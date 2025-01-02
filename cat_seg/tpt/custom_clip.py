
import math
from typing import List, Tuple
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc 

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from clip import load, tokenize
from third_party.clip import load, tokenize
from tpt.simple_tokenizer import SimpleTokenizer as _Tokenizer
from tpt.data.imagnet_prompts import imagenet_classes
from tpt.data.fewshot_datasets import fewshot_datasets
from tpt.data.cls_to_names import *

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT='~/.cache/clip'

# class ClipImageEncoder(nn.Module):
#     def __init__(self, device, arch="ViT-L/14", image_resolution=224, n_class=1000):
#         super(ClipImageEncoder, self).__init__()
#         # clip, embed_dim, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
#         clip, clip_preprocess = load(arch, device=device, jit=False)
#         self.encoder = clip.visual
#         del clip.transformer
#         torch.cuda.empty_cache()
        
#         # self.cls_head = nn.Linear(embed_dim, n_class)
    
#     @property
#     def dtype(self):
#         return self.encoder.conv1.weight.dtype

#     def forward(self, image):
#         x = self.encoder(image.type(self.dtype), dense=True)
#         # output = self.cls_head(x)
#         return x


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
 
        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, num_templates=None, n_ctx=16, template_init=None, ctx_position='end', learned_cls=False, attribute_embedding=None):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.num_templates = num_templates
        self.attribute_embedding = attribute_embedding

        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)
        if self.num_templates is not None:
            ctx_vectors = []
            for ctx_init in template_init:
                # use given words to initialize context vectors
                print("Initializing the contect with given words: [{}]".format(ctx_init))
                ctx_init = ctx_init.replace("_", " ")
                if '[CLS]' in ctx_init:
                    ctx_list = ctx_init.split(" ")
                    split_idx = ctx_list.index("[CLS]")
                    ctx_init = ctx_init.replace("[CLS] ", "")
                    ctx_position = "middle"
                else:
                    split_idx = None
                self.split_idx = split_idx
                # n_ctx = len(ctx_init.split(" "))
                prompt = tokenize(ctx_init).to(self.device) # (1, 77)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype) # (1, 77, 768)
                ctx_vectors_per_template = embedding[0, 1 : 1 + n_ctx, :] # (n_ctx, 768)
                ctx_vectors.append(ctx_vectors_per_template)
            ctx_vectors = torch.stack(ctx_vectors, dim=0) # (80, n_ctx, 768)
            self.prompt_prefix = template_init
        else:
            ctx_init = template_init
            if ctx_init:
                # use given words to initialize context vectors
                print("Initializing the contect with given words: [{}]".format(ctx_init))
                ctx_init = ctx_init.replace("_", " ")
                if '[CLS]' in ctx_init:
                    ctx_list = ctx_init.split(" ")
                    split_idx = ctx_list.index("[CLS]")
                    ctx_init = ctx_init.replace("[CLS] ", "")
                    ctx_position = "middle"
                else:
                    split_idx = None
                self.split_idx = split_idx
                n_ctx = len(ctx_init.split(" "))
                prompt = tokenize(ctx_init).to(self.device)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
                prompt_prefix = ctx_init
            else:
                print("Random initialization: initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
            
            self.prompt_prefix = prompt_prefix

            print(f'Initial context: "{prompt_prefix}"')
            print(f"Number of context words (tokens): {n_ctx}")

            # batch-wise prompt tuning for test-time adaptation
            # if self.num_templates is not None: 
            #     ctx_vectors = ctx_vectors.repeat(self.num_templates, 1, 1)  #(N, L, D) # (4, 768)
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized # (80, n_ctx, 768)

        if torch.isnan(self.ctx).any() or torch.isinf(self.ctx).any():
            print("NaN or Inf in the self.ctx 117")

        if not self.learned_cls: # self.learned_cls = False
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            if self.num_templates is not None:
                prompts = [[prompt_prefix + " " + name + "." for name in classnames] for prompt_prefix in self.prompt_prefix]
            else:
                prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            if self.attribute_embedding is not None:
                print("Initialize with catseg text embedding")
                cls_vectors = self.attribute_embedding.unsqueeze(0).repeat(self.num_templates, 1, 1, 1)
            else:
                print("Random initialization: initializing a learnable class token")
                cls_vectors = torch.empty(self.num_templates, n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
                nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            if self.num_templates is not None:
                prompts = [[prompt_prefix + " " + _ + "." for _ in classnames] for prompt_prefix in self.prompt_prefix]
            else:
                prompts = [self.prompt_prefix + " " + _ + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized

        if self.num_templates is not None:
            tokenized_prompts = torch.cat([torch.cat([tokenize(p) for p in ps]).to(self.device).unsqueeze(0) for ps in prompts]).to(self.device) # (80, classnames, 77)
        else:
            tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) # (80, classnames, 77, 768)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS # (classnames, 1, 768)
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS # (classnames, 71, 768)
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS # (classnames, 72, 768)

        self.ctx_init = ctx_init # 'a photo of a'
        self.tokenized_prompts = tokenized_prompts  # (80, classnames, 77)
        self.name_lens = name_lens # list of number of tokens in each class name ex: ['others', 'nuclei in cells'] => [1, 4]
        self.class_token_position = ctx_position # 'end'
        self.n_cls = n_cls # number of classes
        self.n_ctx = n_ctx # number of tokens in the prompt
        self.classnames = classnames # list of class names

        # empty the cache
        # torch.cuda.empty_cache()
        del embedding, tokenized_prompts, prompts, ctx_vectors, ctx_init, classnames, name_lens, cls_vectors, cls_token

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            if self.num_templates is not None:
                prompts = [[prompt_prefix + " " + name + "." for name in classnames] for prompt_prefix in self.prompt_prefix]
            else:
                prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            if self.num_templates is not None:
                prompts = [[prompt_prefix + " " + _ + "." for _ in classnames] for prompt_prefix in self.prompt_prefix]
            else:
                prompts = [self.prompt_prefix + " " + _ + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()

        if self.num_templates is not None:
            tokenized_prompts = torch.cat([torch.cat([tokenize(p) for p in ps]).to(self.device).unsqueeze(0) for ps in prompts]).to(self.device) # (80, classnames, 77)
        else:
            tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        # clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)
        clip, clip_preprocess = load(arch, device=self.device, jit=False)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        if self.num_templates is not None:
            self.token_prefix = embedding[:, :, :1, :]
            if self.learned_cls:
                self.token_suffix = embedding[:, :, 2 + self.n_ctx :, :]  # CLS, EOS
            else:
                self.token_suffix = embedding[:, :, 1 + self.n_ctx :, :]  # CLS, EOS
        else:
            self.token_prefix = embedding[:, :1, :]
            self.token_suffix = embedding[:, 1 + self.n_ctx :, :]

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

        del clip, clip_preprocess, embedding, tokenized_prompts, prompts, cls_vectors, cls_token, classnames, name_lens

    def forward(self, init=None):
        
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
            if torch.isnan(init).any() or torch.isinf(init).any():
                print("NaN or Inf in the init 209")
        else:
            ctx = self.ctx # (80, n_ctx, 768)

        if torch.isnan(ctx).any() or torch.isinf(ctx).any():
            print("NaN or Inf in the self.ctx 206")

        
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        # elif not ctx.size()[0] == self.n_cls:
        ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)   #
        # ctx = (classnames, 4, 768)

        prefix = self.token_prefix # (classnames, 1, 768)
        suffix = self.token_suffix # (classnames, 72, 768)
        # if self.num_templates is not None: 
        #     # This way only works for single-gpu setting (could pass batch size as an argument for forward())
        #     prefix = prefix.repeat(self.num_templates, 1, 1, 1)
        #     suffix = suffix.repeat(self.num_templates, 1, 1, 1)

        if torch.isnan(ctx).any() or torch.isinf(ctx).any():
            print("NaN or Inf in the ctx 225")

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                # breakpoint()
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim) # (200, 1, 512)
                        ctx,     # (n_cls, n_ctx, dim) # (200, 4, 512)  # 5,8,7,768
                        cls,     # (n_cls, 1, dim) # (200, 1, 512)
                        suffix,  # (n_cls, *, dim) # (200, 72, 512)
                    ],
                    dim=-2,
                )
            else:
                with torch.enable_grad():

                    prompts = torch.cat(
                        [
                            prefix,  # (n_cls, 1, dim)
                            ctx,     # (n_cls, n_ctx, dim)
                            suffix,  # (n_cls, *, dim)
                        ],
                        dim=-2,
                    )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts # (classnames, 77, 768)


class ClipTestTimeTuning(nn.Module):
    def __init__(self, device, classnames, num_templates, criterion='cosine', arch="ViT-L/14",
                        n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False, attribute_embedding=None):
        super(ClipTestTimeTuning, self).__init__()
        # clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        clip, clip_preprocess = load(arch, device=device, jit=False)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        self.num_templates = num_templates
        # prompt tuning
        self.prompt_learner = PromptLearner(clip, classnames, num_templates, n_ctx, ctx_init, ctx_position, learned_cls, attribute_embedding)
        self.criterion = criterion

        del clip, clip_preprocess
        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner() # (classnames, 77, 768)    grad=False //// (80, classnames, 77, 768)    
        tokenized_prompts = self.prompt_learner.tokenized_prompts # (classnames, 77)   grad=False
        if self.num_templates is not None:
            # t_features = torch.empty((prompts.shape[0], prompts.shape[1], prompts.shape[3]), dtype = prompts.dtype, device = prompts.device, requires_grad=True)
            # for i in range(prompts.shape[0]):
            #     with torch.no_grad():
            #         t_features[i] = self.text_encoder(prompts[i], tokenized_prompts)
            #     torch.cuda.empty_cache()
            prompts = prompts.reshape(prompts.size(0) * prompts.size(1), prompts.size(2), prompts.size(3))
            tokenized_prompts = tokenized_prompts.reshape(tokenized_prompts.size(0) * tokenized_prompts.size(1), tokenized_prompts.size(2))
            t_features = self.text_encoder(prompts, tokenized_prompts)  # (80 * classnames, 768) 
            text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
            text_features = torch.stack(text_features, dim=0)
            out = torch.mean(text_features, dim=0).reshape(self.num_templates, prompts.shape[0] // self.num_templates, t_features.shape[-1])
        else:
            t_features = self.text_encoder(prompts, tokenized_prompts) # (classnames, 1024)    grad=False
            text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
            text_features = torch.stack(text_features, dim=0)
            out = torch.mean(text_features, dim=0)
        return out

    def inference(self, image):
        with torch.no_grad():        
            image_features = self.image_encoder(image.type(self.dtype), dense=True) # arch=RN50 >> image=(64, 3, 224, 224) => image_features=(64, 1024)
        text_features = self.get_text_features() # arch=RN50 => (200, 1024)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        # image features: (64, 577, 768), text features: (80, n_classses, 768)
        if self.num_templates is not None:
            img_shape = image_features.shape
            text_shape = text_features.shape
            image_features = image_features.reshape(image_features.size(0) * image_features.size(1), image_features.size(2))
            text_features = text_features.reshape(text_features.size(0) * text_features.size(1), text_features.size(2))
            logits = logit_scale * image_features @ text_features.T
            logits = logits.reshape(img_shape[0], img_shape[1], text_shape[0], text_shape[1])
        else:
            logits = logit_scale * image_features @ text_features.T

        if self.num_templates is not None:
            out = logits, text_features.view(text_shape).permute(1, 0, 2).unsqueeze(0)
        else:
            out = logits, text_features

        return out
    
    def forward(self, input):
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input)


def get_coop(clip_arch, classnames, num_templates, device, n_ctx, ctx_init, learned_cls=False, attribute_embedding=None):
    # if test_set in fewshot_datasets:
    #     classnames = eval("{}_classes".format(test_set.lower()))
    # elif test_set == 'bongard':
    #     if learned_cls:
    #         classnames = ['X', 'X']
    #     else:
    #         classnames = ['True', 'False']
    # else:
    #     classnames = imagenet_classes

    model = ClipTestTimeTuning(device, classnames, num_templates=num_templates, arch=clip_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls, attribute_embedding=attribute_embedding)

    return model