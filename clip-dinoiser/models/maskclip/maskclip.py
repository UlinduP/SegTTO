# ---------------------------------------------------------------------------------------------------
# CLIP-DINOiser
# authors: Monika Wysoczanska, Warsaw University of Technology

# Copyright (c) OpenMMLab. All rights reserved.
# Modified version of the original MaskCLIP code: https://github.com/chongzhou96/MaskCLIP/tree/master
# ---------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.ops import resize
from typing import List, Tuple
from torch import Tensor
from open_clip import get_tokenizer,  create_model_from_pretrained
from models.builder import MODELS
import torchvision.transforms as T
from .utils.prompt_templates import imagenet_templates
import json
from PIL import Image

from ..tpt.tpt_classification import TPT

OPENAI_NORMALIZE = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


@MODELS.register_module()
class MaskClip(nn.Module):
    def __init__(
            self,
            backbone,
            decode_head,
            clip_model,
            class_names
        ):
        super(MaskClip, self).__init__()

        self.decode_head = eval(decode_head.get('type'))(clip_model, class_names, **decode_head)
        self.patch_size = backbone.get('patch_size')
        self.img_size = tuple([backbone.get('img_size', 224)]*2)
        pretrained = decode_head.get("pretrained")
        model, _ = create_model_from_pretrained(clip_model, pretrained=pretrained)
        model.eval()
        self.clip_T = OPENAI_NORMALIZE
        self.hook_features = {}
        self.backbone = model
        def hook_fn_forward(module, input, output):
            self.hook_features["v"] = output
        self.backbone.visual.transformer.resblocks[-2].register_forward_hook(hook_fn_forward)
        self._positional_embd = nn.Parameter(self.backbone.visual.positional_embedding.data.clone())

    @torch.no_grad()
    def extract_feat(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features from images."""
        pos_embed = self.backbone.visual.positional_embedding

        B, C, H, W = inputs.shape
        hw_shape = (H // self.patch_size, W // self.patch_size)
        x_len, pos_len = hw_shape[0]*hw_shape[1], pos_embed.shape[0]

        if x_len != pos_len:
            if pos_len == (self.img_size[0] // self.patch_size) * (self.img_size[1] // self.patch_size) + 1:
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError(
                    '{}, {}'.format(x_len, pos_len))

            self.backbone.visual.positional_embedding.data = self.resize_pos_embed(
                self._positional_embd[None], hw_shape,  (pos_h, pos_w), 'bicubic')[0]

        _ = self.backbone(inputs)
        v = self.hook_features["v"]
        v = self.extract_v(v, self.backbone.visual.transformer.resblocks[-1]).permute(1, 0, 2)
        v = self.backbone.visual.ln_post(v)
        v = v[:, 1:]
        v = v.reshape(B, hw_shape[0], hw_shape[1], -1).permute(0, 3, 1, 2).contiguous()

        self.backbone.visual.positional_embedding.data = self._positional_embd
        return v

    def extract_v(self, x, block):
        y = block.ln_1(x)
        y = torch.nn.functional.linear(y, block.attn.in_proj_weight, block.attn.in_proj_bias)
        B, N, C = y.shape
        y = y.view(B, N, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * B, N, C // 3)
        y = F.linear(y, block.attn.out_proj.weight, block.attn.out_proj.bias)
        q, k, v = y.tensor_split(3, dim=0)
        v += x
        v += block.mlp(block.ln_2(v))
        return v


    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed
    
    def save_tensor_as_image(self, tensor, filename='output_image.png'):
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
        
        tensor = tensor.squeeze(0).cpu()
        tensor = tensor * std[:, None, None] + mean[:, None, None]
        tensor = tensor.clamp(0, 1)
        tensor = (tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
        
        image = Image.fromarray(tensor)
        image.save(filename)

    def forward(self, inputs: Tensor, return_feat=False) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        images = inputs
        inputs = self.clip_T(inputs)
        x = self.extract_feat(inputs)
        if return_feat:
            seg_logits, feats = self.decode_head(x, images, return_feat, tpt_tuning=False)
            return seg_logits, feats
        else:
            seg_logits = self.decode_head(x, images, tpt_tuning=False)
        return seg_logits

class MaskClipHead(nn.Module):
    def __init__(self, clip_model, class_names, in_channels=3, text_channels=512, use_templates=False, pretrained=None, json_path=None,
                 **kwargs):
        super(MaskClipHead, self).__init__()

        self.text_channels = text_channels
        self.clip_model = clip_model
        self.pretrained = pretrained
        self.class_names = class_names
        self.in_channels = in_channels
        self.use_templates = use_templates
        self.tokenizer = get_tokenizer(clip_model)
        model, _ = create_model_from_pretrained(clip_model, pretrained=pretrained)
        model.eval()
        self.device = "cuda"
        self.weighted_attr = False
        
        self.attributes_dict = self.load_attributes(json_path) # Load attributes from json file
        self.register_buffer("class_embeddings", self._get_class_embeddings(model, class_names))
        self.proj = nn.Conv2d(self.in_channels, text_channels, 1, bias=False)
        self.proj.weight = nn.Parameter(model.visual.proj.t()[:, :, None, None])
        
        
        # tpt tunning part
        # templates_to_tune = ['a_photo_of_a', 'a_photo_of_a_clean', 'a_close-up_photo_of_a', 'a_blurry_photo_of_the', 'a_bright_photo_of_the']
        processed_templates = [template.replace(" {}.", "").replace(" ", "_").rstrip("_") for template in imagenet_templates]
        
        self.prompt_tuner = TPT(class_names, use_templates, templates_to_tune=processed_templates[:5])
        self.tpt_tuning = True
        
        self.attr_embeddings = torch.stack([self._embed_attributes(model, label, self.attributes_dict) for label in self.class_names])
        self.aug_embeddings = torch.stack([self._embed_label(model, label, self.attributes_dict, True) for label in self.class_names])

    @torch.no_grad()
    def update_vocab(self, class_names):
        model, _ = create_model_from_pretrained(self.clip_model, pretrained=self.pretrained )
        model.eval()
        self.class_embeddings = self._get_class_embeddings(model, class_names)
        
    @torch.no_grad()
    def load_attributes(self, json_path: str) -> dict:
        with open(json_path, 'r') as f:
            attributes = json.load(f)
        return attributes

    @torch.no_grad()
    def _embed_label(self, text_model: torch.nn.Module, label: str, attributes_dict: dict, tpt_tuning: bool) -> torch.Tensor:
        """
        Encode label name into a single vector
        """
        if self.use_templates:
            templates = imagenet_templates
        elif "laion" in self.pretrained:
            templates = ['a photo of a {}', 'a photo of an {}']
        else:
            templates = ['a {}']
        all_prompts = [self.tokenizer(template.format(label)) for template in templates]
        
        attributes = False
        if attributes:
            if label in attributes_dict:
                attributes = attributes_dict[label]
                for attr in attributes:
                    # print(f"Adding attribute for {label}: {attr} ")
                    formatted_templates = [
                        template.format(label).rstrip(".") + f" featuring {attr}" for template in templates
                    ]
                    
                    # Print each formatted template
                    # for formatted_template in formatted_templates:
                    #     print(formatted_template)
                        
                    attribute_prompts = [self.tokenizer(formatted_template) for formatted_template in formatted_templates]
                    all_prompts.extend(attribute_prompts)
                    
            out = text_model.encode_text(torch.cat(all_prompts))
            out /= out.norm(dim=-1, keepdim=True)
            # out = 0.4*out[:80, :].mean(dim=0) + 0.6*out[80:, :].mean(dim=0)
            out = out.mean(dim=0)
            return out
                    
        
        out = text_model.encode_text(torch.cat(all_prompts))
        out /= out.norm(dim=-1, keepdim=True)
        if tpt_tuning:
            return out
        out = out.mean(dim=0)
        return out
    
    # @torch.no_grad()
    # def _embed_attributes(self, text_model: torch.nn.Module, label: str, attributes_dict: dict) -> torch.Tensor:
    #     """
    #     Encode attribute descriptions into a single vector.
    #     """
    #     if not attributes_dict:
    #         return None
        
    #     if self.use_templates:
    #         templates = imagenet_templates
    #         # templates = ['a photo of a {}', 'a photo of an {}']
    #     elif "laion" in self.pretrained:
    #         templates = ['a photo of a {}', 'a photo of an {}']
    #     else:
    #         templates = ['a {}']
        
    #     all_attribute_prompts = []
    #     if label in attributes_dict:
    #         attributes = attributes_dict[label]
    #         for attr in attributes:
    #             print(f"Adding attribute for {label}: {attr} ")
    #             formatted_templates = [template.format(label).rstrip(".") + f" featuring {attr}" for template in templates]
    #             attribute_prompts = [self.tokenizer(formatted_template) for formatted_template in formatted_templates]
    #             all_attribute_prompts.extend(attribute_prompts)
    #     else:
    #         assert False, f"Attributes not found for label: {label}"
        
    #     attribute_out = text_model.encode_text(torch.cat(all_attribute_prompts))
    #     attribute_out /= attribute_out.norm(dim=-1, keepdim=True)
    #     attribute_embed = attribute_out.mean(dim=0)
    #     return attribute_embed
    @torch.no_grad()
    def _embed_attributes(self, text_model: torch.nn.Module, label: str, attributes_dict: dict) -> torch.Tensor:
        """
        Encode attribute descriptions into a single vector.
        """
        if not attributes_dict:
            return None
        
        if self.use_templates:
            templates = imagenet_templates
        elif "laion" in self.pretrained:
            templates = ['a photo of a {}', 'a photo of an {}']
        else:
            templates = ['a photo of a {}']
        
        # templates = ['a photo of a {}']
        
        all_attribute_embeddings = []
        all_weights = []
        weight_attributes = True
        if label in attributes_dict:
            attributes = attributes_dict[label]
            
            # Encode label templates and average them to get label embedding
            label_prompts = [self.tokenizer(template.format(label)) for template in templates]
            label_embeds = text_model.encode_text(torch.cat(label_prompts))
            label_embeds /= label_embeds.norm(dim=-1, keepdim=True)
            label_embed = label_embeds.mean(dim=0, keepdim=True)  

            for attr in attributes:
                attr_prompts = [self.tokenizer(template.format(label).rstrip(".") + f" featuring {attr}") for template in templates]
                attr_embeds = text_model.encode_text(torch.cat(attr_prompts))  
                attr_embeds /= attr_embeds.norm(dim=-1, keepdim=True)
                
                attr_embed = attr_embeds.mean(dim=0)
                all_attribute_embeddings.append(attr_embed)
                
                if weight_attributes:
                    cos_sim = torch.nn.functional.cosine_similarity(attr_embed.unsqueeze(0), label_embed).item()
                    all_weights.append(cos_sim)
        
        else:
            raise ValueError(f"Attributes not found for label: {label}")

        attribute_out = torch.stack(all_attribute_embeddings, dim=0)  
        
        if weight_attributes and all_weights:
            weights = torch.tensor(all_weights)
            weights = torch.nn.functional.softmax(weights, dim=0)  
            attribute_embed = torch.sum(attribute_out * weights.unsqueeze(-1), dim=0)  
        else:
            attribute_embed = attribute_out.mean(dim=0)  
        
        return attribute_embed

    def _get_class_embeddings(self, text_model: torch.nn.Module, class_names: List[str]):
        aug_embeddings = torch.stack([self._embed_label(text_model, label, self.attributes_dict, False) for label in class_names])
        aug_embeddings = aug_embeddings / aug_embeddings.norm(dim=-1, keepdim=True)
        return aug_embeddings.squeeze(1)

    def forward(self, inputs, images, return_feat=False, tpt_tuning=False):
        v = inputs
        feat = self.proj(v)
        output = self.cls_seg(feat, tpt_tunning=False)
        if return_feat:
            return output, feat
        return output

    def cls_seg(self, feat, image=None, tpt_tunning=False):
        # tpt_tunning = False
        if tpt_tunning and image is not None:
            with torch.enable_grad():
                class_embeddings = self.prompt_tuner.tpt_tuning(image)  # torch.Size([1, 3, 5, 512])
                
            class_embeddings = class_embeddings.squeeze(0)
            # model, _ = create_model_from_pretrained(self.clip_model, pretrained=self.pretrained )
            # linear_layer = nn.Linear(768, 512)
            # linear_layer = linear_layer.to('cuda:0')
            # class_embeddings = linear_layer(class_embeddings)
            aug_embeddings = self.aug_embeddings / self.aug_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = torch.cat((class_embeddings, aug_embeddings[:, -75:, :].to(self.device)), dim=1)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=1)
            
            attribute = True
            if attribute:
                if not self.weighted_attr:
                    final_embedding = (class_embeddings + self.attr_embeddings.to(self.device)) / 2
                else:
                    final_embedding = (0.4*class_embeddings + 0.6*self.attr_embeddings.to(self.device))
                self.class_embeddings = final_embedding
            else:
                self.class_embeddings = class_embeddings
                
        feat = feat / feat.norm(dim=1, keepdim=True)
        output = F.conv2d(feat, self.class_embeddings[:, :, None, None])
        output = F.softmax(output * 100, dim=1)
        return output
