# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified by Jian Ding from: https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
import fvcore.nn.weight_init as weight_init
import torch
import os

from torch import nn
from torch.nn import functional as F

from cat_seg.tpt.tpt_segmentation import TPT

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .model import Aggregator
from cat_seg.third_party import clip
from cat_seg.third_party import imagenet_templates
from detectron2.data.catalog import MetadataCatalog

import numpy as np
import open_clip
class CATSegPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        # train_class_json: str,
        # test_class_json: str,
        train_dataset: str,
        test_dataset: str,
        clip_pretrained: str,
        prompt_ensemble_type: str,
        text_guidance_dim: int,
        text_guidance_proj_dim: int,
        appearance_guidance_dim: int,
        appearance_guidance_proj_dim: int,
        prompt_depth: int,
        prompt_length: int,
        decoder_dims: list,
        decoder_guidance_dims: list,
        decoder_guidance_proj_dims: list,
        num_heads: int,
        num_layers: tuple,
        hidden_dims: tuple,
        pooling_sizes: tuple,
        feature_resolution: tuple,
        window_sizes: tuple,
        attention_type: str,
        device: str,
        prompts: list,
        attr_path: str,
        tpt_steps: int,
        loss: str,
        gpu: int,
    ):
        """
        Args:
            
        """
        super().__init__()
        
        import json
        self.device = device
        print(self.device)

        self.prompt_ensemble_type = prompt_ensemble_type        

        if self.prompt_ensemble_type == "imagenet_select":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == "imagenet":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == "single":
            prompt_templates = ['A photo of a {} in the scene',]
        else:
            raise NotImplementedError
  
        self.tokenizer = None
        if clip_pretrained == "ViT-G" or clip_pretrained == "ViT-H":
            # for OpenCLIP models
            name, pretrain = ('ViT-H-14', 'laion2b_s32b_b79k') if clip_pretrained == 'ViT-H' else ('ViT-bigG-14', 'laion2b_s39b_b160k')
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                name, 
                pretrained=pretrain, 
                device=device, 
                force_image_size=336,)
        
            self.tokenizer = open_clip.get_tokenizer(name)
        else:
            # for OpenAI models
            clip_model, clip_preprocess = clip.load(clip_pretrained, device=device, jit=False, prompt_depth=prompt_depth, prompt_length=prompt_length)
        
        self.clip_model = clip_model.float()
        self.clip_preprocess = clip_preprocess

        # use class_texts in train_forward, and test_class_texts in test_forward
        # with open(train_class_json, 'r') as f_in:
        #     self.class_texts = json.load(f_in)
        # with open(test_class_json, 'r') as f_in:
        #     self.test_class_texts = json.load(f_in)
        self.class_texts = MetadataCatalog.get(train_dataset).stuff_classes          

        self.is_tpt_tuning = True
        self.tpt_tuning_classnames_only = True
        self.classnames_plus_attributes = True
        self.imagenet_templates_tuning = True
        self.catseg_plus = True
        self.attribute_embedding_tuning = False

        self.templates_to_tune =  prompts#['a_bad_photo_of_a', 'a_photo_of_many', 'a_sculpture_of_a', 'a_photo_of_the_hard_to_see', 'a_low_resolution_photo_of_the', 'a_rendering_of_a', 'graffiti_of_a', 'a_bad_photo_of_the', 'a_cropped_photo_of_the', 'a_tattoo_of_a', 'the_embroidered', 'a_photo_of_a_hard_to_see', 'a_bright_photo_of_a', 'a_photo_of_a_clean', 'a_photo_of_a_dirty', 'a_dark_photo_of_the', 'a_drawing_of_a', 'a_photo_of_my', 'the_plastic', 'a_photo_of_the_cool', 'a_close-up_photo_of_a', 'a_black_and_white_photo_of_the', 'a_painting_of_the', 'a_painting_of_a', 'a_pixelated_photo_of_the', 'a_sculpture_of_the', 'a_bright_photo_of_the', 'a_cropped_photo_of_a', 'a_plastic', 'a_photo_of_the_dirty', 'a_jpeg_corrupted_photo_of_a', 'a_blurry_photo_of_the', 'a_photo_of_the', 'a_good_photo_of_the', 'a_rendering_of_the', 'a_video_of_a', 'a_photo_of_one', 'a_doodle_of_a', 'a_close-up_photo_of_the', 'a_photo_of_a', 'the_origami', 'the_video_of_a', 'a_sketch_of_a', 'a_doodle_of_the', 'a_origami', 'a_low_resolution_photo_of_a', 'the_toy', 'a_rendition_of_the', 'a_photo_of_the_clean', 'a_photo_of_a_large', 'a_rendition_of_a', 'a_photo_of_a_nice', 'a_photo_of_a_weird', 'a_blurry_photo_of_a', 'a_cartoon', 'art_of_a', 'a_sketch_of_the', 'a_embroidered', 'a_pixelated_photo_of_a', 'itap_of_the', 'a_jpeg_corrupted_photo_of_the', 'a_good_photo_of_a', 'a_plushie', 'a_photo_of_the_nice', 'a_photo_of_the_small', 'a_photo_of_the_weird', 'the_cartoon', 'art_of_the', 'a_drawing_of_the', 'a_photo_of_the_large', 'a_black_and_white_photo_of_a', 'the_plushie', 'a_dark_photo_of_a', 'itap_of_a', 'graffiti_of_the', 'a_toy', 'itap_of_my', 'a_photo_of_a_cool', 'a_photo_of_a_small', 'a_tattoo_of_the']
#['a_photo_of_a','a_photo_of_a_clean', 'a_close-up_photo_of_a', 'a_blurry_photo_of_the', 'a_bright_photo_of_the']  #, 'a_photo_of_a_clean', 'a_close-up_photo_of_a', 'a_blurry_photo_of_the', 'a_bright_photo_of_the'


        if self.catseg_plus:
            # catseg-plus (class names featuring attributes)
            with open(attr_path+f"{test_dataset}.json", 'r') as f_in:
                data = json.load(f_in)
            # with open(f"datasets/atlantis (1).json", 'r') as f_in:
            #     data = json.load(f_in)
            self.test_class_texts = [[f"{key} featuring {attribute}" for attribute in attributes] for key, attributes in data.items()]
            attribute_tuning = [f"A photo of a {key} featuring " + ", ".join(attributes[:2]) for key, attributes in data.items()]
            sc_variant = False
            hierachy = ['']
            if sc_variant:
                super_class = [0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
                hierachy_all_classes = []
                for i in super_class:
                    if i == 0:
                        hierachy_all_classes.append([' '] * len(hierachy))
                    else:
                        hierachy_all_classes.append(hierachy)
            else:
                hierachy_all_classes = len(self.test_class_texts) * [hierachy]
            self.test_class_texts = [list(data.keys()) , self.test_class_texts, hierachy_all_classes, attribute_tuning]
            if self.attribute_embedding_tuning:
                self.attribute_embedding = self.class_embeddings_attribute_tuning(self.test_class_texts, prompt_templates, clip_model)
            else:
                self.text_features_test, self.prompts_75 = self.class_embeddings_75(self.test_class_texts, prompt_templates, clip_model)
                # self.text_features_test= self.class_embeddings(self.test_class_texts, prompt_templates, clip_model)
            if self.tpt_tuning_classnames_only:
                self.tpt_tuning = TPT(self.test_class_texts[0], self.imagenet_templates_tuning, templates_to_tune=self.templates_to_tune,tpt_steps=tpt_steps,loss=loss,gpu=gpu)
            else:
                self.tpt_tuning = TPT(self.test_class_texts[0], self.imagenet_templates_tuning, self.attribute_embedding, templates_to_tune=self.templates_to_tune,tpt_steps=tpt_steps,loss=loss,gpu=gpu)
        else:
            self.test_class_texts = MetadataCatalog.get(test_dataset).stuff_classes
            self.text_features_test = self.class_embeddings(self.test_class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()  

        assert self.class_texts != None
        if self.test_class_texts == None:
            self.test_class_texts = self.class_texts
        # device = "cuda" if torch.cuda.is_available() else "cpu"

        # self.text_features = self.class_embeddings(self.class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
        # self.text_features_test = self.class_embeddings(self.test_class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
        
        transformer = Aggregator(
            text_guidance_dim=text_guidance_dim,
            text_guidance_proj_dim=text_guidance_proj_dim,
            appearance_guidance_dim=appearance_guidance_dim,
            appearance_guidance_proj_dim=appearance_guidance_proj_dim,
            decoder_dims=decoder_dims,
            decoder_guidance_dims=decoder_guidance_dims,
            decoder_guidance_proj_dims=decoder_guidance_proj_dims,
            num_layers=num_layers,
            nheads=num_heads, 
            hidden_dim=hidden_dims,
            pooling_size=pooling_sizes,
            feature_resolution=feature_resolution,
            window_size=window_sizes,
            attention_type=attention_type
            )
        self.transformer = transformer

    @classmethod
    def from_config(cls, cfg):#, in_channels, mask_classification):
        ret = {}

        # ret["train_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON
        # ret["test_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON
        assert len(cfg.DATASETS.TRAIN) == 1, "Current implementation of CATSeg only supports one training dataset"
        ret["train_dataset"] = cfg.DATASETS.TRAIN[0]
        assert len(cfg.DATASETS.TEST) == 1, "Current implementation of CATSeg only supports one test dataset"
        ret["test_dataset"] = cfg.DATASETS.TEST[0]
        ret["clip_pretrained"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED
        ret["prompt_ensemble_type"] = cfg.MODEL.PROMPT_ENSEMBLE_TYPE

        # Aggregator parameters:
        ret["text_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM
        ret["text_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM
        ret["appearance_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM
        ret["appearance_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM

        ret["decoder_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS
        ret["decoder_guidance_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS
        ret["decoder_guidance_proj_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS

        ret["prompt_depth"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH
        ret["prompt_length"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH

        ret["num_layers"] = cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS
        ret["num_heads"] = cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS
        ret["hidden_dims"] = cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS
        ret["pooling_sizes"] = cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES
        ret["feature_resolution"] = cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION
        ret["window_sizes"] = cfg.MODEL.SEM_SEG_HEAD.WINDOW_SIZES
        ret["attention_type"] = cfg.MODEL.SEM_SEG_HEAD.ATTENTION_TYPE

        ret["device"] = cfg.MODEL.DEVICE

        ret["prompts"] = cfg.SEG_TTO.PROMPTS
        ret["attr_path"] = cfg.SEG_TTO.ATTR_PATH
        ret["tpt_steps"] = cfg.SEG_TTO.TPT_STEPS
        ret["loss"] = cfg.SEG_TTO.LOSS
        ret["gpu"] = cfg.SEG_TTO.GPU

        return ret

    def forward(self, x, vis_guidance, global_image = None, OOM = False, tpt_tuning = False, tuned_text_embedding = None):
        # x.shape = (1, 768, 24, 24)
        # vis_guidance = {'res2': (1, 128, 96, 96), 'res3': (1, 256, 48, 48), 'res4': (1, 512, 24, 24)}
        vis = [vis_guidance[k] for k in vis_guidance.keys()][::-1]  

        # if x.device == "cuda:6":
        #     global_image = global_image.to("cuda:7")
        # else:
        #     global_image = global_image.to("cuda:6")
        self.is_tpt_tuning = tpt_tuning
        if self.is_tpt_tuning:
            if self.tpt_tuning_classnames_only:
                with torch.enable_grad():
                    class_embedding = self.tpt_tuning.tpt_tuning(global_image)
                    class_embedding = class_embedding.to(x.device)
                    if class_embedding.shape[2] != 80:
                        # # repeat until 80
                        # repeats = (80 // class_embedding.shape[2]) + 1
                        # class_embedding = class_embedding.repeat(1, 1, repeats, 1)[:, :, :80, :]
                        # add remaining prompts
                        class_embedding = torch.cat([class_embedding, self.prompts_75.unsqueeze(0)], dim=2)
                    if self.imagenet_templates_tuning:
                        class_embedding = class_embedding.repeat(x.shape[0], 1, 1, 1) # (n_classes, 80, 768)->(5, n_classes, 80, 768)
                    else:
                        # repeat 80 times
                        class_embedding = class_embedding.unsqueeze(0).repeat(x.shape[0], 1, 1) # (n_classes, 768)->(5, n_classes, 768)
                        class_embedding = class_embedding.unsqueeze(2).repeat(1, 1, 80, 1) # (5, n_classes, 768)->(5, n_classes, 80, 768)
                    class_embedding = class_embedding.to(x.dtype)
                    if self.classnames_plus_attributes:
                        attribute_embedding = self.text_features_test.repeat(x.shape[0], 1, 1, 1)
                        text = 0.4 * class_embedding + 0.6 * attribute_embedding
                    else:
                        text = class_embedding
            else:
                with torch.enable_grad():
                    class_embedding = self.tpt_tuning.tpt_tuning(global_image)
                    class_embedding = class_embedding.to(x.device)
                    if class_embedding.shape[2] != 80:
                        # repeat until 80
                        repeats = (80 // class_embedding.shape[2]) + 1
                        class_embedding = class_embedding.repeat(1, 1, repeats, 1)[:, :, :80, :]
                        # # add remaining prompts
                        # class_embedding = torch.cat([class_embedding, self.prompts_75.unsqueeze(0)], dim=2)
                    if self.imagenet_templates_tuning:
                        class_embedding = class_embedding.repeat(x.shape[0], 1, 1, 1) # (n_classes, 80, 768)->(5, n_classes, 80, 768)
                    else:
                        # repeat 80 times
                        class_embedding = class_embedding.unsqueeze(0).repeat(x.shape[0], 1, 1) # (n_classes, 768)->(5, n_classes, 768)
                        class_embedding = class_embedding.unsqueeze(2).repeat(1, 1, 80, 1) # (5, n_classes, 768)->(5, n_classes, 80, 768)
                    class_embedding = class_embedding.to(x.dtype)
                    text = class_embedding

                    # attribute_embedding = self.tpt_tuning.tpt_tuning(global_image)
                    # attribute_embedding = attribute_embedding.to(x.device)
                    # attribute_embedding = attribute_embedding.unsqueeze(1).repeat(x.shape[0], 1, 1, 1) # (n_classes, 80, 768)->(5, n_classes, 80, 768)
                    # attribute_embedding = attribute_embedding.to(x.dtype)
                    # text = attribute_embedding
                    # # class_embedding = self.tpt_tuning.tpt_tuning(global_image)
                    # # class_embedding = class_embedding.unsqueeze(1).repeat(x.shape[0], 1, 1, 1) # (n_classes, 80, 768)->(5, n_classes, 80, 768)
                    # # class_embedding = class_embedding.to(x.dtype)
                    # # attribute_embeddings = []
                    # # for i in range(len(self.test_class_texts[0])):
                    # #     # change init
                    # #     attribute_embedding = self.tpt_tuning.tpt_tuning(global_image, self.test_class_texts[1][i])
                    # #     attribute_embedding = attribute_embedding.mean(dim=0)
                    # #     attribute_embeddings.append(attribute_embedding)
                    # # attribute_embedding = torch.stack(attribute_embeddings, dim=0)
                    # # print(attribute_embedding.shape)
                    # # attribute_embedding = attribute_embedding.unsqueeze(1).repeat(x.shape[0], 1, 1, 1)
                    # # attribute_embedding = attribute_embedding.to(x.dtype)
                    # # print(attribute_embedding.shape)
                    # # text = 0.4 * class_embedding + 0.6 * attribute_embedding
        elif tuned_text_embedding is not None:
            text = tuned_text_embedding[:5,:,:,:]
        else:
            text = self.text_features_test 
            text = text.repeat(x.shape[0], 1, 1, 1) # (5, n_classes, 80, 768)  
        # print("tpt tuning done")                            
        print(f"allocated memory: {torch.cuda.memory_allocated() / 1e6} MB")
        if OOM == False:
            # print(text.shape)
            # breakpoint()
            out = self.transformer(x, text, vis)
        else:
            # size = [0,16,32,48,64]
            out = []
            # print(f"vis[o].shape: {vis[0].shape}")
            for i in range(0, 32):
                viss = [vis[0][i,:,:,:].unsqueeze(0), vis[1][i,:,:,:].unsqueeze(0), vis[2][i,:,:,:].unsqueeze(0)]
                out.append(self.transformer(x[i,:,:,:].unsqueeze(0), text[i,:,:,:].unsqueeze(0), viss))
            out = torch.cat(out, dim=0)
        # out = self.transformer(x, text, vis)
        return out, text

    # @torch.no_grad()
    # def class_embeddings(self, classnames, templates, clip_model):
    #     zeroshot_weights = []
    #     for classname in classnames:
    #         if ', ' in classname:
    #             classname_splits = classname.split(', ')
    #             texts = []
    #             for template in templates:
    #                 for cls_split in classname_splits:
    #                     texts.append(template.format(cls_split))
    #         else:
    #             texts = [template.format(classname) for template in templates]  # format with class
    #         if self.tokenizer is not None:
    #             texts = self.tokenizer(texts).cuda()
    #         else: 
    #             texts = clip.tokenize(texts).cuda()
    #         class_embeddings = clip_model.encode_text(texts)
    #         class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    #         if len(templates) != class_embeddings.shape[0]:
    #             class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
    #             class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    #         class_embedding = class_embeddings
    #         zeroshot_weights.append(class_embedding)
    #     zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    #     return zeroshot_weights

    # with attributes
    @torch.no_grad()
    def class_embeddings(self, classnames, templates, clip_model):
        classesonly = classnames[0]
        classattributes = classnames[1]  
        hierachy = classnames[2] 

        zeroshot_weights_lst = []
        for classname in classesonly:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                texts = [template.format(classname) for template in templates]  # format with class
            if self.tokenizer is not None:
                texts = self.tokenizer(texts).to(self.device)
            else: 
                texts = clip.tokenize(texts).to(self.device)

            class_embeddings = clip_model.encode_text(texts) #(80, 768) This has become 80 because of the 80 imageNET templates
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            if len(templates) != class_embeddings.shape[0]:
                class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings
            zeroshot_weights_lst.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights_lst, dim=0).float().to(self.device)

        final_embeddings = []
        for _class in classattributes:
            tokens = []
            index = classattributes.index(_class)
            for attribute in _class:
                attribute = [template.format(attribute) for template in templates]  # format with class
                if self.tokenizer is not None:
                    texts = self.tokenizer(attribute).to(self.device)
                else: 
                    texts = clip.tokenize(attribute).to(self.device)
                class_embeddings = clip_model.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                tokens.append(class_embeddings)
            tokens = torch.stack(tokens, dim=0).squeeze(1)
            tokens = tokens.to(torch.float32)
            if self.prompt_ensemble_type == "single":
                cosine_similarity = F.cosine_similarity(tokens, zeroshot_weights_lst[index], dim=-1)
                cosine_similarity = cosine_similarity / torch.sum(cosine_similarity)
                cosine_similarity = cosine_similarity.view(cosine_similarity.shape[0], 1)
            else:
                cosine_similarity = F.cosine_similarity(tokens, zeroshot_weights_lst[index], dim=-1).mean(dim=-1)
                cosine_similarity = cosine_similarity / torch.sum(cosine_similarity)
                cosine_similarity = cosine_similarity.view(cosine_similarity.shape[0], 1, 1)
            tokens = tokens * cosine_similarity
            tokens = tokens.mean(dim=0)
            if len(tokens.shape) == 1:
                tokens = tokens.unsqueeze(0)
            final_embeddings.append(tokens)
        final_embeddings = torch.stack(final_embeddings, dim=0).to(self.device)

        hierachy_embeddings = []
        for _class in hierachy:
            tokens = []
            for _attribute in _class:
                attribute = [template.format(_attribute) for template in templates]  # format with class
                if self.tokenizer is not None:
                    texts = self.tokenizer(attribute).to(self.device)
                else: 
                    texts = clip.tokenize(attribute).to(self.device)
                class_embeddings = clip_model.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                if _attribute == ' ':
                    class_embeddings = torch.zeros_like(class_embeddings)
                tokens.append(class_embeddings)
            tokens = torch.stack(tokens, dim=0).squeeze(1)
            tokens = tokens.to(torch.float32)
            tokens = tokens.mean(dim=0)
            if len(tokens.shape) == 1:
                tokens = tokens.unsqueeze(0)
            hierachy_embeddings.append(tokens)
        hierachy_embeddings = torch.stack(hierachy_embeddings, dim=0).to(self.device)


        zeroshot_weights = 0.0 * zeroshot_weights + 1.0 * final_embeddings + 0.0 * hierachy_embeddings # (n_classes, 80, 768)

        return zeroshot_weights
    
    # 75 prompts
    @torch.no_grad()
    def class_embeddings_75(self, classnames, templates, clip_model):
        classesonly = classnames[0]
        classattributes = classnames[1]  

        indices_lst = []
        remove =  self.templates_to_tune #['a_photo_of_a', 'a_photo_of_a_clean', 'a_close-up_photo_of_a', 'a_blurry_photo_of_the', 'a_bright_photo_of_the']
        for r in remove:
            r = " ".join(r.split("_")) + " {}."
            if r in templates:
                indices_lst.append(templates.index(r))

        zeroshot_weights_lst = []
        for classname in classesonly:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                texts = [template.format(classname) for template in templates]  # format with class
            if self.tokenizer is not None:
                texts = self.tokenizer(texts).to(self.device)
            else: 
                texts = clip.tokenize(texts).to(self.device)

            class_embeddings = clip_model.encode_text(texts) #(80, 768) This has become 80 because of the 80 imageNET templates
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            if len(templates) != class_embeddings.shape[0]:
                class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings # (80, 768)
            zeroshot_weights_lst.append(class_embedding) # [(80, 768), (80, 768), ...]
        zeroshot_weights = [torch.stack([clss[i] for i in range(clss.shape[0]) if i not in indices_lst]) for clss in zeroshot_weights_lst]
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).float().to(self.device)

        final_embeddings = []
        for _class in classattributes:
            tokens = []
            index = classattributes.index(_class)
            for attribute in _class:
                attribute = [template.format(attribute) for template in templates]  # format with class
                if self.tokenizer is not None:
                    texts = self.tokenizer(attribute).to(self.device)
                else: 
                    texts = clip.tokenize(attribute).to(self.device)
                class_embeddings = clip_model.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                tokens.append(class_embeddings)
            tokens = torch.stack(tokens, dim=0).squeeze(1)
            tokens = tokens.to(torch.float32)
            if self.prompt_ensemble_type == "single":
                cosine_similarity = F.cosine_similarity(tokens, zeroshot_weights_lst[index], dim=-1)
                cosine_similarity = cosine_similarity / torch.sum(cosine_similarity)
                cosine_similarity = cosine_similarity.view(cosine_similarity.shape[0], 1)
            else:
                cosine_similarity = F.cosine_similarity(tokens, zeroshot_weights_lst[index], dim=-1).mean(dim=-1)
                cosine_similarity = cosine_similarity / torch.sum(cosine_similarity)
                cosine_similarity = cosine_similarity.view(cosine_similarity.shape[0], 1, 1)
            tokens = tokens * cosine_similarity
            tokens = tokens.mean(dim=0)
            if len(tokens.shape) == 1:
                tokens = tokens.unsqueeze(0)
            final_embeddings.append(tokens)
        final_embeddings = torch.stack(final_embeddings, dim=0).to(self.device)

        return final_embeddings, zeroshot_weights
    
    # with attributes
    @torch.no_grad()
    def class_embeddings_attribute_tuning(self, classnames, templates, clip_model):
        classesonly = classnames[0]
        classattributes = classnames[1]  

        zeroshot_weights_lst = []
        for classname in classesonly:
            # if ', ' in classname:
            #     classname_splits = classname.split(', ')
            #     texts = []
            #     for template in templates:
            #         for cls_split in classname_splits:
            #             texts.append(template.format(cls_split))
            # else:
            #     texts = [template.format(classname) for template in templates]  # format with class
            texts = classname
            if self.tokenizer is not None:
                texts = self.tokenizer(texts).to(self.device)
            else: 
                texts = clip.tokenize(texts).to(self.device)

            class_embeddings = clip_model.encode_text(texts) #(80, 768) This has become 80 because of the 80 imageNET templates
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # if len(templates) != class_embeddings.shape[0]:
            #     class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
            #     class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings
            zeroshot_weights_lst.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights_lst, dim=0).float().to(self.device)

        final_embeddings = []
        for _class in classattributes:
            tokens = []
            index = classattributes.index(_class)
            for attribute in _class:
                # attribute = [template.format(attribute) for template in templates]  # format with class
                if self.tokenizer is not None:
                    texts = self.tokenizer(attribute).to(self.device)
                else: 
                    texts = clip.tokenize(attribute).to(self.device)
                class_embeddings = clip_model.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                tokens.append(class_embeddings)
            tokens = torch.stack(tokens, dim=0).squeeze(1)
            tokens = tokens.to(torch.float32)
            cosine_similarity = F.cosine_similarity(tokens, zeroshot_weights_lst[index], dim=-1)
            cosine_similarity = cosine_similarity / torch.sum(cosine_similarity)
            cosine_similarity = cosine_similarity.view(cosine_similarity.shape[0], 1)
            tokens = tokens * cosine_similarity
            tokens = tokens.mean(dim=0)
            if len(tokens.shape) == 1:
                tokens = tokens.unsqueeze(0)
            final_embeddings.append(tokens)
        final_embeddings = torch.stack(final_embeddings, dim=0).to(self.device)

        zeroshot_weights = 0.0 * zeroshot_weights + 1.0 * final_embeddings + 0.0 # (n_classes, 80, 768)

        return zeroshot_weights


# # Copyright (c) Facebook, Inc. and its affiliates.
# # Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# # Modified by Jian Ding from: https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
# import fvcore.nn.weight_init as weight_init
# import torch
# import os

# from torch import nn
# from torch.nn import functional as F

# from detectron2.config import configurable
# from detectron2.layers import Conv2d

# from .model import Aggregator
# from cat_seg.third_party import clip
# from cat_seg.third_party import imagenet_templates
# from detectron2.data.catalog import MetadataCatalog

# import numpy as np
# import open_clip
# class CATSegPredictor(nn.Module):
#     @configurable
#     def __init__(
#         self,
#         *,
#         # train_class_json: str,
#         # test_class_json: str,
#         train_dataset: str,
#         test_dataset: str,
#         clip_pretrained: str,
#         prompt_ensemble_type: str,
#         text_guidance_dim: int,
#         text_guidance_proj_dim: int,
#         appearance_guidance_dim: int,
#         appearance_guidance_proj_dim: int,
#         prompt_depth: int,
#         prompt_length: int,
#         decoder_dims: list,
#         decoder_guidance_dims: list,
#         decoder_guidance_proj_dims: list,
#         num_heads: int,
#         num_layers: tuple,
#         hidden_dims: tuple,
#         pooling_sizes: tuple,
#         feature_resolution: tuple,
#         window_sizes: tuple,
#         attention_type: str,
#     ):
#         """
#         Args:
            
#         """
#         super().__init__()
        
#         import json
#         # use class_texts in train_forward, and test_class_texts in test_forward
#         # with open(train_class_json, 'r') as f_in:
#         #     self.class_texts = json.load(f_in)
#         # with open(test_class_json, 'r') as f_in:
#         #     self.test_class_texts = json.load(f_in)
#         self.class_texts = MetadataCatalog.get(train_dataset).stuff_classes
#         # self.test_class_texts = MetadataCatalog.get(test_dataset).stuff_classes
#         # catseg-plus (class names featuring attributes)
#         with open(f"/home/kanchana/data/kanchana/ood_seg/descriptors/{test_dataset}.json", 'r') as f_in:
#             data = json.load(f_in)
#         # with open(f"datasets/atlantis (1).json", 'r') as f_in:
#         #     data = json.load(f_in)
        
#         # with attributes
#         self.test_class_texts = [[f"A photo of a {key} featuring {attribute}" for attribute in attributes] for key, attributes in data.items()]
#         sc_variant = False
#         hierachy = ['']
#         if sc_variant:
#             super_class = [0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
#             hierachy_all_classes = []
#             for i in super_class:
#                 if i == 0:
#                     hierachy_all_classes.append([' '] * len(hierachy))
#                 else:
#                     hierachy_all_classes.append(hierachy)
#         else:
#             hierachy_all_classes = len(self.test_class_texts) * [hierachy]
#         self.test_class_texts = [list(data.keys()) , self.test_class_texts, hierachy_all_classes]
        
#         assert self.class_texts != None
#         if self.test_class_texts == None:
#             self.test_class_texts = self.class_texts
#         device = "cuda" if torch.cuda.is_available() else "cpu"
  
#         self.tokenizer = None
#         if clip_pretrained == "ViT-G" or clip_pretrained == "ViT-H":
#             # for OpenCLIP models
#             name, pretrain = ('ViT-H-14', 'laion2b_s32b_b79k') if clip_pretrained == 'ViT-H' else ('ViT-bigG-14', 'laion2b_s39b_b160k')
#             clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
#                 name, 
#                 pretrained=pretrain, 
#                 device=device, 
#                 force_image_size=336,)
        
#             self.tokenizer = open_clip.get_tokenizer(name)
#         else:
#             # for OpenAI models
#             clip_model, clip_preprocess = clip.load(clip_pretrained, device=device, jit=False, prompt_depth=prompt_depth, prompt_length=prompt_length)
    
#         self.prompt_ensemble_type = prompt_ensemble_type        

#         if self.prompt_ensemble_type == "imagenet_select":
#             prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
#         elif self.prompt_ensemble_type == "imagenet":
#             prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
#         elif self.prompt_ensemble_type == "single":
#             prompt_templates = ['A photo of a {} in the scene',]
#         else:
#             raise NotImplementedError
#         # self.text_features = self.class_embeddings(self.class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
#         # self.text_features_test = self.class_embeddings(self.test_class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()

#         # with attributes
#         self.text_features_test = self.class_embeddings(self.test_class_texts, prompt_templates, clip_model)
        
#         self.clip_model = clip_model.float()
#         self.clip_preprocess = clip_preprocess
        
#         transformer = Aggregator(
#             text_guidance_dim=text_guidance_dim,
#             text_guidance_proj_dim=text_guidance_proj_dim,
#             appearance_guidance_dim=appearance_guidance_dim,
#             appearance_guidance_proj_dim=appearance_guidance_proj_dim,
#             decoder_dims=decoder_dims,
#             decoder_guidance_dims=decoder_guidance_dims,
#             decoder_guidance_proj_dims=decoder_guidance_proj_dims,
#             num_layers=num_layers,
#             nheads=num_heads, 
#             hidden_dim=hidden_dims,
#             pooling_size=pooling_sizes,
#             feature_resolution=feature_resolution,
#             window_size=window_sizes,
#             attention_type=attention_type
#             )
#         self.transformer = transformer

#     @classmethod
#     def from_config(cls, cfg):#, in_channels, mask_classification):
#         ret = {}

#         # ret["train_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON
#         # ret["test_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON
#         assert len(cfg.DATASETS.TRAIN) == 1, "Current implementation of CATSeg only supports one training dataset"
#         ret["train_dataset"] = cfg.DATASETS.TRAIN[0]
#         assert len(cfg.DATASETS.TEST) == 1, "Current implementation of CATSeg only supports one test dataset"
#         ret["test_dataset"] = cfg.DATASETS.TEST[0]
#         ret["clip_pretrained"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED
#         ret["prompt_ensemble_type"] = cfg.MODEL.PROMPT_ENSEMBLE_TYPE

#         # Aggregator parameters:
#         ret["text_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM
#         ret["text_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM
#         ret["appearance_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM
#         ret["appearance_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM

#         ret["decoder_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS
#         ret["decoder_guidance_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS
#         ret["decoder_guidance_proj_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS

#         ret["prompt_depth"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH
#         ret["prompt_length"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH

#         ret["num_layers"] = cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS
#         ret["num_heads"] = cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS
#         ret["hidden_dims"] = cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS
#         ret["pooling_sizes"] = cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES
#         ret["feature_resolution"] = cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION
#         ret["window_sizes"] = cfg.MODEL.SEM_SEG_HEAD.WINDOW_SIZES
#         ret["attention_type"] = cfg.MODEL.SEM_SEG_HEAD.ATTENTION_TYPE

#         return ret

#     def forward(self, x, vis_guidance, global_image = None, OOM = False, tpt_tuning = False, tuned_text_embedding = None):
#         # x.shape = (1, 768, 24, 24)
#         # vis_guidance = {'res2': (1, 128, 96, 96), 'res3': (1, 256, 48, 48), 'res4': (1, 512, 24, 24)}
#         # print("in catsegpredictor forward")
#         vis = [vis_guidance[k] for k in vis_guidance.keys()][::-1]  
#         # print(f"vis shape: {vis.shape}")                            # vis[0].shape = (1, 512, 24, 24)
#         text = self.text_features if self.training else self.text_features_test                        # text.shape = (1, 80, 768)
#         # breakpoint()
#         text = text.repeat(x.shape[0], 1, 1, 1)      
#         print(f"text shape: {text.shape}")                                          # (n_classes, 80, 768)   ->    (1, n_classes, 80, 768)
#         # print(f"x shape: {x.shape}")
#         # breakpoint() 
#         # temp = 0
#         if OOM == False:
#             out = self.transformer(x, text, vis)

#         else:
#             size = [0,16,32,48,64]
#             # print(f"vis[o].shape: {vis[0].shape}")
#             out = []
#             for i in range(0, 32):
#                 viss = [vis[0][i,:,:,:].unsqueeze(0), vis[1][i,:,:,:].unsqueeze(0), vis[2][i,:,:,:].unsqueeze(0)]
#                 out.append(self.transformer(x[i,:,:,:].unsqueeze(0), text[i,:,:,:].unsqueeze(0), viss))
#             out = torch.cat(out, dim=0)
#             # breakpoint()
#         # out = self.transformer(x, text, vis)
#         # print(f"out shape: {out.shape}")
#         return out

#     # @torch.no_grad()
#     # def class_embeddings(self, classnames, templates, clip_model):
#     #     zeroshot_weights = []
#     #     for classname in classnames:
#     #         if ', ' in classname:
#     #             classname_splits = classname.split(', ')
#     #             texts = []
#     #             for template in templates:
#     #                 for cls_split in classname_splits:
#     #                     texts.append(template.format(cls_split))
#     #         else:
#     #             texts = [template.format(classname) for template in templates]  # format with class
#     #         if self.tokenizer is not None:
#     #             texts = self.tokenizer(texts).cuda()
#     #         else: 
#     #             texts = clip.tokenize(texts).cuda()
#     #         class_embeddings = clip_model.encode_text(texts)
#     #         class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
#     #         if len(templates) != class_embeddings.shape[0]:
#     #             class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
#     #             class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
#     #         class_embedding = class_embeddings
#     #         zeroshot_weights.append(class_embedding)
#     #     zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
#     #     return zeroshot_weights

#     # with attributes
#     @torch.no_grad()
#     def class_embeddings(self, classnames, templates, clip_model):
#         classesonly = classnames[0]
#         classattributes = classnames[1]  
#         hierachy = classnames[2] 

#         zeroshot_weights_lst = []
#         for classname in classesonly:
#             if ', ' in classname:
#                 classname_splits = classname.split(', ')
#                 texts = []
#                 for template in templates:
#                     for cls_split in classname_splits:
#                         texts.append(template.format(cls_split))
#             else:
#                 texts = [template.format(classname) for template in templates]  # format with class
#             if self.tokenizer is not None:
#                 texts = self.tokenizer(texts).cuda()
#             else: 
#                 texts = clip.tokenize(texts).cuda()

#             class_embeddings = clip_model.encode_text(texts) #(80, 768) This has become 80 because of the 80 imageNET templates
#             class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
#             if len(templates) != class_embeddings.shape[0]:
#                 class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
#                 class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
#             class_embedding = class_embeddings
#             zeroshot_weights_lst.append(class_embedding)
#         zeroshot_weights = torch.stack(zeroshot_weights_lst, dim=0).float().cuda()

#         final_embeddings = []
#         for _class in classattributes:
#             tokens = []
#             index = classattributes.index(_class)
#             for attribute in _class:
#                 attribute = [template.format(attribute) for template in templates]  # format with class
#                 if self.tokenizer is not None:
#                     texts = self.tokenizer(attribute).cuda()
#                 else: 
#                     texts = clip.tokenize(attribute).cuda()
#                 class_embeddings = clip_model.encode_text(texts)
#                 class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
#                 tokens.append(class_embeddings)
#             tokens = torch.stack(tokens, dim=0).squeeze(1)
#             tokens = tokens.to(torch.float32)
#             cosine_similarity = F.cosine_similarity(tokens, zeroshot_weights_lst[index], dim=-1).mean(dim=-1)
#             cosine_similarity = cosine_similarity / torch.sum(cosine_similarity)
#             cosine_similarity = cosine_similarity.view(cosine_similarity.shape[0], 1, 1)
#             tokens = tokens * cosine_similarity
#             tokens = tokens.mean(dim=0)
#             if len(tokens.shape) == 1:
#                 tokens = tokens.unsqueeze(0)
#             final_embeddings.append(tokens)
#         final_embeddings = torch.stack(final_embeddings, dim=0).cuda()

#         hierachy_embeddings = []
#         for _class in hierachy:
#             tokens = []
#             for _attribute in _class:
#                 attribute = [template.format(_attribute) for template in templates]  # format with class
#                 if self.tokenizer is not None:
#                     texts = self.tokenizer(attribute).cuda()
#                 else: 
#                     texts = clip.tokenize(attribute).cuda()
#                 class_embeddings = clip_model.encode_text(texts)
#                 class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
#                 if _attribute == ' ':
#                     class_embeddings = torch.zeros_like(class_embeddings)
#                 tokens.append(class_embeddings)
#             tokens = torch.stack(tokens, dim=0).squeeze(1)
#             tokens = tokens.to(torch.float32)
#             tokens = tokens.mean(dim=0)
#             if len(tokens.shape) == 1:
#                 tokens = tokens.unsqueeze(0)
#             hierachy_embeddings.append(tokens)
#         hierachy_embeddings = torch.stack(hierachy_embeddings, dim=0).cuda()


#         zeroshot_weights = 1 * zeroshot_weights + 0 * final_embeddings + 0.0 * hierachy_embeddings # (n_classes, 80, 768)

#         return zeroshot_weights