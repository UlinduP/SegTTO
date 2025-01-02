from line_profiler import profile
import random
import numpy as np
from PIL import Image
import torch
from einops import rearrange
import gc
import torch.nn as nn


import os
import sys
from copy import deepcopy
import math
from PIL import Image
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import time
import numpy as np
import gc

from .pcgrad import PCGrad

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tpt.custom_clip import get_coop
from tpt.data.datautils import AugMixAugmenter
from tpt.utils.tools import set_random_seed

class TPT:
    # @profile
    def __init__(self, classnames, imagenet_templates_tuning, tpt_steps=2, gpu=0, loss="pcgrad", templates_to_tune = None,attribute_embedding = None):
        self.data = 'data'
        self.test_sets = 'A/R/V/K/I'
        self.dataset_mode = 'test'
        self.arch = 'ViT-L/14@336px'
        self.resolution = 336
        self.workers = 4
        self.num_templates = 80 # None or 80
        self.lr = 5e-3
        self.print_freq = 200
        self.gpu = gpu
        self.tpt = True
        self.selection_p = 0.2
        self.tta_steps = tpt_steps
        self.n_ctx = 7
        self.cocoop = False
        self.load = None
        self.seed = 0
        self.num_augmentations = 64
        self.classnames = classnames
        self.learned_cls = True
        self.attribute_embedding = attribute_embedding
        self.loss = loss

        self.first_img = True
        self.threshold = 0.9
        self.loop_high = False
        self.loop_low = False

        # self.combine_loss = AutomaticWeightedLoss(2)  # GradNormLoss(2) 

        if imagenet_templates_tuning:
            self.num_templates = len(templates_to_tune)
        else:
            self.num_templates = None

        if self.num_templates and templates_to_tune is not None:
            # self.ctx_init = ['a_bad_photo_of_a', 'a_photo_of_many', 'a_sculpture_of_a', 'a_photo_of_the_hard_to_see', 'a_low_resolution_photo_of_the', 'a_rendering_of_a', 'graffiti_of_a', 'a_bad_photo_of_the', 'a_cropped_photo_of_the', 'a_tattoo_of_a', 'the_embroidered', 'a_photo_of_a_hard_to_see', 'a_bright_photo_of_a', 'a_photo_of_a_clean', 'a_photo_of_a_dirty', 'a_dark_photo_of_the', 'a_drawing_of_a', 'a_photo_of_my', 'the_plastic', 'a_photo_of_the_cool', 'a_close-up_photo_of_a', 'a_black_and_white_photo_of_the', 'a_painting_of_the', 'a_painting_of_a', 'a_pixelated_photo_of_the', 'a_sculpture_of_the', 'a_bright_photo_of_the', 'a_cropped_photo_of_a', 'a_plastic', 'a_photo_of_the_dirty', 'a_jpeg_corrupted_photo_of_a', 'a_blurry_photo_of_the', 'a_photo_of_the', 'a_good_photo_of_the', 'a_rendering_of_the', 'a_video_of_a', 'a_photo_of_one', 'a_doodle_of_a', 'a_close-up_photo_of_the', 'a_photo_of_a', 'the_origami', 'the_video_of_a', 'a_sketch_of_a', 'a_doodle_of_the', 'a_origami', 'a_low_resolution_photo_of_a', 'the_toy', 'a_rendition_of_the', 'a_photo_of_the_clean', 'a_photo_of_a_large', 'a_rendition_of_a', 'a_photo_of_a_nice', 'a_photo_of_a_weird', 'a_blurry_photo_of_a', 'a_cartoon', 'art_of_a', 'a_sketch_of_the', 'a_embroidered', 'a_pixelated_photo_of_a', 'itap_of_the', 'a_jpeg_corrupted_photo_of_the', 'a_good_photo_of_a', 'a_plushie', 'a_photo_of_the_nice', 'a_photo_of_the_small', 'a_photo_of_the_weird', 'the_cartoon', 'art_of_the', 'a_drawing_of_the', 'a_photo_of_the_large', 'a_black_and_white_photo_of_a', 'the_plushie', 'a_dark_photo_of_a', 'itap_of_a', 'graffiti_of_the', 'a_toy', 'itap_of_my', 'a_photo_of_a_cool', 'a_photo_of_a_small', 'a_tattoo_of_the']
            self.ctx_init = templates_to_tune #['a_photo_of_a', 'a_photo_of_a_clean', 'a_close-up_photo_of_a', 'a_blurry_photo_of_the', 'a_bright_photo_of_the']
            self.ctx_init = self.ctx_init[:self.num_templates]
        else:
            self.ctx_init = 'a_photo_of_a'

        set_random_seed(self.seed)
        model = get_coop(self.arch, self.classnames, self.num_templates, self.gpu, self.n_ctx, self.ctx_init, self.learned_cls, self.attribute_embedding)
        if self.load is not None:
            print("Use pre-trained soft prompt (CoOp) as initialization")
            pretrained_ctx = torch.load(self.load)['state_dict']['ctx']
            assert pretrained_ctx.size()[0] == self.n_ctx
            with torch.no_grad():
                model.prompt_learner[0].ctx.copy_(pretrained_ctx)
                model.prompt_learner[0].ctx_init_state = pretrained_ctx
        self.model_state = None
        model.reset_classnames(self.classnames, self.arch)
        self.model = model

        for name, param in model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        
        print("=> Model created: visual backbone {}".format(self.arch))
        
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
        else:
            assert self.gpu is not None
            torch.cuda.set_device(self.gpu)
            self.model = model.cuda(self.gpu)

        trainable_param = self.model.prompt_learner.parameters()

        if self.loss=="pcgrad":
            self.optimizer = PCGrad(torch.optim.AdamW(trainable_param, self.lr))
            self.optim_state = deepcopy(self.optimizer.optimizer.state_dict())
        
        elif loss=="entropy_only":
            self.optimizer = torch.optim.AdamW(trainable_param, self.lr)
            self.optim_state = deepcopy(self.optimizer.state_dict())

        # setup automatic mixed-precision (Amp) loss scaling
        self.scaler = GradScaler(init_scale=1000)

        print('=> Using native Torch AMP. Training in mixed precision.')
        # mixed precision training code here

        # norm stats from clip.load()
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])

        base_transform = transforms.Compose([
            transforms.Resize(self.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(self.resolution)])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        self.data_transform = AugMixAugmenter(base_transform, preprocess, n_views=self.num_augmentations-1)
        # batchsize = 1
        del model, trainable_param
    # @profile
    def tpt_tuning(self, image):
        torch.backends.cudnn.benchmark = True
        # This codebase has only been tested under the single GPU setting
        assert self.gpu is not None
        # print("Use GPU: {} for training".format(self.gpu))
        
        image = transforms.ToPILImage()(image.squeeze(0))
        transformed_images = self.data_transform(image)
            
        text_embedding = self.test_time_adapt_eval(transformed_images, self.optimizer, self.optim_state, self.scaler)
        del transformed_images, image
        return text_embedding

    # @profile
    def test_time_adapt_eval(self, images, optimizer, optim_state, scaler):
        self.model.eval()
        with torch.no_grad():
            self.model.reset()
        assert self.gpu is not None

        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(self.gpu, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(self.gpu, non_blocking=True)
            image = images
        images = torch.stack(images, dim=0)

        if self.tta_steps > 0:
            with torch.no_grad():
                self.model.reset()

        if self.loss=="pcgrad":
            optimizer.optimizer.load_state_dict(optim_state)
        elif self.loss=="entropy_only":
            optimizer.load_state_dict(optim_state)

        self.test_time_tuning(images, optimizer, scaler)

        with torch.no_grad():
            image = image.unsqueeze(0)
            logits, text_features = self.model(image)
        del image, images, optimizer, optim_state, scaler, logits
        return text_features
    # @profile
    def select_confident_samples(self, logits, top): # logits = (64, 577, num_classes) //// (64, 577, 80, num_classes)
        if self.num_templates is not None:
            logits_ = logits[:, 1:, :, :].permute(0, 2, 3, 1) # (64, 80, num_classes, 576)
            logits_ = logits_.view(logits_.shape[0], logits_.shape[1], logits_.shape[2], int(math.sqrt(logits_.shape[3])), int(math.sqrt(logits_.shape[3]))) # (64, 80, num_classes, 24, 24)
            patch_entropy = -(logits_.softmax(2) * logits_.log_softmax(2)).mean(3).mean(3).mean(1).mean(1) # (64,)
            idx = torch.argsort(patch_entropy, descending=False)[:int(patch_entropy.size()[0] * top)] # (6,)
            selected_logits = logits[idx][:, :, :, :] # (6, 5, 577, num_classes)
        else:
            logits_ = logits[:, 1:, :].permute(0, 2, 1) # (64, num_classes, 576)
            logits_ = logits_.view(logits_.shape[0], logits_.shape[1], int(math.sqrt(logits_.shape[2])), int(math.sqrt(logits_.shape[2]))) # (64, num_classes, 24, 24)
            # batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
            # idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
            # return logits[idx], idx
            # batch_entropy = -(logits.softmax(2) * logits.log_softmax(2)).sum(2).sum(2).sum(2).sum(1)
            patch_entropy = -(logits_.softmax(1) * logits_.log_softmax(1)).mean(2).mean(2) # (64, num_classes)
            idx = torch.argsort(patch_entropy, descending=False)[:int(patch_entropy.size()[0] * top)]
            # idx = torch.argsort(avg_patch_entropy.view(avg_patch_entropy.size(0), -1), dim=0, descending=False)[:int(patch_entropy.size()[0] * top)]
            selected_logits = logits[idx][:, :, :]
            # plot_entropy = selected_logits.mean(0).mean(0)        
            # import matplotlib.pyplot as plt
            # import time
            # plt.figure(figsize=(10, 5))
            # plt.bar(range(len(plot_entropy.detach().cpu().numpy())), plot_entropy.detach().cpu().numpy())
            # plt.savefig(f'plots/plot_entropy_{time.time()}.png')
            # plt.show()
        del logits_, patch_entropy, logits
        return selected_logits, idx
    # @profile
    def avg_entropy(self, outputs):
        #outout shape = (selected aug, num_classes) //// (selected aug, 80, num_classes)
        # remove dim=2 for logits                           (6,577,5,c) 
        outputs = outputs[:,0,:,:] #                      (6,577,5,c)
        logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
        avg_logits = logits.logsumexp(dim=0) - torch.log(torch.tensor(logits.shape[0])) # avg_logits = logits.mean(0) [1, 1000]
        min_real = torch.finfo(avg_logits.dtype).min
        logits = torch.clamp(avg_logits, min=min_real)
        # scale = torch.log(torch.tensor(logits.shape[-1])).to(self.gpu)
        # breakpoint()
        value = (-(logits * torch.exp(logits)).sum(dim=-1).mean(0))
        # breakpoint()
        del outputs, logits, avg_logits
        return value
    
    def cross_entropy(self, outputs):
        outputs = outputs[:,:,:,:] #                      (6,577,5,c)
        b, wh, p, c = outputs.shape
        print(outputs.shape)
        outputs = outputs.permute(0,3,1,2) #              (6,5,577,c)
        logits = torch.softmax(outputs/1,dim=1) # logits = outputs.log_softmax(dim=1) [N, 1000]  ########## gamma = 20
        # avg_logits = logits.logsumexp(dim=0) - torch.log(torch.tensor(logits.shape[0])) # avg_logits = logits.mean(0) [1, 1000]
        min_real = torch.finfo(logits.dtype).min
        logits = torch.clamp(logits, min=min_real)
        # gt = torch.argmax(logits, dim=-1, keepdim=True)
        # target = torch.zeros_like(logits).scatter_(-1, gt, 1)
        if self.first_img == True:
            while True:
                target = (logits>self.threshold).float()
                ones = torch.sum(target==1).item()
                print(f"number of ones: {ones} from {b*p*wh}")
                if ones < int(8300*p/5) and self.threshold > 0.7 and not self.loop_low:
                    self.threshold -= 0.01
                    self.loop_high = True
                elif ones > int(8700*p/5) and self.threshold > 0.7 and not self.loop_high:
                    self.threshold += 0.01
                    self.loop_low = True
                else:
                    print("......................")
                    print(f"threshold: {self.threshold}")
                    self.first_img = False
                    break
        target = (logits>self.threshold).float()
        ones = torch.sum(target==1).item()
        print(f"number of ones: {ones} from {b*p*wh}")
        # loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)/(b*p)
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(logits,target)/(b*p)
        # breakpoint()
        return loss

    # @profile
    def test_time_tuning(self, inputs, optimizer, scaler):
        for j in range(self.tta_steps+1):
            optimizer.zero_grad()
            with autocast():
                output, _ = self.model(inputs)
                # global_image = output[0].unsqueeze(0).repeat(int(output.size(0) * self.selection_p), 1, 1)
                output_conf, selected_idx = self.select_confident_samples(output, self.selection_p)
                if j<self.tta_steps:
                    if self.loss=="pcgrad":
                        loss_ent = self.avg_entropy(output_conf)
                        loss_cross = self.cross_entropy(output_conf)
                        loss = [loss_ent, loss_cross]
                        print(f'loss ent: {loss_ent} loss cross: {loss_cross}')
                    elif self.loss=="entropy_only":
                        loss_ent = self.avg_entropy(output_conf)
                        loss = loss_ent
                        print(f'loss ent: {loss_ent}')
                    # assert not torch.isnan(loss[1]).any(), "NaN found in loss"
                    # assert not torch.isinf(loss[1]).any(), "Inf found in loss"
                    # assert not torch.isnan(loss[0]).any(), "NaN found in loss"
                    # assert not torch.isinf(loss[0]).any(), "Inf found in loss"
                else:
                    loss_cross = self.cross_entropy(output_conf)
                    if self.loss=="pcgrad":
                        loss = [loss_cross]
                    elif self.loss=="entropy_only":
                        loss = loss_cross
                    
                    # assert not torch.isnan(loss[0]).any(), "NaN found in loss"
                    # assert not torch.isinf(loss[0]).any(), "Inf found in loss"
                    print(f'loss cross: {loss_cross}')                
                # loss = self.combine_loss(loss_ent, loss_cross)
                # loss = (loss_ent + 100*loss_cross)
                # cross entropy loss global image vs selected images
                # loss = torch.nn.functional.cross_entropy(global_image, output)
                # loss.backward()    
            # optimizer.step()
            # plt.figure(figsize=(10, 5))
            # plt.bar(range(len(torch.softmax(output[0,0,:,:].mean(0),dim=-1).detach().cpu().numpy())), torch.softmax(output[0,0].mean(0),dim=-1).detach().cpu().numpy())
            # plt.savefig(f'plots/target_{time.time()}.png')
            # # plt.show()    
            # print('before step', self.model.prompt_learner.cls.mean())
            if self.loss == "pcgrad":
                optimizer.pc_backward(loss) 
                optimizer.step()
            elif self.loss == "entropy_only":
                scaler.scale(loss).backward()          
                scaler.step(optimizer)
                scaler.update()
                 
            # print('after step', self.model.prompt_learner.cls.mean())
            # Assert no NaN or Inf in loss 

            # Assert no NaN or Inf in gradients of model parameters
            # for name, param in self.model.named_parameters():   
            #     if param.grad is not None:
            #         assert not torch.isnan(param.grad).any(), f"NaN found in gradients of {name}"
            #         assert not torch.isinf(param.grad).any(), f"Inf found in gradients of {name}"
        return
    

    

def random_crop_and_resize(image, resize_to, num_crops=64):
    """
    image : torch.Tensor
    resize_to : tuple (width, height) to resize the image to

    Returns:
    croped_images : list of torch.Tensor
    crop_coords : list of tuples
    """
    # Get the original size
    # width, height = image.size
    # crop_size=random.randint(24,224)

    # # Randomly select a starting point for the crop
    # x = random.randint(0, width - crop_size)
    # y = random.randint(0, height - crop_size)
    random.seed(0)
    image = image.cpu().numpy()
    
    if isinstance(image, np.ndarray):
        image = np.transpose(image, (1, 2, 0))
        # Convert numpy array to PIL Image
        image = Image.fromarray(image.astype(np.uint8))

    cropped_images = []
    cropped_coords = []
    for i in range(num_crops):
        # Randomly select a starting point for the crop
        #######
        x1,y1 = random.randint(0,639), random.randint(0,639)
        crop_size = random.randint(1, 640 - max(x1,y1))
        #######

        # Perform the crop
        cropped_image = image.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        # Resize back to 224x224
        resized_image = cropped_image.resize(resize_to)
        resized_image = torch.Tensor(np.transpose(resized_image.copy(), (2, 0, 1)))
        # print(resized_image.shape)

        cropped_images.append(resized_image)
        cropped_coords.append((x1, y1, x1 + crop_size, y1 + crop_size))
        
        # if i % 10 == 0:
        #     gc.collect()
        #     torch.cuda.empty_cache()
    
    cropped_images = torch.stack(cropped_images, dim=0)
    # print(cropped_images.shape)
    
    del image, cropped_image, resized_image
    gc.collect()
    torch.cuda.empty_cache()

    # Return the resized images and crop coordinates
    return cropped_images, cropped_coords

def select_confident_samples(logits, top, patch_level=False, aggregration="median",model="catseg"):
    """
    args:
        logits: torch.Tensor shape [batch_size, num_classes, height, width]
        top: float, percentage of samples to select
    
    Returns:
        selected_logits: torch.Tensor [selected_batch_size, num_classes, height, width]
        idx: torch.Tensor [selected_batch_size]
    """
    if patch_level:
        print("Patch level selection:", aggregration)
        print(f"logits shape: {logits.shape}")
        patch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        print(f"patch_entropy shape: {patch_entropy.shape}")
        
        if aggregration == "mean":
            aggregated_entropy = patch_entropy.mean(dim=1).mean(dim=1)
        elif aggregration == "max":
            aggregated_entropy = patch_entropy.max(dim=1)[0].max(dim=1)[0] 
        elif aggregration == "min":
            aggregated_entropy = patch_entropy.min(dim=1)[0].min(dim=1)[0]
        elif aggregration == "median":
            aggregated_entropy = patch_entropy.median(dim=1)[0].median(dim=1)[0]
        
        print(f"aggregated_entropy shape: {aggregated_entropy.shape}")
        idx = torch.argsort(aggregated_entropy, descending=False)[:int(aggregated_entropy.size()[0] * top)]
        print(f"idx shape: {idx.shape}")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return None,idx
    else:
        batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(2).sum(2).sum(1)
        # print(f"batch_entropy shape: {batch_entropy.shape}")
        idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
        # print(f"idx shape: {idx.shape}")
        gc.collect()
        torch.cuda.empty_cache()
        return logits[idx], idx  


@profile
def folding(crop_features, output_size = (640, 640), kernel_size = (384, 384), overlap = 0.333):    
    
    patches = rearrange(crop_features, 'b c h w -> c (h w) b')
    stride = int(kernel_size[0] * (1 - overlap))  

    fold = nn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride)
    # folded_feature = fold(patches)

    # ones_input = torch.ones_like(patches)  
    # unfold_ones = fold(ones_input)
    # folded_feature /= unfold_ones
    
    folded_feature = fold(patches) / fold(torch.ones_like(patches))

    folded_feature = rearrange(folded_feature, 'd b h w -> b d h w')

    del patches
    gc.collect()
    torch.cuda.empty_cache()

    return folded_feature

@profile
def unfolding(folded_feature, h_w=(384, 384), kernel_size=(384, 384), overlap=0.333):
    folded_feature = rearrange(folded_feature, 'b d h w -> d b h w')
    
    height, width = h_w
    stride = int(kernel_size[0] * (1 - overlap))  

    unfold = nn.Unfold(kernel_size=kernel_size, stride=stride)
    unfolded = unfold(folded_feature)
    unfolded = rearrange(unfolded, 'd (h w) b -> b d h w', h=height, w=width)

    del folded_feature
    gc.collect()
    torch.cuda.empty_cache()

    return unfolded

@profile
def process_clip_feature(clip_feature, aug_coords, interpolation="bilinear", selected_idxes=None, size=24):
    # aug_coords = [[x1, y1, x1 + crop_size, y1 + crop_size], [x2, y2, x2 + crop_size, y2 + crop_size], ...]
    # patch 1 = [0, 0, 384, 384], patch 2 = [256, 0, 640, 384], patch 3 = [0, 256, 384, 640], patch 4 = [256, 256, 640, 640]
    
    orig_feature = clip_feature[4,1:,:].unsqueeze(0) # [1, 576, 768]
    orig_feature = rearrange(orig_feature, "b (h w) c -> b c h w", h=24, w=24) # [1, 768, 24, 24]
    orig_feature = torch.nn.functional.interpolate(orig_feature, size=(640, 640), mode=interpolation) # [1, 768, 640, 640]
    
    crop_feature = clip_feature[:4,1:,:] # [4, 576, 768]`
    # print(f"crop_feature shape: {crop_feature.shape}")
    crop_feature = rearrange(crop_feature, "b (h w) c -> b c h w", h=24, w=24) # [4, 768, 24, 24]
    crop_feature = torch.nn.functional.interpolate(crop_feature, size=(384, 384), mode=interpolation) # [4, 768, 384, 384]
    crop_feature = folding(crop_feature)

    # print(f"orig_feature shape: {orig_feature.shape}, crop_feature_fold shape: {crop_feature_fold.shape}")

    count_tensor = torch.ones((640, 640)).to(clip_feature.device)

    for idx in selected_idxes:
        crop_size = aug_coords[idx][2] - aug_coords[idx][0]
        y1, y2 = aug_coords[idx][1], aug_coords[idx][3]
        x1, x2 = aug_coords[idx][0], aug_coords[idx][2]

        feature = clip_feature[idx+5,1:,:].unsqueeze(0) # [1, 576, 768]
        feature = rearrange(feature, "b (h w) c -> b c h w", h=24, w=24) # [1, 768, 24, 24]
        feature = torch.nn.functional.interpolate(feature, size=(crop_size, crop_size), mode=interpolation) # [1, 768, crop_size, crop_size]
    
        orig_feature[:, :, y1:y2, x1:x2] = (orig_feature[:, :, y1:y2, x1:x2] + feature)
        crop_feature[:, :, y1:y2, x1:x2] = (crop_feature[:, :, y1:y2, x1:x2] + feature)

        count_tensor[y1:y2, x1:x2] += 1

    orig_feature = orig_feature / count_tensor.unsqueeze(0).unsqueeze(0)
    crop_feature = crop_feature / count_tensor.unsqueeze(0).unsqueeze(0)
    
    crop_feature = unfolding(crop_feature)
    
    orig_feature = torch.nn.functional.interpolate(orig_feature, size=(24, 24), mode=interpolation) # [1, 768, 24, 24]
    crop_feature = torch.nn.functional.interpolate(crop_feature, size=(24, 24), mode=interpolation) # [4, 768, 24, 24]
    
    crop_feature = torch.cat((crop_feature, orig_feature), dim=0)
    
    crop_feature = rearrange(crop_feature, "b d h w -> b (h w) d") # [5, 576, 768]
    # breakpoint()
    crop_feature = torch.cat((clip_feature[:5,:1,:], crop_feature), dim=1) # [5, 577, 768]
    # clip_feature_updated = torch.nn.functional.interpolate(clip_feature_updated, size=(24, 24), mode=interpolation) # [5, 768, 24, 24]

    del orig_feature, clip_feature
    gc.collect()
    torch.cuda.empty_cache()

    return crop_feature

@profile
def process_backbone_feature(features, aug_coords, interpolation="bilinear", selected_idxes=None, size=24):
    """
    Processes the backbone features (res2, res3, res4) and returns the aggregated feature.

    Args:
        features: dict of torch.Tensor shape 
        aug_coords: list of tuples (x1, y1, x2, y2)
        interpolation: nearest, bilinear
    """

    orig_features_res2 = torch.nn.functional.interpolate(features["res2"][4,:,:,:].unsqueeze(0), size=(640,640), mode=interpolation) # [1, 128, 640, 640]
    orig_features_res3 = torch.nn.functional.interpolate(features["res3"][4,:,:,:].unsqueeze(0), size=(640,640), mode=interpolation) # [1, 256, 640, 640]
    orig_features_res4 = torch.nn.functional.interpolate(features["res4"][4,:,:,:].unsqueeze(0), size=(640,640), mode=interpolation) # [1, 512, 640, 640]
    
    crop_features_res2 = torch.nn.functional.interpolate(features["res2"][:4,:,:,:], size=(384,384), mode=interpolation) # [4, 128, 384, 384]
    crop_features_res3 = torch.nn.functional.interpolate(features["res3"][:4,:,:,:], size=(384,384), mode=interpolation) # [4, 256, 384, 384]
    crop_features_res4 = torch.nn.functional.interpolate(features["res4"][:4,:,:,:], size=(384,384), mode=interpolation) # [4, 512, 384, 384]

    crop_features_res2 = folding(crop_features_res2)
    crop_features_res3 = folding(crop_features_res3)
    crop_features_res4 = folding(crop_features_res4)

    # print(f"orig_features_res2 shape: {orig_features_res2.shape}, crop_features_fold_res2 shape: {crop_features_fold_res2.shape}")

    count_tensor = torch.ones((640, 640)).to(crop_features_res2.device)

    for idx in selected_idxes:
        crop_size = aug_coords[idx][2] - aug_coords[idx][0]
        y1, y2 = aug_coords[idx][1], aug_coords[idx][3]
        x1, x2 = aug_coords[idx][0], aug_coords[idx][2]

        feature_res2 = torch.nn.functional.interpolate(features["res2"][idx+5,:,:,:].unsqueeze(0), size=(crop_size,crop_size), mode=interpolation) # [1, 128, crop_size, crop_size]
        feature_res3 = torch.nn.functional.interpolate(features["res3"][idx+5,:,:,:].unsqueeze(0), size=(crop_size,crop_size), mode=interpolation) # [1, 256, crop_size, crop_size]
        feature_res4 = torch.nn.functional.interpolate(features["res4"][idx+5,:,:,:].unsqueeze(0), size=(crop_size,crop_size), mode=interpolation) # [1, 512, crop_size, crop_size]

        orig_features_res2[:, :, y1:y2, x1:x2] = (orig_features_res2[:, :, y1:y2, x1:x2] + feature_res2)
        orig_features_res3[:, :, y1:y2, x1:x2] = (orig_features_res3[:, :, y1:y2, x1:x2] + feature_res3)
        orig_features_res4[:, :, y1:y2, x1:x2] = (orig_features_res4[:, :, y1:y2, x1:x2] + feature_res4)

        crop_features_res2[:, :, y1:y2, x1:x2] = (crop_features_res2[:, :, y1:y2, x1:x2] + feature_res2)
        crop_features_res3[:, :, y1:y2, x1:x2] = (crop_features_res3[:, :, y1:y2, x1:x2] + feature_res3)
        crop_features_res4[:, :, y1:y2, x1:x2] = (crop_features_res4[:, :, y1:y2, x1:x2] + feature_res4)

        count_tensor[y1:y2, x1:x2] += 1
    
    orig_features_res2 = orig_features_res2 / count_tensor.unsqueeze(0).unsqueeze(0)
    orig_features_res3 = orig_features_res3 / count_tensor.unsqueeze(0).unsqueeze(0)
    orig_features_res4 = orig_features_res4 / count_tensor.unsqueeze(0).unsqueeze(0)

    crop_features_res2 = crop_features_res2 / count_tensor.unsqueeze(0).unsqueeze(0)
    crop_features_res3 = crop_features_res3 / count_tensor.unsqueeze(0).unsqueeze(0)
    crop_features_res4 = crop_features_res4 / count_tensor.unsqueeze(0).unsqueeze(0)

    # crop_features_unfold_res2 = unfolding(crop_features_fold_res2)
    # crop_features_unfold_res3 = unfolding(crop_features_fold_res3)
    # crop_features_unfold_res4 = unfolding(crop_features_fold_res4)
    
    orig_features_res2 = torch.nn.functional.interpolate(orig_features_res2, size=(96,96), mode=interpolation)
    orig_features_res3 = torch.nn.functional.interpolate(orig_features_res3, size=(48,48), mode=interpolation)
    orig_features_res4 = torch.nn.functional.interpolate(orig_features_res4, size=(24,24), mode=interpolation)
    
    crop_features_res2 = torch.nn.functional.interpolate(unfolding(crop_features_res2), size=(96,96), mode=interpolation)
    crop_features_res3 = torch.nn.functional.interpolate(unfolding(crop_features_res3), size=(48,48), mode=interpolation)
    crop_features_res4 = torch.nn.functional.interpolate(unfolding(crop_features_res4), size=(24,24), mode=interpolation)
    
    crop_features_res2 = torch.cat((crop_features_res2, orig_features_res2), dim=0)
    crop_features_res3 = torch.cat((crop_features_res3, orig_features_res3), dim=0)
    crop_features_res4 = torch.cat((crop_features_res4, orig_features_res4), dim=0)
    
    # crop_features_unfold_res2 = torch.cat((crop_features_unfold_res2, orig_features_res2), dim=0)
    # crop_features_unfold_res3 = torch.cat((crop_features_unfold_res3, orig_features_res3), dim=0)
    # crop_features_unfold_res4 = torch.cat((crop_features_unfold_res4, orig_features_res4), dim=0)
    
    # crop_features_unfold_res2 = torch.nn.functional.interpolate(crop_features_unfold_res2, size=(96,96), mode=interpolation)
    # crop_features_unfold_res3 = torch.nn.functional.interpolate(crop_features_unfold_res3, size=(48,48), mode=interpolation)
    # crop_features_unfold_res4 = torch.nn.functional.interpolate(crop_features_unfold_res4, size=(24,24), mode=interpolation)

    del orig_features_res2, orig_features_res3, orig_features_res4
    gc.collect()
    torch.cuda.empty_cache()

    return {"res2": crop_features_res2, "res3": crop_features_res3, "res4": crop_features_res4}