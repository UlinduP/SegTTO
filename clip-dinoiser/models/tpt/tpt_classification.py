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

# from .pcgrad import PCGrad

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

class GradNormLoss(torch.nn.Module):
    def __init__(self, num_of_task, alpha=1.5):
        super(GradNormLoss, self).__init__()
        self.num_of_task = num_of_task  # Total number of tasks
        self.alpha = alpha  # Alpha value for adjusting relative losses
        self.w = torch.nn.Parameter(torch.ones(num_of_task, dtype=torch.float))  # Task-specific weights
        self.l1_loss = torch.nn.L1Loss()  # L1 Loss for regularization
        self.L_0 = None  # Reference to initial losses for each task
        self.optimizer = torch.optim.Adam(self.w, lr=0.025)

    def forward(self, L_t: torch.Tensor):
        """ Compute the total weighted loss for the tasks. """
        # Initialize the initial losses `Li_0` if not already done
        if self.L_0 is None:
            self.L_0 = L_t.detach()  # Detach to prevent gradients
        # Compute the weighted losses `w_i(t) * L_i(t)`
        self.additional_forward_and_backward(self.w,self.optimizer,L_t)
        self.wL_t = L_t * self.w
        # Sum up the weighted losses
        self.total_loss = self.wL_t.sum()
        return self.total_loss

    def additional_forward_and_backward(self, grad_norm_weights: torch.nn.Module, optimizer: torch.optim.Optimizer, L_t: torch.Tensor):
        """ Perform additional forward and backward pass to adjust task weights. 
            grad_norm_weights: the layers for computing gradients
            optimizer: optimizer for updating weights of losses      
        """
        # Perform standard backward pass on the total loss
        self.total_loss.backward(retain_graph=True)
        # Reset gradients for task weights as they shouldn't be updated in this pass
        self.w.grad.zero_()

        # Calculate the gradients of the loss w.r.t. the shared parameters and their norms
        GW_t = [torch.norm(torch.autograd.grad(
                self.wL_t[i], grad_norm_weights.parameters(), retain_graph=True, create_graph=True)[0])
                for i in range(self.num_of_task)]
        self.GW_t = torch.stack(GW_t)  # Stack to create a single tensor

        # Calculate average gradient norms
        self.bar_GW_t = self.GW_t.mean().detach()
        # Calculate normalized losses and relative inverse training rates
        self.tilde_L_t = (L_t / self.L_0).detach()
        self.r_t = self.tilde_L_t / self.tilde_L_t.mean()
        # Calculate the gradient normalization loss
        grad_loss = self.l1_loss(self.GW_t, self.bar_GW_t * (self.r_t ** self.alpha))

        # Compute gradients for the task weights
        self.w.grad = torch.autograd.grad(grad_loss, self.w, only_inputs=True)[0]
        # Update weights using optimizer
        optimizer.step()

        # Re-normalize weights to keep their sum constant
        self.w.data = self.w.data / self.w.data.sum() * self.num_of_task

        # Clear intermediate variables to free memory
        self.GW_t, self.bar_GW_t, self.tilde_L_t, self.r_t, self.wL_t = None, None, None, None, None


class AutomaticWeightedLoss(torch.nn.Module):
    """
    Automatically weighted multi-task loss.

    Params:
        num: int
            The number of loss functions to combine.
        x: tuple
            A tuple containing multiple task losses.

    Examples:
        loss1 = 1
        loss2 = 2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        # Initialize parameters for weighting each loss, with gradients enabled
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *losses):
        """
        Forward pass to compute the combined loss.

        Args:
            *losses: Variable length argument list of individual loss values.

        Returns:
            torch.Tensor: The combined weighted loss.
        """
        loss_sum = 0
        for i, loss in enumerate(losses):
            # Compute the weighted loss component for each task
            weighted_loss = 0.5 / (self.params[i] ** 2) * loss
            # Add a regularization term to encourage the learning of useful weights
            regularization = torch.log(1 + self.params[i] ** 2)
            # Sum the weighted loss and the regularization term
            loss_sum += weighted_loss + regularization

        return loss_sum

class TPT:
    # @profile
    def __init__(self, classnames, imagenet_templates_tuning, templates_to_tune = None,attribute_embedding = None):
        self.data = 'data'
        self.test_sets = 'A/R/V/K/I'
        self.dataset_mode = 'test'
        self.arch = 'ViT-B/32'
        self.resolution = 336
        self.workers = 4
        self.num_templates = 80 # None or 80
        self.lr = 5e-3
        self.print_freq = 200
        self.gpu = 0
        self.tpt = True
        self.selection_p = 0.2
        self.tta_steps = 2
        self.n_ctx = 7
        self.cocoop = False
        self.load = None
        self.seed = 0
        self.num_augmentations = 64
        self.classnames = classnames
        self.learned_cls = True
        self.attribute_embedding = attribute_embedding

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
        self.optimizer = torch.optim.AdamW(trainable_param, self.lr)
        # self.optimizer = PCGrad(torch.optim.AdamW(trainable_param, self.lr))
        self.optim_state = deepcopy(self.optimizer.state_dict())
        # self.optim_state = deepcopy(self.optimizer.optimizer.state_dict())

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

    # @profile
    def tpt_tuning(self, image):
        torch.backends.cudnn.benchmark = True
        # This codebase has only been tested under the single GPU setting
        assert self.gpu is not None
        # print("Use GPU: {} for training".format(self.gpu))
        
        image = transforms.ToPILImage()(image.squeeze(0))
        transformed_images = self.data_transform(image)
            
        text_embedding = self.test_time_adapt_eval(transformed_images, self.optimizer, self.optim_state, self.scaler)
        
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
        optimizer.load_state_dict(optim_state)
        # optimizer.optimizer.load_state_dict(optim_state)
        self.test_time_tuning(images, optimizer, scaler)

        with torch.no_grad():
            image = image.unsqueeze(0)
            logits, text_features = self.model(image)
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

    # def entropy_loss(self, v):
    #     v = v[:,1:,:,:]
    #     v = v.permute(0,3,1,2)
    #     l1_norm = torch.norm(v, p=1, dim=1, keepdim=True)
    #     v = v/l1_norm
    #     # breakpoint()
    #     assert v.dim() == 4
    #     n, c, hw, p = v.size()
    #     return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * hw * p * np.log2(c))

    # @profile
    def test_time_tuning(self, inputs, optimizer, scaler):
        for j in range(self.tta_steps):
            optimizer.zero_grad()
            with autocast():
                output, _ = self.model(inputs)
                # global_image = output[0].unsqueeze(0).repeat(int(output.size(0) * self.selection_p), 1, 1)
                output_conf, selected_idx = self.select_confident_samples(output, self.selection_p)
                if j<self.tta_steps:
                    loss_ent = self.avg_entropy(output_conf)
                    # loss_cross = self.cross_entropy(output_conf)
                    # loss = self.combine_loss(loss_ent, loss_cross)
                    loss = loss_ent
                    # print(f'loss ent: {loss_ent}')
                    # assert not torch.isnan(loss[1]).any(), "NaN found in loss"
                    # assert not torch.isinf(loss[1]).any(), "Inf found in loss"
                    # assert not torch.isnan(loss).any(), "NaN found in loss"
                    # assert not torch.isinf(loss).any(), "Inf found in loss"
                else:
                    loss_cross = self.cross_entropy(output_conf)
                    # loss = loss_cross
                    loss = [loss_cross]
                    # print(f'loss cross: {loss_cross}')  
                    # assert not torch.isnan(loss[0]).any(), "NaN found in loss"
                    # assert not torch.isinf(loss[0]).any(), "Inf found in loss"              
                # loss = self.combine_loss(loss_ent, loss_cross)
                # loss = (loss_ent + 100*loss_cross)
                # cross entropy loss global image vs selected images
                # loss = torch.nn.functional.cross_entropy(global_image, output)
                # loss.backward()    
            # optimizer.step()
            # plt.figure(figsize=(10, 5))
            # plt.bar(range(len(torch.softmax(output[0,0,:,:].mean(0),dim=-1).detach().cpu().numpy())), torch.softmax(output[0,0].mean(0),dim=-1).detach().cpu().numpy())
            # plt.savefig(f'plots/target_{time.time()}.png')
            # plt.show()
            scaler.scale(loss).backward()  
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)          
            scaler.step(optimizer)
            scaler.update()    
            # print('before step', self.model.prompt_learner.cls.mean())
            # optimizer.pc_backward(loss) 
            # optimizer.step() 
            # print('after step', self.model.prompt_learner.cls.mean())
            # Assert no NaN or Inf in loss

            # Assert no NaN or Inf in gradients of model parameters
            # for name, param in self.model.named_parameters():   
            #     if param.grad is not None:
            #         assert not torch.isnan(param.grad).any(), f"NaN found in gradients of {name}"
            #         assert not torch.isinf(param.grad).any(), f"Inf found in gradients of {name}"

        return