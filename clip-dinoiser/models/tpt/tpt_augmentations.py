import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def random_crop_and_resize(image: torch.Tensor, min_crop_size=1, max_crop_size=448, resize_size=(448, 448), num_crops=64):
    """
    Perform random square crops of random sizes on an image and resize them to a specified size.
    
    Args:
        image (torch.Tensor): The input image of shape (1, 3, H, W).
        min_crop_size (int): The minimum size of the crop (height and width).
        max_crop_size (int): The maximum size of the crop (height and width).
        resize_size (tuple): The size to resize each crop (height, width).
        num_crops (int): The number of random crops to generate.
    
    Returns:
        cropped_imgs (torch.Tensor): The cropped and resized images of shape (num_crops, 3, resize_size[0], resize_size[1]).
        cropped_coords (list): A list of coordinates for each crop in the format (x1, y1, x2, y2).
    """
    random.seed(0)
    _, C, H, W = image.shape
    cropped_imgs = [image.squeeze(0)]
    cropped_coords = []

    for _ in range(num_crops):
        # Ensure the crop size is square
        crop_size = random.randint(min_crop_size, min(max_crop_size, min(H, W)))

        x1 = random.randint(0, W - crop_size)
        y1 = random.randint(0, H - crop_size)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        crop = image[:, :, y1:y2, x1:x2]
        cropped_coords.append((x1, y1, x2, y2))

        resized_crop = TF.resize(crop, resize_size)
        cropped_imgs.append(resized_crop.squeeze(0))  

    cropped_imgs = torch.stack(cropped_imgs)

    return cropped_imgs, cropped_coords

# def random_crop_and_resize(image, resize_to, num_crops=64):
#     """
#     image : torch.Tensor
#     resize_to : tuple (width, height) to resize the image to

#     Returns:
#     croped_images : list of torch.Tensor
#     crop_coords : list of tuples
#     """
#     random.seed(0)
#     image = image.cpu().numpy()
    
#     if isinstance(image, np.ndarray):
#         image = np.transpose(image, (1, 2, 0))
#         # Convert numpy array to PIL Image
#         image = Image.fromarray(image.astype(np.uint8))

#     cropped_images = []
#     cropped_coords = []
#     for i in range(num_crops):
#         # Randomly select a starting point for the crop
#         #######
#         x1,y1 = random.randint(0,447), random.randint(0,447)
#         crop_size = random.randint(1, 448 - max(x1,y1))
#         #######

#         cropped_image = image.crop((x1, y1, x1 + crop_size, y1 + crop_size))

#         resized_image = cropped_image.resize(resize_to)
#         resized_image = torch.Tensor(np.transpose(resized_image.copy(), (2, 0, 1)))
#         cropped_images.append(resized_image)
#         cropped_coords.append((x1, y1, x1 + crop_size, y1 + crop_size))
        
    
#     cropped_images = torch.stack(cropped_images, dim=0)

#     return cropped_images, cropped_coords

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
        patch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        
        if aggregration == "mean":
            aggregated_entropy = patch_entropy.mean(dim=1).mean(dim=1)
        elif aggregration == "max":
            aggregated_entropy = patch_entropy.max(dim=1)[0].max(dim=1)[0] 
        elif aggregration == "min":
            aggregated_entropy = patch_entropy.min(dim=1)[0].min(dim=1)[0]
        elif aggregration == "median":
            aggregated_entropy = patch_entropy.median(dim=1)[0].median(dim=1)[0]
        idx = torch.argsort(aggregated_entropy, descending=False)[:int(aggregated_entropy.size()[0] * top)]        
        return None,idx
    else:
        # breakpoint()
        if logits.dim() != 4:
            batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).sum(1)
        else:
            batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(2).sum(2).sum(1)
        idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
        return logits[idx], idx  
    
def feature_update(feature, aug_coords, selected_idxes):
    # print(f"Orig feature shape: {feature[0].shape}")
    orig_feats = feature[0].unsqueeze(0)
    orig_feats = F.interpolate(orig_feats, size=(448, 448), mode="bilinear", align_corners=False)
    
    count_tensor = torch.ones_like(orig_feats)
    
    for idx in selected_idxes:
        crop_size = aug_coords[idx][2] - aug_coords[idx][0]
        y1, y2 = aug_coords[idx][1], aug_coords[idx][3]
        x1, x2 = aug_coords[idx][0], aug_coords[idx][2]
        
        crop_feats = feature[idx + 1]
        
        crop_feats = F.interpolate(crop_feats.unsqueeze(0), size=(crop_size, crop_size), mode="bilinear")
        # print(f"orig_feats[:, :, {y1}:{y2}, {x1}:{x2}].shape: {orig_feats[:, :, y1:y2, x1:x2].shape}")
        # print(f"crop_feats.shape: {crop_feats.shape}")
        orig_feats[:, :, y1:y2, x1:x2] = (orig_feats[:, :, y1:y2, x1:x2] + crop_feats)
        count_tensor[:, :, y1:y2, x1:x2] += 1
        # breakpoint()
    orig_feats = orig_feats / count_tensor
    orig_feats = F.interpolate(orig_feats, size=(28,28), mode="bilinear", align_corners=False)
        
    return orig_feats    
        
        