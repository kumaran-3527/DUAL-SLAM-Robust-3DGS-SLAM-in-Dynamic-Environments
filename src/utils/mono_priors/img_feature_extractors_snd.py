from typing import Dict, List, Tuple, Union
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm

def get_feature_extractor(cfg: Dict) -> nn.Module:
    """
    Get the SnD DINOv2 feature extractor model.
    """
    device = cfg["device"]
    
    # Load into timm model
    model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=False,
        num_classes=0,
        dynamic_img_size=True,
        dynamic_img_pad=False,
    )
    
    # Load weights
    weights_path = "pretrained/dinov2_small_snd_no_blending.pth"
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location='cpu')
    else:
        url = "https://huggingface.co/david-shavin/SnD/resolve/main/dinov2_small_snd.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        
    model.load_state_dict(state_dict, strict=False)
    
    return model.to(device).eval()

@torch.no_grad()
def predict_img_features(
    model: nn.Module,
    idx: int,
    input_tensor: torch.Tensor,
    cfg: Dict,
    device: str,
    save_feat: bool = True,
    suffix: str = "",
) -> torch.Tensor:
    """
    Predict image features using the SnD DINOv2 model.

    Args:
        model (nn.Module): The feature extractor model.
        idx (int): Image index.
        input_tensor (torch.Tensor): Input image tensor of shape (1, 3, H, W).
        cfg (Dict): Configuration dictionary.
        device (str): Device to run the model on.
        save_feat (bool): Whether to save the features.
        suffix (str): Suffix for the output file name.

    Returns:
        torch.Tensor: Extracted features [H, W, C].
    """
    stride = 14
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=mean, std=std)
    image_resized = process_image(input_tensor, stride, normalize, device)

    # Extract features using timm's forward_features
    # This outputs a tensor of shape [B, N, C]
    features = model.forward_features(image_resized)
    
    h_feat = image_resized.shape[2] // stride
    w_feat = image_resized.shape[3] // stride
    
    # Depending on timm model configuration, the CLS token might be present.
    # If the sequence length is exactly h_feat * w_feat + 1, it has a CLS token at index 0.
    if features.shape[1] == h_feat * w_feat + 1:
        features = features[:, 1:, :] # Remove CLS token
        
    # Reshape from [B, H*W, C] -> [H, W, C]
    features = features.view(h_feat, w_feat, -1)

    if save_feat:
        _save_features(features, cfg, idx, suffix)

    return features

def process_image(
    image: torch.Tensor, stride: int, transforms: nn.Module, device: str = "cuda"
) -> torch.Tensor:
    """
    Process the input image for feature extraction.
    """
    image_tensor = transforms(image).float().to(device)
    h, w = image_tensor.shape[2:]
    height_int = (h // stride) * stride
    width_int = (w // stride) * stride
    return F.interpolate(image_tensor, size=(height_int, width_int), mode="bilinear")

def _save_features(features: torch.Tensor, cfg: Dict, idx: int, suffix: str) -> None:
    """
    Save the extracted features to a file.
    """
    output_dir = f"{cfg['data']['output']}/{cfg['scene']}"
    output_path = f"{output_dir}/mono_priors/features/{idx:05d}{suffix}.npy"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_feat = features.detach().cpu().float().numpy()
    np.save(output_path, final_feat)
