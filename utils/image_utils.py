#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from PIL import Image

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def save_periodic_render(model_path, iteration, image_tensor, image_name=None):
    """Save the current training render without performing another render pass."""
    render_dir = os.path.join(model_path, "render_test")
    os.makedirs(render_dir, exist_ok=True)
    safe_name = str(image_name).replace("\\", "_").replace("/", "_") if image_name else ""
    suffix = f"_{safe_name}" if safe_name else ""
    render_path = os.path.join(render_dir, f"iter_{iteration:06d}{suffix}.png")
    img8 = (image_tensor.detach().clamp(0, 1) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
    Image.fromarray(img8).save(render_path)
