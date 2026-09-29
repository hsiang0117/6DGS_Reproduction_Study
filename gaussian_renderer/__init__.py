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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Slice 6DGS to 3DGS
    direction = pc.view_direction(viewpoint_camera.camera_center)
    means3D_cond, cov3D_precomp, opacity_cond = pc.slice_to_3dgs(viewpoint_camera.camera_center, direction)
    cov3D_precomp = cov3D_precomp * (scaling_modifier ** 2)
    
    means3D = means3D_cond
    means2D = screenspace_points
    opacity = opacity_cond
    scales = None
    rotations = None

    # Always use one PyTorch SH path: CUDA's built-in SH uses a different
    # activation and recomputes directions from the conditional positions.
    # Legacy convert_SHs_python/separate_sh switches no longer alter semantics.
    if override_color is None:
        sh = pc.get_features.transpose(1, 2)
        colors_precomp = torch.sigmoid(eval_sh(pc.active_sh_degree, sh, direction))
    else:
        colors_precomp = override_color

    rendered_image, radii, depth_image = rasterizer(
        means3D=means3D, means2D=means2D, shs=None,
        colors_precomp=colors_precomp, opacities=opacity,
        scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp)

    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "depth" : depth_image
        }
    
    return out
