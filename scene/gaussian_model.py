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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import C0
from utils.gaussian6d_utils import conditional_parameters, pack_covariance, covariance_scale_rotation
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        
        self._L_diag = torch.empty(0)
        self._L_offdiag = torch.empty(0)
        self._mu_d = torch.empty(0)
        self._lambda_opa = torch.empty(0)
        self._lambda_trainable = False
        
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return {
            "format": "6dgs_sigmoid_sh",
            "active_sh_degree": self.active_sh_degree,
            "parameters": {name: getattr(self, name) for name in (
                "_xyz", "_features_dc", "_features_rest", "_L_diag", "_L_offdiag",
                "_mu_d", "_opacity", "_lambda_opa", "_exposure")},
            "max_radii2D": self.max_radii2D,
            "xyz_gradient_accum": self.xyz_gradient_accum,
            "denom": self.denom,
            "optimizer": self.optimizer.state_dict(),
            "exposure_optimizer": self.exposure_optimizer.state_dict(),
            "exposure_mapping": self.exposure_mapping,
            "pretrained_exposures": self.pretrained_exposures,
            "spatial_lr_scale": self.spatial_lr_scale,
        }

    def restore(self, model_args, training_args):
        if not isinstance(model_args, dict) or model_args.get("format") != "6dgs_sigmoid_sh":
            raise ValueError("Legacy checkpoints use a different SH activation; retrain with the corrected implementation.")
        for name, value in model_args["parameters"].items():
            setattr(self, name, nn.Parameter(value.detach().requires_grad_(True)))
        self.active_sh_degree = model_args["active_sh_degree"]
        self.spatial_lr_scale = model_args["spatial_lr_scale"]
        self.exposure_mapping = model_args["exposure_mapping"]
        self.pretrained_exposures = model_args["pretrained_exposures"]
        self.training_setup(training_args)
        for name in ("max_radii2D", "xyz_gradient_accum", "denom"):
            setattr(self, name, model_args[name])
        self.optimizer.load_state_dict(model_args["optimizer"])
        self.exposure_optimizer.load_state_dict(model_args["exposure_optimizer"])

    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_lambda_opa(self):
        logits = self._lambda_opa if self._lambda_trainable else self._lambda_opa.detach()
        return logits.sigmoid()

    def view_direction(self, camera_center):
        return torch.nn.functional.normalize(self.get_xyz - camera_center, dim=-1)

    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
            
    @property
    def get_S_and_R(self):
        covariance, _, _ = conditional_parameters(self.get_L)
        return covariance_scale_rotation(covariance)

    @property
    def get_L(self):
        diag = torch.exp(self._L_diag)
        offdiag = 2.0 * torch.sigmoid(self._L_offdiag) - 1.0
        
        L = torch.diag_embed(diag)
        indices = torch.tril_indices(6, 6, offset=-1, device=diag.device)
        L[:, indices[0], indices[1]] = offdiag
        return L

    def slice_to_3dgs(self, camera_center, direction=None):
        d = self.view_direction(camera_center) if direction is None else direction
        covariance, regression, factor = conditional_parameters(self.get_L)
        mu_d = torch.nn.functional.normalize(self._mu_d, dim=-1)
        x = d - mu_d
        mu_cond = self.get_xyz + (regression @ x.unsqueeze(-1)).squeeze(-1)
        whitened = torch.linalg.solve_triangular(factor, x.unsqueeze(-1), upper=False)
        distance = whitened.square().sum(dim=(-2, -1)).unsqueeze(-1)
        alpha_cond = self.get_opacity * torch.exp(-self.get_lambda_opa * distance)
        return mu_cond, pack_covariance(covariance), alpha_cond

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = float(spatial_lr_scale)
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        color = torch.tensor(np.asarray(pcd.colors)).float().cuda().clamp(1e-6, 1 - 1e-6)
        fused_color = torch.logit(color) / C0
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        dir_scales = torch.ones_like(scales) * math.log(0.3)
        L_diag = torch.cat((scales, dir_scales), dim=1)
        
        L_offdiag = torch.zeros((fused_point_cloud.shape[0], 15), device="cuda")
        
        v = torch.randn((fused_point_cloud.shape[0], 3), device="cuda")
        mu_d = torch.nn.functional.normalize(v, dim=-1)

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._L_diag = nn.Parameter(L_diag.requires_grad_(True))
        self._L_offdiag = nn.Parameter(L_offdiag.requires_grad_(True))
        self._mu_d = nn.Parameter(mu_d.requires_grad_(True))
        self._lambda_opa = nn.Parameter(torch.full_like(opacities, math.log(0.35 / 0.65)))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        if self.optimizer_type != "default":
            raise ValueError("6D parameters require the default Adam optimizer; sparse_adam is not supported.")
        self.lambda_opa_lr = training_args.lambda_opa_lr
        self.lambda_opa_from_iter = training_args.lambda_opa_from_iter
        self.lambda_opa_until_iter = training_args.lambda_opa_until_iter
        self._lambda_trainable = False
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._L_diag], 'lr': 1e-2, "name": "L_diag"},
            {'params': [self._L_offdiag], 'lr': 1e-2, "name": "L_offdiag"},
            {'params': [self._mu_d], 'lr': 1e-3, "name": "mu_d"},
            {'params': [self._lambda_opa], 'lr': 0.0, "name": "lambda_opa"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        self._lambda_trainable = self.lambda_opa_from_iter <= iteration < self.lambda_opa_until_iter
        if not self._lambda_trainable:
            self._lambda_opa.grad = None
        for group in self.optimizer.param_groups:
            if group["name"] == "lambda_opa":
                group['lr'] = self.lambda_opa_lr if self._lambda_trainable else 0.0
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = float(self.exposure_scheduler_args(iteration))

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = float(self.xyz_scheduler_args(iteration))
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._L_diag.shape[1]):
            l.append('l_diag_{}'.format(i))
        for i in range(self._L_offdiag.shape[1]):
            l.append('l_offdiag_{}'.format(i))
        for i in range(self._mu_d.shape[1]):
            l.append('mu_d_{}'.format(i))
        l.append('lambda_opa_logit')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        L_diag = self._L_diag.detach().cpu().numpy()
        L_offdiag = self._L_offdiag.detach().cpu().numpy()
        mu_d = self._mu_d.detach().cpu().numpy()
        lambda_opa = self._lambda_opa.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, L_diag, L_offdiag, mu_d, lambda_opa), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el], comments=["6dgs_color_activation sigmoid"]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if "6dgs_color_activation sigmoid" not in plydata.comments:
            raise ValueError("This PLY lacks sigmoid-SH metadata. Legacy 6DGS models require retraining.")
        self.pretrained_exposures = None
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        L_diag_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("l_diag_")]
        L_diag_names = sorted(L_diag_names, key = lambda x: int(x.split('_')[-1]))
        L_diags = np.zeros((xyz.shape[0], len(L_diag_names)))
        for idx, attr_name in enumerate(L_diag_names):
            L_diags[:, idx] = np.asarray(plydata.elements[0][attr_name])

        L_offdiag_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("l_offdiag_")]
        L_offdiag_names = sorted(L_offdiag_names, key = lambda x: int(x.split('_')[-1]))
        L_offdiags = np.zeros((xyz.shape[0], len(L_offdiag_names)))
        for idx, attr_name in enumerate(L_offdiag_names):
            L_offdiags[:, idx] = np.asarray(plydata.elements[0][attr_name])

        mu_d_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("mu_d_")]
        mu_d_names = sorted(mu_d_names, key = lambda x: int(x.split('_')[-1]))
        mu_ds = np.zeros((xyz.shape[0], len(mu_d_names)))
        for idx, attr_name in enumerate(mu_d_names):
            mu_ds[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._L_diag = nn.Parameter(torch.tensor(L_diags, dtype=torch.float, device="cuda").requires_grad_(True))
        self._L_offdiag = nn.Parameter(torch.tensor(L_offdiags, dtype=torch.float, device="cuda").requires_grad_(True))
        self._mu_d = nn.Parameter(torch.tensor(mu_ds, dtype=torch.float, device="cuda").requires_grad_(True))
        lambda_logits = np.asarray(plydata.elements[0]["lambda_opa_logit"]).copy()
        self._lambda_opa = nn.Parameter(torch.tensor(lambda_logits[:, None], dtype=torch.float, device="cuda"))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                if stored_state is not None:
                    self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._L_diag = optimizable_tensors["L_diag"]
        self._L_offdiag = optimizable_tensors["L_offdiag"]
        self._mu_d = optimizable_tensors["mu_d"]
        self._lambda_opa = optimizable_tensors["lambda_opa"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_L_diag, new_L_offdiag, new_mu_d, new_lambda_opa, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "L_diag" : new_L_diag,
        "L_offdiag" : new_L_offdiag,
        "mu_d" : new_mu_d,
        "lambda_opa": new_lambda_opa}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._L_diag = optimizable_tensors["L_diag"]
        self._L_offdiag = optimizable_tensors["L_offdiag"]
        self._mu_d = optimizable_tensors["mu_d"]
        self._lambda_opa = optimizable_tensors["lambda_opa"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # Preserve old statistics until screen-space pruning. New children start fresh.
        self.max_radii2D = torch.cat((self.max_radii2D, torch.zeros_like(new_tmp_radii)))

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        
        S, R = self.get_S_and_R
        
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(S, dim=1).values > self.percent_dense*scene_extent)

        stds = S[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = R[selected_pts_mask].repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        
        new_L_diag = self._L_diag[selected_pts_mask].repeat(N,1)
        shrink = 0.8 * N
        new_L_diag[:, :3] -= math.log(shrink)
        
        new_L_offdiag = self._L_offdiag[selected_pts_mask].repeat(N,1)
        # The first three packed off-diagonals belong to L's spatial 3x3 block.
        spatial_offdiag = (2 * new_L_offdiag[:, :3].sigmoid() - 1) / shrink
        new_L_offdiag[:, :3] = torch.logit((spatial_offdiag + 1) * 0.5)
        new_mu_d = self._mu_d[selected_pts_mask].repeat(N,1)
        new_lambda_opa = self._lambda_opa[selected_pts_mask].repeat(N,1)
        
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_L_diag, new_L_offdiag, new_mu_d, new_lambda_opa, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        S, _ = self.get_S_and_R
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(S, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_L_diag = self._L_diag[selected_pts_mask]
        new_L_offdiag = self._L_offdiag[selected_pts_mask]
        new_mu_d = self._mu_d[selected_pts_mask]
        new_lambda_opa = self._lambda_opa[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_L_diag, new_L_offdiag, new_mu_d, new_lambda_opa, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze(-1)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            S, _ = self.get_S_and_R
            big_points_ws = S.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        self.tmp_radii = None
        self.max_radii2D.zero_()

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
