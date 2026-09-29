"""Numerical and state-transition regressions for the corrected reproduction."""
import io
import math
import sys
import tempfile
import unittest
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
from arguments import OptimizationParams
from gaussian_renderer import render
from scene.cameras import Camera, MiniCam
from scene.gaussian_model import GaussianModel
from utils.gaussian6d_utils import conditional_parameters
from utils.graphics_utils import BasicPointCloud, getProjectionMatrix
from utils.sh_utils import C0, eval_sh


def options():
    parser = ArgumentParser()
    group = OptimizationParams(parser)
    return group.extract(parser.parse_args([]))


def model(n=16):
    torch.manual_seed(123)
    points = torch.randn(n, 3, device='cuda') * .2
    points[:, 2] += 3
    m = GaussianModel(3)
    m.create_from_pcd(BasicPointCloud(points.cpu().numpy(), np.full((n, 3), .6), np.zeros((n, 3))),
                      [SimpleNamespace(image_name='test')], 1.)
    m.training_setup(options())
    with torch.no_grad():
        m._L_diag[:] = torch.log(torch.tensor([.2,.3,.15,1.,1.,1.], device='cuda'))
        m._L_offdiag[:] = torch.randn_like(m._L_offdiag)*.8
        m._opacity[:] = 0
        m._mu_d[:] = torch.tensor([1.,0.,0.], device='cuda')
    return m


def render_args():
    view = torch.eye(4, device='cuda')
    proj = getProjectionMatrix(.01,100.,1.,1.).T.cuda()
    return (MiniCam(64,64,1.,1.,.01,100.,view,proj),
            SimpleNamespace(debug=False, antialiasing=False, convert_SHs_python=False, compute_cov3D_python=False),
            torch.zeros(3, device='cuda'))


class ConditioningTests(unittest.TestCase):
    def test_value_and_derivatives(self):
        torch.manual_seed(1)
        L = torch.tril(torch.randn(2,6,6,dtype=torch.double)*.1) + torch.eye(6,dtype=torch.double)
        L.requires_grad_()
        cov, regression, _ = conditional_parameters(L)
        sigma = L @ L.transpose(-1,-2)
        ref_reg = torch.linalg.solve(sigma[:,3:,3:], sigma[:,:3,3:].transpose(-1,-2)).transpose(-1,-2)
        ref_cov = sigma[:,:3,:3] - ref_reg @ sigma[:,3:,:3]
        torch.testing.assert_close(cov, ref_cov, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(regression, ref_reg, atol=1e-12, rtol=1e-12)
        self.assertTrue(torch.autograd.gradcheck(conditional_parameters, (L,), fast_mode=True))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA extensions required')
class ReproductionTests(unittest.TestCase):
    def test_small_direction_covariance(self):
        m = model(1)
        with torch.no_grad():
            m._L_offdiag.zero_()
            m._L_diag[:,3:] = math.log(1e-3)
            m._xyz[:] = torch.tensor([0.,0.,3.],device='cuda')
            m._mu_d[:] = torch.tensor([.001,0.,math.sqrt(1-1e-6)],device='cuda')
        alpha = m.slice_to_3dgs(torch.zeros(3,device='cuda'))[2]
        self.assertAlmostEqual(float(alpha.detach()), .35234399, places=5)

    def test_initial_color_and_all_render_switches(self):
        m = model()
        torch.testing.assert_close(torch.sigmoid(m._features_dc[:,0,:]*C0), torch.full((16,3),.6,device='cuda'))
        m.active_sh_degree = 3
        with torch.no_grad(): m._features_rest.normal_(0,.2)
        cam, pipe, bg = render_args()
        color = torch.sigmoid(eval_sh(3,m.get_features.transpose(1,2),m.view_direction(cam.camera_center)))
        reference = render(cam,m,pipe,bg,override_color=color)['render']
        for flag in (False,True):
            pipe.convert_SHs_python=flag
            for separate in (False,True):
                torch.testing.assert_close(render(cam,m,pipe,bg,separate_sh=separate)['render'],reference)
        loss = reference.square().mean()
        loss.backward()
        for name in ('_xyz','_L_diag','_L_offdiag','_mu_d','_features_dc','_features_rest'):
            grad=getattr(m,name).grad
            self.assertTrue(grad is not None and torch.isfinite(grad).all() and grad.norm()>0,name)

    def test_split_covariance_and_direction_distribution(self):
        m=model()
        with torch.no_grad():
            parent_cov=conditional_parameters(m.get_L)[0]
            parent_sigma=m.get_L @ m.get_L.transpose(-1,-2)
            m.tmp_radii=torch.ones(16,device='cuda')
            m.densify_and_split(torch.ones(16,1,device='cuda'),.1,1.)
            child_cov=conditional_parameters(m.get_L)[0]
            child_sigma=m.get_L @ m.get_L.transpose(-1,-2)
            torch.testing.assert_close(child_cov,parent_cov.repeat(2,1,1)/1.6**2,atol=2e-7,rtol=2e-5)
            torch.testing.assert_close(child_sigma[:,3:,3:],parent_sigma[:,3:,3:].repeat(2,1,1))
            self.assertEqual(m._lambda_opa.shape,(32,1))
            torch.testing.assert_close(m.get_lambda_opa,torch.full((32,1),.35,device='cuda'))

    def test_screen_radius_pruning(self):
        m=model()
        with torch.no_grad():
            m.max_radii2D[:]=100
            m.denom[:]=1
            m.densify_and_prune(1.,.01,100.,20,torch.ones(16,device='cuda'))
            self.assertEqual(len(m.get_xyz),0)

    def test_lambda_schedule_and_optimizer_state_resize(self):
        m=model()
        original=m._lambda_opa.detach().clone()
        for step in (14999,15000,27999,28000,28001):
            m.update_learning_rate(step)
            before=m._lambda_opa.detach().clone()
            m.slice_to_3dgs(torch.zeros(3,device='cuda'))[2].sum().backward()
            active=15000<=step<28000
            self.assertEqual(m._lambda_opa.grad is not None,active)
            m.optimizer.step();m.optimizer.zero_grad(set_to_none=True)
            self.assertEqual(torch.equal(before,m._lambda_opa),not active)
        self.assertFalse(torch.equal(original,m._lambda_opa))
        with torch.no_grad():
            m.tmp_radii=torch.ones(16,device='cuda')
            m.percent_dense=10
            m.densify_and_clone(torch.ones(16,1,device='cuda'),.1,1.)
            m.prune_points(torch.arange(32,device='cuda')%2==0)
        state=m.optimizer.state[m._lambda_opa]
        self.assertEqual(state['exp_avg'].shape,m._lambda_opa.shape)
        m.update_learning_rate(28002)
        before=m._lambda_opa.detach().clone()
        m.slice_to_3dgs(torch.zeros(3,device='cuda'))[2].sum().backward()
        m.optimizer.step()
        torch.testing.assert_close(m._lambda_opa,before,rtol=0,atol=0)

    def test_checkpoint_and_ply_roundtrip(self):
        m=model();m.active_sh_degree=3;m.update_learning_rate(16000)
        m.slice_to_3dgs(torch.zeros(3,device='cuda'))[2].sum().backward()
        m.optimizer.step();m.optimizer.zero_grad(set_to_none=True)
        buf=io.BytesIO();torch.save(m.capture(),buf);buf.seek(0)
        restored=GaussianModel(3)
        restored.restore(torch.load(buf,weights_only=True),options())
        restored.update_learning_rate(16001);m.update_learning_rate(16001)
        for obj in (m,restored):
            obj.slice_to_3dgs(torch.zeros(3,device='cuda'))[2].sum().backward()
            obj.optimizer.step();obj.optimizer.zero_grad(set_to_none=True)
        for name in m.capture()['parameters']:
            torch.testing.assert_close(getattr(m,name),getattr(restored,name),rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as temp:
            path=str(Path(temp)/'point_cloud.ply');m.save_ply(path)
            loaded=GaussianModel(3);loaded.load_ply(path)
            cam,pipe,bg=render_args()
            with torch.no_grad(): torch.testing.assert_close(render(cam,m,pipe,bg)['render'],render(cam,loaded,pipe,bg)['render'])
            ply=PlyData.read(path,mmap=False);ply.comments=[];ply.write(path)
            with self.assertRaisesRegex(ValueError,'metadata'): loaded.load_ply(path)

    def test_rgba_and_rgb_targets(self):
        rgba=np.zeros((8,8,4),dtype=np.uint8);rgba[:]=[204,102,51,64]
        for white in (False,True):
            cam=Camera((8,8),0,np.eye(3),np.zeros(3),1.,1.,None,Image.fromarray(rgba),None,'rgba',0,white_background=white)
            expected=torch.tensor([.8,.4,.2],device='cuda')*(64/255)+float(white)*(1-64/255)
            torch.testing.assert_close(cam.original_image[:,0,0],expected)
            self.assertTrue((cam.alpha_mask==1).all())

    def test_lpips_mapping_and_cache(self):
        import lpipsPyTorch as metric
        torch.manual_seed(19)
        x=torch.rand(1,3,64,64,device='cuda');y=(x*.8+.05).clamp(0,1)
        with torch.no_grad():
            value=metric.lpips(x,y,net_type='vgg')
            cached=metric._metric('vgg','0.1','cuda:0')
            expected=cached(2*x-1,2*y-1)
            torch.testing.assert_close(value,expected)
            self.assertIs(cached,metric._metric('vgg','0.1','cuda:0'))


if __name__=='__main__': unittest.main(verbosity=2)
