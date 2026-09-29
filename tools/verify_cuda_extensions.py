"""Environment smoke check: real CUDA kernels and a synthetic 6DGS optimization step."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from argparse import ArgumentParser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torchvision
assert Path(sys.prefix).resolve() == (ROOT / '.venv').resolve(), 'Use this repository venv'
assert Path(torch.__file__).resolve().is_relative_to((ROOT / '.venv').resolve())
from simple_knn._C import distCUDA2
from fused_ssim import fused_ssim
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from utils.graphics_utils import BasicPointCloud, getProjectionMatrix
from arguments import OptimizationParams

torch.manual_seed(123)
torch.cuda.init()
points = torch.randn(128, 3, device="cuda") * 0.2
points[:, 2] += 3.0
d2 = distCUDA2(points)
assert torch.isfinite(d2).all() and (d2 > 0).all()
expected = torch.cdist(points, points).square()
expected.fill_diagonal_(float("inf"))
expected = expected.topk(3, largest=False).values.mean(dim=1)
torch.testing.assert_close(d2, expected, atol=2e-5, rtol=2e-3)

model = GaussianModel(3)
pcd = BasicPointCloud(points.cpu().numpy(), np.full((128, 3), 0.6), np.zeros((128, 3)))
model.create_from_pcd(pcd, [SimpleNamespace(image_name="smoke")], 1.0)
parser = ArgumentParser()
opt = OptimizationParams(parser).extract(parser.parse_args([]))
model.training_setup(opt)
view = torch.eye(4, device="cuda")
proj = getProjectionMatrix(0.01, 100.0, 1.0, 1.0).transpose(0, 1).cuda()
camera = MiniCam(64, 64, 1.0, 1.0, 0.01, 100.0, view, proj)
pipe = SimpleNamespace(debug=False, antialiasing=False, convert_SHs_python=False, compute_cov3D_python=True)
package = render(camera, model, pipe, torch.zeros(3, device="cuda"))
img = package["render"]
assert img.shape == (3, 64, 64) and torch.isfinite(img).all() and img.max() > 0
target = torch.full_like(img, 0.1)
ssim = fused_ssim(img[None], target[None])
loss = (img - target).abs().mean() + 0.2 * (1 - ssim)
loss.backward()
grads = {}
for name in ("_xyz", "_L_diag", "_L_offdiag", "_mu_d", "_opacity", "_features_dc"):
    grad = getattr(model, name).grad
    assert grad is not None and torch.isfinite(grad).all(), name
    grads[name] = float(grad.norm())
    assert grads[name] > 0, name
model.optimizer.step()
torch.cuda.synchronize()
result = {"torch": torch.__version__, "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
          "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
          "knn_matches_brute_force": True, "render_shape": list(img.shape),
          "loss": float(loss.detach()), "parameter_gradient_norms": grads,
          "optimizer_step": "passed"}
(ROOT / "temporary-build").mkdir(exist_ok=True)
(ROOT / "temporary-build" / "smoke-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
