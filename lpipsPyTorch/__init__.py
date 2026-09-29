import torch
from functools import lru_cache

from .modules.lpips import LPIPS


@lru_cache(maxsize=4)
def _metric(net_type, version, device):
    return LPIPS(net_type, version).to(device).eval().requires_grad_(False)


def lpips(x: torch.Tensor,
          y: torch.Tensor,
          net_type: str = 'alex',
          version: str = '0.1'):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    # This convenience API accepts renderer RGB in [0,1]. The LPIPS module
    # itself follows the official [-1,1] convention.
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if y.ndim == 3:
        y = y.unsqueeze(0)
    criterion = _metric(net_type, version, str(x.device))
    return criterion((2 * x - 1).contiguous(), (2 * y - 1).contiguous())
