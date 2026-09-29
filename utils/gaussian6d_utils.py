"""Differentiable 6D conditioning; all operations use stock PyTorch kernels."""
import torch


def conditional_parameters(L):
    """Return conditional covariance, regression, and direction Cholesky factor.

    Relative roundoff guards scale with covariance and dtype. There is no
    absolute determinant offset, so changing scene units does not change them.
    """
    sigma = L @ L.transpose(-1, -2)
    spatial = sigma[:, :3, :3]
    cross = sigma[:, :3, 3:]
    direction = sigma[:, 3:, 3:]
    eye = torch.eye(3, dtype=L.dtype, device=L.device)
    eps = 8 * torch.finfo(L.dtype).eps
    direction_scale = direction.diagonal(dim1=-2, dim2=-1).mean(-1)
    factor = torch.linalg.cholesky(direction + (eps * direction_scale)[:, None, None] * eye)
    regression = torch.cholesky_solve(cross.transpose(-1, -2), factor).transpose(-1, -2)
    covariance = spatial - regression @ cross.transpose(-1, -2)
    covariance = (covariance + covariance.transpose(-1, -2)) * 0.5
    spatial_scale = spatial.diagonal(dim1=-2, dim2=-1).mean(-1)
    covariance = covariance + (eps * spatial_scale)[:, None, None] * eye
    return covariance, regression, factor


def pack_covariance(covariance):
    return covariance[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]].contiguous()


def covariance_scale_rotation(covariance):
    # Refinement runs without gradients; eigh preserves the sign of eigenvalues.
    values, rotation = torch.linalg.eigh(covariance)
    rotation = rotation.clone()
    rotation[:, :, -1] *= torch.linalg.det(rotation).sign().unsqueeze(-1)
    return values.clamp_min(0).sqrt(), rotation
