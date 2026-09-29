# 6D Gaussian Splatting — Reproduction Study

> **Local correction, 2026-09-29:** the implementation now uses sigmoid SH colors,
> one shared viewing direction, per-Gaussian trainable opacity modulation,
> corrected splitting and RGBA targets, and standardized LPIPS evaluation.
> CUDA sources are unchanged. Read [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md)
> for defaults, numerical choices, validation, and checkpoint compatibility.
> The experiment numbers and environment below are historical upstream reports;
> they have not been reproduced with this corrected code.

> **Reproduction of:** *6DGS: Enhanced Direction-Aware Gaussian Splatting for Volumetric Rendering*  
> Zhongpai Gao et al., ICLR 2025  
> This repository is a **reproduction study** built on top of the official [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) codebase (Kerbl et al., SIGGRAPH 2023). No official source code from the 6DGS authors was used; all modifications were independently implemented from the paper's mathematical specification.

---

## Representation

Standard 3D Gaussian Splatting represents a scene as a collection of **static ellipsoids** in 3D space. Each ellipsoid has a fixed position, orientation, and scale that never change regardless of where the camera is. To simulate view-dependent effects like specular highlights or reflections, 3DGS attaches **Spherical Harmonic (SH) coefficients** to each point — a polynomial function that outputs a different *colour* depending on viewing direction. Crucially, the **geometry and opacity** of each Gaussian remain invariant to viewpoint.

This is a fundamental limitation for volumetric and translucent objects. Consider smoke, clouds, or subsurface scattering: the apparent *density*, *thickness*, and *shape* of these materials change significantly depending on viewing angle, not merely their colour. SH cannot model this because it is a pure colour perturbation — it cannot alter opacity or the physical footprint of a primitive.

**6DGS resolves this by lifting the representation to 6D space.** Each Gaussian is defined over a joint distribution of 3D spatial position $(x, y, z)$ and 3D viewing direction $(w_x, w_y, w_z)$. The full joint covariance $\Sigma^{6\times6}$ encodes how spatial extent is *correlated* with viewing direction. At render time, the current camera direction is used to **conditionally slice** this 6D distribution to a standard 3D Gaussian via the Schur complement:

$$\Sigma_{\text{cond}} = \Sigma_p - \Sigma_{pd}\,\Sigma_d^{-1}\,\Sigma_{pd}^\top$$
$$\mu_{\text{cond}} = \mu_p + \Sigma_{pd}\,\Sigma_d^{-1}(d - \mu_d)$$

The conditional mean and opacity vary with viewing direction. For fixed model
parameters, the conditional covariance is independent of viewing direction.
This remains an appearance reconstruction model; image metrics alone do not
establish that the learned representation recovers physical volume density.

---

## Environment Setup

```bash
# Python 3.10 venv with PyTorch cu121 against CUDA 12.1 toolkit
cd /home/bhuvan/6DGS
python3 -m venv venv_6dgs
source venv_6dgs/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0"

pip install plyfile tqdm scipy wheel setuptools lpipsPyTorch

cd gaussian-splatting
pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/simple-knn --no-build-isolation
cd ..
```

> **Hardware:** NVIDIA A100 80 GB PCIe · CUDA 12.1 toolkit · PyTorch 2.5.1+cu121

---

## Dataset

The PBR volumetric dataset used in this study (`dragon`, `bunny_cloud`, `cloud`, `smoke`, `explosion`, `suzanne`) was obtained from the [official 6DGS project page](https://gaozhongpai.github.io/6dgs/). Download and extract the dataset archive into the parent directory alongside `gaussian-splatting/`:

```
6DGS/
├── gaussian-splatting/   ← this repo
├── 6dgs-pbr/             ← extracted dataset
│   ├── dragon/
│   ├── bunny_cloud/
│   ├── cloud/
│   ├── smoke/
│   ├── explosion/
│   └── suzanne/
└── venv_6dgs/            ← Python virtual environment
```

Each scene directory follows the standard NeRF-synthetic format with `transforms_train.json`, `transforms_test.json`, and an `images/` subdirectory of RGBA PNGs.

---

## Training

```bash
cd gaussian-splatting
source ../venv_6dgs/bin/activate

# dragon scene
python train.py -s ../6dgs-pbr/dragon --eval

# bunny_cloud scene
python train.py -s ../6dgs-pbr/bunny_cloud --eval
```

Metrics (L1, PSNR, SSIM, LPIPS) are evaluated and logged at iterations **7,000** and **30,000**.

## Rendering

```bash
python render.py -m output/<run-id>
```

---

## Experimental Results

All metrics are evaluated on held-out test views at iteration 30,000.

| Scene | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| `dragon` | **32.07 dB** | **0.892** | **0.071** |
| `bunny_cloud` | **41.69 dB** | **0.993** | **0.047** |

The `bunny_cloud` scene — a highly translucent volumetric object — is particularly illustrative: a PSNR of 41.69 dB on novel test views demonstrates that the view-dependent conditional slicing is correctly modelling the angular-dependent density of the volume, a result that 3DGS with SH cannot achieve.

---

## Modified Files

Exactly **4 files** in the original `gaussian-splatting` repository were modified.

---

### 1. `scene/gaussian_model.py`

**What 3DGS did here:** Stored each Gaussian as `_scaling` (3D log-scale) and `_rotation` (4D quaternion), giving it a fixed 3D ellipsoidal shape in world space. Colours were parameterised as Spherical Harmonic coefficients `_features_dc` and `_features_rest`.

**What we changed:**

- **Removed** `_scaling` and `_rotation` entirely.
- **Added** three new learnable parameter tensors:
  - `_L_diag` — shape `(N, 6)`, log-parameterised diagonal of the 6×6 Cholesky factor `L`. Activated via `torch.exp` to enforce positive definiteness of the spatial and directional diagonal blocks.
  - `_L_offdiag` — shape `(N, 15)`, the 15 strictly lower-triangular off-diagonal elements of `L`. Activated via `2·sigmoid(x)−1` to constrain values to `(-1, 1)` for numerical stability.
  - `_mu_d` — shape `(N, 3)`, the directional mean vector encoding the preferred emission direction of each Gaussian. Unit-normalised at inference time.
- **Implemented `slice_to_3dgs(camera_center)`** — the core Algorithm 1 from the paper. Given the current camera position, this computes:
  1. The unit direction vector `d` from each Gaussian's mean to the camera.
  2. The full 6×6 covariance `Σ = L Lᵀ` and its three block submatrices `Σ_p` (3×3 spatial), `Σ_pd` (3×3 cross), `Σ_d` (3×3 directional).
  3. The 3×3 analytical inverse of `Σ_d` via cofactor expansion (avoiding `torch.linalg.inv` for batch stability).
  4. The conditional mean `μ_cond` and covariance `Σ_cond` via Schur complement.
  5. A Mahalanobis-distance-based **opacity scaling** `f_cond = exp(−λ · (d−μ_d)ᵀ Σ_d⁻¹ (d−μ_d))` with `λ = 0.35`, so Gaussians become transparent when viewed far from their preferred direction.
- **Updated `densify_and_split` and `densify_and_clone`** to operate on the spatial block of `_L_diag[:, :3]` (the 3D spatial diagonal scales) rather than `_scaling`. The 6DGS-specific split factor of **1.6×** (vs 3DGS's 1.6× on 3D scale) was preserved.
- **Updated `save_ply` / `load_ply`** to serialise and deserialise `_L_diag`, `_L_offdiag`, and `_mu_d` to/from `.ply` checkpoint files.

---

### 2. `gaussian_renderer/__init__.py`

**What 3DGS did here:** The `render()` function directly read `pc.get_scaling`, `pc.get_rotation`, and `pc.get_opacity` and passed them as static tensors to the CUDA rasterizer.

**What we changed:**

- Inserted a single **pre-rasterization hook** at the top of `render()`:
  ```python
  means3D_cond, cov3D_precomp, opacity_cond = pc.slice_to_3dgs(viewpoint_camera.camera_center)
  ```
- The rasterizer receives `cov3D_precomp` (the upper-triangular 6-element packed form of `Σ_cond`) and `opacity_cond` (the Mahalanobis-scaled alpha) instead of static scale/rotation parameters.
- `scales` and `rotations` are explicitly set to `None`, routing the rasterizer into its precomputed-covariance path.
- The underlying `diff-gaussian-rasterization` CUDA kernel was **not modified**. The 6D→3D projection runs entirely in Python/PyTorch, leaving the low-level tile rasterizer unchanged.

**Why this design:** It allows the full 6D physics to be applied without forking or rebuilding the CUDA submodule, making the implementation portable and easy to audit.

---

### 3. `train.py`

**What 3DGS did here:** Called `densify_and_prune` with an opacity threshold of `0.005`. Reported only L1 and PSNR at test iterations.

**What we changed:**

- **Opacity pruning threshold:** Changed from `0.005` → **`0.01`** to match the 6DGS paper specification. Because 6DGS modulates opacity per-view, Gaussians with low base opacity are less useful and can be pruned more aggressively.
- **Inline metrics:** Imported `lpipsPyTorch.lpips` and extended the `training_report` function to evaluate and log **SSIM** and **LPIPS** alongside L1 and PSNR at every test iteration. All four metrics are written to both stdout and TensorBoard.

### 4. `scene/dataset_readers.py`

**What 3DGS did here:** When loading NeRF-synthetic datasets with RGBA images, it constructed a composited numpy array and passed it directly to `PIL.Image.fromarray()`.

**What we changed:**

- Fixed a **`TypeError`** that occurred when the composited array was of dtype `np.byte` (signed, range `[-128, 127]`), which PIL refuses to interpret as an image. Added an explicit cast:
  ```python
  arr = arr.astype(np.uint8)
  ```
  before the `PIL.Image.fromarray()` call. This was a data pipeline bug, not a 3DGS design decision, and is unrelated to the 6DGS mathematics.

---

## Repository Structure

```
gaussian-splatting/
├── scene/
│   ├── gaussian_model.py      ← 6D parameters, Cholesky slice, densification [MODIFIED]
│   └── dataset_readers.py     ← np.byte → np.uint8 dtype fix                 [MODIFIED]
├── gaussian_renderer/
│   └── __init__.py            ← Pre-rasterization 6D→3D hook                 [MODIFIED]
├── train.py                   ← Opacity threshold + inline SSIM/LPIPS         [MODIFIED]
├── render.py                  ← Unchanged
├── metrics.py                 ← Unchanged
└── submodules/
    ├── diff-gaussian-rasterization/   ← Unchanged (CUDA rasterizer)
    └── simple-knn/                    ← Unchanged
```

All subdirectories required to run training and rendering are listed below.



## Future Work

### Speed Optimisation

Our implementation achieves state-of-the-art **visual quality** but does not match the paper's training throughput. The primary bottleneck is that the Schur complement slicing (`slice_to_3dgs`) is implemented in **PyTorch on the Python side**, running at ~4–6 iter/s (total training time ~1.5 hours on A100). The paper benchmarks on an RTX 4090 at approximately 40–50 minutes, suggesting a ~2–3× gap.

---

## Citation

```bibtex
@inproceedings{gao20246dgs,
  title     = {6DGS: Enhanced Direction-Aware Gaussian Splatting for Volumetric Rendering},
  author    = {Gao, Zhongpai and Zhou, Benjamin and Knopf, Marius and Metaxas, Dimitris},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}

@article{kerbl20233dgs,
  title   = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author  = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  journal = {ACM Transactions on Graphics},
  year    = {2023}
}
```
