# Windows CUDA 扩展构建

本机验证配置：Windows、Python 3.12.10、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、CUDA Toolkit 12.8、MSVC 14.44、RTX 5060 Laptop（sm_120）。
所有扩展必须使用本仓库 `.venv` 编译和运行。脚本不接受其他仓库的 Python。

## 环境准备

从仓库根目录，在普通 PowerShell 控制台中执行：

```powershell
git submodule update --init --recursive
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install 'setuptools<82' wheel ninja
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install numpy scipy pillow plyfile tqdm tensorboard opencv-python
```

另需安装 CUDA Toolkit 12.8，以及 Visual Studio 的 C++ 工具和 MSVC 14.44。
VS 2026 的默认 14.50 不在此 CUDA 版本接受的范围内，应安装并选择 14.44，不能使用 `-allow-unsupported-compiler` 跳过检查。

## 头文件补丁

NVCC/MSVC 编译 `spatial.cu` 时，PyTorch 的 `compiled_autograd.h` 在 `IValuePacker<T>::packed_type()` 的字符串分支报 C2872：`std` 符号不明确。
本机 Cloud-GS、原版 3DGS 和 Vol3DGS 已使用的兼容补丁是注释以下两行：

```cpp
//     } else if constexpr (::std::is_same_v<T, ::std::string>) {
//       return at::StringType::get();
```

`tools/patch_torch_header.py` 只修改当前仓库 venv 的头文件；检查 PyTorch 版本和完整文件 SHA256，拒绝未知版本，保留原始换行符，并自动备份到 `environment-backups/<头文件名>.original`。重复应用无额外修改。

这是有范围限制的本地兼容补丁，会移除此模板的字符串类型映射；不是通用等价修复。不修改高斯渲染、KNN 或 SSIM 算法。若以后需要该模板处理字符串，或升级/重装 PyTorch，应重新评估。

```powershell
# 单独应用；构建脚本也会自动调用
.\.venv\Scripts\python.exe -I tools/patch_torch_header.py
# 恢复原头文件；已安装二进制不会随之重编译
.\.venv\Scripts\python.exe -I tools/patch_torch_header.py --restore
```

此外，`fused-ssim` 编译时，Windows SDK 的 `rpcndr.h` 定义了 `#define small char`，将 `CUDACachingAllocator.h` 的构造函数形参 `bool small` 展开为非法的 `bool char`。
脚本同时将该形参及初始化列表中的引用重命名为 `is_small_flag`，与另外三个本机工程一致。这个重命名不改变函数签名、内存布局或计算行为。
两个头文件均先校验 SHA256，再备份和修改；`--restore` 恢复两者。

## 构建与验证

```powershell
.\tools\build_cuda_extensions.ps1
# 若无法自动发现 Visual Studio，可显式指定：
.\tools\build_cuda_extensions.ps1 -VcVars 'D:\Program Files\Microsoft Visual Studio\2026\Community\VC\Auxiliary\Build\vcvarsall.bat'
```

脚本使用本仓库 Python、MSVC 14.44 和 `TORCH_CUDA_ARCH_LIST=12.0`；用 `--no-build-isolation --force-reinstall --no-deps` 构建安装三个本仓库子模块：
`diff-gaussian-rasterization`、`simple-knn`、`fused-ssim`。
普通控制台设置代码页 936，避免中文 MSVC 输出与 PyTorch OEM 解码不一致；不修改 Python/PyTorch 解码函数。
构建来源记录和各扩展日志位于 `temporary-build/build-environment.json` 及 `*-own-venv.log`。

构建结束自动检查实际 GPU 运算：KNN 对照暴力最近邻计算、6DGS 渲染、fused SSIM 反向传播、参数梯度和一次优化器更新，并执行 `pip check`。
也可独立验证：

```powershell
.\.venv\Scripts\python.exe -I tools/verify_cuda_extensions.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_reproduction.py -v
```

## 此前构建记录

此前的构建脚本允许传入 `-BuildPython`，日志显示曾使用原版 3DGS 的 venv（相同 PyTorch 版本）构建本仓库源码的 wheel，再安装到本仓库 venv。因此，6DGS 自身头文件未打补丁时也能运行已构建的扩展，但无法独立复现构建。
本次将补丁和构建步骤纳入版本管理，并改为只使用自身 venv；虚拟环境、备份、构建日志和二进制不提交到 Git。

## 本机验证记录（2026-09-29）

三个扩展均已使用本仓库 `.venv` 编译安装成功。128 点 KNN 与暴力参考一致；64×64 的 6DGS 渲染、fused SSIM 反向传播、六组参数的有限非零梯度以及一次优化器更新均通过，`pip check` 无错误。
此次验证未进行完整重训，也不改变既有 Zenith 实验结果。
