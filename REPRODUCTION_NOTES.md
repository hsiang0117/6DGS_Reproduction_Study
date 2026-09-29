# 本地修正版说明

基于 Bhuvan171 的提交 `e8885cd566285a9089cbdd44d74f0dc3cbe012d9`。
本次只修改 Python 层，`submodules` 中的 CUDA/C++ 源码保持不变，无需重新编译。
此代码仍是独立复现，尚未验证论文完整数据集指标。

## 默认行为

| 项目 | 当前实现 |
|---|---|
| SH 颜色 | PyTorch 中计算 `sigmoid(SH(d))`，DC 初始化使用 `logit(RGB)/C0` |
| 方向 | 切片及 SH 共用从相机指向原始位置的单位方向 |
| λ | 每高斯一个 logit，sigmoid 映射到 (0,1)，初始 0.35 |
| λ 学习时段 | `[15000,28000)`；区间外从计算图分离，避免 Adam 动量继续更新 |
| λ 学习率 | `0.001`，论文未给出，属于本复现选择 |
| 分裂 | 完整空间 Cholesky 块除以 `0.8*N`，使条件协方差等比缩小 |
| 剪枝 | 保留既有点的屏幕半径累计量，直到本次剪枝完成后清零；子点从零累计 |
| GT | straight-alpha RGBA 按黑/白背景合成一次，RGB 保持原值 |
| LPIPS | 便利函数输入 [0,1]，内部转 [-1,1]，按设备缓存评估网络 |
| 优化器 | 标准 Adam；不支持面向原版参数布局的 sparse Adam |

`--convert_SHs_python`、`--compute_cov3D_python` 和渲染接口中的 `separate_sh`
保留用于兼容原入口，但不再改变颜色/协方差的计算方式。
`scaling_modifier` 调整条件高斯的协方差 footprint，不修改条件均值。

需要做固定 λ=0.35 消融时，可设 `--lambda_opa_from_iter 0 --lambda_opa_until_iter 0`。
这不是默认的可学习版本。三个 λ 训练选项均会写入训练配置。

## 数值选择

使用 Cholesky 分解和线性求解，代替 `adj(D)/(det(D)+1e-15)`。
方向矩阵和条件协方差均采用随尺度变化的舍入保护：
`8 * finfo(dtype).eps * mean(diagonal(reference_covariance)) * I`。
这是明确的数值正则，不是论文新增的物理项。没有固定绝对行列式偏置。

方向均值归一化、方向尺度初始 0.3、初始交叉块为零沿用第三方实现；论文未完整
交代这些选择，暂不把它们改为未经验证的新假设。场景初始化仍需检查已有
`points3d.ply` 或默认 [-1.3,1.3] 范围是否合适。

## 数据与评估

透明度用于合成 GT，不再只给预测图乘一次 alpha。半图曝光训练的显式 holdout
mask 同时作用于预测和 GT，独立于源图 alpha。默认关闭该曝光模式。
这里假设 PNG 为 straight-alpha；预乘 alpha 资产应先明确转换。

训练 loss 仍为 L1 + DSSIM，没有增加 LPIPS loss。LPIPS 仅用于指定迭代的评估。
正式比较继续使用一致的测试划分、背景和外部统一评估脚本，并传 `--eval`。
README 中原作者报告的数字不代表本修正版的结果。

## 保存及兼容性

PLY 新增 `lambda_opa_logit` 属性，并写入 `6dgs_color_activation sigmoid` 注释。
旧版 SH 系数不能直接当作 sigmoid 的系数使用，因此缺少该标记的旧 PLY 会明确
拒绝加载，而不是静默改变外观。旧 tuple checkpoint 同样要求重新训练。

新 checkpoint 保存模型、λ、优化器、曝光和增密统计。学习率以 Python float
保存，支持 PyTorch `weights_only=True` 加载。恢复时使用同一训练配置；测试
覆盖相同梯度下恢复后的下一次 Adam 更新，不承诺恢复相机抽样/RNG 的逐位连续性。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_reproduction.py -v
```

覆盖条件公式及自动微分、小行列式、颜色/方向路径、分裂、剪枝、λ 冻结与状态
增删、checkpoint/PLY 往返、RGBA、LPIPS 输入与缓存。GPU 测试依赖本仓库扩展。
本地 `.local-setup` 保存构建日志、数值诊断和小型合成训练检查，不纳入正式实验结果。

2026-09-29 本机验证结果：9 项回归测试全部通过；64×64、4 个训练视角和
2 个测试视角的合成数据完成 240 步训练，覆盖增密、opacity 重置、λ 学习与冻结、
评估及保存。另从第 120 步 checkpoint 通过实际命令行恢复到第 160 步，
并完成 `render.py` 和 `metrics.py` 离线流程。短跑将 λ 学习窗口临时设为
`[80,160)`，不改变正式训练的默认窗口。尚未进行真实场景完整重训。

论文来源：https://arxiv.org/html/2410.04974v3
