# FaceSR_CPP 当前代码实现状态

本文档是 README、快速开始和答辩材料的事实基准。若其他历史文档中的描述与本文档冲突，以当前源码和本文档为准。

## 已实现

- C++17 + CMake 构建，核心依赖为 LibTorch、OpenCV、Qt Widgets。
- 生成器为 RRDBNet，默认 23 个 RRDB 块，支持可选 CBAM attention。
- 生成器上采样方式为“最近邻插值 + 卷积”，不是 PixelShuffle，也不是转置卷积。
- 判别器为 VGG-style Discriminator，使用卷积、BatchNorm、LeakyReLU 和全连接输出真假 logits。
- 损失函数已实现 Pixel Loss、Perceptual Loss、GAN Loss，组合损失由 `CombinedLoss` 管理。
- GAN 类型枚举支持 vanilla、lsgan、wgan、wgan-gp、hinge；默认配置为 vanilla。当前代码没有实现 Relativistic GAN。
- 训练流程由 C++ `Trainer` 实现，支持从头训练、checkpoint 加载、训练状态保存、优化器状态保存和 Ctrl+C 安全中断。
- 训练阶段由 `phase1_epochs`、`phase2_epochs` 控制：阶段 1 仅像素损失，阶段 2 加入感知损失，阶段 3 加入 GAN 损失。
- 数据集模块支持从 HR 图像在线 bicubic 下采样生成 LR，支持水平翻转和 90 度旋转增强。
- 推理模块支持单图和文件夹顺序批处理，优先尝试加载 TorchScript，失败后回退到 LibTorch 原生 `torch::load`。
- 命令行推理支持 `--model`、`--input`、`--output`、`--scale`、`--cpu`、`--attention`。
- 生成器 forward 当前先执行一次 2× 上采样，只有 `scale == 4` 时再执行第二次 2× 上采样；因此项目主路径应按 4× 超分使用。
- GUI 支持打开图像、加载模型、开始重建、保存结果、选择目录批量处理，并使用后台线程避免界面阻塞。
- GUI 当前使用 `DeviceType::Auto` 创建推理器，界面上没有 attention 参数开关；启用 CBAM 的 LibTorch 原生模型建议使用命令行 `--attention` 推理。
- Python 评估脚本用于离线评估，支持 PSNR、SSIM、LPIPS、分层分析和报告生成。

## 已预留但当前未完整实现

- `Inference` 头文件中预留了 `DeviceType::Hybrid`、`BoundedQueue` 和流水线接口，但 `src/inference.cpp` 当前文件夹处理仍是顺序逐张处理。
- 配置中有 `frequency_weight` 和 `gradient_weight`，训练器会读取和打印，但 `CombinedLoss` 当前没有实际计算频域损失和梯度损失。
- 配置中有 `use_spectral_norm`，判别器构造函数接收该参数，但当前没有对卷积层实际应用 spectral norm。
- `GANType::WGAN_GP` 当前在 `GANLoss` 中按 WGAN 形式计算，没有实现 WGAN-GP 的梯度惩罚；训练器实现的是可选 R1 penalty。
- `DeviceType` 枚举包含 auto/cuda/hybrid/cpu 概念，但 `main_test.cpp` 当前没有 `--device` 参数，只有默认自动使用 CUDA 和 `--cpu` 强制 CPU。
- 当前代码没有实现 FP16 推理、TensorRT、模型量化、梯度裁剪、学习率调度器、真实退化建模或视频超分。
- GUI 当前没有拖拽上传、PSNR/SSIM 实时显示、参数设置面板或精确批处理进度百分比。
- 推理器不直接加载普通 Python `state_dict`；若从 Python 侧导出模型，应导出 TorchScript，或保证保存格式与当前 C++ RRDBNet 加载逻辑匹配。

## 文档撰写口径

答辩和说明文档应使用以下表述：

- “当前核心推理实现支持 CUDA/CPU，文件夹批处理为顺序处理；混合流水线是预留扩展方向。”
- “当前上采样使用最近邻插值加卷积，避免转置卷积常见棋盘格伪影。”
- “当前核心损失为像素损失、感知损失和 GAN 损失；频域损失和梯度损失是后续增强项。”
- “当前默认 GAN 为 vanilla GAN，可配置 hinge、lsgan 等类型；代码没有实现 Relativistic GAN。”
- “当前项目用 C++/LibTorch 完成模型定义、训练和推理；Python 主要用于离线评估和报告生成。”
- “当前 GUI 适合演示默认结构模型；需要 attention 结构开关时使用命令行推理。”
