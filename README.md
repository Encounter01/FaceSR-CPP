# 超分辨率人脸图像重建系统 (C++版本)

基于 LibTorch、OpenCV 和 Qt 的人脸图像 4 倍超分辨率重建系统。当前源码实现了 C++ 训练、命令行推理、Qt GUI 和 Python 离线评估。

> 当前实现细节以 [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) 为准。历史文档中关于 PixelShuffle、完整 GPU+CPU 混合流水线、frequency/gradient loss、spectral norm、Relativistic GAN 等描述如与源码不一致，以当前源码为准。

## 环境要求

- **编译器**: Visual Studio 2019/2022 (MSVC)
- **CMake**: 3.18+
- **LibTorch**: 2.x+ (CUDA版本推荐)
- **OpenCV**: 4.5+
- **Qt**: 5.15+ 或 6.x+
- **CUDA**: 11.8+ (推荐，用于GPU加速)

### 实际验证环境

| 组件 | 版本 |
|------|------|
| LibTorch | 2.7.1+cu118 |
| OpenCV | 4.12.0 |
| Qt | 6.10.1 |
| CUDA Toolkit | 11.8 |
| MSVC | Visual Studio 2022 |

## 依赖库安装

### 1. 下载LibTorch

从PyTorch官网下载LibTorch (C++ API):
```
https://pytorch.org/get-started/locally/
```

选择:
- PyTorch Build: Stable
- OS: Windows
- Package: LibTorch
- Language: C++/Java
- CUDA: 选择你安装的CUDA版本（推荐11.8）

下载后解压到 `C:\libtorch` (或其他位置)

### 2. 安装OpenCV

方法1: 使用vcpkg
```bash
vcpkg install opencv4:x64-windows
```

方法2: 从官网下载预编译版本
```
https://opencv.org/releases/
```

### 3. 安装Qt

从Qt官网下载安装器:
```
https://www.qt.io/download
```

安装时选择 MSVC 2019/2022 64-bit 组件。CMake会自动检测Qt版本（优先Qt6，回退Qt5）。

## 构建项目

### 使用CMake命令行

```bash
cd FaceSR_CPP

# 创建构建目录
mkdir build && cd build

# 配置 (修改路径为你的实际安装路径)
cmake .. -G "Visual Studio 17 2022" -A x64 \
    -DCMAKE_PREFIX_PATH="C:/libtorch;C:/Qt/6.10.1/msvc2022_64;C:/opencv/build"

# 构建
cmake --build . --config Release
```

### 使用Visual Studio

1. 打开Visual Studio 2022
2. 选择 "打开本地文件夹"
3. 选择 `FaceSR_CPP` 目录
4. VS会自动检测CMakeLists.txt并配置项目
5. 在CMakeSettings.json中设置依赖库路径
6. 选择 "生成" -> "全部生成"

## 设备与并行现状

当前代码的训练和推理都支持 CPU/CUDA。数据读取、图像预处理、数据增强和图像保存由 CPU/OpenCV 完成；神经网络前向、反向传播和参数更新在 LibTorch 选择的设备上执行。

| 任务 | 设备 | 说明 |
|------|------|------|
| 模型前向/反向传播 | GPU | 神经网络计算密集型操作 |
| 参数更新 | GPU | 梯度更新在GPU上完成 |
| 数据加载 | CPU | 多线程并行读取与预处理 |
| 数据增强 | CPU | 图像变换等预处理操作 |
| 图像后处理/保存 | CPU | IO密集型操作 |

`include/inference.h` 中预留了 Hybrid 流水线接口和有界队列，但 `src/inference.cpp` 当前的文件夹处理是顺序逐张处理，不是完整的 CPU 预处理、GPU 推理、CPU 后处理三阶段并行流水线。

当前命令行推理默认自动使用 CUDA（如果可用），可通过 `--cpu` 强制 CPU。

## 使用方法

### 训练模型

```bash
# 基本用法
facesr_train --train-hr data/train/HR --epochs 200

# 完整参数
facesr_train \
    --train-hr data/train/HR \
    --train-lr data/train/LR \
    --val-hr data/val/HR \
    --batch-size 16 \
    --epochs 200 \
    --lr 0.0002 \
    --config config/train_config.ini

# 从检查点恢复训练
facesr_train --resume checkpoints --epochs 200
```

**训练参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train-hr <路径>` | data/train/HR | 训练集HR图像目录 |
| `--train-lr <路径>` | data/train/LR | 训练集LR图像目录 |
| `--val-hr <路径>` | data/val/HR | 验证集HR图像目录 |
| `--batch-size <数值>` | 12 | 批次大小 |
| `--epochs <数值>` | 300 | 训练轮数 |
| `--lr <数值>` | 0.0002 | 学习率 |
| `--config <路径>` | - | 配置文件路径 |
| `--resume <路径>` | - | 从检查点恢复训练 |
| `--cpu` | - | 强制 CPU 训练 |

其他训练参数（如 `val_lr_dir`、`num_workers`、阶段边界和损失权重）通过 `config/train_config.ini` 配置。

### 推理/测试

```bash
# 处理单张图像
facesr_test --model checkpoints/generator_epoch190.pt --input image.jpg --output result.png

# 批量处理
facesr_test --model checkpoints/generator_epoch190.pt --input input_folder --output output_folder

# 仅使用CPU
facesr_test --model checkpoints/generator_epoch190.pt --input image.jpg --cpu
```

> 当前仓库内 `results/eval_reports/checkpoint_comparison_summary.csv` 的 3000 张测试集复评结果显示：
> `generator_epoch190.pt` 的平均 `PSNR/SSIM` 为 `29.552015 / 0.843166`，
> 高于 `generator_best.pt` 的 `29.441479 / 0.840618`。
> 因此，当前推理默认推荐优先使用 `checkpoints/generator_epoch190.pt`。

**推理参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model <路径>` | **必需** | 模型文件路径 |
| `--input <路径>` | **必需** | 输入图像或文件夹路径 |
| `--output <路径>` | 单图默认生成同目录 `原名_sr.ext`；目录默认生成 `输入目录_sr/` | 输出路径 |
| `--scale <数值>` | 4 | 放大倍数；当前生成器代码主要按 4× 设计，非 4 时只执行一次 2× 上采样 |
| `--cpu` | - | 仅使用CPU |
| `--attention` | - | 启用 CBAM attention，必须与训练时模型结构一致 |

### GUI应用

```bash
facesr_gui_app
```

注意：当前 GUI 通过 `DeviceType::Auto` 创建推理器，界面上没有 `--attention` 等结构参数开关。若加载的是启用 CBAM 的 LibTorch 原生权重，优先使用命令行 `facesr_test --attention`；TorchScript 模型则由文件本身携带网络结构。

## 项目结构

```
FaceSR_CPP/
├── include/                    # 头文件
│   ├── common/                # 公共工具
│   │   ├── config.h
│   │   ├── config_parser.h
│   │   ├── logger.h
│   │   └── random.h
│   ├── models/                # 模型定义
│   │   ├── generator.h
│   │   ├── discriminator.h
│   │   └── losses.h
│   ├── utils/                 # 工具函数
│   │   ├── dataset.h
│   │   ├── image_utils.h
│   │   └── metrics.h
│   ├── gui/                   # GUI组件
│   │   ├── main_window.h
│   │   ├── style_config.h
│   │   ├── background_widget.h
│   │   └── frosted_widget.h
│   ├── trainer.h
│   └── inference.h
├── src/                        # 源文件
│   ├── models/
│   │   ├── generator.cpp
│   │   ├── discriminator.cpp
│   │   └── losses.cpp
│   ├── utils/
│   │   ├── dataset.cpp
│   │   ├── image_utils.cpp
│   │   └── metrics.cpp
│   ├── gui/
│   │   ├── main_window.cpp
│   │   ├── style_config.cpp
│   │   ├── background_widget.cpp
│   │   └── frosted_widget.cpp
│   ├── trainer.cpp
│   ├── inference.cpp
│   ├── main_train.cpp          # 训练入口
│   ├── main_test.cpp           # 测试/推理入口
│   └── main_gui.cpp            # GUI入口
├── config/                     # 配置文件
│   ├── train_config.ini        # 训练配置
│   └── ui_config.ini           # UI样式配置
├── scripts/                    # 辅助脚本
│   ├── evaluation/            # 评估脚本
│   │   ├── evaluate_model.py
│   │   ├── evaluate_model_final.py
│   │   └── diagnose_sr_issue.py
│   ├── analysis/              # 分析与报告脚本
│   │   ├── analyze_checkpoint_quality.py
│   │   ├── training_decision_analysis.py
│   │   ├── generate_thesis_assessment.py
│   │   ├── generate_root_cause_analysis.py
│   │   └── generate_final_summary.py
│   ├── data/                  # 数据处理脚本
│   │   ├── extract_and_prepare.py
│   │   └── gen_lr.py
│   ├── autodl_setup.sh
│   ├── finetune.sh
│   ├── train.sh
│   └── train.bat
├── docs/                       # 项目文档
│   ├── QUICKSTART.md
│   └── EVALUATION.md
├── thesis/                     # 论文与答辩材料
├── models/                     # 预训练特征模型
├── data/                       # 数据目录
│   ├── train/
│   │   ├── HR/
│   │   └── LR/
│   ├── val/
│   │   ├── HR/
│   │   └── LR/
│   └── test/
│       ├── HR/
│       └── LR/
├── checkpoints/                # 模型权重
├── results/                    # 输出结果
├── CMakeLists.txt              # CMake配置
└── README.md
```

## 网络架构

### 生成器 (RRDB Net)
- 23个RRDB块
- 最近邻插值 + 卷积上采样（两次2倍上采样，实现4倍放大）
- 可选CBAM注意力模块
- 参数量: ~16.7M

### 判别器 (VGG Style)
- VGG风格多层卷积
- BatchNorm + LeakyReLU
- 全连接输出层
- 参数量: 约17M（以运行时 `get_num_parameters()` 日志为准）

## 训练策略

三阶段渐进式训练（与 `config/train_config.ini` 对齐）：

| 阶段 | 轮次范围 | 损失函数 | 说明 |
|------|----------|----------|------|
| 阶段1 | 0 - `phase1_epochs` | 仅像素损失 | 学习基础重建映射 |
| 阶段2 | `phase1_epochs` - `phase2_epochs` | 像素损失 + VGG感知损失 | 引入感知特征提升纹理结构 |
| 阶段3 | `phase2_epochs` 之后 | 像素损失 + VGG感知损失 + GAN损失 | 对抗训练提升视觉真实感 |

阶段边界可在 `config/train_config.ini` 中通过 `phase1_epochs` 和 `phase2_epochs` 参数调整。

### 更锐利的训练配置

如果你更在意清晰度和面部细节，而不是只追求更平滑的 `PSNR`，优先查看：

- `config/train_config_sharper.ini`
- `config/finetune_phase_a.ini`
- `config/finetune_phase_b.ini`
- `config/finetune_phase_c.ini`

这组配置会尝试启用 `models/vgg19_features.pt`、CBAM attention、hinge GAN、R1 penalty 等更锐利的训练设定。

注意：当前代码已实现 CBAM、VGG感知损失、hinge GAN 和可选 R1 penalty；`frequency_weight`、`gradient_weight` 和 `use_spectral_norm` 目前只是配置预留项，核心损失中尚未实际计算 frequency/gradient loss，判别器也未实际应用 spectral norm。

## 配置文件说明

### config/train_config.ini

训练相关参数配置，包括数据路径、训练超参数、损失权重、输出设置和模型开关。命令行参数会覆盖配置文件中的同名设置。

主要配置节：
- `[data]` - 训练/验证数据路径和图像尺寸
- `[training]` - batch_size、epochs、学习率、阶段边界
- `[loss]` - 损失函数权重和类型
- `[output]` - 检查点和结果输出目录、保存间隔
- `[model]` - VGG权重路径、CBAM attention、谱归一化预留开关

### config/ui_config.ini

GUI界面样式配置，支持运行时读取，无需重新编译。

主要配置节：
- `[general]` - 启用/禁用毛玻璃和自定义背景
- `[background]` - 背景类型（solid/gradient/image）、渐变参数
- `[frosted]` - 毛玻璃效果参数（模糊半径、不透明度、色调）
- `[theme]` - 主题色、强调色、文本色、边框色

## UI自定义

GUI应用支持现代化毛玻璃（Frosted Glass）质感界面，通过编辑 `config/ui_config.ini` 即可自定义。

### 背景模式

支持三种背景：纯色（solid）、渐变（gradient，含线性和径向）、图片（image）。

```ini
# 线性渐变（默认）
[background]
type=gradient
gradient_type=linear
start_color=#2C3E50
end_color=#3498DB

# 图片背景
[background]
type=image
image_path=D:/background.jpg
```

### 毛玻璃效果

使用栈模糊（Stack Blur）算法，O(n)复杂度，支持0-50可调模糊半径。

```ini
[frosted]
enabled=true
blur_radius=25
opacity=200
tint_color=#FFFFFF
```

### 主题色

```ini
[theme]
primary_color=#4CAF50
accent_color=#45a049
text_color=#FFFFFF
```

### 降级方案

如果需要禁用视觉特效（兼容模式）：

```ini
[general]
enable_frosted=false
enable_custom_background=false
```

## 注意事项

1. **显存需求**: 训练需要约8-12GB显存
2. **LibTorch版本**: 确保 LibTorch、CUDA 和模型保存环境版本兼容
3. **模型格式**: 推理器优先加载 TorchScript `.pt`，失败后按当前 RRDBNet 结构加载 C++ `torch::save` 保存的 LibTorch 权重；普通 Python `state_dict` 不能直接当作完整模型加载
4. **扩展项**: `frequency_weight`、`gradient_weight`、`use_spectral_norm` 等配置项当前为预留能力，详见 `docs/IMPLEMENTATION_STATUS.md`

## 常见问题

### Q: 编译时找不到LibTorch
A: 在CMake配置时设置正确的CMAKE_PREFIX_PATH。

### Q: 运行时缺少DLL
A: 将LibTorch/lib下的DLL复制到可执行文件目录。

### Q: CUDA不可用
A: 检查CUDA Toolkit和cuDNN是否正确安装。

### Q: CUDA 11.8与高版本MSVC的兼容性问题
A: 当前 `CMakeLists.txt` 的 `project(... LANGUAGES CXX)` 只启用 C++，项目本身不编译 `.cu` 文件，CUDA 主要来自预编译的 LibTorch CUDA 库。若本地 CMake 或 LibTorch 仍触发 CUDA 工具链检测，应以实际报错为准调整 CUDA/MSVC/LibTorch 版本组合。

### Q: Qt版本选择
A: CMake会自动检测，优先使用Qt6，回退Qt5。无需手动指定版本。

## 预训练模型

训练好的模型权重文件（`.pt`）不包含在本仓库中，请从 [Releases](../../releases) 页面下载。

下载后放置到 `checkpoints/` 目录：
```
checkpoints/
  generator_epoch190.pt
  generator_best.pt
```

如果只保留一个当前推荐的推理权重，优先使用 `generator_epoch190.pt`。

## 贡献

欢迎提交 Issue 和 Pull Request。

## 许可证

本项目采用 [CC BY-NC-SA 4.0](LICENSE) 许可证。
允许在注明出处且非商业的前提下自由使用和修改，衍生作品须使用相同许可证。

本项目使用的数据集版权归原作者所有：
- [FFHQ](https://github.com/NVlabs/ffhq-dataset) — CC BY-NC-SA 4.0
- [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) — 请参阅原始许可证
