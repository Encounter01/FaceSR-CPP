# 超分辨率人脸图像重建系统 (C++版本)

基于LibTorch和Qt的人脸图像4倍超分辨率重建系统，支持GPU+CPU混合加速管线。

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

## GPU+CPU混合模式

系统支持GPU+CPU混合加速管线，最大化硬件利用率：

| 任务 | 设备 | 说明 |
|------|------|------|
| 模型前向/反向传播 | GPU | 神经网络计算密集型操作 |
| 参数更新 | GPU | 梯度更新在GPU上完成 |
| 数据加载 | CPU | 多线程并行读取与预处理 |
| 数据增强 | CPU | 图像变换等预处理操作 |
| 图像后处理/保存 | CPU | IO密集型操作 |

推理时使用三阶段流水线并行：CPU线程负责图像加载/预处理/后处理/保存，GPU线程负责神经网络推理，二者并发执行。

### 设备模式

| 模式 | 说明 |
|------|------|
| `auto` | 自动检测，有GPU时使用混合模式 |
| `hybrid` | GPU+CPU混合模式（推荐） |
| `gpu` | 仅GPU模式 |
| `cpu` | 仅CPU模式 |

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
    --val-lr data/val/LR \
    --batch-size 16 \
    --epochs 200 \
    --lr 0.0002 \
    --device auto \
    --workers 4 \
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
| `--val-lr <路径>` | data/val/LR | 验证集LR图像目录 |
| `--batch-size <数值>` | 12 | 批次大小 |
| `--epochs <数值>` | 300 | 训练轮数 |
| `--lr <数值>` | 0.0002 | 学习率 |
| `--device <类型>` | auto | 设备模式：auto/hybrid/gpu/cpu |
| `--workers <数值>` | 4 | 数据加载线程数 |
| `--no-pin-memory` | - | 禁用锁页内存（默认启用） |
| `--config <路径>` | - | 配置文件路径 |
| `--resume <路径>` | - | 从检查点恢复训练 |

### 推理/测试

```bash
# 处理单张图像
facesr_test --model checkpoints/generator_epoch190.pt --input image.jpg --output result.png

# 批量处理
facesr_test --model checkpoints/generator_epoch190.pt --input input_folder --output output_folder

# 指定设备模式
facesr_test --model checkpoints/generator_epoch190.pt --input image.jpg --device hybrid

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
| `--output <路径>` | results | 输出路径 |
| `--scale <数值>` | 4 | 放大倍数 |
| `--device <类型>` | auto | 设备模式：auto/hybrid/gpu/cpu |
| `--cpu` | - | 仅使用CPU（等效于 `--device cpu`） |

### GUI应用

```bash
facesr_gui_app
```

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
- PixelShuffle上采样
- 参数量: ~16.7M

### 判别器 (VGG Style)
- VGG风格多层卷积
- 全连接输出层
- 参数量: ~8.3M

## 训练策略

三阶段渐进式训练（与 `config/train_config.ini` 对齐）：

| 阶段 | 轮次范围 | 损失函数 | 说明 |
|------|----------|----------|------|
| 阶段1 | 0 - 30 | 仅L1像素损失 | 稳定初始化网络权重 |
| 阶段2 | 30 - 80 | L1 + VGG感知损失 | 引入感知损失提升视觉质量 |
| 阶段3 | 80 - 200 | L1 + VGG + GAN对抗损失 | 对抗训练生成高频细节 |

阶段边界可在 `config/train_config.ini` 中通过 `phase1_epochs` 和 `phase2_epochs` 参数调整。

### 更锐利的训练配置

如果你更在意清晰度和面部细节，而不是只追求更平滑的 `PSNR`，优先查看：

- `config/train_config_sharper.ini`
- `config/finetune_phase_a.ini`
- `config/finetune_phase_b.ini`
- `config/finetune_phase_c.ini`

这组配置会启用 `models/vgg19_features.pt`、CBAM attention、spectral norm，
并打开 frequency / gradient loss，用来抑制“看起来发糊、边缘偏软”的问题。

## 配置文件说明

### config/train_config.ini

训练相关的所有参数配置，包括数据路径、训练超参数、损失权重、设备模式和输出设置。命令行参数会覆盖配置文件中的同名设置。

主要配置节：
- `[data]` - 训练/验证数据路径和图像尺寸
- `[training]` - batch_size、epochs、学习率、阶段边界
- `[loss]` - 损失函数权重和类型
- `[device]` - 设备模式（auto/hybrid/gpu/cpu）和pin_memory
- `[output]` - 检查点和结果输出目录、保存间隔

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
2. **LibTorch版本**: 确保LibTorch版本与PyTorch训练版本兼容
3. **模型格式**: 支持.pt和.pth格式的PyTorch模型
4. **pin_memory**: 混合模式下启用pin_memory可加速CPU到GPU的数据传输

## 常见问题

### Q: 编译时找不到LibTorch
A: 在CMake配置时设置正确的CMAKE_PREFIX_PATH。

### Q: 运行时缺少DLL
A: 将LibTorch/lib下的DLL复制到可执行文件目录。

### Q: CUDA不可用
A: 检查CUDA Toolkit和cuDNN是否正确安装。

### Q: CUDA 11.8与高版本MSVC的兼容性问题
A: CUDA 11.8的nvcc不支持MSVC 19.40+。本项目不编译.cu文件（仅链接预编译的LibTorch CUDA库），CMakeLists.txt中已通过 `TORCH_CUDA_SKIP_NVCC` 选项跳过nvcc编译器检测，无需降级MSVC版本。

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
