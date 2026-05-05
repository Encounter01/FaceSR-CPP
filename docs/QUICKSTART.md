# FaceSR_CPP 快速开始指南

## 项目简介

**超分辨率人脸图像重建系统 (C++版本)**

- 基于 RRDB 神经网络的 4 倍人脸超分辨率
- 使用 LibTorch + Qt 实现 GPU 加速
- 支持 GUI 应用和命令行工具
- 完整的性能评估框架

> 当前源码实现边界见 `docs/IMPLEMENTATION_STATUS.md`。本文只描述当前代码已经支持的启动路径。

### 核心特性

- ✓ 当前复评 `generator_epoch190.pt`：PSNR 29.552 dB、SSIM 0.8432
- ✓ CUDA/CPU 推理，默认有 CUDA 时自动使用 GPU
- ✓ 当前可信评估指标：3000 张测试集 PSNR/SSIM；LPIPS 需重新完整评估后再引用
- ✓ 现代化 GUI 应用

---

## 5 分钟快速开始

### 0. 检查系统要求

```bash
# Windows 11 + Visual Studio 2022 + CUDA 11.8
```

### 1. 克隆和初始化

```bash
git clone <repository>
cd FaceSR_CPP
mkdir build && cd build
```

### 2. 编译

```bash
# 方式 A: CMake 命令行
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release

# 方式 B: Visual Studio IDE
# 打开 Visual Studio
# 文件 -> 打开文件夹 -> 选择 FaceSR_CPP
# 自动检测 CMakeLists.txt 并配置
# 构建 -> 生成全部
```

### 3. 下载模型

从 Releases 页面下载预训练模型，放到 `checkpoints/` 目录：
```
checkpoints/
  ├── generator_latest.pt
  ├── generator_best.pt
  └── generator_epoch190.pt   (当前 README 推荐优先尝试)
```

### 4. 快速推理

```bash
# 推理单张图像
./build/bin/Release/facesr_test.exe \
    --model checkpoints/generator_epoch190.pt \
    --input input.jpg \
    --output output.png

# 批量推理
./build/bin/Release/facesr_test.exe \
    --model checkpoints/generator_epoch190.pt \
    --input input_folder/ \
    --output output_folder/

# 强制 CPU 推理
./build/bin/Release/facesr_test.exe \
    --model checkpoints/generator_epoch190.pt \
    --input input.jpg \
    --output output.png \
    --cpu
```

### 5. 启动 GUI

```bash
./build/bin/Release/facesr_gui_app.exe
```

GUI 适合演示默认结构模型。若模型训练时启用了 CBAM attention，请优先使用命令行并添加 `--attention`。

---

## 验证安装

运行评估脚本检查系统是否正常：

```bash
cd FaceSR_CPP
python scripts/evaluation/evaluate_model.py
```

预期输出：
- ✓ 生成或更新评估汇总文件
- ✓ 生成性能对比图表
- ✓ 打印性能指标（PSNR ~29 dB）

---

## 项目结构

```
FaceSR_CPP/
├── README.md               项目概览
│
├── include/                C++ 头文件
├── src/                    C++ 源文件
├── config/                 配置文件
│
├── checkpoints/            预训练模型 (下载)
├── data/                   数据目录 (下载或生成)
│
├── results/                评估结果
│   └── eval_reports/       (关键文件在这里)
│
├── scripts/                辅助脚本
│   ├── evaluation/         评估脚本
│   └── utils/              工具脚本
│
├── docs/                   技术文档
│   ├── QUICKSTART.md       快速开始
│   ├── EVALUATION.md       评估说明
│   ├── IMPLEMENTATION_STATUS.md 当前实现状态
│   └── FACESR_LEARNING_TO_DEFENSE_GUIDE.md 学习到答辩指南
│
└── thesis/                 论文相关文件
    ├── figures/            论文图表
    └── references/         参考文献
```

---

## 常见问题

### Q1: 编译失败，找不到 LibTorch

**A:** 设置 CMAKE_PREFIX_PATH：
```bash
cmake .. -DCMAKE_PREFIX_PATH="C:/libtorch;C:/Qt/6.10.1/msvc2022_64"
```

### Q2: 运行时缺少 DLL

**A:** 将 `C:\libtorch\lib\*.dll` 复制到 `build\bin\Release\` 目录

### Q3: CUDA 不可用，如何用 CPU 推理？

**A:** 添加 `--cpu` 参数：
```bash
facesr_test.exe --model model.pt --input image.jpg --cpu
```

### Q4: 如何使用自己的数据进行训练？

**A:** 准备 `data/train/hr` 和 `data/val/hr` 目录，运行：

```bash
facesr_train --config config/train_config.ini
```

### Q5: 模型性能不佳怎么办？

**A:** 先确认使用的模型结构是否与推理参数一致，例如训练时启用了 CBAM，则推理也需要 `--attention`。再通过 `docs/EVALUATION.md` 中的评估脚本复查 PSNR/SSIM；LPIPS 需要和当前 checkpoint 同批次重新计算后再引用。

---

## 下一步

- 当前实现状态：见 `docs/IMPLEMENTATION_STATUS.md`
- 从学习到答辩：见 `docs/FACESR_LEARNING_TO_DEFENSE_GUIDE.md`
- 评估方法：见 `docs/EVALUATION.md`

---

## 性能指标 (3000 张 CelebA 测试集)

当前 `results/eval_reports/checkpoint_comparison_summary.csv` 中可验证的 PSNR/SSIM 结果：

| Checkpoint | PSNR | SSIM |
|------------|-----:|-----:|
| Bicubic | 28.463390 | 0.819148 |
| generator_best.pt | 29.441479 | 0.840618 |
| generator_latest.pt | 29.258062 | 0.832859 |
| generator_epoch190.pt | 29.552015 | 0.843166 |

**说明**: 当前推理示例优先使用 `generator_epoch190.pt`。LPIPS、分层分析和可视化报告应重新运行 `docs/EVALUATION.md` 中的评估脚本后再用于论文或答辩。

---

## 获取帮助

1. 查看当前实现状态：`docs/IMPLEMENTATION_STATUS.md`
2. 查看学习到答辩指南：`docs/FACESR_LEARNING_TO_DEFENSE_GUIDE.md`
3. 查看评估方法：`docs/EVALUATION.md`

---

**开始使用**
