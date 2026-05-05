# FaceSR_CPP 评估方法说明

## 概述

本项目包含完整的性能评估框架，用于评估超分辨率模型的性能。当前可直接引用的最新指标以
`results/eval_reports/checkpoint_comparison_summary.csv` 为准，该文件记录了 3000 张测试图像上的
PSNR/SSIM 复评结果。

> 当前 C++ 指标模块实现了 PSNR 和 SSIM；LPIPS、分层分析、可视化报告由 Python 评估脚本完成。评估脚本属于离线分析工具，不是 `facesr_test` 可执行程序的一部分。当前代码实现边界见 `docs/IMPLEMENTATION_STATUS.md`。

### 评估指标

| 指标 | 说明 | 范围 | 越大越好 |
|------|------|------|---------|
| PSNR | 峰值信噪比 (dB) | 0-50 | ✓ 是 |
| SSIM | 结构相似度 | 0-1 | ✓ 是 |
| LPIPS | 感知相似度 | 0-1 | ✗ 否 |

---

## 评估流程

### 步骤 1: 准备测试数据

```bash
# 使用 3000 张 CelebA 人脸测试集
data/test/HR/        # 256x256 原始图像 3000 张
data/test/LR/        # 64x64 低分图像 3000 张
```

### 步骤 2: 生成超分结果

```bash
# 使用模型进行推理
python scripts/evaluation/evaluate_model.py \
    --checkpoint latest \
    --n_images 3000
```

推荐使用 `scripts/evaluation/final_recompute_eval.py` 重新生成最终复评输出。完整评估可执行：
```bash
python scripts/evaluation/final_recompute_eval.py \
    --n-images 3000 \
    --checkpoints best,latest,epoch190 \
    --device cpu
```

评估流程包括：
1. 数据加载和预处理
2. 模型推理（best/latest/epoch190）
3. 指标计算（当前正式口径为 PSNR/SSIM）
4. 输出汇总文件与必要的中间结果

注意：`scripts/evaluation/evaluate_model.py` 仍可用于完整离线分析，但历史 LPIPS、分层分析和失败率报告已经不再作为当前结论引用。若需要 LPIPS 或失败率，应与当前 checkpoint 同批次重新生成。

### 步骤 3: 查看结果

生成的文件：
```
results/eval_reports/
├── checkpoint_comparison_summary.csv    # 当前可信 PSNR/SSIM 汇总
├── checkpoint_comparison_bar.png        # 性能对比图
├── model_selection_and_sharper_plan.md  # 当前模型选择和锐化计划
├── worst_cases_best.csv                 # best checkpoint 的低增益样本
├── OLD_RESULTS_CLEARED.md               # 旧评估结果清理说明
└── panels/                              # 对比面板 (240 张)
    ├── latest/
    ├── best/
    ├── epoch190/
    └── bicubic/
```

---

## 分层分析

旧版分层分析来自已清理的历史评估批次，不再作为当前论文或答辩结论引用。若需要按亮度、边缘密度或
失败案例统计，应重新运行完整评估，并确保分层文件、SR 输出、checkpoint 文件和论文表格来自同一次评估。

---

## 质量评估

### 性能等级判断

```
PSNR > 30 dB:  超高质量 (专业应用)
PSNR 28-30:    高质量   (通用用途) ← Epoch190: 29.55 dB
PSNR 26-28:    合格     (可用)
PSNR < 26:     低质量   (需改进)
```

### 失败率分析

当前仓库不再保留与 `generator_epoch190.pt` 最新复评同批次的失败率文件。若论文或答辩需要失败率，
应先定义阈值（例如 PSNR < 25 dB），再用当前 checkpoint 的同批次逐图指标重新统计。

---

## 对标基线方法

### Bicubic 插值基线

```
PSNR: 28.46 dB
SSIM: 0.8191
```

### 当前 checkpoint 对比

`results/eval_reports/checkpoint_comparison_summary.csv` 中当前可验证的 3000 张测试集 PSNR/SSIM：

| Checkpoint | PSNR | PSNR gain | SSIM | SSIM gain |
|------------|-----:|----------:|-----:|----------:|
| Bicubic | 28.463390 | - | 0.819148 | - |
| Best | 29.441479 | +0.978087 | 0.840618 | +0.021470 |
| Latest | 29.258062 | +0.794669 | 0.832859 | +0.013711 |
| Epoch190 | 29.552015 | +1.088623 | 0.843166 | +0.024018 |

**结论**: 当前 PSNR/SSIM 口径下，`generator_epoch190.pt` 优先于 `generator_best.pt` 和 `generator_latest.pt`。LPIPS 结果应以重新运行评估脚本后的新报告为准，不要混用旧报告中的 LPIPS 表格。

---

## 根因分析

### 历史报告与当前复评不一致

**发现**:
- 已清理的旧版评估报告曾记录 Best、Epoch190 的异常低 PSNR。
- 当前 `checkpoint_comparison_summary.csv` 重新评估后，Best 为 29.441479 dB，Epoch190 为 29.552015 dB。
- 因此旧报告中的异常结论不应继续作为当前结论使用。

**验证过程**:
1. 检查当前汇总文件：`results/eval_reports/checkpoint_comparison_summary.csv`
2. 对比当前 README 推荐模型：`generator_epoch190.pt`
3. 回看源码：`Trainer::save_checkpoint` 当前 best 是基于验证 PSNR 更新，不是基于 training loss

**根本原因**:
- 旧报告和后续复评结果来自不同评估批次或不同中间文件，不能混合引用。
- 当前源码中的 best 保存逻辑已经是 `if (psnr > best_psnr) save_checkpoint(epoch, "best")`。
- 如果正式论文或答辩需要引用指标，应重新运行完整评估并统一使用同一次输出。

**改进方案**:
1. 删除或标注过期评估报告，避免与当前 CSV 冲突
2. 每次替换 checkpoint 后重新生成 `checkpoint_comparison_summary.csv`
3. 在报告中注明 checkpoint 文件名、评估脚本参数和评估日期

---

## 如何复现结果

### 快速复现 (10 分钟)

```bash
# 使用已有的 SR 输出目录
python scripts/evaluation/evaluate_model.py \
    --n_images 3000 \
    --force_recompute_metrics
```

预期结果:
- Epoch190 PSNR 约 29.55 dB
- Epoch190 SSIM 约 0.8432
- 若启用 LPIPS，应以本次新生成的报告为准

### 完整复现 (1 小时)

```bash
# 从零开始，包括生成 SR 图像
python scripts/evaluation/evaluate_model.py \
    --n_images 3000 \
    --force_regen_lr \
    --force_regen_sr
```

需要:
- GPU (推荐 CUDA 11.8+)
- 3000 张 HR 测试图像
- 约 30GB 临时空间

---

## 性能指标解释

### PSNR (Peak Signal-to-Noise Ratio)

- 衡量像素级相似度
- 单位：dB (分贝)
- 越高越好
- 典型范围：20-40 dB
- 局限：只看像素差异，不看感知质量

### SSIM (Structural Similarity Index)

- 衡量结构相似度（比 PSNR 更接近人眼感知）
- 范围：0-1
- 越接近 1 越好
- 考虑亮度、对比度、结构三个方面

### LPIPS (Learned Perceptual Image Patch Similarity)

- 使用预训练 CNN (AlexNet) 计算感知相似度
- 范围：0-1
- 越低越好
- 最符合人类视觉感知
- 推荐用于超分辨率评估

---

## 结论

完整的 3000 张图像评估表明：
1. ✓ Epoch190 模型当前 PSNR/SSIM 最优 (PSNR 29.552 dB, SSIM 0.8432)
2. ✓ 相比 Bicubic 基线有明显改进
3. ✓ Best 与 Latest 均优于 Bicubic，但当前推荐优先使用 Epoch190
4. ✓ 具备演示和实验验证价值
5. △ LPIPS、分层分析和失败率需用当前 checkpoint 重新完整评估后再引用

上述 PSNR/SSIM 数值来自 `checkpoint_comparison_summary.csv`。若重新训练或替换 checkpoint，应重新运行评估脚本，并以新生成的同批次文件为准。

---

## 参考文献

- PSNR: Huynh-Thu & Ghanbari (2008)
- SSIM: Wang et al. (2004)
- LPIPS: Zhang et al. (2018)

---

最后更新: 2026-05-03
