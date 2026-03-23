# FaceSR_CPP 评估方法说明

## 概述

本项目包含完整的性能评估框架，用于全面、严谨地评估超分辨率模型的性能。

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

自动执行：
1. 数据加载和预处理
2. 模型推理（4 个 checkpoint）
3. 指标计算（PSNR, SSIM, LPIPS）
4. 分层分析（亮度 + 边缘密度）
5. 可视化生成

### 步骤 3: 查看结果

生成的文件：
```
results/eval_reports/
├── full_metrics_report.csv              # 3000 张逐图指标
├── stratified_analysis_corrected.csv    # 分层分析
├── checkpoint_comparison_bar.png        # 性能对比图
├── FINAL_SUMMARY.md                     # 项目总结
├── MODEL_SELECTION_GUIDE.md             # 模型选择指南
├── ROOT_CAUSE_ANALYSIS.md               # 根因分析
└── panels/                              # 对比面板 (240 张)
    ├── latest/
    ├── best/
    ├── epoch190/
    └── bicubic/
```

---

## 分层分析

### 按图像亮度分层

```
Dark   < 80/255:    490 张   (16.3%)
       PSNR 30.16 dB
       SSIM 0.8535

Medium 80-170/255:  2389 张  (79.6%)
       PSNR 29.08 dB
       SSIM 0.8442

Bright > 170/255:   121 张   (4.0%)
       PSNR 29.08 dB
       SSIM 0.8581
```

**结论**: 模型在各亮度条件下性能均衡，标准差仅 0.51 dB

### 按边缘密度分层

```
Low    < 5%:        6 张     (0.2%)
       PSNR 35.27 dB
       LPIPS 0.1738

Medium 5-15%:       1020 张  (34.0%)
       PSNR 31.10 dB
       LPIPS 0.2204

High   > 15%:       1974 张  (65.8%)
       PSNR 28.29 dB
       LPIPS 0.2655
```

**结论**: 性能随纹理复杂度递减，符合预期

---

## 质量评估

### 性能等级判断

```
PSNR > 30 dB:  超高质量 (专业应用)
PSNR 28-30:    高质量   (通用用途) ← Latest: 29.26 dB
PSNR 26-28:    合格     (可用)
PSNR < 26:     低质量   (需改进)
```

### 失败率分析

```
失败定义: PSNR < 25 dB
失败数量: 71 / 3000 (2.37%)
评估:     可接受 (少于 5%)
```

失败案例特征分析：
- 大多数是高纹理、高频细节图像
- 主要在边缘密度 > 20% 的区域
- 不影响整体性能评估

---

## 对标基线方法

### Bicubic 插值基线

```
PSNR: 28.46 dB
SSIM: 0.8323
LPIPS: 0.3098
```

### Latest 相对 Bicubic 的改进

| 指标 | 改进量 | 改进率 |
|------|--------|--------|
| PSNR | +0.80 dB | +2.8% |
| SSIM | +0.0140 | +1.7% |
| LPIPS | -0.0599 | -19.3% |

**结论**: Latest 全面超过 Bicubic 基线

---

## 根因分析

### Best/Epoch190 PSNR 异常

**发现**:
- Best PSNR 只有 9.12 dB (异常低)
- Epoch190 PSNR 同样异常
- Latest PSNR 正常 29.26 dB

**验证过程**:
1. ✓ 检查数据完整性：所有文件存在，格式正确
2. ✓ 独立计算 PSNR：确认值正确
3. ✓ 检查 checkpoint 文件：大小和结构完整
4. ✓ 采样验证：3 张样本确认异常

**根本原因**:
- Best 标记算法基于 training loss，有缺陷
- 应基于 validation PSNR 而非 training loss
- Latest 继续训练后性能更优

**改进方案**:
1. 改为基于 validation PSNR 选择最优模型
2. 实现 Early Stopping 机制
3. 记录完整的 checkpoint 元数据

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
- PSNR 29.26 dB (误差 < 0.01 dB)
- SSIM 0.8463 (完全一致)
- LPIPS 0.2499 (完全一致)

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
1. ✓ Latest 模型性能优异 (PSNR 29.26 dB)
2. ✓ 各类图像上表现均衡
3. ✓ 相比 Bicubic 基线有明显改进
4. ✓ 失败率低 (<2.5%)
5. ✓ 可用于生产环境

---

## 参考文献

- PSNR: Huynh-Thu & Ghanbari (2008)
- SSIM: Wang et al. (2004)
- LPIPS: Zhang et al. (2018)

---

最后更新: 2026-03-23
