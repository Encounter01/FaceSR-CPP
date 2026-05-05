# FaceSR_CPP 消融实验方案

## 目的

本实验用于回答四个问题：

1. 仅使用 L1 像素损失能达到什么重建上限。
2. 加入 VGG 感知损失是否提升结构相似度或视觉细节。
3. GAN 从第 0 轮直接加入，与三阶段渐进训练相比是否更稳定。
4. CBAM 注意力模块是否带来可量化收益。

当前正式对照指标来自 `results/eval_reports/checkpoint_comparison_summary.csv`：

| Model | PSNR | SSIM |
|---|---:|---:|
| Bicubic | 28.463390 | 0.819148 |
| generator_best.pt | 29.441479 | 0.840618 |
| generator_latest.pt | 29.258062 | 0.832859 |
| generator_epoch190.pt | 29.552015 | 0.843166 |

LPIPS、失败率、分层分析不再引用旧报告；如需使用，必须和本次消融 checkpoint 在同一批 SR 输出上重新计算。

## 实验组

| 编号 | 配置文件 | 变量 | 训练阶段 | 注意力 |
|---|---|---|---|---|
| A1 | `config/ablations/a1_l1_only.ini` | L1 only | 300 epoch 全程 L1 | 否 |
| A2 | `config/ablations/a2_l1_perceptual.ini` | L1 + VGG | 300 epoch 全程 L1 + perceptual | 否 |
| A3 | `config/ablations/a3_full_nonprogressive.ini` | L1 + VGG + GAN | 300 epoch 全程 full loss | 否 |
| A4 | `config/ablations/a4_three_stage.ini` | 三阶段训练 | 0-49 L1，50-149 L1+VGG，150-299 full loss | 否 |
| A5 | `config/ablations/a5_three_stage_attention.ini` | 三阶段 + CBAM | 同 A4 | 是 |

控制变量：训练集、验证集、测试集、HR/LR 尺寸、batch size、学习率、epoch 数、VGG 权重、GAN 类型均保持一致。输出目录全部隔离到 `checkpoints_ablation/` 和 `results/ablation_runs/`。

## 训练命令

先预览将执行的命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ablation/run_ablation_plan.ps1
```

运行单组实验：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ablation/run_ablation_plan.ps1 -Run -Experiment a1_l1_only
```

按 A1 到 A5 顺序运行全部实验：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ablation/run_ablation_plan.ps1 -Run -Experiment all
```

如需强制 CPU：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ablation/run_ablation_plan.ps1 -Run -Experiment a1_l1_only -Cpu
```

完整 5 组 300 epoch 训练耗时较长，正式论文数据建议全部跑满；调试流程时可临时复制配置并降低 `num_epochs`，但调试结果不能写入正式结论。

## 评估命令

训练完成后，先预览评估命令：

```bash
python scripts/ablation/evaluate_ablation_checkpoints.py --checkpoint-type best --n-images 3000 --device cpu
```

执行正式评估：

```bash
python scripts/ablation/evaluate_ablation_checkpoints.py --checkpoint-type best --n-images 3000 --device cpu --run
```

同时评估 `best` 和 `latest`：

```bash
python scripts/ablation/evaluate_ablation_checkpoints.py --checkpoint-type both --n-images 3000 --device cpu --run
```

辅助脚本会自动给 A5 加上 `--attention` 推理标记。也可以直接使用通用复评脚本评估任意 checkpoint：

```bash
python scripts/evaluation/final_recompute_eval.py \
  --n-images 3000 \
  --checkpoints a1=checkpoints_ablation/a1_l1_only/generator_best.pt,a5=checkpoints_ablation/a5_three_stage_attention/generator_best.pt \
  --attention-checkpoints a5 \
  --device cpu
```

## 结果记录模板

| 实验 | Checkpoint | PSNR | PSNR gain vs Bicubic | SSIM | SSIM gain vs Bicubic | Negative PSNR gain count | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| A1 L1 only | best |  |  |  |  |  |  |
| A2 L1 + VGG | best |  |  |  |  |  |  |
| A3 full non-progressive | best |  |  |  |  |  |  |
| A4 three-stage | best |  |  |  |  |  |  |
| A5 three-stage + CBAM | best |  |  |  |  |  |  |

正式论文优先报告验证 PSNR 最高的 `generator_best.pt`。如果 `best` 与 `latest` 差距较大，应同时列出两者，并说明选择依据。

## 判读规则

- A2 相比 A1 的差异，说明感知损失的贡献。
- A3 相比 A4 的差异，说明渐进式训练是否提升稳定性。
- A5 相比 A4 的差异，说明 CBAM 注意力的净贡献。
- A4/A5 与当前 `generator_epoch190.pt` 的差异，可判断重新训练是否超过现有推荐模型。

每次更新 checkpoint 后，都应重新运行 `scripts/evaluation/final_recompute_eval.py` 或上述消融评估脚本，不要混用旧 CSV、旧 LPIPS 或旧失败率报告。
