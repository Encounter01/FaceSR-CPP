#!/usr/bin/env python3
"""Generate a thesis readiness note using the current PSNR/SSIM metrics."""

from csv import DictReader
from datetime import datetime
from pathlib import Path


SUMMARY_CSV = Path("results/eval_reports/checkpoint_comparison_summary.csv")
OUTPUT_PATH = Path("results/eval_reports/THESIS_ASSESSMENT.md")


def _load_rows():
    with SUMMARY_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["checkpoint"]: row for row in DictReader(f)}


def _f(value):
    return f"{float(value):.6f}"


def generate_thesis_assessment():
    rows = _load_rows()
    bicubic = rows["bicubic"]
    epoch190 = rows["epoch190"]
    best = rows["best"]
    latest = rows["latest"]

    report = f"""# FaceSR_CPP 本科毕业设计完成度评估

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 当前可信实验结果

当前正式定量结论仅引用 `{SUMMARY_CSV.as_posix()}` 中可直接核对的 PSNR/SSIM。

| 模型 | PSNR | SSIM | 说明 |
|------|-----:|-----:|------|
| Bicubic | {_f(bicubic['avg_psnr'])} | {_f(bicubic['avg_ssim'])} | 插值基线 |
| generator_epoch190.pt | {_f(epoch190['avg_psnr'])} | {_f(epoch190['avg_ssim'])} | 当前推荐 |
| generator_best.pt | {_f(best['avg_psnr'])} | {_f(best['avg_ssim'])} | 对比/备用 |
| generator_latest.pt | {_f(latest['avg_psnr'])} | {_f(latest['avg_ssim'])} | 对比/备用 |

`generator_epoch190.pt` 相比 Bicubic 的提升为 PSNR `{_f(epoch190['avg_psnr_gain_vs_bicubic'])}` dB、
SSIM `{_f(epoch190['avg_ssim_gain_vs_bicubic'])}`。

## 毕设完成度判断

- 理论基础：具备。项目覆盖人脸超分辨率、RRDB、GAN 训练和多指标评估。
- 系统实现：具备。C++/LibTorch 训练与推理、Qt GUI、CMake 构建路径均已实现。
- 实验验证：具备。当前有 3000 张测试集的 PSNR/SSIM 复评结果，且推荐模型优于 Bicubic 基线。
- 工程交付：具备。命令行推理、GUI 和模型文件均可运行。

## 需要在论文中谨慎处理的内容

- LPIPS、失败率、分层分析不再引用旧报告数值。
- 若论文需要这些指标，应重新运行完整评估，并确保 checkpoint、SR 输出、逐图指标和论文表格来自同一批次。
- 已清理的旧评估报告不再作为模型选择、答辩或论文证据。

## 建议写法

当前可直接写入论文和答辩材料的结论是：在 3000 张 CelebA 测试图像上，
`generator_epoch190.pt` 达到 PSNR `{_f(epoch190['avg_psnr'])}` dB、SSIM `{_f(epoch190['avg_ssim'])}`，
相较 Bicubic 基线分别提升 `{_f(epoch190['avg_psnr_gain_vs_bicubic'])}` dB 和
`{_f(epoch190['avg_ssim_gain_vs_bicubic'])}`。
"""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"[OK] Thesis assessment generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_thesis_assessment()
