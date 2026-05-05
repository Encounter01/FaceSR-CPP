#!/usr/bin/env python3
"""Analyze whether retraining is needed using current PSNR/SSIM summary metrics."""

from csv import DictReader
from pathlib import Path


SUMMARY_CSV = Path("results/eval_reports/checkpoint_comparison_summary.csv")


def _load_rows():
    with SUMMARY_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["checkpoint"]: row for row in DictReader(f)}


def _num(row, key):
    return float(row[key])


def analyze_training_necessity():
    rows = _load_rows()
    bicubic = rows["bicubic"]
    epoch190 = rows["epoch190"]
    best = rows["best"]
    latest = rows["latest"]

    print("=" * 80)
    print("训练需求分析（当前 PSNR/SSIM 口径）")
    print("=" * 80)
    print(f"数据来源: {SUMMARY_CSV}")
    print()

    print("[1] 当前推荐模型")
    print("-" * 80)
    print("推荐: generator_epoch190.pt")
    print(f"PSNR: {_num(epoch190, 'avg_psnr'):.6f} dB")
    print(f"SSIM: {_num(epoch190, 'avg_ssim'):.6f}")
    print(f"PSNR gain vs Bicubic: {_num(epoch190, 'avg_psnr_gain_vs_bicubic'):+.6f} dB")
    print(f"SSIM gain vs Bicubic: {_num(epoch190, 'avg_ssim_gain_vs_bicubic'):+.6f}")
    print()

    print("[2] Checkpoint 排名")
    print("-" * 80)
    ranked = sorted(
        [("epoch190", epoch190), ("best", best), ("latest", latest)],
        key=lambda item: _num(item[1], "avg_psnr"),
        reverse=True,
    )
    for idx, (name, row) in enumerate(ranked, start=1):
        print(f"{idx}. {name:8s} PSNR={_num(row, 'avg_psnr'):.6f}, SSIM={_num(row, 'avg_ssim'):.6f}")
    print()

    print("[3] 是否需要重新训练")
    print("-" * 80)
    psnr = _num(epoch190, "avg_psnr")
    if psnr >= 30:
        verdict = "当前质量较高，短期无需重新训练。"
    elif psnr >= 29:
        verdict = "当前质量可用于演示和论文验证；若追求更高质量，可规划下一轮训练。"
    else:
        verdict = "建议重新训练或调整退化模型与损失设计。"
    print(verdict)
    print()

    print("[4] 后续改进")
    print("-" * 80)
    print("- 不再引用旧 LPIPS、失败率或分层分析数值。")
    print("- 若需要感知质量结论，应重新运行完整评估并绑定 checkpoint、SR 输出和指标文件。")
    print("- 下一轮训练应保存每个 checkpoint 的 PSNR/SSIM/LPIPS 元数据，避免批次混用。")


if __name__ == "__main__":
    analyze_training_necessity()
