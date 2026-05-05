#!/usr/bin/env python3
"""Generate a current PSNR/SSIM summary from checkpoint_comparison_summary.csv."""

from csv import DictReader
from datetime import datetime
from pathlib import Path


SUMMARY_CSV = Path("results/eval_reports/checkpoint_comparison_summary.csv")
OUTPUT_PATH = Path("results/eval_reports/FINAL_SUMMARY.md")


def _load_rows():
    with SUMMARY_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        return list(DictReader(f))


def _fmt(value, digits=6):
    if value in ("", None):
        return "-"
    return f"{float(value):.{digits}f}"


def generate_summary():
    rows = _load_rows()
    by_name = {row["checkpoint"]: row for row in rows}
    recommended = by_name["epoch190"]

    lines = [
        "# FaceSR_CPP 当前评估总结",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 当前可信口径",
        "",
        f"- 数据来源：`{SUMMARY_CSV.as_posix()}`",
        "- 测试集：3000 张 CelebA 测试图像",
        "- 当前正式引用指标：PSNR / SSIM",
        "- LPIPS、分层分析和失败率需使用当前 checkpoint 同批次重新评估后再引用",
        "",
        "## Checkpoint 对比",
        "",
        "| Checkpoint | PSNR | PSNR Gain | SSIM | SSIM Gain | Count |",
        "|------------|-----:|----------:|-----:|----------:|------:|",
    ]

    order = ["bicubic", "best", "latest", "epoch190"]
    for key in order:
        row = by_name[key]
        label = "generator_epoch190.pt" if key == "epoch190" else (
            f"generator_{key}.pt" if key in {"best", "latest"} else "Bicubic"
        )
        lines.append(
            f"| {label} | {_fmt(row['avg_psnr'])} | {_fmt(row['avg_psnr_gain_vs_bicubic'])} | "
            f"{_fmt(row['avg_ssim'])} | {_fmt(row['avg_ssim_gain_vs_bicubic'])} | {row['count']} |"
        )

    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 当前 PSNR/SSIM 口径下，推荐优先使用 `checkpoints/generator_epoch190.pt`。",
            f"- Epoch190 PSNR/SSIM 为 `{_fmt(recommended['avg_psnr'])}` / `{_fmt(recommended['avg_ssim'])}`。",
            "- `generator_best.pt` 和 `generator_latest.pt` 仍可用于对比或回退，但不是当前 PSNR/SSIM 最优项。",
            "- 已清理的旧 LPIPS、失败率和异常根因报告不再作为论文或答辩依据。",
            "",
        ]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Current summary generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_summary()
