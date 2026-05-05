#!/usr/bin/env python3
"""Generate a current model-selection guide from checkpoint_comparison_summary.csv.

Historical versions of this script generated reports based on an invalid evaluation batch.
The current script intentionally uses only the latest PSNR/SSIM summary CSV.
"""

from csv import DictReader
from datetime import datetime
from pathlib import Path


SUMMARY_CSV = Path("results/eval_reports/checkpoint_comparison_summary.csv")
OUTPUT_PATH = Path("results/eval_reports/MODEL_SELECTION_GUIDE.md")


def _load_rows():
    with SUMMARY_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["checkpoint"]: row for row in DictReader(f)}


def _f(value):
    return f"{float(value):.6f}"


def generate_model_selection_guide():
    rows = _load_rows()
    epoch190 = rows["epoch190"]
    best = rows["best"]
    latest = rows["latest"]
    bicubic = rows["bicubic"]

    guide = f"""# FaceSR_CPP Model Selection Guide

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Recommendation

Use `checkpoints/generator_epoch190.pt` first for current inference demos and thesis metrics.

## Current Evaluation Source

- Source file: `{SUMMARY_CSV.as_posix()}`
- Test images: 3000
- Official current metrics: PSNR / SSIM
- LPIPS, stratified analysis, and failure rate must be recomputed from the same checkpoint outputs before citation.

## Performance Comparison

| Checkpoint | PSNR | PSNR Gain | SSIM | SSIM Gain | Status |
|------------|-----:|----------:|-----:|----------:|--------|
| Bicubic | {_f(bicubic['avg_psnr'])} | - | {_f(bicubic['avg_ssim'])} | - | Baseline |
| generator_epoch190.pt | {_f(epoch190['avg_psnr'])} | {_f(epoch190['avg_psnr_gain_vs_bicubic'])} | {_f(epoch190['avg_ssim'])} | {_f(epoch190['avg_ssim_gain_vs_bicubic'])} | Recommended |
| generator_best.pt | {_f(best['avg_psnr'])} | {_f(best['avg_psnr_gain_vs_bicubic'])} | {_f(best['avg_ssim'])} | {_f(best['avg_ssim_gain_vs_bicubic'])} | Backup / comparison |
| generator_latest.pt | {_f(latest['avg_psnr'])} | {_f(latest['avg_psnr_gain_vs_bicubic'])} | {_f(latest['avg_ssim'])} | {_f(latest['avg_ssim_gain_vs_bicubic'])} | Backup / comparison |

## Notes

- Do not cite historical reports that marked best or epoch190 as abnormal.
- Do not mix old LPIPS or failure-rate tables with this PSNR/SSIM summary.
- Re-run `scripts/evaluation/final_recompute_eval.py` after replacing any checkpoint.
"""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(guide, encoding="utf-8")
    print(f"[OK] Model selection guide saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_model_selection_guide()
