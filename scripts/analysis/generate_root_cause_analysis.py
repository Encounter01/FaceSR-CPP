#!/usr/bin/env python3
"""Generate the current explanation for historical evaluation inconsistency."""

from csv import DictReader
from datetime import datetime
from pathlib import Path


SUMMARY_CSV = Path("results/eval_reports/checkpoint_comparison_summary.csv")
OUTPUT_PATH = Path("results/eval_reports/ROOT_CAUSE_ANALYSIS.md")


def _load_rows():
    with SUMMARY_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["checkpoint"]: row for row in DictReader(f)}


def _fmt(value):
    return f"{float(value):.6f}"


def generate_root_cause_analysis():
    rows = _load_rows()
    epoch190 = rows["epoch190"]
    best = rows["best"]
    latest = rows["latest"]

    report = f"""# FaceSR_CPP 历史评估不一致说明

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 当前结论

当前可核对的结果来自 `{SUMMARY_CSV.as_posix()}`。在 3000 张 CelebA 测试图像上：

| Checkpoint | PSNR | SSIM |
|------------|-----:|-----:|
| generator_epoch190.pt | {_fmt(epoch190['avg_psnr'])} | {_fmt(epoch190['avg_ssim'])} |
| generator_best.pt | {_fmt(best['avg_psnr'])} | {_fmt(best['avg_ssim'])} |
| generator_latest.pt | {_fmt(latest['avg_psnr'])} | {_fmt(latest['avg_ssim'])} |

当前 PSNR/SSIM 口径下，`generator_epoch190.pt` 是推荐优先使用的推理权重。

## 为什么旧结论失效

旧版评估报告和后续复评使用了不同批次的中间文件或不同评估链路，不能与当前 CSV 混合引用。
旧报告中的 checkpoint 异常判断已经被当前复评结果推翻，因此不再作为论文、答辩或模型选择依据。

## 后续要求

- 每次替换 checkpoint 后重新生成同批次评估结果。
- 报告中同时记录 checkpoint 文件名、评估脚本参数、评估日期和输出目录。
- LPIPS、分层分析和失败率必须与当前 checkpoint 的 SR 输出来自同一次评估。
- 不再把已清理的历史报告作为正式证据。
"""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"[OK] Current inconsistency note generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_root_cause_analysis()
