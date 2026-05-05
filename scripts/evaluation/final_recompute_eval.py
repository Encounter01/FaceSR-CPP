#!/usr/bin/env python3
"""Recompute final FaceSR metrics in an isolated output directory.

This script intentionally does not reuse historical CSV files. It creates a
fresh LR set, runs the C++ inference binary for selected checkpoints, computes
PSNR/SSIM from the generated SR files, and writes one authoritative report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


REPO_ROOT = Path(__file__).resolve().parents[2]
HR_DIR = REPO_ROOT / "data" / "test" / "HR"
BINARY = REPO_ROOT / "build" / "bin" / "Release" / "facesr_test.exe"
CHECKPOINTS = {
    "best": REPO_ROOT / "checkpoints" / "generator_best.pt",
    "latest": REPO_ROOT / "checkpoints" / "generator_latest.pt",
    "epoch190": REPO_ROOT / "checkpoints" / "generator_epoch190.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fresh final FaceSR evaluation")
    parser.add_argument("--n-images", type=int, default=3000)
    parser.add_argument(
        "--checkpoints",
        default="best,latest,epoch190",
        help=(
            "Comma-separated checkpoint keys or label=path specs. "
            "Built-ins: best,latest,epoch190"
        ),
    )
    parser.add_argument(
        "--attention-checkpoints",
        default="",
        help="Comma-separated checkpoint labels that must be loaded with --attention",
    )
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-inference", action="store_true")
    return parser.parse_args()


def parse_label_list(labels: str) -> set[str]:
    return {label.strip() for label in labels.split(",") if label.strip()}


def parse_checkpoint_specs(spec: str) -> dict[str, Path]:
    checkpoints: dict[str, Path] = {}
    unknown: list[str] = []
    for item in [part.strip() for part in spec.split(",") if part.strip()]:
        if "=" in item:
            label, raw_path = [part.strip() for part in item.split("=", 1)]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
                raise SystemExit(
                    f"Invalid checkpoint label '{label}'. Use letters, numbers, '_' or '-'."
                )
            path = Path(raw_path)
            checkpoints[label] = path if path.is_absolute() else (REPO_ROOT / path).resolve()
        elif item in CHECKPOINTS:
            checkpoints[item] = CHECKPOINTS[item]
        else:
            unknown.append(item)
    if unknown:
        raise SystemExit(f"Unknown checkpoint keys: {unknown}")
    if not checkpoints:
        raise SystemExit("No checkpoints selected")
    return checkpoints


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Unable to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"Unable to write image: {path}")


def generate_lr_and_bicubic(hr_files: list[Path], lr_dir: Path, bicubic_dir: Path) -> None:
    lr_dir.mkdir(parents=True, exist_ok=True)
    bicubic_dir.mkdir(parents=True, exist_ok=True)
    for hr_path in hr_files:
        hr = read_rgb(hr_path)
        lr = cv2.resize(hr, (64, 64), interpolation=cv2.INTER_CUBIC)
        up = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
        write_rgb(lr_dir / hr_path.name, lr)
        write_rgb(bicubic_dir / hr_path.name.replace(".png", "_sr.png"), up)


def run_inference(
    key: str,
    ckpt: Path,
    lr_dir: Path,
    out_dir: Path,
    device: str,
    use_attention: bool,
) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BINARY),
        "--model",
        str(ckpt),
        "--input",
        str(lr_dir),
        "--output",
        str(out_dir),
    ]
    if device == "cpu":
        cmd.append("--cpu")
    if use_attention:
        cmd.append("--attention")
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
    elapsed = time.perf_counter() - start
    (out_dir / "_inference_stdout.txt").write_text(proc.stdout, encoding="utf-8", errors="replace")
    (out_dir / "_inference_stderr.txt").write_text(proc.stderr, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"Inference failed for {key}: return code {proc.returncode}\n{proc.stderr}")
    return elapsed


def compute_one(hr_path: Path, sr_path: Path) -> tuple[float, float]:
    hr = read_rgb(hr_path)
    sr = read_rgb(sr_path)
    if sr.shape != hr.shape:
        sr = cv2.resize(sr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
    hr_f = hr.astype(np.float32) / 255.0
    sr_f = sr.astype(np.float32) / 255.0
    psnr = peak_signal_noise_ratio(hr_f, sr_f, data_range=1.0)
    ssim = structural_similarity(hr_f, sr_f, channel_axis=2, data_range=1.0)
    return float(psnr), float(ssim)


def summarize(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    args = parse_args()
    selected_checkpoints = parse_checkpoint_specs(args.checkpoints)
    attention_checkpoints = parse_label_list(args.attention_checkpoints)
    keys = list(selected_checkpoints.keys())
    unknown_attention = sorted(attention_checkpoints.difference(selected_checkpoints.keys()))
    if unknown_attention:
        raise SystemExit(f"Attention labels not selected: {unknown_attention}")
    if not BINARY.exists():
        raise SystemExit(f"Missing inference binary: {BINARY}")
    if not HR_DIR.exists():
        raise SystemExit(f"Missing HR directory: {HR_DIR}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (REPO_ROOT / "results" / f"final_eval_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    hr_files = sorted(HR_DIR.glob("*.png"))[: args.n_images]
    if len(hr_files) != args.n_images:
        raise SystemExit(f"Requested {args.n_images} images, found {len(hr_files)}")

    lr_dir = output_dir / "lr"
    bicubic_dir = output_dir / "sr_bicubic"
    generate_lr_and_bicubic(hr_files, lr_dir, bicubic_dir)

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "n_images": args.n_images,
        "device": args.device,
        "binary": str(BINARY),
        "binary_sha256": file_sha256(BINARY),
        "checkpoints": {},
        "inference_seconds": {},
    }

    sr_dirs = {"bicubic": bicubic_dir}
    for key in keys:
        ckpt = selected_checkpoints[key]
        use_attention = key in attention_checkpoints
        if not ckpt.exists():
            raise SystemExit(f"Missing checkpoint {key}: {ckpt}")
        metadata["checkpoints"][key] = {
            "path": str(ckpt),
            "use_attention": use_attention,
            "sha256": file_sha256(ckpt),
            "size_bytes": ckpt.stat().st_size,
            "mtime": datetime.fromtimestamp(ckpt.stat().st_mtime).isoformat(timespec="seconds"),
        }
        out_dir = output_dir / f"sr_{key}"
        sr_dirs[key] = out_dir
        if not args.skip_inference:
            metadata["inference_seconds"][key] = run_inference(
                key, ckpt, lr_dir, out_dir, args.device, use_attention
            )

    rows: list[dict[str, str | float]] = []
    summary_rows: list[dict[str, str | float | int]] = []
    all_keys = ["bicubic"] + keys
    metric_values: dict[str, dict[str, list[float]]] = {
        key: {"psnr": [], "ssim": []} for key in all_keys
    }

    for hr_path in hr_files:
        row: dict[str, str | float] = {"file": hr_path.name}
        for key in all_keys:
            sr_path = sr_dirs[key] / hr_path.name.replace(".png", "_sr.png")
            if not sr_path.exists():
                row[f"{key}_missing"] = 1
                row[f"{key}_psnr"] = math.nan
                row[f"{key}_ssim"] = math.nan
                continue
            psnr, ssim = compute_one(hr_path, sr_path)
            row[f"{key}_psnr"] = psnr
            row[f"{key}_ssim"] = ssim
            metric_values[key]["psnr"].append(psnr)
            metric_values[key]["ssim"].append(ssim)
        rows.append(row)

    per_image_csv = output_dir / "per_image_metrics.csv"
    with per_image_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    bicubic_psnr = np.asarray(metric_values["bicubic"]["psnr"], dtype=np.float64)
    bicubic_ssim = np.asarray(metric_values["bicubic"]["ssim"], dtype=np.float64)
    for key in all_keys:
        psnr_stats = summarize(metric_values[key]["psnr"])
        ssim_stats = summarize(metric_values[key]["ssim"])
        row = {
            "checkpoint": key,
            "count": psnr_stats["count"],
            "avg_psnr": psnr_stats["mean"],
            "std_psnr": psnr_stats["std"],
            "median_psnr": psnr_stats["median"],
            "min_psnr": psnr_stats["min"],
            "max_psnr": psnr_stats["max"],
            "avg_ssim": ssim_stats["mean"],
            "std_ssim": ssim_stats["std"],
            "median_ssim": ssim_stats["median"],
            "min_ssim": ssim_stats["min"],
            "max_ssim": ssim_stats["max"],
        }
        if key != "bicubic":
            psnr_arr = np.asarray(metric_values[key]["psnr"], dtype=np.float64)
            ssim_arr = np.asarray(metric_values[key]["ssim"], dtype=np.float64)
            row["avg_psnr_gain_vs_bicubic"] = float(np.mean(psnr_arr - bicubic_psnr))
            row["avg_ssim_gain_vs_bicubic"] = float(np.mean(ssim_arr - bicubic_ssim))
            row["negative_psnr_gain_count"] = int(np.sum(psnr_arr < bicubic_psnr))
            row["negative_ssim_gain_count"] = int(np.sum(ssim_arr < bicubic_ssim))
        else:
            row["avg_psnr_gain_vs_bicubic"] = ""
            row["avg_ssim_gain_vs_bicubic"] = ""
            row["negative_psnr_gain_count"] = ""
            row["negative_ssim_gain_count"] = ""
        summary_rows.append(row)

    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_md = output_dir / "summary.md"
    lines = ["# Final FaceSR Evaluation", ""]
    lines.append(f"- Generated at: `{metadata['generated_at']}`")
    lines.append(f"- Images: `{args.n_images}`")
    lines.append(f"- Device: `{args.device}`")
    lines.append("")
    lines.append("| Checkpoint | Count | PSNR mean | SSIM mean | PSNR gain vs Bicubic | SSIM gain vs Bicubic | Negative PSNR gain |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            "| {checkpoint} | {count} | {avg_psnr:.6f} | {avg_ssim:.6f} | {psnr_gain} | {ssim_gain} | {neg} |".format(
                checkpoint=row["checkpoint"],
                count=row["count"],
                avg_psnr=row["avg_psnr"],
                avg_ssim=row["avg_ssim"],
                psnr_gain=(
                    ""
                    if row["avg_psnr_gain_vs_bicubic"] == ""
                    else f"{row['avg_psnr_gain_vs_bicubic']:.6f}"
                ),
                ssim_gain=(
                    ""
                    if row["avg_ssim_gain_vs_bicubic"] == ""
                    else f"{row['avg_ssim_gain_vs_bicubic']:.6f}"
                ),
                neg=row["negative_psnr_gain_count"],
            )
        )
    lines.append("")
    lines.append("This directory is self-contained and does not reuse historical evaluation CSV files.")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] Final evaluation written to: {output_dir}")
    print(summary_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
