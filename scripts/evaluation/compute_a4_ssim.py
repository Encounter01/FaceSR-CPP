#!/usr/bin/env python3
"""Compute PSNR and SSIM for A4 formal checkpoint on the 3000-image validation set.

Usage (from repo root):
    python scripts/evaluation/compute_a4_ssim.py
    python scripts/evaluation/compute_a4_ssim.py --n-images 100  # quick test
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

REPO_ROOT = Path(__file__).resolve().parents[2]
VAL_HR_DIR = REPO_ROOT / "data" / "val" / "hr"
A4_CHECKPOINT = REPO_ROOT / "checkpoints" / "final" / "facesr_a4_best_psnr28.6019.pt"
BINARY = REPO_ROOT / "build" / "bin" / "Release" / "facesr_test.exe"
DEFAULT_WORK_DIR = REPO_ROOT / "results" / "revision" / "a4_best_full_val"


def bicubic_sr(lr_img: np.ndarray, scale: int = 4) -> np.ndarray:
    h, w = lr_img.shape[:2]
    return cv2.resize(lr_img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def run_inference(checkpoint: Path, input_dir: Path, output_dir: Path) -> None:
    cmd = [
        str(BINARY),
        "--model", str(checkpoint),
        "--input", str(input_dir),
        "--output", str(output_dir),
    ]
    print(f"Running inference: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Inference failed:\n{result.stderr}")


def evaluate(hr_dir: Path, sr_dir: Path, n_images: int) -> dict:
    hr_files = sorted(hr_dir.glob("*.png"))[:n_images]
    psnr_list, ssim_list = [], []

    for hr_path in hr_files:
        sr_path = sr_dir / hr_path.name
        if not sr_path.exists():
            print(f"  WARNING: missing SR for {hr_path.name}")
            continue

        hr = cv2.imread(str(hr_path))
        sr = cv2.imread(str(sr_path))
        if hr is None or sr is None:
            continue
        if sr.shape != hr.shape:
            sr = cv2.resize(sr, (hr.shape[1], hr.shape[0]))

        psnr = peak_signal_noise_ratio(hr, sr, data_range=255)
        ssim = structural_similarity(hr, sr, channel_axis=2, data_range=255)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    return {
        "count": len(psnr_list),
        "avg_psnr": float(np.mean(psnr_list)),
        "avg_ssim": float(np.mean(ssim_list)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=3000)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--keep-work-dir", action="store_true")
    args = parser.parse_args()

    if not VAL_HR_DIR.exists():
        raise FileNotFoundError(f"Validation HR directory not found: {VAL_HR_DIR}")
    if not A4_CHECKPOINT.exists():
        raise FileNotFoundError(f"A4 checkpoint not found: {A4_CHECKPOINT}")
    if not BINARY.exists():
        raise FileNotFoundError(f"Inference binary not found: {BINARY}")

    hr_files = sorted(VAL_HR_DIR.glob("*.png"))[: args.n_images]
    print(f"Evaluating {len(hr_files)} images from {VAL_HR_DIR}")

    work_dir = args.work_dir
    if work_dir.exists() and not args.keep_work_dir:
        shutil.rmtree(work_dir)

    lr_dir = work_dir / "lr"
    sr_dir = work_dir / "sr"
    lr_dir.mkdir(parents=True, exist_ok=True)
    sr_dir.mkdir(parents=True, exist_ok=True)

    # Generate LR images via Bicubic downsampling
    print("Generating LR images...")
    bicubic_psnr, bicubic_ssim = [], []
    for hr_path in hr_files:
        hr = cv2.imread(str(hr_path))
        lr = cv2.resize(hr, (hr.shape[1] // 4, hr.shape[0] // 4),
                        interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(lr_dir / hr_path.name), lr)

        bic = bicubic_sr(lr, scale=4)
        bicubic_psnr.append(peak_signal_noise_ratio(hr, bic, data_range=255))
        bicubic_ssim.append(structural_similarity(hr, bic, channel_axis=2, data_range=255))

    print(f"\nBicubic baseline ({len(bicubic_psnr)} images):")
    print(f"  PSNR: {np.mean(bicubic_psnr):.4f} dB")
    print(f"  SSIM: {np.mean(bicubic_ssim):.6f}")

    # Run A4 inference
    run_inference(A4_CHECKPOINT, lr_dir, sr_dir)

    # Evaluate A4
    a4_metrics = evaluate(VAL_HR_DIR, sr_dir, args.n_images)
    print(f"\nA4 formal checkpoint ({a4_metrics['count']} images):")
    print(f"  PSNR: {a4_metrics['avg_psnr']:.4f} dB")
    print(f"  SSIM: {a4_metrics['avg_ssim']:.6f}")

    # Summary for thesis table
    print("\n" + "=" * 50)
    print("THESIS TABLE 6.2 SUPPLEMENT:")
    print(f"  Bicubic  | PSNR {np.mean(bicubic_psnr):.2f} | SSIM {np.mean(bicubic_ssim):.3f}")
    print(f"  A4 best  | PSNR {a4_metrics['avg_psnr']:.2f} | SSIM {a4_metrics['avg_ssim']:.3f}")


if __name__ == "__main__":
    main()
