#!/usr/bin/env python3
"""Compute same-batch LPIPS for the final A4 revision outputs.

The script uses the already generated revision outputs:
- HR: data/test/HR/*.png
- LR: results/revision/a4_best_full_val/lr/*.png
- A4 SR: results/revision/a4_best_full_val/sr/*_sr.png

It recomputes Bicubic from the saved LR files and compares both Bicubic and A4
against the same HR images, so the reported LPIPS values are batch-consistent.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
REVISION_DIR = REPO_ROOT / "results" / "revision" / "a4_best_full_val"
HR_DIR = REPO_ROOT / "data" / "test" / "HR"
LR_DIR = REVISION_DIR / "lr"
A4_SR_DIR = REVISION_DIR / "sr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Bicubic/A4 LPIPS on revision outputs")
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def choose_device(raw: str) -> torch.device:
    if raw == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available")
        return torch.device("cuda")
    if raw == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_lpips_tensor(images: list[np.ndarray], device: torch.device) -> torch.Tensor:
    arr = np.stack(images, axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)
    return tensor.mul(2.0).sub(1.0).to(device)


def summarize(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    for path in (HR_DIR, LR_DIR, A4_SR_DIR):
        if not path.exists():
            raise SystemExit(f"Missing directory: {path}")

    hr_files = sorted(HR_DIR.glob("*.png"))[: args.limit]
    if len(hr_files) != args.limit:
        raise SystemExit(f"Requested {args.limit} images, found {len(hr_files)}")

    metric = lpips.LPIPS(net="alex", verbose=False).to(device)
    metric.eval()

    rows: list[dict[str, str | float]] = []
    bicubic_values: list[float] = []
    a4_values: list[float] = []

    with torch.no_grad():
        for start in range(0, len(hr_files), args.batch_size):
            batch_files = hr_files[start : start + args.batch_size]
            hr_images: list[np.ndarray] = []
            bicubic_images: list[np.ndarray] = []
            a4_images: list[np.ndarray] = []

            for hr_path in batch_files:
                lr_path = LR_DIR / hr_path.name
                a4_path = A4_SR_DIR / hr_path.name.replace(".png", "_sr.png")
                if not lr_path.exists():
                    raise RuntimeError(f"Missing LR image: {lr_path}")
                if not a4_path.exists():
                    raise RuntimeError(f"Missing A4 SR image: {a4_path}")

                hr = read_rgb(hr_path)
                lr = read_rgb(lr_path)
                a4 = read_rgb(a4_path)
                if a4.shape[:2] != hr.shape[:2]:
                    a4 = cv2.resize(a4, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
                bicubic = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)

                hr_images.append(hr)
                bicubic_images.append(bicubic)
                a4_images.append(a4)

            hr_t = to_lpips_tensor(hr_images, device)
            bicubic_t = to_lpips_tensor(bicubic_images, device)
            a4_t = to_lpips_tensor(a4_images, device)

            bicubic_batch = metric(bicubic_t, hr_t).view(-1).detach().cpu().numpy()
            a4_batch = metric(a4_t, hr_t).view(-1).detach().cpu().numpy()

            for hr_path, bicubic_lpips, a4_lpips in zip(batch_files, bicubic_batch, a4_batch):
                bicubic_value = float(bicubic_lpips)
                a4_value = float(a4_lpips)
                bicubic_values.append(bicubic_value)
                a4_values.append(a4_value)
                rows.append(
                    {
                        "file": hr_path.name,
                        "bicubic_lpips": bicubic_value,
                        "a4_lpips": a4_value,
                    }
                )

            if (start // args.batch_size) % 20 == 0:
                print(f"processed {min(start + args.batch_size, len(hr_files))}/{len(hr_files)}")

    summary = {
        "device": str(device),
        "net": "alex",
        "source": {
            "hr_dir": str(HR_DIR.relative_to(REPO_ROOT)),
            "lr_dir": str(LR_DIR.relative_to(REPO_ROOT)),
            "a4_sr_dir": str(A4_SR_DIR.relative_to(REPO_ROOT)),
        },
        "bicubic": summarize(bicubic_values),
        "a4_best": summarize(a4_values),
    }

    csv_path = REVISION_DIR / "a4_bicubic_lpips_per_image.csv"
    json_path = REVISION_DIR / "a4_bicubic_lpips_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "bicubic_lpips", "a4_lpips"])
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {csv_path}")
    print(f"saved: {json_path}")


if __name__ == "__main__":
    main()
