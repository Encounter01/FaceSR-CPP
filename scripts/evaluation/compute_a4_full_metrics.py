#!/usr/bin/env python3
"""Compute full validation PSNR/SSIM for the final A4 checkpoint.

The final checkpoint is a TorchScript archive without a callable forward method,
but it contains the generator state_dict. This script reconstructs the RRDBNet
architecture in PyTorch, loads that state_dict on CPU, and evaluates the same
3000-image CelebA validation split used in the thesis.
"""

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
VAL_HR_DIR = REPO_ROOT / "data" / "val" / "HR"
CHECKPOINT = REPO_ROOT / "checkpoints" / "final" / "facesr_a4_best_psnr28.6019.pt"
OUTPUT_DIR = REPO_ROOT / "results" / "revision" / "a4_best_full_val"


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, padding=1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, padding=1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, padding=1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, padding=1)
        self.beta = 0.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x2 = F.leaky_relu(self.conv2(torch.cat((x, x1), dim=1)), negative_slope=0.2)
        x3 = F.leaky_relu(self.conv3(torch.cat((x, x1, x2), dim=1)), negative_slope=0.2)
        x4 = F.leaky_relu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)), negative_slope=0.2)
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x + x5 * self.beta


class RRDB(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.beta = 0.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * self.beta


class RRDBNet(nn.Module):
    def __init__(self, num_feat: int = 64, num_block: int = 23, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(3, num_feat, 3, padding=1)
        self.body = nn.Sequential(*(RRDB(num_feat, num_grow_ch) for _ in range(num_block)))
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = F.interpolate(feat, scale_factor=2, mode="nearest")
        feat = F.leaky_relu(self.conv_up1(feat), negative_slope=0.2)
        feat = F.interpolate(feat, scale_factor=2, mode="nearest")
        feat = F.leaky_relu(self.conv_up2(feat), negative_slope=0.2)
        out = F.leaky_relu(self.conv_hr(feat), negative_slope=0.2)
        return self.conv_last(out)


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Unable to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def image_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    arr = rgb.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def compute_metrics(hr: np.ndarray, sr: np.ndarray) -> tuple[float, float]:
    hr_f = hr.astype(np.float32) / 255.0
    sr_f = sr.astype(np.float32) / 255.0
    psnr = peak_signal_noise_ratio(hr_f, sr_f, data_range=1.0)
    ssim = structural_similarity(hr_f, sr_f, channel_axis=2, data_range=1.0)
    return float(psnr), float(ssim)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = None
    if not args.metrics_only:
        archive = torch.jit.load(str(CHECKPOINT), map_location="cpu")
        state_dict = archive.state_dict()
        model = RRDBNet()
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Unexpected state_dict mismatch: missing={missing}, unexpected={unexpected}")
        model.eval()

    hr_files = sorted(VAL_HR_DIR.glob("*.png"))[:3000]
    if len(hr_files) != 3000:
        raise RuntimeError(f"Expected 3000 validation images, found {len(hr_files)}")

    rows: list[dict[str, float | str]] = []
    bicubic_psnr: list[float] = []
    bicubic_ssim: list[float] = []
    a4_psnr: list[float] = []
    a4_ssim: list[float] = []

    sr_dir = OUTPUT_DIR / "sr"

    with torch.inference_mode():
        for idx, hr_path in enumerate(hr_files, start=1):
            hr = read_rgb(hr_path)
            lr = cv2.resize(hr, (hr.shape[1] // 4, hr.shape[0] // 4), interpolation=cv2.INTER_CUBIC)
            bicubic = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
            b_psnr, b_ssim = compute_metrics(hr, bicubic)

            if args.metrics_only:
                sr_path = sr_dir / hr_path.name.replace(".png", "_sr.png")
                sr_np = read_rgb(sr_path)
            else:
                sr = model(image_to_tensor(lr)).clamp(0.0, 1.0)
                sr_np = (sr.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            p, s = compute_metrics(hr, sr_np)

            bicubic_psnr.append(b_psnr)
            bicubic_ssim.append(b_ssim)
            a4_psnr.append(p)
            a4_ssim.append(s)
            rows.append(
                {
                    "file": hr_path.name,
                    "bicubic_psnr": b_psnr,
                    "bicubic_ssim": b_ssim,
                    "a4_psnr": p,
                    "a4_ssim": s,
                }
            )

            if idx % 100 == 0:
                print(f"{idx}/3000 A4 PSNR {np.mean(a4_psnr):.4f} SSIM {np.mean(a4_ssim):.6f}")

    summary = {
        "count": len(rows),
        "checkpoint": str(CHECKPOINT.relative_to(REPO_ROOT)),
        "bicubic": {
            "psnr": float(np.mean(bicubic_psnr)),
            "ssim": float(np.mean(bicubic_ssim)),
        },
        "a4_best": {
            "psnr": float(np.mean(a4_psnr)),
            "ssim": float(np.mean(a4_ssim)),
        },
    }

    with (OUTPUT_DIR / "a4_best_full_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (OUTPUT_DIR / "a4_best_full_per_image_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nTHESIS TABLE 6.2 SUPPLEMENT")
    print(f"Bicubic | PSNR {summary['bicubic']['psnr']:.2f} | SSIM {summary['bicubic']['ssim']:.3f}")
    print(f"A4 best | PSNR {summary['a4_best']['psnr']:.2f} | SSIM {summary['a4_best']['ssim']:.3f}")


if __name__ == "__main__":
    main()
