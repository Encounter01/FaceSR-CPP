#!/usr/bin/env python3
"""
诊断脚本：调查 Best/Epoch190 PSNR 异常
=======================================
目标：
1. 检查 SR 图像像素值范围
2. 验证图像尺寸是否正确
3. 对比 Best/Latest 的像素分布
4. 找出问题根因
"""

import numpy as np
from pathlib import Path
from PIL import Image
import sys

def diagnose_sr_images():
    """诊断 SR 图像问题"""

    base = Path("tmp_eval")
    checkpoints = {
        "best": base / "val_best_3000",
        "latest": base / "val_latest_3000",
        "epoch190": base / "val_epoch190_3000",
    }

    print("=" * 80)
    print("SR 图像诊断报告")
    print("=" * 80)

    results = {}

    for ckpt_name, ckpt_dir in checkpoints.items():
        print(f"\n【{ckpt_name.upper()}】")
        print("-" * 80)

        sr_files = sorted(ckpt_dir.glob("*_sr.png"))

        if not sr_files:
            print(f"错误：未找到 _sr.png 文件！")
            continue

        print(f"文件数量: {len(sr_files)}")
        print(f"第一个文件: {sr_files[0].name}")
        print(f"最后一个文件: {sr_files[-1].name}")

        # 采样 10 张图像进行分析
        sample_indices = [0, len(sr_files)//4, len(sr_files)//2, 3*len(sr_files)//4, len(sr_files)-1]

        pixel_min_vals = []
        pixel_max_vals = []
        pixel_mean_vals = []
        dimensions = []

        for idx in sample_indices:
            try:
                img = Image.open(sr_files[idx])
                arr = np.array(img, dtype=np.float32)

                dimensions.append(arr.shape)
                pixel_min_vals.append(arr.min())
                pixel_max_vals.append(arr.max())
                pixel_mean_vals.append(arr.mean())

                if idx == 0:
                    print(f"  样本 1 尺寸: {arr.shape}")
                    print(f"  样本 1 像素范围: {arr.min():.1f} - {arr.max():.1f}")
                    print(f"  样本 1 数据类型: {arr.dtype}")

            except Exception as e:
                print(f"  文件读取错误 {sr_files[idx].name}: {e}")

        if pixel_min_vals:
            print(f"\n  采样 {len(sample_indices)} 张图像统计:")
            print(f"    最小像素值范围: {min(pixel_min_vals):.1f} - {max(pixel_min_vals):.1f}")
            print(f"    最大像素值范围: {min(pixel_max_vals):.1f} - {max(pixel_max_vals):.1f}")
            print(f"    平均像素值范围: {min(pixel_mean_vals):.1f} - {max(pixel_mean_vals):.1f}")

            # 判断像素值范围
            overall_max = max(pixel_max_vals)
            overall_min = min(pixel_min_vals)

            if overall_max <= 1.0 and overall_min >= 0:
                print(f"    像素值范围: [0.0, 1.0] (浮点型)")
                status = "VALID"
            elif overall_max <= 255 and overall_min >= 0:
                print(f"    像素值范围: [0, 255] (uint8)")
                status = "VALID"
            else:
                print(f"    像素值范围: [{overall_min:.1f}, {overall_max:.1f}] 异常!")
                status = "ANOMALY"

            results[ckpt_name] = {
                'pixel_max': max(pixel_max_vals),
                'pixel_min': min(pixel_min_vals),
                'status': status,
                'dimensions': list(set(dimensions))
            }

    # ===== 对比分析 =====
    print("\n" + "=" * 80)
    print("对比分析：Best vs Latest")
    print("=" * 80)

    if "best" in results and "latest" in results:
        best_stat = results["best"]
        latest_stat = results["latest"]

        print(f"\n最大像素值:")
        print(f"  Best:   {best_stat['pixel_max']:.1f}")
        print(f"  Latest: {latest_stat['pixel_max']:.1f}")

        print(f"\nBest 像素值范围异常?")
        if best_stat['pixel_max'] > 1.5 or best_stat['pixel_min'] < 0:
            print(f"  >>> 是的！像素值异常 [{best_stat['pixel_min']:.1f}, {best_stat['pixel_max']:.1f}]")
        else:
            print(f"  >>> 否，像素值正常")

        print(f"\nBest 图像尺寸:")
        print(f"  {best_stat['dimensions']}")
        print(f"\nLatest 图像尺寸:")
        print(f"  {latest_stat['dimensions']}")

    # ===== 直接计算几张图像的 PSNR 进行验证 =====
    print("\n" + "=" * 80)
    print("PSNR 直接计算验证")
    print("=" * 80)

    try:
        from skimage.metrics import peak_signal_noise_ratio
        import cv2

        hr_dir = Path("data/test/HR")
        hr_files = sorted(hr_dir.glob("celeba_*.png"))[:3]

        for hr_file in hr_files:
            hr_img = cv2.imread(str(hr_file))
            if hr_img is None:
                continue

            hr_rgb = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
            hr_float = hr_rgb.astype(np.float32) / 255.0

            img_id = hr_file.stem

            print(f"\n{img_id}:")
            print(f"  HR 像素范围: [{hr_float.min():.3f}, {hr_float.max():.3f}]")

            for ckpt_name in ["best", "latest"]:
                sr_path = checkpoints[ckpt_name] / f"{img_id}_sr.png"
                if sr_path.exists():
                    sr_img = cv2.imread(str(sr_path))
                    sr_rgb = cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB)
                    sr_float = sr_rgb.astype(np.float32) / 255.0

                    psnr = peak_signal_noise_ratio(hr_float, sr_float, data_range=1.0)
                    print(f"  {ckpt_name:8s} - PSNR: {psnr:.2f} dB | 像素范围: [{sr_float.min():.3f}, {sr_float.max():.3f}]")

    except Exception as e:
        print(f"PSNR 验证失败: {e}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    diagnose_sr_images()
