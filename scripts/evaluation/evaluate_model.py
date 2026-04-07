#!/usr/bin/env python3
"""
FaceSR_CPP 综合性能评估脚本

功能：
  1. LR 图像生成（从 HR 通过 bicubic 缩放）
  2. SR 图像生成（调用 C++ 推理二进制）
  3. Bicubic 基线（LR 放大为 HR 尺寸）
  4. 指标计算：PSNR, SSIM, LPIPS
  5. 分层分析（按亮度、边缘密度分桶）
  6. 可视化：对比面板 + 柱状图
  7. 输出报告（CSV、Markdown）
"""

import sys
import argparse
import subprocess
import inspect
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Tuple, List
import importlib.util
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import cv2

# ===============================================================================
# Step 0: 模块级常量
# ===============================================================================

PROJECT_ROOT = Path(__file__).parent.resolve()
HR_DIR = PROJECT_ROOT / "data/test/HR"
LR_DIR = PROJECT_ROOT / "data/test/LR"
CKPT_DIR = PROJECT_ROOT / "checkpoints"

BINARY_CANDIDATES = [
    PROJECT_ROOT / "build/bin/Release/facesr_test.exe",
    PROJECT_ROOT / "build/Release/facesr_test.exe",
]

EXISTING_METRICS_CSV = PROJECT_ROOT / "results/eval_reports/per_image_checkpoint_metrics.csv"

CHECKPOINT_MAP = {
    "best": CKPT_DIR / "generator_best.pt",
    "latest": CKPT_DIR / "generator_latest.pt",
    "epoch190": CKPT_DIR / "generator_epoch190.pt",
}

# 已存在的 SR 输出目录，跳过重新生成
EXISTING_SR_DIRS = {
    "best": PROJECT_ROOT / "tmp_eval/val_best_3000",
    "epoch190": PROJECT_ROOT / "tmp_eval/val_epoch190_3000",
}

# 亮度分桶阈值（[0,1] 归一化值）
BRIGHTNESS_BINS = {
    "dark": (0, 80 / 255),
    "medium": (80 / 255, 170 / 255),
    "bright": (170 / 255, 1.0),
}

# 边缘密度分桶
EDGE_DENSITY_BINS = {
    "low": (0, 5),           # < 5% 的像素是边缘
    "medium": (5, 15),       # 5-15% 的像素是边缘
    "high": (15, 100),       # > 15% 的像素是边缘
}

# ===============================================================================
# Step 1: 依赖检查
# ===============================================================================

def check_dependencies():
    """检查必要的 Python 依赖"""
    required_packages = [
        "lpips",
        "skimage",
        "cv2",
        "pandas",
        "matplotlib",
        "tqdm",
        "torch",
    ]
    missing = []
    for pkg in required_packages:
        # 映射某些包名
        check_name = {"cv2": "opencv-python", "skimage": "scikit-image"}.get(pkg, pkg)
        if importlib.util.find_spec(pkg) is None:
            missing.append(check_name)

    if missing:
        print("[ERROR] 缺失以下依赖包:")
        for pkg in missing:
            print(f"  pip install {pkg}")
        sys.exit(1)


check_dependencies()

import torch
import lpips as lpips_lib
from skimage import metrics as skimage_metrics
import matplotlib.pyplot as plt


# ===============================================================================
# Step 2: CLI 参数解析
# ===============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="FaceSR_CPP 综合性能评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        choices=["best", "latest", "epoch190", "all"],
        default="all",
        help="要评估的 checkpoint（默认评估所有）",
    )
    parser.add_argument(
        "--n_images",
        type=int,
        default=3000,
        help="评估的图像数量（默认 3000）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "results/eval_reports"),
        help="输出目录（默认 results/eval_reports/）",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu"],
        default="auto",
        help="计算设备（默认 auto）",
    )
    parser.add_argument(
        "--force_regen_lr",
        action="store_true",
        help="强制重新生成 LR 图像",
    )
    parser.add_argument(
        "--force_regen_sr",
        action="store_true",
        help="强制重新生成 SR 图像",
    )
    parser.add_argument(
        "--force_recompute_metrics",
        action="store_true",
        help="强制重新计算所有指标",
    )
    parser.add_argument(
        "--skip_panels",
        action="store_true",
        help="跳过生成对比面板（加快速度）",
    )
    parser.add_argument(
        "--skip_summary",
        action="store_true",
        help="跳过生成汇总图表",
    )

    return parser.parse_args()


# ===============================================================================
# Step 3: LR 图像生成
# ===============================================================================

def generate_lr_images(n_images: int, force_regen: bool = False):
    """
    从 HR 图像生成 LR 图像（64x64 bicubic 缩放）

    参数：
      n_images: 生成的图像数量
      force_regen: 是否强制重新生成
    """
    LR_DIR.mkdir(parents=True, exist_ok=True)

    # 列出所有 HR 图像
    hr_files = sorted(HR_DIR.glob("celeba_*.png"))[:n_images]

    if not force_regen and len(list(LR_DIR.glob("*.png"))) >= n_images:
        print(f"[INFO] LR 目录已存在 >= {n_images} 张图像，跳过重新生成")
        return

    print(f"[INFO] 生成 {len(hr_files)} 张 LR 图像...")

    def process_image(hr_path: Path) -> Tuple[str, bool]:
        """处理单张图像"""
        lr_path = LR_DIR / hr_path.name
        try:
            img = Image.open(hr_path).convert("RGB")
            lr_img = img.resize((64, 64), Image.BICUBIC)
            lr_img.save(lr_path)
            return hr_path.name, True
        except Exception as e:
            warnings.warn(f"处理 {hr_path.name} 失败: {e}")
            return hr_path.name, False

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_image, p) for p in hr_files]
        for _ in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="生成 LR 图像",
            dynamic_ncols=True,
        ):
            pass

    actual_count = len(list(LR_DIR.glob("*.png")))
    print(f"[INFO] 生成完成：{actual_count} 张 LR 图像")


# ===============================================================================
# Step 4: SR 图像生成
# ===============================================================================

def find_binary() -> Path:
    """查找 C++ 推理二进制"""
    for candidate in BINARY_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"未找到 facesr_test.exe，尝试过的路径：{BINARY_CANDIDATES}"
    )


def generate_sr_images(
    ckpt_key: str, n_images: int, device: str, force_regen: bool = False
) -> Path:
    """
    生成 SR 图像

    幂等逻辑（优先级从高到低）：
    1. 目标目录已有 >= n_images 张 *_sr.png → 直接返回
    2. EXISTING_SR_DIRS 中预存目录满足数量 → 返回该目录（复用）
    3. 否则调用 C++ 二进制

    返回：SR 输出目录
    """
    output_dir = PROJECT_ROOT / "tmp_eval" / f"val_{ckpt_key}_{n_images}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查 1: 目标目录是否已有足够图像
    existing_sr = list(output_dir.glob("*_sr.png"))
    if not force_regen and len(existing_sr) >= n_images:
        print(f"[INFO] {ckpt_key} SR 目录已存在 >= {n_images} 张图像，跳过重新生成")
        return output_dir

    # 检查 2: 预存目录是否可复用
    if ckpt_key in EXISTING_SR_DIRS and not force_regen:
        prebuilt_dir = EXISTING_SR_DIRS[ckpt_key]
        if prebuilt_dir.exists():
            prebuilt_count = len(list(prebuilt_dir.glob("*_sr.png")))
            if prebuilt_count >= n_images:
                print(f"[INFO] 复用预存 {ckpt_key} SR 目录：{prebuilt_dir}")
                return prebuilt_dir

    # 检查 3: 生成 SR
    ckpt_path = CHECKPOINT_MAP[ckpt_key]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint 不存在：{ckpt_path}")

    binary = find_binary()
    print(f"[INFO] 使用 {ckpt_key} checkpoint 生成 SR 图像...")
    print(f"       Binary: {binary}")
    print(f"       Checkpoint: {ckpt_path}")
    print(f"       Output: {output_dir}")

    cmd = [
        str(binary),
        "--model",
        str(ckpt_path),
        "--input",
        str(LR_DIR),
        "--output",
        str(output_dir),
    ]
    if device == "cpu":
        cmd.append("--cpu")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode != 0:
            print(f"[ERROR] C++ 推理失败：{result.stderr}")
            raise RuntimeError(f"推理失败，返回码 {result.returncode}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"C++ 推理超时 (3600s)")
    except Exception as e:
        raise RuntimeError(f"推理异常：{e}")

    generated_count = len(list(output_dir.glob("*_sr.png")))
    print(f"[INFO] SR 生成完成：{generated_count} 张图像")
    return output_dir


# ===============================================================================
# Step 5: Bicubic 基线生成
# ===============================================================================

def generate_bicubic_baseline(n_images: int, force_regen: bool = False) -> Path:
    """
    生成 Bicubic 基线（LR 放大到 HR 尺寸）

    输出文件名：celeba_NNNNN.png（无 _sr 后缀）
    """
    output_dir = PROJECT_ROOT / "tmp_eval" / f"val_bicubic_256_{n_images}"
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_bicubic = list(output_dir.glob("*.png"))
    if not force_regen and len(existing_bicubic) >= n_images:
        print(f"[INFO] Bicubic 基线已存在 >= {n_images} 张图像，跳过重新生成")
        return output_dir

    print(f"[INFO] 生成 {n_images} 张 Bicubic 基线图像...")

    lr_files = sorted(LR_DIR.glob("*.png"))[:n_images]

    def process_bicubic(lr_path: Path) -> Tuple[str, bool]:
        """处理单张 bicubic 放大"""
        output_path = output_dir / lr_path.name
        try:
            img = Image.open(lr_path).convert("RGB")
            upsampled = img.resize((256, 256), Image.BICUBIC)
            upsampled.save(output_path)
            return lr_path.name, True
        except Exception as e:
            warnings.warn(f"处理 {lr_path.name} 失败: {e}")
            return lr_path.name, False

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_bicubic, p) for p in lr_files]
        for _ in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="生成 Bicubic 基线",
            dynamic_ncols=True,
        ):
            pass

    actual_count = len(list(output_dir.glob("*.png")))
    print(f"[INFO] Bicubic 基线生成完成：{actual_count} 张图像")
    return output_dir


# ===============================================================================
# Step 6: SR 路径解析辅助函数
# ===============================================================================

def resolve_sr_path(ckpt_key: str, hr_name: str, sr_dirs: Dict[str, Path]) -> Path:
    """
    解析 SR 图像路径

    参数：
      ckpt_key: checkpoint 名称（bicubic/best/latest/epoch190）
      hr_name: HR 文件名（celeba_NNNNN.png）
      sr_dirs: checkpoint -> SR 目录的映射

    返回：SR 图像的完整路径
    """
    folder = sr_dirs[ckpt_key]
    if ckpt_key == "bicubic":
        # Bicubic 基线不带 _sr 后缀
        return folder / hr_name
    else:
        # 模型输出带 _sr 后缀
        stem = Path(hr_name).stem
        return folder / f"{stem}_sr.png"


# ===============================================================================
# Step 7: 单对图像指标计算
# ===============================================================================

def compute_metrics_for_pair(
    hr_path: Path,
    sr_path: Path,
    lpips_net: Optional[object],
    device: str = "cpu",
) -> Dict[str, Optional[float]]:
    """
    计算单对图像的 PSNR, SSIM, LPIPS

    参数：
      hr_path: HR 图像路径
      sr_path: SR 图像路径
      lpips_net: 预加载的 LPIPS 网络
      device: 计算设备

    返回：{psnr, ssim, lpips} 字典，如果文件不存在则返回 None 值
    """
    result = {"psnr": None, "ssim": None, "lpips": None}

    if not hr_path.exists() or not sr_path.exists():
        if not hr_path.exists():
            warnings.warn(f"HR 文件不存在：{hr_path}")
        if not sr_path.exists():
            warnings.warn(f"SR 文件不存在：{sr_path}")
        return result

    try:
        # 加载图像为 RGB float32 [0,1]
        hr_img = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)
        sr_img = cv2.imread(str(sr_path), cv2.IMREAD_COLOR)

        if hr_img is None or sr_img is None:
            return result

        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        sr_img = cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # PSNR
        try:
            psnr_val = skimage_metrics.peak_signal_noise_ratio(
                hr_img, sr_img, data_range=1.0
            )
            result["psnr"] = float(psnr_val)
        except Exception as e:
            warnings.warn(f"PSNR 计算失败 ({hr_path.name}): {e}")

        # SSIM（兼容新旧 skimage API）
        try:
            sig = inspect.signature(skimage_metrics.structural_similarity)
            if "channel_axis" in sig.parameters:
                # 新版 API (skimage >= 0.19)
                ssim_val = skimage_metrics.structural_similarity(
                    hr_img, sr_img, data_range=1.0, channel_axis=2
                )
            else:
                # 旧版 API
                ssim_val = skimage_metrics.structural_similarity(
                    hr_img, sr_img, data_range=1.0, multichannel=True
                )
            result["ssim"] = float(ssim_val)
        except Exception as e:
            warnings.warn(f"SSIM 计算失败 ({hr_path.name}): {e}")

        # LPIPS（主线程执行，后续在 compute_all_metrics 中批量处理）
        if lpips_net is not None:
            try:
                # 转换为 torch tensor: [1, 3, H, W], [-1, 1] 范围
                hr_t = torch.from_numpy(
                    (hr_img * 2 - 1).transpose(2, 0, 1)[None, ...]
                ).float()
                sr_t = torch.from_numpy(
                    (sr_img * 2 - 1).transpose(2, 0, 1)[None, ...]
                ).float()

                if device != "cpu" and torch.cuda.is_available():
                    hr_t = hr_t.to("cuda")
                    sr_t = sr_t.to("cuda")

                with torch.no_grad():
                    lpips_val = lpips_net(hr_t, sr_t).item()
                result["lpips"] = float(lpips_val)
            except Exception as e:
                warnings.warn(f"LPIPS 计算失败 ({hr_path.name}): {e}")

    except Exception as e:
        warnings.warn(f"图像加载失败 ({hr_path.name}): {e}")

    return result


# ===============================================================================
# Step 8: 图像统计量
# ===============================================================================

def compute_image_stats(hr_path: Path) -> Dict[str, Optional[float]]:
    """
    计算 HR 图像的统计量：亮度、边缘密度

    返回：{mean_brightness, edge_density}
    """
    result = {"mean_brightness": None, "edge_density": None}

    if not hr_path.exists():
        return result

    try:
        img = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)
        if img is None:
            return result

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 亮度：灰度均值 / 255（[0, 1]）
        mean_brightness = float(gray.mean() / 255.0)
        result["mean_brightness"] = mean_brightness

        # 边缘密度：Canny 边缘非零像素比例
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(edges.sum() / edges.size)
        result["edge_density"] = edge_density

    except Exception as e:
        warnings.warn(f"统计量计算失败 ({hr_path.name}): {e}")

    return result


# ===============================================================================
# Step 9: 全量指标编排
# ===============================================================================

def compute_all_metrics(
    checkpoint_keys: List[str],
    n_images: int,
    sr_dirs: Dict[str, Path],
    device: str,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    计算所有图像的所有指标

    参数：
      checkpoint_keys: 要评估的 checkpoint 列表
      n_images: 图像数量
      sr_dirs: checkpoint -> SR 目录的映射
      device: 计算设备
      force_recompute: 是否强制重新计算

    返回：包含所有指标的 DataFrame
    """
    # 加载现有 CSV（如果存在）
    existing_df = None
    if EXISTING_METRICS_CSV.exists() and not force_recompute:
        try:
            existing_df = pd.read_csv(EXISTING_METRICS_CSV)
            print(f"[INFO] 加载现有指标 CSV：{EXISTING_METRICS_CSV}")
        except Exception as e:
            warnings.warn(f"加载现有 CSV 失败：{e}")

    # 列出 HR 图像
    hr_files = sorted(HR_DIR.glob("celeba_*.png"))[:n_images]

    # 初始化 LPIPS 网络（仅主线程使用）
    lpips_net = None
    if device != "cpu" and torch.cuda.is_available():
        print("[INFO] 初始化 LPIPS 网络（GPU）...")
        lpips_net = lpips_lib.LPIPS(net="alex").to("cuda")
    else:
        print("[INFO] 初始化 LPIPS 网络（CPU）...")
        lpips_net = lpips_lib.LPIPS(net="alex")
    lpips_net.eval()

    # 构建数据
    data_rows = []
    print(f"[INFO] 计算 {len(hr_files)} 张图像的指标...")

    # 步骤 1: 计算图像统计量和已有指标（PSNR/SSIM）
    print("[INFO] 第 1 阶段：图像统计量和 PSNR/SSIM...")

    def compute_stats_and_basic_metrics(hr_path: Path) -> Dict:
        """计算统计量和基础指标"""
        row = {"file": hr_path.name}

        # 图像统计量
        stats = compute_image_stats(hr_path)
        row.update(stats)

        # PSNR/SSIM for each checkpoint
        for ckpt_key in checkpoint_keys:
            sr_path = resolve_sr_path(ckpt_key, hr_path.name, sr_dirs)
            metrics = compute_metrics_for_pair(hr_path, sr_path, None, device)
            row[f"{ckpt_key}_psnr"] = metrics["psnr"]
            row[f"{ckpt_key}_ssim"] = metrics["ssim"]

        return row

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(compute_stats_and_basic_metrics, p) for p in hr_files]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="统计量和基础指标",
            dynamic_ncols=True,
        ):
            data_rows.append(future.result())

    # 步骤 2: 计算 LPIPS（主线程顺序执行）
    print("[INFO] 第 2 阶段：LPIPS 指标（顺序执行）...")

    for i, hr_path in enumerate(tqdm(hr_files, desc="LPIPS 计算", dynamic_ncols=True)):
        for ckpt_key in checkpoint_keys:
            sr_path = resolve_sr_path(ckpt_key, hr_path.name, sr_dirs)
            lpips_metrics = compute_metrics_for_pair(
                hr_path, sr_path, lpips_net, device
            )
            # 更新对应行的 LPIPS 值
            for row in data_rows:
                if row["file"] == hr_path.name:
                    row[f"{ckpt_key}_lpips"] = lpips_metrics["lpips"]
                    break

        # 每 100 张图像后清空 CUDA 缓存
        if (i + 1) % 100 == 0 and device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 最后清空缓存
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 构建 DataFrame
    df = pd.DataFrame(data_rows)

    # 确保列顺序
    col_order = ["file", "mean_brightness", "edge_density"]
    for ckpt in checkpoint_keys:
        col_order.extend([f"{ckpt}_psnr", f"{ckpt}_ssim", f"{ckpt}_lpips"])

    # 只选择存在的列
    existing_cols = [c for c in col_order if c in df.columns]
    df = df[existing_cols]

    print(f"[INFO] 指标计算完成：{len(df)} 行数据")
    return df


# ===============================================================================
# Step 10: 分层分析
# ===============================================================================

def compute_stratified_analysis(
    df: pd.DataFrame, checkpoint_keys: List[str]
) -> pd.DataFrame:
    """
    按亮度和边缘密度分层分析

    返回：分层统计 DataFrame
    """
    stratified_rows = []

    # 亮度分层
    for bin_name, (low, high) in BRIGHTNESS_BINS.items():
        subset = df[
            (df["mean_brightness"] >= low) & (df["mean_brightness"] < high)
        ]
        for ckpt_key in checkpoint_keys:
            psnr_col = f"{ckpt_key}_psnr"
            ssim_col = f"{ckpt_key}_ssim"
            lpips_col = f"{ckpt_key}_lpips"

            if psnr_col in subset.columns:
                valid_psnr = subset[psnr_col].dropna()
                valid_ssim = subset[ssim_col].dropna() if ssim_col in subset.columns else None
                valid_lpips = subset[lpips_col].dropna() if lpips_col in subset.columns else None

                stratified_rows.append(
                    {
                        "category": "brightness",
                        "bin": bin_name,
                        "checkpoint": ckpt_key,
                        "count": len(subset),
                        "psnr_mean": valid_psnr.mean() if len(valid_psnr) > 0 else None,
                        "psnr_std": valid_psnr.std() if len(valid_psnr) > 0 else None,
                        "ssim_mean": valid_ssim.mean() if valid_ssim is not None and len(valid_ssim) > 0 else None,
                        "ssim_std": valid_ssim.std() if valid_ssim is not None and len(valid_ssim) > 0 else None,
                        "lpips_mean": valid_lpips.mean() if valid_lpips is not None and len(valid_lpips) > 0 else None,
                        "lpips_std": valid_lpips.std() if valid_lpips is not None and len(valid_lpips) > 0 else None,
                    }
                )

    # 边缘密度分层
    for bin_name, (low, high) in EDGE_DENSITY_BINS.items():
        subset = df[(df["edge_density"] >= low) & (df["edge_density"] < high)]
        for ckpt_key in checkpoint_keys:
            psnr_col = f"{ckpt_key}_psnr"
            ssim_col = f"{ckpt_key}_ssim"
            lpips_col = f"{ckpt_key}_lpips"

            if psnr_col in subset.columns:
                valid_psnr = subset[psnr_col].dropna()
                valid_ssim = subset[ssim_col].dropna() if ssim_col in subset.columns else None
                valid_lpips = subset[lpips_col].dropna() if lpips_col in subset.columns else None

                stratified_rows.append(
                    {
                        "category": "edge_density",
                        "bin": bin_name,
                        "checkpoint": ckpt_key,
                        "count": len(subset),
                        "psnr_mean": valid_psnr.mean() if len(valid_psnr) > 0 else None,
                        "psnr_std": valid_psnr.std() if len(valid_psnr) > 0 else None,
                        "ssim_mean": valid_ssim.mean() if valid_ssim is not None and len(valid_ssim) > 0 else None,
                        "ssim_std": valid_ssim.std() if valid_ssim is not None and len(valid_ssim) > 0 else None,
                        "lpips_mean": valid_lpips.mean() if valid_lpips is not None and len(valid_lpips) > 0 else None,
                        "lpips_std": valid_lpips.std() if valid_lpips is not None and len(valid_lpips) > 0 else None,
                    }
                )

    stratified_df = pd.DataFrame(stratified_rows)
    return stratified_df


# ===============================================================================
# Step 11: 可视化
# ===============================================================================

def generate_panels(
    df: pd.DataFrame,
    checkpoint_keys: List[str],
    n_images: int,
    output_dir: Path,
    sr_dirs: Dict[str, Path],
):
    """
    生成对比面板图：worst-20 / best-20 / random-20（按 PSNR 排序）

    每张图：3 列子图 [LR↑ | SR | HR]
    """
    panels_dir = output_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    lr_files = sorted(LR_DIR.glob("*.png"))[:n_images]

    for ckpt_key in checkpoint_keys:
        print(f"[INFO] 生成 {ckpt_key} 的对比面板...")

        ckpt_dir = panels_dir / ckpt_key
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        psnr_col = f"{ckpt_key}_psnr"
        if psnr_col not in df.columns:
            warnings.warn(f"列 {psnr_col} 不存在，跳过面板生成")
            continue

        # 按 PSNR 排序，取 worst-20 / best-20
        df_sorted = df.dropna(subset=[psnr_col]).sort_values(psnr_col)
        worst_20 = df_sorted.head(20)
        best_20 = df_sorted.tail(20)
        random_20 = df_sorted.sample(min(20, len(df_sorted)))

        for subset_name, subset_df in [
            ("worst_20", worst_20),
            ("best_20", best_20),
            ("random_20", random_20),
        ]:
            for idx, row in tqdm(
                subset_df.iterrows(),
                total=len(subset_df),
                desc=f"{ckpt_key}/{subset_name}",
                dynamic_ncols=True,
            ):
                hr_name = row["file"]
                hr_path = HR_DIR / hr_name
                lr_path = LR_DIR / hr_name
                sr_path = resolve_sr_path(ckpt_key, hr_name, sr_dirs)

                if not all(p.exists() for p in [hr_path, lr_path, sr_path]):
                    continue

                try:
                    hr_img = Image.open(hr_path).convert("RGB")
                    lr_img = Image.open(lr_path).convert("RGB")
                    sr_img = Image.open(sr_path).convert("RGB")

                    # LR 放大以匹配尺寸显示
                    lr_upsampled = lr_img.resize(hr_img.size, Image.BICUBIC)

                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                    axes[0].imshow(lr_upsampled)
                    axes[0].set_title(
                        f"LR ↑\n{row.get('mean_brightness', '?'):.3f} bright"
                    )
                    axes[0].axis("off")

                    axes[1].imshow(sr_img)
                    psnr_val = row.get(psnr_col, None)
                    ssim_col = f"{ckpt_key}_ssim"
                    lpips_col = f"{ckpt_key}_lpips"
                    ssim_val = row.get(ssim_col, None)
                    lpips_val = row.get(lpips_col, None)
                    title = f"SR ({ckpt_key})\n"
                    if psnr_val is not None:
                        title += f"PSNR: {psnr_val:.2f}dB "
                    if ssim_val is not None:
                        title += f"SSIM: {ssim_val:.3f}"
                    if lpips_val is not None:
                        title += f"\nLPIPS: {lpips_val:.4f}"
                    axes[1].set_title(title)
                    axes[1].axis("off")

                    axes[2].imshow(hr_img)
                    axes[2].set_title("HR (GT)")
                    axes[2].axis("off")

                    plt.tight_layout()
                    output_file = (
                        ckpt_dir / f"{subset_name}_{idx:04d}_{Path(hr_name).stem}.png"
                    )
                    plt.savefig(output_file, dpi=80, bbox_inches="tight")
                    plt.close(fig)

                except Exception as e:
                    warnings.warn(f"生成面板失败 ({hr_name}): {e}")

        print(f"[INFO] {ckpt_key} 面板生成完成：{ckpt_dir}")


def generate_summary_chart(
    df: pd.DataFrame, checkpoint_keys: List[str], output_file: Path
):
    """
    生成汇总柱状图：4 个 checkpoint × 3 指标（PSNR/SSIM/LPIPS）

    LPIPS 用次坐标轴（值越低越好）
    """
    print("[INFO] 生成汇总柱状图...")

    fig, ax1 = plt.subplots(figsize=(14, 6))

    metrics_data = {}
    for ckpt in checkpoint_keys:
        psnr_col = f"{ckpt}_psnr"
        ssim_col = f"{ckpt}_ssim"
        lpips_col = f"{ckpt}_lpips"

        metrics_data[ckpt] = {
            "psnr": df[psnr_col].dropna().mean() if psnr_col in df.columns else None,
            "ssim": df[ssim_col].dropna().mean() if ssim_col in df.columns else None,
            "lpips": df[lpips_col].dropna().mean() if lpips_col in df.columns else None,
        }

    ckpts = list(metrics_data.keys())
    x = np.arange(len(ckpts))
    width = 0.25

    # PSNR 和 SSIM（左坐标轴）
    psnrs = [metrics_data[c]["psnr"] for c in ckpts]
    ssims = [metrics_data[c]["ssim"] for c in ckpts]

    ax1.bar(x - width, psnrs, width, label="PSNR (dB)", color="steelblue", alpha=0.8)
    ax1.bar(x, ssims, width, label="SSIM", color="coral", alpha=0.8)
    ax1.set_xlabel("Checkpoint", fontsize=12)
    ax1.set_ylabel("PSNR (dB) / SSIM", fontsize=12)
    ax1.set_title("FaceSR_CPP Model Comparison", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ckpts)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    # LPIPS（右坐标轴）
    ax2 = ax1.twinx()
    lpips = [metrics_data[c]["lpips"] for c in ckpts]
    ax2.bar(x + width, lpips, width, label="LPIPS (↓)", color="mediumseagreen", alpha=0.8)
    ax2.set_ylabel("LPIPS", fontsize=12)
    ax2.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] 汇总图表保存：{output_file}")


# ===============================================================================
# Step 12: 保存输出
# ===============================================================================

def save_outputs(
    df: pd.DataFrame,
    stratified_df: pd.DataFrame,
    checkpoint_keys: List[str],
    output_dir: Path,
):
    """
    保存所有输出文件
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 全量逐图指标 CSV
    full_metrics_file = output_dir / "full_metrics_report.csv"
    df.to_csv(full_metrics_file, index=False)
    print(f"[INFO] 保存全量指标：{full_metrics_file}")

    # 2. 汇总表（Markdown）
    summary_rows = []
    for ckpt in checkpoint_keys:
        psnr_col = f"{ckpt}_psnr"
        ssim_col = f"{ckpt}_ssim"
        lpips_col = f"{ckpt}_lpips"

        summary_rows.append(
            {
                "Checkpoint": ckpt,
                "PSNR (dB)": (
                    f"{df[psnr_col].dropna().mean():.4f} ± {df[psnr_col].dropna().std():.4f}"
                    if psnr_col in df.columns
                    else "N/A"
                ),
                "SSIM": (
                    f"{df[ssim_col].dropna().mean():.4f} ± {df[ssim_col].dropna().std():.4f}"
                    if ssim_col in df.columns
                    else "N/A"
                ),
                "LPIPS": (
                    f"{df[lpips_col].dropna().mean():.4f} ± {df[lpips_col].dropna().std():.4f}"
                    if lpips_col in df.columns
                    else "N/A"
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_md_file = output_dir / "summary_table.md"
    with open(summary_md_file, "w") as f:
        f.write("# FaceSR_CPP Evaluation Summary\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n")
    print(f"[INFO] 保存汇总表：{summary_md_file}")

    # 3. 分层统计 CSV
    stratified_file = output_dir / "stratified_analysis.csv"
    stratified_df.to_csv(stratified_file, index=False)
    print(f"[INFO] 保存分层分析：{stratified_file}")

    # 4. 柱状图
    comparison_chart = output_dir / "checkpoint_comparison_bar.png"
    generate_summary_chart(df, checkpoint_keys, comparison_chart)

    # 5. 更新现有 CSV（如果存在）
    if EXISTING_METRICS_CSV.exists():
        try:
            old_df = pd.read_csv(EXISTING_METRICS_CSV)
            merged_df = pd.merge(old_df, df, on="file", how="outer", suffixes=("", "_new"))
            # 简化：直接用新计算的值覆盖
            for col in df.columns:
                if col in merged_df.columns:
                    merged_df[col] = df[col]
            # 保留原 CSV 备份
            backup_file = EXISTING_METRICS_CSV.with_suffix(".csv.backup")
            EXISTING_METRICS_CSV.rename(backup_file)
            merged_df.to_csv(EXISTING_METRICS_CSV, index=False)
            print(f"[INFO] 更新原始指标 CSV：{EXISTING_METRICS_CSV}")
            print(f"[INFO] 备份保存：{backup_file}")
        except Exception as e:
            warnings.warn(f"更新原始 CSV 失败：{e}")


# ===============================================================================
# Main 入口
# ===============================================================================

def main():
    args = parse_args()

    print("=" * 80)
    print("FaceSR_CPP 综合性能评估脚本")
    print("=" * 80)
    print(f"[INFO] 项目路径：{PROJECT_ROOT}")
    print(f"[INFO] 评估图像数：{args.n_images}")
    print(f"[INFO] 输出目录：{args.output_dir}")
    print(f"[INFO] 设备：{args.device}")
    print(f"[INFO] 要评估的 checkpoint：{args.checkpoint}")
    print("=" * 80)

    # 确定要评估的 checkpoint
    if args.checkpoint == "all":
        checkpoint_keys = ["bicubic", "best", "latest", "epoch190"]
    else:
        checkpoint_keys = [args.checkpoint]

    # 如果没有指定具体 checkpoint，添加 bicubic 作为基线
    if "bicubic" not in checkpoint_keys:
        checkpoint_keys = ["bicubic"] + checkpoint_keys

    # Step 1: 生成 LR 图像
    generate_lr_images(args.n_images, args.force_regen_lr)

    # Step 2: 生成 Bicubic 基线
    bicubic_dir = generate_bicubic_baseline(args.n_images, args.force_regen_lr)

    # Step 3: 生成各 checkpoint 的 SR 图像
    sr_dirs = {"bicubic": bicubic_dir}
    for ckpt_key in checkpoint_keys:
        if ckpt_key != "bicubic":
            sr_dir = generate_sr_images(
                ckpt_key,
                args.n_images,
                args.device,
                args.force_regen_sr,
            )
            sr_dirs[ckpt_key] = sr_dir

    # Step 4: 计算所有指标
    df = compute_all_metrics(
        checkpoint_keys,
        args.n_images,
        sr_dirs,
        args.device,
        args.force_recompute_metrics,
    )

    # Step 5: 分层分析
    stratified_df = compute_stratified_analysis(df, checkpoint_keys)

    # Step 6: 生成可视化
    output_dir = Path(args.output_dir)
    if not args.skip_panels:
        generate_panels(df, checkpoint_keys, args.n_images, output_dir, sr_dirs)

    if not args.skip_summary:
        generate_summary_chart(
            df, checkpoint_keys, output_dir / "checkpoint_comparison_bar.png"
        )

    # Step 7: 保存输出
    save_outputs(df, stratified_df, checkpoint_keys, output_dir)

    print("=" * 80)
    print("[SUCCESS] 评估完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
