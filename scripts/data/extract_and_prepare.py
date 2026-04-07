"""
FaceSR_CPP 数据提取与准备脚本（一键式）
功能：
  1. 从 HuggingFace snapshot_download 下载的 parquet/图像中提取图像
  2. 将 FFHQ 图像转为 256x256 -> data/train/HR
  3. 将 CelebA-HQ 划分为 val/HR 和 test/HR
  4. LR 由 C++ 训练程序自动生成

使用方式：
  python extract_and_prepare.py
"""

import os
import sys
import random
import struct
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from PIL import Image
except ImportError:
    print("需要安装 Pillow: pip install Pillow")
    sys.exit(1)

# ===================== 配置 =====================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(PROJECT_DIR, "datasets")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

FFHQ_RAW_DIR = os.path.join(DATASETS_DIR, "ffhq_raw")
CELEBAHQ_RAW_DIR = os.path.join(DATASETS_DIR, "celebahq_raw")

TRAIN_HR_DIR = os.path.join(DATA_DIR, "train", "HR")
VAL_HR_DIR = os.path.join(DATA_DIR, "val", "HR")
TEST_HR_DIR = os.path.join(DATA_DIR, "test", "HR")

TARGET_SIZE = 256
VAL_COUNT = 3000
TEST_COUNT = 3000
RANDOM_SEED = 42
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
MAX_WORKERS = 8


def find_all_images(directory):
    """递归查找目录下所有图像文件"""
    images = []
    for root, dirs, files in os.walk(directory):
        for f in sorted(files):
            if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS:
                images.append(os.path.join(root, f))
    return images


def try_extract_parquet_images(raw_dir, output_dir, label):
    """尝试从 parquet 文件中提取图像"""
    parquet_files = []
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if f.endswith('.parquet'):
                parquet_files.append(os.path.join(root, f))

    if not parquet_files:
        return False

    print(f"\n[{label}] 发现 {len(parquet_files)} 个 parquet 文件，提取图像...")

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("  安装 pyarrow...")
        os.system(f"{sys.executable} -m pip install pyarrow")
        import pyarrow.parquet as pq

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for pf in sorted(parquet_files):
        print(f"  处理: {os.path.basename(pf)}")
        table = pq.read_table(pf)

        # 查找图像列
        image_col = None
        for col_name in table.column_names:
            if col_name.lower() in ('image', 'img', 'pixel_values', 'data'):
                image_col = col_name
                break

        if image_col is None:
            # 尝试第一列
            print(f"    列名: {table.column_names}")
            # 检查是否有 bytes 类型的列
            for col_name in table.column_names:
                col = table.column(col_name)
                if hasattr(col, 'type'):
                    type_str = str(col.type)
                    if 'struct' in type_str or 'binary' in type_str:
                        image_col = col_name
                        break

        if image_col is None:
            print(f"    未找到图像列，跳过")
            continue

        col = table.column(image_col)
        for i in range(len(col)):
            item = col[i].as_py()
            if isinstance(item, dict) and 'bytes' in item:
                img_bytes = item['bytes']
            elif isinstance(item, bytes):
                img_bytes = item
            else:
                continue

            import io
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            if img.size != (TARGET_SIZE, TARGET_SIZE):
                img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)
            img.save(os.path.join(output_dir, f"{count:05d}.png"), quality=95)
            count += 1

            if count % 2000 == 0:
                print(f"    已提取 {count} 张图像")

    print(f"  共提取 {count} 张图像")
    return count > 0


def copy_and_resize_images(src_images, dst_dir, label):
    """复制图像并 resize 到目标尺寸"""
    os.makedirs(dst_dir, exist_ok=True)
    total = len(src_images)
    success = 0

    print(f"\n[{label}] 处理 {total} 张图像 -> {dst_dir}")

    def process_one(args):
        idx, src_path = args
        try:
            img = Image.open(src_path).convert('RGB')
            if img.size != (TARGET_SIZE, TARGET_SIZE):
                img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)
            dst_path = os.path.join(dst_dir, f"{idx:05d}.png")
            img.save(dst_path, quality=95)
            return True
        except Exception as e:
            return False

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_one, (i, p)) for i, p in enumerate(src_images)]
        for i, f in enumerate(as_completed(futures), 1):
            if f.result():
                success += 1
            if i % 2000 == 0 or i == total:
                print(f"  进度: {i}/{total}")

    print(f"  完成: {success}/{total}")
    return success


def main():
    print("=" * 50)
    print("FaceSR_CPP 数据提取与准备")
    print("=" * 50)

    # 确保目录存在
    for d in [TRAIN_HR_DIR, VAL_HR_DIR, TEST_HR_DIR]:
        os.makedirs(d, exist_ok=True)

    # ========== 处理 FFHQ ==========
    print("\n" + "=" * 40)
    print("处理 FFHQ -> 训练集")
    print("=" * 40)

    train_existing = len([f for f in os.listdir(TRAIN_HR_DIR) if f.endswith('.png')])
    if train_existing >= 60000:
        print(f"训练集已有 {train_existing} 张图像，跳过")
    else:
        # 先尝试直接找图像文件
        ffhq_images = find_all_images(FFHQ_RAW_DIR)
        if len(ffhq_images) >= 1000:
            print(f"找到 {len(ffhq_images)} 张 FFHQ 图像文件")
            copy_and_resize_images(ffhq_images, TRAIN_HR_DIR, "FFHQ -> train/HR")
        else:
            # 尝试从 parquet 提取
            print(f"仅找到 {len(ffhq_images)} 张图像文件，尝试从 parquet 提取...")
            if not try_extract_parquet_images(FFHQ_RAW_DIR, TRAIN_HR_DIR, "FFHQ"):
                print("[ERROR] FFHQ 数据提取失败，请检查 datasets/ffhq_raw/ 目录")

    # ========== 处理 CelebA-HQ ==========
    print("\n" + "=" * 40)
    print("处理 CelebA-HQ -> 验证集 + 测试集")
    print("=" * 40)

    val_existing = len([f for f in os.listdir(VAL_HR_DIR) if f.endswith('.png')])
    test_existing = len([f for f in os.listdir(TEST_HR_DIR) if f.endswith('.png')])
    if val_existing >= VAL_COUNT and test_existing >= TEST_COUNT:
        print(f"验证集 ({val_existing}) 和测试集 ({test_existing}) 已就绪，跳过")
    else:
        # 先提取所有 CelebA-HQ 图像到临时目录
        celebahq_temp = os.path.join(DATASETS_DIR, "celebahq_extracted")
        celebahq_images = find_all_images(CELEBAHQ_RAW_DIR)

        if len(celebahq_images) >= 1000:
            print(f"找到 {len(celebahq_images)} 张 CelebA-HQ 图像文件")
            all_images = celebahq_images
        else:
            print(f"仅找到 {len(celebahq_images)} 张图像，尝试从 parquet 提取...")
            os.makedirs(celebahq_temp, exist_ok=True)
            if try_extract_parquet_images(CELEBAHQ_RAW_DIR, celebahq_temp, "CelebA-HQ"):
                all_images = find_all_images(celebahq_temp)
            else:
                print("[ERROR] CelebA-HQ 数据提取失败")
                all_images = []

        if len(all_images) >= VAL_COUNT + TEST_COUNT:
            random.seed(RANDOM_SEED)
            shuffled = all_images.copy()
            random.shuffle(shuffled)

            val_images = shuffled[:VAL_COUNT]
            test_images = shuffled[VAL_COUNT:VAL_COUNT + TEST_COUNT]

            copy_and_resize_images(val_images, VAL_HR_DIR, "CelebA-HQ -> val/HR")
            copy_and_resize_images(test_images, TEST_HR_DIR, "CelebA-HQ -> test/HR")
        else:
            print(f"[ERROR] CelebA-HQ 图像不足: 需要 {VAL_COUNT + TEST_COUNT}, 只有 {len(all_images)}")

    # ========== 最终统计 ==========
    train_count = len([f for f in os.listdir(TRAIN_HR_DIR) if f.endswith('.png')])
    val_count = len([f for f in os.listdir(VAL_HR_DIR) if f.endswith('.png')])
    test_count = len([f for f in os.listdir(TEST_HR_DIR) if f.endswith('.png')])

    print("\n" + "=" * 50)
    print("数据准备完成!")
    print("=" * 50)
    print(f"  data/train/HR: {train_count} 张  (FFHQ)")
    print(f"  data/val/HR:   {val_count} 张  (CelebA-HQ)")
    print(f"  data/test/HR:  {test_count} 张  (CelebA-HQ)")
    print(f"  LR 图像: 由训练程序自动生成")
    print("=" * 50)

    if train_count > 0 and val_count > 0:
        print("\n可以开始训练:")
        print(f"  facesr_train.exe --train-hr data/train/HR --epochs 200")


if __name__ == "__main__":
    main()
