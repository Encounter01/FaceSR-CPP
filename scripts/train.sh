#!/bin/bash
# ============================================================================
# FaceSR 云端训练脚本 (AutoDL 全自动化 · 国内镜像加速版)
#
# AutoDL 目录结构:
#   /root/autodl-tmp/    数据盘 (实例释放后保留, 存放数据集和模型)
#   /root/autodl-fs/     文件存储 (跨实例共享)
#   /root/               系统盘 (实例释放后清空)
#
# 用法:
#   bash train.sh setup     # 一键安装依赖 + 编译 + 准备数据 (全自动)
#   bash train.sh build     # 仅编译
#   bash train.sh start     # 开始训练
#   bash train.sh resume    # 从 checkpoint 恢复训练
#   bash train.sh download  # 显示下载模型的命令
#   bash train.sh data      # 仅下载并准备数据集
#   bash train.sh status    # 查看 GPU/磁盘/训练状态
#   bash train.sh help      # 帮助
#
# 特性:
#   - 国内镜像加速 (pip清华源, HuggingFace镜像, CMake/LibTorch备用源)
#   - 自动修复 AutoDL conda cmake 冲突
#   - 自动检测 CUDA 版本匹配 LibTorch
#   - 自动检测 GPU 显存调整 batch_size
#   - 数据集全自动下载 + 提取 (FFHQ + CelebA-HQ)
#   - 断点续传, 失败重试
# ============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
BIN_DIR="$BUILD_DIR/bin"
EXE="$BIN_DIR/facesr_train"
CONFIG="$PROJECT_DIR/config/train_config.ini"

# ============================================================================
# AutoDL 路径配置
# 数据集和 checkpoint 放在数据盘, 实例释放后不丢失
# ============================================================================
AUTODL_DATA="/root/autodl-tmp"
CHECKPOINT_DIR="$AUTODL_DATA/facesr_checkpoints"
DATA_DIR="$AUTODL_DATA/facesr_data"
RESULT_DIR="$AUTODL_DATA/facesr_results"
LIBTORCH_DIR="$AUTODL_DATA/libtorch"

# ============================================================================
# 国内镜像配置
# ============================================================================
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
HF_MIRROR="https://hf-mirror.com"
# CMake 国内镜像 (GitHub Proxy)
CMAKE_MIRROR="https://ghfast.top/https://github.com/Kitware/CMake/releases/download"

# ============================================================================
# 颜色输出
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()    { echo -e "${CYAN}[$1]${NC} $2"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }

# ============================================================================
# 修复 AutoDL conda cmake 冲突
# AutoDL 的 miniconda 中有一个损坏的 cmake python 包装器
# 它会覆盖系统的 /usr/bin/cmake, 导致 cmake 无法使用
# ============================================================================
fix_cmake() {
    log_info "检查 cmake 环境..."

    # 检测 cmake 是否能正常工作
    if cmake --version &>/dev/null; then
        local cmake_ver
        cmake_ver=$(cmake --version | head -1 | grep -oP '[0-9]+\.[0-9]+\.[0-9]+')
        local cmake_major cmake_minor
        cmake_major=$(echo "$cmake_ver" | cut -d. -f1)
        cmake_minor=$(echo "$cmake_ver" | cut -d. -f2)

        if [ "$cmake_major" -ge 3 ] && [ "$cmake_minor" -ge 18 ]; then
            log_success "cmake $cmake_ver 可用 (>= 3.18)"
            return 0
        else
            log_warn "cmake $cmake_ver 版本过低, 需要 >= 3.18"
        fi
    else
        log_warn "cmake 不可用, 可能被 conda 包装器覆盖"
    fi

    # 删除 conda/pip 中损坏的 cmake
    log_info "清理损坏的 cmake..."
    pip uninstall -y cmake 2>/dev/null || true
    conda uninstall -y cmake 2>/dev/null || true
    rm -f /root/miniconda3/bin/cmake /root/miniconda3/bin/cmake3 2>/dev/null || true
    rm -f /root/anaconda3/bin/cmake /root/anaconda3/bin/cmake3 2>/dev/null || true
    hash -r

    # 检查系统 cmake 是否可用且版本足够
    if /usr/bin/cmake --version &>/dev/null; then
        local sys_ver
        sys_ver=$(/usr/bin/cmake --version | head -1 | grep -oP '[0-9]+\.[0-9]+\.[0-9]+')
        local sys_major sys_minor
        sys_major=$(echo "$sys_ver" | cut -d. -f1)
        sys_minor=$(echo "$sys_ver" | cut -d. -f2)
        if [ "$sys_major" -ge 3 ] && [ "$sys_minor" -ge 18 ]; then
            log_success "使用系统 cmake $sys_ver"
            return 0
        fi
    fi

    # 需要安装新版 cmake
    log_info "安装 cmake 3.28.3..."

    local cmake_script="cmake-3.28.3-linux-x86_64.sh"
    local cmake_installed=false

    # 尝试国内镜像
    log_info "从国内镜像下载 cmake..."
    if wget -q --show-progress --timeout=30 -O "/tmp/$cmake_script" \
        "${CMAKE_MIRROR}/v3.28.3/$cmake_script" 2>/dev/null; then
        cmake_installed=true
    fi

    # 国内镜像失败, 尝试官方源
    if [ "$cmake_installed" = false ]; then
        log_warn "国内镜像不可用, 尝试官方源..."
        if wget -q --show-progress --timeout=60 -O "/tmp/$cmake_script" \
            "https://github.com/Kitware/CMake/releases/download/v3.28.3/$cmake_script" 2>/dev/null; then
            cmake_installed=true
        fi
    fi

    if [ "$cmake_installed" = true ]; then
        bash "/tmp/$cmake_script" --skip-license --prefix=/usr/local
        rm -f "/tmp/$cmake_script"
        hash -r
    else
        # 最后的后备: 使用 pip 安装
        log_warn "下载失败, 使用 pip 安装 cmake..."
        pip install cmake -i "$PIP_MIRROR" --root-user-action=ignore
        hash -r
    fi

    # 验证
    if cmake --version &>/dev/null; then
        local final_ver
        final_ver=$(cmake --version | head -1 | grep -oP '[0-9]+\.[0-9]+\.[0-9]+')
        log_success "cmake $final_ver 安装成功"
    else
        log_error "cmake 安装失败, 请手动安装"
        exit 1
    fi
}

# ============================================================================
# 自动检测 CUDA 版本, 选择匹配的 LibTorch
# ============================================================================
detect_cuda_and_libtorch() {
    # 检测 CUDA 版本
    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    elif [ -f /usr/local/cuda/version.txt ]; then
        CUDA_VER=$(cat /usr/local/cuda/version.txt | grep -oP '[0-9]+\.[0-9]+')
    elif command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
    else
        CUDA_VER="12.4"
        log_warn "无法检测 CUDA 版本, 默认使用 $CUDA_VER"
    fi

    log_info "检测到 CUDA 版本: $CUDA_VER"

    # 根据 CUDA 版本选择 LibTorch 下载链接
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)

    if [ "$CUDA_MAJOR" -ge 13 ] || ([ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 8 ]); then
        LIBTORCH_CUDA="cu128"
        LIBTORCH_VER="2.7.0"
    elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
        LIBTORCH_CUDA="cu124"
        LIBTORCH_VER="2.5.1"
    elif [ "$CUDA_MAJOR" -eq 12 ]; then
        LIBTORCH_CUDA="cu121"
        LIBTORCH_VER="2.4.1"
    else
        LIBTORCH_CUDA="cu118"
        LIBTORCH_VER="2.1.0"
    fi

    LIBTORCH_FILENAME="libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VER}%2B${LIBTORCH_CUDA}.zip"
    LIBTORCH_URL_OFFICIAL="https://download.pytorch.org/libtorch/${LIBTORCH_CUDA}/${LIBTORCH_FILENAME}"

    log_info "将使用 LibTorch ${LIBTORCH_VER}+${LIBTORCH_CUDA}"
}

# ============================================================================
# 自动检测 GPU 显存, 调整 batch_size
# ============================================================================
detect_gpu_and_batch_size() {
    local gpu_mem_mb
    gpu_mem_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')

    if [ -z "$gpu_mem_mb" ]; then
        RECOMMENDED_BATCH=16
        log_warn "无法检测 GPU 显存, 默认 batch_size=$RECOMMENDED_BATCH"
        return
    fi

    local gpu_mem_gb=$(( gpu_mem_mb / 1024 ))
    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs)

    log_info "GPU: $gpu_name (${gpu_mem_gb}GB)"

    if [ "$gpu_mem_gb" -ge 30 ]; then
        RECOMMENDED_BATCH=48
        RECOMMENDED_WORKERS=20
    elif [ "$gpu_mem_gb" -ge 22 ]; then
        RECOMMENDED_BATCH=32
        RECOMMENDED_WORKERS=16
    elif [ "$gpu_mem_gb" -ge 14 ]; then
        RECOMMENDED_BATCH=16
        RECOMMENDED_WORKERS=12
    elif [ "$gpu_mem_gb" -ge 10 ]; then
        RECOMMENDED_BATCH=8
        RECOMMENDED_WORKERS=8
    else
        RECOMMENDED_BATCH=4
        RECOMMENDED_WORKERS=4
    fi

    log_info "推荐参数: batch_size=$RECOMMENDED_BATCH, workers=$RECOMMENDED_WORKERS"
}

# ============================================================================
# 下载 LibTorch (带重试和镜像回退)
# ============================================================================
download_libtorch() {
    if [ -d "$LIBTORCH_DIR" ] && [ -f "$LIBTORCH_DIR/share/cmake/Torch/TorchConfig.cmake" ]; then
        log_success "LibTorch 已存在: $LIBTORCH_DIR, 跳过下载"
        return 0
    fi

    local zip_file="$AUTODL_DATA/libtorch.zip"

    # 如果之前下载了一半, 支持断点续传
    local wget_opts="-q --show-progress --timeout=60 --tries=3 -c"

    log_info "下载 LibTorch ${LIBTORCH_VER}+${LIBTORCH_CUDA} (~2.3GB)..."

    local download_ok=false

    # 尝试 PyTorch 官方源 (AutoDL 对 pytorch 官方有较好的连通性)
    log_info "尝试 PyTorch 官方源..."
    if wget $wget_opts -O "$zip_file" "$LIBTORCH_URL_OFFICIAL"; then
        download_ok=true
    fi

    if [ "$download_ok" = false ]; then
        log_error "LibTorch 下载失败"
        log_info "请手动下载 LibTorch 并解压到 $LIBTORCH_DIR"
        log_info "下载链接: $LIBTORCH_URL_OFFICIAL"
        exit 1
    fi

    log_info "解压 LibTorch..."
    cd "$AUTODL_DATA"
    unzip -q -o "$zip_file"
    rm -f "$zip_file"
    log_success "LibTorch 已安装到: $LIBTORCH_DIR"
}

# ============================================================================
# 生成 AutoDL 适配的训练配置 (自动适配 GPU 显存)
# ============================================================================
generate_autodl_config() {
    log_info "生成 AutoDL 适配配置..."

    detect_gpu_and_batch_size

    cat > "$PROJECT_DIR/config/train_config_autodl.ini" << CONF
# FaceSR Training Configuration (AutoDL Cloud)
# 人脸超分辨率训练配置 — AutoDL 云端优化版
# 自动生成于 $(date '+%Y-%m-%d %H:%M:%S')

[data]
# 数据目录指向 AutoDL 数据盘 (实例释放后保留)
train_hr_dir = ${DATA_DIR}/train/HR
train_lr_dir = ${DATA_DIR}/train/LR
val_hr_dir = ${DATA_DIR}/val/HR
val_lr_dir = ${DATA_DIR}/val/LR

# 图像尺寸
hr_size = 256
lr_size = 64
scale = 4

[training]
# 根据 GPU 显存自动推荐的参数
batch_size = ${RECOMMENDED_BATCH}
num_epochs = 200
num_workers = ${RECOMMENDED_WORKERS}

# 学习率
lr_g = 0.0002
lr_d = 0.0002

# 阶段训练边界 (epoch)
# 阶段1: 仅像素损失 (0 - phase1_epochs)
# 阶段2: 像素+感知损失 (phase1_epochs - phase2_epochs)
# 阶段3: 全部损失 (phase2_epochs - num_epochs)
phase1_epochs = 30
phase2_epochs = 80

# 学习率调度器
lr_scheduler = cosine
lr_min = 1e-7
lr_warmup_epochs = 0
lr_decay_gamma = 0.5
lr_decay_step = 50

[loss]
pixel_weight = 1.0
perceptual_weight = 1.0
gan_weight = 0.1
pixel_type = l1
gan_type = vanilla

[device]
device_type = gpu
pin_memory = true

[output]
# 输出目录指向 AutoDL 数据盘
checkpoint_dir = ${CHECKPOINT_DIR}
result_dir = ${RESULT_DIR}

save_interval = 10
latest_save_interval = 1
val_interval = 2
log_interval = 100
auto_resume = true

[model]
vgg_weights =
CONF

    CONFIG="$PROJECT_DIR/config/train_config_autodl.ini"
    log_success "配置文件: $CONFIG"
    log_info "  batch_size=$RECOMMENDED_BATCH, workers=$RECOMMENDED_WORKERS"
}

# ============================================================================
# 全自动数据集下载与准备 (国内镜像)
# ============================================================================
prepare_data() {
    log_info "检查数据集..."

    local train_count val_count test_count
    train_count=$(find "$DATA_DIR/train/HR" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
    val_count=$(find "$DATA_DIR/val/HR" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
    test_count=$(find "$DATA_DIR/test/HR" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)

    log_info "当前数据: 训练=${train_count}, 验证=${val_count}, 测试=${test_count}"

    if [ "$train_count" -ge 60000 ] && [ "$val_count" -ge 2000 ]; then
        log_success "数据集已就绪, 跳过"
        return 0
    fi

    # 确保 Python 依赖
    pip install -q Pillow pyarrow huggingface_hub -i "$PIP_MIRROR" --root-user-action=ignore 2>/dev/null

    # --- 方式1: 项目自带原始数据 ---
    if [ -d "$PROJECT_DIR/datasets" ] && [ "$(ls -A "$PROJECT_DIR/datasets" 2>/dev/null)" ]; then
        log_info "从项目 datasets/ 目录提取数据..."
        cd "$PROJECT_DIR"
        DATA_DIR="$DATA_DIR" python3 -c "
import os, sys
sys.path.insert(0, '.')
import extract_and_prepare as ep
ep.DATA_DIR = os.environ['DATA_DIR']
ep.TRAIN_HR_DIR = os.path.join(ep.DATA_DIR, 'train', 'HR')
ep.VAL_HR_DIR = os.path.join(ep.DATA_DIR, 'val', 'HR')
ep.TEST_HR_DIR = os.path.join(ep.DATA_DIR, 'test', 'HR')
ep.main()
"
        return 0
    fi

    # --- 方式2: 从 HuggingFace 镜像自动下载 ---
    log_info "从 HuggingFace 镜像自动下载数据集..."
    log_info "镜像地址: $HF_MIRROR"

    # 创建下载目录
    local ffhq_raw="$AUTODL_DATA/facesr_datasets/ffhq_raw"
    local celebahq_raw="$AUTODL_DATA/facesr_datasets/celebahq_raw"
    mkdir -p "$ffhq_raw" "$celebahq_raw"

    cd "$PROJECT_DIR"

    # 下载 + 提取一步完成
    HF_ENDPOINT="$HF_MIRROR" python3 << 'PYEOF'
import os, sys, random

# 设置 HuggingFace 镜像
hf_mirror = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = hf_mirror

data_dir = os.environ.get("DATA_DIR", "/root/autodl-tmp/facesr_data")
autodl_data = os.environ.get("AUTODL_DATA", "/root/autodl-tmp")

TRAIN_HR = os.path.join(data_dir, "train", "HR")
VAL_HR   = os.path.join(data_dir, "val", "HR")
TEST_HR  = os.path.join(data_dir, "test", "HR")
for d in [TRAIN_HR, VAL_HR, TEST_HR]:
    os.makedirs(d, exist_ok=True)

TARGET_SIZE = 256
VAL_COUNT = 3000
TEST_COUNT = 3000

try:
    from PIL import Image
except ImportError:
    print("[ERROR] Pillow 未安装")
    sys.exit(1)

def count_images(d):
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.lower().endswith(('.png','.jpg','.jpeg'))])

# ========== 下载 FFHQ ==========
train_existing = count_images(TRAIN_HR)
if train_existing >= 60000:
    print(f"[OK] 训练集已有 {train_existing} 张, 跳过")
else:
    print(f"\n{'='*50}")
    print("下载 FFHQ 数据集 (训练集, ~70000张人脸)...")
    print(f"{'='*50}")
    try:
        from huggingface_hub import snapshot_download
        ffhq_dir = os.path.join(autodl_data, "facesr_datasets", "ffhq_raw")
        snapshot_download(
            "FFHQ/FFHQ",
            repo_type="dataset",
            local_dir=ffhq_dir,
            resume_download=True,  # 断点续传
        )
        print("[OK] FFHQ 下载完成, 开始提取...")

        # 提取: 先尝试 parquet, 再尝试直接图像
        import io
        extracted = 0

        # 查找 parquet 文件
        parquet_files = []
        for root, dirs, files in os.walk(ffhq_dir):
            for f in sorted(files):
                if f.endswith('.parquet'):
                    parquet_files.append(os.path.join(root, f))

        if parquet_files:
            print(f"  发现 {len(parquet_files)} 个 parquet 文件")
            import pyarrow.parquet as pq
            for pf in sorted(parquet_files):
                print(f"  处理: {os.path.basename(pf)}")
                table = pq.read_table(pf)
                # 查找图像列
                image_col = None
                for cn in table.column_names:
                    if cn.lower() in ('image', 'img', 'pixel_values', 'data'):
                        image_col = cn
                        break
                if image_col is None:
                    for cn in table.column_names:
                        col = table.column(cn)
                        ts = str(col.type)
                        if 'struct' in ts or 'binary' in ts:
                            image_col = cn
                            break
                if image_col is None:
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
                    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    if img.size != (TARGET_SIZE, TARGET_SIZE):
                        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)
                    img.save(os.path.join(TRAIN_HR, f"{extracted:05d}.png"), quality=95)
                    extracted += 1
                    if extracted % 5000 == 0:
                        print(f"    已提取 {extracted} 张")
        else:
            # 直接查找图像文件
            from concurrent.futures import ThreadPoolExecutor
            exts = {'.jpg', '.jpeg', '.png', '.bmp'}
            imgs = []
            for root, dirs, files in os.walk(ffhq_dir):
                for f in sorted(files):
                    if os.path.splitext(f)[1].lower() in exts:
                        imgs.append(os.path.join(root, f))
            print(f"  发现 {len(imgs)} 张图像文件")
            def process(args):
                idx, src = args
                try:
                    img = Image.open(src).convert('RGB')
                    if img.size != (TARGET_SIZE, TARGET_SIZE):
                        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)
                    img.save(os.path.join(TRAIN_HR, f"{idx:05d}.png"), quality=95)
                    return True
                except:
                    return False
            with ThreadPoolExecutor(max_workers=8) as ex:
                results = list(ex.map(process, enumerate(imgs)))
            extracted = sum(results)

        print(f"[OK] FFHQ 提取完成: {extracted} 张图像")
    except Exception as e:
        print(f"[ERROR] FFHQ 下载/提取失败: {e}")
        print("  可手动将 256x256 人脸 PNG 放入:", TRAIN_HR)

# ========== 下载 CelebA-HQ ==========
val_existing = count_images(VAL_HR)
test_existing = count_images(TEST_HR)
if val_existing >= VAL_COUNT and test_existing >= TEST_COUNT:
    print(f"[OK] 验证集 ({val_existing}) 和测试集 ({test_existing}) 已就绪, 跳过")
else:
    print(f"\n{'='*50}")
    print("下载 CelebA-HQ 数据集 (验证/测试集)...")
    print(f"{'='*50}")
    try:
        from huggingface_hub import snapshot_download
        celebahq_dir = os.path.join(autodl_data, "facesr_datasets", "celebahq_raw")
        snapshot_download(
            "mattymchen/CelebA-HQ",
            repo_type="dataset",
            local_dir=celebahq_dir,
            resume_download=True,
        )
        print("[OK] CelebA-HQ 下载完成, 开始提取...")

        # 提取所有图像
        all_images = []
        import io
        # 先尝试 parquet
        parquet_files = []
        for root, dirs, files in os.walk(celebahq_dir):
            for f in sorted(files):
                if f.endswith('.parquet'):
                    parquet_files.append(os.path.join(root, f))
        if parquet_files:
            import pyarrow.parquet as pq
            temp_dir = os.path.join(autodl_data, "facesr_datasets", "celebahq_temp")
            os.makedirs(temp_dir, exist_ok=True)
            count = 0
            for pf in sorted(parquet_files):
                print(f"  处理: {os.path.basename(pf)}")
                table = pq.read_table(pf)
                image_col = None
                for cn in table.column_names:
                    if cn.lower() in ('image', 'img', 'pixel_values', 'data'):
                        image_col = cn
                        break
                if image_col is None:
                    for cn in table.column_names:
                        ts = str(table.column(cn).type)
                        if 'struct' in ts or 'binary' in ts:
                            image_col = cn
                            break
                if image_col is None:
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
                    path = os.path.join(temp_dir, f"{count:05d}.png")
                    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    if img.size != (TARGET_SIZE, TARGET_SIZE):
                        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)
                    img.save(path, quality=95)
                    all_images.append(path)
                    count += 1
                    if count % 5000 == 0:
                        print(f"    已提取 {count} 张")
        else:
            exts = {'.jpg', '.jpeg', '.png', '.bmp'}
            for root, dirs, files in os.walk(celebahq_dir):
                for f in sorted(files):
                    if os.path.splitext(f)[1].lower() in exts:
                        all_images.append(os.path.join(root, f))

        print(f"  CelebA-HQ 共 {len(all_images)} 张图像")

        if len(all_images) >= VAL_COUNT + TEST_COUNT:
            random.seed(42)
            shuffled = all_images.copy()
            random.shuffle(shuffled)
            # 验证集
            for i, src in enumerate(shuffled[:VAL_COUNT]):
                img = Image.open(src).convert('RGB')
                if img.size != (TARGET_SIZE, TARGET_SIZE):
                    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)
                img.save(os.path.join(VAL_HR, f"{i:05d}.png"), quality=95)
            print(f"  验证集: {VAL_COUNT} 张")
            # 测试集
            for i, src in enumerate(shuffled[VAL_COUNT:VAL_COUNT+TEST_COUNT]):
                img = Image.open(src).convert('RGB')
                if img.size != (TARGET_SIZE, TARGET_SIZE):
                    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)
                img.save(os.path.join(TEST_HR, f"{i:05d}.png"), quality=95)
            print(f"  测试集: {TEST_COUNT} 张")
        else:
            print(f"[WARN] 图像不足 ({len(all_images)}), 需要 {VAL_COUNT+TEST_COUNT}")
            if len(all_images) > 0:
                half = len(all_images) // 2
                for i, src in enumerate(all_images[:half]):
                    img = Image.open(src).convert('RGB')
                    if img.size != (TARGET_SIZE, TARGET_SIZE):
                        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)
                    img.save(os.path.join(VAL_HR, f"{i:05d}.png"), quality=95)
                for i, src in enumerate(all_images[half:]):
                    img = Image.open(src).convert('RGB')
                    if img.size != (TARGET_SIZE, TARGET_SIZE):
                        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)
                    img.save(os.path.join(TEST_HR, f"{i:05d}.png"), quality=95)
                print(f"  已使用全部 {len(all_images)} 张平分验证/测试")
        print("[OK] CelebA-HQ 提取完成")
    except Exception as e:
        print(f"[ERROR] CelebA-HQ 下载/提取失败: {e}")
        print("  可手动将 256x256 人脸 PNG 放入:", VAL_HR, "和", TEST_HR)

# 最终统计
t = count_images(TRAIN_HR)
v = count_images(VAL_HR)
te = count_images(TEST_HR)
print(f"\n{'='*50}")
print(f"数据集准备完成!")
print(f"  训练集: {t} 张  ({TRAIN_HR})")
print(f"  验证集: {v} 张  ({VAL_HR})")
print(f"  测试集: {te} 张  ({TEST_HR})")
print(f"  LR 图像: 由训练程序自动 bicubic 降采样生成")
print(f"{'='*50}")
PYEOF
}

# ============================================================================
# 编译
# ============================================================================
build() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  编译 FaceSR 训练程序${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""

    # 先修复 cmake
    fix_cmake

    mkdir -p "$BUILD_DIR"

    export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-$LIBTORCH_DIR}"
    export LD_LIBRARY_PATH="$LIBTORCH_DIR/lib:$LD_LIBRARY_PATH"

    log_step "1/2" "CMake 配置..."
    cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$LIBTORCH_DIR" \
        -DBUILD_GUI=OFF \
        -DTORCH_CUDA_SKIP_NVCC=ON

    echo ""
    log_step "2/2" "编译 ($(nproc) 线程)..."
    cmake --build "$BUILD_DIR" --config Release -j"$(nproc)"

    echo ""
    if [ -f "$EXE" ]; then
        log_success "编译成功: $EXE"
    else
        log_error "编译失败, 请检查错误信息"
        exit 1
    fi
}

# ============================================================================
# 一键 setup
# ============================================================================
setup() {
    echo ""
    echo -e "${CYAN}========================================================${NC}"
    echo -e "${CYAN}  FaceSR 一键环境配置 (AutoDL · 国内镜像加速版)${NC}"
    echo -e "${CYAN}========================================================${NC}"
    echo ""

    local start_time=$SECONDS

    # 检测 CUDA
    detect_cuda_and_libtorch

    # 检测 GPU
    echo ""
    log_info "GPU 信息:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  无法获取"
    echo ""

    # 创建 AutoDL 数据盘目录
    mkdir -p "$DATA_DIR/train/HR" "$DATA_DIR/train/LR"
    mkdir -p "$DATA_DIR/val/HR" "$DATA_DIR/val/LR"
    mkdir -p "$DATA_DIR/test/HR" "$DATA_DIR/test/LR"
    mkdir -p "$CHECKPOINT_DIR"
    mkdir -p "$RESULT_DIR"

    # 1. 安装系统依赖
    echo ""
    log_step "1/6" "安装系统依赖..."
    apt update -qq 2>/dev/null
    apt install -y -qq build-essential libopencv-dev unzip wget tmux 2>/dev/null
    pip install -q Pillow pyarrow huggingface_hub -i "$PIP_MIRROR" --root-user-action=ignore 2>/dev/null
    log_success "系统依赖安装完成"

    # 2. 修复 cmake
    echo ""
    log_step "2/6" "配置 CMake..."
    fix_cmake

    # 3. 下载 LibTorch
    echo ""
    log_step "3/6" "配置 LibTorch..."
    download_libtorch
    export CMAKE_PREFIX_PATH="$LIBTORCH_DIR"
    export LD_LIBRARY_PATH="$LIBTORCH_DIR/lib:$LD_LIBRARY_PATH"

    # 4. 生成 AutoDL 配置
    echo ""
    log_step "4/6" "生成训练配置..."
    generate_autodl_config

    # 5. 编译
    echo ""
    log_step "5/6" "编译项目..."
    build

    # 6. 准备数据
    echo ""
    log_step "6/6" "准备数据集 (自动下载)..."
    prepare_data

    local elapsed=$(( SECONDS - start_time ))
    local minutes=$(( elapsed / 60 ))
    local seconds=$(( elapsed % 60 ))

    echo ""
    echo -e "${GREEN}========================================================${NC}"
    echo -e "${GREEN}  环境配置完成! (耗时 ${minutes}分${seconds}秒)${NC}"
    echo -e "${GREEN}========================================================${NC}"
    echo ""
    echo "  开始训练:"
    echo "    tmux new -s train"
    echo "    bash train.sh start"
    echo ""
    echo "  数据盘文件 (实例释放后保留):"
    echo "    数据集:     $DATA_DIR"
    echo "    Checkpoint: $CHECKPOINT_DIR"
    echo "    LibTorch:   $LIBTORCH_DIR"
    echo ""
}

# ============================================================================
# 开始训练
# ============================================================================
start() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  开始训练${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""

    if [ ! -f "$EXE" ]; then
        log_error "训练程序未找到, 请先运行 'bash train.sh setup'"
        exit 1
    fi

    export LD_LIBRARY_PATH="$LIBTORCH_DIR/lib:$LD_LIBRARY_PATH"

    # 使用 AutoDL 专用配置 (如果存在)
    local use_config="$PROJECT_DIR/config/train_config_autodl.ini"
    if [ ! -f "$use_config" ]; then
        use_config="$CONFIG"
    fi

    # 显示 GPU 信息
    log_info "GPU 信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu --format=csv,noheader 2>/dev/null || echo "  无法获取"
    echo ""

    # 检查数据是否就绪
    local train_count
    train_count=$(find "$DATA_DIR/train/HR" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
    log_info "训练数据: $train_count 张图像"
    if [ "$train_count" -lt 100 ]; then
        log_error "训练数据不足, 请先准备数据集"
        log_info "运行: bash train.sh setup"
        exit 1
    fi

    log_info "配置文件: $use_config"
    log_info "Checkpoint: $CHECKPOINT_DIR"
    echo ""

    # 启动训练
    "$EXE" --config "$use_config" "$@"
}

# ============================================================================
# 恢复训练
# ============================================================================
resume() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  恢复训练${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""

    if [ ! -d "$CHECKPOINT_DIR" ] || [ -z "$(ls -A "$CHECKPOINT_DIR" 2>/dev/null)" ]; then
        log_error "未找到 checkpoint 文件"
        log_info "目录: $CHECKPOINT_DIR"
        exit 1
    fi

    # 如果编译产物不存在 (系统盘被清空), 自动重编译
    if [ ! -f "$EXE" ]; then
        log_warn "训练程序不存在, 自动重新编译..."
        build
    fi

    log_info "找到 checkpoint:"
    ls -lh "$CHECKPOINT_DIR"/*.pt 2>/dev/null | tail -5
    echo ""

    export LD_LIBRARY_PATH="$LIBTORCH_DIR/lib:$LD_LIBRARY_PATH"

    local use_config="$PROJECT_DIR/config/train_config_autodl.ini"
    if [ ! -f "$use_config" ]; then
        use_config="$CONFIG"
    fi

    "$EXE" --config "$use_config" --resume "$CHECKPOINT_DIR" "$@"
}

# ============================================================================
# 下载训练结果
# ============================================================================
download() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  下载训练结果${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    echo "在本地电脑执行以下命令下载模型:"
    echo ""
    echo "# 1. 查看 AutoDL SSH 信息 (在 AutoDL 控制台获取)"
    echo "#    格式: ssh -p <端口> root@<地址>"
    echo ""
    echo "# 2. 下载最优模型 (推理只需这个)"
    echo "scp -P <端口> root@<地址>:$CHECKPOINT_DIR/generator_best.pt      ./checkpoints/"
    echo "scp -P <端口> root@<地址>:$CHECKPOINT_DIR/discriminator_best.pt  ./checkpoints/"
    echo ""
    echo "# 3. 下载所有 checkpoint (可选)"
    echo "scp -P <端口> -r root@<地址>:$CHECKPOINT_DIR/ ./checkpoints_cloud/"
    echo ""
    echo "# 4. 下载验证结果图 (可选)"
    echo "scp -P <端口> -r root@<地址>:$RESULT_DIR/ ./results_cloud/"
    echo ""

    # 显示当前 checkpoint 状态
    if [ -d "$CHECKPOINT_DIR" ]; then
        echo "当前 Checkpoint 文件:"
        ls -lh "$CHECKPOINT_DIR"/*.pt 2>/dev/null || echo "  暂无"
        echo ""
        # 显示最优模型的修改时间
        if [ -f "$CHECKPOINT_DIR/generator_best.pt" ]; then
            local best_time
            best_time=$(stat -c '%y' "$CHECKPOINT_DIR/generator_best.pt" 2>/dev/null | cut -d. -f1)
            echo "  最优模型更新时间: $best_time"
        fi
    fi
}

# ============================================================================
# 系统状态
# ============================================================================
status() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  系统状态 (AutoDL)${NC}"
    echo -e "${CYAN}========================================${NC}"

    echo ""
    echo "=== GPU 状态 ==="
    nvidia-smi 2>/dev/null || echo "nvidia-smi 不可用"

    echo ""
    echo "=== 磁盘空间 ==="
    echo "系统盘:"
    df -h /root 2>/dev/null | tail -1
    echo "数据盘:"
    df -h /root/autodl-tmp 2>/dev/null | tail -1 || echo "  autodl-tmp 不可用"

    echo ""
    echo "=== 数据集 ==="
    local train_c val_c test_c
    train_c=$(find "$DATA_DIR/train/HR" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
    val_c=$(find "$DATA_DIR/val/HR" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
    test_c=$(find "$DATA_DIR/test/HR" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
    echo "  训练集: $train_c 张"
    echo "  验证集: $val_c 张"
    echo "  测试集: $test_c 张"

    echo ""
    echo "=== Checkpoint ==="
    if [ -d "$CHECKPOINT_DIR" ]; then
        ls -lh "$CHECKPOINT_DIR"/*.pt 2>/dev/null || echo "  暂无 checkpoint"
        if [ -f "$CHECKPOINT_DIR/train_state.bin" ]; then
            echo "  训练状态文件存在 (可恢复训练)"
        fi
    else
        echo "  暂无 checkpoint 目录"
    fi

    echo ""
    echo "=== 训练进程 ==="
    pgrep -a facesr_train 2>/dev/null || echo "  无训练进程运行"

    echo ""
    echo "=== 环境 ==="
    echo "  cmake: $(cmake --version 2>/dev/null | head -1 || echo '未安装')"
    echo "  训练程序: $([ -f "$EXE" ] && echo '已编译' || echo '未编译')"
    echo "  LibTorch: $([ -d "$LIBTORCH_DIR" ] && echo '已安装' || echo '未安装')"
}

# ============================================================================
# 帮助
# ============================================================================
help() {
    echo ""
    echo -e "${CYAN}FaceSR AutoDL 云端训练脚本 (国内镜像加速版)${NC}"
    echo ""
    echo "用法: bash train.sh <命令>"
    echo ""
    echo "命令:"
    echo "  setup    - 一键全自动配置 (安装依赖 + 编译 + 下载数据集)"
    echo "  build    - 仅编译训练程序"
    echo "  start    - 开始训练"
    echo "  resume   - 从 checkpoint 恢复训练 (自动重编译)"
    echo "  data     - 仅下载并准备数据集"
    echo "  download - 显示下载模型的命令"
    echo "  status   - 查看 GPU/磁盘/训练状态"
    echo "  help     - 显示帮助"
    echo ""
    echo -e "${GREEN}====== 快速开始 (只需3条命令) ======${NC}"
    echo ""
    echo "  # 1. 一键配置 (首次, 全自动)"
    echo "  bash train.sh setup"
    echo ""
    echo "  # 2. 在 tmux 中训练"
    echo "  tmux new -s train"
    echo "  bash train.sh start"
    echo "  # Ctrl+B, D 断开 (训练继续运行)"
    echo ""
    echo "  # 3. 训练完成后下载模型"
    echo "  bash train.sh download"
    echo ""
    echo -e "${YELLOW}====== 重新开机后 ======${NC}"
    echo ""
    echo "  bash train.sh resume  # 自动重编译 + 从断点恢复"
    echo ""
    echo "====== 国内镜像配置 ======"
    echo ""
    echo "  pip:          $PIP_MIRROR"
    echo "  HuggingFace:  $HF_MIRROR"
    echo "  CMake:        $CMAKE_MIRROR"
    echo ""
    echo "====== AutoDL 数据盘 (实例释放后保留) ======"
    echo ""
    echo "  数据集:     $DATA_DIR"
    echo "  Checkpoint: $CHECKPOINT_DIR"
    echo "  LibTorch:   $LIBTORCH_DIR"
    echo ""
}

# ============================================================================
# 入口
# ============================================================================

case "${1:-help}" in
    setup)    setup ;;
    build)    build ;;
    start)    shift; start "$@" ;;
    resume)   shift; resume "$@" ;;
    data)     prepare_data ;;
    download) download ;;
    status)   status ;;
    help)     help ;;
    *)
        echo "未知命令: $1"
        help
        exit 1
        ;;
esac
