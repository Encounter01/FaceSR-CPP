#!/bin/bash
# ============================================================================
# autodl_setup.sh — AutoDL 环境一键编译脚本
#
# 用法（在 AutoDL 终端执行，需在项目根目录）:
#   cd /root/FaceSR_CPP
#   bash scripts/autodl_setup.sh
# ============================================================================

set -e

# ---------- 1. 自动定位 LibTorch ----------
echo "=== 检测 LibTorch 路径 ==="

TORCH_CMAKE_PREFIX=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null || true)

if [ -z "$TORCH_CMAKE_PREFIX" ]; then
    TORCH_INSTALL=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))" 2>/dev/null || true)
    TORCH_CMAKE_PREFIX="$TORCH_INSTALL"
fi

if [ -z "$TORCH_CMAKE_PREFIX" ]; then
    echo "ERROR: 找不到 PyTorch/LibTorch。请确认已激活正确的 conda 环境。"
    echo "手动设置方式: export CMAKE_PREFIX_PATH=/path/to/libtorch && bash scripts/autodl_setup.sh"
    exit 1
fi

echo "LibTorch 路径: $TORCH_CMAKE_PREFIX"

# ---------- 2. 检查 OpenCV ----------
echo ""
echo "=== 检测 OpenCV ==="
if pkg-config --exists opencv4 2>/dev/null; then
    echo "OpenCV: $(pkg-config --modversion opencv4)"
else
    echo "未检测到 OpenCV，尝试安装..."
    apt-get install -y libopencv-dev 2>/dev/null || pip install opencv-python-headless
fi

# ---------- 3. CMake 配置 ----------
echo ""
echo "=== CMake 配置 ==="

mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX" \
    -DBUILD_GUI=OFF

# ---------- 4. 编译 ----------
echo ""
echo "=== 编译（使用 $(nproc) 核心）==="
make -j$(nproc)

cd ..

echo ""
echo "=== 编译完成 ==="
echo "可执行文件: build/bin/facesr_train  build/bin/facesr_test"
echo ""
echo "快速验证:"
echo "  ./build/bin/facesr_train --help"
echo ""
echo "运行消融实验:"
echo "  ./build/bin/facesr_train --config config/ablations_quick/a1_l1_only_50e.ini"
echo "  ./build/bin/facesr_train --config config/ablations_quick/a2_l1_perceptual_50e.ini"
echo "  ./build/bin/facesr_train --config config/ablations_quick/a4_three_stage_50e.ini"
