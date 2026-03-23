#!/bin/bash
# ============================================================================
# autodl_setup.sh — AutoDL 环境一键编译脚本
#
# 用法（在 AutoDL 终端执行）:
#   cd /root/FaceSR_CPP
#   bash scripts/autodl_setup.sh
# ============================================================================

set -e

echo "=== 编译 FaceSR_CPP ==="

# 创建并进入 build 目录
mkdir -p build
cd build

# CMake 配置（Release 模式，关闭 GUI）
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_GUI=OFF

# 多线程编译
make -j$(nproc)

echo ""
echo "=== 编译完成 ==="
echo ""
echo "训练命令:"
echo "  cd /root/FaceSR_CPP && bash scripts/finetune.sh"
echo ""
