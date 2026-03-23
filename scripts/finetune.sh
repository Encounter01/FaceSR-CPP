#!/bin/bash
# ============================================================================
# finetune.sh — 三阶段微调训练脚本
#
# 用法:
#   cd FaceSR_CPP
#   bash scripts/finetune.sh
#
# 前提:
#   1. 已编译: mkdir build && cd build && cmake .. && make -j$(nproc)
#   2. 已有 best checkpoint: checkpoints/generator_best.pt
#   3. 数据在 data/train/hr 和 data/val/hr
# ============================================================================

set -e  # 出错即停

TRAIN_BIN="./build/facesr_train"
CHECKPOINT_DIR="checkpoints"

# 检查可执行文件
if [ ! -f "$TRAIN_BIN" ]; then
    echo "[错误] 找不到 $TRAIN_BIN，请先编译项目"
    echo "  mkdir -p build && cd build && cmake .. && make -j\$(nproc)"
    exit 1
fi

# 检查 best checkpoint
if [ ! -f "$CHECKPOINT_DIR/generator_best.pt" ]; then
    echo "[错误] 找不到 $CHECKPOINT_DIR/generator_best.pt"
    exit 1
fi

echo "============================================"
echo "  FaceSR 三阶段微调训练"
echo "============================================"
echo ""

# ============================================================================
# 阶段A: Pixel + Frequency + Gradient (20 epochs, lr=1e-4)
# ============================================================================
echo "[阶段A] Pixel + Frequency + Gradient — 20 epochs, lr=1e-4"
echo "  从 best checkpoint 开始微调..."
echo ""

$TRAIN_BIN --config config/finetune_phase_a.ini --resume $CHECKPOINT_DIR

# 检查是否正常完成
if [ $? -ne 0 ]; then
    echo "[错误] 阶段A训练失败"
    exit 1
fi
echo ""
echo "[阶段A] 完成！"
echo ""

# ============================================================================
# 阶段B: + Perceptual (30 epochs, lr=5e-5)
# ============================================================================
echo "[阶段B] + Perceptual — 30 epochs, lr=5e-5"
echo "  从阶段A的 latest checkpoint 继续..."
echo ""

$TRAIN_BIN --config config/finetune_phase_b.ini --resume $CHECKPOINT_DIR

if [ $? -ne 0 ]; then
    echo "[错误] 阶段B训练失败"
    exit 1
fi
echo ""
echo "[阶段B] 完成！"
echo ""

# ============================================================================
# 阶段C: + GAN(hinge) + R1 (50 epochs, lr=2e-5)
# ============================================================================
echo "[阶段C] + GAN(hinge) + R1 — 50 epochs, lr=2e-5"
echo "  从阶段B的 latest checkpoint 继续..."
echo ""

$TRAIN_BIN --config config/finetune_phase_c.ini --resume $CHECKPOINT_DIR

if [ $? -ne 0 ]; then
    echo "[错误] 阶段C训练失败"
    exit 1
fi
echo ""
echo "[阶段C] 完成！"
echo ""

# ============================================================================
# 训练完成总结
# ============================================================================
echo "============================================"
echo "  三阶段微调训练全部完成！"
echo "============================================"
echo ""
echo "模型保存位置:"
echo "  最新模型: $CHECKPOINT_DIR/generator_latest.pt"
echo "  最佳模型: $CHECKPOINT_DIR/generator_best.pt"
echo ""
echo "下一步: 运行推理对比效果"
echo "  ./build/facesr_test --model $CHECKPOINT_DIR/generator_best.pt --input data/val/hr --output results/finetuned --attention"
echo ""
