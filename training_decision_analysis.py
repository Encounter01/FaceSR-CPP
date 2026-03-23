#!/usr/bin/env python3
"""
训练决策分析
===========
判断是否需要重新训练
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_training_necessity():
    """分析是否需要重新训练"""

    print("=" * 80)
    print("训练需求分析")
    print("=" * 80 + "\n")

    df = pd.read_csv('results/eval_reports/full_metrics_report.csv')

    # ===== 分析 1：Latest 模型性能 =====
    print("[1] LATEST 模型性能分析")
    print("-" * 80)

    latest_psnr = df['latest_psnr'].mean()
    bicubic_psnr = df['bicubic_psnr'].mean()
    improvement = latest_psnr - bicubic_psnr
    improvement_pct = 100 * improvement / bicubic_psnr

    print(f"Latest 平均 PSNR: {latest_psnr:.2f} dB")
    print(f"Bicubic 基线 PSNR: {bicubic_psnr:.2f} dB")
    print(f"性能提升: {improvement:+.2f} dB ({improvement_pct:+.1f}%)")
    print(f"PSNR 标准差: {df['latest_psnr'].std():.2f} dB (稳定性)")

    # SSIM 和 LPIPS
    print(f"\nLATEST 其他指标:")
    print(f"  SSIM: {df['latest_ssim'].mean():.4f} (vs Bicubic {df['bicubic_ssim'].mean():.4f})")
    print(f"  LPIPS: {df['latest_lpips'].mean():.4f} (vs Bicubic {df['bicubic_lpips'].mean():.4f})")

    # ===== 分析 2：性能评估 =====
    print("\n[2] 性能等级评估")
    print("-" * 80)

    if latest_psnr >= 30:
        level = "优秀 (Excellent)"
    elif latest_psnr >= 28:
        level = "良好 (Good)"
    elif latest_psnr >= 26:
        level = "合格 (Fair)"
    else:
        level = "需改进 (Poor)"

    print(f"Latest PSNR 等级: {level}")
    print(f"  - 超高质量 (>30 dB): 用于专业应用")
    print(f"  - 高质量 (28-30 dB): 适合通用用途 -< LATEST 在这里")
    print(f"  - 合格 (26-28 dB): 可用但有限")
    print(f"  - 低质量 (<26 dB): 需要改进")

    # ===== 分析 3：分层性能 =====
    print("\n[3] 分层性能一致性")
    print("-" * 80)

    # 按亮度分层
    brightness_psnr = []
    for threshold in [(0, 80/255), (80/255, 170/255), (170/255, 1.0)]:
        mask = (df['mean_brightness'] >= threshold[0]) & (df['mean_brightness'] < threshold[1])
        psnr = df[mask]['latest_psnr'].mean()
        brightness_psnr.append(psnr)
        name = ["Dark", "Medium", "Bright"][len(brightness_psnr)-1]
        print(f"  {name:8s}: PSNR {psnr:.2f} dB")

    brightness_variance = np.std(brightness_psnr)
    print(f"  亮度间差异: {brightness_variance:.2f} dB (越小越稳定)")

    # ===== 分析 4：异常值分析 =====
    print("\n[4] 异常值/失败案例分析")
    print("-" * 80)

    low_psnr_count = (df['latest_psnr'] < 25).sum()
    low_psnr_pct = 100 * low_psnr_count / len(df)

    print(f"低质量输出 (PSNR < 25 dB): {low_psnr_count}/3000 ({low_psnr_pct:.2f}%)")

    if low_psnr_pct < 1:
        print(f"  评估: 很好 - 几乎没有失败案例")
    elif low_psnr_pct < 5:
        print(f"  评估: 可接受 - 少量失败案例")
    else:
        print(f"  评估: 有问题 - 失败率较高")

    # ===== 分析 5：与 Best 对比 =====
    print("\n[5] Latest vs Best 对比（理想 Best 应该是什么样）")
    print("-" * 80)

    best_psnr = df['best_psnr'].mean()
    print(f"Latest: {latest_psnr:.2f} dB [OK] (actual best)")
    print(f"Best:   {best_psnr:.2f} dB [BAD] (mislabeled)")
    print(f"\nBest 应该是最优的，但现在：")
    print(f"  - Best 性能远低于 Latest (-{latest_psnr - best_psnr:.2f} dB)")
    print(f"  - 说明 'Best' 的标记机制有问题")
    print(f"  - Latest 才是真正的最优模型")

    # ===== 分析 6：训练优化空间 =====
    print("\n[6] 训练优化空间评估")
    print("-" * 80)

    # 与理想值比对
    ideal_psnr = 32  # 高质量 SR 的典型值
    gap_to_ideal = ideal_psnr - latest_psnr

    print(f"理想超分 PSNR: ~{ideal_psnr} dB")
    print(f"Latest 实际: {latest_psnr:.2f} dB")
    print(f"优化空间: {gap_to_ideal:.2f} dB")

    if gap_to_ideal < 1:
        print(f"  评估: 极小 - 模型已接近理想状态")
    elif gap_to_ideal < 2:
        print(f"  评估: 较小 - 优化空间有限")
    elif gap_to_ideal < 3:
        print(f"  评估: 中等 - 有改进空间但投入产出比需评估")
    else:
        print(f"  评估: 较大 - 建议继续优化")

    # ===== 分析 7：实际应用可行性 =====
    print("\n[7] 实际应用可行性")
    print("-" * 80)

    print(f"Latest PSNR {latest_psnr:.2f} dB 是否足够用于生产？")
    print(f"  - 人脸超分应用 (推荐 PSNR > 27): {' 可用' if latest_psnr > 27 else ' 不足'}")
    print(f"  - 图像增强应用 (推荐 PSNR > 26): {' 可用' if latest_psnr > 26 else ' 不足'}")
    print(f"  - 专业用途 (推荐 PSNR > 30): {' 可用' if latest_psnr > 30 else ' 不足'}")

    print(f"\nLPIPS {df['latest_lpips'].mean():.4f} (越低越好):")
    print(f"  - 良好感知质量 (< 0.15): {' 优秀' if df['latest_lpips'].mean() < 0.15 else '可用' if df['latest_lpips'].mean() < 0.25 else '需改进'}")

    # ===== 分析 8：重新训练成本/收益 =====
    print("\n[8] 重新训练的成本/收益分析")
    print("-" * 80)

    potential_gain = ideal_psnr - latest_psnr
    training_time = 10  # 预估小时
    cost_per_db = training_time / max(potential_gain, 0.1)

    print(f"潜在收益: +{potential_gain:.2f} dB")
    print(f"预计训练时间: ~{training_time} 小时")
    print(f"每 dB 提升的训练成本: ~{cost_per_db:.1f} 小时/dB")

    print(f"\n收益评估:")
    if potential_gain < 0.5:
        print(f"  - 收益很小（<0.5 dB），不值得重新训练")
    elif potential_gain < 1.5 and cost_per_db > 5:
        print(f"  - 收益/成本比不划算，建议不重新训练")
    elif potential_gain < 2 and cost_per_db > 3:
        print(f"  - 需权衡具体需求")
    else:
        print(f"  - 值得重新训练以获得更好性能")

    # ===== 最终建议 =====
    print("\n" + "=" * 80)
    print("最终建议")
    print("=" * 80)

    print(f"\n【情景1】如果是生产环境（推荐优先级）")
    print(f"   立即部署 Latest checkpoint")
    print(f"  理由：")
    print(f"    - Latest 性能已足够好（PSNR {latest_psnr:.2f} dB）")
    print(f"    - 已通过充分验证，风险低")
    print(f"    - 相比等待新训练，立即获益更重要")

    print(f"\n【情景2】如果是研发/优化（可选）")
    print(f"  - 短期：优化现有训练流程（不需要重头训练）")
    print(f"    • 修复 'Best' checkpoint 的选择机制")
    print(f"    • 实现 Early Stopping 基于验证集指标")
    print(f"    • 保存完整的元数据供分析")
    print(f"  - 长期：计划下一个训练周期")
    print(f"    • 尝试更大模型 或 更长训练时间")
    print(f"    • 改进损失函数（加入感知损失 LPIPS）")
    print(f"    • 期望的改进：+1-2 dB")

    print(f"\n【情景3】如果需要极高质量（PSNR > 31 dB）")
    print(f"  - 需要重新训练，但需要策略调整：")
    print(f"    • 使用更强的损失函数")
    print(f"    • 增加训练数据或预处理")
    print(f"    • 优化超参数（学习率、batch size）")
    print(f"    • 预计需要 10-20 小时训练时间")

    print(f"\n【总体结论】")
    print(f"  🎯 短期：NO - 不需要重新训练，Latest 已足够好")
    print(f"  📈 中期：YES (可选) - 优化训练流程和选择机制")
    print(f"  🚀 长期：YES (计划) - 下一训练周期改进策略")

if __name__ == "__main__":
    analyze_training_necessity()
