#!/usr/bin/env python3
"""
分析 Checkpoint 质量
====================
目标：加载每个 checkpoint，检查其权重统计和训练元数据
"""

import torch
from pathlib import Path
import sys

def analyze_checkpoint(ckpt_path):
    """分析单个 checkpoint"""

    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        info = {
            'path': str(ckpt_path),
            'keys': list(checkpoint.keys()),
            'size_mb': ckpt_path.stat().st_size / (1024*1024)
        }

        # 检查模型权重
        if 'generator_state_dict' in checkpoint:
            state_dict = checkpoint['generator_state_dict']

            # 计算权重统计
            total_params = 0
            weight_stats = {}

            for name, param in state_dict.items():
                total_params += param.numel()
                if param.numel() > 0:
                    weight_stats[name] = {
                        'shape': tuple(param.shape),
                        'min': param.min().item(),
                        'max': param.max().item(),
                        'mean': param.mean().item(),
                        'std': param.std().item(),
                    }

            info['total_params'] = total_params
            info['num_layers'] = len(state_dict)

            # 前 5 层的权重统计
            sample_layers = list(weight_stats.items())[:3]
            info['sample_layers'] = sample_layers

        # 检查是否有训练信息
        if 'epoch' in checkpoint:
            info['epoch'] = checkpoint['epoch']
        if 'best_loss' in checkpoint:
            info['best_loss'] = checkpoint['best_loss']
        if 'train_losses' in checkpoint:
            losses = checkpoint['train_losses']
            if isinstance(losses, list) and len(losses) > 0:
                info['final_loss'] = losses[-1]
                info['num_epochs'] = len(losses)

        return info

    except Exception as e:
        return {'path': str(ckpt_path), 'error': str(e)}


def main():
    print("=" * 80)
    print("Checkpoint 质量分析")
    print("=" * 80 + "\n")

    ckpt_dir = Path("checkpoints")
    checkpoints = {
        'best': ckpt_dir / "generator_best.pt",
        'latest': ckpt_dir / "generator_latest.pt",
        'epoch190': ckpt_dir / "generator_epoch190.pt",
    }

    results = {}

    for name, path in checkpoints.items():
        print(f"【{name.upper()}】{path.name}")
        print("-" * 80)

        info = analyze_checkpoint(path)
        results[name] = info

        if 'error' in info:
            print(f"ERROR: {info['error']}\n")
            continue

        print(f"文件大小: {info['size_mb']:.1f} MB")
        print(f"Checkpoint 包含的键: {', '.join(info['keys'])}")

        if 'total_params' in info:
            print(f"总参数数: {info['total_params']:,}")
            print(f"模型层数: {info['num_layers']}")

        if 'epoch' in info:
            print(f"训练 Epoch: {info['epoch']}")

        if 'best_loss' in info:
            print(f"最佳 Loss: {info['best_loss']:.6f}")

        if 'final_loss' in info:
            print(f"最终 Loss: {info['final_loss']:.6f}")

        if 'num_epochs' in info:
            print(f"记录的 Loss 数量: {info['num_epochs']}")

        if 'sample_layers' in info:
            print(f"\n权重统计 (前 3 层):")
            for layer_name, stats in info['sample_layers']:
                print(f"  {layer_name}:")
                print(f"    形状: {stats['shape']}")
                print(f"    值范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"    均值: {stats['mean']:.4f}, 标准差: {stats['std']:.4f}")

        print()

    # ===== 对比分析 =====
    print("=" * 80)
    print("对比分析")
    print("=" * 80 + "\n")

    print("参数检查:")
    for name in ['best', 'latest', 'epoch190']:
        if 'total_params' in results[name]:
            params = results[name]['total_params']
            print(f"  {name:10s}: {params:,} 参数")

    print("\n关键观察:")
    if results['best'].get('final_loss') and results['latest'].get('final_loss'):
        best_loss = results['best'].get('final_loss', float('inf'))
        latest_loss = results['latest'].get('final_loss', float('inf'))

        print(f"  Best 最终 Loss:   {best_loss:.6f}")
        print(f"  Latest 最终 Loss: {latest_loss:.6f}")

        if best_loss > latest_loss:
            print(f"  >>> Latest 训练损失更低（更好）")
        else:
            print(f"  >>> Best 训练损失更低（更好）")

    print("\n结论:")
    print("  - 检查所有 checkpoint 的参数数和结构是否一致")
    print("  - 验证权重值范围是否正常（非零、无 NaN）")
    print("  - 对比 Best vs Latest 的训练损失趋势")

if __name__ == "__main__":
    main()
