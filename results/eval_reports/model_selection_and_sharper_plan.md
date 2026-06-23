# 当前模型结论与减糊方案

## 1. 当前推理默认推荐

在 `data/test/HR` 的 3000 张测试图上重新评估后，当前最值得优先用于推理的权重不是
`generator_best.pt`，而是 `generator_epoch190.pt`。

平均结果如下：

| checkpoint | PSNR | SSIM |
| --- | ---: | ---: |
| bicubic | 28.463390 | 0.819148 |
| generator_best.pt | 29.441479 | 0.840618 |
| generator_latest.pt | 29.258062 | 0.832859 |
| generator_epoch190.pt | 29.552015 | 0.843166 |

因此，当前推荐：

```bash
facesr_test --model checkpoints/generator_epoch190.pt --input image.jpg --output result.png
```

## 2. 为什么还是会感觉模糊

主要原因不是单点问题，而是几项因素叠加：

1. 任务本身是 `64x64 -> 256x256` 的 4x 人脸超分，信息缺失很严重，模型只能“猜”细节。
2. 当前主训练配置 `config/train_config.ini` 偏保守，容易得到更平滑的重建：
   - `frequency_weight = 0.0`
   - `gradient_weight = 0.0`
   - `use_attention = false`
   - `use_spectral_norm = false`
   - `vgg_weights_path =` 为空
3. 现有 `generator_best.pt` 是按验证 `PSNR` 选出来的，`PSNR` 往往更偏好平滑结果，不一定更锐利。

## 3. 具体减糊训练方案

仓库里已经补齐了一套更适合“提细节”的配置：

- `config/train_config_sharper.ini`
- `config/finetune_phase_a.ini`
- `config/finetune_phase_b.ini`
- `config/finetune_phase_c.ini`

核心思路：

- 保留 `pixel_weight = 1.0`
- 保留 `perceptual_weight = 1.0`
- 使用 `gan_weight = 0.05`
- 打开 `frequency_weight = 0.1`
- 打开 `gradient_weight = 0.1`
- 使用 `gan_type = hinge`
- 使用 `r1_weight = 10.0`
- 打开 `use_attention = true`
- 打开 `use_spectral_norm = true`
- 指向现成的 `models/vgg19_features.pt`

## 4. 使用建议

- 当前直接推理：优先用 `checkpoints/generator_epoch190.pt`
- 想继续提细节：优先用 `config/train_config_sharper.ini`
- 想分阶段微调：用 `config/finetune_phase_a.ini` / `b` / `c`

补充说明：

- 这套方案的目标是“更锐利、少发糊”，不保证 `PSNR` 一定继续上升。
- 如果你更在意视觉观感而不是纯指标，这套配置比当前默认训练配置更合适。
