# Configuration Reference

Configuration files use INI syntax:

```ini
[section]
key = value
```

Comments can start with `#` or `;`.

## data

| Key | Description |
| --- | --- |
| `train_hr_dir` | Training high-resolution image directory. |
| `train_lr_dir` | Optional training low-resolution image directory. |
| `val_hr_dir` | Validation high-resolution image directory. |
| `val_lr_dir` | Optional validation low-resolution image directory. |
| `hr_size` | Target high-resolution image size. |
| `lr_size` | Target low-resolution image size. |
| `scale` | Upscaling factor. |

## training

| Key | Description |
| --- | --- |
| `batch_size` | Images per training step. |
| `num_epochs` | Number of epochs for the current run. |
| `num_workers` | Dataset loading worker count. |
| `lr_g` | Generator learning rate. |
| `lr_d` | Discriminator learning rate. |
| `phase1_epochs` | Boundary for pixel-only training. |
| `phase2_epochs` | Boundary for enabling perceptual loss before the full loss set. |

## loss

| Key | Description |
| --- | --- |
| `pixel_weight` | Pixel reconstruction loss weight. |
| `perceptual_weight` | Perceptual feature loss weight. |
| `gan_weight` | Adversarial loss weight. |
| `pixel_type` | Pixel loss type: `l1` or `l2`. |
| `gan_type` | GAN loss type: `vanilla`, `lsgan`, `wgan`, `wgan-gp`, or `hinge`. |
| `frequency_weight` | Reserved by some presets; verify implementation before relying on it. |
| `gradient_weight` | Reserved by some presets; verify implementation before relying on it. |
| `r1_weight` | R1 regularization weight. `0` disables it. |

## output

| Key | Description |
| --- | --- |
| `checkpoint_dir` | Directory for model checkpoints and training state. |
| `result_dir` | Directory for validation outputs. |
| `save_interval` | Save checkpoint every N epochs. |
| `val_interval` | Run validation every N epochs. |
| `log_interval` | Print training logs every N steps. |

## model

| Key | Description |
| --- | --- |
| `vgg_weights_path` | Optional VGG feature model path. |
| `use_attention` | Enables the CBAM attention module. Inference must use the same native-model layout. |
| `use_spectral_norm` | Reserved by some presets; verify implementation before relying on it. |

## GUI Configuration

`config/ui_config.ini` controls optional GUI visual settings:

- `general`: visual feature toggles.
- `background`: solid, gradient, or image background settings.
- `frosted`: blur, opacity, and tint settings.
- `theme`: primary, accent, text, border, and panel colors.
