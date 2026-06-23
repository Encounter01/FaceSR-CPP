# FaceSR-CPP

FaceSR-CPP is a C++ face image super-resolution toolkit built with LibTorch, OpenCV, and an optional Qt desktop interface. It provides command-line training, single-image inference, batch inference, and GUI-based image enhancement.

This repository is maintained as an engineering-oriented open source project. The public entry points focus on buildability, usage, configuration, maintainability, and contribution workflow.

## Features

- C++17 implementation with LibTorch neural network modules.
- OpenCV-based image loading, preprocessing, postprocessing, and output.
- Command-line tools for training and inference.
- Optional Qt Widgets desktop GUI.
- INI-based configuration for data paths, training options, loss weights, outputs, and model options.
- Checkpoint save and resume support.
- Python utility scripts for local evaluation, analysis, and ablation workflows.

## Requirements

| Dependency | Recommended Version | Required For |
| --- | --- | --- |
| CMake | 3.18+ | Build configuration |
| MSVC / C++ compiler | C++17 capable | All C++ targets |
| LibTorch | 2.x | Training and inference |
| OpenCV | 4.x | Image I/O and preprocessing |
| Qt | 5.15+ or 6.x | Optional GUI |
| CUDA | Matching LibTorch build | Optional GPU acceleration |

Validated Windows setups typically use Visual Studio 2022, CUDA-enabled LibTorch, OpenCV 4.x, and Qt 6.x.

## Build

Configure with explicit dependency paths:

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_PREFIX_PATH="C:/libtorch;C:/Qt/6.10.1/msvc2022_64;C:/opencv/build"
```

Build all targets:

```bat
cmake --build build --config Release
```

Build command-line targets only:

```bat
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_GUI=OFF ^
  -DCMAKE_PREFIX_PATH="C:/libtorch;C:/opencv/build"
cmake --build build --config Release --parallel
```

See [Build Guide](docs/build.md) for setup details and troubleshooting.

## Usage

### Training

```bat
build\bin\Release\facesr_train.exe --config config\train_config.ini
```

Override selected settings:

```bat
build\bin\Release\facesr_train.exe ^
  --train-hr data\train\HR ^
  --val-hr data\val\HR ^
  --batch-size 8 ^
  --epochs 100
```

Resume from a checkpoint directory:

```bat
build\bin\Release\facesr_train.exe --config config\train_config.ini --resume checkpoints
```

### Inference

Process one image:

```bat
build\bin\Release\facesr_test.exe ^
  --model checkpoints\generator_epoch190.pt ^
  --input input.jpg ^
  --output output.png
```

Process a folder:

```bat
build\bin\Release\facesr_test.exe ^
  --model checkpoints\generator_epoch190.pt ^
  --input input_folder ^
  --output output_folder
```

Use CPU inference:

```bat
build\bin\Release\facesr_test.exe --model checkpoints\generator_epoch190.pt --input input.jpg --cpu
```

If the model was trained with CBAM attention enabled, pass `--attention` during C++ native-weight inference.

### GUI

```bat
build\bin\Release\facesr_gui_app.exe
```

The GUI target is available when Qt is found and `BUILD_GUI=ON`.

See [Usage Guide](docs/usage.md) for complete command examples.

## Data And Checkpoints

Large datasets, generated outputs, and training checkpoint directories are not tracked in Git. Pretrained model weights are distributed through GitHub Releases instead of the repository history.

Recommended release assets:

- `facesr_a4_best_psnr28.6019.pt`: recommended final A4 model.
- `facesr_a4_best_cpu_forward.pt`: CPU-forward-compatible final A4 model.
- `generator_epoch190.pt`: optional legacy generator checkpoint.
- `generator_epoch190_traced.pt`: optional TorchScript legacy generator checkpoint.

Download the required `.pt` files from the `v1.0-models` release and place them under `checkpoints/` or pass their paths with `--model`.

Default training layout:

```text
data/
  train/
    HR/
    LR/
  val/
    HR/
    LR/
```

Default checkpoint layout:

```text
checkpoints/
  generator_epoch190.pt
  generator_best.pt
```

When LR directories are omitted in configuration, the dataset loader can generate LR images from HR images at load time.

## Configuration

Primary runtime configuration:

- `config/train_config.ini`: default training configuration.
- `config/train_config_sharper.ini`: sharper-output training preset.
- `config/finetune_phase_*.ini`: staged fine-tuning presets.
- `config/ui_config.ini`: GUI visual configuration.

See [Configuration Reference](docs/configuration.md).

## Repository Layout

```text
FaceSR-CPP/
  CMakeLists.txt
  CMakePresets.json
  include/                 Public headers
  src/                     C++ implementation
  config/                  Runtime and training configuration
  scripts/                 Build, training, evaluation, and analysis utilities
  models/                  Small auxiliary model assets
  checkpoints/             Local checkpoints, not tracked except .gitkeep
  docs/                    Project documentation
```

## Documentation

- [Build Guide](docs/build.md)
- [Usage Guide](docs/usage.md)
- [Configuration Reference](docs/configuration.md)
- [Maintainer Notes](docs/maintainer_notes.md)
- [Changelog](CHANGELOG.md)

## Contributing

Issues and pull requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes.

## License

This project is released under the [CC BY-NC-SA 4.0](LICENSE) license.

Dataset and third-party assets may have their own licenses. Verify the relevant dataset and model terms before redistribution.
