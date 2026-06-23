# Usage Guide

## Training

Run with the default configuration:

```bat
build\bin\Release\facesr_train.exe --config config\train_config.ini
```

Override selected settings:

```bat
build\bin\Release\facesr_train.exe ^
  --config config\train_config.ini ^
  --batch-size 8 ^
  --epochs 100
```

Use CPU only:

```bat
build\bin\Release\facesr_train.exe --config config\train_config.ini --cpu
```

Resume from checkpoint files:

```bat
build\bin\Release\facesr_train.exe --config config\train_config.ini --resume checkpoints
```

## Inference

Single image:

```bat
build\bin\Release\facesr_test.exe ^
  --model checkpoints\generator_epoch190.pt ^
  --input input.jpg ^
  --output output.png
```

Folder:

```bat
build\bin\Release\facesr_test.exe ^
  --model checkpoints\generator_epoch190.pt ^
  --input input_folder ^
  --output output_folder
```

CPU only:

```bat
build\bin\Release\facesr_test.exe --model checkpoints\generator_epoch190.pt --input input.jpg --cpu
```

Attention-enabled native model:

```bat
build\bin\Release\facesr_test.exe ^
  --model checkpoints\generator_epoch190.pt ^
  --input input.jpg ^
  --output output.png ^
  --attention
```

When `--output` is omitted, single-image output defaults to `name_sr.ext`; folder output defaults to `input_folder_sr`.

## GUI

```bat
build\bin\Release\facesr_gui_app.exe
```

GUI visual settings are read from `config/ui_config.ini`.

## Evaluation Utilities

Python scripts under `scripts/evaluation/`, `scripts/analysis/`, and `scripts/ablation/` are optional local utilities. They are not required for the C++ build or inference workflow.

Use them only with local checkpoints and datasets that match the script assumptions.
