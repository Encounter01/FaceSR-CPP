# Contributing

Thanks for helping improve FaceSR-CPP.

## Setup

1. Install the dependencies listed in `docs/build.md`.
2. Configure the project with CMake.
3. Build at least the command-line targets before opening a pull request.

Example:

```bat
cmake -S . -B build -DBUILD_GUI=OFF -DCMAKE_PREFIX_PATH="C:/libtorch;C:/opencv/build"
cmake --build build --config Release --parallel
```

## Pull Request Guidelines

- Keep each pull request focused.
- Update documentation when changing build commands, CLI behavior, configuration keys, or runtime assumptions.
- Do not commit local datasets, generated results, logs, or checkpoints.
- Include the validation commands you ran.

## Code Guidelines

- Use C++17.
- Prefer existing project patterns.
- Keep errors actionable.
- Avoid broad formatting-only churn in functional changes.

## Bug Reports

Please include:

- Operating system and compiler.
- CMake configure command.
- LibTorch, OpenCV, Qt, and CUDA versions where relevant.
- Exact command that failed.
- Relevant log output.
