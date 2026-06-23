# Build Guide

FaceSR-CPP uses CMake and builds these targets:

- `facesr_train`: command-line training executable.
- `facesr_test`: command-line inference executable.
- `facesr_gui_app`: optional Qt desktop executable.

## Dependencies

Install:

- CMake 3.18 or newer.
- A C++17-capable compiler.
- LibTorch 2.x.
- OpenCV 4.x.
- Qt 5.15+ or Qt 6.x for GUI builds.

LibTorch must match the compiler and CUDA runtime you intend to use. CPU-only LibTorch is acceptable for CPU-only workflows.

## Windows Example

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_PREFIX_PATH="C:/libtorch;C:/Qt/6.10.1/msvc2022_64;C:/opencv/build"

cmake --build build --config Release
```

Disable the GUI:

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
  -DBUILD_GUI=OFF ^
  -DCMAKE_PREFIX_PATH="C:/libtorch;C:/opencv/build"

cmake --build build --config Release --parallel
```

## Helper Scripts

Windows:

```bat
scripts\train.bat build
```

Linux or remote environments:

```bash
bash scripts/train.sh
```

## Output Paths

Visual Studio generators usually place executables under:

```text
build/bin/Release/
```

Single-configuration generators usually place executables under:

```text
build/bin/
```

## Troubleshooting

If CMake cannot find LibTorch, set `CMAKE_PREFIX_PATH` to the LibTorch root directory or set `Torch_DIR` to the directory containing `TorchConfig.cmake`.

If OpenCV is not found, set `OpenCV_DIR` or add the OpenCV package root to `CMAKE_PREFIX_PATH`.

If Qt is not found, install Qt development components and add the Qt CMake package path to `CMAKE_PREFIX_PATH`, or configure with `-DBUILD_GUI=OFF`.
