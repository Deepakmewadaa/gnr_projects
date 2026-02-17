# CustomDL – C++ Deep Learning Framework with Python Bindings

This project implements a custom deep learning framework in C++ with:

The framework can be used in two ways:
1. Pure C++ (train.exe / evaluate.exe)
2. Directly from Python (`import customdl`)

---

# System Requirements

- Windows 10 / 11
- Visual Studio 2026 Build Tools (C++ workload)
- CMake ≥ 3.20
- Python 3.10+
- Git
- vcpkg (for dependency management)
---

# Install vcpkg

Clone and bootstrap vcpkg:

```powershell
git clone https://github.com/microsoft/vcpkg
cd vcpkg
.\bootstrap-vcpkg.bat
```

```powershell
setx VCPKG_ROOT "C:\path\to\vcpkg"
```
---

# Clone This Repository

```powershell
git clone <your-repository-url>
cd CustomDL
```

---

# Build the Project (Manifest Mode)

⚠️ IMPORTANT: Run this from the project root (where `CMakeLists.txt` exists)

```powershell
cmake -S . -B build `
  -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-windows `
  -DVCPKG_FEATURE_FLAGS=manifests
```

Then build:

```powershell
cmake --build build --config Release
```

This will:
- Automatically install OpenCV
- Install pybind11
- Build train.exe
- Build evaluate.exe
- Build Python module `customdl.pyd`

---

# Verify Python Module

Go to the build output folder:

```powershell
cd build\Release
python
```

Inside Python:

```python
import customdl
```

#  Training from Python

```python
import customdl

accuracy = customdl.train(
    dataset_path="path/to/dataset",
    num_classes=10"
)


During training it prints:

- Dataset loading time
- Train/Test split sizes
- Total number of parameters
- Loss per epoch
- Train accuracy per epoch
- Test accuracy per epoch
- Model save confirmation

---

# Evaluating from Python

```python
accuracy = customdl.evaluate(
    dataset_path="path/to/hidden_dataset",
    weights_path="model.bin",
    num_classes=10
)

This prints:

- Dataset loading time
- Total parameters
- MACs
- FLOPs
- Final test accuracy

---

# Running Pure C++ Executables (Optional)

After building:

```powershell
build\Release\train.exe <dataset_path> <num_classes>
build\Release\evaluate.exe <dataset_path> <num_classes> <weights_path>
```

---

# Project Structure

```
framework/
    conv2d.cpp
    relu.cpp
    maxpool.cpp
    linear.cpp
    loss.cpp
    optimizer.cpp
    tensor.cpp
    dataset.cpp
    weight_io.cpp

train.cpp
evaluate.cpp
bindings.cpp
CMakeLists.txt
vcpkg.json
README.md
```

---

# Dataset Format

Dataset folder should be structured as:

```
dataset/
    class_1/
        img1.png
        img2.png
    class_2/
        img1.png
        img2.png
```

Each folder name represents a class label.

---

# Common Issues & Fixes

## OpenCV Not Found

Ensure you used:

```
-DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake
```

## DLL Load Failed

Run Python from:

```
build\Release
```

OR add to PATH:

```
build\vcpkg_installed\x64-windows\bin
```

## Runtime Library Mismatch

Use:

```
-DVCPKG_TARGET_TRIPLET=x64-windows
```

Do NOT use `x64-windows-static`.

---

# Instructions

1. Build project using instructions above.
2. Train the model:
   ```python
   customdl.train(...)
   ```
3. Evaluate on hidden dataset:
   ```python
   customdl.evaluate(...)
   ```
4. Efficiency metrics are automatically printed.

---

# Reproducibility

This project:
- Uses vcpkg manifest mode
- Does NOT depend on system-specific paths
- Automatically installs dependencies
- Can be built on any Windows machine with VS Build Tools + CMake

---

# Notes

Do NOT commit:
```
build/
Release/
*.pyd
*.exe
```

Only commit:
```
framework/
bindings.cpp
train.cpp
evaluate.cpp
CMakeLists.txt
vcpkg.json
README.md
```

---

# Author

Deepak Mewada Kjala Kumari
Custom Deep Learning Framework – GNR 638 Assignment
