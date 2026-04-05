# 🎯 Real-Time Face Recognition System

A high-performance face detection and recognition system built with **C++17** and **OpenCV**.  
Achieves **~30 FPS** real-time detection with **~90% accuracy** under normal lighting conditions.

---

## ✨ Features

| Feature                  | Description                                           |
|--------------------------|-------------------------------------------------------|
| **Face Detection**       | Haar Cascade Classifier with tuned parameters         |
| **Face Recognition**     | LBPH (Local Binary Patterns Histograms) algorithm     |
| **Real-time Processing** | ~25–30 FPS on modern hardware                         |
| **Multi-face Support**   | Detects and recognizes multiple faces simultaneously  |
| **Dataset Creator**      | Interactive webcam-based dataset capture with progress bar |
| **Model Training**       | One-click LBPH model training with auto-labeling      |
| **FPS Overlay**          | Live FPS counter with color-coded performance display |
| **Snapshot Saving**      | Save detected frames and individual face crops        |
| **Interactive Controls** | Keyboard shortcuts for pause, sensitivity, snapshots  |
| **Error Handling**       | Graceful camera, file, and model error handling       |

---

## 📋 Prerequisites

### Required Software

1. **CMake** ≥ 3.15  
   Download: https://cmake.org/download/

2. **C++17 Compiler**  
   - Windows: Visual Studio 2019+ (MSVC) or MinGW-w64  
   - Linux: GCC 8+ or Clang 7+

3. **OpenCV** ≥ 4.5 with `opencv_contrib` modules  
   The `face` module (for LBPH) is part of `opencv_contrib`.

### Install OpenCV on Windows

#### Option A: Pre-built (Recommended for quick setup)

```powershell
# Using vcpkg (recommended)
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install opencv4[contrib]:x64-windows
```

#### Option B: Download from OpenCV.org

1. Go to https://opencv.org/releases/
2. Download the latest Windows release
3. Extract to `C:\opencv`
4. Add `C:\opencv\build\x64\vc16\bin` to your system PATH
5. For `opencv_contrib` (needed for LBPH):
   - Download from https://github.com/opencv/opencv_contrib
   - Rebuild OpenCV from source with `-DOPENCV_EXTRA_MODULES_PATH=<path_to_contrib>/modules`

#### Option C: Build from source (Full control)

```powershell
# Clone repositories
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

# Create build directory
cd opencv
mkdir build && cd build

# Configure with contrib modules
cmake -G "Visual Studio 17 2022" -A x64 ^
    -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ^
    -DBUILD_EXAMPLES=OFF ^
    -DBUILD_TESTS=OFF ^
    ..

# Build
cmake --build . --config Release --parallel

# Install
cmake --install . --config Release
```

### Install OpenCV on Linux

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install libopencv-dev libopencv-contrib-dev

# Or build from source for latest version
sudo apt install build-essential cmake git libgtk-3-dev
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv && mkdir build && cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ..
make -j$(nproc)
sudo make install
```

### Haar Cascade File

The system needs `haarcascade_frontalface_default.xml`. It ships with OpenCV and is auto-detected.  
If not found, download it manually:

```
https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml
```

Place it in the project root directory.

---

## 🔨 Build Instructions

### Using CMake (Cross-platform)

```powershell
# Navigate to project directory
cd "Face Recogniztion"

# Create build directory
mkdir build
cd build

# Configure (adjust OpenCV_DIR if needed)
cmake -DOpenCV_DIR="C:/opencv/build" ..

# If using vcpkg:
# cmake -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake" ..

# Build
cmake --build . --config Release

# Run
.\Release\FaceRecognitionSystem.exe
```

### Using g++ directly (Linux/MinGW)

```bash
g++ -std=c++17 -O2 \
    src/main.cpp \
    src/FaceDetector.cpp \
    src/FaceRecognizer.cpp \
    -o FaceRecognitionSystem \
    $(pkg-config --cflags --libs opencv4) \
    -lstdc++fs

./FaceRecognitionSystem
```

### Using Visual Studio directly

```powershell
# Open Developer Command Prompt for Visual Studio
cl /std:c++17 /EHsc /O2 ^
    /I "C:\opencv\build\include" ^
    src\main.cpp src\FaceDetector.cpp src\FaceRecognizer.cpp ^
    /link /LIBPATH:"C:\opencv\build\x64\vc16\lib" ^
    opencv_world4100.lib ^
    /OUT:FaceRecognitionSystem.exe
```

---

## 🚀 Usage

### Main Menu

When you run the program, you'll see this interactive menu:

```
  ==================================================
     REAL-TIME FACE RECOGNITION SYSTEM  v1.0
  ==================================================
     Built with OpenCV 4.x + LBPH
  ==================================================

  [1] Face Detection Only    (real-time webcam)
  [2] Face Recognition       (detect + identify)
  [3] Create Dataset         (capture face samples)
  [4] Train Model            (train LBPH recognizer)
  [5] Exit

  Select option [1-5]:
```

### Workflow for Face Recognition

1. **Create Dataset** (Option 3):  
   - Enter person's name → webcam captures 30 face samples  
   - Repeat for each person you want to recognize  
   - Move your head slowly during capture for varied training data

2. **Train Model** (Option 4):  
   - Trains LBPH on all captured datasets  
   - Model saved to `models/lbph_model.yml`

3. **Run Recognition** (Option 2):  
   - Real-time face detection + identification  
   - Shows person name and confidence percentage

### Keyboard Controls (During Live Feed)

| Key    | Action                                     |
|--------|--------------------------------------------|
| `q`    | Quit / Return to menu                      |
| `ESC`  | Quit / Return to menu                      |
| `s`    | Save snapshot (frame + individual faces)   |
| `p`    | Pause / Resume video                       |
| `+`    | Increase detection sensitivity (stricter)  |
| `-`    | Decrease detection sensitivity (looser)    |

---

## 📁 Project Structure

```
Face Recogniztion/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This file
├── src/
│   ├── main.cpp                # Application entry point + menu
│   ├── FaceDetector.h          # Face detection class header
│   ├── FaceDetector.cpp        # Haar Cascade detection implementation
│   ├── FaceRecognizer.h        # Face recognition class header
│   └── FaceRecognizer.cpp      # LBPH recognition implementation
├── dataset/                    # Training face images (auto-created)
│   ├── PersonA/
│   │   ├── face_0001.jpg
│   │   └── ...
│   └── PersonB/
│       └── ...
├── models/                     # Trained models (auto-created)
│   ├── lbph_model.yml
│   └── label_map.csv
├── saved_faces/                # Snapshots (auto-created)
└── build/                      # CMake build directory
```

---

## ⚙️ Performance Tuning

### Detection Parameters (`DetectionParams`)

| Parameter       | Default | Range  | Effect                              |
|-----------------|---------|--------|-------------------------------------|
| `scaleFactor`   | 1.2     | 1.05–1.4 | Lower = more accurate, slower     |
| `minNeighbors`  | 6       | 1–15   | Higher = fewer false positives      |
| `minFaceSize`   | 80×80   | 30–200 | Filters small noise detections      |

### Recognition Threshold

- Default: `80.0` (confidence below this = recognized)
- Lower threshold = stricter matching (fewer false recognitions)
- Adjust via `recognizer.setConfidenceThreshold()`

### Tips for ~90% Accuracy

1. Capture **30+ samples** per person with varied head positions
2. Ensure **good, even lighting** during capture and recognition
3. Set `minNeighbors = 5–7` for optimal false-positive filtering
4. Use `scaleFactor = 1.1` if speed isn't critical (more thorough scanning)
5. Face camera at eye-level, ~50–80cm distance

---

## 🔧 Troubleshooting

| Problem                          | Solution                                      |
|----------------------------------|-----------------------------------------------|
| "Cannot open webcam"             | Check camera connection, close other apps      |
| "Haar Cascade not found"         | Download XML file and place in project root    |
| Low FPS                          | Increase `scaleFactor`, increase `minFaceSize` |
| Too many false positives         | Increase `minNeighbors` (press `+` key)        |
| "No trained model"               | Run option 3 then 4 first                     |
| OpenCV not found by CMake        | Set `-DOpenCV_DIR=<path>` in cmake command     |
| Missing `face` module            | Install `opencv_contrib` modules               |

---

## 📊 Architecture

```
┌──────────────────────────────────────────────────────┐
│                      main.cpp                        │
│  ┌─────────┐  ┌────────────┐  ┌───────────────────┐ │
│  │  Menu   │  │  Detection │  │   Recognition      │ │
│  │  System │──│  Mode      │──│   Mode             │ │
│  └─────────┘  └─────┬──────┘  └────────┬──────────┘ │
│                     │                  │             │
│            ┌────────▼────────┐  ┌──────▼──────────┐ │
│            │  FaceDetector   │  │ FaceRecognizer   │ │
│            │  - Haar Cascade │  │ - LBPH Algorithm │ │
│            │  - Preprocess   │  │ - Dataset I/O    │ │
│            │  - Draw Boxes   │  │ - Train/Predict  │ │
│            └────────┬────────┘  └──────┬──────────┘ │
│                     │                  │             │
│            ┌────────▼──────────────────▼──────────┐ │
│            │         OpenCV Library               │ │
│            │  imgproc | objdetect | face | highgui│ │
│            └──────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

---

## 📄 License

This project is open-source. Use it for educational purposes.
