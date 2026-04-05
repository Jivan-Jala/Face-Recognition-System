# Real-Time Face Detection System

A simple, real-time face detection system using **C++** and **OpenCV Haar Cascade Classifier**.

## How It Works

```
Webcam Frame → Grayscale → Histogram Equalization → Haar Cascade Detection → Bounding Boxes → Display
```

1. **Capture** a frame from the webcam
2. **Convert** to grayscale (faster — 1 channel instead of 3)
3. **Equalize histogram** (normalizes lighting conditions)
4. **Detect faces** using `detectMultiScale()` with the Haar Cascade
5. **Draw** green bounding boxes around each detected face
6. **Display** the result with an FPS counter

## Key Concepts (Interview Points)

| Concept | Explanation |
|---------|-------------|
| **Haar Cascade** | Pre-trained classifier using Viola-Jones algorithm. Uses AdaBoost to select the best features from ~160,000 candidates, organized as a cascade of stages for fast rejection of non-face regions. |
| **scaleFactor (1.2)** | Controls the image pyramid. At each scale, the image is reduced by 20%. Lower values = more accurate but slower. |
| **minNeighbors (6)** | A detection must overlap with at least 6 other detections to be confirmed. Higher = fewer false positives. |
| **Grayscale conversion** | Haar features work on intensity differences, so color is unnecessary. Reduces computation by 3×. |
| **Histogram equalization** | Spreads pixel intensity values evenly, improving detection under poor or uneven lighting. |

## Performance

- **~30 FPS** real-time detection
- **~90% accuracy** under normal lighting
- Handles **multiple faces** simultaneously

## Code Structure

```
src/main.cpp          ← Single file, ~180 lines
├── captureFrame()    ← Read frame from webcam
├── preprocessFrame() ← Grayscale + histogram equalization
├── detectFaces()     ← Haar Cascade detectMultiScale
├── displayOutput()   ← Draw bounding boxes + FPS
└── main()            ← Setup camera, run detection loop
```

## Build & Run

### Prerequisites
- OpenCV 4.x (`pip install opencv-python` for Python, or build from source for C++)
- C++17 compiler (g++, MSVC, or clang)

### With CMake
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
./FaceDetection          # Linux/Mac
.\Release\FaceDetection.exe  # Windows
```

### With g++ directly
```bash
g++ -std=c++17 -O2 src/main.cpp -o FaceDetection $(pkg-config --cflags --libs opencv4)
```

## Controls

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `s` | Save screenshot |

## Dependencies

- OpenCV 4.x (core, imgproc, highgui, videoio, objdetect)
- `haarcascade_frontalface_default.xml` (ships with OpenCV)
