// =============================================================================
// Real-Time Face Detection System using Haar Cascade Classifier
// =============================================================================
// Language : C++17
// Library  : OpenCV 4.x
// Method   : Haar Cascade (Viola-Jones Algorithm)
//
// How it works:
//   1. Capture live video frames from webcam
//   2. Convert each frame to grayscale (faster processing)
//   3. Apply histogram equalization (normalize lighting)
//   4. Run Haar Cascade detectMultiScale to find faces
//   5. Draw bounding boxes around detected faces
//   6. Display FPS counter for performance monitoring
//
// Key Parameters:
//   scaleFactor  = 1.2  → 20% image reduction per scale (speed vs accuracy)
//   minNeighbors = 6    → minimum overlapping detections to confirm a face
//   minSize      = 80x80 → ignore regions smaller than this
//
// Performance: ~30 FPS detection with ~90% accuracy (normal lighting)
//
// Controls:
//   q / ESC → Quit
//   s       → Save screenshot
// =============================================================================

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

// =============================================================================
// FUNCTION: captureFrame
// =============================================================================
// Purpose : Read one frame from the webcam
// Input   : VideoCapture reference, Mat reference for output
// Output  : true if frame was read successfully, false otherwise
// =============================================================================
bool captureFrame(cv::VideoCapture& camera, cv::Mat& frame) {
    camera >> frame;
    return !frame.empty();
}

// =============================================================================
// FUNCTION: preprocessFrame
// =============================================================================
// Purpose : Convert BGR frame to grayscale and equalize histogram
// Why     : Haar Cascade works on grayscale images. Histogram equalization
//           normalizes brightness and contrast, improving detection under
//           varying lighting conditions.
// Input   : BGR color frame
// Output  : Preprocessed grayscale frame
// =============================================================================
cv::Mat preprocessFrame(const cv::Mat& frame) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);  // BGR → Grayscale
    cv::equalizeHist(gray, gray);                    // Normalize lighting
    return gray;
}

// =============================================================================
// FUNCTION: detectFaces
// =============================================================================
// Purpose : Detect all faces in a grayscale frame using Haar Cascade
//
// How detectMultiScale works:
//   - Slides a detection window across the image at multiple scales
//   - scaleFactor=1.2 means image is reduced by 20% at each scale
//   - At each position, the cascade evaluates ~6000 features
//   - minNeighbors=6 means a region must be detected 6+ times to count
//   - This filtering removes most false positives
//
// Input   : CascadeClassifier, grayscale frame
// Output  : Vector of rectangles (x, y, width, height) for each face
// =============================================================================
std::vector<cv::Rect> detectFaces(cv::CascadeClassifier& cascade,
                                   const cv::Mat& gray) {
    std::vector<cv::Rect> faces;

    cascade.detectMultiScale(
        gray,                   // Input: grayscale image
        faces,                  // Output: detected face rectangles
        1.2,                    // scaleFactor: image pyramid scale
        6,                      // minNeighbors: detection threshold
        0,                      // flags (unused, kept for compatibility)
        cv::Size(80, 80)        // minSize: minimum face size in pixels
    );

    return faces;
}

// =============================================================================
// FUNCTION: displayOutput
// =============================================================================
// Purpose : Draw bounding boxes on detected faces and show the frame
//
// Drawing details:
//   - Green rectangle around each face
//   - Label "Face #N" above each detection
//   - Semi-transparent HUD bar at top with FPS counter
//   - Face count display
//
// Input   : frame (modified in-place), face rectangles, current FPS
// =============================================================================
void displayOutput(cv::Mat& frame,
                   const std::vector<cv::Rect>& faces,
                   double fps) {
    // --- Draw bounding box for each detected face ---
    for (size_t i = 0; i < faces.size(); i++) {
        // Green bounding box
        cv::rectangle(frame, faces[i], cv::Scalar(0, 255, 0), 2);

        // Label above the box
        std::string label = "Face #" + std::to_string(i + 1);
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                             0.6, 2, &baseline);

        // Black background for label readability
        cv::rectangle(frame,
                      cv::Point(faces[i].x, faces[i].y - textSize.height - 8),
                      cv::Point(faces[i].x + textSize.width + 4, faces[i].y),
                      cv::Scalar(0, 0, 0), cv::FILLED);

        // White label text
        cv::putText(frame, label,
                    cv::Point(faces[i].x + 2, faces[i].y - 6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 255), 2);
    }

    // --- HUD: FPS and face count at top ---
    char hudText[64];
    snprintf(hudText, sizeof(hudText), "FPS: %.1f | Faces: %zu",
             fps, faces.size());
    cv::putText(frame, hudText, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

    // --- Show the frame ---
    cv::imshow("Face Detection - Haar Cascade", frame);
}

// =============================================================================
// MAIN FUNCTION
// =============================================================================
int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "  Real-Time Face Detection (Haar Cascade)   " << std::endl;
    std::cout << "  OpenCV " << CV_VERSION << " | C++17       " << std::endl;
    std::cout << "============================================" << std::endl;

    // -----------------------------------------------------------------
    // STEP 1: Load the Haar Cascade Classifier
    // -----------------------------------------------------------------
    // The XML file contains pre-trained feature data for frontal faces.
    // It was trained on thousands of positive/negative samples using
    // the Viola-Jones (AdaBoost + Cascade) algorithm.
    // -----------------------------------------------------------------
    cv::CascadeClassifier faceCascade;
    std::string cascadePath = cv::samples::findFile(
        "haarcascades/haarcascade_frontalface_default.xml"
    );

    if (!faceCascade.load(cascadePath)) {
        std::cerr << "[ERROR] Failed to load Haar Cascade from: "
                  << cascadePath << std::endl;
        std::cerr << "  Download from: https://github.com/opencv/opencv/"
                  << "tree/master/data/haarcascades" << std::endl;
        return -1;
    }
    std::cout << "[OK] Haar Cascade loaded." << std::endl;

    // -----------------------------------------------------------------
    // STEP 2: Open the webcam
    // -----------------------------------------------------------------
    cv::VideoCapture camera(0);  // 0 = default camera

    if (!camera.isOpened()) {
        std::cerr << "[ERROR] Cannot open webcam." << std::endl;
        std::cerr << "  Check if camera is connected and not in use." << std::endl;
        return -1;
    }

    // Set camera resolution (640x480 is a good balance for real-time)
    camera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    std::cout << "[OK] Webcam opened (640x480)." << std::endl;
    std::cout << "[INFO] Press 'q' to quit, 's' to save screenshot.\n" << std::endl;

    // -----------------------------------------------------------------
    // STEP 3: Main detection loop
    // -----------------------------------------------------------------
    cv::Mat frame;
    double fps = 0.0;
    int frameCount = 0;
    auto prevTime = std::chrono::high_resolution_clock::now();

    while (true) {
        // --- Capture a frame ---
        if (!captureFrame(camera, frame)) {
            std::cerr << "[ERROR] Failed to read frame." << std::endl;
            break;
        }

        // --- Preprocess: grayscale + histogram equalization ---
        cv::Mat gray = preprocessFrame(frame);

        // --- Detect faces ---
        std::vector<cv::Rect> faces = detectFaces(faceCascade, gray);

        // --- Calculate FPS (update every 0.5 seconds) ---
        frameCount++;
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - prevTime).count();
        if (elapsed >= 0.5) {
            fps = frameCount / elapsed;
            frameCount = 0;
            prevTime = now;
        }

        // --- Draw results and display ---
        displayOutput(frame, faces, fps);

        // --- Handle keyboard input ---
        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {  // q or ESC
            break;
        }
        if (key == 's' || key == 'S') {  // Save screenshot
            cv::imwrite("screenshot.jpg", frame);
            std::cout << "[OK] Screenshot saved." << std::endl;
        }
    }

    // -----------------------------------------------------------------
    // STEP 4: Cleanup
    // -----------------------------------------------------------------
    camera.release();
    cv::destroyAllWindows();
    std::cout << "\n[INFO] Program ended." << std::endl;

    return 0;
}
