// =============================================================================
// main.cpp - Real-Time Face Recognition System
// =============================================================================
// A complete face detection and recognition system built with OpenCV.
//
// Features:
//   1. Real-time face detection using Haar Cascade Classifier
//   2. Face recognition using LBPH (Local Binary Patterns Histograms)
//   3. Dataset creation from webcam
//   4. Model training and persistence
//   5. Live FPS display
//   6. Multiple face handling
//   7. Saved face snapshots
//
// Controls (during live feed):
//   q / ESC  - Quit / Return to menu
//   s        - Save snapshot of current frame
//   p        - Pause / Resume
//   +/-      - Increase / Decrease detection sensitivity
//
// Author:  Face Recognition System
// Version: 1.0.0
// =============================================================================

#include "FaceDetector.h"
#include "FaceRecognizer.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <filesystem>
#include <sstream>

namespace fs = std::filesystem;

// =============================================================================
// GLOBAL CONFIGURATION
// =============================================================================
// Paths to Haar Cascade XML files (tried in order)
const std::vector<std::string> CASCADE_PATHS = {
    // Local copy in project directory
    "haarcascade_frontalface_default.xml",
    // OpenCV installed data directories (common locations)
    "data/haarcascade_frontalface_default.xml",
    // OpenCV samples path (auto-detected by OpenCV)
    cv::samples::findFile("haarcascades/haarcascade_frontalface_default.xml", false),
    // Common Windows OpenCV install paths
    "C:/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml",
    "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml",
};

const std::string DATASET_PATH = "dataset";
const std::string MODEL_PATH = "models/lbph_model.yml";
const std::string SAVED_FACES_PATH = "saved_faces";

// Camera settings
const int CAMERA_WIDTH = 640;
const int CAMERA_HEIGHT = 480;
const int CAMERA_ID = 0;

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

// ---- findCascadePath: Locate the Haar Cascade XML file --------------------
std::string findCascadePath() {
    for (const auto& path : CASCADE_PATHS) {
        if (!path.empty() && fs::exists(path)) {
            std::cout << "[OK] Found cascade at: " << path << std::endl;
            return path;
        }
    }

    // Last resort: ask the user
    std::cerr << "\n[ERROR] Haar Cascade file not found in any default location."
              << std::endl;
    std::cerr << "[INFO]  Expected: haarcascade_frontalface_default.xml"
              << std::endl;
    std::cerr << "[INFO]  Download from: https://github.com/opencv/opencv/tree/"
              << "master/data/haarcascades" << std::endl;
    std::cerr << "\nEnter the full path to haarcascade_frontalface_default.xml: ";

    std::string userPath;
    std::getline(std::cin, userPath);

    if (fs::exists(userPath)) {
        return userPath;
    }

    return "";
}

// ---- captureFrame: Read a frame from the camera ---------------------------
bool captureFrame(cv::VideoCapture& camera, cv::Mat& frame) {
    camera >> frame;
    return !frame.empty();
}

// ---- displayOutput: Show a frame in a named window -----------------------
void displayOutput(const std::string& windowName, const cv::Mat& frame) {
    cv::imshow(windowName, frame);
}

// ---- getTimestamp: Get current timestamp string ----------------------------
std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
    return ss.str();
}

// ---- drawHUD: Draw heads-up display overlay (FPS, mode, face count) ------
void drawHUD(cv::Mat& frame, double fps, const std::string& mode,
             int faceCount) {
    // Semi-transparent overlay bar at the top
    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, cv::Point(0, 0),
                  cv::Point(frame.cols, 50),
                  cv::Scalar(20, 20, 20), cv::FILLED);
    cv::addWeighted(overlay, 0.7, frame, 0.3, 0, frame);

    // FPS indicator with color coding
    cv::Scalar fpsColor;
    if (fps >= 25.0)      fpsColor = cv::Scalar(0, 255, 100);   // Green
    else if (fps >= 15.0) fpsColor = cv::Scalar(0, 200, 255);   // Yellow
    else                  fpsColor = cv::Scalar(0, 80, 255);    // Red

    char fpsText[32];
    snprintf(fpsText, sizeof(fpsText), "FPS: %.1f", fps);
    cv::putText(frame, fpsText, cv::Point(15, 35),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, fpsColor, 2);

    // Mode indicator
    cv::putText(frame, "Mode: " + mode,
                cv::Point(180, 35),
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(200, 200, 200), 1);

    // Face count
    std::string countText = "Faces: " + std::to_string(faceCount);
    cv::putText(frame, countText,
                cv::Point(frame.cols - 150, 35),
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(200, 200, 200), 1);

    // Bottom info bar
    cv::Mat bottomOverlay = frame.clone();
    cv::rectangle(bottomOverlay, cv::Point(0, frame.rows - 30),
                  cv::Point(frame.cols, frame.rows),
                  cv::Scalar(20, 20, 20), cv::FILLED);
    cv::addWeighted(bottomOverlay, 0.7, frame, 0.3, 0, frame);

    cv::putText(frame, "Q:Quit  S:Snapshot  P:Pause  +/-:Sensitivity",
                cv::Point(10, frame.rows - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.4,
                cv::Scalar(150, 150, 150), 1);
}

// ---- openCamera: Initialize webcam with error handling --------------------
bool openCamera(cv::VideoCapture& camera) {
    std::cout << "[INFO] Opening camera " << CAMERA_ID << "..." << std::endl;

    camera.open(CAMERA_ID);

    if (!camera.isOpened()) {
        std::cerr << "[ERROR] Cannot open webcam (ID=" << CAMERA_ID << ")."
                  << std::endl;
        std::cerr << "[INFO]  Possible causes:" << std::endl;
        std::cerr << "         - No camera connected" << std::endl;
        std::cerr << "         - Camera in use by another application" << std::endl;
        std::cerr << "         - Incorrect camera ID" << std::endl;
        return false;
    }

    // Set resolution for performance
    camera.set(cv::CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT);

    // Try to set higher FPS
    camera.set(cv::CAP_PROP_FPS, 30);

    std::cout << "[OK] Camera opened: "
              << static_cast<int>(camera.get(cv::CAP_PROP_FRAME_WIDTH)) << "x"
              << static_cast<int>(camera.get(cv::CAP_PROP_FRAME_HEIGHT))
              << " @ " << camera.get(cv::CAP_PROP_FPS) << " FPS"
              << std::endl;

    return true;
}

// =============================================================================
// MODE 1: DETECTION ONLY
// =============================================================================
// Runs real-time face detection without recognition.
// Draws bounding boxes on all detected faces.
// =============================================================================
void runDetectionMode(FaceDetector& detector) {
    cv::VideoCapture camera;
    if (!openCamera(camera)) return;

    std::cout << "\n=== DETECTION MODE ===" << std::endl;
    std::cout << "Press 'q' or ESC to return to menu.\n" << std::endl;

    // FPS calculation variables
    auto prevTime = std::chrono::high_resolution_clock::now();
    double fps = 0.0;
    int frameCount = 0;
    bool paused = false;
    cv::Mat frame, displayFrame;

    while (true) {
        if (!paused) {
            if (!captureFrame(camera, frame)) {
                std::cerr << "[ERROR] Failed to capture frame." << std::endl;
                break;
            }
        }

        // Make a copy for display
        displayFrame = frame.clone();

        // Detect faces
        auto faces = detector.detectFaces(displayFrame);

        // Draw bounding boxes
        detector.drawDetections(displayFrame, faces);

        // Calculate FPS (rolling average)
        frameCount++;
        auto currentTime = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(
            currentTime - prevTime).count();

        if (elapsed >= 0.5) {  // Update FPS every 0.5 seconds
            fps = frameCount / elapsed;
            frameCount = 0;
            prevTime = currentTime;
        }

        // Draw HUD
        drawHUD(displayFrame, fps, "DETECTION",
                static_cast<int>(faces.size()));

        if (paused) {
            cv::putText(displayFrame, "PAUSED",
                        cv::Point(displayFrame.cols / 2 - 80,
                                  displayFrame.rows / 2),
                        cv::FONT_HERSHEY_SIMPLEX, 1.5,
                        cv::Scalar(0, 0, 255), 3);
        }

        // Show the frame
        displayOutput("Face Detection", displayFrame);

        // Handle keyboard input
        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) break;

        if (key == 'p' || key == 'P') {
            paused = !paused;
        }

        if (key == 's' || key == 'S') {
            // Save snapshot
            std::string filename = SAVED_FACES_PATH + "/snapshot_" +
                                   getTimestamp() + ".jpg";
            cv::imwrite(filename, frame);
            std::cout << "[OK] Snapshot saved: " << filename << std::endl;
        }

        // Adjust sensitivity
        if (key == '+' || key == '=') {
            detector.getParams().minNeighbors =
                std::min(15, detector.getParams().minNeighbors + 1);
            std::cout << "[INFO] minNeighbors = "
                      << detector.getParams().minNeighbors << std::endl;
        }
        if (key == '-' || key == '_') {
            detector.getParams().minNeighbors =
                std::max(1, detector.getParams().minNeighbors - 1);
            std::cout << "[INFO] minNeighbors = "
                      << detector.getParams().minNeighbors << std::endl;
        }
    }

    camera.release();
    cv::destroyAllWindows();
}

// =============================================================================
// MODE 2: RECOGNITION MODE
// =============================================================================
// Runs face detection + LBPH recognition in real-time.
// Shows detected person names with confidence scores.
// =============================================================================
void runRecognitionMode(FaceDetector& detector, FaceRecognizer& recognizer) {
    // Ensure model is trained
    if (!recognizer.isModelTrained()) {
        std::cout << "[INFO] No trained model found. Attempting to load..."
                  << std::endl;
        if (!recognizer.loadModel()) {
            std::cerr << "[ERROR] No trained model available." << std::endl;
            std::cerr << "[INFO]  Run option 3 (Create Dataset) first, "
                      << "then option 4 (Train Model)." << std::endl;
            return;
        }
    }

    cv::VideoCapture camera;
    if (!openCamera(camera)) return;

    std::cout << "\n=== RECOGNITION MODE ===" << std::endl;
    std::cout << "Press 'q' or ESC to return to menu.\n" << std::endl;

    auto prevTime = std::chrono::high_resolution_clock::now();
    double fps = 0.0;
    int frameCount = 0;
    bool paused = false;
    cv::Mat frame, displayFrame;

    while (true) {
        if (!paused) {
            if (!captureFrame(camera, frame)) {
                std::cerr << "[ERROR] Failed to capture frame." << std::endl;
                break;
            }
        }

        displayFrame = frame.clone();

        // Detect faces
        auto faces = detector.detectFaces(displayFrame);

        // Recognize each face
        std::vector<std::string> labels;
        cv::Mat gray = detector.preprocessFrame(frame);

        for (const auto& faceRect : faces) {
            cv::Mat faceROI = detector.extractFaceROI(gray, faceRect, 200);

            if (!faceROI.empty()) {
                RecognitionResult result = recognizer.recognize(faceROI);

                // Format label with name and confidence
                std::string label;
                if (result.recognized) {
                    char conf[16];
                    snprintf(conf, sizeof(conf), "%.0f%%",
                             100.0 - result.confidence);
                    label = result.name + " (" + std::string(conf) + ")";
                } else {
                    label = "Unknown";
                }
                labels.push_back(label);
            } else {
                labels.push_back("Unknown");
            }
        }

        // Draw detections with recognition labels
        detector.drawDetections(displayFrame, faces, labels);

        // Calculate FPS
        frameCount++;
        auto currentTime = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(
            currentTime - prevTime).count();

        if (elapsed >= 0.5) {
            fps = frameCount / elapsed;
            frameCount = 0;
            prevTime = currentTime;
        }

        // Draw HUD
        drawHUD(displayFrame, fps, "RECOGNITION",
                static_cast<int>(faces.size()));

        if (paused) {
            cv::putText(displayFrame, "PAUSED",
                        cv::Point(displayFrame.cols / 2 - 80,
                                  displayFrame.rows / 2),
                        cv::FONT_HERSHEY_SIMPLEX, 1.5,
                        cv::Scalar(0, 0, 255), 3);
        }

        displayOutput("Face Recognition", displayFrame);

        // Handle keyboard input
        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) break;

        if (key == 'p' || key == 'P') paused = !paused;

        if (key == 's' || key == 'S') {
            std::string filename = SAVED_FACES_PATH + "/recognized_" +
                                   getTimestamp() + ".jpg";
            cv::imwrite(filename, frame);

            // Also save individual faces
            for (size_t i = 0; i < faces.size(); i++) {
                cv::Mat faceROI = detector.extractFaceROI(gray, faces[i], 200);
                if (!faceROI.empty()) {
                    std::string faceFile = SAVED_FACES_PATH + "/face_" +
                                           std::to_string(i) + "_" +
                                           getTimestamp() + ".jpg";
                    cv::imwrite(faceFile, faceROI);
                }
            }
            std::cout << "[OK] Snapshot and faces saved." << std::endl;
        }
    }

    camera.release();
    cv::destroyAllWindows();
}

// =============================================================================
// MODE 3: DATASET CREATION
// =============================================================================
// Interactive menu to create training datasets for new people.
// =============================================================================
void runDatasetCreation(FaceRecognizer& recognizer) {
    std::cout << "\n=== DATASET CREATION ===" << std::endl;

    // Show existing datasets
    if (fs::exists(DATASET_PATH)) {
        std::cout << "\nExisting datasets:" << std::endl;
        int count = 0;
        for (const auto& entry : fs::directory_iterator(DATASET_PATH)) {
            if (entry.is_directory()) {
                int numImages = 0;
                for (const auto& img : fs::directory_iterator(entry)) {
                    if (img.is_regular_file()) numImages++;
                }
                std::cout << "  " << entry.path().filename().string()
                          << " (" << numImages << " images)" << std::endl;
                count++;
            }
        }
        if (count == 0) {
            std::cout << "  (none)" << std::endl;
        }
    }

    // Get person's name
    std::cout << "\nEnter the person's name (or 'cancel' to go back): ";
    std::string name;
    std::getline(std::cin, name);

    if (name.empty() || name == "cancel") {
        std::cout << "[INFO] Dataset creation cancelled." << std::endl;
        return;
    }

    // Get label ID
    int label = 0;
    auto labelMap = recognizer.getLabelMap();
    if (!labelMap.empty()) {
        label = labelMap.rbegin()->first + 1;
    }

    // Check if person already exists
    for (const auto& [l, n] : labelMap) {
        if (n == name) {
            label = l;
            std::cout << "[INFO] Person '" << name
                      << "' already exists with label " << label
                      << ". Adding more samples." << std::endl;
            break;
        }
    }

    // Get number of samples
    std::cout << "Number of samples to capture (default 30): ";
    std::string numStr;
    std::getline(std::cin, numStr);
    int numSamples = numStr.empty() ? 30 : std::stoi(numStr);
    numSamples = std::clamp(numSamples, 5, 200);

    std::cout << "\n[INFO] Will capture " << numSamples
              << " samples for '" << name << "' (label=" << label << ")"
              << std::endl;
    std::cout << "[INFO] Press ENTER to start, or 'c' to cancel: ";

    std::string confirm;
    std::getline(std::cin, confirm);
    if (confirm == "c" || confirm == "C") {
        std::cout << "[INFO] Cancelled." << std::endl;
        return;
    }

    // Capture!
    recognizer.captureDataset(name, label, numSamples);
}

// =============================================================================
// MODE 4: TRAIN MODEL
// =============================================================================
void runTraining(FaceRecognizer& recognizer) {
    std::cout << "\n=== MODEL TRAINING ===" << std::endl;

    if (!fs::exists(DATASET_PATH)) {
        std::cerr << "[ERROR] No dataset directory found." << std::endl;
        std::cerr << "[INFO]  Create a dataset first (option 3)." << std::endl;
        return;
    }

    // Check for images
    int totalImages = 0;
    int totalPeople = 0;
    for (const auto& entry : fs::directory_iterator(DATASET_PATH)) {
        if (entry.is_directory()) {
            totalPeople++;
            for (const auto& img : fs::directory_iterator(entry)) {
                if (img.is_regular_file()) totalImages++;
            }
        }
    }

    if (totalImages == 0) {
        std::cerr << "[ERROR] No training images found." << std::endl;
        return;
    }

    std::cout << "[INFO] Found " << totalImages << " images for "
              << totalPeople << " people." << std::endl;
    std::cout << "[INFO] Training LBPH model..." << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();

    bool success = recognizer.trainModel();

    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(endTime - startTime).count();

    if (success) {
        std::cout << "[OK] Training completed in " << std::fixed
                  << std::setprecision(2) << elapsed << " seconds."
                  << std::endl;
    } else {
        std::cerr << "[ERROR] Training failed." << std::endl;
    }
}

// =============================================================================
// MAIN MENU
// =============================================================================
void printMenu() {
    std::cout << "\n";
    std::cout << "  =================================================="
              << std::endl;
    std::cout << "     REAL-TIME FACE RECOGNITION SYSTEM  v1.0"
              << std::endl;
    std::cout << "  =================================================="
              << std::endl;
    std::cout << "     Built with OpenCV " << CV_VERSION
              << " + LBPH" << std::endl;
    std::cout << "  =================================================="
              << std::endl;
    std::cout << std::endl;
    std::cout << "  [1] Face Detection Only    (real-time webcam)"
              << std::endl;
    std::cout << "  [2] Face Recognition       (detect + identify)"
              << std::endl;
    std::cout << "  [3] Create Dataset         (capture face samples)"
              << std::endl;
    std::cout << "  [4] Train Model            (train LBPH recognizer)"
              << std::endl;
    std::cout << "  [5] Exit" << std::endl;
    std::cout << std::endl;
    std::cout << "  Select option [1-5]: ";
}

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================
int main() {
    std::cout << "\n[INFO] Initializing Face Recognition System...\n"
              << std::endl;

    // ---- Step 1: Find the Haar Cascade file ----
    std::string cascadePath = findCascadePath();
    if (cascadePath.empty()) {
        std::cerr << "[FATAL] Cannot find Haar Cascade file. Exiting."
                  << std::endl;
        return -1;
    }

    // ---- Step 2: Initialize the Face Detector ----
    FaceDetector detector(cascadePath);
    if (!detector.isLoaded()) {
        std::cerr << "[FATAL] Failed to initialize face detector. Exiting."
                  << std::endl;
        return -1;
    }

    // ---- Step 3: Initialize the Face Recognizer ----
    FaceRecognizer recognizer(DATASET_PATH, MODEL_PATH);

    // Try to load existing model
    if (fs::exists(MODEL_PATH)) {
        recognizer.loadModel();
    }

    // ---- Step 4: Create required directories ----
    if (!fs::exists(DATASET_PATH))    fs::create_directories(DATASET_PATH);
    if (!fs::exists(SAVED_FACES_PATH)) fs::create_directories(SAVED_FACES_PATH);
    if (!fs::exists("models"))        fs::create_directories("models");

    std::cout << "[OK] System initialized successfully." << std::endl;

    // ---- Step 5: Main menu loop ----
    bool running = true;
    while (running) {
        printMenu();

        std::string choice;
        std::getline(std::cin, choice);

        if (choice == "1") {
            runDetectionMode(detector);
        }
        else if (choice == "2") {
            runRecognitionMode(detector, recognizer);
        }
        else if (choice == "3") {
            runDatasetCreation(recognizer);
        }
        else if (choice == "4") {
            runTraining(recognizer);
        }
        else if (choice == "5") {
            running = false;
            std::cout << "\n[INFO] Shutting down. Goodbye!\n" << std::endl;
        }
        else {
            std::cout << "[WARNING] Invalid option. Please enter 1-5."
                      << std::endl;
        }
    }

    return 0;
}
