// =============================================================================
// FaceRecognizer.cpp - LBPH Face Recognition Implementation
// =============================================================================

#include "FaceRecognizer.h"
#include "FaceDetector.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

// =============================================================================
// Constructor: Initialize LBPH recognizer with default parameters
// =============================================================================
// LBPH Parameters:
//   radius=1      : Radius of circular LBP pattern
//   neighbors=8   : Number of sample points in LBP
//   grid_x=8      : Horizontal grid cell count
//   grid_y=8      : Vertical grid cell count
//   threshold=DBL_MAX : Recognition confidence threshold
// =============================================================================
FaceRecognizer::FaceRecognizer(const std::string& datasetPath,
                               const std::string& modelPath)
    : m_datasetPath(datasetPath)
    , m_modelPath(modelPath)
    , m_confidenceThreshold(80.0)   // Default: below 80 = recognized
    , m_trained(false)
{
    // Create LBPH face recognizer
    m_recognizer = cv::face::LBPHFaceRecognizer::create(
        1,    // radius
        8,    // neighbors
        8,    // grid_x
        8     // grid_y
    );

    // Ensure required directories exist
    ensureDirectory(m_datasetPath);
    ensureDirectory("models");
    ensureDirectory("saved_faces");

    // Load existing label map if available
    loadLabelMap();

    std::cout << "[OK] Face Recognizer initialized." << std::endl;
    std::cout << "     Dataset path: " << m_datasetPath << std::endl;
    std::cout << "     Model path:   " << m_modelPath << std::endl;
}

// =============================================================================
// captureDataset: Capture face samples from webcam for training
// =============================================================================
// Opens the webcam, detects faces, and saves grayscale face ROIs to:
//   dataset/<personName>/face_0001.jpg, face_0002.jpg, ...
// The user should move their head slightly for varied training data.
// =============================================================================
bool FaceRecognizer::captureDataset(const std::string& personName,
                                     int personLabel,
                                     int numSamples) {
    // Create directory for this person
    std::string personDir = m_datasetPath + "/" + personName;
    ensureDirectory(personDir);

    // Add to label map
    m_labelMap[personLabel] = personName;
    saveLabelMap();

    // Open webcam
    cv::VideoCapture camera(0);
    if (!camera.isOpened()) {
        std::cerr << "[ERROR] Cannot open webcam for dataset capture."
                  << std::endl;
        return false;
    }

    // Set camera properties for speed
    camera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // Create a face detector for extraction
    // Try multiple cascade paths
    std::vector<std::string> cascadePaths = {
        "haarcascade_frontalface_default.xml",
        cv::samples::findFile("haarcascades/haarcascade_frontalface_default.xml", false),
    };

    FaceDetector detector(cascadePaths[0]);
    if (!detector.isLoaded()) {
        for (size_t i = 1; i < cascadePaths.size(); i++) {
            detector = FaceDetector(cascadePaths[i]);
            if (detector.isLoaded()) break;
        }
    }

    if (!detector.isLoaded()) {
        std::cerr << "[ERROR] Cannot load face detector for capture."
                  << std::endl;
        camera.release();
        return false;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "  DATASET CAPTURE: " << personName << std::endl;
    std::cout << "  Label ID:        " << personLabel << std::endl;
    std::cout << "  Samples needed:  " << numSamples << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Look at the camera. Move your head slowly." << std::endl;
    std::cout << "Press 'q' to cancel.\n" << std::endl;

    int captured = 0;
    int frameCount = 0;
    cv::Mat frame;

    while (captured < numSamples) {
        camera >> frame;
        if (frame.empty()) break;

        frameCount++;

        // Only capture every 5th frame for variety
        bool shouldCapture = (frameCount % 5 == 0);

        // Detect faces
        auto faces = detector.detectFaces(frame);

        if (!faces.empty() && shouldCapture) {
            // Use the largest face
            auto largest = std::max_element(faces.begin(), faces.end(),
                [](const cv::Rect& a, const cv::Rect& b) {
                    return a.area() < b.area();
                });

            // Extract grayscale face ROI
            cv::Mat gray = detector.preprocessFrame(frame);
            cv::Mat faceROI = detector.extractFaceROI(gray, *largest, 200);

            if (!faceROI.empty()) {
                // Save the face image
                char filename[256];
                snprintf(filename, sizeof(filename),
                         "%s/face_%04d.jpg", personDir.c_str(), captured + 1);
                cv::imwrite(filename, faceROI);
                captured++;

                // Flash green border to indicate capture
                cv::rectangle(frame, *largest,
                              cv::Scalar(0, 255, 0), 4);
            }
        }

        // Draw all detected faces
        detector.drawDetections(frame, faces);

        // Draw progress bar
        float progress = static_cast<float>(captured) / numSamples;
        int barWidth = 300;
        int barHeight = 25;
        int barX = (frame.cols - barWidth) / 2;
        int barY = frame.rows - 50;

        // Background
        cv::rectangle(frame,
                      cv::Point(barX, barY),
                      cv::Point(barX + barWidth, barY + barHeight),
                      cv::Scalar(50, 50, 50), cv::FILLED);
        // Fill
        cv::rectangle(frame,
                      cv::Point(barX, barY),
                      cv::Point(barX + static_cast<int>(barWidth * progress),
                                barY + barHeight),
                      cv::Scalar(0, 200, 100), cv::FILLED);
        // Border
        cv::rectangle(frame,
                      cv::Point(barX, barY),
                      cv::Point(barX + barWidth, barY + barHeight),
                      cv::Scalar(200, 200, 200), 1);

        // Progress text
        std::string progressText = "Captured: " + std::to_string(captured) +
                                   " / " + std::to_string(numSamples);
        cv::putText(frame, progressText,
                    cv::Point(barX + 10, barY + 18),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1);

        // Display
        cv::imshow("Dataset Capture - " + personName, frame);

        // Check for quit key
        int key = cv::waitKey(30);
        if (key == 'q' || key == 'Q' || key == 27) {
            std::cout << "[INFO] Capture cancelled by user." << std::endl;
            camera.release();
            cv::destroyWindow("Dataset Capture - " + personName);
            return false;
        }
    }

    camera.release();
    cv::destroyWindow("Dataset Capture - " + personName);

    std::cout << "\n[OK] Captured " << captured << " samples for '"
              << personName << "'." << std::endl;
    return true;
}

// =============================================================================
// trainModel: Load all dataset images and train the LBPH model
// =============================================================================
bool FaceRecognizer::trainModel() {
    std::vector<cv::Mat> images;
    std::vector<int> labels;

    std::cout << "\n[INFO] Loading dataset images..." << std::endl;

    if (!loadDatasetImages(images, labels)) {
        std::cerr << "[ERROR] Failed to load dataset images." << std::endl;
        return false;
    }

    if (images.empty()) {
        std::cerr << "[ERROR] No training images found in: "
                  << m_datasetPath << std::endl;
        return false;
    }

    std::cout << "[INFO] Training LBPH model with " << images.size()
              << " images..." << std::endl;

    // Train the recognizer
    m_recognizer->train(images, labels);

    // Save the trained model
    ensureDirectory("models");
    m_recognizer->save(m_modelPath);
    saveLabelMap();

    m_trained = true;

    std::cout << "[OK] Model trained and saved to: " << m_modelPath
              << std::endl;
    std::cout << "[OK] Labels: " << m_labelMap.size() << " people registered."
              << std::endl;

    for (const auto& [label, name] : m_labelMap) {
        std::cout << "     Label " << label << " -> " << name << std::endl;
    }

    return true;
}

// =============================================================================
// loadModel: Load a previously saved LBPH model
// =============================================================================
bool FaceRecognizer::loadModel() {
    if (!fs::exists(m_modelPath)) {
        std::cerr << "[WARNING] No trained model found at: " << m_modelPath
                  << std::endl;
        return false;
    }

    try {
        m_recognizer->read(m_modelPath);
        m_trained = true;
        loadLabelMap();

        std::cout << "[OK] Model loaded from: " << m_modelPath << std::endl;
        std::cout << "[OK] " << m_labelMap.size() << " people in model."
                  << std::endl;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "[ERROR] Failed to load model: " << e.what() << std::endl;
        return false;
    }
}

// =============================================================================
// recognize: Predict a face's identity
// =============================================================================
// Returns a RecognitionResult with:
//   - label:      numeric ID
//   - name:       person's name (or "Unknown")
//   - confidence: distance score (lower = better)
//   - recognized: true if confidence < threshold
// =============================================================================
RecognitionResult FaceRecognizer::recognize(const cv::Mat& faceROI) {
    RecognitionResult result;
    result.label = -1;
    result.name = "Unknown";
    result.confidence = 999.0;
    result.recognized = false;

    if (!m_trained || faceROI.empty()) {
        return result;
    }

    // Ensure the image is grayscale and correct size
    cv::Mat processed;
    if (faceROI.channels() > 1) {
        cv::cvtColor(faceROI, processed, cv::COLOR_BGR2GRAY);
    } else {
        processed = faceROI;
    }
    cv::resize(processed, processed, cv::Size(200, 200));

    // Predict
    int predictedLabel = -1;
    double confidence = 0.0;
    m_recognizer->predict(processed, predictedLabel, confidence);

    result.label = predictedLabel;
    result.confidence = confidence;

    // Check if confidence is below threshold (recognized)
    if (confidence < m_confidenceThreshold) {
        result.recognized = true;
        auto it = m_labelMap.find(predictedLabel);
        if (it != m_labelMap.end()) {
            result.name = it->second;
        } else {
            result.name = "Person_" + std::to_string(predictedLabel);
        }
    }

    return result;
}

// =============================================================================
// isModelTrained: Check model readiness
// =============================================================================
bool FaceRecognizer::isModelTrained() const {
    return m_trained;
}

// =============================================================================
// Label map accessors
// =============================================================================
const std::map<int, std::string>& FaceRecognizer::getLabelMap() const {
    return m_labelMap;
}

void FaceRecognizer::addLabelMapping(int label, const std::string& name) {
    m_labelMap[label] = name;
}

void FaceRecognizer::setConfidenceThreshold(double threshold) {
    m_confidenceThreshold = threshold;
}

std::string FaceRecognizer::getDatasetPath() const {
    return m_datasetPath;
}

std::string FaceRecognizer::getModelPath() const {
    return m_modelPath;
}

// =============================================================================
// loadDatasetImages: Scan dataset/ directory for face images
// =============================================================================
// Expected structure:
//   dataset/
//     PersonA/       <- label read from label_map.csv
//       face_0001.jpg
//       face_0002.jpg
//     PersonB/
//       face_0001.jpg
// =============================================================================
bool FaceRecognizer::loadDatasetImages(std::vector<cv::Mat>& images,
                                        std::vector<int>& labels) {
    if (!fs::exists(m_datasetPath)) {
        std::cerr << "[ERROR] Dataset directory not found: "
                  << m_datasetPath << std::endl;
        return false;
    }

    // Iterate through person directories
    for (const auto& personEntry : fs::directory_iterator(m_datasetPath)) {
        if (!personEntry.is_directory()) continue;

        std::string personName = personEntry.path().filename().string();

        // Find the label for this person
        int label = -1;
        for (const auto& [l, n] : m_labelMap) {
            if (n == personName) {
                label = l;
                break;
            }
        }

        // If not in map, assign a new label
        if (label == -1) {
            label = m_labelMap.empty() ? 0 :
                    m_labelMap.rbegin()->first + 1;
            m_labelMap[label] = personName;
        }

        int count = 0;
        // Load all images in this person's directory
        for (const auto& imgEntry : fs::directory_iterator(personEntry)) {
            if (!imgEntry.is_regular_file()) continue;

            std::string ext = imgEntry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" &&
                ext != ".bmp") {
                continue;
            }

            cv::Mat img = cv::imread(imgEntry.path().string(),
                                     cv::IMREAD_GRAYSCALE);
            if (img.empty()) continue;

            // Resize to standard size
            cv::resize(img, img, cv::Size(200, 200));

            images.push_back(img);
            labels.push_back(label);
            count++;
        }

        std::cout << "  Loaded " << count << " images for '"
                  << personName << "' (label=" << label << ")" << std::endl;
    }

    return !images.empty();
}

// =============================================================================
// saveLabelMap: Save label-to-name mapping as CSV
// =============================================================================
void FaceRecognizer::saveLabelMap() {
    std::string mapPath = "models/label_map.csv";
    ensureDirectory("models");

    std::ofstream file(mapPath);
    if (!file.is_open()) {
        std::cerr << "[WARNING] Cannot save label map to: " << mapPath
                  << std::endl;
        return;
    }

    for (const auto& [label, name] : m_labelMap) {
        file << label << "," << name << "\n";
    }

    file.close();
}

// =============================================================================
// loadLabelMap: Load label-to-name mapping from CSV
// =============================================================================
void FaceRecognizer::loadLabelMap() {
    std::string mapPath = "models/label_map.csv";

    std::ifstream file(mapPath);
    if (!file.is_open()) return;

    m_labelMap.clear();
    std::string line;
    while (std::getline(file, line)) {
        size_t commaPos = line.find(',');
        if (commaPos == std::string::npos) continue;

        int label = std::stoi(line.substr(0, commaPos));
        std::string name = line.substr(commaPos + 1);
        m_labelMap[label] = name;
    }

    file.close();
}

// =============================================================================
// ensureDirectory: Create directory recursively if needed
// =============================================================================
void FaceRecognizer::ensureDirectory(const std::string& path) {
    if (!fs::exists(path)) {
        fs::create_directories(path);
    }
}
