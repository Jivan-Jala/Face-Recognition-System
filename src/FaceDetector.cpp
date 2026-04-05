// =============================================================================
// FaceDetector.cpp - Face Detection Implementation
// =============================================================================

#include "FaceDetector.h"
#include <iostream>

// =============================================================================
// Constructor: Load Haar Cascade from file
// =============================================================================
FaceDetector::FaceDetector(const std::string& cascadePath)
    : m_loaded(false)
{
    // Attempt to load the Haar Cascade classifier
    if (!m_cascade.load(cascadePath)) {
        std::cerr << "[ERROR] Failed to load Haar Cascade from: "
                  << cascadePath << std::endl;
        std::cerr << "[INFO]  Make sure the path to "
                  << "haarcascade_frontalface_default.xml is correct."
                  << std::endl;
        m_loaded = false;
    } else {
        std::cout << "[OK] Haar Cascade loaded successfully." << std::endl;
        m_loaded = true;
    }
}

// =============================================================================
// preprocessFrame: Convert to grayscale + histogram equalization
// =============================================================================
// Grayscale conversion reduces computation (1 channel vs 3).
// Histogram equalization normalizes lighting for better detection.
// =============================================================================
cv::Mat FaceDetector::preprocessFrame(const cv::Mat& frame) {
    cv::Mat gray;

    // Convert BGR to grayscale
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame.clone();
    }

    // Equalize histogram to normalize brightness/contrast
    cv::equalizeHist(gray, gray);

    return gray;
}

// =============================================================================
// detectFaces: Run Haar Cascade detection on a frame
// =============================================================================
// Uses detectMultiScale for multi-scale face detection.
// Parameters are tuned for real-time performance with good accuracy:
//   - scaleFactor=1.2  : 20% size reduction per scale (fast, decent accuracy)
//   - minNeighbors=6   : Requires 6 overlapping detections (reduces false pos.)
//   - minFaceSize=80x80: Ignores very small regions (noise filtering)
// =============================================================================
std::vector<cv::Rect> FaceDetector::detectFaces(const cv::Mat& frame) {
    std::vector<cv::Rect> faces;

    if (!m_loaded) {
        std::cerr << "[ERROR] Cascade not loaded. Cannot detect faces."
                  << std::endl;
        return faces;
    }

    // Preprocess: grayscale + histogram equalization
    cv::Mat gray = preprocessFrame(frame);

    // Run multi-scale detection
    m_cascade.detectMultiScale(
        gray,                    // Input grayscale image
        faces,                   // Output vector of face rectangles
        m_params.scaleFactor,    // Scale factor (1.2 = 20% reduction)
        m_params.minNeighbors,   // Min neighbors for filtering
        0,                       // Flags (0 = default)
        m_params.minFaceSize,    // Minimum face size
        m_params.maxFaceSize     // Maximum face size
    );

    return faces;
}

// =============================================================================
// drawDetections: Draw bounding boxes and labels on the frame
// =============================================================================
// Draws rounded-corner-style rectangles with labels for each detected face.
// If labels are provided, they are displayed above each bounding box.
// =============================================================================
void FaceDetector::drawDetections(cv::Mat& frame,
                                   const std::vector<cv::Rect>& faces,
                                   const std::vector<std::string>& labels) {
    for (size_t i = 0; i < faces.size(); i++) {
        const cv::Rect& face = faces[i];

        // Draw the bounding box
        cv::rectangle(frame, face, m_params.boxColor, m_params.boxThickness);

        // Draw corner accents for a modern look
        int cornerLen = std::min(face.width, face.height) / 5;
        cv::Scalar accentColor(0, 200, 255); // Orange-yellow accent

        // Top-left corner
        cv::line(frame, face.tl(),
                 cv::Point(face.x + cornerLen, face.y),
                 accentColor, 3);
        cv::line(frame, face.tl(),
                 cv::Point(face.x, face.y + cornerLen),
                 accentColor, 3);

        // Top-right corner
        cv::line(frame, cv::Point(face.x + face.width, face.y),
                 cv::Point(face.x + face.width - cornerLen, face.y),
                 accentColor, 3);
        cv::line(frame, cv::Point(face.x + face.width, face.y),
                 cv::Point(face.x + face.width, face.y + cornerLen),
                 accentColor, 3);

        // Bottom-left corner
        cv::line(frame, cv::Point(face.x, face.y + face.height),
                 cv::Point(face.x + cornerLen, face.y + face.height),
                 accentColor, 3);
        cv::line(frame, cv::Point(face.x, face.y + face.height),
                 cv::Point(face.x, face.y + face.height - cornerLen),
                 accentColor, 3);

        // Bottom-right corner
        cv::line(frame, face.br(),
                 cv::Point(face.x + face.width - cornerLen, face.y + face.height),
                 accentColor, 3);
        cv::line(frame, face.br(),
                 cv::Point(face.x + face.width, face.y + face.height - cornerLen),
                 accentColor, 3);

        // Draw label if available
        std::string label;
        if (i < labels.size() && !labels[i].empty()) {
            label = labels[i];
        } else {
            label = "Face #" + std::to_string(i + 1);
        }

        // Background rectangle for the label text
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                             m_params.fontScale, 2, &baseline);
        cv::Point textOrg(face.x, face.y - 10);

        // Ensure label doesn't go off-screen
        if (textOrg.y - textSize.height < 0) {
            textOrg.y = face.y + face.height + textSize.height + 10;
        }

        // Draw semi-transparent background for label
        cv::rectangle(frame,
                      cv::Point(textOrg.x - 2, textOrg.y - textSize.height - 4),
                      cv::Point(textOrg.x + textSize.width + 4, textOrg.y + 4),
                      cv::Scalar(0, 0, 0), cv::FILLED);

        // Draw the label text
        cv::putText(frame, label, textOrg,
                    cv::FONT_HERSHEY_SIMPLEX, m_params.fontScale,
                    m_params.boxColor, 2);
    }
}

// =============================================================================
// extractFaceROI: Extract and resize a face region for recognition
// =============================================================================
cv::Mat FaceDetector::extractFaceROI(const cv::Mat& grayFrame,
                                      const cv::Rect& faceRect,
                                      int targetSize) {
    // Clamp the rectangle to frame boundaries
    cv::Rect safeRect = faceRect & cv::Rect(0, 0, grayFrame.cols, grayFrame.rows);

    if (safeRect.empty()) {
        return cv::Mat();
    }

    // Extract the face region
    cv::Mat faceROI = grayFrame(safeRect);

    // Resize to standard size for consistent recognition
    cv::Mat resized;
    cv::resize(faceROI, resized, cv::Size(targetSize, targetSize));

    return resized;
}

// =============================================================================
// isLoaded: Check if cascade classifier is ready
// =============================================================================
bool FaceDetector::isLoaded() const {
    return m_loaded;
}

// =============================================================================
// Parameter accessors
// =============================================================================
DetectionParams& FaceDetector::getParams() {
    return m_params;
}

void FaceDetector::setParams(const DetectionParams& params) {
    m_params = params;
}
