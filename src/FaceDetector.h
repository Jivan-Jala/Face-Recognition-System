// =============================================================================
// FaceDetector.h - Face Detection using Haar Cascade Classifier
// =============================================================================
// Provides real-time face detection from webcam frames using OpenCV's
// Haar Cascade classifier. Optimized for ~30 FPS with ~90% accuracy.
// =============================================================================

#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <string>
#include <vector>

// =============================================================================
// Detection parameters - tuned for accuracy/performance balance
// =============================================================================
struct DetectionParams {
    // Scale factor for multi-scale detection.
    // Lower = more accurate but slower. 1.1-1.3 recommended.
    double scaleFactor = 1.2;

    // Minimum number of neighbor rectangles to retain a detection.
    // Higher = fewer false positives. 5-7 gives ~90% accuracy.
    int minNeighbors = 6;

    // Minimum face size in pixels (filters out noise)
    cv::Size minFaceSize = cv::Size(80, 80);

    // Maximum face size (0,0 = no limit)
    cv::Size maxFaceSize = cv::Size(0, 0);

    // Bounding box color (BGR format) - vibrant green
    cv::Scalar boxColor = cv::Scalar(0, 255, 100);

    // Bounding box thickness
    int boxThickness = 2;

    // Label font scale
    double fontScale = 0.7;
};

// =============================================================================
// FaceDetector class
// =============================================================================
class FaceDetector {
public:
    // -------------------------------------------------------------------------
    // Constructor: loads the Haar Cascade classifier from the given XML path
    // -------------------------------------------------------------------------
    explicit FaceDetector(const std::string& cascadePath);

    // -------------------------------------------------------------------------
    // detectFaces: Detects all faces in a given frame
    // Input:  BGR frame from camera
    // Output: Vector of bounding rectangles for detected faces
    // -------------------------------------------------------------------------
    std::vector<cv::Rect> detectFaces(const cv::Mat& frame);

    // -------------------------------------------------------------------------
    // drawDetections: Draws bounding boxes and labels on the frame
    // Modifies the frame in-place
    // -------------------------------------------------------------------------
    void drawDetections(cv::Mat& frame,
                        const std::vector<cv::Rect>& faces,
                        const std::vector<std::string>& labels = {});

    // -------------------------------------------------------------------------
    // preprocessFrame: Converts frame to grayscale and equalizes histogram
    // for better detection performance
    // -------------------------------------------------------------------------
    cv::Mat preprocessFrame(const cv::Mat& frame);

    // -------------------------------------------------------------------------
    // extractFaceROI: Extracts a face region from the frame, resized to a
    // standard size for recognition training
    // -------------------------------------------------------------------------
    cv::Mat extractFaceROI(const cv::Mat& grayFrame, const cv::Rect& faceRect,
                           int targetSize = 200);

    // -------------------------------------------------------------------------
    // isLoaded: Returns true if the cascade classifier loaded successfully
    // -------------------------------------------------------------------------
    bool isLoaded() const;

    // -------------------------------------------------------------------------
    // Getters/Setters for detection parameters
    // -------------------------------------------------------------------------
    DetectionParams& getParams();
    void setParams(const DetectionParams& params);

private:
    cv::CascadeClassifier m_cascade;   // Haar cascade classifier
    DetectionParams m_params;           // Detection parameters
    bool m_loaded;                      // Whether cascade loaded successfully
};

#endif // FACE_DETECTOR_H
