// =============================================================================
// FaceRecognizer.h - Face Recognition using LBPH
// =============================================================================
// Provides face recognition capabilities using Local Binary Patterns
// Histograms (LBPH). Supports dataset creation, model training, and
// real-time face identification.
// =============================================================================

#ifndef FACE_RECOGNIZER_H
#define FACE_RECOGNIZER_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <string>
#include <vector>
#include <map>

// =============================================================================
// Recognition result structure
// =============================================================================
struct RecognitionResult {
    int label;              // Numeric label (person ID)
    std::string name;       // Person's name
    double confidence;      // Confidence score (lower = better match)
    bool recognized;        // True if confidence is below threshold
};

// =============================================================================
// FaceRecognizer class
// =============================================================================
class FaceRecognizer {
public:
    // -------------------------------------------------------------------------
    // Constructor
    // datasetPath: root directory for storing face images
    // modelPath:   path to save/load the trained LBPH model
    // -------------------------------------------------------------------------
    FaceRecognizer(const std::string& datasetPath = "dataset",
                   const std::string& modelPath = "models/lbph_model.yml");

    // -------------------------------------------------------------------------
    // captureDataset: Capture face images from webcam for a person
    // Saves 'numSamples' grayscale face images to dataset/<personName>/
    // -------------------------------------------------------------------------
    bool captureDataset(const std::string& personName,
                        int personLabel,
                        int numSamples = 30);

    // -------------------------------------------------------------------------
    // trainModel: Train the LBPH recognizer on all collected datasets
    // Reads all images from dataset/ subdirectories and trains the model
    // -------------------------------------------------------------------------
    bool trainModel();

    // -------------------------------------------------------------------------
    // loadModel: Load a previously trained model from disk
    // -------------------------------------------------------------------------
    bool loadModel();

    // -------------------------------------------------------------------------
    // recognize: Predict the identity of a face image
    // Input:  Grayscale face ROI (should be same size as training images)
    // Output: RecognitionResult with label, name, confidence
    // -------------------------------------------------------------------------
    RecognitionResult recognize(const cv::Mat& faceROI);

    // -------------------------------------------------------------------------
    // isModelTrained: Check if a model is loaded and ready for recognition
    // -------------------------------------------------------------------------
    bool isModelTrained() const;

    // -------------------------------------------------------------------------
    // getLabelMap: Returns the label-to-name mapping
    // -------------------------------------------------------------------------
    const std::map<int, std::string>& getLabelMap() const;

    // -------------------------------------------------------------------------
    // addLabelMapping: Manually add a label->name mapping
    // -------------------------------------------------------------------------
    void addLabelMapping(int label, const std::string& name);

    // -------------------------------------------------------------------------
    // setConfidenceThreshold: Set the max confidence for a positive match
    // Lower threshold = stricter matching
    // -------------------------------------------------------------------------
    void setConfidenceThreshold(double threshold);

    // -------------------------------------------------------------------------
    // getDatasetPath / getModelPath
    // -------------------------------------------------------------------------
    std::string getDatasetPath() const;
    std::string getModelPath() const;

private:
    // -------------------------------------------------------------------------
    // loadDatasetImages: Scan dataset directory and load all face images
    // -------------------------------------------------------------------------
    bool loadDatasetImages(std::vector<cv::Mat>& images,
                           std::vector<int>& labels);

    // -------------------------------------------------------------------------
    // saveLabelMap / loadLabelMap: Persist label-name mapping to disk
    // -------------------------------------------------------------------------
    void saveLabelMap();
    void loadLabelMap();

    // -------------------------------------------------------------------------
    // ensureDirectory: Create a directory if it doesn't exist
    // -------------------------------------------------------------------------
    void ensureDirectory(const std::string& path);

    cv::Ptr<cv::face::LBPHFaceRecognizer> m_recognizer;  // LBPH recognizer
    std::map<int, std::string> m_labelMap;                // Label -> Name
    std::string m_datasetPath;                            // Dataset root dir
    std::string m_modelPath;                              // Trained model path
    double m_confidenceThreshold;                         // Recognition threshold
    bool m_trained;                                       // Model trained flag
};

#endif // FACE_RECOGNIZER_H
