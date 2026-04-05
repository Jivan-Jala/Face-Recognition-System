# =============================================================================
# face_recognizer.py - Face Recognition using LBPH
# =============================================================================
# Provides face recognition capabilities using Local Binary Patterns
# Histograms (LBPH). Supports dataset creation, model training, and
# real-time face identification.
# =============================================================================

import cv2
import numpy as np
import os
import csv
from dataclasses import dataclass, field
from face_detector import FaceDetector


@dataclass
class RecognitionResult:
    """Result from a face recognition prediction."""
    label: int = -1            # Numeric label (person ID)
    name: str = "Unknown"      # Person's name
    confidence: float = 999.0  # Confidence score (lower = better match)
    recognized: bool = False   # True if confidence is below threshold


class FaceRecognizer:
    """
    Face recognition engine using LBPH (Local Binary Patterns Histograms).

    LBPH Parameters:
      radius=1      : Radius of circular LBP pattern
      neighbors=8   : Number of sample points in LBP
      grid_x=8      : Horizontal grid cell count
      grid_y=8      : Vertical grid cell count
    """

    def __init__(self, dataset_path: str = "dataset",
                 model_path: str = "models/lbph_model.yml"):
        """
        Initialize the LBPH face recognizer.

        Args:
            dataset_path: Root directory for storing face images
            model_path: Path to save/load the trained LBPH model
        """
        self._dataset_path = dataset_path
        self._model_path = model_path
        self._confidence_threshold = 80.0  # Below this = recognized
        self._trained = False
        self._label_map: dict[int, str] = {}

        # Create LBPH face recognizer
        self._recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8
        )

        # Ensure required directories exist
        os.makedirs(self._dataset_path, exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("saved_faces", exist_ok=True)

        # Load existing label map if available
        self._load_label_map()

        print("[OK] Face Recognizer initialized.")
        print(f"     Dataset path: {self._dataset_path}")
        print(f"     Model path:   {self._model_path}")

    def capture_dataset(self, person_name: str, person_label: int,
                        num_samples: int = 30) -> bool:
        """
        Capture face samples from webcam for training.

        Opens the webcam, detects faces, and saves grayscale face ROIs to:
          dataset/<person_name>/face_0001.jpg, face_0002.jpg, ...
        The user should move their head slightly for varied training data.

        Args:
            person_name: Name of the person
            person_label: Numeric label ID
            num_samples: Number of face images to capture

        Returns:
            True if capture was successful
        """
        # Create directory for this person
        person_dir = os.path.join(self._dataset_path, person_name)
        os.makedirs(person_dir, exist_ok=True)

        # Add to label map
        self._label_map[person_label] = person_name
        self._save_label_map()

        # Open webcam
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("[ERROR] Cannot open webcam for dataset capture.")
            return False

        # Set camera properties
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Create a face detector
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = FaceDetector(cascade_path)

        if not detector.is_loaded:
            print("[ERROR] Cannot load face detector for capture.")
            camera.release()
            return False

        print(f"\n{'=' * 40}")
        print(f"  DATASET CAPTURE: {person_name}")
        print(f"  Label ID:        {person_label}")
        print(f"  Samples needed:  {num_samples}")
        print(f"{'=' * 40}")
        print("Look at the camera. Move your head slowly.")
        print("Press 'q' to cancel.\n")

        captured = 0
        frame_count = 0

        while captured < num_samples:
            ret, frame = camera.read()
            if not ret or frame is None:
                break

            frame_count += 1
            should_capture = (frame_count % 5 == 0)  # Every 5th frame for variety

            # Detect faces
            faces = detector.detect_faces(frame)

            if faces and should_capture:
                # Use the largest face
                largest = max(faces, key=lambda f: f[2] * f[3])

                # Extract grayscale face ROI
                gray = detector.preprocess_frame(frame)
                face_roi = detector.extract_face_roi(gray, largest, 200)

                if face_roi is not None:
                    # Save the face image
                    filename = os.path.join(person_dir, f"face_{captured + 1:04d}.jpg")
                    cv2.imwrite(filename, face_roi)
                    captured += 1

                    # Flash green border to indicate capture
                    x, y, w, h = largest
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

            # Draw all detected faces
            detector.draw_detections(frame, faces)

            # --- Draw progress bar ---
            progress = captured / num_samples
            bar_width = 300
            bar_height = 25
            bar_x = (frame.shape[1] - bar_width) // 2
            bar_y = frame.shape[0] - 50

            # Background
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + bar_width, bar_y + bar_height),
                          (50, 50, 50), cv2.FILLED)
            # Fill
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + int(bar_width * progress), bar_y + bar_height),
                          (0, 200, 100), cv2.FILLED)
            # Border
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + bar_width, bar_y + bar_height),
                          (200, 200, 200), 1)
            # Progress text
            cv2.putText(frame, f"Captured: {captured} / {num_samples}",
                        (bar_x + 10, bar_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display
            cv2.imshow(f"Dataset Capture - {person_name}", frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                print("[INFO] Capture cancelled by user.")
                camera.release()
                cv2.destroyWindow(f"Dataset Capture - {person_name}")
                return False

        camera.release()
        cv2.destroyWindow(f"Dataset Capture - {person_name}")

        print(f"\n[OK] Captured {captured} samples for '{person_name}'.")
        return True

    def train_model(self) -> bool:
        """
        Train the LBPH recognizer on all collected datasets.

        Reads all images from dataset/ subdirectories and trains the model.
        Saves the trained model to disk.

        Returns:
            True if training was successful
        """
        images, labels = self._load_dataset_images()

        if not images:
            print(f"[ERROR] No training images found in: {self._dataset_path}")
            return False

        print(f"[INFO] Training LBPH model with {len(images)} images...")

        # Train the recognizer
        self._recognizer.train(images, np.array(labels))

        # Save the trained model
        os.makedirs(os.path.dirname(self._model_path), exist_ok=True)
        self._recognizer.save(self._model_path)
        self._save_label_map()

        self._trained = True

        print(f"[OK] Model trained and saved to: {self._model_path}")
        print(f"[OK] Labels: {len(self._label_map)} people registered.")
        for label, name in self._label_map.items():
            print(f"     Label {label} -> {name}")

        return True

    def load_model(self) -> bool:
        """
        Load a previously trained model from disk.

        Returns:
            True if model was loaded successfully
        """
        if not os.path.exists(self._model_path):
            print(f"[WARNING] No trained model found at: {self._model_path}")
            return False

        try:
            self._recognizer.read(self._model_path)
            self._trained = True
            self._load_label_map()

            print(f"[OK] Model loaded from: {self._model_path}")
            print(f"[OK] {len(self._label_map)} people in model.")
            return True
        except cv2.error as e:
            print(f"[ERROR] Failed to load model: {e}")
            return False

    def recognize(self, face_roi: np.ndarray) -> RecognitionResult:
        """
        Predict the identity of a face image.

        Args:
            face_roi: Grayscale face ROI (will be resized to 200x200)

        Returns:
            RecognitionResult with label, name, confidence, recognized flag
        """
        result = RecognitionResult()

        if not self._trained or face_roi is None:
            return result

        # Ensure grayscale and correct size
        if len(face_roi.shape) == 3:
            processed = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            processed = face_roi
        processed = cv2.resize(processed, (200, 200))

        # Predict
        predicted_label, confidence = self._recognizer.predict(processed)

        result.label = predicted_label
        result.confidence = confidence

        # Check threshold
        if confidence < self._confidence_threshold:
            result.recognized = True
            result.name = self._label_map.get(predicted_label,
                                               f"Person_{predicted_label}")
        return result

    @property
    def is_model_trained(self) -> bool:
        """Check if a model is loaded and ready for recognition."""
        return self._trained

    @property
    def label_map(self) -> dict:
        """Returns the label-to-name mapping."""
        return self._label_map.copy()

    @property
    def dataset_path(self) -> str:
        return self._dataset_path

    @property
    def model_path(self) -> str:
        return self._model_path

    def set_confidence_threshold(self, threshold: float):
        """Set the max confidence for a positive match. Lower = stricter."""
        self._confidence_threshold = threshold

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _load_dataset_images(self) -> tuple:
        """
        Scan dataset/ directory for face images.

        Expected structure:
          dataset/
            PersonA/
              face_0001.jpg
            PersonB/
              face_0001.jpg

        Returns:
            Tuple of (images_list, labels_list)
        """
        images = []
        labels = []

        if not os.path.exists(self._dataset_path):
            print(f"[ERROR] Dataset directory not found: {self._dataset_path}")
            return images, labels

        for person_name in os.listdir(self._dataset_path):
            person_dir = os.path.join(self._dataset_path, person_name)
            if not os.path.isdir(person_dir):
                continue

            # Find label for this person
            label = -1
            for l, n in self._label_map.items():
                if n == person_name:
                    label = l
                    break

            # If not in map, assign new label
            if label == -1:
                label = max(self._label_map.keys(), default=-1) + 1
                self._label_map[label] = person_name

            count = 0
            for img_file in os.listdir(person_dir):
                ext = os.path.splitext(img_file)[1].lower()
                if ext not in ('.jpg', '.jpeg', '.png', '.bmp'):
                    continue

                img_path = os.path.join(person_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                img = cv2.resize(img, (200, 200))
                images.append(img)
                labels.append(label)
                count += 1

            print(f"  Loaded {count} images for '{person_name}' (label={label})")

        return images, labels

    def _save_label_map(self):
        """Save label-to-name mapping as CSV."""
        map_path = "models/label_map.csv"
        os.makedirs("models", exist_ok=True)

        with open(map_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for label, name in self._label_map.items():
                writer.writerow([label, name])

    def _load_label_map(self):
        """Load label-to-name mapping from CSV."""
        map_path = "models/label_map.csv"
        if not os.path.exists(map_path):
            return

        self._label_map.clear()
        with open(map_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    self._label_map[int(row[0])] = row[1]
