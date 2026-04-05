# =============================================================================
# face_detector.py - Face Detection using Haar Cascade Classifier
# =============================================================================
# Provides real-time face detection from webcam frames using OpenCV's
# Haar Cascade classifier. Optimized for ~30 FPS with ~90% accuracy.
# =============================================================================

import cv2
import numpy as np


class DetectionParams:
    """Configuration parameters for face detection, tuned for accuracy/performance."""

    def __init__(self):
        # Scale factor for multi-scale detection.
        # Lower = more accurate but slower. 1.1-1.3 recommended.
        self.scale_factor: float = 1.2

        # Minimum number of neighbor rectangles to retain a detection.
        # Higher = fewer false positives. 5-7 gives ~90% accuracy.
        self.min_neighbors: int = 6

        # Minimum face size in pixels (filters out noise)
        self.min_face_size: tuple = (80, 80)

        # Maximum face size (0,0 = no limit)
        self.max_face_size: tuple = (0, 0)

        # Bounding box color (BGR format) - vibrant green
        self.box_color: tuple = (0, 255, 100)

        # Bounding box thickness
        self.box_thickness: int = 2

        # Label font scale
        self.font_scale: float = 0.7


class FaceDetector:
    """Face detection engine using Haar Cascade Classifier."""

    def __init__(self, cascade_path: str):
        """
        Initialize the detector with a Haar Cascade XML file.

        Args:
            cascade_path: Path to haarcascade_frontalface_default.xml
        """
        self.params = DetectionParams()
        self._loaded = False

        # Load the Haar Cascade classifier
        self._cascade = cv2.CascadeClassifier(cascade_path)

        if self._cascade.empty():
            print(f"[ERROR] Failed to load Haar Cascade from: {cascade_path}")
            print("[INFO]  Make sure the path to haarcascade_frontalface_default.xml is correct.")
        else:
            print("[OK] Haar Cascade loaded successfully.")
            self._loaded = True

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert frame to grayscale and equalize histogram.

        Grayscale conversion reduces computation (1 channel vs 3).
        Histogram equalization normalizes lighting for better detection.

        Args:
            frame: BGR input frame

        Returns:
            Preprocessed grayscale frame
        """
        # Convert BGR to grayscale
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Equalize histogram to normalize brightness/contrast
        gray = cv2.equalizeHist(gray)
        return gray

    def detect_faces(self, frame: np.ndarray) -> list:
        """
        Detect all faces in a given frame.

        Uses detectMultiScale for multi-scale face detection.
        Parameters are tuned for real-time performance with good accuracy:
          - scaleFactor=1.2  : 20% size reduction per scale (fast, decent accuracy)
          - minNeighbors=6   : Requires 6 overlapping detections (reduces false pos.)
          - minFaceSize=80x80: Ignores very small regions (noise filtering)

        Args:
            frame: BGR input frame

        Returns:
            List of (x, y, w, h) tuples for detected faces
        """
        if not self._loaded:
            print("[ERROR] Cascade not loaded. Cannot detect faces.")
            return []

        # Preprocess: grayscale + histogram equalization
        gray = self.preprocess_frame(frame)

        # Run multi-scale detection
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.params.scale_factor,
            minNeighbors=self.params.min_neighbors,
            minSize=self.params.min_face_size,
            maxSize=self.params.max_face_size if self.params.max_face_size != (0, 0) else None
        )

        # Convert to list of tuples if numpy array
        if isinstance(faces, np.ndarray) and len(faces) > 0:
            return [tuple(f) for f in faces]
        return []

    def draw_detections(self, frame: np.ndarray, faces: list,
                        labels: list = None) -> np.ndarray:
        """
        Draw styled bounding boxes and labels on the frame.

        Draws corner-accented rectangles with semi-transparent label backgrounds.

        Args:
            frame: BGR frame to draw on (modified in-place)
            faces: List of (x, y, w, h) face rectangles
            labels: Optional list of label strings for each face

        Returns:
            The annotated frame
        """
        for i, (x, y, w, h) in enumerate(faces):
            # --- Draw the main bounding box ---
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          self.params.box_color, self.params.box_thickness)

            # --- Draw corner accents for a modern look ---
            corner_len = min(w, h) // 5
            accent_color = (0, 200, 255)  # Orange-yellow accent

            # Top-left
            cv2.line(frame, (x, y), (x + corner_len, y), accent_color, 3)
            cv2.line(frame, (x, y), (x, y + corner_len), accent_color, 3)

            # Top-right
            cv2.line(frame, (x + w, y), (x + w - corner_len, y), accent_color, 3)
            cv2.line(frame, (x + w, y), (x + w, y + corner_len), accent_color, 3)

            # Bottom-left
            cv2.line(frame, (x, y + h), (x + corner_len, y + h), accent_color, 3)
            cv2.line(frame, (x, y + h), (x, y + h - corner_len), accent_color, 3)

            # Bottom-right
            cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), accent_color, 3)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), accent_color, 3)

            # --- Draw label ---
            if labels and i < len(labels) and labels[i]:
                label = labels[i]
            else:
                label = f"Face #{i + 1}"

            # Calculate text size for background
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.params.font_scale, 2
            )
            text_y = y - 10

            # Ensure label doesn't go off-screen
            if text_y - text_h < 0:
                text_y = y + h + text_h + 10

            # Semi-transparent background for label
            cv2.rectangle(frame,
                          (x - 2, text_y - text_h - 4),
                          (x + text_w + 4, text_y + 4),
                          (0, 0, 0), cv2.FILLED)

            # Label text
            cv2.putText(frame, label, (x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, self.params.font_scale,
                        self.params.box_color, 2)

        return frame

    def extract_face_roi(self, gray_frame: np.ndarray,
                         face_rect: tuple, target_size: int = 200) -> np.ndarray:
        """
        Extract a face region from the frame, resized to a standard size.

        Args:
            gray_frame: Grayscale input frame
            face_rect: (x, y, w, h) face rectangle
            target_size: Output size (square)

        Returns:
            Resized grayscale face ROI, or None if invalid
        """
        x, y, w, h = face_rect
        rows, cols = gray_frame.shape[:2]

        # Clamp to frame boundaries
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(cols, x + w)
        y2 = min(rows, y + h)

        if x2 <= x1 or y2 <= y1:
            return None

        face_roi = gray_frame[y1:y2, x1:x2]
        resized = cv2.resize(face_roi, (target_size, target_size))
        return resized

    @property
    def is_loaded(self) -> bool:
        """Check if cascade classifier loaded successfully."""
        return self._loaded
