# =============================================================================
# main.py - Real-Time Face Recognition System
# =============================================================================
# A complete face detection and recognition system built with OpenCV.
#
# Features:
#   1. Real-time face detection using Haar Cascade Classifier
#   2. Face recognition using LBPH (Local Binary Patterns Histograms)
#   3. Dataset creation from webcam
#   4. Model training and persistence
#   5. Live FPS display with color-coded indicator
#   6. Multiple face handling
#   7. Saved face snapshots
#
# Controls (during live feed):
#   q / ESC  - Quit / Return to menu
#   s        - Save snapshot of current frame
#   p        - Pause / Resume
#   +/-      - Increase / Decrease detection sensitivity
#
# Usage:
#   python main.py
#
# =============================================================================

import cv2
import numpy as np
import os
import time
from datetime import datetime

from face_detector import FaceDetector
from face_recognizer import FaceRecognizer

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
DATASET_PATH = "dataset"
MODEL_PATH = "models/lbph_model.yml"
SAVED_FACES_PATH = "saved_faces"

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_ID = 0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_cascade_path() -> str:
    """Locate the Haar Cascade XML file."""
    # OpenCV ships with cascades — use the built-in path
    builtin = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(builtin):
        print(f"[OK] Found cascade at: {builtin}")
        return builtin

    # Check local directory
    local = "haarcascade_frontalface_default.xml"
    if os.path.exists(local):
        print(f"[OK] Found cascade at: {local}")
        return local

    # Ask the user
    print("\n[ERROR] Haar Cascade file not found in any default location.")
    print("[INFO]  Download from: https://github.com/opencv/opencv/tree/master/data/haarcascades")
    user_path = input("\nEnter the full path to haarcascade_frontalface_default.xml: ").strip()
    if os.path.exists(user_path):
        return user_path

    return ""


def capture_frame(camera: cv2.VideoCapture):
    """Read a frame from the camera."""
    ret, frame = camera.read()
    if ret and frame is not None:
        return frame
    return None


def display_output(window_name: str, frame: np.ndarray):
    """Show a frame in a named window."""
    cv2.imshow(window_name, frame)


def get_timestamp() -> str:
    """Get current timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def draw_hud(frame: np.ndarray, fps: float, mode: str, face_count: int):
    """
    Draw heads-up display overlay (FPS, mode, face count).

    Features:
      - Semi-transparent top bar
      - Color-coded FPS indicator (green/yellow/red)
      - Mode name and face count
      - Keyboard shortcut bar at bottom
    """
    h, w = frame.shape[:2]

    # --- Semi-transparent overlay bar at the top ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # --- FPS indicator with color coding ---
    if fps >= 25.0:
        fps_color = (0, 255, 100)    # Green - great
    elif fps >= 15.0:
        fps_color = (0, 200, 255)    # Yellow - okay
    else:
        fps_color = (0, 80, 255)     # Red - slow

    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)

    # --- Mode indicator ---
    cv2.putText(frame, f"Mode: {mode}", (180, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # --- Face count ---
    cv2.putText(frame, f"Faces: {face_count}", (w - 150, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # --- Bottom info bar ---
    bottom_overlay = frame.copy()
    cv2.rectangle(bottom_overlay, (0, h - 30), (w, h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(bottom_overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "Q:Quit  S:Snapshot  P:Pause  +/-:Sensitivity",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)


def open_camera() -> cv2.VideoCapture:
    """Initialize webcam with error handling."""
    print(f"[INFO] Opening camera {CAMERA_ID}...")

    camera = cv2.VideoCapture(CAMERA_ID)

    if not camera.isOpened():
        print(f"[ERROR] Cannot open webcam (ID={CAMERA_ID}).")
        print("[INFO]  Possible causes:")
        print("         - No camera connected")
        print("         - Camera in use by another application")
        print("         - Incorrect camera ID")
        return None

    # Set resolution for performance
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera.get(cv2.CAP_PROP_FPS)

    print(f"[OK] Camera opened: {actual_w}x{actual_h} @ {actual_fps:.0f} FPS")
    return camera


# =============================================================================
# MODE 1: DETECTION ONLY
# =============================================================================
def run_detection_mode(detector: FaceDetector):
    """
    Real-time face detection without recognition.
    Draws bounding boxes on all detected faces.
    """
    camera = open_camera()
    if camera is None:
        return

    print("\n=== DETECTION MODE ===")
    print("Press 'q' or ESC to return to menu.\n")

    # FPS calculation
    prev_time = time.time()
    fps = 0.0
    frame_count = 0
    paused = False
    frame = None

    while True:
        if not paused:
            frame = capture_frame(camera)
            if frame is None:
                print("[ERROR] Failed to capture frame.")
                break

        display_frame = frame.copy()

        # Detect faces
        faces = detector.detect_faces(display_frame)

        # Draw bounding boxes
        detector.draw_detections(display_frame, faces)

        # Calculate FPS (rolling average, updated every 0.5s)
        frame_count += 1
        current_time = time.time()
        elapsed = current_time - prev_time

        if elapsed >= 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            prev_time = current_time

        # Draw HUD
        draw_hud(display_frame, fps, "DETECTION", len(faces))

        if paused:
            cv2.putText(display_frame, "PAUSED",
                        (display_frame.shape[1] // 2 - 80,
                         display_frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Show frame
        display_output("Face Detection", display_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q') or key == 27:
            break
        elif key == ord('p') or key == ord('P'):
            paused = not paused
        elif key == ord('s') or key == ord('S'):
            filename = f"{SAVED_FACES_PATH}/snapshot_{get_timestamp()}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[OK] Snapshot saved: {filename}")
        elif key == ord('+') or key == ord('='):
            detector.params.min_neighbors = min(15, detector.params.min_neighbors + 1)
            print(f"[INFO] minNeighbors = {detector.params.min_neighbors}")
        elif key == ord('-') or key == ord('_'):
            detector.params.min_neighbors = max(1, detector.params.min_neighbors - 1)
            print(f"[INFO] minNeighbors = {detector.params.min_neighbors}")

    camera.release()
    cv2.destroyAllWindows()


# =============================================================================
# MODE 2: RECOGNITION MODE
# =============================================================================
def run_recognition_mode(detector: FaceDetector, recognizer: FaceRecognizer):
    """
    Real-time face detection + LBPH recognition.
    Shows detected person names with confidence scores.
    """
    # Ensure model is trained
    if not recognizer.is_model_trained:
        print("[INFO] No trained model found. Attempting to load...")
        if not recognizer.load_model():
            print("[ERROR] No trained model available.")
            print("[INFO]  Run option 3 (Create Dataset) first, then option 4 (Train Model).")
            return

    camera = open_camera()
    if camera is None:
        return

    print("\n=== RECOGNITION MODE ===")
    print("Press 'q' or ESC to return to menu.\n")

    prev_time = time.time()
    fps = 0.0
    frame_count = 0
    paused = False
    frame = None

    while True:
        if not paused:
            frame = capture_frame(camera)
            if frame is None:
                print("[ERROR] Failed to capture frame.")
                break

        display_frame = frame.copy()

        # Detect faces
        faces = detector.detect_faces(display_frame)

        # Recognize each face
        labels = []
        gray = detector.preprocess_frame(frame)

        for face_rect in faces:
            face_roi = detector.extract_face_roi(gray, face_rect, 200)

            if face_roi is not None:
                result = recognizer.recognize(face_roi)

                if result.recognized:
                    conf_pct = max(0, 100.0 - result.confidence)
                    label = f"{result.name} ({conf_pct:.0f}%)"
                else:
                    label = "Unknown"
                labels.append(label)
            else:
                labels.append("Unknown")

        # Draw detections with recognition labels
        detector.draw_detections(display_frame, faces, labels)

        # Calculate FPS
        frame_count += 1
        current_time = time.time()
        elapsed = current_time - prev_time
        if elapsed >= 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            prev_time = current_time

        # Draw HUD
        draw_hud(display_frame, fps, "RECOGNITION", len(faces))

        if paused:
            cv2.putText(display_frame, "PAUSED",
                        (display_frame.shape[1] // 2 - 80,
                         display_frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        display_output("Face Recognition", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break
        elif key == ord('p') or key == ord('P'):
            paused = not paused
        elif key == ord('s') or key == ord('S'):
            # Save full frame and individual faces
            filename = f"{SAVED_FACES_PATH}/recognized_{get_timestamp()}.jpg"
            cv2.imwrite(filename, frame)
            for i, face_rect in enumerate(faces):
                face_roi = detector.extract_face_roi(gray, face_rect, 200)
                if face_roi is not None:
                    face_file = f"{SAVED_FACES_PATH}/face_{i}_{get_timestamp()}.jpg"
                    cv2.imwrite(face_file, face_roi)
            print("[OK] Snapshot and faces saved.")

    camera.release()
    cv2.destroyAllWindows()


# =============================================================================
# MODE 3: DATASET CREATION
# =============================================================================
def run_dataset_creation(recognizer: FaceRecognizer):
    """Interactive menu to create training datasets for new people."""
    print("\n=== DATASET CREATION ===")

    # Show existing datasets
    if os.path.exists(DATASET_PATH):
        print("\nExisting datasets:")
        count = 0
        for entry in os.listdir(DATASET_PATH):
            entry_path = os.path.join(DATASET_PATH, entry)
            if os.path.isdir(entry_path):
                num_images = len([f for f in os.listdir(entry_path)
                                  if os.path.isfile(os.path.join(entry_path, f))])
                print(f"  {entry} ({num_images} images)")
                count += 1
        if count == 0:
            print("  (none)")

    # Get person's name
    name = input("\nEnter the person's name (or 'cancel' to go back): ").strip()

    if not name or name.lower() == 'cancel':
        print("[INFO] Dataset creation cancelled.")
        return

    # Get label ID
    label_map = recognizer.label_map
    label = max(label_map.keys(), default=-1) + 1

    # Check if person already exists
    for l, n in label_map.items():
        if n == name:
            label = l
            print(f"[INFO] Person '{name}' already exists with label {label}. Adding more samples.")
            break

    # Get number of samples
    num_str = input("Number of samples to capture (default 30): ").strip()
    num_samples = int(num_str) if num_str.isdigit() else 30
    num_samples = max(5, min(200, num_samples))

    print(f"\n[INFO] Will capture {num_samples} samples for '{name}' (label={label})")
    confirm = input("[INFO] Press ENTER to start, or 'c' to cancel: ").strip()

    if confirm.lower() == 'c':
        print("[INFO] Cancelled.")
        return

    recognizer.capture_dataset(name, label, num_samples)


# =============================================================================
# MODE 4: TRAIN MODEL
# =============================================================================
def run_training(recognizer: FaceRecognizer):
    """Train the LBPH model on all collected datasets."""
    print("\n=== MODEL TRAINING ===")

    if not os.path.exists(DATASET_PATH):
        print("[ERROR] No dataset directory found.")
        print("[INFO]  Create a dataset first (option 3).")
        return

    # Count images
    total_images = 0
    total_people = 0
    for entry in os.listdir(DATASET_PATH):
        entry_path = os.path.join(DATASET_PATH, entry)
        if os.path.isdir(entry_path):
            total_people += 1
            total_images += len([f for f in os.listdir(entry_path)
                                 if os.path.isfile(os.path.join(entry_path, f))])

    if total_images == 0:
        print("[ERROR] No training images found.")
        return

    print(f"[INFO] Found {total_images} images for {total_people} people.")
    print("[INFO] Training LBPH model...")

    start_time = time.time()
    success = recognizer.train_model()
    elapsed = time.time() - start_time

    if success:
        print(f"[OK] Training completed in {elapsed:.2f} seconds.")
    else:
        print("[ERROR] Training failed.")


# =============================================================================
# MAIN MENU
# =============================================================================
def print_menu():
    """Display the main interactive menu."""
    print(f"""
  {'=' * 50}
     REAL-TIME FACE RECOGNITION SYSTEM  v1.0
  {'=' * 50}
     Built with OpenCV {cv2.__version__} + LBPH
  {'=' * 50}

  [1] Face Detection Only    (real-time webcam)
  [2] Face Recognition       (detect + identify)
  [3] Create Dataset         (capture face samples)
  [4] Train Model            (train LBPH recognizer)
  [5] Exit
""")


def main():
    """Main entry point for the Face Recognition System."""
    print("\n[INFO] Initializing Face Recognition System...\n")

    # ---- Step 1: Find the Haar Cascade file ----
    cascade_path = find_cascade_path()
    if not cascade_path:
        print("[FATAL] Cannot find Haar Cascade file. Exiting.")
        return

    # ---- Step 2: Initialize the Face Detector ----
    detector = FaceDetector(cascade_path)
    if not detector.is_loaded:
        print("[FATAL] Failed to initialize face detector. Exiting.")
        return

    # ---- Step 3: Initialize the Face Recognizer ----
    recognizer = FaceRecognizer(DATASET_PATH, MODEL_PATH)

    # Try to load existing model
    if os.path.exists(MODEL_PATH):
        recognizer.load_model()

    # ---- Step 4: Create required directories ----
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(SAVED_FACES_PATH, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("[OK] System initialized successfully.")

    # ---- Step 5: Main menu loop ----
    while True:
        print_menu()
        choice = input("  Select option [1-5]: ").strip()

        if choice == "1":
            run_detection_mode(detector)
        elif choice == "2":
            run_recognition_mode(detector, recognizer)
        elif choice == "3":
            run_dataset_creation(recognizer)
        elif choice == "4":
            run_training(recognizer)
        elif choice == "5":
            print("\n[INFO] Shutting down. Goodbye!\n")
            break
        else:
            print("[WARNING] Invalid option. Please enter 1-5.")


if __name__ == "__main__":
    main()
