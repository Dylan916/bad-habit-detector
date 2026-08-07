"""
Main entry point for Bad Habit Detector.
Runs real-time webcam processing loop, integrating landmark tracking, YOLO object detection,
rules engine, non-blocking audio alerts, CSV logging, and HUD UI rendering.
"""

import cv2
import time
import sys
import config
from audio_notifier import AudioNotifier
from logger import HabitLogger
from detectors.landmark_detector import LandmarkDetector
from detectors.yolo_detector import YoloPhoneDetector
from detectors.habit_rules import HabitRulesEngine
from ui_overlay import UIOverlay

def main():
    print("============================================")
    print("      BAD HABIT DETECTOR - STARTING         ")
    print("============================================")
    print(f"Log file: {config.LOG_FILE_PATH}")
    print("Press 'd' to toggle debug overlay.")
    print("Press 'q' to quit application.")
    print("--------------------------------------------")

    # 1. Initialize Webcam (AVFoundation backend preferred on macOS)
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("Warning: Could not open camera device 0 with AVFoundation. Trying default backend...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Trying camera device 1...")
            cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                print("\n" + "="*60)
                print(" ERROR: CAMERA PERMISSION REQUIRED ON MACOS")
                print("="*60)
                print("macOS blocked camera access for your terminal application.")
                print("\nTo grant permission:")
                print("1. Open System Settings -> Privacy & Security -> Camera")
                print("2. Toggle ON permission for Terminal / iTerm2 / VS Code")
                print("3. Restart your terminal window and run `uv run main.py` again!")
                print("="*60 + "\n")
                sys.exit(1)


    # Set camera resolution (optional, default 1280x720 if supported)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 2. Initialize Detectors & Components
    print("Loading MediaPipe Holistic landmarks model...")
    landmark_detector = LandmarkDetector()

    print("Loading YOLOv8n object detector for cell phone tracking...")
    yolo_detector = YoloPhoneDetector()

    audio = AudioNotifier()
    logger = HabitLogger()
    rules_engine = HabitRulesEngine(audio, logger)
    ui = UIOverlay()

    debug_mode = config.DEFAULT_DEBUG_MODE
    frame_count = 0
    start_time = time.time()
    fps = 0.0

    # Phone detection state memory across check intervals
    phone_detected = False
    phone_conf = 0.0
    phone_box = None

    print("Initialization complete! Starting video stream loop...")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to capture frame from webcam.")
                break

            # Mirror frame horizontally for intuitive webcam view
            frame = cv2.flip(frame, 1)
            frame_count += 1

            # FPS calculation
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                if elapsed > 0:
                    fps = 10.0 / elapsed
                start_time = time.time()

            # 3. MediaPipe Landmark Extraction
            lm_data = landmark_detector.process(frame)

            # Draw raw MediaPipe landmarks if debug mode is active
            if debug_mode:
                landmark_detector.draw_landmarks(frame, lm_data["results"])

            # 4. YOLO Cell Phone Detection (Interval Skipped)
            if frame_count % config.PHONE_CHECK_INTERVAL == 0:
                phone_detected, phone_conf, phone_box = yolo_detector.detect(frame)

            # 5. Evaluate Rules & Trigger Alerts
            rules_engine.evaluate_scratching(lm_data["head_bbox"], lm_data["hand_points"])
            rules_engine.evaluate_biting(lm_data["mouth_center"], lm_data["hand_points"], lm_data["mouth_gap"])
            rules_engine.evaluate_phone(phone_detected, phone_conf, phone_box, lm_data["hand_points"], frame.shape)

            # 6. Render HUD Overlay
            summary = rules_engine.get_status_summary()
            frame = ui.render(
                frame=frame,
                summary=summary,
                fps=fps,
                debug_mode=debug_mode,
                head_top=lm_data["head_top"],
                head_bbox=lm_data["head_bbox"],
                mouth_center=lm_data["mouth_center"],
                mouth_gap=lm_data["mouth_gap"],
                scratch_jitter=rules_engine.current_jitter
            )

            # 7. Display Frame & Listen to Keyboard Input
            cv2.imshow("Bad Habit Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\nQuit requested by user.")
                break
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"Debug overlay toggled: {'ON' if debug_mode else 'OFF'}")

    finally:
        print("\nCleaning up resources...")
        cap.release()
        landmark_detector.close()
        cv2.destroyAllWindows()

        # Session Summary Report
        summary = rules_engine.get_status_summary()
        print("\n============================================")
        print("          SESSION SUMMARY REPORT            ")
        print("============================================")
        print(f"Hair Scratching occurrences: {summary['total_scratch']}")
        print(f"Finger/Nail Biting occurrences: {summary['total_biting']}")
        print(f"Phone Use occurrences:         {summary['total_phone']}")
        print(f"Log saved to: {config.LOG_FILE_PATH}")
        print("============================================\n")

if __name__ == "__main__":
    main()
