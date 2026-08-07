"""
MediaPipe Holistic Landmark Detector Module.
Extracts face mesh landmarks (head top, mouth center) and left/right hand landmarks.
"""

import cv2
import mediapipe as mp
from typing import Dict, Any, List, Optional, Tuple

class LandmarkDetector:
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, frame_bgr: cv2.Mat) -> Dict[str, Any]:
        """
        Processes a BGR image frame and extracts normalized landmark coordinates.
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(frame_rgb)

        head_top: Optional[Tuple[float, float]] = None
        mouth_center: Optional[Tuple[float, float]] = None
        lower_lip: Optional[Tuple[float, float]] = None
        mouth_gap: float = 0.0
        # Head bounding box as (x_min, y_min, x_max, y_max) in normalized coords
        head_bbox: Optional[Tuple[float, float, float, float]] = None
        hand_points: List[Tuple[float, float]] = []

        # Extract Face Landmarks
        # Key indices:
        #   10  = Top of forehead / head
        #   152 = Bottom of chin
        #   234 = Right temple (right side of face)
        #   454 = Left temple (left side of face)
        #   13  = Inner upper lip
        #   14  = Inner lower lip
        if results.face_landmarks:
            landmarks = results.face_landmarks.landmark
            if len(landmarks) > 454:
                head_top = (landmarks[10].x, landmarks[10].y)
                mouth_center = (landmarks[13].x, landmarks[13].y)
                lower_lip = (landmarks[14].x, landmarks[14].y)

                # Build head bounding box from forehead, chin, and temples
                top_y = landmarks[10].y
                bottom_y = landmarks[152].y
                left_x = landmarks[454].x   # Left temple (viewer's right due to mirror)
                right_x = landmarks[234].x  # Right temple (viewer's left due to mirror)
                # Ensure correct min/max regardless of mirror flip
                x_min = min(left_x, right_x)
                x_max = max(left_x, right_x)
                head_bbox = (x_min, top_y, x_max, bottom_y)

                # Calculate vertical mouth opening gap
                import math
                mouth_gap = math.hypot(landmarks[13].x - landmarks[14].x, landmarks[13].y - landmarks[14].y)


        # Extract Left Hand Landmarks
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                hand_points.append((lm.x, lm.y))

        # Extract Right Hand Landmarks
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                hand_points.append((lm.x, lm.y))

        return {
            "results": results,
            "head_top": head_top,
            "head_bbox": head_bbox,
            "mouth_center": mouth_center,
            "mouth_gap": mouth_gap,
            "hand_points": hand_points,
            "has_face": head_top is not None,
            "has_hands": len(hand_points) > 0
        }

    def draw_landmarks(self, frame_bgr: cv2.Mat, results: Any):
        """
        Draws MediaPipe facial and hand landmark wireframes on the frame for debugging.
        """
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                frame_bgr,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )

        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame_bgr,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame_bgr,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

    def close(self):
        self.holistic.close()
