import math
import time
from collections import deque
from typing import Dict, Any, Optional, List, Tuple
import config
from audio_notifier import AudioNotifier
from logger import HabitLogger

class HabitRulesEngine:
    def __init__(self, audio_notifier: AudioNotifier, logger: HabitLogger):
        self.audio = audio_notifier
        self.logger = logger

        # Consecutive frame counters
        self.scratch_counter = 0
        self.biting_counter = 0
        self.phone_counter = 0

        # Total event occurrence counts
        self.total_scratch_count = 0
        self.total_biting_count = 0
        self.total_phone_count = 0

        # Last alert timestamps (for cooldown management)
        self.last_scratch_alert = 0.0
        self.last_biting_alert = 0.0
        self.last_phone_alert = 0.0

        # Active state flags (True while habit is actively detected)
        self.scratch_active = False
        self.biting_active = False
        self.phone_active = False

        # Latest phone bounding box
        self.last_phone_box: Optional[List[int]] = None

        # --- Scratch motion tracking ---
        # Stores recent hand centroid positions (normalized) for jitter calculation
        self.hand_history: deque = deque(maxlen=config.SCRATCH_JITTER_WINDOW)
        self.current_jitter: float = 0.0

    def _get_hand_centroid_near_head(
        self, hand_points: List[Tuple[float, float]], head_bbox: Tuple[float, float, float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        Returns the centroid (avg x, avg y) of hand landmarks that fall within
        the expanded head zone, or None if no hand points are near the head.
        """
        margin = config.SCRATCH_HEAD_ZONE_MARGIN
        x_min, y_min, x_max, y_max = head_bbox
        # Expand the box: widen horizontally and extend upward (for hair above forehead)
        zone_x_min = x_min - margin
        zone_x_max = x_max + margin
        zone_y_min = y_min - margin * 2  # Extra upward margin for hair above head
        zone_y_max = y_min + (y_max - y_min) * 0.3  # Only upper 1/3 of face (forehead/hair, NOT cheeks)

        near_points = []
        for hx, hy in hand_points:
            if zone_x_min <= hx <= zone_x_max and zone_y_min <= hy <= zone_y_max:
                near_points.append((hx, hy))

        if not near_points:
            return None

        avg_x = sum(p[0] for p in near_points) / len(near_points)
        avg_y = sum(p[1] for p in near_points) / len(near_points)
        return (avg_x, avg_y)

    def _compute_jitter(self) -> float:
        """
        Computes average frame-to-frame Euclidean displacement of the hand centroid
        over the recent history window. High jitter = scratching motion.
        """
        if len(self.hand_history) < 2:
            return 0.0
        total_disp = 0.0
        count = 0
        for i in range(1, len(self.hand_history)):
            prev = self.hand_history[i - 1]
            curr = self.hand_history[i]
            if prev is not None and curr is not None:
                total_disp += math.hypot(curr[0] - prev[0], curr[1] - prev[1])
                count += 1
        return total_disp / count if count > 0 else 0.0

    def evaluate_scratching(
        self, head_bbox: Optional[Tuple[float, float, float, float]], hand_points: List[tuple]
    ) -> bool:
        """
        Hair Scratching Rule:
        1. Hand must be within the expanded head bounding box (proximity check).
        2. Hand must be jittering (rapid small movements = scratching motion).
        Both conditions must hold for N consecutive frames to trigger an alert.
        """
        centroid = None
        if head_bbox and hand_points:
            centroid = self._get_hand_centroid_near_head(hand_points, head_bbox)

        # Track hand position history for jitter
        self.hand_history.append(centroid)
        self.current_jitter = self._compute_jitter()

        is_near_head = centroid is not None
        is_jittering = self.current_jitter >= config.SCRATCH_JITTER_THRESH
        is_scratching_this_frame = is_near_head and is_jittering

        if is_scratching_this_frame:
            self.scratch_counter += 1
        else:
            self.scratch_counter = max(0, self.scratch_counter - 1)

        self.scratch_active = self.scratch_counter >= config.SCRATCH_CONSECUTIVE_FRAMES

        # Check alert trigger & cooldown
        now = time.time()
        if self.scratch_active and (now - self.last_scratch_alert >= config.ALERT_COOLDOWN_SEC):
            self.last_scratch_alert = now
            self.total_scratch_count += 1
            self.audio.play_beep(config.SCRATCH_BEEP_FREQ, config.BEEP_DURATION)
            self.logger.log_habit("Hair Scratching", f"Jitter: {self.current_jitter:.4f}")
            return True

        return False

    def evaluate_biting(self, mouth_center: Optional[tuple], hand_points: List[tuple], mouth_gap: float = 0.0) -> bool:
        """
        Finger/Nail Biting Rule: Any hand landmark stays within MOUTH_DIST_THRESH
        (normalized distance) of the mouth center for N consecutive frames AND
        (if REQUIRE_MOUTH_OPEN_FOR_BITING) mouth_gap >= MOUTH_OPEN_THRESH.
        """
        is_biting_this_frame = False
        is_mouth_open = (mouth_gap >= config.MOUTH_OPEN_THRESH) if config.REQUIRE_MOUTH_OPEN_FOR_BITING else True

        if mouth_center and hand_points and is_mouth_open:
            mx, my = mouth_center

            for hx, hy in hand_points:
                dist = math.hypot(hx - mx, hy - my)
                if dist <= config.BITING_MOUTH_THRESH:
                    is_biting_this_frame = True
                    break


        if is_biting_this_frame:
            self.biting_counter += 1
        else:
            self.biting_counter = max(0, self.biting_counter - 1)

        self.biting_active = self.biting_counter >= config.BITING_CONSECUTIVE_FRAMES

        # Check alert trigger & cooldown
        now = time.time()
        if self.biting_active and (now - self.last_biting_alert >= config.ALERT_COOLDOWN_SEC):
            self.last_biting_alert = now
            self.total_biting_count += 1
            self.audio.play_beep(config.BITING_BEEP_FREQ, config.BEEP_DURATION)
            self.logger.log_habit("Finger/Nail Biting", f"Counter: {self.biting_counter}")
            return True

        return False

    def evaluate_phone(self, phone_detected: bool, conf: float, box: Optional[List[int]], hand_points: List[tuple] = [], frame_shape: tuple = (720, 1280)) -> bool:
        """
        Phone Use Rule: YOLO detect cell phone held for N consecutive check intervals.
        Optionally requires hand landmarks to overlap with the phone bounding box.
        """
        is_held_by_hand = True
        if phone_detected and box and config.PHONE_REQUIRE_HAND_HOLDING:
            is_held_by_hand = False
            h, w = frame_shape[:2] if len(frame_shape) >= 2 else (720, 1280)
            x1, y1, x2, y2 = box
            margin = 30  # Margin around phone box in pixels

            for hx, hy in hand_points:
                px, py = int(hx * w), int(hy * h)
                if (x1 - margin) <= px <= (x2 + margin) and (y1 - margin) <= py <= (y2 + margin):
                    is_held_by_hand = True
                    break

        valid_phone_detection = phone_detected and is_held_by_hand

        if valid_phone_detection:
            self.phone_counter += 1
            self.last_phone_box = box
        else:
            self.phone_counter = max(0, self.phone_counter - 1)
            if self.phone_counter == 0:
                self.last_phone_box = None

        self.phone_active = self.phone_counter >= config.PHONE_CONSECUTIVE_CHECKS

        # Check alert trigger & cooldown
        now = time.time()
        if self.phone_active and (now - self.last_phone_alert >= config.ALERT_COOLDOWN_SEC):
            self.last_phone_alert = now
            self.total_phone_count += 1
            self.audio.play_beep(config.PHONE_BEEP_FREQ, config.BEEP_DURATION)
            self.logger.log_habit("Phone Use", f"Confidence: {conf:.2f}")
            return True

        return False

    def get_status_summary(self) -> Dict[str, Any]:
        return {
            "scratch_active": self.scratch_active,
            "biting_active": self.biting_active,
            "phone_active": self.phone_active,
            "total_scratch": self.total_scratch_count,
            "total_biting": self.total_biting_count,
            "total_phone": self.total_phone_count,
            "scratch_counter": self.scratch_counter,
            "biting_counter": self.biting_counter,
            "phone_counter": self.phone_counter,
            "phone_box": self.last_phone_box
        }

