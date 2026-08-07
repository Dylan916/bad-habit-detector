"""
Habit Detection Rules Engine.
Evaluates frame landmark data and YOLO results against configurable thresholds,
managing state counters, alert cooldowns, audio playback, and CSV logging.
"""

import math
import time
from typing import Dict, Any, Optional, List
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

    def evaluate_scratching(self, head_top: Optional[tuple], hand_points: List[tuple]) -> bool:
        """
        Hair Scratching Rule: Any hand landmark's y-coordinate stays above
        (head_top_y - SCRATCH_HEAD_MARGIN) for N consecutive frames.
        """
        is_scratching_this_frame = False
        if head_top and hand_points:
            head_y = head_top[1]
            threshold_y = head_y - config.SCRATCH_HEAD_MARGIN

            for _, hy in hand_points:
                if hy < threshold_y:
                    is_scratching_this_frame = True
                    break

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
            self.logger.log_habit("Hair Scratching", f"Counter: {self.scratch_counter}")
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

    def evaluate_phone(self, phone_detected: bool, conf: float, box: Optional[List[int]]) -> bool:
        """
        Phone Use Rule: YOLO detect cell phone held for N consecutive check intervals.
        """
        if phone_detected:
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

