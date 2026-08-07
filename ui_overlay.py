"""
UI Overlay Renderer Module.
Draws a modern HUD banner, real-time habit status badges, lifetime counters,
FPS overlay, phone bounding boxes, and debug guides on OpenCV video frames.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import config

class UIOverlay:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        # Colors (BGR)
        self.COLOR_BG = (20, 20, 25)
        self.COLOR_TEXT_WHITE = (255, 255, 255)
        self.COLOR_TEXT_MUTED = (180, 180, 180)
        self.COLOR_OK = (40, 200, 80)          # Green
        self.COLOR_ALERT = (40, 40, 240)       # Red
        self.COLOR_PHONE = (240, 160, 40)      # Amber/Cyan
        self.COLOR_DEBUG_GUIDE = (255, 255, 0) # Cyan/Yellow

    def render(
        self,
        frame: cv2.Mat,
        summary: Dict[str, Any],
        fps: float,
        debug_mode: bool,
        head_top: Optional[Tuple[float, float]] = None,
        mouth_center: Optional[Tuple[float, float]] = None
    ) -> cv2.Mat:
        """
        Renders the complete HUD overlay onto frame.
        """
        h, w, _ = frame.shape

        # 1. Semi-transparent top header bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 85), self.COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Title & FPS
        cv2.putText(frame, "BAD HABIT DETECTOR", (15, 30), self.font, 0.75, self.COLOR_TEXT_WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 30), self.font, 0.55, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        # 2. Habit Badges (Scratching, Biting, Phone)
        card_w = (w - 40) // 3
        card_y = 42

        cards = [
            ("SCRATCHING", summary["scratch_active"], summary["total_scratch"], summary["scratch_counter"], config.SCRATCH_CONSECUTIVE_FRAMES),
            ("NAIL BITING", summary["biting_active"], summary["total_biting"], summary["biting_counter"], config.BITING_CONSECUTIVE_FRAMES),
            ("PHONE USE", summary["phone_active"], summary["total_phone"], summary["phone_counter"], config.PHONE_CONSECUTIVE_CHECKS)
        ]

        for i, (name, is_active, total, current_counter, max_counter) in enumerate(cards):
            card_x = 15 + i * (card_w + 5)
            badge_color = self.COLOR_ALERT if is_active else self.COLOR_OK
            status_text = "ALERT!" if is_active else "CLEAR"

            # Badge background
            cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + 35), (40, 40, 45), -1)
            cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + 35), badge_color, 1)

            # Badge text
            cv2.putText(frame, f"{name}: ", (card_x + 8, card_y + 22), self.font, 0.45, self.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
            cv2.putText(frame, status_text, (card_x + 95, card_y + 22), self.font, 0.45, badge_color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"[{total}]", (card_x + card_w - 30, card_y + 22), self.font, 0.45, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        # 3. Phone Bounding Box
        phone_box = summary.get("phone_box")
        if phone_box:
            x1, y1, x2, y2 = phone_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_PHONE, 3)
            label = "CELL PHONE"
            cv2.putText(frame, label, (x1, max(20, y1 - 10)), self.font, 0.6, self.COLOR_PHONE, 2, cv2.LINE_AA)

        # 4. Debug Guides (If Debug Mode is Enabled)
        if debug_mode:
            # Hair threshold line
            if head_top:
                hx_px, hy_px = int(head_top[0] * w), int(head_top[1] * h)
                thresh_y_px = int((head_top[1] - config.SCRATCH_HEAD_MARGIN) * h)
                cv2.line(frame, (0, thresh_y_px), (w, thresh_y_px), self.COLOR_DEBUG_GUIDE, 1, cv2.LINE_AA)
                cv2.putText(frame, "Scratch Threshold Line", (10, max(15, thresh_y_px - 5)), self.font, 0.4, self.COLOR_DEBUG_GUIDE, 1, cv2.LINE_AA)

            # Mouth threshold circle
            if mouth_center:
                mx_px, my_px = int(mouth_center[0] * w), int(mouth_center[1] * h)
                radius_px = int(config.BITING_MOUTH_THRESH * min(w, h))
                cv2.circle(frame, (mx_px, my_px), radius_px, self.COLOR_DEBUG_GUIDE, 1, cv2.LINE_AA)
                cv2.putText(frame, "Mouth Zone", (mx_px - 35, my_px), self.font, 0.4, self.COLOR_DEBUG_GUIDE, 1, cv2.LINE_AA)

        # 5. Bottom Shortcut Legend
        cv2.rectangle(frame, (0, h - 30), (w, h), self.COLOR_BG, -1)
        legend = f"Shortcuts:  [D] Debug Overlay: {'ON' if debug_mode else 'OFF'}  |  [Q] Quit"
        cv2.putText(frame, legend, (15, h - 10), self.font, 0.45, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        return frame
