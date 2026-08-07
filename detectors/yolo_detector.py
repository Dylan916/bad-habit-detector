"""
YOLOv8 Object Detector Module for Cell Phone Detection.
Uses Ultralytics pretrained YOLOv8n to detect cell phones in webcam frames.
"""

import cv2
from typing import Tuple, List, Optional
from ultralytics import YOLO
import config

class YoloPhoneDetector:
    def __init__(self, model_name: str = "yolov8s.pt"):
        # Load pretrained YOLOv8 model (auto-downloads on first run if missing)
        # yolov8s = "small" variant, 11.2M params — better accuracy at distance than nano (3.2M)
        self.model = YOLO(model_name)
        self.cell_phone_class_ids = config.PHONE_COCO_CLASS_IDS

    def detect(self, frame_bgr: cv2.Mat, conf_threshold: float = config.PHONE_CONF_THRESH) -> Tuple[bool, float, Optional[List[int]]]:
        """
        Runs YOLOv8 detection on frame_bgr.
        Returns:
            - phone_detected (bool)
            - max_confidence (float)
            - bounding_box [x1, y1, x2, y2] or None
        """
        results = self.model(frame_bgr, verbose=False, conf=conf_threshold, imgsz=480)
        
        phone_detected = False
        max_conf = 0.0
        best_box = None

        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls_id in self.cell_phone_class_ids:
                    if conf > max_conf:
                        phone_detected = True
                        max_conf = conf
                        xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                        best_box = xyxy

        return phone_detected, max_conf, best_box
