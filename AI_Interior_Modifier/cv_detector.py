
"""
cv_detector.py
--------------
Room element detection using YOLOv8.
Detects walls, doors, windows, and furniture in a room image.
"""

import cv2
import numpy as np
from ultralytics import YOLO

class RoomDetector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Initialize the YOLOv8 model.
        :param model_path: Path to the YOLOv8 model weights.
        """
        self.model = YOLO(model_path)

    def detect(self, image_path):
        """
        Detect room elements in the image.
        :param image_path: Path to the room image.
        :return: List of detected objects with class, confidence, and bounding box.
        """
        results = self.model(image_path)
        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                detections.append({
                    "class": cls,
                    "confidence": conf,
                    "bbox": xyxy
                })
        return detections

    def parse_detections(self, detections, class_map):
        """
        Convert YOLO detections to structured room objects.
        :param detections: List of YOLO detections.
        :param class_map: Dict mapping class indices to names.
        :return: List of room objects with type and bounding box.
        """
        objects = []
        for det in detections:
            obj = {
                "type": class_map.get(det["class"], "unknown"),
                "bbox": det["bbox"]
            }
            objects.append(obj)
        return objects