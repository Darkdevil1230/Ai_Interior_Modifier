"""
cv_detector.py
--------------
Room element detection using YOLOv8 (pretrained fallback).

Robustness added: detect and ignore text/placeholders at weights/best.pt
so the app falls back to 'yolov8n.pt' instead of crashing with UTF-8 decode errors.
"""
import os
from typing import List, Dict, Tuple
import numpy as np
from ultralytics import YOLO


# Expanded COCO -> domain mapping (abbreviated)
COCO_TO_DOMAIN = {
    "chair": "chair", "sofa": "sofa", "couch": "sofa", "dining table": "table",
    "table": "table", "desk": "table", "bed": "bed", "window": "window", "door": "door"
}

DEFAULT_CLASS_MAP = {0: "wall", 1: "door", 2: "window", 3: "bed", 4: "table", 5: "chair"}


def _is_probably_binary_weights(path: str, min_size_bytes: int = 1024) -> bool:
    """
    Heuristic check to decide whether a file is likely a binary weights file:
     - file must exist
     - file size should be >= min_size_bytes (tiny text placeholders are suspect)
     - file should contain at least one null byte or non-UTF8-like byte pattern
    This is defensive and avoids attempting to load plain-text placeholders.
    """
    try:
        if not os.path.isfile(path):
            return False
        size = os.path.getsize(path)
        if size < min_size_bytes:
            return False
        with open(path, "rb") as f:
            head = f.read(4096)
        # If contains a null byte, it's almost certainly binary
        if b"\x00" in head:
            return True
        # Count bytes outside common UTF-8 printable range; many such bytes => binary
        non_printable = sum(1 for b in head if b < 0x09 or (0x0D < b < 0x20) or b > 0x7F)
        if non_printable > len(head) * 0.05:  # >5% suspicious bytes
            return True
        # Otherwise treat as text/placeholder
        return False
    except Exception:
        return False


class RoomDetector:
    def __init__(self, model_path: str = "weights/best.pt"):
        """
        Try to load custom weights first; if they appear to be a text placeholder
        or clearly not a binary checkpoint, skip and fall back to 'yolov8n.pt'.
        """
        self.model_path = model_path
        self.model = None
        self.names = None
        self.model_name = None

        use_custom = _is_probably_binary_weights(model_path)
        if use_custom:
            try:
                self.model = YOLO(model_path)
                self.names = getattr(self.model, "names", None)
                self.model_name = model_path
                return
            except Exception:
                # If loading fails, fall back below
                pass

        # Fallback to official small pretrained model
        self.model = YOLO("yolov8n.pt")
        self.names = getattr(self.model, "names", None)
        self.model_name = "yolov8n.pt"

    def get_model_name(self) -> str:
        return self.model_name

    def detect(self, image_path: str, conf_threshold: float = 0.35) -> List[Dict]:
        results = self.model(image_path)
        detections = []
        for r in results:
            for box in getattr(r, "boxes", []):
                try:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                except Exception:
                    continue
                label = None
                if self.names is not None:
                    if isinstance(self.names, dict):
                        label = self.names.get(cls, None)
                    elif isinstance(self.names, (list, tuple)) and cls < len(self.names):
                        label = self.names[cls]
                detections.append({
                    "class": cls,
                    "label": label,
                    "confidence": conf,
                    "bbox": xyxy
                })
        # filter by confidence threshold
        detections = [d for d in detections if d["confidence"] >= conf_threshold]
        return detections

    def _map_label_to_domain(self, label: str, class_idx: int = None) -> str:
        if not label and class_idx is not None:
            return DEFAULT_CLASS_MAP.get(class_idx, "unknown")
        if label:
            l = label.lower()
            return COCO_TO_DOMAIN.get(l, l)
        return "unknown"

    def parse_detections(self, detections: List[Dict], class_map: Dict[int, str] = None,
                         room_dims: Tuple[float, float] = (400.0, 300.0),
                         image_shape: Tuple[int, int] = (480, 640)) -> List[Dict]:
        img_h, img_w = image_shape
        room_w, room_h = room_dims
        objects = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)
            obj_w = max(1.0, (x2 - x1) / float(img_w) * float(room_w))
            obj_h = max(1.0, (y2 - y1) / float(img_h) * float(room_h))
            obj_x = max(0.0, x1 / float(img_w) * float(room_w))
            obj_y = max(0.0, y1 / float(img_h) * float(room_h))

            model_label = det.get("label")
            mapped = self._map_label_to_domain(model_label, det.get("class"))
            if mapped == "unknown" and class_map is not None:
                mapped = class_map.get(det.get("class"), "unknown")

            obj = {
                "type": mapped,
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "x": obj_x,
                "y": obj_y,
                "w": obj_w,
                "h": obj_h
            }
            objects.append(obj)
        return objects
