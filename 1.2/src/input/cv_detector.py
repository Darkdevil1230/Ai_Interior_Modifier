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
from src.input.architectural_detector import enhance_detections_with_architecture


# Comprehensive COCO -> domain mapping for furniture and objects
COCO_TO_DOMAIN = {
    # Furniture
    "chair": "chair", "dining chair": "chair", "office chair": "chair",
    "sofa": "sofa", "couch": "sofa", "sectional": "sofa",
    "dining table": "table", "table": "table", "desk": "table", "office desk": "table",
    "bed": "bed", "double bed": "bed", "single bed": "bed", "king bed": "bed", "queen bed": "bed",
    "wardrobe": "wardrobe", "closet": "wardrobe", "armoire": "wardrobe",
    "bookshelf": "bookshelf", "shelf": "bookshelf", "bookcase": "bookshelf",
    "cabinet": "cabinet", "storage cabinet": "cabinet", "kitchen cabinet": "cabinet",
    "dresser": "dresser", "chest of drawers": "dresser", "bureau": "dresser",
    "nightstand": "nightstand", "bedside table": "nightstand",
    "coffee table": "coffee table", "side table": "side table", "end table": "side table",
    "ottoman": "ottoman", "footstool": "ottoman", "pouf": "ottoman",
    "bench": "bench", "storage bench": "bench",
    "stool": "stool", "bar stool": "stool", "kitchen stool": "stool",
    "tv stand": "tv stand", "tv cabinet": "tv stand", "entertainment center": "tv stand",
    "mirror": "mirror", "floor mirror": "mirror", "wall mirror": "mirror",
    
    # Electronics & Appliances
    "tv": "tv", "television": "tv", "monitor": "tv",
    "laptop": "laptop", "computer": "laptop",
    "phone": "phone", "mobile phone": "phone", "cell phone": "phone",
    "keyboard": "keyboard", "computer keyboard": "keyboard",
    "mouse": "mouse", "computer mouse": "mouse",
    "remote": "remote", "tv remote": "remote",
    "speaker": "speaker", "bluetooth speaker": "speaker",
    "lamp": "lamp", "table lamp": "lamp", "floor lamp": "lamp", "desk lamp": "lamp",
    "clock": "clock", "wall clock": "clock", "alarm clock": "clock",
    
    # Storage & Organization
    "backpack": "backpack", "school bag": "backpack",
    "handbag": "handbag", "purse": "handbag", "tote bag": "handbag",
    "suitcase": "suitcase", "luggage": "suitcase",
    "basket": "basket", "storage basket": "basket", "laundry basket": "basket",
    "box": "box", "storage box": "box", "cardboard box": "box",
    "drawer": "drawer", "storage drawer": "drawer",
    
    # Books & Media
    "book": "book", "books": "book", "novel": "book", "textbook": "book",
    "magazine": "magazine", "newspaper": "magazine",
    "notebook": "notebook", "journal": "notebook",
    "pen": "pen", "pencil": "pen", "marker": "pen",
    
    # Decorative Items
    "vase": "vase", "flower vase": "vase",
    "picture": "picture", "photo": "picture", "painting": "picture", "artwork": "picture",
    "frame": "frame", "picture frame": "frame",
    "candle": "candle", "candlestick": "candle",
    "plant": "plant", "potted plant": "plant", "houseplant": "plant",
    "flower": "flower", "bouquet": "flower",
    "sculpture": "sculpture", "statue": "sculpture",
    "ornament": "ornament", "decoration": "ornament",
    
    # Kitchen Items
    "cup": "cup", "mug": "cup", "coffee cup": "cup",
    "bowl": "bowl", "soup bowl": "bowl", "cereal bowl": "bowl",
    "plate": "plate", "dinner plate": "plate",
    "bottle": "bottle", "water bottle": "bottle", "wine bottle": "bottle",
    "glass": "glass", "wine glass": "glass", "drinking glass": "glass",
    "spoon": "spoon", "fork": "spoon", "knife": "spoon", "cutlery": "spoon",
    "pot": "pot", "cooking pot": "pot", "saucepan": "pot",
    "pan": "pan", "frying pan": "pan", "skillet": "pan",
    
    # Clothing & Personal Items
    "shirt": "shirt", "t-shirt": "shirt", "blouse": "shirt",
    "pants": "pants", "jeans": "pants", "trousers": "pants",
    "dress": "dress", "gown": "dress",
    "shoes": "shoes", "sneakers": "shoes", "boots": "shoes", "sandals": "shoes",
    "hat": "hat", "cap": "hat", "baseball cap": "hat",
    "jacket": "jacket", "coat": "jacket", "blazer": "jacket",
    "tie": "tie", "necktie": "tie",
    "belt": "belt", "leather belt": "belt",
    "watch": "watch", "wristwatch": "watch",
    "glasses": "glasses", "eyeglasses": "glasses", "sunglasses": "glasses",
    
    # Toys & Entertainment
    "toy": "toy", "doll": "toy", "teddy bear": "toy",
    "ball": "ball", "sports ball": "ball", "basketball": "ball", "football": "ball",
    "game": "game", "board game": "game", "puzzle": "game",
    "guitar": "guitar", "piano": "guitar", "violin": "guitar",
    
    # Sports & Fitness
    "bicycle": "bicycle", "bike": "bicycle",
    "skateboard": "skateboard", "skate": "skateboard",
    "tennis racket": "tennis racket", "racket": "tennis racket",
    "baseball bat": "baseball bat", "bat": "baseball bat",
    "frisbee": "frisbee", "disc": "frisbee",
    
    # Tools & Equipment
    "scissors": "scissors", "shears": "scissors",
    "hammer": "hammer", "mallet": "hammer",
    "screwdriver": "screwdriver", "tool": "screwdriver",
    "wrench": "wrench", "spanner": "wrench",
    
    # Room Features
    "window": "window", "glass window": "window",
    "door": "door", "entrance": "door",
    "wall": "wall", "partition": "wall",
    "pillar": "pillar", "column": "pillar", "support column": "pillar",
    "shelf": "shelf", "wall shelf": "shelf", "bookshelf shelf": "shelf",
    "floor": "floor", "carpet": "floor", "rug": "floor",
    "ceiling": "ceiling", "roof": "ceiling",
    
    # Miscellaneous
    "umbrella": "umbrella", "parasol": "umbrella",
    "towel": "towel", "bath towel": "towel", "hand towel": "towel",
    "pillow": "pillow", "cushion": "pillow",
    "blanket": "blanket", "throw": "blanket", "quilt": "blanket",
    "curtain": "curtain", "drape": "curtain", "blind": "curtain",
    "fan": "fan", "ceiling fan": "fan", "desk fan": "fan",
    "heater": "heater", "radiator": "heater",
    "air conditioner": "air conditioner", "ac": "air conditioner",
    
    # Fallback for unknown objects
    "object": "object", "item": "object", "thing": "object"
}

# Enhanced default class map for custom models
DEFAULT_CLASS_MAP = {
    0: "wall", 1: "door", 2: "window", 3: "bed", 4: "table", 5: "chair",
    6: "sofa", 7: "tv", 8: "wardrobe", 9: "lamp", 10: "desk",
    11: "bookshelf", 12: "cabinet", 13: "dresser", 14: "nightstand",
    15: "coffee table", 16: "ottoman", 17: "bench", 18: "stool",
    19: "tv stand", 20: "mirror", 21: "plant", 22: "picture",
    23: "vase", 24: "clock", 25: "backpack", 26: "handbag",
    27: "book", 28: "laptop", 29: "phone", 30: "cup",
    31: "bottle", 32: "bowl", 33: "plate", 34: "glass",
    35: "shirt", 36: "pants", 37: "shoes", 38: "hat",
    39: "toy", 40: "ball", 41: "scissors", 42: "umbrella",
    43: "towel", 44: "pillow", 45: "blanket", 46: "curtain",
    47: "fan", 48: "heater", 49: "object"
}


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


# Categories for better object organization
ARCHITECTURAL_ELEMENTS = {"window", "door", "wall", "floor", "ceiling", "pillar", "shelf"}
LARGE_FURNITURE = {"bed", "sofa", "wardrobe", "dresser", "bookshelf", "cabinet"}
SMALL_FURNITURE = {"chair", "table", "desk", "nightstand", "tv stand", "lamp", "plant"}


class RoomDetector:
    def __init__(self, model_path: str = "weights/best.pt", names_override: dict = None):
        """
        Try to load custom weights first; if they appear to be a text placeholder
        or clearly not a binary checkpoint, skip and fall back to 'yolov8n.pt'.
        """
        self.model_path = model_path
        self.model = None
        self.names = None
        self.model_name = None
        self.is_custom_model = False

        use_custom = _is_probably_binary_weights(model_path)
        if use_custom:
            try:
                print(f"[RoomDetector] Loading custom model: {model_path}")
                self.model = YOLO(model_path)
                self.names = getattr(self.model, "names", None)
                self.model_name = model_path
                self.is_custom_model = True
                print(f"[RoomDetector] ✓ Custom model loaded with classes: {self.names}")
                return
            except Exception as e:
                print(f"[RoomDetector] ✗ Custom model failed to load: {e}")
                # If loading fails, fall back below
                pass

        # Fallback to official small pretrained model
        print(f"[RoomDetector] Falling back to pretrained model: yolov8n.pt")
        self.model = YOLO("yolov8n.pt")
        self.names = getattr(self.model, "names", None)
        self.model_name = "yolov8n.pt"
        self.is_custom_model = False
        print(f"[RoomDetector] ⚠ Using COCO model - may not detect architectural elements well")

        # Auto-load names override JSON if present next to weights or in weights/ directory
        if not names_override:
            try:
                import json, os
                candidates = []
                if isinstance(self.model_path, str):
                    base = os.path.dirname(self.model_path)
                    candidates.extend([
                        os.path.join(base, "classes.json"),
                        os.path.join(base, "class_map.json"),
                        os.path.join(base, "names.json"),
                    ])
                candidates.extend([
                    os.path.join("weights", "classes.json"),
                    os.path.join("weights", "class_map.json"),
                    os.path.join("weights", "names.json"),
                ])
                for p in candidates:
                    if os.path.exists(p):
                        with open(p, "r") as f:
                            self.names = json.load(f)
                            print(f"[RoomDetector] Loaded class map: {p}")
                            break
            except Exception:
                pass
        else:
            self.names = names_override

    def get_model_name(self) -> str:
        return self.model_name
    
    def is_architectural_element(self, obj_type: str) -> bool:
        """Check if an object is a fixed architectural element."""
        return obj_type.lower() in ARCHITECTURAL_ELEMENTS
    
    def categorize_object(self, obj_type: str) -> str:
        """Categorize object for better layout handling."""
        obj_lower = obj_type.lower()
        if obj_lower in ARCHITECTURAL_ELEMENTS:
            return "architectural"
        elif obj_lower in LARGE_FURNITURE:
            return "large_furniture"
        elif obj_lower in SMALL_FURNITURE:
            return "small_furniture"
        else:
            return "other"

    def detect(self, image_path: str, conf_threshold: float = 0.35) -> List[Dict]:
        results = self.model(image_path)
        detections = []
        architectural_count = 0
        furniture_count = 0
        
        for r in results:
            masks_xy = []
            try:
                # ultralytics returns r.masks.xy as list of Nx2 arrays (float)
                if hasattr(r, "masks") and getattr(r.masks, "xy", None) is not None:
                    masks_xy = r.masks.xy
            except Exception:
                masks_xy = []
            for idx, box in enumerate(getattr(r, "boxes", [])):
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
                
                # Track detection categories
                if label and self.is_architectural_element(label):
                    architectural_count += 1
                elif label:
                    furniture_count += 1
                    
                det = {
                    "class": cls,
                    "label": label,
                    "confidence": conf,
                    "bbox": xyxy,
                    "category": self.categorize_object(label) if label else "other"
                }
                # Attach mask polygon if present
                try:
                    if masks_xy and idx < len(masks_xy) and masks_xy[idx] is not None:
                        poly = masks_xy[idx]
                        if poly is not None:
                            det["mask"] = [(float(x), float(y)) for x, y in poly]
                except Exception:
                    pass
                detections.append(det)
        
        # Confidence filtering with architectural grace
        arch_threshold = max(0.05, conf_threshold * 0.5)
        filtered_detections = []
        for d in detections:
            is_arch = d.get("category") == "architectural" or str(d.get("label") or "").lower() in ARCHITECTURAL_ELEMENTS
            thr = arch_threshold if is_arch else conf_threshold
            if float(d.get("confidence", 0.0)) >= thr:
                d["type"] = d.get("label")  # normalize for downstream
                filtered_detections.append(d)
        
        # Print detection summary
        print(f"[Detection] Found {len(filtered_detections)} objects total:")
        print(f"  - Architectural elements: {architectural_count}")
        print(f"  - Furniture items: {furniture_count}")
        
        # FALLBACK: If using COCO model, enhance with image processing for architectural elements
        if not self.is_custom_model:
            print("[Detection] Using COCO model - applying architectural detection fallback")
            detections = enhance_detections_with_architecture(
                yolo_detections=detections,
                image_path=image_path,
                force_architectural=False  # Only add if none found
            )
        
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
            # CRITICAL: Some detectors (architectural fallback) provide 'type' not 'label'.
            # Prefer explicit 'type' if present to keep windows/doors/walls.
            if isinstance(det.get("type"), str) and det.get("type"): 
                mapped = det.get("type").lower()

            obj = {
                "type": mapped,
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "x": obj_x,
                "y": obj_y,
                "w": obj_w,
                "h": obj_h
            }
            # Preserve category if supplied (e.g., 'architectural')
            if det.get("category"):
                obj["category"] = det.get("category")
            objects.append(obj)
        return objects
