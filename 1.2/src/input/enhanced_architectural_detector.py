"""
Enhanced Architectural Detector
Combines depth estimation (MiDaS), vision-language analysis (optional Florence-2/LLaVA),
and classical CV to detect walls, windows, doors, and lighting.

This module is designed to run with graceful degradation:
- If transformers models are unavailable, falls back to CV heuristics only.
- If CUDA is unavailable, runs on CPU.
"""
from typing import Dict, List, Tuple, Optional
import os
import re
import json

import numpy as np
import cv2

try:
    import torch
except Exception:
    torch = None  # type: ignore

try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    _has_transformers = True
except Exception:
    _has_transformers = False

class EnhancedArchitecturalDetector:
    def __init__(self, device: Optional[str] = None, use_vlm: bool = True):
        self.device = device or ("cuda" if (torch and torch.cuda.is_available()) else "cpu")
        self.use_vlm = use_vlm and _has_transformers
        self.vlm_processor = None
        self.vlm_model = None
        self.depth_model = None
        self._init_models()

    def _init_models(self) -> None:
        # Depth: try MiDaS via torch.hub if torch available
        if torch is not None:
            try:
                self.depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
                self.depth_model.eval()
                if self.device == "cuda":
                    self.depth_model.to(self.device)
            except Exception:
                self.depth_model = None
        # VLM: optional Florence-2 (requires trust_remote_code)
        if self.use_vlm:
            try:
                self.vlm_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
                dtype = None
                if torch is not None and self.device == "cuda":
                    dtype = torch.float16
                self.vlm_model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/Florence-2-large",
                    trust_remote_code=True,
                    torch_dtype=dtype,
                )
                if torch is not None:
                    self.vlm_model.to(self.device)
            except Exception:
                self.vlm_processor = None
                self.vlm_model = None
                self.use_vlm = False

    def detect(self, image_path: str) -> Dict:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image at {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        depth = self._get_depth(img_rgb)
        vlm = self._analyze_with_vlm(img_rgb) if self.use_vlm else {}

        walls = self._detect_walls(img_rgb, depth, vlm)
        windows = self._detect_windows(img_rgb, depth, vlm)
        doors = self._detect_doors(img_rgb, depth, vlm)
        lighting = self._analyze_lighting(img_rgb, vlm)

        # Build unified detections list (image pixel coordinates)
        detected = []
        for src_key, t in (("windows", "window"), ("doors", "door")):
            for obj in (vlm.get(src_key) or []):
                try:
                    if all(k in obj for k in ("x", "y", "width", "height")):
                        x = float(obj["x"]); y = float(obj["y"])
                        w = float(obj["width"]); h = float(obj["height"])
                        cx = x + w / 2.0
                        cy = y + h / 2.0
                        detected.append({
                            "type": t,
                            "bbox": [x, y, x + w, y + h],
                            "center": [cx, cy],
                            "confidence": float(obj.get("confidence", 0.8)),
                        })
                except Exception:
                    continue

        # Depth summary for downstream use
        depth_summary = None
        if depth is not None:
            try:
                dmin = float(depth.min())
                dmax = float(depth.max())
                dmean = float(depth.mean())
                depth_summary = {"min": dmin, "max": dmax, "mean": dmean}
            except Exception:
                depth_summary = None

        return {
            "room_type": vlm.get("room_type", "unknown"),
            "walls": walls,
            "windows": windows,
            "doors": doors,
            "lighting": lighting,
            # Unified detection schema for downstream pipeline
            "detected": detected,
            "depth_map": depth,
            "depth_map_summary": depth_summary,
            "vlm_description": vlm,
        }

    def _get_depth(self, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        if self.depth_model is None or torch is None:
            return None
        try:
            inp = cv2.resize(img_rgb, (384, 384))
            inp = inp.astype(np.float32) / 255.0
            tensor = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0)
            if self.device == "cuda":
                tensor = tensor.to(self.device)
            with torch.no_grad():
                pred = self.depth_model(tensor)
                if pred.ndim == 4:
                    pred = pred.squeeze(0)
                if pred.ndim == 3:
                    pred = pred.mean(0)
                depth = pred.detach().cpu().numpy()
            depth = cv2.resize(depth, (img_rgb.shape[1], img_rgb.shape[0]))
            # normalize
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            return depth
        except Exception:
            return None

    def _analyze_with_vlm(self, img_rgb: np.ndarray) -> Dict:
        if not (self.vlm_processor and self.vlm_model and _has_transformers):
            return {}
        try:
            from PIL import Image
            image = Image.fromarray(img_rgb)
            prompt = (
                "<DETAILED_CAPTION> Provide JSON with keys: room_type, walls, windows, doors, "
                "lighting (natural_light_sources, artificial_lights, brightness_level, color_temperature), colors."
            )
            inputs = self.vlm_processor(images=image, text=prompt, return_tensors="pt")
            if torch is not None and self.device == "cuda":
                inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            with torch.no_grad():
                output = self.vlm_model.generate(**inputs, max_new_tokens=512, do_sample=False)
            text = self.vlm_processor.decode(output[0], skip_special_tokens=True)
            return self._parse_vlm_json(text)
        except Exception:
            return {}

    def _parse_vlm_json(self, text: str) -> Dict:
        try:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return {}
            return json.loads(m.group(0))
        except Exception:
            return {}

    def _detect_walls(self, img_rgb: np.ndarray, depth: Optional[np.ndarray], vlm: Dict) -> List[Dict]:
        walls: List[Dict] = []
        if isinstance(vlm.get("walls"), list):
            walls.extend(vlm["walls"])  # assume items are dicts with start/end or boxes
        # Depth/CV based straight line detection as fallback
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=120, maxLineGap=8)
        if lines is not None:
            for ln in lines.squeeze(axis=1) if len(lines.shape) == 3 else lines:
                x1, y1, x2, y2 = map(float, ln[:4])
                walls.append({"type": "wall", "start": [x1, y1], "end": [x2, y2], "source": "cv"})
        return walls

    def _detect_windows(self, img_rgb: np.ndarray, depth: Optional[np.ndarray], vlm: Dict) -> List[Dict]:
        wins: List[Dict] = []
        if isinstance(vlm.get("windows"), list):
            wins.extend(vlm["windows"])  # expect dicts {x,y,width,height} in image coords or normalized
        # Simple brightness/edge heuristic near walls could be added later
        return wins

    def _detect_doors(self, img_rgb: np.ndarray, depth: Optional[np.ndarray], vlm: Dict) -> List[Dict]:
        drs: List[Dict] = []
        if isinstance(vlm.get("doors"), list):
            drs.extend(vlm["doors"])  # expect dicts {x,y,width,height}
        return drs

    def _analyze_lighting(self, img_rgb: np.ndarray, vlm: Dict) -> Dict:
        lighting = {
            "natural_light_sources": [],
            "artificial_lights": [],
            "brightness_level": "unknown",
            "color_temperature": "unknown",
        }
        if isinstance(vlm.get("lighting"), dict):
            lighting.update({k: v for k, v in vlm["lighting"].items() if k in lighting})
        return lighting
