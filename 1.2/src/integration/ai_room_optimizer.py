"""
ai_room_optimizer.py
--------------------
Main integration module that combines CNN architectural detection,
genetic algorithm optimization, and enhanced 2D layout generation.
"""

import os
import tempfile
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image
import io

# Import our custom modules
from src.input.cnn_architectural_detector import CNNArchitecturalDetector
from src.input.cv_detector import RoomDetector
from src.input.enhanced_detector import EnhancedDetector
try:
    from src.input.enhanced_architectural_detector import EnhancedArchitecturalDetector
except Exception:
    EnhancedArchitecturalDetector = None  # type: ignore
from src.visualization.enhanced_layout_generator import EnhancedLayoutGenerator
from src.optimization.cnn_guided_optimizer import CNNGuidedOptimizer
from pipeline import run_pipeline, LLMUnavailableError

class AIRoomOptimizer:
    """
    Main AI Room Optimizer that integrates CNN detection, genetic algorithms,
    and enhanced visualization for optimal furniture placement.
    """
    
    def __init__(self, room_dims: Tuple[float, float], device: str = "cpu", room_type: Optional[str] = None,
                 known_length_cm: Optional[float] = None, known_length_type: Optional[str] = None,
                 show_detections_overlay: bool = False,
                 model_path: Optional[str] = None,
                 class_map: Optional[Dict] = None,
                 conf_threshold: float = 0.35,
                 manual_corners: Optional[List[Tuple[float, float]]] = None,
                 manual_detections: Optional[List[Dict]] = None,
                 detector_backend: str = "cnn"):
        self.room_dims = room_dims
        self.device = device
        self.room_type = room_type
        self.known_length_cm = known_length_cm
        self.known_length_type = known_length_type or "door_width"
        self.scale_cm_per_px: Optional[float] = None
        self.show_detections_overlay: bool = bool(show_detections_overlay)
        self.model_path = model_path or "weights/best.pt"
        self.class_map = class_map or None
        self.conf_threshold = float(conf_threshold)
        self.manual_corners = manual_corners or []
        self.manual_detections = manual_detections or []
        
        # Initialize components
        self.detector_backend = str(detector_backend or "cnn").lower().strip()
        if self.detector_backend == "enhanced" and EnhancedArchitecturalDetector is not None:
            try:
                self.architectural_detector = EnhancedArchitecturalDetector(device=device)
            except Exception:
                print("[AI Room Optimizer] Falling back to CNNArchitecturalDetector (failed to init EnhancedArchitecturalDetector)")
                self.detector_backend = "cnn"
                self.architectural_detector = CNNArchitecturalDetector(device=device)
        else:
            if self.detector_backend == "enhanced" and EnhancedArchitecturalDetector is None:
                print("[AI Room Optimizer] EnhancedArchitecturalDetector not available. Using CNN backend.")
            self.detector_backend = "cnn"
            self.architectural_detector = CNNArchitecturalDetector(device=device)
        self.layout_generator = EnhancedLayoutGenerator(room_dims)
        # Propagate room type to layout generator for title rendering
        if isinstance(room_type, str):
            self.layout_generator.room_type = room_type
        
        # Results storage
        self.room_analysis = None
        self.optimized_layout = None
        self.architectural_elements = []
        self.furniture_layout = []
        self.optimizer_diagnostics: List[Dict] = []
        self.last_method: str = "none"  # 'llm_constraint_solver' or 'genetic_algorithm_fallback'
    
    def analyze_room(self, image_path: str) -> Dict:
        """
        Analyze room using CNN to detect architectural elements and understand layout.
        """
        print("[AI Room Optimizer] Analyzing room...")
        if self.detector_backend == "enhanced" and hasattr(self.architectural_detector, "detect"):
            # Use the new enhanced detector and normalize to the existing schema
            try:
                enhanced = self.architectural_detector.detect(image_path)
            except Exception as e:
                print(f"[AI Room Optimizer] Enhanced detect failed ({e}). Falling back to CNN pipeline.")
                self.detector_backend = "cnn"
                enhanced = None
            if isinstance(enhanced, dict):
                self.room_analysis = self._normalize_enhanced_analysis(enhanced, image_path)
            else:
                # Fallback to CNN pipeline
                self.architectural_elements = self.architectural_detector.detect_architectural_elements(image_path)
                self.room_analysis = self.architectural_detector.analyze_room_layout(image_path)
        else:
            # Original CNN pipeline
            # Detect architectural elements
            self.architectural_elements = self.architectural_detector.detect_architectural_elements(image_path)
            # Perform comprehensive room analysis
            self.room_analysis = self.architectural_detector.analyze_room_layout(image_path)

        # Phase 1.2: detect other objects (furniture, plants, décor) and merge
        try:
            rd = RoomDetector(model_path=self.model_path, names_override=self.class_map or None)
            yolo_dets = rd.detect(image_path, conf_threshold=self.conf_threshold)
            # Also run enhanced multi-pass heuristics (edges for windows/paintings)
            ed = EnhancedDetector(base_detector=rd)
            multi = ed.multi_pass_detection(image_path)
            multi_dets = multi.get("detections", [])

            # Merge with existing detections
            base = list(self.room_analysis.get("detections", []))
            # Merge manual detections (high confidence, preferred) if any
            manual_norm = []
            for m in (self.manual_detections or []):
                if not isinstance(m, dict):
                    continue
                t = str(m.get("type") or m.get("label") or "").lower()
                bb = m.get("bbox")
                if isinstance(bb, (list, tuple)) and len(bb) == 4 and t:
                    mm = {"type": t, "label": t, "bbox": list(bb), "confidence": float(m.get("confidence", 0.99)), "category": "architectural"}
                    manual_norm.append(mm)
            combined = self._dedup_merge(manual_norm + base + yolo_dets + multi_dets, iou_threshold=0.5)
            # Persist combined detections back into analysis for downstream consumers
            self.room_analysis["detections"] = combined
            # Try to estimate homography before projection
            try:
                H = None
                # If manual corners provided, use them directly
                if isinstance(self.manual_corners, list) and len(self.manual_corners) == 4:
                    import numpy as np, cv2
                    pts = np.array(self.manual_corners, dtype=np.float32)
                    # order (tl, tr, br, bl) if not guaranteed
                    s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
                    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
                    tr = pts[np.argmin(diff)]; bl = pts[np.argmax(diff)]
                    src = np.array([tl, tr, br, bl], dtype=np.float32)
                    dst = np.array([[0,0],[self.room_dims[0],0],[self.room_dims[0],self.room_dims[1]],[0,self.room_dims[1]]], dtype=np.float32)
                    H = cv2.getPerspectiveTransform(src, dst)
                else:
                    H = self._estimate_homography(image_path)
                if H is not None:
                    self.room_analysis.setdefault("metrics", {})["homography"] = H.tolist()
            except Exception:
                H = None

            # Project to room coordinates (cm) using homography if available, else scale fallback
            self._project_detections_to_room(image_path)
        except Exception as e:
            print(f"[AI Room Optimizer] Non-fatal: general object detection merge failed: {e}")

        # Estimate cm-per-pixel scale if user provided a known length and a matching detection exists
        try:
            if isinstance(self.known_length_cm, (int, float)) and self.known_length_cm > 0:
                doors = [d for d in self.room_analysis.get("doors", [])] or [d for d in self.room_analysis.get("detections", []) if d.get("type") == "door"]
                if doors:
                    # Approximate door width as the smaller of bbox width/height in pixels
                    dx1, dy1, dx2, dy2 = doors[0]["bbox"]
                    px_w = abs(dx2 - dx1)
                    px_h = abs(dy2 - dy1)
                    door_px = min(px_w, px_h) if min(px_w, px_h) > 0 else None
                    if door_px and door_px > 0:
                        self.scale_cm_per_px = float(self.known_length_cm) / float(door_px)
                        # Attach to analysis for downstream consumers
                        self.room_analysis.setdefault("metrics", {})
                        self.room_analysis["metrics"]["scale_cm_per_px"] = self.scale_cm_per_px
        except Exception:
            pass
        
        # After merging and projection, rebuild architectural elements from room coords
        try:
            # Ensure projection is run once merged detections are available
            self._project_detections_to_room(image_path)
        except Exception:
            pass
        # Snap openings to perimeter and dedupe
        try:
            self._snap_and_infer_perimeter_openings()
        except Exception:
            pass
        # If still no windows/doors, try edge-gap inference from the image
        try:
            if not any((str(d.get("type") or d.get("label") or "").lower() in {"window","door"}) for d in self.room_analysis.get("detections", [])):
                self._infer_openings_from_image(image_path)
                # re-snap after inference
                self._snap_and_infer_perimeter_openings()
        except Exception:
            pass
        self.architectural_elements = self._elements_from_detections(self.room_analysis.get("detections", []))
        print(f"[AI Room Optimizer] Detected {len(self.architectural_elements)} architectural elements")
        print(f"[AI Room Optimizer] Room analysis complete")
        
        return self.room_analysis

    def _normalize_enhanced_analysis(self, enhanced: Dict, image_path: str) -> Dict:
        """Convert EnhancedArchitecturalDetector outputs to the internal room_analysis format.
        - Build detections list with bbox in image px.
        - Add simple recommendations skeleton to keep downstream consumers happy.
        """
        dets: List[Dict] = []
        # Windows/doors as rects
        for key in ("windows", "doors"):
            for obj in enhanced.get(key, []) or []:
                try:
                    if all(k in obj for k in ("x", "y", "width", "height")):
                        x = float(obj["x"]); y = float(obj["y"]) 
                        w = float(obj["width"]); h = float(obj["height"]) 
                        dets.append({
                            "type": key[:-1],
                            "label": key[:-1],
                            "bbox": [x, y, x + w, y + h],
                            "confidence": float(obj.get("confidence", 0.8)),
                            "category": "architectural",
                        })
                except Exception:
                    continue
        # Walls as line segments -> approximate thin rectangles, but filter to perimeter-long segments only
        wall_rects: List[Dict] = []
        for wall in enhanced.get("walls", []) or []:
            try:
                sx, sy = wall.get("start", [0, 0])
                ex, ey = wall.get("end", [0, 0])
                sx, sy, ex, ey = map(float, [sx, sy, ex, ey])
                x1, y1, x2, y2 = min(sx, ex), min(sy, ey), max(sx, ex), max(sy, ey)
                # add small thickness
                if abs(x2 - x1) < abs(y2 - y1):
                    # vertical wall
                    x2 = x1 + 6.0
                else:
                    # horizontal wall
                    y2 = y1 + 6.0
                wall_rects.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(wall.get("confidence", 0.7)),
                })
            except Exception:
                continue
        # Keep at most 4 perimeter-aligned walls (top/bottom/left/right)
        try:
            import cv2
            img = cv2.imread(image_path)
            ih, iw = img.shape[:2] if img is not None else (1000, 1000)
            margin = max(10.0, 0.03 * min(iw, ih))
            def side_of(bb):
                x1,y1,x2,y2 = bb
                cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
                # distance to borders
                d_left, d_right = cx - 0.0, iw - cx
                d_top, d_bottom = cy - 0.0, ih - cy
                dmin = min(d_left, d_right, d_top, d_bottom)
                if dmin == d_left:
                    return "left", d_left
                if dmin == d_right:
                    return "right", d_right
                if dmin == d_top:
                    return "top", d_top
                return "bottom", d_bottom
            buckets = {"left": [], "right": [], "top": [], "bottom": []}
            for wr in wall_rects:
                bb = wr["bbox"]
                side, dist = side_of(bb)
                if dist > margin:
                    continue  # not near perimeter
                w = abs(bb[2]-bb[0]); h = abs(bb[3]-bb[1])
                long_side = max(w, h)
                # require long wall (>60% of corresponding room image side)
                need = 0.6 * (ih if side in ("left","right") else iw)
                if long_side < need:
                    continue
                buckets[side].append(wr)
            # pick longest per side
            for side, items in buckets.items():
                if not items:
                    continue
                items.sort(key=lambda r: max(abs(r["bbox"][2]-r["bbox"][0]), abs(r["bbox"][3]-r["bbox"][1])), reverse=True)
                keep = items[0]
                dets.append({
                    "type": "wall",
                    "label": "wall",
                    "bbox": keep["bbox"],
                    "confidence": keep.get("confidence", 0.7),
                    "category": "architectural",
                })
        except Exception:
            # If filtering fails, keep none (outer frame suffices)
            pass
        # Lighting summary
        lighting = enhanced.get("lighting", {}) or {}
        analysis: Dict = {
            "room_type": enhanced.get("room_type") or self.room_type,
            "detections": dets,
            "recommendations": {
                "lighting_considerations": {
                    "natural_light": lighting.get("brightness_level", "unknown"),
                    "color_temperature": lighting.get("color_temperature", "unknown"),
                },
                "furniture_placement_zones": [],
                "traffic_flow": {"pathways": [], "clearance_required": 60},
            },
            "metrics": {}
        }
        return analysis

    def _elements_from_detections(self, detections: List[Dict]) -> List[Dict]:
        """Create simple architectural_elements rectangles for visualization from detections in room coords if available,
        otherwise image px; visualization uses room_dims scale anyway after projection step.
        """
        elems: List[Dict] = []
        room_w, room_h = self.room_dims
        # thresholds in cm
        min_wall_len = max(30.0, min(room_w, room_h) * 0.08)   # at least 8% of short side or 30cm
        min_wall_thk = 3.0
        def _area(bb):
            x1,y1,x2,y2 = bb; return max(0.0, x2-x1) * max(0.0, y2-y1)
        def _iou(a, b):
            ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
            ix1,iy1 = max(ax1,bx1), max(ay1,by1)
            ix2,iy2 = min(ax2,bx2), min(ay2,by2)
            if ix2<=ix1 or iy2<=iy1: return 0.0
            inter = (ix2-ix1)*(iy2-iy1)
            uni = _area(a)+_area(b)-inter
            return inter/uni if uni>0 else 0.0
        for d in detections:
            t = str(d.get("type") or d.get("label") or "").lower()
            # include core architectural items and fixed decor that must be respected
            if t not in {"window", "door", "wall", "plant", "potted plant", "vase"}:
                continue
            bb = d.get("room_bbox") or d.get("bbox")
            if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                continue
            x1, y1, x2, y2 = map(float, bb)
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if t == "wall":
                # treat as elongated rectangle representing a wall segment
                long_side = max(w, h)
                short_side = min(w, h)
                if long_side < min_wall_len or short_side < min_wall_thk:
                    continue
            # plants/vase are modeled as fixed decor to avoid overlap with movable furniture
            is_decor = t in {"plant", "potted plant", "vase"}
            elems.append({
                "type": ("plant" if t in {"potted plant", "vase"} else t),
                "x": x1,
                "y": y1,
                "w": w,
                "h": h,
                "fixed": True,
                "architectural": (not is_decor),
            })
        # Deduplicate walls by IoU and keep top-N by area
        walls = [e for e in elems if e["type"] == "wall"]
        others = [e for e in elems if e["type"] != "wall"]
        if walls:
            walls = sorted(walls, key=lambda e: _area((e["x"], e["y"], e["x"]+e["w"], e["y"]+e["h"])) , reverse=True)
            dedup: List[Dict] = []
            for wci in walls:
                bb = (wci["x"], wci["y"], wci["x"] + wci["w"], wci["y"] + wci["h"])
                if all(_iou(bb, (wcj["x"], wcj["y"], wcj["x"]+wcj["w"], wcj["y"]+wcj["h"])) < 0.6 for wcj in dedup):
                    dedup.append(wci)
                if len(dedup) >= 8:
                    break
            walls = dedup
        return others + walls

    def _project_detections_to_room(self, image_path: str) -> None:
        """Project detection bboxes (and masks if present) into room coordinates (cm).
        Uses homography if available, else anisotropic scaling from image size to room dims.
        """
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                return
            ih, iw = img.shape[:2]
            sx = float(self.room_dims[0]) / max(1.0, float(iw))
            sy = float(self.room_dims[1]) / max(1.0, float(ih))
            # Homography (image -> room cm coordinates)
            H = None
            try:
                H_list = self.room_analysis.get("metrics", {}).get("homography")
                if isinstance(H_list, list):
                    import numpy as np
                    H = np.array(H_list, dtype=float)
            except Exception:
                H = None
            dets = self.room_analysis.get("detections", [])
            for d in dets:
                bb = d.get("bbox")
                if H is not None:
                    # Project bbox corners through H
                    import numpy as np
                    pts = np.array([[[bb[0], bb[1]]], [[bb[2], bb[1]]], [[bb[2], bb[3]]], [[bb[0], bb[3]]]], dtype=np.float32)
                    wpts = cv2.perspectiveTransform(pts, H)
                    xs = [float(p[0][0]) for p in wpts]
                    ys = [float(p[0][1]) for p in wpts]
                    d["room_bbox"] = [min(xs), min(ys), max(xs), max(ys)]
                    if isinstance(d.get("mask"), (list, tuple)) and len(d["mask"]) >= 3:
                        poly = np.array(d["mask"], dtype=np.float32).reshape((-1,1,2))
                        wpoly = cv2.perspectiveTransform(poly, H)
                        d["room_mask"] = [(float(p[0][0]), float(p[0][1])) for p in wpoly]
                else:
                    if isinstance(bb, (list, tuple)) and len(bb) == 4:
                        x1, y1, x2, y2 = bb
                        rb = [float(x1) * sx, float(y1) * sy, float(x2) * sx, float(y2) * sy]
                        d["room_bbox"] = rb
                    if isinstance(d.get("mask"), (list, tuple)) and len(d["mask"]) >= 3:
                        room_poly = [(float(x) * sx, float(y) * sy) for (x, y) in d["mask"]]
                        d["room_mask"] = room_poly
        except Exception:
            pass

    def _estimate_homography(self, image_path: str):
        """Estimate homography mapping image (floor/wall rectangle) to room rectangle in cm.
        Approach: Canny -> find largest contour -> approx to 4-point polygon -> order points -> get H.
        Returns 3x3 matrix H (float32) mapping image px to room cm.
        """
        try:
            import cv2
            import numpy as np
            img = cv2.imread(image_path)
            if img is None:
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5,5), 0)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            # Largest contour by area
            cnt = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) != 4:
                return None
            pts = approx.reshape(4, 2).astype(np.float32)
            # Order points (top-left, top-right, bottom-right, bottom-left)
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            tl = pts[np.argmin(s)]
            br = pts[np.argmax(s)]
            tr = pts[np.argmin(diff)]
            bl = pts[np.argmax(diff)]
            src = np.array([tl, tr, br, bl], dtype=np.float32)
            dst = np.array([[0,0], [self.room_dims[0], 0], [self.room_dims[0], self.room_dims[1]], [0, self.room_dims[1]]], dtype=np.float32)
            H = cv2.getPerspectiveTransform(src, dst)
            return H
        except Exception:
            return None

    def _dedup_merge(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Simple IoU-based deduplication keeping higher-confidence items and preferring architectural types."""
        def iou(b1, b2):
            x1_1, y1_1, x2_1, y2_1 = b1
            x1_2, y1_2, x2_2, y2_2 = b2
            xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
            xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
            if xi2 <= xi1 or yi2 <= yi1:
                return 0.0
            inter = (xi2 - xi1) * (yi2 - yi1)
            a1 = max(0, x2_1 - x1_1) * max(0, y2_1 - y1_1)
            a2 = max(0, x2_2 - x1_2) * max(0, y2_2 - y1_2)
            uni = a1 + a2 - inter
            return inter / uni if uni > 0 else 0.0

        # Sort by preference: architectural first, then confidence desc
        def is_arch(d):
            t = str(d.get("type") or d.get("label") or "").lower()
            return t in {"window", "door", "wall", "floor", "ceiling"}
        dets = sorted(detections, key=lambda d: (not is_arch(d), -(d.get("confidence", 0.0) or 0.0)))
        kept: List[Dict] = []
        for d in dets:
            bb = d.get("bbox")
            if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                continue
            drop = False
            for k in kept:
                if iou(bb, k.get("bbox", [0,0,0,0])) > iou_threshold:
                    drop = True
                    break
            if not drop:
                kept.append(d)
        return kept
    
    def validate_furniture_selection(self, furniture_list: List[Dict]) -> List[Dict]:
        """
        Validate furniture selection to ensure items fit in the room.
        """
        room_area = self.room_dims[0] * self.room_dims[1]
        validated_furniture = []
        total_furniture_area = 0
        
        print(f"[AI Room Optimizer] Validating furniture for {self.room_dims[0]}x{self.room_dims[1]}cm room ({room_area:.0f}cm²)")
        
        for furniture in furniture_list:
            furniture_area = furniture.get("w", 0) * furniture.get("h", 0)
            furniture_name = furniture.get("name", "Unknown")
            
            # Check if individual furniture is too large
            if furniture_area > room_area * 0.6:  # More than 60% of room
                print(f"[WARNING] Skipping {furniture_name} - too large ({furniture_area:.0f}cm²)")
                continue
            
            # Check if adding this furniture would exceed room capacity
            if total_furniture_area + furniture_area > room_area * 0.8:  # More than 80% total
                print(f"[WARNING] Skipping {furniture_name} - room would be overcrowded")
                continue
            
            validated_furniture.append(furniture)
            total_furniture_area += furniture_area
            print(f"[OK] Added {furniture_name} ({furniture_area:.0f}cm²)")
        
        print(f"[AI Room Optimizer] Validated {len(validated_furniture)}/{len(furniture_list)} furniture items")
        print(f"[AI Room Optimizer] Total furniture area: {total_furniture_area:.0f}cm² ({total_furniture_area/room_area*100:.1f}% of room)")
        
        return validated_furniture
    
    def optimize_furniture_placement(self, furniture_list: List[Dict], 
                                    user_preferences: Dict = None) -> List[Dict]:
        """Optimize furniture placement using LLM architect + constraint solver pipeline.

        Falls back to the GA-based CNNGuidedOptimizer if the LLM is unavailable or
        if unexpected errors occur in the pipeline.
        """
        if not self.room_analysis:
            raise ValueError("Room analysis must be performed first. Call analyze_room() before optimization.")

        import logging

        logger = logging.getLogger(__name__)

        # Validate furniture selection first
        validated_furniture = self.validate_furniture_selection(furniture_list)

        if not validated_furniture:
            raise ValueError("No valid furniture items selected for this room size.")

        logger.info("[AI Room Optimizer] Starting LLM+CP-SAT optimization pipeline...")

        # Prepare inputs for the pipeline
        room_data = dict(self.room_analysis or {})
        detected_openings: List[Dict] = room_data.get("detections", [])
        furniture_catalog = pd.DataFrame(validated_furniture)

        combined_layout: List[Dict]

        # TRY: LLM Architect + CP-SAT Pipeline
        try:
            logger.info("🚀 Attempting LLM + Constraint Solver pipeline...")
            layout = run_pipeline(
                room_data=room_data,
                room_dims=self.room_dims,
                detected_openings=detected_openings,
                furniture_catalog=furniture_catalog,
                room_type=self.room_type or "living_room",
                api_key=None,
            )
            combined_layout = layout
            logger.info("✅ LLM pipeline succeeded")
            self.optimizer_diagnostics = []
            self.last_method = "llm_constraint_solver"
        except LLMUnavailableError as e:
            # Expected fallback case
            logger.warning("⚠️ LLM unavailable, falling back to GA: %s", e)
        except Exception as e:  # noqa: BLE001
            # Unexpected error - log but don't hide it
            logger.exception("❌ Unexpected error in LLM pipeline: %s", e)
            logger.warning("⚠️ Falling back to GA optimizer due to unexpected error")

        # FALLBACK: Genetic Algorithm (if combined_layout not set)
        if "combined_layout" not in locals():
            logger.info("🧬 Using GA optimizer fallback...")
            import time
            import hashlib

            room_signature = f"{self.room_dims[0]}_{self.room_dims[1]}_{len(validated_furniture)}_{time.time()}"
            seed_hash = int(hashlib.md5(room_signature.encode()).hexdigest()[:8], 16)
            logger.info("[AI Room Optimizer] Using GA fallback with seed: %d", seed_hash)

            optimizer = CNNGuidedOptimizer(
                room_dims=self.room_dims,
                objects=validated_furniture,
                room_analysis=self.room_analysis,
                user_prefs=user_preferences,
                population_size=200,
                generations=300,
                seed=seed_hash,
            )
            combined_layout = optimizer.optimize()
            try:
                self.optimizer_diagnostics = getattr(optimizer, "diagnostics", []) or []
            except Exception:
                self.optimizer_diagnostics = []
            self.last_method = "genetic_algorithm_fallback"
        
        # Separate furniture from architectural elements in the optimized layout
        self.furniture_layout = []
        detected_arch_elements = []
        
        for item in combined_layout:
            if item.get("architectural", False) or item.get("fixed", False):
                detected_arch_elements.append(item)
            else:
                self.furniture_layout.append(item)
        
        # Update architectural elements with newly detected ones (from fallback detector)
        if detected_arch_elements:
            print(f"[AI Room Optimizer] Found {len(detected_arch_elements)} architectural elements in optimization")
            self.architectural_elements.extend(detected_arch_elements)

        # Ensure no furniture overlaps with architectural elements (hard constraint repair)
        try:
            self.furniture_layout = self._repair_arch_overlaps(self.furniture_layout)
        except Exception:
            pass
        
        # Optional: compute and store metrics for the final layout
        try:
            metrics = self._calculate_metrics(self.furniture_layout, self.room_dims, self.room_analysis.get("detections", []))
            setattr(self, "last_metrics", metrics)
        except Exception:
            pass

        print(f"[AI Room Optimizer] Optimization complete. Generated layout with {len(self.furniture_layout)} furniture items + {len(self.architectural_elements)} architectural elements")
        
        return self.furniture_layout
    
    def generate_2d_layout(self, save_path: Optional[str] = None, 
                          save_buffer: Optional[io.BytesIO] = None) -> io.BytesIO:
        """
        Generate enhanced 2D layout showing architectural elements and optimized furniture.
        """
        if not self.architectural_elements and not self.furniture_layout:
            raise ValueError("No layout data available. Run analysis and optimization first.")
        
        print("[AI Room Optimizer] Generating enhanced 2D layout...")
        
        # Ensure room type is reflected in generator
        if isinstance(self.room_type, str):
            self.layout_generator.room_type = self.room_type

        # Generate layout
        buffer = self.layout_generator.generate_layout(
            architectural_elements=self.architectural_elements,
            furniture_layout=self.furniture_layout,
            save_path=save_path,
            save_buffer=save_buffer,
            detections=None,  # keep the final output clean
            overlay_detections=False,
            diagnostics=self.optimizer_diagnostics,
            overlay_violations=True
        )
        
        print("[AI Room Optimizer] 2D layout generation complete")
        
        return buffer

    def _repair_arch_overlaps(self, layout: List[Dict]) -> List[Dict]:
        """Shift furniture slightly to avoid overlap with windows/doors/walls.
        Keeps inside room bounds, tries minimal translation.
        """
        if not layout:
            return layout
        room_w, room_h = self.room_dims
        arch_bboxes = []
        for d in self.room_analysis.get("detections", []):
            t = str(d.get("type") or d.get("label") or "").lower()
            if t in {"window", "door", "wall", "plant", "potted plant", "vase"}:
                bb = d.get("room_bbox") or d.get("bbox")
                if isinstance(bb, (list, tuple)) and len(bb) == 4:
                    arch_bboxes.append(tuple(map(float, bb)))

        def rect_overlap(a, b) -> float:
            ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            return (ix2 - ix1) * (iy2 - iy1)

        repaired = []
        for obj in layout:
            x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
            moved = False
            for _ in range(50):  # limited attempts
                ob = (x, y, x + w, y + h)
                overlaps = [rect_overlap(ob, bb) for bb in arch_bboxes]
                if not overlaps or max(overlaps) <= 0:
                    break
                # push away along the smallest move among four directions
                step = 5.0
                candidates = [
                    (x + step, y),
                    (x - step, y),
                    (x, y + step),
                    (x, y - step),
                ]
                best = (x, y); best_overlap = float("inf")
                for cx, cy in candidates:
                    cx = max(0.0, min(room_w - w, cx))
                    cy = max(0.0, min(room_h - h, cy))
                    cob = (cx, cy, cx + w, cy + h)
                    ov = sum(rect_overlap(cob, bb) for bb in arch_bboxes)
                    if ov < best_overlap:
                        best, best_overlap = (cx, cy), ov
                if best_overlap < sum(overlaps):
                    x, y = best
                    moved = True
                else:
                    break
            new_obj = dict(obj)
            new_obj["x"], new_obj["y"] = float(x), float(y)
            repaired.append(new_obj)
        return repaired

    def _snap_and_infer_perimeter_openings(self) -> None:
        """Snap window/door detections to nearest room edge in room coordinates and keep at most
        one segment per side per type. If a detection is within margin from an edge, clamp it to that edge
        and trim its thickness to a narrow band.
        """
        dets = list(self.room_analysis.get("detections", []))
        if not dets:
            return
        room_w, room_h = self.room_dims
        margin = max(8.0, 0.03 * min(room_w, room_h))
        thickness = 6.0
        buckets = {("window","left"):[], ("window","right"):[], ("window","top"):[], ("window","bottom"):[],
                   ("door","left"):[], ("door","right"):[], ("door","top"):[], ("door","bottom"):[]}
        out: List[Dict] = []
        def side_of(bb):
            x1,y1,x2,y2 = bb
            cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
            d_left, d_right = cx - 0.0, room_w - cx
            d_top, d_bottom = cy - 0.0, room_h - cy
            dmin = min(d_left, d_right, d_top, d_bottom)
            if dmin == d_left:
                return "left", d_left
            if dmin == d_right:
                return "right", d_right
            if dmin == d_top:
                return "top", d_top
            return "bottom", d_bottom
        for d in dets:
            t = str(d.get("type") or d.get("label") or "").lower()
            if t not in {"window","door"}:
                out.append(d); continue
            bb = d.get("room_bbox") or d.get("bbox")
            if not isinstance(bb, (list,tuple)) or len(bb) != 4:
                continue
            x1,y1,x2,y2 = map(float, bb)
            side, dist = side_of([x1,y1,x2,y2])
            if dist > margin:
                # Not close enough to perimeter: drop to avoid floating openings
                continue
            # Snap to the side and clamp thickness
            if side in ("left","right"):
                y1c, y2c = max(0.0, min(y1,y2)), min(room_h, max(y1,y2))
                if y2c - y1c < 20.0:  # too small -> drop
                    continue
                if side == "left":
                    x1c, x2c = 0.0, min(thickness, room_w)
                else:
                    x1c, x2c = max(0.0, room_w - thickness), room_w
                snapped = [x1c, y1c, x2c, y2c]
            else:
                x1c, x2c = max(0.0, min(x1,x2)), min(room_w, max(x1,x2))
                if x2c - x1c < 20.0:
                    continue
                if side == "top":
                    y1c, y2c = 0.0, min(thickness, room_h)
                else:
                    y1c, y2c = max(0.0, room_h - thickness), room_h
                snapped = [x1c, y1c, x2c, y2c]
            buckets[(t, side)].append({"bbox": snapped, "src": d})
        # for each bucket, keep the longest span
        for key, items in buckets.items():
            if not items:
                continue
            if key[1] in ("left","right"):
                items.sort(key=lambda it: abs(it["bbox"][3]-it["bbox"][1]), reverse=True)
            else:
                items.sort(key=lambda it: abs(it["bbox"][2]-it["bbox"][0]), reverse=True)
            keep = items[0]
            t, _side = key
            src = keep["src"]
            snapped_bb = keep["bbox"]
            # Relabel heuristic: if any raw detection of type 'window' overlaps this snapped opening -> window
            def _iou(a,b):
                ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
                ix1,iy1=max(ax1,bx1),max(ay1,by1)
                ix2,iy2=min(ax2,bx2),min(ay2,by2)
                if ix2<=ix1 or iy2<=iy1: return 0.0
                inter=(ix2-ix1)*(iy2-iy1)
                ua=(ax2-ax1)*(ay2-ay1); ub=(bx2-bx1)*(by2-by1)
                uni=ua+ub-inter
                return inter/uni if uni>0 else 0.0
            raw = dets  # pre-snapping raw list
            win_overlap = max((_iou(snapped_bb, (r.get("room_bbox") or r.get("bbox")))
                               for r in raw if str(r.get("type") or r.get("label") or "").lower()=="window"), default=0.0)
            door_overlap = max((_iou(snapped_bb, (r.get("room_bbox") or r.get("bbox")))
                                for r in raw if str(r.get("type") or r.get("label") or "").lower()=="door"), default=0.0)
            label = t
            if win_overlap > 0.2 and win_overlap >= door_overlap:
                label = "window"
            elif door_overlap > 0.2 and door_overlap > win_overlap:
                label = "door"
            else:
                # Span-based heuristic
                x1,y1,x2,y2 = snapped_bb
                span_v = abs(y2-y1)
                span_h = abs(x2-x1)
                if _side in ("left","right"):
                    # Tall vertical span likely a door; shorter is window
                    label = "door" if span_v >= 0.6*room_h else "window"
                else:
                    # Horizontal span windows are common; treat as window by default
                    label = "window"
            out.append({
                "type": label,
                "label": label,
                "bbox": snapped_bb,
                "confidence": float(src.get("confidence", 0.9)),
                "category": "architectural",
            })
        self.room_analysis["detections"] = out

    def _infer_openings_from_image(self, image_path: str) -> None:
        """Heuristic: if no openings detected, infer one opening per closest edge using Hough lines near borders.
        Creates detections in image pixels; snapping will convert to room coords.
        """
        try:
            import cv2
            import numpy as np
            img = cv2.imread(image_path)
            if img is None:
                return
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5,5), 0)
            edges = cv2.Canny(gray, 50, 150)
            h, w = edges.shape[:2]
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=int(0.25*min(h,w)), maxLineGap=20)
            if lines is None:
                return
            # Buckets by side
            margin_px = max(10, int(0.06*min(w,h)))
            sides = {"left": [], "right": [], "top": [], "bottom": []}
            for l in lines[:,0,:]:
                x1,y1,x2,y2 = map(int, l)
                # classify by proximity to an image border
                cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
                d_left, d_right, d_top, d_bottom = cx, w-cx, cy, h-cy
                dm = min(d_left, d_right, d_top, d_bottom)
                if dm > margin_px:
                    continue
                if dm == d_left:
                    sides["left"].append((x1,y1,x2,y2))
                elif dm == d_right:
                    sides["right"].append((x1,y1,x2,y2))
                elif dm == d_top:
                    sides["top"].append((x1,y1,x2,y2))
                else:
                    sides["bottom"].append((x1,y1,x2,y2))
            dets = list(self.room_analysis.get("detections", []))
            # helper to add opening bbox in image px
            def add_opening(side:str, segs:list, label:str):
                if not segs:
                    return
                # choose longest span on that side
                if side in ("left","right"):
                    segs.sort(key=lambda s: abs(s[3]-s[1]), reverse=True)
                    x = 0 if side=="left" else w
                    y1 = max(0, min(segs[0][1], segs[0][3])); y2 = min(h, max(segs[0][1], segs[0][3]))
                    # expand a bit
                    pad = int(0.02*h)
                    y1 = max(0, y1 - pad); y2 = min(h, y2 + pad)
                    if y2 - y1 < int(0.1*h):
                        return
                    # make a thin bbox at that edge
                    thick = int(0.02*w)
                    if side=="left":
                        bb = [0, y1, min(thick, w), y2]
                    else:
                        bb = [max(0, w-thick), y1, w, y2]
                else:
                    segs.sort(key=lambda s: abs(s[2]-s[0]), reverse=True)
                    y = 0 if side=="top" else h
                    x1 = max(0, min(segs[0][0], segs[0][2])); x2 = min(w, max(segs[0][0], segs[0][2]))
                    pad = int(0.02*w)
                    x1 = max(0, x1 - pad); x2 = min(w, x2 + pad)
                    if x2 - x1 < int(0.2*w*0.2):
                        return
                    thick = int(0.02*h)
                    if side=="top":
                        bb = [x1, 0, x2, min(thick, h)]
                    else:
                        bb = [x1, max(0, h-thick), x2, h]
                dets.append({
                    "type": label,
                    "label": label,
                    "bbox": list(map(float, bb)),
                    "confidence": 0.6,
                    "category": "architectural",
                })
            # Prefer door on left/right, window on any long edge if available
            add_opening("left", sides["left"], "door")
            add_opening("right", sides["right"], "door")
            add_opening("top", sides["top"], "window")
            add_opening("bottom", sides["bottom"], "window")
            self.room_analysis["detections"] = dets
        except Exception:
            return
    
    def process_empty_room(self, image_path: str, selected_furniture: List[Dict],
                          user_preferences: Dict = None) -> Dict:
        """
        Complete pipeline for processing an empty room:
        1. Detect architectural elements
        2. Optimize furniture placement
        3. Generate 2D layout
        """
        print("[AI Room Optimizer] Processing empty room...")
        
        # Step 1: Analyze room
        room_analysis = self.analyze_room(image_path)
        
        # Step 2: Optimize furniture placement
        optimized_furniture = self.optimize_furniture_placement(selected_furniture, user_preferences)
        
        # Step 3: Generate 2D layout
        layout_buffer = self.generate_2d_layout()
        
        # Calculate metrics with real-world dimensions
        room_area = (self.room_dims[0] * self.room_dims[1]) / 10000  # m²

        def _footprint_wh(obj: Dict) -> Tuple[float, float]:
            """Robustly get furniture footprint (w,h) in cm from various keys."""
            w = float(
                obj.get("w",
                        obj.get("w_cm",
                                obj.get("width", 0.0)))
            )
            h = float(
                obj.get("h",
                        obj.get("h_cm",
                                obj.get("depth", obj.get("height", 0.0))))
            )
            return max(0.0, w), max(0.0, h)

        furniture_area = sum(_footprint_wh(obj)[0] * _footprint_wh(obj)[1] for obj in optimized_furniture) / 10000  # m²
        space_utilization = (furniture_area / room_area) * 100 if room_area > 0 else 0
        
        # Calculate total furniture volume for better space analysis
        total_furniture_volume = sum(
            _footprint_wh(obj)[0] * _footprint_wh(obj)[1] * float(obj.get("height", obj.get("height_cm", 80)))
            for obj in optimized_furniture
        ) / 1000000  # m³
        
        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(optimized_furniture, room_analysis)
        
        return {
            "room_analysis": room_analysis,
            "optimized_furniture": optimized_furniture,
            "layout_image": layout_buffer,
            "metrics": {
                "room_area_m2": room_area,
                "furniture_area_m2": furniture_area,
                "furniture_volume_m3": total_furniture_volume,
                "space_utilization_percent": space_utilization,
                "optimization_score": optimization_score,
                "architectural_elements_count": len(self.architectural_elements),
                "furniture_items_count": len(optimized_furniture),
                "average_furniture_size_cm": (
                    sum(_footprint_wh(obj)[0] * _footprint_wh(obj)[1] for obj in optimized_furniture) / len(optimized_furniture)
                    if optimized_furniture else 0
                ),
                "optimizer_method": getattr(self, "last_method", "unknown"),
            },
            "room_type": self.room_type,
            "recommendations": self._generate_recommendations(room_analysis, optimized_furniture)
        }
    
    def _calculate_optimization_score(self, furniture_layout: List[Dict], room_analysis: Dict) -> float:
        """Calculate overall optimization score."""
        score = 0.0
        
        # Base score for successful placement
        score += 1000.0
        
        # Penalty for overlaps (should be 0)
        overlap_penalty = 0
        for i in range(len(furniture_layout)):
            for j in range(i + 1, len(furniture_layout)):
                if self._overlap(furniture_layout[i], furniture_layout[j]):
                    overlap_penalty += 1000
        
        score -= overlap_penalty
        
        # Bonus for good furniture relationships
        relationship_bonus = self._calculate_relationship_bonus(furniture_layout)
        score += relationship_bonus
        
        # Bonus for architectural compliance
        architectural_bonus = self._calculate_architectural_bonus(furniture_layout, room_analysis)
        score += architectural_bonus
        
        # Bonus for space utilization
        space_bonus = self._calculate_space_utilization_bonus(furniture_layout)
        score += space_bonus
        
        return max(0, score)
    
    def _footprint_wh_safe(self, obj: Dict) -> Tuple[float, float]:
        """Robustly get furniture footprint (w,h) in cm from various keys.

        Supports legacy GA layouts (w/h), solver outputs (w_cm/h_cm), and
        more generic width/depth/height-style keys. Never raises KeyError.
        """
        try:
            w = float(
                obj.get("w",
                        obj.get("w_cm",
                                obj.get("width", 0.0)))
            )
            h = float(
                obj.get("h",
                        obj.get("h_cm",
                                obj.get("depth", obj.get("height", 0.0))))
            )
        except Exception:
            w, h = 0.0, 0.0
        return max(0.0, w), max(0.0, h)

    def _calculate_relationship_bonus(self, furniture_layout: List[Dict]) -> float:
        """Calculate bonus for good furniture relationships."""
        bonus = 0.0
        
        # Define furniture relationships
        relationships = {
            "sofa": ["coffee table", "tv", "lamp"],
            "bed": ["nightstand", "dresser", "wardrobe"],
            "desk": ["chair", "lamp"],
            "dining table": ["chair"],
            "tv": ["tv stand", "sofa"]
        }
        
        for obj1 in furniture_layout:
            obj1_type = obj1.get("type", "").lower()
            related_types = relationships.get(obj1_type, [])
            
            for obj2 in furniture_layout:
                if obj1 == obj2:
                    continue
                
                obj2_type = obj2.get("type", "").lower()
                if obj2_type in related_types:
                    distance = self._center_distance(obj1, obj2)
                    if distance < 200:  # Within 2m
                        bonus += 50 * (1 - distance / 200)
        
        return bonus
    
    def _calculate_architectural_bonus(self, furniture_layout: List[Dict], room_analysis: Dict) -> float:
        """Calculate bonus for architectural compliance."""
        bonus = 0.0
        def _dtype(det: Dict) -> str:
            return str(det.get("type") or det.get("label") or "").lower().replace("_", " ")
        windows = [d for d in room_analysis.get("detections", []) if _dtype(d) == "window"]
        doors = [d for d in room_analysis.get("detections", []) if _dtype(d) == "door"]
        
        # Bonus for placing seating near windows
        for obj in furniture_layout:
            if obj.get("type", "").lower() in ["chair", "sofa", "desk"]:
                w, h = self._footprint_wh_safe(obj)
                x = float(obj.get("x", 0.0)); y = float(obj.get("y", 0.0))
                obj_center = (x + w/2, y + h/2)
                
                for window in windows:
                    wx1, wy1, wx2, wy2 = window["bbox"]
                    window_center = ((wx1 + wx2) / 2, (wy1 + wy2) / 2)
                    distance = np.sqrt((obj_center[0] - window_center[0])**2 + 
                                     (obj_center[1] - window_center[1])**2)
                    
                    if distance < 150:  # Within 1.5m
                        bonus += 30 * (1 - distance / 150)
        
        # Penalty for blocking doors
        for obj in furniture_layout:
            w, h = self._footprint_wh_safe(obj)
            x = float(obj.get("x", 0.0)); y = float(obj.get("y", 0.0))
            obj_center = (x + w/2, y + h/2)
            
            for door in doors:
                dx1, dy1, dx2, dy2 = door["bbox"]
                door_center = ((dx1 + dx2) / 2, (dy1 + dy2) / 2)
                distance = np.sqrt((obj_center[0] - door_center[0])**2 + 
                                 (obj_center[1] - door_center[1])**2)
                
                if distance < 80:  # Too close to door
                    bonus -= 100
        
        return bonus
    
    def _calculate_space_utilization_bonus(self, furniture_layout: List[Dict]) -> float:
        """Calculate bonus for good space utilization."""
        if not furniture_layout:
            return 0.0
        
        # Calculate furniture area
        furniture_area = 0.0
        for obj in furniture_layout:
            w, h = self._footprint_wh_safe(obj)
            furniture_area += w * h
        room_area = self.room_dims[0] * self.room_dims[1]
        utilization = furniture_area / room_area if room_area > 0 else 0
        
        # Optimal utilization is around 30-50%
        if 0.3 <= utilization <= 0.5:
            return 100
        elif utilization < 0.3:
            return 50 * utilization / 0.3
        else:
            return max(0, 100 - (utilization - 0.5) * 200)

    def _calculate_metrics(self, layout: List[Dict], room_dims: Tuple[float, float], detected_openings: List[Dict]) -> Dict:
        """Calculate layout quality metrics including room coverage.

        Room coverage is expressed as a percentage of floor area occupied by
        furniture footprints.
        """

        room_width, room_length = room_dims
        room_area = room_width * room_length

        # Calculate furniture coverage
        total_furniture_area = 0.0
        for item in layout:
            w, h = self._footprint_wh_safe(item)
            total_furniture_area += w * h
        coverage_ratio = (total_furniture_area / room_area) if room_area > 0 else 0.0

        metrics: Dict[str, Any] = {
            "furniture_count": len(layout),
            "room_coverage": round(coverage_ratio * 100, 1),  # percentage
            "coverage_status": "optimal"
            if 0.35 < coverage_ratio < 0.60
            else "too_crowded"
            if coverage_ratio > 0.60
            else "sparse",
        }

        return metrics
    
    def _generate_recommendations(self, room_analysis: Dict, furniture_layout: List[Dict]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Lighting recommendations
        lighting_analysis = room_analysis.get("recommendations", {}).get("lighting_considerations", {})
        if lighting_analysis.get("natural_light") == "low":
            recommendations.append("Consider adding more lighting fixtures near seating areas")
        
        # Space utilization recommendations
        furniture_area = 0.0
        for obj in furniture_layout:
            w, h = self._footprint_wh_safe(obj)
            furniture_area += w * h
        room_area = self.room_dims[0] * self.room_dims[1]
        utilization = furniture_area / room_area if room_area > 0 else 0
        
        if utilization < 0.2:
            recommendations.append("Room has plenty of space - consider adding more furniture or decorative elements")
        elif utilization > 0.6:
            recommendations.append("Room is quite full - consider removing some items for better flow")
        
        # Furniture relationship recommendations
        has_sofa = any("sofa" in obj.get("type", "").lower() for obj in furniture_layout)
        has_coffee_table = any("coffee table" in obj.get("type", "").lower() for obj in furniture_layout)
        
        if has_sofa and not has_coffee_table:
            recommendations.append("Consider adding a coffee table near the sofa for better functionality")
        
        return recommendations
    
    def _overlap(self, a: Dict, b: Dict) -> bool:
        """Check if two objects overlap."""
        ax = float(a.get("x", 0.0)); ay = float(a.get("y", 0.0))
        bx = float(b.get("x", 0.0)); by = float(b.get("y", 0.0))
        aw, ah = self._footprint_wh_safe(a)
        bw, bh = self._footprint_wh_safe(b)
        if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
            return False
        return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)
    
    def _center_distance(self, a: Dict, b: Dict) -> float:
        """Calculate distance between centers of two objects."""
        aw, ah = self._footprint_wh_safe(a)
        bw, bh = self._footprint_wh_safe(b)
        ax = float(a.get("x", 0.0)) + aw / 2.0
        ay = float(a.get("y", 0.0)) + ah / 2.0
        bx = float(b.get("x", 0.0)) + bw / 2.0
        by = float(b.get("y", 0.0)) + bh / 2.0
        return np.sqrt((ax - bx)**2 + (ay - by)**2)
    
    def export_results(self, output_path: str) -> Dict:
        """Export complete results to JSON file."""
        results = {
            "room_dimensions": {
                "width_cm": self.room_dims[0],
                "height_cm": self.room_dims[1],
                "area_m2": (self.room_dims[0] * self.room_dims[1]) / 10000
            },
            "architectural_elements": self.architectural_elements,
            "furniture_layout": self.furniture_layout,
            "room_analysis": self.room_analysis,
            "optimization_metrics": self._calculate_optimization_score(
                self.furniture_layout, self.room_analysis
            ) if self.furniture_layout and self.room_analysis else 0
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
