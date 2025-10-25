"""
enhanced_detector.py
-------------------
Advanced multi-pass detection system with edge detection for paintings/artwork
and intelligent object suggestion system for real-world room images.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
from PIL import Image

class EnhancedDetector:
    """Enhanced detection with edge detection and intelligent suggestions."""
    
    def __init__(self, base_detector):
        self.base_detector = base_detector
    
    def detect_paintings_artwork(self, image_path: str, image_array: np.ndarray = None) -> List[Dict]:
        """
        Detect rectangular objects on walls (paintings, pictures, artwork, mirrors, doors)
        using edge detection and contour analysis.
        """
        if image_array is None:
            image_array = cv2.imread(image_path)
        
        if image_array is None:
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply Canny edge detection - EXTREMELY sensitive to catch architectural elements
        edges = cv2.Canny(blurred, 15, 70)  # More sensitive for doors/mirrors
        
        # Second pass with different thresholds for variety
        edges2 = cv2.Canny(blurred, 30, 100)
        edges = cv2.bitwise_or(edges, edges2)  # Combine both passes
        
        # Apply morphological closing to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        paintings = []
        img_h, img_w = gray.shape
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Look for quadrilaterals (4 corners)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter by aspect ratio and size
                aspect_ratio = w / h if h > 0 else 0
                
                # Paintings/artwork typically:
                # - Are rectangular (aspect ratio between 0.4 and 4.0) - wider range
                # - Cover ~0.2% to 20% of image area (even lower to catch more)
                # - Are relatively flat (small depth in 3D)
                
                min_area = img_w * img_h * 0.002  # 0.2% - lowered
                max_area = img_w * img_h * 0.20   # 20% - increased
                
                if (0.3 < aspect_ratio < 5.0 and  # Even wider for doors/mirrors
                    min_area < area < max_area and
                    w > 15 and h > 15):  # Even lower minimum size for architectural elements
                    
                    # Calculate position in image coordinates
                    x1, y1 = x, y
                    x2, y2 = x + w, y + h
                    
                    # Quick exclusions to avoid floor shadows being treated as paintings
                    # Discard very low regions (likely floor) - more lenient
                    if (y2 > img_h * 0.95) or (y1 > img_h * 0.85 and h > img_h * 0.08):
                        continue

                    # Check if it's likely on a wall (upper ~95% of image) - very lenient for doors/mirrors
                    center_y = y + h / 2
                    if center_y < img_h * 0.95:
                        crop = image_array[y1:y2, x1:x2]
                        
                        # CRITICAL: Priority-based classification to prevent confusion
                        # Check in order: Door -> Window -> Mirror -> Painting
                        # This prevents mislabeling by checking most specific first
                        
                        # Priority 1: DOOR (very tall objects with panels)
                        if h > 100 and aspect_ratio < 0.8:  # Very tall, narrow
                            is_door = self._is_door_like(crop, w, h)
                            if is_door:
                                paintings.append({
                                    "type": "door",
                                    "bbox": [x1, y1, x2, y2],
                                    "confidence": 0.75,
                                    "width_px": w,
                                    "height_px": h,
                                    "method": "edge_detection_door"
                                })
                                continue
                        
                        # Priority 2: DOOR (medium tall with strong door characteristics)
                        if h > 80 and w > 40 and aspect_ratio < 1.2:
                            is_door = self._is_door_like(crop, w, h)
                            # Additional check: doors are typically darker than windows
                            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            mean_intensity = float(np.mean(crop_gray))
                            is_door_dark = mean_intensity < 160  # Doors are darker
                            
                            if is_door and is_door_dark:
                                paintings.append({
                                    "type": "door",
                                    "bbox": [x1, y1, x2, y2],
                                    "confidence": 0.70,
                                    "width_px": w,
                                    "height_px": h,
                                    "method": "edge_detection_door"
                                })
                                continue
                        
                        # Priority 3: WINDOW (must pass strict criteria)
                        is_window, window_conf = self._is_window_like(crop, w, h)
                        if is_window and window_conf > 0.65:  # Higher threshold
                            # Additional validation: windows are typically brighter
                            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            mean_intensity = float(np.mean(crop_gray))
                            if mean_intensity > 120:  # Windows are bright
                                paintings.append({
                                    "type": "window",
                                    "bbox": [x1, y1, x2, y2],
                                    "confidence": float(max(0.65, min(0.95, window_conf))),
                                    "width_px": w,
                                    "height_px": h,
                                    "method": "edge_detection_window"
                                })
                                continue
                        
                        # Priority 4: MIRROR (square-ish bright objects)
                        if 0.7 < aspect_ratio < 1.4 and w > 30 and h > 30:
                            is_mirror = self._is_mirror_like(crop, w, h)
                            if is_mirror:
                                paintings.append({
                                    "type": "mirror",
                                    "bbox": [x1, y1, x2, y2],
                                    "confidence": 0.65,
                                    "width_px": w,
                                    "height_px": h,
                                    "method": "edge_detection_mirror"
                                })
                                continue
                        
                        # Priority 5: PAINTING (everything else - decorative art)
                        # Only classify as painting if doesn't match above categories
                        crop_gray_for_paint = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop is not None and crop.size > 0 else None
                        if crop_gray_for_paint is not None:
                            mean_i = float(np.mean(crop_gray_for_paint))
                            std_i = float(np.std(crop_gray_for_paint))
                            # Extremely bright and uniform -> likely light/shadow on floor/wall, not a painting
                            if mean_i > 195 and std_i < 8:  # More strict exclusion criteria
                                continue
                            # High variance suggests artwork texture, classify as painting
                            if std_i > 30:  # Painting has texture
                                paintings.append({
                                    "type": "painting",
                                    "bbox": [x1, y1, x2, y2],
                                    "confidence": 0.75,  # Higher confidence for textured items
                                    "width_px": w,
                                    "height_px": h,
                                    "method": "edge_detection"
                                })
                            elif std_i > 20:  # Medium variance
                                paintings.append({
                                    "type": "painting",
                                    "bbox": [x1, y1, x2, y2],
                                    "confidence": 0.65,
                                    "width_px": w,
                                    "height_px": h,
                                    "method": "edge_detection"
                                })
        
        # Remove overlapping detections (keep larger ones)
        paintings = self._remove_overlapping_paintings(paintings)

        # If only one window found, try to recover a symmetric partner
        try:
            windows = [p for p in paintings if p.get("type") == "window"]
            if len(windows) == 1:
                candidate = self._find_symmetric_window_candidate(image_array, windows[0], img_w, img_h)
                if candidate is not None:
                    paintings.append(candidate)
        except Exception:
            pass
        
        return paintings

    def _is_window_like(self, crop_bgr: np.ndarray, w: int, h: int) -> Tuple[bool, float]:
        """Heuristic to decide if a rectangular region is likely a window.
        Signals used:
        - Brightness: windows are often brighter than walls
        - Grid/lines: vertical/horizontal muntins detected via HoughLinesP
        - Edge density: interior edges suggest panes
        - Color variance: paintings have more texture/color variance than windows
        - Uniformity: windows are more uniform, paintings have artwork/texture
        Returns (is_window, confidence)
        """
        try:
            if crop_bgr is None or crop_bgr.size == 0:
                return False, 0.0
            crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            mean_intensity = float(np.mean(crop_gray))
            std_intensity = float(np.std(crop_gray))
            
            # Convert to HSV for better color analysis
            crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            mean_saturation = float(np.mean(crop_hsv[:, :, 1]))
            mean_value = float(np.mean(crop_hsv[:, :, 2]))

            # Canny on crop
            e = cv2.Canny(crop_gray, 50, 150)
            # Hough lines
            lines = cv2.HoughLinesP(
                e,
                1,
                np.pi/180,
                threshold=max(20, int(min(w, h) * 0.06)),
                minLineLength=max(6, int(min(w, h) * 0.25)),
                maxLineGap=12,
            )
            v_count = 0
            h_count = 0
            if lines is not None:
                for ln in lines:
                    x1, y1, x2, y2 = ln[0]
                    dx = abs(x2 - x1)
                    dy = abs(y2 - y1)
                    if dx < dy:  # vertical-ish
                        v_count += 1
                    else:
                        h_count += 1

            edge_density = float(np.count_nonzero(e)) / float(e.size)

            # KEY FIX: Distinguish paintings from windows
            # Paintings typically have:
            # - High color variance (artwork has texture/patterns)
            # - More edges (artistic details)
            # - Higher saturation (colorful artwork)
            # - Less uniform intensity
            
            # Windows typically have:
            # - Lower color variance (glass is uniform)
            # - Grid-like patterns (muntins)
            # - High brightness but uniform
            # - Lower saturation (clear or white frames)
            
            is_painting_like = (
                std_intensity > 35 or  # High intensity variance (artwork texture)
                edge_density > 0.15 or  # Too many edges (too artistic)
                mean_saturation > 100 or  # Too colorful (artwork)
                (std_intensity > 25 and edge_density > 0.10)  # High variance + edges
            )
            
            # If it looks like a painting, it's NOT a window
            if is_painting_like:
                return False, 0.0
            
            # Window detection criteria (VERY strict)
            bright = mean_intensity > 140  # bright region
            has_grid = (v_count >= 2 and h_count >= 1) or (v_count + h_count >= 4)  # Clear grid pattern
            pane_like = 0.03 < edge_density < 0.12  # Moderate edge density (not too low, not too high)
            uniform = std_intensity < 30  # Uniform (not artistic texture)
            low_saturation = mean_saturation < 80  # Low color saturation
            
            # STRICT: Must pass ALL criteria to be a window
            # Windows are very different from doors (bright vs dark)
            is_window = (
                bright and      # Must be bright (not door)
                uniform and     # Must be uniform (not painting)
                low_saturation and  # Must be low saturation (not colorful painting)
                (has_grid or pane_like)  # Must have grid/panes (not door)
            )
            
            # Confidence based on signals observed
            conf = 0.3
            if bright:
                conf += 0.15
            if has_grid:
                conf += 0.25
            if pane_like:
                conf += 0.15
            if uniform:
                conf += 0.10
            if low_saturation:
                conf += 0.05
                
            return bool(is_window), float(min(0.99, max(0.0, conf)))
        except Exception:
            return False, 0.0

    def _is_door_like(self, crop_bgr: np.ndarray, w: int, h: int) -> bool:
        """Check if region looks like a door - STRICT criteria with artistic pattern exclusion."""
        try:
            if crop_bgr is None or crop_bgr.size == 0:
                return False
            crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            
            # Convert to HSV for saturation analysis
            crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            mean_saturation = float(np.mean(crop_hsv[:, :, 1]))
            
            # Doors typically have:
            # - Vertical lines (panels) - CRITICAL
            # - Medium brightness (darker than windows)
            # - Some texture but not artistic
            # - Usually darker than surrounding walls
            # - LOW saturation (not colorful artwork!)
            
            edges = cv2.Canny(crop_gray, 30, 100)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                   threshold=max(15, int(min(w, h) * 0.1)),
                                   minLineLength=max(10, int(h * 0.3)),
                                   maxLineGap=20)
            
            vertical_lines = 0
            line_positions = []
            if lines is not None:
                for ln in lines:
                    x1, y1, x2, y2 = ln[0]
                    if abs(x2 - x1) < abs(y2 - y1):  # vertical
                        vertical_lines += 1
                        # Store line positions to check for patterns
                        if abs(x2 - x1) < abs(y2 - y1):
                            line_positions.append((x1 + x2) / 2)
            
            # CRITICAL: Check if vertical lines form structural panels or artistic patterns
            # Doors have evenly spaced structural panels
            # Paintings have wavy, irregular artistic lines
            has_structural_panels = False
            if len(line_positions) >= 2:
                # Check if lines are evenly spaced (structural) vs irregular (artistic)
                line_positions.sort()
                spacings = [line_positions[i+1] - line_positions[i] for i in range(len(line_positions)-1)]
                if len(spacings) > 0:
                    mean_spacing = sum(spacings) / len(spacings)
                    spacing_variance = sum((s - mean_spacing)**2 for s in spacings) / len(spacings)
                    # Structural panels have low variance in spacing (evenly distributed)
                    # Artistic patterns have high variance (irregular)
                    has_structural_panels = spacing_variance < mean_spacing * 0.3
            
            # STRICT: Must have clear structural vertical panels
            has_panels = vertical_lines >= 2 and has_structural_panels
            
            # Check brightness (doors are darker than windows)
            mean_intensity = float(np.mean(crop_gray))
            std_intensity = float(np.std(crop_gray))
            
            # Doors are medium-dark, not very bright (bright = window)
            is_door_brightness = 70 < mean_intensity < 160
            
            # Doors have moderate texture (not artistic paintings)
            has_moderate_texture = 15 < std_intensity < 45
            
            # CRITICAL: Doors have LOW saturation (not colorful artwork)
            # Paintings are colorful (high saturation)
            is_door_color = mean_saturation < 80  # Not colorful
            
            # EXCLUDE artistic patterns - paintings have wavy/curved lines, not straight structural
            # Check edge curvature - paintings have more curved edges
            edges_binary = edges > 0
            contours, _ = cv2.findContours(edges_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            has_wavy_patterns = False
            if len(contours) > 3:  # Too many small contours = artistic patterns
                # Calculate curvature/high-frequency changes
                total_contour_length = sum(cv2.arcLength(cnt, False) for cnt in contours)
                if total_contour_length > w * h * 0.1:  # Too much detail = artistic
                    has_wavy_patterns = True
            
            # Combine all criteria - ALL must pass
            # PRIME REQUIREMENT: Not artistic (not colorful, not wavy patterns)
            is_not_artistic = is_door_color and not has_wavy_patterns
            
            return has_panels and is_door_brightness and has_moderate_texture and is_not_artistic
        except Exception:
            return False
    
    def _is_mirror_like(self, crop_bgr: np.ndarray, w: int, h: int) -> bool:
        """Check if region looks like a mirror."""
        try:
            if crop_bgr is None or crop_bgr.size == 0:
                return False
            
            # Mirrors typically have:
            # - Reflections (color diversity)
            # - Bright/shiny appearance
            # - Frame around edges
            
            crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            mean_intensity = float(np.mean(crop_gray))
            std_intensity = float(np.std(crop_gray))
            
            # Mirrors are bright and somewhat uniform
            is_bright = mean_intensity > 100
            is_somewhat_uniform = std_intensity < 50
            
            # Check for frame-like edges
            edges = cv2.Canny(crop_gray, 30, 100)
            edge_density = float(np.count_nonzero(edges)) / float(edges.size)
            
            # Moderate edge density (frame + some reflection details)
            has_frame_pattern = 0.05 < edge_density < 0.20
            
            return is_bright and is_somewhat_uniform and has_frame_pattern
        except Exception:
            return False
    
    def _find_symmetric_window_candidate(self, img_bgr: np.ndarray, ref_window: Dict, img_w: int, img_h: int) -> Dict:
        """Given one detected window, search horizontally for a symmetric counterpart.
        Strategy:
        - Mirror the x-position around image center to estimate expected location
        - Search in a horizontal band near the reference center_y with similar height
        - Use window-like heuristic with relaxed thresholds
        Returns a detection dict if found, else None.
        """
        x1, y1, x2, y2 = ref_window["bbox"]
        w = x2 - x1
        h = y2 - y1
        cy = (y1 + y2) / 2.0
        cx = (x1 + x2) / 2.0
        mirror_cx = img_w - cx

        # Define search ROI around mirrored position
        band_h = int(max(20, h * 0.35))
        band_y1 = max(0, int(cy - band_h))
        band_y2 = min(img_h, int(cy + band_h))
        band_w = int(max(30, w * 1.2))
        band_x1 = max(0, int(mirror_cx - band_w))
        band_x2 = min(img_w, int(mirror_cx + band_w))

        roi = img_bgr[band_y1:band_y2, band_x1:band_x2]
        if roi is None or roi.size == 0:
            return None

        # Edge-based rectangle search in ROI
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 7, 60, 60)
        e = cv2.Canny(blur, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(e, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = 0.0
        for cnt in contours:
            x, y, ww, hh = cv2.boundingRect(cnt)
            # Map to image coords
            gx1 = band_x1 + x
            gy1 = band_y1 + y
            gx2 = gx1 + ww
            gy2 = gy1 + hh

            # Basic filters
            if hh < h * 0.6 or hh > h * 1.4:
                continue
            aspect = ww / float(hh)
            if not (0.5 < aspect < 3.0):
                continue

            crop = img_bgr[gy1:gy2, gx1:gx2]
            is_win, conf = self._is_window_like(crop, ww, hh)
            if is_win:
                # Score by closeness of center_y to reference and size similarity
                ccy = (gy1 + gy2) / 2.0
                y_sim = 1.0 - min(1.0, abs(ccy - cy) / max(1.0, h))
                size_sim = 1.0 - min(1.0, abs(hh - h) / max(1.0, h))
                score = conf * 0.6 + y_sim * 0.25 + size_sim * 0.15
                if score > best_score:
                    best_score = score
                    best = [gx1, gy1, gx2, gy2]

        if best is None or best_score < 0.55:
            return None

        return {
            "type": "window",
            "bbox": best,
            "confidence": float(min(0.95, 0.6 + best_score * 0.4)),
            "width_px": best[2] - best[0],
            "height_px": best[3] - best[1],
            "method": "edge_detection_window_symmetry"
        }
    
    def _remove_overlapping_paintings(self, paintings: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Remove overlapping painting detections."""
        if len(paintings) <= 1:
            return paintings
        
        # Sort by area (largest first)
        paintings_sorted = sorted(paintings, 
                                 key=lambda p: p['width_px'] * p['height_px'], 
                                 reverse=True)
        
        keep = []
        for i, p1 in enumerate(paintings_sorted):
            overlap = False
            for p2 in keep:
                iou = self._calculate_iou(p1['bbox'], p2['bbox'])
                if iou > iou_threshold:
                    overlap = True
                    break
            if not overlap:
                keep.append(p1)
        
        return keep
    
    def _calculate_iou(self, bbox1: List, bbox2: List) -> float:
        """Calculate Intersection over Union."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def suggest_missing_objects(self, detected_objects: List[str], room_type: str = "living_room") -> List[Dict]:
        """
        Intelligently suggest objects that are typically found in rooms
        but weren't detected.
        """
        
        # Define typical object sets for different room types
        typical_objects = {
            "living_room": [
                {"type": "sofa", "priority": "high", "typical_size": (200, 90)},
                {"type": "coffee table", "priority": "high", "typical_size": (100, 60)},
                {"type": "tv", "priority": "medium", "typical_size": (120, 70)},
                {"type": "tv stand", "priority": "medium", "typical_size": (140, 40)},
                {"type": "lamp", "priority": "medium", "typical_size": (30, 30)},
                {"type": "plant", "priority": "low", "typical_size": (40, 40)},
                {"type": "painting", "priority": "low", "typical_size": (60, 80)},
                {"type": "bookshelf", "priority": "low", "typical_size": (80, 180)},
            ],
            "bedroom": [
                {"type": "bed", "priority": "high", "typical_size": (200, 160)},
                {"type": "nightstand", "priority": "high", "typical_size": (50, 50)},
                {"type": "wardrobe", "priority": "medium", "typical_size": (120, 60)},
                {"type": "dresser", "priority": "medium", "typical_size": (100, 50)},
                {"type": "lamp", "priority": "medium", "typical_size": (25, 25)},
                {"type": "mirror", "priority": "low", "typical_size": (60, 80)},
                {"type": "chair", "priority": "low", "typical_size": (50, 50)},
            ],
            "office": [
                {"type": "desk", "priority": "high", "typical_size": (140, 70)},
                {"type": "chair", "priority": "high", "typical_size": (60, 60)},
                {"type": "bookshelf", "priority": "medium", "typical_size": (80, 180)},
                {"type": "laptop", "priority": "medium", "typical_size": (35, 25)},
                {"type": "lamp", "priority": "medium", "typical_size": (25, 25)},
                {"type": "filing cabinet", "priority": "low", "typical_size": (50, 70)},
            ],
            "dining_room": [
                {"type": "table", "priority": "high", "typical_size": (180, 90)},
                {"type": "chair", "priority": "high", "typical_size": (50, 50), "quantity": 4},
                {"type": "cabinet", "priority": "medium", "typical_size": (100, 50)},
                {"type": "plant", "priority": "low", "typical_size": (40, 40)},
            ]
        }
        
        expected = typical_objects.get(room_type, typical_objects["living_room"])
        detected_types = [obj.lower() for obj in detected_objects]
        
        suggestions = []
        for obj in expected:
            obj_type = obj["type"].lower()
            # Check if this type is already detected
            if not any(obj_type in det for det in detected_types):
                suggestions.append({
                    "type": obj["type"],
                    "priority": obj["priority"],
                    "suggested_width": obj["typical_size"][0],
                    "suggested_height": obj["typical_size"][1],
                    "reason": f"Typically found in {room_type.replace('_', ' ')}"
                })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return suggestions[:10]  # Return top 10 suggestions
    
    def multi_pass_detection(self, image_path: str, confidence_levels: List[float] = None) -> Dict:
        """
        Perform multiple detection passes with different confidence levels
        and combine results intelligently.
        """
        if confidence_levels is None:
            confidence_levels = [0.20, 0.10, 0.05]  # High, Medium, Low confidence - very aggressive for better recall
        
        image_array = cv2.imread(image_path)
        all_detections = {}
        
        # Pass 1: Standard YOLO detection with varying confidence
        for conf_level in confidence_levels:
            detections = self.base_detector.detect(image_path, conf_threshold=conf_level)
            for det in detections:
                det_key = f"{det['label']}_{int(det['bbox'][0])}_{int(det['bbox'][1])}"
                # Keep highest confidence detection for same object
                if det_key not in all_detections or det['confidence'] > all_detections[det_key]['confidence']:
                    all_detections[det_key] = det
        
        # Pass 2: Edge detection for paintings/artwork and windows
        paintings = self.detect_paintings_artwork(image_path, image_array)
        for i, painting in enumerate(paintings):
            det_key = f"painting_edge_{i}"
            if det_key not in all_detections:
                lbl = painting.get("type", "painting")
                all_detections[det_key] = {
                    "class": -1,
                    "label": lbl,
                    "confidence": painting["confidence"],
                    "bbox": painting["bbox"],
                    "category": "architectural" if lbl == "window" else "other",
                    "method": painting.get("method", "edge_detection")
                }
        
        # Convert back to list
        combined_detections = list(all_detections.values())
        
        # Remove duplicates (overlapping detections)
        combined_detections = self._deduplicate_detections(combined_detections)
        
        # Get suggestions
        detected_types = [d.get('label', 'unknown') for d in combined_detections]
        suggestions = self.suggest_missing_objects(detected_types, room_type="living_room")
        
        return {
            "detections": combined_detections,
            "suggestions": suggestions,
            "total_detected": len(combined_detections),
            "methods_used": ["yolo", "edge_detection", "intelligent_suggestions"]
        }
    
    def _deduplicate_detections(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Remove duplicate overlapping detections, keeping highest confidence."""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (highest first)
        detections_sorted = sorted(detections, key=lambda d: d.get('confidence', 0), reverse=True)
        
        keep = []
        for det in detections_sorted:
            overlap = False
            for kept_det in keep:
                iou = self._calculate_iou(det['bbox'], kept_det['bbox'])
                if iou > iou_threshold:
                    overlap = True
                    break
            if not overlap:
                keep.append(det)
        
        return keep
