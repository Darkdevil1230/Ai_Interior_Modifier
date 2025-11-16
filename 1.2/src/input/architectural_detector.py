"""
architectural_detector.py
-------------------------
Fallback architectural element detection using image processing.
Used when YOLO model doesn't have architectural element detection capability.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple


class ArchitecturalDetector:
    """Detect windows, doors, and walls using image processing techniques."""
    
    def __init__(self):
        self.min_window_area = 1000  # Minimum pixels for a window (lowered from 2000)
        self.min_door_area = 3000    # Minimum pixels for a door (lowered from 5000)
        
    def detect_architectural_elements(self, image_path: str) -> List[Dict]:
        """
        Detect windows and doors using edge detection and contour analysis.
        Returns list of detections in same format as YOLO detector.
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ArchitecturalDetector] Failed to load image: {image_path}")
            return []
        
        height, width = img.shape[:2]
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect windows (usually bright/white areas with rectangular shapes)
        windows = self._detect_windows(gray, width, height)
        detections.extend(windows)
        
        # Detect doors (usually darker rectangular areas near edges)
        doors = self._detect_doors(gray, width, height)
        detections.extend(doors)
        
        # Detect walls (image boundaries)
        walls = self._detect_walls(width, height)
        detections.extend(walls)
        
        print(f"[ArchitecturalDetector] Found {len(windows)} windows, {len(doors)} doors, {len(walls)} walls")
        
        return detections
    
    def _detect_windows(self, gray: np.ndarray, img_width: int, img_height: int) -> List[Dict]:
        """Detect windows using improved strategies with better accuracy."""
        detections = []
        seen_boxes = set()  # Avoid duplicates
        
        # Strategy 1: Bright areas (windows are typically bright)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Strategy 2: Edge detection for window frames
        edges = cv2.Canny(gray, 50, 150)
        
        # Strategy 3: Template matching for rectangular shapes
        # Find rectangular contours that could be windows
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Look for rectangular shapes (4 corners)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter by size and aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                
                # Windows typically:
                # - Are rectangular (aspect ratio between 0.1 and 5.0) - wider range
                # - Cover 0.1% to 20% of image area - more flexible
                # - Are in upper portion of image (not floor level)
                # - Have reasonable size relative to image
                
                min_area = img_width * img_height * 0.001  # 0.1% of image - much lower
                max_area = img_width * img_height * 0.20  # 20% of image - higher
                
                if (0.1 < aspect_ratio < 5.0 and  # Much wider range
                    min_area < area < max_area and
                    w > 15 and h > 15 and  # Lower minimum size
                    y < img_height * 0.8):  # Allow lower in image
                    
                    # Check if it's bright (window characteristic)
                    roi = gray[y:y+h, x:x+w]
                    if roi.size > 0:
                        mean_brightness = np.mean(roi)
                        
                        # Windows are typically bright
                        if mean_brightness > 150:  # Bright threshold
                            # Create unique signature to avoid duplicates
                            box_sig = (x // 20, y // 20, w // 20, h // 20)  # Rounded to nearest 20
                            if box_sig not in seen_boxes:
                                seen_boxes.add(box_sig)
                                
                                # Calculate confidence based on brightness and shape
                                brightness_score = min(1.0, mean_brightness / 255.0)
                                shape_score = 1.0 - abs(aspect_ratio - 1.5) / 1.5  # Prefer ~1.5 aspect ratio
                                confidence = (brightness_score * 0.7 + shape_score * 0.3)
                                
                                detections.append({
                                    "type": "window",
                                    "bbox": [x, y, x + w, y + h],
                                    "confidence": confidence,
                                    "category": "architectural"
                                })
        
        return detections
    
    def _detect_doors(self, gray: np.ndarray, img_width: int, img_height: int) -> List[Dict]:
        """Detect doors using improved shape and position analysis."""
        detections = []
        
        # Edge detection for door frames
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect door frame edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Look for rectangular shapes
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter by size and aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                
                # Doors typically:
                # - Are tall and narrow (aspect ratio < 1.0)
                # - Cover 0.5% to 25% of image area - more flexible
                # - Are tall (height > 20% of image height) - more flexible
                # - Are near edges (walls)
                
                min_area = img_width * img_height * 0.005  # 0.5% of image - much lower
                max_area = img_width * img_height * 0.25  # 25% of image - higher
                
                if (0.1 < aspect_ratio < 1.2 and  # Tall and narrow - wider range
                    min_area < area < max_area and
                    w > 20 and h > 50 and  # Lower minimum size
                    h > img_height * 0.2):  # More flexible height requirement
                    
                    # Check if near edges (doors usually at room perimeter) - more flexible
                    near_edge = (x < img_width * 0.2 or x + w > img_width * 0.8 or
                               y < img_height * 0.2 or y + h > img_height * 0.8)
                    
                    # Also consider doors that are reasonably positioned
                    reasonable_position = (x > img_width * 0.1 and x + w < img_width * 0.9 and
                                         y > img_height * 0.1 and y + h < img_height * 0.9)
                    
                    if near_edge or reasonable_position:
                        # Check brightness (doors are typically darker than windows)
                        roi = gray[y:y+h, x:x+w]
                        if roi.size > 0:
                            mean_brightness = np.mean(roi)
                            
                            # Doors are typically darker than windows
                            if 50 < mean_brightness < 180:  # Not too bright, not too dark
                                # Calculate confidence based on shape and brightness
                                shape_score = 1.0 - abs(aspect_ratio - 0.5) / 0.5  # Prefer ~0.5 aspect ratio
                                brightness_score = 1.0 - abs(mean_brightness - 120) / 120  # Prefer medium brightness
                                confidence = (shape_score * 0.6 + brightness_score * 0.4)
                                
                                detections.append({
                                    "type": "door",
                                    "bbox": [x, y, x + w, y + h],
                                    "confidence": confidence,
                                    "category": "architectural"
                                })
        
        return detections
    
    def _detect_walls(self, img_width: int, img_height: int) -> List[Dict]:
        """Detect walls as the image boundaries."""
        detections = []
        
        wall_thickness = 10  # Pixels
        
        # Top wall
        detections.append({
            "type": "wall",
            "bbox": [0, 0, img_width, wall_thickness],
            "confidence": 1.0,
            "category": "architectural"
        })
        
        # Bottom wall
        detections.append({
            "type": "wall",
            "bbox": [0, img_height - wall_thickness, img_width, img_height],
            "confidence": 1.0,
            "category": "architectural"
        })
        
        # Left wall
        detections.append({
            "type": "wall",
            "bbox": [0, 0, wall_thickness, img_height],
            "confidence": 1.0,
            "category": "architectural"
        })
        
        # Right wall
        detections.append({
            "type": "wall",
            "bbox": [img_width - wall_thickness, 0, img_width, img_height],
            "confidence": 1.0,
            "category": "architectural"
        })
        
        return detections


def enhance_detections_with_architecture(
    yolo_detections: List[Dict],
    image_path: str,
    force_architectural: bool = False
) -> List[Dict]:
    """
    Enhance YOLO detections with architectural elements if none were found.
    
    Args:
        yolo_detections: Detections from YOLO model
        image_path: Path to the room image
        force_architectural: Force architectural detection even if some found
        
    Returns:
        Combined detections list
    """
    # Check if any architectural elements were detected by YOLO
    has_architectural = any(
        d.get("category") == "architectural" or 
        d.get("type", "").lower() in ["window", "door", "wall"]
        for d in yolo_detections
    )
    
    if has_architectural and not force_architectural:
        print("[ArchitecturalDetector] YOLO found architectural elements, skipping fallback detection")
        return yolo_detections
    
    # Use fallback detection
    print("[ArchitecturalDetector] No architectural elements from YOLO, using image processing fallback")
    detector = ArchitecturalDetector()
    arch_detections = detector.detect_architectural_elements(image_path)
    
    # Combine detections
    combined = yolo_detections + arch_detections
    
    print(f"[ArchitecturalDetector] Total detections: {len(combined)} (YOLO: {len(yolo_detections)}, Architectural: {len(arch_detections)})")
    
    return combined
