"""
cnn_architectural_detector.py
----------------------------
CNN-based architectural element detection for enhanced accuracy.
Uses deep learning models to detect windows, doors, walls, and other fixed elements.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from PIL import Image
import os

class ArchitecturalCNN(nn.Module):
    """CNN model for detecting architectural elements in room images."""
    
    def __init__(self, num_classes=5):  # wall, door, window, floor, ceiling
        super(ArchitecturalCNN, self).__init__()
        
        # Encoder (feature extraction)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet-like blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        
        # Final classification
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Decoder
        x = self.upconv4(x)
        x = self.upconv3(x)
        x = self.upconv2(x)
        x = self.upconv1(x)
        
        # Final classification
        x = self.final_conv(x)
        x = self.softmax(x)
        
        return x

class CNNArchitecturalDetector:
    """CNN-based detector for architectural elements with enhanced accuracy."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.model = None
        self.class_names = ["wall", "door", "window", "floor", "ceiling"]
        self.load_model(model_path)
    
    def load_model(self, model_path: Optional[str] = None):
        """Load or create the CNN model."""
        if model_path and os.path.exists(model_path):
            try:
                self.model = torch.load(model_path, map_location=self.device)
                print(f"[CNN Detector] Loaded model from {model_path}")
            except Exception as e:
                print(f"[CNN Detector] Failed to load model: {e}")
                self.model = None
        else:
            # Create a new model (will need training in practice)
            self.model = ArchitecturalCNN(num_classes=len(self.class_names))
            print("[CNN Detector] Created new model (untrained)")
        
        if self.model:
            self.model.to(self.device)
            self.model.eval()
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for CNN input."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to standard input size
        target_size = (512, 512)
        image = cv2.resize(image, target_size)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def detect_architectural_elements(self, image_path: str, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect architectural elements using CNN.
        Returns list of detected elements with bounding boxes and confidence scores.
        """
        if self.model is None:
            print("[CNN Detector] No model loaded, falling back to traditional methods")
            return self._fallback_detection(image_path)
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_path)
            
            # Get predictions
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            # Convert predictions to numpy
            pred_np = predictions.squeeze(0).cpu().numpy()
            
            # Find regions with high confidence for each class
            detections = []
            original_image = cv2.imread(image_path)
            h, w = original_image.shape[:2]
            
            for class_idx, class_name in enumerate(self.class_names):
                class_map = pred_np[class_idx]
                
                # Find connected components with high confidence
                binary_map = (class_map > confidence_threshold).astype(np.uint8)
                contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Get bounding box
                    x, y, w_bbox, h_bbox = cv2.boundingRect(contour)
                    
                    # Scale back to original image size
                    x = int(x * w / 512)
                    y = int(y * h / 512)
                    w_bbox = int(w_bbox * w / 512)
                    h_bbox = int(h_bbox * h / 512)
                    
                    # Calculate confidence as max value in the region
                    region = class_map[y:y+h_bbox, x:x+w_bbox]
                    confidence = float(np.max(region))
                    
                    if confidence > confidence_threshold and w_bbox > 10 and h_bbox > 10:
                        detections.append({
                            "type": class_name,
                            "bbox": [x, y, x + w_bbox, y + h_bbox],
                            "confidence": confidence,
                            "category": "architectural",
                            "method": "cnn_detection"
                        })
            
            return detections
            
        except Exception as e:
            print(f"[CNN Detector] Error in CNN detection: {e}")
            return self._fallback_detection(image_path)
    
    def _fallback_detection(self, image_path: str) -> List[Dict]:
        """Fallback to traditional computer vision methods."""
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        detections = []
        h, w = image.shape[:2]
        
        # Simple edge-based detection for windows and doors
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find rectangular contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:  # Rectangular shape
                x, y, w_bbox, h_bbox = cv2.boundingRect(contour)
                
                # Filter by size and aspect ratio
                aspect_ratio = w_bbox / h_bbox if h_bbox > 0 else 0
                area = w_bbox * h_bbox
                
                if (area > w * h * 0.01 and  # At least 1% of image
                    area < w * h * 0.3 and   # At most 30% of image
                    0.2 < aspect_ratio < 5.0):  # Reasonable aspect ratio
                    
                    # Classify based on position and characteristics
                    center_y = y + h_bbox / 2
                    
                    if center_y < h * 0.3:  # Upper third - likely window
                        detections.append({
                            "type": "window",
                            "bbox": [x, y, x + w_bbox, y + h_bbox],
                            "confidence": 0.6,
                            "category": "architectural",
                            "method": "fallback_edge"
                        })
                    elif h_bbox > w_bbox * 1.5:  # Tall and narrow - likely door
                        detections.append({
                            "type": "door",
                            "bbox": [x, y, x + w_bbox, y + h_bbox],
                            "confidence": 0.5,
                            "category": "architectural",
                            "method": "fallback_edge"
                        })
        
        return detections
    
    def analyze_room_layout(self, image_path: str) -> Dict:
        """
        Analyze the overall room layout to understand constraints and opportunities.
        Returns analysis including wall positions, window/door locations, and layout recommendations.
        """
        detections = self.detect_architectural_elements(image_path)
        
        # Analyze room structure
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        # Find walls (edges of the room)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        
        # Detect horizontal and vertical lines (walls)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        walls = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Classify as horizontal or vertical wall
                if abs(y2 - y1) < abs(x2 - x1):  # Horizontal
                    walls.append({"type": "wall", "orientation": "horizontal", "y": (y1 + y2) / 2})
                else:  # Vertical
                    walls.append({"type": "wall", "orientation": "vertical", "x": (x1 + x2) / 2})
        
        # Analyze window and door positions
        windows = [d for d in detections if d["type"] == "window"]
        doors = [d for d in detections if d["type"] == "door"]
        
        # Calculate room metrics
        room_area = w * h
        window_area = sum((d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]) for d in windows)
        door_area = sum((d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]) for d in doors)
        
        # Layout recommendations
        recommendations = {
            "furniture_placement_zones": self._identify_furniture_zones(w, h, windows, doors),
            "traffic_flow": self._analyze_traffic_flow(windows, doors, w, h),
            "lighting_considerations": self._analyze_lighting(windows, w, h),
            "room_balance": self._analyze_room_balance(windows, doors, w, h)
        }
        
        return {
            "detections": detections,
            "walls": walls,
            "windows": windows,
            "doors": doors,
            "room_metrics": {
                "area": room_area,
                "window_area": window_area,
                "door_area": door_area,
                "window_to_room_ratio": window_area / room_area if room_area > 0 else 0
            },
            "recommendations": recommendations
        }
    
    def _identify_furniture_zones(self, w: int, h: int, windows: List[Dict], doors: List[Dict]) -> List[Dict]:
        """Identify optimal zones for furniture placement."""
        zones = []
        
        # Zone 1: Near windows (for seating, plants)
        for window in windows:
            x1, y1, x2, y2 = window["bbox"]
            zones.append({
                "type": "window_zone",
                "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                "size": (x2 - x1, y2 - y1),
                "suitable_for": ["chair", "plant", "table", "desk"],
                "priority": "high"
            })
        
        # Zone 2: Wall areas (for large furniture)
        wall_zones = [
            {"center": (w * 0.1, h / 2), "size": (w * 0.15, h * 0.8), "suitable_for": ["bed", "sofa", "wardrobe"]},
            {"center": (w * 0.9, h / 2), "size": (w * 0.15, h * 0.8), "suitable_for": ["bed", "sofa", "wardrobe"]},
            {"center": (w / 2, h * 0.1), "size": (w * 0.8, h * 0.15), "suitable_for": ["tv", "bookshelf"]},
            {"center": (w / 2, h * 0.9), "size": (w * 0.8, h * 0.15), "suitable_for": ["tv", "bookshelf"]}
        ]
        
        for zone in wall_zones:
            zones.append({
                "type": "wall_zone",
                "center": zone["center"],
                "size": zone["size"],
                "suitable_for": zone["suitable_for"],
                "priority": "medium"
            })
        
        # Zone 3: Center area (for coffee tables, ottomans)
        zones.append({
            "type": "center_zone",
            "center": (w / 2, h / 2),
            "size": (w * 0.4, h * 0.4),
            "suitable_for": ["coffee table", "ottoman", "rug"],
            "priority": "low"
        })
        
        return zones
    
    def _analyze_traffic_flow(self, windows: List[Dict], doors: List[Dict], w: int, h: int) -> Dict:
        """Analyze traffic flow patterns in the room."""
        # Identify main pathways
        pathways = []
        
        # Pathway from doors to center
        for door in doors:
            door_center = ((door["bbox"][0] + door["bbox"][2]) / 2, (door["bbox"][1] + door["bbox"][3]) / 2)
            pathways.append({
                "start": door_center,
                "end": (w / 2, h / 2),
                "width": 60,  # cm
                "importance": "high"
            })
        
        return {
            "pathways": pathways,
            "clearance_required": 60,  # cm minimum clearance
            "main_circulation": "center_to_doors"
        }
    
    def _analyze_lighting(self, windows: List[Dict], w: int, h: int) -> Dict:
        """Analyze lighting conditions and recommendations."""
        if not windows:
            return {"natural_light": "low", "recommendations": ["add_artificial_lighting"]}
        
        # Calculate total window area
        total_window_area = sum((w["bbox"][2] - w["bbox"][0]) * (w["bbox"][3] - w["bbox"][1]) for w in windows)
        room_area = w * h
        window_ratio = total_window_area / room_area if room_area > 0 else 0
        
        if window_ratio > 0.15:
            light_level = "high"
        elif window_ratio > 0.08:
            light_level = "medium"
        else:
            light_level = "low"
        
        return {
            "natural_light": light_level,
            "window_ratio": window_ratio,
            "recommendations": [
                "place_seating_near_windows" if light_level in ["medium", "high"] else "add_lighting",
                "consider_window_treatments" if light_level == "high" else "maximize_natural_light"
            ]
        }
    
    def _analyze_room_balance(self, windows: List[Dict], doors: List[Dict], w: int, h: int) -> Dict:
        """Analyze room balance and symmetry."""
        # Check for symmetry in window placement
        window_centers = [((w["bbox"][0] + w["bbox"][2]) / 2, (w["bbox"][1] + w["bbox"][3]) / 2) for w in windows]
        
        # Analyze distribution
        left_windows = [w for w in window_centers if w[0] < w / 2]
        right_windows = [w for w in window_centers if w[0] > w / 2]
        
        balance_score = 1.0 - abs(len(left_windows) - len(right_windows)) / max(1, len(window_centers))
        
        return {
            "balance_score": balance_score,
            "symmetry": "good" if balance_score > 0.7 else "needs_improvement",
            "recommendations": [
                "maintain_visual_balance" if balance_score > 0.7 else "consider_symmetrical_arrangement"
            ]
        }
