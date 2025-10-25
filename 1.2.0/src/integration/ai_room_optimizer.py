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
from PIL import Image
import io

# Import our custom modules
from src.input.cnn_architectural_detector import CNNArchitecturalDetector
from src.optimization.cnn_guided_optimizer import CNNGuidedOptimizer
from src.visualization.enhanced_layout_generator import EnhancedLayoutGenerator

class AIRoomOptimizer:
    """
    Main AI Room Optimizer that integrates CNN detection, genetic algorithms,
    and enhanced visualization for optimal furniture placement.
    """
    
    def __init__(self, room_dims: Tuple[float, float], device: str = "cpu"):
        self.room_dims = room_dims
        self.device = device
        
        # Initialize components
        self.architectural_detector = CNNArchitecturalDetector(device=device)
        self.layout_generator = EnhancedLayoutGenerator(room_dims)
        
        # Results storage
        self.room_analysis = None
        self.optimized_layout = None
        self.architectural_elements = []
        self.furniture_layout = []
    
    def analyze_room(self, image_path: str) -> Dict:
        """
        Analyze room using CNN to detect architectural elements and understand layout.
        """
        print("[AI Room Optimizer] Analyzing room with CNN...")
        
        # Detect architectural elements
        self.architectural_elements = self.architectural_detector.detect_architectural_elements(image_path)
        
        # Perform comprehensive room analysis
        self.room_analysis = self.architectural_detector.analyze_room_layout(image_path)
        
        print(f"[AI Room Optimizer] Detected {len(self.architectural_elements)} architectural elements")
        print(f"[AI Room Optimizer] Room analysis complete")
        
        return self.room_analysis
    
    def optimize_furniture_placement(self, furniture_list: List[Dict], 
                                    user_preferences: Dict = None) -> List[Dict]:
        """
        Optimize furniture placement using CNN-guided genetic algorithm.
        """
        if not self.room_analysis:
            raise ValueError("Room analysis must be performed first. Call analyze_room() before optimization.")
        
        print("[AI Room Optimizer] Starting CNN-guided optimization...")
        
        # Initialize CNN-guided optimizer
        optimizer = CNNGuidedOptimizer(
            room_dims=self.room_dims,
            objects=furniture_list,
            room_analysis=self.room_analysis,
            user_prefs=user_preferences,
            population_size=200,  # Increased for better results
            generations=300,      # Increased for convergence
            seed=42
        )
        
        # Run optimization
        self.furniture_layout = optimizer.optimize()
        
        print(f"[AI Room Optimizer] Optimization complete. Generated layout with {len(self.furniture_layout)} furniture items")
        
        return self.furniture_layout
    
    def generate_2d_layout(self, save_path: Optional[str] = None, 
                          save_buffer: Optional[io.BytesIO] = None) -> io.BytesIO:
        """
        Generate enhanced 2D layout showing architectural elements and optimized furniture.
        """
        if not self.architectural_elements and not self.furniture_layout:
            raise ValueError("No layout data available. Run analysis and optimization first.")
        
        print("[AI Room Optimizer] Generating enhanced 2D layout...")
        
        # Generate layout
        buffer = self.layout_generator.generate_layout(
            architectural_elements=self.architectural_elements,
            furniture_layout=self.furniture_layout,
            save_path=save_path,
            save_buffer=save_buffer
        )
        
        print("[AI Room Optimizer] 2D layout generation complete")
        
        return buffer
    
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
        furniture_area = sum(obj["w"] * obj["h"] for obj in optimized_furniture) / 10000  # m²
        space_utilization = (furniture_area / room_area) * 100 if room_area > 0 else 0
        
        # Calculate total furniture volume for better space analysis
        total_furniture_volume = sum(obj["w"] * obj["h"] * obj.get("height", 80) for obj in optimized_furniture) / 1000000  # m³
        
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
                "average_furniture_size_cm": sum(obj["w"] * obj["h"] for obj in optimized_furniture) / len(optimized_furniture) if optimized_furniture else 0
            },
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
        windows = [d for d in room_analysis.get("detections", []) if d["type"] == "window"]
        doors = [d for d in room_analysis.get("detections", []) if d["type"] == "door"]
        
        # Bonus for placing seating near windows
        for obj in furniture_layout:
            if obj.get("type", "").lower() in ["chair", "sofa", "desk"]:
                obj_center = (obj["x"] + obj["w"]/2, obj["y"] + obj["h"]/2)
                
                for window in windows:
                    wx1, wy1, wx2, wy2 = window["bbox"]
                    window_center = ((wx1 + wx2) / 2, (wy1 + wy2) / 2)
                    distance = np.sqrt((obj_center[0] - window_center[0])**2 + 
                                     (obj_center[1] - window_center[1])**2)
                    
                    if distance < 150:  # Within 1.5m
                        bonus += 30 * (1 - distance / 150)
        
        # Penalty for blocking doors
        for obj in furniture_layout:
            obj_center = (obj["x"] + obj["w"]/2, obj["y"] + obj["h"]/2)
            
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
        furniture_area = sum(obj["w"] * obj["h"] for obj in furniture_layout)
        room_area = self.room_dims[0] * self.room_dims[1]
        utilization = furniture_area / room_area if room_area > 0 else 0
        
        # Optimal utilization is around 30-50%
        if 0.3 <= utilization <= 0.5:
            return 100
        elif utilization < 0.3:
            return 50 * utilization / 0.3
        else:
            return max(0, 100 - (utilization - 0.5) * 200)
    
    def _generate_recommendations(self, room_analysis: Dict, furniture_layout: List[Dict]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Lighting recommendations
        lighting_analysis = room_analysis.get("recommendations", {}).get("lighting_considerations", {})
        if lighting_analysis.get("natural_light") == "low":
            recommendations.append("Consider adding more lighting fixtures near seating areas")
        
        # Space utilization recommendations
        furniture_area = sum(obj["w"] * obj["h"] for obj in furniture_layout)
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
        return not (a["x"] + a["w"] <= b["x"] or b["x"] + b["w"] <= a["x"] or 
                   a["y"] + a["h"] <= b["y"] or b["y"] + b["h"] <= a["y"])
    
    def _center_distance(self, a: Dict, b: Dict) -> float:
        """Calculate distance between centers of two objects."""
        ax = a["x"] + a["w"] / 2.0
        ay = a["y"] + a["h"] / 2.0
        bx = b["x"] + b["w"] / 2.0
        by = b["y"] + b["h"] / 2.0
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
