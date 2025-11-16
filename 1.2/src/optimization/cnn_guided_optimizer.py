"""
cnn_guided_optimizer.py
-----------------------
Enhanced genetic algorithm optimizer that uses CNN analysis to guide furniture placement.
Integrates architectural constraints and room analysis for better optimization.
"""

import random
import math
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from copy import deepcopy
import json

class CNNGuidanceSystem:
    """System that provides CNN-based guidance for furniture optimization."""
    
    def __init__(self, room_analysis: Dict):
        self.room_analysis = room_analysis
        self.architectural_elements = room_analysis.get("detections", [])
        self.furniture_zones = room_analysis.get("recommendations", {}).get("furniture_placement_zones", [])
        self.traffic_flow = room_analysis.get("recommendations", {}).get("traffic_flow", {})
        self.lighting_analysis = room_analysis.get("recommendations", {}).get("lighting_considerations", {})
        
    def get_placement_guidance(self, furniture_type: str) -> Dict:
        """Get CNN-guided placement recommendations for specific furniture."""
        guidance = {
            "preferred_zones": [],
            "avoid_zones": [],
            "orientation_hints": [],
            "size_constraints": {},
            "lighting_requirements": []
        }
        
        # Analyze furniture type and get recommendations
        furniture_lower = furniture_type.lower()
        
        # Find suitable zones for this furniture type
        for zone in self.furniture_zones:
            if furniture_lower in [f.lower() for f in zone.get("suitable_for", [])]:
                guidance["preferred_zones"].append(zone)
        
        # Lighting requirements
        if furniture_lower in ["desk", "table", "chair"]:
            if self.lighting_analysis.get("natural_light") in ["medium", "high"]:
                guidance["lighting_requirements"].append("near_window")
        
        # Size constraints based on room analysis
        room_metrics = self.room_analysis.get("room_metrics", {})
        room_area = room_metrics.get("area", 0)
        
        if room_area > 0:
            # Calculate maximum furniture size as percentage of room
            max_furniture_ratio = 0.3  # 30% of room area max for single piece
            max_area = room_area * max_furniture_ratio
            guidance["size_constraints"]["max_area"] = max_area
        
        return guidance
    
    def calculate_architectural_fitness(self, furniture: Dict, position: Tuple[float, float]) -> float:
        """Calculate fitness score based on architectural constraints."""
        x, y = position
        w, h = furniture.get("w", 0), furniture.get("h", 0)
        furniture_type = furniture.get("type", "").lower().replace("_", " ")
        
        score = 0.0
        def _dtype(det: Dict) -> str:
            return str(det.get("type") or det.get("label") or "").lower().replace("_", " ")
        
        # Check proximity to windows (good for seating, desks)
        if furniture_type in ["chair", "desk", "table", "dining table", "dining chair"]:
            for window in [d for d in self.architectural_elements if _dtype(d) == "window"]:
                wx1, wy1, wx2, wy2 = window["bbox"]
                window_center = ((wx1 + wx2) / 2, (wy1 + wy2) / 2)
                distance = math.sqrt((x + w/2 - window_center[0])**2 + (y + h/2 - window_center[1])**2)
                
                # Bonus for being near windows (within 100cm)
                if distance < 100:
                    score += 50 * (1 - distance / 100)
        
        # Check wall proximity (good for large furniture)
        if furniture_type in ["bed", "sofa", "wardrobe", "dresser"]:
            # Distance to nearest wall
            wall_distance = min(x, y, 400 - x - w, 300 - y - h)  # Assuming 400x300 room
            if wall_distance < 50:  # Close to wall
                score += 30
        
        # Avoid blocking doors (leave clear walking space near door)
        for door in [d for d in self.architectural_elements if _dtype(d) == "door"]:
            dx1, dy1, dx2, dy2 = door["bbox"]
            door_center = ((dx1 + dx2) / 2, (dy1 + dy2) / 2)
            distance_to_door = math.sqrt((x + w/2 - door_center[0])**2 + (y + h/2 - door_center[1])**2)
            
            # Penalty for blocking door access
            if distance_to_door < 80:  # Too close to door
                score -= 150

            # Stronger: keep-clear rectangle in front of door (walking space)
            clear_margin = 90  # cm
            expand = 40
            clear_rect = (max(0, dx1 - expand), max(0, dy1 - expand), dx2 + expand + clear_margin, dy2 + expand + clear_margin)
            fx1, fy1, fx2, fy2 = x, y, x + w, y + h
            if not (fx2 <= clear_rect[0] or fx1 >= clear_rect[2] or fy2 <= clear_rect[1] or fy1 >= clear_rect[3]):
                score -= 250
        
        # Traffic flow consideration
        clearance_required = self.traffic_flow.get("clearance_required", 60)
        for pathway in self.traffic_flow.get("pathways", []):
            # Check if furniture blocks pathway
            pathway_start = pathway["start"]
            pathway_end = pathway["end"]
            pathway_width = pathway.get("width", 60)
            
            # Simple line intersection check
            if self._blocks_pathway((x, y, w, h), pathway_start, pathway_end, pathway_width):
                score -= 200  # Heavy penalty for blocking traffic
        
        return score
    
    def _blocks_pathway(self, furniture_rect: Tuple, start: Tuple, end: Tuple, width: float) -> bool:
        """Check if furniture blocks a pathway."""
        fx, fy, fw, fh = furniture_rect
        
        # Create pathway corridor
        path_length = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        if path_length == 0:
            return False
        
        # Check if furniture intersects with pathway corridor
        # Simplified: check if furniture center is too close to pathway line
        furniture_center = (fx + fw/2, fy + fh/2)
        
        # Distance from point to line
        A = end[1] - start[1]
        B = start[0] - end[0]
        C = end[0] * start[1] - start[0] * end[1]
        
        distance = abs(A * furniture_center[0] + B * furniture_center[1] + C) / math.sqrt(A**2 + B**2)
        
        return distance < width / 2 + 30  # 30cm buffer

class CNNGuidedOptimizer:
    """Enhanced genetic algorithm optimizer with CNN guidance."""
    
    def __init__(self, room_dims: Tuple[float, float], objects: List[Dict], 
                 room_analysis: Dict, user_prefs: Dict = None,
                 population_size: int = 150, generations: int = 300, seed: int = None):
        self.room_dims = tuple(map(float, room_dims))
        self.base_objects = deepcopy(objects)
        self.room_analysis = room_analysis
        self.guidance_system = CNNGuidanceSystem(room_analysis)
        self.user_prefs = user_prefs or {}
        self.population_size = max(10, int(population_size))
        self.generations = max(1, int(generations))
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.diagnostics: List[Dict] = []
    
    def cnn_guided_fitness(self, layout: List[Dict]) -> float:
        """
        Enhanced fitness function that incorporates CNN analysis.
        Combines traditional optimization with architectural intelligence.
        """
        base_score = 10000.0
        
        # Traditional constraints (overlaps, bounds)
        num_overlaps = 0
        num_out_of_bounds = 0
        
        for obj in layout:
            if not self._in_bounds(obj):
                num_out_of_bounds += 1
                base_score -= 5000.0
        
        for i in range(len(layout)):
            for j in range(i + 1, len(layout)):
                if self._overlap(layout[i], layout[j]):
                    num_overlaps += 1
                    overlap_area = self._overlap_area(layout[i], layout[j])
                    base_score -= 10000.0 * (1 + overlap_area / 1000.0)
        
        if num_overlaps > 0:
            base_score -= 50000.0
        
        # CNN-guided architectural fitness
        if num_overlaps == 0 and num_out_of_bounds == 0:
            architectural_score = 0.0
            
            for obj in layout:
                position = (obj["x"], obj["y"])
                arch_fitness = self.guidance_system.calculate_architectural_fitness(obj, position)
                architectural_score += arch_fitness
            
            # Room balance and harmony
            balance_score = self._calculate_room_balance(layout)
            lighting_score = self._calculate_lighting_optimization(layout)
            traffic_score = self._calculate_traffic_flow_score(layout)
            
            base_score += architectural_score + balance_score + lighting_score + traffic_score
        
        # Furniture grouping and relationships
        grouping_score = self._calculate_furniture_grouping(layout)
        base_score += grouping_score
        hard_penalty = self._hard_constraint_penalty(layout)
        base_score -= hard_penalty
        # Additional rule penalties (walkways, window blocking, clearances)
        rule_pen = self._rule_penalties(layout)
        base_score -= rule_pen
        
        return base_score
    
    def _calculate_room_balance(self, layout: List[Dict]) -> float:
        """Calculate room balance score based on furniture distribution."""
        if not layout:
            return 0.0
        
        # Calculate center of mass
        total_area = sum(obj["w"] * obj["h"] for obj in layout)
        if total_area == 0:
            return 0.0
        
        center_x = sum(obj["x"] * obj["w"] * obj["h"] for obj in layout) / total_area
        center_y = sum(obj["y"] * obj["w"] * obj["h"] for obj in layout) / total_area
        
        room_center_x = self.room_dims[0] / 2
        room_center_y = self.room_dims[1] / 2
        
        # Distance from room center (closer is better for balance)
        distance_from_center = math.sqrt((center_x - room_center_x)**2 + (center_y - room_center_y)**2)
        max_distance = math.sqrt(room_center_x**2 + room_center_y**2)
        
        balance_score = 100 * (1 - distance_from_center / max_distance)
        return balance_score
    
    def _calculate_lighting_optimization(self, layout: List[Dict]) -> float:
        """Optimize furniture placement for lighting conditions."""
        score = 0.0
        def _dtype(det: Dict) -> str:
            return str(det.get("type") or det.get("label") or "").lower().replace("_", " ")
        windows = [d for d in self.room_analysis.get("detections", []) if _dtype(d) == "window"]
        
        if not windows:
            return 0.0
        
        for obj in layout:
            furniture_type = obj.get("type", "").lower().replace("_", " ")
            if furniture_type in ["desk", "chair", "table"]:
                # Check proximity to windows
                obj_center = (obj["x"] + obj["w"]/2, obj["y"] + obj["h"]/2)
                
                for window in windows:
                    wx1, wy1, wx2, wy2 = window["bbox"]
                    window_center = ((wx1 + wx2) / 2, (wy1 + wy2) / 2)
                    distance = math.sqrt((obj_center[0] - window_center[0])**2 + 
                                        (obj_center[1] - window_center[1])**2)
                    
                    if distance < 150:  # Within 1.5m of window
                        score += 50 * (1 - distance / 150)
        
        return score
    
    def _calculate_traffic_flow_score(self, layout: List[Dict]) -> float:
        """Calculate score based on traffic flow optimization."""
        score = 0.0
        pathways = self.room_analysis.get("recommendations", {}).get("traffic_flow", {}).get("pathways", [])
        
        for pathway in pathways:
            pathway_clear = True
            pathway_start = pathway["start"]
            pathway_end = pathway["end"]
            pathway_width = pathway.get("width", 60)
            
            for obj in layout:
                if self.guidance_system._blocks_pathway(
                    (obj["x"], obj["y"], obj["w"], obj["h"]), 
                    pathway_start, pathway_end, pathway_width
                ):
                    pathway_clear = False
                    break
            
            if pathway_clear:
                score += 100  # Bonus for clear pathways
        
        return score
    
    def _calculate_furniture_grouping(self, layout: List[Dict]) -> float:
        """Calculate score for appropriate furniture grouping."""
        score = 0.0
        
        # Define furniture relationships
        relationships = {
            "sofa": ["coffee table", "tv", "lamp"],
            "bed": ["nightstand", "dresser", "wardrobe"],
            "desk": ["chair", "lamp"],
            "dining table": ["chair"],
            "tv": ["tv stand", "sofa", "chair"]
        }
        
        for obj1 in layout:
            obj1_type = obj1.get("type", "").lower().replace("_", " ")
            related_types = relationships.get(obj1_type, [])
            
            for obj2 in layout:
                if obj1 == obj2:
                    continue
                
                obj2_type = obj2.get("type", "").lower().replace("_", " ")
                if obj2_type in related_types:
                    distance = self._center_distance(obj1, obj2)
                    if distance < 200:  # Within 2m
                        score += 30 * (1 - distance / 200)
        
        return score

    def _min_wall_distance(self, obj: Dict) -> float:
        x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
        return min(x, y, self.room_dims[0] - (x + w), self.room_dims[1] - (y + h))

    def _center(self, obj: Dict) -> Tuple[float, float]:
        return (obj["x"] + obj["w"] / 2.0, obj["y"] + obj["h"] / 2.0)

    def _hard_constraint_penalty(self, layout: List[Dict]) -> float:
        penalty = 0.0
        if not layout:
            return penalty
        def T(o):
            return o.get("type", "").lower().replace("_", " ")
        def _dtype(det: Dict) -> str:
            return str(det.get("type") or det.get("label") or "").lower().replace("_", " ")
        sofas = [o for o in layout if ("sofa" in T(o) or "couch" in T(o))]
        tvs = [o for o in layout if "tv" in T(o)]
        mirrors = [o for o in layout if "mirror" in T(o)]
        coffee_tables = [o for o in layout if "coffee table" in T(o)]
        dining_tables = [o for o in layout if "dining table" in T(o)]
        generic_tables = [o for o in layout if ("table" in T(o) and o not in dining_tables and o not in coffee_tables)]
        desks = [o for o in layout if "desk" in T(o)]
        beds = [o for o in layout if "bed" in T(o)]
        nightstands = [o for o in layout if ("nightstand" in T(o) or "bedside table" in T(o))]
        fridges = [o for o in layout if ("refrigerator" in T(o) or "fridge" in T(o))]
        tv_stands = [o for o in layout if "tv stand" in T(o)]
        office_chairs = [o for o in layout if "office chair" in T(o)]
        dining_chairs = [o for o in layout if "dining chair" in T(o)]
        any_chairs = [o for o in layout if "chair" in T(o)]

        windows = [d for d in self.room_analysis.get("detections", []) if _dtype(d) == "window"]
        def _nearest_window_dist(cx: float, cy: float) -> float:
            if not windows:
                return float("inf")
            dmin = float("inf")
            for w in windows:
                x1, y1, x2, y2 = w.get("bbox", (0, 0, 0, 0))
                wx, wy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                d = math.hypot(cx - wx, cy - wy)
                if d < dmin:
                    dmin = d
            return dmin

        if sofas and tvs:
            for s in sofas:
                sc = self._center(s)
                nearest_tv = min(tvs, key=lambda t: math.hypot(self._center(t)[0] - sc[0], self._center(t)[1] - sc[1]))
                tc = self._center(nearest_tv)
                d = math.hypot(sc[0] - tc[0], sc[1] - tc[1])
                target_min, target_max = 150.0, 300.0
                if d < target_min:
                    penalty += (target_min - d) * 50.0
                elif d > target_max:
                    penalty += (d - target_max) * 20.0
                align_err = min(abs(sc[0] - tc[0]), abs(sc[1] - tc[1]))
                if align_err > 60.0:
                    penalty += (align_err - 60.0) * 10.0

                # Avoid mirrors in the viewing corridor between sofa and TV
                if mirrors:
                    # corridor width in cm (approx line-of-sight buffer)
                    corridor_w = 60.0
                    def point_segment_distance(px, py, ax, ay, bx, by):
                        # distance from P to segment AB
                        vx, vy = bx - ax, by - ay
                        wx, wy = px - ax, py - ay
                        c1 = vx * wx + vy * wy
                        if c1 <= 0:
                            return math.hypot(px - ax, py - ay), 0.0
                        c2 = vx * vx + vy * vy
                        if c2 <= 1e-6:
                            return math.hypot(px - ax, py - ay), 0.0
                        t = max(0.0, min(1.0, c1 / c2))
                        projx, projy = ax + t * vx, ay + t * vy
                        return math.hypot(px - projx, py - projy), t
                    ax, ay = sc
                    bx, by = tc
                    for m in mirrors:
                        mx, my = self._center(m)
                        dist_line, tpos = point_segment_distance(mx, my, ax, ay, bx, by)
                        if 0.0 <= tpos <= 1.0 and dist_line < corridor_w/2.0:
                            # Heavy penalty: mirror in viewing corridor
                            penalty += 40000.0 + (corridor_w/2.0 - dist_line) * 500.0

        for t in tvs + tv_stands:
            if self._min_wall_distance(t) > 20.0:
                penalty += 6000.0

        for s in sofas:
            if coffee_tables:
                ct = min(coffee_tables, key=lambda c: math.hypot(self._center(c)[0] - self._center(s)[0], self._center(c)[1] - self._center(s)[1]))
                d = math.hypot(self._center(ct)[0] - self._center(s)[0], self._center(ct)[1] - self._center(s)[1])
                if d > 120.0:
                    penalty += (d - 120.0) * 15.0

        for b in beds:
            if self._min_wall_distance(b) > 30.0:
                penalty += 8000.0
            if nightstands:
                ns_d = min(math.hypot(self._center(n)[0] - self._center(b)[0], self._center(n)[1] - self._center(b)[1]) for n in nightstands) if nightstands else float("inf")
                if ns_d > 100.0:
                    penalty += (ns_d - 100.0) * 20.0

        for f in fridges:
            if self._min_wall_distance(f) > 20.0:
                penalty += 6000.0

        if dining_tables:
            target_tables = dining_tables
        else:
            target_tables = generic_tables + desks
        if target_tables:
            for ch in dining_chairs if dining_chairs else [o for o in any_chairs if "office" not in o.get("type", "").lower()]:
                tbl = min(target_tables, key=lambda t: math.hypot(self._center(t)[0] - self._center(ch)[0], self._center(t)[1] - self._center(ch)[1]))
                d = math.hypot(self._center(tbl)[0] - self._center(ch)[0], self._center(tbl)[1] - self._center(ch)[1])
                if d > 120.0:
                    penalty += (d - 120.0) * 15.0

        for dsk in desks:
            cx, cy = self._center(dsk)
            wd = _nearest_window_dist(cx, cy)
            if wd > 150.0:
                penalty += (wd - 150.0) * 10.0

        for oc in office_chairs:
            if desks:
                dsk = min(desks, key=lambda d1: math.hypot(self._center(d1)[0] - self._center(oc)[0], self._center(d1)[1] - self._center(oc)[1]))
                d = math.hypot(self._center(dsk)[0] - self._center(oc)[0], self._center(dsk)[1] - self._center(oc)[1])
                if d > 80.0:
                    penalty += (d - 80.0) * 20.0

        # Sofa under the fan: encourage proximity to ceiling fan if present
        fans = [o for o in layout if ("ceiling fan" in T(o) or ("fan" in T(o) and "ceiling" in T(o)))]
        if fans and sofas:
            fan_c = self._center(fans[0])
            for s in sofas:
                sc = self._center(s)
                d = math.hypot(sc[0] - fan_c[0], sc[1] - fan_c[1])
                if d > 120.0:
                    penalty += (d - 120.0) * 10.0

        # Dressing table near lighting/window: treat dresser as dressing table
        lights = [o for o in layout if any(k in T(o) for k in ["lamp", "chandelier", "pendant light", "wall sconce"])]
        dressers = [o for o in layout if any(k in T(o) for k in ["dresser", "dressing table"])]
        if dressers:
            for dr in dressers:
                cx, cy = self._center(dr)
                wd = _nearest_window_dist(cx, cy)
                # nearest light
                ld = min((math.hypot(self._center(l)[0] - cx, self._center(l)[1] - cy) for l in lights), default=float("inf"))
                near_any = min(wd, ld)
                if near_any > 150.0:
                    penalty += (near_any - 150.0) * 8.0

        # Door swing/keep-clear hard constraint
        doors = [d for d in self.room_analysis.get("detections", []) if _dtype(d) == "door"]
        for door in doors:
            dx1, dy1, dx2, dy2 = door.get("bbox", (0, 0, 0, 0))
            px_w = abs(dx2 - dx1)
            px_h = abs(dy2 - dy1)
            swing_depth = max(40.0, float(min(px_w, px_h)))
            expand = 20.0
            clear_rect = (
                min(dx1, dx2) - expand,
                min(dy1, dy2) - expand,
                max(dx1, dx2) + expand + swing_depth,
                max(dy1, dy2) + expand + swing_depth,
            )
            for o in layout:
                if o.get("fixed") or o.get("architectural"):
                    continue
                fx1, fy1 = o["x"], o["y"]
                fx2, fy2 = o["x"] + o["w"], o["y"] + o["h"]
                if not (fx2 <= clear_rect[0] or fx1 >= clear_rect[2] or fy2 <= clear_rect[1] or fy1 >= clear_rect[3]):
                    penalty += 50000.0

        return penalty

    def _rule_penalties(self, layout: List[Dict]) -> float:
        """Soft and medium penalties computed via occupancy grid and overlaps.
        - Walkway width from each door to room center
        - Window blocking percent
        - Furniture-furniture clearance (basic dilation overlap)
        """
        if not layout:
            return 0.0
        penalty = 0.0
        room_w, room_h = self.room_dims
        # Occupancy grid at 5 cm resolution
        res = 5.0
        gw, gh = int(max(1, round(room_w / res))), int(max(1, round(room_h / res)))
        occ = np.zeros((gh, gw), dtype=np.uint8)
        def to_grid_rect(o):
            x1 = int(max(0, math.floor(o["x"] / res)))
            y1 = int(max(0, math.floor(o["y"] / res)))
            x2 = int(min(gw - 1, math.ceil((o["x"] + o["w"]) / res)))
            y2 = int(min(gh - 1, math.ceil((o["y"] + o["h"]) / res)))
            return x1, y1, x2, y2
        # Rasterize movable furniture only
        movables = [o for o in layout if not o.get("fixed") and not o.get("architectural")]
        for o in movables:
            x1, y1, x2, y2 = to_grid_rect(o)
            occ[y1:y2, x1:x2] = 255
        # Distance transform to compute clearance (in grid cells)
        inv = (occ == 0).astype(np.uint8) * 255
        dist = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)
        dist_cm = dist * res
        # Walkway: door -> room center minimum clearance along straight ray samples
        doors = [d for d in self.room_analysis.get("detections", []) if d.get("type") == "door"]
        cx, cy = room_w / 2.0, room_h / 2.0
        walkway_min = 70.0  # cm
        for door in doors:
            dx1, dy1, dx2, dy2 = door.get("bbox", (0, 0, 0, 0))
            dxc, dyc = (dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0
            # sample N points from door center to room center
            N = 100
            min_clear = float("inf")
            for t in np.linspace(0.0, 1.0, N):
                px = dxc * (1 - t) + cx * t
                py = dyc * (1 - t) + cy * t
                gx = int(min(gw - 1, max(0, round(px / res))))
                gy = int(min(gh - 1, max(0, round(py / res))))
                min_clear = min(min_clear, float(dist_cm[gy, gx]))
            if min_clear < walkway_min:
                penalty += (walkway_min - min_clear) * 100.0
        # Window blocking percent
        windows = [d for d in self.room_analysis.get("detections", []) if d.get("type") == "window"]
        def rect_overlap(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            return (ix2 - ix1) * (iy2 - iy1)
        for wdet in windows:
            wb = wdet.get("bbox", (0, 0, 0, 0))
            warea = max(0.0, (wb[2] - wb[0]) * (wb[3] - wb[1]))
            if warea <= 0:
                continue
            blocked = 0.0
            for o in movables:
                ob = (o["x"], o["y"], o["x"] + o["w"], o["y"] + o["h"])
                blocked += rect_overlap(ob, wb)
            pct = 100.0 * blocked / warea if warea > 0 else 0.0
            if pct > 25.0:
                penalty += (pct - 25.0) * 200.0
        # Basic clearance between furniture via dilation (avoid tight gaps)
        clearance_req = 40.0  # cm
        kern = int(max(1, round(clearance_req / res)))
        if kern % 2 == 0:
            kern += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern, kern))
        dil = cv2.dilate(occ, kernel)
        # if any two items overlap after dilation, add penalty proportional to overlap area
        overlap_area = int(np.sum((dil == 255))) - int(np.sum((occ == 255)))
        if overlap_area > 0:
            penalty += overlap_area * (res * res)  # scale by cm^2
        return penalty
    
    def cnn_guided_mutation(self, layout: List[Dict], mutation_rate: float = 0.3) -> List[Dict]:
        """Enhanced mutation that considers CNN guidance."""
        new_layout = deepcopy(layout)
        
        for i, obj in enumerate(new_layout):
            if random.random() < mutation_rate:
                furniture_type = obj.get("type", "").lower()
                
                # Get CNN guidance for this furniture type
                guidance = self.guidance_system.get_placement_guidance(furniture_type)
                
                # Try to place in preferred zones
                if guidance["preferred_zones"]:
                    zone = random.choice(guidance["preferred_zones"])
                    zone_center = zone["center"]
                    zone_size = zone["size"]
                    
                    # Place within zone with some randomness
                    new_x = zone_center[0] + random.gauss(0, zone_size[0] * 0.2)
                    new_y = zone_center[1] + random.gauss(0, zone_size[1] * 0.2)
                    
                    # Ensure bounds
                    new_x = max(0, min(self.room_dims[0] - obj["w"], new_x))
                    new_y = max(0, min(self.room_dims[1] - obj["h"], new_y))
                    
                    obj["x"] = new_x
                    obj["y"] = new_y
                else:
                    # Fallback to traditional mutation
                    dx = random.gauss(0, self.room_dims[0] * 0.05)
                    dy = random.gauss(0, self.room_dims[1] * 0.05)
                    obj["x"] = max(0, min(self.room_dims[0] - obj["w"], obj["x"] + dx))
                    obj["y"] = max(0, min(self.room_dims[1] - obj["h"], obj["y"] + dy))
        
        # Repair overlaps
        if self._has_overlaps(new_layout):
            new_layout = self._repair_layout(new_layout)
        
        return new_layout
    
    def cnn_guided_crossover(self, parent1: List[Dict], parent2: List[Dict]) -> List[Dict]:
        """Enhanced crossover that considers furniture relationships."""
        child = []
        
        # Group furniture by relationships
        furniture_groups = self._group_related_furniture(parent1)
        
        for group_name, group_indices in furniture_groups.items():
            if random.random() < 0.5:
                # Take entire group from parent1
                for idx in group_indices:
                    if idx < len(parent1):
                        child.append(deepcopy(parent1[idx]))
            else:
                # Take entire group from parent2
                for idx in group_indices:
                    if idx < len(parent2):
                        child.append(deepcopy(parent2[idx]))
        
        # Fill remaining slots
        while len(child) < len(parent1):
            if len(child) < len(parent1):
                child.append(deepcopy(parent1[len(child)]))
            if len(child) < len(parent2):
                child.append(deepcopy(parent2[len(child)]))
        
        return child[:len(parent1)]
    
    def _group_related_furniture(self, layout: List[Dict]) -> Dict:
        """Group furniture by relationships."""
        groups = {}
        
        for i, obj in enumerate(layout):
            obj_type = obj.get("type", "").lower()
            
            # Define furniture groups
            if obj_type in ["sofa", "coffee table", "tv", "tv stand"]:
                group = "living_room"
            elif obj_type in ["bed", "nightstand", "dresser", "wardrobe"]:
                group = "bedroom"
            elif obj_type in ["desk", "chair", "lamp"]:
                group = "office"
            elif obj_type in ["dining table", "chair"]:
                group = "dining"
            else:
                group = "other"
            
            if group not in groups:
                groups[group] = []
            groups[group].append(i)
        
        return groups
    
    def optimize(self) -> List[Dict]:
        """Main optimization loop with CNN guidance."""
        # Safety check: ensure we have furniture to place
        if not self.base_objects:
            print("[WARNING] No furniture objects provided. Returning empty layout with architectural elements only.")
            return self._add_architectural_elements([])
        
        # Initialize population with CNN guidance
        population = []
        for _ in range(self.population_size):
            layout = self._cnn_guided_initialization()
            if layout:  # Only add non-empty layouts
                population.append(layout)
        
        # Safety check: ensure we have at least one valid layout
        if not population:
            print("[ERROR] Failed to initialize any valid layouts. Creating fallback layout.")
            # Create a simple fallback layout with minimal placement
            fallback_layout = []
            for obj in self.base_objects:
                test_obj = dict(obj)
                test_obj.update({
                    "x": random.uniform(0, max(1.0, self.room_dims[0] - obj["w"])),
                    "y": random.uniform(0, max(1.0, self.room_dims[1] - obj["h"]))
                })
                fallback_layout.append(test_obj)
            population = [fallback_layout]
        
        best_so_far = deepcopy(population[0])  # Initialize with first layout
        best_score = self.cnn_guided_fitness(best_so_far)
        
        for generation in range(self.generations):
            # Calculate fitness scores
            scores = [self.cnn_guided_fitness(ind) for ind in population]
            
            # Track best
            gen_best_idx = int(np.argmax(scores))
            if scores[gen_best_idx] > best_score:
                best_score = scores[gen_best_idx]
                best_so_far = deepcopy(population[gen_best_idx])
            
            # Elitism: keep top 3
            idx_sorted = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            elites = [deepcopy(population[i]) for i in idx_sorted[:3]]
            
            # Selection and reproduction
            new_population = elites.copy()
            
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_select(population, scores, k=3)
                parent2 = self._tournament_select(population, scores, k=3)
                
                # Crossover
                child = self.cnn_guided_crossover(parent1, parent2)
                
                # Mutation
                child = self.cnn_guided_mutation(child, mutation_rate=0.3)
                
                new_population.append(child)
            
            population = new_population
        
        # Safety check before final cleanup
        if not best_so_far:
            print("[ERROR] No valid solution found. Returning fallback layout.")
            best_so_far = []
            for obj in self.base_objects:
                test_obj = dict(obj)
                test_obj.update({
                    "x": random.uniform(0, max(1.0, self.room_dims[0] - obj["w"])),
                    "y": random.uniform(0, max(1.0, self.room_dims[1] - obj["h"]))
                })
                best_so_far.append(test_obj)
        
        # Final cleanup
        for obj in best_so_far:
            obj["x"] = float(max(0.0, min(self.room_dims[0] - obj["w"], obj["x"])))
            obj["y"] = float(max(0.0, min(self.room_dims[1] - obj["h"], obj["y"])))
        
        # Final repair
        if self._has_overlaps(best_so_far):
            best_so_far = self._repair_layout(best_so_far)
        
        # Include architectural elements in output
        best_so_far = self._add_architectural_elements(best_so_far)
        # Collect diagnostics for final layout
        try:
            self.diagnostics = self._collect_diagnostics(best_so_far)
        except Exception:
            self.diagnostics = []
        
        return best_so_far
    
    def optimize_multiple(self, count: int = 3) -> List[Dict]:
        """Generate multiple alternative layouts for user selection."""
        results = []
        base_seed = self.seed if self.seed is not None else random.randint(0, 1000000)
        
        for i in range(count):
            seed_i = base_seed + i + 1
            random.seed(seed_i)
            np.random.seed(seed_i)
            
            layout = self.optimize()
            score = float(self.cnn_guided_fitness(layout))
            
            results.append({
                "layout": layout,
                "score": score,
                "seed": seed_i
            })
            
            print(f"[CNNGuidedOptimizer] Generated layout {i+1}/{count} (score: {score:.2f})")
        
        results.sort(key=lambda r: r["score"], reverse=True)
        return results
    
    def _add_architectural_elements(self, layout: List[Dict]) -> List[Dict]:
        """Add fixed architectural elements (windows, doors, walls) to layout."""
        enhanced_layout = deepcopy(layout)
        
        for detection in self.room_analysis.get("detections", []):
            det_type = detection.get("type", "").lower()
            
            # Only add architectural elements (immovable)
            if det_type in ["window", "door", "wall"]:
                bbox = detection.get("bbox", [0, 0, 0, 0])
                x1, y1, x2, y2 = bbox
                
                # Convert to room coordinates
                arch_element = {
                    "type": det_type,
                    "x": float(x1),
                    "y": float(y1),
                    "w": float(x2 - x1),
                    "h": float(y2 - y1),
                    "confidence": detection.get("confidence", 1.0),
                    "fixed": True,  # Mark as immovable
                    "architectural": True
                }
                enhanced_layout.append(arch_element)
        
        return enhanced_layout
    
    def _cnn_guided_initialization(self) -> List[Dict]:
        """Initialize layout using CNN guidance. ENSURES ALL FURNITURE IS PLACED."""
        layout = []
        placed_objects = set()
        
        # Sort objects by importance (size and type)
        sorted_objects = sorted(enumerate(self.base_objects), 
                               key=lambda x: x[1]["w"] * x[1]["h"], reverse=True)
        
        for orig_idx, obj in sorted_objects:
            if orig_idx in placed_objects:
                continue
            
            furniture_type = obj.get("type", "").lower().replace("_", " ")
            guidance = self.guidance_system.get_placement_guidance(furniture_type)
            
            # Try to place in preferred zones first
            placed = False
            for zone in guidance["preferred_zones"]:
                if self._try_place_in_zone(obj, zone, layout):
                    layout.append(obj)
                    placed_objects.add(orig_idx)
                    placed = True
                    break
            
            # Fallback to random placement with more attempts
            if not placed:
                for attempt in range(100):
                    x = random.uniform(0, max(1.0, self.room_dims[0] - obj["w"]))
                    y = random.uniform(0, max(1.0, self.room_dims[1] - obj["h"]))
                    
                    test_obj = dict(obj)
                    test_obj.update({"x": x, "y": y})
                    
                    if not any(self._overlap(test_obj, existing) for existing in layout):
                        layout.append(test_obj)
                        placed_objects.add(orig_idx)
                        placed = True
                        break
            
            # CRITICAL: Check if furniture is too large for room before forcing placement
            if not placed:
                furniture_area = obj["w"] * obj["h"]
                room_area = self.room_dims[0] * self.room_dims[1]
                
                # If furniture takes up more than 60% of room, skip it
                if furniture_area > room_area * 0.6:
                    print(f"[WARNING] Skipping {obj.get('type', 'unknown')} - too large for room ({furniture_area:.0f}cm² vs {room_area:.0f}cm² room)")
                    continue
                
                # Force placement with overlap (will be repaired by genetic algorithm)
                print(f"[WARNING] Could not place {obj.get('type', 'unknown')} without overlap. Forcing placement.")
                test_obj = dict(obj)
                test_obj.update({
                    "x": random.uniform(0, max(1.0, self.room_dims[0] - obj["w"])),
                    "y": random.uniform(0, max(1.0, self.room_dims[1] - obj["h"]))
                })
                layout.append(test_obj)
                placed_objects.add(orig_idx)
        
        # Verify all objects were placed
        if len(layout) != len(self.base_objects):
            print(f"[ERROR] Only placed {len(layout)}/{len(self.base_objects)} objects!")
        
        return layout
    
    def _try_place_in_zone(self, obj: Dict, zone: Dict, existing_layout: List[Dict]) -> bool:
        """Try to place object in a specific zone."""
        zone_center = zone["center"]
        zone_size = zone["size"]
        
        # Calculate placement area within zone
        x_min = max(0, zone_center[0] - zone_size[0] / 2)
        x_max = min(self.room_dims[0] - obj["w"], zone_center[0] + zone_size[0] / 2)
        y_min = max(0, zone_center[1] - zone_size[1] / 2)
        y_max = min(self.room_dims[1] - obj["h"], zone_center[1] + zone_size[1] / 2)
        
        if x_max <= x_min or y_max <= y_min:
            return False
        
        # Try multiple positions within zone
        for _ in range(20):
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            
            test_obj = dict(obj)
            test_obj.update({"x": x, "y": y})
            
            if not any(self._overlap(test_obj, existing) for existing in existing_layout):
                obj.update({"x": x, "y": y})
                return True
        
        return False
    
    # Helper methods (same as original optimizer)
    def _overlap(self, a, b):
        return not (a["x"] + a["w"] <= b["x"] or b["x"] + b["w"] <= a["x"] or 
                   a["y"] + a["h"] <= b["y"] or b["y"] + b["h"] <= a["y"])
    
    def _overlap_area(self, a, b):
        if not self._overlap(a, b):
            return 0.0
        overlap_x = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
        overlap_y = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
        return overlap_x * overlap_y
    
    def _in_bounds(self, obj):
        return (0 <= obj["x"] <= self.room_dims[0] - obj["w"] and 
                0 <= obj["y"] <= self.room_dims[1] - obj["h"])
    
    def _center_distance(self, a, b):
        ax = a["x"] + a["w"] / 2.0
        ay = a["y"] + a["h"] / 2.0
        bx = b["x"] + b["w"] / 2.0
        by = b["y"] + b["h"] / 2.0
        return math.hypot(ax - bx, ay - by)
    
    def _has_overlaps(self, layout):
        for i in range(len(layout)):
            for j in range(i + 1, len(layout)):
                if self._overlap(layout[i], layout[j]):
                    return True
        return False
    
    def _repair_layout(self, layout):
        """Repair layout by moving overlapping objects."""
        repaired = deepcopy(layout)
        max_attempts = 100
        
        for attempt in range(max_attempts):
            if not self._has_overlaps(repaired):
                break
            
            for i in range(len(repaired)):
                for j in range(i + 1, len(repaired)):
                    if self._overlap(repaired[i], repaired[j]):
                        # Move objects apart
                        dx = random.uniform(-50, 50)
                        dy = random.uniform(-50, 50)
                        
                        repaired[i]["x"] = max(0, min(self.room_dims[0] - repaired[i]["w"], 
                                                    repaired[i]["x"] + dx))
                        repaired[i]["y"] = max(0, min(self.room_dims[1] - repaired[i]["h"], 
                                                    repaired[i]["y"] + dy))
                        
                        repaired[j]["x"] = max(0, min(self.room_dims[0] - repaired[j]["w"], 
                                                    repaired[j]["x"] - dx))
                        repaired[j]["y"] = max(0, min(self.room_dims[1] - repaired[j]["h"], 
                                                    repaired[j]["y"] - dy))
        
        return repaired
    
    def _tournament_select(self, population, scores, k=3):
        aspirants = random.sample(list(range(len(population))), k)
        best_idx = max(aspirants, key=lambda idx: scores[idx])
        return deepcopy(population[best_idx])

    def _collect_diagnostics(self, layout: List[Dict]) -> List[Dict]:
        """Summarize key violations/metrics for overlay rendering."""
        diags: List[Dict] = []
        room_w, room_h = self.room_dims
        # Door swing intersections
        doors = [d for d in self.room_analysis.get("detections", []) if d.get("type") == "door"]
        for door in doors:
            dx1, dy1, dx2, dy2 = door.get("bbox", (0, 0, 0, 0))
            px_w = abs(dx2 - dx1)
            px_h = abs(dy2 - dy1)
            swing_depth = max(40.0, float(min(px_w, px_h)))
            expand = 20.0
            clear_rect = (
                min(dx1, dx2) - expand,
                min(dy1, dy2) - expand,
                max(dx1, dx2) + expand + swing_depth,
                max(dy1, dy2) + expand + swing_depth,
            )
            for o in layout:
                if o.get("fixed") or o.get("architectural"):
                    continue
                fx1, fy1 = o["x"], o["y"]
                fx2, fy2 = o["x"] + o["w"], o["y"] + o["h"]
                if not (fx2 <= clear_rect[0] or fx1 >= clear_rect[2] or fy2 <= clear_rect[1] or fy1 >= clear_rect[3]):
                    diags.append({
                        "kind": "door_swing_intersection",
                        "door_bbox": [dx1, dy1, dx2, dy2],
                        "clear_rect": list(clear_rect),
                        "object": {"type": o.get("type"), "bbox": [fx1, fy1, fx2, fy2]}
                    })

        # Walkway minimum clearance (door->center)
        res = 5.0
        gw, gh = int(max(1, round(room_w / res))), int(max(1, round(room_h / res)))
        occ = np.zeros((gh, gw), dtype=np.uint8)
        for o in [oo for oo in layout if not oo.get("fixed") and not oo.get("architectural")]:
            x1 = int(max(0, math.floor(o["x"] / res))); y1 = int(max(0, math.floor(o["y"] / res)))
            x2 = int(min(gw - 1, math.ceil((o["x"] + o["w"]) / res)))
            y2 = int(min(gh - 1, math.ceil((o["y"] + o["h"]) / res)))
            occ[y1:y2, x1:x2] = 255
        inv = (occ == 0).astype(np.uint8) * 255
        dist_cm = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3) * res
        cx, cy = room_w / 2.0, room_h / 2.0
        for door in doors:
            dx1, dy1, dx2, dy2 = door.get("bbox", (0,0,0,0))
            dxc, dyc = (dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0
            N = 100
            min_clear = float("inf")
            for t in np.linspace(0.0, 1.0, N):
                px = dxc * (1 - t) + cx * t
                py = dyc * (1 - t) + cy * t
                gx = int(min(gw - 1, max(0, round(px / res))))
                gy = int(min(gh - 1, max(0, round(py / res))))
                min_clear = min(min_clear, float(dist_cm[gy, gx]))
            diags.append({
                "kind": "walkway_clearance",
                "door_center": [dxc, dyc],
                "to": [cx, cy],
                "min_clearance_cm": min_clear
            })

        # Window blocking > threshold
        windows = [d for d in self.room_analysis.get("detections", []) if d.get("type") == "window"]
        def rect_overlap(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            return (ix2 - ix1) * (iy2 - iy1)
        for wdet in windows:
            wb = wdet.get("bbox", (0,0,0,0))
            warea = max(0.0, (wb[2]-wb[0])*(wb[3]-wb[1]))
            if warea <= 0:
                continue
            blocked = 0.0
            offenders = []
            for o in [oo for oo in layout if not oo.get("fixed") and not oo.get("architectural")]:
                ob = (o["x"], o["y"], o["x"]+o["w"], o["y"]+o["h"])
                ov = rect_overlap(ob, wb)
                if ov > 0:
                    blocked += ov
                    offenders.append({"type": o.get("type"), "bbox": list(ob), "overlap": ov})
            pct = 100.0 * blocked / warea if warea > 0 else 0.0
            if pct > 25.0:
                diags.append({
                    "kind": "window_block",
                    "window_bbox": list(wb),
                    "blocked_pct": pct,
                    "offenders": offenders
                })
        return diags
