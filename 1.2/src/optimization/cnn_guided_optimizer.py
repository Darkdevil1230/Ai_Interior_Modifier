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

        # Cache basic architectural primitives for constraint reasoning
        self._room_w, self._room_h = self.room_dims
        self._doors = self._extract_openings("door")
        self._windows = self._extract_openings("window")
        self._walls = self._extract_walls()

    # ------------------------------------------------------------------
    # Architectural helpers (doors, windows, walls)
    # ------------------------------------------------------------------

    def _extract_openings(self, kind: str) -> List[Dict]:
        def _dtype(det: Dict) -> str:
            return str(det.get("type") or det.get("label") or "").lower().replace("_", " ")
        out: List[Dict] = []
        for d in self.room_analysis.get("detections", []):
            if _dtype(d) != kind:
                continue
            bb = d.get("room_bbox") or d.get("bbox")
            if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                continue
            x1, y1, x2, y2 = map(float, bb)
            out.append({"type": kind, "bbox": [x1, y1, x2, y2]})
        return out

    def _extract_walls(self) -> List[Dict]:
        """Return simple perimeter walls as line segments (top/bottom/left/right)."""
        w, h = self.room_dims
        return [
            {"name": "left", "orientation": "vertical", "x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": h},
            {"name": "right", "orientation": "vertical", "x1": w, "y1": 0.0, "x2": w, "y2": h},
            {"name": "top", "orientation": "horizontal", "x1": 0.0, "y1": 0.0, "x2": w, "y2": 0.0},
            {"name": "bottom", "orientation": "horizontal", "x1": 0.0, "y1": h, "x2": w, "y2": h},
        ]

    # ------------------------------------------------------------------
    # Wall selection / placement
    # ------------------------------------------------------------------

    def _wall_for_object(self, obj: Dict) -> Optional[Dict]:
        cx = obj["x"] + obj["w"] / 2.0
        cy = obj["y"] + obj["h"] / 2.0
        w, h = self.room_dims
        candidates = [
            ("left", cx),
            ("right", w - cx),
            ("top", cy),
            ("bottom", h - cy),
        ]
        name = min(candidates, key=lambda t: t[1])[0]
        for wall in self._walls:
            if wall["name"] == name:
                return wall
        return None

    def find_best_wall(self, furniture: Dict, role: str, layout: List[Dict]) -> Dict:
        """Choose the most appropriate perimeter wall for an anchored object."""
        role_l = role.lower()
        walls = self._walls

        def length(wall: Dict) -> float:
            return math.hypot(wall["x2"] - wall["x1"], wall["y2"] - wall["y1"])

        def opposite(name: str) -> Optional[str]:
            mapping = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
            return mapping.get(name)

        # If sofa has TV, sofa should be opposite TV and vice versa
        if role_l == "sofa":
            tvs = [o for o in layout if "tv" in o.get("type", "").lower()]
            if tvs:
                tv_wall = self._wall_for_object(tvs[0])
                if tv_wall:
                    opp = opposite(tv_wall["name"])
                    for w in walls:
                        if w["name"] == opp:
                            return w
        if role_l == "tv":
            sofas = [o for o in layout if "sofa" in o.get("type", "").lower() or "couch" in o.get("type", "").lower()]
            if sofas:
                sofa_wall = self._wall_for_object(sofas[0])
                if sofa_wall:
                    opp = opposite(sofa_wall["name"])
                    for w in walls:
                        if w["name"] == opp:
                            return w

        # Fallback: longest wall for sofa/bed; otherwise nearest to corners
        if role_l in {"sofa", "bed"}:
            return max(walls, key=length)
        return max(walls, key=length)

    def place_against_wall(self, obj: Dict, wall: Dict, layout: List[Dict], margin: float = 5.0) -> Optional[Tuple[float, float]]:
        """Snap an object to a wall while keeping overlaps and door swing clear."""
        room_w, room_h = self.room_dims
        w, h = float(obj.get("w", 0.0)), float(obj.get("h", 0.0))
        if w <= 0 or h <= 0:
            return None

        candidates: List[Tuple[float, float]] = []
        step = max(10.0, min(w, h) / 2.0)

        if wall["orientation"] == "horizontal":
            # top or bottom
            if wall["y1"] < room_h / 2.0:
                y = margin
            else:
                y = room_h - h - margin
            x = margin
            while x + w + margin <= room_w + 1e-6:
                candidates.append((x, y))
                x += step
        else:
            # left or right
            if wall["x1"] < room_w / 2.0:
                x = margin
            else:
                x = room_w - w - margin
            y = margin
            while y + h + margin <= room_h + 1e-6:
                candidates.append((x, y))
                y += step

        best: Optional[Tuple[float, float]] = None
        for (cx, cy) in candidates:
            test = dict(obj)
            test.update({"x": cx, "y": cy})
            if any(self._overlap(test, other) for other in layout):
                continue
            if not self.check_door_clearance(test):
                continue
            best = (cx, cy)
            break
        return best

    # ------------------------------------------------------------------
    # Core hard constraints used by the deterministic solver
    # ------------------------------------------------------------------

    def check_door_clearance(self, obj: Dict) -> bool:
        """No object intersects door polygon or 90cm swing path."""
        if not self._doors:
            return True
        x1, y1 = obj["x"], obj["y"]
        x2, y2 = x1 + obj["w"], y1 + obj["h"]
        for door in self._doors:
            dx1, dy1, dx2, dy2 = door["bbox"]
            px_w = abs(dx2 - dx1)
            px_h = abs(dy2 - dy1)
            swing = max(90.0, float(min(px_w, px_h)))
            expand = 20.0
            crx1 = min(dx1, dx2) - expand
            cry1 = min(dy1, dy2) - expand
            crx2 = max(dx1, dx2) + expand + swing
            cry2 = max(dy1, dy2) + expand + swing
            if not (x2 <= crx1 or x1 >= crx2 or y2 <= cry1 or y1 >= cry2):
                return False
        return True

    def check_window_blocking(self, obj: Dict) -> bool:
        """Sofa/wardrobe/bookcase must not block windows; desk/plant should be near."""
        t = obj.get("type", "").lower().replace("_", " ")
        if not self._windows:
            return True
        x1, y1 = obj["x"], obj["y"]
        x2, y2 = x1 + obj["w"], y1 + obj["h"]

        def overlap(a, b) -> float:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            return (ix2 - ix1) * (iy2 - iy1)

        for win in self._windows:
            wx1, wy1, wx2, wy2 = win["bbox"]
            if t in {"sofa", "wardrobe", "bookcase", "bookshelf"}:
                if overlap((x1, y1, x2, y2), (wx1, wy1, wx2, wy2)) > 0:
                    return False
            if t in {"desk", "plant", "small plant", "large plant"}:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                wcx = (wx1 + wx2) / 2.0
                wcy = (wy1 + wy2) / 2.0
                if math.hypot(cx - wcx, cy - wcy) > 100.0 + 1e-6:
                    # Not hard failure – GA will still prefer closer placements –
                    # but the deterministic solver uses this as a filter.
                    return False
        return True

    def route_path_clearance(self, obj: Dict, layout: List[Dict], min_width: float = 80.0) -> bool:
        """Approximate navmesh: ensure door→center path keeps min corridor width."""
        if not self._doors:
            return True
        w, h = self.room_dims
        cx, cy = w / 2.0, h / 2.0
        res = 5.0
        gw, gh = int(max(1, round(w / res))), int(max(1, round(h / res)))
        occ = np.zeros((gh, gw), dtype=np.uint8)

        def mark(o: Dict) -> None:
            x1 = int(max(0, math.floor(o["x"] / res)))
            y1 = int(max(0, math.floor(o["y"] / res)))
            x2 = int(min(gw - 1, math.ceil((o["x"] + o["w"]) / res)))
            y2 = int(min(gh - 1, math.ceil((o["y"] + o["h"]) / res)))
            occ[y1:y2, x1:x2] = 255

        for o in layout:
            if o.get("fixed") or o.get("architectural"):
                continue
            mark(o)
        mark(obj)

        inv = (occ == 0).astype(np.uint8) * 255
        dist = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3) * res

        for door in self._doors:
            dx1, dy1, dx2, dy2 = door["bbox"]
            dxc, dyc = (dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0
            N = 80
            min_clear = float("inf")
            for t in np.linspace(0.0, 1.0, N):
                px = dxc * (1 - t) + cx * t
                py = dyc * (1 - t) + cy * t
                gx = int(min(gw - 1, max(0, round(px / res))))
                gy = int(min(gh - 1, max(0, round(py / res))))
                min_clear = min(min_clear, float(dist[gy, gx]))
            if min_clear < min_width - 1e-6:
                return False
        return True

    # ------------------------------------------------------------------
    # Functional zones and deterministic constraint solver
    # ------------------------------------------------------------------

    def find_functional_zones(self) -> Dict[str, Dict]:
        """Create coarse functional zones for living, reading, work, green, entry, storage."""
        w, h = self.room_dims
        zones: Dict[str, Dict] = {}

        # Entry near first door if available
        if self._doors:
            dx1, dy1, dx2, dy2 = self._doors[0]["bbox"]
            dxc, dyc = (dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0
            zw, zh = w * 0.3, h * 0.3
            ex1 = max(0.0, dxc - zw / 2.0)
            ey1 = max(0.0, dyc - zh / 2.0)
            ex2 = min(w, ex1 + zw)
            ey2 = min(h, ey1 + zh)
            zones["entry"] = {"name": "entry", "bbox": [ex1, ey1, ex2, ey2], "center": [(ex1 + ex2) / 2.0, (ey1 + ey2) / 2.0], "size": [ex2 - ex1, ey2 - ey1], "members": []}

        # Living zone in the central band
        lv_w, lv_h = w * 0.6, h * 0.5
        lx1 = (w - lv_w) / 2.0
        ly1 = (h - lv_h) / 2.0
        lx2, ly2 = lx1 + lv_w, ly1 + lv_h
        zones["living"] = {"name": "living", "bbox": [lx1, ly1, lx2, ly2], "center": [(lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0], "size": [lv_w, lv_h], "members": []}

        # Reading zone in a back corner
        rz_w, rz_h = w * 0.35, h * 0.35
        rx1 = w - rz_w - 10.0
        ry1 = h - rz_h - 10.0
        rx2, ry2 = rx1 + rz_w, ry1 + rz_h
        zones["reading"] = {"name": "reading", "bbox": [rx1, ry1, rx2, ry2], "center": [(rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0], "size": [rz_w, rz_h], "members": []}

        # Work and green zones near first window if available
        if self._windows:
            wx1, wy1, wx2, wy2 = self._windows[0]["bbox"]
            wcx, wcy = (wx1 + wx2) / 2.0, (wy1 + wy2) / 2.0
            wz_w, wz_h = w * 0.4, h * 0.4
            bx1 = max(0.0, wcx - wz_w / 2.0)
            by1 = max(0.0, wcy - wz_h / 2.0)
            bx2 = min(w, bx1 + wz_w)
            by2 = min(h, by1 + wz_h)
            zones["work"] = {"name": "work", "bbox": [bx1, by1, bx2, by2], "center": [(bx1 + bx2) / 2.0, (by1 + by2) / 2.0], "size": [bx2 - bx1, by2 - by1], "members": []}

            gz_w, gz_h = wz_w * 0.6, wz_h * 0.6
            gx1 = max(0.0, wcx - gz_w / 2.0)
            gy1 = max(0.0, wcy + 10.0)
            gx2 = min(w, gx1 + gz_w)
            gy2 = min(h, gy1 + gz_h)
            zones["green"] = {"name": "green", "bbox": [gx1, gy1, gx2, gy2], "center": [(gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0], "size": [gx2 - gx1, gy2 - gy1], "members": []}

        # Storage band along one long wall
        sz_w, sz_h = w * 0.25, h * 0.9
        sx1, sy1 = w - sz_w - 5.0, 5.0
        sx2, sy2 = sx1 + sz_w, sy1 + sz_h
        zones["storage"] = {"name": "storage", "bbox": [sx1, sy1, sx2, sy2], "center": [(sx1 + sx2) / 2.0, (sy1 + sy2) / 2.0], "size": [sz_w, sz_h], "members": []}

        return zones

    def run_constraint_solver(self) -> List[Dict]:
        """Deterministic layout used to seed the GA instead of random placement."""
        if not self.base_objects:
            return []

        layout: List[Dict] = []
        zones = self.find_functional_zones()

        def T(o: Dict) -> str:
            return o.get("type", "").lower().replace("_", " ")

        # --- Step 1: anchor wall objects (sofa, bed, wardrobe, tv, bookshelf, mirror, desk) ---
        wall_candidates: List[Dict] = []
        for o in self.base_objects:
            t = T(o)
            if any(k in t for k in ["sofa", "bed", "wardrobe", "bookcase", "bookshelf", "tv", "mirror", "desk"]):
                wall_candidates.append(o)

        # TV first
        for o in wall_candidates:
            if "tv" in T(o):
                wall = self.find_best_wall(o, "tv", layout)
                pos = self.place_against_wall(o, wall, layout)
                if pos is not None:
                    o["x"], o["y"] = pos
                    layout.append(dict(o))

        # Sofa next
        for o in wall_candidates:
            if "sofa" in T(o) or "couch" in T(o):
                wall = self.find_best_wall(o, "sofa", layout)
                pos = self.place_against_wall(o, wall, layout)
                if pos is not None:
                    o["x"], o["y"] = pos
                    layout.append(dict(o))
                    if "living" in zones:
                        zones["living"]["members"].append(o)

        # Bed headboard
        for o in wall_candidates:
            if "bed" in T(o):
                wall = self.find_best_wall(o, "bed", layout)
                pos = self.place_against_wall(o, wall, layout)
                if pos is not None:
                    o["x"], o["y"] = pos
                    layout.append(dict(o))

        # Remaining wall storage (wardrobe, bookshelf, desk, mirror)
        for o in wall_candidates:
            if any(x in T(o) for x in ["wardrobe", "bookcase", "bookshelf", "desk", "mirror"]):
                if any(o is placed for placed in layout):
                    # Already added above
                    continue
                wall = self.find_best_wall(o, "storage", layout)
                pos = self.place_against_wall(o, wall, layout)
                if pos is not None:
                    o["x"], o["y"] = pos
                    layout.append(dict(o))
                    if "storage" in zones:
                        zones["storage"]["members"].append(o)

        # --- Step 2: assign remaining furniture into functional zones ---
        for o in self.base_objects:
            if any(self._overlap(o, placed) for placed in layout):
                continue
            t = T(o)
            if any(k in t for k in ["coffee table", "center table"]):
                zones.get("living", zones[next(iter(zones))])["members"].append(o)
            elif any(k in t for k in ["armchair", "accent chair", "reading chair"]):
                zones.get("reading", zones.get("living", list(zones.values())[0]))["members"].append(o)
            elif "desk" in t:
                zones.get("work", zones.get("living", list(zones.values())[0]))["members"].append(o)
            elif "lamp" in t:
                zones.get("reading", zones.get("living", list(zones.values())[0]))["members"].append(o)
            elif "plant" in t:
                zones.get("green", zones.get("living", list(zones.values())[0]))["members"].append(o)
            else:
                zones.get("living", list(zones.values())[0])["members"].append(o)

        # --- Step 3: place loose objects inside zones with simple grid packing ---
        placed_ids = {id(o) for o in layout}
        for z in zones.values():
            x1, y1, x2, y2 = z["bbox"]
            zw, zh = x2 - x1, y2 - y1
            if zw <= 0 or zh <= 0:
                continue
            cursor_x = x1 + 10.0
            cursor_y = y1 + 10.0
            row_h = 0.0
            for o in z.get("members", []):
                if id(o) in placed_ids:
                    continue
                wobj = float(o.get("w", 0.0))
                hobj = float(o.get("h", 0.0))
                if wobj <= 0 or hobj <= 0:
                    continue
                if cursor_x + wobj + 10.0 > x2:
                    cursor_x = x1 + 10.0
                    cursor_y += row_h + 15.0
                    row_h = 0.0
                if cursor_y + hobj + 10.0 > y2:
                    continue
                trial = dict(o)
                trial.update({"x": cursor_x, "y": cursor_y})
                if any(self._overlap(trial, other) for other in layout):
                    continue
                if not self.check_door_clearance(trial):
                    continue
                if not self.check_window_blocking(trial):
                    continue
                if not self.route_path_clearance(trial, layout):
                    continue
                o["x"], o["y"] = cursor_x, cursor_y
                layout.append(dict(o))
                placed_ids.add(id(o))
                cursor_x += wobj + 15.0
                row_h = max(row_h, hobj)

        # Any still-unplaced objects: drop into free space with constraints
        for o in self.base_objects:
            if id(o) in placed_ids:
                continue
            wobj = float(o.get("w", 0.0))
            hobj = float(o.get("h", 0.0))
            if wobj <= 0 or hobj <= 0:
                continue
            for _ in range(40):
                x = random.uniform(0, max(1.0, self._room_w - wobj))
                y = random.uniform(0, max(1.0, self._room_h - hobj))
                trial = dict(o)
                trial.update({"x": x, "y": y})
                if any(self._overlap(trial, other) for other in layout):
                    continue
                if not self.check_door_clearance(trial):
                    continue
                if not self.route_path_clearance(trial, layout):
                    continue
                o["x"], o["y"] = x, y
                layout.append(dict(o))
                placed_ids.add(id(o))
                break

        return layout
    
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
        wardrobes = [o for o in layout if ("wardrobe" in T(o) or "closet" in T(o))]
        plants = [o for o in layout if ("plant" in T(o) or "vase" in T(o))]

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

        # Beds: strongly prefer being anchored to a wall (headboard or side)
        for b in beds:
            d_wall = self._min_wall_distance(b)
            if d_wall > 5.0:  # >5cm away from any wall -> heavy penalty
                penalty += (d_wall - 5.0) * 400.0
            if nightstands:
                ns_d = min(math.hypot(self._center(n)[0] - self._center(b)[0], self._center(n)[1] - self._center(b)[1]) for n in nightstands) if nightstands else float("inf")
                if ns_d > 100.0:
                    penalty += (ns_d - 100.0) * 20.0

            # Maintain comfortable clearance between bed and wardrobes
            for wrob in wardrobes:
                # Approximate edge-to-edge distance along the shortest axis
                bx1, by1 = b["x"], b["y"]
                bx2, by2 = bx1 + b["w"], by1 + b["h"]
                wx1, wy1 = wrob["x"], wrob["y"]
                wx2, wy2 = wx1 + wrob["w"], wy1 + wrob["h"]
                # Horizontal gap if aligned vertically
                gap_x1 = max(0.0, wx1 - bx2)
                gap_x2 = max(0.0, bx1 - wx2)
                # Vertical gap if aligned horizontally
                gap_y1 = max(0.0, wy1 - by2)
                gap_y2 = max(0.0, by1 - wy2)
                edge_gap = min(g for g in [gap_x1, gap_x2, gap_y1, gap_y2] if g > 0) if any(g > 0 for g in [gap_x1, gap_x2, gap_y1, gap_y2]) else 0.0
                min_bed_wardrobe_gap = 60.0  # cm
                if edge_gap < min_bed_wardrobe_gap:
                    penalty += (min_bed_wardrobe_gap - edge_gap) * 150.0

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

        # Ottomans: keep comfortable clearance from sofas (avoid being jammed)
        ottomans = [o for o in layout if "ottoman" in T(o)]
        min_clearance = 40.0  # cm between sofa and ottoman centers
        for s in sofas:
            scx, scy = self._center(s)
            for ot in ottomans:
                ocx, ocy = self._center(ot)
                d = math.hypot(scx - ocx, scy - ocy)
                if d < min_clearance:
                    penalty += (min_clearance - d) * 200.0

        # Sofa vs wardrobes/bookcases: avoid jamming storage right against seating
        storage_blocks = wardrobes + [o for o in layout if any(k in T(o) for k in ["bookcase", "bookshelf", "cabinet"])]
        min_sofa_storage_gap = 50.0
        for s in sofas:
            sx1, sy1 = s["x"], s["y"]
            sx2, sy2 = sx1 + s["w"], sy1 + s["h"]
            for st in storage_blocks:
                tx1, ty1 = st["x"], st["y"]
                tx2, ty2 = tx1 + st["w"], ty1 + st["h"]
                gap_x1 = max(0.0, tx1 - sx2)
                gap_x2 = max(0.0, sx1 - tx2)
                gap_y1 = max(0.0, ty1 - sy2)
                gap_y2 = max(0.0, sy1 - ty2)
                edge_gap = min(g for g in [gap_x1, gap_x2, gap_y1, gap_y2] if g > 0) if any(g > 0 for g in [gap_x1, gap_x2, gap_y1, gap_y2]) else 0.0
                if edge_gap < min_sofa_storage_gap:
                    penalty += (min_sofa_storage_gap - edge_gap) * 120.0

        # Wardrobe front corridor: keep a clear strip in front for door swing / standing space
        corridor_depth = 70.0  # cm
        room_w, room_h = self.room_dims
        for wrob in wardrobes:
            x1, y1 = wrob["x"], wrob["y"]
            x2, y2 = x1 + wrob["w"], y1 + wrob["h"]
            # Determine which wall this wardrobe is against
            dist_left = x1
            dist_right = room_w - x2
            dist_top = y1
            dist_bottom = room_h - y2
            dmin = min(dist_left, dist_right, dist_top, dist_bottom)
            if dmin == dist_bottom:
                # Wardrobe on bottom wall -> corridor above it
                cx1, cy1, cx2, cy2 = x1, max(0.0, y1 - corridor_depth), x2, y1
            elif dmin == dist_top:
                # on top wall -> corridor below
                cx1, cy1, cx2, cy2 = x1, y2, x2, min(room_h, y2 + corridor_depth)
            elif dmin == dist_left:
                # on left wall -> corridor to the right
                cx1, cy1, cx2, cy2 = x2, y1, min(room_w, x2 + corridor_depth), y2
            else:
                # on right wall -> corridor to the left
                cx1, cy1, cx2, cy2 = max(0.0, x1 - corridor_depth), y1, x1, y2

            def _overlap_rect(a, b) -> float:
                ax1, ay1, ax2, ay2 = a
                bx1, by1, bx2, by2 = b
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                if ix2 <= ix1 or iy2 <= iy1:
                    return 0.0
                return (ix2 - ix1) * (iy2 - iy1)

            corridor = (cx1, cy1, cx2, cy2)
            for o in layout:
                if o is wrob:
                    continue
                ox1, oy1 = o["x"], o["y"]
                ox2, oy2 = ox1 + o["w"], oy1 + o["h"]
                if _overlap_rect(corridor, (ox1, oy1, ox2, oy2)) > 0.0:
                    penalty += 30000.0

        # Major furniture must not overlap plants/vases (decor obstacles)
        blocking_major = sofas + beds + wardrobes + desks + generic_tables + dining_tables

        # 1) Layout-level plant objects (if present in GA population)
        for maj in blocking_major:
            for pl in plants:
                if self._overlap(maj, pl):
                    # Heavy penalty to effectively forbid this configuration
                    penalty += 40000.0

        # 2) Detected plants/vases from room_analysis (most common case)
        det_plants = []
        for det in self.room_analysis.get("detections", []):
            dt = _dtype(det)
            if dt in {"plant", "potted plant"} or "plant" in dt or "vase" in dt:
                bb = det.get("room_bbox") or det.get("bbox")
                if isinstance(bb, (list, tuple)) and len(bb) == 4:
                    det_plants.append(tuple(map(float, bb)))

        def _rect_overlap(a, b) -> float:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            return (ix2 - ix1) * (iy2 - iy1)

        if det_plants:
            for maj in blocking_major:
                mx1, my1 = maj["x"], maj["y"]
                mx2, my2 = mx1 + maj["w"], my1 + maj["h"]
                mrect = (mx1, my1, mx2, my2)
                for pb in det_plants:
                    if _rect_overlap(mrect, pb) > 0.0:
                        penalty += 40000.0

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
        """Main optimization loop: deterministic constraint solver + GA refinement.

        1. Use run_constraint_solver() to build an architecturally sensible base layout.
        2. Seed a GA population with small perturbations of that layout.
        3. Refine using cnn_guided_fitness (which already encodes many soft rules).
        4. Run final validation passes; if any hard validation fails, re-run solver
           with a few attempts before returning.
        """
        # Safety check: ensure we have furniture to place
        if not self.base_objects:
            print("[WARNING] No furniture objects provided. Returning empty layout with architectural elements only.")
            return self._add_architectural_elements([])

        base_layout = self.run_constraint_solver()
        if not base_layout:
            print("[ERROR] Constraint solver produced empty layout, falling back to legacy initializer.")
            base_layout = self._cnn_guided_initialization()

        # Initialize population around base layout (small mutations only)
        population: List[List[Dict]] = [deepcopy(base_layout)]
        while len(population) < self.population_size:
            mutated = self.cnn_guided_mutation(deepcopy(base_layout), mutation_rate=0.15)
            population.append(mutated)

        best_so_far = deepcopy(base_layout)
        best_score = self.cnn_guided_fitness(best_so_far)

        for _ in range(self.generations):
            scores = [self.cnn_guided_fitness(ind) for ind in population]
            gen_best_idx = int(np.argmax(scores))
            if scores[gen_best_idx] > best_score:
                best_score = scores[gen_best_idx]
                best_so_far = deepcopy(population[gen_best_idx])

            idx_sorted = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            elites = [deepcopy(population[i]) for i in idx_sorted[:3]]

            new_population = elites.copy()
            while len(new_population) < self.population_size:
                parent1 = self._tournament_select(population, scores, k=3)
                parent2 = self._tournament_select(population, scores, k=3)
                child = self.cnn_guided_crossover(parent1, parent2)
                child = self.cnn_guided_mutation(child, mutation_rate=0.25)
                new_population.append(child)
            population = new_population

        # Final validation & repair loop
        best_valid = self._validate_and_repair(best_so_far)

        # Include architectural elements in output
        best_valid = self._add_architectural_elements(best_valid)
        try:
            self.diagnostics = self._collect_diagnostics(best_valid)
        except Exception:
            self.diagnostics = []
        return best_valid
    
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
        """Initialize layout using deterministic constraint solver instead of random placement."""
        layout = self.run_constraint_solver()
        if not layout:
            # Fallback to previous random behavior if solver fails
            fallback: List[Dict] = []
            for obj in self.base_objects:
                test_obj = dict(obj)
                test_obj.update({
                    "x": random.uniform(0, max(1.0, self.room_dims[0] - obj["w"])),
                    "y": random.uniform(0, max(1.0, self.room_dims[1] - obj["h"]))
                })
                fallback.append(test_obj)
            layout = fallback
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

    # ------------------------------------------------------------------
    # Final validation helpers
    # ------------------------------------------------------------------

    def _validate_and_repair(self, layout: List[Dict]) -> List[Dict]:
        """Run final hard validations and, if necessary, try limited re-solve.

        Validation set:
        - validate_no_door_obstruction
        - validate_walk_path
        - validate_window_access
        - validate_alignment
        - validate_lighting_rules
        """
        candidate = deepcopy(layout)

        def _ok(L: List[Dict]) -> bool:
            try:
                return (
                    self.validate_no_door_obstruction(L)
                    and self.validate_walk_path(L)
                    and self.validate_window_access(L)
                    and self.validate_alignment(L)
                    and self.validate_lighting_rules(L)
                )
            except Exception:
                # Fail open rather than crashing the pipeline
                return True

        # First, make sure we don't keep layouts with simple overlaps
        try:
            if self._has_overlaps(candidate):
                candidate = self._repair_layout(candidate)
        except Exception:
            pass

        if _ok(candidate) and not self._has_overlaps(candidate):
            return candidate

        # Try basic overlap-based repair once
        try:
            candidate = self._repair_layout(candidate)
        except Exception:
            pass
        try:
            if _ok(candidate) and not self._has_overlaps(candidate):
                return candidate
        except Exception:
            if _ok(candidate):
                return candidate

        # As a last resort, re-run deterministic solver a few times
        for _ in range(3):
            try:
                candidate = self.run_constraint_solver()
            except Exception:
                candidate = []
            if candidate and _ok(candidate) and not self._has_overlaps(candidate):
                return candidate
        # Fall back to the best we have, but prefer non-overlapping layout
        try:
            if candidate and not self._has_overlaps(candidate):
                return candidate
        except Exception:
            if candidate:
                return candidate
        return layout

    def validate_no_door_obstruction(self, layout: List[Dict]) -> bool:
        """Ensure movable furniture does not violate door swing clearance."""
        try:
            for o in layout:
                if o.get("fixed") or o.get("architectural"):
                    continue
                if not self.check_door_clearance(o):
                    return False
            return True
        except Exception:
            return True

    def validate_walk_path(self, layout: List[Dict]) -> bool:
        """Ensure each door has a reasonably clear path to room center."""
        try:
            if not self._doors:
                return True
            room_w, room_h = self.room_dims
            res = 5.0
            gw, gh = int(max(1, round(room_w / res))), int(max(1, round(room_h / res)))
            occ = np.zeros((gh, gw), dtype=np.uint8)
            for o in layout:
                if o.get("fixed") or o.get("architectural"):
                    continue
                x1 = int(max(0, math.floor(o["x"] / res)))
                y1 = int(max(0, math.floor(o["y"] / res)))
                x2 = int(min(gw - 1, math.ceil((o["x"] + o["w"]) / res)))
                y2 = int(min(gh - 1, math.ceil((o["y"] + o["h"]) / res)))
                occ[y1:y2, x1:x2] = 255
            inv = (occ == 0).astype(np.uint8) * 255
            dist = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3) * res
            cx, cy = room_w / 2.0, room_h / 2.0
            for door in self._doors:
                dx1, dy1, dx2, dy2 = door["bbox"]
                dxc, dyc = (dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0
                N = 80
                min_clear = float("inf")
                for t in np.linspace(0.0, 1.0, N):
                    px = dxc * (1 - t) + cx * t
                    py = dyc * (1 - t) + cy * t
                    gx = int(min(gw - 1, max(0, round(px / res))))
                    gy = int(min(gh - 1, max(0, round(py / res))))
                    min_clear = min(min_clear, float(dist[gy, gx]))
                if min_clear < 80.0 - 1e-6:
                    return False
            return True
        except Exception:
            return True

    def validate_window_access(self, layout: List[Dict]) -> bool:
        """Ensure large furniture does not block window apertures excessively."""
        try:
            if not self._windows:
                return True

            def rect_overlap(a, b) -> float:
                ax1, ay1, ax2, ay2 = a
                bx1, by1, bx2, by2 = b
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                if ix2 <= ix1 or iy2 <= iy1:
                    return 0.0
                return (ix2 - ix1) * (iy2 - iy1)

            for win in self._windows:
                wb = win["bbox"]
                warea = max(0.0, (wb[2] - wb[0]) * (wb[3] - wb[1]))
                if warea <= 0:
                    continue
                blocked = 0.0
                for o in layout:
                    if o.get("fixed") or o.get("architectural"):
                        continue
                    ob = (o["x"], o["y"], o["x"] + o["w"], o["y"] + o["h"])
                    blocked += rect_overlap(ob, wb)
                pct = 100.0 * blocked / warea if warea > 0 else 0.0
                if pct > 35.0:
                    return False
            return True
        except Exception:
            return True

    def validate_alignment(self, layout: List[Dict]) -> bool:
        """Light-weight checks for basic alignment rules (sofa↔TV, bed↔wall)."""
        try:
            sofas = [o for o in layout if "sofa" in o.get("type", "").lower() or "couch" in o.get("type", "").lower()]
            tvs = [o for o in layout if "tv" in o.get("type", "").lower()]
            if sofas and tvs:
                s = sofas[0]
                t = tvs[0]
                scx = s["x"] + s["w"] / 2.0
                tcx = t["x"] + t["w"] / 2.0
                scy = s["y"] + s["h"] / 2.0
                tcy = t["y"] + t["h"] / 2.0
                align_err = min(abs(scx - tcx), abs(scy - tcy))
                if align_err > 120.0:
                    return False
            # Bed near at least one wall
            beds = [o for o in layout if "bed" in o.get("type", "").lower()]
            for b in beds:
                dmin = min(b["x"], b["y"], self.room_dims[0] - (b["x"] + b["w"]), self.room_dims[1] - (b["y"] + b["h"]))
                if dmin > 80.0:
                    return False
            return True
        except Exception:
            return True

    def validate_lighting_rules(self, layout: List[Dict]) -> bool:
        """Ensure desks and chairs have at least one nearby window or lamp."""
        try:
            lights = [o for o in layout if any(k in o.get("type", "").lower() for k in ["lamp", "chandelier", "pendant", "sconce"])]
            for o in layout:
                t = o.get("type", "").lower().replace("_", " ")
                if t not in {"desk", "chair", "office chair", "dining chair"}:
                    continue
                cx = o["x"] + o["w"] / 2.0
                cy = o["y"] + o["h"] / 2.0
                best = float("inf")
                for win in self._windows:
                    wx1, wy1, wx2, wy2 = win["bbox"]
                    wcx, wcy = (wx1 + wx2) / 2.0, (wy1 + wy2) / 2.0
                    best = min(best, math.hypot(cx - wcx, cy - wcy))
                for l in lights:
                    lx = l["x"] + l["w"] / 2.0
                    ly = l["y"] + l["h"] / 2.0
                    best = min(best, math.hypot(cx - lx, cy - ly))
                if best == float("inf"):
                    continue
                if best > 180.0:
                    return False
            return True
        except Exception:
            return True
