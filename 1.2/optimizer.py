"""
optimizer.py
------------
Improved Genetic Algorithm for optimizing furniture placement.

Improvements:
 - Configurable population/generations/seed.
 - Tournament selection + elitism.
 - Penalty for violating minimum clearance between objects.
 - Better crossover & mutation (per-object mutation).
 - Exports richer JSON with metadata.
"""

import random
import math
import json
import os
from copy import deepcopy

import numpy as np
from rule_learner import RuleWeightLearner

# Absolute paths for data files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RULES_PATH = os.path.join(BASE_DIR, "data", "architectural_rules.json")


class LayoutOptimizer:
    def __init__(self, room_dims, objects, user_prefs=None, population_size=100, generations=250, seed=None):
        """
        :param room_dims: (width_cm, height_cm) tuple.
        :param objects: List of objects with keys: type, w, h (x,y optional).
        :param user_prefs: Dict of preferences (bed_near_wall, table_near_window, min_distance).
        :param population_size: GA population size (default 100, optimized for quality).
        :param generations: Number of generations (default 250, optimized for convergence).
        :param seed: Optional RNG seed for reproducibility.
        """
        self.room_dims = tuple(map(float, room_dims))
        self.base_objects = deepcopy(objects)
        self.user_prefs = user_prefs or {}
        self.population_size = max(10, int(population_size))
        self.generations = max(1, int(generations))
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Rules configuration and adaptive learner
        self.rules = self._load_arch_rules()
        self.rule_learner = RuleWeightLearner(os.path.join(BASE_DIR, "data", "rule_memory.json"))
        print("Architectural optimizer active 🧠")

    def _load_arch_rules(self):
        path = RULES_PATH
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        print(f"Architectural rules loaded ✅ ({len(data.get('group_rules', []))} group rules)")
                        return data
        except Exception:
            pass
        return {"group_rules": [], "position_rules": [], "global_constraints": {}}

    def _distance(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _angle_deg(self, a, b):
        dx, dy = (b[0] - a[0]), (b[1] - a[1])
        ang = math.degrees(math.atan2(dy, dx))
        return (ang + 360.0) % 360.0

    def _center(self, obj):
        return (obj["x"] + obj["w"] / 2.0, obj["y"] + obj["h"] / 2.0)

    def _by_type(self, layout, key):
        k = key.lower()
        return [o for o in layout if k in o.get("type", "").lower()]

    def _nearest(self, layout, src, key):
        cand = self._by_type(layout, key)
        if not cand:
            return None
        sc = self._center(src)
        return min(cand, key=lambda o: self._distance(sc, self._center(o)))

    def architectural_rule_score(self, layout):
        score = 0.0
        # Group rules
        for idx, rule in enumerate(self.rules.get("group_rules", [])):
            fid = str(idx)
            wmult = self.rule_learner.get_weight_adjustment(fid)
            rel = str(rule.get("relation", "")).lower()
            fset = self._by_type(layout, rule.get("furniture", ""))
            if not fset:
                continue
            if rel == "faces":
                tset = self._by_type(layout, rule.get("target", ""))
                if not tset:
                    continue
                rng = rule.get("distance", [0, 9e9])
                tol = float(rule.get("angle_tolerance", 25))
                base_w = float(rule.get("weight", 10))
                for a in fset:
                    b = self._nearest(layout, a, rule.get("target", ""))
                    if not b:
                        continue
                    ca, cb = self._center(a), self._center(b)
                    d = self._distance(ca, cb)
                    if rng[0] <= d <= rng[1]:
                        ang = abs(((self._angle_deg(ca, cb) + 360) % 180) - 90)
                        # Prefer near 0 (facing across) -> within tolerance
                        sat = 1.0 if ang <= tol else max(0.0, 1.0 - (ang - tol) / 90.0)
                        delta = base_w * sat * wmult
                        score += delta
                        self.rule_learner.update_memory(fid, delta)
            elif rel == "avoid_line_of_sight":
                tset = self._by_type(layout, rule.get("target", ""))
                if not tset:
                    continue
                # Use sofa if present to define LOS; else any furniture center line to target
                sofas = self._by_type(layout, "sofa")
                tv = self._nearest(layout, sofas[0] if sofas else fset[0], rule.get("target", "")) if (sofas or tset) else None
                if not tv:
                    continue
                sc = self._center(sofas[0] if sofas else fset[0])
                tc = self._center(tv)
                pen = float(rule.get("penalty", 20))
                def _ccw(A,B,C):
                    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
                def _segint(A,B,C,D):
                    return _ccw(A,C,D) != _ccw(B,C,D) and _ccw(A,B,C) != _ccw(A,B,D)
                for m in fset:
                    x,y,w,h = m["x"], m["y"], m["w"], m["h"]
                    r1=(x,y); r2=(x+w,y); r3=(x+w,y+h); r4=(x,y+h)
                    hit = _segint(sc, tc, r1, r2) or _segint(sc, tc, r2, r3) or _segint(sc, tc, r3, r4) or _segint(sc, tc, r4, r1)
                    if hit:
                        delta = -pen * wmult
                        score += delta
                        self.rule_learner.update_memory(fid, delta)
            elif rel == "near_light_source":
                t = self._by_type(layout, rule.get("target", ""))
                if not t:
                    continue
                rng = rule.get("distance", [0, 100])
                base_w = float(rule.get("weight", 10))
                for a in fset:
                    b = self._nearest(layout, a, rule.get("target", ""))
                    if not b:
                        continue
                    d = self._distance(self._center(a), self._center(b))
                    if rng[0] <= d <= rng[1]:
                        delta = base_w * wmult
                        score += delta
                        self.rule_learner.update_memory(fid, delta)
            elif rel == "avoid_glare":
                t = self._by_type(layout, rule.get("target", ""))
                if not t:
                    continue
                pen = float(rule.get("penalty", 20))
                tol = float(rule.get("angle_tolerance", 45))
                for a in fset:
                    b = self._nearest(layout, a, rule.get("target", ""))
                    if not b:
                        continue
                    ang = abs(((self._angle_deg(self._center(b), self._center(a)) + 360) % 180) - 90)
                    if ang <= tol:
                        delta = -pen * wmult
                        score += delta
                        self.rule_learner.update_memory(fid, delta)
            elif rel == "adjacent":
                t = self._by_type(layout, rule.get("target", ""))
                if not t:
                    continue
                rng = rule.get("distance", [0, 80])
                base_w = float(rule.get("weight", 10))
                for a in fset:
                    b = self._nearest(layout, a, rule.get("target", ""))
                    if not b:
                        continue
                    d = self._distance(self._center(a), self._center(b))
                    if rng[0] <= d <= rng[1]:
                        delta = base_w * wmult
                        score += delta
                        self.rule_learner.update_memory(fid, delta)

        # Position rules
        for jdx, rule in enumerate(self.rules.get("position_rules", [])):
            fid = f"p{jdx}"
            wmult = self.rule_learner.get_weight_adjustment(fid)
            rel = str(rule.get("relation", "")).lower()
            fset = self._by_type(layout, rule.get("furniture", ""))
            if not fset:
                continue
            if rel == "wall_touch":
                off = float(rule.get("offset", 15))
                base_w = float(rule.get("weight", 15))
                for a in fset:
                    x,y,w,h = a["x"], a["y"], a["w"], a["h"]
                    dist = min(x, y, self.room_dims[0]- (x+w), self.room_dims[1]- (y+h))
                    if dist <= off:
                        delta = base_w * wmult
                        score += delta
                        self.rule_learner.update_memory(fid, delta)
            elif rel == "away_from":
                pen = float(rule.get("penalty", 20))
                min_d = float(rule.get("min_distance", 100))
                targets = self._by_type(layout, rule.get("target", ""))
                if not targets:
                    continue
                for a in fset:
                    b = self._nearest(layout, a, rule.get("target", ""))
                    if not b:
                        continue
                    d = self._distance(self._center(a), self._center(b))
                    if d < min_d:
                        delta = -pen * wmult
                        score += delta
                        self.rule_learner.update_memory(fid, delta)
            elif rel == "center_alignment":
                base_w = float(rule.get("weight", 10))
                t = self._by_type(layout, rule.get("target", ""))
                if not t:
                    continue
                tol = float(rule.get("tolerance", 30))
                for a in fset:
                    b = self._nearest(layout, a, rule.get("target", ""))
                    if not b:
                        continue
                    ca, cb = self._center(a), self._center(b)
                    align_err = min(abs(ca[0]-cb[0]), abs(ca[1]-cb[1]))
                    if align_err <= tol:
                        delta = base_w * wmult
                        score += delta
                        self.rule_learner.update_memory(fid, delta)

        # Global constraints
        gc = self.rules.get("global_constraints", {})
        max_center_density = float(gc.get("max_center_density", 0.6))
        cx1, cy1 = self.room_dims[0]*0.2, self.room_dims[1]*0.2
        cx2, cy2 = self.room_dims[0]*0.8, self.room_dims[1]*0.8
        center_area = (cx2-cx1)*(cy2-cy1)
        occ = 0.0
        for a in layout:
            ax1, ay1 = max(cx1, a["x"]), max(cy1, a["y"])
            ax2, ay2 = min(cx2, a["x"]+a["w"]), min(cy2, a["y"]+a["h"])
            if ax2>ax1 and ay2>ay1:
                occ += (ax2-ax1)*(ay2-ay1)
        density = occ / max(1.0, center_area)
        if density > max_center_density:
            over = density - max_center_density
            fid = "gc_center"
            mult = self.rule_learner.get_weight_adjustment(fid)
            delta = -50.0 * over * mult
            score += delta
            self.rule_learner.update_memory(fid, delta)

        if bool(gc.get("prefer_symmetry", True)):
            left_area = sum(a["w"]*a["h"] for a in layout if self._center(a)[0] < self.room_dims[0]/2)
            right_area = sum(a["w"]*a["h"] for a in layout if self._center(a)[0] >= self.room_dims[0]/2)
            total = max(1.0, left_area + right_area)
            imbalance = abs(left_area - right_area) / total
            fid = "gc_sym"
            mult = self.rule_learner.get_weight_adjustment(fid)
            delta = (1.0 - imbalance) * 20.0 * mult
            score += delta
            self.rule_learner.update_memory(fid, delta)

        return score

    def compute_architectural_score(self, layout):
        try:
            print(f"🧩 Running architectural brain: {len(self.rules.get('group_rules', []))} rules, {len(self.rules.get('position_rules', []))} position rules.")
        except Exception:
            print("🧩 Running architectural brain: 0 rules, 0 position rules.")
        return self.architectural_rule_score(layout)

    # --- Fitness and constraints ---
    def fitness(self, layout):
        """
        Composite fitness:
         - Start from positive base (sum of free-space incentives)
         - Large negative penalties for overlaps and out-of-bounds
         - Add rewards for user preferences
         - Penalize close proximities below min_distance
         - Add spread score to maximize comfort / accessibility
        Higher is better.
        """
        score = 10000.0  # base - increased for better scaling
        
        # Count violations
        num_overlaps = 0
        num_out_of_bounds = 0
        
        # Out-of-bounds penalty (severe)
        for obj in layout:
            if not self._in_bounds(obj):
                num_out_of_bounds += 1
                score -= 5000.0  # Much larger penalty

        # Overlap penalty (severe and quadratic)
        for i in range(len(layout)):
            for j in range(i + 1, len(layout)):
                if self._overlap(layout[i], layout[j]):
                    num_overlaps += 1
                    # Calculate overlap area for proportional penalty
                    overlap_area = self._overlap_area(layout[i], layout[j])
                    score -= 10000.0 * (1 + overlap_area / 1000.0)  # Quadratically worse

        # If there are any overlaps, severely penalize
        if num_overlaps > 0:
            score -= 50000.0  # Make layouts with overlaps unacceptable

        # Min distance penalty (clearance) - improved calculation
        min_distance = float(self.user_prefs.get("min_distance", 20.0))
        for i in range(len(layout)):
            for j in range(i + 1, len(layout)):
                dist = self._center_distance(layout[i], layout[j])
                # Use diagonal distance for more accurate clearance
                min_req = min_distance + (layout[i]["w"] + layout[i]["h"]) / 4.0 + (layout[j]["w"] + layout[j]["h"]) / 4.0
                if dist < min_req:
                    # quadratic penalty for being too close
                    score -= (min_req - dist) ** 2 * 15

        # User prefs rewards (only if no violations)
        if num_overlaps == 0 and num_out_of_bounds == 0:
            for obj in layout:
                t = obj.get("type", "").lower()
                if t == "bed" and self.user_prefs.get("bed_near_wall"):
                    # reward if center is within 10% of room width/height from any wall
                    cx = obj["x"] + obj["w"] / 2.0
                    cy = obj["y"] + obj["h"] / 2.0
                    if (cx < 0.1 * self.room_dims[0]) or (cx > 0.9 * self.room_dims[0]) or (cy < 0.1 * self.room_dims[1]) or (cy > 0.9 * self.room_dims[1]):
                        score += 200.0
                if t == "table" and self.user_prefs.get("table_near_window"):
                    # favor near any wall / window area (simple heuristic)
                    if obj["y"] < 0.15 * self.room_dims[1] or obj["y"] + obj["h"] > 0.85 * self.room_dims[1]:
                        score += 150.0

            # Spread score (encourage evenly spaced objects)
            score += self._spread_score(layout) * 0.8
        
        # Bonus for furniture grouping (sofa near coffee table, etc.)
        for i in range(len(layout)):
            for j in range(i + 1, len(layout)):
                if self._are_complementary(layout[i], layout[j]):
                    dist = self._center_distance(layout[i], layout[j])
                    if dist < 150:  # Within reasonable grouping distance
                        score += 50 / (1 + dist / 50)
        
        # Bonus for clear pathways (open space in room center)
        center_x, center_y = self.room_dims[0] / 2, self.room_dims[1] / 2
        pathway_size = min(self.room_dims) * 0.3
        objects_near_center = sum(1 for obj in layout 
                                  if abs(obj["x"] + obj["w"]/2 - center_x) < pathway_size/2
                                  and abs(obj["y"] + obj["h"]/2 - center_y) < pathway_size/2)
        if objects_near_center == 0:
            score += 100  # Bonus for clear pathway
        
        hard_penalty = self._hard_constraint_penalty(layout)
        score -= hard_penalty
        
        return score

    def _overlap(self, a, b):
        return not (a["x"] + a["w"] <= b["x"] or b["x"] + b["w"] <= a["x"] or a["y"] + a["h"] <= b["y"] or b["y"] + b["h"] <= a["y"])
    
    def _overlap_area(self, a, b):
        """Calculate the area of overlap between two rectangles."""
        if not self._overlap(a, b):
            return 0.0
        overlap_x = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
        overlap_y = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
        return overlap_x * overlap_y

    def _in_bounds(self, obj):
        return 0 <= obj["x"] <= self.room_dims[0] - obj["w"] and 0 <= obj["y"] <= self.room_dims[1] - obj["h"]

    def _center_distance(self, a, b):
        ax = a["x"] + a["w"] / 2.0
        ay = a["y"] + a["h"] / 2.0
        bx = b["x"] + b["w"] / 2.0
        by = b["y"] + b["h"] / 2.0
        return math.hypot(ax - bx, ay - by)

    def _spread_score(self, layout):
        dists = []
        for i in range(len(layout)):
            for j in range(i + 1, len(layout)):
                dists.append(self._center_distance(layout[i], layout[j]))
        return float(np.mean(dists)) if dists else 0.0
    
    def _are_complementary(self, obj1, obj2):
        """Check if two objects are complementary (should be near each other)."""
        complementary_pairs = [
            ("sofa", "coffee table"), ("couch", "coffee table"),
            ("chair", "table"), ("sofa", "tv"), ("couch", "tv"),
            ("bed", "nightstand"), ("desk", "chair")
        ]
        t1, t2 = obj1.get("type", "").lower(), obj2.get("type", "").lower()
        for pair in complementary_pairs:
            if (t1 in pair[0] and t2 in pair[1]) or (t1 in pair[1] and t2 in pair[0]):
                return True
        return False
    
    def _min_wall_distance(self, obj):
        x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
        return min(x, y, self.room_dims[0] - (x + w), self.room_dims[1] - (y + h))
    
    def _center(self, obj):
        return (obj["x"] + obj["w"] / 2.0, obj["y"] + obj["h"] / 2.0)
    
    def _hard_constraint_penalty(self, layout):
        penalty = 0.0
        if not layout:
            return penalty
        sofas = [o for o in layout if ("sofa" in o.get("type", "").lower() or "couch" in o.get("type", "").lower())]
        tvs = [o for o in layout if "tv" in o.get("type", "").lower()]
        coffee_tables = [o for o in layout if "coffee table" in o.get("type", "").lower()]
        dining_tables = [o for o in layout if "dining table" in o.get("type", "").lower()]
        generic_tables = [o for o in layout if ("table" in o.get("type", "").lower() and o not in dining_tables and o not in coffee_tables)]
        desks = [o for o in layout if "desk" in o.get("type", "").lower()]
        beds = [o for o in layout if "bed" in o.get("type", "").lower()]
        nightstands = [o for o in layout if ("nightstand" in o.get("type", "").lower() or "bedside table" in o.get("type", "").lower())]
        fridges = [o for o in layout if ("refrigerator" in o.get("type", "").lower() or "fridge" in o.get("type", "").lower())]
        tv_stands = [o for o in layout if "tv stand" in o.get("type", "").lower()]
        office_chairs = [o for o in layout if "office chair" in o.get("type", "").lower()]
        dining_chairs = [o for o in layout if "dining chair" in o.get("type", "").lower()]
        any_chairs = [o for o in layout if "chair" in o.get("type", "").lower()]

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

                # Penalize obstructions between sofa and TV (e.g., mirrors)
                # Define small helpers for segment-rectangle intersection
                def _ccw(A, B, C):
                    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

                def _seg_intersect(A, B, C, D):
                    return _ccw(A, C, D) != _ccw(B, C, D) and _ccw(A, B, C) != _ccw(A, B, D)

                def _segment_intersects_rect(P, Q, rx, ry, rw, rh):
                    # rectangle corners
                    r1 = (rx, ry)
                    r2 = (rx + rw, ry)
                    r3 = (rx + rw, ry + rh)
                    r4 = (rx, ry + rh)
                    # check against 4 edges
                    return (
                        _seg_intersect(P, Q, r1, r2) or
                        _seg_intersect(P, Q, r2, r3) or
                        _seg_intersect(P, Q, r3, r4) or
                        _seg_intersect(P, Q, r4, r1)
                    )

                mirrors = [o for o in layout if "mirror" in o.get("type", "").lower()]
                for m in mirrors:
                    mx, my, mw, mh = m["x"], m["y"], m["w"], m["h"]
                    # If the mirror intersects the view line between sofa and TV, apply a strong penalty
                    if _segment_intersects_rect(sc, tc, mx, my, mw, mh):
                        penalty += 12000.0
                    else:
                        # Otherwise, softly penalize if the mirror sits very close (<30cm) to the line of sight
                        # Compute distance from rectangle center to segment
                        mcx, mcy = mx + mw/2.0, my + mh/2.0
                        # Distance from point to segment
                        x1, y1 = sc
                        x2, y2 = tc
                        if (x1, y1) == (x2, y2):
                            dist = math.hypot(mcx - x1, mcy - y1)
                        else:
                            # projection
                            dx, dy = x2 - x1, y2 - y1
                            t = ((mcx - x1) * dx + (mcy - y1) * dy) / max(1e-6, (dx*dx + dy*dy))
                            t = max(0.0, min(1.0, t))
                            px, py = x1 + t * dx, y1 + t * dy
                            dist = math.hypot(mcx - px, mcy - py)
                        if dist < 30.0:
                            penalty += (30.0 - dist) * 50.0

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

        for oc in office_chairs:
            if desks:
                dsk = min(desks, key=lambda d1: math.hypot(self._center(d1)[0] - self._center(oc)[0], self._center(d1)[1] - self._center(oc)[1]))
                d = math.hypot(self._center(dsk)[0] - self._center(oc)[0], self._center(dsk)[1] - self._center(oc)[1])
                if d > 80.0:
                    penalty += (d - 80.0) * 20.0

        return penalty
    
    def _has_overlaps(self, layout):
        """Check if layout has any overlaps."""
        for i in range(len(layout)):
            for j in range(i + 1, len(layout)):
                if self._overlap(layout[i], layout[j]):
                    return True
        return False
    
    def _repair_layout(self, layout, aggressive=False):
        """Repair a layout by moving overlapping objects with multiple strategies."""
        repaired = deepcopy(layout)
        max_attempts = 200 if aggressive else 100
        min_distance = float(self.user_prefs.get("min_distance", 20.0))
        
        for attempt in range(max_attempts):
            if not self._has_overlaps(repaired):
                break
            
            # Strategy 1: Push apart overlapping objects
            moved = False
            for i in range(len(repaired)):
                for j in range(i + 1, len(repaired)):
                    if self._overlap(repaired[i], repaired[j]):
                        obj_a, obj_b = repaired[i], repaired[j]
                        
                        # Calculate centers
                        cx_a = obj_a["x"] + obj_a["w"] / 2.0
                        cy_a = obj_a["y"] + obj_a["h"] / 2.0
                        cx_b = obj_b["x"] + obj_b["w"] / 2.0
                        cy_b = obj_b["y"] + obj_b["h"] / 2.0
                        
                        # Calculate separation vector
                        dx = cx_b - cx_a
                        dy = cy_b - cy_a
                        
                        if abs(dx) < 1 and abs(dy) < 1:
                            # Objects are nearly on top of each other, use random direction
                            angle = random.uniform(0, 2 * math.pi)
                            dx = math.cos(angle)
                            dy = math.sin(angle)
                        else:
                            # Normalize
                            length = math.sqrt(dx*dx + dy*dy)
                            if length > 0:
                                dx /= length
                                dy /= length
                        
                        # Calculate required separation distance
                        required_sep = math.hypot(obj_a["w"] + obj_b["w"], obj_a["h"] + obj_b["h"]) / 2.0 + min_distance
                        
                        # Move both objects apart (more aggressive)
                        move_dist = required_sep * 0.6
                        
                        # Move obj_a
                        new_x_a = obj_a["x"] - dx * move_dist
                        new_y_a = obj_a["y"] - dy * move_dist
                        new_x_a = max(0, min(self.room_dims[0] - obj_a["w"], new_x_a))
                        new_y_a = max(0, min(self.room_dims[1] - obj_a["h"], new_y_a))
                        
                        # Move obj_b
                        new_x_b = obj_b["x"] + dx * move_dist
                        new_y_b = obj_b["y"] + dy * move_dist
                        new_x_b = max(0, min(self.room_dims[0] - obj_b["w"], new_x_b))
                        new_y_b = max(0, min(self.room_dims[1] - obj_b["h"], new_y_b))
                        
                        repaired[i]["x"] = new_x_a
                        repaired[i]["y"] = new_y_a
                        repaired[j]["x"] = new_x_b
                        repaired[j]["y"] = new_y_b
                        moved = True
            
            # Strategy 2: If still overlapping after many attempts, use grid placement
            if attempt > max_attempts // 2 and self._has_overlaps(repaired):
                repaired = self._grid_based_repair(repaired)
        
        # Final check: if still has overlaps, use emergency grid placement
        if self._has_overlaps(repaired):
            print(f"[WARNING] Layout still has overlaps after {max_attempts} repair attempts. Using grid placement.")
            repaired = self._grid_based_repair(repaired)
        
        return repaired
    
    def _grid_based_repair(self, layout):
        """Emergency repair using grid-based placement to guarantee no overlaps."""
        repaired = deepcopy(layout)
        min_distance = float(self.user_prefs.get("min_distance", 20.0))
        
        # Sort objects by size (largest first)
        sorted_indices = sorted(range(len(repaired)), 
                               key=lambda i: repaired[i]["w"] * repaired[i]["h"], 
                               reverse=True)
        
        # Create grid cells
        grid_size = 50  # cm
        grid_w = int(self.room_dims[0] / grid_size) + 1
        grid_h = int(self.room_dims[1] / grid_size) + 1
        occupied = [[False] * grid_h for _ in range(grid_w)]
        
        def mark_occupied(x, y, w, h):
            """Mark grid cells as occupied."""
            x1_cell = max(0, int(x / grid_size))
            y1_cell = max(0, int(y / grid_size))
            x2_cell = min(grid_w - 1, int((x + w + min_distance) / grid_size))
            y2_cell = min(grid_h - 1, int((y + h + min_distance) / grid_size))
            
            for gx in range(x1_cell, x2_cell + 1):
                for gy in range(y1_cell, y2_cell + 1):
                    if 0 <= gx < grid_w and 0 <= gy < grid_h:
                        occupied[gx][gy] = True
        
        def find_free_position(w, h):
            """Find a free position for object of given size."""
            # Try to find a position with minimal grid cell conflicts
            best_pos = None
            min_conflicts = float('inf')
            
            attempts = 100
            for _ in range(attempts):
                x = random.uniform(0, max(1, self.room_dims[0] - w))
                y = random.uniform(0, max(1, self.room_dims[1] - h))
                
                # Check conflicts
                x1_cell = int(x / grid_size)
                y1_cell = int(y / grid_size)
                x2_cell = int((x + w) / grid_size)
                y2_cell = int((y + h) / grid_size)
                
                conflicts = 0
                for gx in range(x1_cell, x2_cell + 1):
                    for gy in range(y1_cell, y2_cell + 1):
                        if 0 <= gx < grid_w and 0 <= gy < grid_h and occupied[gx][gy]:
                            conflicts += 1
                
                if conflicts == 0:
                    return x, y
                
                if conflicts < min_conflicts:
                    min_conflicts = conflicts
                    best_pos = (x, y)
            
            # Return best position found
            return best_pos if best_pos else (0, 0)
        
        # Place each object
        for idx in sorted_indices:
            obj = repaired[idx]
            w, h = obj["w"], obj["h"]
            
            # Find free position
            x, y = find_free_position(w, h)
            
            # Ensure bounds
            x = max(0, min(self.room_dims[0] - w, x))
            y = max(0, min(self.room_dims[1] - h, y))
            
            # Update position
            repaired[idx]["x"] = x
            repaired[idx]["y"] = y
            
            # Mark as occupied
            mark_occupied(x, y, w, h)
        
        return repaired

    # --- Initialization / genetic operators ---
    def _random_layout(self):
        """Generate a random layout with smart placement to minimize overlaps."""
        layout = []
        max_attempts = 50  # Increased from 20
        min_distance = float(self.user_prefs.get("min_distance", 20.0))
        
        # Sort objects by size (largest first for better packing)
        sorted_objects = sorted(enumerate(self.base_objects), 
                               key=lambda x: x[1]["w"] * x[1]["h"], 
                               reverse=True)
        
        for orig_idx, obj in sorted_objects:
            w, h = float(obj["w"]), float(obj["h"])
            placed = False
            
            # Try multiple times to find a non-overlapping position
            for attempt in range(max_attempts):
                x = random.uniform(0, max(1.0, self.room_dims[0] - w))
                y = random.uniform(0, max(1.0, self.room_dims[1] - h))
                
                entry = dict(obj)
                entry.update({"x": x, "y": y, "w": w, "h": h})
                
                # Check if this position overlaps with existing objects
                overlaps = False
                for existing in layout:
                    if self._overlap(entry, existing):
                        overlaps = True
                        break
                    
                    # Also check minimum distance
                    dist = self._center_distance(entry, existing)
                    min_req = (w + existing["w"] + h + existing["h"]) / 4.0 + min_distance / 2.0
                    if dist < min_req:
                        overlaps = True
                        break
                
                if not overlaps:
                    layout.append(entry)
                    placed = True
                    break
            
            # If couldn't find position, use last attempt and will be repaired later
            if not placed:
                entry = dict(obj)
                entry.update({"x": x, "y": y, "w": w, "h": h})
                layout.append(entry)
        
        # Initial repair if needed
        if self._has_overlaps(layout):
            layout = self._repair_layout(layout)
        
        return layout

    def _mutate(self, layout, mutation_rate=0.3):
        new_layout = deepcopy(layout)
        for i in range(len(new_layout)):
            if random.random() < mutation_rate:
                # small gaussian perturbation
                dx = random.gauss(0, self.room_dims[0] * 0.05)
                dy = random.gauss(0, self.room_dims[1] * 0.05)
                new_layout[i]["x"] = min(max(0, new_layout[i]["x"] + dx), self.room_dims[0] - new_layout[i]["w"])
                new_layout[i]["y"] = min(max(0, new_layout[i]["y"] + dy), self.room_dims[1] - new_layout[i]["h"])
        
        # Repair if there are overlaps
        if self._has_overlaps(new_layout):
            new_layout = self._repair_layout(new_layout)
        
        return new_layout

    def _crossover(self, p1, p2):
        # Uniform crossover per-object
        child = []
        for a, b in zip(p1, p2):
            if random.random() < 0.5:
                child.append(deepcopy(a))
            else:
                child.append(deepcopy(b))
        return child

    def _tournament_select(self, population, scores, k=3):
        selected = []
        for _ in range(len(population)):
            aspirants = random.sample(list(range(len(population))), k)
            best_idx = max(aspirants, key=lambda idx: scores[idx])
            selected.append(deepcopy(population[best_idx]))
        return selected

    # --- Main GA loop ---
    def optimize(self):
        # initialize
        population = [self._random_layout() for _ in range(self.population_size)]
        best_so_far = None
        best_score = -1e9

        for gen in range(self.generations):
            scores = [self.fitness(ind) + self.compute_architectural_score(ind) for ind in population]
            # track best
            gen_best_idx = int(np.argmax(scores))
            if scores[gen_best_idx] > best_score:
                best_score = scores[gen_best_idx]
                best_so_far = deepcopy(population[gen_best_idx])

            # elitism: keep top 2
            idx_sorted = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            elites = [deepcopy(population[i]) for i in idx_sorted[:2]]

            # selection
            selected = self._tournament_select(population, scores, k=3)

            # produce children
            children = []
            while len(children) < self.population_size - len(elites):
                a, b = random.sample(selected, 2)
                child = self._crossover(a, b)
                child = self._mutate(child, mutation_rate=0.35)
                children.append(child)

            population = elites + children

        # final best cleanup: ensure integer-ish positions and in-bounds
        for obj in best_so_far:
            obj["x"] = float(max(0.0, min(self.room_dims[0] - obj["w"], obj["x"])))
            obj["y"] = float(max(0.0, min(self.room_dims[1] - obj["h"], obj["y"])))
        
        # Post-processing: STRICT overlap elimination
        repair_attempts = 0
        max_repair_attempts = 5
        
        while self._has_overlaps(best_so_far) and repair_attempts < max_repair_attempts:
            print(f"[Optimizer] Repairing overlaps (attempt {repair_attempts + 1}/{max_repair_attempts})...")
            best_so_far = self._repair_layout(best_so_far, aggressive=True)
            repair_attempts += 1
        
        # FINAL VALIDATION: Absolutely no overlaps allowed
        if self._has_overlaps(best_so_far):
            print("[Optimizer] WARNING: Final layout still has overlaps. Forcing grid-based placement.")
            best_so_far = self._grid_based_repair(best_so_far)
        
        # Triple-check: verify no overlaps
        if self._has_overlaps(best_so_far):
            raise Exception("CRITICAL ERROR: Failed to eliminate all overlaps. Please reduce number of objects or increase room size.")
        
        print("[Optimizer] Final layout validated: ZERO overlaps")
        return best_so_far

    def optimize_multiple(self, count: int = 3, runs: int = None):
        """
        Generate multiple alternative optimized layouts by varying RNG seeds.
        Returns a list of dicts: {"layout": <list>, "score": <float>} sorted by score desc.
        """
        k = max(1, int(count))
        runs = max(k, int(runs)) if runs is not None else k

        results = []
        signatures = set()

        # Base seed: use provided seed or a random base
        base_seed = self.seed if self.seed is not None else random.randint(0, 1_000_000)

        for i in range(runs):
            seed_i = base_seed + i + 1
            # Reseed RNGs to diversify runs
            random.seed(seed_i)
            np.random.seed(seed_i)

            layout = self.optimize()
            score = float(self.fitness(layout) + self.compute_architectural_score(layout))

            # Create a simple signature to avoid near-duplicates
            # Round positions to 1 decimal place and types/size to ints
            sig = tuple(
                (
                    obj.get("type", ""),
                    round(float(obj["x"]), 1),
                    round(float(obj["y"]), 1),
                    int(round(float(obj["w"]))),
                    int(round(float(obj["h"]))),
                )
                for obj in sorted(layout, key=lambda o: (o.get("type", ""), o.get("w", 0), o.get("h", 0)))
            )
            if sig in signatures:
                continue
            signatures.add(sig)
            results.append({"layout": layout, "score": score})

        # Sort by score (desc) and take top-k
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:k]

    def export_json(self, layout, filename="layout.json", metadata=None):
        payload = {"metadata": metadata or {}, "layout": layout}
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return filename 

if __name__ == "__main__":
    # Quick debug run for architectural scoring
    sample_layout = [
        {"type": "sofa", "x": 150.0, "y": 200.0, "w": 160.0, "h": 80.0},
        {"type": "tv", "x": 160.0, "y": 40.0, "w": 100.0, "h": 20.0},
        {"type": "mirror", "x": 180.0, "y": 140.0, "w": 60.0, "h": 10.0},
        {"type": "plant", "x": 40.0, "y": 90.0, "w": 30.0, "h": 30.0},
        {"type": "window", "x": 30.0, "y": 70.0, "w": 20.0, "h": 80.0}
    ]
    opt = LayoutOptimizer(room_dims=(400, 300), objects=[])
    s = opt.compute_architectural_score(sample_layout)
    print(" Test Architectural Score:", s)