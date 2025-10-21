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
from copy import deepcopy

import numpy as np


class LayoutOptimizer:
    def __init__(self, room_dims, objects, user_prefs=None, population_size=50, generations=200, seed=None):
        """
        :param room_dims: (width_cm, height_cm) tuple.
        :param objects: List of objects with keys: type, w, h (x,y optional).
        :param user_prefs: Dict of preferences (bed_near_wall, table_near_window, min_distance).
        :param population_size: GA population size.
        :param generations: Number of generations.
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
        score = 1000.0  # base
        # Out-of-bounds penalty
        for obj in layout:
            if not self._in_bounds(obj):
                score -= 1000.0

        # Overlap penalty
        for i in range(len(layout)):
            for j in range(i + 1, len(layout)):
                if self._overlap(layout[i], layout[j]):
                    score -= 1500.0

        # Min distance penalty (clearance)
        min_distance = float(self.user_prefs.get("min_distance", 20.0))
        for i in range(len(layout)):
            for j in range(i + 1, len(layout)):
                dist = self._center_distance(layout[i], layout[j])
                allowed = (min_distance + layout[i]["w"] / 2.0 + layout[j]["w"] / 2.0) / 2.0
                if dist < allowed:
                    # quadratic penalty for being too close
                    score -= (allowed - dist) ** 2

        # User prefs rewards
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
        score += self._spread_score(layout) * 0.5
        return score

    def _overlap(self, a, b):
        return not (a["x"] + a["w"] <= b["x"] or b["x"] + b["w"] <= a["x"] or a["y"] + a["h"] <= b["y"] or b["y"] + b["h"] <= a["y"])

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

    # --- Initialization / genetic operators ---
    def _random_layout(self):
        layout = []
        for obj in self.base_objects:
            w, h = float(obj["w"]), float(obj["h"])
            x = random.uniform(0, max(0.0, self.room_dims[0] - w))
            y = random.uniform(0, max(0.0, self.room_dims[1] - h))
            entry = dict(obj)
            entry.update({"x": x, "y": y, "w": w, "h": h})
            layout.append(entry)
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
            scores = [self.fitness(ind) for ind in population]
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
        return best_so_far

    def export_json(self, layout, filename="layout.json", metadata=None):
        payload = {"metadata": metadata or {}, "layout": layout}
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return filename 