"""
Bootstrap rule memory by simulating synthetic rooms and running the optimizer.
This pre-fills data/rule_memory.json so the optimizer starts with informed rule weights.

Usage:
  python scripts/bootstrap_rules.py --scenarios 500 --generations 80 --population 120

Notes:
- Uses small generations by default to keep it fast. Increase for higher quality.
- Progress is printed; you can stop anytime, memory persists incrementally.
"""
import argparse
import os
import random
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm

# Local imports from project root
from optimizer import LayoutOptimizer


FURNITURE_POOLS = {
    "living": [
        {"type": "sofa", "w": 160, "h": 90},
        {"type": "tv", "w": 120, "h": 40},
        {"type": "coffee table", "w": 100, "h": 60},
        {"type": "lamp", "w": 30, "h": 30},
        {"type": "plant", "w": 40, "h": 40},
        {"type": "large mirror", "w": 120, "h": 10},
        {"type": "chair", "w": 50, "h": 50},
        {"type": "table", "w": 120, "h": 60},
    ],
    "bedroom": [
        {"type": "bed", "w": 200, "h": 150},
        {"type": "wardrobe", "w": 180, "h": 60},
        {"type": "nightstand", "w": 50, "h": 40},
        {"type": "lamp", "w": 25, "h": 25},
        {"type": "chair", "w": 50, "h": 50},
        {"type": "mirror", "w": 80, "h": 10},
        {"type": "desk", "w": 140, "h": 70},
    ],
    "office": [
        {"type": "desk", "w": 140, "h": 70},
        {"type": "office chair", "w": 60, "h": 60},
        {"type": "bookshelf", "w": 80, "h": 30},
        {"type": "plant", "w": 30, "h": 30},
        {"type": "tv", "w": 120, "h": 40},
        {"type": "lamp", "w": 25, "h": 25},
    ],
}


def rand_room_dims() -> Tuple[float, float]:
    w = random.choice([320, 360, 400, 450, 500])
    h = random.choice([260, 300, 320, 360, 400])
    return float(w), float(h)


def rand_arch_elements(room_dims: Tuple[float, float]) -> List[Dict]:
    """Return fixed architectural elements to inject as detections if desired.
    We emulate windows/doors as rectangles near perimeter that the GA must respect.
    (Optimizer already respects architectural elements indirectly via rules; this is optional.)
    """
    w, h = room_dims
    elems: List[Dict] = []
    # 0-2 windows
    for _ in range(random.randint(0, 2)):
        ww = random.uniform(60, 140)
        wh = random.uniform(30, 90)
        x = random.uniform(10, max(10, w - ww - 10))
        y = random.choice([5.0, h - wh - 5.0])
        elems.append({"type": "window", "x": x, "y": y, "w": ww, "h": wh, "fixed": True, "architectural": True})
    # 0-1 door
    if random.random() < 0.8:
        dw = random.uniform(70, 100)
        dh = random.uniform(170, 210)
        x = random.choice([5.0, w - dw - 5.0])
        y = random.uniform(5.0, max(5.0, h - dh - 5.0))
        elems.append({"type": "door", "x": x, "y": y, "w": dw, "h": dh, "fixed": True, "architectural": True})
    return elems


def sample_furniture(kind: str) -> List[Dict]:
    pool = FURNITURE_POOLS[kind]
    k = random.randint(3, min(7, len(pool)))
    picks = random.sample(pool, k)
    # shallow copies
    return [{**p} for p in picks]


def prepare_objects(furn: List[Dict]) -> List[Dict]:
    # Ensure required keys; positions will be optimized
    out = []
    for p in furn:
        out.append({
            "type": p["type"],
            "w": float(p["w"]),
            "h": float(p["h"]),
            "x": 0.0,
            "y": 0.0,
        })
    return out


def run_scenario(room_dims: Tuple[float, float], objects: List[Dict], generations: int, population: int) -> None:
    # User prefs minimal; rules engine does most heavy lifting now
    user_prefs = {"min_distance": 20.0, "bed_near_wall": True, "table_near_window": True}
    opt = LayoutOptimizer(room_dims=room_dims, objects=objects, user_prefs=user_prefs,
                          generations=generations, population_size=population)
    # Running optimize() triggers scoring and learner updates internally
    try:
        _ = opt.optimize()
    except Exception:
        # Even if a scenario fails to converge, continue; memory already accumulated for many evaluations
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, default=500)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=120)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs("data", exist_ok=True)

    kinds = list(FURNITURE_POOLS.keys())

    for _ in tqdm(range(args.scenarios), desc="Bootstrapping rules"):
        kind = random.choice(kinds)
        room = rand_room_dims()
        furn = sample_furniture(kind)
        objects = prepare_objects(furn)
        run_scenario(room, objects, generations=args.generations, population=args.population)

    print("\nBootstrap complete. Rule memory saved at data/rule_memory.json")


if __name__ == "__main__":
    main()
