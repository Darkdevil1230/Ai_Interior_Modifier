"""
Train YOLOv8 segmentation on the custom dataset and export:
- weights/best.pt (copied from latest exp)
- weights/class_map.json (derived from data.yaml names order)
"""
import os
import json
import shutil
from glob import glob
from typing import List

import yaml
from ultralytics import YOLO


def _latest_exp_weights() -> str:
    """Return path to latest runs/train/exp*/weights/best.pt if exists."""
    cand = sorted(glob(os.path.join("runs", "train", "exp*", "weights", "best.pt")), key=os.path.getmtime)
    return cand[-1] if cand else ""


def _write_class_map_from_yaml(data_yaml: str, out_json: str) -> None:
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names: List[str] = cfg.get("names") or []
    if isinstance(names, dict):
        # Some exports use dict {0:"class",1:"class"}
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    mapping = {str(i): str(n).lower() for i, n in enumerate(names)}
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"📝 Wrote class map JSON -> {out_json}")


def train_yolo(data_yaml: str = "data/data.yaml", epochs: int = 100, imgsz: int = 960, batch: int = 16, model_name: str = "yolov8s-seg.pt"):
    """Train a segmentation model suitable for architectural elements.
    - Use a -seg backbone for masks (better geometry than boxes)
    - Exports weights to weights/best.pt and class_map.json next to it
    """
    os.makedirs("weights", exist_ok=True)

    model = YOLO(model_name)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=0.01,
        cos_lr=True,
        patience=30,
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        device=0 if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else None,
    )
    # Export to PyTorch .pt to ensure compatibility
    model.export(format="pt")

    best_src = _latest_exp_weights()
    if best_src and os.path.exists(best_src):
        dst = os.path.join("weights", "best.pt")
        shutil.copy2(best_src, dst)
        print(f"✅ Copied best weights -> {dst}")
    else:
        print("⚠ Could not locate best.pt in runs/train/exp*/weights/")

    # Create class_map.json based on data.yaml names order
    _write_class_map_from_yaml(data_yaml, os.path.join("weights", "class_map.json"))
    print("✅ Training complete.")


if __name__ == "__main__":
    os.makedirs("weights", exist_ok=True)
    # Example: train_yolo()  # uses defaults
    train_yolo()