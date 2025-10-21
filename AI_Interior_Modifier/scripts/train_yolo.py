"""
Train YOLOv8 on the custom dataset.
"""
import os
from ultralytics import YOLO

def train_yolo(data_yaml="data/data.yaml", epochs=50):
    model = YOLO("yolov8n.pt")
    model.train(data=data_yaml, epochs=epochs, imgsz=640, batch=16)
    model.export(format="pt")
    print("✅ Training complete. Weights saved in runs/train/exp/weights/best.pt")

if __name__ == "__main__":
    os.makedirs("weights", exist_ok=True)
    train_yolo()