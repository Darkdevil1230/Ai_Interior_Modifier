"""
Quick test script to verify window detection on your image.
Save your room image and update the path below.
"""

import cv2
from src.input.architectural_detector import ArchitecturalDetector
from src.input.cv_detector import RoomDetector

def test_detection(image_path: str):
    """Test both YOLO and fallback detection on an image."""
    
    print("=" * 60)
    print(f"Testing detection on: {image_path}")
    print("=" * 60)
    
    # Test 1: YOLO Detection
    print("\n1️⃣ Testing YOLO Detection...")
    detector = RoomDetector()
    yolo_detections = detector.detect(image_path, conf_threshold=0.35)
    
    yolo_windows = [d for d in yolo_detections if d.get("type") == "window"]
    yolo_doors = [d for d in yolo_detections if d.get("type") == "door"]
    yolo_furniture = [d for d in yolo_detections if d.get("category") != "architectural"]
    
    print(f"   Windows: {len(yolo_windows)}")
    print(f"   Doors: {len(yolo_doors)}")
    print(f"   Furniture: {len(yolo_furniture)}")
    
    # Test 2: Fallback Architectural Detection
    print("\n2️⃣ Testing Fallback Architectural Detection...")
    arch_detector = ArchitecturalDetector()
    arch_detections = arch_detector.detect_architectural_elements(image_path)
    
    windows = [d for d in arch_detections if d.get("type") == "window"]
    doors = [d for d in arch_detections if d.get("type") == "door"]
    walls = [d for d in arch_detections if d.get("type") == "wall"]
    
    print(f"   Windows: {len(windows)}")
    print(f"   Doors: {len(doors)}")
    print(f"   Walls: {len(walls)}")
    
    # Show window details
    if windows:
        print("\n📐 Window Details:")
        for i, window in enumerate(windows):
            bbox = window["bbox"]
            conf = window["confidence"]
            print(f"   Window {i+1}: bbox={bbox}, confidence={conf:.2f}")
    else:
        print("\n⚠️ No windows detected!")
    
    # Test 3: Visualize detections
    print("\n3️⃣ Creating visualization...")
    img = cv2.imread(image_path)
    if img is not None:
        # Draw windows
        for window in windows:
            x1, y1, x2, y2 = window["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue for windows
            cv2.putText(img, "WINDOW", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw doors
        for door in doors:
            x1, y1, x2, y2 = door["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green for doors
            cv2.putText(img, "DOOR", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Save result
        output_path = "detection_result.jpg"
        cv2.imwrite(output_path, img)
        print(f"   ✅ Saved visualization to: {output_path}")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"   Total detections: {len(yolo_detections) + len(arch_detections)}")
    print(f"   Windows found: {len(windows)} ✅" if windows else "   Windows found: 0 ❌")
    print(f"   Doors found: {len(doors)}")
    print(f"   Walls found: {len(walls)}")
    print("=" * 60)


if __name__ == "__main__":
    # UPDATE THIS PATH to your room image
    image_path = "c:\\Users\\Sai talluri\\Downloads\\img1.jpg"  # ⬅️ CHANGE THIS
    
    import sys
    import os
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("\nUsage:")
        print(f"  1. Save your room image")
        print(f"  2. Update image_path in this script")
        print(f"  3. Run: python test_window_detection.py")
        sys.exit(1)
    
    # Run test
    test_detection(image_path)
    
    print("\n💡 If windows were detected, check 'detection_result.jpg' to see them marked!")
    print("💡 If not detected, try adjusting thresholds in architectural_detector.py")
