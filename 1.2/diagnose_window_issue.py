"""
Diagnostic script to check why windows aren't showing in the layout.
This will trace the entire flow from detection to visualization.
"""

import sys
import os
from src.input.cv_detector import RoomDetector
from src.input.architectural_detector import enhance_detections_with_architecture

def diagnose_window_detection(image_path: str):
    """Diagnose window detection and visualization issues."""
    
    print("=" * 70)
    print("WINDOW DETECTION DIAGNOSTIC")
    print("=" * 70)
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Step 1: YOLO Detection
    print("\n📍 STEP 1: YOLO Detection")
    print("-" * 70)
    detector = RoomDetector()
    yolo_detections = detector.detect(image_path, conf_threshold=0.35)
    
    print(f"✓ Total YOLO detections: {len(yolo_detections)}")
    
    # Categorize
    windows_yolo = [d for d in yolo_detections if d.get("type") == "window"]
    doors_yolo = [d for d in yolo_detections if d.get("type") == "door"]
    walls_yolo = [d for d in yolo_detections if d.get("type") == "wall"]
    furniture_yolo = [d for d in yolo_detections if d.get("category") not in ["architectural", None]]
    
    print(f"  - Windows from YOLO: {len(windows_yolo)}")
    print(f"  - Doors from YOLO: {len(doors_yolo)}")
    print(f"  - Walls from YOLO: {len(walls_yolo)}")
    print(f"  - Furniture from YOLO: {len(furniture_yolo)}")
    
    # Step 2: Check if fallback was triggered
    print("\n📍 STEP 2: Checking Enhanced Detections")
    print("-" * 70)
    
    has_arch_before = len(windows_yolo) + len(doors_yolo) + len(walls_yolo)
    print(f"Architectural elements before enhancement: {has_arch_before}")
    
    if has_arch_before == 0:
        print("✓ Fallback detector should have been triggered!")
        print("  (YOLO had no architectural elements)")
    else:
        print("⚠️ YOLO found some architectural elements")
        print("  Fallback might have been skipped")
    
    # Step 3: Check final detections
    print("\n📍 STEP 3: Final Detection Results")
    print("-" * 70)
    
    all_detections = yolo_detections  # This should include fallback results
    
    windows_final = [d for d in all_detections if d.get("type") == "window"]
    doors_final = [d for d in all_detections if d.get("type") == "door"]
    walls_final = [d for d in all_detections if d.get("type") == "wall"]
    
    print(f"✓ Total final detections: {len(all_detections)}")
    print(f"  - Windows: {len(windows_final)}")
    print(f"  - Doors: {len(doors_final)}")
    print(f"  - Walls: {len(walls_final)}")
    
    if windows_final:
        print("\n  Window Details:")
        for i, window in enumerate(windows_final):
            bbox = window.get("bbox", [0,0,0,0])
            conf = window.get("confidence", 0)
            category = window.get("category", "unknown")
            architectural = window.get("architectural", False)
            fixed = window.get("fixed", False)
            print(f"    Window {i+1}:")
            print(f"      - BBox: {bbox}")
            print(f"      - Confidence: {conf:.2f}")
            print(f"      - Category: {category}")
            print(f"      - Architectural flag: {architectural}")
            print(f"      - Fixed flag: {fixed}")
    
    # Step 4: Check what would be passed to optimizer
    print("\n📍 STEP 4: What Optimizer Would Receive")
    print("-" * 70)
    
    # Simulate room_analysis structure
    room_analysis = {
        "detections": all_detections,
        "room_dims": (400, 300)  # Example
    }
    
    arch_in_analysis = [d for d in room_analysis["detections"] 
                        if d.get("category") == "architectural" or 
                        d.get("type", "").lower() in ["window", "door", "wall"]]
    
    print(f"Architectural elements in room_analysis: {len(arch_in_analysis)}")
    for elem in arch_in_analysis:
        print(f"  - {elem.get('type', 'unknown')}: category={elem.get('category')}, "
              f"architectural={elem.get('architectural', False)}, "
              f"fixed={elem.get('fixed', False)}")
    
    # Step 5: Recommendations
    print("\n📍 STEP 5: Diagnosis & Recommendations")
    print("=" * 70)
    
    if len(windows_final) == 0:
        print("❌ NO WINDOWS DETECTED")
        print("\nRecommendations:")
        print("  1. Check if fallback detector was called (see console above)")
        print("  2. Try lowering thresholds in architectural_detector.py:")
        print("     self.min_window_area = 500")
        print("  3. Verify image has visible windows")
        print("  4. Check image brightness/contrast")
    elif len(arch_in_analysis) == 0:
        print("⚠️ WINDOWS DETECTED but not flagged as 'architectural'")
        print("\nRecommendations:")
        print("  1. Check that detected windows have 'architectural': True")
        print("  2. Verify category is set to 'architectural'")
        print("  3. May need to fix flag assignment in detector")
    else:
        print("✅ WINDOWS DETECTED and properly flagged")
        print(f"   {len(windows_final)} window(s) should appear in layout")
        print("\nIf windows still don't show in visualization:")
        print("  1. Check console output from layout generator")
        print("  2. Verify architectural_elements are passed to visualizer")
        print("  3. Run app and check for these log messages:")
        print("     '[LayoutGenerator] Received X architectural elements'")
        print("     '[LayoutGenerator] Drawing Y architectural elements'")
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    # Update this path to your room image
    image_path = "c:\\Users\\Sai talluri\\Downloads\\img1.jpg"
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Usage: python diagnose_window_issue.py <image_path>")
        print(f"\nExample:")
        print(f"  python diagnose_window_issue.py room.jpg")
        sys.exit(1)
    
    diagnose_window_detection(image_path)
