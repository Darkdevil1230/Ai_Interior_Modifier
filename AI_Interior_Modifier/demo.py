"""
AI Interior Modifier - Demo Script
==================================
Demonstrates how to use the system programmatically without the UI.
"""

import tempfile
import json
from PIL import Image
import numpy as np

from src.input.cv_detector import RoomDetector
from optimizer import LayoutOptimizer
from plot2d import plot_layout

def demo_with_image(image_path=None):
    """Demo using a real image."""
    print("🏠 AI Interior Modifier - Programmatic Demo")
    print("=" * 50)
    
    # Create a sample room image if none provided
    if image_path is None:
        print("📸 Creating sample room image...")
        # Create a simple room-like image
        room_img = Image.new('RGB', (800, 600), color='lightblue')
        
        # Add some simple furniture-like shapes
        from PIL import ImageDraw
        draw = ImageDraw.Draw(room_img)
        
        # Draw a bed-like rectangle
        draw.rectangle([100, 200, 300, 350], fill='brown', outline='black', width=2)
        draw.text((200, 275), "BED", fill='white', anchor='mm')
        
        # Draw a table-like rectangle  
        draw.rectangle([500, 100, 650, 200], fill='tan', outline='black', width=2)
        draw.text((575, 150), "TABLE", fill='black', anchor='mm')
        
        # Draw a chair-like rectangle
        draw.rectangle([400, 300, 450, 400], fill='darkgreen', outline='black', width=2)
        draw.text((425, 350), "CHAIR", fill='white', anchor='mm')
        
        # Save temporary image
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        room_img.save(tmp.name)
        image_path = tmp.name
        print(f"✅ Sample image created: {image_path}")
    
    # Step 1: Detection
    print("\n🔍 Step 1: Detecting furniture...")
    detector = RoomDetector()
    detections = detector.detect(image_path, conf_threshold=0.1)
    
    print(f"   Model: {detector.get_model_name()}")
    print(f"   Detections: {len(detections)}")
    
    for i, det in enumerate(detections):
        label = det.get("label", "unknown")
        conf = det["confidence"]
        print(f"   {i+1}. {label} (confidence: {conf:.2f})")
    
    # Step 2: Parse detections
    print("\n📐 Step 2: Parsing detections...")
    room_dims = (400, 300)  # 4m x 3m room
    image_shape = (600, 800)  # height, width
    
    objects = detector.parse_detections(detections, {}, room_dims, image_shape)
    
    # If no detections, use default furniture
    if not objects:
        print("   No detections found, using default furniture...")
        objects = [
            {"type": "bed", "w": 200, "h": 150},
            {"type": "table", "w": 120, "h": 80},
            {"type": "chair", "w": 50, "h": 50}
        ]
    
    print(f"   Parsed {len(objects)} furniture pieces:")
    for obj in objects:
        print(f"   - {obj['type']}: {obj['w']}x{obj['h']} cm")
    
    # Step 3: Optimization
    print("\n🧬 Step 3: Optimizing layout...")
    ga = LayoutOptimizer(
        room_dims=room_dims,
        objects=objects,
        user_prefs={
            "bed_near_wall": True,
            "table_near_window": True,
            "min_distance": 20
        },
        population_size=50,
        generations=100,
        seed=42
    )
    
    layout = ga.optimize()
    print(f"   ✅ Optimization complete!")
    print(f"   Optimized {len(layout)} objects")
    
    # Step 4: Visualization
    print("\n📊 Step 4: Generating visualization...")
    buf = plot_layout(room_dims, layout, save_buffer=True)
    print("   ✅ Visualization generated")
    
    # Step 5: Export results
    print("\n💾 Step 5: Exporting results...")
    result = {
        "metadata": {
            "room_width_cm": room_dims[0],
            "room_height_cm": room_dims[1],
            "model": detector.get_model_name(),
            "optimization_method": "genetic_algorithm",
            "furniture_count": len(layout)
        },
        "layout": layout
    }
    
    # Save JSON
    with open("demo_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("   ✅ Results saved to demo_result.json")
    
    # Display layout
    print("\n📐 Optimized Layout:")
    print("-" * 30)
    for i, obj in enumerate(layout):
        print(f"{i+1}. {obj['type'].upper()}")
        print(f"   Position: ({obj['x']:.1f}, {obj['y']:.1f}) cm")
        print(f"   Size: {obj['w']:.1f} x {obj['h']:.1f} cm")
        print()
    
    # Calculate metrics
    room_area = room_dims[0] * room_dims[1]
    furniture_area = sum(obj["w"] * obj["h"] for obj in layout)
    coverage = (furniture_area / room_area) * 100
    
    print("📊 Room Metrics:")
    print(f"   Room Area: {room_area/10000:.1f} m²")
    print(f"   Furniture Area: {furniture_area/10000:.1f} m²") 
    print(f"   Coverage: {coverage:.1f}%")
    print(f"   Free Space: {(room_area - furniture_area)/10000:.1f} m²")
    
    print("\n🎉 Demo complete! Check demo_result.json for detailed results.")
    return result

def demo_without_image():
    """Demo without image detection."""
    print("🏠 AI Interior Modifier - Quick Demo (No Image)")
    print("=" * 50)
    
    # Define furniture
    objects = [
        {"type": "bed", "w": 200, "h": 150},
        {"type": "sofa", "w": 180, "h": 80},
        {"type": "table", "w": 120, "h": 80},
        {"type": "chair", "w": 50, "h": 50},
        {"type": "chair", "w": 50, "h": 50}
    ]
    
    room_dims = (500, 400)  # 5m x 4m room
    
    print(f"📐 Room: {room_dims[0]}x{room_dims[1]} cm")
    print(f"🪑 Furniture: {len(objects)} pieces")
    
    # Optimize
    ga = LayoutOptimizer(
        room_dims=room_dims,
        objects=objects,
        user_prefs={
            "bed_near_wall": True,
            "table_near_window": True,
            "min_distance": 30
        },
        population_size=100,
        generations=200,
        seed=123
    )
    
    print("🧬 Optimizing...")
    layout = ga.optimize()
    
    # Results
    print("✅ Optimization complete!")
    print("\n📐 Final Layout:")
    for i, obj in enumerate(layout):
        print(f"{i+1}. {obj['type'].upper()}: ({obj['x']:.0f}, {obj['y']:.0f}) - {obj['w']:.0f}x{obj['h']:.0f}cm")
    
    return layout

def main():
    """Run demos."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--no-image":
        demo_without_image()
    else:
        demo_with_image()

if __name__ == "__main__":
    main()
