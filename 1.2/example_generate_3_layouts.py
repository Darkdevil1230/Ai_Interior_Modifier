"""
Example: Generate 3 Layout Options with Architectural Elements
================================================================
This script demonstrates:
1. All furniture items guaranteed to be placed
2. Generate 3 alternative layouts
3. Include architectural elements (windows, doors, walls)
"""

import json
from src.optimization.cnn_guided_optimizer import CNNGuidedOptimizer
from optimizer import LayoutOptimizer

# Example 1: Using CNNGuidedOptimizer with architectural detection
def example_cnn_guided():
    print("=" * 60)
    print("Example 1: CNN-Guided Optimizer with Architectural Elements")
    print("=" * 60)
    
    # Sample furniture items
    furniture = [
        {"type": "bed", "w": 200, "h": 180},
        {"type": "nightstand", "w": 50, "h": 40},
        {"type": "dresser", "w": 120, "h": 50},
        {"type": "wardrobe", "w": 150, "h": 60},
        {"type": "desk", "w": 120, "h": 60},
        {"type": "chair", "w": 50, "h": 50},
    ]
    
    # Mock CNN analysis with detected architectural elements
    room_analysis = {
        "detections": [
            {
                "type": "window",
                "bbox": [0, 50, 10, 150],  # Left wall window
                "confidence": 0.85
            },
            {
                "type": "door",
                "bbox": [350, 0, 400, 80],  # Door on right wall
                "confidence": 0.92
            },
            {
                "type": "window",
                "bbox": [100, 290, 200, 300],  # Bottom wall window
                "confidence": 0.78
            }
        ],
        "recommendations": {
            "furniture_placement_zones": [],
            "traffic_flow": {"pathways": [], "clearance_required": 60},
            "lighting_considerations": {"natural_light": "medium"}
        },
        "room_metrics": {"area": 120000}
    }
    
    # Create optimizer
    optimizer = CNNGuidedOptimizer(
        room_dims=(400, 300),
        objects=furniture,
        room_analysis=room_analysis,
        population_size=100,
        generations=200,
        seed=42
    )
    
    # Generate 3 alternative layouts
    print("\n🔄 Generating 3 alternative layouts...")
    results = optimizer.optimize_multiple(count=3)
    
    # Display results
    print("\n✅ Generation complete!\n")
    for i, result in enumerate(results):
        layout = result["layout"]
        score = result["score"]
        seed = result["seed"]
        
        # Separate furniture from architectural elements
        furniture_items = [obj for obj in layout if not obj.get("architectural", False)]
        arch_items = [obj for obj in layout if obj.get("architectural", False)]
        
        print(f"📐 Layout Option {i+1} (Score: {score:.2f}, Seed: {seed})")
        print(f"   - Furniture: {len(furniture_items)} items")
        print(f"   - Architectural: {len(arch_items)} elements (fixed)")
        
        # Show furniture types
        furniture_types = [obj.get("type") for obj in furniture_items]
        print(f"   - Items: {', '.join(furniture_types)}")
        
        # Show architectural elements
        arch_types = [f"{obj.get('type')} (fixed)" for obj in arch_items]
        if arch_types:
            print(f"   - Fixed: {', '.join(arch_types)}")
        print()
    
    # Save best layout
    best_layout = results[0]["layout"]
    with open("best_layout_cnn.json", "w") as f:
        json.dump({
            "room_dimensions": [400, 300],
            "layout": best_layout,
            "score": results[0]["score"],
            "seed": results[0]["seed"]
        }, f, indent=2)
    print("💾 Best layout saved to: best_layout_cnn.json\n")
    
    return results


# Example 2: Using classic LayoutOptimizer
def example_classic_optimizer():
    print("=" * 60)
    print("Example 2: Classic Optimizer (No Architectural Detection)")
    print("=" * 60)
    
    # Sample furniture items - living room
    furniture = [
        {"type": "sofa", "w": 200, "h": 90},
        {"type": "coffee table", "w": 100, "h": 60},
        {"type": "tv", "w": 120, "h": 15},
        {"type": "tv stand", "w": 140, "h": 40},
        {"type": "chair", "w": 70, "h": 70},
        {"type": "chair", "w": 70, "h": 70},
    ]
    
    # Create optimizer
    optimizer = LayoutOptimizer(
        room_dims=(400, 300),
        objects=furniture,
        user_prefs={
            "bed_near_wall": False,
            "table_near_window": True,
            "min_distance": 20
        },
        population_size=100,
        generations=200,
        seed=42
    )
    
    # Generate 3 alternative layouts
    print("\n🔄 Generating 3 alternative layouts...")
    results = optimizer.optimize_multiple(count=3, runs=5)
    
    # Display results
    print("\n✅ Generation complete!\n")
    for i, result in enumerate(results):
        layout = result["layout"]
        score = result["score"]
        
        print(f"📐 Layout Option {i+1} (Score: {score:.2f})")
        print(f"   - Furniture: {len(layout)} items")
        
        # Show furniture types
        furniture_types = [obj.get("type") for obj in layout]
        print(f"   - Items: {', '.join(furniture_types)}")
        print()
    
    # Save best layout
    best_layout = results[0]["layout"]
    with open("best_layout_classic.json", "w") as f:
        json.dump({
            "room_dimensions": [400, 300],
            "layout": best_layout,
            "score": results[0]["score"]
        }, f, indent=2)
    print("💾 Best layout saved to: best_layout_classic.json\n")
    
    return results


# Example 3: Verify all furniture was placed
def verify_all_furniture_placed(furniture, layout):
    print("=" * 60)
    print("Verification: All Furniture Placed?")
    print("=" * 60)
    
    # Filter out architectural elements
    furniture_items = [obj for obj in layout if not obj.get("architectural", False)]
    
    expected_count = len(furniture)
    actual_count = len(furniture_items)
    
    print(f"\nExpected furniture: {expected_count}")
    print(f"Placed furniture: {actual_count}")
    
    if actual_count == expected_count:
        print("✅ SUCCESS: All furniture items placed!")
    else:
        print(f"❌ WARNING: {expected_count - actual_count} items missing!")
        
        # Show what's missing
        expected_types = [obj.get("type") for obj in furniture]
        placed_types = [obj.get("type") for obj in furniture_items]
        
        expected_set = set(expected_types)
        placed_set = set(placed_types)
        missing = expected_set - placed_set
        
        if missing:
            print(f"   Missing types: {missing}")
    
    print()


if __name__ == "__main__":
    print("\n🏠 Layout Generation Examples\n")
    
    # Run Example 1: CNN-Guided with Architectural Elements
    results_cnn = example_cnn_guided()
    
    # Verify all furniture placed
    furniture_cnn = [
        {"type": "bed", "w": 200, "h": 180},
        {"type": "nightstand", "w": 50, "h": 40},
        {"type": "dresser", "w": 120, "h": 50},
        {"type": "wardrobe", "w": 150, "h": 60},
        {"type": "desk", "w": 120, "h": 60},
        {"type": "chair", "w": 50, "h": 50},
    ]
    verify_all_furniture_placed(furniture_cnn, results_cnn[0]["layout"])
    
    # Run Example 2: Classic Optimizer
    results_classic = example_classic_optimizer()
    
    # Verify all furniture placed
    furniture_classic = [
        {"type": "sofa", "w": 200, "h": 90},
        {"type": "coffee table", "w": 100, "h": 60},
        {"type": "tv", "w": 120, "h": 15},
        {"type": "tv stand", "w": 140, "h": 40},
        {"type": "chair", "w": 70, "h": 70},
        {"type": "chair", "w": 70, "h": 70},
    ]
    verify_all_furniture_placed(furniture_classic, results_classic[0]["layout"])
    
    print("=" * 60)
    print("All examples complete! Check JSON files for detailed output.")
    print("=" * 60)
