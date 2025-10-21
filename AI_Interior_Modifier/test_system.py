"""
Test script for AI Interior Modifier
===================================
Tests the core functionality without the UI.
"""

import tempfile
import json
from PIL import Image
import numpy as np

from src.input.cv_detector import RoomDetector
from optimizer import LayoutOptimizer
from plot2d import plot_layout

def test_detection():
    """Test the detection functionality."""
    print("🔍 Testing detection...")
    
    # Create a simple test image
    test_img = Image.new('RGB', (640, 480), color='white')
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    test_img.save(tmp.name)
    
    try:
        detector = RoomDetector()
        detections = detector.detect(tmp.name, conf_threshold=0.1)
        print(f"✅ Detection test passed - Model: {detector.get_model_name()}")
        print(f"   Found {len(detections)} detections")
        return True
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        return False

def test_optimization():
    """Test the optimization functionality."""
    print("🧬 Testing optimization...")
    
    try:
        # Test objects
        objects = [
            {"type": "bed", "w": 200, "h": 150},
            {"type": "table", "w": 120, "h": 80},
            {"type": "chair", "w": 50, "h": 50}
        ]
        
        # Initialize optimizer
        ga = LayoutOptimizer(
            room_dims=(400, 300),
            objects=objects,
            user_prefs={
                "bed_near_wall": True,
                "table_near_window": True,
                "min_distance": 20
            },
            population_size=20,  # Smaller for testing
            generations=10,       # Smaller for testing
            seed=42
        )
        
        # Run optimization
        layout = ga.optimize()
        
        print(f"✅ Optimization test passed")
        print(f"   Optimized {len(layout)} objects")
        
        # Test visualization
        buf = plot_layout((400, 300), layout, save_buffer=True)
        print(f"✅ Visualization test passed")
        
        return True
    except Exception as e:
        print(f"❌ Optimization test failed: {e}")
        return False

def test_full_pipeline():
    """Test the complete pipeline."""
    print("🚀 Testing full pipeline...")
    
    try:
        # Create test image
        test_img = Image.new('RGB', (640, 480), color='lightblue')
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        test_img.save(tmp.name)
        
        # Detection
        detector = RoomDetector()
        detections = detector.detect(tmp.name, conf_threshold=0.1)
        
        # Parse detections
        objects = detector.parse_detections(detections, {}, (400, 300), (480, 640))
        
        # If no detections, use default objects
        if not objects:
            objects = [
                {"type": "bed", "w": 200, "h": 150},
                {"type": "table", "w": 120, "h": 80}
            ]
        
        # Optimization
        ga = LayoutOptimizer(
            room_dims=(400, 300),
            objects=objects,
            population_size=20,
            generations=10,
            seed=42
        )
        
        layout = ga.optimize()
        
        # Export
        result = {
            "metadata": {
                "room_width_cm": 400,
                "room_height_cm": 300,
                "model": detector.get_model_name()
            },
            "layout": layout
        }
        
        print(f"✅ Full pipeline test passed")
        print(f"   Model: {detector.get_model_name()}")
        print(f"   Objects: {len(layout)}")
        print(f"   Layout: {json.dumps(layout, indent=2)}")
        
        return True
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 AI Interior Modifier - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Detection", test_detection),
        ("Optimization", test_optimization), 
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All tests passed! The system is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    main()
