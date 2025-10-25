"""
test_enhanced_system.py
-----------------------
Test script to verify the enhanced AI Room Optimizer system works correctly.
"""

import os
import sys
import numpy as np
from PIL import Image
import tempfile

# Add src to path
sys.path.append('src')

def create_test_room_image(width=400, height=300):
    """Create a simple test room image."""
    # Create a simple room image with basic elements
    img = Image.new('RGB', (width, height), color='white')
    
    # Add some basic room elements (simplified)
    # This is just for testing - in real usage, you'd upload actual room photos
    return img

def test_cnn_architectural_detector():
    """Test the CNN architectural detector."""
    print("Testing CNN Architectural Detector...")
    
    try:
        from src.input.cnn_architectural_detector import CNNArchitecturalDetector
        
        # Create test image
        test_img = create_test_room_image()
        
        # Save temporary image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            test_img.save(tmp.name)
            tmp_path = tmp.name
        
        # Initialize detector
        detector = CNNArchitecturalDetector(device="cpu")
        
        # Test detection
        detections = detector.detect_architectural_elements(tmp_path)
        print(f"✓ Detected {len(detections)} architectural elements")
        
        # Test room analysis
        room_analysis = detector.analyze_room_layout(tmp_path)
        print(f"✓ Room analysis completed")
        
        # Clean up
        os.unlink(tmp_path)
        
        return True
        
    except Exception as e:
        print(f"✗ CNN Architectural Detector test failed: {e}")
        return False

def test_cnn_guided_optimizer():
    """Test the CNN-guided optimizer."""
    print("Testing CNN-Guided Optimizer...")
    
    try:
        from src.optimization.cnn_guided_optimizer import CNNGuidedOptimizer
        
        # Create test furniture
        furniture = [
            {"name": "Bed", "width": 200, "depth": 150},
            {"name": "Sofa", "width": 150, "depth": 80},
            {"name": "Table", "width": 120, "depth": 60}
        ]
        
        # Create mock room analysis
        room_analysis = {
            "detections": [
                {"type": "window", "bbox": [50, 50, 150, 100], "confidence": 0.8},
                {"type": "door", "bbox": [300, 50, 350, 200], "confidence": 0.7}
            ],
            "recommendations": {
                "furniture_placement_zones": [
                    {"type": "window_zone", "center": (100, 75), "size": (100, 50), "suitable_for": ["chair", "desk"]},
                    {"type": "wall_zone", "center": (200, 150), "size": (100, 200), "suitable_for": ["bed", "sofa"]}
                ],
                "traffic_flow": {"pathways": [], "clearance_required": 60},
                "lighting_considerations": {"natural_light": "medium"}
            }
        }
        
        # Initialize optimizer
        optimizer = CNNGuidedOptimizer(
            room_dims=(400, 300),
            objects=furniture,
            room_analysis=room_analysis,
            population_size=50,  # Small for testing
            generations=10      # Small for testing
        )
        
        # Test optimization
        optimized_layout = optimizer.optimize()
        print(f"✓ Optimized layout with {len(optimized_layout)} furniture items")
        
        return True
        
    except Exception as e:
        print(f"✗ CNN-Guided Optimizer test failed: {e}")
        return False

def test_enhanced_layout_generator():
    """Test the enhanced layout generator."""
    print("Testing Enhanced Layout Generator...")
    
    try:
        from src.visualization.enhanced_layout_generator import EnhancedLayoutGenerator
        
        # Create test data
        architectural_elements = [
            {"type": "window", "x": 50, "y": 50, "w": 100, "h": 50},
            {"type": "door", "x": 300, "y": 50, "w": 50, "h": 150}
        ]
        
        furniture_layout = [
            {"type": "bed", "x": 100, "y": 100, "w": 200, "h": 150},
            {"type": "sofa", "x": 50, "y": 200, "w": 150, "h": 80}
        ]
        
        # Initialize generator
        generator = EnhancedLayoutGenerator(room_dims=(400, 300))
        
        # Test layout generation
        layout_buffer = generator.generate_layout(
            architectural_elements=architectural_elements,
            furniture_layout=furniture_layout
        )
        
        print(f"✓ Generated layout image (buffer size: {len(layout_buffer.getvalue())} bytes)")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced Layout Generator test failed: {e}")
        return False

def test_ai_room_optimizer():
    """Test the main AI Room Optimizer integration."""
    print("Testing AI Room Optimizer Integration...")
    
    try:
        from src.integration.ai_room_optimizer import AIRoomOptimizer
        
        # Create test image
        test_img = create_test_room_image()
        
        # Save temporary image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            test_img.save(tmp.name)
            tmp_path = tmp.name
        
        # Initialize optimizer
        optimizer = AIRoomOptimizer(room_dims=(400, 300), device="cpu")
        
        # Test room analysis
        room_analysis = optimizer.analyze_room(tmp_path)
        print(f"✓ Room analysis completed")
        
        # Test furniture optimization
        furniture = [
            {"name": "Bed", "width": 200, "depth": 150},
            {"name": "Sofa", "width": 150, "depth": 80}
        ]
        
        optimized_layout = optimizer.optimize_furniture_placement(furniture)
        print(f"✓ Optimized {len(optimized_layout)} furniture items")
        
        # Test layout generation
        layout_buffer = optimizer.generate_2d_layout()
        print(f"✓ Generated 2D layout")
        
        # Clean up
        os.unlink(tmp_path)
        
        return True
        
    except Exception as e:
        print(f"✗ AI Room Optimizer test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Enhanced AI Room Optimizer System")
    print("=" * 50)
    
    tests = [
        ("CNN Architectural Detector", test_cnn_architectural_detector),
        ("CNN-Guided Optimizer", test_cnn_guided_optimizer),
        ("Enhanced Layout Generator", test_enhanced_layout_generator),
        ("AI Room Optimizer Integration", test_ai_room_optimizer)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * 30)
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced system is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
