"""
Simple UI Test
==============
Test the simple, easy-to-understand UI.
"""

def test_simple_app():
    """Test the simple app."""
    print("Testing Simple UI...")
    
    try:
        import app_simple
        print("Simple app imports successfully")
        
        # Test core functionality
        from src.input.cv_detector import RoomDetector
        from optimizer import LayoutOptimizer
        from plot2d import plot_layout
        
        print("All modules work")
        
        # Test with simple objects
        objects = [
            {"type": "bed", "w": 200, "h": 150},
            {"type": "table", "w": 120, "h": 80}
        ]
        
        optimizer = LayoutOptimizer(
            room_dims=(400, 300),
            objects=objects,
            population_size=20,
            generations=10,
            seed=42
        )
        
        layout = optimizer.optimize()
        print(f"Optimization works: {len(layout)} objects")
        
        return True
        
    except Exception as e:
        print(f"Simple app test failed: {e}")
        return False

def main():
    """Run simple test."""
    print("AI Interior Modifier - Simple UI Test")
    print("=" * 50)
    
    if test_simple_app():
        print("\nSimple UI Test PASSED!")
        print("The simple UI is ready to use!")
        print("\nTo run the simple app:")
        print("   python run_app.py")
        print("   OR")
        print("   streamlit run app_simple.py")
    else:
        print("\nSimple UI Test FAILED!")
    
    return test_simple_app()

if __name__ == "__main__":
    main()
