"""
AI Interior Modifier - Launcher Script
======================================
Simple launcher for the improved Streamlit app.
"""

import subprocess
import sys
import os

def main():
    """Launch the improved Streamlit app."""
    print("AI Interior Modifier - Starting...")
    print("=" * 50)
    
    # Check if simple app exists
    if not os.path.exists("app_simple.py"):
        print("Error: app_simple.py not found!")
        print("   Please make sure the simple app file exists.")
        return False
    
    try:
        print("Launching Simple AI Interior Modifier...")
        print("   The app will open in your default browser.")
        print("   Press Ctrl+C to stop the server.")
        print("=" * 50)
        
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app_simple.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
        return True
    except KeyboardInterrupt:
        print("\nApp stopped by user.")
        return True
    except Exception as e:
        print(f"Error launching app: {e}")
        return False

if __name__ == "__main__":
    main()
