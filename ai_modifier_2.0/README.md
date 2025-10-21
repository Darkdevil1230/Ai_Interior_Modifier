# 🏠 AI Interior Modifier

An AI-powered interior design optimization system that uses computer vision (YOLOv8) for furniture detection and genetic algorithms for optimal furniture placement.

## ✨ Features

- **🔍 AI-Powered Detection**: Uses YOLOv8 to automatically detect 50+ furniture and object types including chairs, tables, beds, sofas, bookshelves, lamps, electronics, personal items, and more
- **🧬 Genetic Algorithm Optimization**: Optimizes furniture placement using advanced genetic algorithms
- **🎨 User Preferences**: Customizable preferences for furniture placement (bed near wall, table near window, etc.)
- **📊 Real-time Visualization**: Interactive 2D layout visualization with detailed metrics
- **💾 Export Options**: Download optimized layouts as JSON or PNG images
- **🎯 Modern UI**: Beautiful, responsive Streamlit interface with enhanced UX

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   # If you have git
   git clone <repository-url>
   cd AI_Interior_Modifier
   
   # Or simply download and extract the project files
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   # Use the launcher script (RECOMMENDED)
   python run_app.py
   
   # OR run directly
   streamlit run app_simple.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, manually navigate to the URL

## 🎯 How to Use

### 1. Upload Room Image
- Click "Choose a room image" and select a clear photo of your room
- Supported formats: JPG, PNG, JPEG
- For best results, use well-lit, high-resolution images

### 2. Configure Settings
- **Room Dimensions**: Enter accurate width and height in centimeters
- **Detection Confidence**: Adjust sensitivity (higher = more confident detections only)
- **Genetic Algorithm Parameters**: Tune population size and generations for optimization quality vs speed
- **Preferences**: Set furniture placement preferences

### 3. Run Optimization
- Click "🚀 Run AI Optimization" to start the process
- The system will:
  - Detect furniture in your image
  - Optimize placement using genetic algorithms
  - Generate a 2D visualization
  - Provide metrics and export options

### 4. Export Results
- Download the optimized layout as JSON (for further processing)
- Download the visualization as PNG image
- View detailed metrics about room utilization


## 📁 Project Structure

```
AI_Interior_Modifier/
├── app_simple.py              # Main user interface (RECOMMENDED)
├── run_app.py                # Launcher script
├── cv_detector.py            # Basic detection (root level)
├── optimizer.py              # Advanced genetic algorithm optimizer
├── plot2d.py                 # 2D visualization
├── requirements.txt          # Dependencies
├── src/                      # Source modules
│   └── input/
│       └── cv_detector.py    # Enhanced detection with error handling
├── scripts/                  # Utility scripts
│   ├── download_dataset.py     # Dataset downloader
│   ├── download_yolo_weights.py # YOLO weights downloader
│   ├── fix_encoding.py        # Encoding fixer
│   └── train_yolo.py           # YOLO training script
├── data/                     # Data and configuration
│   ├── data.yaml            # YOLO training config
│   ├── furniture_catalog.csv # Furniture specifications
│   └── datasets/            # Training datasets
└── weights/                  # Model weights
    ├── best.pt              # Custom trained model
    └── yolov8n.pt           # Pretrained YOLOv8 model
```

## 🔧 Technical Details

### Detection Engine
- **Model**: YOLOv8 (ultralytics)
- **Fallback**: Automatic fallback from custom weights to pretrained model
- **Object Types**: Supports 50+ object types including:
  - **Furniture**: Beds, sofas, chairs, tables, desks, wardrobes, bookshelves, cabinets, dressers, nightstands, coffee tables, ottomans, benches, stools, TV stands, mirrors
  - **Electronics**: TVs, laptops, phones, speakers, clocks, keyboards, mice, remotes
  - **Storage**: Backpacks, handbags, suitcases, baskets, boxes, drawers
  - **Personal Items**: Books, cups, bottles, bowls, plates, glasses, shirts, pants, shoes, hats, toys
  - **Decorative**: Plants, pictures, vases, candles, sculptures, ornaments
  - **Tools & Equipment**: Scissors, hammers, screwdrivers, wrenches
  - **Room Features**: Windows, doors, walls, floors, ceilings
- **Confidence**: Adjustable detection threshold
- **Categorization**: Objects are automatically categorized by type and size for better organization

### Optimization Algorithm
- **Method**: Genetic Algorithm with tournament selection
- **Features**: 
  - Elitism (preserves best solutions)
  - Configurable population size and generations
  - Penalty systems for overlaps and constraints
  - User preference rewards
  - Minimum clearance enforcement

### Visualization
- **Engine**: Matplotlib with custom styling
- **Features**: Color-coded objects, confidence display, size annotations
- **Export**: PNG images and JSON data

## 🎨 Simple UI Features

The **simple version** (`app_simple.py`) features:

- **📱 Easy to Use**: Step-by-step process that anyone can follow
- **🎯 Clear Steps**: 4 simple steps to get your optimized layout
- **👀 Visual Feedback**: See detection boxes and results clearly
- **⚡ Fast Results**: Quick optimization with sensible defaults
- **📊 Simple Metrics**: Easy-to-understand room statistics
- **💾 Easy Export**: One-click download of results
- **🎨 Clean Design**: Minimal, distraction-free interface

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'ultralytics'"**
   ```bash
   pip install ultralytics
   ```

2. **"Model loading failed"**
   - The app automatically falls back to pretrained models
   - Check your internet connection for model downloads

3. **"Detection returns no results"**
   - Try lowering the confidence threshold
   - Ensure your image is clear and well-lit
   - Try different camera angles

4. **"Optimization takes too long"**
   - Reduce population size and generations in settings
   - Use fewer furniture pieces for testing

### Performance Tips

- **For faster processing**: Reduce GA parameters (population: 20-50, generations: 50-100)
- **For better quality**: Increase GA parameters (population: 100-200, generations: 200-500)
- **For testing**: Use default furniture instead of uploading images

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python test_system.py`
5. Submit a pull request

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for the detection model
- **Streamlit**: For the web framework
- **Genetic Algorithms**: Inspired by optimization research
- **Computer Vision**: OpenCV and PIL for image processing

---

**🎉 Enjoy optimizing your interior spaces with AI!**
