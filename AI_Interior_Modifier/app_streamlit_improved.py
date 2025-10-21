"""
AI Interior Modifier - Enhanced Streamlit App
============================================
Advanced UI with better UX, error handling, and visual improvements.
Features:
- Modern, responsive design
- Better error handling and user feedback
- Detection overlay visualization
- Progress indicators
- Improved layout and spacing
- Enhanced visual elements
"""

from io import BytesIO
import json
import tempfile
import cv2
import numpy as np

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from src.input.cv_detector import RoomDetector
from optimizer import LayoutOptimizer
from plot2d import plot_layout
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="AI Interior Modifier",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #e8f4fd;
        border-left: 4px solid #3498db;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Friendly mapping for common pretrained labels -> domain labels
COCO_TO_DOMAIN = {
    "sofa": "sofa",
    "couch": "sofa", 
    "chair": "chair",
    "dining table": "table",
    "bed": "bed",
    "table": "table",
    "tv": "tv",
    "potted plant": "plant",
    "laptop": "laptop",
    "book": "book",
    "bottle": "bottle",
    "cup": "cup",
    "bowl": "bowl",
    "banana": "banana",
    "apple": "apple",
    "sandwich": "sandwich",
    "orange": "orange",
    "broccoli": "broccoli",
    "carrot": "carrot",
    "hot dog": "hot dog",
    "pizza": "pizza",
    "donut": "donut",
    "cake": "cake",
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "airplane": "airplane",
    "bus": "bus",
    "train": "train",
    "truck": "truck",
    "boat": "boat",
    "traffic light": "traffic light",
    "fire hydrant": "fire hydrant",
    "stop sign": "stop sign",
    "parking meter": "parking meter",
    "bench": "bench",
    "bird": "bird",
    "cat": "cat",
    "dog": "dog",
    "horse": "horse",
    "sheep": "sheep",
    "cow": "cow",
    "elephant": "elephant",
    "bear": "bear",
    "zebra": "zebra",
    "giraffe": "giraffe",
    "backpack": "backpack",
    "umbrella": "umbrella",
    "handbag": "handbag",
    "tie": "tie",
    "suitcase": "suitcase",
    "frisbee": "frisbee",
    "skis": "skis",
    "snowboard": "snowboard",
    "sports ball": "sports ball",
    "kite": "kite",
    "baseball bat": "baseball bat",
    "baseball glove": "baseball glove",
    "skateboard": "skateboard",
    "surfboard": "surfboard",
    "tennis racket": "tennis racket",
    "wine glass": "wine glass",
    "fork": "fork",
    "knife": "knife",
    "spoon": "spoon",
    "bowl": "bowl",
    "clock": "clock",
    "vase": "vase",
    "scissors": "scissors",
    "teddy bear": "teddy bear",
    "hair drier": "hair drier",
    "toothbrush": "toothbrush"
}

# Fallback mapping by index for custom models
CLASS_MAP = {
    0: "wall",
    1: "door", 
    2: "window",
    3: "bed",
    4: "table",
    5: "chair",
    6: "sofa",
    7: "tv",
    8: "wardrobe",
    9: "lamp",
    10: "desk"
}

def map_label_to_domain(label: str):
    """Map COCO labels to domain-specific labels."""
    if not label:
        return None
    label_lower = label.lower()
    return COCO_TO_DOMAIN.get(label_lower, label_lower)

def draw_detection_overlay(image, detections, class_map):
    """Draw bounding boxes and labels on the image."""
    img_array = np.array(image)
    img_with_boxes = img_array.copy()
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = det.get("label") or class_map.get(det["class"], "unknown")
        confidence = det["confidence"]
        
        # Draw bounding box
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label and confidence
        text = f"{label}: {confidence:.2f}"
        cv2.putText(img_with_boxes, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return Image.fromarray(img_with_boxes)

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 AI Interior Modifier</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>🎯 What this app does:</strong> Upload a room image and get AI-optimized furniture placement using 
        YOLOv8 computer vision detection and genetic algorithm optimization.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Room dimensions
        st.markdown("#### 📏 Room Dimensions")
        room_width = st.number_input("Room Width (cm)", min_value=100, max_value=2000, value=400, help="Width of your room in centimeters")
        room_height = st.number_input("Room Height (cm)", min_value=100, max_value=2000, value=300, help="Height of your room in centimeters")
        
        # Detection settings
        st.markdown("#### 🔍 Detection Settings")
        conf_thresh = st.slider("Detection Confidence", min_value=0.1, max_value=0.9, value=0.35, step=0.05, 
                              help="Higher values = more confident detections only")
        show_detections_overlay = st.checkbox("Show Detection Overlay", value=True, 
                                            help="Display bounding boxes on uploaded image")
        
        # GA parameters
        st.markdown("#### 🧬 Genetic Algorithm")
        pop_size = st.number_input("Population Size", min_value=10, max_value=500, value=60, 
                                 help="Number of layout candidates per generation")
        generations = st.number_input("Generations", min_value=10, max_value=2000, value=250,
                                   help="Number of optimization iterations")
        seed = st.number_input("Random Seed (0 = random)", min_value=0, value=0,
                             help="For reproducible results")
        
        # Preferences
        st.markdown("#### 🎨 Preferences")
        bed_near_wall = st.checkbox("Prefer bed near wall", value=True)
        table_near_window = st.checkbox("Prefer table near window", value=True)
        min_distance = st.number_input("Minimum Clearance (cm)", min_value=0, max_value=200, value=20,
                                     help="Minimum space between furniture pieces")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">📸 Upload Room Image</h2>', unsafe_allow_html=True)
        image_file = st.file_uploader(
            "Choose a room image", 
            type=["jpg", "png", "jpeg"],
            help="Upload a clear photo of your room for furniture detection"
        )
        
        if image_file is not None:
            img = Image.open(image_file).convert("RGB")
            st.image(img, caption="Uploaded Room Image", use_container_width=True)
            
            # Initialize detector
            with st.spinner("🔍 Loading AI model and detecting furniture..."):
                try:
                    detector = RoomDetector()
                    model_name = detector.get_model_name()
                    
                    # Show model info
                    st.markdown(f"""
                    <div class="info-box">
                        <strong>🤖 Model:</strong> {model_name}<br>
                        <strong>📊 Confidence Threshold:</strong> {conf_thresh}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Save image temporarily for detection
                    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                    img.save(tmp.name)
                    
                    # Run detection
                    detections = detector.detect(tmp.name, conf_threshold=conf_thresh)
                    
                    if detections:
                        st.success(f"✅ Found {len(detections)} furniture items!")
                        
                        # Show detection overlay if requested
                        if show_detections_overlay:
                            img_with_overlay = draw_detection_overlay(img, detections, CLASS_MAP)
                            st.image(img_with_overlay, caption="Detection Results", use_container_width=True)
                        
                        # Parse detections
                        image_shape = img.size[::-1]  # (height, width)
                        objects = detector.parse_detections(detections, CLASS_MAP, (room_width, room_height), image_shape)
                        
                        # Display detected objects
                        st.markdown('<h3 class="section-header">🪑 Detected Furniture</h3>', unsafe_allow_html=True)
                        
                        for i, (det, obj) in enumerate(zip(detections, objects)):
                            label = det.get("label") or CLASS_MAP.get(det["class"], "unknown")
                            domain_label = map_label_to_domain(label) or obj["type"]
                            
                            with st.expander(f"🪑 {domain_label.capitalize()} {i+1} (Confidence: {det['confidence']:.2f})"):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    w = st.number_input(f"Width (cm)", min_value=20, max_value=int(room_width), 
                                                      value=int(max(30, obj["w"])), key=f"w_{i}")
                                with col_b:
                                    h = st.number_input(f"Depth (cm)", min_value=20, max_value=int(room_height), 
                                                      value=int(max(30, obj["h"])), key=f"h_{i}")
                                obj["w"], obj["h"] = w, h
                                obj["type"] = domain_label
                    else:
                        st.warning("⚠️ No furniture detected. You can proceed with default furniture or try a different image.")
                        objects = []
                        
                except Exception as e:
                    st.error(f"❌ Detection failed: {str(e)}")
                    objects = []
        else:
            st.markdown("""
            <div class="warning-box">
                <strong>💡 No image uploaded</strong> - Using default furniture set. 
                Upload an image for AI-powered detection!
            </div>
            """, unsafe_allow_html=True)
            
            # Default furniture
            objects = [
                {"type": "bed", "w": 200, "h": 150},
                {"type": "table", "w": 120, "h": 80}, 
                {"type": "chair", "w": 50, "h": 50}
            ]
            
            st.markdown('<h3 class="section-header">🪑 Default Furniture</h3>', unsafe_allow_html=True)
            for i, obj in enumerate(objects):
                col_a, col_b = st.columns(2)
                with col_a:
                    obj["w"] = st.number_input(f"{obj['type'].capitalize()} width (cm)", 
                                            min_value=20, max_value=int(room_width), 
                                            value=int(obj["w"]), key=f"def_w_{i}")
                with col_b:
                    obj["h"] = st.number_input(f"{obj['type'].capitalize()} depth (cm)", 
                                            min_value=20, max_value=int(room_height), 
                                            value=int(obj["h"]), key=f"def_h_{i}")
    
    with col2:
        st.markdown('<h2 class="section-header">🎯 Optimization Results</h2>', unsafe_allow_html=True)
        
        # Run optimization button
        if st.button("🚀 Run AI Optimization", type="primary", use_container_width=True):
            if objects:
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("🧬 Initializing genetic algorithm...")
                    progress_bar.progress(10)
                    
                    # Initialize optimizer
                    ga = LayoutOptimizer(
                        room_dims=(room_width, room_height),
                        objects=objects,
                        user_prefs={
                            "bed_near_wall": bed_near_wall,
                            "table_near_window": table_near_window,
                            "min_distance": min_distance
                        },
                        population_size=int(pop_size),
                        generations=int(generations),
                        seed=(None if int(seed) == 0 else int(seed))
                    )
                    
                    status_text.text("🔄 Optimizing layout...")
                    progress_bar.progress(30)
                    
                    # Run optimization
                    layout = ga.optimize()
                    
                    status_text.text("📊 Generating visualization...")
                    progress_bar.progress(80)
                    
                    # Create visualization
                    buf = BytesIO()
                    plot_layout((room_width, room_height), layout, save_buffer=buf)
                    buf.seek(0)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Optimization complete!")
                    
                    # Display results
                    st.markdown('<h3 class="section-header">📐 Optimized Layout</h3>', unsafe_allow_html=True)
                    st.image(buf, caption="AI-Optimized Room Layout", use_container_width=True)
                    
                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Room Area", f"{room_width * room_height / 10000:.1f} m²")
                    with col_b:
                        st.metric("Furniture Pieces", len(layout))
                    with col_c:
                        total_furniture_area = sum(obj["w"] * obj["h"] for obj in layout)
                        st.metric("Furniture Coverage", f"{total_furniture_area / (room_width * room_height) * 100:.1f}%")
                    
                    # Export functionality
                    st.markdown('<h3 class="section-header">💾 Export Results</h3>', unsafe_allow_html=True)
                    
                    result = {
                        "metadata": {
                            "room_width_cm": room_width,
                            "room_height_cm": room_height,
                            "model": model_name if 'model_name' in locals() else "yolov8n.pt",
                            "ga_population": pop_size,
                            "ga_generations": generations,
                            "optimization_timestamp": str(pd.Timestamp.now()) if 'pd' in globals() else "unknown"
                        },
                        "layout": layout
                    }
                    
                    json_bytes = json.dumps(result, indent=2).encode("utf-8")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.download_button(
                            "📄 Download JSON", 
                            data=json_bytes, 
                            file_name="optimized_layout.json", 
                            mime="application/json",
                            use_container_width=True
                        )
                    with col_b:
                        st.download_button(
                            "🖼️ Download Image", 
                            data=buf.getvalue(), 
                            file_name="optimized_layout.png", 
                            mime="image/png",
                            use_container_width=True
                        )
                    
                    st.markdown("""
                    <div class="success-box">
                        <strong>🎉 Optimization Complete!</strong><br>
                        Your room layout has been optimized using AI. Download the results above.
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Optimization failed: {str(e)}")
                    st.exception(e)
            else:
                st.warning("⚠️ Please add some furniture items before running optimization.")
        
        # Tips section
        st.markdown('<h3 class="section-header">💡 Tips for Best Results</h3>', unsafe_allow_html=True)
        st.markdown("""
        - **📸 Clear Images**: Use well-lit, high-resolution room photos
        - **🎯 Multiple Angles**: Try different camera angles for better detection
        - **⚙️ Adjust Settings**: Tune confidence threshold and GA parameters
        - **🔄 Experiment**: Try different preferences and see how they affect layout
        - **📏 Accurate Dimensions**: Enter precise room measurements for realistic results
        """)

if __name__ == "__main__":
    main()
