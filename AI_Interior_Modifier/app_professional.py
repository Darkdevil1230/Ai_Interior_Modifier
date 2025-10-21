"""
AI Interior Modifier - Professional UI
=====================================
Clean, professional interface focused on core functionality.
"""

import streamlit as st
import tempfile
import json
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

from src.input.cv_detector import RoomDetector
from optimizer import LayoutOptimizer
from plot2d import plot_layout

# Page configuration
st.set_page_config(
    page_title="AI Interior Modifier",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .main {
        padding-top: 2rem;
    }
    
    /* Header */
    .header {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .header h1 {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .header p {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 300;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    
    .card h3 {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 0 0 1rem 0;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    /* Metrics */
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    
    .metric {
        text-align: center;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin: 0.25rem 0 0 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    /* File uploader */
    .stFileUploader > div > div {
        border: 2px dashed #3498db;
        border-radius: 8px;
        background: #f8f9fa;
    }
    
    /* Success/Error messages */
    .success-message {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724;
    }
    
    .error-message {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #721c24;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom spacing */
    .spacer {
        height: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# COCO to domain mapping
COCO_TO_DOMAIN = {
    "sofa": "sofa", "couch": "sofa", "chair": "chair", "dining table": "table",
    "bed": "bed", "table": "table", "tv": "tv", "potted plant": "plant"
}

CLASS_MAP = {0: "wall", 1: "door", 2: "window", 3: "bed", 4: "table", 5: "chair"}

def map_label_to_domain(label: str):
    if not label:
        return None
    return COCO_TO_DOMAIN.get(label.lower(), label.lower())

def draw_detection_overlay(image, detections):
    """Draw detection overlay on image."""
    img_array = np.array(image)
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = det.get("label", "object")
        confidence = det["confidence"]
        
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_array, f"{label}: {confidence:.2f}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return Image.fromarray(img_array)

def main():
    # Professional header
    st.markdown("""
    <div class="header">
        <h1>🏠 AI Interior Modifier</h1>
        <p>Transform your space with AI-powered furniture optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content in two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>📸 Upload Room Image</h3>', unsafe_allow_html=True)
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Choose a room image",
            type=['jpg', 'png', 'jpeg'],
            help="Upload a clear photo of your room"
        )
        
        # Room dimensions
        st.markdown("**Room Dimensions**")
        room_col1, room_col2 = st.columns(2)
        with room_col1:
            room_width = st.number_input("Width (cm)", min_value=100, max_value=2000, value=400)
        with room_col2:
            room_height = st.number_input("Height (cm)", min_value=100, max_value=2000, value=300)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Settings card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>⚙️ Settings</h3>', unsafe_allow_html=True)
        
        # Detection settings
        confidence = st.slider("Detection Confidence", 0.1, 0.9, 0.35, 0.05)
        show_overlay = st.checkbox("Show Detection Overlay", value=True)
        
        # Optimization settings
        st.markdown("**Optimization**")
        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            population = st.number_input("Population", min_value=20, max_value=200, value=50)
        with opt_col2:
            generations = st.number_input("Generations", min_value=50, max_value=500, value=100)
        
        # Preferences
        st.markdown("**Preferences**")
        pref_col1, pref_col2 = st.columns(2)
        with pref_col1:
            bed_wall = st.checkbox("Bed near wall", value=True)
        with pref_col2:
            table_window = st.checkbox("Table near window", value=True)
        
        min_distance = st.slider("Minimum Clearance (cm)", 0, 100, 20)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Results card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>🎯 Results</h3>', unsafe_allow_html=True)
        
        # Initialize variables
        objects = []
        model_name = "yolov8n.pt"
        
        # Process uploaded image
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            
            # Save temporary file for detection
            tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            image.save(tmp_file.name)
            
            # Run detection
            with st.spinner("🔍 Detecting furniture..."):
                try:
                    detector = RoomDetector()
                    model_name = detector.get_model_name()
                    detections = detector.detect(tmp_file.name, conf_threshold=confidence)
                    
                    if detections:
                        st.success(f"✅ Found {len(detections)} furniture items")
                        
                        # Show overlay if requested
                        if show_overlay:
                            overlay_img = draw_detection_overlay(image, detections)
                            st.image(overlay_img, caption="Detection Results", use_container_width=True)
                        
                        # Parse detections
                        image_shape = image.size[::-1]
                        objects = detector.parse_detections(detections, CLASS_MAP, (room_width, room_height), image_shape)
                        
                        # Display detected objects
                        st.markdown("**Detected Furniture:**")
                        for i, (det, obj) in enumerate(zip(detections, objects)):
                            label = det.get("label", "unknown")
                            domain_label = map_label_to_domain(label) or obj["type"]
                            obj["type"] = domain_label
                            
                            with st.expander(f"🪑 {domain_label.capitalize()} (Confidence: {det['confidence']:.2f})"):
                                size_col1, size_col2 = st.columns(2)
                                with size_col1:
                                    obj["w"] = st.number_input("Width (cm)", min_value=20, max_value=room_width, 
                                                            value=int(max(30, obj["w"])), key=f"w_{i}")
                                with size_col2:
                                    obj["h"] = st.number_input("Depth (cm)", min_value=20, max_value=room_height, 
                                                            value=int(max(30, obj["h"])), key=f"h_{i}")
                    else:
                        st.warning("⚠️ No furniture detected")
                        objects = []
                        
                except Exception as e:
                    st.error(f"❌ Detection failed: {str(e)}")
                    objects = []
        else:
            st.info("📷 Upload an image to detect furniture")
            
            # Default furniture
            objects = [
                {"type": "bed", "w": 200, "h": 150},
                {"type": "table", "w": 120, "h": 80},
                {"type": "chair", "w": 50, "h": 50}
            ]
            
            st.markdown("**Default Furniture:**")
            for i, obj in enumerate(objects):
                size_col1, size_col2 = st.columns(2)
                with size_col1:
                    obj["w"] = st.number_input(f"{obj['type'].capitalize()} width", 
                                            min_value=20, max_value=room_width, 
                                            value=int(obj["w"]), key=f"def_w_{i}")
                with size_col2:
                    obj["h"] = st.number_input(f"{obj['type'].capitalize()} depth", 
                                            min_value=20, max_value=room_height, 
                                            value=int(obj["h"]), key=f"def_h_{i}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Optimization button and results
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    
    if st.button("🚀 Optimize Layout", type="primary", use_container_width=True):
        if objects:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🧬 Initializing optimization...")
                progress_bar.progress(20)
                
                # Initialize optimizer
                optimizer = LayoutOptimizer(
                    room_dims=(room_width, room_height),
                    objects=objects,
                    user_prefs={
                        "bed_near_wall": bed_wall,
                        "table_near_window": table_window,
                        "min_distance": min_distance
                    },
                    population_size=population,
                    generations=generations,
                    seed=42
                )
                
                status_text.text("🔄 Optimizing layout...")
                progress_bar.progress(50)
                
                # Run optimization
                layout = optimizer.optimize()
                
                status_text.text("📊 Generating visualization...")
                progress_bar.progress(80)
                
                # Create visualization
                buf = BytesIO()
                plot_layout((room_width, room_height), layout, save_buffer=buf)
                buf.seek(0)
                
                progress_bar.progress(100)
                status_text.text("✅ Optimization complete!")
                
                # Display results
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<h3>📐 Optimized Layout</h3>', unsafe_allow_html=True)
                
                # Show visualization
                st.image(buf, caption="AI-Optimized Room Layout", use_container_width=True)
                
                # Metrics
                room_area = room_width * room_height
                furniture_area = sum(obj["w"] * obj["h"] for obj in layout)
                coverage = (furniture_area / room_area) * 100
                
                st.markdown("""
                <div class="metric-container">
                    <div class="metric">
                        <div class="metric-value">{:.1f} m²</div>
                        <div class="metric-label">Room Area</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Furniture Pieces</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{:.1f}%</div>
                        <div class="metric-label">Coverage</div>
                    </div>
                </div>
                """.format(room_area/10000, len(layout), coverage), unsafe_allow_html=True)
                
                # Export options
                st.markdown("**Export Results:**")
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    result_data = {
                        "metadata": {
                            "room_width_cm": room_width,
                            "room_height_cm": room_height,
                            "model": model_name,
                            "optimization_settings": {
                                "population": population,
                                "generations": generations,
                                "preferences": {
                                    "bed_near_wall": bed_wall,
                                    "table_near_window": table_window,
                                    "min_distance": min_distance
                                }
                            }
                        },
                        "layout": layout
                    }
                    
                    json_bytes = json.dumps(result_data, indent=2).encode("utf-8")
                    st.download_button(
                        "📄 Download JSON",
                        data=json_bytes,
                        file_name="optimized_layout.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with export_col2:
                    st.download_button(
                        "🖼️ Download Image",
                        data=buf.getvalue(),
                        file_name="optimized_layout.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Success message
                st.markdown("""
                <div class="success-message">
                    <strong>🎉 Optimization Complete!</strong><br>
                    Your room layout has been optimized using AI. Download the results above.
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-message">
                    <strong>❌ Optimization Failed</strong><br>
                    {str(e)}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please add furniture items before optimizing.")

if __name__ == "__main__":
    main()
