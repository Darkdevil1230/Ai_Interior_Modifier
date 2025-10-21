"""
AI Interior Modifier - Simple UI
===============================
Ultra-simple, easy-to-understand interface.
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
    layout="wide"
)

# Simple, clean CSS
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    
    .header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 1.5rem;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        border-radius: 10px;
        color: white;
    }
    
    .header h1 {
        font-size: 2rem;
        margin: 0;
    }
    
    .header p {
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .step {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #4CAF50;
    }
    
    .step h3 {
        color: #2E7D32;
        margin: 0 0 1rem 0;
        font-size: 1.2rem;
    }
    
    .result-box {
        background: #E8F5E8;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #4CAF50;
    }
    
    .metric {
        background: #F5F5F5;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2E7D32;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Enhanced object categorization for better organization
FURNITURE_CATEGORIES = {
    "seating": ["chair", "sofa", "bench", "stool", "ottoman"],
    "tables": ["table", "desk", "coffee table", "nightstand", "side table"],
    "storage": ["wardrobe", "cabinet", "dresser", "bookshelf", "tv stand"],
    "bedroom": ["bed", "nightstand", "dresser", "wardrobe"],
    "electronics": ["tv", "laptop", "phone", "speaker", "clock"],
    "lighting": ["lamp", "floor lamp", "table lamp"],
    "decorative": ["plant", "picture", "vase", "mirror", "sculpture"],
    "storage_items": ["backpack", "handbag", "suitcase", "basket", "box"],
    "personal": ["book", "cup", "bottle", "shirt", "shoes", "hat"],
    "small_objects": ["phone", "pen", "scissors", "toy", "ball"]
}

# Object size categories for better layout optimization
SIZE_CATEGORIES = {
    "large": ["bed", "sofa", "wardrobe", "dresser", "bookshelf"],
    "medium": ["table", "desk", "chair", "tv stand", "cabinet"],
    "small": ["lamp", "plant", "picture", "vase", "clock"],
    "tiny": ["phone", "cup", "pen", "toy", "scissors"]
}

CLASS_MAP = {0: "wall", 1: "door", 2: "window", 3: "bed", 4: "table", 5: "chair"}

def map_label_to_domain(label):
    if not label:
        return None
    # Import the comprehensive mapping from the detector
    from src.input.cv_detector import COCO_TO_DOMAIN
    return COCO_TO_DOMAIN.get(label.lower(), label.lower())

def categorize_object(obj_type):
    """Categorize objects for better organization."""
    obj_type_lower = obj_type.lower()
    for category, items in FURNITURE_CATEGORIES.items():
        if obj_type_lower in items:
            return category
    return "other"

def get_size_category(obj_type):
    """Get size category for layout optimization."""
    obj_type_lower = obj_type.lower()
    for size, items in SIZE_CATEGORIES.items():
        if obj_type_lower in items:
            return size
    return "medium"

def get_object_icon(obj_type):
    """Get appropriate icon for object type."""
    icons = {
        "bed": "🛏️", "sofa": "🛋️", "chair": "🪑", "table": "🪑", "desk": "🪑",
        "tv": "📺", "lamp": "💡", "plant": "🌱", "book": "📚", "laptop": "💻",
        "phone": "📱", "cup": "☕", "bottle": "🍼", "shirt": "👕", "shoes": "👟",
        "hat": "🎩", "toy": "🧸", "ball": "⚽", "scissors": "✂️", "umbrella": "☂️",
        "towel": "🧻", "pillow": "🛏️", "blanket": "🛏️", "curtain": "🪟",
        "fan": "🌀", "heater": "🔥", "wardrobe": "🚪", "cabinet": "🚪",
        "dresser": "🚪", "bookshelf": "📚", "mirror": "🪞", "picture": "🖼️",
        "vase": "🏺", "clock": "🕐", "backpack": "🎒", "handbag": "👜"
    }
    return icons.get(obj_type.lower(), "📦")

def draw_simple_overlay(image, detections):
    """Draw simple detection overlay."""
    img_array = np.array(image)
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = det.get("label", "object")
        confidence = det["confidence"]
        
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img_array, f"{label}: {confidence:.1f}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return Image.fromarray(img_array)

def main():
    # Simple header
    st.markdown("""
    <div class="header">
        <h1>🏠 AI Interior Modifier</h1>
        <p>Upload your room photo and get AI-optimized furniture placement</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 1: Upload Image
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown('<h3>📸 Step 1: Upload Room Photo</h3>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a room image",
        type=['jpg', 'png', 'jpeg'],
        help="Take a clear photo of your room"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your Room", use_container_width=True)
    else:
        st.info("👆 Please upload a room photo to get started")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 2: Room Size
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown('<h3>📏 Step 2: Room Size</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        room_width = st.number_input("Room Width (cm)", min_value=100, max_value=2000, value=400)
    with col2:
        room_height = st.number_input("Room Height (cm)", min_value=100, max_value=2000, value=300)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 3: Settings
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown('<h3>⚙️ Step 3: Settings</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        confidence = st.slider("Detection Sensitivity", 0.1, 0.9, 0.35, 0.05, 
                             help="Higher = more confident detections only")
        show_overlay = st.checkbox("Show Detection Boxes", value=True)
    
    with col2:
        bed_wall = st.checkbox("Put bed near wall", value=True)
        table_window = st.checkbox("Put table near window", value=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process image and detect furniture
    objects = []
    model_name = "yolov8n.pt"
    
    if uploaded_file:
        # Save temporary file
        tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        image.save(tmp_file.name)
        
        # Detect furniture
        with st.spinner("🔍 Detecting furniture..."):
            try:
                detector = RoomDetector()
                model_name = detector.get_model_name()
                detections = detector.detect(tmp_file.name, conf_threshold=confidence)
                
                if detections:
                    st.success(f"✅ Found {len(detections)} furniture items!")
                    
                    # Show overlay if requested
                    if show_overlay:
                        overlay_img = draw_simple_overlay(image, detections)
                        st.image(overlay_img, caption="Detected Furniture", use_container_width=True)
                    
                    # Parse detections
                    image_shape = image.size[::-1]
                    objects = detector.parse_detections(detections, CLASS_MAP, (room_width, room_height), image_shape)
                    
                    # Show detected objects organized by category
                    st.markdown("**Detected Objects:**")
                    
                    # Group objects by category
                    categorized_objects = {}
                    for i, (det, obj) in enumerate(zip(detections, objects)):
                        label = det.get("label", "unknown")
                        domain_label = map_label_to_domain(label) or obj["type"]
                        obj["type"] = domain_label
                        obj["confidence"] = det["confidence"]
                        obj["index"] = i
                        
                        category = categorize_object(domain_label)
                        if category not in categorized_objects:
                            categorized_objects[category] = []
                        categorized_objects[category].append(obj)
                    
                    # Display objects by category
                    for category, objs in categorized_objects.items():
                        st.markdown(f"**{category.replace('_', ' ').title()}:**")
                        for obj in objs:
                            icon = get_object_icon(obj["type"])
                            size_cat = get_size_category(obj["type"])
                            
                            with st.expander(f"{icon} {obj['type'].capitalize()} (Confidence: {obj['confidence']:.2f}) - {size_cat.title()}"):
                                size_col1, size_col2 = st.columns(2)
                                with size_col1:
                                    obj["w"] = st.number_input("Width (cm)", min_value=5, max_value=room_width, 
                                                            value=int(max(10, obj["w"])), key=f"w_{obj['index']}")
                                with size_col2:
                                    obj["h"] = st.number_input("Depth (cm)", min_value=5, max_value=room_height, 
                                                            value=int(max(10, obj["h"])), key=f"h_{obj['index']}")
                else:
                    st.warning("⚠️ No furniture detected. Using default furniture.")
                    objects = []
                    
            except Exception as e:
                st.error(f"❌ Detection failed: {str(e)}")
                objects = []
    else:
        # Default objects with more variety
        objects = [
            {"type": "bed", "w": 200, "h": 150},
            {"type": "table", "w": 120, "h": 80},
            {"type": "chair", "w": 50, "h": 50},
            {"type": "bookshelf", "w": 80, "h": 30},
            {"type": "lamp", "w": 30, "h": 30},
            {"type": "plant", "w": 25, "h": 25}
        ]
        
        st.markdown("**Default Objects:**")
        for i, obj in enumerate(objects):
            icon = get_object_icon(obj["type"])
            size_cat = get_size_category(obj["type"])
            st.write(f"{icon} {obj['type'].capitalize()} - {size_cat.title()}")
            size_col1, size_col2 = st.columns(2)
            with size_col1:
                obj["w"] = st.number_input("Width (cm)", min_value=5, max_value=room_width, 
                                        value=int(obj["w"]), key=f"def_w_{i}")
            with size_col2:
                obj["h"] = st.number_input("Depth (cm)", min_value=5, max_value=room_height, 
                                        value=int(obj["h"]), key=f"def_h_{i}")
    
    # Step 4: Optimize
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown('<h3>🚀 Step 4: Optimize Layout</h3>', unsafe_allow_html=True)
    
    if st.button("🎯 Optimize My Room Layout", type="primary", use_container_width=True):
        if objects:
            # Simple progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🧬 Optimizing layout...")
                progress_bar.progress(50)
                
                # Run optimization
                optimizer = LayoutOptimizer(
                    room_dims=(room_width, room_height),
                    objects=objects,
                    user_prefs={
                        "bed_near_wall": bed_wall,
                        "table_near_window": table_window,
                        "min_distance": 20
                    },
                    population_size=50,
                    generations=100,
                    seed=42
                )
                
                layout = optimizer.optimize()
                
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                
                # Show results
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown('<h3>📐 Your Optimized Layout</h3>', unsafe_allow_html=True)
                
                # Create visualization
                buf = BytesIO()
                plot_layout((room_width, room_height), layout, save_buffer=buf)
                buf.seek(0)
                
                st.image(buf, caption="AI-Optimized Room Layout", use_container_width=True)
                
                # Simple metrics
                room_area = room_width * room_height
                furniture_area = sum(obj["w"] * obj["h"] for obj in layout)
                coverage = (furniture_area / room_area) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric">
                        <div class="metric-value">{room_area/10000:.1f}</div>
                        <div class="metric-label">Room Area (m²)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric">
                        <div class="metric-value">{len(layout)}</div>
                        <div class="metric-label">Furniture Pieces</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric">
                        <div class="metric-value">{coverage:.0f}%</div>
                        <div class="metric-label">Space Used</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Simple export
                st.markdown("**Download Results:**")
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    result_data = {
                        "room": {"width": room_width, "height": room_height},
                        "furniture": layout
                    }
                    json_bytes = json.dumps(result_data, indent=2).encode("utf-8")
                    st.download_button(
                        "📄 Download Layout Data",
                        data=json_bytes,
                        file_name="room_layout.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with export_col2:
                    st.download_button(
                        "🖼️ Download Layout Image",
                        data=buf.getvalue(),
                        file_name="room_layout.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.success("🎉 Your room layout has been optimized!")
                
            except Exception as e:
                st.error(f"❌ Optimization failed: {str(e)}")
        else:
            st.warning("⚠️ Please add some furniture first.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Simple footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🏠 AI Interior Modifier - Transform your space with AI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
