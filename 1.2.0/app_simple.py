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
import pandas as pd
import os

from src.input.cv_detector import RoomDetector
from src.input.enhanced_detector import EnhancedDetector
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

def draw_enhanced_overlay(image, detections):
    """Draw enhanced detection overlay with distinction between architectural and furniture."""
    img_array = np.array(image).copy()
    
    # Architectural elements (windows, doors) get different colors
    architectural_elements = {"window", "door", "wall", "floor", "ceiling"}
    
    # Create semi-transparent overlay
    overlay = img_array.copy()
    
    # Separate detections by type
    arch_dets = []
    furn_dets = []
    
    for det in detections:
        label = det.get("label", "object")
        if label and label.lower() in architectural_elements:
            arch_dets.append(det)
        else:
            furn_dets.append(det)
    
    # Draw architectural elements first (blue)
    for det in arch_dets:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = det.get("label", "object")
        confidence = det["confidence"]
        category = det.get("category", "unknown")
        
        # Blue color for architectural elements
        color = (30, 144, 255)  # Dodger blue
        
        # Draw filled rectangle with transparency
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        
        # Draw border
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 4)
        
        # Add label with background
        text = f"[FIXED] {label.upper()}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(img_array, (x1, y1 - text_height - 10), 
                     (x1 + text_width + 10, y1), color, -1)
        cv2.putText(img_array, text, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Confidence on bottom
        conf_text = f"{confidence:.2f}"
        cv2.putText(img_array, conf_text, (x1 + 5, y2 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw furniture (green)
    for det in furn_dets:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = det.get("label", "object")
        confidence = det["confidence"]
        
        # Green color for furniture
        color = (50, 205, 50)  # Lime green
        
        # Draw filled rectangle with transparency
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        
        # Draw border
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 3)
        
        # Add label with background
        text = f"{label.title()}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(img_array, (x1, y1 - text_height - 8), 
                     (x1 + text_width + 8, y1), color, -1)
        cv2.putText(img_array, text, (x1 + 4, y1 - 4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Confidence
        conf_text = f"{confidence:.2f}"
        cv2.putText(img_array, conf_text, (x1 + 5, y2 - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Blend overlay with original image
    alpha = 0.3
    img_array = cv2.addWeighted(overlay, alpha, img_array, 1 - alpha, 0)
    
    # Add legend
    legend_y = 30
    # Architectural elements legend
    cv2.rectangle(img_array, (10, legend_y), (40, legend_y + 20), (30, 144, 255), -1)
    cv2.putText(img_array, "Fixed Elements (windows/doors)", (50, legend_y + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img_array, "Fixed Elements (windows/doors)", (50, legend_y + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Furniture legend
    legend_y += 30
    cv2.rectangle(img_array, (10, legend_y), (40, legend_y + 20), (50, 205, 50), -1)
    cv2.putText(img_array, "Movable Furniture", (50, legend_y + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img_array, "Movable Furniture", (50, legend_y + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return Image.fromarray(img_array)

def load_furniture_catalog():
    """Load furniture catalog from CSV."""
    try:
        catalog_path = "data/furniture_catalog.csv"
        if os.path.exists(catalog_path):
            df = pd.read_csv(catalog_path)
            furniture_list = []
            for _, row in df.iterrows():
                furniture_list.append({
                    "name": row["name"],
                    "width": float(row["width_m"]) * 100,  # Convert to cm
                    "depth": float(row["depth_m"]) * 100,  # Convert to cm
                    "prefer": row.get("prefer", "center")
                })
            return furniture_list
        else:
            # Default furniture if catalog not found
            return get_default_furniture()
    except Exception as e:
        st.warning(f"Could not load furniture catalog: {e}")
        return get_default_furniture()

def get_default_furniture():
    """Return default furniture list."""
    return [
        {"name": "Bed", "width": 200, "depth": 150, "prefer": "wall"},
        {"name": "Sofa", "width": 150, "depth": 80, "prefer": "center"},
        {"name": "Table", "width": 120, "depth": 60, "prefer": "window"},
        {"name": "Wardrobe", "width": 180, "depth": 60, "prefer": "wall"},
        {"name": "Desk", "width": 120, "depth": 70, "prefer": "window"},
        {"name": "Chair", "width": 50, "depth": 50, "prefer": "center"},
        {"name": "TV Stand", "width": 120, "depth": 40, "prefer": "wall"},
        {"name": "Bookshelf", "width": 80, "depth": 30, "prefer": "wall"},
        {"name": "Coffee Table", "width": 100, "depth": 60, "prefer": "center"},
        {"name": "Lamp", "width": 30, "depth": 30, "prefer": "center"},
        {"name": "Plant", "width": 25, "depth": 25, "prefer": "window"},
        {"name": "Mirror", "width": 80, "depth": 10, "prefer": "wall"},
        {"name": "Cabinet", "width": 100, "depth": 50, "prefer": "wall"},
        {"name": "Dresser", "width": 120, "depth": 50, "prefer": "wall"},
        {"name": "Nightstand", "width": 50, "depth": 40, "prefer": "wall"}
    ]

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
        st.image(image, caption="Your Room", width='stretch')
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
        use_multi_pass = st.checkbox("🚀 Multi-Pass Detection (Recommended)", value=True,
                                    help="Uses YOLO + Edge Detection for paintings/artwork - IMPROVED for better accuracy")
        confidence = st.slider("Detection Sensitivity", 0.05, 0.9, 0.20, 0.05, 
                             help="Lower = more detections (may include false positives), Higher = fewer but more confident detections. Recommended: 0.15-0.25")
        show_overlay = st.checkbox("Show Detection Boxes", value=True)
        st.caption("💡 Try lowering sensitivity or enable Multi-Pass if objects are missed")
    
    with col2:
        bed_wall = st.checkbox("Put bed near wall", value=True)
        table_window = st.checkbox("Put table near window", value=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process image and detect furniture
    objects = []
    model_name = "yolov8n.pt"
    is_empty_room = False
    
    if uploaded_file:
        # Save temporary file
        tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        image.save(tmp_file.name)
        tmp_file_path = tmp_file.name
        
        try:
            # Detect furniture
            with st.spinner("🔍 Analyzing room with AI..." if not use_multi_pass else "🔍 Running multi-pass detection (YOLO + Edge Detection)..."):
                try:
                    detector = RoomDetector()
                    model_name = detector.get_model_name()
                    
                    # Use enhanced detector if multi-pass is enabled
                    if use_multi_pass:
                        enhanced_detector = EnhancedDetector(detector)
                        result = enhanced_detector.multi_pass_detection(
                            tmp_file_path,
                            confidence_levels=[confidence, max(0.1, confidence - 0.1), max(0.05, confidence - 0.15)]
                        )
                        detections = result['detections']
                        suggestions = result.get('suggestions', [])
                        st.success(f"✨ Multi-pass detection completed! Detected {len(detections)} objects using YOLO + Edge Detection")
                    else:
                        detections = detector.detect(tmp_file_path, conf_threshold=confidence)
                        suggestions = []
                    
                    if detections:
                        st.success(f"✅ Successfully detected {len(detections)} objects in your room!")
                        
                        # Separate architectural elements from furniture
                        arch_dets = [d for d in detections if d.get('category') == 'architectural']
                        furn_dets = [d for d in detections if d.get('category') != 'architectural']
                        
                        # Display model info
                        if detector.is_custom_model:
                            st.info(f"🎯 Using custom-trained model: {model_name}")
                        else:
                            st.warning(f"⚠️ Using general COCO model: {model_name} - May not detect windows/doors accurately")
                        
                        # Show detection summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🪟 Fixed Elements", len(arch_dets), 
                                     help="Windows, doors, and other architectural features")
                        with col2:
                            st.metric("🪑 Furniture Items", len(furn_dets),
                                     help="Movable furniture pieces")
                        with col3:
                            st.metric("📊 Total Objects", len(detections))
                        
                        # Show overlay if requested
                        if show_overlay:
                            overlay_img = draw_enhanced_overlay(image, detections)
                            st.image(overlay_img, caption="🔍 AI Detection Results - Blue=Fixed, Green=Furniture", width='stretch')
                        
                        # Parse detections
                        image_shape = image.size[::-1]
                        objects = detector.parse_detections(detections, CLASS_MAP, (room_width, room_height), image_shape)
                        
                        # Show detected objects organized by category
                        st.markdown("---")
                        st.markdown("### 📋 Detected Objects Details")
                        
                        # Group objects by category
                        categorized_objects = {}
                        architectural_objects = []
                        
                        for i, (det, obj) in enumerate(zip(detections, objects)):
                            label = det.get("label", "unknown")
                            domain_label = map_label_to_domain(label) or obj["type"]
                            obj["type"] = domain_label
                            obj["confidence"] = det["confidence"]
                            obj["index"] = i
                            obj["category"] = det.get("category", "other")
                            
                            # Separate architectural elements
                            if obj["category"] == "architectural":
                                architectural_objects.append(obj)
                            else:
                                category = categorize_object(domain_label)
                                if category not in categorized_objects:
                                    categorized_objects[category] = []
                                categorized_objects[category].append(obj)
                        
                        # Display architectural elements first (if any)
                        if architectural_objects:
                            st.markdown("#### 🏗️ Fixed Architectural Elements")
                            st.info("These elements are detected but won't be moved during optimization.")
                            for obj in architectural_objects:
                                icon = "🪟" if "window" in obj["type"].lower() else "🚪" if "door" in obj["type"].lower() else "🏛️"
                                st.markdown(f"- {icon} **{obj['type'].upper()}** - Confidence: {obj['confidence']:.2f} - Size: {int(obj['w'])}×{int(obj['h'])}cm")
                        
                        # Display movable furniture by category
                        if categorized_objects:
                            st.markdown("#### 🪑 Movable Furniture (Will be Optimized)")
                            for category, objs in categorized_objects.items():
                                st.markdown(f"**{category.replace('_', ' ').title()}:**")
                                for obj in objs:
                                    icon = get_object_icon(obj["type"])
                                    size_cat = get_size_category(obj["type"])
                                    
                                    with st.expander(f"{icon} {obj['type'].capitalize()} (Confidence: {obj['confidence']:.2f}) - {size_cat.title()}"):
                                        # Allow type correction
                                        obj_types = ["bed", "sofa", "chair", "ottoman", "table", "desk", "wardrobe", 
                                                    "cabinet", "bookshelf", "tv stand", "lamp", "plant", "nightstand",
                                                    "dresser", "coffee table", "side table", "bench", "stool"]
                                        current_type = obj["type"].lower()
                                        if current_type not in obj_types:
                                            obj_types.insert(0, current_type)
                                        
                                        new_type = st.selectbox(
                                            "Object Type (correct if wrong)",
                                            obj_types,
                                            index=obj_types.index(current_type) if current_type in obj_types else 0,
                                            key=f"type_{obj['index']}"
                                        )
                                        obj["type"] = new_type
                                        
                                        size_col1, size_col2 = st.columns(2)
                                        with size_col1:
                                            obj["w"] = st.number_input("Width (cm)", min_value=5, max_value=room_width, 
                                                                    value=int(max(10, obj["w"])), key=f"w_{obj['index']}")
                                        with size_col2:
                                            obj["h"] = st.number_input("Depth (cm)", min_value=5, max_value=room_height, 
                                                                    value=int(max(10, obj["h"])), key=f"h_{obj['index']}")
                                        
                                        # Option to remove false detection
                                        if st.checkbox("Remove this detection (false positive)", key=f"remove_{obj['index']}"):
                                            obj["removed"] = True
                        
                        # Show AI suggestions if available
                        if use_multi_pass and suggestions:
                            st.markdown("---")
                            st.markdown("#### 💡 AI Smart Suggestions")
                            st.info("🤖 **AI Analysis:** The following objects are typically found in rooms but weren't detected. Add them if they exist in your image:")
                            
                            for i, sug in enumerate(suggestions[:5]):
                                priority_emoji = "🔴" if sug["priority"] == "high" else "🟡" if sug["priority"] == "medium" else "🟢"
                                st.markdown(f"- {priority_emoji} **{sug['type'].title()}** ({sug['suggested_width']}×{sug['suggested_height']}cm) - *{sug['reason']}*")
                        
                        # Add manual object addition option
                        st.markdown("---")
                        st.markdown("#### ➕ Add Missing Objects")
                        with st.expander("Manually add objects that weren't detected"):
                            add_col1, add_col2, add_col3 = st.columns(3)
                            with add_col1:
                                manual_type = st.selectbox(
                                    "Object Type",
                                    ["lamp", "painting", "mirror", "shelf", "cabinet", "tv", 
                                     "chair", "table", "sofa", "bed", "plant", "vase"],
                                    key="manual_type"
                                )
                            with add_col2:
                                manual_w = st.number_input("Width (cm)", min_value=5, max_value=room_width, 
                                                          value=50, key="manual_w")
                            with add_col3:
                                manual_h = st.number_input("Depth (cm)", min_value=5, max_value=room_height, 
                                                          value=50, key="manual_h")
                            
                            if st.button("➕ Add This Object", key="add_manual"):
                                manual_obj = {
                                    "type": manual_type,
                                    "w": manual_w,
                                    "h": manual_h,
                                    "confidence": 1.0,
                                    "index": len(objects),
                                    "category": "furniture",
                                    "manual": True
                                }
                                objects.append(manual_obj)
                                st.success(f"✅ Added {manual_type}!")
                        
                        # Store architectural elements separately (they won't be optimized but will be shown)
                        # Create a session state to preserve architectural elements
                        if 'architectural_elements' not in st.session_state:
                            st.session_state.architectural_elements = []
                        st.session_state.architectural_elements = architectural_objects
                        
                        # Update objects list to only include furniture (for optimization) and remove marked items
                        objects = [obj for obj in objects 
                                 if obj.get("category") != "architectural" 
                                 and not obj.get("removed", False)]
                    else:
                        st.info("🏠 No furniture detected! This appears to be an empty room. Let's select furniture to place.")
                        is_empty_room = True
                        objects = []
                        
                except Exception as e:
                    st.error(f"❌ Detection failed: {str(e)}")
                    objects = []
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
            except Exception:
                pass
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
    
    # Furniture Selection Step for Empty Rooms
    if is_empty_room:
        st.markdown('<div class="step">', unsafe_allow_html=True)
        st.markdown('<h3>📦 Step 3.5: Select Furniture to Place</h3>', unsafe_allow_html=True)
        
        # Load furniture catalog
        furniture_catalog = load_furniture_catalog()
        
        # Create a multiselect with furniture categorized by type
        st.markdown("**Choose furniture items to place in your room:**")
        
        # Group furniture by category
        furniture_by_category = {
            "Large Furniture": ["Bed", "Sofa", "Wardrobe", "Dresser"],
            "Tables & Surfaces": ["Table", "Desk", "Coffee Table", "Nightstand", "TV Stand"],
            "Seating": ["Chair", "Bench", "Stool", "Ottoman"],
            "Storage": ["Bookshelf", "Cabinet", "Backpack", "Handbag", "Suitcase", "Basket", "Box"],
            "Electronics": ["TV", "Laptop", "Phone", "Speaker", "Clock"],
            "Lighting": ["Lamp"],
            "Decorative": ["Mirror", "Plant", "Picture", "Vase", "Candle", "Sculpture"],
            "Small Items": ["Book", "Cup", "Bottle", "Bowl", "Plate", "Glass"]
        }
        
        selected_furniture = []
        
        # Create tabs for better organization
        tabs = st.tabs(["🏠 Large Furniture", "🪑 Tables & Seating", "📦 Storage", "💡 Electronics & Lighting", "🎨 Decorative"])
        
        # Tab 1: Large Furniture
        with tabs[0]:
            st.markdown("**Large Furniture:**")
            large_items = ["Bed", "Sofa", "Wardrobe", "Dresser", "Cabinet"]
            for item in large_items:
                if item in [f["name"] for f in furniture_catalog]:
                    if st.checkbox(f"🛏️ {item}", key=f"cb_{item}_large"):
                        furniture_item = next((f for f in furniture_catalog if f["name"] == item), None)
                        if furniture_item:
                            selected_furniture.append(furniture_item)
        
        # Tab 2: Tables & Seating
        with tabs[1]:
            st.markdown("**Tables & Surfaces:**")
            table_items = ["Table", "Desk", "Coffee Table", "Nightstand", "TV Stand"]
            for item in table_items:
                if item in [f["name"] for f in furniture_catalog]:
                    if st.checkbox(f"🪑 {item}", key=f"cb_{item}_table"):
                        furniture_item = next((f for f in furniture_catalog if f["name"] == item), None)
                        if furniture_item:
                            selected_furniture.append(furniture_item)
            
            st.markdown("**Seating:**")
            seating_items = ["Chair", "Bench", "Stool", "Ottoman"]
            for item in seating_items:
                if item in [f["name"] for f in furniture_catalog]:
                    if st.checkbox(f"🪑 {item}", key=f"cb_{item}_seat"):
                        furniture_item = next((f for f in furniture_catalog if f["name"] == item), None)
                        if furniture_item:
                            selected_furniture.append(furniture_item)
        
        # Tab 3: Storage
        with tabs[2]:
            st.markdown("**Storage Solutions:**")
            storage_items = ["Bookshelf", "Cabinet", "Backpack", "Handbag", "Suitcase", "Basket", "Box"]
            for item in storage_items:
                if item in [f["name"] for f in furniture_catalog]:
                    if st.checkbox(f"📦 {item}", key=f"cb_{item}_storage"):
                        furniture_item = next((f for f in furniture_catalog if f["name"] == item), None)
                        if furniture_item:
                            selected_furniture.append(furniture_item)
        
        # Tab 4: Electronics & Lighting
        with tabs[3]:
            st.markdown("**Electronics:**")
            electronics_items = ["TV", "Laptop", "Phone", "Speaker", "Clock"]
            for item in electronics_items:
                if item in [f["name"] for f in furniture_catalog]:
                    if st.checkbox(f"📺 {item}", key=f"cb_{item}_elec"):
                        furniture_item = next((f for f in furniture_catalog if f["name"] == item), None)
                        if furniture_item:
                            selected_furniture.append(furniture_item)
            
            st.markdown("**Lighting:**")
            lighting_items = ["Lamp"]
            for item in lighting_items:
                if item in [f["name"] for f in furniture_catalog]:
                    if st.checkbox(f"💡 {item}", key=f"cb_{item}_light"):
                        furniture_item = next((f for f in furniture_catalog if f["name"] == item), None)
                        if furniture_item:
                            selected_furniture.append(furniture_item)
        
        # Tab 5: Decorative
        with tabs[4]:
            st.markdown("**Decorative Items:**")
            decorative_items = ["Mirror", "Plant", "Picture", "Vase", "Candle", "Sculpture"]
            for item in decorative_items:
                if item in [f["name"] for f in furniture_catalog]:
                    if st.checkbox(f"🎨 {item}", key=f"cb_{item}_decor"):
                        furniture_item = next((f for f in furniture_catalog if f["name"] == item), None)
                        if furniture_item:
                            selected_furniture.append(furniture_item)
        
        # Show selected furniture summary
        if selected_furniture:
            st.success(f"✅ Selected {len(selected_furniture)} furniture items!")
            st.markdown("**Selected Items:**")
            for item in selected_furniture:
                icon = get_object_icon(item["name"].lower())
                st.write(f"{icon} {item['name']} - {item['width']}cm × {item['depth']}cm")
            
            # Convert selected furniture to objects format
            objects = []
            for i, item in enumerate(selected_furniture):
                obj = {
                    "type": item["name"].lower(),
                    "w": item["width"],
                    "h": item["depth"]
                }
                objects.append(obj)
        else:
            st.warning("⚠️ Please select at least one furniture item to place in your room.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 4: Optimize
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown('<h3>🚀 Step 4: Optimize Layout</h3>', unsafe_allow_html=True)
    
    # Initialize session state for layouts and architectural elements
    if 'optimized_layouts' not in st.session_state:
        st.session_state.optimized_layouts = None
    if 'optimized_scores' not in st.session_state:
        st.session_state.optimized_scores = None
    if 'room_dimensions' not in st.session_state:
        st.session_state.room_dimensions = None
    if 'architectural_elements' not in st.session_state:
        st.session_state.architectural_elements = []
    
    if st.button("🎯 Optimize My Room Layout", type="primary", width='stretch'):
        if objects:
            # Simple progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🧬 Optimizing multiple layouts...")
                progress_bar.progress(20)

                # Run optimization (generate alternatives)
                optimizer = LayoutOptimizer(
                    room_dims=(room_width, room_height),
                    objects=objects,
                    user_prefs={
                        "bed_near_wall": bed_wall,
                        "table_near_window": table_window,
                        "min_distance": 20
                    },
                    population_size=150,  # Increased for better optimization
                    generations=250,  # Increased for better results
                    seed=42
                )

                alt_results = optimizer.optimize_multiple(count=3, runs=5)
                layouts = [r["layout"] for r in alt_results] if alt_results else []
                scores = [r["score"] for r in alt_results] if alt_results else []

                progress_bar.progress(80)
                status_text.text("✅ Generated alternatives")

                if not layouts:
                    # Fallback to single optimize if for some reason none were produced
                    layout = optimizer.optimize()
                    layouts = [layout]
                    scores = [float(optimizer.fitness(layout))]

                # Store in session state
                st.session_state.optimized_layouts = layouts
                st.session_state.optimized_scores = scores
                st.session_state.room_dimensions = (room_width, room_height)
                
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                st.success("🎉 Multiple layout options generated with ZERO overlaps guaranteed!")
                
            except Exception as e:
                st.error(f"❌ Optimization failed: {str(e)}")
                st.session_state.optimized_layouts = None
                st.session_state.optimized_scores = None
        else:
            st.warning("⚠️ Please add some furniture first.")
    
    # Display results if they exist in session state
    if st.session_state.optimized_layouts is not None:
        layouts = st.session_state.optimized_layouts
        scores = st.session_state.optimized_scores
        stored_room_width, stored_room_height = st.session_state.room_dimensions
        
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown('<h3>📐 Choose Your Optimized Layout</h3>', unsafe_allow_html=True)
        
        # Show info about what's included in the layout
        architectural_elems = st.session_state.get('architectural_elements', [])
        if architectural_elems:
            st.info(f"💡 **Layout includes:** {len(architectural_elems)} fixed architectural element(s) (windows/doors shown in blue with dashed borders) + optimized furniture placement (shown in various colors)")
        
        st.success("✅ **All layouts are validated with ZERO overlaps** - furniture is optimally placed with proper spacing")
        
        # Render each alternative in tabs with preview
        tabs = st.tabs([f"Option {i+1}" for i in range(len(layouts))])
        preview_buffers = []
        room_area = stored_room_width * stored_room_height
        for idx, (tab, layout) in enumerate(zip(tabs, layouts)):
            with tab:
                # Combine architectural elements with optimized furniture for display
                architectural_elems = st.session_state.get('architectural_elements', [])
                combined_layout = architectural_elems + layout
                
                buf = BytesIO()
                plot_layout((stored_room_width, stored_room_height), combined_layout, save_buffer=buf)
                buf.seek(0)
                preview_buffers.append(buf)
                
                # Show info about what's displayed
                if architectural_elems:
                    st.image(buf, caption=f"Layout Option {idx+1} - Showing {len(architectural_elems)} fixed elements + {len(layout)} optimized furniture", width='stretch')
                else:
                    st.image(buf, caption=f"Layout Option {idx+1}", width='stretch')

                furniture_area = sum(obj["w"] * obj["h"] for obj in layout)
                coverage = (furniture_area / room_area) * 100
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class=\"metric\">
                        <div class=\"metric-value\">{room_area/10000:.1f}</div>
                        <div class=\"metric-label\">Room Area (m²)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class=\"metric\">
                        <div class=\"metric-value\">{len(layout)}</div>
                        <div class=\"metric-label\">Furniture Pieces</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class=\"metric\">
                        <div class=\"metric-value\">{coverage:.0f}%</div>
                        <div class=\"metric-label\">Space Used</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.caption(f"Score: {scores[idx]:.0f}")

        # Selection and export
        selected_idx = 0
        if len(layouts) > 1:
            selected_label = st.radio(
                "Select a layout to export",
                [f"Option {i+1}" for i in range(len(layouts))],
                index=0,
                horizontal=True,
                key="layout_selector"
            )
            selected_idx = int(selected_label.split()[-1]) - 1

        selected_layout = layouts[selected_idx]
        selected_buf = preview_buffers[selected_idx]

        st.markdown("**Download Selected Result:**")
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            # Include architectural elements in export
            architectural_elems = st.session_state.get('architectural_elements', [])
            result_data = {
                "room": {"width": stored_room_width, "height": stored_room_height},
                "architectural_elements": architectural_elems,
                "furniture": selected_layout,
                "score": scores[selected_idx] if scores else None
            }
            json_bytes = json.dumps(result_data, indent=2).encode("utf-8")
            st.download_button(
                "📄 Download Layout Data",
                data=json_bytes,
                file_name=f"room_layout_option_{selected_idx+1}.json",
                mime="application/json",
                width='stretch'
            )
        with export_col2:
            st.download_button(
                "🖼️ Download Layout Image",
                data=selected_buf.getvalue(),
                file_name=f"room_layout_option_{selected_idx+1}.png",
                mime="image/png",
                width='stretch'
            )

        st.markdown('</div>', unsafe_allow_html=True)
    
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
