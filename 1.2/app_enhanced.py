"""
app_enhanced.py
---------------
Enhanced Streamlit app that integrates CNN architectural detection,
genetic algorithm optimization, and professional 2D layout generation.
"""

import streamlit as st
import tempfile
import json
import os
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import numpy as np

# Import our enhanced modules
from src.integration.ai_room_optimizer import AIRoomOptimizer
from src.input.cnn_architectural_detector import CNNArchitecturalDetector

# Page configuration
st.set_page_config(
    page_title="AI Room Optimizer - Enhanced",
    page_icon="🏠",
    layout="wide"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    
    .header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .header h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .step {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
    }
    
    .step h3 {
        color: #2c5aa0;
        margin: 0 0 1.5rem 0;
        font-size: 1.4rem;
        font-weight: 600;
    }
    
    .result-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border: 2px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2c5aa0;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        font-weight: 500;
    }
    
    .cnn-badge {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .genetic-badge {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .ai-badge {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #2c5aa0;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def load_furniture_catalog():
    """Load furniture catalog from CSV with real-world dimensions."""
    try:
        catalog_path = "data/furniture_catalog.csv"
        if os.path.exists(catalog_path):
            import pandas as pd
            df = pd.read_csv(catalog_path, comment='#')  # Skip comment lines
            furniture_list = []
            for _, row in df.iterrows():
                # Skip empty rows
                if pd.isna(row["name"]) or str(row["name"]).strip() == "":
                    continue
                    
                furniture_list.append({
                    "name": str(row["name"]).strip(),
                    "width": float(row["width_m"]) * 100,  # Convert to cm for internal calculations
                    "depth": float(row["depth_m"]) * 100,  # Convert to cm for internal calculations
                    "height": float(row.get("height_m", 0.8)) * 100,  # Convert to cm, default 0.8m
                    "prefer": row.get("prefer", "center"),
                    "category": row.get("category", "furniture"),
                    "description": row.get("description", "")
                })
            return furniture_list
        else:
            return get_default_furniture()
    except Exception as e:
        st.warning(f"Could not load furniture catalog: {e}")
        return get_default_furniture()

def get_default_furniture():
    """Return default furniture list with real-world dimensions in meters."""
    return [
        {"name": "Queen Bed", "width": 160, "depth": 203, "height": 60, "prefer": "wall", "category": "bedroom", "description": "Queen size bed (1.6m x 2.03m)"},
        {"name": "2-Seat Sofa", "width": 160, "depth": 90, "height": 80, "prefer": "center", "category": "living", "description": "2-seater sofa (1.6m x 0.9m)"},
        {"name": "Coffee Table", "width": 120, "depth": 60, "height": 40, "prefer": "center", "category": "living", "description": "Coffee table"},
        {"name": "Wardrobe", "width": 180, "depth": 60, "height": 210, "prefer": "wall", "category": "bedroom", "description": "Full wardrobe"},
        {"name": "Desk", "width": 140, "depth": 70, "height": 75, "prefer": "window", "category": "office", "description": "Office desk"},
        {"name": "Dining Chair", "width": 50, "depth": 50, "height": 90, "prefer": "center", "category": "dining", "description": "Dining room chair"},
        {"name": "TV Stand", "width": 120, "depth": 40, "height": 60, "prefer": "wall", "category": "living", "description": "TV stand"},
        {"name": "Bookshelf", "width": 80, "depth": 30, "height": 200, "prefer": "wall", "category": "living", "description": "Tall bookshelf"},
        {"name": "Table Lamp", "width": 20, "depth": 20, "height": 50, "prefer": "center", "category": "living", "description": "Table lamp"},
        {"name": "Small Plant", "width": 30, "depth": 30, "height": 80, "prefer": "window", "category": "living", "description": "Small potted plant"},
        {"name": "Wall Mirror", "width": 80, "depth": 10, "height": 120, "prefer": "wall", "category": "living", "description": "Wall mirror"}
    ]

def main():
    # Enhanced header
    st.markdown("""
    <div class="header">
        <h1>🏠 AI Room Optimizer</h1>
        <p>Advanced CNN + Genetic Algorithm Furniture Placement System</p>
        <div style="margin-top: 1rem;">
            <span class="cnn-badge">CNN Detection</span>
            <span class="genetic-badge">Genetic Algorithm</span>
            <span class="ai-badge">AI Optimization</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'room_optimizer' not in st.session_state:
        st.session_state.room_optimizer = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'optimization_complete' not in st.session_state:
        st.session_state.optimization_complete = False
    
    # Step 1: Upload Image
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown('<h3>📸 Step 1: Upload Room Image</h3>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a room image (JPG, PNG, JPEG)",
        type=['jpg', 'png', 'jpeg'],
        help="Upload a clear photo of your room for AI analysis"
    )
    
    image = None
    if uploaded_file is not None:
        # Temporary debug information about the uploaded file
        try:
            st.caption(f"[DEBUG] Uploaded file: name={uploaded_file.name}, type={uploaded_file.type}, size={uploaded_file.size} bytes")
        except Exception:
            pass

        try:
            # Read the raw bytes from the uploaded file and wrap in a BytesIO
            file_bytes = uploaded_file.getvalue()
            from io import BytesIO
            image = Image.open(BytesIO(file_bytes)).convert("RGB")
        except UnidentifiedImageError:
            st.error("Could not read the uploaded file as an image. Please upload a valid JPG or PNG file.")
            image = None
        except Exception as e:
            st.error(f"Unexpected error while reading image: {e}")
            image = None

    if image is not None:
        st.image(image, caption="Your Room Image", use_container_width=True)
    else:
        st.info("👆 Please upload a room image to get started")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 2: Room Configuration
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown('<h3>📏 Step 2: Room Configuration</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        room_width = st.number_input("Room Width (m)", min_value=1.0, max_value=20.0, value=4.0, step=0.1,
                                   help="Enter the width of your room in meters")
    with col2:
        room_height = st.number_input("Room Height (m)", min_value=1.0, max_value=20.0, value=3.0, step=0.1,
                                    help="Enter the height of your room in meters")
    
    # Room type selection
    room_type = st.selectbox(
        "Select Room Type",
        ["Living Room", "Bedroom", "Kitchen", "Dining Room", "Office", "Bathroom"],
        index=0,
        help="This will guide furniture choices and architectural constraints"
    )
    
    # Convert to cm for internal calculations
    room_width_cm = room_width * 100
    room_height_cm = room_height * 100
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            use_cnn_detection = st.checkbox("Enable CNN Architectural Detection", value=True,
                                          help="Use deep learning for better architectural element detection")
            confidence_threshold = st.slider("Detection Confidence", 0.1, 0.9, 0.5, 0.05,
                                           help="Higher values = more confident detections, lower values = more detections")
        with col2:
            population_size = st.number_input("Genetic Algorithm Population", min_value=50, max_value=500, value=200,
                                            help="Larger population = better optimization but slower")
            generations = st.number_input("Optimization Generations", min_value=50, max_value=1000, value=300,
                                        help="More generations = better results but slower")

        st.markdown("---")
        col3, col4 = st.columns(2)
        with col3:
            enable_known_length = st.checkbox("Use known length for scale (door width)", value=False,
                                             help="If enabled, we will use the door width you provide to estimate cm-per-pixel from detections.")
        with col4:
            known_door_width_cm = st.number_input("Known Door Width (cm)", min_value=50.0, max_value=120.0, value=90.0, step=1.0,
                                                help="Typical interior doors are ~80–90 cm")

        show_detections_overlay = st.checkbox(
            "Overlay raw detections on layout",
            value=False,
            help="Draw outlines/labels of detected objects (windows, doors, furniture) on the floorplan output."
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 3: Furniture Selection
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown('<h3>🪑 Step 3: Select Furniture</h3>', unsafe_allow_html=True)
    
    # Load furniture catalog
    furniture_catalog = load_furniture_catalog()
    
    # Furniture selection interface
    selected_furniture = []
    
    # Create tabs for furniture categories
    tabs = st.tabs(["🏠 Large Furniture", "🪑 Tables & Seating", "📦 Storage", "💡 Electronics & Lighting", "🎨 Decorative"])
    
    # Define allowed categories by room type
    roomtype_to_categories = {
        "Living Room": {"living", "lighting", "electronics", "decorative", "architectural"},
        "Bedroom": {"bedroom", "lighting", "decorative", "architectural"},
        "Kitchen": {"kitchen", "dining", "lighting", "architectural"},
        "Dining Room": {"dining", "living", "lighting", "architectural"},
        "Office": {"office", "lighting", "electronics", "decorative", "architectural"},
        "Bathroom": {"bathroom", "lighting", "architectural"},
    }
    allowed_categories = roomtype_to_categories.get(room_type, {"living", "bedroom", "office", "dining"})

    # Tab 1: Large Furniture
    with tabs[0]:
        st.markdown("**Large Furniture Items:**")
        large_items = ["King Bed", "Queen Bed", "Double Bed", "3-Seat Sofa", "2-Seat Sofa", "Wardrobe", "Dresser", "Recliner", "Loveseat"]
        for item in large_items:
            if item in [f["name"] for f in furniture_catalog]:
                furniture_item = next((f for f in furniture_catalog if f["name"] == item), None)
                if furniture_item:
                    if str(furniture_item.get('category', '')).lower() not in allowed_categories:
                        continue
                    width_m = furniture_item['width'] / 100  # Convert to meters for display
                    depth_m = furniture_item['depth'] / 100  # Convert to meters for display
                    height_m = furniture_item.get('height', 80) / 100  # Convert to meters for display
                    description = furniture_item.get('description', '')
                    
                    # Get appropriate icon
                    icon = "🛏️" if "Bed" in item else "🛋️" if "Sofa" in item or "Loveseat" in item else "🚪" if "Wardrobe" in item else "📦"
                    
                    if st.checkbox(f"{icon} {item} ({width_m:.1f}×{depth_m:.1f}×{height_m:.1f}m)", key=f"cb_{item}_large"):
                        selected_furniture.append(furniture_item)
                        st.caption(f"📏 {description}")
    
    # Tab 2: Tables & Seating
    with tabs[1]:
        st.markdown("**Tables & Surfaces:**")
        table_items = ["Dining Table", "Small Dining Table", "Coffee Table", "Desk", "Writing Desk", "Computer Desk", "Nightstand", "TV Stand"]
        for item in table_items:
            if item in [f["name"] for f in furniture_catalog]:
                furniture_item = next((f for f in furniture_catalog if f["name"] == item), None)
                if furniture_item:
                    if str(furniture_item.get('category', '')).lower() not in allowed_categories:
                        continue
                    width_m = furniture_item['width'] / 100  # Convert to meters for display
                    depth_m = furniture_item['depth'] / 100  # Convert to meters for display
                    height_m = furniture_item.get('height', 80) / 100  # Convert to meters for display
                    description = furniture_item.get('description', '')
                    
                    icon = "🪑" if "Table" in item else "💻" if "Desk" in item else "📺" if "TV" in item else "🛏️" if "Nightstand" in item else "📦"
                    
                    if st.checkbox(f"{icon} {item} ({width_m:.1f}×{depth_m:.1f}×{height_m:.1f}m)", key=f"cb_{item}_table"):
                        selected_furniture.append(furniture_item)
                        st.caption(f"📏 {description}")
        
        st.markdown("**Seating:**")
        seating_items = ["Dining Chair", "Office Chair", "Armchair", "Accent Chair", "Bar Stool", "Bench", "Ottoman"]
        for item in seating_items:
            if item in [f["name"] for f in furniture_catalog]:
                furniture_item = next((f for f in furniture_catalog if f["name"] == item), None)
                if furniture_item:
                    if str(furniture_item.get('category', '')).lower() not in allowed_categories:
                        continue
                    width_m = furniture_item['width'] / 100  # Convert to meters for display
                    depth_m = furniture_item['depth'] / 100  # Convert to meters for display
                    height_m = furniture_item.get('height', 80) / 100  # Convert to meters for display
                    description = furniture_item.get('description', '')
                    
                    icon = "🪑" if "Chair" in item else "🪑" if "Stool" in item else "🛋️" if "Bench" in item else "🪑" if "Ottoman" in item else "📦"
                    
                    if st.checkbox(f"{icon} {item} ({width_m:.1f}×{depth_m:.1f}×{height_m:.1f}m)", key=f"cb_{item}_seat"):
                        selected_furniture.append(furniture_item)
                        st.caption(f"📏 {description}")
    
    # Tab 3: Storage
    with tabs[2]:
        st.markdown("**Storage Solutions:**")
        storage_items = ["Wardrobe", "Closet", "Dresser", "Chest of Drawers", "Bookshelf", "Bookcase", "TV Stand", "Entertainment Center", "Cabinet", "Filing Cabinet"]
        for item in storage_items:
            if item in [f["name"] for f in furniture_catalog]:
                furniture_item = next((f for f in furniture_catalog if f["name"] == item), None)
                if furniture_item:
                    if str(furniture_item.get('category', '')).lower() not in allowed_categories:
                        continue
                    width_m = furniture_item['width'] / 100  # Convert to meters for display
                    depth_m = furniture_item['depth'] / 100  # Convert to meters for display
                    height_m = furniture_item.get('height', 80) / 100  # Convert to meters for display
                    description = furniture_item.get('description', '')
                    
                    icon = "🚪" if "Wardrobe" in item or "Closet" in item else "📚" if "Book" in item else "📺" if "TV" in item else "📦"
                    
                    if st.checkbox(f"{icon} {item} ({width_m:.1f}×{depth_m:.1f}×{height_m:.1f}m)", key=f"cb_{item}_storage"):
                        selected_furniture.append(furniture_item)
                        st.caption(f"📏 {description}")
    
    # Tab 4: Electronics & Lighting
    with tabs[3]:
        st.markdown("**Electronics:**")
        electronics_items = ["TV 55 inch", "TV 65 inch", "TV 75 inch", "Soundbar", "Speaker", "Floor Speaker", "Subwoofer", "Laptop", "Desktop Computer", "Monitor", "Printer"]
        for item in electronics_items:
            if item in [f["name"] for f in furniture_catalog]:
                furniture_item = next((f for f in furniture_catalog if f["name"] == item), None)
                if furniture_item:
                    if str(furniture_item.get('category', '')).lower() not in allowed_categories:
                        continue
                    width_m = furniture_item['width'] / 100  # Convert to meters for display
                    depth_m = furniture_item['depth'] / 100  # Convert to meters for display
                    height_m = furniture_item.get('height', 80) / 100  # Convert to meters for display
                    description = furniture_item.get('description', '')
                    
                    icon = "📺" if "TV" in item else "🔊" if "Speaker" in item or "Soundbar" in item else "💻" if "Computer" in item or "Laptop" in item else "🖨️" if "Printer" in item else "📺"
                    
                    if st.checkbox(f"{icon} {item} ({width_m:.1f}×{depth_m:.1f}×{height_m:.1f}m)", key=f"cb_{item}_elec"):
                        selected_furniture.append(furniture_item)
                        st.caption(f"📏 {description}")
        
        st.markdown("**Lighting:**")
        lighting_items = ["Floor Lamp", "Table Lamp", "Desk Lamp", "Chandelier", "Pendant Light", "Wall Sconce", "Ceiling Fan"]
        for item in lighting_items:
            if item in [f["name"] for f in furniture_catalog]:
                furniture_item = next((f for f in furniture_catalog if f["name"] == item), None)
                if furniture_item:
                    # Allow lighting in all room types
                    pass
                    width_m = furniture_item['width'] / 100  # Convert to meters for display
                    depth_m = furniture_item['depth'] / 100  # Convert to meters for display
                    height_m = furniture_item.get('height', 80) / 100  # Convert to meters for display
                    description = furniture_item.get('description', '')
                    
                    icon = "💡" if "Lamp" in item else "🕯️" if "Chandelier" in item else "🌀" if "Fan" in item else "💡"
                    
                    if st.checkbox(f"{icon} {item} ({width_m:.1f}×{depth_m:.1f}×{height_m:.1f}m)", key=f"cb_{item}_light"):
                        selected_furniture.append(furniture_item)
                        st.caption(f"📏 {description}")
    
    # Tab 5: Decorative
    with tabs[4]:
        st.markdown("**Decorative Items:**")
        decorative_items = ["Large Mirror", "Wall Mirror", "Picture Frame", "Artwork", "Large Plant", "Small Plant", "Vase", "Sculpture", "Candle", "Clock"]
        for item in decorative_items:
            if item in [f["name"] for f in furniture_catalog]:
                furniture_item = next((f for f in furniture_catalog if f["name"] == item), None)
                if furniture_item:
                    if str(furniture_item.get('category', '')).lower() not in allowed_categories:
                        continue
                    width_m = furniture_item['width'] / 100  # Convert to meters for display
                    depth_m = furniture_item['depth'] / 100  # Convert to meters for display
                    height_m = furniture_item.get('height', 80) / 100  # Convert to meters for display
                    description = furniture_item.get('description', '')
                    
                    icon = "🪞" if "Mirror" in item else "🌱" if "Plant" in item else "🖼️" if "Picture" in item or "Artwork" in item else "🏺" if "Vase" in item else "🕯️" if "Candle" in item else "🕐" if "Clock" in item else "🎨"
                    
                    if st.checkbox(f"{icon} {item} ({width_m:.1f}×{depth_m:.1f}×{height_m:.1f}m)", key=f"cb_{item}_decor"):
                        selected_furniture.append(furniture_item)
                        st.caption(f"📏 {description}")
    
    # Show selected furniture summary with real-world dimensions
    if selected_furniture:
        st.success(f"✅ Selected {len(selected_furniture)} furniture items!")
        st.markdown("**Selected Items with Real-World Dimensions:**")
        
        # Create a more detailed display
        for item in selected_furniture:
            width_m = item['width'] / 100  # Convert to meters for display
            depth_m = item['depth'] / 100  # Convert to meters for display
            height_m = item.get('height', 80) / 100  # Convert to meters for display
            category = item.get('category', 'furniture')
            description = item.get('description', '')
            
            # Get appropriate icon based on category
            icon_map = {
                'bedroom': '🛏️', 'living': '🛋️', 'office': '💻', 'dining': '🍽️',
                'kitchen': '🍳', 'bathroom': '🚿', 'outdoor': '🌳', 'architectural': '🏗️'
            }
            icon = icon_map.get(category, '📦')
            
            # Display with real-world context in meters
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; margin: 5px 0;">
                <strong>{icon} {item['name']}</strong><br>
                <span style="color: #666;">Dimensions: {width_m:.1f}m × {depth_m:.1f}m × {height_m:.1f}m</span><br>
                <span style="color: #888; font-size: 0.9em;">{description}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please select at least one furniture item to place in your room.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 4: AI Analysis and Optimization
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown('<h3>🤖 Step 4: AI Analysis & Optimization</h3>', unsafe_allow_html=True)
    # Backend selection for detector
    detector_backend_label = st.selectbox(
        "Architectural detection backend",
        ["Enhanced (VLM + Depth)", "CNN"],
        index=1,
        help="Enhanced uses a vision-language model + depth + CV heuristics. Falls back to CNN if unavailable."
    )
    _backend_key = "enhanced" if detector_backend_label.startswith("Enhanced") else "cnn"
    
    if uploaded_file and selected_furniture:
        if st.button("🚀 Start AI Room Optimization", type="primary", use_container_width=True):
            # Initialize AI Room Optimizer
            with st.spinner("🧠 Initializing AI Room Optimizer..."):
                st.session_state.room_optimizer = AIRoomOptimizer(
                    room_dims=(room_width_cm, room_height_cm),
                    device="cpu",  # Use CPU for compatibility
                    room_type=room_type,
                    known_length_cm=known_door_width_cm if enable_known_length else None,
                    known_length_type="door_width",
                    show_detections_overlay=show_detections_overlay,
                    detector_backend=_backend_key
                )
            
            # Process the room
            try:
                # Save uploaded file temporarily
                tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                image.save(tmp_file.name)
                tmp_file_path = tmp_file.name
                
                # Run complete pipeline
                with st.spinner(f"🔍 Analyzing room with {'Enhanced' if _backend_key=='enhanced' else 'CNN'} detector..."):
                    print(f"[DEBUG] Starting optimization with {len(selected_furniture)} furniture items")
                    print(f"[DEBUG] Room dimensions: {room_width_cm}x{room_height_cm} cm")
                    
                    # Ensure furniture objects have required fields
                    processed_furniture = []
                    for item in selected_furniture:
                        processed_item = {
                            "name": item.get("name", "Unknown"),
                            "type": item.get("name", "Unknown").lower().replace(" ", "_"),
                            "w": float(item.get("width", 100)),
                            "h": float(item.get("depth", 100)),
                            "x": 0,  # Will be optimized
                            "y": 0,  # Will be optimized
                            "height": float(item.get("height", 80)),
                            "prefer": item.get("prefer", "center"),
                            "category": item.get("category", "furniture")
                        }
                        processed_furniture.append(processed_item)
                    
                    print(f"[DEBUG] Processed furniture: {len(processed_furniture)} items")
                    for item in processed_furniture:
                        print(f"  - {item['name']}: {item['w']}x{item['h']}x{item['height']} cm")
                    
                    results = st.session_state.room_optimizer.process_empty_room(
                        image_path=tmp_file_path,
                        selected_furniture=processed_furniture,
                        user_preferences={
                            "bed_near_wall": True,
                            "table_near_window": True,
                            "min_distance": 20
                        }
                    )
                    
                    print(f"[DEBUG] Optimization completed. Results keys: {list(results.keys())}")
                    print(f"[DEBUG] Optimized furniture count: {len(results.get('optimized_furniture', []))}")
                    print(f"[DEBUG] Layout image size: {len(results.get('layout_image', b'').getvalue()) if results.get('layout_image') else 0} bytes")
                
                st.session_state.analysis_complete = True
                st.session_state.optimization_complete = True
                st.session_state.results = results
                
                # Clean up temporary file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                
                st.success("🎉 AI Room Optimization Complete!")
                
            except Exception as e:
                st.error(f"❌ Optimization failed: {str(e)}")
                st.error(f"Error type: {type(e).__name__}")
                
                # Show detailed error information in expander
                with st.expander("🔍 Detailed Error Information"):
                    import traceback
                    st.code(traceback.format_exc())
                
                st.session_state.analysis_complete = False
                st.session_state.optimization_complete = False
    else:
        st.info("👆 Please upload an image and select furniture to start optimization")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 5: Results Display
    if st.session_state.get('analysis_complete', False) and st.session_state.get('optimization_complete', False):
        results = st.session_state.get('results', {})
        
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown('<h3>📊 Optimization Results</h3>', unsafe_allow_html=True)
        
        # Display metrics with real-world dimensions
        metrics = results.get('metrics', {})
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric">
                <div class="metric-value">{metrics.get('room_area_m2', 0):.1f}</div>
                <div class="metric-label">Room Area (m²)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric">
                <div class="metric-value">{metrics.get('furniture_items_count', 0)}</div>
                <div class="metric-label">Furniture Items</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric">
                <div class="metric-value">{metrics.get('space_utilization_percent', 0):.0f}%</div>
                <div class="metric-label">Space Used</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric">
                <div class="metric-value">{metrics.get('furniture_volume_m3', 0):.1f}</div>
                <div class="metric-label">Volume (m³)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric">
                <div class="metric-value">{metrics.get('optimization_score', 0):.0f}</div>
                <div class="metric-label">AI Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display optimized layout
        st.markdown("### 🏠 Optimized Room Layout")
        layout_image = results.get('layout_image')
        if layout_image:
            st.image(layout_image, caption="AI-Optimized Room Layout", use_container_width=True)

        # Raw detections summary
        with st.expander("🔍 Show detected objects (from image)"):
            dets = results.get('room_analysis', {}).get('detections', [])
            if dets:
                # Compact table-like listing
                for d in dets[:200]:
                    t = str(d.get('type') or d.get('label') or 'unknown')
                    c = float(d.get('confidence', 0.0))
                    st.write(f"• {t} — conf {c:.2f}")
                if len(dets) > 200:
                    st.caption(f"… and {len(dets) - 200} more")
            else:
                st.info("No detections to display.")
        
        # Display recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            st.markdown("### 💡 AI Recommendations")
            for rec in recommendations:
                st.info(f"💡 {rec}")
        
        # Export options
        st.markdown("### 📥 Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            # Export layout data
            export_data = {
                "room_dimensions": {"width": room_width, "height": room_height, "width_cm": room_width_cm, "height_cm": room_height_cm},
                "architectural_elements": results.get('room_analysis', {}).get('detections', []),
                "furniture_layout": results.get('optimized_furniture', []),
                "metrics": metrics
            }
            json_bytes = json.dumps(export_data, indent=2).encode("utf-8")
            st.download_button(
                "📄 Download Layout Data (JSON)",
                data=json_bytes,
                file_name="ai_optimized_layout.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # Export layout image
            if layout_image:
                st.download_button(
                    "🖼️ Download Layout Image (PNG)",
                    data=layout_image.getvalue(),
                    file_name="ai_optimized_layout.png",
                    mime="image/png",
                    use_container_width=True
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🏠 AI Room Optimizer - Powered by CNN + Genetic Algorithms</p>
        <p>Transform your space with advanced AI technology</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
