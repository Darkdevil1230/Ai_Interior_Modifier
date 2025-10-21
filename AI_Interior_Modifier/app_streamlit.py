
from io import BytesIO
import json
import tempfile

import streamlit as st
from PIL import Image

from src.input.cv_detector import RoomDetector
from optimizer import LayoutOptimizer
from plot2d import plot_layout

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
    # Add more mappings as needed
}

# Fallback mapping by index for custom models (kept for backward compatibility)
CLASS_MAP = {
    0: "wall",
    1: "door",
    2: "window",
    3: "bed",
    4: "table",
    5: "chair",
}


def map_label_to_domain(label: str):
    if not label:
        return None
    label_lower = label.lower()
    return COCO_TO_DOMAIN.get(label_lower, label_lower)


def main():
    st.title("AI Interior Modifier — Improved")
    st.write("Upload a room image and tune detection / optimization parameters. Uses YOLOv8 detection + Genetic Algorithm optimization.")

    # Image upload + room dims
    image_file = st.file_uploader("Upload Room Image", type=["jpg", "png", "jpeg"])
    room_width = st.number_input("Room Width (cm)", min_value=100, max_value=2000, value=400)
    room_height = st.number_input("Room Height (cm)", min_value=100, max_value=2000, value=300)

    # Detection / GA settings
    conf_thresh = st.slider("Detection confidence threshold", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
    show_detections_overlay = st.checkbox("Show detection overlay on uploaded image", value=True)

    st.markdown("### Genetic Algorithm parameters")
    pop_size = st.number_input("Population size", min_value=10, max_value=500, value=60)
    generations = st.number_input("Generations", min_value=10, max_value=2000, value=250)
    seed = st.number_input("Random seed (0 = random)", min_value=0, value=0)

    # Preferences
    st.markdown("### Preferences")
    bed_near_wall = st.checkbox("Prefer bed near wall", value=True)
    table_near_window = st.checkbox("Prefer table near window", value=True)
    min_distance = st.number_input("Minimum clearance between objects (cm)", min_value=0, max_value=200, value=20)

    objects = []
    image_shape = None
    detector = None
    model_name = None

    if image_file is not None:
        img = Image.open(image_file).convert("RGB")
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(tmp.name)
        image_shape = img.size[::-1]  # (height, width)
        # Updated: replaced deprecated use_container_width with width
        st.image(img, caption="Uploaded Room Image", width="stretch")

        # Initialize detector and run detection (with spinner)
        with st.spinner("Loading model and running detection..."):
            try:
                detector = RoomDetector()
                model_name = detector.get_model_name()
                st.info(f"Model loaded: {model_name}")
                detections = detector.detect(tmp.name, conf_threshold=conf_thresh)
            except Exception as e:
                st.error(f"Detection failed: {e}")
                detections = []

        if not detections:
            st.warning("No detections found. You can proceed with default objects or try another image.")
        else:
            st.subheader("Detected Objects")
            # Convert raw detections into domain objects and allow adjustments
            objects = detector.parse_detections(detections, CLASS_MAP, (room_width, room_height), image_shape)
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write("Detected (label — confidence)")
                for i, det in enumerate(detections):
                    label = det.get("label") or CLASS_MAP.get(det["class"], "unknown")
                    st.write(f"{i+1}. {label} — {det['confidence']:.2f}")
            with col2:
                st.write("Adjust sizes (cm) — detected sizes are proportional to room dims")
                for i, obj in enumerate(objects):
                    # Map label to domain term (prefer model label if available)
                    domain_label = map_label_to_domain(obj["type"]) or obj["type"]
                    obj["type"] = domain_label
                    if domain_label in ["bed", "table", "chair", "sofa"]:
                        w = st.number_input(f"{domain_label.capitalize()} {i+1} width (cm)", min_value=20, max_value=int(room_width), value=int(max(30, obj["w"])))
                        h = st.number_input(f"{domain_label.capitalize()} {i+1} depth (cm)", min_value=20, max_value=int(room_height), value=int(max(30, obj["h"])))
                        obj["w"], obj["h"] = w, h

    else:
        st.info("No image uploaded — using default furniture set. You can also upload an image for detection.")
        objects = [
            {"type": "bed", "w": 200, "h": 150},
            {"type": "table", "w": 120, "h": 80},
            {"type": "chair", "w": 50, "h": 50}
        ]
        for i, obj in enumerate(objects):
            obj["w"] = st.number_input(f"{obj['type'].capitalize()} {i+1} width (cm)", min_value=20, max_value=int(room_width), value=int(obj["w"]))
            obj["h"] = st.number_input(f"{obj['type'].capitalize()} {i+1} depth (cm)", min_value=20, max_value=int(room_height), value=int(obj["h"]))

    # Run optimization
    run = st.button("Run AI Modifier")
    if run and objects:
        with st.spinner("Optimizing layout (this may take a moment)..."):
            try:
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
                layout = ga.optimize()
                # Export JSON data to memory for safe download
                result = {
                    "metadata": {
                        "room_width_cm": room_width,
                        "room_height_cm": room_height,
                        "model": model_name or (detector.model_path if detector else None),
                        "ga_population": pop_size,
                        "ga_generations": generations
                    },
                    "layout": layout
                }
                json_bytes = json.dumps(result, indent=2).encode("utf-8")
                # Visualization
                buf = BytesIO()
                plot_layout((room_width, room_height), layout, save_buffer=buf)
                buf.seek(0)
                st.subheader("Optimized Layout")
                # Updated: replaced deprecated use_container_width with width
                st.image(buf, caption="Optimized Layout (2D)", width="stretch")
                # Download button
                st.download_button("Download optimized layout (JSON)", data=json_bytes, file_name="optimized_layout.json", mime="application/json")
                st.success("Optimization complete.")
            except Exception as e:
                st.error(f"Optimization failed: {e}")


if __name__ == "__main__":
    main()