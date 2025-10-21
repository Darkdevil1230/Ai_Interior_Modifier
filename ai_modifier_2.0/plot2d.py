"""
plot2d.py
---------
2D visualization of the optimized room layout using matplotlib.

Improvements:
 - Supports saving into a BytesIO buffer for direct Streamlit display.
 - Draws objects with distinct colors and labels, shows confidence when present.
"""

import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

def _color_for_label(label: str):
    random.seed(hash(label) & 0xFFFFFFFF)
    return (random.random() * 0.7 + 0.15, random.random() * 0.7 + 0.15, random.random() * 0.7 + 0.15)

def plot_layout(room_dims, layout, save_path=None, save_buffer=None):
    """
    Plot the room and furniture layout.
    :param room_dims: (width_cm, height_cm)
    :param layout: List of objects with type, x, y, w, h, optional confidence
    :param save_path: optional filesystem path to save PNG
    :param save_buffer: optional BytesIO buffer to write PNG (preferred for Streamlit)
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, room_dims[0])
    ax.set_ylim(0, room_dims[1])
    ax.set_aspect("equal")
    ax.invert_yaxis()  # so origin (0,0) appears top-left like images

    # Room boundary
    boundary = patches.Rectangle((0, 0), room_dims[0], room_dims[1], linewidth=2, edgecolor="black", facecolor="none")
    ax.add_patch(boundary)

    # Draw each object
    for obj in layout:
        label = str(obj.get("type", "obj"))
        x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
        color = _color_for_label(label)
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="black", facecolor=color, alpha=0.6)
        ax.add_patch(rect)
        txt = label
        if "confidence" in obj:
            txt += f"\n{obj['confidence']:.2f}"
        txt += f"\n{int(w)}x{int(h)}cm"
        ax.text(x + w / 2.0, y + h / 2.0, txt, ha="center", va="center", fontsize=8, color="black")

    ax.set_title("Optimized Layout (2D)")
    ax.set_xlabel("Width (cm)")
    ax.set_ylabel("Depth (cm)")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if save_buffer is not None:
        save_buffer = save_buffer if hasattr(save_buffer, "write") else io.BytesIO()
        plt.savefig(save_buffer, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    # If save_buffer given, write bytes and leave pointer at end for reading by caller
    if save_buffer is not None:
        try:
            save_buffer.seek(0)
        except Exception:
            pass
        return save_buffer