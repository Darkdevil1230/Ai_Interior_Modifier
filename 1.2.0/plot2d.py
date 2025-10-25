"""plot2d.py
---------
2D visualization of the optimized room layout using matplotlib.

Improvements:
 - Supports saving into a BytesIO buffer for direct Streamlit display.
 - Draws objects with distinct colors and labels, shows confidence when present.
 - Distinguishes architectural elements (windows, doors) from furniture.
"""

import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import math

# Architectural elements that should be displayed differently
ARCHITECTURAL_ELEMENTS = {"window", "door", "wall", "floor", "ceiling", "fireplace", "entry", "entrance", "step", "stairs"}

# Color schemes for different object types
ARCHITECTURAL_COLORS = {
    "window": (0.6, 0.8, 1.0),  # Light blue
    "door": (0.8, 0.6, 0.4),     # Brown
    "wall": (0.7, 0.7, 0.7),     # Gray
    "floor": (0.9, 0.9, 0.8),    # Beige
    "ceiling": (0.95, 0.95, 0.95) # Off-white
}

def _color_for_label(label: str):
    """Get color for an object based on its type."""
    label_lower = label.lower()
    
    # Check if it's an architectural element
    if label_lower in ARCHITECTURAL_COLORS:
        return ARCHITECTURAL_COLORS[label_lower]
    
    # For furniture, use deterministic but varied colors
    random.seed(hash(label) & 0xFFFFFFFF)
    return (random.random() * 0.5 + 0.3, random.random() * 0.5 + 0.3, random.random() * 0.5 + 0.3)

def _is_architectural(label: str) -> bool:
    """Check if an object is an architectural element."""
    return label.lower() in ARCHITECTURAL_ELEMENTS

def plot_layout(room_dims, layout, save_path=None, save_buffer=None):
    """
    Plot the room and furniture layout with clear distinction between
    architectural elements and furniture.
    :param room_dims: (width_cm, height_cm)
    :param layout: List of objects with type, x, y, w, h, optional confidence
    :param save_path: optional filesystem path to save PNG
    :param save_buffer: optional BytesIO buffer to write PNG (preferred for Streamlit)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    # Add margins for dimension callouts outside the room
    margin = max(40, min(room_dims) * 0.15)
    ax.set_xlim(-margin, room_dims[0] + margin)
    ax.set_ylim(-margin, room_dims[1] + margin)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # so origin (0,0) appears top-left like images

    # Floor-plan style: thick outer walls and hatched floor area
    wall_thickness = max(6, min(room_dims) * 0.012)
    # Outer wall
    outer = patches.Rectangle((0, 0), room_dims[0], room_dims[1],
                              linewidth=6, edgecolor="black",
                              facecolor="none")
    ax.add_patch(outer)
    # Inner floor area with hatch
    inset = max(2, wall_thickness)
    floor = patches.Rectangle((inset, inset), room_dims[0]-2*inset, room_dims[1]-2*inset,
                              linewidth=1.0, edgecolor="#999",
                              facecolor="white", alpha=1.0)
    ax.add_patch(floor)

    # ---------- helpers ----------
    def _draw_dimension(ax, p1, p2, offset_vec, text, text_offset=8):
        """Draw a double-arrow dimension between p1 and p2, offset by offset_vec (dx, dy)."""
        (x1, y1), (x2, y2) = p1, p2
        dx, dy = offset_vec
        ax.annotate("", xy=(x1+dx, y1+dy), xytext=(x2+dx, y2+dy),
                    arrowprops=dict(arrowstyle="<->", color="#222", lw=1.5))
        tx = (x1 + x2)/2 + dx
        ty = (y1 + y2)/2 + dy
        ax.text(tx, ty - text_offset, text, ha="center", va="center", fontsize=9, color="#111")

    def _fmt_cm(v):
        return f"{int(round(v))} cm"

    def _draw_bed(ax, x, y, w, h, edge="#000", face="#ffffff"):
        rr = max(6, min(18, min(w, h) * 0.18))
        bed = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={rr}",
                                     linewidth=1.8, edgecolor=edge, facecolor=face, alpha=0.95)
        ax.add_patch(bed)
        # pillows at the top (assume y is top since inverted axis)
        pillow_h = h * 0.22
        pillow_w = w * 0.38
        gap = w * 0.06
        px1 = x + w*0.1
        px2 = x + w - w*0.1 - pillow_w
        py = y + 4
        for px in (px1, px2):
            ax.add_patch(patches.FancyBboxPatch((px, py), pillow_w, pillow_h,
                          boxstyle=f"round,pad=0.02,rounding_size={rr*0.6}",
                          linewidth=1.2, edgecolor="#777", facecolor="#fff", alpha=0.95))
    
    def _draw_chair(ax, x, y, w, h, edge="#000", face="#ffffff"):
        """Draw a realistic chair shape."""
        rr = max(4, min(12, min(w, h) * 0.2))
        # Chair back
        back_h = h * 0.6
        back_x = x + w * 0.2
        back_w = w * 0.6
        ax.add_patch(patches.FancyBboxPatch((back_x, y), back_w, back_h,
                      boxstyle=f"round,pad=0.02,rounding_size={rr*0.7}",
                      linewidth=1.5, edgecolor=edge, facecolor=face, alpha=0.95))
        # Chair seat
        seat_h = h * 0.4
        seat_y = y + back_h
        ax.add_patch(patches.FancyBboxPatch((x, seat_y), w, seat_h,
                      boxstyle=f"round,pad=0.02,rounding_size={rr*0.5}",
                      linewidth=1.5, edgecolor=edge, facecolor=face, alpha=0.95))
        # Armrests
        arm_h = seat_h * 0.7
        arm_w = w * 0.12
        arm_y = seat_y + seat_h * 0.15
        ax.add_patch(patches.FancyBboxPatch((x, arm_y), arm_w, arm_h,
                      boxstyle=f"round,pad=0.02,rounding_size={rr*0.4}",
                      linewidth=1.2, edgecolor=edge, facecolor=face, alpha=0.95))
        ax.add_patch(patches.FancyBboxPatch((x + w - arm_w, arm_y), arm_w, arm_h,
                      boxstyle=f"round,pad=0.02,rounding_size={rr*0.4}",
                      linewidth=1.2, edgecolor=edge, facecolor=face, alpha=0.95))
    
    def _draw_recliner(ax, x, y, w, h, edge="#000", face="#ffffff"):
        """Draw a recliner chair shape."""
        rr = max(5, min(15, min(w, h) * 0.15))
        # Main body
        body = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={rr}",
                                     linewidth=1.6, edgecolor=edge, facecolor=face, alpha=0.95)
        ax.add_patch(body)
        # Backrest
        back_h = h * 0.65
        back_x = x + w * 0.15
        back_w = w * 0.7
        ax.add_patch(patches.FancyBboxPatch((back_x, y), back_w, back_h,
                      boxstyle=f"round,pad=0.02,rounding_size={rr*0.8}",
                      linewidth=1.3, edgecolor=edge, facecolor="#b8a898", alpha=0.95))
        # Footrest
        foot_h = h * 0.25
        foot_y = y + h - foot_h
        ax.add_patch(patches.FancyBboxPatch((x + w*0.15, foot_y), w*0.7, foot_h,
                      boxstyle=f"round,pad=0.02,rounding_size={rr*0.6}",
                      linewidth=1.2, edgecolor=edge, facecolor="#b8a898", alpha=0.95))
    
    def _draw_accent_chair(ax, x, y, w, h, edge="#000", face="#ffffff"):
        """Draw an accent chair (decorative style)."""
        rr = max(5, min(14, min(w, h) * 0.18))
        # Main body with rounded top
        main_h = h * 0.8
        ax.add_patch(patches.FancyBboxPatch((x, y), w, main_h,
                      boxstyle=f"round,pad=0.02,rounding_size={rr}",
                      linewidth=1.5, edgecolor=edge, facecolor=face, alpha=0.95))
        # Decorative back
        back_h = h * 0.6
        back_x = x + w * 0.25
        back_w = w * 0.5
        ax.add_patch(patches.FancyBboxPatch((back_x, y), back_w, back_h,
                      boxstyle=f"round,pad=0.02,rounding_size={rr*0.9}",
                      linewidth=1.3, edgecolor=edge, facecolor="#d5c5b5", alpha=0.95))
        # Base
        base_h = h * 0.25
        base_y = y + main_h
        ax.add_patch(patches.FancyBboxPatch((x + w*0.15, base_y), w*0.7, base_h,
                      boxstyle=f"round,pad=0.02,rounding_size={rr*0.5}",
                      linewidth=1.2, edgecolor=edge, facecolor="#d5c5b5", alpha=0.95))

    def _draw_sofa(ax, x, y, w, h, edge="#000", face="#ffffff"):
        rr = max(6, min(18, min(w, h) * 0.2))
        body = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={rr}",
                                      linewidth=1.8, edgecolor=edge, facecolor=face, alpha=0.95)
        ax.add_patch(body)
        # backrest at top
        back_h = h * 0.22
        ax.add_patch(patches.FancyBboxPatch((x+2, y+2), w-4, back_h,
                      boxstyle=f"round,pad=0.02,rounding_size={rr*0.6}",
                      linewidth=1.2, edgecolor="#666", facecolor="#e6e6e6", alpha=0.95))
        # armrests
        arm_w = max(6, w * 0.08)
        ax.add_patch(patches.FancyBboxPatch((x+2, y+back_h+2), arm_w, h-back_h-4,
                      boxstyle=f"round,pad=0.02,rounding_size={rr*0.5}", linewidth=1.2,
                      edgecolor="#666", facecolor="#e6e6e6", alpha=0.95))
        ax.add_patch(patches.FancyBboxPatch((x+w-arm_w-2, y+back_h+2), arm_w, h-back_h-4,
                      boxstyle=f"round,pad=0.02,rounding_size={rr*0.5}", linewidth=1.2,
                      edgecolor="#666", facecolor="#e6e6e6", alpha=0.95))
        # Cushion divisions (3 cushions for sofa/couch)
        if w > 40:
            cushion_div1 = x + w / 3
            cushion_div2 = x + 2 * w / 3
            ax.plot([cushion_div1, cushion_div1], [y+back_h+2, y+h-2], color="#999", linewidth=1, linestyle="--")
            ax.plot([cushion_div2, cushion_div2], [y+back_h+2, y+h-2], color="#999", linewidth=1, linestyle="--")
    
    def _draw_loveseat(ax, x, y, w, h, edge="#000", face="#ffffff"):
        """Draw a loveseat (2-seater sofa)."""
        rr = max(5, min(16, min(w, h) * 0.2))
        body = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={rr}",
                                      linewidth=1.8, edgecolor=edge, facecolor=face, alpha=0.95)
        ax.add_patch(body)
        # backrest
        back_h = h * 0.25
        ax.add_patch(patches.FancyBboxPatch((x+2, y+2), w-4, back_h,
                      boxstyle=f"round,pad=0.02,rounding_size={rr*0.6}",
                      linewidth=1.2, edgecolor="#666", facecolor="#d8d8d8", alpha=0.95))
        # armrests (shorter)
        arm_w = max(5, w * 0.1)
        ax.add_patch(patches.FancyBboxPatch((x+2, y+back_h+2), arm_w, h-back_h-4,
                      boxstyle=f"round,pad=0.02,rounding_size={rr*0.5}", linewidth=1.2,
                      edgecolor="#666", facecolor="#d8d8d8", alpha=0.95))
        ax.add_patch(patches.FancyBboxPatch((x+w-arm_w-2, y+back_h+2), arm_w, h-back_h-4,
                      boxstyle=f"round,pad=0.02,rounding_size={rr*0.5}", linewidth=1.2,
                      edgecolor="#666", facecolor="#d8d8d8", alpha=0.95))
        # Cushion division (2 cushions for loveseat)
        cushion_div = x + w / 2
        ax.plot([cushion_div, cushion_div], [y+back_h+2, y+h-2], color="#999", linewidth=1, linestyle="--")

    def _draw_circle_table(ax, x, y, w, h, edge="#333", face="#efe7da"):
        r = min(w, h)/2
        circ = patches.Ellipse((x + w/2, y + h/2), width=min(w, h), height=min(w, h),
                               linewidth=1.6, edgecolor=edge, facecolor=face, alpha=0.95)
        ax.add_patch(circ)
    
    def _draw_table(ax, x, y, w, h, edge="#000", face="#ffffff"):
        """Draw a rectangular table."""
        rr = max(4, min(12, min(w, h) * 0.1))
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h,
                      boxstyle=f"round,pad=0.02,rounding_size={rr}",
                      linewidth=1.6, edgecolor=edge, facecolor=face, alpha=0.95))
        # Table legs (corners)
        leg_size = min(w, h) * 0.08
        for cx, cy in [(x+leg_size/2, y+h-leg_size/2), (x+w-leg_size/2, y+h-leg_size/2),
                        (x+leg_size/2, y+leg_size/2), (x+w-leg_size/2, y+leg_size/2)]:
            ax.add_patch(patches.Rectangle((cx-leg_size/2, cy-leg_size/2), leg_size, leg_size,
                          linewidth=1, edgecolor="#555", facecolor="#666", alpha=0.8))
    
    def _draw_coffee_table(ax, x, y, w, h, edge="#000", face="#ffffff"):
        """Draw a coffee table (low table in front of sofa)."""
        rr = max(5, min(14, min(w, h) * 0.12))
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h,
                      boxstyle=f"round,pad=0.02,rounding_size={rr}",
                      linewidth=1.6, edgecolor=edge, facecolor=face, alpha=0.95))
        # Decorative details
        if w > 60:
            # Center design element
            cx, cy = x + w/2, y + h/2
            ax.add_patch(patches.Circle((cx, cy), min(w, h)*0.15,
                          linewidth=1.2, edgecolor=edge, facecolor="#c8b8a8", alpha=0.6))
    
    def _draw_accent_table(ax, x, y, w, h, edge="#000", face="#ffffff"):
        """Draw an accent table (small decorative table, rotated 45 degrees)."""
        # Rotate 45 degrees by drawing a diamond shape
        cx, cy = x + w/2, y + h/2
        # Calculate rotated corners
        size = min(w, h) / 2 * 0.85  # Scale down slightly
        # Four corners of rotated square
        corners = [
            (cx, cy - size),  # top
            (cx + size, cy),  # right
            (cx, cy + size),  # bottom
            (cx - size, cy)   # left
        ]
        ax.add_patch(patches.Polygon(corners, closed=True,
                      linewidth=1.6, edgecolor=edge, facecolor=face, alpha=0.95))

    def _draw_rect_furniture(ax, x, y, w, h, edge="#000", face="#ffffff"):
        rr = max(6, min(16, min(w, h) * 0.15))
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h,
                    boxstyle=f"round,pad=0.02,rounding_size={rr}", linewidth=1.6,
                    edgecolor=edge, facecolor=face, alpha=0.9))
    
    def _draw_sofa_table(ax, x, y, w, h, edge="#000", face="#ffffff"):
        """Draw a sofa table (long narrow table behind sofa)."""
        rr = max(3, min(8, min(w, h) * 0.08))
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h,
                      boxstyle=f"round,pad=0.02,rounding_size={rr}",
                      linewidth=1.6, edgecolor=edge, facecolor=face, alpha=0.95))

    def _draw_plant(ax, x, y, w, h):
        cx, cy = x + w/2, y + h/2
        r = min(w, h) * 0.35
        ax.add_patch(patches.Circle((cx, cy), r, edgecolor="#000", facecolor="#fff", lw=1.2, alpha=1.0))
    
    def _draw_bookshelf(ax, x, y, w, h, edge="#000", face="#ffffff"):
        """Draw a bookshelf with shelves."""
        _draw_rect_furniture(ax, x, y, w, h, edge=edge, face=face)
        # Shelf lines
        n = max(2, int(h // 25))
        for i in range(1, n):
            yy = y + i * (h / n)
            ax.plot([x+3, x+w-3], [yy, yy], color="#111", lw=1)
        # Draw some vertical divisions
        if w > 40:
            div_x = x + w / 3
            ax.plot([div_x, div_x], [y+3, y+h-3], color="#111", lw=1)
            div_x2 = x + 2*w / 3
            ax.plot([div_x2, div_x2], [y+3, y+h-3], color="#999", lw=1)
    
    def _draw_tv(ax, x, y, w, h, edge="#000", face="#ffffff"):
        """Draw a TV/Television."""
        rr = max(3, min(8, min(w, h) * 0.08))
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h,
                      boxstyle=f"round,pad=0.02,rounding_size={rr}",
                      linewidth=1.6, edgecolor=edge, facecolor=face, alpha=0.95))
        # Screen center
        cx, cy = x + w/2, y + h/2
        screen_w = w * 0.85
        screen_h = h * 0.85
        ax.add_patch(patches.Rectangle((cx-screen_w/2, cy-screen_h/2), screen_w, screen_h,
                      linewidth=1.0, edgecolor="#000", facecolor="#fff", alpha=1.0))
        # Stand base
        stand_w = w * 0.6
        stand_h = h * 0.15
        stand_x = cx - stand_w/2
        stand_y = y + h - stand_h
        ax.add_patch(patches.Rectangle((stand_x, stand_y), stand_w, stand_h,
                      linewidth=1, edgecolor="#000", facecolor="#1a1a1a", alpha=0.95))
    
    def _draw_speaker(ax, x, y, w, h, edge="#000", face="#ffffff"):
        """Draw a speaker."""
        rr = max(3, min(6, min(w, h) * 0.1))
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h,
                      boxstyle=f"round,pad=0.02,rounding_size={rr}",
                      linewidth=1.5, edgecolor=edge, facecolor=face, alpha=0.95))
        # Speaker grille
        grille_x = x + w * 0.2
        grille_y = y + h * 0.15
        grille_w = w * 0.6
        grille_h = h * 0.7
        ax.add_patch(patches.Rectangle((grille_x, grille_y), grille_w, grille_h,
                      linewidth=1, edgecolor="#000", facecolor="#fff", alpha=1.0))
        # Speaker cone (dots)
        cx, cy = grille_x + grille_w/2, grille_y + grille_h/2
        for i in range(3):
            for j in range(3):
                dot_x = cx - grille_w/3 + i * grille_w/3
                dot_y = cy - grille_h/3 + j * grille_h/3
                ax.add_patch(patches.Circle((dot_x, dot_y), min(w, h)*0.05,
                              edgecolor="#000", facecolor="#fff", alpha=1.0))
    
    def _draw_fireplace(ax, x, y, w, h, edge="#000", face="#ffffff"):
        """Draw a fireplace."""
        rr = max(8, min(20, min(w, h) * 0.3))
        # Main body with curved front
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h,
                      boxstyle=f"round,pad=0.02,rounding_size={rr}",
                      linewidth=2, edgecolor=edge, facecolor=face, alpha=0.95))
        # Opening in center
        opening_w = w * 0.5
        opening_h = h * 0.6
        opening_x = x + (w - opening_w) / 2
        opening_y = y + h * 0.1
        ax.add_patch(patches.Rectangle((opening_x, opening_y), opening_w, opening_h,
                      linewidth=1.5, edgecolor="#333", facecolor="#8a7a68", alpha=0.9))
        # Mantle shelf
        mantle_h = h * 0.1
        mantle_y = y + h * 0.85
        ax.add_patch(patches.Rectangle((x-2, mantle_y), w+4, mantle_h,
                      linewidth=1.5, edgecolor="#444", facecolor="#d8c8b8", alpha=0.95))
    
    def _draw_entry(ax, x, y, w, h, edge="#000", face="#ffffff"):
        """Draw an entry/entrance."""
        rr = max(3, min(8, min(w, h) * 0.15))
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h,
                      boxstyle=f"round,pad=0.02,rounding_size={rr}",
                      linewidth=2, edgecolor=edge, facecolor=face, alpha=0.95))
        # Entry opening
        opening_w = w * 0.7
        opening_h = h * 0.5
        opening_x = x + (w - opening_w) / 2
        opening_y = y + h * 0.25
        ax.add_patch(patches.Rectangle((opening_x, opening_y), opening_w, opening_h,
                      linewidth=1.5, edgecolor="#666", facecolor="#e8e8e8", alpha=0.8))
    
    def _draw_step(ax, x, y, w, h, edge="#777", face="#d8d8d8"):
        """Draw a step/stairs."""
        # Step platform
        ax.add_patch(patches.Rectangle((x, y), w, h,
                      linewidth=1.5, edgecolor=edge, facecolor=face, alpha=0.95))
        # Step riser lines for depth indication
        for i in range(3):
            line_y = y + i * (h / 3)
            ax.plot([x, x+w], [line_y, line_y], color="#999", linewidth=1)

    def _draw_shelves(ax, x, y, w, h):
        _draw_rect_furniture(ax, x, y, w, h, edge="#444", face="#e8e8e8")
        # shelf lines
        n = max(2, int(h // 20))
        for i in range(1, n):
            yy = y + i * (h / n)
            ax.plot([x+4, x+w-4], [yy, yy], color="#999", lw=1)

    # Separate architectural elements from furniture for better rendering
    architectural_objs = []
    furniture_objs = []
    
    for obj in layout:
        label = str(obj.get("type", "obj"))
        if _is_architectural(label):
            architectural_objs.append(obj)
        else:
            furniture_objs.append(obj)
    
    # Draw architectural elements first (they're fixed/background)
    for obj in architectural_objs:
        label = str(obj.get("type", "obj"))
        x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
        color = _color_for_label(label)
        
        label_l = label.lower()
        # Windows: draw slim element aligned to its rectangle
        if label_l == "window":
            rect = patches.Rectangle((x, y), w, h,
                                     linewidth=2.0,
                                     edgecolor="#0066cc",
                                     facecolor="#b7d8ff",
                                     alpha=0.9)
            ax.add_patch(rect)
            ax.plot([x, x+w], [y+h/2, y+h/2], color="#004c99", linewidth=2)
        # Doors: draw swing arc and leaf
        elif label_l == "door":
            # Approximate a hinge at the rectangle's lower-left corner in image coords
            rx, ry = x, y+h
            r = max(10, min(w, h))
            arc = patches.Arc((rx, ry), 2*r, 2*r, angle=0, theta1=270, theta2=270+90,
                               linewidth=2, color="#8b5a2b")
            ax.add_patch(arc)
            ax.plot([rx, rx+r], [ry, ry], color="#8b5a2b", linewidth=3)
            ax.plot([rx+r, rx+r], [ry, ry-r], color="#8b5a2b", linewidth=3)
        elif label_l == "fireplace":
            _draw_fireplace(ax, x, y, w, h)
        elif label_l in ["entry", "entrance"]:
            _draw_entry(ax, x, y, w, h)
        elif label_l in ["step", "stairs"]:
            _draw_step(ax, x, y, w, h)
        else:
            # Generic architectural element as dashed shape
            rect = patches.Rectangle((x, y), w, h,
                                     linewidth=2,
                                     edgecolor="darkblue",
                                     facecolor=color,
                                     alpha=0.6,
                                     linestyle="--")
            ax.add_patch(rect)
        
        # Minimal label
        txt = f"{label.upper()}"
        ax.text(x + w / 2.0, y + h / 2.0, txt,
                ha="center", va="center",
                fontsize=8, color="#253858",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white",
                          edgecolor="#b0c4de",
                          alpha=0.8))

    
    # Draw furniture (movable objects)
    for obj in furniture_objs:
        label = str(obj.get("type", "obj"))
        x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
        color = _color_for_label(label)
        
        # Choose a specific renderer per type
        t = label.lower()
        if "bed" in t:
            _draw_bed(ax, x, y, w, h)
        elif "loveseat" in t:
            _draw_loveseat(ax, x, y, w, h)
        elif "sofa" in t or "couch" in t:
            _draw_sofa(ax, x, y, w, h)
        elif "recliner" in t:
            _draw_recliner(ax, x, y, w, h)
        elif "accent chair" in t or ("chair" in t and h < 60):
            _draw_accent_chair(ax, x, y, w, h)
        elif "chair" in t:
            _draw_chair(ax, x, y, w, h)
        elif "coffee table" in t:
            _draw_coffee_table(ax, x, y, w, h)
        elif "sofa table" in t:
            _draw_sofa_table(ax, x, y, w, h)
        elif "accent table" in t:
            _draw_accent_table(ax, x, y, w, h)
        elif "round table" in t or ("table" in t and abs(w - h) < 0.3 * max(w, h)):
            _draw_circle_table(ax, x, y, w, h)
        elif "table" in t:
            _draw_table(ax, x, y, w, h)
        elif "plant" in t:
            _draw_plant(ax, x, y, w, h)
        elif "bookshelf" in t or "bookshelf" in t:
            _draw_bookshelf(ax, x, y, w, h)
        elif "shelf" in t:
            _draw_shelves(ax, x, y, w, h)
        elif "tv" in t or "television" in t:
            _draw_tv(ax, x, y, w, h)
        elif "speaker" in t:
            _draw_speaker(ax, x, y, w, h)
        else:
            _draw_rect_furniture(ax, x, y, w, h)
        
        # Label - positioned appropriately based on furniture type
        txt = label.title()
        # For small items or architectural elements, use text without box
        is_small = w < 30 or h < 30
        is_arch = _is_architectural(label)
        
        if is_small or is_arch:
            ax.text(x + w / 2.0, y + h / 2.0, txt,
                    ha="center", va="center",
                    fontsize=7, color="#333", weight="bold")
        else:
            ax.text(x + w / 2.0, y + h / 2.0, txt,
                    ha="center", va="center",
                    fontsize=8.5, color="#111",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white",
                              alpha=0.75))

    # clean axes
    ax.grid(False)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.axis('off')

    # Room dimension annotation in the center
    area_text = f"{int((room_dims[0]*room_dims[1])/10000)}.00 sq. m" if room_dims[0] and room_dims[1] else ""
    if area_text:
        ax.text(room_dims[0]/2, room_dims[1]/2, area_text,
                ha="center", va="center", fontsize=11, color="#4a4a4a", alpha=0.9)

    # Perimeter dimensions (architectural-style)
    # Top horizontal dimension
    off = - (inset + 18)
    _draw_dimension(ax, (0, 0), (room_dims[0], 0), (0, off), _fmt_cm(room_dims[0]), text_offset=0)
    # Bottom horizontal dimension
    off_b = (inset + 26)
    _draw_dimension(ax, (0, room_dims[1]), (room_dims[0], room_dims[1]), (0, off_b), _fmt_cm(room_dims[0]), text_offset=0)
    # Left vertical dimension
    off_l = - (inset + 22)
    _draw_dimension(ax, (0, 0), (0, room_dims[1]), (off_l, 0), _fmt_cm(room_dims[1]), text_offset=0)
    # Right vertical dimension
    off_r = (inset + 30)
    _draw_dimension(ax, (room_dims[0], 0), (room_dims[0], room_dims[1]), (off_r, 0), _fmt_cm(room_dims[1]), text_offset=0)

    # Extension ticks from corners to dimension lines
    tick = 10
    # top/bottom
    for x in (0, room_dims[0]):
        ax.plot([x, x], [0, off+tick], color="#444", lw=1)
        ax.plot([x, x], [room_dims[1], room_dims[1]+off_b-tick], color="#444", lw=1)
    # left/right
    for y in (0, room_dims[1]):
        ax.plot([0, off_l+tick], [y, y], color="#444", lw=1)
        ax.plot([room_dims[0], room_dims[0]+off_r-tick], [y, y], color="#444", lw=1)

    # simple scale bar
    try:
        scale_len_cm = max(50, (room_dims[0] // 5 // 10) * 10)
        sb_w = scale_len_cm
        sb_h = max(3, min(room_dims) * 0.01)
        sb_x = room_dims[0] - inset - sb_w
        sb_y = room_dims[1] + off_b - 2*sb_h
        ax.add_patch(patches.Rectangle((sb_x, sb_y), sb_w, sb_h, facecolor="#111", edgecolor="#111"))
        ax.text(sb_x + sb_w/2, sb_y - 6, f"{int(scale_len_cm)} cm", ha="center", va="top", fontsize=9, color="#111")
    except Exception:
        pass

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white")
    if save_buffer is not None:
        save_buffer = save_buffer if hasattr(save_buffer, "write") else io.BytesIO()
        plt.savefig(save_buffer, format="png", bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    # If save_buffer given, write bytes and leave pointer at end for reading by caller
    if save_buffer is not None:
        try:
            save_buffer.seek(0)
        except Exception:
            pass
        return save_buffer