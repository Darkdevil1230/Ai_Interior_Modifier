"""
enhanced_layout_generator.py
---------------------------
Enhanced 2D layout generator that clearly shows architectural elements and optimized furniture.
Creates professional floor plan visualizations with detailed annotations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon, Arc
import numpy as np
from typing import List, Dict, Tuple, Optional
import io

class EnhancedLayoutGenerator:
    """Enhanced layout generator with professional floor plan visualization."""
    
    def __init__(self, room_dims: Tuple[float, float]):
        self.room_dims = room_dims
        self.architectural_elements = []
        self.furniture_layout = []
        
    def generate_layout(self, architectural_elements: List[Dict], 
                       furniture_layout: List[Dict], 
                       save_path: Optional[str] = None,
                       save_buffer: Optional[io.BytesIO] = None) -> io.BytesIO:
        """
        Generate enhanced 2D layout with clear architectural and furniture distinction.
        """
        self.architectural_elements = architectural_elements
        self.furniture_layout = furniture_layout
        
        # Create figure with professional styling
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(-50, self.room_dims[0] + 50)
        ax.set_ylim(-50, self.room_dims[1] + 50)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # Top-left origin like architectural drawings
        
        # Draw room structure
        self._draw_room_structure(ax)
        
        # Draw architectural elements first (fixed elements)
        self._draw_architectural_elements(ax)
        
        # Draw furniture layout (movable elements)
        self._draw_furniture_layout(ax)
        
        # Add professional annotations
        self._add_annotations(ax)
        
        # Add legend and scale
        self._add_legend_and_scale(ax)
        
        # Clean up axes
        ax.set_title("AI-Optimized Room Layout", fontsize=16, fontweight='bold', pad=20)
        ax.grid(False)
        ax.axis('off')
        
        # Save or return buffer
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white")
        
        if save_buffer is not None:
            plt.savefig(save_buffer, format="png", bbox_inches="tight", dpi=300, facecolor="white")
            save_buffer.seek(0)
            return save_buffer
        
        # If no buffer provided, create one
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight", dpi=300, facecolor="white")
        buffer.seek(0)
        plt.close(fig)
        return buffer
    
    def _draw_room_structure(self, ax):
        """Draw the basic room structure with walls and floor."""
        # Outer walls (thick black lines)
        wall_thickness = 8
        outer_wall = Rectangle((0, 0), self.room_dims[0], self.room_dims[1],
                             linewidth=wall_thickness, edgecolor="black",
                             facecolor="none", alpha=1.0)
        ax.add_patch(outer_wall)
        
        # Floor area (light background)
        floor = Rectangle((2, 2), self.room_dims[0]-4, self.room_dims[1]-4,
                         linewidth=1, edgecolor="#cccccc",
                         facecolor="#f8f8f8", alpha=0.8)
        ax.add_patch(floor)
        
        # Add dimension lines
        self._add_dimension_lines(ax)
    
    def _draw_architectural_elements(self, ax):
        """Draw architectural elements (windows, doors, walls) with professional styling."""
        for element in self.architectural_elements:
            element_type = element.get("type", "").lower()
            x, y, w, h = element["x"], element["y"], element["w"], element["h"]
            
            if element_type == "window":
                self._draw_window(ax, x, y, w, h)
            elif element_type == "door":
                self._draw_door(ax, x, y, w, h)
            elif element_type == "wall":
                self._draw_wall(ax, x, y, w, h)
            elif element_type == "fireplace":
                self._draw_fireplace(ax, x, y, w, h)
            else:
                self._draw_generic_architectural(ax, x, y, w, h, element_type)
    
    def _draw_window(self, ax, x, y, w, h):
        """Draw a window with professional architectural styling."""
        # Window frame (thick border)
        frame = Rectangle((x, y), w, h,
                         linewidth=3, edgecolor="#2c5aa0",
                         facecolor="#b8d4f0", alpha=0.9)
        ax.add_patch(frame)
        
        # Window panes (grid lines)
        if w > 20 and h > 20:  # Only for larger windows
            # Vertical dividers
            if w > 40:
                v_divider = x + w / 2
                ax.plot([v_divider, v_divider], [y, y + h], 
                       color="#1a4480", linewidth=2)
            
            # Horizontal dividers
            if h > 40:
                h_divider = y + h / 2
                ax.plot([x, x + w], [h_divider, h_divider], 
                       color="#1a4480", linewidth=2)
        
        # Window label
        ax.text(x + w/2, y + h/2, "WINDOW", 
               ha="center", va="center", fontsize=8, 
               color="#1a4480", weight="bold",
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    def _draw_door(self, ax, x, y, w, h):
        """Draw a door with swing arc."""
        # Door frame
        door_frame = Rectangle((x, y), w, h,
                              linewidth=3, edgecolor="#8b4513",
                              facecolor="#d2b48c", alpha=0.9)
        ax.add_patch(door_frame)
        
        # Door swing arc
        swing_radius = min(w, h) * 0.8
        swing_center_x = x + w
        swing_center_y = y + h
        
        # Draw swing arc
        arc = Arc((swing_center_x, swing_center_y), 2*swing_radius, 2*swing_radius,
                 angle=0, theta1=0, theta2=90,
                 linewidth=2, color="#654321", linestyle="--")
        ax.add_patch(arc)
        
        # Door leaf
        door_leaf = Rectangle((x + w*0.1, y + h*0.1), w*0.8, h*0.8,
                            linewidth=2, edgecolor="#654321",
                            facecolor="#f4e4bc", alpha=0.9)
        ax.add_patch(door_leaf)
        
        # Door label
        ax.text(x + w/2, y + h/2, "DOOR", 
               ha="center", va="center", fontsize=8, 
               color="#654321", weight="bold",
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    def _draw_wall(self, ax, x, y, w, h):
        """Draw a wall element."""
        wall = Rectangle((x, y), w, h,
                        linewidth=2, edgecolor="#666666",
                        facecolor="#e0e0e0", alpha=0.8,
                        linestyle="--")
        ax.add_patch(wall)
        
        ax.text(x + w/2, y + h/2, "WALL", 
               ha="center", va="center", fontsize=7, 
               color="#666666", weight="bold")
    
    def _draw_fireplace(self, ax, x, y, w, h):
        """Draw a fireplace."""
        # Fireplace body
        fireplace = FancyBboxPatch((x, y), w, h,
                                  boxstyle="round,pad=0.1",
                                  linewidth=2, edgecolor="#8b4513",
                                  facecolor="#d2b48c", alpha=0.9)
        ax.add_patch(fireplace)
        
        # Fireplace opening
        opening_w = w * 0.6
        opening_h = h * 0.7
        opening_x = x + (w - opening_w) / 2
        opening_y = y + h * 0.1
        
        opening = Rectangle((opening_x, opening_y), opening_w, opening_h,
                           linewidth=1, edgecolor="#333333",
                           facecolor="#2c2c2c", alpha=0.8)
        ax.add_patch(opening)
        
        ax.text(x + w/2, y + h/2, "FIREPLACE", 
               ha="center", va="center", fontsize=7, 
               color="#8b4513", weight="bold")
    
    def _draw_generic_architectural(self, ax, x, y, w, h, element_type):
        """Draw generic architectural element."""
        element = Rectangle((x, y), w, h,
                           linewidth=2, edgecolor="#666666",
                           facecolor="#f0f0f0", alpha=0.7,
                           linestyle=":")
        ax.add_patch(element)
        
        ax.text(x + w/2, y + h/2, element_type.upper(), 
               ha="center", va="center", fontsize=7, 
               color="#666666", weight="bold")
    
    def _draw_furniture_layout(self, ax):
        """Draw furniture layout with detailed styling."""
        for furniture in self.furniture_layout:
            furniture_type = furniture.get("type", "").lower()
            x, y, w, h = furniture["x"], furniture["y"], furniture["w"], furniture["h"]
            
            # Choose appropriate drawer based on furniture type
            if "bed" in furniture_type:
                self._draw_bed(ax, x, y, w, h)
            elif "sofa" in furniture_type or "couch" in furniture_type:
                self._draw_sofa(ax, x, y, w, h)
            elif "chair" in furniture_type:
                self._draw_chair(ax, x, y, w, h)
            elif "table" in furniture_type:
                self._draw_table(ax, x, y, w, h)
            elif "desk" in furniture_type:
                self._draw_desk(ax, x, y, w, h)
            elif "wardrobe" in furniture_type or "closet" in furniture_type:
                self._draw_wardrobe(ax, x, y, w, h)
            elif "tv" in furniture_type:
                self._draw_tv(ax, x, y, w, h)
            elif "lamp" in furniture_type:
                self._draw_lamp(ax, x, y, w, h)
            elif "plant" in furniture_type:
                self._draw_plant(ax, x, y, w, h)
            else:
                self._draw_generic_furniture(ax, x, y, w, h, furniture_type)
    
    def _draw_bed(self, ax, x, y, w, h):
        """Draw a detailed bed."""
        # Bed frame
        bed = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.05",
                            linewidth=2, edgecolor="#8b4513",
                            facecolor="#f5deb3", alpha=0.9)
        ax.add_patch(bed)
        
        # Pillows
        pillow_h = h * 0.25
        pillow_w = w * 0.3
        pillow1_x = x + w * 0.1
        pillow2_x = x + w * 0.6
        pillow_y = y + 5
        
        for px in [pillow1_x, pillow2_x]:
            pillow = FancyBboxPatch((px, pillow_y), pillow_w, pillow_h,
                                   boxstyle="round,pad=0.02",
                                   linewidth=1, edgecolor="#8b4513",
                                   facecolor="#ffffff", alpha=0.9)
            ax.add_patch(pillow)
        
        # Bed label
        ax.text(x + w/2, y + h/2, "BED", 
               ha="center", va="center", fontsize=9, 
               color="#8b4513", weight="bold",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    def _draw_sofa(self, ax, x, y, w, h):
        """Draw a detailed sofa."""
        # Sofa body
        sofa = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.05",
                              linewidth=2, edgecolor="#654321",
                              facecolor="#d2691e", alpha=0.9)
        ax.add_patch(sofa)
        
        # Backrest
        backrest_h = h * 0.3
        backrest = FancyBboxPatch((x + 3, y + 3), w - 6, backrest_h,
                                 boxstyle="round,pad=0.02",
                                 linewidth=1, edgecolor="#654321",
                                 facecolor="#cd853f", alpha=0.9)
        ax.add_patch(backrest)
        
        # Cushion divisions
        if w > 60:
            for i in range(1, 3):
                div_x = x + i * w / 3
                ax.plot([div_x, div_x], [y + backrest_h, y + h - 3], 
                       color="#8b4513", linewidth=1, linestyle="--")
        
        ax.text(x + w/2, y + h/2, "SOFA", 
               ha="center", va="center", fontsize=9, 
               color="#654321", weight="bold",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    def _draw_chair(self, ax, x, y, w, h):
        """Draw a detailed chair."""
        # Chair seat
        seat = FancyBboxPatch((x, y + h*0.4), w, h*0.6,
                             boxstyle="round,pad=0.02",
                             linewidth=2, edgecolor="#8b4513",
                             facecolor="#f5deb3", alpha=0.9)
        ax.add_patch(seat)
        
        # Chair back
        back = FancyBboxPatch((x + w*0.2, y), w*0.6, h*0.6,
                              boxstyle="round,pad=0.02",
                              linewidth=2, edgecolor="#8b4513",
                              facecolor="#f5deb3", alpha=0.9)
        ax.add_patch(back)
        
        # Chair legs (small squares at corners)
        leg_size = min(w, h) * 0.15
        for leg_x, leg_y in [(x + 2, y + h - 2), (x + w - leg_size - 2, y + h - 2),
                            (x + 2, y + h*0.4), (x + w - leg_size - 2, y + h*0.4)]:
            leg = Rectangle((leg_x, leg_y), leg_size, leg_size,
                           linewidth=1, edgecolor="#654321",
                           facecolor="#8b4513", alpha=0.8)
            ax.add_patch(leg)
        
        ax.text(x + w/2, y + h/2, "CHAIR", 
               ha="center", va="center", fontsize=8, 
               color="#8b4513", weight="bold",
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
    
    def _draw_table(self, ax, x, y, w, h):
        """Draw a detailed table."""
        # Table top
        table = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.02",
                              linewidth=2, edgecolor="#8b4513",
                              facecolor="#f5deb3", alpha=0.9)
        ax.add_patch(table)
        
        # Table legs
        leg_size = min(w, h) * 0.1
        for leg_x, leg_y in [(x + leg_size/2, y + leg_size/2), 
                            (x + w - leg_size/2, y + leg_size/2),
                            (x + leg_size/2, y + h - leg_size/2),
                            (x + w - leg_size/2, y + h - leg_size/2)]:
            leg = Rectangle((leg_x - leg_size/2, leg_y - leg_size/2), leg_size, leg_size,
                           linewidth=1, edgecolor="#654321",
                           facecolor="#8b4513", alpha=0.8)
            ax.add_patch(leg)
        
        ax.text(x + w/2, y + h/2, "TABLE", 
               ha="center", va="center", fontsize=8, 
               color="#8b4513", weight="bold",
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
    
    def _draw_desk(self, ax, x, y, w, h):
        """Draw a detailed desk."""
        # Desk top
        desk = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.02",
                             linewidth=2, edgecolor="#2c5aa0",
                             facecolor="#e6f3ff", alpha=0.9)
        ax.add_patch(desk)
        
        # Drawer (if wide enough)
        if w > 40:
            drawer_h = h * 0.2
            drawer = Rectangle((x + 5, y + h - drawer_h - 2), w - 10, drawer_h,
                             linewidth=1, edgecolor="#1a4480",
                             facecolor="#cce6ff", alpha=0.9)
            ax.add_patch(drawer)
            
            # Drawer handle
            handle = Circle((x + w/2, y + h - drawer_h/2), min(w, h) * 0.05,
                           linewidth=1, edgecolor="#1a4480",
                           facecolor="#ffffff", alpha=0.9)
            ax.add_patch(handle)
        
        ax.text(x + w/2, y + h/2, "DESK", 
               ha="center", va="center", fontsize=8, 
               color="#2c5aa0", weight="bold",
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
    
    def _draw_wardrobe(self, ax, x, y, w, h):
        """Draw a detailed wardrobe."""
        # Wardrobe body
        wardrobe = FancyBboxPatch((x, y), w, h,
                                 boxstyle="round,pad=0.02",
                                 linewidth=2, edgecolor="#8b4513",
                                 facecolor="#f5deb3", alpha=0.9)
        ax.add_patch(wardrobe)
        
        # Wardrobe doors
        if w > 30:
            door_w = w / 2
            door1 = Rectangle((x + 2, y + 2), door_w - 4, h - 4,
                            linewidth=1, edgecolor="#654321",
                            facecolor="#ffffff", alpha=0.8)
            ax.add_patch(door1)
            
            door2 = Rectangle((x + door_w + 2, y + 2), door_w - 4, h - 4,
                             linewidth=1, edgecolor="#654321",
                             facecolor="#ffffff", alpha=0.8)
            ax.add_patch(door2)
            
            # Door handles
            handle1 = Circle((x + door_w/2, y + h/2), min(w, h) * 0.05,
                            linewidth=1, edgecolor="#654321",
                            facecolor="#8b4513", alpha=0.9)
            ax.add_patch(handle1)
            
            handle2 = Circle((x + door_w + door_w/2, y + h/2), min(w, h) * 0.05,
                            linewidth=1, edgecolor="#654321",
                            facecolor="#8b4513", alpha=0.9)
            ax.add_patch(handle2)
        
        ax.text(x + w/2, y + h/2, "WARDROBE", 
               ha="center", va="center", fontsize=8, 
               color="#8b4513", weight="bold",
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
    
    def _draw_tv(self, ax, x, y, w, h):
        """Draw a detailed TV."""
        # TV screen
        tv = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.02",
                           linewidth=2, edgecolor="#000000",
                           facecolor="#1a1a1a", alpha=0.9)
        ax.add_patch(tv)
        
        # Screen
        screen_w = w * 0.9
        screen_h = h * 0.8
        screen_x = x + (w - screen_w) / 2
        screen_y = y + (h - screen_h) / 2
        
        screen = Rectangle((screen_x, screen_y), screen_w, screen_h,
                          linewidth=1, edgecolor="#000000",
                          facecolor="#000000", alpha=0.9)
        ax.add_patch(screen)
        
        # Stand
        stand_h = h * 0.2
        stand_w = w * 0.6
        stand_x = x + (w - stand_w) / 2
        stand_y = y + h - stand_h
        
        stand = Rectangle((stand_x, stand_y), stand_w, stand_h,
                         linewidth=1, edgecolor="#333333",
                         facecolor="#333333", alpha=0.9)
        ax.add_patch(stand)
        
        ax.text(x + w/2, y + h/2, "TV", 
               ha="center", va="center", fontsize=8, 
               color="#ffffff", weight="bold",
               bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.8))
    
    def _draw_lamp(self, ax, x, y, w, h):
        """Draw a detailed lamp."""
        # Lamp base
        base = Circle((x + w/2, y + h/2), min(w, h) * 0.4,
                     linewidth=2, edgecolor="#8b4513",
                     facecolor="#f5deb3", alpha=0.9)
        ax.add_patch(base)
        
        # Lamp shade
        shade = Circle((x + w/2, y + h/2), min(w, h) * 0.3,
                      linewidth=1, edgecolor="#654321",
                      facecolor="#ffffff", alpha=0.8)
        ax.add_patch(shade)
        
        ax.text(x + w/2, y + h/2, "LAMP", 
               ha="center", va="center", fontsize=7, 
               color="#8b4513", weight="bold")
    
    def _draw_plant(self, ax, x, y, w, h):
        """Draw a detailed plant."""
        # Plant pot
        pot = Circle((x + w/2, y + h/2), min(w, h) * 0.4,
                    linewidth=2, edgecolor="#8b4513",
                    facecolor="#f5deb3", alpha=0.9)
        ax.add_patch(pot)
        
        # Plant leaves (simple circles)
        leaf_size = min(w, h) * 0.15
        for i in range(3):
            angle = i * 120
            leaf_x = x + w/2 + leaf_size * np.cos(np.radians(angle))
            leaf_y = y + h/2 + leaf_size * np.sin(np.radians(angle))
            leaf = Circle((leaf_x, leaf_y), leaf_size * 0.5,
                        linewidth=1, edgecolor="#228b22",
                        facecolor="#90ee90", alpha=0.8)
            ax.add_patch(leaf)
        
        ax.text(x + w/2, y + h/2, "PLANT", 
               ha="center", va="center", fontsize=7, 
               color="#228b22", weight="bold")
    
    def _draw_generic_furniture(self, ax, x, y, w, h, furniture_type):
        """Draw generic furniture."""
        furniture = FancyBboxPatch((x, y), w, h,
                                  boxstyle="round,pad=0.02",
                                  linewidth=2, edgecolor="#666666",
                                  facecolor="#f0f0f0", alpha=0.9)
        ax.add_patch(furniture)
        
        ax.text(x + w/2, y + h/2, furniture_type.upper(), 
               ha="center", va="center", fontsize=8, 
               color="#666666", weight="bold",
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
    
    def _add_dimension_lines(self, ax):
        """Add dimension lines to the layout."""
        # Room dimensions
        room_w, room_h = self.room_dims
        
        # Horizontal dimension (top)
        ax.annotate("", xy=(0, -20), xytext=(room_w, -20),
                   arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.5))
        ax.text(room_w/2, -25, f"{int(room_w)} cm", 
               ha="center", va="center", fontsize=10, color="#333333")
        
        # Vertical dimension (left)
        ax.annotate("", xy=(-20, 0), xytext=(-20, room_h),
                   arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.5))
        ax.text(-30, room_h/2, f"{int(room_h)} cm", 
               ha="center", va="center", fontsize=10, color="#333333", rotation=90)
    
    def _add_annotations(self, ax):
        """Add professional annotations to the layout."""
        # Room area calculation
        room_area = (self.room_dims[0] * self.room_dims[1]) / 10000  # Convert to m²
        ax.text(self.room_dims[0]/2, self.room_dims[1]/2, 
               f"Room Area: {room_area:.1f} m²", 
               ha="center", va="center", fontsize=12, 
               color="#2c5aa0", weight="bold",
               bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))
        
        # Furniture count
        furniture_count = len(self.furniture_layout)
        ax.text(10, self.room_dims[1] - 20, 
               f"Furniture Items: {furniture_count}", 
               ha="left", va="top", fontsize=10, 
               color="#666666",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def _add_legend_and_scale(self, ax):
        """Add legend and scale to the layout."""
        # Legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor="#b8d4f0", edgecolor="#2c5aa0", label="Windows"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#d2b48c", edgecolor="#8b4513", label="Doors"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#f5deb3", edgecolor="#8b4513", label="Furniture"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#f8f8f8", edgecolor="#cccccc", label="Floor")
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(0.98, 0.98), fontsize=9)
        
        # Scale bar
        scale_length = 100  # 100cm scale
        scale_x = self.room_dims[0] - 150
        scale_y = 20
        
        ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y], 
               color="#333333", linewidth=3)
        ax.text(scale_x + scale_length/2, scale_y - 10, 
               f"{int(scale_length)} cm", 
               ha="center", va="top", fontsize=9, color="#333333")
