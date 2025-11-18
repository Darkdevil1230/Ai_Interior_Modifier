from typing import Dict, List, Any

try:
    from ortools.sat.python import cp_model
except Exception:  # pragma: no cover
    cp_model = None  # type: ignore


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(round(float(x)))
    except Exception:
        return default


def enforce_constraints(room_w: float, room_h: float, detections: List[Dict[str, Any]], proposal: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Apply a CP-SAT model to clean up the LLM layout proposal.

    The solver enforces:
    - keep furniture fully inside room bounds
    - rectangle non-overlap
    - door swing clearance (~90cm radius) using L1 distance approximation
    - simple window clearance for tall objects (no overlap with window bbox)
    - bed/wardrobe must touch at least one wall
    - minimize deviation from LLM proposal positions

    If OR-Tools is unavailable or the model is infeasible, the original
    proposal layout_proposal is returned.
    """
    items = list(proposal.get("layout_proposal", []))
    if not items or cp_model is None:
        return items

    room_w_i = _safe_int(room_w)
    room_h_i = _safe_int(room_h)

    model = cp_model.CpModel()

    n = len(items)
    # We model centers in integer cm for easier constraints
    Cx = [model.NewIntVar(0, room_w_i, f"cx_{i}") for i in range(n)]
    Cy = [model.NewIntVar(0, room_h_i, f"cy_{i}") for i in range(n)]

    W = [_safe_int(it.get("w_cm", it.get("w") or 0)) for it in items]
    H = [_safe_int(it.get("h_cm", it.get("h") or 0)) for it in items]

    # Inside bounds
    for i in range(n):
        half_w = W[i] // 2
        half_h = H[i] // 2
        model.Add(Cx[i] >= half_w)
        model.Add(Cx[i] <= room_w_i - half_w)
        model.Add(Cy[i] >= half_h)
        model.Add(Cy[i] <= room_h_i - half_h)

    # Door swing clearance (approximate) and collect window bboxes in cm
    door_centers: List[tuple[int, int]] = []
    window_bboxes: List[List[int]] = []
    for det in detections or []:
        t = str(det.get("type") or det.get("label") or "").lower()
        bb = det.get("room_bbox") or det.get("bbox")
        if not isinstance(bb, (list, tuple)) or len(bb) != 4:
            continue
        x1, y1, x2, y2 = map(_safe_int, bb)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        if t == "door":
            door_centers.append((cx, cy))
        elif t == "window":
            window_bboxes.append([x1, y1, x2, y2])

    clearance = 90
    for (dx, dy) in door_centers:
        for i in range(n):
            # |Cx[i] - dx| + |Cy[i] - dy| >= clearance (L1 approx of circle)
            abs_x = model.NewIntVar(0, room_w_i, f"abs_dx_{i}_{dx}_{dy}")
            abs_y = model.NewIntVar(0, room_h_i, f"abs_dy_{i}_{dx}_{dy}")
            model.AddAbsEquality(abs_x, Cx[i] - dx)
            model.AddAbsEquality(abs_y, Cy[i] - dy)
            model.Add(abs_x + abs_y >= clearance)

    # Walkway bands: keep a corridor from each door towards room center mostly free.
    # We approximate this as a fixed-width axis-aligned rectangle between the door center
    # and the geometric room center, and require each furniture item to not overlap it.
    if door_centers:
        cx_room = room_w_i // 2
        cy_room = room_h_i // 2
        walkway_half_width = 40  # ~80cm total corridor width

        corridor_bboxes: List[List[int]] = []
        for (dx, dy) in door_centers:
            # Decide orientation by which side the door is closer to
            if dx < room_w_i * 0.25 or dx > room_w_i * 0.75:
                # Door on left/right wall -> horizontal corridor
                x1 = min(dx, cx_room)
                x2 = max(dx, cx_room)
                y1 = max(0, cy_room - walkway_half_width)
                y2 = min(room_h_i, cy_room + walkway_half_width)
            else:
                # Door near top/bottom -> vertical corridor
                y1 = min(dy, cy_room)
                y2 = max(dy, cy_room)
                x1 = max(0, cx_room - walkway_half_width)
                x2 = min(room_w_i, cx_room + walkway_half_width)
            if x2 > x1 and y2 > y1:
                corridor_bboxes.append([x1, y1, x2, y2])

        # For each corridor bbox, enforce non-overlap with every furniture item
        for (cx1, cy1, cx2, cy2) in corridor_bboxes:
            for i in range(n):
                b1 = model.NewBoolVar(f"obj_left_corr_{i}_{cx1}_{cy1}")
                b2 = model.NewBoolVar(f"obj_right_corr_{i}_{cx1}_{cy1}")
                b3 = model.NewBoolVar(f"obj_above_corr_{i}_{cx1}_{cy1}")
                b4 = model.NewBoolVar(f"obj_below_corr_{i}_{cx1}_{cy1}")
                model.Add(Cx[i] * 2 + W[i] <= 2 * cx1).OnlyEnforceIf(b1)
                model.Add(2 * cx2 <= Cx[i] * 2 - W[i]).OnlyEnforceIf(b2)
                model.Add(Cy[i] * 2 + H[i] <= 2 * cy1).OnlyEnforceIf(b3)
                model.Add(2 * cy2 <= Cy[i] * 2 - H[i]).OnlyEnforceIf(b4)
                model.AddBoolOr([b1, b2, b3, b4])

    # Non-overlap constraints using disjunction (axis-aligned rectangles)
    for i in range(n):
        for j in range(i + 1, n):
            b1 = model.NewBoolVar(f"left_{i}_{j}")
            b2 = model.NewBoolVar(f"right_{i}_{j}")
            b3 = model.NewBoolVar(f"above_{i}_{j}")
            b4 = model.NewBoolVar(f"below_{i}_{j}")
            # i is left of j
            model.Add(Cx[i] * 2 + W[i] <= Cx[j] * 2 - W[j]).OnlyEnforceIf(b1)
            # i is right of j
            model.Add(Cx[j] * 2 + W[j] <= Cx[i] * 2 - W[i]).OnlyEnforceIf(b2)
            # i is above j
            model.Add(Cy[i] * 2 + H[i] <= Cy[j] * 2 - H[j]).OnlyEnforceIf(b3)
            # i is below j
            model.Add(Cy[j] * 2 + H[j] <= Cy[i] * 2 - H[i]).OnlyEnforceIf(b4)
            model.AddBoolOr([b1, b2, b3, b4])

    # Window clearance for tall objects: do not overlap window bbox
    for i, it in enumerate(items):
        if H[i] <= 80 and W[i] <= 80:
            continue
        for (wx1, wy1, wx2, wy2) in window_bboxes:
            # Same disjunction-style non-overlap between item and window rect
            b1 = model.NewBoolVar(f"obj_left_win_{i}")
            b2 = model.NewBoolVar(f"obj_right_win_{i}")
            b3 = model.NewBoolVar(f"obj_above_win_{i}")
            b4 = model.NewBoolVar(f"obj_below_win_{i}")
            model.Add(Cx[i] * 2 + W[i] <= 2 * wx1).OnlyEnforceIf(b1)
            model.Add(2 * wx2 <= Cx[i] * 2 - W[i]).OnlyEnforceIf(b2)
            model.Add(Cy[i] * 2 + H[i] <= 2 * wy1).OnlyEnforceIf(b3)
            model.Add(2 * wy2 <= Cy[i] * 2 - H[i]).OnlyEnforceIf(b4)
            model.AddBoolOr([b1, b2, b3, b4])

    # Bed and wardrobe must touch at least one wall (in a relaxed sense)
    wall_margin = 5
    for i, it in enumerate(items):
        t = str(it.get("type") or "").lower()
        if t not in {"bed", "wardrobe", "closet"}:
            continue
        on_left = model.NewBoolVar(f"on_left_{i}")
        on_right = model.NewBoolVar(f"on_right_{i}")
        on_top = model.NewBoolVar(f"on_top_{i}")
        on_bottom = model.NewBoolVar(f"on_bottom_{i}")
        model.Add(Cx[i] <= W[i] // 2 + wall_margin).OnlyEnforceIf(on_left)
        model.Add(Cx[i] >= room_w_i - W[i] // 2 - wall_margin).OnlyEnforceIf(on_right)
        model.Add(Cy[i] <= H[i] // 2 + wall_margin).OnlyEnforceIf(on_top)
        model.Add(Cy[i] >= room_h_i - H[i] // 2 - wall_margin).OnlyEnforceIf(on_bottom)
        model.AddBoolOr([on_left, on_right, on_top, on_bottom])

    # Sofa–TV relationship: viewing distance and rough facing alignment
    sofa_indices: List[int] = []
    tv_indices: List[int] = []
    desk_indices: List[int] = []
    plant_indices: List[int] = []
    mirror_indices: List[int] = []
    for i, it in enumerate(items):
        t = str(it.get("type") or "").lower()
        if "sofa" in t or t in {"couch"}:
            sofa_indices.append(i)
        if t in {"tv", "television", "monitor"}:
            tv_indices.append(i)
        if "desk" in t or t in {"work desk", "office desk"}:
            desk_indices.append(i)
        if t in {"plant", "potted plant"} or "plant" in t:
            plant_indices.append(i)
        if "mirror" in t:
            mirror_indices.append(i)

    # Enforce sofa–TV pairing if at least one of each exists
    for si in sofa_indices:
        for ti in tv_indices:
            abs_dx = model.NewIntVar(0, room_w_i, f"st_dx_{si}_{ti}")
            abs_dy = model.NewIntVar(0, room_h_i, f"st_dy_{si}_{ti}")
            model.AddAbsEquality(abs_dx, Cx[si] - Cx[ti])
            model.AddAbsEquality(abs_dy, Cy[si] - Cy[ti])
            # Viewing distance 150–300 cm in L1 metric (approximate)
            dist_l1 = model.NewIntVar(0, room_w_i + room_h_i, f"st_l1_{si}_{ti}")
            model.Add(dist_l1 == abs_dx + abs_dy)
            model.Add(dist_l1 >= 150)
            model.Add(dist_l1 <= 450)

            # Rough axis alignment: either vertically aligned (sofa below/above TV)
            # or horizontally aligned (sofa left/right of TV)
            b_vert = model.NewBoolVar(f"st_vert_{si}_{ti}")
            b_horiz = model.NewBoolVar(f"st_horiz_{si}_{ti}")

            # Vertical arrangement: centers roughly aligned in X, separated in Y
            model.Add(abs_dx <= 60).OnlyEnforceIf(b_vert)
            model.Add(abs_dy >= 150).OnlyEnforceIf(b_vert)
            model.Add(abs_dy <= 300).OnlyEnforceIf(b_vert)

            # Horizontal arrangement: centers roughly aligned in Y, separated in X
            model.Add(abs_dy <= 60).OnlyEnforceIf(b_horiz)
            model.Add(abs_dx >= 150).OnlyEnforceIf(b_horiz)
            model.Add(abs_dx <= 300).OnlyEnforceIf(b_horiz)

            model.AddBoolOr([b_vert, b_horiz])

    # Desk near window: ensure each desk is within ~150cm L1 of at least one window center if windows exist
    for di in desk_indices:
        if not window_bboxes:
            break
        near_any = []
        for (wx1, wy1, wx2, wy2) in window_bboxes:
            wxc = (wx1 + wx2) // 2
            wyc = (wy1 + wy2) // 2
            abs_dx = model.NewIntVar(0, room_w_i, f"desk_dx_{di}_{wxc}_{wyc}")
            abs_dy = model.NewIntVar(0, room_h_i, f"desk_dy_{di}_{wxc}_{wyc}")
            model.AddAbsEquality(abs_dx, Cx[di] - wxc)
            model.AddAbsEquality(abs_dy, Cy[di] - wyc)
            b_near = model.NewBoolVar(f"desk_near_{di}_{wxc}_{wyc}")
            model.Add(abs_dx + abs_dy <= 150).OnlyEnforceIf(b_near)
            near_any.append(b_near)
        if near_any:
            model.AddBoolOr(near_any)

    # Plants: keep within 100cm L1 of some window center when windows exist
    for pi in plant_indices:
        if not window_bboxes:
            break
        near_any = []
        for (wx1, wy1, wx2, wy2) in window_bboxes:
            wxc = (wx1 + wx2) // 2
            wyc = (wy1 + wy2) // 2
            abs_dx = model.NewIntVar(0, room_w_i, f"plant_dx_{pi}_{wxc}_{wyc}")
            abs_dy = model.NewIntVar(0, room_h_i, f"plant_dy_{pi}_{wxc}_{wyc}")
            model.AddAbsEquality(abs_dx, Cx[pi] - wxc)
            model.AddAbsEquality(abs_dy, Cy[pi] - wyc)
            b_near = model.NewBoolVar(f"plant_near_{pi}_{wxc}_{wyc}")
            model.Add(abs_dx + abs_dy <= 100).OnlyEnforceIf(b_near)
            near_any.append(b_near)
        if near_any:
            model.AddBoolOr(near_any)

    # Mirrors: place near entry (doors) or wardrobes, not floating randomly
    for mi in mirror_indices:
        near_any = []
        # near doors
        for (dx, dy) in door_centers:
            abs_dx = model.NewIntVar(0, room_w_i, f"mirror_dx_door_{mi}_{dx}_{dy}")
            abs_dy = model.NewIntVar(0, room_h_i, f"mirror_dy_door_{mi}_{dx}_{dy}")
            model.AddAbsEquality(abs_dx, Cx[mi] - dx)
            model.AddAbsEquality(abs_dy, Cy[mi] - dy)
            b_near = model.NewBoolVar(f"mirror_near_door_{mi}_{dx}_{dy}")
            model.Add(abs_dx + abs_dy <= 200).OnlyEnforceIf(b_near)
            near_any.append(b_near)
        # near wardrobes
        for wi, it in enumerate(items):
            t = str(it.get("type") or "").lower()
            if t not in {"wardrobe", "closet"}:
                continue
            abs_dx = model.NewIntVar(0, room_w_i, f"mirror_dx_ward_{mi}_{wi}")
            abs_dy = model.NewIntVar(0, room_h_i, f"mirror_dy_ward_{mi}_{wi}")
            model.AddAbsEquality(abs_dx, Cx[mi] - Cx[wi])
            model.AddAbsEquality(abs_dy, Cy[mi] - Cy[wi])
            b_near = model.NewBoolVar(f"mirror_near_ward_{mi}_{wi}")
            model.Add(abs_dx + abs_dy <= 150).OnlyEnforceIf(b_near)
            near_any.append(b_near)
        if near_any:
            model.AddBoolOr(near_any)

    # Objective: stay close to LLM suggestion
    objective_terms = []
    for i, it in enumerate(items):
        sx = _safe_int(it.get("x", 0))
        sy = _safe_int(it.get("y", 0))
        dx = model.NewIntVar(0, room_w_i, f"dx_{i}")
        dy = model.NewIntVar(0, room_h_i, f"dy_{i}")
        model.AddAbsEquality(dx, Cx[i] - (sx + W[i] // 2))
        model.AddAbsEquality(dy, Cy[i] - (sy + H[i] // 2))
        objective_terms.extend([dx, dy])

    # Soft room coverage constraint: discourage over-filling the room beyond ~60%
    # of its area with furniture footprints. We compute the total furniture area
    # from the current item dimensions and penalize any excess over the target.
    if items:
        room_area_cm2 = room_w_i * room_h_i
        total_furniture_area = sum(max(1, W[i] * H[i]) for i in range(n))
        max_coverage = 0.6
        max_allowed_area = int(room_area_cm2 * max_coverage)
        if total_furniture_area > max_allowed_area:
            over_coverage = model.NewIntVar(0, room_area_cm2, "over_coverage")
            model.Add(over_coverage >= total_furniture_area - max_allowed_area)
            # Add with a relatively high weight so that the solver prefers
            # layouts that reduce overcrowding when possible.
            objective_terms.append(10 * over_coverage)

    if objective_terms:
        model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return items

    final: List[Dict[str, Any]] = []
    for i, it in enumerate(items):
        cx = solver.Value(Cx[i])
        cy = solver.Value(Cy[i])
        w = W[i]
        h = H[i]
        x = cx - w // 2
        y = cy - h // 2
        out = dict(it)
        out.update({
            "x": int(x),
            "y": int(y),
            "w_cm": int(w),
            "h_cm": int(h),
        })
        final.append(out)
    return final
