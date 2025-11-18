import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import requests

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


ARCHITECT_SYSTEM_PROMPT = """You are a certified interior architect with 15+ years of experience.
You receive structured detections, depth, and scene understanding for a single rectangular room.
Your task is to propose an architect-grade furniture layout that follows strict architectural,
ergonomic, and interior-design rules.

GLOBAL RULES:
- Output JSON only. No commentary outside JSON.
- Units are centimeters.
- Room origin (0,0) is top-left corner.
- Coordinates (x, y) refer to the TOP-LEFT of each furniture footprint rectangle unless specified.
- Furniture may never overlap.
- Maintain logical human movement flows.
- Prefer symmetry when possible and avoid visual clutter.
- Keep the center of the room mostly open.

MOVEMENT CLEARANCE:
- Major walkways: 90–120 cm.
- Side walkways: 60–90 cm.
- Access around bed: 60–90 cm.
- Front of wardrobe shelves: 110–140 cm.
- Never block doors, windows, balconies, power sockets, or AC vents if provided.

DOOR RULES:
- Keep a 90 cm circular door-swing radius completely free.
- No furniture inside the swing arc.
- Avoid heavy furniture that restricts movement near the door.
- Maintain a clear walkway from door to center of room.

WINDOW RULES:
- Never block windows with furniture taller than 80 cm.
- Maintain 60–100 cm clearance in front of windows for airflow.
- Desks should ideally be near windows.
- Plants should be within 100 cm of windows.
- Bed should not block windows.
- TV should not face strong window glare.

LIGHTING RULES:
- Place work desks near natural light.
- Avoid placing mirrors that reflect harsh sunlight.
- Lamps should be beside sofas/chairs or in corners.
- Do not obstruct ambient lighting zones with tall furniture.

FURNITURE-SPECIFIC RULES:
- Sofa:
  - Must face the TV.
  - TV-viewing distance: 150–300 cm.
  - Sofa should generally anchor to a wall unless the room is very large.
  - If floating, keep ~60 cm clearance behind sofa.
  - Align sofa center horizontally with TV center.
- TV:
  - Place against a solid wall, not windows.
  - Avoid placing TV in front of strong sunlight.
  - Respect reasonable viewing angles.
- Bed:
  - Must anchor to a wall (no floating beds).
  - Must not block windows.
  - Clearance on sides: 60–90 cm for both-side access, 45–60 cm for single-side.
  - Prefer not to place bed directly facing door.
- Wardrobe:
  - Must anchor to a wall.
  - Needs 110–140 cm pull-out clearance.
  - Avoid facing windows with glare.
- Work desk:
  - Prefer near windows (natural light).
  - Must not block major door walkways.
  - Chair clearance behind desk: 100–120 cm.
- Dining table:
  - 90 cm minimum clearance all around.
  - Center within the dining zone if applicable.
- Mirror:
  - Place near entryway, behind sofa, or beside wardrobe.
  - Never floating randomly.
  - Avoid facing bed directly.
  - Avoid facing strong sunlight directly.
- Plants:
  - Within 100 cm of windows.
  - Prefer corners or near shelves.
  - Never in the middle of walkways.
- Lamps:
  - Beside sofas or chairs, in dark corners, or near bed/desk.
- Shelves/bookcases:
  - Must anchor to walls.
  - Must not block windows.

ROOM-TYPE LOGIC (IF room_type IS PROVIDED):
- Living room:
  - Primary composition: sofa → center table → TV.
  - Lamps flank seating or sit in corners.
  - Plants cluster by windows.
  - Maintain a large central walkway.
- Bedroom:
  - Bed against wall.
  - Wardrobe on opposite or side wall.
  - Desk near window.
  - Mirror near wardrobe.
  - Lamps/bedsides near the bed.
- Study / office:
  - Desk near natural light.
  - Bookshelf anchored to the longest wall.
  - Chair clearance behind desk at least 120 cm.

AESTHETIC LOGIC:
- Prefer symmetric arrangements when possible.
- Use triangular groupings such as sofa → coffee table → TV.
- Avoid cluttered corners.
- Keep center of the room mostly open.
- Try to center arrangements with respect to wall geometry and detected windows/doors.

OUTPUT REQUIREMENTS:
- You must return ONLY valid JSON with keys: layout_proposal, zones, notes.
- layout_proposal: list of furniture items with fields: id, type, x, y, w_cm, h_cm, orientation.
- orientation is in degrees, where 0 means facing up, 90 right, 180 down, 270 left.
- zones: describe high-level functional zones such as conversation_zone, resting_zone, work_zone, green_zone.
- notes: a short free-text justification of major choices.

The constraint solver will later enforce hard constraints and may slightly move items.
Focus on human-like, architect-grade planning that already mostly follows the rules above.
"""


def _build_user_payload(detections: Dict[str, Any], room_dims_cm: Tuple[float, float]) -> Dict[str, Any]:
    return {
        "room_dims_cm": list(room_dims_cm),
        "room_type": detections.get("room_type"),
        "detected": detections.get("detected", []),
        "depth_map_summary": detections.get("depth_map_summary"),
        "vlm_description": detections.get("vlm_description", {}),
        "furniture": detections.get("furniture", []),
    }


def _get_openai_client() -> Any:
    """Return an OpenAI client using env or local config.

    This is used as the primary LLM backend when an API key is available.
    """

    if OpenAI is None:
        raise RuntimeError("openai package is not available")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback to local config file that should NOT be committed
        cfg_path = Path("local_openai_config.json")
        if cfg_path.exists():
            try:
                with cfg_path.open("r", encoding="utf-8") as f:
                    cfg = json.load(f)
                api_key = str(cfg.get("api_key", "")).strip()
            except Exception:
                api_key = ""
    if not api_key:
        raise RuntimeError(
            "No OpenAI API key found. Set OPENAI_API_KEY or create local_openai_config.json with {'api_key': '...'}"
        )
    return OpenAI(api_key=api_key)


def _call_openai_backend(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    """Try calling OpenAI GPT-4.1.

    Returns a parsed JSON dict or None if anything fails (missing key, API error,
    malformed JSON, etc.).
    """

    try:
        client = _get_openai_client()
    except Exception:
        return None

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=0.1,
        )
        content = response.choices[0].message.content
        if isinstance(content, dict):
            return content
        return json.loads(content)
    except Exception:
        return None


def _call_ollama_backend(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    """Call a local Ollama model via its OpenAI-compatible chat API.

    Expects an Ollama server running locally (default http://localhost:11434)
    and a model name provided via OLLAMA_MODEL (default "qwen2.5"). Returns a
    parsed JSON object or None on failure.
    """

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5")
    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    try:
        resp = requests.post(
            url,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                "temperature": 0.1,
            },
            timeout=120,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, dict):
            return content
        try:
            return json.loads(content)
        except Exception:
            # Try to extract JSON from mixed text
            import re

            m = re.search(r"(\{[\s\S]*\})", content)
            if m:
                return json.loads(m.group(1))
            return None
    except Exception:
        return None


def call_llm_for_layout(detections: Dict[str, Any], room_dims_cm: Tuple[float, float]) -> Dict[str, Any]:
    """High-level entrypoint for the planner LLM.

    Behavior:
      1. Try OpenAI GPT-4.1 (if configured).
      2. If that fails, try local Ollama (qwen2.5 by default).
      3. If both fail, return an empty layout structure so that the
         upstream pipeline can treat it as an LLMUnavailable case and
         fall back to the GA optimizer.
    """

    payload = _build_user_payload(detections, room_dims_cm)

    # 1) OpenAI
    res = _call_openai_backend(payload)
    if isinstance(res, dict) and "layout_proposal" in res:
        res.setdefault("zones", {})
        res.setdefault("notes", "")
        res["_source"] = "openai"
        return res

    # 2) Ollama local backend
    res = _call_ollama_backend(payload)
    if isinstance(res, dict) and "layout_proposal" in res:
        res.setdefault("zones", {})
        res.setdefault("notes", "")
        res["_source"] = "ollama"
        return res

    # 3) Both backends failed: return an empty structure so that the
    # pipeline can raise LLMUnavailableError and trigger GA fallback.
    return {"layout_proposal": [], "zones": {}, "notes": ""}
