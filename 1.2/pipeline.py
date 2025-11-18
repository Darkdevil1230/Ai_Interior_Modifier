import json
from typing import Dict, Any, Tuple, Optional, List

import logging

import openai  # for error types only
import pandas as pd

from llm_architect import call_llm_for_layout
from constraint_solver import enforce_constraints


class LLMUnavailableError(Exception):
    """Raised when LLM API is unavailable or fails."""


class LLMArchitect:
    """Thin wrapper around call_llm_for_layout to match the pipeline interface.

    This class currently ignores the api_key argument because llm_architect.py
    reads configuration from environment/local file. The parameter is accepted
    for future extension without breaking callers.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key

    def propose_layout(
        self,
        room_data: Dict[str, Any],
        room_dims: Tuple[float, float],
        detected_openings: List[Dict[str, Any]],
        furniture_catalog: pd.DataFrame,
        room_type: str = "living_room",
    ) -> Dict[str, Any]:
        # Build detections payload expected by call_llm_for_layout
        furniture_list: List[Dict[str, Any]] = []
        if isinstance(furniture_catalog, pd.DataFrame):
            furniture_list = furniture_catalog.to_dict(orient="records")

        detections_payload: Dict[str, Any] = {
            "room_type": room_data.get("room_type", room_type),
            "detected": room_data.get("detections") or room_data.get("detected") or detected_openings,
            "depth_map_summary": room_data.get("depth_map_summary")
            or room_data.get("metrics", {}).get("depth_summary"),
            "vlm_description": room_data.get("vlm_description", {}),
            "furniture": furniture_list,
        }
        return call_llm_for_layout(detections_payload, room_dims)


def run_pipeline(
    room_data: Dict,
    room_dims: Tuple[float, float],
    detected_openings: List[Dict[str, Any]],
    furniture_catalog: pd.DataFrame,
    room_type: str = "living_room",
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """LLM Architect + CP-SAT Constraint Solver Pipeline.

    Raises
    ------
    LLMUnavailableError
        When LLM API fails (quota, network, etc.).
    """

    logger = logging.getLogger(__name__)

    try:
        # Step 1: LLM proposes layout
        logger.info("🤖 Calling LLM Architect...")
        llm_architect = LLMArchitect(api_key=api_key)
        proposal = llm_architect.propose_layout(
            room_data=room_data,
            room_dims=room_dims,
            detected_openings=detected_openings,
            furniture_catalog=furniture_catalog,
            room_type=room_type,
        )

        # Consider proposals with no items as failure so that the caller can
        # fall back to the GA optimizer when both OpenAI and Ollama fail or
        # return unusable results.
        if (
            not proposal
            or "layout_proposal" not in proposal
            or not proposal.get("layout_proposal")
        ):
            raise LLMUnavailableError("LLM returned empty or invalid proposal")

        furniture_items = proposal.get("layout_proposal", [])
        logger.info("✅ LLM proposed %d items", len(furniture_items))

        # Step 2: Enforce constraints with CP-SAT
        logger.info("⚙️ Running constraint solver...")
        room_w, room_h = room_dims
        final_layout = enforce_constraints(room_w, room_h, detected_openings, proposal)

        logger.info("✅ Constraint solver placed %d items", len(final_layout))
        return final_layout

    except openai.APIError as e:  # type: ignore[attr-defined]
        logger.error("OpenAI API Error: %s", e)
        raise LLMUnavailableError(f"OpenAI API failed: {e}") from e

    except openai.RateLimitError as e:  # type: ignore[attr-defined]
        logger.error("OpenAI Rate Limit: %s", e)
        raise LLMUnavailableError(f"OpenAI rate limit exceeded: {e}") from e

    except Exception as e:  # noqa: BLE001
        # Only catch API-related errors as LLMUnavailableError
        msg = str(e).lower()
        if "api" in msg or "quota" in msg or "rate limit" in msg:
            logger.error("LLM service unavailable: %s", e)
            raise LLMUnavailableError(f"LLM service error: {e}") from e
        # Re-raise programming errors
        logger.exception("Unexpected error in pipeline")
        raise
