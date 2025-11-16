import json
import os
from typing import Dict


class RuleWeightLearner:
    """Simple persistent rule weight memory.
    Stores per-rule cumulative deltas and counts and exposes a multiplier [0.5, 2.0].
    """

    def __init__(self, path: str = "data/rule_memory.json") -> None:
        self.path = path
        self.memory: Dict[str, Dict[str, float]] = self.load_memory()

    def load_memory(self) -> Dict[str, Dict[str, float]]:
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
        except Exception:
            pass
        return {}

    def update_memory(self, rule_id: str, score_delta: float) -> None:
        try:
            if rule_id not in self.memory:
                self.memory[rule_id] = {"count": 0, "total_delta": 0.0}
            self.memory[rule_id]["count"] = int(self.memory[rule_id].get("count", 0)) + 1
            self.memory[rule_id]["total_delta"] = float(self.memory[rule_id].get("total_delta", 0.0)) + float(score_delta)
            self.save_memory()
        except Exception:
            # Avoid breaking optimization if write fails
            pass

    def get_weight_adjustment(self, rule_id: str) -> float:
        try:
            item = self.memory.get(rule_id)
            if not item:
                return 1.0
            count = max(1, int(item.get("count", 0)))
            avg_delta = float(item.get("total_delta", 0.0)) / float(count)
            # Map avg_delta to [0.5, 2.0] softly
            mult = 1.0 + (avg_delta / 100.0)
            return max(0.5, min(2.0, mult))
        except Exception:
            return 1.0

    def save_memory(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
