import math
from typing import Dict, Any

DEFAULT_CALIBRATION = {"mu": 0.85, "sigma": 0.25}

TARGET_BANDS = {
    "easy":   (0.20, 0.40),
    "medium": (0.45, 0.65),
    "hard":   (0.70, 0.90),
}

def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))

def compute_hardness(difficulty: str, cards: int, solver_stats: Dict[str, Any],
                     mu: float = DEFAULT_CALIBRATION["mu"],
                     sigma: float = DEFAULT_CALIBRATION["sigma"]) -> Dict[str, Any]:
    backtracks = float(solver_stats.get("backtracks", 0))
    nodes = float(solver_stats.get("nodesVisited", 0))
    time_ms = float(solver_stats.get("timeMs", 0))
    depth = float(solver_stats.get("maxDepth", 0))

    c = max(1, int(cards))
    bt_per_card = backtracks / c
    nv_per_card = nodes / c
    t_per_card = time_ms / c

    x_bt = math.log10(1.0 + bt_per_card)
    x_nv = math.log10(1.0 + nv_per_card)
    x_t  = math.log10(1.0 + t_per_card)

    raw = 0.55 * x_bt + 0.30 * x_nv + 0.15 * x_t
    hardness01 = _sigmoid((raw - mu) / sigma)

    lo, hi = TARGET_BANDS[difficulty]
    passed = (lo <= hardness01 <= hi)

    return {
        "hardness01": round(hardness01, 4),
        "components": {
            "btPerCard": round(bt_per_card, 2),
            "nvPerCard": round(nv_per_card, 2),
            "tPerCardMs": round(t_per_card, 2),
            "raw": round(raw, 4),
            "mu": mu,
            "sigma": sigma,
            "maxDepth": int(depth)
        },
        "targetBand": [lo, hi],
        "passedBand": passed
    }
