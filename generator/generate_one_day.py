import hashlib
import json
from datetime import datetime
from typing import Dict, Any

from .config import SPECS, GENERATOR_VERSION, GIT_COMMIT, GLOBAL_SALT
from .solver_unique import solve_count
from .difficulty_score import compute_hardness


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build_seed(date_utc: str, difficulty: str) -> str:
    seed_input = f"{date_utc}|{difficulty}|{GLOBAL_SALT}"
    return sha256_hex(seed_input)



def generate_candidate(date_utc: str, difficulty: str, attempt: int) -> Dict[str, Any]:
    """
    TEMP generator: creates a simple connected, domino-tileable active shape 
    so the solver can be developed and tested. 
    Replace later with real shape/regions/cards/rules generator.
    """
    spec = SPECS[difficulty]
    seed = build_seed(date_utc, difficulty)
    rows, cols = spec.rows, spec.cols
    
    # Simple deterministic shapes (domino-tileable)
    active = []
    
    if difficulty == "easy":
        # 4x4 minus last column => 12 cells
        for r in range(rows):
            for c in range(cols - 1):
                active.append([r, c])
                
    elif difficulty == "medium":
        # 6x6 minus last row => 30 cells
        for r in range(rows - 1):
            for c in range(cols):
                active.append([r, c])
                
    else:
        # 8x8 minus last two rows => 48 cells
        for r in range(rows - 2):
            for c in range(cols):
                active.append([r, c])

    # Blockers = everything else
    active_set = {tuple(x) for x in active}
    blocked = []
    blocker_types = ["tree", "house", "tractor"]
    k = 0
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in active_set:
                blocked.append({
                    "cell": [r, c], 
                    "type": blocker_types[k % 3]
                })
                k += 1

    # TEMP cards: correct count, neutral content (null/null)
    # Fence is hint-only but we still fill it.
    fences = ["white", "brown", "black"]
    cards = []
    
    for i in range(spec.cards):
        cards.append({
            "id": f"C{i+1:02d}",
            "fence": fences[(i + attempt) % 3],
            "a": None,
            "b": None
        })

    # TEMP regions/rules: empty for now (solver stage 1)
    return {
        "dateUtc": date_utc,
        "difficulty": difficulty,
        "schemaVersion": 1,
        "generatorVersion": GENERATOR_VERSION,
        "grid": {
            "rows": rows,
            "cols": cols,
            "activeCellsCoords": active,
            "blocked": blocked
        },
        "regions": [],
        "cards": cards,
        "_internal": {
            "seed": "sha256:" + seed, 
            "attempt": attempt
        }
    }



def generate_unique(date_utc: str, difficulty: str, max_attempts: int = 300) -> Dict[str, Any]:
    """Generate a puzzle that is uniquely solvable.

    TEMP: Hardness gating disabled until real generator + CSP solver are implemented.
    """
    spec = SPECS[difficulty]

    for attempt in range(max_attempts):
        puzzle = generate_candidate(date_utc, difficulty, attempt)

        # Uniqueness check
        solutions, stats = solve_count(puzzle, stop_at=2)
        if solutions != 1:
            continue

        # TEMP: disable hardness gating for now
        score = compute_hardness(difficulty, spec.cards, stats)
        score["passedBand"] = True

        puzzle["_internal"]["uniqueSolution"] = True
        puzzle["_internal"]["solverStats"] = stats
        puzzle["_internal"]["difficultyScore"] = score
        return puzzle

    raise RuntimeError(f"Could not generate acceptable puzzle for {date_utc} {difficulty}")


def build_meta(date_utc: str, puzzles_by_diff: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    meta = {
        "dateUtc": date_utc,
        "schemaVersion": 1,
        "generatorVersion": GENERATOR_VERSION,
        "gitCommit": GIT_COMMIT,
        "generatedAtUtc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "notes": {
            "fenceColor": "white/brown/black only (hint-only; not used in rules)",
            "animals": ["Chicken", "Cow", "Horse", "Dog", "Cat", "Bee", "Spider", "Snail"],
            "emptyAllowed": True,
            "counting": "rules evaluate per CELL; cards may cross regions",
            "ui": {
                "snapToValid": True,
                "checkEnabledOnlyWhenAllCardsPlaced": True
            }
        },
        "difficulties": {}
    }

    for diff, puzzle in puzzles_by_diff.items():
        spec = SPECS[diff]
        stats = puzzle["_internal"]["solverStats"]
        score = puzzle["_internal"]["difficultyScore"]

        public_puzzle = {k: v for k, v in puzzle.items() if k != "_internal"}
        puzzle_hash = "sha256:" + sha256_hex(json.dumps(public_puzzle, sort_keys=True))

        meta["difficulties"][diff] = {
            "grid": {
                "rows": public_puzzle["grid"]["rows"],
                "cols": public_puzzle["grid"]["cols"],
                "activeCells": len(public_puzzle["grid"]["activeCellsCoords"]),
                "cards": spec.cards
            },
            "seed": puzzle["_internal"]["seed"],
            "uniqueSolution": True,
            "solver": {
                "type": stats.get("type"),
                "stopAt": stats.get("stopAt", 2),
                "stats": {
                    "solutionsFound": stats.get("solutionsFound"),
                    "nodesVisited": stats.get("nodesVisited"),
                    "backtracks": stats.get("backtracks"),
                    "maxDepth": stats.get("maxDepth"),
                    "timeMs": stats.get("timeMs")
                }
            },
            "difficultyScore": score,
            "hash": {"puzzleJson": puzzle_hash}
        }

    return meta
