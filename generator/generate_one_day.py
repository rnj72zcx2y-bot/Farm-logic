import hashlib
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import random
import networkx as nx

from .config import SPECS, GENERATOR_VERSION, GIT_COMMIT, GLOBAL_SALT, FENCE_COLORS, ANIMALS
from .solver_unique import solve_count
from .difficulty_score import compute_hardness

# ------------------------------------------------------------
# DEV-STABLE (FAST) MODE
# Goal: finish runs quickly and reliably.
# - Tries for uniqueness first, but with STRICT attempt caps per difficulty.
# - If uniqueness isn't found quickly, falls back to the first solvable candidate.
# - Always logs solutionsFound + uniqueSolution + difficultyScore.
# ------------------------------------------------------------


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build_seed_str(date_utc: str, difficulty: str) -> str:
    return "sha256:" + sha256_hex(f"{date_utc}|{difficulty}|{GLOBAL_SALT}")


def build_seed_int(date_utc: str, difficulty: str, attempt: int) -> int:
    seed_input = f"{date_utc}|{difficulty}|{attempt}|{GLOBAL_SALT}"
    return int(sha256_hex(seed_input)[:16], 16)


def neighbors(cell: Tuple[int, int], rows: int, cols: int) -> List[Tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            out.append((rr, cc))
    return out


def make_simple_active_shape(difficulty: str, rows: int, cols: int) -> List[List[int]]:
    """Deterministic domino-tileable shapes (dev stable)."""
    active: List[List[int]] = []
    if difficulty == "easy":
        for r in range(rows):
            for c in range(cols - 1):
                active.append([r, c])
    elif difficulty == "medium":
        for r in range(rows - 1):
            for c in range(cols):
                active.append([r, c])
    else:
        for r in range(rows - 2):
            for c in range(cols):
                active.append([r, c])
    return active


def make_blockers(rows: int, cols: int, active_cells: List[List[int]], rng: random.Random) -> List[Dict[str, Any]]:
    active_set = {tuple(x) for x in active_cells}
    types = ["tree", "house", "tractor"]
    blocked = []
    k = 0
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in active_set:
                blocked.append({"cell": [r, c], "type": types[k % 3]})
                k += 1
    rng.shuffle(blocked)
    return blocked


def bipartite_matching_tiling(active_cells: List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Return one domino tiling as list of cell pairs using bipartite matching."""
    U = [cell for cell in active_cells if (cell[0] + cell[1]) % 2 == 0]
    V = [cell for cell in active_cells if (cell[0] + cell[1]) % 2 == 1]

    if len(active_cells) % 2 == 1 or len(U) != len(V):
        return []

    active_set = set(active_cells)

    G = nx.Graph()
    G.add_nodes_from(U, bipartite=0)
    G.add_nodes_from(V, bipartite=1)

    for (r, c) in U:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in active_set:
                G.add_edge((r, c), nb)

    matching = nx.algorithms.bipartite.maximum_matching(G, top_nodes=U)

    pairs = []
    used = set()
    for u in U:
        if u in matching:
            v = matching[u]
            if u in used or v in used:
                continue
            used.add(u)
            used.add(v)
            pairs.append((u, v))

    if len(used) != len(active_cells):
        return []

    return pairs


def _balance_regions(regions: List[set], active_set: set, rows: int, cols: int, rng: random.Random,
                     min_size: int, max_size: int) -> None:
    """Reduce tiny regions by moving boundary cells from larger regions."""

    def boundary_cells(reg: set) -> List[Tuple[int, int]]:
        b = []
        for cell in reg:
            for nb in neighbors(cell, rows, cols):
                if nb in active_set and nb not in reg:
                    b.append(cell)
                    break
        return b

    for _ in range(40):
        small = [i for i, r in enumerate(regions) if len(r) < min_size]
        if not small:
            return
        big = [i for i, r in enumerate(regions) if len(r) > max_size]
        if not big:
            big = [i for i, r in enumerate(regions) if len(r) > min_size]
            if not big:
                return

        si = rng.choice(small)
        bi = rng.choice(big)
        b_cells = boundary_cells(regions[bi])
        if not b_cells:
            continue
        cell = rng.choice(b_cells)
        regions[bi].remove(cell)
        regions[si].add(cell)


def partition_regions(active_cells: List[Tuple[int, int]], region_count: int, rows: int, cols: int,
                      rng: random.Random, min_size: int, max_size: int) -> List[List[List[int]]]:
    active_set = set(active_cells)
    seeds = rng.sample(active_cells, k=min(region_count, len(active_cells)))
    regions = [set([s]) for s in seeds]
    unassigned = active_set - set(seeds)

    frontiers = [set([s]) for s in seeds]
    while unassigned:
        expandable = []
        for i, fr in enumerate(frontiers):
            if len(regions[i]) >= max_size:
                continue
            for cell in fr:
                for nb in neighbors(cell, rows, cols):
                    if nb in unassigned:
                        expandable.append(i)
                        break
                if i in expandable:
                    break

        if not expandable:
            for cell in list(unassigned):
                i = min(range(len(regions)), key=lambda k: len(regions[k]))
                regions[i].add(cell)
                unassigned.remove(cell)
            break

        i = rng.choice(expandable)
        cand = set()
        for cell in frontiers[i]:
            for nb in neighbors(cell, rows, cols):
                if nb in unassigned:
                    cand.add(nb)
        if not cand:
            continue

        nb = rng.choice(list(cand))
        regions[i].add(nb)
        unassigned.remove(nb)
        frontiers[i].add(nb)

        if len(frontiers[i]) > 18:
            frontiers[i] = set(rng.sample(list(frontiers[i]), 9))

    _balance_regions(regions, active_set, rows, cols, rng, min_size=min_size, max_size=max_size)

    return [[[r, c] for (r, c) in sorted(reg)] for reg in regions]


def pick_animals_for_card(rng: random.Random, difficulty: str) -> Tuple[Optional[str], Optional[str]]:
    # Reduce empties in medium/hard to reduce symmetry and speed up solve
    empty_p = {"easy": 0.10, "medium": 0.06, "hard": 0.04}[difficulty]
    names = [a for a, _legs in ANIMALS]

    w = {name: 1.0 for name in names}
    w["Chicken"] = 1.2
    w["Bee"] = 0.95
    w["Spider"] = 0.85 if difficulty != "hard" else 1.05
    w["Snail"] = 0.7 if difficulty == "easy" else 0.9

    weights = [w[n] for n in names]

    def sample_one() -> Optional[str]:
        if rng.random() < empty_p:
            return None
        return rng.choices(names, weights=weights, k=1)[0]

    # avoid None/None
    for _ in range(8):
        a, b = sample_one(), sample_one()
        if not (a is None and b is None):
            return a, b
    return sample_one(), sample_one()


def legs_of(animal: Optional[str]) -> int:
    if animal is None:
        return 0
    for n, l in ANIMALS:
        if n == animal:
            return l
    return 0


def choose_rule_for_region(rng: random.Random, difficulty: str,
                           region_cells: List[Tuple[int, int]],
                           cell_animal: Dict[Tuple[int, int], Optional[str]]) -> Dict[str, Any]:
    animals = [cell_animal[c] for c in region_cells]
    legs_sum = sum(legs_of(a) for a in animals)
    animal_count = sum(1 for a in animals if a is not None)

    species = [a for a in animals if a is not None]
    has_empty = any(a is None for a in animals)

    strong_unique_ok = (len(region_cells) >= 3) and (not has_empty) and (len(species) == len(region_cells)) and (len(set(species)) == len(species))
    only_ok = (not has_empty) and (len(species) == len(region_cells)) and (len(set(species)) == 1)

    size = len(region_cells)

    if difficulty == "easy":
        choices = ["legs_eq", "animals_eq", "unique", "only"]
        weights = [0.62, 0.33, 0.04, 0.01]
    elif difficulty == "medium":
        choices = ["legs_eq", "animals_eq", "unique"]
        weights = [0.50, 0.33, 0.17]
    else:
        choices = ["unique", "legs_eq", "animals_eq"]
        weights = [0.55, 0.25, 0.20]

    if size <= 2:
        choices = ["legs_eq", "animals_eq"]
        weights = [0.7, 0.3]

    filtered, filtered_w = [], []
    for ch, w in zip(choices, weights):
        if ch == "unique" and not strong_unique_ok:
            continue
        if ch == "only" and not only_ok:
            continue
        filtered.append(ch)
        filtered_w.append(w)

    if not filtered:
        filtered, filtered_w = ["legs_eq"], [1.0]

    rule_type = rng.choices(filtered, weights=filtered_w, k=1)[0]

    if rule_type == "legs_eq":
        return {"type": "legs", "op": "=", "value": legs_sum}
    if rule_type == "animals_eq":
        return {"type": "animals", "op": "=", "value": animal_count}
    if rule_type == "unique":
        return {"type": "uniqueSpecies"}
    if rule_type == "only":
        return {"type": "onlySpecies", "species": species[0]}

    return {"type": "legs", "op": "=", "value": legs_sum}


def generate_candidate(date_utc: str, difficulty: str, attempt: int) -> Dict[str, Any]:
    spec = SPECS[difficulty]
    rng = random.Random(build_seed_int(date_utc, difficulty, attempt))

    rows, cols = spec.rows, spec.cols
    active_ll = make_simple_active_shape(difficulty, rows, cols)
    active = [tuple(x) for x in active_ll]

    blocked = make_blockers(rows, cols, active_ll, rng)

    pairs = bipartite_matching_tiling(active)

    # reduce duplicate card pairs (symmetry)
    max_dupes = {"easy": 2, "medium": 1, "hard": 1}[difficulty]
    pair_counts: Dict[Tuple[str, str], int] = {}

    fences = [FENCE_COLORS[i % 3] for i in range(spec.cards)]
    rng.shuffle(fences)

    cards = []
    for i in range(spec.cards):
        for _ in range(50):
            a, b = pick_animals_for_card(rng, difficulty)
            key = tuple(sorted([(a or "_"), (b or "_")]))
            pair_counts[key] = pair_counts.get(key, 0) + 1
            if pair_counts[key] <= max_dupes:
                cards.append({
                    "id": f"C{i+1:02d}",
                    "fence": fences[i],
                    "a": a,
                    "b": b,
                })
                break
            pair_counts[key] -= 1
        else:
            cards.append({"id": f"C{i+1:02d}", "fence": fences[i], "a": "Snail", "b": None})

    rng.shuffle(pairs)
    cell_animal: Dict[Tuple[int, int], Optional[str]] = {}
    for i, (u, v) in enumerate(pairs[:spec.cards]):
        card = cards[i]
        if rng.random() < 0.5:
            cell_animal[u] = card["a"]
            cell_animal[v] = card["b"]
        else:
            cell_animal[u] = card["b"]
            cell_animal[v] = card["a"]

    for c in active:
        cell_animal.setdefault(c, None)

    if difficulty == "hard":
        min_size, max_size = 3, 4
    elif difficulty == "medium":
        min_size, max_size = 2, 4
    else:
        min_size, max_size = 2, 5

    region_cells_ll = partition_regions(active, spec.regions, rows, cols, rng, min_size=min_size, max_size=max_size)

    regions = []
    for idx, cell_list in enumerate(region_cells_ll):
        rcells = [tuple(x) for x in cell_list]
        rule = choose_rule_for_region(rng, difficulty, rcells, cell_animal)
        regions.append({"id": f"R{idx+1}", "cells": cell_list, "rule": rule})

    return {
        "dateUtc": date_utc,
        "difficulty": difficulty,
        "schemaVersion": 1,
        "generatorVersion": GENERATOR_VERSION,
        "grid": {"rows": rows, "cols": cols, "activeCellsCoords": active_ll, "blocked": blocked},
        "regions": regions,
        "cards": cards,
        "_internal": {"seed": build_seed_str(date_utc, difficulty), "attempt": attempt},
    }


def generate_unique(date_utc: str, difficulty: str) -> Dict[str, Any]:
    """FAST: strict attempt caps so workflow finishes quickly."""
    spec = SPECS[difficulty]

    attempt_caps = {"easy": 250, "medium": 40, "hard": 60}
    timeout_caps = {"easy": 8, "medium": 5, "hard": 8}

    max_attempts = attempt_caps.get(difficulty, 60)
    max_timeouts = timeout_caps.get(difficulty, 6)

    best_solvable = None
    best_stats = None
    best_score = None

    timeouts = 0

    for attempt in range(max_attempts):
        puzzle = generate_candidate(date_utc, difficulty, attempt)

        solutions, stats = solve_count(puzzle, stop_at=2)

        if stats.get("timedOut"):
            timeouts += 1
            if timeouts >= max_timeouts and best_solvable is not None:
                break

        if solutions <= 0:
            continue

        score = compute_hardness(difficulty, spec.cards, stats)

        if best_solvable is None:
            best_solvable, best_stats, best_score = puzzle, stats, score

        if solutions == 1:
            puzzle["_internal"]["uniqueSolution"] = True
            puzzle["_internal"]["solverStats"] = stats
            puzzle["_internal"]["difficultyScore"] = score
            return puzzle

    # fallback
    if best_solvable is None:
        best_solvable = generate_candidate(date_utc, difficulty, 0)
        best_stats = {"type": "CSP-backtracking", "stopAt": 2, "solutionsFound": 0, "nodesVisited": 0, "backtracks": 0, "maxDepth": 0, "timeMs": 0, "timedOut": False}
        best_score = {"hardness01": 0.0, "passedBand": False}

    best_solvable["_internal"]["uniqueSolution"] = False
    best_solvable["_internal"]["solverStats"] = best_stats
    best_solvable["_internal"]["difficultyScore"] = best_score
    return best_solvable


def build_meta(date_utc: str, puzzles_by_diff: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    meta = {
        "dateUtc": date_utc,
        "schemaVersion": 1,
        "generatorVersion": GENERATOR_VERSION,
        "gitCommit": GIT_COMMIT,
        "generatedAtUtc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "notes": {
            "animals": ["Chicken", "Cow", "Horse", "Dog", "Cat", "Bee", "Spider", "Snail"],
            "counting": "rules evaluate per CELL; cards may cross regions",
            "emptyAllowed": True,
            "fenceColor": "white/brown/black only (hint-only; not used in rules)",
            "ui": {"checkEnabledOnlyWhenAllCardsPlaced": True, "snapToValid": True},
        },
        "difficulties": {},
    }

    for diff, puzzle in puzzles_by_diff.items():
        spec = SPECS[diff]
        stats = puzzle["_internal"].get("solverStats", {})
        score = puzzle["_internal"].get("difficultyScore", {})
        unique_flag = puzzle["_internal"].get("uniqueSolution", False)

        public_puzzle = {k: v for k, v in puzzle.items() if k != "_internal"}
        puzzle_hash = "sha256:" + sha256_hex(json.dumps(public_puzzle, sort_keys=True))

        meta["difficulties"][diff] = {
            "grid": {
                "rows": public_puzzle["grid"]["rows"],
                "cols": public_puzzle["grid"]["cols"],
                "activeCells": len(public_puzzle["grid"]["activeCellsCoords"]),
                "cards": spec.cards,
            },
            "seed": puzzle["_internal"]["seed"],
            "uniqueSolution": bool(unique_flag),
            "solver": {
                "type": stats.get("type"),
                "stopAt": stats.get("stopAt", 2),
                "stats": {
                    "solutionsFound": stats.get("solutionsFound"),
                    "nodesVisited": stats.get("nodesVisited"),
                    "backtracks": stats.get("backtracks"),
                    "maxDepth": stats.get("maxDepth"),
                    "timeMs": stats.get("timeMs"),
                    "timedOut": stats.get("timedOut"),
                    "timeLimitSec": stats.get("timeLimitSec"),
                },
            },
            "difficultyScore": score,
            "hash": {"puzzleJson": puzzle_hash},
        }

    return meta
