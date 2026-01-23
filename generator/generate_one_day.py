import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import random
import networkx as nx

from .config import SPECS, GENERATOR_VERSION, GIT_COMMIT, GLOBAL_SALT, FENCE_COLORS, ANIMALS
from .solver_unique import solve_count
from .difficulty_score import compute_hardness

# ------------------------------------------------------------
# generate_one_day.py (Flexible regions inside this file)
#
# FULL DROP-IN REPLACEMENT
#
# Added in this version:
# - Quick-Score pre-solver filter to reduce symmetry early
#
# Everything else:
# - identical architecture
# - identical JSON schema
# - identical solver usage
# ------------------------------------------------------------


# ============================================================
# Utilities
# ============================================================

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


# ============================================================
# Grid / Shape
# ============================================================

def make_simple_active_shape(difficulty: str, rows: int, cols: int) -> List[List[int]]:
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


# ============================================================
# Domino tiling
# ============================================================

def bipartite_matching_tiling(active_cells: List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
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


# ============================================================
# Flexible Regions
# ============================================================

def region_ranges(difficulty: str) -> Tuple[int, int]:
    return (3, 6) if difficulty == "easy" else (8, 14) if difficulty == "medium" else (12, 20)


def region_size_limits(difficulty: str) -> Tuple[int, int]:
    return (2, 6) if difficulty == "hard" else (2, 5)


def sample_region_count_and_sizes(n_cells: int, difficulty: str, rng: random.Random) -> Tuple[int, List[int]]:
    rmin, rmax = region_ranges(difficulty)
    min_size, max_size = region_size_limits(difficulty)

    for _ in range(300):
        R = rng.randint(rmin, rmax)
        if R * min_size <= n_cells <= R * max_size:
            sizes = [min_size] * R
            remaining = n_cells - R * min_size
            while remaining > 0:
                i = rng.randrange(R)
                if sizes[i] < max_size:
                    sizes[i] += 1
                    remaining -= 1
            rng.shuffle(sizes)
            return R, sizes

    R = max(rmin, min(rmax, n_cells // min_size))
    sizes = [min_size] * R
    remaining = n_cells - R * min_size
    i = 0
    while remaining > 0:
        if sizes[i] < max_size:
            sizes[i] += 1
            remaining -= 1
        i = (i + 1) % R
    rng.shuffle(sizes)
    return R, sizes


def choose_seed_strategy(rng: random.Random) -> str:
    return rng.choice(["spread", "cluster", "border", "center"])


def choose_mode_weights(rng: random.Random) -> Dict[str, float]:
    a, b, c = rng.random(), rng.random(), rng.random()
    s = a + b + c
    return {"island": a / s, "snake": b / s, "zigzag": c / s}


def sample_modes(R: int, weights: Dict[str, float], rng: random.Random) -> List[str]:
    modes = ["island", "snake", "zigzag"]
    w = [weights[m] for m in modes]
    return [rng.choices(modes, weights=w, k=1)[0] for _ in range(R)]


def build_regions_flexible(active, rows, cols, difficulty, rng, max_retries=220):
    active_set = set(active)
    for _ in range(max_retries):
        R, sizes = sample_region_count_and_sizes(len(active), difficulty, rng)
        modes = sample_modes(R, choose_mode_weights(rng), rng)
        seeds = random.sample(active, R)
        regions = [set([seeds[i]]) for i in range(R)]
        remaining = [sizes[i] - 1 for i in range(R)]
        unassigned = set(active) - set(seeds)

        while unassigned:
            progress = False
            for i in range(R):
                if remaining[i] <= 0:
                    continue
                frontier = {nb for cell in regions[i] for nb in neighbors(cell, rows, cols) if nb in unassigned}
                if not frontier:
                    continue
                pick = rng.choice(list(frontier))
                regions[i].add(pick)
                remaining[i] -= 1
                unassigned.remove(pick)
                progress = True
            if not progress:
                break

        if not unassigned and all(r == 0 for r in remaining):
            return [[[r, c] for (r, c) in reg] for reg in regions], modes
    return None


# ============================================================
# Cards / Animals
# ============================================================

def legs_of(animal: Optional[str]) -> int:
    for n, l in ANIMALS:
        if n == animal:
            return l
    return 0


def pick_animals_for_card(rng: random.Random, difficulty: str):
    empty_p = {"easy": 0.05, "medium": 0.05, "hard": 0.04}[difficulty]
    names = [a for a, _ in ANIMALS]

    def pick():
        return None if rng.random() < empty_p else rng.choice(names)

    for _ in range(10):
        a, b = pick(), pick()
        if not (a is None and b is None):
            return a, b
    return pick(), pick()


# ============================================================
# Rules
# ============================================================

def choose_op(rng: random.Random, difficulty: str) -> str:
    p = {"easy": 0.97, "medium": 0.92, "hard": 0.75}[difficulty]
    if rng.random() < p:
        return "="
    return "<" if rng.random() < 0.5 else ">"


def choose_rule_for_region(rng, difficulty, region_cells, cell_animal):
    animals = [cell_animal[c] for c in region_cells]
    legs_sum = sum(legs_of(a) for a in animals if a)
    animal_count = sum(1 for a in animals if a)

    if difficulty == "easy":
        rule_type = rng.choices(["legs", "animals"], [0.7, 0.3])[0]
    elif difficulty == "medium":
        rule_type = rng.choices(["legs", "animals"], [0.6, 0.4])[0]
    else:
        rule_type = rng.choices(["legs", "animals"], [0.5, 0.5])[0]

    op = choose_op(rng, difficulty)
    val = legs_sum if rule_type == "legs" else animal_count
    return {"type": rule_type, "op": op, "value": int(val)}


# ============================================================
# Candidate Generation
# ============================================================

def generate_candidate(date_utc: str, difficulty: str, attempt: int):
    spec = SPECS[difficulty]
    rng = random.Random(build_seed_int(date_utc, difficulty, attempt))

    rows, cols = spec.rows, spec.cols
    active_ll = make_simple_active_shape(difficulty, rows, cols)
    active = [tuple(x) for x in active_ll]

    blocked = make_blockers(rows, cols, active_ll, rng)
    pairs = bipartite_matching_tiling(active)
    if not pairs:
        return None

    cards = []
    fences = [FENCE_COLORS[i % 3] for i in range(spec.cards)]
    rng.shuffle(fences)

    for i in range(spec.cards):
        a, b = pick_animals_for_card(rng, difficulty)
        cards.append({"id": f"C{i+1:02d}", "fence": fences[i], "a": a, "b": b})

    rng.shuffle(pairs)
    cell_animal = {}
    for i, (u, v) in enumerate(pairs[:spec.cards]):
        card = cards[i]
        if rng.random() < 0.5:
            cell_animal[u], cell_animal[v] = card["a"], card["b"]
        else:
            cell_animal[u], cell_animal[v] = card["b"], card["a"]

    regions_built = build_regions_flexible(active, rows, cols, difficulty, rng)
    if not regions_built:
        return None

    region_cells_ll, _modes = regions_built
    regions = []
    for i, cells in enumerate(region_cells_ll):
        rc = [tuple(c) for c in cells]
        rule = choose_rule_for_region(rng, difficulty, rc, cell_animal)
        regions.append({"id": f"R{i+1}", "cells": cells, "rule": rule})

    solution_cells = {f"{r},{c}": cell_animal.get((r, c)) for (r, c) in active}

    return {
        "dateUtc": date_utc,
        "difficulty": difficulty,
        "schemaVersion": 1,
        "generatorVersion": GENERATOR_VERSION,
        "grid": {"rows": rows, "cols": cols, "activeCellsCoords": active_ll, "blocked": blocked},
        "regions": regions,
        "cards": cards,
        "_internal": {
            "seed": build_seed_str(date_utc, difficulty),
            "attempt": attempt,
            "solutionCells": solution_cells,
        },
    }


# ============================================================
# QUICK SCORE (NEW)
# ============================================================

QUICK_SCORE_MIN = {"easy": 2.5, "medium": 4.5, "hard": 4.0}


def quick_score_candidate(puzzle: Dict[str, Any]) -> float:
    regs = puzzle.get("regions", [])
    sol = puzzle.get("_internal", {}).get("solutionCells", {})
    if not regs or not sol:
        return -999.0

    score = 0.0
    sizes = [len(r["cells"]) for r in regs]
    score += len(set(sizes)) * 0.8

    rule_types = [r["rule"]["type"] for r in regs]
    score += len(set(rule_types)) * 1.5

    legsums = []
    for r in regs:
        s = 0
        for rr, cc in [tuple(c) for c in r["cells"]]:
            a = sol.get(f"{rr},{cc}")
            if a:
                s += legs_of(a)
        legsums.append(s)

    freq = {}
    for v in legsums:
        freq[v] = freq.get(v, 0) + 1
    score -= sum(v - 1 for v in freq.values() if v > 1) * 1.2

    return score


# ============================================================
# Generate Unique
# ============================================================

def generate_unique(date_utc: str, difficulty: str) -> Dict[str, Any]:
    spec = SPECS[difficulty]

    max_attempts = {"easy": 260, "medium": 120, "hard": 110}[difficulty]
    max_timeouts = {"easy": 12, "medium": 8, "hard": 12}[difficulty]

    diag_enabled = os.getenv("SOLVER_DIAG_ENABLED", "true").lower() == "true"
    diag_time_limit = float(os.getenv("SOLVER_DIAG_TIME_LIMIT_SEC", "2.0"))

    best = None
    best_stats = None
    best_score = None
    timeouts = 0

    for attempt in range(max_attempts):
        puzzle = generate_candidate(date_utc, difficulty, attempt)
        if puzzle is None:
            continue

        qs = quick_score_candidate(puzzle)
        puzzle["_internal"]["quickScore"] = qs
        if qs < QUICK_SCORE_MIN[difficulty]:
            continue

        solutions2, stats2 = solve_count(puzzle, stop_at=2)

        if stats2.get("timedOut"):
            timeouts += 1
            if timeouts >= max_timeouts and best:
                break

        if solutions2 == 1:
            puzzle["_internal"]["uniqueSolution"] = True
            puzzle["_internal"]["solverStats"] = stats2
            puzzle["_internal"]["difficultyScore"] = compute_hardness(difficulty, spec.cards, stats2)
            return puzzle

        if solutions2 > 0:
            score2 = compute_hardness(difficulty, spec.cards, stats2)
            if best is None:
                best, best_stats, best_score = puzzle, stats2, score2

    if best is None:
        raise RuntimeError(f"No solvable puzzle for {date_utc} {difficulty}")

    best["_internal"]["uniqueSolution"] = False
    best["_internal"]["solverStats"] = best_stats
    best["_internal"]["difficultyScore"] = best_score
    return best


# ============================================================
# META
# ============================================================

def build_meta(date_utc: str, puzzles_by_diff: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    meta = {
        "dateUtc": date_utc,
        "schemaVersion": 1,
        "generatorVersion": GENERATOR_VERSION,
        "gitCommit": GIT_COMMIT,
        "generatedAtUtc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "difficulties": {},
    }

    for diff, puzzle in puzzles_by_diff.items():
        stats = puzzle["_internal"].get("solverStats", {})
        score = puzzle["_internal"].get("difficultyScore", {})
        meta["difficulties"][diff] = {
            "uniqueSolution": puzzle["_internal"].get("uniqueSolution"),
            "solver": stats,
            "difficultyScore": score,
        }

    return meta
