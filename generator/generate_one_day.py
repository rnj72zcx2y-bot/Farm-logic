import hashlib
import json
import os
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
# User requirement:
# - Region count should be flexible per difficulty (within a range)
# - Region sizes, criteria, and location should vary a lot
# - Shapes should include islands / zigzags / snakes
# - Keep JSON structure: regions[] with cells + rule
# - UI does not need to show region count
#
# Uniqueness Gate (best practice): stop_at=2
# Diagnostic (enabled): if solutionsFound>=2, run stop_at=3 with small time budget
#
# Diagnostic controls (env):
#   SOLVER_DIAG_ENABLED=true|false (default true)
#   SOLVER_DIAG_TIME_LIMIT_SEC=2.0 (default 2.0)
# ------------------------------------------------------------


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build_seed_str(date_utc: str, difficulty: str) -> str:
    return "sha256:" + sha256_hex(f"{date_utc}|{difficulty}|{GLOBAL_SALT}")


def build_seed_int(date_utc: str, difficulty: str, attempt: int) -> int:
    seed_input = f"{date_utc}|{difficulty}|{attempt}|{GLOBAL_SALT}"
    return int(sha256_hex(seed_input)[:16], 16)


def cell_key(cell: Tuple[int, int]) -> str:
    return f"{cell[0]},{cell[1]}"


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


# ----------------------------
# Flexible regions (core)
# ----------------------------

def region_ranges(difficulty: str) -> Tuple[int, int]:
    if difficulty == "easy":
        return 3, 6
    if difficulty == "medium":
        return 8, 14
    return 12, 20


def region_size_limits(difficulty: str) -> Tuple[int, int]:
    if difficulty == "hard":
        return 2, 6
    return 2, 5


def sample_region_count_and_sizes(n_cells: int, difficulty: str, rng: random.Random) -> Tuple[int, List[int]]:
    """Pick regionCount and exact target sizes such that sum(sizes)=n_cells.

    We enforce min/max region size. We resample regionCount until feasible.
    """
    rmin, rmax = region_ranges(difficulty)
    min_size, max_size = region_size_limits(difficulty)

    for _ in range(200):
        R = rng.randint(rmin, rmax)
        if R * min_size > n_cells:
            continue
        if R * max_size < n_cells:
            continue

        sizes = [min_size] * R
        remaining = n_cells - R * min_size

        # Randomly distribute remaining cells without exceeding max_size.
        buckets = list(range(R))
        while remaining > 0:
            rng.shuffle(buckets)
            placed_any = False
            for i in buckets:
                if remaining <= 0:
                    break
                if sizes[i] < max_size:
                    sizes[i] += 1
                    remaining -= 1
                    placed_any = True
            if not placed_any:
                break

        if remaining == 0:
            rng.shuffle(sizes)
            return R, sizes

    # Fallback: fixed R nearest feasible
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
    # flexible weights per puzzle
    a, b, c = rng.random(), rng.random(), rng.random()
    s = a + b + c
    return {
        "island": a / s,
        "snake": b / s,
        "zigzag": c / s,
    }


def sample_modes(R: int, weights: Dict[str, float], rng: random.Random) -> List[str]:
    modes = ["island", "snake", "zigzag"]
    w = [weights[m] for m in modes]
    return [rng.choices(modes, weights=w, k=1)[0] for _ in range(R)]


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def farthest_point_seeds(active: List[Tuple[int, int]], R: int, rng: random.Random) -> List[Tuple[int, int]]:
    """Greedy farthest-point sampling (spread seeds)."""
    seeds = [rng.choice(active)]
    while len(seeds) < R:
        best = None
        best_d = -1
        for cell in active:
            d = min(manhattan(cell, s) for s in seeds)
            if d > best_d:
                best_d = d
                best = cell
        seeds.append(best)
    return seeds


def select_seeds(active: List[Tuple[int, int]], R: int, rows: int, cols: int, rng: random.Random, strategy: str) -> List[Tuple[int, int]]:
    active_set = set(active)

    if strategy == "spread":
        return farthest_point_seeds(active, R, rng)

    if strategy == "cluster":
        # pick a start, then bias toward nearby cells
        seeds = [rng.choice(active)]
        while len(seeds) < R:
            # candidate pool = neighbors within radius 2..4 of existing seeds
            pool = []
            for cell in active:
                d = min(manhattan(cell, s) for s in seeds)
                if d <= rng.randint(2, 4):
                    pool.append(cell)
            if not pool:
                pool = active
            seeds.append(rng.choice(pool))
        return seeds

    if strategy == "border":
        # bias to border-ish active cells
        scored = []
        for cell in active:
            r, c = cell
            dist_border = min(r, c, rows - 1 - r, cols - 1 - c)
            scored.append((dist_border, cell))
        scored.sort(key=lambda x: x[0])
        pool = [c for _, c in scored[: max(R * 3, min(len(scored), 30))]]
        if len(pool) < R:
            pool = active
        return rng.sample(pool, k=R)

    # center
    cr, cc = (rows - 1) / 2.0, (cols - 1) / 2.0
    scored = []
    for cell in active:
        r, c = cell
        dist = abs(r - cr) + abs(c - cc)
        scored.append((dist, cell))
    scored.sort(key=lambda x: x[0])
    pool = [c for _, c in scored[: max(R * 3, min(len(scored), 30))]]
    if len(pool) < R:
        pool = active
    return rng.sample(pool, k=R)


def perimeter_delta(region: set, cand: Tuple[int, int], rows: int, cols: int, active_set: set) -> int:
    """Approximate perimeter change if cand is added.

    We treat perimeter as count of region edges that touch outside.
    Delta computed locally around cand.
    """
    # for cand: each side that touches outside adds +1; each side that touches region removes 1 from cand's contribution
    delta = 0
    for nb in neighbors(cand, rows, cols):
        if nb not in active_set:
            continue
        if nb in region:
            delta -= 1
        else:
            delta += 1
    return delta


def neighbor_in_region_count(region: set, cand: Tuple[int, int], rows: int, cols: int, active_set: set) -> int:
    return sum(1 for nb in neighbors(cand, rows, cols) if nb in active_set and nb in region)


def score_candidate(mode: str, region: set, cand: Tuple[int, int], rows: int, cols: int, active_set: set, rng: random.Random) -> float:
    """Heuristic scoring for shape modes."""
    n_in = neighbor_in_region_count(region, cand, rows, cols, active_set)
    pd = perimeter_delta(region, cand, rows, cols, active_set)

    # add small noise so shapes vary even for similar scores
    noise = (rng.random() - 0.5) * 0.15

    if mode == "island":
        # compact: prefer many neighbors in region, small perimeter increase
        return 1.2 * n_in - 0.35 * pd + noise

    if mode == "snake":
        # stringy: prefer few neighbors, larger perimeter increase
        return -1.0 * n_in + 0.45 * pd + noise

    # zigzag: middle ground
    return 0.4 * n_in + 0.15 * pd + noise


def build_regions_flexible(active: List[Tuple[int, int]], rows: int, cols: int, difficulty: str, rng: random.Random,
                           max_retries: int = 80) -> Tuple[List[List[List[int]]], List[str]]:
    """Return region cell lists and region modes (for debugging)."""
    active_set = set(active)
    n_cells = len(active)

    for attempt in range(max_retries):
        R, sizes = sample_region_count_and_sizes(n_cells, difficulty, rng)
        strategy = choose_seed_strategy(rng)
        weights = choose_mode_weights(rng)
        modes = sample_modes(R, weights, rng)

        seeds = select_seeds(active, R, rows, cols, rng, strategy)

        # ensure uniqueness of seeds; if duplicates, resample by adding random jitter selection
        seen = set()
        uniq_seeds = []
        for s in seeds:
            if s not in seen:
                seen.add(s)
                uniq_seeds.append(s)
        while len(uniq_seeds) < R:
            cand = rng.choice(active)
            if cand not in seen:
                seen.add(cand)
                uniq_seeds.append(cand)
        seeds = uniq_seeds

        regions = [set([seeds[i]]) for i in range(R)]
        remaining_cap = [sizes[i] - 1 for i in range(R)]
        unassigned = set(active) - set(seeds)

        # frontiers
        frontiers: List[set] = []
        for i in range(R):
            fr = set()
            for nb in neighbors(seeds[i], rows, cols):
                if nb in unassigned:
                    fr.add(nb)
            frontiers.append(fr)

        stalled = 0
        while unassigned and stalled < 5000:
            # choose a region that still needs cells and has frontier
            candidates = [i for i in range(R) if remaining_cap[i] > 0 and frontiers[i]]
            if not candidates:
                # try to refresh frontiers for regions that still need cells
                refreshed = False
                for i in range(R):
                    if remaining_cap[i] <= 0:
                        continue
                    fr = set()
                    for cell in regions[i]:
                        for nb in neighbors(cell, rows, cols):
                            if nb in unassigned:
                                fr.add(nb)
                    frontiers[i] = fr
                    if fr:
                        refreshed = True
                if not refreshed:
                    break
                continue

            i = rng.choice(candidates)

            # score frontier candidates
            scored = []
            for cand in list(frontiers[i]):
                sc = score_candidate(modes[i], regions[i], cand, rows, cols, active_set, rng)
                scored.append((sc, cand))

            scored.sort(key=lambda x: x[0], reverse=True)

            # pick among top-k to keep variety
            k = min(5, len(scored))
            pick = rng.choice(scored[:k])[1]

            regions[i].add(pick)
            remaining_cap[i] -= 1
            unassigned.remove(pick)

            # update frontier i
            frontiers[i].discard(pick)
            for nb in neighbors(pick, rows, cols):
                if nb in unassigned:
                    frontiers[i].add(nb)

            # remove pick from other frontiers
            for j in range(R):
                if j != i:
                    frontiers[j].discard(pick)

            stalled += 1

        # success if all assigned and caps satisfied
        if not unassigned and all(rc == 0 for rc in remaining_cap):
            region_cells_ll = [[[r, c] for (r, c) in sorted(reg)] for reg in regions]
            return region_cells_ll, modes

    # Fallback: single generic grow (should rarely happen)
    region_cells_ll = [[[r, c] for (r, c) in active]]
    return region_cells_ll, ["fallback"]


# ----------------------------
# Cards / Animals
# ----------------------------

def legs_of(animal: Optional[str]) -> int:
    if animal is None:
        return 0
    for n, l in ANIMALS:
        if n == animal:
            return l
    return 0


def pick_animals_for_card(rng: random.Random, difficulty: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (a,b) where values are animal names or None (empty)."""
    # Keep empties lower on easy to reduce ambiguity
    empty_p = {"easy": 0.06, "medium": 0.06, "hard": 0.04}[difficulty]
    names = [a for a, _legs in ANIMALS]

    weights = []
    for n in names:
        if n in ("Dog", "Cat", "Cow", "Horse"):
            weights.append(0.85 if difficulty == "easy" else 0.9)
        elif n == "Chicken":
            weights.append(1.25)
        elif n == "Bee":
            weights.append(1.05)
        elif n == "Spider":
            weights.append(1.05 if difficulty == "hard" else 0.95)
        elif n == "Snail":
            weights.append(0.9)
        else:
            weights.append(1.0)

    def sample_one() -> Optional[str]:
        if rng.random() < empty_p:
            return None
        return rng.choices(names, weights=weights, k=1)[0]

    for _ in range(12):
        a, b = sample_one(), sample_one()
        if not (a is None and b is None):
            return a, b
    return sample_one(), sample_one()


def ensure_leg_diversity_easy(cards: List[Dict[str, Any]]) -> bool:
    leg_counts = set()
    for c in cards:
        for half in (c.get("a"), c.get("b")):
            l = legs_of(half)
            if l > 0:
                leg_counts.add(l)
    return len(leg_counts) >= 2


# ----------------------------
# Rules
# ----------------------------

def choose_op(rng: random.Random, difficulty: str) -> str:
    # lots of '=' for uniqueness, sprinkle < and > for variety
    if difficulty == "easy":
        return "=" if rng.random() < 0.90 else ("<" if rng.random() < 0.5 else ">")
    if difficulty == "medium":
        return "=" if rng.random() < 0.80 else ("<" if rng.random() < 0.5 else ">")
    return "=" if rng.random() < 0.75 else ("<" if rng.random() < 0.5 else ">")


def choose_rule_for_region(rng: random.Random, difficulty: str,
                           region_cells: List[Tuple[int, int]],
                           cell_animal: Dict[Tuple[int, int], Optional[str]]) -> Dict[str, Any]:
    animals = [cell_animal[c] for c in region_cells]
    legs_sum = sum(legs_of(a) for a in animals)
    animal_count = sum(1 for a in animals if a is not None)

    species = [a for a in animals if a is not None]
    has_empty = any(a is None for a in animals)

    strong_unique_ok = (len(region_cells) >= 3) and (not has_empty) and (len(species) == len(region_cells)) and (len(set(species)) == len(species))

    # Rule mix depends on difficulty and size (variety but still solvable)
    size = len(region_cells)

    if difficulty == "easy":
        choices = ["legs", "animals", "unique"]
        weights = [0.62, 0.34, 0.04]
    elif difficulty == "medium":
        choices = ["legs", "animals", "unique"]
        weights = [0.52, 0.33, 0.15]
    else:
        choices = ["unique", "legs", "animals"]
        weights = [0.48, 0.30, 0.22]

    if size <= 2:
        choices = ["legs", "animals"]
        weights = [0.65, 0.35]

    filtered, filtered_w = [], []
    for ch, w in zip(choices, weights):
        if ch == "unique" and not strong_unique_ok:
            continue
        filtered.append(ch)
        filtered_w.append(w)

    if not filtered:
        filtered, filtered_w = ["legs"], [1.0]

    rule_type = rng.choices(filtered, weights=filtered_w, k=1)[0]

    if rule_type == "legs":
        op = choose_op(rng, difficulty)
        # Keep the hidden solution valid:
        # - For '=', keep exact
        # - For '<' or '>', choose a bound that the solution satisfies but still constrains a bit
        if op == "=":
            val = legs_sum
        elif op == "<":
            val = max(legs_sum + 1, legs_sum + rng.randint(1, 4))
        else:
            val = max(0, legs_sum - rng.randint(1, 4))
        return {"type": "legs", "op": op, "value": int(val)}

    if rule_type == "animals":
        op = choose_op(rng, difficulty)
        if op == "=":
            val = animal_count
        elif op == "<":
            val = min(len(region_cells) + 1, animal_count + rng.randint(1, 2))
        else:
            val = max(0, animal_count - rng.randint(1, 2))
        return {"type": "animals", "op": op, "value": int(val)}

    return {"type": "uniqueSpecies"}


# ----------------------------
# Candidate generation
# ----------------------------

def generate_candidate(date_utc: str, difficulty: str, attempt: int) -> Optional[Dict[str, Any]]:
    spec = SPECS[difficulty]
    rng = random.Random(build_seed_int(date_utc, difficulty, attempt))

    rows, cols = spec.rows, spec.cols
    active_ll = make_simple_active_shape(difficulty, rows, cols)
    active = [tuple(x) for x in active_ll]

    blocked = make_blockers(rows, cols, active_ll, rng)
    pairs = bipartite_matching_tiling(active)
    if not pairs:
        return None

    # Reduce duplicates to decrease symmetry
    max_dupes = {"easy": 1, "medium": 1, "hard": 1}[difficulty]
    pair_counts: Dict[Tuple[str, str], int] = {}

    fences = [FENCE_COLORS[i % 3] for i in range(spec.cards)]
    rng.shuffle(fences)

    # Build cards; retry a few times for easy leg diversity
    for _deck_try in range(12):
        cards = []
        pair_counts.clear()

        for i in range(spec.cards):
            for _ in range(80):
                a, b = pick_animals_for_card(rng, difficulty)
                key = tuple(sorted([(a or "_"), (b or "_")]))
                pair_counts[key] = pair_counts.get(key, 0) + 1
                if pair_counts[key] <= max_dupes:
                    cards.append({"id": f"C{i+1:02d}", "fence": fences[i], "a": a, "b": b})
                    break
                pair_counts[key] -= 1
            else:
                cards.append({"id": f"C{i+1:02d}", "fence": fences[i], "a": "Chicken", "b": None})

        if difficulty != "easy":
            break
        if ensure_leg_diversity_easy(cards):
            break

    if difficulty == "easy" and not ensure_leg_diversity_easy(cards):
        return None

    # Hidden assignment: assign cards onto a domino tiling
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

    # Flexible regions (ignore spec.regions here)
    region_cells_ll, region_modes = build_regions_flexible(active, rows, cols, difficulty, rng)

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
        "_internal": {
            "seed": build_seed_str(date_utc, difficulty),
            "attempt": attempt,
            "regionModes": region_modes,
        },
    }


# ----------------------------
# generate_unique (DEV FAST SAFE)
# ----------------------------

def generate_unique(date_utc: str, difficulty: str) -> Dict[str, Any]:
    """DEV FAST SAFE

    - Gate uses stop_at=2
    - Diagnostic pass (stop_at=3) for non-unique cases
    """
    spec = SPECS[difficulty]

    attempt_caps = {"easy": 180, "medium": 60, "hard": 80}
    timeout_caps = {"easy": 12, "medium": 8, "hard": 12}

    max_attempts = attempt_caps.get(difficulty, 60)
    max_timeouts = timeout_caps.get(difficulty, 6)

    diag_enabled = os.getenv("SOLVER_DIAG_ENABLED", "true").lower() == "true"
    diag_time_limit = float(os.getenv("SOLVER_DIAG_TIME_LIMIT_SEC", "2.0"))

    best_solvable = None
    best_stats = None
    best_score = None

    timeouts = 0

    for attempt in range(max_attempts):
        if attempt % 10 == 0:
            print(f"[TRY] date={date_utc} diff={difficulty} attempt={attempt}/{max_attempts} timeouts={timeouts}", flush=True)

        puzzle = generate_candidate(date_utc, difficulty, attempt)
        if puzzle is None:
            continue

        solutions2, stats2 = solve_count(puzzle, stop_at=2)

        solver_diag = None
        if diag_enabled and solutions2 >= 2:
            solutions3, stats3 = solve_count(
                puzzle,
                stop_at=3,
                time_limit_override_sec=diag_time_limit,
                verbose_override=False,
                progress_every_override=10**9,
            )
            solver_diag = {
                "stopAt": 3,
                "solutionsFound": solutions3,
                "timedOut": stats3.get("timedOut"),
                "timeMs": stats3.get("timeMs"),
            }
        puzzle["_internal"]["solverDiag"] = solver_diag

        if stats2.get("timedOut"):
            timeouts += 1
            if timeouts >= max_timeouts and best_solvable is not None:
                print(f"[TRY] date={date_utc} diff={difficulty} too many timeouts -> fallback", flush=True)
                break

        if solutions2 <= 0:
            continue

        score2 = compute_hardness(difficulty, spec.cards, stats2)

        if best_solvable is None:
            best_solvable, best_stats, best_score = puzzle, stats2, score2

        if solutions2 == 1:
            puzzle["_internal"]["uniqueSolution"] = True
            puzzle["_internal"]["solverStats"] = stats2
            puzzle["_internal"]["difficultyScore"] = score2
            return puzzle

    if best_solvable is None:
        print(f"[TRY] date={date_utc} diff={difficulty} no solvable candidate found -> forced fallback", flush=True)
        # try a few attempts to get at least something solvable
        for a in range(5):
            best_solvable = generate_candidate(date_utc, difficulty, a)
            if best_solvable:
                break
        if best_solvable is None:
            raise RuntimeError(f"Could not generate any candidate for {date_utc} {difficulty}")
        best_stats = {"type": "CSP-backtracking", "stopAt": 2, "solutionsFound": 0, "nodesVisited": 0, "backtracks": 0, "maxDepth": 0, "timeMs": 0, "timedOut": False}
        best_score = {"hardness01": 0.0, "passedBand": False}

    best_solvable["_internal"]["uniqueSolution"] = False
    best_solvable["_internal"]["solverStats"] = best_stats
    best_solvable["_internal"]["difficultyScore"] = best_score
    return best_solvable


# ----------------------------
# meta
# ----------------------------

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
        diag = puzzle["_internal"].get("solverDiag")
        unique_flag = puzzle["_internal"].get("uniqueSolution", False)

        public_puzzle = {k: v for k, v in puzzle.items() if k != "_internal"}
        puzzle_hash = "sha256:" + sha256_hex(json.dumps(public_puzzle, sort_keys=True))

        meta["difficulties"][diff] = {
            "grid": {
                "rows": public_puzzle["grid"]["rows"],
                "cols": public_puzzle["grid"]["cols"],
                "activeCells": len(public_puzzle["grid"]["activeCellsCoords"]),
                "cards": spec.cards,
                "regions": len(public_puzzle.get("regions", [])),
            },
            "seed": puzzle["_internal"].get("seed"),
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
                "diagnostic": diag,
            },
            "difficultyScore": score,
            "hash": {"puzzleJson": puzzle_hash},
        }

    return meta
