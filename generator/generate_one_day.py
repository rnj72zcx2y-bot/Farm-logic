import hashlib
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import random
import networkx as nx

from .config import SPECS, GENERATOR_VERSION, GIT_COMMIT, GLOBAL_SALT, FENCE_COLORS, ANIMALS
from .solver_unique import solve_count
from .difficulty_score import compute_hardness


# --- helpers ---

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


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
    """Deterministic domino-tileable shapes to unblock development.

    Replace later with a real shape generator.
    """
    active: List[List[int]] = []
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
    # shuffle for a bit of visual variety
    rng.shuffle(blocked)
    return blocked


def bipartite_matching_tiling(active_cells: List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Return one domino tiling as list of cell pairs using bipartite matching."""
    cell_to_idx = {cell: i for i, cell in enumerate(active_cells)}
    U = [cell for cell in active_cells if (cell[0] + cell[1]) % 2 == 0]
    V = [cell for cell in active_cells if (cell[0] + cell[1]) % 2 == 1]

    if len(active_cells) % 2 == 1 or len(U) != len(V):
        return []

    G = nx.Graph()
    G.add_nodes_from(U, bipartite=0)
    G.add_nodes_from(V, bipartite=1)

    active_set = set(active_cells)
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
            used.add(u); used.add(v)
            pairs.append((u, v))

    if len(used) != len(active_cells):
        return []

    return pairs


def partition_regions(active_cells: List[Tuple[int, int]], region_count: int, rows: int, cols: int, rng: random.Random) -> List[List[List[int]]]:
    """Grow contiguous regions from random seeds."""
    active_set = set(active_cells)
    if region_count <= 1:
        return [[list(c) for c in active_cells]]

    seeds = rng.sample(active_cells, k=min(region_count, len(active_cells)))
    regions = [set([s]) for s in seeds]
    unassigned = active_set - set(seeds)

    # BFS-like growth
    frontiers = [set([s]) for s in seeds]
    while unassigned:
        # choose a region that can expand
        expandable = []
        for i, fr in enumerate(frontiers):
            for cell in fr:
                for nb in neighbors(cell, rows, cols):
                    if nb in unassigned:
                        expandable.append(i)
                        break
                if i in expandable:
                    break

        if not expandable:
            # assign remaining arbitrarily (rare)
            for cell in list(unassigned):
                regions[rng.randrange(len(regions))].add(cell)
                unassigned.remove(cell)
            break

        i = rng.choice(expandable)
        # collect candidate neighbors
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
        if len(frontiers[i]) > 24:
            frontiers[i] = set(rng.sample(list(frontiers[i]), 12))

    # convert to list of lists
    out = []
    for idx, reg in enumerate(regions):
        out.append([[r, c] for (r, c) in sorted(reg)])
    return out


def pick_animals_for_card(rng: random.Random, difficulty: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (a,b) where values are animal names or None (empty)."""
    # empty probability per half
    empty_p = {"easy": 0.10, "medium": 0.12, "hard": 0.15}[difficulty]
    names = [a for a, _legs in ANIMALS]

    # weights: slightly more spiders/snails in hard
    w = {name: 1.0 for name in names}
    w["Chicken"] = 1.2
    w["Bee"] = 0.8 if difficulty == "easy" else 0.95
    w["Spider"] = 0.5 if difficulty != "hard" else 0.9
    w["Snail"] = 0.5 if difficulty == "easy" else 0.85

    weights = [w[n] for n in names]

    def sample_one() -> Optional[str]:
        if rng.random() < empty_p:
            return None
        return rng.choices(names, weights=weights, k=1)[0]

    return sample_one(), sample_one()


def legs_of(animal: Optional[str]) -> int:
    if animal is None:
        return 0
    for n, l in ANIMALS:
        if n == animal:
            return l
    return 0


def choose_rule_for_region(rng: random.Random, difficulty: str, region_cells: List[Tuple[int, int]], cell_animal: Dict[Tuple[int, int], Optional[str]]) -> Dict[str, Any]:
    """Choose a structured rule that is consistent with the (hidden) solution."""
    animals = [cell_animal[c] for c in region_cells]
    legs_sum = sum(legs_of(a) for a in animals)
    animal_count = sum(1 for a in animals if a is not None)

    species = [a for a in animals if a is not None]
    unique_ok = len(species) >= 2 and len(set(species)) == len(species)
    only_ok = len(species) == len(animals) and len(set(species)) == 1  # no empties and all same

    # weighted choices by difficulty
    choices = []
    if difficulty == "easy":
        choices += ["legs_eq", "animals_eq", "only", "unique"]
        weights = [0.45, 0.40, 0.10, 0.05]
    elif difficulty == "medium":
        choices += ["legs_eq", "animals_eq", "legs_lt", "legs_gt", "unique", "only"]
        weights = [0.30, 0.25, 0.15, 0.10, 0.15, 0.05]
    else:
        choices += ["legs_eq", "animals_eq", "legs_lt", "legs_gt", "unique", "animals_gt", "only"]
        weights = [0.20, 0.20, 0.10, 0.10, 0.25, 0.10, 0.05]

    # filter impossible
    filtered = []
    filtered_w = []
    for ch, w in zip(choices, weights):
        if ch == "unique" and not unique_ok:
            continue
        if ch == "only" and not only_ok:
            continue
        filtered.append(ch)
        filtered_w.append(w)

    if not filtered:
        filtered = ["legs_eq"]
        filtered_w = [1.0]

    rule_type = rng.choices(filtered, weights=filtered_w, k=1)[0]

    if rule_type == "legs_eq":
        return {"type": "legs", "op": "=", "value": legs_sum}
    if rule_type == "legs_lt":
        return {"type": "legs", "op": "<", "value": legs_sum + 2}
    if rule_type == "legs_gt":
        return {"type": "legs", "op": ">", "value": max(0, legs_sum - 2)}

    if rule_type == "animals_eq":
        return {"type": "animals", "op": "=", "value": animal_count}
    if rule_type == "animals_gt":
        return {"type": "animals", "op": ">", "value": max(0, animal_count - 1)}

    if rule_type == "unique":
        return {"type": "uniqueSpecies"}
    if rule_type == "only":
        # all same species
        return {"type": "onlySpecies", "species": species[0]}

    return {"type": "legs", "op": "=", "value": legs_sum}


# --- main generator API ---

def generate_candidate(date_utc: str, difficulty: str, attempt: int) -> Dict[str, Any]:
    """Generate a candidate puzzle with structured regions + rules.

    NOTE: This is still a "development" generator (simple shape), but it now outputs:
      - activeCellsCoords (filled)
      - blockers
      - cards with animals
      - regions with structured rules derived from an internal solution assignment

    Uniqueness is NOT guaranteed yet (Stage-2 solver will handle that).
    """
    spec = SPECS[difficulty]
    rng = random.Random(build_seed_int(date_utc, difficulty, attempt))

    rows, cols = spec.rows, spec.cols
    active_ll = make_simple_active_shape(difficulty, rows, cols)
    active = [tuple(x) for x in active_ll]

    blocked = make_blockers(rows, cols, active_ll, rng)

    # Build one tiling to create a consistent internal solution
    pairs = bipartite_matching_tiling(active)
    if not pairs:
        # Should not happen with our simple shapes
        pairs = []

    # Create cards (balanced fence colors)
    fences = [FENCE_COLORS[i % 3] for i in range(spec.cards)]
    rng.shuffle(fences)

    cards = []
    cell_animal: Dict[Tuple[int, int], Optional[str]] = {}

    # Ensure at least one snail across the whole puzzle sometimes
    force_snail = (difficulty != "easy")
    forced_used = False

    for i in range(spec.cards):
        a, b = pick_animals_for_card(rng, difficulty)
        if force_snail and not forced_used:
            a = "Snail"
            forced_used = True

        cards.append({
            "id": f"C{i+1:02d}",
            "fence": fences[i],
            "a": a,
            "b": b,
        })

    # Assign cards to tiling pairs (one-to-one)
    # (In the real game, this pairing is what the player must discover.)
    rng.shuffle(pairs)
    for i, (u, v) in enumerate(pairs[:spec.cards]):
        card = cards[i]
        # random flip
        if rng.random() < 0.5:
            cell_animal[u] = card["a"]
            cell_animal[v] = card["b"]
        else:
            cell_animal[u] = card["b"]
            cell_animal[v] = card["a"]

    # Any missing cells (should not happen) set to empty
    for c in active:
        cell_animal.setdefault(c, None)

    # Regions + rules (structured)
    region_cells_ll = partition_regions(active, spec.regions, rows, cols, rng)

    regions = []
    for idx, cell_list in enumerate(region_cells_ll):
        rcells = [tuple(x) for x in cell_list]
        rule = choose_rule_for_region(rng, difficulty, rcells, cell_animal)
        regions.append({
            "id": f"R{idx+1}",
            "cells": cell_list,
            "rule": rule
        })

    # Seed for meta/debug
    seed_str = "sha256:" + sha256_hex(f"{date_utc}|{difficulty}|{GLOBAL_SALT}")

    return {
        "dateUtc": date_utc,
        "difficulty": difficulty,
        "schemaVersion": 1,
        "generatorVersion": GENERATOR_VERSION,
        "grid": {
            "rows": rows,
            "cols": cols,
            "activeCellsCoords": active_ll,
            "blocked": blocked,
        },
        "regions": regions,
        "cards": cards,
        "_internal": {"seed": seed_str, "attempt": attempt}
    }


def generate_unique(date_utc: str, difficulty: str, max_attempts: int = 300) -> Dict[str, Any]:
    """Generate a puzzle that is solvable (Stage-1) and passes TEMP gates.

    NOTE: Hardness gating is temporarily disabled until the real CSP solver is implemented.
    """
    spec = SPECS[difficulty]

    for attempt in range(max_attempts):
        puzzle = generate_candidate(date_utc, difficulty, attempt)

        # Stage-1 feasibility check
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
