import os
import time
from typing import Dict, Any, Tuple, List, Optional

import networkx as nx

from .config import ANIMALS

SPECIES = [name for name, _legs in ANIMALS]
LEGS = {name: legs for name, legs in ANIMALS}
SPECIES_ID = {name: i for i, name in enumerate(SPECIES)}


def _feasible_domino_tiling(active_cells: List[Tuple[int, int]]) -> bool:
    n = len(active_cells)
    if n % 2 == 1:
        return False

    U = [cell for cell in active_cells if (cell[0] + cell[1]) % 2 == 0]
    V = [cell for cell in active_cells if (cell[0] + cell[1]) % 2 == 1]
    if len(U) != len(V):
        return False

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
    matched = sum(1 for u in U if u in matching)
    return matched == len(U)


def _domino_edges(active_cells: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], List[List[int]]]:
    cell_to_idx = {cell: i for i, cell in enumerate(active_cells)}
    edges: List[Tuple[int, int]] = []
    edges_by_cell: List[List[int]] = [[] for _ in range(len(active_cells))]

    for (r, c), i in cell_to_idx.items():
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in cell_to_idx:
                j = cell_to_idx[nb]
                if i < j:
                    eid = len(edges)
                    edges.append((i, j))
                    edges_by_cell[i].append(eid)
                    edges_by_cell[j].append(eid)

    return edges, edges_by_cell


def _legs_of(animal: Optional[str]) -> int:
    if animal is None:
        return 0
    return LEGS.get(animal, 0)


def _parse_rule(rule: Dict[str, Any]) -> Tuple[str, Any]:
    t = (rule or {}).get("type")
    if t == "legs":
        return "legs", (rule.get("op"), int(rule.get("value")))
    if t == "animals":
        return "animals", (rule.get("op"), int(rule.get("value")))
    if t == "uniqueSpecies":
        return "unique", None
    if t == "onlySpecies":
        return "only", rule.get("species")
    return "none", None


def solve_count(
    puzzle: Dict[str, Any],
    stop_at: int = 2,
    time_limit_override_sec: Optional[float] = None,
    verbose_override: Optional[bool] = None,
    progress_every_override: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    """CSP solver with Stop-at-N and optional time-limit override.

    - stop_at=2 is best practice for uniqueness gating (as soon as 2 solutions exist => not unique).
    - time_limit_override_sec can be used for cheap diagnostic second pass (e.g., stop_at=3, time_limit=2s).

    Env vars (defaults):
      SOLVER_VERBOSE=true|false
      SOLVER_PROGRESS_EVERY=20000
      SOLVER_TIME_LIMIT_EASY=5
      SOLVER_TIME_LIMIT_MEDIUM=15
      SOLVER_TIME_LIMIT_HARD=25
    """
    t0 = time.time()

    diff = puzzle.get("difficulty", "medium")

    env_verbose = os.getenv("SOLVER_VERBOSE", "false").lower() == "true"
    verbose = env_verbose if verbose_override is None else bool(verbose_override)

    env_progress_every = int(os.getenv("SOLVER_PROGRESS_EVERY", "20000"))
    progress_every = env_progress_every if progress_every_override is None else int(progress_every_override)

    default_limits = {
        "easy": float(os.getenv("SOLVER_TIME_LIMIT_EASY", "5")),
        "medium": float(os.getenv("SOLVER_TIME_LIMIT_MEDIUM", "15")),
        "hard": float(os.getenv("SOLVER_TIME_LIMIT_HARD", "25")),
    }
    time_limit = default_limits.get(diff, 15.0) if time_limit_override_sec is None else float(time_limit_override_sec)

    grid = puzzle.get("grid", {})
    active_ll = grid.get("activeCellsCoords", [])
    active_cells: List[Tuple[int, int]] = [tuple(x) for x in active_ll]

    if not active_cells:
        return 0, {
            "type": "CSP-backtracking",
            "stopAt": stop_at,
            "solutionsFound": 0,
            "nodesVisited": 0,
            "backtracks": 0,
            "maxDepth": 0,
            "timeMs": int((time.time() - t0) * 1000),
            "timedOut": False,
            "timeLimitSec": time_limit,
        }

    if not _feasible_domino_tiling(active_cells):
        return 0, {
            "type": "CSP-backtracking",
            "stopAt": stop_at,
            "solutionsFound": 0,
            "nodesVisited": 0,
            "backtracks": 0,
            "maxDepth": 0,
            "timeMs": int((time.time() - t0) * 1000),
            "timedOut": False,
            "timeLimitSec": time_limit,
        }

    n_cells = len(active_cells)
    cards = puzzle.get("cards", [])
    n_cards = len(cards)

    edges, edges_by_cell = _domino_edges(active_cells)

    regions = puzzle.get("regions", [])
    R = len(regions)
    cell_to_idx = {cell: i for i, cell in enumerate(active_cells)}

    region_of_cell = [-1] * n_cells
    region_cells: List[List[int]] = [[] for _ in range(R)]
    rule_kind = ["none"] * R
    rule_payload = [None] * R

    for rid, reg in enumerate(regions):
        for rc in reg.get("cells", []):
            idx = cell_to_idx[tuple(rc)]
            region_of_cell[idx] = rid
            region_cells[rid].append(idx)
        k, p = _parse_rule(reg.get("rule", {}) or {})
        rule_kind[rid] = k
        rule_payload[rid] = p

    region_size = [len(region_cells[r]) for r in range(R)]
    card_halves: List[Tuple[Optional[str], Optional[str]]] = [(c.get("a"), c.get("b")) for c in cards]

    occupied = [False] * n_cells
    animal_at: List[Optional[str]] = [None] * n_cells
    used_card = [False] * n_cards

    sum_legs = [0] * R
    count_animals = [0] * R
    remaining = region_size[:]
    species_counts = [[0] * len(SPECIES) for _ in range(R)]

    nodes_visited = 0
    backtracks = 0
    max_depth = 0
    solutions = 0
    timed_out = False

    def apply_cell(ci: int, animal: Optional[str]):
        rid = region_of_cell[ci]
        if rid == -1:
            return
        remaining[rid] -= 1
        sum_legs[rid] += _legs_of(animal)
        if animal is not None:
            count_animals[rid] += 1
            sid = SPECIES_ID.get(animal)
            if sid is not None:
                species_counts[rid][sid] += 1

    def undo_cell(ci: int, animal: Optional[str]):
        rid = region_of_cell[ci]
        if rid == -1:
            return
        remaining[rid] += 1
        sum_legs[rid] -= _legs_of(animal)
        if animal is not None:
            count_animals[rid] -= 1
            sid = SPECIES_ID.get(animal)
            if sid is not None:
                species_counts[rid][sid] -= 1

    def can_place(ci: int, animal: Optional[str]) -> bool:
        rid = region_of_cell[ci]
        if rid == -1:
            return True
        k = rule_kind[rid]
        p = rule_payload[rid]
        if k == "only":
            return animal is not None and animal == p
        if k == "unique":
            if animal is None:
                return True
            sid = SPECIES_ID.get(animal)
            return sid is None or species_counts[rid][sid] == 0
        return True

    def region_possible(rid: int) -> bool:
        k = rule_kind[rid]
        p = rule_payload[rid]
        rem = remaining[rid]
        if k == "none":
            return True
        if k == "unique":
            return all(cnt <= 1 for cnt in species_counts[rid])
        if k == "only":
            target = p
            for cell in region_cells[rid]:
                if occupied[cell] and animal_at[cell] != target:
                    return False
            return True
        if k == "legs":
            op, val = p
            cur = sum_legs[rid]
            min_possible = cur
            max_possible = cur + 8 * rem
            if op == "=":
                return min_possible <= val <= max_possible
            if op == "<":
                return min_possible < val
            if op == ">":
                return max_possible > val
            return True
        if k == "animals":
            op, val = p
            cur = count_animals[rid]
            min_possible = cur
            max_possible = cur + rem
            if op == "=":
                return min_possible <= val <= max_possible
            if op == "<":
                return cur < val
            if op == ">":
                return max_possible > val
            return True
        return True

    def all_regions_possible() -> bool:
        return all(region_possible(r) for r in range(R))

    def final_check() -> bool:
        for rid in range(R):
            k = rule_kind[rid]
            p = rule_payload[rid]
            if k == "none":
                continue
            if k == "unique":
                if any(cnt > 1 for cnt in species_counts[rid]):
                    return False
            elif k == "only":
                target = p
                for cell in region_cells[rid]:
                    if animal_at[cell] != target:
                        return False
            elif k == "legs":
                op, val = p
                cur = sum_legs[rid]
                if op == "=" and cur != val:
                    return False
                if op == "<" and not (cur < val):
                    return False
                if op == ">" and not (cur > val):
                    return False
            elif k == "animals":
                op, val = p
                cur = count_animals[rid]
                if op == "=" and cur != val:
                    return False
                if op == "<" and not (cur < val):
                    return False
                if op == ">" and not (cur > val):
                    return False
        return True

    def pick_next_cell() -> int:
        best = -1
        best_options = 10**9
        for ci in range(n_cells):
            if occupied[ci]:
                continue
            opt = 0
            for eid in edges_by_cell[ci]:
                u, v = edges[eid]
                other = v if u == ci else u
                if not occupied[other]:
                    opt += 1
            if opt < best_options:
                best_options = opt
                best = ci
            if best_options == 0:
                break
        return best

    def dfs(depth: int):
        nonlocal nodes_visited, backtracks, max_depth, solutions, timed_out

        if (time.time() - t0) > time_limit:
            timed_out = True
            return

        if solutions >= stop_at:
            return

        nodes_visited += 1
        if verbose and nodes_visited % progress_every == 0:
            elapsed = time.time() - t0
            print(f"[SOLVER] diff={diff} nodes={nodes_visited} depth={depth} sol={solutions} t={elapsed:.1f}s", flush=True)

        max_depth = max(max_depth, depth)

        if all(occupied):
            if final_check():
                solutions += 1
            return

        ci = pick_next_cell()
        if ci == -1:
            if final_check():
                solutions += 1
            return

        any_branch = False

        for eid in edges_by_cell[ci]:
            u, v = edges[eid]
            other = v if u == ci else u
            if occupied[other]:
                continue

            for card_idx in range(n_cards):
                if used_card[card_idx]:
                    continue

                a, b = card_halves[card_idx]
                flips = [(a, b)]
                if a != b:
                    flips.append((b, a))

                for left, right in flips:
                    if not can_place(ci, left):
                        continue
                    if not can_place(other, right):
                        continue

                    used_card[card_idx] = True
                    occupied[ci] = occupied[other] = True
                    animal_at[ci] = left
                    animal_at[other] = right

                    apply_cell(ci, left)
                    apply_cell(other, right)

                    if all_regions_possible():
                        any_branch = True
                        dfs(depth + 1)
                        if timed_out or solutions >= stop_at:
                            undo_cell(ci, left)
                            undo_cell(other, right)
                            occupied[ci] = occupied[other] = False
                            animal_at[ci] = animal_at[other] = None
                            used_card[card_idx] = False
                            return

                    undo_cell(ci, left)
                    undo_cell(other, right)
                    occupied[ci] = occupied[other] = False
                    animal_at[ci] = animal_at[other] = None
                    used_card[card_idx] = False

        if not any_branch:
            backtracks += 1

    if verbose:
        print(f"[SOLVER] start diff={diff} timeLimit={time_limit}s", flush=True)

    dfs(0)

    sf = solutions if solutions < stop_at else stop_at
    # if timed out and didn't reach 2, treat as non-unique
    if timed_out and sf < 2:
        sf = min(stop_at, 2)

    stats = {
        "type": "CSP-backtracking",
        "stopAt": stop_at,
        "solutionsFound": sf,
        "nodesVisited": nodes_visited,
        "backtracks": backtracks,
        "maxDepth": max_depth,
        "timeMs": int((time.time() - t0) * 1000),
        "timedOut": timed_out,
        "timeLimitSec": time_limit,
    }

    return sf, stats
