import time
from typing import Dict, Any, Tuple, List, Optional

import networkx as nx

from .config import ANIMALS

# Map species name -> legs, and index for bitmask
SPECIES = [name for name, _legs in ANIMALS]
LEGS = {name: legs for name, legs in ANIMALS}
SPECIES_ID = {name: i for i, name in enumerate(SPECIES)}
EMPTY = None  # empty grass in JSON


def _domino_edges(active_cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Return list of edges (i,j) between adjacent active cells by index."""
    cell_to_idx = {cell: i for i, cell in enumerate(active_cells)}
    edges = []
    for (r, c), i in cell_to_idx.items():
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in cell_to_idx:
                j = cell_to_idx[nb]
                if i < j:
                    edges.append((i, j))
    return edges


def _feasible_domino_tiling(active_cells: List[Tuple[int, int]]) -> bool:
    """Quick feasibility: perfect matching exists."""
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


def _legs_of(animal: Optional[str]) -> int:
    if animal is None:
        return 0
    return LEGS.get(animal, 0)


def _is_animal(animal: Optional[str]) -> bool:
    return animal is not None


def _parse_rule(rule: Dict[str, Any]) -> Tuple[str, Any]:
    """Normalize rule dict into a (kind, payload) tuple."""
    t = rule.get("type")
    if t == "legs":
        return "legs", (rule.get("op"), int(rule.get("value")))
    if t == "animals":
        return "animals", (rule.get("op"), int(rule.get("value")))
    if t == "uniqueSpecies":
        return "unique", None
    if t == "onlySpecies":
        return "only", rule.get("species")
    # Unknown / empty rule
    return "none", None


def solve_count(puzzle: Dict[str, Any], stop_at: int = 2) -> Tuple[int, Dict[str, Any]]:
    """Stage-2 CSP solver: counts solutions up to stop_at (Stop-at-2).

    - Variables: assign each card to a domino edge + flip
    - Constraints: regions rules (legs/animals/unique/only) evaluated per CELL
    - Early stop after 2 solutions

    Returns (solutionsFound, stats)
      solutionsFound in {0,1,2} where 2 means 2+ found.
    """
    t0 = time.time()

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
        }

    # Quick feasibility
    if not _feasible_domino_tiling(active_cells):
        return 0, {
            "type": "CSP-backtracking",
            "stopAt": stop_at,
            "solutionsFound": 0,
            "nodesVisited": 0,
            "backtracks": 0,
            "maxDepth": 0,
            "timeMs": int((time.time() - t0) * 1000),
        }

    n_cells = len(active_cells)
    n_cards = len(puzzle.get("cards", []))

    # Map cell coords to index
    cell_to_idx = {cell: i for i, cell in enumerate(active_cells)}
    idx_to_cell = active_cells

    # Edges
    edges = _domino_edges(active_cells)
    edges_by_cell: List[List[int]] = [[] for _ in range(n_cells)]
    for eid, (u, v) in enumerate(edges):
        edges_by_cell[u].append(eid)
        edges_by_cell[v].append(eid)

    # Regions
    regions = puzzle.get("regions", [])
    R = len(regions)
    region_of_cell = [-1] * n_cells
    region_cells: List[List[int]] = [[] for _ in range(R)]
    region_rule_kind: List[str] = ["none"] * R
    region_rule_payload: List[Any] = [None] * R

    for rid, reg in enumerate(regions):
        # cells
        for rc in reg.get("cells", []):
            ci = cell_to_idx[tuple(rc)]
            region_of_cell[ci] = rid
            region_cells[rid].append(ci)
        # rule
        kind, payload = _parse_rule(reg.get("rule", {}) or {})
        region_rule_kind[rid] = kind
        region_rule_payload[rid] = payload

    region_size = [len(region_cells[r]) for r in range(R)]

    # Card halves (a,b)
    cards = puzzle.get("cards", [])
    card_halves: List[Tuple[Optional[str], Optional[str]]] = []
    for c in cards:
        card_halves.append((c.get("a"), c.get("b")))

    # --- CSP State ---
    occupied = [False] * n_cells
    animal_at: List[Optional[str]] = [None] * n_cells
    used_card = [False] * n_cards

    # Region trackers
    sum_legs = [0] * R
    count_animals = [0] * R
    remaining = region_size[:]

    # species counts per region (8 species)
    species_counts = [[0] * len(SPECIES) for _ in range(R)]

    # stats
    nodes_visited = 0
    backtracks = 0
    max_depth = 0
    solutions = 0

    def apply_cell(ci: int, animal: Optional[str]):
        rid = region_of_cell[ci]
        if rid == -1:
            return
        remaining[rid] -= 1
        sum_legs[rid] += _legs_of(animal)
        if _is_animal(animal):
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
        if _is_animal(animal):
            count_animals[rid] -= 1
            sid = SPECIES_ID.get(animal)
            if sid is not None:
                species_counts[rid][sid] -= 1

    def region_possible(rid: int) -> bool:
        kind = region_rule_kind[rid]
        payload = region_rule_payload[rid]
        rem = remaining[rid]

        # No rule
        if kind == "none":
            return True

        # Unique species (ignores EMPTY)
        if kind == "unique":
            # If any species count > 1 => fail
            for cnt in species_counts[rid]:
                if cnt > 1:
                    return False
            return True

        # Only species
        if kind == "only":
            target = payload
            # No empty allowed, all placed must be target
            for cell in region_cells[rid]:
                if occupied[cell]:
                    if animal_at[cell] != target:
                        return False
            # Also: cannot end with empties, but since we enforce at placement, bounds are enough
            return True

        # Legs bounds
        if kind == "legs":
            op, val = payload
            cur = sum_legs[rid]

            # bounds depend on only rule: but if kind is legs, we use generic bounds
            min_possible = cur + 0 * rem
            max_possible = cur + 8 * rem

            if op == "=":
                return min_possible <= val <= max_possible
            if op == "<":
                return min_possible < val
            if op == ">":
                return max_possible > val
            return True

        # Animals bounds
        if kind == "animals":
            op, val = payload
            cur = count_animals[rid]

            # generic bounds (empty allowed)
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
        for rid in range(R):
            if not region_possible(rid):
                return False
        return True

    def can_place_in_cell(ci: int, animal: Optional[str]) -> bool:
        rid = region_of_cell[ci]
        if rid == -1:
            return True
        kind = region_rule_kind[rid]
        payload = region_rule_payload[rid]

        if kind == "only":
            # empty forbidden
            return animal is not None and animal == payload

        if kind == "unique":
            if animal is None:
                return True
            sid = SPECIES_ID.get(animal)
            if sid is None:
                return True
            return species_counts[rid][sid] == 0

        return True

    def is_complete_solution() -> bool:
        # All cells occupied
        if not all(occupied):
            return False
        # Final region check (exact)
        for rid in range(R):
            kind = region_rule_kind[rid]
            payload = region_rule_payload[rid]

            if kind == "none":
                continue
            if kind == "unique":
                for cnt in species_counts[rid]:
                    if cnt > 1:
                        return False
            elif kind == "only":
                target = payload
                for cell in region_cells[rid]:
                    if animal_at[cell] != target:
                        return False
            elif kind == "legs":
                op, val = payload
                cur = sum_legs[rid]
                if op == "=" and cur != val:
                    return False
                if op == "<" and not (cur < val):
                    return False
                if op == ">" and not (cur > val):
                    return False
            elif kind == "animals":
                op, val = payload
                cur = count_animals[rid]
                if op == "=" and cur != val:
                    return False
                if op == "<" and not (cur < val):
                    return False
                if op == ">" and not (cur > val):
                    return False
        return True

    def pick_next_cell() -> int:
        # MRV-lite: choose free cell with fewest free-neighbor edges
        best = -1
        best_options = 10**9
        for ci in range(n_cells):
            if occupied[ci]:
                continue
            # count free neighbors
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
        nonlocal nodes_visited, backtracks, max_depth, solutions

        if solutions >= stop_at:
            return

        nodes_visited += 1
        if depth > max_depth:
            max_depth = depth

        # If all cards placed or all cells occupied, check solution
        if all(occupied):
            if is_complete_solution():
                solutions += 1
            return

        ci = pick_next_cell()
        if ci == -1:
            if is_complete_solution():
                solutions += 1
            return

        any_branch = False

        # For each adjacent free cell
        for eid in edges_by_cell[ci]:
            u, v = edges[eid]
            other = v if u == ci else u
            if occupied[other]:
                continue

            # Try each unused card
            for card_idx in range(n_cards):
                if used_card[card_idx]:
                    continue

                a, b = card_halves[card_idx]

                # Try flips
                flips = [(a, b)]
                if a != b:
                    flips.append((b, a))

                for left, right in flips:
                    # Check local placement constraints
                    if not can_place_in_cell(ci, left):
                        continue
                    if not can_place_in_cell(other, right):
                        continue

                    # Apply
                    used_card[card_idx] = True
                    occupied[ci] = True
                    occupied[other] = True
                    animal_at[ci] = left
                    animal_at[other] = right

                    apply_cell(ci, left)
                    apply_cell(other, right)

                    if all_regions_possible():
                        any_branch = True
                        dfs(depth + 1)
                        if solutions >= stop_at:
                            # Undo and early stop
                            undo_cell(ci, left)
                            undo_cell(other, right)
                            occupied[ci] = False
                            occupied[other] = False
                            animal_at[ci] = None
                            animal_at[other] = None
                            used_card[card_idx] = False
                            return

                    # Undo
                    undo_cell(ci, left)
                    undo_cell(other, right)
                    occupied[ci] = False
                    occupied[other] = False
                    animal_at[ci] = None
                    animal_at[other] = None
                    used_card[card_idx] = False

        if not any_branch:
            backtracks += 1

    dfs(0)

    # solutions is exact up to stop_at
    sf = solutions if solutions < stop_at else stop_at

    stats = {
        "type": "CSP-backtracking",
        "stopAt": stop_at,
        "solutionsFound": sf,
        "nodesVisited": nodes_visited,
        "backtracks": backtracks,
        "maxDepth": max_depth,
        "timeMs": int((time.time() - t0) * 1000),
    }

    return sf, stats
