import time
from typing import Dict, Any, Tuple, List

import networkx as nx


def _domino_tileable(active_cells: List[List[int]]) -> Tuple[bool, int]:
    """Check if the active cell set is domino-tileable using bipartite matching.

    Returns (tileable, matched_pairs).
    """
    cells = [tuple(rc) for rc in active_cells]
    n = len(cells)
    if n % 2 == 1:
        return False, 0

    cell_to_idx = {cell: i for i, cell in enumerate(cells)}

    # Bipartition by chessboard parity
    U, V = [], []
    for (r, c), idx in cell_to_idx.items():
        (U if (r + c) % 2 == 0 else V).append(idx)

    if len(U) != len(V):
        return False, 0

    G = nx.Graph()
    G.add_nodes_from(U, bipartite=0)
    G.add_nodes_from(V, bipartite=1)

    for (r, c), u in cell_to_idx.items():
        if u not in U:
            continue
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in cell_to_idx:
                v = cell_to_idx[nb]
                G.add_edge(u, v)

    matching = nx.algorithms.bipartite.maximum_matching(G, top_nodes=U)
    matched = sum(1 for u in U if u in matching)
    return matched == len(U), matched


def solve_count(puzzle: Dict[str, Any], stop_at: int = 2) -> Tuple[int, Dict[str, Any]]:
    """Stage-1 solver: feasibility only.

    This implementation is meant to keep the pipeline running while the real CSP solver
    (Stop-at-2 with full region rules) is implemented.

    Returns:
      solutionsFound: 0 if not tileable, 1 if tileable (NOT uniqueness yet)
      stats: minimal stats for meta.json
    """
    t0 = time.time()

    grid = puzzle.get("grid", {})
    active = grid.get("activeCellsCoords", [])

    if not active:
        stats = {
            "type": "tiling-feasibility",
            "stopAt": stop_at,
            "solutionsFound": 0,
            "nodesVisited": 0,
            "backtracks": 0,
            "maxDepth": 0,
            "timeMs": int((time.time() - t0) * 1000),
        }
        return 0, stats

    ok, matched_pairs = _domino_tileable(active)
    solutions_found = 1 if ok else 0

    stats = {
        "type": "tiling-feasibility",
        "stopAt": stop_at,
        "solutionsFound": solutions_found,
        "nodesVisited": matched_pairs,
        "backtracks": 0,
        "maxDepth": len(active) // 2,
        "timeMs": int((time.time() - t0) * 1000),
    }
    return solutions_found, stats
