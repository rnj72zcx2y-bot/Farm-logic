import os
from dataclasses import dataclass

from . import __version__ as GENERATOR_VERSION

GIT_COMMIT = os.getenv("GITHUB_SHA", "dev")[:7]
GLOBAL_SALT = os.getenv("GLOBAL_SALT", "CHANGE_ME")

@dataclass(frozen=True)
class DifficultySpec:
    name: str
    rows: int
    cols: int
    active_cells: int
    cards: int
    regions: int

SPECS = {
    "easy":   DifficultySpec("easy",   4, 4, 12,  6,  4),
    "medium": DifficultySpec("medium", 6, 6, 30, 15, 10),
    "hard":   DifficultySpec("hard",   8, 8, 48, 24, 16),
}

FENCE_COLORS = ["white", "brown", "black"]

ANIMALS = [
    ("Chicken", 2),
    ("Cow", 4),
    ("Horse", 4),
    ("Dog", 4),
    ("Cat", 4),
    ("Bee", 6),
    ("Spider", 8),
    ("Snail", 0),
]
