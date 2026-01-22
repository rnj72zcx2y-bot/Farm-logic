import json
import os
from typing import Dict, Any

def write_json(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)

def strip_internal(puzzle: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in puzzle.items() if k != "_internal"}
