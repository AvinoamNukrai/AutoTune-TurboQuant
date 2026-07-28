# src/profiles.py
"""Profile configuration loader.

Workload profiles define how the tuner weighs speed vs memory and what
PPL degradation threshold is acceptable. Defaults are baked in; a JSON
config file can override them.
"""
from __future__ import annotations

import json
from pathlib import Path

_DEFAULT_CONFIG_PATH = Path("configs/profiles.json")

DEFAULT_PROFILES: dict[str, dict] = {
    "chat": {
        "terms": [("s_tpot", 0.7), ("r_mem", 0.3)],
        "ppl_threshold": 0.005,
    },
    "rag": {
        "terms": [("r_mem", 0.5), ("s_ttft", 0.5)],
        "ppl_threshold": 0.01,
    },
    "batch": {
        "terms": [("s_tp", 0.8), ("r_mem", 0.2)],
        "ppl_threshold": 0.02,
    },
}


def load_profiles(path: Path | str | None = None) -> dict[str, dict]:
    """Load profile configs from JSON, falling back to built-in defaults."""
    if path is None:
        path = _DEFAULT_CONFIG_PATH
    path = Path(path)
    if path.exists():
        raw = json.loads(path.read_text())
        profiles = {}
        for name, cfg in raw.items():
            profiles[name] = {
                "terms": [tuple(t) for t in cfg["terms"]],
                "ppl_threshold": float(cfg["ppl_threshold"]),
            }
        return profiles
    return dict(DEFAULT_PROFILES)
