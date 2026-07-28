# tests/test_profiles.py
"""Tests for profile config loading."""

import json
import pytest
from pathlib import Path
from src.profiles import load_profiles, DEFAULT_PROFILES


def test_default_profiles_have_all_three():
    profiles = DEFAULT_PROFILES
    assert set(profiles.keys()) == {"chat", "rag", "batch"}


def test_each_profile_has_required_fields():
    for name, cfg in DEFAULT_PROFILES.items():
        assert "terms" in cfg, f"{name} missing 'terms'"
        assert "ppl_threshold" in cfg, f"{name} missing 'ppl_threshold'"
        assert isinstance(cfg["ppl_threshold"], float)
        assert len(cfg["terms"]) >= 2
        for term_name, weight in cfg["terms"]:
            assert isinstance(term_name, str)
            assert isinstance(weight, (int, float))


def test_load_profiles_from_file(tmp_path):
    custom = {
        "chat": {
            "terms": [["s_tpot", 0.6], ["r_mem", 0.4]],
            "ppl_threshold": 0.01,
        },
        "rag": {
            "terms": [["r_mem", 0.5], ["s_ttft", 0.5]],
            "ppl_threshold": 0.02,
        },
        "batch": {
            "terms": [["s_tp", 0.8], ["r_mem", 0.2]],
            "ppl_threshold": 0.03,
        },
    }
    f = tmp_path / "profiles.json"
    f.write_text(json.dumps(custom))
    loaded = load_profiles(f)
    assert loaded["chat"]["ppl_threshold"] == 0.01
    assert loaded["chat"]["terms"][0][1] == 0.6


def test_load_profiles_falls_back_to_defaults():
    loaded = load_profiles(Path("/nonexistent/path.json"))
    assert loaded == DEFAULT_PROFILES


def test_ppl_thresholds_are_ordered():
    p = DEFAULT_PROFILES
    assert p["chat"]["ppl_threshold"] < p["rag"]["ppl_threshold"]
    assert p["rag"]["ppl_threshold"] < p["batch"]["ppl_threshold"]
