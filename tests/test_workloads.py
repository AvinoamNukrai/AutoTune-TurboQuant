"""Trace generation must be deterministic and profile-shaped (SPEC §5.1)."""

from src.workloads import PROFILES, generate_trace


def test_traces_are_deterministic():
    for profile in PROFILES:
        a = generate_trace(profile, seed=123)
        b = generate_trace(profile, seed=123)
        assert a == b, f"{profile}: same seed must give identical traces"


def test_different_seeds_differ():
    a = generate_trace("chat", seed=1)
    b = generate_trace("chat", seed=2)
    assert a["requests"] != b["requests"]


def test_profile_shapes():
    chat = generate_trace("chat", seed=7)
    assert all(len(r["prompt"].split()) < 100 for r in chat["requests"])
    assert all(200 <= r["max_tokens"] <= 500 for r in chat["requests"])

    rag = generate_trace("rag", seed=7)
    for r in rag["requests"]:
        # ~4k-8.4k tokens ≈ at least 2500 words of context
        assert len(r["prompt"].split()) > 2500
        assert r["max_tokens"] <= 32
        assert "internal reference code" in r["prompt"]

    batch = generate_trace("batch", seed=7)
    assert all(r["max_tokens"] == 128 for r in batch["requests"])


def test_scale_shrinks_requests():
    full = generate_trace("chat", seed=7)
    small = generate_trace("chat", seed=7, scale=0.25)
    assert small["n_requests"] < full["n_requests"]
    assert small["n_requests"] >= 4
