"""Cell hashing (checkpoint identity) and latency statistics."""

from src.harness import CellConfig
from src.metrics import RequestLatency, percentile, summarize_latencies


def test_cell_hash_stable_and_sensitive():
    a = CellConfig(kv_cache_dtype="turboquant_3bit_nc", rep=0)
    b = CellConfig(kv_cache_dtype="turboquant_3bit_nc", rep=0)
    assert a.cell_hash() == b.cell_hash(), "identical configs must collide"

    for changed in [
        CellConfig(kv_cache_dtype="turboquant_4bit_nc", rep=0),
        CellConfig(kv_cache_dtype="turboquant_3bit_nc", rep=1),
        CellConfig(kv_cache_dtype="turboquant_3bit_nc", skip_layers=("10",)),
        CellConfig(kv_cache_dtype="turboquant_3bit_nc", trace_seed=999),
    ]:
        assert changed.cell_hash() != a.cell_hash(), f"{changed} must differ"


def test_percentile():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(vals, 0.5) == 3.0
    assert percentile(vals, 0.0) == 1.0
    assert percentile(vals, 1.0) == 5.0
    assert percentile([], 0.5) is None


def test_tpot_derivation():
    lat = RequestLatency(ttft_s=0.5, e2e_s=2.5, n_output_tokens=21)
    assert abs(lat.tpot_s - 0.1) < 1e-9  # (2.5-0.5)/20


def test_summarize_handles_missing_metrics():
    lats = [RequestLatency(ttft_s=None, e2e_s=None, n_output_tokens=10)]
    s = summarize_latencies(lats)
    assert s["ttft_s"]["mean"] is None
    assert s["total_output_tokens"] == 10
