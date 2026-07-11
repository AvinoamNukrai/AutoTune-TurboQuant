# AutoTuneTurboQuant

Configuration-space characterization and workload-aware auto-tuning of
**TurboQuant KV-cache quantization** in stock [vLLM](https://github.com/vllm-project/vllm).

Given a (model, GPU, workload) combination, AutoTuneTurboQuant finds the optimal
TurboQuant configuration — preset, layer-protection set, and kernel parameters —
instead of relying on vLLM's hand-picked defaults. The layer-protection policy is
derived from measured per-layer sensitivity statistics (see [SPEC.md](SPEC.md) for
the full technical specification and experimental program).

**Baseline:** unmodified vLLM 0.20.2 (TurboQuant merged upstream in
[vllm#38479](https://github.com/vllm-project/vllm/pull/38479)).
**Primary target:** Qwen3-4B on NVIDIA RTX 3090 (24 GB).

## Setup (one-time, on a GPU machine)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Phase 1 — smoke test

Verifies all TurboQuant presets on the actual GPU, probes the
`--kv-cache-dtype-skip-layers` behavior (including whether the hard-coded
boundary-protection floor can be disabled), and measures real per-cell cost
for the experiment budget.

```bash
python scripts/smoke_test.py
```

Output: `results/smoke_test.json` + a summary table to stdout.
Exit code 0 = all configs passed.

## Repository layout

```
SPEC.md                   full technical specification & experimental program
scripts/smoke_test.py     Phase 1: preset verification + cost measurement
docker/Dockerfile         reproducibility image (vLLM base + analysis deps)
src/                      (Phase 2+) profiler, harness, tuner, advisor
results/                  benchmark outputs (JSON/CSV)
```

## How to benchmark

Full instructions land with the Phase-2 harness. The design contract:
every experiment cell is checkpointed by config-hash — reruns skip
completed cells, and all analysis runs offline from the CSVs (zero GPU cost
to iterate on figures/statistics).
