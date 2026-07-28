# Project Refactoring Design

**Date:** 2026-07-28
**Goal:** Systematic refactoring to strengthen the project's academic framing, eliminate unnecessary reimplementation, and align with course guidelines.

---

## Context

Through critical review, we identified four issues:

1. The profiler reimplements ~100 lines of TurboQuant math (Hadamard, Lloyd-Max, quantize/dequant) when it could import from vLLM or use vLLM's engine directly
2. The project is framed as "a TurboQuant auto-tuner" but should be framed as "investigating whether compression policy matters for KV cache management"
3. Utility function parameters (PPL thresholds, exponents) are hard-coded with no way to override
4. SPEC.md and docstrings don't reflect the hypothesis-driven framing

## Section 1 — Profiler Refactor

### Current state

`src/profiler.py` contains four reimplemented functions (lines 47-157):
- `_solve_lloyd_max_normal()` — Lloyd-Max centroids
- `hadamard_matrix()` — Walsh-Hadamard matrix construction
- `simulate_turbo_quant_keys()` — Hadamard rotation + Lloyd-Max quantization
- `simulate_turbo_quant_values()` — per-vector uniform quantization

These are used for:
1. Feature computation (features 2 and 5 in `compute_features()`)
2. Ground-truth sensitivity measurement (`_compute_hf_ppl_chunked()`)

### Changes

**1a. Import vLLM's quantization utils instead of reimplementing**

Replace our implementations with imports from vLLM where available. Wrap in a try/except so the profiler still works if vLLM isn't installed (e.g., on a CPU-only machine for unit tests).

```python
try:
    from vllm.model_executor.layers.quantization.utils.hadamard import hadamard_matrix
    from vllm.model_executor.layers.quantization.utils.centroids import compute_centroids
    _VLLM_QUANT_AVAILABLE = True
except ImportError:
    _VLLM_QUANT_AVAILABLE = False
    # Fall back to our implementations (kept as _fallback_*)
```

The exact import paths depend on the installed vLLM version. We'll probe the actual module structure on the cluster and adjust. If vLLM doesn't expose clean Python-level imports for these, we keep our implementation but clearly document it as "ported from vLLM's <exact file>" with version pinning.

**1b. Add vLLM-based ground-truth profiling mode**

New `--mode vllm` that generates a per-layer sensitivity manifest and runs it through the existing harness. For a model with L layers, this creates L cells:

```
For each layer i in [0, L-1]:
    Cell: kv_cache_dtype="turboquant_4bit_nc"
          skip_layers = [all layers EXCEPT i]  # only layer i is quantized
    → measure PPL
    → ΔPPL = PPL(layer_i_quantized) - PPL(all_FP16)
```

This uses vLLM's actual CUDA kernels with zero simulation. Output: `results/exp0/sensitivity_vllm.json` with the same format as the current sensitivity output.

Implementation: a new function `generate_sensitivity_manifest()` in `src/profiler.py` that:
1. Detects number of layers from the model config (HuggingFace AutoConfig)
2. Generates a JSON manifest of L+1 cells (L single-layer-quantized + 1 baseline)
3. The user runs it through `python -m src.harness --manifest`
4. A new `--mode vllm-analyze` reads the results and computes the sensitivity ranking

This mode is slower (~1.5-2 hours for 36 layers) but provides ground truth.

**1c. Rename existing mode**

- `--mode features` stays (fast feature extraction)
- `--mode sensitivity` becomes `--mode sim-sensitivity` (simulation-based)
- `--mode full` becomes `--mode sim-full` (simulation-based full pipeline)
- New `--mode vllm` generates manifest
- New `--mode vllm-analyze` analyzes results

### Files changed

- `src/profiler.py` — refactored imports, new manifest generation, mode renaming

---

## Section 2 — Project Reframing

### Current framing (weak)

"AutoTuneTurboQuant auto-selects the optimal TurboQuant config."
Problem: only 4 presets, user could try manually. Sounds over-engineered.

### New framing (strong)

"We investigate whether KV cache compression policy matters — not just WHETHER to compress, but WHERE (which layers) and HOW MUCH (which bit-width). We prove that:
1. Sensitivity to compression is layer-dependent (some layers tolerate 3-bit, others don't)
2. Sensitivity is model-dependent (3-bit works on 4B models, destroys 1.7B models)
3. The optimal compression strategy depends on the workload profile (chat vs batch have different quality tolerances)

This establishes that adaptive compression depth is a necessary complement to eviction-based KV cache management. TurboQuant is the experimental vehicle; the finding and methodology are the contribution."

### Key narrative shifts

| Old | New |
|-----|-----|
| "We build an auto-tuner" | "We prove compression policy matters and build a tool that implements it" |
| "TurboQuant is our feature" | "TurboQuant is our experimental vehicle" |
| "Framework saves time" | "Framework prevents quality failures the user can't predict" |
| "We reimplemented the algorithm" | "Simulation mode documents its math as a vLLM replication; vLLM mode bypasses it entirely" |

### Related work to cite

- **KIVI** (Liu et al., 2024) — per-channel KV quantization with sensitivity analysis
- **KVQuant** (Hooper et al., 2024) — outlier-aware KV cache quantization
- **HAWQ** (Dong et al., 2019) — Hessian-Aware mixed-precision quantization (weight domain, same principle)
- **SqueezeLLM** (Kim et al., 2024) — sensitivity-based mixed-precision

Our differentiation: applying layer-sensitivity analysis specifically to vLLM's TurboQuant with workload-aware utility functions and a practical recommendation tool.

### Files changed

- `SPEC.md` — rewrite sections 1.1-1.4 with new framing
- `src/profiler.py` — update module docstring
- `src/tuner.py` — update module docstring
- `src/advisor.py` — update module docstring

---

## Section 3 — Configurable Utility Parameters

### Current state

Hard-coded in `src/tuner.py` lines 44-57:
```python
PROFILES_CFG = {
    "chat": {"terms": [("s_tpot", 0.7), ("r_mem", 0.3)], "ppl_threshold": 0.005},
    "rag":  {"terms": [("r_mem", 0.5), ("s_ttft", 0.5)], "ppl_threshold": 0.01},
    "batch": {"terms": [("s_tp", 0.8), ("r_mem", 0.2)], "ppl_threshold": 0.02},
}
```

Same values hard-coded in `analysis/exp3_validation.py`, `analysis/exp4_generalization.py`, and `src/advisor.py`.

### Changes

**3a. Extract to a config file**

Create `configs/profiles.json`:
```json
{
  "chat": {
    "description": "Interactive chat — TPOT-sensitive",
    "terms": [["s_tpot", 0.7], ["r_mem", 0.3]],
    "ppl_threshold": 0.005
  },
  "rag": {
    "description": "Retrieval-augmented generation — memory+TTFT",
    "terms": [["r_mem", 0.5], ["s_ttft", 0.5]],
    "ppl_threshold": 0.01
  },
  "batch": {
    "description": "Bulk offline processing — throughput",
    "terms": [["s_tp", 0.8], ["r_mem", 0.2]],
    "ppl_threshold": 0.02
  }
}
```

**3b. Load from config with CLI override**

All modules that use profile configs load from this file:
```python
def load_profiles(path="configs/profiles.json"):
    return json.loads(Path(path).read_text())
```

Add CLI flags for override:
```
--ppl-threshold chat=0.01   # override chat threshold
--profiles-config path.json  # use custom profiles file
```

**3c. Single source of truth**

Remove duplicated constants from `analysis/exp3_validation.py`, `analysis/exp4_generalization.py`, and `src/advisor.py`. All import from the same config loader.

### Files changed

- New: `configs/profiles.json`
- `src/tuner.py` — load from config, remove hard-coded dict
- `src/advisor.py` — load from config
- `analysis/exp3_validation.py` — load from config
- `analysis/exp4_generalization.py` — load from config

---

## Section 4 — SPEC.md and Documentation Updates

### Changes

- Rewrite SPEC.md §1 (Executive Summary) with hypothesis-driven framing
- Add §1.5 Related Work section citing KIVI, KVQuant, HAWQ
- Update the "Claim to Fame" to emphasize the FINDING over the TOOL
- Update component descriptions to reflect profiler refactor
- Clarify that VramSampler already measures actual GPU memory (it does!)

### Files changed

- `SPEC.md` — sections 1.1-1.5, component A description

---

## What we are NOT changing

- **Experiment results** — all existing results remain valid
- **Harness** (`src/harness.py`) — no changes needed, already uses VramSampler
- **Metrics** (`src/metrics.py`) — already has NVML-based VRAM measurement
- **Workloads** (`src/workloads.py`) — no changes
- **Analysis scripts** — only change is import path for profile configs
- **Manifests** — exp4_all.json already updated with correct models

## Risks

- **vLLM import paths**: The exact module structure for quantization utils may differ across vLLM versions. Mitigation: try/except with fallback to our implementations.
- **vLLM-mode profiler speed**: ~1.5-2 hours for 36 layers. Acceptable as an optional ground-truth mode.
- **Config file adds complexity**: One more file to maintain. But eliminates 4x duplicated constants.
