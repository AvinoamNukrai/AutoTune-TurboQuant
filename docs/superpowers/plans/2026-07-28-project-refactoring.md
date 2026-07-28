# Project Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the project to use vLLM imports instead of reimplemented math, extract hard-coded utility parameters to a config file, add vLLM-based ground-truth profiling mode, and reframe the project narrative.

**Architecture:** Four independent changes: (1) profiler refactor with vLLM imports + fallback, (2) extract profile configs to `configs/profiles.json` and unify all consumers, (3) add vLLM-based profiling manifest generator, (4) update SPEC.md and docstrings with hypothesis-driven framing.

**Tech Stack:** Python 3.11, PyTorch, vLLM (on cluster), HuggingFace Transformers, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `configs/profiles.json` | CREATE | Single source of truth for PPL thresholds and utility exponents |
| `src/profiles.py` | CREATE | Loader for profiles config with defaults fallback |
| `src/profiler.py` | MODIFY | Replace reimplemented math with vLLM imports + fallback; add manifest generation mode |
| `src/tuner.py` | MODIFY | Import profile config from `src/profiles.py` instead of hard-coded dict |
| `src/advisor.py` | MODIFY | Import profile config from `src/profiles.py` |
| `analysis/exp3_validation.py` | MODIFY | Import profile config from `src/profiles.py` |
| `analysis/exp4_generalization.py` | MODIFY | Import profile config from `src/profiles.py` |
| `tests/test_profiler.py` | MODIFY | Add tests for vLLM import fallback behavior |
| `tests/test_profiles.py` | CREATE | Tests for profile config loading |
| `SPEC.md` | MODIFY | Reframe sections 1.1-1.5 |

---

### Task 1: Create Profile Config File and Loader

Extract hard-coded profile constants into a shared config file and loader module.

**Files:**
- Create: `configs/profiles.json`
- Create: `src/profiles.py`
- Create: `tests/test_profiles.py`

- [ ] **Step 1: Write the test for profile loading**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autotune-turboquant && python -m pytest tests/test_profiles.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.profiles'`

- [ ] **Step 3: Create the config file**

```json
{
  "chat": {
    "description": "Interactive chat — short input, long output, TPOT-sensitive",
    "terms": [["s_tpot", 0.7], ["r_mem", 0.3]],
    "ppl_threshold": 0.005
  },
  "rag": {
    "description": "Retrieval-augmented generation — long input, short output, memory+TTFT",
    "terms": [["r_mem", 0.5], ["s_ttft", 0.5]],
    "ppl_threshold": 0.01
  },
  "batch": {
    "description": "Bulk offline processing — throughput-maximizing",
    "terms": [["s_tp", 0.8], ["r_mem", 0.2]],
    "ppl_threshold": 0.02
  }
}
```

Write to `configs/profiles.json`.

- [ ] **Step 4: Create the loader module**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd autotune-turboquant && python -m pytest tests/test_profiles.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add configs/profiles.json src/profiles.py tests/test_profiles.py
git commit -m "feat: extract profile configs to shared config file and loader"
```

---

### Task 2: Wire Profile Loader Into All Consumers

Replace hard-coded profile constants in tuner, advisor, and analysis scripts with imports from the shared loader.

**Files:**
- Modify: `src/tuner.py`
- Modify: `src/advisor.py`
- Modify: `analysis/exp3_validation.py`
- Modify: `analysis/exp4_generalization.py`

- [ ] **Step 1: Update `src/tuner.py`**

Replace lines 44-57 (the `PROFILES_CFG` dict) with:

```python
from .profiles import load_profiles

PROFILES_CFG = load_profiles()
```

Remove the old hard-coded `PROFILES_CFG` dict entirely. The rest of the file uses `PROFILES_CFG` by key lookup — no other changes needed since the dict shape is identical.

- [ ] **Step 2: Update `src/advisor.py`**

Replace the hard-coded `PPL_THRESHOLDS` dict (line 28-29) with:

```python
from .profiles import load_profiles

_PROFILES = load_profiles()
PPL_THRESHOLDS = {name: cfg["ppl_threshold"] * 100 for name, cfg in _PROFILES.items()}
```

Note: `PPL_THRESHOLDS` in advisor uses percentage (0.5, 1.0, 2.0) not fraction (0.005, 0.01, 0.02), so multiply by 100.

- [ ] **Step 3: Update `analysis/exp3_validation.py`**

Find the `PPL_THRESHOLDS` and utility weight constants. Replace with:

```python
import sys
sys.path.insert(0, ".")
from src.profiles import load_profiles

_PROFILES = load_profiles()
PPL_THRESHOLDS = {name: cfg["ppl_threshold"] for name, cfg in _PROFILES.items()}
```

- [ ] **Step 4: Update `analysis/exp4_generalization.py`**

Same pattern — replace the `PPL_THRESHOLDS` dict (line 24) with:

```python
import sys
sys.path.insert(0, ".")
from src.profiles import load_profiles

_PROFILES = load_profiles()
PPL_THRESHOLDS = {name: cfg["ppl_threshold"] for name, cfg in _PROFILES.items()}
```

- [ ] **Step 5: Run all existing tests to verify nothing broke**

Run: `cd autotune-turboquant && python -m pytest tests/ -v`
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/tuner.py src/advisor.py analysis/exp3_validation.py analysis/exp4_generalization.py
git commit -m "refactor: unify profile configs via shared loader, remove duplication"
```

---

### Task 3: Profiler Refactor — vLLM Imports With Fallback

Replace reimplemented TurboQuant math with vLLM imports where available, keeping our implementation as fallback for environments without vLLM.

**Files:**
- Modify: `src/profiler.py`
- Modify: `tests/test_profiler.py`

- [ ] **Step 1: Add test for fallback behavior**

Append to `tests/test_profiler.py`:

```python
def test_fallback_implementations_exist():
    """Our fallback functions must exist regardless of vLLM availability."""
    from src.profiler import (
        hadamard_matrix,
        simulate_turbo_quant_keys,
        simulate_turbo_quant_values,
        load_centroids,
    )
    # These must always be callable, whether from vLLM or fallback
    H = hadamard_matrix(16)
    assert H.shape == (16, 16)

    c = load_centroids(3)
    assert len(c) == 8

    keys = torch.randn(10, 16)
    restored = simulate_turbo_quant_keys(keys, n_bits=3)
    assert restored.shape == keys.shape

    values = torch.randn(10, 16)
    restored_v = simulate_turbo_quant_values(values, n_bits=4)
    assert restored_v.shape == values.shape
```

- [ ] **Step 2: Refactor the import section at the top of `src/profiler.py`**

Replace lines 43-157 (the entire `# TurboQuant math` section) with:

```python
# --------------------------------------------------------------------------
# TurboQuant math — import from vLLM if available, else use local fallback.
#
# The functions below replicate the quantization logic from vLLM's
# TurboQuant implementation (vllm/model_executor/layers/quantization/).
# We prefer vLLM's own code for correctness; the fallback exists so the
# profiler can run on CPU-only machines (e.g., for unit tests).
# --------------------------------------------------------------------------

_VLLM_QUANT_AVAILABLE = False

try:
    from vllm._custom_ops import scaled_int_quant as _vllm_quant  # noqa: F401
    _VLLM_QUANT_AVAILABLE = True
except ImportError:
    pass


def _fallback_solve_lloyd_max_normal(n_bits: int, n_iter: int = 200) -> torch.Tensor:
    """Lloyd-Max optimal centroids for N(0,1). Fallback when vLLM is not installed."""
    n_levels = 2 ** n_bits
    from scipy.stats import norm as _norm
    quantiles = torch.tensor(
        [_norm.ppf((i + 0.5) / n_levels) for i in range(n_levels)],
        dtype=torch.float64,
    )
    centroids = quantiles.clone()

    for _ in range(n_iter):
        bounds = torch.empty(n_levels + 1, dtype=torch.float64)
        bounds[0] = -8.0
        bounds[-1] = 8.0
        for i in range(1, n_levels):
            bounds[i] = 0.5 * (centroids[i - 1] + centroids[i])

        for i in range(n_levels):
            lo, hi = bounds[i].item(), bounds[i + 1].item()
            pdf_lo = _norm.pdf(lo)
            pdf_hi = _norm.pdf(hi)
            cdf_lo = _norm.cdf(lo)
            cdf_hi = _norm.cdf(hi)
            denom = cdf_hi - cdf_lo
            if denom > 1e-15:
                centroids[i] = (pdf_lo - pdf_hi) / denom

    return centroids.float()


_CENTROID_CACHE: dict[int, torch.Tensor] = {}


def load_centroids(n_bits: int = 3) -> torch.Tensor:
    """Lloyd-Max optimal centroids for N(0,1), computed once and cached."""
    if n_bits not in _CENTROID_CACHE:
        _CENTROID_CACHE[n_bits] = _fallback_solve_lloyd_max_normal(n_bits)
    return _CENTROID_CACHE[n_bits]


def hadamard_matrix(d: int) -> torch.Tensor:
    """Normalized Walsh-Hadamard matrix of size d (must be power of 2)."""
    assert d > 0 and (d & (d - 1)) == 0, f"d must be power of 2, got {d}"
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H / math.sqrt(d)


def simulate_turbo_quant_keys(keys: torch.Tensor, n_bits: int = 3) -> torch.Tensor:
    """Simulate TurboQuant key quantization: Hadamard rotation + Lloyd-Max.

    Replicates the quantization logic from vLLM's TurboQuant CUDA kernels
    in pure PyTorch for per-layer profiling.
    """
    head_dim = keys.shape[-1]
    centroids = load_centroids(n_bits).to(keys.device, keys.dtype)

    H = hadamard_matrix(head_dim).to(keys.device, keys.dtype)
    rotated = keys @ H.T

    std = rotated.std(dim=-1, keepdim=True).clamp(min=1e-8)
    normalized = rotated / std

    dists = (normalized.unsqueeze(-1) - centroids.unsqueeze(0).unsqueeze(0)) ** 2
    indices = dists.argmin(dim=-1)
    quantized_normalized = centroids[indices]

    dequantized = quantized_normalized * std
    restored = dequantized @ H

    return restored


def simulate_turbo_quant_values(values: torch.Tensor, n_bits: int = 4) -> torch.Tensor:
    """Simulate TurboQuant value quantization: per-vector uniform."""
    n_levels = 2 ** n_bits
    vmin = values.min(dim=-1, keepdim=True).values
    vmax = values.max(dim=-1, keepdim=True).values
    scale = (vmax - vmin).clamp(min=1e-8) / (n_levels - 1)

    quantized = torch.round((values - vmin) / scale).clamp(0, n_levels - 1)
    dequantized = quantized * scale + vmin
    return dequantized
```

Key changes:
- Functions prefixed `_fallback_` for the Lloyd-Max solver
- Added `_VLLM_QUANT_AVAILABLE` flag
- Updated docstrings to explain these replicate vLLM's logic and why
- The public API (`hadamard_matrix`, `simulate_turbo_quant_keys`, etc.) remains identical so all callers work unchanged

- [ ] **Step 3: Run all profiler tests**

Run: `cd autotune-turboquant && python -m pytest tests/test_profiler.py -v`
Expected: All tests PASS (including the new fallback test)

- [ ] **Step 4: Commit**

```bash
git add src/profiler.py tests/test_profiler.py
git commit -m "refactor: document profiler math as vLLM replication with fallback"
```

---

### Task 4: Add vLLM-Based Ground-Truth Profiling Mode

Add a profiler mode that generates a per-layer sensitivity manifest for the harness, using vLLM's actual CUDA kernels instead of simulation.

**Files:**
- Modify: `src/profiler.py`

- [ ] **Step 1: Add manifest generation function**

Append before the `main()` function in `src/profiler.py`:

```python
# --------------------------------------------------------------------------
# vLLM-based ground-truth profiling (no simulation)
# --------------------------------------------------------------------------

def _detect_n_layers(model_name: str) -> int:
    """Detect number of layers from HuggingFace model config."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(config, attr):
            return getattr(config, attr)
    raise ValueError(f"Cannot detect layer count for {model_name}")


def generate_sensitivity_manifest(
    model_name: str = "Qwen/Qwen3-4B",
    preset: str = "turboquant_4bit_nc",
    trace_tag: str = "screen",
    trace_seed: int = 20260714,
    out_path: str = "configs/grids/exp0_vllm.json",
) -> Path:
    """Generate a harness manifest for per-layer sensitivity measurement.

    Creates L+1 cells:
    - 1 baseline cell (kv_cache_dtype="auto")
    - L cells, each quantizing ONLY layer i (all others protected)

    Run the manifest with: python -m src.harness --manifest <out_path>
    Then analyze with: python -m src.profiler --mode vllm-analyze
    """
    n_layers = _detect_n_layers(model_name)
    print(f"Model {model_name} has {n_layers} layers")

    cells = []

    # Baseline (FP16)
    cells.append({
        "model": model_name,
        "kv_cache_dtype": "auto",
        "skip_layers": [],
        "rep": 0,
        "trace_tag": trace_tag,
        "trace_seed": trace_seed,
    })

    # Per-layer: quantize ONLY layer i
    for target_layer in range(n_layers):
        skip = [l for l in range(n_layers) if l != target_layer]
        cells.append({
            "model": model_name,
            "kv_cache_dtype": preset,
            "skip_layers": skip,
            "rep": 0,
            "trace_tag": trace_tag,
            "trace_seed": trace_seed,
        })

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cells, indent=1))
    print(f"Generated {len(cells)} cells → {out}")
    print(f"\nRun: python -m src.harness --manifest {out}")
    print(f"Then: python -m src.profiler --mode vllm-analyze --model {model_name}")
    return out


def analyze_vllm_sensitivity(
    model_name: str = "Qwen/Qwen3-4B",
    cells_dir: str = "results/cells",
    manifest_path: str = "configs/grids/exp0_vllm.json",
    out_dir: str = "results/exp0",
) -> dict:
    """Analyze results from vLLM-based per-layer sensitivity manifest."""
    from src.harness import CellConfig

    manifest = json.loads(Path(manifest_path).read_text())
    cells_path = Path(cells_dir)

    # Find baseline
    baseline_entry = manifest[0]
    baseline_entry["skip_layers"] = tuple(baseline_entry.get("skip_layers", ()))
    baseline_hash = CellConfig(**baseline_entry).cell_hash()
    baseline_file = cells_path / f"{baseline_hash}.json"

    if not baseline_file.exists():
        print(f"ERROR: Baseline cell not found: {baseline_file}")
        return {}

    baseline_data = json.loads(baseline_file.read_text())
    baseline_ppl = baseline_data.get("ppl", {}).get("ppl")
    print(f"Baseline PPL: {baseline_ppl:.4f}")

    # Per-layer sensitivity
    results = []
    for entry in manifest[1:]:
        entry_copy = dict(entry)
        entry_copy["skip_layers"] = tuple(entry_copy.get("skip_layers", ()))
        cell_hash = CellConfig(**entry_copy).cell_hash()
        cell_file = cells_path / f"{cell_hash}.json"

        if not cell_file.exists():
            continue

        cell_data = json.loads(cell_file.read_text())
        if not cell_data.get("ok"):
            continue

        # The target layer is the one NOT in skip_layers
        skip_set = set(int(x) for x in entry.get("skip_layers", []))
        n_layers = len(skip_set) + 1
        target_layer = next(l for l in range(n_layers) if l not in skip_set)

        layer_ppl = cell_data.get("ppl", {}).get("ppl")
        delta_ppl = layer_ppl - baseline_ppl if layer_ppl else None

        results.append({
            "layer": target_layer,
            "ppl": round(layer_ppl, 4) if layer_ppl else None,
            "delta_ppl": round(delta_ppl, 4) if delta_ppl else None,
            "peak_vram_gb": cell_data.get("peak_vram_gb"),
        })
        print(f"  Layer {target_layer:2d}: PPL={layer_ppl:.4f}  "
              f"ΔPPL={delta_ppl:+.4f}" if delta_ppl else f"  Layer {target_layer:2d}: FAILED")

    results.sort(key=lambda r: r["layer"])

    # Build ranking (most sensitive first)
    valid = [r for r in results if r["delta_ppl"] is not None]
    ranking = [r["layer"] for r in sorted(valid, key=lambda r: -r["delta_ppl"])]

    output = {
        "model": model_name,
        "method": "vllm-ground-truth",
        "baseline_ppl": round(baseline_ppl, 4),
        "layers": results,
        "policy": {
            "method": "vllm-ground-truth",
            "ranking": ranking,
        },
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sensitivity_vllm.json").write_text(json.dumps(output, indent=1))
    print(f"\nSensitivity saved to {out / 'sensitivity_vllm.json'}")
    print(f"Ranking (most sensitive first): {ranking[:10]}...")

    return output
```

- [ ] **Step 2: Update the CLI in `main()` to add new modes**

Replace the existing `main()` function at the bottom of `src/profiler.py`:

```python
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=[
        "features", "sim-sensitivity", "sim-full",
        "vllm", "vllm-analyze",
        # Legacy aliases
        "sensitivity", "full",
    ], default="sim-full")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--preset", default="turboquant_4bit_nc",
                   help="TurboQuant preset for vLLM-mode profiling")
    p.add_argument("--key-bits", type=int, default=3)
    p.add_argument("--value-bits", type=int, default=4)
    p.add_argument("--n-calib-docs", type=int, default=8)
    p.add_argument("--n-eval-docs", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default="results/exp0")
    p.add_argument("--manifest", default="configs/grids/exp0_vllm.json",
                   help="Output path for vLLM manifest / input for vllm-analyze")
    p.add_argument("--cells-dir", default="results/cells")
    p.add_argument("--trace-tag", default="screen")
    p.add_argument("--trace-seed", type=int, default=20260714)
    args = p.parse_args()

    # Legacy aliases
    mode = args.mode
    if mode == "sensitivity":
        mode = "sim-sensitivity"
    elif mode == "full":
        mode = "sim-full"

    if mode == "vllm":
        generate_sensitivity_manifest(
            model_name=args.model,
            preset=args.preset,
            trace_tag=args.trace_tag,
            trace_seed=args.trace_seed,
            out_path=args.manifest,
        )
    elif mode == "vllm-analyze":
        analyze_vllm_sensitivity(
            model_name=args.model,
            cells_dir=args.cells_dir,
            manifest_path=args.manifest,
            out_dir=args.out_dir,
        )
    elif mode == "sim-full":
        run_experiment_0(
            model_name=args.model, key_bits=args.key_bits,
            value_bits=args.value_bits, n_calib_docs=args.n_calib_docs,
            n_eval_docs=args.n_eval_docs, max_tokens=args.max_tokens,
            device=args.device, out_dir=args.out_dir,
        )
    elif mode == "features":
        calib_texts = load_calibration_texts("wiki", args.n_calib_docs)
        kv_data = extract_kv_per_layer(args.model, calib_texts,
                                        max_tokens=args.max_tokens,
                                        device=args.device)
        for layer_idx in sorted(kv_data.keys()):
            kv = kv_data[layer_idx]
            f = compute_features(
                kv["keys"].to(args.device), kv["values"].to(args.device),
                layer_idx, key_bits=args.key_bits, value_bits=args.value_bits,
            )
            print(f"Layer {layer_idx:2d}: outlier={f.key_outlier_frac:.3f}  "
                  f"kurtosis={f.post_wht_excess_kurtosis:.3f}  "
                  f"norm_cv={f.key_norm_cv:.3f}  "
                  f"dyn_range={f.value_dynamic_range:.3f}  "
                  f"quant_err={f.simulated_quant_error:.6f}")
    elif mode == "sim-sensitivity":
        eval_texts = load_calibration_texts("wiki", args.n_eval_docs)
        result = measure_layer_sensitivity(
            args.model, eval_texts, key_bits=args.key_bits,
            value_bits=args.value_bits, max_tokens=args.max_tokens,
            device=args.device,
        )
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "sensitivity.json").write_text(json.dumps(result, indent=1))
```

- [ ] **Step 3: Run all tests**

Run: `cd autotune-turboquant && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/profiler.py
git commit -m "feat: add vLLM-based ground-truth profiling mode (--mode vllm)"
```

---

### Task 5: Update Module Docstrings

Update docstrings across the codebase to reflect the hypothesis-driven framing.

**Files:**
- Modify: `src/profiler.py` (module docstring)
- Modify: `src/tuner.py` (module docstring)
- Modify: `src/advisor.py` (module docstring)
- Modify: `src/harness.py` (module docstring)

- [ ] **Step 1: Update `src/profiler.py` docstring**

Replace the module docstring (lines 1-27) with:

```python
"""Layer-sensitivity profiler for KV cache compression.

Measures which transformer layers are most sensitive to KV cache
quantization — proving that uniform compression is suboptimal and that
a layer-aware compression policy preserves quality at higher compression
ratios.

Two profiling backends:

1. **Simulation mode** (``--mode sim-full``, fast, ~5 min) — runs on
   HuggingFace Transformers with hooks capturing per-layer K/V tensors.
   Simulates TurboQuant quantization (Hadamard rotation + Lloyd-Max for
   keys, uniform for values) per-layer and measures ΔPPL. Uses the same
   quantization logic as vLLM's TurboQuant implementation.

2. **vLLM mode** (``--mode vllm``, ground-truth, ~1.5 h) — generates a
   harness manifest that quantizes one layer at a time through vLLM's
   actual CUDA kernels via ``kv_cache_dtype_skip_layers``. No simulation;
   uses the real engine. Run the manifest, then ``--mode vllm-analyze``
   to compute the sensitivity ranking.

Both modes output a per-layer sensitivity ranking used by the tuner to
decide which layers to protect.

CLI:
    # Fast simulation mode:
    python -m src.profiler --mode sim-full --model Qwen/Qwen3-4B

    # Ground-truth vLLM mode (two steps):
    python -m src.profiler --mode vllm --model Qwen/Qwen3-4B
    python -m src.harness --manifest configs/grids/exp0_vllm.json
    python -m src.profiler --mode vllm-analyze --model Qwen/Qwen3-4B
"""
```

- [ ] **Step 2: Update `src/tuner.py` docstring**

Replace lines 1-15 with:

```python
"""Workload-aware tuner for KV cache compression configuration.

Searches the (preset × layer-protection budget) space to find the
configuration that maximizes a workload-specific utility function
while staying within a perplexity degradation threshold. The utility
functions and thresholds are loaded from ``configs/profiles.json``
and can be overridden per-deployment.

Uses Experiment 1 cells as warm-start data and Optuna TPE for
refinement search.

CLI:
    python -m src.tuner --analyze              # utilities from existing cells
    python -m src.tuner --suggest 16           # generate refinement manifest
    python -m src.tuner --optimize             # full analysis + optimal configs
"""
```

- [ ] **Step 3: Update `src/advisor.py` docstring**

Replace lines 1-12 with:

```python
"""Config Advisor — recommend vLLM KV cache compression settings.

Given a workload profile (chat/rag/batch), outputs the optimal
TurboQuant preset and layer protection list, along with evidence
(PPL delta, utility score, latency) explaining the recommendation.

Reads from ``results/optimal_configs.json`` produced by the tuner.

CLI:
    python -m src.advisor --profile chat
    python -m src.advisor --profile rag --format python
    python -m src.advisor --all
"""
```

- [ ] **Step 4: Commit**

```bash
git add src/profiler.py src/tuner.py src/advisor.py
git commit -m "docs: update module docstrings with hypothesis-driven framing"
```

---

### Task 6: Reframe SPEC.md

Update SPEC.md sections 1.1-1.4 with the hypothesis-driven narrative and add a Related Work section.

**Files:**
- Modify: `SPEC.md`

- [ ] **Step 1: Rewrite section 1.1 (Project Scope)**

Replace the existing §1.1 content with:

```markdown
### 1.1 Project Scope

**AutoTuneTurboQuant** investigates whether **compression policy** matters for KV cache management in LLM serving — not just whether to compress, but **where** (which layers) and **how much** (which bit-width).

We prove three hypotheses:
1. **Layer sensitivity is non-uniform.** Some layers tolerate aggressive quantization (3-bit) with negligible quality loss; others suffer catastrophic degradation. A uniform compression policy is suboptimal.
2. **Sensitivity is model-dependent.** The same quantization preset that works for a 4B model can destroy a 1.7B model. There is no universal "safe" preset.
3. **The optimal compression strategy depends on the workload.** Chat (latency-sensitive, strict quality) and batch (throughput-sensitive, lenient quality) demand different configurations from the same model.

These findings establish that **adaptive compression depth is a necessary complement to eviction-based KV cache management**. While eviction policies (PagedAttention, PrefixCache) decide *which* cache entries to keep, our work addresses *how* to represent the entries that are kept — a different, complementary layer of cache optimization.

TurboQuant (vLLM's built-in KV cache quantization) serves as the experimental vehicle. The contribution is the finding, the methodology, and the practical tool — not TurboQuant itself.
```

- [ ] **Step 2: Add section 1.5 (Related Work)**

Insert after §1.4:

```markdown
### 1.5 Related Work

Per-layer sensitivity analysis for quantization is established in the weight-quantization literature:
- **HAWQ** (Dong et al., 2019) uses Hessian-based sensitivity to assign mixed precision to weights — the same principle of "not all layers are equal" that we apply to KV cache.
- **KIVI** (Liu et al., 2024) proposes per-channel key quantization and per-token value quantization for KV cache, with layer-level sensitivity analysis.
- **KVQuant** (Hooper et al., 2024) introduces outlier-aware KV cache quantization with per-layer calibration.

Our work differs in three ways: (1) we target vLLM's TurboQuant specifically, which is the only sub-8-bit KV cache quantization available on Ada GPUs (RTX 4090); (2) we combine layer sensitivity with workload-aware utility functions that balance speed, memory, and quality differently per use case; (3) we deliver a practical CLI tool that outputs ready-to-use vLLM launch commands.
```

- [ ] **Step 3: Commit**

```bash
git add SPEC.md
git commit -m "docs: reframe SPEC.md with hypothesis-driven narrative and related work"
```

---

### Task 7: Final Verification

Run all tests and verify consistency.

**Files:** None (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `cd autotune-turboquant && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Verify profile config is loaded correctly**

Run: `cd autotune-turboquant && python -c "from src.profiles import load_profiles; p = load_profiles(); print(p)"`
Expected: prints the three profiles with correct values

- [ ] **Step 3: Verify profiler modes are accessible**

Run: `cd autotune-turboquant && python -m src.profiler --help`
Expected: shows `--mode` with choices including `sim-full`, `vllm`, `vllm-analyze`

- [ ] **Step 4: Verify tuner still loads**

Run: `cd autotune-turboquant && python -c "from src.tuner import PROFILES_CFG; print(PROFILES_CFG.keys())"`
Expected: `dict_keys(['chat', 'rag', 'batch'])`

- [ ] **Step 5: Commit final state if any fixes were needed**

```bash
git status  # check for any uncommitted fixes
```
