# KVCompressionTune — Project Reframing & Final Direction

**Date:** 2026-07-30
**Context:** After professor meeting feedback and comprehensive literature survey

---

## The Problem

vLLM ships TurboQuant with 4 fixed presets and a hard-coded layer protection rule (first 2 + last 2 layers, engine-level, not TurboQuant itself). A user who wants to compress their KV cache has to pick a preset blindly — they can't see PPL in production, they don't know which presets crash on their GPU, and they don't know that a preset safe on one model destroys another.

## The Research Question

**Does the choice of compression preset matter, and if so — for whom?**

Not "how to compress better" (that's TurboQuant's job). Not "which layers to protect" (our data shows it's marginal). Just: given the existing presets, does the right choice depend on the workload and model?

---

## Key Findings

| Finding | Evidence | Implication |
|---------|----------|-------------|
| Preset selection dominates layer protection (4x) | Exp 1, INSIGHTS #30 | Focus on preset choice, not layer tuning |
| Optimal preset is workload-dependent | Exp 2-3, INSIGHTS #37, #44 | No single default serves all workloads |
| Optimal preset is model-dependent | Exp 4 (4 models, 3 families) | Llama/Mistral tolerate 3bit; Qwen-1.7B needs k8v4 |
| Larger models absorb quantization better | Exp 4 | Mistral-7B handles 3bit for RAG (dPPL +0.69%); Qwen-1.7B batch utility <1 |
| k8v4 crashes on Ampere (RTX 3090) | Smoke test, INSIGHTS #6 | Preset feasibility is hardware-dependent |
| TurboQuant incompatible with Gemma architecture | Exp 4 (Gemma-2-2B) | KV page size mismatch — compatibility is also architecture-dependent |
| Naive 4bit_nc on chat exceeds quality threshold silently | Exp 3, INSIGHTS #45 | Users need guidance, not guessing |
| Per-layer protection is marginal | Exp 1-3, INSIGHTS #46-47 | Negative result — layer protection doesn't beat positional |

---

## What We Contribute

1. **Empirical characterization** of TurboQuant's preset space across workloads (3 profiles), models (4 models, 3 architecture families), and hardware — first systematic study of this configuration space
2. **Workload-aware utility framework** — chat/RAG/batch each have different quality thresholds and performance weights. No existing paper (KVTuner, KVmix, RateQuant) considers workload
3. **Practical decision tool** on unmodified stock vLLM — input: (model, GPU, workload) → output: recommended `--kv-cache-dtype` flags with evidence and rejected alternatives
4. **Negative result on layer protection** — with explanation grounded in error propagation literature

## What We Do NOT Claim

- We do NOT propose a new quantization method
- We do NOT improve compression quality beyond what TurboQuant already offers
- We do NOT claim per-layer protection is a major lever (our own data refutes this)
- The "improvement" is in the **decision**, not in the algorithm

---

## Professor's Feedback & Response

### "Error propagates across layers, so layer protection is pointless"

The professor is right about the mechanism. KV cache quantization errors accumulate across both layers (spatial) and generation steps (temporal). Multiple papers confirm this:
- KVarN (arxiv 2606.03458) — explicit error accumulation model
- KVTuner (ICML 2025, arxiv 2502.04420) — Equation 3: error at layer l depends on all previous layers
- QEP (NeurIPS 2025, arxiv 2504.09629) — error grows exponentially even when downstream layers stay FP16

Our own data confirms: preset selection has 4x more impact than layer protection (INSIGHTS #30), and stats-guided protection didn't beat positional default (INSIGHTS #47).

**Response:** "You're right. Our Experiments 1-3 confirm this — preset selection has 4x more impact than layer protection. We report this as a negative result. The project's contribution is workload-aware preset selection, which is independent of the per-layer question."

### "Choosing from 4 presets is trivial"

**Response:** "The choice is trivial if you know the impact. But PPL isn't visible in production, preset feasibility is hardware-dependent, and the same preset has 16x different impact across models. The system makes the informed choice and prevents silent quality violations."

### "What's new here?"

**Response:** "No existing paper combines TurboQuant characterization + workload-specific utility functions + practical stock-vLLM tool. KVTuner/KVmix propose new methods requiring custom kernels. We characterize and automate the policy over an existing method that millions of vLLM users already have access to."

---

## Positioning vs. Related Work

### Directly Relevant (must cite and differentiate)

| Paper | What they do | How we differ |
|-------|-------------|---------------|
| **KVTuner** (Li et al., ICML 2025, arxiv 2502.04420) | Per-layer mixed precision KV cache quantization, KIVI-based, multi-objective optimization | We study preset selection (not per-layer), TurboQuant-specific, workload-aware utility |
| **KVmix** (Li et al., AAAI 2026 Oral, arxiv 2506.08018) | Gradient-based per-layer importance allocation | Different method, no workload awareness |
| **RateQuant** (arxiv 2605.06675) | Rate-distortion theory for optimal bit allocation | Theoretical framework; we're empirical + practical on stock vLLM |
| **Red Hat/vLLM TurboQuant blog** (May 2026) | TurboQuant benchmark across 4 models on 5 benchmarks | They characterize; we characterize + automate + add workload dimension |
| **"Hold Onto That Thought"** (NeurIPS 2025 workshop, arxiv 2512.12008) | Shows no singular compression strategy fits all tasks | They benchmark many methods; we go deep on one (TurboQuant) with utility functions |

### Error Propagation (supports professor's argument)

| Paper | Key finding |
|-------|------------|
| **KVarN** (arxiv 2606.03458) | Errors accumulate across layers AND generation steps |
| **QEP** (NeurIPS 2025, arxiv 2504.09629) | Error grows exponentially even with downstream layers at full precision |
| **Structural Sensitivity** (arxiv 2603.20991) | Residual connections contract errors (Lyapunov theory), but contraction has limits |

### Broader KV Cache Quantization (background)

- **KIVI** (Liu et al., 2024, arxiv 2402.02750) — per-channel keys, per-token values
- **KVQuant** (Hooper et al., NeurIPS 2024, arxiv 2401.18079) — outlier-aware, per-layer sensitivity-weighted
- **TurboQuant** (Zandieh et al., ICLR 2026, arxiv 2504.19874) — the algorithm we use
- **HAWQ** (Dong et al., ICCV 2019) — Hessian-aware mixed precision (weight domain)
- **MoE-nD** (arxiv 2604.17695) — per-layer routing to compression configs
- **PM-KVQ** (arxiv 2505.18610) — progressive mixed-precision KV cache
- **CoopQ** (arxiv 2509.15455) — Shapley values for inter-layer interactions

---

## Direction Change Summary

### What we DROP (or demote to negative result)
- Per-layer protection as headline contribution
- "Statistics-guided protection beats positional" hypothesis (H1b — falsified)
- The framing "we optimize which layers to protect"

### What we KEEP as the main story
- Workload-aware preset selection (optimal preset varies by workload AND model)
- Model-dependent sensitivity confirmed across 3 families (Qwen, Llama, Mistral)
- Hardware-dependent feasibility (k8v4 crashes on Ampere)
- Architecture-dependent compatibility (TurboQuant incompatible with Gemma)
- Practical tool on stock vLLM

### What we REFRAME
- Layer sensitivity analysis (Exp 0) → background work that motivated investigation
- Negative result on H1b → honest finding, explained by error propagation
- "AutoTuner" → "empirical characterization + decision tool"

---

## Optional Additional Experiment

**PPL degradation vs. output length** (~1 GPU-hour):
Run 4bit_nc at output lengths 50, 100, 200, 500 tokens. If degradation grows with length, this shows temporal error accumulation is workload-relevant (chat generates more tokens → more accumulation → needs milder preset). A one-paragraph observation, not a full study.

---

## Honest Assessment: Is the System Broad Enough?

### What's solid right now
- Rigorous methodology (5 reps, held-out validation, Holm-Bonferroni correction across 30 tests)
- Real findings backed by data across 5 experiments
- Honest negative result on layer protection
- Working end-to-end pipeline (profiler → harness → tuner → advisor)

### What's thin
1. ~~**Only 2 models, same family (Qwen).**~~ **RESOLVED.** Exp 4 ran on 4 models across 3 families: Qwen3-4B, Qwen3-1.7B, Llama-3.1-8B, Mistral-7B-v0.3. Gemma-2-2B was attempted but TurboQuant is incompatible with its architecture (KV page size mismatch — itself a finding).
2. **Only 1 primary GPU (RTX 4090).** The "hardware-dependent" claim relies on a single observation (k8v4 crashes on 3090). Valid but thin. Acceptable for a course project since GPU access is limited.
3. **The advisor is a lookup table.** It can only recommend configs it has already profiled. Give it an unseen model → it has nothing to say. Could at least estimate risk based on model properties (parameter count, num_layers, architecture family).
4. **No cost translation.** "2.25x compression" is abstract. "Saves $X per month" or "serves Y more concurrent users" is concrete. One paragraph with back-of-envelope math would make practical value tangible.

### Priority additions to make it comprehensive

| Addition | Effort | Impact | Status |
|----------|--------|--------|--------|
| Run Exp 4 on Llama + Mistral | ~4 GPU-hours | **High** — validates "model-dependent" across families | **DONE** (Llama, Mistral, Qwen-1.7B; Gemma incompatible) |
| PPL vs output length experiment | ~1 GPU-hour | **Medium** — connects error propagation to workload | Not started |
| Cost/capacity calculation in advisor output | ~2 hours code | **Medium** — makes practical value concrete | Not started |
| System architecture diagram in report | ~1 hour | **Medium** — makes it look like a real system | Not started |
| Decision heuristic for unseen models | ~3 hours code | **Medium** — makes advisor generalizable | Not started |

**Critical path:** ~~The Llama/Mistral runs are essential.~~ **DONE.** Cross-family validation complete. Everything else is polish.

### The framing matters more than the mechanism

The project is an **applied systems/MLOps contribution**, not an algorithmic one. This is legitimate and valuable:

- **Algorithmic contribution** (what we're NOT doing): "We invented a new quantization method with better distortion rates." That's KVTuner/KVmix territory, requires CUDA kernels, and is a PhD-level effort.
- **Systems contribution** (what we ARE doing): "vLLM has a powerful compression tool that millions of users have access to. Nobody knows how to configure it correctly for their workload without silently breaking quality. We built the framework that maps the trade-offs and makes the decision automatically."

The finding that "the complicated solution (per-layer protection) is unnecessary, and the simple solution (right preset per workload) is what actually matters" is not a weakness — it's the central insight. In science, proving that complexity is unnecessary and providing a validated simple alternative is a strong result.

**The "boring" version:** "We tried 4 presets and picked the best one per workload."

**The professional version:** "We performed a comprehensive empirical characterization of TurboQuant's configuration space in vLLM, mapping the trade-offs between VRAM, latency, and PPL across workloads, models, and hardware. We discovered that the prevailing assumption about per-layer protection is secondary (4x less impact) to preset selection, and that the optimal preset is workload-dependent and model-dependent. We developed a Workload-Aware Decision Engine based on utility functions that prevents silent quality failures in production."

Same work. Different framing. The second version is accurate and honest — it just uses precise language.

---

## Code Gaps for the Reframing

### What's missing for the reframed project to be complete:

#### 1. Advisor doesn't accept GPU input (HIGH)
`src/advisor.py` takes `--model` and `--profile` but has NO `--gpu` flag. It can't warn about k8v4 feasibility on Ampere vs Ada. The "hardware-dependent" claim has no code support.
**Fix:** Add `--gpu` argument. Maintain a simple feasibility map (e.g., `k8v4` requires SM >= 8.9). Output a warning or filter infeasible presets.

#### 2. Advisor is single-model lookup (HIGH)
The advisor reads from `results/optimal_configs.json`, which is produced by the tuner for ONE model. If you profile Qwen3-4B, the advisor only knows about Qwen3-4B. It can't say anything about Llama or Mistral without re-running the entire pipeline.
**Fix (minimal):** Support multiple model configs in `optimal_configs.json` keyed by model name. The advisor takes `--model Qwen/Qwen3-4B` and looks up the right entry.
**Fix (better):** Add a heuristic fallback: if the model isn't profiled, estimate risk based on parameter count / num_layers (smaller models → more sensitive → recommend milder presets).

#### 3. Tuner is hard-coded to Qwen3-4B layout (MEDIUM)
`src/tuner.py` has `N_LAYERS = 36` and `FLOOR_LAYERS = frozenset({0, 1, 34, 35})` hard-coded at the top. Won't work for Llama (32 layers) or any other model without editing the source.
**Fix:** Detect num_layers from model config (the profiler already has `_detect_n_layers()`), or accept `--n-layers` as CLI arg. Compute floor dynamically: `{0, 1, L-2, L-1}`.

#### 4. No plots module (MEDIUM)
No visualization code exists. The report needs charts: utility per workload, PPL vs preset, model sensitivity comparison. Currently all analysis outputs are text tables.
**Fix:** Add `analysis/plots.py` (or a notebook) that reads cell results and produces matplotlib/seaborn figures for the report.

#### 5. Missing tests for tuner and advisor (MEDIUM — 40% of grade)
Tests exist for: profiler, profiles, harness, workloads. NO tests for `tuner.py` or `advisor.py`.
**Fix:** Add `tests/test_tuner.py` (utility computation, r_mem formula, cell averaging) and `tests/test_advisor.py` (format_cli, format_python, load_configs).

#### 6. ~~No Exp 4 cross-family results yet~~ — DONE
Exp 4 completed on Llama-3.1-8B, Mistral-7B-v0.3, and Qwen3-1.7B. Gemma-2-2B incompatible with TurboQuant (KV page size mismatch in vLLM). Results confirm model-dependent preset selection across 3 architecture families.

**Exp 4 results summary:**

| Model | Chat | RAG | Batch | Key observation |
|-------|------|-----|-------|-----------------|
| Qwen3-4B | k8v4 | 4bit_nc | 4bit_nc | Original baseline (Exp 1-3) |
| Qwen3-1.7B | k8v4 | k8v4 | k8v4 (U=0.99) | Small model barely benefits; batch is break-even |
| Llama-3.1-8B | 4bit_nc | 4bit_nc | 3bit_nc | Tolerates aggressive compression |
| Mistral-7B-v0.3 | 4bit_nc | 3bit_nc | 3bit_nc | Most compression-tolerant model |

#### 7. No cost/capacity estimation (LOW-MEDIUM)
The advisor outputs "2.25x compression" but not "serves X more concurrent users" or "saves $Y/month." Adding a back-of-envelope capacity calculation would make the practical value concrete.
**Fix:** In advisor output, add: `Max concurrent sequences: ~N (vs ~M baseline)` based on KV cache bytes per token × max_model_len × gpu_memory_utilization.

#### 8. PPL vs output length experiment (LOW — optional but nice)
The temporal error accumulation experiment: measure PPL degradation at increasing output lengths (50, 100, 200, 500 tokens) under 4bit_nc. Would explain WHY chat needs milder presets.
**Fix:** New manifest + small analysis script. ~1 GPU-hour.

#### 9. Docstrings/framing still say "AutoTune" in places (LOW)
Some internal comments and the tuner's Optuna study names still reference the old "auto-tuner" framing. Should reflect "workload-aware preset selection."
**Fix:** Grep and update.

### Priority order for remaining work:

1. ~~**Run Exp 4 on Llama + Mistral**~~ — **DONE**
2. ~~**Make tuner model-agnostic**~~ — **DONE** (Jad branch)
3. ~~**Add GPU input to advisor + feasibility filtering**~~ — **DONE** (Jad branch)
4. ~~**Add tests for tuner + advisor**~~ — **DONE** (Jad branch)
5. ~~**Add plots module**~~ — **DONE** (Jad branch)
6. ~~**Multi-model advisor support**~~ — **DONE** (Jad branch)
7. **Cost/capacity estimation** — polish
8. **PPL vs output length experiment** — bonus finding
9. **Docstring cleanup** — cosmetic
10. **Fix: tuner `load_cells` doesn't filter by model** — bug found during Exp 4 (workaround: per-model cells directories)

---

## One-Sentence Positioning

*vLLM ships TurboQuant with hand-picked defaults; this project shows those defaults are workload-dependent and model-dependent, and provides a tool that selects the right configuration with quality guarantees on unmodified stock vLLM.*
