# Insights Log — material for the final report

Short, dated findings collected during the project. Each entry: what we found,
how we know, and why it matters for the report.

## Baseline & configuration space (from vLLM source reading, 2026-07-11)

1. **TurboQuant is upstream in vLLM** (merged 2026-04-15,
   [vllm#38479](https://github.com/vllm-project/vllm/pull/38479)) but exposed
   only as **4 fixed presets** (`k8v4`, `4bit_nc`, `k3v4_nc`, `3bit_nc`).
   Key-bits / value-bits / norm-correction are **confounded inside the preset**
   — the full factorial exists in the code (`TurboQuantConfig` dataclass) but is
   not reachable from the CLI. → Motivation + PR opportunity.
2. **Layer protection is hard-coded**: vLLM force-protects the first 2 and last
   2 layers for every `turboquant_*` dtype, validated by its authors on Qwen3-4B
   (their comment: removing it costs ~30 GSM8K points at aggressive presets).
3. **Protection is expensive on small models**: protecting s of L layers cuts
   effective compression to `L/(s+(L−s)/r)`. Qwen3-4B (36L) at `3bit_nc`
   (4.9×): the default n=2 leaves only 3.5× — ~29% of the compression paid for.
   On an 80-layer model the same rule is nearly free → optimal protection is
   model-size-dependent.
4. **Community context (Red Hat / vLLM study, May 2026)**: no single config
   wins — FP8 KV is the best default on Hopper+, 3-bit collapses on reasoning
   at long context, Qwen-class models need more K bits than V bits, and layer-0
   has ~20% outlier channels vs 4–6% mid-stack. Nobody automated the choice.

## Smoke test on RTX 3090, Qwen3-4B, vLLM 0.21.0 (2026-07-14)

5. **The n=2 protection floor cannot be disabled on stock vLLM** (measured):
   passing an empty `--kv-cache-dtype-skip-layers` still yields effective
   protection `[0, 1, 34, 35]`; user-supplied layers are *unioned* with the
   floor (`[10,11]` → `[0,1,10,11,34,35]`). You can protect **more, never
   fewer**. → Shapes the experiment grid (floor + additions); strong PR
   motivation; report-worthy limitation of the shipped implementation.
6. **`turboquant_k8v4` fails to initialize on RTX 3090** (Ampere, SM 8.6 — no
   native FP8), while all three `_nc` presets work and generate coherent
   output. Pending root-cause confirmation, the **feasible preset space itself
   is hardware-dependent** — hardware-awareness demonstrated before any tuning.
7. **All `_nc` presets pass sanity on 3090**: 8/8 distinct coherent outputs,
   correct 4K-token retrieval answer, ~460–470 tok/s short-gen (vs 498 tok/s
   FP16 baseline — ~6% decode overhead at trivial scale; real differences are
   expected under long context / concurrency, not here).
8. **Per-cell cost measured**: ~85 s TQ engine load (226 s only on first cold
   weight read), ~103 s per config with mini-workloads. Confirms the
   ~30–40 GPU-h budget for the full experimental program, with margin.

## Smoke test on RTX 4090, Qwen3-4B, vLLM 0.21.0 (2026-07-14)

11. **7/7 configs pass on RTX 4090 (Ada, SM 8.9), including `k8v4`** — the
    preset that fails to initialize on the 3090. Combined with #6: **the
    feasible preset space is hardware-dependent**, measured from both sides
    (same model, same vLLM, same scripts; only the GPU differs). AutoTune must
    treat feasibility, not just optimality, as GPU-specific.
12. **`k8v4` is the slowest TQ preset on the 4090 at small scale** (562 tok/s
    vs ~612 for all `_nc` presets, 642 FP16 baseline) — the *lightest*
    quantization costs the most decode speed. Single mini-run → hypothesis for
    Experiment 1, not a claim yet.
13. **The n=2 protection floor reproduces identically on the 4090**
    (empty skip list → `[0,1,34,35]`) — finding #5 generalizes across GPUs.

## Harness validation — baseline vs. 3bit_nc, RTX 4090, Qwen3-4B (2026-07-24)

Source: Phase 2 harness (`src/harness.py`), two cells — `auto` (FP16 baseline)
vs. `turboquant_3bit_nc`, both rep 0 on RTX 4090.

14. **TTFT paradox: 3-bit quantization is 8× faster for chat TTFT** (78 ms vs.
    636 ms baseline). Compressed KV means more tokens fit per chunked-prefill
    batch → less queuing → faster time-to-first-token under concurrent
    scheduling. This is a workload-dependent trade-off: TTFT improves while
    per-token decode slows down. → Core motivation for workload-aware tuning.
15. **TPOT is 19% slower with 3-bit** (13.6 ms vs. 11.5 ms baseline). Each
    generated token pays quantization/dequantization overhead. Chat workloads
    (decode-heavy) are hurt; RAG workloads (prefill-heavy, short output) are
    less affected.
16. **PPL degrades 3.4% under 3bit_nc** (10.70 vs. 10.35 baseline). Exceeds
    the SPEC's 0.5% chat tolerance but within the 2% batch threshold. →
    Confirms that aggressive quantization is acceptable for some workloads but
    not others — the utility function per-profile design is necessary.
17. **Peak VRAM is identical between configs** (21.64 GB 3-bit vs. 21.55 GB
    baseline). vLLM pre-allocates a fixed KV-cache pool based on
    `gpu_memory_utilization` (0.85 × 24 GB). Quantization does not reduce
    *peak memory used* — it increases *capacity* (more tokens fit in the same
    pool). **Peak VRAM is the wrong metric for measuring compression benefit**;
    the correct metric is the number of KV-cache blocks allocated or max
    concurrent sequences. → Must add block-count extraction to harness before
    Experiment 1.
18. **Needle accuracy is 1.0 (20/20) under 3bit_nc** — the model retrieves the
    correct embedded code from 4K–8K token contexts even at aggressive
    quantization. Combined with finding #16: 3-bit hurts PPL (global quality)
    but not simple retrieval (local attention still works).
19. **Per-cell wall time is ~70–100 s on RTX 4090** (67 s for 3bit_nc, 99 s for
    auto). FP16 baseline is slower due to longer engine init (59 s vs. 27 s) —
    larger per-token KV means more CUDA graph compilations. Budget for Exp 1
    (28 cells × 2 reps × ~85 s) ≈ 1.3 GPU-hours — well within the 8–12 h
    allocation.

## Experiment 0 — Layer sensitivity profiler, Qwen3-4B (2026-07-25)

Source: `src/profiler.py --mode full`, HuggingFace Transformers (not vLLM),
8 WikiText calibration docs, 4 eval docs, 3-bit keys / 4-bit values,
chunked PPL with 256-token chunks. Results in `results/exp0/`.

22. **Layer 0 is 10× more sensitive than any other layer** (ΔPPL=+4.59 vs.
    next-worst layer 34 at +0.24). quant_error=6.84 vs ~0.15 median. Layer 0
    must always be protected regardless of budget.
23. **`simulated_quant_error` is the only feature that predicts sensitivity**
    (Spearman ρ=+0.40, p=0.016). The other four candidates — key outlier
    fraction, post-WHT excess kurtosis, key-norm CV, value dynamic range — all
    fail (p>0.10). This is because TurboQuant's Hadamard rotation specifically
    neutralizes the outlier/kurtosis properties those features measure.
24. **vLLM's positional protection is suboptimal on Qwen3-4B.** The fixed
    "first 2 + last 2" rule protects {0,1,34,35}. Layer 1 is insensitive
    (ΔPPL=+0.01), wasting budget. Layers 30-33 are sensitive (ΔPPL=+0.07
    to +0.15) but unprotected. A statistics-guided policy protecting
    {0,32,33,34} at the same 4-layer budget covers 3× more ΔPPL.
25. **Layer 5 breaks the correlation** — second-highest quant_error (0.85) but
    zero sensitivity (ΔPPL=-0.007). The quantization damage doesn't propagate
    downstream, likely because early layers have enough residual capacity to
    compensate. This limits single-feature prediction to ρ≈0.4.
26. **Correct Lloyd-Max centroids matter.** Initial hardcoded centroids (wrong
    3-bit values, only 12/16 for 4-bit) gave ρ=0.50. Switching to dynamically
    computed centroids via the iterative Lloyd-Max algorithm for N(0,1) dropped
    it to ρ=0.40 — the earlier "better" result was an artifact of wrong math.
27. **Value dynamic range increases monotonically with depth** — from 0.24
    (layer 0) to 26.3 (layer 34), dropping to 18.0 at layer 35. Despite this
    100× spread, it doesn't correlate with sensitivity (ρ=+0.24, p=0.15),
    because TurboQuant uses per-vector min-max scaling that absorbs range.
28. **Some mid-layers show negative ΔPPL** (layers 18, 20: ΔPPL=-0.08, -0.04).
    Quantization acts as regularization — the added noise slightly improves
    generalization on the eval set. Not reproducible across seeds; noise floor.
29. **Modern transformers (4.48+) uses `DynamicLayer` cache API**, not the older
    `DynamicCache.key_cache`/`value_cache` lists. Cache entries are accessed via
    `past_kv.layers[i].keys` / `.values`. Iteration yields 3-tuples, not
    2-tuples.

## Methodology / infrastructure lessons

9. **vLLM v1 is multi-process**: `torch.cuda.max_memory_allocated()` in the
   client process reads 0 — VRAM must be measured device-level via NVML.
   (Smoke-test `peak_vram_gb: 0.0` is this bug, not a real number.)
10. **Reproducibility trap**: `vllm==0.20.2` ships sdist-only → pip silently
    attempts an hours-long CUDA source build. Pin `0.21.0` (prebuilt wheel)
    and install with `--only-binary :all:`.
20. **vLLM V1 offline `LLM.generate()` requires `disable_log_stats=False`** to
    populate `RequestOutput.metrics`. With `True` (which we initially set to
    reduce noise), all per-request latencies are `None`. The V1 metrics object
    is `RequestStateStats`, not the V0 `RequestMetrics` — field names differ
    (`first_token_latency` vs. `first_token_time`; monotonic timestamps vs.
    wall-clock). Source: Phase 2 harness debugging.
21. **HuggingFace `datasets` namespace change**: `load_dataset("wikitext", ...)`
    fails on recent `huggingface_hub` versions — must use
    `"Salesforce/wikitext"`. Silent breakage if not caught. Source: harness
    PPL computation failure on cluster.
