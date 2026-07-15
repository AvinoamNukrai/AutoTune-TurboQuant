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

## Methodology / infrastructure lessons

9. **vLLM v1 is multi-process**: `torch.cuda.max_memory_allocated()` in the
   client process reads 0 — VRAM must be measured device-level via NVML.
   (Smoke-test `peak_vram_gb: 0.0` is this bug, not a real number.)
10. **Reproducibility trap**: `vllm==0.20.2` ships sdist-only → pip silently
    attempts an hours-long CUDA source build. Pin `0.21.0` (prebuilt wheel)
    and install with `--only-binary :all:`.
