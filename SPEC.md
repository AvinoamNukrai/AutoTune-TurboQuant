# KVCompressionTune: Workload-Aware KV-Cache Compression Policy for vLLM

**Technical Specification & Execution Plan**

| | |
|---|---|
| **Author** | Avinoam Nukrai |
| **Course** | LLM Caching — Final Project |
| **Version** | 2.0 |
| **Date** | July 11, 2026 |
| **Baseline framework** | Stock vLLM (≥ 0.20) with in-tree TurboQuant KV-cache quantization |

---

## 1. Executive Summary & Motivation

### 1.1 Project Scope

**KVCompressionTune** investigates whether **compression policy** matters for KV cache management in LLM serving — not just whether to compress, but **where** (which layers) and **how much** (which bit-width).

We prove three hypotheses:
1. **Layer sensitivity is non-uniform.** Some layers tolerate aggressive quantization (3-bit) with negligible quality loss; others suffer catastrophic degradation. A uniform compression policy is suboptimal.
2. **Sensitivity is model-dependent.** The same quantization preset that works for a 4B model can destroy a 1.7B model. There is no universal "safe" preset.
3. **The optimal compression strategy depends on the workload.** Chat (latency-sensitive, strict quality) and batch (throughput-sensitive, lenient quality) demand different configurations from the same model.

These findings establish that **adaptive compression depth is a necessary complement to eviction-based KV cache management**. While eviction policies (PagedAttention, PrefixCache) decide *which* cache entries to keep, our work addresses *how* to represent the entries that are kept — a different, complementary layer of cache optimization.

TurboQuant (vLLM's built-in KV cache quantization) serves as the experimental vehicle. The contribution is the finding, the methodology, and the practical tool — not TurboQuant itself.

### 1.2 Problem Statement

TurboQuant KV-cache quantization ([Zandieh et al., ICLR 2026](https://arxiv.org/abs/2504.19874)) was merged into upstream vLLM in April 2026 ([vllm-project/vllm#38479](https://github.com/vllm-project/vllm/pull/38479)). It ships as **four fixed presets** (`turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_3bit_nc`) plus a **hard-coded layer-protection rule** (skip the first 2 and last 2 layers), validated on a handful of dense Qwen models.

Meanwhile, the community's own findings show that the right configuration is strongly context-dependent:

- The [Red Hat AI / vLLM study (May 2026)](https://blog.vllm.ai/2026/05/11/turboquant.html) found 3-bit modes lose ~20 accuracy points on reasoning benchmarks at long context, while being the right choice for memory-bound serving.
- Per-layer sensitivity varies drastically (layer 0 has ~20% outlier channels vs. 4–6% in middle layers), so a fixed boundary-protection rule is unlikely to be universally right.
- K/V bit-budget needs differ per model family (Qwen-class models need more key bits than value bits).
- FP8 alternatives win on Hopper+ GPUs but not on older hardware — the optimum is hardware-dependent.

Yet vLLM users today must pick a preset by hand, and the layer-protection rule is not even a choice — it is hard-coded. **Nobody has systematically mapped this configuration space or automated the choice.** That is the gap this project fills.

### 1.3 The Claim to Fame

> We prove that KV-cache compression policy matters — layer sensitivity is non-uniform, model-dependent, and workload-dependent. We present the first systematic characterization of TurboQuant's configuration space in vLLM, showing that the optimal configuration depends measurably on the model, the GPU, and the workload. The system turns this empirical map into a workload-aware tuner that outperforms vLLM's hand-picked defaults on every profile tested.

### 1.4 Positioning: A Research Contribution, Not a Wrapper

A wrapper calls an existing API with parameters someone else chose. This project's contribution is precisely the part that does not exist anywhere:

1. **The empirical map.** vLLM ships the mechanism; no one — including the vLLM maintainers, per their own hard-coded `n=2` boundary rule — has published a sensitivity analysis of the TurboQuant parameter space across models, GPUs, and workloads. The characterization study is new knowledge, not repackaging.
2. **The measured challenge to shipped defaults.** The project directly tests assumptions baked into upstream code (fixed presets, fixed boundary width) and quantifies when they are wrong and by how much.
3. **The automation.** KVCompressionTune converts the map into a decision procedure: given (model, GPU, workload), it outputs the configuration and the evidence. vLLM has no such capability.
4. **The tuner itself is evidence-driven.** Its search space and priors are outputs of the characterization phase — a methodological contribution over blind hyperparameter search.

**One-sentence positioning:** *vLLM ships TurboQuant with hand-picked defaults; this project proves those defaults are suboptimal and replaces them with configurations chosen from a measured understanding of what actually matters — per model, per GPU, per workload.*

### 1.5 Related Work

Per-layer sensitivity analysis for quantization is established in the weight-quantization literature:
- **HAWQ** (Dong et al., 2019) uses Hessian-based sensitivity to assign mixed precision to weights — the same principle of "not all layers are equal" that we apply to KV cache.
- **KIVI** (Liu et al., 2024) proposes per-channel key quantization and per-token value quantization for KV cache, with layer-level sensitivity analysis.
- **KVQuant** (Hooper et al., 2024) introduces outlier-aware KV cache quantization with per-layer calibration.

Our work differs in three ways: (1) we target vLLM's TurboQuant specifically, which is the only sub-8-bit KV cache quantization available on Ada GPUs (RTX 4090); (2) we combine layer sensitivity with workload-aware utility functions that balance speed, memory, and quality differently per use case; (3) we deliver a practical CLI tool that outputs ready-to-use vLLM launch commands.

---

## 2. System Architecture

The system is a pipeline of four decoupled components, all operating on **unmodified vLLM engines**:

```
┌─────────────────────────────────────────────────────────────────┐
│ (A) Layer Sensitivity Profiler  src/profiler.py                 │
│     HF-side, outside vLLM: hooks capture post-RoPE K/V (what    │
│     vLLM actually caches) over a calibration batch; computes    │
│     candidate per-layer features (§3.4) with streaming stats.   │
│     Features are VALIDATED against measured ground truth        │
│     (Experiment 0) before use. Output: per-layer ranking that   │
│     collapses the 2^L protection space to a scalar budget k.    │
├─────────────────────────────────────────────────────────────────┤
│ (B) Benchmark Harness          src/harness.py, src/workloads.py │
│     Launches stock vLLM with a given TurboQuant configuration,  │
│     replays frozen workload traces, collects all metrics.       │
├─────────────────────────────────────────────────────────────────┤
│ (C) Characterization Engine    src/characterize.py              │
│     Runs the screening experiment grid; computes main effects,  │
│     two-way interactions (ANOVA), and parameter importance      │
│     (fANOVA). Output: the empirical sensitivity map.            │
├─────────────────────────────────────────────────────────────────┤
│ (D) Auto-Tuner                 src/tuner.py                     │
│     Optuna Bayesian search over (preset, protection budget k,   │
│     kv-splits), guided by (A)'s ranking and (C)'s importance    │
│     map. Output: optimal config per (model, GPU, workload).     │
├─────────────────────────────────────────────────────────────────┤
│ (E) Config Advisor             src/advisor.py                   │
│     optimal_configs.json lookup + CLI: given (model, GPU,       │
│     workload) → recommended vLLM launch flags + evidence trail. │
└─────────────────────────────────────────────────────────────────┘
```

**Division of labor between (A) and (D):** the profiler *ranks* layers but cannot price the accuracy-vs-compression trade-off; the tuner *decides* the budget k jointly with the preset, because the right k depends on the workload utility (RAG prices memory, chat prices accuracy), interacts with the preset (protection is likely worthless at `k8v4`, critical at `3bit_nc`), and shifts with hardware. Statistics propose; optimization disposes.

### 2.1 Configuration Application Granularity

All TurboQuant parameters in vLLM are **engine-startup parameters** (the KV-cache pool is allocated once, with fixed bytes-per-token). Accordingly, KVCompressionTune applies configurations at **engine-launch granularity**: the advisor emits the flags for `vllm serve` / `LLM(...)`, and "workload-aware" means the right configuration is selected per deployment context. Live per-request re-configuration is out of scope (see Section 10, Future Work) — it would require modifying vLLM's cache allocator and attention kernels, conflicting with the stock-vLLM constraint.

---

## 3. The Configuration Space Under Study

All parameters below are reachable on stock vLLM. **None is excluded a priori** — the characterization phase measures each one's impact before the tuner narrows focus.

### 3.1 TurboQuant-specific parameters

| # | Parameter | Values | vLLM surface |
|---|-----------|--------|--------------|
| 1 | **Preset** = (key bits, value bits, norm correction) | `turboquant_k8v4` (FP8 keys + 4-bit values), `turboquant_4bit_nc` (4/4+NC), `turboquant_k3v4_nc` (3/4+NC), `turboquant_3bit_nc` (3/3+NC) | `--kv-cache-dtype` |
| 2 | **Layer-protection set** (which layers stay FP16) | any subset of layers; parameterized as boundary widths (n_first, n_ last) ∈ {0..4}², plus sensitivity-guided non-boundary sets | `--kv-cache-dtype-skip-layers` |
| 3 | **Decode-kernel split count** | {8, 16, 32, 64} | `AttentionConfig.tq_max_kv_splits_for_cuda_graph` |

Note on parameter 1: on stock vLLM, key bits, value bits, and norm-correction are *confounded within the preset* — the presets do not cover the full factorial (e.g., no `k4v3`, no `4bit` without NC). The characterization therefore treats the preset as a single 4-level factor; isolating the confounded sub-factors requires the bonus PR track (Section 9).

### 3.2 Interacting serving parameters

| # | Parameter | Values | Role |
|---|-----------|--------|------|
| 4 | Cache block size | {16, 32} | paging granularity |
| 5 | `gpu_memory_utilization` | [0.7 – 0.95] | KV pool size — mediates how compression converts to capacity |
| 6 | `max_num_seqs` | {32 … 512} | concurrency ceiling — mediates how capacity converts to throughput |

These are not TurboQuant parameters, but they determine how TurboQuant's memory savings translate into throughput; the characterization measures whether they interact with the quantization choice or merely scale it.

### 3.3 Context axes (not tuned — characterized across)

- **Model:** primary = **Qwen3-4B** (fits the 24 GB primary GPU with KV headroom; deliberately the same model on which vLLM's hard-coded n=2 rule was validated — beating the default on its own reference model is the strongest form of the claim). A second model (e.g., Llama-3.1-8B) receives only a small **spot-check** (~5 configs) demonstrating that the optimal configuration differs across models.
- **GPU:** primary = **NVIDIA RTX 3090 (24 GB, Ampere/SM 8.6)** on the SLURM cluster; vLLM auto-selects the `fp8e4b15` key format for SM < 8.9, so all presets run, and FP8-key behavior on Ampere-vs-newer is itself part of the hardware-dependence story. A second GPU generation may receive the same spot-check treatment.
- **Workload profile:** Section 4. All three profiles are measured in every cell at no extra engine-launch cost (one engine session replays all traces), and every profile's utility is computed from the same measurements.

**Compute budget:** the entire experimental program is sized to **~30–40 GPU-hours total** (Section 6) — one GPU over a weekend, or a day on two GPUs. Cells are independent, so wall-clock divides by available GPUs. Phase 1 measures the true per-cell cost and the manifest is re-budgeted from actuals.

### 3.4 The Central Hypothesis: Statistics-Guided Layer Protection

The layer-protection dimension carries a quantifiable two-sided trade-off. Protecting *s* of *L* layers leaves them FP16, reducing effective compression from *r* to `L / (s + (L−s)/r)`:

- Qwen3-4B (36 layers), `3bit_nc` (r = 4.9×), default n=2: effective compression drops to **3.5×** — boundary protection costs ~29% of the compression paid for.
- An 80-layer model under the same rule: 4.2× — nearly free.

vLLM's own code comments document that removing boundary protection costs ~30 GSM8K points at aggressive presets on Qwen3-4B, while at `k8v4` the effect is expected to be negligible — a strong candidate **preset × layer-protection interaction** the screening experiment is designed to capture.

vLLM's hard-coded rule protects layers by *position* (first 2 + last 2). Community measurements suggest position is only a proxy: per-layer outlier-channel fractions vary from ~20% (layer 0) to 4–6% (middle layers), and not monotonically. This motivates the project's central hypothesis — stated in two parts, **both of which are tested, not assumed**:

> **H1a (feature validity):** cheap per-layer statistics computed from one calibration forward pass correlate strongly (rank correlation) with *measured* per-layer quantization sensitivity, and the correlation is stable across calibration domains.
>
> **H1b (policy value):** a protection set ranked by validated features achieves better accuracy than positional protection at the *same* protection budget — and therefore a better accuracy/compression frontier.

**Candidate features** (each targeting a specific failure mode of TurboQuant's math): (1) key outlier-channel fraction — violates the N(0, 1/d) Gaussian assumption under which vLLM's Lloyd-Max centroids are solved (empirical basis: LLM.int8(), KIVI/KVQuant show keys carry systematic, input-persistent channel outliers); (2) post-WHT excess kurtosis of key coordinates — the direct Gaussianity test (precedent: kurtosis-aware mixed precision in community TurboQuant ports); (3) key-norm magnitude/dispersion — logit error scales with ‖k‖·‖q‖ × angular quantization error; (4) per-vector value dynamic range — uniform value quantization error is proportional to (max−min); (5) **direct simulated quantization error** — TurboQuant's exact math (Hadamard + vLLM's own centroids + uniform values) applied offline to captured K/V, measuring per-layer attention-output error.

**Validation protocol (Experiment 0):** ground-truth per-layer sensitivity is measured by simulated quantization of one layer at a time (ΔPPL per layer); H1a is then Spearman correlation of each feature against this curve, plus a stability report (rank correlation of feature-derived rankings across disjoint, cross-domain calibration slices). The ranking policy is built **only from features that pass both tests**. H1b is then tested end-to-end on real vLLM in Experiment 1 via matched-budget cells — necessary because simulated quantization reproduces vLLM's math but not its kernels.

**Fallback:** if no feature passes (H1a fails) or the ranking does not beat position at matched budgets (H1b fails), KVCompressionTune ships with tuned positional protection — a negative result on H1 is itself a reportable finding, and the system's framing is unchanged either way.

---

## 4. Workload Profiles & Utility Functions

The system is characterized and tuned separately for three workload profiles.

### 4.0 Normalization Framework

Raw metrics have incompatible units, so a weighted sum of their inverses is dimensionally meaningless. Every metric is normalized against the **uncompressed (auto/FP16 KV) baseline** measured on the *same model, workload, and GPU*, yielding dimensionless gain ratios where higher is better and 1.0 means "identical to baseline":

$$S_{TPOT} = \frac{TPOT_{base}}{TPOT_{cfg}}, \qquad S_{TTFT} = \frac{TTFT_{base}}{TTFT_{cfg}}, \qquad S_{TP} = \frac{TP_{cfg}}{TP_{base}}, \qquad R_{mem} = \frac{KV_{base}}{KV_{cfg}}$$

**Accuracy is a hard constraint, not a weighted term.** Using $1/\Delta PPL$ in a sum is numerically unstable (it explodes as $\Delta PPL \to 0$). Each profile defines a maximum tolerated relative perplexity degradation $\delta$:

$$\frac{PPL_{cfg} - PPL_{base}}{PPL_{base}} \le \delta$$

Trials violating the constraint (or crashing with OOM) are **pruned** by Optuna (assigned $Utility = 0$), so the search only ranks configurations that are already accurate enough.

Each profile's utility is a **weighted geometric mean** of the relevant gain ratios — scale-free, so the exponents act as true relative-importance weights. Complementarily, the tuner also runs in Optuna's **multi-objective mode (NSGA-II)** over $(S_{speed}, R_{mem}, \Delta PPL)$ to expose the full Pareto frontier per profile; the scalar utilities are the operating-point selectors reported in the paper.

### 4.1 Profile A: Interactive Chat

- **Characteristics:** short input (< 100 tokens), long output (200–500 tokens). Sensitive to token-generation speed.

$$Utility_{Chat} = S_{TPOT}^{\,0.7} \cdot R_{mem}^{\,0.3} \qquad \text{s.t. } \Delta PPL / PPL_{base} \le 0.5\%$$

### 4.2 Profile B: Retrieval-Augmented Generation (RAG)

- **Characteristics:** long input (4,000–16,000 tokens), short output (20–50 tokens). Memory-bound; OOM risk; TTFT-sensitive.

$$Utility_{RAG} = R_{mem}^{\,0.5} \cdot S_{TTFT}^{\,0.5} \qquad \text{s.t. } \Delta PPL / PPL_{base} \le 1\%$$

### 4.3 Profile C: Bulk Offline Batch Processing

- **Characteristics:** thousands of documents, no human waiting; total completion rate is the only latency metric that matters.

$$Utility_{Batch} = S_{TP}^{\,0.8} \cdot R_{mem}^{\,0.2} \qquad \text{s.t. } \Delta PPL / PPL_{base} \le 2\%$$

---

## 5. Metrics & Measurement Infrastructure

The benchmark harness collects five hard metrics per run:

| # | Metric | Unit | What it measures |
|---|--------|------|------------------|
| 1 | **TTFT** (Time to First Token) | ms | prefill phase |
| 2 | **TPOT** (Time Per Output Token) | ms | decoding phase |
| 3 | **Throughput** | tokens/sec | completed work under concurrent load |
| 4 | **KV-cache memory / peak VRAM** | GB | from vLLM's own accounting + `torch.cuda` peak stats |
| 5 | **Perplexity (PPL)** | — | WikiText-2, verifying language quality |

Latency metrics are reported as mean, p95, and p99. Because the characterization showed layer-protection effects manifest on *reasoning* rather than perplexity, a lightweight **GSM8K subset (200 questions)** accuracy check is added for configurations on the recommendation shortlist.

### 5.1 Statistical Methodology

All comparative claims in the final report follow this protocol:

- **Paired experimental design.** Each workload trace (exact prompt sequence, lengths, arrival times) is generated once with a fixed seed and frozen into the repository. Every configuration replays the *identical* trace, so comparisons are paired per-request.
- **Repetitions.** Screening cells (Experiment 1) run **n = 2** times with distinct seeds — enough for effect directions and magnitudes; validation cells (Experiment 3), where the headline claims are made, run **n = 5**. The first *k* requests of each run are discarded as warm-up.
- **Reporting.** Scalar metrics as **mean ± 95% CI** (Student's *t*, *n−1* df); latencies additionally as p50/p95/p99 over pooled per-request measurements.
- **Screening analysis.** Main effects and two-way interactions via **factorial ANOVA** on the screening grid; parameter importance via **fANOVA** over all collected trials. These two analyses *are* the characterization deliverable.
- **Significance testing.** Default-vs-tuned differences via **Wilcoxon signed-rank** on paired per-request latencies and paired *t*-tests on run-level scalars, with effect sizes (Cliff's delta / Cohen's *d*).
- **Multiple-comparison correction.** **Holm–Bonferroni** across the metric × profile family; only corrected p < 0.05 is claimed significant.
- **Raw data retention.** All per-request measurements go to CSV; the entire analysis is a deterministic script (`analysis/stats.py`) over those CSVs.

---

## 6. Experimental Program

Three experiments, each a standalone deliverable, each feeding the next:

### Experiment 0 — Layer-Sensitivity Ground Truth & Feature Study (H1a) — ≈ 2–3 GPU-h

- **Ground truth:** simulated TurboQuant quantization (vLLM's exact math, ported to HF hooks) applied to **one layer at a time**; ΔPPL per layer on a small eval set → the true per-layer sensitivity curve (~L cheap evaluations, no vLLM engine involved).
- **Feature study:** each §3.4 candidate feature computed from one calibration pass; Spearman correlation against the ground-truth curve; stability analysis across disjoint cross-domain calibration slices (encyclopedic / code / dialogue) and sample sizes.
- **Output:** validated feature set + the ranking policy used by the profiler — or a documented H1a rejection with fallback to positional protection. The sensitivity curve and correlation table are report figures regardless of outcome.

### Experiment 1 — Configuration-Space Characterization (the map) — ≈ 8–12 GPU-h

- **Design:** compact factorial screening on the primary (model, GPU): **4 presets × 6 layer-protection sets** — positional n ∈ {0, 1, 2, 4} plus **statistics-guided sets at budgets matched to n=2 and n=4** (same number of protected layers, layers chosen by the profiler's ranking; this is the direct H1 test) = 24 core configs, plus a **4-config kv-splits arm** (latency-only knob, tested for the no-interaction assumption rather than crossed with the full grid). Serving knobs are fixed at vLLM defaults — they are not TurboQuant parameters. Short screening traces (~30–50 requests per profile), n = 2 repetitions; every cell measures all three profiles and PPL in a single engine session (~12–15 min/cell).
- **Analysis:** main effect per parameter per metric; two-way interaction terms (explicitly including preset × layer-protection); fANOVA variance decomposition.
- **Output:** the empirical sensitivity map — *which parameters matter, by how much, for which metric, and which interactions exist.* This is the report's centerpiece figure set and satisfies the course's ablation-study requirement by construction.

### Experiment 2 — Auto-Tuning (the system) — ≈ 8 GPU-h

- Optuna TPE search per profile on the primary (model, GPU) over **(preset, protection budget k under the profiler's ranking, kv-splits)**, **warm-started with all Experiment-1 cells as completed trials**, adding ~10–15 refinement trials per profile on the dimensions Experiment 1 proved significant; parameters that measured flat are frozen at defaults, with that decision documented by data.
- Hard PPL constraint + OOM pruning per Section 4.0; NSGA-II over the pooled trials exposes Pareto frontiers.
- **Output:** `optimal_configs.json` + convergence and Pareto plots; fANOVA importance recomputed over all trials as a headline result ("parameter X explains Y% of utility variance").

### Experiment 3 — Validation (the payoff) — ≈ 6 GPU-h

- Head-to-head under the full statistical protocol (n = 5, full-length **held-out** traces not used during screening or tuning), per profile: **uncompressed baseline** vs. **best vLLM default preset** (hard-coded positional n=2 rule) vs. **AutoTune with positional protection** vs. **AutoTune with statistics-guided protection** — separating the value of tuning from the value of the H1 policy.
- **Output:** relative improvements with corrected significance — the "claim to fame" numbers.

### Experiment 4 — Generalization spot-check — ≈ 3 GPU-h

- The tuned configuration and ~5 grid cells re-run on a second model (and/or second GPU generation). The only claim: **the sensitivity map and the optimum differ across contexts** — establishing that auto-tuning per context is necessary, without paying for a second full map.

---

## 7. Directory Tree & Code Structure

```
├── docker/
│   └── Dockerfile               # CUDA + stock vLLM (pinned) + deps
├── configs/
│   ├── grids/                   # Experiment-1 screening grids (YAML)
│   └── profiles.yaml            # workload profile definitions
├── data/
│   ├── wikitext/                # PPL reference data
│   └── traces/                  # frozen workload traces (seeded)
├── src/
│   ├── __init__.py
│   ├── profiler.py              # per-layer sensitivity scoring (H1 policy)
│   ├── harness.py               # vLLM engine launch + benchmark client
│   ├── workloads.py             # trace generation (Chat / RAG / Batch)
│   ├── metrics.py               # TTFT, TPOT, throughput, VRAM, PPL, GSM8K-200
│   ├── characterize.py          # Experiment 1: grid runner
│   ├── tuner.py                 # Experiment 2: Optuna search
│   └── advisor.py               # config lookup + recommendation CLI
├── analysis/
│   ├── stats.py                 # ANOVA, fANOVA, CIs, significance tests
│   └── plots.py                 # all report figures, regenerable
├── tests/
│   ├── test_metrics.py          # measurement correctness
│   ├── test_workloads.py        # trace determinism
│   └── test_advisor.py          # config selection logic
├── run_benchmark.py             # end-to-end: Exp 1 → 2 → 3 → figures
├── results/                     # CSV logs (committed samples)
└── requirements.txt
```

---

## 8. Implementation Roadmap

### Phase 1: Infrastructure (Days 1–3)
- **1.1** Dockerfile with pinned stock vLLM; verify all four TurboQuant presets + skip-layers flag work on each cluster GPU type with a small model.
- **1.2** Smoke-test measurement of one full config cell end-to-end.

### Phase 2: Harness & Instrumentation (Days 4–7)
- **2.1** `workloads.py`: seeded, frozen trace generation for the three profiles.
- **2.2** `metrics.py`: TTFT/TPOT via `time.perf_counter()`, VRAM via vLLM accounting + `torch.cuda.max_memory_allocated()`, WikiText-2 PPL, GSM8K-200 harness.
- **2.3** `harness.py`: engine lifecycle + trace replay + CSV emission; unit tests for trace determinism and metric correctness.
- **2.4** `profiler.py`: HF hook-based K/V capture (post-RoPE), streaming feature computation, and the simulated-TurboQuant module (vLLM math ported verbatim, validated against vLLM's `centroids.py` outputs in a unit test); run Experiment 0 (ground truth + feature correlations + stability).

### Phase 3: Experiment 1 — Characterization (Days 8–13)
- **3.1** `characterize.py`: grid runner with resumability (cluster jobs fail).
- **3.2** Run the compact screening grid (~28 cells × 2 reps) on the primary (model, GPU).
- **3.3** `analysis/stats.py`: ANOVA + fANOVA; produce the sensitivity map and interaction figures. **Checkpoint: decide the tuner's search space from data.**

### Phase 4: Experiment 2 — Auto-Tuner (Days 14–17)
- **4.1** `tuner.py`: Optuna objective with PPL constraint + OOM pruning; TPE and NSGA-II modes.
- **4.2** Run per (model, GPU, profile); export `optimal_configs.json`; convergence + Pareto plots.
- **4.3** `advisor.py`: recommendation CLI with evidence trail.

### Phase 5: Experiment 3 + Report (Days 18–21)
- **5.1** Held-out validation runs, full statistical protocol.
- **5.2** `run_benchmark.py` end-to-end automation; regenerate every figure from raw CSVs.
- **5.3** Final report (8–12 pages) per course structure.

---

## 9. Bonus Track (time-permitting): Upstream Contribution

The characterization will likely identify parameters whose *frozen* status in vLLM is unjustified — prime candidates: generic `turboquant_k{K}v{V}[_nc]` preset parsing (the config dataclass and kernels already support the combinations) and exposing the boundary width `n` currently hard-coded at its call site. If Experiment 1 shows these matter, a small, test-covered PR to vLLM is prepared, using our measurements as motivation. This targets the course's exceptional-grade criterion for adopted open-source contributions — but the core project stands entirely without it.

---

## 10. Success Criteria Mapping

| Criterion | Weight | How this project satisfies it |
|-----------|--------|-------------------------------|
| **Correctness** | 40% | Unit tests for metric computation, trace determinism, and advisor logic; PPL/GSM8K guards verify quantized runs produce sane model output; all experiments run on unmodified vLLM, so cache correctness itself rests on vLLM's own tested implementation — our correctness surface is the measurement and decision code, and it is fully tested. |
| **Reproducibility** | 30% | Stock vLLM (pinned version) + Docker + frozen seeded traces + one-command `run_benchmark.py`; reproducible on any CUDA machine with a supported GPU. No forks or patches to reproduce against. |
| **Performance Gain** | 15% | Experiment 3's head-to-head vs. vLLM defaults with paired tests, effect sizes, and Holm–Bonferroni-corrected significance, on held-out traces. |
| **Clarity** | 15% | The report follows the research arc — map the space, understand it, then tune it — with every figure regenerable from committed raw data; docstrings and typed code throughout. |

### 10.1 Future Work (grounded)

- Per-request configuration within a single engine (requires vLLM allocator + kernel changes — the natural sequel once the per-deployment tuner proves the configs differ).
- Calibrated, per-model Lloyd-Max centroids (vLLM's centroids assume Gaussian rotated coordinates; outlier-heavy layers violate this).
- Non-boundary layer-protection sets guided by per-layer outlier statistics rather than position.
