"""Benchmark harness: run one experiment cell (engine config × traces × PPL).

A *cell* is one TurboQuant configuration evaluated on all workload traces plus
perplexity, in a single engine session. Cells are content-addressed: the cell
config is hashed, and results/cells/<hash>.json is the checkpoint — rerunning
a manifest skips completed cells (SPEC §6 resumability).

Each cell runs in a fresh subprocess (clean VRAM, crash isolation), same
pattern as the smoke test.

CLI:
    # one cell, inline config:
    python -m src.harness --kv-cache-dtype turboquant_3bit_nc --rep 0

    # a manifest of cells (JSON list of cell configs):
    python -m src.harness --manifest configs/grids/exp1.json
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

RESULT_MARKER = "===CELL_RESULT_JSON==="
DEFAULT_TRACE_DIR = "data/traces"
CELL_TIMEOUT_S = 3600


@dataclass(frozen=True)
class CellConfig:
    """Everything that identifies one experiment cell (hash = checkpoint key)."""

    model: str = "Qwen/Qwen3-4B"
    kv_cache_dtype: str = "auto"
    skip_layers: tuple[int | str, ...] = ()  # user-supplied additions (floor is implicit)
    max_model_len: int = 10240
    gpu_memory_utilization: float = 0.85
    chunk_size: int = 512                  # max_num_batched_tokens (chunked prefill)
    trace_tag: str = "screen"
    trace_seed: int = 20260714
    rep: int = 0                           # repetition index (distinct engine seed)
    ppl_docs: int = 4

    def cell_hash(self) -> str:
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def result_path(cfg: CellConfig, out_dir: str | Path = "results/cells") -> Path:
    return Path(out_dir) / f"{cfg.cell_hash()}.json"


# --------------------------------------------------------------------------
# In-subprocess execution of one cell
# --------------------------------------------------------------------------

def run_cell_inprocess(cfg: CellConfig) -> dict:
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    from .metrics import (  # noqa: PLC0415
        PPLConfig, RequestLatency, VramSampler, compute_ppl,
        extract_latencies, summarize_latencies,
    )
    from .workloads import PROFILES, load_trace  # noqa: PLC0415

    result: dict = {
        "config": dataclasses.asdict(cfg),
        "cell_hash": cfg.cell_hash(),
        "ok": False,
    }

    with VramSampler() as vram:
        t0 = time.perf_counter()
        engine_kwargs: dict = dict(
            model=cfg.model,
            kv_cache_dtype=cfg.kv_cache_dtype,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            enable_chunked_prefill=True,
            max_num_batched_tokens=cfg.chunk_size,
            seed=1000 + cfg.rep,
            disable_log_stats=False,
        )
        if cfg.skip_layers:
            engine_kwargs["kv_cache_dtype_skip_layers"] = [str(x) for x in cfg.skip_layers]
        llm = LLM(**engine_kwargs)
        result["engine_load_s"] = round(time.perf_counter() - t0, 2)

        try:
            cc = llm.llm_engine.vllm_config.cache_config
            result["effective_skip_layers"] = list(cc.kv_cache_dtype_skip_layers)
        except AttributeError:
            result["effective_skip_layers"] = None

        # --- Workload traces (all profiles, one engine session) ---
        result["profiles"] = {}
        for profile in PROFILES:
            trace_file = (Path(DEFAULT_TRACE_DIR)
                          / f"{profile}_{cfg.trace_tag}_seed{cfg.trace_seed}.json")
            trace = load_trace(trace_file)
            prompts = [r["prompt"] for r in trace["requests"]]
            sps = [SamplingParams(temperature=0.0, max_tokens=r["max_tokens"])
                   for r in trace["requests"]]
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, sps)
            wall = time.perf_counter() - t0

            lats = extract_latencies(outputs)
            summary = summarize_latencies(lats)
            summary["wall_s"] = round(wall, 2)
            summary["throughput_tok_s"] = round(
                summary["total_output_tokens"] / wall, 1)
            if profile == "rag":
                hits = sum(
                    1 for req, out in zip(trace["requests"], outputs)
                    if _extract_code(req["prompt"]) in out.outputs[0].text
                )
                summary["needle_accuracy"] = hits / len(outputs)
            result["profiles"][profile] = summary

        # --- Perplexity (KV-cache-sensitive; see metrics.py docstring) ---
        result["ppl"] = compute_ppl(
            llm, PPLConfig(n_docs=cfg.ppl_docs, chunk_size=cfg.chunk_size))

    result["peak_vram_gb"] = round(vram.peak_gb, 2)
    result["baseline_vram_gb"] = round(vram.baseline_gb, 2)
    result["ok"] = True
    return result


def _extract_code(prompt: str) -> str:
    """Recover the needle code embedded by workloads.gen_rag."""
    marker = "internal reference code for this document is "
    start = prompt.index(marker) + len(marker)
    return prompt[start:start + 7].rstrip(".")


# --------------------------------------------------------------------------
# Parent orchestration: subprocess per cell + checkpointing
# --------------------------------------------------------------------------

def run_cell(cfg: CellConfig, out_dir: str | Path = "results/cells",
             force: bool = False) -> dict:
    """Run one cell in a subprocess, unless its checkpoint already exists."""
    path = result_path(cfg, out_dir)
    if path.exists() and not force:
        print(f"[{cfg.cell_hash()}] checkpoint exists — skipping", flush=True)
        return json.loads(path.read_text())

    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "src.harness",
           "--run-single-json", json.dumps(dataclasses.asdict(cfg))]
    print(f"[{cfg.cell_hash()}] {cfg.kv_cache_dtype} skip+{list(cfg.skip_layers)} "
          f"rep{cfg.rep} — launching", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=CELL_TIMEOUT_S)
    wall = round(time.perf_counter() - t0, 1)

    if RESULT_MARKER in proc.stdout:
        payload = proc.stdout.split(RESULT_MARKER, 1)[1].strip()
        result = json.loads(payload.splitlines()[0])
    else:
        result = {
            "config": dataclasses.asdict(cfg),
            "cell_hash": cfg.cell_hash(),
            "ok": False,
            "error": (proc.stderr or proc.stdout)[-4000:],
        }
    result["cell_wall_s"] = wall
    path.write_text(json.dumps(result, indent=1))
    print(f"[{cfg.cell_hash()}] {'OK' if result['ok'] else 'FAILED'} in {wall}s",
          flush=True)
    return result


def run_manifest(manifest_path: str | Path, out_dir: str | Path = "results/cells",
                 force: bool = False) -> None:
    cells = json.loads(Path(manifest_path).read_text())
    print(f"manifest: {len(cells)} cells", flush=True)
    n_ok = 0
    for entry in cells:
        entry["skip_layers"] = tuple(entry.get("skip_layers", ()))
        cfg = CellConfig(**entry)
        res = run_cell(cfg, out_dir, force=force)
        n_ok += bool(res.get("ok"))
    print(f"done: {n_ok}/{len(cells)} ok", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-single-json", default=None,
                   help="internal: run one cell in-process from a JSON config")
    p.add_argument("--manifest", default=None, help="JSON list of cell configs")
    p.add_argument("--force", action="store_true", help="ignore checkpoints")
    # Inline single-cell flags:
    p.add_argument("--kv-cache-dtype", default="auto")
    p.add_argument("--skip-layers", nargs="*", default=[])
    p.add_argument("--model", default=CellConfig.model)
    p.add_argument("--rep", type=int, default=0)
    args = p.parse_args()

    if args.run_single_json:
        entry = json.loads(args.run_single_json)
        entry["skip_layers"] = tuple(entry.get("skip_layers", ()))
        result = run_cell_inprocess(CellConfig(**entry))
        print(RESULT_MARKER)
        print(json.dumps(result))
    elif args.manifest:
        run_manifest(args.manifest, force=args.force)
    else:
        cfg = CellConfig(model=args.model, kv_cache_dtype=args.kv_cache_dtype,
                         skip_layers=tuple(args.skip_layers), rep=args.rep)
        run_cell(cfg, force=args.force)


if __name__ == "__main__":
    main()
