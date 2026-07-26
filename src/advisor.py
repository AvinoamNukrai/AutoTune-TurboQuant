"""Config Advisor — recommend vLLM launch flags for a given workload.

Loads the optimal configs produced by the tuner (results/optimal_configs.json)
and emits the vLLM CLI flags or Python kwargs, along with an evidence trail
explaining *why* this config was selected.

Usage:
    python -m src.advisor --profile chat
    python -m src.advisor --profile rag --format python
    python -m src.advisor --profile batch --format cli
    python -m src.advisor --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_CONFIGS_PATH = Path("results/optimal_configs.json")

PROFILE_DESCRIPTIONS = {
    "chat": "Interactive chat — short input, long output, TPOT-sensitive (dPPL ≤ 0.5%)",
    "rag": "Retrieval-augmented generation — long input, short output, memory+TTFT (dPPL ≤ 1.0%)",
    "batch": "Bulk offline processing — throughput-maximizing (dPPL ≤ 2.0%)",
}

PPL_THRESHOLDS = {"chat": 0.5, "rag": 1.0, "batch": 2.0}


def load_configs(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: {path} not found. Run the tuner first:", file=sys.stderr)
        print(f"  python -m src.tuner --optimize", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text())


def format_cli(cfg: dict, model: str) -> str:
    parts = [
        "vllm serve", model,
        f"--kv-cache-dtype {cfg['kv_cache_dtype']}",
    ]
    if cfg.get("skip_layers"):
        layers = " ".join(str(x) for x in cfg["skip_layers"])
        parts.append(f"--kv-cache-dtype-skip-layers {layers}")
    parts.append("--enable-chunked-prefill")
    return " \\\n    ".join(parts)


def format_python(cfg: dict, model: str) -> str:
    kwargs = [
        f'    model="{model}",',
        f'    kv_cache_dtype="{cfg["kv_cache_dtype"]}",',
    ]
    if cfg.get("skip_layers"):
        layers = [str(x) for x in cfg["skip_layers"]]
        kwargs.append(f"    kv_cache_dtype_skip_layers={layers},")
    kwargs.append("    enable_chunked_prefill=True,")
    return "LLM(\n" + "\n".join(kwargs) + "\n)"


def print_recommendation(profile: str, cfg: dict, model: str,
                         output_format: str) -> None:
    desc = PROFILE_DESCRIPTIONS.get(profile, profile)
    print(f"{'=' * 70}")
    print(f"RECOMMENDATION: {profile.upper()}")
    print(f"  {desc}")
    print(f"{'=' * 70}")

    if cfg.get("viable") is False:
        print(f"\n  No viable config found: {cfg.get('reason', 'unknown')}")
        print(f"  Recommendation: use FP16 baseline (--kv-cache-dtype auto)")
        return

    print(f"\n  Preset:     {cfg['kv_cache_dtype']}")
    skip = cfg.get("skip_layers", [])
    eff = cfg.get("effective_skip_layers", [])
    print(f"  Extra skip: {skip if skip else '(none — floor only)'}")
    print(f"  Effective:  {eff} ({len(eff)} layers protected)")
    print(f"  R_mem:      {cfg.get('r_mem', '?')}x compression")

    print(f"\n  --- Evidence ---")
    ppl_d = cfg.get("ppl_delta_pct")
    threshold = PPL_THRESHOLDS.get(profile)
    if ppl_d is not None:
        status = "PASS" if abs(ppl_d) <= threshold else "FAIL"
        print(f"  PPL delta:  {ppl_d:+.2f}% (threshold: ≤{threshold}%) [{status}]")
    print(f"  Utility:    {cfg.get('utility', '?')}")

    ev = cfg.get("evidence", {})
    if profile == "chat" and ev.get("chat_tpot_ms"):
        print(f"  Chat TPOT:  {ev['chat_tpot_ms']:.2f} ms")
    if profile == "rag":
        if ev.get("rag_ttft_ms"):
            print(f"  RAG TTFT:   {ev['rag_ttft_ms']:.1f} ms")
        if ev.get("needle_acc") is not None:
            print(f"  Needle acc: {ev['needle_acc']:.2f}")
    if profile == "batch" and ev.get("batch_tps"):
        print(f"  Batch TPS:  {ev['batch_tps']:.1f} tok/s")

    print(f"\n  --- Launch command ---")
    if output_format == "cli":
        print(f"  {format_cli(cfg, model)}")
    else:
        print(f"  {format_python(cfg, model)}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", choices=["chat", "rag", "batch"],
                   help="workload profile to recommend for")
    p.add_argument("--all", action="store_true",
                   help="show recommendations for all profiles")
    p.add_argument("--format", choices=["cli", "python"], default="cli",
                   dest="output_format",
                   help="output format: vLLM CLI flags or Python kwargs")
    p.add_argument("--model", default="Qwen/Qwen3-4B",
                   help="model name for the launch command")
    p.add_argument("--configs", default=str(DEFAULT_CONFIGS_PATH),
                   help="path to optimal_configs.json")
    p.add_argument("--json", action="store_true",
                   help="output raw JSON instead of formatted text")
    args = p.parse_args()

    if not args.profile and not args.all:
        p.error("specify --profile or --all")

    configs = load_configs(Path(args.configs))
    profiles = ["chat", "rag", "batch"] if args.all else [args.profile]

    if args.json:
        out = {pr: configs[pr] for pr in profiles if pr in configs}
        print(json.dumps(out, indent=2))
        return

    baseline = configs.get("baseline", {})
    if baseline:
        print(f"Baseline PPL: {baseline.get('ppl', '?')}")
        print()

    for profile in profiles:
        cfg = configs.get(profile)
        if cfg is None:
            print(f"No config found for profile '{profile}'")
            continue
        print_recommendation(profile, cfg, args.model, args.output_format)


if __name__ == "__main__":
    main()
