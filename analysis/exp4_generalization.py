"""Experiment 4 — Multi-model generalization spot-check.

Compares Qwen3-4B (primary) against multiple spot-check models on the same
configs, showing that the sensitivity map and optimum differ across models
and architectures. Establishes that per-context auto-tuning is necessary.

Usage:
    python -m analysis.exp4_generalization [--cells-dir results/cells]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

EXP3_MANIFEST = Path("configs/grids/exp3.json")
EXP4_MANIFESTS = [
    Path("configs/grids/exp4_all.json"),
    Path("configs/grids/exp4.json"),
]

PROFILES = ["chat", "rag", "batch"]
PPL_THRESHOLDS = {"chat": 0.005, "rag": 0.01, "batch": 0.02}

BYTES_PER_KV = {
    "auto": 4.0,
    "turboquant_k8v4": 1.5,
    "turboquant_4bit_nc": 1.0,
    "turboquant_k3v4_nc": 0.875,
    "turboquant_3bit_nc": 0.75,
}


def config_key(cfg: dict) -> tuple:
    return (cfg["kv_cache_dtype"],
            tuple(sorted(int(x) for x in cfg.get("skip_layers", []))))


def detect_n_layers(cells: list[dict]) -> int | None:
    for c in cells:
        eff = c.get("effective_skip_layers", [])
        if eff and len(eff) >= 4:
            return max(int(x) for x in eff) + 1
    return None


def load_manifest_cells(manifest_path: Path, cells_dir: Path) -> list[dict]:
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text())
    from src.harness import CellConfig
    hashes = set()
    for entry in manifest:
        e = dict(entry)
        e["skip_layers"] = tuple(e.get("skip_layers", ()))
        hashes.add(CellConfig(**e).cell_hash())

    results = []
    for f in sorted(cells_dir.glob("*.json")):
        if f.stem in hashes:
            data = json.loads(f.read_text())
            if data.get("ok"):
                results.append(data)
    return results


def extract_metrics(cell: dict) -> dict:
    chat = cell.get("profiles", {}).get("chat", {})
    rag = cell.get("profiles", {}).get("rag", {})
    batch = cell.get("profiles", {}).get("batch", {})
    tpot = chat.get("tpot_s", {}).get("mean")
    ttft_rag = rag.get("ttft_s", {}).get("mean")
    return {
        "ppl": cell.get("ppl", {}).get("ppl"),
        "chat_tpot_ms": tpot * 1000 if tpot else None,
        "chat_tps": chat.get("throughput_tok_s"),
        "rag_ttft_ms": ttft_rag * 1000 if ttft_rag else None,
        "needle_acc": rag.get("needle_accuracy"),
        "batch_tps": batch.get("throughput_tok_s"),
    }


def compute_r_mem(dtype: str, skip: tuple, n_layers: int) -> float:
    if dtype == "auto":
        return 1.0
    b = BYTES_PER_KV[dtype]
    floor = {0, 1, n_layers - 2, n_layers - 1}
    n_protect = len(floor | set(int(x) for x in skip))
    return (n_layers * 4.0) / (n_protect * 4.0 + (n_layers - n_protect) * b)


def compute_utility(profile: str, m: dict, base: dict,
                    dtype: str, skip: tuple, n_layers: int) -> float | None:
    ppl = m.get("ppl")
    base_ppl = base.get("ppl")
    if ppl is None or base_ppl is None or base_ppl == 0:
        return None
    delta = (ppl - base_ppl) / base_ppl
    if delta > PPL_THRESHOLDS[profile]:
        return 0.0
    r_mem = compute_r_mem(dtype, skip, n_layers)

    if profile == "chat":
        bt = base.get("chat_tpot_ms")
        ct = m.get("chat_tpot_ms")
        if not bt or not ct or ct == 0:
            return None
        return (bt / ct) ** 0.7 * r_mem ** 0.3
    if profile == "rag":
        bt = base.get("rag_ttft_ms")
        ct = m.get("rag_ttft_ms")
        if not bt or not ct or ct == 0:
            return None
        return r_mem ** 0.5 * (bt / ct) ** 0.5
    if profile == "batch":
        bt = base.get("batch_tps")
        ct = m.get("batch_tps")
        if not bt or bt == 0:
            return None
        return (ct / bt) ** 0.8 * r_mem ** 0.2 if ct else None
    return None


def fmt(v, digits=2):
    return f"{v:.{digits}f}" if v is not None else "N/A"


def short_model_name(model: str) -> str:
    return model.split("/")[-1]


def analyze_model(cells: list[dict], n_layers: int):
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for c in cells:
        key = config_key(c["config"])
        grouped[key].append(extract_metrics(c))

    averaged: dict[tuple, dict] = {}
    for key, reps in grouped.items():
        avg = {}
        for field in ["ppl", "chat_tpot_ms", "chat_tps", "rag_ttft_ms",
                       "needle_acc", "batch_tps"]:
            vals = [r[field] for r in reps if r[field] is not None]
            avg[field] = sum(vals) / len(vals) if vals else None
        averaged[key] = avg

    baseline_key = ("auto", ())
    base = averaged.get(baseline_key, {})

    results = {}
    for key in sorted(averaged.keys()):
        dtype, skip = key
        m = averaged[key]
        label = dtype.replace("turboquant_", "")
        if skip:
            label += f"+L{','.join(str(s) for s in skip)}"

        r_mem = compute_r_mem(dtype, skip, n_layers)
        dppl = ((m["ppl"] - base["ppl"]) / base["ppl"] * 100
                if m.get("ppl") and base.get("ppl") else None)

        row = {"label": label, "ppl": m["ppl"], "dppl_pct": dppl, "r_mem": r_mem}
        for profile in PROFILES:
            u = compute_utility(profile, m, base, dtype, skip, n_layers)
            row[f"util_{profile}"] = u
        results[key] = row

    return results, base


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells-dir", default="results/cells")
    args = p.parse_args()
    cells_dir = Path(args.cells_dir)

    # Load primary model (Qwen3-4B from Exp 3)
    exp3_cells = load_manifest_cells(EXP3_MANIFEST, cells_dir)
    primary, _ = analyze_model(exp3_cells, n_layers=36)

    # Load all spot-check cells from any exp4 manifest
    all_exp4_cells = []
    for manifest in EXP4_MANIFESTS:
        all_exp4_cells.extend(load_manifest_cells(manifest, cells_dir))

    # Deduplicate by cell_hash
    seen = set()
    deduped = []
    for c in all_exp4_cells:
        h = c.get("cell_hash", id(c))
        if h not in seen:
            seen.add(h)
            deduped.append(c)
    all_exp4_cells = deduped

    if not all_exp4_cells:
        print("No Exp 4 cells found. Run the manifest first.")
        return

    # Group by model
    models: dict[str, list[dict]] = defaultdict(list)
    for c in all_exp4_cells:
        model = c["config"]["model"]
        models[model].append(c)

    model_names = sorted(models.keys())
    short_names = {m: short_model_name(m) for m in model_names}

    print(f"Primary: Qwen3-4B ({len(exp3_cells)} cells)")
    for m in model_names:
        print(f"Spot-check: {short_names[m]} ({len(models[m])} cells)")
    print()

    # Analyze each model
    model_results = {}
    for model in model_names:
        cells = models[model]
        n_layers = detect_n_layers(cells)
        if n_layers is None:
            print(f"WARNING: Could not detect layer count for {short_names[model]}, skipping")
            continue
        results, _ = analyze_model(cells, n_layers)
        model_results[model] = {"results": results, "n_layers": n_layers}

    if not model_results:
        print("No valid models found.")
        return

    # -----------------------------------------------------------------------
    # PPL sensitivity comparison across all models
    # -----------------------------------------------------------------------
    all_models = ["Qwen/Qwen3-4B"] + list(model_results.keys())
    all_results = {"Qwen/Qwen3-4B": primary}
    all_results.update({m: model_results[m]["results"] for m in model_results})
    all_short = {"Qwen/Qwen3-4B": "Qwen3-4B"}
    all_short.update(short_names)

    common_keys = set(primary.keys())
    for m in model_results:
        common_keys &= set(model_results[m]["results"].keys())
    common_keys = sorted(common_keys)

    col_w = 12
    header_models = "".join(f"{all_short[m]:>{col_w}}" for m in all_models)

    print("=" * (20 + col_w * len(all_models) * 2 + 10))
    print("dPPL% BY CONFIG × MODEL")
    print("=" * (20 + col_w * len(all_models) * 2 + 10))
    print(f"{'Config':<20}" + header_models)
    print("-" * (20 + col_w * len(all_models)))

    for key in common_keys:
        label = all_results[all_models[0]][key]["label"]
        vals = ""
        for m in all_models:
            dppl = all_results[m][key].get("dppl_pct")
            vals += f"{fmt(dppl, 2) + '%':>{col_w}}"
        print(f"{label:<20}{vals}")
    print()

    # -----------------------------------------------------------------------
    # Per-profile utility × model matrix
    # -----------------------------------------------------------------------
    for profile in PROFILES:
        print("=" * (20 + col_w * len(all_models) + 20))
        print(f"UTILITY — {profile.upper()} (threshold dPPL ≤ {PPL_THRESHOLDS[profile]*100:.1f}%)")
        print("=" * (20 + col_w * len(all_models) + 20))
        print(f"{'Config':<20}" + header_models + f"{'  Optimal on':>{20}}")
        print("-" * (20 + col_w * len(all_models) + 20))

        best_per_model = {}
        for m in all_models:
            best_u = -1
            best_label = "?"
            for key in common_keys:
                u = all_results[m][key].get(f"util_{profile}")
                if u is not None and u > best_u:
                    best_u = u
                    best_label = all_results[m][key]["label"]
            best_per_model[m] = best_label

        for key in common_keys:
            label = all_results[all_models[0]][key]["label"]
            vals = ""
            wins = []
            for m in all_models:
                u = all_results[m][key].get(f"util_{profile}")
                vals += f"{fmt(u, 4):>{col_w}}"
                if best_per_model[m] == label:
                    wins.append(all_short[m])
            win_str = ", ".join(wins) if wins else ""
            print(f"{label:<20}{vals}  {win_str:>{18}}")

        print()
        print(f"  Optimal per model:")
        for m in all_models:
            print(f"    {all_short[m]:<25} → {best_per_model[m]}")

        unique_optima = set(best_per_model.values())
        if len(unique_optima) > 1:
            print(f"  OPTIMUM DIFFERS across models ({len(unique_optima)} distinct)")
        else:
            print(f"  Same optimum across all models")
        print()

    # -----------------------------------------------------------------------
    # Summary: which profiles show model-dependent optima?
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("CONCLUSION — MULTI-MODEL GENERALIZATION")
    print("=" * 80)

    any_differs = False
    for profile in PROFILES:
        best_per_model = {}
        for m in all_models:
            best_u = -1
            best_key = None
            for key in common_keys:
                u = all_results[m][key].get(f"util_{profile}")
                if u is not None and u > best_u:
                    best_u = u
                    best_key = key
            best_per_model[m] = best_key

        unique = set(best_per_model.values())
        if len(unique) > 1:
            any_differs = True
            print(f"  {profile}: DIFFERS — {len(unique)} distinct optimal configs")
        else:
            print(f"  {profile}: same optimal across all models")

    if any_differs:
        print()
        print("  -> The optimal TurboQuant config is model-dependent.")
        print("     Per-context auto-tuning is necessary.")
    else:
        print()
        print("  -> Same optimum across models, but dPPL% differs significantly.")
        print("     The safety margin is model-dependent.")

    # Save
    out = Path("results/exp4_generalization.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for m in all_models:
        data[all_short[m]] = {
            str(k): v for k, v in all_results[m].items() if k in common_keys
        }
    out.write_text(json.dumps(data, indent=1, default=str))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
