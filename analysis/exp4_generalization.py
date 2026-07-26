"""Experiment 4 — Generalization spot-check.

Compares Qwen3-4B (primary) vs Qwen3-1.7B (spot-check) on the same configs
to show that the sensitivity map and optimum differ across models, establishing
that per-context auto-tuning is necessary.

Usage:
    python -m analysis.exp4_generalization [--cells-dir results/cells]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

EXP3_MANIFEST = Path("configs/grids/exp3.json")
EXP4_MANIFEST = Path("configs/grids/exp4.json")

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


def load_manifest_cells(manifest_path: Path, cells_dir: Path) -> list[dict]:
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


def analyze_model(cells: list[dict], model_name: str, n_layers: int):
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

    exp3_cells = load_manifest_cells(EXP3_MANIFEST, cells_dir)
    exp4_cells = load_manifest_cells(EXP4_MANIFEST, cells_dir)

    if not exp4_cells:
        print("No Exp 4 cells found. Run the manifest first.")
        return

    print(f"Exp 3 cells: {len(exp3_cells)}, Exp 4 cells: {len(exp4_cells)}\n")

    primary, _ = analyze_model(exp3_cells, "Qwen3-4B", n_layers=36)
    spot, _ = analyze_model(exp4_cells, "Qwen3-1.7B", n_layers=28)

    # Side-by-side comparison
    print("=" * 110)
    print("GENERALIZATION SPOT-CHECK: Qwen3-4B (primary) vs Qwen3-1.7B")
    print("=" * 110)
    print(f"{'Config':<20} {'PPL (4B)':>9} {'PPL (1.7B)':>10}  "
          f"{'dPPL% 4B':>8} {'dPPL% 1.7B':>10}  "
          f"{'R_mem 4B':>8} {'R_mem 1.7B':>10}")
    print("-" * 110)

    common_keys = sorted(set(primary.keys()) & set(spot.keys()))
    for key in common_keys:
        a = primary[key]
        b = spot[key]
        print(f"{a['label']:<20} {fmt(a['ppl'], 3):>9} {fmt(b['ppl'], 3):>10}  "
              f"{fmt(a['dppl_pct'], 2):>7}% {fmt(b['dppl_pct'], 2):>9}%  "
              f"{a['r_mem']:>7.2f}x {b['r_mem']:>9.2f}x")
    print()

    # Per-profile utility comparison
    for profile in PROFILES:
        print(f"{'=' * 80}")
        print(f"UTILITY — {profile.upper()} (threshold dPPL ≤ {PPL_THRESHOLDS[profile]*100:.1f}%)")
        print(f"{'=' * 80}")
        print(f"{'Config':<20} {'Qwen3-4B':>12} {'Qwen3-1.7B':>12}  {'Same rank?':>10}")
        print("-" * 80)

        items = []
        for key in common_keys:
            a = primary[key]
            b = spot[key]
            ua = a.get(f"util_{profile}")
            ub = b.get(f"util_{profile}")
            items.append((key, a["label"], ua, ub))

        ranked_a = sorted(items, key=lambda x: -(x[2] or -1))
        ranked_b = sorted(items, key=lambda x: -(x[3] or -1))
        rank_a = {item[0]: i for i, item in enumerate(ranked_a)}
        rank_b = {item[0]: i for i, item in enumerate(ranked_b)}

        for key, label, ua, ub in items:
            ra = rank_a.get(key, -1) + 1
            rb = rank_b.get(key, -1) + 1
            same = "YES" if ra == rb else f"#{ra}→#{rb}"
            print(f"{label:<20} {fmt(ua, 4):>12} {fmt(ub, 4):>12}  {same:>10}")

        best_a = ranked_a[0][1] if ranked_a else "?"
        best_b = ranked_b[0][1] if ranked_b else "?"
        differs = best_a != best_b
        print(f"\n  Best for 4B:   {best_a}")
        print(f"  Best for 1.7B: {best_b}")
        print(f"  Optimum differs: {'YES' if differs else 'NO'}")
        print()

    # Headline
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    any_differs = False
    for profile in PROFILES:
        items = []
        for key in common_keys:
            a = primary[key]
            b = spot[key]
            ua = a.get(f"util_{profile}")
            ub = b.get(f"util_{profile}")
            items.append((key, ua, ub))
        best_a = max(items, key=lambda x: x[1] or -1)[0] if items else None
        best_b = max(items, key=lambda x: x[2] or -1)[0] if items else None
        if best_a != best_b:
            any_differs = True
            print(f"  {profile}: optimal config DIFFERS between models")
        else:
            print(f"  {profile}: same optimal config on both models")

    if any_differs:
        print("\n  -> Per-context auto-tuning is necessary: the optimal config")
        print("     is model-dependent. A config tuned for one model does not")
        print("     transfer to another.")
    else:
        print("\n  -> Optimal configs match across models. The PPL sensitivity")
        print("     profile may still differ (check dPPL% columns above).")

    out = Path("results/exp4_generalization.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "primary": {str(k): v for k, v in primary.items()},
        "spot_check": {str(k): v for k, v in spot.items()},
    }
    out.write_text(json.dumps(data, indent=1, default=str))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
