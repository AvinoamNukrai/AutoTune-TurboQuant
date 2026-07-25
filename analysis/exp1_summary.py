"""Experiment 1 analysis: screening grid summary + ANOVA.

Usage:
    python -m analysis.exp1_summary [--cells-dir results/cells]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from itertools import groupby

MANIFEST = Path("configs/grids/exp1.json")


def load_cells(cells_dir: Path) -> list[dict]:
    manifest = json.loads(MANIFEST.read_text())
    hashes: set[str] = set()
    for entry in manifest:
        from src.harness import CellConfig
        entry["skip_layers"] = tuple(entry.get("skip_layers", ()))
        hashes.add(CellConfig(**entry).cell_hash())

    results = []
    for f in sorted(cells_dir.glob("*.json")):
        if f.stem in hashes:
            data = json.loads(f.read_text())
            if data.get("ok"):
                results.append(data)
    return results


def protection_label(skip: list) -> str:
    if not skip:
        return "floor"
    skip_set = set(int(x) for x in skip)
    if skip_set == {2, 3, 32, 33}:
        return "pos_n4"
    if skip_set == {4, 5, 8, 33}:
        return "stat_b8"
    if skip_set == {5, 33}:
        return "stat_b6"
    return f"custom_{sorted(skip_set)}"


def fmt(v, digits=2):
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells-dir", default="results/cells")
    args = p.parse_args()

    cells = load_cells(Path(args.cells_dir))
    if not cells:
        print("No completed cells found.")
        return

    rows = []
    for c in cells:
        cfg = c["config"]
        dtype = cfg["kv_cache_dtype"]
        skip = cfg.get("skip_layers", [])
        prot = protection_label(skip)
        eff = c.get("effective_skip_layers", [])
        rep = cfg["rep"]

        ppl_val = c.get("ppl", {}).get("ppl")
        scored = c.get("ppl", {}).get("scored_tokens", 0)

        chat = c.get("profiles", {}).get("chat", {})
        rag = c.get("profiles", {}).get("rag", {})
        batch = c.get("profiles", {}).get("batch", {})

        ttft_chat = chat.get("ttft_s", {}).get("mean")
        tpot_chat = chat.get("tpot_s", {}).get("mean")
        tp_chat = chat.get("throughput_tok_s")
        needle = rag.get("needle_accuracy")
        ttft_rag = rag.get("ttft_s", {}).get("mean")
        tp_batch = batch.get("throughput_tok_s")

        rows.append({
            "dtype": dtype.replace("turboquant_", ""),
            "protection": prot,
            "eff_skip": eff,
            "rep": rep,
            "ppl": ppl_val,
            "scored_tokens": scored,
            "chat_ttft_ms": ttft_chat * 1000 if ttft_chat else None,
            "chat_tpot_ms": tpot_chat * 1000 if tpot_chat else None,
            "chat_tps": tp_chat,
            "rag_ttft_ms": ttft_rag * 1000 if ttft_rag else None,
            "needle_acc": needle,
            "batch_tps": tp_batch,
        })

    # --- Summary table ---
    print("=" * 110)
    print("EXPERIMENT 1 — SCREENING GRID RESULTS")
    print("=" * 110)
    print(f"{'dtype':<12} {'protection':<10} {'rep':>3}  {'PPL':>7}  "
          f"{'chat_TTFT':>9}  {'chat_TPOT':>9}  {'chat_TPS':>8}  "
          f"{'needle':>6}  {'batch_TPS':>9}  eff_skip")
    print("-" * 110)

    rows.sort(key=lambda r: (r["dtype"], r["protection"], r["rep"]))
    for r in rows:
        eff_str = str(r["eff_skip"]) if r["eff_skip"] else "[]"
        print(f"{r['dtype']:<12} {r['protection']:<10} {r['rep']:>3}  "
              f"{fmt(r['ppl'], 2):>7}  "
              f"{fmt(r['chat_ttft_ms'], 1):>9}  "
              f"{fmt(r['chat_tpot_ms'], 2):>9}  "
              f"{fmt(r['chat_tps'], 1):>8}  "
              f"{fmt(r['needle_acc'], 2):>6}  "
              f"{fmt(r['batch_tps'], 1):>9}  "
              f"{eff_str}")
    print()

    # --- Averaged over reps ---
    print("=" * 100)
    print("AVERAGED OVER REPS (mean of rep 0 + rep 1)")
    print("=" * 100)
    print(f"{'dtype':<12} {'protection':<10}  {'PPL':>7}  "
          f"{'chat_TTFT':>9}  {'chat_TPOT':>9}  {'chat_TPS':>8}  "
          f"{'needle':>6}  {'batch_TPS':>9}")
    print("-" * 100)

    key_fn = lambda r: (r["dtype"], r["protection"])
    rows.sort(key=key_fn)
    avg_rows = []
    for (dtype, prot), group in groupby(rows, key=key_fn):
        grp = list(group)
        n = len(grp)

        def avg(field):
            vals = [r[field] for r in grp if r[field] is not None]
            return sum(vals) / len(vals) if vals else None

        ar = {
            "dtype": dtype, "protection": prot,
            "ppl": avg("ppl"),
            "chat_ttft_ms": avg("chat_ttft_ms"),
            "chat_tpot_ms": avg("chat_tpot_ms"),
            "chat_tps": avg("chat_tps"),
            "needle_acc": avg("needle_acc"),
            "batch_tps": avg("batch_tps"),
            "n": n,
        }
        avg_rows.append(ar)
        print(f"{dtype:<12} {prot:<10}  {fmt(ar['ppl'], 2):>7}  "
              f"{fmt(ar['chat_ttft_ms'], 1):>9}  "
              f"{fmt(ar['chat_tpot_ms'], 2):>9}  "
              f"{fmt(ar['chat_tps'], 1):>8}  "
              f"{fmt(ar['needle_acc'], 2):>6}  "
              f"{fmt(ar['batch_tps'], 1):>9}")
    print()

    # --- Baseline comparison ---
    baseline = next((r for r in avg_rows if r["dtype"] == "auto"), None)
    if baseline and baseline["ppl"]:
        print("=" * 80)
        print(f"DELTA vs BASELINE (auto, PPL={fmt(baseline['ppl'], 2)})")
        print("=" * 80)
        print(f"{'dtype':<12} {'protection':<10}  {'dPPL':>7} {'dPPL%':>6}  "
              f"{'dTTFT%':>7}  {'dTPOT%':>7}  {'needle':>6}")
        print("-" * 80)
        for r in avg_rows:
            if r["dtype"] == "auto":
                continue
            dppl = r["ppl"] - baseline["ppl"] if r["ppl"] else None
            dppl_pct = (dppl / baseline["ppl"] * 100) if dppl is not None else None
            dttft = ((r["chat_ttft_ms"] - baseline["chat_ttft_ms"]) /
                     baseline["chat_ttft_ms"] * 100
                     if r["chat_ttft_ms"] and baseline["chat_ttft_ms"] else None)
            dtpot = ((r["chat_tpot_ms"] - baseline["chat_tpot_ms"]) /
                     baseline["chat_tpot_ms"] * 100
                     if r["chat_tpot_ms"] and baseline["chat_tpot_ms"] else None)
            print(f"{r['dtype']:<12} {r['protection']:<10}  "
                  f"{fmt(dppl, 2):>7} {fmt(dppl_pct, 1):>5}%  "
                  f"{fmt(dttft, 1):>6}%  "
                  f"{fmt(dtpot, 1):>6}%  "
                  f"{fmt(r['needle_acc'], 2):>6}")
        print()

    # --- Protection effect per dtype ---
    print("=" * 80)
    print("PROTECTION EFFECT: PPL by dtype × protection")
    print("=" * 80)
    dtypes = sorted(set(r["dtype"] for r in avg_rows if r["dtype"] != "auto"))
    prots = ["floor", "pos_n4", "stat_b8", "stat_b6"]
    header = f"{'dtype':<12} " + "  ".join(f"{p:>10}" for p in prots)
    print(header)
    print("-" * 80)
    for dt in dtypes:
        vals = []
        for pr in prots:
            match = next((r for r in avg_rows
                          if r["dtype"] == dt and r["protection"] == pr), None)
            vals.append(fmt(match["ppl"], 3) if match and match["ppl"] else "N/A")
        print(f"{dt:<12} " + "  ".join(f"{v:>10}" for v in vals))
    print()

    # --- Key findings ---
    print("=" * 80)
    print("KEY OBSERVATIONS")
    print("=" * 80)

    if baseline and baseline["ppl"]:
        best_ppl = min((r for r in avg_rows if r["dtype"] != "auto" and r["ppl"]),
                       key=lambda r: r["ppl"])
        worst_ppl = max((r for r in avg_rows if r["dtype"] != "auto" and r["ppl"]),
                        key=lambda r: r["ppl"])
        print(f"  Baseline PPL:  {fmt(baseline['ppl'], 3)}")
        print(f"  Best TQ PPL:   {fmt(best_ppl['ppl'], 3)} "
              f"({best_ppl['dtype']}/{best_ppl['protection']})")
        print(f"  Worst TQ PPL:  {fmt(worst_ppl['ppl'], 3)} "
              f"({worst_ppl['dtype']}/{worst_ppl['protection']})")
        print()

    for dt in dtypes:
        dt_rows = [r for r in avg_rows if r["dtype"] == dt]
        floor_r = next((r for r in dt_rows if r["protection"] == "floor"), None)
        stat8_r = next((r for r in dt_rows if r["protection"] == "stat_b8"), None)
        pos4_r = next((r for r in dt_rows if r["protection"] == "pos_n4"), None)
        if floor_r and stat8_r and floor_r["ppl"] and stat8_r["ppl"]:
            improvement = floor_r["ppl"] - stat8_r["ppl"]
            print(f"  {dt}: stat_b8 vs floor = {improvement:+.3f} PPL "
                  f"({'better' if improvement > 0 else 'worse'})")
        if pos4_r and stat8_r and pos4_r["ppl"] and stat8_r["ppl"]:
            diff = pos4_r["ppl"] - stat8_r["ppl"]
            print(f"  {dt}: stat_b8 vs pos_n4 = {diff:+.3f} PPL "
                  f"({'stat wins' if diff > 0 else 'pos wins'})")

    # --- Write raw data for further analysis ---
    out_path = Path("results/exp1_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"per_cell": rows, "averaged": avg_rows}, indent=1))
    print(f"\nRaw data saved to {out_path}")


if __name__ == "__main__":
    main()
