#!/usr/bin/env python3
"""Generic per-run table for v3 campaigns (partial load, AI classes, NeuralRx-primary cells).

usage: analyze_runs.py JOB PREFIX [--json out.json]
Prints one row per run whose tag starts with PREFIX.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot")


def run_row(path: str) -> dict:
    d = json.loads(Path(path).read_text())
    cfg, h, a = d["config"], d["headline"], d["all"]
    work = Path(path.replace(".json", "_work"))
    seconds = int(cfg["periods"]) * float(cfg["period_ms"]) / 1e3
    gpus = cfg["num_gpus"]
    skip = int(cfg.get("skip_periods", 20))
    d2 = float(cfg.get("rescue_deadline_ms", cfg["deadline_ms"])) * 1e6
    period = float(cfg["period_ms"]) * 1e6
    primary = {int(c["cell"]) for c in cfg["cells"] if c.get("nrx_mode") == "primary"}
    row = {
        "tag": re.sub(r"_c\d+_.*", "", Path(path).name), "cells": h["cells"], "per_gpu": h["cells"] / gpus,
        "ai_policy": cfg.get("ai_policy"), "nrx_policy": cfg["nrx_policy"],
        "activity": (cfg.get("activity") or {}).get("prob", 1.0),
        "tbs": h["tbs"], "l1_late": h["late_tbs"], "l1_late_pct": 100 * h["late_tbs"] / max(1, h["tbs"]),
        "conv_p50": a["conv_done_ms"]["p50"], "conv_p99": a["conv_done_ms"]["p99"],
        "conv_p999": a["conv_done_ms"]["p999"],
        "nrx_runs": h["nrx_runs"], "nrx_late": h["nrx_late"], "rescues": h["nrx_rescues_on_time"],
        "retx": h["retransmissions"], "false_pass": h["false_crc_pass"],
        "primary_cells": len(primary),
    }
    if primary:
        # Primary TBs: NeuralRx must finish within the rescue deadline of every active TB.
        from_conv = {}
        for cell in primary:
            conv = json.loads((work / f"conv{cell}.json").read_text())
            for r in conv["records"]:
                if r[0] >= skip:
                    from_conv[(cell, r[0])] = r[2]
        on_time = 0
        for lane in work.glob("lane*.json"):
            for cell, k, start, done, ok, payload in json.loads(lane.read_text())["records"]:
                if (cell, k) in from_conv and done <= from_conv[(cell, k)] + d2:
                    on_time += 1
        row["primary_tbs"] = len(from_conv)
        row["primary_on_time_pct"] = 100 * on_time / max(1, len(from_conv))
    ai = h.get("ai_total") or {}
    row["ai_slo"] = ai.get("tokens_within_slo_per_s", 0.0)
    row["ai_tokens"] = ai.get("tokens_per_s", 0.0)
    row["ai_busy"] = ai.get("gpu_busy_ms", 0.0) / 1e3 / seconds / gpus
    classes = {}
    for name, summary in (d.get("ai") or {}).items():
        for cname, c in (summary or {}).get("classes", {}).items():
            agg = classes.setdefault(cname, {"kind": c["kind"], "arrived": 0, "completed": 0, "within_slo": 0,
                                             "tokens": 0.0, "slo_tokens": 0.0, "rejected": 0, "p50": []})
            agg["arrived"] += c["arrived"]
            agg["completed"] += c["completed"]
            agg["rejected"] += c["rejected"]
            agg["within_slo"] += c.get("within_slo", 0)
            agg["tokens"] += c["tokens_per_s"] + c.get("partial_tokens_per_s", 0.0)
            agg["slo_tokens"] += c.get("tokens_within_slo_per_s", 0.0)
            if c["latency_ms"].get("p50") is not None:
                agg["p50"].append(c["latency_ms"]["p50"])
    for c in classes.values():
        c["p50"] = float(np.mean(c["p50"])) if c["p50"] else None
    row["classes"] = classes
    ctl = (d.get("controller") or {}).get("counters", {})
    row["dropped"] = ctl.get("dropped")
    row["queue_blocked"] = ctl.get("queue_blocked")
    return row


def main() -> None:
    job, prefix = sys.argv[1], sys.argv[2]
    rows = [run_row(p) for p in sorted(glob.glob(str(ROOT / "raw" / f"{prefix}*_j{job}.json")))]
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(rows, indent=2))
    print("| tag | cells (per GPU) | load p | AI policy | conv p50/p99/p99.9 | L1 late (%) | NRx runs | NRx late | rescues | primary on time | AI SLO tok/s | AI all tok/s | AI busy | classes |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        prim = f"{r['primary_on_time_pct']:.1f}% of {r['primary_tbs']}" if r.get("primary_tbs") else "-"
        cls = "; ".join(
            f"{n}: {c['slo_tokens']:.0f} in SLO ({c['within_slo']}/{c['arrived']})" if c["kind"] == "interactive"
            else f"{n}: {c['tokens']:.0f} tok/s" for n, c in r["classes"].items())
        print(f"| {r['tag']} | {r['cells']} ({r['per_gpu']:.0f}) | {r['activity']} | {r['ai_policy']} | "
              f"{r['conv_p50']:.2f}/{r['conv_p99']:.2f}/{r['conv_p999']:.2f} | {r['l1_late']} ({r['l1_late_pct']:.3f}) | "
              f"{r['nrx_runs']} | {r['nrx_late']} | {r['rescues']} | {prim} | {r['ai_slo']:.0f} | {r['ai_tokens']:.0f} | "
              f"{100*r['ai_busy']:.0f}% | {cls} |")


if __name__ == "__main__":
    main()
