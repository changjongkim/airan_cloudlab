#!/usr/bin/env python3
"""Cells-per-GPU sweep: radio latency, L1 misses and NeuralRx rescue yield per density."""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot")
JOB = sys.argv[1]
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "dn"


def main() -> None:
    rows = []
    for path in sorted(glob.glob(str(ROOT / "raw" / f"{PREFIX}*_j{JOB}.json"))):
        d = json.loads(Path(path).read_text())
        cfg, h, a = d["config"], d["headline"], d["all"]
        weak = d["per_profile"].get("weak_rank1", {})
        strong = d["per_profile"].get("strong_rank2", {})
        times = []
        for lane in glob.glob(path.replace(".json", "_work") + "/lane*.json"):
            times += [(r[3] - r[2]) / 1e6 for r in json.loads(Path(lane).read_text())["records"]]
        # Candidates = weak TBs whose conventional decode failed with at most one bad code block
        # (read from the conventional workers; the controller's "dropped" also counts TBs whose
        # conventional result was not in by the latest start).
        skip, max_cb = int(cfg.get("skip_periods", 20)), int(cfg.get("nrx_max_cb_fail", 1))
        candidates = 0
        for cell in cfg["cells"]:
            if cell["profile"] != "weak_rank1":
                continue
            conv = json.loads(Path(path.replace(".json", "_work") + f"/conv{cell['cell']}.json").read_text())
            candidates += sum(1 for r in conv["records"] if r[0] >= skip and not r[5] and r[7] <= max_cb)
        not_started = max(0, candidates - h["nrx_runs"])
        ai = h.get("ai_total") or {}
        rows.append({
            "tag": re.sub(r"_c\d+_.*", "", Path(path).name),
            "cells": h["cells"], "gpus": cfg["num_gpus"], "per_gpu": h["cells"] / cfg["num_gpus"],
            "weak_cells": sum(c["profile"] == "weak_rank1" for c in cfg["cells"]),
            "lanes": cfg["lanes_per_gpu"], "d2": cfg["rescue_deadline_ms"],
            "ai_policy": cfg.get("ai_policy"), "tbs": h["tbs"],
            "conv_p50": a["conv_done_ms"]["p50"], "conv_p99": a["conv_done_ms"]["p99"],
            "conv_p999": a["conv_done_ms"]["p999"],
            "weak_conv_p99": weak.get("conv_done_ms", {}).get("p99"),
            "strong_conv_p99": strong.get("conv_done_ms", {}).get("p99"),
            "l1_late": h["late_tbs"], "l1_late_pct": 100 * h["late_tbs"] / h["tbs"],
            "nrx_runs": h["nrx_runs"], "nrx_time_p50": float(np.percentile(times, 50)) if times else None,
            "nrx_time_p99": float(np.percentile(times, 99)) if times else None,
            "candidates": candidates, "dropped": not_started,
            "dropped_pct": 100 * not_started / max(1, candidates),
            "rescues_per_candidate_pct": 100 * h["nrx_rescues_on_time"] / max(1, candidates),
            "nrx_late": h["nrx_late"], "rescues": h["nrx_rescues_on_time"],
            "rescues_per_weak_tb_pct": 100 * h["nrx_rescues_on_time"] / max(1, weak.get("tbs", 0)),
            "false_pass": h["false_crc_pass"], "retx": h["retransmissions"],
            "ai_slo": ai.get("tokens_within_slo_per_s"),
        })
    (ROOT / f"density_{PREFIX}_j{JOB}.json").write_text(json.dumps(rows, indent=2))
    print("| tag | cells | GPUs | cells/GPU | weak cells | lanes/GPU | D2 | conv p50/p99/p99.9 ms | L1 late (%) | NRx time p50/p99 ms | candidates | not started (%) | NRx late | rescues (per weak TB) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["gpus"], r["weak_cells"] / r["cells"], r["lanes"], r["d2"], r["cells"])):
        nrx = f"{r['nrx_time_p50']:.2f}/{r['nrx_time_p99']:.2f}" if r["nrx_time_p50"] else "-"
        print(f"| {r['tag']} | {r['cells']} | {r['gpus']} | {r['per_gpu']:.0f} | {r['weak_cells']} | {r['lanes']} | {r['d2']} | "
              f"{r['conv_p50']:.2f}/{r['conv_p99']:.2f}/{r['conv_p999']:.2f} | {r['l1_late']} ({r['l1_late_pct']:.3f}) | {nrx} | "
              f"{r['candidates']} | {r['dropped']} ({r['dropped_pct']:.1f}) | {r['nrx_late']} | "
              f"{r['rescues']} ({r['rescues_per_weak_tb_pct']:.2f}%) |")


if __name__ == "__main__":
    main()
