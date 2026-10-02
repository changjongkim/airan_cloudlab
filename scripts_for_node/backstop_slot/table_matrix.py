#!/usr/bin/env python3
"""Collect slot-scale run summaries into one comparison table."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def row(path: Path) -> dict:
    d = json.loads(path.read_text())
    h = d["headline"]
    everything = d["all"]
    ai = h.get("ai_total") or {}
    ttft = [v["ttft_ms"] for v in (d.get("ai") or {}).values() if v and v["ttft_ms"].get("n")]
    weak = d["per_profile"].get("weak_rank1", {})
    return {
        "run": path.stem,
        "cells": h["cells"],
        "nrx": h["nrx_policy"],
        "ai": h["ai_policy"],
        "tbs": h["tbs"],
        "late_tbs": h["late_tbs"],
        "on_time": h["ul_indication_on_time"],
        "decoded_on_time": h["decoded_on_time"],
        "rescues": h["nrx_rescues_on_time"],
        "decoded_final": h.get("decoded_final"),
        "retx": h.get("retransmissions"),
        "nrx_runs": h["nrx_runs"],
        "nrx_late": h.get("nrx_late"),
        "nrx_dropped": weak.get("nrx_start_reasons", {}).get("dropped"),
        "conv_p99_ms": everything["conv_done_ms"]["p99"],
        "conv_max_ms": everything["conv_done_ms"]["max"],
        "ai_tokens_per_s": ai.get("tokens_per_s", 0.0),
        "ai_slo_tokens_per_s": ai.get("tokens_within_slo_per_s", 0.0),
        "ai_completed": ai.get("completed", 0),
        "ai_arrived": ai.get("arrived", 0),
        "ai_overruns": ai.get("piece_overruns", 0),
        "ai_pieces": ai.get("pieces", 0),
        "ttft_p50_ms_gpu_mean": (sum(t["p50"] for t in ttft) / len(ttft)) if ttft else None,
        "ttft_p99_ms_gpu_max": max(t["p99"] for t in ttft) if ttft else None,
    }


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:-1]]
    output = Path(sys.argv[-1])
    rows = [row(p) for p in paths]
    output.write_text(json.dumps(rows, indent=2))
    cols = ["cells", "nrx", "ai", "late_tbs", "decoded_final", "retx", "rescues", "nrx_runs",
            "nrx_late", "nrx_dropped", "conv_p99_ms", "ai_tokens_per_s",
            "ai_slo_tokens_per_s", "ai_completed", "ai_arrived", "ai_overruns",
            "ttft_p50_ms_gpu_mean", "ttft_p99_ms_gpu_max"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                v = f"{v:.4f}" if c in ("decoded_on_time", "decoded_final") else f"{v:.2f}" if v < 100 else f"{v:.0f}"
            cells.append(str(v))
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
