#!/usr/bin/env python3
"""Per-run trace of the recovery path: every candidate, when it became known, whether it got a
neural receiver, and how long that run took.  Input of the recovery-loss model (loss_model.py).

A candidate is a slot of a two-user cell whose conventional decode failed with at most K failed
code blocks.  Times are in ms from the arrival of the slot.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RAW = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw")
CACHE = RAW.parent / "model_cache"


def run_path(name: str) -> Path:
    """Result JSON of a run given its name (with or without the policy suffix and job)."""
    path = RAW / f"{name}.json"
    if path.is_file():
        return path
    matches = sorted(p for p in RAW.glob(f"{name}*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{name}: {len(matches)} matches")
    return matches[0]


def trace(name: str, refresh: bool = False) -> dict:
    path = run_path(name)
    cached = CACHE / (path.stem + ".npz")
    if cached.is_file() and not refresh:
        data = np.load(cached, allow_pickle=False)
        return {k: data[k] for k in data.files}
    result = json.loads(path.read_text())
    config = result["config"]
    work = Path(str(path)[:-5] + "_work")
    periods, skip = int(config["periods"]), int(config.get("skip_periods", 20))
    period_ms = float(config["period_ms"])
    rescue_ms = float(config["rescue_deadline_ms"])
    k_max = int(config.get("nrx_max_cb_fail", 1))
    weak = [c for c in config["cells"] if c.get("nrx_gpu") is not None]
    lanes = {}
    for lane in sorted(work.glob("lane*.json")):
        data = json.loads(lane.read_text())
        for r in data["records"]:
            lanes[(r[0], r[1])] = (int(data["gpu"]), r)
    rows = []
    active = np.zeros((len(weak), periods), dtype=bool)
    for w, cell in enumerate(weak):
        index, ues = int(cell["cell"]), int(cell.get("num_ue", 1))
        every = (1 << ues) - 1
        for r in json.loads((work / f"conv{index}.json").read_text())["records"]:
            period, release, done = r[0], r[2], r[4]
            active[w, period] = True
            if period < skip:
                continue
            good = r[9] if ues > 1 else (every if r[6] else 0)
            failed = good != every
            cb_fail = int(r[7])
            if not failed or cb_fail > k_max:
                continue
            ran = lanes.get((index, period))
            start = duration = -1.0
            gpu, recovered = -1, 0
            if ran is not None:
                gpu, rec = ran
                start, duration = (rec[2] - release) / 1e6, (rec[3] - rec[2]) / 1e6
                if rec[3] - release <= rescue_ms * 1e6:
                    nrx_good = rec[7] if ues > 1 else (every if rec[5] else 0)
                    recovered = bin(nrx_good & ~good & every).count("1")
            rows.append((w, period, (done - release) / 1e6, cb_fail, start, duration, gpu, recovered))
    rows.sort(key=lambda r: (r[1], r[0]))
    out = {
        "cell": np.array([r[0] for r in rows], dtype=np.int32),
        "period": np.array([r[1] for r in rows], dtype=np.int32),
        "known_ms": np.array([r[2] for r in rows]),
        "cb_fail": np.array([r[3] for r in rows], dtype=np.int32),
        "start_ms": np.array([r[4] for r in rows]),
        "run_ms": np.array([r[5] for r in rows]),
        "gpu": np.array([r[6] for r in rows], dtype=np.int32),
        "recovered": np.array([r[7] for r in rows], dtype=np.int32),
        "weak_cells": np.int32(len(weak)), "lanes": np.int32(len(list(work.glob("lane*.json")))),
        "periods": np.int32(periods), "skip": np.int32(skip), "period_ms": np.float64(period_ms),
        "rescue_ms": np.float64(rescue_ms), "bound_ms": np.float64(config["nrx_bound_ms"]),
        "active": active,
    }
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(cached, **out)
    return out


if __name__ == "__main__":
    import sys
    for name in sys.argv[1:]:
        t = trace(name, refresh=True)
        ran = t["start_ms"] >= 0
        slots = (int(t["periods"]) - int(t["skip"])) * int(t["weak_cells"])
        print(f"{name}: candidates {len(ran)} ({len(ran) / slots:.3f} per two-user slot), ran {ran.sum()}, "
              f"dropped {(~ran).sum()} ({100 * (~ran).mean():.1f}%), recovered {t['recovered'].sum()}; "
              f"known p50 {np.median(t['known_ms']):.2f}, start p50 {np.median(t['start_ms'][ran]):.2f}, "
              f"run p50 {np.median(t['run_ms'][ran]):.2f} ms")
