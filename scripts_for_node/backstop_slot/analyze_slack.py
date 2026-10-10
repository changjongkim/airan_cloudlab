#!/usr/bin/env python3
"""Room of each neural receiver run before the next boundary, and the AI left beside it.

usage: analyze_slack.py OUT.json TAG CELLS POLICY[,POLICY...] [JOB,JOB...]
POLICY "n" is the run without AI.  All seeds and jobs of the tag are pooled per policy.

For every neural receiver run: start offset s and length S (ms from the arrival of its slot),
and the room  2P + L - (s + S)  to the latest start of the candidates two slots later
(L = recovery deadline - time limit of that run's configuration).  A negative room is a run
that holds a third slot.  For runs next to AI: the time an AI piece was still on that GPU after
the neural receiver started (the stop tail) and the AI time inside the whole run.

Printed per policy: the distribution of the room, the share of three-slot runs split into
"late start" (s + the median length without AI already passes the boundary) and "lengthened"
(passes only because S grew), the lengthened ones split by whether AI was beside the run, and
the same count if the stop tail were cut to a cap (S reduced by the fitted ms of S per ms of tail).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RAW = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw")


def one(path: Path) -> dict:
    result = json.loads(path.read_text())
    config = result["config"]
    work = Path(str(path)[:-5] + "_work")
    period = float(config["period_ms"])
    periods = int(config["periods"])
    skip = max(int(config.get("skip_periods", 20)), periods // 10 if periods >= 8000 else 0)
    late_start = float(config["rescue_deadline_ms"]) - float(config["nrx_bound_ms"])
    weak = [int(c["cell"]) for c in config["cells"] if c.get("nrx_gpu") is not None]
    release, known, candidates = {}, {}, 0
    k_max = int(config.get("nrx_max_cb_fail", 1))
    for cell in weak:
        ues = next(int(c.get("num_ue", 1)) for c in config["cells"] if int(c["cell"]) == cell)
        every = (1 << ues) - 1
        for r in json.loads((work / f"conv{cell}.json").read_text())["records"]:
            release[(cell, r[0])] = r[2]
            known[(cell, r[0])] = (r[4] - r[2]) / 1e6
            good = r[9] if ues > 1 else (every if r[6] else 0)
            if r[0] >= skip and good != every and int(r[7]) <= k_max:
                candidates += 1
    rows = []
    for lane in sorted(work.glob("lane*.json")):
        data = json.loads(lane.read_text())
        gpu = int(data["gpu"])
        pieces = []
        ai_file = work / f"ai{gpu}.json"
        if ai_file.is_file():
            pieces = [(p[0], p[1]) for p in json.loads(ai_file.read_text()).get("pieces", [])]
        pieces.sort()
        starts = np.array([p[0] for p in pieces], dtype=np.int64)
        ends = np.array([p[1] for p in pieces], dtype=np.int64)
        for r in data["records"]:
            cell, k, a, b = int(r[0]), int(r[1]), int(r[2]), int(r[3])
            if k < skip or (cell, k) not in release:
                continue
            s = (a - release[(cell, k)]) / 1e6
            length = (b - a) / 1e6
            tail = inside = 0.0
            if len(starts):
                lo = np.searchsorted(ends, a, side="right")
                hi = np.searchsorted(starts, b, side="left")
                for i in range(lo, hi):
                    inside += (min(ends[i], b) - max(starts[i], a)) / 1e6
                    if starts[i] <= a:
                        tail = max(tail, (min(ends[i], b) - a) / 1e6)
            rows.append((s, length, 2 * period + late_start - (s + length), tail, inside,
                         s - known[(cell, k)], known[(cell, k)]))
    return {"rows": np.array(rows).reshape(-1, 7), "candidates": candidates, "late_start": late_start,
            "period": period, "bound": float(config["nrx_bound_ms"])}


def quantiles(v, qs=(10, 50, 90, 99)):
    return [round(float(np.percentile(v, q)), 2) for q in qs] if len(v) else []


def main() -> None:
    out, tag, cells = sys.argv[1], sys.argv[2], sys.argv[3]
    policies = sys.argv[4].split(",")
    jobs = sys.argv[5].split(",") if len(sys.argv) > 5 else None
    report = {"tag": tag, "policies": {}}
    alone_median = alone_known = None
    for policy in policies:
        name = f"{tag}[0-9]*n_c{cells}_*" if policy == "n" else f"{tag}[0-9]*r[0-9]*{policy}_c{cells}_*"
        paths = [p for p in sorted(RAW.glob(name + ".json"))
                 if jobs is None or any(p.stem.endswith("_j" + j) for j in jobs)]
        if policy == "n":
            paths = [p for p in paths if "_none_" in p.name]
        if not paths:
            print(f"{policy}: no runs")
            continue
        parts = [one(p) for p in paths]
        rows = np.concatenate([p["rows"] for p in parts])
        s, length, room, tail, inside, waited, known_ms = rows.T
        late_start, period = parts[0]["late_start"], parts[0]["period"]
        boundary = 2 * period + late_start
        if policy == "n":
            alone_median = float(np.median(length))
            alone_known = float(np.median(known_ms))
        ref = alone_median if alone_median is not None else float(np.median(length))
        three = room < 0
        late = three & (s + ref > boundary)
        longer = three & ~late
        beside = longer & (inside > 0.02)
        entry = {
            "runs": len(paths), "nrx_runs": int(len(rows)), "candidates": int(sum(p["candidates"] for p in parts)),
            "no_nrx_pct": round(100 * (1 - len(rows) / max(1, sum(p["candidates"] for p in parts))), 2),
            "late_start_ms": late_start, "time_limit_ms": parts[0]["bound"],
            "known_ms": quantiles(known_ms), "start_after_known_ms": quantiles(waited),
            "start_offset_ms": quantiles(s), "run_ms": quantiles(length), "room_ms": quantiles(room, (1, 10, 50, 90)),
            "room_below": {str(x): round(100 * float((room < x).mean()), 1) for x in (0, 0.2, 0.43, 0.84, 1.2)},
            "waited_for_nrx_pct": round(100 * float((waited > 0.3).mean()), 1),
            "three_slot_pct": round(100 * float(three.mean()), 1),
            "three_slot_late_start_pct": round(100 * float(late.mean()), 1),
            "three_slot_lengthened_pct": round(100 * float(longer.mean()), 1),
            "lengthened_with_ai_beside_pct": round(100 * float(beside.mean()), 1),
            "ai_tail_share_pct": round(100 * float((tail > 0.02).mean()), 1),
            "ai_tail_ms": quantiles(tail[tail > 0.02]),
            "ai_inside_ms": quantiles(inside[inside > 0.02]),
        }
        if policy != "n" and (tail > 0.02).sum() > 50:
            slope, base = np.polyfit(tail, length, 1)
            entry["run_ms_per_tail_ms"] = round(float(slope), 2)
            entry["run_ms_without_tail"] = round(float(base), 2)
            entry["run_ms_by_tail"] = {
                label: [int(mask.sum()), round(float(np.median(length[mask])), 2), round(100 * float(three[mask].mean()), 1)]
                for label, mask in (("none", tail <= 0.02), ("to 0.3", (tail > 0.02) & (tail <= 0.3)),
                                    ("0.3-0.6", (tail > 0.3) & (tail <= 0.6)), ("0.6-1.0", (tail > 0.6) & (tail <= 1.0)),
                                    ("over 1.0", tail > 1.0)) if mask.sum()}
            for cap in (0.37, 0.2, 0.0):
                shorter = length - max(0.0, float(slope)) * np.maximum(0.0, tail - cap)
                entry[f"three_slot_pct_if_tail_capped_{cap}"] = round(100 * float((s + shorter > boundary).mean()), 1)
            if alone_median is not None:
                earlier = float(np.median(known_ms)) - alone_known
                no_tail = length - max(0.0, float(slope)) * tail
                entry["known_later_than_alone_ms"] = round(earlier, 2)
                entry["three_slot_pct_if_known_as_alone"] = round(100 * float((s - earlier + length > boundary).mean()), 1)
                entry["three_slot_pct_if_both"] = round(100 * float((s - earlier + no_tail > boundary).mean()), 1)
        report["policies"][policy] = entry
        print(f"\n{tag} {policy}: {len(paths)} runs, {len(rows)} neural receiver runs, "
              f"L = {late_start:.1f} ms, candidates without a neural receiver {entry['no_nrx_pct']}%")
        for key in ("known_ms", "start_after_known_ms", "start_offset_ms", "run_ms", "room_ms", "room_below", "waited_for_nrx_pct", "three_slot_pct",
                    "three_slot_late_start_pct", "three_slot_lengthened_pct", "lengthened_with_ai_beside_pct",
                    "ai_tail_share_pct", "ai_tail_ms", "ai_inside_ms", "run_ms_per_tail_ms", "run_ms_without_tail",
                    "run_ms_by_tail", "three_slot_pct_if_tail_capped_0.37", "three_slot_pct_if_tail_capped_0.2",
                    "three_slot_pct_if_tail_capped_0.0", "known_later_than_alone_ms",
                    "three_slot_pct_if_known_as_alone", "three_slot_pct_if_both"):
            if key in entry:
                print(f"  {key}: {entry[key]}")
    Path(out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
