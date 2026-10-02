#!/usr/bin/env python3
"""Why a run with AI rescues fewer TBs than the same seed without AI.

For every (cell, period) whose TBs the no-AI run rescued, the AI run either rescued them too,
or lost them for one of these reasons:
  conv_late   the conventional result arrived after the latest NeuralRx start (AI delayed it)
  lane_busy   the result was in time but no lane could take the TB before its latest start
  nrx_late    NeuralRx ran and finished after the rescue deadline
  other       anything else (for example NeuralRx was not started by the value rule)
usage: analyze_loss.py JOB TAG CELLS [VARIANT_TAG ...]
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw")
PASS, FAIL, SKIPPED, DROPPED = 1, 2, 3, 5


def load(path: Path) -> dict:
    result = json.loads(path.read_text())
    config = result["config"]
    work = Path(str(path)[:-5] + "_work")
    rescue_ns = float(config["rescue_deadline_ms"]) * 1e6
    bound_ns = float(config["nrx_bound_ms"]) * 1e6
    lanes = {}
    for lane in work.glob("lane*.json"):
        for r in json.loads(lane.read_text())["records"]:
            lanes[(r[0], r[1])] = r
    final = np.load(work / "final_cells.npy")
    slots = {}
    for cell in config["cells"]:
        if cell.get("nrx_gpu") is None:
            continue
        index, ues = int(cell["cell"]), int(cell.get("num_ue", 1))
        for r in json.loads((work / f"conv{index}.json").read_text())["records"]:
            period, release, done = r[0], r[2], r[4]
            conv_good = r[9] if ues > 1 else r[6]
            if conv_good == (1 << ues) - 1:
                continue
            ran = lanes.get((index, period))
            state = {"conv_late": done - release > rescue_ns - bound_ns, "rescued": 0, "ran": ran is not None,
                     "late": False, "status": int(final[index, period, 3])}
            if ran is not None:
                good = ran[7] if ues > 1 else ran[5]
                state["late"] = ran[3] - release > rescue_ns
                if not state["late"]:
                    state["rescued"] = bin(good & ~conv_good).count("1")
            slots[(index, period)] = state
    ai = result["headline"].get("ai_total") or {}
    return {"slots": slots, "slo": ai.get("tokens_within_slo_per_s", 0.0),
            "rescues": result["headline"]["nrx_rescues_on_time"]}


def main() -> None:
    job, tag, cells = sys.argv[1], sys.argv[2], sys.argv[3]
    tags = [tag] + sys.argv[4:]
    refs = {}
    for path in sorted(ROOT.glob(f"{tag}[0-9]n_c{cells}_*_j{job}.json")):
        refs[int(path.name[len(tag)])] = load(path)
    table = {}
    for t in tags:
        for path in sorted(ROOT.glob(f"{t}[0-9]r*_c{cells}_*_j{job}.json")):
            m = re.match(rf"{t}(\d)r(\d+)(\w+?)_c", path.name)
            seed = int(m.group(1))
            if seed not in refs:
                continue
            run, ref = load(path), refs[seed]
            total = sum(s["rescued"] for s in ref["slots"].values())
            lost = {"conv_late": 0, "lane_busy": 0, "nrx_late": 0, "other": 0}
            gained = 0
            for key, base in ref["slots"].items():
                now = run["slots"].get(key)
                if now is None:
                    continue
                diff = base["rescued"] - now["rescued"]
                if diff < 0:
                    gained -= diff
                elif diff > 0:
                    if now["ran"] and now["late"]:
                        lost["nrx_late"] += diff
                    elif not now["ran"] and now["conv_late"]:
                        lost["conv_late"] += diff
                    elif not now["ran"] and now["status"] == DROPPED:
                        lost["lane_busy"] += diff
                    else:
                        lost["other"] += diff
            row = table.setdefault((t, m.group(3), int(m.group(2))), [])
            row.append({"slo": run["slo"], "gained": 100 * gained / total,
                        **{k: 100 * v / total for k, v in lost.items()}})
    print("| run | AI load | tokens/s | lost: conventional result too late | lost: lane busy | lost: NeuralRx late | lost: other | gained |")
    print("|---|---|---|---|---|---|---|---|")
    out = []
    for (t, policy, rate), rows in sorted(table.items()):
        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        out.append({"tag": t, "policy": policy, "ai_rate": rate, "seeds": len(rows), **mean})
        print(f"| {t} {policy} | {rate} | {mean['slo']:.0f} | {mean['conv_late']:.1f}% | {mean['lane_busy']:.1f}% | "
              f"{mean['nrx_late']:.1f}% | {mean['other']:.1f}% | {mean['gained']:.1f}% |")
    Path(ROOT.parent / f"loss_{tag}_c{cells}_j{job}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
