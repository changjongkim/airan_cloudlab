#!/usr/bin/env python3
"""Runs whose load alternates between full and partial phases: rescued TBs and AI served per phase.

usage: analyze_phases.py JOB[,JOB...] TAG CELLS      (seeds may be spread over several jobs)
Rescued TBs are compared with the no-AI run of the same seed, phase by phase.  AI tokens are
credited to the phase in which the request arrived.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw")


def load(path: Path) -> dict:
    result = json.loads(path.read_text())
    config = result["config"]
    work = Path(str(path)[:-5] + "_work")
    phase = int(config["activity"]["phase_periods"])
    period_ns = float(config["period_ms"]) * 1e6
    rescue_ns = float(config["rescue_deadline_ms"]) * 1e6
    deadline_ns = float(config["deadline_ms"]) * 1e6
    skip = int(config.get("skip_periods", 20))
    lanes = {}
    for lane in work.glob("lane*.json"):
        for r in json.loads(lane.read_text())["records"]:
            lanes[(r[0], r[1])] = r
    out = {"rescued": [0, 0], "late": [0, 0], "tbs": [0, 0], "tokens": [0.0, 0.0], "seconds": [0.0, 0.0]}
    for cell in config["cells"]:
        index, ues = int(cell["cell"]), int(cell.get("num_ue", 1))
        for r in json.loads((work / f"conv{index}.json").read_text())["records"]:
            period, release, done = r[0], r[2], r[4]
            if period < skip:
                continue
            kind = (period // phase) % 2
            out["tbs"][kind] += ues
            out["late"][kind] += ues * int(done - release > deadline_ns)
            if cell.get("nrx_gpu") is None:
                continue
            conv_good = r[9] if ues > 1 else r[6]
            ran = lanes.get((index, period))
            if ran is not None and ran[3] - release <= rescue_ns:
                good = ran[7] if ues > 1 else ran[5]
                out["rescued"][kind] += bin(good & ~conv_good & ((1 << ues) - 1)).count("1")
    periods = int(config["periods"])
    for k in range(skip, periods):
        out["seconds"][(k // phase) % 2] += period_ns / 1e9
    slo_ns = float(config.get("ai", {}).get("slo_ms", 200.0)) * 1e6
    for path_ai in work.glob("ai*.json"):
        data = json.loads(path_ai.read_text())
        epoch = int(data["epoch_ns"])
        for arrival, length, _, finished in data.get("requests", []):
            if finished and finished - (epoch + arrival) <= slo_ns:
                k = int(arrival // period_ns)
                if skip <= k < periods:
                    out["tokens"][(k // phase) % 2] += length
    return out


def main() -> None:
    jobs, tag, cells = sys.argv[1].split(","), sys.argv[2], sys.argv[3]
    job = "_".join(jobs)
    refs = {int(p.name[len(tag)]): load(p) for j in jobs
            for p in sorted(ROOT.glob(f"{tag}[0-9]n_c{cells}_*_j{j}.json"))}
    rows = {}
    for path in sorted(p for j in jobs for p in ROOT.glob(f"{tag}[0-9]r*_c{cells}_*_j{j}.json")):
        m = re.match(rf"{tag}(\d)r(\d+)(\w+?)_c", path.name)
        seed = int(m.group(1))
        if seed not in refs:
            continue
        run, ref = load(path), refs[seed]
        rows.setdefault((m.group(3), int(m.group(2))), []).append({
            **{f"ai{k}": run["tokens"][k] / run["seconds"][k] for k in (0, 1)},
            **{f"rescue{k}": 100 * run["rescued"][k] / max(1, ref["rescued"][k]) for k in (0, 1)},
            **{f"l1{k}": 100 * run["late"][k] / max(1, run["tbs"][k]) for k in (0, 1)},
            "ai": sum(run["tokens"]) / sum(run["seconds"]),
            "rescue": 100 * sum(run["rescued"]) / max(1, sum(ref["rescued"])),
        })
    print("| policy | AI load | full-load phases: tokens/s, rescued, L1 late | partial phases: tokens/s, rescued, L1 late | whole run: tokens/s, rescued |")
    print("|---|---|---|---|---|")
    out = []
    for (policy, rate), items in sorted(rows.items()):
        mean = {k: float(np.mean([i[k] for i in items])) for k in items[0]}
        spread = {"ai_std": float(np.std([i["ai"] for i in items])), "rescue_std": float(np.std([i["rescue"] for i in items])),
                  "rescue_min": float(min(i["rescue"] for i in items)), "ai_min": float(min(i["ai"] for i in items)),
                  "ai_max": float(max(i["ai"] for i in items)), "rescue_max": float(max(i["rescue"] for i in items)),
                  "l1_full_max": float(max(i["l10"] for i in items))}
        out.append({"policy": policy, "ai_rate": rate, "seeds": len(items), **mean, **spread})
        print(f"| {policy} | {rate} | {mean['ai0'] / 1e3:.1f}k, {mean['rescue0']:.1f}%, {mean['l10']:.3f}% | "
              f"{mean['ai1'] / 1e3:.1f}k, {mean['rescue1']:.1f}%, {mean['l11']:.3f}% | {mean['ai'] / 1e3:.1f}k, {mean['rescue']:.1f}% "
              f"(n={len(items)}, AI {spread['ai_min'] / 1e3:.1f}-{spread['ai_max'] / 1e3:.1f}k, "
              f"rescued {spread['rescue_min']:.1f}-{spread['rescue_max']:.1f}%) |")
    (ROOT.parent / f"phases_{tag}_c{cells}_j{job}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
