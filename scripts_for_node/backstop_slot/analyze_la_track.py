#!/usr/bin/env python3
"""Does the outer loop follow an Es/No that changes from slot to slot?

usage: analyze_la_track.py OUT.json JOB[,JOB] TAG:CELLS[:LABEL] ...     (env SEEDS=1,2)

For the runs on a dataset of make_esno_dataset.py (every slot has its own Es/No, meta key esno_true_db): the slots of
the two-user cells are grouped by the Es/No of the slot in bins of 1 dB.  Per policy and bin: the mean MCS of the
slots and the share of the TBs that need a retransmission (not decoded and not recovered before the recovery
deadline).  Policies: x (no recovery path), n (recovery path, no AI), wm (the rule), p30, p70, p100.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from loss_trace import RAW

WARM = 800
POLICIES = ("x", "n", "wm", "p30", "p70", "p100")


def run_bins(path: Path, edges: np.ndarray) -> dict:
    config = json.loads(path.read_text())["config"]
    la = config["la"]
    levels = len(la["levels"])
    mcs = np.asarray(la["mcs"], dtype=float)
    meta = json.loads((Path(config["dataset"]) / f"{la['levels'][0]}_meta.json").read_text())
    esno = np.asarray(meta["esno_true_db"])
    rescue_ns = float(config["rescue_deadline_ms"]) * 1e6
    work = Path(str(path)[:-5] + "_work")
    lanes = {}
    for lane in sorted(work.glob("lane*.json")):
        for r in json.loads(lane.read_text())["records"]:
            lanes[(r[0], r[1])] = r
    slots = np.zeros(len(edges) - 1); mcs_sum = np.zeros(len(edges) - 1); tbs = np.zeros(len(edges) - 1); retx = np.zeros(len(edges) - 1)
    for cell in config["cells"]:
        if cell.get("nrx_gpu") is None:
            continue
        data = json.loads((work / f"conv{cell['cell']}.json").read_text())
        index = np.asarray(data["dataset_indices"])
        for r in data["records"]:
            if r[0] < WARM:
                continue
            b = int(np.searchsorted(edges, esno[index[r[1]]], side="right")) - 1
            b = min(max(b, 0), len(slots) - 1)
            ran = lanes.get((cell["cell"], r[0]))
            saved = ran[7] & ~r[9] & 3 if ran is not None and ran[3] - r[2] <= rescue_ns else 0
            slots[b] += 1; mcs_sum[b] += mcs[r[10] % levels]; tbs[b] += 2
            retx[b] += 2 - bin(r[9] | saved).count("1")
    return {"slots": slots, "mcs": mcs_sum, "tbs": tbs, "retx": retx}


def main() -> None:
    out_path, jobs = Path(sys.argv[1]), sys.argv[2].split(",")
    keep = os.environ.get("SEEDS", "").split(",") if os.environ.get("SEEDS") else None
    out = {}
    for spec in sys.argv[3:]:
        tag, cells, *label = spec.split(":")
        pattern = re.compile(rf"^{re.escape(tag)}(\d)(x|n|r32([a-z0-9]+))_c{cells}_")
        edges = np.arange(13.0, 21.0, 1.0)
        acc: dict[str, dict] = {}
        for job in jobs:
            for path in sorted(RAW.glob(f"{tag}[0-9]*_c{cells}_*_j{job}.json")):
                m = pattern.match(path.name)
                if not m or (keep and m.group(1) not in keep):
                    continue
                policy = m.group(3) or m.group(2)
                if policy not in POLICIES:
                    continue
                b = run_bins(path, edges)
                a = acc.setdefault(policy, {k: np.zeros(len(edges) - 1) for k in b})
                for k in b:
                    a[k] += b[k]
        if not acc:
            continue
        print(f"## {label[0] if label else tag}\n")
        present = [p for p in POLICIES if p in acc]
        print("| Es/No of the slot (dB) | share of the slots | " + " | ".join(f"{p}: mean MCS / retransmissions" for p in present) + " |")
        print("|---|---|" + "---|" * len(present))
        rows = []
        some = acc[present[0]]
        for i in range(len(edges) - 1):
            if some["slots"][i] == 0:
                continue
            cells_out = [f"{acc[p]['mcs'][i] / max(1, acc[p]['slots'][i]):.2f} / {100 * acc[p]['retx'][i] / max(1, acc[p]['tbs'][i]):.1f}%" for p in present]
            print(f"| {edges[i]:.0f}-{edges[i + 1]:.0f} | {100 * some['slots'][i] / some['slots'].sum():.0f}% | " + " | ".join(cells_out) + " |")
            rows.append({"lo": float(edges[i]), "share": float(some["slots"][i] / some["slots"].sum()),
                         **{p: {"mcs": float(acc[p]["mcs"][i] / max(1, acc[p]["slots"][i])),
                                "retx_pct": float(100 * acc[p]["retx"][i] / max(1, acc[p]["tbs"][i]))} for p in present}})
        total = "| all | 100% | " + " | ".join(f"{acc[p]['mcs'].sum() / acc[p]['slots'].sum():.2f} / {100 * acc[p]['retx'].sum() / acc[p]['tbs'].sum():.1f}%" for p in present) + " |"
        print(total + "\n")
        out[tag] = rows
    out_path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
