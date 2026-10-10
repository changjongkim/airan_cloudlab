#!/usr/bin/env python3
"""What the recoveries are worth to the user: uplink goodput, retransmissions and delivery delay.

usage: mac_goodput.py OUT.json JOB[,JOB]:TAG:CELLS [...]      (runs named as in analyze_sweep.py)

MAC-level accounting on the per-TB records of a run.  The run gives, for every TB, whether the
conventional receiver decoded it and whether the neural receiver recovered it in time.  The
schedule of a TB that is not decoded follows the TDD pattern of the paper:
  slot start T0, L1 result at T0 + 4.5 ms; a retransmission takes the uplink slot at T0 + 7.5 ms
  (three uplink periods later) and each further retransmission 7.5 ms more;
  a TB whose neural receiver run was started waits for it until the recovery deadline (11.5 ms
  after the samples arrive, T0 + 12.0 ms).  The MAC fixes an uplink grant about three slots before
  it goes on air (Aerial: slot_advance 3), so a decision at T0 + 12.0 ms reaches the uplink slot at
  T0 + 15.0 ms, three uplink periods after the conventional retransmission slot (RESCUE_RETX_MS;
  12.5 would assume no lead time, 17.5 is the slot with a lead of five or six slots).
A candidate that got no neural receiver is known before the L1 result is sent (latest start
3.9 ms after arrival), so it is retransmitted at T0 + 7.5 ms like a TB that was never a candidate.
Every retransmission uses an uplink slot of that user that would have carried new data
(full-buffer users), and it succeeds with probability ``RETX_OK`` (soft combining).

Outputs per policy, for the users of the two-user cells and for all users:
  goodput        share of the uplink slots that deliver new data
  retransmissions per 100 TBs
  delay          mean and 99th percentile from slot start to delivery (ms)
  HARQ processes in use per user (a TB holds one from its slot until it is delivered; NR has 16)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from analyze_sweep import policy_name
from loss_trace import RAW

PERIOD_MS = 2.5
RETX_OK = float(__import__("os").environ.get("RETX_OK", "0.95"))
# uplink slot (ms after the slot start) of the retransmission of a TB whose recovery failed
RESCUE_RETX_MS = float(__import__("os").environ.get("RESCUE_RETX_MS", "15.0"))


def tb_rows(path: Path):
    """(weak cell?, conventional ok, recovered in time, neural receiver started, delivery offset of a recovery)"""
    result = json.loads(path.read_text())
    config = result["config"]
    work = Path(str(path)[:-5] + "_work")
    skip = int(config.get("skip_periods", 20))
    rescue_ms = float(config["rescue_deadline_ms"])
    lanes = {}
    for lane in work.glob("lane*.json"):
        for r in json.loads(lane.read_text())["records"]:
            lanes[(r[0], r[1])] = r
    weak, conv_ok, recovered, started, nrx_ms, user, slot = [], [], [], [], [], [], []
    for cell in config["cells"]:
        index, ues = int(cell["cell"]), int(cell.get("num_ue", 1))
        is_weak = cell.get("nrx_gpu") is not None
        for r in json.loads((work / f"conv{index}.json").read_text())["records"]:
            if r[0] < skip:
                continue
            ran = lanes.get((index, r[0]))
            for ue in range(ues):
                ok = bool((r[9] >> ue) & 1) if ues > 1 else bool(r[6])
                rec, ms = False, 0.0
                if ran is not None and not ok:
                    ms = (ran[3] - r[2]) / 1e6
                    good = bool((ran[7] >> ue) & 1) if ues > 1 else bool(ran[5])
                    rec = good and ms <= rescue_ms
                user.append(index * 4 + ue)
                slot.append(r[0])
                weak.append(is_weak)
                conv_ok.append(ok)
                recovered.append(rec)
                started.append(ran is not None and not ok)
                nrx_ms.append(ms)
    return (np.array(weak), np.array(conv_ok), np.array(recovered), np.array(started), np.array(nrx_ms),
            np.array(user), np.array(slot))


def harq_processes(delay, user, slot) -> tuple[float, float, int]:
    """HARQ processes in use per user: a TB holds one from its slot until it is delivered.
    Returns (mean, 99.9th percentile, maximum) over the (user, slot) pairs."""
    hold = np.ceil(delay / PERIOD_MS).astype(int)          # uplink periods a TB holds its process
    counts = []
    for u in np.unique(user):
        pick = user == u
        first, last = int(slot[pick].min()), int(slot[pick].max())
        line = np.zeros(last - first + int(hold[pick].max()) + 2, dtype=np.int32)
        np.add.at(line, slot[pick] - first, 1)
        np.add.at(line, slot[pick] - first + hold[pick], -1)
        counts.append(np.cumsum(line)[:last - first + 1])
    counts = np.concatenate(counts)
    return float(counts.mean()), float(np.percentile(counts, 99.9)), int(counts.max())


def account(conv_ok, recovered, started, nrx_ms, recovery_path: bool, rng, user=None, slot=None) -> dict:
    """Goodput share, retransmissions per 100 TBs, delivery delays and HARQ processes in use."""
    n = len(conv_ok)
    delay = np.full(n, 4.5)
    retx = np.zeros(n)
    if recovery_path:
        done = recovered
        delay[done] = 0.5 + nrx_ms[done]
        first = np.where(started & ~recovered, RESCUE_RETX_MS, 7.5)       # slot of the first retransmission
    else:
        done = np.zeros(n, dtype=bool)
        first = np.full(n, 7.5)
    need = ~conv_ok & ~done
    # Number of retransmissions until one succeeds (at most four).
    count = np.minimum(rng.geometric(RETX_OK, size=n), 4)
    retx[need] = count[need]
    delay[need] = first[need] + (count[need] - 1) * 7.5 + 4.5
    slots = n + retx.sum()
    processes = harq_processes(delay, user, slot) if user is not None else (0.0, 0.0, 0)
    return {
        "harq_processes_mean": processes[0], "harq_processes_p999": processes[1], "harq_processes_max": processes[2],
        "goodput_share": float(n / slots), "retx_per_100": float(100 * retx.sum() / n),
        "not_decoded_first_pct": float(100 * need.mean()),
        "delay_mean_ms": float(delay.mean()), "delay_p99_ms": float(np.percentile(delay, 99)),
        "delay_p999_ms": float(np.percentile(delay, 99.9)),
    }


def main() -> None:
    out_path = Path(sys.argv[1])
    out = {}
    for spec in sys.argv[2:]:
        jobs, tag, cells = spec.split(":")
        pattern = re.compile(rf"^{re.escape(tag)}(\d)(?:n|r(\d+)([a-z0-9]+))_c{cells}_")
        groups: dict[str, list] = {}
        for job in jobs.split(","):
            for path in sorted(RAW.glob(f"{tag}[0-9]*_c{cells}_*_j{job}.json")):
                m = pattern.match(path.name)
                if not m or (m.group(2) and int(m.group(2)) != 32):
                    continue
                groups.setdefault(m.group(3) or "n", []).append(path)
        rows = {}
        for policy, paths in groups.items():
            parts = {"weak": [], "all": []}
            base = {"weak": [], "all": []}
            for path in paths:
                weak, conv_ok, recovered, started, nrx_ms, user, slot = tb_rows(path)
                rng = np.random.default_rng(1)
                for scope, mask in (("weak", weak), ("all", np.ones(len(weak), dtype=bool))):
                    parts[scope].append(account(conv_ok[mask], recovered[mask], started[mask], nrx_ms[mask], True, rng,
                                                user[mask], slot[mask]))
                    if policy == "n":
                        base[scope].append(account(conv_ok[mask], recovered[mask], started[mask], nrx_ms[mask], False, rng,
                                                   user[mask], slot[mask]))
            mean = lambda items: {k: float(np.mean([i[k] for i in items])) for k in items[0]}
            rows[policy] = {"name": "No AI" if policy == "n" else policy_name(policy), "seeds": len(paths),
                            **{scope: mean(parts[scope]) for scope in parts}}
            if policy == "n":
                rows["none"] = {"name": "No recovery path, no AI", "seeds": len(paths),
                                **{scope: mean(base[scope]) for scope in base}}
        out[f"{tag}_c{cells}"] = rows
        ref, none = rows["n"], rows["none"]
        print(f"## {tag} c{cells} (retransmission succeeds with probability {RETX_OK})")
        print("| policy | seeds | two-user cells: goodput vs no recovery path | retransmissions per 100 TBs | delay mean / p99 ms | all cells: goodput vs no recovery path | delay mean / p99 ms | two-user cells: HARQ processes in use, mean / p99.9 / max |")
        print("|---|---|---|---|---|---|---|---|")
        order = ["none", "n"] + sorted(k for k in rows if k not in ("none", "n"))
        for key in order:
            r = rows[key]
            w, a = r["weak"], r["all"]
            print(f"| {r['name']} | {r['seeds']} | {100 * (w['goodput_share'] / none['weak']['goodput_share'] - 1):+.2f}% | "
                  f"{w['retx_per_100']:.2f} | {w['delay_mean_ms']:.2f} / {w['delay_p99_ms']:.1f} | "
                  f"{100 * (a['goodput_share'] / none['all']['goodput_share'] - 1):+.2f}% | "
                  f"{a['delay_mean_ms']:.2f} / {a['delay_p99_ms']:.1f} | "
                  f"{w['harq_processes_mean']:.2f} / {w['harq_processes_p999']:.0f} / {w['harq_processes_max']:.0f} |")
    out_path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
