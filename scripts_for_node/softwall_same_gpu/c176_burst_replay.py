#!/usr/bin/env python3.11
"""C176 measured-distribution replay of bursty AI arrivals in the P180/D155 mode.

The replay drives the admission policies of the physical C176 coordinator with
service times drawn from the Q2 two-node campaign: Qwen prefill execution per
prompt length, recovery path time, the per-period number of required
recoveries, and the decision-to-launch control delay. It predicts the physical
campaign and extends it to patterns that are not run on GPUs. Every policy
sees the same arrivals, prompt lengths, recovery counts, and service draws.

Policies (identical deadlines, guard, bounds, and FIFO visiting order):
- backstop: AI leases first while the all-fail recovery schedule still ends by
  the guard and each lease ends by its request deadline;
- recovery_first: every required recovery first at its bound, then AI leases;
- idle_time: AI leases while the lane is idle at the decision, ignoring pending
  recovery; the contract is broken when the leases plus the required
  recoveries at their bounds end after the guard;
- static: holds the recovery of every admitted TB until expiry.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

PERIOD_MS = 180.0
DECISION_MS = 45.0
GUARD_END_MS = 153.0
RECOVERY_BOUND_MS = 25.0
CONTROL_MS = 5.0
ADMITTED_TBS = 4
CLASS_BOUND_MS = {16: 35, 32: 35, 64: 35, 128: 40, 256: 65, 512: 75}
POLICIES = ("backstop", "recovery_first", "idle_time", "static")


def load_q2(raw: Path) -> dict:
    """Empirical service distributions from the Q2 coordinator logs."""
    execution = {c: [] for c in CLASS_BOUND_MS}
    recovery_ms, required, launch_ms = [], [], []
    for name in ("confirm159q2f_batch_dev_j58857672_coordinator.json",
                 "confirm159q2g_batch_holdout_j58858194_coordinator.json"):
        run = json.loads((raw / name).read_text())
        for record in run["physical_recoveries"]:
            recovery_ms.append((record["actual_completed_ns"] - record["actual_start_ns"]) / 1e6)
        for row in run["rounds"]:
            required.append(len(row["outcome_transition"]["unresolved_obligations"]))
            qwen = row["qwen"] if row["lease_accepted"] else None
            if qwen:
                execution[row["context_length"]].append(qwen["execution_ms"])
                # Decision (outcome batch) to worker acceptance.
                decision_ns = row["release_wall_ns"] + row["outcome_transition"]["applied_at_ns"]
                launch_ms.append((qwen["worker_accepted_ns"] - decision_ns) / 1e6)
    return {"execution": execution, "recovery": recovery_ms, "required": required, "launch": launch_ms}


def plan(policy: str, queue: list, decision_ms: float, required: int, release_ms: float,
         slo_ms: float, max_leases: int) -> tuple[list, bool]:
    """Leases (request, certified start, certified end) and the contract flag."""
    guard = release_ms + GUARD_END_MS
    leases = []
    if policy == "static":
        # Static reservation holds all admitted TBs: 4 x 25 ms of the 108 ms window.
        cursor = decision_ms
        reserve = ADMITTED_TBS * RECOVERY_BOUND_MS
    elif policy == "recovery_first":
        cursor = decision_ms + required * RECOVERY_BOUND_MS
        reserve = 0.0
    elif policy == "backstop":
        cursor = decision_ms
        reserve = required * RECOVERY_BOUND_MS
    else:  # idle_time ignores pending recovery
        cursor = decision_ms
        reserve = 0.0
    for request in queue:
        if len(leases) >= max_leases:
            break
        end = cursor + CONTROL_MS + CLASS_BOUND_MS[request["context_length"]]
        if end + reserve > guard or end > request["arrival_ms"] + slo_ms:
            continue
        leases.append((request, cursor, end))
        cursor = end
    broken = False
    if policy == "idle_time" and leases:
        broken = leases[-1][2] + required * RECOVERY_BOUND_MS > guard
    return leases, broken


def replay(trace: dict, policy: str, dist: dict, slo_ms: float, seed: int, max_leases: int,
           periods: int, padded: bool = False) -> dict:
    rng = random.Random(seed)
    requests = sorted(trace["requests"], key=lambda r: r["arrival_ms"])
    pending, served, dropped = [], [], []
    broken_periods = physical_misses = 0
    index = 0
    for k in range(periods):
        release = k * PERIOD_MS
        decision = release + DECISION_MS
        while index < len(requests) and requests[index]["arrival_ms"] <= decision:
            pending.append(requests[index]); index += 1
        # Drop requests that can no longer meet their deadline.
        keep = []
        for request in pending:
            fastest = decision + CONTROL_MS + CLASS_BOUND_MS[request["context_length"]]
            (keep if fastest <= request["arrival_ms"] + slo_ms else dropped).append(request)
        pending = keep
        required = rng.choice(dist["required"])
        leases, broken = plan(policy, pending, decision, required, release, slo_ms, max_leases)
        broken_periods += broken
        # Work-conserving execution in certificate order with measured times.
        clock = decision + rng.choice(dist["launch"])
        recoveries = [RECOVERY_BOUND_MS if padded else rng.choice(dist["recovery"]) for _ in range(required)]
        if policy == "recovery_first":
            clock += sum(recoveries)
        for request, _, _ in leases:
            clock += (CLASS_BOUND_MS[request["context_length"]] if padded
                      else rng.choice(dist["execution"][request["context_length"]]))
            served.append({"request": request, "completion_ms": clock})
            pending.remove(request)
        if policy != "recovery_first":
            clock += sum(recoveries)
        physical_misses += clock > release + GUARD_END_MS and required > 0
    dropped.extend(pending)
    dropped.extend(requests[index:])
    duration_s = periods * PERIOD_MS / 1000.0
    latencies = [s["completion_ms"] - s["request"]["arrival_ms"] for s in served]
    on_time = [s for s, lat in zip(served, latencies) if lat <= slo_ms]
    return {
        "policy": policy, "served": len(served), "on_time": len(on_time), "dropped": len(dropped),
        "requests": len(requests),
        "on_time_fraction": len(on_time) / len(requests),
        "on_time_tokens_per_s": sum(s["request"]["value_tokens"] for s in on_time) / duration_s,
        "ttft_p50_ms": statistics.median(latencies) if latencies else None,
        "ttft_p99_ms": sorted(latencies)[int(0.99 * (len(latencies) - 1))] if latencies else None,
        "contract_broken_periods": broken_periods, "physical_guard_misses": physical_misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--slo-ms", type=float, nargs="+", default=[300.0])
    parser.add_argument("--seed", type=int, default=17600101)
    parser.add_argument("--max-leases", type=int, default=4)
    parser.add_argument("--periods", type=int, default=600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--padded", action="store_true", help="every unit and recovery takes its declared bound")
    args = parser.parse_args()
    dist = load_q2(args.raw)
    rows = []
    for path in args.traces:
        trace = json.loads(path.read_text())
        for slo in args.slo_ms:
            for policy in POLICIES:
                result = replay(trace, policy, dist, slo, args.seed, args.max_leases, args.periods, args.padded)
                result.update({"pattern": trace["summary"]["pattern"], "slo_ms": slo,
                               "interarrival_cv": trace["summary"]["interarrival_cv"]})
                rows.append(result)
    if args.output:
        args.output.write_text(json.dumps({"schema": "softwall-c176-replay-v1", "rows": rows}, indent=1) + "\n")
    for row in rows:
        print(f"{row['pattern']:10s} slo={row['slo_ms']:5.0f} {row['policy']:15s} "
              f"on-time={100 * row['on_time_fraction']:5.1f}% tok/s={row['on_time_tokens_per_s']:7.1f} "
              f"p50={row['ttft_p50_ms'] or 0:6.1f} p99={row['ttft_p99_ms'] or 0:6.1f} "
              f"broken={row['contract_broken_periods']:3d} phys={row['physical_guard_misses']}")


if __name__ == "__main__":
    main()
