#!/usr/bin/env python3.11
"""Summarize the C176 bursty-arrival campaign.

For every run (pattern, policy, execution mode) the analyzer reports the AI
requests that arrived, were served, and finished within their SLO; the on-time
AI tokens per second; the time to first token of served requests; the periods
whose admitted schedule breaks the guard at the declared bounds; the periods
whose last recovery physically ended after the guard; and the radio commits
that missed the result expiry. It also checks that every policy of a pattern
and mode saw the same NeuralRx outcome sequence, which makes the comparison
paired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

LABEL = re.compile(r"^(?P<prefix>c176)_(?P<pattern>[a-z0-9_]+?)_(?P<policy>backstop|recovery_first|idle_time)_"
                   r"(?P<mode>natural|padded)_j(?P<job>\d+)_coordinator\.json$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize_run(path: Path, match) -> dict:
    run = json.loads(path.read_text())
    owners = sorted(path.parent.glob(path.name.replace("_coordinator.json", "_home*_owner.json")))
    if len(owners) != 4:
        raise SystemExit(f"{path.name}: expected four owner logs, found {len(owners)}")
    commits = misses = 0
    for owner in owners:
        records = json.loads(owner.read_text())["records"]
        commits += sum(record["commit_count"] for record in records)
        misses += sum(bool(record["deadline_miss"]) for record in records)
    rounds = run["rounds"]
    duration_s = run["iterations"] * run["period_ms"] / 1000.0
    served = [row for row in run["requests"] if row["status"] == "served"]
    on_time = [row for row in served if row.get("on_time")]
    trace_requests = run["ai_trace"]["requests"]
    outcome_sequence = [sorted(tuple(key) for key in row["success_keys"]) for row in rounds]
    return {
        "label": path.name.replace("_coordinator.json", ""),
        "pattern": match["pattern"], "policy": match["policy"], "mode": match["mode"], "job": match["job"],
        "host": run["host"], "error": run["error"], "completed_rounds": run["completed_rounds"],
        "iterations": run["iterations"], "slo_ms": run["slo_ms"], "max_leases": run["max_leases"],
        "trace_requests": trace_requests, "requests_arrived": run["requests_arrived"],
        "requests_served": len(served), "requests_on_time": len(on_time),
        "requests_refused_at_launch": sum(1 for row in run["requests"] if row["status"] == "refused"),
        "requests_expired": sum(1 for row in run["requests"] if row["status"] == "expired"),
        "on_time_fraction": len(on_time) / trace_requests,
        "on_time_tokens_per_s": sum(row["value_tokens"] for row in on_time) / duration_s,
        "offered_tokens_per_s": None,
        "ttft_p50_ms": percentile([row["ttft_ms"] for row in served], 0.5),
        "ttft_p99_ms": percentile([row["ttft_ms"] for row in served], 0.99),
        "ai_units_per_period": len(served) / max(1, len(rounds)),
        "contract_broken_periods": sum(bool(row["contract_broken"]) for row in rounds),
        "physical_guard_misses": sum(bool(row["physical_guard_miss"]) for row in rounds),
        "latest_recovery_complete_ms": max((row["last_recovery_complete_ms"] for row in rounds
                                            if row["last_recovery_complete_ms"] is not None), default=None),
        "radio_commits": commits, "radio_deadline_misses": misses,
        "required_recoveries": sum(len(row["unresolved_keys"]) for row in rounds),
        "late_launches": sum(1 for row in rounds for attempt in row["lease_attempts"] if not attempt["launched"]),
        "outcome_sequence_sha256": hashlib.sha256(json.dumps(outcome_sequence).encode()).hexdigest(),
        "coordinator_sha256": sha256(path),
        "owner_sha256": {owner.name: sha256(owner) for owner in owners},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--jobs", nargs="+", required=True, help="Slurm job ids of the campaign")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = []
    for path in sorted(args.raw.glob("c176_*_coordinator.json")):
        match = LABEL.match(path.name)
        if match and match["job"] in args.jobs:
            runs.append(summarize_run(path, match))
    if not runs:
        raise SystemExit("no C176 runs found for the given jobs")
    trace_summary = json.loads((args.traces / "c176_trace_summary.json").read_text())
    for run in runs:
        trace = json.loads((args.traces / f"c176_trace_{run['pattern']}.json").read_text())
        run["offered_tokens_per_s"] = (sum(row["value_tokens"] for row in trace["requests"])
                                       / (run["iterations"] * 0.18))
        run["interarrival_cv"] = trace["summary"]["interarrival_cv"]
        run["trace_sha256"] = sha256(args.traces / f"c176_trace_{run['pattern']}.json")

    # Paired radio: one NeuralRx outcome sequence per pattern, mode, and node.
    paired = {}
    for run in runs:
        paired.setdefault((run["pattern"], run["mode"], run["host"]), set()).add(run["outcome_sequence_sha256"])
    comparisons = []
    for (pattern, mode) in sorted({(run["pattern"], run["mode"]) for run in runs}):
        group = {run["policy"]: run for run in runs if run["pattern"] == pattern and run["mode"] == mode}
        if {"backstop", "recovery_first"} <= set(group):
            backstop, recovery = group["backstop"], group["recovery_first"]
            comparisons.append({
                "pattern": pattern, "mode": mode,
                "backstop_over_recovery_first_tokens": (backstop["on_time_tokens_per_s"]
                                                         / max(recovery["on_time_tokens_per_s"], 1e-9)),
                "backstop_over_idle_time_tokens": (backstop["on_time_tokens_per_s"]
                                                    / max(group["idle_time"]["on_time_tokens_per_s"], 1e-9)
                                                    if "idle_time" in group else None),
            })
    result = {
        "schema": "softwall-c176-burst-campaign-v1",
        "jobs": args.jobs,
        "trace_source": trace_summary["source"],
        "runs": sorted(runs, key=lambda run: (run["mode"], run["interarrival_cv"], run["policy"])),
        "paired_outcomes_identical": {f"{k[0]}|{k[1]}|{k[2]}": len(v) == 1 for k, v in paired.items()},
        "comparisons": comparisons,
        "all_runs_completed": all(run["error"] is None and run["completed_rounds"] == run["iterations"]
                                  for run in runs),
        "certified_policies_safe": all(run["contract_broken_periods"] == 0 and run["physical_guard_misses"] == 0
                                       and run["radio_deadline_misses"] == 0
                                       for run in runs if run["policy"] != "idle_time"),
        "claim_scope": ("Two-node P180/D155 mode, one node per execution mode, one run per pattern and policy. "
                        "Natural runs use measured service times; padded runs hold every AI unit and recovery on "
                        "the GPU until its declared bound (bound-realization diagnostic). Static reservation "
                        "admits no AI unit in this mode and is not run."),
    }
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    for run in result["runs"]:
        print(f"{run['mode']:7s} {run['pattern']:10s} cv={run['interarrival_cv']:.2f} {run['policy']:15s} "
              f"on-time={100 * run['on_time_fraction']:5.1f}% tok/s={run['on_time_tokens_per_s']:6.1f} "
              f"p50={run['ttft_p50_ms'] or 0:6.1f} p99={run['ttft_p99_ms'] or 0:6.1f} "
              f"broken={run['contract_broken_periods']:3d} guard={run['physical_guard_misses']:3d} "
              f"miss={run['radio_deadline_misses']:3d} err={run['error']}")
    print("paired outcomes identical:", result["paired_outcomes_identical"])
    print("certified policies safe:", result["certified_policies_safe"], "all completed:", result["all_runs_completed"])


if __name__ == "__main__":
    main()
