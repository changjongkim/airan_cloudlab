#!/usr/bin/env python3.11
"""Service-path tails of every run of a placement, failed and padded runs included.

The paired comparison covers completed natural runs. Qualification needs
every execution: this script reads the Qwen, owner, and coordinator logs of
every run whose coordinator log matches the given globs and reports, per
service path, the sample count, p50, p99, maximum, declared bound, and the
executions above the bound. It also counts runs whose first period lost a
NeuralRx outcome at the cutoff and runs that stopped with an error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from c159_q2_classes import CLASS_BOUNDS_MS

NRX_BOUND_MS = 45.0
RECOVERY_BOUND_MS = 25.0


def stats(values, bound):
    values = sorted(values)
    if not values:
        return None
    pick = lambda q: values[round((len(values) - 1) * q)]
    return {"count": len(values), "p50": pick(0.5), "p99": pick(0.99), "max": values[-1], "bound": bound,
            "over_bound": sum(1 for value in values if value > bound)}


def placement(raw: Path, globs: list) -> dict:
    paths = sorted({path for pattern in globs for path in raw.glob(pattern)})
    ai, nrx, recovery = {}, [], []
    first_missing, failed, overruns = [], [], []
    radio_misses, commits = [], 0
    for path in paths:
        label = path.name.replace("_coordinator.json", "")
        coordinator = json.loads(path.read_text())
        if coordinator["error"]:
            failed.append({"label": label, "error": coordinator["error"], "completed_rounds": coordinator["completed_rounds"]})
        if coordinator["rounds"] and coordinator["rounds"][0]["missing_outcomes_at_cutoff"]:
            first_missing.append({"label": label, "missing": len(coordinator["rounds"][0]["missing_outcomes_at_cutoff"])})
        recovery += [row["prepad_path_ms"] for row in coordinator["physical_recoveries"]]
        for owner in raw.glob(f"{label}_home*_owner.json"):
            records = json.loads(owner.read_text())["records"]
            nrx += [row["nrx_release_to_complete_ms"] for row in records
                    if row.get("nrx_release_to_complete_ms") is not None]
            commits += sum(row["commit_count"] for row in records)
            radio_misses += [{"label": label, "sequence": row["sequence"],
                              "release_to_commit_ms": row.get("release_to_commit_ms")}
                             for row in records if row.get("deadline_miss")]
        qwen = raw / f"{label}_qwen.json"
        if qwen.exists():
            log = json.loads(qwen.read_text())
            for row in log.get("requests") or log.get("records") or []:
                if row.get("completed_ns") is None:
                    continue
                wall = (row["completed_ns"] - row["accepted_ns"]) / 1e6
                ai.setdefault(row["context_length"], []).append(wall)
                if wall > CLASS_BOUNDS_MS[row["context_length"]]:
                    overruns.append({"label": label, "context_length": row["context_length"], "ms": round(wall, 3),
                                     "bound_ms": CLASS_BOUNDS_MS[row["context_length"]]})
    return {
        "runs": len(paths),
        "nrx_path_ms": stats(nrx, NRX_BOUND_MS),
        "recovery_path_ms": stats(recovery, RECOVERY_BOUND_MS),
        "ai_execution_ms": {str(context): stats(values, CLASS_BOUNDS_MS[context])
                            for context, values in sorted(ai.items())},
        "ai_executions": sum(len(values) for values in ai.values()),
        "ai_overruns": overruns,
        "first_period_missing_outcomes": first_missing,
        "failed_runs": failed,
        "radio_commits": commits,
        "radio_deadline_misses": radio_misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--multi", nargs="+", required=True, help="coordinator-log globs of the multi-GPU runs")
    parser.add_argument("--single", nargs="+", required=True, help="coordinator-log globs of the single-GPU runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema": "softwall-c177-tail-summary-v1",
        "multi": placement(args.raw, args.multi),
        "single": placement(args.raw, args.single),
        "globs": {"multi": args.multi, "single": args.single},
        "note": ("AI execution is the Qwen worker's accepted-to-completed time against the class bound; the NeuralRx "
                 "path is release to result and the recovery path is the pre-padding service path."),
    }
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    for name in ("multi", "single"):
        part = result[name]
        print(name, "runs", part["runs"], "AI executions", part["ai_executions"], "overruns", len(part["ai_overruns"]),
              "first-period missing", len(part["first_period_missing_outcomes"]), "failed", len(part["failed_runs"]),
              "radio misses", len(part["radio_deadline_misses"]), "of", part["radio_commits"])
        print("   NRx", part["nrx_path_ms"], "\n   recovery", part["recovery_path_ms"])


if __name__ == "__main__":
    main()
