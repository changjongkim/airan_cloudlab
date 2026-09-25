#!/usr/bin/env python3.11
"""Audit certified scheduler accuracy and 1--64 debt decision latency."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import math
import random
import time
from pathlib import Path

from c162_certified_scheduler_v1 import (
    certificate_digest,
    certified_schedule,
    verify_certificate,
)
from c162_feasibility_model_v1 import AI_BOUNDS_MS, EnvelopePoint, exact_flags
from shared_recovery_certificate_v1 import (
    RecoveryObligation,
    SharedAILease,
    exact_certificate,
)


MS = 1_000_000
SOURCES = (
    "c162_certified_scheduler_v1.py",
    "test_c162_certified_scheduler_v1.py",
    "run_c162_scheduler_scalability_v1.py",
    "c162_feasibility_model_v1.py",
    "shared_recovery_certificate_v1.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)] if ordered else None


def summary(values):
    return {
        "count": len(values), "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, .50), "p99": percentile(values, .99),
        "max": max(values) if values else None,
    }


def small_exact_audit(seed: int, cases_per_size: int = 100):
    rng = random.Random(seed)
    rows = []
    for count in range(1, 7):
        for sample in range(cases_per_size):
            capacity = rng.choice(tuple(range(1, min(count, 3) + 1)))
            now = rng.randint(0, 10) * MS
            jobs = []
            for index in range(count):
                release = rng.randint(0, 40) * MS
                service = rng.choice((5, 10, 15, 25)) * MS
                deadline = release + service + rng.randint(0, 90) * MS
                jobs.append(RecoveryObligation(
                    f"h{index % 3}", f"r{sample}-{index}",
                    release, deadline, service,
                ))
            leases = ()
            if rng.random() < .4:
                start = rng.randint(5, 45) * MS
                leases = (SharedAILease(
                    f"a{sample}", "global", start,
                    start + rng.randint(5, 20) * MS,
                ),)
            exact = exact_certificate(jobs, capacity, leases, now)
            started = time.perf_counter_ns()
            candidate = certified_schedule(jobs, capacity, leases, now)
            latency_us = (time.perf_counter_ns() - started) / 1e3
            valid = candidate is None or verify_certificate(
                jobs, candidate, capacity, leases, now
            )[0]
            rows.append({
                "count": count, "capacity": capacity,
                "exact_accept": exact is not None,
                "certified_accept": candidate is not None,
                "certificate_valid": valid, "latency_us": latency_us,
            })
    return rows


def current_mode_grid():
    contexts = (None,) + tuple(AI_BOUNDS_MS)
    mismatches = 0
    count = 0
    for accepted in range(6):
        for unresolved in range(accepted + 1):
            for decision in range(45, 154):
                for context in contexts:
                    point = EnvelopePoint(accepted, unresolved, decision, context)
                    exact_all, exact_current, exact_ai, _ = exact_flags(point)
                    all_jobs = tuple(RecoveryObligation(
                        f"h{i % 2}", f"a{i}", 45 * MS, 153 * MS, 25 * MS
                    ) for i in range(accepted))
                    current_jobs = tuple(RecoveryObligation(
                        f"h{i % 2}", f"u{i}", 45 * MS, 153 * MS, 25 * MS
                    ) for i in range(unresolved))
                    fast_all = certified_schedule(all_jobs, 1, now_ns=45 * MS) is not None
                    fast_current = certified_schedule(
                        current_jobs, 1, now_ns=decision * MS
                    ) is not None
                    fast_ai = False
                    if (context is not None and fast_all and fast_current):
                        finish = decision + 5 + AI_BOUNDS_MS[context]
                        if finish <= 153:
                            lease = SharedAILease(
                                "ai", "global", decision * MS, finish * MS
                            )
                            fast_ai = certified_schedule(
                                current_jobs, 1, (lease,), decision * MS
                            ) is not None
                    mismatches += int((fast_all, fast_current, fast_ai)
                                      != (exact_all, exact_current, exact_ai))
                    count += 1
    return count, mismatches


def scale_audit(seed: int, samples: int = 100):
    rng = random.Random(seed)
    rows = []
    for count in (1, 2, 4, 8, 16, 32, 64):
        for capacity in (1, 2, 4, 8):
            for sample in range(samples):
                now = 5 * MS
                leases = ()
                blackout = 0
                if sample % 3:
                    start = (20 + sample % 10) * MS
                    blackout = 5 + sample % 11
                    leases = (SharedAILease(
                        f"scale-a{count}-{capacity}-{sample}", "global",
                        start, start + blackout * MS,
                    ),)
                services = [rng.choice((2, 4, 6, 8, 10)) for _ in range(count)]
                work = sum(services) / capacity
                pressure = (.75, 1.0, 1.25)[sample % 3]
                horizon = max(25, math.ceil(work * pressure + blackout + 15))
                jobs = tuple(RecoveryObligation(
                    f"h{i % min(16, max(1, count))}",
                    f"r{count}-{capacity}-{sample}-{i}",
                    rng.randint(0, 15) * MS,
                    max(20, horizon - rng.randint(0, 8)) * MS,
                    services[i] * MS,
                ) for i in range(count))
                started = time.perf_counter_ns()
                candidate = certified_schedule(jobs, capacity, leases, now)
                decided = time.perf_counter_ns()
                valid, reason = ((True, "rejected") if candidate is None
                                 else verify_certificate(jobs, candidate, capacity, leases, now))
                verified = time.perf_counter_ns()
                repeated = certified_schedule(jobs, capacity, leases, now)
                deterministic = (
                    (candidate is None and repeated is None)
                    or (candidate is not None and repeated is not None
                        and certificate_digest(candidate) == certificate_digest(repeated))
                )
                rows.append({
                    "debts": count, "capacity": capacity, "sample": sample,
                    "accepted": candidate is not None,
                    "certificate_valid": valid, "verify_reason": reason,
                    "deterministic": deterministic,
                    "decision_us": (decided - started) / 1e3,
                    "verify_us": (verified - decided) / 1e3,
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=16200001)
    args = parser.parse_args()
    small = small_exact_audit(args.seed)
    grid_count, grid_mismatch = current_mode_grid()
    scale = scale_audit(args.seed + 1)
    false_safe = sum(row["certified_accept"] and not row["exact_accept"] for row in small)
    false_conservative = sum(row["exact_accept"] and not row["certified_accept"] for row in small)
    by_debt = {}
    for debt in (1, 2, 4, 8, 16, 32, 64):
        subset = [row for row in scale if row["debts"] == debt]
        by_debt[str(debt)] = {
            "states": len(subset), "accepted": sum(row["accepted"] for row in subset),
            "decision_us": summary([row["decision_us"] for row in subset]),
            "verify_us": summary([row["verify_us"] for row in subset if row["accepted"]]),
        }
    debt64 = [row for row in scale if row["debts"] == 64]
    gates = {
        "unit_scale_small_exact_600": len(small) == 600,
        "small_exact_false_safe_zero": false_safe == 0,
        "every_returned_small_certificate_verifies": all(row["certificate_valid"] for row in small),
        "current_mode_16023_acceptance_mismatch_zero": grid_count == 16023 and grid_mismatch == 0,
        "large_grid_2800_states": len(scale) == 2800,
        "every_returned_large_certificate_verifies": all(row["certificate_valid"] for row in scale),
        "deterministic_certificate": all(row["deterministic"] for row in scale),
        "large_grid_has_accept_and_reject": any(row["accepted"] for row in scale) and any(not row["accepted"] for row in scale),
        "debt64_decision_p99_below_5ms": percentile([row["decision_us"] for row in debt64], .99) < 5000,
        "debt64_verify_p99_below_5ms": percentile([row["verify_us"] for row in debt64 if row["accepted"]], .99) < 5000,
    }
    value = {
        "schema": "softwall-c162-certified-scheduler-scalability-v1",
        "status": "C162_SCHEDULER_SCALABILITY_PASS" if all(gates.values()) else "C162_SCHEDULER_SCALABILITY_FAIL",
        "all_pass": all(gates.values()), "gates": gates, "seed": args.seed,
        "small_exact": {
            "states": len(small), "exact_accept": sum(row["exact_accept"] for row in small),
            "certified_accept": sum(row["certified_accept"] for row in small),
            "false_safe": false_safe, "false_conservative": false_conservative,
            "latency_us": summary([row["latency_us"] for row in small]),
        },
        "current_qualified_mode": {"states": grid_count, "acceptance_mismatch": grid_mismatch},
        "large_grid": {
            "states": len(scale), "accepted": sum(row["accepted"] for row in scale),
            "rejected": sum(not row["accepted"] for row in scale),
            "by_debt": by_debt,
        },
        "scope": "CPU finite-run control-plane scalability. A verified returned schedule is safe for supplied bounds; rejection may be conservative. Latency is not a production WCET.",
        "source_sha256": {name: sha256(args.scripts_root / name) for name in SOURCES},
        "small_rows": small, "scale_rows": scale,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(args.output)
    print(json.dumps({
        "status": value["status"], "gates": gates,
        "small_exact": value["small_exact"],
        "current_qualified_mode": value["current_qualified_mode"],
        "large_grid": {key: value["large_grid"][key] for key in ("states", "accepted", "rejected", "by_debt")},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
