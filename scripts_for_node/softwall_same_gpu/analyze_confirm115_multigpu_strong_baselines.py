#!/usr/bin/env python3
"""Audit the frozen same-budget multi-GPU strong-baseline campaign."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
RESULTS = ROOT / "results/softwall_multigpu"
RAW = RESULTS / "raw"
PROTOCOL = RESULTS / "confirm115_multigpu_strong_baselines_protocol.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def radio_signature(result: dict) -> list[tuple]:
    return [
        (
            row["index"], row["cell"], row["gate_skipped"], row["admitted"],
            row["forced_nrx_failure"], row["commit_kind"],
        )
        for row in result["records"]
    ]


def timely_by_request(result: dict) -> dict[str, int]:
    return {
        row["request_id"]: row["value_tokens"] if row["timely"] else 0
        for row in result["background_records"]
    }


def compare_pair(pair: dict, arms: dict, trace: dict, seed: int) -> dict:
    soft = [arms[name] for name in pair["softwall"]]
    work = [arms[name] for name in pair["work_conserving"]]
    soft_values = [timely_by_request(item) for item in soft]
    work_values = [timely_by_request(item) for item in work]
    blocks = defaultdict(float)
    soft_total = 0.0
    work_total = 0.0
    for request in trace["requests"]:
        request_id = request["request_id"]
        soft_value = sum(v.get(request_id, 0) for v in soft_values) / len(soft_values)
        work_value = sum(v.get(request_id, 0) for v in work_values) / len(work_values)
        blocks[request["source_timestamp_s"]] += soft_value - work_value
        soft_total += soft_value
        work_total += work_value
    block_values = list(blocks.values())
    rng = random.Random(seed)
    bootstrap = [sum(rng.choice(block_values) for _ in block_values) for _ in range(10_000)]
    delta = soft_total - work_total
    effect_pct = 100.0 * delta / work_total if work_total else 0.0
    lower = percentile(bootstrap, 0.025)
    upper = percentile(bootstrap, 0.975)
    return {
        "name": pair["name"],
        "softwall_arms": pair["softwall"],
        "work_conserving_arms": pair["work_conserving"],
        "softwall_mean_timely_value": soft_total,
        "work_conserving_mean_timely_value": work_total,
        "delta_tokens": delta,
        "effect_pct": effect_pct,
        "paired_source_second_bootstrap_95_ci": [lower, upper],
        "effect_at_least_2pct": effect_pct >= 2.0,
        "bootstrap_lower_positive": lower > 0,
    }


def main() -> None:
    protocol = read(PROTOCOL)
    trace_path = ROOT / protocol["trace"]["path"]
    trace = read(trace_path)
    source_audit = {
        relative: {
            "expected": expected,
            "observed": sha256(ROOT / relative),
            "match": sha256(ROOT / relative) == expected,
        }
        for relative, expected in protocol["source_sha256_before_run"].items()
    }
    arms = {}
    audit = {}
    for spec in protocol["arms"]:
        wrapper_path = RESULTS / f"{spec['label']}_result.json"
        wrapper = read(wrapper_path)
        controller_path = Path(wrapper["controller"])
        controller = read(controller_path)
        arms[spec["name"]] = controller
        artifact_audit = {
            path: {"expected": expected, "observed": sha256(Path(path)),
                   "match": sha256(Path(path)) == expected}
            for path, expected in wrapper["artifact_sha256"].items()
        }
        audit[spec["name"]] = {
            "wrapper": str(wrapper_path.relative_to(ROOT)),
            "wrapper_sha256": sha256(wrapper_path),
            "system_expected": spec["system"],
            "system_observed": controller["system"],
            "all_arm_gates_pass": wrapper["all_pass"],
            "artifacts_unchanged": all(x["match"] for x in artifact_audit.values()),
            "timely_value_tokens": controller["trace_summary"]["timely_value_tokens"],
            "timely_requests": controller["trace_summary"]["timely_requests"],
            "correct_cells": controller["correct_cells"],
            "recovery_retime_count": controller["recovery_retime_count"],
            "joint_lease_retired_count": controller["joint_lease_retired_count"],
            "remote_worker_completed": read(Path(wrapper["remote_worker"]))["completed_units"],
        }
    parity = {}
    for seed_name, names in protocol["radio_parity_groups"].items():
        reference = radio_signature(arms[names[0]])
        parity[seed_name] = {
            "arms": names,
            "structural_signature_equal": all(radio_signature(arms[name]) == reference for name in names[1:]),
            "correct_cells": {name: arms[name]["correct_cells"] for name in names},
        }
    comparisons = [
        compare_pair(pair, arms, trace, 115_000 + index)
        for index, pair in enumerate(protocol["paired_comparisons"])
    ]
    provenance = (
        sha256(trace_path) == protocol["trace"]["sha256"]
        and all(item["match"] for item in source_audit.values())
        and all(item["artifacts_unchanged"] for item in audit.values())
        and all(item["system_expected"] == item["system_observed"] for item in audit.values())
    )
    safety = all(item["all_arm_gates_pass"] for item in audit.values())
    structural = all(item["structural_signature_equal"] for item in parity.values())
    outcome = all(x["effect_at_least_2pct"] and x["bootstrap_lower_positive"] for x in comparisons)
    result = {
        "schema":"softwall-confirm115-multigpu-strong-baselines-v1",
        "protocol_sha256":sha256(PROTOCOL),
        "trace_sha256_observed":sha256(trace_path),
        "source_audit":source_audit,
        "arms":audit,
        "radio_parity":parity,
        "paired_comparisons":comparisons,
        "gates":{
            "provenance":provenance,
            "safety":safety,
            "radio_structural_parity":structural,
            "outcome":outcome,
            "all_pass":provenance and safety and structural and outcome,
        },
        "interpretation":"The outcome gate is deliberately identical to C113. A failure ends the multi-GPU throughput-superiority claim but preserves a passing transport-independent safety result.",
    }
    output=RESULTS/f"confirm115_multigpu_strong_baselines_job{protocol['job']}.json"
    output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result["gates"],indent=2))


if __name__ == "__main__":
    main()
