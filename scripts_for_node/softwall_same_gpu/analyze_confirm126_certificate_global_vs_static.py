#!/usr/bin/env python3
"""Audit the conv25 ABBA global-broker versus static-partition campaign."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
RES = ROOT / "results/softwall_multigpu"
PROTOCOL = RES / "confirm126_certificate_global_vs_static_protocol.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def radio_signature(result: dict) -> list:
    signature = []
    for home in result["homes"]:
        controller = read(Path(home["controller"]))
        signature.extend([
            home["home"], row["index"], row["cell"], row["gate_skipped"],
            row["admitted"], row["endpoint_id"], row["forced_nrx_failure"],
            row["commit_kind"],
        ] for row in controller["records"])
    return signature


def timely_by_second(broker: dict, seconds: int = 60) -> list[float]:
    values = [0.0] * seconds
    for request in broker["requests"]:
        if request["state"] != "completed":
            continue
        second = min(seconds - 1, int(math.floor(request["arrival_ms"] / 1000.0)))
        values[second] += request["value_tokens"]
    return values


def bootstrap_ci(delta: list[float], seed: int, draws: int = 20000) -> list[float]:
    rng = random.Random(seed)
    n = len(delta)
    values = sorted(sum(delta[rng.randrange(n)] for _ in range(n)) for _ in range(draws))
    return [values[round((draws - 1) * 0.025)],
            values[round((draws - 1) * 0.975)]]


def main() -> None:
    protocol = read(PROTOCOL)
    source_audit = {
        path: sha256(ROOT / path) == expected
        for path, expected in protocol["source_sha256_before_run"].items()
    }
    arms = []
    by_seed = defaultdict(lambda: defaultdict(list))
    correct_by_seed = defaultdict(lambda: defaultdict(list))
    signatures = defaultdict(list)
    for spec in protocol["arms"]:
        path = RES / f"{spec['label']}_result.json"
        result = read(path)
        broker = read(Path(result["broker"]))
        artifacts_unchanged = all(
            sha256(Path(name)) == expected
            for name, expected in result["artifact_sha256"].items()
        )
        values = timely_by_second(broker)
        by_seed[spec["seed"]][spec["policy"]].append(values)
        correct = sum(home["summary"]["correct_cells"] for home in result["homes"])
        correct_by_seed[spec["seed"]][spec["policy"]].append(correct)
        signatures[spec["seed"]].append(radio_signature(result))
        arms.append({
            "label": spec["label"],
            "seed": spec["seed"],
            "policy": spec["policy"],
            "order": spec["order"],
            "wrapper_pass": result["all_pass"],
            "artifacts_unchanged": artifacts_unchanged,
            "timely_requests": broker["summary"]["timely_requests"],
            "timely_value_tokens": broker["summary"]["timely_value_tokens"],
            "radio_correct": correct,
            "expired_requests": broker["summary"]["states"].get("expired", 0),
            "per_home": broker["summary"]["per_home"],
            "duplicate_commit_count": broker["summary"]["duplicate_commit_count"],
            "outstanding_tokens": broker["summary"]["outstanding_tokens"],
            "result": str(path),
            "result_sha256": sha256(path),
        })
    comparisons = []
    for seed in sorted(by_seed):
        policy_values = by_seed[seed]
        global_vectors = policy_values["global"]
        static_vectors = policy_values["static_partition"]
        if len(global_vectors) != 2 or len(static_vectors) != 2:
            raise RuntimeError("each seed needs two arms per policy")
        global_mean = [sum(items) / 2 for items in zip(*global_vectors)]
        static_mean = [sum(items) / 2 for items in zip(*static_vectors)]
        delta = [left - right for left, right in zip(global_mean, static_mean)]
        global_total = sum(global_mean)
        static_total = sum(static_mean)
        effect_pct = 100.0 * (global_total - static_total) / static_total
        ci = bootstrap_ci(delta, 126000 + seed)
        global_correct = sum(correct_by_seed[seed]["global"]) / 2
        static_correct = sum(correct_by_seed[seed]["static_partition"]) / 2
        radio_noninferior = global_correct >= static_correct
        comparisons.append({
            "seed": seed,
            "global_mean_tokens": global_total,
            "static_mean_tokens": static_total,
            "delta_tokens": global_total - static_total,
            "effect_percent": effect_pct,
            "source_second_bootstrap95_delta_tokens": ci,
            "global_mean_radio_correct": global_correct,
            "static_mean_radio_correct": static_correct,
            "radio_noninferior": radio_noninferior,
            "outcome_pass": effect_pct >= 2.0 and ci[0] > 0 and radio_noninferior,
        })
    structural_gates = {
        "protocol_sources": all(source_audit.values()),
        "eight_arms": len(arms) == 8,
        "all_arm_gates": all(arm["wrapper_pass"] for arm in arms),
        "artifacts_unchanged": all(arm["artifacts_unchanged"] for arm in arms),
        "broker_drained_unique": all(
            arm["duplicate_commit_count"] == 0 and arm["outstanding_tokens"] == 0
            for arm in arms
        ),
        "radio_decision_parity": all(
            all(signature == values[0] for signature in values[1:])
            for values in signatures.values()
        ),
        "balanced_design": all(
            len(values["global"]) == len(values["static_partition"]) == 2
            for values in by_seed.values()
        ),
    }
    outcome_gate = all(item["outcome_pass"] for item in comparisons)
    output = {
        "schema": "softwall-confirm126-certificate-global-vs-static-v1",
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
        "source_audit": source_audit,
        "arms": arms,
        "comparisons": comparisons,
        "structural_gates": structural_gates,
        "structural_pass": all(structural_gates.values()),
        "outcome_gate": outcome_gate,
        "all_pass": all(structural_gates.values()) and outcome_gate,
        "decision": (
            "global-routing-throughput-superiority-supported"
            if outcome_gate else
            "global-routing-throughput-superiority-not-supported"
        ),
        "claim_limit": (
            "The structural broker/certificate result stands independently of "
            "the throughput outcome. A failed outcome gate forbids a global "
            "routing performance claim."
        ),
    }
    out = RES / "confirm126_certificate_global_vs_static_result.json"
    temporary = out.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2), encoding="utf-8")
    temporary.replace(out)
    print(json.dumps({"structural_gates": structural_gates,
                      "outcome_gate": outcome_gate,
                      "comparisons": comparisons}, indent=2))
    if not output["structural_pass"]:
        raise SystemExit("Confirm126 structural gate failed")


if __name__ == "__main__":
    main()
