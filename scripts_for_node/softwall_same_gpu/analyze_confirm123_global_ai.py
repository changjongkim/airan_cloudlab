#!/usr/bin/env python3
"""Audit two frozen global-AI broker arms over two recovery homes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
RES = ROOT / "results/softwall_multigpu"
PROTOCOL = RES / "confirm123_global_ai_broker_protocol.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def active_interval(controller: dict) -> tuple[int, int]:
    return (
        min(row["feature_begin_ns"] for row in controller["records"]),
        max(row["commit_return_ns"] for row in controller["records"]),
    )


def event_sequences(broker: dict) -> dict[int, list[str]]:
    result = defaultdict(list)
    for event in broker["events"]:
        result[event["request_id"]].append(event["event"])
    return dict(result)


def legal_event_sequence(sequence: list[str]) -> bool:
    """Accept zero or more aborted holds followed by at most one completion."""
    index = 0
    while sequence[index:index + 2] == ["prepare", "abort"]:
        index += 2
    tail = sequence[index:]
    return tail in ([], ["prepare", "commit", "complete"])


def main() -> None:
    protocol = read(PROTOCOL)
    source_audit = {
        relative: {
            "expected": expected,
            "observed": sha256(ROOT / relative),
            "match": expected == sha256(ROOT / relative),
        }
        for relative, expected in protocol["source_sha256_before_run"].items()
    }
    arm_source_map = {
        "/softwall/global_trace_baseline_controller.py":
            "scripts_for_node/softwall_same_gpu/global_trace_baseline_controller.py",
        "/softwall/global_trace_lease_broker.py":
            "scripts_for_node/softwall_same_gpu/global_trace_lease_broker.py",
        "/softwall/same_request_ipc_worker.py":
            "scripts_for_node/softwall_same_gpu/same_request_ipc_worker.py",
        "/softwall/trace_qwen_worker.py":
            "scripts_for_node/softwall_same_gpu/trace_qwen_worker.py",
        "/softwall_task1/isca_v2/dart_runtime.py":
            "scripts_for_node/task1/isca_v2/dart_runtime.py",
        "/softwall_task1/isca_v2/cuda_ipc_channel.py":
            "scripts_for_node/task1/isca_v2/cuda_ipc_channel.py",
    }
    arms = []
    for spec in protocol["arms"]:
        label = spec["label"]
        result_path = RES / f"{label}_result.json"
        result = read(result_path)
        broker = read(Path(result["broker"]))
        controllers = [read(Path(home["controller"])) for home in result["homes"]]
        intervals = [active_interval(controller) for controller in controllers]
        overlap_ns = max(0, min(end for _, end in intervals)
                         - max(start for start, _ in intervals))
        ids_by_home = [
            [row["request_id"] for row in controller["background_records"]]
            for controller in controllers
        ]
        all_ids = [request_id for values in ids_by_home for request_id in values]
        sequences = event_sequences(broker)
        completed_sequences_valid = all(
            legal_event_sequence(sequence) for sequence in sequences.values()
        )
        completed_sequence_count = sum(
            sequence[-3:] == ["prepare", "commit", "complete"]
            for sequence in sequences.values()
        )
        artifact_audit = {
            path: sha256(Path(path)) == expected
            for path, expected in result["artifact_sha256"].items()
        }
        arm_protocol = read(RES / f"{label}_protocol.json")
        summary = broker["summary"]
        home_value = [home["summary"]["ai_timely_value_tokens"]
                      for home in result["homes"]]
        arm_gates = {
            "wrapper_pass": result["all_pass"],
            "same_release": len({controller["first_release_ns"]
                                 for controller in controllers}) == 1,
            "physical_overlap": overlap_ns > 0,
            "system_safety": all(
                all(home["safety"].values()) for home in result["homes"]
            ),
            "unique_execution": len(all_ids) == len(set(all_ids))
                == summary["timely_requests"],
            "legal_event_sequences": completed_sequences_valid
                and completed_sequence_count == summary["timely_requests"],
            "broker_drained": summary["outstanding_tokens"] == 0
                and summary["duplicate_commit_count"] == 0,
            "both_homes_served": set(summary["per_home"]) == {"0", "1"}
                and all(value > 0 for value in home_value),
            "value_conservation": sum(home_value) == summary["timely_value_tokens"],
            "artifacts_unchanged": all(artifact_audit.values()),
            "arm_sources_match_frozen": all(
                arm_protocol["source_sha256"][key]
                == protocol["source_sha256_before_run"][relative]
                for key, relative in arm_source_map.items()
            ),
        }
        arms.append({
            "label": label,
            "result": str(result_path),
            "result_sha256": sha256(result_path),
            "release_delta_ns": abs(
                controllers[0]["first_release_ns"]
                - controllers[1]["first_release_ns"]
            ),
            "physical_overlap_ms": overlap_ns / 1e6,
            "request_counts_by_home": [len(values) for values in ids_by_home],
            "value_by_home": home_value,
            "broker_summary": summary,
            "event_counts": dict(Counter(
                event["event"] for event in broker["events"]
            )),
            "artifact_audit": artifact_audit,
            "gates": arm_gates,
            "all_pass": all(arm_gates.values()),
        })
    seed_tuples = {
        (arm["payload_seed0"], arm["payload_seed1"],
         arm["channel_seed0"], arm["channel_seed1"])
        for arm in protocol["arms"]
    }
    gates = {
        "protocol_sources": all(item["match"] for item in source_audit.values()),
        "two_independent_arms": len(arms) == 2 and len(seed_tuples) == 2,
        "both_arms_pass": all(arm["all_pass"] for arm in arms),
    }
    output = {
        "schema": "softwall-confirm123-global-ai-broker-v1",
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
        "source_audit": source_audit,
        "arms": arms,
        "gates": gates,
        "all_pass": all(gates.values()),
        "scope": (
            "one globally serialized BurstGPT request queue routed to two "
            "disjoint four-cell recovery homes and two physical A100 GPUs"
        ),
        "claim_limit": (
            "This validates global request uniqueness and local certificate/lease "
            "composition. It is not a throughput-superiority, WCET, shared-NRx, "
            "or production-DU result."
        ),
    }
    out = RES / "confirm123_global_ai_broker_result.json"
    temporary = out.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2), encoding="utf-8")
    temporary.replace(out)
    print(json.dumps(gates, indent=2))
    if not output["all_pass"]:
        raise SystemExit("Confirm123 global AI broker gate failed")


if __name__ == "__main__":
    main()
