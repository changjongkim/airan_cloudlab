#!/usr/bin/env python3
"""Aggregate SoftWall multi-GPU G0--G3 evidence into one strict result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
RESULT = ROOT / "results/softwall_multigpu"
RAW = RESULT / "raw"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": ordered[round((len(ordered) - 1) * 0.50)],
        "p99": ordered[round((len(ordered) - 1) * 0.99)],
        "max": ordered[-1],
    }


def host_source(container_path: str) -> Path:
    if container_path.startswith("/softwall/"):
        return ROOT / "scripts_for_node/softwall_same_gpu" / container_path.removeprefix("/softwall/")
    if container_path.startswith("/softwall_task1/"):
        return ROOT / "scripts_for_node/task1" / container_path.removeprefix("/softwall_task1/")
    raise ValueError(f"unknown container source path: {container_path}")


def verify_protocol_sources(protocol: dict) -> dict[str, bool]:
    return {
        key: host_source(key).is_file() and sha256(host_source(key)) == expected
        for key, expected in protocol["source_sha256"].items()
    }


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    topology_path = RESULT / "topology_peer_gate_job58815435.json"
    g1_path = RESULT / "g1a_formal_job58815435_result.json"
    g2_path = RESULT / "g2a_formal_job58815435_result.json"
    topology = load(topology_path)
    g1 = load(g1_path)
    g2 = load(g2_path)

    g2_protocol_path = RESULT / "g2a_formal_job58815435_protocol.json"
    g2_controller_path = RESULT / "g2a_formal_job58815435_controller.json"
    g2_worker_path = RESULT / "g2a_formal_job58815435_worker.json"
    g2_protocol = load(g2_protocol_path)

    arms = []
    for seed in ("s1", "s2"):
        prefix = f"g3_formal_{seed}_job58815435"
        arm_result_path = RESULT / f"{prefix}_result.json"
        protocol_path = RESULT / f"{prefix}_protocol.json"
        controller_path = RAW / f"{prefix}_controller.json"
        remote_path = RAW / f"{prefix}_remote_worker.json"
        arm = load(arm_result_path)
        protocol = load(protocol_path)
        controller = load(controller_path)
        remote = load(remote_path)
        remote_timed_count = sum(
            row.get("endpoint_id") == "nrx1" for row in controller["records"]
        )
        warmup_count = remote["completed_units"] - remote_timed_count
        timed_records = remote["records"][warmup_count:]
        source_checks = verify_protocol_sources(protocol)
        artifact_checks = {
            path_text: Path(path_text).is_file()
            and sha256(Path(path_text)) == expected
            for path_text, expected in arm["artifact_sha256"].items()
        }
        arms.append({
            "seed": seed,
            "result": str(arm_result_path),
            "result_sha256": sha256(arm_result_path),
            "protocol_source_checks": source_checks,
            "artifact_checks": artifact_checks,
            "all_arm_gates_pass": bool(arm["all_pass"]),
            "summary": arm["summary"],
            "remote_timed_requests": remote_timed_count,
            "remote_warmup_requests": warmup_count,
            "remote_timed_worker_path_ms": summarize([
                row["worker_path_ms"] for row in timed_records
            ]),
            "remote_timed_forward_gpu_us": summarize([
                row["forward_gpu_us"] for row in timed_records
            ]),
            "remote_timed_nrx_gpu_ms": summarize([
                row["nrx_gpu_ms"] for row in timed_records
            ]),
            "remote_timed_backward_gpu_us": summarize([
                row["backward_gpu_us"] for row in timed_records
            ]),
        })

    gates = {
        "g0_topology": topology["device_count"] == 4
        and bool(topology["all_bidirectional_peer_pairs"]),
        "g1_cross_process_transport": bool(g1["all_pass"])
        and g1["iterations"] == 10000,
        "g2_actual_nrx_path": bool(g2["all_pass"])
        and g2_protocol["iterations"] == 1000,
        "g2_sources_frozen": all(verify_protocol_sources(g2_protocol).values()),
        "g3_two_independent_arms": len(arms) == 2
        and all(arm["all_arm_gates_pass"] for arm in arms),
        "g3_sources_frozen": all(
            all(arm["protocol_source_checks"].values()) for arm in arms
        ),
        "g3_artifacts_unchanged": all(
            all(arm["artifact_checks"].values()) for arm in arms
        ),
        "g3_remote_timed_path_complete": all(
            arm["remote_timed_requests"] > 0
            and arm["remote_timed_worker_path_ms"]["count"]
            == arm["remote_timed_requests"]
            for arm in arms
        ),
    }

    artifacts = [
        topology_path,
        g1_path,
        g2_path,
        g2_protocol_path,
        g2_controller_path,
        g2_worker_path,
    ]
    result = {
        "schema": "softwall-confirm114-multigpu-v1",
        "status": "G0/G1a/G2a/G3 complete; G1 host-staging and G4+ remain",
        "slurm_job_id": "58815435",
        "g0": {
            "device_count": topology["device_count"],
            "all_bidirectional_peer_pairs": topology["all_bidirectional_peer_pairs"],
            "topology_artifact": str(topology_path),
        },
        "g1": {
            "iterations": g1["iterations"],
            "payload": g1["payload"],
            "steady_timing_us": g1["steady_timing_us"],
            "cold_first_unit_us": g1["cold_first_unit_us"],
            "all_pass": g1["all_pass"],
        },
        "g2": {
            "iterations": g2_protocol["iterations"],
            "deadline_ms": g2_protocol["deadline_ms"],
            "controller_response_ms": g2["controller_response_ms"],
            "forward_gpu_us": g2["forward_gpu_us"],
            "nrx_gpu_ms": g2["nrx_gpu_ms"],
            "backward_gpu_us": g2["backward_gpu_us"],
            "all_pass": g2["all_pass"],
            "source_checks": verify_protocol_sources(g2_protocol),
        },
        "g3_arms": arms,
        "gates": gates,
        "all_pass": all(gates.values()),
        "claim": (
            "On one 4xA100 NVLink node, a remote P2P NeuralRx endpoint was "
            "integrated with a local CUDA-IPC endpoint, four-cell SoftWall "
            "recovery, MPS Qwen, and mixed injected failures in two independent "
            "arms without observed deadline, bound, guard, fault, or credit "
            "violations. This is finite-sample transport-independence evidence, "
            "not a WCET, production-DU, or multi-GPU performance-superiority claim."
        ),
        "remaining": [
            "host-staging transport baseline",
            "component-wise co-run service-bound qualification",
            "more than two heterogeneous endpoints",
            "same-budget multi-GPU strong-baseline comparison",
            "prospective 1/2/4-GPU feasibility-envelope grid",
            "remote-GPU Qwen lease",
            "production DU d_MAC and independent hardware",
        ],
        "input_artifact_sha256": {
            str(path): sha256(path) for path in artifacts
        },
    }
    output = RESULT / "confirm114_multigpu_gates_job58815435.json"
    atomic_json(output, result)
    print(json.dumps(result, indent=2))
    if not result["all_pass"]:
        raise SystemExit("Confirm114 aggregate gate failed")


if __name__ == "__main__":
    main()

