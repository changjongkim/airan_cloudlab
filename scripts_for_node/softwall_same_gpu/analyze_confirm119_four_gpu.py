#!/usr/bin/env python3
"""Audit two frozen four-GPU/four-endpoint SoftWall arms."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
RES = ROOT / "results/softwall_multigpu"
PROTOCOL = RES / "confirm119_four_gpu_protocol.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "p50": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": len(ordered),
        "p50": ordered[round((len(ordered) - 1) * 0.50)],
        "p99": ordered[round((len(ordered) - 1) * 0.99)],
        "max": ordered[-1],
    }


def bound_gate(values: list[float], bound: float) -> dict:
    return {
        "bound": bound,
        "summary": summarize(values),
        "violations": sum(value > bound for value in values),
        "pass": bool(values) and all(value <= bound for value in values),
    }


def main() -> None:
    protocol = read(PROTOCOL)
    bounds = protocol["candidate_component_bounds"]
    source_audit = {
        path: {
            "expected": expected,
            "observed": sha(ROOT / path),
            "match": expected == sha(ROOT / path),
        }
        for path, expected in protocol["source_sha256_before_run"].items()
    }
    arm_results = []
    arm_source_map = {
        "/softwall/four_cell_trace_baseline_controller.py": "scripts_for_node/softwall_same_gpu/four_cell_trace_baseline_controller.py",
        "/softwall/same_request_ipc_worker.py": "scripts_for_node/softwall_same_gpu/same_request_ipc_worker.py",
        "/softwall/multigpu_p2p_nrx_worker.py": "scripts_for_node/softwall_same_gpu/multigpu_p2p_nrx_worker.py",
        "/softwall/trace_qwen_worker.py": "scripts_for_node/softwall_same_gpu/trace_qwen_worker.py",
        "/softwall/multigpu_p2p_ipc_gate.py": "scripts_for_node/softwall_same_gpu/multigpu_p2p_ipc_gate.py",
        "/softwall_task1/isca_v2/cuda_ipc_channel.py": "scripts_for_node/task1/isca_v2/cuda_ipc_channel.py",
    }
    for spec in protocol["arms"]:
        label = spec["label"]
        wrapper_path = RES / f"{label}_result.json"
        wrapper = read(wrapper_path)
        controller = read(Path(wrapper["controller"]))
        arm_protocol = read(RES / f"{label}_protocol.json")
        admitted_rows = [row for row in controller["records"] if row["endpoint_id"] is not None]
        admissions = Counter(row["endpoint_id"] for row in admitted_rows)
        home_component = {
            "front_gpu_ms": bound_gate([row["nrx_prepare_gpu_ms"] for row in admitted_rows], bounds["front_gpu_ms"]),
            "back_gpu_ms": bound_gate([row["nrx_back_gpu_ms"] for row in admitted_rows], bounds["back_gpu_ms"]),
            "end_to_end_nrx_ms": bound_gate([row["nrx_response_ms"] for row in admitted_rows], bounds["end_to_end_nrx_ms"]),
        }
        remote = {}
        for endpoint in ("nrx1", "nrx2", "nrx3"):
            worker = read(Path(wrapper[f"remote{endpoint[-1]}_worker"]))
            warm_records = protocol["warmup"] + 1
            records = worker["records"][warm_records:]
            remote[endpoint] = {
                "expected_timed": admissions[endpoint],
                "observed_timed": len(records),
                "count_match": len(records) == admissions[endpoint],
                "forward_gpu_us": bound_gate([row["forward_gpu_us"] for row in records], bounds["remote_forward_gpu_us"]),
                "nrx_gpu_ms": bound_gate([row["nrx_gpu_ms"] for row in records], bounds["remote_nrx_gpu_ms"]),
                "backward_gpu_us": bound_gate([row["backward_gpu_us"] for row in records], bounds["remote_backward_gpu_us"]),
                "worker_path_ms": bound_gate([row["worker_path_ms"] for row in records], bounds["remote_worker_path_ms"]),
            }
        local_worker = read(Path(wrapper["local_worker"]))
        local_expected_total = admissions["nrx0"] + protocol["warmup"] + 1
        local_count_match = local_worker["completed_units"] == local_expected_total
        artifacts_unchanged = all(
            sha(Path(path)) == expected
            for path, expected in wrapper["artifact_sha256"].items()
        )
        remote_pass = all(
            value["count_match"]
            and all(
                value[key]["pass"]
                for key in ("forward_gpu_us", "nrx_gpu_ms", "backward_gpu_us", "worker_path_ms")
            )
            for value in remote.values()
        )
        arm_results.append({
            "label": label,
            "wrapper": str(wrapper_path),
            "wrapper_sha256": sha(wrapper_path),
            "all_arm_gates_pass": wrapper["all_pass"],
            "all_four_endpoints_used": set(admissions) == {"nrx0", "nrx1", "nrx2", "nrx3"},
            "endpoint_admissions": dict(admissions),
            "local_worker_count_match": local_count_match,
            "remote": remote,
            "home_component": home_component,
            "component_pass": all(value["pass"] for value in home_component.values()) and remote_pass,
            "arm_protocol_source_checks": {
                key: expected == sha(ROOT / arm_source_map[key])
                for key, expected in arm_protocol["source_sha256"].items()
            },
            "artifacts_unchanged": artifacts_unchanged,
            "summary": wrapper["summary"],
        })
    gates = {
        "protocol_sources": all(value["match"] for value in source_audit.values()),
        "two_independent_arms": len(arm_results) == 2,
        "arm_sources": all(all(arm["arm_protocol_source_checks"].values()) for arm in arm_results),
        "artifacts_unchanged": all(arm["artifacts_unchanged"] for arm in arm_results),
        "system_safety": all(arm["all_arm_gates_pass"] for arm in arm_results),
        "all_four_endpoints_used": all(arm["all_four_endpoints_used"] for arm in arm_results),
        "worker_count_match": all(
            arm["local_worker_count_match"]
            and all(value["count_match"] for value in arm["remote"].values())
            for arm in arm_results
        ),
        "component_bounds": all(arm["component_pass"] for arm in arm_results),
    }
    result = {
        "schema": "softwall-confirm119-four-gpu-v1",
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha(PROTOCOL),
        "source_audit": source_audit,
        "arms": arm_results,
        "gates": gates,
        "all_pass": all(gates.values()),
        "scope": "four physical GPUs, four NRx endpoints, warm lazy-module-loading MPS/Qwen/fault mode; finite sample, not WCET",
    }
    output = RES / "confirm119_four_gpu_job58815435.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(output)
    print(json.dumps(gates, indent=2))
    if not result["all_pass"]:
        raise SystemExit("Confirm119 four-GPU gate failed")


if __name__ == "__main__":
    main()
