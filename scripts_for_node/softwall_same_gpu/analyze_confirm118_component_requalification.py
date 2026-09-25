#!/usr/bin/env python3
"""Audit prospectively frozen revised component bounds on independent arms."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
RES = ROOT / "results/softwall_multigpu"
PROTOCOL = RES / "confirm118_component_requalification_protocol.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values: list[float]) -> dict[str, float | int]:
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


def gate(values: list[float], bound: float) -> dict:
    return {
        "bound": bound,
        "summary": summary(values),
        "violations": sum(value > bound for value in values),
        "pass": all(value <= bound for value in values),
    }


def main() -> None:
    protocol = read(PROTOCOL)
    bounds = protocol["candidate_bounds"]
    source_audit = {
        path: {
            "expected": expected,
            "observed": sha(ROOT / path),
            "match": expected == sha(ROOT / path),
        }
        for path, expected in protocol["source_sha256_before_run"].items()
    }
    arms = []
    source_map = {
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
        rows = [row for row in controller["records"] if row["endpoint_id"] is not None]
        admissions = Counter(row["endpoint_id"] for row in rows)
        component = {
            "front_gpu_ms": gate([row["nrx_prepare_gpu_ms"] for row in rows], bounds["front_gpu_ms"]),
            "back_gpu_ms": gate([row["nrx_back_gpu_ms"] for row in rows], bounds["back_gpu_ms"]),
            "end_to_end_nrx_ms": gate([row["nrx_response_ms"] for row in rows], bounds["end_to_end_nrx_ms"]),
        }
        remote = {}
        for endpoint, key in (("nrx1", "remote1_worker"), ("nrx2", "remote2_worker")):
            worker = read(Path(wrapper[key]))
            endpoint_index = int(endpoint[-1])
            cells = sum(cell % 3 == endpoint_index for cell in range(4))
            warm_records = (protocol["warmup"] + 1) * cells
            records = worker["records"][warm_records:]
            remote[endpoint] = {
                "expected_timed": admissions[endpoint],
                "observed_timed": len(records),
                "count_match": len(records) == admissions[endpoint],
                "forward_gpu_us": gate([row["forward_gpu_us"] for row in records], bounds["remote_forward_gpu_us"]),
                "nrx_gpu_ms": gate([row["nrx_gpu_ms"] for row in records], bounds["remote_nrx_gpu_ms"]),
                "backward_gpu_us": gate([row["backward_gpu_us"] for row in records], bounds["remote_backward_gpu_us"]),
                "worker_path_ms": gate([row["worker_path_ms"] for row in records], bounds["remote_worker_path_ms"]),
            }
        artifacts_unchanged = all(
            sha(Path(path)) == expected
            for path, expected in wrapper["artifact_sha256"].items()
        )
        component_pass = all(value["pass"] for value in component.values())
        remote_pass = all(
            value["count_match"]
            and all(
                value[key]["pass"]
                for key in ("forward_gpu_us", "nrx_gpu_ms", "backward_gpu_us", "worker_path_ms")
            )
            for value in remote.values()
        )
        arms.append({
            "label": label,
            "wrapper": str(wrapper_path),
            "wrapper_sha256": sha(wrapper_path),
            "all_arm_gates_pass": wrapper["all_pass"],
            "arm_protocol_source_checks": {
                key: expected == sha(ROOT / source_map[key])
                for key, expected in arm_protocol["source_sha256"].items()
            },
            "artifacts_unchanged": artifacts_unchanged,
            "endpoint_admissions": dict(admissions),
            "component": component,
            "remote": remote,
            "component_pass": component_pass and remote_pass,
            "summary": wrapper["summary"],
        })
    gates = {
        "protocol_sources": all(value["match"] for value in source_audit.values()),
        "two_independent_arms": len(arms) == 2,
        "arm_sources": all(all(arm["arm_protocol_source_checks"].values()) for arm in arms),
        "artifacts_unchanged": all(arm["artifacts_unchanged"] for arm in arms),
        "system_safety": all(arm["all_arm_gates_pass"] for arm in arms),
        "revised_component_bounds": all(arm["component_pass"] for arm in arms),
    }
    result = {
        "schema": "softwall-confirm118-component-requalification-v1",
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha(PROTOCOL),
        "source_audit": source_audit,
        "arms": arms,
        "gates": gates,
        "all_pass": all(gates.values()),
        "scope": "prospective finite-sample requalification after C117 rejected the 2 ms home-component bounds; not WCET or production timing",
    }
    output = RES / "confirm118_component_requalification_job58815435.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(output)
    print(json.dumps(gates, indent=2))
    if not result["all_pass"]:
        raise SystemExit("Confirm118 component requalification failed")


if __name__ == "__main__":
    main()
