#!/usr/bin/env python3
"""Audit two prospectively frozen two-home/eight-cell SoftWall arms."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
RES = ROOT / "results/softwall_multigpu"
PROTOCOL = RES / "confirm121_sharded_two_home_protocol.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def active_interval(controller: dict) -> tuple[int, int]:
    """Return the timed PHY interval, excluding model/receiver warmup."""
    starts = [row["feature_begin_ns"] for row in controller["records"]]
    ends = [row["commit_return_ns"] for row in controller["records"]]
    return min(starts), max(ends)


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
        "/softwall/four_cell_trace_baseline_controller.py":
            "scripts_for_node/softwall_same_gpu/four_cell_trace_baseline_controller.py",
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
        arm_protocol_path = RES / f"{label}_protocol.json"
        arm_protocol = read(arm_protocol_path)
        controllers = [
            read(Path(home["controller"]))
            for home in result["homes"]
        ]
        intervals = [active_interval(controller) for controller in controllers]
        overlap_ns = max(
            0,
            min(interval[1] for interval in intervals)
            - max(interval[0] for interval in intervals),
        )
        artifact_audit = {
            path: {
                "expected": expected,
                "observed": sha256(Path(path)),
                "match": expected == sha256(Path(path)),
            }
            for path, expected in result["artifact_sha256"].items()
        }
        homes = []
        for home, controller in zip(result["homes"], controllers):
            homes.append({
                "home": home["home"],
                "host": controller["host"],
                "first_release_ns": controller["first_release_ns"],
                "records": len(controller["records"]),
                "timed_interval_ns": list(active_interval(controller)),
                "summary": home["summary"],
                "safety": home["safety"],
                "workers": home["workers"],
            })
        arm_gates = {
            "wrapper_pass": result["all_pass"],
            "two_homes": len(homes) == 2,
            "same_host": len({home["host"] for home in homes}) == 1,
            "same_first_release": len({home["first_release_ns"] for home in homes}) == 1,
            "physical_time_overlap": overlap_ns > 0,
            "eight_cell_record_count": sum(home["records"] for home in homes)
                == 8 * protocol["iterations"],
            "all_safety": all(
                all(home["safety"].values()) for home in homes
            ),
            "worker_count_match": all(
                worker["count_match"]
                for home in homes for worker in home["workers"]
            ),
            "atomic_exchange_each_home": all(
                home["summary"]["atomic_exchange"] > 0 for home in homes
            ),
            "timely_ai_each_home": all(
                home["summary"]["ai_timely_value_tokens"] > 0 for home in homes
            ),
            "artifacts_unchanged": all(
                item["match"] for item in artifact_audit.values()
            ),
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
            "home_release_delta_ns": abs(
                homes[0]["first_release_ns"] - homes[1]["first_release_ns"]
            ),
            "physical_overlap_ms": overlap_ns / 1e6,
            "homes": homes,
            "artifact_audit": artifact_audit,
            "gates": arm_gates,
            "all_pass": all(arm_gates.values()),
        })
    seed_tuples = {
        (
            arm["payload_seed0"], arm["payload_seed1"],
            arm["channel_seed0"], arm["channel_seed1"],
        )
        for arm in protocol["arms"]
    }
    gates = {
        "protocol_sources": all(item["match"] for item in source_audit.values()),
        "two_independent_arms": len(arms) == 2 and len(seed_tuples) == 2,
        "both_arms_pass": all(arm["all_pass"] for arm in arms),
    }
    output = {
        "schema": "softwall-confirm121-sharded-two-home-v1",
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
        "source_audit": source_audit,
        "arms": arms,
        "gates": gates,
        "all_pass": all(gates.values()),
        "scope": (
            "two disjoint four-cell recovery homes on two physical A100 GPUs, "
            "synchronized releases, per-home local NRx/Qwen, MPS; finite sample"
        ),
        "claim_limit": (
            "This validates disjoint certificate composition and horizontal "
            "memory scaling; it does not validate shared endpoints, a global "
            "AI lease pool, WCET, or production DU timing."
        ),
    }
    out = RES / "confirm121_sharded_two_home_result.json"
    temporary = out.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2), encoding="utf-8")
    temporary.replace(out)
    print(json.dumps(gates, indent=2))
    if not output["all_pass"]:
        raise SystemExit("Confirm121 sharded two-home gate failed")


if __name__ == "__main__":
    main()
