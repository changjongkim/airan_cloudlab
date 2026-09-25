#!/usr/bin/env python3.11
"""Check C156 Nsight GPU activity against the committed V17 timeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

from build_confirm156_timeline_protocol import source_hashes
from integrated_shared_recovery_holdout_plan_v1 import build_branch


ACTIVITY_TABLES = (
    ("kernel", "CUPTI_ACTIVITY_KIND_KERNEL"),
    ("memcpy", "CUPTI_ACTIVITY_KIND_MEMCPY"),
    ("memset", "CUPTI_ACTIVITY_KIND_MEMSET"),
)
PHASES = ("forward_p2p", "install", "conventional", "backward_p2p")
CLOCK_TOLERANCE_NS = 250_000


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(path: Path) -> tuple[bool, list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    return (
        len(rows) == 4
        and all("A100" in row.get("name", "") for row in rows)
        and all(row.get("mig.mode.current", "").lower() == "disabled" for row in rows),
        rows,
    )


def profile_activity(
    path: Path, clock_anchor: dict[str, int] | None = None
) -> tuple[int, list[dict], list[dict]]:
    events = []
    nvtx = []
    with sqlite3.connect(path) as database:
        clock_columns = {
            row[1] for row in database.execute(
                "PRAGMA table_info(TARGET_INFO_SESSION_START_TIME)"
            )
        }
        if "systemClockNs" in clock_columns:
            base_row = database.execute(
                "SELECT systemClockNs FROM TARGET_INFO_SESSION_START_TIME"
            ).fetchone()
            clock_kind = "perf_counter"
        elif "utcEpochNs" in clock_columns:
            base_row = database.execute(
                "SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME"
            ).fetchone()
            clock_kind = "utc_epoch"
        else:
            base_row = None
            clock_kind = "unknown"
        if base_row is None:
            raise RuntimeError(f"profile has no session clock: {path}")
        base = int(base_row[0])

        def normalize(relative_ns: int) -> int:
            absolute_ns = base + int(relative_ns)
            if clock_kind == "perf_counter":
                return absolute_ns
            if clock_anchor is None:
                raise RuntimeError(
                    f"UTC-only Nsight profile requires a clock anchor: {path}"
                )
            return (
                int(clock_anchor["perf_counter_ns"])
                + absolute_ns - int(clock_anchor["utc_epoch_ns"])
            )
        tables = {
            row[0] for row in database.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        strings = {}
        if "StringIds" in tables:
            strings = dict(database.execute("SELECT id, value FROM StringIds"))
        for kind, table in ACTIVITY_TABLES:
            if table not in tables:
                continue
            columns = {
                row[1] for row in database.execute(f"PRAGMA table_info({table})")
            }
            selected = ["start", "end", "deviceId"]
            for optional in ("shortName", "bytes", "srcDeviceId", "dstDeviceId"):
                if optional in columns:
                    selected.append(optional)
            for row in database.execute(
                f"SELECT {','.join(selected)} FROM {table} ORDER BY start"
            ):
                value = dict(zip(selected, row))
                name_id = value.pop("shortName", None)
                value.update({
                    "kind": kind,
                    "start_ns": normalize(int(value.pop("start"))),
                    "end_ns": normalize(int(value.pop("end"))),
                    "name": strings.get(name_id, str(name_id)) if name_id else None,
                })
                events.append(value)
        if "NVTX_EVENTS" in tables:
            for start, end, text, text_id in database.execute(
                "SELECT start, end, text, textId FROM NVTX_EVENTS "
                "WHERE end IS NOT NULL ORDER BY start"
            ):
                name = text if text is not None else strings.get(text_id)
                if name and str(name).startswith("c156:"):
                    nvtx.append({
                        "name": str(name),
                        "start_ns": normalize(int(start)),
                        "end_ns": normalize(int(end)),
                    })
    events.sort(key=lambda row: (row["start_ns"], row["end_ns"], row["kind"]))
    return base, events, nvtx


def contained(events: list[dict], window: dict) -> list[dict]:
    return [
        row for row in events
        if row["start_ns"] >= window["start_ns"] - CLOCK_TOLERANCE_NS
        and row["end_ns"] <= window["end_ns"] + CLOCK_TOLERANCE_NS
    ]


def contained_exact(events: list[dict], window: dict) -> list[dict]:
    return [
        row for row in events
        if row["start_ns"] >= window["start_ns"]
        and row["end_ns"] <= window["end_ns"]
    ]


def intersecting(events: list[dict], lower: int, upper: int) -> list[dict]:
    return [
        row for row in events
        if (row["start_ns"] < upper + CLOCK_TOLERANCE_NS
            and row["end_ns"] > lower - CLOCK_TOLERANCE_NS)
    ]


def overlap_ns(left: list[dict], right: list[dict]) -> int:
    total = 0
    for first in left:
        for second in right:
            total += max(
                0,
                min(first["end_ns"], second["end_ns"])
                - max(first["start_ns"], second["start_ns"]),
            )
    return total


def phases_are_ordered(windows: dict[str, dict]) -> bool:
    return all(
        windows[left]["start_ns"] <= windows[left]["end_ns"]
        <= windows[right]["start_ns"] <= windows[right]["end_ns"]
        for left, right in zip(PHASES, PHASES[1:])
    )


def analyze_phases(recoveries: list[dict], events: list[dict], nvtx: list[dict]) -> dict:
    rows = []
    all_windows = []
    for recovery in recoveries:
        windows = recovery.get("phase_windows", {})
        expected_nvtx = {
            f"c156:{recovery['key'][0]}:{recovery['key'][1]}:{name}"
            for name in PHASES
        }
        observed_rows = {
            row["name"]: row for row in nvtx
            if row["name"] in expected_nvtx
        }
        observed_nvtx = set(observed_rows)
        nvtx_windows = {
            name: observed_rows.get(
                f"c156:{recovery['key'][0]}:{recovery['key'][1]}:{name}"
            )
            for name in PHASES
        }
        phase_events = {
            name: contained_exact(events, nvtx_windows[name])
            if nvtx_windows[name] is not None else []
            for name in PHASES
        }
        nvtx_host_aligned = all(
            nvtx_windows[name] is not None
            and nvtx_windows[name]["start_ns"]
                >= windows[name]["start_ns"] - CLOCK_TOLERANCE_NS
            and nvtx_windows[name]["end_ns"]
                <= windows[name]["end_ns"] + CLOCK_TOLERANCE_NS
            for name in PHASES
        )
        nonempty = {
            "forward_p2p": any(row["kind"] == "memcpy" for row in phase_events["forward_p2p"]),
            "install": bool(phase_events["install"]),
            "conventional": any(row["kind"] == "kernel" for row in phase_events["conventional"]),
            "backward_p2p": any(row["kind"] == "memcpy" for row in phase_events["backward_p2p"]),
        }
        gpu_order = True
        for left, right in zip(PHASES, PHASES[1:]):
            if phase_events[left] and phase_events[right]:
                gpu_order = gpu_order and (
                    max(row["end_ns"] for row in phase_events[left])
                    <= min(row["start_ns"] for row in phase_events[right])
                )
        rows.append({
            "key": recovery["key"],
            "host_phase_order": phases_are_ordered(windows),
            "gpu_phase_order": gpu_order,
            "phase_nonempty": nonempty,
            "phase_activity_counts": {
                name: {
                    kind: sum(row["kind"] == kind for row in phase_events[name])
                    for kind, _ in ACTIVITY_TABLES
                }
                for name in PHASES
            },
            "nvtx_expected": sorted(expected_nvtx),
            "nvtx_observed": sorted(observed_nvtx),
            "nvtx_complete": observed_nvtx == expected_nvtx,
            "nvtx_host_aligned": nvtx_host_aligned,
        })
        all_windows.extend(
            window for window in nvtx_windows.values() if window is not None
        )
    if recoveries:
        lower = min(row["actual_start_ns"] for row in recoveries)
        upper = max(row["actual_completed_ns"] for row in recoveries)
        runtime = intersecting(events, lower, upper)
    else:
        runtime = []
    uncovered = [
        row for row in runtime
        if not any(
            row["start_ns"] >= window["start_ns"]
            and row["end_ns"] <= window["end_ns"]
            for window in all_windows
        )
    ]
    return {"rows": rows, "runtime_events": runtime, "uncovered": uncovered}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--peer-spec", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--worker-sqlite", type=Path, required=True)
    parser.add_argument("--qwen-sqlite", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--task1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load(args.protocol)
    spec = load(args.peer_spec)
    worker = load(args.worker)
    qwen = load(args.qwen)
    owners = [load(Path(row["owner_output"])) for row in spec["peers"]]
    expected = build_branch("conditional_open")
    expected_keys = [list(key) for key in expected["physical_recovery_keys"]]
    inventory_ok, inventory_rows = inventory(args.inventory)
    clock_anchor = worker.get("clock_anchor")
    worker_base, worker_events, worker_nvtx = profile_activity(
        args.worker_sqlite, clock_anchor
    )
    qwen_base, qwen_events, _ = profile_activity(args.qwen_sqlite, clock_anchor)
    recoveries = worker.get("physical_recoveries", [])
    phase_audit = analyze_phases(recoveries, worker_events, worker_nvtx)

    qwen_record = qwen.get("records", [None])[0]
    release_ns = int(worker["release_wall_ns"])
    lease_interval = worker["lease_interval"]
    ai_lower = release_ns + int(lease_interval["start_ns"])
    ai_upper = release_ns + int(lease_interval["finish_ns"])
    qwen_runtime = intersecting(
        [row for row in qwen_events if row["kind"] == "kernel"],
        int(qwen_record["accepted_ns"]), int(qwen_record["completed_ns"]),
    ) if qwen_record else []
    recovery_kernels = [
        row for row in phase_audit["runtime_events"] if row["kind"] == "kernel"
    ]
    forbidden_overlap_ns = overlap_ns(qwen_runtime, recovery_kernels)

    decisions = worker.get("reservation_decisions", [])
    decision_keys = [row.get("key") for row in decisions]
    rejected = [row.get("key") for row in decisions if not row.get("accepted")]
    success = [row.get("key") for row in worker.get("success_outcomes", [])]
    owner_keys = [[f"home{row['home_id']}", row["request_id"]] for row in owners]
    physical_keys = [row.get("key") for row in recoveries]
    same_host = (
        all(row.get("host") == worker.get("host") for row in owners)
        and qwen.get("host") == worker.get("host")
    )
    per_recovery_ok = all(
        row["host_phase_order"] and row["gpu_phase_order"]
        and all(row["phase_nonempty"].values()) and row["nvtx_complete"]
        and row["nvtx_host_aligned"]
        for row in phase_audit["rows"]
    )
    recovery_gpu_order = all(
        recoveries[index]["actual_completed_ns"]
        <= recoveries[index + 1]["actual_start_ns"]
        for index in range(len(recoveries) - 1)
    )
    source_ok = source_hashes(args.scripts_root, args.task1_root) == protocol["source_sha256"]
    gates = {
        "source_hash_match": source_ok,
        "profile_files_and_session_clocks": (
            args.worker_sqlite.is_file() and args.qwen_sqlite.is_file()
            and worker_base > 0 and qwen_base > 0
        ),
        "node_inventory_and_provenance": (
            inventory_ok and same_host
            and worker.get("host") not in protocol["excluded_nodes"]
            and str(worker.get("slurm_job_id")) == str(qwen.get("slurm_job_id"))
        ),
        "controlled_certificate_decisions": (
            worker.get("branch") == "conditional_open"
            and decision_keys == [list(key) for key in expected["submitted_keys"]]
            and rejected == [list(key) for key in expected["rejected_keys"]]
            and success == [list(key) for key in expected["success_keys"]]
            and worker.get("lease_decision", {}).get("accepted") is True
        ),
        "qwen_fence_and_lease": (
            qwen.get("completed_units") == 1 and not qwen.get("rejected")
            and worker.get("qwen", {}).get("fence_confirmed") is True
            and worker.get("launch_control_bound_ms")
                == protocol["mode"]["launch_control_bound_ms"]
            and worker.get("qwen", {}).get("launch_control_bound_violation") is False
            and worker.get("qwen", {}).get("lease_interval_violation") is False
            and worker.get("lease_retire", {}).get("accepted") is True
            and bool(qwen_runtime)
            and all(row["start_ns"] >= ai_lower and row["end_ns"] <= ai_upper
                    for row in qwen_runtime)
        ),
        "recovery_certificate_and_gpu_order": (
            physical_keys == expected_keys
            and worker.get("certificate_order") == expected_keys
            and owner_keys == expected_keys
            and recovery_gpu_order
            and all(row.get("correct") for row in recoveries)
        ),
        "phase_gpu_causality_and_nvtx": per_recovery_ok,
        "worker_runtime_activity_fully_attributed": (
            bool(phase_audit["runtime_events"]) and not phase_audit["uncovered"]
        ),
        "qwen_recovery_kernel_overlap_zero": forbidden_overlap_ns == 0,
        "home_commit_correct_and_timely": (
            all(row.get("correct") and row.get("input_ready_before_release") for row in owners)
            and not any(row.get("deadline_miss") for row in owners)
            and all(row.get("termination_acknowledged") for row in owners)
        ),
        "p2p_ipc_and_process_lifecycle": (
            worker.get("ipc_handles_closed_before_ack") is True
            and worker.get("qwen_stop_acknowledged") is True
            and bool(worker.get("peer_access"))
            and all(bool(value) for value in worker.get("peer_access", {}).values())
            and worker.get("error") is None
            and all(row.get("error") is None for row in owners)
        ),
    }
    value = {
        "schema": "softwall-confirm156-gpu-timeline-result-v1",
        "status": (
            "GPU_TIMELINE_SEMANTICS_PASS_CONTROLLED_OUTCOME"
            if all(gates.values()) else "GPU_TIMELINE_GATE_FAIL"
        ),
        "protocol": str(args.protocol),
        "job_id": worker.get("slurm_job_id"),
        "node": worker.get("host"),
        "gates": gates,
        "all_pass": all(gates.values()),
        "summary": {
            "qwen_runtime_kernels": len(qwen_runtime),
            "worker_runtime_events": len(phase_audit["runtime_events"]),
            "recovery_runtime_kernels": len(recovery_kernels),
            "nvtx_ranges": len(worker_nvtx),
            "uncovered_worker_events": len(phase_audit["uncovered"]),
            "forbidden_qwen_recovery_kernel_overlap_ns": forbidden_overlap_ns,
            "physical_recoveries": len(recoveries),
            "correct_home_commits": sum(bool(row.get("correct")) for row in owners),
            "deadline_misses": sum(bool(row.get("deadline_miss")) for row in owners),
        },
        "lease_interval_ns": {"start_ns": ai_lower, "end_ns": ai_upper},
        "qwen_profile_session_start_ns": qwen_base,
        "worker_profile_session_start_ns": worker_base,
        "recovery_phase_audit": phase_audit["rows"],
        "inventory": inventory_rows,
        "artifact_sha256": {
            str(path): sha256(path)
            for path in (
                args.protocol, args.peer_spec, args.worker, args.qwen,
                args.worker_sqlite, args.qwen_sqlite, args.inventory,
            )
        } | {
            str(Path(row["owner_output"])): sha256(Path(row["owner_output"]))
            for row in spec["peers"]
        },
        "scope": protocol["scope"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps({
        "status": value["status"],
        "all_pass": value["all_pass"],
        "summary": value["summary"],
    }, indent=2, sort_keys=True))
    if not value["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
