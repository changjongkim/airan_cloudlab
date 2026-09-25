#!/usr/bin/env python3
"""S0 temporal protection: run AI units only in measured post-RAN slack."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from pathlib import Path

from softwall_phy import ConventionalRan, summary, wait_until


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cells", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--period-ms", type=float, default=10.0)
    parser.add_argument("--deadline-ms", type=float, default=5.0)
    parser.add_argument("--ai-budget-ms", type=float, required=True)
    parser.add_argument("--guard-ms", type=float, default=0.5)
    parser.add_argument("--ai-rpc-timeout-ms", type=float)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--ran-input", choices=("synthetic", "paired"), default="synthetic")
    args = parser.parse_args()
    if args.ai_budget_ms <= 0 or args.guard_ms < 0:
        parser.error("AI budget must be positive and guard non-negative")
    if args.ai_rpc_timeout_ms is not None and args.ai_rpc_timeout_ms <= 0:
        parser.error("AI RPC timeout must be positive")

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(args.socket)
    rpc_timeout_ms = (
        args.ai_rpc_timeout_ms
        if args.ai_rpc_timeout_ms is not None
        else args.ai_budget_ms + args.guard_ms
    )
    client.settimeout(rpc_timeout_ms / 1000.0)
    channel = client.makefile("rwb", buffering=0)
    if args.ran_input == "paired":
        from paired_phy import PairedConventionalRan

        ran = PairedConventionalRan(seed=args.seed)
    else:
        ran = ConventionalRan(seed=args.seed)
    for _ in range(args.warmup):
        warmup_result = ran.run_cells(args.cells)
        if args.ran_input == "paired" and not warmup_result[1]:
            raise RuntimeError("paired-radio warmup produced an invalid transport block")

    ran_records = []
    ai_records = []
    ai_faults = []
    ai_enabled = True
    first_release_ns = time.perf_counter_ns() + 50_000_000
    period_ns = round(args.period_ms * 1e6)
    budget_ns = round(args.ai_budget_ms * 1e6)
    guard_ns = round(args.guard_ms * 1e6)
    for index in range(args.iterations):
        release_ns = first_release_ns + index * period_ns
        wait_until(release_ns)
        start_ns = time.perf_counter_ns()
        ran_result = ran.run_cells(args.cells)
        if args.ran_input == "paired":
            gpu_ms, correct_tb, crc_failures, payload_mismatches = ran_result
        else:
            gpu_ms = ran_result
            correct_tb = None
            crc_failures = 0
            payload_mismatches = 0
        completed_ns = time.perf_counter_ns()
        response_ms = (completed_ns - release_ns) / 1e6
        ran_records.append({
            "index": index,
            "release_ns": release_ns,
            "start_lateness_ms": (start_ns - release_ns) / 1e6,
            "gpu_ms": gpu_ms,
            "service_ms": (completed_ns - start_ns) / 1e6,
            "response_ms": response_ms,
            "deadline_miss": response_ms > args.deadline_ms,
            "correct_tb": correct_tb,
            "crc_failures": crc_failures,
            "payload_mismatches": payload_mismatches,
        })
        if index + 1 == args.iterations:
            continue
        next_release_ns = release_ns + period_ns
        while (
            ai_enabled
            and time.perf_counter_ns() + budget_ns + guard_ns <= next_release_ns
        ):
            admitted_ns = time.perf_counter_ns()
            try:
                channel.write(b'{"op":"run"}\n')
                payload = channel.readline()
                if not payload:
                    raise ConnectionError("AI worker closed the RPC channel")
                response = json.loads(payload)
                if not response.get("ok"):
                    raise RuntimeError(f"AI worker rejected unit: {response}")
            except (BrokenPipeError, ConnectionError, OSError, ValueError, RuntimeError) as error:
                ai_faults.append({
                    "slot_index": index,
                    "detected_ns": time.perf_counter_ns(),
                    "type": type(error).__name__,
                    "message": str(error),
                })
                ai_enabled = False
                break
            returned_ns = time.perf_counter_ns()
            execution_ms = (returned_ns - admitted_ns) / 1e6
            record = {
                "slot_index": index,
                "admitted_ns": admitted_ns,
                "returned_ns": returned_ns,
                "gpu_ms": response["gpu_ms"],
                "execution_ms": execution_ms,
                "budget_ms": args.ai_budget_ms,
                "budget_violation": execution_ms > args.ai_budget_ms,
                "crossed_next_release": returned_ns > next_release_ns,
            }
            ai_records.append(record)
            if record["budget_violation"] or record["crossed_next_release"]:
                ai_faults.append({
                    "slot_index": index,
                    "detected_ns": returned_ns,
                    "type": "AIContractViolation",
                    "message": (
                        f"execution_ms={execution_ms:.6f}, "
                        f"crossed_next_release={record['crossed_next_release']}"
                    ),
                })
                ai_enabled = False
                break

    if ai_enabled:
        try:
            channel.write(b'{"op":"stop"}\n')
            channel.readline()
        except (BrokenPipeError, OSError):
            pass
    try:
        channel.close()
    except OSError:
        pass
    client.close()
    result = {
        "schema": "softwall-protected-s0-v2",
        "label": args.label,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "mps_active_thread_percentage": os.environ.get(
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
        ),
        "mps_client_priority": os.environ.get("CUDA_MPS_CLIENT_PRIORITY"),
        "cells": args.cells,
        "iterations": args.iterations,
        "period_ms": args.period_ms,
        "deadline_ms": args.deadline_ms,
        "ai_budget_ms": args.ai_budget_ms,
        "guard_ms": args.guard_ms,
        "ai_rpc_timeout_ms": rpc_timeout_ms,
        "ran_path": "CE+NI+EQ+de-rate-match+LDPC+CRC",
        "input": (
            "Aerial PdschTx-generated valid 273-PRB PUSCH slot"
            if args.ran_input == "paired"
            else "fixed-seed synthetic complex RX tensor"
        ),
        "ran_response_ms": summary([x["response_ms"] for x in ran_records]),
        "ran_service_ms": summary([x["service_ms"] for x in ran_records]),
        "ran_deadline_misses": sum(x["deadline_miss"] for x in ran_records),
        "ran_correct_transport_blocks": sum(x["correct_tb"] is not False for x in ran_records),
        "ran_crc_failures": sum(x["crc_failures"] for x in ran_records),
        "ran_payload_mismatches": sum(x["payload_mismatches"] for x in ran_records),
        "ai_completed_units": len(ai_records),
        "ai_gpu_ms": summary([x["gpu_ms"] for x in ai_records]),
        "ai_execution_ms": summary([x["execution_ms"] for x in ai_records]),
        "ai_budget_violations": sum(x["budget_violation"] for x in ai_records),
        "ai_crossed_next_release": sum(x["crossed_next_release"] for x in ai_records),
        "ai_disabled": not ai_enabled,
        "ai_fault_count": len(ai_faults),
        "ai_faults": ai_faults,
        "ran_records": ran_records,
        "ai_records": ai_records,
    }
    atomic_json(Path(args.output), result)
    print(
        f"[PROTECTED-S0] {args.label} ran_p99={result['ran_response_ms']['p99']:.3f}ms "
        f"ran_max={result['ran_response_ms']['max']:.3f}ms "
        f"miss={result['ran_deadline_misses']} ai_units={len(ai_records)} "
        f"ai_budget_violations={result['ai_budget_violations']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
