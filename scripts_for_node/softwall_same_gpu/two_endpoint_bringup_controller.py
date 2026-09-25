#!/usr/bin/env python3
"""Two same-GPU MPS NeuralRx endpoints with mandatory-first radio credits.

This intentionally waits for both physical NRx completions before running any
conventional fallback. It qualifies the two-endpoint path; it is not a joint
admission/AI policy or a strong-baseline comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import cupy as cp

from dual_receiver_phy import PairedDualReceiver
from isca_v2.cuda_ipc_channel import CudaIpcOwner, TERMINATE_SEQ
from isca_v2.dart_runtime import (
    DartRequest, DartRuntime, EndpointState, FallbackCalendar,
    ProfileTable, ServiceProfile,
)
from s2_runner import BackgroundClient
from softwall_phy import summary, wait_until


RX_ELEMENTS = 1 * 3276 * 12 * 4
CE_ELEMENTS = 1 * 4914 * 1 * 4
FWD_ELEMENTS = 2 * RX_ELEMENTS + 2 * CE_ELEMENTS
LLR_ELEMENTS = 2 * 1 * 3276 * 12


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--ipc-dir", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--period-ms", type=float, default=150)
    parser.add_argument("--deadline-ms", type=float, default=130)
    parser.add_argument("--nrx-bound-ms", type=float, default=50)
    parser.add_argument("--conv-bound-ms", type=float, default=25)
    parser.add_argument("--commit-guard-ms", type=float, default=2)
    parser.add_argument("--endpoint-timeout-ms", type=float, default=100)
    parser.add_argument("--ai-budget-ms", type=float, default=40)
    parser.add_argument("--ai-guard-ms", type=float, default=2)
    parser.add_argument("--ai-rpc-timeout-ms", type=float, default=35)
    parser.add_argument("--seed", type=int, default=20321055)
    parser.add_argument("--snr-db", type=float, default=-8.5)
    parser.add_argument("--channel-seed-base", type=int, default=20355000)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0:
        parser.error("invalid count")
    if min(
        args.period_ms, args.deadline_ms, args.nrx_bound_ms,
        args.conv_bound_ms, args.commit_guard_ms, args.ai_budget_ms,
        args.ai_guard_ms, args.ai_rpc_timeout_ms,
    ) <= 0:
        parser.error("invalid timing parameter")
    if args.nrx_bound_ms + 2 * args.conv_bound_ms + args.commit_guard_ms > args.deadline_ms:
        parser.error("two full NRx bounds cannot leave both recovery windows")
    if 2 * args.conv_bound_ms > args.period_ms:
        parser.error("two all-fail conventional paths exceed sustained period")

    cp.cuda.runtime.setDevice(0)
    contexts = []
    for cell in range(2):
        forward = cp.empty(FWD_ELEMENTS, dtype=cp.float32)
        backward = cp.empty(LLR_ELEMENTS, dtype=cp.float32)
        owner = CudaIpcOwner(
            f"{args.tag_prefix}_c{cell}", forward, backward,
            directory=args.ipc_dir,
        )
        receiver = PairedDualReceiver(
            args.engine, seed=args.seed + cell, enable_local_neural=False
        )
        contexts.append({
            "owner": owner, "forward": forward, "backward": backward,
            "receiver": receiver, "sequence": 0,
        })
    profiles = ProfileTable({
        (f"nrx{cell}", 0, 0, "isolated"): ServiceProfile(
            forward_ns=0,
            service_ns=round(args.nrx_bound_ms * 1e6),
            backward_ns=0,
        )
        for cell in range(2)
    })
    runtime = DartRuntime(
        # Each IPC owner exports exactly one forward/backward buffer pair.
        # Advertising a second credit here would permit an unsafe second
        # request before the first worker has physically completed.
        endpoints=[EndpointState("nrx0", ring_depth=1), EndpointState("nrx1", ring_depth=1)],
        profiles=profiles,
        commit_guard_ns=round(args.commit_guard_ms * 1e6),
        fallback_calendar=FallbackCalendar(
            capacity=1,
            service_ns=round(args.conv_bound_ms * 1e6),
            commit_guard_ns=round(args.commit_guard_ms * 1e6),
        ),
    )
    background = None
    records = []
    warmup_correct = [0, 0]
    conventional_warmup_correct = [0, 0]
    faults = []
    try:
        for ctx in contexts:
            ctx["owner"].wait_ready(120.0)
        for _ in range(args.warmup):
            for cell, ctx in enumerate(contexts):
                conventional_warmup_correct[cell] += int(ctx["receiver"].run_conventional()[1])
                ctx["sequence"] += 1
                ctx["receiver"].prepare_neural_ipc(ctx["forward"])
                ctx["owner"].publish_forward(ctx["sequence"])
                ctx["owner"].wait_backward(ctx["sequence"], 2.0)
                warmup_correct[cell] += int(ctx["receiver"].complete_neural_ipc(ctx["backward"])[1])
        if warmup_correct != [args.warmup] * 2 or conventional_warmup_correct != [args.warmup] * 2:
            raise RuntimeError("clean receiver warmup failed")
        for cell, ctx in enumerate(contexts):
            ctx["receiver"].apply_rayleigh_awgn(args.snr_db, args.channel_seed_base - 2 + cell)
            ctx["sequence"] += 1
            ctx["receiver"].prepare_neural_ipc(ctx["forward"])
            ctx["owner"].publish_forward(ctx["sequence"])
            ctx["owner"].wait_backward(ctx["sequence"], 2.0)
            ctx["receiver"].complete_neural_ipc(ctx["backward"])
            ctx["receiver"].run_conventional()

        background = BackgroundClient(args.socket, args.ai_rpc_timeout_ms)
        period_ns = round(args.period_ms * 1e6)
        deadline_ns = round(args.deadline_ms * 1e6)
        conv_bound_ns = round(args.conv_bound_ms * 1e6)
        guard_ns = round(args.commit_guard_ms * 1e6)
        nrx_bound_ns = round(args.nrx_bound_ms * 1e6)
        ai_budget_ns = round(args.ai_budget_ms * 1e6)
        ai_guard_ns = round(args.ai_guard_ms * 1e6)
        first_release_ns = time.perf_counter_ns() + 200_000_000
        for index in range(args.iterations):
            release_ns = first_release_ns + index * period_ns
            for cell, ctx in enumerate(contexts):
                ctx["receiver"].apply_rayleigh_awgn(
                    args.snr_db, args.channel_seed_base + index * 2 + cell
                )
            wait_until(release_ns)
            transactions = []
            radio = []
            for cell in range(2):
                cutoff_ns = release_ns + deadline_ns - guard_ns - (2 - cell) * conv_bound_ns
                request = DartRequest(
                    slot_id=index * 2 + cell, epoch=1, graph_id=0, tensor_class=0,
                    release_ns=release_ns, deadline_ns=release_ns + deadline_ns,
                    fallback_latest_start_ns=cutoff_ns,
                    # Bring-up pins each radio PHY to its own IPC worker.
                    # Queue-aware cross-routing needs separate request-owned
                    # input buffers and is deliberately a later comparison.
                    candidate_bitmap=1 << cell,
                )
                tx = runtime.reserve_mandatory(request, time.perf_counter_ns())
                if tx is None:
                    raise RuntimeError(f"mandatory credit unavailable at release={index} cell={cell}")
                admitted = runtime.admit_nrx(tx, time.perf_counter_ns())
                transactions.append(tx)
                radio.append({
                    "index": index, "cell": cell, "slot_id": request.slot_id,
                    "release_ns": release_ns, "channel_seed": args.channel_seed_base + index * 2 + cell,
                    "cutoff_ns": cutoff_ns, "endpoint_id": (
                        tx.reservation.endpoint_id if admitted else None
                    ),
                    "admitted": admitted, "nrx_correct": None,
                    "nrx_response_ms": None, "nrx_bound_violation": False,
                    "nrx_commit": False, "fallback": False,
                    "fallback_early": False, "retime_rejected": False,
                    "fallback_actual_start_ns": None, "fallback_started": None,
                    "conventional_gpu_ms": None, "conv_bound_violation": False,
                    "commit_kind": None, "correct": False, "completed_ns": None,
                })
            for cell, tx in enumerate(transactions):
                if tx.reservation is None:
                    continue
                ctx = contexts[int(tx.reservation.endpoint_id[-1])]
                ctx["sequence"] += 1
                ctx["receiver"].prepare_neural_ipc(ctx["forward"])
                ctx["owner"].publish_forward(ctx["sequence"])
            for cell, tx in enumerate(transactions):
                if tx.reservation is None:
                    continue
                ctx = contexts[int(tx.reservation.endpoint_id[-1])]
                try:
                    ctx["owner"].wait_backward(
                        ctx["sequence"], args.endpoint_timeout_ms / 1000.0
                    )
                except TimeoutError as error:
                    faults.append({"index": index, "cell": cell, "kind": "endpoint_timeout", "message": str(error)})
                    raise RuntimeError("physical NRx completion unconfirmed; stop before conventional") from error
                neural = ctx["receiver"].complete_neural_ipc(ctx["backward"])
                finish_ns = time.perf_counter_ns()
                neural_correct = bool(neural[1])
                visible = neural_correct and finish_ns <= tx.request.fallback_latest_start_ns
                committed = runtime.complete_nrx(
                    tx, tx.request.slot_id, tx.request.epoch, finish_ns,
                    payload_visible=visible,
                )
                row = radio[cell]
                row["nrx_correct"] = neural_correct
                row["nrx_response_ms"] = (finish_ns - release_ns) / 1e6
                row["nrx_bound_violation"] = finish_ns - release_ns > nrx_bound_ns
                row["nrx_commit"] = committed
                if committed:
                    row["commit_kind"] = "nrx"
                    row["correct"] = neural_correct
                    row["completed_ns"] = finish_ns
            for cell, tx in enumerate(transactions):
                row = radio[cell]
                if row["commit_kind"] == "nrx":
                    continue
                row["fallback"] = True
                start_ns = time.perf_counter_ns()
                if start_ns < tx.request.fallback_latest_start_ns:
                    started = runtime.start_fallback_early(tx, start_ns)
                    row["fallback_early"] = started
                    if not started:
                        row["retime_rejected"] = True
                        wait_until(tx.request.fallback_latest_start_ns)
                        start_ns = time.perf_counter_ns()
                        started = runtime.start_fallback(tx, start_ns)
                else:
                    started = runtime.start_fallback(tx, start_ns)
                row["fallback_actual_start_ns"] = start_ns
                row["fallback_started"] = started
                if not started:
                    raise RuntimeError(f"mandatory fallback failed to start index={index} cell={cell}")
                conventional = contexts[cell]["receiver"].run_conventional()
                row["conventional_gpu_ms"] = conventional[0]
                row["conv_bound_violation"] = conventional[0] > args.conv_bound_ms
                finish_ns = time.perf_counter_ns()
                committed = runtime.complete_conventional(tx, finish_ns)
                row["commit_kind"] = "conventional" if committed else "none"
                row["correct"] = bool(conventional[1]) and committed
                row["completed_ns"] = finish_ns
            for row in radio:
                row["response_ms"] = (row["completed_ns"] - release_ns) / 1e6
                row["deadline_miss"] = row["completed_ns"] > release_ns + deadline_ns
                records.append(row)
            if index + 1 < args.iterations:
                next_release_ns = release_ns + period_ns
                while (
                    background.enabled
                    and time.perf_counter_ns() + ai_budget_ns + ai_guard_ns <= next_release_ns
                ):
                    background.run(index, next_release_ns, args.ai_budget_ms)
    finally:
        if background is not None:
            background.close()
        for ctx in contexts:
            try:
                ctx["owner"].publish_forward(TERMINATE_SEQ)
            except Exception:
                pass
        time.sleep(0.2)
        for ctx in contexts:
            ctx["owner"].close()

    background_records = background.records if background is not None else []
    result = {
        "schema": "softwall-two-endpoint-bringup-v1",
        "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "iterations": args.iterations, "cells": 2, "warmup": args.warmup,
        "warmup_correct": warmup_correct,
        "conventional_warmup_correct": conventional_warmup_correct,
        "period_ms": args.period_ms, "deadline_ms": args.deadline_ms,
        "nrx_bound_ms": args.nrx_bound_ms, "conv_bound_ms": args.conv_bound_ms,
        "commit_guard_ms": args.commit_guard_ms,
        "endpoint_timeout_ms": args.endpoint_timeout_ms,
        "ai_budget_ms": args.ai_budget_ms, "ai_guard_ms": args.ai_guard_ms,
        "ai_rpc_timeout_ms": args.ai_rpc_timeout_ms,
        "payload_seed": args.seed, "snr_db": args.snr_db,
        "channel_seed_base": args.channel_seed_base,
        "correct_cells": sum(row["correct"] for row in records),
        "deadline_misses": sum(row["deadline_miss"] for row in records),
        "nrx_bound_violations": sum(row["nrx_bound_violation"] for row in records),
        "conv_bound_violations": sum(row["conv_bound_violation"] for row in records),
        "admission_rejections": sum(not row["admitted"] for row in records),
        "retime_rejections": sum(row["retime_rejected"] for row in records),
        "early_fallbacks": sum(row["fallback_early"] for row in records),
        "fallbacks": sum(row["fallback"] for row in records),
        "nrx_commits": sum(row["commit_kind"] == "nrx" for row in records),
        "conv_commits": sum(row["commit_kind"] == "conventional" for row in records),
        "background_units": len(background_records),
        "background_budget_violations": sum(row["budget_violation"] for row in background_records),
        "background_release_crossings": sum(row["crossed_next_release"] for row in background_records),
        "background_faults": background.faults if background is not None else [],
        "endpoint_faults": faults,
        "runtime_metrics": dict(runtime.metrics),
        "fallback_calendar_final": runtime.fallback_calendar.snapshot(),
        "endpoint_final": {name: endpoint.snapshot() for name, endpoint in runtime.endpoints.items()},
        "response_ms": summary([row["response_ms"] for row in records]),
        "nrx_response_ms": summary([
            row["nrx_response_ms"] for row in records if row["nrx_response_ms"] is not None
        ]),
        "records": records,
        "background_records": background_records,
    }
    atomic_json(args.output, result)
    print(
        f"[TWO-ENDPOINT] correct={result['correct_cells']}/{len(records)} "
        f"miss={result['deadline_misses']} nrx_bound={result['nrx_bound_violations']} "
        f"fallback={result['fallbacks']} ai={result['background_units']} "
        f"p99={result['response_ms']['p99']:.3f}ms",
        flush=True,
    )


if __name__ == "__main__":
    main()
