#!/usr/bin/env python3
"""Four-cell / two-NeuralRx-endpoint bring-up with atomic AI lease.

This qualifies one execution primitive. The fixed low-feature NeuralRx gate
is not a PHY-value/AI joint policy and this is not its performance comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import threading
import time
from pathlib import Path

import cupy as cp

from dual_receiver_phy import PairedDualReceiver
from isca_v2.cuda_ipc_channel import CudaIpcOwner, TERMINATE_SEQ
from isca_v2.dart_runtime import (
    BackgroundUnit, DartRequest, DartRuntime, EndpointState, FallbackCalendar,
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


def finish_pre_ai(thread, background, prior_count, cutoff_ns, guard_ns):
    """Join one in-flight AI RPC before any unqualified conventional overlap."""
    if thread is None:
        return
    thread.join(timeout=1.0)
    if thread.is_alive() or not background.enabled or len(background.records) != prior_count + 1:
        raise RuntimeError("asynchronous pre-radio AI completion unconfirmed")
    record = background.records[-1]
    record["phase"] = "before_nrx_observation"
    record["earliest_fallback_start_ns"] = cutoff_ns
    record["fallback_guard_violation"] = record["returned_ns"] + guard_ns > cutoff_ns
    if record["fallback_guard_violation"]:
        raise RuntimeError("asynchronous AI crossed earliest recovery guard")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--ipc-dir", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--cells", type=int, choices=(4,), default=4)
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
    parser.add_argument("--gate-mode", choices=("always", "low_threshold"), required=True)
    parser.add_argument("--early-mandatory", choices=("off", "on"), required=True)
    parser.add_argument("--ai-during-nrx", choices=("sync_one", "async_one"), required=True)
    parser.add_argument("--ai-aware-recovery", choices=("joint",), required=True)
    parser.add_argument("--ai-deadline-ms", type=float, required=True)
    parser.add_argument("--inject-correlated-failure-every", type=int, default=0)
    parser.add_argument("--gate-threshold", type=float, required=True)
    parser.add_argument("--alternate-admission-order", action="store_true")
    parser.add_argument("--clean-channel", action="store_true")
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0 or args.inject_correlated_failure_every < 0:
        parser.error("invalid count")
    if args.ai_deadline_ms <= 0 or args.ai_deadline_ms > args.period_ms:
        parser.error("AI deadline must be within one release period")
    if min(
        args.period_ms, args.deadline_ms, args.nrx_bound_ms,
        args.conv_bound_ms, args.commit_guard_ms, args.ai_budget_ms,
        args.ai_guard_ms, args.ai_rpc_timeout_ms,
    ) <= 0:
        parser.error("invalid timing parameter")
    # This only rejects deadlines that fail even under an optimistic
    # B_pair == per-endpoint bound assumption. Passing it does not qualify
    # the simultaneous pair bound on MPS; that needs separate evidence.
    if args.nrx_bound_ms + args.cells * args.conv_bound_ms + args.commit_guard_ms > args.deadline_ms:
        parser.error("even perfectly overlapped NRx bounds leave insufficient recovery time")
    if args.cells * args.conv_bound_ms > args.period_ms:
        parser.error("all-fail conventional paths exceed sustained period")

    cp.cuda.runtime.setDevice(0)
    contexts = []
    for cell in range(args.cells):
        forward = cp.empty(FWD_ELEMENTS, dtype=cp.float32) if cell < 2 else None
        backward = cp.empty(LLR_ELEMENTS, dtype=cp.float32) if cell < 2 else None
        owner = (
            CudaIpcOwner(
                f"{args.tag_prefix}_c{cell}", forward, backward,
                directory=args.ipc_dir,
            ) if cell < 2 else None
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
        endpoints=[EndpointState("nrx0", ring_depth=1), EndpointState("nrx1", ring_depth=1),
                   EndpointState("ai", ring_depth=1)],
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
    recovery_decisions = []
    warmup_correct = [0] * args.cells
    conventional_warmup_correct = [0] * args.cells
    faults = []
    try:
        for ctx in contexts[:2]:
            ctx["owner"].wait_ready(120.0)
        for _ in range(args.warmup):
            for cell, ctx in enumerate(contexts):
                conventional_warmup_correct[cell] += int(ctx["receiver"].run_conventional()[1])
                endpoint_ctx = contexts[cell % 2]
                endpoint_ctx["sequence"] += 1
                ctx["receiver"].prepare_neural_ipc(endpoint_ctx["forward"])
                endpoint_ctx["owner"].publish_forward(endpoint_ctx["sequence"])
                endpoint_ctx["owner"].wait_backward(endpoint_ctx["sequence"], 2.0)
                warmup_correct[cell] += int(
                    ctx["receiver"].complete_neural_ipc(endpoint_ctx["backward"])[1]
                )
        if (warmup_correct != [args.warmup] * args.cells
                or conventional_warmup_correct != [args.warmup] * args.cells):
            raise RuntimeError("clean receiver warmup failed")
        for ctx in contexts:
            ctx["receiver"].restore_clean_slot()
            ctx["receiver"].observed_channel_features()
        for cell, ctx in enumerate(contexts):
            if args.clean_channel:
                ctx["receiver"].restore_clean_slot()
            else:
                ctx["receiver"].apply_rayleigh_awgn(
                    args.snr_db, args.channel_seed_base - 2 + cell,
                    noise_reference="pre_fading",
                )
            endpoint_ctx = contexts[cell % 2]
            endpoint_ctx["sequence"] += 1
            ctx["receiver"].prepare_neural_ipc(endpoint_ctx["forward"])
            endpoint_ctx["owner"].publish_forward(endpoint_ctx["sequence"])
            endpoint_ctx["owner"].wait_backward(endpoint_ctx["sequence"], 2.0)
            ctx["receiver"].complete_neural_ipc(endpoint_ctx["backward"])
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
                if args.clean_channel:
                    ctx["receiver"].restore_clean_slot()
                else:
                    ctx["receiver"].apply_rayleigh_awgn(
                        args.snr_db, args.channel_seed_base + index * args.cells + cell,
                        noise_reference="pre_fading",
                    )
            wait_until(release_ns)
            features = []
            for ctx in contexts:
                feature_begin_ns = time.perf_counter_ns()
                feature = ctx["receiver"].observed_channel_features()
                feature_return_ns = time.perf_counter_ns()
                features.append((feature, feature_begin_ns, feature_return_ns))
            transactions = []
            radio = []
            for cell in range(args.cells):
                cutoff_ns = (
                    release_ns + deadline_ns - guard_ns
                    - (args.cells - cell) * conv_bound_ns
                )
                request = DartRequest(
                    slot_id=index * args.cells + cell, epoch=1, graph_id=0, tensor_class=0,
                    release_ns=release_ns, deadline_ns=release_ns + deadline_ns,
                    fallback_latest_start_ns=cutoff_ns,
                    candidate_bitmap=0b11,
                )
                tx = runtime.reserve_mandatory(request, time.perf_counter_ns())
                if tx is None:
                    raise RuntimeError(f"mandatory credit unavailable at release={index} cell={cell}")
                feature, feature_begin_ns, feature_return_ns = features[cell]
                gate_skipped = (
                    args.gate_mode == "low_threshold"
                    and feature["channel_estimate_power"] < args.gate_threshold
                )
                transactions.append(tx)
                radio.append({
                    "index": index, "cell": cell, "slot_id": request.slot_id,
                    "release_ns": release_ns,
                    "channel_seed": args.channel_seed_base + index * args.cells + cell,
                    "cutoff_ns": cutoff_ns, "endpoint_id": None,
                    "admitted": False, "nrx_correct": None,
                    "forced_nrx_failure": False,
                    "gate_skipped": gate_skipped,
                    "channel_estimate_power": feature["channel_estimate_power"],
                    "feature_gpu_ms": feature["gpu_ms"],
                    "feature_host_ms": (feature_return_ns - feature_begin_ns) / 1e6,
                    "feature_begin_ns": feature_begin_ns,
                    "feature_return_ns": feature_return_ns,
                    "nrx_response_ms": None, "nrx_bound_violation": False,
                    "nrx_commit": False, "fallback": False,
                    "fallback_early": False, "retime_rejected": False,
                    "nrx_dispatch_begin_ns": None,
                    "nrx_dispatched_ns": None, "nrx_observed_ns": None,
                    "fallback_actual_start_ns": None, "fallback_started": None,
                    "conventional_gpu_ms": None, "conv_bound_violation": False,
                    "conventional_host_path_ms": None, "conv_path_bound_violation": False,
                    "host_overlap_with_other_nrx": False,
                    "commit_kind": None, "correct": False, "completed_ns": None,
                    "commit_return_ns": None,
                })
            admission_order = (
                list(reversed(range(args.cells)))
                if args.alternate_admission_order and index % 2
                else list(range(args.cells))
            )
            for cell in admission_order:
                row = radio[cell]
                if row["gate_skipped"]:
                    continue
                tx = transactions[cell]
                admitted = runtime.admit_nrx(tx, time.perf_counter_ns(), policy="shortest_queue")
                row["admitted"] = admitted
                row["endpoint_id"] = tx.reservation.endpoint_id if admitted else None
            for cell, tx in enumerate(transactions):
                if tx.reservation is None:
                    continue
                endpoint_ctx = contexts[int(tx.reservation.endpoint_id[-1])]
                endpoint_ctx["sequence"] += 1
                contexts[cell]["receiver"].prepare_neural_ipc(endpoint_ctx["forward"])
                radio[cell]["nrx_dispatch_begin_ns"] = time.perf_counter_ns()
                endpoint_ctx["owner"].publish_forward(endpoint_ctx["sequence"])
                radio[cell]["nrx_dispatched_ns"] = time.perf_counter_ns()
            async_ai_thread = None
            async_ai_prior_count = None
            async_ai_cutoff_ns = None
            if any(tx.reservation is not None for tx in transactions):
                earliest_cutoff_ns = min(
                    tx.request.fallback_latest_start_ns for tx in transactions
                )
                if time.perf_counter_ns() + ai_budget_ns + ai_guard_ns <= earliest_cutoff_ns:
                    if args.ai_during_nrx == "sync_one":
                        prior_count = len(background.records)
                        background.run(index, release_ns + period_ns, args.ai_budget_ms)
                        if not background.enabled or len(background.records) != prior_count + 1:
                            raise RuntimeError("pre-radio AI physical completion unconfirmed")
                        ai_record = background.records[-1]
                        ai_record["phase"] = "before_nrx_observation"
                        ai_record["earliest_fallback_start_ns"] = earliest_cutoff_ns
                        ai_record["fallback_guard_violation"] = (
                            ai_record["returned_ns"] + ai_guard_ns > earliest_cutoff_ns
                        )
                        if ai_record["fallback_guard_violation"]:
                            raise RuntimeError("pre-radio AI crossed earliest recovery guard")
                    else:
                        async_ai_prior_count = len(background.records)
                        async_ai_cutoff_ns = earliest_cutoff_ns
                        async_ai_thread = threading.Thread(
                            target=background.run,
                            args=(index, release_ns + period_ns, args.ai_budget_ms),
                            daemon=False,
                        )
                        async_ai_thread.start()
            if args.early_mandatory == "on":
                if any(tx.reservation is None for tx in transactions):
                    finish_pre_ai(async_ai_thread, background, async_ai_prior_count,
                                  async_ai_cutoff_ns, ai_guard_ns)
                    async_ai_thread = None
                for cell, tx in enumerate(transactions):
                    if tx.reservation is not None:
                        continue
                    row = radio[cell]
                    row["fallback"] = True
                    start_ns = time.perf_counter_ns()
                    started = runtime.start_fallback_early(tx, start_ns)
                    row["fallback_actual_start_ns"] = start_ns
                    row["fallback_started"] = started
                    row["fallback_early"] = started
                    if not started:
                        raise RuntimeError(f"early mandatory calendar conflict index={index} cell={cell}")
                    conventional = contexts[cell]["receiver"].run_conventional()
                    row["conventional_gpu_ms"] = conventional[0]
                    row["conv_bound_violation"] = conventional[0] > args.conv_bound_ms
                    finish_ns = time.perf_counter_ns()
                    committed = runtime.complete_conventional(tx, finish_ns)
                    commit_return_ns = time.perf_counter_ns()
                    if not committed:
                        raise RuntimeError(f"early mandatory commit rejected index={index} cell={cell}")
                    row["conventional_host_path_ms"] = (commit_return_ns - start_ns) / 1e6
                    row["conv_path_bound_violation"] = row["conventional_host_path_ms"] > args.conv_bound_ms
                    row["commit_kind"] = "conventional"
                    row["correct"] = bool(conventional[1])
                    row["completed_ns"] = finish_ns
                    row["commit_return_ns"] = commit_return_ns
            for cell, tx in enumerate(transactions):
                if tx.reservation is None:
                    continue
                endpoint_ctx = contexts[int(tx.reservation.endpoint_id[-1])]
                try:
                    endpoint_ctx["owner"].wait_backward(
                        endpoint_ctx["sequence"], args.endpoint_timeout_ms / 1000.0
                    )
                except TimeoutError as error:
                    faults.append({"index": index, "cell": cell, "kind": "endpoint_timeout", "message": str(error)})
                    raise RuntimeError("physical NRx completion unconfirmed; stop before conventional") from error
                neural = contexts[cell]["receiver"].complete_neural_ipc(endpoint_ctx["backward"])
                finish_ns = time.perf_counter_ns()
                radio[cell]["nrx_observed_ns"] = finish_ns
                neural_correct = bool(neural[1])
                forced_failure = (
                    args.inject_correlated_failure_every > 0
                    and index % args.inject_correlated_failure_every == 0
                )
                visible = (
                    neural_correct and not forced_failure
                    and finish_ns <= tx.request.fallback_latest_start_ns
                )
                committed = runtime.complete_nrx(
                    tx, tx.request.slot_id, tx.request.epoch, finish_ns,
                    payload_visible=visible,
                )
                commit_return_ns = time.perf_counter_ns()
                row = radio[cell]
                row["nrx_correct"] = neural_correct
                row["forced_nrx_failure"] = forced_failure
                row["nrx_response_ms"] = (finish_ns - release_ns) / 1e6
                row["nrx_bound_violation"] = finish_ns - release_ns > nrx_bound_ns
                row["nrx_commit"] = committed
                if committed:
                    row["commit_kind"] = "nrx"
                    row["correct"] = neural_correct
                    row["completed_ns"] = finish_ns
                    row["commit_return_ns"] = commit_return_ns
            finish_pre_ai(async_ai_thread, background, async_ai_prior_count,
                          async_ai_cutoff_ns, ai_guard_ns)
            async_ai_thread = None
            pending = [tx for cell, tx in enumerate(transactions)
                       if radio[cell]["commit_kind"] is None]
            recovery_action = {
                "release_index": index,
                "pending_cells": [cell for cell, tx in enumerate(transactions)
                                  if tx in pending],
                "retimed": False,
                "ai_lease": False,
                "ai_completed_before_recovery": False,
            }
            recovery_action["joint_lease_retired"] = False
            if pending:
                moves = []
                if len(pending) == 1:
                    tx = pending[0]
                    latest_start_ns = tx.request.deadline_ns - guard_ns - conv_bound_ns
                    if latest_start_ns > tx.fallback_reservation.start_ns:
                        moves = [(tx, latest_start_ns, tx.fallback_reservation.lane)]
                planned_start_ns = {
                    id(tx): start_ns for tx, start_ns, _ in moves
                }
                earliest_candidate_ns = min(
                    planned_start_ns.get(id(tx), tx.fallback_reservation.start_ns)
                    for tx in pending
                )
                ai_deadline_ns = release_ns + round(args.ai_deadline_ms * 1e6)
                now_ns = time.perf_counter_ns()
                if (background.enabled and
                    now_ns + ai_budget_ns + ai_guard_ns
                        <= min(earliest_candidate_ns, ai_deadline_ns)):
                    joint = runtime.replan_and_lease(
                        moves, "ai", now_ns,
                        ai_deadline_ns - now_ns, ai_guard_ns,
                        [BackgroundUnit(f"before_recovery_{index}", ai_budget_ns, 1.0)],
                    )
                    if joint is not None:
                        replacements, lease = joint
                        recovery_action["retimed"] = bool(replacements)
                        recovery_action["ai_lease"] = True
                        earliest_recovery_ns = min(
                            tx.fallback_reservation.start_ns for tx in pending
                        )
                        prior_count = len(background.records)
                        background.run(index, ai_deadline_ns, args.ai_budget_ms)
                        if not background.enabled or len(background.records) != prior_count + 1:
                            raise RuntimeError("joint AI physical completion unconfirmed; lease retained")
                        ai_record = background.records[-1]
                        ai_record["phase"] = "after_nrx_before_recovery"
                        ai_record["earliest_fallback_start_ns"] = earliest_recovery_ns
                        ai_record["joint_lease_unit_id"] = lease.unit_id
                        ai_record["fallback_guard_violation"] = (
                            ai_record["returned_ns"] + ai_guard_ns > earliest_recovery_ns
                        )
                        if ai_record["fallback_guard_violation"]:
                            raise RuntimeError("joint AI crossed recovery guard; lease retained")
                        if not runtime.retire_joint_lease(lease, gpu_fence_confirmed=True):
                            raise RuntimeError("confirmed joint AI lease could not be retired")
                        recovery_action["joint_lease_retired"] = True
                        recovery_action["ai_completed_before_recovery"] = True
            recovery_decisions.append(recovery_action)
            for cell, tx in enumerate(transactions):
                row = radio[cell]
                if row["commit_kind"] in ("nrx", "conventional"):
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
                # Legacy field name: host decision time before the mandatory
                # GPU call, not an observed GPU kernel-start timestamp.
                row["fallback_actual_start_ns"] = start_ns
                row["fallback_started"] = started
                if not started:
                    raise RuntimeError(f"mandatory fallback failed to start index={index} cell={cell}")
                conventional = contexts[cell]["receiver"].run_conventional()
                row["conventional_gpu_ms"] = conventional[0]
                row["conv_bound_violation"] = conventional[0] > args.conv_bound_ms
                finish_ns = time.perf_counter_ns()
                committed = runtime.complete_conventional(tx, finish_ns)
                commit_return_ns = time.perf_counter_ns()
                row["conventional_host_path_ms"] = (commit_return_ns - start_ns) / 1e6
                row["conv_path_bound_violation"] = (
                    row["conventional_host_path_ms"] > args.conv_bound_ms
                )
                row["commit_kind"] = "conventional" if committed else "none"
                row["correct"] = bool(conventional[1]) and committed
                row["completed_ns"] = finish_ns
                row["commit_return_ns"] = commit_return_ns
            for row in radio:
                if row["commit_kind"] == "conventional" and args.early_mandatory == "on":
                    row["host_overlap_with_other_nrx"] = any(
                        other["cell"] != row["cell"]
                        and other["nrx_dispatched_ns"] is not None
                        and other["nrx_dispatched_ns"] < row["commit_return_ns"]
                        and row["fallback_actual_start_ns"] < other["nrx_observed_ns"]
                        for other in radio
                    )
                # Keep legacy pre-commit response_ms for historical schema
                # comparison, and expose the conservative after-return time.
                row["response_ms"] = (row["completed_ns"] - release_ns) / 1e6
                row["commit_response_ms"] = (row["commit_return_ns"] - release_ns) / 1e6
                # An after-return observation is conservative for the actual
                # in-process commit instant, which the old trace did not log.
                row["deadline_miss"] = row["commit_return_ns"] > release_ns + deadline_ns
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
        for ctx in contexts[:2]:
            try:
                ctx["owner"].publish_forward(TERMINATE_SEQ)
            except Exception:
                pass
        time.sleep(0.2)
        for ctx in contexts[:2]:
            ctx["owner"].close()

    background_records = background.records if background is not None else []
    result = {
        "schema": "softwall-four-cell-two-endpoint-v1",
        "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "iterations": args.iterations, "cells": args.cells, "warmup": args.warmup,
        "warmup_correct": warmup_correct,
        "conventional_warmup_correct": conventional_warmup_correct,
        "period_ms": args.period_ms, "deadline_ms": args.deadline_ms,
        "nrx_bound_ms": args.nrx_bound_ms, "conv_bound_ms": args.conv_bound_ms,
        "commit_guard_ms": args.commit_guard_ms,
        "endpoint_timeout_ms": args.endpoint_timeout_ms,
        "ai_budget_ms": args.ai_budget_ms, "ai_guard_ms": args.ai_guard_ms,
        "ai_rpc_timeout_ms": args.ai_rpc_timeout_ms,
        "payload_seed": args.seed, "snr_db": args.snr_db,
        "gate_mode": args.gate_mode, "gate_threshold": args.gate_threshold,
        "early_mandatory": args.early_mandatory,
        "ai_during_nrx": args.ai_during_nrx,
        "ai_aware_recovery": args.ai_aware_recovery,
        "inject_correlated_failure_every": args.inject_correlated_failure_every,
        "ai_deadline_ms": args.ai_deadline_ms,
        "recovery_decisions": recovery_decisions,
        "recovery_retime_count": sum(row["retimed"] for row in recovery_decisions),
        "joint_lease_retired_count": sum(
            row["joint_lease_retired"] for row in recovery_decisions
        ),
        "ai_before_recovery_units": sum(
            row.get("phase") == "after_nrx_before_recovery" for row in background_records
        ),
        "ai_timely_units": sum(
            row["returned_ns"] <= records[args.cells * row["release_index"]]["release_ns"]
            + round(args.ai_deadline_ms * 1e6)
            for row in background_records
        ),
        "pre_radio_ai_units": sum(
            row.get("phase") == "before_nrx_observation" for row in background_records
        ),
        "pre_radio_ai_guard_violations": sum(
            row.get("fallback_guard_violation", False) for row in background_records
        ),
        "host_overlapped_early_mandatory": sum(
            row["host_overlap_with_other_nrx"] for row in records
        ),
        "routing_policy": "shortest_queue",
        "alternate_admission_order": args.alternate_admission_order,
        "channel_mode": "clean" if args.clean_channel else "pre_fading_rayleigh_awgn",
        "noise_reference": "pre_fading",
        "channel_seed_base": args.channel_seed_base,
        "gate_skips": sum(row["gate_skipped"] for row in records),
        "feature_gpu_ms": summary([row["feature_gpu_ms"] for row in records]),
        "feature_host_ms": summary([row["feature_host_ms"] for row in records]),
        "correct_cells": sum(row["correct"] for row in records),
        "deadline_misses": sum(row["deadline_miss"] for row in records),
        "nrx_bound_violations": sum(row["nrx_bound_violation"] for row in records),
        "conv_bound_violations": sum(row["conv_bound_violation"] for row in records),
        "conv_path_bound_violations": sum(row["conv_path_bound_violation"] for row in records),
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
        "commit_response_ms": summary([row["commit_response_ms"] for row in records]),
        "nrx_response_ms": summary([
            row["nrx_response_ms"] for row in records if row["nrx_response_ms"] is not None
        ]),
        "records": records,
        "background_records": background_records,
    }
    atomic_json(args.output, result)
    print(
        f"[FOUR-CELL/TWO-ENDPOINT] correct={result['correct_cells']}/{len(records)} "
        f"miss={result['deadline_misses']} nrx_bound={result['nrx_bound_violations']} "
        f"fallback={result['fallbacks']} ai={result['background_units']} "
        f"p99={result['response_ms']['p99']:.3f}ms",
        flush=True,
    )


if __name__ == "__main__":
    main()
