#!/usr/bin/env python3
"""Same-GPU CUDA-IPC NeuralRx with an actual DART recovery transaction."""

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
from isca_v2.dart_runtime import DartRequest
from s2_runner import BackgroundClient, make_runtime
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
    parser.add_argument("--tag", required=True)
    parser.add_argument("--ipc-dir", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--period-ms", type=float, default=60.0)
    parser.add_argument("--deadline-ms", type=float, default=35.0)
    parser.add_argument("--endpoint-timeout-ms", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20321001)
    parser.add_argument("--snr-db", type=float)
    parser.add_argument("--channel-seed-base", type=int, default=20322000)
    parser.add_argument(
        "--policy", choices=("s2", "eager", "conventional"), default="s2"
    )
    parser.add_argument(
        "--conventional-warmup", choices=("on", "off"), default="on"
    )
    parser.add_argument("--socket")
    parser.add_argument("--ai-budget-ms", type=float, default=6.0)
    parser.add_argument("--ai-guard-ms", type=float, default=1.0)
    parser.add_argument("--ai-rpc-timeout-ms", type=float, default=7.0)
    parser.add_argument("--nrx-bound-ms", type=float, default=30.0)
    parser.add_argument("--conv-bound-ms", type=float, default=25.0)
    parser.add_argument("--commit-guard-ms", type=float, default=2.0)
    parser.add_argument("--recovery-gap-ai", action="store_true")
    parser.add_argument(
        "--fallback-timing", choices=("latest", "early_if_clear"), default="latest"
    )
    args = parser.parse_args()
    if min(
        args.iterations,
        args.period_ms,
        args.deadline_ms,
        args.endpoint_timeout_ms,
    ) <= 0 or args.warmup < 0:
        parser.error("invalid timing/count parameter")
    if min(args.ai_budget_ms, args.ai_guard_ms, args.ai_rpc_timeout_ms) <= 0:
        parser.error("AI lease timing parameters must be positive")
    if min(args.nrx_bound_ms, args.conv_bound_ms, args.commit_guard_ms) <= 0:
        parser.error("transaction timing parameters must be positive")
    if args.nrx_bound_ms + args.conv_bound_ms + args.commit_guard_ms > args.deadline_ms:
        parser.error("NRx bound must leave a reserved conventional recovery interval")
    if args.policy != "s2":
        parser.error("transactional controller currently supports only the S2 policy")

    cp.cuda.runtime.setDevice(0)
    forward = cp.empty(FWD_ELEMENTS, dtype=cp.float32)
    backward = cp.empty(LLR_ELEMENTS, dtype=cp.float32)
    owner = CudaIpcOwner(
        args.tag, forward, backward, directory=args.ipc_dir
    )
    receiver = PairedDualReceiver(
        args.engine, seed=args.seed, enable_local_neural=False
    )
    sequence = 0
    endpoint_enabled = True
    records = []
    endpoint_faults = []
    warmup_correct = 0
    conventional_warmup_correct = 0
    noisy_warmup_units = 0
    noisy_conventional_warmup_units = 0
    background = None
    runtime = make_runtime(
        round(args.nrx_bound_ms * 1e6),
        round(args.conv_bound_ms * 1e6),
        round(args.commit_guard_ms * 1e6),
        1,
    )
    try:
        owner.wait_ready(120.0)
        for _ in range(args.warmup):
            if args.conventional_warmup == "on":
                conventional_warmup_correct += int(
                    receiver.run_conventional()[1]
                )
            sequence += 1
            receiver.prepare_neural_ipc(forward)
            owner.publish_forward(sequence)
            owner.wait_backward(sequence, 2.0)
            result = receiver.complete_neural_ipc(backward)
            warmup_correct += int(result[1])
        if warmup_correct != args.warmup:
            raise RuntimeError(
                f"same-request endpoint warmup failed "
                f"{warmup_correct}/{args.warmup}"
            )
        if (args.conventional_warmup == "on"
                and conventional_warmup_correct != args.warmup):
            raise RuntimeError(
                f"same-request conventional warmup failed "
                f"{conventional_warmup_correct}/{args.warmup}"
            )
        if args.snr_db is not None:
            # Compile channel/random kernels and exercise the IPC path before
            # the timed epoch. A transition-SNR request may validly fail CRC.
            receiver.apply_rayleigh_awgn(
                args.snr_db, args.channel_seed_base - 1
            )
            sequence += 1
            receiver.prepare_neural_ipc(forward)
            owner.publish_forward(sequence)
            owner.wait_backward(sequence, 2.0)
            receiver.complete_neural_ipc(backward)
            noisy_warmup_units = 1
            if args.conventional_warmup == "on":
                receiver.run_conventional()
                noisy_conventional_warmup_units = 1

        period_ns = round(args.period_ms * 1e6)
        deadline_ns = round(args.deadline_ms * 1e6)
        budget_ns = round(args.ai_budget_ms * 1e6)
        ai_guard_ns = round(args.ai_guard_ms * 1e6)
        nrx_bound_ns = round(args.nrx_bound_ms * 1e6)
        conv_bound_ns = round(args.conv_bound_ms * 1e6)
        commit_guard_ns = round(args.commit_guard_ms * 1e6)
        background = (
            BackgroundClient(args.socket, args.ai_rpc_timeout_ms)
            if args.socket else None
        )
        first_release_ns = time.perf_counter_ns() + 200_000_000
        for index in range(args.iterations):
            release_ns = first_release_ns + index * period_ns
            if args.snr_db is not None:
                receiver.apply_rayleigh_awgn(
                    args.snr_db, args.channel_seed_base + index
                )
            wait_until(release_ns)
            started_ns = time.perf_counter_ns()
            expiry_ns = release_ns + deadline_ns
            fallback_start_ns = expiry_ns - commit_guard_ns - conv_bound_ns
            # A late request never consumes optional endpoint capacity. The
            # mandatory conventional path remains the fail-closed default.
            can_admit = (
                endpoint_enabled
                and started_ns + nrx_bound_ns <= fallback_start_ns
            )
            request = DartRequest(
                slot_id=index,
                epoch=1,
                graph_id=0,
                tensor_class=0,
                release_ns=release_ns,
                deadline_ns=expiry_ns,
                fallback_latest_start_ns=fallback_start_ns,
                candidate_bitmap=0xFFFF if can_admit else 0,
            )
            transaction = runtime.submit(request, started_ns)
            admitted = transaction.reservation is not None
            front_ms = None
            post_ms = None
            neural_correct = None
            conventional_gpu_ms = None
            conventional_correct = None
            fallback = False
            fallback_actual_start_ns = None
            fallback_started = None
            fallback_early = False
            early_retime_rejected = False
            endpoint_timeout = False
            nrx_bound_violation = False
            conv_bound_violation = False
            duplicate_commit = False
            gap_ai_units = 0
            gap_ai_deadline_ns = None
            commit_kind = None

            if admitted:
                sequence += 1
                front_ms = receiver.prepare_neural_ipc(forward)
                owner.publish_forward(sequence)
                try:
                    owner.wait_backward(
                        sequence, args.endpoint_timeout_ms / 1000.0
                    )
                except TimeoutError as error:
                    endpoint_timeout = True
                    endpoint_enabled = False
                    endpoint_faults.append({
                        "release_index": index,
                        "sequence": sequence,
                        "type": type(error).__name__,
                        "message": str(error),
                    })
                else:
                    neural = receiver.complete_neural_ipc(backward)
                    post_ms = neural[0]
                    neural_correct = neural[1]
                nrx_finish_ns = time.perf_counter_ns()
                nrx_bound_violation = (
                    nrx_finish_ns - started_ns > nrx_bound_ns
                )
                if not endpoint_timeout:
                    committed = runtime.complete_nrx(
                        transaction, index, 1, nrx_finish_ns,
                        payload_visible=(
                            neural_correct is True
                            and nrx_finish_ns <= fallback_start_ns
                        ),
                    )
                    if committed:
                        commit_kind = "nrx"

            if commit_kind != "nrx":
                fallback = admitted
                if admitted:
                    # A known CRC failure can be recovered before the reserved
                    # latest-start window, but only when the full conventional
                    # bound and guard fit before the next release. The current
                    # controller handles one cell/one transaction at a time,
                    # so no other fallback reservation can conflict here.
                    next_release_ns = release_ns + period_ns
                    fallback_early = (
                        args.fallback_timing == "early_if_clear"
                        and neural_correct is False
                        and not endpoint_timeout
                        and (index + 1 == args.iterations or
                             time.perf_counter_ns() + conv_bound_ns + commit_guard_ns
                             <= next_release_ns)
                    )
                    # Late recovery retains the prior gap-lease behavior. An
                    # early recovery relinquishes that gap to protect the next
                    # request's radio opportunity.
                    if not fallback_early and args.recovery_gap_ai and background is not None:
                        gap_ai_deadline_ns = min(
                            fallback_start_ns, next_release_ns
                        )
                        before = len(background.records)
                        while (
                            background.enabled
                            and time.perf_counter_ns() + budget_ns + ai_guard_ns
                            <= gap_ai_deadline_ns
                        ):
                            background.run(
                                index, gap_ai_deadline_ns, args.ai_budget_ms
                            )
                        gap_ai_units = len(background.records) - before
                    if not fallback_early:
                        wait_until(fallback_start_ns)
                    fallback_actual_start_ns = time.perf_counter_ns()
                    if fallback_early:
                        fallback_started = runtime.start_fallback_early(
                            transaction, fallback_actual_start_ns
                        )
                        if not fallback_started:
                            # Keep the old reserved window when an earlier
                            # physical recovery lane cannot be obtained.
                            early_retime_rejected = True
                            fallback_early = False
                            wait_until(fallback_start_ns)
                            fallback_actual_start_ns = time.perf_counter_ns()
                            fallback_started = runtime.start_fallback(
                                transaction, fallback_actual_start_ns
                            )
                    else:
                        fallback_started = runtime.start_fallback(
                            transaction, fallback_actual_start_ns
                        )
                conventional = receiver.run_conventional()
                conventional_gpu_ms = conventional[0]
                conventional_correct = conventional[1]
                conv_bound_violation = conventional_gpu_ms * 1e6 > conv_bound_ns
                committed = runtime.complete_conventional(
                    transaction, time.perf_counter_ns()
                )
                if committed:
                    commit_kind = "conventional"
                if admitted and not endpoint_timeout:
                    duplicate_commit = runtime.complete_nrx(
                        transaction, index, 1, time.perf_counter_ns(),
                        payload_visible=True,
                    )
                correct = bool(conventional_correct) and committed
            else:
                correct = True

            completed_ns = time.perf_counter_ns()
            response_ms = (completed_ns - release_ns) / 1e6
            records.append({
                "index": index,
                "sequence": sequence if front_ms is not None else None,
                "release_ns": release_ns,
                "channel_seed": (
                    args.channel_seed_base + index
                    if args.snr_db is not None else None
                ),
                "start_lateness_ms": (started_ns - release_ns) / 1e6,
                "front_gpu_ms": front_ms,
                "post_gpu_ms": post_ms,
                "neural_correct": neural_correct,
                "endpoint_timeout": endpoint_timeout,
                "admitted": admitted,
                "admission_rejected": not admitted,
                "fallback_start_ns": fallback_start_ns if fallback else None,
                "fallback_actual_start_ns": fallback_actual_start_ns,
                "fallback_start_lateness_ms": (
                    (fallback_actual_start_ns - fallback_start_ns) / 1e6
                    if fallback_actual_start_ns is not None else None
                ),
                "fallback_started": fallback_started,
                "fallback_early": fallback_early,
                "early_retime_rejected": early_retime_rejected,
                "fallback": fallback,
                "conventional_gpu_ms": conventional_gpu_ms,
                "conventional_correct": conventional_correct,
                "commit_kind": commit_kind,
                "nrx_bound_violation": nrx_bound_violation,
                "conv_bound_violation": conv_bound_violation,
                "duplicate_commit": duplicate_commit,
                "recovery_gap_ai_units": gap_ai_units,
                "recovery_gap_ai_deadline_ns": gap_ai_deadline_ns,
                "correct": bool(correct),
                "response_ms": response_ms,
                "deadline_miss": response_ms > args.deadline_ms,
            })
            if background is not None and index + 1 < args.iterations:
                next_release_ns = release_ns + period_ns
                while (
                    background.enabled
                    and time.perf_counter_ns() + budget_ns + ai_guard_ns
                    <= next_release_ns
                ):
                    background.run(index, next_release_ns, args.ai_budget_ms)
    finally:
        # This is deliberately outside the timed RAN epoch. Confirm31 shows
        # that destroying an initialized MPS client during the epoch is unsafe.
        try:
            if background is not None:
                background.close()
            owner.publish_forward(TERMINATE_SEQ)
            time.sleep(0.2)
        finally:
            owner.close()

    result = {
        "schema": "softwall-same-request-transaction-controller-v1",
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "tag": args.tag,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "warmup_correct": warmup_correct,
        "conventional_warmup_correct": conventional_warmup_correct,
        "noisy_warmup_units": noisy_warmup_units,
        "noisy_conventional_warmup_units": noisy_conventional_warmup_units,
        "period_ms": args.period_ms,
        "deadline_ms": args.deadline_ms,
        "endpoint_timeout_ms": args.endpoint_timeout_ms,
        "policy": "s2_transactional",
        "nrx_bound_ms": args.nrx_bound_ms,
        "conv_bound_ms": args.conv_bound_ms,
        "commit_guard_ms": args.commit_guard_ms,
        "recovery_gap_ai": args.recovery_gap_ai,
        "fallback_timing_policy": args.fallback_timing,
        "conventional_warmup": args.conventional_warmup,
        "payload_seed": args.seed,
        "snr_db": args.snr_db,
        "channel_seed_base": args.channel_seed_base,
        "lifecycle": "persistent during RAN epoch; teardown after final release",
        "correct_releases": sum(item["correct"] for item in records),
        "deadline_misses": sum(item["deadline_miss"] for item in records),
        "nrx_commits": sum(item["commit_kind"] == "nrx" for item in records),
        "conv_commits": sum(item["commit_kind"] == "conventional" for item in records),
        "admission_rejections": sum(item["admission_rejected"] for item in records),
        "nrx_bound_violations": sum(item["nrx_bound_violation"] for item in records),
        "conv_bound_violations": sum(item["conv_bound_violation"] for item in records),
        "duplicate_commits": sum(item["duplicate_commit"] for item in records),
        "recovery_gap_ai_units": sum(item["recovery_gap_ai_units"] for item in records),
        "runtime_metrics": dict(runtime.metrics),
        "fallback_calendar_final": runtime.fallback_calendar.snapshot(),
        "endpoint_final": runtime.endpoints["nrx0"].snapshot(),
        "fallbacks": sum(item["fallback"] for item in records),
        "early_fallbacks": sum(item["fallback_early"] for item in records),
        "early_retime_rejections": sum(
            item["early_retime_rejected"] for item in records
        ),
        "fallback_not_started": sum(
            item["fallback_started"] is False for item in records
        ),
        "fallback_start_late_gt_1ms": sum(
            item["fallback_start_lateness_ms"] is not None
            and item["fallback_start_lateness_ms"] > 1.0
            for item in records
        ),
        "conventional_runs": sum(
            item["conventional_gpu_ms"] is not None for item in records
        ),
        "endpoint_requests": sum(
            item["front_gpu_ms"] is not None for item in records
        ),
        "endpoint_timeouts": sum(item["endpoint_timeout"] for item in records),
        "endpoint_faults": endpoint_faults,
        "ai_budget_ms": args.ai_budget_ms if background else None,
        "ai_guard_ms": args.ai_guard_ms if background else None,
        "ai_rpc_timeout_ms": args.ai_rpc_timeout_ms if background else None,
        "background_units": len(background.records) if background else 0,
        "background_gpu_ms": summary([
            item["gpu_ms"] for item in background.records
        ]) if background else None,
        "background_budget_violations": sum(
            item["budget_violation"] for item in background.records
        ) if background else 0,
        "background_release_crossings": sum(
            item["crossed_next_release"] for item in background.records
        ) if background else 0,
        "background_faults": background.faults if background else [],
        "background_records": background.records if background else [],
        "response_ms": summary([item["response_ms"] for item in records]),
        "front_gpu_ms": summary([
            item["front_gpu_ms"] for item in records
            if item["front_gpu_ms"] is not None
        ]),
        "post_gpu_ms": summary([
            item["post_gpu_ms"] for item in records
            if item["post_gpu_ms"] is not None
        ]),
        "records": records,
    }
    atomic_json(args.output, result)
    print(
        f"[SAME-REQUEST] correct={result['correct_releases']}/{args.iterations} "
        f"miss={result['deadline_misses']} nrx={result['nrx_commits']} "
        f"fallback={result['fallbacks']} "
        f"bg={result['background_units']} "
        f"p99={result['response_ms']['p99']:.3f}ms",
        flush=True,
    )


if __name__ == "__main__":
    main()
