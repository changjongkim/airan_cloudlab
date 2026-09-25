#!/usr/bin/env python3
"""S2 optional NeuralRx, reserved fallback, single commit, and AI slack recovery."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from pathlib import Path

from dual_receiver_phy import PairedDualReceiver
from isca_v2.dart_runtime import (
    DartRequest,
    DartRuntime,
    EndpointState,
    FallbackCalendar,
    ProfileTable,
    ServiceProfile,
)
from softwall_phy import summary, wait_until


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


class BackgroundClient:
    def __init__(self, path: str, timeout_ms: float) -> None:
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.connect(path)
        self.socket.settimeout(timeout_ms / 1000.0)
        self.channel = self.socket.makefile("rwb", buffering=0)
        self.enabled = True
        self.records: list[dict] = []
        self.faults: list[dict] = []

    def run(self, release_index: int, next_release_ns: int, budget_ms: float) -> None:
        admitted_ns = time.perf_counter_ns()
        try:
            self.channel.write(b'{"op":"run"}\n')
            payload = self.channel.readline()
            if not payload:
                raise ConnectionError("background worker closed the RPC channel")
            response = json.loads(payload)
            if not response.get("ok"):
                raise RuntimeError(f"background worker rejected unit: {response}")
        except (BrokenPipeError, ConnectionError, OSError, ValueError, RuntimeError) as error:
            self.enabled = False
            self.faults.append({
                "release_index": release_index,
                "detected_ns": time.perf_counter_ns(),
                "type": type(error).__name__,
                "message": str(error),
            })
            return
        returned_ns = time.perf_counter_ns()
        execution_ms = (returned_ns - admitted_ns) / 1e6
        record = {
            "release_index": release_index,
            "admitted_ns": admitted_ns,
            "returned_ns": returned_ns,
            "gpu_ms": response["gpu_ms"],
            "execution_ms": execution_ms,
            "budget_ms": budget_ms,
            "budget_violation": execution_ms > budget_ms,
            "crossed_next_release": returned_ns > next_release_ns,
        }
        self.records.append(record)
        if record["budget_violation"] or record["crossed_next_release"]:
            self.enabled = False
            self.faults.append({
                "release_index": release_index,
                "detected_ns": returned_ns,
                "type": "AIContractViolation",
                "message": json.dumps(record),
            })

    def close(self) -> None:
        if self.enabled:
            try:
                self.channel.write(b'{"op":"stop"}\n')
                self.channel.readline()
            except (BrokenPipeError, OSError):
                pass
        try:
            self.channel.close()
        except OSError:
            pass
        self.socket.close()


def run_conventional(
    receiver: PairedDualReceiver, cells: int, conv_bound_ms: float
) -> dict:
    results = [receiver.run_conventional() for _ in range(cells)]
    return {
        "correct": all(item[1] for item in results),
        "nrx_branch_correct": [None] * cells,
        "conv_branch_correct": [bool(item[1]) for item in results],
        "gpu_ms": sum(item[0] for item in results),
        "crc_failures": sum(item[2] for item in results),
        "payload_mismatches": sum(item[3] for item in results),
        "nrx_commits": 0,
        "conv_commits": cells,
        "injected_nrx_failures": 0,
        "duplicate_commits": 0,
        "fallback_started": 0,
        "admission_rejections": 0,
        "nrx_bound_violations": 0,
        "conv_bound_violations": sum(
            item[0] > conv_bound_ms for item in results
        ),
    }


def run_nrx_only(
    receiver: PairedDualReceiver,
    cells: int,
    inject_failure: bool,
    nrx_bound_ms: float,
) -> dict:
    results = [receiver.run_neural() for _ in range(cells)]
    injected = cells if inject_failure else 0
    return {
        "correct": all(item[1] for item in results) and not inject_failure,
        "nrx_branch_correct": [
            bool(item[1]) and not inject_failure for item in results
        ],
        "conv_branch_correct": [None] * cells,
        "gpu_ms": sum(item[0] for item in results),
        "crc_failures": sum(item[2] for item in results),
        "payload_mismatches": sum(item[3] for item in results),
        "nrx_commits": cells - injected,
        "conv_commits": 0,
        "injected_nrx_failures": injected,
        "duplicate_commits": 0,
        "fallback_started": 0,
        "admission_rejections": 0,
        "nrx_bound_violations": sum(item[0] > nrx_bound_ms for item in results),
        "conv_bound_violations": 0,
    }


def run_eager_dual(
    receiver: PairedDualReceiver,
    cells: int,
    inject_failure: bool,
    nrx_bound_ms: float,
    conv_bound_ms: float,
) -> dict:
    nrx = [receiver.run_neural() for _ in range(cells)]
    conventional = [receiver.run_conventional() for _ in range(cells)]
    nrx_valid = [item[1] and not inject_failure for item in nrx]
    selected = [
        nrx[index] if nrx_valid[index] else conventional[index]
        for index in range(cells)
    ]
    nrx_commits = sum(nrx_valid)
    return {
        "correct": all(item[1] for item in selected),
        "nrx_branch_correct": [
            bool(item[1]) and not inject_failure for item in nrx
        ],
        "conv_branch_correct": [bool(item[1]) for item in conventional],
        "gpu_ms": sum(item[0] for item in nrx + conventional),
        "crc_failures": sum(item[2] for item in selected),
        "payload_mismatches": sum(item[3] for item in selected),
        "nrx_commits": nrx_commits,
        "conv_commits": cells - nrx_commits,
        "injected_nrx_failures": cells if inject_failure else 0,
        "duplicate_commits": 0,
        "fallback_started": 0,
        "admission_rejections": 0,
        "nrx_bound_violations": sum(item[0] > nrx_bound_ms for item in nrx),
        "conv_bound_violations": sum(item[0] > conv_bound_ms for item in conventional),
    }


def make_runtime(
    nrx_bound_ns: int,
    conv_bound_ns: int,
    commit_guard_ns: int,
    cells: int,
) -> DartRuntime:
    return DartRuntime(
        endpoints=[EndpointState("nrx0", ring_depth=max(2, cells))],
        profiles=ProfileTable({
            ("nrx0", 0, 0, "isolated"): ServiceProfile(
                forward_ns=0,
                service_ns=nrx_bound_ns,
                backward_ns=0,
            )
        }),
        commit_guard_ns=commit_guard_ns,
        fallback_calendar=FallbackCalendar(
            capacity=1,
            service_ns=conv_bound_ns,
            commit_guard_ns=commit_guard_ns,
        ),
    )


def run_s2(
    receiver: PairedDualReceiver,
    runtime: DartRuntime,
    release_index: int,
    release_ns: int,
    deadline_ns: int,
    cells: int,
    inject_failure: bool,
    nrx_bound_ns: int,
    conv_bound_ns: int,
    commit_guard_ns: int,
) -> dict:
    transactions = []
    for cell in range(cells):
        # Reserve sequential conventional recovery slots ending before the
        # common deadline and commit guard.
        fallback_start_ns = (
            deadline_ns
            - commit_guard_ns
            - conv_bound_ns * (cells - cell)
        )
        request = DartRequest(
            slot_id=release_index * cells + cell,
            epoch=1,
            graph_id=0,
            tensor_class=0,
            release_ns=release_ns,
            deadline_ns=deadline_ns,
            fallback_latest_start_ns=fallback_start_ns,
            input_slot=cell,
            output_slot=cell,
        )
        transactions.append(runtime.submit(request, time.perf_counter_ns()))

    nrx_results = []
    fallbacks = []
    duplicate_commits = 0
    admission_rejections = 0
    nrx_bound_violations = 0
    for cell, transaction in enumerate(transactions):
        if transaction.reservation is None:
            admission_rejections += 1
            conventional = receiver.run_conventional()
            committed = runtime.complete_conventional(
                transaction, time.perf_counter_ns(), result_slot=cell
            )
            nrx_results.append((None, conventional, committed))
            continue
        neural = receiver.run_neural()
        completed_ns = time.perf_counter_ns()
        nrx_bound_violations += int(neural[0] * 1e6 > nrx_bound_ns)
        visible = (
            neural[1]
            and not inject_failure
            and completed_ns <= transaction.request.fallback_latest_start_ns
        )
        committed = runtime.complete_nrx(
            transaction,
            transaction.request.slot_id,
            transaction.request.epoch,
            completed_ns,
            payload_visible=visible,
        )
        nrx_results.append((neural, None, committed))
        if not committed:
            fallbacks.append((transaction.request.fallback_latest_start_ns, cell, transaction))

    conv_bound_violations = 0
    fallback_started = 0
    for fallback_start_ns, cell, transaction in sorted(fallbacks):
        wait_until(fallback_start_ns)
        if not runtime.start_fallback(transaction, time.perf_counter_ns()):
            continue
        fallback_started += 1
        conventional = receiver.run_conventional()
        conv_bound_violations += int(conventional[0] * 1e6 > conv_bound_ns)
        runtime.complete_conventional(
            transaction, time.perf_counter_ns(), result_slot=cell
        )
        prior = nrx_results[cell]
        nrx_results[cell] = (prior[0], conventional, prior[2])
        # A completion from the abandoned NRx branch must never commit after
        # conventional recovery has closed the transaction.
        duplicate_commits += int(runtime.complete_nrx(
            transaction,
            transaction.request.slot_id,
            transaction.request.epoch,
            time.perf_counter_ns(),
            payload_visible=True,
        ))

    correct = True
    crc_failures = 0
    payload_mismatches = 0
    nrx_commits = 0
    conv_commits = 0
    gpu_ms = 0.0
    for transaction, (neural, conventional, initially_committed) in zip(
        transactions, nrx_results
    ):
        if neural is not None:
            gpu_ms += neural[0]
        if conventional is not None:
            gpu_ms += conventional[0]
        if transaction.commit_state.value == "nrx_won":
            nrx_commits += 1
            correct = correct and bool(neural and neural[1]) and not inject_failure
            crc_failures += neural[2] if neural else 1
            payload_mismatches += neural[3] if neural else 1
        elif transaction.commit_state.value == "conv_won":
            conv_commits += 1
            correct = correct and bool(conventional and conventional[1])
            crc_failures += conventional[2] if conventional else 1
            payload_mismatches += conventional[3] if conventional else 1
        else:
            correct = False
    return {
        "correct": correct,
        "nrx_branch_correct": [
            (bool(item[0][1]) and not inject_failure)
            if item[0] is not None
            else None
            for item in nrx_results
        ],
        "conv_branch_correct": [
            bool(item[1][1]) if item[1] is not None else None
            for item in nrx_results
        ],
        "gpu_ms": gpu_ms,
        "crc_failures": crc_failures,
        "payload_mismatches": payload_mismatches,
        "nrx_commits": nrx_commits,
        "conv_commits": conv_commits,
        "injected_nrx_failures": cells if inject_failure else 0,
        "duplicate_commits": duplicate_commits,
        "fallback_started": fallback_started,
        "admission_rejections": admission_rejections,
        "nrx_bound_violations": nrx_bound_violations,
        "conv_bound_violations": conv_bound_violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=("conventional_only", "nrx_only", "eager_dual", "s2_reserved"),
        required=True,
    )
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--cells", type=int, default=1)
    parser.add_argument("--period-ms", type=float, default=50.0)
    parser.add_argument("--deadline-ms", type=float, default=40.0)
    parser.add_argument("--nrx-bound-ms", type=float, default=14.0)
    parser.add_argument("--conv-bound-ms", type=float, default=24.0)
    parser.add_argument("--commit-guard-ms", type=float, default=2.0)
    parser.add_argument("--inject-failure-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20270420)
    parser.add_argument("--snr-db", type=float)
    parser.add_argument("--channel-seed-base", type=int, default=20270750)
    parser.add_argument("--inject-start-delay-every", type=int, default=0)
    parser.add_argument("--inject-start-delay-ms", type=float, default=0.0)
    parser.add_argument("--socket")
    parser.add_argument("--ai-budget-ms", type=float, default=6.0)
    parser.add_argument("--ai-guard-ms", type=float, default=1.0)
    parser.add_argument("--ai-rpc-timeout-ms", type=float, default=7.0)
    args = parser.parse_args()
    if min(
        args.iterations,
        args.cells,
        args.period_ms,
        args.deadline_ms,
        args.nrx_bound_ms,
        args.conv_bound_ms,
        args.commit_guard_ms,
    ) <= 0:
        parser.error("counts and timing parameters must be positive")
    if args.inject_start_delay_every < 0 or args.inject_start_delay_ms < 0:
        parser.error("start-delay injection parameters must be non-negative")
    # Even an immediate NRx abort cannot recover a correlated all-fail burst
    # if its single conventional lane needs more time than this deadline.
    if (
        args.policy == "s2_reserved"
        and args.cells * args.conv_bound_ms + args.commit_guard_ms
        > args.deadline_ms
    ):
        parser.error(
            "s2_reserved cannot fit all correlated conventional fallbacks "
            "on one lane within the deadline"
        )

    receiver = PairedDualReceiver(args.engine, seed=args.seed)
    for _ in range(args.warmup):
        receiver.run_conventional()
        receiver.run_neural()
    if args.snr_db is not None:
        # CuPy lazily compiles the random/channel kernels on their first use.
        # Keep that one-time setup outside the release timeline, then exercise
        # both receiver paths once with a noisy input as well.
        receiver.apply_rayleigh_awgn(
            args.snr_db, args.channel_seed_base - 1
        )
        receiver.run_conventional()
        receiver.run_neural()

    nrx_bound_ns = round(args.nrx_bound_ms * 1e6)
    conv_bound_ns = round(args.conv_bound_ms * 1e6)
    commit_guard_ns = round(args.commit_guard_ms * 1e6)
    runtime = make_runtime(
        nrx_bound_ns, conv_bound_ns, commit_guard_ns, args.cells
    )
    background = (
        BackgroundClient(args.socket, args.ai_rpc_timeout_ms)
        if args.socket
        else None
    )
    records = []
    first_release_ns = time.perf_counter_ns() + 200_000_000
    period_ns = round(args.period_ms * 1e6)
    deadline_delta_ns = round(args.deadline_ms * 1e6)
    budget_ns = round(args.ai_budget_ms * 1e6)
    ai_guard_ns = round(args.ai_guard_ms * 1e6)
    for index in range(args.iterations):
        release_ns = first_release_ns + index * period_ns
        if args.snr_db is not None:
            receiver.apply_rayleigh_awgn(
                args.snr_db, args.channel_seed_base + index
            )
        wait_until(release_ns)
        injected_start_delay = (
            args.inject_start_delay_every > 0
            and index % args.inject_start_delay_every == 0
        )
        if injected_start_delay:
            wait_until(
                release_ns + round(args.inject_start_delay_ms * 1e6)
            )
        start_ns = time.perf_counter_ns()
        inject_failure = (
            args.inject_failure_every > 0
            and index % args.inject_failure_every == 0
        )
        if args.policy == "conventional_only":
            outcome = run_conventional(receiver, args.cells, args.conv_bound_ms)
        elif args.policy == "nrx_only":
            outcome = run_nrx_only(
                receiver, args.cells, inject_failure, args.nrx_bound_ms
            )
        elif args.policy == "eager_dual":
            outcome = run_eager_dual(
                receiver,
                args.cells,
                inject_failure,
                args.nrx_bound_ms,
                args.conv_bound_ms,
            )
        else:
            outcome = run_s2(
                receiver,
                runtime,
                index,
                release_ns,
                release_ns + deadline_delta_ns,
                args.cells,
                inject_failure,
                nrx_bound_ns,
                conv_bound_ns,
                commit_guard_ns,
            )
        completed_ns = time.perf_counter_ns()
        response_ms = (completed_ns - release_ns) / 1e6
        records.append({
            "index": index,
            "channel_seed": (
                args.channel_seed_base + index
                if args.snr_db is not None
                else None
            ),
            "release_ns": release_ns,
            "start_lateness_ms": (start_ns - release_ns) / 1e6,
            "injected_start_delay": injected_start_delay,
            "completed_ns": completed_ns,
            "response_ms": response_ms,
            "deadline_miss": response_ms > args.deadline_ms,
            **outcome,
        })
        if background is not None and index + 1 < args.iterations:
            next_release_ns = release_ns + period_ns
            while (
                background.enabled
                and time.perf_counter_ns() + budget_ns + ai_guard_ns
                <= next_release_ns
            ):
                background.run(index, next_release_ns, args.ai_budget_ms)

    if background is not None:
        background.close()
    result = {
        "schema": "softwall-s2-hardware-v1",
        "policy": args.policy,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cells": args.cells,
        "iterations": args.iterations,
        "period_ms": args.period_ms,
        "deadline_ms": args.deadline_ms,
        "nrx_bound_ms": args.nrx_bound_ms,
        "conv_bound_ms": args.conv_bound_ms,
        "commit_guard_ms": args.commit_guard_ms,
        "inject_failure_every": args.inject_failure_every,
        "payload_seed": args.seed,
        "snr_db": args.snr_db,
        "channel_seed_base": args.channel_seed_base,
        "inject_start_delay_every": args.inject_start_delay_every,
        "inject_start_delay_ms": args.inject_start_delay_ms,
        "response_ms": summary([item["response_ms"] for item in records]),
        "gpu_ms": summary([item["gpu_ms"] for item in records]),
        "deadline_misses": sum(item["deadline_miss"] for item in records),
        "correct_releases": sum(item["correct"] for item in records),
        "incorrect_releases": sum(not item["correct"] for item in records),
        "nrx_commits": sum(item["nrx_commits"] for item in records),
        "conv_commits": sum(item["conv_commits"] for item in records),
        "injected_nrx_failures": sum(
            item["injected_nrx_failures"] for item in records
        ),
        "fallback_started": sum(item["fallback_started"] for item in records),
        "duplicate_commits": sum(item["duplicate_commits"] for item in records),
        "admission_rejections": sum(
            item["admission_rejections"] for item in records
        ),
        "nrx_bound_violations": sum(
            item["nrx_bound_violations"] for item in records
        ),
        "conv_bound_violations": sum(
            item["conv_bound_violations"] for item in records
        ),
        "background_units": len(background.records) if background else 0,
        "background_gpu_ms": summary(
            [item["gpu_ms"] for item in background.records]
        ) if background else None,
        "background_budget_violations": sum(
            item["budget_violation"] for item in background.records
        ) if background else 0,
        "background_release_crossings": sum(
            item["crossed_next_release"] for item in background.records
        ) if background else 0,
        "background_faults": background.faults if background else [],
        "runtime_metrics": dict(runtime.metrics),
        "records": records,
        "background_records": background.records if background else [],
    }
    atomic_json(Path(args.output), result)
    print(
        f"[S2] policy={args.policy} n={args.iterations} "
        f"p99={result['response_ms']['p99']:.3f}ms "
        f"max={result['response_ms']['max']:.3f}ms "
        f"miss={result['deadline_misses']} correct={result['correct_releases']} "
        f"nrx={result['nrx_commits']} conv={result['conv_commits']} "
        f"bg={result['background_units']} duplicate={result['duplicate_commits']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
