#!/usr/bin/env python3
"""Trace-driven, same-radio comparison of three safe recovery substrates.

All arms use the same max-radio/low-feature policy, PHY inputs, MPS caps,
service bounds, physical fences, and fault pattern.  The only treatment is how
qualified recovery slack may be exposed to the background-Qwen request queue.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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
from softwall_phy import summary, wait_until
from fault_contained_global_trace_client import FaultContainedGlobalTraceClient
from trace_background_client import TraceBackgroundClient, TraceRequestQueue


RX_ELEMENTS = 1 * 3276 * 12 * 4
CE_ELEMENTS = 1 * 4914 * 1 * 4
FWD_ELEMENTS = 2 * RX_ELEMENTS + 2 * CE_ELEMENTS
LLR_ELEMENTS = 2 * 1 * 3276 * 12


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synchronized_first_release(barrier_dir: Path, member: int, participants: int) -> int:
    """Create a same-node process barrier and return a shared monotonic release."""
    barrier_dir.mkdir(parents=True, exist_ok=True)
    ready = barrier_dir / f"ready_{member}.json"
    atomic_json(ready, {"member": member, "ready_ns": time.perf_counter_ns()})
    release_file = barrier_dir / "first_release_ns"
    deadline = time.monotonic() + 120.0
    if member == 0:
        while not all((barrier_dir / f"ready_{index}.json").is_file()
                      for index in range(participants)):
            if time.monotonic() > deadline:
                raise TimeoutError("start barrier participants did not become ready")
            time.sleep(0.01)
        release_ns = time.perf_counter_ns() + 1_000_000_000
        temporary = release_file.with_suffix(".tmp")
        temporary.write_text(str(release_ns), encoding="utf-8")
        temporary.replace(release_file)
    while not release_file.is_file():
        if time.monotonic() > deadline:
            raise TimeoutError("start barrier release was not published")
        time.sleep(0.01)
    release_ns = int(release_file.read_text(encoding="utf-8"))
    if release_ns <= time.perf_counter_ns():
        raise RuntimeError("start barrier release is already late")
    return release_ns


def parse_bound_map(text: str) -> dict[int, int]:
    """Parse `length:milliseconds,...` into nanosecond bounds."""
    try:
        pairs = [item.split(":", 1) for item in text.split(",")]
        result = {int(length): round(float(ms) * 1e6) for length, ms in pairs}
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("invalid AI bound map") from error
    if not result or any(length <= 0 or bound <= 0 for length, bound in result.items()):
        raise argparse.ArgumentTypeError("AI bounds must be positive")
    return result


def select_request(queue, now_ns, horizon_ns, guard_ns, bounds_ns):
    return queue.peek_edf_fitting(
        now_ns, horizon_ns, guard_ns, bounds_ns
    )


def commit_selected(queue, request):
    """Commit a prepared global request; local queues need no extra step."""
    commit = getattr(queue, "commit", None)
    if commit is None:
        return True
    result = commit(request)
    return True if result is None else bool(result)


def release_selected(queue, request):
    """Abort a prepared global request after local lease rejection."""
    release = getattr(queue, "release", None)
    if release is None:
        return True
    result = release(request)
    return True if result is None else bool(result)


def complete_selected(queue, request, returned_ns):
    """Retire a completed request without retrying an ambiguous global RPC."""
    result = queue.complete(request, returned_ns)
    return True if result is None else bool(result)


def global_queue_status(queue):
    return {
        "enabled": getattr(queue, "enabled", True),
        "faults": list(getattr(queue, "faults", [])),
    }


def execute_request(background, queue, request, phase, bounds_ns, horizon_ns, guard_ns):
    bound_ns = bounds_ns[request["context_length"]]
    if not commit_selected(queue, request):
        return None
    record = background.run_request(request, phase, bound_ns / 1e6)
    if record is None:
        raise RuntimeError("trace background RPC failed")
    record["global_complete_confirmed"] = complete_selected(
        queue, request, record["returned_ns"]
    )
    record["horizon_ns"] = horizon_ns
    record["horizon_guard_violation"] = record["returned_ns"] + guard_ns > horizon_ns
    if record["bound_violation"] or record["horizon_guard_violation"]:
        raise RuntimeError("trace AI exceeded its certified execution window")
    return record


def finish_pre_ai(thread, holder, background, queue, request, cutoff_ns, guard_ns):
    """Join one in-flight AI RPC before any unqualified conventional overlap."""
    if thread is None:
        return
    thread.join(timeout=1.0)
    if thread.is_alive() or not background.enabled or holder.get("record") is None:
        raise RuntimeError("asynchronous pre-radio AI completion unconfirmed")
    record = holder["record"]
    record["global_complete_confirmed"] = complete_selected(
        queue, request, record["returned_ns"]
    )
    record["earliest_fallback_start_ns"] = cutoff_ns
    record["horizon_ns"] = cutoff_ns
    record["horizon_guard_violation"] = record["returned_ns"] + guard_ns > cutoff_ns
    if record["bound_violation"] or record["horizon_guard_violation"]:
        raise RuntimeError("asynchronous AI crossed earliest recovery guard")


def injected_failure_cells(pattern: str, every: int, index: int, admitted: list[int]) -> set[int]:
    if every <= 0 or index % every or not admitted:
        return set()
    if pattern == "correlated":
        return set(admitted)
    # Alternate a single failure and a correlated failure without using an
    # outcome or system-arm label.  The admitted set is identical by design.
    event = index // every
    return set(admitted if event % 2 == 0 else admitted[:1])


def compact_recovery_tail(pending, guard_ns: int, service_ns: int):
    """Place the remaining same-deadline credits at the certified tail.

    All four requests in one release share a deadline and one conventional
    lane.  NRx successes remove credits from the original four-slot chain.
    Packing the still-live credits against the common deadline exposes exactly
    the conditional prefix slack while preserving an executable all-fail plan.
    """
    if not pending:
        return []
    deadlines = {tx.request.deadline_ns for tx in pending}
    if len(deadlines) != 1:
        raise RuntimeError("trace mode requires one common radio deadline")
    ordered = sorted(pending, key=lambda tx: tx.request.slot_id)
    tail_end_ns = next(iter(deadlines)) - guard_ns
    first_start_ns = tail_end_ns - len(ordered) * service_ns
    return [
        (tx, first_start_ns + position * service_ns, 0)
        for position, tx in enumerate(ordered)
    ]


def fallback_is_early(transaction, now_ns: int) -> bool:
    """Use the live credit, which may have moved after NRx observation."""
    reservation = transaction.fallback_reservation
    if reservation is None:
        raise RuntimeError("mandatory fallback has no live recovery credit")
    return now_ns < reservation.start_ns


def certificate_order(transactions, radio):
    """Return open recovery work in the executable certificate order.

    A request that did not enter NeuralRx and a request whose NeuralRx result
    failed are both ready conventional work.  Treating those classes in two
    separate loops can try to move a later credit across an earlier live
    credit.  The recovery calendar is the authority: execute every open item
    by its current reserved start, then by deadline and slot for deterministic
    ties.
    """
    pending = [
        (cell, tx) for cell, tx in enumerate(transactions)
        if radio[cell]["commit_kind"] is None
    ]
    if any(tx.fallback_reservation is None for _, tx in pending):
        raise RuntimeError("open recovery transaction has no certificate credit")
    return sorted(
        pending,
        key=lambda item: (
            item[1].fallback_reservation.start_ns,
            item[1].request.deadline_ns,
            item[1].request.slot_id,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--ipc-dir", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--trace-sha256", required=True)
    parser.add_argument("--system", choices=("static", "work_conserving", "softwall"), required=True)
    parser.add_argument("--ai-bound-map", type=parse_bound_map, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--cells", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--nrx-endpoints", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--period-ms", type=float, default=150)
    parser.add_argument("--deadline-ms", type=float, default=130)
    parser.add_argument("--nrx-bound-ms", type=float, default=50)
    parser.add_argument("--conv-bound-ms", type=float, default=25)
    parser.add_argument("--commit-guard-ms", type=float, default=2)
    parser.add_argument("--endpoint-timeout-ms", type=float, default=100)
    parser.add_argument("--ai-guard-ms", type=float, default=2)
    parser.add_argument("--ai-rpc-timeout-ms", type=float, default=100)
    parser.add_argument("--seed", type=int, default=20321055)
    parser.add_argument("--snr-db", type=float, default=-8.5)
    parser.add_argument("--channel-seed-base", type=int, default=20355000)
    parser.add_argument("--gate-mode", choices=("always", "low_threshold"), required=True)
    parser.add_argument("--inject-correlated-failure-every", type=int, default=0)
    parser.add_argument("--fault-pattern", choices=("correlated", "mixed"), default="mixed")
    parser.add_argument("--gate-threshold", type=float, required=True)
    parser.add_argument("--gc-mode", choices=("on", "off"), required=True)
    parser.add_argument("--alternate-admission-order", action="store_true")
    parser.add_argument("--clean-channel", action="store_true")
    parser.add_argument("--start-barrier-dir", type=Path)
    parser.add_argument("--start-barrier-member", type=int)
    parser.add_argument("--start-barrier-participants", type=int)
    parser.add_argument("--global-trace-broker")
    parser.add_argument("--global-home-id", type=int)
    args = parser.parse_args()
    if (args.iterations <= 0 or args.warmup < 0
            or args.inject_correlated_failure_every < 0
            or not 1 <= args.nrx_endpoints <= args.cells):
        parser.error("invalid count")
    barrier_values = (
        args.start_barrier_dir,
        args.start_barrier_member,
        args.start_barrier_participants,
    )
    if any(value is not None for value in barrier_values):
        if any(value is None for value in barrier_values):
            parser.error("all start-barrier arguments are required together")
        if not 0 <= args.start_barrier_member < args.start_barrier_participants:
            parser.error("invalid start-barrier member/participant count")
    if (args.global_trace_broker is None) != (args.global_home_id is None):
        parser.error("global broker path and home id are required together")
    if args.global_home_id is not None and args.global_home_id < 0:
        parser.error("invalid global home id")
    if min(
        args.period_ms, args.deadline_ms, args.nrx_bound_ms,
        args.conv_bound_ms, args.commit_guard_ms,
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
    if not args.trace.is_file() or sha256(args.trace) != args.trace_sha256:
        parser.error("sealed trace is missing or changed")
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    trace_lengths = {int(value) for value in trace["mapping"]["allowed_context_lengths"]}
    if set(args.ai_bound_map) != trace_lengths:
        parser.error("AI bound map must cover exactly the sealed trace lengths")

    cp.cuda.runtime.setDevice(0)
    endpoint_contexts = []
    for endpoint in range(args.nrx_endpoints):
        forward = cp.empty(FWD_ELEMENTS, dtype=cp.float32)
        backward = cp.empty(LLR_ELEMENTS, dtype=cp.float32)
        owner = CudaIpcOwner(
            f"{args.tag_prefix}_c{endpoint}", forward, backward,
            directory=args.ipc_dir,
        )
        endpoint_contexts.append({
            "endpoint_id": f"nrx{endpoint}", "owner": owner,
            "forward": forward, "backward": backward, "sequence": 0,
        })
    endpoint_by_id = {
        ctx["endpoint_id"]: ctx for ctx in endpoint_contexts
    }
    contexts = []
    for cell in range(args.cells):
        receiver = PairedDualReceiver(
            args.engine, seed=args.seed + cell, enable_local_neural=False
        )
        contexts.append({"receiver": receiver})
    profiles = ProfileTable({
        (f"nrx{cell}", 0, 0, "isolated"): ServiceProfile(
            forward_ns=0,
            service_ns=round(args.nrx_bound_ms * 1e6),
            backward_ns=0,
        )
        for cell in range(args.nrx_endpoints)
    })
    runtime = DartRuntime(
        # Each IPC owner exports exactly one forward/backward buffer pair.
        # Advertising a second credit here would permit an unsafe second
        # request before the first worker has physically completed.
        endpoints=[
            *[EndpointState(f"nrx{index}", ring_depth=1)
              for index in range(args.nrx_endpoints)],
            EndpointState("ai", ring_depth=1),
        ],
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
    release_stage_records = []
    gc_events = []
    gc_open = {}
    def gc_probe(phase, info):
        now_ns = time.perf_counter_ns()
        generation = info["generation"]
        if phase == "start":
            gc_open[generation] = now_ns
        else:
            begin_ns = gc_open.pop(generation, None)
            if begin_ns is not None:
                gc_events.append({"generation": generation, "begin_ns": begin_ns,
                                  "end_ns": now_ns, "collected": info["collected"]})
    recovery_decisions = []
    warmup_correct = [0] * args.cells
    conventional_warmup_correct = [0] * args.cells
    faults = []
    index = -1
    transactions = []
    radio = []
    trace_queue = None
    current_phase = "worker-warmup"
    try:
        for ctx in endpoint_contexts:
            ctx["owner"].wait_ready(120.0)
        for _ in range(args.warmup):
            for cell, ctx in enumerate(contexts):
                conventional_warmup_correct[cell] += int(ctx["receiver"].run_conventional()[1])
                endpoint_ctx = endpoint_contexts[cell % args.nrx_endpoints]
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
            endpoint_ctx = endpoint_contexts[cell % args.nrx_endpoints]
            endpoint_ctx["sequence"] += 1
            ctx["receiver"].prepare_neural_ipc(endpoint_ctx["forward"])
            endpoint_ctx["owner"].publish_forward(endpoint_ctx["sequence"])
            endpoint_ctx["owner"].wait_backward(endpoint_ctx["sequence"], 2.0)
            ctx["receiver"].complete_neural_ipc(endpoint_ctx["backward"])
            ctx["receiver"].run_conventional()

        background = TraceBackgroundClient(args.socket, args.ai_rpc_timeout_ms)
        period_ns = round(args.period_ms * 1e6)
        deadline_ns = round(args.deadline_ms * 1e6)
        conv_bound_ns = round(args.conv_bound_ms * 1e6)
        guard_ns = round(args.commit_guard_ms * 1e6)
        nrx_bound_ns = round(args.nrx_bound_ms * 1e6)
        ai_guard_ns = round(args.ai_guard_ms * 1e6)
        gc.callbacks.append(gc_probe)
        if args.gc_mode == "off":
            gc.disable()
        else:
            gc.enable()
        current_phase = "start-barrier"
        first_release_ns = (
            synchronized_first_release(
                args.start_barrier_dir,
                args.start_barrier_member,
                args.start_barrier_participants,
            )
            if args.start_barrier_dir is not None
            else time.perf_counter_ns() + 200_000_000
        )
        trace_queue = (
            FaultContainedGlobalTraceClient(
                args.global_trace_broker, args.global_home_id,
                first_release_ns, args.trace_sha256,
            )
            if args.global_trace_broker is not None
            else TraceRequestQueue(trace, first_release_ns)
        )
        static_first_offset_ns = deadline_ns - guard_ns - args.cells * conv_bound_ns
        static_end_offset_ns = deadline_ns - guard_ns
        for index in range(args.iterations):
            current_phase = "release-setup"
            release_ns = first_release_ns + index * period_ns
            stage = {"index": index, "release_ns": release_ns,
                     "channel_prep_begin_ns": time.perf_counter_ns()}
            for cell, ctx in enumerate(contexts):
                if args.clean_channel:
                    ctx["receiver"].restore_clean_slot()
                else:
                    ctx["receiver"].apply_rayleigh_awgn(
                        args.snr_db, args.channel_seed_base + index * args.cells + cell,
                        noise_reference="pre_fading",
                    )
            stage["channel_prep_end_ns"] = time.perf_counter_ns()
            stage["wait_begin_ns"] = time.perf_counter_ns()
            wait_until(release_ns)
            stage["wait_return_ns"] = time.perf_counter_ns()
            release_stage_records.append(stage)
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
                    candidate_bitmap=(1 << args.nrx_endpoints) - 1,
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
                    "nrx_admission_host_ms": None,
                    "nrx_prepare_begin_ns": None,
                    "nrx_prepare_return_ns": None,
                    "nrx_prepare_host_ms": None,
                    "nrx_prepare_gpu_ms": None,
                    "nrx_back_gpu_ms": None,
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
            stage["admission_loop_begin_ns"] = time.perf_counter_ns()
            for cell in admission_order:
                row = radio[cell]
                if row["gate_skipped"]:
                    continue
                tx = transactions[cell]
                admission_begin_ns = time.perf_counter_ns()
                admitted = runtime.admit_nrx(tx, admission_begin_ns, policy="shortest_queue")
                row["nrx_admission_host_ms"] = (time.perf_counter_ns() - admission_begin_ns) / 1e6
                row["admitted"] = admitted
                row["endpoint_id"] = tx.reservation.endpoint_id if admitted else None
            stage["admission_loop_end_ns"] = time.perf_counter_ns()
            for cell, tx in enumerate(transactions):
                if tx.reservation is None:
                    continue
                endpoint_ctx = endpoint_by_id[tx.reservation.endpoint_id]
                endpoint_ctx["sequence"] += 1
                radio[cell]["nrx_prepare_begin_ns"] = time.perf_counter_ns()
                front_gpu_ms = contexts[cell]["receiver"].prepare_neural_ipc(endpoint_ctx["forward"])
                radio[cell]["nrx_prepare_return_ns"] = time.perf_counter_ns()
                radio[cell]["nrx_prepare_host_ms"] = (
                    radio[cell]["nrx_prepare_return_ns"] - radio[cell]["nrx_prepare_begin_ns"]
                ) / 1e6
                radio[cell]["nrx_prepare_gpu_ms"] = front_gpu_ms
                radio[cell]["nrx_dispatch_begin_ns"] = time.perf_counter_ns()
                endpoint_ctx["owner"].publish_forward(endpoint_ctx["sequence"])
                radio[cell]["nrx_dispatched_ns"] = time.perf_counter_ns()
            admitted_cells = [
                cell for cell, tx in enumerate(transactions)
                if tx.reservation is not None
            ]
            forced_failure_cells = injected_failure_cells(
                args.fault_pattern, args.inject_correlated_failure_every,
                index, admitted_cells,
            )
            async_ai_thread = None
            async_ai_holder = {}
            async_ai_request = None
            async_ai_cutoff_ns = None
            if any(tx.reservation is not None for tx in transactions):
                earliest_cutoff_ns = min(
                    tx.request.fallback_latest_start_ns for tx in transactions
                )
                now_ns = time.perf_counter_ns()
                async_ai_request = select_request(
                    trace_queue, now_ns, earliest_cutoff_ns,
                    ai_guard_ns, args.ai_bound_map,
                )
                if async_ai_request is not None:
                    async_ai_cutoff_ns = earliest_cutoff_ns
                    bound_ms = args.ai_bound_map[
                        async_ai_request["context_length"]
                    ] / 1e6
                    if commit_selected(trace_queue, async_ai_request):
                        def run_pre_radio():
                            async_ai_holder["record"] = background.run_request(
                                async_ai_request, "before_nrx_observation", bound_ms
                            )
                        async_ai_thread = threading.Thread(
                            target=run_pre_radio, daemon=False
                        )
                        async_ai_thread.start()
            for cell, tx in enumerate(transactions):
                if tx.reservation is None:
                    continue
                endpoint_ctx = endpoint_by_id[tx.reservation.endpoint_id]
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
                forced_failure = cell in forced_failure_cells
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
                row["nrx_back_gpu_ms"] = neural[0]
                row["forced_nrx_failure"] = forced_failure
                row["nrx_response_ms"] = (finish_ns - release_ns) / 1e6
                row["nrx_bound_violation"] = finish_ns - release_ns > nrx_bound_ns
                row["nrx_commit"] = committed
                if committed:
                    row["commit_kind"] = "nrx"
                    row["correct"] = neural_correct
                    row["completed_ns"] = finish_ns
                    row["commit_return_ns"] = commit_return_ns
            if any(tx.reservation is None for tx in transactions):
                finish_pre_ai(
                    async_ai_thread, async_ai_holder, background, trace_queue,
                    async_ai_request, async_ai_cutoff_ns, ai_guard_ns,
                )
                async_ai_thread = None
            finish_pre_ai(
                async_ai_thread, async_ai_holder, background, trace_queue,
                async_ai_request, async_ai_cutoff_ns, ai_guard_ns,
            )
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
                "system": args.system,
                "selected_request_id": None,
            }
            recovery_action["joint_lease_retired"] = False
            if pending:
                moves = (
                    compact_recovery_tail(pending, guard_ns, conv_bound_ns)
                    if args.system == "softwall" else []
                )
                planned_start_ns = {
                    id(tx): start_ns for tx, start_ns, _ in moves
                }
                earliest_candidate_ns = min(
                    planned_start_ns.get(id(tx), tx.fallback_reservation.start_ns)
                    for tx in pending
                )
                now_ns = time.perf_counter_ns()
                horizon_ns = (
                    release_ns + static_first_offset_ns
                    if args.system == "static" else earliest_candidate_ns
                )
                request = select_request(
                    trace_queue, now_ns, horizon_ns,
                    ai_guard_ns, args.ai_bound_map,
                )
                if background.enabled and request is not None:
                    recovery_action["selected_request_id"] = request["request_id"]
                    if args.system == "softwall":
                        bound_ns = args.ai_bound_map[request["context_length"]]
                        joint = runtime.replan_and_lease(
                            moves, "ai", now_ns,
                            request["deadline_ns"] - now_ns, ai_guard_ns,
                            [BackgroundUnit(
                                request["request_id"], bound_ns,
                                float(request["value_tokens"]),
                            )],
                        )
                    else:
                        joint = None
                    if args.system == "softwall" and joint is not None:
                        replacements, lease = joint
                        recovery_action["retimed"] = bool(replacements)
                        recovery_action["ai_lease"] = True
                        earliest_recovery_ns = min(
                            tx.fallback_reservation.start_ns for tx in pending
                        )
                        ai_record = execute_request(
                            background, trace_queue, request,
                            "after_nrx_before_recovery", args.ai_bound_map,
                            earliest_recovery_ns, ai_guard_ns,
                        )
                        if ai_record is None:
                            # The broker applied commit but its reply may have
                            # been lost.  No GPU work was launched, so the
                            # local execution credit is safe to retire.  The
                            # global token remains quarantined and is never
                            # retried or reassigned.
                            if not runtime.retire_joint_lease(
                                    lease, gpu_fence_confirmed=True):
                                raise RuntimeError(
                                    "unlaunched joint AI lease could not be retired"
                                )
                            recovery_action["joint_lease_retired"] = True
                            recovery_action["broker_commit_ambiguous"] = True
                        else:
                            ai_record["earliest_fallback_start_ns"] = earliest_recovery_ns
                            ai_record["joint_lease_unit_id"] = lease.unit_id
                            if not runtime.retire_joint_lease(
                                    lease, gpu_fence_confirmed=True):
                                raise RuntimeError(
                                    "confirmed joint AI lease could not be retired"
                                )
                            recovery_action["joint_lease_retired"] = True
                            recovery_action["ai_completed_before_recovery"] = True
                    elif args.system != "softwall":
                        execute_request(
                            background, trace_queue, request,
                            "after_nrx_before_recovery", args.ai_bound_map,
                            horizon_ns, ai_guard_ns,
                        )
                        recovery_action["ai_completed_before_recovery"] = True
                    else:
                        release_selected(trace_queue, request)
                elif request is not None:
                    release_selected(trace_queue, request)
            recovery_decisions.append(recovery_action)
            current_phase = "mandatory-fallback-dispatch"
            for cell, tx in certificate_order(transactions, radio):
                row = radio[cell]
                row["fallback"] = True
                start_ns = time.perf_counter_ns()
                if fallback_is_early(tx, start_ns):
                    started = runtime.start_fallback_early(tx, start_ns)
                    row["fallback_early"] = started
                    if not started:
                        row["retime_rejected"] = True
                        # Another live credit can prevent the early retime.
                        # Wait for this transaction's current certified slot,
                        # which may differ from the immutable NRx cutoff.
                        wait_until(tx.fallback_reservation.start_ns)
                        start_ns = time.perf_counter_ns()
                        started = runtime.start_fallback(tx, start_ns)
                else:
                    started = runtime.start_fallback(tx, start_ns)
                # Legacy field name: host decision time before the mandatory
                # GPU call, not an observed GPU kernel-start timestamp.
                row["fallback_actual_start_ns"] = start_ns
                row["fallback_started"] = started
                if not started:
                    reservation = tx.fallback_reservation
                    raise RuntimeError(
                        "mandatory fallback failed to start "
                        f"index={index} cell={cell} "
                        f"now_after_release_ms={(start_ns - release_ns) / 1e6:.3f} "
                        f"deadline_ms={(tx.request.deadline_ns - release_ns) / 1e6:.3f} "
                        f"reserved_start_ms="
                        f"{((reservation.start_ns - release_ns) / 1e6) if reservation else None} "
                        f"commit_state={tx.commit_state.name}"
                    )
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
                if row["commit_kind"] == "conventional":
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
            next_release_ns = release_ns + period_ns
            if args.system == "static":
                static_release_ns = release_ns + static_end_offset_ns
                if time.perf_counter_ns() < static_release_ns:
                    wait_until(static_release_ns)
            while background.enabled:
                now_ns = time.perf_counter_ns()
                request = select_request(
                    trace_queue, now_ns, next_release_ns,
                    ai_guard_ns, args.ai_bound_map,
                )
                if request is None:
                    break
                execute_request(
                    background, trace_queue, request, "after_radio",
                    args.ai_bound_map, next_release_ns, ai_guard_ns,
                )
        trace_queue.finalize(first_release_ns + args.iterations * period_ns)
    except Exception as error:
        now_ns = time.perf_counter_ns()
        transaction_state = []
        for cell, tx in enumerate(transactions):
            reservation = tx.fallback_reservation
            transaction_state.append({
                "cell": cell,
                "slot_id": tx.request.slot_id,
                "commit_state": tx.commit_state.name,
                "nrx_observed_ns": tx.nrx_observed_ns,
                "fallback_started_ns": tx.fallback_started_ns,
                "deadline_ns": tx.request.deadline_ns,
                "fallback_latest_start_ns": tx.request.fallback_latest_start_ns,
                "fallback_reservation": None if reservation is None else {
                    "lane": reservation.lane,
                    "start_ns": reservation.start_ns,
                    "predicted_finish_ns": reservation.predicted_finish_ns,
                },
            })
        atomic_json(args.output, {
            "schema": "softwall-four-cell-trace-baseline-failure-v1",
            "system": args.system,
            "trace_sha256": args.trace_sha256,
            "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "failed": True,
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "phase": current_phase,
                "release_index": index,
                "detected_ns": now_ns,
            },
            "transaction_state": transaction_state,
            "current_radio": radio,
            "completed_radio_records": records,
            "release_stage_records": release_stage_records,
            "recovery_decisions": recovery_decisions,
            "background_records": [] if background is None else background.records,
            "background_faults": [] if background is None else background.faults,
            "trace_summary": None if trace_queue is None else trace_queue.summary(),
            "global_queue_status": (
                None if trace_queue is None else global_queue_status(trace_queue)
            ),
            "runtime_metrics": dict(runtime.metrics),
            "fallback_calendar": runtime.fallback_calendar.snapshot(),
            "endpoint_state": {
                name: endpoint.snapshot()
                for name, endpoint in runtime.endpoints.items()
            },
        })
        raise
    finally:
        if background is not None:
            background.close()
        for ctx in endpoint_contexts:
            try:
                ctx["owner"].publish_forward(TERMINATE_SEQ)
            except Exception:
                pass
        time.sleep(0.2)
        for ctx in endpoint_contexts:
            ctx["owner"].close()

    background_records = background.records if background is not None else []
    result = {
        "schema": "softwall-global-trace-home-fault-contained-v1",
        "system": args.system,
        "trace": str(args.trace),
        "trace_sha256": args.trace_sha256,
        "trace_selection": trace["selection"],
        "trace_mapping": trace["mapping"],
        "trace_summary": trace_queue.summary(),
        "global_queue_status": global_queue_status(trace_queue),
        "gc_mode": args.gc_mode,
        "release_stage_records": release_stage_records,
        "gc_events": gc_events,
        "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "iterations": args.iterations, "cells": args.cells, "warmup": args.warmup,
        "first_release_ns": first_release_ns,
        "global_trace_broker": args.global_trace_broker,
        "global_home_id": args.global_home_id,
        "start_barrier": None if args.start_barrier_dir is None else {
            "directory": str(args.start_barrier_dir),
            "member": args.start_barrier_member,
            "participants": args.start_barrier_participants,
        },
        "nrx_endpoints": args.nrx_endpoints,
        "warmup_correct": warmup_correct,
        "conventional_warmup_correct": conventional_warmup_correct,
        "period_ms": args.period_ms, "deadline_ms": args.deadline_ms,
        "nrx_bound_ms": args.nrx_bound_ms, "conv_bound_ms": args.conv_bound_ms,
        "commit_guard_ms": args.commit_guard_ms,
        "endpoint_timeout_ms": args.endpoint_timeout_ms,
        "ai_bound_ms": {
            str(length): bound_ns / 1e6
            for length, bound_ns in sorted(args.ai_bound_map.items())
        },
        "ai_guard_ms": args.ai_guard_ms,
        "ai_rpc_timeout_ms": args.ai_rpc_timeout_ms,
        "payload_seed": args.seed, "snr_db": args.snr_db,
        "gate_mode": args.gate_mode, "gate_threshold": args.gate_threshold,
        "early_mandatory": "on",
        "ai_during_nrx": "async_one_if_edf_fit",
        "ai_aware_recovery": args.system == "softwall",
        "inject_correlated_failure_every": args.inject_correlated_failure_every,
        "fault_pattern": args.fault_pattern,
        "recovery_decisions": recovery_decisions,
        "recovery_retime_count": sum(row["retimed"] for row in recovery_decisions),
        "joint_lease_retired_count": sum(
            row["joint_lease_retired"] for row in recovery_decisions
        ),
        "ai_before_recovery_units": sum(
            row.get("phase") == "after_nrx_before_recovery" for row in background_records
        ),
        "ai_timely_units": sum(row["timely"] for row in background_records),
        "ai_timely_value_tokens": sum(
            row["value_tokens"] for row in background_records if row["timely"]
        ),
        "pre_radio_ai_units": sum(
            row.get("phase") == "before_nrx_observation" for row in background_records
        ),
        "pre_radio_ai_guard_violations": sum(
            row.get("horizon_guard_violation", False) for row in background_records
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
        "background_budget_violations": sum(row["bound_violation"] for row in background_records),
        "background_horizon_violations": sum(
            row.get("horizon_guard_violation", False) for row in background_records
        ),
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
        f"[TRACE/{args.system}] correct={result['correct_cells']}/{len(records)} "
        f"miss={result['deadline_misses']} nrx_bound={result['nrx_bound_violations']} "
        f"fallback={result['fallbacks']} ai={result['background_units']} "
        f"value={result['ai_timely_value_tokens']} "
        f"p99={result['response_ms']['p99']:.3f}ms",
        flush=True,
    )


if __name__ == "__main__":
    main()
