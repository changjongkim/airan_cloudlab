#!/usr/bin/env python3
"""C178 coordinator: the C176 bursty-arrival coordinator plus early AI slots.

C178 adds one policy to the C176 policies below:

- backstop_early: at release, before any NeuralRx outcome, an AI slot is
  committed only when the all-fail schedule of every admitted TB, with the
  slot as an AI busy interval, stays executable (c178_early_plan.py); from the
  NeuralRx deadline on, the period runs exactly as backstop. On one GPU the
  early slot shares the GPU with NeuralRx, and a NeuralRx result that it
  delays only turns into a required recovery that the schedule already holds.

The C176 description follows.

C176 bursty-arrival coordinator for the P180/D155 two-node mode.

The radio path is the C159-Q2 path: pre-staged owners, the persistent
TensorRT NeuralRx worker on GPU3, the shared cuPHY recovery lane on GPU2, and
the Qwen worker on GPU2 behind an absolute latest-start gate. The AI side is
driven by a frozen arrival trace. Requests queue in FIFO order; at each
outcome batch one admission policy chooses the requests that receive a lease.
Every policy sees the same trace, deadlines, class bounds, guard, and visiting
order:

- backstop: an AI lease is committed only when the all-fail certificate of
  the required recoveries still holds with every committed lease as a
  blackout, and the lease ends by its request deadline and by the guard;
- recovery_first: the required recoveries take their bounds first, and a
  lease follows only when it still ends by its deadline and by the guard;
- idle_time: a lease is committed while it ends by its deadline and by the
  guard, ignoring the pending recoveries; the period breaks the contract when
  the leases plus the required recoveries at their bounds end after the guard.

Execution is work-conserving in certificate order: the next action starts when
the previous one returns, which is never later than its certified start. With
--padded, every AI unit and recovery occupies the GPU until its declared bound
(the C172 bound-realization diagnostic).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import cupy as cp

from bounded_launch_revalidation import bounded_revalidate
from c159_q2_classes import CLASS_BOUNDS_MS
from c172_gpu_bound_pad import GpuBoundPad
from actual_nrx_integrated_coordinator import (
    load_spec,
    wait_json,
)
from actual_nrx_repeated_control import (
    ACTION_RECOVERY,
    ACTION_SUCCESS,
    DecisionPeer,
)
from dual_receiver_phy import PairedDualReceiver
from integrated_shared_recovery_plan_v1 import IntegratedScenarioConfig
from integrated_shared_recovery_launch_plan_v1 import LAUNCH_CONTROL_BOUND_MS
from integrated_shared_recovery_worker import connect_qwen, wait_until_ns
from isca_v2.cuda_ipc_channel import CudaIpcPeer, TERMINATE_SEQ
from multigpu_p2p_ipc_gate import (
    P2PCopier,
    close_peer_before_ack,
    enable_peer_access,
    summarize,
)
from shared_conventional_worker import (
    RX_ELEMENTS,
    atomic_json,
    install_window,
    run_conventional,
    wait_sequence,
)
from c176_burst_plan import POLICIES as C176_POLICIES, lease_ns, plan_period
from c178_early_plan import mandatory_infeasible_plan, plan_early


MS = 1_000_000
POLICIES = C176_POLICIES + ("backstop_early",)
CERTIFIED = ("backstop", "backstop_early")


def qwen_request_with_guard(
    channel, request_id: str, context_length: int, latest_start_ns: int
) -> dict:
    sent_ns = time.perf_counter_ns()
    channel.write(json.dumps({
        "op": "run",
        "request_id": request_id,
        "context_length": context_length,
        "latest_start_ns": latest_start_ns,
    }).encode() + b"\n")
    raw = channel.readline()
    returned_ns = time.perf_counter_ns()
    if not raw:
        raise ConnectionError("Qwen worker closed before response")
    response = json.loads(raw)
    if not response.get("ok"):
        raise RuntimeError(f"Qwen worker rejected request: {response}")
    if response.get("launched") is False:
        return {
            "request_id": request_id,
            "context_length": context_length,
            "sent_ns": sent_ns,
            "returned_ns": returned_ns,
            "launched": False,
            "latest_start_ns": latest_start_ns,
            "worker_accepted_ns": int(response["accepted_ns"]),
            "reason": response["reason"],
            "fence_confirmed": True,
        }
    return {
        "request_id": request_id,
        "context_length": context_length,
        "sent_ns": sent_ns,
        "returned_ns": returned_ns,
        "launched": True,
        "latest_start_ns": latest_start_ns,
        "execution_ms": (returned_ns - sent_ns) / 1e6,
        "gpu_ms": float(response["gpu_ms"]),
        "worker_completed_ns": int(response["completed_ns"]),
        "worker_accepted_ns": int(response["accepted_ns"]),
        "fence_confirmed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--peer-spec", type=Path, required=True)
    parser.add_argument("--ipc-dir", type=Path, required=True)
    parser.add_argument("--schedule-file", type=Path, required=True)
    parser.add_argument("--qwen-socket", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--ai-trace", type=Path, required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--slo-ms", type=float, required=True)
    parser.add_argument("--max-leases", type=int, default=4)
    parser.add_argument("--padded", action="store_true")
    parser.add_argument("--ai-padding-margin-ms", type=float, default=0.50)
    parser.add_argument("--recovery-padding-margin-ms", type=float, default=0.50)
    parser.add_argument("--period-ms", type=float, default=180.0)
    parser.add_argument("--destination-device", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--ready-timeout-s", type=float, default=300.0)
    parser.add_argument("--unit-timeout-s", type=float, default=5.0)
    parser.add_argument("--release-lead-ms", type=float, default=3000.0)
    args = parser.parse_args()
    if args.iterations <= 0 or args.period_ms <= 0 or args.slo_ms <= 0 or args.max_leases <= 0:
        parser.error("iterations, period, SLO, and max leases must be positive")
    trace = json.loads(args.ai_trace.read_text())
    trace_requests = sorted(trace["requests"], key=lambda row: (row["arrival_ms"], row["request_id"]))
    for row in trace_requests:
        if row["context_length"] not in CLASS_BOUNDS_MS:
            raise ValueError(f"unknown prompt class {row['context_length']}")
    specs = load_spec(args.peer_spec)
    config = IntegratedScenarioConfig(period_ms=round(args.period_ms))
    config.validate()

    peers = []
    for row in specs:
        source = int(row["source_device"])
        cp.cuda.runtime.setDevice(source)
        peers.append(CudaIpcPeer(
            row["recovery_tag"], args.ready_timeout_s, directory=args.ipc_dir
        ))
    decisions = [
        DecisionPeer(Path(row["decision_control"]), args.ready_timeout_s)
        for row in specs
    ]
    access = {}
    for source in sorted({int(row["source_device"]) for row in specs}):
        access.update(enable_peer_access(source, args.destination_device))

    contexts = []
    recovery_padder = ai_padder = None
    if args.padded:
        recovery_padder = GpuBoundPad(device=args.destination_device, margin_ms=args.recovery_padding_margin_ms)
        ai_padder = GpuBoundPad(device=args.destination_device, margin_ms=args.ai_padding_margin_ms)
    with cp.cuda.Device(args.destination_device):
        for row in specs:
            receiver = PairedDualReceiver(
                args.engine,
                seed=int(row["receiver_seed"]),
                device=args.destination_device,
                enable_local_neural=False,
            )
            remote_forward = cp.empty(2 * RX_ELEMENTS, dtype=cp.float32)
            remote_backward = cp.empty(
                len(receiver.reference_tb) + 1, dtype=cp.uint8
            )
            verify_backward = cp.empty(
                len(receiver.reference_tb) + 1, dtype=cp.uint8
            )
            for _ in range(args.warmup):
                if not receiver.run_conventional()[1]:
                    raise RuntimeError(f"recovery warmup failed {row['key']}")
            contexts.append({
                "receiver": receiver,
                "remote_forward": remote_forward,
                "remote_backward": remote_backward,
                "verify_backward": verify_backward,
                "forward": P2PCopier(
                    int(row["source_device"]), args.destination_device
                ),
                "backward": P2PCopier(
                    args.destination_device, int(row["source_device"])
                ),
            })

    # Exercise the entire recovery path before readiness, as in C159-Q2.
    preflight_echo = []
    for peer, context in zip(peers, contexts):
        context["forward"].copy(context["remote_forward"], peer.forward.view(cp.float32))
        context["backward"].copy(peer.forward.view(cp.float32), context["remote_forward"])
        install_window(context["receiver"], context["remote_forward"])
        decoded = run_conventional(context["receiver"], context["remote_backward"])
        with cp.cuda.Device(args.destination_device):
            context["remote_backward"][0] = cp.uint8(int(decoded["crc_failures"] != 0))
            cp.cuda.get_current_stream().synchronize()
        context["backward"].copy(peer.backward.view(cp.uint8), context["remote_backward"])
        context["forward"].copy(context["verify_backward"], peer.backward.view(cp.uint8))
        with cp.cuda.Device(args.destination_device):
            preflight_echo.append(bool(cp.array_equal(
                context["verify_backward"], context["remote_backward"]).item()))
    if not all(preflight_echo):
        raise RuntimeError("full recovery path preflight failed")

    def run_recovery(key, sequence: int, release_ns: int, by_key) -> dict:
        row, peer, context = by_key[key]
        wait_sequence(peer, sequence, args.unit_timeout_s)
        started_ns = time.perf_counter_ns()
        context["forward"].copy(context["remote_forward"], peer.forward.view(cp.float32))
        context["backward"].copy(peer.forward.view(cp.float32), context["remote_forward"])
        install_window(context["receiver"], context["remote_forward"])
        decoded = run_conventional(context["receiver"], context["remote_backward"])
        with cp.cuda.Device(args.destination_device):
            context["remote_backward"][0] = cp.uint8(int(decoded["crc_failures"] != 0))
            cp.cuda.get_current_stream().synchronize()
        context["backward"].copy(peer.backward.view(cp.uint8), context["remote_backward"])
        context["forward"].copy(context["verify_backward"], peer.backward.view(cp.uint8))
        with cp.cuda.Device(args.destination_device):
            echo_equal = bool(cp.array_equal(context["verify_backward"], context["remote_backward"]).item())
        if not echo_equal:
            raise RuntimeError(f"recovery response echo mismatch sequence={sequence} key={key}")
        service_ns = time.perf_counter_ns()
        padding = None
        if recovery_padder is not None:
            padding = recovery_padder.pad_elapsed(started_ns, config.conventional_bound_ms)
        completed_ns = time.perf_counter_ns()
        peer.publish_backward(sequence)
        return {
            "sequence": sequence,
            "key": list(key),
            "actual_start_ns": started_ns,
            "actual_completed_ns": completed_ns,
            "release_to_start_ms": (started_ns - release_ns) / 1e6,
            "release_to_complete_ms": (completed_ns - release_ns) / 1e6,
            "prepad_path_ms": (service_ns - started_ns) / 1e6,
            "conventional_gpu_ms": decoded["gpu_ms"],
            "padding": padding,
        }

    qwen_socket = None
    qwen_channel = None
    handles_closed = False
    rounds = []
    recoveries = []
    request_state = {}
    result = {
        "schema": "softwall-c178-early-coordinator-v1",
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "iterations": args.iterations,
        "period_ms": args.period_ms,
        "policy": args.policy,
        "slo_ms": args.slo_ms,
        "max_leases": args.max_leases,
        "padded": args.padded,
        "ai_trace": {"path": str(args.ai_trace), "summary": trace.get("summary"),
                     "requests": len(trace_requests)},
        "class_bounds_ms": CLASS_BOUNDS_MS,
        "launch_control_bound_ms": LAUNCH_CONTROL_BOUND_MS,
        "peer_specs": specs,
        "peer_access": access,
        "preflight_echo_equal": preflight_echo,
        "ipc_handles_closed_before_ack": False,
        "qwen_stop_acknowledged": False,
        "error": None,
    }
    try:
        qwen_socket, qwen_channel = connect_qwen(args.qwen_socket, args.ready_timeout_s)
        for peer in peers:
            peer.mark_ready()
        for row in specs:
            ready = wait_json(Path(row["ready_file"]), args.ready_timeout_s)
            if ready.get("key") != row["key"] or not ready.get("ready"):
                raise RuntimeError(f"invalid owner readiness {row['key']}")

        first_release_ns = time.perf_counter_ns() + round(args.release_lead_ms * 1e6)
        period_ns = round(args.period_ms * 1e6)
        atomic_json(args.schedule_file, {
            "schema": "softwall-c178-schedule-v1",
            "first_release_wall_ns": first_release_ns,
            "period_ns": period_ns,
            "iterations": args.iterations,
            "release_semantics": "synthetic t_IQ_ready after pre-staged bank copy",
        })
        by_key = {tuple(row["key"]): (row, peer, context) for row, peer, context in zip(specs, peers, contexts)}
        slo_ns = round(args.slo_ms * MS)
        next_request = 0
        queue = []

        def admit_arrivals(wall_ns: int) -> None:
            nonlocal next_request
            while (next_request < len(trace_requests)
                   and first_release_ns + round(trace_requests[next_request]["arrival_ms"] * MS) <= wall_ns):
                row = trace_requests[next_request]
                arrival_ns = first_release_ns + round(row["arrival_ms"] * MS)
                request_state[row["request_id"]] = {
                    "request_id": row["request_id"], "context_length": row["context_length"],
                    "value_tokens": row["value_tokens"], "arrival_ns": arrival_ns,
                    "deadline_ns": arrival_ns + slo_ns, "status": "queued",
                }
                queue.append(request_state[row["request_id"]])
                next_request += 1

        def run_slots(leases, certificate, release_ns: int, sequence: int, early: bool) -> list:
            attempts_out = []
            for lease in leases:
                request = request_state[lease["request_id"]]
                latest_start_ns = release_ns + lease["start_ns"] + LAUNCH_CONTROL_BOUND_MS * MS
                attempt = qwen_request_with_guard(qwen_channel, lease["request_id"],
                                                  lease["context_length"], latest_start_ns)
                attempt["certified_start_ms"] = lease["start_ns"] / 1e6
                attempt["certified_finish_ms"] = lease["finish_ns"] / 1e6
                attempt["early"] = early
                if attempt["launched"] and ai_padder is not None:
                    attempt["padding"] = ai_padder.pad_elapsed(
                        attempt["worker_accepted_ns"], CLASS_BOUNDS_MS[lease["context_length"]])
                    attempt["returned_ns"] = time.perf_counter_ns()
                    attempt["worker_completed_ns"] = attempt["returned_ns"]
                attempt["release_to_return_ms"] = (attempt["returned_ns"] - release_ns) / 1e6
                request["status"] = "served" if attempt["launched"] else "refused"
                request["served_sequence"] = sequence
                request["early"] = early
                if attempt["launched"]:
                    request["completion_ns"] = attempt["worker_completed_ns"]
                    request["ttft_ms"] = (attempt["worker_completed_ns"] - request["arrival_ns"]) / 1e6
                    request["on_time"] = attempt["worker_completed_ns"] <= request["deadline_ns"]
                if args.policy in CERTIFIED:
                    retire = certificate.retire_lease(
                        lease["request_id"], gpu_fence_confirmed=attempt["fence_confirmed"],
                        expected_generation=certificate.generation,
                        now_ns=max(0, attempt["returned_ns"] - release_ns))
                    if not retire.accepted:
                        raise RuntimeError(f"lease retire failed sequence={sequence}: {retire.reason}")
                attempts_out.append(attempt)
            return attempts_out

        for index in range(args.iterations):
            sequence = index + 1
            release_ns = first_release_ns + index * period_ns
            early_plan, early_attempts, early_decision_ms = None, [], None
            if args.policy == "backstop_early":
                wait_until_ns(release_ns)
                admit_arrivals(time.perf_counter_ns())
                visible_early = [{"request_id": request["request_id"], "context_length": request["context_length"],
                                  "deadline_rel_ns": request["deadline_ns"] - release_ns} for request in queue]
                early_plan, _, _, early_revalidation = bounded_revalidate(
                    lambda launch_now_ns: plan_early(launch_now_ns, visible_early, config, args.max_leases),
                    release_ns=release_ns,
                    lower_bound_ns=0,
                    budget_ms=LAUNCH_CONTROL_BOUND_MS,
                )
                early_decision_ms = early_revalidation[-1]["launch_now_ns"] / 1e6
                early_ids = {lease["request_id"] for lease in early_plan["leases"]}
                queue = [request for request in queue if request["request_id"] not in early_ids]
                early_attempts = run_slots(early_plan["leases"], early_plan["coordinator"], release_ns,
                                           sequence, early=True)
            wait_until_ns(release_ns + config.recovery_release_ns)
            observed, missing, timely_successes = [], [], []
            for row, decision in zip(specs, decisions):
                value = decision.read_outcome(sequence)
                if value is None:
                    missing.append(row["key"])
                    continue
                value["key"] = row["key"]
                value["release_to_complete_ms"] = (value["completed_ns"] - release_ns) / 1e6
                observed.append(value)
                if value.get("timely_success") is True:
                    timely_successes.append(tuple(row["key"]))

            # Requests that have arrived join the FIFO queue; requests that can
            # no longer meet their deadline leave it.
            decision_wall_ns = time.perf_counter_ns()
            admit_arrivals(decision_wall_ns)
            now_rel_ns = max(config.recovery_release_ns, decision_wall_ns - release_ns)
            kept = []
            for request in queue:
                fastest = release_ns + max(config.ai_start_ns, now_rel_ns) + lease_ns(request["context_length"])
                if fastest > request["deadline_ns"]:
                    request["status"] = "expired"
                    request["expired_sequence"] = sequence
                else:
                    kept.append(request)
            queue = kept
            visible = [{"request_id": request["request_id"], "context_length": request["context_length"],
                        "deadline_rel_ns": request["deadline_ns"] - release_ns} for request in queue]

            try:
                plan, revalidation_started_ns, revalidation_completed_ns, attempts = bounded_revalidate(
                    lambda launch_now_ns: plan_period("backstop" if args.policy == "backstop_early" else args.policy,
                                                      timely_successes, launch_now_ns, visible,
                                                      config, args.max_leases),
                    release_ns=release_ns,
                    lower_bound_ns=config.recovery_release_ns,
                    budget_ms=LAUNCH_CONTROL_BOUND_MS,
                )
            except RuntimeError as error:
                if "certificate infeasible at current time" not in str(error):
                    raise
                revalidation_started_ns = revalidation_completed_ns = time.perf_counter_ns()
                late_ns = max(config.recovery_release_ns, revalidation_started_ns - release_ns)
                plan = mandatory_infeasible_plan(timely_successes, late_ns, config)
                attempts = [{"launch_now_ns": late_ns}]
            coordinator = plan["coordinator"]
            success_set = set(map(tuple, plan["success_keys"]))
            recovery_set = set(map(tuple, plan["unresolved_keys"]))
            for row, decision in zip(specs, decisions):
                key = tuple(row["key"])
                if key in success_set:
                    action = ACTION_SUCCESS
                elif key in recovery_set:
                    action = ACTION_RECOVERY
                else:
                    raise RuntimeError(f"accepted request has no action: {key}")
                decision.publish_dispatch(sequence, action, coordinator.generation)

            admitted_ids = {lease["request_id"] for lease in plan["leases"]}
            queue = [request for request in queue if request["request_id"] not in admitted_ids]

            def run_leases() -> list:
                return run_slots(plan["leases"], coordinator, release_ns, sequence, early=False)

            def run_recoveries() -> list:
                if args.policy in CERTIFIED and not plan.get("mandatory_infeasible"):
                    placements = sorted(coordinator.snapshot()["placements"], key=lambda row: (
                        row["start_ns"], row["lane"], row["home_id"], row["request_id"]))
                    keys = [(row["home_id"], row["request_id"]) for row in placements]
                else:
                    keys = [tuple(key) for key in plan["unresolved_keys"]]
                if sorted(keys) != sorted(tuple(key) for key in plan["unresolved_keys"]):
                    raise RuntimeError("recovery order does not match the unresolved set")
                done = []
                for key in keys:
                    record = run_recovery(key, sequence, release_ns, by_key)
                    recoveries.append(record)
                    done.append(record)
                return done

            if args.policy == "recovery_first":
                round_recoveries = run_recoveries()
                lease_attempts = run_leases()
            else:
                lease_attempts = run_leases()
                round_recoveries = run_recoveries()
            last_recovery_ms = max((row["release_to_complete_ms"] for row in round_recoveries), default=None)
            rounds.append({
                "sequence": sequence,
                "release_wall_ns": release_ns,
                "missing_outcomes_at_cutoff": missing,
                "observed_outcomes": observed,
                "success_keys": [list(key) for key in plan["success_keys"]],
                "unresolved_keys": [list(key) for key in plan["unresolved_keys"]],
                "rejected_keys": [list(key) for key in plan["rejected_keys"]],
                "decision_ms": attempts[-1]["launch_now_ns"] / 1e6,
                "revalidation_host_ms": (revalidation_completed_ns - revalidation_started_ns) / 1e6,
                "queue_visible": len(visible),
                "leases": plan["leases"],
                "lease_attempts": lease_attempts,
                "early_decision_ms": early_decision_ms,
                "early_leases": early_plan["leases"] if early_plan else [],
                "early_attempts": early_attempts,
                "early_all_fail_end_ms": early_plan["all_fail_end_ns"] / 1e6 if early_plan else None,
                "bound_end_ms": plan["bound_end_ns"] / 1e6,
                "contract_broken": plan["contract_broken"],
                "mandatory_infeasible": bool(plan.get("mandatory_infeasible")),
                "physical_recoveries": [row["key"] for row in round_recoveries],
                "last_recovery_complete_ms": last_recovery_ms,
                "physical_guard_miss": (last_recovery_ms is not None
                                        and last_recovery_ms > config.recovery_deadline_ns / 1e6),
            })

        for request in queue:
            request["status"] = "unserved_at_end"
        for decision in decisions:
            decision.close()
        for peer in peers:
            wait_sequence(peer, TERMINATE_SEQ, args.ready_timeout_s)
        for row, peer in zip(specs, peers):
            cp.cuda.runtime.setDevice(int(row["source_device"]))
            close_peer_before_ack(peer)
        handles_closed = True
        result["ipc_handles_closed_before_ack"] = True
        qwen_channel.write(b'{"op":"stop"}\n')
        stopped = json.loads(qwen_channel.readline())
        result["qwen_stop_acknowledged"] = bool(stopped.get("stopped"))
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        served = [row for row in request_state.values() if row["status"] == "served"]
        result.update({
            "completed_rounds": len(rounds),
            "requests_arrived": len(request_state),
            "requests_not_arrived": len(trace_requests) - len(request_state),
            "requests_served": len(served),
            "requests_served_early": sum(1 for row in served if row.get("early")),
            "requests_on_time": sum(1 for row in served if row.get("on_time")),
            "on_time_value_tokens": sum(row["value_tokens"] for row in served if row.get("on_time")),
            "contract_broken_periods": sum(row["contract_broken"] for row in rounds),
            "mandatory_infeasible_periods": sum(row["mandatory_infeasible"] for row in rounds),
            "physical_guard_misses": sum(row["physical_guard_miss"] for row in rounds),
            "ttft_ms": summarize([row["ttft_ms"] for row in served]),
            "physical_recovery_count": len(recoveries),
            "requests": sorted(request_state.values(), key=lambda row: row["arrival_ns"]),
            "rounds": rounds,
            "physical_recoveries": recoveries,
            "completed_ns": time.perf_counter_ns(),
        })
        atomic_json(args.output, result)
        if qwen_channel is not None:
            try:
                qwen_channel.close()
            except OSError:
                pass
        if qwen_socket is not None:
            qwen_socket.close()
        if not handles_closed:
            for decision in decisions:
                try:
                    decision.close()
                except BaseException:
                    pass
            for peer in peers:
                try:
                    peer.close()
                except BaseException:
                    pass


if __name__ == "__main__":
    main()
