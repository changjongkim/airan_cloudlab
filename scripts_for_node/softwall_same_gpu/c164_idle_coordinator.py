#!/usr/bin/env python3
"""C162 boundary coordinator with a measured pre-schedule lifecycle idle."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import time
from pathlib import Path

import cupy as cp

from bounded_launch_revalidation import bounded_revalidate
from c159_q2_classes import CLASS_BOUNDS_MS
from c162_boundary_cases import CASES, case_for_sequence
from c162_boundary_plan import build_boundary_plan
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
from integrated_shared_recovery_plan_v1 import REQUEST_ORDER, IntegratedScenarioConfig
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
    parser.add_argument("--period-ms", type=float, default=600.0)
    parser.add_argument("--destination-device", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--ready-timeout-s", type=float, default=300.0)
    parser.add_argument("--unit-timeout-s", type=float, default=5.0)
    parser.add_argument("--release-lead-ms", type=float, default=3000.0)
    parser.add_argument("--pre-schedule-idle-s", type=float, required=True)
    parser.add_argument("--lifecycle-marker", type=Path, required=True)
    parser.add_argument("--reverse-cases", action="store_true")
    args = parser.parse_args()
    if (args.iterations <= 0 or args.period_ms <= 0
            or args.pre_schedule_idle_s <= 0):
        parser.error("iterations and period must be positive")
    specs = load_spec(args.peer_spec)
    base_config = IntegratedScenarioConfig(period_ms=round(args.period_ms))
    base_config.validate()

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

    # Exercise the entire recovery path before readiness. Merely warming the
    # receiver does not compile CuPy's window-install kernels or initialize all
    # P2P/event paths. The owner has not published a request, so these temporary
    # bytes are not consumable radio state. This explicitly defines the
    # qualified lifecycle as warm persistent operation.
    preflight_echo = []
    recovery_preflight = []
    for peer, context in zip(peers, contexts):
        started_ns = time.perf_counter_ns()
        forward = context["forward"].copy(
            context["remote_forward"], peer.forward.view(cp.float32)
        )
        input_echo = context["backward"].copy(
            peer.forward.view(cp.float32), context["remote_forward"]
        )
        install_started_ns = time.perf_counter_ns()
        install_window(context["receiver"], context["remote_forward"])
        install_completed_ns = time.perf_counter_ns()
        decoded = run_conventional(
            context["receiver"], context["remote_backward"]
        )
        normalize_started_ns = time.perf_counter_ns()
        with cp.cuda.Device(args.destination_device):
            context["remote_backward"][0] = cp.uint8(
                int(decoded["crc_failures"] != 0)
            )
            cp.cuda.get_current_stream().synchronize()
        normalize_completed_ns = time.perf_counter_ns()
        backward = context["backward"].copy(
            peer.backward.view(cp.uint8), context["remote_backward"]
        )
        echo = context["forward"].copy(
            context["verify_backward"], peer.backward.view(cp.uint8)
        )
        with cp.cuda.Device(args.destination_device):
            echo_equal = bool(cp.array_equal(
                context["verify_backward"], context["remote_backward"]
            ).item())
        completed_ns = time.perf_counter_ns()
        preflight_echo.append(echo_equal)
        recovery_preflight.append({
            "path_ms": (completed_ns - started_ns) / 1e6,
            "forward_host_us": forward["host_us"],
            "input_echo_host_us": input_echo["host_us"],
            "install_window_host_ms": (
                install_completed_ns - install_started_ns
            ) / 1e6,
            "conventional_host_ms": decoded["host_ms"],
            "normalize_host_us": (
                normalize_completed_ns - normalize_started_ns
            ) / 1e3,
            "backward_host_us": backward["host_us"],
            "backward_echo_host_us": echo["host_us"],
            "backward_echo_equal": echo_equal,
        })
    if not all(preflight_echo):
        raise RuntimeError("full recovery path preflight failed")

    qwen_socket = None
    qwen_channel = None
    handles_closed = False
    rounds = []
    recoveries = []
    result = {
        "schema": "softwall-c162-boundary-coordinator-v1",
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "iterations": args.iterations,
        "period_ms": args.period_ms,
        "boundary_cases": [dataclasses.asdict(case) for case in CASES],
        "class_bounds_ms": CLASS_BOUNDS_MS,
        "peer_specs": specs,
        "peer_access": access,
        "preflight_echo_equal": preflight_echo,
        "recovery_preflight": recovery_preflight,
        "lifecycle": "idle_30s_first",
        "required_pre_schedule_idle_s": args.pre_schedule_idle_s,
        "ipc_handles_closed_before_ack": False,
        "qwen_stop_acknowledged": False,
        "error": None,
    }
    try:
        qwen_socket, qwen_channel = connect_qwen(
            args.qwen_socket, args.ready_timeout_s
        )
        for peer in peers:
            peer.mark_ready()
        for row in specs:
            ready = wait_json(Path(row["ready_file"]), args.ready_timeout_s)
            if ready.get("key") != row["key"] or not ready.get("ready"):
                raise RuntimeError(f"invalid owner readiness {row['key']}")

        idle_started_ns = time.perf_counter_ns()
        atomic_json(args.lifecycle_marker, {
            "schema": "softwall-c164-lifecycle-idle-marker-v1",
            "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "clock": "time.perf_counter_ns",
            "status": "idle_started_after_all_readiness",
            "idle_started_ns": idle_started_ns,
            "required_idle_ns": round(args.pre_schedule_idle_s * 1e9),
        })
        time.sleep(args.pre_schedule_idle_s)
        idle_completed_ns = time.perf_counter_ns()
        observed_idle_ns = idle_completed_ns - idle_started_ns
        if observed_idle_ns < round(args.pre_schedule_idle_s * 1e9):
            raise RuntimeError("pre-schedule lifecycle idle was shorter than required")
        first_release_ns = time.perf_counter_ns() + round(args.release_lead_ms * 1e6)
        period_ns = round(args.period_ms * 1e6)
        atomic_json(args.lifecycle_marker, {
            "schema": "softwall-c164-lifecycle-idle-marker-v1",
            "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "clock": "time.perf_counter_ns",
            "status": "idle_complete_before_schedule_publish",
            "idle_started_ns": idle_started_ns,
            "idle_completed_ns": idle_completed_ns,
            "observed_idle_ns": observed_idle_ns,
            "required_idle_ns": round(args.pre_schedule_idle_s * 1e9),
            "idle_gate": observed_idle_ns >= round(args.pre_schedule_idle_s * 1e9),
            "first_release_wall_ns": first_release_ns,
        })
        atomic_json(args.schedule_file, {
            "schema": "softwall-c162-boundary-schedule-v1",
            "first_release_wall_ns": first_release_ns,
            "period_ns": period_ns,
            "iterations": args.iterations,
            "release_semantics": "synthetic t_IQ_ready after pre-staged bank copy",
            "staging_semantics": (
                "all channel inputs and validation-only local oracles are built "
                "before readiness; only bank-to-live-buffer copy occurs after "
                "the preceding 155 ms expiry"
            ),
            "lifecycle_semantics": (
                "all persistent endpoints and owners were ready, then remained "
                "idle before this schedule was published"
            ),
        })
        by_key = {
            tuple(row["key"]): (row, peer, context)
            for row, peer, context in zip(specs, peers, contexts)
        }

        for index in range(args.iterations):
            sequence = index + 1
            boundary = case_for_sequence(sequence, args.reverse_cases)
            context_length = boundary.context_length
            config = IntegratedScenarioConfig(
                period_ms=round(args.period_ms),
                ai_bound_ms=CLASS_BOUNDS_MS.get(context_length, 35),
            )
            config.validate()
            release_ns = first_release_ns + index * period_ns
            cutoff_ns = release_ns + round(boundary.target_decision_ms * 1e6)
            wait_until_ns(cutoff_ns)
            observed = []
            missing = []
            physical_timely_successes = []
            for row, decision in zip(specs, decisions):
                value = decision.read_outcome(sequence)
                if value is None:
                    missing.append(row["key"])
                    continue
                value["key"] = row["key"]
                value["release_to_complete_ms"] = (
                    value["completed_ns"] - release_ns
                ) / 1e6
                observed.append(value)
                if value.get("timely_success") is True:
                    physical_timely_successes.append(tuple(row["key"]))

            # All four NRx paths still execute physically.  Stable +20 dB
            # inputs must succeed, then the frozen boundary case converts a
            # suffix to failures before the atomic transition.  This isolates
            # debt count from random channel outcomes.
            accepted_keys = tuple(REQUEST_ORDER[:4])
            if set(physical_timely_successes) != set(accepted_keys):
                raise RuntimeError(
                    f"boundary physical NRx was not stable sequence={sequence}: "
                    f"{physical_timely_successes}"
                )
            injected_successes = accepted_keys[:boundary.success_count]

            (
                plan, revalidation_started_ns, revalidation_completed_ns,
                revalidation_attempts,
            ) = bounded_revalidate(
                lambda launch_now_ns: build_boundary_plan(
                    injected_successes, launch_now_ns, config,
                    offer_ai=context_length is not None,
                ),
                release_ns=release_ns,
                lower_bound_ns=config.recovery_release_ns,
                budget_ms=LAUNCH_CONTROL_BOUND_MS,
            )
            coordinator = plan["coordinator"]
            dispatch = {
                "schema": "softwall-c162-boundary-dispatch-v1",
                "sequence": sequence,
                "success_keys": [list(key) for key in plan["success_keys"]],
                "physical_recovery_keys": [
                    list(key) for key in plan["physical_recovery_keys"]
                ],
                "rejected_keys": [list(key) for key in plan["rejected_keys"]],
                "generation": coordinator.generation,
            }
            success_set = set(plan["success_keys"])
            recovery_set = set(plan["physical_recovery_keys"])
            for row, decision in zip(specs, decisions):
                key = tuple(row["key"])
                if key in success_set:
                    action = ACTION_SUCCESS
                elif key in recovery_set:
                    action = ACTION_RECOVERY
                else:
                    raise RuntimeError(f"accepted request has no action: {key}")
                decision.publish_dispatch(
                    sequence, action, coordinator.generation
                )

            qwen = None
            qwen_attempt = None
            lease_retire = None
            if plan["ai_expected"]:
                lease = plan["lease_interval"]
                latest_start_ns = (
                    release_ns + lease["start_ns"]
                    + round(plan["launch_control_bound_ms"] * 1e6)
                )
                qwen_attempt = qwen_request_with_guard(
                    qwen_channel,
                    f"{plan['lease_id']}-c{context_length}-{sequence:06d}",
                    context_length,
                    latest_start_ns,
                )
                qwen_attempt["release_to_send_ms"] = (
                    qwen_attempt["sent_ns"] - release_ns
                ) / 1e6
                qwen_attempt["release_to_return_ms"] = (
                    qwen_attempt["returned_ns"] - release_ns
                ) / 1e6
                qwen_attempt["dispatch_after_revalidation_ms"] = (
                    qwen_attempt["sent_ns"] - revalidation_started_ns
                ) / 1e6
                qwen_attempt["launch_control_bound_violation"] = (
                    qwen_attempt["launched"]
                    and qwen_attempt["dispatch_after_revalidation_ms"]
                    > plan["launch_control_bound_ms"]
                )
                if qwen_attempt["launched"]:
                    qwen_attempt["bound_violation"] = (
                        qwen_attempt["execution_ms"] > config.ai_bound_ms
                    )
                    qwen_attempt["lease_interval_violation"] = not (
                        lease["start_ns"]
                            <= qwen_attempt["worker_accepted_ns"] - release_ns
                        and qwen_attempt["returned_ns"] - release_ns
                            <= lease["finish_ns"]
                    )
                    qwen = qwen_attempt
                else:
                    qwen_attempt["bound_violation"] = False
                    qwen_attempt["lease_interval_violation"] = False
                retire_now_ns = max(
                    0, qwen_attempt["returned_ns"] - release_ns
                )
                retire = coordinator.retire_lease(
                    plan["lease_id"],
                    gpu_fence_confirmed=qwen_attempt["fence_confirmed"],
                    expected_generation=coordinator.generation,
                    now_ns=retire_now_ns,
                )
                lease_retire = {
                    "accepted": retire.accepted,
                    "reason": retire.reason,
                    "generation": retire.generation,
                    "now_ns": retire_now_ns,
                }
                if not retire.accepted:
                    raise RuntimeError(
                        f"lease retire failed sequence={sequence}: {retire.reason}"
                    )

            final_snapshot = coordinator.snapshot()
            ordered = sorted(final_snapshot["placements"], key=lambda row: (
                row["start_ns"], row["lane"], row["home_id"], row["request_id"]
            ))
            round_recoveries = []
            for placement in ordered:
                key = (placement["home_id"], placement["request_id"])
                row, peer, context = by_key[key]
                wait_sequence(peer, sequence, args.unit_timeout_s)
                wait_until_ns(release_ns + placement["start_ns"])
                started_ns = time.perf_counter_ns()
                forward = context["forward"].copy(
                    context["remote_forward"], peer.forward.view(cp.float32)
                )
                input_echo = context["backward"].copy(
                    peer.forward.view(cp.float32), context["remote_forward"]
                )
                install_started_ns = time.perf_counter_ns()
                install_window(context["receiver"], context["remote_forward"])
                install_completed_ns = time.perf_counter_ns()
                decoded = run_conventional(
                    context["receiver"], context["remote_backward"]
                )
                # The pyAerial CRC result buffer can expose a non-normalized
                # byte in heavily corrupted cases. Serialize the boolean
                # decision already consumed by _verify as the wire status so
                # the response contract is unambiguously 0=pass, 1=fail.
                normalize_started_ns = time.perf_counter_ns()
                with cp.cuda.Device(args.destination_device):
                    raw_response_crc = int(
                        context["remote_backward"][0].item()
                    )
                    serialized_crc_status = int(decoded["crc_failures"] != 0)
                    context["remote_backward"][0] = cp.uint8(
                        serialized_crc_status
                    )
                    cp.cuda.get_current_stream().synchronize()
                normalize_completed_ns = time.perf_counter_ns()
                backward = context["backward"].copy(
                    peer.backward.view(cp.uint8), context["remote_backward"]
                )
                echo = context["forward"].copy(
                    context["verify_backward"], peer.backward.view(cp.uint8)
                )
                with cp.cuda.Device(args.destination_device):
                    backward_echo_equal = bool(cp.array_equal(
                        context["verify_backward"], context["remote_backward"]
                    ).item())
                if not backward_echo_equal:
                    raise RuntimeError(
                        f"recovery response echo mismatch sequence={sequence} "
                        f"key={key}"
                    )
                completed_ns = time.perf_counter_ns()
                peer.publish_backward(sequence)
                record = {
                    "sequence": sequence,
                    "key": list(key),
                    "model_lane": placement["lane"],
                    "model_start_ns": placement["start_ns"],
                    "model_finish_ns": placement["finish_ns"],
                    "actual_start_ns": started_ns,
                    "actual_completed_ns": completed_ns,
                    "release_to_start_ms": (started_ns - release_ns) / 1e6,
                    "release_to_complete_ms": (completed_ns - release_ns) / 1e6,
                    "forward_gpu_us": forward["gpu_us"],
                    "forward_host_us": forward["host_us"],
                    "input_echo_gpu_us": input_echo["gpu_us"],
                    "input_echo_host_us": input_echo["host_us"],
                    "install_window_host_ms": (
                        install_completed_ns - install_started_ns
                    ) / 1e6,
                    "conventional_gpu_ms": decoded["gpu_ms"],
                    "conventional_host_ms": decoded["host_ms"],
                    "normalize_host_us": (
                        normalize_completed_ns - normalize_started_ns
                    ) / 1e3,
                    "conventional_reference_correct": decoded["correct"],
                    "crc_failures": decoded["crc_failures"],
                    "payload_mismatches": decoded["payload_mismatches"],
                    "raw_response_crc": raw_response_crc,
                    "serialized_crc_status": serialized_crc_status,
                    "backward_gpu_us": backward["gpu_us"],
                    "backward_host_us": backward["host_us"],
                    "backward_echo_gpu_us": echo["gpu_us"],
                    "backward_echo_host_us": echo["host_us"],
                    "backward_echo_equal": backward_echo_equal,
                    "declared_path_bound_violation": (
                        (completed_ns - started_ns) / 1e6
                        > config.conventional_bound_ms
                    ),
                }
                recoveries.append(record)
                round_recoveries.append(record)

            rounds.append({
                "sequence": sequence,
                "release_wall_ns": release_ns,
                "boundary_case": boundary.case_id,
                "target_decision_ms": boundary.target_decision_ms,
                "expected_state": boundary.expected_state,
                "expected_lease": boundary.expected_lease,
                "context_length": context_length,
                "ai_bound_ms": config.ai_bound_ms,
                "missing_outcomes_at_cutoff": missing,
                "observed_outcomes": observed,
                "physical_timely_successes": [
                    list(key) for key in physical_timely_successes
                ],
                "injected_failure_keys": [
                    list(key) for key in accepted_keys
                    if key not in set(injected_successes)
                ],
                "success_keys": [list(key) for key in plan["success_keys"]],
                "outcome_transition": plan["outcome_transition"],
                "rejected_keys": [list(key) for key in plan["rejected_keys"]],
                "lease_accepted": plan["lease_decision"]["accepted"],
                "lease_reason": plan["lease_decision"]["reason"],
                "launch_revalidation_host_ms": (
                    revalidation_completed_ns - revalidation_started_ns
                ) / 1e6,
                "launch_revalidation_attempts": revalidation_attempts,
                "qwen": qwen,
                "qwen_attempt": qwen_attempt,
                "lease_retire": lease_retire,
                "certificate_order": [
                    [row["home_id"], row["request_id"]] for row in ordered
                ],
                "physical_recoveries": [row["key"] for row in round_recoveries],
            })

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
        qwen_rows = [row["qwen"] for row in rounds if row["qwen"] is not None]
        result.update({
            "completed_rounds": len(rounds),
            "missing_outcomes": sum(
                len(row["missing_outcomes_at_cutoff"]) for row in rounds
            ),
            "actual_successes": sum(len(row["success_keys"]) for row in rounds),
            "qwen_units": len(qwen_rows),
            "qwen_execution_ms": summarize([
                row["execution_ms"] for row in qwen_rows
            ]),
            "physical_recovery_count": len(recoveries),
            "conventional_path_ms": summarize([
                (row["actual_completed_ns"] - row["actual_start_ns"]) / 1e6
                for row in recoveries
            ]),
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
