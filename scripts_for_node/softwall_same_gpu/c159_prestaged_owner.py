#!/usr/bin/env python3
"""Pre-staged RAN-home owner for C159 P180 qualification."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import cupy as cp
import numpy as np

from actual_nrx_repeated_control import (
    ACTION_RECOVERY,
    ACTION_SUCCESS,
    DecisionOwner,
)
from actual_nrx_integrated_owner import (
    conventional_oracle,
    wait_json,
    wait_until_ns,
)
from dual_receiver_phy import PairedDualReceiver
from isca_v2.cuda_ipc_channel import CudaIpcOwner, TERMINATE_SEQ
from multigpu_p2p_nrx_worker import FWD_ELEMENTS, LLR_ELEMENTS
from shared_conventional_owner import RX_ELEMENTS, atomic_json, prepare_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--source-device", type=int, required=True)
    parser.add_argument("--nrx-tag", required=True)
    parser.add_argument("--recovery-tag", required=True)
    parser.add_argument("--ipc-dir", type=Path, required=True)
    parser.add_argument("--schedule-file", type=Path, required=True)
    parser.add_argument("--decision-control", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--receiver-seed", type=int, required=True)
    parser.add_argument("--channel-seed-base", type=int, required=True)
    parser.add_argument("--snr-db", type=float, required=True)
    parser.add_argument("--nrx-cutoff-ms", type=float, default=45.0)
    parser.add_argument("--expiry-ms", type=float, default=155.0)
    parser.add_argument("--ready-timeout-s", type=float, default=300.0)
    parser.add_argument("--unit-timeout-s", type=float, default=5.0)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("iterations must be positive")

    cp.cuda.runtime.setDevice(args.source_device)
    receiver = PairedDualReceiver(
        args.engine,
        seed=args.receiver_seed,
        device=args.source_device,
        enable_local_neural=False,
    )
    nrx_forward = cp.empty(FWD_ELEMENTS, dtype=cp.float32)
    nrx_forward_shadow = cp.empty(FWD_ELEMENTS, dtype=cp.float32)
    nrx_backward = cp.empty(LLR_ELEMENTS, dtype=cp.float32)
    recovery_forward = cp.empty(2 * RX_ELEMENTS, dtype=cp.float32)
    recovery_forward_shadow = cp.empty(2 * RX_ELEMENTS, dtype=cp.float32)
    recovery_backward = cp.empty(
        len(receiver.reference_tb) + 1, dtype=cp.uint8
    )
    # All expensive synthetic channel generation and the validation-only local
    # oracle run before readiness. Timed epochs only copy one already-built row
    # into the stable CUDA-IPC buffers.
    nrx_bank = cp.empty((args.iterations, FWD_ELEMENTS), dtype=cp.float32)
    recovery_bank = cp.empty(
        (args.iterations, 2 * RX_ELEMENTS), dtype=cp.float32
    )
    nrx_owner = CudaIpcOwner(
        args.nrx_tag, nrx_forward, nrx_backward, directory=args.ipc_dir
    )
    recovery_owner = CudaIpcOwner(
        args.recovery_tag,
        recovery_forward,
        recovery_backward,
        directory=args.ipc_dir,
    )
    decision = DecisionOwner(args.decision_control)
    # Compile the exact input round-trip monitor before endpoint readiness.
    nrx_forward.fill(cp.float32(0))
    recovery_forward.fill(cp.float32(0))
    cp.copyto(nrx_forward_shadow, nrx_forward)
    cp.copyto(recovery_forward_shadow, recovery_forward)
    if not bool(cp.array_equal(nrx_forward, nrx_forward_shadow).item()):
        raise RuntimeError("NRx input shadow preflight failed")
    if not bool(cp.array_equal(
        recovery_forward, recovery_forward_shadow
    ).item()):
        raise RuntimeError("recovery input shadow preflight failed")
    key = [f"home{args.home_id}", args.request_id]
    safe_key = f"{key[0]}_{key[1]}"
    records = []
    result = {
        "schema": "softwall-c159-prestaged-owner-v1",
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "key": key,
        "source_device": args.source_device,
        "receiver_seed": args.receiver_seed,
        "channel_seed_base": args.channel_seed_base,
        "snr_db": args.snr_db,
        "iterations": args.iterations,
        "nrx_termination_acknowledged": False,
        "recovery_termination_acknowledged": False,
        "error": None,
        "mismatch_diagnostic": None,
    }
    try:
        prestage_started_ns = time.perf_counter_ns()
        staged = []
        for index in range(args.iterations):
            channel_seed = args.channel_seed_base + index
            receiver.apply_rayleigh_awgn(
                args.snr_db,
                channel_seed,
                noise_reference="pre_fading",
            )
            oracle, oracle_correct, oracle_crc_failures, oracle_payload_mismatches = (
                conventional_oracle(receiver)
            )
            nrx_prepare_gpu_ms = receiver.prepare_neural_ipc(nrx_bank[index])
            recovery_prepare_gpu_ms = prepare_forward(
                receiver, recovery_bank[index]
            )
            cp.cuda.get_current_stream().synchronize()
            staged.append({
                "channel_seed": channel_seed,
                "oracle": oracle,
                "oracle_correct": oracle_correct,
                "oracle_crc_failures": oracle_crc_failures,
                "oracle_payload_mismatches": oracle_payload_mismatches,
                "nrx_prepare_gpu_ms": nrx_prepare_gpu_ms,
                "recovery_prepare_gpu_ms": recovery_prepare_gpu_ms,
            })
        prestage_completed_ns = time.perf_counter_ns()
        nrx_owner.wait_ready(args.ready_timeout_s)
        recovery_owner.wait_ready(args.ready_timeout_s)
        atomic_json(args.ready_file, {
            "schema": "softwall-c159-prestaged-owner-ready-v1",
            "key": key,
            "ready": True,
            "pre_staged_iterations": len(staged),
            "prestage_completed_ns": prestage_completed_ns,
        })
        schedule = wait_json(args.schedule_file, args.ready_timeout_s)
        first_release_ns = int(schedule["first_release_wall_ns"])
        period_ns = int(schedule["period_ns"])
        if int(schedule["iterations"]) != args.iterations:
            raise RuntimeError("schedule iteration mismatch")

        for index in range(args.iterations):
            sequence = index + 1
            release_ns = first_release_ns + index * period_ns
            if sequence > 1:
                previous_release_ns = release_ns - period_ns
                wait_until_ns(
                    previous_release_ns + round(args.expiry_ms * 1e6)
                )
            stage = staged[index]
            channel_seed = stage["channel_seed"]
            oracle = stage["oracle"]
            oracle_correct = stage["oracle_correct"]
            oracle_crc_failures = stage["oracle_crc_failures"]
            oracle_payload_mismatches = stage["oracle_payload_mismatches"]
            nrx_prepare_gpu_ms = stage["nrx_prepare_gpu_ms"]
            recovery_prepare_gpu_ms = stage["recovery_prepare_gpu_ms"]
            preparation_started_ns = time.perf_counter_ns()
            bank_copy_begin = cp.cuda.Event()
            bank_copy_end = cp.cuda.Event()
            bank_copy_begin.record()
            cp.copyto(nrx_forward, nrx_bank[index])
            cp.copyto(recovery_forward, recovery_bank[index])
            cp.copyto(nrx_forward_shadow, nrx_forward)
            cp.copyto(recovery_forward_shadow, recovery_forward)
            bank_copy_end.record()
            bank_copy_end.synchronize()
            bank_copy_gpu_ms = float(cp.cuda.get_elapsed_time(
                bank_copy_begin, bank_copy_end
            ))
            preparation_completed_ns = time.perf_counter_ns()
            wait_until_ns(release_ns)
            published_ns = time.perf_counter_ns()
            nrx_owner.publish_forward(sequence)
            nrx_owner.wait_backward(sequence, args.unit_timeout_s)
            cp.cuda.runtime.deviceSynchronize()
            nrx_returned_ns = time.perf_counter_ns()
            nrx_input_roundtrip_equal = bool(cp.array_equal(
                nrx_forward, nrx_forward_shadow
            ).item())
            if not nrx_input_roundtrip_equal:
                raise RuntimeError(
                    f"NRx input round-trip mismatch sequence={sequence}"
                )
            neural = receiver.complete_neural_ipc(nrx_backward)
            nrx_completed_ns = time.perf_counter_ns()
            neural_correct = bool(neural[1])
            outcome = {
                "schema": "softwall-c159-prestaged-outcome-v1",
                "sequence": sequence,
                "key": key,
                "release_wall_ns": release_ns,
                "completed_ns": nrx_completed_ns,
                "release_to_complete_ms": (nrx_completed_ns - release_ns) / 1e6,
                "neural_correct": neural_correct,
                "crc_failures": int(neural[2]),
                "payload_mismatches": int(neural[3]),
                "timely_success": (
                    neural_correct
                    and nrx_completed_ns
                    <= release_ns + round(args.nrx_cutoff_ms * 1e6)
                ),
            }
            decision.publish_outcome(
                sequence,
                timely_success=outcome["timely_success"],
                neural_correct=neural_correct,
                completed_ns=nrx_completed_ns,
                crc_failures=int(neural[2]),
                payload_mismatches=int(neural[3]),
            )
            dispatch = decision.wait_dispatch(sequence, args.ready_timeout_s)
            recovery_published_ns = None
            recovery_returned_ns = None
            recovery_crc = None
            recovery_payload_equal = None
            recovery_payload_matches_reference = None
            recovery_correct = None
            recovery_input_roundtrip_equal = None
            recovery_contract_valid = None
            recovery_local_class_equal = None
            recovery_equivalent = None
            if dispatch["action"] == ACTION_SUCCESS:
                if not neural_correct:
                    raise RuntimeError("incorrect NRx outcome was committed")
                commit_source = "actual_nrx"
            elif dispatch["action"] == ACTION_RECOVERY:
                recovery_published_ns = time.perf_counter_ns()
                recovery_owner.publish_forward(sequence)
                recovery_owner.wait_backward(sequence, args.unit_timeout_s)
                cp.cuda.runtime.deviceSynchronize()
                recovery_returned_ns = time.perf_counter_ns()
                recovery_input_roundtrip_equal = bool(cp.array_equal(
                    recovery_forward, recovery_forward_shadow
                ).item())
                observed = cp.asnumpy(recovery_backward)
                recovery_crc = int(observed[0])
                recovery_payload_equal = bool(
                    np.array_equal(observed[1:], oracle[1:])
                )
                recovery_payload_matches_reference = bool(
                    np.array_equal(observed[1:], receiver.reference_tb)
                )
                recovery_correct = (
                    recovery_crc == 0 and recovery_payload_matches_reference
                )
                recovery_contract_valid = (
                    recovery_crc != 0 or recovery_payload_matches_reference
                )
                recovery_local_class_equal = recovery_correct == oracle_correct
                recovery_equivalent = recovery_local_class_equal
                if (not recovery_input_roundtrip_equal
                        or not recovery_contract_valid):
                    result["mismatch_diagnostic"] = {
                        "sequence": sequence,
                        "key": key,
                        "channel_seed": channel_seed,
                        "oracle_crc": int(oracle[0]),
                        "recovery_crc": recovery_crc,
                        "payload_equal": recovery_payload_equal,
                        "payload_matches_reference": (
                            recovery_payload_matches_reference
                        ),
                        "recovery_correct": recovery_correct,
                        "input_roundtrip_equal": (
                            recovery_input_roundtrip_equal
                        ),
                        "contract_valid": recovery_contract_valid,
                        "oracle_correct": oracle_correct,
                    }
                    raise RuntimeError(
                        f"recovery input/contract invalid sequence={sequence}"
                    )
                commit_source = "shared_conventional"
            else:
                raise RuntimeError(
                    f"accepted request has no action sequence={sequence}"
                )
            commit_ns = time.perf_counter_ns()
            records.append({
                "sequence": sequence,
                "channel_seed": channel_seed,
                "release_wall_ns": release_ns,
                "preparation_started_ns": preparation_started_ns,
                "preparation_completed_ns": preparation_completed_ns,
                "input_ready_before_release": preparation_completed_ns <= release_ns,
                "bank_copy_gpu_ms": bank_copy_gpu_ms,
                "nrx_prepare_gpu_ms": nrx_prepare_gpu_ms,
                "recovery_prepare_gpu_ms": recovery_prepare_gpu_ms,
                "nrx_published_ns": published_ns,
                "nrx_returned_ns": nrx_returned_ns,
                "nrx_completed_ns": nrx_completed_ns,
                "nrx_input_roundtrip_equal": nrx_input_roundtrip_equal,
                "nrx_release_to_complete_ms": (
                    nrx_completed_ns - release_ns
                ) / 1e6,
                "neural_correct": neural_correct,
                "neural_post_gpu_ms": float(neural[0]),
                "timely_success": outcome["timely_success"],
                "oracle_crc": int(oracle[0]),
                "oracle_correct": oracle_correct,
                "oracle_crc_failures": oracle_crc_failures,
                "oracle_payload_mismatches": oracle_payload_mismatches,
                "recovery_published_ns": recovery_published_ns,
                "recovery_returned_ns": recovery_returned_ns,
                "recovery_crc": recovery_crc,
                "recovery_payload_equal_to_local_oracle": recovery_payload_equal,
                "recovery_payload_matches_reference": (
                    recovery_payload_matches_reference
                ),
                "recovery_correct": recovery_correct,
                "recovery_input_roundtrip_equal": (
                    recovery_input_roundtrip_equal
                ),
                "recovery_contract_valid": recovery_contract_valid,
                "recovery_local_class_equal": recovery_local_class_equal,
                "commit_source": commit_source,
                "commit_count": 1,
                "commit_ns": commit_ns,
                "release_to_commit_ms": (commit_ns - release_ns) / 1e6,
                "deadline_miss": (
                    commit_ns > release_ns + round(args.expiry_ms * 1e6)
                ),
            })
        nrx_owner.publish_forward(TERMINATE_SEQ)
        recovery_owner.publish_forward(TERMINATE_SEQ)
        nrx_owner.wait_backward(TERMINATE_SEQ, args.ready_timeout_s)
        result["nrx_termination_acknowledged"] = True
        recovery_owner.wait_backward(TERMINATE_SEQ, args.ready_timeout_s)
        result["recovery_termination_acknowledged"] = True
    except BaseException as error:
        result["error"] = repr(error)
        for owner in (nrx_owner, recovery_owner):
            try:
                owner.publish_forward(TERMINATE_SEQ)
            except BaseException:
                pass
        raise
    finally:
        result.update({
            "prestage_started_ns": locals().get("prestage_started_ns"),
            "prestage_completed_ns": locals().get("prestage_completed_ns"),
            "prestage_ms": (
                (prestage_completed_ns - prestage_started_ns) / 1e6
                if "prestage_completed_ns" in locals() else None
            ),
            "pre_staged_iterations": len(locals().get("staged", [])),
            "bank_bytes": int(
                nrx_bank.nbytes + recovery_bank.nbytes
            ),
            "completed_iterations": len(records),
            "actual_nrx_commits": sum(
                row["commit_source"] == "actual_nrx" for row in records
            ),
            "recovery_commits": sum(
                row["commit_source"] == "shared_conventional" for row in records
            ),
            "oracle_equivalent_recoveries": sum(
                row["recovery_local_class_equal"] is True
                for row in records
            ),
            "contract_valid_recoveries": sum(
                row["recovery_contract_valid"] is True for row in records
            ),
            "deadline_misses": sum(row["deadline_miss"] for row in records),
            "records": records,
        })
        atomic_json(args.output, result)
        nrx_owner.close()
        recovery_owner.close()
        decision.close()


if __name__ == "__main__":
    main()
