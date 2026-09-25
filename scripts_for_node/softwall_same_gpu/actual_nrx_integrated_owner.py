#!/usr/bin/env python3
"""RAN-home owner for an actual-NRx-driven shared recovery transaction."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import cupy as cp
import numpy as np

from dual_receiver_phy import PairedDualReceiver
from isca_v2.cuda_ipc_channel import CudaIpcOwner, TERMINATE_SEQ
from multigpu_p2p_nrx_worker import FWD_ELEMENTS, LLR_ELEMENTS
from shared_conventional_owner import RX_ELEMENTS, atomic_json, prepare_forward


def wait_json(path: Path, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.002)
    raise TimeoutError(f"timed out waiting for {path}")


def wait_until_ns(target_ns: int) -> None:
    while True:
        remaining = target_ns - time.perf_counter_ns()
        if remaining <= 0:
            return
        if remaining > 2_000_000:
            time.sleep((remaining - 1_000_000) / 1e9)
        else:
            time.sleep(0)


def conventional_oracle(receiver) -> tuple[np.ndarray, bool, int, int]:
    with cp.cuda.Device(receiver.device), receiver.stream:
        output = receiver.conventional_once()
        receiver.stream.synchronize()
        blocks, crc_values = output
        payload = cp.asnumpy(
            cp.asarray(blocks[0]).reshape(-1).astype(cp.uint8)
        )
        crc = int(cp.asnumpy(
            cp.asarray(crc_values[0]).reshape(-1).astype(cp.uint8)
        )[0])
        response = np.empty(payload.size + 1, dtype=np.uint8)
        response[0] = crc
        response[1:] = payload
        correct, crc_failures, payload_mismatches = receiver._verify(output)
        return response, bool(correct), int(crc_failures), int(payload_mismatches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--source-device", type=int, required=True)
    parser.add_argument("--nrx-tag", required=True)
    parser.add_argument("--recovery-tag", required=True)
    parser.add_argument("--ipc-dir", type=Path, required=True)
    parser.add_argument("--release-file", type=Path, required=True)
    parser.add_argument("--dispatch-file", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--outcome-file", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receiver-seed", type=int, required=True)
    parser.add_argument("--channel-seed", type=int, required=True)
    parser.add_argument("--snr-db", type=float, required=True)
    parser.add_argument("--nrx-cutoff-ms", type=float, default=45.0)
    parser.add_argument("--expiry-ms", type=float, default=155.0)
    parser.add_argument("--ready-timeout-s", type=float, default=240.0)
    parser.add_argument("--unit-timeout-s", type=float, default=5.0)
    args = parser.parse_args()

    cp.cuda.runtime.setDevice(args.source_device)
    receiver = PairedDualReceiver(
        args.engine,
        seed=args.receiver_seed,
        device=args.source_device,
        enable_local_neural=False,
    )
    nrx_forward = cp.empty(FWD_ELEMENTS, dtype=cp.float32)
    nrx_backward = cp.empty(LLR_ELEMENTS, dtype=cp.float32)
    recovery_forward = cp.empty(2 * RX_ELEMENTS, dtype=cp.float32)
    recovery_backward = cp.empty(
        len(receiver.reference_tb) + 1, dtype=cp.uint8
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
    key = [f"home{args.home_id}", args.request_id]
    result = {
        "schema": "softwall-actual-nrx-integrated-owner-v1",
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "key": key,
        "source_device": args.source_device,
        "receiver_seed": args.receiver_seed,
        "channel_seed": args.channel_seed,
        "snr_db": args.snr_db,
        "commit_count": 0,
        "nrx_termination_acknowledged": False,
        "recovery_termination_acknowledged": False,
        "error": None,
    }
    try:
        nrx_owner.wait_ready(args.ready_timeout_s)
        recovery_owner.wait_ready(args.ready_timeout_s)
        preparation_started_ns = time.perf_counter_ns()
        receiver.apply_rayleigh_awgn(
            args.snr_db,
            args.channel_seed,
            noise_reference="pre_fading",
        )
        oracle, oracle_correct, oracle_crc_failures, oracle_payload_mismatches = (
            conventional_oracle(receiver)
        )
        nrx_prepare_gpu_ms = receiver.prepare_neural_ipc(nrx_forward)
        recovery_prepare_gpu_ms = prepare_forward(receiver, recovery_forward)
        preparation_completed_ns = time.perf_counter_ns()
        atomic_json(args.ready_file, {
            "schema": "softwall-actual-nrx-owner-ready-v1",
            "key": key,
            "prepared": True,
            "preparation_completed_ns": preparation_completed_ns,
        })

        release = wait_json(args.release_file, args.ready_timeout_s)
        release_ns = int(release["release_wall_ns"])
        wait_until_ns(release_ns)
        nrx_published_ns = time.perf_counter_ns()
        nrx_owner.publish_forward(1)
        nrx_owner.wait_backward(1, args.unit_timeout_s)
        nrx_returned_ns = time.perf_counter_ns()
        neural = receiver.complete_neural_ipc(nrx_backward)
        nrx_completed_ns = time.perf_counter_ns()
        neural_correct = bool(neural[1])
        outcome = {
            "schema": "softwall-actual-nrx-outcome-v1",
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
        atomic_json(args.outcome_file, outcome)

        dispatch = wait_json(args.dispatch_file, args.ready_timeout_s)
        success_keys = dispatch["success_keys"]
        recovery_keys = dispatch["physical_recovery_keys"]
        commit_source = None
        recovery_equivalent = None
        recovery_published_ns = None
        recovery_returned_ns = None
        if key in success_keys:
            if not neural_correct:
                raise RuntimeError("coordinator accepted an incorrect NeuralRx result")
            commit_source = "actual_nrx"
        elif key in recovery_keys:
            recovery_published_ns = time.perf_counter_ns()
            recovery_owner.publish_forward(1)
            recovery_owner.wait_backward(1, args.unit_timeout_s)
            recovery_returned_ns = time.perf_counter_ns()
            observed = cp.asnumpy(recovery_backward)
            recovery_payload_equal = bool(np.array_equal(observed[1:], oracle[1:]))
            recovery_crc = int(observed[0])
            # A failed-CRC payload is not consumable radio data and its decoded
            # garbage bits need not be stable across cuPHY contexts/devices.
            # Successful results must match exactly; failed results must retain
            # the same failure status.
            recovery_equivalent = (
                recovery_crc == 0 and int(oracle[0]) == 0
                and recovery_payload_equal
            ) or (
                recovery_crc != 0 and int(oracle[0]) != 0
            )
            if not recovery_equivalent:
                raise RuntimeError("shared recovery differs from local conventional oracle")
            commit_source = "shared_conventional"
        else:
            raise RuntimeError(f"accepted request has no dispatch action: {key}")
        commit_ns = time.perf_counter_ns()
        result["commit_count"] = 1
        result.update({
            "release_wall_ns": release_ns,
            "preparation_started_ns": preparation_started_ns,
            "preparation_completed_ns": preparation_completed_ns,
            "input_ready_before_release": preparation_completed_ns <= release_ns,
            "nrx_prepare_gpu_ms": nrx_prepare_gpu_ms,
            "recovery_prepare_gpu_ms": recovery_prepare_gpu_ms,
            "nrx_published_ns": nrx_published_ns,
            "nrx_returned_ns": nrx_returned_ns,
            "nrx_completed_ns": nrx_completed_ns,
            "nrx_release_to_complete_ms": (nrx_completed_ns - release_ns) / 1e6,
            "neural_correct": neural_correct,
            "neural_post_gpu_ms": float(neural[0]),
            "neural_crc_failures": int(neural[2]),
            "neural_payload_mismatches": int(neural[3]),
            "timely_success": outcome["timely_success"],
            "oracle_crc": int(oracle[0]),
            "oracle_correct": oracle_correct,
            "oracle_crc_failures": oracle_crc_failures,
            "oracle_payload_mismatches": oracle_payload_mismatches,
            "dispatch_success": key in success_keys,
            "dispatch_recovery": key in recovery_keys,
            "recovery_published_ns": recovery_published_ns,
            "recovery_returned_ns": recovery_returned_ns,
            "recovery_crc": recovery_crc if recovery_returned_ns else None,
            "recovery_payload_equal_to_local_oracle": (
                recovery_payload_equal if recovery_returned_ns else None
            ),
            "recovery_equivalent_to_local_oracle": recovery_equivalent,
            "commit_source": commit_source,
            "commit_ns": commit_ns,
            "release_to_commit_ms": (commit_ns - release_ns) / 1e6,
            "deadline_miss": commit_ns > release_ns + round(args.expiry_ms * 1e6),
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
        atomic_json(args.output, result)
        nrx_owner.close()
        recovery_owner.close()


if __name__ == "__main__":
    main()
