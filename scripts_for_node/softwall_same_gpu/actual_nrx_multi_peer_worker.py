#!/usr/bin/env python3
"""One persistent TensorRT NeuralRx endpoint serving several RAN owners."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import cupy as cp

from isca_v2.cuda_ipc_channel import CudaIpcPeer, TERMINATE_SEQ
from multigpu_p2p_ipc_gate import (
    P2PCopier,
    close_peer_before_ack,
    enable_peer_access,
    summarize,
)
from multigpu_p2p_nrx_worker import (
    CE_ELEMENTS,
    CE_SHAPE,
    FWD_ELEMENTS,
    LLR_ELEMENTS,
    LLR_SHAPE,
    RX_ELEMENTS,
    RX_SHAPE,
    atomic_json,
    section,
)
from nrx_trt_direct import DirectNrx
from shared_conventional_worker import wait_sequence


def load_spec(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "softwall-actual-nrx-peer-spec-v1":
        raise ValueError("invalid actual-NRx peer specification")
    peers = value.get("peers", [])
    if not peers or len({tuple(row["key"]) for row in peers}) != len(peers):
        raise ValueError("empty or duplicate actual-NRx peer specification")
    return peers


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--peer-spec", type=Path, required=True)
    parser.add_argument("--ipc-dir", type=Path, required=True)
    parser.add_argument("--release-file", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--destination-device", type=int, default=3)
    parser.add_argument("--ready-timeout-s", type=float, default=240.0)
    parser.add_argument("--unit-timeout-s", type=float, default=5.0)
    parser.add_argument("--graph-warmup", type=int, default=20)
    parser.add_argument("--release-prewarm-lead-ms", type=float, default=100.0)
    args = parser.parse_args()
    specs = load_spec(args.peer_spec)

    peers = []
    for row in specs:
        source = int(row["source_device"])
        cp.cuda.runtime.setDevice(source)
        peers.append(CudaIpcPeer(
            row["nrx_tag"], args.ready_timeout_s, directory=args.ipc_dir
        ))
    access = {}
    for source in sorted({int(row["source_device"]) for row in specs}):
        access.update(enable_peer_access(source, args.destination_device))

    with cp.cuda.Device(args.destination_device):
        remote_forward = cp.empty(FWD_ELEMENTS, dtype=cp.float32)
        remote_backward = cp.empty(LLR_ELEMENTS, dtype=cp.float32)
        runtime = DirectNrx(args.engine)
        offset = 0
        for name, count, shape in (
            ("rx_slot_real", RX_ELEMENTS, RX_SHAPE),
            ("rx_slot_imag", RX_ELEMENTS, RX_SHAPE),
            ("h_hat_real", CE_ELEMENTS, CE_SHAPE),
            ("h_hat_imag", CE_ELEMENTS, CE_SHAPE),
        ):
            runtime.bind_tensor(name, section(remote_forward, offset, count, shape))
            offset += count
        runtime.bind_tensor("output_1", remote_backward.reshape((1,) + LLR_SHAPE))
        runtime.inputs["active_dmrs_ports"].fill(1)
        runtime.inputs["dmrs_ofdm_pos"][:] = cp.asarray([[0, 5, 9]], dtype=cp.int32)
        runtime.inputs["dmrs_subcarrier_pos"][:] = cp.asarray(
            [[0, 2, 4, 6, 8, 10]], dtype=cp.int32
        )
        runtime.capture_graph()
        for _ in range(args.graph_warmup):
            runtime.launch(use_graph=True)
        runtime.stream.synchronize()

    contexts = []
    for row, peer in zip(specs, peers):
        source = int(row["source_device"])
        contexts.append({
            "forward": P2PCopier(source, args.destination_device),
            "backward": P2PCopier(args.destination_device, source),
            "peer": peer,
            "spec": row,
        })

    # Importing an IPC allocation and establishing its peer mapping can add a
    # large first-use host tail even when the memcpy CUDA event is short.  The
    # owner has not published a request yet, so exercise every distinct peer
    # path before readiness; payload values are deliberately irrelevant here.
    for context in contexts:
        peer = context["peer"]
        context["forward"].copy(
            remote_forward, peer.forward.view(cp.float32)
        )
        with cp.cuda.Device(args.destination_device):
            runtime.launch(use_graph=True)
            runtime.stream.synchronize()
        context["backward"].copy(
            peer.backward.view(cp.float32), remote_backward
        )
    # Compile the safety-monitor reduction outside the timed epoch.  Without
    # this call the first cp.isfinite invocation can add tens of milliseconds
    # of CuPy JIT latency that is unrelated to the NRx data path.
    with cp.cuda.Device(args.destination_device):
        if not bool(cp.all(cp.isfinite(remote_backward)).item()):
            raise RuntimeError("non-finite output during monitor warmup")

    records = []
    handles_closed = False
    error = None
    try:
        for peer in peers:
            peer.mark_ready()
        release = wait_json(args.release_file, args.ready_timeout_s)
        release_wall_ns = int(release["release_wall_ns"])
        wait_until_ns(
            release_wall_ns - round(args.release_prewarm_lead_ms * 1e6)
        )
        release_prewarm_started_ns = time.perf_counter_ns()
        for context in contexts:
            peer = context["peer"]
            context["forward"].copy(
                remote_forward, peer.forward.view(cp.float32)
            )
            with cp.cuda.Device(args.destination_device):
                runtime.launch(use_graph=True)
                runtime.stream.synchronize()
            context["backward"].copy(
                peer.backward.view(cp.float32), remote_backward
            )
        release_prewarm_completed_ns = time.perf_counter_ns()
        for context in contexts:
            peer = context["peer"]
            row = context["spec"]
            # Owners initialize a full cuPHY receiver and prepare both the NRx
            # and recovery views before the common release.  That pre-release
            # phase is readiness, not service time, so do not charge it to the
            # per-request path timeout.
            wait_sequence(peer, 1, args.ready_timeout_s)
            observed_ns = time.perf_counter_ns()
            forward = context["forward"].copy(
                remote_forward, peer.forward.view(cp.float32)
            )
            with cp.cuda.Device(args.destination_device):
                begin = cp.cuda.Event()
                end = cp.cuda.Event()
                wall_begin_ns = time.perf_counter_ns()
                with runtime.stream:
                    begin.record()
                    runtime.launch(use_graph=True)
                    end.record()
                end.synchronize()
                nrx_gpu_ms = float(cp.cuda.get_elapsed_time(begin, end))
                nrx_host_ms = (time.perf_counter_ns() - wall_begin_ns) / 1e6
                finite = bool(cp.all(cp.isfinite(remote_backward)).item())
            if not finite:
                raise RuntimeError(f"non-finite actual NRx output {row['key']}")
            backward = context["backward"].copy(
                peer.backward.view(cp.float32), remote_backward
            )
            completed_ns = time.perf_counter_ns()
            peer.publish_backward(1)
            records.append({
                "key": row["key"],
                "observed_ns": observed_ns,
                "completed_ns": completed_ns,
                "forward_gpu_us": forward["gpu_us"],
                "forward_host_us": forward["host_us"],
                "nrx_gpu_ms": nrx_gpu_ms,
                "nrx_host_ms": nrx_host_ms,
                "backward_gpu_us": backward["gpu_us"],
                "backward_host_us": backward["host_us"],
                "worker_path_ms": (completed_ns - observed_ns) / 1e6,
                "output_finite": finite,
            })
        for peer in peers:
            wait_sequence(peer, TERMINATE_SEQ, args.ready_timeout_s)
        for row, peer in zip(specs, peers):
            cp.cuda.runtime.setDevice(int(row["source_device"]))
            close_peer_before_ack(peer)
        handles_closed = True
    except BaseException as caught:
        error = repr(caught)
        raise
    finally:
        result = {
            "schema": "softwall-actual-nrx-multi-peer-worker-v1",
            "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "destination_device": args.destination_device,
            "peer_access": access,
            "release_prewarm_lead_ms": args.release_prewarm_lead_ms,
            "release_prewarm_started_ns": locals().get(
                "release_prewarm_started_ns"
            ),
            "release_prewarm_completed_ns": locals().get(
                "release_prewarm_completed_ns"
            ),
            "completed_units": len(records),
            "finite_units": sum(row["output_finite"] for row in records),
            "ipc_handles_closed_before_ack": handles_closed,
            "forward_gpu_us": summarize([row["forward_gpu_us"] for row in records]),
            "nrx_gpu_ms": summarize([row["nrx_gpu_ms"] for row in records]),
            "backward_gpu_us": summarize([row["backward_gpu_us"] for row in records]),
            "worker_path_ms": summarize([row["worker_path_ms"] for row in records]),
            "records": records,
            "error": error,
        }
        atomic_json(args.output, result)
        if not handles_closed:
            for peer in peers:
                try:
                    peer.close()
                except BaseException:
                    pass


if __name__ == "__main__":
    main()
