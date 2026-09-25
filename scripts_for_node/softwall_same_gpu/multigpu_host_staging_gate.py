#!/usr/bin/env python3
"""Cross-process CUDA-IPC transport through pinned host staging."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import cupy as cp
import numpy as np

from isca_v2.cuda_ipc_channel import BWD_OFFSET, CudaIpcOwner, CudaIpcPeer, TERMINATE_SEQ


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


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "count": len(values), "mean": sum(values) / len(values),
        "p50": ordered[round((len(ordered) - 1) * 0.50)],
        "p99": ordered[round((len(ordered) - 1) * 0.99)], "max": ordered[-1],
    }


def pattern(sequence: int) -> float:
    return float((sequence * 17) % 251 + 1)


def device_name(index: int) -> str:
    value = cp.cuda.runtime.getDeviceProperties(index)["name"]
    return value.decode() if isinstance(value, bytes) else str(value)


def common_result(args: argparse.Namespace) -> dict:
    script = Path(__file__).resolve()
    channel = Path(args.channel_source).resolve()
    return {
        "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "role": args.role, "tag": args.tag, "iterations": args.iterations,
        "source_device": args.source_device, "destination_device": args.destination_device,
        "forward_elements": FWD_ELEMENTS, "forward_bytes": FWD_ELEMENTS * 4,
        "backward_elements": LLR_ELEMENTS, "backward_bytes": LLR_ELEMENTS * 4,
        "source_sha256": {str(script): sha256(script), str(channel): sha256(channel)},
    }


class HostStagingCopier:
    def __init__(self, source_device: int, destination_device: int, nbytes: int) -> None:
        self.source_device = source_device
        self.destination_device = destination_device
        self.pinned = cp.cuda.alloc_pinned_memory(nbytes)
        self.host = np.frombuffer(self.pinned, dtype=np.uint8, count=nbytes)
        self.host_ptr = int(self.host.ctypes.data)
        with cp.cuda.Device(source_device):
            self.source_stream = cp.cuda.Stream(non_blocking=True)
        with cp.cuda.Device(destination_device):
            self.destination_stream = cp.cuda.Stream(non_blocking=True)

    def copy(self, destination: cp.ndarray, source: cp.ndarray) -> dict[str, float]:
        if destination.nbytes != source.nbytes or destination.nbytes != self.host.nbytes:
            raise RuntimeError("host-staging size mismatch")
        host_begin_ns = time.perf_counter_ns()
        with cp.cuda.Device(self.source_device):
            d2h_begin = cp.cuda.Event(); d2h_end = cp.cuda.Event()
            d2h_begin.record(self.source_stream)
            cp.cuda.runtime.memcpyAsync(
                self.host_ptr, source.data.ptr, source.nbytes,
                cp.cuda.runtime.memcpyDeviceToHost, self.source_stream.ptr,
            )
            d2h_end.record(self.source_stream); d2h_end.synchronize()
            d2h_us = float(cp.cuda.get_elapsed_time(d2h_begin, d2h_end) * 1000.0)
        with cp.cuda.Device(self.destination_device):
            h2d_begin = cp.cuda.Event(); h2d_end = cp.cuda.Event()
            h2d_begin.record(self.destination_stream)
            cp.cuda.runtime.memcpyAsync(
                destination.data.ptr, self.host_ptr, destination.nbytes,
                cp.cuda.runtime.memcpyHostToDevice, self.destination_stream.ptr,
            )
            h2d_end.record(self.destination_stream); h2d_end.synchronize()
            h2d_us = float(cp.cuda.get_elapsed_time(h2d_begin, h2d_end) * 1000.0)
        return {
            "d2h_gpu_us": d2h_us, "h2d_gpu_us": h2d_us,
            "gpu_us": d2h_us + h2d_us,
            "host_us": (time.perf_counter_ns() - host_begin_ns) / 1000.0,
        }


def close_peer_before_ack(peer: CudaIpcPeer) -> None:
    cp.cuda.runtime.ipcCloseMemHandle(peer.forward_ptr)
    cp.cuda.runtime.ipcCloseMemHandle(peer.backward_ptr)
    peer.control.write(BWD_OFFSET, TERMINATE_SEQ)
    peer.control.close()


def run_owner(args: argparse.Namespace) -> None:
    cp.cuda.runtime.setDevice(args.source_device)
    forward = cp.empty(FWD_ELEMENTS, dtype=cp.float32)
    backward = cp.empty(LLR_ELEMENTS, dtype=cp.float32)
    owner = CudaIpcOwner(args.tag, forward, backward, directory=args.ipc_dir)
    records = []; error = None; termination_acknowledged = False
    started = time.perf_counter()
    try:
        owner.wait_ready(args.ready_timeout_s)
        for sequence in range(1, args.iterations + 1):
            expected = pattern(sequence)
            with cp.cuda.Device(args.source_device):
                forward.fill(cp.float32(expected)); cp.cuda.get_current_stream().synchronize()
            published_ns = time.perf_counter_ns()
            owner.publish_forward(sequence)
            owner.wait_backward(sequence, args.unit_timeout_s)
            returned_ns = time.perf_counter_ns()
            with cp.cuda.Device(args.source_device):
                backward_ok = bool(cp.all(backward == cp.float32(expected + 0.5)).item())
            records.append({"sequence": sequence, "backward_ok": backward_ok,
                            "round_trip_us": (returned_ns - published_ns) / 1000.0})
            if not backward_ok:
                raise RuntimeError(f"backward integrity failure sequence={sequence}")
        owner.publish_forward(TERMINATE_SEQ)
        owner.wait_backward(TERMINATE_SEQ, args.ready_timeout_s)
        termination_acknowledged = True
    except BaseException as caught:
        error = repr(caught)
        try: owner.publish_forward(TERMINATE_SEQ)
        except BaseException: pass
        raise
    finally:
        result = common_result(args)
        result.update({
            "schema":"softwall-multigpu-host-staging-owner-v1",
            "device_name":device_name(args.source_device), "completed_units":len(records),
            "integrity_errors":sum(not x["backward_ok"] for x in records),
            "sequences_contiguous":[x["sequence"] for x in records] == list(range(1,len(records)+1)),
            "termination_acknowledged":termination_acknowledged,
            "round_trip_us":summarize([x["round_trip_us"] for x in records]),
            "wall_s":time.perf_counter()-started, "error":error, "records":records,
        })
        atomic_json(args.output,result); owner.close()


def run_worker(args: argparse.Namespace) -> None:
    if args.source_device == args.destination_device:
        raise ValueError("host-staging gate requires distinct devices")
    cp.cuda.runtime.setDevice(args.source_device)
    peer = CudaIpcPeer(args.tag,args.ready_timeout_s,directory=args.ipc_dir)
    with cp.cuda.Device(args.destination_device):
        remote_forward=cp.empty(FWD_ELEMENTS,dtype=cp.float32)
        remote_backward=cp.empty(LLR_ELEMENTS,dtype=cp.float32)
    forward_copy=HostStagingCopier(args.source_device,args.destination_device,remote_forward.nbytes)
    backward_copy=HostStagingCopier(args.destination_device,args.source_device,remote_backward.nbytes)
    records=[]; error=None; handles_closed=False; started=time.perf_counter()
    try:
        peer.mark_ready(); last=0
        while True:
            sequence=peer.read_forward()
            if sequence==TERMINATE_SEQ: break
            if sequence<=last: time.sleep(0); continue
            if sequence!=last+1: raise RuntimeError(f"sequence skip expected={last+1} observed={sequence}")
            expected=pattern(sequence)
            forward=forward_copy.copy(remote_forward,peer.forward.view(cp.float32))
            with cp.cuda.Device(args.destination_device):
                forward_ok=bool(cp.all(remote_forward==cp.float32(expected)).item())
                remote_backward.fill(cp.float32(expected+0.5)); cp.cuda.get_current_stream().synchronize()
            if not forward_ok: raise RuntimeError(f"forward integrity failure sequence={sequence}")
            backward=backward_copy.copy(peer.backward.view(cp.float32),remote_backward)
            peer.publish_backward(sequence)
            records.append({
                "sequence":sequence,"forward_ok":forward_ok,
                "forward_gpu_us":forward["gpu_us"],"forward_host_us":forward["host_us"],
                "forward_d2h_gpu_us":forward["d2h_gpu_us"],"forward_h2d_gpu_us":forward["h2d_gpu_us"],
                "backward_gpu_us":backward["gpu_us"],"backward_host_us":backward["host_us"],
                "backward_d2h_gpu_us":backward["d2h_gpu_us"],"backward_h2d_gpu_us":backward["h2d_gpu_us"],
            }); last=sequence
        close_peer_before_ack(peer); handles_closed=True
    except BaseException as caught:
        error=repr(caught); raise
    finally:
        result=common_result(args)
        result.update({
            "schema":"softwall-multigpu-host-staging-worker-v1",
            "source_device_name":device_name(args.source_device),
            "destination_device_name":device_name(args.destination_device),
            "completed_units":len(records),"integrity_errors":sum(not x["forward_ok"] for x in records),
            "sequences_contiguous":[x["sequence"] for x in records] == list(range(1,len(records)+1)),
            "ipc_handles_closed_before_ack":handles_closed,
            "forward_gpu_us":summarize([x["forward_gpu_us"] for x in records]),
            "forward_host_us":summarize([x["forward_host_us"] for x in records]),
            "backward_gpu_us":summarize([x["backward_gpu_us"] for x in records]),
            "backward_host_us":summarize([x["backward_host_us"] for x in records]),
            "wall_s":time.perf_counter()-started,"error":error,"records":records,
        })
        atomic_json(args.output,result)
        if not handles_closed:
            try: peer.close()
            except BaseException: pass


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--role",choices=("owner","worker"),required=True)
    parser.add_argument("--tag",required=True); parser.add_argument("--ipc-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True); parser.add_argument("--iterations",type=int,required=True)
    parser.add_argument("--source-device",type=int,default=0); parser.add_argument("--destination-device",type=int,default=1)
    parser.add_argument("--ready-timeout-s",type=float,default=120.0); parser.add_argument("--unit-timeout-s",type=float,default=2.0)
    parser.add_argument("--channel-source",default="/softwall_task1/isca_v2/cuda_ipc_channel.py")
    args=parser.parse_args()
    if args.iterations<=0 or min(args.ready_timeout_s,args.unit_timeout_s)<=0: parser.error("positive counts/timeouts required")
    args.ipc_dir.mkdir(parents=True,exist_ok=True)
    run_owner(args) if args.role=="owner" else run_worker(args)


if __name__=="__main__": main()

