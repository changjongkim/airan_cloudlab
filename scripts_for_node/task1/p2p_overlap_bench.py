#!/usr/bin/env python3
"""Fair same-MIG vs cross-MIG P2P overlap benchmark.

The previous request/response benchmark serialized L1 and NRx, so there was no
same-partition overlap for MIG isolation to remove.  This benchmark keeps two
slots in flight and reports L1 active CUDA time separately from end-to-end
latency.

Modes:
  standalone  L1 front/back only on CUDA device 0.
  same        L1 and NRx on device 0, using distinct CUDA streams.
  p2p         L1 on device 0 and NRx on device 1, using cudaMemcpyPeerAsync.

The runner exposes either one 4g MIG (same) or two 2g MIGs (p2p).  Both overlap
modes use the same FP16 TensorRT engine, tensor contract, warm-up count, ring
depth, and wall/CUDA timing code.
"""

import argparse
import datetime
import json
import os
import queue
import sys
import threading
import time
import traceback

import cupy as cp
import numpy as np
from cuda.bindings import runtime as cudart

sys.path.insert(0, "/opt/nvidia/cuBB/pyaerial/src")

from aerial.phy5g.algorithms import ChannelEstimator, TrtEngine, TrtTensorPrms
from aerial.phy5g.config import PuschConfig, PuschUeConfig
from aerial.phy5g.ldpc import (
    CrcChecker,
    LdpcDecoder,
    LdpcDeRateMatch,
    get_mcs,
    get_tb_size,
)
from aerial.util.cuda import get_cuda_stream


NUM_PRBS = 273
NUM_SC = NUM_PRBS * 12
NUM_RX_ANT = 4
START_SYM = 2
NUM_SYMBOLS = 12
RX_SHAPE = (1, NUM_SC, NUM_SYMBOLS, NUM_RX_ANT)
CE_SHAPE = (1, 4914, 1, NUM_RX_ANT)
LLR_SHAPE = (8, 1, NUM_SC, NUM_SYMBOLS)
RX_ELEMS = int(np.prod(RX_SHAPE))
CE_ELEMS = int(np.prod(CE_SHAPE))
LLR_ELEMS = int(np.prod(LLR_SHAPE))
FWD_ELEMS = RX_ELEMS * 2 + CE_ELEMS * 2
FWD_BYTES = FWD_ELEMS * np.dtype(np.float32).itemsize
BWD_BYTES = LLR_ELEMS * np.dtype(np.float32).itemsize
DEFAULT_WARMUP = 20
DEFAULT_ITERATIONS = 30
DEFAULT_RING_DEPTH = 2
QUEUE_TIMEOUT_S = 600.0

assert FWD_BYTES == 1_415_232, FWD_BYTES
assert BWD_BYTES == 1_257_984, BWD_BYTES


def make_pusch_config():
    dmrs_syms = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    mod_order, code_rate = get_mcs(2, 1)
    tb_size = get_tb_size(
        mod_order=mod_order,
        code_rate=code_rate,
        dmrs_syms=dmrs_syms,
        num_prbs=NUM_PRBS,
        start_sym=START_SYM,
        num_symbols=NUM_SYMBOLS,
        num_layers=1,
    )
    ue = PuschUeConfig(
        scid=0,
        layers=1,
        dmrs_ports=1,
        rnti=1234,
        data_scid=0,
        mcs_table=0,
        mcs_index=2,
        code_rate=int(code_rate * 10),
        mod_order=mod_order,
        tb_size=tb_size // 8,
    )
    config = PuschConfig(
        ue_configs=[ue],
        num_dmrs_cdm_grps_no_data=2,
        dmrs_scrm_id=41,
        start_prb=0,
        num_prbs=NUM_PRBS,
        dmrs_syms=dmrs_syms,
        dmrs_max_len=1,
        dmrs_add_ln_pos=2,
        start_sym=START_SYM,
        num_symbols=NUM_SYMBOLS,
    )
    data_mask = np.asarray(
        [dmrs_syms[START_SYM + index] == 0 for index in range(NUM_SYMBOLS)],
        dtype=bool,
    )
    return [config], np.where(data_mask)[0]


PUSCH_CONFIGS, DATA_SYMBOL_INDICES = make_pusch_config()


def percentile_summary(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def cuda_timed(device, external_stream, function):
    with cp.cuda.Device(device), external_stream:
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        result = function()
        end.record()
        end.synchronize()
        elapsed_ms = float(cp.cuda.get_elapsed_time(start, end))
    return elapsed_ms, result


def flat_section(flat, start, count, shape):
    view = flat[start:start + count].reshape(shape, order="C")
    if (
        view.dtype != cp.float32
        or view.size != count
        or not view.flags.c_contiguous
    ):
        raise RuntimeError(f"invalid staging section shape={shape}")
    return view


class L1Ops:
    def __init__(self, device):
        self.device = device
        with cp.cuda.Device(device):
            cudart.cudaSetDevice(device)
            self.stream_handle = get_cuda_stream()
            self.stream = cp.cuda.ExternalStream(int(self.stream_handle))
            self.ch_est = ChannelEstimator(
                num_rx_ant=NUM_RX_ANT,
                ch_est_algo=3,
                cuda_stream=self.stream_handle,
            )
            self.derm = LdpcDeRateMatch(
                enable_scrambling=True, cuda_stream=self.stream_handle)
            self.decoder = LdpcDecoder(cuda_stream=self.stream_handle)
            self.crc = CrcChecker(cuda_stream=self.stream_handle)
            self.rx_slot = cp.asarray(
                (
                    np.random.randn(NUM_SC, 14, NUM_RX_ANT)
                    + 1j * np.random.randn(NUM_SC, 14, NUM_RX_ANT)
                ).astype(np.complex64),
                order="F",
            )
            self.data_symbol_indices = cp.asarray(DATA_SYMBOL_INDICES)

    def front(self, fwd_flat):
        def work():
            h_est = self.ch_est.estimate(
                rx_slot=self.rx_slot, slot=0, pusch_configs=PUSCH_CONFIGS)
            rx_window = self.rx_slot[
                None, :, START_SYM:START_SYM + NUM_SYMBOLS, :]
            h0 = h_est[0]
            ce_input = cp.transpose(h0, (0, 3, 1, 2)).reshape(
                h0.shape[0] * h0.shape[3], h0.shape[1], h0.shape[2]
            )[None, ...]
            if rx_window.shape != RX_SHAPE or ce_input.shape != CE_SHAPE:
                raise RuntimeError(
                    f"front shape mismatch rx={rx_window.shape} ce={ce_input.shape}")
            offset = 0
            cp.copyto(flat_section(
                fwd_flat, offset, RX_ELEMS, RX_SHAPE), rx_window.real)
            offset += RX_ELEMS
            cp.copyto(flat_section(
                fwd_flat, offset, RX_ELEMS, RX_SHAPE), rx_window.imag)
            offset += RX_ELEMS
            cp.copyto(flat_section(
                fwd_flat, offset, CE_ELEMS, CE_SHAPE), ce_input.real)
            offset += CE_ELEMS
            cp.copyto(flat_section(
                fwd_flat, offset, CE_ELEMS, CE_SHAPE), ce_input.imag)
            return None

        elapsed_ms, _ = cuda_timed(self.device, self.stream, work)
        return elapsed_ms

    def back(self, llr_flat):
        def work():
            llrs_full = llr_flat.reshape(LLR_SHAPE, order="C")
            llrs = cp.take(llrs_full, self.data_symbol_indices, axis=3)
            coded = self.derm.derate_match(
                input_llrs=[llrs], pusch_configs=PUSCH_CONFIGS)
            blocks = self.decoder.decode(
                input_llrs=coded, pusch_configs=PUSCH_CONFIGS)
            self.crc.check_crc(input_bits=blocks, pusch_configs=PUSCH_CONFIGS)
            return None

        elapsed_ms, _ = cuda_timed(self.device, self.stream, work)
        return elapsed_ms


class NRxOps:
    def __init__(self, device):
        self.device = device
        with cp.cuda.Device(device):
            cudart.cudaSetDevice(device)
            self.stream_handle = get_cuda_stream()
            self.stream = cp.cuda.ExternalStream(int(self.stream_handle))
            engine_path = "/tmp/nrx_p2p_fp16.trt"
            onnx_path = "/opt/nvidia/cuBB/pyaerial/models/neural_rx.onnx"
            if not os.path.exists(engine_path):
                print(f"[P2P-BENCH] building FP16 TRT engine on device {device}", flush=True)
                command = (
                    f"CUDA_VISIBLE_DEVICES={device} trtexec "
                    f"--onnx={onnx_path} --saveEngine={engine_path} "
                    f"--skipInference --fp16 --memPoolSize=workspace:4096 "
                    f"--inputIOFormats="
                    f"fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,int32:chw,int32:chw "
                    f"--outputIOFormats=fp32:chw,fp32:chw "
                    f"--shapes=rx_slot_real:1x3276x12x4,"
                    f"rx_slot_imag:1x3276x12x4,"
                    f"h_hat_real:1x4914x1x4,h_hat_imag:1x4914x1x4 "
                    f"> /tmp/trtexec_p2p.log 2>&1"
                )
                if os.system(command) != 0:
                    raise RuntimeError(
                        "FP16 TRT engine build failed; see /tmp/trtexec_p2p.log")
            self.engine = TrtEngine(
                trt_model_file=engine_path,
                max_batch_size=1,
                cuda_stream=self.stream_handle,
                input_tensors=[
                    TrtTensorPrms("rx_slot_real", (3276, 12, 4), np.float32),
                    TrtTensorPrms("rx_slot_imag", (3276, 12, 4), np.float32),
                    TrtTensorPrms("h_hat_real", (4914, 1, 4), np.float32),
                    TrtTensorPrms("h_hat_imag", (4914, 1, 4), np.float32),
                    TrtTensorPrms("active_dmrs_ports", (1,), np.float32),
                    TrtTensorPrms("dmrs_ofdm_pos", (3,), np.int32),
                    TrtTensorPrms("dmrs_subcarrier_pos", (6,), np.int32),
                ],
                output_tensors=[
                    TrtTensorPrms("output_1", (8, 1, 3276, 12), np.float32),
                    TrtTensorPrms("output_2", (1, 3276, 12, 8), np.float32),
                ],
            )
            self.active = cp.ones((1, 1), dtype=cp.float32)
            self.dmrs_ofdm = cp.asarray([[2, 2, 2]], dtype=cp.int32)
            self.dmrs_subcarrier = cp.asarray(
                [[0, 2, 4, 6, 8, 10]], dtype=cp.int32)

    def infer(self, fwd_flat, bwd_staging=None):
        offset = 0
        rx_real = flat_section(fwd_flat, offset, RX_ELEMS, RX_SHAPE)
        offset += RX_ELEMS
        rx_imag = flat_section(fwd_flat, offset, RX_ELEMS, RX_SHAPE)
        offset += RX_ELEMS
        ce_real = flat_section(fwd_flat, offset, CE_ELEMS, CE_SHAPE)
        offset += CE_ELEMS
        ce_imag = flat_section(fwd_flat, offset, CE_ELEMS, CE_SHAPE)

        def work():
            output = self.engine.run({
                "rx_slot_real": rx_real,
                "rx_slot_imag": rx_imag,
                "h_hat_real": ce_real,
                "h_hat_imag": ce_imag,
                "active_dmrs_ports": self.active,
                "dmrs_ofdm_pos": self.dmrs_ofdm,
                "dmrs_subcarrier_pos": self.dmrs_subcarrier,
            })
            llrs = cp.ravel(output["output_1"], order="C")
            if llrs.dtype != cp.float32 or llrs.size != LLR_ELEMS:
                raise RuntimeError(
                    f"invalid NRx output shape={llrs.shape} dtype={llrs.dtype}")
            if bwd_staging is not None:
                cp.copyto(bwd_staging, llrs)
                return bwd_staging
            return llrs

        return cuda_timed(self.device, self.stream, work)


class P2PCopier:
    def __init__(self, src_device, dst_device):
        self.src_device = src_device
        self.dst_device = dst_device
        with cp.cuda.Device(src_device):
            self.stream = cp.cuda.Stream(non_blocking=True)

    def copy(self, destination, source):
        if destination.nbytes != source.nbytes:
            raise RuntimeError("P2P copy size mismatch")
        started = time.perf_counter_ns()
        with cp.cuda.Device(self.src_device):
            cp.cuda.runtime.memcpyPeerAsync(
                destination.data.ptr,
                self.dst_device,
                source.data.ptr,
                self.src_device,
                source.nbytes,
                self.stream.ptr,
            )
            self.stream.synchronize()
        return (time.perf_counter_ns() - started) / 1e3


def enable_peer_access(first, second):
    for source, destination in ((first, second), (second, first)):
        if not cp.cuda.runtime.deviceCanAccessPeer(source, destination):
            raise RuntimeError(
                f"CUDA P2P unavailable from device {source} to {destination}")
        with cp.cuda.Device(source):
            try:
                cp.cuda.runtime.deviceEnablePeerAccess(destination)
            except cp.cuda.runtime.CUDARuntimeError as error:
                already_enabled = getattr(
                    cp.cuda.runtime, "errorPeerAccessAlreadyEnabled", 704)
                if error.status != already_enabled:
                    raise


def allocate_ring(mode, ring_depth, l1_device, nrx_device):
    with cp.cuda.Device(l1_device):
        l1_forward = [cp.empty(FWD_ELEMS, dtype=cp.float32)
                      for _ in range(ring_depth)]
        l1_backward = [cp.empty(LLR_ELEMS, dtype=cp.float32)
                       for _ in range(ring_depth)]
    if mode == "p2p":
        with cp.cuda.Device(nrx_device):
            nrx_forward = [cp.empty(FWD_ELEMS, dtype=cp.float32)
                           for _ in range(ring_depth)]
            nrx_backward = [cp.empty(LLR_ELEMS, dtype=cp.float32)
                            for _ in range(ring_depth)]
    else:
        nrx_forward = l1_forward
        nrx_backward = [None] * ring_depth
    return l1_forward, l1_backward, nrx_forward, nrx_backward


def run_standalone(label, iterations, warmup):
    l1 = L1Ops(0)
    with cp.cuda.Device(0):
        forward = cp.empty(FWD_ELEMS, dtype=cp.float32)
        fixed_llrs = cp.zeros(LLR_ELEMS, dtype=cp.float32)
    for _ in range(warmup):
        l1.front(forward)
        l1.back(fixed_llrs)

    records = []
    wall_start = time.perf_counter_ns()
    for sequence in range(1, iterations + 1):
        started = time.perf_counter_ns()
        front_ms = l1.front(forward)
        back_ms = l1.back(fixed_llrs)
        completed = time.perf_counter_ns()
        records.append({
            "sequence": sequence,
            "l1_front_ms": front_ms,
            "l1_back_ms": back_ms,
            "l1_active_ms": front_ms + back_ms,
            "e2e_ms": (completed - started) / 1e6,
            "completed_ns": completed,
        })
    wall_end = time.perf_counter_ns()
    return make_result(
        label, "standalone", records, iterations, warmup, 1,
        wall_start, wall_end)


def run_overlap(label, mode, iterations, warmup, ring_depth):
    if mode not in ("same", "p2p"):
        raise ValueError(mode)
    device_count = cp.cuda.runtime.getDeviceCount()
    expected_devices = 2 if mode == "p2p" else 1
    if device_count != expected_devices:
        raise RuntimeError(
            f"mode={mode} requires {expected_devices} visible CUDA devices, "
            f"found {device_count}")

    l1_device = 0
    nrx_device = 1 if mode == "p2p" else 0
    if mode == "p2p":
        enable_peer_access(l1_device, nrx_device)
    l1 = L1Ops(l1_device)
    nrx = NRxOps(nrx_device)
    l1_fwd, l1_bwd, nrx_fwd, nrx_bwd = allocate_ring(
        mode, ring_depth, l1_device, nrx_device)
    fwd_copier = P2PCopier(l1_device, nrx_device) if mode == "p2p" else None
    bwd_copier = P2PCopier(nrx_device, l1_device) if mode == "p2p" else None

    # Serial warm-up is outside all reported timing.
    for _ in range(warmup):
        l1.front(l1_fwd[0])
        if mode == "p2p":
            fwd_copier.copy(nrx_fwd[0], l1_fwd[0])
        _, llrs = nrx.infer(nrx_fwd[0], nrx_bwd[0])
        if mode == "p2p":
            bwd_copier.copy(l1_bwd[0], nrx_bwd[0])
            llrs = l1_bwd[0]
        l1.back(llrs)

    forward_queue = queue.Queue(maxsize=ring_depth)
    result_queue = queue.Queue(maxsize=ring_depth)
    stop_token = object()

    def nrx_worker():
        try:
            with cp.cuda.Device(nrx_device):
                while True:
                    item = forward_queue.get(timeout=QUEUE_TIMEOUT_S)
                    if item is stop_token:
                        return
                    sequence, buffer_id, submitted_ns, front_ms, fwd_us = item
                    nrx_started_ns = time.perf_counter_ns()
                    nrx_ms, llrs = nrx.infer(
                        nrx_fwd[buffer_id], nrx_bwd[buffer_id])
                    bwd_us = 0.0
                    if mode == "p2p":
                        bwd_us = bwd_copier.copy(
                            l1_bwd[buffer_id], nrx_bwd[buffer_id])
                        llrs = l1_bwd[buffer_id]
                    result_queue.put({
                        "sequence": sequence,
                        "buffer_id": buffer_id,
                        "submitted_ns": submitted_ns,
                        "front_ms": front_ms,
                        "fwd_us": fwd_us,
                        "nrx_ms": nrx_ms,
                        "nrx_started_ns": nrx_started_ns,
                        "nrx_finished_ns": time.perf_counter_ns(),
                        "bwd_us": bwd_us,
                        "llrs": llrs,
                    }, timeout=QUEUE_TIMEOUT_S)
        except BaseException as error:  # propagate worker failure to main thread
            result_queue.put({
                "error": repr(error),
                "traceback": traceback.format_exc(),
            })

    worker = threading.Thread(target=nrx_worker, name="nrx-worker", daemon=True)
    worker.start()

    def submit(sequence, buffer_id):
        submitted_ns = time.perf_counter_ns()
        front_ms = l1.front(l1_fwd[buffer_id])
        fwd_us = 0.0
        if mode == "p2p":
            fwd_us = fwd_copier.copy(nrx_fwd[buffer_id], l1_fwd[buffer_id])
        forward_queue.put(
            (sequence, buffer_id, submitted_ns, front_ms, fwd_us),
            timeout=QUEUE_TIMEOUT_S,
        )

    records = []
    next_sequence = 1
    wall_start = time.perf_counter_ns()
    try:
        for buffer_id in range(min(ring_depth, iterations)):
            submit(next_sequence, buffer_id)
            next_sequence += 1

        while len(records) < iterations:
            item = result_queue.get(timeout=QUEUE_TIMEOUT_S)
            if "error" in item:
                raise RuntimeError(
                    f"NRx worker failed: {item['error']}\n{item['traceback']}")
            back_ms = l1.back(item["llrs"])
            completed_ns = time.perf_counter_ns()
            front_ms = item["front_ms"]
            records.append({
                "sequence": item["sequence"],
                "buffer_id": item["buffer_id"],
                "l1_front_ms": front_ms,
                "l1_back_ms": back_ms,
                "l1_active_ms": front_ms + back_ms,
                "nrx_ms": item["nrx_ms"],
                "fwd_copy_us": item["fwd_us"],
                "bwd_copy_us": item["bwd_us"],
                "transport_us": item["fwd_us"] + item["bwd_us"],
                "e2e_ms": (completed_ns - item["submitted_ns"]) / 1e6,
                "submitted_ns": item["submitted_ns"],
                "nrx_started_ns": item["nrx_started_ns"],
                "nrx_finished_ns": item["nrx_finished_ns"],
                "completed_ns": completed_ns,
            })
            if next_sequence <= iterations:
                submit(next_sequence, item["buffer_id"])
                next_sequence += 1
    finally:
        try:
            forward_queue.put(stop_token, timeout=5.0)
        except queue.Full:
            pass
        worker.join(timeout=10.0)
    wall_end = time.perf_counter_ns()
    if worker.is_alive():
        raise RuntimeError("NRx worker did not stop")
    records.sort(key=lambda record: record["sequence"])
    return make_result(
        label, mode, records, iterations, warmup, ring_depth,
        wall_start, wall_end)


def make_result(
        label, mode, records, iterations, warmup, ring_depth,
        wall_start, wall_end):
    completion_times = [record["completed_ns"] for record in records]
    completion_intervals_ms = [
        (right - left) / 1e6
        for left, right in zip(completion_times, completion_times[1:])
    ]
    keys = (
        "l1_front_ms",
        "l1_back_ms",
        "l1_active_ms",
        "nrx_ms",
        "fwd_copy_us",
        "bwd_copy_us",
        "transport_us",
        "e2e_ms",
    )
    summaries = {
        key: percentile_summary([
            record[key] for record in records if key in record])
        for key in keys
    }
    wall_seconds = (wall_end - wall_start) / 1e9
    return {
        "label": label,
        "mode": mode,
        "iterations": iterations,
        "warmup": warmup,
        "ring_depth": ring_depth,
        "visible_cuda_devices": cp.cuda.runtime.getDeviceCount(),
        "fwd_bytes": FWD_BYTES,
        "bwd_bytes": BWD_BYTES,
        "wall_seconds": wall_seconds,
        "completion_throughput_slots_s": iterations / wall_seconds,
        "completion_interval_ms": percentile_summary(completion_intervals_ms),
        "metrics": summaries,
        "raw": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("standalone", "same", "p2p"))
    parser.add_argument("label")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--ring-depth", type=int, default=DEFAULT_RING_DEPTH)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0 or args.ring_depth <= 0:
        parser.error("iterations/ring-depth must be positive and warmup non-negative")
    if args.ring_depth > args.iterations:
        parser.error("ring-depth cannot exceed iterations")

    print(
        f"[P2P-BENCH] mode={args.mode} label={args.label} "
        f"devices={cp.cuda.runtime.getDeviceCount()} iterations={args.iterations} "
        f"warmup={args.warmup} ring={args.ring_depth}",
        flush=True,
    )
    if args.mode == "standalone":
        result = run_standalone(args.label, args.iterations, args.warmup)
    else:
        result = run_overlap(
            args.label, args.mode, args.iterations, args.warmup, args.ring_depth)

    output_dir = os.environ.get("RESULTS_DIR", ".")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        output_dir, f"p2p_overlap_{args.label}_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)

    active = result["metrics"]["l1_active_ms"]
    e2e = result["metrics"]["e2e_ms"]
    nrx = result["metrics"]["nrx_ms"]
    transport = result["metrics"]["transport_us"]
    print(
        f"[P2P-BENCH] RESULT label={args.label} mode={args.mode} "
        f"l1_active_mean={active['mean']:.3f}ms "
        f"l1_active_p99={active['p99']:.3f}ms "
        f"e2e_mean={e2e['mean']:.3f}ms "
        f"nrx_mean={nrx['mean'] if nrx['mean'] is not None else float('nan'):.3f}ms "
        f"transport_mean={transport['mean'] if transport['mean'] is not None else 0.0:.2f}us "
        f"throughput={result['completion_throughput_slots_s']:.3f}slot/s",
        flush=True,
    )
    print(f"[P2P-BENCH] saved {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
