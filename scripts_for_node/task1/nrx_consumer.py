"""NRx consumer process — runs on MIG 3g partition.

Receives each new slot's (rx_slot_window, h_est) over RDMA, runs the
Neural Receiver TRT engine, and publishes LLRs over a return RDMA channel.

Usage: nrx_consumer.py <channel_tag>
"""
import os
import sys
import time
import struct
import atexit

import numpy as np
import cupy as cp

sys.path.insert(0, "/opt/nvidia/cuBB/pyaerial/src")

from aerial.phy5g.algorithms import TrtEngine, TrtTensorPrms
from aerial.util.cuda import get_cuda_stream
from cuda.bindings import runtime as cudart
from rdma_channel import RdmaEndpoint

CHANNEL_TAG = sys.argv[1] if len(sys.argv) > 1 else "airan"

num_prbs = 273
num_rx_ant = 4
num_symbols = 12
NUM_SC = num_prbs * 12
RX_ELEMS = 1 * NUM_SC * num_symbols * num_rx_ant
CE_ELEMS = 1 * 4914 * 1 * num_rx_ant
LLR_ELEMS = 8 * 1 * NUM_SC * num_symbols
F32 = 4

RX_NBYTES = RX_ELEMS * F32
CE_NBYTES = CE_ELEMS * F32
FWD_DATA_SIZE = 2 * RX_NBYTES + 2 * CE_NBYTES
BWD_DATA_SIZE = LLR_ELEMS * F32
SEQ_SIZE = struct.calcsize("!Q")
TERMINATE_SEQ = (1 << 64) - 1
WAIT_TIMEOUT_S = float(os.environ.get("RDMA_SLOT_TIMEOUT_S", "600"))

assert FWD_DATA_SIZE == 1_415_232, FWD_DATA_SIZE
assert BWD_DATA_SIZE == 1_257_984, BWD_DATA_SIZE

fwd_ep = None
bwd_ep = None


def _close_endpoint(ep):
    """Release verbs resources in dependency order and remove our info file."""
    if ep is None:
        return
    role = "prod" if ep.is_producer else "cons"
    try:
        os.unlink(ep._info_path(role))
    except OSError:
        pass
    for resource_name in ("qp", "mr", "cq", "pd", "ctx"):
        close = getattr(getattr(ep, resource_name, None), "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass


def _cleanup_endpoints():
    global fwd_ep, bwd_ep
    _close_endpoint(bwd_ep)
    bwd_ep = None
    _close_endpoint(fwd_ep)
    fwd_ep = None


fwd_ep = RdmaEndpoint(
    buf_size=FWD_DATA_SIZE, tag=f"{CHANNEL_TAG}_fwd", is_producer=False)
atexit.register(_cleanup_endpoints)
bwd_ep = RdmaEndpoint(
    buf_size=BWD_DATA_SIZE, tag=f"{CHANNEL_TAG}_bwd", is_producer=True)

cudart.cudaSetDevice(0)
stream = get_cuda_stream()

trt_file = "/tmp/nrx.trt"
onnx_file = "/opt/nvidia/cuBB/pyaerial/models/neural_rx.onnx"
if not os.path.exists(trt_file):
    print(f"[NRx] building TRT engine ({onnx_file})", flush=True)
    cmd = (f"trtexec --onnx={onnx_file} --saveEngine={trt_file} --skipInference --fp16 "
           f"--memPoolSize=workspace:4096 "
           f"--inputIOFormats=fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,int32:chw,int32:chw "
           f"--outputIOFormats=fp32:chw,fp32:chw "
           f"--shapes=rx_slot_real:1x3276x12x4,rx_slot_imag:1x3276x12x4,"
           f"h_hat_real:1x4914x1x4,h_hat_imag:1x4914x1x4 > /tmp/trtexec.log 2>&1")
    if os.system(cmd) != 0:
        sys.exit("[NRx] trtexec build failed; see /tmp/trtexec.log")

trt_engine = TrtEngine(
    trt_model_file=trt_file, max_batch_size=1, cuda_stream=stream,
    input_tensors=[
        TrtTensorPrms('rx_slot_real', (3276, 12, 4), np.float32),
        TrtTensorPrms('rx_slot_imag', (3276, 12, 4), np.float32),
        TrtTensorPrms('h_hat_real', (4914, 1, 4), np.float32),
        TrtTensorPrms('h_hat_imag', (4914, 1, 4), np.float32),
        TrtTensorPrms('active_dmrs_ports', (1,), np.float32),
        TrtTensorPrms('dmrs_ofdm_pos', (3,), np.int32),
        TrtTensorPrms('dmrs_subcarrier_pos', (6,), np.int32),
    ],
    output_tensors=[
        TrtTensorPrms('output_1', (8, 1, 3276, 12), np.float32),
        TrtTensorPrms('output_2', (1, 3276, 12, 8), np.float32),
    ],
)
active = cp.ones((1, 1), dtype=cp.float32)
dofdm = cp.array([[2, 2, 2]], dtype=cp.int32)
dsub = cp.array([[0, 2, 4, 6, 8, 10]], dtype=cp.int32)

# Warmup: run a dummy inference so the first real slot doesn't pay JIT cost.
dummy_rx = cp.zeros((1, 3276, 12, 4), dtype=cp.float32)
dummy_ce = cp.zeros((1, 4914, 1, 4), dtype=cp.float32)
for _ in range(20):
    trt_engine.run({
        "rx_slot_real": dummy_rx, "rx_slot_imag": dummy_rx,
        "h_hat_real": dummy_ce, "h_hat_imag": dummy_ce,
        "active_dmrs_ports": active, "dmrs_ofdm_pos": dofdm, "dmrs_subcarrier_pos": dsub,
    })
cp.cuda.runtime.deviceSynchronize()

print(f"[NRx] ready; RDMA fwd={FWD_DATA_SIZE} B "
      f"bwd={BWD_DATA_SIZE} B", flush=True)
last_seen = 0
last_progress = time.monotonic()
try:
    while True:
        seq = struct.unpack(
            "!Q", fwd_ep.mr.read(SEQ_SIZE, offset=FWD_DATA_SIZE))[0]
        if seq == TERMINATE_SEQ:
            print("[NRx] shutdown signal", flush=True)
            break
        if seq <= last_seen:
            if time.monotonic() - last_progress >= WAIT_TIMEOUT_S:
                raise TimeoutError(
                    f"timed out waiting for a slot; last seq={last_seen}")
            continue

        # The sequence marker is a separate RDMA WRITE issued only after the
        # payload WRITE completes, so observing it makes this snapshot valid.
        payload = fwd_ep.mr.read(FWD_DATA_SIZE, offset=0)
        if len(payload) != FWD_DATA_SIZE:
            raise ValueError(
                f"forward payload is {len(payload)} B, expected "
                f"{FWD_DATA_SIZE} B")
        rx_re = cp.asarray(np.frombuffer(
            payload, dtype=np.float32, count=RX_ELEMS, offset=0
        ).reshape(1, 3276, 12, 4))
        rx_im = cp.asarray(np.frombuffer(
            payload, dtype=np.float32, count=RX_ELEMS, offset=RX_NBYTES
        ).reshape(1, 3276, 12, 4))
        ce_re = cp.asarray(np.frombuffer(
            payload, dtype=np.float32, count=CE_ELEMS, offset=2 * RX_NBYTES
        ).reshape(1, 4914, 1, 4))
        ce_im = cp.asarray(np.frombuffer(
            payload, dtype=np.float32, count=CE_ELEMS,
            offset=2 * RX_NBYTES + CE_NBYTES
        ).reshape(1, 4914, 1, 4))
        out = trt_engine.run({
            "rx_slot_real": rx_re, "rx_slot_imag": rx_im,
            "h_hat_real": ce_re, "h_hat_imag": ce_im,
            "active_dmrs_ports": active, "dmrs_ofdm_pos": dofdm,
            "dmrs_subcarrier_pos": dsub,
        })
        # Depending on wrapper batching, output_1 is either
        # (1, 8, 1, 3276, 12) or (8, 1, 3276, 12); its flattened layout agrees.
        llrs_np = np.ascontiguousarray(
            cp.asnumpy(out["output_1"]), dtype=np.float32).ravel()
        if llrs_np.size != LLR_ELEMS or llrs_np.nbytes != BWD_DATA_SIZE:
            raise ValueError(
                f"LLR output is {llrs_np.size} elements/{llrs_np.nbytes} B; "
                f"expected {LLR_ELEMS}/{BWD_DATA_SIZE}")
        bwd_ep.mr.write(llrs_np.tobytes(order="C"), BWD_DATA_SIZE, offset=0)
        bwd_ep.rdma_write(0, BWD_DATA_SIZE, 0)
        marker = struct.pack("!Q", seq)
        bwd_ep.mr.write(marker, SEQ_SIZE, offset=BWD_DATA_SIZE)
        bwd_ep.rdma_write(BWD_DATA_SIZE, SEQ_SIZE, BWD_DATA_SIZE)
        last_seen = seq
        last_progress = time.monotonic()
finally:
    _cleanup_endpoints()
