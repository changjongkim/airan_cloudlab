"""Neural Receiver consumer using zero-copy GPUDirect RDMA bindings.

The forward registered GPU allocation is viewed directly as the four TRT
inputs. Caller-owned TensorRT bindings write LLRs directly into the registered
backward allocation. CUDA Graph replay has no layout-conversion or staging
copy, and no payload traverses host memory.

Usage: nrx_consumer_gdr.py <channel_tag>
"""
import atexit
import os
import sys
import time

import cupy as cp
import numpy as np

sys.path.insert(0, "/opt/nvidia/cuBB/pyaerial/src")

from cuda.bindings import runtime as cudart
from gdr_rdma_channel import GdrRdmaEndpoint, flush_gpudirect_writes
from nrx_trt_direct import DirectNrx


CHANNEL_TAG = sys.argv[1] if len(sys.argv) > 1 else "airan"
WAIT_TIMEOUT_S = float(os.environ.get("RDMA_SLOT_TIMEOUT_S", "600"))
TERMINATE_SEQ = (1 << 64) - 1

num_prbs = 273
num_rx_ant = 4
num_symbols = 12
NUM_SC = num_prbs * 12
RX_SHAPE = (1, NUM_SC, num_symbols, num_rx_ant)
CE_SHAPE = (1, 4914, 1, num_rx_ant)
LLR_SHAPE = (2, 1, NUM_SC, num_symbols)
RX_ELEMS = int(np.prod(RX_SHAPE))
CE_ELEMS = int(np.prod(CE_SHAPE))
LLR_ELEMS = int(np.prod(LLR_SHAPE))
F32 = np.dtype(np.float32).itemsize
FWD_DATA_SIZE = 2 * RX_ELEMS * F32 + 2 * CE_ELEMS * F32
BWD_DATA_SIZE = LLR_ELEMS * F32

assert FWD_DATA_SIZE == 1_415_232, FWD_DATA_SIZE
assert BWD_DATA_SIZE == 314_496, BWD_DATA_SIZE

fwd_ep = None
bwd_ep = None


def _cleanup_endpoints():
    global fwd_ep, bwd_ep
    if bwd_ep is not None:
        bwd_ep.close()
        bwd_ep = None
    if fwd_ep is not None:
        fwd_ep.close()
        fwd_ep = None


def _section(flat, start, count, shape, name):
    view = flat[start:start + count].reshape(shape, order="C")
    expected_ptr = int(flat.data.ptr) + start * F32
    if (
        view.shape != shape
        or view.dtype != cp.float32
        or view.size != count
        or view.nbytes != count * F32
        or not view.flags.c_contiguous
        or int(view.data.ptr) != expected_ptr
    ):
        raise RuntimeError(f"invalid registered GPU view: {name}")
    return view


# Endpoint order and roles mirror L1 exactly: forward first, then backward.
fwd_ep = GdrRdmaEndpoint(
    FWD_DATA_SIZE, tag=f"{CHANNEL_TAG}_fwd", role="cons")
atexit.register(_cleanup_endpoints)
bwd_ep = GdrRdmaEndpoint(
    BWD_DATA_SIZE, tag=f"{CHANNEL_TAG}_bwd", role="prod")

fwd_flat = fwd_ep.gpu_array.view(cp.float32)
bwd_flat = bwd_ep.gpu_array.view(cp.float32)
if fwd_flat.nbytes != FWD_DATA_SIZE or bwd_flat.nbytes != BWD_DATA_SIZE:
    raise RuntimeError("GDR endpoint allocation size mismatch")

offset = 0
fwd_rx_re = _section(fwd_flat, offset, RX_ELEMS, RX_SHAPE, "rx_real")
offset += RX_ELEMS
fwd_rx_im = _section(fwd_flat, offset, RX_ELEMS, RX_SHAPE, "rx_imag")
offset += RX_ELEMS
fwd_ce_re = _section(fwd_flat, offset, CE_ELEMS, CE_SHAPE, "ce_real")
offset += CE_ELEMS
fwd_ce_im = _section(fwd_flat, offset, CE_ELEMS, CE_SHAPE, "ce_imag")
offset += CE_ELEMS
assert offset * F32 == FWD_DATA_SIZE

bwd_llrs = _section(bwd_flat, 0, LLR_ELEMS, LLR_SHAPE, "llrs")
bwd_output = bwd_llrs.reshape((1,) + LLR_SHAPE, order="C")

cudart.cudaSetDevice(0)
trt_file = os.environ.get("NRX_ENGINE", "/engines/neural_rx_fp16_4g.trt")
trt_engine = DirectNrx(trt_file)
trt_engine.bind_tensor("rx_slot_real", fwd_rx_re)
trt_engine.bind_tensor("rx_slot_imag", fwd_rx_im)
trt_engine.bind_tensor("h_hat_real", fwd_ce_re)
trt_engine.bind_tensor("h_hat_imag", fwd_ce_im)
trt_engine.bind_tensor("output_1", bwd_output)
trt_engine.capture_graph()
for _ in range(20):
    trt_engine.launch(use_graph=True)
trt_engine.stream.synchronize()

print(
    f"[NRx-GDR] ready; fwd={FWD_DATA_SIZE} B bwd={BWD_DATA_SIZE} B",
    flush=True,
)

last_seen = 0
last_progress = time.monotonic()
try:
    while True:
        seq = fwd_ep.read_sequence()
        if seq == TERMINATE_SEQ:
            print("[NRx-GDR] shutdown signal", flush=True)
            break
        if seq <= last_seen:
            if time.monotonic() - last_progress >= WAIT_TIMEOUT_S:
                raise TimeoutError(
                    f"timed out waiting for a slot; last seq={last_seen}")
            continue
        if seq != last_seen + 1:
            raise RuntimeError(
                f"forward sequence skipped: expected={last_seen + 1} "
                f"observed={seq}")

        # The host marker follows the completed GPU payload WRITE on the same
        # RC QP. Flush peer writes before CUDA consumes the registered views.
        flush_gpudirect_writes()
        trt_engine.launch(use_graph=True)
        trt_engine.stream.synchronize()
        if bwd_output.dtype != cp.float32 or bwd_output.size != LLR_ELEMS:
            raise ValueError(
                f"LLR output mismatch: shape={bwd_output.shape} "
                f"dtype={bwd_output.dtype} size={bwd_output.size}, "
                f"expected size={LLR_ELEMS}")

        # The graph completion guarantees TRT consumed fwd_flat and directly
        # finished bwd_flat before the NIC reads the response payload.
        bwd_ep.write_payload()
        bwd_ep.write_sequence(seq)
        last_seen = seq
        last_progress = time.monotonic()
finally:
    _cleanup_endpoints()
