"""Neural Receiver consumer using staging-based GPUDirect RDMA.

The forward registered GPU allocation is viewed directly as the four TRT
inputs. The public TrtEngine wrapper still performs its internal F-order input
copies. Its LLR output is copied on GPU into a persistent registered backward
staging allocation; no payload traverses host memory.

Usage: nrx_consumer_gdr.py <channel_tag>
"""
import atexit
import os
import sys
import time

import cupy as cp
import numpy as np

sys.path.insert(0, "/opt/nvidia/cuBB/pyaerial/src")

from aerial.phy5g.algorithms import TrtEngine, TrtTensorPrms
from aerial.util.cuda import get_cuda_stream
from cuda.bindings import runtime as cudart
from gdr_rdma_channel import GdrRdmaEndpoint, flush_gpudirect_writes


CHANNEL_TAG = sys.argv[1] if len(sys.argv) > 1 else "airan"
WAIT_TIMEOUT_S = float(os.environ.get("RDMA_SLOT_TIMEOUT_S", "600"))
TERMINATE_SEQ = (1 << 64) - 1

num_prbs = 273
num_rx_ant = 4
num_symbols = 12
NUM_SC = num_prbs * 12
RX_SHAPE = (1, NUM_SC, num_symbols, num_rx_ant)
CE_SHAPE = (1, 4914, 1, num_rx_ant)
LLR_SHAPE = (8, 1, NUM_SC, num_symbols)
RX_ELEMS = int(np.prod(RX_SHAPE))
CE_ELEMS = int(np.prod(CE_SHAPE))
LLR_ELEMS = int(np.prod(LLR_SHAPE))
F32 = np.dtype(np.float32).itemsize
FWD_DATA_SIZE = 2 * RX_ELEMS * F32 + 2 * CE_ELEMS * F32
BWD_DATA_SIZE = LLR_ELEMS * F32

assert FWD_DATA_SIZE == 1_415_232, FWD_DATA_SIZE
assert BWD_DATA_SIZE == 1_257_984, BWD_DATA_SIZE

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

# The public TrtEngine output is flattened in logical C order to preserve the
# CPU-RDMA payload contract. L1 creates the matching C-order LLR view.
bwd_llrs = _section(bwd_flat, 0, LLR_ELEMS, LLR_SHAPE, "llrs")

cudart.cudaSetDevice(0)
stream = get_cuda_stream()
external_stream = cp.cuda.ExternalStream(int(stream))

trt_file = "/tmp/nrx.trt"
onnx_file = "/opt/nvidia/cuBB/pyaerial/models/neural_rx.onnx"
if not os.path.exists(trt_file):
    print(f"[NRx-GDR] building TRT engine ({onnx_file})", flush=True)
    command = (
        f"trtexec --onnx={onnx_file} --saveEngine={trt_file} "
        f"--skipInference --fp16 --memPoolSize=workspace:4096 "
        f"--inputIOFormats="
        f"fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,int32:chw,int32:chw "
        f"--outputIOFormats=fp32:chw,fp32:chw "
        f"--shapes=rx_slot_real:1x3276x12x4,"
        f"rx_slot_imag:1x3276x12x4,"
        f"h_hat_real:1x4914x1x4,h_hat_imag:1x4914x1x4 "
        f"> /tmp/trtexec.log 2>&1"
    )
    if os.system(command) != 0:
        sys.exit("[NRx-GDR] trtexec build failed; see /tmp/trtexec.log")

trt_engine = TrtEngine(
    trt_model_file=trt_file,
    max_batch_size=1,
    cuda_stream=stream,
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
active = cp.ones((1, 1), dtype=cp.float32)
dofdm = cp.array([[2, 2, 2]], dtype=cp.int32)
dsub = cp.array([[0, 2, 4, 6, 8, 10]], dtype=cp.int32)

# Warm the public wrapper and TRT engine before consuming the first slot.
dummy_rx = cp.zeros(RX_SHAPE, dtype=cp.float32)
dummy_ce = cp.zeros(CE_SHAPE, dtype=cp.float32)
for _ in range(20):
    trt_engine.run({
        "rx_slot_real": dummy_rx,
        "rx_slot_imag": dummy_rx,
        "h_hat_real": dummy_ce,
        "h_hat_imag": dummy_ce,
        "active_dmrs_ports": active,
        "dmrs_ofdm_pos": dofdm,
        "dmrs_subcarrier_pos": dsub,
    })
cp.cuda.runtime.deviceSynchronize()

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
        output = trt_engine.run({
            "rx_slot_real": fwd_rx_re,
            "rx_slot_imag": fwd_rx_im,
            "h_hat_real": fwd_ce_re,
            "h_hat_imag": fwd_ce_im,
            "active_dmrs_ports": active,
            "dmrs_ofdm_pos": dofdm,
            "dmrs_subcarrier_pos": dsub,
        })
        llrs = output["output_1"]
        if llrs.dtype != cp.float32 or llrs.size != LLR_ELEMS:
            raise ValueError(
                f"LLR output mismatch: shape={llrs.shape} dtype={llrs.dtype} "
                f"size={llrs.size}, expected size={LLR_ELEMS}")

        # Public TrtEngine has no caller-provided output buffer. Preserve the
        # baseline's logical C flattening and copy only GPU→GPU into the
        # persistent registered backward allocation.
        with external_stream:
            llrs_c_flat = cp.ravel(llrs, order="C")
            if (
                llrs_c_flat.dtype != cp.float32
                or llrs_c_flat.size != LLR_ELEMS
                or llrs_c_flat.nbytes != BWD_DATA_SIZE
                or not llrs_c_flat.flags.c_contiguous
            ):
                raise ValueError("invalid flattened LLR output")
            cp.copyto(bwd_flat, llrs_c_flat)

        # Completion here guarantees both that TRT consumed fwd_flat and that
        # the NIC sees a finished bwd_flat. The backward response is therefore
        # the ACK allowing L1 to reuse its forward staging on the next slot.
        external_stream.synchronize()
        bwd_ep.write_payload()
        bwd_ep.write_sequence(seq)
        last_seen = seq
        last_progress = time.monotonic()
finally:
    _cleanup_endpoints()
