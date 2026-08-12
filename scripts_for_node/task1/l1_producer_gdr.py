"""L1 producer using GPU-resident staging and GPUDirect RDMA.

The forward path packs cuPHY tensors into a persistent registered CuPy
allocation.  The backward path consumes a persistent registered CuPy
allocation directly, with no CPU payload bounce in either direction.

Usage: l1_producer_gdr.py <label> <iterations> <channel_tag>
"""
import atexit
import datetime
import json
import os
import sys
import time

import cupy as cp
import numpy as np

sys.path.insert(0, "/opt/nvidia/cuBB/pyaerial/src")

from aerial.phy5g.algorithms import ChannelEstimator
from aerial.phy5g.config import PuschConfig, PuschUeConfig
from aerial.phy5g.ldpc import (
    CrcChecker, LdpcDecoder, LdpcDeRateMatch, get_mcs, get_tb_size,
)
from aerial.util.cuda import get_cuda_stream
from cuda.bindings import runtime as cudart
from gdr_rdma_channel import GdrRdmaEndpoint, flush_gpudirect_writes


LABEL = sys.argv[1] if len(sys.argv) > 1 else "l1_prod_gdr"
ITERATIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
CHANNEL_TAG = sys.argv[3] if len(sys.argv) > 3 else "airan"
NUM_WARMUP = 20
WAIT_TIMEOUT_S = float(os.environ.get("RDMA_SLOT_TIMEOUT_S", "600"))
TERMINATE_SEQ = (1 << 64) - 1

RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- PUSCH config (must match nrx_consumer_gdr.py) --------------------------
num_prbs = 273
num_rx_ant = 4
mcs_index = 2
mod_order, code_rate = get_mcs(mcs_index, 1)
dmrs_syms = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
start_sym = 2
num_symbols = 12
tb_size = get_tb_size(
    mod_order=mod_order, code_rate=code_rate, dmrs_syms=dmrs_syms,
    num_prbs=num_prbs, start_sym=start_sym, num_symbols=num_symbols,
    num_layers=1,
)
NUM_SC = num_prbs * 12

pusch_ue = PuschUeConfig(
    scid=0, layers=1, dmrs_ports=1, rnti=1234, data_scid=0,
    mcs_table=0, mcs_index=mcs_index, code_rate=int(code_rate * 10),
    mod_order=mod_order, tb_size=tb_size // 8,
)
pusch_configs = [PuschConfig(
    ue_configs=[pusch_ue], num_dmrs_cdm_grps_no_data=2, dmrs_scrm_id=41,
    start_prb=0, num_prbs=num_prbs, dmrs_syms=dmrs_syms,
    dmrs_max_len=1, dmrs_add_ln_pos=2,
    start_sym=start_sym, num_symbols=num_symbols,
)]

# --- Registered GPU payload layouts ----------------------------------------
# Forward uses the same logical C-order section serialization as CPU-RDMA.
# The public TrtEngine wrapper on NRx converts these inputs to F-order.
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


def _wait_for_sequence(endpoint, expected):
    deadline = time.monotonic() + WAIT_TIMEOUT_S
    observed = 0
    while time.monotonic() < deadline:
        observed = endpoint.read_sequence()
        if observed == expected:
            return
        if observed > expected:
            raise RuntimeError(
                f"sequence skipped: expected={expected} observed={observed}")
    raise TimeoutError(
        f"timed out waiting for sequence {expected}; observed={observed}")


# Both processes construct forward and then backward, preventing a two-channel
# rendezvous deadlock. Consumer-owned session epochs reject stale info files.
fwd_ep = GdrRdmaEndpoint(
    FWD_DATA_SIZE, tag=f"{CHANNEL_TAG}_fwd", role="prod")
atexit.register(_cleanup_endpoints)
bwd_ep = GdrRdmaEndpoint(
    BWD_DATA_SIZE, tag=f"{CHANNEL_TAG}_bwd", role="cons")

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

# --- GPU init ---------------------------------------------------------------
cudart.cudaSetDevice(0)
stream = get_cuda_stream()
external_stream = cp.cuda.ExternalStream(int(stream))
ch_est = ChannelEstimator(
    num_rx_ant=num_rx_ant, ch_est_algo=3, cuda_stream=stream)
derm = LdpcDeRateMatch(enable_scrambling=True, cuda_stream=stream)
dec = LdpcDecoder(cuda_stream=stream)
crc = CrcChecker(cuda_stream=stream)

rx_slot = cp.asarray(
    (np.random.randn(NUM_SC, 14, num_rx_ant)
     + 1j * np.random.randn(NUM_SC, 14, num_rx_ant)).astype(np.complex64),
    order="F",
)
data_sym_mask = np.array(
    [dmrs_syms[start_sym + k] == 0 for k in range(num_symbols)], dtype=bool)
data_sym_idx = cp.asarray(np.where(data_sym_mask)[0])

print(
    f"[L1-GDR] connected fwd={FWD_DATA_SIZE} B bwd={BWD_DATA_SIZE} B",
    flush=True,
)


def send_slot(seq):
    """Pack cuPHY output on GPU, then publish payload and ordered marker."""
    h_est = ch_est.estimate(
        rx_slot=rx_slot, slot=0, pusch_configs=pusch_configs)
    with external_stream:
        rx_win = rx_slot[None, :, start_sym:start_sym + num_symbols, :]
        h0 = h_est[0]
        ce_in = cp.transpose(h0, (0, 3, 1, 2)).reshape(
            h0.shape[0] * h0.shape[3], h0.shape[1], h0.shape[2]
        )[None, ...]
        if rx_win.shape != RX_SHAPE or ce_in.shape != CE_SHAPE:
            raise ValueError(
                f"forward shape mismatch: rx={rx_win.shape} ce={ce_in.shape}")
        cp.copyto(fwd_rx_re, rx_win.real)
        cp.copyto(fwd_rx_im, rx_win.imag)
        cp.copyto(fwd_ce_re, ce_in.real)
        cp.copyto(fwd_ce_im, ce_in.imag)
    # The NIC must not read the registered source until CUDA packing finishes.
    external_stream.synchronize()
    fwd_ep.write_payload()
    fwd_ep.write_sequence(seq)


def recv_llrs(seq):
    """Wait for NRx's ordered CPU marker and expose its GPU payload view."""
    _wait_for_sequence(bwd_ep, seq)
    flush_gpudirect_writes()
    return bwd_llrs


def run_one_slot(seq):
    send_slot(seq)
    llrs_full = recv_llrs(seq)
    with external_stream:
        llrs = cp.take(llrs_full, data_sym_idx, axis=3)
    coded = derm.derate_match(input_llrs=[llrs], pusch_configs=pusch_configs)
    blocks = dec.decode(input_llrs=coded, pusch_configs=pusch_configs)
    crc.check_crc(input_bits=blocks, pusch_configs=pusch_configs)
    # NRx may reuse/overwrite the remote backward allocation only after L1
    # sends the next forward slot. Ensure every consumer of bwd_llrs is done
    # before that implicit request/response ACK.
    cp.cuda.runtime.deviceSynchronize()


try:
    print("[L1-GDR] warmup...", flush=True)
    for i in range(NUM_WARMUP):
        run_one_slot(i + 1)
    print("[L1-GDR] measuring...", flush=True)

    latencies = []
    base = NUM_WARMUP
    for i in range(ITERATIONS):
        t0 = time.perf_counter_ns()
        run_one_slot(base + i + 1)
        latencies.append((time.perf_counter_ns() - t0) / 1e6)

    arr = np.asarray(latencies)
    result = {
        "label": LABEL,
        "iterations": ITERATIONS,
        "transport": "gpudirect_rdma_staging",
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "miss_1ms": int(np.sum(arr > 1.0)),
        "raw_ms": [float(value) for value in latencies],
    }
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        RESULTS_DIR, f"l1prod_{LABEL}_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print(
        f"[L1-GDR] {LABEL}: mean={arr.mean():.3f}ms "
        f"p95={np.percentile(arr, 95):.3f}ms "
        f"p99={np.percentile(arr, 99):.3f}ms n={ITERATIONS}",
        flush=True,
    )
    print(f"[L1-GDR] saved: {output_path}", flush=True)
finally:
    if fwd_ep is not None:
        try:
            fwd_ep.write_sequence(TERMINATE_SEQ)
        except Exception as exc:
            print(f"[L1-GDR] failed to send shutdown marker: {exc}", flush=True)
    _cleanup_endpoints()
