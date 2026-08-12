"""L1 producer process — runs on MIG 4g partition.

Simulates the L1 side of a modular AI-RAN split:
  1. cuPHY LS channel estimation on rx_slot.
  2. RDMA WRITE (rx_slot data window, h_est) to NRx.
  3. Receive LLRs from NRx over a second RDMA channel.
  4. cuPHY LDPC de-rate-match + decode + CRC.

Usage: l1_producer.py <label> <iterations> <channel_tag>
"""
import os
import sys
import json
import time
import datetime
import struct
import atexit

import numpy as np
import cupy as cp

sys.path.insert(0, "/opt/nvidia/cuBB/pyaerial/src")

from aerial.phy5g.algorithms import ChannelEstimator
from aerial.phy5g.ldpc import (
    LdpcDeRateMatch, LdpcDecoder, CrcChecker, get_mcs, get_tb_size,
)
from aerial.phy5g.config import PuschConfig, PuschUeConfig
from aerial.util.cuda import get_cuda_stream
from cuda.bindings import runtime as cudart
from rdma_channel import RdmaEndpoint

LABEL = sys.argv[1] if len(sys.argv) > 1 else "l1_prod"
ITERATIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
CHANNEL_TAG = sys.argv[3] if len(sys.argv) > 3 else "airan"
NUM_WARMUP = 20

RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- PUSCH config (must match nrx_consumer.py) -------------------------------
num_prbs = 273
num_rx_ant = 4
mcs_index = 2
mod_order, code_rate = get_mcs(mcs_index, 1)
dmrs_syms = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
start_sym = 2
num_symbols = 12
tb_size = get_tb_size(
    mod_order=mod_order, code_rate=code_rate, dmrs_syms=dmrs_syms,
    num_prbs=num_prbs, start_sym=start_sym, num_symbols=num_symbols, num_layers=1,
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

# --- RDMA channel layout -----------------------------------------------------
# Each direction has one fixed-size payload followed by an 8-byte sequence
# marker. RdmaEndpoint allocates the marker in addition to buf_size.
# rx_slot_window: (1, 3276, 12, 4) complex64 -> real+imag as float32 pair.
# h_hat:          (1, 4914, 1, 4) complex64 -> real+imag as float32 pair.
# llrs (return):  (8, 1, 3276, 12) float32.
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

# These values also cross-check the deployed NRx model's fixed tensor shapes.
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


# Construct both directions in the same order in both processes. The first
# handshake is L1 producer↔NRx consumer; the second reverses the roles.
fwd_ep = RdmaEndpoint(
    buf_size=FWD_DATA_SIZE, tag=f"{CHANNEL_TAG}_fwd", is_producer=True)
atexit.register(_cleanup_endpoints)
bwd_ep = RdmaEndpoint(
    buf_size=BWD_DATA_SIZE, tag=f"{CHANNEL_TAG}_bwd", is_producer=False)

# --- GPU init ---------------------------------------------------------------
cudart.cudaSetDevice(0)
stream = get_cuda_stream()
ch_est = ChannelEstimator(num_rx_ant=num_rx_ant, ch_est_algo=3, cuda_stream=stream)
derm = LdpcDeRateMatch(enable_scrambling=True, cuda_stream=stream)
dec = LdpcDecoder(cuda_stream=stream)
crc = CrcChecker(cuda_stream=stream)

rx_slot = cp.asarray(
    (np.random.randn(NUM_SC, 14, num_rx_ant) +
     1j * np.random.randn(NUM_SC, 14, num_rx_ant)).astype(np.complex64),
    order='F',
)
data_sym_mask = np.array([dmrs_syms[start_sym + k] == 0 for k in range(num_symbols)],
                         dtype=bool)
data_sym_idx = cp.asarray(np.where(data_sym_mask)[0])

print(f"[L1] RDMA connected fwd={FWD_DATA_SIZE} B "
      f"bwd={BWD_DATA_SIZE} B", flush=True)


def send_slot(seq):
    """Compute CE, RDMA WRITE window+h_est, then publish sequence."""
    h_est = ch_est.estimate(rx_slot=rx_slot, slot=0, pusch_configs=pusch_configs)
    rx_win = rx_slot[None, :, start_sym:start_sym + num_symbols, :]
    h0 = h_est[0]
    ce_in = cp.transpose(h0, (0, 3, 1, 2)).reshape(
        h0.shape[0] * h0.shape[3], h0.shape[1], h0.shape[2])[None, ...]
    arrays = (
        cp.asnumpy(cp.ascontiguousarray(rx_win.real).astype(cp.float32)),
        cp.asnumpy(cp.ascontiguousarray(rx_win.imag).astype(cp.float32)),
        cp.asnumpy(cp.ascontiguousarray(ce_in.real).astype(cp.float32)),
        cp.asnumpy(cp.ascontiguousarray(ce_in.imag).astype(cp.float32)),
    )
    payload = b"".join(a.tobytes(order="C") for a in arrays)
    if len(payload) != FWD_DATA_SIZE:
        raise ValueError(
            f"forward payload is {len(payload)} B, expected {FWD_DATA_SIZE} B")
    fwd_ep.mr.write(payload, FWD_DATA_SIZE, offset=0)
    fwd_ep.rdma_write(0, FWD_DATA_SIZE, 0)
    marker = struct.pack("!Q", seq)
    fwd_ep.mr.write(marker, SEQ_SIZE, offset=FWD_DATA_SIZE)
    fwd_ep.rdma_write(FWD_DATA_SIZE, SEQ_SIZE, FWD_DATA_SIZE)


def recv_llrs(seq):
    """Spin-wait for consumer to publish LLRs for `seq`, return cupy tensor."""
    deadline = time.monotonic() + WAIT_TIMEOUT_S
    while True:
        received = struct.unpack(
            "!Q", bwd_ep.mr.read(SEQ_SIZE, offset=BWD_DATA_SIZE))[0]
        if received >= seq:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for LLR seq={seq}; last seq={received}")
    payload = bwd_ep.mr.read(BWD_DATA_SIZE, offset=0)
    if len(payload) != BWD_DATA_SIZE:
        raise ValueError(
            f"backward payload is {len(payload)} B, expected {BWD_DATA_SIZE} B")
    llrs_np = np.frombuffer(payload, dtype=np.float32).reshape(
        8, 1, NUM_SC, num_symbols)
    return cp.asarray(llrs_np)


def run_one_slot(seq):
    send_slot(seq)
    llrs_full = recv_llrs(seq)
    llrs = cp.take(llrs_full, data_sym_idx, axis=3)
    coded = derm.derate_match(input_llrs=[llrs], pusch_configs=pusch_configs)
    blocks = dec.decode(input_llrs=coded, pusch_configs=pusch_configs)
    crc.check_crc(input_bits=blocks, pusch_configs=pusch_configs)


# --- Warmup + measure --------------------------------------------------------
try:
    print("[L1] warmup...", flush=True)
    for i in range(NUM_WARMUP):
        run_one_slot(i + 1)
    print("[L1] measuring...", flush=True)

    latencies = []
    base = NUM_WARMUP
    for i in range(ITERATIONS):
        t0 = time.perf_counter_ns()
        run_one_slot(base + i + 1)
        latencies.append((time.perf_counter_ns() - t0) / 1e6)

    arr = np.array(latencies)
    result = {
        "label": LABEL,
        "iterations": ITERATIONS,
        "transport": "cpu_rdma",
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "miss_1ms": int(np.sum(arr > 1.0)),
        "raw_ms": [float(x) for x in latencies],
    }
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULTS_DIR, f"l1prod_{LABEL}_{ts}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[L1] {LABEL}: mean={arr.mean():.3f}ms "
          f"p95={np.percentile(arr,95):.3f}ms "
          f"p99={np.percentile(arr,99):.3f}ms  n={ITERATIONS}", flush=True)
    print(f"[L1] saved: {out}", flush=True)
finally:
    # Publish shutdown through the same ordered marker path. A completed RDMA
    # WRITE means the marker is visible before verbs resources are released.
    if fwd_ep is not None:
        try:
            marker = struct.pack("!Q", TERMINATE_SEQ)
            fwd_ep.mr.write(marker, SEQ_SIZE, offset=FWD_DATA_SIZE)
            fwd_ep.rdma_write(FWD_DATA_SIZE, SEQ_SIZE, FWD_DATA_SIZE)
        except Exception as exc:
            print(f"[L1] failed to send shutdown marker: {exc}", flush=True)
    _cleanup_endpoints()
