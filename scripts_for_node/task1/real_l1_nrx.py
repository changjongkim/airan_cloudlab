"""
Real cuPHY-integrated Neural Receiver pipeline.

Unlike run_neural_rx_stress.py (dummy random tensors), this script wires the
Neural Receiver into the actual cuPHY PUSCH RX chain following NVIDIA's
example_neural_receiver.ipynb recipe:

  1. cuPHY ChannelEstimator(ch_est_algo=3, LS)  -->  h_est
  2. slice rx_slot[start_sym:start_sym+num_symbols]
  3. NRx TRT engine (feeds real cuPHY h_est + real rx data)  -->  LLRs
  4. drop LLRs at DMRS symbol positions
  5. cuPHY LdpcDeRateMatch --> LdpcDecoder --> CrcChecker  -->  decoded TBs

We measure per-slot end-to-end latency of this REAL pipeline.

Usage:
  python3 real_l1_nrx.py <label> <num_cells> <iterations>
"""
import os
import sys
import json
import datetime

import numpy as np
import cupy as cp

sys.path.insert(0, "/opt/nvidia/cuBB/pyaerial/src")

from aerial.phy5g.algorithms import (
    ChannelEstimator, TrtEngine, TrtTensorPrms,
)
from aerial.phy5g.ldpc import (
    LdpcDeRateMatch, LdpcDecoder, CrcChecker,
    get_mcs, get_tb_size,
)
from aerial.phy5g.config import PuschConfig, PuschUeConfig
from aerial.util.cuda import get_cuda_stream
from cuda.bindings import runtime as cudart

LABEL = sys.argv[1] if len(sys.argv) > 1 else "real_l1_nrx"
NUM_CELLS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
ITERATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
NUM_WARMUP = 20
RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 5G NR PUSCH config (single UE, 273 PRB, MCS 2, rank 1, 4 rx antennas).
cell_id = 41
num_tx_ant = int(os.environ.get("NUM_TX_ANT", "4"))
num_rx_ant = int(os.environ.get("NUM_RX_ANT", "4"))
mcs_index = int(os.environ.get("MCS_INDEX", "2"))
mcs_table = 0
mod_order, code_rate = get_mcs(mcs_index, 1)
layers = 1
dmrs_ports = 1
dmrs_scrm_id = 41
rnti = 1234
scid = 0
data_scid = 0
num_dmrs_cdm_grps_no_data = 2
start_prb = 0
num_prbs = int(os.environ.get("NUM_PRBS", "273"))
start_sym = 2
num_symbols = 12
# 3 DMRS positions (primary + 2 additional) to match the pretrained neural_rx
# model which expects h_hat of shape (1, 4914, 1, 4) where 4914 = 273 PRB *
# 18 = 273 * (6 pilots/sym * 3 DMRS symbols).
dmrs_syms = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
dmrs_max_len = 1
dmrs_add_ln_pos = 2
enable_pusch_tdi = 0
eq_coeff_algo = 1

tb_size = get_tb_size(
    mod_order=mod_order, code_rate=code_rate, dmrs_syms=dmrs_syms,
    num_prbs=num_prbs, start_sym=start_sym, num_symbols=num_symbols, num_layers=layers,
)
print(f"[realL1NRx] cells={NUM_CELLS} iters={ITERATIONS} prbs={num_prbs} "
      f"mcs={mcs_index} tb_size={tb_size} rx_ant={num_rx_ant}", flush=True)

cudart.cudaSetDevice(0)
cuda_stream = get_cuda_stream()

pusch_ue = PuschUeConfig(
    scid=scid, layers=layers, dmrs_ports=dmrs_ports, rnti=rnti, data_scid=data_scid,
    mcs_table=mcs_table, mcs_index=mcs_index, code_rate=int(code_rate * 10),
    mod_order=mod_order, tb_size=tb_size // 8,
)
pusch_configs = [PuschConfig(
    ue_configs=[pusch_ue], num_dmrs_cdm_grps_no_data=num_dmrs_cdm_grps_no_data,
    dmrs_scrm_id=dmrs_scrm_id, start_prb=start_prb, num_prbs=num_prbs,
    dmrs_syms=dmrs_syms, dmrs_max_len=dmrs_max_len, dmrs_add_ln_pos=dmrs_add_ln_pos,
    start_sym=start_sym, num_symbols=num_symbols,
)]

# --- Build NRx TRT engine from ONNX ------------------------------------------
onnx_file = "/opt/nvidia/cuBB/pyaerial/models/neural_rx.onnx"
trt_file = "/tmp/neural_rx.trt"
if not os.path.exists(trt_file):
    print(f"[realL1NRx] Building TRT engine from {onnx_file}", flush=True)
    cmd = (
        f"trtexec --onnx={onnx_file} --saveEngine={trt_file} --skipInference "
        f"--inputIOFormats=fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,int32:chw,int32:chw "
        f"--outputIOFormats=fp32:chw,fp32:chw "
        f"--shapes=rx_slot_real:1x3276x12x4,rx_slot_imag:1x3276x12x4,"
        f"h_hat_real:1x4914x1x4,h_hat_imag:1x4914x1x4 "
        f"> /tmp/trtexec.log 2>&1"
    )
    ret = os.system(cmd)
    if ret != 0:
        sys.exit(f"[realL1NRx] trtexec failed ({ret}); see /tmp/trtexec.log")
    print("[realL1NRx] TRT engine built.", flush=True)

trt_engine = TrtEngine(
    trt_model_file=trt_file,
    max_batch_size=1,
    cuda_stream=cuda_stream,
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
    ]
)

# --- cuPHY receiver components ----------------------------------------------
# LS estimation (algo=3) — matches what NRx model was trained against.
ch_est = ChannelEstimator(num_rx_ant=num_rx_ant, ch_est_algo=3, cuda_stream=cuda_stream)
derm = LdpcDeRateMatch(enable_scrambling=True, cuda_stream=cuda_stream)
dec = LdpcDecoder(cuda_stream=cuda_stream)
crc = CrcChecker(cuda_stream=cuda_stream)
print("[realL1NRx] cuPHY components + TRT engine ready", flush=True)

# --- Synthetic rx_slot -------------------------------------------------------
# Real deployment would receive rx_slot from a fronthaul frame; we substitute
# random IQ of the same shape so we can measure pipeline latency without a live
# transmitter. Task 2 will feed rx_slot over NIC RDMA instead of memory.
NUM_SC = num_prbs * 12
rx_slot = cp.asarray(
    (np.random.randn(NUM_SC, 14, num_rx_ant) +
     1j * np.random.randn(NUM_SC, 14, num_rx_ant)).astype(np.complex64),
    order='F',
)
print(f"[realL1NRx] synthetic rx_slot shape={rx_slot.shape}", flush=True)

# --- Neural receiver runtime tensors (GPU-resident to avoid CPU round-trip) --
# TrtEngine accepts cupy arrays; using them keeps data on GPU through the whole
# NRx invocation. The small metadata tensors stay on host.
active_dmrs_ports_cp = cp.ones((1, 1), dtype=cp.float32)
dmrs_ofdm_pos_cp = cp.array([[2, 2, 2]], dtype=cp.int32)
dmrs_subcarrier_pos_cp = cp.array([[0, 2, 4, 6, 8, 10]], dtype=cp.int32)
data_sym_mask = np.array(
    [dmrs_syms[start_sym + k] == 0 for k in range(num_symbols)],
    dtype=bool,
)
data_sym_idx = cp.asarray(np.where(data_sym_mask)[0])


def run_one_slot():
    """One PUSCH slot through the FULL cuPHY-integrated NRx pipeline (GPU-resident)."""
    # 1. cuPHY LS channel estimation.
    h_est = ch_est.estimate(rx_slot=rx_slot, slot=0, pusch_configs=pusch_configs)

    # 2. Slice data-symbol window (stays on GPU).
    rx_slot_in = rx_slot[None, :, start_sym:start_sym + num_symbols, :]  # (1,3276,12,4)

    # 3. Reshape cuPHY h_est to NRx expected layout on GPU.
    h0 = h_est[0]                                               # cupy (rx, L, sc, DMRS)
    ch_est_in = cp.transpose(h0, (0, 3, 1, 2))
    ch_est_in = ch_est_in.reshape(h0.shape[0] * h0.shape[3], h0.shape[1], h0.shape[2])
    ch_est_in = ch_est_in[None, ...]                            # (1, 4914, 1, 4)

    # 4. TRT inference (all inputs cupy -> outputs stay cupy).
    outputs = trt_engine.run({
        "rx_slot_real": cp.ascontiguousarray(rx_slot_in.real).astype(cp.float32),
        "rx_slot_imag": cp.ascontiguousarray(rx_slot_in.imag).astype(cp.float32),
        "h_hat_real": cp.ascontiguousarray(ch_est_in.real).astype(cp.float32),
        "h_hat_imag": cp.ascontiguousarray(ch_est_in.imag).astype(cp.float32),
        "active_dmrs_ports": active_dmrs_ports_cp,
        "dmrs_ofdm_pos": dmrs_ofdm_pos_cp,
        "dmrs_subcarrier_pos": dmrs_subcarrier_pos_cp,
    })

    # 5. Drop LLRs at DMRS symbol positions (on GPU).
    llrs = cp.take(outputs["output_1"][0, ...], data_sym_idx, axis=3)

    # 6. cuPHY LDPC de-rate-match + decode + CRC (cupy input accepted).
    coded = derm.derate_match(input_llrs=[llrs], pusch_configs=pusch_configs)
    blocks = dec.decode(input_llrs=coded, pusch_configs=pusch_configs)
    tbs, _ = crc.check_crc(input_bits=blocks, pusch_configs=pusch_configs)
    return tbs


# --- Warm-up -----------------------------------------------------------------
print("[realL1NRx] warmup...", flush=True)
for _ in range(NUM_WARMUP):
    for _ in range(NUM_CELLS):
        run_one_slot()
cp.cuda.runtime.deviceSynchronize()
print("[realL1NRx] measuring...", flush=True)

# --- Measure -----------------------------------------------------------------
latencies = []
start_event = cp.cuda.Event()
end_event = cp.cuda.Event()
for _ in range(ITERATIONS):
    start_event.record()
    for _ in range(NUM_CELLS):
        run_one_slot()
    end_event.record()
    end_event.synchronize()
    latencies.append(cp.cuda.get_elapsed_time(start_event, end_event))

arr = np.array(latencies)
miss_1ms = int(np.sum(arr > 1.0))
result = {
    "label": LABEL,
    "num_cells": NUM_CELLS,
    "iterations": ITERATIONS,
    "num_rx_ant": num_rx_ant,
    "num_prbs": num_prbs,
    "mcs_index": mcs_index,
    "mean_ms": float(arr.mean()),
    "p50_ms": float(np.percentile(arr, 50)),
    "p95_ms": float(np.percentile(arr, 95)),
    "p99_ms": float(np.percentile(arr, 99)),
    "min_ms": float(arr.min()),
    "max_ms": float(arr.max()),
    "miss_1ms": miss_1ms,
    "raw_ms": [float(x) for x in latencies],
}
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_file = os.path.join(RESULTS_DIR, f"realL1NRx_{LABEL}_{ts}.json")
with open(out_file, "w") as f:
    json.dump(result, f, indent=2)

print(f"[realL1NRx] {LABEL}: mean={arr.mean():.3f}ms "
      f"p95={np.percentile(arr,95):.3f}ms p99={np.percentile(arr,99):.3f}ms "
      f"miss1ms={miss_1ms}/{ITERATIONS}", flush=True)
print(f"[realL1NRx] saved: {out_file}", flush=True)
