"""
real_l1_graph.py — Approach 1: L1 wrapped in CUDA Graph capture-replay.

Structural difference from real_l1.py:
  - Warmup as normal
  - Capture ONE iteration (NUM_CELLS × run_one_cell) into a graph
  - Instantiate graph_exec once
  - Measurement loop: single graph_exec.launch(stream) per iteration
    → no per-iteration cudaMalloc / cudaFree / kernel-launch API calls
  - Buffer allocations happen ONCE inside capture; graph replays same buffers

Tests whether wrapping the L1 pipeline in a CUDA Graph avoids the
per-frame temporal-sync path that Chain 9 confirmed cannot be
bypassed at the CUDA runtime API level.

Usage:
  python3 real_l1_graph.py <label> <num_cells> <iterations>
"""
import os
import sys
import json
import datetime

import numpy as np
import cupy as cp

from aerial.phy5g.algorithms import (
    ChannelEstimator, ChannelEqualizer, NoiseIntfEstimator,
)
from aerial.phy5g.ldpc import (
    LdpcDeRateMatch, LdpcDecoder, CrcChecker,
    get_mcs, get_tb_size,
)
from aerial.phy5g.config import (
    PuschConfig, PuschUeConfig,
)
from aerial.util.cuda import get_cuda_stream

LABEL = sys.argv[1] if len(sys.argv) > 1 else "real_l1_graph"
NUM_CELLS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
ITERATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
NUM_WARMUP = 20

RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")
os.makedirs(RESULTS_DIR, exist_ok=True)

cell_id = 41
num_tx_ant = int(os.environ.get("NUM_TX_ANT", "4"))
num_rx_ant = int(os.environ.get("NUM_RX_ANT", "4"))
mcs_table = 0
mcs_index = int(os.environ.get("MCS_INDEX", "2"))
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
dmrs_syms = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
dmrs_max_len = 1
dmrs_add_ln_pos = 0
enable_pusch_tdi = 0
eq_coeff_algo = 1

tb_size = get_tb_size(
    mod_order=mod_order, code_rate=code_rate, dmrs_syms=dmrs_syms,
    num_prbs=num_prbs, start_sym=start_sym, num_symbols=num_symbols, num_layers=layers,
)
print(f"[realL1_graph] cells={NUM_CELLS} iters={ITERATIONS} prbs={num_prbs} mcs={mcs_index} tb_size={tb_size}", flush=True)

cuda_stream = get_cuda_stream()

NUM_SUBCARRIERS = num_prbs * 12
NUM_OFDM_SYMS = 14
rx_slot = cp.asarray(
    (np.random.randn(NUM_SUBCARRIERS, NUM_OFDM_SYMS, num_rx_ant) +
     1j * np.random.randn(NUM_SUBCARRIERS, NUM_OFDM_SYMS, num_rx_ant)).astype(np.complex64),
    order='F',
)
print(f"[realL1_graph] rx_slot shape={rx_slot.shape}", flush=True)

pusch_ue_config = PuschUeConfig(
    scid=scid, layers=layers, dmrs_ports=dmrs_ports, rnti=rnti, data_scid=data_scid,
    mcs_table=mcs_table, mcs_index=mcs_index, code_rate=int(code_rate * 10),
    mod_order=mod_order, tb_size=tb_size // 8,
)
pusch_configs = [PuschConfig(
    ue_configs=[pusch_ue_config], num_dmrs_cdm_grps_no_data=num_dmrs_cdm_grps_no_data,
    dmrs_scrm_id=dmrs_scrm_id, start_prb=start_prb, num_prbs=num_prbs,
    dmrs_syms=dmrs_syms, dmrs_max_len=dmrs_max_len, dmrs_add_ln_pos=dmrs_add_ln_pos,
    start_sym=start_sym, num_symbols=num_symbols,
)]

print("[realL1_graph] building RX components...", flush=True)
ch_est = ChannelEstimator(num_rx_ant=num_rx_ant, cuda_stream=cuda_stream)
ch_eq = ChannelEqualizer(
    num_rx_ant=num_rx_ant, enable_pusch_tdi=enable_pusch_tdi,
    eq_coeff_algo=eq_coeff_algo, cuda_stream=cuda_stream,
)
ni_est = NoiseIntfEstimator(
    num_rx_ant=num_rx_ant, eq_coeff_algo=eq_coeff_algo, cuda_stream=cuda_stream,
)
derm = LdpcDeRateMatch(enable_scrambling=True, cuda_stream=cuda_stream)
dec = LdpcDecoder(cuda_stream=cuda_stream)
crc = CrcChecker(cuda_stream=cuda_stream)
print("[realL1_graph] RX components ready", flush=True)


def run_one_cell():
    h_est = ch_est.estimate(rx_slot=rx_slot, slot=0, pusch_configs=pusch_configs)
    lw_inv, n_var = ni_est.estimate(
        rx_slot=rx_slot, channel_est=h_est, slot=0, pusch_configs=pusch_configs,
    )
    llrs, sym = ch_eq.equalize(
        rx_slot=rx_slot, channel_est=h_est, lw_inv=lw_inv,
        noise_var_pre_eq=n_var, pusch_configs=pusch_configs,
    )
    return llrs


# Warm-up.
print("[realL1_graph] warmup...", flush=True)
for _ in range(NUM_WARMUP):
    for _ in range(NUM_CELLS):
        run_one_cell()
cp.cuda.runtime.deviceSynchronize()

# CUDA Graph capture
print("[realL1_graph] capturing graph...", flush=True)
stream = cp.cuda.Stream(non_blocking=True)
use_graph = True
try:
    with stream:
        stream.begin_capture()
        for _ in range(NUM_CELLS):
            run_one_cell()
        graph = stream.end_capture()
    print(f"[realL1_graph] graph captured — nodes={graph.debug_dot_str() if hasattr(graph,'debug_dot_str') else '?'}", flush=True)
except Exception as e:
    print(f"[realL1_graph] GRAPH CAPTURE FAILED: {e} — falling back to eager mode", flush=True)
    use_graph = False

print("[realL1_graph] measuring...", flush=True)
latencies = []
start_event = cp.cuda.Event()
end_event = cp.cuda.Event()

if use_graph:
    for i in range(ITERATIONS):
        start_event.record(stream)
        graph.launch(stream=stream)
        end_event.record(stream)
        end_event.synchronize()
        latencies.append(cp.cuda.get_elapsed_time(start_event, end_event))
else:
    for i in range(ITERATIONS):
        start_event.record()
        for _ in range(NUM_CELLS):
            run_one_cell()
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
    "num_tx_ant": num_tx_ant,
    "num_prbs": num_prbs,
    "mcs_index": mcs_index,
    "mean_ms": float(arr.mean()),
    "p50_ms": float(np.percentile(arr, 50)),
    "p95_ms": float(np.percentile(arr, 95)),
    "p99_ms": float(np.percentile(arr, 99)),
    "min_ms": float(arr.min()),
    "max_ms": float(arr.max()),
    "miss_1ms": miss_1ms,
    "used_graph": use_graph,
    "raw_ms": [float(x) for x in latencies],
}

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_file = os.path.join(RESULTS_DIR, f"realL1_graph_{LABEL}_{ts}.json")
with open(out_file, "w") as f:
    json.dump(result, f, indent=2)

mode_tag = "graph" if use_graph else "eager"
print(f"[realL1_graph] {LABEL} ({mode_tag}): mean={arr.mean():.3f}ms p95={np.percentile(arr,95):.3f}ms "
      f"p99={np.percentile(arr,99):.3f}ms miss1ms={miss_1ms}/{ITERATIONS}", flush=True)
print(f"[realL1_graph] saved: {out_file}", flush=True)
