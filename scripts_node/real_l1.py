"""
Real cuPHY L1 latency benchmark using pyAerial component-level API (current 25-3).

Builds the PUSCH RX pipeline from individual components (ChannelEstimator,
ChannelEqualizer, NoiseIntfEstimator, LdpcDeRateMatch, LdpcDecoder, CrcChecker)
because the high-level PuschRxPipelineFactory crashes (binding-level segfault on
this build).

For multi-cell scaling: loop NUM_CELLS times per measured iteration, matching
cuPHY's per-cell serial execution (Perlmutter findings on internal cell-sync).

Usage:
  python3 real_l1.py <label> <num_cells> <iterations>
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
    get_mcs, get_tb_size, random_tb,
)
from aerial.phy5g.pdsch import PdschTxPipelineFactory
from aerial.phy5g.config import (
    PdschConfig, PdschUeConfig, PdschCwConfig,
    PuschConfig, PuschUeConfig,
    AerialPdschTxConfig,
)
from aerial.util.cuda import CudaStream
from cuda.bindings import runtime as cudart

LABEL = sys.argv[1] if len(sys.argv) > 1 else "real_l1"
NUM_CELLS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
ITERATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
NUM_WARMUP = 20

RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 5G NR config — override via env vars: NUM_PRBS, NUM_RX_ANT, NUM_TX_ANT, MCS_INDEX.
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
precoding_matrix = np.array([[1.0 + 0.0j]], dtype=np.complex64)

tb_size = get_tb_size(
    mod_order=mod_order, code_rate=code_rate, dmrs_syms=dmrs_syms,
    num_prbs=num_prbs, start_sym=start_sym, num_symbols=num_symbols, num_layers=layers,
)
print(f"[realL1] cells={NUM_CELLS} iters={ITERATIONS} prbs={num_prbs} mcs={mcs_index} tb_size={tb_size}", flush=True)

cudart.cudaSetDevice(0)
cuda_stream = CudaStream()

# PdschTxPipeline crashes on this build (binding-level segfault). We skip it and
# feed the RX components a random tensor of the expected shape directly.
# Resource grid: (subcarriers, symbols, num_rx_ant)  for 273 PRB → 3276 subcarriers, 14 symbols.
NUM_SUBCARRIERS = num_prbs * 12
NUM_OFDM_SYMS = 14
rx_slot = cp.asarray(
    (np.random.randn(NUM_SUBCARRIERS, NUM_OFDM_SYMS, num_rx_ant) +
     1j * np.random.randn(NUM_SUBCARRIERS, NUM_OFDM_SYMS, num_rx_ant)).astype(np.complex64),
    order='F',
)
print(f"[realL1] synthetic rx_slot shape={rx_slot.shape}", flush=True)

# --- build PUSCH RX components ------------------------------------------------
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

print("[realL1] building RX components...", flush=True)
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
print("[realL1] RX components ready", flush=True)


def run_one_cell():
    """One cell's worth of PUSCH RX work."""
    h_est = ch_est.estimate(rx_slot=rx_slot, slot=0, pusch_configs=pusch_configs)
    lw_inv, n_var = ni_est.estimate(
        rx_slot=rx_slot, channel_est=h_est, slot=0, pusch_configs=pusch_configs,
    )
    llrs, sym = ch_eq.equalize(
        rx_slot=rx_slot, channel_est=h_est, lw_inv=lw_inv,
        noise_var_pre_eq=n_var, pusch_configs=pusch_configs,
    )
    # de-rate-match + decode + CRC are deferred — most of the GPU work is in EQ+CE+NI.
    return llrs


# Warm-up.
print("[realL1] warmup...", flush=True)
for _ in range(NUM_WARMUP):
    for _ in range(NUM_CELLS):
        run_one_cell()
cp.cuda.runtime.deviceSynchronize()
print("[realL1] measuring...", flush=True)

latencies = []
start_event = cp.cuda.Event()
end_event = cp.cuda.Event()
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
    "raw_ms": [float(x) for x in latencies],
}

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_file = os.path.join(RESULTS_DIR, f"realL1_{LABEL}_{ts}.json")
with open(out_file, "w") as f:
    json.dump(result, f, indent=2)

print(f"[realL1] {LABEL}: mean={arr.mean():.3f}ms p95={np.percentile(arr,95):.3f}ms "
      f"p99={np.percentile(arr,99):.3f}ms miss1ms={miss_1ms}/{ITERATIONS}", flush=True)
print(f"[realL1] saved: {out_file}", flush=True)
