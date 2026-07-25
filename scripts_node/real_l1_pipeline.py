"""
real_l1_pipeline.py — Approach A: Use pyaerial's high-level PuschRxPipelineFactory
(which the original real_l1.py comment said crashes with segfault).

If it works, this dramatically reduces per-iter API-call count because the
whole pipeline is dispatched via one .run() call rather than N component
calls.

Attempts to use the PuschRxPipelineFactory + AerialPuschRxConfig chain.
Falls back gracefully with a clear message if it segfaults.

Usage:
  python3 real_l1_pipeline.py <label> <num_cells> <iterations>
"""
import os, sys, json, datetime
import numpy as np
import cupy as cp
import traceback

LABEL = sys.argv[1] if len(sys.argv) > 1 else "pipeline_test"
NUM_CELLS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
ITERATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
NUM_WARMUP = 5

RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Config parameters (same as real_l1.py)
num_tx_ant = int(os.environ.get("NUM_TX_ANT", "4"))
num_rx_ant = int(os.environ.get("NUM_RX_ANT", "4"))
num_prbs = int(os.environ.get("NUM_PRBS", "273"))

print(f"[pipeline] cells={NUM_CELLS} iters={ITERATIONS}", flush=True)

# Try to import PuschRxPipelineFactory
try:
    from aerial.phy5g.pusch.pusch_rx import PuschRxPipelineFactory, PuschRx
    from aerial.phy5g.config import (
        AerialPuschRxConfig, PuschConfig, PuschUeConfig,
    )
    from aerial.phy5g.ldpc import get_mcs, get_tb_size
    from aerial.util.cuda import get_cuda_stream
    print("[pipeline] Pipeline factory available", flush=True)
except ImportError as e:
    print(f"[pipeline] IMPORT ERROR: {e}", flush=True)
    sys.exit(2)

mcs_index = int(os.environ.get("MCS_INDEX", "2"))
mod_order, code_rate = get_mcs(mcs_index, 1)
start_prb = 0
start_sym = 2
num_symbols = 12
dmrs_syms = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
dmrs_max_len = 1
dmrs_add_ln_pos = 0

tb_size = get_tb_size(
    mod_order=mod_order, code_rate=code_rate, dmrs_syms=dmrs_syms,
    num_prbs=num_prbs, start_sym=start_sym, num_symbols=num_symbols, num_layers=1,
)

cuda_stream = get_cuda_stream()

NUM_SUBCARRIERS = num_prbs * 12
NUM_OFDM_SYMS = 14
rx_slot = cp.asarray(
    (np.random.randn(NUM_SUBCARRIERS, NUM_OFDM_SYMS, num_rx_ant) +
     1j * np.random.randn(NUM_SUBCARRIERS, NUM_OFDM_SYMS, num_rx_ant)).astype(np.complex64),
    order='F',
)

# Config for one UE
pusch_ue_config = PuschUeConfig(
    scid=0, layers=1, dmrs_ports=1, rnti=1234, data_scid=0,
    mcs_table=0, mcs_index=mcs_index, code_rate=int(code_rate * 10),
    mod_order=mod_order, tb_size=tb_size // 8,
)
pusch_configs = [PuschConfig(
    ue_configs=[pusch_ue_config], num_dmrs_cdm_grps_no_data=2,
    dmrs_scrm_id=41, start_prb=start_prb, num_prbs=num_prbs,
    dmrs_syms=dmrs_syms, dmrs_max_len=dmrs_max_len, dmrs_add_ln_pos=dmrs_add_ln_pos,
    start_sym=start_sym, num_symbols=num_symbols,
)]

# Try creating the pipeline
print("[pipeline] Creating PuschRx pipeline...", flush=True)
try:
    pusch_rx = PuschRx(
        cell_id=41,
        num_rx_ant=num_rx_ant,
        num_tx_ant=num_tx_ant,
        cuda_stream=cuda_stream,
    )
    print(f"[pipeline] PuschRx created: {type(pusch_rx).__name__}", flush=True)
except Exception as e:
    print(f"[pipeline] PuschRx CREATE FAILED: {e}", flush=True)
    traceback.print_exc()
    sys.exit(3)

# Warmup
print("[pipeline] warmup...", flush=True)
try:
    for _ in range(NUM_WARMUP):
        for _ in range(NUM_CELLS):
            _ = pusch_rx.run(rx_slot=rx_slot, slot=0, pusch_configs=pusch_configs)
    cp.cuda.runtime.deviceSynchronize()
except Exception as e:
    print(f"[pipeline] warmup FAILED: {e}", flush=True)
    traceback.print_exc()
    sys.exit(4)

print("[pipeline] measuring...", flush=True)
latencies = []
start_event = cp.cuda.Event()
end_event = cp.cuda.Event()
for i in range(ITERATIONS):
    start_event.record()
    for _ in range(NUM_CELLS):
        _ = pusch_rx.run(rx_slot=rx_slot, slot=0, pusch_configs=pusch_configs)
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
out_file = os.path.join(RESULTS_DIR, f"realL1_pipeline_{LABEL}_{ts}.json")
with open(out_file, "w") as f:
    json.dump(result, f, indent=2)

print(f"[pipeline] {LABEL}: mean={arr.mean():.3f}ms p95={np.percentile(arr,95):.3f}ms "
      f"miss1ms={miss_1ms}/{ITERATIONS}", flush=True)
print(f"[pipeline] saved: {out_file}", flush=True)
