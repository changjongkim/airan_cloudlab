"""
PDSCH TX (transmit) pipeline latency measurement — counterpart to PUSCH RX (real_l1.py).
Goal: verify L1 disturbance generalizes beyond PUSCH RX.

Usage:
  python3 real_pdsch_tx.py <label> <num_cells> <iterations>
"""
import os, sys, json, datetime, statistics, time
import numpy as np
import cupy as cp

from aerial.phy5g.pdsch import PdschTxPipelineFactory
from aerial.phy5g.config import (
    PdschConfig, PdschUeConfig, PdschCwConfig, AerialPdschTxConfig,
)
from aerial.phy5g.ldpc import random_tb, get_mcs, get_tb_size
from aerial.util.cuda import get_cuda_stream

LABEL = sys.argv[1] if len(sys.argv) > 1 else "real_pdsch_tx"
NUM_CELLS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
ITERATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 30

NUM_RX_ANT = int(os.environ.get("NUM_RX_ANT", "4"))
NUM_TX_ANT = int(os.environ.get("NUM_TX_ANT", "4"))
NUM_PRBS = int(os.environ.get("NUM_PRBS", "273"))
MCS = int(os.environ.get("MCS", "2"))
TB_SIZE = get_tb_size(num_prb=NUM_PRBS, num_layers=1, mcs_index=MCS,
                     num_re_symbol=12, num_dmrs_re=4*4)

print(f"[realPdschTx] cells={NUM_CELLS} iters={ITERATIONS} prbs={NUM_PRBS} mcs={MCS} tb_size={TB_SIZE}", flush=True)

cuda_stream = get_cuda_stream()

# Build PDSCH TX pipeline (per-cell)
cw_cfg = PdschCwConfig(tb_size_bytes=TB_SIZE // 8, mcs_index=MCS, num_layers=1, rv=0)
ue_cfg = PdschUeConfig(scid=0, rnti=1, dmrs_ports=1, cws=[cw_cfg], dmrs_scrm_id=0)
pdsch_cfg = PdschConfig(
    slot_number=0, sfn=0, num_dl_dmrs_symbols=4, dmrs_max_len=1,
    dmrs_add_pos=2, num_layers=1, num_pdsch_symbols=12, start_prb=0,
    num_prbs=NUM_PRBS, start_symbol=2, ues=[ue_cfg],
)
factory = PdschTxPipelineFactory()
pipeline = factory.build_pipeline(
    AerialPdschTxConfig(carrier_freq_ghz=3.5, dl_bwp_size_prbs=NUM_PRBS, num_tx_ants=NUM_TX_ANT),
    cuda_stream=cuda_stream,
)

# Synthetic transport block
tb_bits = random_tb(TB_SIZE)
tb_words = cp.array(np.packbits(tb_bits).view(np.uint8), dtype=cp.uint8)

print(f"[realPdschTx] warmup...", flush=True)
for _ in range(5):
    out = pipeline.run(cell_idx=0, pdsch_configs=[pdsch_cfg], tb=tb_words, slot=0)
cuda_stream.synchronize()

print(f"[realPdschTx] measuring...", flush=True)
times_ms = []
for _ in range(ITERATIONS):
    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record(cuda_stream)
    for c in range(NUM_CELLS):
        out = pipeline.run(cell_idx=c, pdsch_configs=[pdsch_cfg], tb=tb_words, slot=0)
    end.record(cuda_stream)
    end.synchronize()
    times_ms.append(cp.cuda.get_elapsed_time(start, end))

times_ms.sort()
n = len(times_ms)
def pct(p): return times_ms[int(n * p / 100)] if n > 0 else 0
result = {
    "label": LABEL, "num_cells": NUM_CELLS, "iterations": ITERATIONS,
    "num_tx_ant": NUM_TX_ANT, "num_prbs": NUM_PRBS, "mcs_index": MCS,
    "mean_ms": statistics.mean(times_ms),
    "p50_ms": statistics.median(times_ms),
    "p95_ms": pct(95), "p99_ms": pct(99),
    "min_ms": times_ms[0], "max_ms": times_ms[-1],
    "miss_1ms": sum(1 for t in times_ms if t > 1.0),
    "raw_ms": times_ms,
}
print(f"[realPdschTx] {LABEL}: mean={result['mean_ms']:.3f}ms p95={result['p95_ms']:.3f}ms p99={result['p99_ms']:.3f}ms", flush=True)

os.makedirs("./results", exist_ok=True)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_file = f"./results/realPdschTx_{LABEL}_{ts}.json"
with open(out_file, "w") as f:
    json.dump(result, f)
print(f"[realPdschTx] saved: {out_file}", flush=True)
