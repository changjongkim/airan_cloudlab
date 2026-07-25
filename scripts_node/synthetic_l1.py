"""
Synthetic L1 benchmark v2 — heavier per-TTI work to actually exercise HBM bandwidth.
Used as a stand-in when full pyAerial L1 cannot be exercised end-to-end (API drift).

Target: per-TTI memory touch ~ a few GB across all cells so HBM bandwidth is the
limiter, matching the cuPHY L1 bottleneck the research targets.

Per-cell, per-TTI work (covers PUSCH-Rx-like memory pattern):
  - cuFFT batched 4096-point FFT on (8, 14, 4096) complex64 buffer (~7 MB)
  - 2x channel estimation proxy: matmul on (16, 16, SC=3276) complex64
  - LDPC decode proxy: 4 element-wise + 2 matmul over soft bits (~64 KB)
  - HARQ d-to-d copy: 4 x 256KB blocks (~1 MB)
  - Output D2H proxy: cudaMemcpy 64 KB

Repeat multiple inner iterations per TTI to amplify HBM access (mimicks cuPHY
running multiple kernels on the same buffers sequentially).

Usage:
  python3 synthetic_l1.py <label> <num_cells> <iterations>
"""
import os
import sys
import json
import time
import datetime

import numpy as np
import torch

LABEL = sys.argv[1] if len(sys.argv) > 1 else "synl1"
NUM_CELLS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
ITERATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
NUM_WARMUP = 30  # heavier warmup to absorb JIT / clock-ramp
INNER_REPEATS = 3  # 3x amplification of per-cell HBM accesses

RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda:0")
torch.cuda.set_device(device)
print(f"[synL1] GPU: {torch.cuda.get_device_name(0)}", flush=True)
free, total = torch.cuda.mem_get_info(0)
print(f"[synL1] HBM free/total: {free/1e9:.1f}/{total/1e9:.1f} GB", flush=True)

# 5G NR proxy params — bigger antennas to push HBM
FFT_SIZE = 4096
N_RX_ANT = 8           # was 4 — doubles FFT memory
N_TX_ANT = 8
N_SYMBOLS_PER_TTI = 14
N_PRBS = 273
SUBCARRIERS = N_PRBS * 12   # 3276
TB_SIZE_BYTES = 8192

# Per-cell footprint:
#   rx     : 8 * 14 * 4096 * 8B = ~3.7 MB
#   chest  : 8 * 8 * 3276 * 8B  = ~1.7 MB
#   harq   : 1 MB
# Total per cell ~6.5 MB. 20 cells = 130 MB. Per TTI (inner_repeats=3): ~400 MB.
per_cell_buffers = []
for c in range(NUM_CELLS):
    rx = torch.randn(N_RX_ANT, N_SYMBOLS_PER_TTI, FFT_SIZE, dtype=torch.complex64, device=device)
    chest = torch.randn(N_RX_ANT, N_TX_ANT, SUBCARRIERS, dtype=torch.complex64, device=device)
    softbits = torch.randn(TB_SIZE_BYTES * 8, dtype=torch.float32, device=device)
    harq = torch.empty(TB_SIZE_BYTES * 8 * 4, dtype=torch.float32, device=device)
    out_buf = torch.empty(TB_SIZE_BYTES // 4, dtype=torch.float32, device=device)
    per_cell_buffers.append((rx, chest, softbits, harq, out_buf))


def tti_proxy():
    """One TTI worth of L1 work across all cells, amplified by INNER_REPEATS."""
    for rx, chest, softbits, harq, out_buf in per_cell_buffers:
        for _ in range(INNER_REPEATS):
            # FFT proxy: PUSCH OFDM demod  (N_RX, 14, 4096)
            f = torch.fft.fft(rx, n=FFT_SIZE, dim=-1)
            # Channel estimation proxy: matmul stays in HBM
            data_syms = f[:, 2:14, :SUBCARRIERS]                      # (N_RX, 12, SC)
            # eq combine: per-subcarrier (N_TX^H * N_RX) — element-wise + reduction
            eq_out = (data_syms.unsqueeze(1) * chest.conj().unsqueeze(2)).sum(dim=0)  # (N_TX, 12, SC)
            # Soft demap proxy
            a = eq_out.real.reshape(-1)
            n = min(a.numel(), softbits.numel())
            softbits[:n].copy_(a[:n])
            torch.tanh(softbits, out=softbits)
            # LDPC iter proxy: 2x soft bit matmul-like operation
            harq[:softbits.numel()].copy_(softbits)
            harq[softbits.numel():2 * softbits.numel()].copy_(softbits)
            # Output D2H proxy (kept on device, just memcpy)
            out_buf.copy_(softbits[:out_buf.numel()])


# Warm up — heavier so kernels and clocks stabilize.
for _ in range(NUM_WARMUP):
    tti_proxy()
torch.cuda.synchronize()

# Measured.
latencies = []
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
for i in range(ITERATIONS):
    start_event.record()
    tti_proxy()
    end_event.record()
    torch.cuda.synchronize()
    latencies.append(start_event.elapsed_time(end_event))

arr = np.array(latencies)
miss_1ms = int(np.sum(arr > 1.0))
result = {
    "label": LABEL,
    "num_cells": NUM_CELLS,
    "iterations": ITERATIONS,
    "inner_repeats": INNER_REPEATS,
    "n_rx_ant": N_RX_ANT,
    "n_tx_ant": N_TX_ANT,
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
out_file = os.path.join(RESULTS_DIR, f"synl1_{LABEL}_{ts}.json")
with open(out_file, "w") as f:
    json.dump(result, f, indent=2)

print(f"[synL1] {LABEL}: mean={arr.mean():.3f}ms p95={np.percentile(arr,95):.3f}ms "
      f"p99={np.percentile(arr,99):.3f}ms miss1ms={miss_1ms}/{ITERATIONS}", flush=True)
print(f"[synL1] saved: {out_file}", flush=True)
