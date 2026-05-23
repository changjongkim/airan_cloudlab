"""
xApp RAN telemetry anomaly detection — realistic AI-RAN workload.

Standard rApp/xApp use case: autoencoder reconstructs RAN KPI time series
and flags anomalies via reconstruction error. Continuous inference on
streaming RAN telemetry.

Model: 1D-CNN autoencoder, ~2-5 MB. Tiny, fits anywhere.
HBM usage: very low. Tests "lots of small AI services co-located" scenario.

Usage: python3 run_xapp_anomaly.py <gpu_id> <duration_sec>

Env:
  INPUT_DIM   — number of KPIs (default 32, e.g. PRB util, CQI, MCS, etc.)
  SEQ_LEN     — telemetry window (default 64 = 64ms at 1ms intervals)
  BATCH_SIZE  — UEs/cells processed in batch (default 16 cells)
"""
import os
import sys
import time

import torch
import torch.nn as nn

gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
duration = int(sys.argv[2]) if len(sys.argv) > 2 else 120

INPUT_DIM = int(os.environ.get("INPUT_DIM", "32"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "64"))
BATCH = int(os.environ.get("BATCH_SIZE", "16"))

torch.cuda.set_device(gpu_id)
device = f"cuda:{gpu_id}"


class Conv1dAutoencoder(nn.Module):
    """1D-CNN autoencoder over RAN telemetry (input_dim KPIs × seq_len timesteps)."""
    def __init__(self, in_dim=INPUT_DIM):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(in_dim, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.Conv1d(64, in_dim, 3, padding=1),
        )

    def forward(self, x):
        # x: (batch, in_dim, seq_len)
        z = self.enc(x)
        out = self.dec(z)
        return out


model = Conv1dAutoencoder().to(device).half()
model.eval()

# Synthetic telemetry: (batch=cells, KPIs, time)
batch = torch.randn(BATCH, INPUT_DIM, SEQ_LEN, dtype=torch.float16, device=device)

# Warmup
with torch.no_grad():
    for _ in range(5):
        out = model(batch)
        err = ((out - batch) ** 2).mean()
torch.cuda.synchronize(gpu_id)

free, total = torch.cuda.mem_get_info(gpu_id)
n_params = sum(p.numel() for p in model.parameters())
print(f"[xApp anomaly] params={n_params/1e6:.2f}M, "
      f"HBM={(total-free)/1e9:.2f}/{total/1e9:.1f}GB, "
      f"batch={BATCH} cells × {SEQ_LEN} TTIs × {INPUT_DIM} KPIs, dur={duration}s",
      flush=True)

c = 0
anomalies = 0
start = time.time()
with torch.no_grad():
    while time.time() - start < duration:
        # Refresh telemetry occasionally (simulating streaming data)
        if c % 50 == 0:
            batch = torch.randn(BATCH, INPUT_DIM, SEQ_LEN,
                                dtype=torch.float16, device=device)
        out = model(batch)
        err = ((out - batch) ** 2).mean(dim=(1, 2))
        # threshold = 1.5 (synthetic — half of variance)
        anomalies += int((err > 1.5).sum().item())
        c += 1
torch.cuda.synchronize(gpu_id)
elapsed = time.time() - start
print(f"[xApp anomaly] done: {c} inferences in {elapsed:.1f}s "
      f"({c/elapsed:.0f} inf/s), flagged {anomalies} anomalies "
      f"across {c*BATCH} cell-snapshots", flush=True)
