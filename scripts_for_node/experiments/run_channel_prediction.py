"""
Channel Prediction LSTM stress — realistic AI-RAN workload.

Real-time channel state prediction for proactive beamforming/scheduling.
Standard architecture: 2-layer LSTM, input = last N channel estimates,
output = predicted next-slot channel.

Model size: ~5-10 MB. Fits in any MIG instance including 1g.5gb.
HBM usage: low (model L2-resident), but bursty inference pattern.

Usage: python3 run_channel_prediction.py <gpu_id> <duration_sec>

Env:
  HIDDEN_SIZE  — LSTM hidden dim (default 256)
  SEQ_LEN      — input sequence length (default 16)
  CHANNEL_DIM  — channel coefficient dim (default 64 = 8 sc × 8 antennas)
  BATCH_SIZE   — inference batch (default 1, real-time scenario)
"""
import os
import sys
import time

import torch
import torch.nn as nn

gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
duration = int(sys.argv[2]) if len(sys.argv) > 2 else 120

HIDDEN = int(os.environ.get("HIDDEN_SIZE", "256"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "16"))
CHAN_DIM = int(os.environ.get("CHANNEL_DIM", "64"))
BATCH = int(os.environ.get("BATCH_SIZE", "1"))

torch.cuda.set_device(gpu_id)
device = f"cuda:{gpu_id}"


class ChannelPredictor(nn.Module):
    """2-layer LSTM predicting next-slot channel coefficients."""
    def __init__(self, in_dim=CHAN_DIM, hidden=HIDDEN, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, in_dim * 2)  # real+imag

    def forward(self, x):
        # x: (batch, seq, chan)
        y, _ = self.lstm(x)
        return self.out(y[:, -1, :])  # predict next from last hidden


model = ChannelPredictor().to(device).half()
model.eval()

# Synthetic input: random channel sequence
batch = torch.randn(BATCH, SEQ_LEN, CHAN_DIM, dtype=torch.float16, device=device)

# Warmup
with torch.no_grad():
    for _ in range(5):
        model(batch)
torch.cuda.synchronize(gpu_id)

free, total = torch.cuda.mem_get_info(gpu_id)
n_params = sum(p.numel() for p in model.parameters())
print(f"[ChanPred LSTM] params={n_params/1e6:.2f}M, "
      f"HBM={(total-free)/1e9:.2f}/{total/1e9:.1f}GB, "
      f"hidden={HIDDEN}, seq={SEQ_LEN}, dur={duration}s", flush=True)

c = 0
start = time.time()
with torch.no_grad():
    while time.time() - start < duration:
        # Real-time scenario: predict every TTI (~1ms)
        # Refresh input occasionally to avoid input caching effects
        if c % 100 == 0:
            batch = torch.randn(BATCH, SEQ_LEN, CHAN_DIM,
                                dtype=torch.float16, device=device)
        model(batch)
        c += 1
torch.cuda.synchronize(gpu_id)
elapsed = time.time() - start
print(f"[ChanPred LSTM] done: {c} predictions in {elapsed:.1f}s "
      f"({c/elapsed:.0f} pred/s, {elapsed/c*1000:.3f}ms/pred)", flush=True)
