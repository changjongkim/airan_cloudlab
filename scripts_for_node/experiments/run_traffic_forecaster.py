"""
Traffic forecaster stress — LSTM-based time-series forecaster (RAN traffic prediction).

Usage:  python3 run_traffic_forecaster.py <gpu_id> <duration_sec> [batch_size] [hidden_dim]
        (02 script invokes it as: ... <dur> 64 384)

Mirrors the F_E_forecaster condition: a recurrent forecaster inference loop, a
"safe-ish" AI workload (moderate, bursty) co-located with L1 on the no-MIG GPU.
"""
import os
import sys
import time

import torch
import torch.nn as nn

gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
duration = int(sys.argv[2]) if len(sys.argv) > 2 else 120
batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 64
hidden_dim = int(sys.argv[4]) if len(sys.argv) > 4 else 384

dev = f"cuda:{gpu_id}"
torch.cuda.set_device(gpu_id)
seq_len = 32
in_dim = 16

print(f"[forecaster] batch={batch_size} hidden={hidden_dim} seq={seq_len} dur={duration}s", flush=True)


class Forecaster(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


model = Forecaster(in_dim, hidden_dim).to(dev).eval()
x = torch.randn(batch_size, seq_len, in_dim, device=dev)

t_end = time.time() + duration
iters = 0
with torch.no_grad():
    while time.time() < t_end:
        for _ in range(20):
            _ = model(x)
        iters += 20
        torch.cuda.synchronize()
print(f"[forecaster] done, {iters} inferences", flush=True)
