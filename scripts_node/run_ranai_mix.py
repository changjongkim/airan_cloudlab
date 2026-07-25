"""Realistic RAN AI multi-instance mix — realistic same-partition AI-RAN load.

CLI: python3 run_ranai_mix.py <label> <duration_s>
     [--n_nrx 2] [--n_csinet 4] [--n_beampred 8]

Design: single process spawns N GPU inference threads simulating a real
xApp deployment where one AI service handles many concurrent UEs/cells.

Each thread runs a small realistic RAN AI workload:
- NRx (Neural Receiver): small TRT-style CNN, per-cell PUSCH demapping
- CsiNet: small transformer, per-UE CSI feedback compression (5G NR standard)
- BeamPred: MLP, per-UE beam selection

All threads share the same GPU context (single process → many CUDA streams),
mimicking a real per-UE inference server. Continuous inference for duration_s.

HBM pressure: ~14 concurrent inference streams → sustained HBM access.
"""
import argparse, time, threading
import torch
import torch.nn as nn

# ─── Small realistic RAN AI models ──────────────────────────────
class TinyNRx(nn.Module):
    """~5M params CNN — Neural Receiver approximation."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 64, 3, padding=1)   # 4 rx antennas → 64 features
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.head = nn.Linear(64, 6)                    # → LLR estimates
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return self.head(x.mean([-1,-2]))

class CsiNet(nn.Module):
    """CSI feedback compression encoder — 5G NR reference (~500K params)."""
    def __init__(self, dim=64, nhead=4, nlayers=2, comp_dim=32):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=nlayers)
        self.comp = nn.Linear(dim, comp_dim)
        self.dec  = nn.Linear(comp_dim, dim)
    def forward(self, x):
        h = self.enc(x)
        z = self.comp(h)
        return self.dec(z)

class BeamPred(nn.Module):
    """Beam prediction MLP — per-UE beam index selection (~50K params)."""
    def __init__(self, in_dim=64, hidden=128, num_beams=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_beams),
        )
    def forward(self, x): return self.mlp(x)

# ─── Worker thread ──────────────────────────────────────────────
class WorkerStats:
    def __init__(self): self.iters = 0

def run_worker(model, sample_x, stop_evt, stats, name):
    """Continuous inference loop on a dedicated CUDA stream."""
    stream = torch.cuda.Stream()
    with torch.inference_mode():
        with torch.cuda.stream(stream):
            # warmup
            for _ in range(2): _ = model(sample_x)
            stream.synchronize()
        while not stop_evt.is_set():
            with torch.cuda.stream(stream):
                _ = model(sample_x)
                stats.iters += 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("duration_s", type=int)
    ap.add_argument("--n_nrx",      type=int, default=2)
    ap.add_argument("--n_csinet",   type=int, default=4)
    ap.add_argument("--n_beampred", type=int, default=8)
    args = ap.parse_args()

    device = "cuda"; dtype = torch.float16
    print(f"[{time.strftime('%H:%M:%S')}] setting up RAN AI mix: "
          f"{args.n_nrx}× NRx + {args.n_csinet}× CsiNet + {args.n_beampred}× BeamPred", flush=True)

    workers = []

    # NRx instances (per-cell)
    for i in range(args.n_nrx):
        m = TinyNRx().to(device, dtype=dtype).eval()
        x = torch.randn(1, 4, 273, 14, device=device, dtype=dtype)  # 1 slot, 4 ant, 273 PRB, 14 sym
        s = WorkerStats()
        t = threading.Thread(target=run_worker, args=(m, x, None, s, f"nrx{i}"), daemon=True)
        workers.append((f"nrx{i}", t, s))

    # CsiNet instances (per-UE)
    for i in range(args.n_csinet):
        m = CsiNet(dim=64).to(device, dtype=dtype).eval()
        x = torch.randn(1, 32, 64, device=device, dtype=dtype)  # 1 UE, 32 subcarriers, 64 features
        s = WorkerStats()
        t = threading.Thread(target=run_worker, args=(m, x, None, s, f"csi{i}"), daemon=True)
        workers.append((f"csi{i}", t, s))

    # BeamPred instances (per-UE)
    for i in range(args.n_beampred):
        m = BeamPred().to(device, dtype=dtype).eval()
        x = torch.randn(1, 64, device=device, dtype=dtype)
        s = WorkerStats()
        t = threading.Thread(target=run_worker, args=(m, x, None, s, f"beam{i}"), daemon=True)
        workers.append((f"beam{i}", t, s))

    stop_evt = threading.Event()
    for name, t, stats in workers:
        t._args = (t._args[0], t._args[1], stop_evt, stats, name)   # inject stop_evt
        t.start()

    print(f"[{time.strftime('%H:%M:%S')}] {len(workers)} workers running", flush=True)
    start = time.time()
    while time.time() - start < args.duration_s:
        time.sleep(3)
        elapsed = time.time() - start
        total = sum(s.iters for _,_,s in workers)
        print(f"[{time.strftime('%H:%M:%S')}] elapsed={elapsed:.0f}s total_iters={total} rate={total/elapsed:.0f}/s", flush=True)

    stop_evt.set()
    time.sleep(1)   # let threads exit
    torch.cuda.synchronize()

    per_type = {"nrx": 0, "csi": 0, "beam": 0}
    for name, _, s in workers:
        for k in per_type:
            if name.startswith(k): per_type[k] += s.iters
    print(f"[{time.strftime('%H:%M:%S')}] DONE "
          f"nrx={per_type['nrx']} csi={per_type['csi']} beam={per_type['beam']} "
          f"total={sum(per_type.values())} label={args.label}", flush=True)

if __name__ == "__main__":
    main()
