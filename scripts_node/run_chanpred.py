"""ChanPred — small PyTorch transformer for CSI feedback prediction.

CLI: python3 run_chanpred.py <label> <duration_s>

Real-world proxy for O-RAN channel prediction xApp:
- 2-layer transformer, ~5M params
- Input: CSI features [B=64, seq=273, dim=32]  (batch of PRBs)
- Output: predicted next-slot CSI
- Compute-bound (small model, tight loop) — distinguishes from NRx (which uses TRT engine)
"""
import argparse, time
import torch
import torch.nn as nn

class ChanPredModel(nn.Module):
    def __init__(self, dim=32, nhead=4, nlayers=2, ffn=256):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, dim_feedforward=ffn, batch_first=True)
        self.enc  = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Linear(dim, dim)
    def forward(self, x):
        h = self.enc(x)
        return self.head(h)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("duration_s", type=int)
    args = ap.parse_args()

    device = "cuda"; dtype = torch.float16
    print(f"[{time.strftime('%H:%M:%S')}] building ChanPred model", flush=True)
    model = ChanPredModel(dim=32, nhead=4, nlayers=2, ffn=256).to(device, dtype=dtype).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{time.strftime('%H:%M:%S')}] params={n_params}", flush=True)

    x = torch.randn(64, 273, 32, device=device, dtype=dtype)

    with torch.inference_mode():
        for _ in range(5):
            _ = model(x); torch.cuda.synchronize()
    print(f"[{time.strftime('%H:%M:%S')}] warmup done", flush=True)

    start = time.time()
    n_iters = 0
    with torch.inference_mode():
        while time.time() - start < args.duration_s:
            _ = model(x)
            torch.cuda.synchronize()
            n_iters += 1
            if n_iters % 500 == 0:
                elapsed = time.time() - start
                print(f"[{time.strftime('%H:%M:%S')}] iter={n_iters}  elapsed={elapsed:.1f}s", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] DONE  iters={n_iters}  label={args.label}", flush=True)

if __name__ == "__main__":
    main()
