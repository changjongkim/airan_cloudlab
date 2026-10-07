"""Time of the prefill units of the AI model on the whole GPU and on a slice of N SMs (QwenUnits.add_slice).

usage: gc_units.py N [N ...]      -> per chunk size: layer unit time on the whole GPU, on the slice, and the ratio
One process per N is started by the caller (a runner has one slice).
"""
import sys, time
import torch
from qwen_units import QwenUnits

sms = int(sys.argv[1])
runner = QwenUnits("Qwen/Qwen2.5-1.5B", 0, chunks=(128, 512, 1024), max_ctx=1024)
got = runner.add_slice(sms)
prompt = torch.randint(0, 32000, (1024,), generator=torch.Generator().manual_seed(1)).to(runner.device)

def layer_ms(c, n=3):
    runner.load_prompt(prompt[:c]); runner.launch(c, "prep", 0); runner.stream.synchronize()
    best = None
    for _ in range(n):
        s = time.perf_counter()
        for i in range(runner.layers):
            runner.launch(c, f"layer{i}", 0)
        runner.stream.synchronize()
        t = 1e3 * (time.perf_counter() - s) / runner.layers
        best = t if best is None else min(best, t)
    return best

rows = []
for c in (128, 512, 1024):
    runner.set_mode(0); full = layer_ms(c)
    runner.set_mode(1); part = layer_ms(c)
    runner.set_mode(0)
    rows.append(f"{c} tokens: {full:.3f} ms -> {part:.3f} ms ({part / full:.2f}x)")
print(f"RESULT slice of {got} SMs, layer unit on the whole GPU -> on the slice: " + "; ".join(rows), flush=True)
