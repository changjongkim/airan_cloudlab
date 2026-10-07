"""cuPHY and the neural receiver inside a green context: run times against the SMs of the context.

usage: gc_receivers.py SPEC DATASET ENGINE [SLOTS]
  SPEC  0       the primary context (all SMs)
        N       a green context on the first N SMs
        rest:N  a green context on the SMs left after the first N (another process can hold the first N)
The context is set again before every stage and every run: the first CuPy allocation and the neural receiver path
leave the thread in the primary context.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import greenctx as g
spec, dataset, engine = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
slots = int(sys.argv[4]) if len(sys.argv) > 4 else 96
dev, primary, res = g.init(0)
want, got = primary, int(res.u.sm.smCount)
if spec != "0":
    rest_side = spec.startswith("rest:")
    _g, want, got, other = g.green(dev, res, int(spec.split(":")[-1]), rest_side)
g.use(want)
import numpy as np, cupy as cp
from cell_ring import CellRing
from slot_radio import conv_path, nrx_path
ring = CellRing(dataset, "nv_mu2_m13", slots, 0); g.use(want)
stream = cp.cuda.Stream(non_blocking=True); g.use(want)
conv = conv_path(ring.profile, ring.tb_bytes, stream, 20); g.use(want)
nrx = nrx_path({"engine": engine, "nrx_ldpc_iterations": 20}, ring.profile, ring.tb_bytes, stream); g.use(want)
conv_s, nrx_s = [], []
for attempt in range(4):
    for i in range(slots):
        g.use(want)
        s = time.perf_counter(); conv.run(ring.views[i], ring.slots[i]); e = time.perf_counter() - s
        if attempt: conv_s.append(e)
    for i in range(slots):
        g.use(want)
        s = time.perf_counter(); nrx.run(ring.views[i], ring.slots[i]); e = time.perf_counter() - s
        if attempt: nrx_s.append(e)
q = lambda v, p: 1e3 * float(np.percentile(v, p))
print(f"RESULT receivers on {got} SMs ({spec}): cuPHY p50 {q(conv_s, 50):.2f} p99 {q(conv_s, 99):.2f} ms, "
      f"neural receiver p50 {q(nrx_s, 50):.2f} p99 {q(nrx_s, 99):.2f} ms", flush=True)
