"""Device memory of one and two PairedDualReceiver objects in one process (diagnostic)."""
import json
import sys

import cupy as cp

sys.path.insert(0, "/softwall")
cp.cuda.runtime.setDevice(0)
pool = cp.get_default_memory_pool()


def used() -> float:
    free, total = cp.cuda.runtime.memGetInfo()
    return round((total - free) / 2**30, 3)


cp.zeros(1)
r = {"context": used()}
from dual_receiver_phy import PairedDualReceiver  # noqa: E402

engine = "/softwall_runtime/engines/neural_rx_fp16_full.trt"
a = PairedDualReceiver(engine, seed=1, device=0, enable_local_neural=False)
r.update(one_receiver=used(), pool_used_1=round(pool.used_bytes() / 2**30, 3), pool_total_1=round(pool.total_bytes() / 2**30, 3))
b = PairedDualReceiver(engine, seed=2, device=0, enable_local_neural=False)
r.update(two_receivers=used(), pool_used_2=round(pool.used_bytes() / 2**30, 3), pool_total_2=round(pool.total_bytes() / 2**30, 3))
pool.free_all_blocks()
r["after_free_all_blocks"] = used()
print("RESULT", json.dumps(r))
