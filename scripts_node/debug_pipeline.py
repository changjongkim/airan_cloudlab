"""Debug PuschRx segfault — mimic the official pytest fixture exactly."""
import sys, os
print(f"step 0: pyaerial import path = {os.environ.get('PYTHONPATH','?')}", flush=True)

print("step 1: import aerial.phy5g.pusch.PuschRx", flush=True)
from aerial.phy5g.pusch import PuschRx

print("step 2: construct PuschRx(cell_id=41, num_rx_ant=4, num_tx_ant=4)  <-- fixture pattern", flush=True)
try:
    rx = PuschRx(cell_id=41, num_rx_ant=4, num_tx_ant=4)
    print(f"step 3: OK -> {type(rx).__name__}", flush=True)
    print(f"step 3.1: cuda_stream = {rx.cuda_stream}", flush=True)
    print(f"step 3.2: pusch_pipeline = {type(rx.pusch_pipeline).__name__}", flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"step 2 EXCEPTION: {e}", flush=True)
    sys.exit(1)

print("step 4: destroy", flush=True)
del rx
print("step 5: done", flush=True)
