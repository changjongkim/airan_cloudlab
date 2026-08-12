"""Minimal 2-process RDMA loopback: producer writes payload to consumer's MR,
consumer polls a sequence counter, prints when message arrives.
"""
import os, sys, time, struct
from rdma_channel import RdmaEndpoint

ROLE = sys.argv[1] if len(sys.argv) > 1 else "prod"
TAG = os.environ.get("RDMA_TAG", "test")
PAYLOAD = 1024 * 1024   # 1 MB

if ROLE == "prod":
    ep = RdmaEndpoint(buf_size=PAYLOAD, tag=TAG, is_producer=True)
    print("[prod] endpoint ready", flush=True)
    for seq in range(1, 11):
        payload = (seq).to_bytes(4, "little") * (PAYLOAD // 4)
        ep.mr.write(payload, PAYLOAD)                      # local buf
        t0 = time.perf_counter_ns()
        ep.rdma_write(0, PAYLOAD, 0)                       # remote data
        ep.mr.write(struct.pack("!Q", seq), 8, offset=PAYLOAD)
        ep.rdma_write(PAYLOAD, 8, PAYLOAD)                 # remote seq marker
        us = (time.perf_counter_ns() - t0) / 1e3
        print(f"[prod] seq={seq} write took {us:.2f} us", flush=True)
        time.sleep(0.05)
    print("[prod] done", flush=True)

else:  # cons
    ep = RdmaEndpoint(buf_size=PAYLOAD, tag=TAG, is_producer=False)
    print("[cons] endpoint ready", flush=True)
    last = 0
    while last < 10:
        seq_bytes = ep.mr.read(8, offset=PAYLOAD)
        seq = struct.unpack("!Q", seq_bytes)[0]
        if seq > last:
            first = int.from_bytes(ep.mr.read(4, offset=0), "little")
            print(f"[cons] got seq={seq} first_word={first}", flush=True)
            last = seq
    print("[cons] done", flush=True)
