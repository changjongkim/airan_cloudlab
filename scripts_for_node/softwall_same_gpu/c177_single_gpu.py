#!/usr/bin/env python3.11
"""C177 single-GPU placement of the C176 pipeline.

The C176 coordinator and the persistent NeuralRx worker enable CUDA peer
access between each owner GPU and their own GPU. On one GPU there is no peer:
CUDA IPC handles open on the same device, and cudaMemcpyPeerAsync with equal
devices is an ordinary device-to-device copy. This launcher replaces only the
peer-access step and runs the unchanged entry point, which keeps the frozen
C159/C176 sources byte-identical.

The coordinator also builds one conventional receiver per cell for its
recovery lane. One PairedDualReceiver holds 8.6 GiB of cuPHY state, and the
multi-GPU placement peaks at 81.7 GiB summed over its GPUs, above one 80 GB
A100. The lane runs one recovery at a time, and a recovery depends only on the
installed IQ window: the CRC result and payload returned to the owner do not
use the receiver's own reference TB. Therefore, on one GPU the lane keeps one
receiver and reuses it for every cell.

    python3 c177_single_gpu.py coordinator --peer-spec ... --destination-device 0 ...
    python3 c177_single_gpu.py nrx-worker --peer-spec ... --destination-device 0 ...
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import multigpu_p2p_ipc_gate

MODULES = {
    "coordinator": "c176_burst_coordinator",
    "coordinator-early": "c178_early_coordinator",
    "nrx-worker": "c159_persistent_nrx_worker",
}


def enable_peer_access_or_local(first: int, second: int) -> dict:
    """Peer access between two GPUs, or a same-device marker on one GPU."""
    if first == second:
        return {f"{first}_to_{second}": "same_device"}
    return multigpu_p2p_ipc_gate.enable_peer_access(first, second)


def shared_receiver(factory):
    """Build one receiver per (device, local-NeuralRx flag) and reuse it."""
    built = {}

    def build(engine_path, *, device=0, enable_local_neural=True, **kwargs):
        key = (engine_path, device, enable_local_neural)
        if key not in built:
            built[key] = factory(engine_path, device=device, enable_local_neural=enable_local_neural, **kwargs)
            print(f"[C177] recovery lane receiver built once on GPU{device} and shared by every cell", flush=True)
        return built[key]

    return build


def single_gpu(argv: list) -> bool:
    """True when every owner runs on the coordinator's own GPU."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--peer-spec", type=Path, required=True)
    parser.add_argument("--destination-device", type=int, default=2)
    args, _ = parser.parse_known_args(argv)
    peers = json.loads(args.peer_spec.read_text())["peers"]
    return all(int(peer["source_device"]) == args.destination_device for peer in peers)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in MODULES:
        raise SystemExit(f"usage: c177_single_gpu.py {{{'|'.join(MODULES)}}} ARGS...")
    module = importlib.import_module(MODULES[sys.argv[1]])
    module.enable_peer_access = enable_peer_access_or_local
    if sys.argv[1].startswith("coordinator") and single_gpu(sys.argv[2:]):
        module.PairedDualReceiver = shared_receiver(module.PairedDualReceiver)
    sys.argv = [module.__file__] + sys.argv[2:]
    module.main()


if __name__ == "__main__":
    main()
