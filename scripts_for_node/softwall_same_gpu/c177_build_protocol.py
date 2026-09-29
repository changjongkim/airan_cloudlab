#!/usr/bin/env python3.11
"""Freeze one C177 single-GPU run from the frozen C159-Q2 builder.

The C159-Q2 builder places cuPHY instance h on GPU h. C177 keeps its radio
protocol (requests, seeds, SNRs, bounds) and moves every owner to one GPU:
the peer specification, the peer TSV, and the protocol placement. The
NeuralRx worker, the recovery lane, and the Qwen worker run on the same GPU
(see run_c177_single_gpu.sh).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import json

import build_confirm159_q2_protocol as q2

C177_SOURCES = (
    "c177_build_protocol.py",
    "c177_single_gpu.py",
    "run_c177_single_gpu.sh",
    "run_c178_single_gpu.sh",
    "c176_burst_coordinator.py",
    "c176_burst_plan.py",
    "c178_early_coordinator.py",
    "c178_early_plan.py",
    "c159_persistent_nrx_worker.py",
    "c159_prestaged_owner.py",
    "c159_q2_qwen_worker.py",
    "multigpu_p2p_ipc_gate.py",
)


def main() -> None:
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--gpu", type=int, default=0)
    own, rest = wrapper.parse_known_args()
    if own.gpu < 0:
        raise SystemExit("--gpu must be a device index")
    paths = argparse.ArgumentParser(add_help=False)
    for name in ("--output", "--peer-spec", "--peer-tsv", "--scripts-root"):
        paths.add_argument(name, type=Path, required=True)
    located, _ = paths.parse_known_args(rest)

    sys.argv = [q2.__file__] + rest
    q2.main()

    spec = json.loads(located.peer_spec.read_text(encoding="utf-8"))
    for peer in spec["peers"]:
        peer["source_device"] = own.gpu
    q2.atomic_json(located.peer_spec, spec)

    rows = []
    for line in located.peer_tsv.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        fields[2] = str(own.gpu)
        rows.append("\t".join(fields) + "\n")
    located.peer_tsv.write_text("".join(rows), encoding="utf-8")

    protocol = json.loads(located.output.read_text(encoding="utf-8"))
    protocol["placement"] = {
        "single_gpu": f"GPU{own.gpu}",
        "cuphy_owners": f"GPU{own.gpu} under MPS 50",
        "persistent_actual_nrx": f"GPU{own.gpu} under MPS 80",
        "shared_cuphy_and_qwen": f"GPU{own.gpu} under MPS 80/20",
        "transport": "CUDA IPC on one device; same-device copies replace NVLink P2P",
        "recovery_lane": ("one conventional receiver shared by every cell; the lane runs one recovery at a "
                          "time (c177_single_gpu.py)"),
    }
    protocol["c177_single_gpu"] = {
        "gpu": own.gpu,
        "base_builder": "build_confirm159_q2_protocol.py",
        "source_sha256": {name: q2.sha256(located.scripts_root / name) for name in C177_SOURCES},
    }
    q2.atomic_json(located.output, protocol)


if __name__ == "__main__":
    main()
