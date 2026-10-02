"""Load one cell's ring of pre-generated received slots into GPU memory."""

from __future__ import annotations

import json
from pathlib import Path

import cupy as cp
import numpy as np

from slot_radio import fortran_slot_views
from ul_profiles import NUM_RX_ANT, NUM_SUBCARRIERS, NUM_SYMBOLS_SLOT, PROFILES

SLOT_BYTES = NUM_SUBCARRIERS * NUM_SYMBOLS_SLOT * NUM_RX_ANT * 8


class CellRing:
    """Raw cudaMalloc ring so the base pointer can be exported over CUDA IPC."""

    def __init__(self, dataset: Path, profile_name: str, ring: int, offset: int,
                 stride: int = 1) -> None:
        self.profile = PROFILES[profile_name]
        meta = json.loads((dataset / f"{profile_name}_meta.json").read_text())
        count = int(meta["count"])
        self.indices = [(offset + stride * i) % count for i in range(ring)]
        self.slots = [int(meta["slots"][i]) for i in self.indices]
        self.esno_db = [float(meta["esno_db"][i]) for i in self.indices]
        blocks = np.load(dataset / f"{profile_name}_tb.npy")
        self.blocks = [blocks[i] for i in self.indices]
        self.tb_bytes = int(meta["tb_bytes"])
        self.ring = ring
        self.ptr = cp.cuda.runtime.malloc(SLOT_BYTES * ring)
        self.views = fortran_slot_views(self.ptr, self, ring)
        received = np.load(dataset / f"{profile_name}_rx.npy", mmap_mode="r")
        for view, index in zip(self.views, self.indices):
            view[...] = cp.asarray(np.ascontiguousarray(received[index]))
        cp.cuda.runtime.deviceSynchronize()

    def ipc_record(self) -> dict:
        return {
            "handle": cp.cuda.runtime.ipcGetMemHandle(self.ptr).hex(),
            "bytes": SLOT_BYTES * self.ring,
            "ring": self.ring,
            "indices": self.indices,
            "profile": self.profile.name,
        }

    def close(self) -> None:
        if self.ptr:
            cp.cuda.runtime.free(self.ptr)
            self.ptr = 0
