#!/usr/bin/env python3
"""Same-device CUDA IPC payload channel with host shared-memory doorbells."""

from __future__ import annotations

import json
import mmap
import os
import struct
import time
import uuid
from pathlib import Path

import cupy as cp


TERMINATE_SEQ = (1 << 64) - 1
CONTROL_SIZE = 32
FWD_OFFSET = 0
BWD_OFFSET = 8
READY_OFFSET = 16


class _Control:
    def __init__(self, path: Path, create: bool):
        flags = os.O_RDWR
        if create:
            flags |= os.O_CREAT | os.O_EXCL
        self.fd = os.open(path, flags, 0o600)
        if create:
            os.ftruncate(self.fd, CONTROL_SIZE)
        self.mapping = mmap.mmap(self.fd, CONTROL_SIZE)

    def read(self, offset: int) -> int:
        return struct.unpack_from("=Q", self.mapping, offset)[0]

    def write(self, offset: int, value: int) -> None:
        struct.pack_into("=Q", self.mapping, offset, value)

    def close(self) -> None:
        self.mapping.close()
        os.close(self.fd)


def wait_value(control: _Control, offset: int, expected: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    observed = control.read(offset)
    while time.monotonic() < deadline:
        observed = control.read(offset)
        if observed == expected:
            return
        if expected != TERMINATE_SEQ and observed > expected:
            raise RuntimeError(f"doorbell skipped expected={expected} observed={observed}")
    raise TimeoutError(f"doorbell timeout expected={expected} observed={observed}")


class CudaIpcOwner:
    def __init__(self, tag: str, forward: cp.ndarray, backward: cp.ndarray):
        self.tag = tag
        self.forward = forward
        self.backward = backward
        self.info_path = Path(f"/dev/shm/cuda_ipc_{tag}.info")
        self.control_path = Path(f"/dev/shm/cuda_ipc_{tag}.ctrl")
        self.session = uuid.uuid4().hex
        self.control = _Control(self.control_path, create=True)
        info = {
            "version": 1,
            "session": self.session,
            "pid": os.getpid(),
            "forward_bytes": forward.nbytes,
            "backward_bytes": backward.nbytes,
            "forward_handle": cp.cuda.runtime.ipcGetMemHandle(
                int(forward.data.ptr)
            ).hex(),
            "backward_handle": cp.cuda.runtime.ipcGetMemHandle(
                int(backward.data.ptr)
            ).hex(),
        }
        temporary = self.info_path.with_suffix(".info.tmp")
        temporary.write_text(json.dumps(info), encoding="utf-8")
        os.replace(temporary, self.info_path)

    def wait_ready(self, timeout_s: float) -> None:
        wait_value(self.control, READY_OFFSET, 1, timeout_s)

    def publish_forward(self, sequence: int) -> None:
        self.control.write(FWD_OFFSET, sequence)

    def wait_backward(self, sequence: int, timeout_s: float) -> None:
        wait_value(self.control, BWD_OFFSET, sequence, timeout_s)

    def close(self) -> None:
        self.control.close()
        for path in (self.info_path, self.control_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


class CudaIpcPeer:
    def __init__(self, tag: str, timeout_s: float):
        self.info_path = Path(f"/dev/shm/cuda_ipc_{tag}.info")
        self.control_path = Path(f"/dev/shm/cuda_ipc_{tag}.ctrl")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not self.info_path.is_file():
            time.sleep(0.001)
        if not self.info_path.is_file():
            raise TimeoutError(f"CUDA IPC owner info timeout: {self.info_path}")
        info = json.loads(self.info_path.read_text(encoding="utf-8"))
        if info.get("version") != 1:
            raise RuntimeError(f"unsupported CUDA IPC info: {info}")
        self.control = _Control(self.control_path, create=False)
        flag = cp.cuda.runtime.cudaIpcMemLazyEnablePeerAccess
        self.forward_ptr = cp.cuda.runtime.ipcOpenMemHandle(
            bytes.fromhex(info["forward_handle"]), flag
        )
        self.backward_ptr = cp.cuda.runtime.ipcOpenMemHandle(
            bytes.fromhex(info["backward_handle"]), flag
        )
        self._forward_memory = cp.cuda.UnownedMemory(
            self.forward_ptr, int(info["forward_bytes"]), self
        )
        self._backward_memory = cp.cuda.UnownedMemory(
            self.backward_ptr, int(info["backward_bytes"]), self
        )
        self.forward = cp.ndarray(
            (int(info["forward_bytes"]),), dtype=cp.uint8,
            memptr=cp.cuda.MemoryPointer(self._forward_memory, 0),
        )
        self.backward = cp.ndarray(
            (int(info["backward_bytes"]),), dtype=cp.uint8,
            memptr=cp.cuda.MemoryPointer(self._backward_memory, 0),
        )

    def mark_ready(self) -> None:
        self.control.write(READY_OFFSET, 1)

    def read_forward(self) -> int:
        return self.control.read(FWD_OFFSET)

    def publish_backward(self, sequence: int) -> None:
        self.control.write(BWD_OFFSET, sequence)

    def close(self) -> None:
        cp.cuda.runtime.ipcCloseMemHandle(self.forward_ptr)
        cp.cuda.runtime.ipcCloseMemHandle(self.backward_ptr)
        self.control.close()
