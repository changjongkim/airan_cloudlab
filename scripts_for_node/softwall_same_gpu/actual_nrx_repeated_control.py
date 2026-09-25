#!/usr/bin/env python3.11
"""Persistent host control page for repeated actual-NRx outcomes and dispatch."""

from __future__ import annotations

import mmap
import os
import struct
import time
from pathlib import Path


CONTROL_SIZE = 64
OUTCOME_SEQ = 0
OUTCOME_FLAGS = 8
OUTCOME_COMPLETED_NS = 16
OUTCOME_CRC_FAILURES = 24
OUTCOME_PAYLOAD_MISMATCHES = 32
DISPATCH_SEQ = 40
DISPATCH_ACTION = 48
DISPATCH_GENERATION = 56

ACTION_SUCCESS = 1
ACTION_RECOVERY = 2


class _Page:
    def __init__(self, path: Path, *, create: bool, timeout_s: float = 0.0):
        self.path = path
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            os.ftruncate(self.fd, CONTROL_SIZE)
        else:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline and not path.is_file():
                time.sleep(0.001)
            if not path.is_file():
                raise TimeoutError(f"decision control timeout: {path}")
            self.fd = os.open(path, os.O_RDWR)
        self.mapping = mmap.mmap(self.fd, CONTROL_SIZE)

    def read(self, offset: int) -> int:
        return struct.unpack_from("=Q", self.mapping, offset)[0]

    def write(self, offset: int, value: int) -> None:
        struct.pack_into("=Q", self.mapping, offset, value)

    def close(self) -> None:
        self.mapping.close()
        os.close(self.fd)


class DecisionOwner:
    def __init__(self, path: Path):
        self.page = _Page(path, create=True)

    def publish_outcome(
        self,
        sequence: int,
        *,
        timely_success: bool,
        neural_correct: bool,
        completed_ns: int,
        crc_failures: int,
        payload_mismatches: int,
    ) -> None:
        flags = int(timely_success) | (int(neural_correct) << 1)
        self.page.write(OUTCOME_FLAGS, flags)
        self.page.write(OUTCOME_COMPLETED_NS, completed_ns)
        self.page.write(OUTCOME_CRC_FAILURES, crc_failures)
        self.page.write(OUTCOME_PAYLOAD_MISMATCHES, payload_mismatches)
        self.page.write(OUTCOME_SEQ, sequence)

    def wait_dispatch(self, sequence: int, timeout_s: float) -> dict:
        deadline = time.monotonic() + timeout_s
        observed = self.page.read(DISPATCH_SEQ)
        while time.monotonic() < deadline:
            observed = self.page.read(DISPATCH_SEQ)
            if observed == sequence:
                action = self.page.read(DISPATCH_ACTION)
                if action not in (ACTION_SUCCESS, ACTION_RECOVERY):
                    raise RuntimeError(f"invalid dispatch action {action}")
                return {
                    "sequence": sequence,
                    "action": action,
                    "generation": self.page.read(DISPATCH_GENERATION),
                }
            if observed > sequence:
                raise RuntimeError(
                    f"dispatch skipped expected={sequence} observed={observed}"
                )
            time.sleep(0)
        raise TimeoutError(
            f"dispatch timeout expected={sequence} observed={observed}"
        )

    def close(self) -> None:
        path = self.page.path
        self.page.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


class DecisionPeer:
    def __init__(self, path: Path, timeout_s: float):
        self.page = _Page(path, create=False, timeout_s=timeout_s)

    def read_outcome(self, sequence: int) -> dict | None:
        observed = self.page.read(OUTCOME_SEQ)
        if observed < sequence:
            return None
        if observed > sequence:
            raise RuntimeError(
                f"outcome skipped expected={sequence} observed={observed}"
            )
        flags = self.page.read(OUTCOME_FLAGS)
        return {
            "sequence": sequence,
            "timely_success": bool(flags & 1),
            "neural_correct": bool(flags & 2),
            "completed_ns": self.page.read(OUTCOME_COMPLETED_NS),
            "crc_failures": self.page.read(OUTCOME_CRC_FAILURES),
            "payload_mismatches": self.page.read(OUTCOME_PAYLOAD_MISMATCHES),
        }

    def publish_dispatch(self, sequence: int, action: int, generation: int) -> None:
        if action not in (ACTION_SUCCESS, ACTION_RECOVERY):
            raise ValueError(f"invalid dispatch action {action}")
        self.page.write(DISPATCH_ACTION, action)
        self.page.write(DISPATCH_GENERATION, generation)
        self.page.write(DISPATCH_SEQ, sequence)

    def close(self) -> None:
        self.page.close()
