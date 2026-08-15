#!/usr/bin/env python3
"""Portable transaction trace schema shared by G7 and G8.

The schema stores measured or replayed *work readiness* separately from control
decisions.  That separation lets Host DART, persistent-device DART, and DART-Q
consume the exact same transaction input without relabeling simulated control
latency as measured hardware performance.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Iterable, Optional


TRACE_SCHEMA = "dart-transaction-input-v1"


@dataclasses.dataclass(frozen=True)
class TransactionSample:
    slot_id: int
    epoch: int
    release_ns: int
    deadline_ns: int
    fallback_latest_start_ns: int
    conventional_service_ns: int
    remote_ready_ns: Optional[int]
    remote_admitted: bool = True
    payload_visible: bool = True
    endpoint_epoch_valid: bool = True
    fallback_reserved: bool = True
    unreserved_fallback_delay_ns: int = 0
    utility: float = 1.0

    def __post_init__(self) -> None:
        if self.slot_id < 0 or not (0 <= self.epoch <= 0xFFFFFFFF):
            raise ValueError("invalid slot or epoch")
        if not (
            self.release_ns
            <= self.fallback_latest_start_ns
            <= self.deadline_ns
        ):
            raise ValueError("invalid release/fallback/deadline ordering")
        if self.conventional_service_ns <= 0:
            raise ValueError("conventional service must be positive")
        if self.remote_ready_ns is not None and self.remote_ready_ns < self.release_ns:
            raise ValueError("remote readiness precedes release")
        if self.unreserved_fallback_delay_ns < 0:
            raise ValueError("fallback queue delay must be non-negative")
        if self.utility < 0:
            raise ValueError("utility must be non-negative")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TransactionSample":
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class TransactionTrace:
    transactions: tuple[TransactionSample, ...]
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        keys = [(item.slot_id, item.epoch) for item in self.transactions]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate (slot, epoch) in transaction trace")

    @classmethod
    def build(
        cls,
        transactions: Iterable[TransactionSample],
        metadata: Optional[dict[str, Any]] = None,
    ) -> "TransactionTrace":
        ordered = tuple(
            sorted(
                transactions,
                key=lambda item: (item.release_ns, item.slot_id, item.epoch),
            )
        )
        return cls(ordered, dict(metadata or {}))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TransactionTrace":
        if value.get("schema") != TRACE_SCHEMA:
            raise ValueError(f"unsupported trace schema: {value.get('schema')}")
        return cls.build(
            (TransactionSample.from_dict(item) for item in value["transactions"]),
            value.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TRACE_SCHEMA,
            "metadata": self.metadata,
            "transactions": [item.to_dict() for item in self.transactions],
        }


def read_trace(path: Path) -> TransactionTrace:
    return TransactionTrace.from_dict(json.loads(path.read_text(encoding="utf-8")))


def write_trace(path: Path, trace: TransactionTrace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(trace.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

