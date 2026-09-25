#!/usr/bin/env python3
"""Unified DART-Rx deadline-transaction runtime.

This module contains the control semantics shared by P2P and GDR data paths.
It intentionally has no CUDA dependency, so transaction correctness and policy
decisions can be tested before binding the runtime to a GPU transport.
"""

from __future__ import annotations

import dataclasses
import enum
import struct
import threading
from collections import Counter
from typing import Dict, Iterable, Mapping, Optional, Tuple


DESCRIPTOR_FORMAT = "!QIHHQQQIIHBBII4x"
DESCRIPTOR_SIZE = struct.calcsize(DESCRIPTOR_FORMAT)
assert DESCRIPTOR_SIZE == 64


class RequestState(enum.Enum):
    FREE = "free"
    RESERVED = "reserved"
    DISPATCHED = "dispatched"
    NRX_RUNNING = "nrx_running"
    NRX_READY = "nrx_ready"
    CLOSED = "closed"


class CommitState(enum.Enum):
    OPEN = "open"
    FALLBACK_RUNNING = "fallback_running"
    NRX_WON = "nrx_won"
    CONV_WON = "conv_won"
    DEADLINE_MISS = "deadline_miss"


class CommitKind(enum.IntEnum):
    NRX = 1
    CONVENTIONAL = 2


@dataclasses.dataclass(frozen=True)
class DartRequest:
    slot_id: int
    epoch: int
    graph_id: int
    tensor_class: int
    release_ns: int
    deadline_ns: int
    fallback_latest_start_ns: int
    input_slot: int = 0
    output_slot: int = 0
    candidate_bitmap: int = 0xFFFF
    utility_class: int = 1
    flags: int = 0x03
    payload_checksum: int = 0
    reserved: int = 0

    def __post_init__(self) -> None:
        if self.release_ns > self.fallback_latest_start_ns:
            raise ValueError("fallback latest start precedes release")
        if self.fallback_latest_start_ns > self.deadline_ns:
            raise ValueError("fallback latest start exceeds deadline")
        if not (0 <= self.epoch <= 0xFFFFFFFF):
            raise ValueError("epoch outside uint32")

    def pack(self) -> bytes:
        return struct.pack(
            DESCRIPTOR_FORMAT,
            self.slot_id,
            self.epoch,
            self.graph_id,
            self.tensor_class,
            self.release_ns,
            self.deadline_ns,
            self.fallback_latest_start_ns,
            self.input_slot,
            self.output_slot,
            self.candidate_bitmap,
            self.utility_class,
            self.flags,
            self.payload_checksum,
            self.reserved,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "DartRequest":
        if len(payload) != DESCRIPTOR_SIZE:
            raise ValueError(f"descriptor must be {DESCRIPTOR_SIZE} bytes")
        values = struct.unpack(DESCRIPTOR_FORMAT, payload)
        return cls(*values)


@dataclasses.dataclass(frozen=True)
class ServiceProfile:
    forward_ns: int
    service_ns: int
    backward_ns: int
    control_ns: int = 0
    positive_error_ns: int = 0

    @property
    def bound_ns(self) -> int:
        values = dataclasses.astuple(self)
        if any(value < 0 for value in values):
            raise ValueError("profile components must be non-negative")
        return sum(values)


class ProfileTable:
    """Conservative profiles keyed by endpoint/tensor/graph/background class."""

    def __init__(self, profiles: Mapping[Tuple[str, int, int, str], ServiceProfile]):
        self._profiles = dict(profiles)

    def get(
        self,
        endpoint_id: str,
        request: DartRequest,
        background_mode: str,
    ) -> ServiceProfile:
        exact = (endpoint_id, request.tensor_class, request.graph_id, background_mode)
        isolated = (endpoint_id, request.tensor_class, request.graph_id, "isolated")
        if exact in self._profiles:
            return self._profiles[exact]
        if isolated in self._profiles:
            return self._profiles[isolated]
        raise KeyError(f"missing DART profile: {exact}")


@dataclasses.dataclass(frozen=True)
class Reservation:
    endpoint_id: str
    slot_id: int
    epoch: int
    input_slot: int
    output_slot: int
    start_ns: int
    predicted_finish_ns: int
    profile: ServiceProfile
    health_epoch: int


@dataclasses.dataclass(frozen=True)
class FallbackReservation:
    """A reserved conventional-receiver recovery window."""

    slot_id: int
    epoch: int
    lane: int
    start_ns: int
    predicted_finish_ns: int


@dataclasses.dataclass(frozen=True)
class LeaseReservation:
    endpoint_id: str
    unit_id: str
    start_ns: int
    predicted_finish_ns: int
    value: float


@dataclasses.dataclass(frozen=True)
class BackgroundUnit:
    unit_id: str
    bound_ns: int
    value: float

    def __post_init__(self) -> None:
        if self.bound_ns <= 0 or self.value < 0:
            raise ValueError("invalid background unit")


class FallbackCalendar:
    """Atomic interval credits for mandatory conventional recovery.

    Remote execution is safe only when its local recovery interval can also be
    reserved.  Exact intervals make correlated fallback bursts visible instead
    of assuming that every speculative NRx request can fall back concurrently.
    """

    def __init__(
        self,
        capacity: int,
        service_ns: int,
        commit_guard_ns: int = 0,
    ):
        if capacity <= 0:
            raise ValueError("fallback capacity must be positive")
        if service_ns <= 0:
            raise ValueError("fallback service time must be positive")
        if commit_guard_ns < 0:
            raise ValueError("fallback commit guard must be non-negative")
        self.capacity = capacity
        self.service_ns = service_ns
        self.commit_guard_ns = commit_guard_ns
        self._reservations: Dict[Tuple[int, int], FallbackReservation] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _overlaps(
        start_ns: int,
        finish_ns: int,
        reservation: FallbackReservation,
    ) -> bool:
        return (
            start_ns < reservation.predicted_finish_ns
            and reservation.start_ns < finish_ns
        )

    def reserve(self, request: DartRequest) -> Optional[FallbackReservation]:
        key = (request.slot_id, request.epoch)
        start_ns = request.fallback_latest_start_ns
        finish_ns = start_ns + self.service_ns
        if finish_ns > request.deadline_ns - self.commit_guard_ns:
            return None
        with self._lock:
            if key in self._reservations:
                return None
            for lane in range(self.capacity):
                conflict = any(
                    current.lane == lane
                    and self._overlaps(start_ns, finish_ns, current)
                    for current in self._reservations.values()
                )
                if conflict:
                    continue
                reservation = FallbackReservation(
                    slot_id=request.slot_id,
                    epoch=request.epoch,
                    lane=lane,
                    start_ns=start_ns,
                    predicted_finish_ns=finish_ns,
                )
                self._reservations[key] = reservation
                return reservation
            return None

    def release(self, reservation: FallbackReservation) -> bool:
        key = (reservation.slot_id, reservation.epoch)
        with self._lock:
            if self._reservations.get(key) != reservation:
                return False
            del self._reservations[key]
            return True

    def retime_earlier(
        self,
        request: DartRequest,
        reservation: FallbackReservation,
        start_ns: int,
    ) -> Optional[FallbackReservation]:
        """Atomically move a live recovery credit to an earlier free interval.

        The old interval stays reserved if the move cannot be made. This is
        required before an early conventional launch when more than one radio
        request may have a recovery obligation on the same physical lane.
        """

        key = (request.slot_id, request.epoch)
        finish_ns = start_ns + self.service_ns
        if (
            key != (reservation.slot_id, reservation.epoch)
            or start_ns < request.release_ns
            or start_ns > reservation.start_ns
            or finish_ns > request.deadline_ns - self.commit_guard_ns
        ):
            return None
        with self._lock:
            if self._reservations.get(key) != reservation:
                return None
            lanes = [reservation.lane] + [
                lane for lane in range(self.capacity) if lane != reservation.lane
            ]
            for lane in lanes:
                if any(
                    other_key != key
                    and current.lane == lane
                    and self._overlaps(start_ns, finish_ns, current)
                    for other_key, current in self._reservations.items()
                ):
                    continue
                moved = FallbackReservation(
                    slot_id=request.slot_id,
                    epoch=request.epoch,
                    lane=lane,
                    start_ns=start_ns,
                    predicted_finish_ns=finish_ns,
                )
                self._reservations[key] = moved
                return moved
            return None

    def replan_atomic(
        self,
        moves: Iterable[Tuple[DartRequest, FallbackReservation, int, int]],
        now_ns: int,
    ) -> Optional[Dict[Tuple[int, int], FallbackReservation]]:
        """Replace several unstarted credits together, including later moves.

        The caller must hold each owning transaction lock. No old interval is
        released unless the complete replacement calendar is conflict-free.
        This books only conventional capacity; AI admission and physical GPU
        completion remain separate obligations of the controller.
        """

        proposed = list(moves)
        if not proposed or now_ns < 0:
            return None
        keys = [(request.slot_id, request.epoch) for request, _, _, _ in proposed]
        if len(set(keys)) != len(keys):
            return None
        with self._lock:
            staged = dict(self._reservations)
            replacements = {}
            for request, old, start_ns, lane in proposed:
                key = (request.slot_id, request.epoch)
                finish_ns = start_ns + self.service_ns
                if (
                    staged.get(key) != old
                    or lane < 0 or lane >= self.capacity
                    or start_ns < max(now_ns, request.release_ns)
                    or finish_ns > request.deadline_ns - self.commit_guard_ns
                ):
                    return None
                replacement = FallbackReservation(
                    slot_id=request.slot_id,
                    epoch=request.epoch,
                    lane=lane,
                    start_ns=start_ns,
                    predicted_finish_ns=finish_ns,
                )
                staged[key] = replacement
                replacements[key] = replacement
            by_lane = {}
            for reservation in staged.values():
                by_lane.setdefault(reservation.lane, []).append(reservation)
            for reservations in by_lane.values():
                ordered = sorted(reservations, key=lambda item: item.start_ns)
                if any(left.predicted_finish_ns > right.start_ns
                       for left, right in zip(ordered, ordered[1:])):
                    return None
            self._reservations = staged
            return replacements

    def snapshot(self) -> dict:
        with self._lock:
            by_lane = [0] * self.capacity
            for reservation in self._reservations.values():
                by_lane[reservation.lane] += 1
            return {
                "capacity": self.capacity,
                "service_ns": self.service_ns,
                "outstanding": len(self._reservations),
                "by_lane": by_lane,
            }


class EndpointState:
    """Atomic endpoint tail and registered tensor-slot credits."""

    def __init__(self, endpoint_id: str, ring_depth: int, health_epoch: int = 1):
        if ring_depth <= 0:
            raise ValueError("ring depth must be positive")
        self.endpoint_id = endpoint_id
        self.ring_depth = ring_depth
        self.health_epoch = health_epoch
        self.tail_ns = 0
        self.blocking_until_ns = 0
        self.healthy = True
        self._input_slots = set(range(ring_depth))
        self._output_slots = set(range(ring_depth))
        self._reservations: Dict[Tuple[int, int], Reservation] = {}
        self._lock = threading.Lock()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "endpoint_id": self.endpoint_id,
                "tail_ns": self.tail_ns,
                "blocking_until_ns": self.blocking_until_ns,
                "healthy": self.healthy,
                "health_epoch": self.health_epoch,
                "free_input_slots": len(self._input_slots),
                "free_output_slots": len(self._output_slots),
                "outstanding": len(self._reservations),
            }

    def predict_finish(self, now_ns: int, profile: ServiceProfile) -> int:
        with self._lock:
            start = max(now_ns, self.tail_ns, self.blocking_until_ns)
            return start + profile.bound_ns

    def reserve(
        self,
        request: DartRequest,
        now_ns: int,
        profile: ServiceProfile,
        commit_guard_ns: int,
        enforce_deadline: bool = True,
        latest_finish_ns: Optional[int] = None,
    ) -> Optional[Reservation]:
        key = (request.slot_id, request.epoch)
        with self._lock:
            if not self.healthy or key in self._reservations:
                return None
            if not self._input_slots or not self._output_slots:
                return None
            start = max(now_ns, self.tail_ns, self.blocking_until_ns)
            finish = start + profile.bound_ns
            if enforce_deadline and finish > request.deadline_ns - commit_guard_ns:
                return None
            if latest_finish_ns is not None and finish > latest_finish_ns:
                return None
            input_slot = min(self._input_slots)
            output_slot = min(self._output_slots)
            self._input_slots.remove(input_slot)
            self._output_slots.remove(output_slot)
            reservation = Reservation(
                endpoint_id=self.endpoint_id,
                slot_id=request.slot_id,
                epoch=request.epoch,
                input_slot=input_slot,
                output_slot=output_slot,
                start_ns=start,
                predicted_finish_ns=finish,
                profile=profile,
                health_epoch=self.health_epoch,
            )
            self._reservations[key] = reservation
            self.tail_ns = finish
            return reservation

    def release(self, reservation: Reservation, actual_finish_ns: int) -> bool:
        key = (reservation.slot_id, reservation.epoch)
        with self._lock:
            current = self._reservations.pop(key, None)
            if current != reservation:
                return False
            self._input_slots.add(reservation.input_slot)
            self._output_slots.add(reservation.output_slot)
            if not self._reservations:
                self.tail_ns = min(self.tail_ns, actual_finish_ns)
            return True

    def completion_valid(self, reservation: Reservation) -> bool:
        key = (reservation.slot_id, reservation.epoch)
        with self._lock:
            return (
                self.healthy
                and reservation.health_epoch == self.health_epoch
                and self._reservations.get(key) == reservation
            )

    def restart(self, now_ns: int) -> None:
        """Invalidate all old completions and expose a fresh endpoint epoch."""

        with self._lock:
            self.health_epoch += 1
            self.healthy = True
            self.tail_ns = now_ns
            self.blocking_until_ns = now_ns
            self._input_slots = set(range(self.ring_depth))
            self._output_slots = set(range(self.ring_depth))
            self._reservations.clear()

    def reserve_lease(
        self,
        now_ns: int,
        next_guard_ns: int,
        drain_guard_ns: int,
        units: Iterable[BackgroundUnit],
    ) -> Optional[LeaseReservation]:
        with self._lock:
            start = max(now_ns, self.tail_ns, self.blocking_until_ns)
            slack = next_guard_ns - start - drain_guard_ns
            feasible = [unit for unit in units if unit.bound_ns <= slack]
            if not self.healthy or not feasible:
                return None
            unit = max(feasible, key=lambda item: (item.value / item.bound_ns, item.value))
            finish = start + unit.bound_ns
            self.tail_ns = finish
            self.blocking_until_ns = finish
            return LeaseReservation(
                endpoint_id=self.endpoint_id,
                unit_id=unit.unit_id,
                start_ns=start,
                predicted_finish_ns=finish,
                value=unit.value,
            )


class DartTransaction:
    """Single request with independent execution and commit state."""

    def __init__(
        self,
        request: DartRequest,
        reservation: Optional[Reservation],
        fallback_reservation: Optional[FallbackReservation],
        submit_ns: int,
    ):
        self.request = request
        self.reservation = reservation
        self.fallback_reservation = fallback_reservation
        has_credit = reservation is not None or fallback_reservation is not None
        self.request_state = RequestState.RESERVED if has_credit else RequestState.CLOSED
        self.commit_state = CommitState.OPEN if has_credit else CommitState.FALLBACK_RUNNING
        self.submit_ns = submit_ns
        self.fallback_started_ns: Optional[int] = (
            None if has_credit else submit_ns
        )
        self.nrx_observed_ns: Optional[int] = None
        self.commit_ns: Optional[int] = None
        self.result_slot: Optional[int] = None
        self._lock = threading.Lock()

    @property
    def closed(self) -> bool:
        return self.commit_state in {
            CommitState.NRX_WON,
            CommitState.CONV_WON,
            CommitState.DEADLINE_MISS,
        }

    def start_fallback(self, now_ns: int, force: bool = False) -> bool:
        with self._lock:
            if self.commit_state is not CommitState.OPEN:
                return False
            if now_ns > self.request.deadline_ns:
                return False
            reserved_start_ns = (
                self.fallback_reservation.start_ns
                if self.fallback_reservation is not None
                else self.request.fallback_latest_start_ns
            )
            if not force and now_ns < reserved_start_ns:
                return False
            self.commit_state = CommitState.FALLBACK_RUNNING
            self.fallback_started_ns = now_ns
            return True

    def attach_nrx(self, reservation: Reservation) -> bool:
        """Attach optional execution only while the mandatory credit is live."""

        with self._lock:
            if (
                self.commit_state is not CommitState.OPEN
                or self.fallback_reservation is None
                or self.reservation is not None
                or reservation.slot_id != self.request.slot_id
                or reservation.epoch != self.request.epoch
            ):
                return False
            self.reservation = reservation
            return True

    def start_fallback_early(
        self, now_ns: int, calendar: FallbackCalendar
    ) -> bool:
        """Start only after atomically reserving the earlier physical lane."""

        with self._lock:
            if (
                self.commit_state is not CommitState.OPEN
                or self.fallback_reservation is None
                or now_ns > self.request.deadline_ns
            ):
                return False
            moved = calendar.retime_earlier(
                self.request, self.fallback_reservation, now_ns
            )
            if moved is None:
                return False
            self.fallback_reservation = moved
            self.commit_state = CommitState.FALLBACK_RUNNING
            self.fallback_started_ns = now_ns
            return True

    def try_commit(
        self,
        kind: CommitKind,
        slot_id: int,
        epoch: int,
        result_slot: int,
        now_ns: int,
    ) -> bool:
        with self._lock:
            if slot_id != self.request.slot_id or epoch != self.request.epoch:
                return False
            if self.closed:
                return False
            if (
                kind is CommitKind.CONVENTIONAL
                and self.commit_state is not CommitState.FALLBACK_RUNNING
            ):
                return False
            if now_ns > self.request.deadline_ns:
                self.commit_state = CommitState.DEADLINE_MISS
                self.commit_ns = now_ns
                self.request_state = RequestState.CLOSED
                return False
            self.commit_state = (
                CommitState.NRX_WON
                if kind is CommitKind.NRX
                else CommitState.CONV_WON
            )
            self.commit_ns = now_ns
            self.result_slot = result_slot
            self.request_state = RequestState.CLOSED
            return True

    def detach_fallback_reservation(
        self,
        after_execution: bool = False,
    ) -> Optional[FallbackReservation]:
        """Return a releasable fallback reservation exactly once.

        Once fallback has started, an NRx win cannot free its compute lane until
        the already-running conventional graph completes.
        """

        with self._lock:
            if self.fallback_reservation is None:
                return None
            if self.fallback_started_ns is not None and not after_execution:
                return None
            reservation = self.fallback_reservation
            self.fallback_reservation = None
            return reservation

    def expire(self, now_ns: int) -> bool:
        with self._lock:
            if self.closed or now_ns <= self.request.deadline_ns:
                return False
            self.commit_state = CommitState.DEADLINE_MISS
            self.commit_ns = now_ns
            self.request_state = RequestState.CLOSED
            return True


class DartRuntime:
    """One framework: admission, reservation, commit, fallback, and lease."""

    def __init__(
        self,
        endpoints: Iterable[EndpointState],
        profiles: ProfileTable,
        commit_guard_ns: int,
        background_mode: str = "isolated",
        fallback_calendar: Optional[FallbackCalendar] = None,
    ):
        if commit_guard_ns < 0:
            raise ValueError("commit guard must be non-negative")
        self.endpoints = {endpoint.endpoint_id: endpoint for endpoint in endpoints}
        if not self.endpoints:
            raise ValueError("at least one endpoint is required")
        self.profiles = profiles
        self.commit_guard_ns = commit_guard_ns
        self.background_mode = background_mode
        self.fallback_calendar = fallback_calendar
        self.metrics = Counter()
        self._round_robin_counter = 0

    def submit(
        self,
        request: DartRequest,
        now_ns: int,
        policy: str = "predicted_finish",
    ) -> DartTransaction:
        if policy not in {"static", "round_robin", "shortest_queue", "predicted_finish"}:
            raise ValueError(f"unsupported routing policy: {policy}")
        ranked = []
        for index, endpoint in enumerate(self.endpoints.values()):
            if not request.candidate_bitmap & (1 << index):
                continue
            try:
                profile = self.profiles.get(
                    endpoint.endpoint_id, request, self.background_mode
                )
            except KeyError:
                self.metrics["missing_profile"] += 1
                continue
            predicted = endpoint.predict_finish(now_ns, profile)
            if (
                self.fallback_calendar is not None
                and predicted > request.fallback_latest_start_ns
            ):
                self.metrics["nrx_cutoff_rejected"] += 1
                continue
            if policy != "predicted_finish" or (
                predicted <= request.deadline_ns - self.commit_guard_ns
            ):
                ranked.append((predicted, endpoint.endpoint_id, endpoint, profile))

        if policy == "static":
            ranked = sorted(ranked, key=lambda item: item[1])[:1]
        elif policy == "round_robin" and ranked:
            ranked = sorted(ranked, key=lambda item: item[1])
            offset = self._round_robin_counter % len(ranked)
            ranked = ranked[offset:] + ranked[:offset]
            self._round_robin_counter += 1
        elif policy == "shortest_queue":
            ranked = sorted(
                ranked,
                key=lambda item: (
                    item[2].snapshot()["outstanding"], item[0], item[1]
                ),
            )
        else:
            ranked = sorted(ranked)

        for _, _, endpoint, profile in ranked:
            fallback_reservation = None
            if self.fallback_calendar is not None:
                fallback_reservation = self.fallback_calendar.reserve(request)
                if fallback_reservation is None:
                    self.metrics["fallback_credit_rejected"] += 1
                    break
            reservation = endpoint.reserve(
                request,
                now_ns,
                profile,
                self.commit_guard_ns,
                enforce_deadline=policy == "predicted_finish",
                latest_finish_ns=(
                    request.fallback_latest_start_ns
                    if self.fallback_calendar is not None else None
                ),
            )
            if reservation is not None:
                self.metrics["admitted"] += 1
                if fallback_reservation is not None:
                    self.metrics["dual_reserved"] += 1
                return DartTransaction(
                    request,
                    reservation,
                    fallback_reservation,
                    now_ns,
                )
            if (
                fallback_reservation is not None
                and self.fallback_calendar.release(fallback_reservation)
            ):
                self.metrics["fallback_credit_rolled_back"] += 1

        self.metrics["rejected_to_conventional"] += 1
        return DartTransaction(request, None, None, now_ns)

    def reserve_mandatory(
        self, request: DartRequest, now_ns: int
    ) -> Optional[DartTransaction]:
        """Reserve the conventional lane before deciding on optional NRx.

        Unlike the legacy one-step ``submit`` rejection path, a mandatory-only
        request also holds a physical recovery credit. A multi-request
        controller must treat ``None`` as unprotected overload, not success.
        """

        if self.fallback_calendar is None:
            raise RuntimeError("mandatory-first admission needs a fallback calendar")
        credit = self.fallback_calendar.reserve(request)
        if credit is None:
            self.metrics["mandatory_credit_rejected"] += 1
            return None
        self.metrics["mandatory_reserved"] += 1
        return DartTransaction(request, None, credit, now_ns)

    def admit_nrx(
        self,
        transaction: DartTransaction,
        now_ns: int,
        policy: str = "predicted_finish",
    ) -> bool:
        """Attach an optional endpoint to an existing mandatory reservation."""

        if policy not in {"static", "round_robin", "shortest_queue", "predicted_finish"}:
            raise ValueError(f"unsupported routing policy: {policy}")
        request = transaction.request
        if transaction.fallback_reservation is None:
            self.metrics["nrx_without_mandatory_credit_rejected"] += 1
            return False
        ranked = []
        for index, endpoint in enumerate(self.endpoints.values()):
            if not request.candidate_bitmap & (1 << index):
                continue
            try:
                profile = self.profiles.get(
                    endpoint.endpoint_id, request, self.background_mode
                )
            except KeyError:
                self.metrics["missing_profile"] += 1
                continue
            predicted = endpoint.predict_finish(now_ns, profile)
            if predicted <= request.fallback_latest_start_ns:
                ranked.append((predicted, endpoint.endpoint_id, endpoint, profile))
            else:
                self.metrics["nrx_cutoff_rejected"] += 1
        if policy == "static":
            ranked = sorted(ranked, key=lambda item: item[1])[:1]
        elif policy == "round_robin" and ranked:
            ranked = sorted(ranked, key=lambda item: item[1])
            offset = self._round_robin_counter % len(ranked)
            ranked = ranked[offset:] + ranked[:offset]
            self._round_robin_counter += 1
        elif policy == "shortest_queue":
            ranked = sorted(
                ranked,
                key=lambda item: (item[2].snapshot()["outstanding"], item[0], item[1]),
            )
        else:
            ranked = sorted(ranked)
        for _, _, endpoint, profile in ranked:
            reservation = endpoint.reserve(
                request,
                now_ns,
                profile,
                self.commit_guard_ns,
                latest_finish_ns=request.fallback_latest_start_ns,
            )
            if reservation is None:
                continue
            if transaction.attach_nrx(reservation):
                self.metrics["nrx_attached_to_mandatory"] += 1
                return True
            endpoint.release(reservation, now_ns)
            break
        self.metrics["nrx_optional_rejected"] += 1
        return False

    def _release_fallback(
        self,
        transaction: DartTransaction,
        after_execution: bool,
    ) -> bool:
        if self.fallback_calendar is None:
            return False
        reservation = transaction.detach_fallback_reservation(after_execution)
        if reservation is None:
            return False
        released = self.fallback_calendar.release(reservation)
        if released:
            self.metrics["fallback_credit_released"] += 1
        return released

    def complete_nrx(
        self,
        transaction: DartTransaction,
        slot_id: int,
        epoch: int,
        now_ns: int,
        payload_visible: bool = True,
    ) -> bool:
        reservation = transaction.reservation
        result_slot = reservation.output_slot if reservation is not None else 0
        endpoint = (
            self.endpoints[reservation.endpoint_id]
            if reservation is not None
            else None
        )
        endpoint_valid = (
            endpoint.completion_valid(reservation)
            if endpoint is not None
            else False
        )
        if not payload_visible:
            self.metrics["visibility_dropped"] += 1
        if reservation is not None and not endpoint_valid:
            self.metrics["endpoint_epoch_dropped"] += 1
        committed = False
        if payload_visible and endpoint_valid:
            committed = transaction.try_commit(
                CommitKind.NRX, slot_id, epoch, result_slot, now_ns
            )
        if (
            reservation is not None
            and slot_id == reservation.slot_id
            and epoch == reservation.epoch
        ):
            if endpoint.release(reservation, now_ns):
                self.metrics["released"] += 1
                with transaction._lock:
                    transaction.nrx_observed_ns = now_ns
        if committed:
            self._release_fallback(transaction, after_execution=False)
        self.metrics["nrx_committed" if committed else "nrx_dropped"] += 1
        return committed

    def replan_fallbacks(
        self,
        moves: Iterable[Tuple[DartTransaction, int, int]],
        now_ns: int,
    ) -> Optional[Dict[Tuple[int, int], FallbackReservation]]:
        """Atomically update calendar and transaction references for a plan.

        Each move is ``(transaction, new_start_ns, lane)``. Optional NRx must
        have physically completed before its credit can be moved. The original
        NRx cutoff in the request is unchanged.
        """

        if self.fallback_calendar is None:
            return None
        plan = list(moves)
        if not plan or len({id(tx) for tx, _, _ in plan}) != len(plan):
            return None
        transactions = sorted((tx for tx, _, _ in plan), key=id)
        for transaction in transactions:
            transaction._lock.acquire()
        try:
            if any(
                tx.commit_state is not CommitState.OPEN
                or tx.fallback_reservation is None
                or tx.fallback_started_ns is not None
                or (tx.reservation is not None and tx.nrx_observed_ns is None)
                for tx, _, _ in plan
            ):
                self.metrics["fallback_replan_rejected"] += 1
                return None
            proposal = [
                (tx.request, tx.fallback_reservation, start_ns, lane)
                for tx, start_ns, lane in plan
            ]
            replacements = self.fallback_calendar.replan_atomic(proposal, now_ns)
            if replacements is None:
                self.metrics["fallback_replan_rejected"] += 1
                return None
            for tx, _, _ in plan:
                tx.fallback_reservation = replacements[(tx.request.slot_id, tx.request.epoch)]
            self.metrics["fallback_replanned"] += len(plan)
            return replacements
        finally:
            for transaction in reversed(transactions):
                transaction._lock.release()

    def start_fallback(self, transaction: DartTransaction, now_ns: int) -> bool:
        started = transaction.start_fallback(now_ns)
        self.metrics["fallback_started" if started else "fallback_not_started"] += 1
        return started

    def start_fallback_early(
        self, transaction: DartTransaction, now_ns: int
    ) -> bool:
        """Retain recovery-calendar capacity when starting before latest time."""

        if self.fallback_calendar is None:
            self.metrics["fallback_early_not_started"] += 1
            return False
        started = transaction.start_fallback_early(now_ns, self.fallback_calendar)
        self.metrics["fallback_started" if started else "fallback_not_started"] += 1
        self.metrics[
            "fallback_early_started" if started else "fallback_early_not_started"
        ] += 1
        return started

    def complete_conventional(
        self, transaction: DartTransaction, now_ns: int, result_slot: int = 0
    ) -> bool:
        committed = transaction.try_commit(
            CommitKind.CONVENTIONAL,
            transaction.request.slot_id,
            transaction.request.epoch,
            result_slot,
            now_ns,
        )
        self._release_fallback(transaction, after_execution=True)
        self.metrics["conv_committed" if committed else "conv_dropped"] += 1
        return committed

    def expire(self, transaction: DartTransaction, now_ns: int) -> bool:
        expired = transaction.expire(now_ns)
        if expired:
            self.metrics["deadline_miss"] += 1
            self._release_fallback(transaction, after_execution=True)
        return expired

    def try_lease(
        self,
        endpoint_id: str,
        now_ns: int,
        earliest_latest_start_ns: int,
        qmax_ns: int,
        drain_guard_ns: int,
        units: Iterable[BackgroundUnit],
    ) -> Optional[LeaseReservation]:
        if qmax_ns <= 0:
            raise ValueError("qmax must be positive")
        next_guard = min(earliest_latest_start_ns, now_ns + qmax_ns)
        lease = self.endpoints[endpoint_id].reserve_lease(
            now_ns, next_guard, drain_guard_ns, units
        )
        self.metrics["lease_admitted" if lease else "lease_rejected"] += 1
        return lease


def request_to_dict(request: DartRequest) -> dict:
    return dataclasses.asdict(request)


def reservation_to_dict(reservation: Optional[Reservation]) -> Optional[dict]:
    return dataclasses.asdict(reservation) if reservation is not None else None
