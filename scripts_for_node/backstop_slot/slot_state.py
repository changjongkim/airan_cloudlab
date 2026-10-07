"""Host shared state for one slot-scale run.

All processes of a run map the same file.  Writers store the timestamp before
the status word, and readers read the status before the timestamp, so a
reader never sees a status without its time on x86.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

HEADER_WORDS = 512
H_EPOCH_NS = 0
H_ABORT = 1
H_READY_BASE = 64          # one word per process slot, up to 448 processes

# Per (cell, period) record.
C_CONV_STATUS = 0          # 0 pending, 1 CRC pass, 2 CRC fail
C_CONV_START = 1
C_CONV_DONE = 2
C_NRX_STATUS = 3           # 0 pending, 1 pass, 2 fail, 3 skipped, 4 running
C_NRX_START = 4
C_NRX_DONE = 5
C_NRX_LANE = 6             # GPU of the lane that ran NeuralRx, -1 if none
C_NRX_REASON = 7           # 1 conv_fail, 2 latest_start, 3 arrival, 4 low_value
C_CONV_CBFAIL = 8          # code blocks whose CRC failed in the conventional decode
C_LEVEL = 9                # link adaptation: index of the MCS level the slot was sent with
C_NRX_MASK = 10            # NeuralRx: UEs whose CRC passed (bit per UE), written before C_NRX_STATUS
C_WORDS = 12

# NeuralRx lane mailbox (controller -> lane), one row of ``lanes`` per lane.
L_ASSIGN_SEQ = 0
L_CELL = 1
L_PERIOD = 2
L_DONE_SEQ = 3
# Per-GPU AI mailbox (controller -> AI worker).
# Words 0-3 are used only by the dynamic-share baseline (several workers per GPU, one active).
A_DYN_SCAN = 0             # next entry of the global request table to look at (shared by the workers)
A_DYN_PULLED = 1           # requests taken from the table on this GPU, all workers together
A_DYN_ACTIVE = 2           # id (1..K) of the worker that may take new requests; 0 = none
A_DYN_OWNER = 3            # id of the worker currently taking requests; 0 = free
A_GRANT_SEQ = 4
A_BUDGET_NS = 5
A_NOT_AFTER = 6
A_DONE_SEQ = 7
A_HAS_WORK = 8
A_RUNNING = 9              # 1 while a granted piece is on the GPU
A_PIECE_END_NS = 10        # expected end of the running piece (start + bound)
A_MAX_CHUNK = 11           # largest AI chunk (tokens) the current grant allows; 0 = any
A_BACKLOG_TOKENS = 12      # AI worker: prompt tokens still to prefill in its queue
A_RATE_TPS = 13            # AI worker: recent prefill rate (tokens per second)
A_PULLED = 14              # AI worker: requests it has taken from the global table
A_CUR_CHUNK = 15           # AI worker: chunk size (tokens) of the chunk in progress, 0 if none
A_STOP = 16                # controller: 1 = end the running piece after the units already on the GPU
A_CHATS_DONE = 17          # AI worker (classes5): responses finished or given up, for the session count
A_BACKLOG_T1 = 18          # AI worker (classes5): backlog of work with a time limit up to 100 ms (ns of unit bounds)
A_BACKLOG_T2 = 19          # AI worker (classes5): the same, time limit up to 250 ms
A_RATE_T1 = 20             # AI worker (classes5): service rate for work with a time limit up to 100 ms
A_SLICE = 21               # controller: 1 = the granted piece runs on the SM slice of the GPU (ai.slice_sms), 0 = on the whole GPU
MAX_REQUESTS = 16384       # global AI request table (dispatch mode)
REQ_WORDS = 4
R_ARRIVAL, R_LENGTH, R_GPU, R_SEQ = 0, 1, 2, 3   # R_GPU: -1 waiting, -2 rejected, else GPU
G_WORDS = 24
MAX_LANES = 64             # NeuralRx lane mailboxes (lane id = gpu * lanes_per_gpu + j)
LANE_WORDS = 8

PENDING, PASS, FAIL, SKIPPED, RUNNING, DROPPED = 0, 1, 2, 3, 4, 5
REASON_CONV_FAIL, REASON_LATEST_START, REASON_ARRIVAL, REASON_LOW_VALUE = 1, 2, 3, 4


def now_ns() -> int:
    return time.monotonic_ns()


class SlotState:
    def __init__(self, path: str | Path, num_cells: int, num_periods: int,
                 num_gpus: int, create: bool = False) -> None:
        self.path = Path(path)
        self.num_cells = num_cells
        self.num_periods = num_periods
        self.num_gpus = num_gpus
        words = (HEADER_WORDS + num_cells * num_periods * C_WORDS + num_gpus * G_WORDS
                 + MAX_LANES * LANE_WORDS + MAX_REQUESTS * REQ_WORDS)
        if create:
            with open(self.path, "wb") as handle:
                handle.truncate(words * 8)
        self._map = np.memmap(self.path, dtype=np.int64, mode="r+", shape=(words,))
        self.header = self._map[:HEADER_WORDS]
        begin = HEADER_WORDS
        end = begin + num_cells * num_periods * C_WORDS
        self.cells = self._map[begin:end].reshape(num_cells, num_periods, C_WORDS)
        gpu_end = end + num_gpus * G_WORDS
        self.gpus = self._map[end:gpu_end].reshape(num_gpus, G_WORDS)
        lane_end = gpu_end + MAX_LANES * LANE_WORDS
        self.lanes = self._map[gpu_end:lane_end].reshape(MAX_LANES, LANE_WORDS)
        self.requests = self._map[lane_end:].reshape(MAX_REQUESTS, REQ_WORDS)

    def mark_ready(self, slot: int) -> None:
        self.header[H_READY_BASE + slot] = 1

    def ready_count(self, count: int) -> int:
        return int(self.header[H_READY_BASE:H_READY_BASE + count].sum())

    def wait_epoch(self, timeout_s: float = 600.0) -> int:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            epoch = int(self.header[H_EPOCH_NS])
            if epoch:
                return epoch
            if self.header[H_ABORT]:
                raise RuntimeError("run aborted before epoch")
            time.sleep(0.0005)
        raise TimeoutError("epoch was never published")

    def set_conv(self, cell: int, period: int, start: int, done: int, ok: bool,
                 cb_fail: int = 0) -> None:
        row = self.cells[cell, period]
        row[C_CONV_CBFAIL] = cb_fail
        row[C_CONV_START] = start
        row[C_CONV_DONE] = done
        row[C_CONV_STATUS] = PASS if ok else FAIL

    def conv_status(self, cell: int, period: int) -> int:
        return int(self.cells[cell, period, C_CONV_STATUS])

    def set_nrx_running(self, cell: int, period: int, start: int) -> None:
        row = self.cells[cell, period]
        row[C_NRX_START] = start
        row[C_NRX_STATUS] = RUNNING

    def set_nrx(self, cell: int, period: int, status: int, done: int) -> None:
        row = self.cells[cell, period]
        row[C_NRX_DONE] = done
        row[C_NRX_STATUS] = status

    def flush(self) -> None:
        self._map.flush()


def spin_until(target_ns: int) -> int:
    """Busy-wait on CLOCK_MONOTONIC, the clock every process of a run shares."""
    current = now_ns()
    while current < target_ns:
        current = now_ns()
    return current


def unlink_quietly(path: str | Path) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
