#!/usr/bin/env python3
"""One AI joint RPC whose GPU completion can survive a delayed response.

Only a matching marker written after worker-side CUDA event synchronization is
accepted. A timeout disables every subsequent AI admission in this epoch.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from s2_runner import BackgroundClient


class FencedBackgroundClient(BackgroundClient):
    def run_joint(self, release_index: int, ai_deadline_ns: int,
                  budget_ms: float, lease_id: str,
                  marker_path: Path, earliest_recovery_ns: int,
                  guard_ns: int) -> bool:
        if not self.enabled:
            return False
        admitted_ns = time.perf_counter_ns()
        response = None
        transport_error = None
        try:
            request = json.dumps({"op": "run", "joint": True,
                                  "lease_id": lease_id}).encode() + b"\n"
            self.channel.write(request)
            payload = self.channel.readline()
            if not payload:
                raise ConnectionError("joint AI worker closed RPC channel")
            response = json.loads(payload)
            if not response.get("ok"):
                raise RuntimeError(f"joint AI worker rejected request: {response}")
        except (BrokenPipeError, ConnectionError, OSError,
                ValueError, RuntimeError) as error:
            transport_error = error
            self.enabled = False

        if transport_error is None:
            completed_ns = int(response["completed_ns"])
            returned_ns = time.perf_counter_ns()
            gpu_ms = float(response["gpu_ms"])
            if (completed_ns < admitted_ns or not math.isfinite(gpu_ms)
                    or returned_ns - admitted_ns > round(budget_ms * 1e6)
                    or returned_ns > ai_deadline_ns
                    or returned_ns + guard_ns > earliest_recovery_ns):
                self.enabled = False
                self.faults.append({"release_index": release_index,
                                    "type": "AIContractViolation",
                                    "detected_ns": returned_ns})
                return False
            self.records.append({
                "release_index": release_index,
                "admitted_ns": admitted_ns, "returned_ns": returned_ns,
                "gpu_completed_ns": completed_ns, "gpu_ms": gpu_ms,
                "execution_ms": (returned_ns - admitted_ns) / 1e6,
                "budget_ms": budget_ms, "budget_violation": False,
                "crossed_next_release": False,
                "fault_fence_confirmed": False,
            })
            return True

        # The worker wrote this file only after synchronizing its CUDA end
        # event. Neither elapsed time nor a socket timeout alone is a fence.
        latest_ns = min(admitted_ns + round(budget_ms * 1e6),
                        ai_deadline_ns, earliest_recovery_ns - guard_ns)
        marker = None
        while time.perf_counter_ns() <= latest_ns:
            try:
                candidate = json.loads(marker_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                candidate = None
            if candidate is not None and candidate.get("lease_id") == lease_id:
                marker = candidate
                break
            time.sleep(0.0002)
        detected_ns = time.perf_counter_ns()
        if marker is None:
            self.faults.append({
                "release_index": release_index,
                "type": "UnfencedJointAI",
                "detected_ns": detected_ns,
                "message": str(transport_error),
            })
            return False
        completed_ns = int(marker["completed_ns"])
        gpu_ms = float(marker["gpu_ms"])
        if (completed_ns < admitted_ns or completed_ns > latest_ns
                or detected_ns > latest_ns
                or detected_ns + guard_ns > earliest_recovery_ns
                or not math.isfinite(gpu_ms) or gpu_ms < 0):
            self.faults.append({
                "release_index": release_index,
                "type": "InvalidJointAIFence",
                "detected_ns": detected_ns,
                "message": str(transport_error),
            })
            return False
        self.records.append({
            "release_index": release_index,
            "admitted_ns": admitted_ns, "returned_ns": detected_ns,
            "gpu_completed_ns": completed_ns, "gpu_ms": gpu_ms,
            "execution_ms": (detected_ns - admitted_ns) / 1e6,
            "budget_ms": budget_ms, "budget_violation": False,
            "crossed_next_release": False,
            "fault_fence_confirmed": True,
            "marker_lease_id": lease_id,
        })
        self.faults.append({
            "release_index": release_index,
            "type": "RPCResponseTimeoutFenced",
            "detected_ns": detected_ns,
            "gpu_completed_ns": completed_ns,
            "lease_id": lease_id,
            "message": str(transport_error),
        })
        return True
