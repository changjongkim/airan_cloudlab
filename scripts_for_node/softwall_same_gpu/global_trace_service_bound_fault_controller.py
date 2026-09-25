#!/usr/bin/env python3
"""Frozen controller plus attempted-RPC telemetry for broker fail-stop arms."""

import global_trace_fully_budgeted_fault_contained_controller as controller
from instrumented_deadline_global_trace_client_v2 import (
    InstrumentedDeadlineGlobalTraceClientV2,
)


def instrumented_global_queue_status(queue):
    status = {"enabled": getattr(queue, "enabled", True),
              "faults": list(getattr(queue, "faults", []))}
    telemetry = getattr(queue, "rpc_telemetry", None)
    if telemetry is not None:
        status["rpc_telemetry"] = telemetry()
    return status


controller.DeadlineFaultContainedGlobalTraceClient = InstrumentedDeadlineGlobalTraceClientV2
controller.global_queue_status = instrumented_global_queue_status


if __name__ == "__main__":
    controller.main()
