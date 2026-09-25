#!/usr/bin/env python3
"""Run the frozen C136 controller with added broker-RPC wall-time telemetry."""

import global_trace_fully_budgeted_fault_contained_controller as controller
from instrumented_deadline_global_trace_client import (
    InstrumentedDeadlineGlobalTraceClient,
)


def instrumented_global_queue_status(queue):
    status = {
        "enabled": getattr(queue, "enabled", True),
        "faults": list(getattr(queue, "faults", [])),
    }
    telemetry = getattr(queue, "rpc_telemetry", None)
    if telemetry is not None:
        status["rpc_telemetry"] = telemetry()
    return status


controller.DeadlineFaultContainedGlobalTraceClient = InstrumentedDeadlineGlobalTraceClient
controller.global_queue_status = instrumented_global_queue_status


if __name__ == "__main__":
    controller.main()
