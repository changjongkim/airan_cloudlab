#!/usr/bin/env python3
"""Controller wrapper separating socket timeout from admission wall bound."""

import os
import sys

import global_trace_fully_budgeted_fault_contained_controller as controller
from instrumented_deadline_global_trace_client_v2 import (
    InstrumentedDeadlineGlobalTraceClientV2,
)


def pop_rpc_bound(argv):
    name = "--global-broker-rpc-bound-ms"
    for index, argument in enumerate(list(argv)):
        if argument == name:
            if index + 1 >= len(argv):
                raise SystemExit("%s requires a value" % name)
            value = float(argv[index + 1])
            del argv[index:index + 2]
            return value
        if argument.startswith(name + "="):
            value = float(argument.split("=", 1)[1])
            del argv[index]
            return value
    environment = os.environ.get("SOFTWALL_BROKER_RPC_ADMISSION_BOUND_MS")
    if environment is not None:
        return float(environment)
    raise SystemExit("%s is required" % name)


RPC_BOUND_MS = pop_rpc_bound(sys.argv)
if RPC_BOUND_MS <= 0:
    raise SystemExit("global broker RPC admission bound must be positive")


def qualified_composed_ai_guard_ns(base_guard_ns, broker_rpc_timeout_ms,
                                   global_broker_enabled):
    del broker_rpc_timeout_ms
    broker_budget_ns = (
        3 * round(RPC_BOUND_MS * 1e6) if global_broker_enabled else 0
    )
    return base_guard_ns + broker_budget_ns, broker_budget_ns


def qualified_global_queue_status(queue):
    status = {
        "enabled": getattr(queue, "enabled", True),
        "faults": list(getattr(queue, "faults", [])),
        "broker_rpc_socket_timeout_ms": getattr(queue, "rpc_timeout_ms", None),
        "broker_rpc_admission_bound_ms": RPC_BOUND_MS,
    }
    telemetry = getattr(queue, "rpc_telemetry", None)
    if telemetry is not None:
        status["rpc_telemetry"] = telemetry()
    return status


controller.DeadlineFaultContainedGlobalTraceClient = (
    InstrumentedDeadlineGlobalTraceClientV2
)
controller.composed_ai_guard_ns = qualified_composed_ai_guard_ns
controller.global_queue_status = qualified_global_queue_status


if __name__ == "__main__":
    controller.main()
