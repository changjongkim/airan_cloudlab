#!/usr/bin/env python3
"""V14 controller: defer prepare/complete and charge only launch commit."""

import os
import sys

import global_trace_fully_budgeted_fault_contained_controller as controller
from pipelined_global_trace_client import PipelinedGlobalTraceClient


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
    raise SystemExit("global broker commit admission bound must be positive")


def pipelined_composed_ai_guard_ns(base_guard_ns, broker_rpc_timeout_ms,
                                  global_broker_enabled):
    del broker_rpc_timeout_ms
    # Prepare and complete are handed to a separate connection.  Only the
    # launch-authorizing commit can delay the RAN executor.
    broker_budget_ns = (
        round(RPC_BOUND_MS * 1e6) if global_broker_enabled else 0
    )
    return base_guard_ns + broker_budget_ns, broker_budget_ns


def pipelined_global_queue_status(queue):
    status = {
        "enabled": getattr(queue, "enabled", True),
        "faults": list(getattr(queue, "faults", [])),
        "control_plane": "pipelined-prepare-and-complete",
        "launch_rule": "physical AI launch only after synchronous commit ACK",
        "broker_rpc_socket_timeout_ms": getattr(queue, "rpc_timeout_ms", None),
        "broker_commit_admission_bound_ms": RPC_BOUND_MS,
    }
    telemetry = getattr(queue, "rpc_telemetry", None)
    if telemetry is not None:
        status["rpc_telemetry"] = telemetry()
    return status


controller.DeadlineFaultContainedGlobalTraceClient = PipelinedGlobalTraceClient
controller.composed_ai_guard_ns = pipelined_composed_ai_guard_ns
controller.global_queue_status = pipelined_global_queue_status


if __name__ == "__main__":
    controller.main()
