#!/usr/bin/env python3
"""Global lease broker with a deterministic fail-stop injection point."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import socketserver
import threading
import time
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


class GlobalTraceLeaseState:
    """One trace queue with prepare/commit/abort/complete transitions."""

    def __init__(self, trace: dict, trace_sha256: str, participants: int,
                 policy: str) -> None:
        if participants <= 0 or policy not in ("global", "static_partition"):
            raise ValueError("invalid broker configuration")
        self.trace = trace
        self.trace_sha256 = trace_sha256
        self.participants = participants
        self.policy = policy
        self.epoch_ns = None
        self.requests = []
        self.tokens = {}
        self.registered_homes = set()
        self.finalized_homes = set()
        self.events = []
        self.generation = 0
        self.duplicate_commit_count = 0
        self.lock = threading.Lock()

    def initialize(self, home: int, epoch_ns: int, trace_sha256: str) -> dict:
        with self.lock:
            if not 0 <= home < self.participants:
                raise ValueError("invalid home")
            if trace_sha256 != self.trace_sha256:
                raise ValueError("trace hash mismatch")
            if self.epoch_ns is None:
                self.epoch_ns = epoch_ns
                for raw in self.trace["requests"]:
                    request = dict(raw)
                    request["arrival_ns"] = epoch_ns + round(request["arrival_ms"] * 1e6)
                    request["deadline_ns"] = epoch_ns + round(request["deadline_ms"] * 1e6)
                    request["state"] = "future"
                    request["owner"] = None
                    request["token"] = None
                    request["committed_owner"] = None
                    self.requests.append(request)
            elif self.epoch_ns != epoch_ns:
                raise ValueError("all homes must use the same trace epoch")
            self.registered_homes.add(home)
            return {"epoch_ns": self.epoch_ns,
                    "registered_homes": sorted(self.registered_homes)}

    def _observe_locked(self, now_ns: int) -> None:
        for request in self.requests:
            if request["state"] == "future" and request["arrival_ns"] <= now_ns:
                request["state"] = "ready"
                request["observed_ns"] = now_ns
            if request["state"] == "ready" and request["deadline_ns"] <= now_ns:
                request["state"] = "expired"
                request["expired_ns"] = now_ns

    def prepare(self, home: int, now_ns: int, horizon_ns: int,
                guard_ns: int, bounds_ns: dict) -> dict | None:
        with self.lock:
            if home not in self.registered_homes:
                raise ValueError("home is not initialized")
            self._observe_locked(now_ns)
            ready = [request for request in self.requests
                     if request["state"] == "ready"]
            if self.policy == "static_partition":
                ready = [request for request in ready
                         if request["source_order"] % self.participants == home]
            if not ready:
                return None
            earliest = min(request["deadline_ns"] for request in ready)
            selected = None
            for request in sorted(
                    (request for request in ready
                     if request["deadline_ns"] == earliest),
                    key=lambda item: item["source_order"]):
                bound = bounds_ns.get(str(request["context_length"]))
                if bound is None:
                    raise KeyError("missing context bound")
                if now_ns + bound + guard_ns <= min(horizon_ns, earliest):
                    selected = request
                    break
            if selected is None:
                return None
            self.generation += 1
            token = f"g{self.generation}:h{home}:r{selected['request_id']}"
            selected["state"] = "held"
            selected["owner"] = home
            selected["token"] = token
            selected["held_ns"] = now_ns
            self.tokens[token] = selected
            self.events.append({"event": "prepare", "token": token,
                                "home": home, "request_id": selected["request_id"],
                                "time_ns": now_ns, "horizon_ns": horizon_ns})
            result = {
                key: selected[key]
                for key in (
                    "request_id", "source_order", "context_length", "value_tokens",
                    "arrival_ns", "deadline_ns",
                )
            }
            result["broker_token"] = token
            return result

    def commit(self, home: int, token: str, now_ns: int) -> dict:
        with self.lock:
            request = self.tokens.get(token)
            if request is None or request["state"] != "held" or request["owner"] != home:
                raise ValueError("invalid held token")
            if request["committed_owner"] is not None:
                self.duplicate_commit_count += 1
                raise ValueError("request was already committed")
            request["state"] = "inflight"
            request["committed_owner"] = home
            request["committed_ns"] = now_ns
            self.events.append({"event": "commit", "token": token,
                                "home": home, "request_id": request["request_id"],
                                "time_ns": now_ns})
            return {"request_id": request["request_id"]}

    def abort(self, home: int, token: str, now_ns: int) -> dict:
        with self.lock:
            request = self.tokens.get(token)
            if request is None or request["state"] != "held" or request["owner"] != home:
                raise ValueError("invalid held token")
            request["owner"] = None
            request["token"] = None
            request["state"] = (
                "expired" if request["deadline_ns"] <= now_ns else "ready"
            )
            if request["state"] == "expired":
                request["expired_ns"] = now_ns
            del self.tokens[token]
            self.events.append({"event": "abort", "token": token,
                                "home": home, "request_id": request["request_id"],
                                "time_ns": now_ns})
            return {"request_id": request["request_id"], "state": request["state"]}

    def complete(self, home: int, token: str, returned_ns: int) -> dict:
        with self.lock:
            request = self.tokens.get(token)
            if request is None or request["state"] != "inflight" or request["owner"] != home:
                raise ValueError("invalid inflight token")
            request["returned_ns"] = returned_ns
            request["state"] = (
                "completed" if returned_ns <= request["deadline_ns"] else "late"
            )
            del self.tokens[token]
            self.events.append({"event": "complete", "token": token,
                                "home": home, "request_id": request["request_id"],
                                "time_ns": returned_ns, "state": request["state"]})
            return {"request_id": request["request_id"], "state": request["state"]}

    def finalize(self, home: int, now_ns: int) -> dict:
        with self.lock:
            if home not in self.registered_homes:
                raise ValueError("home is not initialized")
            self._observe_locked(now_ns)
            self.finalized_homes.add(home)
            return self._summary_locked()

    def _summary_locked(self) -> dict:
        offered = [request for request in self.requests
                   if request["state"] != "future"]
        completed = [request for request in self.requests
                     if request["state"] == "completed"]
        counts = {}
        for request in self.requests:
            counts[request["state"]] = counts.get(request["state"], 0) + 1
        per_home = {}
        for request in self.requests:
            owner = request.get("committed_owner")
            if owner is None:
                continue
            key = str(owner)
            bucket = per_home.setdefault(key, {"committed": 0, "completed": 0,
                                                "timely_value_tokens": 0})
            bucket["committed"] += 1
            if request["state"] == "completed":
                bucket["completed"] += 1
                bucket["timely_value_tokens"] += request["value_tokens"]
        return {
            "offered_requests": len(offered),
            "offered_value_tokens": sum(request["value_tokens"] for request in offered),
            "timely_requests": len(completed),
            "timely_value_tokens": sum(request["value_tokens"] for request in completed),
            "states": counts,
            "per_home": per_home,
            "registered_homes": sorted(self.registered_homes),
            "finalized_homes": sorted(self.finalized_homes),
            "outstanding_tokens": len(self.tokens),
            "duplicate_commit_count": self.duplicate_commit_count,
        }

    def snapshot(self) -> dict:
        with self.lock:
            return self._summary_locked()

    def output(self) -> dict:
        with self.lock:
            return {
                "schema": "softwall-global-trace-lease-broker-v1",
                "trace_sha256": self.trace_sha256,
                "participants": self.participants,
                "policy": self.policy,
                "epoch_ns": self.epoch_ns,
                "summary": self._summary_locked(),
                "events": list(self.events),
                "requests": list(self.requests),
            }


class BrokerServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path: str, state: GlobalTraceLeaseState,
                 crash_after_commit_home: int | None = None,
                 crash_after_commit_number: int | None = None):
        self.state = state
        self.stop_requested = False
        self.crash_after_commit_home = crash_after_commit_home
        self.crash_after_commit_number = crash_after_commit_number
        self.commit_counts = {}
        self.faults = []
        self.fault_lock = threading.Lock()
        super().__init__(path, BrokerHandler)

    def should_crash_after_commit(self, request: dict, result: dict) -> bool:
        """Apply a commit once, then terminate before sending its reply.

        The state transition precedes this hook, so the client cannot know
        whether commit succeeded.  That is the ambiguity the runtime must
        contain without retrying the AI request.
        """
        home = request["home"]
        with self.fault_lock:
            count = self.commit_counts.get(home, 0) + 1
            self.commit_counts[home] = count
            if (home != self.crash_after_commit_home
                    or count != self.crash_after_commit_number):
                return False
            self.faults.append({
                "fault": "broker_exit_after_commit_apply",
                "home": home,
                "commit_number": count,
                "token": request["token"],
                "request_id": result["request_id"],
                "injected_ns": time.perf_counter_ns(),
            })
            return True


class BrokerHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            raw = self.rfile.readline()
            if not raw:
                return
            try:
                request = json.loads(raw)
                op = request["op"]
                if op == "init":
                    result = self.server.state.initialize(
                        request["home"], request["epoch_ns"], request["trace_sha256"]
                    )
                elif op == "prepare":
                    result = self.server.state.prepare(
                        request["home"], request["now_ns"], request["horizon_ns"],
                        request["guard_ns"], request["bounds_ns"],
                    )
                elif op == "commit":
                    result = self.server.state.commit(
                        request["home"], request["token"], request["now_ns"]
                    )
                    if self.server.should_crash_after_commit(request, result):
                        # The C128 injector wrote a marker on the shared
                        # filesystem here and stalled both controller RPCs.
                        # Exit directly so observation cannot perturb the
                        # timed fault path.  The parent records exit status;
                        # clients record the ambiguous token independently.
                        os._exit(86)
                elif op == "abort":
                    result = self.server.state.abort(
                        request["home"], request["token"], request["now_ns"]
                    )
                elif op == "complete":
                    result = self.server.state.complete(
                        request["home"], request["token"], request["returned_ns"]
                    )
                elif op == "finalize":
                    result = self.server.state.finalize(
                        request["home"], request["now_ns"]
                    )
                elif op == "snapshot":
                    result = self.server.state.snapshot()
                elif op == "stop":
                    result = self.server.state.snapshot()
                    self.server.stop_requested = True
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                else:
                    raise ValueError("unknown operation")
                response = {"ok": True, "result": result}
            except Exception as error:
                response = {"ok": False, "type": type(error).__name__,
                            "error": str(error)}
            self.wfile.write(json.dumps(response).encode() + b"\n")
            self.wfile.flush()
            if request.get("op") == "stop":
                return


class GlobalTraceLeaseClient:
    """Queue-compatible client used by one sharded home controller."""

    def __init__(self, path: str, home: int, epoch_ns: int,
                 trace_sha256: str) -> None:
        self.path = path
        self.home = home
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.connect(path)
        self.channel = self.socket.makefile("rwb", buffering=0)
        self.lock = threading.Lock()
        self._rpc({"op": "init", "home": home, "epoch_ns": epoch_ns,
                   "trace_sha256": trace_sha256})

    def _rpc(self, payload: dict):
        with self.lock:
            self.channel.write(json.dumps(payload).encode() + b"\n")
            raw = self.channel.readline()
        if not raw:
            raise ConnectionError("global broker closed")
        response = json.loads(raw)
        if not response.get("ok"):
            raise RuntimeError(f"global broker rejected request: {response}")
        return response["result"]

    def peek_edf_fitting(self, now_ns: int, horizon_ns: int, guard_ns: int,
                         bounds_ns: dict) -> dict | None:
        return self._rpc({
            "op": "prepare", "home": self.home, "now_ns": now_ns,
            "horizon_ns": horizon_ns, "guard_ns": guard_ns,
            "bounds_ns": {str(key): value for key, value in bounds_ns.items()},
        })

    def commit(self, request: dict) -> None:
        self._rpc({"op": "commit", "home": self.home,
                   "token": request["broker_token"],
                   "now_ns": time.perf_counter_ns()})

    def release(self, request: dict) -> None:
        self._rpc({"op": "abort", "home": self.home,
                   "token": request["broker_token"],
                   "now_ns": time.perf_counter_ns()})

    def complete(self, request: dict, returned_ns: int) -> None:
        self._rpc({"op": "complete", "home": self.home,
                   "token": request["broker_token"],
                   "returned_ns": returned_ns})

    def finalize(self, now_ns: int) -> None:
        self._rpc({"op": "finalize", "home": self.home, "now_ns": now_ns})

    def summary(self) -> dict:
        return self._rpc({"op": "snapshot"})

    def close(self) -> None:
        try:
            self.channel.close()
        finally:
            self.socket.close()


def stop_broker(path: str) -> None:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(path)
    channel = client.makefile("rwb", buffering=0)
    channel.write(b'{"op":"stop"}\n')
    raw = channel.readline()
    if not raw or not json.loads(raw).get("ok"):
        raise RuntimeError("broker stop failed")
    channel.close()
    client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--stop-only", action="store_true")
    parser.add_argument("--trace")
    parser.add_argument("--trace-sha256")
    parser.add_argument("--participants", type=int)
    parser.add_argument("--policy", choices=("global", "static_partition"), default="global")
    parser.add_argument("--crash-after-commit-home", type=int)
    parser.add_argument("--crash-after-commit-number", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.stop_only:
        stop_broker(args.socket)
        return
    if not all((args.trace, args.trace_sha256, args.participants, args.output)):
        parser.error("serve mode requires trace, hash, participants, and output")
    trace_path = Path(args.trace)
    observed_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    if observed_sha != args.trace_sha256:
        raise RuntimeError("sealed trace hash mismatch")
    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass
    state = GlobalTraceLeaseState(
        json.loads(trace_path.read_text(encoding="utf-8")), observed_sha,
        args.participants, args.policy,
    )
    if ((args.crash_after_commit_home is None)
            != (args.crash_after_commit_number is None)):
        parser.error("both broker-crash fault arguments are required")
    if args.crash_after_commit_number is not None and args.crash_after_commit_number <= 0:
        parser.error("broker-crash fault number must be positive")
    if (args.crash_after_commit_home is not None
            and not 0 <= args.crash_after_commit_home < args.participants):
        parser.error("broker-crash fault home is outside participant range")
    server = BrokerServer(
        str(socket_path), state,
        args.crash_after_commit_home, args.crash_after_commit_number,
    )
    try:
        server.serve_forever(poll_interval=0.05)
    finally:
        server.server_close()
        output = state.output()
        output["schema"] = "softwall-global-trace-lease-broker-fault-v1"
        output["fault_configuration"] = {
            "crash_after_commit_home": args.crash_after_commit_home,
            "crash_after_commit_number": args.crash_after_commit_number,
        }
        output["faults"] = list(server.faults)
        output["commit_counts"] = dict(server.commit_counts)
        atomic_json(Path(args.output), output)
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
