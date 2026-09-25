#!/usr/bin/env python3
"""Strict analyzer for the cross-process pinned-host staging gate."""

from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def summarize(values: list[float]) -> dict[str,float|int]:
    if not values: return {"count":0,"mean":0.0,"p50":0.0,"p99":0.0,"max":0.0}
    v=sorted(values)
    return {"count":len(v),"mean":sum(v)/len(v),"p50":v[round((len(v)-1)*.5)],"p99":v[round((len(v)-1)*.99)],"max":v[-1]}
def main() -> None:
    p=argparse.ArgumentParser()
    for name in ("protocol","owner","worker","output"): p.add_argument(f"--{name}",type=Path,required=True)
    a=p.parse_args(); protocol=json.loads(a.protocol.read_text()); owner=json.loads(a.owner.read_text()); worker=json.loads(a.worker.read_text())
    n=int(protocol["iterations"]); warm=min(int(protocol["timing_warmup_units"]),n)
    gates={
      "source_hashes":protocol["source_sha256"]==owner["source_sha256"]==worker["source_sha256"],
      "iteration_count":owner["completed_units"]==worker["completed_units"]==n,
      "integrity":owner["integrity_errors"]==worker["integrity_errors"]==0,
      "sequence":owner["sequences_contiguous"] and worker["sequences_contiguous"],
      "termination_lifecycle":owner["termination_acknowledged"] and worker["ipc_handles_closed_before_ack"],
      "payload":owner["forward_bytes"]==worker["forward_bytes"]==1415232 and owner["backward_bytes"]==worker["backward_bytes"]==314496,
      "no_error":owner["error"] is None and worker["error"] is None,
    }
    result={
      "schema":"softwall-multigpu-host-staging-gate-v1","protocol":str(a.protocol),"protocol_sha256":sha256(a.protocol),
      "owner":str(a.owner),"owner_sha256":sha256(a.owner),"worker":str(a.worker),"worker_sha256":sha256(a.worker),
      "iterations":n,"timing_warmup_units":warm,
      "cold_first_unit_us":{"round_trip":owner["records"][0]["round_trip_us"],"forward_gpu":worker["records"][0]["forward_gpu_us"],"backward_gpu":worker["records"][0]["backward_gpu_us"]},
      "steady_timing_us":{
        "round_trip":summarize([x["round_trip_us"] for x in owner["records"][warm:]]),
        "forward_gpu":summarize([x["forward_gpu_us"] for x in worker["records"][warm:]]),
        "forward_host":summarize([x["forward_host_us"] for x in worker["records"][warm:]]),
        "backward_gpu":summarize([x["backward_gpu_us"] for x in worker["records"][warm:]]),
        "backward_host":summarize([x["backward_host_us"] for x in worker["records"][warm:]]),
      },
      "gates":gates,"all_pass":all(gates.values()),"analyzer_sha256":sha256(Path(__file__).resolve()),
      "scope":"cross-process CUDA IPC plus pinned-host staging transport integrity; no NeuralRx, MPS, or deadline claim",
    }
    tmp=a.output.with_suffix(a.output.suffix+".tmp"); tmp.write_text(json.dumps(result,indent=2)); tmp.replace(a.output)
    print(json.dumps(result,indent=2))
    if not result["all_pass"]: raise SystemExit("host-staging gate failed")
if __name__=="__main__": main()

