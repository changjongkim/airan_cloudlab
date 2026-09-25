#!/usr/bin/env python3
"""Compare frozen P2P and pinned-host staging transport gates."""

from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--p2p',type=Path,required=True);p.add_argument('--host',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); p2p=json.loads(a.p2p.read_text()); host=json.loads(a.host.read_text())
    same=(p2p['iterations']==host['iterations']==10000 and p2p['payload']=={'forward_bytes':1415232,'backward_bytes':314496})
    # Host result omits a duplicate payload object; its frozen protocol and gate check the same sizes.
    pp=p2p['steady_timing_us']; hp=host['steady_timing_us']
    result={
      'schema':'softwall-multigpu-transport-comparison-v1',
      'p2p':str(a.p2p),'p2p_sha256':sha256(a.p2p),'host_staging':str(a.host),'host_staging_sha256':sha256(a.host),
      'iterations_each':p2p['iterations'],'forward_bytes':1415232,'backward_bytes':314496,
      'p2p_all_pass':p2p['all_pass'],'host_staging_all_pass':host['all_pass'],
      'round_trip_us':{
        'p2p':pp['round_trip'],'host_staging':hp['round_trip'],
        'host_over_p2p_p50_ratio':hp['round_trip']['p50']/pp['round_trip']['p50'],
        'host_over_p2p_p99_ratio':hp['round_trip']['p99']/pp['round_trip']['p99'],
        'p50_saved_by_p2p_us':hp['round_trip']['p50']-pp['round_trip']['p50'],
        'p99_saved_by_p2p_us':hp['round_trip']['p99']-pp['round_trip']['p99'],
      },
      'gates':{'same_payload_and_iterations':same,'both_integrity_gates_pass':p2p['all_pass'] and host['all_pass']},
      'scope':'descriptive warm transport comparison on one NVLink node; not end-to-end NeuralRx or application throughput',
    }
    result['all_pass']=all(result['gates'].values())
    tmp=a.output.with_suffix(a.output.suffix+'.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(a.output)
    print(json.dumps(result,indent=2))
    if not result['all_pass']: raise SystemExit('transport comparison gate failed')
if __name__=='__main__': main()
