#!/usr/bin/env python3
"""Aggregate two formal local+two-remote SoftWall arms."""

from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path

ROOT=Path('/pscratch/sd/s/sgkim/kcj/airan_cloudlab')
RESULTS=ROOT/'results/softwall_multigpu'
def read(p:Path)->dict:return json.loads(p.read_text())
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def summarize(v:list[float])->dict[str,float|int]:
    if not v:return {'count':0,'mean':0.0,'p50':0.0,'p99':0.0,'max':0.0}
    s=sorted(v);return {'count':len(s),'mean':sum(s)/len(s),'p50':s[round((len(s)-1)*.5)],'p99':s[round((len(s)-1)*.99)],'max':s[-1]}
def main()->None:
    p=argparse.ArgumentParser();p.add_argument('--job',default='58815435');p.add_argument('--output',type=Path);a=p.parse_args()
    arms=[]
    for seed in ('s1','s2'):
        label=f'g4_three_endpoint_{seed}_job{a.job}'
        wrapper_path=RESULTS/f'{label}_result.json'; wrapper=read(wrapper_path)
        protocol_path=RESULTS/f'{label}_protocol.json'; protocol=read(protocol_path)
        controller=read(Path(wrapper['controller']))
        workers={
          'nrx0':read(Path(wrapper['local_worker'])),
          'nrx1':read(Path(wrapper['remote1_worker'])),
          'nrx2':read(Path(wrapper['remote2_worker'])),
        }
        timed=Counter(row['endpoint_id'] for row in controller['records'] if row['endpoint_id'] is not None)
        worker_audit={}
        for index,(endpoint,worker) in enumerate(workers.items()):
            cells_for_endpoint=sum(cell%3==index for cell in range(4))
            warmup=(protocol['warmup']+1)*cells_for_endpoint
            expected=warmup+timed[endpoint]
            records=worker.get('records')
            timed_rows=[] if records is None else records[warmup:]
            worker_audit[endpoint]={
              'completed':worker['completed_units'],'expected_completed':expected,
              'warmup_requests':warmup,'timed_requests':timed[endpoint],
              'count_matches':worker['completed_units']==expected and (
                  records is None or len(timed_rows)==timed[endpoint]
              ),
              'worker_path_ms':None if records is None else summarize(
                  [x['worker_path_ms'] for x in timed_rows]
              ),
            }
        source_checks={}
        mapping={
          '/softwall/four_cell_trace_baseline_controller.py':'scripts_for_node/softwall_same_gpu/four_cell_trace_baseline_controller.py',
          '/softwall/same_request_ipc_worker.py':'scripts_for_node/softwall_same_gpu/same_request_ipc_worker.py',
          '/softwall/multigpu_p2p_nrx_worker.py':'scripts_for_node/softwall_same_gpu/multigpu_p2p_nrx_worker.py',
          '/softwall/trace_qwen_worker.py':'scripts_for_node/softwall_same_gpu/trace_qwen_worker.py',
          '/softwall/multigpu_p2p_ipc_gate.py':'scripts_for_node/softwall_same_gpu/multigpu_p2p_ipc_gate.py',
          '/softwall_task1/isca_v2/cuda_ipc_channel.py':'scripts_for_node/task1/isca_v2/cuda_ipc_channel.py',
        }
        for key,relative in mapping.items():source_checks[key]=sha(ROOT/relative)==protocol['source_sha256'][key]
        artifacts={path:sha(Path(path))==expected for path,expected in wrapper['artifact_sha256'].items()}
        arms.append({
          'seed':seed,'wrapper':str(wrapper_path),'wrapper_sha256':sha(wrapper_path),
          'all_arm_gates_pass':wrapper['all_pass'],'source_checks':source_checks,'artifact_checks':artifacts,
          'endpoint_admissions':dict(timed),'worker_audit':worker_audit,'summary':wrapper['summary'],
        })
    gates={
      'two_independent_arms':len(arms)==2 and all(x['all_arm_gates_pass'] for x in arms),
      'sources_frozen':all(all(x['source_checks'].values()) for x in arms),
      'artifacts_unchanged':all(all(x['artifact_checks'].values()) for x in arms),
      'all_three_endpoints_used':all(all(x['endpoint_admissions'].get(f'nrx{i}',0)>0 for i in range(3)) for x in arms),
      'worker_counts_complete':all(all(v['count_matches'] for v in x['worker_audit'].values()) for x in arms),
    }
    result={
      'schema':'softwall-confirm116-three-endpoint-v1','job':a.job,'arms':arms,'gates':gates,'all_pass':all(gates.values()),
      'claim':'On one 4xA100 NVLink node, one local CUDA-IPC and two remote P2P NeuralRx endpoints shared the four-cell SoftWall recovery/Qwen/fault path in two independent arms without observed safety or credit violations.',
      'scope':'finite-sample three-endpoint lifecycle and integration evidence; no WCET, production timing, or throughput-superiority claim',
    }
    out=a.output or RESULTS/f'confirm116_three_endpoint_gates_job{a.job}.json'
    tmp=out.with_suffix(out.suffix+'.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(out)
    print(json.dumps(gates,indent=2))
    if not result['all_pass']:raise SystemExit('Confirm116 gate failed')
if __name__=='__main__':main()
