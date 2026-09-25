#!/usr/bin/env python3
"""Audit prospectively frozen component bounds for three-endpoint SoftWall."""

from __future__ import annotations
import hashlib,json
from collections import Counter
from pathlib import Path

ROOT=Path('/pscratch/sd/s/sgkim/kcj/airan_cloudlab');RES=ROOT/'results/softwall_multigpu';PROTOCOL=RES/'confirm117_component_bounds_protocol.json'
def read(p:Path)->dict:return json.loads(p.read_text())
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def summary(v:list[float])->dict[str,float|int]:
    s=sorted(v)
    if not s:return {'count':0,'mean':0.0,'p50':0.0,'p99':0.0,'max':0.0}
    return {'count':len(s),'mean':sum(s)/len(s),'p50':s[round((len(s)-1)*.5)],'p99':s[round((len(s)-1)*.99)],'max':s[-1]}
def gate(values:list[float],bound:float)->dict:
    return {'bound':bound,'summary':summary(values),'violations':sum(x>bound for x in values),'pass':all(x<=bound for x in values)}
def main()->None:
    p=read(PROTOCOL);b=p['candidate_bounds'];source={k:{'expected':v,'observed':sha(ROOT/k),'match':v==sha(ROOT/k)} for k,v in p['source_sha256_before_run'].items()}
    arms=[]
    for spec in p['arms']:
        label=spec['label'];wrap_path=RES/f'{label}_result.json';w=read(wrap_path);c=read(Path(w['controller']));arm_protocol=read(RES/f'{label}_protocol.json')
        rows=[x for x in c['records'] if x['endpoint_id'] is not None]
        admissions=Counter(x['endpoint_id'] for x in rows)
        component={
          'front_gpu_ms':gate([x['nrx_prepare_gpu_ms'] for x in rows],b['front_gpu_ms']),
          'back_gpu_ms':gate([x['nrx_back_gpu_ms'] for x in rows],b['back_gpu_ms']),
          'end_to_end_nrx_ms':gate([x['nrx_response_ms'] for x in rows],b['end_to_end_nrx_ms']),
        }
        remote={}
        for endpoint,key in (('nrx1','remote1_worker'),('nrx2','remote2_worker')):
            worker=read(Path(w[key]));idx=int(endpoint[-1]);cells=sum(cell%3==idx for cell in range(4));warm=(p['warmup']+1)*cells;rr=worker['records'][warm:]
            remote[endpoint]={
              'expected_timed':admissions[endpoint],'observed_timed':len(rr),
              'count_match':len(rr)==admissions[endpoint],
              'forward_gpu_us':gate([x['forward_gpu_us'] for x in rr],b['remote_forward_gpu_us']),
              'nrx_gpu_ms':gate([x['nrx_gpu_ms'] for x in rr],b['remote_nrx_gpu_ms']),
              'backward_gpu_us':gate([x['backward_gpu_us'] for x in rr],b['remote_backward_gpu_us']),
              'worker_path_ms':gate([x['worker_path_ms'] for x in rr],b['remote_worker_path_ms']),
            }
        artifacts={path:sha(Path(path))==expected for path,expected in w['artifact_sha256'].items()}
        component_pass=all(x['pass'] for x in component.values())
        remote_pass=all(v['count_match'] and all(v[k]['pass'] for k in ('forward_gpu_us','nrx_gpu_ms','backward_gpu_us','worker_path_ms')) for v in remote.values())
        arms.append({'label':label,'wrapper':str(wrap_path),'wrapper_sha256':sha(wrap_path),'all_arm_gates_pass':w['all_pass'],'arm_protocol_source_checks':{k:arm_protocol['source_sha256'][k]==sha(ROOT/{'/softwall/four_cell_trace_baseline_controller.py':'scripts_for_node/softwall_same_gpu/four_cell_trace_baseline_controller.py','/softwall/same_request_ipc_worker.py':'scripts_for_node/softwall_same_gpu/same_request_ipc_worker.py','/softwall/multigpu_p2p_nrx_worker.py':'scripts_for_node/softwall_same_gpu/multigpu_p2p_nrx_worker.py','/softwall/trace_qwen_worker.py':'scripts_for_node/softwall_same_gpu/trace_qwen_worker.py','/softwall/multigpu_p2p_ipc_gate.py':'scripts_for_node/softwall_same_gpu/multigpu_p2p_ipc_gate.py','/softwall_task1/isca_v2/cuda_ipc_channel.py':'scripts_for_node/task1/isca_v2/cuda_ipc_channel.py'}[k]) for k in arm_protocol['source_sha256']},'artifacts_unchanged':all(artifacts.values()),'endpoint_admissions':dict(admissions),'component':component,'remote':remote,'component_pass':component_pass and remote_pass,'summary':w['summary']})
    gates={'protocol_sources':all(x['match'] for x in source.values()),'two_arms':len(arms)==2,'arm_sources':all(all(x['arm_protocol_source_checks'].values()) for x in arms),'artifacts_unchanged':all(x['artifacts_unchanged'] for x in arms),'system_safety':all(x['all_arm_gates_pass'] for x in arms),'component_bounds':all(x['component_pass'] for x in arms)}
    result={'schema':'softwall-confirm117-component-bounds-v1','protocol':str(PROTOCOL),'protocol_sha256':sha(PROTOCOL),'source_audit':source,'arms':arms,'gates':gates,'all_pass':all(gates.values()),'scope':'finite-sample component qualification for the warm local1+remote2 MPS/Qwen/fault mode; not WCET or production timing'}
    out=RES/'confirm117_component_bounds_job58815435.json';tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(out);print(json.dumps(gates,indent=2))
    if not result['all_pass']:raise SystemExit('Confirm117 component gate failed')
if __name__=='__main__':main()
