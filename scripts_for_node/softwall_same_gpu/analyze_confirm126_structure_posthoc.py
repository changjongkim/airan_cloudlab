#!/usr/bin/env python3
"""Post-hoc decomposition of C126 safety evidence and comparison validity."""
from __future__ import annotations
import hashlib,json
from collections import defaultdict
from pathlib import Path
ROOT=Path('/pscratch/sd/s/sgkim/kcj/airan_cloudlab')
RES=ROOT/'results/softwall_multigpu'
PROTOCOL=RES/'confirm126_certificate_global_vs_static_protocol.json'
FROZEN=RES/'confirm126_certificate_global_vs_static_result.json'

def read(p): return json.loads(Path(p).read_text())
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def decision_rows(result):
 out=[]
 for h in result['homes']:
  c=read(h['controller'])
  for row in c['records']:
   out.append((h['home'],row['index'],row['cell'],row['gate_skipped'],row['admitted'],row['endpoint_id'],row['forced_nrx_failure'],row['commit_kind']))
 return out

def main():
 p=read(PROTOCOL);frozen=read(FROZEN);arms=[];by=defaultdict(lambda:defaultdict(list));totals=defaultdict(int);maxima=defaultdict(float)
 for spec in p['arms']:
  path=RES/(spec['label']+'_result.json');r=read(path);rows=decision_rows(r);by[spec['seed']][spec['policy']].append((spec['label'],rows))
  arm={'label':spec['label'],'seed':spec['seed'],'policy':spec['policy'],'wrapper_pass':r['all_pass'],'tb':0,'radio_correct':0,'nrx_commits':0,'conv_commits':0,'atomic_exchanges':0,'ai_tokens':r['broker_summary']['timely_value_tokens']}
  for h in r['homes']:
   c=read(h['controller']);arm['tb']+=len(c['records']);arm['radio_correct']+=c['correct_cells'];arm['nrx_commits']+=c['nrx_commits'];arm['conv_commits']+=c['conv_commits'];arm['atomic_exchanges']+=c['joint_lease_retired_count']
   for key in ('deadline_misses','nrx_bound_violations','conv_bound_violations','conv_path_bound_violations','background_budget_violations','background_horizon_violations'):
    totals[key]+=c[key]
   maxima['nrx_response_ms']=max(maxima['nrx_response_ms'],c['nrx_response_ms']['max'])
   maxima['conv_host_ms']=max(maxima['conv_host_ms'],max(x['conventional_host_path_ms'] or 0 for x in c['records']))
   maxima['radio_commit_response_ms']=max(maxima['radio_commit_response_ms'],c['commit_response_ms']['max'])
  totals['tb']+=arm['tb'];totals['atomic_exchanges']+=arm['atomic_exchanges'];arms.append(arm)
 pairwise=[]
 for seed,policies in sorted(by.items()):
  for policy,items in sorted(policies.items()):
   left,right=items
   diffs=[]
   for a,b in zip(left[1],right[1]):
    if a[3:]!=b[3:]: diffs.append({'home':a[0],'index':a[1],'cell':a[2],'left':a[3:],'right':b[3:]})
   pairwise.append({'seed':seed,'policy':policy,'left':left[0],'right':right[0],'decision_difference_count':len(diffs),'differences':diffs})
 value={'schema':'softwall-confirm126-structure-posthoc-v1','status':'posthoc','protocol':str(PROTOCOL),'frozen_result':str(FROZEN),'frozen_structural_pass':frozen['structural_pass'],'safety_mechanism':{'all_arm_wrappers_pass':all(x['wrapper_pass'] for x in arms),'totals':dict(totals),'maxima':dict(maxima),'broker_all_drained_unique':all(x['outstanding_tokens']==0 and x['duplicate_commit_count']==0 for x in frozen['arms'])},'comparison_validity':{'strict_radio_decision_parity':frozen['structural_gates']['radio_decision_parity'],'within_policy_repeat_differences':pairwise,'interpretation':'All arms validate fail-closed substrate safety for the exact mode. Timing-dependent radio admission differences invalidate a strict same-radio throughput comparison, independently of the already-failed >=2% outcome gate.'},'arms':arms,'artifact_sha256':{str(PROTOCOL):digest(PROTOCOL),str(FROZEN):digest(FROZEN),**{str(RES/(x['label']+'_result.json')):digest(RES/(x['label']+'_result.json')) for x in arms}}}
 out=RES/'confirm126_structure_posthoc.json';tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(out)
 print(json.dumps({'safety_mechanism':value['safety_mechanism'],'comparison_validity':value['comparison_validity']},indent=2))
if __name__=='__main__':main()
