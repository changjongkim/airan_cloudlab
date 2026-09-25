# C142--C144: V14 pipelined control과 AI45 결과

**상태:** 2026-09-24, 당시 모든 사전 고정 gate PASS; 후속 ownership 감사로 V14 mode는 현재 UQ  
**결론:** `prepare/complete`를 RAN critical path 밖으로 옮기고 launch `commit`만 동기
확인하면, V13의 AI35 대신 exchange-only AI45를 두 A100 node에서 실행하면서 세 broker
failure point 뒤에도 local RAN certificate를 계속 실행할 수 있다.

후속 two-request 회귀가 retained staged token 뒤 두 번째 prepare의 untracked ownership
분기를 재현했다. 이 문서의 물리 timing/fault 결과는 보존하지만 현재 자격 근거는
[V14→V15 교정](SOFTWALL_V14_OWNERSHIP_CORRECTION_KO.md)과
[C145/C146 V15 재자격](SOFTWALL_CONFIRM145_146_V15_SINGLE_TOKEN_RESULT_KO.md)이다.

## C142: AI45 conditional class

Node `nid001361`, 두 독립 seed, 두 home·8셀 구성에서 다음 결과를 얻었다.

| 항목 | 결과 |
|---|---:|
| Arm | 2/2 PASS |
| Radio TB | 2,560 |
| Target branch | 305 |
| AI45 conditional exchange | 9 |
| 최소 physical guarded-horizon margin | 53.114836 ms |
| Commit 최대 | 0.612807 ms |
| Deferred RPC 최대 | 9.780886 ms |
| Deadline/bound/credit/duplicate 위반 | 0 |

모델상 static margin은 `53−54=−1 ms`, conditional margin은 `58−54=+4 ms`다. 모든 target
event는 즉시 complete ACK로 표기하지 않았고, timed traffic 뒤 broker drain에서 최종
`completed` 상태를 확인했다.

첫 canary는 staged token을 짧은 pre-radio horizon에서 즉시 abort해 AI 0이 됐다. 실패를
보존하고 request 자체 SLO가 남아 있으면 token을 유지해 다음 local window에서 재검사하도록
수정한 뒤 정식 protocol을 고정했다.

## C143: 세 control point fault matrix

같은 node에서 broker가 home0의 state transition을 적용한 직후 reply 전에 종료되도록
`prepare`, `commit`, `complete` 각각 두 arm을 실행했다.

| 항목 | 결과 |
|---|---:|
| Fault arm | 6/6 PASS |
| Radio TB | 16,320 |
| Fault 탐지 뒤 radio TB | 14,184 |
| 전체 RPC record | 1,924 |
| Synchronous commit record | 340 |
| Commit 최대 / 7 ms 초과 | 5.148988 ms / 0 |
| Deferred record | 1,584 |
| Deferred 최대 / 7 ms 초과 | 9.614963 ms / 50 |
| Deadline/bound/credit/duplicate 위반 | 0 |

Commit-after-apply reply loss에서는 target Qwen 실행이 0이었다. Complete-after-apply reply
loss에서는 target Qwen이 정확히 한 번 실행됐고 token은 재사용되지 않았다. Prepare ambiguity는
물리 launch 없이 격리됐다. 모든 arm에서 두 home 모두 fault 뒤 radio를 계속 처리했다.

## C144: 독립 node 재자격

이전 node를 제외하고 `nid002049`에서 AI45 한 arm과 세 fault point를 각각 한 arm 실행했다.

| 항목 | 결과 |
|---|---:|
| 전체 radio TB | 6,080 |
| AI45 exchange | 2 |
| Fault arm | 3/3 PASS |
| Fault 탐지 뒤 radio TB | 3,730 |
| Source hash mismatch | 0 |
| Safety/duplicate 위반 | 0 |

따라서 V14의 현재 판정은 **two-node finite-sample same-family PASS**다.

## V14 envelope

Envelope V14는 기존 V13 synchronous mode를 보존하고 pipelined mode를 별도 scenario로
추가한다.

```text
AI35: 35 + 7 + 2 = 44 ms  -> static 가능
AI40: 40 + 7 + 2 = 49 ms  -> static 가능
AI45: 45 + 7 + 2 = 54 ms  -> static 53 ms 불가, conditional 58 ms 가능
```

분류는 `QSU6/QSN0/MI3/UQ11`이고 validation gate 10개가 모두 통과했다. 유한 protocol
model은 16 states·20 transitions에서 before/after-apply failure를 전수 열거해 invariant
violation 0을 냈다.

## 주장 한계

- 전체 결과는 synthetic P180/D155 timing mode다.
- 7 ms는 commit의 finite-sample admission bound이며 WCET가 아니다.
- Deferred operation에는 7 ms bound를 주장하지 않는다.
- Broker crash 뒤 token reconciliation/restart는 구현하지 않았다.
- 처리량 최적화 우위나 production HARQ deadline 보장이 아니다.

## 권위 artifact

- [C142 protocol](../../results/softwall_multigpu/confirm142_v14_pipelined_protocol.json)
- [C142 result](../../results/softwall_multigpu/confirm142_v14_pipelined_result.json)
- [C143 protocol](../../results/softwall_multigpu/confirm143_v14_control_faults_protocol.json)
- [C143 result](../../results/softwall_multigpu/confirm143_v14_control_faults_result.json)
- [C144 protocol](../../results/softwall_multigpu/confirm144_v14_independent_node_protocol.json)
- [C144 result](../../results/softwall_multigpu/confirm144_v14_independent_node_result.json)
- [V14 grid](../../results/softwall_multigpu/softwall_sharded_home_envelope_grid_v14.json)
- [V14 prediction](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v14.json)
- [V14 validation](../../results/softwall_multigpu/softwall_envelope_v14_validation_summary.json)
- [V14 immutable manifest](../../results/softwall_multigpu/softwall_v14_pipelined_control_manifest.json)
