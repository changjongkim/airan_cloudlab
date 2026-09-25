# C141: corrected V13 독립 node 재자격

**상태:** 2026-09-24, 8/8 arm PASS  
**결론:** C139/C140에서 수정한 `socket timeout 5 ms`, `RPC admission bound 7 ms`,
`AI35+control21+completion2=58 ms` 계약이 새 A100 node와 새 process lifecycle에서도
재현됐다.

## 목적

C139와 C140은 같은 node `nid001252`에서 실행됐다. 따라서 7 ms control bound와 AI35
conditional class가 하나의 node 상태에 함께 의존했을 가능성이 남아 있었다. C141은 이전
검증 node 여덟 개를 allocation에서 제외하고 두 요소를 같은 새 node에서 연속 재자격했다.
Protocol, seed, source hash, 7 ms 실패 기준은 GPU formal arm 전에 고정했다.

## 환경과 설계

- Slurm job: `58832462`
- Node: `nid001372`
- GPU: NVIDIA A100-SXM4-40GB 4개
- MIG OFF, MPS ON
- Control fault: prepare/commit/complete 각 두 arm
- Conditional class: context64 AI35 두 arm
- 실패 규칙: 첫 safety 또는 bound 실패에서 중단, 사후 bound 확대 금지
- Compute Shifter canary: 4 tests PASS

## Control fault 결과

| 항목 | C141 결과 |
|---|---:|
| Arm | 6/6 PASS |
| Radio TB | 16,320 |
| Fault 탐지 뒤 radio TB | 14,914 |
| 실제 시도 RPC | 2,010 |
| Socket timeout 뒤 반환한 fault call | 12 |
| Prepare 최대 | 5.343565 ms |
| Commit 최대 | 5.775660 ms |
| Complete 최대 | 5.434791 ms |
| 전체 최대 | 5.775660 ms |
| 7 ms 초과 | 0 |
| Deadline/bound/credit/duplicate 위반 | 0 |

5 ms socket timeout 뒤 controller로 복귀하는 추가 시간이 새 node에서도 재현됐다. 동시에
모든 faulting call은 사전 고정 7 ms admission bound 안에 들어왔다.

## AI35 conditional class 결과

| 항목 | C141 결과 |
|---|---:|
| Arm | 2/2 PASS |
| Radio TB | 2,560 |
| Target branch | 293 |
| Atomic exchange | 17 |
| AI35 target exchange | 7 |
| 최소 physical guarded-horizon margin | 44.160249 ms |
| Static/conditional model margin | −5 / 0 ms |
| Safety/bound/credit 위반 | 0 |

일곱 target event 모두 static all-fail slack 53 ms에는 들어가지 않았고, conditional
decision window 58 ms에는 정확히 들어갔다. 실제 margin이 큰 것은 45/35 ms 상한보다
물리 NRx/AI가 빨리 끝났기 때문이며 bound를 낮추는 근거로 사용하지 않는다.

## 두 node 결합 판정

C139/C140과 C141의 corrected exact mode를 합치면 다음과 같다.

| 증거 단위 | 두 node 결합 |
|---|---:|
| Physical node | 2 |
| Control fault arm | 12 |
| Control-path radio TB | 32,640 |
| Fault 탐지 뒤 radio TB | 29,833 |
| Attempted RPC | 4,030 |
| RPC 최대 | 5.775660 ms |
| 7 ms 초과 | 0 |
| AI35 arm | 4 |
| AI35 radio TB | 5,120 |
| Target branch | 602 |
| AI35 exchange | 15 |
| 전체 선언 safety 위반 | 0 |

현재 판정은 `FINITE_SAMPLE_TWO_NODE_CORRECTED_CONTROL_PASS`다. 두 node만으로 hardware
population reliability나 WCET를 주장하지 않는다. Node를 IID라고 가정한 zero-failure
95% 상한도 77.64%로 넓다.

## 주장 한계

- 같은 A100 family 안의 두 node 재자격이다.
- OS scheduler, socket stall과 GPU 실행의 deterministic upper bound는 아니다.
- 실제 DU `d_MAC`, cold/restart/GC ON, 다른 GPU family는 미검증이다.
- AI35 exchange는 mechanism 증거이며 처리량 우위가 아니다.

## 권위 artifact

- [Control protocol](../../results/softwall_multigpu/confirm141_control_protocol.json)
- [Control result](../../results/softwall_multigpu/confirm141_control_requalification_result.json)
- [AI35 protocol](../../results/softwall_multigpu/confirm141_ai35_protocol.json)
- [AI35 result](../../results/softwall_multigpu/confirm141_ai35_requalification_result.json)
- [Combined result](../../results/softwall_multigpu/confirm141_v13_independent_requalification_result.json)
- [Service-bound qualification v4](../../results/softwall_multigpu/softwall_service_bound_qualification_v4.json)
- [C141 artifact manifest](../../results/softwall_multigpu/confirm141_artifact_manifest.json)
- [V13 cross-node manifest](../../results/softwall_multigpu/softwall_v13_cross_node_manifest.json)
