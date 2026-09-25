> **보관 사유 (2026-09-25):** 이 문서는 실험 이력으로 보존한다. 최신 스킴과 claim boundary는 `docs/current`의 원고·모델·production gate 문서가 권위 기준이며, 과거 GPU0 전용, MIG, CloudLab 또는 4-GPU pool 가정은 현재 논문의 근거가 아니다.

# Confirm137: broker control budget의 on-path 계측

**상태:** 2026-09-24, 6/6 arm PASS  
**물리 환경:** Slurm job `58830573`, `nid001177`, NVIDIA A100-SXM4-40GB 4개  
**목적:** V12가 admission에서 청구하는 broker RPC 5 ms×3을 실제 wall clock으로 계측

## 결론

C137은 새 A100 node의 6개 process/seed arm에서 모든 `prepare/commit/complete` RPC를
계측했다. 총 5,993개 호출의 최대 wall time은 2.593595 ms였고 선언한 per-RPC 5 ms를
넘은 호출은 0개였다. 같은 실행에서 7,680 radio TB, 목표 분기 904회, AI40 조건부 교환
28회가 발생했고 deadline·NRx·conventional·AI·horizon·credit 위반은 0이었다.

이 결과로 “5 ms를 산술에 넣었지만 실제 control path를 측정하지 않았다”는 C135/C136의
telemetry 공백은 C137 mode에서 닫혔다. 관측 최대를 WCET로 해석하지 않으며, RPC telemetry
추가 자체가 timing mode를 바꾸므로 C135/C136과 C137을 동일한 exact mode로 합치지 않는다.

## 사전 고정과 canary

첫 canary는 컨테이너 `unittest`에 파일명 `.py`를 넘겨 test loader가 모듈 속성으로
해석하면서 formal arm 전에 실패했다. 호출을 module name으로 고친 뒤 같은 두 unit test가
통과했다. Campaign protocol은 **성공한 canary 뒤, formal arm 전에** source hash와 여섯
seed pair를 고정했다. 따라서 첫 canary 실패는 물리 outcome도 아니고 protocol 사후 변경도
아니다. Allocation log에 두 시도를 모두 보존했다.

동결 조건은 다음과 같다.

```text
P/D                  = 180/155 ms
NRx/conv/AI bound    = 45/25/40 ms
broker RPC bound     = prepare/commit/complete 각각 5 ms
transaction budget  = 15 ms
AI completion guard = 2 ms
complete transaction= 57 ms
homes/cells          = 2 homes × 4 cells
endpoint             = home당 2, ring depth 1
```

## RPC 결과

| Operation | 호출 수 | 최대 wall time | 5 ms 초과 | fault |
|---|---:|---:|---:|---:|
| prepare | 5,745 | 2.593595 ms | 0 | 0 |
| commit | 124 | 0.388313 ms | 0 | 0 |
| complete | 124 | 1.046257 ms | 0 | 0 |
| **전체** | **5,993** | **2.593595 ms** | **0** | **0** |

Prepare 수가 큰 이유는 보이는 global request가 현재 certificate window에 들어가는지 매
decision에서 질의하기 때문이다. Commit과 complete는 실제 선택된 request에만 발생한다.
각 arm의 양 home 모두 prepare·commit·complete를 최소 한 번 이상 실행했다.

## 같은 실행의 component bound

| Component | 선언 bound | 표본 | 최대 | headroom | 초과 |
|---|---:|---:|---:|---:|---:|
| NeuralRx response | 45 ms | 3,295 | 25.735641 ms | 19.264359 ms | 0 |
| Conventional host path | 25 ms | 5,232 | 11.591275 ms | 13.408725 ms | 0 |
| Qwen context-128 execution | 40 ms | 124 | 30.656990 ms | 9.343010 ms | 0 |

각 숫자는 telemetry가 켜진 한 node·6 arm의 관측치다. Headroom을 근거로 선언 bound를
낮추지 않는다.

## Conditional-class 재현

| 항목 | C137 |
|---|---:|
| Arm | 6 |
| Radio TB | 7,680 |
| Home release | 1,920 |
| Atomic exchange | 52 |
| 정확한 목표 분기 | 904 |
| AI40 target exchange | 28 |
| 최소 physical guarded-horizon margin | 46.052573 ms |
| 선언 safety violation | 0 |

28/28 사건은 static 53 ms slack에서 완전한 57 ms transaction을 거절하고 conditional
58 ms window에서는 수락했다. C135/C136과 C137을 mechanism 수준에서 합치면 세 A100
node·14 arm·17,920 TB·목표 분기 2,100회·AI40 exchange 66회다. 다만 C137은 telemetry
instrumentation mode이므로 이 합계는 cross-mode reproduction이고 exact-mode reliability
표본 수가 아니다.

## 판정 범위

허용:

> In an RPC-instrumented third-node mode, all 5,993 observed broker calls,
> including prepare, commit, and complete on every arm, returned within the
> frozen 5 ms per-call budget while the conditional AI40 and radio-safety gates
> passed.

허용하지 않음:

- 2.593595 ms를 WCET로 사용
- 한 telemetry node에서 A100 population 또는 다른 GPU family로 일반화
- socket timeout이 GPU kernel preemption이나 MPS hard isolation을 제공한다고 주장
- synthetic D155를 production DU `d_MAC`으로 해석

## Artifact

- [Frozen campaign protocol](../../results/softwall_multigpu/confirm137_service_bound_telemetry_protocol.json)
- [Campaign result](../../results/softwall_multigpu/confirm137_service_bound_telemetry_result.json)
- [Allocation environment](../../results/softwall_multigpu/raw/confirm137_environment_job58830573.json)
- [Allocation log](../../results/softwall_multigpu/confirm137_allocation_job58830573.log)
- [Service-bound qualification v2](../../results/softwall_multigpu/softwall_service_bound_qualification_v2.json)
- [C137 artifact manifest](../../results/softwall_multigpu/confirm137_artifact_manifest.json)
- [Service-bound v2 manifest](../../results/softwall_multigpu/softwall_service_bound_v2_manifest.json)
- [Instrumented client](../../scripts_for_node/softwall_same_gpu/instrumented_deadline_global_trace_client.py)
- [Controller wrapper](../../scripts_for_node/softwall_same_gpu/global_trace_service_bound_controller.py)
- [Campaign analyzer](../../scripts_for_node/softwall_same_gpu/analyze_confirm137_service_bound_telemetry.py)
