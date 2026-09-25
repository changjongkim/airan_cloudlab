> **보관 사유 (2026-09-25):** 이 문서는 실험 이력으로 보존한다. 최신 스킴과 claim boundary는 `docs/current`의 원고·모델·production gate 문서가 권위 기준이며, 과거 GPU0 전용, MIG, CloudLab 또는 4-GPU pool 가정은 현재 논문의 근거가 아니다.

# SoftWall C134 독립 allocation/node 재자격 결과

**상태:** 2026-09-24, 6/6 prospective arm PASS  
**새 환경:** job 58823497, node nid002688, 4×NVIDIA A100-SXM4-40GB  
**기존 환경:** C132/C133 job 58819952, node nid001049  
**Mode:** MIG OFF, MPS ON, 2 recovery home×4 cell, home별 NeuralRx endpoint 2개와 Qwen worker 1개

## 결론

C132/C133의 full-control-budget broker fail-stop 결과는 같은 node의 우연에 머물지 않았다.
독립 allocation과 다른 A100 node에서 prepare, commit, complete 적용 직후 reply 전
broker 종료를 각각 두 새 seed로 반복했고 6개 arm이 모두 통과했다.

각 broker RPC 상한은 5 ms, 전체 control budget은 15 ms, physical guard는 2 ms다.
이 17 ms를 AI admission 전에 recovery horizon에서 차감했다. Broker fault 뒤에는 모든
global-AI client가 fail-closed로 전환했지만 두 home의 local RAN certificate 실행은
끝까지 계속됐다.

## 결과

| Fault point | Arm | RAN TB | Crash 탐지 뒤 TB | Containment 전 AI unit | NRx response max | Conv host-path max | AI execution max |
|---|---:|---:|---:|---:|---:|---:|---:|
| prepare | 2 | 5,440 | 4,980 | 113 | 11.573 ms | 6.499 ms | 62.415 ms |
| commit | 2 | 5,440 | 4,970 | 120 | 17.042 ms | 13.413 ms | 62.997 ms |
| complete | 2 | 5,440 | 4,976 | 121 | 22.301 ms | 8.016 ms | 58.136 ms |
| **합계** | **6** | **16,320** | **14,926** | **354** | — | — | — |

다음 항목은 모두 0이었다.

- RAN deadline miss
- NeuralRx 45 ms bound 위반
- conventional 25 ms bound 위반
- context별 AI bound 위반
- recovery horizon 위반
- duplicate physical AI execution
- 종료 뒤 남은 recovery 또는 joint lease credit

Prepare ambiguity는 physical AI를 시작하지 않았고, commit ambiguity도 ACK가 없으므로
시작하지 않았다. Complete ambiguity의 대상은 이미 물리 완료된 요청으로 정확히 한 번만
실행됐다.

## C133보다 강해진 점

C133은 한 node에서 control-point coverage를 완성했다. C134는 runtime source와 contract를
그대로 두고 allocation, node와 여섯 seed를 바꿨다. 따라서 현재 근거는 두 독립 A100
allocation/node에서 각 control point 네 arm, 합계 12 arm이다. C132/C133과 C134를 합치면
총 32,640 RAN TB이며 모든 선언 safety gate 위반은 0이다.

이는 cross-family 일반화가 아니다. 두 node 모두 A100이고 같은 Aerial, CUDA, MPS와
Qwen stack을 썼다. Production d_MAC, WCET, network partition, broker restart, durable
token reconciliation, GPU hang과 exactly-once는 여전히 범위 밖이다.

## 분석기 환경 기록

여섯 physical arm은 모두 결과 파일을 쓴 뒤 끝났다. 이어진 frozen aggregate analyzer는
compute host Python 3.6이 future annotations 문법을 지원하지 않아 한 번 실패했다.
원 로그와 실패 기록을 보존하고 analyzer 소스를 수정하지 않은 채 Aerial 25.3 container의
Python 3.10에서 같은 artifact를 분석해 PASS aggregate를 만들었다. Physical arm 결과와
protocol/source hash에는 영향이 없다.

## 권위 artifact

- [C134 frozen protocol](../../results/softwall_multigpu/confirm134_independent_node_control_matrix_protocol.json)
- [C134 aggregate result](../../results/softwall_multigpu/confirm134_independent_node_control_matrix_result.json)
- [C134 artifact manifest](../../results/softwall_multigpu/confirm134_artifact_manifest.json)
- [C134 validation summary](../../results/softwall_multigpu/confirm134_validation_summary.json)
- [Analyzer environment failure record](../../results/softwall_multigpu/confirm134_analyzer_environment_failure.json)
- [Envelope v8 prediction](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v8.json)
- [Envelope v8 manifest](../../results/softwall_multigpu/softwall_sharded_home_envelope_manifest_v8.json)
