# C162 predictive feasibility envelope 결과

**상태:** 2026-09-25, model grid·retrospective audit·두-node physical boundary·64-debt scheduler scalability 모두 PASS  
**주장 범위:** 자격화된 warm `P180/D155`, NRx45/recovery25/control5, MIG-off MPS,
4 accepted debt와 여섯 Qwen class의 유한 표본 예측  
**주장하지 않는 범위:** WCET, cold/long-idle, 임의 A100 node, production `d_MAC`,
8-receiver 실행 가능성, cross-GPU-family 일반화

![C162 predictive envelope and certified scheduler latency](figures/softwall_c162_envelope_scalability.svg)

## 1. 무엇을 검증했는가

C162의 질문은 “SoftWall이 한 번 안전하게 실행됐는가”가 아니다. 실행 전에 관측 가능한
mode와 state만으로 다음 네 상태를 구분할 수 있는지를 검증한다.

| 상태 | 뜻 |
|---|---|
| QSU | all-fail mandatory schedule이 안전하고 요청 AI class도 함께 완료 가능 |
| QSN | mandatory schedule은 안전하지만 해당 AI class를 넣을 certified slack이 없음 |
| MI | 초기 all-fail mandatory admission 자체가 불가능 |
| UQ | timing, lifecycle, node, memory 또는 fault provenance가 자격화되지 않음 |

한 recovery lane, 동일 deadline과 동일 service bound에서는 accepted debt `A`, 현재 unresolved
debt `U`, 판단시각 `t`, AI class bound `B_AI`에 대해 다음 식을 사용한다.

```text
initial all-fail finish = 45 + 25 A
current mandatory finish = max(t,45) + 25 U
AI-and-recovery finish = max(t,45) + 5 + B_AI + 25 U
recovery deadline = 155 - 2 = 153 ms
```

첫 식이 153 ms를 넘으면 MI다. 첫 식과 둘째 식이 안전하지만 셋째 식이 넘으면 QSN이고,
셋째 식까지 만족하면 QSU다. 빈 debt 집합은 판단시각이 늦어도 mandatory-safe이지만, AI
unit 자체는 153 ms 전에 끝나야 한다. C162 전수 대조 과정에서 이 두 edge case를 발견해
analytic model과 exact oracle 양쪽을 교정한 뒤 결과를 다시 생성했다.

## 2. 전체 상태공간 대조

[C162 grid](../../results/softwall_multigpu/c162_feasibility_grid_v1.json)는 다음 qualified
상태 16,023개를 열거한다.

- accepted debt `0..5`, unresolved debt `0..accepted`
- 판단시각 `45..153 ms`
- AI class `none, 16, 32, 64, 128, 256, 512`
- NRx45, recovery25, control5, `P180/D155`, capacity1

결과는 QSU 2,664, QSN 5,281, MI 8,078이며, analytic classifier와 bounded exhaustive exact
certificate의 불일치는 **0/16,023**이다.

별도 provenance grid 162개는 recovery bound `{12,25}`, lifecycle
`{warm,cold,long-idle}`, node 상태 `{qualified,failed,unknown}`, fault class
`{none,post-fence delay,pre-fence loss}`, receiver 수 `{4,6,8}`을 조합했다. 과거 false
qualification인 recovery12, cold/long-idle, 실패·미확인 node, receiver5--7은 UQ로,
receiver8의 관측 OOM은 MI로 분류했다. 자격 없는 provenance를 QSU/QSN으로 승격한 경우는
0개다.

## 3. 기존 물리 실행과의 retrospective audit

[C159-Q2 retrospective 결과](../../results/softwall_multigpu/c162_q2_retrospective_validation.json)는
동결된 두 독립 node의 1,200 physical epoch를 C162 model에 다시 넣었다.

| node | 예측 허가=실제 허가 | 예측 거절=실제 거절 | 불일치 |
|---|---:|---:|---:|
| `nid001204` | 544 | 56 | 0 |
| `nid001632` | 533 | 67 | 0 |
| 합계 | 1,077 | 123 | **0/1,200** |

이는 이미 실행된 동일 계약의 정합성 검사이므로 독립 C162 holdout으로 세지 않는다. 그 역할은
다음 물리 boundary campaign이 담당한다.

## 4. 사전 고정 물리 boundary

모든 round에서 4개의 TensorRT NeuralRx가 GPU3에서 실제 실행되고, 두 RAN home의 input과
응답은 CUDA IPC/P2P로 이동한다. +20 dB 안정 입력의 네 NRx가 NRx45 안에 성공해야만 round를
계속한다. 그 뒤 사전 고정한 suffix를 실패로 바꿔 unresolved debt 수를 통제한다. 실패로
주입된 요청은 GPU2의 shared cuPHY conventional path를 실제 실행하고, 허가된 AI는 같은
GPU2의 MPS client에서 Qwen2.5-1.5B를 실제 실행한다.

이 주입은 NeuralRx 정확도를 평가하기 위한 것이 아니다. 채널 성공률의 우연성을 제거하고
debt 수, AI class, 판단시각이라는 envelope 축을 물리 경로에서 검증하기 위한 controlled
fault다.

| 점 | 물리 조건 | 예측과 관측 |
|---|---|---|
| E1 | unresolved4, AI 없음 | QSN, 네 recovery 실행 |
| E2 | 다섯 번째 all-fail debt | MI, 매 round 원자적 reject |
| E3 | unresolved2, context128 | QSU, lease/Qwen 실행 |
| E4 | unresolved2, context256 | QSN, lease 거절 |
| E5 | unresolved1, context512 | QSU, lease/Qwen 실행 |
| E6a | unresolved1, context64, 보수적 판단 88 ms | QSU, lease/Qwen 실행 |
| E6b | 동일하되 판단 89 ms 이상 | QSN, lease 거절 |

Development는 `nid001265`, seed43M, 정방향 60 rounds였고 holdout은 개발 node를 제외한
`nid001353`, seed44M, 역방향 120 rounds였다. Frozen source hash는 두 campaign에서 같다.

| 합계 | 결과 |
|---|---:|
| physical rounds | 180 |
| actual NeuralRx | 720 |
| injected physical recoveries | 330 |
| physical Qwen units | 90 |
| single radio commits | 720 |
| deadline miss | **0** |
| 각 E1/E3/E4/E5/E6a/E6b 표본 | 30 |
| model/lease decision mismatch | **0/180** |

전체 최대값은 NRx release-to-complete 28.407 ms, recovery whole path 8.950 ms, Qwen host
execution 67.976 ms, radio commit 126.169 ms였다. 각각의 class bound와 NRx45/recovery25/D155
gate를 모두 통과했다. E6a는 30/30 모두 conservative integer time 88 ms였고 E6b는 30/30
모두 89 ms 이상이었다. [두-node 판정](../../results/softwall_multigpu/c162_boundary_two_node.json)은
10개 combined gate를 모두 통과했다.

## 5. 1--64 debt certified scheduler scalability

Exact search는 small-state reference로 유지하되, runtime 확장을 위해 polynomial candidate
scheduler와 독립 executable verifier를 분리했다. Candidate scheduler는 deadline, laxity,
release, short-job, long-job의 다섯 deterministic order로 list schedule을 만들고, verifier는
모든 obligation의 key, release/now, service, deadline, lane overlap과 full-GPU AI blackout을
독립 검사한다. Runtime은 verifier를 통과한 schedule만 사용할 수 있으므로 heuristic의
오류는 unsafe admission이 아니라 보수적 거절로 나타난다.

[Scalability 결과](../../results/softwall_multigpu/c162_scheduler_scalability_v1.json)는 다음과 같다.

| 감사 | 결과 |
|---|---:|
| 임의 이질 small state와 exact 대조 | 600 |
| exact accept / certified accept | 504 / 494 |
| false-safe | **0** |
| false-conservative | 10 |
| 현재 qualified mode 판정 차이 | **0/16,023** |
| debt1--64, capacity1/2/4/8 large grid | 2,800 states |
| accepted / rejected | 2,161 / 639 |
| 반환된 certificate verifier 실패 | **0** |
| debt64 decision latency p99 / max | 1.494 / 2.197 ms |
| debt64 verifier latency p99 / max | 0.131 / 0.142 ms |

따라서 현재 mode에서는 exact와 같은 결정을 내리면서 64 debt까지 5 ms control budget 안에서
certificate를 만들고 검증했다. 일반 이질 상태에서는 exact-feasible 504개 중 10개를
보수적으로 거절하므로 optimal scheduler라고 주장하지 않는다. 이 latency는 한 CPU 실행의
finite-run 측정이며 production WCET가 아니다.

## 6. 해석

C162가 추가한 핵심은 안전한 예시 하나가 아니라 **예측 가능한 경계**다. 같은 radio
admission과 같은 MPS cap에서도 unresolved debt가 2개이면 context128은 허가되고 context256은
거절되며, debt가 1개로 줄면 context512가 허가된다. 같은 context64와 debt1에서도 판단시각
88/89 ms 사이에서 결정이 바뀐다. 이 선택은 future CRC나 실제 kernel duration을 보지 않고,
동결한 whole-path bound와 현재 debt state만 사용한다.

MI와 UQ도 구분한다. 다섯 번째 debt는 알려진 bound 안에서도 all-fail schedule이 없으므로
MI다. 반면 `nid001044`에서 같은 최종 C161 source가 NRx45를 47.470 ms로 깬 사례는 모델
산술 실패가 아니라 node/lifecycle qualification 실패이므로 UQ다. Recovery12, cold/long-idle,
receiver5--7도 같은 이유로 UQ이며, receiver8 OOM은 관측된 resource infeasibility다.

따라서 현재 방어 가능한 기여 문장은 다음과 같다.

> SoftWall은 optional NeuralRx가 만드는 조건부 conventional recovery debt를 executable
> all-fail certificate로 유지하고, bounded external-AI lease와 recovery retiming을 원자적으로
> 처리한다. 자격화된 warm P180/D155 mode에서는 debt 수, AI class와 실제 decision time으로
> QSU/QSN/MI 경계를 사전에 계산할 수 있으며, exact state 16,023개와 두 독립 A100 node의
> 180 physical boundary rounds에서 false-safe를 관측하지 않았다.

이 문장은 확률 0 또는 hard real-time WCET를 뜻하지 않는다. 현재 증거는 명시된 node
preflight와 warm lifecycle을 통과한 두 A100 node의 유한 표본이다. 실제 DU/FAPI expiry,
cold/restart/long-idle, durable process restart, 다른 GPU family는 별도 자격이 필요하다.

## 7. 권위 artifact

- [Model grid](../../results/softwall_multigpu/c162_feasibility_grid_v1.json)
- [C159-Q2 retrospective validation](../../results/softwall_multigpu/c162_q2_retrospective_validation.json)
- [Development result](../../results/softwall_multigpu/c162a_boundary_dev_j58860486_result.json)
- [Holdout result](../../results/softwall_multigpu/c162b_boundary_holdout_j58860535_result.json)
- [Two-node combined result](../../results/softwall_multigpu/c162_boundary_two_node.json)
- [Certified scheduler scalability](../../results/softwall_multigpu/c162_scheduler_scalability_v1.json)
- [Paper figure PDF](figures/softwall_c162_envelope_scalability.pdf)
- [83-file artifact manifest](../../results/softwall_multigpu/c162_artifact_manifest.json)
