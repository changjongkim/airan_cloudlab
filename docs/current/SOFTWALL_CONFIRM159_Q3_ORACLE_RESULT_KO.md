# C159-Q3 calibration oracle opportunity screen

**판정:** `C159_Q3_ORACLE_SCREEN_STOP_PERFORMANCE`  
**Confirmatory performance holdout:** 열지 않음  
**사전 MDE:** strong safe baseline 대비 timely token value 5%  
**Prespec:** `results/softwall_multigpu/confirm159_q3_oracle_prespec_v1.json`  
**결과:** `results/softwall_multigpu/confirm159_q3_oracle_screen_v1.json`  
**Manifest:** `results/softwall_multigpu/confirm159_q3_oracle_manifest.json`

## 질문

C159-Q2는 여섯 Qwen class와 actual NeuralRx/shared recovery 경로를 물리적으로 자격화했다.
다음 질문은 그 안전한 실행 순서가 강한 safe baseline보다 실질적인 external-AI value를 만들
가능성이 있는가였다. 이 가능성이 5%에도 못 미치면 새 GPU holdout을 반복하는 것은 성능
주장을 만들지 못하고 표본만 늘리게 된다.

Q3는 confirmatory trace를 열기 전에 calibration partition만 사용해 이 기회를 판정했다.

## 공정한 비교

두 시스템에 다음을 똑같이 줬다.

- 같은 BurstGPT request arrival, 1초 synthetic SLO, context class와 token value
- 같은 Q2 actual NeuralRx outcome state와 class별 frozen bound
- 같은 radio all-fail admission inequality
- 미래 arrival와 deadline을 모두 아는 같은 exact offline weighted request selector
- P180 epoch당 최대 Qwen unit 한 개

유일한 차이는 실행 순서다.

| 시스템 | 순서 |
|---|---|
| SoftWall | Qwen lease를 먼저 원자 commit하고 unresolved recovery를 deadline 안으로 retime |
| Strong Event-driven | unresolved recovery를 물리 완료한 뒤 현재 slack에 Qwen 실행 |

Event-driven에 약한 online EDF를 주고 SoftWall에 oracle을 주는 비교는 하지 않았다. 그렇게
생긴 차이는 일반 queue scheduling 이득이며 conditional-recovery substrate의 이득이 아니기
때문이다.

## 입력과 계산

Calibration 입력은 서로 겹치지 않는 60초 창 네 개, request 4,290건, offered value
1,129,504 token이다. Q2 development와 independent-node holdout의 600-epoch state stream을
각각 처음/마지막 334 epoch로 고정 매핑했다. 한 node 안에서 68 state가 겹치므로 네 window를
독립 통계 표본처럼 사용하지 않는다. 이 실행은 confidence interval이 아니라 사전 opportunity
screen이다.

각 request와 radio epoch 사이에 다음 조건을 만족할 때만 bipartite edge를 만들고, min-cost
flow로 전체 timely token value를 정확히 최대화했다.

```text
request_arrival <= epoch_decision
policy_specific_completion <= request_deadline
class_bound + control + unresolved_recovery_bound <= radio recovery deadline
```

Matcher는 작은 전수 조합과 비교해 최대 weight가 일치함을 별도 확인했다.

## 결과

| 모델 | Event-driven | SoftWall | SoftWall 개선 |
|---|---:|---:|---:|
| Q2에서 관측한 full recovery path 시간 | 931 request / 385,262 token | 931 / 385,262 | **0.000%** |
| unresolved recovery마다 25 ms 전체 bound 청구 | 930 / 385,007 | 931 / 385,262 | **0.066%** |

Empirical 비교는 네 window 모두 0.000%였다. 보수적 contract sensitivity에서조차 이득은
token 255개, 0.066%로 5% MDE의 약 1/76이다.

## 왜 0이 나왔는가

SoftWall은 recovery보다 먼저 AI를 시작할 수 있다. 그러나 이 mode에서 실제 recovery full
path는 대부분 약 3--4 ms이고 최대도 Q2 결합 표본에서 13.465 ms였다. AI SLO는 1,000 ms다.
Recovery-first 기준선이 recovery를 실제 완료한 직후 같은 요청을 실행해도 deadline eligibility가
바뀌지 않았다. Offline selector는 두 시스템에 같은 931개 요청을 배정했다.

즉 조건부 capacity와 실행 순서 차이는 실제로 존재하지만, 현재 trace의 1초 SLO에서는 그
차이가 timely token value로 변환되지 않는다. 모든 recovery가 선언 25 ms를 소비한다고
과장해도 단 한 요청만 달라졌다.

## 연구 판정

현재 mode에서 “SoftWall이 strong recovery-first baseline보다 external-AI throughput을 5%
늘린다”는 가설은 진행하지 않는다. Confirmatory holdout을 materialize하지 않고 online ABBA도
실행하지 않는다. 이 판단은 처리량 우위를 만들기 위해 SLO나 baseline을 사후 약화하는 것을
막는다.

유지되는 기여는 다음이다.

1. Multi-home optional NeuralRx의 conditional mandatory debt와 local-safe/global-unsafe 반례
2. Global all-fail executable certificate
3. Recovery calendar와 external-AI lease의 generation-safe atomic transaction
4. Common-cutoff outcome batch, bounded revalidation과 physical latest-start lifecycle
5. 여섯 Qwen class의 실제 cuPHY/NeuralRx co-run feasibility boundary

다음 실험은 C160/C161 integrated fault containment와 C162 predicted-versus-measured envelope다.
더 짧은 AI SLO가 실제 운영 요구에서 독립적으로 정당화될 때만 SLO sensitivity를 별도
envelope 축으로 평가하며, 현재 1초-SLO 성능 주장을 대체하는 사후 mode로 쓰지 않는다.
