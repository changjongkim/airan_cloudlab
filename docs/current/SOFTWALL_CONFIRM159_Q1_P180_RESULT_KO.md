# C159-Q1 P180 실제 NeuralRx 재자격 결과

**최종 판정:** `C159_Q1_TWO_NODE_P180_PASS`  
**개발:** job `58857317`, node `nid001177`, label `confirm159q1a6_dev_j58857317`  
**독립 holdout:** job `58857402`, node `nid001308`, label `confirm159q1b_holdout_j58857402`  
**결합 결과:** `results/softwall_multigpu/confirm159_q1_p180_two_node.json`  
**135-file manifest:** `results/softwall_multigpu/confirm159_q1_p180_manifest.json`

## 1. 이 실험이 해결한 공백

C158은 실제 TensorRT NeuralRx outcome, global all-fail certificate, atomic Qwen lease와
shared cuPHY recovery를 두 node에서 반복 검증했지만 주기가 600 ms였다. 따라서 D155 radio
contract는 검증했어도 60초 BurstGPT replay에 사용할 180 ms cadence는 아직 자격화되지
않았다.

C159-Q1은 expensive synthetic channel 생성과 validation-only local oracle를 readiness 전에
모두 만들고, 각 epoch에는 이미 만든 GPU bank를 stable CUDA-IPC buffer로 복사한다. 이전
epoch의 D155 뒤 다음 P180 release까지 남는 25 ms만 이 복사에 쓴다. 그 결과 timed 구간은
다음 물리 경로만 포함한다.

```text
P180 release
  -> GPU0/1의 stable input publish
  -> GPU3 persistent TensorRT NeuralRx
  -> actual CRC outcome publish
  -> V17.1 global certificate/replan
  -> GPU2 MPS: context-64 Qwen lease + 필요한 shared cuPHY recovery
  -> P2P response fence
  -> GPU0/1 single radio commit by D155
```

따라서 Q1의 질문은 “C158의 기계적 전이를 P180에서도 같은 bound로 실행할 수 있는가”다.
Trace throughput 우위나 variable-context workload는 아직 Q1의 평가 대상이 아니다.

## 2. 동결한 mode

| 항목 | 값 |
|---|---:|
| 주기 / radio expiry | P180 / D155 ms |
| NeuralRx 전체 release-to-complete bound | 45 ms |
| shared conventional 전체 path bound | 25 ms |
| Qwen context / bound | 64 token / 35 ms |
| control bound / recovery guard | 5 / 2 ms |
| GPU0 / GPU1 | 두 RAN home, stable IPC input/output owner |
| GPU2 | shared cuPHY cap80 + Qwen cap20, MPS |
| GPU3 | persistent actual NeuralRx cap80 |
| node당 표본 | 250 epoch, actual NeuralRx 1,000건 |

두 `+20 dB` key와 두 `-8.75 dB` transition key를 매 epoch 실제 NeuralRx에 제출한다. 제때
CRC-correct인 결과만 recovery debt를 해제하며 나머지는 shared conventional recovery를
실행한다. 다섯 번째 요청은 global capacity certificate로 거절한다.

## 3. 두 독립 node 결과

| 지표 | 개발 `nid001177` | holdout `nid001308` | 합계/최대 |
|---|---:|---:|---:|
| epoch | 250 | 250 | 500 |
| actual NeuralRx | 1,000 | 1,000 | 2,000 |
| 제때 성공 | 702 | 716 | 1,418 |
| physical recovery | 298 | 284 | 582 |
| Qwen lease | 250 | 250 | 500 |
| radio commit | 1,000 | 1,000 | 2,000 |
| NRx 최대 | 23.882 ms | 23.091 ms | 23.882 ms |
| recovery 최대 | 5.600 ms | 11.555 ms | 11.555 ms |
| Qwen 최대 | 26.771 ms | 25.704 ms | 26.771 ms |
| radio commit 최대 | 99.509 ms | 103.402 ms | 103.402 ms |
| deadline miss | 0 | 0 | 0 |

결합 판정의 9개 gate가 모두 통과했다.

- 두 개별 실행의 13개 gate가 전부 통과했다.
- job, node, seed가 서로 다르고 개발 node는 holdout exclusion에 사전 포함됐다.
- 두 protocol의 22개 source hash가 서로 같고 현재 source와도 같다.
- P180/D155와 accepted workload가 두 실행에서 같다.
- input round-trip, response echo, recovery semantic contract, deadline 위반은 모두 0이다.
- cold preflight 8개가 모두 완료된 뒤에만 timed release를 시작했다.

## 4. cold lifecycle 반례와 교정

첫 250-epoch 개발 실행 `confirm159q1a4_j58857203`은 13개 중 recovery gate 하나를
실패했다. 286 recovery 중 첫 recovery 한 건이 28.731 ms로 25 ms bound를 넘었다. 해당
건의 cuPHY conventional host 시간은 7.040 ms였으므로 계산 kernel 자체의 28 ms tail이
아니었다. 당시 warmup은 receiver만 실행했고 P2P input, complex-window 설치, CRC
normalization, P2P response를 합친 전체 경로는 처음 timed request에서 초기화됐다.

이를 사후 bound 완화로 처리하지 않았다. 각 recovery context에서 전체
`P2P input -> window install -> cuPHY -> response normalization -> P2P response/echo`를
readiness 전에 한 번 실행하고 그 cold 비용도 기록하도록 lifecycle을 새 source hash로
고쳤다. 개발/holdout의 cold preflight 최대는 각각 59.082/64.750 ms였고, 이후 warm timed
recovery 582건의 p99/max는 4.317/11.555 ms였다.

이 결과는 두 사실을 함께 보여 준다.

1. warm persistent P180 mode에서는 선언한 25 ms recovery bound가 두 node의 유한 표본에서
   유지됐다.
2. cold/restart/long-idle은 같은 mode로 간주하면 안 된다. 59--65 ms initialization을
   deadline 밖 maintenance 단계에서 끝내거나 별도 unqualified state로 차단해야 한다.

## 5. 실패 이력의 처리

| 단계 | 판정 | 처리 |
|---|---|---|
| attempt1 | peer schema 불일치, timed round 0 | prelaunch failure 보존 |
| attempt2 | 10-epoch pilot 통과 | 이후 socket source 변경으로 개발 자료만 사용 |
| attempt3 | AF_UNIX path 길이 초과, timed round 0 | failure 보존, 짧은 job socket으로 수정 |
| attempt4 | 250 epoch, first full-path cold tail 1건 | primary FAIL 보존 |
| attempt5 | full-path preflight 20-epoch pilot 통과 | 개발 검증, 최종 표본 제외 |
| attempt6 | 250-epoch development 통과 | 권위 개발 arm |
| holdout | 새 job/node/seed, 동일 source 통과 | 권위 holdout arm |

## 6. 논문 주장에 추가되는 것과 남은 것

C159-Q1은 C158의 P600 staging 제한을 제거하고, **실제 cuPHY/NeuralRx/Qwen을 사용한
conditional-recovery substrate가 P180 cadence에서도 두 A100 node에서 안전 gate를
통과했다**는 근거를 추가한다. 특히 whole-path preflight를 mode contract에 포함해야 한다는
lifecycle boundary도 실측했다.

아직 다음 주장은 할 수 없다.

- BurstGPT 도착과 16--512 token Qwen request에서 SoftWall이 strong baseline보다 더 많은
  deadline 내 token value를 만든다는 주장
- cold/restart/long-idle, worker crash, lost fence를 포함한 P180 fault 보장
- 유한 표본 최대값을 WCET로 해석하는 주장
- synthetic D155를 production DU의 `d_MAC`으로 해석하는 주장

## 7. 다음 실험 순서

1. **C159-Q2 variable-context qualification:** 16/32/64/128/256/512 token 각각의 isolated 및
   cuPHY co-run 전체 path bound를 두 node에서 자격화한다. 512-token 요청이 D155 내부에
   구조적으로 들어갈 수 없는 경우 admission class에서 제외하고 그 경계를 결과로 남긴다.
2. **C159-Q3 offline oracle screen:** calibration 네 window에서 미래를 아는 oracle조차
   static-safe 또는 safe work-conserving 대비 사전 MDE를 만들 수 있는지 계산한다. 상한이
   MDE 미만이면 성능 우위 주장은 열지 않고 envelope 결과로 종료한다.
3. **C159-Q4 online replay:** oracle screen을 통과한 경우에만 동일 max-radio controller로
   static safe, safe work-conserving, SoftWall을 ABBA 실행한다. 완료 request, deadline 내
   token value, expired/late/rejected와 모든 radio safety gate를 함께 보고한다.
4. **C160/C161 fault matrix:** NRx late/fail correlation, Qwen response delay, worker crash,
   stale/duplicate reply, fence loss를 P180 mode에 주입한다.
5. **C162 envelope:** cell/home 수, context class, cap, lifecycle, memory에 대한
   qualified-safe-useful / safe-no-slack / infeasible / unqualified 경계를 예측하고 holdout으로
   검증한다.

C159-Q2가 다음 즉시 실행 gate다. BurstGPT holdout 구간은 Q2와 oracle screen, 최종 protocol이
동결되기 전까지 materialize하지 않는다.
