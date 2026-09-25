> **보관 사유 (2026-09-25):** 이 문서는 실험 이력으로 보존한다. 최신 스킴과 claim boundary는 `docs/current`의 원고·모델·production gate 문서가 권위 기준이며, 과거 GPU0 전용, MIG, CloudLab 또는 4-GPU pool 가정은 현재 논문의 근거가 아니다.

# C159-Q2 variable-context 실제 경로 재자격 결과

**최종 판정:** `C159_Q2_TWO_NODE_VARIABLE_P180_PASS`  
**개발:** job `58857672`, node `nid001204`, label `confirm159q2f_batch_dev_j58857672`  
**독립 holdout:** job `58858194`, node `nid001632`, label `confirm159q2g_batch_holdout_j58858194`  
**결합 결과:** `results/softwall_multigpu/confirm159_q2_variable_two_node.json`  
**162-file manifest:** `results/softwall_multigpu/confirm159_q2_variable_manifest.json`

## 1. Q2가 검증한 질문

C159-Q1은 context-64 Qwen만 사용했다. 실제 BurstGPT trace는 대부분 256/512-token class이므로,
고정 35 ms AI unit으로 얻은 안전 결과를 trace replay에 그대로 적용할 수 없었다. Q2는 같은
`P180/D155` 실제 NeuralRx 경로에서 Qwen context class마다 독립 physical-completion bound를
사용하고, 현재 unresolved recovery debt가 그 class를 수용할 수 있을 때만 AI lease를 발급한다.

동결한 class와 bound는 다음과 같다.

| context token | bound | 두 recovery debt가 남은 상태 | 한 개 이하 debt가 남은 상태 |
|---:|---:|---|---|
| 16 | 35 ms | 수용 가능 | 수용 가능 |
| 32 | 35 ms | 수용 가능 | 수용 가능 |
| 64 | 35 ms | 수용 가능 | 수용 가능 |
| 128 | 40 ms | 수용 가능 | 수용 가능 |
| 256 | 65 ms | certificate가 거절할 수 있음 | 수용 가능 |
| 512 | 75 ms | certificate가 거절할 수 있음 | 수용 가능 |

이 표의 거절은 오류가 아니다. NRx outcome으로 recovery obligation이 충분히 해소되지 않은
epoch에서 긴 AI를 시작하지 않는 것이 바로 conditional admission이다.

## 2. Q2에서 추가된 실행 의미론

Q2 개발 과정에서 certificate의 시간 의미를 두 군데 강화했다.

### 2.1 Bounded launch-time revalidation과 physical start gate

Certificate가 계산된 뒤 host가 오래 정지하면 계산 당시의 lease interval은 더 이상 유효하지
않다. Runtime은 certificate 계산과 dispatch 사이가 5 ms를 넘으면 결과를 버리고 현재 clock으로
다시 계산한다. 최대 세 번까지만 허용한다. Worker에는 절대 `latest_start_ns`를 전달하며, 그
시각이 지난 뒤에는 Qwen kernel을 시작하지 않고 빈 lease를 fence-confirmed 상태로 반환한다.

### 2.2 Common-cutoff outcome의 atomic batch transition

네 NeuralRx 결과는 한 common cutoff에서 함께 관측한다. 이미 관측한 success를 재검증 때 하나씩
적용하면 실제로 존재하지 않은 partial-outcome 상태가 중간에 나타난다. 늦어진 current time에서
그 중간 상태가 불가능하면 최종 observed state가 가능한데도 잘못 거절할 수 있다. Q2 최종
runtime은 common cutoff에서 관측한 success set 전체를 한 transaction으로 제거하고, 남은
unresolved obligation만으로 recovery와 AI certificate를 다시 만든다.

따라서 최종 transaction의 의미는 다음과 같다.

```text
observed success set at cutoff
  -> remove the full set atomically
  -> rebuild unresolved recovery calendar at current time
  -> check class-specific Qwen lease
  -> commit calendar + lease in one generation
  -> physical worker enforces absolute latest-start
```

## 3. 두 독립 node의 권위 결과

| 지표 | 개발 `nid001204` | holdout `nid001632` | 합계/최대 |
|---|---:|---:|---:|
| epoch | 600 | 600 | 1,200 |
| actual NeuralRx | 2,400 | 2,400 | 4,800 |
| 제때 NRx success | 1,750 | 1,738 | 3,488 |
| physical recovery | 650 | 662 | 1,312 |
| Qwen completion | 544 | 533 | 1,077 |
| radio commit | 2,400 | 2,400 | 4,800 |
| NRx 최대 | 18.052 ms | 18.123 ms | 18.123 ms |
| recovery 최대 | 4.933 ms | 13.465 ms | 13.465 ms |
| radio commit 최대 | 117.236 ms | 118.022 ms | 118.022 ms |
| launch revalidation retry epoch | 1 | 1 | 2 |
| deadline miss | 0 | 0 | 0 |

두 protocol은 같은 source hash를 사용한다. Job, node, seed는 서로 다르고 개발 node는 holdout
protocol에서 사전에 제외됐다. 결합 판정의 13개 gate가 모두 통과했다.

- 1,200 epoch 전부에서 common-cutoff success를 atomic batch로 적용했다.
- 4,800 NRx input round-trip, 1,312 recovery contract/response echo, 4,800 single commit의
  위반은 모두 0이다.
- 두 node에서 각각 한 번씩 5 ms revalidation이 실제로 발동했고, 새 certificate로 안전하게
  진행했다.
- Physical latest-start 뒤 시작한 Qwen kernel은 0건이다.
- BurstGPT confirmatory holdout은 아직 materialize하지 않았다.

## 4. Class별 결과

| context | offered | completed | certificate reject | 최대 execution | bound |
|---:|---:|---:|---:|---:|---:|
| 16 | 200 | 200 | 0 | 28.831 ms | 35 ms |
| 32 | 200 | 200 | 0 | 24.550 ms | 35 ms |
| 64 | 200 | 200 | 0 | 25.466 ms | 35 ms |
| 128 | 200 | 200 | 0 | 25.770 ms | 40 ms |
| 256 | 200 | 145 | 55 | 40.579 ms | 65 ms |
| 512 | 200 | 132 | 68 | 67.338 ms | 75 ms |

각 node에서 각 class가 최소 50개의 physical completion을 확보했다. 모든 physical Qwen
execution은 class bound 안에 있었다. 256/512의 123건은 worker failure가 아니라 현재
recovery certificate가 긴 unit을 수용하지 못해 GPU launch 전에 거절한 결과다. 이 차이가
trace-driven 평가에서 SoftWall이 회수할 수 있는 conditional capacity의 근거가 된다.

## 5. 보존한 두 실패와 설계 교정

| 실행 | 관측 | 원인 | 교정 |
|---|---|---|---|
| `q2b` 600 epoch | seq352에서 stale lease 1건 | certificate 뒤 host control tail 23.530 ms | 5 ms 초과 certificate 폐기·최대 3회 재계산 + worker latest-start |
| `q2d` 349 epoch 뒤 중단 | `actual success transition failed` | 동시 관측 success를 순차 재적용하여 허구의 partial state 생성 | observed-success set 원자 제거 후 unresolved debt만 재구성 |

`q2b`의 문제 epoch에서 context-128 Qwen 실행 자체는 23.894 ms로 40 ms bound 안이었다. 실패는
GPU service bound가 아니라 계산 후 dispatch까지의 certificate freshness 위반이다. 이를 bound를
늘려 숨기지 않고 lease lifecycle을 보강했다. `q2d`도 표본에서 제외하고 partial execution과
stack trace를 보존했다.

별도 증거 파일은 다음과 같다.

- `results/softwall_multigpu/confirm159q2b_launch_control_failure.json`
- `results/softwall_multigpu/confirm159q2d_sequential_revalidation_failure.json`

## 6. 논문 주장에 추가되는 것

Q2는 SoftWall이 고정 길이 background job만 다루는 calendar가 아니라 다음을 함께 수행한다는
물리 근거를 추가한다.

1. 실제 NeuralRx outcome에 따라 unresolved recovery debt를 바꾼다.
2. Context별 Qwen completion bound를 certificate에 넣는다.
3. 여러 동시 outcome과 recovery retiming, AI lease를 하나의 현재-time transaction으로 만든다.
4. Host tail이 있으면 stale certificate를 버리고, 실제 worker의 kernel start 시각까지 lease를
   강제한다.

이는 새로운 joint optimizer의 우월성 주장이 아니다. 같은 max-radio 선택을 유지하면서
conditional recovery obligation을 안전한 external-AI capacity로 가상화하는 substrate 주장이다.

## 7. 해석 한계와 다음 gate

현재 결과는 warm synthetic `P180/D155` mode에서 두 A100 node로 얻은 유한 표본 자격이다.
WCET, cold/long-idle, production `d_MAC`, integrated crash/fence-loss fault, 실제 BurstGPT online
throughput을 아직 증명하지 않는다.

다음 C159-Q3는 calibration trace에서 미래를 아는 offline oracle의 상한을 먼저 계산한다.
Oracle조차 Event-driven safe baseline보다 5% 이상 적시 token value를 만들 수 없으면 성능 우위
holdout을 열지 않고 `qualified-safe but no material optimization headroom`이라는 envelope 결과로
종료한다. 5% 이상이면 그때만 source, arm order, window 수를 고정하고 confirmatory BurstGPT
holdout을 materialize한다.
