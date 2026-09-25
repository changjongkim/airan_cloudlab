# C159 actual-NRx trace 실험 사전 명세

**상태:** 2026-09-25 prespec 고정, C159-Q1/Q2 두-node P180 PASS, Q3 oracle screen에서 성능 holdout 중단  
**선행 gate:** C159-Q1 `C159_Q1_TWO_NODE_P180_PASS`  
**목적:** V17.1 shared-recovery substrate의 `P180/D155` online mode를 먼저 자격화하고,
같은 radio 정책을 쓰는 강한 안전 비교군 대비 적시 external-AI value와 비용을 판정한다.

## 1. C158에서 이어받는 것과 다시 검증할 것

C158에서 이어받는 실행 의미론은 다음과 같다.

- GPU0/1: 두 RAN home과 final radio commit
- GPU2: persistent shared cuPHY conventional worker와 MPS Qwen worker
- GPU3: persistent TensorRT NeuralRx worker
- global all-fail certificate, launch-time `control+AI` revalidation, atomic recovery
  retiming/AI lease와 generation-safe retire
- persistent mmap decision page, byte-exact input round-trip, response read-back echo,
  normalized CRC wire status와 readiness 전 P2P/monitor preflight

C158의 `P600`은 validation-only oracle와 다음 epoch input staging을 deadline 밖에 두기 위한
qualification 간격이었다. C159에서는 다음을 새로 자격화한다.

1. 모든 noisy PUSCH input과 transmitted payload를 campaign 시작 전에 준비한다.
2. Timed loop에서 local validation oracle와 파일 생성·rename을 제거한다.
3. Radio를 후보 `P180/D155`로 실행한다.
4. Context `{16,32,64,128,256,512}` Qwen class를 shared cuPHY co-run mode에서 다시
   자격화한다. C113의 bound를 자동 상속하지 않는다.
5. BurstGPT arrival를 online queue로 넣고 미래 arrival·CRC·service time을 숨긴다.

## 2. 고정할 timing contract

| 항목 | 후보 값 | 판정 |
|---|---:|---|
| radio period / expiry | 180 / 155 ms | C159-Q1 두 node 재자격 PASS |
| NeuralRx end-to-end | 45 ms | Q1 actual NRx2,000, 최대23.882 ms |
| shared recovery path | 25 ms | Q1 recovery582, 최대11.555 ms, input/output fence 포함 |
| launch-control | 5 ms | actual decision부터 GPU submit까지 |
| recovery guard | 2 ms | 다음 recovery와 AI physical completion 분리 |
| Qwen 16/32/64 | 35/35/35 ms | Q2 두 node 최대28.831/24.550/25.466 ms PASS |
| Qwen 128/256/512 | 40/65/75 ms | Q2 두 node 최대25.770/40.579/67.338 ms PASS |
| AI request SLO | primary 1,000 ms | BurstGPT 원본에 없는 synthetic mapping |

Runtime은 요청의 context class에 대응하는 전체 physical-completion bound를 사용한다. 해당
class가 C159-Q에서 자격을 얻지 못하면 그 class를 더 짧은 bound로 재해석하지 않고
`unqualified class`로 거절한다.

## 3. Workload split과 holdout 보호

원본은 `data/public/burstgpt/BurstGPT_1.csv`이고 SHA-256은
`46fc9480ef0b748ecb2b51d512ff08c196b031782cbe6f78e28044d768e86d5a`다. Positive request와
response token만 사용한다. Timestamp 범위 `[5, 5,269,973]`을 source time으로 세 구간에
나눈다.

| 구간 | source timestamp | 용도 |
|---|---:|---|
| development | `[5, 1,756,661)` | 구현·실패 수정 전용 |
| calibration | `[1,756,661, 3,513,317)` | C159-Q, 분산·표본 수 결정 |
| confirmatory holdout | `[3,513,317, 5,269,974)` | frozen C159-P에서만 사용 |

각 구간에서는 valid request 수가 많은 60초 창을 deterministic하게 고른다. 여러 창이
필요하면 request count 내림차순, start timestamp 오름차순으로 정렬하고 서로 겹치지 않는
창만 순서대로 선택한다. Calibration outcome으로 holdout 창을 바꾸지 않는다.

원본 timestamp가 1초 해상도이므로 primary replay는 같은 초의 요청을 원래 source order로
동시에 도착시킨다. 같은 요청을 그 초 안에 deterministic midpoint로 분산한 replay를
time-resolution sensitivity로 별도 보고한다. Primary claim은 burst replay를 기준으로 한다.

## 4. 비교군과 유일한 차이

세 safe system 모두 같은 actual PUSCH input ID, max-radio rule, NeuralRx endpoint,
global all-fail 검사, service profile, MPS cap, AI request와 single-commit/fault rule을 쓴다.

| 시스템 | 동작 |
|---|---|
| Static global-safe | all-fail recovery interval을 고정하고 처음부터 보이는 안전 gap에만 AI 실행 |
| Event-driven global-safe | NRx 결과 뒤 credit을 삭제하고 현재 contiguous gap을 사용하되 여러 recovery를 재정렬하지 않음 |
| SoftWall | 남은 recovery를 global certificate 안에서 retime하고 새 calendar와 AI lease를 한 generation에서 원자 commit |

`Uncontrolled MPS`와 `local-only certificate`는 실패 원인을 보이는 diagnostic arm으로만
사용하며 timely-token 성능 순위에 포함하지 않는다. Offline oracle은 미래 outcome/arrival를
보는 상한이며 online baseline으로 세지 않는다.

## 5. 실행 단계

### C159-D — 구현과 개발

- Development partition만 사용한다.
- Pre-staged PUSCH bank, full recovery-path preflight, variable-context Qwen worker와 class별
  certificate telemetry를 구현했다. Online 세-policy replay는 Q3 중단 gate로 실행하지 않는다.
- 오류 수정 때마다 새 attempt label과 source hash를 사용한다.
- 이 표본은 bound, power 또는 성능 CI에 사용하지 않는다.

### C159-Q1 — context-64와 `P180` mode 자격: 완료

- 서로 다른 두 A100 node에서 node당 actual NRx request 1,000건을 실행했다.
- Context-64 Qwen, NRx45, recovery25, P180/D155와 full-path warm lifecycle을 자격화했다.
- 합계 actual NRx2,000, recovery582, Qwen500, commit2,000에서 deadline, class bound,
  exact input/output fence, generation, single commit, residual credit 위반은 모두 0이었다.
- 첫 250-epoch primary FAIL의 cold path tail을 보존하고 whole-path preflight로 lifecycle을
  교정했다. [Q1 결과](SOFTWALL_CONFIRM159_Q1_P180_RESULT_KO.md)를 따른다.

### C159-Q2 — variable-context component 자격: 완료

- `{16,32,64,128,256,512}`를 각각 `35/35/35/40/65/75 ms`로 고정하고 Q1과 같은
  P180/D155, placement, MPS cap, actual-NRx/fence semantics에서 실행했다.
- 두 독립 node 합계 1,200 epoch, actual NRx4,800, recovery1,312, Qwen1,077, radio
  commit4,800에서 deadline·transport·recovery contract 위반은 0이었다.
- 각 node에서 class별 physical completion을 최소 50개 확보했고 전체 class 최대는
  28.831/24.550/25.466/25.770/40.579/67.338 ms로 frozen bound 안이었다.
- 256/512는 unresolved recovery가 많을 때 각각 55/68회 certificate가 launch 전에 거절했다.
  이는 class failure가 아니라 conditional envelope 관측이다.
- 개발 중 stale certificate 한 건과 sequential outcome replay 반례를 보존했다. 이를 5 ms
  bounded revalidation, physical latest-start gate와 common-cutoff success-set atomic transition으로
  교정한 뒤 개발/holdout 16개 gate를 모두 통과했다.
- 상세는 [Q2 결과](SOFTWALL_CONFIRM159_Q2_VARIABLE_RESULT_KO.md)와
  [결합 판정](../../results/softwall_multigpu/confirm159_q2_variable_two_node.json)을 따른다.

### C159-Q3 — oracle screen: 완료, 성능 holdout 중단

Q2의 두 실제 outcome stream과 calibration 네 60초 창을 사전에 고정해 exact offline weighted
matching을 수행했다. 두 정책은 같은 radio admission inequality와 같은 미래를 아는 request
selector를 사용한다. 차이는 SoftWall이 AI를 unresolved recovery보다 먼저 원자 배치하고,
강한 Event-driven 기준선은 recovery를 물리 완료한 뒤 AI를 시작한다는 실행 순서뿐이다.

| 비교 | Event-driven | SoftWall | 개선 |
|---|---:|---:|---:|
| Q2 실측 recovery 시간을 사용한 강한 기준선 | 385,262 token / 931 request | 385,262 / 931 | **0.000%** |
| recovery마다 선언 25 ms를 전부 청구한 보수적 sensitivity | 385,007 / 930 | 385,262 / 931 | **0.066%** |

네 window의 실측-recovery 비교가 모두 0.000%였고 사전 MDE 5%에 미달했다. 따라서 Q3의
사전 중단 규칙에 따라 C159-P confirmatory performance holdout은 materialize하지 않는다.
이 결과는 online 성능 CI가 아니라 현재 trace/SLO/mode에 material optimization headroom이
없다는 calibration-only upper-bound 판정이다. Prespec과 결과는 각각
[`confirm159_q3_oracle_prespec_v1.json`](../../results/softwall_multigpu/confirm159_q3_oracle_prespec_v1.json)과
[`confirm159_q3_oracle_screen_v1.json`](../../results/softwall_multigpu/confirm159_q3_oracle_screen_v1.json)에 있다.

### C159-P — 동결 holdout 비교: Q3 gate로 열지 않음

- 아래 절차는 Q3가 5% gate를 통과할 때만 적용하기로 사전 정의했다. Q3 결과가 0.000%여서 실행하지 않았다.
- C159-Q2/Q3 뒤 source, container, engine, mode, trace-selection code, holdout window 수,
  seed block, arm order와 analyzer hash를 protocol에 고정한다.
- 두 node에서 각 window를 다음 balanced order로 실행한다.

```text
node A: Static -> EventDriven -> SoftWall -> SoftWall -> EventDriven -> Static
node B: SoftWall -> Static -> EventDriven -> EventDriven -> Static -> SoftWall
```

- 각 paired arm의 offered radio input ID와 AI request ID·arrival·deadline·context·value,
  mode fingerprint는 byte-identical이어야 한다.
- C159-Q와 development 결과는 confirmatory CI에 합치지 않는다.

## 6. Metric과 통계 단위

Primary metric은 deadline 안에 끝난 `value_tokens`다. Request 하나나 TB 하나를 독립 표본으로
세지 않는다. 기본 paired 단위는 `source 1-second block × trace window × node × seed block`이다.

Secondary metric은 다음과 같다.

- timely request 수, reject/expire/late 수와 context별 completion
- GPU busy time, Qwen/recovery kernel time, certificate fragmentation
- controller replan/commit latency와 atomic exchange 수
- offered/admitted/completed radio utility, deadline miss, correct TB와 commit kind
- 가능할 때 energy와 joule/timely-token

Primary 비교는 SoftWall 대 Event-driven이다. Static 비교는 recovery reservation 비용을
보여 준다. Paired hierarchical bootstrap은 먼저 window/node를 resample하고 그 안에서 source
second block을 resample한다. 효과 크기, 95% CI와 raw per-window 방향을 함께 보고한다.

## 7. 사전 gate와 해석

### Safety gate

- 모든 safe arm의 radio deadline miss, NRx/recovery/Qwen bound 위반 0
- input round-trip, response echo, generation/credit, single-commit 위반 0
- run 종료 시 outstanding obligation, lease, endpoint/transport credit 0
- physical Qwen launch는 committed lease와 일치

### Comparability gate

- paired offered trace와 radio input signature 완전 동일
- 같은 model/engine/container/MPS cap과 warm lifecycle
- 각 arm 시작 전 readiness와 preflight 통과
- analyzer가 현재 source hash와 protocol hash를 재검증

### Performance gate

- SoftWall radio utility가 Event-driven에 비열등
- 두 node와 두 실행 방향에서 효과 부호가 일치
- hierarchical paired-bootstrap 95% CI 하한 `> 0`
- 중앙 timely-token 개선율 `>= 5%`

Performance gate가 실패하면 처리량 우위를 주장하지 않는다. 그 경우에도 safety gate를 통과한
C159은 실제 trace 아래 substrate 동작과 overhead/no-benefit 영역의 증거로 사용한다.

## 8. C159 뒤 실험

1. **C160/C161 integrated fault matrix:** correlated NRx fail/late, stale/duplicate result,
   Qwen/recovery response delay, pre-fence timeout, broker post-apply reply loss와 worker exit.
2. **C162 envelope:** home/debt 수, P/D, recovery capacity, context class, cap, lifecycle와
   memory 축에서 QSU/QSN/MI/UQ 예측 후 경계점 holdout.
3. **C163 sensitivity/scalability:** trace time-resolution, SLO, load와 certificate 계산시간.
4. 실제 DU/FAPI `d_MAC`을 얻으면 synthetic D155와 분리한 production-mode 재자격.

이 순서는 성능 결과로 safety bound를 바꾸거나, 실패한 mode를 더 긴 period의 결과로 덮는
것을 막는다.
