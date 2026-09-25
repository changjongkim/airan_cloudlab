> **보관 사유 (2026-09-25):** 이 문서는 실험 이력으로 보존한다. 최신 스킴과 claim boundary는 `docs/current`의 원고·모델·production gate 문서가 권위 기준이며, 과거 GPU0 전용, MIG, CloudLab 또는 4-GPU pool 가정은 현재 논문의 근거가 아니다.

# SoftWall C160--C162 실험 계획: 통합 fault와 feasibility envelope 완결

**상태:** 2026-09-25, C160 pure-state PASS / C161 A0--A6 qualified-node PASS /
C162 predictive envelope two-node PASS  
**현재 권위 mode:** 4×A100, MIG OFF, GPU2 cuPHY cap80+Qwen cap20, GPU3 actual NeuralRx,
GPU0/1 RAN home, `P180/D155`, warm persistent lifecycle  
**목적:** 처리량 우위를 더 찾지 않고 conditional-recovery substrate의 고장 의미론과 예측 가능한
safe/useful/infeasible 경계를 완결한다.

## 1. 남은 두 주장

1. **Fault containment:** 결과·응답·fence가 지연되거나 중복돼도 미확정 recovery/AI credit을
   재사용하지 않고, mandatory RAN은 자격화된 fault bound 안에서 계속 처리한다.
2. **Envelope prediction:** `(debt, decision time, class, lifecycle, memory)`를 입력하면 QSU/QSN/
   MI/UQ를 실행 전에 예측하며, 물리 holdout에서 false-safe가 없어야 한다.

C159-Q3가 strong Event-driven 대비 token headroom 0.000%를 냈으므로 C160--C162의 성공을
새 처리량 우위로 표현하지 않는다.

## 2. 공통 불변식

모든 safe arm은 다음을 만족해야 한다.

```text
I1  admitted TB마다 D155 이전 single commit 정확히 1회
I2  unresolved NRx debt 전부를 포함하는 executable all-fail certificate 유지
I3  stale/duplicate outcome은 generation을 바꾸거나 credit을 해제하지 않음
I4  Qwen lease는 physical completion fence 또는 검증된 non-launch 전에는 retire하지 않음
I5  common cutoff에서 관측한 success set은 한 atomic batch로 적용
I6  bounded revalidation을 통과한 최신 certificate만 physical launch 가능
I7  fault 뒤 outstanding debt/lease/IPC credit 수가 analyzer 계산과 일치
```

Safety violation 한 건은 평균으로 상쇄하지 않는다. 해당 `(fault, mode, lifecycle)`을 UQ로
내리고 원인을 교정한 새 source hash에서 처음부터 다시 자격화한다.

## 3. C160 — CPU/state-machine fault closure: 완료

GPU 실행 전에 deterministic controller와 transport double로 다음 전이를 전수 검사한다.

| ID | 주입 지점 | 기대 동작 |
|---|---|---|
| F1 | common cutoff 뒤 이전 epoch NRx success 재전송 | stale epoch 폐기, debt/세대 변화 0 |
| F2 | 같은 success 두 번 전송 | 첫 batch만 반영, duplicate로 추가 credit 해제 0 |
| F3 | 네 NRx outcome 중 일부/전부 cutoff 이후 도착 | late 결과는 success로 사용하지 않고 해당 recovery 실행 |
| F4 | certificate 계산 뒤 dispatch가 5 ms 초과 | stale certificate 폐기, 최대 3회 현재 clock 재계산 |
| F5 | Qwen worker가 `latest_start` 뒤 request 수신 | kernel launch 0, empty lease fence-confirmed retire |
| F6 | Qwen kernel 완료 뒤 reply만 유실/지연 | matching completion marker가 있을 때만 retire; 이후 AI quarantine |
| F7 | Qwen launch 뒤 fence/marker 없이 channel 단절 | lease 미반환, 이후 AI admission 차단, RAN debt 보존 |
| F8 | Recovery response stale/duplicate | owner epoch/generation 검사, radio duplicate commit 0 |
| F9 | Atomic update 적용 뒤 ACK 유실 | 같은 transaction ID 재시도는 결과 조회만, 중복 lease/recovery 0 |

필수 model-check 상태는 `epoch relation(old/current/future) × generation relation × launched ×
fence × ACK`의 유효 조합이다. 최소 gate는 모든 유효 state/edge에서 invariant violation 0,
reject 경로 state mutation 0, duplicate physical execution 0이다.

C160 v1은 unit test8개와 51개 state-transition product에서 invariant violation0과 reject-path
mutation0으로 통과했다. 첫 analyzer의 fence-history 누락 실패는 보존했다. 상세는
[C160 결과](SOFTWALL_C160_FAULT_MODEL_RESULT_KO.md)에 있다.

## 4. C161 — P180 실제 GPU 통합 fault matrix

### 4.1 Arm

| Arm | 물리 주입 | 주 표본 | 판정 목적 |
|---|---|---:|---|
| A0 | no fault | 300 epoch/node | 같은 frozen source의 control |
| A1 | 매 25 epoch에 accepted NRx 네 건의 success 공개를 모두 억제 | fault 12회/node | correlated all-fail→4 recovery |
| A2 | 매 30 epoch에 old/duplicate NRx outcome | 각 10회/node | stale/duplicate credit release 차단 |
| A3 | 매 40 epoch에 Qwen post-fence reply를 control bound 밖으로 지연 | 7회/node | marker 기반 retire와 AI quarantine |
| A4 | 매 50 epoch에 latest-start 전 host hold | 6회/node | physical non-launch와 empty lease retire |
| A5 | 한 번 Qwen channel을 pre-fence에서 단절 | 1회/node | uncertain lease 보존, 이후 AI 차단, RAN continuation |
| A6 | 매 30 epoch에 recovery reply duplicate/stale | 각 10회/node | owner single-commit/generation |

각 arm은 16/128/512 context를 고정 순환해 짧은 class와 conditional long class를 모두 포함한다.
A5는 GPU hang을 가장하지 않는다. Process/channel fail-stop과 확인 가능한 fence 부재까지만
대상으로 하며, driver hang은 UQ로 남긴다.

### 4.2 순서와 독립성

```text
development node: A0 A1 A2 A3 A4 A5 A6
holdout node:     A6 A5 A4 A3 A2 A1 A0
```

- Development에서 semantics를 수정한 뒤 source와 analyzer를 동결한다.
- Holdout은 새 allocation/node/seed를 사용하고 development node를 사전 제외한다.
- 각 fault는 epoch와 transaction ID를 protocol에 미리 고정한다.
- Development/pilot은 holdout 합계나 fault-rate CI에 포함하지 않는다.

### 4.3 Gate

- 모든 arm의 deadline miss, wrong commit, duplicate commit, input/response mismatch 0
- A1에서 fault epoch마다 recovery 정확히 4개, Qwen admission은 certificate 결과와 일치
- A2/A6에서 stale/duplicate 수신 수는 양수, state mutation과 duplicate physical action 0
- A3에서 injected reply delay는 양수, matching fence 없는 lease retire 0
- A4에서 injected late-start 수는 양수, physical Qwen kernel launch 0
- A5 fault 뒤 새 Qwen launch 0, 이후 최소 50 radio epoch와 모든 mandatory recovery 지속
- 종료 시 analyzer가 설명하지 못한 obligation/lease/IPC credit 0

### 4.4 1단계 결과와 남은 범위

C161 1단계는 A0/A1/A4를 frozen source로 두 독립 A100 node에서 반대 arm 순서로 실행했다.
합계 180 epoch·actual NeuralRx720·radio commit720에서 deadline miss는 0이었다. A1은 물리적으로
관측된 success 공개를 common cutoff에서 억제해 네 debt를 유지했고, 두 node에서 정확히
240건의 correlated conventional recovery를 실행했다. A4는 commit된 lease의 latest-start 뒤
dispatch를 52회 만들었으며 Qwen worker는 모두 physical non-launch로 거절했다. 두 campaign은
각각 15/15 gate를 통과했다.

2단계는 A2/A3/A5/A6를 frozen source로 두 추가 A100 node에서 반대 순서로 실행했다. 합계
520 epoch·actual NeuralRx2,080·radio commit2,080에서 miss0이었다. A2/A6 duplicate/stale
pair는 각각20회, A3/A5 terminal fault는4회, fault 뒤 radio continuation은260 epoch였다.
따라서 C160과 C161 1·2단계를 합친 A0--A6 matrix는 qualified warm node에서 통과했다.

같은 최종 source의 별도 node `nid001044`는 sequence-1 owner-observed NRx가 47.470 ms로
45 ms bound를 초과했다. 이를 숨기지 않고 해당 node/lifecycle을 UQ로 남긴다. 상세는
[C161 1단계 결과](SOFTWALL_C161_PHASE1_RESULT_KO.md)와
[C161 2단계 결과](SOFTWALL_C161_PHASE2_RESULT_KO.md)에 있다.

## 5. C162 — 예측 모델과 물리 boundary holdout

### 5.1 Mode vector

```text
theta = (
  homes, accepted_debts, unresolved_debts,
  decision_time, B_NRx, B_recovery, B_control,
  Qwen_class_bound, P, D, capacity,
  MPS caps, lifecycle, resident_memory, fault_class
)
```

분류는 네 가지다.

| 분류 | 정의 |
|---|---|
| QSU | Mandatory all-fail safe이고 하나 이상의 자격화된 AI class가 deadline 내 완료 가능 |
| QSN | Mandatory safe지만 요청 class를 넣을 certified slack이 없음 |
| MI | Mandatory all-fail schedule 자체가 불가능 |
| UQ | timing/memory/lifecycle/fault bound가 없거나 물리 gate 실패 |

### 5.2 먼저 계산할 grid

- unresolved debt `0..5`
- decision time `45..153 ms`, 1 ms 간격
- Qwen class `{16,32,64,128,256,512}`
- recovery bound `{12,25}` ms에서 12 ms는 과거 false qualification으로 UQ 표시
- lifecycle `{warm, cold/restart, long-idle}`
- resident receiver/home topology `{1×4, 2×4, 4×3}`
- fault class `{none, bounded response delay, ambiguous fence}`

Exact small-state oracle와 analytic classifier가 모든 bounded state에서 같아야 한다. Runtime이
사용하지 않는 12 ms나 cold mode를 계산상 safe로 올리지 않도록 qualification provenance를
분류 입력에 포함한다.

### 5.3 물리 holdout 점

| 점 | 예상 | 이유 |
|---|---|---|
| E1: debt4, unresolved4, AI 없음 | QSN | mandatory all-fail의 마지막 safe point |
| E2: debt5 | MI/reject | `45+5×25+2 >155` |
| E3: unresolved2, context128 | QSU | `45+2×25+5+40 <=153` |
| E4: unresolved2, context256 | QSN | `45+2×25+5+65 >153` |
| E5: unresolved1, context512 | QSU | `45+25+5+75 <=153` |
| E6: late decision boundary 양쪽 | QSU/QSN 전이 | launch-time revalidation 경계 |
| E7: warm/cold 같은 산술 | warm QSU, cold UQ | whole-path preflight 64--72 ms 반례 |
| E8: 8 receiver single-home | memory MI/UQ | C84 pretraffic OOM 경계 |

각 점은 development와 독립 holdout에서 같은 분류가 나와야 한다. 가장 심각한 오류는 예측
QSU인데 물리 safety gate가 깨지는 false-safe다. False-conservative는 별도로 세고 slack 손실을
보고한다.

### 5.4 결과

C162 model은 qualified grid 16,023개에서 analytic/exact mismatch 0을 얻었다. QSU 2,664,
QSN 5,281, MI 8,078이었다. Lifecycle·node·bound·memory provenance 162개에서도 미자격
mode를 safe로 승격한 경우가 없었다. 과거 C159-Q2 두-node 1,200 epoch의 lease decision을
사후 재생한 정합성 검사는 1,200/1,200 일치했다.

새 물리 boundary는 development `nid001265` 60 rounds와 이 node를 제외한 holdout
`nid001353` 120 rounds를 서로 다른 seed·반대 case 순서로 실행했다. 합계 actual NRx720,
injected physical recovery330, Qwen90, single commit720, deadline miss0이다. 여섯 점은 각각
30회였고 model/lease mismatch는0이었다. E6a는 30/30 모두 conservative 88 ms에 허가,
E6b는 30/30 모두 89 ms 이상에서 거절됐다. 상세 수치와 한계는
[C162 결과](SOFTWALL_C162_PREDICTIVE_ENVELOPE_RESULT_KO.md)와
[두-node 판정](../../results/softwall_multigpu/c162_boundary_two_node.json)에 있다.

후속 certified scheduler는 임의 이질 small state600개에서 exact 대비 false-safe0,
false-conservative10을 냈고 현재 mode16,023개에서는 판정 차이0이었다. Debt1--64의
large grid2,800개에서 반환 schedule의 verifier 실패0, debt64 decision/verifier p99는
1.494/0.131 ms였다. 이는 5 ms control budget 안의 finite CPU 결과이며 WCET가 아니다.

## 6. 완료 기준과 논문 연결

| 결과 | 논문에 들어가는 주장 |
|---|---|
| C160/C161 PASS | Stale/duplicate/late/fence-loss의 제한된 fail-closed containment |
| C162 false-safe 0 | Mode-qualified predictive feasibility envelope |
| C161 fault FAIL | 해당 fault/lifecycle UQ, 일반 fault-tolerance 문구 삭제 |
| C162 false-safe | 모델 수정 후 새 holdout 전까지 envelope claim 보류 |

최종 논문에서 성능 결과는 C159-Q3의 no-headroom 판정으로 보고한다. 핵심 positive result는
“더 많은 token을 처리한다”가 아니라 “conditional recovery debt를 물리 resource와 함께
안전하게 가상화하고, 가능한 영역과 불가능한 영역을 실행 전에 구분한다”이다.
