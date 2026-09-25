# SoftWall novelty defense matrix

**상태:** 2026-09-25 공개 선행 연구 재검색과 C162/C164 claim freeze 반영  
**논문축:** certified conditional-recovery substrate + predictive feasibility envelope  
**제외한 축:** MPS 최초성, AI-RAN GPU sharing 최초성, 새 optimizer/처리량 우위

## 1. 논문이 실제로 새롭다고 주장할 대상

SoftWall의 novelty는 구성 요소 목록이 아니라 다음 세 계약의 연결에 있다. 증거 강도에 따라
refinement chain과 predictive envelope를 먼저 두며, recovery debt 자체를 새로운 일반
primary/backup scheduling 이론으로 주장하지 않는다.

### N1. Certificate-to-physical-execution refinement

Certificate는 계산 결과만으로 안전하지 않다. SoftWall은 다음을 한 generation-safe
transaction으로 묶는다.

```text
observed NRx outcome batch
  + unresolved recovery calendar
  + endpoint/ring/IPC/P2P credit
  + external-AI ownership token
  + physical latest-start and completion fence
  -> atomic commit or no state change
```

이 연결이 필요한 이유는 구현 반례로 확인됐다. Stale lease는 certificate 이후 23.530 ms
host tail에서 깨졌고, sequential outcome replay는 존재하지 않는 partial state를 만들었으며,
fixed lease는 launch control을 누락했고, V14는 두 번째 prepare가 untracked token을 만들었다.
각 반례는 bounded revalidation, common-cutoff batch, launch-time blackout, single-token
ownership으로 수정됐다. 따라서 기여는 RPC, CAS, fence 각각의 최초성이 아니라
same-TB recovery certificate의 의미를 실제 CUDA/P2P 실행까지 보존하는 refinement다.

### N2. Provenance-qualified predictive feasibility envelope

SoftWall은 한 실험점의 miss0을 일반화하지 않는다. `(debt, decision time, AI class,
resource capacity, bound provenance, lifecycle)`을 입력으로 다음 네 상태를 분류한다.

- QSU: RAN-safe이며 자격화된 AI unit도 허가 가능
- QSN: RAN-safe이나 해당 AI unit의 안전한 여유 없음
- MI: mandatory all-fail schedule 자체가 불가능
- UQ: 필요한 service/lifecycle bound가 자격화되지 않음

C162는 qualified state 16,023개에서 exact mismatch0, 과거 physical decision 1,200개 일치,
두 독립 node의 prespecified boundary 180개 일치를 얻었다. Context64의 88 ms 허가와
89 ms 거절을 각30회 재현했고, 64-debt certified scheduler는 p99 1.494 ms였다. 일반 이질
small state에서는 exact-feasible 504개 중 10개를 보수적으로 거절했지만 false-safe는0이었다.

### N3. AI-RAN specialization of failure-contingent same-TB recovery debt

Optional NeuralRx가 unresolved인 동안 같은 TB의 conventional receiver는 아직 queue에
도착하지 않았어도 미래 mandatory obligation이다. SoftWall은 여러 RAN home의 obligation을
하나의 executable all-fail schedule로 유지한다. All-fail schedule에서 성공한 NRx의
obligation을 삭제하면 모든 failure subset도 가능하다는 deletion-monotonicity가 핵심이다.

Shared recovery GPU에서는 각 home이 따로 안전하다는 사실로 충분하지 않다. C17 모델의
`3 debt + 2 debt`, `c=25 ms`, `D=100 ms` 반례는 local calendars가 각각 가능하지만 합집합
125 ms는 불가능함을 보인다. SoftWall은 이 상태를 global certificate로 거절한다.

## 2. 가장 가까운 선행 연구와 정확한 차이

| 선행 연구 | 이미 해결한 것 | SoftWall이 중복 주장하지 않는 것 | 남는 차이 |
|---|---|---|---|
| YinYangRAN, INFOCOM'24 | GPU PHY+ML, MPS 자원 비율, CPU fallback | AI-RAN GPU sharing, fallback, RAN 우선 보호 | 동일 TB의 unresolved recovery debt와 executable all-fail calendar 없음 |
| CloudRIC, MobiCom'24 | Queue/deadline 기반 accelerator pool 배치 | 요청 단위 endpoint 선택, 실행시간 예측 | Optional NRx 결과에 따라 생멸하는 same-TB mandatory debt와 AI lease의 원자 교환 없음 |
| Concordia, SIGCOMM'21 | vRAN CPU 예약과 남는 자원 회수 | slack reclamation, mandatory-first 사고 | Non-preemptive GPU fence/transport lifecycle과 outcome-contingent recovery 없음 |
| DARIS, DAC'25 | MPS·stream·stage 기반 priority DNN admission | MPS+bounded work unit, 높은 우선순위 보호 | PHY semantic recovery와 all-fail certificate가 없음 |
| ARCHES, 2026 preprint | Channel별 neural/conventional PHY expert 선택 | NeuralRx/conventional 선택 | 선택 실패 뒤 같은 TB recovery schedule과 외부 AI lease가 없음 |
| OCUDU, 2026 preview | In-DU dApp timing class, conventional path, result/lifecycle validation | Inline AI timing contract와 fallback 자체 | 여러 TB/home의 all-fail recovery calendar와 external-AI compute lease 교환은 공개 범위에서 없음 |
| Interplay/CAORA | SAC·forecasting 기반 MIG AI/RAN orchestration | Dynamic AI-and-RAN allocation, RAN-priority fulfillment | Workload/MIG 추상화이며 same-TB recovery debt와 physical transaction 없음 |
| HAF, 2026 preprint | Slow placement+fast deadline-aware GPU/CPU allocation | 계층형 시간 규모와 deadline resource allocation | Simulation-level allocation이며 cuPHY/NRx recovery certificate가 없음 |
| SMEC, NSDI'26 | RAN과 MEC의 decoupled SLO scheduling, edge MPS priority | 5G+MPS+deadline-aware scheduling | PHY와 외부 AI가 같은 GPU에서 만드는 same-TB recovery obligation을 다루지 않음 |
| nvtaskset/GCAPS/REEF/XSched | GPU isolation, context scheduling, preemption | GPU scheduling primitive나 context overhead 최초성 | Radio outcome-conditioned obligation과 RAN commit semantics가 없음 |

이 표에서 한 칸의 부재만으로 novelty가 성립하는 것은 아니다. SoftWall의 주장은 N1–N3를
같은 구현과 물리 evidence chain으로 연결한 데 한정한다.

## 3. “기존 요소의 단순 결합”이 아닌 이유

단순 결합이면 각 부품을 이어도 안전 의미가 보존되어야 한다. 실제로는 다음 compositional
failure가 발생했다.

| 반례 | 단순 결합이 실패한 이유 | SoftWall의 수정 |
|---|---|---|
| Local-safe/global-unsafe | 두 local calendar가 한 shared recovery lane의 합집합을 보지 못함 | Global obligation set과 shared-lane certificate |
| Certificate/dispatch gap | 계산 시각의 slack이 physical launch 전 host tail 동안 사라짐 | 5 ms freshness, 재계산, worker latest-start |
| Sequential outcome replay | 같은 cutoff의 성공을 순차 적용해 허구의 infeasible partial state 생성 | Common-cutoff success batch 원자 적용 |
| Fixed lease | AI service만 예약하고 launch control이 recovery blackout을 침범 | Launch-time `control+AI` blackout 재검증 |
| Broker reply loss | 적용 여부가 모호한 token을 재시도하면 duplicate execution 가능 | Generation/ownership quarantine와 fail-closed |
| V14 two-prepare | 일반 at-most-once RPC만으로 client가 모르는 broker-held token 발생 | Home별 single unlaunched token invariant |
| Bound reuse across lifecycle | Warm bound를 GC/restart/cold에 상속하면 false-safe | Provenance fingerprint와 QSU/QSN/MI/UQ |

이 반례들은 N1–N3 사이의 인터페이스가 연구 문제임을 보여 준다. 각 수정은 unit/model
검사뿐 아니라 실제 TensorRT NeuralRx, cuPHY recovery, Qwen, CUDA IPC/P2P와 MPS 경로에서
검증됐다.

## 4. Claim–evidence 연결

| 원고 claim | 필수 증거 | 현재 권위 evidence | 허용 범위 |
|---|---|---|---|
| All-fail certificate가 failure subsets를 지배 | Lemma와 exact audit | Lemma 1/1c, V17 3,400 state + local-safe/global-unsafe10 | 모델 가정 안 |
| Atomic transaction이 실제 경로에서 동작 | Actual NRx·recovery·AI·fence·commit | C159-Q2 actual NRx4,800, recovery1,312, Qwen1,077, commit4,800, violation0 | Warm P180/D155 A100 |
| Fault가 있어도 qualified mode safety 유지 | 각 fault point와 post-fault continuation | C161 A0–A6, actual NRx2,800, recovery942, commit2,800, miss0 | 명시 fault model |
| 모델이 feasibility boundary를 예측 | Prespecified adjacent physical cases | C162 180/180 match, 88/89 ms 각30 | Qualified provenance |
| Current-idle admission이 future debt를 보호하지 못함 | Mandatory-safe이지만 AI-first는 false-safe인 상태 | C162 E4/E6b contract 초과12/1ms, SoftWall 60/60 launch 전 거절 | Contract witness; observed debt-blind miss 아님 |
| Scheduler가 small-state exact semantics를 보존 | Exact comparison+independent verifier | 16,023 mismatch0; debt64 p99 1.494 ms; verifier failure0 | 현재 mode; 일반 optimality 없음 |
| Lifecycle provenance가 bound reuse를 막음 | Token model+representative physical subsets | C164 qualified/partial5, UQ5 | Universal lifecycle 보장 아님 |

Zero violation은 finite-sample qualification이다. WCET, probability-zero miss 또는 production
HARQ guarantee로 표현하지 않는다.

## 5. 예상 reviewer objection과 답변

### “Fallback을 미리 예약한 것뿐이다.”

한 request의 fallback slot 하나가 아니다. 여러 home의 아직 queue에 없는 contingent jobs,
shared recovery capacity, endpoint/transport credit과 AI blackout을 함께 스케줄한다. Local
reservation이 false-safe가 되는 반례와 exact global certificate가 있다.

### “MPS scheduling paper다.”

MPS는 concurrency mechanism이다. C26/C46은 cap과 lifecycle만으로 deadline contract가
생기지 않음을 보인다. SoftWall의 안전 조건은 MPS share가 아니라 executable recovery
certificate, qualified whole-path bound와 physical fence다.

### “EDF/knapsack/exact solver가 새롭지 않다.”

맞다. Exact solver는 small-state oracle/certificate generator이고 정책 novelty가 아니다.
C78/C79/C102와 C159-Q3에서 optimizer/throughput 우위를 기각했다. 새 상태 표현과
certificate-to-execution refinement가 기여다.

### “처리량 이득이 없는데 왜 필요한가.”

현재 trace에서는 certificate-preserving recovery-first Event-driven과 SoftWall의 timely token
value가 385,262로 같았다. 이 결과는 AI-first retiming의 추가 처리량을 기각하며 certificate
자체를 기각하지 않는다. Recovery-first는 같은 초기 all-fail admission을 사용하고 recovery를
먼저 drain하는 안전한 보수 정책이다.

Certificate의 필요성은 별도 비교로 보인다. C162 E4/E6b에서 mandatory-only는 feasible하지만
current-idle debt-blind AI admission은 선언 bound 안에서 radio guard를 12/1ms 넘길 수 있다.
SoftWall은 두 상태를 합계60/60회 launch 전에 거절했다. 이는 observed debt-blind miss가 아닌
contract-level witness다. 따라서 정확한 주장은 “어떤 공유에도 SoftWall이 필요하다”가 아니라
“unresolved recovery보다 AI를 먼저 실행하면서 contract를 유지하려면 executable witness가
필요하다”다. 자세한 판정은
[필요성 반론 감사](../archive/SOFTWALL_NECESSITY_GAP_DECISION_KO.md)에 있다.

### “Synthetic D155라 실제 RAN 보장이 아니다.”

맞다. 원고는 production deadline을 주장하지 않는다. D155는 reproducible contract이고,
production `d_MAC` ingestion/validator는 준비됐지만 trace가 없다. 이 한계는 external validity에
영향을 주며, N1–N3의 내부 타당성을 무효화하지는 않는다. 여기서 `P180`은 harness의
inter-release period이지 NR slot 길이가 아니다. 따라서 0.5/1 ms slot을 `P`에 직접 대입한
배수 계산은 현재 실험의 production feasibility 판정이 아니다. C163의 raw-event builder,
clock/provenance 검사, request-specific exact/certified bridge와 expiry validator는 단위시험
24개를 통과했다. 실제 trace가 들어오면 같은 contract 의미론으로 request별 expiry를 판정할
수 있지만, 새 timing vector와 mode의 물리 재자격은 여전히 필요하다.

### “외부 채널에서 NeuralRx가 동작하지 않았다.”

맞다. Aerial TDL-A와 1×4 기하 정정 canary는 각각 conventional 10/10, NeuralRx 0/10이었다.
따라서 외부 채널의 NeuralRx decoding gain과 PHY 일반화는 주장하지 않는다. 현재 증거가
입증하는 것은 qualified synthetic input에서 TensorRT NeuralRx의 실제 실행, CRC 기반 상태전이,
recovery와 Qwen의 물리 lifecycle이다. Tensor/channel/DMRS 호환성 문제가 해결되면 외부
channel mode가 같은 service-vector/envelope qualification을 다시 통과해야 한다.

### “강한 기준선보다 처리량 이득이 없다.”

맞다. 이 결과 때문에 optimizer와 throughput superiority를 기여에서 제외했다. 실무적 가치는
실행 전에 QSU/QSN/MI/UQ를 분류하고, 실제 launch 시 certificate를 깨는 작업을 차단하며,
mode 변경이 재자격을 요구하는 시점을 알려 주는 데 있다. 이 프레이밍이 설득력을 얻으려면
원고가 안전성과 예측 가능성을 중심에 두고 385,262-token 동률을 숨기지 않아야 한다.

### “왜 lifecycle 10/10이 아닌가.”

논문 claim은 warm mode와 네 대표 subset에 한정된다. C164는 claim-scoped complete이며 나머지
다섯 mode를 UQ로 표시한다. Process replacement를 포함한 범용 crash recovery는 다른 fault
model과 mechanism을 요구하므로 future work다.

## 6. Novelty 판정

공개된 정식 학회 논문과 2026-09-25까지의 직접 preprint/preview를 검토한 범위에서는
N1–N3의 정확한 결합을 확인하지 못했다. 이 판정은 유사 연구의 절대 부재 증명이 아니다.

현재 novelty는 다음 조건에서 방어 가능하다.

1. 제목·초록·Introduction에서 GPU sharing이나 MPS 자체를 주 기여로 쓰지 않는다.
2. Optimizer와 처리량 우위가 기각됐음을 명시한다.
3. All-fail debt, atomic physical lifecycle, predictive envelope를 하나의 causal chain으로 쓴다.
4. Synthetic finite-sample/A100/fault/lifecycle 범위를 모든 정리와 평가 문장에 유지한다.
5. 기존 내부 DART calendar와 SoftWall의 추가 multi-home/shared-recovery refinement를 구분한다.

이 조건을 지키면 novelty의 중심은 분명하다. 가장 큰 제출 위험은 아이디어 중복보다
production timing 부재와 많은 refinement history를 한 편의 명확한 이야기로 압축하는 일이다.
