# SoftWall 필요성 반론 감사와 추가 실험 결정

**상태:** 2026-09-25, C162 사전 고정 경계의 사후 contract 감사 완료  
**결론:** 필요성 반론은 원고에서 반드시 다루되, `recovery-first`를 깨는 correlation sweep은
실행하지 않는다. 비교군 의미를 바로잡고 이미 확보한 C162/C153/C26/C46 증거를 하나의
necessity chain으로 연결한다.

## 1. 지적에서 맞는 부분

C159-Q3에서 SoftWall과 recovery-first는 적시 token value가 `385,262 = 385,262`로 같다.
따라서 원고가 이 결과만 보여 주면 리뷰어는 “더 단순한 기준선과 결과가 같은데 이
substrate가 왜 필요한가?”라고 물을 수 있다. 이 질문을 처리하지 않고 조합 novelty만
강조하면 필요성 논증이 약하다.

또한 conditional recovery debt 자체는 primary/backup scheduling과 가까우므로 새로운
일반 스케줄링 이론으로 전면에 두지 않는다. 가장 강한 기여 순서는 다음으로 바꾼다.

1. 측정된 일곱 반례에서 도출한 certificate-to-physical-execution refinement
2. provenance가 포함된 QSU/QSN/MI/UQ predictive envelope
3. 위 두 기여를 표현하는 AI-RAN 특화 conditional-recovery model

## 2. 제안된 비교에서 잘못된 부분

C159의 recovery-first는 certificate가 없는 정책이 아니다. 이 기준선은 SoftWall과 같은
초기 all-fail radio admission을 사용하고, 미해결 recovery를 물리적으로 모두 처리한 뒤
AI를 실행한다. 따라서 정확한 이름은 **certificate-preserving recovery-first**다.

이 기준선은 보수적이지만 구조적으로 안전하다. failure correlation을 높여 이 기준선을
깨뜨리려 하면 다음 둘 중 하나가 된다.

- recovery-first의 정의를 바꿔 AI를 recovery보다 먼저 실행한다.
- initial all-fail admission을 제거해 안전 비교군을 unsafe diagnostic으로 바꾼다.

어느 쪽도 기존 strong baseline과 같은 정책이 아니다. 따라서 `385,262 = 385,262`가
기각한 것은 **certificate의 필요성**이 아니라 **AI-first atomic retiming의 추가 처리량
이득**이다.

## 3. 실제로 필요한 비교

세 정책을 분리한다.

| 정책 | all-fail 검사 | unresolved debt보다 AI를 먼저 실행 | 논문에서의 역할 |
|---|---:|---:|---|
| Debt-blind current-idle | 없음 | 가능 | unsafe diagnostic |
| Certificate-preserving recovery-first | 있음 | 안 함 | 강한 안전·성능 기준선 |
| SoftWall | 있음 | certificate가 있을 때만 | 제안 substrate |

SoftWall의 필요성 주장은 다음과 같이 한정한다.

> **현재 GPU가 비었다는 이유만으로 unresolved same-TB recovery보다 AI를 먼저 실행하면서
> radio contract를 보장할 수는 없다. AI-first sharing을 하려면 executable recovery witness가
> 필요하다.**

“어떤 shared-GPU 실행에도 SoftWall이 필요하다” 또는 “recovery-first보다 처리량이 높다”는
주장은 하지 않는다.

## 4. C162에 이미 존재하는 necessity witness

C162에서 아래 점들은 물리 실행 전에 고정됐다. 이번 분석은 새 outcome을 연 것이 아니라
그 사전 고정 점들의 **사후 contract 해석**이다.

| C162 점 | 상태 | Debt-blind가 AI를 넣은 bound-respecting 완료 | Radio guard | 초과 | SoftWall 물리 판정 |
|---|---|---:|---:|---:|---:|
| E4, debt 2, context 256, decision 45 ms | QSN | 165 ms | 153 ms | 12 ms | 30/30 launch 전 거절 |
| E6b, debt 1, context 64, decision 89 ms | QSN | 154 ms | 153 ms | 1 ms | 30/30 launch 전 거절 |

두 점 모두 mandatory-only all-fail schedule은 feasible이다. 즉 MI나 과부하를 AI 실패로
위장한 사례가 아니다. 차이는 unresolved recovery를 포함해 AI transaction을 검사했는지다.

이 결과는 C162에서 debt-blind deadline miss를 **관측했다는 뜻이 아니다.** SoftWall이
unsafe launch를 막았기 때문에 해당 Qwen kernel은 실행되지 않았다. 정확한 주장은 선언된
상한 안에 radio guard를 넘는 허용 실행이 존재하므로 debt-blind admission이 그 contract를
보장할 수 없다는 것이다.

기계 감사 결과는
[softwall_necessity_witness_v2.json](../../results/softwall_multigpu/softwall_necessity_witness_v2.json)에
있다. 두 사전 고정 false-safe 상태, 60회 물리 거절, 최소/최대 contract 초과 1/12 ms가
확인됐다.

### 4.1 선언 bound 민감도

이 witness는 자격화된 `B_conv=25 ms` 계약에 대한 판정이다. 다른 항을 그대로 두고
`B_conv`만 변화시키면 경계는 정확히 다음과 같다.

| `B_conv` 구간 | E4 | E6b |
|---|---|---|
| `B_conv > 24 ms` | witness 유지 | witness 유지 |
| `19 ms < B_conv <= 24 ms` | witness 유지 | 사라짐 |
| `B_conv <= 19 ms` | 사라짐 | 사라짐 |

E6b는 1 ms 경계 사례이므로 `24 ms` 이하에서 사라지지만, E4는 `19 ms`까지 낮춰야
사라진다. 현재 권위 Q2 전체 recovery path의 표본 최대는 `13.465105 ms`, 중앙값은
`3.40418 ms`다. 따라서 `25 ms`는 현재 표본 최대의 `1.8567×`이며, 과거 약 `3.1 ms`를
최대값으로 두고 `8×`라 부르는 것은 현재 mode에 맞지 않는다. C162 boundary 자체의 최대는
`8.950475 ms`로 비율은 `2.7931×`다.

표본 최대를 새 service bound로 바로 바꿀 수는 없다. `B_conv < 25 ms`는 complete calendar,
workload, placement, lifecycle을 다시 자격화하기 전까지 모델이 명시적으로 `UQ`로 분류하는
counterfactual이다. 만약 `13.465105 ms`를 가상 bound로 승격하면 이 두 고정점은 모두
witness가 아니게 된다. 다만 양의 recovery charge가 있는 한 Proposition 2의 false-safe
구간은 없어지는 것이 아니라 더 늦은 decision time으로 이동한다. 그 새 경계는 별도의
사전 고정 물리 검증이 필요하다. 민감도는 witness가 선언 bound에 얼마나 의존하는지 공개하며,
더 작은 bound의 자격 검증을 대신하지 않는다.

## 5. 필요성 증거의 세 층

| 층 | 증거 | 무엇을 입증하는가 |
|---|---|---|
| MPS 자체 | C26 cap100 2/1,500, cap20 1/1,500 miss; C46 GPU/MPS 12/10,000 대 CPU 0/10,000 | MPS cap/lifecycle만으로 deadline contract를 만들 수 없음 |
| Global obligation | C153 local-safe/global-unsafe에서 local-only fifth debt를 global coordinator가 launch 전 거절 | 공유 recovery lane에는 global debt 집합이 필요함 |
| AI-first admission | C162 E4/E6b의 60/60 QSN reject와 1/12 ms bound-respecting 초과 | current idleness만으로 early AI를 허가할 수 없음 |

C161의 correlated failure campaign은 certificate를 사용하는 물리 경로가 실제 동시 실패를
처리한다는 양의 증거다. 위 세 층과 합치면 “MPS가 실패한다”에서 끝나지 않고 “어떤 상태를
표현하고 어떤 admission을 막아야 하는가”까지 연결된다.

## 6. Failure-correlation sweep을 하지 않는 이유

All-fail certificate는 실패 확률 분포에 의존하지 않는 계약이다. 허용된 `rho=1` 분기 하나가
있으면 current-idle 정책의 보장 불가능성을 보이기에 충분하다. `rho` sweep은 expected miss
rate나 expected AI utility를 평가할 때 의미가 있지만, 현재 논문은 그 확률 모델이나 처리량
우위를 주장하지 않는다.

현재 qualified mode의 관측 실행시간은 선언 bound보다 짧다. 따라서 debt-blind kernel을
실제로 실행해도 finite sample에서는 deadline miss가 없을 가능성이 크다. miss를 만들기 위해
deadline을 사후 축소하거나 unqualified AI class를 넣거나 sleep을 주입하면, “실제 mode에서
필요하다”가 아니라 인위적 실패를 만든다. 이는 현재의 contract witness보다 약한 증거다.

새 물리 실험은 원고 검토에서 다음 두 조건을 모두 만족하는 경우에만 연다.

1. 리뷰어 반론을 처리하려면 observed unsafe launch가 반드시 필요하다.
2. 기존 qualified work class와 사전 고정 deadline 안에서, outcome을 보기 전에 계산한
   empirical power analysis가 material miss 확률을 예측한다.

현재는 두 번째 조건이 없다.

## 7. 원고 반영 결정

- Failure-derived refinement chain을 Model 앞에 배치한다.
- 기여 순서를 refinement → envelope → conditional model → physical evaluation으로 바꾼다.
- `current idleness is insufficient` proposition과 짧은 증명을 추가한다.
- C162 E4/E6b 60회 necessity witness를 별도 evaluation subsection으로 추가한다.
- recovery-first를 항상 `certificate-preserving recovery-first`로 부른다.
- `385,262 = 385,262`는 certificate 기각이 아니라 AI-first throughput claim 기각으로 쓴다.
- correlation sweep을 새 필수 실험으로 만들지 않는다.

이 결정은 실험을 덜 하기 위한 편의가 아니다. 안전 기준선과 unsafe diagnostic을 섞지 않고,
이미 사전 고정한 경계가 실제로 입증하는 범위만 사용하는 선택이다.
