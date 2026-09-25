# SoftWall 최종 실험 계획: shared recovery certificate에서 AI-and-RAN system claim까지

**상태:** 2026-09-25 C160--C162 PASS, C163 readiness와 C164 idle30/MPS-restart/Qwen-reload subset 반영  
**현재 출발점:** V17 global model PASS, C159-Q1/Q2 P180 actual-NRx qualification PASS, Q3 performance holdout 중단, C160 fault-state PASS, C161 A0--A6 physical fault PASS on qualified nodes, C162 16,023-state exact agreement·180-round physical boundary·64-debt certified scheduling PASS, C163 production trace UQ, C164 idle30·quiescent MPS-restart·Qwen-reload mandatory continuity two-node physical PASS  
**목적:** 설계의 새로움을 구성요소 목록이 아니라 반증 가능한 시스템 가설과 물리 증거로 고정한다.

---

## 1. 최종 논문 주장을 먼저 고정한다

SoftWall의 주장은 새로운 GPU scheduler나 joint optimizer의 우월성이 아니다. 최종 주장은
다음 세 문장으로 제한한다.

1. Optional per-TB NeuralRx는 결과가 알려지기 전까지 conventional recovery라는
   **조건부 mandatory debt**를 만든다. 여러 RAN home이 recovery GPU를 공유하면
   home-local 안전 판정의 합성은 false-safe가 될 수 있다.
2. SoftWall은 모든 home의 debt를 한 executable all-fail calendar로 유지하고, recovery
   retiming과 bounded external-AI lease를 generation-safe 원자 transaction으로 확정한다.
   GPU/IPC/P2P 자원은 물리 completion fence 전에는 반환하지 않는다.
3. 이 계약은 MIG를 사용하지 않는 MPS GPU에서 실제 cuPHY conventional recovery와 Qwen
   inference를 함께 운용할 수 있는 safe/useful/infeasible envelope를 만든다.

세 번째 문장의 `useful`은 “외부 AI completion이 존재하는 qualified-safe-useful 영역”이라는
envelope 의미로 제한한다. 강한 기준선 대비 추가 처리량 우위 H5는 Q3에서 기각됐다. 첫째와
둘째 문장의 correctness substrate 및 feasibility-envelope 결과는 유지한다.

## 2. 가설과 필요한 증거

| ID | 반증 가능한 가설 | 현재 근거 | 닫아야 할 실험 |
|---|---|---|---|
| H1 | 각 home에서 feasible인 debt 집합도 shared recovery lane의 합집합에서는 infeasible일 수 있다. | 25-state grid의 local-safe/global-unsafe 10개, 3,400-state oracle mismatch 0; C153 이후 global fifth-debt physical reject | qualified-node 통합 canary 완료 |
| H2 | Global certificate가 직접 발급한 순서로 두 home의 cuPHY recovery를 실행하고도 모든 radio commit을 expiry 안에 끝낼 수 있다. | C161 full: actual NRx2,800·recovery942·commit2,800·miss0; C162 recovery330·commit720·miss0 | qualified-node 완료 |
| H3 | NRx success로 debt가 해소될 때만 Qwen lease가 열리고, lease commit·GPU fence·credit retire가 recovery calendar와 원자적으로 일치한다. | C161 A3 matching-fence retire2, A4 non-launch52, A5 fenceless retain2, post-fault continuation260 | qualified-node 완료 |
| H4 | Certificate가 예측한 QSU/QSN/MI 경계가 holdout 물리 mode에서 false-safe 없이 재현된다. | C162 exact mismatch0/16,023; retrospective1,200/1,200; 새 두-node physical boundary180/180; 64-debt decision p99 1.494ms | qualified warm envelope와 finite CPU scalability 완료; production mode UQ |
| H5 | 동일한 radio policy와 bound에서 SoftWall이 가장 강한 safe online baseline보다 적시 AI token value를 실용적으로 늘린다. | Q3 exact calibration oracle: 실측 recovery 기준 385,262 대 385,262, **0.000%**; contract sensitivity 0.066% | **현재 P180/1초-SLO mode에서 기각, confirmatory holdout 중단** |
| H6 | Broker/worker/response fault 뒤에도 불명확한 credit을 재사용하지 않고 RAN을 계속 처리한다. | C160 state51; C161 A0--A6, event pair40·terminal fault4·continuation260·miss0 | qualified-node 완료; durable restart UQ |

H1--H4와 H6은 안전·모델링 기여다. H5만 성능 기여다. H5 실패를 H1--H4 성공과 섞어
전체 연구 실패 또는 optimizer 성공으로 재해석하지 않는다.

## 3. 최종 물리 mode

첫 통합 mode는 한 4×A100 NVLink node에서 다음 배치를 사용한다.

| GPU | 역할 |
|---|---|
| GPU0 | RAN home 0의 PUSCH owner와 radio commit |
| GPU1 | RAN home 1의 PUSCH owner와 radio commit |
| GPU2 | 두 home이 공유하는 persistent cuPHY conventional worker + MPS Qwen worker |
| GPU3 | 실제 optional NeuralRx endpoint; 초기 mechanism canary에서는 결과 주입만 수행 가능 |

대표 context-64 timing contract는 C159-Q1에서 두 node 자격을 얻은 `P=180 ms`, `D=155 ms`,
`B_NRx=45 ms`, `B_conv_path=25 ms`, `B_AI(context64)=35 ms`, `guard=2 ms`다.
다음 식을 만족한다.

```text
all-fail four-debt path: 45 + 4×25 + 2 = 147 ms <= 155 ms
five-debt path:          45 + 5×25 + 2 = 172 ms > 155 ms
two-success AI path:     45 + 2×25 + 35 + 2 = 132 ms <= 155 ms
```

따라서 두 home의 3+2 debt는 각 local calendar에서는 가능하지만 global union에서는 다섯
번째 debt를 거절해야 한다. 네 debt 중 두 NRx가 성공하면 두 recovery credit이 사라지고
context-64 Qwen lease가 열린다. 이 세 상태가 한 mode 안에서 H1--H3을 모두 노출한다.

새 mode의 service bound는 과거 mode에서 자동 상속하지 않는다. 동일 placement, MPS cap,
worker lifecycle, trace class에서 다시 자격화하며 관측 최대를 WCET로 부르지 않는다.

## 4. 비교군

모든 안전 비교군에 같은 PUSCH 입력, max-radio rule, NRx endpoint, service profile, MPS cap,
Qwen worker, request arrival/deadline과 single-commit rule을 준다.

| 비교군 | 안전 규칙 | 목적 |
|---|---|---|
| Uncontrolled MPS | certificate 없이 보이는 idle에 Qwen 실행 | MPS-only 실패 진단; 안전 성능 비교군이 아님 |
| Local-only certificate | home별 calendar만 검사 | shared resource에서의 false-safe 반례 진단; 안전 비교군이 아님 |
| Static global-safe | global all-fail recovery 구간을 고정하고 남은 구간만 AI에 사용 | 안전하지만 보수적인 하한 |
| Event-driven global-safe | NRx 결과 뒤 credit은 회수하되 기존 recovery 순서를 유지하고 현재 contiguous gap만 사용 | 강한 work-conserving online baseline |
| SoftWall | global replan, multi-credit retiming, AI lease와 generation/fence를 한 transaction으로 처리 | 제안 substrate |
| Offline oracle | 미래 NRx 결과와 AI arrival을 알고 최대로 배치 | 개선 가능성의 상한; 실현 가능한 비교군이 아님 |

`Local-only`와 `Uncontrolled MPS`의 deadline miss를 이용해 SoftWall의 처리량 우위를
부풀리지 않는다. 성능 주 비교는 `Static global-safe`, `Event-driven global-safe`,
`SoftWall` 세 시스템이다.

## 5. 단계별 실행 계획

### C153 — V17 통합 falsification canary — 완료

목적은 처리량이 아니라 model과 물리 path의 원자 결합이다.

1. 같은 release에 home0의 debt 3개와 home1의 debt 2개를 제출한다.
2. Local-only 판정은 다섯 개를 모두 feasible로 표시해야 한다.
3. Global coordinator는 네 개만 commit하고 다섯 번째를 GPU launch 전에 거절해야 한다.
4. 네 accepted request 중 두 NRx success를 관측해 debt 두 개를 release한다.
5. 같은 generation update에서 recovery replan과 context-64 Qwen lease를 commit한다.
6. Qwen CUDA fence 뒤 lease를 retire하고 남은 두 conventional recovery를 certificate 순서로
   shared cuPHY worker에서 실행한다.
7. 두 decoded TB/CRC를 원 home에 돌려주고 한 TB당 결과 한 개만 expiry 전에 commit한다.

필수 gate는 다음과 같다.

- local/global decision divergence가 정확히 한 번 이상 존재
- global-rejected request의 CUDA/IPC launch 0
- worker 실행 순서가 certificate `(start, deadline, generation)`과 일치
- Qwen launch generation과 committed lease generation 일치
- CUDA fence 전 retire 0, stale/duplicate retire의 state mutation 0
- accepted radio의 deadline·declared bound·single-commit·credit 위반 0
- 실제 conventional payload/CRC 오류 0

C153 job `58852924`/`nid001348`은 10/10 gate를 통과했다. Qwen host/GPU는
23.059/21.733 ms, 두 home release-to-commit은 82.412/97.644 ms였고 deadline miss는
0이었다. 앞선 socket-path 실패와 잘못된 release 정의의 D155 실패도 보존했다. C153은
development canary이므로 QSU가 아니며 source와 bound를 동결한 holdout으로 넘어간다.

### C154/C155 — 두 독립 node의 integrated holdout — controlled-outcome 완료

C153에서 고친 source, seed 범위, mode fingerprint와 analyzer를 먼저 hash로 고정한다.
기존 C149--C152 node를 제외한 서로 다른 두 A100 node에서 다음 arm을 실행한다.

| Arm | 결과 패턴 | 반드시 관측할 분기 |
|---|---|---|
| all-fail | 네 accepted NRx를 모두 fail/late로 처리 | AI 거절, 네 shared conventional recovery |
| conditional-open | 두 success + 두 fail | debt release 뒤 Qwen lease와 두 recovery |
| all-success | 네 credit을 release | 최대 conditional slack, recovery launch 0 |
| overload | local 3+2 request | fifth global rejection, 기존 네 request 정상 완료 |

각 node에서 독립 seed와 반대 arm 순서를 사용했다. C154/C155의 8개 arm은 제출36,
수락32, global reject4, success12, Qwen4, physical recovery/home commit20/20, deadline miss0으로
모든 gate를 통과했다. 결과는 controlled-outcome `FINITE_SAMPLE_INTEGRATED_PASS`로만
표현한다. C154/C155 자체에는 actual NeuralRx-driven claim이 없으며, 그 후속 gate는
C157A/C157B에서 별도로 통과했다.

### C156 — GPU timeline conformance — 완료

작은 frozen arm을 Nsight Systems로 계측한다. Host event뿐 아니라 GPU kernel interval로
다음을 확인한다.

- Qwen은 committed lease interval에서만 실행됨
- full-GPU-blackout 계약에서는 Qwen과 shared conventional kernel의 금지 overlap이 0
- recovery worker의 kernel 순서가 certificate 순서와 일치
- CUDA IPC/P2P forward, cuPHY, P2P backward, home commit의 인과 순서가 유지됨

Profiler 표본은 service bound로 사용하지 않는다. 이는 실행 semantics 증거다.

Job `58853926`/`nid001085`의 첫 시도는 Nsight 13.2 clock schema 분석 오류로 판정 전
제외했고, 두 번째 시도는 기존 `[45,80] ms` 고정 lease를 profiler 지연 아래 3.153 ms
넘어 실제 gate가 실패했다. 이 결과로 launch control이 lease 밖에 있던 공백을 찾았다.
V17.1은 실제 decision 시각에서 `control5+AI35` blackout과 recovery를 함께 재계산한다.
세 번째 시도는 revalidation 1.745 ms, dispatch 1.768 ms, Qwen kernel1,224,
recovery kernel106, NVTX range8, 미귀속 event0, 금지 kernel overlap0 ns, correct commit2/2,
D155 miss0으로 11개 gate를 모두 통과했다. 상세는
[C156 결과](SOFTWALL_CONFIRM156_GPU_TIMELINE_RESULT_KO.md)에 있다. Attempt3 자체는
controlled-outcome 한 node의 profiler semantics이며 service bound나 actual-NRx 자격은 아니다.

C156b job `58854401`/`nid001064`는 첫 passing node를 제외하고 새 seed로 같은 source와
gate를 재실행해 11/11을 통과했다. 두 node 합계 Qwen kernel2,448, recovery kernel212,
미귀속 event0, forbidden overlap0 ns, correct commit4/4, deadline miss0이다. 따라서 V17.1
controlled GPU semantics는 same-family two-node finite-sample PASS다.

### C157A/C157B — actual NeuralRx outcome integration — 완료

GPU0/1 owner의 같은 noisy TB를 GPU3 persistent TensorRT NeuralRx와 GPU2 shared cuPHY
recovery에 함께 연결했다. Outcome은 branch injection이 아니라 45 ms cutoff 전 실제 NRx
CRC다. `+20 dB` 두 key와 `-15 dB` 두 key, seed와 기대 분기를 결과를 열기 전에 고정했다.

C157A job `58854900`/node `nid001109`의 최종 development source는 12/12 gate를 통과했다.
C157B는 그 source를 바꾸지 않고 development node를 제외한 job `58855194`/node
`nid001085`, 새 seed에서 12/12를 다시 통과했다. 두 node 합계 debt 제출10, 수락8,
global reject2, actual NRx success4, shared recovery4, same-input conventional-oracle equivalent4,
Qwen2, radio single commit8, miss0이다. 최대 NRx release-to-complete는 37.660 ms, 최대
radio commit은 107.674 ms였다. 상세는
[C157 결과](SOFTWALL_CONFIRM157_ACTUAL_NRX_RESULT_KO.md)에 있다.

자격 범위는 warm synthetic actual-NRx transition이다. Monitor의 첫 CuPy JIT가 45 ms
cutoff를 넘긴 attempt를 포함해 여섯 development failure를 보존했다. WCET, cold/long-idle,
production `d_MAC`, integrated fault와 throughput 우위는 이 결과에 포함하지 않는다.

### C158 — 반복 actual-NeuralRx 자격 — 완료

C158은 C157의 단일 transition을 장기 persistent mode로 확장했다. 두 독립 A100 node에서
각 250 epoch를 실행해 actual TensorRT NeuralRx request 2,000건, 제때 success 1,440건,
shared cuPHY recovery 560건, Qwen unit 500건과 radio single commit 2,000건을 얻었다.
Deadline·declared bound·credit·input round-trip·response echo·recovery contract 위반은 모두
0이었다. 결합 p99/max는 NRx 8.439/26.385 ms, recovery 4.468/17.763 ms,
Qwen 29.871/32.948 ms, radio commit 98.840/107.941 ms였다.

이 실험에서 deadline control metadata를 `/pscratch`의 per-epoch JSON으로 전달할 수 없다는
것, host doorbell만으로 P2P response consumption을 증명할 수 없다는 것, safety monitor의
첫 CuPy JIT도 lifecycle contract에 포함해야 한다는 것을 확인했다. 최종 구현은 persistent
mmap control page, input round-trip, response read-back echo와 모든 P2P/monitor preflight를
사용한다. 상세는 [C158 결과](SOFTWALL_CONFIRM158_REPEATED_QUALIFICATION_KO.md)에 있다.

C158의 주기는 `P=600 ms`이고 timed radio expiry는 `D=155 ms`다. 600 ms는 synthetic input과
validation-only local oracle를 다음 release 전에 staging하기 위한 qualification 간격이다.
따라서 C158 표본은 C159 성능 CI에 합치지 않고, 임의 trace arrival 또는 `P180` 처리량을
증명한다고 해석하지 않는다.

### C159-Q — trace mode 자격과 실험 크기 고정

C159의 성능 arm을 열기 전에 별도 qualification run으로 다음 차이를 닫는다.
상세 workload split, arm order와 통계 단위는
[C159 사전 명세](SOFTWALL_CONFIRM159_TRACE_PROTOCOL_KO.md)에 고정한다.

**Q1 완료:** 모든 noisy PUSCH input과 validation-only oracle를 readiness 전에 seed-indexed
GPU bank로 만들고, D155 뒤 다음 P180 release까지의 25 ms에 stable IPC buffer만 갱신했다.
두 독립 node에서 actual NRx2,000, recovery582, Qwen500, commit2,000, miss·bound·fence·
contract 위반0으로 통과했다. Whole-path cold preflight 최대 64.750 ms와 이후 warm recovery
최대 11.555 ms를 분리해 warm lifecycle을 명시했다. 상세는
[C159-Q1 결과](SOFTWALL_CONFIRM159_Q1_P180_RESULT_KO.md)에 있다.

**Q2 완료:** 16/32/64/128/256/512 token을 35/35/35/40/65/75 ms로 고정했다.
두 독립 node의 1,200 epoch에서 actual NRx4,800, recovery1,312, Qwen1,077, commit4,800,
위반0으로 16/16 gate를 통과했다. 256/512의 certificate reject55/68은 긴 class가 unresolved
recovery와 함께 들어가지 않는 실제 conditional boundary다. Stale launch certificate와 순차
outcome replay 반례를 각각 bounded revalidation/physical latest-start와 common-cutoff atomic
batch transition으로 교정했다. 상세는
[C159-Q2 결과](SOFTWALL_CONFIRM159_Q2_VARIABLE_RESULT_KO.md)에 있다.

**Q3 완료:** Calibration 4개 창, request4,290, offered token1,129,504에서 같은 exact offline
selector를 SoftWall과 recovery-first Event-driven에 적용했다. Q2 실측 recovery 시간을 사용하면
두 정책은 931 request·385,262 token으로 완전 동일했다. 매 recovery가 선언 25 ms를 모두 쓴다는
보수적 sensitivity에서도 SoftWall 이득은 0.066%였다. 사전 MDE 5%를 충족하지 못했으므로
confirmatory performance holdout을 materialize하지 않는다. H5는 현재 trace/mode에서 종료하고
C160/C161 fault와 C162 envelope로 진행한다.

### C159-P — 실제 AI-and-RAN trace의 강한 baseline 비교: Q3 gate로 중단

아래 설계는 Q3의 5% gate를 통과할 때만 실행하도록 고정했다. Q3가 0.000%였으므로
holdout은 열지 않았다. 통과했다면 AI는 보존한 BurstGPT densest 60초 trace를 사용하고 request arrival, context bucket,
deadline과 input-token value를 동일하게 replay한다. Qwen2.5-1.5B prefill을 실제 실행한다.
Radio는 두 home에 같은 frozen PUSCH/channel trace를 제공하고 미래 CRC를 정책 입력으로
사용하지 않는다. C157에서 연결한 GPU3 actual TensorRT NeuralRx와 GPU2 shared cuPHY
recovery path를 모든 비교군에 그대로 사용하며 controlled outcome으로 되돌리지 않는다.

두 독립 node에서 각 seed block을 다음처럼 교대한다.

```text
node A: Static -> EventDriven -> SoftWall -> SoftWall -> EventDriven -> Static
node B: SoftWall -> Static -> EventDriven -> EventDriven -> Static -> SoftWall
```

최종 반복 수는 C159-Q의 분산으로 power analysis한 뒤 test trace를 열기 전에 고정한다.
C159-Q는 성능 추정치로 합치지 않는다. 각 paired block에는 동일한 radio input ID, AI request
ID·arrival·deadline·context·offered value가 들어가야 하며 이 parity 자체를 hard gate로 둔다.

Primary metric은 deadline 안에 끝난 input-token value다. Secondary metric은 timely request,
GPU busy time, rejected/expired/late request, recovery fragmentation, scheduler overhead와
energy가 계측 가능할 경우 joule/token이다. Radio metric은 deadline miss, correct TB,
NRx/conventional commit, single-commit과 offered radio utility다.

성능 주장의 사전 gate는 다음과 같다.

- 모든 safe arm에서 safety violation 0
- 모든 paired arm에서 offered radio/AI trace와 mode fingerprint가 byte-identical
- SoftWall radio utility가 best safe baseline에 비열등
- 각 node와 실행 순서에서 SoftWall의 timely token value 방향이 일치
- paired block bootstrap 95% CI 하한 `> 0`
- best safe online baseline 대비 중앙 개선율 `>= 5%`

5%는 과거 1% 부근 효과가 네 번 부호 반전한 뒤 정한 실용적 최소 크기다. 이 gate를
통과하지 못하면 throughput superiority를 주장하지 않고, 비용과 동일 결과를 그대로
보고한다.

### C160/C161 — integrated fault matrix

정상 holdout source에서 fault만 사전 고정해 두 node에 각각 실행한다.

| Fault | 기대 동작 |
|---|---|
| 두 home의 correlated NRx fail/late | certificate 순서로 모든 accepted recovery 실행 |
| stale/duplicate NRx success | generation 검사로 무상태 폐기 |
| Qwen GPU 완료 뒤 RPC response delay | fence가 일치할 때만 retire, 이후 정책대로 지속 |
| Qwen timeout before fence | lease 유지, 신규 AI fail-closed, RAN recovery 지속 |
| shared worker GPU 완료 뒤 response delay | physical credit 유지 후 fence/response 규칙대로 한 번 commit |
| broker post-apply reply loss | ambiguous generation quarantine, duplicate launch 0 |
| worker process exit | 미완료 credit 미반환, 신규 optional work 차단, 가능한 mandatory 경로만 지속 |

각 state-changing operation `reserve`, `resolve`, `replan+lease`, `retire`에 최소 두 독립
physical fault arm을 요구한다. 각 arm은 주입 시각 전후의 generation, outstanding physical
credit, quarantine 상태와 radio continuation을 모두 보존한다. GPU hang/driver reset은 별도
unqualified mode로 남긴다.

C160은 unit8과 51개 state-transition product에서 invariant violation0·reject mutation0으로
통과했다. C161 1·2단계는 A0--A6를 qualified node의 paired campaign으로 실행해 합계 actual
NRx2,800, recovery942, radio commit2,800, deadline miss0을 얻었다. A2/A6 event replay pair40,
A3/A5 terminal fault4와 post-fault continuation260도 통과했다. 한 별도 node의 first-round
NRx47.470 ms는 NRx45 bound를 깨 해당 lifecycle을 UQ로 남긴다.

### C162 — feasibility envelope와 scheduler scalability

Envelope 축은 home 수, debt 수, recovery capacity, `P/D`, NRx/conv/AI class, MPS cap,
fault correlation, GC/cold/restart와 resident memory다. 전체 조합을 물리 실행하지 않고
모델 grid를 먼저 계산한 뒤 다음 boundary point를 holdout으로 선택한다.

- QSU 내부점과 경계점
- QSN: radio는 안전하지만 자격화된 Qwen이 들어가지 않는 점
- MI: mandatory all-fail schedule 자체가 불가능한 점
- UQ: component bound 또는 lifecycle evidence가 없는 점

물리 결과와 예측의 confusion matrix를 보고한다. **False-safe는 0이어야 한다.** False-unsafe는
안전성을 깨지 않지만 활용률 비용이므로 별도 보고한다.

C162 small-state 단계는 완료했다. Qualified grid 16,023개에서 analytic/exact mismatch0,
C159-Q2 retrospective decision 1,200/1,200 일치, 새 development/holdout 두 node의 사전 고정
boundary 180/180 일치를 얻었다. Actual NRx720·physical recovery330·Qwen90·single commit720,
deadline miss0이다. E6 context64는 conservative 88 ms 허가와 89 ms 거절을 각30회 재현했다.
[상세 결과](SOFTWALL_C162_PREDICTIVE_ENVELOPE_RESULT_KO.md)를 따른다.

현재 exhaustive solver는 small-state reference다. 실제 runtime에는 schedule을 반환하는
deterministic certified heuristic 또는 bounded branch-and-bound가 필요하다. 반환된 schedule은
독립 verifier로 검사한다. 작은 상태에서는 exact oracle과 acceptance gap을, 큰 상태에서는
home/debt 1--64의 decision latency와 memory를 측정한다. 목표는 false-safe 0과 사전 고정한
control budget 안의 p99 decision latency다. 이 결과가 없으면 SIGMETRICS식 규모 모델링
주장은 small-state envelope로 제한한다.

이 scalability gate도 완료했다. 임의 이질 small state600개에서 false-safe0,
false-conservative10이었고 현재 qualified state16,023개에서는 exact와 판정 차이0이었다.
Debt1--64의 2,800-state grid에서 반환 certificate verifier 실패0, debt64 decision/verifier
p99는1.494/0.131 ms였다. 이는 production WCET가 아닌 finite CPU 실행 결과다.

### C163--C165 — production timing, lifecycle, cross-family

C163의 목적은 synthetic `D155`를 실제 DU의 request별 `d_MAC`으로 교체하는 것이다. Raw
DU event, clock calibration, explicit expiry와 same-topology mode qualification을 별도
artifact로 보존하고, v2 validator가 initial all-fail schedule과 decision-time AI lease를
request별 절대 시간으로 재검사한다. Builder·bridge·validator의 24개 단위시험은 완료했지만
실제 trace는 아직 없어 `UQ_NO_PRODUCTION_TRACE`다. 세부 표본, baseline과 중단 규칙은
[C162 이후 실험 계획](SOFTWALL_POST_C162_EXPERIMENT_PLAN_KO.md)에 사전 고정한다.

C164는 warm, cold, restart, long-idle과 GC/model reload를 서로 다른 mode로 자격화한다.
현재 lifecycle-token 모델, `idle_30s_first`와 quiescent restart 뒤 재자격한
`mps_restart_first` two-node boundary subset은 통과했다. Restart 구간의 optional service는
닫으며 무중단 restart는 주장하지 않는다. NeuralRx process/model cold, worker reconnect,
5분/30분 idle, GC, reload 중 optional availability와 full six-class vector는 아직 UQ다.
C164 Qwen reload subset은 같은 recovery GPU에서 fresh model load와 6-class warmup을 node당
30회 수행하는 동안 4-cell mandatory path를 계속 실행했다. 합계 release3,334·cell13,336에서
miss와 cell25 위반은 0이지만 optional inference는 닫았으므로 reload availability는 UQ다.
C164의 same-worker channel reconnect는 두 node·token120에서 durable terminal reconciliation과
mandatory release145·miss0을 통과했다. [통합 lifecycle matrix](../../results/softwall_multigpu/c164_lifecycle_qualification_summary_v1.json)는
qualified/partial subset 5개와 UQ mode 5개를 분리해 `C164_LIFECYCLE_MATRIX_PARTIAL`로 판정한다.
5/10은 claim-scoped 종료점이다. 남은 lifecycle mode는 UQ/future work로 유지하고, 새 journal
recovery mechanism이나 독립 process-replacement holdout은 첫 원고 검토에서 claim-critical
gap이 확인되지 않는 한 수행하지 않는다. Claim/evidence freeze와 audited 영문 초고는
완료했으며 다음 내부 gate는 venue 원고 압축, 도표·bibliography와 reviewer audit이다.
C165는 A100이 아닌 GPU family에서 service vector 전체를 다시 측정한다. C163 PASS 없이
production RAN deadline 문구를 추가하지 않고, C165 PASS 없이 cross-family 일반화를 하지
않는다.

## 6. 통계와 재현 규칙

1. Protocol, source SHA-256, trace hash, node exclusion, seeds, arm order와 gate를 실행 전에
   JSON으로 고정한다.
2. Development, pilot, confirmatory holdout을 다른 파일과 seed range로 분리한다.
3. 동일 process/node 안의 TB를 IID로 세지 않는다. Paired 분석 단위는 고정 길이 time block,
   arm, node 순으로 계층화한다.
4. Safety는 평균이나 p99가 아니라 violation count와 원시 maximum을 함께 보고한다.
5. 0회 초과는 WCET 증명이 아니다. Event-IID, arm-IID, node-IID 가정별 one-sided 상한을
   민감도로만 보고한다.
6. 실패, smoke, analyzer bug와 protocol mismatch를 덮어쓰지 않고 ledger에 보존한다.
7. Test 결과를 본 뒤 bound, 최소 효과 크기, seed 또는 비교군을 바꾸지 않는다.

## 7. 실행 우선순위와 중단 규칙

```text
C153 integrated canary
  -> 실패: model/path glue 수정 후 새 development ID
  -> 통과: source와 mode freeze

C154/C155 two-node holdout
  -> safety/bound 한 건이라도 실패: V17 UQ 유지, 성능 실험 중단
  -> 통과: C156 semantics + C157A/C157B actual-NRx

C156 GPU timeline
  -> 고정 lease FAIL: launch control budget을 포함한 V17.1로 교정
  -> V17.1 PASS: C157A/C157B actual-NRx·독립-node gate로 진행

C157A/C157B actual-NRx
  -> 두 node PASS: C158 repeated qualification으로 진행
  -> actual outcome/cutoff FAIL: 해당 lifecycle UQ, 성능 실험 중단

C158 repeated qualification
  -> 두 node·2,000 request PASS: C159-Q P180 trace-mode qualification으로 진행
  -> lifecycle/bound FAIL: C157 mechanism canary만 유지

C159-Q1 P180 context64 qualification
  -> 두 node·2,000 request PASS 완료

C159-Q2 variable-context qualification
  -> 두 node·1,200 epoch·4,800 NRx PASS 완료

C159-Q3 oracle screen
  -> empirical 0.000%, contract sensitivity 0.066% < 5% MDE
  -> C159-P를 열지 않고 no-material-headroom 경계로 종료 완료

C159-P performance
  -> >=5%와 CI gate 통과: usefulness claim 추가
  -> 미통과: performance superiority 삭제, substrate/envelope claim 유지

C160/C161 faults + C162 small-state envelope
  -> false-safe 또는 credit reuse 발생: 해당 mode UQ
  -> 모두 통과 완료: V17 qualified-warm finite-sample system claim 후보

C162 large-state scheduler scalability
  -> home/debt 1--64의 verifier·decision latency가 control budget 통과: scalability claim 추가
  -> 통과 완료; production WCET로 확대하지 않음

C163 production timing
  -> actual trace/clock/expiry/mode artifact가 없으면 UQ 유지
  -> false-safe, deadline miss, decision mismatch가 한 건이면 해당 mode UQ
  -> development PASS 뒤 source·bound·gate를 freeze하고 독립-node holdout

C164 lifecycle / C165 cross-family
  -> 각 lifecycle·GPU family를 별도 mode로 자격화
  -> warm A100 숫자를 복사하지 않으며 실패 mode만 UQ로 격리
```

실험 수를 늘려 실패한 gate를 평균으로 덮지 않는다. 같은 원인의 세 번째 반복보다 원인에
맞는 mode 수정이나 주장 축소를 먼저 한다.

## 8. 논문 그림과 표로 연결

| 논문 결과 | 입력 실험 |
|---|---|
| Figure 1: optional NRx가 만드는 조건부 debt와 global false-safe 반례 | H1 model + C153 |
| Figure 2: global certificate, launch-time revalidation, shared cuPHY worker와 Qwen lease | C153/C156 |
| Table 1: 두-node correctness와 component timing | C154/C155 |
| Figure 3: predicted vs measured QSU/QSN/MI envelope | C162 |
| Figure 4: class별 conditional admission과 oracle no-headroom screen | C159-Q2/Q3 |
| Table 2: integrated fault containment | C160/C161 |
| Figure 5: certificate decision latency와 acceptance gap | C162 |
| Figure 6: 실제 request별 expiry에서 예측/물리 QSU·QSN·MI confusion matrix | C163 |
| Table 3: warm/cold/restart/idle 및 GPU-family별 qualified mode vector | C164/C165 |

현재 C163 raw-event builder, request-specific exact/certified bridge, hash/clock/expiry/mode validator와
24개 시험은 준비됐다. 다음 실행 우선순위는 실제 DU/FAPI `d_MAC` artifact 확보와
same-topology qualification이며, 그 뒤 cross-family/lifecycle 재자격이다. Small-state
QSU/QSN/MI/UQ 예측, 두-node physical holdout과 64-debt certified scheduler의 finite CPU
scalability는 완료했다.
C159-P 성능 holdout은 Q3 사전 중단 규칙에 따라 열지 않는다.
