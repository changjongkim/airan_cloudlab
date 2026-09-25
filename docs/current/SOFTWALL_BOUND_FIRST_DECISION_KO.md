# SoftWall bound-first 의사결정 — 2026-09-22

**현재 판정:** 선언 상한이 경합 선택 공간을 좁힌다는 가설은 맞지만, 관측 최대치를
바로 낮은 상한으로 대체할 수 없다. 기존 4셀 `P180/D155`는 `B_NRx50 +
4×B_conv25 + guard2 = 152 ms`로 3 ms만 남긴다. 상한을 줄여 실제 예약표를
바꾸면 서비스 mode도 바뀐다. 따라서 **상한 자격 → 실제 예약표 재검증 → 경합
정책 비교** 순서가 필요하다.

| 근거 | 확인한 사실 | 판정 |
|---|---|---|
| [과거 mode 재감사](../../results/softwall_same_gpu/pre_confirm82_bound_mode_audit.json) | 1-endpoint cap80 장시간 run의 NRx30 초과 6건은 GPU front/post 구간까지 수십 ms; 미예열 첫 conventional GPU 최대 35.303 ms | 짧은 예열 run의 3.1 ms를 전역 8 ms 상한으로 사용할 수 없음 |
| [Confirm82](../../results/softwall_same_gpu/confirm82_bound_campaign_job58738952.json) | 4셀 예열 co-run 두 seed×3,000 release, 기존 50/25 ms 계약 gate 통과. NRx 최대 38.179/44.589 ms로 25/30 ms 후보 실패. Conventional 전체 host 최대 5.732/7.978 ms로 8 ms 후보 관측 통과 | 8 ms 최저 여유 0.022 ms; 해당 mode의 표본 결과일 뿐 WCET 아님 |
| [Confirm83](../../results/softwall_same_gpu/confirm83_conv8_fourcell_job58738952.json) | 실제 복구 예약표를 8 ms로 바꾼 4셀 3,000 release에서 conventional 최대 6.551 ms·miss 0, 그러나 NRx 최대 51.151 ms로 50 ms 위반 2건 | **전체 frozen gate 실패**. 한 경로의 예약 축소가 다른 경로 시간도 바꿈 |
| [Confirm84](../../results/softwall_same_gpu/confirm84_pretraffic_oom_failure.json) | 8셀·복구 12 ms 새 mode는 독립 PHY receiver 생성 중 GPU OOM, timed release 0 | `50+8×12+2=148<D155`는 메모리 가능성도 보장하지 않음 |
| [Confirm85](../../results/softwall_same_gpu/confirm85_tail_probe_job58738952.json) | 4셀 release 616에서 NRx 44.038 ms 재현; admission·NRx 준비·worker는 정상, 첫 feature가 release보다 27.192 ms 늦게 시작 | 이 반복의 지연은 NRx worker 자체가 아님 |
| [Confirm86 사전 판정](../../results/softwall_same_gpu/confirm86_stage_gc_probe_job58738952.json)·[사후 진단](../../results/softwall_same_gpu/confirm86_gc_overlap_posthoc.json) | `NRx>30 ms` 재현 gate는 실패. release 608의 채널 준비 30.271 ms와 Python 2세대 GC 27.267 ms가 겹침 | GC는 원인 후보. 사전 실패를 유지하고 ON/OFF 개입으로 인과 판정 필요 |

NRx tail의 위치도 한 가지가 아니다. Confirm82/83의 큰 응답은 feature 완료 후 첫
dispatch 전 22.7–34.4 ms 공백에 있었고, Confirm85는 feature 시작 전 지연이었다.
이를 모두 GPU kernel 또는 모두 GC라고 단정하지 않는다. [Confirm87 사전
protocol](../../results/softwall_same_gpu/confirm87_gc_intervention_protocol.json)은
새 node의 두 동일-trace seed pair에서 GC ON/OFF/OFF/ON을 비교했다.
[frozen 감사](../../results/softwall_same_gpu/confirm87_gc_intervention_job58743005.json)는
네 arm의 기존 계약 통과와 **사전 원인 gate 실패**를 기록한다. 원인 gate는
worker 게시 뒤 controller 관측 대기를 지표에서 누락했다.
[사후 같은 요청 짝비교](../../results/softwall_same_gpu/confirm87_matched_gc_posthoc.json)는
각 pair의 3,200개 PHY feature와 NRx 수락이 전부 일치하고, 두 ON arm의
2세대 GC 28.845/27.672 ms가 worker 게시와 controller 관측 사이에 있으며,
해당 요청의 ON−OFF 응답 차이가 +27.3–28.5 ms임을 보여 준다. 강한 인과
단서지만 frozen 실패를 수정하지 않으며 독립 사전 재현이 필요하다.
[Confirm88](../../results/softwall_same_gpu/confirm88_gc_off_long_bounds_job58743005.json)은
GC OFF mode를 두 독립 5,000-release run에서 장시간 검사했다. 기존 50/25 ms
계약의 모든 frozen gate가 통과했고, NRx 14,796건의 최대는 21.223/20.116 ms,
conventional 29,597건의 최대는 8.458/5.136 ms였다. 표본 후보 25/12 ms는
초과 0으로 통과했지만, 복구 8 ms는 한 건 초과로 탈락했다. 이에 따라
[Confirm90 사전 protocol](../../results/softwall_same_gpu/confirm90_tight_calendar_gc_off_protocol.json)을
고정하고 **실제 25/12 ms 예약표**의 두 새 seed×3,000 release 물리 시험을 했다.
[frozen 분석](../../results/softwall_same_gpu/confirm90_tight_calendar_gc_off_job58743005.json)은
첫 반복의 NRx 25 ms 초과 1건(최대 26.875 ms) 때문에 **전체 실패**다. 두 번째
반복은 NRx 최대 24.051 ms로 통과했고, 두 반복 모두 deadline miss 0,
all-fail 네 conventional commit 461/467회와 복구 전 원자 AI 1,031/1,047회를
확인했다. 첫 초과는 사후 제외하지 않는다.
두 arm의 NRx 최댓값은 모두 warmup 뒤 **첫 timed release**에서 나왔다
(26.875/24.051 ms). 이 시작 전환 구간을 사후 버리면 상한을 과소평가하므로
후속 30/12 ms 시험에도 포함한다.
해당 release 0에서는 NRx worker가 release+16.118 ms에 결과를 게시했지만,
controller가 복구 전 AI 반환과 나머지 세 셀의 필수 conventional commit을 처리한
뒤 release+26.875 ms에 관측했다. GPU NRx 실행 지연이 아니라 현재 **관측·필수 작업
순서가 상한에 포함되는 사례**다. [사후 시간 분해](../../results/softwall_same_gpu/confirm90_tail_posthoc.json)는
원 frozen 실패와 별개다. 두 arm 모두 NRx 최대가 30 ms 아래이고 다른 관측 안전
위반이 없어, [Confirm91 새 protocol](../../results/softwall_same_gpu/confirm91_conv12_nrx30_gc_off_protocol.json)을
사전 고정해 실제 30/12 ms 달력을 새 seed 두 번에서 시험한다.
[Confirm91 frozen 분석](../../results/softwall_same_gpu/confirm91_conv12_nrx30_gc_off_job58743005.json)은
두 독립 4셀×3,000 release의 실제 `B_NRx30/B_conv12` 달력에서 모든 gate를
통과했다. NRx 5,181/5,164건의 최대는 20.549/21.261 ms, conventional
8,365/8,362건의 최대는 4.217/5.231 ms였다. deadline·상한·fault·credit
위반 0, 주입 all-fail 네 conventional 457/473회, 복구 전 물리 AI lease
1,063/1,025회다. `30+4×12+2=80 ms`라 D155 예약 여유는 75 ms이지만,
이것은 **그 예열 4셀·GC OFF mode의 유한 표본 자격**이다. 다른 간섭·GC ON·
cold/restart·더 많은 셀·production DU의 WCET로 확장할 수 없고, AI 처리량
우위나 공동 정책 노벨리티도 이 실험에서는 비교하지 않았다.
[Confirm89 사전 protocol](../../results/softwall_same_gpu/confirm89_gc_causal_replication_protocol.json)은
Confirm87에서 빠진 **worker 완료 뒤 host 관측**을 NRx 응답 자체로 판정하도록
수정하고, 새로운 두 seed pair에서 다시 ON/OFF/OFF/ON을 실행한다.
[Confirm89 frozen 분석](../../results/softwall_same_gpu/confirm89_gc_causal_replication_job58743005.json)은
**전체 실패**다. 첫 ON/OFF pair는 PHY feature와 수락 3,200/3,200이 동일하고
GC ON 2세대 정지 40.343 ms, OFF 0, 같은 release의 NRx 응답 차이 +38.797 ms로
사전 메커니즘 gate를 통과했다. 하지만 ON의 NRx 응답은 최대 55.323 ms로 기존
50 ms bound를 2건 넘겨 안전 gate가 실패했다. 역순 OFF/ON pair도 feature·수락이
3,200/3,200 같았고 GC ON 정지 20.824 ms, OFF 0이었지만 짝지은 응답 차이는
+8.069 ms라 사전 +10 ms 기준에 못 미쳤다. 두 번째 GC의 대부분은 해당
release 시작 **이전**에 있었고 해당 release와 겹친 시간은 5.733 ms뿐이다.
이는 사후 설명일 뿐 frozen 메커니즘 실패를 수정하지 않는다. 따라서 GC의
host 지연 기전은 강한 단서가 있으나 독립 두 pair의 사전 재현 gate는 통과하지
못했고, GC ON의 기존 50 ms 계약도 새 표본에서 반증됐다.

따라서 처음 제안한 `B_conv=8 ms`를 넣어 `4→12셀`로 늘리는 경로는 현 표본에서
성립하지 않는다. `B_NRx=25 ms, B_conv=12 ms`가 나중에 실제 달력에서
통과하더라도 `25+10×12+2=147 ms`라는 시간 산술만 가능성을 말할 뿐이다.
현 구현은 독립 PHY receiver를 셀마다 상주시켜 8셀 초기화에서 OOM이 났다.
메모리 구조를 바꾸거나 더 작은 셀 수를 물리 검증해야 하며, 4셀의 상한만
줄이면 AI 여유가 커져 오히려 기존 두 AI unit의 선택 문제는 더 퇴화할 수 있다.

현재 실험으로 선택할 수 있는 계약 범위는 다음과 같다.

| 후보 | 판정 | 이유 |
|---|---|---|
| conventional 8 ms | 탈락 | GC OFF 장시간 Confirm88에서 8.458 ms 한 건; 전 mode 공통 상한 불가 |
| NRx 25 ms / conventional 12 ms, GC OFF 4셀 | 탈락 | 실제 달력 Confirm90의 NRx 26.875 ms 한 건 |
| NRx 30 ms / conventional 12 ms, GC OFF 예열 4셀 | 유한 표본 통과 | 실제 달력 Confirm91의 두 새 seed×3,000 release에서 안전·물리 coverage gate 통과. WCET 아님 |
| NRx 50 ms / conventional 25 ms, GC ON 4셀 | 새 표본에서 탈락 | Confirm89 첫 ON arm NRx 55.323 ms, 50 ms 초과 2건 |
| 8셀 / conventional 12 ms | 물리 실행 전 실패 | 현 receiver-per-cell 구현의 GPU OOM |

## 연구 스킴에 반영할 계약

실행 mode를 `(셀 수, resident PHY context 수, MPS cap/동시 작업,
worker 수명·예열 상태, AI unit class, 예약표, host 제어 경로)`로 정의한다.
각 mode에서 NRx 관측 시간뿐 아니라 **release 전 준비, 관측 feature, admission,
NRx 준비/worker/후처리, conventional 결정→commit 반환, AI 물리 완료/반환**을
계측한다. 낮은 후보 상한을 쓰려면 그 값으로 실제 예약표를 운영한 독립 trace·
간섭·고장 시험까지 통과해야 한다. cold start나 worker 재시작은 별도 mode로
격리하고 재자격 전에는 보수 mode로 전환한다. 통계적 표본 최대만으로 hard
WCET를 선언하지 않는다.

이 계약 위에서 PHY의 *추가 구조 가치* `q_i^use`, 여러 셀의 동시 all-fail
conventional 복구 의무, AI 도착·마감·가치를 한 계획에서 결정한다. 복구 calendar
변경과 AI lease 발급은 원자적으로 확정하고, GPU 완료 fence 전에는 credit을
반환하지 않는다. **NRx가 cutoff를 넘겨도 mandatory 복구가 실행될 수 있는지**는
현재 보장 공백이다. 미완료 NRx와 conventional의 물리 동시 실행 상한, IPC buffer
수명, 늦은 결과의 single commit을 검증하지 못하면 해당 NRx admission을
제한해야 한다. MPS는 이 제어의 GPU 실행 기반이며 단독 격리 보장은 아니다.

다음 의사결정은 명확하다. Confirm87의 누락 지표를 바로잡은 독립 사전 검증과
Confirm88의 장시간 bound 감사로 host tail과 GPU tail이 남는 mode를 분리한다.
그 후 receiver 메모리 구조를 줄여 셀 수를
늘릴 수 있는지, 독립 노드·장시간·lifecycle·AI 간섭에서 **같은 mode 계약**이
성립하는지 검사한다. 물리적으로 가능한 경합 영역이 확보되면 그때 실제 AI
도착/마감과 DU `d_MAC`를 넣어 강한 AI 인지 greedy와 `q_i^use` 기반 공동 정책을
동일 정보·안전 조건으로 비교한다. 단순 일반 packing 차이와 수치 동점은
AI-RAN 노벨리티로 세지 않는다. [Confirm79b](../../results/softwall_same_gpu/confirm79_contention_screen_v2.json)는
가상 경합을 넣어도 exact와 greedy의 차이가 0이었으므로, bound 개선만으로
공동 정책의 성능 우위가 생긴다고 가정하지 않는다.

현재 [오프라인 결정 모델](../../scripts_for_node/softwall_same_gpu/offline_contingent_phy_ai.py)은
AI 작업이 모두 시각 0에 알려지고 선택한 NeuralRx의 관측을 하나의 공통 event로
묶으며 최대 4셀·3 AI unit만 평가한다. 이 구조에서는 사건 사이 새 AI 도착이나
셀마다 다른 PHY 결과가 복구 의무를 순차적으로 바꾸는 효과를 표현하지 못한다.
후속 [Confirm92 사전 고정 CPU 스크린](../../results/softwall_same_gpu/confirm92_qualified_fourcell_screen.json)은
실제 표본 자격을 얻은 30/12/15 ms와 AI 4 unit을 사용했지만, D155의 124개
실행 가능 사례에서 exact와 강한 AI-aware 1-swap greedy의 NRx 선택·기대 AI 값이
완전히 같았다. 물리 자격이 없는 후보 D115, AI 가치 동일/상이 조건에서도 같은
결과였다. 이는 **현 4-unit 부하의 결정 퇴화**라는 음의 결과이고, D115의
물리 가능성이나 공동 정책 우위를 입증하지 않는다.
7-unit [Confirm93](../../results/softwall_same_gpu/confirm93_seven_ai_capacity_screen.json)은
all-fail에서는 일곱 번째 AI를 모두 선허가할 수 없고 NRx 성공 후에는 가능해지는
**실제 조건부 선택 압력**을 추상 모델에서 만들었다. 그런데 네 조건 각각 41개
실행 가능 상태에서 exact와 강한 greedy는 다시 동일했다. [사후 감사](../../results/softwall_same_gpu/confirm93_choice_surface_posthoc.json)는
31/41 상태에 NRx 집합별 AI 기대값 차이(최대 0.663)가 있음을 확인해 단순한
무경합 설명을 배제한다. [폐형식](../../results/softwall_same_gpu/confirm93_closed_form_posthoc.json)은
164/164 상태의 값이 `Σ AI 가치−최저 가치×Π(1−p_i)`로 축약됨을 보여 준다.
따라서 현 단일 event 모델에서 **exact subset 탐색 자체는 차별 알고리즘이
아니다.**
다음 비교는 관측 가능한 도착·마감만 사용하는 다중 사건 모델이어야 한다.
미래 AI 도착을 미리 아는 exact 상한과 online greedy를 직접 비교해 우위를
주장하지 말고, 동일 online 정보의 공동 정책과 greedy를 먼저 비교한다.

현 계약에서 공동 결정이 계속 퇴화하면 그 자체를 **feasibility envelope**
경계 결과로 보고한다. 노벨리티 확보 판정에는 안전한 다른 `(NRx 선택, 복구
시각, AI 허가)` 행동이 반복해서 생기고, 그 원인이 조건부 PHY 가치·복구
의무의 결합이며, 동일 GPU paired 실험에서 무선 비열등성과 AI 적시 완료량
증가가 독립 seed·역순으로 유지돼야 한다.

## C97–98: 실제 Qwen 부하에서의 새 mode

[C97](../../results/softwall_same_gpu/confirm97_qwen_fourcell_failure_job58747070.json)은
예열·GC OFF 4셀에 Qwen2.5-1.5B prefill을 함께 실행했으나 NRx30 위반
705건과 다른 arm의 RPC timeout35/복구 전 물리 완료 미확인으로 실패했다.
이는 C91/95의 NRx30 자격을 **다른 AI 작업 class**로 확장하면 안 된다는
직접 반례다. 실패를 보존한 채 NRx45/복구12/AI50·RPC45로 새 달력을 고정한
[C98](../../results/softwall_same_gpu/confirm98_qwen_nrx45_ai50_job58747070.json)은
독립 새 seed 두 arm×400 release에서 안전·coverage gate를 통과했다. NRx
최대 43.412/42.818 ms와 conventional host 최대 9.161/4.868 ms이므로
이 mode도 각각 1.588/2.182 ms, 2.839/7.132 ms의 **표본 최대 기준 여유**만
있다. 특히 conventional 첫 arm의 12 ms 여유는 2.839 ms다. 이 숫자를
hard WCET 또는 다른 노드·cold/GC ON·재시작 mode의 자격으로 승격하지 않는다.

`45+4×12+2=95 < D155`라는 예약 산술은 유효하지만, 실제 Qwen 단위의
연속 backlog가 네 셀과 동시 실행될 때에도 AI7의 조건부 선택을 만든다는
뜻은 아니다. [C98 사후 분기 감사](../../results/softwall_same_gpu/confirm98_branch_posthoc.json)에서
NRx가 0개인 release까지 일곱 번째 Qwen 완료는 0건이었다. 현재 수요는
**일반 AI 포화와 6번째 단위의 복구 민감도**를 분리해야 하는 단계다. 따라서
초기 제안의 “3.1 ms 관측으로 8 ms 상한→12셀” 경로도, “무조건 7개 작업으로
경합 증명” 경로도 사용하지 않는다. 같은 PHY/AI 입력의 고장 ON/OFF 개입을
C99에서 실행했고, 이후 개별 작업 도착·마감 및 안전한 선택 차이가 남았다.

[C99 사전 고정 ABBA](../../results/softwall_same_gpu/confirm99_qwen_fault_abba_job58747070.json)은
같은 PHY 입력의 강제 실패 OFF/ON/ON/OFF로 6번째 Qwen 완료 민감도를
시험했다. 일치한 두 NRx 수락 26+26 release에서 실패 ON−OFF 적시 AI
차이는 −22/−9개였으나, 마지막 OFF arm의 NRx45 초과 1건(최대 48.605 ms)으로
**frozen 전체 실패**다. [사후 분해](../../results/softwall_same_gpu/confirm99_tail_posthoc.json)는
첫 timed release의 worker 결과 게시가 +16.990 ms, controller 관측이
+48.605 ms였음을 보여 준다. 그 사이 +15.797~39.862 ms의 Qwen RPC와
세 conventional commit(+39.917~47.733 ms)이 있었다. 그래서 짝비교의
방향은 탐색 단서이고, NRx45 자격 또는 정책 이득의 근거가 아니다. 다음
mode 검증은 GPU 실행 시간뿐 아니라 **결과를 언제 소비하는가**를 제어해야 한다.
기존 순서에서 `NRx release→controller 관측` 상한에는 feature·준비 시간 뒤에
`AI RPC 대기 + 먼저 실행한 필수 conventional 수×그 host 경로 + NRx 후처리`가
들어갈 수 있다. 따라서 worker kernel/게시의 표본 최대만 줄여도 이 상한은
자동으로 줄지 않는다. 같은 작업 class에서도 **관측 순서가 바뀌면 서로 다른
실행 mode**로 자격화한다. C100은 기존 소스를 보존한 별도 제어기에서 비동기
AI와 겹치는 NRx 결과를 필수 conventional보다 먼저 소비하는 후보이며, 같은
NRx45·복구12·AI50 계약을 새 seed의 실제 달력으로 검사한다.
[C100 frozen 감사](../../results/softwall_same_gpu/confirm100_observe_first_qwen_job58747620.json)는
두 새 seed×1,000 release에서 출처·안전·coverage gate를 모두 통과했다.
NRx 최대 12.968/14.562 ms, conventional host 최대 5.251/4.230 ms,
Qwen host 최대 26.083/26.801 ms, deadline·bound·guard·credit/fault 위반 0,
혼합 수락/필수 복구 release의 관측 우선 순서 위반 0이다. 복구 전 원자 AI
lease 318/350개도 물리 완료 뒤 반환됐다. **정책 순서가 mode 상한을
결정한다**는 직접적인 시스템 증거이지만, 다른 seed의 C99 숫자와 직접
paired 성능 개선율로 비교할 수 없고 WCET·production DU·AI-RAN 작업
일반화는 아니다.
[C101 frozen 고장 ABBA](../../results/softwall_same_gpu/confirm101_observe_first_fault_abba_job58747620.json)는
이 새 mode에서 동일 PHY 입력의 N/F/F/N 각 300 release를 실행했다. 네 arm
모두 안전·출처·coverage gate 통과, 일치한 두 NRx 수락 release 49+49개에서
강제 동시 실패 ON−OFF의 D153 전 Qwen 반환은 −7/−14개다. 이는 **복구
의무가 실제로 적시 AI 완료를 줄인다**는 좁은 물리 증거다. 아직 개별 AI
작업의 가치·도착·마감이나 서로 다른 정책 행동의 우위는 검증하지 않았다.

## AI 작업 상한도 경합보다 먼저 자격화

[Confirm93](../../results/softwall_same_gpu/confirm93_seven_ai_capacity_screen.json)의
7×15 ms 조건부 용량 경합은 추상 달력에서만 확인됐다. [C91 원자료의 사후
물리 대조](../../results/softwall_same_gpu/confirm91_ai7_physical_posthoc.json)는
실제 4셀·GC OFF 두 seed의 AI RPC 261,031건에서 host 최대 6.724/6.887 ms,
8 ms 초과 0건을 기록했다. 주입된 두 NRx 동시 실패 457/473 release **전부**에서
D153 전에 완료된 AI 작업은 최소 36개였다. 원자료는 AI15 달력으로 실행됐고
일곱 개 작업의 원자 사전 lease를 구현하지 않았으므로, AI8 자격·안전 보증으로
격상하지 않는다. 그러나 현재 7-AI 경합을 물리 현상이라고 주장할 근거도 없다.

그래서 [Confirm95 사전 protocol](../../results/softwall_same_gpu/confirm95_ai8_fourcell_protocol.json)은
새 두 seed에서 실제 AI8 host budget·NRx30/복구12·D155 달력을 운영하고,
동시 실패·복구 전 lease·D153 전 실제 AI 반환 및 모든 안전 지표를 다시 확인했다.
[두 arm의 frozen 결과](../../results/softwall_same_gpu/confirm95_ai8_fourcell_job58746343.json)는
각 1,000 release의 출처·계약·표본 안전·coverage gate를 **모두 통과**했다.
NRx 최대 18.950/19.649 ms, conventional host 최대 4.290/4.508 ms,
AI host 최대 6.790/4.999 ms, deadline·bound·guard·credit/fault 위반 0이다.
동시 실패 주입 153/143 release에서도 D153 전 AI 완료는 최소 38개였고,
복구 전 원자 AI lease는 356/345개 완료·반환됐다. 첫 B 실행은 Slurm step 생성
전 timeout이므로 [원 로그](../../results/softwall_same_gpu/confirm95_b_ai8_job58746343.log)를
보존하고 같은 protocol·seed의 [재시도](../../results/softwall_same_gpu/confirm95_b_ai8_job58746343_retry1.log)로 끝냈다.
이것은 예열 4셀·GC OFF의 유한 표본 자격이며 WCET나 일곱 작업의 원자 선허가
검증은 아니다. 이 조건에서는 `30+4×12+7×8+2=136 ms < D155`라 7개 단위의
all-fail 조건부 용량 압력이 보수적 직렬 산술에서도 사라진다.
[AI8 모델 사전 재검사](../../results/softwall_same_gpu/confirm96_ai8_capacity_screen.json)는
같은 C94 독립 PHY quartet 100건 중 실행 가능 70건 **모두**에서 공동 exact,
공동 greedy, 최소-NRx 및 max-radio 단계별 정책이 AI 7개 전부를 수용했다고
확인했다. 공동 정책의 기대 AI 우위는 0건이다. 이는 이미 사용한 PHY trace의
**서비스 시간 민감도 검사**이며 새 PHY 반복은 아니지만, 현 7-unit 단일 사건
모델의 경합·성능 가설을 이 자격 mode에서 명확히 기각한다.
다음 경합 workload는 **물리적으로 긴 AI class** 또는 다른 도착·마감 구조를
먼저 실측·사전 등록해야 한다. 선언 bound와 실제 서비스 길이가 크게 다른
작업을 임의로 많이 쌓아 정책 우위로 해석하지 않는다.
