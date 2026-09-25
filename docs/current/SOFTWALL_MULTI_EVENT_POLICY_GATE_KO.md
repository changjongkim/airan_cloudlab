# SoftWall 공동 정책의 다음 판정 gate — 2026-09-22

> **최종 판정:** C102의 calibration/structural gate는 통과했지만 outcome gate는
> 실패했고, guarded joint와 max-radio가 39/39 동일했다. 추가 optimizer 탐색은
> 종료한다. C102 해석과 후속 연구축은 [논문 구조 검토본](SOFTWALL_PAPER_STRUCTURE_KO.md)을
> 따른다.

**현재 상태:** [Confirm91](../../results/softwall_same_gpu/confirm91_conv12_nrx30_gc_off_job58743005.json)은
예열·GC OFF·4셀의 실제 `B_NRx30/B_conv12/AI15`, `P180/D155` 달력에서 두 독립
seed의 물리·안전 표본 gate를 통과했다. 이는 실행 기반의 일부일 뿐이다.
[Confirm78/79b](../../results/softwall_same_gpu/confirm79_contention_screen_v2.json)와
[Confirm92](../../results/softwall_same_gpu/confirm92_qualified_fourcell_screen.json)는
강한 AI-aware 1-swap greedy와 exact의 NRx 선택·기대 AI 가치 차이를 찾지 못했다.
Confirm92의 4개 AI unit은 D155뿐 아니라 물리 미자격 D115 후보에서도 같았다.
후속 [Confirm93](../../results/softwall_same_gpu/confirm93_seven_ai_capacity_screen.json)은
7개 AI unit의 조건부 용량 경합을 넣었지만 네 조건 각각 41개 실행 가능 상태에서
exact–greedy 차이 0이었다. [사후 선택면 감사](../../results/softwall_same_gpu/confirm93_choice_surface_posthoc.json)는
그중 31개에서 NRx 집합별 기대 AI 값이 실제로 달랐음(최대 차이 0.663)을
보여 준다. 따라서 결정 공간 자체가 공집합인 것만은 아니다.
[사후 폐형식 감사](../../results/softwall_same_gpu/confirm93_closed_form_posthoc.json)는
실행 가능 164건 모두의 exact 값이 `Σ AI 가치 − min(AI 가치) × Π(1−p_i)`와
`1.8×10⁻¹⁵` 이내로 일치함을 확인했다. 현 단일 관측 event 모델은 결국
“선택한 NRx 중 하나라도 성공하는가”로 축약되어 강한 greedy가 충분했다.
[사후 단계별 비교](../../results/softwall_same_gpu/confirm93_decomposed_baselines_posthoc.json)는
각 조건의 41개 실행 가능 상태 중 27개에서 공동 선택의 기대 AI 가치가
`PHY 선택→AI 최적 배치`보다 높았고, 그중 24개는 예측 무선 이득도 낮지
않았음을 보였다. 이 비교는 **기존 표본을 보고 설계한 사후 가설**이다.
새 독립 PHY seed의 [Confirm94 사전 protocol](../../results/softwall_same_gpu/confirm94_independent_phy_protocol.json)에서
재현 여부와 숨긴 CRC 정답의 짝비교를 검사했다. [결과](../../results/softwall_same_gpu/confirm94_independent_phy_screen.json)는
100개 새 quartet 중 70개 실행 가능, 그중 50개에서 공동 선택의 기대 AI 가치가
최소-NRx 단계별 정책보다 높았다(평균 +0.2098 unit). [NRx 수 분해](../../results/softwall_same_gpu/confirm94_endpoint_count_posthoc.json)에
따르면 양의 차이 50건 중 48건은 단계별 정책이 NRx 하나만 쓰고 공동 선택은
두 번째 endpoint까지 사용하는 경우다. 둘 다 NRx 두 개를 쓰는 22건에서는
AI 이득이 2건, 평균 +0.00125 unit이었다. 공동 exact와 AI-aware
1-swap은 다시 70/70 같았다. 그러나 공동 선택은 **70개 모두 두 NRx endpoint를
사용**했고, 무선 기대 이득을 최대화한 단계별 정책과 비교하면 AI 평균 +0.0166
unit에 무선 기대 이득 평균 −0.0316 correct/4TB였다. AI 이득 24건 중 무선 손실
≤0.01은 2건뿐이다. [사후 숨긴 정답·AI proxy 분석](../../results/softwall_same_gpu/confirm94_paired_outcomes_posthoc.json)은
공동/최소-NRx/max-radio 정답 150/149/153건, 7-unit 단일 사건 모델 내 AI
완료 478/465/476건이다. 이것은 GPU 실측이나 무선 비열등성 증명이 아니다.
또한 [C91 물리 trace 사후 감사](../../results/softwall_same_gpu/confirm91_ai7_physical_posthoc.json)는
두 seed의 동시 실패 주입 930 release **모두 D153 전에 AI 작업 최소 36개가 실제
완료**됐음을 보여 준다. AI RPC host 최대는 6.724/6.887 ms였다. 따라서
C93의 AI15 단위 7-job 경합은 현재 물리 작업 길이에서 성립한다고 볼 수 없다.
후속 [AI8 실제 달력 frozen 시험](../../results/softwall_same_gpu/confirm95_ai8_fourcell_job58746343.json)은
새 두 seed×1,000 release에서 출처·계약·안전·coverage gate를 모두 통과했다.
AI host 최대 6.790/4.999 ms, RAN miss·NRx30·복구12·AI8·guard·credit/fault
위반 0, 주입 동시 실패 153/143 release에서 D153 전 AI 완료 최소 38개였다.
이는 예열 4셀·GC OFF의 **유한 표본 자격**이며 일곱 작업의 원자 선허가나
WCET 증명이 아니다. [AI8 모델 재검사](../../results/softwall_same_gpu/confirm96_ai8_capacity_screen.json)는
이 물리 gate 통과를 조건으로 같은 독립 PHY trace를 재평가했다. 실행 가능
70/70 사례에서 공동 exact·공동 greedy·두 단계별 정책이 모두 AI 7개 전부를
수용했고, 공동 정책의 AI 우위는 **0건**이었다. 기존 15 ms 선언으로 만든
7-unit 단일 사건 성능 가설은 이 mode에서 기각한다. 이미 사용한 PHY trace의
서비스 시간 민감도 검사지 새 독립 PHY 반복은 아니다.
이것은 다른 도착·마감·상관 결과 모델에 대한 정리가 아니며 frozen 음성 결과를
성공으로 재해석하지 않는다.
따라서 **현 4셀·7-AI·단일 사건 workload의 공동 정책 AI 우위는 없다.** 다른
물리 workload·온라인 다중 사건에서의 정책 우위와 논문 노벨리티도 아직 미입증이다.

## 고정할 온라인 정보와 행동

매 사건 `t`에서 양쪽 정책에 같은 정보만 준다: 도착한 TB의 관측 가능 feature와
train-only `q_i^PHY`, 그 시각까지 도착한 AI unit의 길이·마감·가치, endpoint queue,
물리적으로 미완료인 kernel/IPC/lease, 이미 확정한 복구 의무와 deadline. 아직
도착하지 않은 AI 작업, 아직 관측하지 않은 NeuralRx 결과, held-out 정답·실제 SNR,
미래 GPU 서비스 시간은 온라인 입력이 아니다.

결정은 `(NRx 수락·endpoint, NRx 결과 관측 우선순위, 필수 conventional 순서·cutoff,
복구 calendar, 지금 발급할 AI lease)`의 첫 비선점 행동이다. NRx 결과 게시와
controller 관측 사이도 서비스 경로에 포함한다. 사건 후 재계획은 가능하지만 이미
제출한 작업을 논리적으로 취소해 credit을 반환할 수 없다. GPU 완료 fence가 확인된
뒤에만 endpoint/AI lease를 반환한다.

**공통 안전 판정기:** 그 시각까지 허용된 AI 작업을 실제 제출해도 모든 미해결 NRx가
늦거나 동시에 실패하는 분기에서, 검증된 mode별 *전체 host 경로* 상한으로 모든
conventional commit이 `d_MAC` 전에 끝나야 한다. 겹친 작업의 상한을 측정하지
않은 mode에서는 그 겹침을 허가하지 않는다. 달력 변경과 lease 발급은 한 원자
transaction이다. 이 판정기와 MPS cap, feature·`q_i`, queue 정보, 관측 순서의
선택 자유도를 모든 비교군에 동일하게 준다.

## 비교군과 성능 목표

1. **안전한 고정 gate:** train-only low-feature gate와 동일한 조기 복구·AI lease.
   Confirm60의 반복 이득을 가진 기준선이므로 생략하지 않는다.
2. **강한 단계별 정책:** PHY 수락 집합을 먼저 정하고, 그 집합에 대해 공동 정책과
   같은 all-fail 안전 판정기·복구 재배치·AI exact recourse를 허용한다. 최소 NRx
   수로 무선 `q` 하한을 만족하면서 같은 수 안에서 `q`를 최대화하는 버전과,
   실행 가능한 집합 중 `q` 자체를 최대화하는 버전을 모두 보고한다. 후자의
   무선 이득을 AI 성능 차이와 함께 표시해 무선 손실을 감추지 않는다.
3. **공동 정책의 효율 구현:** 보이는 AI 수요를 이용한 NRx add/drop/1-swap,
   endpoint, 관측·복구 순서 및 AI 허가의 재배치다. 모든 후보는 같은 분기별
   안전 판정기를 통과하고 정확한 보이는 AI recourse를 평가한다. 이 greedy는
   이미 조건부 PHY·복구·AI 결합을 구현하므로 **경쟁 기준선인 동시에 공동
   정책의 효율 구현**이다. exact joint solver는 작은 상태의 최적성 참고값이다.
   greedy와 exact가 같으면 exact 탐색 자체의 우위는 주장하지 않는다.
4. **다중 사건 공동 정책 후보:** 보수적으로 보정한 `q_i^use`와 조건부 복구 의무의
   shadow cost를 함께 평가하여 동일 무선 비열등 하한 아래 적시 AI 가치를
   최대화한다. 새 도착·개별 결과 관측에서 3번보다 다른 행동을 보일 때에만
   추가 알고리즘 기여를 주장한다. 미래 도착 oracle과 online greedy를 직접
   비교해 성능 주장을 만들지 않는다.

가장 먼저 CPU 동일 trace에서 반복되는 **다른 안전한 행동**을 보여야 한다.
차이가 있는 사례마다 `q_i^use`를 균등화하거나 조건부 복구 의무를 항상 필수로
고정했을 때 차이가 사라지는지 확인한다. AI 길이·가치만 바꿔 생기는 일반
packing gap은 노벨리티로 세지 않는다. 차이가 없으면 GPU 처리량 비교를
추가 seed로 밀어붙이지 않고 공동 알고리즘 우위 주장을 중단한다.

## 물리 실행 순서와 중단 기준

1. **경합 영역 자격:** C91의 D155에서 AI unit 수요·도착·마감이 복구 의무와
   실제로 경쟁하는 상태를 찾는다. 먼저 NRx·conventional뿐 아니라 **AI host
   작업의 실제 mode별 bound**를 자격화한다. C93 추상 모델에서는 6×15 ms가
   한 early AI와 한 early mandatory
   conventional 이후 모든 실패 복구와 함께 들어가지만, 7×15 ms는 같은 배치의
   all-fail 분기에서 모두 선허가할 수 없다. 두 NRx 성공 뒤에는 7개가 가능하다.
   [전수 일정 검사](../../results/softwall_same_gpu/confirm93_conditional_capacity_witness.json)는
   early conventional·AI 배치를 모두 열거해 6개 all-fail의 가능한 계획 19개,
   7개 all-fail 0개, 두 NRx 성공 뒤 7개 가능한 계획 24개를 확인했다.
   [7-unit CPU screen protocol](../../results/softwall_same_gpu/confirm93_seven_ai_capacity_protocol.json)은
   이 조건의 **탐색**이며 AI arrival을 모두 시각 0에 보이게 하므로 온라인 결과가 아니다.
   C91의 실제 AI가 훨씬 짧으므로 이 용량 증인을 물리 경합으로 승격하지 않는다.
   AI8이 실제 달력의 유한 표본 gate를 통과했으므로 `30+4×12+7×8+2=136<D155`라 현 7-unit
   all-fail 경합은 보수적인 직렬 계산에서도 사라진다. 그때는 작업을 임의로
   15 ms로 가정하지 말고, **실제 물리 서비스가 긴 AI class 또는 개별 도착·
   마감이 촘촘한 workload**를 먼저 사전 등록해 새 경합을 찾아야 한다.
2. **실제 부하 경로:** AI worker가 7개 작업의 개별 도착·마감·완료 fence를 기록하고,
   controller가 그중 발급한 lease와 복구 의무를 원자적으로 연결하는 물리 구현이
   필요하다. C91의 AI 15 ms 표본 자격을 7개 연속 burst의 간섭 상한으로 자동
   승계하지 않는다. 먼저 AI class의 실제 길이, 연속 burst, 독립 seed·고장·시작
   전환을 포함해 같은 mode를 자격화한다.
3. **paired 성능 gate:** 실제 물리 서비스 길이에서 다른 안전한 행동이 확인된
   입력으로 동일 PHY/AI trace, GPU, cap, 안전 검사, 무선 품질 하한을 고정한다.
   공동 greedy와 단계별 최소-NRx 및 **무선 이득 최대화** 기준선을 ABBA 순서·
   독립 seed에서 비교한다. 두 단계별 정책에는 같은 정확한 복구·AI recourse를
   제공한다. radio miss·상한·credit/fault 위반 0과 사전 정의한 무선 비열등성,
   적시 AI 완료 가치의 양의 차이를 요구한다. 새 다중 사건 알고리즘을 주장하려면
   공동 greedy와도 같은 정보·안전 판정으로 직접 비교한다. 방향이 뒤집히면
   성능 우위를 주장하지 않는다.
4. **일반화 gate:** 실제 대상 DU의 `d_MAC`, GC ON/OFF 및 cold/restart/장시간
   lifecycle, 8셀에서 실패한 receiver 상주 메모리 구조를 별도로 해결한다.
   synthetic P180/D155 통과를 production RAN 보증으로 환산하지 않는다.

exact와 공동 greedy가 계속 같다면 **exact 탐색의 추가 알고리즘 우위는 없다**.
실제 서비스 길이에서 단계별·무선 최대화 기준선과도 우위가 없다면 남는 결과는
실행 mode의 **feasibility envelope**다. 이를 논문화하려면 단순 실패 사례 수집을
넘어 시간·메모리·수명 mode별 경계를 예측하는 모델과 독립 물리 검증이 더 필요하다.

## Confirm102 독립 trace 봉인 상태 — 2026-09-23

새 다중 사건 정책을 기존 Confirm94 PHY 표본에 맞춰 설계하는 것을 피하기 위해
`confirm102_blinded_phy_protocol.json`을 먼저 고정하고 job `58788060`, node
`nid001193`에서 seed `20994102`, SNR `-10/-9/-8/-7 dB` 각 250건의 새 PHY trace를
생성했다. 개별 feature와 CRC label record는 열지 않았다. 다만 변경하지 않은 기존
생성기가 SNR별 conventional/NeuralRx 정확도 합계를 stdout에 출력했으므로 완전한
blind라고 표현하지 않는다. 이 aggregate 노출은 capture 문서에 기록하며 정책 설계에
사용하지 않는다.

이 trace를 읽기 전 별도 protocol에 AI 도착·개별 deadline·가치, 온라인 가시 정보,
low-feature 고정 gate, 두 단계별 정책, joint add/drop/1-swap, 제안 다중 사건 정책,
작은 상태 exact 참고값, 조건부 복구 제거와 `q` 균등화 ablation, radio 비열등성과
적시 AI 가치의 go/no-go 기준을 모두 고정해야 한다. 이 순서를 지키지 못하면
Confirm102는 독립 확인 표본이 아니라 탐색 표본으로 강등한다.

## Confirm102 독립 판정과 새 비용 항 — 2026-09-23

[사전 protocol](../../results/softwall_same_gpu/confirm102_multi_event_policy_protocol.json)은
동일 크기·가치·마감의 AI 7개를 시각 `0×5, 60, 90 ms`에 도착시키고, 시각 0에서
보이지 않은 두 작업은 어떤 정책에도 주지 않았다. admission은 C100 선언값
`NRx45/conv12/AI50`으로 검사하고, replay만 C100 관측값에서 올림한
`NRx15/conv6/AI22`를 사용했다. max-radio 예측 이득에서 0.01 이상 벗어나지 않는
guarded joint exact/1-swap, unguarded joint 1-swap, staged-min, max-radio,
Confirm60 low gate를 같은 all-fail 판정기와 exact AI recourse로 비교했다. 모든 AI가
동일하므로 이 결과를 이질적 knapsack packing 이득으로 설명할 수 없다.

[독립 결과](../../results/softwall_same_gpu/confirm102_independent_multi_event_policy.json)는
calibration과 구조 gate를 통과했다. 새 PHY 1,000건의 rescue Brier는 0.02761로
train prevalence 상수 0.05203보다 낮고 MACE는 0.01740이었다. 39개 실행 가능
quartet 중 NRx 집합별 조건부 AI 용량이 다른 경우는 37개였고, 모든 복구를 항상
필수로 고정하면 39/39 용량 차이가 0이었다. exact와 guarded 1-swap도 39/39 같았다.

그러나 **성능 gate는 실패했다.** guarded 정책과 max-radio+exact recourse가 39/39
같은 NRx 집합을 골라 적시 AI·실제 correct 차이가 모두 0이었다. staged-min 대비
AI +19·correct +5, low gate 대비 AI +15·correct +12였지만 가장 강한 max-radio
기준선에 대한 우위는 아니다. guard 없는 joint는 max-radio보다 AI 4개를 더
완료한 대신 correct TB 4개를 잃어 Pareto tradeoff일 뿐 비열등 우위가 아니다.
따라서 C102를 공동 정책 노벨리티 성공으로 세지 않는다.

이 결과가 드러낸 모델 누락은 NRx 수가 AI 물리 서비스에 주는 비용이다. C102는
NRx를 하나 더 선택해도 AI 시간이 같다고 두어, endpoint가 남으면 성공 확률만
증가하는 무료 행동으로 취급했다. [C103 사전 ABBA](../../results/softwall_same_gpu/confirm103_nrx_count_abba_protocol.json)는
같은 PHY에서 NRx를 정확히 1개 또는 2개 실행했다. [결과](../../results/softwall_same_gpu/confirm103_nrx_count_abba_job58788499.json)는
네 arm의 안전 gate와 두 방향 평균 부호 gate를 통과했고 2−1 NRx의 pre-observation
Qwen 실행시간 평균이 +0.034/+0.207 ms였다. 다만 사후 bootstrap 95% CI가 첫 pair는
0을 포함하고 둘째만 양수이므로 안정적 크기로 확정하지 않는다. 독립 1,000-release
ABBA인 [C104](../../results/softwall_same_gpu/confirm104_nrx_count_long_abba_job58788756.json)는
첫 pair `+1.062 ms [1.013,1.110]`와 역순 pair `−0.164 ms
[-0.224,−0.106]`로 **부호가 반전되어 frozen gate에 실패했다.** 적시 AI도
2−1 NRx가 −151 대 +8로 반전했다. 따라서 서로 재시작한 MPS/Qwen arm 사이의
상태 드리프트를 NRx-count 비용으로 모델에 넣지 않는다. C105는 한 persistent
process에서 동일 PHY를 인접 release에 재생하고 `1→2, 2→1`을 교대해 이 비용을
다시 분리한다. C105에서도 네 run/order stratum의 bootstrap 하한이 일치하지 않으면
NRx-count scalar cost 가설을 폐기한다.

[C105 결과](../../results/softwall_same_gpu/confirm105_interleaved_count_job58789084.json)는
안전·동일 PHY pairing·정확 count gate를 모두 통과했지만 성능 gate에 실패했다.
두 독립 run의 `1→2/2→1` stratum에서 2−1 NRx pre-Qwen 평균과 bootstrap95는
각각 `+0.112 [0.027,0.198]`, `+0.071 [−0.0067,0.151]`,
`−0.001 [−0.094,0.091]`, `−0.114 [−0.204,−0.023] ms`였다.
적시 AI 차이도 네 stratum 모두 0을 포함했다. 반면 radio correct는 네 stratum
모두 2-NRx가 양수였다. 따라서 **안정적인 NRx-count AI slowdown scalar 가설은
기각**하며 정책 노벨리티 입력으로 사용하지 않는다.

C102와 C105를 합치면 현 증거가 지지하는 것은 새 optimizer의 우월성이 아니라
조건부 복구 의무를 안전하게 해제·재배치하는 runtime substrate다. 이후 물리 비교는
실제 AI-and-RAN 요청 계약에서 (a) low gate, (b) staged-min, (c) max-radio+
동일 exact recourse, (d) unguarded joint 1-swap, (e) SoftWall 원자 credit/lease를
모두 둔다. (e)가 (c)와 같은 행동이면 알고리즘 성능 주장을 하지 않고,
안전한 work-conserving substrate와 feasibility envelope만 평가한다.
