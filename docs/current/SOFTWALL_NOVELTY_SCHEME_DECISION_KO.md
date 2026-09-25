# SoftWall의 연구 중심: 조건부 PHY 복구 의무와 AI 실행을 공동 제어

> **역사 문서:** 이 문서의 joint-policy 가설은 C102에서 max-radio와 39/39 동일해
> 종료됐다. 현재 기여는 [논문 구조 검토본](SOFTWALL_PAPER_STRUCTURE_KO.md)의
> certified recovery substrate와 feasibility envelope다.

**판정 (2026-09-22): 구체적인 논문 가설이며, 독립 노벨리티와 성능 우위는 아직 입증되지 않았다.**
MPS는 같은 GPU에서 PHY와 허용 AI를 실행하는 수단이다. MPS 단독의 격리 실패나 cap 조정은
이 논문의 새 알고리즘이 아니다. 목표는 *허용한 AI 작업 범위에서 RAN deadline과 무선 품질
요구를 지키고 AI 완료 처리량을 늘리는 것*이다.

## 단일 핵심 기여와 구현 결정

**핵심 기여 후보는 `PHY 가치 기반 조건부 복구·AI 공동 admission`이다.** 같은 GPU의
여러 PUSCH 중 NeuralRx가 conventional 대비 *추가로 살릴 가능성이 높은* TB만 고르되,
그 선택이 남기는 conventional 복구 의무를 모든 허용 실패 분기에 대해 예약한다. 그리고
현재 허용된 AI 작업의 실행 가능성까지 본 뒤 `(NeuralRx 선택·endpoint, 복구 순서·시각,
AI unit 허가)`를 **한 상태 전이로 확정**한다. MPS는 이 전이를 실제 단일 GPU에서
실행하는 기반이고, 연구의 독립 변수는 이 공동 결정이다.

구현은 요청/NRx 결과/AI 완료 사건마다 아래 순서로 한다.

1. 실제 MAC expiry와 관측 가능한 채널 feature로 각 TB의 `q_i = Pr(NRx 제때 성공 ∧
   conventional 실패 | feature)`의 보수적 추정치를 계산한다. 사용 가능한 AI unit의
   도착 시각, 작업 길이, 마감과 가치를 함께 읽는다.
2. TB별 `conventional-only` 또는 `(NRx, endpoint)` 후보를 만들고, 각 후보에 대해
   모든 미확정 NeuralRx가 실패·지연해도 conventional 전체 경로가 마감 전 끝나는
   복구 calendar를 구성한다. 이미 제출된 비선점 작업과 물리적으로 미완료인 작업은
   점유 상태로 남긴다.
3. 검증한 실행 mode별 상한으로 다음 AI unit을 놓아도 모든 복구 분기와 endpoint/IPC
   수명이 안전한 후보만 남긴다. 그 안에서 사전 고정한 무선 품질 하한을 만족하며
   *마감 내 AI 완료 가치*가 가장 큰 계획을 선택한다. 첫 행동만 제출하고 다음 사건에서
   다시 계산한다.
4. 복구 calendar와 AI lease는 원자적으로 확정한다. NeuralRx 성공으로 복구가
   불필요해졌다는 **물리 완료 확인** 전에는 해당 credit을 돌려주지 않는다.

**상한 자격에서 드러난 추가 행동:** 결과가 worker에 게시된 뒤 controller가 이를
읽는 시각도 결정 변수다. 실제 `B_NRx`는 GPU worker 시간뿐 아니라 feature,
AI 반환 대기, 조기 필수 conventional 실행, 결과 관측·후처리까지의
`release→controller observation` 전체 경로다. [Confirm90 첫 실제 25/12 ms
달력](../../results/softwall_same_gpu/confirm90_tight_calendar_gc_off_protocol.json)에서는
worker 게시가 release+16.118 ms였지만 AI 반환과 세 필수 conventional commit 뒤
관측이 +26.875 ms라 25 ms 상한이 한 건 깨졌다. [두 seed의 frozen 감사](../../results/softwall_same_gpu/confirm90_tight_calendar_gc_off_job58743005.json)는
첫 seed의 이 초과 때문에 전체 실패다. 향후 공동 계획은
**NRx 결과 관측 우선순위와 조기 필수 복구 순서**도 함께 고르고, 같은 자유도를
강한 baseline에도 제공해야 한다. 단순히 관측 순서를 바꾼 것 자체를 독립
노벨리티로 세지 않는다.

강한 기준선에도 같은 MPS, 채널 feature, 조기 복구, mode별 상한, AI 작업 정보와
**AI 수요를 보는 greedy 복구 재배치**를 제공한다. 공동 정책은 *채널 가치 때문에
선택한 NRx 집합과 여러 셀의 조건부 복구 순서가 AI 허가를 함께 바꾸는 사건*에서
이 기준선과 다른 안전한 행동을 내야 한다. 그 사건이 없거나 처리량 이득이 없다면
MPS+부품 결합의 유용성은 보고할 수 있어도 새 공동 알고리즘의 우위는 주장하지 않는다.

## 우리가 실제로 만들 스킴: PHY 가치가 있는 복구 의무의 공동 배치

핵심 단위는 GPU 점유율이 아니라 **TB별 조건부 복구 의무**다. 요청 `i`에 NeuralRx를
선택하면 그 결과가 제때 성공으로 확인될 때까지 conventional 전체 경로를 `d_i` 전에
끝낼 수 있는 권리를 예약한다. 여러 요청의 의무는 *동시에 모두 실패하는 경우에도*
서로 충돌하지 않아야 한다. 다만 그 의무의 실행 시각·순서는 사건이 관측될 때마다
바꿀 수 있다. AI unit은 이 조건부 의무와 물리적으로 미완료인 GPU 작업을 모두 고려해
검사한 뒤에만 허가한다.

온라인 결정은 다음 네 가지를 **같은 계획에서** 고르는 것이다: `(1) 어떤 TB에
NeuralRx를 시도할지, (2) 어느 endpoint에 보낼지, (3) 모든 허용 실패 분기의
conventional 복구 순서와 cutoff, (4) 지금 제출할 AI unit`. 먼저 보수적인 완료시간
상한으로 모든 분기의 무선 마감과 자원 수명을 검사한다. 통과한 후보 중에서 사전 정한
무선 비열등 조건을 지키며 제시간 완료 AI 가치가 가장 큰 것을 고른다. NeuralRx의
무선 추가 가치는 `Pr(NRx 성공 ∧ conventional 실패 ∧ NRx 결과가 d_i 전에 사용 가능 | x_i)`로
계산한다. 두 복호기의 단순 성공률 차이나 생성기의 미래 정답을 온라인 점수로 쓰지 않는다.

NRx 성공·실패, 늦은 완료, 새 요청, AI 완료가 오면 예약표를 다시 계산한다. 새 표와 AI
허가를 **한 transaction**으로 확정하고, 실제 GPU 완료 fence가 오기 전에는 예전
작업의 credit을 반환하지 않는다. 첫 비선점 행동만 실행하고 다음 사건에서 다시 결정한다.
이것이 구현할 정책의 중심이다. MPS cap·채널 gate·조기 복구·짧은 AI 작업은 이 정책에
필요한 실행 부품이지만 각각을 독립 기여로 주장하지 않는다.

**노벨리티 판정선:** 같은 관측 정보, MPS 예산, 실패 허용 범위와 복구 안전 검사로
AI 수요를 보며 복구 예약을 재배치하는 *강한 greedy*까지 비교한다. 공동 정책이 다른
안전한 `(NRx, 복구, AI)` 결정을 반복해서 만들고, 무선 비열등성과 deadline을 지키면서
유효 AI 완료량을 늘려야 결합의 연구 기여가 성립한다. 차이가 단순 AI packing이나
greedy의 허술한 안전 검사 때문이면 기여로 세지 않는다. 현재는 이 판정선을 통과하지
못했으므로 **제안한 노벨리티**이지 **확보한 노벨리티**는 아니다.

## 논문에서 검증할 한 문장

여러 셀의 PUSCH가 같은 GPU에 도착할 때, 현재 채널에서 NeuralRx가 **conventional만으로는
실패할 TB를 추가로 살릴 가능성**과, 아직 결과가 나오지 않은 NeuralRx마다 필요한
**conventional 복구 의무**를 함께 계산한다. 정책은 `(NeuralRx 선택, endpoint, 복구 일정,
허용 AI unit)`을 한 결정으로 고르고, 모든 허용 실패·지연 분기에서 MAC expiry 전의 복구
가능성을 보존하는 계획에만 AI 실행권을 준다. 성공이 물리적으로 확정되면 풀린 복구 의무를
다른 셀과 AI에 다시 배분한다. 이 방식이 같은 정보와 실행 계약을 가진 강한 결합 정책보다
무선 요구를 유지하면서 유효 AI 완료량을 늘리는지가 **검증할 가설**이다.

## 왜 이 결합이 단순 부품 나열보다 구체적인가

1. **무선 이득을 NeuralRx 성공 확률만으로 보지 않는다.** 관측 가능한 DMRS/채널 feature
   `x_i`로 `q_i^PHY = Pr(NRx가 CRC 성공하고 conventional은 실패 | x_i)`를 훈련 trace에서
   추정하고, 분리된 test trace에서 보정 정도를 확인한다. 실제 수락의 효용은 선택한
   endpoint·일정에서 NRx 결과가 MAC expiry 전 단일 commit에 **사용 가능할 확률**까지
   반영한 `q_i^use`로 계산한다. 이는 NeuralRx의 *추가 무선 구조 가치*다. 두 복호기가 모두 실패하는 채널과 둘 다 성공하는 채널에서 NRx 실행은 값비싼
   speculative GPU 작업일 수 있다. PHY 결과·미래 채널·true SNR은 online 입력에 넣지 않는다.
   실제 선택에서는 confidence bound와 전체 BLER 비열등 조건을 사용한다.
   [독립 seed 탐색](../../results/softwall_same_gpu/conditional_phy_value_confirm57_exploratory.json)에서는
   train에서 정한 가운데 feature quintile에 추가 NeuralRx 성공이 train 33/100,
   test 29/82로 집중됐다. 이는 다음 모델의 입력 가설이며 synthetic fixed-noise 채널에서
   본 사후 탐색이므로 production 정책의 검증 결과가 아니다.
   후속 [사전 고정한 5,000/5,000 독립 보정](../../results/softwall_same_gpu/confirm74_phy_value_job58734412.json)은
   같은 synthetic −8.5 dB 조건의 train 기반 가운데 quintile에서 추가 성공
   273/1,000(`q=0.2732`), held-out 256/1,058(`q=0.2420`)을 관측했고 5구간 가중 보정
   오차 0.00708, Brier 0.04158(훈련 전체 prevalence 상수 0.05092)을 기록해 frozen
   gate를 통과했다. 이 입력은 **해당 synthetic 채널 안에서만** 검증된 것이다.
2. **NRx 선택은 복구 빚을 발생시킨다.** 각 수락 TB에 대해, NRx가 늦거나 실패해도
   conventional 전체 경로와 commit이 `d_MAC` 전에 끝나는 복구 구간을 확보한다. 여러 셀의
   구간은 동시에 실패해도 충돌하면 안 된다. 따라서 `q_i`가 높아도 다른 셀의 복구 여유를
   망치면 NRx를 수락하지 않는다.
3. **AI 실행은 복구 빚과 교환하는 계약이다.** AI unit의 최대 비선점 blocking, MPS
   간섭을 포함한 실행 mode별 서비스 상한, endpoint/IPC 수명, controller 비용을 계획에
   넣는다. 어느 허용 분기에서도 복구 가능성이 남는 경우에만 unit을 제출한다. NRx가
   성공하여 빚이 소멸하거나 실패하여 복구를 실제 실행할 때마다 다른 셀의 구간을 앞뒤로
   이동하고 AI 허가를 다시 계산한다. 복구 calendar 변경과 AI lease 발급은 하나의 원자적
   commit이어야 한다. 논리적 취소만으로 GPU 자원을 반환한 것으로 계산하지 않는다.

이 세 결정을 분리하면 `좋은 채널이면 NRx 생략` 같은 고정 gate가 실제 비용을 거꾸로
판단할 수 있다. [Confirm58의 사전 고정 online ABBA 비교](../../results/softwall_same_gpu/confirm58_online_gate_abba_job58729926.json)에서
고 feature gate는 두 paired 반복 모두 AI 처리량이 감소했고, 첫 반복의 사전 정의된
무선 비열등 gate도 실패했다. 이 결과는 해당 gate의 실패이며 공동 정책의 성공 증거가 아니다.
특히 NRx를 생략한 TB의 conventional 실행을 다른 셀의 NRx 완료 *뒤*에 둔 현재
wait-both controller에서는, NRx 작업을 줄여도 복구 직렬 시간이 늘어 AI에 돌아갈
시간이 오히려 줄 수 있다. 따라서 PHY gate와 복구 순서·AI 허가를 함께 고르는 가설이
구체적인 시험 대상이 된다. 이 결합의 성능 우위는 아직 측정하지 않았다.
[Confirm60의 별도 사전 고정 비교](../../results/softwall_same_gpu/confirm60_low_gate_abba_job58729926.json)에서는
train-only 낮은 feature gate가 독립 두 쌍 모두 무선 정답을 보존하면서 AI를
+0.943%/+1.877% 늘렸다. 이것은 공동 정책과 비교할 **더 강한 채널 baseline**이며,
고정 gate 자체를 우리 노벨리티로 계산하지 않는다.

## 온라인 알고리즘의 정확한 입력·행동·안전 판정

- **입력:** release 시각, 대상 DU가 정한 요청별 MAC/CRC expiry `d_i`, 현재까지 관측 가능한
   PHY feature와 `q_i^PHY`의 보정 구간, 현재 endpoint queue와 물리 미완료 작업, AI unit의 길이·
  마감·가치, 실행 mode별 검증된 *host 결정부터 commit 반환까지* 상한.
- **행동:** 각 TB에 대해 conventional-only 또는 NRx endpoint 선택; 미해결 TB의 복구
  순서·시각; 이번 사건에서 실제 제출할 AI unit 또는 NRx stage. 이미 제출된 비선점 GPU
  작업의 취소는 행동 집합에 넣지 않는다.
- **안전 판정:** 모든 허용 NRx 실패·late 결과와 AI 최대 blocking 분기에서, 충돌 없는
  복구 calendar가 각 `d_i`와 IPC/endpoint credit을 만족해야 한다. 실행 mode가 달라져
  간섭 상한이 달라지는 경우에는 단순 all-fail 일정 하나로 충분하다고 가정하지 않고
  분기별 검증 또는 보수적 지배 상한을 쓴다. 검사·원자적 commit 비용도 deadline에 포함한다.
  MPS worker는 RAN epoch 동안 유지하고, AI unit이 종료/timeout 되어도 GPU 물리 완료를
  확인할 때까지 점유로 처리한다. 동시 실행 상한을 아직 검증하지 못한 mode에서는 AI를
  해당 필수 경로와 겹치게 제출하지 않는다.
- **목표:** 먼저 무선 BLER/CRC 효용의 사전 정의된 비열등 하한을 만족시키고, 그 안에서
  deadline 내 완료한 AI unit 가치의 기대값을 최대화한다. 작은 동시 셀 수에는 유한
  시계열 사건의 exact solver로 상한을 구하고, online에는 같은 안전 판정기를 통과하는
  bounded lookahead/heuristic을 쓴다. 단순히 AI 작업 여러 개를 잘 담는 일반 knapsack
  성능 차이를 AI-RAN 노벨리티로 세지 않는다.

## 이 결과가 나오지 않으면 노벨리티를 주장하지 않는다

강한 비교군에도 같은 채널 feature, `q_i^PHY`와 사용 가능 시각 모델, MPS cap, queue-aware routing, 조기 복구,
AI 단위와 서비스 상한, 물리 수명·fault 처리, 그리고 AI 수요를 보는 **greedy 복구 재배치**를
제공한다. 네 행동 `(NRx 수락, endpoint, 복구 시각, AI 허가)` 중 어느 것이 언제 달라졌는지
trace에 기록한다. 단순 복구 자리 이동은 이미 greedy가 재현했고, 일반 AI packing
차이는 AI-RAN 고유 기여가 아니다. 고정 gate, AI 인지 greedy, 공동 정책, 작은 상태의
exact optimum을 동일 trace로 비교한다.

다음 네 증거가 모두 필요하다.

1. **정책 차이:** 실제 또는 사전 고정 stress trace에서 강한 greedy와 다른 *안전한*
   NRx/복구/AI 결정이 나오고, 차이의 원인이 `q_i^use`와 여러 셀의 조건부 복구 의무임을
   counterfactual로 보여야 한다. 단순 AI packing 반례는 제외한다.
2. **물리 실행 계약:** 실제 DU의 `d_MAC`, 필수 conventional 용량, NRx–복구·AI–PHY
   co-run별 전체 경로 상한과 GPU 완료 fence, IPC 수명을 검증해야 한다. MPS cap만으로
   상한을 가정하지 않는다. 현 P150/D130은 production deadline이 아니다.
3. **paired 결과:** 같은 trace·GPU 자원·무선 요구에서 deadline miss/credit leak 없이,
   공동 정책의 완료 AI가 강한 greedy보다 사전 정의한 크기만큼 반복해서 증가해야 한다.
   채널/AI 부하·동시 실패·late result를 분리해 보고한다. 현 P150/D130은 넉넉해서
   정책 차이가 거의 없는 진단 부하다. 물리 계약이 성립한 뒤에만 모든 실패의 지속
   conventional 용량도 수용할 수 있는 더 높은 offered load를 사전 고정한다. 현재
   `B_conv_path=25 ms`, 2셀·단일 lane 가정에서는 최소한 `P≥50 ms`가 필요하다.
4. **기여 분해:** `q_i^use` 제거, 조건부 복구 교환 제거, AI lease 공동 결정 제거를 각각
   ablation하고, exact optimum과 gap을 제시한다. 하나의 일반 최적화 기법만으로 같은
   결과가 나면 새 PHY-aware 스킴이라는 주장을 접는다.

현재 구현은 두 셀 GPU bring-up, 온라인 feature 비교, 복구 calendar·AI lease의 원자 admission까지다.
후속 [Confirm65](../../results/softwall_same_gpu/confirm65_async_job58729926.json)는 미해결
NeuralRx 중 비동기 AI 한 unit의 **성공 경로**를 확인했고,
[Confirm68](../../results/softwall_same_gpu/confirm68_kernel_overlap_union_posthoc.json)은 두
MPS client의 실제 GPU kernel 중첩을 짧은 profiled sample에서 확인했다.
[Confirm69](../../results/softwall_same_gpu/confirm69_value_gate_job58734412.json)의 훈련 기반
추가 PHY 가치 구간 gate는 두 paired seed에서 요청별 무선 정답을 보존했지만 AI 처리량 변화는
+0.034%/−0.067%로 반복 이득이 없었다. 따라서 *gate를 더 합치는 것*은 해결책이 아니다.
[Confirm71](../../results/softwall_same_gpu/confirm71_greedy_retime_job58734412.json)은
복구 전 AI와 greedy 복구 재배치를 함께 켰을 때 AI가 +1.321%/+0.319% 늘어난 첫 canary지만,
[Confirm72](../../results/softwall_same_gpu/confirm72_retime_ablation_job58734412.json)에서
복구 전 AI를 양쪽에 동일하게 주고 **자리 이동만** 비교하자 +0.187%/−0.420%로
이득이 재현되지 않았다. 이는 현 P150/D130에서 *단순 retime이 노벨리티*라는 주장을
기각하는 근거다. 다음에는 실제 AI 마감·도착 다양성과 여러 셀의 복구 의무가
동시에 제약이 되는, 사전 고정한 **물리적으로 실행 가능한** 상태에서 위 공동 결정이
강한 AI 인지 greedy와 갈라지는지 먼저 보여야 한다.
[Confirm73](../../results/softwall_same_gpu/confirm73_atomic_lease_job58734412.json)은
복구 calendar와 AI lease를 한 번에 확정하고 AI worker의 GPU 동기 완료 응답 뒤에만
lease를 반환하는 물리 성공 경로 8건(동시 retime 4건)을 검증했다. 이 controller의
NeuralRx 선택은 여전히 고정 low gate이므로 **공동 정책의 차별 성능 증거는 아니다.**
[Confirm75](../../results/softwall_same_gpu/confirm75_three_cell_job58738952.json)는 3셀 중
두 NeuralRx가 동시에 실패하는 주입 분기에서 세 conventional commit 18회를 확인했지만
40 ms AI 선언 예산에서는 복구 전 원자 AI가 0회였다.
[Confirm76](../../results/softwall_same_gpu/confirm76_ai15_job58738952.json)은 별도 seed의
3셀·500 release에서 15 ms AI 선언 예산으로 복구 전 원자 AI 170회와 동시 retime 78회를
확인했다. 이 15 ms는 표본 관측 최대 6.017 ms보다 크지만 **WCET 자격을 얻은 상한은 아니다.**
다음 정책 비교는 이 실행 가능한 원자 경로에서 훈련 전용
[혼합 SNR PHY 가치 모델](../../results/softwall_same_gpu/confirm77_train_only_q_model.json)을 사용하고,
강한 AI 인지 greedy와 실제로 다른 `(NRx, 복구, AI)` 결정을 먼저 찾아야 한다.
[Confirm77](../../results/softwall_same_gpu/confirm77_multisnr_job58738952.json)은 서로 다른
4개 synthetic SNR의 train/test 각 8,000건에서 관측 가능한 두 feature의 추가 PHY 가치
보정 gate를 통과했다. 그러나 [Confirm78](../../results/softwall_same_gpu/confirm78_trace_decision_screen.json)의
3셀 638개 실행 가능 결정과 [Confirm79b](../../results/softwall_same_gpu/confirm79_contention_screen_v2.json)의
가상 경합을 넣은 4셀 365개 실행 가능 결정에서 exact와 강한 1회 교환 greedy가 같은
NRx 선택과 기대 AI 값을 얻었다. 따라서 현재 작은 상태·입력 분포에서는 **공동 탐색의
독립 성능 이득을 발견하지 못했다.** 더 많은 셀과 실제 AI 도착·마감 trace, 품질 위험
예산, `q_i^use`를 넣되, 먼저 물리적으로 실행 가능한 경합 조건과 동등 baseline의
차이를 blind trace에서 사전 고정해야 한다. 차이가 일반 local-search gap뿐이면 알고리즘
노벨리티로 세지 않는다.

Confirm78의 동일 결정은 우연한 숫자만은 아니다. 두 AI 단위의 선언 시간 합 30 ms는
NRx 관측까지의 50 ms 안에 들어가고, 관측 후 3개 all-fail conventional 의무
`3×25=75 ms`도 남은 radio window `130−50=80 ms` 안에 들어간다. 이 구성에서는 AI를
먼저 처리하고 모든 복구를 뒤에 넣을 수 있어 조건부 복구 credit의 선택 가치가 거의
드러나지 않는다. 다음 실험은 **AI 수요가 사전 구간보다 크되 all-fail 복구는 여전히
가능한** 마감/도착 상태를 먼저 측정으로 확인해야 한다. Confirm79b는 이를 가상
작업으로 강화했지만 365개 실행 가능 사례에서도 강한 greedy와 동일했다.

[Confirm80](../../results/softwall_same_gpu/confirm80_four_cell_job58738952.json)은 실제
4셀·2 endpoint·300 release에서 주입 all-fail 4 conventional 37회와 복구 전 원자 AI
완료 92회를 확인했다. [Confirm81](../../results/softwall_same_gpu/confirm81_fenced_fault_job58738952.json)은
AI의 GPU 작업이 끝난 뒤 RPC 응답만 지연한 고장 한 건을 별도 CUDA 완료 표식으로
회수하고 새 AI 허가를 차단해 99개 후속 release를 처리했다. 이는 공동 정책의 **물리·
특정 fault 실행 기반**을 강화하지만, 강한 greedy보다 높은 AI 처리량을 보여주지
않는다. GPU hang/worker crash, `q_i^use`, 실제 DU는 별도 입증이 필요하다.

**다음 판정은 경합 trace 생성보다 서비스 bound 자격 검증이 먼저다.** 현재 4셀
`D155`의 `B_NRx50 + 4×B_conv25 = 150 ms` 예약은 명목 5 ms,
2 ms commit guard 뒤 3 ms만 남겨 새로운 선택
상태와 셀 확장을 압박한다. 그러나 관측 최대치로 선언 상한을 즉시 대체하면
tail과 실행 mode를 누락한다. [원자료 mode 감사](../../results/softwall_same_gpu/pre_confirm82_bound_mode_audit.json)에서는
Confirm43/44의 10,000-release 구 실행 mode에서 NRx 30 ms 초과 6건의 GPU
front/post 꼬리가 확인되고, Confirm40의 미예열 첫 conventional GPU도 최대
35.303 ms였다. 짧은 4셀 예열 실행에서 복구 host 최대 4.935 ms였다는 사실은
`B_conv=8 ms`의 *후보* 근거일 뿐이다. [Confirm82 사전 protocol](../../results/softwall_same_gpu/confirm82_bound_campaign_protocol.json)은
현재 50/25 ms 계약을 유지한 채 두 독립 3,000-release 4셀 run에서 NRx
25/30 ms·복구 8 ms의 관측 위반을 따로 판정했다. [결과](../../results/softwall_same_gpu/confirm82_bound_campaign_job58738952.json)는
기존 계약 gate 통과, NRx 25/30 ms 탈락(최대 38.179/44.589 ms), 예열 4셀
복구 8 ms 관측 통과(최대 5.732/7.978 ms)다. 후자는 두 번째 run의 여유가
0.022 ms뿐이고, 같은 allocation의 별도 conventional-only 재자격에서는 GPU
단독 10.770 ms가 관측됐다. [Confirm83](../../results/softwall_same_gpu/confirm83_conv8_fourcell_job58738952.json)에서
4셀 실제 예약표를 8 ms로 바꾸자 conventional 8 ms 위반은 0이었으나 NRx
50 ms 위반이 release 621에서 2건 발생해 **전체 gate가 실패**했다. 예약표
축소가 다른 서비스 경로의 시간까지 바꾼다는 경고다. 8셀·복구 12 ms를
새 mode로 검사하려던 [Confirm84](../../results/softwall_same_gpu/confirm84_pretraffic_oom_failure.json)는
독립 PHY receiver 생성 중 GPU OOM으로 timed release가 0건이었다. 현재 구현은
8셀 메모리 조건부터 통과하지 못했다. 12셀은 `50+12×8=146<D155`(2 ms guard
뒤 7 ms 여유)라는 한
필요조건의 산술 결과이며, endpoint·feature·GPU 간섭·지속 부하 실증 전에는
가능하다고 단정하지 않는다. bound가 줄지 않으면 현재 계약의 정책 결정이
퇴화한다는 **feasibility envelope** 자체를 경계 결과로 보고한다. 줄어도
Confirm79b처럼 가상 경합에서 강한 greedy가 같은 답을 얻을 수 있으므로,
상한 조정만으로 알고리즘 노벨리티가 확보되지는 않는다.

온라인 `q_i^use`의 제때 commit 확률 보정과 외부 채널 이전, 미해결 AI의 fault continuation,
물리 co-run **서비스 상한**, 실제 MAC expiry, 강한 AI 인지 greedy 대비 공동 정책 성능은 남아 있다. 따라서 현재의
정확한 표현은 **검증 가능한 신규 시스템 가설**이지 **이미 확보한 노벨리티**가 아니다.

## 선행연구와의 경계

[YinYangRAN](https://jaayala.github.io/papers/24_infocom_2.pdf)은 GPU 기반 vRAN/ML 자원
배분 자체를 다뤘다. [CloudRIC](https://dspace.networks.imdea.org/bitstream/handle/20.500.12761/1795/mobicom24-final428_authors_v.pdf?sequence=1)은
여러 DU의 공유 가속기 접근과 무선 스케줄링을 이미 공동 제어했다. 따라서 단순히
"무선과 GPU를 함께 최적화"한다는 문구는 차별점이 아니다.
[ARCHES](https://arxiv.org/abs/2604.23397)는 AI와 conventional PHY
expert의 온라인 선택을 제시한다. [DARIS](https://arxiv.org/abs/2504.08795)는 MPS 기반
실시간 DNN 시공간 스케줄링을 다룬다. [OCUDU dApp](https://arxiv.org/abs/2609.07843)은
conventional 경로를 보존하는 실제 DU 통합을 보고한다. 이 선행연구가 있으므로 논문의
차별점은 이들의 기능을 단순 병치하는 것이 아니라 **조건부 PHY 복구 의무를 가진 여러
PUSCH의 무선 구조 가치·deadline·AI 허가를 한 온라인 결정으로 결합하고, 그 이득을
동등한 강한 결합 비교군에서 증명하는 것**이어야 한다. 현재 조사로 최초성은 단정하지 않는다.
