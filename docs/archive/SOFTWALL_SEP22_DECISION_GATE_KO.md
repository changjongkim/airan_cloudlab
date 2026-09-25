> **보관 사유 (2026-09-25):** 이 문서는 실험 이력으로 보존한다. 최신 스킴과 claim boundary는 `docs/current`의 원고·모델·production gate 문서가 권위 기준이며, 과거 GPU0 전용, MIG, CloudLab 또는 4-GPU pool 가정은 현재 논문의 근거가 아니다.

# SoftWall 2026-09-22 의사결정 gate

현재 후보 기여는 **관측 가능한 추가 PHY 구조 가치**를 사용해 NeuralRx 요청을 고르고,
선택된 모든 NeuralRx가 실패해도 마감을 지키는 conventional 복구 증명서를 유지하면서,
결과가 확정될 때마다 남는 시간을 GPU 완료 fence가 붙은 AI lease로 교환하는 제어기다.
MPS는 동일 GPU 실행 기반이다. MPS만으로 격리를 주장하지 않는다.

## 현재 판정

1. **물리 부품은 유의미하다.** Confirm76의 3셀·2 NeuralRx endpoint·500 release에서
   복구 전 원자 AI lease 170개가 실제 완료·반환됐고, 주입 동시 실패 뒤 3 conventional
   commit 54회, 관측 마감·선언 상한·credit 위반 0이었다. 이는 synthetic P150/D130
   표본 성공 경로이며 AI timeout 뒤 지속, WCET, 실제 DU 계약은 아니다.
2. **채널 가치 입력도 진전했다.** Confirm77은 −10/−9/−8/−7 dB synthetic 혼합에서
   train/test 각 8,000건을 분리했고, train-only 두 feature의 추가 구조 가치 예측이
   held-out 상수 예측보다 나았다(Brier 0.03773 대 0.04038). 실제 DU/TDL 및 적시
   commit을 포함한 `q_use`는 아직 보정되지 않았다.
3. **성능 알고리즘 노벨리티는 아직 없다.** Confirm78의 3셀 638개 실행 가능 상태와
   Confirm79b의 4셀 가상 경합 365개 실행 가능 상태에서 정확 공동 탐색과 동일 정보·
   all-fail 검사·AI recourse를 가진 1회 교환 greedy는 NRx 선택과 기대 AI 값이
   모두 같았다. Confirm79 첫 실행의 선택 차이 2건은 1e−16 부동소수점 동점 결함으로
   판정하고 수정 재실행했다. 더 약한 고정 예약만 이기는 결과는 논문 핵심 근거가 아니다.

## 다음 실험의 순서와 중단 기준

1. **4셀 물리 경로 자격 완료:** Confirm80의 P180/D155·2 endpoint·모든 실패 시 네
   conventional 작업·15 ms AI lease가 300 release의 frozen gate를 통과했다. 복구 전
   AI 완료·lease 반환 92회, 주입 all-fail 네 conventional 37회, 관측 마감·선언 bound·
   credit 위반 0이다. 이것은 다음 정책 비교의 실행 가능성만 판단한다.
2. **서비스 bound 자격을 먼저 검증:** 현재 예약값 `B_NRx=50 ms`,
   `B_conv_path=25 ms`가 어떤 실행 mode와 실패·간섭 범위를 포함하는지 명시한다.
   [기존 원자료 재감사](../../results/softwall_same_gpu/pre_confirm82_bound_mode_audit.json)에서
   2셀 Confirm56의 NRx/복구 host 최대 9.030/3.114 ms와 4셀 Confirm80의
   19.548/4.935 ms는 짧은 표본이었다. 반면 과거 1-endpoint·cap80 장시간
   Confirm43/44는 NRx 30 ms 초과 6건을 보였고, 그중 GPU front/post 자체가
   수십 ms였다. Confirm40의 미예열 첫 conventional GPU 경로도 최대
   35.303 ms였다. 따라서 다른 mode의 짧은 최대치로 8 ms 복구나 25/30 ms
   NRx를 선언하지 않는다. 사전 고정한 [Confirm82 protocol](../../results/softwall_same_gpu/confirm82_bound_campaign_protocol.json)의
   예열·지속 4셀 동시 실행 두 독립 seed×3,000 release [감사 결과](../../results/softwall_same_gpu/confirm82_bound_campaign_job58738952.json)는
   기존 50/25 ms 계약 gate를 모두 통과했다. NRx 응답 최대는 38.179/44.589 ms로
   후보 25/30 ms는 탈락했다. Conventional 전체 host 경로 최대는 5.732/7.978 ms로
   8 ms 후보를 관측상 통과했지만 두 번째 여유는 0.022 ms뿐이다. 같은 allocation의
   1셀 conventional-only 재자격에서 GPU 단독 최대 10.770 ms도 관측돼 8 ms를
   전역 상한으로 사용할 수 없다. [사후 지연 위치 감사](../../results/softwall_same_gpu/confirm82_tail_localization_posthoc.json)는
   NRx 30 ms 초과 release의 feature 완료→첫 dispatch 사이 22.721/28.249 ms
   빈 구간과 정상 범위의 endpoint worker GPU 약 1.7–2.1 ms를 보여 준다.
   admission·`prepare_neural_ipc`를 직접 분리 계측하기 전에는 원인을 단정하지
   않는다. 후보가 표본에서 통과해도 WCET가 아니므로
   cold start·worker 수명 변경·간섭·fault를 별도 mode로 격리하고,
   [Confirm83 실제 8 ms 예약표 protocol](../../results/softwall_same_gpu/confirm83_conv8_fourcell_protocol.json)의
   frozen 물리 캠페인을 거쳤다. [실제 결과](../../results/softwall_same_gpu/confirm83_conv8_fourcell_job58738952.json)는
   conventional 8 ms 위반 0·deadline miss 0이지만 NRx 50 ms 초과 2건(최대
   51.151 ms)으로 **전체 gate 실패**다. 세 독립 4셀 run의 지연이 모두 release
   약 620–623에서 다시 나타나므로 mode 전이/전처리 구간 계측이 우선이다.
   [Confirm84](../../results/softwall_same_gpu/confirm84_eightcell_conv12_protocol.json)는
   별도 8셀·복구 12 ms 실행 가능성 탐색이었으나, [독립 PHY receiver 생성 중
   GPU OOM](../../results/softwall_same_gpu/confirm84_pretraffic_oom_failure.json)으로
   timed release 시작 전 실패했다. 4셀 후보의 보증을 8셀에 계승할 수 없으며
   현재 구현의 resident 메모리 용량이 시간 조건보다 먼저 막는다.
   50+12×8=146 ms는
   D155에서 2 ms commit guard 뒤 7 ms가 남는 단순 산술일 뿐 12셀 실증이나
   AI 여유 보증이 아니다.
   [Confirm85 직접 계측](../../results/softwall_same_gpu/confirm85_tail_probe_job58738952.json)은
   새 seed의 release 616에서 NRx 44.038 ms tail을 재현했지만, admission·
   NeuralRx 준비·worker는 각각 정상 범위였고 첫 feature 시작이 release보다
   27.192 ms 늦었다. C82/83의 feature 완료 후 dispatch 전 지연과 동일 원인인지
   아직 모른다. [Confirm86의 frozen 진단](../../results/softwall_same_gpu/confirm86_stage_gc_probe_job58738952.json)은
   `NRx>30 ms` tail 재현 gate에는 실패했지만, [사후 시간 분해](../../results/softwall_same_gpu/confirm86_gc_overlap_posthoc.json)에서
   release 608의 채널 준비 30.271 ms와 2세대 Python GC 27.267 ms가
   거의 같은 구간임을 확인했다. 인과는 [Confirm87 ON/OFF 순서 반전
   개입](../../results/softwall_same_gpu/confirm87_gc_intervention_protocol.json)으로
   별도 판정한다.
   이후 Confirm87은 네 arm의 기존 안전 gate를 통과했지만 사전 인과 지표가
   worker 완료 뒤 controller 관측 대기를 누락해 **원인 gate 실패**로 남겼다.
   [Confirm88 장시간 독립 반복](../../results/softwall_same_gpu/confirm88_gc_off_long_bounds_job58743005.json)은
   GC OFF 4셀×5,000 release 두 seed에서 NRx 최대 21.223/20.116 ms,
   conventional host 최대 8.458/5.136 ms, 기존 계약·deadline 위반 0을
   보였다. 25/12 ms는 **관측 후보**, 8 ms 복구는 초과 1건으로 탈락이다.
   [Confirm90 실제 달력](../../results/softwall_same_gpu/confirm90_tight_calendar_gc_off_job58743005.json)은
   두 새 seed 중 첫 arm의 NRx 26.875 ms 한 건으로 **25/12 ms 전체 gate 실패**다.
   worker 게시 뒤 AI 반환·세 필수 conventional commit이 결과 관측을 늦췄다.
   [Confirm91 실제 30/12 ms 달력](../../results/softwall_same_gpu/confirm91_conv12_nrx30_gc_off_job58743005.json)은
   두 새 seed×3,000 release의 모든 frozen gate를 통과했지만 예열 4셀·GC OFF
   유한 표본 자격일 뿐이다. [Confirm89 독립 ON/OFF](../../results/softwall_same_gpu/confirm89_gc_causal_replication_job58743005.json)은
   첫 ON arm의 NRx 50 ms 초과 2건과 역순 pair의 사전 인과 기준 미달로 전체
   실패했다. 따라서 GC ON 기존 50/25 ms와 GC OFF 30/12 ms를 같은 계약으로
   묶지 않는다.
   낮은 상한을 얻어도 8셀 독립 receiver OOM과 강한 greedy 대비 동일 결정은
   별도 장벽이므로 경합 정책의 성능 우위를 아직 주장하지 않는다.
3. **AI bound도 먼저 검증:** 15 ms로 선언한 AI unit은 C91 실제 trace의 261,031
   RPC에서 host 최대 6.724/6.887 ms였다. [사후 원자료 감사](../../results/softwall_same_gpu/confirm91_ai7_physical_posthoc.json)는
   주입 동시 실패 930 release 모두 D153 전 AI 완료 최소 36개를 확인했다.
   C93의 7×15 ms 조건부 경합을 물리 경합으로 부르지 않는다. [Confirm95 frozen
   결과](../../results/softwall_same_gpu/confirm95_ai8_fourcell_job58746343.json)는
   실제 AI8 예산을 넣은 두 새 seed×1,000 release의 4셀 달력·동시 실패·guard·
   credit gate를 모두 통과했다. AI host 최대 6.790/4.999 ms,
   deadline·bound·guard·credit/fault 위반 0이다. 예열·GC OFF 표본 자격이지
   WCET나 7개 원자 선허가 검증은 아니다. 이 mode에서
   `30+4×12+7×8+2=136<D155`이므로
   기존 7-unit 경합 모델은 다시 퇴화한다. [Confirm96 계산 검증](../../results/softwall_same_gpu/confirm96_ai8_capacity_screen.json)은
   동일 C94 PHY quartet의 실행 가능 70/70건에서 공동·greedy·두 단계별 정책이
   AI 7개를 전부 수용해 공동 AI 우위 0임을 확인했다. 이 단일 사건 branch는
   종료한다. 새 경합은 실제 서비스가 긴 AI class 또는 개별 도착·마감에서
   찾아야 한다.
4. **공동 선택의 비교 대상 교정:** C93의 사후 PHY-first 단계별 대조에서는
   최소-NRx baseline보다 공동 선택의 AI 가치가 높았다. [사전 고정한 Confirm94의
   새 PHY seed](../../results/softwall_same_gpu/confirm94_independent_phy_screen.json)에서도
   실행 가능 70건 중 50건의 양의 차이가 재현됐다. 그러나 공동 exact와
   AI-aware 1-swap greedy는 70/70 같고, 공동 정책은 70/70 두 endpoint를 사용한다.
   무선 이득 최대화 단계별 baseline 대비 AI 평균 +0.0166 unit을 얻는 동안
   기대 무선 이득 평균 −0.0316 correct/4TB를 잃었다. AI 이득 24건 중 무선
   손실 ≤0.01은 2건이다. 따라서 최소-NRx 기준선 이득을 새 알고리즘의
   노벨리티로 세지 않는다. 실제 무선 비열등성 기준과 조건부 PHY·복구 의무의
   기여를 분리해 다시 시험한다.
5. **실제 경합과 paired 검증:** AI 도착 시각·개별 마감·물리 작업 길이·셀별
   `d_MAC`·두 endpoint queue를 사전에 고정한다. 양쪽에 같은 관측 정보와
   all-fail 안전 판정기를 준다. 공동 greedy 대 최소-NRx 및 max-radio 단계별
   정책을 독립 seed·ABBA 물리 실행으로 비교하되, 사전 정의한 무선 비열등성,
   적시 AI 완료량, deadline·bound·credit/fault를 모두 통과해야 성능을 주장한다.
   새 다중 사건 알고리즘은 공동 greedy 자체보다도 다른 안전한 행동을 내야
   추가 알고리즘 기여가 된다. 일반 AI packing이나 부동소수점 동점은 세지 않는다.
6. **보증 경로:** Confirm81은 AI GPU 완료 뒤 RPC 응답만 지연한 경우 별도 완료 표식으로
   lease를 회수하고 새 AI를 차단한 채 99개 후속 release를 처리했다. GPU hang이나
   marker가 없는 worker 장애 뒤 미완료 작업의 fence/credit 수명은 여전히 해결해야
   한다. co-run mode별 필수 conventional 전체 경로 상한과 실제 대상
   DU `d_MAC`를 검증한다. 현재 Aerial TDL-A 입력 호환성 실패를 해결하기 전에는
   production DU로 일반화하지 않는다.

이 순서를 통과하면 **PHY의 추가 가치와 여러 조건부 복구 의무를 AI lease와 함께
제어한 것**을 기여로 검토할 수 있다. 통과하지 못해도 안전한 MPS 공유 메커니즘은
남지만, strong greedy 대비 우월한 새 스케줄링 알고리즘이라고 부르지 않는다.
