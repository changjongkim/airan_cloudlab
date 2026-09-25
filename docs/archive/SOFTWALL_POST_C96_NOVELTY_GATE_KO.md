> **보관 사유 (2026-09-25):** 이 문서는 실험 이력으로 보존한다. 최신 스킴과 claim boundary는 `docs/current`의 원고·모델·production gate 문서가 권위 기준이며, 과거 GPU0 전용, MIG, CloudLab 또는 4-GPU pool 가정은 현재 논문의 근거가 아니다.

# SoftWall C96 이후 노벨리티 판정과 다음 실험 gate — 2026-09-22

**판정:** MIG OFF·MPS 4셀에서 복구 credit의 원자 재배치와 AI lease는 실제로
동작한다. 그러나 현재 **동시에 보이는 7개 AI 작업·단일 NRx 결과 사건**의 공동
정책 성능 우위는 없다. 이 branch에서 seed를 더 늘리거나 7개 작업만으로 GPU
ABBA를 수행하지 않는다. 목표는 그대로 *허용된 AI 작업 안에서 RAN deadline과
무선 품질을 유지하고, 남는 GPU 자원으로 적시 AI 완료량을 늘리는 것*이다.

| 근거 | 판정 |
|---|---|
| [C91 실제 30/12/15 달력](../../results/softwall_same_gpu/confirm91_conv12_nrx30_gc_off_job58743005.json) | 예열·GC OFF 4셀 두 seed의 물리·안전 표본 gate 통과. WCET/DU 보증은 아님 |
| [C93 7-AI 모델](../../results/softwall_same_gpu/confirm93_seven_ai_capacity_screen.json) | 15 ms 선언으로 만든 조건부 용량 압력은 있지만 exact=공동 1-swap greedy |
| [C94 독립 PHY](../../results/softwall_same_gpu/confirm94_independent_phy_screen.json) | 최소-NRx 단계별 정책보다 AI 기대값 평균 +0.2098, 그러나 [NRx 수 분해](../../results/softwall_same_gpu/confirm94_endpoint_count_posthoc.json)상 양의 차이 50건 중 48건은 단순한 두 번째 endpoint 사용. max-radio 대비 평균 AI +0.0166에 기대 무선 이득 −0.0316 correct/4TB |
| [C91 AI 물리 사후 대조](../../results/softwall_same_gpu/confirm91_ai7_physical_posthoc.json) | 두 seed의 동시 실패 930 release 모두 D153 전 AI 완료 최소 36개. 7×15 ms 경합은 현 물리 작업을 대표하지 못함 |
| [C95 실제 AI8 달력](../../results/softwall_same_gpu/confirm95_ai8_fourcell_job58746343.json) | 새 두 seed×1,000 release의 출처·계약·안전·coverage gate 전부 통과, AI host 최대 6.790/4.999 ms. 예열·GC OFF의 유한 표본 자격 |
| [C96 AI8 모델 재검사](../../results/softwall_same_gpu/confirm96_ai8_capacity_screen.json) | 동일 독립 PHY 70개 실행 가능 상태에서 공동·greedy·두 단계별 정책 모두 AI 7개 전부 수용. AI 우위 0 |
| [C97 실제 Qwen 경합 canary](../../results/softwall_same_gpu/confirm97_qwen_fourcell_failure_job58747070.json) | Qwen2.5-1.5B b1c16 prefill cap20은 첫 arm에서 동시 실패 63/63 release의 D153 전 완료를 5–6개로 낮췄지만 NRx30 위반 705건. 둘째 arm은 Qwen GPU 36.246 ms 단위가 RPC timeout35를 넘긴 직후 복구 전 AI 완료 미확인으로 controller가 중단. 전체 **FAIL**. 실제 경합 단서는 얻었으나 새 mode 자격 없음 |
| [C98 Qwen 새 mode](../../results/softwall_same_gpu/confirm98_qwen_nrx45_ai50_job58747070.json) | C97 실패 뒤 고정한 실제 NRx45/복구12/AI50·RPC45 달력에서 새 seed 두 arm×400 release의 frozen 안전·coverage gate **PASS**. NRx 최대 43.412/42.818 ms, 복구 최대 9.161/4.868 ms, AI host 최대 28.481/28.041 ms, deadline·bound·credit/fault 위반 0. 동시 실패 55/64 release에서 D153 전 Qwen 5–6개 반환. 예열·GC OFF·연속 backlog의 유한 표본 결과 |
| [C98 분기별 사후 감사](../../results/softwall_same_gpu/confirm98_branch_posthoc.json) | 강제 실패 주입이 없는 release와 NRx 0개 release에서도 D153 전 Qwen은 최대 **6개**. 따라서 7번째 작업의 탈락은 RAN 복구 때문에 생긴 조건부 경합 증인이 아니다. 6번째 완료율 차이는 보이지만 짝지은 고장 개입·정책 대조 전에는 인과나 노벨리티 근거로 쓰지 않는다 |
| [C99 동일 PHY 고장 ON/OFF ABBA](../../results/softwall_same_gpu/confirm99_qwen_fault_abba_job58747070.json) | 같은 seed·입력의 N/F/F/N 각 200 release. 일치한 두 NRx 수락 26+26개 release에서 고장 ON−OFF의 D153 전 Qwen 반환은 −22/−9개로 양쪽 순서에서 감소. 그러나 마지막 OFF arm 첫 timed release의 NRx45가 **48.605ms**로 1건 초과해 frozen 안전·coverage 전체 **FAIL**. 이 차이를 자격 mode의 성능/인과 결론으로 승격하지 않는다 |
| [C100 관측 우선 mode](../../results/softwall_same_gpu/confirm100_observe_first_qwen_job58747620.json) | 별도 제어기에서 비동기 Qwen 실행 중 NRx 결과를 필수 conventional보다 먼저 소비했다. 새 seed 두 arm×1,000 release의 frozen 출처·안전·coverage gate **PASS**. NRx 최대 12.968/14.562ms, 복구 host 최대 5.251/4.230ms, AI host 최대 26.083/26.801ms, deadline·bound·guard·credit/fault 위반 0. 혼합 경로의 관측 순서 모두 충족. 예열·GC OFF·4셀·Qwen class의 유한 표본 자격이며 C99 실패 수정이나 정책 우위 아님 |
| [C101 관측 우선 mode의 고장 ABBA](../../results/softwall_same_gpu/confirm101_observe_first_fault_abba_job58747620.json) | 새 동일 PHY seed의 N/F/F/N 각 300 release. 네 arm 모두 frozen 출처·안전·coverage gate **PASS**. 일치한 두 NRx 수락 release가 각 짝 49개이며, 강제 실패 ON−OFF 적시 Qwen 반환은 −7/−14개. 복구가 허용 AI 작업의 적시 완료를 밀어내는 방향이 양쪽 순서에서 재현됐다. 한 seed의 고장 메커니즘 대조이며 새 정책/AI-RAN 앱/AI 가치 우위가 아님 |

## 연구 기여의 정확한 후보

현재 구현된 시스템 기여 후보는 **조건부 복구 증명서를 유지하면서 AI 실행권을
원자적으로 발급·회수하는 same-GPU 제어기**다. MPS cap은 실행 도구이며
단독 격리나 새 스케줄링 알고리즘이 아니다. 추가 알고리즘 기여 후보는
**개별 PHY 결과·AI 도착·마감이 바뀔 때 `(NRx 선택·endpoint, 관측 순서,
복구 순서·cutoff, AI lease)`를 함께 다시 결정하는 온라인 정책**이다.
현재 single-event exact는 강한 joint greedy와 같으므로 그 탐색 자체에
새로움을 부여하지 않는다.

C98은 실제 Qwen 부하를 넣으면 안전한 4셀 mode와 GPU 수요 압력이 동시에
가능하다는 첫 좁은 증거다. 그러나 이 workload는 일반 Qwen co-tenant이며
강제 실패 주입이 없어도 일곱 번째 단위가 마감 전에 끝나지 않는다. 다음 핵심 질문은
**6번째 단위의 적시 완료를 복구 위험에 따라 다르게 허가해야 하는 상태가
있는가**다. 먼저 동일 PHY 입력의 고장 ON/OFF 짝비교로 복구가 실제로 그
단위를 밀어내는지 검증하고, 그다음 개별 도착·마감·가치를 가진 AI-RAN
작업에서 안전한 서로 다른 계획과 강한 baseline 차이를 보여야 한다.
후속 C99의 역순 고장 개입은 그 6번째 단위가 복구에 민감하다는 단서를
재현했지만, 같은 run의 NRx45 위반으로 전체 자격에는 실패했다. [사후 시간
분해](../../results/softwall_same_gpu/confirm99_tail_posthoc.json)에서 worker는
release+16.990ms에 결과를 게시했고, controller가 Qwen RPC 완료와 세 필수
conventional commit을 거친 뒤 +48.605ms에 관측했다. GPU kernel만의 긴 tail이
아니라 **관측 순서가 서비스 상한에 들어가는 구조적 경로**다.
[C100](../../results/softwall_same_gpu/confirm100_observe_first_qwen_job58747620.json)은
NRx 결과 관측을 앞당긴 별도 mode를 두 새 seed×1,000 release에서 통과시켰다.
기존 45 ms 선언을 유지했지만, C99와는 PHY seed·제어 순서가 달라 두 최댓값을
paired 개선율로 계산할 수 없다. Qwen은 AI-RAN 전용 작업이 아니므로 이
순서 변경만으로 공동 정책 노벨리티가 성립하지 않는다.
[C101](../../results/softwall_same_gpu/confirm101_observe_first_fault_abba_job58747620.json)은
그 새 mode 안에서 강제 동시 실패가 적시 AI 반환을 줄이는 물리 효과를
안전 gate와 함께 재현했다. 이로써 **경합은 실제이고 조건부 복구가 영향을
준다는 점**까지 확보했다. 다음 novelty gate는 서로 다른 안전한 온라인
`(NRx 선택·관측·복구·AI 작업 허가)` 행동을 만들고, 같은 정보·안전 검사·
AI recourse를 가진 강한 joint greedy와 max-radio/최소-NRx 기준선보다
무선 비열등성과 적시 AI 가치가 유지되는지 비교하는 것이다. 단순 고장 ON/OFF
차이를 새 정책의 성능 이득으로 환산하지 않는다.

온라인 상태는 현재 시각, TB별 실제 MAC expiry와 관측 feature/보수적
`q_i^use`, endpoint·IPC·GPU 완료 fence, 미확정 복구 의무, 지금까지 도착한
AI 작업의 길이·마감·가치, 자격을 얻은 실행 mode별 host 전체 경로 상한으로
구성한다. 미래 AI 도착, 미관측 NRx 결과, held-out CRC, 실현 GPU 서비스
시간은 입력에서 제외한다. 매 사건에는 이미 제출한 비선점 작업을 유지하고,
허용한 AI를 지금 제출해도 모든 미확정 NRx가 실패·지연하는 분기에서 모든
conventional commit이 `d_MAC` 전에 끝나는 후보만 남긴다. 그 후보 안에서
사전 고정한 무선 비열등 조건 아래 적시 AI 가치를 최대화한다. 복구 달력 변경과
lease 발급은 한 transaction으로 확정하며 실제 GPU 완료 fence 전에는 credit을
반환하지 않는다.

## 다음 실험을 여는 조건

1. **물리 수요 자격:** 실제 사용할 AI application/class의 모델·입력 크기·배치,
   개별 도착·마감·가치와 MPS cap을 먼저 고정한다. 4셀 radio co-run에서
   AI의 `admission→RPC 반환` 상한과 연속 burst, NRx `release→controller 관측`,
   conventional `결정→commit 반환`, GC·시작/재시작 mode를 각각 측정한다.
   단순 `--repeats`를 키워 만든 길이는 stress 진단으로만 쓰고 실제 AI-RAN
   workload의 성능 근거로 승격하지 않는다. 새 bound는 그 값으로 **실제 달력**을
   운영한 독립 seed·동시 실패·간섭 시험을 통과해야 한다.
2. **경합 증인:** 실측 자격 mode에서 허용 AI 수요가 모든 실패 복구와 함께
   들어가는지 계산하고, 실제 반환 시각·작업별 deadline으로 검증한다.
   모두 들어가면 이 workload의 정책 비교를 중단한다. 들어가지 않는다면
   최소한 두 개의 안전한 계획이 서로 다른 `(NRx, 복구, AI 허가)` 행동을
   갖는 사건을 독립 trace에서 보여 준다. 일반 AI packing, 임의로 부풀린
   bound, 부동소수점 동점은 증인에서 제외한다.
3. **강한 비교:** 고정 low-feature gate, PHY-first 최소-NRx, PHY-first
   max-radio를 모두 구현하고 동일 정보·동일 all-fail 안전 검사·동일 정확한
   복구/AI recourse를 제공한다. 공동 1-swap greedy는 현 결합 정책의 효율
   구현이자 작은 상태의 exact와 대조하는 기준이다. 새 다중 사건 알고리즘의
   추가 우위는 이 joint greedy보다도 보여야 한다.
4. **사전 성능 gate:** 실제 DU의 `d_MAC`와 무선 비열등 마진을 먼저 정한다.
   동일 PHY/AI trace의 ABBA·독립 seed에서 radio miss·bound·credit/fault
   위반 0, max-radio 정책 대비 사전 무선 비열등성, 적시 AI 완료 가치의
   반복 양의 차이를 모두 요구한다. `q_i^use`를 균등화하거나 조건부 복구를
   항상 필수로 고정한 대조에서 이득 원인을 분리한다. 어느 하나 실패하면
   공동 정책 **성능 노벨리티** 주장을 중단한다.

현 대상 DU의 `d_MAC`, GC ON·cold/restart 수명 mode, NRx cutoff 뒤 미완료
GPU 작업의 fence/IPC 수명, 8셀 receiver OOM, 외부 Aerial TDL-A 호환성은
별도 미완료 조건이다. 이들이 남아 있는 동안 synthetic P180/D155의 표본
통과를 production hard deadline 보장으로 쓰지 않는다. 성능 gate가 계속
음성이면 연구 방향은 **mode별 feasibility envelope**로 바꾸되, 논문에는
경계 예측 모델과 독립 물리 검증이 필요하다.
