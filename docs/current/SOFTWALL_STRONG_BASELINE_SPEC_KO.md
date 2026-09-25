# SoftWall 강한 결합 baseline의 공정 비교 명세 — 2026-09-21

> **상태 갱신:** 정책 우위 gate는 C102에서 종료됐다. 최종 비교는
> [논문 구조 검토본](SOFTWALL_PAPER_STRUCTURE_KO.md)의 `static safe`,
> `safe work-conserving`, `SoftWall substrate`에 동일 max-radio controller를 준다.
> 아래 내용은 그 결정을 만든 과거 baseline 설계 기록이다.

목적은 “MPS+복구 예약+짧은 AI 작업을 조합하기만 해도 같은 결과가 나는가?”를 반증 가능하게 만드는 것이다. [Confirm50](../../results/softwall_same_gpu/confirm50_requalified_gap_abba_protocol.json)은 같은 한 셀·한 endpoint에서 gap lease만 제거한 **구성요소 비교**다. 아래 전체 결합 baseline의 완료를 대신하지 않는다.

## 비교 대상의 정보와 안전장치

| 항목 | 결합 baseline에 제공할 것 | SoftWall과의 공정성 조건 |
|---|---|---|
| 무선 선택 | 실제 요청에서 deadline 전 관측 가능한 DMRS/채널·잡음 feature에 기반한 고정 NRx gate. 미래 CRC, 생성기의 true SNR, test trace 정답 사용 금지 | feature 추출과 gate 계산 시간도 deadline에 포함. 두 정책이 같은 feature·품질 목표를 받음 |
| Endpoint 선택 | 후보별 현재 queue tail과 최근 서비스 profile을 보고 가장 이른 feasible 완료를 선택 | queue-aware 선택은 최소 2개의 실제 endpoint가 있을 때만 비교. 한 endpoint 결과를 배치 기여로 주장하지 않음 |
| 복구 | 모든 수락 NRx에 conventional 실행 시간·commit guard를 고정 보수적 calendar로 예약. 실패가 cutoff 전에 확정되고 다음 release 전에 상한대로 끝낼 수 있으면 즉시 시작하는 선택도 허용 | 같은 fallback 서비스 bound, 동시 실패 가정, 단일 commit 및 buffer/credit 수명 규칙을 제공. 조기 CRC 실패 복구 자체를 SoftWall만의 이점으로 계산하지 않음 |
| Background AI | 같은 MPS cap·우선순위·worker 수·work unit, 40 ms host budget·35 ms socket timeout·2 ms guard로 bounded admission | AI 완료 유효 unit만 처리량으로 계산. 종료·timeout 뒤 물리 미완료 작업은 양쪽 모두 추적 |
| 배치 | 같은 GPU 수·SM/MPS cap·PHY/NRx 모델·cell 수·arrival trace·deadline | 전용 GPU, 단일 GPU, 다중 GPU 결과를 섞지 않고 동일 구성 안에서 paired 비교 |

Baseline은 채널 gate, queue-aware endpoint 선택, **고정** 복구 여유, 독립된 bounded AI 허가를 각각 실행한다. SoftWall의 추가 가설은 NRx 수락·복구 calendar·AI 허가량을 함께 결정해서 고정 결합의 과잉 예약 또는 충돌을 줄인다는 것이다. 두 정책 모두 late/stale/duplicate 결과 차단과 물리 완료 전 credit 점유를 지켜야 한다. 이 안전장치 없이 baseline만 실패하게 만드는 비교는 novelty 증거가 아니다.

**노벨리티 판정에는 더 강한 비교군을 추가한다.** 고정 calendar와 별도로, 동일한 관측 정보·서비스 상한·all-fail 안전 검사를 가진 AI 수요 인지 greedy 정책이 성공/실패 사건마다 복구 예약을 앞뒤로 재배치하고 AI unit을 허가하게 한다. 제안하는 [조건부 복구 credit 교환 정책](SOFTWALL_JOINT_SCHEME_PROPOSAL_KO.md)이 이 greedy와 같은 결정을 내리거나 같은 무선 효용·유효 AI 완료량을 얻으면, 단순 예약 재배치나 구성요소 결합을 알고리즘 기여로 주장하지 않는다. 작은 동시 요청 집합의 정확 최적해도 함께 계산해 남은 개선 여지를 확인한다.

**2026-09-22 구성요소 갱신:** [Confirm60](../../results/softwall_same_gpu/confirm60_low_gate_abba_job58729926.json)의
train-only 낮은 feature skip은 독립 두 쌍에서 무선 정답을 유지하고 AI를 +0.943%/+1.877%
늘려 사전 gate를 통과했다. [Confirm61](../../results/softwall_same_gpu/confirm61_corun_job58729926.json)의
조기 conventional은 동일 채널 200 release에서 정답을 유지하고 AI를 +0.806% 늘렸으며,
다른 셀 NRx가 host 관측상 미완료인 구간과 99회 겹쳤다. 두 부품은 강한 baseline에 준다.
이 둘과 Confirm59의 queue-aware 교차 라우팅을 한 controller에 묶은
[Confirm62](../../results/softwall_same_gpu/confirm62_combined_job58729926.json)는 200회 paired
release에서 무선 정답 214/400 동일, AI +2.071%, 교차 라우팅 117건, 조기 conventional
host 중첩 101건으로 사전 gate를 통과했다. 다만 endpoint queue는 idle이고 AI는 무선
후에만 실행했다. Confirm62 자체는 AI 수요 인지 greedy 복구 재배치, 미해결 복구 의무 중
AI 허가, 실제 GPU kernel overlap, production MAC expiry를 평가하지 않았다.

**후속 강화:** [Confirm69](../../results/softwall_same_gpu/confirm69_value_gate_job58734412.json)는
훈련 trace의 추가 무선 성공 구간만 NeuralRx에 보내는 value-bin gate를 저 feature gate와
두 새 seed·역순으로 비교했다. 더 많은 NRx를 생략하고 요청별 정답도 같았지만 AI 완료량은
+0.034%/−0.067%로 부호가 뒤집혔다. 따라서 이 gate도 강한 채널 선택 baseline의 한
후보로 보유하되, 게이트 결합만으로 AI 이득이나 공동 정책의 기여를 주장하지 않는다.
[Confirm68](../../results/softwall_same_gpu/confirm68_kernel_overlap_union_posthoc.json)은
허용한 사전 AI와 NeuralRx가 같은 GPU에서 실제 kernel 시간 구간을 겹쳐 실행하는 표본을
확인했다. [Confirm70](../../results/softwall_same_gpu/confirm70_conv_nrx_job58734412.json)은
조기 conventional과 다른 셀 NeuralRx의 실제 GPU kernel 중첩도 확인했다.

**AI 인지 greedy canary 갱신 (2026-09-22):**
[Confirm71](../../results/softwall_same_gpu/confirm71_greedy_retime_job58734412.json)은
조기 복구·비동기 사전 AI가 동일한 두 arm에서 복구 전 AI 한 unit 허가와 남은 단일 복구
credit의 greedy 자리 이동을 함께 켰다. 안전·무선 gate는 두 seed에서 통과했고 총 AI는
+1.321%/+0.319%였지만 두 요소의 효과가 섞였다.
[Confirm72](../../results/softwall_same_gpu/confirm72_retime_ablation_job58734412.json)는
복구 전 AI 허가를 양쪽에 주고 자리 이동만 분리했다. ON은 각 seed에서 14번 옮겼으나
AI는 +0.187%/−0.420%로 부호가 갈렸다. 단순 retime은 강한 baseline의 기능으로
유지하며, 그 자체의 독립 처리량 기여를 주장하지 않는다. 두 비교 모두 P150/D130,
동질적인 AI unit·idle queue의 진단 부하다. 전체 공동 정책, 원자 AI lease,
co-run 서비스 상한과 실제 MAC expiry는 아직 없다.

**채널 입력의 현재 자격:** 이전 499/500 중복 seed calibration은 무효다. 별도의
[Confirm57](../../results/softwall_same_gpu/confirm57_disjoint_feature_gate_job58729926.json)은
완전히 분리된 train/test 각 500개, prewarmed feature에서 train threshold를 고정한 뒤
held-out NRx skip 215/500, neural-only 정답 손실 2/500, feature GPU 시간 최대 3.453 ms로
새 사전 gate를 통과했다. 이는 `channel_estimate_power`를 *현재 synthetic 고정 잡음
workload의 오프라인 baseline 후보*로 허용한다. 실제 요청에서 feature가 준비되는 시각과
host-to-decision 비용, 온라인 gate 실행, Aerial TDL 호환성은 아직 검증하지 않았다.
비교 때에는 baseline과 공동 정책 모두 같은 threshold·feature 계산 비용을 받게 한다.

**결정 차이의 현재 판정:** [오프라인 단순 교환 대조](../../results/softwall_same_gpu/offline_exchange_example_result.json)에서
고정 예약은 AI 작업을 못 받았지만 AI 수요 인지 greedy 재배치와 정확 공동 탐색은 모두
같은 AI 작업을 받았다. 따라서 복구 credit 뒤로 이동 하나를 제안 스킴만의 성과로
계산하지 않는다. 별도 [AI 묶음 선택 대조](../../results/softwall_same_gpu/offline_generic_packing_control_result.json)의
정확 탐색 우위도 일반 packing에서 생겨 AI-RAN 노벨리티 근거가 아니다.

Confirm53의 조기 복구 비교는 현재 controller의 병목을 고치는 **내부 메커니즘 ablation**이다. 성공하면 이 동작을 결합 baseline에도 제공한 뒤, 동시 요청에서 복구 시점·NRx 수락·AI 허가량을 함께 정하는 정책이 별도 이득을 내는지 비교한다. 한 셀에서 `latest`보다 `early_if_clear`가 낫다는 결과만으로 공동 제어의 노벨리티를 주장하지 않는다.

Confirm53의 최초 한 셀 구현은 조기 실행 시 calendar에 남은 늦은 예약 시각을 이동하지 않았다. 다중 요청에 그대로 적용하면 물리적 conventional 경로가 다른 요청의 예약 구간에 겹칠 수 있다. 이후 `FallbackCalendar.retime_earlier`와 `DartRuntime.start_fallback_early`를 추가해 원자적으로 앞선 빈 구간으로 예약을 이동시키고, 충돌하면 원래 예약을 유지하도록 했다. 동시 이동·충돌 단위 테스트 14개가 통과했고, [Confirm54](../../results/softwall_same_gpu/confirm54_retimed_recovery_canary.md)의 한 셀 GPU 통합 gate도 통과했다. 이 사실은 **복구 예약표의 논리적 안전성**에 한정된다. 다른 NeuralRx GPU 작업과 conventional이 동시에 실행될 때의 25 ms 서비스 상한과 미완료 입력 buffer 수명은 별도로 검증해야 한다.

## 두 셀 비교를 시작할 수 있는 계약

현재 단일 conventional lane의 개별 `B_NRx=50`, 예약용 `B_conv=25`, guard `=2` ms를 유지한다. `B_conv_path`를 host가 복구를 시작하기로 한 시각부터 commit 반환까지의 **전체 경로** 상한이라고 하면, 두 셀 모든-실패의 필요조건은 `2×B_conv_path+2 ≤ D`, 지속 부하에는 `2×B_conv_path ≤ P`가 필요하다. 현 `두 NRx 대기 → 복구` 순서의 deadline 조건은 **`B_pair+2×B_conv_path+2 ≤ D`**다. 여기서 `B_pair`는 두 요청 중 늦게 끝나는 NRx의 *동시 실행·후처리·commit 반환 포함* 상한이다. `B_conv_path=25 ms`와 `B_pair=50 ms`를 각각 별도로 자격 검증한다는 **가정 아래에서만** 수치는 `50+50+2=102 ≤ D`가 된다. 개별 `B_NRx=50 ms`만으로 `B_pair=50 ms`를 가정할 수 없다. 기존 protocol의 25 ms gate는 `conventional_gpu_ms`에만 적용했고 NRx 응답 시각은 commit 호출 전이므로 두 전체 경로 상한 모두 아직 보증이 아니다.

`B_conv_path=25 ms`를 **전체 경로 상한으로 검증했다는 가정** 아래에서 P150/D130의 현 순서가 허용하는 공동 상한은 `B_pair ≤ 130−2×25−2 = 78 ms`다. 두 작업이 MPS에서 직렬화돼 각 50 ms를 모두 쓰는 허용 사례라면 `B_pair=100 ms`이고, 이 가정에서도 복구 포함 `100+50+2=152>D130`이어서 **현 순서**는 계약을 지킬 수 없다. 한 NRx가 끝난 뒤 다른 NRx가 미완료인 동안 복구를 시작하는 다른 정책은 NRx–conventional 동시 서비스 상한의 별도 증명이 필요하다. [Confirm55 원자료](../../results/softwall_same_gpu/raw/confirm55_two_endpoint_job58693395_controller.json)의 실제 1,000회에서 두 NRx의 **commit 호출 전** 관측 최대는 13.503 ms였지만, 이 표본 최댓값은 전체 공동 실행의 방어 가능한 상한이 아니다. 새로운 비교는 동시 NRx·fallback·AI phase별 간섭 상한을 먼저 정해야 한다.

아래 수치표는 `B_conv_path=25 ms`를 **가정한 조건부 계산**이며 현재 GPU-event gate가 그 가정을 입증하지 않는다.

| 2셀 계약 | 계산상 판정 | 다음 행동 |
|---|---|---|
| P45/D80 | `2×25>P`이며 wait-both에는 `B_pair≤28 ms` 필요 | all-fail 지속 부하 불가능. 이 점에서 강한 baseline 우위를 주장하지 않음 |
| P90/D80 | 지속 복구는 가능하나 wait-both에는 `B_pair≤28 ms` 필요 | 개별 50 ms 상한 두 개만으로 보증 불가. 첫 fallback cutoff도 release+28 ms이므로 별도의 물리 완료/포기 계약 필요 |
| P90/D130 | 지속 복구는 가능하고 wait-both에는 `B_pair≤78 ms` 필요. 다음 release가 이전 late recovery 구간과 겹침 | 요청 간 동시 실행의 mandatory bound를 별도 측정하기 전에는 비교 부하로 사용하지 않음 |
| **P150/D130** | 두 복구를 78–103/103–128 ms에 둘 수 있고 wait-both에는 `B_pair≤78 ms` 필요 | 두 endpoint·IPC·동시 실패 bring-up의 첫 부하. 실제 co-run bound와 buffer 수명을 검증한 뒤 비교로 승격 |

P150/D130에서는 같은 release의 첫·둘째 요청에 **서로 다른 fallback latest start**를 줘야 한다. 둘 다 103 ms로 예약하면 capacity 1 calendar는 하나를 거절한다. 78/103 ms의 두 예약은 아래 Confirm55에서 실제 GPU로 검증했다. 단, 그 실험은 두 NRx의 물리 완료를 확인한 다음 conventional을 시작했으므로 **NRx와 conventional이 겹칠 때의 25 ms 서비스 상한은 검증하지 않았다.** 이후 baseline과 joint 정책은 같은 검증된 상한과 조기 복구 primitive를 사용해야 한다.

[Confirm55 물리 bring-up](../../results/softwall_same_gpu/confirm55_two_endpoint_bringup.md)은 P150/D130의 두 cap40 NeuralRx endpoint를 같은 GPU에서 동시에 dispatch하고, cap20 AI worker를 남는 구간에 둔 1,000회에서 frozen gate를 통과했다. RAN 1,958/2,000 정답·miss 0, NRx max 13.503 ms(<50), conventional GPU bound 위반 0, AI 29,910 units, 잔여 credit 0이었다. 그러나 endpoint별 worker 시각이 없어 두 GPU kernel의 **실제 overlap 구간은 별도 계측되지 않았고**, 두 셀 동시 자연 fallback은 1회뿐이었다. 이는 P150/D130 고정 배선의 통합 확인이지 queue-aware 선택, worst-case correlated recovery 또는 강한 baseline 대비 joint policy의 이득이 아니다.
구현 감사에서 `DartRuntime.submit`의 기존 predicted-finish 검사가 expiry−guard만 보고 **요청별 fallback latest start보다 NRx가 먼저 끝나는지는 확인하지 않는** 문제를 발견했다. 모든 라우팅 정책에 `predicted_NRx_finish ≤ cutoff`를 추가하고, 실제 endpoint 예약 때도 같은 제한을 원자적으로 다시 검사하도록 수정했다. 이 검사를 포함한 런타임 단위 테스트 15개가 통과했다. 다중 셀 controller는 각 요청에 78/103 ms cutoff를 정확히 배정해야 하며, GPU에서 관측한 실제 NRx 시간과 이 예측 상한의 일치 여부는 별도 자격 검증이다. 이 수정은 Confirm54의 GPU 실행 **이후** 이루어졌으므로 Confirm54를 수정 코드의 GPU 자격 검증으로 사용하지 않는다.

또한 기존 한 단계 `submit`은 NRx를 거절하면 conventional을 즉시 실행한다고 가정해 **그 mandatory-only 작업의 calendar credit을 보유하지 않는다.** 두 셀에서는 이것이 필수 경로끼리 충돌하는 원인이 된다. 새 `reserve_mandatory`→`admit_nrx` 두 단계 API는 먼저 모든 요청의 conventional 구간을 확보하고, cutoff 안에 끝날 수 있는 NRx만 선택적으로 붙인다. optional이 거절돼도 mandatory credit이 유지된다. 두 셀에 서로 다른 cutoff와 endpoint를 붙이는 사례를 포함한 **17개 단위 테스트**가 통과했고, 기존 경로의 **10,000회 fault 회귀 검사**도 통과했다. 후자는 새 두 단계 API의 모든 고장 상태를 검증한 시험이 아니다. Confirm55는 이 API를 GPU의 고정 2셀 배선에 사용했으나, `reserve_mandatory`가 실패하는 overload와 자유로운 endpoint 재배치는 평가하지 않았다. 같은 overload 규칙을 baseline과 joint 정책에 적용해야 한다.

## 구현·평가 순서

1. **정보 사용 시각을 고정:** 어떤 feature가 request release 전에 알려지고 어느 계산이 release 이후 수행되는지 trace에 기록한다. 현재 synthetic `snr_db`는 생성기 입력이므로 온라인 gate에 그대로 주지 않는다.
2. **검증 trace에서 baseline 튜닝:** gate threshold, queue profile, 고정 복구 여유, AI unit/budget을 사전 지정한 범위에서 탐색한다. Test seed·채널·실행 순서는 튜닝에 사용하지 않는다.
3. **단일 GPU 확인:** Confirm50으로 고정 복구+bounded AI에 gap lease를 더한 순증과 안전성 비용을 본다. 이는 전체 baseline의 하한이 아니라 메커니즘 한 개의 비교다.
4. **다중 endpoint 비교:** 동일한 full-PHY 요청과 최소 두 endpoint에서 독립 정책과 joint 정책을 같은 GPU 예산으로 실행한다. 동시 fallback·burst·co-tenant phase 변화를 포함하고 순서와 seed를 교대한다.
5. **사전 판정:** 모든 정책의 RAN deadline miss, 실제 fallback 시작 지연, NRx/mandatory bound, radio utility의 paired 비열등성, AI 완료량, admission 거절·credit 잔여를 함께 보고한다. SoftWall이 같은 안전·무선 요구에서 baseline보다 더 많은 AI를 완료하지 못하면 공동 제어 우월성 주장을 철회한다.

현재 GPU 통합은 2셀·2 endpoint의 **고정 배선**까지 평가했다. 요청 소유 PHY 입력과 endpoint IPC buffer를 분리한 queue-aware 재배치, 반복 동시 실패, 실제 worker kernel overlap 계측은 아직 없다. Confirm55의 `fallback_actual_start_ns`는 명칭과 달리 **host가 복구를 시작하려고 한 시각**이며 GPU kernel 시작을 직접 관측한 값이 아니다. 비교할 새 정책에는 같은 host 시각과 GPU 실행 시각의 구분된 계측이 필요하다. 원 계획의 0.5/1 ms slot 보장은 P90/D80 또는 P150/D130 결과와 별도 게이트다.
