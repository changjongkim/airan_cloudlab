# SoftWall endpoint·sharded-home multi-GPU envelope

**상태:** 2026-09-24 과거 checker 보존, control wall-bound 교정 v13과 C138--140 완료  
**범위:** Endpoint-only 구조와 disjoint sharded-home multi-GPU 구조의 mode별 envelope

## 결과

`P180/D155`, recovery 12 ms, guard 2 ms, AI class 35/40/65/75 ms의
`GPU 1/2/4 × cell 4/8/12` grid를 계산했다.

| Cell 수 | 1 GPU | 2 GPU | 4 GPU | 지배 경계 |
|---:|---|---|---|---|
| 4 | QSU | QSU | QSU | 자격화된 현재 mode |
| 8 | MI | MI | MI | GPU0 receiver residency OOM |
| 12 | MI | MI | MI | 8-cell OOM의 동일 single-home 구조 |

4-cell의 mandatory recovery demand는 `4×12=48 ms`, guard 뒤 capacity는 153 ms라
all-fail base slack은 105 ms다. 모든 평가 AI class가 이 구간에 이미 들어간다. 따라서
이 grid에는 `W < B_eff ≤ W+Delta`라는 용량 필요조건의 exchange-only AI class가 없다. 이는 C113/C115에서
atomic exchange가 실제로 반복 실행됐지만 safe work-conserving 대비 추가 token 이득이
거의 0이었던 결과와 일치한다.

8/12-cell은 시간 산술만 보면 각각 base slack 57/9 ms이고 일부 AI class가 조건부 회수
구간에 들어간다. 그러나 현재 구조는 모든 Aerial receiver와 conventional recovery를
GPU0에 유지한다. C84가 8 receiver를 timed traffic 전에 OOM으로 기각했으므로 이 timing
schedule은 counterfactual로 표시하고 최종 상태는 `MI`다. Endpoint GPU 수를 늘려도 home
receiver memory는 줄지 않는다.

## 설계 결론

현재 endpoint-offload 구조에서 4-GPU 확장은 transport/lifecycle 일반화에는 유효하지만
cell scale-out 해법은 아니다. 8/12 cell을 검증하려면 다음 v2가 필요하다.

1. RAN home receiver와 conventional recovery lane을 여러 GPU에 분산한다.
2. 각 home GPU가 local all-fail certificate를 유지한다.
3. Global coordinator는 shared endpoint/ring과 GPU별 AI lease의 product state만 원자화한다.
4. GPU별 certificate의 합성 정리와 cross-home fault를 먼저 증명한다.
5. 새 memory/service profile을 자격화한 뒤에만 8/12-cell 물리 grid를 실행한다.

이 v1은 synchronized batch의 deterministic model이며 일반 arrival queueing model이나
production WCET가 아니다.

## Sharded-home v2 prospective 결과

같은 checker를 각 GPU가 별도 RAN home/recovery lane을 갖는 구조로 확장했다. 현재 물리
구현이 없으므로 multi-home timing/lifecycle profile은 의도적으로 `UQ`로 둔다.

| Cell 수 | 1 home GPU | 2 home GPU | 4 home GPU |
|---:|---|---|---|
| 4 | QSU `[4]` | UQ `[2,2]` | UQ `[1,1,1,1]` |
| 8 | MI `[8]` | UQ `[4,4]` | UQ `[2,2,2,2]` |
| 12 | MI `[12]` | MI `[6,6]` | UQ `[3,3,3,3]` |

대괄호는 home별 cell 수다. 현재 A100의 경험적 receiver residency 한계를 home당 4개로
두면 2-home은 8 cell, 4-home은 12 cell의 memory 산술을 만족한다. 그러나 이를 QSU로
승격하지 않았다. GPU별 Aerial/Qwen/MPS co-run bound, global endpoint coordinator,
cross-home fault와 product-credit lifecycle이 아직 물리 자격을 받지 않았기 때문이다.

이 예측은 다음 물리 실험점을 사전 선택한다.

- 내부점: 2 home×2 cell = 4 cell
- memory 경계: 2 home×4 cell = 8 cell
- scale 점: 4 home×3 cell = 12 cell
- 실패 대조: 2 home×6 cell = 12 cell

각 점은 별도 source/protocol을 고정하고, home별 local certificate와 global shared-credit
검사를 모두 통과해야 한다.

## Sharded-home v3 물리 판정

C121/C122의 실제 구조는 shared endpoint pool이 아니라 home별 NRx endpoint 두 개와
Qwen worker 하나를 가진 disjoint pool이다. 기존 v2 입력과 checker hash는 보존했다.
새 checker v2는 `home_endpoint_bounds_ms`를 받아 각 home의 endpoint queue를 따로 계산한다.

| Cell 수 | 1 home GPU | 2 home GPU | 4 home GPU |
|---:|---|---|---|
| 4 | **QSU** `[4]` | UQ `[2,2]` | UQ `[1,1,1,1]` |
| 8 | **MI** `[8]` | **QSU** `[4,4]` | UQ `[2,2,2,2]` |
| 12 | **MI** `[12]` | **MI** `[6,6]` | **QSU** `[3,3,3,3]` |

C121은 2-home×4셀의 두 독립 arm, 총 5,440 TB를 release 오차 0 ns로 실행했고 safety
위반이 없었다. C122는 4-home×3셀의 두 arm, 총 8,160 TB에서 같은 판정을 얻었다. 따라서
사전에 UQ였던 두 점만 QSU로 승격했다. 직접 실행하지 않은 `[2,2]`, `[1,1,1,1]`,
`[2,2,2,2]`는 mode별 co-run 자격 원칙에 따라 UQ로 남겼다.

이 결과는 disjoint certificate의 수평 합성이다. Shared endpoint 또는 global AI lease가
있는 경우에는 v3의 곱 구조를 그대로 사용할 수 없고 multi-resource broker가 필요하다.

## Sharded-home v4: bound와 executor도 mode다

후속 실험은 v3의 2-home×4-cell QSU를 두 단계로 반증했다. C124에서 conventional
host-to-commit이 `12.535096 ms`로 conv12 계약을 넘었고, C125에서는 conv25 산술이
가능해도 executor가 뒤 ready class를 먼저 실행하려 해 live credit과 충돌했다.
Certificate-ordered executor를 쓴 C126은 8개 arm, 21,760 TB에서 모든 safety gate를
통과했다.

따라서 v4는 같은 topology도 `(conv bound, executor)`에 따라 다른 mode로 나눈다.

| Mode | 상태 | 근거 |
|---|---|---|
| 1-home×4-cell conv12 | QSU | C113 exact mode |
| 2-home×4-cell conv12 legacy executor | UQ | C121 pass 뒤 C124 bound 반증 |
| 2-home×4-cell conv25 class-order executor | UQ | C125 ordering fail-closed |
| 2-home×4-cell conv25 certificate executor | QSU | C126 8/8 arm safety PASS |
| 4-home×3-cell conv12 | QSU | C122 exact mode |

전체 v4 count는 `QSU3/QSN0/MI3/UQ5`다. C126의 QSU는 safety와 양의 admitted AI를
뜻하며 global routing 처리량 우위를 뜻하지 않는다. C126 strict same-radio parity와 +2%
outcome gate는 실패했다.

## Sharded-home v5: control-plane fault class도 mode다

동일한 2-home×4-cell, conv25, certificate-ordered executor라도 허용하는 broker fault와
ambiguity policy가 다르면 같은 mode로 볼 수 없다. Checker v3는
`control_fault_model`, `control_fault_qualified`, `ambiguous_token_policy`를 별도 축으로
기록한다.

| Mode | 상태 | 근거 |
|---|---|---|
| Nominal broker, certificate executor | QSU | C126 safety/usefulness |
| Post-apply commit reply loss, 처리 규칙 없음 | UQ | 재시도는 중복 실행, controller 종료는 RAN 전파 가능 |
| 같은 reply loss, no-retry·affected-home AI quarantine | QSU | C127 두 arm 5,440 TB, duplicate 0, local credit drain |

C127 QSU는 검증한 **한 client의 post-apply commit reply loss**에만 적용된다. Broker
crash/restart, complete ambiguity, partition, 여러 동시 fault는 이 분류에 포함되지 않는다.
전체 v5 count는 `QSU4/QSN0/MI3/UQ6`이다.

## Sharded-home v6: synchronous control budget도 capacity다

Checker v4는 broker fail-stop을 허용하는 mode에 대해 `control_rpc_bound_ms`,
`control_rpc_count`, `control_transaction_budget_ms`와 실제 admission 반영 여부를 검사한다.
AI class의 effective bound도 `B_AI+B_control`로 계산한다.

| Mode | 상태 | 근거 |
|---|---|---|
| Synchronous marker를 쓰는 crash injector | UQ | C128 양 home D155 miss |
| Direct fail-stop, unbounded RPC | UQ | C129 complete RPC 뒤 fallback +160.844 ms |
| 5 ms/RPC timeout, admission charge 없음 | UQ | C130 physical PASS이나 proof incomplete |
| 3 RPC 중 commit+complete 10 ms만 반영 | UQ | C131 physical PASS이나 prepare 5 ms 누락 |
| 3 RPC×5 ms=15 ms 전부 반영 | QSU | C132 두 arm 5,440 TB, crash 뒤 4,976 TB, safety 위반0 |

전체 v6 count는 `QSU5/QSN0/MI3/UQ10`이다. C130/C131을 UQ로 분류한 것은 물리 실행이
실패했기 때문이 아니라, 선언 theorem의 모든 synchronous path가 admission inequality에
포함되지 않았기 때문이다. 이 구분이 표본 PASS를 안전 증명으로 과대해석하지 않게 한다.

## Sharded-home v7: full-budget mode의 fault-point coverage

분류 count는 v6와 같은 `QSU5/QSN0/MI3/UQ10`이다. 달라진 것은 C132 QSU point의 증거
coverage다. C133이 post-prepare와 post-complete fault를 두 arm씩 추가했고, frozen C132의
post-commit 두 arm과 합쳐 `prepare/commit/complete` 각각 두 arm, 총 16,320 TB와 crash
탐지 뒤 14,938 TB에서 safety 위반 0을 확인했다. Prepare ambiguity는 physical AI를
launch하지 않았고 complete ambiguity의 target AI는 정확히 한 번 실행됐다.

## Sharded-home v8: 독립 allocation/node 재자격

분류 count는 v7과 같은 QSU5/QSN0/MI3/UQ10이다. C134는 runtime과 full 15 ms control
budget을 그대로 두고 새 allocation 58823497과 다른 A100 node nid002688에서
prepare/commit/complete를 각각 두 arm으로 재실행했다. 총 16,320 TB, crash 탐지 뒤
14,926 TB와 containment 전 AI 354개에서 deadline·bound·horizon·duplicate 위반은 0이었다.
따라서 해당 QSU point는 두 독립 A100 node에서 재현됐다. 다른 GPU family와 production
timing은 여전히 UQ다.

## Sharded-home v9: home-local slack 교정과 exact oracle

V4 checker까지는 system 전체에서 NRx가 하나라도 수락되면 모든 home의 조건부 회수량을
`(cells_h−1)×B_conv`로 계산했다. 이 식은 대칭 실험점에서는 드러나지 않지만 비대칭
endpoint 상태에서 한 home의 NRx 수락을 다른 home의 slack으로 옮기는 오류다.

V5는 home `h`의 실제 NRx 수락 수를 `k_h`라 하고, recovery가 하나 이상 남아 atomic
exchange를 gate하는 현재 실행 phase에서 다음 상한만 허용한다.

```text
Delta_h <= max(0, k_h - 1) B_conv
```

Synthetic regression은 두 home의 수락 수가 `[0,1]`인 상태를 만들었다. V4는 존재하지 않는
`[75,75] ms`를 부여해 QSU로 분류했지만, V5는 `[0,0] ms`와 QSN으로 교정했다. 이는 물리
실험 결과를 바꾼 것이 아니라 envelope의 false-usefulness 예측을 제거한 것이다. 기존
18개 물리 mode의 count는 `QSU5/QSN0/MI3/UQ10`으로 같고, 일부 NRx만 수락되는 MI
counterfactual 세 점의 과대 회수량은 감소했다.

Endpoint list schedule도 checker 코드와 독립된 exact oracle로 감사했다. 현재 18개
scenario의 39개 disjoint/shared endpoint pool에서 checker의 수락 수와 exact 최대값이
모두 같았고 schedule 오류는 0이었다. 추가로 home 수, deadline, recovery bound와
endpoint service bound를 바꾼 3,640개 bounded 상태를 전수 열거했으며 cardinality 차이는
0이었다. 이 결과의 범위는 동시 release, endpoint별 job-independent bound인 현재 모델이다.
Online arrival, job별 path bound와 일반 queueing 최적성은 포함하지 않는다.

## Sharded-home v10: 용량 기하와 실제 결정시각 분리 — 이후 교정됨

V9까지의 `W < B_eff <= W+Delta_h`는 recovery credit을 삭제했을 때 생길 수 있는
**용량 기하의 필요조건**이다. 현재 controller는 모든 admitted NRx 결과를 관측하고
rejected NRx 요청의 conventional recovery를 실행한 뒤에야 exchange를 시도한다. 따라서
그때까지 경과한 시간을 빼지 않으면 AI usefulness를 과대평가한다.

V6 checker는 home별로 다음 값을 계산한다.

```text
T_dec,h = latest admitted-NRx finish bound
          + rejected_NRx_h * B_conv,h
H_1,h   = D_h - guard_h - B_conv,h
safe exchange only if B_eff <= H_1,h - T_dec,h
```

`B_eff`에는 AI와 admission에 부과한 control transaction budget이 함께 들어간다. 현재
18개 scenario에서 용량 기하만 보면 exchange-only home×AI-class 후보가 42개였지만,
decision-time 조건까지 만족한 후보는 **0개**였다. `D80/NRx45/conv12/AI35` 반례에서
V5는 QSU였고 V6는 QSN이다. 기존 18개 mode의 최종 count는
`QSU5/QSN0/MI3/UQ10`으로 변하지 않았다. QSU 다섯 점은 static window에 들어가는
AI class가 이미 있기 때문이다.

이 판정은 V10 입력의 가정 아래에서는 맞지만 현재 구현을 정확히 나타내지 않았다.
V10은 endpoint가 한 release에서 여러 completion slot을 소비할 수 있다고 보았고,
모든 executor가 rejected recovery를 exchange 전에 실행한다고 가정했다. 다음 V11에서
이 두 가정을 구현과 맞췄으므로 “후보 0”은 현재 결론이 아니라 보존된 중간 감사 결과다.

## Sharded-home v11→v12: finite ring, executor phase와 완전한 AI guard

실제 CUDA-IPC endpoint는 `ring_depth=1`이므로 두 endpoint가 있는 home의 동시 NRx
admission은 최대 두 개다. 또한 C125 이후 certificate executor는 NRx가 거절된 요청의
recovery를 exchange 시점에도 live credit으로 유지한다. V7 checker는 두 항을 mode
vector로 추가한다.

Conditional release 상한은 phase에 따라 다음처럼 달라진다.

```text
rejected recovery를 먼저 실행: Delta_h <= max(0, k_h-1) B_conv,h
rejected recovery가 live:       Delta_h <= min(k_h, n_h-1) B_conv,h
```

Full-control 두-home mode에서 V10은 home당 `4 admitted/0 rejected`, decision window
38 ms를 계산했다. V11은 물리 ring을 따라 `2 admitted/2 rejected`로 교정한다. 두 NRx가
성공해 50 ms recovery credit을 삭제해도 recovery 두 개가 남으므로

```text
T_dec = 45 ms
H_2 = 155 - 2 - 2*25 = 103 ms
safe transaction window = 103 - 45 = 58 ms
AI40 + prepare/commit/complete 15 ms = 55 ms
AI completion guard = 2 ms
complete effective transaction = 57 ms
corrected modeled margin = 1 ms
```

V11은 마지막 completion guard 2 ms를 별도로 더하지 않아 55 ms와 3 ms 여유로
기록했다. V12는 실제 runtime의 `admission_ai_guard=17 ms`와 맞춰 위 산술을 57 ms와
1 ms 여유로 교정했다. 기존 18개 mode의 분류 수는 `QSU5/QSN0/MI3/UQ10`으로 유지되며,
decision-time exchange-only 후보는 0개에서 두 개로 바뀐다. 두 후보는 같은 full-control
mode의 두 home에 각각 하나다.

독립 exact oracle은 finite ring과 executor phase를 포함한 10,920개 bounded 상태,
18개 scenario와 39개 endpoint pool에서 checker와 최대 수락 수 차이 0, schedule 오류
0을 얻었다. Incremental observation은 새 service class를 만들지 않아 구현하지 않았다.

C135는 이 예측을 실행 전에 동결하고 context-128 Qwen request를 사용해 두 독립 arm에서
검증했다. 정확한 `2 admitted/2 success/2 rejected-live` 분기는 288회, 그 분기의 AI40
exchange는 arm별 5회씩 총 10회 발생했다. 2,560 TB에서 deadline·bound·horizon·credit
위반은 0이었다. 이는 모델이 구현을 설명하는 데서 나아가 이전 trace가 노출하지 못한
service class를 예측해 물리화한 결과다. Synthetic mechanism workload와 유한 표본이므로
BurstGPT 처리량 우위, WCET 또는 production deadline 증거로 확대하지 않는다.
사건별 반사실 감사에서 완전한 57 ms transaction은 10/10 모두 static 53 ms slack에는
들어가지 않았고 conditional 58 ms window에는 들어갔다.

C136은 C135 node를 제외한 `nid002817`에서 새 seed와 process lifecycle의 6개 arm을
재실행했다. 7,680 TB, 목표 분기 908회, AI40 exchange 28회에서 모든 선언 gate가
통과했다. C135와 합치면 두 A100 node·8 arm·10,240 TB·목표 분기 1,196회·AI40
exchange 38회이며 safety 위반은 0이다. 다만 zero-failure 95% 상한은 독립 단위를
TB로 놓으면 0.0293%, arm으로 놓으면 31.23%, node로 놓으면 77.64%다. 상관을 무시한
TB 수치를 hardware-population 보장으로 사용하지 않고 두-node finite-sample
requalification으로만 해석한다.

C137은 정상 경로의 RPC 5,993회가 모두 5 ms 안에 반환됨을 보였지만, 이 결과만으로
socket timeout을 controller-return wall bound로 쓸 수는 없었다. C138의 첫 prospective
prepare-crash arm에서 faulting prepare와 동시에 진행 중이던 complete가 각각
5.205092/5.830741 ms에 반환되어 `3×5=15 ms` admission 계약을 반증했다. Radio safety는
유지됐지만 V12 AI40 timing mode는 UQ로 내렸다.

V13은 socket timeout `T_sock=5 ms`와 admission wall bound `B_rpc=7 ms`를 분리한다.
Prepare/commit/complete의 full charge는 21 ms이고 completion guard 2 ms까지 포함한다.
C139는 새 node의 prepare/commit/complete 각 두 fault arm, 총 16,320 TB와 실제 시도 RPC
2,020회에서 최대 5.653235 ms, 7 ms 초과 0, safety 위반 0을 얻었다. 이 교정으로 AI40은
`40+21+2=63 ms`라 58 ms conditional window에도 들어가지 않는다. 대신 AI35가
`35+21+2=58 ms`로 정확히 경계에 놓인다. C140의 두 arm은 이 class를 8회 실행했고
모든 사건에서 static margin −5 ms, conditional margin 0 ms, physical guarded-horizon
위반 0이었다. C141은 독립 node `nid001372`에서 control fault 6 arm과 AI35 두 arm을
다시 실행했다. 추가 RPC2,010회의 최대는 5.775660 ms, 7 ms 초과0이었고 AI35 exchange7,
safety 위반0이었다. Corrected exact mode의 현재 범위는 같은 A100 family 두 node다.

## Artifact

- [Checker input](../../results/softwall_multigpu/softwall_endpoint_offload_envelope_grid_v1.json)
- [Prediction output](../../results/softwall_multigpu/softwall_endpoint_offload_envelope_prediction_v1.json)
- [Artifact manifest](../../results/softwall_multigpu/softwall_endpoint_offload_envelope_manifest_v1.json)
- [Checker source](../../scripts_for_node/softwall_same_gpu/softwall_envelope_checker.py)
- [단위시험](../../scripts_for_node/softwall_same_gpu/test_softwall_envelope_checker.py)
- [Sharded-home v2 input](../../results/softwall_multigpu/softwall_sharded_home_envelope_grid_v2.json)
- [Sharded-home v2 prediction](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v2.json)
- [Sharded-home v2 manifest](../../results/softwall_multigpu/softwall_sharded_home_envelope_manifest_v2.json)
- [Sharded-home v3 input](../../results/softwall_multigpu/softwall_sharded_home_envelope_grid_v3.json)
- [Sharded-home v3 prediction](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v3.json)
- [Sharded-home v3 manifest](../../results/softwall_multigpu/softwall_sharded_home_envelope_manifest_v3.json)
- [Sharded-home v4 input](../../results/softwall_multigpu/softwall_sharded_home_envelope_grid_v4.json)
- [Sharded-home v4 prediction](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v4.json)
- [Sharded-home v4 manifest](../../results/softwall_multigpu/softwall_sharded_home_envelope_manifest_v4.json)
- [Sharded-home v5 input](../../results/softwall_multigpu/softwall_sharded_home_envelope_grid_v5.json)
- [Sharded-home v5 prediction](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v5.json)
- [Sharded-home v5 manifest](../../results/softwall_multigpu/softwall_sharded_home_envelope_manifest_v5.json)
- [Sharded-home v6 input](../../results/softwall_multigpu/softwall_sharded_home_envelope_grid_v6.json)
- [Sharded-home v6 prediction](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v6.json)
- [Sharded-home v6 manifest](../../results/softwall_multigpu/softwall_sharded_home_envelope_manifest_v6.json)
- [Sharded-home v7 input](../../results/softwall_multigpu/softwall_sharded_home_envelope_grid_v7.json)
- [Sharded-home v7 prediction](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v7.json)
- [Sharded-home v7 manifest](../../results/softwall_multigpu/softwall_sharded_home_envelope_manifest_v7.json)
- [Sharded-home v8 input](../../results/softwall_multigpu/softwall_sharded_home_envelope_grid_v8.json)
- [Sharded-home v8 prediction](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v8.json)
- [Sharded-home v8 manifest](../../results/softwall_multigpu/softwall_sharded_home_envelope_manifest_v8.json)
- [Sharded-home v9 input](../../results/softwall_multigpu/softwall_sharded_home_envelope_grid_v9.json)
- [Sharded-home v9 prediction](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v9.json)
- [Sharded-home v9 manifest](../../results/softwall_multigpu/softwall_sharded_home_envelope_manifest_v9.json)
- [V9 validation summary](../../results/softwall_multigpu/softwall_envelope_v9_validation_summary.json)
- [Sharded-home v10 input](../../results/softwall_multigpu/softwall_sharded_home_envelope_grid_v10.json)
- [Sharded-home v10 prediction](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v10.json)
- [Sharded-home v10 manifest](../../results/softwall_multigpu/softwall_sharded_home_envelope_manifest_v10.json)
- [V10 validation summary](../../results/softwall_multigpu/softwall_envelope_v10_validation_summary.json)
- [Sharded-home v11 input](../../results/softwall_multigpu/softwall_sharded_home_envelope_grid_v11.json)
- [Sharded-home v11 prediction](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v11.json)
- [Sharded-home v11 manifest](../../results/softwall_multigpu/softwall_sharded_home_envelope_manifest_v11.json)
- [V11 validation summary](../../results/softwall_multigpu/softwall_envelope_v11_validation_summary.json)
- [Checker v7](../../scripts_for_node/softwall_same_gpu/softwall_envelope_checker_v7.py)
- [V10→V11 ring/phase regression](../../results/softwall_multigpu/softwall_envelope_v10_v11_ring_phase_regression_v1.json)
- [Sharded-home v12 input](../../results/softwall_multigpu/softwall_sharded_home_envelope_grid_v12.json)
- [Sharded-home v12 prediction](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v12.json)
- [V12 validation summary](../../results/softwall_multigpu/softwall_envelope_v12_validation_summary.json)
- [V12 artifact manifest](../../results/softwall_multigpu/softwall_sharded_home_envelope_manifest_v12.json)
- [Checker v8](../../scripts_for_node/softwall_same_gpu/softwall_envelope_checker_v8.py)
- [V11→V12 AI guard regression](../../results/softwall_multigpu/softwall_envelope_v11_v12_ai_guard_regression_v1.json)
- [Exact finite-ring oracle v3](../../results/softwall_multigpu/softwall_endpoint_admission_exact_oracle_audit_v3.json)
- [C135 결과](SOFTWALL_CONFIRM135_V11_AI40_RESULT_KO.md)
- [C135 campaign result](../../results/softwall_multigpu/confirm135_v11_ai40_campaign_result.json)
- [C135 artifact manifest](../../results/softwall_multigpu/confirm135_artifact_manifest.json)
- [C135 static counterfactual](../../results/softwall_multigpu/confirm135_static_counterfactual_audit_v1.json)
- [C136 독립 node 결과](SOFTWALL_CONFIRM136_V12_REQUALIFICATION_RESULT_KO.md)
- [C136 campaign result](../../results/softwall_multigpu/confirm136_v12_ai40_requalification_result.json)
- [C136 artifact manifest](../../results/softwall_multigpu/confirm136_artifact_manifest.json)
- [C135/C136 결합 감사](../../results/softwall_multigpu/confirm135_136_combined_v12_qualification.json)
- [V12 cross-node validation summary](../../results/softwall_multigpu/softwall_envelope_v12_cross_node_validation_summary.json)
- [V12 cross-node manifest](../../results/softwall_multigpu/softwall_envelope_v12_cross_node_manifest.json)
- [C137 service-bound telemetry](SOFTWALL_CONFIRM137_SERVICE_BOUND_TELEMETRY_RESULT_KO.md)
- [Service-bound qualification v2](../../results/softwall_multigpu/softwall_service_bound_qualification_v2.json)
- [C137 artifact manifest](../../results/softwall_multigpu/confirm137_artifact_manifest.json)
- [Service-bound v2 manifest](../../results/softwall_multigpu/softwall_service_bound_v2_manifest.json)
- [C138--C140 control-bound correction](SOFTWALL_CONFIRM138_140_CONTROL_BOUND_CORRECTION_KO.md)
- [Sharded-home envelope v13](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v13.json)
- [V12→V13 control-bound regression](../../results/softwall_multigpu/softwall_envelope_v12_v13_control_bound_regression_v1.json)
- [V13 validation summary](../../results/softwall_multigpu/softwall_envelope_v13_validation_summary.json)
- [Service-bound qualification v3](../../results/softwall_multigpu/softwall_service_bound_qualification_v3.json)
- [V13 corrected-control manifest](../../results/softwall_multigpu/softwall_v13_corrected_control_manifest.json)
- [C141 독립-node 재자격](SOFTWALL_CONFIRM141_V13_INDEPENDENT_RESULT_KO.md)
- [Service-bound qualification v4](../../results/softwall_multigpu/softwall_service_bound_qualification_v4.json)
- [과거 trace evidence audit](../../results/softwall_multigpu/softwall_v11_candidate_physical_evidence_audit_v1.json)
- [Incremental observation 반증](../../results/softwall_multigpu/softwall_incremental_exchange_prospective_v2.json)
- [Checker v6](../../scripts_for_node/softwall_same_gpu/softwall_envelope_checker_v6.py)
- [Decision-time audit](../../results/softwall_multigpu/softwall_decision_time_envelope_audit_v1.json)
- [V5→V6 decision-time regression](../../results/softwall_multigpu/softwall_envelope_v5_v6_decision_time_regression_v1.json)
- [Checker v5](../../scripts_for_node/softwall_same_gpu/softwall_envelope_checker_v5.py)
- [Exact endpoint oracle](../../scripts_for_node/softwall_same_gpu/softwall_endpoint_admission_exact_oracle.py)
- [Exact oracle audit v2](../../results/softwall_multigpu/softwall_endpoint_admission_exact_oracle_audit_v2.json)
- [V4→V5 home-local regression](../../results/softwall_multigpu/softwall_envelope_v4_v5_home_local_regression_v1.json)
- [C134 독립 node 결과](SOFTWALL_CONFIRM134_INDEPENDENT_NODE_RESULT_KO.md)
- [C134 artifact manifest](../../results/softwall_multigpu/confirm134_artifact_manifest.json)
- [C125/C126 artifact manifest](../../results/softwall_multigpu/confirm125_126_artifact_manifest.json)
- [Disjoint-home checker v2](../../scripts_for_node/softwall_same_gpu/softwall_envelope_checker_v2.py)
- [C121/C122 물리 결과](SOFTWALL_CONFIRM121_122_SHARDED_HOME_RESULT_KO.md)
- [C123--126 global AI/executor 결과](SOFTWALL_CONFIRM123_126_GLOBAL_AI_RESULT_KO.md)
- [C127 broker fault containment](SOFTWALL_CONFIRM127_BROKER_FAULT_RESULT_KO.md)
- [C128--132 broker fail-stop/control-budget](SOFTWALL_CONFIRM128_132_BROKER_CRASH_RESULT_KO.md)
- [C128--133 full control-point matrix](SOFTWALL_CONFIRM128_133_CONTROL_FAULT_RESULT_KO.md)
