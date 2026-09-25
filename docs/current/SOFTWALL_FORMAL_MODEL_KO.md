# SoftWall 형식 모델과 입증 의무

**상태:** 2026-09-25 모델 v16 물리 자격 및 V17 shared-recovery C158 반복 자격 반영  
**목적:** 시스템 구현의 안전 불변식, 유용성 경계와 멀티 GPU 확장을 하나의 검증 가능한
모델로 고정한다. 이 문서의 정리는 자격화된 service bound를 가정한 조건부 결과이며
현재 측정치를 WCET로 바꾸지 않는다.

## 1. 문제의 최소 모델

RAN 요청 `i`는 다음 tuple이다.

```text
J_i = (r_i, d_i, h_i, c_i, K_i, q_i)
```

- `r_i`: 요청 release
- `d_i`: radio result가 유효한 expiry
- `h_i`: conventional recovery와 최종 commit이 실행되는 home resource
- `c_i`: 자격화된 conventional host-to-commit bound
- `K_i`: 선택 가능한 optional NeuralRx endpoint 집합
- `q_i`: NeuralRx 성공 시 얻는 radio value

Endpoint `e`는 `(GPU, transport, queue, ring)`의 조합이다. `z`는 home/remote GPU에서
동시에 실행되는 workload class다. 예를 들어 `isolated`, `remote-P2P-overlap`,
`home-Qwen-overlap`, `fault-recovery-overlap`을 구별한다. 요청 `i`를 endpoint `e`에
보낼 때의 mode별 전체 path bound는 다음이다.

```text
B_sum(i,e,z) = f_i,z + x_fwd,i,e,z + w_i,e,z + n_i,e,z
               + x_bwd,i,e,z + b_i,z + g_i,e,z
```

여기서 `f,b`는 home GPU의 Aerial front/back, `x_fwd,x_bwd`는 transport, `w`는 endpoint
queue, `n`은 NeuralRx, `g`는 관측·commit guard다. Runtime이 사용하는 path contract
`p_i,e,z`는 두 방법 중 하나로 자격화한다.

```text
p_i,e,z = directly qualified end-to-end physical-completion bound
       or a conservatively composed B_sum whose every component is qualified
```

`p_i,e,z`가 request별 fallback cutoff 안에 들어오는 endpoint만 eligible하다. Component
vector는 transport 선택과 실패 진단에 항상 기록하지만, 한 component 후보가 깨졌다고
더 큰 end-to-end 계약까지 자동으로 깨졌다고 판정하지 않는다.

`z`를 생략할 수 있는 것은 모든 허용 co-run class를 지배하는 bound가 따로 자격화된
경우뿐이다. C117에서 isolated에 가까운 기존 관측치 위로 잡은 home front/back 2 ms
후보가 각각 한 번씩 깨진 것은 이 생략이 false-safe를 만들 수 있다는 반례다. C118은
동일한 full co-run mode에서 3 ms 후보를 새 seed로 재자격했지만 WCET로 승격하지 않는다.
C119/C120의 4-GPU mode에서도 system/end-to-end gate는 네 arm 모두 통과한 반면 backward
P2P 100 us와 home back 3 ms 후보가 차례로 한 건씩 깨졌다. 따라서 현재 4-GPU runtime
계약은 직접 자격화한 전체 path bound를 사용하고 component vector는 진단용으로 둔다.

외부 AI 요청 `a`는 다음 tuple이다.

```text
A_a = (u_a, v_a, ell_a, b_a, g_a)
```

- `u_a`: 관측 가능한 arrival
- `v_a`: deadline 안에 완료했을 때의 value
- `ell_a`: AI deadline
- `b_a`: 해당 context/work class의 자격화된 physical-completion bound
- `g_a`: 다음 recovery와 분리할 guard

Admission 전 offered request에는 완료 보장이 없다. Admission된 non-preemptive unit만
`b_a`와 `ell_a` 계약의 대상이다.

## 2. 조건부 recovery debt

Optional NRx가 unresolved인 동안 요청 `i`는 conventional recovery job `R_i`를 빚진다.

```text
R_i = (a_i(t), d_i - g_i, c_i, h_i)
```

`a_i(t)`는 현재 lifecycle에서 conventional을 합법적으로 시작할 수 있는 가장 이른
시각이다. NRx를 포기한 early fallback이면 `t`, 결과를 기다리는 mode이면 자격화된 cutoff다.

시각 `t`의 unresolved 집합을 `U(t)`라 한다. All-fail certificate `S_t`는 각
`i ∈ U(t)`에 interval `[s_i, s_i+c_i]`를 부여하며 다음을 만족하는 실행 가능한 일정이다.

```text
a_i(t) <= s_i
s_i + c_i <= d_i - g_i
같은 home recovery lane의 interval은 겹치지 않음
```

Endpoint queue에 conventional job이 아직 나타나지 않아도 `R_i`는 certificate에 존재한다.
이 점이 현재 도착한 작업만 보는 일반 queue scheduler와 다른 상태 표현이다.

## 3. All-fail dominance

**Lemma 1 — deletion monotonicity.** 동일 mode에서 모든 recovery job의 resource demand가
비음수가 아니고, NRx 성공이 새로운 mandatory job을 만들지 않으며 `S_t`가 all-fail
집합 `U(t)`에 대해 실행 가능하면, 실패 subset `F ⊆ U(t)`도 실행 가능하다.

**증명 개요.** `S_t`에서 `U(t)\F`의 interval을 삭제한다. 남은 interval의 release,
deadline과 non-overlap은 바뀌지 않는다. 따라서 모든 실패 subset을 `2^|U|`개 열거하지
않아도 all-fail 한 개가 robust certificate가 된다.

이 축약이 성립하지 않는 mode는 별도로 다뤄야 한다. 예를 들어 한 NRx 성공이 다른
mandatory 후처리를 추가하거나, 실패 subset에 따라 `c_i`가 증가하거나, 공유 transport가
recovery와 역방향으로 결합되면 deletion monotonicity를 다시 증명해야 한다.

**Lemma 1b — disjoint-home composition.** Home GPU `h`마다 recovery lane이 서로 disjoint이고
각 local all-fail certificate `S_t^h`가 유효하며, shared endpoint/AI credit이 mandatory
recovery lane을 점유하지 않거나 이미 local certificate에 비용으로 포함돼 있으면
`S_t = product_h S_t^h`도 유효하다.

**증명 개요.** 각 요청은 정확히 한 home partition에 속한다. Partition 안의 release,
deadline과 non-overlap은 `S_t^h`가 만족한다. 서로 다른 home의 recovery interval은
서로 다른 resource에서 실행되므로 교차 충돌이 없다. 따라서 local schedule의 합집합이
global all-fail 분기의 모든 요청을 만족한다. Shared resource가 mandatory recovery와
경합하거나 recovery를 home 사이에 migration하면 이 lemma의 가정이 깨지므로 별도
multi-resource certificate가 필요하다.

**Lemma 1c — shared-recovery global certificate (control-plane candidate).** 여러 home의
mandatory recovery가 같은 capacity-`m` resource를 사용한다고 하자. 모든 unresolved
obligation의 합집합 `U_G(t)=union_h U_h(t)`에 대해 release, deadline, lane non-overlap과
허가된 external-AI blackout non-overlap을 만족하는 한 global schedule `S_t^G`를 유지하고,
obligation·schedule·AI lease 변경을 한 generation에서 원자 commit하면, local certificate가
각각 가능하다는 조건보다 강한 all-fail safety condition을 얻는다.

**필요성 반례.** `c_i=25 ms`, `d_i=100 ms`, shared lane 하나에서 home 0의 job 3개와
home 1의 job 2개는 각 local calendar에서는 75/50 ms로 가능하지만 합집합은 125 ms라
불가능하다. 따라서 shared recovery를 local 판정의 곱으로 허가하면 false-safe다.

V17 후보 reference model은 bounded exhaustive search로 `S_t^G`를 만들고 generation CAS,
rejected-update rollback과 fence-required lease retire를 구현했다. Equal-deadline 25개
상태에서 analytic capacity와 차이 0, local-safe/global-unsafe 반례 10개, variable
release/deadline/AI-blackout 3,400개 상태(capacity 1/2)에서 독립 slot oracle와 차이 0이었다.
C151/C152는 이 debt를 실행할 한 shared cuPHY worker와 두 home의 P2P data path를 두 A100
node에서 합계 1,000 request·decode/order/lifecycle 오류0으로 검증했다. C153은 reference
model이 global fifth debt를 거절하고, 두 success 뒤 AI35를 lease한 다음 physical Qwen fence와
두 certificate-ordered cuPHY recovery를 D155 안에 구동하는 one-node canary를 통과했다.
C154/C155는 같은 source의 네 controlled branch를 서로 다른 두 A100 node에서 반대 순서와
독립 seed로 실행했다. 여덟 arm에서 submitted36, accepted32, global reject4, success12,
Qwen4, certificate-ordered physical recovery20, correct home commit20, D155 miss0이었다.
따라서 controlled-outcome integrated mode는 finite-sample PASS였다. C157A/C157B는
같은 noisy TB를 GPU3 actual TensorRT NeuralRx와 GPU2 shared recovery에 동시에 연결하고,
45 ms cutoff 전 실제 CRC success만 `resolve_success` 입력으로 사용했다. 두 node 합계
actual success4, Qwen lease2, remaining recovery4, single commit8, D155 miss0이었고
same-input conventional semantic oracle도 4/4 일치했다. Warm synthetic actual-NRx
transition은 finite-sample PASS이며 integrated control/data fault mode는 UQ다.
C158은 같은 V17.1 전이를 두 node·500 epoch에서 actual NRx2,000, recovery560,
Qwen500과 single commit2,000으로 반복했다. Deadline·bound·credit·input round-trip·response
echo violation은 0이었다. C159-Q1은 pre-staged GPU bank와 whole-path preflight를 추가해
`P180/D155` context-64 mode를 두 새 node에서 actual NRx2,000, recovery582, Qwen500,
commit2,000, 위반0으로 재자격했다. C159-Q2는 context별 `B_a={35,35,35,40,65,75} ms`와
두-node 1,200 epoch를 추가해 actual NRx4,800, recovery1,312, Qwen1,077, commit4,800,
위반0을 얻었다. Q3 calibration oracle은 같은 초기 all-fail admission을 사용하는
certificate-preserving recovery-first Event-driven과 SoftWall의 exact optimum을 385,262
token으로 동일하게 판정했다. 이 동률은 AI-first retiming의 추가 처리량을 기각하며
certificate 자체를 기각하지 않는다. Variable-context warm qualification은
통과했다. C160은 51개 epoch/generation/fence state transition에서 violation0으로 fault
의미론을 고정했다. C161 1·2단계는 qualified node에서 actual NRx2,800, recovery942,
commit2,800, miss0을 얻어 A0--A6를 자격화했다. Event replay pair40, terminal fault4와
post-fault continuation260도 통과했다. 같은 source의 `nid001044` NRx45 lifecycle과 WCET는
자격화하지 않는다.

**Lemma 4c — common-cutoff batch outcome.** 한 cutoff `t_c` 전에 관측된 success 집합을 `S`라
하자. Runtime이 `S`를 임의 순서로 하나씩 적용하며 매 중간 상태를 현재시각 `t>t_c`에서
재검사하면, 실제로 존재하지 않은 부분집합 debt가 infeasible하여 최종 $U \setminus S$가 feasible인데도
false reject할 수 있다. `S` 전체를 한 generation에서 제거하고 unresolved 집합 $U \setminus S$만
현재시각 `t`로 재구성하면 이러한 순서 의존성이 없다. 단, 최초 all-fail admission과 물리
outcome fence는 그대로 유지해야 한다.

**Lemma 4d — lease freshness and physical start.** Certificate 계산 시작부터 dispatch까지
`B_ctl`을 넘으면 그 certificate를 폐기하고 현재 clock으로 다시 계산한다. 또한 worker가
committed `latest_start` 뒤에는 kernel을 시작하지 않는다면, host scheduling tail은 오래된
AI blackout을 물리 실행으로 바꾸지 못한다. Empty non-launch lease도 fence-confirmed retire를
거쳐야 한다. C159-Q2의 23.530 ms stale-dispatch 반례와 교정 후 두-node retry 각1회가 이
조건의 필요성과 실행을 보였다.
C156은 fixed lease가 실제 decision/dispatch 지연을 포함하지 않는 반례를 추가했다. 실제
launch 시각 `t_l`에 `B_launch_control=5 ms`와 `B_AI=35 ms`를 합친 blackout을 만들고
남은 debt와 함께 재검사하는 V17.1은 두-node Nsight arm에서 Qwen kernel2,448,
recovery kernel212, forbidden overlap0 ns와 D155 miss0을 통과했다. 너무 늦은
`t_l=64 ms` conditional 상태는 상태 변경 없이 거절된다.
상세는 [V17 shared-recovery 모델](../archive/SOFTWALL_SHARED_RECOVERY_V17_MODEL_KO.md)과
[C151/C152 물리 path](../archive/SOFTWALL_CONFIRM151_152_SHARED_RECOVERY_PATH_KO.md),
[C153 통합 결과](../archive/SOFTWALL_CONFIRM153_INTEGRATED_RESULT_KO.md),
[C154/C155 holdout](../archive/SOFTWALL_CONFIRM154_155_CONTROLLED_HOLDOUT_KO.md),
[C156 GPU timeline](../archive/SOFTWALL_CONFIRM156_GPU_TIMELINE_RESULT_KO.md),
[C157 actual-NRx 결과](../archive/SOFTWALL_CONFIRM157_ACTUAL_NRX_RESULT_KO.md),
[C158 반복 자격](../archive/SOFTWALL_CONFIRM158_REPEATED_QUALIFICATION_KO.md),
[C159-Q1 P180 결과](../archive/SOFTWALL_CONFIRM159_Q1_P180_RESULT_KO.md)와
[C159-Q2 variable-context 결과](../archive/SOFTWALL_CONFIRM159_Q2_VARIABLE_RESULT_KO.md)에 있다.

C121은 두 disjoint home의 `[4,4]` 8셀, C122는 네 home의 `[3,3,3,3]` 12셀에서 이
가정을 물리화했다. 각 home은 별도 recovery lane, NRx endpoint 두 개와 Qwen queue를
가졌고 첫 release는 arm 안에서 0 ns 차이였다. 두 실험의 네 formal arm, 총 13,600 TB에서
관측된 safety violation은 0이었다. 이는 Lemma 1b의 전제 아래 유한 표본 검증이며, shared
endpoint가 있는 경우의 multi-resource 합성 증명은 아니다. C123은 mandatory recovery를
공유하지 않는 두 home 사이에서 **AI request ownership만 공유**하는 제한된 합성을 추가했다.
Global broker의 request hold가 직렬화되고, 선택한 home의 local certificate/lease가 성공한
뒤에만 request가 commit되므로 request 중복은 없었다. 이는 shared mandatory resource까지
포함하는 일반 multi-resource certificate가 아니다.

## 4. Credit 상태와 원자 전이

Runtime 상태는 다음 product state다.

```text
X_t = (S_t, Q_t, T_t, L_t, E_t, C_t, G_t)
```

- `S_t`: recovery certificate
- `Q_t`: endpoint queue/input/output credit
- `T_t`: local IPC 또는 remote P2P ring/transport generation
- `L_t`: 발급된 AI lease
- `E_t`: GPU completion fence와 worker epoch
- `C_t`: 요청별 single-commit 상태
- `G_t`: multi-home AI request의 global hold/owner/generation 상태

`replan_and_lease`는 후보 상태 `X'`를 먼저 private하게 만든다.

```text
VALID(X') = recovery_intervals_feasible
          ∧ endpoint_and_transport_credits_unique
          ∧ AI_finish <= min(AI_deadline, earliest_recovery - guard)
          ∧ all_generations_match
```

`VALID(X')`가 참이면 한 generation에서 `X_t→X'`를 공개한다. 거짓이면 `X_t`를 유지한다.
GPU launch/copy는 상태 commit 뒤 비동기로 일어날 수 있지만, fence가 확인되기 전에는
그 physical credit을 새 generation에 재사용하지 않는다. C158의 shared-recovery mode에서
`E_t`는 CUDA 완료뿐 아니라 다음 transport fence도 포함한다.

```text
F_in(i,g):  worker가 받은 input == owner가 generation g에 보존한 input shadow
F_out(i,g): worker output의 P2P copy 뒤 read-back == worker output shadow
accept_result(i,g) => F_in(i,g) ∧ F_out(i,g) ∧ generation_match(i,g)
```

두 equality는 byte-exact다. Input fence는 같은 TB가 worker까지 갔음을, output fence는
completion doorbell 전에 response copy가 실제 관측 가능했음을 검사한다. 이 조건은 decoder
알고리즘의 정답 자체를 증명하지 않으므로 CRC pass이면 transmitted payload와 별도 비교하고,
CRC fail이면 payload를 radio 결과로 소비하지 않는다.

Global AI queue가 있는 mode에서는 `G_t`가 `ready -> held -> inflight -> completed/late`로
전이한다. `held`만 commit 전 abort로 `ready/expired`에 돌아갈 수 있다. Local replan/lease가
거절되면 global hold를 abort하고, local commit이 성공한 뒤 global token을 commit한다.
Commit 뒤의 불확실한 실행은 다른 home에 중복 재할당하지 않는다.

Broker가 commit을 적용했지만 reply가 유실되면 global truth는 `inflight`, client view는
`ambiguous`가 된다. 이 상태는 다음과 같이 fail-closed로 흡수한다.

```text
G_broker(token) = inflight
G_home(token)   = ambiguous/quarantined
new_global_AI(home) = disabled
retry(token) = false
reassign(token) = false
```

Qwen launch는 commit 성공 reply 뒤에만 일어난다. 따라서 이 fault point에서 response가
없으면 새 physical AI work가 제출되지 않았고 local execution lease는 confirmed-unlaunched로
회수할 수 있다. `S_t`, RAN request state와 다른 home의 local certificate는 바꾸지 않는다.

Global broker와 동기적으로 통신하는 fail-stop mode에서는 socket timeout을 fault handler로만
두면 충분하지 않다. AI transaction의 제어 경로 자체를 다음처럼 정의한다.

```text
B_ctl = B_prepare_rpc + B_commit_rpc + B_complete_rpc
t_admit + B_ctl + b_a + g_a
    <= min(ell_a, earliest_recovery_start)
```

`prepare`는 AI request를 고르는 함수 안에 숨어 있어도 synchronous critical path다. `commit`은
physical AI launch 전, `complete`는 physical return 뒤 recovery executor로 돌아오기 전에
호출된다. 따라서 세 항 중 하나라도 무제한이거나 admission에서 빠지면, AI kernel 자체가
bound 안에 끝나도 recovery dispatch가 늦을 수 있다. C132의 exact mode는 각 RPC 5 ms,
`B_ctl=15 ms`, `g_a=2 ms`를 사용한다.

**Lemma 2 — abort preservation.** 후보 transaction이 실패하고 기존 상태를 변경하지 않으면
기존 certificate의 실행 가능성은 보존된다.

**Lemma 3 — executor conformance.** 같은 recovery lane의 ready obligation 집합 `R(t)`를
현재 certificate의 `(reserved_start, deadline, slot_id)` 순서로 실행하고, 조기 시작할
interval이 더 이른 live credit과 겹치면 이동을 거절하면, 실제 dispatch prefix는 항상
certificate의 실행 가능한 refinement다.

**필요성.** Certificate가 존재해도 ready 원인별로 별도 queue를 먼저 실행하면 이 조건이
깨질 수 있다. C125에서 no-NRx 셀 2를 `[+33.106,+58.106] ms`에 당기려 한 실행은 이미
ready인 failed-NRx 셀 0의 `[+53,+78] ms` credit과 겹쳤다. Runtime은 GPU launch 전에
fail-closed했다. C126 executor는 모든 ready class를 합쳐 live-credit 순서로 실행한다.

**Lemma 3b — generation-scoped transport containment.** 각 request의 input/output buffer가
generation별로 유일하고, `F_in`, `F_out`과 generation match 전에는 결과를 accept하거나
physical credit을 반환하지 않는다고 하자. 그러면 이전 generation의 stale input 또는
response가 현재 request의 정상 completion으로 승인될 수 없다.

**증명 개요.** Buffer reuse 전에는 이전 generation의 fence가 필요하고, 현재 result 승인에는
현재 generation의 두 byte-exact equality가 모두 필요하다. 이전 byte가 남아 있으면 적어도
generation match 또는 해당 equality 하나가 거짓이므로 accept transition이 없다. 이는
transport/ownership 성질이며 PHY decoder의 수학적 correctness는 별도 CRC/payload gate다.

**Lemma 4 — one-home global-AI ambiguity containment.** (i) mandatory recovery resource가
home별로 분리돼 있고, (ii) 모든 synchronous `prepare/commit/complete`가 선언 bound 안에
성공 또는 timeout하며 `B_ctl` 전체가 admission inequality에 포함되고, (iii) Qwen launch는
commit 성공 reply 뒤에만 일어나고, (iv) ambiguous token을 재시도·재할당하지 않으며,
(v) local AI lease는 launch 부재 또는 physical fence가 확인될 때만 회수한다고 하자.
그러면 한 home의 global-AI transition ambiguity는 그 home의 새 AI admission에만 영향을 주며
기존 local all-fail certificate의 실행 가능성과 global AI request의 at-most-once 실행을
보존한다.

**증명 개요.** 조건 (ii)에 의해 fault 감지까지의 모든 동기 제어 지연과 physical AI가
earliest recovery guard 전에 끝난다. Ambiguity 처리 전후 `S_t`와 mandatory endpoint
state는 동일하다. Commit
reply가 없으면 조건 (iii)에 의해 새 GPU work가 없으므로 local AI lease 회수가 recovery
resource를 침범하지 않는다. 조건 (iv)에 의해 broker의 `inflight` request는 다른 execution으로
복제되지 않는다. 다른 home은 disjoint `S_t^h`와 별도 client connection을 사용하므로
Lemma 1b의 local certificate를 계속 실행할 수 있다.

Lemma 4는 세 control RPC를 모두 RAN executor가 기다리는 **V13 synchronous mode**에
대한 정리다. V14가 pipelined mode를 도입했고, V15 runtime은 single-token ownership을
추가해 다음 정리의 전체 조건을 만족한다.

**Lemma 4b — pipelined global-control containment.** 다음 조건을 가정한다.

1. `prepare`, `abort`, `complete`는 RAN executor와 다른 connection·worker에서 비동기로
   실행되며 executor는 그 반환을 기다리지 않는다.
2. `prepare`로 받은 token은 즉시 실행 권한이 아니다. Runtime은 실제 launch 직전에 현재
   all-fail certificate와 AI deadline으로 token을 다시 검사한다.
3. 현재 recovery horizon이 짧아도 AI request deadline이 남아 있으면 token을 staged로
   유지한다. Request deadline 안에서도 실행할 수 없을 때만 비동기 abort한다.
4. 한 home은 launch 전 token을 `staged` 또는 `offered` 중 최대 하나만 보유한다. 이 token이
   있거나 prepare가 pending이면 새 prepare를 발행하지 않는다.
5. `commit`만 RAN critical path에서 동기 실행하고, 그 wall bound `B_commit`을 admission에
   청구한다. Commit ACK 전에는 CUDA work를 제출하지 않는다.
6. 물리 CUDA fence 전에는 local lease를 회수하지 않고, complete ACK 전에는 global token을
   reusable로 만들지 않는다.
7. 어떤 비동기 operation의 적용 여부가 불명확해지면 token을 quarantine하고 신규 global
   AI admission을 닫되 local RAN certificate는 계속 실행한다.

그러면 admission 조건은

```text
t_launch + B_commit + b_a + g_a
    <= min(ell_a, earliest_recovery_start)
```

로 충분하며, 비동기 `prepare/abort/complete`의 반환 지연은 local all-fail certificate의
실행 가능성과 global request의 at-most-once physical launch를 깨지 않는다.

**증명 개요.** 조건 1 때문에 세 비동기 operation의 대기는 recovery dispatch prefix에
삽입되지 않는다. Staged token은 조건 2의 launch-time revalidation을 통과하기 전에는
GPU resource를 점유하지 않고, 조건 3은 짧은 pre-radio horizon을 abort로 오판하지 않게
한다. 조건 4는 정상 ACK 경로에서 broker-held launch 전 token과 client state 사이의
일대일 대응을 유지한다. Prepare reply가 유실돼 token ID를 알 수 없는 분기는 조건 7의
home-level quarantine으로 들어가며, 직렬 prepare와 즉시 fail-closed 때문에 그 뒤 두 번째
token을 만들지 않는다.
실제 launch 전 유일한 동기 구간은 commit이며 위 부등식이 그 지연과 AI execution,
completion guard를 모두 첫 recovery 전에 둔다. ACK가 없으면 launch하지 않고, ACK 뒤에는
fence와 complete 상태가 확정될 때까지 재사용하지 않으므로 중복 실행이 없다. Ambiguity는
조건 7에 따라 AI ownership 상태만 축소하며 `S_t`를 변경하지 않는다.

[유한 상태 모델 v2](../../results/softwall_multigpu/softwall_pipelined_control_model_v2.json)는
prepare/abort/commit/complete의 before/after-apply reply-loss를 포함한 16개 상태·20개
전이와 single-token ownership의 4개 상태·8개 전이를 함께 검사했다. At-most-once launch,
commit-ACK-before-launch, ambiguous token non-reuse, local certificate, one-unlaunched-token
불변식 위반은 0이었다. 이 모델은 timing bound나 GPU 자체 동작을 대신하지 않는다.
두 하위 모델은 fault protocol과 정상 single-token ownership을 각각 전수한 audit이며 전체
thread/interleaving의 Cartesian-product model check는 아니다. C145/C146의 fault arm과
물리 branch counter가 두 모델 사이의 구현 연결을 검사한다.

**Lemma 5 — broker fail-stop composition.** Lemma 4의 조건과 Lemma 1b의 disjoint-home
조건이 모든 home에 성립하고 broker process가 fail-stop하면, 각 home은 현재 synchronous
operation의 bound 안에 fault를 감지하고 새 global AI admission을 닫는다. 이때 이미
admission한 AI transaction의 `B_ctl+b_a+g_a`가 local horizon에 포함돼 있으므로 모든
local certificate는 계속 실행 가능하다.

**증명 개요.** Broker failure 뒤 새 global hold는 생성되지 않는다. 각 client는 최대 현재
RPC bound만큼만 controller를 지연시키며, 이 지연은 admission 시 이미 local certificate에서
차감됐다. Timeout 뒤에는 global queue operation 없이 certificate-ordered recovery executor로
복귀한다. Home별 `S_t^h`가 disjoint이므로 모든 home의 AI가 닫혀도 mandatory schedule의
합집합은 유지된다. 이는 broker 상태 복구나 exactly-once 완료를 증명하지 않는다.

C132는 post-commit, C133은 post-prepare와 post-complete state transition 뒤 reply 전
fail-stop을 각각 두 arm에서 물리화했다. Prepare ambiguity에서는 Qwen launch가 0이고,
commit ambiguity에서도 ACK 전 launch 규칙으로 0이며, complete ambiguity에서는 이미
끝난 target AI가 정확히 한 번만 기록됐다. 세 지점 합계 6 arm·16,320 TB에서 관측된
safety 위반과 duplicate execution은 0이었다.
C134는 같은 가정과 runtime을 새 allocation의 다른 A100 node에서 세 지점 각각 두 arm으로
재자격했다. 추가 16,320 TB와 crash 탐지 뒤 14,926 TB에서도 safety와 duplicate 위반은
0이었다. 이는 theorem 가정의 same-family 재현이며 WCET 또는 cross-family 증명은 아니다.

**Theorem 1 — conditional safety.** 다음 가정 아래 모든 admitted RAN request의 허용된
NRx success/fail/late 분기에서 conventional commit은 expiry 전에 완료된다.

1. 각 mode의 실제 end-to-end NRx physical-completion, recovery와 AI 시간이 선언한 runtime
   bound 안이다. Component 합성 bound를 사용할 경우에는 그 모든 항도 자격화돼야 한다.
2. 초기 `S_0`는 유효하다.
3. 새 RAN 요청은 mandatory recovery를 `S_t`에 넣은 뒤에만 수락한다.
4. 상태 변경은 `VALID` transaction 또는 obligation 삭제만으로 일어난다.
5. 미완료 physical credit은 fence 전에 재사용하지 않는다. Transport를 사용하는 mode는
   Lemma 3b의 generation-scoped input/output fence도 만족한다.
6. 자격 밖 hang/restart에서는 새 admission을 닫는다.
7. 실제 recovery dispatch는 Lemma 3의 certificate-conformance를 만족한다.
8. Global-AI control fault를 허용하는 mode에서는 synchronous V13이면 Lemma 4/5의
   모든 control RPC bound를, pipelined V15이면 Lemma 4b의 launch-time revalidation,
   single-token ownership과 synchronous commit bound만 admission에 포함하고 각 mode의
   launch·fence·quarantine 조건을 만족한다.

**증명 개요.** 사건 수에 대한 귀납법을 쓴다. Mandatory-first admission은 새 요청 뒤에도
유효한 `S`를 만든다. NRx 성공은 Lemma 1의 삭제이고, 실패/late는 이미 예약된 interval을
실행한다. Replan은 `VALID`일 때만 새 유효 certificate로 바뀌며 abort는 Lemma 2를 따른다.
Lemma 3으로 계산된 interval schedule이 실제 dispatch prefix에 의해 깨지지 않는다. Bound
가정으로 각 예약 interval 안에 물리 완료하므로 commit은 `d_i`를 넘지 않는다. Global-AI
ambiguity가 생기면 Lemma 4가 `S_t`를 보존하고 새 AI admission만 닫으므로 같은 귀납식이
계속 성립한다.

## 5. 조건부 slack과 no-benefit 영역

### 5.1 Endpoint admission의 제한된 최적성

현재 envelope의 동기 batch에서는 모든 NRx job의 release가 0이고, endpoint `e`의 path
bound `p_e`는 job에 무관하다. Endpoint `e`의 물리 ring credit을 `L_e`라 하면 한 batch가
사용할 수 있는 완료 slot은 `p_e, 2p_e, ..., L_e p_e`다. Checker는 cutoff가 빠른 job부터
아직 쓰지 않은 가장 이른 완료 slot에 대응시킨다. 현재 CUDA-IPC 구현은 endpoint별
`L_e=1`이다.

**Lemma 6 — finite-credit deadline-slot matching.** Job별 cutoff를 오름차순으로 처리하고
가장 작은 미사용 endpoint completion slot이 cutoff 안이면 대응시키고, 아니면 그 job을
거절하는 알고리즘은 위 제한된 finite-slot 모델에서 수락 개수를 최대화한다.

**증명 개요.** 가장 작은 slot이 현재 cutoff보다 크면 다른 모든 slot도 커서 그 job은
어떤 schedule에서도 수락할 수 없다. 가장 작은 slot이 cutoff 안이면, 그 slot을 뒤의
느슨한-deadline job에 준 임의의 최적 matching에서 두 job을 바꿔 현재 job에 줄 수 있다.
따라서 현재 대응을 포함하는 최적해가 항상 존재하며 job 수에 대한 귀납으로 성립한다.

이 lemma는 job별 endpoint bound, 서로 다른 release, copy-engine 경합 또는 online arrival을
포함하지 않는다. 별도 exact oracle은 현재 grid 18개 scenario의 39개 endpoint pool과
`home/deadline/recovery/endpoint/ring depth/executor phase`를 바꾼 10,920개 작은 상태를
전수 열거했다. Checker와 oracle의 수락 수 차이 및 schedule 오류는 모두 0이었다. 이는
lemma 구현 감사이며 일반 queueing 최적성이나 WCET 증명이 아니다.

### 5.2 Home-local conditional slack

Home `h`의 요청이 `n_h`개이고 optional NRx가 `k_h`개 수락됐다고 하자. Conditional
exchange 때 recovery가 적어도 하나 남아 있어야 하므로, NRx 성공으로 삭제할 수 있는
credit 상한은 executor phase에 따라 달라진다.

```text
rejected recovery를 exchange 전에 실행:
    Delta_h <= max(0, k_h - 1)c_h

rejected recovery가 exchange 때 live:
    Delta_h <= min(k_h, n_h - 1)c_h
```

첫 식에서는 수락하지 않은 NRx의 conventional 작업을 이미 지불했으므로 수락된 NRx 중
하나는 recovery로 남겨야 한다. 둘째 식에서는 거절된 요청도 live obligation이므로 수락된
NRx `k_h`개가 모두 성공해도 `n_h-k_h`개 recovery가 transaction을 gate할 수 있다.
`k_h=0`인 home은 다른 home의 성공으로 slack을 얻지 않는다.

**Lemma 7 — phase-aware deletion bound.** Home `h`의 homogeneous recovery obligation이
`n_h`개, optional NRx admission이 `k_h`개이고 conditional exchange가 적어도 한 recovery를
남겨야 한다고 하자. Rejected recovery를 exchange 전에 모두 완료하는 executor가 삭제할
수 있는 최대 credit은 `max(0,k_h-1)c_h`다. Rejected recovery를 live로 유지하는 executor의
최대값은 `min(k_h,n_h-1)c_h`다.

**증명 개요.** Rejected-first phase에는 NRx가 수락된 `k_h`개 obligation만 남는다.
Transaction boundary를 제공할 하나를 남기면 성공으로 삭제 가능한 수는 최대 `k_h-1`이다.
Live-rejected phase에는 전체 `n_h`개가 남고 NRx 성공이 삭제할 수 있는 것은 수락된
`k_h`개뿐이다. 전체 obligation 하나를 남긴다는 조건과 함께 삭제 수는
`min(k_h,n_h-1)`이다. 각 credit 길이가 `c_h`이므로 위 식을 얻는다. 이 값은 성공 결과와
tail-compaction schedule이 실제로 존재할 때 달성 가능하며, 서로 다른 recovery bound에는
가중 interval 식이 필요하다.

Checker v4는 전체 system의 수락 수가 1개 이상이면 모든 home에 `(n_h-1)c_h`를 부여했다.
비대칭 synthetic regression에서 home별 수락이 `[0,1]`인데도 `[75,75] ms`를 부여해
QSN을 QSU로 잘못 분류했다. Checker v5는 `[0,0] ms`로 교정한다. 기존 v8 물리 mode의
분류 count는 대칭 구성이라 `QSU5/QSN0/MI3/UQ10`으로 유지되지만, 수락이 일부뿐인
counterfactual 세 점의 과대 slack은 줄었다.
Checker v7은 여기에 finite ring credit과 `rejected_recovery_before_exchange`를 추가해
실제 executor phase와 위 식을 일치시켰다.

### 5.3 용량 기하와 decision-time 조건

동일 deadline의 `n`개 recovery credit이 tail에 있고 결과 관측 뒤 `m`개만 남았다고 하자.
각 bound가 `c`이면 tail compaction으로 새로 열 수 있는 최대 prefix는

```text
Delta <= (n-m)c
```

기존 safe work-conserving baseline이 이미 가진 연속 slack을 `W`, 보이는 AI transaction의
전체 bound를 `b_eff`라 하면, compaction 때문에만 새 admission이 생기는 **용량 기하의
필요조건**은 다음이다.

```text
W < b_eff <= W + Delta
```

이 식만으로 현재 controller가 실제로 그 작업을 시작할 수 있다고 결론 내리면 안 된다.
현재 구현은 수락한 NRx 결과를 모두 관측한 뒤 exchange를 시도하지만, 거절된 요청의
conventional recovery를 먼저 실행하는지는 executor mode에 따라 다르다. 동기 release-0
home `h`에 대해 `r_h`를 exchange 전에 실행한 rejected recovery 수,
`m_h`를 exchange 때 남은 전체 recovery 수라 두면

```text
T_dec,h = max admitted-NRx predicted finish
          + r_h * c_h
H_m,h   = d_h - g_h - m_h * c_h
```

이고 현재 phase의 충분조건은 다음이다.

```text
b_eff <= max(0, H_m,h - T_dec,h)
```

여기서 `b_eff = b_AI + b_control + g_AI`이며, raw AI service bound, 자격화된
synchronous control budget, 물리 완료 뒤 요구하는 AI completion guard를 모두 포함한다.
Radio commit guard `g_h`는 recovery tail의 deadline capacity에서 별도로 차감한다. Conditional credit을
하나도 삭제할 수 없으면 exchange-only 후보가 아니다. 이 조건은 현재 observe-all 실행
순서와 mode별 rejected-recovery phase에 대한 보수적 충분조건이며, 일반 online-arrival
최적조건은 아니다.
V12 mode는 `g_AI`를 반드시 명시해야 하며, 누락을 0으로 해석하지 않고 입력 오류로
거절한다.

**Lemma 8 — decision-window sufficiency.** 시각 `T_dec,h`에 길이 `c_h`인 recovery
obligation `m_h>=1`개가 동일 deadline `d_h`의 tail에 실행 가능하게 compact되어 있고,
AI, synchronous control, AI completion guard를 합친 non-preemptive transaction bound가
`b_eff`라고 하자.
다른 공유 resource 제약이 이미 `b_eff` 또는 certificate에 포함됐다면

```text
T_dec,h + b_eff <= d_h - g_h - m_h c_h
```

일 때 transaction을 먼저 실행하고 기존 recovery tail을 그대로 실행하는 schedule이
존재한다.

**증명 개요.** 우변은 compacted tail의 첫 recovery 시작시각이다. Transaction을
`T_dec,h`에 시작하면 가정한 부등식에 의해 그 시각을 넘지 않고 끝난다. 이후 기존 tail의
모든 interval과 deadline은 바뀌지 않는다. 부등식이 깨지면 이 고정 decision time과
non-preemptive prefix 구조에서는 transaction이 첫 recovery와 겹치므로 같은 구조의
schedule은 존재하지 않는다. 이는 임의 release·이질적 bound의 일반 필요충분조건이 아니다.

**Proposition 9 — current-idleness insufficiency.** 하나의 non-preemptive recovery lane에서
현재 시각을 `t`, 공통 radio guard boundary를 `D`, unresolved recovery bound를
`c_1,...,c_m`, AI transaction bound를 `b_eff`라 하자. Mandatory-only schedule은
feasible하지만

```text
t + b_eff + sum_i c_i > D
```

이면 현재 GPU가 idle이라는 사실만 보고 AI를 허가하는 debt-blind policy는 radio contract를
보장할 수 없다.

**증명 개요.** AI와 모든 recovery가 각각 선언 bound를 사용하는 허용 실행을 택하면 마지막
recovery가 `D` 뒤에 끝난다. 위반 원인은 현재 queue가 아니라 아직 materialize되지 않은
future mandatory work다. 이 명제는 모든 finite sample이 실제 miss를 낸다는 뜻이 아니라,
해당 policy가 qualified guarantee를 제공할 수 없다는 뜻이다.

C162의 사전 고정 E4는 `t=45 ms`, debt2, context256에서 bound-respecting finish165ms로
guard153ms를 12ms 넘는다. E6b는 `t=89 ms`, debt1, context64에서 finish154ms로 1ms
넘는다. 두 점은 mandatory-only feasible인 QSN이며 SoftWall은 두 node 합계60/60회 AI를
launch 전에 거절했다. 이는 사후 contract 해석이고 C162에서 debt-blind miss를 관측했다는
주장이 아니다. 상세 감사는
[필요성 반론 결정](../archive/SOFTWALL_NECESSITY_GAP_DECISION_KO.md)에 있다.

**Corollary 1 — too-large-unit no benefit.** 모든 보이는 AI unit이 `b_eff > W+Delta`이면
atomic compaction은 안전하더라도 추가 admission을 만들 수 없다.

**Corollary 2 — no-pressure no benefit.** Baseline이 deadline 전 offered demand를 이미
모두 수용하면 추가 slack은 timely value를 늘리지 못한다.

C113은 두 경계를 실제로 관측했다. Conv12의 4셀 tail에서 `Delta≤36 ms`인데 trace의
99.824%가 65/75 ms unit이어서 too-large 영역이었다. 35 ms 소형 모델은 기하에 들어왔지만
218개 중 216개를 baseline도 처리해 no-pressure 영역이었다. 고부하 30 ms 후보는 실제
43.628 ms로 bound를 깨 unqualified 영역이었다.

Checker v6의 v10 감사에서는 기존 18개 scenario에 `W+Delta`를 만족하는 home×AI-class
쌍이 42개 있었지만 decision-time 충분조건을 만족한 쌍은 0개였다. 다만 V10은 실제
`ring_depth=1` endpoint를 무제한 completion slot처럼 계산했고 모든 executor가 rejected
recovery를 먼저 실행한다고 보았다. Checker v7은 이 두 추상화 오류를 교정했다.

V11은 기존 mode count `QSU5/QSN0/MI3/UQ10`을 유지하면서 C132--134 full-control mode에서
home당 수락/거절이 `4/0`에서 `2/2`로 바뀌고 rejected recovery 두 개가 live인 실제 phase를
반영한다. 이때 `T_dec=45 ms`, `m=2`, `H_m=103 ms`라 transaction window는 58 ms다.
후속 V12는 runtime이 실제 청구한 AI completion guard 2 ms를 별도로 포함한다. 따라서
`b_eff=40+15+2=57 ms`가 1 ms 여유로 들어가는 후보가 두 home에 각각 하나씩 생긴다.
C135는 이 분기를 독립 두 arm에서 288회 만들고 AI40
exchange를 5회씩 완료했으며, 2,560 TB의 deadline·bound·credit 위반은 0이었다. 이는
선언 bound를 사용한 모델 예측과 finite-sample mechanism 실행의 일치이며 WCET 증명은 아니다.
C136은 이전 node를 제외한 새 A100 node에서 6 arm을 추가했다. 결합 결과는 두 node,
8 arm, 10,240 TB, branch 1,196회와 AI40 exchange 38회이며 선언 위반은 0이다. Node 단위
zero-failure 95% 상한은 77.64%이므로 same-family population 보장으로 확대하지 않는다.
C137은 RPC telemetry를 추가한 별도 mode에서 세 번째 A100 node의 6 arm을 실행했다.
`prepare/commit/complete` 5,993회의 최대 wall time은 2.593595 ms이고 5 ms 초과는 0이었다.
같은 실행의 7,680 TB와 AI40 exchange 28회도 통과했다. 이는 control budget의 관측 공백을
닫지만 deterministic bound는 아니다.

후속 C138은 V12가 동일시한 두 시간을 분리해야 함을 보였다. `socket.settimeout(5 ms)`인
faulting prepare와 complete가 controller에는 5.205092 ms와 5.830741 ms 뒤 반환됐다.
따라서 C132–C137의 safety 실행은 보존하지만 `B_rpc=5 ms`인 시간 mode는 UQ다.

V13은 다음 보정 계약을 사용한다.

```text
T_sock = 5 ms
B_rpc = 7 ms
B_control = 3 B_rpc = 21 ms
B_eff = B_AI + B_control + g_AI
```

C139는 이 후보를 사전 고정하고 새 node의 prepare/commit/complete 각 두 arm에서
2,020개 실제 호출을 기록했다. Faulting call 12개는 모두 5 ms보다 늦게 반환됐지만
최대 5.653235 ms였고 7 ms 초과는 0이었다. 보정 뒤 AI40은
`40+21+2=63 ms > 58 ms`라 탈락한다. AI35는 `35+21+2=58 ms`로 static slack 53 ms에는
들어가지 않고 conditional decision window 58 ms에만 들어간다. C140 두 arm은 이
class를 8회 실제 실행해 모든 사건의 static margin −5 ms, conditional margin 0 ms와
physical guarded-horizon을 확인했다. V13 count는 `QSU5/QSN0/MI3/UQ11`이다.
C141은 같은 theorem 가정과 mode를 독립 A100 node에서 다시 실행했다. 추가 control
6 arm·RPC2,010회와 AI35 두 arm·exchange7에서 7 ms 초과와 safety 위반은 0이었다.
이는 theorem의 mode-specific 가정을 same-family 두 node에서 재현한 것이며 WCET 또는
cross-family 증명은 아니다.

V14는 세 RPC를 모두 직렬로 청구하던 V13의 보수성과, 제어 RPC 지연이 local recovery를
직접 막는 구조를 함께 제거한다.

```text
V13 synchronous: B_eff = B_AI + 3 B_rpc + g_AI
V15 pipelined:    B_eff = B_AI + B_commit + g_AI
```

V14/V15에서 `prepare/abort/complete`는 전용 control worker가 처리하고 `commit`만 launch 직전에
동기화한다. 따라서 `AI45+commit7+guard2=54 ms`는 static all-fail slack 53 ms에는
들어가지 않지만 conditional decision window 58 ms에는 4 ms 여유로 들어간다. Checker
당시 V14의 분류는 `QSU6/QSN0/MI3/UQ11`이었다.

C142는 새 node `nid001361`의 두 arm·2,560 TB에서 목표 분기 305회와 AI45 exchange 9회를
실행했고 선언 위반은 0이었다. Commit 최대는 0.612807 ms였고, RAN 경로 밖 deferred RPC는
9.780886 ms까지 늘어났다. C143의 prepare/commit/complete fault 6 arm에서는 commit
340회 최대 5.148988 ms가 7 ms 안이었던 반면 deferred RPC 1,584회 중 50회가 7 ms를
넘었다. 그럼에도 16,320 TB, fault 뒤 14,184 TB의 safety·duplicate 위반은 0이었다.
이는 deferred RPC가 실제로 길어진 상황에서도 local safety가 그 wall bound에 의존하지
않는다는 분리 증거다. C144는 이전 node를 제외한 `nid002049`에서 AI45 한 arm과 세 fault
point 한 arm씩을 재자격해 6,080 TB, AI45 exchange 2회, fault 뒤 3,730 TB에서 모든 gate를
통과했다. 두 node의 결과는 유한 표본 same-A100-family 근거이며 WCET·production
`d_MAC`·cross-family 보장이 아니다.

그러나 V14의 한-request finite model은 staged token 뒤 두 번째 prepare를 표현하지 않았다.
결정적 two-request 구현 감사는 broker-held token 2개 중 client-tracked token 1개인 반례를
재현했다. 실제 C142/C144 arm은 outstanding 0으로 끝났지만 이 전이를 사전 gate하지
않았으므로 V14 mode는 현재 UQ다. V15는 `staged ∨ offered`이면 prepare를 금지한다.
Model v2는 기존 16-state/20-edge fault protocol과 4-state/8-edge single-token ownership
model을 함께 검사해 violation 0을 얻었다. C145/C146은 두 새 A100 node에서 정상 AI45와
prepare/commit/complete fault 8개 arm을 실행했다. 합계 12,160 TB, AI45 exchange12,
fault 뒤 radio7,463, prepare 억제537회, maximum unlaunched token1, safety·duplicate
위반0이다. 이때 abort는 정상 경로에는 있었지만 post-apply fault로 주입되지 않았다.
C147/C148은 서로 다른 새 A100 node에서 abort 적용 뒤 reply-loss를 주입해 3,200 TB,
fault 뒤 radio2,662, target 실행0, prepare 억제129회와 위반0을 얻었다. V16은 모든
state-changing operation의 물리 fault coverage를 요구한다. 이 기준의 grid는
`QSU6/QSN0/MI3/UQ13`이고 V14 ownership row와 V15 three-point row는 UQ, 네 operation마다
두 arm을 가진 V16 row는 QSU다.
Verifier의 set 직렬화 순서는 Python hash seed에 따라 달랐으나, 16개 seed에서 canonical
state/edge multiset, count와 violation 판정은 모두 같았다. Raw JSON byte 순서를 형식 결과로
세지 않고 canonical semantics를 재현성 gate로 사용한다.

### 5.4 자격 증거의 독립 단위

0회 실패를 관측한 독립 단위가 `N_u`개일 때 failure probability의 exact one-sided
`1-alpha` 상한은 다음과 같다.

```text
p_u <= 1 - alpha^(1/N_u)
```

그러나 TB는 같은 release, process, node의 상태를 공유하므로 자동으로 독립 표본이 되지
않는다. SoftWall은 증거를 `TB < home release < arm/process lifecycle < physical node <
GPU family`의 계층으로 보고하고, 주장하려는 모집단에 대응하는 가장 거친 방어 가능한
단위를 사용한다. C135/C136에서 `N_TB=10,240`, `N_release=2,560`, `N_arm=8`, `N_node=2`이며
95% 상한은 각각 0.0293%, 0.1170%, 31.23%, 77.64%다. 이 네 값은 서로 다른 IID 가정의
민감도이며 동시에 성립하는 네 보장이 아니다. 따라서 현재 결론은 두 A100 node에서의
mode requalification이고, 표본 수가 큰 TB-level 수치로 WCET나 hardware-population
reliability를 주장하지 않는다.

계측 코드도 `theta`의 lifecycle/instrumentation 축이다. 따라서 C135/C136과 C137의
논리 계약 필드가 같아도 exact physical mode는 구분한다. 세 node 합계는 mechanism
reproduction에는 사용할 수 있지만, RPC telemetry가 없는 두 node를 계측 완료 표본으로
소급하지 않는다.

## 6. Transport 선택 경계

Local endpoint `l`과 remote endpoint `e` 중 remote가 eligible하려면

```text
p_i,e <= fallback_cutoff_i - now
```

이고 성능상 선택할 이유가 있으려면 보수적으로

```text
local_queue_bound - remote_queue_bound
  > x_fwd,e + x_bwd,e + remote_interference_penalty
```

여야 한다. 첫 식은 safety 조건이고 둘째 식은 utility 조건이다. C114는 첫 식이 만족하는
한 warm mode를 물리 검증했다. G5는 두 식이 예측한 local/remote 경계를 사전에 고른 뒤
false-safe와 false-conservative를 측정한다.

## 7. Feasibility envelope

Mode vector를 다음처럼 둔다.

```text
theta = (GPU topology, cells, endpoint placement, transport, co-run class,
         MPS caps/priorities, calendar, fault correlation,
         NRx/conv/AI class, memory residency, lifecycle)
```

모델은 `theta`를 네 집합으로 분류한다.

```text
QSU: qualified-safe-useful
QSN: qualified-safe-no-slack
MI : mandatory-infeasible
UQ : unqualified
```

물리 판정과 비교할 때 다음 오류를 분리한다.

- **false-safe:** 모델은 QSU/QSN인데 물리 safety gate가 깨짐
- **false-conservative:** 모델은 MI/UQ인데 물리적으로 안전한 admitted service가 존재
- **utility prediction error:** safety는 맞지만 timely value 또는 local/remote 선택 경계가 틀림

False-safe를 가장 심각한 오류로 센다. 표본에서 false-safe 0이라고 확률 0이나 WCET가
되는 것은 아니며, mode qualification의 관측 결과로만 보고한다.

## 8. 멀티 GPU에서 새로 필요한 credit

C114의 remote endpoint는 다음 credit을 한 lifecycle로 묶는다.

1. Home GPU의 conventional recovery interval
2. Endpoint input/output slot
3. GPU0 IPC allocation handle
4. GPU0→GPU1 forward P2P operation
5. GPU1 NRx graph generation
6. GPU1→GPU0 backward P2P operation
7. Home GPU back/LDPC/CRC와 single commit
8. 같은 horizon에서 허가한 external-AI lease

Control state의 commit만 원자적이다. 여러 GPU kernel이 하드웨어에서 한순간에 원자 실행된다고
주장하지 않는다. 실제 작업은 단계별 fence가 끝날 때까지 점유 상태로 남는다.

## 9. 입증 의무와 현재 증거

| 입증 의무 | 현재 증거 | 상태 |
|---|---|---|
| Mandatory-first와 all-fail schedule 보존 | runtime unit 21, fault regression 10,000 | 통과 |
| 복수 credit+AI lease commit/abort | C73/C76/C80, C113 | 4셀 유한 표본 통과 |
| Physical completion 전 credit 재사용 금지 | C81, IPC generation/fence 검사 | 제한적 fault 통과 |
| 실제 동시 GPU 실행 | C68/C70 Nsight | 확인 |
| No-benefit 조건 | C113 envelope와 strong baseline | 관측 일치 |
| Transport-independent endpoint | C114 G1a/G2a/G3 | 2-GPU 유한 표본 통과 |
| Same-budget multi-GPU utility | C115 10-arm ABBA, +0.080%/+0.138%, 두 CI 0 포함 | 우위 기각 |
| Three-endpoint generation lifecycle | C116 local1+remote2, 두 seed, worker count 일치 | 유한 표본 통과 |
| Co-run component vector | C117 home 2 ms 후보 FAIL; C118 home 3 ms·remote component PASS | exact warm mode의 유한 표본 경계 |
| Four-GPU/four-endpoint lifecycle | C119/C120 네 formal arm safety·count PASS; fine component gate 두 번 FAIL | 통합 경로 성립, component vector 일반화 기각 |
| 1/2/4 GPU endpoint-only boundary prediction | deterministic checker: 4-cell QSU 3점, 8/12-cell memory MI 6점 | v1 완료 |
| Disjoint sharded-home composition | C121 2-home×8 총 5,440 TB; C122 4-home×12 총 8,160 TB; release delta 0 ns, safety 위반0 | 물리 합성 통과; C124 후 C121 conv12 자격은 철회 |
| Global AI ownership/local certificate 합성 | C123 두 arm, global request 2,268회 완료, duplicate/outstanding 0 | 제한된 shared-AI 합성 통과 |
| 12 ms 2-home mode falsification | C124 마지막 arm conv host 12.535096 ms > 12 ms | C121 exact mode를 UQ로 강등 |
| Executor conformance | C125 ordering 반례 보존; 단위시험2와 C126 8-arm 21,760 TB·2,496 exchange, safety 위반0 | conv25 certificate executor QSU |
| Global routing utility | C124 음수; C126 +0.0049%/+0.0236%, 두 CI 하한0·strict radio parity FAIL | 처리량 우위 기각 |
| Global commit ambiguity containment | C127 두 arm 5,440 TB, fault 뒤 home0 2,490 TB·home1 AI 1,668, duplicate 0 | 한 post-apply reply-loss fault의 at-most-once 격리 통과 |
| Unbounded control-path 반례 | C129 Qwen 반환 뒤 complete RPC가 97.259 ms 더 막혀 fallback +160.844 ms | broker fail-stop mode UQ |
| Former 5 ms control charge | C132–C134 safety/fail-closed는 통과했지만 C138 faulting RPC 5.831 ms | 15 ms 시간 mode는 UQ, fault semantics 자료로만 보존 |
| Full control-point coverage | C132/C133 prepare·commit·complete 각2 arm, 총16,320 TB·crash 뒤14,938 TB, duplicate/safety 위반0 | bounded volatile fail-stop lemma 유한 표본 통과 |
| Independent-node requalification | C134 다른 A100 node에서 prepare·commit·complete 각2 arm, 16,320 TB·crash 뒤14,926 TB, 위반0 | 같은 hardware family의 allocation/node 재자격 통과 |
| Corrected control bound | C139 6 arm·16,320 TB·RPC2,020, max5.653 ms<7 ms; safety 위반0 | `T_sock=5`, `B_rpc=7`, control21 finite-sample PASS |
| Sharded-home envelope v13 | QSU5/QSN0/MI3/UQ11; 과거 mode UQ, corrected mode QSU | AI35 effective58 exchange-only, AI40 effective63 탈락 |
| Corrected conditional class | C140 2 arm·2,560 TB·branch309·AI35 exchange8 | static margin−5, conditional margin0, 위반0 |
| Corrected V13 독립 node | C141 8 arm·18,880 TB·RPC2,010·AI35 exchange7 | 같은 A100 family 두 번째 node, 위반0 |
| V14 control model과 반례 | 16 state·20 transition은 violations0, two-request 회귀는 untracked held token1 재현 | timing/fault 증거 보존, V14 mode UQ |
| V15 composed finite model | fault 16 state·20 edge + ownership 4 state·8 edge, violations0 | reply-loss와 single-token invariant 전수 통과 |
| Pipelined conditional class | C145/C146 두 node·정상 arm2,560 TB·AI45 exchange12 | static margin−1, conditional margin+4, 위반0 |
| Pipelined control fault/ownership | C145/C146 fault6 arm·9,600 TB·post-fault7,463; prepare 억제537·max token1 | commit-only critical budget, local RAN 분리와 ownership branch 통과 |
| Four-point physical fault closure | C147/C148 abort2 arm·3,200 TB·post-fault2,662; C145--C148 결합 fault8 arm·radio15,360·post-fault10,125·억제666 | prepare/abort/commit/complete 각각 두 arm, V16 QSU |
| Shared-recovery global certificate/path | 3,425 state oracle; C159-Q2 actual NRx4,800; C160 state51; C161 actual NRx2,800·recovery942·event pair40·terminal fault4·miss0 | P180 variable-context와 A0--A6 qualified-node PASS; one lifecycle UQ |
| Predictive debt/time/class envelope | C162 qualified state16,023 analytic/exact mismatch0; retrospective1,200/1,200; two-node physical180/180 | warm P180/D155 QSU/QSN/MI/UQ PASS; production UQ |
| Certified scheduler/verifier scale | small exact600 false-safe0·false-conservative10; large2,800 verifier 실패0; debt64 decision p991.494ms | finite CPU control budget PASS; WCET 아님 |
| Lifecycle qualification token | C164 model unsafe admission0; idle30·MPS-restart 각각 two-node physical180·miss0; Qwen reload60 중 mandatory release3,334·miss0 | 두 first-request와 reload mandatory-continuity subset PASS; optional reload availability/cold/reconnect/full class UQ |
| Production expiry와 WCET | G7 | 미완료 |

## 10. SIGMETRICS 수준으로 남은 모델링 작업

1. **Endpoint-only v1 완료:** `theta`별 recovery, endpoint, AI와 home-memory constraint를
   받는 deterministic checker와 5개 단위시험을 구현했다.
2. **Endpoint-only grid 완료:** GPU 1/2/4, cell 4/8/12에서 4-cell QSU, 8/12-cell MI를
   예측했다. 후자는 C84 memory 경계 때문에 timing schedule을 counterfactual로 표시한다.
3. **Sharded-home v14 완료:** Lemma 1b, disjoint endpoint pool과 bound/executor/control-fault
   mode를 구현했다. C124가 C121 conv12를 UQ로 내렸고, C125 ordering 반례 뒤 C126
   conv25 certificate executor를 QSU로 올렸다. C127은 unhandled reply loss를 UQ,
   no-retry quarantine mode를 QSU로 분리했다. C129의 무제한 broker RPC와 C131의
   2-RPC 부분 예산은 UQ다. C132--134의 세 RPC 15 ms mode는 safety 의미론을 검증했지만
   C138의 5 ms wall-bound 반례 뒤 UQ로 내렸다. C139의 corrected mode는 5 ms socket
   timeout과 7 ms admission wall bound를 분리하고 세 RPC 21 ms를 청구한다. V5 checker는 conditional slack을
   home별 실제 NRx 수락 수에서 계산했고, V6는 `W+Delta`와 observe-all decision time을
   분리했다. V7은 실제 endpoint ring depth와 rejected-recovery executor phase를 추가했다.
   독립 exact oracle은 현재 39개 pool과 10,920개 bounded 상태에서 endpoint 수락
   cardinality와 schedule consistency를 검증했다. V13은 corrected-control mode에서
   완전한 58 ms AI35 transaction 후보 두 개를 유지하고 63 ms AI40을 제거한다.
   C140 두 arm에서 AI35 class는 8회 물리 실행됐다. C137의 정상-path telemetry와
   C138의 반례, C139의 fault-path qualification을 분리해 관리한다.
   V14는 prepare/abort/complete를 비동기화하고 launch-time revalidation 뒤 commit7만
   청구해 54 ms AI45 후보 두 개를 추가했다. 후속 two-request 반례로 V14를 UQ로 내리고,
   V15가 single-token ownership을 추가했다. C145/C146은 두 새 node에서 AI45 exchange,
   세 fault point와 retained-token branch를 검증했고 model v2가 bounded ambiguity와
   ownership 상태를 감사했다. C147/C148은 남은 abort post-apply fault를 두 추가 node에서
   검증했다. V16은 이 four-point physical coverage를 별도 자격 gate로 사용한다.
   C122 3-cell/home exact mode는 유지한다.
4. **Global AI ownership과 제한된 fault containment 완료:** C123은 한 AI queue를 local
   certificate와 합성했고 C127은 post-apply reply loss, C132/C133은 prepare·commit·complete
   각각의 broker process fail-stop을 처리했다. C134는 이 전체 matrix를 다른 A100 node에서
   재자격했다. 이들을 bounded control transaction으로 local RAN boundary에 격리했다.
   Shared recovery의 non-separable control-plane은 V17 후보에서 모든 home generation,
   obligation과 AI lease를 함께 검사하도록 구현했고, 3,425개 schedule state와 원자 전이
   단위시험을 통과했다. C151/C152는 shared cuPHY worker와 두-home P2P/IPC payload를 두
   node에서 통과시켰다. C153은 global placement가 worker dispatch와 Qwen lease를 한
   deadline transaction으로 구동하는 development 경로까지 통과했다. C154/C155는 같은
   controlled-outcome 경로를 두 독립 node, 반대 branch 순서, 여덟 arm으로 재현했다.
   C156은 launch control5 ms를 lease에 포함하는 V17.1로 fixed-lease 반례를 닫고 GPU
   timeline order와 blackout을 직접 검증했고 C156b가 이를 다른 node에서 재현했다.
   Actual NRx outcome과 C161 A0--A6 fault matrix는 qualified node에서 통과했다. C162는
   debt/time/class/provenance state 16,023개와 두-node physical boundary180개를 mismatch0으로
   통과했다. Certified scheduler/verifier는 64 debt decision p991.494ms를 얻었다. C164는
   qualification을 lifecycle fingerprint와 epoch/generation token에 묶고 30초-idle 및
   quiescent MPS-restart 뒤 재자격 boundary subset을 각각 두 node에서 통과했다. Restart 중
   availability와 worker process replacement는 claim 밖의 UQ로 남긴다. C164 durable reconnect
   상태모델은 token/lifecycle/payload/worker epoch와 journal sequence를 묶고, quiesced worker의
   prepared→nonlaunch-fence 또는 fenced record에서만 retire한다. 43개 전이 조합에서 retire2,
   fail-closed41, invariant violation0을 얻었다. Same-worker channel reconnect는 두 node의
   token120·physical launch60·terminal fence120과 mandatory release145에서 miss0으로 통과했다.
   Worker process replacement 단일-node development canary는 exploratory evidence일 뿐
   formal claim으로 승격하지 않는다. Process/model cold/full-class와 production
   node/lifecycle 예측은 UQ/future work다. Qwen reload 중
   mandatory continuity는 두 node·60 fresh process에서 통과했지만 optional inference는 닫았다.
5. Endpoint queue arrival을 trace에서 추출해 transport choice 식의 예측 오차를 검증한다.
6. 고장 상관도 0→all-fail sensitivity에서 reservation cost와 admitted AI value를 함께 그린다.
7. Bound uncertainty를 구간 또는 quantile sensitivity로 바꾸되, hard guarantee와 확률적
   qualification을 섞지 않는다.
8. Theorem의 가정을 runtime assertion과 artifact field에 일대일로 연결한다.

이 작업이 끝나야 “모델이 구현을 설명한다”를 넘어 “모델이 보지 않은 경계를 예측한다”는
주장을 할 수 있다.

- [C134 독립 node 결과](../archive/SOFTWALL_CONFIRM134_INDEPENDENT_NODE_RESULT_KO.md)
- [Sharded-home envelope v11](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v11.json)
- [Sharded-home envelope v12](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v12.json)
- [V12 validation summary](../../results/softwall_multigpu/softwall_envelope_v12_validation_summary.json)
- [V12 artifact manifest](../../results/softwall_multigpu/softwall_sharded_home_envelope_manifest_v12.json)
- [V11 validation summary](../../results/softwall_multigpu/softwall_envelope_v11_validation_summary.json)
- [V11 artifact manifest](../../results/softwall_multigpu/softwall_sharded_home_envelope_manifest_v11.json)
- [V10→V11 ring/phase regression](../../results/softwall_multigpu/softwall_envelope_v10_v11_ring_phase_regression_v1.json)
- [V11→V12 AI guard regression](../../results/softwall_multigpu/softwall_envelope_v11_v12_ai_guard_regression_v1.json)
- [C135 사건별 static counterfactual](../../results/softwall_multigpu/confirm135_static_counterfactual_audit_v1.json)
- [C135 물리 결과](../archive/SOFTWALL_CONFIRM135_V11_AI40_RESULT_KO.md)
- [C136 독립 node 결과](../archive/SOFTWALL_CONFIRM136_V12_REQUALIFICATION_RESULT_KO.md)
- [C135/C136 결합 감사](../../results/softwall_multigpu/confirm135_136_combined_v12_qualification.json)
- [V12 cross-node validation summary](../../results/softwall_multigpu/softwall_envelope_v12_cross_node_validation_summary.json)
- [V12 cross-node manifest](../../results/softwall_multigpu/softwall_envelope_v12_cross_node_manifest.json)
- [C137 service-bound telemetry](../archive/SOFTWALL_CONFIRM137_SERVICE_BOUND_TELEMETRY_RESULT_KO.md)
- [Service-bound qualification 방법론](SOFTWALL_SERVICE_BOUND_QUALIFICATION_KO.md)
- [Service-bound qualification v2](../../results/softwall_multigpu/softwall_service_bound_qualification_v2.json)
- [C137 artifact manifest](../../results/softwall_multigpu/confirm137_artifact_manifest.json)
- [Service-bound v2 manifest](../../results/softwall_multigpu/softwall_service_bound_v2_manifest.json)
- [C138–C140 control-bound correction](../archive/SOFTWALL_CONFIRM138_140_CONTROL_BOUND_CORRECTION_KO.md)
- [Sharded-home envelope v13](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v13.json)
- [V12→V13 control-bound regression](../../results/softwall_multigpu/softwall_envelope_v12_v13_control_bound_regression_v1.json)
- [V13 validation summary](../../results/softwall_multigpu/softwall_envelope_v13_validation_summary.json)
- [Service-bound qualification v3](../../results/softwall_multigpu/softwall_service_bound_qualification_v3.json)
- [V13 corrected-control manifest](../../results/softwall_multigpu/softwall_v13_corrected_control_manifest.json)
- [C141 독립-node 재자격](../archive/SOFTWALL_CONFIRM141_V13_INDEPENDENT_RESULT_KO.md)
- [Service-bound qualification v4](../../results/softwall_multigpu/softwall_service_bound_qualification_v4.json)
- [V14 pipelined-control 설계](../archive/SOFTWALL_PIPELINED_CONTROL_DESIGN_KO.md)
- [C142--C144 결과](../archive/SOFTWALL_CONFIRM142_144_V14_PIPELINED_RESULT_KO.md)
- [V14 prediction](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v14.json)
- [V14 validation](../../results/softwall_multigpu/softwall_envelope_v14_validation_summary.json)
- [Pipelined-control finite model](../../results/softwall_multigpu/softwall_pipelined_control_model_v1.json)
- [V14 immutable manifest](../../results/softwall_multigpu/softwall_v14_pipelined_control_manifest.json)
- [V14 ownership 반례와 V15 교정](../archive/SOFTWALL_V14_OWNERSHIP_CORRECTION_KO.md)
- [C145--C146 V15 결과](../archive/SOFTWALL_CONFIRM145_146_V15_SINGLE_TOKEN_RESULT_KO.md)
- [V14→V15 ownership regression](../../results/softwall_multigpu/softwall_v14_v15_staged_ownership_regression_v1.json)
- [Pipelined-control finite model v2](../../results/softwall_multigpu/softwall_pipelined_control_model_v2.json)
- [Finite-model semantic reproducibility](../../results/softwall_multigpu/softwall_pipelined_control_model_v2_reproducibility_v1.json)
- [V15 prediction](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v15.json)
- [V15 validation](../../results/softwall_multigpu/softwall_envelope_v15_validation_summary.json)
- [V15 immutable manifest](../../results/softwall_multigpu/softwall_v15_single_token_manifest.json)
- [C147--C148 V16 four-point 결과](../archive/SOFTWALL_CONFIRM147_148_V16_FOUR_POINT_RESULT_KO.md)
- [V16 prediction](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v16.json)
- [V16 validation](../../results/softwall_multigpu/softwall_envelope_v16_validation_summary.json)
- [V16 immutable manifest](../../results/softwall_multigpu/softwall_v16_four_point_manifest.json)
- [Decision-time audit](../../results/softwall_multigpu/softwall_decision_time_envelope_audit_v1.json)
- [Exact finite-ring endpoint oracle 감사](../../results/softwall_multigpu/softwall_endpoint_admission_exact_oracle_audit_v3.json)
- [V4→V5 home-local regression](../../results/softwall_multigpu/softwall_envelope_v4_v5_home_local_regression_v1.json)
- [V5→V6 decision-time regression](../../results/softwall_multigpu/softwall_envelope_v5_v6_decision_time_regression_v1.json)
