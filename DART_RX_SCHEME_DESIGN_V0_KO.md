# DART-Rx v0 · 실행 가능한 scheme 설계

> **Canonical novelty와 problem statement:**
> `DART_RX_CANONICAL_NOVELTY_THESIS_KO.md`  
> 본 문서의 P/F/R/C/J/L 및 S0--S10은 구현·ablation 내부 label이며 서로 다른
> proposal이 아니다. Novelty claim과 dual recovery reservation 정의는 canonical
> 문서를 우선한다.

작성일: 2026-08-13  
상태: Day-1 implement/measure specification  
관련 architecture: `ISCA_ARCHITECTURE_V2_KO.md`

Day-1 실행 결과와 수정된 결론:
`results/isca_v2/day1_20260813T0523Z/REPORT_KO.md`

> 구현 후 용어 정리: DART-Rx는 P/F/R/C/J/L 여섯 개의 독립 scheme이 아니라
> 하나의 slot transaction runtime이다. 이들은 각각 profile, feasibility,
> reservation, commit, fallback, lease를 담당하는 내부 stage이고 P2P/GDR는
> 교체 가능한 transport adapter다.

machine-readable ablation: `DART_RX_SCHEME_ABLATION_V0.csv`

## 0. scheme의 정확한 역할

DART-Rx는 MIG topology를 바꾸는 scheduler가 아니다. 고정된 L1 partition과
resident NRx endpoint pool 사이에서 각 slot을 하나의 **deadline transaction**으로
실행한다.

```text
radio/MAC lookahead
        |
        v
 [DART-F admission] ---- reject ----> conventional path
        |
        | reserve endpoint + tensor slot
        v
 [P2P/GDR dispatch] --> [resident NRx endpoint]
        |                         |
        |                         v
        |                  private result slot
        |                         |
        v                         v
 [DART-C watchdog] <------ completion(epoch)
        |
        +-- latest-safe-start --> conventional fallback
        |
        v
 atomic winner commit --> LDPC/CRC

 [DART-L] leases only bounded endpoint slack to background AI
```

핵심은 remote execution, fallback, background sharing을 따로 구현하지 않고
하나의 transaction state와 deadline으로 묶는 것이다.

이 scheme은 `predicted-finish scheduler` 하나와 같지 않다. scheduler가 endpoint를
잘 골라도 concurrent submit의 over-admission, late result의 stale-buffer overwrite,
endpoint failure, conventional fallback의 시작 시각, 이미 enqueue된 background
kernel의 blocking은 해결되지 않는다. DART-Rx의 단위는 routing decision이 아니라
reservation부터 single-winner commit까지의 transaction이다.

## 1. scheme 구성 요소

### DART-P · Profile table

다음 key별 conservative profile을 유지한다.

```text
(endpoint_id, tensor_class, graph_id, background_mode)
```

값:

- forward transport p99
- NRx service p99
- backward transport p99
- control/launch p99
- prediction positive-error p99
- background non-preemptible residual bound

초기 profile은 isolated measurement로 만든다. online에서는 최근 window의
quantile과 prediction error를 업데이트하되, 값이 낮아지는 속도에는 제한을 둔다.
갑작스러운 optimistic profile 때문에 deadline을 놓치지 않도록 한다.

### DART-F · Feasibility router

각 NRx request의 deadline을 만족할 endpoint만 선택한다.

### DART-R · Reservation/credit

endpoint마다 다음을 관리한다.

```text
tail_e                 predicted end of reserved work
free_input_slots_e     registered input ring credits
free_output_slots_e    registered output ring credits
health_epoch_e         endpoint generation
blocking_until_e       currently running non-preemptible quantum bound
```

queue tail을 읽기만 하고 여러 request를 동시에 보내면 over-admission이 생긴다.
따라서 endpoint 선택과 동시에 tail/tensor slot을 atomic reserve한다. reservation에
실패하면 모든 candidate의 feasibility를 다시 계산한다.

### DART-C · Commit guard

NRx와 conventional receiver는 각자 private output slot에 쓴다. downstream은
atomic winner descriptor가 가리키는 buffer만 읽는다.

### DART-J · Just-in-time fallback

request arrival 시 conventional receiver를 무조건 중복 실행하지 않는다. NRx가
제시간에 끝날 가능성이 있는 동안 기다리고, 마지막 안전 시점에만 conventional
graph를 실행한다.

### DART-L · Grant-ahead background lease

RAN의 UL grant/slot calendar는 가까운 미래의 work release를 알려준다. DART-L은
이 lookahead와 admitted NRx latest-start를 이용해 endpoint slack을 background
graph unit에 빌려준다.

일반 GPU utilization predictor가 아니라 **RAN schedule로 끝 시각이 정해진
lease**라는 점이 중요하다.

### DART-Q · Command/completion primitive

위 기능을 persistent polling 없이 실행하기 위한 hardware queue다. Day-1에는
host와 device-emulated scheme을 먼저 구현하고 control gap을 측정한다.

## 2. request descriptor

```c
enum dart_flags : uint8_t {
    DART_ALLOW_REMOTE   = 1 << 0,
    DART_ALLOW_FALLBACK = 1 << 1,
    DART_REQUIRE_NRX    = 1 << 2,
};

struct alignas(64) dart_request {
    uint64_t slot_id;
    uint32_t epoch;
    uint16_t graph_id;
    uint16_t tensor_class;

    uint64_t release_ns;
    uint64_t deadline_ns;
    uint64_t fallback_latest_start_ns;

    uint32_t input_slot;
    uint32_t output_slot;
    uint16_t candidate_bitmap;
    uint8_t  utility_class;
    uint8_t  flags;

    uint32_t payload_checksum;
    uint32_t reserved;
};
static_assert(sizeof(dart_request) == 64);
```

실제 pointer와 rkey는 startup registration table에 둔다. request descriptor가
process-specific raw pointer를 직접 운반하지 않는다.

## 3. deadline 계산

deployment deadline `D`는 임의로 1 ms라고 두지 않는다. Aerial pipeline에서
NRx가 사용할 수 있는 stage budget을 측정하고 sensitivity `{2,5,10 ms}`도 함께
보고한다.

fallback latest start:

```text
LFS(r) = D(r) - C_conv^99(tensor_class) - G_commit
```

endpoint predicted finish:

```text
ready_e = max(now, tail_e, blocking_until_e)

F_e(r) = ready_e
       + T_fwd^99(e,t)
       + S_nrx^99(e,t,bg)
       + T_bwd^99(e,t)
       + T_control^99(e)
       + E_positive^99(e,t,bg)
```

NRx가 fallback과 겹쳐도 되는 경우의 feasibility:

```text
F_e(r) <= D(r) - G_commit
```

fallback 실행 전 NRx 완료를 기대하는 stricter feasibility:

```text
F_e(r) <= LFS(r) - G_notify
```

Day-1에는 둘 다 구현하여 aggressive/conservative admission을 비교한다.

## 4. endpoint 선택 algorithm

```text
procedure SUBMIT(r):
    if radio_utility(r) < U_min and not REQUIRE_NRX:
        schedule_conventional(r)
        return CONVENTIONAL_ONLY

    candidates = []
    for e in enabled_endpoints(r.candidate_bitmap):
        if no_credit(e) or unhealthy(e):
            continue
        finish = predict_finish(e, r)
        if finish <= r.deadline - commit_guard:
            lost_bg = lease_displacement_cost(e, r)
            score = finish
                  + alpha * lost_bg
                  + beta  * energy_cost(e, r)
                  + gamma * transport_risk(e, r)
            candidates.append(e, score)

    for e in candidates ordered by score:
        if atomic_reserve(e, r):
            publish_payload_and_descriptor(e, r)
            arm_watchdog(r)
            return ADMITTED(e)
        recompute_candidates()

    schedule_conventional(r)
    return REJECTED_TO_CONVENTIONAL
```

Day-1의 기본 score는 `earliest predicted finish`다. `alpha/beta/gamma`를 처음부터
학습시키지 않는다. lost-background utility와 energy term은 ablation에서 하나씩
추가한다.

## 5. endpoint service order

endpoint queue는 FIFO 대신 `earliest latest-start first`를 사용한다.

```text
latest_start_nrx(r,e) = deadline(r)
                      - remaining_service_bound(r,e)
                      - commit_guard
```

동일 latest-start에서는 먼저 도착한 request를 선택한다. 이미 시작한 TensorRT
graph는 preempt할 수 있다고 가정하지 않는다.

head-of-line blocking을 줄이기 위해 tensor class가 다른 request reorder를 허용하되,
reservation 시 계산한 predecessor set을 event log에 저장한다.

## 6. transaction state machine

Request state:

```text
FREE
  -> RESERVED
  -> DISPATCHED
  -> NRX_RUNNING
  -> NRX_READY
  -> CLOSED
```

Commit state는 독립적으로 관리한다.

```text
OPEN
  -> FALLBACK_RUNNING
  -> NRX_WON | CONV_WON | DEADLINE_MISS
  -> CLOSED
```

`FALLBACK_RUNNING` 상태에서도 deadline 전 NRx가 먼저 끝나면 NRx가 이길 수 있다.
winner는 다음 atomic word 하나로 결정한다.

```text
winner = {slot_id_low, epoch, kind, result_slot}
```

### NRx completion

```text
1. payload DMA/kernel write 완료
2. device/NIC visibility fence
3. completion의 slot_id와 epoch 검증
4. now <= deadline 검증
5. CAS(OPEN or FALLBACK_RUNNING -> NRX_WON)
6. 실패하면 private result를 drop/recycle
```

### Fallback watchdog

```text
at LFS:
    if state == OPEN:
        CAS(OPEN -> FALLBACK_RUNNING)
        launch conventional graph

on conventional completion:
    CAS(OPEN or FALLBACK_RUNNING -> CONV_WON)
```

### Deadline watchdog

deadline에 winner가 없으면 `DEADLINE_MISS`로 닫는다. 이후 도착한 모든 completion은
epoch가 맞더라도 commit하지 않는다.

## 7. background lease algorithm

### 7.1 background unit contract

DART-L이 관리하는 background application은 model 전체를 한 번에 enqueue하지
않고 bounded graph unit을 제공해야 한다.

```text
unit_id, graph_exec, p99_duration, application_value, preemption_safe_boundary
```

예:

- Qwen: decode step 또는 transformer layer group
- training: micro-batch
- video: frame/engine segment
- speech: encoder chunk/decoder step

unmodified application이 unbounded work를 미리 enqueue하면 strict DART guarantee를
제공하지 않는다. 이를 compatibility limitation으로 명시한다.

### 7.2 lease window

```text
next_guard_e = min(
    earliest admitted NRx latest-start on e,
    earliest grant-ahead request latest-start,
    now + Q_max
)

slack_e = next_guard_e - max(now, tail_e) - drain_guard
```

`Q_max`는 unexpected arrival의 maximum blocking을 제한한다.

### 7.3 admission

```text
procedure TRY_LEASE(e, units):
    slack = compute_slack(e)
    feasible = [u for u in units if p99(u) + error_guard(u) <= slack]
    if empty(feasible): return NO_LEASE
    u = argmax(application_value(u) / p99(u))
    reserve tail_e for u
    dispatch u with lease_expiry=now+p99(u)+guard
```

새 NRx가 lease 중 도착해도 blocking은 선택한 unit bound 이하이다. DART-L은
kernel preemption을 claim하지 않는다.

## 8. service-time adaptation

각 completion 후:

```text
error = actual_finish - predicted_finish
histogram[key].add(actual_component_times)
positive_error_hist[key].add(max(0,error))
```

profile update:

- 상승: 즉시 반영
- 하강: 최소 1,000 samples와 rate limit 후 반영
- background class/quantum이 바뀌면 별도 key
- thermal/clock state가 manifest 범위를 벗어나면 profile invalid

underprediction이 연속 `K`회 발생하면 endpoint를 `DEGRADED`로 표시하고
conservative profile 또는 conventional-only로 전환한다.

## 9. transport integration

P2P와 GDR 모두 같은 DART transaction API를 쓴다.

```text
reserve_tensor_slot()
write_payload()
publish_descriptor_after_payload()
receive_completion_after_result_visibility()
release_slot_after_commit_or_drop()
```

P2P:

- `cudaMemcpyPeerAsync` 또는 peer-access kernel
- CUDA event/stream ordering

GDR:

- startup GPU MR registration
- payload WRITE 후 small descriptor/sequence WRITE
- GPU visibility flush
- completion immediate/doorbell

transport에 따라 policy semantics가 바뀌면 안 된다.

## 10. v0 implementation boundary

오늘 실제 구현하는 범위:

### v0-H host

- descriptor와 endpoint table
- atomic reservation
- predicted-finish admission
- shortest-queue/round-robin baseline
- private result slots와 epoch validation
- watchdog/JIT fallback
- trace/event output

### v0-D device

- GPU-visible ring
- descriptor observe timestamp
- device atomic commit micro-path
- stale/duplicate completion fault test
- device/conditional graph capability가 허용하는 최소 launch path

### v0-L lease

- Qwen decode step과 training micro-batch 두 unit class
- fixed p99 profile
- grant-ahead trace replay
- `Q_max` sweep

오늘 하지 않는 것:

- learned routing policy
- full DART-Q RTL
- 모든 application의 arbitrary kernel preemption
- radio utility model 학습

## 11. scheme ablation

| Variant | Mechanism | 질문 |
|---|---|---|
| S0 Static-1 | one endpoint FIFO | fixed isolation의 한계 |
| S1 Round-robin | multiple endpoints | capacity만 늘리면 충분한가 |
| S2 Shortest-queue | observed queue | simple work conservation |
| S3 Predicted-finish | p99 service+transport | heterogeneity/deadline awareness |
| S4 +Reservation | atomic tail/credit | over-admission 제거 |
| S5 +Epoch commit | private slot+CAS | late/stale correctness |
| S6 +JIT fallback | latest-safe-start | miss 차단 대 duplicate work |
| S7 +Fixed quantum | constant background unit | 단순 cooperative gating |
| S8 +Grant lease | RAN lookahead+p99 unit | background utility 회수 |
| S9 Device DART | device observe/commit | host control tail 제거 |
| S10 DART-Q model | command/completion hardware | polling resource 제거 |

각 row는 이전 row에 mechanism 하나만 추가한다.

## 12. parameter sweep

### Admission

- average, p95, p99, p99.9 profile
- prediction error guard 0/5/10/20%
- queue depth 1/2/4/8/16

### Fallback

- commit guard 10/25/50/100 us
- conventional p99 multiplier 1.0/1.1/1.25
- aggressive/conservative admission

### Lease

- Qmax 25/50/100/250/500 us와 actual application units
- RAN lookahead 0/1/2/4 slots
- background offered load 30/50/70/90%

### Endpoint

- 1/2/3 resident endpoints
- P2P/GDR
- endpoint capacity heterogeneity
- injected slowdown/failure

## 13. scheme experiment와 claim 연결

| 실험 | 설계에 주는 값 | 검증 mechanism |
|---|---|---|
| NRx service profile | `S_nrx^99`, error | DART-P/F |
| transport depth | `T_fwd/bwd^99` | DART-F |
| residual injection | `blocking_until`, Qmax | DART-L |
| control-path CDF | control guard | v0-D/DART-Q |
| endpoint scale-out | credit/tail capacity | DART-R |
| fallback calibration | `C_conv^99`, LFS | DART-J |
| stale/fault injection | epoch/CAS | DART-C |
| background app profile | unit p99/value | DART-L |

즉 characterization을 별도로 하고 나중에 heuristic을 만드는 것이 아니다. 각
profile 결과가 바로 scheme table/guard로 들어가고, 다음 experiment에서 그
mechanism을 검증한다.

## 14. Day-1 success

오늘 scheme 관점의 성공 조건:

1. 동일 trace replay에서 reservation/admission 결정이 deterministic하다.
2. duplicate/stale/delayed completion 1,000건에서 wrong commit이 0이다.
3. predicted-finish가 round-robin/shortest-queue보다 deadline miss를 줄인다.
4. JIT fallback이 block-on-NRx보다 miss를 줄이고 immediate-dual보다 GPU work를
   줄인다.
5. Qwen 또는 training lease가 L1/NRx deadline을 유지하면서 nonzero application
   utility를 회수한다.
6. device path의 control p99와 resource cost를 host path와 비교할 수 있다.

결과가 나쁘더라도 scheme을 유리하게 바꾸기 전에 raw event와 failed hypothesis를
보존한다.
