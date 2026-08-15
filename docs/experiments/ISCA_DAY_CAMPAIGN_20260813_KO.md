# MIGRx ISCA 실험 캠페인 · 하루 실행판

> **상태: v1 / 대체됨.** 이 문서는 초기 software-router 가설을 보존하기 위한 기록이다.
> 논문 방향과 실행 우선순위는 `ISCA_ARCHITECTURE_V2_KO.md`와
> `ISCA_EXPERIMENT_PLAN_V2_KO.md`를 따른다.

작성일: 2026-08-13  
대상: CloudLab d8545 · A100-SXM4-40GB 4장 · ConnectX-6 Dx · Aerial 25.3.2  
목표 venue: ISCA급 architecture/system paper  
상태: 기존 characterization을 보존하고, mechanism을 뒷받침할 kernel/API 증거를 수집하는 단계

## 0. 오늘의 결론부터

오늘 검증할 architecture thesis는 다음 한 문장이다.

> 고정 MIG는 L1을 공간적으로 보호하지만 NRx burst를 흡수할 service capacity와 이미
> 실행 중인 background kernel을 선점할 수단은 제공하지 않는다. MIGRx는 topology를
> 바꾸지 않고 resident NRx endpoint, GPU-resident transport, deadline-aware routing,
> bounded background work lease를 결합하여 이 isolation-elasticity gap을 메운다.

오늘은 `MIG vs MPS vs P2P vs GDR` 막대그래프를 더 만드는 날이 아니다. 다음 인과관계를
각각 hardware trace로 닫는 날이다.

```text
공간 격리는 정상이다
        ↓
고정 endpoint의 service rate를 넘으면 queue가 발산한다
        ↓
MIG 재구성은 수 초라 fast path가 될 수 없다
        ↓
남는 partition에 NRx와 background AI를 resident로 둬야 한다
        ↓
그러나 긴 AI kernel은 priority만으로 선점되지 않는다
        ↓
AI를 bounded quantum으로 만들고, deadline slack이 있을 때만 lease한다
        ↓
resident NRx endpoint를 P2P/GDR로 묶어 burst 때 수평 확장한다
        ↓
예상 완료가 deadline을 넘으면 conventional receiver로 fallback한다
```

오늘 종료 시 반드시 있어야 할 산출물은 다섯 개다.

1. Qwen/NRx의 CUDA API와 kernel taxonomy, kernel duration CDF
2. NRx arrival이 background kernel 때문에 기다린 정확한 blocking time
3. background quantum 크기와 NRx p99/Qwen utility 사이의 Pareto curve
4. fixed topology에서 lease controller가 queue cliff를 막는 timeline
5. 위 결과가 MIGRx의 각 mechanism을 정당화하는지에 대한 Go/No-Go 판정

오늘 하루에 radio-aware predictor와 production C++ Aerial integration까지 완성되었다고
주장하지 않는다. 그것들은 오늘의 mechanism gate가 통과한 뒤 수행할 paper campaign이다.

## 1. 기존 증거와 아직 비어 있는 증거

| Pathology | 현재 관측 | 현재 상태 | 오늘 필요한 증거 | 대응 mechanism |
|---|---:|---|---|---|
| P0. wrapper artifact | public wrapper 약 103 ms, direct TRT+Graph 4g 약 1.34 ms | 원인 일부 확인 | API/kernel별 시간과 copy/sync 개수 | caller-owned bindings, persistent buffers, CUDA Graph |
| P1. fixed capacity | 4g NRx 약 745 slot/s | 확인 | clock-fixed 재현 및 service CDF | resident independent endpoint pool |
| P2. queue cliff | 750/s p99 15–19 ms, 800/s p99 214–218 ms | 확인 | periodic/synchronized arrival에서도 재현 | finish-time routing, admission, fallback |
| P3. topology change | warmed endpoint outage 평균 3.643 s | 확인 | 반복 불필요, 대조군으로만 사용 | measured interval topology invariant |
| P4. background blocking | adaptive reclaim 5.53–16.40 ms | 현상만 확인 | 어떤 AI kernel/API가 얼마를 막았는지 | bounded AI graph/kernel lease |
| P5. transport | P2P RTT 76.84 us, GDR full-chain P2P 대비 +0.438 ms | component gate | 동일 depth·payload·load 공정 비교 | topology-aware GPU-resident path |
| P6. radio utility | 아직 없음 | 치명적 공백 | paired conventional/NRx CRC·BLER dataset | utility-aware admission/fallback |
| P7. actual deadline | 10 ms queue target만 사용 | 치명적 공백 | Aerial schedule/HARQ budget 확인 | absolute-deadline contract |

ISCA 관점에서 오늘의 최우선 공백은 P4다. `CUDA stream priority`는 pending kernel 선택에
영향을 주는 hint이지 이미 실행 중인 kernel을 보장된 방식으로 선점하지 않는다. 그러므로
5–16 ms reclaim delay를 kernel-level에서 설명하지 못하면 bounded lease 설계의 근거가 없다.

P6와 P7이 끝나기 전에는 논문에서 `hard real-time RAN deadline`이나 `radio-optimal`을
완료된 claim으로 쓰지 않는다.

기존 `QUANTUM_SWEEP`은 Qwen full-forward의 sequence length `64/128/256/512`를 바꾼
실험이다. whole forward가 한 cooperative boundary였으므로 blocking-size sensitivity를
보이는 데는 유효하지만, model을 layer/prefill micro-quantum으로 분할한 결과는 아니다.
논문에서 이를 `bounded micrograph` 결과로 부르지 않는다. 또한 현재 P2P/GDR full-chain
timing harness는 random receive tensor를 LDPC/CRC API까지 통과시키지만 transmitted
ground-truth TB와 CRC success를 비교하지 않는다. 따라서 이는 API-complete timing path이지
radio-correct full-chain evidence가 아니다.

## 2. 제안 architecture: MIGRx

### 2.1 배치

Primary deployment는 다음처럼 고정한다.

```text
Physical GPU 0: fixed MIG geometry, run 중 변경 금지

  4g protected partition
  ┌──────────────────────────────────────────────────────┐
  │ cuPHY L1 front/CE ─ Router ─ L1 back/LDPC/CRC       │
  │             └ conventional receiver fallback        │
  └──────────────────────────────────────────────────────┘
                │ slot descriptor + GPU tensor
                │
        ┌───────┴─────────┬─────────────────────┐
        │                 │                     │
  sibling 3g         remote endpoint 1     remote endpoint N
  ┌──────────────┐   ┌────────────────┐    ┌────────────────┐
  │ resident NRx │   │ resident NRx   │    │ resident NRx   │
  │ resident AI  │   │ optional AI    │    │ optional AI    │
  │ bounded lease│   │ bounded lease  │    │ bounded lease  │
  └──────────────┘   └────────────────┘    └────────────────┘
       P2P upper bound      GDR/P2P path           GDR/P2P path
```

현재 node의 GPU0은 `4g+3g`, GPU1–3은 full GPU다. 오늘 policy gate에서는 기존
4g-primary-NRx/3g-overflow-NRx+Qwen harness를 재사용할 수 있지만, 이것은 L1이 포함되지
않은 compute-only 축소 모델이다. 논문 primary topology는 반드시 `4g L1-only + 3g NRx`
또는 실제 service map으로 정한 동등한 protected-L1 배치를 사용한다.

같은 physical GPU의 서로 다른 MIG GI 사이 P2P는 R570 이상에서 지원되지만, 서로 다른 GI
사이 CUDA IPC는 지원되지 않는다. 따라서 다음처럼 deployment 의미를 분리한다.

- 한 service process가 두 device context를 소유할 수 있는 경우: same-GPU P2P fast path
- process/container isolation이 필요한 경우: NIC GDR service path
- CPU shared memory: compatibility 및 CPU-bounce 대조군
- 다른 physical GPU/MIG 조합에서 P2P가 지원되지 않는 경우: GDR

현재 GPU0–NIC 및 GPU0–GPU1/2/3은 모두 `SYS` topology다. 이 node의 GDR 절대 latency는
PCIe-local NIC 시스템의 대표값이 아니라 구현 가능한 conservative point로 보고한다.

### 2.2 Control plane

각 NRx endpoint `e`는 run 전에 다음을 resident 상태로 만든다.

- TensorRT engine와 execution context
- caller-owned input/output GPU buffers
- captured CUDA Graph
- slot ring와 sequence/generation marker
- P2P mapping 또는 GDR MR/QP
- class별 service-time quantile table

각 job `j`는 최소 다음 metadata를 가진다.

```text
j = {cell, slot, grant, arrival, absolute_deadline,
     tensor_class, radio_features, generation}
```

endpoint별 predicted finish는 다음으로 정의한다.

```text
F_e(j) = max(now, V_e)
         + Q99(T_fwd[e, class])
         + Q99(S_nrx[e, class])
         + Q99(T_bwd[e, class])
         + Q99(S_l1_back[class])

slack_e(j) = deadline_j - F_e(j) - guard_e
```

`V_e`는 이미 배정한 작업의 virtual finish time이다. `guard_e`에는 service prediction
오차, clock 변동, polling jitter의 high quantile을 포함한다.

선택 규칙은 첫 구현에서 단순하고 검증 가능하게 유지한다.

1. `slack_e(j) >= 0`인 NRx endpoint만 eligible로 둔다.
2. eligible endpoint 중 `F_e(j)`가 가장 빠른 곳을 선택한다.
3. 동시에 대기 중인 job은 absolute deadline 순으로 처리한다.
4. 여러 eligible job이 자원을 경쟁하면 predicted NRx radio gain per service-time을 tie-break로 쓴다.
5. 어떤 NRx path도 feasible하지 않으면 conventional path의 finish를 계산한다.
6. conventional만 feasible하면 즉시 fallback한다. 이미 늦은 NRx result를 queue에 남기지 않는다.

첫날에는 radio gain을 상수로 두어 deadline-only controller를 검증한다. Radio dataset이
생긴 뒤 `gain_hat(j)`을 넣는다. RL은 필요하지 않다.

#### Just-in-time receiver commit

Remote/sibling NRx가 예상보다 늦어져도 protected L1을 기다리게 하면 service fabric이
deadline risk를 L1에 전파한다. 이를 막기 위해 각 job에 receiver commit time을 둔다.

```text
commit_j = deadline_j
           - Q99(S_conventional[j])
           - Q99(S_l1_back[j])
           - guard_l1
```

- NRx result가 `commit_j` 전에 도착하면 NRx LLR을 선택한다.
- NRx가 아직 끝나지 않았으면 그 시점에 reserved conventional graph를 launch한다.
- conventional launch 뒤 도착한 NRx는 generation을 확인하고 폐기한다.
- deadline까지 conventional조차 불가능한 job은 admission 시점에 inevitable miss로 표시한다.

이는 NRx를 mandatory blocking stage가 아니라 deadline-bounded quality enhancement로 바꾼다.
항상 conventional을 동시에 실행하는 speculation은 안전하지만 L1 GPU work를 낭비한다.
MIGRx는 measured conventional p99로 latest-safe-start를 계산하여 그 낭비를 줄인다.
실제 deadline이 확정되기 전 오늘은 2/5/10 ms synthetic budget에서 mechanism property만
검증한다.

### 2.3 Bounded temporal lease

NRx와 Qwen이 같은 opportunistic partition에 있을 때 spatial isolation은 없다. 이
partition에서는 다음 state machine으로 time을 공유한다.

```text
IDLE ──safe slack──> AI_LEASE ──lease end──> IDLE
  │                     │ NRx pressure
  └──NRx arrival──> NRX_ACTIVE <──QUIESCING
```

중요한 것은 `pause()` 호출 자체가 아니라 이미 제출된 GPU work의 최대 잔여 시간이다.

```text
H_e = next_known_arrival
      - now
      - Q99(T_fwd + S_nrx + T_bwd + S_l1_back)
      - guard_e

AI quantum q is admissible iff
Q99(q_gpu_time) + Q99(revoke_to_idle) <= H_e
and projected endpoint utilization <= rho_guard.
```

RAN slot boundary와 scheduled grant는 Poisson request와 달리 가까운 미래의 arrival을 알 수
있다. MIGRx는 이 deterministic look-ahead를 이용해 idle window보다 짧은 AI quantum만
enqueue한다. NRx pressure가 오면 새 quantum 제출을 멈추고 현재 quantum 종료 뒤 NRx를
실행한다. kernel preemption을 주장하지 않는다.

Qwen 실행 단위 후보는 다음 순서로 구현한다.

1. decode token boundary: correctness가 쉬운 baseline, quantum이 길 수 있음
2. prefill chunk: fixed token bucket `64/128/256/512`
3. transformer layer group: `1/2/4/8/whole` layers
4. fixed-shape CUDA Graph bucket: KV length bucket별 capture

Layer-group executor는 full Hugging Face forward의 logits와 `allclose` 및 next-token exact
match를 통과해야 한다. Dynamic KV length를 하나의 graph로 억지 capture하지 않고,
고정 length bucket별 graph 또는 eager fallback을 사용한다.

### 2.4 Data-plane contract

- payload는 CPU를 통하지 않는 persistent GPU ring에 둔다.
- payload completion 뒤 같은 ordered path에서 sequence/generation marker를 publish한다.
- consumer는 marker와 generation이 모두 일치할 때만 tensor를 사용한다.
- endpoint queue는 유한하다. full이면 무한 대기하지 않고 router로 backpressure를 보낸다.
- deadline 이후 도착한 NRx output은 폐기하며 LDPC/CRC에 전달하지 않는다.
- endpoint failure, CQ timeout, checksum/finite check 실패는 conventional fallback으로 연결한다.

P2P와 GDR의 목적은 single-slot latency 경쟁이 아니라 resident endpoint를 하나의 bounded
service pool로 만드는 것이다.

### 2.5 ISCA novelty guardrail

다음 mechanism은 그 자체로 novelty가 아니다.

- service-time prediction과 SLO scheduling: Clockwork 등 DNN serving 연구가 존재
- generic kernel priority/slicing/preemption: REEF, Orion, Tally 등이 존재
- fixed MIG 사이 분산 실행: Flex-MIG가 존재
- MIG로 baseband와 AI를 격리: AI-RAN fixed-partition 연구가 존재

따라서 논문의 architecture contribution은 다음 묶음으로만 방어한다.

1. `cuPHY front → selectable receiver → LDPC/CRC`라는 dependent GPU service chain을
   protected L1과 isolated NRx endpoint 사이에 실제 tensor로 연결
2. 늦은 neural result가 PHY deadline을 전파하지 못하도록 하는 just-in-time conventional
   receiver commit과 generation-safe late-result discard
3. deterministic slot look-ahead, measured NRx service, predicted radio gain을 함께 쓰는 admission
4. static spatial isolation 위에서만 동작하는 bounded AI lease와 resident NRx scale-out
5. Aerial/TensorRT/P2P/GDR hardware implementation과 p99/CRC/BLER evidence

즉 `bounded Qwen quantum`은 contribution 하나가 아니라 JIT receiver architecture가
deadline-safe하게 남는 accelerator time을 회수하기 위한 mechanism이다. Radio correctness와
commit 실험이 없으면 논문은 generic GPU sharing과 충분히 구분되지 않는다.

## 3. 측정 원칙

### 3.1 측정 run과 profiling run 분리

프로파일러가 붙은 결과를 primary latency로 사용하지 않는다.

| Run 종류 | 목적 | 반복 | profiler |
|---|---|---:|---|
| Timing | p50/p95/p99, throughput, miss ratio | 오늘 3회, paper 5회 이상 | 없음, CUDA event + monotonic clock만 |
| Nsys graph | API/kernel concurrency와 critical path | 대표 config 1회, 50–200 job | graph-level |
| Nsys node | CUDA Graph 내부 node와 blocking | 대표 window 5–20 job | node-level, 높은 overhead 허용 |
| NCU | top kernel의 SM/memory/occupancy 원인 | isolated kernel 1–3 launch | background workload 금지 |
| CUPTI production trace | 저비용 장기 correlation | 후속 구현 | concurrent-kernel activity |

### 3.2 clock와 환경

각 run은 시작 전에 다음 manifest를 저장한다.

- git/source SHA256, container image ID
- driver/CUDA/TensorRT/Aerial/Nsys/NCU version
- physical GPU UUID, MIG UUID/profile/placement, CUDA ordinal, SM count
- `nvidia-smi topo -m`, NIC/GID/link state
- GPU clock, power, temperature, ECC, running process
- CPU affinity와 host load

현재 supported maximum은 memory 1215 MHz, graphics 1410 MHz다. timing campaign은 먼저
idle 상태에서 clock-lock 권한을 smoke하고, 성공한 GPU만 1410 MHz로 고정한다. 모든
정책에 같은 clock policy를 적용한다. lock 실패 시 application clock을 섞지 말고
warm-up 후 실제 clock을 각 sample과 함께 기록한다.

GPU0 local CPU affinity는 `18-23,66-71`, NUMA affinity는 3이다. `numactl`은 현재 host에
없으므로 오늘은 `taskset`으로 router/L1 polling thread를 해당 CPU set에 pin하고, NUMA
memory policy는 limitation으로 남긴다.

### 3.3 arrival와 deadline

Primary arrival는 open-loop다. 이전 job completion이 다음 arrival을 늦추는 closed-loop
loop를 primary queue 결과에 사용하지 않는다.

- periodic slot boundary: 1.0 ms 후보, 실제 numerology 확인 후 고정
- synchronized multi-cell batch: slot마다 `1/2/4/8` grants
- normalized load: calibrated aggregate capacity의 `0.5/0.75/0.9/1.0/1.1/1.25x`
- trace: low → short burst → low, periodic burst, synchronized cell burst
- Poisson: robustness ablation만 수행

실제 Aerial/HARQ budget이 확인되기 전 오늘은 deadline `2/5/10 ms` sensitivity로 mechanism을
검증하며, 이것을 5G deadline 준수 claim으로 쓰지 않는다.

### 3.4 통계

- 오늘 core sweep: 동일 trace/seed로 paired 3 trials
- paper result: randomized order, 5 trials 이상, 합계 100,000 job 이상
- p50/p95/p99/max, deadline miss, consecutive miss burst, bootstrap 95% CI
- 0 miss는 sample count와 one-sided upper bound를 함께 보고
- overload run은 측정 window 끝 backlog와 drain time을 반드시 기록
- 평균 service time만으로 안정성을 판정하지 않음

## 4. CUDA API·kernel instrumentation

### 4.1 NVTX domain과 range

모든 실행은 아래 이름을 공통으로 사용한다. 이름이 바뀌면 분석 script가 실패하도록 한다.

| Domain | Range | 의미 |
|---|---|---|
| `MIGRX.SLOT` | `slot_total` | arrival부터 usable result까지 |
| `MIGRX.L1` | `l1_front_ce` | common front/channel estimate |
| `MIGRX.ROUTER` | `route_decision` | state snapshot과 endpoint 결정 |
| `MIGRX.TRANSPORT` | `fwd_p2p`, `fwd_gdr`, `bwd_p2p`, `bwd_gdr` | transport submission/completion |
| `MIGRX.NRX` | `nrx_enqueue`, `nrx_graph`, `nrx_sync` | TensorRT critical path |
| `MIGRX.L1` | `l1_back_ldpc_crc` | common back stage |
| `MIGRX.AI` | `ai_quantum`, `ai_prefill`, `ai_decode`, `ai_quiesce` | background work와 회수 |
| `MIGRX.CTRL` | `lease_admit`, `lease_reject`, `fallback` | control action |

Range text에는 최소 `run`, `slot`, `cell`, `endpoint`, `generation`, `policy`를 넣는다.
현재 `airan:25-3-final`에는 별도 Python `nvtx` package가 없고 CuPy NVTX API만 있다.
따라서 첫날 구현은 default domain에서 range name 앞에 domain을 붙이며, package를 image에
추가하거나 NVTX C API를 감싼 뒤 실제 domain으로 승격한다. Structured payload를 지원하지
않는 경로에서는 다음 고정 grammar를 쓴다.

```text
MIGRX.SLOT::slot_total|run=R|slot=42|cell=1|ep=E0|gen=3|policy=migrx
```

### 4.2 slot event schema

각 job에 대해 host monotonic raw timestamp와 CUDA event elapsed를 함께 기록한다.

```text
run_id,trial,seed,policy,topology_hash
cell_id,slot_id,grant_id,tensor_class,generation
arrival_ns,absolute_deadline_ns,route_start_ns,route_end_ns
endpoint_id,path,queue_depth_at_admit,virtual_finish_ns,predicted_finish_ns
fwd_submit_ns,fwd_done_ns,nrx_submit_ns,nrx_gpu_ms,nrx_done_ns
bwd_submit_ns,bwd_done_ns,l1_back_done_ns,completion_ns
deadline_miss,fallback_reason,stale_drop,crc_ok
ai_state,ai_quantum_id,ai_kernel_residual_us
```

CUDA event는 동일 context/stream 안의 GPU elapsed를 재는 데 사용한다. 서로 다른 GPU의
event timestamp를 직접 빼지 않는다. Cross-device end-to-end는 synchronized host
`CLOCK_MONOTONIC_RAW`와 protocol marker로 잰다.

### 4.3 lease event schema

```text
run_id,endpoint_id,quantum_id,kind,shape_bucket
decision_ns,start_ns,end_ns,gpu_ms
predicted_p99_us,observed_us
headroom_us,admitted,reject_reason
stop_request_ns,quiescent_ns,revoke_to_idle_us
tokens_or_layers,ai_utility
```

### 4.4 Nsight Systems pass

Node image에는 Nsys 2025.3.1이 있다. 대표 timing과 별도로 아래 두 pass를 쓴다.

```bash
# graph-level: 50~200 job critical window
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none \
  --cuda-graph-trace=graph:host-only \
  --cuda-event-trace=true \
  --stats=true --force-overwrite=true \
  --output=/results/nsys/<run>_graph \
  <instrumented command>

# node-level: 5~20 job만, overhead가 크므로 원인 규명 전용
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node:host-only \
  --cuda-event-trace=true \
  --stats=true --force-overwrite=true \
  --output=/results/nsys/<run>_node \
  <short instrumented command>
```

긴 process는 code에서 `cudaProfilerStart/Stop`으로 burst window만 감싸고
`--capture-range=cudaProfilerApi --capture-range-end=repeat:3`를 사용한다. GPU metric
sampling은 별도 짧은 pass에서만 켠다.

SQLite export에서 최소 다음을 추출한다.

- `CUPTI_ACTIVITY_KIND_KERNEL`: kernel start/end, context, stream, graphNodeId
- `CUPTI_ACTIVITY_KIND_RUNTIME`: launch/sync/memcpy API와 correlationId
- `CUPTI_ACTIVITY_KIND_MEMCPY`: kind, bytes, stream, graph node
- `NVTX_EVENTS`: slot/NRx/AI range
- context/stream/device metadata

핵심 derived metric은 다음과 같다.

```text
arrival_to_first_nrx_kernel
last_ai_kernel_end - nrx_arrival
cuda_api_submission_overhead
explicit_sync_time
kernel_gap_and_gpu_idle
NRx/AI overlap fraction
longest_nonpreemptible_ai_kernel
memcpy bytes per slot by direction
```

### 4.5 Nsight Compute pass

먼저 Nsys에서 누적 GPU time 상위 NRx kernel 5개와 Qwen decode/prefill kernel 5개를 뽑는다.
그 뒤 background concurrency가 없는 isolated replay에서만 NCU를 실행한다.

```bash
ncu \
  --target-processes all \
  --replay-mode kernel \
  --kernel-name-base demangled \
  --kernel-name 'regex:<escaped-kernel-regex>' \
  --launch-skip <warmup-matches> --launch-count 1 \
  --section LaunchStats \
  --section Occupancy \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section SchedulerStats \
  --force-overwrite \
  --export /results/ncu/<kernel> \
  <microbenchmark command>
```

각 kernel에 대해 다음을 분류한다.

- kernel duration과 launch geometry
- achieved occupancy 및 theoretical occupancy
- SM/Tensor utilization
- DRAM/L2 bandwidth와 hit rate
- eligible/active warps, issue stall
- register/shared-memory 제한
- `launch__uses_mps`, context/stream

MIG에서는 shared-resource profiling counter 일부가 지원되지 않을 수 있다. unavailable
metric은 0으로 해석하지 않고 `unsupported`로 기록한다.

### 4.6 CUPTI 후속 pass

장기 run에 Nsys를 붙이지 않는다. 후속 low-overhead collector는
`RUNTIME`, `DRIVER`, `CONCURRENT_KERNEL`, `MEMCPY/MEMCPY2`, `SYNCHRONIZATION`,
`OVERHEAD`, `MARKER` activity를 비동기로 수집한다. `KERNEL` activity는 concurrency를
serialize할 수 있으므로 사용하지 않는다. API correlation ID, graph ID/node ID,
context/stream ID를 slot NVTX external correlation과 연결한다.

## 5. Background AI workload

Background 결과를 `Qwen it/s` 한 숫자로 끝내지 않는다.

| Workload | Primary 목적 | Shape |
|---|---|---|
| Qwen decode | 지속적 latency-sensitive inference | batch 1, KV bucket 128/512/2048 |
| Qwen prefill | 긴 blocking kernel과 burst형 AI | prompt 128/512/2048 |
| Qwen full forward | 최악 간섭 stress control | 기존 seq 64/128/256/512 |
| BERT/Whisper | architecture generality | Qwen gate 통과 후 1개씩 |

Qwen primary utility는 decode token/s와 inter-token latency다. Prefill은 request/s,
time-to-first-token, chunk overhead를 기록한다. Full-forward iteration/s는 stress ablation으로만
사용한다.

Background sharing 비교군은 다음과 같다.

| ID | 정책 | 무엇을 검증하는가 |
|---|---|---|
| AI0 | NRx dedicated, AI off | RAN lower bound |
| AI1 | Qwen always-on MPS | uncontrolled interference |
| AI2 | MPS cap 10/20/30/50% | static spatial/time cap |
| AI3 | priority only | pending-work priority의 한계 |
| AI4 | token-boundary gate | 현재 cooperative baseline |
| AI5 | bounded layer/prefill quantum | quantum 크기 효과 |
| AI6 | MIGRx predictive lease | 제안 방식 |
| AIO | future-arrival oracle lease | upper bound |

## 6. 오늘 12시간 실행 순서

시간은 절대 시각보다 `T+` 기준으로 사용한다. 앞 gate가 실패하면 뒤 결과를 무리하게
생성하지 않는다.

CSV의 `estimated_wall_min`은 각 row를 완전히 독립 실행할 때의 보수적 시간이라 합계가
12시간을 넘는다. 실제 critical path는 아래 720분이다. 같은 model warm-up, Nsys capture,
paired trace를 공유해 여러 row를 한 block에서 얻는다. `P0`가 오늘의 필수 lane이며,
`P1`은 gate가 일찍 끝나면 별도 idle GPU에서 병행하거나 overnight로 넘긴다. `P2`와
`paper` row는 오늘 강제로 실행하지 않는다.

### T+0:00–0:40 · Freeze와 manifest

- active container/process가 없는지 확인
- GPU0 `4g+3g` UUID와 CUDA ordinal/SM count를 실제 container 안에서 기록
- NIC loopback, GID3, link, peermem, topology 기록
- image/source SHA와 clock policy 저장
- deterministic input seed 및 result root 생성
- 기존 결과 디렉터리는 절대 overwrite하지 않음

산출물: `manifest.json`, `topology.txt`, `versions.txt`, `source_sha256.txt`.

Abort:

- MIG UUID/profile 불일치
- NIC inactive 또는 GID 변경
- stale workload/container
- CUDA ordinal을 UUID로 검증하지 못함

### T+0:40–1:40 · Instrumentation smoke

- DirectNrx, Qwen gate, queue harness에 공통 NVTX range 삽입
- slot/lease event CSV writer 추가
- profiler capture window를 warm-up 밖에 배치
- 20-slot short run으로 range nesting과 event count 확인
- direct vs wrapper output, graph vs non-graph output allclose 재확인

Gate I:

- slot N개면 `slot_total`, `nrx_graph`, completion event도 정확히 N개
- profiler 밖 timing과 profiler 안 timing 파일이 분리
- correctness mismatch 0

### T+1:40–3:10 · E1: NRx API/kernel stack

Timing matrix:

- wrapper
- direct caller-owned, graph off
- direct caller-owned, graph on
- 각 3 trials, warm-up 200, measured 5,000 이상
- 3g/4g/full 중 오늘은 4g primary, full sanity

Profiling matrix:

- wrapper graph-level 50 iterations
- direct graph-level 200 iterations
- direct node-level 10 iterations

답해야 할 질문:

1. wrapper의 103 ms 중 conversion/copy/sync/API overhead 비율은 얼마인가?
2. graph는 host enqueue와 GPU kernel time 중 무엇을 줄이는가?
3. direct NRx의 top kernels는 compute-bound인가 memory-bound인가?

### T+3:10–4:20 · E2: NCU top-kernel roofline

- E1 Nsys에서 누적 GPU time 상위 NRx kernel 5개 선택
- Qwen decode와 prefill 각각 상위 kernel 3–5개 선택
- isolated NCU 1 launch, 필요한 kernel만 3 launches 반복
- first run은 `basic`, 결정적 kernel만 지정 section으로 확장

Gate II:

- replay가 output/cuda graph state를 깨면 해당 kernel은 standalone microbenchmark로 추출
- NCU가 MIG counter를 지원하지 않으면 full GPU에서 kernel taxonomy만 얻고 MIG timing과
  혼합하지 않음

### T+4:20–5:30 · E3: non-preemption 실험

같은 3g partition 안에 NRx와 Qwen을 MPS로 두고, Qwen kernel의 임의 phase에 NRx를
주입한다.

- Qwen: decode, prefill 128/512/2048
- NRx stream: default와 highest priority
- Qwen client priority: normal/low
- MPS cap: uncapped와 20%
- 각 cell 100 injections timing, 대표 20 injections Nsys

Primary metric:

```text
blocking_us = first_nrx_kernel_start - nrx_arrival
residual_ai_us = max(0, last_inflight_ai_kernel_end - nrx_arrival)
```

H3 판정:

- priority-only blocking이 `residual_ai_us`에 묶이면 bounded lease가 필요함
- priority만으로 p99가 target 아래가 되면 lease의 novelty를 재검토함

### T+5:30–7:10 · E4: background quantum sweep

구현/validation 순서:

1. token boundary baseline
2. prefill chunk 64/128/256/512
3. layer group 1/2/4/8/whole
4. fixed-shape graph bucket 가능 여부

각 quantum에서 측정:

- quantum GPU p50/p95/p99/max
- stop request → quiescent p99
- Qwen tokens/s 또는 prefill request/s
- NRx injected p99/max와 miss ratio
- graph launch/API overhead
- correctness: final logits allclose, next token exact match

오늘의 선택 규칙:

```text
q* = AI utility가 가장 크면서
     Q99(quantum + quiescence) <= lease budget의 가장 큰 quantum
```

### T+7:10–7:40 · E5a: conventional fallback calibration

- protected 4g에서 current MCS2 conventional receiver의 persistent path 측정
- service p50/p95/p99와 front/back 중복 범위를 분리
- synthetic 2/5/10 ms budget의 `commit_j` table 생성
- 이 단계는 service timing gate이며 ground-truth radio correctness claim이 아님

### T+7:40–9:10 · E5b: fixed-MIG policy sweep

고정 `4g primary NRx + 3g overflow NRx/Qwen` compute-only harness에서 먼저 controller를
검증한다. 이 결과를 full-L1 결과로 표기하지 않는다.

Trace:

- low 600/s 2 s → burst 1,100/s 2 s → low 600/s 2 s
- 추가 normalized boundary: aggregate capacity의 0.9/1.0/1.1x
- deadline sensitivity: 2/5/10 ms
- paired seeds 201/204/207

정책:

- static one + AI
- naive two + AI
- priority-only
- token gate
- shortest queue + token gate
- predicted finish + bounded lease (MIGRx)
- predicted finish + bounded lease + JIT conventional commit
- dedicated two
- oracle lease

필수 timeline:

- arrival rate, queue depth, endpoint assignment
- AI lease start/end, stop request, quiescent
- NRx p99 window, miss, fallback
- Qwen token progress
- NRx-before-commit 비율, conventional launch 비율, late-result drop, extra L1 GPU work

Gate III:

- MIGRx가 shortest-queue/token-gate보다 p99 또는 AI utility의 Pareto 축 하나도 개선하지
  못하면 controller를 central contribution으로 사용하지 않음
- queue가 안정되더라도 background utility가 0이면 dedicated baseline과 동일하므로 실패

### T+9:10–10:10 · E6: transport fair-depth sweep

동일 payload와 ring depth로만 비교한다.

- current MCS2/QPSK service: forward 1,415,232 B, backward 314,496 B
- payload-size sensitivity: backward 1,257,984 B는 max-output buffer control로 별도 측정
- depth 1/2/4/8
- P2P same-GPU in-process upper bound
- GDR cross-process service path
- CPU staging baseline
- 1,000 iterations, 100 warm-up, 3 trials

기록:

- one-way/round-trip p50/p95/p99
- achieved slot/s, CQ polling CPU time
- CPU memory bytes/copies
- NIC counters before/after
- full-chain latency는 transport-only와 별도 열

현재 direct engine의 `output_1`은 QPSK 2-bit LLR shape
`(2, 1, 3276, 12)`이므로 production-path backward payload는 314,496 B다. 과거
1,257,984 B 수치는 8-value output buffer를 가정한 generic transport gate다. 두 값을
같은 service configuration의 결과처럼 섞지 않는다.

GDR에서 payload bytes가 NIC counter에 기대값으로 나타나는지 확인한다. Nsys에는 NIC DMA
자체가 CUDA memcpy로 보이지 않을 수 있으므로 NVTX submission/completion과 verbs CQ를
ground truth로 쓴다.

### T+10:10–11:10 · E7: full L1 component profile와 integration smoke

- CE/front, transport, direct NRx, LDPC/CRC/back을 각각 NVTX로 표시
- local/sibling P2P/GDR에서 deterministic tensor checksum, finite output, stale marker 검증
- first goal은 100-slot correctness smoke
- 가능하면 ring depth 2/4로 periodic arrival 1,000 slots
- protected L1 active time과 end-to-end sojourn을 분리

이 단계에서 Python/Aerial wrapper의 common front/back capacity가 1,000 slot/s보다 낮으면
이를 숨기지 않는다. `routing failure`가 아니라 production-path optimization prerequisite로
분류한다.

이 smoke의 `correctness`는 tensor transport와 API contract에 한정한다. Random RX tensor에
대한 CRC API 반환을 decoded-TB correctness로 해석하지 않는다. Ground-truth TB를 생성하고
channel을 통과시킨 paired CRC/BLER evidence는 R1에서 별도로 만든다.

### T+11:10–12:00 · 분석과 verdict

- unprofiled timing만 primary CSV로 aggregate
- Nsys/NCU는 causal evidence figure로 분리
- 각 pathology row에 supporting artifact를 링크
- 실패/unsupported metric도 결과로 보존
- 다음 overnight run은 Gate I–III를 통과한 config만 5 trials/long run으로 예약

최종 verdict는 세 문장으로 답한다.

1. 실제 bottleneck은 무엇인가?
2. MIGRx의 어떤 mechanism이 그 bottleneck을 제거했는가?
3. 남은 bottleneck 때문에 central claim을 어디까지 제한해야 하는가?

## 7. Paper campaign 전체 매트릭스

오늘 이후 primary paper 실험은 다음 여섯 축으로 구성한다.

### X1. Isolation-elasticity gap

- full GPU MPS, same-partition MIG+MPS, protected cross-MIG
- NRx endpoint capacity 이하/근처/초과
- background offered load 0/30/50/100%
- 결과: protected L1 active p99, NRx queue cliff, AI utility

### X2. Non-preemption과 lease

- priority, MPS cap, token gate, bounded quantum, proposed predictive lease
- decode/prefill 및 BERT/Whisper diversity
- 결과: arrival-to-first-NRx-kernel, quiescence, deadline/AI Pareto

### X3. Endpoint scale-out

- resident endpoints 1/2/3
- sibling MIG, remote GPU/MIG
- periodic and synchronized multi-cell arrival
- 결과: aggregate capacity, backlog, drain time, imbalance

### X4. Transport

- P2P, GDR, CPU staging
- depth 1/2/4/8, topology locality
- 결과: transport p99, CPU traffic, full-chain contribution

### X5. Deadline/radio-aware routing

- round-robin, shortest queue, EDF feasible, queue threshold, MIGRx, oracle
- paired conventional/NRx channel traces
- 결과: deadline miss vs decoded TB goodput/BLER Pareto

### X6. End-to-end/long run

- actual slot schedule, real full L1, 30 min 및 가능하면 2 h
- power/clock/temperature/ECC/NIC counter
- 결과: p99/miss burst, L1 restart 0, topology change 0, AI utility

## 8. 예상 figure와 각각의 논리

| Figure | x/y | 입증하는 것 | 대응 mechanism |
|---|---|---|---|
| F1 Causal chain | isolation→queue→reconfig→lease | problem이 실제로 연속되는가 | 전체 design |
| F2 Kernel timeline | time, Qwen/NRx streams | priority가 긴 kernel을 못 자르는가 | bounded quantum |
| F3 Quantum Pareto | Qwen utility vs NRx p99/miss | 최적 quantum이 존재하는가 | lease controller |
| F4 Queue phase transition | offered load vs p99/backlog | fixed capacity cliff | endpoint scale-out |
| F5 Reclaim timeline | time vs queue/route/AI state | online controller가 burst 전에 반응하는가 | prediction/headroom |
| F6 Transport decomposition | path/depth vs stage latency | GDR/P2P의 정확한 역할 | GPU-resident service fabric |
| F7 Deadline-radio Pareto | miss vs TB goodput | queue-only보다 radio-aware가 좋은가 | admission/fallback |
| F8 Long-run | time series | 안정성, no drain/restart | fixed topology invariant |

F2가 architecture paper의 핵심 microarchitectural evidence다. F3와 F5가 proposed mechanism의
효과다. F7이 없으면 radio-aware contribution은 삭제한다.

## 9. Go/No-Go와 pivot

### G1. Long background kernel이 원인인가?

- Go: NRx blocking이 in-flight AI kernel residual과 강하게 일치
- Pivot: CPU/MPS submission 또는 sync가 지배하면 kernel chunking 대신 launch/control path를 개선

### G2. Quantum을 줄이면 p99가 줄어드는가?

- Go: AI utility 손실보다 NRx tail 개선이 큰 Pareto knee 존재
- Pivot: kernel 자체가 이미 짧다면 endpoint routing/headroom만 남기고 micrograph claim 삭제

### G3. Predictive lease가 단순 threshold보다 좋은가?

- Go: 같은 miss budget에서 AI utility 증가 또는 같은 utility에서 miss 감소
- Pivot: deterministic periodic arrival에서 threshold가 동등하면 simpler threshold system으로 축소

### G4. Scale-out이 common L1 병목 전에 효과가 있는가?

- Go: aggregate NRx capacity 증가와 queue 안정
- Pivot: common front/back이 먼저 포화하면 L1 pipeline/graph optimization이 central problem

### G5. Radio benefit heterogeneity가 있는가?

- Go: channel/MCS별 NRx incremental goodput 분산이 충분
- Pivot: utility predictor를 버리고 deadline-safe all-NRx/fallback만 유지

### G6. JIT commit이 deadline risk를 실제로 차단하는가?

- Go: injected NRx tail/endpoint failure에서 block-on-NRx 대비 miss를 줄이고, immediate
  speculation보다 conventional GPU work를 줄임
- Pivot: conventional service 자체가 budget에 들어오지 않으면 commit 설계를 버리고
  pre-admission-only fallback 또는 더 빠른 classical path가 필요

## 10. Claim discipline

다음 표현은 evidence가 생길 때까지 금지한다.

- `5–10 us GDR loopback`
- `zero overhead cross-MIG`
- `1 ms hard deadline satisfied`
- `P2P production path` — 현재 별도-process limitation을 해결하기 전
- `radio-aware` — paired CRC/BLER dataset 전
- `kernel preemption` — MIGRx는 cooperative bounded work이지 preemption이 아님
- `dynamic MIG allocation` — geometry는 고정

방어 가능한 현재 claim은 다음이다.

> MIGRx converts fixed isolated GPU partitions into an elastic NRx service pool without
> changing MIG geometry. It does so by routing deadline-feasible slots to resident endpoints
> and admitting only bounded background GPU work that fits measured slack.

## 11. 오늘 사용할 authoritative source

- NVIDIA MIG deployment guide: R570부터 same-GPU MIG P2P, CUDA IPC/GDR 제약
  <https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/deployment-considerations.html>
- CUDA Programming Guide: stream priority는 hint이며 실행 중 work를 보장되게 선점하지 않음
  <https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html>
- Nsight Systems User Guide: graph/node trace, capture range와 overhead
  <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>
- Nsight Compute Profiling Guide: LaunchStats, Occupancy, SOL, Memory, Scheduler sections
  <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>
- CUPTI Activity API: concurrent kernel/API/memcpy/sync correlation
  <https://docs.nvidia.com/cupti/13.3.0/api/group__CUPTI__ACTIVITY__API.html>
- NVIDIA MPS guide: active-thread partitioning과 client priority
  <https://docs.nvidia.com/deploy/mps/when-to-use-mps.html>
- Clockwork: predictable DNN execution과 request-level SLO scheduling
  <https://www.usenix.org/conference/osdi20/presentation/gujarati>
- REEF: concurrent DNN inference를 위한 microsecond-scale preemption
  <https://www.usenix.org/system/files/osdi22-han.pdf>
- Tally: block-level kernel slicing/preemption 기반 non-intrusive isolation
  <https://arxiv.org/abs/2410.07381>
- Flex-MIG: fixed MIG 사이 one-to-many 분산 실행과 host shared-memory collective
  <https://arxiv.org/abs/2511.09143>
