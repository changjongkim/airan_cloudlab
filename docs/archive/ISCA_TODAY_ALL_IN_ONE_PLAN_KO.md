# DART-Rx Day-1 · 설계부터 five-way/background 실험까지

작성일: 2026-08-13  
목표 시간: T+0부터 T+17시간  
대상: CloudLab d8545, A100 40GB ×4, Aerial 25.3.2

scheme specification: `DART_RX_SCHEME_DESIGN_V0_KO.md`

실행 결과: `results/isca_v2/day1_20260813T0523Z/REPORT_KO.md`  
raw/자동 요약: `results/isca_v2/day1_20260813T0523Z/{SUMMARY.json,SUMMARY.md}`

Day-1에서 preflight, scheme fault test, direct/wrapper/Nsight, fixed P2P/GDR,
replica queue, 4-workload reclaim, MPS와 proper two-client MIG+MPS diagnostic을
실행했다. 최초 background run의 device mapping 오류는 invalid marker로 제외하고
corrected suite를 별도 보존했다.

## 0. 오늘의 목표

오늘은 paper 전체 full sweep을 끝내는 날이 아니다. 다음 세 가지를 한 번에
완성하는 날이다.

1. `MIG/MPS/MIG+MPS/P2P/GDR`의 공정한 canonical comparison
2. realistic background workload가 만드는 architecture problem의 원인 증거
3. 그 문제와 직접 연결되는 DART-Rx software design의 첫 A/B 결과

하루가 끝났을 때 다음 문장을 데이터로 판단할 수 있어야 한다.

> 같은 GPU에 모두 넣으면 work conservation은 좋지만 L1 tail이 background
> kernel에 막힌다. MIG로 나누면 L1은 보호되지만 NRx endpoint의 queue와 idle
> capacity가 분리된다. P2P/GDR는 data movement를 해결하지만 deadline과
> result-validity를 해결하지 않는다. DART-Rx가 이 마지막 semantic gap을
> 해결하는가?

### 0.1 오늘의 co-design 방식

스킴을 마지막 90분에 붙이지 않는다. 다음 loop를 반복한다.

```text
measure one mechanism boundary
        -> derive one conservative parameter/profile
        -> implement one DART component
        -> run its causal ablation on the same trace
        -> keep or reject the mechanism
```

| 측정 | 바로 설계하는 component | 같은 날 검증 |
|---|---|---|
| NRx/transport p99 | DART-P/F profile과 feasibility | predicted finish vs SQ |
| endpoint capacity/queue | DART-R reservation/credit | concurrent burst over-admission |
| stale/delayed completion | DART-C epoch commit | 1,000 fault injections |
| conventional p99 | DART-J latest-safe-start | block vs dual vs JIT |
| background residual CDF | DART-L Qmax/unit profile | fixed gate vs grant lease |
| host/device control CDF | device DART와 DART-Q target | p99와 reserved-SM cost |

따라서 Stage 1부터 scheme source가 만들어지고 이후 측정 결과로 profile을 채운다.

co-design 규율:

- algorithm/state/invariant는 첫 measurement 전에 `scheme_version=v0`로 freeze
- measurement가 자동으로 채우는 값은 p99 profile, error guard, Qmax뿐
- 결과가 나쁘다고 v0 heuristic을 덮어쓰지 않음
- 구조 변경은 `v1`과 새 experiment ID로 실행하고 v0 raw result를 보존
- 동일 trace의 tuning split과 evaluation split을 분리

## 1. 오늘 만들 figure

### Figure A · Five-way isolation/communication frontier

x축: background application utility  
y축: L1/slot p99 또는 deadline miss  
marker: MPS, MIG, MIG+MPS, P2P, GDR  
색: background workload class

이 figure는 각 approach의 위치를 보여준다.

- MPS: 높은 utilization, 높은/불안정한 L1 tail 가능성
- MIG: 안정적 background isolation, L1+NRx local contention
- MIG+MPS: partition 내부 quota의 제한적 효과
- P2P: L1 isolation과 작은 transport cost
- GDR: P2P 불가/remote endpoint를 위한 transport cost

### Figure B · Background non-preemption cause

x축: injection 시점의 background kernel residual time  
y축: NRx arrival-to-first-kernel delay  
shape: Qwen prefill/decode, video, speech, training  
line: `y=x`

이 figure가 맞으면 background problem은 단순 평균 SM utilization이 아니라
**이미 dispatch된 non-preemptible work**다.

### Figure C · Isolation-elasticity contradiction

한 time-series에 다음을 겹친다.

- cell/NRx arrivals
- endpoint queue depth
- deadline miss
- 각 endpoint utilization
- idle eligible GPU capacity

miss와 idle capacity가 같은 시각에 존재해야 DART-Rx의 problem이 성립한다.

### Figure D · Transport는 얼마만큼 문제인가

stacked bar:

```text
pack | forward | visibility | queue | TensorRT | return | commit | LDPC/CRC
```

SHM, P2P, GDR를 동일 current payload와 topology에서 비교한다.

### Figure E · DART first result

static, shortest queue, predicted-finish host, device commit/JIT fallback의
deadline miss 대 background utility Pareto를 그린다.

## 2. 공정한 canonical placement

### C-MPS

- physical GPU3 full A100
- L1, NRx, background를 별도 process로 MPS 공유
- Qwen/background cap만 바꾸고 L1/NRx는 100% request 가능
- full GPU3의 L1-only baseline을 별도로 측정

### C-MIG

- GPU0 fixed 4g+3g
- 4g: L1→NRx를 한 process, 두 stream/CUDA Graph로 실행
- 3g: background workload 전용
- 이 구성은 background interference로부터 L1+NRx 전체를 격리한다.

### C-MIG+MPS

- GPU0 fixed 4g+3g
- 4g: L1 producer와 NRx service를 별도 process로 MPS 공유
- 3g: background workload 전용
- L1/NRx process boundary와 quota 효과를 C-MIG와 비교한다.

### C-P2P

- GPU0 fixed 4g+3g
- 4g: L1 전용
- 3g: NRx service와 background를 MPS 공유
- 4g↔3g payload는 CUDA P2P
- L1은 완전히 격리되지만 NRx service capacity는 background와 경쟁한다.

### C-GDR

- C-P2P와 동일 placement
- payload transport만 ConnectX-6 Dx loopback GPUDirect RDMA로 교체
- same-GPU GDR capability가 실패하면 GPU1 full endpoint를 사용하고 별도
  topology control로 표기한다.

### C-DART-P2P/GDR

- C-P2P/GDR의 data placement 유지
- GPU1/2의 resident NRx endpoint를 elastic pool로 추가
- predicted finish, epoch commit, JIT fallback을 활성화

## 3. fairness rule

### 3.1 동일한 것

- NRx ONNX/TRT engine과 precision
- caller-owned input/output binding
- current QPSK payload와 layout
- arrival timestamp trace와 random seed
- background request trace
- clock/power/thermal policy
- warm-up, measurement window, trial 수
- correctness checksum

### 3.2 current payload

- forward: 1,415,232 B
- backward: 314,496 B

generic 8-value output의 1,257,984 B와 섞지 않는다.

### 3.3 두 종류의 load

각 topology의 capacity가 다르므로 두 sweep을 모두 한다.

1. **absolute traffic:** 같은 cell/slot arrival trace를 모든 approach에 입력
2. **normalized stress:** 각 approach의 isolated NRx capacity 대비
   50/85/95/105% load

논문의 deployment comparison은 absolute traffic을 우선하고, queue mechanism
설명에는 normalized load를 쓴다.

### 3.4 두 종류의 latency

- absolute end-to-end latency
- 각 physical topology의 L1-only 대비 slowdown

GPU3 full과 GPU0 4g의 성능 차이를 transport 차이로 해석하지 않는다.

### 3.5 GPU budget

C-MPS와 GPU0 MIG approaches는 각각 physical A100 한 대의 7g-equivalent
budget을 쓴다. DART scale-out은 추가 endpoint GPU-seconds를 별도 비용으로
보고하고, dedicated peak provisioning baseline과 같은 총 budget에서 비교한다.

## 4. background workload qualification

오늘 기존 background script를 그대로 돌리지 않는다. 먼저 다음 gate를 통과해야
`realistic` label을 붙인다.

### BG-Q · Generative AI

- cached Qwen2.5-7B weights
- real prompt JSONL
- prefill과 decode event를 분리
- open-loop request arrival
- TTFT, TPOT, token/s
- random token은 kernel microbenchmark에만 허용

### BG-V · Video

- 실제 compressed clip decode
- pretrained detector weights
- decode, preprocess, H2D, TRT enqueue를 분리
- 1/4 streams, 15/30 FPS
- frame p99, drop, FPS

현재 `weights=None` ResNet/random image script는 사용할 수 없다.

### BG-S · Speech

- 실제 audio chunk가 wall-clock arrival
- streaming-capable path
- 5초 audio를 30초로 강제 padding한 batch Whisper 결과는 사용하지 않는다.
- chunk p99, backlog, real-time factor

### BG-T · Online training

- real dataset subset과 pretrained initialization
- forward/backward/optimizer NVTX 분리
- micro-batch 1/4
- samples/s와 iteration p99

### 오늘 실행 tier

- Tier 0 필수: Qwen decode/prefill + online training
- Tier 1: video가 model/data gate를 통과하면 포함
- Tier 1: speech가 true-streaming gate를 통과하면 포함
- 실패한 Tier 1을 synthetic 대체재로 조용히 바꾸지 않는다. `BLOCKED_DATA`
  상태로 남기고 five-way claim에서 제외한다.

## 5. 오늘 구현할 software

### 5.1 공통 event schema

각 request는 JSONL 또는 Parquet event를 남긴다.

```text
run_id, approach, workload, trial,
slot_id, epoch, tensor_class,
arrival_ns, l1_start_ns, l1_end_ns,
pack_start_ns, transport_publish_ns, remote_visible_ns,
queue_enter_ns, nrx_start_ns, nrx_end_ns,
return_visible_ns, fallback_start_ns, commit_ns,
committed_kind, deadline_ns, miss,
endpoint_id, endpoint_queue_depth,
background_state, background_request_id
```

GPU timestamps는 calibration record와 함께 저장한다.

### 5.2 `nrx_service_v2`

- replica별 독립 TRT context/stream/buffer/graph exec
- fixed-size tensor-slot ring
- open-loop request queue
- service start/end timestamp
- endpoint credit와 queue tail export

### 5.3 `background_server_v2`

workload 공통 RPC/control protocol:

```text
LOAD -> READY -> SET_RATE -> PAUSE_ENQUEUE -> RESUME -> STOP
```

model은 resident 상태를 유지한다. `PAUSE_ENQUEUE`는 이미 실행 중인 kernel을
중단한 척하지 않고 신규 work submit만 막는다.

### 5.4 `placement_runner_v2`

한 configuration을 실행하고 다음을 보장한다.

- UUID/profile 검증
- topology별 L1 baseline association
- exact arrival trace replay
- background offered rate replay
- timeout/fault cleanup
- checksum/epoch validation
- summary 생성 전 raw event count 확인

### 5.5 `dart_host_v0`

오늘 최소 구현:

- p99 profile table
- predicted-finish admission
- shortest-queue baseline
- epoch-safe result table
- JIT fallback simulator 또는 실제 conventional graph hook

### 5.6 `dart_device_probe`

오늘 full DART-Q를 구현하는 것이 아니다. 다음 hardware/software boundary를
측정한다.

- host descriptor→graph start
- GPU-visible ring→persistent kernel observe
- device commit CAS
- conditional/device graph launch capability
- persistent polling SM cost

### 5.7 DART scheme modules

오늘 구현 단위는 다음과 같다.

- `dart_profile.py`: component p99와 positive prediction error table
- `dart_router.py`: earliest predicted finish와 deadline feasibility
- `dart_reservation.py`: endpoint tail, input/output credit atomic reservation
- `dart_commit.cu`: private result slot, slot/epoch validation, winner CAS
- `dart_fallback.py`: conventional p99에서 latest-safe-start 계산 및 watchdog
- `dart_lease.py`: grant lookahead, Qmax, bounded graph-unit admission
- `transport_p2p.py` / `transport_gdr.py`: 같은 transaction API

scheme의 전체 algorithm/state/invariant는 `DART_RX_SCHEME_DESIGN_V0_KO.md`를
따른다.

## 6. one-shot orchestrator 설계

최종 entry point:

```bash
cd /mydata/aerial-cuda-accelerated-ran/pyaerial
nohup bash run_isca_day1_all.sh \
  --manifest /opt/nvidia/cuBB/pyaerial/isca_day1_matrix.csv \
  --results /mydata/results/isca_v2/day1_$(date -u +%Y%m%dT%H%M%SZ) \
  </dev/null > /mydata/results/isca_v2/day1_master.log 2>&1 &
```

### 6.1 상태 머신

```text
PENDING -> PREFLIGHT -> WARMING -> RUNNING -> VALIDATING -> COMPLETE
                                      |             |
                                      v             v
                                    FAILED       INVALID
```

각 job은 `status.json`을 atomic rename으로 갱신한다. `COMPLETE` marker가 있는
job은 resume 시 건너뛰고 source/model/trace hash가 다르면 재사용을 거부한다.

### 6.2 failure policy

- correctness, topology, thermal, container leak: 즉시 전체 중단
- Tier-1 model/data unavailable: 해당 workload만 `BLOCKED_DATA`, 독립 job 계속
- profiler failure: measurement run은 유지, profile dependent job만 block
- P2P/GDR capability failure: 실패 자체를 artifact로 저장하고 해당 dependent
  configuration만 block
- deadline result가 나쁘다는 이유로 중단하지 않는다.

### 6.3 topology safety

- GPU0의 4g+3g는 오늘 변경하지 않는다.
- full-MPS는 GPU3에서 실행한다.
- GPU1/2는 remote endpoint에 사용한다.
- dynamic-MIG downtime은 기존 3.64 s 결과를 재사용하되 manifest/hash를 확인한다.
- runner 종료 시 모든 MPS daemon/container를 scoped name으로 정리한다.
- NIC loopback은 시작/각 GDR phase 전에 검증한다.

### 6.4 동시 실행 규칙

실험 measurement는 서로 오염시키지 않도록 직렬 실행한다. 병렬 가능한 것은:

- CPU-side model/data download와 hash
- 이전 run 분석
- GPU를 사용하지 않는 trace generation

GPU experiment 두 개를 편의상 병렬 실행하지 않는다.

## 7. 오늘의 실행 단계

## Stage 0 · Freeze and preflight · T+0:00--0:30

수행:

- source/image/model/data/trace hash
- `nvidia-smi -L`, clocks, power, temperature
- NIC loopback, GID3, GDR registration
- running containers/processes
- disk space
- existing result provenance

pass:

- GPU0 4g+3g, GPU1--3 full
- no unknown GPU process
- NIC active 100G
- engine load 가능

## Stage 1 · DART transaction skeleton과 architecture boundary · T+0:30--2:00

구현/검증:

- event schema와 timestamp calibration
- 64-byte request descriptor와 transaction state machine
- endpoint/tensor-slot reservation unit test
- CUDA device/conditional graph capability probe
- GPU-visible descriptor doorbell
- commit CAS/epoch fault unit test
- C++ host launch와 persistent GPU polling microbench

산출물:

- `CAPABILITY.json`
- `CONTROL_PATH.csv`
- `COMMIT_CORRECTNESS.json`
- host-side DART transaction replay test

## Stage 2 · NRx direct path and service profile · T+2:00--3:30

- wrapper/raw/caller-owned input/caller-owned I/O/graph 비교
- Nsys 짧은 trace
- top kernel NCU
- replica 1/2/4/8 quick sweep
- 4g, 3g, full GPU 각각 capacity 측정

산출물:

- tensor-class p99 service table
- hidden copy/sync table
- endpoint capacity table

즉시 scheme 반영:

- 결과를 `(endpoint,tensor_class,background_mode)` DART-P table로 compile
- average/p95/p99 admission을 trace replay하고 optimistic error를 확인
- DART-F predicted-finish와 DART-R reservation unit test

## Stage 3 · Background qualification · T+3:30--5:30

Qwen과 training은 필수, video/speech는 qualification gate를 통과한 경우 실행한다.

각 workload에서:

- isolated 3g/full-GPU capacity calibration
- offered load 50/90% 생성
- Nsys kernel duration distribution
- longest kernel과 synchronization
- application metric

산출물:

- `BACKGROUND_MANIFEST.csv`
- `BACKGROUND_KERNEL_CDF.csv`
- workload별 application baseline

즉시 scheme 반영:

- workload별 bounded unit과 p99 duration을 DART-L table로 compile
- bounded unit을 만들 수 없는 workload는 strict lease incompatible로 표시
- measured longest/residual kernel로 initial Qmax와 drain guard 설정

## Stage 4 · Five-way compute-only · T+5:30--7:00

순서:

1. MPS full GPU
2. MIG local
3. MIG+MPS
4. P2P
5. GDR

각각:

- L1-only
- NRx-only capacity
- L1+NRx at 50/85/95/105% normalized load
- 3 trials quick discovery

background를 끄고 pure placement/transport 비용을 분리한다.

즉시 scheme 반영:

- P2P/GDR를 동일 DART transport interface에 연결
- per-transport p99를 DART-P에 주입
- simultaneous submit burst로 reservation 없는 path와 DART-R을 비교

## Stage 5 · Background causal experiment · T+7:00--9:00

full MPS에서 NRx를 background kernel의 임의 offset에 inject한다.

- Qwen decode 500 injections
- Qwen prefill 500
- training forward/backward 각 500
- qualified video/speech 각 500
- CUDA priority off/on

Nsys는 trial 하나의 짧은 window에만 사용하고 나머지는 lightweight events로
측정한다.

pass/fail:

- residual-kernel 대 dispatch-delay 관계가 보이면 lease hypothesis 유지
- 관계가 약하면 host/control/queue 원인으로 pivot

즉시 scheme 반영:

- residual CDF에서 Qmax candidate를 생성
- fixed gate, fixed quantum, grant-ahead lease의 trace-level replay
- Qwen decode step과 training micro-batch용 DART-L v0 구현

## Stage 6 · Five-way realistic-background comparison · T+9:00--12:00

오늘 discovery sweep:

- approaches 5개
- Tier-0 workload 2개
- background offered load 50/90%
- NRx absolute trace low/burst/high
- 3 trials, trial당 90초

Tier-1 workload는 각 approach 1 trial smoke 후, 통과하면 overnight full sweep
queue에 추가한다.

각 run은 PHY와 application metric을 동시에 저장한다.

baseline five-way와 함께 `P2P+fixed quantum` 및 `P2P+grant lease`를 실행하여
DART-L이 characterization에서 관찰된 tradeoff를 실제로 움직이는지 확인한다.

### 이 stage가 보여야 할 triangle

1. MPS: background utility는 좋지만 L1 tail risk
2. MIG/MIG+MPS: L1+NRx는 보호되지만 fixed service capacity
3. P2P/GDR: L1 보호는 유지되지만 NRx+background queue에서 contention

이 triangle이 없으면 DART-Rx motivation을 재검토한다.

## Stage 7 · Isolation-elasticity killer trace · T+12:00--13:30

고정 arrival trace:

```text
30% 60s -> 85% 60s -> 105% 10s burst -> 60% 60s
```

정책:

- static one endpoint
- dedicated peak endpoints
- round robin
- shortest queue
- predicted finish
- predicted finish + atomic reservation

GPU1/2의 endpoint를 resident로 유지한다. queue miss와 idle endpoint GPU-seconds가
동시에 존재하는지 확인한다.

## Stage 8 · DART v0 cumulative ablation · T+13:30--15:15

같은 killer trace에서:

- host predicted-finish admission
- admission + atomic reservation/credit
- host admission + epoch guard
- host admission + JIT fallback
- JIT fallback + fixed quantum
- JIT fallback + grant-ahead background lease
- GPU commit probe path
- P2P와 GDR

각 variant는 직전 variant에 mechanism 하나만 추가한다. 순서는
`static -> RR -> SQ -> predicted finish -> reservation -> epoch -> JIT -> fixed
quantum -> grant lease -> device path`다.

## Stage 9 · Analysis and decision · T+15:15--16:15

자동 생성:

- five-way summary
- residual-delay scatter
- isolation-elasticity timeline
- transport/compute/control stacked bar
- DART first Pareto
- failure/blocked table

마지막 `DAY1_VERDICT.md`에는 hypothesis마다 `PASS/FAIL/INCONCLUSIVE`와 다음
필요 evidence를 기록한다.

## Stage 10 · Validation reserve와 overnight freeze · T+16:15--17:00

- correctness/infra failure만 scoped rerun
- 성능이 나쁘다는 이유의 rerun 금지
- discovery 결과로 paper-quality rate/deadline/guard를 freeze
- five-way 5 trials×5분과 heterogeneous workload overnight matrix 생성

## 8. 오늘 discovery matrix의 크기

GPU measurement 예상:

- capability/control: 약 30분
- NRx stack/profile: 약 60분
- background calibration: 약 80분
- five-way compute: 약 80분
- background causal: 약 90분
- five-way Tier-0: 약 180분
- killer+DART: 약 140분

model load, cleanup, validation margin을 포함하면 약 12--14 GPU-hours다. CPU-side
profile compilation과 scheme validation은 warm-up/분석과 겹쳐 전체 wall-clock을
T+17시간으로 잡는다.

paper-quality full 5 trials/5분 sweep은 오늘 discovery 결과로 rate와 deadline을
freeze한 뒤 overnight에 별도 수행한다.

## 9. monitor와 alert

master log는 다음 event를 한 줄로 출력한다.

```text
[START] job_id config
[PROGRESS] job_id trial completed/total
[RESULT] job_id key metrics
[BLOCKED_DATA] job_id reason
[ALERT] job_id invariant
[DONE] job_id artifact
```

즉시 alert 조건:

- GPU temperature/clock deviation
- wrong checksum/epoch
- stale result commit
- missing arrivals 또는 coordinated omission
- background actual offered load ±5% 초과
- unexpected topology/UUID
- NIC link/GID change
- container exit/non-zero
- result row 수 부족

## 10. 오늘 결과 해석 표

| 관측 | 해석 | 다음 행동 |
|---|---|---|
| MPS만 L1 tail 증가 | shared execution/non-preemption 문제 | lease와 DART isolation 유지 |
| MIG local도 L1 tail 증가 | L1-NRx local contention | cross-partition 필요 |
| P2P/GDR L1 안정, NRx miss | isolation은 성공, service elasticity 문제 | DART endpoint admission |
| GDR가 P2P보다 크게 느림 | NIC/NUMA/flush/binding 확인 | transport 최적화, claim 축소 |
| P2P/GDR 모두 transport 작음 | 예상대로 compute/queue가 핵심 | DART-Q/NRx 최적화 집중 |
| idle+miss 동시 없음 | fragmentation problem 약함 | ISCA framing 중단/수정 |
| residual과 delay 상관 없음 | non-preemption hypothesis 약함 | host/queue/launch gap 분석 |
| host와 device control 동일 | DART-Q hardware 이점 약함 | systems-only 방향 검토 |
| JIT fallback이 miss 차단 | commit semantic 유효 | radio utility 단계 진행 |

## 11. 오늘 끝날 때 go/no-go

### GO-A · comparison

five-way 모두에서 output correctness와 application/PHY metric이 생성됐다.

### GO-B · background problem

최소 두 workload에서 같은 root cause 또는 같은 placement tradeoff가 재현됐다.

### GO-C · fragmentation

같은 trace에서 deadline pressure와 idle eligible capacity가 동시에 관측됐다.

### GO-D · DART

predicted-finish/epoch/JIT 중 적어도 하나가 strongest simple baseline의 Pareto를
개선했다.

### GO-E · architecture

device control path가 host 대비 tail 또는 reserved resource에서 hardware assist를
검토할 만한 gap을 보였다.

GO-C나 GO-E가 실패하면 ISCA title/claim을 그대로 밀지 않는다.

## 12. 오늘 수정·생성할 파일

```text
cloudlab_aerial/task1/isca_v2/
  cuda_graph_capability.cu
  control_path_bench.cu
  dart_descriptor.h
  dart_commit.cu
  nrx_service_v2.py
  placement_runner_v2.py
  background_server_v2.py
  dart_host_v0.py
  run_isca_day1_all.sh
  isca_day1_matrix.csv

cloudlab_results/results/isca_v2/day1_<UTC>/
  MANIFEST.json
  STATUS.json
  raw/
  profiles/
  summary/
  figures/
  DAY1_VERDICT.md
```

## 13. 기존 script 재사용 범위

재사용 가능:

- `nrx_trt_direct.py`
- `p2p_overlap_bench.py`
- `p2p_copy_gate.py`
- `gdr_rdma_channel.py`
- fixed-MIG/open-loop endpoint sweep의 trace generator
- existing Nsys SQLite analyzer

수정 없이 paper baseline으로 사용 불가:

- random-token closed-loop Qwen stress
- `weights=None` ResNet/random frame
- 5초 audio를 30초로 pad한 Whisper batch API
- whole Qwen sequence length를 micro-quantum이라 부르는 기존 sweep

## 14. 논문에서 five-way 비교를 쓰는 방식

Section 순서는 다음이 자연스럽다.

1. **Characterization:** MPS/MIG/MIG+MPS/P2P/GDR
2. **Problem:** isolation과 work conservation을 동시에 얻지 못함
3. **Root cause:** non-preemption, fixed endpoint queue, deadline-unaware command
4. **Design:** DART descriptor, admission, commit, fallback, lease
5. **Architecture:** DART-Q
6. **Evaluation:** five-way+DART와 heterogeneous workloads

따라서 비교 결과는 빠지는 것이 아니라 논문 problem을 세우는 첫 번째 핵심
section이 된다.
