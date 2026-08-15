# DART-Rx 실제 다중 엔드포인트 통합 계획

상태: 348-run pool campaign 완료 · Gate A/B 통과 · 실제 multi-cell open-loop 통합 대기  
작성일: 2026-08-14  
대상: Aerial cuPHY L1 + 선택적 Neural Receiver + 고정 MIG 자원 풀

## 1. 이번 통합의 정확한 목표

현재 GDR pool 캠페인은 **여러 격리 NRx 엔드포인트를 실제 GPUDirect RDMA로 동시에 구동할 수 있는지**와 정책별 queue/deadline 거동을 측정한다. 그러나 payload는 deterministic synthetic tensor이고 conventional receiver를 실행하지 않는다.

다음 통합은 측정기를 실제 스킴으로 바꾼다.

> 보호된 L1 MIG에서 실제 cuPHY channel estimation과 conventional receiver를 계속 실행하면서, radio utility가 있는 요청만 여러 상주 NRx 엔드포인트 중 deadline 안에 끝날 경로로 보내고, 유효한 NRx 결과 또는 conventional 결과를 정확히 한 번 commit한다.

이 단계에서 증명할 것은 단순 RDMA latency가 아니다.

1. MIG를 재구성하거나 L1을 재시작하지 않는다.
2. L1과 NRx를 같은 partition에 두지 않아 L1 interference를 막는다.
3. 다른 partition/GPU의 유휴 NRx capacity를 실제 radio request가 빌린다.
4. 늦거나 stale인 AI 결과는 PHY 상태를 오염시키지 않는다.
5. AI를 거절해도 conventional receiver로 정상 복구한다.

## 2. 현재 구현과 실제 스킴 사이의 공백

| 항목 | 현재 구현 | 통합 후 |
|---|---|---|
| `dart_rx_gdr_pool.py` | synthetic GPU payload, 실제 TRT NRx, 다중 endpoint/process | 실제 cuPHY CE tensor, 실제 NRx/LDPC/CRC |
| `dart_rx_radio_l1.py` | 실제 radio chain, 다중 GDR endpoint 통합 완료 | open-loop multi-cell arrival와 heterogeneous path selection |
| conventional path | radio vertical slice에만 존재, eager 실행 | 모든 요청의 보호된 recovery path |
| scheduling | synthetic trace의 finish-time 예측 | radio utility + endpoint/path tail + expiry |
| commit | 단일 endpoint에서 검증 | 모든 endpoint에 대해 epoch/expiry/single-commit |
| GPU data ownership | endpoint process가 synthetic buffer를 직접 채움 | L1이 CUDA IPC로 endpoint buffer를 직접 pack/read |

따라서 pool 결과만으로 논문의 전체 스킴이 구현되었다고 주장하지 않는다. Pool은 problem/capacity/path gate이고, 이 문서의 통합이 end-to-end design gate다.

## 3. Overall architecture

```text
Per-cell slot arrival
        |
        v
+---------------- Protected L1 MIG ----------------+
| cuPHY CE + conventional recovery                  |
|                                                   |
|  utility gate -> deadline/risk admission          |
|                       |                           |
|                endpoint/path selector             |
|                       |                           |
|  CUDA-IPC mapped registered GPU rings             |
|                       |                           |
|  expiry + epoch + CRC checked single commit       |
+-----------------------|---------------------------+
                        |
              control only: local IPC queue
              payload: GPU memory only
                        |
        +---------------+----------------+
        |               |                |
  endpoint agent 0 endpoint agent 1 endpoint agent 2
  QP/MR progress    QP/MR progress    QP/MR progress
        |               |                |
      P2P/GDR          GDR              GDR
        |               |                |
  resident NRx 0   resident NRx 1   resident NRx 2
  isolated GPU/MIG isolated GPU/MIG isolated GPU/MIG
```

### 3.1 고정 isolation plane

- L1은 고정 MIG partition에 상주한다.
- NRx 엔진은 다른 허용된 MIG/GPU partition에 미리 로드하고 warm-up한다.
- 실험 중 MIG geometry를 변경하지 않는다.
- 같은 physical GPU의 서로 다른 MIG instance는 CUDA P2P가 차단되므로 GDR가 일반 경로다.
- 하드웨어가 P2P를 허용하는 별도 GPU 경로는 fast tier로 사용한다.
- MPS는 NRx pool 내부의 best-effort/background sharing에만 선택적으로 사용하며 L1과 NRx를 같은 scheduling domain에 다시 섞지 않는다.

### 3.2 resident receiver fabric

각 NRx endpoint는 다음을 상주시킨다.

- TensorRT engine과 persistent input/output buffer
- forward/bacward RC QP와 GPU MR
- endpoint health epoch
- service-time telemetry와 queue credit
- transport path class (`P2P`, `GDR-local-host`, `GDR-remote-host`)

P2P/GDR는 novelty 자체가 아니라, 고정 partition 위에서 request 단위 elasticity를 제공하는 data plane이다.

### 3.3 DART control/commit plane

요청은 `(slot, cell, epoch, release, expiry, radio utility, tensor class)`를 가진다. Controller는 다음 순서로 동작한다.

1. NRx의 예상 radio gain이 없으면 conventional-only로 보낸다.
2. conventional recovery가 deadline 전에 가능한지 보장한다.
3. endpoint별 `transport + queue + NRx + return + commit` 완료 분포를 예측한다.
4. risk budget 안에서 가장 높은 expected radio utility를 주는 endpoint를 선택한다.
5. 도착 결과의 slot/epoch/health epoch/expiry/payload visibility/CRC를 검증한다.
6. NRx 또는 conventional 결과 하나만 commit한다.

## 4. GPU memory와 프로세스 소유권

### 4.1 왜 endpoint별 progress process가 필요한가

`pyverbs` completion polling을 Python thread로 병렬화했을 때 GIL 때문에 여러 endpoint가 사실상 직렬화되었다. 현재 pool은 endpoint별 `multiprocessing`으로 이를 제거했고, 3 endpoint RR에서 실제 병렬 capacity를 확인했다.

실제 통합도 endpoint별 process를 유지한다. 단, CPU queue에는 descriptor와 completion만 보내고 payload는 보내지 않는다.

### 4.2 선택한 zero-CPU-payload 구조

각 source-side endpoint agent가 다음 자원을 생성하고 소유한다.

- forward source GPU buffer/MR
- backward destination GPU buffer/MR
- QP/CQ/PD
- inter-process CUDA event

agent는 두 GPU allocation의 CUDA IPC memory handle을 L1에 넘긴다. L1은 handle을 열어 CuPy view를 만들고:

- CE 결과를 forward buffer에 GPU-to-GPU pack한다.
- 완료 event 이후 backward LLR buffer를 GPU에서 직접 읽는다.

agent는 control queue에서 `(sequence, descriptor, buffer_slot)`만 받고 RDMA를 post한다. NIC payload가 CPU DRAM을 통과하지 않는다.

### 4.3 ordering contract

Forward:

1. L1 stream에서 payload pack
2. inter-process CUDA ready event record
3. agent가 event wait
4. RDMA payload WRITE completion
5. descriptor WRITE
6. sequence/valid marker WRITE

Backward:

1. NRx output publish
2. payload WRITE completion
3. completion descriptor WRITE
4. sequence marker WRITE
5. source agent가 marker 관측
6. `cuFlushGPUDirectRDMAWrites`
7. IPC completion event record
8. L1 stream이 event wait 후 LDPC/CRC

같은 RC QP에서 payload보다 marker가 먼저 보이면 안 된다. GPU MR에는 `MR.read()`/`MR.write()`를 호출하지 않는다.

### 4.4 buffer depth

- 최초 correctness 구현은 endpoint당 one-in-flight로 제한한다.
- queue에 request metadata만 쌓고 GPU payload를 덮어쓰지 않는다.
- pool 실측이 one-in-flight로 deadline을 만족하지 못하는 경우에만 registered ring을 2/4/8로 확장한다.
- ring slot은 completion/expiry 이후에도 agent ACK 전에는 재사용하지 않는다.

이 순서는 불필요한 multi-buffer 복잡성을 피하면서 실제 deadline capacity 요구로 ring depth를 정한다.

## 5. Policy V2

현재 `predicted`는 late result를 거의 제거하지만 많은 요청을 conventional fallback으로 보낸다. 현재 결과의 `no timely NRx`는 PHY deadline miss와 동일하지 않으며, reject는 의도된 fallback이다.

V2는 고정 tail multiplier 대신 관측 기반 risk budget을 사용한다.

### 5.1 endpoint prediction

각 endpoint에 대해 다음을 별도로 추적한다.

- forward transport distribution
- NRx enqueue/service distribution
- backward transport distribution
- outstanding request와 predicted release time
- recent late/expired ratio
- health/circuit-break epoch

예상 완료시간은 단일 평균이 아니라 upper quantile과 uncertainty로 계산한다.

```text
finish_e = max(now, available_e)
         + Q_path_e(risk_budget)
         + Q_service_e(risk_budget)
         + commit_guard
```

### 5.2 admission objective

```text
maximize  expected_radio_gain(request)
          - lambda_late * P(finish > expiry)
          - lambda_load * fragmentation_cost(endpoint)
```

제약:

- protected L1 conventional recovery deadline
- target remote late/expired ratio
- endpoint credit/ring availability
- endpoint health epoch

정책 sweep은 `risk_budget`, `utility threshold`, `commit guard`를 독립 변수로 한다. 목표는 무조건 많은 NRx를 보내는 것이 아니라, 허용 late rate 안에서 추가 correct-TB gain을 최대화하는 것이다.

## 6. 구현 순서와 각 gate

### Gate A — CUDA IPC + GDR buffer ownership microtest

- agent가 GPU allocation/MR를 만들고 IPC handle export
- L1 process가 mapping 후 pattern을 GPU에서 기록
- agent가 pattern을 GDR WRITE
- peer가 checksum 확인
- backward도 대칭 검증
- 10,000회에 stale marker, checksum error, CUDA error 0

실패하면 실제 radio 통합을 시작하지 않고 C++/Cython progress engine 또는 동일-process async verbs로 경계를 재설계한다.

실측 결과:

| payload | iterations | integrity errors | steady mean | steady p99 |
|---:|---:|---:|---:|---:|
| 64 KiB | 100 | 0 | 135.62 us | 161.79 us |
| 1,415,232 B | 10,000 | 0 | 1,405.97 us | 1,432.08 us |

두 번째 수치는 forward request와 backward echo를 모두 포함한다. 기존 독립 측정의 forward
약 758 us + backward 약 665 us와 일치하므로 CUDA IPC process boundary가 유의미한 추가
millisecond overhead를 만들지 않는다.

### Gate B — 실제 single-request multi-endpoint correctness

- 기존 `dart_rx_radio_l1.py` trace를 고정 replay
- endpoint 1/2/3 각각 같은 request를 단독 실행
- conventional/NRx CRC와 TB correctness 비교
- endpoint별 100회, wrong commit/stale result 0

완료 결과:

| endpoints | mode | trials | median NRx requests | median correct TB | decision p50 | decision p99 | miss/late |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | all | 1 | 100 | 0.800 | 2.264 ms | 4.795 ms | 0/0 |
| 2 | all | 1 | 100 | 0.800 | 2.745 ms | 5.375 ms | 0/0 |
| 2 | utility | 1 | 75 | 0.800 | 2.634 ms | 5.094 ms | 0/0 |
| 3 | all | 3 | 100 | 0.800 | 2.567 ms | 5.139 ms | 0/0 |
| 3 | utility | 3 | 75 | 0.800 | 2.636 ms | 5.050 ms | 0/0 |
| 3 | conventional-only | 3 | 0 | 0.620 | 1.045 ms | 1.292 ms | 0/0 |

세 paired trace에서 conventional-only는 62/67/61 correct TB, 3-endpoint utility는
80/79/80이었다. Utility mode는 all-NRx와 같은 median correctness를 NRx 요청 25% 감소로
얻었고, 각 trial에서 endpoint를 정확히 25/25/25회 사용했다. 이것은 synchronous
correctness/radio-utility gate이며 replica throughput claim은 Gate C에서만 한다.

비-profiling 3-trial actual path median 분해:

| mode | CE+pack→dispatch p50 | remote exchange p50/p99 | NRx service p50 | transport/control p50 | conventional 이후 residual p50/p99 |
|---|---:|---:|---:|---:|---:|
| all | 1.331 ms | 2.004/2.777 ms | 1.111 ms | 0.895 ms | 0.861/1.365 ms |
| utility | 1.372 ms | 2.013/2.967 ms | 1.111 ms | 0.902 ms | 0.838/1.574 ms |

따라서 GDR control만 제거해서 얻을 수 있는 상한은 작다. Conversion/pack, remote NRx,
conventional과의 overlap이 모두 같은 millisecond scale이다.

### Gate C — 실제 multi-cell scheduled radio pipeline

현재 synchronous Python/Aerial vertical slice의 CE+pack→dispatch p50가 1.33--1.37 ms라서
한 process로 1 request/ms open-loop arrival을 생성하면 NRx 이전에 source가 병목이 된다.
Gate C는 이를 숨기지 않고 다음 중 하나로 구현해야 한다.

- 미리 capture한 actual CE tensors를 absolute-time replay하여 receiver pool만 측정
- cell별 L1 source process/CUDA stream을 분리
- Aerial의 multi-cell batch interface로 front-end를 묶음

첫 번째는 scheduler/capacity gate, 두 번째/세 번째는 end-to-end architecture gate다.

- single-cell periodic
- 2/4/8-cell phase-offset aggregate
- selective NRx 10/25/50/75/100%
- ON/OFF burst와 measured trace replay
- policy: static-one, static-cell, RR, shortest, predicted V1, risk-budget V2

필수 결과:

- correct TB/s와 additional correct TB due to NRx
- p50/p95/p99 commit latency
- conventional fallback ratio
- remote late/expired/stale ratio
- endpoint utilization 및 eligible-idle overlap
- L1 kernel/slot latency와 deadline miss

### Gate D — heterogeneous path ablation

동일 trace와 동일 NRx 엔진으로 다음을 비교한다.

- same-partition L1+NRx
- MPS co-location
- fixed MIG + host staging
- permitted CUDA P2P
- CPU-buffer RDMA
- GPUDirect RDMA
- DART-Rx mixed P2P/GDR selection

비교의 핵심은 microsecond transfer만이 아니라 isolation, usable NRx capacity, correct-TB utility와 deadline이다.

### Gate E — architecture-level profiling

- Nsight Systems: CUDA API, memcpy, TRT enqueue, stream sync, RDMA progress annotation
- raw TensorRT enqueue vs public wrapper
- caller-owned binding/persistent buffers
- CUDA Graph on/off
- NRx replica 수 1/2/3과 throughput/queue stability
- transport, queue, TRT, LDPC/CRC, commit 단계별 critical path

### Gate F — fault/recovery/background workload

- endpoint stall, process kill, delayed completion, stale epoch, corrupted descriptor
- Qwen만이 아니라 ResNet/BERT/Whisper/LLM 혼합 background
- NRx pool lease on/off와 burst arrival
- L1 isolation, recovery correctness, background work 보존율 측정

## 7. 현재 캠페인 종료 전 할 일

- [x] 분석기가 `04_full`과 nested replica results를 읽도록 수정
- [x] NaN-safe median 및 reject/late/expired 분리
- [x] 정책 gap CSV와 dominance/load-band 표 추가
- [x] 원격 자동 분석 전에 최신 분석기 배포
- [x] CUDA IPC GPU-MR 양방향 microtest 및 재현 runner 구현·원격 배포·10,000회 통과
- [x] `gdr_endpoint_agent.py`로 transport/control 및 CUDA IPC mapping 분리
- [x] Policy V2 risk-budget pure-Python 인터페이스와 deterministic unit test 6개
- [x] actual-radio 1/3-endpoint runner와 3-trial paired correctness matrix
- [x] warm-up 제외 CUDA Profiler API + NVTX Nsight capture
- [ ] fault injection runner 작성

## 8. 캠페인 종료 후 판정

캠페인은 348/348 full run, 전체 412 validated run으로 완료됐다. `predicted_finish`는
`static_one`보다 87/87, `static_cell`보다 86/87 trace에서 우수했다. 반면 전체 median
reject ratio가 0.8131이므로 현재 V1은 안전하지만 과보수적이다.

1. 모든 trace/policy pair 검증: 완료.
2. offered load별 `reject`, `late`, `expired` 분리: 완료. `correct fallback`은 actual radio
   통합에서 추가한다.
3. V1이 지나치게 보수적인 영역을 찾음: 완료. V2 target risk sweep은 미실행.
4. endpoint 1/2/3 measured capacity 확보: 완료.
5. Gate A는 통과했다. 다음 실행은 actual radio 다중 endpoint 통합이다.

현재 결과가 보여주는 잠정 방향은 긍정적이다. 고정 endpoint는 다른 NRx가 놀아도 deadline을 놓치며, 다중 pool은 같은 고정 MIG topology에서 usable capacity를 늘린다. 다만 현재 policy의 reject를 곧바로 failure로 해석해서는 안 되고, 최종 novelty는 Gate C에서 **격리 유지 + 추가 radio utility + safe fallback + drain-free elasticity**를 함께 보일 때 성립한다.
