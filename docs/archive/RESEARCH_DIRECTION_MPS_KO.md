> **보관 사유 (2026-09-25):** 이 문서는 실험 이력으로 보존한다. 최신 스킴과 claim boundary는 `docs/current`의 원고·모델·production gate 문서가 권위 기준이며, 과거 GPU0 전용, MIG, CloudLab 또는 4-GPU pool 가정은 현재 논문의 근거가 아니다.

# 연구 방향 정리: MIG는 제약, MPS는 최적화 대상

**Updated:** 2026-09-19 KST
**Status:** 방향 확정. Perlmutter 관련 항목은 아직 확인 전.
**근거 데이터:** `task1_final/gdr_pool_20260814T014651Z/`, `results/isca_v2/mig_causal_20260813T1138Z/`,
발표 슬라이드 Page 1–7 (NVIDIA 미팅용)

## 1. 결정

- **MIG는 제약 조건으로 다룬다.** L1을 격리하는 데는 효과가 있지만, 파티션이 고정되어 있어
  NRx capacity가 쪼개지고, 실행 중에 재구성할 수 없고, 다른 GPU의 MIG 파티션과는 P2P가 되지 않는다.
  지금까지의 MIG 실험은 "MIG가 어디까지 되고 어디서 막히는가"를 보여주는 근거로 쓴다.
- **최적화는 MPS 위에서 한다.** MPS는 파티션이 고정되지 않고, context 사이 CUDA P2P가 되며,
  concurrency를 동적으로 조절할 수 있다. 현재 MPS가 L1을 보호하지 못하는 것은 SW 스케줄링으로
  개선할 여지로 본다.
- **순서:** 단일 GPU에서 MPS 스케줄링(Direction 1) → Perlmutter에서 MPS + RDMA 분산(Direction 2).

## 2. 이 결정의 근거 (실측)

| 관찰 | 수치 | 의미 |
|---|---|---|
| Same-partition(Full MPS / MIG local / MIG+MPS) L1 active-time | 1.601× / 1.621× / 1.702× | NRx와 같은 방에 두면 L1이 느려진다 |
| Cross-partition(P2P / NIC GDR) L1 active-time | 1.043× / 1.103× | 파티션을 나누면 L1이 baseline 근처로 돌아온다 |
| NRx 8개 동시 실행 시 L1 p99 | Full MPS 4.5×, 4g MIG 안 MPS 10.7× | MIG 안에서 MPS를 써도 보호되지 않는다 |
| L1 kernel 사이 대기 시간 (median) | 1.1 → 119.7 → 379.1 μs (NRx 4 → 6 → 8) | 공유 queue에 NRx 작업이 쌓인다 |
| L1 GPU duty / kernel duration | 31.6% → 13.8% / 6 → 15 μs | 사용률이 낮은데도 contention에 민감하다 |
| Cross-partition end-to-end request time | P2P 5.83 ms, GDR 6.33 ms | 두 전송 경로의 차이는 작다 |
| Background AI(Qwen) throughput | Full MPS 11.14 it/s, 나머지 약 10.2 it/s | MPS가 background 처리량에서는 유리하다 |

정리하면, MIG는 격리는 되지만 유연하지 않고, MPS는 유연하고 처리량이 좋지만 지금 상태로는
L1을 보호하지 못한다. 연구 기회는 MPS 쪽에 있다.

## 3. Future Direction (슬라이드 문구)

```
1) Improving L1 Kernel Protection via Full-GPU MPS Scheduling Orchestration
   - Full-GPU MPS is inherently flexible — CUDA P2P works across contexts,
     concurrency is dynamic, and there is no rigid partitioning — leaving
     significant room for software-level scheduling optimization that MIG
     cannot offer.
   - Our future work builds on this flexibility to design admission control
     and scheduling orchestration that enhance L1 latency protection while
     preserving MPS's throughput and utilization benefits.

2) Scaling AI-RAN via MPS + RDMA on Large-scale HPC Systems
   - Multi-cell 5G deployments exceed single-node capacity, requiring
     distributed GPUs to absorb burst NRx demand.
   - Combining MPS's flexible per-node multiplexing with RDMA's low-latency
     cross-node transport extends Direction 1 to distributed HPC systems
     (e.g., Perlmutter), forming a scalable AI-RAN framework.
```

Direction 1을 설명할 때 "MPS가 이미 L1을 보호한다"고 말하면 Page 5–6 결과와 충돌한다.
"MPS는 유연하므로 SW 스케줄링으로 개선할 여지가 있다"는 방식으로 설명한다.

## 4. Perlmutter로 옮길 때 확인할 것

아래는 확인 전 항목이다. NERSC 문서와 짧은 테스트 job으로 확인한다.

| 항목 | 예상 | 영향 |
|---|---|---|
| 네트워크 | HPE Slingshot-11 (InfiniBand 아님) | pyverbs/ConnectX RC QP 코드가 그대로 돌지 않는다. libfabric(CXI) 또는 GPU-aware MPI로 전송 계층을 다시 짜야 할 수 있다 |
| MIG 설정 권한 | 사용자에게 없을 가능성이 큼 (root 필요) | MIG와 MPS 비교는 CloudLab에서, Perlmutter는 MPS만 쓴다 |
| Aerial 전체 스택 | 드라이버/DPDK/NIC 요구사항 때문에 어려울 수 있음 | cuPHY 커널만 떼어 쓰거나 L1 부하 trace를 재생한다 |
| MPS daemon | job 안에서 띄울 수 있는지 확인 필요 | `MPS_ACTIVE_THREAD_PERCENTAGE` 등을 조절할 수 있는지도 같이 본다 |
| 노드 안 GPU 연결 | A100 4장이 NVLink로 연결 | MIG를 쓰지 않으면 GPU 사이 P2P가 되므로 Direction 1 근거를 노드 하나에서 확인할 수 있다 |

첫 테스트 job에서 확인할 세 가지: MPS daemon 실행, 노드 안 GPU 사이 P2P, Slingshot에서
GPU 메모리로 직접 RDMA(GDR).

## 5. NVIDIA 미팅 질문 (최종 문구)

**Page 3 (placement):**
> We confirmed that P2P works between MIG partitions on the same GPU. However, we cannot access
> a MIG partition on another GPU. Is this a hardware-level limitation?

**Page 5 (root cause):**
> Is L1 slowdown caused mainly by MPS work-queue contention, SM re-allocation, or memory
> contention? Can MPS prioritize latency-critical L1 kernels beyond MPS_ACTIVE_THREAD_PERCENTAGE?

**Additional Question:**
1. **CUDA Green Context as a Flexible MIG Alternative** — Does Green Context provide per-context
   hardware queues and priority isolation, making it a flexible alternative to MIG?
2. **Fault Isolation Between MIG Partitions** — Can a crash or local hardware fault in one MIG
   partition affect the latency or execution of a sibling L1 partition on the same GPU?

Green Context는 Future Direction에 넣지 않고 질문으로만 둔다. 방향은 우리가 실측한 결과를
바탕으로 하고, Green Context는 NVIDIA 답에 따라 나중에 검토한다.

## 6. 발표 슬라이드 순서

1. Overview of AI-RAN (1): L1이 GPU로 이동, underutilization과 accuracy ceiling
2. Overview of AI-RAN (2): NRx 소개, cuPHY와 같은 GPU에 있어야 하는 이유, background AI와의 패턴 차이
3. Solutions using five different placements: same-partition 3종, cross-partition 2종, P2P 질문
4. Same-partition Placement Fails to Protect L1
5. Root Cause: NRx Concurrency Builds GPU Queue Backlog, MPS 질문
6. Our Future Direction
7. Additional Question

슬라이드 1–2에는 TTI, cuPHY, waterfall SNR, LDPC, slot 같은 용어가 설명 없이 나온다. AI-RAN을
모르는 청중을 위해 첫 등장 시 괄호로 짧게 정의하거나, 발표 초반 30초에 말로 설명한다.

## 7. 개념 정리 (이번 논의에서 확정한 것)

- **Slot과 request:** slot은 시간 단위(0.5 ms)이고 곧 deadline이다. request는 (cell, slot) 쌍이며
  실제로 처리되는 단위다. multi-cell 동기화는 같은 slot에 여러 cell의 request가 몰리는 상황이다.
- **cuPHY와 NRx:** cuPHY는 모든 request에 대해 항상 실행된다(FB). NRx는 admission을 통과한
  request에만 실행된다(RES). 둘은 channel estimation까지 공유하고 그 뒤에서 나뉜다.
- **Commit 규칙:** deadline 전에 먼저 도착한 CRC-pass 결과를 쓴다. FB가 먼저 왔지만 CRC fail이면
  RES를 기다린다.
- **Admission 조건:** utility(이 request가 NRx로 이득을 보는가)와 credit(NRx pool에 여유가 있는가)이
  둘 다 만족해야 NRx를 부른다. credit은 정확한 counter이고, utility는 추정값이다.
- **Utility gate의 현재 한계:** 지금 실험은 Sionna로 설정한 Es/N0 값을 알고 있는 oracle 방식이다.
  실제 시스템에서는 DMRS channel estimate, CQI, HARQ 이력으로 추정해야 한다(future work).
- **Es/N0 sweep(−4.0 ~ −3.2 dB, 0.2 dB 간격):** waterfall 구간과 Shannon limit 근처를 함께 담도록
  잡은 범위다. −4.0 dB에서는 conventional과 NRx 모두 실패하므로 NRx 성공률 상한이 0.80이 된다.
- **Selective가 필요한 이유:** 채널이 좋으면 NRx가 필요 없고, 너무 나쁘면 NRx도 실패한다.
  NRx가 이득을 주는 건 waterfall 구간뿐이다.

## 8. 남은 결정

- 지금 MIG 기반 DART-Rx 논문을 먼저 마무리할지, 바로 MPS 방향으로 넘어갈지 정해야 한다.
  이전 타임라인은 "MIG 기반 논문 먼저, 분산은 그 다음"이었다.
