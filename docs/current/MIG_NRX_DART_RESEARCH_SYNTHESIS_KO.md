# Fixed-MIG NRx service fabric 연구 종합

**기준 시각:** 2026-08-14 17:40 KST  
**상태:** actual multi-endpoint GDR pool 348-run full sweep 완료 · 실제 radio multi-endpoint 통합 준비  
**역할:** 문제 정의, 측정 증거, 설계, novelty boundary, 남은 실험을 하나로 연결하는 현재 기준 문서

---

## 0. 먼저 결론

이 연구의 문제는 `MIG 대 MPS`, `P2P 대 GDR`, 또는 `빠른 RDMA`가 아니다.

> **고정 MIG는 cuPHY L1을 co-tenant 간섭으로부터 보호하지만, dependency-coupled NRx
> service capacity도 partition별로 고정한다. Multi-cell/selective burst가 특정 NRx queue에
> 몰리면 다른 isolated accelerator가 놀아도 deadline miss가 발생한다. L1을 중단하거나
> MIG를 재구성하지 않고 이 유휴 capacity를 빌리려면, remote 실행의 deadline feasibility,
> result expiry, conventional recovery, stale-result commit을 하나의 contract로 다뤄야 한다.**

따라서 제안 방향은 다음이다.

```text
고정된 protected L1 MIG
  + 고정된 resident NRx MIG/GPU pool
  + P2P/GDR registered service fabric
  + utility/deadline/tail-aware admission and placement
  + conventional recovery를 보존하는 epoch/expiry single commit
  + NRx headroom에서만 실행되는 bounded background lease
```

MIG는 연구에서 멀어진 것이 아니라 문제와 해법의 물리적 기반이다. 다만 **MIG 자체가
novelty는 아니고**, `static spatial isolation`과 `dynamic dependent-stage demand`의 충돌을
재구성 없이 해결하는 architecture가 novelty 후보이다.

현재 결과는 이 방향을 지지한다. 특히:

- 단일 NRx endpoint는 capacity를 조금만 넘겨도 p99가 수백 ms로 붕괴한다.
- static placement에서는 한 endpoint가 miss하는 동안 다른 두 3g MIG가 약 66.7% idle인
  경우가 실제 hardware에서 관측됐다.
- compute-only pool에서는 predicted-finish가 queue collapse를 크게 줄였다.
- actual full-size zero-copy GDR pool에서도 process-isolated 3 endpoints가 단일 endpoint보다
  capacity를 실제로 합쳤다.
- P2P/GDR transport는 필요하지만 전체 병목은 NRx service와 queueing이다.
- Qwen뿐 아니라 ResNet, BERT, Whisper에서도 bounded background gating이 queue collapse를
  크게 줄이면서 89--99%의 background work를 보존했다.

actual GDR 87-trace full sweep은 완료됐다. `predicted_finish`는 `static_one`보다 87/87,
`static_cell`보다 86/87 trace에서 no-timely ratio가 낮았다. 그러나 전체 부하에서는 요청의
81.31%를 conventional 대상으로 미리 거부했다. 따라서 fragmentation과 pooling 효과는
확인됐지만, 현재 정책은 최종 해법이 아니다. 이후 actual multi-endpoint radio chain,
risk-budget admission의 open-loop actual-radio 통합과 fault/background 평가가 남아 있다.

actual radio 다중 endpoint 통합의 가장 큰 구현 위험도 별도 gate를 통과했다. QP/MR를
endpoint-agent process가 소유하고 L1 process가 CUDA IPC로 그 GPU allocation을 직접
매핑하는 구조에서 1,415,232B 양방향 GDR echo 10,000회를 오류 없이 수행했다. steady
round-trip은 평균 1.406 ms, p99 1.432 ms였고 CPU payload bounce는 없었다.

이 경계를 실제 radio chain에도 통합했다. 보호된 4g MIG의 cuPHY CE/conventional과 세
full-GPU resident NRx endpoint를 CUDA-IPC mapped GPU MR + GDR로 연결하고, LLR를 다시
LDPC/CRC 및 expiry/epoch single commit에 넣었다. 세 paired trace에서 conventional-only는
62/67/61 correct TB였고, 3-endpoint utility mode는 NRx를 75/100만 호출하면서
80/79/80 correct TB를 얻었다. 요청은 각 endpoint에 25/25/25로 분배됐고 late/expired와
deadline miss는 0이었다. All-NRx 100/100 호출도 median 0.800으로 같았다.

warm-up 제외 Nsight Systems capture에서 source-side NVTX 평균은 CE+GPU pack 1.178 ms,
conventional 1.214 ms, conventional 실행 뒤 남은 remote wait 1.277 ms, NRx 결과의
LDPC/CRC postprocess 1.201 ms, dispatch control 40 us였다. GPU kernel 시간은
float/half conversion이 약 46.4%, 주요 LDPC kernel이 12.7%를 차지했다. 따라서 다음
최적화 우선순위는 QP doorbell이 아니라 conversion/binding, overlapping, NRx service다.

profiling overhead가 없는 100-request run의 중앙값도 같은 결론이다. CE+pack에서 dispatch까지
1.331--1.372 ms, remote exchange 2.004--2.013 ms, 그중 worker NRx service 1.111 ms,
나머지 transport/control 약 0.895--0.902 ms였다. Conventional과 겹치고도 남는 remote
residual은 p50 0.838--0.861 ms였다. 즉 NIC path는 충분히 작지만 0은 아니며, 현재 실제
source wrapper도 1 request/ms보다 느리다. 다음 multi-cell open-loop 실험은 captured tensor
replay와 cell-parallel/batched L1을 분리해야 NRx pool 효과를 왜곡하지 않는다.

---

## 1. 정확히 어떤 시스템을 다루는가

### 1.1 하나의 request

단위 request는 일반 inference query가 아니라 PHY dependency chain의 일부다.

```text
(cell, slot, scheduled PUSCH allocation)
  -> protected cuPHY front / channel estimation
  -> {conventional receiver, local NRx, remote NRx}
  -> LDPC / CRC
  -> downstream absolute expiry
```

NRx는 L1 output tensor를 받고 LLR를 되돌려준다. 따라서 다른 MIG/GPU에 두면 compute
placement뿐 아니라 양방향 tensor path와 결과 유효성 규칙이 필요하다.

### 1.2 현실적인 arrival model

실험 workload는 세 종류를 포함한다.

1. **Single-cell periodic:** 0.5/1.0 ms마다 request가 도착한다.
2. **Multi-cell aggregate:** 1/2/4/8 cells가 synchronized 또는 staggered offset으로
   도착한다.
3. **Selective/bursty NRx:** NRx invocation을 10/25/50/75/100%로 바꾸고 hard-channel
   episode를 Markov burst로 만든다.

세 번째가 중요한 이유는 NRx utility가 모든 channel에서 같지 않기 때문이다. 실제 paired
radio test에서 NRx incremental success는 waterfall SNR 구간에 집중됐다. 좋은 channel과
완전히 나쁜 channel에서는 NRx가 추가 success를 거의 만들지 않으면서 capacity만 소비할 수
있다.

### 1.3 왜 background AI가 포함되는가

통신사업자는 peak NRx demand만 보고 GPU를 비워두지 않는다. 남는 AI partition에는 다음이
올라갈 수 있다.

- Qwen-7B prefill/decode
- ResNet-50 vision inference/training
- BERT-base inference
- Whisper-base streaming speech
- CSI/beam prediction과 같은 RAN-native AI

따라서 올바른 목표는 NRx 전용 overprovisioning이 아니라, L1/NRx deadline을 지키면서
headroom을 background work에 돌려주는 것이다.

---

## 2. 근본 모순과 기존 선택지의 한계

### 2.1 MPS: work-conserving하지만 보호 계약이 약하다

MPS는 local pointer와 공유 실행을 제공하지만 L1, NRx, background kernel이 SM, memory
system, launch queue를 공유한다. Percentage quota는 평균적인 active-thread allocation이지
L1 p99를 보장하는 temporal isolation contract가 아니다.

### 2.2 MIG local: 보호하지만 L1과 NRx의 내부 경합은 남는다

L1과 NRx를 같은 4g MIG에 넣으면 sibling background workload와는 격리된다. 그러나 L1과
NRx는 여전히 같은 제한된 SM/HBM slice를 사용한다. 또한 NRx compute는 pipeline의 실제
stage이므로 E2E가 cuPHY-only 39 ms와 같아질 수는 없다. 비교해야 할 것은:

- protected L1 active time이 own-partition L1-only baseline에 가까운가
- NRx를 포함한 E2E가 deadline 안에 있는가

두 metric이다.

### 2.3 Fixed MIG: spatial isolation과 runtime elasticity가 충돌한다

MIG geometry를 바꾸면 resident model, CUDA context, graph, registered memory를 다시 만들어야
한다. 이는 slot/burst timescale보다 훨씬 느리고 protected L1 drain을 요구할 수 있다.
따라서 fast path에서는 MIG를 바꾸지 않는다.

문제는 demand가 topology를 따르지 않는다는 것이다. 한 cell의 NRx burst가 endpoint 0에
몰려도 endpoint 1/2의 capacity는 자동으로 사용되지 않는다.

### 2.4 P2P/GDR만 붙여도 해결되지 않는다

P2P와 GDR은 isolated endpoint에 도달하게 해주는 data plane이다. 하지만 단순 remote write는
다음을 답하지 못한다.

- 이 request가 deadline 안에 끝날 수 있는가
- 늦을 때 conventional receiver를 언제 시작할 것인가
- 여러 late NRx가 동시에 fallback storm을 만들지 않는가
- 이전 epoch 결과가 재사용된 slot에 commit되지 않는가
- payload visibility와 graph completion 이후 어떤 결과를 공개할 것인가

즉 `reachability`와 `safe elasticity`는 다른 문제다.

---

## 3. 지금까지 실험이 증명한 것

### 3.1 NRx service가 queue cliff를 만든다

단일 resident NRx endpoint의 measured saturation과 open-loop 결과는 다음과 같다.

| endpoint | saturation | 95% load p99 | 105% load p99 | 105% miss |
|---|---:|---:|---:|---:|
| 3g MIG | 621.7 req/s | 1.79 ms | 834.25 ms | 99.61% |
| 4g MIG | 739.7 req/s | 1.53 ms | 803.00 ms | 99.55% |
| full A100 | 1164.1 req/s | 1.01 ms | 643.00 ms | 약 99.6% |

핵심은 평균 service time이 아니다. `lambda`가 effective capacity에 가까워지면 작은 burst와
tail outlier가 backlog를 만들고, p99가 세 자릿수 ms로 뛴다.

동일 MIG GI 안에 TensorRT context를 2/4/8개 추가해도 capacity가 수평 확장되지 않았다.
독립 capacity는 별도 MIG/GPU endpoint에서 나와야 한다.

### 3.2 MIG/MPS/MIG+MPS/P2P/GDR 비교가 말하는 것

3-trial median 기반 low-load/queue-stability 결과다.

| placement | low-load E2E | last stable point | 다음 point의 p99 | 의미 |
|---|---:|---:|---:|---|
| MPS full GPU | 4.242 ms @50/s | >=350/s | 2.551 ms @350/s | 빠르지만 isolation 없음 |
| MIG local | 4.678 ms @50/s | 300/s | 1124.981 ms @350/s | fixed 4g capacity cliff |
| MIG+MPS | 4.979 ms @50/s | 250/s | 259.106 ms @300/s | 같은 GI context는 scale-out 아님 |
| P2P split | 5.269 ms @50/s | 250/s | 451.798 ms @300/s | transport보다 remote service가 지배 |
| zero-copy GDR | 6.109 ms @50/s | 180/s | 1048.695 ms @250/s | single-slot speedup이 아님 |

P2P payload round trip은 약 78--110 us였고 동일-depth GDR는 약 0.438 ms를 더 썼다. 그러나
E2E는 수 ms이고 overload tail은 수백--수천 ms였다.

**판정:** `통신이 전체 병목`이라는 가설은 기각한다. GDR의 가치는 단일 slot을 5 us로
만드는 것이 아니라 CPU bounce 없이 다른 isolated resident endpoint를 service pool에
포함하는 것이다.

### 3.3 Fixed-placement fragmentation은 실제로 존재한다

3개의 물리적 3g MIG endpoint와 87개 multi-cell/selective trace에서 analyzer는
`problem_gate=True`를 냈다.

| trace | static-one no-timely | predicted-finish no-timely | 관측 |
|---|---:|---:|---|
| 1 cell, 1 ms, NRx 100% | 99.97% | 0.13% | static queue가 붕괴하고 2 endpoints idle |
| 4 cells, 1 ms, bursty 10% | 64.85% | 1.61% | aggregate capacity는 있으나 placement가 나쁨 |
| 4 cells, 0.5 ms, bursty 10% | 99.90% | 7.89% | 더 높은 load에서는 fallback도 필요 |

static miss가 발생하는 동안 다른 endpoint idle fraction은 대표적으로 66.7%였다. 이것이
우리 problem의 가장 직관적인 그림이다.

### 3.4 Background contention은 Qwen-only 현상이 아니다

`500 -> 1100 -> 500 req/s` burst에서 naive sharing과 NRx-priority cooperative gating을
비교했다.

| background | naive burst p99 / miss | gated burst p99 / miss | 보존한 work |
|---|---:|---:|---:|
| ResNet-50 | 2211.394 ms / 67.00% | 6.723 ms / 1.24% | 94.0% |
| BERT-base | 1602.273 ms / 57.27% | 2.772 ms / 0.39% | 89.3% |
| Whisper-base | 471.312 ms / 49.64% | 3.070 ms / 0.39% | 98.6% |
| Qwen-7B decode | 2270.974 ms / 67.67% | 5.724 ms / 1.12% | 93.0% |

7--11%의 background work를 양보해 queue collapse를 제거했다. 이는 idle capacity reclaim이
실제 가치가 있지만, arbitrary kernel preemption이 아니라 bounded cooperative work unit이
필요하다는 뜻이다.

### 3.5 Radio utility가 selective admission을 정당화한다

MCS 7 Rayleigh paired test에서 NRx gain은 waterfall 구간에 집중됐다. 실제 single-endpoint
integrated vertical slice의 3 trials에서는:

| mode | NRx requests / 100 | correct-TB ratio | decision p99 |
|---|---:|---:|---:|
| conventional | 0 | 0.64 | 1.074 ms |
| all NRx | 100 | 0.80 | 5.798 ms |
| utility admission | 75 | 0.80 | 5.691 ms |

utility gate는 all-NRx와 같은 delivered outcome을 유지하면서 NRx request를 25% 줄였다.
단, 이 timing gate는 12 ms expiry였으며 production deadline 주장이 아니다.

### 3.6 진행 중인 actual three-endpoint GDR pool

현재 data path는 endpoint마다 독립 GPU MR과 RC QP를 사용한다.

```text
source 4g GPU MR
  1,415,232 B request -> descriptor -> doorbell
     resident 3g direct TensorRT/CUDA Graph NRx
  314,496 B LLR <- completion <- doorbell
```

첫 구현은 endpoint별 Python thread를 사용했다. pyverbs polling이 한 GIL에서 직렬화되어
3 endpoints의 capacity를 합치지 못했다. 이를 endpoint별 독립 process/CUDA context/QP로
수정했다.

1 cell, 1 request/ms, 20,000 requests, experimental 5 ms threshold의 중간 결과:

| implementation/policy | timely | no-timely | timely sojourn p99 |
|---|---:|---:|---:|
| threaded 3-endpoint round-robin | 12 / 20,000 | 99.94% | 4.992 ms |
| process 3-endpoint static-one | 1 / 20,000 | 99.995% | 4.188 ms |
| process 3-endpoint round-robin | 18,684 / 20,000 | 6.58% | 4.976 ms |
| process 3-endpoint shortest-queue | 11,192 / 20,000 | 44.04% | 4.979 ms |
| process 3-endpoint predicted-finish | 15,081 / 20,000 | 24.595% | 3.190 ms |
| process 3-endpoint tail-aware | 16,720 / 20,000 | 16.40% | 4.067 ms |

해석 시 p99는 timely completion에 대해서만 계산되므로 반드시 no-timely와 함께 봐야 한다.

독립 replica 수만 바꾼 현재 sweep의 같은 single-cell trace 중간값도 capacity 경계를
직접 보여준다.

| physical replicas | round-robin timely | predicted-finish timely | predicted p99 |
|---:|---:|---:|---:|
| 1 | 1 / 20,000 | 7,701 / 20,000 | 4.349 ms |
| 2 | 3 / 20,000 | 13,056 / 20,000 | 3.064 ms |
| 3 | 19,441 / 20,000 | 17,605 / 20,000 | 3.042 ms |

1/2-replica RR은 offered rate보다 service capacity가 낮아 queue가 붕괴했다. 3 replicas에서
RR은 97.2% timely까지 올라갔고, predicted-finish는 12.0%를 미리 baseline 대상으로 돌려
remote queue tail을 제한했다. 이 표는 현재 single trial 중간 결과이며 campaign 종료 뒤
반복성과 다른 trace를 함께 판정한다.

- process 분리로 aggregate capacity가 실제로 증가했다.
- queue를 무시한 RR은 93.42%를 timely로 처리하지만 598 remote-expired와 718 late-at-source를
  만들었다.
- predicted-finish는 4,918건을 conventional fallback 대상으로 미리 거부해 remote expiry를
  1건으로 줄이고 p99를 3.19 ms로 낮췄다.
- tail-aware는 3,267건만 거부하면서 83.60% timely와 p99 4.067 ms를 얻었다.
- shortest-queue는 service tail과 deadline slack을 모르므로 충분하지 않았다.

이는 두 가지를 동시에 보여준다.

1. **독립 endpoint pool은 효과가 있다.**
2. **replica 수만 늘리는 것으로는 부족하고 admission의 목표를 명시해야 한다.** RR은 더
   많은 NRx를 시도하고, predicted/tail-aware는 late work를 줄여 baseline recovery를
   가능하게 한다.

1/2/3-replica sweep, 6-trace representative matrix, 87-trace full matrix는 모두 완료됐다.
전체 412개 run 중 full matrix는 348개다. full matrix의 최종 paired 결과는 다음과 같다.

| policy | traces | no-timely median | reject median | late/expired median | timely p99 median |
|---|---:|---:|---:|---:|---:|
| static-one | 87 | 1.0000 | 0.0000 | 37,849 | 4.343 ms |
| static-cell | 87 | 1.0000 | 0.0000 | 61,208 | 4.950 ms |
| predicted-finish | 87 | 0.8133 | 0.8131 | 9 | 4.151 ms |
| tail-aware | 87 | 0.8432 | 0.8422 | 56 | 4.769 ms |

`no-timely`에는 conventional path로 의도적으로 reject한 요청이 포함되므로 PHY miss와 같지
않다. 이 결과는 `predicted-finish`가 쓸모없는 remote late work를 강하게 억제하지만,
상당한 usable NRx opportunity도 버린다는 것을 보여준다. 이것이 risk-budget Policy V2와
actual radio utility 통합의 직접적인 동기다.

---

## 4. 제안 architecture: DART-Rx

논문에서는 메커니즘별 약어를 늘리지 않고 overall architecture와 세 subsection으로만
설명한다.

### 4.1 Fixed isolation and resident receiver fabric

배포 시 topology를 정하고 fast path에서는 변경하지 않는다.

```text
GPU0 4g: protected cuPHY/L1-side source
GPU0/1/2 3g: resident NRx endpoints
remaining isolated headroom: bounded background AI
P2P or GDR: pre-registered tensor service fabric
```

각 endpoint는 다음 상태를 갖는다.

- resident model, TensorRT context, CUDA Graph
- pre-registered input/output ring
- independent queue/QP and health epoch
- tensor credit and service-bound history

P2P는 지원되는 same-node topology에서 낮은 transport baseline이다. GDR은 process/host/GPU
boundary를 통일하고 CPU bounce 없이 remote endpoint를 연결하는 일반 fabric이다. 어느 것도
그 자체로 contribution이라고 주장하지 않는다.

### 4.2 Utility-, deadline-, and tail-aware pool control

요청은 다음 정보를 가진다.

```text
r = (cell, slot, epoch, release, expiry,
     utility class, tensor class, candidate endpoints)
```

controller는 다음 순서로 결정한다.

1. channel/decoder history에서 NRx incremental utility가 양수인지 판단한다.
2. healthy endpoint별 `predicted queue tail + transport + NRx + guard`를 계산한다.
3. expiry 전에 끝날 endpoint의 tensor credit만 reserve한다.
4. feasible endpoint 중 earliest conservative finish를 선택한다.
5. endpoint outlier가 bound를 넘으면 circuit을 닫고 completion feedback까지 새 request를
   보내지 않는다.
6. feasible endpoint가 없으면 remote work를 만들지 않고 conventional path를 선택한다.
7. NRx slack이 충분할 때만 cooperative background work unit을 lease한다.

이 control의 목적은 NRx utilization 최대화 하나가 아니다. 우선순위는:

1. protected L1 tail
2. PHY completion/deadline
3. timely radio utility
4. background utility

이다.

### 4.3 Expiring alternative-result transaction

Remote NRx는 일반 inference result가 아니라 expiry가 있는 alternative result다. admit할
때 다음을 하나의 transaction으로 관리한다.

- remote endpoint/tensor credit
- conventional recovery credit 또는 latest-fallback-start window
- `(slot, epoch)` commit entry

안전한 초기 구현은 conventional baseline을 eager하게 유지한다. 최적화된 구현은 recovery
credit을 먼저 reserve하고 `latest_fallback_start`에 JIT conventional graph를 시작할 수
있다. JIT 버전은 fallback storm 실험을 통과하기 전에는 최종 주장에 포함하지 않는다.

Commit 조건은 다음과 같다.

- slot ID, epoch, endpoint/health epoch 일치
- GPU payload visibility 보장
- completion status OK
- absolute expiry 이전
- LDPC/CRC와 radio validity 통과
- transaction이 아직 open

NRx와 conventional 중 먼저 유효하게 commit한 하나만 architectural state에 공개한다.
late/stale/duplicate result는 private buffer에 도착해도 다음 slot의 state를 바꾸지 못한다.

---

## 5. 왜 이 설계 요소가 필요한가

| 측정된 문제 | 필요한 설계 |
|---|---|
| MPS co-tenant가 L1 tail에 들어옴 | protected fixed MIG |
| dynamic MIG가 burst timescale보다 느림 | no-reconfiguration resident pool |
| static endpoint miss + 다른 endpoint idle | earliest feasible placement |
| capacity 근처의 queue cliff | admission과 conventional fallback |
| rare service/GDR/control outlier | online tail feedback와 circuit breaker |
| NRx utility가 channel별로 다름 | radio-utility admission |
| P2P/GDR completion만으로 결과 의미를 모름 | epoch/expiry/visibility commit guard |
| correlated remote delay가 fallback storm 생성 | recovery credit/reservation |
| background kernel이 NRx queue를 막음 | bounded cooperative lease/gating |
| threaded host dispatcher가 replica capacity를 직렬화 | endpoint별 독립 process, 이후 device queue 후보 |

마지막 행은 중요한 구현 교훈이지만 그 자체가 연구 problem은 아니다. 다만 host polling과
multi-QP orchestration 비용을 정량화해 device-resident queue/DPU/hardware scheduler의 필요성을
판정하는 근거가 된다.

---

## 6. 우리에게 생기는 효과

### 6.1 L1 isolation

L1과 NRx/background를 다른 spatial partition에 두어 same-partition kernel/SM/HBM contention을
제거한다. 목표 metric은 cross-partition L1 active p99가 own-partition L1-only baseline의
1.05배 이내인지다.

### 6.2 Drain-free NRx elasticity

MIG geometry와 resident model을 그대로 둔 채 request placement만 바꿔 burst를 여러
endpoint에 흡수한다. 이는 dynamic MIG보다 빠른 runtime control plane이다.

### 6.3 Queue stability와 useful-work efficiency

끝날 수 없는 NRx를 queue에 넣지 않는다. remote-expired work를 줄이고 conventional path가
deadline을 지킬 시간을 보존한다. NRx throughput 자체보다 timely useful NRx와 delivered radio
utility를 최적화한다.

### 6.4 Background work conservation

peak NRx 기준으로 자원을 영구 유휴화하지 않는다. completion feedback으로 headroom이 있을
때 bounded AI work를 실행하고 burst 전후로 억제한다.

### 6.5 Correctness under remote execution

late result를 단순 SLO violation으로 기록하는 것이 아니라 architectural state에서 폐기한다.
epoch/expiry/single-commit이 없으면 remote pooling은 빠르더라도 PHY pipeline에 안전하지 않다.

---

## 7. Novelty를 어디에 두어야 하는가

### 7.1 novelty가 아닌 것

- MIG가 MPS보다 좋다는 비교
- P2P 또는 NIC GDR 구현
- shortest-queue/earliest-finish routing 단독
- background workload를 pause/resume하는 heuristic 단독
- 실행 중 MIG geometry를 바꾸지 않는다는 사실 단독
- Python GIL을 process로 우회한 구현

이들은 baseline, substrate 또는 engineering requirement다.

### 7.2 strongest novelty candidate

> **Fixed spatial isolation 때문에 파편화된, deadline-coupled optional NRx capacity를
> 재구성 없이 하나의 service pool로 만들고, radio utility, conservative finish,
> mandatory recovery, expiring result commit을 하나의 accelerator transaction으로
> 결합한다.**

일반 GPU load balancing과 구별되는 속성은 다음 조합이다.

- protected real-time producer stage
- dependency-coupled remote consumer stage
- result가 optional하지만 radio utility를 가짐
- deadline 뒤에는 성능 저하가 아니라 invalid result가 됨
- conventional recovery가 반드시 존재해야 함
- late/stale DMA completion을 commit하면 correctness bug가 됨
- fixed isolation을 유지하면서 background utility도 회수함

Flex-MIG류 fixed-partition 분산 실행과 비교할 때 단순 `drain-free`만으로는 부족하다. 우리의
차별점은 batch makespan이 아니라 **PHY deadline, alternative radio result, recovery reservation,
versioned commit, actual cuPHY/TensorRT/GDR integration**에 있어야 한다.

### 7.3 ISCA급 architecture로 만들기 위한 hardware question

현재 software prototype이 기능적 contract를 검증한다. ISCA contribution으로 강화하려면
Nsight/CPU profiling 결과를 기반으로 최소 hardware state를 제안해야 한다.

후보는 다음 정도로 제한한다.

- endpoint tail/credit scoreboard
- absolute-expiry compare와 admission gate
- registered tensor-ring descriptor cache
- payload-visible completion에서 CUDA Graph launch로 이어지는 doorbell path
- `(slot, epoch)` commit table와 stale completion filter
- background lease revoke/drain indication

이 중 host overhead로 실제 critical path가 된 항목만 hardware mechanism으로 채택한다.
측정 없이 새로운 ISA 전체를 먼저 주장하지 않는다.

---

## 8. 논문 구성

### Section 1 · Motivation and problem

- AI-RAN의 protected L1 + selective NRx + background consolidation
- MPS isolation 실패와 fixed-MIG capacity fragmentation
- `miss + eligible idle endpoint` timeline
- dynamic MIG가 맞지 않는 timescale

### Section 2 · Characterization

- MIG/MPS/MIG+MPS/P2P/GDR fair matrix
- L1 active, transport, NRx service, queue wait, E2E 분해
- single-endpoint queue cliff
- multi-cell/selective fragmentation
- four background workloads의 blocking

### Section 3 · Overall DART-Rx architecture

1. Fixed isolation and resident receiver fabric
2. Utility/deadline/tail-aware pool control
3. Expiring alternative-result transaction

### Section 4 · Implementation

- A100 MIG, Aerial/cuPHY, caller-owned TensorRT/CUDA Graph
- P2P/GDR GPU MR ring과 descriptor/doorbell ordering
- process-per-endpoint source agents
- software commit table와 background cooperative units
- 측정으로 정당화된 경우 DART queue-engine model

### Section 5 · Evaluation

- L1 p99/p99.9 isolation
- 1/2/3 endpoint capacity scaling
- 1/2/4/8-cell, periodic/synchronized/staggered/selective/bursty
- no-timely decomposition: rejected, remote-expired, late, overflow
- BLER/CRC/delivered utility
- background useful work
- CPU overhead와 kernel/copy/sync attribution
- fault injection: late, stale epoch, endpoint failure, correlated outlier

---

## 9. 현재 판정과 남은 실험

### 9.1 현재 가장 타당한 branch

현재 증거는 다음 두 원인이 결합된 방향을 지지한다.

- **A: fixed-capacity fragmentation**이 primary problem이다.
- **C: background cooperative-unit blocking**이 NRx tail을 증폭한다.

P2P/GDR transport 자체와 같은-GI context 부족은 primary solution이 아니다.

### 9.2 actual GDR pool campaign 완료

결과 root:

```text
/mydata/results/isca_v2/gdr_pool_20260814T014651Z
```

완료 상태:

1. process 기반 3-endpoint smoke: 완료
2. 1/2/3 physical endpoint replica sweep: 완료
3. 6 representative traces x 6 policies: 완료
4. 87 traces x 4 policies = 348 runs: 완료
5. analyzer 412-run validation: 완료
6. GPU 0 원래 4g+3g MIG, GPU 1/2 full mode, NIC loopback 원상 복구: 완료

로컬 raw 결과:

```text
/Users/changjongkim/New_research/cloudlab_results/task1_final/gdr_pool_20260814T014651Z
```

### 9.3 이 campaign이 답한 것

- actual full-size GDR에서도 independent endpoint 추가가 usable capacity를 늘린다.
- 1 request/ms single-cell trace에서 1/2 replica RR은 붕괴하고 3 replica RR은 97.2%
  timely로 올라갔다. 최소 stable boundary는 workload와 tail에 따라 달라진다.
- compute-only에서 관측한 fragmentation은 real GDR transport에서도 유지됐다.
- 현재 구현에서는 fixed predicted-finish가 runtime tail-aware보다 full matrix median과
  dominance가 좋았다.
- late work를 거의 제거할 수 있지만 과도한 reject가 발생하므로 radio utility를 포함한
  explicit risk budget이 필요하다.

### 9.4 이후 반드시 남는 것

1. multi-endpoint GDR pool과 actual cuPHY/conventional/LDPC/CRC vertical slice 통합
2. measured CQI/DMRS/decoder history 기반 radio-derived selective trace
3. eager baseline과 JIT reserved fallback 비교 및 fallback-storm fault injection
4. NRx pool과 4종 background workload를 동시에 실행
5. Nsight Systems/Compute와 CPU profile로 CUDA API/kernel/copy/sync/polling 분해
6. 실제 gNB scheduling/HARQ에서 primary expiry 확정
7. software host queue와 device/DPU/hardware queue의 cost-benefit 비교

---

## 10. 지금 말할 수 있는 것과 아직 말하면 안 되는 것

### 말할 수 있는 것

- fixed MIG는 L1 isolation과 NRx elasticity 사이의 실제 tension을 만든다.
- static placement fragmentation과 endpoint idle 공존은 actual A100에서 관측됐다.
- independent physical endpoints는 같은 GI의 context 증가와 다르게 capacity를 합칠 수 있다.
- transport보다 NRx service/queue가 주 병목이다.
- deadline-aware admission은 remote-expired work를 크게 줄인다.
- background gating 효과는 Qwen뿐 아니라 vision/language/speech에서 반복된다.
- actual radio test는 utility-selective NRx의 필요성을 지지한다.

### 아직 말하면 안 되는 것

- 현재 5 ms가 production PHY deadline이다.
- current GDR pool이 full cuPHY end-to-end architecture를 완성했다.
- tail-aware가 모든 trace에서 최선이다.
- NIC GDR가 P2P보다 빠르다.
- 현재 software prototype이 hard real-time guarantee를 제공한다.
- DART queue hardware가 반드시 필요하다.
- synthetic Markov burst가 실제 radio traffic을 완전히 대표한다.

---

## 11. 한 문장 research direction

> **MIG를 동적으로 바꾸는 대신 고정된 protected L1과 resident NRx pool을 유지하고,
> radio utility와 measured tail을 이용해 request·recovery·commit을 deadline-aware transaction으로
> 제어함으로써, L1 isolation을 잃지 않으면서 bursty NRx와 background AI의 유휴 capacity를
> drain-free하게 회수한다.**
