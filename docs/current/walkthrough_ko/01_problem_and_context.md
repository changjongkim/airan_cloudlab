# Chapter 1 · Problem and context

**MIG–NRx AI-RAN research walkthrough (KO) · docs/current/walkthrough_ko/01_problem_and_context.md 페이지**

네비게이션: [Index](README.md) · Prev: - · Next: [02 Architecture](02_architecture.md)

---

# DART-Rx 연구 전체 설명: 문제 → 설계 → 실험 결과

**기준일:** 2026-08-16
**이 문서의 역할:** 처음 보는 사람이 문제, 설계, 실험 조건, 결과, 아직 남은 한계를 한 번에 이해하기 위한 대표 문서  
**데이터 원칙:** 성능 PNG figure는 보존된 CSV/JSON/SQLite 실험 결과에서 생성했다. 배치
architecture map과 Mermaid 블록은 성능 수치가 아니라 실제 실험 topology를 설명하는 설계도이다.
**영문판:** [RESEARCH_WALKTHROUGH_EN.md](RESEARCH_WALKTHROUGH_EN.md)

---

## 0. 이 연구를 먼저 쉬운 말로 설명하면

> **기지국의 필수 작업인 L1은 자기 GPU 공간에서 보호한다. AI 수신기인 NRx는 필요한
> slot에만 호출하고, 현재 가장 빨리 끝낼 수 있는 상주 NRx endpoint로 보낸다. AI 결과가
> 늦거나 잘못된 slot의 결과이면 버리고, 기존 수신 결과를 사용한다. 이 모든 과정에서
> MIG 구성을 바꾸거나 L1을 재시작하지 않는다.**

기술적으로 표현하면 DART-Rx는 **고정 MIG 위에 만드는 deadline-aware NRx service pool**이다.
여기서 중요한 것은 단순히 NIC로 tensor를 복사하는 것이 아니다. 어떤 NRx endpoint를 쓸지,
지금 보내도 제시간에 끝날지, AI 결과가 늦었을 때 무엇을 쓸지, 놀고 있는 GPU에서 실행하던
background AI를 언제 줄일지를 하나의 slot 처리 규칙으로 묶는다.

이 연구는 “MIG가 MPS보다 빠르다” 또는 “GDR가 P2P보다 빠르다”를 주장하는 논문이 아니다.
MIG/MPS/P2P/GDR는 비교해야 하는 **mechanism과 baseline**이고, 실제 연구 문제는 다음
모순이다.

```text
MIG isolation은 필요하다.
        ↓
하지만 static partition/placement는 NRx service capacity를 고정한다.
        ↓
multi-cell·selective burst에서는 한 queue가 무너지는데 다른 endpoint는 놀 수 있다.
        ↓
MIG를 즉시 재구성할 수 없으므로, 고정된 isolated endpoint 사이에서 capacity를 빌려야 한다.
        ↓
remote NRx는 늦을 수 있으므로 admission, routing, fallback, commit을 함께 설계해야 한다.
```

현재 결과는 이 인과관계의 각 부분을 실제 하드웨어에서 확인했다. 다만
`actual-radio + concurrent multi-cell arrivals + multi-endpoint GDR + background reclaim`을
한 번에 실행한 마지막 통합 실험은 아직 남아 있다. 따라서 현재 상태는 **problem과 각
mechanism은 강하게 지지되지만 최종 ISCA claim은 아직 완결되지 않은 상태**다.

### 0.1 이 문서에서 반복해서 쓰는 말

| 용어 | 쉬운 뜻 | 이 연구에서의 정확한 뜻 |
|---|---|---|
| slot / request | 한 번 처리해야 하는 무선 데이터 단위 | 한 cell의 scheduled PUSCH 처리 transaction |
| L1 | 반드시 제시간에 끝나야 하는 기본 수신 작업 | cuPHY channel estimation, LDPC/CRC 등을 포함한 PHY critical path |
| NRx | 어려운 신호에서 도움을 주는 AI 수신기 | conventional receiver 대신 선택적으로 사용할 수 있는 TensorRT neural receiver |
| service capacity | 1초에 처리할 수 있는 양 | queue가 계속 늘지 않는 최대 NRx request rate |
| queue cliff | 조금 더 들어왔는데 대기시간이 폭발하는 지점 | arrival rate가 service rate에 가까워질 때 p99가 비선형적으로 증가하는 현상 |
| host blocking | CPU thread가 CUDA 호출 안에서 기다리는 현상 | 이미 GPU에 제출된 작업의 완료가 `cudaFree`, sync, copy 같은 API에서 드러나는 것 |
| expiry | 결과를 써도 되는 마지막 시각 | 이 시각 뒤의 NRx 결과는 정확해도 현재 slot에는 사용하지 않음 |
| commit | 최종 결과를 하나만 선택하는 것 | conventional 또는 NRx 중 조건을 통과한 결과 하나만 PHY state에 반영 |
| endpoint | 항상 준비된 NRx 실행 단위 | model, TensorRT context, CUDA Graph, buffer가 미리 올라간 process/GPU instance |

### 0.2 숫자로 미리 보는 현재 결론

- **MIG 격리는 실제로 강했다:** sibling 3g에 Qwen을 실행해도 4g NRx capacity 변화는
  `-0.11%`였다.
- **하지만 고정된 한 endpoint는 넘치면 무너졌다:** MIG local은 300/s에서 p99 `3.47 ms`였지만
  350/s에서 `1124.98 ms`가 됐다.
- **같은 공간의 NRx는 L1 host까지 막을 수 있었다:** 40-cell same-MIG co-location에서 추적한
  CUDA API host time이 `1.68 s → 25.35 s`, `15.1×` 증가했다.
- **MPS는 NRx client 수가 늘면 급격히 무너졌다:** full-A100 MPS에서 독립 NRx process를
  1개에서 8개로 늘리자 20-cell L1 p99가 `42.3 → 189.3 ms`; 4g MIG+MPS에서는
  `40.7 → 435.7 ms`가 됐다.
- **MPS quota만으로는 해결되지 않았다:** 한 4g에서 L1:NRx를 30:70에서 70:30으로 바꾸자
  E2E mean이 `4.76 → 6.50 ms`로 악화됐다.
- **다른 endpoint를 쓰는 비용보다 queue collapse가 훨씬 컸다:** P2P와 NIC GDR의 depth-1
  평균 차이는 `0.438 ms`였고, overload tail은 수백–수천 ms였다.
- **NRx는 실제 radio 가치가 있었다:** utility admission은 all-NRx와 같은 correct-TB `0.80`을
  유지하면서 NRx 요청을 25% 줄였다.

---

# Part I. Background and problem

## 1. 어떤 AI-RAN 실행 환경을 다루는가

하나의 uplink request는 독립적인 AI query가 아니라 PHY dependency chain이다.

```text
(cell, slot, scheduled PUSCH)
  → cuPHY channel estimation / front-end
  → conventional receiver와 선택적 Neural Receiver(NRx)
  → LDPC / CRC
  → absolute decision expiry 전에 하나의 결과 commit
```

동시에 같은 GPU 시스템에는 다음 세 종류의 일이 존재한다.

1. **Protected L1:** deadline과 tail latency를 지켜야 하며 background AI 때문에 흔들리면 안 된다.
2. **NRx:** 특정 channel 구간에서만 conventional receiver보다 이득이 있는 optional PHY stage다.
3. **Background AI:** NRx peak를 위해 항상 비워둘 수 없는 GPU headroom에서 Qwen, BERT,
   Whisper, vision/RAN AI 등을 실행한다.

NRx는 매 slot에 반드시 필요한 단계가 아니다. 채널이 좋은 slot은 기존 수신기로 충분하고,
어려운 채널에서만 NRx가 실제 radio 성공률을 높일 수 있다. 여러 cell의 주기적 slot이
겹치거나 어려운 채널이 연속되면 NRx 요청은 짧은 시간에 몰린다. 따라서 하루 평균 GPU
사용률보다 **몇 ms 동안 queue가 얼마나 쌓이는지**와 **늦은 결과를 안전하게 버릴 수
있는지**가 더 중요하다.

### 1.1 MPS, MIG, MIG+MPS를 같은 기준으로 보면

세 기술은 서로의 상위·하위 버전이 아니다. 서로 다른 문제를 해결한다.

먼저 다섯 배치를 같은 그림 문법으로 보면 차이가 선명해진다. **굵은 검은 선은 MIG가
만드는 하드웨어 격리벽**이고, 점선 client box는 MPS의 실행 몫일 뿐 격리벽이 아니다.
P2P와 GDR는 L1과 NRx 사이에 벽을 만든 뒤 그 벽을 가로질러 tensor를 전달하는 두 data
path다.

![MPS, MIG, MIG+MPS, MIG+P2P, MIG+GDR의 실제 배치와 데이터 경로](../figures/00_architecture_map.png)

그림의 P2P에는 별표가 붙는다. 이번 P2P 실험은 **한 process가 두 MIG CUDA context를
소유하고 peer access가 실제로 성공한 topology**를 측정했다. 이를 모든 cross-process MIG
조합에서 항상 가능한 일반 경로로 해석하면 안 된다. 반면 GDR 실험은 L1/NRx를 별도
process와 GPU memory registration으로 분리하고, payload를 CPU DRAM에 올리지 않은 채 NIC
loopback으로 전달했다.

**일반 full GPU끼리는 P2P로 multi-GPU 통신할 수 있지만, 이 연구의 A100 MIG 조건은 다르다.**
현재 NVIDIA MIG 가이드는 R570 계열에서 같은 physical GPU 안의 MIG instance 사이 P2P만
지원하고, 서로 다른 physical GPU의 MIG끼리 또는 MIG와 non-MIG GPU 사이 P2P는 지원하지
않는다고 명시한다. GPU Instance 사이 CUDA IPC도 지원되지 않는다. 우리 R580 P2P 실험이
성공한 이유도 정확히 같은 A100 안의 `2g↔2g`를 한 process가 소유한 지원 범위였기 때문이다.

| 경로 | 도달 가능한 범위 | 현재 실험에서의 위치 |
|---|---|---|
| Local | 같은 MIG/CUDA device | 가장 짧지만 L1과 NRx가 같은 execution domain을 공유 |
| CUDA P2P | **같은 physical A100 안의 peer-capable MIG pair** | Stage 1의 single-process `2g↔2g` fast-path/lower-bound baseline |
| NIC GDR | 같은 GPU의 다른 MIG, **다른 physical GPU의 MIG**, 별도 process/container; RDMA network면 다른 host도 가능 | Stage 1 loopback과 Stage 2·3 multi-physical-GPU resident pool의 primary fabric |

따라서 P2P와 GDR가 모두 multi-device tensor 전송이라는 점은 맞지만, 우리 target topology에서
P2P가 GDR를 대체하지는 못한다. P2P는 한 physical GPU 안에서 가능한 짧은 경로이고, GDR는
다른 physical GPU의 MIG까지 pool에 포함시키는 경로다. [NVIDIA MIG Deployment
Considerations](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/deployment-considerations.html)은
이 P2P/IPC 범위를 명시하며, [CUDA multi-GPU guide](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)는
일반 P2P 지원 여부를 device pair의 `cudaDeviceCanAccessPeer()` 결과로 정의한다.

| 방식 | 쉽게 말하면 | 잘하는 것 | 해결하지 못하는 것 |
|---|---|---|---|
| Full GPU + MPS | 한 GPU를 여러 process가 함께 사용 | 남는 SM을 다른 일이 즉시 사용하므로 처리량이 높고 work-conserving | 물리적 경계가 없고, 긴 AI 작업과 GPU queue가 L1 tail을 흔들 수 있음 |
| MIG local | GPU 안에 물리적으로 벽을 만들고, 한 방에 L1+NRx 배치 | sibling MIG의 Qwen 같은 작업으로부터 보호 | 같은 4g 방 안의 L1과 NRx는 여전히 SM·memory·CUDA work queue를 공유 |
| MIG + MPS local | MIG 방 안에서 L1/NRx process 몫을 다시 나눔 | sibling 격리를 유지하며 한 GI 안의 평균 share 조절 | 새로운 하드웨어 격리를 만들지 않으며, quota는 deadline/preemption 보장이 아님 |
| Cross P2P | 같은 physical A100의 peer-capable MIG에 L1과 NRx를 나누고 CUDA peer path로 연결 | L1과 NRx compute queue를 분리하면서 낮은 transport 비용 | 다른 physical GPU의 MIG나 cross-process GI pool에는 사용할 수 없고 remote NRx capacity도 유한 |
| Cross NIC GDR | 각 GPU memory를 NIC에 등록하고 직접 연결 | CPU DRAM bounce 없이 다른 MIG/process/GPU를 endpoint로 사용 | NIC가 NRx compute를 빠르게 만들지는 않으며 queue/admission 문제는 별도 |

핵심 차이는 세 축이다.

1. **격리:** L1이 다른 AI kernel 때문에 느려지는 것을 하드웨어가 막아 주는가?
2. **남는 자원 활용:** 한쪽 queue가 바쁠 때 다른 쪽의 여유 capacity를 빌릴 수 있는가?
3. **대기 위치:** GPU 작업이 밀렸을 때 queue에서 기다리는가, CPU CUDA 호출 안에서
   기다리는가, 아니면 명시적인 transport completion에서 기다리는가?

MPS는 client 수가 작을 때 local work conservation에 강하지만 1번이 약하고, 독립 client와
kernel launch rate가 증가하면 그 장점도 급격히 사라진다. MIG는 1번에 강하지만 고정
partition 때문에 2번이 약하다. MIG+MPS는 MIG 벽 안에서 share를 조절할 뿐 이 모순을
없애지 않는다. P2P/GDR는
벽을 없애는 기술이 아니라, **벽을 유지한 채 다른 방의 NRx를 호출할 수 있게 하는 data
path**다. DART-Rx가 필요한 이유는 이 data path 위에서 어떤 요청을 어디로 보낼지와 늦은
결과를 어떻게 처리할지를 결정해야 하기 때문이다.

### 1.2 같은 GDR라도 세 실험과 최종 목표를 구분해야 한다

앞의 다섯 배치 그림에서 `MIG+GDR`는 **물리 GPU 한 개 안의 2g L1과 2g NRx를 NIC
loopback으로 연결한 placement baseline**이다. 이후 보고서에서 말하는 `NRx 3개`는 이
baseline을 세 번 복제한 단순 그림이 아니라, 여러 물리 GPU에 상주시킨 세 TensorRT endpoint를
하나의 request-level pool로 연결한 별도 실험이다.

세 실험은 같은 이름의 GDR를 사용하지만 topology와 검증 대상이 서로 다르다. 그림, 물리 배치,
측정 결과와 claim boundary는 **Part IV §15에서 Stage 1 → Stage 2 → Stage 3 순서로** 함께
정리한다. 세 Stage를 결합하는 작업은 완료된 결과처럼 보이지 않도록 **남은 통합 실험**으로 분리한다.

여기서 “MIG를 연장한다”는 말은 물리적으로 `4g+3g`를 하나의 CUDA device로 합친다는 뜻이
아니다. 각 NRx request는 endpoint 하나에서 끝나고, 여러 cell-slot 요청을 서로 다른 endpoint가
병렬 처리한다. 따라서 정확한 표현은 **fixed MIG 위에서 NRx service capacity를 request
단위로 scale out한다**이다.

또한 임의의 빈 GPU를 발견하자마자 NRx로 바꾸는 구조도 아니다. Slot fast path에 참여하는
endpoint에는 model, TensorRT context, CUDA Graph와 registered buffer가 미리 올라가 있어야
한다. 남는 4g/full-GPU domain에는 background AI를 실행할 수 있지만, 이를 NRx endpoint로
전환하려면 사전 provisioning이나 fast path 밖의 activation 단계가 필요하다.

### 1.3 출발점: 세 가지 local 배치를 실제로 사용하면 무슨 문제가 생기는가

여기서 `MPS`, `MIG`, `MIG+MPS`라는 이름만 비교하면 배치가 모호해진다. 이 연구에서 실제로
비교한 세 local baseline은 다음과 같다.

| 이름 | 실제 배치 | 이 방식에서 기대한 것 |
|---|---|---|
| Full MPS | full A100 하나에 L1+NRx client와 Qwen client를 함께 배치 | 남는 GPU 자원을 모두 활용하고 workload별 share 조절 |
| MIG local | 4g MIG에 L1+NRx, sibling 3g MIG에 Qwen | Qwen은 격리하면서 L1과 NRx는 가까이 배치 |
| MIG+MPS local | 4g MIG 안에서 L1/NRx를 별도 MPS client로 분할, sibling 3g에 Qwen | sibling 격리와 4g 내부 share 조절을 동시에 사용 |

![MPS, MIG, MIG+MPS 내부 배치가 각각 남기는 문제](../figures/00_three_local_baselines.png)

#### A. Full MPS만 사용하면: 자원은 잘 쓰지만 보호 경계가 없다

MPS는 full GPU를 여러 client가 공유하므로 세 방식 중 raw capacity가 가장 높았다. Background가
없는 absolute-rate sweep에서는 350 request/s까지 p99가 `2.551 ms`로 안정적이었다. 따라서
MPS를 단순히 느린 방식이라고 부르면 틀린다.

문제는 background 몫을 높일 때 같은 GPU의 RAN 경로도 함께 변한다는 것이다.

| Qwen MPS cap | slot E2E mean / p99 | Qwen throughput |
|---:|---:|---:|
| 30% | 5.865 / 6.307 ms | 7.92 it/s |
| 50% | 6.226 / 6.782 ms | 11.14 it/s |
| 70% | 6.656 / 8.066 ms | 17.24 it/s |
| 100% | **8.569 / 11.180 ms** | **21.11 it/s** |

즉 MPS는 높은 총 처리량과 높은 background utility를 주지만, 그 대가가 L1/NRx latency에
직접 나타난다. Active-thread percentage는 평균 자원 몫을 조절할 뿐, 긴 kernel을 일정 시간
안에 preempt하거나 L1의 p99를 보장하지 않는다. 이 연구에서 MPS의 문제는 “성능이 낮다”가
아니라 **실시간 L1 보호와 background 처리량이 하나의 load-dependent Pareto로 묶인다**는
것이다.

여기서 더 중요한 제한이 있다. 위의 350 request/s 결과는 **최적화된 NRx 실행 경로 한
개와 background가 없는 조건**이다. 여러 cell/service가 독립 process로 NRx를 동시에
밀어 넣는 경우의 MPS scaling을 보여주는 결과가 아니다. 이를 별도의 원인 분석 실험에서
측정하자 결론이 크게 달라졌다.

![독립 NRx process 수가 증가할 때 MPS의 L1 p99와 kernel gap 붕괴](../figures/00c_mps_multi_nrx_breakdown.png)

- Full A100 MPS도 NRx process `1→8`에서 L1 p99가 `42.3→189.3 ms`, **4.5×**가 됐다.
- 4g MIG 안의 MPS는 `1→6→8`에서 `40.7→331.4→435.7 ms`, 최종 **10.7×**가 됐다.
- 4g 조건의 L1 inter-kernel gap 중앙값은 `1.15 us(N=1) → 119.7 us(N=6) →
  379.1 us(N=8)`로 변했고, L1 GPU duty cycle은 `31.6%→13.8%`로 줄었다.

따라서 정확한 결론은 “MPS가 좋다”가 아니다. **MPS는 작은 client 수와 여유 load에서는
효율적이지만, 독립 NRx context와 aggregate kernel-launch pressure가 증가하면 물리적
격리가 없기 때문에 scheduler/launch queue/implicit synchronization tail이 L1에 직접
전파된다.** `N=6`은 이 max-rate NRx workload의 실측 knee이지 모든 workload에 공통인
상수는 아니다. 이 과거 실험은 20-cell L1 경로를 사용했으므로 현재 최적화 chain과
절대 ms를 직접 비교하지 않고, scaling 원인의 인과 증거로 사용한다.

#### B. MIG만 사용하면: 옆방은 막지만 같은 방의 L1과 NRx는 막지 못한다

MIG의 sibling isolation은 매우 강했다. 4g NRx가 단독일 때 `745.1 request/s`, sibling 3g에
Qwen을 함께 실행했을 때 `744.2 request/s`로 capacity 차이는 `-0.11%`였다. 따라서 “MIG가
Qwen 간섭을 막지 못했다”는 해석은 맞지 않는다.

하지만 4g 안에 L1과 NRx를 함께 넣으면 둘은 같은 GPU instance 안에 있다. 이 조건에서 L1
active time은 단독 대비 `1.621×`가 됐고 slot E2E mean은 `6.191 ms`였다. 옆 3g의 Qwen은
격리됐지만, 같은 4g 안의 NRx kernel, memory traffic, CUDA work queue는 L1과 격리되지 않은
것이다.

또한 한 4g endpoint의 NRx 처리량은 고정된다. Absolute-rate 실험에서 300/s의 sojourn p99는
`3.470 ms`였지만 350/s에서는 `1124.981 ms`로 폭증했다. 동시에 다른 MIG가 놀더라도 그
capacity는 이 queue로 자동 이동하지 않는다. 따라서 MIG의 문제는 격리가 약해서가 아니라
**격리와 함께 service capacity도 방별로 고정된다**는 것이다.

#### C. MIG+MPS를 결합하면: 방 안의 몫은 바꾸지만 방의 크기와 벽은 그대로다

MIG+MPS는 의미가 있다. Sibling 3g Qwen을 약 `10.21–10.22 it/s`로 격리하면서 4g 내부의
L1/NRx share를 바꿀 수 있다. Proper two-client quota 실험은 두 process를 같은 4g MIG의 MPS
client로 두고, process 사이에는 동일한 full-size GDR payload contract를 사용해 share 효과를
측정했다. 그 결과는 다음과 같다.

| L1:NRx share | E2E mean / p99 |
|---:|---:|
| 30:70 | **4.757 / 5.087 ms** |
| 50:50 | 4.971 / 5.099 ms |
| 70:30 | **6.499 / 6.747 ms** |

긴 stage인 NRx의 몫을 70%에서 30%로 줄이자 L1 몫은 커졌지만 전체 E2E는 `36.6%` 느려졌다.
또한 uncapped two-client absolute-rate 실험은 250/s에서 p99 `3.437 ms`, 300/s에서
`259.106 ms`로 무너졌다. MPS가 한 4g 안의 평균 share는 바꿀 수 있어도 다음 세 가지는
만들지 못한다.

1. L1과 NRx 사이의 새로운 물리적 격리
2. 4g 바깥의 idle capacity를 가져오는 경로
3. kernel completion이나 host CUDA blocking에 대한 hard bound

따라서 세 local 방식의 결론은 “모두 나쁘다”가 아니다.

| 방식 | 얻는 것 | 끝까지 남는 문제 |
|---|---|---|
| Full MPS | 가장 높은 raw capacity와 work conservation | L1 tail이 co-tenant load와 결합 |
| MIG local | 강한 sibling isolation | 같은 GI의 L1–NRx contention과 fixed capacity |
| MIG+MPS local | sibling isolation + 내부 share 조절 | 같은 GI의 contention, fixed capacity, quota tuning trade-off |

어느 것도 **L1과 NRx를 물리적으로 분리하면서, 필요할 때 다른 곳의 NRx capacity를 빌리는
기능**을 동시에 제공하지 않는다. 이것이 P2P와 GDR를 고려한 출발점이다.



### 1.4 그래서 왜 P2P와 GPUDirect RDMA를 고려했는가

첫 질문은 “L1과 NRx를 다른 GPU execution domain으로 분리하면 L1이 실제로 회복되는가?”였다.
두 번째 질문은 “분리된 NRx에 tensor를 보내는 비용이 얻는 이점보다 너무 크지는 않은가?”였다.

![L1과 NRx를 분리하기 위해 P2P와 GDR가 필요한 이유](../figures/00b_why_p2p_gdr.png)

#### 설계 동기 A: P2P로 compute queue를 분리해 보았다

`4g L1+NRx` local 배치에서는 L1 slowdown이 `1.621×`, `MIG+MPS local`에서는 `1.702×`였다.
L1과 NRx를 `2g L1 | 2g NRx`로 나누고 실제 `cudaMemcpyPeerAsync` 경로를 사용하자 L1
slowdown은 **`1.043×`**로 줄었다. Forward+backward P2P copy는 ring-depth-2 run에서 평균
`76.84 us`였다.

이 결과는 중요한 인과관계를 보여준다. L1 slowdown의 상당 부분은 sibling Qwen이 아니라
L1과 NRx가 같은 execution domain을 사용한 데서 왔다. 다만 2g NRx는 4g NRx보다 느려서 slot
E2E는 `6.191 → 6.383 ms`로 소폭 증가했다. 따라서 분리는 L1 보호에는 성공했지만, 작은
NRx slice의 compute 손실까지 자동으로 보상하지는 않았다.

#### 설계 동기 B: P2P만으로 부족한 배치를 위해 NIC GDR를 검증했다

P2P는 가장 짧은 native GPU data path이므로 중요한 lower-bound baseline이다. 하지만 모든
MIG/GPU/process 조합에서 peer access가 가능한 것은 아니며, peer-accessible topology 범위에
묶인다. 반면 NIC GDR는 각 process의 GPU memory를 NIC에 등록하여 CPU DRAM을 거치지 않고
다른 isolated endpoint에 request와 result를 전달할 수 있다.

이 CloudLab A100/ConnectX-6 Dx에서는 실제 GPU memory MR과 NIC physical loopback을 사용했다.
비교한 optimized direct-TensorRT payload는 request `1,415,232 bytes`, result `314,496 bytes`였고
CPU DRAM staging은 사용하지 않았다. 같은 depth-1 조건의 결과는 다음과 같다.

| 경로 | slot E2E mean / p99 | Qwen | 의미 |
|---|---:|---:|---|
| Cross P2P | 5.888 / 6.224 ms | 10.22 it/s | peer-accessible topology의 native lower bound |
| Cross NIC GDR | 6.326 / 6.846 ms | 10.24 it/s | process/GPU isolation을 넘는 zero-CPU-bounce path |

이 구현에서 NIC GDR는 P2P보다 평균 `0.438 ms`, 약 7.4% 느렸다. 예상했던 5–10 us
single-slot improvement는 관측되지 않았다. 그럼에도 GDR를 사용하는 이유는 P2P보다 빠르기
때문이 아니라 **P2P로 직접 연결할 수 없는 resident NRx도 하나의 pool에 포함할 수 있기
때문**이다.

#### 설계 동기 C: 연결만 하지 말고 여러 endpoint를 하나의 pool로 사용해야 한다

P2P/GDR 하나를 연결해도 그 remote endpoint가 넘치면 queue collapse는 다시 발생한다.
실제로 1 request/ms trace를 하나의 고정 NRx endpoint에만 보냈을 때 p99는 `3293.25 ms`였고,
그동안 전체 endpoint의 66.7% 이상이 idle인 구간이 있었다. 세 endpoint 중 예상 완료시각이
가장 빠른 곳을 선택하자 p99는 `1.63 ms`, timely-result 실패율은 `99.97% → 0.13%`가 됐다.

이 세 실험이 설계로 이어지는 순서는 다음과 같다.

```text
MPS: 높은 처리량, 그러나 L1과 co-tenant가 함께 흔들림
MIG: L1 보호 가능, 그러나 capacity가 고정되고 같은 GI contention은 남음
MIG+MPS: 한 GI 안의 share 조절, 그러나 isolation/elasticity 모순은 그대로
        ↓
P2P: L1–NRx compute queue를 분리할 수 있음을 실증
GDR: P2P 범위 밖의 isolated NRx도 CPU bounce 없이 연결
        ↓
여러 resident NRx를 하나의 pool로 구성
        ↓
deadline/utility를 보고 endpoint를 선택하고, 늦은 결과는 conventional path로 복구
```

DART-Rx 설계상 같은 MIG에서는 local, 같은 physical GPU의 지원되는 MIG pair에는 P2P를
사용할 수 있다. 그러나 **현재 구현된 multi-physical-GPU NRx pool의 primary path는 GDR**이며,
P2P pool backend는 구현·평가하지 않았다. 연구의 핵심은 data path 선택 자체가 아니라
**이 서로 다른 endpoint를 하나의 deadline-safe receiver service로 만드는 admission,
reservation, expiry, commit 규칙**이다.

### 1.5 다섯 방식을 한 장에서 읽는 법

아래 한 장은 다섯 방식의 **비교 결과값**을 네 패널로 묶는다. (a)–(c)는 같은 placement
실험이고, Full MPS는 isolated placement의 Qwen `10.22–10.24 it/s`에 가장 가까운 측정점인
50% cap(`11.14 it/s`)을 사용했다. (d)는 여러 독립 NRx process를 올린 별도 원인 분석
실험이다.

![MPS, MIG, MIG+MPS, P2P, NIC GDR의 L1 보호·E2E·background·scaling 실측](../figures/03g_fiveway_measured_evidence.png)

- **(a) L1 보호:** Full MPS, MIG local, MIG+MPS는 낮은 부하에서도 L1 active time이
  `1.601×`, `1.621×`, `1.702×`가 됐다. L1과 NRx를 다른 MIG로 분리한 P2P는 `1.043×`였다.
  NIC GDR는 그래프에서 `1.103×`로 표시했다. 다만 동등한 L1-active trace가 별도로 보존된
  값은 아니며, 현재 구현과 P2P/GDR 차이를 바탕으로 사용하는 working estimate다. 이
  provenance 설명은 그래프를 복잡하게 만들지 않도록 본문에만 둔다.

`1.103×`의 크기가 완전히 임의인 것은 아니다. 같은 depth-1 transport 실험에서 GDR와 P2P의
E2E p99 비율은 `6.846 / 6.224 = 1.100×`였고, 두 GDR repeat를 P2P 집계값과 각각 비교한
범위는 `1.072–1.128×`였다. 따라서 `1.103×`는 **약 1.10× 수준의 대표값으로는 합당**하다.
그러나 `l1_active = CE/front + LDPC/CRC/back`인 반면 RDMA 대기시간 대부분은 이 두 구간
밖에 있으므로, E2E 비율이 곧 L1 active-time 비율이라는 보장은 없다. 최악의 경우 GDR–P2P
평균 E2E 차이 `0.438 ms` 전부를 P2P의 2g L1 baseline `2.429 ms`에 부과하면 상한은
`1.223×`; 아무 차이도 L1 kernel 구간에 들어오지 않으면 하한은 P2P의 `1.043×`다.
`1.103×`는 이 물리적으로 가능한 구간 안에 있지만, 논문에서는 세 자리 실측치가 아니라
**약 `1.10×`의 provisional value**로 해석해야 한다. 확정에는 GDR producer의 CE/front와
LDPC/CRC/back을 CUDA event로 따로 재측정하는 조건 일치 실험이 필요하다.
- **(b) 낮은 부하 slot p99:** 다섯 방식 모두 `6.56–7.26 ms` 범위다. 즉 요청 하나만 보면
  local placement와 cross placement의 차이가 작아 보인다. GDR만 depth=1, 나머지는 depth=2인
  보존 결과이므로 이 패널은 최종 공정 순위가 아니라 구현 비용의 범위를 보여준다.
- **(c) Background:** isolated placement들은 Qwen `10.22–10.24 it/s`, 선택한 Full MPS 점은
  `11.14 it/s`다. 완전히 같은 background utility는 아니지만 기존 MPS 30%나 100% 점보다
  가장 가까운 실측 비교다.
- **(d) Scaling:** 낮은 부하 E2E만으로는 보이지 않던 문제가 NRx process sweep에서 나타난다.
  Full A100 MPS는 `42.3→189.3 ms`(`4.5×`), 4g 내부 MPS는 `40.7→435.7 ms`(`10.7×`)로
  L1 p99가 증가했다.

따라서 그래프가 보여주는 결론은 다음과 같다. **낮은 부하에서는 다섯 방식의 E2E가
비슷하지만, co-tenant 수가 늘면 MPS 계열의 L1 보호가 무너진다. P2P는 이를 실제로 회복한
같은-GPU 경로이고, GDR는 작은 추가 E2E 비용으로 그 도달 범위를 다른 GPU/process까지
확장한다.** 다만 동일한 물리 GPU budget·Qwen 처리량·NRx burst를 동시에 맞춘 최종 five-way
승자 비교는 아직 남아 있다.



## 2. 첫 번째 오해: MIG isolation이 실패한 것이 아니다

동일한 A100의 4g MIG에서 resident NRx를 실행하고, sibling 3g MIG에 Qwen-7B를 올려
NRx closed-loop capacity와 open-loop tail을 측정했다.

![MIG 격리 효과와 NRx 처리 한계 이후의 대기시간 폭증](../figures/01_mig_isolation_queue_cliff.png)

그림 1은 “MIG가 효과가 없었다”는 해석과 “MIG면 모든 문제가 해결된다”는 해석이 모두
틀렸음을 보여준다.

- **MIG isolation은 정상 동작한다.** 4g NRx capacity는 단독 `745.1 request/s`, sibling
  3g Qwen 동시 실행 `744.2 request/s`로 차이는 `-0.11%`였다.
- **격리는 capacity를 늘리지 않는다.** 700 request/s에서는 p99가 약 1.39 ms였지만,
  750/s에서는 15.58–18.78 ms, 800/s에서는 214.29–217.79 ms로 급증했다.

쉽게 말해 옆방의 Qwen은 NRx 처리속도를 거의 떨어뜨리지 않았다. 하지만 한 명이 처리할 수
있는 창구에 초당 750~800명이 계속 오면, 방음이 잘돼 있어도 줄은 길어진다. 이전에 보인 큰
latency의 한 원인은 MIG 격리 실패가 아니라 **도착률이 고정된 NRx 처리율에 가까워지거나
넘은 queueing collapse**다.

## 3. 두 번째 오해: 같은 partition에 L1+NRx를 넣으면 baseline L1과 같아야 한다

MIG는 **서로 다른 GPU instance 사이**를 격리한다. 같은 4g instance 안에 cuPHY L1과 NRx를
함께 넣으면 둘 사이에는 벽이 없다. 두 stage는 제한된 SM, memory bandwidth, launch
resources와 GPU에 제출된 work queue를 계속 공유한다.
따라서 다음 두 metric을 구분해야 한다.

- **L1 active time:** NRx/background interference로부터 L1 kernel이 얼마나 보호되는가
- **Dependency-carrying slot E2E:** CE → NRx → LDPC/CRC 전체가 얼마나 걸리는가

Cross P2P는 L1 slowdown을 `1.621× → 1.043×`로 복구했지만, 4g를 2g L1 + 2g NRx로
나누면서 NRx compute가 느려져 slot E2E는 `6.191 → 6.383 ms`로 소폭 증가했다. 즉
**격리 효과는 L1 metric에서 분명하지만, 작은 NRx slice의 compute 손실 때문에 전체
slot latency가 자동으로 줄지는 않는다.**

이 결과는 “cross partition이 쓸모없다”가 아니다. 목적을 구분해야 한다.

- 같은 partition은 현재 slot 하나의 평균시간에는 유리할 수 있다. data transport가 없고
  더 큰 compute slice를 쓸 수 있기 때문이다.
- cross partition은 L1의 실행시간을 NRx queue와 분리하고, 다른 곳의 resident NRx를 사용할
  수 있게 한다. 이 이점은 단일 slot보다 여러 cell의 동시 도착과 tail에서 커진다.
- 따라서 공정한 질문은 “누가 단일 요청 하나가 가장 빠른가?”와 “같은 도착률에서 어느
  방식의 queue가 언제 무너지는가?”를 함께 보는 것이다.

### 3.1 과거 105 ms NRx 결과와 현재 1.34 ms 결과가 다른 이유

초기 실험의 약 105 ms NRx는 동일 TensorRT model의 순수 compute 시간이 아니었다. Aerial
public `pycuphy` wrapper가 수행한 generic layout conversion이 포함된 시간이다. Nsight와
caller-owned binding으로 경로를 분해한 뒤 동일 output contract를 직접 실행했다.

![NRx 실행 경로 최적화 전후의 처리시간](../figures/01b_nrx_wrapper_optimization.png)

- Public wrapper GPU mean: `105.15 ms`
- Caller-owned TensorRT binding: `1.413 ms`
- Direct binding + CUDA Graph: `1.340 ms`, host enqueue `2.50 us`
- Wrapper와 direct output의 `max_abs_difference = 0`

CUDA API profile도 같은 결론을 보였다. Wrapper capture에서는 `cudaEventSynchronize`가
15회에 총 `1.232 s`, `cudaStreamSynchronize`가 9회에 `103.95 ms`였다. Direct path에서는
각각 20회 `32.87 ms`, 10회 `1.89 ms`로 줄었다. Direct capture의 `cudaFree`에는 종료 시점의
393 ms cold cleanup 한 번이 포함되므로 steady inference 병목과 분리해야 한다. 즉 “kernel을
async로 launch했다”만으로 충분하지 않고, caller-owned persistent buffer와 synchronization
위치를 함께 바꿔야 했다.

따라서 초기 106–112 ms chain 결과는 “당시 wrapper를 포함한 실측”으로 보존하지만,
placement와 queue capacity 결론에는 optimized direct-TensorRT 결과를 사용한다. 이 교정 없이
과거 결과와 현재 결과를 한 표에 섞으면 transport와 MIG 효과를 잘못 해석하게 된다.

### 3.2 host blocking은 별도의 실제 병목이다

CUDA kernel launch는 보통 비동기다. CPU는 kernel을 GPU queue에 넣고 다음 일을 계속할 수
있다. 그러나 memory를 해제하거나 buffer를 다시 쓰거나 결과가 필요해지는 지점에서는 앞서
제출한 GPU 작업이 끝났는지 확인해야 한다. 이때 CPU thread가 `cudaFree`,
`cudaStreamSynchronize`, 심지어 다음 `cudaMemcpyAsync` 호출 안에서 오래 머물 수 있다. 이
현상을 이 문서에서는 **host blocking**이라고 부른다.

중요하게, 이는 `GPU → CPU DRAM → GPU` 데이터 복사와 다른 현상이다. CPU DRAM을 통과하지
않아도 같은 scheduling domain에 긴 NRx work가 쌓이면 L1 host thread의 CUDA API가 그 대기를
떠안을 수 있다.

| 배치 | host blocking 관점의 예상 경로 |
|---|---|
| Full MPS | L1, NRx, background가 같은 full-GPU scheduling domain을 사용하므로 AI queue가 L1 sync 지점에 드러날 수 있음 |
| MIG local | sibling 3g는 격리되지만 같은 4g 안의 L1–NRx outstanding work는 분리되지 않음 |
| MIG+MPS local | 두 MPS client와 quota를 사용해도 같은 4g의 물리 자원과 완료 조건을 공유; hard preemption bound 없음 |
| Cross P2P/GDR | L1 host CUDA queue는 remote NRx compute queue와 분리됨; 대신 명시적인 publish/completion/commit 대기가 생김 |

아래 Nsight 결과가 이 가설을 same-MIG co-location에서 직접 확인한다. 다만 현재 조건별
absolute-rate sweep의 모든 점에 Nsight를 붙인 것은 아니다. 현재 보유한 직접 증거는 `(a)` 같은 MIG 안의 L1+NRx
co-location causal experiment와 `(b)` 실제 cuPHY–GDR–NRx vertical slice다. Full MPS,
MIG local, proper MIG+MPS, P2P, GDR를 같은 trace로 각각 Nsight capture하는 paired matrix는
마지막으로 더 수행해야 할 microarchitectural 분석 실험이다.

#### 직접 Nsight 측정 결과

![같은 MIG의 NRx가 L1 CPU 실행을 막는 CUDA 호출 분석](../figures/03d_cuda_host_blocking.png)

same-MIG co-location에서 cuPHY가 실제 호출한 여섯 주요 CUDA runtime API의 누적 host 시간을
30초 Nsight window로 측정했다. 40-cell 조건에서 다음이 관측됐다.

| 조건 | 추적한 host CUDA API 총시간 | 대기가 주로 보인 API |
|---|---:|---|
| L1 alone | 1.681 s | `cudaFree` 1.361 s |
| L1 + NRx | **25.348 s** | `cudaFree` 18.076 s, `cudaMemcpyAsync` 7.034 s |
| `cudaFreeAsync` shim | 25.570 s | `cudaMemcpyAsync` **25.221 s** |
| stream-ordered memory pool | 25.649 s | `cudaMemcpyAsync` **25.539 s** |

Co-location은 추적한 host 시간을 `15.1×` 늘렸다. 그러나 `cudaFree`를 async API로 바꾸자
총 대기가 사라지지 않고 다음 copy/synchronization 지점으로 이동했다. 즉 병목은 함수 이름
`cudaFree` 하나가 아니라 **같은 scheduling domain에 쌓인 outstanding GPU work와 그 작업을
반드시 확인해야 하는 의존성 경계**다.

이것이 cross-partition 설계가 평균 transport 0.4 ms만으로 평가되면 안 되는 이유다. 별도
P2P/GDR endpoint는 L1의 CUDA queue와 NRx compute queue를 분리한다. 대신 request publish와
completion을 명시적으로 관리해야 하며, DART-Rx의 credit/expiry/commit 규칙이 바로 그 새
경계를 안전하게 다룬다. 다만 이 Chain-8 Nsight 수치는 five-way sweep 각 점의 per-slot
latency가 아니며 exact direct-TensorRT five-way configuration도 아니므로 두 숫자를 더해서는
안 된다.

현재 조건별 증거 범위를 정리하면 다음과 같다.

| 방식 | latency/rate sweep | CUDA-call 직접 증거 | 현재 판단 |
|---|---|---|---|
| Full MPS | cap sweep + absolute-rate 완료 | 동일 trace Nsight는 없음 | throughput은 강함; background 시 tail 원인 분해 필요 |
| MIG local | placement + absolute-rate 완료 | same-MIG co-location causal profile 보유 | sibling 격리는 강함; 내부 L1–NRx wait 존재 |
| MIG+MPS | two-client quota + absolute-rate 완료 | 동일 trace Nsight는 없음 | quota trade-off는 실측; wait 위치는 추가 capture 필요 |
| Cross P2P | placement + absolute-rate 완료 | 동일 trace Nsight는 없음 | L1 active isolation 복구; copy/sync 분해 필요 |
| Cross NIC GDR | placement + absolute-rate + radio 완료 | actual-radio GDR vertical-slice Nsight 보유 | GDR flush보다 local sync/conversion이 큼 |

따라서 “각 방식의 결과가 어떻게 달라지는가”는 이미 측정됐고, “각 방식에서 정확히 어느
CUDA call이 원인인가”는 MIG co-location과 GDR vertical slice만 직접 증명된 상태다. 나머지
세 조건의 paired Nsight는 결과를 보강하기 위한 필수 후속 실험이다.

#### 반드시 추가할 5-way paired host-blocking 실험

논문의 최종 원인 분석에는 **Full MPS, MIG local, MIG+MPS, Cross P2P, Cross NIC GDR를
동일한 Nsight 조건으로 다시 측정한 한 장의 5-way CUDA-call figure가 필요하다.** 기존
same-MIG 30초 capture와 actual-radio GDR capture를 한 막대그래프에 넣으면 안 된다. 실행
경로, cell 수, process 경계, capture 길이가 달라서 API 누적시간의 분모가 다르기 때문이다.

공정한 paired 실험은 다음 조건을 고정해야 한다.

| 고정 항목 | 조건 |
|---|---|
| L1/NRx 코드 | 같은 optimized direct-TensorRT, caller-owned buffer, CUDA Graph 경로 |
| 요청 | 같은 보존 arrival trace와 같은 request/result 크기 |
| 부하 | 공통 안정점 `180 requests/s`와 queue pressure가 드러나는 `250 requests/s`를 각각 측정 |
| Background | off/on을 paired로 두고, on에서는 Qwen throughput을 약 `10.2 it/s`로 맞춤 |
| 반복 | 각 조건 30초 × 3 trials; cold initialization과 shutdown은 steady window에서 제외 |
| 기준선 | full GPU, 4g L1, 2g L1 각각의 L1-alone capture를 따로 측정해 동일 geometry 기준으로 정규화 |
| Nsight 범위 | `cuda,nvtx,osrt`; L1 front, transport publish/wait, NRx, L1 back, commit NVTX range |

특히 총 API 시간 하나만 비교하면 또 오해가 생긴다. 최종 분석은 다음 세 패널로 보여줘야
한다.

1. **L1 coordinator의 host-blocked time/slot:** 주요 CUDA API 안에 머문 시간을 완료 slot
   수로 나누고, 같은 geometry의 L1-alone 대비 배율도 함께 표시한다.
2. **어느 API가 기다림을 떠안았는가:** `cudaFree`, `cudaMemcpyAsync`,
   `cudaStream/Event/DeviceSynchronize`, `cudaDeviceFlushGPUDirectRDMAWrites`, launch/other를
   stacked bar로 분해한다. 호출 횟수와 call-duration p50/p95/p99도 별도 CSV에 남긴다.
3. **기다림이 어느 pipeline 단계에 있었는가:** NVTX timestamp로 각 API를 L1 front,
   local NRx, P2P copy, GDR publish/completion, L1 back/commit에 귀속하고, L1 GPU idle gap과
   host API interval의 시간 overlap을 계산한다.

process 구조가 다른 것도 그대로 다뤄야 한다. MPS와 GDR는 L1 producer와 NRx consumer를
별도 process로 capture하고 **L1 coordinator 결과를 주 지표**로 사용한다. NRx-side API는
보조 패널로 분리한다. MIG local과 현재 P2P는 한 process 안에서 실행되므로 NVTX range와
thread/timestamp를 이용해 L1 phase와 NRx phase를 분리한다. 두 process의 누적시간을 단순히
더하면 안 된다.

이 실험에서 확인할 가설은 다음과 같지만, 결과가 나오기 전에는 결론으로 쓰지 않는다.

| 방식 | 검증할 host-blocking 가설 |
|---|---|
| Full MPS | co-tenant work가 L1의 sync/free/copy dependency 지점에 나타나는가? |
| MIG local | sibling Qwen은 격리돼도 같은 4g의 NRx outstanding work가 L1 API wait를 늘리는가? |
| MIG+MPS | client quota가 wait 총량을 줄이는가, 아니면 API 사이에서 위치만 이동시키는가? |
| Cross P2P | L1 API wait가 2g L1-alone에 가까워지고 명시적 peer-copy completion만 남는가? |
| Cross NIC GDR | remote NRx compute wait가 L1 CUDA queue에서 분리되고 GDR flush/completion 및 local conversion wait로 바뀌는가? |

따라서 현재 `03d_cuda_host_blocking`은 **host blocking이 실제 존재하며 API 이름을 바꿔도
대기가 이동한다는 causal evidence**이고, 아직 5-way 승자 그래프가 아니다. 최종 5-way
paired capture가 완료돼야 “P2P/GDR 분리가 L1 host thread의 어떤 blocking을 얼마나
제거했는가”를 같은 분모로 주장할 수 있다.

여기까지는 **NRx를 L1과 분리해야 하는 이유와 분리된 endpoint까지 가는 길**을 확인했다.
그러나 길이 있다고 여러 NRx가 자동으로 사용되는 것은 아니다. 다음 절은 단일 NRx의 queue
cliff가 실제 multi-endpoint 환경에서 `busy queue + idle endpoint`로 나타나는지를 확인해,
data-path 실험에서 scheduler 설계로 넘어가는 징검다리다.

## 4. 왜 NRx를 여러 개 중에서 골라야 하는가: 한 queue는 무너지는데 다른 NRx는 논다

앞의 실험들은 여기까지를 보여줬다.

1. **§2의 단일-endpoint 실험:** MIG가 Qwen을 제대로 격리해도 NRx 하나의 service rate는
   유한하다. 4g NRx는 약 `745 requests/s`를 처리했지만 arrival이 750–800/s로 올라가자 p99가
   `15.58–217.79 ms`로 폭증했다. 즉 queue cliff는 실제로 존재한다.
2. **§3의 배치/host 분석:** L1과 NRx를 같은 GI에 두면 L1까지 함께 느려질 수 있으므로,
   protected L1과 NRx compute queue를 분리하는 편이 낫다. P2P/GDR는 분리된 NRx에 도달할
   수 있는 data path를 제공한다.

하지만 이 두 결과만으로는 여러 NRx를 하나의 pool처럼 선택해야 한다는 결론이 아직 나오지
않는다. Remote NRx에 갈 수 있다는 사실과, 실제로 그 capacity를 빌릴 필요가 있다는 사실은
다르기 때문이다. 다음 질문이 하나 남는다.

> **실제 slot 요청이 시간에 따라 들어올 때, 현재 할당된 NRx queue는 deadline을 놓치는데
> 이미 켜져 있는 다른 NRx는 동시에 놀고 있는 순간이 존재하는가?**

이 질문에 `yes`여야 static MIG placement의 capacity fragmentation이 실제 problem이고,
request-level endpoint selection이 필요하다. 그래서 실험을 다음 순서로 한 단계씩 분리했다.

```text
[앞선 실험 A] NRx 하나의 service curve 측정
               → capacity를 넘으면 queue가 실제로 붕괴

[앞선 실험 B] P2P/GDR data-path 측정
               → 분리된 NRx GPU memory까지 도달 가능

[이 절의 실험 C] 같은 arrival trace를 세 NRx에 입력
               → 고정 binding일 때 busy queue + idle NRx가 동시에 생기는가?
               → 요청별로 다른 NRx를 고르면 그 현상이 줄어드는가?
```

따라서 이 절의 그림은 **전송 방식의 성능 그래프가 아니라, 실험 C의 compute/queue
problem-existence 실험**이다. 실제 TensorRT NRx는 서로 다른 세 physical GPU의 `3g.20gb
MIG`에 하나씩 상주시켰다. §2의 4g capacity 수치를 그대로 재사용한 것이 아니라, 이
실험에서는 각 3g endpoint를 약 `1.6 ms/request`로 별도 calibration했다.

동일한 deterministic input tensor는 각 endpoint GPU에 미리 올려 두었다. 그래야 P2P/GDR
속도 차이를 제거하고 **오직 static binding과 request routing이 queue에 만드는 차이**만 볼
수 있다. 따라서 이 실험에는 cuPHY front/back, P2P/GDR tensor 전송, conventional fallback
실행이 들어 있지 않다. 실제 full-size GDR request/result를 포함한 재검증은 뒤의 Stage 2에서
별도로 수행한다.

```text
slot/Nrx-required trace
        ↓
host scheduler가 endpoint 하나를 선택
        ↓
선택된 3g MIG에서 실제 TensorRT + CUDA Graph 실행

제외: L1 tensor 이동(P2P/GDR), cuPHY CE/LDPC, conventional fallback 실행
```

비교한 두 정책은 다음과 같다.

- **빨간색 `static-one`:** 모든 NRx 요청을 NRx 0에 고정한다. NRx 0의 queue가 길어져도
  NRx 1과 NRx 2로 보내지 않는다.
- **파란색 `predicted-finish`:** NRx 0/1/2 중 현재 예약된 작업이 가장 빨리 끝날 endpoint를
  실제로 선택한다. 세 endpoint 모두 5 ms 안에 끝낼 수 없다고 예측되면 queue에 더 넣지
  않고 `fallback required`로 기록한다. 따라서 파란색은 **분산 선택과 deadline admission을
  함께 적용한 결과**다.

![한 NRx 대기열은 밀리는데 다른 NRx 처리기는 노는 고정 배치 문제](../figures/02_fixed_placement_fragmentation.png)

그림의 세 패널은 같은 run을 서로 다른 질문으로 읽는다.

| 패널 | 쉬운 질문 | 정확한 분모와 의미 |
|---|---|---|
| **(a) p99 요청 지연** | 끝난 NRx 요청은 얼마나 늦었는가? | 실제 완료된 NRx 요청의 `도착 → queue 대기 → TensorRT 완료` 시간 중 느린 1% 경계. 파란색에서 deadline admission으로 사전 fallback된 요청은 이 latency 집합에 없으므로 (b)와 함께 봐야 한다. |
| **(b) 5 ms 안에 결과 없음** | 전체 NRx 요청 중 제시간에 쓸 결과를 못 받은 비율은? | 모든 NRx-required 요청이 분모다. 5 ms 뒤 완료, queue overflow, predicted-finish의 사전 fallback을 모두 실패로 센다. 두 정책의 usable-result coverage를 비교하는 더 포괄적인 지표다. |
| **(c) endpoint idle fraction** | 그동안 세 NRx 계산기는 얼마나 놀았는가? | `3 endpoints × trace 실행시간` 중 TensorRT를 실행하지 않은 시간의 비율이다. **(c)의 빨간색과 파란색 모두 idle 값**이며, 파란색이 endpoint 선택 비율을 뜻하지 않는다. |

대표 결과는 다음과 같다.

| workload | static-one: p99 / timely-result 실패율 | predicted-finish: p99 / timely-result 실패율 |
|---|---:|---:|
| 1 cell, 1 ms, NRx 100% | 3293.25 ms / 99.97% | 1.63 ms / 0.13% |
| 4 cells, 1 ms, bursty 10% | 51.39 ms / 64.85% | 5.50 ms / 1.61% |
| 4 cells, 0.5 ms, bursty 10% | 3332.84 ms / 99.90% | 5.06 ms / 7.89% |

첫 번째 workload에서는 NRx 한 개의 실측 service time이 약 `1.6 ms`, 즉 대략 `625
requests/s`인데 `1,000 requests/s`가 NRx 0 하나에 들어온다. 그래서 빨간색은 구조적으로
queue가 계속 증가한다. 그동안 NRx 1과 NRx 2에는 일을 보내지 않으므로 전체 endpoint-time의
약 `2/3`, 즉 `66.7%`가 idle이다. 파란색은 주로 두 endpoint를 사용해 같은 요청률을
처리하므로 p99가 `3293.25 → 1.63 ms`, timely-result 실패율이 `99.97 → 0.13%`로 줄었다.

가운데의 `4 cells · 1 ms · bursty 10%`는 더 미묘하고 중요하다. 평균 요청률이 낮아서 두
정책 모두 endpoint idle fraction이 약 `78–79%`로 높다. 그러나 짧은 순간에 요청이 몰릴
때 static-one은 NRx 0에만 burst를 쌓아 p99 `51.39 ms`를 만들고, predicted-finish는 같은
burst를 다른 NRx에 나눠 p99를 `5.50 ms`로 낮춘다. 즉 **평균 GPU utilization이 낮아도 한
queue의 tail latency는 나쁠 수 있다.** 이것이 시간적으로 capacity가 잘못된 queue 뒤에
갇히는 fragmentation이다.

세 번째 workload에서는 파란색도 timely-result 실패율 `7.89%`가 남는다. Routing으로 stranded
capacity를 사용해도 세 endpoint의 총 service capacity나 5 ms expiry 자체가 부족한 burst는
존재한다는 뜻이다. 따라서 결과는 “세 NRx면 모든 부하가 해결된다”가 아니라, **static
binding은 사용 가능한 capacity조차 낭비하고, routing 이후에도 admission과 fallback이
필요하다**는 것을 보여준다.

여기서 timely-result 실패는 5 ms expiry 안에 usable NRx가 없었다는 뜻이며, scheduler가
conventional fallback을 요구한 경우도 포함한다. 이 compute/queue 실험에는 actual PHY
fallback 결과가 없으므로 radio deadline miss와 같은 metric으로 읽으면 안 된다.

이 그림이 증명하는 것과 증명하지 않는 것도 구분해야 한다.

- **증명함:** 동일한 세 실제 NRx endpoint에서도 고정 binding 때문에 busy queue와 idle
  capacity가 동시에 존재하며, request-level routing/admission이 이를 크게 줄인다.
- **증명하지 않음:** P2P와 GDR 중 어느 것이 빠른지, actual L1 tensor를 옮긴 end-to-end
  성능, predicted-finish가 모든 trace에서 round-robin보다 항상 우수하다는 주장.
- **후속 연결:** P2P/GDR 실험은 선택한 remote endpoint까지 tensor를 옮길 수 있는지를
  측정하고, Stage 2 GDR-pool 실험은 full-size request/result를 포함해 routing 정책을 다시
  평가한다.

