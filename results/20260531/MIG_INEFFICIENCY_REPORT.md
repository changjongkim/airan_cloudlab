# MIG(Multi-Instance GPU) 비효율성 및 격리 한계 종합 분석 보고서

본 보고서는 CloudLab 환경(NVIDIA A100-SXM4)에서 수행된 4개의 상세 분석 문서(`EXPERIMENTS_20260531.md`, `NCU_ANALYSIS_20260531.md`, `NSYS_ANALYSIS_20260531.md`, `NSYS_DETAILED_ANALYSIS.md`)를 종합하여, 실시간 통신 워크로드(cuPHY L1)와 AI 워크로드를 동시 실행할 때 발생하는 **MIG의 구조적 비효율성 및 격리 실패 원인**을 진단합니다.

---

## 1. 개요 (Executive Summary)

MIG(Multi-Instance GPU)는 하드웨어 자원을 쪼개어 워크로드를 격리하는 기술이지만, 실시간(latency-critical) 시스템에서는 두 가지 근본적인 비효율성을 보입니다:
1. **단일 파티션 크기 한계에 따른 하드웨어 병목:** AI 코테넌트(Co-tenant)가 없는 단독 실행(alone) 상태에서도, 파티션이 작아질수록 메모리 I/O 병목이 기하급수적으로 증가합니다.
2. **공유 소프트웨어 스택 경합에 의한 격리 실패:** MIG는 메모리와 SM 등의 **하드웨어 실행 단위는 완벽하게 격리**하지만, **드라이버 레벨의 커널 런치 큐(Launch Queue) 및 메모리 컨트롤러는 전체 GPU가 공유**합니다. 이로 인해 불규칙한 메모리 패턴을 가진 AI 워크로드 동작 시 커널 간 대기 시간(Inter-kernel gap)이 폭증합니다.

---

## 2. 근본 원인 1: MIG 자체의 구조적 하드웨어 패널티

MIG를 통해 파티션을 작게 나눌수록, AI 간섭이 없는 상태에서도 근본적인 성능 저하가 발생합니다 (`NCU_ANALYSIS_20260531.md`).

* **MIO Throttle 폭증:** 7g 파티션에서 0.02%에 불과했던 MIO(Memory I/O) Throttle이 2g 파티션에서는 0.66%로 **33배 폭증**합니다.
* **L2 캐시 압박:** 동일한 cuPHY 작업임에도 2g 파티션에서는 L2 트래픽이 7g 대비 **30배 증가**(kernel당 12KB → 378KB)합니다. L2 캐시 슬라이스가 좁아져 워킹셋을 감당하지 못하고 병목이 발생합니다.
* **영향:** 결과적으로 파티션을 7g에서 2g로 줄이면, AI 워크로드가 개입하지 않아도 전체 실행 시간이 37ms에서 51ms로 **약 40% 증가**합니다.

---

## 3. 근본 원인 2: 비대칭적/불완전한 시스템 격리 (공유 자원 경합)

`ncu`(하드웨어 레벨 프로파일러)와 `nsys`(타임라인 프로파일러)의 데이터를 교차 검증한 결과, MIG 격리 실패의 정확한 메커니즘이 규명되었습니다 (`NSYS_DETAILED_ANALYSIS.md`, `NSYS_ANALYSIS_20260531.md`).

### 3.1. 커널 실행(Hardware)은 보호되나 대기 시간(Driver Queue)이 망가짐
* **하드웨어는 안전함:** `ncu` 지표(L2 hit rate, DRAM throughput 등)는 AI 코테넌트 추가 시 변동폭이 2% 미만입니다.
* **소프트웨어 큐의 붕괴:** NVIDIA 드라이버의 커널 런치 큐는 GPU 단위로 공유됩니다. AI 워크로드가 추가되면 `nsys` 타임라인 상에서 **커널과 커널 사이의 대기 시간(post-gap)**이 급증합니다.

### 3.2. 구체적인 병목 지점 및 지연 수치
* **`memcpy` 지연:** GPU 메모리 컨트롤러 큐가 밀리면서, 메모리 복사 작업 시간이 **최대 +317~400% 폭증**합니다.
* **Transition 지연:** cuPHY 파이프라인 중 `cupy_copy`(메모리 전송)에서 `convert_kernel`(포맷 변환)로 넘어가는 과정의 gap이 AI(3 AI 또는 NeuralRx) 개입 시 **+398~509% (약 5~6배) 증가**합니다.
* **영향:** 이러한 미세한 커널 간 지연들이 누적되어, 최종적으로 L1 통신 애플리케이션 레벨(Tier1)에서 **Wall-clock latency가 +33% (평균) ~ +377% (P99)** 증가하는 누수(Leakage)가 발생합니다.

---

## 4. 워크로드의 특성(Memory Pattern)에 따른 차별적 간섭

MIG의 간섭은 코테넌트로 실행되는 AI 워크로드의 성격에 따라 극명하게 갈립니다.

* **Chaotic 패턴 (Qwen LLM, NeuralRx, LSTM):** 
  * 토큰마다 메모리 접근이 달라지거나 잦은 커널 런치가 발생합니다.
  * 드라이버 큐를 독점하려 하여 L1의 p99 latency를 폭증(+69% ~ +377%)시킵니다.
* **Uniform 패턴 (ResNet, sat_compute 등):** 
  * 단조로운 GEMM이나 예측 가능한 메모리 복사를 수행합니다.
  * 드라이버 큐에 큰 충격을 주지 않아 L1에 미치는 영향이 거의 없습니다(0%).
* **시사점:** 변수는 단순히 'GPU 연산량'이 아니라 **'Memory Access Pattern의 변동성(Volatility)'**입니다.

---

## 5. 기존 가설의 수정: 측정 지표(Throughput vs Latency)의 착시

초기 분석에서는 "AI 워크로드는 격리가 잘 되고, 실시간 통신(L1) 커널만 피해를 본다"고 생각했으나, 이는 **측정 방식이 만든 착시**임이 밝혀졌습니다 (`EXPERIMENTS_20260531.md`).

* **AI Throughput (초당 처리량):** 30초 단위로 적분된 처리량을 측정하므로, 간헐적인 드라이버 큐 간섭이 평균치에 묻혀 0.1~1%의 미미한 변화로 보였습니다.
* **Per-op Latency (개별 작업의 꼬리 지연):** AI 워크로드 역시 L1과 동일하게 **per-operation 단위의 p99 latency를 측정해 본 결과, 최대 +27%(chanpred)까지 지연**되는 현상이 확인되었습니다.
* **최종 결론:** MIG는 하드웨어 연산(Throughput/Mean)은 성공적으로 격리하지만, 양방향 모두 **꼬리 지연시간(Tail Latency, p99) 격리에는 실패**합니다. 

---

## 6. AI-RAN 환경 운영 시사점 및 대책 방안

본 분석을 기반으로 실시간 통신(L1)과 AI를 동시 운영(AI-RAN)할 때의 가이드라인을 도출할 수 있습니다.

1. **파티션 크기 꼼수 불가:** L1을 보호하기 위해 L1용 파티션을 3g에서 4g로 늘려도 간섭은 막지 못합니다. (하드웨어 용량 부족이 아니라 큐 경합의 문제이기 때문)
2. **동거(Co-tenant) 워크로드 선별:** L1과 같은 GPU에는 예측 가능하고 단조로운(Uniform) AI 모델(예: ResNet 기반 비전 워크로드)만 배치해야 합니다. LLM(Qwen)이나 통신 유사 신경망(NeuralRx) 배치는 치명적입니다.
3. **근본적 대책:** 완벽한 실시간성을 보장하려면 p99 latency에 민감한 L1은 물리적으로 분리된 별도의 GPU에 할당하는 것이 가장 안전합니다.

---

## 7. **NSYS v3 확장 분석 (10개 추가 시나리오)** — 5/31 23:30 추가

기존 v2 16개 + 새 10 시나리오 = **총 26 scenarios × 3 runs = 78 nsys traces**.

### 7.1 추가된 v3 시나리오

| ID | 설명 | MIG 레이아웃 |
|---|---|---|
| S27 | 3g L1 + chanpred (on 2g) | 3g+2g |
| S28 | 3g L1 + ResNet fp16 (on 2g) | 3g+2g |
| S29 | 3g L1 + Forecaster (on 2g) | 3g+2g |
| S30 | 3g L1 + xapp (on 2g) | 3g+2g |
| S31 | 3g L1 + **ResNet + chanpred** (on 2g+2g, M5c equiv) | 3g+2g+2g |
| S32 | 3g L1 + **ResNet + Forecaster** (on 2g+2g, M8a equiv) | 3g+2g+2g |
| S33 | 4g L1 + chanpred (on 2g) | 4g+2g+1g |
| S34 | 4g L1 + ResNet (on 2g) | 4g+2g+1g |
| S35 | **2g L1 + chanpred (on 3g)** | 2g+3g |
| S36 | 4g L1 + Forecaster (on 2g) | 4g+2g+1g |

### 7.2 Steady-state p99 inflation (W3-W9 평균, vs S5 alone 843us)

| Scenario | Steady p99 | **vs S5** | 주요 발견 |
|---|---|---|---|
| S5 alone | 843 us | — | baseline |
| S27 chanpred | 913 | +8% | 2g placement = 작은 효과 |
| **S28 ResNet** | **1064** | **+26%** ⭐ | **ResNet도 단독 시 L1 p99 disturb** |
| **S29 Forecaster** | 1000 | **+19%** | sparse attention도 영향 있음 |
| S30 xapp | 932 | +11% | autoencoder, 작은 효과 |
| **S31 ResNet+chanpred (M5c)** | 872 | **+3.4%** ⭐ | **multi-AI smoothing!** |
| **S32 ResNet+Forecaster (M8a)** | 902 | **+7%** | multi-AI smoothing |
| S33 4g + chanpred | 905 | +7% | 4g L1 robust |
| **S34 4g + ResNet** | 833 | **-1.3%** ⭐ | **4g + ResNet = NEAR-ZERO** |
| **S35 2g + chanpred** | **1402** | **+66%** ⭐⭐⭐ | **catastrophe (worst case)** |
| S36 4g + Forecaster | 866 | +3% | 4g robust |

### 7.3 **5/31 기존 가설의 추가 수정 사항**

#### 수정 A: **ResNet도 L1 p99을 inflate한다** (이전 framing 보완)
- 기존 (Stage 4): "ResNet은 격리됨 (AI side p99 0.2% only)"
- **새 (NSYS v3 S28)**: 단독 ResNet on 2g → **L1 p99 +26%**
- → mean (throughput)에는 안 잡히지만 **L1 tail은 ResNet도 disturb**
- → "burst-mode p99 leak"이 모든 AI workload에 universal한 패턴

#### 수정 B: **Multi-AI heterogeneous는 단일 AI보다 누설 작음** (smoothing 효과)
- S31 (ResNet+chanpred, M5c equiv): **+3.4%** (vs 단독 ResNet +26%)
- S32 (ResNet+Forecaster, M8a equiv): **+7%** (vs 단독 Forecaster +19%)
- → **이전 가설 "Multi-AI = 더 큰 disturbance" 틀림**
- → 새 메커니즘: 두 AI가 서로 driver queue traffic을 평준화 (chaos canceling)
- → Stage 2 M5c/M8a wall-clock 0% 결과 메커니즘 확인

#### 수정 C: **2g L1 + chanpred = catastrophic** (가장 큰 누설)
- S35: 2g L1 + chanpred (on 3g AI partition) → **+66% steady p99**
- 변수 결합: 작은 L1 (MIG mio_throttle 33x baseline) + chaotic LSTM AI + 큰 AI partition (3g)
- **Paper의 worst-case figure 후보**

#### 수정 D: **4g L1 universal robust**
- 4g L1 + ResNet (S34): **-1.3%** (오히려 약간 빠름)
- 4g L1 + Forecaster (S36): +3%
- 4g L1 + chanpred (S33): +7%
- → "4g L1을 쓰면 어떤 AI에도 robust" — paper의 mitigation strategy
- → 단, Tier1 Phase2 M3 (4g L1 + multi-AI) +38% wall-clock과 다름 (steady p99 vs wall-clock mean 차이)

---

## 8. **NSYS Advanced Analyses** (SQLite 추가 분석)

### 8.1 Long-tail dominance

| Scenario | Top 1% gap | Top 0.1% gap |
|---|---|---|
| **모든 scenario** | **45-50%** | **31-41%** |

→ **누설 idle의 절반은 top 1% burst events**  
→ "tail-bounded disturbance" — mean이 아니라 burst가 dominant

### 8.2 Burst events 독립성 (auto-correlation)

| 모든 scenario lag-1 corr | < ±0.002 |

→ **gap_i와 gap_{i+1}이 완전 무상관**  
→ Burst events 독립적, cascading 없음  
→ **URLLC SLA 절대 보장 불가** (predictable 패턴 없음)

### 8.3 Time-series (10 windows)

Steady-state (W3-W9) median gap:
- 모든 scenario: **1.18 us → 1.22 us** (변화 거의 없음)

Steady-state p99 gap:
- S5 alone: 770-810 us 안정
- **S6 Qwen / S7 NRx / S9 3 AI**: 간헐적 spike (964-2681 us)
- S13 sat: alone과 동일

→ **typical kernel은 격리됨, worst-case 만 무작위 burst** ⭐

### 8.4 Distribution shape (long-tailed)

| Scenario | p75/p25 | p90/p10 |
|---|---|---|
| 모든 scenario | 15-26x | **130-138x** |

→ 정규분포 (1.3x) 대비 극히 long-tailed  
→ **Power-law / heavy-tail distribution**  
→ Paper claim: "mean 측정 무의미, p99 sentinel 필수"

### 8.5 CUDA Runtime API (driver-level overhead)

| Scenario | API calls | p99 (us) |
|---|---|---|
| 7g alone (S2) | 108999 | 521 |
| 3g alone (S5) | 108999 | 524 |
| **2g alone (S10)** | 108999 | **1115** |
| 4g + sat (S15) | 108999 | 621 |

→ **109K kernel launch API calls per run**  
→ **2g L1의 API p99 = 4g L1의 2배** (1115 vs 520)  
→ Partition-size dependent driver overhead = mio_throttle 결과와 일관

---

## 9. **Memory operations 분석**

| Scenario | memcpy total (ms) | memcpy median (us) | **memcpy p99 (us)** |
|---|---|---|---|
| S5 alone | 140 | 4.19 | 33 |
| S6 +Qwen | 394 (+181%) | 10.24 (+144%) | **165 (+400%)** |
| **S7 +NRx** | **585 (+317%)** ⭐ | **14.30 (+241%)** ⭐ | **171 (+418%)** ⭐ |
| S13 +sat | 272 (+94%) | 6.78 (+61%) | 164 (+398%) |

→ **memcpy 자체가 chaotic AI 추가 시 3.4x 느려짐**  
→ GPU memory controller queue 압박 직접 증거

---

## 10. **Kernel-pair transition vulnerabilities**

가장 vulnerable한 transition (vs S5 alone p99):

| From → To | Count | S6 Qwen | S7 NRx | S9 3 AI | S13 sat |
|---|---|---|---|---|---|
| `cupy_copy → convert_kernel` | 3840 | +59% | **+399%** | **+499%** | -25% |
| `convert_kernel → noiseIntfEst` | 1920 | +26% | **+292%** | **+509%** | +2% |
| `convert_kernel → windowedCh` | 1920 | +23% | **+134%** | **+137%** | +15% |

→ **memory-heavy transitions (memcpy ↔ format convert ↔ cuPHY compute) 가장 vulnerable**  
→ uniform AI (sat_compute) 추가 시 변화 거의 없음 (격리 작동)

---

## 11. **종합 메커니즘 (3-layer measurement evidence)**

```
Layer 1 (NCU per-kernel):
  - L2 hit rate <5% change
  - DRAM throughput <6%
  - mio_throttle: AI 영향 거의 없음 (partition size에는 33x 영향)
  → 하드웨어 capacity isolation 작동
  
Layer 2 (NSYS inter-kernel):
  - kernel median gap: 1.18 → 1.22 us (격리)
  - kernel p99 gap: +20~509% (chaotic AI)
  - memcpy time: +180~317%
  - specific transitions (cupy_copy→convert): +399~499%
  → driver/runtime/memory pipeline 누설
  
Layer 3 (Tier1 wall-clock):
  - L1 mean: +30~50%
  - L1 p99: +69~377%
  - AI per-op p99: +9~27% (chanpred), +0.4% (ResNet)
  → production 누적 결과
```

---

## 12. **AI workload별 universal pattern** (모든 측정 종합)

| AI 종류 | Memory pattern | L1 wall-clock Δ | NSYS steady p99 Δ | Stage 4 per-op p99 Δ |
|---|---|---|---|---|
| **Qwen-7B** | Chaotic (KV cache) | +33% mean, +69% p99 | +20% | +5-7% |
| **NeuralRx** | Chaotic (cuPHY-like) | **+50% mean, +377% p99** ⭐ | +52% | +13% |
| **chanpred** | Chaotic (LSTM) | +34% mean, +72% p99 | +8-66% (partition-dep) | +20-27% |
| **xapp** | Mixed | +32% mean, +67% p99 | +11% | (안 측정) |
| **ResNet (fp16)** | Uniform (Tensor Core) | 0% (Stage 2) | **+26% single, +3.4% multi** | +0.2-0.4% |
| **Forecaster** | Uniform-ish (sparse attn) | (안 측정) | +19% single, +7% multi | (안 측정) |
| **sat_compute** | Uniform (GEMM) | 0% | +1% | (안 측정) |
| **sat_hbm** | Uniform (memcpy) | 0~+3% | +2% | (안 측정) |
| **Multi-AI het (M5c/M8a)** | Mixed | 0% | **+3-7%** ⭐ (smoothing) | — |

---

## 13. **최종 메시지** (paper용)

> **"NVIDIA A100 MIG는 mean throughput/latency 격리는 양호하지만 p99 tail latency 격리는 부족하다. 누설은 burst-mode (top 1% events가 전체 idle 45-50% 차지), 독립적 (auto-correlation ~0), workload-pattern-dependent (chaotic AI > uniform AI), partition-size-dependent (2g L1 catastrophic, 4g L1 robust), 그리고 AI count/mix의 비선형 함수 (multi-AI het = smoothing).  
>   
> 메커니즘: cross-partition driver kernel launch queue + GPU memory controller queue의 burst-mode contention. NCU per-kernel은 변화 <2%이지만 NSYS inter-kernel post-gap이 +20~509% 폭증.  
>   
> 가장 큰 누설: 2g L1 + chanpred on 3g = +66% steady p99.  
> 가장 robust: 4g L1 + ResNet = -1.3%.  
> Multi-AI heterogeneous (M5c/M8a): chaos canceling = +3-7%만.  
>   
> AI-RAN production deployment 시 sub-ms slot deadline (URLLC SLA) 보장 불가능. 4g L1 + uniform-pattern AI 권장, NeuralRx 등 chaotic AI는 별도 GPU 분리 필요."**
