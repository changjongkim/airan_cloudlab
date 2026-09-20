# SoftWall 연구 계획: 하드웨어 파티션 없이 AI-RAN GPU에 deadline 계약을 부여하기

**작성:** 2026-09-20 KST
**상태:** 계획 초안. 스킴은 확정 전이며, P0/P1 게이트 결과에 따라 수정한다.
**목표 수준:** SIGMETRICS / NSDI
**작업 코드네임:** SoftWall (MIG의 하드웨어 벽을 소프트웨어 계약으로 대체) — 잠정
**선행 문서:** [`RESEARCH_DIRECTION_MPS_KO.md`](RESEARCH_DIRECTION_MPS_KO.md), [`MIG_NRX_DART_RESEARCH_SYNTHESIS_KO.md`](MIG_NRX_DART_RESEARCH_SYNTHESIS_KO.md),
[`../../results/20260803/MIG_MPS_COMBINED_REPORT.md`](../../results/20260803/MIG_MPS_COMBINED_REPORT.md),
[`../../results/visual_evidence/PERLMUTTER_NOMIG_VISUAL_EVIDENCE_KR.md`](../../results/visual_evidence/PERLMUTTER_NOMIG_VISUAL_EVIDENCE_KR.md)

---

## 0. 이 계획의 한 문장

> **MIG를 쓸 수 없는 AI-RAN 배포 환경에서, 하드 실시간 L1과 deadline이 걸린 optional NRx와
> best-effort LLM을 하나의 GPU 노드에 함께 올리면서, 하드웨어 파티션 없이 소프트웨어만으로
> deadline 계약을 성립시킨다.**

---

## 1. 전제: 앞선 결론 중 무엇을 가져가고 무엇을 버리는가

### 1.1 가져가는 것 (측정으로 확정된 것)

| 사실 | 수치 | 출처 |
|---|---|---|
| GPU 전용 배치는 간섭이 0이다 | L1 p99 35.9–40.2ms (AI N=1/4/6/8), baseline 38.5ms | chain19 Exp 12 (L1 GPU0 + AI GPU1) |
| 공유 배치는 최적 튜닝으로도 못 막는다 | pct=30에서 N=4 72.9ms / N=6 145.9ms / N=8 287.3ms | chain19 Exp 11 실측 |
| 기본값 공유는 파국이다 | pct=100, N=6에서 411.3ms (10.7×) | chain19 Exp 11 |
| MIG CP+MPS는 N=16까지 평탄하다 | 41.8–43.8ms | chain19 Exp 5 |
| 독립 client 증가가 L1을 무너뜨린다 | N=1→8에서 L1 p99 42.3→189.3ms, kernel gap 1.15→379μs, duty 31.6→13.8% | five-way 원인 분석 |
| 간섭 경로는 두 개다 | compute 경합=gap만 팽창 / memory 경합=gap+per-op bimodal | Perlmutter §10.5, §10.7 |
| 메커니즘은 4레벨로 닫혀 있다 | GPU gap ↔ cudaFree 블로킹(겹침 85%) ↔ convert 경계 ↔ ioctl | Perlmutter §10.5–10.9 |
| 메모리 대역폭 경합에서 MPS는 더 위험하다 | sat_hbm: default 426ms → MPS 6985ms (16.4×), bistable | Perlmutter §8 |
| NRx 처리율 | full A100 1164.1 req/s (direct TRT + CUDA Graph, 서비스 1.34ms) | NRX_CAPACITY.csv |
| 독립 endpoint는 용량이 합쳐진다 | 3 replica round-robin 97.2% timely (1/2 replica는 붕괴) | GDR pool replica sweep |
| bounded lease는 작동한다 | background 89–99% 보존하며 p99 2211→6.7ms | 4종 background gating |
| NRx는 채널 선택적으로만 가치가 있다 | utility admission 75/100 호출로 all-NRx와 동일한 correct-TB 0.80 | actual radio paired test |
| CUDA IPC + 대용량 전송은 검증됐다 | 1,415,232B 양방향 10,000회 무오류, 왕복 평균 1.406ms | gdr_cuda_ipc_gate |
| P2P 전송 비용은 작다 | 1.4MB 왕복 76.84μs, GDR는 P2P 대비 +0.438ms | placement/depth-1 비교 |

### 1.2 버리는 것

- **loopback NIC GDR을 노드 내 경로로 쓰는 설계.** 이건 성능 선택이 아니라 MIG 제약(다른 물리 GPU의
  MIG 간 P2P 불가, GI 간 CUDA IPC 불가)을 우회하려던 것이다. MIG를 안 쓰면 NVLink P2P와 CUDA IPC가
  모두 열리므로 노드 내에서 NIC를 거칠 이유가 없다. NIC/libfabric은 노드 간 확장에만 남긴다.
- **"MPS knob을 더 돌리면 된다"는 가설.** pct 10–100% 10단계(chain18 Part 4)와 pct×N(chain19 Exp 11)이
  이미 끝났고, 최선이 baseline의 1.9–3.8배다. 정적 knob 공간은 탐색이 종료됐다.
- **duty cycle을 SLA 지표로 쓰는 것.** Full GPU MPS는 duty 62%로 건강해 보이지만 p99는 63.6ms다.

### 1.3 정정해야 할 것 (작업 착수 전)

1. `MIG_MPS_COMBINED_REPORT.md`의 "SP + MPS pct=30 = 45ms (budget fallback)"은 측정값이 아니라
   [`analyze_mig_mps_combined.py:198`](../../results/20260803/analysis_chain19/analyze_mig_mps_combined.py)에
   하드코딩된 상수다. 같은 파일 L398 캡션과 L534 캡션("5-10% penalty")도 동일 오류다.
   실측은 N=6에서 145.9ms(3.79×)이며, F11 히트맵 자체는 실데이터를 그리고 있어 본문과 그림이 어긋나 있다.
   → **prose를 실측으로 교체하고, "tuned SP를 fallback으로 써도 된다"는 권고를 철회한다.**
2. `RESEARCH_DIRECTION_MPS_KO.md` §4의 "Perlmutter MPS daemon 실행 가능 여부 확인 필요"는 이미 해결됐다
   (2026-06-05, `run_F_nomig_mps.sbatch`, 계정 m1248_g, shifter 내부에서 `nvidia-cuda-mps-control -d`, 7조건 측정 완료).
3. 같은 문서가 근거로 인용한 `task1_final/gdr_pool_20260814T014651Z/`와
   `results/isca_v2/mig_causal_20260813T1138Z/`가 이 레포에 없다. 레포를 단일 기준으로 삼기로 한 이상
   가져오거나 경로를 정정한다.
4. 8/3 리포트("MIG CP + MPS만이 유일한 해답")와 9/19 방향("MIG는 제약, MPS 위에서 최적화")의 충돌을
   명시적으로 해소한다. 본 계획의 입장은 §2.3이다.

---

## 2. 문제 정의

### 2.1 대상 시스템

하나의 uplink request는 독립 추론 질의가 아니라 PHY dependency chain의 일부다.

```text
(cell, slot, scheduled PUSCH)
  → cuPHY channel estimation / front-end          [필수, 하드 deadline]
  → {conventional receiver | Neural Receiver}      [NRx는 optional, 채널 의존적 가치]
  → LDPC / CRC
  → absolute expiry 이전에 결과 하나만 commit
```

같은 GPU 노드에 동시에 존재하는 세 부류:

| 클래스 | 계약 | 특성 |
|---|---|---|
| **L1 (cuPHY)** | hard — 매 slot, 지연은 곧 프레임 손실 | 주기적, duty 31.6%로 낮지만 tail에 극도로 민감 |
| **NRx** | soft deadline + optional + utility | waterfall SNR 구간에서만 이득. expiry 후 결과는 **무효** |
| **background AI (LLM 등)** | best-effort, 취소 가능 | GPU 투자 회수의 근거. peak를 위해 비워둘 수 없음 |

### 2.2 왜 어려운가 (논문의 난이도 논거)

**(D1) 세 클래스의 계약이 서로 비교 불가능하다.**
고전 실시간 스케줄링은 preemption과 WCET를 가정하지만 GPU는 둘 다 주지 않는다. L1은 선점할 수 없고,
NRx는 "해도 되고 안 해도 되는데 하면 가치가 있는" 작업이며, LLM은 언제든 뺏어도 되지만 뺏는 데
시간이 걸린다. 단일 utility 함수로 환원되지 않는다.

**(D2) 조절 가능한 채널과 실제 간섭 채널이 다르다.**
MPS가 노출하는 유일한 knob은 SM 몫(active thread %)인데, 측정된 실패 경로는 (a) launch queue와
host sync 결합, (b) HBM 대역폭 고갈이다. **둘 다 pct로 규제되지 않는다.** 이게 정적 튜닝이 실패하는
구조적 이유이고, chain18/19가 이미 실험적으로 증명했다.

**(D3) optional 작업이 부채를 만든다.**
NRx 요청을 받아들이는 것은 용량을 쓰는 동시에 **conventional recovery 경로를 살려둘 의무**를 만든다.
그 recovery 용량은 LLM에게 팔 수 없다. 즉 admission은 단순 수락/거절이 아니라 *reservation을 동반한
2자원 결정*이다.

**(D4) 시간 규모가 어긋난다.**
수요는 multi-cell slot 동기화 때문에 ms 단위로 몰리는데, 재구성 primitive(MIG 재구성, MPS client
재시작, 모델 재적재)는 초 단위다. 평균 기반 provisioning은 무의미하고, 제어는 ms에서 닫혀야 한다.

**(D5) 확장하면 correctness 문제로 바뀐다.**
NRx 용량을 늘리려면 GPU를 건너야 하고, 그 순간 "늦은 결과"는 느린 게 아니라 **틀린 것**이 된다.
expiry 이후 도착한 LLR을 commit하면 PHY 상태가 오염된다. 이건 SLO 위반이 아니라 정확성 버그다.
GPU 공유 문헌 대부분이 다루지 않는 축이다.

**(D6) 예측 불가능성 자체가 실패다.**
sat_hbm MPS는 5회 중 2회만 붕괴하는 bistable이다(p50 6820ms). 운영자에게 "평균은 좋은데 가끔 7초"는
사용 불가다. 따라서 목표 함수에 run-to-run 분산이 포함되어야 한다.

### 2.3 8/3 리포트와의 관계 (충돌 해소)

8/3의 결론("MIG CP + MPS가 유일한 production 해답")은 **MIG를 쓸 수 있는 환경에서는 유효**하다.
본 연구는 그 결론과 싸우지 않고 **전제를 바꾼다**.

> MIG를 쓸 수 없는 배포가 존재한다: HPC 시스템은 일반 사용자에게 MIG를 노출하지 않고(Perlmutter),
> 멀티테넌트 클라우드는 런타임에 파티션을 재구성할 수 없으며, MIG 자체가 L1–NRx 간 P2P/IPC를
> 금지해 NRx pool 구성을 막는다.

따라서 연구 질문은 다음과 같다.

> **하드웨어 파티션 없이, 소프트웨어 제어만으로 MIG cross-partition 수준의 격리에 얼마나 근접할 수 있는가.
> 그리고 그 대가로 회수되는 유휴 용량은 얼마인가.**

목표 수치는 이미 데이터에 있다: **현재 최선 145.9ms → 목표 42ms(MIG CP 수준)**, baseline 38.5ms.

---

## 3. 배포 환경 요구사항

### 3.1 배포 형태

| 티어 | GPU 구성 | 본 연구의 대응 |
|---|---|---|
| Cell site | 1 GPU | L1 전용 불가 → 공유 필수. 가장 어려운 케이스 (P5 시나리오 C) |
| **Edge DC** | **4 GPU 1노드 (NVLink)** | **주 타깃.** CloudLab d8545, Perlmutter GPU 노드 모두 이 형태 |
| Regional / HPC | 다중 노드 | 확장(Direction 2), Slingshot/libfabric |

### 3.2 운영자가 보는 지표 (평가 지표를 여기서 역산한다)

1. **deadline miss ratio** — TTI(0.5ms) 기준, 0에 수렴해야 함. p99가 아니라 p99.9와 최댓값.
2. **cells per node** — TCO 직결. "이 노드로 몇 셀을 서비스할 수 있나".
3. **reclaimed background work** — 같은 쇠붙이로 얼마나 많은 AI 수익 작업을 돌리는가 (tokens/s, it/s).
4. **predictability** — run-to-run 분산, bistability 발생 확률. §2.2 D6.
5. **fault/upgrade 내성** — AI 프로세스 crash, rolling restart가 L1에 전파되는가.
6. **trust boundary** — 협조하지 않는 third-party AI 테넌트에서도 성립하는가.

### 3.3 워크로드 현실성

- L1: cuPHY 25.3, 273 PRB, 4-antenna, multi-cell. TTI 0.5ms (μ=1, 30kHz SCS).
- NRx: 실제 pyAerial neural receiver (ONNX→TensorRT), 요청 1,415,232B / 응답 314,496B.
- background: Qwen 2.5-3B, Whisper large-v3, BERT-large, Qwen2-VL, ResNet-50, CsiNet/BeamPred 등
  RAN-native AI. 이미 전부 구동 스크립트 보유.
- 채널: Sionna 기반 Es/N0 sweep(−4.0 ~ −3.2 dB), Markov burst.

---

## 4. 스킴: SoftWall

### 4.1 배치 평면 (deployment 시 결정, fast path에서 불변)

```text
GPU0 ── L1 전용 (cuPHY)                      MPS 미사용 (단일 client)
  │      · MPS 서버를 fault domain에서 제거
  │      · 유휴 ~68%의 회수는 "회수 사다리"로 별도 판정 (P4-E1)
  │
  │  NVLink P2P + CUDA IPC · persistent ring · NRx측 pull
  ▼
GPU1 ── NRx service 1개 (상주 TRT + CUDA Graph + 내부 큐/배치) + background lease
GPU2 ── 〃                                    MPS on (client 수 고정)
GPU3 ── 〃
```

세 가지 설계 규칙과 각각의 측정 근거:

| 규칙 | 근거 |
|---|---|
| **R1. L1은 GPU를 독점한다** | Exp 12에서 전용 배치 시 AI 부하와 무관하게 35.9–40.2ms 평탄 |
| **R2. NRx 확장은 GPU를 늘려서 하고, GPU 안에서는 client를 늘리지 않는다** | client N=1→8에서 4.5× 붕괴(gap 1.15→379μs) vs 독립 endpoint 3개는 용량 합산 성공 |
| **R3. peer copy는 NRx측에서 당긴다(pull)** | L1은 측정당 H2D memcpy 12,618회를 하고 gap이 convert/copy 경계에 국소화됨. GPU0의 copy engine과 launch queue를 비워둬야 함 |

### 4.2 데이터 평면

- GPU0 메모리에 endpoint별 persistent request/result ring을 두고 CUDA IPC로 NRx service에 매핑.
- descriptor + doorbell, epoch 태그. NRx service가 peer copy로 pull → 계산 → 결과 push.
- 노드 내 NIC 경유 없음. 노드 간은 별도(§7 P6).
- **미확정:** fp16 직결 가능 여부. L1 kernel 시간의 87–90%가 `convert_kernel<__half2,float2>`이고
  gap이 그 경계에 붙는다. NRx가 fp16을 그대로 받으면 전송량 절반 + convert 1회 제거가 동시에 가능하다.
  이 convert가 wrapper 산물인지 cuPHY 구조상 필수인지 P1에서 판정한다.

### 4.3 제어 평면 — 세 개의 루프

| 루프 | 주기 | 입력 | 출력 |
|---|---|---|---|
| **L1. Admission** | 요청당 (μs) | utility(채널 추정), endpoint feasibility | NRx 호출 / conventional |
| **L2. Placement** | 요청당 (μs) | 각 GPU의 queue tail + **co-tenant 상태** + 전송 비용 | 목적지 endpoint |
| **L3. Lease** | burst당 (ms) | NRx headroom, L1 slack | background work unit 발급/회수 |

**L2가 기존 대비 새로운 지점:** 현재 `predicted_finish`는 큐 길이 기반 추정이다. SoftWall에서는
목적지 GPU의 **co-tenant 상태(어떤 클래스가 얼마나 올라와 있고 최근 gap 분포가 어떤가)**를 추정에
넣어야 한다. 이게 §5의 C2(모델)이고, 논문의 정량적 중심이다.

### 4.4 정확성 계약

- commit 조건: slot ID + epoch 일치, payload visibility 보장, completion OK, absolute expiry 이전,
  LDPC/CRC 통과, transaction이 아직 open.
- conventional과 NRx 중 먼저 유효하게 commit한 하나만 공개. late/stale/duplicate는 private buffer에
  도착해도 다음 slot 상태를 바꾸지 못한다.
- endpoint health epoch로 재시작/crash 후의 stale completion을 걸러낸다.

### 4.5 벽이 없는 상태에서 보호가 성립하는 논리

| MIG가 주던 것 | SoftWall의 대체물 | 강도 |
|---|---|---|
| L1 파티션에 아무도 못 들어옴 | GPU0 전용화 + 배치 강제 | 협조적 환경에서 동등, 비협조 테넌트에는 약함 |
| 파티션별 고정 용량 | endpoint pool + admission | 더 유연 (fragmentation 해소) |
| 하드웨어 fault 격리 | 프로세스/GPU 분리 + health epoch | crash 전파는 막으나 driver-level 장애는 미보장 |
| 무제한 점유 방지 | bounded work unit + 회수 가능성 | **계약이지 강제가 아님** → 정직한 한계, 그리고 하드웨어 제안의 출발점 |

마지막 행이 논문의 솔직한 한계이자 "이걸 강제하려면 최소 어떤 하드웨어 지원이 필요한가"라는
후속 질문의 진입점이다.

---

## 5. 기여 구조 (탑 컨퍼런스 관점)

| # | 기여 | 성격 | 주 타깃 |
|---|---|---|---|
| **C1** | GPU가 실제로 노출하는 두 간섭 채널(launch-queue/host-sync 결합, HBM 대역폭 고갈)의 특성화와, 유일한 knob이 둘 다 규제하지 못한다는 증명 | 측정 연구 | SIGMETRICS |
| **C2** | co-tenant 상태를 입력으로 받는 완료시간 예측 모델. 요청당 μs 내 평가 가능 | 모델링 | SIGMETRICS |
| **C3** | SoftWall 런타임: 배치/데이터/제어 평면 | 시스템 | NSDI |
| **C4** | expiry·epoch·single-commit 정확성 계약과 fault injection 검증 | 시스템 | NSDI |
| **C5** | 실제 AI-RAN 스택 위 평가: 파티션 없이 MIG CP 수준 격리에 근접 + 회수 용량 정량화 | 평가 | 양쪽 |

**C1은 단독으로도 측정 논문이 된다.** 이미 273조건 + 4레벨 메커니즘 체인이 확보돼 있으므로,
P0/P1이 실패해도 C1만으로 fallback 투고가 가능하다. 이게 이 계획의 안전망이다.

---

## 6. 보유 자산과 공백

### 6.1 재사용 (다시 만들지 않는다)

- 메트릭 파이프라인: nsys gputrace → `{gap_med, gap_p95, gap_p99, duty, launch_rate}`
  (`results/20260803/chain19_gapstats/` 스키마, 코드 `results/20260725/analyze_kernel_gaps.py`)
- L1 프레임 JSON 스키마(`p99_ms`, `miss_1ms`, `raw_ms[]`) — 3년치 런과 호환. 유지한다.
- 정책 엔진 `scripts_for_node/task1/isca_v2/dart_runtime.py` (admission/reservation/commit/lease,
  4정책, 유닛테스트 16.9KB) — MPS 인지 기능만 추가하면 된다.
- background gating 프로토콜 `background_gated.py`, `qwen7b_gated.py` (파일 게이트 1바이트,
  work-unit 경계 동기화)
- NRx 직결 경로 `nrx_trt_direct.py` (caller-owned binding + `capture_graph()`)
- Perlmutter 구동 레시피 `results/perlmutter_handoff/03_mps_compare.sh`
  (`$SCRATCH` pipe-dir + shifter 자동 마운트), `ncu --mps client` 플래그
- 비교표 포맷 `results/20260813_nrx_placement/PLACEMENT_SUMMARY.csv`

### 6.2 공백 (전부 신규)

| 공백 | 확인 |
|---|---|
| CUDA stream priority / `cudaStreamCreateWithPriority` | 레포 전체 0건 |
| `CUDA_MPS_CLIENT_PRIORITY` 사용 | 0건 (시스템에는 존재 확인, 0/1 두 단계) |
| `CUDA_DEVICE_MAX_CONNECTIONS` | 0건 |
| 실행 중 MPS pct 조정 | 0건 — 모든 sweep이 컨테이너 시작 시 고정 |
| MPS를 실제로 조작하는 admission | 0건 — 정책 엔진은 MPS를 전혀 모름 |
| 대역폭 인지 제어 | 0건 |
| L1측 CUDA Graph / persistent buffer | 0건 (NRx측에만 존재) |
| slot-paced 실시간 harness | 0건 — 현재 `miss_1ms`는 설계상 항상 100% |
| NRx 배치(batch>1) 처리율 곡선 | 0건 |
| 노드 내 NVLink P2P NRx pool | 0건 (모두 NIC loopback 기반) |

---

## 7. 단계별 계획

각 단계는 **게이트(다음으로 넘어가는 조건)**와 **실패 시 분기**를 갖는다.

### P0 — 측정 타당성 확보 (2주) · 선행 필수

| 항목 | 내용 |
|---|---|
| 작업 | (1) slot-paced L1 harness: 0.5/1ms 주기 도착 + absolute deadline + miss율. (2) 프레임당 4,263회 malloc/free 제거 (NRx에서 성공한 caller-owned + CUDA Graph 방식). (3) §1.3의 문서·수치 정정. |
| 산출 | 실시간 신뢰 가능한 L1 baseline, 정정된 authoritative 문서 |
| **게이트** | **malloc/free를 제거해도 MPS 실패 결론이 유지되는가?** |
| 실패 분기 | 유지되지 않으면 → C1의 메커니즘 서술을 전면 수정. "harness 아티팩트였다"는 리뷰어 지적을 선제적으로 처리하는 것이 이 게이트의 목적이다. |

> 근거: 현재 `miss_1ms`가 항상 100%이므로 "deadline을 지켰다"는 주장이 구조적으로 불가능하다.
> 또한 20셀 38.5ms = 셀당 1.92ms는 0.5ms TTI 대비 실시간이 아니다. 이 간극의 정체를 P0에서 밝힌다.

### P1 — 특성화 완성 (3–4주) → C1

| 작업 | 목적 |
|---|---|
| N=1 불일치 해소: Perlmutter L1+NeuralRx 1개 = 40ms vs CloudLab Full-GPU MPS N=1 = 63.6ms | 같은 N인데 반대 결과. 워크로드 구성 때문인지 플랫폼 때문인지 확정 |
| compute/memory 판별자 확립 | 온라인으로 co-tenant 클래스를 판정할 수 있는 최소 신호 (per-op bimodality, gap 확산 패턴, DCGM 카운터) |
| bistability 특성화 | 발생 조건과 확률. §2.2 D6의 정량화 |
| knob 무력함의 정식화 | chain18 Part4 + chain19 Exp11 재분석 (신규 측정 불필요) |
| convert 경계 판정 | fp16 직결 가능 여부 → §4.2 확정 |
| **게이트** | **co-tenant 클래스를 μs~ms 내에 온라인 판정할 수 있는가?** 불가하면 L2 모델은 사전 프로파일 기반으로 축소 |

### P2 — 간섭 인지 완료시간 모델 (3–4주) → C2

- 입력: 목적지 GPU의 co-tenant 클래스 구성, outstanding lease, 최근 gap 분포, queue tail, 전송 비용
- 출력: 보수적 완료시각 (분포 또는 상위 분위수)
- 제약: 요청당 μs 내 평가. 무거운 추론 모델은 불가
- 검증: held-out 트레이스에서 예측 오차, 그리고 **과소예측(=deadline miss 유발) 비율**을 별도로 본다
- **게이트:** 기존 queue-only `predicted_finish` 대비 miss 감소가 유의한가? 아니면 C2를 철회하고
  C3 시스템 기여에 집중한다.

### P3 — 데이터 평면 (3주)

| 실험 | 내용 |
|---|---|
| E3-a | GPU0↔GPU1-3 NVLink P2P 1.4MB/314KB 왕복 측정. push vs pull. loopback 대비 |
| E3-b | copy를 GPU0에서 낼 때 vs NRx측에서 낼 때 L1 p99 차이 → R3 검증 |
| E3-c | GPU당 NRx service 1개(내부 배치) vs client N개. **batch size sweep** |
| **게이트** | **batch로 처리율이 올라가는가?** 1164 req/s가 천장이면 노드당 cell 수가 1.7에 묶이고, 스토리가 "scale-out 불가피"로 이동한다 |

> 용량 산수: 1 cell을 0.5ms slot마다 NRx 호출 = 2000 req/s > 단일 A100 1164 req/s.
> **한 장으로 cell 하나도 못 버틴다.** 3 GPU 풀 ≈ 3500 req/s ≈ 1.7 cell.
> 따라서 (i) selective admission, (ii) 배치, (iii) scale-out 중 최소 하나는 필수다.
> 이 숫자가 논문의 motivation 중 가장 강한 한 줄이 될 가능성이 높다.

### P4 — 제어 평면 (5–6주) → C3 + C4

| 실험 | 내용 |
|---|---|
| E4-a | L1/L2/L3 세 루프 구현, `dart_runtime.py`에 MPS actuator 연결 |
| E4-b | **회수 사다리**: GPU0 유휴 68%를 (a)미회수 (b)LLM 무제어 (c)LLM+lease (d)+memory-bound 배제 (e)+priority로 단계 회수. 기준: L1 p99가 (a)의 1.05배 이내 |
| E4-c | NRx GPU 위 background: static pct vs lease. 여기서 `CUDA_MPS_CLIENT_PRIORITY`와 stream priority 최초 측정 |
| E4-d | 정확성 계약: expiry/epoch/commit + fault injection (SIGKILL, 컨테이너 강제 종료, endpoint 지연) |
| **게이트** | **(c)~(e)가 L1 p99 1.05× 이내에서 유의미한 background work를 회수하는가?** 못 하면 "GPU0 전용화는 회수 불가"라는 negative result로 정직하게 보고하고 NRx GPU 회수에 집중 |

### P5 — 평가 (4–5주) → C5

§8 참조.

### P6 — 노드 간 확장 preview (3주)

- Perlmutter 다중 노드. Slingshot-11이므로 pyverbs/ConnectX 코드 재사용 불가.
- 후보: cray-mpich GPU-aware MPI (환경에 `cray-mpich/9.1.0`, `libfabric/2.3.1` 적재 확인), libfabric CXI.
- 측정: 1.4MB/314KB 전송 지연을 노드 내 P2P와 동일 조건에서 비교. deadline feasibility에 들어갈 수
  있는지 판정. **본 논문에서는 preview/확장성 섹션까지만.** 전체 분산은 후속.

### P7 — 집필 및 novelty 방어 (4주)

§10 리스크 R1 대응 포함.

**총 소요: 약 6개월.** 투고 시점은 대상 학회의 현재 CFP를 확인해 역산한다(본 문서에 날짜를 박지 않는다).

---

## 8. 평가 설계

### 8.1 비교 대상 (baseline)

| # | baseline | 역할 |
|---|---|---|
| B1 | L1 단독 (전용 GPU) | 하한(이상값) |
| B2 | MIG cross-partition + MPS | **상한 목표** — 42ms. MIG를 쓸 수 있을 때의 정답 |
| B3 | Full-GPU MPS 기본값 | 순진한 대안 — 411ms |
| B4 | Full-GPU MPS 최적 튜닝(pct=30) | 정적 knob의 최선 — 145.9ms |
| B5 | time-slicing (MPS off) | 최악 참조 |
| B6 | 선행 시스템 정책 1종 재구현 | **novelty 방어에 필수.** 우선순위/선점 기반 co-location 스케줄러 계열 |
| B7 | SoftWall (제안) | — |

B6를 빼면 "기존 GPU 공유 스케줄러와 뭐가 다른가"에 답할 수 없다. 이건 선택이 아니라 필수다.

### 8.2 지표

**1차 (SLA)**
- deadline miss ratio (TTI 기준), p99 / **p99.9** / max
- no-timely 분해: rejected / remote-expired / late / overflow — 의도적 거부와 실패를 섞지 않는다

**2차 (용량)**
- cells per node (deadline miss 0 조건에서)
- NRx timely 처리율, delivered radio utility (correct-TB), BLER
- reclaimed background work (tokens/s, it/s) 및 보존율

**3차 (운영)**
- run-to-run 분산, bistability 발생률
- fault 후 복구 시간, lease 회수 지연(quantum 상한 검증)
- CPU 오버헤드, host-blocked time/slot

**금지:** duty cycle을 SLA 게이트로 쓰지 않는다(§1.2).

### 8.3 시나리오

| # | 시나리오 | 검증 대상 |
|---|---|---|
| S-A | 단일 cell, 주기적 | 기본 보호 |
| S-B | multi-cell 동기/스태거 (1/2/4/8 cell) | fragmentation, burst 흡수 |
| S-C | **1 GPU만 있는 cell-site** | 전용화가 불가능한 최악 조건 |
| S-D | 6종 AI 스택 동시 (Qwen/Whisper/BERT/VL/CsiNet/BeamPred) | 배포 현실성 |
| S-E | memory-bound 테넌트 혼입 | §2.2 D2, sat_hbm |
| S-F | fault injection (crash, rolling restart, endpoint 지연) | C4 |
| S-G | **비협조 테넌트** (lease 무시) | trust boundary, 정직한 한계 |
| S-H | 채널 sweep (Es/N0 −4.0~−3.2dB, Markov burst) | utility admission의 근거 |

### 8.4 주장 ↔ 실험 매핑

| 주장 | 근거 실험 |
|---|---|
| 파티션 없이도 L1을 보호할 수 있다 | E3-b, E4-b, S-A/S-B vs B2/B4 |
| 정적 knob으로는 불가능하다 | chain18 P4 + chain19 Exp11 (보유) + B4 |
| 간섭 채널은 두 개이며 knob은 둘 다 못 잡는다 | P1 전체, S-E |
| NRx는 GPU 단위로만 확장된다 | E3-c, R2 근거 데이터 |
| 노드 내에서 NIC는 불필요하다 | E3-a |
| 유휴 용량을 안전하게 회수할 수 있다 | E4-b/c, S-D |
| 늦은 결과는 안전하게 폐기된다 | E4-d, S-F |
| 예측 가능하다 | 전 시나리오의 run-to-run 분산, S-E |

---

## 9. 논문 구성안

```
§1 Motivation
   AI-RAN 배포: protected L1 + selective NRx + background consolidation
   "한 장으로 cell 하나도 못 버틴다" (2000 req/s vs 1164 req/s)
   MIG를 쓸 수 없는 배포가 존재한다

§2 Characterization (C1)
   두 간섭 채널: launch-queue/host-sync, HBM 대역폭
   4레벨 메커니즘 체인, bistability
   정적 knob 공간의 탐색 종료 (10단계 pct × N sweep)

§3 Model (C2)
   co-tenant 인지 완료시간 예측, μs 평가 제약

§4 SoftWall Design (C3)
   배치 평면 / 데이터 평면(NVLink P2P + IPC, pull) / 제어 평면(3 루프)

§5 Correctness (C4)
   expiry · epoch · single commit

§6 Implementation
   cuPHY/Aerial, TensorRT+CUDA Graph, MPS actuator, lease 프로토콜

§7 Evaluation (C5)
   B1–B7 × S-A–S-H

§8 Limitations
   협조 가정, 단일 플랫폼 범위, 실시간 harness의 한계

§9 Related Work
   GPU 공유/선점 스케줄러, 실시간 GPU, 추론 서빙, RAN 가속
```

---

## 10. 리스크 레지스터

| # | 리스크 | 심각도 | 대응 |
|---|---|---|---|
| **R1** | **novelty 중복** — 간섭 인지 co-location 스케줄링은 기존 연구가 많다 | **최고** | 차별점을 세 개로 고정: (1) optional + utility를 가진 deadline 작업, (2) 선점 불가한 하드 실시간 producer와의 공존, (3) 늦은 결과 = 정확성 위험. **B6 재구현 비교를 반드시 포함.** related work를 P1 시점에 먼저 쓴다 |
| R2 | L1 harness가 실시간이 아님 | 높음 | P0 게이트. 해결 못 하면 "상대적 보호"로 주장 범위를 축소하고 명시 |
| R3 | 협조적 lease 가정이 깨짐 | 높음 | S-G에서 비협조 케이스를 측정하고 trust boundary를 명시. 최소 하드웨어 지원 제안으로 연결 |
| R4 | batch가 안 먹혀 용량이 천장 | 중간 | P3 게이트. 스토리를 scale-out 필연성으로 전환(오히려 motivation 강화) |
| R5 | GPU0 회수가 불가능 | 중간 | negative result로 정직하게 보고. NRx GPU 회수만으로도 기여 성립 |
| R6 | 단일 플랫폼 | 중간 | CloudLab A100 + Perlmutter A100 2환경. 가능하면 3세대 GPU 1종 추가 |
| R7 | Aerial 라이선스/재현성 | 중간 | 아티팩트는 정책 엔진 + 트레이스 + 분석 파이프라인 공개, cuPHY는 컨테이너 참조 |
| R8 | 두 authoritative 문서의 충돌이 남음 | 낮음(즉시 해결 가능) | §1.3, P0에서 처리 |

---

## 11. 지금 말할 수 있는 것 / 아직 말하면 안 되는 것

**말할 수 있다**
- GPU 전용 배치는 AI 부하와 무관하게 L1을 baseline에 유지한다 (Exp 12).
- 공유 배치는 정적 튜닝의 최선으로도 baseline의 1.9–3.8배다 (Exp 11 실측).
- 간섭 경로는 compute와 memory 두 가지로 분리되며 시그니처가 다르다.
- GPU idle gap의 약 85%가 host의 cudaFree/memcpy 블로킹과 시간적으로 겹친다.
- 독립 client 증가는 launch queue 중재를 포화시킨다.
- MPS는 메모리 대역폭 경합에서 time-slicing보다 위험하다.

**아직 말하면 안 된다**
- SoftWall이 MIG CP 수준에 도달한다 (P4 전).
- 현재 harness의 ms 수치가 production TTI 예산을 대표한다 (P0 전).
- 노드 간 확장이 deadline 안에 들어온다 (P6 전).
- 비협조 테넌트에서도 보호가 성립한다 (S-G 전).
- batch로 NRx 용량이 늘어난다 (E3-c 전).

---

## 12. 즉시 착수 3건

1. **P0-3 문서 정정** (반나절) — 45ms 하드코딩 → 실측 145.9ms, 두 문서 충돌 해소, 누락 데이터셋 경로 정리.
   지금 상태로는 어떤 실험 결과도 해석 기준이 흔들린다.
2. **P0-1 slot-paced harness** (1주) — deadline 주장의 전제.
3. **E3-a P2P 측정** (2~3일) — loopback을 걷어낼 수 있는지 확정. 스킴의 데이터 평면이 여기서 결정된다.

1과 3은 병렬 가능하다. 2가 가장 길므로 먼저 착수한다.
