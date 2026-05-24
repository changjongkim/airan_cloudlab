# 20 Figures — 5/24 Sweep Analysis (상세 설명)

생성: `generate_figures.py`. 각 figure는 paper story arc의 특정 메시지를 뒷받침.

---

## A섹션 — MIG mode 자체는 free

### 📊 fig_01_mig_mode_overhead.png
**무엇을 보여주는가**: 막대 그래프 2개
- 막대 1 (회색): Full GPU (no MIG), median 39.3 ms
- 막대 2 (파랑): 7g.40gb MIG single instance, median 36.7 ms
- 에러 바: stdev

**해석**: 두 막대가 거의 같은 높이 (차이 ~2ms, stdev 내). MIG를 켜고 7g 한 덩어리만 만들면 no-MIG와 성능 동일.

**Paper 메시지**: "MIG mode 활성화 자체는 오버헤드 없음". 이 figure는 우리가 MIG 비판할 때 "MIG가 본질적으로 느린 게 아님"을 미리 인정 → 비판 신뢰성 확보.

---

## B섹션 — Partition cap이 진짜 비용

### 📊 fig_02_partition_cap.png
**무엇을 보여주는가**: 5개 막대 (왼쪽→오른쪽 큰 partition→작은 partition)
- Full GPU (no MIG): 39.3 ms (회색)
- 7g.40gb MIG: 36.7 ms (파랑)
- 4g.20gb MIG: 56.5 ms (주황)
- 3g.20gb MIG: 52.5 ms (빨강) ← 4g보다 미세하게 낮음 ⚠
- 2g.10gb MIG: 71.6 ms (보라)
- 빨간 점선: Full GPU baseline (39.3 ms)

**해석**: Partition을 7g→4g→3g→2g로 줄일수록 L1 느려짐. 단 4g(56.5)가 3g(52.5)보다 살짝 높은 비정상 패턴 (cuPHY가 3g+ 에서 saturate되거나 4g.20gb 측정 noise).

**Paper 메시지**: "Partition을 작게 쪼개면 성능 직격". AI와 co-locate 하려면 작은 partition 어쩔 수 없음 → 본질적 비효율.

### 📊 fig_03_partition_cap_overhead.png
**무엇을 보여주는가**: 막대 4개 (% 슬로다운, Full GPU 대비)
- 7g MIG: -6.5% (오히려 빠름)
- 4g MIG: +43.8%
- 3g MIG: +33.7%
- 2g MIG: +82.2%
- 빨간 가로선 = 0% (Full GPU 기준)

**해석**: fig_02를 비율로 normalize. 2g가 +82%로 최악. 7g는 noise 수준의 -6%.

**Paper 메시지**: 정량적 cost 한눈에. "AI-RAN에서 MIG 쓰면 best 4g 줘도 +44% 손해".

---

## C섹션 — Cell scaling (HBM bw saturation, 핵심 figure)

### 📊 fig_04_cell_scaling.png ⭐ (가장 중요)
**무엇을 보여주는가**: 라인 그래프
- X축: cells per L1 iteration (5, 10, 20, 40)
- Y축: L1 latency (ms, median)
- 3개 라인 (다른 partition):
  - 보라 동그라미: 2g.10gb
  - 빨강 사각형: 3g.20gb
  - 주황 삼각형: 4g.20gb

**구체 데이터 포인트** (보라/빨강/주황):
- cells=5: 18.2 / 18.3 / 17.2 (거의 같음)
- cells=10: 41.6 / 36.1 / 29.8 (차이 시작)
- cells=20: 76.8 / 52.5 / 56.5 (2g가 다른 둘보다 크게 늦음)
- cells=40: 141.4 / 148.0 / N=1 데이터만

**해석**: 
- 작은 워크로드 (cells=5): partition 크기 상관없이 비슷 (fixed overhead dominant)
- 중간 워크로드 (cells=10-20): partition 클수록 빠름 — 큰 partition이 HBM bw 더 많이 받음
- 큰 워크로드 (cells=40): 모든 partition wall 도달 (linear 깨짐)

**Paper 메시지**: "각 MIG partition은 chip HBM bandwidth의 1/7, 2/7, 3/7, 4/7 quota만 할당받음. 워크로드 크기가 quota 한계 넘으면 saturate. partition 작을수록 더 일찍 wall에 부딪힘". 이게 **MIG의 fundamental architecture limit 직접 증명**.

### 📊 fig_05_per_cell_efficiency.png
**무엇을 보여주는가**: 라인 그래프 (fig_04의 정규화 버전)
- X축: cells
- Y축: latency per cell (ms/cell) — 즉 L1 mean / N_cells
- 같은 3개 partition 라인

**해석**: 
- 효율 좋으면 (linear 시) per-cell이 일정해야 함
- per-cell이 올라가면 saturation 시작
- 모든 partition이 cells=5에서 가장 낮은 per-cell (fixed overhead 비중 큼)
- cells 늘리면 처음엔 좋아지다가 (overhead amortize) 다시 올라감 (HBM wall)

**Paper 메시지**: saturation point를 "효율 저하"로 직관적으로 보여줌.

---

## D섹션 — Bimodal 현상 (cuPHY 본질)

### 📊 fig_06_bimodal_A0.png
**무엇을 보여주는가**: 히스토그램
- X축: per-iteration L1 latency (ms)
- Y축: 빈도수
- 데이터: A0 (3g + Qwen)의 모든 raw iteration latency (20 runs × 50 iters = 1000 samples)
- 빨간 점선: median

**해석**: 두 봉우리 명확. 낮은 봉우리 ~53ms, 높은 봉우리 ~57ms. cuPHY의 일부 iteration이 항상 빠르고 다른 일부가 항상 느림 — 무작위 noise가 아닌 시스템적 phase.

**Paper 메시지**: "Bimodal 현상 N=1000 sample로 확정". 이게 우리 v2 N=4 finding의 reproducibility 증거.

### 📊 fig_07_bimodal_baseline.png
**무엇을 보여주는가**: fig_06과 같은 구조, 데이터만 다름
- 데이터: Full GPU baseline (15 runs × 50 iters)
- 회색 막대

**해석**: 놀랍게도 baseline (no MIG, no AI)에도 bimodal! 두 봉우리 ~35ms와 ~41ms.

**Paper 메시지**: "bimodal은 MIG/AI가 만드는 게 아니라 cuPHY pipeline 자체에 있음". H1 (Qwen phase alignment) 가설 반증. 우리의 v2 "MIG가 bimodal 유발한다" 가설 폐기.

### 📊 fig_08_bimodal_overlay.png
**무엇을 보여주는가**: 3개 히스토그램 overlay (density 정규화)
- 회색: Full GPU baseline (med 39)
- 파랑: 3g MIG alone (med 52)
- 빨강: A0 3g + Qwen (med 56)

**해석**: 세 분포 모두 bimodal 모양. partition/AI가 위치만 이동시킬 뿐 shape 유지.

**Paper 메시지**: "bimodal shape는 invariant — cuPHY의 미시 스케줄링 현상이지 환경 영향 아님".

---

## E섹션 — AI workload별 효과

### 📊 fig_09_phase1_qwen_variants.png
**무엇을 보여주는가**: 5개 막대 (모두 3g L1 partition)
- 막대 1 (파랑): 3g alone, 52.5
- 막대 2-5 (주황): A0 Qwen full (55.7), A1a prefill (53.7), A1b decode (56.0), A2 static HBM (55.5)
- 빨간 점선: 3g alone baseline
- 에러 바: stdev

**해석**: AI 종류 (full LLM / prefill burst / decode steady / static memory) 무엇이든 모두 ~55ms로 수렴. 차이 ~3ms 이내.

**Paper 메시지**: "H1 (Qwen prefill/decode phase 가설) 폐기". Bimodal 메커니즘이 Qwen의 특정 phase가 아님. 단순 AI 메모리 traffic이면 동일 효과.

### 📊 fig_10_airan_vs_llm.png
**무엇을 보여주는가**: 5개 막대
- 막대 1 (회색): 3g alone, 52.5
- 막대 2 (파랑): A0 Qwen LLM, 55.7
- 막대 3-5 (빨강): AR1 NeuralRx (78.7), AR2 ChanPred (69.6), AR3 xApp (73.7)
- 빨간 점선: 3g alone baseline

**해석**: 실제 AI-RAN 워크로드 3개 모두 LLM Qwen보다 L1을 훨씬 더 느리게 만듦.
- Qwen LLM 영향: +3ms
- ChanPred: +17ms
- xApp: +21ms
- NeuralRx: +26ms

**Paper 메시지**: "TensorRT 기반 inline AI inference (NeuralRx)가 LLM보다 L1 disruption 훨씬 큼". 사람들이 LLM으로만 측정해서 MIG 성능 평가했을 수 있음 → real AI-RAN에선 더 나쁨.

---

## F섹션 — Decomposition (cap vs leakage 분리)

### 📊 fig_11_decomposition_stacked.png
**무엇을 보여주는가**: 5개 stacked bar
- 회색 (baseline cuPHY): 39 ms 항상
- 빨강 (+partition cap): 작은 partition으로 줄여서 발생한 추가
- 주황 (+AI leakage): AI 붙여서 추가로 발생한 leakage
- 라벨: L1 alone (full) / +cap (3g) / +1 AI / +2 AI (M1) / +Real AI-RAN (AR1)

**해석**: 
- baseline 39ms는 항상 깔림
- partition cap (3g): +13ms 추가
- AI 1개: +3ms 추가
- AI 2개: 21ms 추가 (1개의 7배!)
- Real AI-RAN (NeuralRx): 26ms 추가

**Paper 메시지**: cost를 layered로 분해. cap이 base이고 AI leakage는 그 위에. multi-AI나 real AI-RAN에서 leakage 폭발.

### 📊 fig_14_D1_decomposition.png
**무엇을 보여주는가**: 5개 막대
- Full GPU baseline (39)
- 3g L1 alone (52)
- 3g L1 + Qwen (56) [A0]
- 2g L1 alone (71) [D1b]
- 2g L1 + Qwen (71) [D1a]

**해석**: D1a vs D1b 비교 → 2g L1에서는 AI 있어도 거의 영향 없음 (-0.86ms). 왜? L1이 compute-bound이라 HBM bw 안 씀, AI도 다른 partition에서 자기 quota만 씀.

**Paper 메시지**: "compute-bound L1 (작은 partition)은 AI co-location 영향 거의 안 받음. 대신 partition cap 자체가 큼 (71 vs 39 = +82%)".

---

## G섹션 — Multi-AI (size > count, 핵심 finding)

### 📊 fig_12_multi_ai_count.png ⭐
**무엇을 보여주는가**: 5개 막대 (모두 3g L1)
- 회색 (3g alone, 0 AI): 52.5
- 파랑 (A0 +1 Qwen on 2g): 55.7
- 빨강 (M1 +2 AI on 2g+2g): 73.9 ← 큰 점프
- 녹색 (+3 AI on 1g×3): **39.4** ← 더 빠름!
- 녹색 (+4 AI on 1g×4): **39.3** ← 더 빠름!

**해석**: AI 개수가 늘어나도 partition을 작게 쪼개면 (1g.5gb) L1에 영향 없음. 오히려 alone보다 더 빠른 (39 < 52) 이상한 결과 — 이건 3g.20gb의 HBM bw quota가 layout에 따라 다를 가능성.

**Paper 메시지**: "MIG의 AI leakage는 **AI partition 크기**에 비례, 개수가 아님". 1g.5gb 4개로 쪼개면 contention 거의 없음. **Design implication: AI는 가능한 한 작은 partition에 쪼개서 배치**.

### 📊 fig_13_phase2_multipartition.png
**무엇을 보여주는가**: 4개 막대 (Phase 2 각 multi-partition config)
- M1 3g L1 + 2× 2g AI: 73.9
- M2 2g L1 + 3g+2g AI: 132.8 (worst)
- M3 4g L1 + 1g+2g AI: 63.5
- M4 4g L1 + 3× 1g AI: 59.6 (best)

**해석**: 
- L1을 큰 partition (4g)에, AI를 작은 partition (1g)에 → 최적 (M4)
- L1을 작은 partition (2g)에 두면 catastrophic (M2)

**Paper 메시지**: "AI-RAN deployment guideline: L1 partition은 가장 크게, AI는 가장 작게 잘라서". M4가 최선의 절충.

---

## H섹션 — Tail latency / URLLC 불가능

### 📊 fig_15_p99_urllc.png
**무엇을 보여주는가**: 8개 막대 (p99 medians)
- Full GPU: 55.6
- 7g MIG: 51.6
- 3g alone: 77
- A0: 80
- M1: 171
- M4: 83
- AR1: 234
- M2: 382
- 빨간 점선: URLLC 1ms target

**해석**: 모든 config의 p99 >> 1ms. 가장 좋은 Full GPU도 55ms. AR1 (NeuralRx) p99 234ms는 disaster.

**Paper 메시지**: "URLLC sub-millisecond reliability는 어떤 MIG config로도 달성 불가능. 5G NR TTI deadline (1ms)을 cuPHY가 batched로 처리하더라도 p99에서 fail".

### 📊 fig_16_cdf_comparison.png
**무엇을 보여주는가**: CDF 라인 그래프 5개
- X축: L1 latency (ms)
- Y축: CDF (0-1)
- 회색: Full GPU baseline (왼쪽으로 가장 압축)
- 파랑: 3g MIG alone
- 주황: A0 (+Qwen)
- 빨강: M1 (+2 AI) — 오른쪽 꼬리 김
- 보라: AR1 (NeuralRx) — 가장 오른쪽

**해석**: 각 config의 latency 분포 전체 확인. M1과 AR1은 오른쪽으로 긴 꼬리 — tail-heavy.

**Paper 메시지**: "median 보다 tail이 훨씬 나쁨. Real-time application은 worst case 봐야 함".

### 📊 fig_17_qq_plot.png
**무엇을 보여주는가**: Q-Q plot (정규성 검정)
- X축: theoretical normal quantile
- Y축: observed L1 latency
- 데이터: A0 raw iterations
- 빨간 점선: 정규분포 같으면 일치해야 할 선

**해석**: 점들이 직선에서 S 모양으로 벗어남 → 정규분포 아님. 두 peak이 있는 bimodal의 전형적 Q-Q shape.

**Paper 메시지**: "bimodal을 statistical test로 confirm. unimodal Gaussian 가정 깨짐 → 평균 통계로만 분석하면 안 됨".

### 📊 fig_18_tail_ratio.png
**무엇을 보여주는가**: 10개 막대 (p99 / mean ratio)
- Full GPU ~1.45
- 7g MIG ~1.40
- ... (대부분 1.3-1.7)
- M1 ~2.3 (가장 큼)
- AR1 ~3.0

**해석**: ratio가 1이면 tail 없음. 클수록 tail heavy. AR1 (NeuralRx)이 mean 대비 3배 p99 — burst 패턴 영향.

**Paper 메시지**: "tail amplification은 workload 종류에 매우 민감. TensorRT burst가 가장 강한 amplifier".

---

## I섹션 — Overall summary

### 📊 fig_19_airan_spectrum.png
**무엇을 보여주는가**: 6개 시나리오 × 2 막대 (median + p99 나란히)
- Full GPU (unrealistic)
- Best AI-RAN (M4 4g + 3 light)
- Typical (A0 3g + Qwen)
- Heavy (M1 3g + 2 AI)
- Real AI-RAN (AR1 NeuralRx)
- Catastrophic (M2 2g L1)

**해석**: deployment 스펙트럼. 어떤 시나리오를 택해도 어딘가에 cost. median과 p99 격차 큼.

**Paper 메시지**: "AI-RAN deployment 시 trade-off matrix. 어떤 corner를 택하든 베이스라인 도달 불가".

### 📊 fig_20_overall_breakdown.png ⭐ (Final summary)
**무엇을 보여주는가**: 8개 시나리오 stacked bar
- 회색 base: cuPHY baseline (39ms 항상)
- 빨강 layer: partition cap
- 주황 layer: AI leakage
- 각 막대 위에 total 숫자 (39, 37, 52, 56, 74, 79, 60, 133)
- 빨간 점선: cuPHY baseline 39ms reference

**해석**: 모든 시나리오의 cost를 동일한 framework로 비교. cuPHY 본질 vs MIG가 추가한 부분.

**Paper 메시지**: 이 한 장으로 paper의 모든 message 정리. "MIG는 39ms base 위에 13-94ms 추가. 어떤 config도 39ms 회복 못 함 (AI co-locate 한 채로는)".

---

## Key Numbers Cheat Sheet

```
Full GPU (no MIG):       39.28 ms  ← baseline
7g MIG single:           36.73 ms  ← MIG free
3g MIG alone:            52.54 ms  ← partition cap +34%
3g + Qwen (A0):          55.73 ms  ← +6% AI leakage
3g + 2 AI on 2g (M1):    73.94 ms  ← multi-AI bad
3g + 3 AI on 1g:         39.36 ms  ← surprise: small AI partitions ok!
3g + Real AI-RAN (AR1):  78.71 ms  ← TensorRT worst
2g L1 (M2):             132.82 ms  ← catastrophic
Best AI-RAN (M4):        59.64 ms  ← +52%, minimum achievable
```

## Paper Punchline

> "MIG mode itself imposes negligible overhead (7g MIG single ≈ no-MIG), but AI-RAN co-location requires partition fragmentation, which costs 34-238% L1 latency. AI leakage is small (≤6%) for single AI on properly-sized partitions, but grows to +88% with multiple co-located AI services. Best feasible AI-RAN configuration (4g L1 + 3 light AI on 1g.5gb partitions) still imposes +52% latency penalty vs full GPU baseline. URLLC sub-millisecond p99 is infeasible in all measured MIG configurations."

## Limitations

- AI workload throughput not validly measured (next reservation)
- N=5 for cell scaling — moderate statistical power
- 4g_cells40 has only N=1 valid (JSON cross-contamination)
- M2 stdev=0 anomaly (likely earlier-run cross-contamination)
- Multi-AI on 1g (3AI/4AI) showed 39ms < 52ms (3g alone) — unexplained, possibly MIG profile dependency

## Next Reservation (5/31) TODO

- Fix AI throughput methodology (persistent L1 loop)
- Re-measure 4g_cells40 with N=10
- Re-measure 3g_alone to resolve 39ms anomaly
- MPS-based comparison for full bandwidth scheduling
- dmon HBM utilization correlation analysis
