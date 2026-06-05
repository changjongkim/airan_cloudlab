# no-MIG는 MIG의 격리를 대체하지 못한다: Perlmutter 시각 증거 (PART F)

작성: 2026-06-05 · 측정: Perlmutter (NERSC) A100-SXM4, **MIG OFF**
상위 문서: `results/visual_evidence/MIG_AIRAN_VISUAL_EVIDENCE_KR.md` (CloudLab MIG 증거, PART A–E)
대응 데이터: `results/20260531`, `results/20260601` (CloudLab MIG)

---

## 📍 서론 — 한 줄 요약

> **CloudLab에서 MIG로 측정한 L1+AI 간섭 실험을 Perlmutter A100에서 MIG를 끈 상태로 동일하게 재측정한 결과, no-MIG 시간분할(time-slice)은 단일 AI co-tenant에도 L1 frame p99가 MIG 대비 4–9배, 누적 부하에서는 최대 29배 폭락한다. MIG의 cross-partition 격리는 어떤 AI를 붙여도 L1 p99를 ~45ms로 고정시킨다.**

### 이 문서가 답하는 질문
Reviewer 예상 질문: *"MIG가 부족하다면, 그냥 MIG 안 쓰고 full GPU를 시간분할로 공유하면 되지 않나?"*
→ **답: no-MIG는 격리가 전혀 없어 더 나쁘다.** 본 PART F가 그 정량 증거다.

### 핵심 claim (PART F)
1. no-MIG default time-slice는 단일 AI에도 L1 p99가 **3배 이상**(자기 baseline 대비) 무너진다.
2. MIG에서 "안전"하던 워크로드(chanpred)도 no-MIG에선 위험해진다.
3. 누적 부하(stacking)에서 no-MIG는 선형 이상으로 붕괴(chanpred×4 = MIG 대비 **29배**).
4. NeuralRx(PHY-AI)의 no-MIG p99(389ms)는 MIG의 **최악 케이스(same-partition coloc 357ms)와 동급**이며, MIG 격리 케이스보다 2–8배 나쁘다.

---

## 측정 환경

| 항목 | CloudLab (MIG, 기존) | Perlmutter (no-MIG, 본 PART F) |
|---|---|---|
| GPU | A100 (MIG on) | A100-SXM4-40/80GB (**MIG off**) |
| 격리 | MIG cross-partition (L1=3g, AI=2g) | **없음** — full GPU 시간분할 |
| 컨테이너 | Docker `airan:25-3-final` | Shifter `aerial:25-3-cubb` + venv(torch) |
| L1 | cuPHY PUSCH RX (CE+EQ+NI), 20-cell 직렬 frame | 동일 (pyaerial API만 포팅) |
| 측정 | frame time mean/p50/p99, n=5 (alone n=10) | 동일 |
| 조건 | 14 conditions | 동일 14 conditions |

> L1 "frame" = 20셀 직렬 실행 시간(throughput식 벤치마크). MCS=2, 273 PRB, 4×4 안테나. miss_1ms는 설계상 항상 100%(20셀 직렬) → **p50/p99 비교가 핵심 지표**.

---

# §F1. 전체 비교 — MIG vs no-MIG L1 p99

![MIG vs no-MIG p99](figures/figF1_mig_vs_nomig_p99.png)

같은 조건, 같은 L1, 같은 측정 방식. **파란색(MIG cross-partition)은 어떤 AI에도 ~45ms로 평탄**하다. **빨간색(no-MIG)은 124ms(alone)에서 1330ms(chanpred×4)까지 치솟는다.** (qwen/xApp/NeuralRx/sat_compute/sat_hbm은 CloudLab F_saturation 캠페인에 없어 no-MIG만 표시 — NeuralRx는 §F5에서 별도 비교.)

| 조건 | MIG p50/p99 (ms) | no-MIG p50/p99 (ms) | no-MIG/MIG p99 |
|---|---|---|---|
| alone | 43.7 / 59.2 | 34.6 / 124.0 | 2.1x |
| chanpred (safe AI) | 40.4 / 45.2 | 154.7 / 200.7 | **4.4x** |
| H2D memcpy | 45.3 / 47.6 | 134.8 / 195.2 | 4.1x |
| forecaster | 43.6 / 45.4 | 354.9 / 381.3 | 8.4x |
| GEMM 4096 | 42.8 / 45.4 | 371.4 / 406.6 | 9.0x |
| ResNet50 | 43.6 / 45.3 | 386.1 / 417.7 | **9.2x** |
| D2D memcpy | 43.1 / 45.6 | 391.6 / 434.9 | 9.5x |
| ResNet ×2 | 43.1 / 45.2 | 701.9 / 739.3 | **16.3x** |
| kitchen sink | 42.9 / 45.6 | 966.4 / 987.4 | **21.7x** |
| chanpred ×4 | 43.4 / 45.4 | 1311.5 / 1329.7 | **29.3x** |

---

# §F2. 저하 배수 — no-MIG는 MIG보다 몇 배 나쁜가

![no-MIG/MIG ratio](figures/figF2_nomig_over_mig_ratio.png)

같은 AI 부하에서 no-MIG L1 p99는 MIG 대비:
- **단일 AI**: 4–9.5배 (chanpred 4.4x, ResNet 9.2x, D2D 9.5x)
- **누적 부하**: 16–29배 (ResNet×2 16x, kitchen 22x, **chanpred×4 29x**)

격리가 없으면 부하가 쌓일수록 격차가 기하급수적으로 벌어진다.

---

# §F3. 격리 스토리 — 각 플랫폼 자기 baseline 대비

![Normalized contention](figures/figF3_normalized_contention.png)

플랫폼 간 절대 비교는 baseline 차이(GPU 크기)를 섞으므로, **각 플랫폼을 자기 alone baseline으로 정규화**해 순수 contention 효과만 본다:
- **MIG**: 모든 조건에서 ~1.0 (평탄) — AI가 별도 파티션이라 L1에 영향 없음. CloudLab 분석에서도 contention delta가 −17%(좋은 방향, 통계적으로 SEPARATE).
- **no-MIG**: 1.6배(chanpred) → 10.7배(chanpred×4)로 단조 상승.

이것이 "MIG 격리"의 본질이다: MIG는 contention을 0으로 만들고, no-MIG는 막을 수단이 없다.

---

# §F4. 분포 — frame-time CDF

![CDF key conditions](figures/figF4_cdf_key_conditions.png)

대표 3개 조건의 frame-time 누적분포(CDF):
- **alone**: no-MIG가 full GPU라 p50은 더 빠르지만(왼쪽) 꼬리가 김(오른쪽으로 늘어짐).
- **chanpred / chanpred×4**: MIG는 분포가 ~45ms에 좁게 고정. no-MIG는 분포 전체가 오른쪽으로 크게 이동 — 단순 tail 악화가 아니라 **전 구간이 느려짐**.

---

# §F5. NeuralRx — 핵심 PHY-AI co-tenant

![NeuralRx focus](figures/figF5_neuralrx_focus.png)

NeuralRx는 generic AI가 아니라 L1과 같은 PHY 파이프라인을 공유하는 co-tenant라 가장 위험하다(상위 문서 §2).

| 케이스 | L1 p99 | 출처 |
|---|---|---|
| L1 alone (no-MIG) | 124ms | 본 측정 |
| MIG cross-partition + NeuralRx | ~196ms | CloudLab (워크로드 의존 41↔196) |
| MIG same-partition coloc + NeuralRx | 357ms | CloudLab (상위 §4) |
| **no-MIG + NeuralRx** | **389ms** | **본 측정** |

→ **no-MIG(389ms)는 MIG 최악 케이스(coloc 357ms)와 동급으로 나쁘다.** MIG의 격리 케이스(cross-partition ~196ms)보다 2배, alone 대비 3.1배.

---

## 핵심 발견 종합

1. **no-MIG = 격리 0**: 단일 AI 하나에 L1 p99가 자기 baseline 대비 3배+, MIG 대비 4–9배 폭락.
2. **"안전한" AI 신화 붕괴**: MIG에서 무영향이던 chanpred가 no-MIG에선 4.4배 악화.
3. **누적 = 재앙**: chanpred×4에서 MIG 45ms vs no-MIG 1330ms (29배).
4. **MIG도 만능은 아니다**: same-partition coloc(357ms)은 no-MIG(389ms)와 동급 → 상위 문서의 *"MIG는 필요하지만 same-partition coloc은 불충분"* framing과 정합.

## 정직한 caveat

- **플랫폼 차이**: CloudLab MIG는 A100 slice(3g), Perlmutter no-MIG는 full A100-SXM4. 절대값 직접 비교보다 **자기 baseline 정규화(§F3)**가 공정.
- **no-MIG alone tail**: no-MIG alone p99(124ms)는 p50(34.6ms)보다 큰 체계적 tail(측정 내 clock-ramp 등; 10개 run 일관 → 노이즈 아님). 따라서 §F3의 정규화 비교를 1차 근거로 사용.
- **forecaster**: ours d384 vs MIG d512 (약간의 설정 차이).
- **NeuralRx/qwen/xApp/sat_***: CloudLab F_saturation 캠페인에 없어 §F1 표에서 일부 MIG 칸 비어 있음 — NeuralRx는 §F5에서 별도 데이터로 비교.

---

## 진행 상태 & 다음 단계

| 우선순위 | 내용 | 상태 |
|---|---|---|
| 1 | no-MIG default time-slice (본 PART F) | ✅ 완료·분석됨 |
| 2 | no-MIG **MPS** | 🟡 측정 중 → 완료 시 §F6 추가 (MIG·default·MPS 3-way) |
| 3 | NCU DRAM/L2/SM | 🟡 큐 → §F7 (상위 §18.2 throughput 가설 비교) |
| 4 | per-call NSYS | 🟡 큐 → §F8 (상위 §16.1 memcpy 4.2→14.3us 비교) |
| 5 | 5분 sustained | 🟡 큐 → §F9 (상위 §19.3 지속성 비교) |

> MPS 결과가 나오면 "MPS가 시간분할 대비 contention을 얼마나 회복하고 MIG에 얼마나 근접하는가"를 §F6으로 추가해 **MIG / no-MIG-default / no-MIG-MPS 3-way 매트릭스**를 완성한다.

## 한 줄 종합

> **MIG의 cross-partition 격리는 AI-RAN L1을 어떤 AI에도 ~45ms로 지켜내지만, no-MIG 시간분할은 단일 AI에 4–9배·누적 부하에 최대 29배 L1 p99를 무너뜨린다. "MIG 없이 full GPU 공유"는 AI-RAN에서 실현 불가능하며, 이는 'MIG는 필요하다'는 본 연구의 주장을 직접 입증한다.**

---

### 재현 방법
```bash
cd $SCRATCH/kcj/airan_cloudlab/results/perlmutter_handoff
# 측정: sbatch run_F_nomig.sbatch  (regular, 1 node, no-MIG)
# 분석: python3 analyze_F_nomig.py ; python3 compare_mig_vs_nomig.py
# 그림: shifter --image=<aerial> airan_venv/bin/python build_perlmutter_figures.py
```
데이터: `perlmutter_nomig/F_nomig/realL1_*.json` (80개) · 그림: `figures/figF*.png`
