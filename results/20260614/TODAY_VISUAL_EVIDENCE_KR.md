# 20260614 CloudLab d8545 재측정 — 시각 증거

생성일: 2026-06-15
소스 범위: `cloudlab_results/results/20260614/E0–E6, A1, A2, C1, C2`
상위 문서: `results/visual_evidence/MIG_AIRAN_VISUAL_EVIDENCE_KR.md`
환경: d8545-10s10501.wisc.cloudlab.us, driver 550.163.01, Aerial 25-3-cubb, pyaerial 25.3.2 빌드.

---

## 📍 서론 — 한 줄 요약

> **"새 환경(driver 550, 새 Aerial 빌드)에서 MIG cross-partition 격리는 어떤 단일 AI 워크로드에도, 다중 AI 스택에도 L1을 ~40-50ms p99로 지킨다. 그러나 같은 partition 안에 L1과 AI를 둔 순간 partition 크기와 무관하게 ~360ms로 폭락한다. 7g (full GPU) coloc도 356ms → 'partition을 더 크게 잡으면 해결된다' 가설 반박, Perlmutter no-MIG 389ms와 정합. 즉 no-MIG ≡ same-partition coloc은 'shared CUDA context contention floor'다."**

### 본 측정의 핵심 claim (오늘)

1. **Partition 크기는 alone에서만 영향, coloc에서는 무관** — alone은 2g 50ms vs 7g 34ms (1.5x), coloc은 모두 ~356-371ms (편차 4%) (§T1)
2. **Cross-partition 격리는 모든 워크로드 + stacking에 견고** — chanpred/NeuralRx/xapp/sat_compute/sat_hbm/+stacking 모두 p99 ≤ 50ms (§T2)
3. **Same-partition coloc은 워크로드 종류와도 무관** — chanpred (E3 365ms) ≈ NeuralRx (E6 360ms) ≈ mixed cp+rn (A2 ~360ms) → "shared CUDA context contention floor" (§T3)
4. **7g coloc은 사실상 no-MIG** — CloudLab 7g+NeuralRx 356ms ≈ Perlmutter no-MIG+NeuralRx 389ms (§T5)
5. **NCU 미세 측정**: alone과 coloc의 DRAM throughput / L2 hit rate 차이 (§T6)

---

# §T1. Partition size — alone에는 영향, coloc에는 없음

![partition alone vs coloc](figures/fig_t01_partition_alone_vs_coloc.png)

| partition | alone p50/p99 (ms) | +NeuralRx coloc p50/p99 (ms) | coloc/alone p99 ratio |
|---|---|---|---|
| 2g.10gb | 50.2 / 56.0 | 360.5 / 371.1 | **6.6x** |
| 3g.20gb | 37.7 / 46.4 | 355.8 / 360.4 | 7.8x |
| 4g.20gb | 37.5 / 48.3 | 350.7 / 358.9 | 7.4x |
| 7g.40gb (full) | 34.4 / 36.1 | 344.5 / 356.3 | **9.9x** |

**Alone 쪽**: 2g가 다른 partition보다 +13-16ms 느림 (SM/HBM 슬라이스 작음 → throughput 한계). 3g/4g는 사실상 동일 (20GB 공유). 7g는 가장 빠르고 tail이 5배 좁음 (full GPU, idle 변동 흡수).

**Coloc 쪽**: 2g(371) vs 7g(356) 차이는 단 15ms (4%). 즉 **partition 자원을 키워도 coloc 폭락은 그대로**. partition 크기가 dominant variable이 아니다.

→ "MIG partition을 크게 잡으면 L1+AI를 같이 둘 수 있다" 가설 반박.

---

# §T2. Cross-partition 격리 — 어떤 AI + 어떤 stacking도 안전

![cross partition matrix](figures/fig_t02_cross_partition_matrix.png)

L1을 3g에 두고 AI를 4g에 둔 모든 조건:

| 조건 | n | p50 | p99 | alone 대비 |
|---|---:|---:|---:|---|
| alone (3g) | 500 | 37.5 | 47.8 | — |
| + chanpred | 500 | 37.7 | 42.2 | **-5.6** |
| + NeuralRx | 1000 | 37.6 | 43.6 | -4.2 |
| + xapp | 500 | 37.4 | 39.4 | -8.4 |
| + sat_compute | 500 | 37.6 | 41.1 | -6.7 |
| + sat_hbm | 500 | 37.6 | 39.4 | -8.4 |
| + chanpred ×4 | 500 | 36.6 | 37.9 | -9.9 |
| + ResNet ×2 | 500 | 35.9 | 37.0 | -10.8 |
| + kitchen (cp+mc+gm) | 500 | 36.4 | 37.7 | -10.1 |

→ 모든 조건에서 p99가 alone(47.8) **이하**이거나 동급. 단일 AI든 다중 AI든 cross-partition 격리는 깨지지 않음.

특이점: 스택할수록 오히려 p50/p99가 약간 **낮아짐** (alone에는 있는 idle tail이 contention으로 인해 안 생기는 것으로 추정 — 별도 분석 필요).

---

# §T3. Same-partition coloc — 워크로드 종류 무관, ~360ms로 폭락

![coloc workload invariant](figures/fig_t03_coloc_workload_invariant.png)

L1을 3g에 두고 AI를 같은 3g 안에 같이 둔 케이스 — 어떤 AI든 비슷한 floor:

| coloc 조건 | n | p50 | p99 | max | 기준 |
|---|---:|---:|---:|---:|---|
| **E3** chanpred | 500 | 353.5 | 365.9 | 366.1 | 6/1 G_2_3gColoc_chanpred 361.1 ✓ |
| **E6-3g** NeuralRx | 1000 | 355.8 | 360.4 | 366.1 | — (오늘 첫 측정) |
| **A2** chanpred + ResNet mixed | 450 | ~352 | ~362 | ~363 | — (in-progress 9/10 runs) |

→ 세 케이스 모두 p99 360-366ms, 편차 6ms 내. **AI 종류, 단일/혼합 무관**.

이는 partition-level shared CUDA context가 dominant — kernel queue arbitration이 partition 단위로 묶여있고, AI 워크로드가 그 큐를 점유하면 L1이 무조건 기다림. AI가 "compute heavy"든 "memory heavy"든 큐 점유 시간이 비슷하다는 뜻.

---

# §T4. 세 가지 regime — CDF로 한눈에

![three regimes CDF](figures/fig_t04_three_regimes_cdf.png)

같은 x축(log scale)에서 다섯 case 분포 비교:

- **녹색 (alone)**: ~37ms 근처에 매우 좁게 분포
- **파란색 실선/점선 (cross-partition single/stacking)**: alone과 거의 겹침 — 격리 성공
- **빨간색 실선/점선 (coloc chanpred/NeuralRx)**: ~360ms 절벽으로 이동, 두 분포 거의 겹침

이 한 장이 "MIG cross-partition은 동작, same-partition coloc은 catastrophic" 두 가지 주장을 동시에 보여주는 가장 강한 그림.

---

# §T5. 7g coloc ≈ Perlmutter no-MIG — "shared context contention floor"

![7g coloc no-MIG](figures/fig_t05_7g_coloc_equals_no_mig.png)

| 조건 | L1 p99 (ms) | 의미 |
|---|---:|---|
| CloudLab 7g alone (isolated) | 36.1 | full GPU 단독 — 최고 성능 |
| Perlmutter no-MIG alone | 124.0 | no-MIG 단독 (clock-ramp tail로 더 큼) |
| CloudLab 3g coloc + NeuralRx | 360.4 | small partition coloc |
| **CloudLab 7g coloc + NeuralRx** | **356.3** | **full GPU coloc** |
| **Perlmutter no-MIG + NeuralRx** | **389.0** | **no-MIG (Perlmutter §F5)** |

→ "MIG를 꺼서 full GPU를 시간분할로 쓰는 것"과 "MIG 7g로 두 프로세스를 한 instance에 colocate하는 것"이 **같은 메커니즘**임을 보여줌. 모두 ~360-389ms — **shared CUDA context contention의 fundamental floor**.

paper 주장 통합:
- "MIG 격리는 필요하다" — cross-partition (§T2 ~40ms) vs no-MIG/coloc (§T5 ~360ms) 격차 9배+
- "MIG는 partition을 키우는 것으론 부족하다" — 7g(=full GPU MIG)도 360ms (§T1, §T5)
- "운영 실수 한 번에 폭락" — same partition에 두기만 하면 즉시 365ms (§T3)

---

# §T6. NCU per-kernel — alone vs coloc DRAM/L2 (preview)

![NCU DRAM alone vs coloc](figures/fig_t06_ncu_dram_alone_vs_coloc.png)

C1에서 측정한 NCU per-kernel 분포:
- **DRAM throughput**: alone과 coloc의 분포가 어떻게 다른지 (kernel level)
- **L2 sector hit rate**: 캐시 효율 변화

이 figure는 §17 (mechanism: queue arbitration)을 새 환경에서 확인하는 보조 증거. 데이터 차이는 PART F의 Perlmutter NCU(§F9)와 비교해 chip 간 일반성 확인에 사용.

---

## 핵심 발견 종합

1. **MIG cross-partition 격리는 robust** — 어떤 AI, 어떤 stacking에도 L1 p99 ≤ 50ms (§T2)
2. **Same-partition coloc은 partition 크기·워크로드와 무관하게 ~360ms** — shared context contention의 fundamental floor (§T1, §T3)
3. **7g (full GPU) coloc = no-MIG full-GPU time-slice** — 같은 메커니즘, 같은 결과 (§T5)
4. **5/31 phase4의 NeuralRx 205ms bistability**는 driver 525 시절 artifact로 재해석 가능 — driver 550 + 새 빌드에선 cross-partition NeuralRx도 43.6ms (§T2)
5. **운영 관점 design rule** (오늘 측정으로 확정):
   - L1과 AI는 반드시 다른 MIG partition에 둘 것 (one rule to rule them all)
   - partition 크기는 L1 alone p99 latency budget에 맞춰 결정 (작을수록 +5-13ms)
   - MIG-off + MPS는 별도 비교 필요 (Phase 4 측정 대기)

## paper 메시지 매핑

| paper claim | 본 측정의 근거 figure |
|---|---|
| "MIG는 필요하다" | §T4 (세 regime CDF), §T2 (cross-partition flat) |
| "MIG도 부족하다 (운영 실수 시)" | §T3 (workload invariant ~360ms), §T1 (partition 크기 무관) |
| "no-MIG full GPU 공유는 답이 아님" | §T5 (7g coloc = no-MIG 메커니즘) |
| "Partition 크기 증가가 해결책 아님" | §T1 우측 (2g→7g 차이 4%만) |

## 진행 상태 & 다음

| 항목 | 상태 |
|---|---|
| 측정: E0-E6, A1, A2 (9/10), C1 NCU, C2 NSYS | ✅ 완료 |
| 측정: B3 ResNet batch sweep, A3 chanpred coloc 2g/4g/7g, B1 four-way, C3 5분 sustained | 🟡 chain 진행 중에 인터럽트, 데이터 부분만 |
| 측정: Phase 4 (MPS, no-MIG default) | ⏸️ pending (reboot 필요) |
| 측정: PDSCH TX, qwen_small | ⏸️ pending |

다음 세션 작업:
1. Phase 4 (chain_phase4_mps.sh) — 3-way 매트릭스 (MIG / no-MIG default / MPS) 완성
2. 남은 partition condition들 보강 (B3 batch, A3 partition × chanpred coloc)
3. C2 NSYS deep-dive 분석 — §17 mechanism update

## 재현

```bash
cd $SCRATCH/airan_cloudlab/results/20260614  # on d8545
# data: E*/, A*/, B*/, C*/ subdirs (realL1_*.json)
# figures: results/20260614/figures/
python3 results/20260614/build_figures.py    # rebuild from local data
```
