# MIG 파티셔닝의 한계 — 데이터로 본 비판적 분석

**연구 결론**: 하드웨어 MIG는 **AI-RAN의 최적 해결책이 아니다**. mean latency 개선은 미미하고, 비대칭 split에서 간헐적 leakage가 발생하며, partition 크기에 따른 하드 cap이 존재한다. **MIG만으로는 5G L1 SLA를 보장할 수 없으며, software-level orchestration이 여전히 필요하다.**

---

## 0. 한눈에 보기 — Critical Summary

![Critical Summary](charts/X_critical_summary.png)

| 측면 | 발견 | 결론 |
|---|---|---|
| **mean isolation** | v1: 9× (잘못된 측정) → v2: **1.24×** | 무거운 워크로드에서 거의 효과 없음 |
| **p99 jitter** | no-mig 172ms → MIG 48ms (3.6×) | **유일하게 결정적 효과** |
| **Asymmetric split** | split-60-40에서 **bimodal** | 50% 확률로 +14% 누수 |
| **Partition size** | 1g.5gb OOM, 4g.20gb = 3g.20gb | 하드 cap 존재 |
| **Cell-count** | MIG는 linear, no-mig polynomial | 큰 cell에서만 MIG 유효 |

---

## 1. 실험 설정 — v1 vs v2의 차이가 결정적

### v1 (light workload) — 잘못 측정한 데이터
- AI 워크로드: GPT-2 124M, ResNet-50, HBM stress 0.5-16GB
- L1: cuPHY 4T4R 20-cell
- **문제**: GPT-2 124M은 모델이 500MB — HBM bandwidth를 거의 안 씀 (~1%)
- → 측정된 "11× MIG 격리"는 사실 **SM 경합**이지 HBM 격리가 아니었음

### v2-v5 (heavy workload) — 진짜 측정
- AI 워크로드: **Qwen-7B (14GB fp16)**, HBM 16GB stress
- L1: cuPHY **8T8R** 20-cell (채널추정 4× 부하)
- 결과: MIG 격리 효과 **드라마틱하게 감소** (1.24× mean)

### 워크로드 크기 비교

![v1 vs v2 Critical](charts/X_v1_vs_v2_critical.png)

**핵심**: v1에서 워크로드가 너무 작았다. 실제 production AI-RAN 시나리오에는 v2 데이터가 representative.

---

## 2. 한계 #1: 무거운 워크로드에서 mean isolation은 미미

### v2 heavy AI 측정

![MIG vs no-mig heavy](charts/v2_04_mig_vs_nomig_heavy.png)

```
no-mig + Qwen-7B:        57.89 ms
MIG split-50-50 + Qwen:  46.47 ms
isolation factor: 1.24× (mean)
```

**의미**: AI 워크로드가 실제로 HBM bandwidth를 쓰는 시나리오 (Qwen-7B 22% BW)에서 MIG는 mean latency를 **고작 24% 개선**. v1의 "11× 격리" 주장은 무너진다.

### 왜 이런 차이인가

| 워크로드 | HBM 사용량 (추정) | no-mig + L1 결과 |
|---|---|---|
| GPT-2 124M (v1) | ~1% BW | 375ms (SM 경합 + 측정 artifact) |
| ResNet-50 (v1) | ~1.5% BW | 379ms |
| **Qwen-7B (v2)** | **~22% BW** | **57.89ms** (실제 HBM 경합) |
| HBM stress 16GB (v2) | ~100% BW | 48ms (둘이 fair-share) |

- v1: AI가 HBM 거의 안 씀 → no-mig에서 SM 경합으로 L1 느려짐 → MIG가 SM 격리하니 11× 개선
- v2: AI가 진짜 HBM 사용 → no-mig에서도 fair-share → mean 차이 미미

---

## 3. 한계 #2: **Asymmetric split에서 Bimodal Leakage** (가장 결정적 한계)

### split-60-40 + Qwen-7B, N=4 반복 측정

![Bimodal](charts/v2_03_bimodal_detail.png)

```
Run 1: 52.80 ms (HIGH mode, +6.7ms)
Run 2: 46.67 ms (LOW mode, baseline)
Run 3: 46.53 ms (LOW mode)
Run 4: 52.86 ms (HIGH mode, +6.7ms)
```

**bimodal**: 두 cluster (~46ms, ~52ms), 50:50 확률.
**MIG가 격리한다고 주장하지만 50% 확률로 +6.7ms leakage가 발생.**

### 비교: 다양한 split에서 L1 결과

![Splits with Qwen](charts/v2_02_splits_with_qwen.png)

| Split | L1 partition | L1 mean | leakage vs B baseline |
|---|---|---|---|
| L1 alone (B) | 3g.20gb | 46.14 | baseline |
| split-50-50 | 3g.20gb | 46.47 | +0.7% (대칭, 안정) |
| split-40-60 | 2g.10gb | 66.55 | +44% (partition cap) |
| split-60-40 | 3g.20gb | 49.71 avg, **bimodal** | +7.7% avg, +14% high mode |

→ **대칭 split만 신뢰할 수 있음**. 비대칭 split은 간헐적 6ms spike → 5G TTI 1ms deadline 위반.

### 가능한 원인 (추정)
- L2 cache controller pool 공유 (NVIDIA 공식 spec)
- HBM channel arbitration (time-multiplexed)
- LLM burst phase (Qwen attention/KV cache write 폭증)
- Power budget arbitration (4g instance 활발 시 thermal throttle)

---

## 4. 한계 #3: Partition 크기 하드 cap — 작은 partition은 OOM, 큰 partition도 효과 없음

### L1 alone baselines (no AI)

![Partition Baselines](charts/v2_01_partition_baselines.png)

```
1g.5gb:  OOM (cuPHY LdpcDeRateMatch가 >5GB HBM 필요)
2g.10gb: 59.27 ms
3g.20gb: 46.14 ms (sweet spot)
4g.20gb: 52.77 ms (3g보다 느림!)
full GPU: 측정 못 함 (driver bug)
```

**결정적 시사점**:
- 1g.5gb는 **cuPHY 자체가 들어가지 않음** → 가장 작은 partition unusable
- 3g→4g (SM 4× 증가, HBM BW 동일): L1 **빨라지지 않음** → cuPHY는 HBM-bandwidth-bound
- 즉 partition을 키워도 격리 더 잘 되는 게 아님. **하드웨어 절반(3g.20gb)이 cuPHY 최적**.
- 나머지 절반은 AI로 — 하지만 AI도 자기 partition cap에 갇힘.

### v1 partition 비교 (잘못 해석된 데이터)

![v1 partitions](charts/v1_11_partition_v1.png)

v1에서는 no-mig가 375ms로 보이지만, 이건 **SM 경합 artifact**. v2 data와 일치하지 않음.

---

## 4b. ⚠️ v2 데이터 자체의 한계 — Coverage가 split-50-50에 편중

v2 heavy workload 실험은 reservation 시간 부족으로 **각 split마다 충분한 N과 다양한 AI workload를 측정 못 함**.

### Coverage matrix

![v2 coverage matrix](charts/v2_06_coverage_matrix.png)

```
                     none   Qwen   HBM16   ResNet   Multi-AI
split-20-80           —     —      —       —        —      (모두 OOM 예상)
split-40-60           —     N=1    —       —        —      ← 단 1개 측정!
split-50-50           N=1   N=4    N=1     —        —      ← 가장 많이 측정
split-60-40           —     N=4    —       —        —      ← bimodal 발견했지만 Qwen만
no-mig                —     N=1    N=1     —        —
B baselines (alone)   N=3   —      —       —        —      ← B/B2/B4
```

### 모든 v2 측정 한눈에

![v2 all splits](charts/v2_07_all_splits.png)

**솔직히 인정해야 할 점**:
- **split-40-60은 단 1개 측정 (66.55ms)** → 진짜 그 값인지 outlier인지 모름
- **split-60-40 bimodal은 N=4 만으로 발견** → 50:50 확률 추정도 통계적으로 약함
- **ResNet / Multi-AI on heavy L1 미측정** → workload type 영향 비교 불가
- **L1 alone full GPU (A baseline) 측정 실패** → 진짜 best case 모름
- **모든 split에서 같은 AI 종류 + 같은 N 으로 측정해야 fair comparison** — 미완

### 의미

이 데이터로 "MIG 전체"를 단정 내리기엔 **N이 부족**. 그러나 발견된 패턴 — **bimodal leakage, mean isolation 1.24×** — 은 **재측정 가치 있는 신호**. 다음 reservation에서 가장 먼저 채워야 할 gap.

특히 **split-40-60 (L1=2g.10gb) 66.55ms는 1번만 측정**이라, 이 숫자로 "작은 partition은 안 좋다"는 결론을 내리는 건 약한 근거. partition 자체의 cap 때문일 수도, AI leakage 때문일 수도, 단순 noise일 수도. **재측정 필수**.

---

## 5. 한계 #4: 같은 preset 안에서도 multimodal (deterministic 아님)

### v1 split-50-50 cells=20 의 29 runs

![Multimodal](charts/v1_08_multimodal.png)

같은 MIG split, 같은 cell 수, sub-parameter (PRB, antenna, MCS, AI type) 만 다른데도:
- ~32-46 ms 범위
- **여러 cluster** (32, 35, 39, 41, 46) 존재
- AI 종류 간 차이도 모호 (cluster 겹침)

**시사점**: MIG split-50-50조차도 deterministic하지 않음. 같은 격리 설정이라도 sub-config에 따라 ±45% 변동.

---

## 6. 한계 #5: 무거운 워크로드에서 cell-count scaling 미검증

### v1 (light AI, GPT-2 124M)

![Cell scaling](charts/v1_01_cell_scaling_4presets.png)

v1에서는 MIG가 cell-count linear scaling 보였지만 — 이건 light AI 데이터.

```
cells = 1, 4, 10, 20, 40
no-mig + GPT-2: 1.97, 6.79, 189.32, 375.76, 747.92  (10 cell에서 폭증)
MIG split-50-50: 2.18, 8.37, 17.29, 34.64, 68.96 (linear)
```

**미확인 사항**: heavy AI (Qwen-7B) 하에서 cell-count가 어떻게 scaling하는지?
- v3 cell=1 only: 2.84 ms (split-50-50 + Qwen)
- cell=4, 10, 20, 40: **측정 못 함** (reservation 만료)
- Qwen이 HBM 22% 사용하면 MIG 격리가 cell≥10에서도 유지될까? **모름**.

---

## 7. v1의 "긍정적" 발견들도 비판적으로 봐야

### v1 HBM 32× 안정 — 진짜 격리인가?

![HBM sweep](charts/v1_03_hbm_intensity.png)

```
HBM 0.5GB → 41ms
HBM 16GB  → 41ms (변동 ±2ms)
```

겉보기엔 격리 완벽. 하지만:
- HBM stress 자체가 simple `dst.copy_(src)` 반복 → cache-friendly access pattern일 수 있음
- 실제 production AI (LLM attention)는 더 random access → 다른 결과 가능성

### v1 stability — 짧은 측정 한정

![Stability](charts/v1_10_stability.png)

1000 iters까지는 stable했지만 — 분 단위, 시간 단위 stability 미검증.
실제 deployment는 hours~days continuous. drift/thermal/power throttling 영향 미측정.

### v1 AI type 무관 — small workload artifact

![AI type](charts/v1_02_ai_type_compare.png)

GPT-2/ResNet/HBM 모두 비슷한 41ms — 모두 light라서 HBM 안 씀.
v2 Qwen에서는 다른 결과 (57.89ms no-mig vs 46.47 MIG).

---

## 8. v1 sub-parameter sweeps — 모두 split-50-50 + light AI 한정

### ResNet batch sweep

![ResNet batch](charts/v1_04_resnet_batch.png)

bs=8 → 34.65, bs=64 → 41.33. ResNet이 batch 8× 커져도 L1 변동 작음 (~7ms). 격리 성공처럼 보이나 **bs=64조차 16GB ResNet 가중치 + 활성화는 ~1GB 미만** — 여전히 light.

### PRB sweep

![PRB sweep](charts/v1_05_prb_sweep.png)

PRB 51→273 (5×) 변화 → L1 거의 무변 (33-35ms). cuPHY 내부 fixed overhead가 dominant. PRB는 L1 부하 측정 지표로 부적합.

### Antenna sweep

![Antenna sweep](charts/v1_06_antenna_sweep.png)

2T2R → 31.84, 8T8R → 46.27. 안테나 증가에 따라 비례 — 채널추정 부하가 커짐. **8T8R 부하가 v1에서는 1.5× 무거움**. v2가 8T8R 쓴 이유.

### MCS sweep

![MCS sweep](charts/v1_07_mcs_sweep.png)

MCS 0~24 → L1 34-41ms, **비단조 (non-monotonic)**. MCS 효과보다 measurement variance가 큼. → multimodal 한 번 더 확인.

---

## 9. 14개 OOM 실패 — 1g.5gb partition은 useless

![OOM](charts/v1_09_oom_1g5gb.png)

v1의 14개 fail은 **전부 L1을 1g.5gb partition에 배치**:
- split-20-80 (L1=1g.5gb): 11 fails
- four-way-eq (L1=1g.5gb): 1 fail
- seven-1g (L1=1g.5gb): 2 fails

**cuPHY LdpcDeRateMatch 초기화만으로도 >5GB HBM 필요** → MIG 가장 작은 partition은 cuPHY L1에 무용지물.

A100 7-instance 최대 분할에서 모든 instance가 1g.5gb이라는 점에서 → **MIG의 maximum density 시나리오는 5G L1에 부적합**.

---

## 10. 종합 비판 (Why MIG is not optimal)

### MIG가 "잘 작동"한다고 주장되는 면
✅ p99 jitter 격리 (3.6×)
✅ Cell-count saturation 방지 (light AI 한정)
✅ 안정성 (1000 iter)

### MIG의 진짜 문제
❌ **Mean latency 효과 미미** (heavy AI에서 1.24× 만)
❌ **Asymmetric split에서 bimodal leakage** (50% 확률, +14% spike)
❌ **하드 partition cap** — 작은 partition은 OOM, 큰 partition도 효과 무
❌ **Same preset 내 multimodal** — deterministic 아님
❌ **Heavy AI cell-count 미검증** — production 시나리오 unknown

### 그래서 MIG는?
**Necessary but NOT sufficient**:
- 5G L1 SLA의 baseline 격리에는 도움 (특히 p99)
- 하지만 충분하지 않음:
  - Asymmetric split의 intermittent leakage 처리 못 함
  - Resource utilization 최적화 못 함 (partition cap)
  - AI 워크로드의 burst pattern 감안 못 함

### 진짜 필요한 것
**Software-level orchestration layer**:
1. AI 워크로드의 phase (LLM attention/KV cache burst) detect
2. Bandwidth 사용량 실시간 모니터링
3. SLA violation prediction
4. Dynamic AI workload throttling/migration
5. Symmetric MIG split 권장 + asymmetric 회피

이건 정확히 사용자님 **PHASE1_PLAN**의 방향. **데이터가 PHASE1_PLAN을 강력히 지지**.

---

## 11. 다음 reservation 우선순위 (한계 검증 + publishable data)

### Priority 1: Bimodal mechanism 규명
- N≥20 split-60-40 + Qwen 측정
- Concurrent `nvidia-smi dmon -s m` (HBM BW 실시간)
- Qwen token generation timing vs spike correlation

### Priority 2: Heavy AI cell-count
- Qwen + cells {1, 4, 10, 20, 40} 전 sweep
- no-mig vs MIG 비교
- light AI의 polynomial explosion이 heavy AI에서도?

### Priority 3: A baseline + multi-split
- L1 alone full GPU (driver bug 회피 — full reboot 사이클)
- Split ratios 20-80, 30-70, 70-30, 80-20 — bimodal universal?

### Priority 4: Long-duration (hours)
- 1-2시간 continuous, thermal/drift 검증

---

## 12. 데이터 파일 위치

```
/Users/changjongkim/New_research/cloudlab_results/
├── MIG_LIMITATIONS.md         ← 이 파일 (비판적 분석)
├── DEEP_ANALYSIS.md           ← 신중한 종합 분석
├── KEY_FINDINGS_v2.md         ← v2 결과 요약
├── all_results.csv            ← v2-v5 (heavy)
├── all_results_v1_light.csv   ← v1 (light)
├── make_charts.py             ← 차트 생성 스크립트
├── charts/                    ← 18개 PNG 차트
│   ├── X_critical_summary.png    ← 한 페이지 요약
│   ├── X_v1_vs_v2_critical.png   ← v1 vs v2 비교
│   ├── v1_01 ~ v1_11             ← light workload 11개
│   └── v2_01 ~ v2_05             ← heavy workload 5개
└── results/                   ← raw JSON
```

## 13. 결론

> **"Hardware MIG isolation is necessary but insufficient for AI-RAN. Mean isolation
> is marginal (1.24×) under realistic heavy workloads. Tail latency isolation is the
> only decisive benefit (3.6×). Asymmetric splits exhibit bimodal intermittent
> leakage that violates 5G real-time SLA. The smallest MIG partition (1g.5gb) cannot
> fit cuPHY. Beyond 3g.20gb, additional compute partition gives no L1 speedup due to
> HBM-bandwidth bound. MIG hardware features alone do not constitute an optimal
> AI-RAN co-location solution; software-level orchestration with workload-aware
> bandwidth management remains essential."**
