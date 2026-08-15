# CloudLab Aerial 실험 — 전체 정리 (2026-05-12 ~ 2026-05-13)

> 다음 reservation (5/23 6PM ~ 5/24 10AM) 시작 전 reference 문서.
> 무엇을 했고, 무엇이 publishable하고, 무엇이 빈 곳인지 한 페이지.

---

## 0. 한 줄 요약

A100 하드웨어 MIG로 5G L1 (cuPHY) ↔ AI 워크로드 co-location 격리 효과를 측정.
v1 (light AI) 결과는 잘못된 결론(11× 격리)을 줬고, v2 (heavy AI) 재측정으로 **실제 mean 격리는 1.24×**, **p99만 3.6× 격리**, **비대칭 split에서 bimodal leakage** 발견.

---

## 1. 환경

- 하드웨어: **CloudLab d8545 Wisconsin** (NVIDIA A100-SXM4-40GB × 4, bare-metal)
- 노드: 이전 reservation = `d8545-10s10501.wisc.cloudlab.us`
- OS: Ubuntu 22.04, driver 550, CUDA 12.4
- Aerial SDK: **25-3-cubb** (Perlmutter 결과와 직접 비교 위해 동일 버전)
- L1 워크로드: cuPHY PUSCH 파이프라인 (component-level, PuschRxPipelineFactory가 segfault라 우회)
  - 8T8R 273 PRB MCS 2, 20 cells (default heavy config)
  - 컴포넌트: ChannelEstimator + ChannelEqualizer + NoiseIntfEstimator + LdpcDeRateMatch + LdpcDecoder + CrcChecker
- AI 워크로드:
  - v1: GPT-2 124M (500MB), ResNet-50 (100MB), HBM stress 0.5~16GB
  - v2: **Qwen-7B fp16 (14GB)**, HBM stress 16GB

---

## 2. 두 단계 실험

### Phase v1 — Light AI 워크로드 (5/12 04:00 ~ 08:00)
- **63 OK + 14 FAIL** datapoints (총 77 runs)
- presets: `no-mig, split-40-60, split-50-50, split-60-40, four-way-bigL1`
- AI types: `gpt2, resnet, hbm, multi (gpt2+resnet+hbm)`
- 파라미터 sweep: cells {1,4,10,20,40}, PRB {51,106,217,273}, antenna {2T2R,4T4R,8T8R}, MCS {0,2,7,16,24}, ResNet batch {8,16,32,64}, HBM alloc {0.5,1,2,4,8,16}GB
- **14 FAIL은 모두 1g.5gb partition에 L1 배치 시 OOM** — cuPHY working set >5GB

### Phase v2-v5 — Heavy AI 워크로드 (5/12 13:42 ~ 5/13 종료)
- **8 OK + 10 FAIL** datapoints (재실행 + 환경 복구 시도 포함)
- presets: `no-mig, split-40-60, split-50-50, split-60-40`
- AI types: `qwen7b (14GB), hbm (16GB), none`
- 8T8R 채널추정 부하 4× 증가 → L1 자체 부하가 v1보다 무거움

---

## 3. 핵심 측정 데이터

### 3.1 L1 alone baselines (no AI, 8T8R 20-cell)

| Partition | SM | HBM 용량 | HBM BW (이론) | L1 mean (ms) | L1 p99 (ms) |
|---|---|---|---|---|---|
| 1g.5gb | 1/7 | 5 GB | ~195 GB/s | **OOM** | — |
| 2g.10gb (B2) | 2/7 | 10 GB | ~390 GB/s | 59.27 | 61.80 |
| 3g.20gb (B) | 3/7 | 20 GB | ~775 GB/s | **46.14** | 47.87 |
| 4g.20gb (B4) | 4/7 | 20 GB | ~775 GB/s | 52.77 | 54.31 |
| 7g.40gb (A, full) | 7/7 | 40 GB | 1.55 TB/s | **미측정** | — |

**중요 관찰**:
- 3g.20gb가 **sweet spot** — SM 더 많은 4g.20gb가 더 느림 (HBM-BW bound + L2 thrash)
- 1g.5gb는 cuPHY 자체가 안 들어감 (LdpcDeRateMatch ~2-3GB buffer)
- **A baseline (full GPU L1 alone) 측정 실패** — MIG 모드 toggle 후 driver RPC timeout

### 3.2 L1 + AI on MIG (heavy Qwen-7B)

| Config | L1 partition | AI partition | L1 mean | partition-correct baseline | leakage |
|---|---|---|---|---|---|
| split-50-50 | 3g.20gb | 3g.20gb | 46.47 | B (46.14) | **+0.7%** (대칭, 안정) |
| split-40-60 | **2g.10gb** | 3g.20gb | 66.55 | **B2 (59.27)** | **+12%** (대부분 partition cap) |
| split-60-40 LOW | 3g.20gb | 4g.20gb | 46.60 | B (46.14) | +1% |
| split-60-40 HIGH | 3g.20gb | 4g.20gb | 52.83 | B (46.14) | **+14%** ⚠ |

### 3.3 Split-60-40 + Qwen-7B Bimodal (N=4)

| Run | mean (ms) | mode |
|---|---|---|
| 1 (orig 08:48) | 52.80 | **HIGH** |
| 2 (09:50) | 46.67 | LOW |
| 3 (09:53) | 46.53 | LOW |
| 4 (09:55) | 52.86 | **HIGH** |

→ 두 mode (46.5 / 52.8), gap = 6.66 ms, **50:50 확률**.

### 3.4 L1 + AI on no-mig (shared GPU)

| AI workload | L1 mean | L1 p99 |
|---|---|---|
| Qwen-7B (14GB) | 57.89 | **172.60** (3.6× jitter spike) |
| HBM stress 16GB | 48.03 | 49.91 |

---

## 4. 다섯 가지 발견 (publishable findings)

### Finding 1: MIG mean isolation은 generation gap 따라 다름
- **v1 (light AI)**: no-mig 375ms vs MIG 41ms = **9.1× 격리** ← 잘못된 결론
- **v2 (heavy AI)**: no-mig 57.89ms vs MIG 46.47ms = **1.24× 격리**
- 원인: light AI는 HBM 1% 사용 → SM 경합만 발생 → MIG가 SM 격리해서 효과 큼
- heavy AI는 HBM 22% 사용 → no-mig에서 SM뿐만 아니라 HBM 경합 → MIG는 HBM channel arbiter chip-level 공유라 격리 한계
- **Publishable**: "Real production AI workloads expose MIG's HBM-bandwidth isolation gap"

### Finding 2: Asymmetric split의 Bimodal Leakage ⭐
- **가장 결정적 발견**. NVIDIA MIG whitepaper의 "isolated bandwidth" 주장에 직접 반례.
- split-60-40 (L1=3g, AI=4g) + Qwen → **50% 확률로 +14% leakage**
- gap = 정확히 6.66 ms (연속 분포 아님, 두 점)
- 가설: NoC contention × Qwen prefill/decode phase alignment
- 검증 가능: N≥30 측정 + nvidia-smi dmon HBM BW 동기

### Finding 3: cuPHY는 HBM-bandwidth-bound (3g.20gb sweet spot)
- 3g (42 SM, 775 GB/s) → 46.14 ms
- 4g (56 SM, 775 GB/s) → 52.77 ms (SM 33% ↑ 했는데 6.6 ms ↓ **반대 방향**)
- 원인: HBM BW 동일 + L2 thrash (4 slice도 working set 80MB 대비 부족)
- 즉 cuPHY는 partition 키워도 안 빨라짐 → MIG profile catalog가 cuPHY-친화적 아님

### Finding 4: 1g.5gb는 cuPHY에 쓸 수 없음
- 14 FAIL 모두 L1=1g.5gb 배치
- LdpcDeRateMatch 단독으로 ~2-3GB HBM 필요 + 전체 working set ~5.5GB
- **A100 7-instance density 시나리오는 5G L1 co-location에 불가능**

### Finding 5: MIG의 진짜 가치 = p99 jitter 격리 (3.6×)
- no-mig + Qwen p99 = **172.60 ms** (5G TTI 1ms 대비 3.45 ms/TTI = 명백한 SLA violation)
- MIG 50:50 p99 = 48.29 ms (0.97 ms/TTI = 마진 3%)
- mean보다 tail이 훨씬 더 큰 isolation 효과 → **real-time SLA 관점에서 MIG는 여전히 valuable**

---

## 5. 한계 / 미해결

| # | 이슈 | 영향 | 다음 reservation 우선순위 |
|---|---|---|---|
| L1 | **A baseline (L1 alone full GPU) 미측정** | 모든 leakage % 진짜 baseline 대비 재계산 못 함 | High |
| L2 | **Bimodal N=4 → 통계적으로 약함** | 50:50 확률 추정의 신뢰구간 넓음 | **Highest** |
| L3 | **Heavy AI cell-count sweep 미측정** (c=1만 있음) | Qwen에서 cell-count saturation 발생하는지 모름 | High |
| L4 | **ResNet/Multi-AI on heavy L1 미측정** | workload type 영향 비교 불가 | Mid |
| L5 | **split-40-60 N=1** | "작은 partition 안 좋다" 결론의 근거 약함 | Mid |
| L6 | **Long-duration (hours) 미검증** | drift/thermal/power throttling 영향 모름 | Low |

---

## 6. 다음 reservation 계획 (5/23 6PM ~ 5/24 10AM = 16시간)

### 시작 환경
- 저장한 이미지 (5/12 4:44PM, 9.4GB) 부팅 — driver/Docker/repos 살아있음
- 새로 해야 할 것: Aerial container pull (/mydata에 있었으면 보존), MIG 활성화 (재부팅 시 풀림)

### 시간 배분 (16시간)

| 시간대 | 작업 | 목표 |
|---|---|---|
| 0:00 ~ 1:00 | 환경 복구 | docker, MIG, Aerial container, cuPHY build 잔존 확인 |
| 1:00 ~ 4:00 | **L2: N=20 bimodal 검증** | split-60-40 + Qwen N=20 + nvidia-smi dmon 동기 |
| 4:00 ~ 5:00 | split-50-50 + Qwen N=20 대조군 | bimodal이 비대칭에만 있는지 확인 |
| 5:00 ~ 9:00 | **L3: heavy AI cell-count sweep** | Qwen × cells {1,4,10,20,40} 양쪽 (MIG, no-mig) |
| 9:00 ~ 11:00 | **L1: A baseline 측정 시도** | driver bug 우회 (전체 reboot + module reload 시퀀스) |
| 11:00 ~ 13:00 | L5: split-40-60 N=10 재측정 | partition cap vs leakage 분리 |
| 13:00 ~ 15:00 | L4: ResNet/Multi-AI on heavy L1 | workload type sensitivity |
| 15:00 ~ 16:00 | 결과 정리 + git push + image snapshot 저장 |

### 측정 인프라 사전 점검 (지금 가능)
- `run_sweep_v2.sh` — 18-phase comprehensive
- `focused_heavy.sh` — heavy workload만 빠르게
- `master_sweep_v2.sh` — MIG 한 번 토글 + 18 phase
- `real_l1.py` — cuPHY 8T8R 273 PRB heavy

새로 추가 필요:
- **`nvidia-smi dmon -s m` 동기 측정 wrapper** (bimodal 원인 규명)
- **N=20 반복 측정 + JSON aggregation** (현재는 N=1~4)
- **A baseline driver reset 자동화** (`rmmod nvidia_uvm nvidia_modeset nvidia_drm + modprobe`)

---

## 7. 파일 인덱스 (cloudlab_results/)

### 분석 문서
- `MASTER_SUMMARY.md` — **이 파일** (전체 정리)
- `KEY_FINDINGS_v2.md` — v2 결과 요약 (heavy workload 중심)
- `MIG_LIMITATIONS.md` — 비판적 분석 (570줄, 메커니즘 깊이 보강됨)
- `DEEP_ANALYSIS.md` — 5개 core finding 종합 분석

### 데이터
- `all_results_v1_light.csv` — 77 runs (5/12 light workload)
- `all_results.csv` — 18 runs (v2-v5 heavy workload)
- `results/`, `20260512_*/` — raw JSON + 로그 디렉토리

### 차트 (20개, charts/)
- `v1_01_cell_scaling_4presets.png` — cell-count linear vs polynomial
- `v1_02 ~ v1_11` — light workload sub-parameter sweeps
- `v2_01_partition_baselines.png` — partition별 L1 alone (mean + p99)
- `v2_02_splits_with_qwen.png` — MIG split 비교
- `v2_03_bimodal_detail.png` — N=4 bimodal 시각화
- `v2_04_mig_vs_nomig_heavy.png` — mean 1.24× vs p99 3.6×
- `v2_05_percentile_fan.png` — p50/p95/p99 fan
- `v2_06_coverage_matrix.png` — v2 데이터 커버리지 (대부분 빈 셀)
- `v2_07_all_splits.png` — 모든 v2 datapoint scatter
- `v2_08_partition_aware_leakage.png` — **partition-correct baseline 기준 leakage**
- `v2_09_keyfindings_summary.png` — KEY_FINDINGS 3 표를 한 차트
- `X_v1_vs_v2_critical.png` — v1 mislead vs v2 reality
- `X_critical_summary.png` — 종합 한 페이지

### 스크립트 (cloudlab_aerial/, repo 별도)
- `00_bootstrap.sh, 01_aerial.sh, 02_mig.sh, 03_workloads.sh, Dockerfile.airan`
- `master_sweep_v2.sh, run_sweep_v2.sh, focused_heavy.sh, final_v5.sh`
- `real_l1.py` — component-level cuPHY 벤치마크
- `synthetic_l1.py` — 백업용 합성 워크로드 (사용 안 함, real_l1 우선)

### Git
- GitHub: https://github.com/changjongkim/airan_cloudlab
- 최신 커밋: `5cd30d8` (partition-aware leakage 차트 + MD 정정)
- 모든 데이터/문서/차트/스크립트 push 완료

---

## 8. Publishable 메시지 후보

1. **"Hardware MIG isolation: not perfect, but predictable"** (현재 thesis)
   - 1.24× mean, 3.6× p99, bimodal asymmetric leakage
2. **"Real production AI exposes MIG's HBM-bandwidth isolation gap"**
   - GPT-2 → Qwen-7B 전환 시 격리 효과 9× → 1.24×
3. **"Bimodal leakage in asymmetric MIG splits — novel finding"**
   - NVIDIA whitepaper 마케팅 vs 실측 불일치
4. **"MIG is necessary but insufficient for AI-RAN — software orchestration still needed"**
   - PHASE1_PLAN 방향성을 데이터가 지지

각 메시지의 강도는 다음 reservation 결과(특히 N=20 bimodal 재현)에 달림.

---

마지막 업데이트: 2026-05-23 5:30PM
다음 reservation: 2026-05-23 6:00PM ~ 2026-05-24 10:00AM (16시간)
