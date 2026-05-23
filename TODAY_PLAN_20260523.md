# 오늘 (2026-05-23 6PM ~ 5/24 10AM, 16h) 실험 계획

> reservation window 짧으니까 **최대 효율 우선순위**.
> 가장 publishable한 발견 = **Bimodal mechanism 규명**에 시간 가장 많이 투입.

---

## 1. 큰 그림 — 우리 연구의 흐름

### 연구 질문
> **Hardware MIG는 AI-RAN co-location에서 5G L1 SLA를 지킬 수 있는가?**

### Perlmutter (선행 연구, github.com/changjongkim/AIRAN_Changjong)
- MPS-emulated SM partitioning은 격리 효과 없음 (3.87~4.06× slowdown 그대로)
- 원인: cuPHY가 cell-sync로 SM 경합을 회피 → SM이 병목 아님
- → 하드웨어 MIG (L2 + HBM bandwidth 격리) 검증 필요

### CloudLab Phase 1 (5/12-13, 완료)
- v1 light AI: 환각 11× 격리
- v2 heavy AI (Qwen-7B): 실제 1.24× mean, 3.6× p99
- **Bimodal leakage 발견** (split-60-40 N=4, 50:50, gap 6.7ms)
- A baseline (L1 alone full GPU) 미측정 (driver bug)

### CloudLab Phase 2 (오늘)
- **Bimodal 통계 신뢰도 확보 (N=4 → N=20)**
- **Bimodal 메커니즘 규명** (phase / NoC / arbiter / L2 directory 중 어느 것)
- A baseline 측정 시도 (driver reset 자동화)
- Sensitivity sweep (model size, partition cap 분리 등)

### 최종 publishable thesis (Phase 2 결과에 따라)
- **"MIG isolation: not perfect, but predictable"** 또는
- **"Asymmetric MIG splits exhibit deterministic phase-aligned bimodal leakage"**

---

## 2. 사용 중인 워크로드 — 완전 정리

### 2.1 L1 워크로드 (5G PHY)

**`real_l1.py`** (cloudlab_aerial/real_l1.py)
- cuPHY PUSCH RX 파이프라인 — **component-level 직접 구성**
  - 이유: `PuschRxPipelineFactory`가 25-3-cubb에서 segfault
- 컴포넌트:
  1. **ChannelEstimator** — DMRS 기반 channel estimation
  2. **ChannelEqualizer** — MMSE equalization
  3. **NoiseIntfEstimator** — noise + interference covariance
  4. **LdpcDeRateMatch** — LDPC rate de-matching
  5. **LdpcDecoder** — layered LDPC decode (5 iter)
  6. **CrcChecker** — TB CRC verification

**파라미터** (env vars):
| 변수 | default | 의미 |
|---|---|---|
| `NUM_CELLS` | 20 | per-iter 처리할 cell 수 |
| `NUM_PRBS` | 273 | (full 100MHz BW) |
| `NUM_RX_ANT` | 4 (v1), **8 (v2)** | RX 안테나 수 |
| `NUM_TX_ANT` | 4 (v1), **8 (v2)** | TX 안테나 수 |
| `MCS_INDEX` | 2 | 5G NR MCS table 0의 인덱스 |
| `L1_ITERATIONS` | 50 | 측정 iteration 수 |

**핵심 부하**:
- 8T8R 273 PRB: working set ~80 MB/slot, peak HBM read ~300-400 GB/s
- 1 ms TTI budget → 평균 80-120 GB/s sustained
- cuPHY는 HBM-bandwidth-bound (4g.20gb가 3g.20gb보다 안 빠른 이유)

### 2.2 AI 간섭 워크로드

| 이름 | 스크립트 | 모델/사이즈 | HBM 사용 | 특징 |
|---|---|---|---|---|
| **gpt2** | run_gpt2_stress.py | GPT-2 124M (500MB) | ~1% BW | light, L2 resident, **v1에서만 사용** |
| **resnet** | run_resnet_stress.py | ResNet-50 | ~1.5% BW | light, batch sweep 가능 |
| **hbm** | run_hbm_stress.py | 합성 `dst.copy_(src)` | **~70-100% BW** | deterministic, **non-phased** (A2 검증용 핵심) |
| **qwen7b** | run_qwen7b_stress.py | Qwen2.5-7B fp16 (14GB) | ~22% BW | **v2 핵심 heavy**, prefill+decode 혼합 |
| **qwen7b_prefill** ⭐ | run_qwen7b_prefill.py | Qwen2.5-7B (14GB) | **burst (peak ~40%)** | prefill만, every iter 512-token forward, `use_cache=False`. **H1 phase HIGH trigger** |
| **qwen7b_decode** ⭐ | run_qwen7b_decode.py | Qwen2.5-7B (14GB) | **idle (~5%)** | 한번 prefill 후 1-token씩 incremental decode. **H1 phase LOW trigger** |
| **neuralrx** | run_neural_rx_stress.py | TensorRT Neural RX | low | 이미지 안에 있음, 미사용 |

⭐ = 오늘 추가된 phase-controlled variant (SENSITIVITY_EXPERIMENTS §A1)

### 2.3 MIG presets

| preset | spec | L1 partition | AI partition(s) | 용도 |
|---|---|---|---|---|
| no-mig | full GPU (108 SM) | 7g.40gb | 7g.40gb (공유) | 격리 없는 baseline |
| split-20-80 | 1g+4g | 1g.5gb | 4g.20gb | **항상 OOM** (cuPHY ≥ 5.5GB) |
| split-40-60 | 2g+3g | 2g.10gb | 3g.20gb | partition cap test |
| **split-50-50** | 3g+3g | 3g.20gb | 3g.20gb | **대칭 baseline, 가장 안정** |
| **split-60-40** | 4g+3g | 3g.20gb | 4g.20gb | **bimodal 발생 ⭐** |
| four-way-eq | 2g×3+1g | 1g.5gb | 2g.10gb×3 | 1g 못 씀 → 미사용 |
| four-way-bigL1 | 4g+1g×3 | 4g.20gb | 1g.5gb×3 | L1만 큰 partition |
| seven-1g | 1g×7 | 1g.5gb | 1g.5gb×6 | 전부 OOM |

오늘 주로 쓸 것: **split-60-40** (bimodal 발생), 비교용 **split-50-50, split-40-60**

### 2.4 측정 인프라

- **`nvidia-smi dmon -s mu`** — HBM utilization + memory 사용량 1초 단위. `dmon_sync.sh start/stop/mark`로 sync.
- **N=20 wrapper** (`run_n20.sh`) — 같은 config N회 반복, 각 run JSON 저장.
- **`bimodal_detect.py`** — 1D 2-means clustering, bimodality score, verdict (BIMODAL/AMBIGUOUS/UNIMODAL), histogram + strip plot PNG.
- **`driver_reset.sh`** — services stop → rmmod nvidia_{uvm,modeset,drm} → modprobe → 재시작. A baseline driver bug 우회.
- **`phase1_sweep.sh`** — A0/A1a/A1b/A2 한 번에 ~1.5-2h.

---

## 3. 오늘 실험의 **중점**

### 핵심 메시지
> **"Bimodal leakage가 진짜 존재한다 + 어떤 mechanism에 의해 발생하는가"**

가장 publishable한 발견이 N=4 한정이라 통계적으로 약함. 오늘 N=20으로 확정 + mechanism까지 짚으면 publishable contribution 두 단계 강화됨.

### Bimodal에 대한 4개 가설 → 오늘 실험으로 분리

| 가설 | 약어 | 검증 실험 | 예측 |
|---|---|---|---|
| **H1: Qwen prefill/decode phase alignment** | phase | A1a (prefill만), A1b (decode만) | prefill만 → 거의 항상 HIGH, decode만 → 거의 항상 LOW |
| H2: GPC NoC contention | NoC | A2 (hbm), C3 (SM occupancy) | HBM stress도 bimodal이면 NoC 의심 |
| H3: HBM crossbar arbiter | arbiter | A2 (hbm) | HBM stress도 bimodal이면 arbiter 의심 |
| H4: L2 directory eviction | L2 | C1 (Nsight L2 hit-rate) | 4g가 3g보다 hit-rate 낮으면 L2 의심 |

**H1이 가장 우아한 가설** — confirm되면 "AI-RAN orchestrator는 LLM phase를 모니터링해야 한다"는 명확한 implication.

### 부차적 목표
- **partition cap vs AI leakage 분리** (split-40-60 N=20, L=2g.10gb alone N=20)
- **A baseline 측정** (driver reset 자동화로 우회 시도)
- **heavy AI cell-count sweep** (Qwen × cells {1,4,10,20,40})

---

## 4. 16시간 시간표 (Priority Order)

| 시간 | 작업 | 산출물 | 우선순위 |
|---|---|---|---|
| **0:00 ~ 1:00** | 환경 복구 | docker, MIG enable, Aerial container, scp scripts | 필수 |
| **1:00 ~ 3:00** | **P1: phase1_sweep.sh** (A0/A1a/A1b/A2 N=20 each) | bimodal mechanism 확정 (4개 verdict) | **⭐ 최우선** |
| **3:00 ~ 4:30** | **P2: D1** split-40-60 N=20 + alone N=20 (partition cap 분리) | leakage % 정확 산출 | **High** |
| **4:30 ~ 6:00** | **P2: B1** model size sweep (1B, 3B, 7B, 14B) | isolation factor curve | High |
| **6:00 ~ 10:00** | **L3: heavy AI cell-count** (split-50-50 + Qwen × cells {1,4,10,20,40}, no-mig × cells 동일) | cell saturation point | Mid |
| **10:00 ~ 12:00** | **L1: A baseline** (driver_reset.sh + L1 alone on full GPU) | 진짜 best-case L1 latency | Mid |
| **12:00 ~ 14:00** | **C1: Nsight L2 hit-rate** (3g vs 4g profile) | L2 thrash 가설 검증 | Mid |
| **14:00 ~ 15:00** | **A3: L1 measurement window sweep** | phase sampling artifact 검증 | Low |
| **15:00 ~ 16:00** | rsync + git push + image snapshot save | 모든 결과 영구 저장 | **필수** |

**최소 달성 목표** (시간 부족 시 여기까지만):
1. 0-1h 환경 복구
2. 1-3h P1 (bimodal mechanism)
3. **15:00-16:00 강제로 정리 + 저장** (snapshot 저장 안 하면 모든 work 사라짐)

---

## 5. 오늘 안 다루는 것 (의도적 제외)

- **gpt2/resnet 워크로드 추가 측정** — v1에서 충분, light AI는 misleading
- **multi-AI mix** — workload 1개씩 격리해야 mechanism 추적 가능
- **MCS / PRB / antenna sweep** — v1에서 충분, L1 부하는 8T8R 273 PRB 고정
- **seven-1g, four-way-eq** — 전부 L1 OOM, useless

---

## 6. 사전 점검 (지금 가능)

### scp 올릴 파일 (노드 ready 즉시)
```bash
# cloudlab_aerial/
scp dmon_sync.sh run_n20.sh driver_reset.sh phase1_sweep.sh run_sweep_v2.sh \
    sgkim@<host>:~/cloudlab_aerial/

# AIRAN_Changjong/experiments/
scp ~/New_research/AIRAN_Changjong/experiments/run_qwen7b_prefill.py \
    ~/New_research/AIRAN_Changjong/experiments/run_qwen7b_decode.py \
    sgkim@<host>:~/AIRAN_Changjong/experiments/

# bimodal_detect.py to home
scp ~/New_research/cloudlab_results/bimodal_detect.py sgkim@<host>:~/
```

### 노드 ready 직후 점검 명령
```bash
ssh sgkim@<host> bash -c '"
nvidia-smi
nvidia-smi --query-gpu=mig.mode.current --format=csv
ls /mydata/
docker images | grep aerial
ls ~/cloudlab_aerial/ ~/AIRAN_Changjong/experiments/run_qwen7b_*.py
df -h /
"'
```

### Phase 1 시작 (한 줄)
```bash
ssh sgkim@<host> 'cd ~/cloudlab_aerial && nohup ./phase1_sweep.sh > /tmp/phase1.log 2>&1 &'
```

---

## 7. 성공 기준

### Must-have (오늘 안에 못 하면 reservation 실패)
- [ ] 환경 복구 (MIG 활성화, Aerial container, real_l1.py 동작 확인)
- [ ] **A0 bimodal 재현 N=20** → BIMODAL/UNIMODAL verdict (어느 쪽이든 가치 있음)
- [ ] **A1a prefill / A1b decode N=20** → H1 phase 가설 confirm or reject
- [ ] 종료 전 모든 results rsync + image snapshot 저장

### Stretch goals (시간 남으면)
- [ ] A2 hbm stress N=20 → H2/H3 vs H1 분리
- [ ] D1 split-40-60 + alone → partition cap 분리
- [ ] A baseline 측정 성공
- [ ] B1 model size curve

### Out-of-scope (다음 reservation으로)
- C1 Nsight L2 profiling
- B3 L2 fit boundary 미세조정
- B2 access pattern sensitivity
- 8-hour stability test

---

마지막 업데이트: 2026-05-23 5:55PM
다음 reservation: 6:00 PM 시작, 5/24 10AM 종료 (16시간)
