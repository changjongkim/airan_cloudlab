# Sensitivity Experiments — MIG의 진짜 원인 규명

> 관찰된 anomaly (bimodal, mean isolation collapse, partition cap)에 대한 **causal attribution**.
> 각 실험은 **한 가지 가설**만 isolate해서 evidence를 만들어냄.

각 실험에는:
- **가설** (어떤 원인을 검증)
- **변인 manipulation** (무엇만 바꿈)
- **예측** (가설 맞으면 어떻게 보일지)
- **반증 조건** (어떤 결과면 가설 reject)

---

## A. Bimodal Leakage 원인 규명 (가장 중요)

현재 가설 4개 후보:
- H1 — **Qwen phase alignment**: prefill (HBM read burst) vs decode (idle) 가 L1 sampling window와 alignment 여부
- H2 — **NoC contention**: 4g instance의 packet 생성률이 3g의 4/3배 → NoC saturate
- H3 — **HBM crossbar queue**: chip-level memory controller arbiter 큐잉
- H4 — **L2 directory eviction**: 4g의 큰 working set이 directory 흔듦

### Exp A1 — Qwen phase 분리 (H1 검증, ⭐ 최우선)

**조작**: Qwen workload를 두 가지로 분리해서 split-60-40에서 N=20씩 측정
- A1a: **Prefill-only** — 1024 토큰 context, 새 토큰 생성 1개만 (KV cache 초기화 후 prefill 후 stop)
- A1b: **Decode-only** — 사전에 prefill 끝낸 KV cache 로드 후 1 토큰씩 generation 반복

**예측 (H1 맞으면)**:
- A1a (prefill burst): **L1 HIGH 모드 100%** (50% 아님)
- A1b (decode idle): **L1 LOW 모드 100%**
- → bimodal이 deterministic하게 갈림

**반증 조건**:
- 둘 다 50:50 bimodal → phase 아니라 다른 원인
- 둘 다 HIGH → 단순히 Qwen 메모리 활성도가 원인 (H2/H3가 더 가능성)
- 둘 다 LOW → 측정 artifact

**필요 시간**: 1 시간 (각 30분)

### Exp A2 — Non-phased workload로 split-60-40 (H1 vs H2/H3 분리)

**조작**: Qwen 대신 **HBM stress (deterministic, no phase)** + **GPT-2 (light, no HBM)** 를 split-60-40 4g 위에 올림

**예측**:
- H1 (phase) → HBM stress는 **bimodal 없음** (deterministic)
- H2 (NoC) / H3 (arbiter) → HBM stress도 **bimodal 발생** (NoC/arbiter는 phase-agnostic)
- GPT-2 → 둘 다 light라 leakage 작아야 함

**관측 매트릭스**:
| Split-60-40 + ... | bimodal? | leakage mean |
|---|---|---|
| Qwen (현재 데이터) | ✓ (50:50) | +7.7% |
| HBM stress | ? | ? |
| GPT-2 | ? | ? |

**반증 조건**: HBM stress가 bimodal이면 H1 reject, H2/H3 채택

**필요 시간**: 1 시간

### Exp A3 — L1 measurement window 길이 sweep (H1 검증 보강)

**조작**: L1 iteration 수를 {10, 50, 200, 1000} iters로 변경. window 길이 = iter × ~1ms.

**예측 (H1 맞으면)**:
- 짧은 window (10 iter = ~10ms): bimodal more pronounced (phase에 정확히 alignment)
- 긴 window (1000 iter = ~1s): bimodal **사라짐** (여러 phase 평균화)

**반증 조건**: window 길이와 무관하게 bimodal 유지 → H1 reject

**필요 시간**: 30분

### Exp A4 — Qwen context length / batch sweep (H1 phase period 검증)

**조작**: split-60-40에서 Qwen의 context length를 {128, 512, 2048, 8192} 토큰으로 변경
- context 클수록 prefill burst 더 김
- → phase period 늘어남

**예측 (H1 맞으면)**:
- 짧은 context: 빠른 phase oscillation → bimodal pronounced
- 긴 context: 한 phase가 측정 윈도우 전체 차지 → bimodal 약화, 대신 일관된 HIGH 또는 LOW

**필요 시간**: 1 시간

---

## B. Mean Isolation Collapse 원인 규명

현재 가설: heavy AI는 HBM BW 22% 사용 → MIG가 HBM channel arbiter chip-level shared

### Exp B1 — Model size monotonic sweep (HBM BW dependence 검증)

**조작**: split-50-50 + AI workload, AI 모델 크기를 다양화:
- 0.5 GB (GPT-2 124M)
- 3 GB (Qwen-1.8B)
- 7 GB (Llama-3B fp16)
- 14 GB (Qwen-7B fp16)
- 28 GB (Qwen-14B fp16) — 들어갈까? 3g.20gb는 20GB max — split-50-50으로는 14GB가 최대
  - 대안: no-mig에서만 28GB 측정

**예측**: isolation factor가 모델 크기에 따라 monotonic decrease
- 0.5GB: 9× (v1과 일치)
- 14GB: 1.24× (v2와 일치)
- 중간 데이터로 curve 그릴 수 있음

**반증 조건**: monotonic 아님 → 다른 변수 존재 (L2 fit/miss boundary?)

**필요 시간**: 2 시간 (각 30분)

### Exp B2 — HBM access pattern sensitivity (cache-friendly vs random)

**조작**: split-50-50 + HBM stress workload. 동일한 16GB 할당 + 동일한 BW 사용. 하지만 access 패턴만 변경:
- B2a: Sequential `dst.copy_(src)` (cache-friendly)
- B2b: Strided (stride 64KB)
- B2c: Random (uniform random indices)

**예측**: pattern이 random일수록 leakage 증가 → MIG 격리가 access pattern에 sensitive
- 만약 그렇다면, "MIG isolates capacity, not pattern" 라는 추가 finding

**필요 시간**: 1 시간

### Exp B3 — L2 fit boundary 측정

**조작**: 모델/workload 크기를 L2 cache (40 MB) 주변 미세조정:
- 32 MB (fit fully)
- 40 MB (fit boundary)
- 48 MB (slight overflow)
- 80 MB (full overflow, HBM-bound)

**예측**: 40MB 이하 = MIG 격리 잘 작동 (L2 partitioned), 40MB 초과부터 isolation 급감
- 이 boundary가 명확하면 "L2 fit는 deterministic MIG, L2 overflow는 stochastic" 라는 깔끔한 결론

**필요 시간**: 1.5 시간

---

## C. 4g < 3g Anomaly 원인 규명 (Finding 3)

현재 가설: L2 thrash / SM under-occupancy / NoC

### Exp C1 — L2 hit-rate Nsight profiling

**조작**: 3g.20gb 와 4g.20gb 각각에서 cuPHY L1을 Nsight Compute로 단일 kernel profile
- target metric: `l2_tex__t_sector_hit_rate`, `lts__t_sector_hit_rate`
- 동일 kernel (ChannelEstimator) 비교

**예측**:
- 4g가 3g보다 **L2 hit-rate 낮으면** → L2 thrash 가설 confirm
- 비슷하면 → 다른 원인 (SM under-occupancy 또는 NoC)

**필요 시간**: 1 시간 (Nsight 설치 + profile)

### Exp C2 — cuPHY workload size scaling

**조작**: L1 working set을 줄여서 L2 안에 들어가게:
- PRB 51 (작음), 8T8R → working set ~25 MB
- PRB 273 (full), 8T8R → working set ~80 MB

**예측**:
- 작은 working set: 3g, 4g 모두 비슷 (L2 hit-rate 둘 다 높음)
- 큰 working set: 4g가 3g보다 느려져야 함 (L2 thrash 발생)

**필요 시간**: 30분

### Exp C3 — SM occupancy 측정

**조작**: Nsight metric `sm__warps_active.avg.pct_of_peak_sustained_active` 측정
- 3g (42 SM) 와 4g (56 SM) 각각

**예측**:
- 4g에서 occupancy <70% → SM under-occupancy 가설 confirm

**필요 시간**: 30분

---

## D. Limitation L5 — split-40-60 재측정 (사용자 요청)

### Exp D1 — split-40-60 + Qwen N=20

**현재 데이터**: N=1, 66.55 ms (B2 baseline 59.27 → +12% leakage)

**의문점**:
- N=1이라 noise일 수 있음
- 12%가 partition cap 효과인지, AI leakage인지 분리 안 됨

**측정 계획**:
- D1a: split-40-60 + Qwen N=20 → 평균 + 표준편차
- D1b: split-40-60 + **none** (L1 alone on 2g.10gb with 3g.20gb 비어있음) N=20 → 진짜 partition cap 단독 효과
- → leakage = D1a - D1b (deterministic 분리)

**예측**:
- D1b ≈ B2 (59.27) → partition cap만으로 13ms 증가 설명됨
- D1a - D1b → 진짜 AI leakage (작을 것)

**필요 시간**: 1.5 시간

### Exp D2 — split-40-60 + HBM/ResNet/GPT-2 (workload sensitivity)

**조작**: split-40-60에서 AI workload만 변경
- HBM 16GB
- ResNet bs=64
- GPT-2

**예측**: 만약 workload 무관하게 비슷 → partition cap dominant. workload 따라 다르면 → AI leakage 비중 큼.

**필요 시간**: 1 시간

---

## E. 종합 — 시간 vs 가치

### 다음 reservation (16h)에서 우선순위

| Priority | Exp | 시간 | 어떤 finding 강화 |
|---|---|---|---|
| **P1** | A1 (Qwen phase 분리) | 1h | Bimodal mechanism 확정 |
| **P1** | A2 (non-phased AI on split-60-40) | 1h | H1 vs H2/H3 분리 |
| **P1** | N=20 split-60-40 + Qwen (기본 재현) | 1h | Bimodal 통계 신뢰도 |
| **P2** | D1 (split-40-60 N=20 + alone) | 1.5h | partition cap vs AI leakage 분리 |
| **P2** | B1 (model size sweep) | 2h | mean isolation curve |
| **P3** | C1 (Nsight L2 hit-rate) | 1h | 4g<3g 원인 확정 |
| **P3** | A3 (window length sweep) | 30m | bimodal sampling 검증 |
| **P3** | A4 (context length sweep) | 1h | phase period 검증 |
| **P4** | C2, C3 (cuPHY scaling, SM occupancy) | 1h | secondary |
| **P4** | B2 (access pattern) | 1h | secondary |
| **P5** | B3 (L2 fit boundary) | 1.5h | additional finding |

**16시간 추천 plan**:
- 0-1h: 환경 복구
- 1-3h: P1 세 개 (bimodal 메커니즘 확정 — 가장 publishable)
- 3-5h: P2 (D1, B1 part 1)
- 5-7h: C1 + Nsight 셋업
- 7-12h: heavy AI cell-count (Master Plan §6의 4-8h 작업)
- 12-14h: A baseline 시도 + B1 나머지
- 14-16h: 정리 + push + snapshot

---

## F. 새 측정 인프라 필요 (지금 작성 가능)

### F1 — Qwen phase 제어 스크립트
```python
# qwen_prefill_only.py
# prefill 만 돌리고 generate(max_new_tokens=1) — 1 토큰만 만들고 stop
# 반복문 안에서 model.generate() 호출

# qwen_decode_only.py
# 한 번 prefill 한 KV cache를 저장
# 이후 past_key_values 인자로 1 토큰씩 generate
```

### F2 — nvidia-smi dmon 동기 logger
```bash
# 백그라운드로 nvidia-smi dmon -s mu -d 1 -o T > dmon.csv 띄우고
# L1 측정 시작/종료 timestamp 기록
# 사후에 dmon.csv와 L1 latency timestamp join
```

### F3 — N=20 자동 반복 wrapper
```bash
# focused_heavy.sh를 N번 반복, 각 run 결과를 N_run_<idx>.json으로 저장
# 마지막에 통계 (mean, std, p99, bimodal cluster detection)
```

### F4 — Driver reset 자동화 (A baseline 위해)
```bash
# rmmod nvidia_uvm nvidia_modeset nvidia_drm
# wait 5s
# modprobe nvidia_drm nvidia_modeset nvidia_uvm
# nvidia-persistenced --user root
# docker restart
```

이 4개는 지금 (노드 ready 전) 미리 작성해두면 6PM부터 바로 돌릴 수 있음. 만들까?
