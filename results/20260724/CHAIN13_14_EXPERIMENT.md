# Chain 13 + Chain 14 — 실험 구성 완전 설명

**환경**: CloudLab d8545-10s10505 · A100-SXM4-40GB × 4 · driver 580.173.02 · CUDA 12.8 · pyaerial 25-3 (x86-64 toolchain build)

**세션 목적**: 20260708에서 발견된 두 가지 상반된 결과 —
1. MPS는 compute-bound coloc에서 sync를 완벽히 우회 (`cudaFree` 10.5× → 1.02×)
2. **MPS + HBM stress에서는 오히려 붕괴** (`cudaFree` 22,645 ms, temporal보다 나쁨)
— **이 gap이 실제 AI-RAN 배포 워크로드에서 어떻게 나타나는가?** 답하기 위한 세션.

---

## 배경 — 왜 Chain 13, 14가 필요한가

지금까지 검증된 사실:
- **cross-process implicit sync** (cudaFree, cudaMemcpyAsync)가 L1 latency budget을 파괴
- **temporal multiplex (TS mode)**에서만 발생 — MPS spatial multiplex로 우회 가능
- 하지만 20260708 4번 섹션에서 **MPS + HBM stress → catastrophic breakdown**

**아직 answered되지 않은 것**:
1. MIG cross-partition은 **memory-bound co-tenant에도 완전 격리를 보장하는가?**
   (기존에 compute-bound NRx로만 검증됨)
2. MPS의 **HBM breakdown이 realistic AI 워크로드**(LLM, VLM, ASR)에서도 재현되는가?
   (20260708은 synth STREAM benchmark)

Chain 13이 1을 확인, Chain 14가 2를 확인.

---

## Chain 13 — MIG same-part vs cross-part × 5 real workloads

### 실험 구성

**MIG 프로파일**: A100-40GB 위에 `4g.20gb + 3g.20gb` 두 파티션 생성.
- 4g.20gb: 56 SMs, 20 GB HBM, ~890 GB/s HBM BW (physical isolation)
- 3g.20gb: 42 SMs, 20 GB HBM, ~665 GB/s HBM BW

**측정 대상**: 4g partition에서 돌아가는 `real_l1.py`의 30초 NSYS 프로파일
- 진짜 pyaerial 컴포넌트 (ChannelEstimator, MMSE, LDPC, CRC)
- cells=20, iters=100, PRB=273 (5G NR full BWP)
- 지표: `cudaFree` 총 시간 = temporal sync penalty 크기

**5개 co-tenant workload**:

| # | 워크로드 | 종류 | 실행 |
|---|---|---|---|
| 1 | NRx | Compute (TRT inference) | pyaerial NeuralRx |
| 2 | ChanPred | Compute (torch small transformer) | 소규모 CSI 예측 모델 |
| 3 | Qwen-RAG | Memory (LLM decode w/ RAG prefix) | Qwen-2.5-7B on vLLM |
| 4 | Whisper | Memory (encoder attention + decode) | whisper-large-v3, HuggingFace |
| 5 | Qwen-VL | Memory (ViT + LLM decoder) | Qwen2-VL-7B on HF Transformers |

**cross-partition에는 항상 Qwen-7B baseline LLM 상주** (실전 배포 = idle 파티션 없음).

### Same-partition (SP) 매트릭스 — 11 조건 × 3 trials = 33 캡처

L1과 co-tenant가 **동일 4g 파티션** 안에서 경합. 3g에는 상시 Qwen 상주.

| 조건 | 4g 내용 | 3g 내용 | MPS on 4g |
|---|---|---|---|
| SP-0 | L1 alone | Qwen-7B (baseline) | off |
| SP_{NRx,ChanPred,Qwen-RAG,Whisper,Qwen-VL}_MPSoff | L1 + co-tenant | Qwen-7B | off |
| SP_{...}_MPSon | L1 + co-tenant | Qwen-7B | **on** |

**검증할 가설**:
- H1: NRx MPSoff → cudaFree 10× 폭발 (20260708 재현)
- H2: NRx MPSon → 1.05×로 소멸 (compute-bound 완벽 회복)
- H3: memory-bound MPSon → 부분 잔여 sync (breakdown 시작점 관찰)

### Cross-partition (CP) 매트릭스 — 7 조건 × 3 trials = 21 캡처

L1 alone in 4g, co-tenant를 3g로 이동.

| 조건 | 4g 내용 | 3g 내용 |
|---|---|---|
| CP-idle | L1 alone | idle |
| CP-{workload} | L1 alone | 각 워크로드 6종 (NRx, ChanPred, Qwen-RAG, Whisper, Qwen-VL, Qwen-LLM baseline) |

**검증할 가설**:
- H4: 모든 co-tenant 종류에도 L1은 baseline 수준 유지 → MIG cross-partition = 물리적 완전 격리

**총 54 캡처 × 30s ≈ 27분 profiling** + docker startup ≈ 실측 40분.

### Chain 13 결과 요약

**CP (cross-partition) — 완벽한 격리 확인 ✓**

| 3g workload | L1 cudaFree | vs CP-idle (2,119) |
|---|---:|---:|
| idle | 2,119 ms | 1.00× |
| NRx | 1,844 ms | 0.87× |
| ChanPred | 1,683 ms | 0.79× |
| Qwen-LLM | 1,889 ms | 0.89× |
| Qwen-RAG | 1,995 ms | 0.94× |
| Qwen-VL | 2,051 ms | 0.97× |
| Whisper | 1,675 ms | 0.79× |

→ 3g에 뭘 두든 4g L1은 **~1,800 ms 상수 유지**. H4 확정.

**SP (same-partition) — MPS 효과의 gradient 관찰**

Baseline (SP0, L1 alone + cross-Qwen): **1,838 ms**

| Same-part co-tenant | MPSoff | MPSon | MPS 효과 |
|---|---:|---:|---|
| NRx (compute) | **18,601 ms (10.1×)** | **1,924 ms (1.05×)** | 완벽 회복 ✓ |
| ChanPred (compute) | 11,803 ms (6.4×) | 1,903 ms (1.03×) | 완벽 회복 ✓ |
| Whisper (mem batch=1) | 8,131 ms (4.4×) | **2,117 ms (1.15×)** | ⚠️ 부분 회복 (breakdown 시작 hint) |
| Qwen-VL (mem batch=1) | 1,855 ms (?) | **2,328 ms (1.27×)** | ⚠️ MPS-on이 오히려 나쁨 (MPS overhead > 이득?) |
| Qwen-RAG | 3 ms (invalid) | 3 ms (invalid) | ❌ L1 OOM |

**주요 인식**:
- H1, H2 확정
- H3 **부분 확인** — Whisper 1.15×, VL 1.27× → **20260708 15× catastrophic까지는 못 가지만 gradient 확인**
- Qwen-RAG는 데이터 손실 — Qwen-7B fp16(14GB) + vLLM overhead(2GB) = **16GB가 20GB 4g 파티션 차지 → L1 5GB 여유 없어 ChannelEqualizer OOM**

### Chain 13이 답하지 못한 것

**"진짜 HBM saturating workload"에서 MPS breakdown 크기는?**
- 지금 memory-bound라 부른 것들도 실제 HBM 사용은 3~30% (peak 1,555 GB/s A100)
- Whisper batch=1: ~30 GB/s = 2%
- Qwen-7B decode batch=1: ~420 GB/s = 27%
- 20260708 STREAM stress: ~1,200 GB/s = 77%
- 그래서 Whisper 1.15× 밖에 관찰 안 됨 (20260708의 15×과 큰 차이)

→ **Chain 14가 필요한 이유**

---

## Chain 14 — HBM saturation gradient로 MPS breakdown threshold 확정

### 실험 구성 (Chain 13에서 무엇을 바꾸는가)

1. **Model down-scaling**: Qwen-7B → Qwen-3B, Qwen-VL-7B → Qwen-VL-2B
   - Same-partition에 L1과 함께 넣어도 20GB 4g에 여유롭게 fit
   - Qwen-3B fp16 = 6 GB, VL-2B fp16 = 4 GB
2. **Batch size scale-up**: continuous batching + 병렬 요청
   - Qwen-RAG: `n=64` concurrent sequences (vLLM continuous batching, production LLM 서빙 패턴)
   - Whisper: batch=4 concurrent 30s audio streams
   - Qwen-VL: batch=4 concurrent 1024×1024 이미지
3. **HBM stress control workload 추가**: STREAM triad kernel
   - 20260708 catastrophic MPS breakdown의 ground truth reference

### 4개 workload로 HBM saturation gradient

| Workload | Est. HBM BW | Peak% | 예상 MPS 효과 |
|---|---:|---:|---|
| **HBM_stress (triad)** | ~1,200 GB/s | 77% | Catastrophic (20260708 재현) |
| **Qwen-3B batch=64** | ~700 GB/s | 45% | 큰 부분 붕괴 |
| **Whisper batch=4** | ~120 GB/s | 8% | 부분 붕괴 |
| **VL-2B batch=4** | ~200 GB/s | 13% | 부분 붕괴 |

### Chain 14 매트릭스 (제안)

Chain 13과 동일 구조. 5개 co-tenant를 다음으로 교체:

| # | 워크로드 | 이전 (Ch13) | Chain 14 |
|---|---|---|---|
| 1 | NRx | 그대로 | 그대로 (compute reference) |
| 2 | ChanPred | 그대로 | 그대로 (compute alt) |
| 3 | Qwen-RAG | 7B batch=1 | **3B n=64 continuous** |
| 4 | Whisper | batch=1 | **batch=4 concurrent** |
| 5 | Qwen-VL | 7B batch=1 | **2B batch=4** |
| **6 (new)** | **HBM_stress** | — | **STREAM triad 8 GB, control** |

SP: 13 조건 × 3 = 39 캡처 (baseline 1 + 6 workloads × 2 MPS = 12 조건, +baseline)
CP: 8 조건 × 3 = 24 캡처 (idle + 6 workloads + Qwen-LLM baseline)
**총 63 캡처 × 30s ≈ 32분 profiling** + docker overhead ≈ 실측 45~50분.

### Chain 14가 확정할 것

1. **MPS + memory-bound catastrophic breakdown threshold** — HBM 몇 % 사용에서 MPS가 붕괴하는지
2. **Realistic LLM production 워크로드**(vLLM continuous batching batch=64)가 MPS를 무너뜨리는지
3. **20260708 STREAM stress 재현** — 같은 노드에서 15× breakdown 재현 (driver 580 upgrade 후에도 성립하는지)

### 예상 결과 및 논문 story

Chain 13 + 14 통합해서 다음 그림 완성:

```
MPS overhead protection vs HBM utilization (of 4g slice):

  L1 cudaFree penalty (vs baseline)
  20× ┤                              ●  HBM_stress (Ch14)
      │                            /
  10× ┤                         /
      │                       /
   5× ┤          ●  Whisper batch=4 (Ch14)
      │       /
   2× ┤     ●   Qwen-3B batch=64 (Ch14 pred)
      │  /
   1× ┤●●●●─ NRx, ChanPred (all MPS-on, Ch13)
      └────────────────────────────────────────
       0%   20%   40%   60%   80% HBM saturation
```

이 curve가 그려지면 **"MPS는 workload-conditional한 조건부 solution, HBM 60% 이상에서 SLA 보장 불가"** 가 실측으로 확정. → **MIG cross-partition만이 AI-RAN 배포 안전한 default** 라는 결론이 real workload로 완전 뒷받침됨.

---

## 파일 인벤토리

**Chain 13** (`cloudlab_results/results/20260724/chain13/`, sync 진행중):
- 54 nsys-rep files (~250 MB)
- SP + CP 로그 파일들
- run.log

**Chain 14 준비물 (노드)**:
- `run_qwen_rag_hbm.py` — Qwen-3B n=64 continuous batching
- `run_whisper_hbm.py` — Whisper batch=4
- `run_qwen_vl_hbm.py` — Qwen-VL-2B batch=4
- `run_hbm_stress.py` — STREAM triad control
- 모델 다운로드 완료: Qwen2.5-3B-Instruct, Qwen2-VL-2B-Instruct

**다음 단계**:
1. Chain 13 sync 완료 대기 (백그라운드)
2. Chain 14 runner (`run_chain14.sh`) 작성
3. Chain 14 sanity check (각 워크로드 20s alone verify)
4. Chain 14 매트릭스 실행 (45~50분)
5. Chain 13 + 14 통합 분석 + figures
