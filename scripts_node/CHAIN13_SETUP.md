# Chain 13 셋업 — MIG same-part vs cross-part × 5 co-tenant workloads

**Target session**: 2026-07-22 CloudLab d8545 (예정)
**목표**: MIG same-partition에서 memory-bound co-tenant일 때 MPS의 붕괴 재현 + MIG cross-partition의 물리적 격리 검증. Cross-part는 항상 Qwen LLM 상주 (실전 배포 조건).

---

## 1. 실험 매트릭스

**MIG 프로파일**: `4g.20gb + 3g.20gb` (기존 20260708과 동일, A100-40GB에서 유효)

### Same-partition (4g에 L1 + co-tenant, 3g에는 Qwen 항상)

| 조건 | 4g content | 3g content | MPS on 4g |
|---|---|---|---|
| SP-0 | baseL1 alone | Qwen-2.5-7B | off |
| SP-1a / SP-1b | baseL1 + **NRx** (compute) | Qwen-2.5-7B | off / **on** |
| SP-2a / SP-2b | baseL1 + **ChanPred** (compute) | Qwen-2.5-7B | off / **on** |
| SP-3a / SP-3b | baseL1 + **Qwen-RAG** (memory) | Qwen-2.5-7B | off / **on** |
| SP-4a / SP-4b | baseL1 + **Whisper** (memory) | Qwen-2.5-7B | off / **on** |
| SP-5a / SP-5b | baseL1 + **Qwen2-VL** (memory) | Qwen-2.5-7B | off / **on** |

11 conditions × 3 trials = **33 캡처**

### Cross-partition (4g에는 L1 alone, 3g에 각 워크로드)

| 조건 | 4g content | 3g content |
|---|---|---|
| CP-0 | baseL1 alone | idle |
| CP-1 | baseL1 alone | NRx |
| CP-2 | baseL1 alone | ChanPred |
| CP-3 | baseL1 alone | Qwen-RAG |
| CP-4 | baseL1 alone | Whisper |
| CP-5 | baseL1 alone | Qwen2-VL |
| CP-6 | baseL1 alone | Qwen-2.5-7B (baseline LLM) |

7 conditions × 3 trials = **21 캡처**

MPS는 cross-part에서 무의미 (4g에 L1만 있으므로 spatial multiplex할 대상 없음).

**총 54 캡처 × 30 s ≈ 27분 profiling** + 컨테이너 startup + MIG 재구성 시간

---

## 2. 워크로드 정의

| 워크로드 | Bound | 모델 / 엔진 | 컨테이너 | 크기 (fp16) | 4g/3g fit |
|---|---|---|---|---|---|
| **baseL1** | compute | real cuPHY (pyaerial) | `airan:25-3-final` | — | ✓ |
| **NRx** | compute | pyaerial NeuralRx TRT | `airan:25-3-final` | small | ✓ |
| **ChanPred** | compute | pyaerial AI-ChEst (`chest_trt.yaml`) | `airan:25-3-final` | small | ✓ |
| **Qwen-RAG** | **memory** | Qwen-2.5-7B + FAISS retrieval | `vllm/vllm-openai:v0.6.6` | 14 GB + 1 GB | ✓ (4g) |
| **Whisper** | **memory** | openai/whisper-large-v3 | HF Transformers 컨테이너 | 3 GB | ✓ |
| **Qwen2-VL** | **memory** | Qwen/Qwen2-VL-7B-Instruct | HF Transformers 컨테이너 | 14 GB | ✓ (4g) |
| **Qwen-baseline** (LLM) | **memory** | Qwen-2.5-7B (RAG 없이 pure decode) | `vllm/vllm-openai:v0.6.6` | 14 GB | ✓ (3g) |

Qwen 통일 이유: (1) Apache 2.0으로 gate 없음, (2) HF/vLLM/TRT-LLM 모두 지원, (3) 7B fp16이 20GB MIG 슬라이스에 딱 맞음, (4) NVIDIA 공식 지원 (Qwen3 통합 블로그).

---

## 3. 사전 셋업 (오늘 실험 시작 전에 미리)

### 3.1 컨테이너 사전 pull (`prepull_containers.sh`)

```bash
docker pull vllm/vllm-openai:v0.6.6
docker pull nvcr.io/nvidia/pytorch:25.06-py3    # Whisper, VLM용
```
`airan:25-3-final`는 이미 노드에 존재.

### 3.2 모델 가중치 사전 다운로드 (`prepull_models.sh`)

`/mydata/hf_cache/`에 다음 모델 사전 다운로드:
- `Qwen/Qwen2.5-7B-Instruct` (14 GB)
- `Qwen/Qwen2-VL-7B-Instruct` (14 GB)
- `openai/whisper-large-v3` (3 GB)

총 **~31 GB 디스크** 필요. `/mydata`는 CloudLab 노드에 넉넉함 (100+ GB).

### 3.3 GPU 환경 사전 조정

기존 chain12 스크립트 그대로:
- Persistence: `sudo nvidia-smi -pm 1`
- Clocks: `sudo nvidia-smi -i 0 -ac 1215,1410`
- Hugepages: `sudo sh -c 'echo 8192 > /proc/sys/vm/nr_hugepages'`

### 3.4 MIG 프로파일 생성 (실험 시작 시)

```bash
sudo nvidia-smi -i 0 -mig 1
sudo nvidia-smi mig -i 0 -cgi 4g.20gb,3g.20gb -C
# 각 인스턴스의 UUID를 캡처해서 CUDA_VISIBLE_DEVICES에 사용
nvidia-smi -L
```

---

## 4. 워크로드 실행 스크립트 (개별)

각 워크로드는 아래 pattern으로 `--duration_s N` 인자를 받아 지정 시간 continuous inference. 각 컨테이너를 특정 MIG UUID에 `--gpus 'device=<UUID>'`로 pin.

파일 (`cloudlab_aerial/` 아래에 새로 작성):

| 파일 | 워크로드 |
|---|---|
| `run_qwen_llm.py` | vLLM Qwen-2.5-7B pure decode (RAG 없음, cross-part 및 baseline용) |
| `run_qwen_rag.py` | Qwen-2.5-7B + FAISS retrieval loop (same-part memory-bound) |
| `run_whisper.py` | Faster-Whisper streaming ASR (30s 오디오 반복) |
| `run_qwen_vl.py` | Qwen2-VL image + prompt inference loop |
| `run_chanpred.py` | pyaerial ChannelEstimator with `chest_trt.yaml`, standalone loop |

각 스크립트는 stdout에 iteration count를 flush해서 nsys 프로파일 밖에서 sanity 확인 가능.

---

## 5. Matrix runner (`run_chain13.sh`)

구조는 `run_chain12.sh` 확장:

```bash
# for each (partition_mode, workload, mps_mode, trial):
#   1. Set MIG partitions (if changed)
#   2. Start co-tenant container on correct partition (background)
#   3. Warmup wait (LLM 특히 필요, 15s)
#   4. nsys profile on L1 for 30s
#   5. Kill co-tenant
#   6. Save to /out/chain13/<label>.nsys-rep
```

MIG 파티션 변경마다 모든 container 종료 → MIG 재구성 → 재시작이 필요 (이 부분 chain12보다 복잡).

---

## 6. Sanity checks (사전에 각 워크로드 alone 실행)

CloudLab 접속 후 실험 시작 전에:

```bash
# 각 워크로드가 30s 동안 실제로 inference를 돌리는지 확인
./sanity_check.sh qwen_llm       # 예상: >100 tokens generated
./sanity_check.sh qwen_rag       # 예상: >50 responses
./sanity_check.sh whisper        # 예상: >5 audio clips
./sanity_check.sh qwen_vl        # 예상: >10 image queries
./sanity_check.sh chanpred       # 예상: >1000 inferences
./sanity_check.sh nrx            # 이미 검증됨
```

각 워크로드가 MIG 파티션 안에서 정상 동작하는지, HBM 초과 안 나는지 검증.

---

## 7. 잠재적 blocker 및 대응

| Blocker | 대응 |
|---|---|
| Qwen fp16 모델이 MIG 3g에서 KV cache 포함하면 20 GB 초과 | `--gpu-memory-utilization 0.85`로 제한, 또는 Qwen-2.5-3B로 다운스케일 |
| vLLM 컨테이너 startup 15초 이상 | warmup 시간 20초로 확대 |
| MIG 재구성 실패 (기존 chain11 issue) | reboot 후 다시 시도, 또는 GPU 1로 이전 |
| pyaerial ChannelEstimator SIGSEGV (Chain 12 Approach A 유사) | driver 570 upgrade 필요 — 이번엔 upgrade 후 실행 |
| HF Hub rate limit | 모델 사전 다운로드로 회피 |
| Container 30 GB 이미지 pull 오래 걸림 | 사전 pull |

---

## 8. 오늘 실험 시작 전 사용자 결정 필요

1. **Driver 570 upgrade 진행?** — pyaerial ChanPred 위해 필요. 아니면 ChanPred는 별도 TRT 모델로 대체
2. **디스크 여유** — `/mydata`에 최소 40 GB 필요 (models + Docker layers)
3. **HuggingFace token 필요 없음** — Qwen은 gate 없음. 확인만 하면 됨
4. **cross-part LLM은 32K context 필요한가?** — 기본 4K로 시작 (KV cache 최소화)

---

## 9. 파일 인벤토리 (커밋 예정)

- `CHAIN13_SETUP.md` — 이 문서
- `prepull_containers.sh` — 컨테이너 pull
- `prepull_models.sh` — 모델 다운로드
- `run_qwen_llm.py` — Qwen pure LLM
- `run_qwen_rag.py` — Qwen + FAISS RAG
- `run_whisper.py` — Whisper ASR loop
- `run_qwen_vl.py` — Qwen2-VL inference
- `run_chanpred.py` — pyaerial AI-ChEst standalone
- `run_chain13.sh` — matrix runner
- `sanity_check.sh` — per-workload standalone verifier

다음 단계: 이 파일들 실제 작성 후 커밋.
