# Perlmutter no-MIG Experiment Handoff — Briefing for executing Claude

작성일: 2026-06-05
대상: Perlmutter (NERSC)에서 본 repo를 가지고 실험을 진행할 Claude 인스턴스
상위 컨텍스트: `results/visual_evidence/MIG_AIRAN_VISUAL_EVIDENCE_KR.md`

---

## 0. 너의 임무 한 줄

> **CloudLab d8545 노드에서 MIG로 측정한 모든 핵심 실험을 Perlmutter A100에서 *MIG 끈 상태로* 동일하게 재측정해서, "MIG 없으면 어떻게 되는가"의 비교 데이터를 만들어라.**

이게 왜 필요한가: 현재 우리 paper는 "MIG는 AI-RAN에 부족하다"라고 주장. Reviewer가 "그럼 MIG 안 쓰면 더 나아? 그냥 full GPU 쓰면 되잖아"라고 물을 텐데, 답할 데이터가 없음. 너의 측정 결과로 "MIG가 필요하긴 한데 부족하다"라는 paper의 가장 강한 framing이 완성됨.

---

## 1. 가장 먼저 확인할 것 — 환경 점검

CloudLab과 Perlmutter는 환경이 많이 다르다. 무엇이 가능한지부터 확인.

```bash
# 1) GPU + MIG 상태 (login node에서는 GPU 안 보일 수도 있음. 그러면 salloc 후 확인)
nvidia-smi --query-gpu=name,driver_version,mig.mode.current --format=csv 2>&1

# 2) SLURM 사용 가능 QoS / partition
sinfo -p gpu --Format=PartitionName,Available,Cpus,Gres 2>&1 | head -10
sacctmgr show qos format=name 2>&1 | head -20

# 3) Container runtime
which shifter podman docker 2>&1
shifterimg images 2>&1 | head -5

# 4) 모듈
module avail cuda 2>&1 | head -5
module avail nsight 2>&1 | head -5

# 5) 작업 경로
echo "SCRATCH=$SCRATCH HOME=$HOME CFS=$CFS"
df -h $SCRATCH 2>/dev/null | tail -1

# 6) 우리 repo 위치 확인
find $SCRATCH $HOME -maxdepth 3 -name "airan_cloudlab" -type d 2>/dev/null | head -3
```

이 결과를 사용자에게 보고하고, 사용자가 ok 사인 주면 다음 단계로.

---

## 2. 핵심 차이점 — CloudLab vs Perlmutter

| | CloudLab (원본) | Perlmutter (너) |
|---|---|---|
| GPU 접근 | 직접 (sudo 권한) | SLURM (`salloc` / `sbatch`) |
| MIG 토글 | `sudo nvidia-smi -mig 1` | 못 함. 노드가 enable/disable 정책 따름. **우리는 MIG OFF 상태가 필요**. |
| Container | Docker | Shifter (`shifter --image=...`) |
| Image | `airan:25-3-final` (local build) | Shifter pull: `nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb` |
| sudo | 가능 | 불가 |
| 영구 저장 | `/users/sgkim/cloudlab_aerial/results/` | `$SCRATCH` (성능 좋음, 28일 purge) 또는 `$CFS` (영구) |

**중요**: Perlmutter A100 노드가 MIG 켜진 상태이면 우리가 못 끔. 그러면 두 가지 옵션:
1. NERSC support에 MIG OFF 노드 요청 (시간 걸림)
2. 그 노드의 단일 MIG instance를 full GPU처럼 사용 (덜 깔끔)

**가장 흔한 경우는 MIG가 처음부터 꺼져있음** (Perlmutter는 MIG를 일반 사용자에게 노출 안 함). 그러면 그대로 진행.

---

## 3. Setup 단계

### 3.1. Shifter image 가져오기

```bash
# Aerial base image (cuPHY 포함, Perlmutter용 25-3-cubb 버전)
shifterimg pull nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb 2>&1 | tail -3

# 결과 확인
shifterimg images 2>&1 | grep aerial
```

만약 인증 에러 나면 NERSC NGC 등록 가이드 따라야 함: <https://docs.nersc.gov/development/containers/shifter/how-to-use/#using-ngc-containers-from-nvidia>

### 3.2. 우리 image 빌드 — Aerial 위에 PyTorch + transformers 추가

CloudLab에서는 `Dockerfile.airan` 으로 빌드. Perlmutter Shifter는 image build 직접 못 함. 두 가지 옵션:

**옵션 A (간단)** — Aerial base image 그대로 쓰고, PyTorch는 runtime에 pip install
```bash
# 매번 컨테이너 띄울 때 pip install (느림)
shifter --image=nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb \
        bash -c "python3 -m pip install --user torch==2.4.1 torchvision==0.19.1 transformers==4.44.2 && python3 your_script.py"
```

**옵션 B (권장)** — 한 번 conda env 만들어두고 거기에 모든 의존성 설치
```bash
# Aerial 컨테이너 안에서 conda env 만들기 (한 번만)
shifter --image=nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb bash
# 안에서:
python3 -m venv $SCRATCH/airan_venv
source $SCRATCH/airan_venv/bin/activate
pip install torch==2.4.1 torchvision==0.19.1 transformers==4.44.2 accelerate==0.34.2 sentencepiece==0.2.0 safetensors
```

**옵션 C (best)** — Podman-HPC로 Dockerfile.airan 빌드 (Perlmutter Podman 가능)
```bash
podman-hpc build -t airan:25-3-final -f scripts_for_node/cloudlab_aerial/Dockerfile.airan .
```

먼저 옵션 A로 시도해보고, 너무 느리면 B나 C로.

### 3.3. NVIDIA Aerial 오픈소스 repo (cuPHY 헤더/예제)

```bash
cd $SCRATCH
git clone --recurse-submodules https://github.com/NVIDIA/aerial-cuda-accelerated-ran.git
cd aerial-cuda-accelerated-ran
git lfs pull
```

---

## 4. SLURM job template

A100 노드 1개 잡고 인터랙티브로 작업하기:

```bash
# 인터랙티브 (디버깅용)
salloc -N 1 -C gpu -G 4 -t 4:00:00 -q regular -A m4243 --reservation=... 

# 배치 (긴 실험용)
cat > $SCRATCH/run_nomig_experiment.sbatch << 'EOF'
#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -t 6:00:00
#SBATCH -q regular
#SBATCH -A m4243
#SBATCH -J nomig_compare
#SBATCH -o nomig_%j.log

cd $SCRATCH/airan_cloudlab
bash scripts/run_all_nomig.sh
EOF

sbatch $SCRATCH/run_nomig_experiment.sbatch
```

※ `-A m4243`는 예시 계정. 실제 사용자 NERSC 계정으로 변경.

---

## 5. 실제 진행할 실험 — 우선순위 순

### 우선순위 1 (paper에 필수) — F equivalent on no-MIG

**원본 (CloudLab MIG)**: 3g L1 + AI on 2g, 40 conditions × n=5
**Perlmutter no-MIG 버전**: full GPU L1 + AI on same GPU (default time-slicing)

핵심 조건 (전체 40개 다 안 해도 됨. 아래 14개면 비교 충분):
```
F_0_alone:                L1 혼자 (no AI)                  — baseline
F_B_D2D_256MB_str4:       L1 + D2D memcpy massive          — generic stress
F_C_H2D_256MB_str4:       L1 + H2D PCIe stress
F_D_GEMM_4096:            L1 + tensor core GEMM
F_E_chanpred_b64:         L1 + chanpred (safe AI)
F_E_resnet_b64:           L1 + ResNet (contention AI)
F_E_neuralrx_default:     L1 + NeuralRx (contention AI)
F_E_qwen_small:           L1 + Qwen small
F_E_forecaster_d384:      L1 + Forecaster
F_E_sat_compute:          L1 + sat_compute
F_E_sat_hbm:              L1 + sat_hbm
F_F_stack_chanpred_x4:    L1 + 4×chanpred
F_F_stack_resnet_x2:      L1 + 2×ResNet
F_G_kitchen:              L1 + chanpred + memcpy + GEMM
```

각 n=5. Measurement: L1 frame time JSON (mean / p99 / max) + 가능하면 nsys 캡처.

### 우선순위 2 — MPS comparison

같은 14개 조건을 **MPS 활성화 상태**에서도 측정:

```bash
# MPS daemon 시작 (allocation 안에서)
nvidia-cuda-mps-control -d
# 작업 실행
# ...
# MPS daemon 종료
echo quit | nvidia-cuda-mps-control
```

이렇게 하면 비교 매트릭스:
- MIG cross-partition (CloudLab 데이터, 있음)
- MIG same-partition coloc (CloudLab 데이터, 있음)
- no-MIG default time-slice (네가 측정할 것)
- no-MIG MPS (네가 측정할 것)

### 우선순위 3 — Per-call NSYS analysis on no-MIG

CloudLab의 §16.1 결과 (per-call 60KB memcpy 4.2us → 14.3us)를 no-MIG에서도 측정:

```bash
# F_E_neuralrx 같은 조건에서 nsys 캡처
shifter --image=... -- nsys profile --trace=cuda --output=nomig_neuralrx \
    python3 real_l1.py nomig_neuralrx 20 100
```

그 후 sqlite로 변환해서 같은 분석 (memcpy direction breakdown, per-call duration distribution).

### 우선순위 4 — NCU DRAM throughput on no-MIG

`--clock-control none` 추가 (MIG 노드에서는 clock lock 불가):

```bash
shifter --image=... -- ncu --target-processes all --replay-mode kernel \
    --clock-control none \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active \
    --csv --log-file ncu_nomig_neuralrx.csv \
    python3 real_l1.py ncu_nomig_neuralrx 20 3
```

L1 alone + 핵심 contention case (L1+NeuralRx, L1+ResNet) 측정.

### 우선순위 5 — 5분 sustained

CloudLab P5와 동등. 9 workload × 5분 × n=2. SLURM job 시간 길어서 batch로 submit.

---

## 6. 스크립트 변환 — Docker → Shifter

CloudLab 스크립트의 docker 명령을 Shifter로 변환하는 패턴:

**CloudLab (원본):**
```bash
docker run --rm --gpus "device=$L1_UUID" --user "$UID_:$GID_" \
  -v "$SDK:/opt/nvidia/cuBB" -v "$SCRIPT:/scripts" -v "$OUT:/out" \
  -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
  -w /scripts "$IMAGE" \
  bash -c "nsys profile --output=/out/L1_test python3 real_l1.py L1_test 20 100"
```

**Perlmutter (변환):**
```bash
shifter --image=nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb \
  --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
  --volume="$SCRATCH/airan_cloudlab:/scripts" \
  --volume="$OUT:/out" \
  --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
  --workdir=/scripts \
  bash -c "nsys profile --output=/out/L1_test python3 real_l1.py L1_test 20 100"
```

주요 변환 규칙:
- `docker run --rm --gpus ...` → `shifter --image=...`
- `-v src:dst` → `--volume=src:dst`
- `--user UID:GID` → 안 씀 (Shifter는 user 그대로)
- `-e KEY=VAL` → `--env=KEY=VAL`
- `-w dir` → `--workdir=dir`
- 마지막 image 이름은 `--image=` 인자에 들어가고, command가 image 자리에 옴

MIG 관련 명령은 **전부 제거**:
- `sudo nvidia-smi mig -dgi/-cgi` → 삭제
- `get_uuid()` 함수에서 MIG UUID 추출 → 그냥 GPU UUID 사용: `nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1`

---

## 7. 출력 위치 + 결과 파일 형식

모든 결과를 `$SCRATCH/airan_cloudlab/results/perlmutter_nomig/` 아래 정리:

```
results/perlmutter_nomig/
├── 환경_점검.md            ← Section 1의 결과 정리
├── F_nomig/                ← F-equivalent results (default time-slice)
│   ├── F_0_alone_run1.nsys-rep
│   ├── F_0_alone_run1.json (real_l1 output)
│   ├── F_E_neuralrx_default_run1.nsys-rep
│   └── ...
├── F_nomig_mps/            ← Same conditions but with MPS
├── NCU_nomig/              ← NCU CSVs
├── P5_nomig/               ← 5-min sustained
└── progress.log
```

JSON 형식 — `real_l1.py` 가 출력하는 그대로 (수정 불필요):
```json
{
  "label": "nomig_F_E_neuralrx_run1",
  "num_cells": 20,
  "iterations": 100,
  "mean_ms": 43.5,
  "p99_ms": 46.0,
  "max_ms": 48.2,
  "miss_1ms": 100,
  "raw_ms": [...]
}
```

이렇게 모이면 CloudLab MIG 데이터와 1:1 비교 가능.

---

## 8. 너가 빠질 함정들 (미리 경고)

1. **MIG 끄려고 시도하지 말 것** — `sudo` 없으므로 어차피 안 됨. 노드가 이미 MIG OFF 상태인 것 확인하고 진행.
2. **Aerial pyaerial Python API 경로 차이** — Aerial repo 안에서 pyaerial은 `pyaerial/src` 에 있음. PYTHONPATH 잘 잡아야 `from aerial.phy5g.algorithms import ChannelEstimator` 가 동작.
3. **nsys 출력 경로** — Shifter는 volume mount된 path에 써야 함. 아니면 컨테이너 종료시 사라짐.
4. **SLURM 시간 초과** — F + G + nsys + NCU 다 하려면 6시간으로 모자랄 수도. 각 단계 별도 batch job으로 쪼개.
5. **NeuralRx TRT engine build** — 첫 실행시 ONNX → TensorRT 빌드에 1-2분 걸림. 이걸 measurement window에서 빼야 함.
6. **MPS 시작 실패** — CloudLab에서 MPS pipe 연결 문제로 실패한 적 있음. 만약 MPS 안 되면 default time-slice만 하고 사용자에게 보고.

---

## 9. 보고 형식 — 매 단계 끝나면 사용자에게 보고

각 우선순위 끝나면 짧은 보고:

```
[priority N] 완료
조건 수: X / Y 끝남
주요 발견: [1-2 줄]
다음 단계: [무엇 할지]
```

특히 **MIG OFF 상태에서 L1 + NeuralRx가 어떻게 나오는지** 가 가장 중요한 finding. 이게 CloudLab MIG same-partition (200x 폭락)과 비교됨:
- no-MIG default time-slice에서 NeuralRx와 L1이 비슷한 정도로 무너지면 → "MIG가 필요 (no-MIG는 더 나쁨)" 주장 가능
- 비슷하거나 더 좋으면 → 우리 paper의 MIG 필요성 주장 약화

---

## 10. 마지막 — 어떤 데이터가 paper에 들어가는가

너의 측정 결과는 `MIG_AIRAN_VISUAL_EVIDENCE_KR.md` 에 새로운 PART F로 추가될 예정:

- §27: MIG vs no-MIG default time-slice L1+AI 비교
- §28: MIG vs no-MIG MPS L1+AI 비교
- §29: 종합 — MIG가 필요한지 / MIG가 어디서 도움 되는지 결판

이걸 위해 너가 모아야 할 핵심 숫자 5개:

1. **no-MIG default + L1 + NeuralRx**: L1 frame p99 → CloudLab MIG coloc 357ms와 비교
2. **no-MIG MPS + L1 + NeuralRx**: 동일 → MPS가 MIG보다 나은지
3. **no-MIG default + L1 + chanpred**: chanpred가 MIG에서는 안전한데 no-MIG에서는?
4. **no-MIG default + L1 + ResNet**: bistable contention이 no-MIG에서도 발생하는지
5. **no-MIG NCU DRAM throughput L1 + NeuralRx**: MIG에서 11% → no-MIG에서 어떻게 변하는지

이 5개만 정확히 나와도 paper의 critical comparison 데이터 완성.

---

## 11. 시작 명령

전부 이해했으면 환경 점검부터 시작:

```bash
echo "=== Perlmutter no-MIG experiment session ==="
date
hostname
whoami
echo
# Section 1의 환경 점검 명령 실행
```

문제 생기면 사용자에게 보고하고 진행 지시 받기. 절대로 sudo 권한 우회 시도, NGC 인증 우회 시도 같은 거 하지 말 것.

행운을 빈다.
