# Perlmutter no-MIG Handoff — quick start

Perlmutter에서 본 repo `git pull` 후 이 디렉터리에 들어가서:

```bash
cd $SCRATCH/airan_cloudlab/results/perlmutter_handoff
```

## 파일 목록

| 파일 | 역할 | 언제 실행 |
| --- | --- | --- |
| **BRIEFING.md** | 전체 상황 + 임무 + 함정 정리 | **가장 먼저 읽기** |
| `00_env_check.sh` | 환경 점검 (GPU/SLURM/Shifter/모듈) | login node에서 바로 실행 |
| `01_setup.sh` | Aerial container pull + repo clone + 작업 dir 생성 | 환경 점검 ok 후 |
| `02_F_saturation_nomig.sh` | F-equivalent on no-MIG (default time-slice) | salloc 안에서 |
| `03_mps_compare.sh` | 같은 조건 MPS로 측정 | salloc 안에서 |
| `04_ncu_nomig.sh` | NCU DRAM/L2/SM hardware counter | salloc 안에서 |
| `05_p5_sustained_nomig.sh` | 5분 sustained × 9 워크로드 | salloc 안 (긴 시간) |
| `run_all.sbatch` | 전체 자동 실행 SLURM batch | 한 번에 다 돌릴 때 |

## 빠른 시작

```bash
# 1. 환경 점검
bash 00_env_check.sh 2>&1 | tee env_check.log

# 2. 점검 결과 사용자에게 보고 → ok 받으면

# 3. Setup
bash 01_setup.sh 2>&1 | tee setup.log

# 4-A. 인터랙티브 (단계별 디버깅)
salloc -N 1 -C gpu -G 1 -t 6:00:00 -q regular -A <account>
bash 02_F_saturation_nomig.sh
# 결과 보고 → 다음 단계 진행

# 4-B. 또는 한번에 (batch)
# run_all.sbatch 안 -A m4243 을 실제 계정으로 수정 후
sbatch run_all.sbatch
```

## 결과 위치

모든 결과는 `$SCRATCH/airan_cloudlab/results/perlmutter_nomig/` 아래:

```
perlmutter_nomig/
├── F_nomig/              ← Section 우선순위 1 결과
├── F_nomig_mps/          ← Section 우선순위 2 결과
├── NCU_nomig/            ← Section 우선순위 3 결과
└── P5_nomig/             ← Section 우선순위 4 결과
```

이거 `rsync`로 사용자 로컬로 가져오면 CloudLab MIG 데이터와 비교 분석 가능.

## 비교 매트릭스 — 최종 목표

| 시나리오 | CloudLab MIG cross-partition | CloudLab MIG same-partition (coloc) | **Perlmutter no-MIG default** | **Perlmutter no-MIG MPS** |
| --- | --- | --- | --- | --- |
| L1 alone | 41ms p99 | — | (측정) | (측정) |
| L1 + chanpred | 41ms (영향 없음) | n/a | (측정) | (측정) |
| L1 + NeuralRx | 41ms ↔ 196ms (워크로드 의존) | **357ms (+537%)** | (측정) | (측정) |
| L1 + ResNet | bistable contention | n/a | (측정) | (측정) |
| L1 + Qwen | contention | n/a | (측정) | (측정) |
| L1 + sat_hbm | contention | n/a | (측정) | (측정) |

이 표가 완성되면 paper의 "MIG는 필요한가?" 질문에 답할 수 있음.

## 막혔을 때

- SLURM 권한 / 계정 문제 → 사용자에게 NERSC 계정 + project 확인 요청
- Shifter image pull 실패 → NGC 등록 필요 (사용자가 처리)
- pyaerial 임포트 실패 → AERIAL_REPO path 또는 PYTHONPATH 확인
- nsys "no GPU activity" 경고 → SLURM이 GPU 제대로 잡았는지 (`echo $SLURM_GPUS`)
- AI workload 사망 → 해당 `*_ai.log` 확인

질문 있으면 항상 사용자에게 물어보고 진행. 자동으로 sudo / 권한 우회 시도하지 말 것.
