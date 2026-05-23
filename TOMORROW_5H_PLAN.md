# 5/24 5시간 reservation — 종합 plan (phase 1+2+3 모두)

> **모든 실험 5시간 안에 돌림**. master_5h_sweep.sh 한 번으로 phase 1+2+3 자동 진행.

## 실험 구성

### Phase 1 — Bimodal mechanism (1h55m, N=20)
가장 publishable. SENSITIVITY_EXPERIMENTS §A1-A2.

| # | preset | AI | 목적 |
|---|---|---|---|
| A0 | split-60-40 | qwen7b | bimodal 재현 N=20 |
| A1a | split-60-40 | qwen7b_prefill | H1 phase HIGH trigger |
| A1b | split-60-40 | qwen7b_decode | H1 phase LOW trigger |
| A2 | split-60-40 | hbm 16GB | H1 vs H2/H3 분리 (non-phased) |

### Phase 2 — Multi-AI partition (45m, N=10)
production AI-RAN 시나리오. 1 L1 + 2-3 AI services.

| # | preset | partitions | L1 | AI 매핑 |
|---|---|---|---|---|
| M1 | 3way-balanced | 2g+2g+3g | 3g | qwen_small × 2 (대칭 AI×2) |
| M2 | 3way-L1small | 2g+2g+3g | 2g | qwen_small + qwen7b (asym AI) |
| M3 | 3way-asym | 1g+2g+4g | 4g | qwen_small + gpt2 (mixed) |
| M4 | 4way-1L1+3AI | 4g+1g+1g+1g | 4g | gpt2 + resnet + hbm_1g (3 light) |

### Phase 3 — D1 + A baseline (45m, N=10)
partition cap 분리 + 진짜 best-case L1.

| # | preset | AI | 목적 |
|---|---|---|---|
| D1a | split-40-60 | qwen7b | 2g.10gb L1 + Qwen leakage |
| D1b | split-40-60 | none | 2g.10gb L1 alone (partition cap 단독) |
| A | no-mig | none | 풀 GPU L1 alone (driver_reset.sh 우회) |

→ AI leakage = D1a - D1b, partition cap = D1b - A

## 5시간 시간표

| 시각 | 분 | 누적 | 작업 |
|---|---|---|---|
| 10:00 ~ 10:05 | 5 | 5m | SSH + 환경 점검 |
| 10:05 ~ 10:15 | 10 | 15m | scp 12개 파일 |
| 10:15 ~ 10:25 | 10 | 25m | MIG enable + sanity test |
| 10:25 ~ 10:30 | 5 | 30m | master_5h_sweep.sh 시작 (백그라운드) |
| 10:30 ~ 12:25 | 115 | 145m | Phase 1 (A0/A1a/A1b/A2 N=20) |
| 12:25 ~ 13:10 | 45 | 190m | Phase 2 (M1/M2/M3/M4 N=10) |
| 13:10 ~ 13:55 | 45 | 235m | Phase 3 (D1ab + A baseline) |
| 13:55 ~ 14:25 | 30 | 265m | buffer / re-run failures |
| **14:25 ~ 14:50** | **25** | **290m** | **rsync + git push** |
| 14:50 ~ 15:00 | 10 | 300m | **CloudLab image snapshot 저장** |

**총 4시간 50분, 10분 여유**.

## scp 올릴 12개 파일

```bash
# cloudlab_aerial/ — 8개 sweep + 1 운영
scp ~/New_research/cloudlab_aerial/{dmon_sync,run_n20,driver_reset,phase1_sweep,phase2_multipartition,phase3_extras,master_5h_sweep,run_sweep_v2}.sh \
    sgkim@<HOST>:~/cloudlab_aerial/

# AIRAN_Changjong/experiments/ — 3 AI workload
scp ~/New_research/AIRAN_Changjong/experiments/{run_qwen7b_prefill,run_qwen7b_decode,run_qwen_small_stress}.py \
    sgkim@<HOST>:~/AIRAN_Changjong/experiments/

# home — bimodal 분석기
scp ~/New_research/cloudlab_results/bimodal_detect.py sgkim@<HOST>:~/
```

## 노드 ready 후 명령 (복붙용)

```bash
# 1. SSH
ssh sgkim@<HOSTNAME>.wisc.cloudlab.us

# 2. 환경 점검 (5초)
nvidia-smi --query-gpu=name,driver_version,mig.mode.current --format=csv,noheader
ls /mydata/
docker images | grep aerial
df -h /

# 3. (로컬에서) 12개 파일 scp — 위 블록 그대로

# 4. (SSH 안) chmod
cd ~/cloudlab_aerial && chmod +x *.sh

# 5. MIG enable 확인
sudo nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader -i 0
# Disabled이면:
bash ./driver_reset.sh
sudo nvidia-smi -i 0 -mig 1
sleep 3

# 6. Sanity test (30초)
N=2 PRESET=split-50-50 AI=none TAG=sanity DURATION=10 bash ./run_n20.sh
cat ~/cloudlab_aerial/results/$(date +%Y%m%d)/n2_sanity/summary.txt
# 예상: mean ~46ms이면 OK

# 7. MASTER SWEEP 시작 (백그라운드)
nohup ./master_5h_sweep.sh > /tmp/master_$(date +%H%M).log 2>&1 &
echo $! > /tmp/master.pid
tail -f /tmp/master_*.log
```

## 종료 직전 명령 (14:25 이후, 로컬에서)

```bash
# 1. 완료 확인
ssh sgkim@<HOST> "tail -20 /tmp/master_*.log"

# 2. rsync
rsync -av sgkim@<HOST>:~/cloudlab_aerial/results/$(date +%Y%m%d)/ \
    ~/New_research/cloudlab_results/results/$(date +%Y%m%d)/

# 3. git
cd ~/New_research/cloudlab_results
git add results/$(date +%Y%m%d)/
git commit -m "$(date +%Y%m%d) 5h sweep: phase1+phase2+phase3"
git push

# 4. CloudLab portal → "Save Image" → small-lan 덮어쓰기
```

## 시나리오별 대응

### A: 모든 phase 성공
- phase 1 H1 verdict, phase 2 multi-AI behavior, phase 3 leakage 분해
- → 5/31 30h에서 B1 model size, C1 Nsight 등 더 sensitive 실험

### B: phase 1만 끝남 (예상)
- 1h55m에 phase 1만 끝남 → 핵심 publishable 확보
- phase 2/3는 5/31로
- master sweep이 phase 1 끝나면 즉시 phase 2 시작하니까 시간 되는 만큼 진행

### C: phase 1 실패
- container/MIG 오류 → 트러블슈트
- N=10으로 줄여 재시도
- 안 되면 5/31에 전면 재시도

## ★ image snapshot이 핵심

5/24 끝나기 전에 snapshot 저장 = 5/31 30시간을 100% 실험에 투입 가능. 안 저장하면 setup 30분이 또 들어가서 비싼 시간 잃음.

마지막 업데이트: 2026-05-23 6:30PM
실행 명령: `nohup ./master_5h_sweep.sh > /tmp/master.log 2>&1 &`
