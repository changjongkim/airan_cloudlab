# 5/24 5시간 reservation — 압축 plan

> **오늘 5/23 reservation은 d8545 풀 점유로 실패**. 다음 슬롯 = 5/24 10AM~3PM (5시간).
> 5시간은 매우 빠듯하니 **publishable 가치 가장 높은 한 가지만** 확정.

## 최우선 목표 (must-have)

**A1 phase 가설 검증** — bimodal의 진짜 원인이 Qwen prefill/decode phase alignment인지.
- A0: split-60-40 + qwen7b (mixed) N=20 — baseline 재현 + 통계 확정
- A1a: split-60-40 + qwen7b_prefill N=20 — H1이 맞으면 거의 항상 HIGH
- A1b: split-60-40 + qwen7b_decode N=20 — H1이 맞으면 거의 항상 LOW
- A2: split-60-40 + hbm 16GB N=20 — non-phased AI로 H1 vs H2/H3 분리

## 5시간 정확한 시간표

| 시간 | 분 | 작업 |
|---|---|---|
| 10:00 ~ 10:05 | 5 | SSH 들어가서 nvidia-smi, ls /mydata, df -h |
| 10:05 ~ 10:15 | 10 | scp 6개 script (dmon_sync, run_n20, driver_reset, phase1_sweep, run_sweep_v2 업데이트, run_qwen7b_{prefill,decode}.py) |
| 10:15 ~ 10:25 | 10 | MIG enable, Aerial container 확인 (pull 필요하면 백그라운드) |
| 10:25 ~ 10:30 | 5 | real_l1.py 한 번 sanity test (L1 alone on 3g.20gb) |
| **10:30 ~ 12:30** | **120** | **phase1_sweep.sh 실행 — A0/A1a/A1b/A2 × N=20** |
| 12:30 ~ 13:00 | 30 | bimodal_detect.py 4개 디렉토리 분석 → verdict 확인 |
| 13:00 ~ 14:00 | 60 | (시간 남으면) D1: split-40-60 N=20 + alone → partition cap 분리 |
| 14:00 ~ 14:30 | 30 | (시간 더 남으면) A baseline 시도 (driver_reset.sh) |
| **14:30 ~ 15:00** | **30** | **강제 종료 — rsync + git push + image snapshot 저장** |

## phase1_sweep 시간 견적

- 각 run = ~60초 (DURATION 30s + L1 measurement 30s + setup 5s)
- N=20 × 60s = 20분 per config
- 4 configs × 20분 = 80분
- + MIG 재구성 between configs (~3min × 4) = 12분
- + dmon overhead + analysis = 약 10분
- **총 ~1.7시간** → 2시간 budget 충분

## 가능한 결과 시나리오

### 시나리오 1: H1 (phase) 강력 confirm
```
A0 qwen mixed       BIMODAL ~50:50          (재현 OK)
A1a prefill         BIMODAL HIGH dominant or UNIMODAL HIGH
A1b decode          UNIMODAL LOW
A2 hbm              UNIMODAL (또는 약한 bimodal)
```
→ 강한 publishable claim: "Bimodal leakage is driven by LLM phase oscillation, not by NoC/arbiter contention"

### 시나리오 2: H1 reject, H2/H3 채택
```
A0 qwen mixed       BIMODAL
A1a prefill         BIMODAL (여전히)
A1b decode          BIMODAL (여전히)
A2 hbm              BIMODAL
```
→ phase 무관, mechanism은 NoC/arbiter chip-level shared resource

### 시나리오 3: bimodal 자체가 noise (재현 실패)
```
A0 qwen mixed       UNIMODAL
```
→ 원래 N=4 발견이 noise였음. 다른 publishable angle 필요 (mean isolation collapse, partition cap 등)

세 시나리오 모두 **publishable** — 결과가 어떻든 의미 있음.

## 환경 복구 명령

```bash
# 0. SSH (hostname은 5/24 reservation 시작 시 부여됨)
ssh sgkim@<hostname>.wisc.cloudlab.us

# 1. 환경 점검
nvidia-smi
nvidia-smi --query-gpu=mig.mode.current --format=csv
df -h / /mydata
docker images | grep aerial
ls /mydata/ ~/cloudlab_aerial/ ~/AIRAN_Changjong/

# 2. (로컬에서) scp 6개 파일
scp ~/New_research/cloudlab_aerial/{dmon_sync,run_n20,driver_reset,phase1_sweep,run_sweep_v2}.sh \
    sgkim@<host>:~/cloudlab_aerial/
scp ~/New_research/AIRAN_Changjong/experiments/run_qwen7b_{prefill,decode}.py \
    sgkim@<host>:~/AIRAN_Changjong/experiments/
scp ~/New_research/cloudlab_results/bimodal_detect.py sgkim@<host>:~/

# 3. MIG enable (이미지 부팅 직후엔 disabled)
sudo nvidia-smi -i 0 -mig 1
# 만약 "in use" 에러 나면:
bash ~/cloudlab_aerial/driver_reset.sh
sudo nvidia-smi -i 0 -mig 1

# 4. Aerial container 확인
docker images | grep aerial
# 이미지에 잔존하면 skip, 없으면:
docker pull nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb &

# 5. Sanity test (1분)
cd ~/cloudlab_aerial
env PRESETS=split-50-50 AI=none CELLS=20 DURATION=10 \
    L1_NUM_TX_ANT=8 L1_NUM_RX_ANT=8 \
    bash ./run_sweep_v2.sh

# 6. phase1_sweep 시작 (백그라운드)
nohup ./phase1_sweep.sh > /tmp/phase1_$(date +%H%M).log 2>&1 &
tail -f /tmp/phase1_*.log
```

## 종료 직전 — 14:30 이후

```bash
# 결과 rsync (로컬에서)
rsync -av sgkim@<host>:~/cloudlab_aerial/results/$(date +%Y%m%d)/ \
    ~/New_research/cloudlab_results/results/$(date +%Y%m%d)/

# git push
cd ~/New_research/cloudlab_results
git add results/$(date +%Y%m%d)/ && \
git commit -m "5/24 5h sweep: phase1 N=20 bimodal mechanism" && \
git push

# (CloudLab portal) image snapshot 저장
# Experiment 페이지 → "Save Image" or "Snapshot Node"
# → 이미지 이름: small-lan-v2 (또는 small-lan 덮어쓰기)
# 다음 5/31 30시간에서 이 이미지로 부팅하면 5/24 작업물 보존
```

마지막 업데이트: 2026-05-23 6:00PM
다음 reservation: 2026-05-24 10:00AM ~ 3:00PM (5시간)
