# 5/24 아침 — Recovery Plan (이미지 불완전 가정)

> 5/12 snapshot은 root filesystem 9.4GB만 = **/mydata 비어있음 가능성 99%**.
> 그러면 Docker 이미지 (Aerial 35GB), cuPHY artifacts, HF cache (Qwen 14GB) 다 사라짐.
> 즉시 master sweep 불가. **재 setup 1~2시간 후 phase 1만 가능**.

## 시나리오별 시간 손실

| 상황 | 확률 | Setup overhead | 실험 가능 |
|---|---|---|---|
| 🟢 BEST (모든 거 살아있음) | 5% | 5분 | Phase 1+2+3+4 (4시간) |
| 🟡 PARTIAL (root+scripts만) | 70% | **60-90분** | **Phase 1 + 부분 Phase 4 (2.5시간)** |
| 🔴 WORST (전부 fresh) | 25% | 120-180분 | Phase 1만 시도 (1.5시간) |

## 아침 절차 — 한 번에 (10:00~10:05 진단 → 분기)

### Step 1: SSH + 진단 (5분, ALWAYS 먼저)

```bash
# 1. SSH
ssh sgkim@<HOSTNAME>.wisc.cloudlab.us

# 2. 진단 — recovery.sh 한 줄
cd ~/cloudlab_aerial 2>/dev/null || (mkdir -p ~/cloudlab_aerial && cd ~/cloudlab_aerial)
# 만약 cloudlab_aerial dir 없으면 → 로컬에서 scp 먼저
# (로컬에서) scp ~/New_research/cloudlab_aerial/recovery.sh sgkim@<HOST>:~/

bash ./recovery.sh    # 출력 마지막 줄: BEST / PARTIAL / WORST
```

### Step 2: 분기 — Branch에 따라 다른 명령

#### 🟢 Branch BEST
```bash
# scp 16개 (한 번에)
# (로컬에서)
scp ~/New_research/cloudlab_aerial/{dmon_sync,run_n20,driver_reset,phase1_sweep,phase2_multipartition,phase3_extras,phase4_airan,master_5h_sweep,run_sweep_v2}.sh sgkim@<HOST>:~/cloudlab_aerial/
scp ~/New_research/AIRAN_Changjong/experiments/{run_qwen7b_prefill,run_qwen7b_decode,run_qwen_small_stress,run_channel_prediction,run_xapp_anomaly,run_neural_rx_stress}.py sgkim@<HOST>:~/AIRAN_Changjong/experiments/
scp ~/New_research/cloudlab_results/analyze_run.py sgkim@<HOST>:~/

# 즉시 sweep
ssh sgkim@<HOST>
cd ~/cloudlab_aerial && chmod +x *.sh
nohup ./master_5h_sweep.sh > /tmp/master.log 2>&1 &
```

#### 🟡 Branch PARTIAL (가장 가능성 큼)
```bash
# scp 18개 (recovery + setup 추가)
scp ~/New_research/cloudlab_aerial/{recovery,quick_setup,dmon_sync,run_n20,driver_reset,phase1_sweep,phase2_multipartition,phase3_extras,phase4_airan,master_5h_sweep,run_sweep_v2}.sh \
    sgkim@<HOST>:~/cloudlab_aerial/
scp ~/New_research/AIRAN_Changjong/experiments/{run_qwen7b_prefill,run_qwen7b_decode,run_qwen_small_stress,run_channel_prediction,run_xapp_anomaly,run_neural_rx_stress}.py \
    sgkim@<HOST>:~/AIRAN_Changjong/experiments/
scp ~/New_research/cloudlab_results/analyze_run.py sgkim@<HOST>:~/

# 자동 setup (60-90분, 백그라운드 docker pull + Qwen download 병렬)
ssh sgkim@<HOST>
cd ~/cloudlab_aerial && chmod +x *.sh
nohup ./quick_setup.sh > /tmp/setup.log 2>&1 &
tail -f /tmp/setup.log

# 끝나면 (Step 10 sanity test 성공 확인 후):
# 시간 부족 → master 대신 phase 1 단독:
N=20 DURATION=30 bash ./phase1_sweep.sh
# 시간 여유 → 전체:
bash ./master_5h_sweep.sh
```

#### 🔴 Branch WORST
```bash
# 첫 1시간: 00_bootstrap.sh 재실행 (driver, docker, NGC CLI 등)
# (이건 5/12 setup 이전에 했던 것)
scp ~/New_research/cloudlab_aerial/00_bootstrap.sh sgkim@<HOST>:~/cloudlab_aerial/
ssh sgkim@<HOST>
cd ~/cloudlab_aerial && chmod +x *.sh
sudo bash ./00_bootstrap.sh   # ~30분
# 후 PARTIAL 절차와 동일하게 quick_setup.sh → phase1만 시도
```

## 시간표 (가장 가능성 큰 PARTIAL 기준)

| 시각 | 분 | 작업 |
|---|---|---|
| 10:00 | 5 | SSH + recovery.sh 진단 |
| 10:05 | 10 | scp 18개 파일 |
| 10:15 | 45 | quick_setup.sh (Aerial pull + Qwen + airan build 병렬) |
| 11:00 | 30 | cuPHY/AI 컨테이너 검증, sanity test |
| 11:30 | 15 | MIG enable + 환경 점검 |
| 11:45 | 130 | **Phase 1 (A0/A1a/A1b/A2 N=20)** ⭐ |
| 13:55 | 30 | Phase 4 (AR1/AR2/AR3 N=10) — buffer 활용 |
| 14:25 | 30 | rsync + git push |
| 14:55 | 5 | **★ CloudLab image snapshot 저장** |
| 15:00 | 0 | reservation 만료 |

**목표 최소치**: Phase 1만이라도 끝내고 snapshot 저장. Phase 4는 bonus.

## sanity test 기준 — 진행 vs 중단 결정

setup 끝나고 11:30쯤 sanity 한 번:
```bash
N=2 PRESET=split-50-50 AI=none TAG=sanity DURATION=10 bash ./run_n20.sh
```

기준:
- **mean ~46ms** (3g.20gb baseline) → ✓ 정상, phase 1 진행
- **mean << 46ms (e.g. 20ms)** → MIG 안 들어간 듯 (full GPU에서 측정됨) → MIG enable 재확인
- **mean >> 46ms (e.g. 100ms)** → 뭔가 잘못됨, Aerial container 또는 cuPHY 의심 → log 확인
- **FAIL (no JSON)** → 컨테이너 못 띄움, 메시지 확인

## 절대 잊으면 안 되는 것 — 14:55 PM

```bash
# 1. master sweep 중지 (필요시)
kill $(cat /tmp/master.pid 2>/dev/null) 2>/dev/null

# 2. rsync (로컬에서)
rsync -av sgkim@<HOST>:~/cloudlab_aerial/results/$(date +%Y%m%d)/ \
    ~/New_research/cloudlab_results/results/$(date +%Y%m%d)/

# 3. git
cd ~/New_research/cloudlab_results
git add results/$(date +%Y%m%d)/ && git commit -m "5/24 5h sweep" && git push

# 4. ★ CloudLab portal → Save Image (덮어쓰기)
#    이번에는 /mydata도 같이 snapshot되도록 옵션 확인
```

## 자기 전 마지막 — 폰에 메모

```
9:55 AM ALARM:
- Dashboard 새로고침 → airankcj state 확인
- ready면 SSH, scheduled면 manual instantiate
- hostname 확인 후:
  ssh sgkim@<HOST>
  bash ~/cloudlab_aerial/recovery.sh
- 결과 BEST/PARTIAL/WORST → MORNING_RECOVERY_PLAN.md 분기 따라가기

14:55 PM CHECKPOINT:
- master sweep 결과 rsync
- git push
- CloudLab Save Image (CRITICAL — 안 하면 5/31에 또 같은 setup 반복)
```

마지막 업데이트: 2026-05-23 7:30 PM
다음 reservation: 2026-05-24 10:00 AM ~ 3:00 PM
