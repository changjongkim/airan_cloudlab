# 5/31 Morning Checklist — CloudLab Reservation 30h

## 사전 준비 (오늘 저녁 / 내일 아침 reservation 시작 전)

- [ ] CloudLab dashboard에서 reservation 시작 시간 확인
- [ ] 5/24 image snapshot 했는지 확인 (했으면 setup 단축 가능)
- [ ] github 마지막 commit에 모든 신규 scripts 포함됐는지 확인
- [ ] 이 문서 (`MORNING_531_CHECKLIST.md`) 출력 또는 모바일에서 열어두기

---

## Reservation 시작 직후 (T+0 ~ T+0:15)

```bash
# 1. SSH 접속
ssh sgkim@<HOSTNAME>.wisc.cloudlab.us

# 2. 진단 — 어떤 상태로 들어왔나
bash ~/cloudlab_aerial/recovery.sh
# 출력 마지막 줄: BEST / PARTIAL / WORST
```

### 분기

| 결과 | 의미 | 후속 |
|---|---|---|
| 🟢 **BEST** | 이미지 + /mydata 다 살아있음 | 바로 실험 가능, T+0:15부터 실험 |
| 🟡 **PARTIAL** | 이미지 OK, /mydata 비어있음 | `post_reboot_setup.sh` 실행 (~45-60분) |
| 🔴 **WORST** | 이미지조차 fresh Ubuntu | `00_bootstrap.sh` + `post_reboot_setup.sh` (~90분) |

---

## PARTIAL 시나리오 (most likely, 70% 확률)

```bash
# Local에서 scp (최신 스크립트들 전송)
scp ~/New_research/cloudlab_aerial/*.sh sgkim@<HOST>:~/cloudlab_aerial/
scp ~/New_research/cloudlab_aerial/real_l1.py sgkim@<HOST>:~/cloudlab_aerial/
scp ~/New_research/cloudlab_aerial/Dockerfile.airan sgkim@<HOST>:~/cloudlab_aerial/
scp ~/New_research/AIRAN_Changjong/experiments/run_*.py sgkim@<HOST>:~/AIRAN_Changjong/experiments/
scp ~/New_research/cloudlab_results/analyze_run.py sgkim@<HOST>:~/

# Node에서
chmod +x ~/cloudlab_aerial/*.sh
nohup bash ~/cloudlab_aerial/post_reboot_setup.sh > /tmp/setup.log 2>&1 &
tail -f /tmp/setup.log
# 45-60분 대기 — Aerial pull + Qwen DL + pyaerial build 병렬 진행
```

setup 끝나면 다음으로:

---

## 실험 단계 (T+1:00 ~ T+27:00)

### 🔴 Tier 1: Paper-Critical (T+1 ~ T+10)

#### A. 8T8R 재측정 (T+1 ~ T+7)
**중요도**: 🔴 paper 모든 figure 재생성
```bash
# run_sweep_v2.sh에 NUM_TX_ANT/NUM_RX_ANT export 추가
# (또는 real_l1.py 호출 시 직접 env 지정)

# 모든 baselines + Phase 1-4 재측정 with 8T8R
cd ~/cloudlab_aerial
nohup bash ./phase1_sweep.sh > /tmp/p1.log 2>&1 &
# (각 phase 끝나면 다음 시작)
```

#### B. AI Throughput 정확한 측정 (T+7 ~ T+9)
**중요도**: 🔴 양방향 영향 paper claim
```bash
nohup bash ./ai_throughput_v2.sh > /tmp/ai_thru.log 2>&1 &
# 4 workloads × 2 setups × N=5 × ~30s = ~30분
```

#### C. Nsight Systems Profiling (T+9 ~ T+10:30)
**중요도**: 🔴 F8 mechanism 직접 증거
```bash
# MIG 상태 조정: 4 scenario별로 다름

# Scenario A: Full GPU (MIG off — GPU 3에서)
GPU=3 bash ./nsys_profile_runner.sh

# Scenario B/C/D: MIG enabled GPU에서
# split-60-40 → C (3g alone) + E (3g + Qwen)
sudo nvidia-smi mig -i 0 -cgi 9,14 -C
GPU=0 bash ./nsys_profile_runner.sh

# split-40-60 → D (2g alone)
sudo nvidia-smi mig -i 0 -dci; sudo nvidia-smi mig -i 0 -dgi
sudo nvidia-smi mig -i 0 -cgi 14,9 -C
GPU=0 bash ./nsys_profile_runner.sh

# 7g single → B
sudo nvidia-smi mig -i 0 -dci; sudo nvidia-smi mig -i 0 -dgi
sudo nvidia-smi mig -i 0 -cgi 0 -C
GPU=0 bash ./nsys_profile_runner.sh
```

#### D. Nsight Compute per-kernel (T+10:30 ~ T+12)
**중요도**: 🔴 L2 cache / DRAM throughput 정량
```bash
# Same scenario rotation as nsys
GPU=0 bash ./ncu_profile_runner.sh
```

### 🟡 Tier 2: Paper-Strengthening (T+12 ~ T+20)

#### E. Multi-AI count 확장 (T+12 ~ T+14)
- 3g + 5/6 AI 측정 (extension of 5/24's 3/4 AI)
- 3g + 3/4 AI N=20 재측정 (이전 N=5 → noise 큼)

#### F. 4g cells=40 N=10 재측정 (T+14 ~ T+14:30)
- 5/24 N=1 → 통계 신뢰성 부족

#### G. Cell scaling 세밀 grid (T+14:30 ~ T+16:30)
- 3g cells=15/25/30/35 (knee 정확히)
- 4g cells=50/60 (saturation 찾기)

#### H. MPS Comparison (T+16:30 ~ T+19:30)
```bash
# MIG off, MPS 켜기
sudo nvidia-smi -i 0 -mig 0
sudo reboot   # MIG disable 적용

# Reboot 후
sudo systemctl start nvidia-mps-control
# 같은 sweep을 MPS에서
```

#### I. MCS/PRB sensitivity (T+19:30 ~ T+21:30)
- MCS 14, 27 / PRB 100, 200

### 🟢 Tier 3: Nice-to-Have (T+21:30 ~ T+27)

#### J. Long-duration stability (T+21:30 ~ T+22:30)
- 1시간 연속 3g + Qwen sweep, drift 관찰

#### K. Real Aerial pipeline (T+22:30 ~ T+26)
- 시간 남으면 시도, 안 되면 skip

---

## 마무리 (T+27 ~ T+30)

### Analysis + Figures (T+27 ~ T+28:30)
```bash
# 모든 결과 분석
python3 ~/analyze_run.py ~/cloudlab_aerial/results/$(date +%Y%m%d)/

# Nsight summary
ls ~/cloudlab_aerial/results/$(date +%Y%m%d)/nsys/
ls ~/cloudlab_aerial/results/$(date +%Y%m%d)/ncu/
```

### Rsync + Git Push (T+28:30 ~ T+29:30)
```bash
# Local에서 (꼭 dmon.csv 포함!)
rsync -av sgkim@<HOST>:~/cloudlab_aerial/results/$(date +%Y%m%d)/ \
    ~/New_research/cloudlab_results/results/$(date +%Y%m%d)/

cd ~/New_research/cloudlab_results
git add results/$(date +%Y%m%d)/
git commit -m "5/31 30h sweep: 8T8R + AI throughput v2 + Nsight profiles + MPS"
git push
```

### CloudLab Image Save (T+29:30 ~ T+30:00) ⭐ 절대 잊지 마
```
1. CloudLab portal → Experiments → 클릭
2. "Save Image" 또는 "Snapshot"
3. Image name: small-lan (덮어쓰기)
4. 시작
```

---

## 신규 스크립트 사전 점검 리스트

| Script | 상태 | 비고 |
|---|---|---|
| `recovery.sh` | ✅ pushed | 진단 |
| `post_reboot_setup.sh` | ⭐ NEW | 모든 5/24 lessons 반영 |
| `real_l1_loop.sh` | ⭐ NEW | persistent L1 wrapper |
| `ai_throughput_v2.sh` | ⭐ NEW | valid AI throughput |
| `nsys_profile_runner.sh` | ⭐ NEW | Nsight Systems |
| `ncu_profile_runner.sh` | ⭐ NEW | Nsight Compute |
| `run_sweep_v2.sh` | ⚠ needs 8T8R env var fix | NUM_TX_ANT export |
| `run_n20.sh` | ✅ pushed | JSON filter ok |
| `multi_ai_count.sh` | ⚠ local만 | github push 필요 |
| `cell_scaling_2g_4g.sh` | ⚠ local만 | github push 필요 |
| `Dockerfile.airan` | ⚠ local만 | github push 필요 |
| `00_bootstrap.sh` | ⚠ local만 | WORST 시나리오용 |
| Phase 1-4 sweeps | ✅ pushed | 그대로 사용 |

---

## 비상 시나리오

### Aerial 컨테이너 pull 실패
- Container disk 부족: containerd 위치 점검 (`ls -la /var/lib/containerd`)
- 네트워크: NGC 직접 접근 시도 또는 25.3.0 / 25.3.1 등 다른 tag

### MIG enable 실패 (in use by another client)
- nvidia-persistenced 정지: `sudo systemctl stop nvidia-persistenced`
- Driver reset: `bash driver_reset.sh`
- 그래도 안 되면: reboot

### pyaerial import 실패
- PYTHONPATH 확인: `/opt/nvidia/cuBB/pyaerial/src`
- 컨테이너에 commit 안 됐을 가능성: `airan:25-3-final` 다시 commit

### Nsight 권한 거부
- `--cap-add=SYS_ADMIN` 추가
- 또는 sudo 사용
- 그래도 안 되면 `nvprof` 시도 (legacy)

---

## 기억해야 할 5/24 lessons

1. **containerd도 /mydata로** — docker root만 옮기면 안 됨
2. **HF cache chmod 777** — 컨테이너 UID 1000이 쓸 수 있게
3. **Aerial repo chown 1000:1000** — 동일 이유
4. **pyaerial PYTHONPATH commit** — 이미지 재실행 시 잃지 않게
5. **`L1_NUM_TX_ANT` ≠ `NUM_TX_ANT`** — env var 이름 정확히
6. **dmon.csv는 rsync 명시적 포함** — 5/24에 exclude되어 분석 못함
7. **MIG enable 시 reboot 필요할 수 있음** — pending state면 reboot
8. **GPU 3 no-MIG는 reboot 후에 가능** — 라이브로 disable 안 됨

---

## 한 줄로

> "Reservation 시작 → `recovery.sh` → PARTIAL이면 `post_reboot_setup.sh` (60분) → Tier 1 실험 (10시간) → Tier 2 (8시간) → 마무리 (3시간) → **CloudLab Image Save 필수**."
