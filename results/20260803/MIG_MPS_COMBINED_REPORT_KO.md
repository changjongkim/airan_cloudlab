# MIG + MPS 결합: AI-RAN을 위한 유일한 GPU 격리 전략

**날짜**: 2026-08-03
**플랫폼**: CloudLab d8545 · NVIDIA A100-SXM4-40GB × 4 · Driver 580.173.02 · CUDA 13.0
**워크로드**: cuPHY 25.3-cubb 5G L1 + 다양한 AI 스택 (Qwen 2.5-3B, Whisper large-v3, BERT, Qwen-VL, NRx, CsiNet, BeamPred)
**데이터**: Chain 17 (108 조건), Chain 18 (다중 파트), Chain 19 (273 조건, 213 per-iter L1 측정)
**그림**: 30장 · `analysis_chain19/figures/mig_mps/`

---

## 핵심 명제

> **MIG 단독도, MPS 단독도 불충분하다. MIG cross-partition + AI 파티션 MPS on 조합만이 5G L1 baseline 지연을 지키면서 AI throughput을 온전히 낼 수 있다.**

기존 duty-cycle 프레이밍은 오도되었다. 올바른 SLA 지표 — **L1 per-iteration p99 지연** — 로 다시 보면 그림이 뒤집힌다:

| 토폴로지                        | L1 p99 (ms) | AI throughput | 판정 |
| ------------------------------ | ----------- | ------------- | ---- |
| Multi-GPU                      | **40**      | 100 %         | ✅ 이상적 (비싸다) |
| **MIG CP + MPS on AI**         | **40**      | 100 %         | ✅ **프로덕션** |
| MIG SP + MPS pct=30 (튜닝)     | 45          | 85 %          | ⚠ 차선 |
| MIG SP + MPS 기본값            | 150+        | 100 %         | ✗ SLA 파괴 |
| Full GPU + MPS on              | 63          | 100 %         | ✗ L1 50 % 페널티 |
| MIG 없음, MPS 없음             | 300+        | 30 %          | ✗ 파국 |

MIG는 **하드웨어 격리** — L1 커널이 AI launch queue에 섞이지 않게 하는 유일한 메커니즘.
MPS는 **논리적 다중화** — N개의 AI 프로세스가 `cudaFree` context switch에서 직렬화되지 않게 하는 필수 도구.

둘 중 하나만 빠지면 조합은 무너진다. 둘 다 있으면 하나의 A100에서 5G L1 SLA + 완전한 AI throughput이 동시에 달성된다.

---

## 요약 (5줄)

1. **Duty cycle은 오도한다.** Full GPU + MPS는 L1 duty 62 %로 "건강해" 보이지만 실제 L1 p99는 63 ms — baseline 42 ms 대비 50 % 나쁨. Duty cycle을 SLA 게이트로 쓰지 말 것.
2. **MIG 단독은 AI throughput을 잃는다.** MIG cross-partition에서 AI측 MPS가 없으면 AI 프로세스가 3g 파티션 위에서 직렬화 → 집계 throughput ~70 % 하락.
3. **MPS 단독은 L1 SLA를 잃는다.** Full GPU + MPS는 baseline에 절대 도달하지 못함. N=1 다이버스 AI에서도 L1 p99 42 → 63 ms. N=6 same-partition + MPS 기본값에서는 150+ ms 파괴.
4. **MIG cross-partition + AI측 MPS on은 불변이다.** Chain 19 Exp 5에서 N ∈ {6, 8, 10, 12, 16} 전 구간에서 L1 mean/p95/p99가 baseline 고정. Qwen aggregate throughput은 선형 확장.
5. **이 결과는 SoftBank AITRAS 스타일 배치에 직접 적용된다.** 1× A100에 5G DU + 6-service AI 스택을 얹으려면 MIG CP + MPS on AI 외에는 답이 없다. 다른 선택은 5G TTI SLA 위반 또는 AI 용량 낭비.

---

## 1장 · 4-사분면 (MIG × MPS)

설계 공간은 2×2 행렬로 축소된다: MIG on/off × MPS on/off.

![F01](analysis_chain19/figures/mig_mps/F01_quadrant_l1_latency.png)

우상단 셀 (MIG cross-partition + MPS on AI) 만이 N=6에서 50 ms SLA proxy 아래 L1 지연을 유지.

![F02](analysis_chain19/figures/mig_mps/F02_quadrant_ai_throughput.png)

MPS는 MIG와 AI throughput 축에서 직교 — MPS on이면 MIG 여부와 무관하게 항상 이김. 즉 MPS는 필수. 단, MPS 단독 (MIG off)은 L1을 희생한다.

![F03](analysis_chain19/figures/mig_mps/F03_combined_verdict.png)

네 가지 판정:
- **FAIL** (MIG 없음, MPS 없음) — 파국
- **PARTIAL** (MIG 없음, MPS on) — L1 페널티 ~50 %
- **PARTIAL** (MIG cross, MPS off) — AI 직렬화
- **OPTIMAL** (MIG cross + MPS on) — 유일한 프로덕션 셀

![F04](analysis_chain19/figures/mig_mps/F04_pareto.png)

Pareto view: L1 지연(x) × AI throughput(y). 좌상단이 이상. **MIG CP + MPS on**과 **Multi-GPU** 만이 지배한다.

**1장 결론**: "MIG인가 MPS인가"가 문제가 아니라 "특정 토폴로지에서 둘을 결합하라"가 정답이다.

---

## 2장 · MIG 단독이 불충분한 이유

MPS를 완전히 빼거나 off로 두면 MIG는 우리에게 무엇을 주는가? AI throughput은 거의 얻지 못하고, L1 도움은 L1이 전용 파티션을 가질 때만 발생한다.

![F05](analysis_chain19/figures/mig_mps/F05_mig_off_mps_effect.png)

**Full GPU (MIG 없음)** — MPS on/off 비교. MPS off는 모든 N에서 파국 (커널 직렬화). MPS on은 낫지만 높은 N에서도 L1이 상승.

![F06](analysis_chain19/figures/mig_mps/F06_mig_same_partition.png)

**MIG on이지만 L1이 AI와 같은 파티션 공유 (same-partition)** — Config A (4g), Config C (3g) 둘 다 MPS off에서 N≥4 파괴. MIG 파티션 경계는 워크로드가 공유되면 도움이 안 된다.

![F07](analysis_chain19/figures/mig_mps/F07_all_configs_mpsoff.png)

**세 MIG 설정 (MPS off)** — 모든 토폴로지가 무너진다. 왜냐면 AI 프로세스가 공유 CUDA context에서 직렬화되어 동일 파티션의 L1 커널 launch를 막기 때문.

![F08](analysis_chain19/figures/mig_mps/F08_mig_alone_summary.png)

**요약**: 낮은 L1 + 높은 AI throughput을 동시에 내는 유일한 MIG 설정은 **MIG cross-partition + AI 파티션 MPS on**. 다른 모든 MIG 변형은 최소 한 축을 잃는다.

**2장 결론**: MIG 단독 (AI측 MPS 없음)은 AI를 직렬화시킴. MIG가 있어도 MPS off면 AI throughput ~70 % 손실. **MIG는 필수지만 불충분하다.**

---

## 3장 · MPS 단독이 불충분한 이유

MIG를 완전히 빼고 (Full GPU, no partition) MPS에만 의존하면?

![F09](analysis_chain19/figures/mig_mps/F09_mps_alone_full_gpu.png)

**Full GPU + MPS on, N 스윕** — N=1에서도 L1 p99가 42 ms → 63 ms (50 % 페널티). MPS thread% 튜닝으로도 baseline 복귀 불가. Launch queue를 공유하기 때문: L1 커널이 AI 커널 뒤에서 대기.

![F10](analysis_chain19/figures/mig_mps/F10_mps_breakdown_curves.png)

**MPS on breakdown 곡선** (세 config). same-partition에서 N=6이면 모든 config가 breakdown zone 진입. L1과 AI가 같은 파티션/GPU에서 공유 MPS scheduler를 쓰면 MPS는 L1을 구할 수 없다.

![F11](analysis_chain19/figures/mig_mps/F11_mps_pct_full_gpu.png)

**MPS thread% 튜닝 (Chain 19 Exp 11)** — (pct, N) 그리드의 L1 p99 heatmap. 최선은 pct=30, N=6에서 45 ms. 기본값 (150+ ms) 보다는 훨씬 좋으나 MIG CP baseline 대비 여전히 12 % 나쁨. **튜닝은 근접시킬 뿐 격리에 도달하지 못한다.**

![F12](analysis_chain19/figures/mig_mps/F12_diverse_vs_identical.png)

**Diverse vs identical 워크로드** MPS. 다양한 구성이 패킹 효율성 (SM/memory 압박 패턴 다양)엔 도움되지만 L1 페널티 제거는 못함.

**3장 결론**: MPS 단독 (MIG 없음)은 N=1에서도 L1 p99를 50 % 상승. 공격적 MPS thread% 튜닝은 페널티를 줄이지만 제로에는 못 감. **MPS는 필수지만 불충분하다.**

---

## 4장 · MIG + MPS 결합 = 승자

L1은 전용 MIG 파티션 (예: 4g.20gb), AI는 반대 파티션 (3g.20gb) + AI 다중화용 MPS on.

![F13](analysis_chain19/figures/mig_mps/F13_cp_l1_invariance.png)

**Chain 19 Exp 5 — CP + MPS 조건에서 L1 지연 불변성**. Mean/p95/p99 모두 N ∈ {6, 8, 10, 12, 16} 전 구간에서 baseline (~40 ms p99) 유지. **AI 부하와 무관하게 L1 페널티 제로.**

![F14](analysis_chain19/figures/mig_mps/F14_cp_ai_scaling.png)

**Qwen aggregate throughput이 N과 함께 확장** — 3g AI 파티션 위의 MPS가 N개의 vLLM 인스턴스를 효율적으로 다중화. L1 격리가 throughput 상한을 걸지 않는다.

![F15](analysis_chain19/figures/mig_mps/F15_cp_pareto.png)

**CP + MPS 조건들의 Pareto view** — 점들이 L1 p99 ≈ 40 ms의 수직선 근처에 몰리고 AI throughput만 위로 증가. 이상적 모양: L1 SLA 고정, AI throughput 자유 확장.

![F16](analysis_chain19/figures/mig_mps/F16_cp_vs_sp_direct.png)

**CP vs SP 직접 비교 (N=6)** — CP + MPS는 baseline 유지 (40 ms). SP는 최선의 pct=30 튜닝에서도 45 ms (12 % 나쁨). SP + 기본값 pct=100은 150+ ms (SLA 파괴).

![F17](analysis_chain19/figures/mig_mps/F17_cp_extreme_scale.png)

**극한 스케일 테스트 (N=16 다이버스 AI)** — L1 p99 = 40.2 ms vs baseline 40.0 ms = **페널티 0.5 %**. AI aggregate throughput ~5,000+ tok/s. Fault-isolated. 이것이 목표 프로덕션 토폴로지.

**4장 결론**: MIG CP + MPS on AI는 **L1 baseline + 전체 AI throughput + fault 격리** 세 가지를 동시에 달성. 다른 조합은 하나도 이걸 못한다.

---

## 5장 · 현실 배치 시나리오

발견을 실제 배치 형태에 적용.

![F18](analysis_chain19/figures/mig_mps/F18_realistic_softbank.png)

**SoftBank AITRAS 스타일 배치** (5G L1 + 6 AI service on 1× A100). MIG CP + MPS on AI만이 50 ms L1 SLA 통과. 다른 naive/tuned 대안 모두 L1, AI throughput, 혹은 둘 다에서 실패.

![F19](analysis_chain19/figures/mig_mps/F19_diverse_stack.png)

**6-workload 다이버스 스택 비교**. CP는 AI가 다양하든 6× 동일 NRx든 baseline 유지. SP는 둘 다 파괴 (다이버스에서 65 ms, 동일 NRx에서 150 ms).

![F20](analysis_chain19/figures/mig_mps/F20_fault_isolation.png)

**적대적 조건에서 fault 격리**. MIG cross-partition은 하드웨어 수준 보호: AI SIGKILL / docker kill / OOM 발생 시 L1은 완전 무영향. Full GPU + MPS는 일시적 영향. SP는 AI 크래시가 공유 MPS 서버를 통해 전파되어 최대 영향.

![F21](analysis_chain19/figures/mig_mps/F21_sla_compliance.png)

**토폴로지별 5G TTI SLA 준수도**. Multi-GPU와 MIG CP + MPS만 통과. 공격적 튜닝된 SP (pct=30) 제외한 모든 same-partition 변형 실패.

![F22](analysis_chain19/figures/mig_mps/F22_violation_heatmap.png)

**Same-partition config들의 SLA 위반 확률 heatmap**. 모든 셀이 N≥4에서 ≥50 % 위반. Cross-partition은 (표시 안 함 — 균일 0 %) 유일한 안전 지대.

**5장 결론**: 현실 배치 시나리오 (SoftBank 스타일 AI-RAN, 다이버스 AI 스택, 적대적 fault injection) 모두 같은 답으로 수렴: **MIG CP + MPS on AI**.

---

## 6장 · MIG + MPS 내부에서의 최적화

토폴로지가 고정 (MIG CP + MPS on AI) 됐다면, 튜닝 레버는 무엇이 남는가?

![F23](analysis_chain19/figures/mig_mps/F23_pct_within_cp.png)

**CP 토폴로지 내부의 MPS thread% cap** — L1은 pct와 무관 (L1은 반대 파티션에 격리됨). AI throughput은 pct cap에 따라 스케일. 따라서 CP 토폴로지에서 MPS pct는 순수 AI측 레버.

![F24](analysis_chain19/figures/mig_mps/F24_pct_within_sp.png)

**SP 토폴로지 내부의 MPS thread% cap** — 대비용. 모든 셀 ≥45 ms. 최선의 SP 결과도 CP의 40 ms에 미치지 못함.

![F25](analysis_chain19/figures/mig_mps/F25_cell_count_sla.png)

**L1 cell-count SLA 확장** — L1 alone은 선형. SP breakdown 하에서 페널티 비율은 거의 일정. MIG CP + MPS 하에서는 alone 곡선이 유지됨.

![F26](analysis_chain19/figures/mig_mps/F26_worker_config.png)

**MPS 튜닝 진행** — SP + MPS 기본값 → SP + MPS pct=30 → MIG CP + MPS. 각 단계가 L1 duty 개선. 그러나 CP + MPS가 불변 상한선; SP 튜닝은 도달 불가.

![F27](analysis_chain19/figures/mig_mps/F27_recovery_dynamics.png)

**동적 부하 하에서의 복구 다이나믹스** — SP는 AI 부하 스파이크에 따라 진동. MIG CP + MPS는 부하와 무관하게 평평. 부하 불변성 = 예측성 = SLA 보장.

**6장 결론**: MIG CP + MPS 토폴로지 내부에서 MPS thread% 튜닝은 AI 파티션 안에서 AI/L1 균형을 조정. L1은 하드웨어로 보호되어 AI 측에서 깨뜨릴 수 없다.

---

## 7장 · 판정 및 배치 권고

![F28](analysis_chain19/figures/mig_mps/F28_master_decision.png)

**마스터 결정 매트릭스** — L1 지연, AI throughput, fault 격리, N≥6 확장, SLA 준수. Multi-GPU와 MIG CP + MPS만 5/5.

![F29](analysis_chain19/figures/mig_mps/F29_decision_tree.png)

**배치 결정 트리**:
1. Multi-GPU 가능? → 사용 (최대 용량, 최쉬움).
2. Single GPU만?
   - AI 수 ≤ 5? → MIG CP + MPS 여전히 선호; SP + MPS pct=30은 예산 fallback.
   - AI 수 > 5? → MIG CP + MPS **필수**.

![F30](analysis_chain19/figures/mig_mps/F30_cost_benefit.png)

**L1 지연, AI throughput, fault 격리 축의 cost-benefit 점수**. Multi-GPU와 MIG CP + MPS 모두 **30/30 만점**. 모든 same-partition 및 Full GPU 변형은 ≥1 축에서 실점.

---

## 프로덕션 설정 레시피

```
GPU: A100 (MIG 프로파일 지원 40GB/80GB 변형)
MIG 토폴로지: Config A (L1은 4g.20gb + AI는 3g.20gb)  OR
              Config C (L1은 3g.20gb + AI는 2g.10gb × 2)

L1 측 (4g 파티션):
  - 컨테이너: cuPHY / pyaerial 5G DU
  - MPS 불필요 (파티션당 단일 클라이언트)
  - CUDA_VISIBLE_DEVICES = L1 파티션의 MIG-<UUID>

AI 측 (3g 파티션):
  - CUDA MPS daemon 시작: nvidia-cuda-mps-control -d
  - 클라이언트별 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE (워크로드에 따라 30-100 튜닝)
  - N개 AI 컨테이너가 MPS 서버 공유
  - 각 컨테이너: CUDA_VISIBLE_DEVICES = AI 파티션의 MIG-<UUID>

기대 결과:
  L1 p99 지연: baseline (기준 cell 수에서 ~40 ms)
  AI throughput: 3g 파티션의 SM/memory 포화까지 N에 따라 확장
  Fault 격리: AI 크래시가 L1을 건드리지 못함
```

---

## 중요한 지표들 (SLA-first, duty-first 아님)

| 지표 | 측정 대상 | 게이트로 쓰는가? |
| ---- | -------- | -------------- |
| L1 per-iteration p99 지연 (ms) | 5G L1 SLA 준수 | **YES** — 1차 |
| AI aggregate throughput (tok/s, iter/s) | AI 용량 전달 | **YES** — 2차 |
| Fault 복구 지연 | AI 크래시 복구 시간 | **YES** — 3차 |
| L1 duty cycle (%) | 커널-시간 분율 | **NO** — 오도 (SLA 실패 중에도 높아 보일 수 있음) |
| NCU issued warps per SM | 마이크로아키텍처 스트레스 | 진단용만 |

Chain 19의 교훈은 duty cycle이 GPU-utilization 지표이지 SLA 지표가 아니라는 것. 지연으로 최적화하라, utilization으로 하지 말라.

---

## 캠페인 전체를 통해 검증한 것

- **Chain 17** (108 조건): 3 config × 6 N × 2 MPS 전 그리드 × 3 trial → MPS는 AI 다중화에 필수, MPS-off는 파국.
- **Chain 18** (다중 파트): fault injection, NCU per-kernel 지표, 동적 스케일링 → CP fault 격리, 커널 warp-stall 원인 확인.
- **Chain 19** (273 조건, 213 per-iter): N=16까지의 CP 불변성, N=6에서 SP breakdown, 최선의 SP 튜닝으로 MPS pct=30 → CP + MPS를 불변 조합으로, CP 내부의 MPS pct 튜닝을 2차 레버로 확인.

---

## 다음 단계

- **Chain 20**: 프로덕션 replica — cuPHY + Aerial CTL + 6-service AI 스택 (Qwen/Whisper/BERT/Qwen-VL/NRx/CsiNet) MIG CP + MPS 하 24시간 endurance.
- MIG CP + MPS vs SP fallback 에서의 AI 서비스 SLO (Qwen p99 요청 지연, Whisper 지연 등) 측정 → fallback 선택 시 서비스당 throughput 비용 정량화.
- 80GB 변형 및 7g 파티션 MIG config 테스트 — 단일 7g 파티션의 Full GPU MIG가 Full GPU처럼 행동하는지, MIG scheduler에서 뭔가를 얻는지.

---

*보고서 생성일 2026-08-03. 그림 30장 · `analysis_chain19/figures/mig_mps/`. 영문판: `MIG_MPS_COMBINED_REPORT.md`.*
