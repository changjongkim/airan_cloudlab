# AI-RAN 공유 GPU 연구 워크스페이스

**기준일:** 2026-09-25
**현재 연구:** SoftWall — MIG를 사용하지 않는 공유 GPU(MPS)에서 optional NeuralRx의 조건부 복구 의무를 인증하는 runtime substrate
**현재 단계:** 원고 provenance 정리 완료, production P2 frozen 개발 gate 995/1,000으로 실패; live-DU timing과 native cuPHY fast path가 남음
**투고 목표:** ACM SIGMETRICS 2027 Winter 사이클 — abstract 2027-01-04 23:59 AoE, paper 2027-01-11 23:59 AoE, 통보 2027-03-10 (공식 CFP, 2026-09-25 확인)
**최신 스냅샷 태그:** `softwall-sigmetrics-snapshot-20260925` (commit `2a97f8b`)

이 파일은 저장소의 단일 진입점이다. 연구 경과, 현재 결론, 기각된 주장, 미해결 사항, 알려진 오류를 모두 기록한다.

---

## 1. 연구 문제

AI-and-RAN 플랫폼은 무선 PHY 처리와 외부 AI 추론을 같은 GPU에서 실행하려 한다. NVIDIA MPS는 여러 CUDA process의 동시 실행을 제공하지만 무선 경로에 대한 시간 계약을 제공하지 않는다.

NeuralRx가 optional인 경우 문제가 추가된다. NeuralRx 결과가 확정되기 전까지 같은 transport block(TB)의 conventional receiver 경로는 실행 여부가 정해지지 않은 **미래의 필수 작업**으로 남는다. 이 작업은 아직 어떤 queue에도 들어가지 않았으므로, 현재 GPU가 비어 있다는 관측만으로 외부 AI를 먼저 실행하면 이후 복구 구간을 침범할 수 있다. 여러 cell과 여러 RAN home의 조건부 복구 의무는 공유 복구 자원에서 충돌할 수 있다.

## 2. 스킴: SoftWall

### 2.1 핵심 정의

- **Recovery debt:** 미확정 NeuralRx 결과 `i`가 유도하는 conventional 복구 작업 `R_i = (a_i(t), d_i − g_i, c_i, h_i)`.
- **All-fail certificate:** 미해결 debt 집합 `U(t)` 전체가 동시에 실패해도 모든 복구가 release, deadline, 자원 용량, 비중첩 조건을 만족하는 실행 가능 일정.
- **런타임 상태:** `X_t = (S, Q, T, L, E, C, G)` — 복구 일정, endpoint credit, IPC/P2P transport generation, AI lease, 물리 fence와 worker epoch, radio commit 상태, 다중 home AI 소유권.

### 2.2 불변식과 명제

- **커밋 조건:** 후보 상태 `X'`는 다음을 모두 만족할 때만 `X_t`를 대체한다. 그렇지 않으면 `X_t`는 변경되지 않는다.
  ```text
  VALID(X') = 복구 일정이 실행 가능
            ∧ endpoint·transport credit이 유일
            ∧ AI 완료가 자기 deadline과 복구 guard보다 앞섬
            ∧ 모든 generation과 worker epoch이 일치
  ```
- **Lemma 1 (all-fail dominance):** NeuralRx 성공이 자기 복구만 삭제하고 남은 작업의 demand를 늘리지 않으면, `U(t)`에 대한 certificate는 모든 실패 부분집합에 대해 유효하다. `2^|U|` 열거가 필요하지 않다. 성공이 새 필수 작업을 만들거나 다른 복구의 bound를 바꾸는 mode에는 적용되지 않는다.
- **Lemma 1b (disjoint-home composition):** 복구 lane이 서로 분리된 home의 certificate는 합집합으로 합성된다. 복구 GPU를 공유하면 합성되지 않으므로 전역 certificate를 구성한다.
- **결정 구간 조건:** home `h`에 동일 길이 복구 `m_h`개가 deadline 꼬리에 남았을 때, AI transaction을 먼저 실행하는 충분조건은
  ```text
  T_dec,h + b_eff ≤ d_h − g_h − m_h · c_h
  ```
  이다. 같은 debt 수와 AI class가 88 ms에서 안전하고 89 ms에서 불안전한 이유를 이 식이 설명한다.
- **Proposition 2 (현재의 idle은 불충분):** 단일 비선점 복구 lane, 공통 guard 경계 `D`, 미해결 복구 bound `c_1…c_m`, AI transaction bound `b_eff`에서 mandatory-only 실행은 가능하지만 `t + b_eff + Σ c_i > D`이면, 시각 `t`의 GPU idle만으로 AI를 수락하는 정책은 무선 계약을 보장할 수 없다.

### 2.3 실행 절차

1. **Mandatory-first admission:** 무선 release 시 conventional 복구를 먼저 예약한다. NeuralRx는 전체 경로 bound가 fallback cutoff 안에 들어가고 endpoint가 generation이 일치하는 ring credit을 가질 때만 수락한다.
2. **Outcome batch 적용:** 공통 cutoff의 NeuralRx 결과를 한 묶음으로 적용한다. 적시 성공은 debt를 삭제하고, 실패·지연·증거 없음은 debt를 유지한다.
3. **AI lease와 launch-time enforcement:** worker에 절대 latest-start 시각을 전달하고, 그 이후 CUDA kernel을 발사하지 않는다. launch control 구간보다 오래된 certificate는 폐기하고 재계산한다.
4. **원자 commit:** 복구 재배치와 AI lease 발급을 한 generation으로 commit한다.
5. **Lease 회수:** 일치하는 물리 완료 fence 또는 자격 있는 non-launch 기록으로만 회수한다. 증거 없는 timeout은 credit을 quarantine한다.

### 2.4 설계 요소의 근거

각 설계 요소는 측정된 반례에서 도입했다.

| 반례 | 깨진 가정 | 도입한 요소 |
|---|---|---|
| Local-safe / global-unsafe 복구 | local certificate가 공유 lane에서 합성된다 | 전역 obligation 집합과 전역 certificate |
| 23.530 ms dispatch tail | certificate가 kernel launch 시점까지 유효하다 | Bounded revalidation과 물리 latest-start |
| 순차 common-cutoff 재생 | 동시 결과를 임의 순서로 적용할 수 있다 | Atomic outcome batch |
| 고정 AI lease | AI 실행시간이 launch control을 포함한다 | Launch-time control과 AI blackout |
| 효과 발생 후 reply 손실 | 모호한 연산의 재시도는 무해하다 | Generation quarantine과 fail-closed admission |
| 두 개의 미완료 prepare | at-most-once RPC가 소유권 가시성을 보장한다 | Single unlaunched-token 불변식 |
| Warm bound 재사용 | 배치와 수명주기가 service tail을 바꾸지 않는다 | Provenance 범위의 자격 검증 |

### 2.5 Feasibility 분류

debt 수, AI class, 판단 시각, 수명주기 provenance를 입력으로 실행 전에 분류한다.

| 분류 | 의미 |
|---|---|
| QSU | 안전하며 AI에 사용할 여유가 있음 |
| QSN | 안전하지만 여유가 없음 |
| MI | 필수 부하만으로 실행 불가능 |
| UQ | 자격 검증되지 않음 |

### 2.6 검증된 mode

```text
P = 180 ms, D = 155 ms, guard = 2 ms
B_NRx = 45 ms, B_conv = 25 ms, B_AI(context64) = 35 ms

4-debt all-fail      45 + 4×25 + 2      = 147 ms ≤ 155   수락
5-debt all-fail      45 + 5×25 + 2      = 172 ms > 155   거절
2 성공 + AI          45 + 2×25 + 35 + 2 = 132 ms         AI lease 가능
```

`P180/D155`는 harness 계약이며 5G NR slot 주기나 production HARQ 기한이 아니다.

## 3. 확정된 결과

모든 결과는 A100-SXM4-40GB, MIG OFF, MPS ON, warm persistent process의 유한 표본이다. WCET가 아니다.

| 실험 | 결과 | 원본 |
|---|---|---|
| C159-Q2 가변 context | 두 노드, 1,200 rounds, 실제 TensorRT NeuralRx 4,800건(성공 3,488), 물리 복구 1,312건, Qwen 1,077 unit, deadline miss 0. NeuralRx release→complete p99 8.10 ms, max 18.12 ms. Certificate가 context256 55건, context512 68건을 거절 | `results/softwall_multigpu/confirm159_q2_variable_two_node.json` |
| C158 반복 자격 검증 | 두 노드, NeuralRx 2,000건(성공 1,440), 물리 복구 560건, Qwen 500 unit, deadline miss 0, recovery contract 위반 0 | `results/softwall_multigpu/confirm158_repeated_actual_nrx_two_node.json` |
| C161 fault 자격 검증 | NeuralRx 2,800건, 물리 복구 942건, correlated all-fail 복구 240건, stale/duplicate NeuralRx 20쌍, stale/duplicate 복구 20쌍, terminal channel fault 4건, deadline miss 0 | `results/softwall_multigpu/c161_full_fault_qualification.json` |
| C156 GPU timeline | Qwen kernel 2,448개와 복구 kernel 212개에서 금지된 중첩 0 ns, 미포착 worker event 0 | `results/softwall_multigpu/confirm156_156b_two_node_gpu_timeline.json` |
| C162 경계 예측 | 두 노드, 독립 seed, holdout에서 case 순서 역전, 180 rounds. NeuralRx 720건, 주입 복구 330건, Qwen 90 unit, deadline miss 0. context64가 88 ms에서 30/30 수락, 89 ms에서 30/30 거절 | `results/softwall_multigpu/c162_boundary_two_node.json` |
| C162 scheduler 확장성 | 현재 mode 16,023개 상태에서 exact checker와 불일치 0. 소규모 600개 상태에서 false-safe 0, false-conservative 10(1.7%). 64-debt 결정 p99 1.494 ms, 검증 p99 0.131 ms | `results/softwall_multigpu/c162_scheduler_scalability_v1.json` |
| C135 static 대비 | 288개 분기에서 static 계약은 거절하고 conditional certificate는 안전하게 수락하는 exchange 10건 | `results/softwall_multigpu/confirm135_static_counterfactual_audit_v1.json` |
| 필요성 증거 | 사전 고정된 C162 상태 2개에서 debt-blind AI 수락 시 선언 bound 기준 완료가 guard를 1–12 ms 초과. SoftWall은 두 노드에서 60/60회 GPU launch 전에 거절 | `results/softwall_multigpu/softwall_necessity_witness_v2.json` |
| Envelope v16 | QSU 6, QSN 0, MI 3, UQ 13. v15의 three-point mode를 강등하고 v16 four-point mode를 승격. prepare/commit/complete/abort 네 연산에 서로 다른 물리 노드 4대에서 fault 8 arm, 모든 fault 뒤 무선 지속. RAN-critical 연산은 commit뿐 | `results/softwall_multigpu/softwall_envelope_v16_validation_summary.json` |
| C121/C122 sharded home | 2 GPU·8 cell 5,440 TB, 4 GPU·12 cell 8,160 TB, home 간 release 차이 0 ns, safety 위반 0 | `docs/current/SOFTWALL_FORMAL_MODEL_KO.md` §9 |
| P3 채널 호환성 | Sionna CDL-D: conventional 153/250, NeuralRx 164/250. CDL-E: 155/250, 163/250. 저 SNR에서 NeuralRx만 성공 31건, conventional만 성공 12건, paired exact p = 0.00540. 범위는 Sionna CDL-D/E, 100 ns, MCS7, FP32 | `results/softwall_same_gpu/sionna_cdl_de_holdout_gate_job58868184.json` |

## 4. 기각된 주장

아래 주장은 실험으로 기각했으며 원고에 음성 결과로 포함한다.

| 주장 | 실험 | 결과 |
|---|---|---|
| Joint optimizer가 max-radio보다 우수하다 | C102 | 실행 가능 39개 상태에서 선택 차이 0, AI 차이 0 |
| AI-first retiming이 처리량을 높인다 | C113, 원고 §6 | BurstGPT 4,290 requests, 1,129,504 offered tokens에서 SoftWall과 certificate-preserving recovery-first 모두 timely 931 requests, 385,262 tokens. 복구 전체 bound 부과 시 +0.066%로 사전 등록한 5% 기준 미달. Confirmatory holdout은 열지 않음 |
| 같은 2-GPU 예산에서 처리량 우위가 있다 | C115 | +0.080% / +0.138%, 두 신뢰구간 모두 0 포함 |
| Global routing이 처리량을 높인다 | C124, C126 | C126 radio decision parity 실패, 우위 미지지 |
| NRx 개수를 안정적 AI 비용 스칼라로 쓸 수 있다 | C104, C105 | 순서별 부호 반전 |
| MPS cap이 격리를 제공한다 | C26 | cap100 2/1,500, cap20 1/1,500 miss |
| 부하를 올리면 eager-dual이 먼저 실패한다 | C51 | 1 cell, P90/45/25/12에서 eager miss 0 |

385,262 동률은 certificate의 필요성을 기각하지 않는다. 비교 대상은 초기 all-fail admission을 사용하는 certificate-preserving 정책이며, 동률은 AI-first retiming이 이 trace에서 추가 처리량을 만들지 못했음을 뜻한다.

## 5. Production exit gate

판정 원본: `results/softwall_multigpu/softwall_production_gates_current_v3.json`, `results/softwall_multigpu/softwall_production_exit_gate_v3.json`

| Gate | 판정 | 내용 |
|---|---|---|
| P1 LIVE_DU_CLOCK | FAIL | Target DU의 timing contract 부재. 첫 번째 차단 요소 |
| P2 FAST_PATH | FAIL | persistent-input path 4.5 ms 이내 995/1,000, 초과 5건. frozen 개발 gate 실패로 독립 holdout 미개방 |
| P3 CHANNEL_COMPATIBILITY | PASS | Sionna CDL-D/E 한정 |
| P4 INTEGRATED_REQUALIFICATION | FAIL | 통합 production 재자격 미수행 |

`claim_decision`: production HARQ 보장 REJECT, 외부 TDL NeuralRx 지원 REJECT, Sionna CDL-D/E NeuralRx 지원 FINITE_SAMPLE_PASS, synthetic qualified substrate RETAIN.

### 5.1 Production timing 기준

Aerial checkout의 testMAC 설정에서 확인한 indication threshold는 다음과 같다.

```text
early HARQ          T0 + 2.0 ms
UL indication       T0 + 4.5 ms
PRACH indication    T0 + 4.5 ms
UCI indication      T0 + 4.5 ms
```

`scf_fapi_handler::validate_indication_timing`은 SFN/slot으로 `T0`를 복원하고 handler 진입 시각이 threshold를 넘으면 late로 센다. NVIDIA는 testMAC을 controlled environment용 L2 개발 도구로 정의하므로 이 값을 field DU의 production `d_MAC`으로 취급하지 않는다.

### 5.2 4.5 ms 경로 진단

Clean synthetic PUSCH, AI 없음, warm persistent process 표본이다. 원본: `docs/current/SOFTWALL_PRODUCTION_EXIT_PLAN_KO.md` §1.2

| 경로 | 4.5 ms 이내 | p50 | p99 | 판정 |
|---|---:|---:|---:|---|
| Local NeuralRx 후 conventional | 0/1,000 | 6.609 ms | 6.984 ms | 순차 fallback 기각 |
| Same-GPU speculative dual path | 0/1,000 | 6.828 ms | 17.552 ms | GPU 경합으로 기각 |
| GPU0 LS + GPU1 NRx + GPU0 conventional | 0/1,000 | 5.942 ms | 11.897 ms | GPU0 직렬화로 기각 |
| Raw-IQ P2P, GPU1 full NeuralRx | 852/1,000 | 3.831 ms | 9.506 ms | tail로 FAIL |
| 위 경로, Python GC OFF | 838/1,000 | 3.988 ms | 10.041 ms | GC 원인 가설 기각 |
| 위 경로, caller-owned same-stream + busy poll | 885/1,000 | 3.929 ms | 11.696 ms | tail로 FAIL |
| 위 경로, persistent input + stream ordering | **995/1,000** | **3.813 ms** | **3.995 ms** | correctness 1,000/1,000, late 5로 FAIL |

### 5.3 Stage 귀속

원본: `results/softwall_multigpu/c163_raw_p2p_v5_stage_profile_analysis_job58868184.json`의 `stages.*.gpu_ms`와 `copy`. 300 timed unit 중 timely 265, late 35. Profiling은 timing을 바꾸므로 기전 진단 전용이다. AI를 실행하지 않은 경로에서 측정했다.

GPU 실행 시간 (ms):

| Stage | p50 | p99 | max | timely 평균 | late 평균 | pair wall과 Pearson |
|---|---:|---:|---:|---:|---:|---:|
| cuPHY LS channel estimation | 0.866 | 4.873 | 9.592 | 0.930 | 3.972 | 0.960 |
| TensorRT graph (NeuralRx) | 0.857 | 0.886 | 0.887 | 0.862 | 0.865 | 0.076 |
| CRC | 0.246 | 0.321 | 0.817 | 0.251 | 0.256 | 0.022 |
| LDPC decode | 0.196 | 0.207 | 0.252 | 0.197 | 0.200 | 0.136 |
| derate_match | 0.154 | 0.168 | 2.254 | 0.153 | 0.216 | 0.081 |
| tensor_preparation | 0.030 | 0.031 | 0.031 | 0.030 | 0.030 | 0.023 |
| dmrs_removal | 0.007 | 0.008 | 0.008 | 0.007 | 0.007 | 0.001 |

P2P copy GPU 시간: forward p50 20.7 µs, p99 69.3 µs, max 85.8 µs. Backward p50 15.1 µs, p99 26.4 µs, max 43.3 µs.

Channel estimation은 timely unit과 late unit 사이에서 GPU 시간이 달라지는 유일한 stage이며(평균 0.930 ms 대 3.972 ms), pair latency와의 상관계수가 0.960이다. TensorRT graph의 GPU 시간은 약 0.86 ms로 4.5 ms 예산의 약 19%를 차지하지만 분산이 작고(p99 0.886 ms), timely와 late 사이의 평균 차이가 0.003 ms이며, pair latency와의 상관계수가 0.076이다. P2P 전송은 100 µs 미만이다.

이 진단 뒤 C165는 per-request complex allocation과 redundant pre-CE host sync를 제거했다. Frozen 개발 gate는 995/1,000으로 개선됐지만 통과하지 못해 holdout을 열지 않았다. Late 5건 중 4건은 GPU0 conventional completion, 1건은 remote NeuralRx였고 pair latency 상관도는 conventional GPU 0.918, remote NeuralRx 0.331이었다. Profiler-only conventional 진단에서는 equalization이 total service와 가장 강하게 연결됐다(`r=0.761`). 따라서 남은 4.5 ms 위반은 remote CE 하나가 아니라 양 GPU의 cuPHY service tail이다.

Host enqueue 시간(`host_enqueue_us`)은 GPU 시간과 별개로 기록되어 있다. Channel estimation host enqueue는 p50 0.578 ms, p99 4.595 ms, max 9.283 ms이고, TensorRT graph host enqueue는 p50 7.3 µs이다. TensorRT의 7.3 µs는 enqueue 비용이며 NeuralRx 실행 시간이 아니다.

## 6. 주장 범위

- MPS는 실행 기반이며 격리 수단이 아니다. 보호는 admission, 예약, 수명주기 계약에서 나온다.
- 모든 bound는 유한 표본 whole-path bound이다. 관측 위반 0은 WCET나 실패 확률 0을 증명하지 않는다.
- 자격 검증은 노드, GPU, 소프트웨어, 배치, 수명주기 지문에 한정된다. 범위 밖 mode는 재자격 검증이 필요하다.
- 하드웨어는 A100 계열만 검증했다. 다른 GPU 계열은 검증하지 않았다.
- Fault model은 임의의 GPU/driver hang과 완전한 process 교체 복구를 제외한다.
- 무선 결과는 synthetic 채널과 Sionna CDL-D/E에 한정된다. Aerial TDL-A와 field IQ는 포함하지 않는다.
- 처리량 optimizer 우위는 주장하지 않는다.

### 6.1 수명주기 자격

원본: `results/softwall_multigpu/c164_lifecycle_qualification_summary_v1.json`. 10개 mode 중 5개가 자격 또는 부분 자격, 5개가 UQ이다.

| Mode | 상태 |
|---|---|
| warm_persistent | QUALIFIED_BOUNDARY_SUBSET |
| idle_30s_first | QUALIFIED_BOUNDARY_SUBSET |
| mps_restart_first | QUALIFIED_AFTER_REQUALIFICATION_SUBSET |
| qwen_reload_first | QUALIFIED_MANDATORY_ONLY_SUBSET |
| worker_channel_reconnect_same_epoch | QUALIFIED_RECONCILIATION_SUBSET |
| cold_first | UQ_NO_WHOLE_MODE_EVIDENCE |
| gc_on | UQ_OBSERVED_BOUND_FAILURE |
| idle_5m_first | UQ_NO_PHYSICAL_SAMPLE |
| idle_30m_first | UQ_NO_PHYSICAL_SAMPLE |
| worker_process_replacement | UQ_SINGLE_NODE_DEVELOPMENT_ONLY |

MPS restart와 Qwen reload 중의 가용성은 주장하지 않는다.

## 7. 알려진 문제와 조치 상태

2026-09-25 감사에서 확인한 항목과 현재 상태이다.

| # | 문제 | 상태 |
|---|---|---|
| 1 | `RESEARCH_PLAN_SOFTWALL_KO.md`의 NeuralRx 용량 수치가 두 출처를 혼합했다. 문서는 "full A100 1164.1 req/s, 서비스 1.34 ms"로 기재했으나 `NRX_CAPACITY.csv`의 full GPU 1 replica 값은 1130.5 req/s, 평균 0.882 ms, p99 1.107 ms이다. 1.34 ms는 4g MIG 행(745.1 req/s)의 값이다. 같은 문서의 P2P 76.84 µs는 같은 GPU 안 MIG 쌍의 단일 process 측정값이다. | 문서를 `docs/archive/`로 이동하고 보관 사유를 첫 줄에 기재했다. 본문 수치는 이력으로 남아 있으므로 인용하지 않는다. |
| 2 | `analyze_mig_mps_combined.py`와 `MIG_MPS_COMBINED_REPORT.md`가 "SP + MPS pct=30"을 45 ms로 기재했다. 실측은 N=6에서 145.9 ms이다. | 두 파일 모두 145.9 ms로 정정했다. |
| 3 | 원고 §9의 long-tail 서술에 수치가 없었다. | `main.tex`에 C158 attempt 4의 NeuralRx 완료 350.948 ms를 기재했다. 원인은 미확정이다. |
| 4 | `fallback_start_ns`는 예약 시각이며 실제 시작 시각이 아니다. | 원고의 시간 판정은 commit return을 기준으로 한다. Fallback의 실제 GPU 시작 시각은 계측하지 않는다. |
| 5 | `docs/current/`의 `SOFTWALL_*.md`가 74개였고 노벨티 판정 문서가 시점별로 중복되었다. | `docs/current/`를 17개 문서로 줄이고 나머지를 `docs/archive/`로 옮겼다. |
| 6 | 일부 이전 문서가 참조하는 `task1_final/gdr_pool_20260814T014651Z/`, `results/isca_v2/mig_causal_20260813T1138Z/`, `cloudlab_final_snapshot_20260814/`가 저장소에 없다. `task1_final/`에는 `chain/`만 있다. | 미해결. 해당 문서는 `docs/archive/`에 있으며 현재 주장의 근거로 사용하지 않는다. |
| 7 | `SOFTWALL_SIGMETRICS27_SUBMISSION_PLAN_KO.md`가 Fall 마감(abstract 2026-10-02, paper 2026-10-09)을 기재했다. | **해결.** 공식 CFP를 재확인하고 Winter abstract 2027-01-04, paper 2027-01-11, notification 2027-03-10 AoE로 갱신했다. |

## 8. 먼저 읽을 문서

| 순서 | 문서 | 내용 |
|---|---|---|
| 1 | `paper/softwall_sigmetrics27/main.pdf` | 투고 원고 (15쪽, 참고문헌 포함) |
| 2 | `docs/current/SOFTWALL_MANUSCRIPT_DRAFT_EN.md` | 원고의 Markdown 원본 |
| 3 | `docs/current/SOFTWALL_FORMAL_MODEL_KO.md` | 형식 모델, Lemma, 입증 의무와 현재 증거 |
| 4 | `docs/current/SOFTWALL_NOVELTY_DEFENSE_MATRIX_KO.md` | 가장 가까운 선행 연구, 중복 주장, 예상 반론 |
| 5 | `docs/current/SOFTWALL_RELATED_WORK_AUDIT_KO.md` | 선행 연구 감사 |
| 6 | `docs/current/SOFTWALL_PRODUCTION_EXIT_PLAN_KO.md` | Production gate와 4.5 ms 진단 |
| 7 | `docs/current/SOFTWALL_STRONG_BASELINE_SPEC_KO.md` | 강한 결합 baseline의 공정 비교 명세 |
| 8 | `docs/current/SOFTWALL_RESEARCH_GUIDELINE_KO.md` | 실험 선정·측정·증거 등급·주장 규칙 |
| 9 | `results/softwall_same_gpu/EXPERIMENT_GATE_LEDGER_KO.md` | 단일 GPU 실험의 사전 고정 PASS/FAIL 원장 |
| 10 | `docs/current/CURRENT_RESEARCH_INDEX_KO.md` | 전체 문서 인덱스 |

### 8.1 기계 감사

| 감사 | 결과 | 원본 |
|---|---|---|
| 원고 주장 감사 | 13/13 PASS (`forbidden_overclaim_absent`, `negative_performance_result` 포함) | `results/softwall_multigpu/softwall_manuscript_claim_audit_v1.json` |
| 투고 감사 | 25/25 PASS | `results/softwall_multigpu/softwall_sigmetrics_submission_audit_v1.json` |
| C162 재현성 manifest | 83개 파일 | `results/softwall_multigpu/c162_artifact_manifest.json` |

## 9. 연구 경과

### 9.1 CloudLab MIG–NRx / DART-Rx (2026-05 ~ 2026-08)

CloudLab d8545(A100 ×4, ConnectX-6 Dx)에서 MIG 기반 L1–NeuralRx 배치를 연구했다.

- MIG local, MIG+MPS, Full MPS, cross-partition P2P, NIC GDR의 다섯 배치를 비교했다. Same-partition 배치의 L1 active-time 증가율은 1.601×(Full MPS), 1.621×(MIG local), 1.702×(MIG+MPS)였고, cross-partition P2P는 1.043×였다.
- 독립 NeuralRx process를 1개에서 8개로 늘리면 full A100 MPS에서 20-cell L1 p99가 42.3 ms에서 189.3 ms로, 4g MIG 안의 MPS에서 40.7 ms에서 435.7 ms로 증가했다.
- chain19(273개 조건)에서 MIG cross-partition + AI측 MPS는 N=6–16에서 L1 p99 39.5–43.8 ms로 baseline 38.5 ms에 근접했다. Same-partition MPS는 pct=30 최적 설정에서도 N=6 145.9 ms, 기본 pct=100에서 411.3 ms였다.
- 고정 MIG에서 NeuralRx capacity가 partition별로 고정되는 fragmentation 문제를 다루는 DART-Rx(admission, reservation, expiry, single commit, lease)를 설계했다.

관련 문서: `docs/archive/RESEARCH_WALKTHROUGH_KO.md`, `docs/archive/MIG_NRX_DART_RESEARCH_SYNTHESIS_KO.md`, `results/20260803/`

### 9.2 Perlmutter no-MIG 측정 (2026-06)

Perlmutter A100에서 MIG OFF 상태로 CloudLab 실험을 재측정했다.

- 기본 time-slicing에서 L1 + NeuralRx p99는 389 ms, MPS에서 40 ms였다.
- 메모리 대역폭 포화 워크로드(sat_hbm)에서 MPS p99는 6,985 ms로 time-slicing 426 ms보다 16.4배 나빴고, 5회 중 2회만 붕괴하는 bistable 거동을 보였다.
- GPU idle gap의 약 85%가 host의 `cudaFree`·memcpy 블로킹 구간과 시간적으로 겹쳤다.

관련 문서: `results/visual_evidence/PERLMUTTER_NOMIG_VISUAL_EVIDENCE_KR.md`, `results/perlmutter_handoff/`

### 9.3 방향 전환 (2026-09-19 ~ 2026-09-20)

- MIG를 제약 조건으로 두고 MPS 위에서 최적화하는 방향을 정했다.
- 4-GPU 노드의 L1 전용 GPU와 NVLink P2P NeuralRx pool을 전제로 한 연구 계획서를 작성했으나, 이후 같은 GPU / MIG OFF / MPS 검증으로 우선순위를 변경했다.

### 9.4 SoftWall 단일 GPU (2026-09-20 ~ 2026-09-22, confirm2–105)

- 같은 입력의 conventional·NeuralRx full receiver를 구성하고 수신기 상보성을 실측했다(−8.5 dB에서 conventional 180/500, NeuralRx 489/500, 합집합 491/500).
- 원자 복구 예약, single commit, bounded AI lease를 별도 MPS endpoint와 CUDA IPC로 연결했다.
- MPS cap(C26), admission quarantine(C29), 즉시 client 종료(C30)가 모두 deadline 보호에 실패함을 확인했다. C46에서 CPU sham 대조군 0/10,000, MPS client 12/10,000 miss(부호검정 p = 0.0039)로 GPU client 수명주기가 위험 요소임을 분리했다. C40 ABBA에서 첫 conventional fallback의 cold start(22.8 / 35.3 ms 대 warm-up 후 2.28 / 2.29 ms)를 확인했다.
- 1 cell 부하 진단(C51)에서 eager-dual은 모든 주기에서 miss 0이었다.
- 복수 recovery credit의 원자적 재배치와 AI lease transaction을 구현했다(runtime unit test 21개, fault regression 10,000회).
- C102에서 joint optimizer가 max-radio와 선택 차이 0으로 판정되어 optimizer 우위 주장을 종료했다.

### 9.5 SoftWall 멀티 GPU와 형식 모델 (2026-09-23 ~ 2026-09-24, C113–C150)

- 실제 BurstGPT trace와 Qwen2.5-1.5B로 강한 baseline을 비교했다(C113).
- local CUDA-IPC endpoint와 remote NVLink P2P endpoint를 같은 certificate에 연결했다(C114, C116, C119/C120).
- Component bound는 isolated scalar가 아니라 co-run class vector여야 함을 확인했다(C117 2 ms 후보 실패, C118 3 ms 재자격).
- Endpoint 추가로는 home receiver 메모리 한계가 줄지 않으므로 sharded home이 필요함을 계산하고 물리 검증했다(C121/C122).
- 형식 모델 v11–v16을 반복 개정했다. 각 개정은 물리 실험으로 강등 또는 승격을 판정받았다.
- Broker fault, control-point fault, abort, single-token 불변식을 검증했다(C127–C148).

### 9.6 통합, 경계 예측, 원고 (2026-09-24 ~ 2026-09-25, C151–C164)

- Shared cuPHY 복구 경로를 두 노드에서 검증했다(C151/C152).
- 전역 certificate의 사전 거절, NRx 성공에 따른 credit 해제, Qwen lease, shared cuPHY 복구를 통합 경로에서 확인했다(C153–C157).
- C158, C159에서 캠페인 규모 자격 검증을 수행했다.
- C161에서 7종 fault를 검증했다.
- C162에서 경계 예측과 scheduler 확장성을 검증했다.
- C163에서 production timing 기준을 Aerial 코드에서 확인하고 exit gate를 정의했다. P3는 Sionna CDL-D/E로 통과했다.
- C164에서 수명주기 matrix를 10개 중 5개까지 자격 검증했다.
- 영문 원고와 SIGMETRICS LaTeX 원고를 작성했다.

## 10. 작업 기록

### 10.1 저장소 관리와 계획·분석 문서

- `git@github.com:changjongkim/airan_cloudlab.git`를 `kcj/airan_cloudlab`에 changjongkim 계정으로 clone하고, 이전 로컬 사본은 `kcj/airan_cloudlab_old_20260919`로 보존했다.
- MPS 방향 연구 계획서(`docs/archive/RESEARCH_PLAN_SOFTWALL_KO.md`), 노벨리티 실행안(`docs/archive/SOFTWALL_NOVELTY_ACTION_PLAN_KO.md`), 연구 진행 가이드라인(`docs/current/SOFTWALL_RESEARCH_GUIDELINE_KO.md`)을 작성했다.
- 필요성 반론의 판정은 `docs/archive/SOFTWALL_NECESSITY_GAP_DECISION_KO.md`에 있으며, 결론은 원고 본문에 반영되었다.
- 실험 결과를 원본 파일과 대조해 수치 불일치, 하드코딩 상수, 누락 데이터셋을 확인했다(§7).
- 2026-09-25에 5일간의 미커밋 작업(2,393개 파일)을 commit `2a97f8b`로 올리고 태그 `softwall-sigmetrics-snapshot-20260925`를 생성했다.

### 10.2 실험, 모델, 원고

- confirm2–C164의 실험 protocol을 사전 고정하고 실행했다. 실패한 시도와 오염된 결과는 삭제하지 않고 보존했다.
- 형식 모델, certified scheduler, feasibility checker, 원고 주장 감사와 투고 감사를 구현했다.
- 영문 원고와 LaTeX 원고를 작성했다.

## 11. 다음 작업

1. 원고 정리: **완료.** 수치 provenance, C158 tail, P3 Results 승격, P2 stage attribution, Winter CFP와 문서 archive를 claim/submission audit으로 고정했다.
2. P1 target-DU timing contract 확보: 최소 schema, 세 획득 경로와 parametric bridge를 timing-contract 문서에 고정했다. 실제 target DU trace가 없으면 `UQ_NO_PRODUCTION_TRACE`를 유지한다.
3. P2 fast path: persistent-input 구현은 remote NeuralRx–pair 상관을 0.9603에서 0.3307로 낮췄지만 frozen gate가 995/1,000으로 실패했다. Native N0 parity fixture는 양 decoder의 같은 1,377-byte TB와 두 raw-IQ layout의 183,456개 복소 원소에 대한 C++ bitwise 일치로 통과했다. C++/CUDA IQ bridge도 fixture bitwise parity와 양 decoder 300/300 correctness를 통과했지만 진단 deadline은 298/300이었다. Persistent monolithic conventional은 300/300 correct였고 p50 1.014 ms였으나 max 5.363 ms였으며, 두 partial-native path를 결합한 C168은 293/300이었다. 따라서 N1/N2 완료나 timing qualification으로 세지 않는다. 다음 구현은 N1 GPU1 fused native NeuralRx와 N2의 fully native phase setup이다. 새 qualification은 live-DU `D`가 확보된 뒤 독립 holdout 1,000/1,000으로만 연다.

다음 항목은 수행하지 않는다: 새 메커니즘 추가, 나머지 수명주기 UQ 5개 채우기, 처리량 우위 재시도, P1 이전의 P4 통합 재자격, 새 문서 생성.

## 12. 실행 환경

| 항목 | 값 |
|---|---|
| 시스템 | NERSC Perlmutter |
| GPU | NVIDIA A100-SXM4-40GB ×4, NVLink, MIG OFF |
| GPU 공유 | NVIDIA MPS (`nvidia-cuda-mps-control`, compute node에서 daemon 실행) |
| 컨테이너 | Shifter, `nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb` |
| PHY | Aerial 25.3.2, pyAerial, cuPHY |
| NeuralRx | pyAerial neural receiver, ONNX → TensorRT, persistent binding과 CUDA Graph |
| AI 워크로드 | Qwen2.5-1.5B, BurstGPT trace |
| Slurm 계정 | `m1248_g`, `m5320_g` |
| MPS pipe | `$SCRATCH` 아래에 두어 Shifter가 같은 절대 경로로 mount하도록 한다 |

### 12.1 Git 설정

이 저장소는 changjongkim 계정으로 push한다. 전역 SSH 설정은 공유 계정 소유자의 것이므로 변경하지 않고 저장소 local 설정을 사용한다.

```text
core.sshCommand = ssh -i ~/.ssh/changjong_ed25519 -o IdentitiesOnly=yes
user.name       = changjongkim
user.email      = changjong5238@gmail.com
```

## 13. 디렉터리 구조

| 경로 | 역할 |
|---|---|
| `paper/softwall_sigmetrics27/` | SIGMETRICS LaTeX 원고와 참고문헌 |
| `docs/current/` | 현재 판단에 사용하는 문서 |
| `docs/current/figures/` | 원고용 그림 |
| `docs/architecture/` | DART-Rx 구조와 novelty 문서 |
| `docs/experiments/` | 이전 실험 계획과 campaign 문서 |
| `docs/setup/` | CloudLab 설치와 복구 절차 |
| `docs/tables/` | 실험 matrix와 요약 CSV |
| `docs/archive/` | 이전 가설과 보고서. 현재 결론으로 사용하지 않음 |
| `results/softwall_same_gpu/` | SoftWall 단일 GPU 실험 결과와 gate ledger |
| `results/softwall_multigpu/` | SoftWall 멀티 GPU, 형식 모델, production gate 결과 |
| `results/perlmutter_handoff/` | 2026-06 Perlmutter no-MIG 측정 스크립트와 결과 |
| `results/visual_evidence/` | MIG와 Perlmutter 시각 증거 문서 |
| `results/20260803/` | chain19 CloudLab MIG+MPS 결과 |
| `results/<날짜>/` | 날짜별 이전 실험 결과 |
| `scripts_for_node/softwall_same_gpu/` | SoftWall controller, runtime, 실행 스크립트 |
| `scripts_for_node/task1/` | DART runtime과 NeuralRx 직결 경로 |
| `scripts/` | Production gate 평가와 채널 분석 스크립트 |
| `scripts_node/` | CloudLab 실행 스크립트 |
| `task1_final/chain/` | 2026-08 chain 결과 |
| `task1_p2p_fair/` | P2P 공정 비교 실험 |
| `data/` | 데이터 카탈로그, BurstGPT 공개 trace |
| `tools/` | 분석 도구 |
| `archive/` | 과거 로그와 초기 실행 사본 |

## 14. 관리 원칙

- 새 연구 문서는 루트에 만들지 않고 `docs/`의 해당 분류에 둔다.
- 실험 protocol은 실행 전에 고정한다. `PASS`는 해당 사전 고정 계약에만 해당한다.
- 실패한 protocol과 오염된 결과는 삭제하지 않는다. 이후의 성공이 이전 실패를 대체하지 않는다.
- Raw result는 덮어쓰지 않고 새 파일로 저장한다. Per-run raw capture(`results/softwall_same_gpu/raw/`, `results/softwall_multigpu/raw/`)는 `.gitignore`로 제외한다.
- 원고의 정량 수치는 원본 파일과 필드로 추적할 수 있어야 한다.
- 결과 문장에는 조건을 붙인다. 관측된 위반 0을 WCET나 보장으로 서술하지 않는다.
