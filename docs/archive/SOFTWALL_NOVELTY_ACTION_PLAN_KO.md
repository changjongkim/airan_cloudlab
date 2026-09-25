> **보관 사유 (2026-09-25):** 이 문서는 실험 이력으로 보존한다. 최신 스킴과 claim boundary는 `docs/current`의 원고·모델·production gate 문서가 권위 기준이며, 과거 GPU0 전용, MIG, CloudLab 또는 4-GPU pool 가정은 현재 논문의 근거가 아니다.

# SoftWall 노벨리티 확보 실행안 — Confirm51 결과와 Confirm52 수명주기 검증 반영

**상태:** 이 문서는 Confirm51/52 시점의 실행안이다. Confirm55 이후의 실험 종료 판정과
후보별 시작·중단 기준은 [최신 결정](SOFTWALL_NOVELTY_DECISION_KO.md)을 우선한다.

**작성:** 2026-09-21 KST
**기준 자료:** [gate ledger](../../results/softwall_same_gpu/EXPERIMENT_GATE_LEDGER_KO.md) confirm2–50,
[S2 보고서](../../results/softwall_same_gpu/S2_NATURAL_CHANNEL_REPORT_KO.md),
[완료 점검](SOFTWALL_COMPLETION_AUDIT_KO.md), [선행연구 감사](SOFTWALL_RELATED_WORK_AUDIT_KO.md),
[기여 논지](SOFTWALL_NOVELTY_THESIS_KO.md)
**성격:** 측정 결과의 판정과 다음 실험의 중단 기준. Confirm52까지 독립 노벨리티는 확보되지 않았다.

---

## 1. 전체 결과를 증거 강도로 재분류하면

기존 ledger는 시간 순서와 PASS/FAIL로 정리돼 있다. 논문 기여 관점에서는 **증거의 강도**로 다시 묶어야 한다.

### Tier A — 대조가 있는 국소 결과 (가장 강함, 현재 2건)

| 실험 | 설계 | 결과 |
|---|---|---|
| **confirm40** | 동일 trace·동일 index 103, warm-up OFF/ON ABBA 4라운드 | 첫 conventional fallback GPU **22.756 / 2.277 / 2.286 / 35.303 ms**, 이후 중앙값 2.27–2.32 ms |
| **confirm46** | CUDA context 없는 CPU-sham 대조군, 10 seed 교대 순서 paired | CPU **0/10,000** 대 MPS **12/10,000** miss, 부호검정 단측 **p = 0.00390625**, miss가 index 7–9에 집중. 실제 worker 종료는 확인하지 않아 내부 원인 단계는 미분리 |

이 둘은 대조군을 둔 국소 비교다. Confirm40도 cold path의 내부 작업을 특정하지 않았고, Confirm46은 CPU와 GPU/MPS 초기화·종료 묶음을 대비했을 뿐 종료의 단일 단계를 분리하지 않았다. 나머지 실험은 관측 또는 조건부 비교다.

### Tier B — 반복·사전고정된 비교 (baseline이 약함)

| 실험 | 비교 | AI 이득 | 비고 |
|---|---|---|---|
| confirm36 | conventional / eager / external, 동일 node·trace 10,000회 | — | correct **3,688 / 9,835 / 9,840**, conventional 실행 10,000 대 **203** |
| confirm38 · 41 | external vs local vs eager, 정순·역순 | +2.93% / +3.06% | 비transactional 구현 |
| **confirm45** | 원자 예약 transaction vs eager, 독립 seed 2개·역순 | **+2.95% / +2.41%** | 전체 gate 통과 |
| **confirm47** | NRx 50 ms 보수 계약 | **+1.38% / +1.11%** | 전체 gate 통과. 외부 RAN p99 **55.8 ms** 대 eager **16.0 ms** |
| confirm43 · 44 | gap lease ON/OFF, 30 ms 계약 | +0.419% / +1.304% | **30 ms bound 위반으로 전체 gate 실패** |
| **confirm50** | gap lease ON/OFF, 50 ms 계약 | **+0.468% / +0.535%** | 전체 gate 통과. gap 752·746개 중 순증 595·679 (전환율 79% / 91%) |

### Tier C — 계약을 정의한 반증

| 실험 | 반증한 것 |
|---|---|
| confirm8 | HBM 포화 2,963/3,000 miss — 대역폭 경합은 cap으로 못 막음 |
| confirm9 | AI budget·crossing이 0인데도 RAN 1/1,500 miss — 회계상 안전 ≠ 실제 안전 |
| confirm26 | cap100 2/1,500, cap20 1/1,500 — **MPS cap은 격리 primitive가 아님** |
| confirm29 | admission 차단 + context 유지로도 cap20 1/5,000 — 논리적 차단만으로 부족 |
| confirm30 | 즉시 client 종료 시 17/10,000, worst 96–100 ms, **17건 전부 종료 이후** |
| confirm39 | 전역 lock·별도 epoch에서도 sham 4/5,000 (양 cap) |

### Tier D — 미해결 신호

| 신호 | 내용 |
|---|---|
| **index 2740–2762 서명** | confirm43은 2762(OFF 두 run 동일), confirm44는 2740/2754/2743. 서로 다른 노드·seed에서 비슷한 구간의 tail을 관측했지만 결정론적 원인은 미확인. confirm48은 첫 GC-ON 4,000회에 30 ms 위반이 없어 사전 GC overlap 가설에 실패하고 조기 중단 |
| **mandatory path 57.9 ms** | confirm36 conventional-only max 57.897 ms 대 worker 없는 confirm28의 max 7.571 ms |
| **fallback 실제 시작 시각 미계측** | `fallback_start_ns`는 예약 시각. 실제 시작 검증 불가 (완료 점검 문서가 스스로 명시) |

---

## 2. 핵심 진단: 무엇이 부족한가

### 2.1 구성 요소는 전부 선행 연구가 있다

선행연구 감사가 이미 확인했다. YinYangRAN(RAN+AI GPU 공유), CloudRIC(queue-aware 요청별 배치),
Concordia(예측·예약·유휴 회수), Nuberu(필수 경로 보존), ARCHES(채널 기반 AI/conventional 선택),
OCUDU(inline timing contract·late result), DARIS(MPS + bounded admission),
**SMEC(NSDI 2026, RAN MAC과 별도 MEC 서버의 application SLO 관리; edge GPU에서 MPS priority 사용)**. SMEC은 같은 물리 GPU의 PHY optional NRx/recovery를 다루지는 않지만, "5G + MPS + deadline-aware GPU"라는 넓은 주장은 이미 선행한다. 결합만으로는 기여가 성립하지 않는다.

### 2.2 현재 주장은 baseline이 여유 있는 영역에서만 측정됐다

Confirm45/47/50의 최신 외부 transaction 비교는 **1 cell · P90 · D80**이다. Confirm24의 2셀 결과는 별도 **P150 · D130** 계약이므로 이 부하 근거를 대체하지 않는다. 이 조건에서 conventional은 period의 약 2.8%이고,
eager-dual은 수신기를 둘 다 돌려도 감당된다. 즉 **eager가 아직 깨지지 않는 영역에서만 비교했다.**

- confirm47에서 external RAN p99는 **55.8 ms**, eager는 **16.0 ms**다. D80이라 통과했지만 D50이면 miss다.
- 이득은 +1.1 ~ +3.0%이고, 보수적 계약으로 갈수록 줄어든다.

따라서 심사에서 나올 질문에 지금 데이터로는 답할 수 없다.

> "RAN p99가 3.5배 나쁜데 AI는 1.4% 더 얻는다. 왜 eager를 쓰지 않는가?"

답이 없는 것이 아니라, **답이 나오는 조건에서 측정하지 않았다.** eager는 매 요청마다 두 수신기를
모두 실행하므로 셀 수가 늘거나 주기가 짧아지면 애초에 들어가지 않는다. SoftWall의 가치는 정확히
그 지점에서 나타난다.

### 2.3 안전 마진 ↔ 이득 ↔ 지연의 trade-off가 흩어져 있다

| NRx 상한 | AI 이득 | 외부 RAN p99 | bound gate |
|---|---|---|---|
| 30 ms (confirm43/44) | +0.42% / +1.30% | — | **위반 3건씩** |
| 30 ms대 (confirm45) | +2.95% / +2.41% | — | 통과 |
| 50 ms (confirm47) | +1.38% / +1.11% | 55.8 ms | 통과 |
| 50 ms (confirm50, gap만) | +0.47% / +0.54% | 55.7 ms | 통과 |

세 실험이 서로 다른 seed·조건이라 곡선으로 읽을 수 없다. **하나의 sweep으로 다시 측정하면
"안전 보장의 비용"이라는 결과 자체가 된다.**

### 2.4 그런데 가장 강한 자산이 아직 주장으로 쓰이지 않고 있다

confirm40의 첫 경로 지연과 confirm46·52의 client 수명주기 근처 tail은 **서로 다른 가설**로 분리해 다룬다.

```
confirm40  처음 쓰는 경로          → 22.8 / 35.3 ms   (정상 2.28 ms)
confirm46  stop 응답 근처 tail     → miss가 index 7–9에 집중, CPU 대조군 0
confirm52  idle/ack/exit 비교     → 9짝 0/6/2 miss, 사전 짝비교 gate 실패
confirm30  live teardown           → worst 96–100 ms, 전부 종료 이후
confirm39  전역 lock에서도 sham    → 4/5,000
```

현재 자료는 초기 RAN release에서 GPU 처리 시간이 비정상적으로 길어지는 사례를 보여주지만,
정상 상태 경합보다 상태 전이가 **주된 원인**인지는 입증하지 못했다. Confirm52에서는 종료 응답 조건의
index7 tail이 반복됐으나 실제 종료 확인 조건에서도 miss가 있었고, 장벽과 추가 대기 시간이 혼입됐다.
GPU context switch 비용 자체는 GCAPS 같은 선행 연구가 이미 다룬다. 여기서 남는 연구 후보는
실제 PHY deadline에 영향을 주는 구체적 전이의 귀속과, 그 전이를 요청별 recovery/AI admission에
반영했을 때 강한 결합 baseline을 넘어서는 이득이다.

---

## 3. 노벨리티 확보 작업

우선순위 순이며, 각 항목은 **실패했을 때의 분기**를 포함한다. seed 추가는 어느 항목에도 없다.

### N1. eager가 깨지는 부하까지 올린다 (최우선)

| | |
|---|---|
| **목적** | SoftWall의 가치가 실제로 나타나는 운영점을 찾는다 |
| **방법** | 셀 수(1→2→4→8)와 주기(90→45→25→12 ms)를 올리며 동일 trace로 eager / external transaction / conventional-only 비교 |
| **기록** | 각 부하에서 세 정책의 deadline miss, correct TB, AI 완료량, RAN p99 |
| **진단 통과** | eager가 miss하기 시작하는데 external transaction은 miss 0이고, 무선 정확도를 유지하며, 완료된 AI 작업이 양수인 부하 구간이 존재. 이 결과만으로 강한 baseline 대비 우위 또는 시스템 논문 성립을 주장하지 않음 |
| **실패 분기** | 사전 고정한 실행 가능 격자에서 위 구간이 없으면 현재 시스템 기여 가설을 중단하고 N3 상태 전이 측정 기여를 평가. seed 추가나 사후 계약 완화로 성공 구간을 만들지 않음 |

이 실험은 시스템 기여의 가능성을 가르는 진단이다. 현재 external controller는 한 셀·한 endpoint만 지원하므로 2/4/8셀 점은 구현 없이 실행할 수 없다. P<42 ms에서는 기존 AI host budget 40 ms + guard 2 ms가 release 사이에 들어가지 않는다. recovery-gap AI도 다음 release를 넘는지 실제 시각으로 검사해야 한다. AI 완료량이 0인 crossover는 사용자가 요구한 "deadline 보호와 남는 자원의 AI 처리"를 입증하지 못한다. eager와 conventional-only는 ablation·진단 대조군이고, §3 N5의 강한 결합 baseline 비교를 대체하지 못한다.

**Confirm51 1셀 진단 결과 (2026-09-21):** [사전 protocol](../../results/softwall_same_gpu/confirm51_n1_single_cell_diagnostic_protocol.json)의 첫 라운드 4주기×3정책 각 1,000회를 완료했다. eager는 P90/45/25/12에서 모두 miss 0, p99 6.618/7.142/8.065/7.189 ms로 D80에서 여유가 컸다. external도 miss 0이지만 correct TB가 979/964/947/916으로 eager 981/980/979/981보다 P45부터 떨어졌고, AI 완료량은 12,723/149/0/0이었다. external admission reject는 0/25/50/110. P45에서 이전 fallback을 기다린 다음 요청이 약 11.5 ms 늦어져 3 ms 수락 여유를 넘긴 사례가 직접 기록됐다. 모든 점에서 eager miss 0이므로 두 라운드 모두 교차점을 요구하는 판정은 이미 불가능해져 역순 라운드를 중단했다. [전체 결과](../../results/softwall_same_gpu/confirm51_n1_single_cell_diagnostic.md)는 **진단 FAIL, 증거 등급 C**다. 이는 전체 다중 셀 N1의 결과가 아니다.

**fallback 뒤 거절 연쇄의 관측:** [사후 요청별 감사](../../results/softwall_same_gpu/confirm51_serial_blocking_mechanism_audit.md)에서 P45의 fallback 25건은 각각 바로 다음 release의 NRx 거절 1건으로 이어졌다. P25에서는 fallback 25건마다 연속 거절 2건, P12에서는 fallback 22건마다 연속 거절 5건이었다. 같은 채널 seed의 거절 요청에서 eager NRx 정답은 P45/25/12 각각 23/25, 48/50, 105/110인데 external conventional 정답은 7/25, 15/50, 42/110이다. 이 연쇄에는 컨트롤러의 직렬 대기와 **현재 50 ms NRx 계약상 복구 구간을 침범하지 않고는 수락할 수 없는 제약**이 함께 있다. P45의 다음 release는 이전 fallback 직전 8 ms에 도착하고, 이전 fallback이 25 ms 상한을 다 쓰면 다음 요청의 NRx cutoff까지 20 ms만 남는다. 두 창 모두 50 ms보다 짧으므로 비직렬 접수만으로 거절을 안전하게 없앨 수 없다. 정책들이 다른 시간 창에 실행됐으므로 eager의 정답을 새 스킴의 반사실적 결과로 간주하지 않는다. P90의 AI RPC p99/max는 3.674/4.409 ms, P45는 3.959/6.822 ms였으나 WCET가 아니므로 기존 40 ms host 계약을 사후 5–8 ms로 줄이는 근거로 사용하지 않는다.

**새 후보 — 결과를 아는 즉시 복구 시점 선택:** Confirm51의 자연 fallback은 P90/45/25/12에서 각각 27/25/25/22건이고 모두 timeout이 아닌 조기 CRC 실패였다. 기존 제어기는 실패를 일찍 알아도 deadline 직전인 release+53 ms까지 기다렸다가 conventional을 시작했다. 실패 확인 시각에 `남은 시간 ≥ conventional 상한 25 ms + guard 2 ms`이고 다음 release 전에 끝낼 수 있으면 즉시 복구하고, 그렇지 않으면 원래 예약 시각을 유지하는 정책을 **별도 사전 고정 비교**로 시험했다. 이 방식은 이미 알려진 실패에만 적용된다. timeout·장기 NRx tail·동시 다중 셀의 문제는 해결하지 못하고, 고정 복구 시점 대비 개선만으로 선행 기법의 결합보다 새롭다고 주장할 수 없다.

**Confirm53 판정 (2026-09-21):** 위 후보를 [사전 고정된 ABBA 비교](../../results/softwall_same_gpu/confirm53_early_fallback_abba_protocol.json)로 실행했다. 같은 P45/D80·단일 GPU MPS epoch에서 두 새 seed의 조기/기존 무선 정답은 983/968과 983/972, NRx 거절은 0/19와 0/19, RAN p99는 조기 5.561/5.579 ms 대 기존 55.746/55.667 ms였다. 네 arm 모두 deadline miss와 측정된 계약 위반 0, 조기 정책의 AI 완료량은 343·221개로 양수였다. 다만 AI 완료량의 조기−기존 차이는 **+146, −163**으로 방향이 바뀌므로 처리량 우위는 입증되지 않았다. [전체 판정](../../results/softwall_same_gpu/confirm53_early_fallback_abba.md)은 B등급 메커니즘 ablation PASS다. 조기 복구를 강한 결합 baseline에도 제공해야 하며, 이 결과만으로 novelty를 선언하지 않는다.

**Confirm54 통합 확인:** Confirm53의 단일 요청용 조기 복구는 원래 늦은 calendar 예약을 그대로 남겼다. 다중 요청에서 이 방식은 다른 conventional 의무와 겹칠 수 있어, 예약을 원자적으로 앞당기고 충돌 시 원래 시각을 유지하는 runtime 경로를 추가했다. 동시 이동·중복 구간 단위 테스트 14개가 통과했다. [한 셀 GPU 재검증](../../results/softwall_same_gpu/confirm54_retimed_recovery_canary.md)은 같은 trace의 1,000회에서 무선 정답 983, miss 0, NRx 거절 0, 조기 복구 21, 예약 이동 거절·잔여 credit 0, AI 322로 사전 통합 gate를 통과했다. 이는 다중 셀의 물리적 GPU 간섭이나 미완료 IPC buffer 안전성을 아직 증명하지 않는다.

**Confirm55 두 endpoint bring-up:** [수정된 사전 protocol](../../results/softwall_same_gpu/confirm55_two_endpoint_bringup_protocol_retry1.json)의 단일 GPU·cap40+40 NRx·cap20 AI, 2셀 P150/D130·1,000 동시 release에서 [전체 gate](../../results/softwall_same_gpu/confirm55_two_endpoint_bringup.md)가 통과했다. 정답 1,958/2,000, miss 0, AI 29,910, NRx 50 ms 및 conventional GPU 25 ms bound 위반 0, 복구 48건·예약 이동 거절·잔여 credit 0이었다. 자연 동시 두 셀 fallback은 단 **1회**여서 상관된 복구 폭주를 검증한 것은 아니다. 두 worker를 함께 dispatch했지만 kernel overlap의 실제 시작·끝 계측은 없다. 이 실행은 안전한 pinned transport의 C등급 출발점이며 queue-aware 선택 또는 강한 결합 baseline 대비 우위가 아니다.

다음 N1 구현은 여러 release/셀을 직렬 fallback 대기로 막지 않는 스케줄링과 함께, **다음 NRx와 이전 mandatory fallback이 같은 GPU에서 겹칠 때의 물리적 안전 계약**을 먼저 갖춰야 한다. 가능한 해법은 검증된 짧은 NRx 서비스 상한, cutoff 전에 실제로 중단되는 분할 실행, 또는 동시 실행 중 mandatory 서비스 상한을 계측·자격 검증한 별도 lane이다. 40 ms AI 단위가 들어갈 수 없는 짧은 주기에서는 더 짧은 작업 단위를 새 protocol로 사전 지정하고 모든 정책에 동일하게 제공해야 한다. 이 경로가 없으면 셀 수만 늘려도 무선 효용을 잃는 현재 구조의 실패를 반복할 가능성이 높다.

**N1 격자 선행 가능성 검사:** 같은 시각에 C개 셀이 release되고 conventional 복구 경로가 하나이며 모든 NRx가 실패할 수 있다면, 복구만의 필요조건은 `C × B_conv + guard ≤ D`다. 현재 `D=80`, `B_conv=25`, `guard=2` ms에서 C=1/2/4/8의 최소 deadline은 각각 **27/52/102/202 ms**다. 따라서 동일 release의 4/8셀을 단일 복구 lane에서 모두 보장하는 것은 현재 D80으로 불가능하다. 2셀은 복구 구간을 28–53 ms와 53–78 ms로 배치할 수 있지만, 첫 셀의 NRx 결과는 50 ms 상한까지 기다릴 수 없고 28 ms에 포기해야 한다. 모든 NRx에 50 ms의 결과 기회를 남기려면 더 강한 조건 `B_NRx + C × B_conv + guard ≤ D`가 필요하며 C=1/2/4/8에 **77/102/152/252 ms**다. 두 조건 모두 host 오버헤드를 뺀 낙관적 하한이다. 주기 쪽도 all-fail 지속 부하에서 단일 복구 경로의 점유율 `C × 25 / P`가 1을 넘으면 보장은 불가능하므로 1셀 P12가 해당한다. P25는 점유율이 정확히 1이라 AI와 NRx를 위한 최악 경우 여유가 없다. 이 계약 구분 없이 원래 1/2/4/8 × P90/45/25/12 격자를 그대로 GPU에서 돌리지 않는다.

다중 셀 N1을 다시 설계하려면 먼저 **NRx를 일찍 포기하는 복구 cutoff와 그 무선 품질 비용**, **물리적으로 검증한 추가 conventional 동시 처리 용량**, **실측 tail을 만족하는 더 짧은 NRx 상한**, **실제 셀별 release 위상과 별도 deadline**, 또는 **명시적인 실패 burst 제한** 중 무엇이 계약인지 정해야 한다. 단순히 calendar `capacity` 숫자만 올려 같은 GPU가 25 ms 복구 여러 개를 병렬 처리한다고 가정해서는 안 된다. deadline을 늘리는 선택은 지연 비용을 같은 표에 적는다. 이 실행 가능한 계약을 고정한 뒤에만 다중 release/셀 제어기를 구현·평가한다.

**다음 스킴의 최소 상태 기계:** (1) release 이벤트는 요청별 입력·expiry·fallback calendar를 확보하고 즉시 반환한다. (2) NRx 완료 이벤트가 cutoff 전에 오면 CRC/LLR 유효성을 확인해 한 번만 commit하고 예약 복구를 해제한다. (3) cutoff에 결과가 없거나 틀리면 실제 계측한 시각에 conventional을 시작하고, 늦은 NRx 결과는 폐기하되 GPU가 물리적으로 끝날 때까지 buffer/credit을 재사용하지 않는다. (4) 다음 NRx와 기존 fallback이 겹칠 수 있으면 **둘의 동시 서비스 bound가 검증된 경우에만** NRx를 수락한다. 그렇지 않으면 NRx를 거절하거나, 검증된 중단점에서 실제로 멈추고 복구한다. (5) AI는 다음 release와 이미 받은 모든 fallback 의무 중 가장 이른 경계 앞에 검증된 host+GPU 작업 상한과 guard가 모두 들어갈 때만 허가한다. MPS client는 timed epoch 동안 준비된 상태로 유지한다. 단일 요청의 `wait_until(fallback_start)`를 제어 루프에서 제거해야 하지만, 이것만으로 안전하거나 새 공동 제어 알고리즘인 것은 아니다.

**실제 구현 의무:** 현 CUDA IPC는 forward/backward 배열이 한 쌍이고 doorbell sequence도 한 개라, 이전 NRx가 물리적으로 완료되기 전에 다음 요청을 쓰면 입력·출력이 섞일 수 있다. 현 `PairedDualReceiver`는 매 요청마다 noisy slot 참조를 교체하므로 이전 fallback용 입력 참조를 완료까지 보존해야 한다. 모든 요청 상태를 expiry까지 보유하는 보수적 설계의 최대 동시 요청 수는 1셀에서 `ceil(D/P)`로 P90/45/25/12에 각각 **1/2/4/7개**다. 실제 IPC 슬롯 수는 물리적 NRx 완료와 buffer 재사용 규칙에 따라 정하며, bound 위반 시 추가 drain/quarantine이 필요하다. 컨트롤러에서 `wait_until` 한 줄만 제거하는 수정은 유효한 비직렬 스킴이 아니다.

**다중 요청 go/no-go:** P45/D80에서 *늦은* fallback을 기다리면 이전 복구 전 8 ms·후 20 ms 창만으로 다음 요청의 50 ms NRx 계약을 보장할 수 없다. Confirm53은 알려진 CRC 실패를 일찍 복구하여 그 특정 연쇄를 제거했지만, timeout·장기 실행이나 다중 셀에서는 여전히 충돌할 수 있다. 다음 GPU 비교 전에 **물리적으로 완료되지 않은 NRx와 mandatory conventional의 공동 서비스 상한**, **요청별 IPC buffer 수명**, **동시 실패 시 복구 예약 이동**을 입증해야 한다. 중단 가능한 NRx 분할이나 짧은 서비스 상한이 필요하다면 별도 자격 계약으로 검증한다. 모든 기계적 안전장치와 AI 작업 단위는 강한 결합 baseline에도 동일하게 제공한다. 이 조건을 설계할 수 없으면 큰 GPU campaign을 시작하지 않는다.

**현 계약의 의사결정:** Confirm53/54로 알려진 실패의 1셀 P45 경로는 회복됐지만, 원래 전체 N1 격자를 그대로 실행해 방어 가능한 우월성 판정을 얻을 수는 없다. 현재 50/25/2 ms·D80에서 4/8셀 동시 복구는 단일 conventional lane으로 불가능하고, 2셀은 조기 cutoff·물리 미완료 NRx 처리 비용이 생긴다. 다음 시스템 경로는 가능한 release·실패·deadline 계약을 먼저 고정하고 그 비용을 강한 결합 baseline에 동일하게 적용한다. 이 경로가 성립하지 않으면 시스템 우월성 대신 N3의 별도 측정 기여 여부를 판단한다.

**N3 Confirm52 판정:** [사전 protocol](../../results/softwall_same_gpu/confirm52_mps_idle_vs_retired_protocol.json)은 같은 GPU/PHY 조건에서 MPS client를 유휴 상태로 유지, 종료 응답만 확인, 실제 실행 래퍼 종료까지 확인한 세 조건을 10개 seed의 순서 교차 triplet으로 비교하도록 했다. [9개 triplet 감사](../../results/softwall_same_gpu/confirm52_mps_lifecycle_boundary_early_stop.md)에서 27,000회 모두 무선 정답, miss는 각 조건 0/6/2, 통제 오류 0이었다. 짝 방향은 5/1/3, 정확검정 p=0.109375이고 남은 한 짝이 가장 유리해도 p=0.0625라 중단했다. 이 결과는 종료 응답 직후 초기 RAN release의 tail 위험을 보여주지만 **사전 primary gate에 실패**했다. 종료 확인 조건은 약 2초 추가 quiet time도 포함하므로 장벽 고유 효과와 대기 시간 효과를 분리하지 못한다. 이 수명주기 대비를 독립 기여로 사용하지 않는다. 추가 N3 GPU 실험을 반복 seed로 연장하지 않고, 시간 맞춤 대조와 실제 AI 작업을 포함한 새 메커니즘이 논문 중심 질문을 바꿀 수 있는지 먼저 검토한다.

### N2. admission bound sweep으로 trade-off 곡선을 만든다

| | |
|---|---|
| **목적** | §2.3의 흩어진 점들을 하나의 결과로 만든다 |
| **방법** | 동일 node·MPS epoch·trace에서 NRx bound ∈ {20, 30, 40, 50, 60} ms |
| **기록** | bound별 (gate 통과 여부, AI 이득, RAN p99, admission reject 수, 자연 recovery 수, bound 위반 수) |
| **산출** | "안전 마진의 비용" 단일 그림. 논문 §Evaluation의 중심 그림 후보 |
| **주의** | N1의 운영점에서 수행한다. P90에서 재면 또 낮은 값만 나온다 |

### N3. 상태 전이 비용을 1급 기여로 승격한다

| | |
|---|---|
| **목적** | confirm40·46·30·39를 하나의 측정 기여로 묶는다 |
| **추가 실험 (a)** | cold path의 정체 분리 — 첫 fallback에 nsys를 붙여 TRT engine page-in / CUDA module load / allocator 증가 중 무엇인지 귀속 |
| **추가 실험 (b)** | teardown의 어느 단계인지 분리 — MPS client 등록 해제 / CUDA context 파괴 / process 종료를 각각 단독 수행. confirm46이 "단일 내부 단계 인과는 특정하지 않음"이라고 남긴 공백 |
| **검증 후 가능한 주장** | 특정 GPU client 전이가 PHY deadline tail을 만드는 단계·빈도·조건을 인과적으로 분리하고, 이를 회피하는 admission/maintenance 정책이 동등한 대기·자원 비용의 baseline보다 낫다 |
| **설계 후보** | 경로 사전 warm-up, live window 중 teardown 제한, 논리 quarantine과 물리 teardown 분리. 각각의 비용과 효과를 시간 맞춤 대조로 검증 |

N1이 실패해도 평가할 수 있는 별도 방향이지만, Confirm52 결과만으로는 단독 논문 기여가 성립하지 않는다.

### N4. confirm48 GC 가설 판정 — 실패, 이 경로 중단

index 2740–2762 부근의 tail은 서로 다른 노드·seed에서 관측됐지만 결정론적 원인은 미확인이다.
GC 가설이 맞아도 bound 위반이 사라지는지는 별도 검증해야 하고, 틀리면 원인을 모른 채 남긴다.
[사전 protocol](../../results/softwall_same_gpu/confirm48_gc_tail_abba_protocol.json)의 첫 GC-ON 조건 4,000회를 job `58692107`/node `nid001128`에서 완료했다. Collection 87건이 있었으나 NRx 30 ms 위반은 0건이고, ≥1 ms collection도 없었다. RAN miss·안전 gate는 0이었지만, 첫 조건에서 필수 overlap 가설이 실패해 나머지 세 조건을 조기 중단했다. 후처리 파일명 오류는 원자료를 그대로 읽어 수정했다. [부분 판정](../../results/softwall_same_gpu/confirm48_gc_tail_abba.md)은 C등급 진단이다. GC가 모든 tail의 원인이 아니라는 반증도, 8 ms 상한의 입증도 아니다. 이 가설로 짧은 NRx 계약을 만드는 경로는 추가 seed 없이 중단한다.

### N5. 강한 결합 baseline을 만든다

| 구성 요소 | 대응 선행 연구 |
|---|---|
| 채널 기반 NRx 선택 gate | ARCHES |
| queue·실행시간 인지 배치 | CloudRIC |
| 고정 recovery 여유 예약 | Concordia |
| bounded work-unit background | DARIS / MobiUK |

같은 GPU 예산·같은 관측 정보·같은 trace를 주고 **각 제어기를 충분히 튜닝한다.**
SoftWall은 같은 deadline·radio utility 요구에서 이 결합보다 나아야 한다.

**N1의 운영점에서 수행한다.** 현재 조건에서 돌리면 셋 다 통과해서 차이가 안 나온다.
이 항목이 가장 크고, N1·N2 다음이다.

### N6. deadline 정의를 문서화한다 (실험 아님)

- release / NRx 분기 가능 시각 / conventional latest start / expiry를 대상 DU의 IQ-ready, FAPI/MAC CRC 소비 시각과 scheduler/HARQ 설정에서 유도. 3GPP `K2`는 DCI→UE PUSCH 전송 오프셋, `N2`는 UE 준비 시간이므로 이 둘만으로 gNB 복호 완료 기한을 유도하지 않음. [타이밍 계약 감사](SOFTWALL_PUSCH_TIMING_CONTRACT_KO.md)
- 현재 P90/D80이 무엇에 해당하는지, production slot 타이밍까지 가려면 무엇이 바뀌어야 하는지 명시
- 투고 전 필수. 지금 상태로는 "왜 80 ms인가"에 답할 근거가 없다

---

## 4. 실행 순서와 분기

1. **현재 판정:** Confirm53은 1셀 P45의 알려진 CRC 실패에서 조기 복구가 고정 최신 복구의 무선 손실을 두 역순 seed 모두 줄인다는 것을 보였다. AI 완료량은 양쪽 모두 양수이나 조기 정책의 처리량 우위는 재현되지 않았다. Confirm52의 lifecycle primary gate와 Confirm48의 GC 가설도 실패했다. 탑컨퍼런스 노벨리티는 아직 없다.
2. **다음 시스템 gate:** 조기 실패 복구를 양 정책에 동일하게 제공하고, [강한 결합 baseline](SOFTWALL_STRONG_BASELINE_SPEC_KO.md)의 채널 gate·실제 2개 이상 endpoint의 queue 선택·고정 복구 예약·bounded AI를 구현한다. 같은 GPU 예산에서 여러 release/셀의 동시 실패가 가능한 범위와 물리 미완료 NRx의 처리 규칙을 먼저 계산·계측한다. 기존 단일 셀 조기 복구 결과를 이 비교의 승리로 계산하지 않는다.
3. **유한 중단 기준:** 위 범위가 현재 50/25/2 ms·D80 계약에서 물리적으로 불가능하거나, 사전 고정 paired 비교에서 같은 deadline·무선 효용을 지키면서 강한 baseline보다 유효 AI 완료량을 늘리지 못하면 공동 제어 우월성 주장을 접는다. 추가 seed나 느슨한 계약으로 이를 대체하지 않는다.
4. **논문 주장 gate:** 실제 PHY expiry와 허용한 AI 작업 집합에서 RAN 정답·miss·AI 완료량을 같은 offered load로 비교한다. 이 단계가 없으면 조건부 시스템 구성 결과로만 보고한다. N3 상태 전이 경로를 재개한다면 별도의 time-matched 인과 대조를 먼저 고정한다.

---

## 5. 주장 문장 교정

| 현재 표현 | 교정 |
|---|---|
| "eager보다 AI 처리량이 +2.95% 높다" | "eager가 감당되는 저부하에서 무선 효용 동등, AI 소폭 우위. 고부하 비교는 N1 이후" |
| "RAN deadline을 보호한다" | "선언한 P90/D80 계약과 허용된 bounded workload에서 관측 miss 0" |
| "MPS 위에서 격리를 달성" | "MPS는 격리 primitive가 아니며(confirm26), 보호는 admission·예약·수명주기 계약에서 나온다" |
| "fallback을 예약 시각에 시작한다" | 실제 시작 시각 미계측. 계측 전까지 사용 금지 |

---

## 6. 하지 않을 것

- seed 추가로 기존 결론 보강
- 현재 P90/D80에서의 추가 정책 변형
- 새 메커니즘 추가로 이득 수치 올리기
- 다중 GPU·노드 간 확장 (N1–N5 완료 전)

현재 부족한 것은 데이터의 양이 아니라 **비교 조건**이다.
