# SoftWall 연구 진행 가이드라인

**작성:** 2026-09-21 KST
**적용:** confirm51 이후의 모든 라운드
**역할:** *무엇을* 할지는 [노벨리티 실행안](../archive/SOFTWALL_NOVELTY_ACTION_PLAN_KO.md)이 정하고, 이 문서는 *어떻게* 할지를 정한다.
**근거:** confirm2–50 라운드의 실제 진행 패턴 분석

---

## 0. 지난 라운드에서 배운 세 가지

36시간에 49개 실험을 실행했고 protocol 규율은 훌륭했다. 그런데 결과의 활용도는 실험 수에
비례하지 않았다. 원인은 셋이다.

1. **운영점이 한 곳에 고정됐다.** 최신 외부 transaction 비교는 1 cell · P90 · D80이었고 (Confirm24의 2셀 비교는 P150 · D130), 그 조건에서는
   비교 상대(eager-dual)가 여유로웠다. 그래서 이득이 구조적으로 1–3%에 갇혔다.
2. **실패 뒤에 조건이 아니라 계약이 느슨해졌다.** 30 ms bound가 깨지자 50 ms로 올렸다.
   gate는 통과했지만 AI 이득이 절반이 되고 RAN p99가 3.5배가 됐다. 이 대가가 어디에도
   결과로 기록되지 않았다.
3. **대조군이 있는 실험이 2건뿐이었다.** confirm40과 confirm46만 인과를 분리했고,
   나머지 47건은 관측 또는 조건부 비교다. 그런데 논문에서 가장 강한 건 그 2건이다.

아래 규칙은 이 셋을 되풀이하지 않기 위한 것이다.

---

## 1. 실험 선정 규칙

### R1. 결정을 바꾸지 못하는 실험은 돌리지 않는다

frozen protocol에 다음 항목을 **추가**한다.

```json
"decision_at_stake": "이 결과가 PASS면 X를 하고 FAIL이면 Y를 한다",
"if_both_outcomes_lead_to_same_action": "그러면 이 실험은 취소한다"
```

지난 라운드의 confirm44는 confirm43과 seed·순서만 달랐고 같은 이유로 같이 실패했다.
두 결과가 같은 행동으로 이어졌다면 한 번만 돌렸어야 한다.

### R2. 한 라운드는 한 질문만 다룬다

confirm2–50은 feasibility, recovery 의미론, 상보성, 수명주기, gap lease, 보수 계약까지
여섯 질문이 섞여 있었다. 다음 라운드는 **N1(부하 스케일링) 하나**다.
라운드 중 새 질문이 생기면 실행하지 않고 다음 라운드 후보로 기록한다.

### R3. 실패 뒤에는 조건을 바꾸고, seed는 바꾸지 않는다

이미 지키고 있는 규칙이며 명문화한다. 단 한 가지를 추가한다.

> **계약을 느슨하게 해서 gate를 통과시킨 경우, 느슨해진 양과 그 대가를 같은 표에 기록한다.**

30 ms → 50 ms 변경은 "통과"가 아니라 "+2.9% → +1.4%, p99 16 ms → 55.8 ms를 지불하고 통과"다.

---

## 2. 측정 조건 규칙

### R4. 운영점은 비교 상대가 불편한 곳에서 고른다

우위를 주장하는 모든 비교는 실행 전에 다음을 기록한다.

| 기록 항목 | 예시 |
|---|---|
| 이 부하에서 baseline의 여유 | "eager의 GPU 점유율 X%, deadline 여유 Y ms" |
| baseline이 깨지는 지점과의 거리 | "eager는 N cell에서 miss 시작, 현재 측정점은 1 cell" |

baseline이 여유롭다면 그 실험은 **우위 주장이 아니라 feasibility 확인**으로 분류한다.
이 구분을 결과 md의 제목에 쓴다.

### R5. 비교 상대는 실행 전에 이름을 붙인다

protocol에 `strongest_realistic_opponent` 항목을 둔다. 후보가 아래 중 하나면 그대로 쓰지 않는다.

- eager-dual (우리가 만든 낭비적 정책)
- conventional-only (NRx를 아예 안 씀)
- 우리 구현끼리의 비교 (local vs external)

이들은 **ablation이지 baseline이 아니다.** 강한 상대는
[결합 baseline 명세](SOFTWALL_STRONG_BASELINE_SPEC_KO.md)를 따른다.

### R6. 계약으로 거는 값은 반드시 계측되는 값이어야 한다

`fallback_start_ns`가 예약 시각이라 실제 시작을 검증할 수 없었던 사례를 반복하지 않는다.
protocol의 모든 gate 항목에 대해 **그 값을 어느 파일의 어느 필드에서 읽는지** 명시한다.
읽을 수 없으면 gate에서 뺀다.

---

## 3. 증거 등급 규칙

### R7. 모든 결과에 등급을 붙인다

| 등급 | 조건 | 논문에서의 위치 |
|---|---|---|
| **A** | 대조군이 있고 교란 요인이 분리됨 (confirm40의 ABBA, confirm46의 CPU sham) | 주 주장 |
| **B** | 사전 고정·반복·역순이지만 비교 상대가 약함 | 보조 근거 |
| **C** | 단일 관측, 진단, smoke | 부록·동기 |

### R8. 새 실험은 대조군을 붙일 수 있는지 먼저 묻는다

Tier A가 2건뿐인 것은 설계 단계에서 대조군을 고려하지 않았기 때문이다.
confirm46이 CPU sham을 붙이자 바로 p=0.0039가 나왔다. 대조군의 비용은 보통 실행 시간 2배이고,
증거 가치는 그보다 훨씬 크다.

---

## 4. 주장 규칙

### R9. 결과 문장에 조건을 붙여 쓴다

| 금지 | 사용 |
|---|---|
| "RAN deadline을 보호한다" | "P90/D80 계약과 허용된 bounded workload에서 관측 miss 0" |
| "MPS에서 격리를 달성했다" | "MPS는 격리 primitive가 아니며, 보호는 admission·예약·수명주기 계약에서 나온다" |
| "eager보다 우수하다" | "eager가 감당되는 저부하에서 무선 효용 동등, AI 소폭 우위" |

### R10. 실패한 protocol은 지우지 않고, 성공이 그것을 덮지 않는다

이미 지키고 있다. 지난 라운드의 confirm19·21·23·37이 좋은 예다.

---

## 5. 문서 규칙

### R11. 목적당 문서 하나

현재 SoftWall 문서가 9개이고 내용이 크게 겹친다. 아래로 정리한다. **사용자 승인 후 적용한다.**

| 목적 | 유지할 문서 |
|---|---|
| 결과 원장 | `results/softwall_same_gpu/EXPERIMENT_GATE_LEDGER_KO.md` |
| 결과 해설 | `INITIAL_FEASIBILITY_REPORT_KO.md`, `S2_NATURAL_CHANNEL_REPORT_KO.md`, `SOFTWALL_COMPONENT_ABLATION_KO.md` |
| 다음 작업 | `SOFTWALL_NOVELTY_ACTION_PLAN_KO.md` |
| 선행 연구 | `SOFTWALL_RELATED_WORK_AUDIT_KO.md` |
| 비교 기준 | `SOFTWALL_STRONG_BASELINE_SPEC_KO.md` |
| 작업 규칙 | 이 문서 |

통합 또는 아카이브 대상:

| 문서 | 처리 |
|---|---|
| `SOFTWALL_NOVELTY_THESIS_KO.md` | 실행안과 선행연구 감사로 내용이 분산됨 → 기여 문장 정의만 남기고 축약 |
| `SOFTWALL_COMPLETION_AUDIT_KO.md` | P0–P7 대조는 원 계획서가 폐기되면 의미가 바뀜 → 실행안에 흡수 |
| `SOFTWALL_MPS_ISOLATION_FEASIBILITY_KO.md` | 실행 우선순위는 실행안이 대체 → 결과 보고서로 흡수 |
| `RESEARCH_PLAN_SOFTWALL_KO.md` | 4-GPU 전용 배치 전제와 정정되지 않은 수치 포함 → `docs/archive/`로 이동 |
| `RESEARCH_PLAN_SOFTWALL_REVIEW_KO.md` | 지적 사항이 모두 반영됨 → `docs/archive/`로 이동 |
| `SOFTWALL_SCHEME_AND_NEXT_STEPS_KO.md` | 다중 GPU 확장안 → `docs/experiments/`로 이동, 현재 판단에서 제외 |

### R12. 폐기된 전제는 문서에 남기지 않는다

`RESEARCH_PLAN_SOFTWALL_KO.md`에는 이미 버린 전제(GPU0 전용화, MIG 비교, CloudLab)와
정정되지 않은 수치(full A100 NRx 1164.1 req/s·1.34 ms — 실제는 1130.5 req/s·0.882 ms,
1.34 ms는 4g MIG 값)가 남아 있다. 아카이브로 옮길 때 **문서 첫 줄에 폐기 사유를 적는다.**

---

## 6. 라운드 운영

```
사전   질문 1개 확정 → 운영점 결정(R4) → 강한 상대 지명(R5)
       → protocol 고정(R1·R6) → 대조군 설계(R8)
실행   전역 GPU lock, 단일 MPS epoch, 동일 node·trace
사후   등급 부여(R7) → ledger 기록 → 실패는 보존(R10)
       → 다음 질문 후보만 기록하고 실행하지 않음(R2)
```

**라운드 종료 조건:** 그 질문의 결정이 내려졌을 때. 통과할 때까지가 아니다.

---

## 7. 다음 라운드(N1) 착수 체크리스트

- [ ] 질문 한 문장: "eager-dual이 deadline을 놓치기 시작하는 부하는 어디이고, 그 지점에서 transaction은 놓치지 않는가?"
- [ ] 운영점 sweep 정의: 셀 수 1/2/4/8 × 주기 90/45/25/12 ms 중 실행 가능한 격자
- [ ] 격자 실행 전 단일 conventional lane의 all-fail 복구 필요조건 `cells × B_conv + guard ≤ D`와 지속 부하 조건 `cells × B_conv ≤ period`를 검사. 모든 NRx 결과에 full bound만큼 기다릴 때는 추가로 `B_NRx + cells × B_conv + guard ≤ D`를 검사한다. 조기 복구 cutoff는 무선 품질 비용을 사전 기록하고, 불가능한 점은 "현재 계약에서 보장 불가능"으로 분류
- [ ] 각 점에서 기록할 값: 세 정책의 miss 수·correct TB·AI 완료량·RAN p99, 그리고 **eager의 여유**
- [ ] `decision_at_stake`: deadline 교차점에서 무선 정확도와 양수의 유효 AI 처리량도 유지하면 N2·N5 검증으로, 그렇지 않으면 현재 시스템 우월성 가설을 중단하고 N3 측정 기여를 검증
- [ ] 현재 external controller는 1셀만 지원하므로 2/4/8셀 점은 다중 셀 구현 후에만 실행. eager와 conventional-only 비교는 진단·ablation으로 분류하고 N5 강한 결합 baseline 우위로 해석하지 않음
- [ ] 대조군: 같은 부하의 conventional-only (RAN 단독 상한)
- [ ] 계측 확인: 모든 gate 값의 출처 필드 명시 (R6)
- [ ] seed 추가 금지 명시

---

## 8. 이 가이드라인이 막으려는 실패

| 실패 양상 | 막는 규칙 |
|---|---|
| 편한 조건에서만 이겨서 이득이 작게 나옴 | R4 |
| 약한 상대를 이기고 우위를 주장함 | R5 |
| 계약을 느슨하게 해서 통과시키고 대가를 안 적음 | R3 |
| 같은 결론으로 가는 실험을 반복함 | R1·R2 |
| 관측을 인과로 읽음 | R7·R8 |
| 측정 못 하는 것을 gate로 검 | R6 |
| 문서가 늘어나 어느 것이 기준인지 모름 | R11·R12 |
