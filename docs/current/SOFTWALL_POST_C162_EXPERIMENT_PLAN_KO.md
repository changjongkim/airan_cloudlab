# SoftWall C162 이후 사전 실험 계획: production timing, lifecycle, cross-family

**상태:** 2026-09-25 C163 v2 ingestion path와 단위시험 24개 완료, 실제 production trace 미확보  
**현재 판정:** `UQ_NO_PRODUCTION_TRACE`  
**고정된 기존 근거:** warm A100 `P180/D155` actual-NRx/shared-recovery mode, C161 fault
qualification, C162 predictive QSU/QSN/MI/UQ envelope와 64-debt certified scheduler  
**목적:** synthetic deadline과 single-family 한계를 줄이고, 어떤 추가 결과가 어떤 논문 문장을
허용하는지 실행 전에 고정한다.

---

## 1. C163에서 답할 질문

| ID | 질문 | 필요한 결과 | 허용되는 주장 |
|---|---|---|---|
| RQ-P1 | 대상 DU의 실제 request별 `d_MAC`에서 mandatory all-fail schedule이 가능한가 | explicit expiry·clock calibration·mandatory-only와 correlated all-fail miss0 | 해당 production topology의 radio safety 유한 표본 |
| RQ-P2 | C162 envelope가 실제 절대 시각과 이질 deadline에서도 맞는가 | QSU/QSN/MI 예측과 물리 판정 mismatch0, false-safe0 | production-envelope 예측 정확성 |
| RQ-P3 | Conditional success가 실제 AI lease를 여는가 | actual NRx outcome, exact unresolved set, Qwen completion과 fence | conditional-credit mechanism |
| RQ-P4 | Static 또는 단순 safe work-conserving 대비 실용적 AI 여지가 있는가 | 사전 5% opportunity gate 뒤 paired holdout | 통과할 때만 성능 우위 |
| RQ-P5 | 계측·제어·certificate 비용이 deadline budget 안에 들어가는가 | tracing sham과 decision latency, whole-path bound | 측정된 mode의 overhead |

RQ-P1--P3가 논문의 substrate/envelope 주장을 강화한다. RQ-P4가 실패해도 안전 기여는
유지하며, 기존 C159-Q3의 no-headroom 결과를 숨기거나 다른 metric으로 바꾸지 않는다.

## 2. 실행 전에 봉인할 artifact

다음 네 입력과 protocol/source hash가 없으면 production run을 시작하지 않는다.

| 입력 | schema/template | 역할 |
|---|---|---|
| Raw DU event trace | [raw template](../../results/softwall_multigpu/c163_raw_du_trace_template_v1.json) | IQ ready, release, submit, CRC, FAPI publish, MAC consume, NRx outcome, single commit, SoftWall decision |
| Clock calibration | [clock template](../../results/softwall_multigpu/c163_clock_calibration_template_v1.json) | timestamp producer 간 최대 변환 오차 |
| Expiry contract | [expiry template](../../results/softwall_multigpu/c163_expiry_contract_template_v1.json) | request별 절대 `d_MAC`과 DU/MAC/FAPI 출처 |
| Mode qualification | [mode template](../../results/softwall_multigpu/c163_mode_qualification_template_v1.json) | topology/lifecycle별 NRx·recovery·control·AI class bound |

[Builder](../../scripts_for_node/softwall_same_gpu/c163_build_du_contract_v2.py)가 원시 event를
[contract v2](../../results/softwall_multigpu/softwall_du_timing_contract_template_v2.json)로
조립하고, [validator](../../scripts_for_node/softwall_same_gpu/du_timing_contract_v2.py)가 모든
artifact hash와 물리 결과를 검사한다. [Production bridge](../../scripts_for_node/softwall_same_gpu/c163_production_envelope_bridge.py)는
각 TB를 절대 release/deadline을 가진 recovery obligation으로 변환해 exact all-fail schedule과
AI blackout을 다시 계산한다.

Expiry는 UE-side K2/N2에서 만들지 않는다. Target MAC이 CRC를 마지막으로 유효하게 소비하는
시각 또는 명시적 DU/FAPI 계약이어야 한다. Timestamp가 여러 host/device clock에 있으면
PTP/TAI calibration의 최대 오차를 expiry에서 차감한다.

## 3. C163 단계별 실행

### C163-A — instrumentation와 clock sham

같은 mandatory-only radio trace를 tracing OFF/ON으로 각 node에서 `ABBA` 네 block 실행한다.
Block당 최소 2,500 TB, 두 독립 A100 node를 목표로 한다. 같은-clock이면
`CLOCK_MONOTONIC_RAW`, 여러 host이면 PTP PHC 또는 calibrated TAI를 사용한다.

통과 조건:

- raw event 누락·중복·identity 변화 0
- timestamp 순서 역전과 duplicate commit 0
- clock calibration artifact와 최대 오차 존재
- tracing ON이 새 deadline miss를 만들지 않음
- tracing overhead의 block-paired median, p99, max를 모두 보고

Tracing 자체가 deadline을 깨면 계측 방식을 수정하고 새 development ID를 사용한다. 해당
실패 trace는 보존한다.

### C163-B — production-topology service qualification

Target DU topology와 동일한 GPU placement, MPS cap, cuPHY/NRx engine, Qwen worker, process
lifecycle에서 다음 whole-path class를 따로 측정한다.

| Class | 최소 개발 표본/node | holdout에서 검사할 bound |
|---|---:|---|
| NeuralRx admitted path | 1,000 | release→outcome observe |
| Conventional recovery | 1,000 | dispatch→home-visible completion |
| Broker/controller | 5,000 operations | decision→physical launch eligibility |
| Qwen 16/32/64/128/256/512 | 각 500 | decision 이후 fence-confirmed completion |

Development node에서 bound 후보와 guard를 정하고 source·mode JSON을 봉인한다. 다른 node의
holdout에서는 bound를 늘리지 않는다. 한 건이라도 넘으면 그 node/lifecycle은 UQ이며, 더 큰
bound로 바꾸려면 새 mode ID와 새 holdout이 필요하다. 이는 finite-sample qualification이지
WCET 증명이 아니다.

### C163-C — mandatory-only와 all-fail production safety

먼저 optional NRx와 AI 없이 target offered load를 node당 최소 10,000 TB 실행한다. 그 뒤
optional NRx를 허가하되 사전 고정한 batch에 correlated fail/late를 주입해 실제 shared cuPHY
recovery를 실행한다. Development과 holdout은 다른 node·seed를 사용한다.

Hard gate:

- `mac_consume + clock_error <= d_MAC` 위반 0
- request당 radio commit 정확히 1
- input/output identity와 conventional correctness 위반 0
- initial exact all-fail certificate 없는 request의 optional launch 0
- physical recovery 순서가 반환 certificate와 일치
- physical completion 전 credit 반환 0

Mandatory-only가 불가능하면 AI/NRx 정책을 바꾸지 않고 해당 offered load를 MI로 판정한다.

### C163-D — prospective production-envelope boundary

Development trace의 실제 expiry 분포에서 다음 case를 **결과를 열기 전에** 선택한다.

1. QSU 내부점: AI와 recovery가 충분한 margin으로 모두 가능
2. QSU/QSN 경계의 양쪽 인접 decision time 또는 AI class
3. QSN: mandatory는 안전하지만 자격화된 AI unit은 불가능
4. MI 직전 최대 debt와 첫 MI debt
5. UQ: cold/restart 또는 bound provenance가 없는 mode

각 case는 development node 50회, frozen holdout node 100회를 목표로 하고 case 순서를
반대로 한다. 실제 NRx kernel, 실제 shared cuPHY recovery와 실제 Qwen을 사용한다. Channel
randomness와 debt count를 분리해야 할 때에는 모든 NRx kernel을 실행한 뒤 사전 고정 suffix의
outcome 공개만 fail로 처리하며, 이를 controlled-outcome이라고 명시한다.

Primary gate는 false-safe 0, model/physical lease-decision mismatch 0, deadline miss 0이다.
Confusion matrix는 QSU/QSN/MI/UQ 전체를 보고하며, false-conservative도 숨기지 않는다.

### C163-E — 강한 baseline과 usefulness screen

모든 안전 시스템에 같은 max-radio subset, observable feature, endpoint queue, service bound,
AI arrival/context/deadline과 fault rule을 준다.

| System | 차이 |
|---|---|
| Static global-safe | all-fail recovery 구간을 고정 예약 |
| Safe work-conserving | success 뒤 현재 contiguous gap만 회수, multi-credit atomic retiming 없음 |
| SoftWall | global multi-credit retiming + atomic AI lease + fence lifecycle |
| Offline oracle | 미래 outcome/arrival을 아는 상한; 비교용 |

Uncontrolled MPS와 local-only certificate는 실패 진단군이며 안전 성능 비교군으로 사용하지
않는다. Calibration trace에서 exact offline oracle로 SoftWall과 best safe online baseline의
적시 token-value headroom을 계산한다.

- Headroom `<5%`: confirmatory performance holdout을 열지 않고 no-material-headroom을 보고한다.
- Headroom `>=5%`: test trace를 열기 전에 paired block 수를 power analysis로 고정한다.
- Holdout은 두 node에서 반대 순서의 balanced Latin/ABBA block을 사용한다.
- Primary metric은 deadline 내 input-token value다.
- Safety violation 0, radio utility 비열등, 각 node 방향 일치, paired bootstrap 95% CI 하한
  `>0`, 중앙 개선율 `>=5%`를 모두 만족해야 성능 우위를 주장한다.

실제 expiry가 현재 P180/D155와 다른 경합을 만들 수 있으므로 opportunity screen 자체는
필요하다. 결과가 작다는 이유로 deadline, AI SLO, trace window 또는 비교군을 사후 변경하지
않는다.

### C163-F — 독립 holdout과 최종 판정

Development 완료 뒤 다음을 hash로 봉인한다.

```text
source + analyzer + raw/clock/expiry/mode schema
service bounds + guard + AI class table
node exclusion + seed range + case/block order
all hard gates + minimum effect size
```

Holdout node는 development node와 C162 boundary node를 제외한다. Source를 수정하면 holdout을
폐기하고 새 ID로 다시 development부터 시작한다. Final production PASS에는 C163-A--D와
holdout이 필수다. C163-E는 성능 claim에만 필수다.

## 4. C164 — lifecycle을 mode 축으로 자격화

Warm bound를 다음 상태에 상속하지 않는다.

| Lifecycle mode | 독립 episode 목표/node | 계측 범위 |
|---|---:|---|
| process/model cold start | 30 | readiness 전 초기화와 첫 20 request |
| MPS daemon/client restart | 30 | restart, reconnect, 첫 20 request |
| worker/broker reconnect | 30 | ambiguous token reconciliation과 RAN continuation |
| 30초 idle | 30 | 첫 20 request |
| 5분 idle | 10 | 첫 20 request |
| 30분 idle | 3 | 첫 20 request |
| Python GC ON | 1,000 request | GC event와 whole-path tail |
| Qwen unload/reload | 30 | memory residency와 첫 fence completion; mandatory-continuity subset 두 node 완료 |

각 mode는 C163-B bound qualification과 C163-D boundary subset을 다시 수행한다. Warm보다
느리더라도 독립 bound로 통과할 수 있다. GPU hang, driver reset 또는 completion evidence가
없는 crash는 fail-open하지 않고 UQ/quarantine으로 남긴다.

**현재 C164 진행:** lifecycle-scoped token 상태모델은 9개 mode와 8개 restart 전이에서
unsafe optional admission0으로 통과했다. `idle_30s_first` boundary subset은 두 독립 A100
node·반대 case order·180 round에서 actual NRx720, shared recovery330, Qwen90, commit720,
deadline miss0으로 통과했다. Ready→schedule idle은 30.029131/30.029712초였다.
`mps_restart_first`도 두 다른 A100 node에서 이전 daemon의 실제 4-GPU client epoch, 완전
종료, 다른 control/server PID와 socket inode의 새 epoch를 확인한 뒤 fresh-client
requalification을 수행했다. 180 round 합계 actual NRx720·recovery330·Qwen90·commit720,
miss0으로 boundary subset을 통과했다. Restart 중 service continuity는 주장하지 않는다.
이 결과는 [C164 결과](SOFTWALL_C164_IDLE30_RESULT_KO.md)에 있으며 표의 process/model cold,
worker process replacement, 5분·30분 idle, GC와 full class vector는 계속 UQ다. 추가로 Qwen reload를
node당 30회 수행하면서 같은 GPU2의 4-cell mandatory cuPHY를 계속 실행했다. 두 node 합계
reload60, mandatory release3,334, cell decode13,336에서 deadline miss와 cell25 위반은 0이었다.
Optional inference는 reload 동안 0으로 닫았으므로 이 결과는 mandatory-continuity subset이며
reload 중 AI availability 자격은 아니다.

Same-worker channel reconnect는 두 독립 A100 node의 token120에서 prepare-loss60,
post-fence-loss60, physical Qwen60, terminal fence120을 처리했고 concurrent mandatory
release145·cell580에서 miss와 cell25 위반0으로 통과했다. 개별 결과를
[C164 lifecycle qualification matrix](../../results/softwall_multigpu/c164_lifecycle_qualification_summary_v1.json)로
합친 현재 판정은 qualified/partial subset 5개와 UQ mode 5개인
`C164_LIFECYCLE_MATRIX_PARTIAL`이다. Matrix는 restart downtime을 availability로, Qwen reload
mandatory continuity를 optional boundary로 승격하지 않는다. 여기서 lifecycle expansion을
중단한다. Qualified/partial 5개가 현재 논문 claim을 지지하며, cold·process replacement·
5분/30분 idle·GC ON은 명시적인 UQ/future-work mode로 남긴다. 이미 시작했던 process-
replacement 단일-node development canary는 PASS했지만 independent holdout을 열지 않고
exploratory evidence로만 보존한다. Claim/evidence freeze와 audited 영문 초고는 완료했다.
다음 내부 gate는 venue 원고 압축, 도표·bibliography와 reviewer-objection audit이다.

## 5. C165 — 다른 GPU family의 외적 타당성

A100 이외 GPU를 확보하면 다음 vector를 새로 만든다.

```text
(GPU family, interconnect, driver, CUDA/MPS version,
 cuPHY/NRx engine, Qwen build, memory residency,
 B_NRx, B_recovery, B_control, B_AI[class], guard)
```

실행 순서는 mandatory-only → service qualification → four-debt all-fail → QSU/QSN 인접 경계
→ first-MI debt → C161 fault subset이다. A100의 bound나 cap을 복사하지 않는다. 같은 family의
개발/holdout 두 node가 가능하면 분리하고, 한 node뿐이면 single-node limitation을 명시한다.

Cross-family 표는 pass rate 평균이 아니라 family별 qualified/unqualified mode vector와 실패
원인을 보고한다. C165가 없으면 논문 범위를 A100 family로 제한한다.

## 6. 통계와 판정 규칙

1. TB를 독립 표본으로 간주한 과도한 유의성 검정을 하지 않는다. 분석 단위는 paired block,
   episode, node다.
2. Safety는 평균이나 p99가 아니라 violation count와 raw maximum을 함께 보고한다.
3. Zero violation은 finite observation이며 WCET가 아니다. Event/episode/node 단위의
   one-sided 상한은 민감도 분석으로만 제공한다.
4. Bound, seed, case, trace window, MDE와 baseline은 holdout 전에 고정한다.
5. 실패, timeout, analyzer bug, source 변경과 re-run을 ledger에 모두 남긴다.
6. False-safe, physical completion 전 credit reuse, duplicate commit은 한 건으로 해당 mode를
   UQ로 내린다.

## 7. 주장 승격표

| 최종 상태 | 논문에 추가할 수 있는 문장 |
|---|---|
| 현재 C162 + C163 readiness | qualified warm A100 synthetic contract의 predictive substrate |
| C163-A--D 두-node PASS | 실제 request별 DU expiry를 사용한 finite-sample AI-RAN timing claim |
| C163-E 성능 gate PASS | best safe online baseline 대비 적시 AI value 우위 |
| C163-E opportunity `<5%` | production mode에서도 no-material-headroom; 안전 기여만 유지 |
| C164 특정 mode PASS | 그 lifecycle에 한정한 보장 |
| C165 PASS | 통과한 두 GPU family 범위의 외적 타당성 |
| 어느 단계든 false-safe | 해당 mode UQ, 이전 qualified mode까지만 주장 |

## 8. 논문 결과물 연결

| 결과물 | 실험 |
|---|---|
| Request별 expiry와 all-fail schedule 그림 | C163-C |
| Predicted/measured QSU·QSN·MI confusion matrix | C163-D |
| Static/work-conserving/SoftWall/oracle token-value 표 | C163-E |
| Clock·tracing·controller overhead 표 | C163-A/B |
| Lifecycle feasibility heatmap | C164 |
| GPU-family mode-vector 표 | C165 |

## 9. 현재 완료 상태와 바로 다음 실행

- 완료: raw-event→contract builder, request-specific exact production bridge, artifact hash,
  calibrated clock error, explicit expiry, physical lease parity validator, 단위시험 24개.
- 미완료: target DU/FAPI raw trace, explicit gNB `d_MAC`, clock calibration, 같은 topology의
  production mode qualification, independent holdout.
- 현재 판정 artifact:
  [C163 readiness v2](../../results/softwall_multigpu/c163_du_timing_readiness_v2.json).

따라서 다음 물리 실행은 임의 synthetic GPU run이 아니다. 먼저 target DU/MAC에서 explicit
expiry artifact를 확보하고 C163-A raw event를 생성해야 한다. 그 입력이 없으면 C164/C165를
별도로 진행할 수는 있어도 production deadline claim은 계속 UQ다.
