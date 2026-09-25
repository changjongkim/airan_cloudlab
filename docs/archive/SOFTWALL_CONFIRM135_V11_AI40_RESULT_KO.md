> **보관 사유 (2026-09-25):** 이 문서는 실험 이력으로 보존한다. 최신 스킴과 claim boundary는 `docs/current`의 원고·모델·production gate 문서가 권위 기준이며, 과거 GPU0 전용, MIG, CloudLab 또는 4-GPU pool 가정은 현재 논문의 근거가 아니다.

# Confirm135: conditional-slack 후보의 물리 검증과 V12 guard 감사

**상태:** 2026-09-24, 실행 전 protocol 동결 후 독립 두 arm PASS, V12 사후 모델 교정 PASS  
**범위:** 두 A100 RAN home, home당 4셀·NRx endpoint 2개·IPC ring depth 1,
MIG OFF/MPS ON, synthetic context-128 Qwen request  
**판정:** `AI 40 ms + control 15 ms + AI completion guard 2 ms`의 완전한 57 ms
exchange-only class가 실제 conditional-recovery 분기에서 두 seed 모두 실행됐다.

## 1. 왜 이 실험이 필요했는가

Envelope V10은 endpoint마다 한 release 안에서 completion slot을 반복 사용할 수 있다고
가정해 home당 NRx 4개를 모두 수락했다. 실제 CUDA-IPC endpoint는 input/output ring이
각각 한 slot이고, 한 release 동안 endpoint당 최대 한 작업만 outstanding이다. V10은
구현보다 많은 NRx를 수락하는 추상화였다.

또한 executor phase가 mode마다 달랐다. C113/C121/C122의 legacy/sharded controller는
NRx가 거절된 요청의 mandatory recovery를 exchange 전에 실행하지만, C125 이후의
certificate executor는 그 recovery를 live credit으로 유지한 채 exchange를 시도한다.
V11은 ring depth와 이 ordering을 mode vector에 넣었다. 이후 일관성 감사에서 V11
checker가 radio commit guard와 별개인 AI completion guard 2 ms를 `B_eff`에 명시적으로
더하지 않은 것을 찾았다. 실제 C135 runtime은 처음부터 `admission_ai_guard=17 ms`로
broker 15 ms와 completion guard 2 ms를 모두 청구했다. V12는 모델만 이 실행과 맞췄다.

두 home·home당 4셀의 full-control mode에서 교정 단계는 다음과 같다.

| 항목 | V10 | V11 | V12 |
|---|---:|---:|---:|
| home당 NRx 수락/거절 | 4/0 | 2/2 | 2/2 |
| endpoint ring depth | 무제한 completion slot처럼 계산 | `[1,1]` | `[1,1]` |
| exchange 때 rejected recovery | 암묵적으로 먼저 실행 | 2개 모두 live | 2개 모두 live |
| conditional release 상한 | 75 ms | 50 ms | 50 ms |
| 결정시각 뒤 안전 transaction window | 38 ms | 58 ms | 58 ms |
| 완전한 AI transaction | completion guard 누락 | 55 ms로 계산 | **57 ms, 수락·여유 1 ms** |

[V10→V11 regression](../../results/softwall_multigpu/softwall_envelope_v10_v11_ring_phase_regression_v1.json)은
ring/phase 변화를, [V11→V12 regression](../../results/softwall_multigpu/softwall_envelope_v11_v12_ai_guard_regression_v1.json)은
completion guard 교정을 고정된 prediction에서 자동 검사한다.

## 2. V12의 정확한 계약

한 home의 동시 요청 네 개에서 endpoint 두 개의 ring depth가 각각 1이므로 NRx는 두 개만
수락된다. 나머지 두 conventional recovery는 certificate에 남는다. Full-control executor는
두 NRx 결과를 최대 45 ms에 관측하며, 두 개가 모두 성공하면 다음 상태가 된다.

```text
요청 수                         n = 4
NRx 수락/성공                   k = 2
계속 남은 recovery              m = 2
conditional credit release      Delta = min(k, n-1) * 25 = 50 ms
두 recovery의 earliest start    155 - 2 guard - 2*25 = 103 ms
observe-all decision bound       45 ms
safe transaction window          103 - 45 = 58 ms
AI + control transaction         40 + 3*5 = 55 ms
AI completion guard              2 ms
complete effective transaction   40 + 15 + 2 = 57 ms
worst-case modeled margin         58 - 57 = 1 ms
```

이 완전한 class는 all-fail base slack 53 ms에는 들어가지 않고, 성공 결과로 50 ms credit을
조건부 삭제한 뒤의 decision window에만 들어간다. 따라서 실제로 실행되면 단순 idle-gap
사용과 구별되는 conditional recovery-to-AI exchange의 직접 증거가 된다.

## 3. 사전 고정 protocol

[Campaign protocol](../../results/softwall_multigpu/confirm135_v11_ai40_campaign_protocol.json)은
두 physical arm 전에 다음을 고정했다.

- `P180/D155`, conventional 25 ms, NRx 45 ms, physical guard 2 ms
- 두 home, home당 4셀, NRx endpoint 2개, endpoint별 ring depth 1
- broker `prepare/commit/complete` 각각 5 ms, 총 control budget 15 ms
- AI completion guard 2 ms를 더한 실제 admission guard 17 ms
- AI context 128, raw execution bound 40 ms
- 독립 payload/channel seed 두 세트, arm당 160 release와 20 warmup
- 각 arm에서 정확한 2-admitted/2-success/2-rejected 분기와 AI40 exchange가 최소 한 번
  나타나야 함
- 첫 deadline/bound/credit 실패에서 중단하고 bound를 사후 확대하지 않음

동결 protocol의 설명 필드 `model_margin_ms=3`은 V11 산술을 보존하므로 수정하지 않았다.
그러나 실행 인자와 gate는 이미 `admission_ai_guard_ms=17`을 사용했다. V12 사후 감사는
동결 입력이나 결과를 바꾸지 않고 이 17 ms를 `control 15 + completion guard 2`로 분해해
계약 여유를 1 ms로 바로잡는다.

기존 BurstGPT trace에는 context-128 request가 하나뿐이어서 이 희귀 분기를 충분히
노출하지 못했다. 동결된 C132--134 artifact 24개, home release 8,160개를 재감사하면
목표 분기는 3,771번 있었지만 그때 제공된 AI bound는 모두 65 또는 75 ms였고 AI40은
0번이었다. 따라서 [context-128 deterministic trace](../../data/current/softwall_context128_mechanism_trace_v1.json)를
별도 mechanism workload로 만들었다. 이는 대표 처리량 workload가 아니라 예측한 경계를
검증하기 위한 입력이다.

## 4. 물리 결과

새 interactive allocation `58826874`, node `nid001245`의 A100 4개 중 두 개를 RAN
home으로 사용했다. 두 독립 arm 모두 모든 사전 gate를 통과했다.

| 항목 | Arm s1 | Arm s2 | 합계/최악값 |
|---|---:|---:|---:|
| Radio TB | 1,280 | 1,280 | 2,560 |
| Atomic exchange | 10 | 9 | 19 |
| 정확한 후보 분기 | 141 | 147 | 288 |
| 후보 분기의 AI40 exchange | 5 | 5 | 10 |
| 최소 실제 reserved-horizon margin | 51.056 ms | 51.287 ms | 51.056 ms |
| 최대 NRx admission/home/release | 2 | 2 | 2 |
| Deadline miss | 0 | 0 | 0 |
| NRx/conv/AI bound 위반 | 0 | 0 | 0 |
| Horizon/credit/uniqueness 위반 | 0 | 0 | 0 |

네 controller 전체에서 관측한 최대값은 NRx response 16.951 ms, conventional host path
5.782 ms, AI host execution 27.756 ms, AI GPU 26.639 ms였다. 이 값들은 선언한
45/25/40 ms 안이지만 표본 최대이며 WCET가 아니다.

실제 최소 horizon margin 약 51 ms가 모델 margin 1 ms보다 큰 이유는 NRx와 Qwen이
각각 선언 bound보다 일찍 끝났기 때문이다. 안전 계약의 여유는 계속 1 ms로 해석한다.
관측 여유를 새 bound나 추가 service guarantee로 승격하지 않는다.

[10개 사건 반사실 감사](../../results/softwall_multigpu/confirm135_static_counterfactual_audit_v1.json)는
각 실제 교환에 같은 선언 bound를 대입했다. 10/10 모두 static all-fail 계약에서는
`53-57=-4 ms`로 거절되고, conditional 계약에서는 `58-57=+1 ms`로 수락되며, 실제
17 ms guard를 차감한 physical horizon margin도 음수가 아니었다. 이는 모델 반사실이며
별도 static baseline을 실행해 얻은 처리량 비교는 아니다.

## 5. 이 결과가 입증하는 것

1. Envelope가 구현의 physical ring credit과 executor ordering을 반영하면 이전 모델이
   놓친 exchange-only service class를 사전에 예측할 수 있다.
2. NRx 성공으로 conventional obligation 두 개를 삭제하면서도 나머지 두 recovery의
   executable certificate를 유지할 수 있다.
3. Raw AI 40 ms와 synchronous broker control 15 ms를 하나의 transaction bound로
   묶고 completion guard 2 ms까지 별도로 admission한 뒤 실제 Qwen을 실행하고 모든
   credit을 회수할 수 있다.
4. 따라서 atomic recovery-credit exchange는 단순한 bookkeeping이 아니라, static
   all-fail slack에는 들어가지 않는 bounded AI class를 물리적으로 허가한다.

이 결과는 safe work-conserving baseline 대비 총 처리량 우위, BurstGPT 대표성, WCET,
실제 DU `d_MAC`, 다른 GPU family, broker restart, durable reconciliation 또는
exactly-once를 입증하지 않는다. 논문의 정직한 주장은 **모델 예측 경계와 mechanism
실현의 일치**다.

## 6. 후속 독립 node 재자격

[C136](SOFTWALL_CONFIRM136_V12_REQUALIFICATION_RESULT_KO.md)은 이전 node들을 제외한
`nid002817`에서 새 seed/process lifecycle 6개 arm을 추가했다. 7,680 TB, candidate branch
908회, AI40 exchange 28회에서 모든 gate가 통과했다. C135와 합치면 두 A100 node,
8 arm, 10,240 TB, candidate branch 1,196회와 AI40 exchange 38회이며 선언 safety 위반은
0이다. 38/38 사건이 static −4 ms, conditional +1 ms 계약을 만족한다.

이 증거도 WCET가 아니다. Zero-failure 95% 상한은 TB를 IID로 보면 0.0293%지만, arm을
IID로 보면 31.23%, node를 IID로 보면 77.64%다. 논문에는 두-node finite-sample
requalification으로만 표현한다.

## 7. 독립 검증

- [C135 campaign result](../../results/softwall_multigpu/confirm135_v11_ai40_campaign_result.json)
- [C135 artifact manifest](../../results/softwall_multigpu/confirm135_artifact_manifest.json)
- [V11 validation summary](../../results/softwall_multigpu/softwall_envelope_v11_validation_summary.json)
- [V12 prediction](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v12.json)
- [V12 validation summary](../../results/softwall_multigpu/softwall_envelope_v12_validation_summary.json)
- [V12 artifact manifest](../../results/softwall_multigpu/softwall_sharded_home_envelope_manifest_v12.json)
- [V11→V12 guard regression](../../results/softwall_multigpu/softwall_envelope_v11_v12_ai_guard_regression_v1.json)
- [C135 static counterfactual audit](../../results/softwall_multigpu/confirm135_static_counterfactual_audit_v1.json)
- [C136 result](../../results/softwall_multigpu/confirm136_v12_ai40_requalification_result.json)
- [C136 artifact manifest](../../results/softwall_multigpu/confirm136_artifact_manifest.json)
- [C135/C136 combined audit](../../results/softwall_multigpu/confirm135_136_combined_v12_qualification.json)
- [Arm s1 result](../../results/softwall_multigpu/confirm135_s1_job58826874_result.json)
- [Arm s2 result](../../results/softwall_multigpu/confirm135_s2_job58826874_result.json)
- [V11 prediction](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v11.json)
- [Exact finite-ring oracle](../../results/softwall_multigpu/softwall_endpoint_admission_exact_oracle_audit_v3.json)
- [과거 trace evidence audit](../../results/softwall_multigpu/softwall_v11_candidate_physical_evidence_audit_v1.json)
- [Incremental observation 반증](../../results/softwall_multigpu/softwall_incremental_exchange_prospective_v2.json)

Exact oracle는 18개 scenario, 39개 endpoint pool과 10,920개 bounded finite-ring 상태에서
checker의 최대 수락 수와 차이 0, schedule 오류 0을 확인했다. Incremental observation은
현재 grid에 새 service class를 만들지 않아 구현하지 않았다. 이는 메커니즘을 추가하기
전에 모델로 효용을 반증한 사례로 보존한다.
