> **보관 사유 (2026-09-25):** 이 문서는 실험 이력으로 보존한다. 최신 스킴과 claim boundary는 `docs/current`의 원고·모델·production gate 문서가 권위 기준이며, 과거 GPU0 전용, MIG, CloudLab 또는 4-GPU pool 가정은 현재 논문의 근거가 아니다.

# SoftWall C123–C126: 글로벌 AI lease와 certificate-ordered 실행 결과

**상태:** 2026-09-24 authoritative 판정  
**범위:** 2×A100, GPU당 4-cell recovery home, MIG OFF, MPS ON, home별 local CUDA-IPC NeuralRx 2개와 Qwen worker 1개, 하나의 BurstGPT 요청 trace

## 1. 무엇을 추가했는가

C121은 두 GPU가 각각 독립 AI trace를 처리했다. C123부터는 외부 AI 요청을 한 번만 제공하는
`GlobalTraceLeaseBroker`를 두 home이 공유한다. Broker의 요청 상태는 다음과 같다.

```text
future -> ready -> held(home, generation) -> inflight -> completed | late
                       \-> ready | expired     (commit 전 abort만 허용)
```

한 home의 실행 순서는 다음과 같다.

```text
global request hold
  -> local recovery calendar + AI lease 원자 검사
  -> global request commit
  -> Qwen physical execution/fence
  -> local lease retire + global complete
```

동일 request는 한 home만 `held`할 수 있고 generation token이 맞아야 commit·abort·complete할
수 있다. Local atomic admission이 실패하면 commit 전 global hold를 반환한다. Commit 뒤의
실행 실패는 다른 home에 재할당하지 않고 fail-closed한다. 이는 global request ownership과
local certificate의 **serializable composition**이며, 여러 GPU kernel을 한 하드웨어
transaction으로 실행한다는 뜻은 아니다.

C125에서 더 중요한 executor 조건이 드러났다. Valid certificate를 계산하는 것만으로는
충분하지 않다. 실행기는 서로 다른 원인으로 ready가 된 recovery를 하나로 합쳐 live
certificate 순서대로 dispatch해야 한다.

```text
ready recovery = no-NRx + NRx fail/late + policy-forced fallback
order = reserved_start -> deadline -> slot_id
```

## 2. C123 — 하나의 global AI queue 물리 통합

두 independent arm이 모두 통과했다.

| 항목 | seed 1 | seed 2 |
|---|---:|---:|
| Offered request/value | 1,136 / 288,591 | 1,136 / 288,591 |
| Timely request/value | 1,134 / 288,209 | 1,134 / 288,209 |
| Home별 request 수 | 556 / 578 | 571 / 563 |
| 첫 release 차이 | 0 ns | 0 ns |
| 두 home 물리 실행 중첩 | 61.034 s | 61.033 s |
| Duplicate / outstanding token | 0 / 0 | 0 / 0 |

각 arm에서 `prepare=commit=complete=1,134`였고 두 home 모두 AI 요청을 받았다. 총 5,440 TB의
radio deadline·bound·fault·credit gate도 통과했다. 이 결과는 global request uniqueness와
local recovery/AI lease의 합성이 실제 두 GPU 경로에서 동작한다는 증거다.

## 3. C124 — 12 ms mode의 반증

Global routing과 static home partition을 두 seed, GSSG/SGGS로 비교했지만 마지막 여덟 번째
arm에서 conventional host-to-commit이 `12.535096 ms`로 12 ms 계약을 한 번 넘었다.
Deadline miss는 0이었어도 exact mode는 `Unqualified`다. C121의 두 통과 arm은 역사적
유한 표본으로 남지만, 2-home×4-cell의 conv12 자격은 C124가 반증했다.

성능 방향도 global이 seed별 `−0.068%`, `−0.132%`였고 bootstrap 하한은 음수였다.
Global routing 처리량 우위는 지지되지 않았다.

## 4. C125 — 산술상 feasible한 certificate를 executor가 어긴 반례

사후 관측치에 맞춘 13 ms 대신 기존 보수적 25 ms recovery class를 사용했다.

```text
D - guard - 4 * B_conv = 155 - 2 - 4*25 = 53 ms
```

그러나 첫 formal arm의 home 1, release 150에서 controller가 실패했다. 두 앞선 NeuralRx가
이미 강제 실패해 recovery-ready였지만, controller는 NRx 미선택 셀을 별도 루프에서 먼저
실행하려 했다.

| 항목 | 값 |
|---|---:|
| 뒤 셀 2의 시도 구간 | `[+33.106, +58.106] ms` |
| 앞 셀 0의 live credit | `[+53, +78] ms` |
| 결과 | interval overlap, launch 전 fail-closed |

즉 `4*25 <= D-guard`와 valid calendar만으로 안전 정리가 완성되지 않는다. Executor가
certificate의 partial order를 지켜야 한다. C125는 첫 실패에서 중단했고 나머지 7개 arm을
실행하지 않았다.

## 5. C126 — certificate-ordered executor

Frozen C123/C125 source를 수정하지 않고 새 controller를 만들었다. 모든 open recovery를
합친 뒤 live reservation 순서대로 dispatch한다. C125 geometry를 재현한 단위시험에서
셀 2의 선실행은 거절되고 0→1→2→3 순서는 outstanding credit 0으로 끝났다. 실패 seed의
release 150을 포함한 180-release canary도 통과했다.

이후 같은 25/45 ms 계약으로 8개 formal arm을 실행했다.

| 안전/구조 항목 | 결과 |
|---|---:|
| Formal arm | 8/8 wrapper PASS |
| 총 TB | 21,760 |
| Atomic recovery/AI exchange | 2,496 |
| Deadline miss | 0 |
| NRx bound 위반 | 0, max 38.394 ms / bound 45 ms |
| Conventional path bound 위반 | 0, max 13.685 ms / bound 25 ms |
| AI bound/horizon 위반 | 0 |
| 최대 radio commit response | 79.502 ms / deadline 155 ms |
| Broker duplicate/outstanding | 0 / 0 |
| Frozen source/raw artifact hash | 전부 일치 |

따라서 **certificate-ordered 2-home×4-cell conv25 mode는 유한 표본 QSU**다. 반면 campaign의
전체 structural gate는 FAIL이다. 동일 seed 반복 사이 timing-dependent NeuralRx admission이
달라 strict radio-decision parity가 깨졌다. 반복 내 결정 차이는 seed1 global 4건,
seed1 static 0건, seed2 global 4건, seed2 static 4건이며 대부분 index 0이었다. Seed1 global의
index 33 한 건만 추가 차이다. 이는 개별 arm의 safety를 깨지 않지만 동일 radio 결정을
전제로 한 처리량 인과 비교는 무효로 만든다.

처리량 outcome도 명확히 실패했다.

| Seed | Global 평균 | Static 평균 | 차이 | 효과 | source-second bootstrap 95% |
|---|---:|---:|---:|---:|---:|
| 1 | 288,209 | 288,195 | +14 | +0.0049% | `[0, 42]` |
| 2 | 288,263 | 288,195 | +68 | +0.0236% | `[0, 204]` |

두 seed 모두 사전 `+2%`와 CI 하한 `>0`을 만족하지 못했다. Global AI routing의 처리량
우위는 주장하지 않는다.

## 6. 모델에 추가된 조건

기존 conditional-safety theorem에는 다음 executor-conformance 조건이 필요하다.

> 같은 recovery lane에서 여러 obligation이 ready이면, 실제 dispatch는 live certificate와
> 양립해야 한다. 구체적으로 새 실행 interval이 더 이른 live credit과 겹치면 안 되며,
> 현재 구현은 `(reserved_start, deadline, slot_id)` 순서로 실행한다.

Certificate 존재는 admission의 안전성을 말하고, conformance는 그 certificate가 실제
실행 경로로 refinement되는지를 말한다. C125는 전자는 참이고 후자는 거짓인 실행 반례다.
C126은 두 조건을 결합한 구현의 물리 검증이다.

## 7. 현재 방어 가능한 novelty

C123–126이 추가한 좁은 차별점은 다음이다.

1. Optional per-TB NeuralRx가 만든 multi-cell all-fail recovery debt를 home별 executable
   certificate로 유지한다.
2. Local/remote endpoint와 bounded Qwen의 physical credit을 fence까지 보유한다.
3. 하나의 외부 AI request queue를 generation-tagged hold로 직렬화하고, 선택된 home의
   local recovery replan/AI lease와 합성한다.
4. 계산된 certificate를 실제 GPU dispatch로 보존하는 executor-conformance 조건과 반례를
   제시한다.
5. 같은 하드웨어에서 conv12 false-safe, class-ordered executor 실패, certificate-ordered
   conv25 통과를 하나의 feasibility envelope로 구분한다.

MPS, CUDA IPC/P2P, EDF, Qwen 배치, multi-GPU 자체는 기여가 아니다. 처리량 optimizer 또는
global routing 우위도 현재 결과가 지지하지 않는다.

## 8. 남은 핵심 한계와 다음 순서

1. 실제 DU/FAPI의 `d_MAC`을 확보해 synthetic D155를 production expiry로 교체한다.
2. Finite-sample bound를 cold/GC/restart와 독립 hardware에서 재자격한다.
3. 후속 C127은 post-apply commit 응답 유실을 affected-home AI 차단과 ambiguous-token
   quarantine으로 격리했다. Broker crash/restart와 complete ambiguity를 손실 없이 복구하려면
   durable epoch/WAL과 worker execution record 기반 reconciliation이 여전히 필요하다.
4. Strict 성능 비교를 다시 하려면 radio admission을 treatment와 독립적으로 고정하거나,
   충분한 cutoff margin을 사전 자격화해야 한다. 현재 자료로 성능 우위를 다시 탐색하지 않는다.
5. Shared remote NRx endpoint가 여러 home의 mandatory path와 경합하는 non-separable mode는
   multi-resource certificate가 필요하다.

## 9. 권위 artifact

- [C123 protocol](../../results/softwall_multigpu/confirm123_global_ai_broker_protocol.json)
- [C123 result](../../results/softwall_multigpu/confirm123_global_ai_broker_result.json)
- [C124 result](../../results/softwall_multigpu/confirm124_global_vs_static_result.json)
- [C125 first-arm failure](../../results/softwall_multigpu/confirm125_conv25_first_arm_failure.json)
- [C126 protocol](../../results/softwall_multigpu/confirm126_certificate_global_vs_static_protocol.json)
- [C126 frozen result](../../results/softwall_multigpu/confirm126_certificate_global_vs_static_result.json)
- [C126 post-hoc decomposition](../../results/softwall_multigpu/confirm126_structure_posthoc.json)
- [Envelope v4](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v4.json)
- [Artifact manifest](../../results/softwall_multigpu/confirm125_126_artifact_manifest.json)
- [C127 broker fault containment](SOFTWALL_CONFIRM127_BROKER_FAULT_RESULT_KO.md)
