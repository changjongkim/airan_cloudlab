# C157A/C157B: 실제 TensorRT NeuralRx가 구동한 shared-recovery transaction

**상태:** 2026-09-24, two-node warm synthetic actual-NRx transition PASS  
**Development:** job `58854900`, node `nid001109`, seed base `25800000`  
**Frozen holdout:** job `58855194`, node `nid001085`, seed base `25900000`  
**권위 판정:** `TWO_NODE_ACTUAL_NRX_TRANSITION_PASS_WARM_SYNTHETIC`

## 1. 닫은 질문

C154/C155와 C156/C156b는 success/fail 결과를 실행 전에 정한 controlled-outcome
실험이었다. C157은 다음 미자격 연결을 실제 경로로 바꿨다.

```text
같은 noisy TB
  -> GPU3 persistent TensorRT NeuralRx의 실제 LLR/LDPC/CRC
  -> CRC success이면 global recovery credit 해소
  -> CRC failure 또는 45 ms cutoff 뒤이면 GPU2 shared cuPHY recovery 유지
  -> 실제 decision 시각에 control5 + Qwen35 lease와 남은 recovery를 원자 재계산
  -> GPU fence 뒤 lease retire
  -> 원 home에 actual NRx 또는 conventional 결과를 한 번만 commit
```

따라서 이 실험의 outcome source는 branch injection이 아니라 실제 TensorRT NeuralRx의
CRC다. 미래 결과를 controller 입력으로 쓰지 않았다.

## 2. 물리 배치와 사전 고정

| GPU | 역할 |
|---|---|
| GPU0 | home0의 세 PUSCH owner |
| GPU1 | home1의 한 accepted PUSCH owner |
| GPU2 | 두 home이 공유하는 persistent cuPHY conventional worker + MPS20 Qwen |
| GPU3 | persistent actual TensorRT NeuralRx endpoint |

각 owner는 같은 noisy PUSCH에서 NeuralRx용 full tensor와 recovery용 IQ window를 만든다.
두 payload는 서로 다른 CUDA IPC handle로 공개되고 NVLink P2P로 GPU3 또는 GPU2에 전달된다.

결과를 열기 전에 다음 규칙을 고정했다.

- 3+2 debt를 제출하고 global certificate는 네 개만 수락한다.
- home마다 하나씩인 두 요청은 `+20 dB`, 나머지 두 요청은 `-15 dB`다.
- 새 seed에서 두 `+20 dB` 요청만 실제 NRx success가 될 것을 primary outcome gate로 둔다.
- 두 `-15 dB` 요청은 실제 NRx failure 뒤 shared recovery를 실행한다.
- 두 success가 45 ms cutoff 안에 관측될 때만 context-64 Qwen lease 하나를 시도한다.
- Holdout은 development node를 제외하고 source/mode hash를 그대로 사용한다.

저 SNR에서 conventional CRC도 실패할 수 있으므로 recovery correctness를 TB 성공으로
왜곡하지 않았다. Local owner가 같은 입력을 isolated conventional로 미리 계산한다.
Oracle CRC가 성공하면 shared result의 CRC와 payload가 모두 같아야 한다. Oracle CRC가
실패하면 shared result도 CRC 실패여야 한다. 실패 CRC의 payload bit는 radio가 소비할 수
없는 값이므로 byte identity를 요구하지 않는다.

## 3. 두 노드 결과

| 항목 | C157A development | C157B frozen holdout |
|---|---:|---:|
| 제출 / 수락 / global reject | 5 / 4 / 1 | 5 / 4 / 1 |
| 실제 NRx success | 2 | 2 |
| Shared cuPHY recovery | 2 | 2 |
| Conventional-oracle equivalent | 2 | 2 |
| Qwen lease / completion | 1 / 1 | 1 / 1 |
| Radio single commit | 4 | 4 |
| Deadline miss | 0 | 0 |
| 최대 NRx release→complete | 15.505 ms | 37.660 ms |
| Qwen execution | 27.780 ms | 26.386 ms |
| 최대 radio release→commit | 107.674 ms | 107.054 ms |
| 전체 gate | 12/12 PASS | 12/12 PASS |

두 노드 합계는 debt 제출10, 수락8, global reject2, actual NRx success4, physical recovery4,
oracle-equivalent recovery4, Qwen2, radio commit8, deadline miss0이다. 두 protocol의 source hash,
mode와 placement는 같고 job, node와 seed는 다르다. Holdout protocol은 development node
`nid001109`를 실행 전에 제외했다.

## 4. 실패가 교정한 mode

Development attempt를 덮어쓰지 않았다.

| Attempt | 관측 | 판정과 교정 |
|---|---|---|
| 1 | Owner 초기화와 release lead를 5 s request timeout에 잘못 포함 | outcome 판정 전 orchestration FAIL; readiness wait로 분리 |
| 2 | 첫 P2P path cold tail, 네 outcome 모두 cutoff 뒤; 실패 CRC payload bit 불일치 | all-fail 안전 동작 확인; failed-CRC semantic equivalence로 수정 |
| 3 | peer mapping prewarm 뒤에도 첫 path 52.057 ms | lifecycle 원인 후보 분리 |
| 4 | release 91.964 ms 전 activation 뒤 첫 path 50.476 ms | 단순 idle 설명 기각 |
| 5 | release 13.956 ms 전 activation 뒤 첫 path 102.618 ms | 촘촘한 activation으로도 해결되지 않음을 확인 |
| 6 | forward host 0.077 ms, TensorRT 1.257 ms, backward 0.049 ms인데 전체 59.237 ms | timed path의 첫 `cp.isfinite` CuPy JIT가 원인; monitor reduction을 readiness 전에 compile |
| 7 | 최대 actual NRx 15.505 ms, 12/12 PASS | development source 동결 |

Attempt6의 component timestamp가 중요하다. GPU service가 아니라 safety monitor의 첫 JIT를
service time에 포함한 것이었다. Monitor kernel을 readiness 전에 compile한 뒤 기존의 검증된
source-side P2P copier로 되돌렸고, 별도 node holdout에서도 통과했다.

## 5. 이 결과가 강화한 주장

C157 이전에는 다음 세 증거가 분리돼 있었다.

1. C114의 실제 remote TensorRT NeuralRx와 P2P path
2. C154/C155의 controlled outcome global certificate·Qwen·shared recovery transaction
3. C156/C156b의 실제 GPU timeline과 V17.1 launch-time revalidation

C157은 이들을 한 event chain으로 연결한다. 같은 TB의 실제 NRx CRC가 global debt를
해소하거나 유지하고, 그 결과가 Qwen lease와 두 shared conventional recovery를 결정하며,
모든 accepted radio request가 한 번만 commit된다. 이는 “실제 NRx는 한 실험, recovery
scheduler는 다른 실험”이라는 이전 공백을 닫는다.

## 6. 남는 범위

이 결과는 다음으로 확장하지 않는다.

- 45/25/35 ms를 WCET로 해석
- cold 또는 장시간 idle lifecycle 자격
- 실제 DU/FAPI `d_MAC`
- broker, worker, timeout과 stale reply의 integrated fault matrix
- static global-safe 또는 event-driven global-safe 대비 Qwen 처리량 우위
- 다른 GPU family와 실제 field channel 일반화

현재 자격은 `warm synthetic actual-NRx transition`이다. 다음 순서는 같은 actual-NRx path를
사용하는 integrated fault matrix와 strong safe baseline trace replay다.

## 7. Artifact

- [두 노드 결합 판정](../../results/softwall_multigpu/confirm157_actual_nrx_two_node.json)
- [Development PASS](../../results/softwall_multigpu/confirm157a_actual_nrx_attempt7_job58854900_result.json)
- [Development protocol](../../results/softwall_multigpu/confirm157a_actual_nrx_attempt7_job58854900_protocol.json)
- [Frozen holdout PASS](../../results/softwall_multigpu/confirm157b_actual_nrx_holdout_job58855194_result.json)
- [Frozen holdout protocol](../../results/softwall_multigpu/confirm157b_actual_nrx_holdout_job58855194_protocol.json)
- [53-file artifact manifest](../../results/softwall_multigpu/confirm157_actual_nrx_manifest.json)
- [Attempt 1 failure](../../results/softwall_multigpu/confirm157a_attempt1_failure.json)
- [Attempt 2 failure](../../results/softwall_multigpu/confirm157a_attempt2_failure.json)
- [Attempt 3 failure](../../results/softwall_multigpu/confirm157a_attempt3_failure.json)
- [Attempt 4 failure](../../results/softwall_multigpu/confirm157a_attempt4_failure.json)
- [Attempt 5 failure](../../results/softwall_multigpu/confirm157a_attempt5_failure.json)
- [Attempt 6 failure](../../results/softwall_multigpu/confirm157a_attempt6_failure.json)
