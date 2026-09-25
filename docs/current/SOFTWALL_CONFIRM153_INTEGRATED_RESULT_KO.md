# C153: global shared-recovery certificate와 Qwen/cuPHY 물리 transaction 통합

**상태:** 2026-09-25, development canary PASS / V17 integrated holdout UQ  
**통과 실행:** job `58852924`, node `nid001348`  
**범위:** injected NeuralRx outcome의 한-release mechanism canary

## 1. 무엇을 새로 연결했는가

C151/C152까지는 두 RAN home의 실제 PUSCH를 한 shared cuPHY conventional worker가 처리했지만
실행 순서는 정적 `(sequence, home)`이었다. V17 global certificate와 Qwen lease는 CPU
모델에만 있었다. C153은 다음 경로를 한 generation chain으로 연결한다.

```text
두 home의 3+2 conditional recovery debt
  -> global all-fail admission
  -> fifth debt reject before launch
  -> 두 NRx success credit release
  -> recovery replan + Qwen35 lease atomic commit
  -> physical Qwen completion fence
  -> lease retire at current time
  -> certificate-issued shared cuPHY/P2P recovery order
  -> decoded TB/CRC return
  -> home commit before D155
```

Mode 후보는 `P180/D155`, `NRx45`, `conv25`, `Qwen35`, `guard2`, shared recovery
capacity 1이다. 각 home의 local debt 3개와 2개는 각각 가능하지만 합집합의 다섯 recovery는
`45+5×25+2=172>D155`라 불가능하다. 네 recovery는 147 ms에 가능하고, 그중 두 success로
debt가 해제되면 `45+2×25+35+2=132 ms`에 Qwen을 넣을 수 있다.

## 2. 보존한 development 실패

| Attempt | 결과 | 판정 |
|---|---|---|
| job `58852725`, `nid001176` | 긴 run-state 아래 Qwen Unix socket이 `AF_UNIX path too long`으로 bind 전 실패 | 물리 메커니즘 미실행. [실패 기록](../../results/softwall_multigpu/confirm153_attempt1_failure.json)만 보존 |
| job `58852876`, `nid001348` | false-safe·AI threshold·Qwen·decode 순서는 통과했지만 release 뒤 합성 `apply_rayleigh_awgn`을 실행해 home commit 290.357/293.912 ms, D155 실패 | release 정의 오류. 실제 계약의 `t_IQ_ready` 전에 합성 채널 준비를 이동하고 원본 FAIL 유지 |
| job `58852924`, `nid001348` | `t_IQ_ready`에서 publish하도록 수정, 10/10 gate 통과 | C153 development mechanism PASS |

두 번째 실패에서 conventional 자체는 GPU 6.640/2.856 ms였고 두 payload도 정답이었다.
deadline miss의 263.820--275.158 ms는 합성 IQ 생성·복사 준비를 release 뒤에 둔 harness
정의에서 발생했다. 수정 실행은 channel preparation 246.589/240.382 ms를 release 전에
완료하고, `t_IQ_ready` 뒤 0.002608/0.002208 ms에 IPC request를 publish했다. 이 변경은
deadline을 느슨하게 만든 것이 아니라 release를 실제 PHY 입력 준비 의미에 맞춘 것이다.

## 3. 통과 결과

| Gate/metric | C153 결과 |
|---|---:|
| Local certificate | home0 3개 feasible, home1 2개 feasible |
| Global admission | 첫 4개 accepted, fifth `h1-r1` rejected |
| Rejected update mutation | 0 |
| Rejected request physical launch | 0 |
| 한 success 뒤 AI35 | `lease_breaks_global_certificate` |
| 두 success 뒤 AI35 | `lease_committed`, generation 7 |
| Qwen context64 host/GPU | 23.059 / 21.733 ms |
| Qwen lease interval | release+45.941 → +69.000 ms, 45--80 ms 안 |
| Lease retire | physical fence 뒤 generation 8 |
| Recovery order | `home0/h0-r2` → `home1/h1-r0`, certificate와 일치 |
| Conventional GPU | 5.997 / 2.642 ms |
| Release→physical recovery complete | 82.250 / 97.420 ms |
| Release→home commit | 82.412 / 97.644 ms |
| Correct TB/CRC | 2/2 |
| Deadline miss | 0/2 |
| Source/config/P2P/IPC/process error | 0 |

Qwen response는 worker-side CUDA synchronization 뒤에 반환되므로 이 response를 physical
completion fence로 사용했다. Lease retire는 response 시각의 현재 `now`로 certificate를
다시 계산했다. 첫 recovery가 실제로 일찍 끝났더라도 둘째 작업은 보수적 25 ms placement
시각인 release+94.000 ms까지 기다렸다. 따라서 executor가 관측 실행시간을 사용해 certificate
순서를 앞당기지 않았다.

## 4. 현재 주장과 남은 gate

C153이 직접 지지하는 주장은 다음이다.

> 한 shared recovery GPU에서 home-local certificate의 합성은 false-safe가 될 수 있다.
> Global executable certificate는 그 fifth debt를 물리 launch 전에 거절하고, 두 success가
> 해제한 credit으로만 bounded Qwen lease를 열며, GPU fence 뒤 남은 debt를 실제 cuPHY/P2P
> recovery와 home deadline commit으로 실행할 수 있다.

이 결과는 injected outcome, 한 node, 한 release의 development canary다. 다음은 아직
주장하지 않는다.

- 실제 NeuralRx output이 직접 발생시킨 success/fail 전이
- 독립 node와 seed의 integrated holdout
- all-fail/all-success/overload 반복 branch coverage
- shared worker 또는 broker fault containment
- service WCET, production `d_MAC`, throughput superiority
- V17 QSU

다음 승격 gate는 [최종 실험 계획](SOFTWALL_FINAL_EXPERIMENT_PLAN_KO.md)의 C154/C155다.
Current source를 별도 holdout protocol에서 동결하고 기존 C153 node를 제외한 두 A100 node에서
all-fail, conditional-open, all-success와 overload를 실행해야 한다.

## 5. Artifact

- [통과 protocol](../../results/softwall_multigpu/confirm153_integrated_shared_recovery_job58852924_protocol.json)
- [통과 result](../../results/softwall_multigpu/confirm153_integrated_shared_recovery_job58852924_result.json)
- [실패 attempt2 result](../../results/softwall_multigpu/confirm153_integrated_shared_recovery_job58852876_result.json)
- [33-file development manifest](../../results/softwall_multigpu/confirm153_development_manifest.json)
- [integrated scenario](../../scripts_for_node/softwall_same_gpu/integrated_shared_recovery_plan_v1.py)
- [physical worker](../../scripts_for_node/softwall_same_gpu/integrated_shared_recovery_worker.py)
- [physical owner](../../scripts_for_node/softwall_same_gpu/integrated_shared_recovery_owner.py)

