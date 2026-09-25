# C156: shared-recovery GPU timeline과 launch-time revalidation

**상태:** 2026-09-25, two-node controlled-outcome GPU timeline semantics PASS  
**Jobs / nodes:** `58853926` / `nid001085`, `58854401` / `nid001064`  
**권위 결과:** `confirm156_timeline_attempt3_job58853926_result.json`

## 1. 검증 질문

C154/C155는 host event와 CUDA event duration으로 global certificate, Qwen lease와 shared
cuPHY recovery의 통합을 검증했다. C156은 Nsight Systems의 실제 GPU activity를 사용해
다음 실행 의미를 직접 확인한다.

1. Qwen kernel은 committed lease 안에서만 실행되는가.
2. Full-GPU-blackout 계약에서 Qwen과 conventional recovery kernel의 overlap이 0인가.
3. 두 home의 recovery가 certificate 순서로 실행되는가.
4. 각 recovery에서 forward P2P, input install, cuPHY conventional, backward P2P 순서가
   GPU activity에서도 보존되는가.

Profiler arm은 `conditional_open` branch다. 두 controlled NRx success 뒤 context-64 Qwen을
실행하고, 남은 두 debt를 GPU2의 shared cuPHY worker에서 처리한다.

## 2. 두 실패가 발견한 공백

### Attempt 1: 분석기 schema 호환성

첫 실행은 Qwen/cuPHY와 `.nsys-rep`/SQLite export까지 끝났지만 분석기가 Nsight 13.2의
`utcEpochNs` column을 처리하지 못해 판정 전에 중단됐다. 이전 Nsight의
`systemClockNs`와 새 UTC schema를 모두 지원하고, worker가 기록한 UTC↔monotonic anchor로
서로 다른 profiler session을 같은 clock에 맞췄다. 이 시도는 성능·안전 판정에서 제외하고
[실패 기록](../../results/softwall_multigpu/confirm156_attempt1_failure.json)으로 보존했다.

### Attempt 2: 고정 lease의 launch-time 공백

호환성 수정 뒤 두 번째 실행은 recovery order, phase causality, home commit과 금지 overlap
gate를 통과했지만 `qwen_fence_and_lease`가 실패했다. Profiler 아래 Qwen은 host
34.204 ms, GPU 32.390 ms였고 release 후 83.153 ms에 반환돼 기존 고정 lease
`[45,80] ms`를 3.153 ms 넘었다. 실제 Qwen kernel 44개도 80 ms 뒤에 끝났다.

원인은 AI bound 위반이 아니었다. Qwen host time은 선언한 35 ms 안이었지만 outcome 처리,
global replan과 RPC dispatch가 release 후 약 49 ms까지 소요됐다. 기존 worker는 개념상
45 ms에서 만든 lease를 launch 직전에 다시 맞추지 않았다. 이 결과는
[attempt2 FAIL](../../results/softwall_multigpu/confirm156_timeline_attempt2_job58853926_result.json)로
보존했다.

## 3. V17.1 수정

V17.1은 실제 launch decision 시각 `t_l`에서 전체 global certificate를 다시 만든다.

```text
lease = [t_l, t_l + B_launch_control + B_AI]
B_launch_control = 5 ms
B_AI             = 35 ms
```

이 blackout과 남은 모든 recovery debt를 한 exact transaction으로 검사한다. Transaction이
끝난 뒤 Qwen dispatch까지 5 ms를 넘으면 launch하지 않는다. Qwen fence가 일찍 오면 lease를
retire하고 현재 시각으로 recovery calendar를 다시 만든다. 늦은 상태에서 lease와 recovery를
모두 넣을 수 없으면 기존 state를 유지하고 AI를 거절한다. 결정적 경계시험에서는
conditional branch의 `t_l=49 ms`는 통과하고 `t_l=64 ms`는
`64+5+35+2×25=154 > 153 ms`라 원자적으로 거절된다.

이 수정은 C154/C155의 봉인된 source를 바꾸지 않고 새
`integrated_shared_recovery_launch_plan_v1.py`와 profiler worker에만 적용했다.

## 4. Attempt 3 결과

| 항목 | 결과 |
|---|---:|
| Launch revalidation host time | 1.745 ms |
| Revalidation→Qwen dispatch | 1.768 ms, 5 ms bound 안 |
| Dynamic lease | release 기준 47.153–87.153 ms |
| Qwen host / GPU | 32.087 / 30.148 ms |
| Qwen fence return | release 기준 81.008 ms |
| Qwen runtime kernels | 1,224 |
| Recovery runtime kernels | 106 |
| Worker runtime GPU events | 154 |
| NVTX phase ranges | 8 |
| Unattributed worker event | 0 |
| Qwen–recovery forbidden kernel overlap | **0 ns** |
| Physical recovery / correct home commit | 2 / 2 |
| 최장 recovery complete | release 기준 119.395 ms |
| D155 miss | 0 |

두 recovery 각각에서 Nsight activity는 다음 순서를 지켰다.

```text
P2P forward memcpy
  -> input install kernels
  -> cuPHY conventional kernels/memcpy/memset
  -> P2P backward memcpy
```

각 conventional 구간은 kernel49, memcpy18, memset4를 포함했고, install은 kernel4,
forward/backward는 각각 P2P memcpy1을 포함했다. 두 recovery의 GPU 구간은 certificate 순서와
같았고 모든 11개 gate가 통과했다.

### 4.1 독립 node holdout

C156b는 `nid001085`를 allocation에서 제외하고 새 job `58854401`, node `nid001064`,
분리된 seed에서 동일 source와 gate를 실행했다. 11/11 gate가 다시 통과했다. 두 node를
결합하면 Qwen kernel2,448, recovery kernel212, worker event308, NVTX range16,
physical recovery4, correct commit4/4이며 미귀속 event, forbidden overlap과 deadline miss는
모두 0이다. Source/mode/profiler hash는 같고 seed와 job/node는 분리됐다.

## 5. 해석 범위

C156/C156b는 controlled outcome에서 **모델이 발급한 lease와 recovery 순서가 실제 GPU
timeline에 보존됨**을 두 A100 node에서 보여 준다. 특히 profiler가 만든 지연으로 고정
lease 공백을 드러내고, launch control budget을 certificate에 포함한 뒤 독립 node에서
재현했다는 점이 중요하다.

다음 범위에는 확장하지 않는다.

- Profiler sample을 service bound나 WCET로 해석
- 실제 TensorRT NeuralRx output-driven transition
- Worker/broker/timeout fault matrix
- Static 또는 event-driven global-safe baseline 대비 처리량 우위
- Production DU `d_MAC`

## 6. Artifact

- [Passing result](../../results/softwall_multigpu/confirm156_timeline_attempt3_job58853926_result.json)
- [Independent-node result](../../results/softwall_multigpu/confirm156b_timeline_holdout_job58854401_result.json)
- [Two-node combined result](../../results/softwall_multigpu/confirm156_156b_two_node_gpu_timeline.json)
- [Passing protocol](../../results/softwall_multigpu/confirm156_timeline_attempt3_job58853926_protocol.json)
- [Attempt 1 failure](../../results/softwall_multigpu/confirm156_attempt1_failure.json)
- [Attempt 2 gate failure](../../results/softwall_multigpu/confirm156_timeline_attempt2_job58853926_result.json)
- [73-file manifest](../../results/softwall_multigpu/confirm156_gpu_timeline_manifest.json)
