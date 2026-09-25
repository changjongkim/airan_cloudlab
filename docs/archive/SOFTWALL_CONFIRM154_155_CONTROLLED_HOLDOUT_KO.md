> **보관 사유 (2026-09-25):** 이 문서는 실험 이력으로 보존한다. 최신 스킴과 claim boundary는 `docs/current`의 원고·모델·production gate 문서가 권위 기준이며, 과거 GPU0 전용, MIG, CloudLab 또는 4-GPU pool 가정은 현재 논문의 근거가 아니다.

# C154/C155: shared-recovery 통합 mode의 두-node controlled-outcome holdout

**상태:** 2026-09-25, two-node controlled integrated holdout PASS / actual NeuralRx integration UQ  
**노드:** C154 `nid001032`, C155 `nid001025`  
**Jobs:** `58853453`, `58853542`

## 1. 검증 대상

C153은 global certificate가 Qwen fence와 shared cuPHY recovery를 실제로 구동하는 첫
one-node development canary였다. C154/C155는 source와 mode를 고정하고 C151--C153의 모든
node를 제외한 두 새 A100 node에서 네 outcome branch를 실행한다.

```text
P180 / D155 / NRx45 / conventional25 / Qwen35 / guard2
shared recovery capacity = 1
GPU0 = home0 owners
GPU1 = home1 owners
GPU2 = MPS cap80 shared cuPHY worker + cap20 Qwen
```

Release는 합성 channel preparation을 마친 `t_IQ_ready`다. 각 physical recovery는 독립 CUDA
IPC buffer와 owner를 가지며, global certificate가 발급한 placement 순서로만 GPU2에서
실행된다. NeuralRx outcome은 branch coverage를 위해 protocol에 사전 고정해 주입했다.

## 2. 사전 고정 branch

| Branch | 제출/수락/거절 | Success | 예상 Qwen | 예상 recovery |
|---|---:|---:|---:|---:|
| all-fail | 4/4/0 | 0 | 0 | 4 |
| conditional-open | 5/4/1 | 2 | 1 | 2 |
| all-success | 4/4/0 | 4 | 1 | 0 |
| overload | 5/4/1 | 0 | 0 | 4 |

C154는 위 순서로, C155는 `overload → all-success → conditional-open → all-fail` 역순으로
실행했다. 두 protocol의 source hash와 mode는 동일하고 seed 집합은 완전히 분리됐다.
C155 protocol은 C154 node를 사전에 제외했다.

## 3. 결과

### 3.1 Arm별 결과

| Node | Branch | Qwen host | Recovery/정답 | 최장 release→home commit | D155 miss |
|---|---|---:|---:|---:|---:|
| nid001032 | all-fail | 없음 | 4/4 | 125.330 ms | 0 |
| nid001032 | conditional-open | 23.354 ms | 2/2 | 101.496 ms | 0 |
| nid001032 | all-success | 23.522 ms | 0/0 | 해당 없음 | 0 |
| nid001032 | overload | 없음 | 4/4 | 127.278 ms | 0 |
| nid001025 | overload | 없음 | 4/4 | 132.129 ms | 0 |
| nid001025 | all-success | 23.385 ms | 0/0 | 해당 없음 | 0 |
| nid001025 | conditional-open | 23.212 ms | 2/2 | 100.798 ms | 0 |
| nid001025 | all-fail | 없음 | 4/4 | 123.621 ms | 0 |

모든 Qwen 실행은 35 ms lease와 45--80 ms interval 안에서 physical fence를 반환했다.
All-fail과 overload에서는 `45+4×25+2=147 ms` certificate를 유지하느라 Qwen을 launch하지
않았다. Conditional-open은 두 success 뒤에만 Qwen을 실행하고 남은 두 recovery를
certificate 순서로 처리했다. All-success는 네 debt를 모두 해제하고 recovery를 launch하지
않았다.

### 3.2 결합 결과

| 항목 | 두 node 합계 |
|---|---:|
| Controlled arms | 8 |
| Submitted / accepted / globally rejected debt | 36 / 32 / 4 |
| Success transition | 12 |
| Qwen physical fence | 4 |
| Shared cuPHY/P2P recovery | 20 |
| Correct home commit | 20/20 |
| Deadline miss | 0 |
| Source/mode/order/seed/node gate | 전부 통과 |

Combined analyzer의 8개 gate와 각 arm gate는 모두 통과했다. 81개 transitive artifact의
manifest를 byte-stable하게 생성했다.

## 4. 무엇이 입증됐는가

이 결과는 다음 문장을 지지한다.

> Frozen synthetic P180/D155 mode의 두 A100 node에서 global certificate는 여러 home의
> controlled NeuralRx outcome에 따라 fifth debt를 launch 전에 거절하고, 조건부로만 Qwen
> lease를 발급·fence-retire하며, 남은 recovery를 certificate 순서의 실제 cuPHY/P2P 실행과
> home commit으로 완료했다.

따라서 V17의 **global model + physical shared recovery path + Qwen transaction**은 분리된
부품이 아니라 한 실행 계약으로 두 node에서 재현됐다. Local-only 합성의 false-safe를
막는 correctness contribution은 C153보다 강한 prospective holdout 근거를 얻었다.

## 5. 주장하지 않는 것

- 실제 TensorRT NeuralRx output이 직접 success/fail event를 만든 통합 경로
- GPU hang, shared-worker crash, stale/duplicate reply와 broker reply loss의 V17 fault matrix
- Hard WCET 또는 production DU `d_MAC`
- Static/global work-conserving baseline 대비 처리량 우위
- 다른 GPU family 또는 cross-node fabric 일반화
- Full-GPU-blackout Qwen lease와 recovery kernel의 동시 실행

이 mode는 Qwen과 recovery가 같은 MPS GPU에 resident하지만 certificate가 두 실행을 시간상
겹치지 않게 한다. 기존 C68/C70의 kernel overlap 증거를 이 shared mode의 overlap으로
재해석하지 않는다.

후속 [C156 Nsight gate](SOFTWALL_CONFIRM156_GPU_TIMELINE_RESULT_KO.md)는 fixed lease의
launch-control 공백을 반증한 뒤 V17.1 dynamic lease에서 kernel overlap0 ns와 phase/order를
통과했다. 당시 다음 gate는 V17.1 독립 node와 실제 NeuralRx result 연결이었다.

그 후 [C157A/C157B](SOFTWALL_CONFIRM157_ACTUAL_NRX_RESULT_KO.md)가 실제 TensorRT NRx
CRC-driven transition을 두 node에서 통과했다. 위 목록은 C154/C155 자체의 주장 범위이며,
C157 이후 남은 핵심 공백은 integrated fault matrix와 production timing이다.

## 6. Artifact

- [Combined result](../../results/softwall_multigpu/confirm154_155_controlled_integrated_holdout.json)
- [C154 result](../../results/softwall_multigpu/confirm154_integrated_holdout_job58853453_result.json)
- [C154 protocol](../../results/softwall_multigpu/confirm154_integrated_holdout_job58853453_protocol.json)
- [C155 result](../../results/softwall_multigpu/confirm155_integrated_holdout_job_result.json)
- [C155 protocol](../../results/softwall_multigpu/confirm155_integrated_holdout_job_protocol.json)
- [81-file manifest](../../results/softwall_multigpu/confirm154_155_integrated_holdout_manifest.json)
