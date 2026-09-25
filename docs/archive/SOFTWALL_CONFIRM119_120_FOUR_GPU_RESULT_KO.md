> **보관 사유 (2026-09-25):** 이 문서는 실험 이력으로 보존한다. 최신 스킴과 claim boundary는 `docs/current`의 원고·모델·production gate 문서가 권위 기준이며, 과거 GPU0 전용, MIG, CloudLab 또는 4-GPU pool 가정은 현재 논문의 근거가 아니다.

# Confirm119/120 4-GPU·4-endpoint 결과

**실행일:** 2026-09-24  
**환경:** job `58815435`, `nid001016`, A100-SXM4 40GB 4장, MIG OFF, MPS ON  
**판정:** **4-GPU integration/system safety PASS; 세부 component-vector qualification FAIL**

## 1. 구성

```text
GPU0: 4-cell Aerial/cupHY + local CUDA-IPC NRx + Qwen2.5-1.5B
GPU1: remote P2P NRx endpoint 1
GPU2: remote P2P NRx endpoint 2
GPU3: remote P2P NRx endpoint 3
```

각 셀은 네 endpoint 중 하나에 request별로 수락된다. Conventional recovery와 최종
LDPC/CRC/commit은 GPU0에 남는다. Controller는 네 endpoint/ring generation, GPU0의
all-fail recovery calendar와 Qwen lease를 같은 lifecycle에서 관리한다.

## 2. 먼저 발견한 memory/lifecycle 경계

첫 C119 canary는 timed traffic 전에 네 번째 `PairedDualReceiver`의 `LdpcDecoder` 생성에서
GPU0 OOM으로 실패했다. C116의 3-GPU 구성과 비교해 추가된 것은 GPU3 remote worker와 그
worker의 GPU0 source context/IPC mapping이다. 이 실패는 추가 endpoint도 home GPU memory를
소비한다는 해석과 일치한다.

NVIDIA가 권고하는 `CUDA_MODULE_LOADING=LAZY`를 mode에 명시한 새 canary는 40 release를
통과했다. Eager 실패는 삭제하지 않았다. 따라서 lazy/eager lifecycle은 서로 다른
자격화 mode다. 실패 cleanup에서 MPS 종료가 멈춘 문제도 worker TERM→bounded wait→KILL→wait
순서로 runner를 수정했다.

## 3. C119 formal 결과

새 source hash와 두 seed, component 후보를 실행 전에 고정했다. 두 340-release arm 모두
다음을 통과했다.

- 네 endpoint 모두 실제 사용
- controller admission과 네 worker completion 수 일치
- atomic retime+AI lease 97/85회
- RAN deadline, NRx45, conventional12, Qwen class, guard, fault, residual credit 위반 0
- end-to-end NRx 최대 12.256/13.422 ms

그러나 seed2의 GPU2→GPU0 backward P2P 한 건이 `109.824 us`로 100 us 후보를 넘었다.
따라서 C119의 전체 component gate는 실패다. System safety 통과와 component 자격 실패를
분리해 기록한다.

## 4. C120 재자격 결과

C119를 보존하고 backward 후보만 125 us로 바꾼 뒤 새로운 두 seed를 고정했다. 두 arm은
다시 system safety, 네 endpoint 사용/count, 125 us backward를 통과했고 atomic exchange는
85/95회였다. 그러나 seed3의 GPU0 back이 한 건 `4.724 ms`로 3 ms 후보를 넘었다.

해당 요청의 NRx dispatch→observe 구간은 GPU0의 256-token Qwen 실행
(`28.000 ms GPU`)과 겹쳤다. 이는 시간상 동시 실행 증거이며 단일 표본으로 인과를
확정하지 않는다. 요청의 end-to-end NRx는 13.695 ms로 기존 25/45 ms 계약 안이었고
deadline miss도 없었다.

## 5. 네 formal arm의 정확한 해석

| 항목 | C119 seed1 | C119 seed2 | C120 seed3 | C120 seed4 |
|---|---:|---:|---:|---:|
| Atomic exchange | 97 | 85 | 85 | 95 |
| Timely token value | 225,486 | 225,147 | 225,210 | 226,143 |
| End-to-end NRx 최대 | 12.256 ms | 13.422 ms | 16.757 ms | 11.717 ms |
| System safety 위반 | 0 | 0 | 0 | 0 |
| Fine component gate | backward 100 us PASS | **109.824 us FAIL** | **back 4.724 ms FAIL** | PASS |

네 실행 모두 45 ms runtime 계약과 system safety를 통과했으므로 4-GPU transport-independent
integration의 유한 표본 증거는 있다. 반면 3-GPU C118의 세부 component vector를 4-GPU
mode에 그대로 이식하는 주장은 두 번 기각됐다.

## 6. 설계 결정

Runtime admission의 안전 계약은 `dispatch 전 front 시작 → physical result 관측/commit`의
**end-to-end path bound**로 유지한다. Component 측정은 다음 용도로 남긴다.

- topology/co-run 변화의 원인 진단
- transport 선택 비용과 guard 구성
- path bound가 깨졌을 때 어느 자원이 원인인지 분해

개별 component의 표본 최대를 더해 hard WCET라고 주장하지 않는다. Fine component
zero-violation vector가 실패해도 end-to-end certificate가 자동으로 실패하는 것은 아니며,
반대로 end-to-end 표본 통과도 WCET는 아니다. 각 mode의 runtime path bound는 별도
prospective qualification을 거쳐야 한다.

## 7. 논문에 주는 의미

4-GPU 확장은 단순히 endpoint를 하나 더 띄우는 문제가 아니었다.

1. remote worker가 home GPU context/mapping을 가져 memory feasibility를 바꿨다.
2. lazy module loading 여부가 pretraffic OOM/PASS를 갈랐다.
3. 네 번째 endpoint 뒤 P2P와 home back의 tail이 기존 3-GPU component 후보를 깼다.
4. 그럼에도 all-fail recovery와 physical credit lifecycle은 네 endpoint에서 유지됐다.

이 결과는 SoftWall의 차별점인 **mode-qualified recovery/transport contract**가 실제 scale-out
경계에서 왜 필요한지 보여준다. 처리량 우월성이나 8/12-cell scale을 입증한 결과는 아니다.

## 8. 권위 artifact

- [C119 eager canary OOM](../../results/softwall_multigpu/confirm119_four_gpu_canary_failure_job58815435.json)
- [C119 사전 protocol](../../results/softwall_multigpu/confirm119_four_gpu_protocol.json)
- [C119 aggregate](../../results/softwall_multigpu/confirm119_four_gpu_job58815435.json)
- [C120 사전 protocol](../../results/softwall_multigpu/confirm120_four_gpu_requalification_protocol.json)
- [C120 aggregate](../../results/softwall_multigpu/confirm120_four_gpu_requalification_job58815435.json)
- [C119/C120 artifact manifest](../../results/softwall_multigpu/confirm119_120_artifact_manifest.json)
