# Confirm117 멀티 GPU component-bound 결과

**실행일:** 2026-09-24  
**실행 환경:** job `58815435`, `nid001016`, A100-SXM4 40GB 3장 사용, MIG OFF, MPS ON  
**판정:** **전체 시스템 safety PASS / 사전 고정 2 ms home-component qualification FAIL**

## 1. 무엇을 검증했나

C117은 C116의 `GPU0 local NRx 1개 + GPU1/GPU2 remote NRx 2개 + GPU0 Qwen` 경로에서
45 ms 전체 NeuralRx 계약을 더 작은 component 계약으로 분해했다. 실행 전 다음 후보를
고정하고, 새 독립 seed 두 개 모두에서 위반이 0이어야 통과하도록 정했다.

| Component | 사전 후보 bound |
|---|---:|
| GPU0 front | 2.0 ms |
| P2P forward | 250 us |
| remote NRx | 2.5 ms |
| P2P backward | 100 us |
| remote worker 전체 | 6.0 ms |
| GPU0 back | 2.0 ms |
| end-to-end NRx | 25.0 ms |

[사전 protocol](../../results/softwall_multigpu/confirm117_component_bounds_protocol.json)은
실행 source hash, 두 seed, 모든 후보 bound와 zero-violation 판정 규칙을 보존한다.

## 2. 판정 결과

두 arm 모두 radio deadline, 기존 NRx/conventional/AI bound, recovery guard, endpoint/ring
credit와 최종 credit 반환 gate를 통과했다. 그러나 component gate는 다음 두 건 때문에
실패했다.

| Arm | 실패 component | 표본 수 | p99 | 최대 | 2 ms 초과 |
|---|---|---:|---:|---:|---:|
| seed1 | GPU0 front | 720 | 0.857 ms | **2.393 ms** | **1** |
| seed2 | GPU0 back | 723 | 1.031 ms | **2.741 ms** | **1** |

나머지 component는 모두 사전 후보 안이었다.

| Component | 두 arm에서 관측한 최대 | 후보 | 판정 |
|---|---:|---:|---|
| end-to-end NRx | 13.291 / 11.941 ms | 25 ms | PASS |
| remote1 forward | 129.088 / 84.512 us | 250 us | PASS |
| remote2 forward | 110.048 / 79.648 us | 250 us | PASS |
| remote NRx | 최대 2.070 ms | 2.5 ms | PASS |
| remote backward | 최대 71.424 us | 100 us | PASS |
| remote worker 전체 | 최대 2.879 ms | 6 ms | PASS |

따라서 C117의 정확한 결론은 “멀티 GPU safety 실패”가 아니다. 이 warm mode의 전체
25/45 ms 계약과 모든 시스템 안전 gate는 통과했지만, **GPU0 front/back 각각을 2 ms로
자격화하려던 더 강한 계약은 거절됐다.**

## 3. 두 tail의 실행 문맥

원자료 timestamp로 다음 동시 실행 관계를 확인했다.

- seed1 front 2.393 ms 요청은 Qwen과 겹치지 않았다. 그 front가 실행되는 동안 GPU1과
  GPU2의 remote worker가 두 앞선 요청의 P2P/NRx 경로를 동시에 처리했다.
- seed2 back 2.741 ms 요청은 GPU0의 256-token Qwen 실행
  (`27.668 ms GPU`)과 시간상 겹쳤다.
- 두 arm 모두 Python GC event는 0건이었다.

이 timestamp 관계는 co-run class가 원인이라는 결정적 인과 증명은 아니다. 다만 같은
GPU0 component를 isolated 수치 하나로 모델링하면 false-safe가 생길 수 있고,
`transport traffic`, `external AI overlap`, `lifecycle`을 mode 변수에 포함해야 한다는
직접적인 반례다.

## 4. 모델에 주는 의미

기존의 단일 scalar `B_NRx`는 다음 vector를 보수적으로 합성한 축약 계약이어야 한다.

```text
B_endpoint(mode)
  = B_front(co-run class)
  + B_forward(transport/topology)
  + B_queue(endpoint)
  + B_NRx(endpoint/co-run class)
  + B_backward(transport/topology)
  + B_back(co-run class)
  + guard
```

C117은 원격 실행 비용 자체보다 home GPU의 front/back tail이 현재 후보 계약을 결정할 수
있음을 보였다. 멀티 GPU가 compute를 밖으로 옮겨도 source GPU의 memory/copy traffic과
Qwen 간섭은 사라지지 않는다. SoftWall의 endpoint 선택기는 앞으로 endpoint 평균시간이
아니라 **현재 co-run class에 자격화된 전체 path bound**를 사용해야 한다.

## 5. 다음 판정

C117은 실패로 고정한다. 관측 후 같은 결과 파일의 bound를 넓히지 않는다. C118은 이
실패를 근거로 GPU0 front/back 후보를 3 ms로 정하고, 다른 모든 후보를 유지한 채 새로운
두 seed를 실행하는 독립적인 prospective requalification이다. C118이 통과하더라도 이는
해당 warm co-run mode의 유한 표본 자격이며 WCET나 production DU 보장은 아니다.

## 6. 권위 artifact

- [C117 aggregate 판정](../../results/softwall_multigpu/confirm117_component_bounds_job58815435.json)
- [C117 사전 protocol](../../results/softwall_multigpu/confirm117_component_bounds_protocol.json)
- [C117 seed1 arm](../../results/softwall_multigpu/c117_components_s1_job58815435_result.json)
- [C117 seed2 arm](../../results/softwall_multigpu/c117_components_s2_job58815435_result.json)
- [C117/C118 artifact manifest](../../results/softwall_multigpu/confirm117_118_artifact_manifest.json)
