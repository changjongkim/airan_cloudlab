# Confirm116: local 1개 + remote 2개 heterogeneous endpoint pool

**상태:** 2026-09-24 formal 두 seed 완료  
**환경:** 4×A100 NVLink node, GPU0/1/2 사용, MIG OFF, MPS ON  
**판정:** G4 three-endpoint lifecycle PASS

## 변경점

기존 controller의 두 endpoint 하드코딩을 제거하고 endpoint 수를 runtime 인자로 만들었다.
Confirm116은 다음 pool을 사용한다.

```text
nrx0: GPU0 same-device CUDA IPC, MPS cap40
nrx1: GPU0↔GPU1 NVLink P2P, GPU1 TensorRT NeuralRx, cap40
nrx2: GPU0↔GPU2 NVLink P2P, GPU2 TensorRT NeuralRx, cap40
ai  : GPU0 Qwen2.5-1.5B, cap20
```

세 NRx endpoint는 같은 `DartRuntime`의 endpoint/ring generation과 같은 4셀 all-fail
recovery calendar를 사용한다. Conventional recovery와 최종 LDPC/CRC/commit은 GPU0에
남는다.

## Formal 결과

각 arm은 340 release, 1,360 TB, `P180/D155`, NRx45, conventional12, guard2 ms와
혼합 single/correlated failure를 사용했다.

| 결과 | seed1 | seed2 |
|---|---:|---:|
| endpoint admission `nrx0/nrx1/nrx2` | 331 / 263 / 126 | 320 / 265 / 134 |
| worker 완료 수, warmup 포함 | 373 / 284 / 147 | 362 / 286 / 155 |
| NRx / conventional commit | 541 / 819 | 547 / 813 |
| atomic retime+AI lease | 90 | 98 |
| timely Qwen units / token value | 889 / 225,372 | 891 / 225,889 |
| 전체 NRx response p99 / max | 10.836 / 12.580 ms | 10.306 / 12.640 ms |
| deadline/bound/guard/fault/credit 위반 | 0 | 0 |

Warmup을 제거한 remote worker path는 다음과 같다.

| Endpoint | seed1 p99 / max | seed2 p99 / max |
|---|---:|---:|
| GPU1 `nrx1` | 2.674 / 2.887 ms | 2.698 / 2.750 ms |
| GPU2 `nrx2` | 2.791 / 2.946 ms | 3.430 / 5.481 ms |

Controller의 endpoint별 timed dispatch, worker의 warmup 수와 전체 completed count가 두
arm의 세 endpoint 모두 정확히 일치했다. 모든 source hash와 artifact hash도 일치했다.

## 의미와 한계

이 결과로 SoftWall의 transport-independent endpoint contract가 두 endpoint에 고정된
구현 우연이라는 우려는 줄었다. 하나의 home recovery calendar가 서로 다른 GPU의 두 P2P
ring과 local IPC ring을 같은 generation-safe lifecycle로 관리한다.

다음은 여전히 주장하지 않는다.

- endpoint가 늘어날수록 처리량이 선형 증가한다는 주장
- 3-endpoint가 2-endpoint보다 우월하다는 paired 성능 주장
- GPU 1/2/4와 cell 4/8/12의 일반 envelope
- independent hardware, production `d_MAC`, WCET

## 권위 artifact

- [Aggregate gate](../../results/softwall_multigpu/confirm116_three_endpoint_gates_job58815435.json)
- [Artifact manifest](../../results/softwall_multigpu/confirm116_artifact_manifest.json)
- [Seed1 arm](../../results/softwall_multigpu/g4_three_endpoint_s1_job58815435_result.json)
- [Seed2 arm](../../results/softwall_multigpu/g4_three_endpoint_s2_job58815435_result.json)
- [Analyzer development failure](../../results/softwall_multigpu/confirm116_analyzer_development_failure_job58815435.json)

