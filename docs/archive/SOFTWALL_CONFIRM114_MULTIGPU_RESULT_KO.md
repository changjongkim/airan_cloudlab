> **보관 사유 (2026-09-25):** 이 문서는 실험 이력으로 보존한다. 최신 스킴과 claim boundary는 `docs/current`의 원고·모델·production gate 문서가 권위 기준이며, 과거 GPU0 전용, MIG, CloudLab 또는 4-GPU pool 가정은 현재 논문의 근거가 아니다.

# Confirm114: 멀티 GPU transport-independent SoftWall 결과

**상태:** 2026-09-24 formal gate 완료  
**실행:** Slurm job `58815435`, node `nid001016`  
**판정:** G0/G1a/G1b/G2a/G3 PASS; 멀티 GPU 성능 우월성은 주장하지 않음

## 무엇을 검증했는가

Confirm114는 P2P copy 자체의 새로움을 주장하는 실험이 아니다. SoftWall의 안전 계약이
same-device CUDA IPC endpoint에 묶이지 않고, 원격 GPU의 NeuralRx endpoint까지 같은
recovery lifecycle로 다룰 수 있는지를 검증한다.

최종 경로는 다음과 같다.

```text
GPU0 Aerial front
  ├─ local endpoint: GPU0 CUDA IPC -> GPU0 TensorRT NeuralRx
  └─ remote endpoint: GPU0 CUDA IPC buffer
                      -> NVLink P2P GPU0→GPU1
                      -> GPU1 TensorRT NeuralRx
                      -> NVLink P2P GPU1→GPU0 LLR
GPU0 de-rate-match/LDPC/CRC/single commit

GPU0에는 동시에 MPS cap20 Qwen2.5-1.5B가 실행된다.
모든 미해결 NRx 요청의 conventional recovery credit은 GPU0에 유지된다.
```

Remote worker가 timeout 또는 오류를 내도 이미 시작한 P2P copy와 NRx kernel의 credit을
논리적으로 먼저 반환하지 않는다. Importer가 CUDA IPC handle을 닫은 뒤 ACK하고, exporter는
그 ACK 뒤에만 allocation을 해제한다.

## G0 — 물리 topology

- NVIDIA A100-SXM4-40GB 4장
- 모든 GPU pair가 `NV4`
- 모든 방향에서 `cudaDeviceCanAccessPeer = 1`

이는 가능성 검사다. 이 결과만으로 cross-process correctness나 deadline을 주장하지 않는다.

## G1a — cross-process P2P transport

별도 process가 GPU0의 실제 SoftWall 크기 buffer를 CUDA IPC로 열고 다음을 10,000회 수행했다.

- forward payload: 1,415,232 byte
- backward payload: 314,496 byte
- GPU0→GPU1 `cudaMemcpyPeerAsync`
- GPU-side pattern 검사와 deterministic reply 생성
- GPU1→GPU0 `cudaMemcpyPeerAsync`
- sequence, payload integrity, stale completion, lifecycle 검사

| 지표 | p50 | p99 | max |
|---|---:|---:|---:|
| steady round trip, 9,980회 | 213.122 us | 248.530 us | 1,307.105 us |
| forward GPU copy | 19.040 us | 44.864 us | 72.288 us |
| backward GPU copy | 21.504 us | 40.224 us | 48.256 us |

오류는 0이었다. 첫 cold round trip은 85.492 ms였으므로 warm mode와 cold lifecycle을 같은
mode로 취급하면 안 된다.

## G1b — pinned-host staging 대조

동일 payload와 10,000회 조건에서 NVLink P2P를 pinned-host staging과 비교했다. 두 transport
모두 integrity·sequence·종료 lifecycle 오류 0으로 통과했다.

| Warm 9,980회 round trip | p50 | p99 | max |
|---|---:|---:|---:|
| P2P | 213.122 us | 248.530 us | 1,307.105 us |
| Pinned host staging | 349.265 us | 377.730 us | 1,647.273 us |

Host staging/P2P 비율은 p50 1.639×, p99 1.520×였고 P2P가 각각 136.143/129.200 us를
줄였다. 이는 한 NVLink node의 warm transport 비용 분리이며 end-to-end NeuralRx나 Qwen
처리량 효과가 아니다.

## G2a — 실제 원격 NeuralRx 경로

G1의 pattern kernel을 실제 GPU1 TensorRT DirectNrx CUDA Graph로 바꾸고, GPU0 Aerial의
front/back과 연결했다. MPS를 끈 isolated mode에서 warmup 20회 뒤 1,000회를 실행했다.

| 지표 | 결과 |
|---|---:|
| correct | 1,000 / 1,000 |
| timeout/deadline miss | 0 / 0 |
| controller response p50 / p99 / max | 3.451 / 3.733 / 5.349 ms |
| forward P2P p50 / p99 / max | 65.376 / 70.720 / 85.920 us |
| NeuralRx GPU p50 / p99 / max | 0.845 / 0.849 / 0.863 ms |
| backward P2P p50 / p99 / max | 20.448 / 24.384 / 64.672 us |

8 ms 사전 deadline을 모두 만족했다. 이는 clean valid PUSCH의 isolated transport/path 자격이며,
co-run service bound나 실제 DU deadline의 증명은 아니다.

## G3 — 4셀 SoftWall/Qwen/fault 통합

두 독립 seed에서 각각 340 release, 1,360 TB를 실행했다. GPU0은 4셀 Aerial,
local NeuralRx endpoint와 Qwen을 실행했고 GPU1은 remote NeuralRx endpoint를 실행했다.
MPS cap은 local NRx 40, remote NRx 40, Qwen 20이었다. Mode는 `P180/D155`, NRx 45 ms,
conventional 12 ms, guard 2 ms이며 5 release마다 single/correlated NRx failure를 섞었다.

| 결과 | seed 1 | seed 2 |
|---|---:|---:|
| radio records | 1,360 | 1,360 |
| correct cells | 653 | 649 |
| NRx / conventional commits | 443 / 917 | 442 / 918 |
| atomic retime + AI lease | 85 | 85 |
| timely Qwen units | 886 | 891 |
| timely token value | 224,691 | 225,817 |
| NRx response p99 / max | 9.231 / 10.235 ms | 9.185 / 17.518 ms |
| timed remote worker p99 / max | 2.320 / 2.348 ms | 2.119 / 2.605 ms |
| deadline/bound/guard/fault/credit 위반 | 0 | 0 |

Remote worker의 raw maximum 67.250/62.071 ms에는 controller timed epoch 전에 실행한 42개
warmup 요청이 포함된다. Timed remote 요청은 258/266개였고 위 표는 이 요청만 분리한 값이다.
Cold 비용을 숨긴 것이 아니라 lifecycle mode를 분리한 것이다.

## 통과한 주장과 통과하지 않은 주장

이 결과가 직접 지지하는 문장은 다음이다.

> 한 4×A100 NVLink node의 자격화된 warm mode에서 SoftWall은 local CUDA-IPC endpoint와
> remote P2P endpoint를 같은 4셀 all-fail recovery contract에 연결하고, GPU0의 MPS Qwen과
> 혼합 NRx failure가 있는 두 독립 실행에서 관측된 deadline·bound·guard·credit 위반 없이
> 동작했다.

다음은 아직 지지하지 않는다.

- 표본 최대를 WCET로 해석하는 주장
- production DU/FAPI `d_MAC` 보장
- P2P 사용 자체의 novelty
- multi-GPU가 single-GPU보다 빠르다는 주장
- SoftWall이 같은 2-GPU 예산의 strong safe baseline보다 처리량이 높다는 주장
- 3개 이상의 heterogeneous endpoint와 remote-GPU Qwen lease

## 노벨티에 주는 의미

C114가 강화한 것은 P2P 기능이 아니라 **계약의 transport independence**다. 기존 단일 GPU
주장은 이제 다음과 같이 확장된다.

> Optional per-TB neural processing이 만든 multi-cell all-fail recovery debt를 home GPU에
> executable schedule로 유지하고, local/remote endpoint·transport/ring credit과 bounded
> external-AI lease를 physical completion까지 하나의 reservation lifecycle로 관리한다.

MPS, CUDA IPC, P2P, fallback, deadline-aware routing은 각각 기존 기술이다. 차별점은 이들을
나열한 것이 아니라, NRx 결과가 미래 mandatory recovery 집합을 바꾸는 AI-RAN 고유 dependency를
all-fail certificate와 원자 credit lifecycle로 만든 데 있다. C114는 그 계약이 실제 remote
NeuralRx data path에서도 유지된다는 첫 물리 증거다.

## 남은 우려와 다음 gate

1. **같은 예산의 성능 기준선:** static safe, safe work-conserving, SoftWall을 같은 2-GPU
   placement와 같은 trace에서 ABBA로 비교해야 한다.
2. **Co-run bound 분해:** isolated G2와 통합 G3 사이에서 copy engine, source HBM, remote NRx,
   host polling 비용을 각각 사전 bound로 자격화해야 한다.
3. **Scale:** controller의 2-endpoint 가정을 제거한 뒤 local 1개와 remote 2개 이상에서
   generation, ring, recovery credit 누수를 검사해야 한다.
4. **Envelope:** GPU 1/2/4, cell 4/8/12, memory, AI class의 경계를 사전에 예측하고 false-safe를
   최우선 오류로 측정해야 한다.
5. **실제 시간 계약:** 실제 DU의 IQ-ready부터 FAPI/MAC consumption까지 같은 clock으로
   측정한 `d_MAC`가 없으므로 production hard-real-time 주장은 보류한다.

## 권위 artifact

- [C114 aggregate gate](../../results/softwall_multigpu/confirm114_multigpu_gates_job58815435.json)
- [C114 artifact manifest](../../results/softwall_multigpu/confirm114_artifact_manifest.json)
- [G0 topology gate](../../results/softwall_multigpu/GATE0_TOPOLOGY_KO.md)
- [G1a transport result](../../results/softwall_multigpu/g1a_formal_job58815435_result.json)
- [G1b host-staging result](../../results/softwall_multigpu/g1b_host_formal_job58815435_result.json)
- [P2P 대 host-staging 비교](../../results/softwall_multigpu/g1_transport_comparison_job58815435.json)
- [G2a actual NRx result](../../results/softwall_multigpu/g2a_formal_job58815435_result.json)
- [G3 seed 1](../../results/softwall_multigpu/g3_formal_s1_job58815435_result.json)
- [G3 seed 2](../../results/softwall_multigpu/g3_formal_s2_job58815435_result.json)
