# Fixed-MIG multi-endpoint GPUDirect NRx pool · 실행 계획

Date: 2026-08-14  
Status: process-per-endpoint implementation deployed; autonomous hardware campaign in progress

## 1. 이번 실행이 답할 질문

> Compute-only replay에서 관측한 fixed-MIG fragmentation이 실제 full-size
> GPUDirect request/result와 resident TensorRT NRx를 통과해도 유지되는가? 그리고
> runtime tail feedback이 static/fixed-profile routing보다 deadline-safe한가?

이번 단계는 기존 MIG/MPS/P2P/GDR placement matrix를 반복하지 않는다. 이미 완료된
characterization에서 확인된 다음 공백만 닫는다.

- compute-only multi-cell queue와 actual GDR data plane의 통합
- 한 source-side endpoint agent process가 각각의 독립 3g MIG endpoint를 구동하는 실제
  RC QP pool. 초기 thread 구현은 Python GIL로 직렬화되어 diagnostic으로 보존했고,
  process/CUDA-context/QP-per-endpoint 구조로 교체했다.
- rare service outlier 동안의 over-admission 방지
- 동일 trace에서 static placement와 tail-aware admission의 paired 비교

## 2. 고정 hardware topology

```text
GPU0 4g: source/protected-L1-side GPU MR
GPU0 3g: resident NRx endpoint 0
GPU1 3g: resident NRx endpoint 1
GPU2 3g: resident NRx endpoint 2
GPU1/2 4g, GPU3 full: 이번 gate에서는 사용하지 않음
ConnectX-6 Dx: RoCE v2 GID 3, physical loopback
```

GPU1/2의 MIG enable에 reboot가 필요할 수 있다. autonomous systemd service가 reboot
후 같은 result root에서 재개한다. 측정 중 topology는 바꾸지 않으며 완료 후 GPU1/2를
full-GPU mode로 복구한다.

## 3. 실제 data path

각 endpoint는 독립 RC QP 두 개를 사용한다.

```text
source GPU MR
  1,415,232 B payload -> request descriptor -> doorbell
      -> resident direct TensorRT/CUDA Graph NRx
  314,496 B LLR <- completion descriptor <- doorbell
```

- bulk tensor: GPU MR, CPU staging 없음
- descriptor/control: host MR
- request와 completion: payload -> descriptor -> sequence 순서
- endpoint별 model/context는 resident
- completion epoch/endpoint/status를 검증
- 이 gate에서는 conventional receiver와 radio utility를 실행하지 않음

## 4. 정책

| policy | 의미 |
|---|---|
| `static_one` | 모든 NRx request를 endpoint 0에 고정 |
| `static_cell` | cell ID를 고정 endpoint에 매핑 |
| `round_robin` | queue/deadline을 보지 않는 순환 배치 |
| `shortest_queue` | pending request 수가 가장 작은 endpoint |
| `predicted_finish` | calibration에서 얻은 고정 p99 bound 기반 admission |
| `tail_aware` | online tail bound, late-debt correction, in-flight circuit breaker |

`tail_aware`는 512-sample window의 p99.5에 1.10 guard를 적용한다. 현재 in-flight
exchange가 bound의 1.25배를 넘으면 completion feedback이 올 때까지 해당 endpoint에 새
request를 보내지 않는다. 실제 completion이 예측보다 늦으면 남은 predicted queue tail에
그 debt를 더한다.

## 5. 실행 단계

### Stage 01 · smoke

- 1 cell, 1 ms, 0.25 s
- `tail_aware`
- 3 endpoints/QP/GPU MR/descriptor/termination correctness 확인

Pass:

- source/worker exit 0
- worker 3개 모두 ready/shutdown
- stale/sequence/checksum/worker error 0
- result/raw artifact와 status conservation 통과

### Stage 02 · physical replica sweep

- physical endpoints: `1/2/3`
- trace: single-cell 1 ms, 2-cell synchronized 1 ms, 4-cell selective-bursty 10%
- policy: round-robin, predicted-finish, tail-aware
- 목적: actual full-size GDR에서 aggregate service capacity와 최소 stable replica 수 측정

### Stage 03 · representative matrix

Trace 6개:

- single 1 ms / 0.5 ms
- 2-cell synchronized/staggered 1 ms
- 4-cell bursty 10%, 1 ms / 0.5 ms

정책 6개를 모두 실행한다. 이 단계에서 GDR 경로의 queue cliff, outlier, 정책 차이를
빠르게 확인하고 실패 시 수정한다.

### Stage 04 · full matrix

- 기존 87개 trace 전부
- `static_one`, `static_cell`, `predicted_finish`, `tail_aware`
- 기존 compute-only trace와 동일한 arrival digest 사용
- 완료된 run은 재시작 시 건너뜀

전체 348 runs이며 각 run은 독립 worker/QP epoch를 사용한다.

### 2026-08-14 15:10 KST 실행 snapshot

- smoke: `1/1` 완료
- physical replica sweep: `27/27` 완료
- representative matrix: `36/36` 완료
- full matrix: `42/348` 완료
- active service: `airan-gdr-pool-autonomous.service`, enabled/activating
- 현재 campaign failure: 0

## 6. 주요 metric

- timely NRx completion ratio
- rejected-to-baseline/remote-expired/late/queue-overflow
- sojourn p99/p99.9
- actual GDR exchange p99
- worker TensorRT service p99
- endpoint별 completed request와 max pending
- service bound 변화와 circuit-break 횟수
- scheduler lateness

## 7. 결과 판정

### GO

- static miss와 다른 endpoint의 사용 가능 capacity가 actual GDR에서도 공존
- `tail_aware`가 static/fixed-prediction 대비 no-timely ratio와 queue tail을 감소
- exchange tail이 전체 overload tail보다 충분히 작음
- endpoint 3개가 동시에 resident/healthy하게 동작

### REDIRECT

- QP/host polling/visibility flush가 service capacity를 지배
- 3개 endpoint의 aggregate capacity가 단일 endpoint 대비 증가하지 않음
- tail feedback이 outlier over-admission을 막지 못함

### KILL

- 실제 GDR payload를 포함하면 eligible idle capacity가 사라짐
- remote endpoint 사용이 static-local보다 deadline 결과를 일관되게 악화
- GPUDirect correctness 또는 visibility를 반복 실행에서 보장하지 못함

## 8. 이번 단계가 끝나도 주장하지 않을 것

- production PHY deadline 충족
- BLER/CRC/radio utility 개선
- full cuPHY end-to-end elastic pool 완성
- hardware queue 또는 새로운 ISA가 필요하다는 주장

다음 gate는 이 결과가 GO일 때만 진행한다: single-endpoint actual radio vertical slice를
`EndpointSession[]`로 확장하고, actual conventional baseline, radio-utility admission,
LDPC/CRC, epoch/expiry single commit을 같은 multi-endpoint pool에 통합한다.
