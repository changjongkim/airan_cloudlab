# ISCA full-day campaign — 2026-08-13

## 왜 새 캠페인인가

`day1_20260813T0523Z`는 기능 bring-up과 가설 확인을 위한 약 50분 quick
campaign이다. 이 결과는 정식 반복 수와 trace 길이를 충족하지 않으므로 논문용 full
evaluation으로 취급하지 않는다. 실행 상태는
`ISCA_EXECUTION_LEDGER_20260813.csv`에 고정했다.

정식 결과 루트는 다음과 같다.

```text
/mydata/results/isca_v2/full_day_20260813T0710Z
```

## 자동 실행 DAG

```text
02 NRx stack
  5 trials × 10,000 slots ×
  wrapper/raw/caller-input/caller-I/O/CUDA-Graph
        |
        v
02b Nsys + NCU
        |
        v
03 workload qualification
  Qwen: full/3g × prefill/decode × duty 50/90% × 3 × 90s
  training: full/3g × microbatch 1/4 × duty 50/90% × 3 × 90s
        |
        v
04 NRx capacity
  4g/3g/full × replica 1/2/4/8 × 5 trials
  each: saturation 60s + load 50/85/95/105% each 60s
        |
        v
05 fair five-way compute
  MPS/MIG/MIG+MPS/P2P/GDR × load 50/85/95/105%
  × 5 trials × 60s; common absolute arrival trace and timing boundary
       / \
      v   v
core analysis    06 DART scheme gates
                 actual NRx routing 5 trials × 4 policies × 5 rates × 60s
                 5 × 10,000 fault tests
                 4 guard levels × 5 × 250s control replay
                        |
                        v
                 DART gate analysis
```

정상 경로는 약 10시간 이상이다. Profiler replay와 saturated drain 시간에 따라 더
길어질 수 있다. 각 stage는 dependency의 `COMPLETE`를 기다리며, dependency의
`FAILED`가 관찰되면 실행하지 않는다.

## 공정성 경계

Five-way 실험은 모든 접근법에서 `L1 front 시작 → NRx → L1 back/LDPC/CRC 완료`를
측정한다. 동일 4g replica-1 capacity로부터 하나의 absolute arrival trace를 만들고
모든 접근법에 그대로 replay한다.

- MPS: full A100의 두 MPS process, CUDA IPC GPU payload
- MIG: 4g의 in-process caller-owned binding
- MIG+MPS: 4g의 두 MPS process, CUDA IPC GPU payload
- P2P: L1 4g → NRx 3g, CUDA P2P
- GDR: L1 4g → NRx 3g, GPUDirect RDMA loopback

CUDA IPC와 GDR/P2P 모두 payload가 host DRAM을 경유하지 않는다. CUDA IPC와 GDR의
host control marker는 8 bytes이다. MPS의 full-GPU resource budget과 MIG의 4g budget은
다르므로 결과에는 absolute resource budget도 함께 표시해야 한다.

## Workload realism 범위

- Qwen은 cached pretrained Qwen2.5-7B와 Stanford Alpaca의 실제 text를 사용한다.
  Alpaca는 public instruction dataset이지 production request trace는 아니다.
- Training은 pretrained torchvision ResNet-50과 CIFAR-10 실제 image를 사용한다.
- Video와 speech는 licensed input/model pipeline이 아직 없으므로 synthetic 결과로
  대체하지 않고 `BLOCKED_DATA` 상태를 유지한다.

## DART는 하나의 framework인가

그렇다. DART-Rx의 request path는 하나이다.

```text
slot descriptor
  → DART-P predicted-finish admission
  → DART-R endpoint/buffer atomic reservation
  → local or remote NRx execution
  → DART-C epoch-checked single-winner commit
  → DART-J latest-safe conventional fallback
```

`DART-L`은 같은 deadline/slack table을 이용해 NRx pool의 빈 구간에 background unit을
grant-ahead lease한다. 즉 별도 data pipeline이 아니라 동일 control plane의 reclaim
기능이다. `DART-Q`는 이 control decision을 device-side queue/doorbell로 가속하는 향후
hardware mechanism이며, 현재 software DART의 필수 구성요소는 아니다.

현재 자동 캠페인의 `06` stage는 actual TensorRT routing/queue와 software
reservation/commit/fallback/lease correctness를 정식 길이로 검증한다. 그러나 actual
NRx routing gate와 P2P/GDR data-path gate가 아직 하나의 end-to-end binary로 통합된
것은 아니다. 분석기는 이를 명시하고 `integrated_end_to_end_dart=false`로 기록한다.

## 완료 판정

- `FULL_CORE_COMPLETE`: NRx/workload/capacity/five-way raw artifact strict validation 통과
- `DART_MECHANISM_GATES_COMPLETE`: routing/fault/control mechanism validation 통과
- 위 두 marker가 있어도 integrated end-to-end DART data plane은 별도 gate가 통과하기
  전에는 완료라고 쓰지 않는다.
