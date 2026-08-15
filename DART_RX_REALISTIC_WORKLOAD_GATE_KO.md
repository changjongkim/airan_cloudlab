# DART-Rx realistic workload gate

작성일: 2026-08-13

## 1. 결론

DART-Rx의 주 문제는 단일 NRx invocation을 빠르게 만드는 것이 아니다.

> 여러 cell/UE에서 발생하는 선택적 NRx demand가 고정 MIG partition의 순간
> service capacity를 넘는 동안 다른 isolated endpoint는 놀 수 있다. L1을
> 재시작하거나 MIG를 재구성하지 않고 resident NRx pool로 이 demand를 deadline
> 안에 흡수할 수 있는가?

Multi-cell PUSCH 자체는 가상의 사용례가 아니다. NVIDIA Aerial은 multi-cell
coordination과 multi-cell OTA Data Lake example을 제공하며, 공식 capacity 자료도
여러 cell을 한 accelerator에서 처리하는 구성을 평가한다.

- Aerial multi-cell OTA PUSCH example:
  <https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/content/notebooks/datalake_pusch_multicell.html>
- Aerial multi-cell capacity:
  <https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/release_notes/multicell_capacity.html>
- Aerial neural receiver validation:
  <https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/pyaerial/examples.html>

그러나 traffic timing이 현실적이라는 것과 radio input/utility가 현실적이라는 것은
다르다. 두 gate를 분리한다.

## 2. Gate A: traffic-realistic hardware execution

실제 TensorRT NRx engine과 독립 CUDA context/stream/buffer를 가진 resident
endpoint pool에서 다음 trace를 open-loop replay한다.

| class | 조건 | 목적 |
|---|---|---|
| single-cell | 1 cell, NRx 100%, slot 0.5/1.0 ms | 한 replica의 queue stability와 최소 capacity |
| multi-cell synchronized | 2/4/8 cells, 같은 slot boundary | gNB의 synchronized burst stress |
| multi-cell staggered | 2/4/8 cells, cell phase offset | 같은 평균 load에서 burst structure 영향 |
| selective IID | 4 cells, NRx 10/25/50/75/100% | 평균 NRx 사용률 sensitivity control |
| selective bursty | 4 cells, NRx 10/25/50/75/100%, two-state burst | fixed assignment fragmentation과 pooling 효과 |

Selective trace의 two-state model은 scheduling sensitivity input이지 measured radio
ground truth가 아니다. 모든 artifact에 다음을 강제로 기록한다.

```text
radio_ground_truth=false
radio_claim_allowed=false
```

동일 trace/hash를 다음 정책에 replay한다.

1. `static_one`: 모든 NRx를 한 endpoint에 고정
2. `static_cell`: cell을 endpoint에 고정 매핑
3. `round_robin`
4. `shortest_queue`
5. `predicted_finish`: provisional deadline을 만족하는 endpoint만 admit

측정 항목:

- arrival-to-completion p50/p95/p99/p99.9
- 실제 NRx service time
- deadline 안에 NRx 결과가 없는 비율
- fallback-required 비율
- per-endpoint queue와 assignment
- 동일 trace window의 idle endpoint-seconds
- open-loop generator lateness
- queue drain time

현재 deadline `5 ms`는 sensitivity value다. Aerial schedule에서 NRx stage budget을
도출하기 전에는 production deadline이라고 부르지 않는다. 결과에는 2/5/10 ms
sensitivity도 함께 저장한다.

### Problem gate

다음 두 현상이 **같은 selective-bursty run**에서 동시에 나타나는지 사전 정의한다.

```text
static_cell의 deadline 내 NRx 미완료 >= 1%
eligible endpoint-time idle fraction >= 10%
```

관측되면 fixed isolation의 capacity fragmentation problem을 지지한다. 관측되지
않으면 DART의 primary motivation을 수정한다. 이 gate만으로 DART가 system deadline을
보장했다고 주장하지 않는다. 실제 conventional fallback이 아직 포함되지 않기 때문이다.

구현:

- `cloudlab_aerial/task1/isca_v2/generate_multicell_trace.py`
- `cloudlab_aerial/task1/isca_v2/nrx_multicell_hardware.py`
- `cloudlab_aerial/task1/isca_v2/analyze_multicell_workloads.py`
- `cloudlab_aerial/task1/isca_v2/run_multicell_workloads_full.sh`

## 3. Gate B: radio-realistic selective NRx

Gate A의 NRx probability sweep을 최종 논문에서 임의 probability로 끝내지 않는다.
실제 선택 label은 paired conventional/NRx radio evaluation에서 만든다.

### B1. Reproducible link simulation

Aerial의 `example_neural_receiver.ipynb` 경로를 그대로 script화한다.

- 3GPP CDL-A/B/C/D/E와 Rayleigh control
- SNR, MCS, PRB, layer, Doppler/speed strata
- 같은 transmitted TB와 channel realization을 conventional과 NRx에 입력
- per-slot CRC, decoded TB, BLER, goodput delta 기록
- 최소 TB error/slot 수를 고정하고 seed/model/data hash 저장

### B2. OTA correctness smoke

SDK에 포함된 Aerial Data Lake example data를 사용한다.

```text
pyaerial/notebooks/data/fapi.parquet
pyaerial/notebooks/data/fh.parquet
```

이 데이터는 두 cell과 commercial UE 기반 OTA capture지만 표본이 작으므로 pipeline
correctness smoke에만 사용한다. capacity 또는 일반 BLER curve로 확대 해석하지 않는다.

### B3. Utility-labelled multi-cell trace

B1/B2에서 다음 label을 만든다.

```text
NRx utility(slot) = expected goodput_NRx - expected goodput_conventional
```

그 후 multi-cell trace의 각 grant에 `channel/MCS/utility`를 붙인다.

```text
utility <= threshold       -> conventional only
utility > threshold        -> NRx candidate
NRx deadline infeasible    -> JIT conventional fallback
NRx completes in time      -> NRx result commit
```

이 단계가 끝나야 `radio-aware DART`와 BLER/goodput claim을 허용한다.

## 4. 최종 paper figure

가장 중요한 problem figure는 같은 시간축에 다음을 겹친다.

```text
multi-cell NRx arrivals / channel-hard burst
per-endpoint queue and utilization
deadline miss or fallback
idle endpoint compute
background AI work
```

비교:

- static one/cell assignment
- peak-provisioned dedicated replicas
- MPS sharing
- shortest queue
- predicted-finish software baseline
- full DART-Rx transaction + fallback + slack controller

최종 승리 조건은 single-slot GDR latency 감소가 아니다.

> 같은 radio-labelled multi-cell trace에서 protected L1을 유지하면서 strongest
> software baseline보다 deadline miss, radio goodput, idle GPU-seconds, background
> application SLO의 Pareto front를 개선해야 한다.
