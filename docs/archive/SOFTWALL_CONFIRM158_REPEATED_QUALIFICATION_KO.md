> **보관 사유 (2026-09-25):** 이 문서는 실험 이력으로 보존한다. 최신 스킴과 claim boundary는 `docs/current`의 원고·모델·production gate 문서가 권위 기준이며, 과거 GPU0 전용, MIG, CloudLab 또는 4-GPU pool 가정은 현재 논문의 근거가 아니다.

# C158 반복 actual-NeuralRx 자격 결과

**최종 판정:** `C158_TWO_NODE_REPEATED_ACTUAL_NRX_PASS`  
**개발:** job `58855711`, node `nid002288`, label `confirm158a_repeated_attempt8_job58855711`  
**동결 holdout:** job `58856474`, node `nid002100`, label `confirm158b_repeated_holdout_job58856474`  
**결합 결과:** `results/softwall_multigpu/confirm158_repeated_actual_nrx_two_node.json`

## 1. 무엇을 새로 검증했는가

C157은 두 node에서 실제 TensorRT NeuralRx CRC가 recovery debt를 해제하거나 shared cuPHY
recovery를 남기는 전이를 한 번씩 통과한 mechanism canary였다. C158은 같은 전이를 persistent
endpoint에서 250 epoch씩 반복했다. 매 epoch에는 네 accepted TB와 한 global reject가 있고,
네 TB 모두 GPU3 actual NeuralRx를 거친다. 제때 성공한 debt는 해제하고 나머지는 GPU2 shared
cuPHY로 복구한다. Certificate가 허용하면 GPU2에서 context-64 Qwen lease를 실행한다.

검증 대상은 다음 결합 전체다.

```text
두 RAN home의 같은-TB 입력
  -> GPU3 persistent TensorRT NeuralRx
  -> 실제 CRC outcome을 mmap control page로 publish
  -> global all-fail certificate 재계산
  -> atomic Qwen lease + recovery retiming
  -> GPU2 shared cuPHY recovery
  -> single radio commit
```

## 2. 최종 mode와 사전 gate

| 항목 | 값 |
|---|---:|
| 반복 주기 | 600 ms |
| radio expiry | 155 ms |
| NeuralRx bound | 45 ms |
| conventional recovery path bound | 25 ms |
| Qwen bound | 35 ms |
| launch control bound | 5 ms |
| recovery guard | 2 ms |
| actual NRx 요청 | node당 1,000, 합계 2,000 |
| 채널 | 두 `+20 dB` stable key, 두 `-8.75 dB` transition key |

600 ms는 radio deadline을 늘린 값이 아니다. Synthetic channel 생성과 validation-only local
conventional oracle가 C157에서 약 225--307 ms 걸렸기 때문에, 다음 입력 준비를 이전 epoch의
155 ms expiry 뒤 staging 구간에 격리하려고 사용했다. Timed radio contract는 계속 D155다.
각 release 100 ms 전에는 선언한 warm-lifecycle activation을 실행하며, activation 전에 네 입력
준비가 끝났는지도 gate로 검사했다.

`-8.75 dB`는 C158 결과를 본 뒤 고른 성공점이 아니다. 기존 Confirm18 독립 500-sample
waterfall sweep에서 conventional 0/500, NeuralRx 271/500이었던 transition point다. 개발 중
사용한 `-15 dB`는 validation replica에서 `CRC=0 + wrong payload` 같은 out-of-range 동작을
반복해 최종 mode에서 제외했다. 최종 source와 mode를 고정한 뒤 holdout은 새 node와
`31,000,000` seed block에서 실행했다.

## 3. 반복 실행이 찾아낸 세 가지 실제 결함

### 3.1 per-epoch JSON은 deadline control plane으로 쓸 수 없다

첫 장기 run의 82번째 epoch에서 owner는 NRx를 release 뒤 2.50 ms에 끝냈지만 `/pscratch`의
JSON create/rename metadata 지연 때문에 coordinator는 45 ms cutoff에서 outcome을 보지
못했다. Recovery는 release 뒤 약 197 ms에 시작했다. 최종 구현은 시작 시 한 번 만든 64-byte
mmap control page에 outcome sequence와 dispatch action을 기록한다. JSON은 최종 계측
보존에만 쓴다.

### 3.2 host doorbell만으로 response consumption을 증명할 수 없다

Worker가 P2P copy를 끝내고 host doorbell을 올린 뒤에도 owner가 이전 response byte를 읽는
사례가 있었다. 최종 경로는 입력과 응답 모두 round-trip echo를 사용한다.

1. Owner는 release 전 원본 input shadow를 보존한다.
2. Worker가 받은 input을 owner buffer로 다시 복사한다.
3. Owner가 shadow와 byte-exact 비교한다.
4. Worker output도 owner로 복사한 뒤 worker가 다시 읽어 원본과 byte-exact 비교한다.
5. 두 방향 확인 뒤에만 completion doorbell을 publish한다.

이 검증 copy와 비교 시간은 45/25 ms path bound 안에 포함된다. C158에서 input round-trip과
response echo 위반은 합계 0이었다.

### 3.3 monitor도 lifecycle contract의 일부다

새 echo에 처음 사용한 `cp.array_equal`은 첫 호출 때 JIT되어 NeuralRx path 337 ms,
recovery path 210 ms tail을 만들었다. 모든 peer의 P2P 방향과 equality monitor를 readiness
전에 preflight한 뒤 이 tail은 사라졌다. 이는 C157의 첫 `cp.isfinite` JIT failure와 같은
종류다. 실제 kernel만 warmup하고 safety monitor를 빼면 warm mode가 아니다.

## 4. CRC wire contract와 같은-TB 의미

심하게 손상된 입력에서 pyAerial이 내놓은 raw CRC byte와 별도 `_verify` 판정이 간헐적으로
달랐다. 따라서 shared recovery wire status는 verified 판정을 `0=pass, 1=fail`로
normalization한다. Raw byte는 telemetry로 보존한다. Holdout에서 raw byte가 0이지만 verified
failure였던 1건을 이 규칙이 fail로 직렬화했다.

Local validation replica와 GPU2 shared cuPHY는 waterfall 경계에서 같은 input에도 decode
성공 class가 한 번 달랐다. Local replica는 production decision path가 아니므로 두 독립
decoder의 출력 일치를 substrate 보장으로 두지 않는다. 대신 다음 직접 조건을 사용한다.

- Worker가 받은 input은 owner shadow와 byte-exact여야 한다.
- Shared CRC pass는 transmitted TB와 payload가 byte-exact여야 한다.
- Shared CRC fail은 radio failure 결과이며 payload byte는 소비하지 않는다.
- Local/shared class disagreement는 PHY telemetry로 별도 보고한다.

개발 node의 560개 중 한 recovery에서 class disagreement 1건, holdout에서는 0건이었다.
Input exact와 shared response contract violation은 두 node 모두 0이다.

## 5. 최종 결과

| 지표 | 개발 `nid002288` | holdout `nid002100` | 결합 |
|---|---:|---:|---:|
| epoch | 250 | 250 | 500 |
| actual NeuralRx 요청 | 1,000 | 1,000 | 2,000 |
| 제때 NRx 성공 | 720 | 720 | 1,440 |
| physical shared recovery | 280 | 280 | 560 |
| Qwen unit | 250 | 250 | 500 |
| radio single commit | 1,000 | 1,000 | 2,000 |
| deadline miss | 0 | 0 | 0 |
| input round-trip 위반 | 0 | 0 | 0 |
| response echo 위반 | 0 | 0 | 0 |
| recovery contract 위반 | 0 | 0 | 0 |
| local/shared class disagreement | 1 | 0 | 1 |

| 시간 | 결합 p99 | 결합 max | bound |
|---|---:|---:|---:|
| NRx release-to-complete | 8.439 ms | 26.385 ms | 45 ms |
| recovery physical path | 4.468 ms | 17.763 ms | 25 ms |
| Qwen execution | 29.871 ms | 32.948 ms | 35 ms |
| radio release-to-commit | 98.840 ms | 107.941 ms | 155 ms |

두 passing protocol의 22개 source SHA-256은 서로 같고 현재 source와도 일치한다. Holdout
protocol의 exclusion set에는 development node `nid002288`이 들어 있다.

## 6. 무엇을 주장할 수 있는가

이 결과는 다음 문장을 지지한다.

> Persistent MIG-off/MPS 4-GPU mode에서 실제 TensorRT NeuralRx outcome이 만든 조건부
> recovery debt, global certificate, atomic Qwen lease, shared cuPHY recovery와 물리
> input/output fence를 두 A100 node의 2,000 actual-NRx request에서 deadline·bound·credit·
> lifecycle 위반 없이 반복 실행했다.

다음은 아직 주장하지 않는다.

- WCET나 deterministic hard real-time guarantee
- cold start, long idle, process restart, GPU hang
- production DU/FAPI `d_MAC`
- 다른 GPU family와 topology
- C159 strong baseline 대비 throughput 우위
- 600 ms qualification을 BurstGPT의 임의 도착률에 그대로 적용할 수 있다는 주장

특히 C159는 validation-only oracle를 online path에서 제거하고 실제 trace arrival에서
staging과 queueing contract를 다시 고정해야 한다. C158 pilot 표본은 performance CI에
합치지 않는다.

## 7. 개발 실패 이력

장기 반복이 아니면 보이지 않던 실패를 보존했다.

| 단계 | 결과 | 설계 수정 |
|---|---|---|
| pilot1, 10 epoch | 40/40 PASS | wiring 확인 |
| 최초 장기 run | outcome file metadata tail, late recovery | per-epoch mmap control |
| `-15 dB` 반복 | validation CRC false-pass/replica divergence | 독립 calibration의 `-8.75 dB` transition mode |
| 최초 response fence | 첫 equality JIT로 337/210 ms tail | monitor와 모든 P2P 방향 preflight |
| raw CRC response | verified failure와 wire byte 불일치 | boolean CRC wire normalization |
| strict local replica equality | 같은 input의 decoder class 차이 | exact input proof와 shared radio contract를 직접 검사 |

개발 중 한 초기 실패 label을 재사용해 그 이전 raw 파일이 덮인 기록상 결함이 한 번 있었다.
이 사실은 manifest limitation에 남긴다. 최종 development와 holdout passing label은 서로
고유하며 실행 전 protocol/source hash와 원시 결과를 모두 보존했다.

