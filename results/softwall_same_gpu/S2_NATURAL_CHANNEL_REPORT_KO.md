# SoftWall S2 동일 GPU 자연 채널 실험 보고서

**실행일:** 2026-09-20 UTC  
**환경:** A100-SXM4-40GB, MIG OFF, 실제 NVIDIA MPS, Aerial 25.3.2  
**Slurm:** `58637416` (`nid002693`), `58660634` (`nid001081`)  
**프로젝트 루트:** `/pscratch/sd/s/sgkim/kcj/airan_cloudlab`

## 판정

S2의 첫 동일 GPU vertical slice는 구현·실행됐다. 같은 valid PUSCH 입력을 conventional full receiver와 NeuralRx full receiver에 넣고, optional NeuralRx와 conventional recovery interval을 원자적으로 예약하며, 한 결과만 commit하고 남는 시간에 별도 MPS AI work unit을 허가한다.

현재 증거가 지지하는 주장은 다음과 같다.

> 선언한 workload·arrival·경험적 실행 상한 안에서 S2는 conventional recovery를 보존하며 자연 NeuralRx 실패를 복구했고, eager dual execution과 비열등한 radio utility를 유지하면서 같은 GPU의 background AI 처리량을 높였다.

이 결과는 MPS만으로 모든 workload를 완전 격리했다는 뜻이 아니다. 보장은 허용된 bounded work unit, 최대 미완료 unit 1개, completion ACK, admission/fallback calendar, 고정 timing contract를 전제로 한다. 측정 상한은 수학적 WCET가 아니므로 현재 deadline 결과는 조건부 경험적 보장이다.

## 구현한 스킴

요청마다 다음 순서로 처리한다.

```text
1. NRx endpoint slot과 conventional fallback interval을 함께 예약
2. 둘 중 하나라도 예약할 수 없으면 optional NRx를 거부하고 conventional 실행
3. NRx full path 실행: LS CE → persistent TensorRT/CUDA Graph → LDPC/CRC
4. NRx가 유효하고 fallback cutoff 전이면 NRx 한 번만 commit
5. 실패·지연이면 예약한 시각에 conventional full path 실행
6. conventional commit 뒤의 late NRx completion은 epoch/single-commit fence로 거부
7. 다음 RAN release 전 안전 여유에만 bounded background unit 발급
```

Conventional path는 `CE → noise/interference estimation → equalization → de-rate-match → LDPC → CRC` 전체를 포함한다. 두 경로는 같은 payload와 같은 noisy resource-grid tensor를 사용하며 CRC와 원 payload byte equality를 모두 검사한다.

## 1. 동일 입력 full receiver 검증

Clean valid PUSCH 1,000개에서 두 경로 모두 1,000/1,000 correct였다.

| receiver | mean GPU | p99 GPU | max GPU | CRC/payload error |
|---|---:|---:|---:|---:|
| conventional full path | 4.265 ms | 11.067 ms | 22.660 ms | 0/1,000 |
| direct persistent NeuralRx full path | 3.198 ms | 5.701 ms | 12.761 ms | 0/1,000 |

pyAerial `TrtEngine` wrapper의 약 104 ms 호출 병목을 제거하고 persistent binding과 CUDA Graph를 사용했다. Direct engine output은 wrapper output과 원소 단위로 일치했다.

## 2. 자연 채널 상보성

채널은 RX antenna별 독립 block-Rayleigh coefficient와 complex AWGN이다. 두 receiver는 매 trial마다 동일한 immutable noisy slot을 받는다. **이 표의 AWGN 전력은 각 trial의 fading 적용 후 신호 전력으로 다시 계산된다.** 즉 nominal SNR을 block마다 고정한 합성 trace이며, noise floor를 고정한 fading channel이나 field channel의 결과로 읽으면 안 된다. 후속 [잡음 기준 민감도 검사](channel_feature_gate_job58693851.md)는 fixed pre-fading noise 조건에서 같은 nominal −8.5 dB의 NeuralRx 정답이 크게 달라짐을 보였다. 해당 검사의 gate train/test는 채널 seed 중복으로 무효이므로 online channel-gate 검증 근거는 아니다.

| SNR | trials | conventional correct | NeuralRx correct | conventional-only | NeuralRx-only | union |
|---:|---:|---:|---:|---:|---:|---:|
| −9.00 dB | 500 | 0 | 5 | 0 | 5 | 5 |
| −8.75 dB | 500 | 0 | 271 | 0 | 271 | 271 |
| **−8.50 dB** | **500** | **180** | **489** | **2** | **311** | **491** |
| −8.25 dB | 500 | 496 | 500 | 0 | 4 | 500 |
| −8.00 dB | 500 | 500 | 500 | 0 | 0 | 500 |

−8.5 dB에서 conventional이 NeuralRx가 놓친 2개 trace를 실제로 복구했다. 이는 합성 timeout이 아니라 measured CRC/payload 결과로 생긴 receiver complementarity다.

## 3. 합성 invalidation으로 recovery 의미론 검증

P50/D40, 1 cell, 3×500 releases에서 매 10번째 NeuralRx 결과를 계산 뒤 invalid로 만들었다. 실제 두 receiver 계산은 그대로 수행했다.

| policy | releases | miss | incorrect | NRx/conv commit | background/s |
|---|---:|---:|---:|---:|---:|
| conventional-only | 1,500 | 0 | 0 | 0/1,500 | 648.9 |
| NRx-only | 1,500 | 0 | 150 | 1,350/0 | 674.4 |
| eager dual | 1,500 | 0 | 0 | 1,350/150 | 592.5 |
| **S2 reserved** | **1,500** | **0** | **0** | **1,350/150** | **646.7** |

S2는 eager와 같은 유효 결과를 내면서 background 처리량을 9.15% 높였고 conventional-only 처리량의 99.66%를 유지했다. S2의 duplicate commit, admission rejection, branch-bound violation, AI budget violation과 다음 release crossing은 모두 0이었다.

## 4. 자연 실패 5,000-release 탐색 실험

Confirm19는 −8.5 dB, P60/D50에서 5×1,000 releases를 네 정책으로 실행했다.

| policy | correct | deadline miss | background/s |
|---|---:|---:|---:|
| conventional-only | 1,820/5,000 | 0 | 649.3 |
| NRx-only | 4,893/5,000 | 0 | 641.8 |
| eager dual | 4,923/5,000 | 0 | 596.5 |
| S2 reserved | 4,919/5,000 | 0 | 638.1 |

S2는 NRx-only 실패 28건을 복구하고 eager보다 background 처리량을 6.96% 높였다. 그러나 S2/eager release별 mismatch가 8건이었고 S2 branch-bound violation이 5건 발생해 사전 정의한 완전 일치 gate는 실패했다.

Mismatch 8개를 각 30회 다시 실행한 결과 입력 tensor mutation은 없고 NeuralRx 판정은 선택된 trace에서 반복적으로 안정적이었다. Conventional 판정은 임계 trace에서 크게 변했다. 예를 들어 세 trace의 conventional success는 각각 10/30, 5/30, 28/30이었다. 별도로 두 trace에서는 S2 NeuralRx 실행이 32–42 ms로 늘어나 기존 cutoff를 넘었다. 따라서 decoder 경계의 수치적 변동과 timing-bound 부족을 분리했다.

## 5. 자연 채널 비열등성

Confirm21은 새 channel seed를 사용하고 timing tail을 포함하도록 P90/D80, NRx bound 50 ms, conventional bound 24 ms로 고정했다. Eager와 S2를 각각 10,000 releases 실행했다.

| metric | result |
|---|---:|
| eager correct | 9,813/10,000 |
| S2 correct | 9,810/10,000 |
| S2−eager | −0.0300 percentage points |
| paired 95% CI | [−0.0819, +0.0219] pp |
| frozen noninferiority margin | −0.25 pp |
| noninferiority | pass |
| S2/eager background | 701.2/s / 679.6/s |
| S2 gain | +3.17% |
| S2 deadline miss / branch-bound violation | 0 / 0 |
| seed mismatch | 0 |

사전 protocol의 전체 gate는 한 번의 admission rejection 때문에 형식상 실패했다. 해당 release는 시작이 28.759 ms 늦어 NRx bound를 fallback cutoff 전에 배치할 수 없었다. Runtime은 NRx를 실행하지 않고 conventional로 전환해 31.737 ms에 올바르게 완료했으며 D80을 지켰다. 즉 안전 실패가 아니라 fail-closed 동작이었지만, protocol이 rejection 0을 요구했으므로 결과를 사후 성공으로 바꾸지 않았다.

## 6. Fail-closed admission 재현

Confirm22는 clean PUSCH 1,000 releases 중 매 10번째 release에 30 ms controller lateness를 주입했다.

- 주입 100건 모두 optional NRx admission 거부
- 100건 모두 conventional commit
- 1,000/1,000 correct, deadline miss 0
- unsafe rejection, duplicate commit, branch-bound violation, AI violation 0

이 결과는 예측 상한을 만족하지 못하는 optional work를 거부하면서 mandatory 결과를 보존하는 계약을 검증한다.

## 7. 두 셀 correlated fallback burst

Confirm23은 같은 noisy trace를 두 logical cell에 적용해 실패가 동시에 발생하도록 만들고, 단일 fallback lane에 `[80,104] ms`, `[104,128] ms` 두 recovery interval을 예약했다. 3×1,000 releases에서 S2와 eager의 correct 수는 각 라운드 985, 987, 984로 완전히 일치했고 S2 deadline/bound/admission/duplicate violation은 모두 0이었다.

Eager r1의 background RPC가 GPU 0.886 ms 작업에 대해 host scheduling 때문에 28.170 ms 걸려 10 ms budget을 한 번 위반했다. 따라서 전체 평균의 +31.62%는 유효 비교가 아니다. AI violation이 없는 r2–r3만 비교하면 S2 729.1/s, eager 699.7/s로 +4.20%다. 이 실패를 보존하고 더 큰 host budget으로 새 protocol을 실행했다.

Confirm24는 AI end-to-end budget 40 ms, guard 2 ms, timeout 45 ms를 고정했다.

| metric | eager | S2 |
|---|---:|---:|
| correct releases | 2,961/3,000 | 2,960/3,000 |
| deadline miss | 0 | 0 |
| AI budget/crossing violation | 0 | 0 |
| background throughput | 527.4/s | 556.6/s |

S2−eager는 −0.0333 pp, paired 95% CI는 [−0.0987, +0.0320] pp였고 frozen −0.5 pp noninferiority margin을 통과했다. S2 background gain은 5.53%였다. S2의 branch-bound violation, admission rejection, duplicate commit은 모두 0이었다. Confirm24의 모든 frozen gate가 통과했다.

## 8. Transaction correctness

DART runtime의 12개 unit test와 5,000-iteration fault campaign을 실행했다.

- wrong, late, stale, invisible, restarted completion의 잘못된 accept: 모두 0
- duplicate commit accept: 0
- fallback storm over-admission: 0
- fallback reservation leak: 0
- 5,000-iteration campaign: pass

이는 CUDA 실행시간 보장이 아니라 admission, reservation rollback, epoch fencing, single commit, fallback credit 수명의 논리적 정확성을 검증한다.

## 9. 별도 MPS client의 물리적 overrun

앞선 S2는 같은 controller context에서 NRx 호출이 반환된 뒤 conventional을 실행했다. Timeout 뒤에도 optional GPU work가 물리적으로 남는 조건을 만들기 위해, 별도 low-priority MPS worker가 persistent NeuralRx CUDA Graph를 40회 연속 실행하게 했다. Controller는 5 ms에 timeout하고 late response를 commit하지 않은 채 clean valid-PUSCH conventional full path를 실행했다.

Cap100 worker는 108 SM을 보고 unit당 평균 약 44.94 ms, cap20 worker는 실제 20 SM을 보고 약 126–127 ms 실행됐다. Cap20 overrun은 여러 후속 RAN release까지 계속됐다.

Confirm26 P60/D25 결과:

| optional cap | releases | physical-overlap releases | deadline miss | active-overlap miss | active response p99 | active worst |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1,500 | 150 | 2 | 1 | 19.405 ms | 25.970 ms |
| 20 | 1,500 | 450 | 1 | 1 | 18.150 ms | 28.763 ms |

모든 TB는 correct였지만 D25 frozen gate는 실패했다. Cap20은 간섭 tail을 낮췄지만 완전 격리를 만들지 못했다. 따라서 이 optional unit은 D25 계약에서 admission하면 안 된다.

Confirm27은 confirm26 worst에 6.237 ms guard를 더한 D35로 5×1,000 releases를 실행했다.

| optional cap | releases | physical-overlap releases | deadline miss | active-overlap miss | active worst |
|---:|---:|---:|---:|---:|
| 100 | 5,000 | 500 | 1 | 0 | 25.091 ms |
| 20 | 5,000 | 1,500 | 0 | 0 | 33.862 ms |

Cap20의 선언된 workload는 1,500 physical-overlap releases에서 D35 miss 0이었지만, 전체 protocol은 cap100의 non-overlap release에서 conventional GPU event가 36.076 ms로 늘어나 실패했다. Worker와 socket을 완전히 제거한 confirm28 conventional-only 10,000 releases에서는 miss 0, response max 7.642 ms, GPU max 7.571 ms였다. 따라서 36 ms event를 순수 RAN WCET로 단정할 수 없고, 반복 overrun client가 존재한 MPS campaign의 residual/system tail로 분류한다.

Confirm29는 첫 timeout 뒤 late response를 drain하고 이후 admission을 모두 막되 worker process와 CUDA context는 idle로 유지했다. Cap100은 0/5,000, cap20은 1/5,000 miss로 전체 gate가 실패했다. Confirm30은 같은 frozen P60/D35 조건에서 drain 직후 worker process를 종료했다.

| optional cap | releases | physical-overlap releases | deadline miss | active-overlap miss | process 종료 뒤 miss | worst response |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 5,000 | 5 | 6 | 0 | 6 | 96.625 ms |
| 20 | 5,000 | 15 | 11 | 0 | 11 | 100.002 ms |

모든 TB는 correct였고 timeout, drain, endpoint retirement gate는 동작했지만 deadline gate는 양 cap에서 실패했다. 17 misses 중 14건은 conventional GPU event 자체가 42.569–99.944 ms로 늘어났고, 3건은 직전 긴 event가 다음 release를 늦춘 연쇄 miss였다. 종료 뒤 20 releases 안에 cap100 5건과 cap20 6건이 집중됐다. 이는 인과관계의 최종 증명은 아니지만, worker 없는 confirm28과 결합하면 **실시간 중 MPS client/context teardown을 안전 동작으로 간주할 수 없다는 강한 반례**다.

동시에 실행되어 같은 confirm30 파일명과 socket을 사용한 중간 campaign은 오염 자료로 분리해 `raw/contaminated_confirm30_parallel/`에 보존했고 집계에서 제외했다. 유효 r1–r2와 MPS 재시작 뒤 단독 실행한 r3–r5만 위 표에 포함했다. 이후 campaign wrapper에는 동일 campaign lock과 PID가 포함된 고유 socket 이름을 추가했다.

Confirm31은 timeout/late drain 뒤 live RAN phase를 먼저 끝내고, worker 종료 후 RAN release가 없는 20초 maintenance window를 둔 뒤 새 conventional process를 시작했다. 그러나 사후 monotonic timestamp 대조에서 별도 sham-retirement campaign의 release 구간과 겹친 사실을 확인했다. 아래 수치는 보존하되 PASS 근거에서는 제외하고 단독 재실행한다.

| prior optional cap | runs | fault-phase releases / miss | post-maintenance releases / miss | post worst response | post worst GPU |
|---:|---:|---:|---:|---:|---:|
| 100 | 3 | 300 / 0 | 3,000 / 0 | 13.410 ms | 13.323 ms |
| 20 | 3 | 300 / 3 | 3,000 / 0 | 16.484 ms | 16.413 ms |

기존 analyzer상 frozen recovery gate는 통과했지만 동시 campaign 오염 때문에 유효 판정이 아니다. Cap20 fault phase의 miss와 post 결과 모두 재실행 결과로 교체하기 전에는 claim에 사용하지 않는다.

Confirm32는 같은 요청의 실제 data plane을 별도 persistent process로 연결했다. Controller는 request-specific PUSCH와 channel estimate를 CUDA IPC forward buffer로 전달하고, cap80 endpoint의 output LLR을 backward buffer로 받아 로컬 LDPC/CRC를 실행했다. Clean 10,000 releases는 correct 10,000/10,000, miss/timeout/fallback 0, response p99 4.125 ms, max 8.730 ms였다. Worker는 86 SM과 10,020 completed units를 기록했고 전체 timed epoch 동안 유지됐다.

Confirm33은 같은 cap80 외부 endpoint에 −8.5 dB block-Rayleigh+AWGN trace를 넣었다. 10,000 releases 중 NeuralRx는 9,779건을 commit했고 221건에서 자연 실패가 발생해 같은 요청의 conventional path를 실행했다. 그중 47건이 복구되어 최종 correct는 9,826건이었다. Deadline miss와 endpoint timeout은 0, response p99/max는 6.004/30.669 ms였고 worker는 10,021 units, 평균 GPU 1.226 ms, 86 visible SM을 기록했다. 따라서 별도 process/CUDA-IPC 경로에서 자연 실패 trigger와 same-request recovery가 실제로 동작했다.

Confirm36은 새 node `nid001069`, 한 Slurm allocation `58676333`과 MPS epoch, 같은 payload/channel seed의 세 10,000-release 조건을 순차 재생했다. Conventional-only 3,688, local eager-dual 9,835, cap80 외부 S2 9,840 correct였고 모두 deadline miss 0이었다. 외부 S2−eager paired 차이 +0.0500 percentage points, 95% CI [−0.0088,+0.1088] pp는 사전 마진 −0.25 pp를 통과했다. Per-release win은 외부 7건, eager 2건이었다. 외부 S2는 natural fallback 203건에서만 conventional을 실행해 43건을 추가로 복구했고, endpoint 10,000회 요청·timeout 0, worker 10,021 units·86 visible SM으로 모든 gate가 통과했다. Eager는 매 요청에 conventional을 실행했으므로 외부 S2가 불필요한 conventional 실행 9,797회를 생략한 사실은 확인됐다. 이후 외부 경로의 background 처리량은 Confirm38·41에서 측정했다. Confirm34의 MPS 오염 시도와 Confirm35의 중단된 baseline은 paired 근거에서 제외한다.

Confirm37의 세 번째 cap20 MPS AI worker는 6 ms host RPC 예산을 한 번 초과해 실패했다. 독립 seed와 40 ms 예산의 Confirm37b는 500회 RAN miss·AI 계약 위반 없이 완료 AI 6,519개로 통과했다. Confirm38은 외부 endpoint, local S2, eager-dual에 동일한 cap20 background worker를 붙여 각각 10,000회를 같은 node·trace에서 비교했다. 세 방식의 RAN miss와 AI 계약 위반은 모두 0이었고, 완료 AI는 외부 130,177개, local S2 127,180개, eager 126,477개였다. 역순 독립 seed Confirm41에서도 세 방식 모두 correct 9,817, miss·AI 계약 위반 0이고 AI 완료량은 외부 130,254개, local 127,231개, eager 126,390개였다. 외부−eager 처리량 차이는 Confirm38 +2.93%, Confirm41 +3.06%; Confirm41 radio paired 95% CI는 [−0.0480,+0.0480] pp로 마진 −0.25 pp를 통과했다. Local S2와 eager의 NRx 20 ms branch-bound 초과 각 2건은 RAN deadline miss와 구분한다. 이 두 외부 경로에는 DART 원자 예약·single commit·gap lease가 아직 없으므로, 이 처리량 차이를 완성된 SoftWall transaction의 이득으로 귀속하지 않는다. 실제 연결을 한 Confirm42 smoke는 통과했지만 Confirm43 첫 OFF 10,000회에서 NRx 30 ms bound 위반 1건이 나왔으며, 완전 결합 baseline 비교는 Confirm45에서 검증해야 한다.

**외부 경로의 구현 경계:** Confirm32–38의 CUDA-IPC controller는 자연 NRx 실패 후 같은 요청의 conventional 처리를 실행했지만, 이 절 앞에서 설명한 로컬 S2의 원자 fallback-calendar 예약과 `DartRuntime` single-commit 상태기계를 외부 경로에 연결하지는 않았다. 외부 결과는 physical endpoint·조건부 fallback·background 동시 실행의 근거다. 전체 transaction의 외부 구현은 Confirm42부터 별도 검증한다.

이 실험의 결론은 분명하다. **MPS cap은 간섭을 줄이는 qualification 수단이지만 완전 격리 primitive가 아니다.** S2는 optional work의 물리적 blocking과 mandatory path의 campaign tail을 함께 상한화할 수 있을 때만 admit해야 하며, D25는 reject 대상이다. Bound 위반 시에는 결과를 즉시 폐기하고 endpoint를 논리 quarantine하지만, process/context teardown은 RAN-free maintenance window까지 미룬다. D35 cap20의 confirm27 overlap 결과만을 hard guarantee로 확대하지 않는다.

## Novelty에 주는 의미

각 구성 요소인 MPS, bounded work unit, primary/backup reservation, NeuralRx/conventional 선택은 개별적으로 새롭지 않다. 현재 결과가 지지하는 결합은 다음과 같다.

1. **실제 radio utility와 recovery 의무의 결합:** 같은 noisy PUSCH에서 NeuralRx와 conventional의 상보성을 측정하고, 미래 CRC를 admission에 사용하지 않은 채 runtime CRC 실패를 recovery trigger로 사용했다.
2. **원자적 optional+mandatory transaction:** NRx endpoint credit과 conventional recovery interval을 함께 예약하고, 늦거나 잘못된 결과를 single-commit/epoch fence로 차단했다.
3. **복구 예약과 background lease의 공동 사용:** recovery capacity를 먼저 보존하고 남는 시간에만 별도 MPS AI unit을 발급해 eager dual보다 3.17–5.53% 높은 유효 처리량을 얻었다.
4. **여러 셀의 correlated recovery:** 두 셀의 fallback interval을 중복 판매하지 않고 한 lane에 직렬 예약해 deadline과 정확도를 유지했다.
5. **예측 실패의 fail-closed 처리:** 늦게 시작한 요청을 억지로 NRx에 넣지 않고 mandatory conventional로 전환했다.

검토한 AI-RAN 연구와의 차이는 “MPS를 사용했다”가 아니라, **같은 요청의 optional neural utility, conventional recovery calendar, single commit, background lease를 동일한 deadline transaction으로 묶고 actual full PHY에서 검증했다**는 데 둔다.

## 아직 주장하지 않는 것

- MPS만으로 임의의 AI kernel, HBM traffic, DMA를 완전 격리했다는 주장
- 비협조적이거나 무한 실행하는 kernel의 강제 선점
- 측정 max를 넘어서는 수학적 WCET 증명
- 실제 0.5/1 ms NR slot production deadline 달성
- 독립적인 실제 두 셀 channel/user distribution 검증
- 다중 GPU/P2P/GDR까지 포함한 최종 시스템 완성

다음 강화 항목은 이 별도 process endpoint에 background lease와 timeout credit을 연결하는 것, maintenance quiet-time sweep과 MPS cleanup tail 원인 계측, 실제 독립 cell trace, 다른 MCS/channel과 Qwen decode/training phase, 그리고 방어 가능한 service bound 산출이다.

## 재현 자료

- 동일 입력 receiver: `dual_receiver_phy.py`, `validate_dual_receiver.py`, `profile_dual_receiver.py`
- 자연 채널 sweep: `sweep_dual_receiver_snr.py`
- S2 runtime: `s2_runner.py`, `run_s2_campaign.sh`
- 분석: `analyze_s2.py`, `analyze_s2_paired.py`, `analyze_s2_noninferiority.py`, `analyze_s2_admission_fault.py`
- 안정성 진단: `probe_receiver_stability.py`
- Frozen protocols: `confirm15`–`confirm24`, `confirm26`–`confirm40` protocol JSON (중단·준비 단계는 ledger 참조)
- 주요 유효 결과: `confirm16`, `confirm18`–`confirm33`, `confirm36` 결과 JSON/Markdown
- 별도 client overrun: `overrun_controller.py`, `run_overrun_campaign.sh`, `analyze_overrun.py`

모든 파일은 이 보고서와 같은 project root 아래에 있고 raw result는 `results/softwall_same_gpu/raw/`에 보존했다.
