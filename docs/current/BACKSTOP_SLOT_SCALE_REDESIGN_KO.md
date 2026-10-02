# Backstop 슬롯 단위 재설계 (8·16셀, MIMO, 4.5 ms)

> **상태(2026-09-30): 과정 기록.** 현재 스킴의 기준 문서는 [설계](BACKSTOP_SLOT_DESIGN_KO.md), [결과](BACKSTOP_SLOT_RESULTS_KO.md), [기존 연구와 다른 점](BACKSTOP_SLOT_DIFFERENTIATION_KO.md)이다. 이 문서는 마감이 하나(4.0 ms)이고 AI가 128토큰 단위·MPS 50%였던 첫 구성을 적은 것이다. 2.1의 "최신 시작 시각까지 결과가 없으면 NeuralRx를 시작"은 admission이 켜져 있으면 실제로는 동작하지 않는다(설계 문서 5장).

작성: 2026-09-29. 코드: `scripts_for_node/backstop_slot/`. 결과: `results/backstop_slot/`.

## 1. 기준과 목표

- 마감: testMAC의 UL indication 한계 T0+4.5 ms. T0는 UL 슬롯 시작이고, IQ는 슬롯이
  끝나는 T0+0.5 ms에 준비된다. 따라서 **데이터 도착부터 4.0 ms 안에** TB 결과(성공 또는
  NACK)를 내야 한다. 표에는 4.5 ms 기준도 함께 적는다.
- 프레임: 100 MHz(273 PRB), 30 kHz SCS, TDD DDDSU. 셀마다 UL 슬롯이 2.5 ms마다 한 번
  온다. TDD 셀은 서로 동기화되어 있으므로 **모든 셀의 UL 슬롯이 같은 순간에 도착한다.**
- 규모: 한 서버(A100 4장)에 8셀, 16셀.
- 셀당 UE: UL 슬롯마다 전 대역 PUSCH 하나.
  - 강한 UE: rank 2(2 layer SU-MIMO), 16QAM(MCS 12), CDL-C, Es/No 20–24 dB. conventional만 복호.
  - 약한 UE(셀 가장자리): rank 1, QPSK(MCS 7), CDL-D, Es/No −3.8 ~ −3.2 dB. NeuralRx 대상.
  - rank adaptation 때문에 SINR이 낮은 UE는 실제로 1 layer로 보낸다. 공개 pyAerial NeuralRx
    모델이 받는 형태(1 layer, QPSK, 4 RX)가 바로 이 경우다.
  - 약한 셀 비율 50%(8셀 중 4, 16셀 중 8).

## 2. 무엇이 바뀌었나

| | 기존 Backstop (P180/D155) | 슬롯 단위 Backstop |
|---|---|---|
| 마감 | 153 ms | 4.0 ms (T0+4.5 ms) |
| 먼저 도는 수신기 | NeuralRx, 실패하면 conventional 복구 | conventional을 모든 TB에 항상 실행 |
| NeuralRx | 모든 TB에 먼저 실행 | conventional이 code block 1개만 실패한 TB를 살림 |
| 대기 중 예약 | NeuralRx 결과를 기다리는 복구 시간 | conventional 결과를 기다리는 NeuralRx 시간 |
| AI 단위 | Qwen prefill 한 번(35–75 ms) | layer 하나 × 128 토큰(0.31 ms) 단위 |
| AI와 무선 | 시간으로 나눔 | 같이 돌되, 여유 없는 NeuralRx와는 겹치지 않음 |

### 2.1 NeuralRx 시작 규칙

약한 TB마다 **NeuralRx 최신 시작 시각 = 도착 + 4.0 ms − NeuralRx bound(2.5 ms)** 이다.

- conventional이 먼저 CRC를 통과하면 NeuralRx를 돌리지 않는다(GPU 시간을 돌려받는다).
- conventional이 실패하면, 실패한 code block 수를 본다. 약한 TB는 code block 4개다.
  NeuralRx가 살린 TB는 거의 모두 conventional이 **1개만** 실패한 경우였다(1,024 TB 중
  살린 29개 가운데 28개). 2개 이상 실패하면 NeuralRx를 돌리지 않고 NACK를 보낸다.
  1개 실패면 그 순간 NeuralRx를 시작한다.
- 최신 시작 시각까지 conventional 결과가 없으면 NeuralRx를 그때 시작한다. 이 경우 두
  수신기가 동시에 돈다.
- 빈 lane이 없어서 최신 시작 시각을 넘기면 그 TB의 NeuralRx는 **시작하지 않는다**.
  늦게 끝날 작업이 다른 TB를 느리게 만들기 때문이다.
- NeuralRx는 GPU마다 하나 있는 lane에서 돈다. 최신 시작 시각이 이른 TB부터 비어 있는 lane에
  배정한다(EDF). 슬롯 IQ는 conventional GPU 메모리에 있고 NeuralRx 시작 시 P2P로 복사한다.

결과 확정(single commit): CRC를 먼저 통과한 결과 하나만 확정한다. 마감까지 통과한 결과가
없으면 마감 시각에 NACK를 보낸다(HARQ 재전송). 늦게 끝난 NeuralRx 결과는 버린다.

### 2.2 AI를 빈 GPU 시간에 넣는 규칙

Qwen2.5-1.5B prefill을 128토큰 chunk × layer 하나 단위로 나누고, 단위마다 CUDA graph 하나로
캡처했다. 단위의 GPU 시간은 위치와 무관하게 거의 일정하다(layer 단위 p50 0.31 ms,
p99.9 0.33 ms → bound 0.35 ms). KV cache와 hidden state는 GPU에 남아 있어서 다음 단위를 다음
빈 시간에 이어서 실행한다. HF 전체 forward와 argmax가 같고 logit 차이는 0.055 이하다.

AI worker는 MPS 50% share로 무선 작업과 같은 GPU에서 **같이 돈다**. 같이 돌면 conventional과
NeuralRx가 조금 느려진다(16셀, NeuralRx p99 2.41 → 2.77 ms, conventional p99 1.33 → 1.51 ms).
그래서 NeuralRx bound를 두 개 둔다: 혼자 돌 때 2.5 ms, AI와 같이 돌 때 2.8 ms.

controller는 AI 조각(최대 1 ms)을 내주기 전에, 그 조각과 겹칠 수 있는 무선 작업이 모두 마감을
지키는지 본다.

1. conventional은 여유가 크다(p99 1.5 ms, 마감 4.0 ms). 항상 같이 돌아도 된다.
2. 이 GPU에서 돌고 있는 NeuralRx는 시작 시각 + 2.8 ms ≤ 마감일 때만 AI와 같이 돈다. 아니면
   그 NeuralRx가 끝날 때까지 AI 조각을 내주지 않는다.
3. 아직 conventional 결과를 기다리는 약한 TB가 이 GPU lane을 쓸 수 있으면, AI 조각은
   도착 + 4.0 − 2.8 ms 전에 끝나야 한다. 그 뒤에 NeuralRx가 시작되면 혼자 돌아야 하기 때문이다.

처음에는 AI가 무선 작업과 전혀 겹치지 않게 했다(strict). 16셀에서 AI TTFT p50이 1.2 s로
무너져서 이 규칙으로 바꿨다(5장).

## 3. 실행 구조

- 셀마다 conventional 프로세스 하나(cuBB의 셀별 스레드에 해당), cuPHY monolithic PUSCH.
- GPU마다 NeuralRx lane 하나, AI worker 하나.
- controller 하나(NeuralRx 배정, AI grant).
- 모든 프로세스는 MPS 아래에서 돌고 CLOCK_MONOTONIC을 공유한다. 수신 슬롯은 Sionna CDL과
  pyAerial 송신 체인으로 미리 만들어 GPU에 올려 두었다(TensorFlow는 timed 경로에 없음).

## 4. 부품 측정 (A100-SXM4-80GB, 단독)

| 부품 | p50 | p99 | 비고 |
|---|---|---|---|
| conventional rank 1 (wall) | 0.74 ms | 0.77 ms | GPU 커널 0.28 ms, 나머지는 host setup |
| conventional rank 2 MCS 12 (wall) | 0.78 ms | 0.81 ms | |
| NeuralRx fp16 (wall) | 2.06 ms | 2.11 ms | p99.9 3.9 ms |
| Qwen layer 단위(128토큰) | 0.31 ms | 0.33 ms (p99.9) | 512토큰 prefill 33 ms |

- rank 2에서 TB가 약 13 code block(MCS 13, 14 KB)을 넘으면 pyAerial PUSCH setup이 호출마다
  cudaMalloc/cudaFree를 해서 host가 최대 11 ms 멈춘다. production cuPHY driver는 이 버퍼를
  미리 잡아 둔다. 그래서 강한 UE는 MCS 12(12.5 KB)로 두었다.
- 약한 UE 대역에서 두 수신기의 결과(256 TB): conventional 171, NeuralRx 134, NeuralRx만 성공 9,
  conventional만 성공 46. P3에서 NeuralRx가 앞섰던 것은 비교 대상이 separable conventional이었기
  때문이다. monolithic cuPHY 수신기와 비교하면 NeuralRx가 추가로 살리는 TB는 적다.

## 5. 결과

설정: A100-SXM4-80GB 4장(nid200453, job 59103692), 실행당 UL 주기 4,000개(10 s), 앞 20주기
제외, 시드 2개 평균. AI는 Qwen2.5-1.5B prefill, BurstGPT 길이(최대 1,024토큰), GPU당 Poisson
도착, TTFT SLO 200 ms. 비교 정책의 AI는 모두 MPS 50% share다. 원자료
`results/backstop_slot/final_campaign_j59103692.json`, 그림 `final_tradeoff_j59103692.png`.

비교 방식:
- **Our Scheme**: 2.1 NeuralRx 규칙 + 2.2 AI 규칙.
- **우리 NeuralRx 규칙 + 고정 share**: NeuralRx는 같고, AI는 무선을 보지 않고 계속 돈다
  (YinYangRAN처럼 한 GPU를 고정 비율로 나눔).
- **두 수신기 항상 동시 + 고정 share**: 약한 TB마다 도착 즉시 NeuralRx(C165 구조).
- **두 수신기 도착 즉시, 마감 넘으면 버림 + 고정 share**: 위 방식에 conventional 성공 시 생략과
  마감 기반 drop을 더한 강한 기준선.
- **NeuralRx 없음 + 고정 share**.

### 16셀

| AI 부하 | 방식 | 마감 안 복호 | NeuralRx가 살린 TB | NeuralRx 실행 | NeuralRx 늦음 | SLO 안 AI 토큰/s | TTFT p50 |
|---|---|---|---|---|---|---|---|
| – | AI 없음 | 86.03% | 1,082 | 6,322 | 19 | – | – |
| 4 | **Our Scheme** | **86.03%** | **1,075** | 6,310 | 40 | 6,959 | 64 ms |
| 4 | 우리 NeuralRx 규칙 + 고정 share | 85.98% | 1,034 | 6,285 | 218 | 7,656 | 41 ms |
| 4 | 도착 즉시, 마감 drop + 고정 share | 85.10% | 490 | 15,414 | 502 | 6,855 | 63 ms |
| 4 | 항상 동시 + 고정 share | 84.50% | 114 | 16,030 | 12,485 | 6,823 | 66 ms |
| 4 | NeuralRx 없음 + 고정 share | 84.34% | 0 | 0 | 0 | 7,656 | 38 ms |
| 8 | **Our Scheme** | **86.04%** | **1,075** | 6,306 | 34 | 8,741 | 108 ms |
| 8 | 우리 NeuralRx 규칙 + 고정 share | 85.93% | 1,006 | 6,273 | 456 | 12,787 | 53 ms |
| 8 | 도착 즉시, 마감 drop + 고정 share | 85.06% | 462 | 14,942 | 991 | 10,190 | 90 ms |
| 8 | 항상 동시 + 고정 share | 84.51% | 112 | 15,706 | 12,450 | 9,787 | 97 ms |
| 8 | NeuralRx 없음 + 고정 share | 84.35% | 0 | 0 | 0 | 12,959 | 49 ms |

### 8셀

| AI 부하 | 방식 | 살린 TB | NeuralRx 실행 | conventional p99 | SLO 안 AI 토큰/s |
|---|---|---|---|---|---|
| 4 | **Our Scheme** | 415 | 2,865 | 1.11 ms | 7,669 |
| 4 | 우리 NeuralRx 규칙 + 고정 share | 418 | 2,867 | 1.16 ms | 7,656 |
| 4 | 도착 즉시, 마감 drop + 고정 share | 418 | 15,400 | 1.59 ms | 6,938 |
| 4 | 항상 동시 + 고정 share | 386 | 15,831 | 1.68 ms | 6,823 |
| 8 | **Our Scheme** | 418 | 2,871 | 1.16 ms | 12,643 |
| 8 | 우리 NeuralRx 규칙 + 고정 share | 418 | 2,867 | 1.19 ms | 12,972 |
| 8 | 도착 즉시, 마감 drop + 고정 share | 419 | 15,232 | 1.58 ms | 10,215 |
| 8 | 항상 동시 + 고정 share | 342 | 15,734 | 1.72 ms | 10,024 |

### 읽는 법

- **NeuralRx를 모든 약한 TB에 동시에 돌리면 16셀에서 무너진다.** NeuralRx 실행이 GPU 시간을
  넘쳐서 대부분 마감 뒤에 끝난다(살린 TB 114). 마감 drop을 넣어도 490이다. Our Scheme은
  conventional 결과를 보고 필요한 TB에만 돌려서 1,075를 살리고, NeuralRx 실행은 60% 적다.
- **8셀에서는 GPU 여유가 있어서** 도착 즉시 방식도 같은 수를 살린다. 대신 NeuralRx를 5.4배
  더 돌리고, 그만큼 conventional이 느려지고(p99 1.11 → 1.59 ms) SLO 안 AI 토큰이 Our Scheme보다
  10–19% 적다.
- **고정 share는 AI를 더 받는다(16셀, 부하 8에서 +46%).** 대신 AI 부하가 오르면 NeuralRx가
  늦어져 살린 TB가 줄어든다(1,082 → 1,034 → 1,006). Our Scheme은 AI 부하와 무관하게
  1,075를 유지한다. 즉 두 방식은 무선 복호와 AI 처리량을 다르게 맞바꾼다. Our Scheme은 AI
  부하가 무선 결과를 바꾸지 못하게 하는 쪽이다.
- UL indication 자체가 마감을 넘은 TB(conventional 결과가 4.0 ms 뒤)는 모든 방식에서 64,000개
  중 6–26개다. Python prototype의 host 꼬리이고, 항상 동시 방식이 가장 많다(24–26).

### 개발 과정에서 버린 것 (같은 job의 m1–m6)

- NeuralRx를 셀마다 프로세스로 두고 항상 동시에 돌리기(v1): 16셀에서 NeuralRx 대기열이 수백
  ms로 밀림.
- AI가 무선과 전혀 겹치지 않게 하기(strict): 16셀에서 SLO 안 AI 토큰 107/s, TTFT p50 1.2 s.
- MPS client priority로 AI를 낮은 우선순위에 두기: NeuralRx 지연 변화 없음.
- value rule 없이 co-run 게이트만: 16셀 부하 4에서 SLO 안 AI 토큰 3,804–4,405/s. value rule을
  넣은 뒤 6,959/s.

## 6. 새로운 점과 증거

| 새로운 점 | 가까운 선행 연구가 하는 것 | 이 설계가 하는 것 | 증거(ablation) |
|---|---|---|---|
| N1. NeuralRx를 마감 있는 rescue 경로로 둔다 | NeuralRx 연구(Cammerer 등, pyAerial 예제)는 수신기를 대체한다. 두 수신기를 같이 돌리는 구조(C165)는 모든 TB에 둘 다 돌린다 | conventional 결과를 기다리다가 실패하거나 최신 시작 시각이 되면 NeuralRx를 시작한다 | 도착 즉시 시작(start=arrival)과 비교 |
| N2. 실패한 code block 수로 NeuralRx 가치를 판단 | 없음. cuPHY는 per-CB CRC를 HARQ용으로만 낸다 | 1개만 실패한 TB에만 NeuralRx | value=0과 비교, 다른 채널·MCS에서 신호 재측정 |
| N3. 마감 기반 NeuralRx admission과 GPU 간 lane 공유 | 마감을 넘을 NeuralRx도 끝까지 돌린다 | 최신 시작 시각을 넘기면 시작하지 않고, 빈 lane이면 어느 GPU든 쓴다 | admit=0, lanes=partner와 비교 |
| N4. co-run bound로 AI와 무선을 같은 GPU에서 겹치게 허용 | YinYangRAN: GPU를 고정 비율로 나누고 1 s 이상 간격으로 조정. Concordia: 무선이 비었을 때만 다른 작업 | 겹칠 수 있는 모든 무선 작업이 co-run bound로 마감을 지킬 때만 AI 조각을 준다 | 고정 비율(20–70%)과 비교, AI가 무선이 비었을 때만 도는 방식(strict)과 비교 |
| N5. LLM prefill을 bound가 있는 GPU 단위로 나눠 틈마다 이어 실행 | LLM serving의 chunked prefill은 처리량용이고 bound가 없다 | layer × 128토큰 CUDA graph, KV cache를 GPU에 두고 다음 틈에 이어서 | 단위 GPU 시간 분포(p50 0.31, p99.9 0.33 ms) |

### 마감 안전성 (bound가 맞을 때)

가정: NeuralRx가 혼자 돌면 B(2.5 ms) 안에, AI와 같이 돌면 B_co(2.8 ms) 안에 끝나고, conventional은
AI와 같이 돌아도 마감보다 훨씬 먼저 끝난다(p99 1.5 ms). NeuralRx TB의 마감을 R + D, 최신 시작을
L = R + D − B라 하자.

주장: 시작이 허용된 NeuralRx(시작 t ≤ L)는 R + D 안에 끝난다.

- AI 조각은 두 경우에만 주어진다. (a) 그 GPU에서 돌고 있는 NeuralRx가 모두 t + B_co ≤ R + D를
  만족할 때, (b) conventional 결과를 기다리는 TB가 있으면 조각이 R + D − B_co 전에 끝날 때.
- t + B_co ≤ R + D이면, 겹치는 AI 조각이 몇 개든 NeuralRx는 t + B_co 안에 끝난다.
- t + B_co > R + D이면(늦게 시작), t 이전에 준 조각은 (b)에 따라 R + D − B_co < t 전에 끝났고,
  t 이후에는 (a) 때문에 조각을 주지 않는다. 따라서 혼자 돌고 t + B ≤ L + B = R + D 안에 끝난다.

그러므로 마감을 넘는 NeuralRx는 측정한 bound를 넘은 경우뿐이다. 실험에서 Our Scheme의 늦은
NeuralRx는 약 6,300회 중 34–40회(0.6%)이고, 같은 NeuralRx 규칙에 고정 비율 AI를 붙이면
218–456회다.

### 6.1 value rule은 다른 조건에서도 성립한다 (N2)

같은 규칙(conventional이 code block 1개만 실패하면 NeuralRx)을 새 채널·SNR·MCS에서 다시 측정했다
(job 59111015, 각 512 TB, `results/backstop_slot/raw/cbcrc_*`).

| 조건 | conventional 실패 | NeuralRx가 살린 TB | 규칙이 남긴 rescue | 규칙 적용 시 NeuralRx 실행 |
|---|---|---|---|---|
| CDL-D, Es/No −3.8~−3.2 dB (평가 조건, 1,024 TB) | 307 | 29 | 28 | 197 (−36%) |
| CDL-E | 166 | 17 | 17 | 107 (−36%) |
| CDL-D, Es/No −4.6~−2.4 dB | 228 | 5 | 4 | 33 (−86%) |
| MCS 4 (code block 3개), −6.8~−4.6 dB | 138 | 6 | 6 | 31 (−78%) |

네 조건 모두에서 rescue의 57개 중 55개가 "1개만 실패"에서 나왔다.

### 6.2 ablation (16셀, 시드 2개 평균)

한 번에 하나씩 빼고 같은 조건에서 돌렸다(`results/backstop_slot/novelty_evidence.json`).

| 뺀 것 | 살린 TB | NeuralRx 실행 | 늦은 NeuralRx | SLO 안 AI 토큰/s (부하 4 / 8) |
|---|---|---|---|---|
| 없음 (Our Scheme) | 1,075 | 6,310 | 40 | 6,959 / 8,741 |
| conventional 먼저 (N1; 도착 즉시 NeuralRx) | **545 (−49%)** | 15,851 | 46 | 3,397 / 3,580 |
| value rule (N2) | 1,084 | **9,724 (+54%)** | 62 | **4,101 / 1,590** |
| 마감 admission (N3) | 1,074 | 6,386 | 80 (×2) | 6,874 / 7,690 |
| lane 공유 (N3; 짝 GPU만) | 1,054 (−2%) | 5,691 | 34 | 7,040 / 8,766 |
| 최신 시작 시각에 시작 (conventional 실패 때만) | 1,068 | 6,304 | 51 | 6,651 / 7,308 |
| co-run 게이트 (N4; 고정 50%) | 1,034 / 1,006 | 6,285 | **218 / 456** | 7,656 / 12,787 |
| co-run 자체 (N4; 무선이 빌 때만 AI, Concordia식) | 1,068 | 6,304 | 72 | **901 / 470** |

- conventional을 먼저 보는 것(N1)이 rescue 수를 정한다. 빼면 절반이 된다.
- value rule(N2)은 rescue를 거의 잃지 않고 NeuralRx 실행을 줄여 AI 시간을 만든다. 빼면 AI가
  41–82% 줄어든다.
- co-run(N4)이 없으면 AI가 거의 돌지 못한다. co-run 게이트를 고정 비율로 바꾸면 AI는 늘지만
  늦은 NeuralRx가 5–13배가 되고 rescue가 4–6% 준다.
- 마감 admission과 lane 공유는 효과가 작다(늦은 NeuralRx 절반, rescue 2%).

### 6.3 AI 부하와 약한 셀 비율을 바꿔도 rescue가 유지된다

| 16셀 | Our Scheme | 고정 30% | 고정 50% | 고정 70% | AI 없음 |
|---|---|---|---|---|---|
| AI 4 req/s: 살린 TB | 1,075 | 1,042 | 1,034 | 1,016 | 1,082 |
| AI 8 req/s | 1,075 | 1,033 | 1,006 | 926 | |
| AI 12 req/s | 1,074 | 820 | 787 | 803 | |
| AI 12 req/s: 늦은 NeuralRx | 36 | 1,337 | 1,368 | 1,358 | 19–24 |
| AI 12 req/s: SLO 안 AI 토큰/s | 4,987 | 10,935 | 17,677 | 19,457 | |

| 16셀, AI 8 req/s | Our Scheme | 우리 NeuralRx 규칙 + 고정 50% | 도착 즉시 + drop + 고정 50% | AI 없음 |
|---|---|---|---|---|
| 약한 셀 25%: 살린 TB | 413 | 393 | 374 | 417 |
| 약한 셀 75%: 살린 TB | 1,339 | 1,104 | 426 | 1,316 |
| 약한 셀 75%: SLO 안 AI 토큰/s | 3,908 | 12,470 | 10,120 | |

고정 비율은 어떤 비율을 골라도 AI 부하가 오르면 rescue를 잃는다(12 req/s에서 24–27%).
YinYangRAN은 1 s 이상 간격으로 비율을 바꾸는데, 무선 부하가 일정한 이 실험에서는 그 선택도 고정
비율 중 하나가 된다. Our Scheme은 부하와 약한 셀 비율에 관계없이 AI 없을 때 수준을 유지한다.
대가는 AI다. 무선이나 AI 부하가 높으면 SLO 안 AI 토큰이 고정 비율보다 크게 적다.

### 6.4 AI 요청 admission을 모든 방식에 넣으면

AI 요청이 도착할 때, 앞에 쌓인 단위 수를 최근 처리 속도로 나눠 SLO(200 ms) 안에 끝날지 예측하고
못 끝나면 받지 않는다. 기존 Backstop의 AI admission과 같은 생각이다. 모든 방식에 똑같이 넣었다
(`results/backstop_slot/admission_evidence.json`).

| 16셀 | 살린 TB | 늦은 NeuralRx | SLO 안 AI 토큰/s (admission 전 → 후) | 거절 비율 |
|---|---|---|---|---|
| Our Scheme, AI 8 | 1,072 | 42 | 8,741 → 10,520 | 10% |
| 고정 50%, AI 8 | 1,002 | 387 | 12,787 → 12,806 | 1.5% |
| 고정 30%, AI 8 | 1,030 | 222 | 10,862 → 11,660 | 5% |
| Our Scheme, AI 12 | 1,071 | 64 | 4,987 → 13,564 | 16.5% |
| 고정 50%, AI 12 | 853 | 1,144 | 17,677 → 18,882 | 3.5% |
| 고정 30%, AI 12 | 928 | 786 | 10,935 → 15,914 | 10% |
| Our Scheme, 약한 셀 75% | 1,320 | 74 | 3,908 → 7,051 | 20% |
| 고정 50%, 약한 셀 75% | 1,164 | 869 | 12,470 → 12,652 | 1.8% |

- Our Scheme은 AI를 받을 수 있는 시간이 무선 상태에 따라 바뀌므로, 받을 수 없는 요청을 미리
  거절하는 것이 크게 도움이 된다(AI 12에서 2.7배).
- 남은 차이: 같은 부하에서 고정 30%보다 AI가 10–15% 적고, 살린 TB는 4–15% 많다.
- **AI 없을 때 rescue의 99% 이상을 지키는 방식은 Our Scheme뿐이다.** 고정 비율은 AI 4 req/s에서도
  96%가 최선이고(30%), 비율을 낮추면 AI가 더 오래 밀려 있어서 오히려 더 잃는다(개발 실행에서
  20%, AI 8 req/s일 때 rescue 300).

### 6.5 새 시드로 다시 (holdout)

bound와 규칙은 앞의 실행으로 정했다. 설정을 바꾸지 않고 처음 쓰는 시드 3, 4(AI 도착과 수신 슬롯
구간이 모두 다름)로 16셀, AI 8 req/s, AI admission을 다시 돌렸다
(`results/backstop_slot/holdout_evidence.json`).

| 방식 | 살린 TB / AI 없음 (시드 3) | (시드 4) | 늦은 NeuralRx | SLO 안 AI 토큰/s |
|---|---|---|---|---|
| Our Scheme | 978 / 983 (99.5%) | 853 / 862 (99.0%) | 22 / 18 | 10,404 / 11,189 |
| 우리 NeuralRx 규칙 + 고정 30% | 94.7% | 96.1% | 289 / 135 | 11,426 / 11,976 |
| 우리 NeuralRx 규칙 + 고정 50% | 93.0% | 92.2% | 372 / 395 | 11,963 / 12,659 |
| 도착 즉시 + drop + 고정 50% | 44% | 40% | 742 / 659 | 11,045 / 11,741 |
| AI 없음 | 983 | 862 | 22 / 15 | – |

앞의 결과와 같은 모양이다. Our Scheme만 AI 없을 때 rescue의 99% 이상을 지키고, 늦은 NeuralRx도
AI 없을 때와 같다. SLO 안 AI 토큰은 고정 30%보다 7–9% 적다.

## 7. 한계

- Python prototype: conventional host setup에 0.07% 정도 4.0 ms를 넘는 꼬리가 있다(모든 정책에
  같이 나타남).
- NeuralRx 모델은 공개 pyAerial 모델(1 layer, QPSK)이다. 2-UE MU-MIMO NeuralRx(NVlabs `nrx_rt`,
  사전학습 가중치 있음)를 쓰려면 외부 코드·가중치 사용 승인이 필요하다.
- 셀당 슬롯마다 UE 하나, fronthaul 없음, 미리 생성한 IQ.
- value rule(실패 code block 1개일 때만 NeuralRx)은 이 채널·MCS·모델에서 측정한 관계다. 다른
  모드에서는 다시 측정해야 한다.
- bound(NeuralRx 2.5/2.8 ms, AI 단위 0.35 ms)는 개발 실행의 분포에서 정했다. 새 시드 2개로 다시
  확인했지만(6.5) 같은 노드 종류(A100 80GB)만 썼다.
- NeuralRx rescue 수 자체는 적다(16셀에서 약한 TB 31,840개 중 약 1,075개, 3.4%). 공개 NeuralRx 모델이 monolithic cuPHY보다
  크게 낫지 않기 때문이다. 더 좋은 NeuralRx 모델이면 같은 구조에서 rescue 차이가 커진다.
