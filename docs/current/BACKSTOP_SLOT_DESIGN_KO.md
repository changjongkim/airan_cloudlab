# Backstop 슬롯 단위 스킴: 설계 (현재 구현 기준)

기준일: 2026-09-30. 코드: `scripts_for_node/backstop_slot/`. 이 문서는 지금 구현된 스킴을 그대로 적는다.
결과는 [BACKSTOP_SLOT_RESULTS_KO.md](BACKSTOP_SLOT_RESULTS_KO.md), 기존 연구와의 차이는
[BACKSTOP_SLOT_DIFFERENTIATION_KO.md](BACKSTOP_SLOT_DIFFERENTIATION_KO.md)에 있다.

## 0. 요약

- 모든 셀의 TB가 2.5 ms마다 동시에 도착한다. **모든 TB는 기존 수신기(cuPHY)로 먼저 복호**하고, 그 결과는
  도착 후 4.0 ms 안에 나와야 한다(L1 마감).
- 기존 수신기가 실패한 TB 가운데 **코드 블록이 1개만 틀린 TB만 NRx로 다시 복호**한다. NRx가 rescue 마감
  (6.5 ms, 32셀은 11.5 ms) 안에 성공하면 그 TB의 재전송이 없어진다.
- AI(Qwen prefill)는 0.3–1.1 ms짜리 단위로 나눠 같은 GPU에서 무선과 같이 돈다. controller가 GPU마다
  "지금 얼마 동안, 어느 크기의 단위까지" 돌려도 되는지를 무선 상태를 보고 정한다.
- 논문 현재판의 45 ms(NRx 마감)와 153 ms(복구 마감)는 이 스킴에 없다(10장).

## 1. 설정

| 항목 | 값 |
|---|---|
| 프레임 | TDD DDDSU, 30 kHz SCS, 100 MHz(273 PRB). 셀마다 UL 슬롯이 2.5 ms에 한 번 |
| 도착 | TDD 셀은 서로 동기화되어 있어서 모든 셀의 UL 슬롯이 같은 순간에 도착 |
| 서버 | A100-SXM4-80GB 4장, MPS |
| 셀 수 | 16, 24, 32 (초기 실험은 8, 16) |
| 강한 UE (셀의 50%) | rank 2, 16QAM MCS 12, CDL-C, Es/No 20–24 dB, TB 12,549바이트. 기존 수신기만 |
| 약한 UE (셀의 50%) | rank 1, QPSK MCS 7, CDL-D, Es/No −3.8 ~ −3.2 dB, TB 3,777바이트(코드 블록 4개). NRx 대상 |
| 셀당 UE | UL 슬롯마다 전 대역 PUSCH 하나 |
| NRx 모델 | 공개 pyAerial NRx(1 layer, QPSK, 4 RX), TensorRT fp16 |
| AI | Qwen2.5-1.5B prefill, BurstGPT 길이(최대 1,024토큰), Poisson 도착, TTFT SLO 200 ms |
| 수신 슬롯 | pyAerial 송신 체인 + Sionna CDL로 미리 만들어 GPU에 올려 둠 |

## 2. 마감 두 개

TB마다 마감이 두 개 있다. 둘 다 "데이터 도착(슬롯 끝)"부터 잰다.

| 마감 | 값 | 무엇이 끝나야 하나 | 설정 키 |
|---|---|---|---|
| L1 마감 | 4.0 ms | 기존 수신기의 CRC 결과 | `deadline_ms` |
| rescue 마감 | 6.5 ms (16·24셀), 11.5 ms (32셀) | NRx의 CRC 결과 | `rescue_deadline_ms` |

**L1 마감의 근거.** testMAC의 UL indication 한계는 슬롯 시작(T0) + 4.5 ms다. IQ는 슬롯이 끝나는 T0 + 0.5 ms에
준비되므로 남는 시간이 4.0 ms다.

**rescue 마감의 근거.** 기존 수신기가 실패한 TB는 재전송된다. NRx가 재전송 grant를 보내기 전에 성공하면
재전송을 취소할 수 있다. DDDSU에서 UL 슬롯은 T0, T0+2.5, T0+5.0, T0+7.5, … 에 온다.

| rescue 마감 (도착 기준) | 결정 시각 | 재전송이 실리는 UL 슬롯 | 비고 |
|---|---|---|---|
| 5.0 ms | T0+5.5 | T0+7.5 | L2 준비 시간 1–2 슬롯을 둔 보수적인 값 |
| 6.5 ms | T0+7.0 | T0+7.5 | 바로 앞 S 슬롯에서 K2=1로 grant, L2 준비 시간 0으로 본 값 |
| 9.0 ms | T0+9.5 | T0+10.0 | 재전송이 UL 기회 하나(2.5 ms) 늦어짐. 실험하지 않음 |
| 11.5 ms | T0+12.0 | T0+12.5 | 재전송이 UL 기회 둘(5 ms) 늦어짐 |

rescue 마감은 설정값이다. K2와 L2 준비 시간은 배포마다 다르므로 값 하나를 정답으로 두지 않고 5.0, 6.5,
11.5 ms에서 실험했다. 상한은 HARQ process 16개가 한 바퀴 도는 약 40 ms로 보았는데, 이 값은 3GPP 규격으로
확인하지 않은 우리 계산이다.

## 3. 구성

| 프로세스 | 수 | 하는 일 | 파일 |
|---|---|---|---|
| 기존 수신기 | 셀마다 1 | 2.5 ms마다 자기 셀 TB를 cuPHY PUSCH로 복호, CRC와 틀린 코드 블록 수를 기록 | `conv_worker.py` |
| NRx lane | GPU마다 1 | controller가 준 TB 하나를 NRx로 복호 | `nrx_lane.py` |
| AI worker | GPU마다 1 | Qwen prefill 단위를 허락받은 만큼 실행 | `ai_worker2.py` |
| controller | 1 | NRx 배정, AI 조각 허락, AI 요청 배치 | `controller.py` |

- 모든 프로세스는 `/dev/shm`의 공유 상태 파일 하나(`slot_state.py`)와 CLOCK_MONOTONIC으로 맞춘다. controller는
  이 상태를 계속 읽는 루프이고 한 바퀴가 평균 15–57 µs다(최대 약 1.2 ms).
- 수신 슬롯은 그 셀의 기존 수신기가 있는 GPU 메모리에 있다. NRx lane은 CUDA IPC로 그 메모리를 열어 두고,
  배정받으면 자기 GPU로 P2P 복사한 뒤 복호한다. 그래서 어느 GPU의 lane이든 어느 셀의 TB든 받을 수 있다.
- 약한 셀의 "짝 GPU"는 기존 수신기 GPU의 다음 GPU다. 같은 TB의 두 수신기가 되도록 다른 GPU에서 돌게 하려는
  기본 배치다.

## 4. TB 하나의 흐름

```
도착 ──► 기존 수신기 시작 (모든 TB)
          │
          ├─ CRC 통과 ───────────────────────────► 확정. NRx 안 함
          │
          └─ CRC 실패 ─ 틀린 코드 블록 2개 이상 ──► 재전송 (NRx 안 함)
                      └ 틀린 코드 블록 1개 ───────► NRx 후보
                                                     │
                              빈 lane을 최신 시작 시각 전에 얻음 ─► NRx 실행
                              │                                      ├ rescue 마감 안에 통과 ─► 재전송 취소
                              │                                      └ 실패 또는 늦음 ────────► 재전송
                              └ 못 얻음 ─────────────────────────────► 재전송
```

24셀 한 실행(약한 TB 47,760개)에서 각 갈래의 크기: 기존 수신기 통과 70%, 코드 블록 2개 이상 실패 10.6%,
NRx 실행 18.6%, lane을 못 얻음 0.9%, NRx가 살림 2.9%.

## 5. NRx 규칙

controller가 매 바퀴마다 약한 TB에 대해 적용한다(`controller.py`의 `rescue_value` preset).

1. **기존 수신기가 통과하면 NRx를 하지 않는다.**
2. **실패하면 틀린 코드 블록 수를 본다.** 1개 이하면 후보가 되고, 2개 이상이면 포기한다(`nrx_max_cb_fail=1`).
   NRx가 살린 TB 57개 중 55개가 "1개만 틀린 경우"였다는 측정에 따른 규칙이다.
3. **최신 시작 시각 = 도착 + rescue 마감 − NRx 시간 한도.** 예: 6.5 − 2.8 = 3.7 ms.
4. **후보는 최신 시작 시각이 이른 순서로 빈 lane에 배정한다.** lane을 고르는 순서는 (a) 일하는 lane이 적은
   GPU, (b) AI가 돌고 있지 않은 GPU, (c) 짝 GPU다.
5. **최신 시작 시각을 넘긴 후보는 포기한다.** 늦게 끝날 NRx가 다른 TB까지 느리게 만들기 때문이다.
6. **NRx가 rescue 마감 뒤에 끝나면 그 결과는 쓰지 않는다.** rescue는 CRC 통과와 payload 일치를 둘 다
   만족하고 마감 안에 끝난 경우만 센다(`summarize_slot.py`).

**기존 수신기 결과가 최신 시작 시각까지 안 나온 TB는 NRx 없이 재전송으로 간다.** 코드에는 "결과 없이 NRx를
시작"하는 경로(`start=fail_or_latest`)가 있지만, 5번 규칙이 켜져 있으면 같은 바퀴에서 후보가 되자마자 최신
시작을 넘긴 것으로 처리되어 포기된다. job 59124351의 166회 실행에서 이 경로로 시작한 NRx는 0건이다. 기존
수신기는 p99 2.1–2.4 ms에 끝나고 최신 시작은 3.7 ms 이후라서 해당하는 TB 자체도 드물다.

## 6. AI 규칙

### 6.1 AI 단위

Qwen prefill을 "chunk 크기 × layer 하나" 단위로 나누고 단위마다 CUDA graph로 만들어 두었다
(`qwen_units.py`). chunk 크기는 128, 512, 1,024토큰이고 KV cache 하나를 같이 쓴다. 한 요청 안에서 chunk마다
크기를 바꿔도 결과가 같다(HF와 argmax 일치, logit 차 0.035 이하).

| chunk 크기 | layer 단위 하나의 GPU 시간 (단독) | 512토큰 prefill |
|---|---|---|
| 128토큰 | 0.31 ms | 33.2 ms |
| 512토큰 | 0.67 ms | 18.5 ms |
| 1,024토큰 | 1.03 ms | – (1,024토큰에 28.7 ms) |

큰 단위는 토큰당 빠르지만 한 번 시작하면 오래 GPU를 잡는다. 그래서 크기를 무선 상태로 고른다.

### 6.2 조각 허락

AI worker는 MPS 제한 없이(100%) 무선과 같은 GPU에서 돈다. 대신 controller가 허락한 조각 안에서만 단위를
제출한다. controller는 GPU마다, 앞 조각이 끝났고 할 일이 있을 때 아래를 본다(`backstop_units`).

큰 크기부터(1,024 → 512 → 128) 다음을 모두 만족하는 첫 크기를 고른다. B_co(c)는 "크기 c의 AI 단위와 같이
돌 때의 NRx 시간 한도"다(7장).

| 조건 | 내용 |
|---|---|
| A. 돌고 있는 NRx | 이 GPU의 lane에서 도는 NRx마다 `NRx 시작 시각 + B_co(c) ≤ 그 TB의 rescue 마감`. 아니면 크기 c 금지 |
| B. 기존 수신기와 겹침 | c가 "기존 수신기와 겹쳐도 되는 크기"가 아니면, 이 GPU의 셀이 복호 중일 때 금지하고 조각은 다음 도착 전에 끝나야 함 |
| C. 곧 시작될 NRx | 이 GPU가 짝인 약한 셀 중 아직 NRx가 시작될 수 있는 TB(기존 수신기 결과 대기 또는 lane 대기)가 있으면, 조각은 `그 TB의 rescue 마감 − B_co(c)` 전에 끝나야 함 |
| D. 길이 | 조각은 최대 3.0 ms. 남은 길이에 그 크기의 layer 단위가 2개 이상 들어가야 함(128토큰은 0.36 ms 이상) |
| E. 양보 (32셀만) | lane을 기다리는 NRx 후보가 하나라도 있으면 어느 GPU에도 조각을 주지 않음 |

어떤 크기도 통과하지 못하면 이번 바퀴에는 그 GPU에 AI를 주지 않는다.

### 6.3 AI worker의 동작 (`ai_worker2.py`)

- 허락은 (길이, 최대 chunk 크기)로 온다. 50 µs 안에 시작하지 못하면 그 허락은 버린다.
- 단위의 시간 한도 합이 허락 길이를 넘지 않는 만큼 단위를 연속으로 제출한다.
- chunk를 새로 시작할 때 크기는 "남은 prompt를 한 번에 덮는 가장 작은 크기" 이하에서, 이번 조각에 layer 단위가
  2개 이상 들어가는 가장 큰 크기로 고른다. 허락된 최대 크기를 넘지 않는다.
- 진행 중인 chunk가 이번에 허락된 크기보다 크면 그 요청은 다음 허락을 기다린다.

### 6.4 AI 요청 받기와 배치

- **받기(admission).** 요청이 오면 `(대기 중인 토큰 + 이 요청의 토큰) ÷ 최근 처리 속도`로 끝나는 시각을 예측하고,
  200 ms를 넘으면 받지 않는다. 비교 방식에도 똑같이 넣었다.
- **배치(dispatch).** controller가 서버 전체의 요청을 받아 예상 완료가 가장 이른 GPU로 보낸다
  (`--ai-dispatch global`). GPU마다 무선 상태가 달라 AI 속도가 다르므로 그 차이를 따라간다. 비교 방식에도
  같은 배치를 넣었다.

## 7. 시간 한도

A100 80GB에서 잰 값으로 정했다.

| 셀 수 | NRx 혼자 (B) | 128토큰과 같이 | 512토큰과 같이 | 1,024토큰과 같이 | 기존 수신기와 겹쳐도 되는 크기 | 양보 | rescue 마감 |
|---|---|---|---|---|---|---|---|
| 16 | 2.5 ms | 3.1 | 3.9 | 4.3 | 128, 512 | 끔 | 6.5 ms |
| 24 | 2.8 ms | 3.8 | 4.5 | 5.2 | 128, 512 | 끔 | 6.5 ms |
| 32 | 2.8 ms | 4.1 | 4.7 | 5.3 | 128 | 켬 | 11.5 ms |

- GPU당 NRx lane은 1개다. 2개로 하면 처리량은 1.5배가 되지만 TB 하나의 시간이 2.08 → 2.72 ms로 늘어 최신 시작
  시각이 당겨진다. 24·32셀에서 1개가 더 나았다.
- "기존 수신기와 겹쳐도 되는 크기"는 측정으로 정했다. 24셀에서 512토큰 단위가 겹쳐도 기존 수신기 p99.9가
  128토큰일 때와 같았고(2.97 대 3.04 ms), 32셀에서는 512토큰에서 L1을 놓친 TB가 38 → 67개로 늘었다. 32셀
  (lane 2개)에서 1,024토큰 단위를 제한 없이 겹치면 L1을 놓친 TB가 1,157개였다.
- 한도는 측정한 p99를 0.1 ms 단위로 올린 값이다. 24·32셀의 1,024토큰 한도(5.2, 5.3 ms)는 lane 1개로 직접
  재지 않고 정한 값이다(결과 문서 3장).

예시(24셀, rescue 마감 6.5 ms): NRx가 1.5 ms에 시작했으면 512토큰은 1.5 + 4.5 = 6.0 ≤ 6.5라 허락, 1,024토큰은
1.5 + 5.2 = 6.7 > 6.5라 금지다. NRx가 2.5 ms에 시작했으면 128토큰(2.5 + 3.8 = 6.3)만 허락된다.

## 8. 마감이 지켜지는 근거와 틈

**L1 마감.** 기존 수신기는 도착 즉시 시작하고 앞에 끼어드는 작업이 없다. 느려지는 원인은 같이 도는 NRx와
AI다. AI는 조건 B로 큰 단위를 막는다. 강제로 끊는 장치는 없고, 4.0 ms를 넘긴 TB는 "L1을 놓침"으로 센다.

**rescue 마감.** 시작이 허락된 NRx(시작 ≤ 최신 시작)는 혼자 돌면 B 안에 끝난다. AI와 겹치는 경우는 조건 A를
통과한 크기뿐이므로 B_co(c) 안에 끝난다. 아직 시작하지 않은 NRx는 조건 C로 "혼자 돌 여유"를 남긴다.

**이 근거가 기대는 것과 알려진 틈.**

1. 한도는 실측 p99다. 넘으면 NRx 결과를 버리거나 L1을 놓친 것으로 집계한다. 보장이 아니라 측정에 맞춘 값이다.
2. **AI 조각은 실제로는 허락받은 길이보다 오래 걸린다.** 단위의 시간 한도는 혼자 돌 때 잰 값인데, 무선과 같이
   돌면 AI도 느려진다. 조각의 실제 시간은 한도 합의 1.0–1.4배(중앙값), 1.5–2.1배(p99)였다(중앙값 기준 16셀
   +0.0 ms, 24셀 +0.3–0.4 ms, 32셀 +1.1 ms). 따라서 조건 B·C의 "~ 전에 끝나야 함"은 정확히 지켜지지 않는다. 조건 A는 조각
   길이와 무관하므로 영향이 없다. 결과의 L1·rescue 수치는 이 상태로 잰 것이다.
3. 조건 C는 "짝 GPU" 기준이다. 실제 배정은 어느 GPU의 lane으로도 갈 수 있다.
4. L1을 놓친 TB 0.01–0.05%는 Python prototype의 host 꼬리이고 AI가 없어도 나온다.

2번은 이번 문서 정리 중에 실행 기록에서 확인했다. 고치려면 AI 단위 한도를 "같이 돌 때"로 다시 재서 넣으면
된다. 스킴은 바꾸지 않았다.

## 9. 코드와 실행

| 파일 | 역할 |
|---|---|
| `make_config.py` | 실행 설정 JSON 생성 (셀 배치, 마감, 한도, 정책) |
| `launch_slot.py`, `run_slot.sh` | MPS 시작, 프로세스 실행, 결과 수집 |
| `conv_worker.py`, `slot_radio.py`(`ConvPath`) | 기존 수신기 |
| `nrx_lane.py`, `slot_radio.py`(`NrxPath`) | NRx lane |
| `controller.py` | NRx 규칙(5장), AI 조각 허락(6.2), AI 요청 배치(6.4) |
| `qwen_units.py`, `ai_worker2.py`, `ai_arrivals.py` | AI 단위, AI worker, 요청 도착 |
| `cell_ring.py`, `slot_state.py`, `ul_profiles.py` | 슬롯 메모리, 공유 상태, UE 설정 |
| `summarize_slot.py` | TB별 확정과 집계 |
| `run_matrix.sh` | 설정 여러 개를 이어서 실행 |

설정 키와 문서 용어:

| 문서 용어 | 설정 키 / 옵션 |
|---|---|
| L1 마감, rescue 마감 | `deadline_ms`, `rescue_deadline_ms` (`--rescue-deadline-ms`) |
| NRx 시간 한도 B | `nrx_bound_ms` (`--nrx-bound-ms`) |
| B_co(c), 겹쳐도 되는 크기, 양보 | `ai_unit_gating` (`--unit-gating "128:3.8,512:4.5,1024:5.2/conv=128,512"`, 양보는 끝에 `+yield`) |
| 틀린 코드 블록 수 기준 | `nrx_max_cb_fail` |
| NRx 규칙 묶음 | `nrx_policy=rescue_value` |
| AI 방식 | `ai_policy`: `backstop_units`(Our Scheme), `static`(고정 비율), `none` |
| AI 단위 크기 | `--ai-chunks 128,512,1024` |
| AI 요청 받기, 배치 | `--ai-admission 1`, `--ai-dispatch global` |

실행(24셀 최종 구성 한 번, 약 40초):

```bash
salloc -N 1 -C "gpu&hbm80g" -q interactive -A m5320_g -t 04:00:00 --no-shell
srun --jobid=$J --overlap -N1 -n1 --gpus-per-node=4 bash -c '
  SLOT_TAG=demo SLOT_PERIODS=4000 SLOT_AI_RATE=20 \
  SLOT_EXTRA="--seed 82000 --ai-admission 1 --lanes-per-gpu 1 --nrx-bound-ms 2.8 --nrx-bound-corun-ms 3.8 \
    --ai-chunks 128,512,1024 --ai-max-piece-ms 3.0 --rescue-deadline-ms 6.5 --ai-dispatch global \
    --static-mps-pct 100 --unit-gating 128:3.8,512:4.5,1024:5.2/conv=128,512" \
  SLOT_MATRIX="c24:rescue_value:backstop_units" bash scripts_for_node/backstop_slot/run_matrix.sh'
```

`--gpus-per-node=4`가 없으면 CUDA error 100이 난다. interactive QOS는 사용자당 job 하나다.

## 10. 논문 현재판(기존 스킴)과 다른 점

| | 논문 현재판 | 이 스킴 |
|---|---|---|
| 도착 | 180 ms마다 TB 4개 | 2.5 ms마다 셀 수만큼(16–32개) |
| 마감 | NRx 45 ms, 복구 153 ms (시험 시간표의 값) | L1 4.0 ms, rescue 6.5/11.5 ms (프로토콜 타이밍에서 나온 값) |
| 먼저 도는 수신기 | NRx | 기존 수신기 |
| 나중에 도는 수신기 | 기존 수신기가 NRx 실패를 복구 | NRx가 기존 수신기 실패를 살림 |
| 나중 수신기를 돌리는 조건 | NRx 실패 시 항상 | 코드 블록 1개만 틀렸을 때 |
| 기다리는 동안 남겨 두는 시간 | 복구 시간(남은 TB 수 × 25 ms) | NRx 시간(TB마다 최신 시작 시각) |
| AI 단위 | prefill 한 번 전체(35–75 ms) | layer × chunk 단위(0.3–1.1 ms), 크기 3종 |
| AI와 무선 | 시간으로 나눔 | 같이 돌되 단위 크기와 시점을 제한 |
| AI 시간 계산 | 153 − 지금 − 남은 수 × 25 | 6.2의 조건 A–E |

순서를 뒤집은 이유는 측정이다. NRx는 TB 하나에 2.08 ms가 들고 GPU 하나에서 초당 약 480개(lane 여러 개로도
약 880개)가 한계다. 16셀의 약한 TB는 초당 3,200개라서 전부 NRx로 먼저 돌릴 수 없다. 기존 수신기는 TB 하나에
0.74 ms다.

## 11. 구현되지 않은 것

- TB 종류별 시간 한도. 지금은 TB가 두 종류뿐이라 NRx 한도가 숫자 하나다. PRB 수와 MCS가 바뀌면 한도도 달라져야 한다.
- 슬롯당 여러 UE, UMa SINR 분포, 링크 적응, HARQ 재전송이 다음 슬롯 부하로 돌아오는 것.
- MU-MIMO NRx(NVlabs `nrx_rt`). 외부 코드와 가중치 사용 승인이 필요하다.
- 실제 L2. rescue 마감은 결정 시각만 기록한다.
- 같이 돌 때의 AI 단위 한도(8장 2번).

## 12. 적용 범위 (GPU당 셀 수)

- 이 설계가 고정 비율보다 나은 범위는 GPU당 6–8셀이다(결과 문서 8장).
- GPU당 10셀 이상에서는 기존 수신기가 주기 대부분을 써서, 가장 작은 AI 단위도 겹치면 L1 목표를 넘는다. AI를
  100%로 같이 돌리는 지금 방식으로는 넣을 틈이 없고, AI 비율을 낮게 묶는 고정 30%가 더 낫다. 높은 밀도에서 쓰려면
  "AI 비율을 묶은 채로 조각을 허락"하는 방식이 필요하다. 구현하지 않았다.
- 셀마다 프로세스 하나를 쓰므로 GPU당 프로세스가 약 15개를 넘으면 MPS에서 실행되지 않는다(GPU당 셀 13개 이하).
- 여러 셀을 cuPHY 호출 하나로 묶는 worker(`conv_group_worker.py`, `cellgroup_radio.py`)를 만들어 시간을 쟀으나
  더 느렸다. 실행 경로에는 연결하지 않았다.

## 13. v3에서 바뀐 것 (2026-10-01)

1–12장은 v2 기준이다. v3 실험(결과 문서 9장, 과정은 [BACKSTOP_SLACK_BUDGET_PLAN_KO.md](BACKSTOP_SLACK_BUDGET_PLAN_KO.md))
뒤의 최종 구성은 아래와 같다. v2 설정은 그대로 돌아간다. 새 기능은 모두 설정 키로 켠다.

### 13.1 최종 구성에 들어간 것

| 바뀐 것 | 내용 | 설정 |
|---|---|---|
| AI의 GPU 비율 제한 | AI worker를 MPS 70%로 띄운다. 겹칠 때 NRx가 덜 늦어져서(1,024토큰 5.2 → 4.0 ms) 큰 단위를 더 자주 허락할 수 있다. NRx 시간 한도는 이 조건에서 다시 잰 값을 쓴다 | `--gated-mps-pct 70` |
| 기존 수신기와 겹쳐도 되는 크기 | 측정으로 넓혔다: GPU당 4·6셀은 128/512/1,024 모두, 8셀은 128/512 | `--unit-gating ".../conv=128,512,1024"` |
| 묶음 크기 선택 | controller가 최근 허락의 70% 이상에서 허용한 가장 큰 크기로 묶음을 시작한다. 진행 중인 묶음은 단위 하나만 들어가도 이어서 돌린다 | `--ai-chunk-choice sustained` |

24셀의 NRx 시간 한도(AI 70%): 혼자 2.8 ms, 128토큰과 3.5, 512토큰과 3.9, 1,024토큰과 4.1 ms.
16셀: 2.5 / 2.9 / 3.4 / 3.4. 32셀: 2.8 / 4.0 / 4.5 / 4.6.

### 13.2 구현했고 조건에 따라 쓰는 것

| 기능 | 내용 | 설정 | 쓰는 조건 |
|---|---|---|---|
| 겹침 예산 | 주기마다 기존 수신기가 늦어져도 되는 양을 예산으로 두고, 크기 c의 AI가 겹친 시간 × `alpha[c]`만큼 쓴다 | `--conv-budget alone=2.8/margin=0.2/1024:1.0` | 32셀의 1,024토큰 |
| AI 단위 한도 자동 보정 | worker가 조각의 실제 시간을 보고 단위 한도를 1.0–3.0배로 키운다 | `--ai-adaptive-bound 1` | 기본은 끔 (AI −10%) |
| NRx를 주 수신기로 쓰는 셀 | 그 셀의 모든 TB를 도착 즉시 NRx로, rescue 마감까지 | `--nrx-primary-cells N` (`controller3.py`) | NRx 수요 실험 |
| 대기열 마감 확인 | lane을 기다리는 NRx와 다음 주기의 주 수신 TB가 마감을 지킬 때만 AI 허락 | `--unit-gating ".../conv=...+queue"` | 주 수신 셀이 있을 때 |
| 둘째 lane을 마감으로 판단 | 한 GPU에서 NRx 두 개를 같이 돌려도 둘 다 마감을 지킬 때만 둘째 lane 사용 | `--nrx-flags second_lane=deadline --nrx-bound-busy 2.8,4.0` | lane 2개일 때 |
| AI 여러 종류 | 종류마다 따로 줄을 서고, 조각 안에서 가장 급한 종류의 들어가는 단위를 고른다 | `--ai-classes chat:1.5B:6:200,small:0.5B:6:100,batch:1.5B` (`ai_worker3.py`) | 여러 종류 실험 |
| 부분 부하 | 셀이 슬롯마다 TB를 보낼 확률 | `--activity-prob 0.5 --activity-mode bursty` | 부분 부하 실험 |

### 13.3 시도했고 쓰지 않는 것

- NRx 여유를 겹친 시간에 비례해 깎기(`+nrxbudget`): 늦은 NRx가 7–40배가 된다.
- NRx 후보를 코드 블록 2개 이상 실패까지 넓히기(`--nrx-max-cb-fail 4`, `rank=value`): 살린 TB −14%.
- 여러 셀을 호출 하나로 복호(`--conv-runtime group`): 호출 하나의 코드 블록이 약 13개를 넘으면 무너진다.

### 13.4 코드

| 파일 | 역할 |
|---|---|
| `controller3.py` | v3 controller: `controller.py`의 기능 + 주 수신 셀, 대기열 마감 확인, 둘째 lane 판단, 부분 부하 |
| `ai_worker3.py` | 여러 종류 AI worker |
| `activity.py` | 부분 부하에서 어느 셀이 어느 주기에 TB를 보내는지 |
| `conv_group_worker.py`, `cellgroup_radio.py` | 여러 셀 묶음 복호 (쓰지 않음) |
| `analyze_v3.py`, `analyze_runs.py`, `analyze_classes.py`, `analyze_partial.py`, `analyze_v3cal.py`, `analyze_load.py` | v3 분석 |
| `run_state/backstop_slot/v3_*.sh` | v3 실험 스크립트 |

## 14. v4·v5에서 바뀐 것 (2026-10-01, job 59164118)

근거와 수치는 [BACKSTOP_V4_VERIFICATION_KO.md](BACKSTOP_V4_VERIFICATION_KO.md)에 있다.

| 항목 | 내용 | 코드 |
|---|---|---|
| NRx 입력 | LS 추정값을 DMRS 심볼 순서로 펼치고 1/√2를 곱한다(모델이 학습된 형식). 이전 형식은 `--nrx-ls-input example` | `slot_radio.NrxPath` |
| 살릴 TB 고르기 | 실패한 code block이 K개 이하인 TB를 NRx로 보낸다(`--nrx-max-cb-fail`). 실패 수가 적은 TB부터 lane을 주고, 실패 수가 많은 TB는 그 주기의 기존 수신기 결과가 다 나온 뒤에 lane을 잡는다 | `controller3.py` |
| AI 단위 크기 | 최근 20 ms 동안 허락받은 크기의 비율로 고른다. 진행 중인 chunk보다 작은 크기만 남은 허락은 내지 않는다 | `ai_worker2.py`, `controller*.py` |
| NRx 대기 중 AI | NRx가 lane을 기다리는 동안 AI를 허락하지 않는다(`+yield`) | `controller3.py` |
| LDPC 반복 횟수 | 기존 수신기와 NRx 뒤 LDPC의 횟수를 설정으로 준다(`--conv-ldpc-iterations`, `--nrx-ldpc-iterations`). 0이면 cuPHY 표(여기서는 10회) | `slot_radio.conv_path`, `nrx_path` |
| MU-MIMO 약한 셀 | 슬롯 하나에 UE 둘(`nv_mu2`). 두 수신기 모두 한 번 실행으로 두 TB를 복호하고, 결과는 UE별로 센다. NRx는 NVlabs `nrx_rt` | `mu_radio.py`, `ul_profiles.NV_MU2`, `summarize_slot.py` |

| NRx 빠른 경로 | LS 추정을 pilot RE × 고정 계수 곱셈으로, rate recovery·LDPC·CRC를 미리 잡은 버퍼로 cuPHY 직접 호출. nrx_rt 3.08 → 2.36 ms | `mu_radio.MuNrxPath(mode="fast")` |
| NRx 모델 고르기 | 엔진 경로로 정한다. `nrx_rt`(1.35 ms)와 `nrx_large`(4.29 ms). 큰 모델은 rescue 마감 11.5 ms 이상, GPU당 4셀까지 | `make_config.py --engine`, `mu_main.sh`의 `ENGINE` |
| 바뀌는 부하 | 일정 주기마다 바쁜 구간과 한가한 구간이 번갈아 온다(`--activity-mode phased --activity-phase N`, 바쁜 구간의 확률은 `--activity-high`, 기본 1). 구간별 집계는 `analyze_phases.py` | `activity.py` |
| 비교 대상: 부하에 맞춰 비율을 바꾸는 방식 | MPS 비율은 client를 띄운 뒤 못 바꾼다. GPU마다 비율이 다른 AI worker를 미리 띄워 두고, controller가 최근 100 ms의 무선 부하(지연 0 또는 1초)를 보고 하나만 새 요청을 받게 한다. worker는 요청을 하나씩 가져가서 대기 중인 요청이 전환 뒤의 worker로 넘어간다. 전환 비용이 없는 가정이라 이 방식에 유리하다 | `ai_worker_dyn.py`, `controller4.py`의 `dynamic_share`, `make_config.py --dynamic-shares` |
| AI를 GPU 몇 개에 몰기 | 요청을 낮은 번호 GPU부터 채운다. 시험한 부하에서는 효과가 없어 기본으로 쓰지 않는다 | `controller4.py`, `ai_dispatch_order: pack` |

MU-MIMO 셀에서 controller가 보는 "실패한 code block 수"는 두 UE의 합이다. NRx 한 번이 두 UE를 같이 복호하므로
lane 배정 단위는 슬롯이다.

## 15. v13에서 바뀐 것 (2026-10-02, job 59192512·59199237)

근거와 수치는 [BACKSTOP_V13_SWEEPS_KO.md](BACKSTOP_V13_SWEEPS_KO.md).

| 항목 | 이전 (v7–v12) | v13 | 코드 |
|---|---|---|---|
| AI의 MPS 우선순위 | 무선과 같음 | 낮음(`CUDA_MPS_CLIENT_PRIORITY=1`) | `make_config.py --ai-mps-priority 1`, `launch_slot.py` |
| AI 비율 한도 | 70% | 없음 | `--gated-mps-pct` 생략 |
| 기존 수신기 옆 AI 단위 | 128 token만 | GPU당 4셀: 128/512/1024 모두, 5셀: 128만 | `--unit-gating .../conv=128,512,1024` |
| NRx 옆 AI | 그 NRx가 마감 안에 끝나면 허용 | 마감 조건 + 빈 lane이 R개 이상(4 lane에서 R=3: 다른 NRx가 없을 때만) | `controller4.py`의 `free_lane_reserve`, `--unit-gating ...+reserve3` |
| 겹침 한도(128/512/1024) | 10.1 / 9.8 / 10.3 ms (p99.9) | 8.1 / 8.5 / 8.7 ms (낮은 우선순위에서 p99 + 여유) | `v13_cal.sh` |
| AI 입장 제어 | 시간 제한 안에 끝난다고 예측되면 받음 | 제한의 75% 안에 끝난다고 예측되면 받음(모든 방식 공통) | `--ai-admission-fraction 0.75` |
| 부하 모양 | 일정, 버스트, 교대(phased) | + 무작위 계단(`steps`) | `activity.py`, `--activity-mode steps` |
| AI 도착 | Poisson | + 간격 변동계수 지정 | `ai_arrivals.py`, `--ai-arrival-cv` |
| 부하 따라 비율(비교 대상) | 비율 2개 | 비율 3개까지(`--dynamic-levels`), 낮은 우선순위 결합 | `controller4.py`, `ai_worker_dyn.py` |

lane 여유 규칙의 동작(`controller4.py`, AI 허락 단계): GPU g에 돌고 있는 NRx가 있고 서버 전체의 빈 lane 수가
R보다 작으면, 이번 루프에서 g에는 AI 조각을 주지 않는다. NRx가 없는 GPU에는 이 규칙이 적용되지 않는다.
이미 준 조각은 끝까지 돈다(낮은 우선순위라 조각 하나가 무선 옆에서 중앙값 2.3 ms, p99 8.7 ms 걸림).

