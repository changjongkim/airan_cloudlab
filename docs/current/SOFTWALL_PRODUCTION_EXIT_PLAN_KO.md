# SoftWall production exit gate와 고도화 계획 — 2026-09-25

## 판정

현재 SoftWall은 **synthetic contract에서 자격 검증된 substrate**다. Production PUSCH/HARQ
보장은 아직 아니다. 이번 감사에서 공백을 limitation 문구가 아니라 P1--P4 exit gate로
승격했다. P3의 좁은 표준화 채널 mode는 새 holdout으로 닫았지만, production을 결정하는
P1·P2와 그 뒤의 P4는 아직 FAIL이다.

| Exit gate | 현재 판정 | 직접 근거 |
|---|---|---|
| P1 live-DU timing | **FAIL** | target DU의 synchronized `d_MAC` trace가 없음 |
| P2 production fast path | **FAIL** | same-stream raw-IQ P2P도 4.5 ms 이내 885/1,000; late 115 |
| P3 supported-channel NeuralRx | **PASS, 범위 제한** | 독립 Sionna CDL-D/E holdout 500건에서 high-SNR 두 pipeline 정상, 저 SNR NeuralRx-only 31 대 conventional-only 12 |
| P4 integrated production mode | **BLOCKED** | P1/P2가 닫힌 뒤 동일 계약으로 재자격화해야 함 |

권위 있는 기계 판정은
[production exit gate v2](../../results/softwall_multigpu/softwall_production_exit_gate_v2.json)와
[fail-closed P1--P4 판정](../../results/softwall_multigpu/softwall_production_gates_current_v2.json)에 있다.

## 1. Production timing에서 새로 확인한 것

### 1.1 실제 코드에 존재하는 중간 계약

현재 Aerial checkout의 testMAC 설정은 다음 threshold를 갖는다.

```text
early HARQ:       T0 + 2.0 ms
UL indication:    T0 + 4.5 ms
PRACH indication: T0 + 4.5 ms
UCI indication:   T0 + 4.5 ms
```

`scf_fapi_handler::validate_indication_timing`은 SFN/slot으로 `T0`를 복원하고 indication
handler 진입시각이 threshold를 넘으면 late로 센다. 이것은 종전의 `P180/D155`보다 훨씬
강한 vendor integration target이다. 다만 NVIDIA가 testMAC을 “개발자가 controlled
environment에서 사용하는 L2 도구”라고 정의하므로, 이 값을 field DU의 production
`d_MAC`이라고 부르지는 않는다.

### 1.2 현재 경로의 4.5 ms 진단

모든 수치는 clean synthetic PUSCH, AI 없음, warm persistent process의 표본 결과다. WCET가
아니며 실제 DU qualification도 아니다.

| 경로 | 4.5 ms 이내 | p50 | p99 | 판정 |
|---|---:|---:|---:|---|
| Local NeuralRx 후 conventional | 0/1,000 | 6.609 ms | 6.984 ms | 순차 fallback 기각 |
| Same-GPU speculative dual path | 0/1,000 | 6.828 ms | 17.552 ms | GPU 경합으로 기각 |
| GPU0 LS 준비 + GPU1 NRx + GPU0 conventional | 0/1,000 | 5.942 ms | 11.897 ms | GPU0 front/post 직렬화로 기각 |
| **Raw-IQ P2P, GPU1 full NeuralRx** | **852/1,000** | **3.831 ms** | 9.506 ms | 구조는 유망, tail 때문에 FAIL |
| 위 경로, Python GC OFF | 838/1,000 | 3.988 ms | 10.041 ms | GC 원인 가설 기각 |
| 위 경로, caller-owned same-stream + busy poll | **885/1,000** | **3.929 ms** | 11.696 ms | 33건 개선, tail 때문에 FAIL |
| GPU1 single-stream public wrapper | 첫 timed request timeout | worker 약 106.9 ms | — | 기각 |

가장 중요한 결과는 raw-IQ full-remote 경로다. GPU0에서 NeuralRx용 LS를 먼저 계산하지 않고
raw frequency-domain IQ만 GPU1로 넘긴 뒤, GPU1에서 channel estimation→TensorRT→LDPC→CRC를
완료하면 중앙값은 4.5 ms 안으로 들어왔다. Caller-owned stream으로 cuPHY→TensorRT의 host
synchronization을 없애면 적시 완료가 852에서 885로 늘었지만, 115/1,000 tail이 남으므로
안전 계약으로 채택할 수 없다.

사후 stage diagnostic 300건은 긴 tail을 remote cuPHY LS channel estimation으로 좁혔다.
Channel estimation GPU 시간은 p50 0.866 ms, p99 4.873 ms, max 9.592 ms였고 pair wall과
Pearson 상관은 0.9603이었다. TensorRT host enqueue 평균은 7.443 µs였고 실제 graph GPU
p99는 0.886 ms였다. Forward/backward P2P GPU copy 평균은 36.261/17.150 µs였다. Derate-match
host call은 평균 1,060.105 µs, p99 1,105.380 µs였지만 GPU p99는 0.168 ms였다. 단위와
계층을 섞지 않으면 transport와 TensorRT graph는 비교적 안정적이고, tail의 지배 단계는
cuPHY LS channel estimation이다.

이때 4.5 ms는 단순 그래프 선이 아니라 mode의 `D` parameter다. Diagnostic mode에서
channel-estimation component 하나의 p99가 이미 `D`를 넘으므로, scheduling이나 Qwen을
붙이기 전에 underlying NRx service class가 자격 미달이다. Profiling event가 timing을
바꾸므로 이 수치는 qualification/WCET가 아니라 다음 C++/CUDA fast-path 구현의 병목 근거다.

### 1.3 설계에 주는 결론

현재의 wait-then-recover production 조건은 다음 부등식을 만족해야 한다.

```text
B_NRx + B_recovery + B_control + guard <= d_MAC - t_release
```

격리된 local 표본 중앙값부터 약 6.6 ms이므로 4.5 ms 후보에서는 성립하지 않는다.
Same-GPU speculative mode도 다음 조건을 만족하지 못했다.

```text
max(B_NRx_corun, B_recovery_corun) + B_control + guard <= D
```

따라서 production-tight mode의 우선 구현은 **GPU0 conventional + GPU1 full NeuralRx**다.
GPU0은 raw-IQ publish 직후 conventional을 시작하고, GPU1은 LS 생성부터 CRC까지 소유한다.
둘 중 유효한 radio 결과를 single-commit하되 두 물리 작업의 credit은 fence까지 유지한다.
이 mode에서 conventional은 speculative shadow이므로 종전의 “NRx 실패 뒤 생기는 recovery
debt”가 이미 실행 중인 obligation으로 바뀐다. SoftWall certificate/fence/ownership은 계속
필요하지만, conditional slack과 AI lease는 해당 tight window 밖에서만 허가한다.

## 2. NeuralRx channel contract에서 새로 확인한 것

다음 원인 후보를 서로 다른 development seed로 검사했다.

| 진단 | Conventional | NeuralRx | 판정 |
|---|---:|---:|---|
| corrected 1×4 raw TDL-A | 10/10 | 0/10 | 실패 보존 |
| raw/CFR RMS/UE power/component power normalization | 각 10/10 | 전부 0/10 | scale 가설 기각 |
| reference DMRS spacing, MCS7, start0, TDI1 | 각 10/10 | 전부 0/10 | radio-profile 가설 기각 |
| exact one-antenna Tx, frequency/time Aerial channel | 각 10/10 | 전부 0/10 | Tx/channel execution 가설 기각 |
| public notebook와 같은 FP32 TensorRT build | 각 10/10 | 전부 0/10 | FP16 가설 기각 |

즉 conventional이 같은 파형을 계속 복호하므로 TDL generator나 tensor shape가 단순히 깨진
상태는 아니다. 제공된 pretrained ONNX는 **Aerial TDL-A에서 여전히 unqualified**다.

그 뒤 공개 notebook의 실제 계약을 그대로 복원했다. 공개 코드는 TDL branch가 없고
Rayleigh와 CDL-A--E를 선언한다. FP32 engine, MCS7, start0, DMRS 0/5/10, 1TX/4RX로 분리한
결과는 다음과 같다.

| 단계 | Conventional | NeuralRx | 해석 |
|---|---:|---:|---|
| clean direct/wrapper | 각 10/10 | 각 10/10 | engine·MCS·wrapper 정상 |
| Sionna default Rayleigh, 30 dB | 20/20 | 20/20 | Sionna→cuPHY interface 정상 |
| CDL-A/100 ns native·Fortran layout | 각 20/20 | 각 0/20 | memory-layout 가설 기각 |
| CDL-A/1, 10, 30, 100 ns | 전부 10/10 | 10/10, 2/10, 0/10, 0/10 | delay/family support boundary 존재 |
| CDL-B/C/D/E, 100 ns | 전부 10/10 | 0/10, 0/10, 10/10, 10/10 | D/E를 holdout 후보로 고정 |

개발 matrix와 분리한 seed로 CDL-D/E holdout을 실행했다. 매 trial에서 payload와 slot을
바꾸고 모델별 5개 Es/N0×50건, 총 500건을 paired 비교했다.

| Holdout | Conventional | NeuralRx | 저 SNR NeuralRx-only | 저 SNR conventional-only |
|---|---:|---:|---:|---:|
| CDL-D/100 ns | 153/250 | 164/250 | 16 | 5 |
| CDL-E/100 ns | 155/250 | 163/250 | 15 | 7 |
| 합계 discordant | — | — | **31** | **12** |

두 모델 모두 10 dB에서 양쪽 pipeline이 50/50 성공했고, 사전 조건인 모델별 NeuralRx-only
5건 이상을 통과했다. 합친 discordant pair의 two-sided exact `p=0.00540`이다. 따라서 P3는
**Sionna CDL-D/E, 100 ns, 이 radio/model/precision 조합에 한해 finite-sample PASS**다.
이는 Aerial TDL-A, field IQ 또는 production timing을 자격화하지 않는다.

## 3. 끝내기 위한 네 gate

### P1 — Live DU clock contract

대상 DU에서 동일 clock으로 다음을 받아야 한다.

```text
t_IQ_ready → t_PHY_submit → t_CRC_visible → t_FAPI_publish
             → t_MAC_consume, absolute d_MAC
```

요청 ID, SFN/slot, single commit, clock conversion error와 expiry provenance를 C163 validator에
넣는다. testMAC 4.5 ms는 그 전까지 conservative integration target으로만 사용한다.

### P2 — Production fast path

raw-IQ P2P 구조를 Python polling prototype에서 persistent C++/CUDA path로 옮긴다.

1. GPU0 raw-IQ publish와 conventional launch를 같은 release event에서 시작한다.
2. GPU1은 channel estimation, TensorRT, LDPC, CRC를 하나의 persistent execution graph 또는
   bounded device-side chain으로 수행한다.
3. 결과는 TB/CRC와 fence만 P2P로 반환한다. LLR 전체를 GPU0으로 돌려보내지 않는다.
4. CPU scheduling, lazy allocation, clock/power state를 mode provenance에 넣는다.
5. P1의 `D` 안에서 all-fail conventional completion과 single commit을 독립 node에서
   재자격화한다.

현재 Python raw-P2P는 구조 선택의 근거일 뿐 qualification 결과가 아니다.

### P3 — Supported-channel NeuralRx — 좁은 mode 완료

Public notebook의 원래 Sionna 1.0.2/TensorFlow 2.19 환경과 official ONNX를 복원했고,
개발 matrix 뒤 disjoint-seed CDL-D/E holdout을 통과했다. P3 artifact는
[CDL-D/E gate](../../results/softwall_same_gpu/sionna_cdl_de_holdout_gate_job58868184.json)다.
검증된 mode 밖에서는 fail-closed한다. Aerial TDL-A나 field trace를 쓰려면 별도 지원 모델을
확보하거나 재학습한 뒤 새 holdout을 열어야 한다.

이번 채택 조건은 high-SNR correctness만이 아니었다.

- frozen external-channel holdout에서 conventional과 NeuralRx pipeline 모두 정상 동작
- 독립 SNR/channel grid에서 BLER와 NeuralRx-only recovery value 보고
- 미래 CRC나 generator truth 없이 observable feature만 사용
- 선택한 channel/model/precision 전체를 service-bound mode로 재자격화

### P4 — Integrated production mode — P1/P2 대기

P1–P3 뒤에만 MPS와 bounded Qwen을 다시 넣는다. 같은 production expiry에서 correlated
all-fail, raw-IQ P2P NeuralRx, conventional single commit, Qwen deadline value와 fence/credit
불변식을 함께 검사한다. 이 gate 전에는 production-HARQ나 external-channel gain을 주장하지
않는다.

## 4. 논문 범위 결정

P3의 제한된 supported mode는 확보했다. 이제 제출 범위를 가르는 것은 P1/P2다.

1. 현재 논문을 synthetic qualification/envelope 논문으로 제출하고 production claim을
   명시적으로 제외한다.
2. Production AI-RAN 논문을 목표로 하면 P1·P2를 닫고 그 exact hash에 P3를 묶은 P4까지
   완료한 뒤 제출한다.

현재 데이터는 channel gap을 전부 뭉뚱그릴 필요가 없음을 보여준다. CDL-D/E는 통과했고
TDL-A는 실패했다. 반면 production timing은 여전히 실제 구현을 기각한다. 다음 구현은
remote channel-estimation tail을 줄이는 persistent C++/CUDA fast path와 live-DU contract
수집에 집중한다. 추가 optimizer나 lifecycle matrix 확장은 이 gate보다 우선순위가 낮다.
