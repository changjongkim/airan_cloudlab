# SoftWall 동일 GPU MPS 초기 실험 보고서

**실행일:** 2026-09-20  
**상태:** P0–P4, 제한적 S1, 동일 GPU S2 vertical slice와 자연 채널/두 셀 검증 완료. 상세 S2 결과는 [S2 자연 채널 보고서](S2_NATURAL_CHANNEL_REPORT_KO.md)를 따른다.

## 결론

같은 A100 GPU에서 MIG를 끄고 MPS를 사용해도, **허용된 AI work unit의 제출 시점과 완료를 통제하면 RAN tail을 보호하면서 AI 처리량을 남길 수 있다.** 실제 valid PUSCH 2개를 25 ms마다 처리하는 조건에서 21 ms deadline을 사전에 고정한 최종 독립 실험 결과는 다음과 같다.

| 구성 | 표본 | RAN miss | 잘못된 TB | AI 처리량 | 최악 RAN 응답 |
|---|---:|---:|---:|---:|---:|
| RAN 단독, 두 독립 campaign 합계 | 10,000 | 0 | 0 | 0 | 20.641 ms |
| 무제어 NeuralRx | 5,000 | 0 | 0 | 927.6/s | 16.589 ms |
| 무제어 NeuralRx, 별도 반복 | 5,000 | 2 | 0 | 931.1/s | 21.256 ms |
| **S0 시간 보호 NeuralRx** | **5,000** | **0** | **0** | **380.9/s** | **14.027 ms** |
| **S1 NeuralRx cap80/p0** | **5,000** | **0** | **0** | **860.3/s** | **19.093 ms** |
| heavy Qwen cap100/p0 | 5,000 | 153 | 0 | 11.4/s | 26.065 ms |
| **S1 heavy Qwen cap80/p0** | **5,000** | **0** | **0** | **9.7/s** | **18.667 ms** |

S1 cap80은 무제어 NeuralRx 처리량의 92.4%를 유지하고 S0보다 2.26배 높은 처리량을 냈다. Heavy Qwen에서도 cap80은 cap100 처리량의 약 85%를 유지했다. 그러나 MPS SM cap은 HBM 대역폭과 launch queue를 강제 제한하지 않는다. 따라서 **S0는 bounded unit과 검증된 RAN recovery bound를 전제로 한 조건부 시간 격리 스킴이고, S1 cap80은 workload별로 확인된 통계적 SLO 후보**로 구분한다.

0 miss 5,000표본이 주는 단측 95% miss 확률 상한은 약 0.06%다. 이는 무조건적인 hard-real-time 증명이 아니다. Hard deadline 주장은 RAN 실행 상한, AI end-to-end unit 상한, 잔여 blocking 및 제어 지연의 방어 가능한 상한이 추가로 필요하다.

## 실행 환경

- Slurm job `58625542`, node `nid001201`, A100-SXM4-40GB 4개 중 GPU0 한 개 사용
- MIG OFF, NVIDIA driver 580.178.04
- 실제 MPS control/server를 실행하고 client 목록을 확인
- Aerial CUDA-Accelerated RAN tag 25.3.2, commit `3bf76a43dceb493b00f2ee75fdfbb87038eab7c6`
- Shifter image `nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb`
- 모든 소스·engine·상태·결과는 `/pscratch/sd/s/sgkim/kcj/airan_cloudlab` 안에 저장

MPS 검증 때 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50` client가 A100의 108 SM 중 54개만 보았고, control server에서 해당 client PID를 확인했다. 긴 Unix socket 경로는 MPS daemon과 RPC worker 양쪽에서 실패했으므로 `$ROOT/mps/$JOB_ID/` 아래의 짧은 경로로 고정했다.

## 실제 workload 검증

### RAN

Aerial 자체 Tx→Rx test와 동일한 파라미터로 `PdschTx`가 273-PRB PUSCH slot과 payload를 만든다. 매 요청은 다음 전체 경로를 실행한다.

```text
valid received slot
  → channel estimation
  → noise/interference estimation
  → equalization
  → LDPC de-rate-match
  → LDPC decode
  → CRC
  → 원 payload와 byte 단위 비교
```

최종 실험의 모든 요청에서 CRC와 payload가 일치했다. Random complex tensor를 사용한 초기 full-path 실험은 timing 탐색용으로만 남기고 radio correctness 근거로 사용하지 않는다.

### NeuralRx

Aerial의 `neural_rx.onnx`를 프로젝트 내부에 TensorRT FP16 engine으로 빌드했다. Direct TensorRT CUDA Graph 경로를 pyAerial wrapper와 비교한 결과 두 output tensor의 최대 절대 오차는 모두 0이었다. 단독 1,000회 추론은 GPU 평균 0.897 ms, p99 1.119 ms였다.

### Qwen

PyTorch 2.6.0+cu126, Transformers 4.49.0과 Qwen2.5-1.5B를 모두 프로젝트 내부 runtime/cache에 설치했다. Batch 1, context 16 prefill은 약 47–49 unit/s였고 MPS cap20에서도 처리량이 거의 줄지 않았다. 이 조건은 weight/HBM 지배 workload라 SM cap만으로 부하량을 잘 표현하지 못한다.

Batch 8, context 512 prefill은 단독 probe에서 11.77 unit/s, 최악 unit GPU 시간 86.69 ms였다. 이 heavy phase를 25 ms RAN period에 연속 overlap한 최종 독립 실험에서는 cap100이 153/5,000 miss를 만들었지만 cap80은 0/5,000 miss와 9.7 unit/s를 기록했다. 같은 모델이라도 batch와 context가 바뀌면 별도 qualification이 필요하다는 직접 근거다.

## 스킴

### S0: quiet-window 시간 보호

각 RAN release 뒤 AI unit을 한 개씩 제출한다. 다음 조건을 만족할 때만 새 unit을 허가한다.

```text
now + B_ai + G <= next_RAN_release
```

- `B_ai`: 제출부터 GPU 완료 ACK까지 포함한 AI unit 계약 시간
- `G`: 측정·IPC·release 오차 guard
- outstanding AI unit은 최대 1개
- ACK를 받은 뒤에만 다음 unit을 제출
- 실제 실행시간이 `B_ai`를 넘거나 다음 release를 침범하면 계약 위반으로 기록

최종 paired 2-cell 실험은 `period=25 ms`, `deadline=21 ms`, `B_ai=6 ms`, `G=1 ms`를 사용했다. NeuralRx 5,000개 RAN 요청에서 budget violation과 release crossing은 모두 0이었다.

### S1: 검증된 workload의 제한 overlap

RAN client는 108 SM 전체와 priority 0을 유지하고, NeuralRx client만 MPS active-thread cap을 적용한다. Cap 10–100% sweep 결과:

- MPS priority 1은 보호 효과가 없었고 일부 조건에서 오히려 miss가 증가했다.
- 10–40% cap은 초기 synthetic-input RAN에서 S0보다 AI 처리량이 낮았다.
- paired 2-cell 최종 gate에서 cap80/priority0은 860.3 inference/s와 0/5,000 miss를 기록했다.
- cap100/priority0은 별도 반복에서 2/5,000 miss를 기록했다.

Cap80은 workload qualification 결과로만 허용한다. HBM phase, client 수, model 또는 tensor shape가 바뀌면 재검증 없이 같은 계약을 적용할 수 없다.

Heavy Qwen에도 같은 원칙을 적용했다. Batch 8/context 512에서는 cap100의 RAN miss가 run별로 7–117/1,000까지 변했고, cap80은 5개 독립 run에서 모두 0/1,000이었다. Phase alignment에 따라 cap100 tail이 크게 변하므로 평균 latency 또는 평균 GPU utilization만으로 admission하면 안 된다.

### S0의 Qwen 경계 사례

Batch 1/context 16 Qwen unit의 GPU 시간은 대략 20–24 ms여서 25 ms period의 quiet window에는 들어가지 않는다. Period를 50 ms로 늘린 사전 고정 실험에서 S0는 AI budget 위반과 다음 release 침범이 모두 0이었지만 RAN 1/1,500 miss를 기록해 gate에 실패했다. 해당 miss의 RAN GPU service는 21.070 ms였고 직전 Qwen unit은 release 약 19.3 ms 전에 끝났다. 즉 동시 overlap만 제거해도 RAN 자체 희귀 tail 또는 이전 workload가 남긴 cache/power 상태까지 자동으로 상한화되지는 않는다. S0의 hard claim에는 workload 전환 후 RAN service bound도 포함해야 한다.

### AI fault fail-closed

NeuralRx RPC worker가 5개 unit 뒤 연결을 닫도록 fault를 주입했다. Controller는 EOF를 감지해 이후 AI admission을 전부 중지하고 RAN을 계속 실행했다. Fault 감지 뒤 499개를 포함한 전체 500개 요청에서 deadline miss 0, correct TB 500/500을 기록했다. 이는 process/channel failure에 대한 fail-closed 동작이다. 이미 실행 중인 무한 GPU kernel을 MPS가 선점할 수 있다는 의미는 아니므로 허용 AI는 여전히 bounded unit 계약을 만족해야 한다.

별도 실험에서는 NeuralRx GPU 완료 뒤 응답만 20 ms 지연하고 controller timeout을 7 ms로 고정했다. Controller는 late response를 accept하지 않고 AI를 disable했으며, 이후 499개를 포함한 500개 RAN 요청에서 miss 0, correct TB 500/500을 기록했다. 이 실험은 stale completion fencing과 timeout 경로를 검증한다.

### S2 optional NeuralRx + reserved recovery

같은 valid PUSCH를 conventional full receiver와 direct persistent NeuralRx full receiver가 처리하는 S2 vertical slice를 추가했다. Clean 입력 1,000개에서 두 경로 모두 1,000/1,000 correct였고, −8.5 dB block-Rayleigh+AWGN에서는 500개 중 conventional 180, NeuralRx 489, union 491로 자연 상보성을 확인했다.

S2는 NRx endpoint와 conventional fallback interval을 원자적으로 예약하고 한 결과만 commit한다. NRx CRC/payload 실패 또는 timing cutoff 때 예약한 conventional을 실행하며, late duplicate completion은 epoch/single-commit fence로 거부한다. 두 셀에서는 단일 fallback lane의 서로 겹치지 않는 두 interval을 예약한다.

주요 최종 결과는 다음과 같다.

| S2 실험 | 표본 | radio utility | RAN miss | background 이득 | 계약 결과 |
|---|---:|---:|---:|---:|---|
| 합성 NRx invalidation, 1 cell | 1,500 | eager와 동일, incorrect 0 | 0 | eager 대비 +9.15% | S2 gate 통과 |
| 자연 채널, 1 cell | 10,000 paired | S2−eager −0.0300 pp, 95% CI [−0.0819,+0.0219] | 0 | +3.17% | utility 비열등성 통과; safe admission reject 1건 때문에 전체 protocol은 형식상 실패 |
| fail-closed start lateness | 1,000 | 1,000/1,000 correct | 0 | 양의 처리량 | 주입 100/100 모두 NRx 거부→conventional, 전체 gate 통과 |
| 자연 채널, 2-cell correlated burst | 3,000 paired | S2−eager −0.0333 pp, 95% CI [−0.0987,+0.0320] | 0 | **+5.53%** | 양 정책 AI violation 0, 전체 gate 통과 |

Confirm19의 더 이른 P60/D50 탐색에서는 NRx timing tail과 conventional decoder의 임계 판정 변동을 발견했다. 실패한 완전 일치 gate를 보존하고, receiver 안정성 진단과 새 seed의 confirm21/24 follow-up으로 두 원인을 분리했다. 세부 protocol, 실패와 raw 결과는 [S2 보고서](S2_NATURAL_CHANNEL_REPORT_KO.md)에 있다.

별도 MPS client의 NRx graph 40회 묶음을 5 ms timeout 뒤에도 물리 실행시킨 confirm26에서는 P60/D25가 cap100 2/1,500, cap20 1/1,500 miss로 실패했다. Cap20도 완전 격리를 만들지 못했으므로 해당 optional unit은 D25에서 reject해야 한다. D35 follow-up에서 cap20은 5,000 releases와 1,500 physical-overlap releases 모두 miss 0이었지만, cap100의 worker 비활성 release에서 36.076 ms conventional GPU tail이 한 번 발생해 전체 gate는 실패했다. Worker 없는 conventional-only 10,000 releases는 miss 0, GPU max 7.571 ms였다. 이 결과는 MPS cap이 qualification 수단이지 hard isolation primitive가 아니며, campaign residual/system tail까지 admission contract에 포함해야 함을 보여준다.

Timeout 뒤 새 admission만 막고 CUDA context를 idle 상태로 남긴 confirm29는 cap100 0/5,000, cap20 1/5,000 miss였다. Late response를 drain한 직후 worker process/MPS client를 종료한 confirm30에서는 상황이 악화됐다. Cap100은 6/5,000, cap20은 11/5,000 miss였고, 17건 모두 worker가 물리적으로 종료된 뒤 발생했다. 14건은 conventional GPU event가 42.569–99.944 ms로 늘어난 직접 miss였고 3건은 직전 긴 event의 연쇄 start-lateness였다. Worker 없는 confirm28과 비교하면, 실시간 구간의 client teardown도 안전한 복구 동작으로 가정할 수 없다. Endpoint는 즉시 논리 quarantine하되 process/context teardown은 RAN-free maintenance window로 미루는 정책이 필요하다.

Confirm31은 이 lifecycle을 분리하려 했지만, 사후 monotonic timestamp 대조에서 별도 sham-retirement campaign과 release 구간이 겹쳤다. 기존 3회×cap100/20 결과와 analyzer PASS는 오염 자료로 보존하되 claim에서 제외하며, 전역 GPU0 experiment lock 아래 단독 재실행한다.

Confirm32는 same-request 데이터 경로를 실제 별도 process로 옮겼다. Controller가 request별 PUSCH와 channel estimate를 CUDA IPC forward buffer에 쓰고, cap80 MPS NeuralRx endpoint가 output LLR을 backward buffer로 반환하며 controller가 LDPC/CRC를 완료했다. Clean 10,000 releases에서 correct 10,000/10,000, miss/timeout/fallback 0, response p99 4.125 ms와 max 8.730 ms였고 worker는 실제 86 SM과 평균 1.081 ms를 기록했다. Endpoint는 timed epoch 전체에 상주하고 마지막 release 뒤에만 종료했다.

Confirm33은 같은 외부 endpoint를 −8.5 dB 자연 채널로 확장했다. 10,000 releases에서 NeuralRx commit 9,779건, 자연 fallback 221건, conventional recovery 47건으로 최종 correct 9,826건을 얻었다. Deadline miss와 endpoint timeout은 모두 0, response p99/max는 6.004/30.669 ms였으며, worker는 예정된 10,021 units와 86 visible SM을 정확히 기록했다. 복구되지 않은 174건은 deadline miss가 아니라 양 receiver가 모두 실패한 radio error다.

Confirm36은 같은 node/job/trace의 세 조건을 모두 완료한 첫 external strong-baseline gate다. Conventional-only 3,688, eager-dual 9,835, external S2 9,840/10,000 correct, 모두 deadline miss 0이었다. External−eager +0.050 pp, paired 95% CI [−0.0088,+0.1088] pp가 사전 비열등성 마진 −0.25 pp를 통과했다. External은 conventional을 자연 실패 203건에만 실행해 43건을 추가 복구했고 endpoint timeout 0, worker 10,021 units/86 SM이었다. Confirm34의 MPS orchestration 충돌과 Confirm35의 allocation 중단은 paired 비교에서 제외하고 실패 이력으로 남겼다.

Confirm37은 세 번째 cap20 NeuralRx MPS worker를 외부 endpoint에 추가했으나 6 ms host-RPC budget을 한 unit이 6.208 ms로 초과해 전체 gate에 실패했다. 그 즉시 새 background admission을 중단했고 RAN 500 releases에서 miss 0, AI release crossing 0이었다. 새로운 seed와 prior host-tail budget 40 ms를 사전 고정한 Confirm37b는 RAN 500회 miss/endpoint timeout 0, background 완료 6,519 units, AI budget/crossing/fault 0으로 통과했다. 실패한 6 ms 계약을 통과한 40 ms 계약으로 소급 변경하지 않는다.

Confirm38은 cap80 외부 endpoint와 cap20 background worker를 동시에 실행하고, 동일 node/job/trace의 local S2+AI 및 eager-dual+AI와 각 10,000회 비교했다. Correct는 external/local/eager 9,845/9,841/9,843, RAN miss·AI budget/crossing/fault 모두 0이었다. AI 완료량은 130,177/127,180/126,477 units로 외부가 eager보다 3,700 units(+2.93%) 많았다. External−eager radio 차이 +0.020 pp, paired 95% CI [−0.0354,+0.0754] pp는 사전 −0.25 pp 마진을 통과했다. Local S2와 eager의 NRx 20 ms branch-bound 초과 각각 2건은 deadline miss가 아니었고 frozen gate에도 없었으나 진단 수치로 보존한다. 역순 독립 seed Confirm41도 세 방식 모두 correct 9,817/10,000·miss 0, 외부 AI 130,254 대 eager 126,390(+3.06%)으로 모든 frozen gate를 통과했다.

Confirm32–38/41 외부 controller의 자연 실패 처리와 AI lease는 실제였지만, 로컬 S2의 원자 `DartRuntime` fallback calendar·single commit을 그 별도-process 경로에 아직 연결하지 않았다. 그러므로 위 외부 처리량 결과를 완성된 공동 예약 스킴의 성능으로 부르지 않는다. Confirm42는 이를 연결한 새 controller를 별도 node `nid001145`에서 500회 검증했다. Correct 486, RAN miss·endpoint timeout·NRx/conv bound 위반·AI budget/crossing/fault·중복 commit 모두 0; 자연 fallback 18건 중 4건이 conventional로 복구됐고, 완료 AI 6,304개 중 52개를 예약된 recovery 시작 전 gap에서 실행했다. Endpoint와 fallback calendar의 잔여 credit도 0이다. 최초 analyzer는 worker 필드명 오류로 종료했지만 원자료를 보존하고 판정기만 수정해 [frozen gate 모두 통과](confirm42_transaction_gap_smoke.md)를 확인했다. 이는 500회 통합 smoke다. Confirm43 ON/OFF ABBA는 각 10,000회 RAN miss 0, gap ON−OFF 합계 AI +1,069 units였으나 NRx 30 ms bound 위반 3건으로 frozen 전체 gate가 실패했다. 독립 seed·역순 Confirm44도 네 조건 모두 RAN miss 0, gap ON−OFF AI +3,317(+1.304%)였으나 ON 2회·OFF 1회 NRx bound 위반으로 전체 gate 실패했다. 실제 transaction 대 eager의 Confirm45는 독립 seed·역순 두 짝에서 AI +2.95%/+2.41%, 모든 frozen gate를 통과했다. 이는 완성된 예약 스킴의 처리량 이득 근거이지만 Confirm43/44 실패를 지우지 않는다.

Lifecycle은 별도 재검증했다. Confirm39의 오염되지 않은 sham-retirement 5×2×1,000회는 cap100·cap20 각각 4/5,000 D35 miss로 실패했다. Worker는 ready 전 GPU warmup 10회를 수행했지만 RAN 구간 optional work 제출은 0건이었다. 같은 job의 별도 MPS epoch에서 fault를 유지보수 창으로 drain·retire하고 재자격을 확인한 경로는 post 6,000회 miss 0, quiet interval 최소 28.623초로 해당 subgroup gate를 통과했다. 전체 Confirm39 gate는 sham 때문에 실패했으며, GPU client 없는 CPU-sham 대조 전에는 종료 원인을 단정하지 않는다. Confirm40 cold-start ABBA는 conventional warm-up OFF 첫 복구 22.756/35.303 ms 대 ON 2.277/2.286 ms, 각 500회 RAN miss 0으로 모든 frozen gate를 통과했다.

## 간섭 종류별 결과

동일한 valid paired 2-cell RAN, `period=25 ms`, `deadline=21 ms`에서 다음 결과를 얻었다.

| AI workload | 구성 | RAN 표본 | miss | AI 처리량 | budget 위반 | release 침범 |
|---|---|---:|---:|---:|---:|---:|
| GEMM 2048² FP32 | 무제어 | 3,000 | 17 | 791.7/s | 해당 없음 | 해당 없음 |
| GEMM 2048² FP32 | S0 | 3,000 | 0 | 316.1/s | 0 | 0 |
| HBM 256 MiB 왕복 copy | 무제어 | 3,000 | 2,963 | 1016.9/s | 해당 없음 | 해당 없음 |
| HBM 256 MiB 왕복 copy | S0 | 3,000 | 0 | 406.5/s | 0 | 0 |

HBM 결과는 MPS active-thread percentage가 memory bandwidth 격리를 제공하지 않는다는 문제를 직접 보여준다. S0는 overlap 자체를 제거하므로 compute와 HBM 간섭을 같은 계약으로 차단한다.

## 실패한 조건과 제외한 자료

- 합성 random RX의 25 ms contract는 RAN 단독에서 1/2,500 miss가 발생해 실패했다.
- valid paired 1-cell의 9 ms contract는 RAN 단독 5/5,000 miss로 실패했다.
- valid paired 2-cell의 18 ms S1 contract는 RAN 단독 2/5,000 miss로 실패했다.
- 보호 GEMM의 초기 2 ms와 5 ms 예산은 end-to-end RPC tail을 포함하지 못해 각각 위반이 발생했다.
- Batch 1/context 16 Qwen의 50 ms-period S0 protocol은 budget violation과 release crossing 없이도 RAN 1/1,500 miss가 발생해 실패했다.
- 긴 MPS/RPC Unix socket 경로로 실패한 실행은 제외했다.
- 중단된 수동 2-cell run의 orphan Shifter child가 다음 start marker를 받아 두 RAN client가 겹친 실행은 오염된 자료로 제외했다.
- 동일 파일명과 socket으로 겹친 confirm30 중간 campaign은 `raw/contaminated_confirm30_parallel/`로 분리해 제외했다. 최종 집계에는 MPS 재시작 뒤 단독 실행한 보충 round만 사용했다.
- MPS long-path daemon이 실제로 뜨지 않은 `p0_mps_alone_job58625542.json`은 MPS 결과에서 제외했다.

실패한 deadline을 사후에 성공값으로 바꾸지 않았다. 각 다음 단계의 deadline과 seed는 protocol JSON에 실행 전에 기록했다.

## 지금 방어 가능한 주장

1. 일반 MPS 공유와 client priority만으로 RAN deadline 격리를 제공할 수 없다.
2. 협조적인 bounded AI unit에 대해 quiet-window admission과 completion ACK를 사용하면, 같은 GPU에서 valid full-PHY RAN의 deadline miss를 관측하지 않으면서 양의 AI 처리량을 회수할 수 있다.
3. 검증된 NeuralRx에서는 cap80 제한 overlap이 S0보다 높은 처리량을 제공했지만, 이는 hard guarantee가 아니라 workload별 통계 계약이다.
4. HBM-heavy workload에는 SM cap만으로 부족하며 S0 또는 별도의 memory-bandwidth 제어가 필요하다.
5. Heavy Qwen에서는 cap80이 cap100 처리량의 약 85%를 유지하면서 RAN miss를 153/5,000에서 0/5,000으로 낮췄다. 반면 light Qwen은 cap20–100에서 처리량과 RAN 결과가 거의 같아 workload shape별 qualification의 필요성을 보였다.
6. 같은 PUSCH의 optional NeuralRx와 conventional recovery를 함께 예약한 S2는 10,000 one-cell paired releases에서 eager 대비 utility 비열등성을 통과하며 background를 3.17% 늘렸고, 보수적인 host-tail budget의 3,000 two-cell releases에서는 deadline/계약 위반 없이 background를 5.53% 늘렸다.
7. 늦게 시작한 optional request를 fail-closed로 거부하고 conventional로 전환하면, 주입한 100건 모두를 포함한 1,000 releases에서 correct 1,000/1,000과 miss 0을 유지했다.
8. Bound 위반 client의 late response를 drain한 뒤 process를 즉시 종료하는 동작도 안전하지 않았다. Confirm30에서 종료 뒤 cap100 6/5,000, cap20 11/5,000 D35 misses가 발생해 teardown을 maintenance window로 분리해야 한다.
9. Confirm31의 maintenance 결과는 다른 campaign과 release 구간이 겹쳐 frozen PASS 근거에서 제외했다. Maintenance 복귀 경로는 단독 재실행 전까지 방어하지 않는다.
10. 별도 persistent cap80 MPS process와 request-specific CUDA IPC payload를 사용한 confirm32는 clean 10,000 releases에서 correct/miss/timeout gate를 모두 통과했다.
11. 같은 외부 endpoint의 confirm33은 자연 fallback 221건 중 47건을 conventional로 복구하면서 10,000 releases의 deadline miss와 timeout을 모두 0으로 유지했다.
12. Confirm36은 동일 node·trace의 외부 S2가 강한 eager-dual 대비 radio utility 비열등성을 만족하며 conventional 실행 9,797회를 생략함을 보였다.
13. Confirm38에서는 동일 조건의 세 방식 모두 RAN miss와 AI budget 위반 0이고, 별도-process external S2가 eager 대비 완료 AI +2.93%, radio utility 비열등성을 동시에 만족했다. 이 처리량 결과는 단일 순서·seed라 반복 전까지 관측 이득이다.
14. Confirm42의 외부 실제 transaction 통합은 500회에서 복구 예약·single commit·AI gap lease와 RAN deadline을 함께 만족했다. 처리량 인과 효과와 희귀 tail은 다음 ablation·반복의 대상이다.

아직 방어할 수 없는 주장은 “모든 AI에 대한 완전 격리”, “미검증 workload에 대한 hard deadline”, “실제 0.5 ms TTI 충족”, “background 회수·다중 endpoint까지 포함한 최종 별도-process 시스템”, “다중 GPU/P2P/GDR까지 포함한 최종 시스템 완성”이다.

## 다음 필수 단계

1. 외부 CUDA-IPC 경로의 실제 recovery 예약과 gap AI lease를 Confirm42에서 연결했다. 이를 끄고 켠 동일조건 ABBA와 독립 seed 반복으로 AI 이득을 분리하고, timeout 뒤 buffer/credit 수명과 late response를 end-to-end 검증한다.
2. 동일 trace를 공유한 logical cell에서 실제 독립 channel/user trace와 비동기 arrival burst로 확장한다.
3. Confirm30에서 재현된 client teardown 뒤 42–100 ms conventional GPU tail을 MPS server cleanup, CUDA context destruction, LDPC launch와 device synchronization으로 분해한다. 실시간 구간에는 논리 quarantine만 수행하고 teardown을 maintenance window로 옮기는 상태기를 구현한다.
4. Qwen decode, batch/context 추가 조합과 client 수를 확장하고 cap80이 실패하면 즉시 S0 또는 reject로 내리는 정책을 평가한다.
5. MCS/channel model, offered load와 deadline을 넓혀 긴 독립 run과 confidence bound를 수행한다.
6. 강한 결합 baseline과 다중 GPU DART/v0/v1을 동일 정보·GPU 예산으로 비교한다.

## 재현 파일

- 환경/공통 실행: `scripts_for_node/softwall_same_gpu/common.sh`, `setup_env.sh`, `mps_runtime.sh`
- RAN: `paired_phy.py`, `paired_ran_fullpath.py`
- AI: `ai_worker.py`
- 보호: `protected_runner.py`
- 캠페인: `run_confirmatory_campaign.sh`, `run_s1_nrx_sweep.sh`
- 분석: `analyze_campaign.py`, `analyze_s1.py`
- 최종 S0 결과: `confirm7_paired2_s0_job58625542_d21.*`
- 최종 S1 결과: `confirm7_paired2_s1_job58625542_d21.*`
- workload 범위 결과: `confirm8_paired2_synthai_job58625542_d21.*`
- Qwen S0 경계 결과: `confirm9_paired2_qwen_prefill_job58625542_d21.*`
- Qwen light S1 sweep: `confirm10_paired2_qwen_s1_job58625542_d21.*`
- Qwen heavy S1 최종 결과: `confirm12_paired2_qwen_heavy_final_job58625542_d21.*`
- AI disconnect fault 결과: `confirm13_ai_disconnect_job58625542.md`
- AI stale-completion fault 결과: `confirm14_ai_stale_completion_job58625542.md`
- S2 종합 보고서: `S2_NATURAL_CHANNEL_REPORT_KO.md`
- 자연 채널/비열등성/두 셀 결과: `confirm18`, `confirm21`, `confirm24`
- fail-closed admission 결과: `confirm22_s2_failclosed_admission.json`
- transaction fault tests: `confirm25_dart_runtime_fault_tests.json`
- physical overrun 및 lifecycle: `confirm26_physical_overrun.*`–`confirm31_maintenance_requalification.*`
- persistent same-request CUDA IPC: `confirm32_same_request_persistent_ipc.*`, `confirm33_same_request_natural_s2.*`, `confirm36_same_node_baselines.*`
