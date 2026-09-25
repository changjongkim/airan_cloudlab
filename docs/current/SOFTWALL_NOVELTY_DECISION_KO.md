# SoftWall 노벨리티 판정과 종료 기준 — 2026-09-21

**2026-09-22 후속:** 아래의 2026-09-21 현황은 당시 종료 판정으로 보존한다.
[후속 검증 기록](SOFTWALL_VALIDATION_PROGRESS_KO.md)의 Confirm56은 복구 전체 host 경로
계측, Confirm57은 새 독립 seed의 synthetic channel feature calibration을 통과했다.
이전 중복 seed 실험의 무효 판정은 유지하며, 단순 복구 자리 이동은 AI 인지 greedy도
재현했다. **독립 공동 정책의 노벨리티는 여전히 미확보다.**

## 현재 판정

**탑컨퍼런스급 독립 노벨리티는 아직 입증되지 않았다.** 같은 GPU에서 두 셀 NeuralRx와 AI를
MPS로 돌린 [Confirm55](../../results/softwall_same_gpu/confirm55_two_endpoint_bringup.md)는
P150/D130·1,000회에서 1,958/2,000 무선 정답, deadline miss 0, AI 29,910 units를 보였다.
이는 구현 가능성의 증거이지 기존 기법 결합보다 낫다는 증거가 아니다. NeuralRx 거절이 0이고,
두 셀 동시 자연 fallback은 1회뿐이다. 마지막 무선 작업 이후 다음 release까지 남은 시간의
중앙값은 144.815 ms(실제 다음 release가 있는 999구간)로, 정책을 구별하기에는 매우 여유 있다.
[원자료 재계산](../../results/softwall_same_gpu/confirm55_novelty_gap.json)에 해시와 집계를 보존했다.
원자료의 2,000개 NRx 완료 시각을 각 요청의 fallback cutoff와 대조하면 최소 여유도
**66.273 ms**이고, 1,000/1,000 release에서 두 NRx 모두 더 이른 셀의 cutoff 전에 끝났다.
즉 이 trace에서는 셀별 복구 순서를 바꿔 cutoff를 재배정하는 정책도 수락 결정의 차이를
만들 기회가 없었다. 이 관측은 다른 부하에서의 효과 부재를 뜻하지 않는다.

**투고 범위도 수정해야 한다.** [NSDI 2027 공식 CFP](https://www.usenix.org/conference/nsdi27/call-for-papers)는 GPU resource scheduling을 명시적 제외 주제로 분류하고, 네트워크 시스템·스택 설계 기여가 없는 원고는 범위 밖이라고 한다. SoftWall을 현재와 같은 GPU 스케줄링 중심으로 쓰면 NSDI 적합성 위험이 높다. 반면 [SIGMETRICS 2027 CFP](https://www.sigmetrics.org/sigmetrics2027/pages/cfp.html)는 measurement·modeling·scheduling·real-time systems를 명시한다. 이것은 게재 가능성이나 노벨리티의 증거가 아니라 주제 적합성의 판단이다.

더 직접적인 미완료 항목은 **baseline과 구별되는 공동 정책 자체가 아직 코드로 정의되지 않았다는 점**이다.
현재 런타임의 mandatory-first 예약과 endpoint 선택은 안전 장치이며, 이 장치만으로는 새로운
목적함수나 정책의 효과를 측정할 수 없다. 원 계획의 0.5/1 ms slot-paced P0도 미완료다.
Confirm55의 무선 응답 중앙값은 약 4.88 ms이므로 P150/D130 성공을 production slot
deadline의 증거로 바꿔 쓸 수 없다.
0.5/1 ms 도착 **주기**와 요청별 HARQ **expiry**를 같은 값으로 취급해서도 안 된다.
P0의 수정된 gate는 expiry를 먼저 유도하고, AI 없는 필수 경로가 그 부하에서 통과하는지
확인한다. Confirm55의 48개 conventional fallback GPU event는 2.224–2.772 ms였으므로
현재 경로에 1 ms 이하의 요청별 expiry를 부여하는 주장은 이 관측과 맞지 않는다.
구체적인 [타이밍 계약 감사](SOFTWALL_PUSCH_TIMING_CONTRACT_KO.md)에서 대상 DU의 `d_MAC`가
프로젝트에 아직 없음을 확인했다. 3GPP의 K2/N2는 각각 UE 전송 슬롯 오프셋/준비 시간이라
gNB의 CRC 소비 기한을 직접 정하지 않는다.
또한 현 예약용 `B_conv=25 ms`·단일 필수 lane에서 모든 NRx 실패를 허용하는 지속 부하는
`셀 수×25/P≤1`이 필요하다. 1셀 P1/P0.5 ms도 각각 25배/50배 용량 초과다.
deadline D를 늘리거나 MPS cap을 조정해도 이 큐 안정성 필요조건은 사라지지 않는다.
이는 현 25 ms 복구 계약에서의 용량 판정이며 최적화한 PHY의 불가능성 증명은 아니다.
생산 슬롯 속도의 모든-실패 보장에는 새 필수 경로 용량 계약이 필요하다.

**채널 계약도 먼저 고정해야 한다.** 기존 Rayleigh trace는 fading 블록마다 그 블록의
신호 전력에 맞춰 AWGN 전력을 다시 설정했다. 별도 [feature calibration](../../results/softwall_same_gpu/channel_feature_gate_job58693851.md)은 같은 nominal −8.5 dB의 고정 pre-fading 잡음 계약에서 NeuralRx 정답이 487/500에서 216/500으로 바뀌는 것을 관측했다. 이는 다른 합성 채널 계약을 기존 radio utility 결과에 그대로 대입할 수 없다는 뜻이며 현장 채널의 정답을 결정하지 않는다. 그 calibration의 train/test 채널 seed 499/500개가 겹쳐 gate 자격 판정은 **무효**, feature 첫 호출 303.932 ms로 사전 비용 gate도 실패했다. 이 자료를 강한 baseline의 검증된 online 채널 gate로 사용하지 않는다.

현재 제어 루프는 두 NeuralRx의 물리 완료를 기다리고 필요한 conventional 복구를 끝낸 다음에
AI를 시작한다. 따라서 AI 허가 시점에는 복구 위험이 남지 않는다. **동일 조기 복구·동일 AI
work-unit과 bound를 가진 강한 결합 baseline은 이 실행 순서를 그대로 재현할 수 있다.**
Confirm55 원자료의 AI 허가 시각 29,910개를 두 셀의 무선 완료 시각과 직접 대조하면,
무선 완료 전 허가는 **0개**다([재계산](../../results/softwall_same_gpu/confirm55_novelty_gap.json)).
이 구조에서 AI 처리량 차이가 나더라도 공동 의사결정 자체의 효과로 해석할 근거가 없다.
주기 P만 줄여도 이 코드에서는 항상 `두 NRx 대기 → 필요한 복구 → AI` 순서이므로
공동 정책의 새 결정은 생기지 않는다. P를 줄여 deadline이나 AI 처리량이 깨지는 지점을
찾는 일은 부하 진단일 수 있지만, 그 자체가 강한 baseline 대비 새 기여의 실험은 아니다.
Confirm53의 알려진 CRC 실패 조기 복구는 무선 정답 손실을 줄였지만, AI 처리량 우위는 두
paired seed에서 방향이 반대였다. 조기 복구는 공정하게 baseline에도 제공한다.

외부 Aerial TDL-A 채널의 [잡음 없는 10-TTI 호환성 canary](../../results/softwall_same_gpu/aerial_tdl_compatibility_job58694167.md)는 conventional 10/10, NeuralRx 0/10으로 사전 gate에 실패했다. 원인 분리 전에는 이 채널에서 noisy SNR·AI·deadline 캠페인을 진행하지 않는다. 이는 NeuralRx의 모든 TDL 조건 실패나 실환경 성능을 뜻하지 않지만, 현재 synthetic Rayleigh 결과의 외부 채널 일반화 주장도 허용하지 않는다.
참고 NeuralRx 예제의 1×4 uplink 기하로 정정한 [독립 canary](../../results/softwall_same_gpu/aerial_tdl_1x4_compatibility_job58694384.md)에서도 conventional 10/10, NeuralRx 0/10이었다. 송신 기하 불일치만으로 첫 실패를 설명할 수 없으며, 텐서·채널 정규화 및 DMRS 계약을 분리하기 전 TDL-A 확장 실험을 하지 않는다.

**현재 실험 단계의 판정은 종료한다: 노벨리티 미확보.** 추가 seed나 같은 경로의 GPU run으로 판정을 연장하지 않는다. 이후 연구는 공동 정책이 강한 결합 baseline과 실제로 다른 결정을 만드는 설계와, 그 차이를 성립시키는 PHY 계약을 갖춘 경우에만 새 단계로 개시한다.

후속 설계 가설은 [복구 의무를 포함한 공동 스킴 후보](SOFTWALL_JOINT_SCHEME_PROPOSAL_KO.md)에
입력·행동·안전 불변식·비교 baseline·유한 go/no-go gate로 구체화했다. 이는 제안이며 현 실험
단계의 실패 판정을 바꾸지 않는다.

### 다음 후보의 즉시 판정

요청마다 비교할 행동은 `(NRx 수락 여부, endpoint, 복구 시작 시각, AI unit 허가 시각)`이다.
동일한 온라인 정보와 물리 상한에서 두 정책의 행동이 다르고, **양쪽 모두 모든 허용 실패에서
복구 가능**해야 GPU 비교를 시작할 이유가 있다. 행동 차이만으로 신규성이나 성능 우위가
입증되는 것은 아니다.

| 후보 | 지금의 판정 | 이유 |
|---|---|---|
| 두 NRx 완료 → 필요 복구 → 남은 구간 AI | **기각** | 현재 실행 순서이고 강한 결합 baseline도 그대로 실행할 수 있다. |
| 채널/queue를 보고 고정 fallback 순서만 정하기 | **단독 기여로 기각** | 같은 정보를 받은 baseline도 순서를 고를 수 있고, 기존 스케줄링의 자연스러운 조합이다. |
| 미완료 NRx가 있을 때 짧은 AI unit을 허가하고 복구 credit을 동적으로 다시 배정하기 | **조건부 후보** | AI–NRx 및 AI–conventional 동시 서비스 상한, 버퍼 수명, 실제 expiry가 아직 없다. 현재 controller는 이 행동을 실행하지 않는다. |
| 먼저 끝난 NRx의 실패를 복구하면서 다른 NRx를 계속 실행하기 | **조건부 후보** | 현 `wait-both`보다 실행 선택지가 늘지만 NRx–conventional 동시 상한이 미검증이다. 조기 복구 primitive 자체는 baseline에도 제공해야 한다. |

따라서 **현재 코드로 즉시 돌릴 수 있는 신규 공동 정책 후보는 0개**다. 위 조건부 후보 중
하나라도 계약과 요청별 행동 차이를 먼저 제시하지 못하면, 강한 baseline GPU campaign을
시작하지 않고 SoftWall의 공동 제어 논지를 종료한다. 후보가 성립해도 기존 결합 대비
효과와 선행 연구 차이를 별도로 검증해야 한다.

**대안 C1 측정 기여도 아직 확정되지 않았다.** 기존 273조건의 `cudaFree`–GPU gap 시간적 겹침은 malloc/free가 많은 harness의 상관관계이며 allocator 제거 개입 전에는 인과 증명이 아니다. 별도 [Confirm8](../../results/softwall_same_gpu/confirm8_paired2_synthai_job58625542_d21.json)의 valid PUSCH에서는 무제어 GEMM 17/3,000 miss, HBM 2,963/3,000 miss로 간섭의 존재가 재확인됐지만 두 결과를 같은 4레벨 원인 체인으로 합칠 수 없다. C1을 독립 논문으로 승격하려면 allocator 전후의 matched 반증과 held-out 예측력·선행연구 차이가 추가로 필요하다.

## 남은 유한 판정

1. **정책 차이를 먼저 명시:** 같은 정보·물리 자원·조기 복구·수명 규칙을 가진
   [강한 결합 baseline](SOFTWALL_STRONG_BASELINE_SPEC_KO.md)과 비교할 때, 공동 정책이
   어느 *미해결 요청 또는 미래 복구 의무* 때문에 다른 NRx 수락·복구 시각·AI 허가 결정을
   하는지 요청별 trace로 확인한다. 차이가 없는 후보는 GPU campaign 전에 기각한다.
   그 전에 고정 잡음·fading·실제 PHY workload 계약과 서로 겹치지 않는 train/test trace를 정한다.
2. **물리 실행 가능성:** 각 IPC endpoint는 현재 forward/backward buffer 한 쌍만 가진다.
   request-owned PHY 입력과 endpoint-owned IPC를 분리하거나 buffer ring을 구현한다.
   NeuralRx–conventional 및 AI–mandatory 동시 실행의 서비스 상한을 같은 GPU에서 측정한다.
   현 두-endpoint controller의 timeout 분기는 물리 완료를 확인하지 못하면 복구를 중지하지만,
   `finally`에서 종료 신호 뒤 0.2초만 기다리고 IPC owner를 닫는다. 미완료 worker가 남은
   고장 경로의 버퍼 수명/정리 보장은 별도로 설계·검증해야 한다.
   특히 P150/D130·두 셀의 현 `두 NRx 대기 → 복구` 순서는 모든-실패 경로에서
   **`B_pair+2×B_conv_path+2≤130 ms`**를 요구한다. `B_conv_path`가 host 결정부터
   commit까지 25 ms 이내로 자격 검증될 때에만 `B_pair≤78 ms`라고 쓸 수 있다.
   기존 25 ms gate는 GPU event에만 적용했다. 개별 50 ms 상한 두 개나 commit 호출 전
   관측 최대 13.503 ms를 전체 공동 상한으로 대체하지 않는다. 다른 실행 순서는
   NRx–복구 동시 bound가 필요하다.
   host 겹침만으로 GPU kernel overlap 또는 WCET를 주장하지 않는다.
3. **사전 고정 paired 비교 한 번:** 실제 경합이 있으면서 분석상 모든 fallback을 수용할
   수 있는 운영점을 먼저 고정한다. 같은 trace·offered load·GPU budget에서 두 정책의
   deadline miss, paired 무선 효용, 완료한 AI units, 실제 복구 시작, bound 위반 및
   credit 잔여를 비교한다. 동시 실패와 late NRx도 포함한다. 테스트 trace를 보고 gate나
   계약을 느슨하게 바꾸지 않는다.
4. **중단 조건:** baseline과 같은 안전·무선 요구에서 공동 정책이 반복 가능한 유효 AI
   이득을 보이지 못하거나, 동시 실행의 물리 bound를 자격 검증할 수 없으면 공동 제어
   우월성 주장을 중단한다. seed를 추가해 판정을 미루지 않는다. 조건부 구성 결과는
   별도로 보고할 수 있지만 “MPS 완벽 격리”나 hard deadline 보장으로 표현하지 않는다.

이 네 단계 이후에도 실제 PHY expiry와 HARQ timing에서 같은 결과를 입증하지 못하면
실운영 AI-RAN 보장 주장을 하지 않는다. 최근 P150/D130 결과를 이 단계의 대체 증거로
계산하지 않는다.
