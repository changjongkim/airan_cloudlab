# Aerial TDL-A 수신기 호환성 canary — 2026-09-21

사전 고정 [retry3 protocol](aerial_tdl_compatibility_protocol_retry3.json)에 따라 job `58694167`의 동일 PUSCH 10 TTI를 Aerial TDL-A(30 ns delay spread, 10 Hz Doppler, 4×4 uplink, AWGN 없음)에 통과시켰다. [원자료](raw/aerial_tdl_compatibility_job58694167.json)와 [실행 로그](aerial_tdl_compatibility_retry3_allocation.log)를 보존한다.

| 수신기 | CRC와 payload 모두 정답 |
|---|---:|
| Conventional | 10/10 |
| NeuralRx | 0/10 |

**Frozen gate FAIL.** Conventional은 같은 입력을 모두 복호했으므로 채널 생성과 전체 PHY 경로가 적어도 conventional 수신기에 대해서는 작동했다. NeuralRx 실패의 원인은 모델의 채널 일반화 부족, 입력 표현/인터페이스 불일치 또는 다른 원인 중 아직 분리되지 않았다. 10 TTI는 BLER 추정이나 field 성능 주장의 근거가 아니다. 이 canary는 MPS, AI 처리량, deadline 또는 공동 정책을 평가하지 않았다.

Protocol의 중단 조건에 따라 이 TDL-A branch의 noisy SNR sweep과 AI/deadline campaign을 실행하지 않는다. 다시 시작하려면 먼저 동일 no-AWGN TTI에서 NeuralRx 입력·채널 응답·학습 계약을 대조하고 10/10 호환성 gate를 새 protocol로 통과해야 한다. 실패한 canary를 합성 Rayleigh 결과의 외부 채널 검증으로 세지 않는다.
