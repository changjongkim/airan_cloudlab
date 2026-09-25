# Aerial TDL-A 1×4 수신기 호환성 canary — 2026-09-21

첫 [4×4 canary](aerial_tdl_compatibility_job58694167.md) 뒤 NVIDIA pyAerial의 [NeuralRx 예제](../../third_party/aerial-cuda-accelerated-ran/pyaerial/notebooks/example_neural_receiver.ipynb)가 1 UE 송신 안테나·4 gNB 수신 안테나를 사용한다는 입력 계약 차이를 발견했다. 이에 별도 [사전 고정 protocol](aerial_tdl_1x4_compatibility_protocol.json)로 원래 한 스트림의 4열 precoding을 정확히 역산한 다음, Aerial TDL-A 1×4 uplink(30 ns, 10 Hz, AWGN 없음)에 10 TTI를 통과시켰다.

Job `58694384`, node `nid001160`의 [원자료](raw/aerial_tdl_1x4_compatibility_job58694384.json)에서 precoding 재구성 최대 절대오차는 **0.0**, 채널 입력은 `[1,1,1,14,3276]`이었다. Conventional은 CRC·payload **10/10**, NeuralRx는 **0/10** 정답이다. [실행 로그](aerial_tdl_1x4_compatibility_allocation.log)와 원자료 SHA-256 `e056a8da7217aa609e4cca68b1095138157785d7025540f51e0abdf60570c7fa`를 보존한다.

**Frozen gate FAIL.** 4×4 대신 1×4 기하를 사용해도 0/10이므로 앞선 실패를 송신 안테나 수 불일치만으로 설명할 수 없다. 이것은 그 가설의 반증이며 NeuralRx 자체의 모든 TDL-A 조건 실패를 입증하지는 않는다. 남은 후보는 입력 신호의 정규화·Aerial TDL 출력의 수치적 계약, NeuralRx의 채널/DMRS 분포, 또는 다른 인터페이스 문제다. 이 10 TTI로 BLER, field 성능, MPS deadline 또는 정책 노벨리티를 추론하지 않는다.

Protocol에 따라 TDL-A noisy sweep과 AI/deadline GPU campaign은 중단한다. 새 실험이 필요하다면 먼저 동일한 송수신기 입력에 대한 텐서 단위 동등성·정규화 검사와 참고 Sionna 채널의 독립 대조를 사전 고정해야 한다.
