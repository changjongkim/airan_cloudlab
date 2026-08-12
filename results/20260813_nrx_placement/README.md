# 2026-08-13 NRx placement experiment bundle

이 디렉터리는 optimized direct-TensorRT cuPHY→NRx placement 연구의 원본 결과와 재현 가능한 분석을 담는다.

- 최종 해석: `REPORT_KO.md`
- novelty 검토: `NOVELTY_AUDIT.md`
- placement 수치: `PLACEMENT_SUMMARY.csv`
- 동일 depth transport 대조: `DEPTH1_TRANSPORT_COMPARISON.csv`
- replica capacity: `NRX_CAPACITY.csv`
- open-loop queue: `NRX_OPEN_LOOP.csv`
- per-device TensorRT tactic sensitivity: `NRX_TACTIC_SENSITIVITY.csv`
- 분석/figure 재생성: `python3 analyze_results.py`
- raw data: `raw/`

중요: 이 bundle의 direct-TensorRT 결과는 과거 `task1_final`의 약 105 ms pycuphy-wrapper 결과를 대체하는 optimized-path measurement다. 과거 결과를 삭제하지 않지만 두 세대를 같은 표에서 직접 비교하지 않는다.
