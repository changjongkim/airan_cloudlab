# 2026-06-14 CloudLab d8545 재측정 — 요약

새 노드(d8545-10s10501.wisc.cloudlab.us), driver 550 / CUDA 12.4 환경. MIG GPU 0 split-60-40 (4g + 3g), L1은 3g (6/1 캠페인과 동일한 partition).

## 주요 발견

| 실험 | 조건 | n | p50 | p99 | max | 기준 | 결과 |
|---|---|---:|---:|---:|---:|---|---|
| **E0** | L1 alone 3g | 500 | 37.5 | **47.8** | 54.5 | 6/1 F_0_alone p99=59.2 | ~14% 빠름 (절대값 다름) |
| **E1** | cross-part + NeuralRx | 1000 | 37.6 | **43.6** | 52.4 | 5/31 phase4 p99=205 | ❌ **205ms 재현 안 됨** |
| **E2** | cross-part + chanpred | 500 | 37.7 | **42.2** | 43.8 | 6/1 F_E_chanpred p99=45.2 | ✅ 거의 일치 |
| **E3** | **same-part coloc** | 500 | 353.5 | **365.9** | 366.1 | 6/1 G_coloc p99=361.1 | ✅ **완벽 재현** |
| E4-a | cross-part + xapp | 500 | 37.4 | 39.4 | 40.7 | (6/1 빈 칸) | flat |
| E4-c | cross-part + sat_compute | 500 | 37.6 | 41.1 | 42.6 | (6/1 빈 칸) | flat |
| E4-d | cross-part + sat_hbm | 500 | 37.6 | 39.4 | 40.5 | (6/1 빈 칸) | flat |
| E4-b | cross-part + qwen_small | — | — | — | — | (HF download 이슈, 미실행) | 보류 |

## 핵심 결론

1. **MIG cross-partition 격리가 어제 자료(5/31)에서 본 것보다 훨씬 잘 작동**:
   - 5/31 phase4 neuralrx에서 본 p99=205ms / max=313ms bistability가 오늘 측정에선 사라짐 (p99=43.6, max=52.4)
   - 모든 cross-partition 단일 AI 워크로드가 baseline(37.5ms p99=47.8)에서 +1~+5ms 범위로 거의 영향 없음
   - 5/31 vs 6/1 vs 6/14 baseline 차이(73 → 45 → 47)도 같은 패턴 — driver/CUDA/build 환경 영향

2. **Same-partition coloc 폭락은 confirmed reproducible**:
   - L1 + chanpred 같은 3g 안: p99=365.9 (6/1 361.1 거의 일치)
   - 이게 "MIG necessary but misconfigurable" 주장의 핵심 증거

3. **F-table MIG 빈 칸 4개 중 3개 채움**:
   - xapp, sat_compute, sat_hbm 모두 cross-partition에서 flat
   - qwen_small은 HF cache 마운트 문제로 미실행 (다음에 해결)

## paper에 미치는 영향

- "MIG cross-partition 격리는 PHY-AI에도 작동" — 강화됨
- "5/31 phase4 NeuralRx 205ms" — driver/build 변동성 영향이거나 5/31 측정 환경의 특수성으로 재해석 필요
- "Same-partition coloc은 8× 폭락" — robust한 재현 가능 메시지
- "MIG 비효율적" 주장은 **(1) same-partition mis-placement (E3)** + **(2) HBM bandwidth share** + **(3) chip-wide PCIe/DMA queue** 세 가닥으로 정리하는 게 더 honest. cross-partition NeuralRx bistability는 빼는 게 안전.

## 환경 노트 (재현용)

- 노드: d8545-10s10501.wisc.cloudlab.us (Wisconsin, A100-SXM4-40GB × 4)
- driver 550.163.01, CUDA 12.4 (host), 12.9.1 (container)
- 컨테이너: `nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb` (public pull, no NGC key)
- 빌드: cmake/ninja inside container, `cmake --build build -t _pycuphy pycuphycpp -j16`, .so를 `pyaerial/src/aerial/pycuphy/`로 복사
- MIG: `nvidia-smi mig -i 0 -cgi 5,9 -C` (4g + 3g, pending state → 두 번째 reboot 필요)
- L1 partition: 3g (UUID MIG-e43dd593-…), AI partition: 4g (UUID MIG-954ae2c3-…)

## 다음

- qwen_small HF cache permission 이슈 해결 후 1회 재측정
- Perlmutter MPS / NCU / NSYS / P5 결과 push되는 대로 비교
- §F의 figure 재구성: NeuralRx + G_coloc 포함해서 그리기 — 오늘 데이터로 보면 NeuralRx는 "cross-partition에서 안전, coloc은 폭락" 그림으로 바뀜
