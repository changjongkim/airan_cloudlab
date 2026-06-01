# NSYS SQLite Deep Root-Cause 분석

생성일: 2026-06-01  
소스: `cloudlab_results/results/20260531/nsys_sqlite_v2/*.sqlite`

## 핵심 발견

기존 분석의 가장 큰 문제는 `kernel-to-kernel gap`을 거의 그대로 scheduling idle처럼 읽었다는 점이다. SQLite 원본을 다시 보면, 긴 kernel gap 상당수는 완전히 빈 시간이 아니라 **kernel 사이에 memcpy/memset activity가 끼어 있는 구간**이다. 따라서 문제 지점은 단순한 GPU idle이 아니라, L1 pipeline의 반복적인 `convert -> copy/memset -> 다음 cuPHY kernel` boundary에서 생기는 temporal gap이다.

이 관점에서는 주장이 더 선명해진다.

- 2g L1은 kernel-only p99 gap이 크게 증가한다. 이것은 small partition이 L1 pipeline의 scheduling headroom을 줄인다는 증거다.
- 3g L1 + NeuralRx는 all-activity busy time과 memory activity가 증가한다. 이것은 NeuralRx가 L1 주변의 copy/memset/runtime boundary를 더 무겁게 만든다는 증거다.
- kernel-only gap p99가 커도 all-activity gap p99는 매우 작을 수 있다. 이 경우는 "GPU가 비어 있었다"가 아니라 "kernel 사이에 memory operation이 끼어 있었다"로 해석해야 한다.

## 1. Kernel-only gap vs all-activity gap

아래 표에서 `kernel_gap_p99_us`는 kernel만 보고 다음 kernel까지의 gap을 계산한 값이다. `all_activity_gap_p99_us`는 kernel, memcpy, memset interval을 모두 합쳐 GPU activity timeline을 만든 뒤 남는 gap이다.

핵심은 두 값의 차이다. kernel-only gap이 수백~천 us인데 all-activity gap은 수 us 수준이면, 그 시간은 진짜 idle이 아니라 memory/copy/set activity가 채우고 있다는 뜻이다.

| condition | runs | kernel_busy_pct | all_gpu_busy_pct | kernel_gap_p99_us | all_activity_gap_p99_us | big_kernel_gaps_ge_1ms | big_kernel_gaps_with_mem_pct | mean_mem_fraction_inside_big_gaps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L1 2g alone | 3 | 15.6 | 47.7 | 1444 | 242.6 | 682 | 96.7 | 87.5 |
| L1 3g + ResNet | 3 | 17.4 | 41.3 | 925 | 226.3 | 69 | 77.8 | 43.6 |
| L1 3g + ResNet+ChanPred | 3 | 19.0 | 42.8 | 874 | 175.2 | 37 | 62.9 | 24.3 |
| L1 3g + ResNet+Forecaster | 3 | 18.7 | 43.9 | 919 | 172.5 | 10 | 59.7 | 16.5 |
| L1 2g + ChanPred | 3 | 16.9 | 51.2 | 1404 | 175.1 | 650 | 99.1 | 92.2 |
| L1 3g alone | 3 | 18.8 | 39.7 | 808 | 189.1 | 19 | 45.3 | 3.6 |
| L1 3g + NeuralRx | 3 | 16.3 | 40.5 | 986 | 228.9 | 83 | 72.4 | 37.8 |

## 2. 긴 gap은 어느 transition에서 생기는가

아래 표들은 각 조건에서 500us 이상 gap이 생기는 transition을 p99 gap 기준으로 정렬한 것이다. 반복적으로 보이는 문제 transition은 `convert_kernel -> ch_est_pre`, `convert_kernel -> noise_intf_est`, `convert_kernel -> eq_coef`, `copy_float32 -> convert_kernel`, `copy_complex64 -> convert_kernel` 계열이다.

이것은 L1의 특정 PHY compute kernel 하나가 오래 걸린다는 뜻이 아니다. 오히려 `convert/copy` 이후 다음 PHY stage로 넘어가는 boundary에서 tail이 생긴다. AI-RAN 관점에서는 이 boundary가 frame pipeline의 fragile point다.

### L1 3g alone

| transition | count | p50_gap_us | p99_gap_us | max_gap_ms | mean_mem_us_inside | mean_mem_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| convert_kernel -> convert_kernel | 6 | 71182.1 | 120616.5 | 120.6 | 22.2 | 0.0% |
| convert_kernel -> copy_complex64 | 3 | 116059.2 | 116854.6 | 116.9 | 9.5 | 0.0% |
| convert_kernel -> copy_float16 | 3 | 74686.3 | 80284.5 | 80.3 | 0.0 | 0.0% |
| ch_est_filter -> copy_complex64 | 3 | 76615.2 | 77921.8 | 77.9 | 0.0 | 0.0% |
| copy_complex64 -> copy_float32 | 6 | 73469.4 | 73966.2 | 74.0 | 0.0 | 0.0% |
| copy_complex64 -> convert_kernel | 15 | 2000.6 | 16564.9 | 16.6 | 0.0 | 0.0% |
| convert_kernel -> eq_coef | 7 | 966.3 | 4808.2 | 4.8 | 17.4 | 1.6% |
| copy_float32 -> convert_kernel | 7 | 1469.0 | 4004.0 | 4.0 | 0.0 | 0.0% |
| convert_kernel -> noise_intf_est | 7 | 962.5 | 1876.7 | 1.9 | 59.0 | 5.5% |
| copy_float32 -> copy_complex64 | 2 | 1238.4 | 1238.4 | 1.2 | 0.8 | 0.1% |
| convert_kernel -> ch_est_pre | 1920 | 681.2 | 876.1 | 4.4 | 632.4 | 88.9% |
### L1 3g + NeuralRx

| transition | count | p50_gap_us | p99_gap_us | max_gap_ms | mean_mem_us_inside | mean_mem_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| convert_kernel -> copy_complex64 | 3 | 109402.1 | 109969.6 | 110.0 | 10.0 | 0.0% |
| copy_complex64 -> convert_kernel | 58 | 1377.3 | 76266.6 | 76.3 | 0.0 | 0.0% |
| ch_est_filter -> copy_complex64 | 3 | 73877.8 | 73992.3 | 74.0 | 0.0 | 0.0% |
| convert_kernel -> copy_float16 | 3 | 72311.1 | 72618.2 | 72.6 | 0.0 | 0.0% |
| copy_complex64 -> copy_float32 | 6 | 70640.6 | 72177.0 | 72.2 | 0.0 | 0.0% |
| convert_kernel -> convert_kernel | 6 | 69468.3 | 70854.0 | 70.9 | 41.0 | 0.1% |
| convert_kernel -> noise_intf_est | 36 | 1387.5 | 69276.6 | 69.3 | 119.2 | 13.2% |
| convert_kernel -> eq_coef | 36 | 1380.1 | 62632.5 | 62.6 | 92.6 | 10.0% |
| copy_float32 -> convert_kernel | 33 | 1318.5 | 4067.8 | 4.1 | 0.0 | 0.0% |
| convert_kernel -> ch_est_pre | 1920 | 851.5 | 2051.7 | 37.8 | 791.9 | 89.4% |
| copy_complex64 -> copy_complex64 | 3 | 591.8 | 757.7 | 0.8 | 0.0 | 0.0% |
### L1 2g alone

| transition | count | p50_gap_us | p99_gap_us | max_gap_ms | mean_mem_us_inside | mean_mem_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| convert_kernel -> copy_complex64 | 3 | 115527.7 | 117223.8 | 117.2 | 10.9 | 0.0% |
| convert_kernel -> noise_intf_est | 32 | 1521.6 | 91139.1 | 91.1 | 117.0 | 7.9% |
| convert_kernel -> copy_float16 | 3 | 75758.1 | 82150.5 | 82.2 | 0.0 | 0.0% |
| copy_complex64 -> copy_float32 | 6 | 74224.4 | 74585.8 | 74.6 | 0.0 | 0.0% |
| ch_est_filter -> copy_complex64 | 3 | 72985.4 | 74243.3 | 74.2 | 0.0 | 0.0% |
| convert_kernel -> convert_kernel | 6 | 70051.6 | 72895.1 | 72.9 | 32.6 | 0.0% |
| copy_float32 -> convert_kernel | 26 | 2005.4 | 39580.4 | 39.6 | 0.0 | 0.0% |
| convert_kernel -> eq_coef | 26 | 3426.3 | 37847.1 | 37.8 | 32.1 | 1.4% |
| copy_complex64 -> convert_kernel | 40 | 1765.0 | 6355.1 | 6.4 | 0.0 | 0.0% |
| convert_kernel -> ch_est_pre | 1920 | 1277.5 | 2390.5 | 5.2 | 1257.7 | 93.1% |
| copy_complex64 -> copy_complex64 | 2 | 819.6 | 819.6 | 0.8 | 0.0 | 0.0% |
| copy_float32 -> copy_complex64 | 1 | 790.0 | 790.0 | 0.8 | 2.0 | 0.3% |
### L1 2g + ChanPred

| transition | count | p50_gap_us | p99_gap_us | max_gap_ms | mean_mem_us_inside | mean_mem_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| convert_kernel -> copy_complex64 | 3 | 119617.6 | 120639.5 | 120.6 | 9.7 | 0.0% |
| convert_kernel -> eq_coef | 2 | 81317.8 | 81317.8 | 81.3 | 18.9 | 0.6% |
| convert_kernel -> copy_float16 | 3 | 79273.8 | 79522.8 | 79.5 | 0.0 | 0.0% |
| ch_est_filter -> copy_complex64 | 3 | 74195.5 | 75246.6 | 75.2 | 0.0 | 0.0% |
| copy_complex64 -> copy_float32 | 6 | 73290.9 | 73716.2 | 73.7 | 0.0 | 0.0% |
| convert_kernel -> convert_kernel | 6 | 69000.1 | 71271.5 | 71.3 | 24.8 | 0.0% |
| copy_complex64 -> convert_kernel | 8 | 615.5 | 2931.4 | 2.9 | 0.0 | 0.0% |
| copy_float32 -> convert_kernel | 2 | 2371.0 | 2371.0 | 2.4 | 0.0 | 0.0% |
| convert_kernel -> ch_est_pre | 1920 | 1273.1 | 1425.6 | 2.9 | 1220.7 | 93.6% |
| convert_kernel -> noise_intf_est | 5 | 829.1 | 1258.2 | 1.3 | 105.9 | 12.0% |
| copy_float32 -> copy_complex64 | 1 | 817.2 | 817.2 | 0.8 | 0.0 | 0.0% |
| copy_complex64 -> copy_complex64 | 1 | 512.1 | 512.1 | 0.5 | 0.0 | 0.0% |
### L1 3g + ResNet

| transition | count | p50_gap_us | p99_gap_us | max_gap_ms | mean_mem_us_inside | mean_mem_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| convert_kernel -> copy_complex64 | 3 | 108639.1 | 114277.3 | 114.3 | 9.6 | 0.0% |
| convert_kernel -> noise_intf_est | 17 | 1578.2 | 89216.6 | 89.2 | 64.7 | 4.6% |
| convert_kernel -> copy_float16 | 3 | 73000.4 | 76796.3 | 76.8 | 0.0 | 0.0% |
| ch_est_filter -> copy_complex64 | 3 | 71871.5 | 74365.8 | 74.4 | 0.0 | 0.0% |
| copy_complex64 -> copy_float32 | 7 | 72244.4 | 73796.2 | 73.8 | 0.0 | 0.0% |
| convert_kernel -> eq_coef | 19 | 2336.6 | 71374.1 | 71.4 | 32.7 | 2.2% |
| convert_kernel -> convert_kernel | 6 | 69961.5 | 70298.2 | 70.3 | 31.5 | 0.0% |
| copy_float32 -> convert_kernel | 12 | 1872.3 | 39117.7 | 39.1 | 0.0 | 0.0% |
| copy_complex64 -> convert_kernel | 27 | 1633.6 | 3793.3 | 3.8 | 0.0 | 0.0% |
| convert_kernel -> ch_est_pre | 1920 | 850.5 | 1093.3 | 3.6 | 749.8 | 89.7% |
| copy_complex64 -> copy_complex64 | 2 | 534.1 | 534.1 | 0.5 | 0.0 | 0.0% |
### L1 3g + ResNet+ChanPred

| transition | count | p50_gap_us | p99_gap_us | max_gap_ms | mean_mem_us_inside | mean_mem_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| convert_kernel -> copy_complex64 | 3 | 114152.1 | 114818.0 | 114.8 | 10.7 | 0.0% |
| ch_est_filter -> copy_complex64 | 3 | 74815.8 | 74888.7 | 74.9 | 0.0 | 0.0% |
| convert_kernel -> copy_float16 | 3 | 74216.4 | 74349.4 | 74.3 | 0.0 | 0.0% |
| copy_complex64 -> copy_float32 | 6 | 71955.5 | 73309.9 | 73.3 | 0.0 | 0.0% |
| convert_kernel -> convert_kernel | 6 | 70005.6 | 71214.8 | 71.2 | 30.3 | 0.0% |
| copy_float32 -> copy_complex64 | 2 | 1488.6 | 1488.6 | 1.5 | 3.7 | 0.4% |
| convert_kernel -> noise_intf_est | 2 | 1224.9 | 1224.9 | 1.2 | 96.0 | 9.1% |
| convert_kernel -> ch_est_pre | 1920 | 745.6 | 1019.4 | 1.0 | 690.7 | 88.8% |
| convert_kernel -> eq_coef | 1 | 766.5 | 766.5 | 0.8 | 21.3 | 2.8% |
| copy_complex64 -> convert_kernel | 3 | 537.4 | 602.7 | 0.6 | 0.0 | 0.0% |
| copy_complex64 -> copy_complex64 | 1 | 555.0 | 555.0 | 0.6 | 0.0 | 0.0% |
### L1 3g + ResNet+Forecaster

| transition | count | p50_gap_us | p99_gap_us | max_gap_ms | mean_mem_us_inside | mean_mem_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| convert_kernel -> copy_complex64 | 3 | 105968.3 | 112247.1 | 112.2 | 9.8 | 0.0% |
| ch_est_filter -> copy_complex64 | 3 | 73777.8 | 78704.0 | 78.7 | 0.0 | 0.0% |
| convert_kernel -> copy_float16 | 3 | 73186.3 | 77349.0 | 77.3 | 0.0 | 0.0% |
| copy_complex64 -> copy_float32 | 6 | 70515.7 | 73714.4 | 73.7 | 0.0 | 0.0% |
| convert_kernel -> convert_kernel | 6 | 71121.4 | 72379.6 | 72.4 | 40.1 | 0.1% |
| convert_kernel -> noise_intf_est | 1 | 1169.9 | 1169.9 | 1.2 | 62.5 | 5.3% |
| convert_kernel -> ch_est_pre | 1920 | 845.9 | 992.8 | 2.0 | 732.8 | 89.7% |
| copy_complex64 -> convert_kernel | 5 | 583.5 | 991.3 | 1.0 | 0.0 | 0.0% |
| copy_complex64 -> copy_complex64 | 2 | 567.7 | 567.7 | 0.6 | 0.0 | 0.0% |
| convert_kernel -> eq_coef | 1 | 518.7 | 518.7 | 0.5 | 248.0 | 47.8% |

## 3. Memory operation 자체가 workload별로 다르다

아래 표는 selected condition에서 Device-to-Device memcpy와 memset을 요약한 것이다. NeuralRx, ResNet, ChanPred 조합은 copy/set duration과 bytes pattern이 다르다. 그래서 generic D2D/H2D saturation으로는 실제 PHY-AI의 temporal pattern을 완전히 재현하기 어렵다.

| condition | op | count | duration_ms | bytes_mb |
| --- | --- | --- | --- | --- |
| L1 2g alone | Device-to-Device | 7 | 0.0 | 0.0 |
| L1 2g alone | memset | 1920 | 814.5 | 301433.1 |
| L1 3g + ResNet | Device-to-Device | 7 | 0.0 | 0.0 |
| L1 3g + ResNet | memset | 1920 | 407.6 | 301433.1 |
| L1 3g + ResNet+ChanPred | Device-to-Device | 7 | 0.0 | 0.0 |
| L1 3g + ResNet+ChanPred | memset | 1920 | 407.9 | 301433.1 |
| L1 3g + ResNet+Forecaster | Device-to-Device | 7 | 0.0 | 0.0 |
| L1 3g + ResNet+Forecaster | memset | 1920 | 407.8 | 301433.1 |
| L1 2g + ChanPred | Device-to-Device | 7 | 0.0 | 0.0 |
| L1 2g + ChanPred | memset | 1920 | 813.9 | 301433.1 |
| L1 3g alone | Device-to-Device | 7 | 0.0 | 0.0 |
| L1 3g alone | memset | 1920 | 408.2 | 301433.1 |
| L1 3g + NeuralRx | Device-to-Device | 7 | 0.0 | 0.0 |
| L1 3g + NeuralRx | memset | 1920 | 407.8 | 301433.1 |

## 4. 수정된 해석

NSYS에서 진짜로 말할 수 있는 것은 다음이다.

1. **문제 지점은 raw HBM bandwidth 하나가 아니다.** Kernel-only gap과 all-activity gap이 크게 다르기 때문에, 긴 gap은 대부분 GPU가 완전히 노는 시간이 아니라 copy/memset/runtime activity가 L1 kernel 사이에 끼어든 결과다.

2. **L1 pipeline의 취약 지점은 convert/copy boundary다.** `convert_kernel` 이후 channel estimation, noise/interference estimation, equalization으로 넘어가는 transition에서 p99 gap이 커진다. 이 boundary가 AI co-tenant와 partition size 변화에 민감하다.

3. **2g L1은 scheduling headroom이 부족하다.** 2g 조건은 L1 alone에서도 kernel gap p99가 3g보다 크다. AI workload가 없어도 작은 partition 자체가 위험한 출발점이다.

4. **NeuralRx는 단순 compute stress가 아니다.** NeuralRx 조건은 memory activity와 runtime/API boundary를 더 무겁게 만든다. 그래서 generic D2D/H2D/GEMM stress가 안전하다는 6/1 F 결과와 NeuralRx/coloc failure는 모순이 아니다.

5. **논문 문장은 이렇게 좁혀야 한다.** "MIG는 bandwidth isolation이 완전히 안 된다"가 아니라, "MIG는 static capacity isolation을 제공하지만, AI-RAN L1 pipeline의 convert/copy/runtime boundary에서 필요한 temporal scheduling guarantee를 제공하지 못한다"가 정확하다.
