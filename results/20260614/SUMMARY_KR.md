# 2026-06-14~15 CloudLab d8545 측정 — 요약 (full)

새 노드(d8545-10s10501.wisc.cloudlab.us), driver 550 / CUDA 12.4. MIG GPU 0.

## 측정 매트릭스 한눈에

### Cross-partition (L1 on 3g, AI on 4g)

| 조건 | n | p50 | p99 | max | 6/1 기준 | 평가 |
|---|---:|---:|---:|---:|---|---|
| E0 alone (L1 on 3g) | 500 | 37.5 | **47.8** | 54.5 | F_0_alone p99=59.2 | -19% (driver/build 차이) |
| E1 + NeuralRx | 1000 | 37.6 | **43.6** | 52.4 | (5/31 phase4: p99=205) | ❌ 205ms 재현 안 됨 |
| E2 + chanpred | 500 | 37.7 | **42.2** | 43.8 | F_E_chanpred p99=45.2 | ✅ 일치 |
| E4 + xapp | 500 | 37.4 | 39.4 | 40.7 | (6/1 빈 칸) | flat |
| E4 + sat_compute | 500 | 37.6 | 41.1 | 42.6 | (6/1 빈 칸) | flat |
| E4 + sat_hbm | 500 | 37.6 | 39.4 | 40.5 | (6/1 빈 칸) | flat |

→ **cross-partition은 어떤 AI에도 ~40-44ms p99 (alone +1~+5ms)**. 격리 성공.

### Partition size sweep — L1 alone

| partition | n | p50 | p99 | max | 비고 |
|---|---:|---:|---:|---:|---|
| **2g.10gb** | 500 | **50.2** | 56.1 | 56.6 | 가장 느림 (SM/HBM 적음) |
| 3g.20gb | 500 | 37.7 | 48.2 | 52.5 | |
| 4g.20gb | 500 | 37.5 | 49.4 | 52.9 | 3g와 사실상 동일 |
| **7g.40gb (full)** | 500 | **34.4** | **36.1** | 36.8 | 가장 빠르고 tail도 가장 짧음 |

→ **3g와 4g는 alone 시 거의 동일** (HBM 20GB 공유, SM 차이만 약간). **2g는 명확히 느림** (~13ms +). **7g는 ~3ms 더 빠르고 tail이 5배 좁음** (cross-partition 격리에 비해 idle GPU 전체를 쓸 수 있어 변동성 없음).

### Partition size sweep — L1 + NeuralRx **같은 partition 안에 coloc** ⭐

| partition | n | p50 | p99 | max | alone 대비 |
|---|---:|---:|---:|---:|---|
| 2g | 1000 | 360.5 | **371.1** | 373.7 | +315ms |
| 3g | 1000 | 355.8 | **360.4** | 366.1 | +312ms |
| 4g | 1000 | 350.7 | **358.9** | 361.5 | +310ms |
| **7g (full GPU)** | 1000 | 344.5 | **356.3** | 356.5 | **+320ms** |

→ **partition을 키워도 coloc 폭락은 안 줄어듦**. 2g→7g 차이는 겨우 15ms. **같은 CUDA context 안에 L1 + AI 있으면 무조건 ~360ms로 떨어짐**.

### 6/14 E3 vs E6 cross-check

| 조건 | n | p99 | max |
|---|---:|---:|---:|
| E3 3g+chanpred coloc | 500 | 365.9 | 366.1 |
| E6 3g+NeuralRx coloc | 1000 | 360.4 | 366.1 |

→ **워크로드 종류 안 가림** (chanpred든 NeuralRx든 같은 partition coloc은 ~360ms). partition 자체의 contention이 dominant.

## 핵심 결론 (paper에 바로 들어갈 수 있는 메시지)

### 1) "MIG cross-partition 격리는 어떤 AI에도 작동한다"
- chanpred / NeuralRx / xapp / sat_compute / sat_hbm 모두 cross-partition에서 L1 alone +1~+5ms
- NeuralRx도 cross-partition이면 chanpred와 차이 없음 (둘 다 ~43-44ms)
- 5/31 phase4의 205ms bistability는 driver 525 시절 artifact로 보임 — driver 550에서 사라짐

### 2) "Same-partition coloc은 partition 크기와 무관하게 ~360ms로 폭락"
- 2g/3g/4g/7g 모두 coloc 시 p99 = 356-371ms (편차 단 ~15ms)
- **7g coloc(=사실상 full GPU)이 356ms** → "큰 GPU를 주면 해결됨" 가설 **반박됨**
- Perlmutter no-MIG NeuralRx p99=389ms와 정합 — **no-MIG = 같은 context coloc**

### 3) "Alone baseline은 partition 크기에 비례하지만 격차는 좁다"
- 2g(50ms) vs 7g(34ms) 약 1.5배 차이
- 3g와 4g는 거의 동일 → SM 보다 HBM 용량/대역폭 슬라이스가 dominant
- **7g의 36ms p99는 매우 깨끗** (variance 5배 작음) — full GPU non-interference state

## 통합 메시지

> **L1 + AI를 같은 CUDA context (= same MIG partition = no MIG full GPU)에 두는 순간 무조건 ~360ms로 폭락한다. partition을 키우는 것으로는 해결 불가. cross-partition MIG 격리만이 L1을 ~40ms로 보호한다.**

→ paper의 "MIG는 필요하다" 주장은 강하게 입증됨.
→ "MIG는 비효율적이다" 주장은 **(a) same-partition mis-placement → +320ms 폭락**, **(b) partition HBM bandwidth slice가 baseline의 1.5배 차이를 만듦**, **(c) chip-wide PCIe/DMA queue 공유 (별도 §16 메커니즘)** 세 가닥으로 정리.

## 환경 노트

- 노드: d8545-10s10501.wisc.cloudlab.us
- driver 550.163.01, CUDA 12.4 host / 12.9.1 container
- 컨테이너: `nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb` + Dockerfile.airan = `airan:25-3`
- pyaerial 25.3.2 source를 cmake/ninja로 빌드, `.so` 5개를 `pyaerial/src/aerial/pycuphy/`로 복사
- 측정: NUM_CELLS=20, ITERATIONS=100, n=5 (alone) / n=10 (NeuralRx coloc)

## 디렉터리 구조

```
results/20260614/
├── E0_baseline_3g/         (5 runs, L1 alone on 3g cross-partition layout)
├── E1_neuralrx/            (10 runs, L1 on 3g + NeuralRx on 4g)
├── E2_chanpred/            (5 runs, L1 on 3g + chanpred on 4g)
├── E3_coloc/               (5 runs, L1+chanpred same 3g)
├── E4_misc/                (xapp, sat_compute, sat_hbm — cross-partition fill)
├── E5_alone_partition/{2g,3g,4g,7g}/  (5 runs each, L1 alone per partition)
├── E6_coloc_neuralrx/{2g,3g,4g,7g}/   (10 runs each, L1+NeuralRx same partition)
└── logs/
```

## 다음

- qwen_small HF cache mount 문제 해결 후 짧게 1회 측정 (F-table 마지막 빈 칸)
- Perlmutter MPS / NCU / NSYS / P5 결과 push 받는 대로 통합
- **paper PART F figF1 재구성** — partition sweep alone vs coloc 같이 표시
- **새 figure**: "L1+NeuralRx coloc은 partition 크기와 무관하게 ~360ms" (2g/3g/4g/7g 모두 ~360ms 막대) — 가장 강한 그림
