# 다음 실험 세션 계획 · AI-RAN GPU 격리 · NIC RDMA 검증

> **2026-08-13 연구 방향 갱신**: 이 문서는 완료된 placement/transport 실험 기록으로
> 보존한다. 후속 central experiment는 dynamic MIG reconfiguration을 전제로 하지 않으며,
> [`DRAIN_FREE_NRX_EXPERIMENT_PLAN.md`](DRAIN_FREE_NRX_EXPERIMENT_PLAN.md)의 고정 MIG
> topology · independent NRx endpoint routing · conventional fallback 설계를 따른다.

**작성일**: 2026-08-05 · **업데이트**: 2026-08-13 (direct TensorRT·P2P/GDR·MPS·queue sweep 완료)
**작성 배경**: Chain 17 · 18 · 19 실험 완료 후 · 논문 novelty 완성을 위한 추가 실측 필요
**전제**: 각 Phase의 검증 gate · 통과 못 하면 다음 진행 안 함 · 원인 파악까지 stop

## 2026-08-13 optimized-path 결과 · 현재 authoritative

- `pycuphy` wrapper NRx 104.456ms의 98.4%는 generic layout-conversion kernel이었다.
- caller-owned TensorRT binding은 4g에서 1.413ms, CUDA Graph 적용 후 1.340ms다.
  wrapper와 두 output 모두 `max_abs_difference=0`으로 일치한다.
- 실제 MCS2/QPSK LLR은 2-bit output이며 backward payload는 314,496B다.
- ring depth 2, Qwen 별도 3g 조건:
  - MIG same 4g: L1 slowdown 1.621×, e2e 6.191ms, 322.8 slot/s
  - Cross P2P 2g|2g: L1 slowdown **1.043×**, e2e 6.383ms,
    P2P round-trip 76.84μs, 312.9 slot/s
  - MIG+MPS same 4g: L1 slowdown 1.702×, e2e 6.383ms
- 동일 depth 1의 Cross P2P/NIC GDR: 5.888/6.326ms e2e,
  169.8/158.1 slot/s. NIC loopback이 P2P보다 0.438ms 느렸다.
- NRx N=1 capacity: 2g 334.9, 4g 745.1, full A100 1,130.5 slot/s.
  같은 slice의 replica N=2~16은 capacity를 늘리지 못했다. 1,000 slot/s
  open-loop는 full A100만 안정적이었다.
- 2g/full native-build TensorRT engine sensitivity는 4g-built shared engine보다
  0.8%/2.9% 빨랐지만 stability와 placement 결론을 바꾸지 않았다.
- Full MPS Qwen cap 30/50/70/100%는 e2e 5.865/6.226/6.656/8.569ms와
  Qwen 7.92/11.14/17.24/21.11 it/s의 Pareto trade-off를 만들었다.
- 해석과 raw bundle: `results/20260813_nrx_placement/REPORT_KO.md`

## Legacy wrapper-path 결과 · 2026-08-12 세션

아래 14-row는 구현·transport bring-up 이력으로 보존한다. 약 105ms NRx wrapper
artifact와 잘못된 8-bit LLR payload 가정을 포함하므로 optimized placement의
최종 성능 결론에는 위 direct-TensorRT 결과를 사용한다.

**Task 1 완료 ✅** — real cuPHY↔NRx 통합 파이프라인 · 최종 14-row 결과 완료

| Config | 배치 · 격리 | L1 mean (ms) | p95 | p99 | Qwen it/s |
|---|---|---|---|---|---|
| 5 · cuPHY 단독 | GPU3 무경합 | 39.0 | 39.4 | 39.4 | — |
| 6 · cuPHY+NRx 단독 | GPU3 무경합 | 108.1 | 108.5 | 108.6 | — |
| **1 · MIG only** | 4g(L1+NRx) + 3g(Qwen) | **105.7** | 106.0 | 106.1 | 10.23 |
| 3 · MIG (collapse) | 동일 | 105.9 | 106.1 | 106.5 | 10.23 |
| 2 · MPS 30% Qwen | Full GPU 공유 | 205.3 | 210.0 | 210.1 | 8.05 |
| 2 · MPS 50% | 동일 | 207.5 | 210.3 | 214.3 | 11.29 |
| 2 · MPS 70% | 동일 | 218.2 | 221.1 | 221.3 | 17.38 |
| 2 · MPS 100% | 동일 | **292.0** | **393.2** | **395.6** | 21.26 |
| **4 · Cross-partition shm** | 2g(L1) + 2g(NRx) + 3g(Qwen) | **111.9** | 112.8 | 113.1 | 10.23 |
| 7 · 동일 (collapse) | 동일 | 109.7 | 110.6 | 110.8 | 10.23 |
| **4 · Cross CPU-buffer RDMA** | 2g(L1) + 2g(NRx) + 3g(Qwen) | **111.1** | 111.6 | 111.7 | 10.22 |
| 7 · Cross CPU-buffer RDMA | 동일 반복 | 111.4 | 111.7 | 111.8 | 10.23 |
| **4 · Cross GDR staging** | 2g(L1) + 2g(NRx) + 3g(Qwen) | **109.687** | 109.861 | 110.223 | 10.23 |
| 7 · Cross GDR staging | 동일 반복 | **109.526** | 109.778 | 110.349 | 10.23 |

**핵심 결론**:
- 동일 Qwen throughput (~10 it/s) 조건에서 MIG (106ms) vs MPS (208ms) → **격리로 2× 우수**
- Cross-partition은 109.7~111.9ms로 안정적이지만 4g monolithic 대비 2g+2g split과 IPC 효과가 함께 섞여 있으므로 차이 전체를 순수 IPC 비용으로 해석할 수 없음
- MPS pct 30% (Qwen 최대 제약) 여도 L1 baseline 1.9× 저하 · HBM 경합은 SM cap 으로 못 막음
- CPU-buffer RDMA 반복 평균 111.24ms vs shm 110.78ms로 우열이 run-to-run 변동 범위 안에 있음. Config 4 pair는 RDMA가 0.77ms 개선됐지만 Config 7 pair는 1.69ms 느림
- GDR staging 반복 평균은 **109.607ms**다. 동일 두 반복의 shm 평균 110.776ms보다 **1.169ms**, CPU-buffer RDMA 평균 111.236ms보다 **1.630ms** 낮았다. Config별 GDR-vs-shm 차이는 각각 -2.178ms와 -0.160ms다
- 위 ms 차이는 전체 L1→NRx→L1 파이프라인 실측값이다. 별도 transport microbenchmark의 μs 수치를 whole-pipeline 통신 오버헤드로 주장하지 않는다

**Task 2 Phase 1 완료 ✅** — CPU-buffer RDMA channel 통합 (shm → NIC RDMA loopback)
- `airan:25-3-rdma` 이미지 빌드 완료 · pyverbs 57.0 동작 확인
- `rdma_test.py` 1MiB × 10회 성공 · warm-up 제외 steady-state 평균 **107.35μs**
- QP INIT → RTR `ENODEV` 원인 해결: **컨테이너 `--network=host` 필요** (RoCE GID는 netdev 네임스페이스 스코프)
- rdma_channel.py 에 `_require_gid_netdev()` 체크 · fail-fast 로직 추가됨
- `l1_producer.py` / `nrx_consumer.py` 양방향 RDMA 통합 · 1,415,232B forward + 1,257,984B backward · 20 warm-up + 1 measured slot smoke **107.861ms**
- Config 4/7 RDMA N=30 완료 · 양쪽 NRx 정상 종료 · Qwen 10.22~10.23 it/s · Topology A 복구 확인

**Task 2 Phase 2 GPUDirect RDMA staging 완료 ✅**
- CPU-buffer 경로의 `cp.asnumpy → MR.write` 및 `MR.read → cp.asarray` payload bounce를 GPU MR staging으로 대체했다
- 4g MIG의 64KiB CuPy allocation을 `MR(address=gpu_ptr)`로 등록 성공 · `CUDA_GDR_POINTER_CAPABLE=1`
- `nvidia_peermem` counter가 등록 중 `num_alloc_mrs=1`, `num_reg_bytes=65536`으로 증가하고 정상 해제됨
- IOMMU domain은 `DMA-FQ` translated 상태지만 실제 두-MIG GPU-memory WRITE를 64KiB와 실제 forward/backward payload 크기로 검증했다. seq 1~10 모두 GPU-side first-word/checksum 검증 성공
- transport steady-state(seq 2~10) 평균: 64KiB **61.38μs**, 1,415,232B forward **758.44μs**, 1,257,984B backward **665.29μs**. 측정 범위는 GPU payload WRITE + CPU sequence-marker WRITE이며 whole-pipeline overhead가 아니다
- Config 4/7 GDR staging N=30 완료: mean/p95/p99 **109.687/109.861/110.223ms**, **109.526/109.778/110.349ms** · Qwen 각 10.23 it/s
- payload는 host memory를 경유하지 않지만 public `TrtEngine` wrapper의 내부 F-order input copy와 LLR output→registered GPU staging copy가 남는다. 따라서 **end-to-end zero-copy로 부르지 않는다**

**Fair direct-P2P overlap 최종 검증 완료 ✅ (2026-08-13)**
- 교정 실험은 caller-owned TensorRT, correct 2-bit LLR, CUDA Graph, warm-up 100,
  N=1,000, 각 topology 3 trials로 다시 수행했다.
- depth 2 same 4g는 L1 active 2.975ms/slowdown 1.621×, e2e 6.191ms다.
  Cross 2g|2g P2P는 L1 active 2.533ms/slowdown **1.043×**, e2e 6.383ms다.
- 즉 P2P는 L1 isolation을 거의 복구하지만 2g NRx service 3.114ms가 4g의
  1.782ms보다 느려 end-to-end 이득을 상쇄한다. 통신 76.84μs가 주 병목이 아니다.
- depth 1 보정에서 Cross P2P 5.888ms/169.8 slot/s, NIC GDR
  6.326ms/158.1 slot/s다. GDR은 P2P보다 0.438ms 느렸으며 5~10μs 예상은
  관측되지 않았다.
- 결과: `results/20260813_nrx_placement/raw/p2p_direct_trt*`

## 진행 로그

**2026-08-12 · Phase 0 최종 상태** (CloudLab d8545 · node0.sgkim-312839)

- [x] 0.1 노드 예약 · SSH · sudo · Ubuntu 22.04.2
- [x] 0.2 NVIDIA driver 580.173.02 · CUDA 13.0 · 4× A100 인식
- [x] 0.3 Docker CE 29.7.2 · nvidia-container-toolkit · /mydata/docker data-root
- [x] 0.4 cuPHY · pyaerial · Aerial 이미지 pull · airan:25-3-final 빌드 · pyaerial C++ 바인딩 build 완료 · import OK
- [x] 0.5 MIG mode enabled GPU 0 · Config A partitions (4g + 3g) 생성 완료
  - MIG L1 UUID: `MIG-dae3f173-7b15-594b-bc80-6cef80687a56`
  - MIG AI UUID: `MIG-80a4659b-f06f-540b-9f4b-1c91f78aaaf3`
- [x] 0.6 MPS daemon L1 partition 시작 정상 (pid 5806)
- [x] 0.7 MOFED 24.10-3.2.5.0 설치 완료 · openibd load
- [x] 0.8 nvidia_peermem · mlx5_ib · ib_uverbs 커널 로드 · 재부팅 후 auto-load (persistent · `/etc/modules-load.d/nvidia_peermem.conf`)
- [x] 0.9 **Sanity 재현 성공** · `real_l1.py sanity_baseline 20 30`
  - **mean 37.4 ms · p95 37.7 ms · p99 37.9 ms · miss1ms 30/30**
  - 이전 baseline (38.5 ms) 과 완전 일치 · 환경 동등성 검증
  - 저장: `/mydata/results/20260812/sanity/realL1_sanity_baseline_20260812_080106.json`
- [x] 0.10 /mydata (1.5T NVMe · 54GB used / 1.4T avail) · `/mydata/results/20260812/`

**Task 1 시작 준비 · 완료** ✅
- 모든 하드웨어 · 소프트웨어 · MIG · MPS · sanity 재현 · gate 통과

**Task 2 (NIC RDMA) 시작 조건 · 해결** ✅ (2026-08-12)
- **PHY internal loopback**으로 CloudLab LAN 없이 link UP 성공
  ```bash
  sudo ip link set enp161s0np0 down
  sudo mlxlink -d /dev/mst/mt4123_pciconf0 --link_mode_force --speeds 100G_4X --yes
  sudo mlxlink -d /dev/mst/mt4123_pciconf0 -l PH --yes
  sudo ip link set enp161s0np0 up
  ```
- `ibstat mlx5_0` · State Active · Rate 100 · Link layer Ethernet
- `ib_write_lat -d mlx5_0 localhost` 2-byte 성공 · **t_avg=1.34 μs · p99=1.41 μs**
- 리부트 후 복구 스크립트: `/mydata/nic_loopback_restore.sh` (수동 실행 필요)
- GPUDirect RDMA는 perftest CUDA 빌드 대신 CuPy GPU MR 기반 `gdr_mr_probe.py` / `gdr_rdma_test.py`로 검증 완료
- Soft-RoCE 경로는 MOFED 심볼 conflict로 사용 불가 · PHY loopback으로 대체함

**Fixes 필요했던 것들 (다음 세션 참고)**:
- Docker.io 29.x containerd metadata 이슈 → docker-ce로 교체
- Aerial repo에서 HDF5 파일이 Git LFS pointer → `git lfs install; git lfs pull` 필요
- pyaerial API 변경 · `CudaStream` → `get_cuda_stream` (real_l1.py 수정 후 로컬 리포에도 push)

---

---

## 배경 · 현재까지 완료된 것

**이미 완료 (Chain 17/18/19 · 2026-07 ~ 2026-08-03)**:
- MIG · MPS · Full GPU 조합 grid 실측 (108 + 273 조건)
- SP (같은 파티션 co-location) · CP (다른 파티션 격리) 배치별 L1 latency
- MPS pct 튜닝 sweep (30/50/70/100)
- Diverse AI (Qwen · Whisper · BERT · NRx · CsiNet · BeamPred) 스택 검증
- MIG + MPS 결합의 유효성 확인 (CP setup에서 L1 baseline 유지)

**당시 핵심 gap · 2026-08-12 해소 상태**:
1. NRx dummy input gap → 실제 cuPHY tensor와 NRx LLR을 교환하는 파이프라인으로 해소
2. CPU bounce-only gap → CPU-buffer RDMA와 두-MIG GDR staging을 모두 구현·실측
3. 남은 최적화 → public TRT wrapper 내부 GPU 복사를 제거하는 caller-owned binding 경로

**논문 novelty 완성 target**:
- 진짜 modular AI-RAN pipeline 구성 (L1 ↔ NRx 실제 데이터 교환)
- NIC RDMA loopback 실측 · CPU bounce 대비 latency 개선 정량화
- "MIG 격리 유지 + tightly-coupled AI-RAN 통신 최적화" 라는 완결된 architectural claim

---

## Phase 0 · 전체 환경 셋업 (MUST COMPLETE BEFORE Task 1 · Task 2)

### 원칙
**Task 1 · Task 2 어느 것도 · Phase 0의 모든 체크박스가 통과되기 전에는 시작 안 함**. 셋업 도중 실패 · 원인 파악까지 stop · 무리하게 다음 단계 안 함. 이 gate가 가장 중요.

### 0.1 · CloudLab 노드 예약 · 초기 확보

- [x] **d8545 노드 예약 확정** (Wisconsin cluster · AIRANSLICING project)
- [x] SSH 접근 확인 · passwordless sudo 확인
- [x] 노드 상태와 잔여 GPU process/container 확인 · 실험 전 clean 상태 확보

### 0.2 · Base OS · 드라이버 · CUDA 스택

- [x] **Ubuntu 22.04** 확인
- [x] **NVIDIA Driver 580.173.02 · 정상 동작**
  ```bash
  nvidia-smi
  cat /proc/driver/nvidia/version
  ```
- [x] **Host CUDA 13.0 · RDMA container CUDA 12.9.1 확인**
  ```bash
  nvcc --version
  ```
- [x] **GPU 4× A100-SXM4-40GB 전체 정상 인식**

### 0.3 · Docker · Container 이미지

- [x] **Docker CE 29.7.2 · nvidia-container-toolkit 설치 · 정상 동작**
  ```bash
  docker --version
  docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
  ```
- [x] **필수 이미지 준비 완료**
  - `airan:25-3-final` · cuPHY + Qwen workload
  - `airan:25-3-rdma` · cuPHY + pyverbs 57.0
- [x] Docker data-root `/mydata/docker` · NVMe 여유 공간 확인

### 0.4 · cuPHY / pyaerial SDK

- [x] **Aerial 25.3.2 · cuPHY/pyaerial 설치 · 정상 동작**
  ```bash
  # 컨테이너 안에서
  python3 -c "import pyaerial; print(pyaerial.__version__)"
  ```
- [x] **`neural_rx.onnx` 모델 파일 존재 확인**
  ```bash
  ls /opt/nvidia/cuBB/pyaerial/models/neural_rx.onnx
  ```
- [x] **`real_l1.py` baseline latency 재현** · sanity mean 37.4ms, config5 mean 38.958ms

### 0.5 · MIG 설정 · Config A/C 검증

- [x] **MIG mode 활성화 · 재시작 후 유지**
  ```bash
  sudo nvidia-smi -mig 1
  ```
- [x] **Topology A (4g + 3g) 파티션 생성 · 정상 동작**
  ```bash
  sudo nvidia-smi mig -cgi 4g.20gb,3g.20gb -C
  nvidia-smi -L | grep MIG
  ```
- [x] **Topology B (3g + 2g + 2g) 파티션 생성 · 정상 동작**
- [x] **각 MIG partition에서 CUDA context/allocation · workload 정상 동작 검증**
- [x] **파티션 UUID 기록** · runner에서 자동 탐색하고 실험 후 Topology A 복원

### 0.6 · MPS 데몬 · pct 설정 검증

- [x] **`nvidia-cuda-mps-control` daemon 정상 시작**
  ```bash
  echo start_server -uid 0 | nvidia-cuda-mps-control
  ```
- [x] **`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` 30/50/70/100 적용 확인**
- [x] **MIG/MPS 조합 실행 확인**
- [x] **MPS on/off 및 Full-GPU workload sweep 완료**

### 0.7 · MOFED · RDMA 스택 (Task 2 필수)

- [x] **MOFED 24.10-3.2.5.0 설치 · 버전 확인**
  ```bash
  ofed_info -s
  # 없으면 · 공식 사이트에서 Ubuntu 22.04용 MOFED download · install
  ```
- [x] **NIC 하드웨어 인식 · ConnectX-6 Dx 확인**
  ```bash
  lspci | grep -i mellanox
  sudo mst status
  ibstat
  ibv_devinfo
  ```
- [x] **NIC port state ACTIVE · physical LinkUp 확인**
  ```bash
  ibstatus
  # State: 4: ACTIVE 여야 함
  ```
- [x] **RoCE v2 GID 3 · PHY internal loopback 활성 설정**
  ```bash
  sudo mlxconfig -d <PCI> query | grep -E "ROCE|LINK"
  sudo mlxconfig -d <PCI> set ROCE_CONTROL=1
  # 필요 시 firmware reset
  sudo mlxfwreset -d <PCI> reset
  ```
- [x] **rdma-core · MOFED libibverbs · perftest · pyverbs 57.0 정상 동작**
  ```bash
  apt list --installed 2>/dev/null | grep -E "rdma|ibverbs|perftest"
  ib_write_lat --version
  ```

### 0.8 · nvidia-peermem · GPUDirect 스택

- [x] **`nvidia-peermem` 커널 모듈 로드 · GPU MR counter로 정상 등록 확인**
  ```bash
  sudo modprobe nvidia_peermem
  lsmod | grep nvidia_peermem
  dmesg | grep -i peermem
  ```
- [x] **`mlx5_ib` · `ib_uverbs` 관련 모듈 로드 확인**
- [x] **IOMMU domain 확인** · `DMA-FQ` translated 경고 상태지만 실제 GPU MR/WRITE gate 통과
- [x] **GPU memory peer registration과 두-MIG RDMA correctness로 실제 경로 확인**

### 0.9 · 기본 sanity 실험 · 이전 결과 재현

- [x] **cuPHY sanity baseline 재현으로 환경 동등성 확인**
  - mean 37.4ms · p95 37.7ms · p99 37.9ms
- [x] **Task 1 통합 sweep의 standalone baselines 확인**
  - config5 cuPHY-only 38.958ms · config6 cuPHY+NRx 108.128ms

### 0.10 · Storage · results 디렉토리 준비

- [x] `/mydata/results/chain/` 생성 · 결과 저장 권한 확인
- [x] Analysis와 node runner canonical copy 준비
- [x] 이전 세션 baseline과 최종 raw 결과 로컬 다운로드 완료

### Phase 0 · 최종 Gate

**아래 모두 통과되어야 Task 1 · Task 2 시작 가능**:

- [x] 0.1 ~ 0.10 · 실제 환경 gate 완료
- [x] Sanity baseline이 이전 데이터와 일치
- [x] CPU memory `ib_write_lat`와 pyverbs 1MiB WRITE 성공
- [x] `nvidia-peermem` · GPU MR · 실제 두-MIG WRITE 검증
- [x] 실험 로그 저장 위치 · 권한 확인
- [x] 결과와 재현 스크립트의 scoped audit 완료 · unrelated local artifacts는 보존

**Phase 0 통과 못하면**:
- 원인 항목 파악 · 문서화
- 필요 시 CloudLab 이미지 재설치 · MOFED 재설치
- 무리하게 Task 진행 하지 말 것
- Setup 실패 자체가 결과이므로 · plan에 기록 · 다음 시도 시 활용

---

## Task 1 · NRx-L1 실제 통신 파이프라인 구성

**전제**: Phase 0 완전 통과 확인.

### 목적
현재 dummy input NRx를 · 실제 L1 output을 입력받아 처리하고 · 결과를 L1로 반환하는 **end-to-end 파이프라인**으로 재구성.

### Phase 1.1 · Pre-requisites 조사 (실험 전 · 완료 필수)

- [x] **cuPHY L1 receiver 파이프라인과 NRx replacement point 확인**
  - `real_l1.py` 및 `pyaerial.phy5g` 코드 분석
  - 어느 stage에서 rx_slot 생성되는지 · 어느 stage가 LLR 소비하는지 정확히 파악
  - PUSCH RX pipeline: FFT → Channel Est → Equalization → **NRx replacement point** → LDPC

- [x] **NRx 입출력 tensor shape · dtype · C-order serialization 확정**
  - Input: `rx_slot_real/imag (3276, 12, 4)` · `h_hat (4914, 1, 4)` · DMRS info
  - Pipeline output: LLR `(8, 1, 3276, 12)` float32 → LDPC de-rate match/decoder
  - Forward 1,415,232B · backward 1,257,984B payload contract를 양쪽 assert로 고정

- [x] **Modular · 별도 L1/NRx container 방식 채택**
  - **Option A · Monolithic 통합**: L1 process 내부에서 TRT engine 로드 · 함수 호출로 NRx 실행. IPC 불필요. 개발 복잡도 낮음. 하지만 modular 시나리오 검증 못 함.
  - **Option B · Modular · IPC 기반** ← **채택**
    - L1과 NRx 별개 컨테이너 · shared memory · Unix socket · RDMA 등으로 데이터 교환
    - 개발 복잡도 높음
    - 통신 오버헤드 실측 가능
    - 논문 novelty 방향에 맞음

### Phase 1.2 · 파이프라인 구현

- [x] **L1 producer 구현** · `l1_producer.py`와 `l1_producer_gdr.py`
  - cuPHY ChannelEstimator의 rx_slot/h_hat을 선택한 transport에 publish
  - NRx 처리 완료 marker 대기 · LLR payload consume
  - LDPC 단계로 넘김

- [x] **NRx consumer 구현** · `nrx_consumer.py`와 `nrx_consumer_gdr.py`
  - SHM/CPU MR/GPU MR에서 rx_slot과 h_hat consume
  - `trt_engine.run()` 실행
  - 결과 LLR payload WRITE 후 ordered u64 sequence marker publish

- [x] **IPC/transport 3종 구현·비교**
  - **SHM baseline**: POSIX shared memory · CPU bounce
  - **CPU-buffer RDMA**: registered host MR · RoCE v2 loopback
  - **GDR staging**: registered persistent CuPy allocation · CPU payload bounce 제거

### Phase 1.3 · 동작 검증 (Gate · 통과 못 하면 다음 진행 안 함)

- [x] **Pipeline execution · tensor size/order · standalone transport checksum 검증**
  - L1 → NRx → L1 파이프라인이 end-to-end 성공 실행되는지
  - NRx output이 zero 또는 NaN 아닌 · 정상 값인지
  - 제한: full-run harness는 per-slot CRC/BER assertion을 기록하지 않으므로 BER claim은 후속 gate로 남김

- [x] **20 warm-up + N=30 latency와 concurrent Qwen throughput 측정**
  - 각 end-to-end config의 mean/p95/p99와 raw 30 slots 저장
  - transport microbenchmark는 pipeline latency와 분리해 기록
  - Qwen progress snapshot으로 concurrent throughput 기록

### Phase 1.4 · 실측 실험 (Task 1 완료 조건)

- [x] **Same-partition MIG와 Full-GPU MPS sweep 완료**
  - MIG 4g(L1+NRx)+3g(Qwen) 독립 반복: 105.725/105.859ms mean
  - Full-GPU MPS Qwen pct 30/50/70/100: 205.327~291.961ms mean

- [x] **Cross-partition SHM · CPU-RDMA · GDR staging 반복 측정 완료**
  - 2g(L1)+2g(NRx)+3g(Qwen), 각 transport config4/7 독립 반복
  - 평균: SHM 110.776ms · CPU-buffer RDMA 111.236ms · GDR staging 109.607ms

### Task 1 latency-study 성공 기준 · 달성
- Real cuPHY CE → NRx TRT → LDPC/CRC pipeline 실행과 tensor transport gate 완료
- SHM · CPU-buffer RDMA · GDR staging의 end-to-end latency 정량화
- MIG same-partition · cross-partition · Full-GPU MPS 조건 실측 완료
- 후속 correctness 항목: full-run per-slot CRC/BER assertion 추가

---

## Task 2 · NIC-based GPUDirect RDMA · MIG 간 통신 구현

**전제**: Phase 0 완전 통과 확인.

### 목적
CPU payload bounce를 우회하는 MIG 파티션 간 GPU-memory RDMA 경로를 검증한다.
2026-08-12에는 persistent GPU staging MR 경로까지 완료했다. public TRT wrapper
내부 GPU 복사가 남아 있으므로 엄밀한 end-to-end zero-copy 단계와 구분한다.

### 하드웨어 요건 (확인 완료)
- **CloudLab d8545** · NVIDIA HGX A100 (4× 40GB SXM4)
- **NIC**: Dual-port Mellanox ConnectX-6 DX 100Gb (200Gb 포트 하나 활용)
- PCIe 4.0 · EPYC 7413 · 512GB RAM

### Phase 2.1 · Pre-requisites 확인 (실험 전 · 완료 필수)

- [x] **CloudLab d8545 노드 예약** · sudo 권한 확보

- [x] **하드웨어 검증**
  ```bash
  # NIC 확인 (ConnectX-6 DX 예상)
  lspci | grep -i mellanox
  sudo mst status
  sudo mlxfwmanager

  # GPU · MIG 확인
  nvidia-smi
  nvidia-smi mig -lgi
  ```

- [x] **소프트웨어 스택 설치 · 로드**
  ```bash
  # MOFED (Mellanox OFED)
  ofed_info -s

  # 커널 모듈
  sudo modprobe nvidia_peermem
  sudo modprobe mlx5_ib
  lsmod | grep -E "nvidia_peermem|mlx5_ib"

  # NIC firmware 설정 · RoCE loopback 활성화
  sudo mlxconfig -d <PCI> query | grep ROCE
  sudo mlxconfig -d <PCI> set ROCE_CONTROL=1
  ```

- [x] **NIC 상태 확인**
  ```bash
  ibstat
  ibv_devinfo
  # Port state = ACTIVE · link layer = Ethernet or InfiniBand 확인
  ```

### Phase 2.2 · 기본 RDMA loopback 검증 (Gate · 통과 못 하면 진행 안 함)

- [x] **CPU memory 기반 RDMA loopback 벤치마크**
  ```bash
  # Server (한 터미널)
  ib_write_lat -d mlx5_0

  # Client (다른 터미널 · 같은 노드)
  ib_write_lat -d mlx5_0 localhost
  ```
  - **실측**: 1MiB WRITE 10회 성공 · cold 5995.51μs · seq 2~10 평균 107.35μs

- [x] **GPU memory · GPUDirect RDMA loopback 벤치마크**
  ```bash
  # 실제 검증 경로: MIG별 컨테이너에서 같은 tag/payload로 cons와 prod 실행
  python3 gdr_mr_probe.py 65536
  python3 gdr_rdma_test.py cons 65536
  python3 gdr_rdma_test.py prod 65536
  # 실제 payload 크기 1415232, 1257984도 각각 반복
  ```
  - **실측**: 두 MIG partition 간 GPU-memory WRITE와 first-word/checksum 검증 완료
  - steady-state 평균: 64KiB 61.38μs · forward 1,415,232B 758.44μs · backward 1,257,984B 665.29μs
  - 이 값은 payload WRITE + CPU marker WRITE 구간이며 왕복 또는 whole-pipeline overhead가 아님

### Phase 2.3 · RDMA-based L1↔NRx 파이프라인 구현

- [x] **Task 1의 IPC 메커니즘을 RDMA staging으로 교체**
  - L1 process · persistent CuPy forward/backward staging buffer를 GPU MR로 등록
  - NRx process · 동일 payload contract의 GPU MR staging buffer를 등록
  - 두 process가 RDMA connection (loopback IP) 통해 · Queue Pair 연결
  - L1 → NRx: GPU payload RDMA WRITE 후 ordered CPU u64 marker WRITE
  - NRx → L1: GPU LLR staging payload RDMA WRITE 후 marker WRITE

- [x] **staging 경로 동작 검증** (Gate)
  - 두 MIG 간 payload가 host memory를 통과하지 않고 N=30 파이프라인 완료
  - GDR microbenchmark는 seq 1~10 first-word/checksum 검증 완료
  - 단, public `TrtEngine`의 F-order input copy와 output→registered staging GPU copy가 남아 end-to-end zero-copy 검증은 아직 아님

### Phase 2.4 · 성능 실측 · Task 1과 비교

- [x] **동일 조건에서 · transport 방식만 다른 실험**
  - **Config X · CP + CPU bounce IPC**: Task 1 결과 (baseline)
  - **Config Y · CP + CPU-buffer NIC RDMA loopback**
  - **Config Z · CP + GDR staging NIC RDMA loopback**

- [x] **측정 지표**
  - L1 p99 latency
  - transport microbenchmark · 전체 파이프라인과 별도 분석
  - NRx aggregate throughput
  - Qwen throughput

- [x] **Observed results**

  | 조건 | Config 4 mean/p99 | Config 7 mean/p99 | 반복 mean 평균 |
  | --- | --- | --- | --- |
  | CP + shm | 111.865/113.119ms | 109.686/110.827ms | 110.776ms |
  | CP + CPU-buffer RDMA | 111.097/111.664ms | 111.375/111.802ms | 111.236ms |
  | CP + GDR staging | 109.687/110.223ms | 109.526/110.349ms | **109.607ms** |

  GDR staging 평균은 shm보다 1.169ms, CPU-buffer RDMA보다 1.630ms 낮다.
  이 비교에는 전체 pipeline과 run-to-run 변동이 포함되며, microbenchmark
  latency를 그대로 pipeline overhead로 대입하지 않는다.

### Task 2 staging 성공 기준 · 달성
- GPUDirect RDMA loopback · MIG 파티션 간 payload별 실측 완료
- L1↔NRx GPU-MR staging 통신 functional 확인 · Config 4/7 N=30 완료
- shm · CPU-buffer RDMA · GDR staging의 반복 평균 비교 완료
- public TRT 내부 복사를 제거한 end-to-end zero-copy와 별도 profiler 검증도
  2026-08-13 direct-TensorRT follow-up에서 완료했다.

### Phase 2.5 · direct TensorRT zero-copy follow-up · 완료

- caller-owned forward input과 `output_1`을 GDR-registered GPU MR에 직접 binding
- CUDA Graph capture/replay 적용, host payload bounce와 wrapper layout copy 제거
- wrapper 대비 output bit-for-bit 일치 검증
- Cross NIC GDR depth 1: mean/p99 6.326/6.846ms, Qwen 10.24 it/s
- 동일 depth Cross P2P: mean/p99 5.888/6.224ms, Qwen 10.22 it/s
- 상세 결과: `results/20260813_nrx_placement/`

---

## 실험 순서 · 의존성

```
┌──────────────────────────────────────────────────┐
│  Phase 0 · 전체 환경 셋업 (Gate · MUST PASS)      │
│  0.1 노드 · 0.2 OS/CUDA · 0.3 Docker · 0.4 cuPHY │
│  0.5 MIG · 0.6 MPS · 0.7 MOFED · 0.8 peermem     │
│  0.9 sanity 재현 · 0.10 storage                   │
└─────────────────┬────────────────────────────────┘
                  ↓ (all boxes checked)
     ┌────────────┴────────────┐
     ↓                          ↓
Phase 1.1 (조사)          Phase 2.1 (환경 확인 · 이미 0.7에서 커버)
     ↓                          ↓
Phase 1.2 (파이프라인)     Phase 2.2 (RDMA 기본 검증)
     ↓                          ↓
Phase 1.3 (검증 gate) ─────────────┐
     ↓                              ↓
Phase 1.4 (SP·CP · CPU bounce)     ↓
                                    ↓
                            Phase 2.3 (RDMA 파이프라인)
                                    ↓
                            Phase 2.4 (비교 · 최종 결과)
```

**Phase 0 이 최우선 · 완전 통과 후에만 Task 1 · Task 2 시작**.
Task 1 · Task 2 상당 부분 병렬 진행 가능. 최종 Phase 2.4에서 통합.

---

## 실험 성공 시 논문 최종 결과

**두 이야기가 완결**:

1. **"MIG isolation is necessary but insufficient for AI-RAN"** (완료)
   - Chain 17 · 18 · 19 실측 완료
   - SP 실패 · CP 성공 (통신 없는 경우)

2. **"Real modular AI-RAN needs both isolation AND fast communication"** (Task 1 + Task 2 staging 완료)
   - Real L1↔NRx 통신 실측
   - MIG 격리 위에서 shm · CPU-buffer RDMA · GDR staging을 직접 비교
   - end-to-end zero-copy 주장은 public TRT 내부 copy 제거 후로 제한
   - **논문 novelty의 핵심 결과와 구현 경계가 함께 확인됨**

---

## 리스크 · 대응

| 리스크 | 대응 |
| --- | --- |
| Task 1.1 · cuPHY pipeline 분석 어려움 | NVIDIA 문서 · pyaerial 소스 · 필요 시 NVIDIA developer forum 질문 |
| Task 1.3 · L1 · NRx integration 안 됨 | 원인 파악까지 · 실험 중단 · 무리하게 다음 진행 안 함 (사용자 지시 준수) |
| Task 2.1 · MOFED · nvidia-peermem 미설치 · 설치 실패 | CloudLab 다른 이미지 시도 · NVIDIA 공식 설치 가이드 · 최악은 재설치 |
| Task 2.2 · RDMA loopback 자체 안 됨 | mlxconfig · RoCE 설정 · 매뉴얼 검토 · Mellanox forum |
| Task 2.3 · MIG partition 간 RDMA 안 됨 | nvidia-peermem 로그 확인 · IOMMU 설정 · driver 버전 |

---

## 산출물

- [x] **코드**: `l1_producer.py` / `nrx_consumer.py` · `l1_producer_gdr.py` / `nrx_consumer_gdr.py` · RDMA channel/runner
- [x] **실측 데이터**: 최종 14-row `SUMMARY.txt` · transport raw logs/`TRANSPORT_SUMMARY.csv`
- [x] **비교 figure**: shm · CPU-buffer RDMA · GDR staging 성능 대비
- [ ] **다음 발표 슬라이드**: 지금 발표의 "Next Plan" 슬라이드가 · 실측 데이터로 채워짐
- [ ] **논문 draft 섹션**: "GPU-Direct RDMA Loopback for Modular AI-RAN" 파트

---

## 우선순위 (주별)

- **1주차 · Phase 0 · 전체 환경 셋업 완료 (Gate)**
  - 0.1-0.10 · sanity 재현까지 · 모든 체크박스 통과
  - 통과 못 하면 여기서 stop · 원인 파악 · 재시도
  - **Task 1 · Task 2 시작 안 함**
- **2주차** · Phase 1.1 · 1.2 · 2.2 (조사 · L1↔NRx 파이프라인 시작 · RDMA 기본 검증 병렬)
- **3주차** · Phase 1.3 · 1.4 (파이프라인 검증 gate · SP·CP CPU bounce 실측)
- **4주차** · Phase 2.3 · 2.4 (RDMA 파이프라인 · 최종 비교 · 결과 정리)

**Gate 규칙 준수** (사용자 지시):
- 각 Phase의 검증 단계 · 통과 못 하면 · 다음 phase 진행 안 함
- Phase 0 실패 시 · Task 1 · Task 2 아예 시작 안 함
- 원인 파악까지 stop · 무리하게 다음 진행 안 함

---

## 참고 · 관련 자료

**우리 이전 데이터셋**:
- Chain 17: `results/20260725/chain17_all_stats.json`
- Chain 18: `results/20260725/chain18/`, `chain18_p*` folders
- Chain 19: `results/20260803/chain19_exp*/`, `chain19_gapstats/`

**분석 스크립트**:
- `results/20260803/analysis_chain19/gen_*.py` 시리즈
- 이전 세션 · 완결된 슬라이드 · figure 다수 생성

**외부 리소스**:
- `NVlabs/neural_rx` GitHub 리포 (NRx 원본 논문 · reference implementation)
- NVIDIA cuPHY / pyaerial documentation
- NVIDIA GPUDirect RDMA 지원 매트릭스
- Mellanox ConnectX-6 DX 매뉴얼 · MOFED 문서

**CloudLab 노드**:
- d8545 (Wisconsin) · A100 SXM4 4× + ConnectX-6 DX
- 예약 시 · sudo 권한 자동 부여 · setup 후 · 실험은 sudo 없이 가능
