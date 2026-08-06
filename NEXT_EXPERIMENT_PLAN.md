# 다음 실험 세션 계획 · AI-RAN GPU 격리 · NIC RDMA 검증

**작성일**: 2026-08-05
**작성 배경**: Chain 17 · 18 · 19 실험 완료 후 · 논문 novelty 완성을 위한 추가 실측 필요
**전제**: 각 Phase의 검증 gate · 통과 못 하면 다음 진행 안 함 · 원인 파악까지 stop

---

## 배경 · 현재까지 완료된 것

**이미 완료 (Chain 17/18/19 · 2026-07 ~ 2026-08-03)**:
- MIG · MPS · Full GPU 조합 grid 실측 (108 + 273 조건)
- SP (같은 파티션 co-location) · CP (다른 파티션 격리) 배치별 L1 latency
- MPS pct 튜닝 sweep (30/50/70/100)
- Diverse AI (Qwen · Whisper · BERT · NRx · CsiNet · BeamPred) 스택 검증
- MIG + MPS 결합의 유효성 확인 (CP setup에서 L1 baseline 유지)

**핵심 gap · 다음 세션에서 해결**:
1. NRx가 dummy input으로 · 실제 L1 데이터 사용 안 함 → **통신 오버헤드 미측정**
2. MIG 파티션 간 통신 = CPU bounce만 있음 → **NIC RDMA 우회 경로 검증 안 됨**

**논문 novelty 완성 target**:
- 진짜 modular AI-RAN pipeline 구성 (L1 ↔ NRx 실제 데이터 교환)
- NIC RDMA loopback 실측 · CPU bounce 대비 latency 개선 정량화
- "MIG 격리 유지 + tightly-coupled AI-RAN 통신 최적화" 라는 완결된 architectural claim

---

## Task 1 · NRx-L1 실제 통신 파이프라인 구성

### 목적
현재 dummy input NRx를 · 실제 L1 output을 입력받아 처리하고 · 결과를 L1로 반환하는 **end-to-end 파이프라인**으로 재구성.

### Phase 1.1 · Pre-requisites 조사 (실험 전 · 완료 필수)

- [ ] **cuPHY L1 receiver 파이프라인 상세 이해**
  - `real_l1.py` 및 `pyaerial.phy5g` 코드 분석
  - 어느 stage에서 rx_slot 생성되는지 · 어느 stage가 LLR 소비하는지 정확히 파악
  - PUSCH RX pipeline: FFT → Channel Est → Equalization → **NRx replacement point** → LDPC

- [ ] **NRx 입출력 tensor 정확한 shape · 의미 확인**
  - Input: `rx_slot_real/imag (3276, 12, 4)` · `h_hat (4914, 1, 4)` · DMRS info
  - Output: `output_1 (8, 1, 3276, 12)` · `output_2 (1, 3276, 12, 8)` → 각각 무엇을 의미? (LLR? soft symbol?)
  - `NVlabs/neural_rx` 원본 논문 · 리포 검토

- [ ] **접근 방식 결정 · 팀 논의 필요**
  - **Option A · Monolithic 통합**: L1 process 내부에서 TRT engine 로드 · 함수 호출로 NRx 실행. IPC 불필요. 개발 복잡도 낮음. 하지만 modular 시나리오 검증 못 함.
  - **Option B · Modular · IPC 기반** ← **채택**
    - L1과 NRx 별개 컨테이너 · shared memory · Unix socket · RDMA 등으로 데이터 교환
    - 개발 복잡도 높음
    - 통신 오버헤드 실측 가능
    - 논문 novelty 방향에 맞음

### Phase 1.2 · 파이프라인 구현

- [ ] **L1 process 수정** · `real_l1.py`
  - Receiver pipeline에서 Equalization 이후 · NRx가 처리할 rx_slot 데이터를 shared memory에 write
  - NRx 처리 완료 대기 · 결과 (LLR) 를 shared memory에서 read
  - LDPC 단계로 넘김

- [ ] **NRx process 수정** · `run_neural_rx_stress.py` 개선
  - Shared memory 에서 rx_slot 읽기
  - `trt_engine.run()` 실행
  - 결과 (LLR) 를 shared memory 에 write
  - L1에 완료 signal

- [ ] **IPC 메커니즘 선택**
  - **1차 · CPU bounce baseline**: POSIX shared memory (`mmap`) + semaphore signal → 가장 단순 · CPU 통과
  - **2차 · Task 2에서 대체**: NIC RDMA로 대체 · 성능 비교

### Phase 1.3 · 동작 검증 (Gate · 통과 못 하면 다음 진행 안 함)

- [ ] **Functional 검증**
  - L1 → NRx → L1 파이프라인이 end-to-end 성공 실행되는지
  - NRx output이 zero 또는 NaN 아닌 · 정상 값인지
  - L1 최종 decoded bits 가 baseline (traditional receiver) 과 comparable 한지 (BER 비교)

- [ ] **성능 baseline 측정**
  - NRx 통합 후 · L1 iteration latency 실측
  - CPU bounce IPC 사용 시 · 통신 왕복 시간 · 몇 μs 인지 정확히
  - N=1, 2, 4 개 NRx service 동시 실행 시 latency 스케일링

### Phase 1.4 · 실측 실험 (Task 1 완료 조건)

- [ ] **SP + real L1↔NRx 통신 · L1 p99**
  - N=1/2/4/6/8 NRx service · 같은 파티션에 co-locate
  - MPS on · pct 30/50/70/100
  - **결과 예상**: 현재 Chain 19 Exp 11 (dummy NRx) 보다 latency 더 증가 (실제 통신 오버헤드 추가)

- [ ] **CP + real L1↔NRx 통신 · L1 p99**
  - N NRx service · 다른 파티션 (3g)
  - **결과 예상**: 격리 이득 있지만 · CPU bounce 통신 오버헤드로 baseline 근처 지연 (~100 μs 추가)
  - Task 2 NIC 실험과 비교할 baseline

### Task 1 성공 기준
- End-to-end pipeline 정상 동작 확인 (BER · latency 둘 다)
- 통신 오버헤드 정량화 (CPU bounce baseline)
- SP · CP 조건별 L1 latency 실측 완료

---

## Task 2 · NIC-based GPUDirect RDMA · MIG 간 통신 구현

### 목적
CPU bounce 우회하여 MIG 파티션 간 **zero-CPU-copy 통신** 실현. 예상 5-40 μs latency로 5G TTI 예산 만족.

### 하드웨어 요건 (확인 완료)
- **CloudLab d8545** · NVIDIA HGX A100 (4× 40GB SXM4)
- **NIC**: Dual-port Mellanox ConnectX-6 DX 100Gb (200Gb 포트 하나 활용)
- PCIe 4.0 · EPYC 7413 · 512GB RAM

### Phase 2.1 · Pre-requisites 확인 (실험 전 · 완료 필수)

- [ ] **CloudLab d8545 노드 예약** · sudo 권한 확보

- [ ] **하드웨어 검증**
  ```bash
  # NIC 확인 (ConnectX-6 DX 예상)
  lspci | grep -i mellanox
  sudo mst status
  sudo mlxfwmanager

  # GPU · MIG 확인
  nvidia-smi
  nvidia-smi mig -lgi
  ```

- [ ] **소프트웨어 스택 설치 · 로드**
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

- [ ] **NIC 상태 확인**
  ```bash
  ibstat
  ibv_devinfo
  # Port state = ACTIVE · link layer = Ethernet or InfiniBand 확인
  ```

### Phase 2.2 · 기본 RDMA loopback 검증 (Gate · 통과 못 하면 진행 안 함)

- [ ] **CPU memory 기반 RDMA loopback 벤치마크**
  ```bash
  # Server (한 터미널)
  ib_write_lat -d mlx5_0

  # Client (다른 터미널 · 같은 노드)
  ib_write_lat -d mlx5_0 localhost
  ```
  - **성공 기준**: latency 측정 완료 · 소용량 (KB) 왕복 5-10 μs 이내

- [ ] **GPU memory · GPUDirect RDMA loopback 벤치마크**
  ```bash
  # perftest에 --use_cuda 옵션 · MIG 파티션 별로 실행
  CUDA_VISIBLE_DEVICES=MIG-<UUID-A> ib_write_lat --use_cuda=0
  CUDA_VISIBLE_DEVICES=MIG-<UUID-B> ib_write_lat --use_cuda=0 <server_ip>
  ```
  - **성공 기준**: GPU memory 기반 RDMA · 두 MIG partition 간 왕복 latency 측정 완료

### Phase 2.3 · RDMA-based L1↔NRx 파이프라인 구현

- [ ] **Task 1의 IPC 메커니즘을 RDMA로 교체**
  - L1 process · rx_slot buffer를 `ibv_reg_mr` 로 NIC에 pin
  - NRx process · LLR buffer를 `ibv_reg_mr` 로 pin
  - 두 process가 RDMA connection (loopback IP) 통해 · Queue Pair 연결
  - L1 → NRx: `ibv_post_send(WRITE_WITH_IMMEDIATE)`
  - NRx → L1: 완료 후 · 결과 RDMA WRITE

- [ ] **동작 검증** (Gate)
  - Zero-CPU-copy 확인 · CPU memory · cache 오염 없는지 profiler로
  - NRx output 정상성 · Task 1과 동일하게 BER 비교

### Phase 2.4 · 성능 실측 · Task 1과 비교

- [ ] **동일 조건에서 · IPC 방식만 다른 두 실험**
  - **Config X · CP + CPU bounce IPC**: Task 1 결과 (baseline)
  - **Config Y · CP + NIC RDMA loopback**: 새 실험

- [ ] **측정 지표**
  - L1 p99 latency
  - IPC 왕복 시간 · profile 분석
  - NRx aggregate throughput
  - 5G TTI 예산 (500 μs) 내 여유 시간

- [ ] **Expected results**

  | 조건 | 통신 latency | L1 p99 | TTI 예산 여유 |
  | --- | --- | --- | --- |
  | CP + CPU bounce | 100+ μs | baseline + 100 μs | 40% 남음 |
  | CP + NIC RDMA | 5-40 μs | baseline + 5-40 μs | 90% 남음 |

### Task 2 성공 기준
- GPUDirect RDMA loopback · MIG 파티션 간 실측 latency 측정
- L1↔NRx 통신을 RDMA로 실현 · functional 확인
- CPU bounce 대비 latency 개선 정량화 (예상 3-10배)

---

## 실험 순서 · 의존성

```
Phase 1.1 (조사)          Phase 2.1 (환경 확인)
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

Task 1 · Task 2 상당 부분 병렬 진행 가능. 최종 Phase 2.4에서 통합.

---

## 실험 성공 시 논문 최종 결과

**두 이야기가 완결**:

1. **"MIG isolation is necessary but insufficient for AI-RAN"** (완료)
   - Chain 17 · 18 · 19 실측 완료
   - SP 실패 · CP 성공 (통신 없는 경우)

2. **"Real modular AI-RAN needs both isolation AND fast communication"** (Task 1 + 2)
   - Real L1↔NRx 통신 실측
   - MIG + MPS + NIC RDMA loopback = 유일한 완결 답
   - **논문 novelty의 핵심 결과**

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

- [ ] **코드**: `pipeline_l1_nrx_cpu.py` · `pipeline_l1_nrx_rdma.py`
- [ ] **실측 데이터**: L1 p99 latency · 통신 latency · throughput · N-scaling
- [ ] **비교 figure**: CPU bounce vs NIC RDMA 성능 대비
- [ ] **다음 발표 슬라이드**: 지금 발표의 "Next Plan" 슬라이드가 · 실측 데이터로 채워짐
- [ ] **논문 draft 섹션**: "GPU-Direct RDMA Loopback for Modular AI-RAN" 파트

---

## 우선순위 (주별)

- **1주차** · Phase 1.1 · 2.1 · 2.2 (조사 · 환경 · 기본 검증)
- **2주차** · Phase 1.2 · 1.3 (L1↔NRx 파이프라인 · 검증 gate)
- **3주차** · Phase 1.4 · 2.3 (실측 · RDMA 통합)
- **4주차** · Phase 2.4 (최종 비교 · 결과 정리)

**Gate 규칙 준수**: 각 Phase의 검증 단계 · 통과 못 하면 · 다음 phase 진행 안 함 · 원인 파악까지.

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
