# CloudLab d8545 · Fresh Node → Task 1/2 실험 준비 완전 가이드

**목적**: 새 CloudLab 노드를 예약했을 때 · 이 문서만 있으면 30~90분 안에
`SUMMARY.txt` 결과가 나오는 상태까지 복원.

**전제**:
- CloudLab `sgkim` 계정 · `AIRANSLICING` project · d8545 (Wisconsin) 노드
- 이미지: 기본 `UBUNTU22-64-STD`
- 로컬 개발기 (macOS) 에 `~/New_research/` 클론 완료

**작성일**: 2026-08-12 · **최종 결과 반영**: 2026-08-13 · 14-row CPU-RDMA/GDR staging 검증 완료

---

## Step 0 · 노드 예약 후 첫 SSH

```bash
ssh sgkim@<new-node-hostname>.wisc.cloudlab.us
# 확인
nvidia-smi -L                  # 4× A100 인식?
ip -o link | grep enp          # Mellanox interface 이름? (예: enp161s0np0)
df -h /mydata                  # 1.5 T NVMe · empty
sudo -v                        # passwordless sudo
```

노드 hostname 이 바뀌면 로컬 memory 의 `project_cloudlab_aerial.md` 업데이트.

---

## Step 1 · 기반 스택 (25-35분)

### 1-1. NVIDIA driver + CUDA
```bash
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers install nvidia:580   # 또는 최신 550+
sudo reboot
# 재접속 후
nvidia-smi   # 4× A100 · driver 580.x
```

### 1-2. Docker CE (docker.io 는 사용 금지 · containerd metadata 버그)
```bash
# docker.io 제거
sudo apt-get remove -y docker.io docker-doc containerd runc || true
# docker-ce 저장소 등록
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo tee /etc/apt/keyrings/docker.gpg.asc >/dev/null
sudo gpg --batch --yes --dearmor -o /etc/apt/keyrings/docker.gpg /etc/apt/keyrings/docker.gpg.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
# /mydata 를 data-root 로
sudo mkdir -p /mydata/docker /mydata/containerd
sudo tee /etc/docker/daemon.json <<EOF
{"data-root": "/mydata/docker"}
EOF
sudo systemctl restart docker
sudo usermod -aG docker $USER
newgrp docker
docker info | grep "Docker Root Dir"   # /mydata/docker
```

### 1-3. nvidia-container-toolkit
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 1-4. MOFED (Mellanox OFED)
```bash
cd /mydata
wget https://content.mellanox.com/ofed/MLNX_OFED-24.10-3.2.5.0/MLNX_OFED_LINUX-24.10-3.2.5.0-ubuntu22.04-x86_64.tgz
tar xzf MLNX_OFED_LINUX-24.10-3.2.5.0-ubuntu22.04-x86_64.tgz
cd MLNX_OFED_LINUX-24.10-3.2.5.0-ubuntu22.04-x86_64
sudo ./mlnxofedinstall --force --without-fw-update
sudo /etc/init.d/openibd restart
# 검증
ibstat mlx5_0 | head       # State: Down (링크 안 나온 상태 · 다음 스텝에서 UP)
```

### 1-5. nvidia-peermem (GPUDirect RDMA 전제)
```bash
sudo modprobe nvidia-peermem
echo nvidia-peermem | sudo tee /etc/modules-load.d/nvidia_peermem.conf
lsmod | grep nvidia_peermem
```

---

## Step 2 · NIC PHY Loopback UP (2분)

CloudLab experiment 에 LAN link 안 뽑음 → PHY internal loopback 으로 강제 UP.
**리부트마다 재실행 필요**.

```bash
sudo mst start
sudo ip link set enp161s0np0 down
sudo mlxlink -d /dev/mst/mt4123_pciconf0 --link_mode_force --speeds 100G_4X --yes
sudo mlxlink -d /dev/mst/mt4123_pciconf0 -l PH --yes
sudo ip link set enp161s0np0 up
# 검증
ibstat mlx5_0 | grep -E "State|Rate"
# 기대: State: Active · Rate: 100
```

**복구 스크립트로 저장**:
```bash
sudo tee /mydata/nic_loopback_restore.sh <<'EOF'
#!/bin/bash
set -e
sudo mst start >/dev/null 2>&1 || true
sudo ip link set enp161s0np0 down
sudo mlxlink -d /dev/mst/mt4123_pciconf0 --link_mode_force --speeds 100G_4X --yes
sudo mlxlink -d /dev/mst/mt4123_pciconf0 -l PH --yes
sudo ip link set enp161s0np0 up
sleep 2
ibstat mlx5_0 | grep -E "State|Rate"
EOF
sudo chmod +x /mydata/nic_loopback_restore.sh
```

**RoCE loopback 검증**:
```bash
ib_write_lat -d mlx5_0 &
sleep 2
ib_write_lat -d mlx5_0 localhost | tail -5
# 기대: t_avg ~1.3 μs · p99 ~1.4 μs
```

---

## Step 2.5 · RDMA 컨테이너 · 시스템 사전 조건 (5분 · Task 2 필수)

### 2.5-1. RoCE GID 확인 · GID index 결정
```bash
show_gids | grep mlx5_0
# 기대 출력 예:
#   mlx5_0  1  0  fe80:...            v1    enp161s0np0
#   mlx5_0  1  1  ::ffff:192.168...   v1    enp161s0np0
#   mlx5_0  1  2  fe80:...            v2    enp161s0np0
#   mlx5_0  1  3  ::ffff:192.168...   v2    enp161s0np0   ← 대개 이게 RoCE v2 IPv4
```
- **RoCE v2 IPv4** GID index 를 `rdma_channel.py` 의 `RDMA_GID_INDEX` (기본 3) 에 매칭.
- 다르면 컨테이너 실행 시 `-e RDMA_GID_INDEX=<n>` 로 오버라이드.

### 2.5-2. pinned memory rlimit · RDMA MR 등록 전제
```bash
# 호스트 확인
ulimit -l           # unlimited 이면 OK · 숫자면 낮은 상한
# 낮으면 /etc/security/limits.conf 에 추가:
echo "* soft memlock unlimited" | sudo tee -a /etc/security/limits.conf
echo "* hard memlock unlimited" | sudo tee -a /etc/security/limits.conf
# 재로그인 필요 · 또는 컨테이너에 --ulimit memlock=-1 로 우회
```

### 2.5-3. netdev 접근 가능 확인 (컨테이너 안에서)
컨테이너가 RoCE GID 해석을 하려면 그 GID 가 바인딩된 netdev
(예: `enp161s0np0`) 가 **컨테이너의 network namespace 에 보여야** 함.
`--device=/dev/infiniband` 만으론 부족 · **반드시 `--network=host`** 사용.

**RDMA 컨테이너 필수 플래그 세트**:
```bash
docker run [...] \
  --network=host                 \  # ← RoCE netdev 접근 (필수)
  --device=/dev/infiniband        \  # ← libibverbs 디바이스 노드
  --cap-add=IPC_LOCK              \  # ← MR pinned memory 등록
  --ulimit memlock=-1             \  # ← 호스트 rlimit 낮아도 우회
  --ipc=host                      \  # ← 두 컨테이너간 /dev/shm peer info 교환
  --gpus '"device=<MIG-UUID>"'    \  # ← MIG partition (또는 device=N)
  [...]
```

### 2.5-4. NIC PHY loopback 사전 검증 (컨테이너 안에서)
```bash
# 컨테이너에서 pyverbs · NIC 접근 성공하는지 sanity check
docker run --rm --network=host --device=/dev/infiniband \
  --cap-add=IPC_LOCK --ulimit memlock=-1 \
  airan:25-3-rdma python3 -c "
import pyverbs.device as vd
ctx = vd.Context(name='mlx5_0')
p = ctx.query_port(1)
print(f'port state={p.state} lid={p.lid}')          # state=4 (Active) 기대
g = ctx.query_gid(1, 3)
print(f'GID index 3: {g}')                          # IP embedded GID 기대
# netdev 가시성
import os
ndev = open('/sys/class/infiniband/mlx5_0/ports/1/gid_attrs/ndevs/3').read().strip()
print(f'netdev: {ndev}, visible: {os.path.exists(\"/sys/class/net/\"+ndev)}')
"
# 세 줄 모두 성공하면 RDMA 실행 준비 완료
```

**실패 시 대응**:
- `port state=1 (Down)`: Step 2 PHY loopback 다시 실행
- `netdev visible=False`: `--network=host` 빠뜨림
- `pyverbs import 실패`: `airan:25-3-rdma` 이미지 재빌드 (Step 3-4)
- `Errno 19 (ENODEV)` on `to_rtr`: 위 sanity check 하나 실패한 상태

### 2.5-5. RDMA 채널 loopback 종단 테스트 (2 컨테이너)
```bash
sudo rm -f /dev/shm/rdma_test_*
docker rm -f rdma_cons 2>/dev/null

# Consumer (백그라운드)
docker run -d --name rdma_cons \
  --network=host --device=/dev/infiniband \
  --cap-add=IPC_LOCK --ulimit memlock=-1 --ipc=host \
  -v /mydata/aerial-cuda-accelerated-ran:/opt/nvidia/cuBB \
  -w /opt/nvidia/cuBB/pyaerial \
  airan:25-3-rdma python3 rdma_test.py cons
sleep 3

# Producer (foreground)
docker run --rm \
  --network=host --device=/dev/infiniband \
  --cap-add=IPC_LOCK --ulimit memlock=-1 --ipc=host \
  -v /mydata/aerial-cuda-accelerated-ran:/opt/nvidia/cuBB \
  -w /opt/nvidia/cuBB/pyaerial \
  airan:25-3-rdma python3 rdma_test.py prod

# 기대: seq 1 cold는 느릴 수 있음 · seq 2~10 약 100μs · consumer 1~10 모두 수신

docker logs rdma_cons | tail -12
docker rm -f rdma_cons
```

검증 완료 값 (2026-08-12): 1MiB 10회 모두 수신 · 첫/cold WRITE 약 6ms,
seq 2~10 steady-state 평균 107.35μs. 이 값은 `mr.write()`로 CPU MR에
복사하는 시간을 제외한 NIC WRITE+marker 구간이다.

### 2.5-6. GPUDirect 사전 gate

GPU MR 등록 전 IOMMU와 peermem을 확인한다.

```bash
cat /sys/kernel/iommu_groups/$(basename $(readlink \
  /sys/class/infiniband/mlx5_0/device/iommu_group))/type
lsmod | grep nvidia_peermem
for f in /sys/kernel/mm/memory_peers/nv_mem/num_*; do echo "$f=$(cat "$f")"; done
```

현재 d8545는 `DMA-FQ` translated domain이다. NVIDIA 권장 조건은 identity/
pass-through지만 이 노드에서는 64KiB CuPy GPU MR 등록과 두-MIG 실제 WRITE가
성공했다. 따라서 경고로 기록하되 등록·데이터 checksum gate로 실제 지원 여부를
판정한다.

```bash
# 1) 64KiB GPU MR 등록만 검증
docker run --rm --gpus '"device=<MIG-UUID>"' \
  --network=host --device=/dev/infiniband --cap-add=IPC_LOCK \
  --ulimit memlock=-1 --ipc=host \
  -v /mydata/aerial-cuda-accelerated-ran:/opt/nvidia/cuBB \
  -w /opt/nvidia/cuBB/pyaerial airan:25-3-rdma \
  python3 gdr_mr_probe.py 65536
# 성공: CUDA_GDR_POINTER_CAPABLE=1, peermem counter 증가, GDR_PROBE_OK

# 2) 두 MIG 간 payload 검증은 gdr_rdma_test.py cons/prod를 같은 tag로 실행
# 성공: seq 1~10 first_word/checksum verified=1, 양쪽 GDR_RDMA_TEST_OK
```

검증 완료 값 (2026-08-12, cold seq 1 제외한 seq 2~10 평균):
- 64KiB: **61.38μs**
- 실제 L1→NRx forward 1,415,232B: **758.44μs**
- 실제 NRx→L1 backward 1,257,984B: **665.29μs**

이 microbenchmark의 측정 범위는 GPU payload RDMA WRITE + ordered CPU sequence-
marker WRITE다. 왕복이나 전체 L1→NRx→L1 pipeline overhead로 해석하지 않는다.
원시 로그와 mean/p50/p95/min/max는
`task1_final/analysis/TRANSPORT_SUMMARY.csv`에 보관한다.

GPU MR에는 pyverbs `MR.read()`/`MR.write()`를 호출하면 안 된다. 이 helper는
CPU memcpy를 사용한다. GPU allocation을 MR보다 오래 유지하고 MR을 먼저 close한다.

---

## Step 3 · Aerial SDK + Docker Images (25-40분 · pull 시간 대부분)

### 3-1. Aerial 25.3.2 SDK repo (Git LFS 포함)
```bash
sudo apt-get install -y git-lfs
cd /mydata
git clone --branch v25.3.2 https://github.com/NVIDIA/aerial-cuda-accelerated-ran.git
cd aerial-cuda-accelerated-ran
git lfs install
git lfs pull   # HDF5 filter 파일 진짜 바이너리 다운로드
sudo chown -R 1000:1000 /mydata/aerial-cuda-accelerated-ran   # 컨테이너 UID 매치
```

### 3-2. Base Aerial 이미지 pull
```bash
docker pull nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
# ~57 GB · 15-25분 소요
```

### 3-3. airan:25-3-final (torch + transformers)
```bash
# Dockerfile: cloudlab_aerial/Dockerfile.airan
scp <local>/cloudlab_aerial/Dockerfile.airan sgkim@<node>:/tmp/
cd /tmp
docker build -t airan:25-3-final -f Dockerfile.airan .
```

### 3-4. airan:25-3-rdma (pyverbs 57.0)
```bash
scp <local>/cloudlab_results/scripts_for_node/Dockerfile.airan.rdma sgkim@<node>:/tmp/
cd /tmp
docker build -t airan:25-3-rdma -f Dockerfile.airan.rdma .
docker run --rm airan:25-3-rdma python3 -c "import pyverbs.device; print('pyverbs OK')"
```

---

## Step 4 · MIG 구성 (2분)

**Topology A (기본 · Config 1/3/6 용)**: `4g.20gb + 3g.20gb`
```bash
sudo pkill -9 -f nvidia-cuda-mps 2>/dev/null || true
sleep 2
sudo nvidia-smi mig -dci -i 0 || true
sudo nvidia-smi mig -dgi -i 0 || true
sudo nvidia-smi -i 0 -mig 1
sleep 2
sudo nvidia-smi mig -cgi 5,9 -C -i 0
nvidia-smi -L | head
# 기대: MIG 4g.20gb + MIG 3g.20gb (UUID 두 개)
```

**Topology B (Config 4/7 용)**: `3g.20gb + 2g.10gb + 2g.10gb`
(A100 40GB memory slice 제약으로 `3g+3g+1g` 불가 · [[reference-mig-topology]] 참고)
```bash
sudo nvidia-smi mig -dci -i 0
sudo nvidia-smi mig -dgi -i 0
sudo nvidia-smi mig -cgi 9,14,14 -C -i 0
```

---

## Step 5 · 스크립트 배포 (2분)

로컬에서 `cloudlab_results/scripts_for_node/task1/` 통째로 노드 pyaerial 디렉토리에 복사:

```bash
# 로컬에서
scp /Users/changjongkim/New_research/cloudlab_results/scripts_for_node/task1/*.py \
    /Users/changjongkim/New_research/cloudlab_results/scripts_for_node/task1/*.sh \
    sgkim@<node>:/tmp/

# 노드에서
sudo cp /tmp/*.py /mydata/aerial-cuda-accelerated-ran/pyaerial/
sudo chown 1000:1000 /mydata/aerial-cuda-accelerated-ran/pyaerial/*.py
sudo mkdir -p /mydata/results/chain
sudo chown -R sgkim /mydata/results/chain
sudo cp /tmp/run_chain.sh /tmp/run_phase_c.sh /mydata/results/chain/
sudo chown sgkim /mydata/results/chain/*.sh
chmod +x /mydata/results/chain/*.sh
```

**HF cache 준비 (Qwen 7B 다운로드, 약 15GB · 1-3분)**:
```bash
sudo mkdir -p /mydata/hf_cache
sudo chown -R 1000:1000 /mydata/hf_cache
sudo chmod -R 777 /mydata/hf_cache
docker run --rm --gpus '"device=3"' \
  -e HF_HOME=/mydata/hf_cache \
  -v /mydata/hf_cache:/mydata/hf_cache \
  airan:25-3-final python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B', torch_dtype='float16')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B')
print('done')
"
```

---

## Step 6 · 실험 체인 실행 (15-20분)

```bash
cd /mydata/results/chain
nohup bash run_chain.sh </dev/null >orchestrator.log 2>&1 &
disown
```

Monitor로 진행상황 확인 (로컬 Claude Code):
```bash
ssh sgkim@<node> 'tail -F /mydata/results/chain/orchestrator.log' | \
  grep -E "\[ALERT\]|===|: mean=|chain complete"
```

**Phase C (Config 4/7 · Cross-partition · 별도)**:
```bash
# Topology B 로 자동 전환됨
cd /mydata/results/chain
nohup bash run_phase_c.sh </dev/null >phase_c.log 2>&1 &
```

**Phase C · CPU-buffer RDMA 채널 (Task 2 Phase 1 · 완성됨)**:
현재 배포된 `l1_producer.py` / `nrx_consumer.py` 는 이미 RDMA 통합됨.
`run_phase_c.sh` 는 L1/NRx에 `airan:25-3-rdma`, Qwen에 torch가 포함된
`airan:25-3-final` 이미지와 RDMA_ARGS 배열
(`--network=host --cap-add=IPC_LOCK --device=/dev/infiniband --ipc=host`) 을
사용하도록 재작성되어 있음. 그대로 실행:
```bash
# 사전: Step 2.5-5 loopback 테스트가 성공했어야 함
sudo rm -f /dev/shm/rdma_*   # 이전 peer info 정리
nohup bash /mydata/results/chain/run_phase_c.sh </dev/null >phase_c_rdma.log 2>&1 &
```
결과는 `SUMMARY.txt` 에 `config4_rdma`, `config7_rdma` 로 추가됨.
- **실측 mean/p95/p99**: config4_rdma 111.097/111.586/111.664ms,
  config7_rdma 111.375/111.694/111.802ms
- Qwen throughput: 10 it/s (Config 4/7 shm 결과와 동일 조건)

CPU-buffer RDMA 반복 평균은 111.236ms, shm 반복 평균은 110.776ms로 일관된
개선이 아니다. 이 경로에는 `cp.asnumpy/MR.write` 및 `MR.read/cp.asarray`가
남아 있다. GPUDirect 결과와 구분한다.

**Phase C · GPUDirect RDMA staging 채널 (Task 2 Phase 2 · 완성됨)**:
`l1_producer_gdr.py` / `nrx_consumer_gdr.py`는 persistent CuPy allocation을
GPU MR로 등록하고, 1,415,232B forward와 1,257,984B backward payload를 두 MIG
사이에서 직접 RDMA WRITE한다. CPU MR은 ordered u64 sequence marker에만 쓴다.

```bash
# 사전: Step 2.5-6의 GPU MR와 두-MIG checksum gate가 성공했어야 함
sudo find /dev/shm -maxdepth 1 -type f -name 'gdr_rdma_*' -delete
nohup bash /mydata/results/chain/run_phase_c_gdr.sh \
  </dev/null >phase_c_gdr.log 2>&1 &
```

최종 `SUMMARY.txt`의 GDR staging 결과:
- config4_gdr mean/p95/p99 **109.687/109.861/110.223ms**, Qwen 10.23 it/s
- config7_gdr mean/p95/p99 **109.526/109.778/110.349ms**, Qwen 10.23 it/s
- 반복 mean 평균 **109.607ms**: shm 평균 110.776ms 대비 **1.169ms 감소**,
  CPU-buffer RDMA 평균 111.236ms 대비 **1.630ms 감소**

이 경로는 payload의 host bounce를 제거했지만 public pyaerial `TrtEngine`
wrapper가 내부 F-order input copy를 수행하고, LLR output도 registered backward
GPU staging buffer로 한 번 GPU→GPU copy한다. 따라서 **end-to-end zero-copy라고
부르지 않는다**. 위 ms 값은 전체 pipeline 실측이며 Step 2.5-6의 μs transport
값과 직접 등치하지 않는다.

**Fair direct-P2P overlap 재현 (별도 follow-up)**:
기존 Phase C는 slot마다 `L1→NRx→L1` request/response를 직렬 실행하므로 MIG가
제거할 same-partition L1/NRx overlap이 없다. 아래 runner는 총 4g 자원을 맞춰
`same 4g`와 `L1 2g + NRx 2g direct CUDA P2P`를 ring depth 2로 비교한다.

```bash
sudo mkdir -p /mydata/results/p2p_fair/final
cd /mydata/results/p2p_fair
RESULTS_ROOT=/mydata/results/p2p_fair/final \
  nohup bash run_p2p_fair.sh </dev/null >run_p2p_fair.log 2>&1 &
```

Runner는 Topology A에서 4g standalone/same-overlap을 측정하고, Topology B
(`3g+2g+2g`)에서 2g standalone/cross-P2P를 측정한 뒤 Topology A를 복구한다.
양쪽 모두 Qwen은 별도 3g에서 60초 warm-up한다. 기본은 warm-up 20, N=30,
ring depth 2이며 stale `COMPLETE` 경로를 덮어쓰지 않는다.

세 반복 aggregate: same 4g L1 CUDA-stream elapsed 96.935ms (자체 baseline 대비
61.12×), cross 2g+2g P2P 3.316ms (1.49×); p99 197.823ms 대 3.695ms.
P2P 실제 2g↔2g payload gate는 forward/backward mean 64.93/59.15μs와 checksum 10/10을
확인했다. raw 결과/validator: `cloudlab_results/task1_p2p_fair/`.

이 direct-P2P benchmark는 단일 process가 두 MIG CUDA context를 소유한다. 별도
process/container endpoint가 필수면 cross-MIG CUDA IPC가 안 되므로 위 NIC GDR
staging 구조를 사용한다.

**시퀀스 · 실패 시**:
1. Qwen 컨테이너 먼저 · 60초 warmup → NRx 컨테이너 · 15초 안에 CPU-RDMA는 `/dev/shm/rdma_<tag>_fwd_cons.info`, GDR은 `/dev/shm/gdr_rdma_<tag>_fwd_cons.info`가 나와야 함
2. L1 컨테이너 (foreground) → NRx 종료 대기
3. 실패 시 해당 phase 로그의 `[ALERT]`와 `nrx.log`의 `[NRx]` 또는 `[NRx-GDR]` ready/shutdown marker 확인

---

## Step 7 · 결과 확인 · 다운로드

```bash
# 노드에서
cat /mydata/results/chain/SUMMARY.txt
# 최종 기대: 10 base/shm + 2 CPU-buffer RDMA + 2 GDR staging = 14 rows

# 로컬로
scp -r sgkim@<node>:/mydata/results/chain/ \
   /Users/changjongkim/New_research/cloudlab_results/task1_final_<날짜>/
```

---

## 흔한 이슈 · 빠른 대응

| 증상 | 원인 | 대응 |
|---|---|---|
| `docker images` 비어있음 (pull 은 됐는데) | docker.io 29.x containerd metadata 버그 | Step 1-2 : docker-ce 재설치 |
| MIG 파티션 생성 `Insufficient Resources` | 이전 MIG 잔여 · MPS daemon 이 GPU 잡음 | `sudo pkill -9 -f nvidia-cuda-mps` 후 `mig -dci -dgi` 재시도 |
| MIG mode 전환 안 됨 (`pending disable`) | 파티션에 프로세스 남음 | 컨테이너 다 `docker rm -f` · 안 되면 재부팅 |
| pyverbs `ENODEV` on `to_rtr` | 컨테이너에 RoCE netdev 없음 | `docker run` 에 **`--network=host`** 추가 필수 |
| Config 2 (Full MPS) 만 Qwen die | MIG 미 해제 상태에서 GPU 0 사용 | Config 2 는 GPU 3 사용 (스크립트에 반영됨) |
| MPS + MIG 조합 sibling partition 접근 불가 | NVIDIA MPS + MIG 알려진 제약 | Config 3/7 sweep 스킵 · 단일 run 으로 collapse |
| HDF5 `file signature not found` | Aerial repo Git LFS 파일이 pointer | `git lfs install; git lfs pull` |
| pyaerial `CudaStream` ImportError | Aerial 25.3.2 API 변경 | `from aerial.util.cuda import get_cuda_stream` (수정 이미 반영) |
| L1 SM 부족 (Topology B) | 2g partition 은 SM 28 | 예상됨 · L1 baseline ~40-50 ms · MPS(292) 보단 훨씬 나음 |
| RDMA `Cannot allocate memory` on MR 등록 | rlimit memlock 낮음 | 컨테이너에 `--ulimit memlock=-1` 추가 (Step 2.5-2) |
| RDMA `IPC_LOCK` capability 없음 | `--cap-add=IPC_LOCK` 빠짐 | Step 2.5-3 RDMA_ARGS 세트 사용 |
| RDMA 컨테이너 안에서 netdev 못 봄 | `--network=host` 빠짐 | Step 2.5-3 필수 플래그 · 확인: `/sys/class/net/enp*` 존재 |
| `rdma_test.py cons` 가 peer 대기만 함 | producer 쪽 `/dev/shm/rdma_*` info 파일 없음 | 두 컨테이너 다 `--ipc=host` · shm 정리 (`sudo rm /dev/shm/rdma_*`) |
| GID index 다름 (RoCE v1/v2 IPv4/IPv6 혼용) | 노드마다 index 배치 다를 수 있음 | Step 2.5-1 `show_gids` 로 확인 · `-e RDMA_GID_INDEX=<n>` 오버라이드 |

---

## Persistence 안 되는 것들 (매 리부트 후 재실행)

- **NIC PHY loopback**: `/mydata/nic_loopback_restore.sh`
- **MPS daemon** (필요 시): `sudo nvidia-cuda-mps-control -d`
- **MIG 파티션**: `sudo nvidia-smi mig -cgi ...`
- **memlock rlimit** (필요 시): 로그인 세션마다 · 컨테이너는 `--ulimit memlock=-1`

Persistent :
- Driver / Docker / MOFED / nvidia-container-toolkit / nvidia-peermem module
- `/etc/modules-load.d/nvidia_peermem.conf` (module auto-load)
- `/etc/security/limits.conf` (memlock unlimited)
- Docker 이미지 (/mydata/docker)
- Aerial SDK · HF cache (/mydata/*)
