# CloudLab 빈 노드 완전 복구 Runbook

작성 기준: 2026-08-14 최종 백업 및 실제 d8545 검증 상태  
대상: Wisconsin `d8545`, Ubuntu 22.04, 4× A100-SXM4-40GB, ConnectX-6 Dx  
목표: 새 SSH 주소와 비어 있는 `/mydata`만 주어졌을 때, 현재의 Aerial·MIG·MPS·P2P·RDMA·GPUDirect·DART-Rx 실험을 다시 실행 가능한 상태로 복구한다.

이 문서는 신규 노드 복구의 canonical entry point다. 과거 설치 이력과 상세 배경은 `FRESH_CLOUDLAB_SETUP.md`에 남아 있지만, 새 노드에서는 이 문서의 gate 순서를 우선한다.

---

## 1. 복구 기준점

### 1.1 로컬 최종 스냅샷

```text
/Users/changjongkim/New_research/cloudlab_results/cloudlab_final_snapshot_20260814/
```

구성:

| 경로 | 내용 |
|---|---|
| `mydata/results/` | 원격의 모든 실험 결과, trace, JSON/NPZ, 로그, Nsight report, TensorRT engine |
| `mydata/aerial-cuda-accelerated-ran/` | 실제 실행한 Aerial 25.3.2 저장소 전체와 배포 코드, `.git`, Git LFS 객체 |
| `mydata/AIRAN_Changjong/` | 기존 AIRAN 자료 전체 |
| `mydata/hf_cache/` | Qwen2.5-7B Hugging Face model cache |
| `mydata/datasets/` | radio/workload datasets |
| `mydata/torch_cache/` | Torch model/cache data |
| `mydata/downloads/` | MOFED 설치 archive 등 offline bootstrap 자료 |
| `mydata/nic_loopback_restore.sh` | NIC PHY loopback 복구 스크립트 |
| `home/sgkim/cloudlab_aerial/` | 원격 home에 남아 있던 기존 source 사본 |
| `system_config/` | Docker, NVIDIA runtime, peermem, limits, systemd 설정 사본 |
| `local_workspace/cloudlab_aerial/task1/` | 가장 최신 로컬 canonical Task 1/DART-Rx 코드와 runner |
| `local_workspace/cloudlab_aerial/Dockerfile.*` | AI/RDMA workload 이미지 Dockerfile |
| `local_workspace/cloudlab_results/` | 연구 문서, 분석 코드, figure, node 배포 사본 |
| `manifests/` | 파일 SHA-256, symlink, GPU/MIG/RDMA/Docker/package/Git 환경 |

백업 검증 결과:

- `/mydata` 최종 범위 일반 파일: 원격/로컬 각각 **13,651개**, SHA-256 전부 일치
- `/mydata` 최종 범위 symlink: 원격/로컬 각각 **12개**, 대상 일치
- `/users/sgkim/cloudlab_aerial`: 일반 파일 **149개**, SHA-256 전부 일치
- 최종 `/tmp` 보존 대상 **16개**와 시스템 설정 **22개**도 원격/로컬 SHA-256 일치
- 원격 결과 데이터: 2,079,804,030 bytes
- Aerial 저장소 logical file size: 4,440,935,712 bytes
- `pyaerial/core.1`은 2026-08-13 crash dump다. 백업에는 보존되어 있지만 신규 노드 실행에는 필요 없다.

검증 파일:

```text
manifests/remote_files.sha256
manifests/local_files.sha256
manifests/remote_symlinks.tsv
manifests/local_symlinks.tsv
manifests/remote_home_files.sha256
manifests/local_home_files.sha256
manifests/remote_tmp_files.sha256
manifests/local_tmp_files.sha256
manifests/remote_system_config_files.sha256
manifests/local_system_config_files.sha256
manifests/snapshot_all_files.sha256
```

### 1.2 검증된 소프트웨어 기준

| 항목 | 기준 |
|---|---|
| OS | Ubuntu 22.04.2 |
| Driver | NVIDIA 580.173.02 |
| CUDA driver capability | CUDA 13.0 |
| Docker | Docker CE 29.7.2 |
| Aerial | repository HEAD `3bf76a43dceb493b00f2ee75fdfbb87038eab7c6`, Aerial 25.3.2 |
| Base image | `nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb` |
| MOFED | 24.10-3.2.5.0 |
| pyverbs | 57.0, `IBVERBS_PRIVATE_57` |
| Container Python/CuPy | Python 3.10.12, CuPy 13.4.1 |
| TensorRT/Nsight | TensorRT 10.12.0.36, Nsight Systems 2025.3.1 |
| NIC | `mlx5_0`, ConnectX-6 Dx, PHY loopback 100GbE |
| GID | 현재 노드는 index 3이 RoCE v2 IPv4였으나 새 노드에서 반드시 재탐색 |

최종 이미지 ID는 `manifests/docker.txt`, 패키지는
`image_final_packages.txt`, `image_rdma_packages.txt`,
`image_radio_packages.txt`, `image_radio_rdma_packages.txt`가 기준이다.
Docker build timestamp와 apt index 때문에 새 이미지 ID가 달라도
package/import/smoke gate가 같으면 재현 성공으로 본다.

---

## 2. 전체 복구 순서와 예상 시간

| Gate | 내용 | 예상 |
|---|---|---:|
| R0 | 노드/디스크/GPU/NIC 식별 | 5분 |
| R1 | driver, Docker CE, NVIDIA container toolkit | 15–30분 + reboot |
| R2 | MOFED, peermem, NIC loopback | 15–30분 + 필요 시 reboot |
| R3 | 로컬 snapshot에서 code/input 복구 | 5–15분 |
| R4 | base pull + 4개 workload image build | 25–60분 |
| R5 | MIG topology와 UUID 동적 생성 | 5분 + 필요 시 reboot |
| R6 | compile/import/unit tests | 5분 |
| R7 | CPU RDMA → GDR MR → CUDA-IPC/GDR payload gate | 10분 |
| R8 | direct TensorRT와 actual-radio smoke | 10–20분 |

인터넷과 NGC pull 속도에 따라 완전한 빈 노드는 1.5–3시간이 현실적이다. 설치와 base image pull, 로컬 snapshot 전송은 가능한 한 병렬로 진행한다.

---

## 3. R0 — 새 노드 식별

로컬에서 새 hostname만 설정한다.

```bash
CLOUDLAB_HOST=sgkim@<new-host>.wisc.cloudlab.us
ssh "$CLOUDLAB_HOST"
```

새 노드에서:

```bash
hostname -f
cat /etc/os-release
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINTS
df -h /mydata
nvidia-smi -L
lspci -nn | grep -Ei 'NVIDIA|Mellanox'
rdma link show 2>/dev/null || true
ip -o link show
sudo -v
```

필수 조건:

- A100 40GB 4개가 보여야 한다.
- ConnectX-6 Dx가 보여야 한다.
- `/mydata`가 약 1.5TB NVMe에 mount되어 있어야 한다.
- `sudo`가 passwordless여야 한다.

`/mydata`가 mount되지 않았다고 해서 즉시 format하지 않는다. `lsblk -f`로 대상 NVMe와 기존 filesystem이 비어 있음을 확인한 후 CloudLab provisioning에 맞춰 mount한다. 이전 `post_reboot_setup.sh`의 무조건적인 `mkfs.ext4 -F`를 새 노드에 그대로 실행하지 않는다.

NIC 이름과 MST device는 노드마다 다시 찾는다.

```bash
rdma link show
readlink -f /sys/class/infiniband/mlx5_0/device/net/*
sudo mst start
sudo mst status -v
```

이 문서의 예시는 `enp161s0np0`, `/dev/mst/mt4123_pciconf0`이다. 다르면 발견한 값을 사용한다.

---

## 4. R1 — Driver, Docker CE, NVIDIA container runtime

### 4.1 NVIDIA driver

이미 580 계열 driver가 정상이라면 재설치하지 않는다.

```bash
nvidia-smi
```

없거나 깨졌을 때:

```bash
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers install nvidia:580
sudo reboot
```

재접속 후 `nvidia-smi -L`이 먼저 통과해야 한다. Driver를 MOFED 뒤에 바꾸면 `nvidia_peermem`과 OFED kernel module을 다시 빌드해야 할 수 있으므로 driver를 먼저 고정한다.

### 4.2 Docker CE

Ubuntu `docker.io`와 Docker CE를 섞지 않는다. 과거 `docker pull` 후 image metadata가 보이지 않는 문제가 있었으므로 Docker CE를 사용한다.

```bash
sudo apt-get remove -y docker.io docker-doc containerd runc || true
sudo apt-get install -y ca-certificates curl gnupg git git-lfs rsync jq zstd
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo tee /etc/apt/keyrings/docker.asc >/dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
```

Docker root를 NVMe로 고정한다.

```bash
sudo install -d -m 0755 /mydata/docker /etc/docker
echo '{"data-root":"/mydata/docker"}' | sudo tee /etc/docker/daemon.json
sudo systemctl restart docker
sudo usermod -aG docker sgkim
docker info | grep 'Docker Root Dir'
```

기대: `/mydata/docker`. 그룹 적용 전에는 `sudo docker`를 사용하거나 다시 로그인한다.

### 4.3 NVIDIA container toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

검증:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi -L
```

---

## 5. R2 — MOFED, nvidia-peermem, NIC PHY loopback

### 5.1 MOFED

Driver 설치가 끝난 뒤 MOFED를 설치한다.

```bash
cd /mydata
if [ -f /mydata/downloads/mofed.tgz ]; then
  cp /mydata/downloads/mofed.tgz \
    MLNX_OFED_LINUX-24.10-3.2.5.0-ubuntu22.04-x86_64.tgz
else
  wget https://content.mellanox.com/ofed/MLNX_OFED-24.10-3.2.5.0/MLNX_OFED_LINUX-24.10-3.2.5.0-ubuntu22.04-x86_64.tgz
fi
tar xzf MLNX_OFED_LINUX-24.10-3.2.5.0-ubuntu22.04-x86_64.tgz
cd MLNX_OFED_LINUX-24.10-3.2.5.0-ubuntu22.04-x86_64
sudo ./mlnxofedinstall --force --without-fw-update
sudo /etc/init.d/openibd restart
```

```bash
modinfo mlx5_ib | head
ibv_devices
ibv_devinfo -d mlx5_0
```

Soft-RoCE `rxe`는 MOFED와 충돌했으며 사용하지 않는다.

### 5.2 nvidia-peermem

```bash
sudo modprobe mlx5_ib
sudo modprobe ib_uverbs
sudo modprobe nvidia-peermem
echo nvidia-peermem | sudo tee /etc/modules-load.d/nvidia_peermem.conf
lsmod | grep -E 'mlx5_ib|ib_uverbs|nvidia_peermem'
```

`nvidia_peermem`이 없으면 driver/MOFED 순서를 확인하고 재부팅 후 다시 load한다.

### 5.3 PHY loopback과 RoCE IPv4

```bash
NIC_IF=enp161s0np0
MST_DEV=/dev/mst/mt4123_pciconf0
sudo mst start
sudo ip link set "$NIC_IF" down
sudo mlxlink -d "$MST_DEV" --link_mode_force --speeds 100G_4X --yes
sudo mlxlink -d "$MST_DEV" -l PH --yes
sudo ip link set "$NIC_IF" up
sudo ip addr replace 192.168.99.1/24 dev "$NIC_IF"
sleep 2
ibv_devinfo -d mlx5_0 | grep -E 'state:|phys_state:|active_mtu:'
```

기대: `PORT_ACTIVE`, `LINK_UP`. 복구 스크립트는 snapshot의 `mydata/nic_loopback_restore.sh`를 복사한 뒤 새 interface/MST device가 다르면 수정한다.

GID index는 hardcode하지 않는다.

```bash
show_gids | grep mlx5_0
for f in /sys/class/infiniband/mlx5_0/ports/1/gids/*; do
  idx=${f##*/}; gid=$(cat "$f")
  type=$(cat "/sys/class/infiniband/mlx5_0/ports/1/gid_attrs/types/$idx" 2>/dev/null || true)
  ndev=$(cat "/sys/class/infiniband/mlx5_0/ports/1/gid_attrs/ndevs/$idx" 2>/dev/null || true)
  printf '%s %s %s %s\n' "$idx" "$gid" "$type" "$ndev"
done
```

`RoCE v2`, IPv4-mapped `192.168.99.1`, 해당 NIC netdev인 index를 이후 `RDMA_GID_INDEX`로 쓴다. 이전 노드는 3이었다.

```bash
export RDMA_GID_INDEX=<discovered-RoCE-v2-IPv4-index>
```

Host loopback sanity:

```bash
ib_write_lat -d mlx5_0 -x 3 -s 2 -n 100 -p 18600 &
server_pid=$!
sleep 1
ib_write_lat -d mlx5_0 -x 3 -s 2 -n 100 -p 18600 192.168.99.1
wait "$server_pid"
```

기대: 성공 및 host verbs latency 약 1–2µs. `-x` 값은 새 GID index로 바꾼다.

---

## 6. R3 — 로컬 snapshot에서 code와 input 복구

### 6.1 실행 저장소 복구

로컬에서:

```bash
CLOUDLAB_HOST=sgkim@<new-host>.wisc.cloudlab.us
SNAP=/Users/changjongkim/New_research/cloudlab_results/cloudlab_final_snapshot_20260814

rsync -aS --exclude 'pyaerial/core.1' \
  "$SNAP/mydata/aerial-cuda-accelerated-ran/" \
  "$CLOUDLAB_HOST:/tmp/aerial-cuda-accelerated-ran/"

ssh "$CLOUDLAB_HOST" \
  'sudo rsync -a /tmp/aerial-cuda-accelerated-ran/ /mydata/aerial-cuda-accelerated-ran/ && sudo chown -R 1000:1000 /mydata/aerial-cuda-accelerated-ran'
```

운영 복구에서는 crash dump `core.1`을 제외한다. 원본은 로컬 snapshot에 유지된다.

가장 최신 로컬 canonical 코드로 overlay한다.

```bash
rsync -a "$SNAP/local_workspace/cloudlab_aerial/task1/" \
  "$CLOUDLAB_HOST:/tmp/task1_current/"
ssh "$CLOUDLAB_HOST" \
  'sudo rsync -a /tmp/task1_current/ /mydata/aerial-cuda-accelerated-ran/pyaerial/ && sudo chown -R 1000:1000 /mydata/aerial-cuda-accelerated-ran/pyaerial'
```

### 6.2 기존 결과와 input 복구

완전한 과거 결과를 새 노드에 올릴 필요가 있으면 새 output과 섞지 않도록 reference root에 둔다.

```bash
ssh "$CLOUDLAB_HOST" \
  'sudo install -d -o sgkim -g sgkim /mydata/reference_results_20260814'
rsync -a "$SNAP/mydata/results/" \
  "$CLOUDLAB_HOST:/mydata/reference_results_20260814/"
```

실험 실행에 필요한 최소 input만 기본 경로에 복사한다.

```bash
ssh "$CLOUDLAB_HOST" '
  sudo install -d -o sgkim -g sgkim /mydata/results/nrx_deep_profile /mydata/results/isca_v2
  rsync -a /mydata/reference_results_20260814/nrx_deep_profile/engines/ /mydata/results/nrx_deep_profile/engines/
  rsync -a /mydata/reference_results_20260814/isca_v2/justification_radio_official_3trials/ /mydata/results/isca_v2/justification_radio_official_3trials/
  rsync -a /mydata/reference_results_20260814/isca_v2/dart_rx_integrated_campaign/ /mydata/results/isca_v2/dart_rx_integrated_campaign/
  rsync -a /mydata/reference_results_20260814/isca_v2/mig_causal_20260813T1138Z/07_multicell_workloads/traces/ /mydata/results/isca_v2/mig_causal_20260813T1138Z/07_multicell_workloads/traces/
  chmod -R a+rX /mydata/results
'
```

핵심 input:

- `neural_rx_fp16_{2g,4g,full}.trt`
- `RADIO_UTILITY_MAP.csv`
- paired actual-radio `trace_t{1,2,3}.npz`
- single/multi-cell/selective/bursty trace set

Qwen/background workload와 dataset을 인터넷에서 다시 받지 않으려면 cache도 복구한다.

```bash
ssh "$CLOUDLAB_HOST" \
  'sudo install -d -o sgkim -g sgkim /mydata/hf_cache /mydata/datasets /mydata/torch_cache /mydata/downloads'
for cache_name in hf_cache datasets torch_cache downloads; do
  rsync -aS "$SNAP/mydata/${cache_name}/" \
    "$CLOUDLAB_HOST:/mydata/${cache_name}/"
done
```

Container에는 필요에 따라 다음을 mount/설정한다.

```text
-e HF_HOME=/mydata/hf_cache -v /mydata/hf_cache:/mydata/hf_cache
-e TORCH_HOME=/mydata/torch_cache -v /mydata/torch_cache:/mydata/torch_cache
-v /mydata/datasets:/mydata/datasets:ro
```

새 실험 output root에는 날짜를 붙인다. 과거 `COMPLETE`, `FINAL_COMPLETE`, `SUMMARY.txt`와 같은 경로를 재사용하지 않는다.

---

## 7. R4 — Docker image 4종 복구

필요하면 먼저 `docker login nvcr.io`를 대화형으로 수행한다. NGC token은 문서, shell script, history에 기록하지 않는다.

Dockerfile을 새 노드로 보낸다.

```bash
ssh "$CLOUDLAB_HOST" 'mkdir -p /tmp/airan-build'
rsync -a "$SNAP/local_workspace/cloudlab_aerial/Dockerfile.airan" \
  "$SNAP/local_workspace/cloudlab_aerial/Dockerfile.airan.rdma" \
  "$CLOUDLAB_HOST:/tmp/airan-build/"
rsync -a "$SNAP/local_workspace/cloudlab_aerial/task1/isca_v2/Dockerfile.radio" \
  "$SNAP/local_workspace/cloudlab_aerial/task1/isca_v2/Dockerfile.radio-rdma" \
  "$CLOUDLAB_HOST:/tmp/airan-build/"
```

새 노드에서:

```bash
docker pull nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
cd /tmp/airan-build
docker build --pull=false -t airan:25-3-final -f Dockerfile.airan .
```

현재 `airan:25-3-final`에는 pyaerial build/install commit이 하나 더 있었다. 아래 절차를 재현한다.

```bash
docker run -d --name aerial-build --gpus all \
  -v /mydata/aerial-cuda-accelerated-ran:/opt/nvidia/cuBB \
  -w /opt/nvidia/cuBB airan:25-3-final bash -lc '
    set -e
    cmake -Bbuild -GNinja -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/native \
      -DNVIPC_FMTLOG_ENABLE=OFF -DASIM_CUPHY_SRS_OUTPUT_FP32=ON
    cmake --build build -t _pycuphy pycuphycpp -j16
    bash /opt/nvidia/cuBB/pyaerial/scripts/install_dev_pkg.sh
    python3 -c "import aerial; print(aerial.__file__)"
  '
docker wait aerial-build
test "$(docker inspect -f '{{.State.ExitCode}}' aerial-build)" = 0
docker commit aerial-build airan:25-3-final
docker rm aerial-build
```

나머지 이미지:

```bash
cd /tmp/airan-build
docker build --pull=false -t airan:25-3-rdma -f Dockerfile.airan.rdma .
docker build --pull=false -t airan:25-3-radio -f Dockerfile.radio .
docker build --pull=false -t airan:25-3-radio-rdma -f Dockerfile.radio-rdma .
```

Import gate:

```bash
docker run --rm --gpus all \
  -v /mydata/aerial-cuda-accelerated-ran:/opt/nvidia/cuBB \
  -w /opt/nvidia/cuBB/pyaerial airan:25-3-final \
  python3 -c 'import aerial,torch,transformers,cupy; from aerial.phy5g.algorithms import ChannelEstimator; print("FINAL_OK")'

docker run --rm airan:25-3-rdma \
  python3 -c 'import pyverbs.device; print("RDMA_OK")'

docker run --rm airan:25-3-radio \
  python3 -c 'import tensorflow,sionna; print(tensorflow.__version__,sionna.__version__,"RADIO_OK")'

docker run --rm airan:25-3-radio-rdma \
  python3 -c 'import tensorflow,sionna,pyverbs.device; print("RADIO_RDMA_OK")'
```

버전 차이는 `manifests/image_*_packages.txt`와 비교한다.

---

## 8. R5 — MIG topology와 UUID

### 8.1 기본 종료/정리

Topology를 바꾸기 전에 외부 GPU 작업이 없어야 한다.

```bash
docker ps
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv
sudo pkill -9 -f nvidia-cuda-mps 2>/dev/null || true
```

### 8.2 기본 실험 상태

- GPU 0: MIG `4g.20gb + 3g.20gb`
- GPU 1, 2, 3: full GPU, MIG disabled

```bash
sudo nvidia-smi -i 0 -mig 1
sudo nvidia-smi mig -i 0 -dci 2>/dev/null || true
sudo nvidia-smi mig -i 0 -dgi 2>/dev/null || true
sudo nvidia-smi mig -i 0 -cgi 4g.20gb,3g.20gb -C
for gpu_idx in 1 2 3; do sudo nvidia-smi -i "$gpu_idx" -mig 0 || true; done
nvidia-smi -L
```

`pending enable/disable`이면 모든 container/process를 정리하고 reboot한다. MIG UUID는 재생성할 때마다 바뀌므로 문서나 runner 기본값을 신뢰하지 않는다.

```bash
nvidia-smi -L > /mydata/results/topology_before_run.txt
```

각 runner에는 `L1_GPU`, `SOURCE_GPU`, `ENDPOINT_GPUS`를 현재 `nvidia-smi -L` 값으로 명시적으로 넘긴다.

기본 topology에서는 다음처럼 현재 UUID를 shell 변수에 넣을 수 있다.

```bash
L1_GPU=$(nvidia-smi -L | awk '
  /^GPU 0:/ {inside=1; next} /^GPU [1-9][0-9]*:/ {inside=0}
  inside && /MIG 4g.20gb/ {match($0,/UUID: MIG-[0-9a-f-]+/); print substr($0,RSTART+6,RLENGTH-6); exit}')
LOCAL_NRX_GPU=$(nvidia-smi -L | awk '
  /^GPU 0:/ {inside=1; next} /^GPU [1-9][0-9]*:/ {inside=0}
  inside && /MIG 3g.20gb/ {match($0,/UUID: MIG-[0-9a-f-]+/); print substr($0,RSTART+6,RLENGTH-6); exit}')
GPU1_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 1)
GPU2_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 2)
GPU3_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 3)
printf 'L1=%s\nLOCAL_NRX=%s\nGPU1=%s\nGPU2=%s\nGPU3=%s\n' \
  "$L1_GPU" "$LOCAL_NRX_GPU" "$GPU1_UUID" "$GPU2_UUID" "$GPU3_UUID"
```

### 8.3 pool topology

Open-loop 3-endpoint fixed-MIG pool 캠페인은 일시적으로 다음을 사용한다.

- GPU 0: source 4g + local endpoint 3g
- GPU 1: 4g + remote endpoint 3g
- GPU 2: 4g + remote endpoint 3g
- GPU 3: full GPU 또는 background workload

`run_dart_rx_gdr_pool_autonomous.sh`는 GPU 1/2를 변경하고 완료 후 full GPU로 복구한다. 실행 전 foreign GPU process가 없어야 한다.

---

## 9. R6 — 소스 compile와 unit tests

로컬 canonical test:

```bash
cd /Users/changjongkim/New_research/cloudlab_aerial/task1
python3 -m py_compile *.py isca_v2/*.py
python3 -m unittest -v test_dart_rx_core.py test_dart_rx_policy_v2.py
```

검증 기준: 현재 DART core/policy test **15개 통과**.
최종 로컬 compile, unit-test, 전체 shell syntax 결과는 snapshot의
`manifests/local_source_validation.txt`에 보존되어 있다.

새 노드 container에서 syntax/import:

```bash
docker run --rm --gpus all \
  -v /mydata/aerial-cuda-accelerated-ran:/opt/nvidia/cuBB \
  -w /opt/nvidia/cuBB/pyaerial airan:25-3-radio-rdma bash -lc '
    python3 -m py_compile *.py isca_v2/*.py
    python3 -m unittest -v test_dart_rx_core.py test_dart_rx_policy_v2.py
  '
```

`core.1`이 생기면 그 run은 성공으로 보지 않는다. 해당 container log와 `dmesg`를 먼저 보존한다.

---

## 10. R7 — RDMA/GDR smoke ladder

아래 gate를 건너뛰고 full campaign을 시작하지 않는다.

### G0. Container에서 RDMA device와 netdev

```bash
docker run --rm --network=host --device=/dev/infiniband \
  --cap-add=IPC_LOCK --ulimit memlock=-1:-1 \
  -e RDMA_GID_INDEX="$RDMA_GID_INDEX" \
  airan:25-3-rdma python3 -c '
import os
from pyverbs.device import Context
c=Context(name="mlx5_0")
print(c.query_port(1))
print(c.query_gid(1,int(os.environ["RDMA_GID_INDEX"])))
print(os.listdir("/sys/class/net"))
'
```

`to_rtr: ENODEV`는 QP attr mask 문제가 아니라 대부분 Docker network namespace 문제였다. RDMA container에는 항상 다음을 함께 쓴다.

```text
--network=host --device=/dev/infiniband --cap-add=IPC_LOCK
--ulimit memlock=-1:-1 --ipc=host
```

GID가 3이 아니면 `RDMA_GID_INDEX`를 새 값으로 설정한다.

### G1. CPU MR RDMA loopback

`rdma_test.py cons`를 먼저 background로 실행하고 `prod`를 실행한다. 두 container 모두 위 RDMA flag 세트를 사용한다.

성공 조건:

- consumer endpoint ready
- seq 1–10, `first_word=N`
- 양쪽 done
- 1MiB steady write 약 100µs 수준

### G2. GPUDirect MR registration

```bash
docker run --rm --gpus "device=$L1_GPU" \
  --network=host --device=/dev/infiniband --cap-add=IPC_LOCK \
  --ulimit memlock=-1:-1 --ipc=host \
  -v /mydata/aerial-cuda-accelerated-ran:/opt/nvidia/cuBB \
  -w /opt/nvidia/cuBB/pyaerial airan:25-3-rdma \
  python3 gdr_mr_probe.py 65536
```

성공 조건: `CUDA_GDR_POINTER_CAPABLE=1`, `GDR_PROBE_OK`, peermem counter 증가.

GPU MR에 pyverbs `MR.read()`/`MR.write()`를 사용하지 않는다. 이 helper는 CPU memcpy이며 CUDA VA를 CPU에서 역참조할 수 있다. GPU allocation을 MR보다 오래 유지하고 MR을 먼저 close한다.

### G3. CUDA IPC mapped source MR → GDR

```bash
cd /mydata/aerial-cuda-accelerated-ran/pyaerial
SOURCE_GPU="$L1_GPU" WORKER_GPU=1 \
PAYLOAD_SIZE=65536 ITERATIONS=100 \
OUT_ROOT=/mydata/results/gdr_cuda_ipc_gate_new \
bash run_gdr_cuda_ipc_gate.sh

SOURCE_GPU="$L1_GPU" WORKER_GPU=1 \
PAYLOAD_SIZE=1415232 ITERATIONS=10000 \
OUT_ROOT=/mydata/results/gdr_cuda_ipc_gate_new \
bash run_gdr_cuda_ipc_gate.sh
```

검증된 이전 기준:

- 64KiB × 100: error 0, mean 135.62µs, p99 161.79µs
- 1,415,232B × 10,000: error 0, mean 1.40597ms, p99 1.43208ms

새 노드 값이 달라도 checksum error는 0이어야 한다.

### G4. Stale state cleanup

비정상 종료 후 새 tag를 쓰거나 다음의 bounded pattern만 정리한다.

```bash
sudo find /dev/shm -maxdepth 1 -type f \
  \( -name 'rdma_*' -o -name 'gdr_rdma_*' \) -delete
```

`rdma_channel.py`의 과거 CPU endpoint는 stale peer generation 검증이 약하므로 cleanup이 필수다. GDR DART endpoint는 session UUID를 사용하지만 cleanup을 유지한다.

---

## 11. R8 — TensorRT와 actual-radio end-to-end gate

### 11.1 Engine 확인

```bash
ls -lh /mydata/results/nrx_deep_profile/engines/
```

필수:

- `neural_rx_fp16_2g.trt`
- `neural_rx_fp16_4g.trt`
- `neural_rx_fp16_full.trt`

Engine은 GPU profile과 TensorRT/runtime 호환성을 확인한다. deserialize 실패 시 이전 engine을 억지로 재사용하지 말고 `nrx_trt_direct.py`의 build path로 해당 profile에서 다시 생성한다.

### 11.2 Actual-radio 1-endpoint smoke

현재 topology에서 UUID를 동적으로 채운다.

```bash
cd /mydata/aerial-cuda-accelerated-ran/pyaerial
L1_GPU="$L1_GPU" \
ENDPOINT_GPUS="$GPU1_UUID" \
MODE=all ROUTING=round_robin ITERATIONS=6 WARMUP=2 DEADLINE_MS=12 \
TRIAL=1 RESULT_ROOT=/mydata/results/isca_v2/dart_rx_radio_pool_new \
bash run_dart_rx_radio_pool.sh
```

성공 조건:

- actual path: cuPHY CE → CUDA-IPC mapped source GPU MR → GDR → resident TensorRT NRx → GDR LLR → LDPC/CRC
- `STATUS.txt`가 PASS
- L1과 worker exit 0
- deadline miss 0, late completion 0
- result의 endpoint count와 실제 사용 endpoint 일치

### 11.3 Actual-radio 3-endpoint correctness/utility

```bash
ENDPOINT_GPUS="$GPU1_UUID,$GPU2_UUID,$GPU3_UUID"
for trace_id in 1 2 3; do
  for mode in none all utility; do
    L1_GPU="$L1_GPU" ENDPOINT_GPUS="$ENDPOINT_GPUS" \
    MODE="$mode" ROUTING=round_robin ITERATIONS=100 WARMUP=2 \
    DEADLINE_MS=12 TRIAL="$trace_id" \
    INPUT_TRACE=/mydata/results/isca_v2/dart_rx_integrated_campaign/paired_final_20260813T104930Z/trace_t${trace_id}.npz \
    RESULT_ROOT=/mydata/results/isca_v2/dart_rx_radio_pool_new \
    bash run_dart_rx_radio_pool.sh
  done
done
python3 analyze_dart_rx_radio_pool.py \
  --root /mydata/results/isca_v2/dart_rx_radio_pool_new
```

이전 paired 기준:

| mode | NRx requests | median correct | decision p99 | miss/late |
|---|---:|---:|---:|---:|
| none | 0 | 0.620 | 1.292ms | 0/0 |
| all | 100 | 0.800 | 5.139ms | 0/0 |
| utility | 75 | 0.800 | 5.050ms | 0/0 |

cuPHY의 알려진 cross-process nondeterminism 때문에 개별 TB가 ±1 정도 달라질 수 있다. paired median과 safety invariant로 판정한다.

### 11.4 Nsight

```bash
NSYS=1 ITERATIONS=6 WARMUP=2 DEADLINE_MS=50 \
L1_GPU="$L1_GPU" \
ENDPOINT_GPUS="$GPU1_UUID,$GPU2_UUID,$GPU3_UUID" \
MODE=all RESULT_ROOT=/mydata/results/isca_v2/dart_rx_radio_pool_new \
bash run_dart_rx_radio_pool.sh
```

NVTX capture trigger가 report를 만들지 못했던 적이 있다. 현재 runner는 warmup 후 CUDA Profiler API range를 사용한다. 성공 시 `nsys_l1.nsys-rep`가 생겨야 한다.

이전 source-side profile의 주요 관측:

- float2↔half2 conversion kernel 합계 46.4%
- CE+pack 약 1.3ms
- remote exchange 약 2.0ms
- worker NRx service 약 1.11ms

---

## 12. 전체 캠페인 runner 지도

| 목적 | runner/entry point | 주요 결과 root |
|---|---|---|
| MIG/MPS/GDR placement matrix | `isca_v2/run_mig_mps_gdr_matrix.sh` | `results/placement_matrix`, `results/isca_v2` |
| CPU-shm/RDMA/GDR chain | `run_chain.sh`, `run_phase_c.sh`, `run_phase_c_gdr.sh`의 archived copy | `results/chain` |
| P2P fair/direct TRT | `run_p2p_fair.sh` | `results/p2p_fair` |
| NRx wrapper/raw profiling | `nrx_deep_profile.py`, `nrx_trt_direct.py`, `isca_v2/run_nrx_profilers_full.sh` | `results/nrx_deep_profile` |
| NRx replica/service sweep | `run_nrx_replica_sweep.sh`, `nrx_independent_endpoint_sweep.py` | `results/drain_free` |
| Fixed MIG/background elasticity | `run_fixed_mig_elastic_experiment.sh`, `isca_v2/run_fixed_mig_background_suite.sh` | `results/drain_free`, `results/isca_v2` |
| Realistic multi-cell traces | `isca_v2/run_multicell_workloads_full.sh` | `results/isca_v2/.../07_multicell_workloads` |
| Radio utility justification | `isca_v2/run_radio_utility_full.sh`, `run_radio_labelled_replay_full.sh` | `results/isca_v2/justification_*` |
| GDR open-loop pool | `run_dart_rx_gdr_pool_autonomous.sh` | dated GDR pool result root |
| Actual-radio multi-endpoint | `run_dart_rx_radio_pool.sh` | `results/isca_v2/dart_rx_radio_pool*` |

Runner의 hardcoded UUID/default output을 그대로 신뢰하지 않는다. 실행 전 source/endpoint UUID와 새로운 result root를 environment로 명시한다.

---

## 13. 실패 진단표

| 증상 | 실제 원인/진단 | 해결 |
|---|---|---|
| `docker pull` 후 image가 사라짐 | Ubuntu docker.io/containerd metadata 문제 | Docker CE로 통일, root `/mydata/docker` 확인 |
| root filesystem 부족 | Docker/containerd가 NVMe가 아님 | `docker info`, `/etc/docker/daemon.json` 확인 |
| HDF5 signature 오류 | Git LFS pointer만 복구 | snapshot repo 사용 또는 `git lfs pull` |
| `import aerial` 실패 | pyaerial build/install commit 없음 | R4의 cmake/build/install/commit 재실행 |
| MIG `Insufficient Resources` | compute instance/MPS/container 잔존 | foreign process 확인 후 dci→dgi→cgi 순서 |
| MIG mode `pending` | driver가 live reset 못 함 | workload 정리 후 reboot |
| UUID가 invalid | MIG 재생성으로 UUID 변경 | `nvidia-smi -L`에서 동적 재주입 |
| `to_rtr` ENODEV | container netns에 RoCE netdev 없음 | `--network=host`; attr mask 변경하지 않음 |
| MR `Cannot allocate memory` | memlock 부족 | `--cap-add=IPC_LOCK --ulimit memlock=-1:-1` |
| GDR MR 등록 실패 | peermem 미load, GPU pointer/SYNC_MEMOPS 문제 | module/counter 확인, `gdr_mr_probe.py`로 분리 |
| GDR checksum mismatch | payload publish 전에 marker 또는 buffer reuse | 같은 RC QP에서 payload completion 후 marker; ACK/fence 유지 |
| worker가 첫 seq를 거부 | 전역 seq를 endpoint-local channel에 사용 | endpoint별 seq counter 사용; 현재 코드에 수정됨 |
| worker가 peer 대기 | source/worker tag 불일치 또는 stale shm | 동일 tag 확인, bounded `/dev/shm` cleanup |
| result directory permission | container UID와 host owner 불일치 | result root `0777` 또는 UID 1000 ownership |
| Nsight `no reports` | NVTX trigger가 warmup/capture와 불일치 | 현재 CUDA Profiler API capture range 사용 |
| single-stream에서 replica latency가 안 줄음 | replica는 concurrency capacity용 | synchronous latency 개선으로 주장하지 않음; open-loop trace 사용 |
| 1ms arrival을 실제 wrapper가 생성 못 함 | CE+pack 자체가 1ms 이상 | pre-captured actual CE tensor absolute-time replay 또는 cell-parallel/batched L1 |

---

## 14. 재부팅 후 최소 복구

```bash
sudo modprobe mlx5_ib
sudo modprobe ib_uverbs
sudo modprobe nvidia-peermem
sudo bash /mydata/nic_loopback_restore.sh
sudo ip addr replace 192.168.99.1/24 dev enp161s0np0
nvidia-smi -L
ibv_devinfo -d mlx5_0 | grep -E 'state:|phys_state:'
docker image ls
```

MIG instance가 유지되지 않았거나 UUID가 바뀌었으면 R5를 다시 실행하고 runner environment를 갱신한다.

---

## 15. 새 실험 종료 후 재백업

실험을 모두 정지시켜 consistent snapshot을 만든다.

```bash
docker ps
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv
```

로컬에서 날짜별 새 경로로 받고 기존 snapshot을 덮어쓰지 않는다.

```bash
CLOUDLAB_HOST=sgkim@<new-host>.wisc.cloudlab.us
BACKUP_ROOT=/Users/changjongkim/New_research/cloudlab_results/cloudlab_snapshot_<YYYYMMDD>
mkdir -p "$BACKUP_ROOT/mydata"
rsync -aS --partial "$CLOUDLAB_HOST:/mydata/results/" "$BACKUP_ROOT/mydata/results/"
rsync -aS --partial "$CLOUDLAB_HOST:/mydata/aerial-cuda-accelerated-ran/" "$BACKUP_ROOT/mydata/aerial-cuda-accelerated-ran/"
```

원격과 로컬에서 같은 상대 경로로 SHA-256을 생성하고 비교한다. file count나 `du`만 같다고 완료 처리하지 않는다. Sparse core dump가 있으면 `rsync -S`를 사용하거나 crash dump를 별도 압축하되 원본 hash를 기록한다.

---

## 16. 최종 Ready 판정

다음이 모두 참이어야 full campaign을 시작한다.

- [ ] 4× A100 인식, 필요한 MIG topology와 현재 UUID 저장
- [ ] Docker root `/mydata/docker`
- [ ] 네 workload image import gate 통과
- [ ] `mlx5_0` `PORT_ACTIVE`, RoCE v2 IPv4 GID 확인
- [ ] `nvidia_peermem` loaded
- [ ] DART core/policy 15 tests 통과
- [ ] CPU RDMA seq 1–10 무오류
- [ ] GDR MR probe 통과
- [ ] 64KiB 및 1,415,232B GDR checksum 무오류
- [ ] actual-radio 1-endpoint smoke PASS
- [ ] output root가 과거 archive와 분리됨
- [ ] foreign GPU/container 없음 또는 명시적으로 실험에 포함됨

이 checklist가 새 노드의 operational checkpoint다.
