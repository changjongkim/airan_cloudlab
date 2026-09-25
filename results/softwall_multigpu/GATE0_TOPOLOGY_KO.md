# SoftWall multi-GPU Gate 0 — topology와 peer capability

**상태:** PASS  
**시각:** 2026-09-24 UTC  
**Slurm:** job 58815435, node `nid001016`

## 판정

현재 allocation에는 NVIDIA A100-SXM4-40GB 네 장이 있고, `nvidia-smi topo -m`은 모든
GPU pair를 `NV4`로 보고했다. CUDA runtime의 `cudaDeviceCanAccessPeer`에 해당하는 CuPy
probe도 4×4 모든 방향에서 1을 반환했다. 따라서 이 node에서 full-GPU CUDA P2P 경로를
구현할 topology 전제는 성립한다.

이 gate는 다음을 아직 증명하지 않는다.

- `cudaMemcpyPeerAsync` 실제 SoftWall payload의 무결성이나 latency
- CUDA IPC handle을 결합한 cross-process P2P
- MPS co-run 상태의 front/transport/NRx/back 전체 bound
- remote endpoint failure 뒤 all-fail recovery와 credit lifecycle

이 항목들은 [멀티 GPU·모델링 로드맵](../../docs/current/SOFTWALL_MULTIGPU_MODELING_ROADMAP_KO.md)의
G1–G4에서 새 protocol로 검증한다. 과거 single-process MIG P2P 결과를 이 경로의
service bound로 재사용하지 않는다.

## Artifact와 SHA-256

| Artifact | SHA-256 |
|---|---|
| `gpu_inventory_job58815435.csv` | `1e2d0f0f3d734152e9914e921929e620baf94ec3737ba68221bab4d180ef9226` |
| `topology_job58815435.txt` | `b2dc87ea57bc99df496ee9d53054b2365cbb6dc15769ea405289cbf6313e8ebe` |
| `topology_peer_gate_job58815435.json` | `3c467e088878ef550cbd465bce05dc9f14b2e9bc1c4d5ee46fa53ab9cb1e4c48` |

