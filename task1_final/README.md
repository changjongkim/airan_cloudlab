# AI-RAN Task 1 · Task 2 final results

실험 노드: CloudLab Wisconsin d8545 · NVIDIA A100-SXM4-40GB · ConnectX-6 Dx RoCE v2 PHY loopback.

## End-to-end results

L1 latency는 20 warm-up 이후 30 slots, Qwen은 동시 실행 조건의 처리량이다. `config4`와 `config7`은 같은 cross-partition 배치의 독립 반복이다.

| Config | Isolation · transport | Mean (ms) | p95 | p99 | Qwen it/s |
|---|---|---:|---:|---:|---:|
| config5 | cuPHY only | 38.958 | 39.368 | 39.416 | — |
| config6 | cuPHY+NRx only | 108.128 | 108.500 | 108.604 | — |
| config1 | MIG 4g(L1+NRx) + 3g(Qwen) | 105.725 | 106.041 | 106.127 | 10.23 |
| config3 | MIG repeat | 105.859 | 106.109 | 106.476 | 10.23 |
| config2 · 30% | Full-GPU MPS | 205.327 | 210.017 | 210.109 | 8.05 |
| config2 · 50% | Full-GPU MPS | 207.511 | 210.335 | 214.291 | 11.29 |
| config2 · 70% | Full-GPU MPS | 218.211 | 221.064 | 221.323 | 17.38 |
| config2 · 100% | Full-GPU MPS | 291.961 | 393.196 | 395.608 | 21.26 |
| config4 | MIG 2g↔2g · shared memory | 111.865 | 112.808 | 113.119 | 10.23 |
| config7 | MIG 2g↔2g · shared memory | 109.686 | 110.630 | 110.827 | 10.23 |
| config4_rdma | MIG 2g↔2g · CPU-buffer RDMA | 111.097 | 111.586 | 111.664 | 10.22 |
| config7_rdma | MIG 2g↔2g · CPU-buffer RDMA | 111.375 | 111.694 | 111.802 | 10.23 |
| config4_gdr | MIG 2g↔2g · GDR staging | 109.687 | 109.861 | 110.223 | 10.23 |
| config7_gdr | MIG 2g↔2g · GDR staging | 109.526 | 109.778 | 110.349 | 10.23 |

Transport별 두 반복 mean 평균:

| Cross-partition transport | Mean average (ms) | vs shared memory |
|---|---:|---:|
| Shared memory | 110.776 | baseline |
| CPU-buffer RDMA | 111.236 | +0.461 ms |
| GPUDirect RDMA staging | 109.607 | -1.169 ms |

GDR staging은 CPU-buffer RDMA 평균보다 1.630 ms 낮았다. Paired 결과는 config4에서 SHM 대비 -2.178 ms, config7에서 -0.160 ms이므로 개선 폭에는 반복 변동이 있다. Cross-partition과 4g monolithic의 차이는 MIG resource split과 transport가 함께 바뀌어 순수 IPC 비용으로 분리할 수 없다.

동일한 Qwen 처리량 약 10.2 it/s에서 monolithic MIG는 약 106 ms를 유지했지만 Full-GPU MPS는 약 205–208 ms였다. 가장 강한 결론은 RDMA transport 자체보다 MIG의 resource isolation 효과다.

## Transport gates

| Test | Payload | Cold seq 1 | Steady mean, seq 2–10 |
|---|---:|---:|---:|
| CPU registered-memory RDMA WRITE | 1 MiB | 5995.51 µs | 107.35 µs |
| GDR GPU-memory WRITE | 64 KiB | 122.26 µs | 61.38 µs |
| GDR forward, 4g→3g | 1,415,232 B | 736.45 µs | 758.44 µs |
| GDR backward, 3g→4g | 1,257,984 B | 662.57 µs | 665.29 µs |

모든 GDR test는 consumer checksum과 sequence 1–10을 검증했다. CPU 1 MiB 수치는 local `MR.write()`를 측정 구간 밖에서 수행한 순수 WRITE latency인 반면, GDR 수치는 GPU payload와 sequence publish를 포함한다. 따라서 두 수치를 직접적인 host-copy 절감량으로 비교하지 않는다.

GDR pipeline은 registered GPU staging buffer를 사용해 CPU payload bounce를 제거한다. 다만 공개 `TrtEngine.run()` 내부 input/output copy와 NRx output→registered backward staging D2D copy가 남아 있으므로 end-to-end 완전 zero-copy 구현은 아니다.

## Artifacts

- Raw summary and logs: [`chain/`](chain/)
- Transport log validator and CSV: [`analysis/`](analysis/)
- Figures: [`figures/`](figures/)
- Reproducible node scripts: [`../scripts_for_node/task1/`](../scripts_for_node/task1/)

RDMA 컨테이너에는 `--network=host`가 필수다. Docker default network namespace에는 RoCE backing netdev가 없어 RC QP `INIT → RTR`가 `ENODEV`로 실패한다.
