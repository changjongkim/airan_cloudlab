# Novelty audit — MIG/MPS/P2P/NIC-GDR for AI-RAN

검색일: 2026-08-13

## 판정

**조건부로 논문화 가치가 있다.** 하지만 “MIG가 MPS보다 격리가 좋다”, “GPUDirect가 host copy보다 빠르다”, “남는 GPU에서 LLM을 돌린다”만으로는 노벨하지 않다. 공개 문헌과 vendor 문서에 이미 존재하는 개별 사실이기 때문이다.

이번 연구에서 차별화 가능한 부분은 다음의 결합이다.

1. 실제 cuPHY→NeuralRx dependency를 가진 pipeline에서 same-partition과 cross-partition을 비교한다.
2. MIG, MPS, MIG+MPS, same-GPU MIG P2P, NIC GPUDirect RDMA를 동일한 workload contract와 GPU budget 아래 측정한다.
3. L1 latency만 보지 않고 NRx service rate/open-loop queue stability와 background AI utility를 동시에 평가한다.
4. Cross placement가 L1을 격리하면서도 NRx slice 축소로 전체 pipeline은 악화될 수 있다는 counter-intuitive resource-allocation 결과를 보인다.
5. 이를 deadline-constrained, communication-aware AI-RAN placement/admission-control 문제로 일반화한다.

이 exact matrix를 다루는 공개 연구는 이번 검색 범위에서 찾지 못했다. 이는 “세계 최초”의 증명은 아니므로 논문에서는 first claim을 피하고, related-work table로 재검증해야 한다.

## 이미 알려진 부분

- AI-RAN Alliance는 AI와 RAN이 공통 accelerated infrastructure를 공유하고 utilization/monetization을 높이는 AI-and-RAN 방향을 명시한다. 따라서 co-tenancy 자체는 새로운 문제가 아니다.
- NVIDIA는 AI-RAN에서 MIG로 RAN과 AI workload를 partition하는 구성을 공개적으로 제안한다.
- NVIDIA MIG 문서상 R570+에서 same-GPU MIG P2P가 지원되고, GPU instance에서 GPUDirect RDMA도 지원된다. CUDA IPC는 서로 다른 GPU instance 간 지원되지 않으며 MPS-on-MIG도 지원된다.
- 일반적인 MIG vs MPS 성능·격리 비교와 AI-RAN용 dynamic MIG orchestration 연구가 이미 존재한다.
- Neural receiver를 TensorRT로 최적화해 A100에서 sub-ms 수준으로 실행한 선행 연구가 있으므로, NRx 자체의 1 ms급 추론도 독립 novelty로 주장하면 안 된다.

## 관련 자료

- [AI-RAN Alliance formation](https://ai-ran.org/press-releases/alliance-formation)
- [AI-and-RAN working group](https://ai-ran.org/working-groups/ai-and-ran)
- [AI-RAN Alliance white paper](https://ai-ran.org/wp-content/uploads/2024/12/AI-RAN_Alliance_Whitepaper.pdf)
- [NVIDIA: Bringing AI-RAN to a Telco Near You](https://developer.nvidia.com/blog/bringing-ai-ran-to-a-telco-near-you/)
- [NVIDIA MIG deployment considerations](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/deployment-considerations.html)
- [NVIDIA MPS on MIG setup](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/getting-started-with-mig.html)
- [Characterizing NVIDIA MPS and MIG](https://arxiv.org/abs/2604.22430)
- [Dynamic MIG orchestration for AI-RAN](https://arxiv.org/abs/2503.07420)
- [AI-RAN resource allocation with MIG](https://arxiv.org/abs/2507.09124)
- [Hardware MIG AI-RAN prototype](https://arxiv.org/abs/2601.16565)
- [TensorRT neural receiver optimization](https://arxiv.org/abs/2409.02912)

## 권장 논문 framing

제목 방향:

> Communication- and Queue-Aware Placement of Dependency-Coupled RAN and AI Workloads on Partitioned GPUs

핵심 research question:

> L1 isolation을 위해 dependency-coupled NRx를 분리할 때, 어떤 MIG geometry와 transport가 RAN deadline, NRx queue stability, background AI utility의 Pareto frontier를 만드는가?

최소 contribution set:

1. dependency-aware placement taxonomy와 공정한 실험 방법론
2. wrapper artifact를 제거한 stage-level characterization
3. P2P/GDR를 포함한 real-hardware transport/placement matrix
4. replica 및 open-loop arrival sweep을 통한 stability boundary
5. 측정 기반 admission/placement policy 또는 offline optimizer

마지막 5번이 구현되면 단순 measurement paper보다 훨씬 강한 systems paper가 된다.

