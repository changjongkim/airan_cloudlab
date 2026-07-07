# PROOF 5 + 6 — 20260622

Generated: 2026-06-28 08:36:08

## File summary
  /users/sgkim/cloudlab_aerial/results/20260622/cudafree_h1h2/p5_callchain_L1.nsys-rep (2.0M)
  /users/sgkim/cloudlab_aerial/results/20260622/cudafree_h1h2/p5_callchain_L1.sqlite (5.0M)
  /users/sgkim/cloudlab_aerial/results/20260622/cudafree_h1h2/p6_defer_L1.nsys-rep (2.0M)
  /users/sgkim/cloudlab_aerial/results/20260622/cudafree_h1h2/p6_defer_L1.sqlite (5.0M)

## L1 latencies
    [realL1] cells=20 iters=20 prbs=273 mcs=2 tb_size=13576
    [realL1] synthetic rx_slot shape=(3276, 14, 4)
    [realL1] building RX components...
    [realL1] RX components ready
    [realL1] warmup...
    [realL1] measuring...
    [realL1] p5_callchain_L1: mean=353.850ms p95=355.627ms p99=355.791ms miss1ms=20/20
    [realL1] saved: ./results/realL1_p5_callchain_L1_20260628_133326.json

## e6 shim stats

## PROOF 5 callchain check
    OSRT_CALLCHAINS
    ENUM_STACK_UNWIND_METHOD

## PROOF 6 cudaFree presence in sqlite (should be gone if shim worked)
    p6 cudaFree calls: 53
