# Chain 2 Summary — 20260622

Generated: 2026-06-28 08:03:09

## X4 retry (chanpred alone on 3g)
  /users/sgkim/cloudlab_aerial/results/20260622/s18_ai_nsys/X4_chanpred_alone_3g_AI.nsys-rep (215M)
  /users/sgkim/cloudlab_aerial/results/20260622/s18_ai_nsys/X4_chanpred_alone_3g_AI.sqlite (705M)

## cudafree_h1h2/
  /users/sgkim/cloudlab_aerial/results/20260622/cudafree_h1h2/e10_callchain_L1.nsys-rep (1.7M)
  /users/sgkim/cloudlab_aerial/results/20260622/cudafree_h1h2/e10_callchain_L1.sqlite (3.9M)
  /users/sgkim/cloudlab_aerial/results/20260622/cudafree_h1h2/e2_defer_L1.nsys-rep (1.6M)
  /users/sgkim/cloudlab_aerial/results/20260622/cudafree_h1h2/e2_defer_L1.sqlite (3.9M)
  /users/sgkim/cloudlab_aerial/results/20260622/cudafree_h1h2/e9_baseline_L1.nsys-rep (1.7M)
  /users/sgkim/cloudlab_aerial/results/20260622/cudafree_h1h2/e9_baseline_L1.sqlite (3.9M)
  /users/sgkim/cloudlab_aerial/results/20260622/cudafree_h1h2/e9_sync_first_L1.nsys-rep (50K)
  /users/sgkim/cloudlab_aerial/results/20260622/cudafree_h1h2/e9_sync_first_L1.sqlite (256K)

## cudaFree comparison (L1 process view)

| Condition | cudaFree calls | cudaFree total ms | avg per call (µs) |
|---|---|---|---|
| e9_baseline_L1 | 2516 | 9235.1 | 3670.5 |
| e9_sync_first_L1 | 0 | 0.0 | 0.0 |
| e10_callchain_L1 | 2516 | 9194.2 | 3654.3 |
| e2_defer_L1 | 53 | 1.3 | 24.7 |

### E9 shim's internal timing (sync vs real_free split)

### E2 shim deferred-free count
    [e2_shim] FINAL skipped 0 cudaFree calls
    [e2_shim] FINAL skipped 0 cudaFree calls
    [e2_shim] FINAL skipped 0 cudaFree calls
