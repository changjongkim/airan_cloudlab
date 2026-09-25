# Same-request CUDA-IPC timeout 탐색 결과

이 자료는 Confirm32 persistent clean gate와 별개인 fault 탐색이다. 자연 채널 `−8.5 dB`, P60/D35에서 동일 channel seed의 conventional baseline, 정상 cap80 CUDA-IPC endpoint, 첫 평가 요청에 30 ms 응답 지연을 넣은 timeout 조건을 비교했다.

## 유효한 두 round

| 조건 | releases | correct | miss | NRx commit | fallback | 자연 recovery |
|---|---:|---:|---:|---:|---:|---:|
| conventional | 2,000 | 756 | 0 | 0 | 2,000 | — |
| 정상 IPC endpoint | 2,000 | 1,979 | 0 | 1,971 | 29 | 8 |
| 첫 요청 timeout 후 conventional-only | 2,000 | 778 | 2 | 0 | 2,000 | — |

정상 endpoint는 별도 MPS process와 request-specific CUDA IPC를 포함하면서 두 round 모두 miss 0이었다. Response p99는 5.330/5.544 ms였고, NeuralRx가 실패한 요청 중 8건을 conventional이 자연 복구했다.

Timeout은 5 ms로 설정했고 두 round 모두 첫 요청에서 정확히 한 번 발생했다. Late NRx는 commit되지 않았으며 이후 모든 요청은 conventional로 처리됐다. 그러나 fault 요청의 conventional GPU event가 각각 58.907 ms와 33.647 ms로 늘어나 response 65.474/40.056 ms, D35 miss 1건씩을 만들었다. 따라서 stale-result fencing과 이후 fail-closed 전환은 맞아도 **현재 fault 요청의 deadline을 MPS+fallback만으로 보장할 수 없다.**

초기 preflight의 20 ms timeout도 conventional 19.597 ms와 합쳐 40.881 ms miss를 만들었다. 5 ms로 cutoff를 앞당겨도 위 물리 tail이 반복됐으므로 단순 guard 조정만의 문제가 아니다.

## 제외한 실행

세 번째 round 시작 전 수동 MPS `quit→start`에서 이전 server shutdown backoff가 완전히 끝나기 전에 새 control plane을 시작했다. Baseline 뒤 endpoint client가 `cudaErrorMpsConnectionFailed`로 실패했으므로 이 round는 radio/timing 결과에 포함하지 않았다. 이후 runtime helper는 MPS server process가 완전히 종료될 때까지 최대 60초 기다리고, 남은 server가 있으면 새 control plane 시작을 거부하도록 수정했다.

이 탐색은 arbitrary endpoint fault에 대한 hard guarantee 실패를 기록한다. 정상 bounded endpoint의 long-run gate는 `confirm32_same_request_persistent_ipc.*`와 `confirm33_same_request_natural_s2.*`에서 별도로 판정한다.
