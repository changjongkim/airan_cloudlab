# SoftWall stale AI completion fault injection

- Protocol: `confirm14_ai_stale_completion_protocol.json`
- Injected response delay: 20 ms after NeuralRx GPU completion
- Controller RPC timeout: 7 ms
- RAN requests: 500
- Requests completed after timeout detection: 499
- RAN deadline misses: 0
- Correct transport blocks: 500/500
- AI responses accepted: 0
- AI fault records: 1 (`TimeoutError`)
- AI disabled after timeout: true
- AI units after timeout: 0
- RAN p99 / max: 10.680 / 14.544 ms

The controller rejected the late response, closed the AI channel, and continued every subsequent RAN release. The injected delay started after GPU completion, so this validates timeout and stale-response handling rather than cancellation of a running GPU kernel.
