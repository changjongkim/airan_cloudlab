# SoftWall AI disconnect fault injection

- Protocol: `confirm13_ai_disconnect_protocol.json`
- RAN requests: 500
- Requests completed after fault detection: 499
- RAN deadline misses: 0
- Correct transport blocks: 500/500
- AI units completed before disconnect: 5
- AI fault records: 1 (`ConnectionError`)
- AI disabled after fault: true
- AI units after fault: 0
- AI budget violations: 0
- AI release crossings: 0
- RAN p99 / max: 10.239 / 14.273 ms

The worker closed the RPC connection before unit 6. The controller detected EOF during slot 0 slack, disabled all further AI admission, and continued every later RAN release. This validates fail-closed handling for a process/channel failure. It does not make an unbounded GPU kernel preemptible; admitted work must still obey the bounded-unit contract.
