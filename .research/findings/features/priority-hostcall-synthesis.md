# Priority-hostcall synthesis

- Shared wire type：`gpu_protocol::Priority`（Low=0、Normal=1、High=2，legacy default Normal）。
- `HostcallMetadata` 保持 16 bytes；packet ABI 保持 32-byte header / 2112-byte stride。
- v3 generation/control CAS 把 submitted timeout 变为 device→host ownership transfer。
- READY winner由device release；FILLED/HOST_OWNED cancel winner由host final-drain/reclaim。
- Metadata采用version+stable control snapshots；release顺序为IDLE→clear→free push。
- Host queues按8H:4N:1L服务；已进入blocking syscall的Low仍不可抢占。
- 旧构造器reserve=0保留全部general credits；High reserve仅显式opt-in。
- High先用shared global reserve再fallback general；Normal/Low不消耗reserve。
- `PRIORITY_ECHO`独立记录nonce、typed task identity、priority、packet/provenance、generation与process/error。
- Quiescent pool audit检查ready empty、idle cardinality、pool masks、duplicates/missing与controls。
- Protocol 9 unit + 7 doctest；runtime debug/release39；host priority23 PASS/1 ignored。
- Final composed oracle 10/10；五mutation exact kill；timeout→reclaim→same packet generation `1→2`。
- Final PTX `fb95559e...`；composed fresh/cached、safety、obstacle、canonical trace同快照PASS。
- Trace 32/32 + 32/32只证明IO route/listener/kernel smoke，不替代priority/identity oracle。
- High reserve是shared global pool，不是per-shard isolation或hard-real-time admission。
- Shutdown/reinit仍要求GPU synchronize/no-new-producer；blocking I/O可使join无界。
- CUDA launch/sync Err后的所有mapped-memory teardown路径尚未由源码证明bounded。
- Final logs：composed `39f3d3a7.../61445c72...`、trace `4080a889...`。
