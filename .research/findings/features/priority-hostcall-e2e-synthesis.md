# Priority-hostcall E2E synthesis

## 已落地 / Implemented
- v3 generation-tagged cancel/ownership protocol 保持 32/2112-byte packet ABI。
- Generic/timeout/print/trace timeout 通过 CAS 把 packet release responsibility 转给 host；READY winner 仍由 device 释放。
- Host fast/slow path 只处理 matching generation；stale queue write 被拒绝，cancelled packet 回到原 general shard 或 shared global High pool。
- Listener shutdown final-drain、IDLE→metadata clear→free push、stable metadata snapshot 与 tagged stack mutation 均已闭合。
- Reinit 使用 drain epoch + dispatch/queue/inflight barrier；仍要求外部 GPU synchronize/no-new-producer。
- Priority echo 独立记录 nonce、typed identity、priority、packet/shared-reserve provenance、generation、process/error。

## 最终验证 / Final verification
- Protocol 9+7、runtime debug/release 39、host priority 23+1 ignored、harness 21、oracle 10、route 6 全绿。
- Composed 正例在 4 个 general Low leases 饱和后使用 shared High packet；五种 exact mutation 与 timeout→reclaim→same-packet reuse 全部通过独立 oracle。
- Final PTX `fb95559e93eec1c...`；composed fresh/cached 与 Phase3 safety、obstacle、trace 同快照 PASS。
- Canonical trace `trace_multithread_test` 与 `trace_assert_test` 各 32/32，日志 SHA-256 `4080a889...`。
- Trace 只证明 IO route/listener/kernel smoke；High reserve、typed metadata 与 cancellation 由 composed/safety gates 证明。
- `priority-hostcall-e2e.3a-c644.md` 是 state task `priority-hostcall-e2e.3` 的 v3 ownership 子证据，不是孤立 task。

## 边界 / Boundary
- High reserve 是 shared global pool，不是 per-shard isolation。
- Blocking stdin/accept 可使 shutdown 无界；Arc+join 只防止 mapped-memory UAF。
- CUDA launch/sync 返回 Err 后的 quiescence 依赖外部 CUDA 语义，源码没有证明所有异常 teardown 都 bounded。
- 本机制没有 deadline 或 hard-real-time 保证。

## 状态 / Status
CANONICAL ROUTE + OPT-IN RESERVE + TYPED ROUND-TRIP COMPLETE。
