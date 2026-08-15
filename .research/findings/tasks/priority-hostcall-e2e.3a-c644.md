# priority-hostcall-e2e.3a-c644：timeout ownership、packet generation 与安全 reinit

**Cycle** c644 | **Feature** `priority-hostcall-e2e` | **Kind** P0 correctness | **Status** complete（mapped to state task `priority-hostcall-e2e.3`）

> Tracking mapping：文件后缀 `.3a` 表示 `priority-hostcall-e2e.3` 的 v3
> ownership 子阶段证据，不是独立/孤儿 task；最终 typed composed round-trip 证据
> 记录在 `typed-priority.2-c644.md` 与 Phase3 final logs。

## Summary

修复了 hostcall timeout 的 packet ABA/串包根因：device 不再在 host 仍 ready、queued 或 in-flight 时把 packet 放回 free stack，而是通过 generation-tagged control CAS 将取消和归还责任原子转给 host。Host 的 fast/slow completion 与 device cancel 竞争同一个 control word，只有一个赢家；慢队列携带 generation、service、metadata 和原 pool 快照，写回前验证。`reinit_packets` 增加 listener drain epoch、local dispatch、queued/inflight barrier，并明确仍需 GPU idle/no-new-submit 外部前提。

## Baseline

- 改动前通用 request timeout 清 metadata 后立即 `hc_release_index`；PRINT/TRACE timeout 则永久保留 packet。两者分别可能造成复用后 stale host write，或无界容量泄漏。
- `IoRequest` 只有 `pkt_idx/service/metadata`，无法识别同一 index 的后续请求。
- `reinit_packets(&self)` 在 listener 可能持有 local drained chain、I/O queue 或阻塞 handler 时直接重建 stack/packet/sideband。
- free/high pop 把 packet 中缓存的旧 tagged `next` 原样 CAS 到 head，tag 可回退；host ready `swap(null_tagged())` 也把 tag 重置为 0。
- 旧 trace 32/32 kernel 会吞掉 `gpu_trace` 错误后无条件计数；它只证明 route/JIT/kernel smoke，不是 hostcall 成功证据。

## Changes

- Wire ABI 保持 buffer header 64 B、packet header 32 B、packet stride 2112 B。Buffer protocol v3；metadata v1 保留 priority-only compatibility，v2 表示 cancellable packet。
- Control 低 5 位为 `READY/ERROR/FILLED/HOST_OWNED/CANCELLED`，高 27 位与 metadata flags 的 16 位组成 43-bit request generation。FREE 只清 state，保留 generation；下一次独占 pop 后递增。
- Device 发布 `(gen,FILLED)`；host CAS 为 `(gen,HOST_OWNED)`。Timeout 只能 CAS `FILLED/HOST_OWNED -> CANCELLED`，成功后立即返回且不再碰 packet。Host completion 只能 CAS `HOST_OWNED -> READY[|ERROR]`；若看到同 generation CANCELLED，则 host 清 metadata 并归还。
- v0/v1 device packet 仍走 legacy host path并由 device 归还。新 runtime 对 v0/v2 buffer 的有限 timeout 在提交前返回 `ERR_UNSUPPORTED`；旧协议不可能同时提供 bounded return、无泄漏与安全 ownership transfer。
- Metadata writer 先写 priority/generation/task，最后写 version，并由 control release 发布；host public `packet_metadata` 用 control 双 acquire 快照，FREE/变化中返回 `None`。
- 所有 device pop/push、host free push/ready drain 都由 `advance_tagged_head` 推进 32-bit tag；cached `next` 只提供新 index，不再恢复旧 tag。Device push 在 head CAS 前显式 `membar.sys`。
- `IoRequest` 保存 generation/service/metadata/pool provenance；dequeue/handler 前和 completion 前校验。取消 packet 按 index 归还原 general shard或一个 shared global High pool。`high_reserved_packets()` 明确 shared total；`*_per_shard` 只是每 shard contribution，不承诺 per-shard admission。
- Reinit 发起 host freeze epoch；listener 强制 drain ready/local chain 后 ack并停读，worker 排空 queue/inflight 后才重建。`try_reinit_packets` 可返回具体 Busy 计数；阻塞 stdin/accept 仍可能使等待式 reinit/shutdown 无界，保留 join+Arc 以避免 mapped-memory UAF。
- 新增 error-visible `processing_metrics()`；host 端测试可观察 completed/error/cancel/stale，不再用无条件 GPU completion count 代替 hostcall 成功。

## Verification / Evidence

```text
cargo test -p gpu-protocol --lib
6 passed (ABI/version/generation + T1/T2 short-ABA model)

cargo test --manifest-path crates/core/gpu-runtime/Cargo.toml --lib
11 passed

cargo test -p gpu-host --lib priority_tests -- --nocapture
15 passed; 1 ignored (cancel-before-dequeue, cancel-inflight, completion race,
reuse/no-stale-write, shared-High return, reinit Busy, stable metadata, metrics)

cargo clippy -p gpu-protocol -- -D warnings
cargo clippy --manifest-path crates/core/gpu-runtime/Cargo.toml --lib -- -D warnings
cargo clippy -p gpu-host --lib -- -D warnings
PASS

cd crates/core/gpu-runtime && cargo check
NVPTX PASS (only repository ptx78 unstable warning)

cd crates/kernel/gpu-kernel-io && cargo build --release
PASS; PTX sha256 3902c4ad16d8eb0efbfda02fd172cdd63e5478719a2d667724cb9b889df3ea6f
size 3,308,685; atom.cas.sys.global.b32=175; b64=433;
st.release.sys.global.u32=429; ld.acquire.sys.global.u32=611; membar.sys=239

cargo test -p gpu-host --lib -- --test-threads=1
120s outer timeout in pre-existing scheduler GPU long test; 80+ tests shown PASS,
no failure before timeout, matching the documented full-lib baseline behavior.
```

Final closure：protocol 9+7、runtime debug/release 39、host priority 23+1、
composed oracle 10、typed matrix 13 与官方 CI lint 全绿。RTX 4070 final PTX
`fb95559e...` 的 composed fresh/cached PASS；timeout→host reclaim→同一
reserved packet generation `1→2` reuse，无 stale write。证据日志 SHA-256 为
`39f3d3a7...` / `61445c72...`，Phase3 safety 为 `8141df0c...`。
旧 `3902c4ad...` 是 v3 implementation baseline，不再称最终 GPU artifact。

Post-evidence source delta：final GPU/trace 证据完成后，completion audit 仅在
`crates/core/gpu-host/src/hostcall.rs` 的 `#[cfg(test)] mod tests`（起始行 3547）
删除 6 个冗余 `as u64`；production handler/listener、wire/control 与 GPU PTX 均未改。
变更前/后 source SHA-256 分别为 `ed37914088a09348d3ae69d379f8da079560cb10aca3a27502c50bba86ed4b58`
与 `4a5d0fd2d1d9f6b61c2770a7af410bb0f69321b123394549245e68bdb2acc5f6`。
精确六处 hunk 与作用域证据见
`priority-hostcall-e2e.3a-post-evidence-host-test-only-delta-c644.log`；因此无需把
该 test-only Clippy 清理误判为生产 handler 变更，也不以它覆盖原 GPU transcript。

## Findings

1. **Timeout 必须是 ownership transfer，不是 error branch。** 只要 host 可能持有 index，device release 就会允许下一请求复用同一 payload；generation 本身不能挽救已经发生的 stale write。CAS winner 加 host-side release 才闭合生命周期。Confidence: 10/10。
2. **Generation 必须与 control state 一起验证。** 仅给 `IoRequest` 加 generation 而不在写前检查 control/metadata，仍会让过期 worker 改写新请求。当前 pre-handler identity check + completion CAS 同时覆盖 queued ABA 和 cancel race。Confidence: 10/10。
3. **Stack tag 必须在每次 head mutation 单调推进。** Pop 安装 cached `next` 或 drain 写固定 null 都会回退 tag；T1 snapshot 与 T2 pop/push 可以在很短交错内形成 ABA。Confidence: 10/10。
4. **Host quiescence 不等于 GPU quiescence。** Drain epoch证明 host 不再持有 local/queued/inflight packet，但不能冻结任意 persistent producer；调用者仍必须 synchronize 并阻止新 launch/submit。Confidence: 10/10。
5. **兼容旧 wire 的安全边界必须显式。** New host 能处理旧 producer；但 old host 不认识 CANCELLED，也不会代 device 归还，因此 new device 的有限 timeout 只能 pre-submit 拒绝，不能假装安全降级。Confidence: 10/10。
6. **High reserve 是 shared global pool。** 每 shard 贡献 tail packet 只决定构建来源，不构成每-shard隔离或 admission 保证。Confidence: 10/10。

## Unexpected Discoveries

- 旧 priority E2E 的 32/32 trace counter 与 hostcall Result 无关；已降级为 module route/kernel smoke。新增 host-side metrics 和 metadata tests 才提供错误可见证据。
- Blocking stdin/accept 无通用安全强制取消；detach worker 会把无界 join 变成 mapped-pointer UAF。当前选择显式记录无界边界并保留 Arc+join。
- PTX 中 generation CAS、release/acquire 与新增 push fence 均实际 lower；不是只在 Rust `Ordering` 层成立。

## Open Questions

- 真 GPU composed test 已传播 timeout/error，并同时断言 host
  `processing_metrics`、packet pool cardinality、metadata generation 与同 packet reuse。
- 若需要 bounded teardown/packet return，stdin/accept/read 需改成 OS 可取消 I/O、超时 socket 或独立资源 worker；当前协议只保证 handler 返回后的安全归还。
- 43-bit generation wrap 需要同一 packet 超过 8.8e12 次复用；若未来需要形式上的不回绕身份，应扩 ABI 或引入独立 generation table。

## Impact on Downstream Tasks

- Priority/typed completion 可安全把 task metadata 穿过 host queue；timeout 不会唤醒错误 generation 的 task。
- Composed E2E 可直接使用 `processing_metrics()` 区分真实 host completion/error/cancel，不能再以 kernel completion counter 代替。
- Persistent pipeline 可继续用等待式 `reinit_packets`；低延迟控制面应优先用 `try_reinit_packets` 并对 Busy 做显式策略。
