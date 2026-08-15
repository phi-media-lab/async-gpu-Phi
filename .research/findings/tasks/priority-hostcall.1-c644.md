# priority-hostcall.1-c644：端到端 hostcall 优先级传播与准入保障

**Cycle** c644 | **Theme/Feature** priority-hostcall | **Kind** implementation + compatibility verification | **Status** complete（原 harness 存在独立 PTX 路由缺陷）

## Summary

在不改变固定 packet 总尺寸与旧 API 签名的前提下，将 executor 的共享 `Priority`、`task_id` 与 `effective_priority` 从 device hostcall packet 贯穿到 host ready path 和阻塞服务调度。主机慢服务改为同级稳定的 8H:4N:1L 分级队列；显式 opt-in 的 High packet reserve 保证 Low/Normal 占满 general pool 时仍有 High 准入 credit。旧构造器保留全部 general credits。

## Baseline

- Git baseline：`73b24a9`；Rust `1.98.0-nightly`；NVIDIA GeForce RTX 4070，driver 595.84，compute capability 8.9。
- 改动前 `cargo test -p gpu-protocol`：0 unit + 7 doctest 通过。
- 改动前 `cargo check --manifest-path crates/core/gpu-runtime/Cargo.toml`：通过。
- 改动前 `cargo test -p gpu-host --lib`：已有套件打印约 66 个通过后超过 60 秒未结束，手动终止为 exit 255；这不是本 feature 的有效 PASS/FAIL baseline。
- 工作树已有 user/其他 agent 改动；本任务未触碰 `CLAUDE.md`、`thread.rs`、`gpu-kernel-core/lib.rs`、`executor.rs`、kernel、测试 harness 或 `docs/`。

## Changes

- `gpu-protocol`：增加 `#[repr(u8)] Priority { Low=0, Normal=1, High=2 }`、16-byte `HostcallMetadata`、协议/metadata version、固定 offsets、编译期 size/alignment/header 断言和 reserve 映射 helper。
- `gpu-runtime::hostcall`：旧 request/timeout/print/trace API 委托 `Normal/task_id=0`；新增对应 `_with_metadata` API；High 先取专用 stack、耗尽后退化 general；release 按 packet 归属返回并清 metadata，防止旧式生产者复用时继承 stale priority。
- `gpu-host::hostcall`：ready Treiber 链在 dispatch 前反转以消除批次内 LIFO；慢服务进入三个稳定 `VecDeque`，按 8H:4N:1L 周期服务并在关闭时 drain。
- 旧 buffer 构造器 reserve 固定为 0；仅四个 `_with_priority_reserve` 构造器显式启用 High reserve。host/device 均提供 completion metadata 查询接口。
- 增加 ABI、wire fallback、各 shard 对 shared global High reserve 的贡献、稳定顺序、服务权重、饥饿防护、关闭 drain、legacy metadata 和实际 packet 链分离测试；不把贡献值解释为 per-shard isolation/admission。

## Verification / Evidence

```text
cargo test -p gpu-protocol
4 unit + 7 doctest: PASS

cargo check --manifest-path crates/core/gpu-runtime/Cargo.toml
cargo test --manifest-path crates/core/gpu-runtime/Cargo.toml --lib
11 passed; 0 failed: PASS

cargo clippy -p gpu-protocol -- -D warnings
cargo clippy --manifest-path crates/core/gpu-runtime/Cargo.toml -- -D warnings
cargo clippy -p gpu-host --lib -- -D warnings
PASS

cargo test -p gpu-host --lib priority_tests -- --nocapture
7 passed; 0 failed; 1 ignored; 72 filtered out

timeout 300s env AUTO_BUILD_KERNEL=0 cargo test -p gpu-host --lib \
  legacy_trace_e2e_uses_io_ptx_and_all_general_credits -- \
  --ignored --nocapture --test-threads=1
KERNEL_IO sha256=603023c75504092a8b26c2982726eac318ecb5b722b75b0c159580733c0c4ba7
phase=module_ready → phase=launch → phase=kernel_complete
phase=pass completed=32/32; 1 passed; real 0.72s: PASS

rustfmt --check（3 个负责文件）+ git diff --check
PASS
```

executor 集成侧直接 `pub use gpu_protocol::Priority`；当前 runtime 11 unit tests、nvptx check、clippy 与 core PTX 均通过。

## Compatibility Fix and Falsified Initial Hypothesis

首次实现让旧构造器默认每 shard 预留 1 个 packet。即使签名不变，这也会减少旧调用者可见的 general credits，因此按兼容原则修复为旧构造器 reserve=0；显式 reserve 的无 CUDA 单测直接验证 4 packets 被拆成 general `0→1→2` 与 High-only `3` 两条互不相交的链。

第一次 `ONLY_TEST=trace` 长时间无输出时，曾初步怀疑 32 lanes 因 reserve 从 32 降到 31 而死锁；源码复核反证了该因果：此测试实际分配 64 packets。reserve=0 后原命令仍在约 191.86 秒 SIGSEGV（exit 139），且尚未打印 `Launching...`。进一步证据显示 harness 把 `KERNEL_PTX=KERNEL_COMPUTE` 传给只存在于 `KERNEL_IO` 的 `trace_multithread_test`（compute symbol count=0，I/O count=1），丢弃 `load_ptx` Result，并在长 JIT/load 前启动 listener。内核日志 fault address 末尾 `+0x18` 对应 `BUF_OFF_SHUTDOWN=24`；IP 解析为 `atomic_load::<u32>`，即 listener 在失败清理期读取已失效的 CUDA mapped buffer。

不修改 harness 的正确路由 gate 先成功 load/resolve 当前 I/O PTX，再启动 RAII listener；同一 RTX 4070 上 32/32 trace 完成。该计数只证明正确 I/O route/JIT/kernel smoke 与 reserve=0 未阻止 kernel 结束，不能证明 hostcall 请求成功、priority、reserve provenance 或 cancellation。新 ABI/ownership 的完成证据来自后续 CPU protocol/host tests、error-visible host metrics、composed E2E 与 timeout→reclaim→same-packet-reuse gate；原 harness 的 compute→I/O 路由/错误处理/lifetime 是独立既有缺陷。

## Findings

1. **优先级必须是共享 wire type。** executor 与 hostcall 直接复用 `gpu_protocol::Priority`，避免两套枚举在 ABI 边界漂移。Confidence: 10/10。
2. **旧 API 的 credit 数量也是兼容性。** 即使函数签名不变，自动 reserve 也会减少 general 容量；因此 reservation 必须 opt-in。该原则成立，但原 trace 卡住不是它的实证。Confidence: 10/10。
3. **稳定优先级队列只能消除 queued inversion。** 8H:4N:1L 给 Normal/Low 明确的有界机会，但已经进入阻塞 syscall 的 Low 请求不能被单 worker 抢占。Confidence: 10/10。
4. **版本门控与 release 清理缺一不可。** version=0/unknown 安全回退 Normal；release 清 version 防止 legacy producer 观察到上次复用留下的 metadata。Confidence: 9/10。
5. **High reserve 是 admission guarantee，不是实时保证。** 它保证有包可提交，不保证 PCIe、OS syscall、GPU cooperative poll 或 host worker 的最坏时延。Confidence: 10/10。

## Unexpected Discoveries

- 原 trace 卡住最初被误归因为 reserve；源码中的 64-packet 配置和正确 I/O 路由的 32/32 PASS 反证了该解释。
- 原 harness 的 `KERNEL_PTX` alias 指向 compute，而 trace kernel 属于 I/O；忽略 `load_ptx` Result 又让路由错误退化为长 JIT 后的 listener 映射崩溃。
- `gpu-host` 的 build script 会因协议/host 源变化重建约 2.9 MiB PTX，定向 CPU 单测也可能需要约两分钟 cold build。
- 阶段历史：本阶段尚未处理 host 仍可能持有请求时的 timeout ownership 风险；后续 `priority-hostcall-e2e.3a` 已用 v3 generation/control CAS、host-side final reclaim 与 same-packet reuse gate 闭合该问题。
- runtime 不带 `--lib` 的全测试被既有 `std_future::block_on` doctest 缺少示例上下文拖成失败；host clippy 加 `--tests` 也会进入未启用 `nn` feature 的既有 integration tests。范围内 gate 分别使用 `cargo test --lib` 与 `cargo clippy --lib`，均通过。

## Open Questions

- 阶段历史：本阶段未跨文件修复原 `ONLY_TEST=trace` 的 PTX 路由、Result 传播与 listener 生命周期；后续 `priority-hostcall-e2e.2` 已完成显式 I/O route、错误传播与 listener RAII，final canonical trace 为 32/32 + 32/32 PASS。
- 若要让 High 越过正在执行的 Low 阻塞 syscall，需要可取消 I/O、多 worker/资源隔离或 OS 级 async I/O；当前队列不声称 preemption。
- 多 shard 只保证每个 drained stack 与进入 host queue 后的稳定顺序，不定义跨 shard 的全局提交序。

## Impact on Downstream Tasks

- executor/hostcall 现在共享一套 priority ABI，completion path 可恢复 task identity/effective priority。
- obstacle/event 场景若要验证 hostcall admission，必须显式使用 `_with_priority_reserve`，检查 `high_reserved_packets()` 与实际 packet provenance；`high_reserved_per_shard()` 只表示每 shard 的构建贡献，不承诺 per-shard isolation/admission。未启用时 High 合法退化到 general pool。
- 本阶段当时只为 priority inheritance、deadline、取消与 multi-worker QoS 提供兼容扩展位。后续 v3 已实现 timeout/cancel ownership；priority inheritance、预配置 deadline compliance 与 multi-worker QoS 仍未实现。

## Post-task closure

本文保留的是 `priority-hostcall.1` 阶段证据，不应按最终仓库状态解读。后续 `priority-hostcall-e2e.2` 闭合 canonical I/O PTX route、所有关键 `Result` 传播与 listener 生命周期；`priority-hostcall-e2e.3a` 闭合 v3 timeout/cancel ownership、generation-aware reclaim、quiescent audit 和 same-packet reuse。最终 reserve 是 shared global High pool，各 shard 仅贡献 packet；`fb95559e...` PTX 上 composed fresh/cached、Phase3 safety 与 canonical trace 均通过。High reserve 仍只是准入语义，不构成 per-shard isolation、硬实时或最坏响应时间保证。
