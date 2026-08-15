# priority-hostcall-e2e.2-c644：canonical harness PTX route / error / lifetime closure

**Cycle** c644 | **Feature** `priority-hostcall-e2e` | **Kind** experiment | **Status** complete

## Summary

`gpu-test-harness` 不再把所有 kernel 隐式视为 compute PTX：全部 `ONLY_TEST` selector 都有显式 route plan，易混淆的入口有可执行的 module→symbol 表和纯文本单测。Trace/std-future 等关键路径先 load+resolve，再分配/启动 listener；`HostcallListener` 以 RAII 保证正常与提前 `?` 返回均 shutdown+join。Canonical `ONLY_TEST=trace` 已在 RTX 4070 上从原 exit139 变为两个 kernel 各 32/32 PASS，obstacle 路由回归亦全绿。

最终 Phase3 同快照 `fb95559e...` 再次验证 canonical trace 0.95 秒
32/32 + 32/32 PASS；该结果只作为 route/listener/kernel smoke，不替代 composed
High-reserve 或 typed cancellation gates。

## Baseline

- Git baseline：`73b24a9`；工作树中 executor/protocol/hostcall/kernel 已有其他 agent 的未提交改动，本任务未改这些路径。
- 设备：NVIDIA GeForce RTX 4070，driver 595.84，compute capability 8.9。
- IO PTX：SHA-256 `603023c75504092a8b26c2982726eac318ecb5b722b75b0c159580733c0c4ba7`，3,038,167 bytes，mtime `2026-08-15 00:58:56 +0800`。
- 已有 baseline（同一机器/同一 PTX）：canonical `AUTO_BUILD_KERNEL=0 ONLY_TEST=trace cargo run...` 约 191.86s 后 SIGSEGV/exit139，且未打印 `Launching...`。
- 独立对照 gate：正确 `KERNEL_IO`、先 load 后 listener 的 ignored gpu-host test 在 0.72s 内 32/32、1/1 PASS；它把原 exit139 定位到 route/listener lifecycle 路径，但该 counter 吞掉 device hostcall Result，不能单独证明 ABI 成功。ABI/ownership 结论另由 protocol/host tests、error-visible metrics 与后续 composed/reuse gates支持。

## Changes

- 删除 harness 内单一 `KERNEL_PTX -> KERNEL_COMPUTE` 依赖，改为 `KERNEL_CORE_PTX / KERNEL_COMPUTE_PTX / KERNEL_IO_PTX / KERNEL_TEST_PTX` 明确归属。
- `kernel_routes.rs`：记录全部 70 个 `ONLY_TEST` alias 的 module/standalone/NVRTC/auto-discovery/host-only plan；对 trace、std_future、warp_e2e、rustc_async、executor、channel、obstacle、compute 等给出 module+symbol route。
- `load_kernel()` 在 CUDA JIT 前检查 `.visible .entry`，传播 `load_ptx` 错误，并在返回前逐 symbol resolve；关键路径在资源/listener 创建前调用它。
- 全 harness 的 `let _ = dev.load_ptx(...)` 已清零；98 处剩余调用统一改为 `...?`，不再用二次 `KernelNotFound` 掩盖原始 load/JIT 错误。
- `harness_support.rs`：`HostcallListener` 在 `finish` 和 `Drop` 中均 signal shutdown 并 join；trace、std-future 与 benchmark manual-listener 路径已接入。无 GPU fault-injection 单测证明提前错误会恰好一次 shutdown+join，显式 finish 会传播 listener panic。Warp E2E 使用已有 RAII `HostcallSession`，但也改为先 load/resolve 后 start session。
- 混合文件逐调用校正：scaling compute 为 COMPUTE，hybrid warp 为 IO，autonomous pipeline 为 TEST，`mt_malloc` 明确为 CORE，thread spawn 明确为 TEST。
- 当当前 embedded compute PTX 没有 `sm_80` entries 时，相关 `ONLY_TEST` 在 JIT/资源创建前返回明确 `ptx_feature_gate` 错误，不再混淆为 route 缺陷。
- Patch review 找到并已修复三处收束缺陷：`fusion_bench` 恢复 COMPUTE ownership、disabled `#[cfg]` selector 不再静默运行全套、`splitk` gate 纳入其实际调用的 `mma_diag`。

## Verification / Evidence

```text
AUTO_BUILD_KERNEL=0 cargo test -p gpu-test-harness kernel_routes -- --nocapture
6 passed; 0 failed; no warnings

AUTO_BUILD_KERNEL=0 cargo test -p gpu-test-harness harness_support -- --nocapture
2 passed; 0 failed; early-error shutdown/join + listener-panic fault injection PASS

AUTO_BUILD_KERNEL=0 cargo check -p gpu-test-harness --bin gpu-tests
PASS

cargo +stable fmt --manifest-path crates/test/gpu-test-harness/Cargo.toml -- --check
PASS

AUTO_BUILD_KERNEL=0 cargo +stable clippy -p gpu-test-harness --bin gpu-tests -- -D warnings
PASS

/usr/bin/time -p timeout 300s env AUTO_BUILD_KERNEL=0 ONLY_TEST=trace \
  cargo run -p gpu-test-harness --bin gpu-tests
route plan=io; trace_multithread 32/32 PASS; trace_assert 32/32 PASS;
exit 0; real 4.17s (includes 3.27s incremental host build)

/usr/bin/time -p timeout 300s env AUTO_BUILD_KERNEL=0 ONLY_TEST=obstacle_stress \
  cargo run -p gpu-test-harness --bin gpu-tests
Low 224/224; High 225→226→227; 223/223 Low; 225/225 terminal;
450 dispatches; all gates=1; exit 0; real 3.89s
final cached regression: exit 0; real 0.48s
```

Final same-PTX evidence：

- PTX SHA-256
  `fb95559e93eec1c07dbb4925c4564e43263cbd55de07a208ffa6b46b51e68531`，
  4,821,968 bytes；
- `priority-hostcall-e2e.2-phase3-final-trace-evidence-c644.log`，SHA-256
  `4080a889df2b5712da19d65acaac4599dc5219849051670bd39514cfec87944a`；
- route 6/6、harness binary 21/21、scoped Clippy 与官方 `scripts/ci-lint.sh`
  全部 PASS；
- 旧 `603023c7...`/4.17s 证据保留为 route-fix 初始快照，不再称 final。

## Findings

### Q1. 原 exit139 是 hostcall priority ABI 回归吗？

不是。Trace symbols 只在 IO PTX，但 canonical harness 加载 compute PTX；忽略 load Result 后提前启动的 listener 在失败清理期读取已失效 mapped buffer，导致 shutdown offset fault。同 PTX 的正确 route 对照与修复后 canonical 均 32/32 PASS。Confidence: 10/10。

### Q2. 只修 `trace` 的常量是否足够？

不足。静态审计发现旧 harness 132 个 load site 中 112 个丢弃 Result，单一 compute alias 造成 52 site/55 symbol-pair 误路由。逐文件简单替换也不安全：`tests_scaling.rs`、`tests_warp.rs`、`tests_pipeline.rs`、`tests_std.rs`、`main.rs` 都是混合 owner。Confidence: 10/10。

### Q3. 当前 absent symbol 是否也是 route 错误？

不是。19 site/24 pair 全部归于 `sm_80` feature 配置漂移：源码仍存在，但 `gpu-host/build.rs` 构建 compute kernel 时未启用 `sm_80`。本任务没有改 kernel/build feature，只把该状态变为可观察的 Unsupported Feature 风格错误。Confidence: 10/10。

### Q4. 这些测试是否证明 hard-real-time？

否。本任务证明的是 module admission/load/resolve 与 listener lifetime correctness；4.17s 包含 host build/JIT，也不是任务响应时上界。Obstacle 仍是 cooperative poll-boundary 语义。Confidence: 10/10。

## Unexpected Discoveries

- `KERNEL_PTX` 误用不仅影响 trace；executor/channel/std-future/warp selector 等长期依赖了“忽略 load error 后再查 symbol”的脆弱模式。
- `gpu-kernel-*` 间会因 dependency 重复一些 infrastructure entries；自动搜索可用于用户 API，但 regression harness 应给它们一个 canonical owner。
- Clippy 首轮仅暴露两个既有 formatting borrow，已做无语义变化修正后 `-D warnings` 通过。

## Open Questions

- `sm_80` kernels 应由 build script 始终启用，还是由 harness 按 capability 显式构建/跳过？这是独立 build-feature 任务。
- 本任务为 trace/std-future/throughput/scalability/file-I/O manual listener 提供 RAII；其他历史 manual-listener 测试可在 codebase-health 中继续迁移到同一 helper。
- 默认“无 `ONLY_TEST` 的全量 suite”仍会触达 21 个未嵌入的 `sm_80` symbol；这需要独立修正 `gpu-host/build.rs` feature 策略或显式跳过，不能把定向 trace/obstacle 绿灯外推成全量绿灯。
- 部分错误路径仍会泄漏 raw mapped allocation（RAII 已消除 listener UAF，但未封装所有原始分配）；建议另建 mapped-memory guard 任务。
- `HostcallSession::shutdown()` 仍丢弃 listener join panic；其 `Drop` 能防 UAF，但若要将 panic 作为 gate failure，需要新的 fallible finish API。

## Impact on Downstream Tasks

- `priority-hostcall-e2e.3` 已在显式 IO route 和安全 listener 上完成 typed metadata/completion round-trip。
- `typed-priority.2` 与 multi-block safety tests 已直接重用 `KernelModule`/`load_kernel`；
  新 symbol 若落在错误 PTX 仍会在 listener/launch 前 fail fast。
- Obstacle stress 的 IO route 保持不变，真 GPU 回归确认 executor 证据未受影响。
