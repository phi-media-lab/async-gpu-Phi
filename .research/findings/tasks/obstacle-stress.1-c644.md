# obstacle-stress.1-c644：障碍事件 priority/admission/waker 真实 GPU 压力测试

**Cycle** c644 | **Theme/Feature** obstacle-stress | **Kind** implementation + falsifiable GPU experiment | **Status** complete

## Summary

在 `gpu-kernel-io` 与 `gpu-test-harness` 增加独立的障碍事件压力场景，验证 Low 饱和后的 High 准入、High first-poll dispatch bound、waker priority、Low eventual completion、有界 non-yielding 负例，以及 stale/deadline watchdog 决策输出。规格明确限定为 cooperative scheduling，不声称 hard real-time，也不控制真实机器人。

## Baseline

- Git baseline：`73b24a9`；机器：NVIDIA GeForce RTX 4070，driver 595.84，compute capability 8.9，CUDA compiler 13.3。
- `AUTO_BUILD_KERNEL=0 cargo check -p gpu-test-harness`：改动前通过。
- `AUTO_BUILD_KERNEL=0 ONLY_TEST=executor timeout 90 cargo run -p gpu-test-harness --bin gpu-tests`：失败为 `KernelNotFound("executor_demo")`。
- 路由证据：`executor_demo` 在 `kernel_io.ptx` 中出现一次，在默认 `KERNEL_PTX` 所指的 `kernel_compute.ptx` 中为零；因此新场景必须显式加载 `KERNEL_IO`。
- 工作树已含其他 agent/user 改动；本任务未触碰 `CLAUDE.md`、`thread.rs`、`gpu-kernel-core/lib.rs`、`executor.rs` 或 hostcall 核心文件。

## Changes

- `crates/kernel/gpu-kernel-io/src/obstacle_stress.rs`：真实 GPU 正例、负例、纯 fallback decision 输出与 `u64[48]` 具名 schema。
- `crates/kernel/gpu-kernel-io/src/lib.rs`：仅注册独立模块。
- `crates/test/gpu-test-harness/src/tests_obstacle.rs`：mapped executors/results、20s harness timeout、结构化输出与 gate。
- `crates/test/gpu-test-harness/src/main.rs`：新增 `ONLY_TEST=obstacle_stress|obstacle`，并仅为该场景增加 `KERNEL_IO_PTX` 路由。
- `docs/obstacle-event-priority-stress.md`：用户可读、可证伪、安全边界完整规格。

## Verification / Evidence

IO PTX 构建成功（2.9 MiB），入口路由证据：

```text
kernel_io.ptx:      obstacle_event_priority_stress count = 1
kernel_compute.ptx: obstacle_event_priority_stress count = 0
```

Host 构建无 warning：

```text
AUTO_BUILD_KERNEL=0 cargo check -p gpu-test-harness
Finished `dev` profile ...
```

RTX 4070 真实 GPU 输出：

```text
admission: Low total=224/224 (work=223), ReservedCapacity rejects=1, other rejects=0, High accepted=1
positive dispatch: inject=225 first=226 gap=1 resume=227 wake-gap=1 latency=19456ns
Low completion: 223/223; executor completed=225/225; total dispatches=450
bounded non-yielding negative: inject=1 low-return=2 High-first=3 latency=8899584ns busy-iters=4096
decisions: fresh(action=1,reason=1,use_gpu=1); stale(action=2,reason=2,use_gpu=0); deadline(action=2,reason=4,use_gpu=0)
gates: reserved=1 first_poll=1 waker=1 low_eventual=1 positive=1 no_midpoll=1 decisions=1 overall=1
PASS
```

首轮通过快照 PTX SHA-256 为 `603023c75504092a8b26c2982726eac318ecb5b722b75b0c159580733c0c4ba7`，3,038,167 bytes，mtime `2026-08-15 00:58:56 +0800`。首轮命令为：

```bash
/usr/bin/time -p timeout 300s env AUTO_BUILD_KERNEL=0 ONLY_TEST=obstacle_stress cargo run -p gpu-test-harness --bin gpu-tests
```

首轮 `real 128.99s user 128.25s sys 0.51s` 并 PASS。第一次外层 90 秒和随后所谓“缓存复跑”60.17 秒都停留在 PTX JIT/load、尚未出现 kernel phase，严格归类 UNKNOWN；跨进程 cache 未在 60 秒内命中。`%globaltimer` latency 只作为诊断值。

最终 Phase3 同快照回归使用 PTX
`fb95559e93eec1c07dbb4925c4564e43263cbd55de07a208ffa6b46b51e68531`
（4,821,968 bytes）：RTX 4070 0.60 秒、exit 0；Low `224/224`、
ReservedCapacity 1、High first/resume gap `1/1`、Low `223/223`、
executor `225/225`、全部 gates=1。机器直录日志
`.research/findings/tasks/obstacle-stress.1-phase3-final-gpu-evidence-c644.log`
SHA-256 为
`32c504e37f4dbd9a003ac402b68c55931d879e909a133c3e1de1d997b0339661`。
旧 `603023c7...` 记录保留为 early/JIT evidence，不再称 final snapshot。

交付门禁：

- `cargo fmt --all -- --check`：PASS。
- `AUTO_BUILD_KERNEL=0 cargo check -p gpu-test-harness`：PASS，无 warning。
- Harness binary 21/21、route 6/6、scoped Clippy `-D warnings`：PASS。
- IO kernel NVPTX `cargo check --release`：PASS；整 crate Clippy `-D warnings`
  仍被 18 个既有 pipeline/hybrid/example lint 阻断，未用 `allow` 掩盖。
- 官方 `AUTO_BUILD_KERNEL=0 /bin/bash scripts/ci-lint.sh`：全部 PASS。
- `git diff --check`（本任务文件）：PASS。

验证命令：

```bash
./scripts/build-kernels.sh io
AUTO_BUILD_KERNEL=0 cargo check -p gpu-test-harness
timeout 300s env AUTO_BUILD_KERNEL=0 ONLY_TEST=obstacle_stress cargo run -p gpu-test-harness --bin gpu-tests
git diff --check
```

## Findings

1. **PTX split 必须在 host route 中显式体现。** 新入口属于 IO crate；默认 compute alias 无法找到它。该结论由 baseline `KernelNotFound` 和 PTX symbol grep 直接支持。Confidence: 10/10。
2. **first-poll bound 是 dispatch bound，不是 wall-clock deadline。** 在只有 Low ready tasks 的实验前提下，High 注入后的 gap 应 ≤1；若混入 Normal，8H:4N:1L 策略理论上可出现最多 5 个 lower-priority polls。Confidence: 9/10。
3. **Priority 不能修复 non-yielding poll。** 负例在 Low poll 内入队 High 后继续有界工作，High 只能在 Low return milestone 后运行；这直接区分 cooperative priority 与 preemption。Confidence: 10/10。
4. **Fallback 必须区分数据来源。** fresh hazard 使用 GPU 结果并输出 `apply_brake_from_fresh_result`；stale/deadline 均丢弃 GPU 结果并输出 `watchdog_conservative_stop`，但 reason 不同。Confidence: 10/10。

## Unexpected Discoveries

- 现有 `run_executor_demo_test` 也从 compute PTX 加载 IO kernel；本任务按范围约束没有顺手修改该无关历史测试，只让 obstacle scenario 使用正确 route。
- 超时后 kernel 可能仍访问 mapped buffers；harness 超时路径故意不 free，避免 use-after-free，并把结果归类 UNKNOWN。
- Cold PTX JIT 在该机器上超过 90 秒，但 kernel 一旦进入后在 20 秒 harness bound 内完成；外层 timeout 与语义 gate 必须分开解释。
- 全 crate strict clippy 暴露了既有无关 lint；为了保留工作树和任务边界，本任务没有修改那些文件，而是记录并采用命令行 scoped exemptions 验证新增代码。

## Open Questions

- mixed Normal、多 warp/multi-block、GPU reset 和端到端 sensor-to-actuator deadline 不在本场景覆盖范围。

## Impact on Downstream Tasks

- 为 priority scheduler/admission/waker 提供真实 GPU 回归 gate。
- 为机器人集成提供明确的 watchdog/freshness 接口语义，但不替代系统级 safety case、WCET 分析或硬件急停。
