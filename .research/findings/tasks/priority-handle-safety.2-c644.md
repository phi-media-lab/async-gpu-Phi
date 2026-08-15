# priority-handle-safety.2：executor authority、waker 与 typed priority 安全闭环
**Cycle**: c644 | **Kind**: implementation/verification | **Status**: done；host/model/NVPTX + final RTX 4070 lifetime independently verified

## Summary

本轮把可复用 slot 从公开 authority 中剥离：私有队列键 `LocalTaskKey(u64)` 只编码 slot/generation，公开 `TaskKey` 是 `owner + incarnation + local` 三个 `u64`，所有公开 getter/setter 先验证 executor 与初始化 epoch，再访问 slot。稳定的 per-slot `WakerData` 实现 RawWaker clone/wake/wake_by_ref/drop 引用所有权；terminal slot 直到全部 waker/typed-context refs 释放后才回到 4×`u64` free bitmap。引用计数达到 `u32::MAX` 后永久 pin，宁可泄漏也不 wrap 后 UAF。

队列状态机同时修复了 priority-lock 下丢失唯一 membership 的两个竞态：dequeue 已消费 entry 后等待同 generation 的瞬时 lock；wake/requeue 的 winner 在 enqueue 成功或稳定转入 `ENQUEUE_RETRY` 前不放弃 publish obligation。`ENQUEUE_RETRY` 即使仍有 retained waker 也会在容量恢复后重试。Typed task 在 publish QUEUED 前固定 runtime priority；异值 setter 返回 `PriorityFixed`，从而保持类型标记与 runtime/wire 一致。

## Scope and compatibility

- 实现文件：`crates/core/gpu-runtime/src/executor.rs`、新 `src/priority.rs`，以及 `src/lib.rs`/`src/prelude.rs` 的局部导出。
- 验证文件：`tests/typed_priority/*.rs` 与 `tests/typed_priority_runner.sh`。
- 保留 `spawn`、`spawn_with_priority`、`spawn_with_options`、`spawn_typed`、三档 admission reserve 与 8:4:1 dispatch cycle。
- `TaskId` 保留为 `TaskKey` alias；`slot()`/`generation()` 仅供诊断，slot-only tuple 构造不再存在。
- 未修改 thread、hostcall、protocol、kernel、harness、`CLAUDE.md` 或 `.research/state.toml`；共享树中这些文件的 dirty change 属于其他任务。

## Findings

### Q: stale waker / stale key 能否命中新 generation、另一个 executor 或 re-init 后实例？
**A — PASS（高置信）**：不能通过支持的 API 命中。公开 `TaskKey` 同时绑定 device-side executor 地址、module/process-lifetime 非回绕 incarnation、以及 slot generation；validate 在读 slot 前依次检查 owner/incarnation/slot。队列只接受私有 `LocalTaskKey`。旧 key 跨 executor 返回 `WrongExecutor`，跨 re-init 返回 `WrongIncarnation`，旧 waker 的 generation 不匹配时不产生 wake。incarnation 分配到 `u64::MAX` 后永久失败，不回到旧 epoch。

边界：owner 是 device pointer identity，因此 `init` 必须在 device 上、且 executor 从首次 spawn/run 到 shutdown、warp join 与 device synchronization 完成之前地址稳定。此 incarnation 域不承诺跨 PTX module unload/reload、跨 device/context 全局唯一；需要这些域的 host correlation 时，caller 必须提供外部唯一 launch namespace。

### Q: RawWaker 的所有权和 terminal reclaim 是否闭合？
**A — PASS（高置信）**：`WakerData` 在 slot 生命周期内地址稳定；clone 增 ref，consuming wake 在动作后减 ref，wake_by_ref 借用，drop 减 ref。terminal owner 先使 context 进入 RETIRING/RETIRED 并 drop future 一次；只有 matching generation、RETIRED、refs=0 的唯一 CAS winner 才发布 FREE bitmap。normal completion、cancel、spawn rollback、last-ref delayed reclaim 与 old-waker-after-reuse 都有模型测试。

refs 使用 sticky saturation：per-slot 或 global 达到 `u32::MAX` 后永不递减，`can_teardown=false` 是故意的 fail-safe。它不是类型级 Pin；稳定地址仍是 unsafe API 与外部生命周期契约。

### Q: priority-lock 与 queue membership 竞态是否会丢任务？
**A — PASS（源码线性化高置信；两项并发测试为受控交错）**：

- dequeue 消费唯一 entry 后，对 same-generation `QUEUED|LOCK` 自旋；解锁后以 `QUEUED→RUNNING` CAS 接管。只有 generation/state 已由合法 owner 改变时才能丢弃 entry，不能把瞬时 lock 当 stale。
- `NOTIFIED→QUEUED` 与 enqueue-failure rollback 使用结构化 `RequeueOutcome`。赢得 transition 的线程直到 entry 已发布，或可靠发布 `ENQUEUE_RETRY`/terminal/stale，才结束 obligation；所有 lock/CAS miss 都 reload。
- scanner 条件拆为 `(PARKED && context refs==0) || ENQUEUE_RETRY`，因此一次已接受但发布失败的 wake 不依赖第二次 wake。

消费 cell 后再持锁的 dequeue 回归是确定性交错；NOTIFIED 与 rollback 测试用 barrier/yield 覆盖竞争窗口，但没有 test-only “已读 lock” hook，不能单凭测试声称每次调度都强制命中该瞬间，结论同时依赖源码状态机审阅。

### Q: priority getter/setter 是否仍有 ABA/TOCTOU，typed priority 能否被降级？
**A — PASS（高置信）**：priority lock 与 generation/state 在同一个 identity word；getter/setter 取得精确 live identity 后读写，再 release 原 state。terminal/recycle/wake 不能跨越 lock。普通 task 若已 QUEUED，异值更新返回 `DeferredUntilRequeue`，保留旧 queue entry 恰好一次，下次 enqueue 才使用新 class。

Typed spawn 在 QUEUED release-publish 前写入 `priority_fixed=1`。同值 setter 允许并按当前 state 返回 `PriorityUpdate`；异值返回 `TaskKeyError::PriorityFixed`。代价是 typed task 不支持动态 inheritance/aging；这里保证的是 type marker 与 runtime/wire priority 一致，不是完整优先级继承协议。

### Q: typed wait 的编译期保证和 runtime completion 到哪一层？
**A — PASS（限定范围，高置信）**：sealed `MayWaitFor` 只允许 Low→Low/Normal/High、Normal→Normal/High、High→High。`PriorityClass<P>` 只选择 spawn class；任意 task 仍可 fire-and-forget 任何 class。`PriorityToken<P>` 由成功 admission 后的 builder 获得且无公开 constructor，`TypedJoinHandle` 使用 caller-owned、single-use `CompletionCell`，不把可复用 TaskId 当结果 capability。Release publish / Acquire read，READY 不会被 cancel 覆盖；normal wait、cancel 与 QueueFull rollback 都释放 task/handle context refs。

这不是普通 Future/lock/channel 的全程序证明。Safe code 可把 token 移交给另一个 task，所以它不是 ambient identity 或信息流证明。另因 typed token/handle refs 会排除 legacy `refs==0` scanner，`CompletionTask` 对每个 inner `Pending` 执行 self-wake，保证“不保存 waker、第二次 Ready”的 legacy future 进展；代价是所有长期 Pending typed IO 都会 cooperative busy-repoll，增加调度份额与功耗，不是事件驱动 park。

### Q: metadata 与 trace/correlation identity 是否可信？
**A — PASS（契约内高置信）**：token/handle metadata 从 live `TaskContextRef` 读取 effective priority；task terminal 后返回 `Terminated`，不再伪装 spawn snapshot。trace ID 为 `(namespace << 32) | nonwrapping_local_sequence`；local 溢出返回 `TraceIdExhausted`。兼容 `init()` 使用匿名 namespace 0，只在该 executor epoch 内有意义，不能称跨 block/launch 全局唯一。

admission/reservation 与 free-slot claim 先于 trace allocation，所以 rejected `ReservedCapacity` 不消耗 local ID：224 个 accepted Low、1 个 rejected Low、16 个 accepted Normal 后，High 的 local ID 是 241。trace 分配后的 QueueFull rollback 可留下明确 gap。

### Q: free-slot ABA、内存序与 teardown 边界是否闭合？
**A — PASS/BOUNDARY（高置信）**：4 个 `u64` free bitmap 取代短 tag Treiber stack；70,001 次复用模型跨过 16-bit tag 空间且旧 key 仍 stale。host 使用 Acquire/Release/AcqRel，NVPTX wrapper 明确生成 system-scope acquire load、release store、acq_rel CAS；20,000 round cross-thread message litmus 通过，当前源码 NVPTX check/clippy 通过。

`is_quiescent_snapshot()` 仅为诊断。`can_teardown()`/`shutdown_complete()` 要求 shutdown 已冻结 admission 且 active/context/reclaim 为零，但仍不统计正在 `run()` 中的 warp，也不能阻止新的 unsafe run/spawn/wake。真正 free/move/re-init 前还必须：所有 executor warps 已退出、没有新入口、外部 device/kernel synchronization 已完成、地址始终稳定。shutdown 会主动取消 Pending task 而不等待另一次 task wake，但 escaped/saturated waker/context refs 可让 `run`/teardown 无限等待。

## Verification

- Runtime host debug/release `cargo test ... --lib` — **39/39 PASS**；release gate
  专门覆盖 slot-return 副作用不能藏在 `debug_assert!`。
- `typed_priority_runner.sh` — **13/13 PASS**：6 valid waits、High spawn Low、
  3 invalid waits、token forge、错误 receiver 反证与 no_std NVPTX graph。
- Host x86 Clippy（lib/tests）、NVPTX check/clippy `-D warnings`、fmt 与
  `git diff --check` — **PASS**。
- Host Phase3 safety oracle/model **3/3**；route tests **6/6**。
- Final IO PTX `fb95559e93eec1c...`（4,821,968 bytes），sm89 ptxas
  196.71s PASS。
- RTX 4070 Phase3 safety **25.10s / exit 0**：255 High + NoFreeSlots、reclaim
  `1→0`、slot 0 generation `1→2`、old key `Stale`、256 tasks各 poll2/complete1。
- Typed Phase3：`Cancelled→AlreadyJoined`、pending `2→1→0`、refs
  `2→1→0`、三阶段 masks `0xfe/0xfe`、final tasks `515/515`。
- Machine-direct evidence：
  `.research/findings/tasks/priority-handle-safety.3-final-gpu-evidence-c644.log`，
  SHA-256 `8141df0cd9e0900aa09db7c9e7299acd3cee441874a03b2da72a00e3fb060f08`。

## Unexpected Discoveries

1. priority accessor 的短锁不是普通“CAS miss 可放弃”：dequeue 已消费 cell 后持有唯一 membership obligation，必须等 lock 后完成 owner transfer。
2. `ENQUEUE_RETRY` 表示 wake 已被接受，不是“等待另一次 wake”；retained RawWaker 不能阻止容量恢复后的 scanner retry。
3. 总 context refcount 不能同时充当“Future 是否注册 RawWaker”的信号；typed token/handle 自身也 pin context。最小兼容方案是在 typed wrapper Pending 时自 wake。
4. 公开 Copy key 即使没有 outstanding ref 也可能长期存在，因此 authority 必须绑定 executor incarnation，不能只靠“不在 teardown 后保存 key”的文档。
5. `can_teardown` 是内部 shutdown snapshot，不是 allocation lifetime 的独立证明；warp exit 与 device synchronization 必须由外部确认。

## Open Questions / residual risks

- final PTX、ptxas 与真实 GPU multi-block stale-waker/generation/typed-shutdown
  stress 已由 `priority-handle-safety.3` 闭合；精确 priority-lock 窗口仍主要依赖
  host/model 与源码线性化审阅，没有声称每一种交错都在 GPU 上被强制命中。
- module reload、device/context 切换后的全局 executor epoch 不由 device static allocator保证；host 必须分配更大域的 launch identity。
- Typed self-wake 保证兼容进展但可能让长期 IO Pending busy-repoll；更优方案需把 raw-waker registration count 与总 context-pin refs 分离。
- `can_teardown` 没有 warp-exit ack；任何下游不得仅凭该 bool free/move executor。
- QUEUED priority 更新是显式 deferred，不做原地 relocation；没有 mid-poll preemption 或 wall-time deadline 保证。
- `TaskKey` 派生 Debug 会显示 owner 地址，属于诊断信息暴露的低优先级清理项。

## Impact on downstream tasks

- hostcall correlation 应使用 live `TaskMetadata.task_id/effective_priority`；anonymous namespace 0 不能当全局 ID。
- kernel/harness 可依赖 stale key/waker 不跨 executor incarnation 命中，并可观测
  `PriorityFixed`/`DeferredUntilRequeue`；final GPU litmus 已验证 generation reuse
  与 typed cancellation，但不替代外部 lifetime 契约。
- host teardown 必须组合 shutdown gate、kernel/warp join、no-new-entry 证明和 device synchronization；不能只看 active count 或 `can_teardown`。
- typed dependency只约束经 `PriorityToken::wait_for` 形成的边；普通 Future、raw channel/lock 与直接 polling 仍需独立 inversion 审计。
- 后续若要事件驱动 typed IO，应优先拆分 raw-waker 注册计数与 context pin ref，而不是移除当前 self-wake 后依赖全表扫描。
