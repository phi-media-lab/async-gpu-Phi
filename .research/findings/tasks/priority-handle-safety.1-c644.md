# priority-handle-safety.1-c644：priority executor / typed handle 安全审计

**Cycle** c644 | **Kind** read-only P0 concurrency audit | **Status** complete | **Code changes** none

> **Historical audit snapshot / 后续闭合说明**：本文正文审计的是文件中所列
> `8d333b...` / `9daa3c...` 旧源码快照；其中“当前版本”“仍没有”等措辞只描述
> 该快照，不是 final repository verdict。后续 `priority-handle-safety.2` 已实现
> generation-tagged `TaskKey`/authority、refcounted stable `WakerData`、delayed
> reclaim、bitmap free slots、显式 system-scope ordering 与 live typed context；
> `priority-handle-safety.3` 在 final `fb95559e...` PTX 上以 multi-block Phase1/2
> stale-key litmus 和 Phase3 typed shutdown 验证 `Stale`、`Cancelled→AlreadyJoined`
> 与 refs/reclaims `2→1→0`。本文保留为发现这些 P0 的阶段证据，不能覆盖后续
> final findings、源码哈希或 GPU verdict。

## Scope and Method

只读审查当前共享工作树：

- `crates/core/gpu-runtime/src/executor.rs`，1315 行，SHA-256 `8d333bb82c73b532e28e3dfb2900b9f962033f87525bc8276c4de1d1ef1012b1`。
- `crates/core/gpu-runtime/src/priority.rs`，535 行，SHA-256 `9daa3c268aa4931082dd29d1b665350312d971ad89766f6e3f290f139ad73927`。
- 为判断实际内存语义，只读核对了 `gpu-atomics` 的 PTX inline-asm wrapper；没有修改它。

按任务约束，本审计没有编译、没有运行测试或 GPU，也没有修改 executor、priority、其他 findings、synthesis 或 `state.toml`。以下“最小复现”均为应新增的确定性交错测试，不声称已经执行。

## Executive Verdict

当前版本不宜把 recyclable `TaskId`、RawWaker 或动态 effective priority 宣称为跨 task-slot 复用安全。主要 stop-ship 项为：

1. **BUG / P0：RawWaker 没有 generation 和所有权。** 旧 waker 可唤醒复用同一 slot 的新任务；executor 映射释放后还可能 UAF。
2. **BUG / P0：`TaskId` priority mutation 存在 ABA/TOCTOU。** 旧 ID 或一次跨回收的 setter 可写坏新 occupant。
3. **UNPROVEN / P0：状态 CAS 和 free-head CAS 没有 PTX `.sem`。** 源码按 acquire/release 推理，但 wrapper 发出 `atom.cas.sys...`，没有可核查的 acq_rel 保证。
4. **BUG / P0：free-slot head 只有 16-bit tag。** 65,536 次成功 head 更新即可回绕，可构造 classic Treiber ABA。
5. **BUG / P0（端到端优先级语义）：typed token/wire metadata 是 spawn snapshot。** scheduler priority 被动态提升后，hostcall 仍发送旧 priority。

CompletionCell 单次结果、同 generation 内的 wake/Ready/Drop、自 wake 合并和 typed handle 单次 join 的局部状态机是合理的，但它们目前被 waker generation、executor lifetime 和内存序前提包围。

## Verdict Matrix

| Audit item | Verdict | Severity | Short reason |
|---|---|---:|---|
| RawWaker cross-slot generation / ABA | **BUG** | P0 | data 只有 executor pointer + 8-bit slot |
| RawWaker executor lifetime | **BUG** | P0 | clone/drop 无 refcount；外部 clone 可越过 executor 映射寿命 |
| Wake vs Ready/Drop, same generation | **PASS** | — | `RUNNING/NOTIFIED -> COMPLETING` CAS 给 terminal owner 唯一权 |
| QueueFull rollback | **UNPROVEN** | P1/P0 if public queue used | 内部 one-entry invariant 下正确；public WorkQueue 可制造 stale/duplicate entry |
| Shutdown state transitions | **UNPROVEN** | P0 | parked/running 路径合理；escaped waker 与 executor quiescence 未解决 |
| CompletionCell prepare/Ready/Cancelled | **PASS** | — | single-use CAS；Ready release/acquire；Drop cancel 不覆盖 Ready |
| Completion polling LICM | **UNPROVEN** | P0 | progress load 使用带 `readonly` 的 inline acquire load |
| Duplicate/self wake, same generation | **PASS** | — | first wake transitions state，subsequent wake coalesces |
| Typed handle repeated wait/reuse | **PASS** | — | `&mut` capability + `joined` + non-resetting cell |
| Trace ID uniqueness within one init epoch | **PASS** | — | independent atomic u64 sequence；slot recycle does not redirect join |
| Effective priority vs wire metadata | **BUG** | P0 semantic | live slot 可变，token/handle metadata 永久为 spawn snapshot |
| `TaskId` getter/setter generation | **BUG** | P0 | recyclable index，没有 generation validation |
| FreeSlotStack tag wrap | **BUG** | P0 long-running runtime | 16-bit tag wrap permits stale CAS success |
| PTX CAS publication ordering | **UNPROVEN** | P0 | no `.acquire/.release/.acq_rel` qualifier or explicit proven fence pair |
| Packed RawWaker pointer provenance | **UNPROVEN** | P1 | pointer→u64 mask/OR→pointer；依赖 256-byte alignment 和 provenance 假设 |

## 1. RawWaker Generation, ABA, and Lifetime — BUG / P0

Relevant code: `executor.rs:520-550`, `598-640`, `818-824`, `1055-1057`.

RawWaker data packs only `(executor pointer, slot index)`. Clone is a bit copy, drop is a no-op, and terminal recycle immediately publishes `SLOT_FREE` and pushes the slot. There is no generation snapshot and no outstanding-waker count.

### Constructible cross-slot interleaving

1. Task A occupies slot `s`; its Future clones `WA` into a channel/timer.
2. A returns Ready. Polling warp CASes `RUNNING|NOTIFIED -> COMPLETING`, drops A, stores `FREE`, and pushes `s`.
3. Task B reuses `s` and reaches `RUNNING` or `PARKED`.
4. The old `WA` fires. It decodes only `s` and either changes B `RUNNING -> NOTIFIED`, or changes B `PARKED -> QUEUED` and enqueues B.
5. B receives a spurious poll/wake belonging to A. `Waker::will_wake` also reports old/new same-slot wakers as equal.

The stronger ABA does not require WA to start after completion: WA may read old `RUNNING`, stall, let A complete and B reuse `s` as `RUNNING`, then its stale `CAS(RUNNING, NOTIFIED)` succeeds against B.

### Constructible executor UAF

1. A stores a cloned waker externally, then completes.
2. `run()` observes active count zero and returns; host/kernel owner frees or reuses the executor mapping.
3. External code calls the safe `Waker::wake` API.
4. vtable code reconstructs and dereferences the freed executor pointer.

The unsafe executor API cannot transfer this lifetime obligation to arbitrary safe Future code that is allowed to clone a `Waker`.

### Minimal repro

- A GPU/model test with one free slot: A saves its waker and returns Ready; B must reuse the slot and park; invoke A's saved waker; assert B was not requeued. Current implementation should fail.
- An instrumented transition test that pauses wake between state load and CAS, performs terminal recycle plus B admission, then resumes the CAS.
- A shutdown lifetime test that retains a waker after `active_count==0`; freeing the backing mapping must be rejected until the reference is gone.

### Exact fix

Use a stable per-slot `WakerData` plus real RawWaker ownership:

- Fields: executor pointer/owner, slot, immutable live generation, atomic refcount, retired/reclaim state.
- Creating the Context waker owns one ref; clone increments; `wake` consumes and decrements; `wake_by_ref` does not consume; drop decrements.
- Remove the current `ManuallyDrop<Waker>` behavior or explicitly release its base ref after each poll.
- Terminal owner first marks the generation retired and drops the Future, but does **not** publish FREE while refs remain.
- The unique `try_reclaim` winner publishes the slot to the free pool only when terminal and refcount zero. Handle the self-waker-drop reentrancy with a reclaim CAS, not recursive free.
- Increment generation only after refcount zero and before the next admission.
- Track global outstanding waker ownership so executor memory cannot be released until shutdown is terminal and all refs are zero. A leaked waker must retain the executor allocation, as an Arc-backed CPU executor would.

Only adding a mutable `slot.generation` check is insufficient: after reuse, an old waker pointing at the same mutable record would read B's new generation unless the waker itself retains A's immutable snapshot.

## 2. Wake vs Ready/Drop Within One Generation — PASS (Conditional)

Relevant code: `executor.rs:608-629`, `831-843`, `1072-1104`.

For a non-recycled generation, the CAS state machine gives one terminal owner:

- Wake wins first: `RUNNING -> NOTIFIED`; Ready path accepts `NOTIFIED -> COMPLETING` and no queue entry was created.
- Completion wins first: `RUNNING -> COMPLETING`; wake retries, observes COMPLETING, and ignores it.
- Pending vs wake: if poll parks first, wake transitions `PARKED -> QUEUED`; if wake notifies first, poll transitions `NOTIFIED -> QUEUED`; only one enqueue wins.

No same-generation double-drop interleaving was found. This PASS depends on fixing Section 1: a stale generation can make the same CAS succeed against a different Future.

### Minimal regression test

Use barriers around wake CAS and terminal CAS for both orderings. Count destructor calls and queue publications; require exactly one destructor and zero post-terminal queue entries.

## 3. QueueFull Rollback — UNPROVEN

Relevant code: `executor.rs:149-267`, `621-625`, `846-855`, `931-969`.

### Internal invariant result

Under the claimed one-entry-per-active-task invariant, each per-priority queue has 256 cells while total active tasks are at most 256. Therefore an internal enqueue of a newly admitted or currently unqueued task should not return QueueFull.

If it nevertheless does:

- Spawn rollback owns an unpolled Future, changes it to COMPLETING, drops it exactly once, frees the slot and active credit, and typed `CompletionTask::drop` cancels its cell.
- Wake/requeue rollback changes QUEUED to ENQUEUE_RETRY; a later wake/compatibility scan retries, and shutdown can cancel it.

### Why the public surface is unproven

`work_queue` and unsafe `WorkQueue::enqueue` are public, but the safety contract does not forbid duplicate valid slot indices. A duplicate/stale entry can survive rollback and later CAS a reused slot from QUEUED to RUNNING. Queue items also carry no generation.

### Minimal repro

Fill/inject the public normal queue with duplicate slot indices, force an internal enqueue failure, recycle that slot, reuse it, then drain the stale entry. Assert it cannot claim the new generation; current u32 entry cannot make that check.

### Exact fix

- Make raw queues private, or explicitly make uniqueness/generation part of the unsafe contract.
- Store a generational `TaskKey` in queue cells (likely u64), and validate key generation before `QUEUED -> RUNNING`.
- Model `ENQUEUEING`/publication explicitly if QueueFull remains recoverable; rollback must invalidate the exact TaskKey, not merely a slot state.
- Add fault-injection tests for failure before queue reservation, after state publication, and during shutdown.

## 4. Shutdown — UNPROVEN / P0 Until Wakers Quiesce

Relevant code: `executor.rs:701-739`, `872-887`, `997-1035`, `1078-1104`, `1114-1139`.

The combined shutdown bit and active count correctly linearize admission: a spawn either reserves before the shutdown CAS and remains an admitted task, or observes the bit and returns Shutdown. Queued tasks are polled; Pending RUNNING/NOTIFIED tasks are claimed by their polling warp; PARKED/ENQUEUE_RETRY tasks are claimed by the shutdown scan.

Remaining blockers:

- Terminal cancellation recycles slots even when escaped RawWakers exist, causing Section 1 ABA/UAF.
- `run()` returning at active zero does not prove global waker refcount zero.
- Outside shutdown, an idle warp may read active zero and exit concurrently with a new spawn reserving `0 -> 1`; if all executor warps take that exit, admitted work remains until another `run()` call. The API does not state a no-concurrent-spawn-at-idle rule.

### Minimal repro

- Populate one task in each QUEUED/RUNNING/NOTIFIED/PARKED/ENQUEUE_RETRY state, race shutdown with admission on both sides of its CAS, and require active zero, exactly-one drop, and correct typed Cancelled/Ready result.
- Retain wakers from cancelled tasks and require shutdown/quiescence to keep executor storage alive.
- Barrier an idle run-loop load of active zero against a concurrent spawn CAS and verify the executor cannot strand the task.

### Exact fix

Add a lifecycle phase plus waker quiescence count. `shutdown_complete` must mean: no admitted logical task, no terminal slot awaiting waker refs, no queue publication in flight, and no outstanding waker that can dereference the allocation. Either reject concurrent idle admission with a sealed RUNNING phase or make spawn restart/notify an executor warp.

## 5. CompletionCell prepare / Ready / Cancelled — PASS

Relevant code: `priority.rs:217-289`, `377-440`, `460-493`.

- `prepare` permits only EMPTY -> RUNNING, so a completion cell is single-use.
- All spawn failure paths drop the wrapper: pre-copy failures drop the parameter; post-copy enqueue rollback calls the stored destructor. The wrapper's Drop attempts RUNNING -> CANCELLED.
- Complete writes the value then release-stores READY; poll acquire-loads READY before volatile-reading the value.
- Cancel is CAS RUNNING -> CANCELLED and cannot overwrite READY.
- Executor state ownership prevents concurrent complete and wrapper Drop in the intended state machine.

### Minimal regression test

Test double prepare, FutureTooLarge/alignment/admission/enqueue rollback, shutdown cancellation, and Ready-vs-cancel order. Require one terminal state and one destructor. This PASS does not cover the LICM concern below.

## 6. Duplicate and Self Wake — PASS (Conditional)

Relevant code: `executor.rs:559-629`, `1085-1101`; `priority.rs:352-373`.

- During RUNNING, first wake creates NOTIFIED; subsequent wakes observe NOTIFIED and are ignored.
- On Pending, NOTIFIED is converted to one QUEUED entry.
- From PARKED/ENQUEUE_RETRY, only one CAS to QUEUED succeeds.
- TypedWait's explicit `wake_by_ref` executes while its waiter is RUNNING, so it follows the one-NOTIFIED path and guarantees another completion poll.

This is PASS only for one generation and an intact one-entry invariant. Stale wakers and public duplicate queue injection invalidate those premises.

### Minimal regression test

Issue N concurrent wakes at each state, including two self-wakes inside one poll; assert at most one queue publication and one next poll. Repeat across a terminal/reuse boundary and require old-generation wakes to be rejected.

## 7. Typed Handle Repeated Wait and Completion Reuse — PASS

Relevant code: `priority.rs:300-375`.

`TypedJoinHandle` is not Clone. `wait_for` requires `&mut`, preventing two safe simultaneous waiters. The first Ready or Cancelled observation sets `joined`; subsequent waits return AlreadyJoined. CompletionCell never returns to EMPTY, so an old handle cannot observe a later spawn through cell reuse.

Dropping a Pending TypedWait permits a later wait on the same handle, which is correct because no result was consumed. Dropping the handle without joining does not affect producer completion.

### Minimal regression test

Poll/drop/recreate a Pending wait, then complete and join once; a second wait must return AlreadyJoined. Attempting a second spawn with the same cell must return CompletionUnavailable.

## 8. Task Trace ID and Wire Metadata

### Trace identity — PASS within one init epoch

Relevant code: `executor.rs:682-688`; `priority.rs:117-193`, `472-493`.

`spawn_typed` derives token metadata, typed handle metadata, runtime class and wire metadata from the same type-level P and one atomic trace ID. The trace ID is independent of the recyclable slot; failed spawn may leave a harmless gap. It wraps only after 2^64 allocations and is reset only by the documented one-time init.

### Dynamic effective priority — BUG / P0 semantic

Relevant code: `executor.rs:1173-1204`; `priority.rs:117-193`.

Constructible mismatch:

1. Spawn typed Normal task with trace ID 42; token stores `(42, Normal)`.
2. Call `set_effective_priority(TaskId(s), High)` while it is live.
3. Its next enqueue uses High, so executor scheduling treats it as High.
4. Inside the task, `token.wire_metadata()` still returns `(42, Normal)` forever.
5. Hostcall scheduling therefore receives a different “effective” priority from the executor.

The module-level comment calls metadata a spawn-time snapshot, so this is documented behavior, but it does not satisfy an end-to-end effective-priority propagation claim. Either rename it `base_priority_snapshot` and prohibit dynamic inheritance for typed hostcalls, or make the context reference a stable live priority record.

### Exact fix

Use a refcounted/generational TaskContext record, separate from recyclable slot bytes, containing trace ID and atomic effective priority. Setter, scheduler and wire conversion must update/read that same record. A token moved to another task must retain the record, or that transfer must be prohibited.

## 9. Recyclable TaskId Priority APIs — BUG / P0

Relevant code: `executor.rs:116-118`, `1157-1204`.

TaskId is only a public u32 slot index. Getters check state once, and setter checks state then separately stores priority.

Constructible setter ABA:

1. Caller retains `TaskId(s)` for A; setter reads A as RUNNING.
2. A completes; slot is cleared, freed and reused by B.
3. Setter resumes and release-stores High into B's effective priority.

Even without pausing inside one call, invoking an old A TaskId after B admission mutates B. Getters can likewise read state from one occupant and priority from another. Typed handle documentation correctly calls this ID diagnostic-only, but the public mutation API accepts it as authority.

### Minimal repro and fix

Reuse one slot and call setter with A's old ID; B must remain unchanged. Replace TaskId authority with `TaskKey { slot, generation }`; validate generation in the same atomic state word before and during mutation. Queue entries and RawWaker identity must use the same key. Getters should validate generation before and after the field read, and reject COMPLETING as non-live.

## 10. FreeSlotStack Short-Cycle ABA — BUG / P0

Relevant code: `executor.rs:366-470`.

The u64 head uses only 16 tag bits and 16 index bits; 32 middle bits are unused. Both successful pop and push increment the u16 tag.

Constructible interleaving:

1. Warp P reads head `(tag=t, index=i)` and the then-current `i.next=j`, then stalls.
2. Other warps perform 65,536 successful head updates and arrange the head back to the same encoded `(t, i)` while `i.next` now differs.
3. P's stale CAS succeeds and publishes its old `j`, losing or duplicating part of the current free list.

For a persistent executor, 65,536 task completions/allocations is not a remote bound.

### Exact fix

Preferred: replace the linked free list with four u64 availability bitmaps. Pop CAS-clears a selected free bit; push atomically sets it and can detect double free. Returning to the same bitmap value is benign because no stale `next` pointer is carried.

Lower-churn alternative: use the currently unused bits for a 48-bit tag and 16-bit index, with acq_rel system-scope CAS. This raises the wrap horizon but does not remove theoretical ABA. Add a deterministic model test that forces tag wrap; do not attempt 65,536 GPU iterations as the only proof.

## 11. Memory Ordering and NVPTX LICM — UNPROVEN / P0

Relevant code: all slot/free-stack CAS sites; `priority.rs:253-270`; `gpu-atomics/src/lib.rs:78-205, 261-316, 370-398`.

### Missing semantic qualifier

The CAS wrapper emits `atom.cas.sys.global.b32/b64` with no `.acquire`, `.release` or `.acq_rel` qualifier, and its documentation promises atomicity but not ordering. Executor comments nevertheless use CAS as if it published/consumed surrounding data.

Concrete ordering that needs proof:

1. Producer warp writes result data and wakes A while A is RUNNING.
2. Wake uses CAS RUNNING -> NOTIFIED.
3. Polling warp reads NOTIFIED and requeues A.
4. The next poll must observe producer writes made before wake. An acquire load cannot synchronize with a relaxed CAS that has no release semantics.

Free pool publication has the same gap: terminal owner clears slot/writes `next`, then a non-release CAS publishes head; pop's acquire head load has no demonstrated release partner.

### LICM/progress loads

`sys_load_acquire_u32` is `#[inline(always)]` with `options(readonly)`, and its own source warns LLVM may hoist it from a spin loop. Completion polling is repeatedly inlined through task polls, and idle/shutdown progress also depends on externally changing state. Indirect poll dispatch probably inhibits some hoisting, but source structure is not a proof.

### NVPTX-lowerable fix

- Add explicit wrappers using PTX forms such as `atom.acq_rel.sys.global.cas.b32` and `.b64` for slot-state/free-pool transitions. Use release or acq_rel on publication and acquire/acq_rel on claims; do not rely on Rust `Ordering` names that never reach PTX.
- If toolchain syntax/support blocks semantic atomics, use a documented `membar.sys` protocol with a precise matching acquire operation, then validate generated PTX/SASS. A fence comment alone is insufficient.
- Add a no-`readonly`, no-sleep `sys_poll_load_acquire_u32` for repeatedly polled progress state. Existing `sys_spin_load_acquire_u32` is suitable where its nanosleep is intended.
- Keep volatile value/pointer accesses, but treat volatile only as an optimizer constraint, not as inter-warp synchronization.

Required PTX gates after implementation:

```text
CHECK: atom.acq_rel.sys.global.cas.b32   # slot state / wake
CHECK: atom.acq_rel.sys.global.cas.b64   # free pool if CAS-based
CHECK: st.release.sys.global.u32         # Completion READY publication
CHECK: ld.acquire.sys.global.u32         # Completion observation
```

Add a FileCheck around the completion/idle loop backedge proving the acquire load remains inside the loop. Also run a cross-warp message-passing litmus: payload store -> wake -> next poll payload load, with many iterations and no same-warp shortcut.

## 12. Packed Waker Pointer — UNPROVEN / P1

`pack_waker_data` converts a pointer to u64, masks/ORs address bits, then recreates a pointer. `repr(align(256))` and the allocation contract support the low-bit layout, but strict pointer provenance and executor-base placement are assumed rather than represented in the type system.

The stable `WakerData` fix removes the need for address packing: RawWaker data becomes a real aligned pointer to a live record. Until then, add layout/runtime assertions and avoid claiming portability beyond the validated NVPTX address model.

## Recommended P0 Fix Order

1. **Freeze unsafe authority surface:** make raw queues private and stop treating slot-only TaskId as a mutation key.
2. **Introduce one generational identity:** `TaskKey(slot,generation)` in state, queues, diagnostics and priority APIs.
3. **Implement stable refcounted WakerData:** correct clone/wake/wake_by_ref/drop, terminal retirement, delayed slot reuse and executor lifetime quiescence.
4. **Replace free list:** prefer 4×u64 bitmap; otherwise 48-bit tag as an interim mitigation.
5. **Make PTX ordering explicit:** acq_rel system CAS, poll-safe loads, and generated-PTX checks/litmus tests.
6. **Unify live effective priority:** scheduler and hostcall wire metadata read the same stable TaskContext record.
7. **Re-audit shutdown/rollback on TaskKey:** include in-flight publication and outstanding-waker counts in terminal criteria.
8. **Add deterministic regression harnesses:** forced ABA barriers, stale waker/TaskId, double wake, queue failure injection, completion cancellation and idle-spawn race.

Do not fix only one of generation, waker refcount or delayed reuse: any two without the third leave a reuse or lifetime hole.

## Post-task closure

- `priority-handle-safety.2` 闭合本审计的 generation/authority/refcount、free-slot
  release side effect、queue rollback 与 NVPTX ordering P0；final executor SHA-256
  为 `de59ac88...`，debug/release runtime 39/39 与 typed compile matrix 13/13 PASS。
- `priority-handle-safety.3` 的 RTX 4070 final log SHA-256 为 `8141df0c...`：旧 key
  在 slot generation `1→2` 后返回 `Stale`，256 tasks 各 poll 2/complete 1；typed
  Pending target 经 shutdown 后得到 `Cancelled`，二次 join 为 `AlreadyJoined`。
- Final trace identity 不是“可回绕 u64 sequence”：wire ID 是
  `(namespace:u32 << 32) | local_sequence:u32`，local 到 `u32::MAX` 返回
  `TraceIdExhausted` 且不回绕。Typed metadata 每次 live 读取 `TaskContextRef`；
  terminal context 报错，不使用 spawn-time snapshot。
- 仍未完成的是 priority inheritance/queued relocation、bounded poll quantum 与
  GPU-specific progress model；上述 final gates不构成 hard-real-time/WCET 证明。

## Files Changed

- `.research/findings/tasks/priority-handle-safety.1-c644.md` only.
