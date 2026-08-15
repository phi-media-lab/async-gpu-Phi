# typed-priority.1: Typed GPU priority dependencies
**Cycle**: 644 | **Theme**: typed-priority | **Kind**: design/experiment | **Status**: done

## Summary
Added a sealed type-level priority API over the existing cooperative executor.
`PriorityClass<P>` selects the spawn target independently from dependency rules,
while an executor-issued `PriorityToken<P>` can create `TypedWait` only when the
closed `MayWaitFor<Target>` relation allows the edge. Low may wait for every
class, Normal for Normal/High, and High only for High. This rejects the requested
priority-inverting waits in the real spawn/join API without claiming control over
ordinary futures, channels, locks, or direct polling.

`spawn_typed` accepts a delayed `FnOnce(PriorityToken<P>) -> Future` builder, so
the non-cloneable token is bound to the priority actually admitted and cannot
escape a rejected spawn. Each spawn also receives a wire `u64` task ID whose
low half is a non-wrapping launch-local `u32` sequence. Results use a caller-owned,
atomically single-use `CompletionCell`, not
the recyclable slot `TaskId`; a typed handle therefore cannot be redirected by
slot reuse. Explicit `TaskMetadata` reads task ID/effective priority from the
live `TaskContextRef` on each call and maps them to
`gpu_protocol::HostcallMetadata`; typed-fixed priority normally keeps the value
equal to admission, but terminal context returns an error rather than a stale snapshot.

## Baseline
- The executor exposed runtime `Priority` and recyclable `TaskId`, but no typed
  dependency relation, result handle, trusted task context, or trace identity.
- Any `Future` could await lower-priority work; the compiler saw no distinction.
- Reusing a bare slot ID for joins would create ABA after slot recycling.
- Shared wire `gpu_protocol::Priority` and `HostcallMetadata` were already
  available; this task consumes those types rather than defining a second wire.

## Findings

### Q: What does the compiler now reject?
**A (high confidence):** Dependency construction through the real
`PriorityToken::wait_for(&mut TypedJoinHandle<...>)` API rejects High→Low,
High→Normal, and Normal→Low because no corresponding sealed `MayWaitFor` impl
exists. Low→High, Normal→High, and High→High compile. The API also demonstrates
that a High task may fire-and-forget a Low spawn; spawn target and wait ordering
are intentionally separate.

### Q: Why is the task token trustworthy?
**A (high confidence):** `PriorityToken` has no public constructor and is not
`Clone`/`Copy`. `spawn_typed(PriorityClass<P>, cell, builder)` derives queue
priority, token marker, typed handle, and metadata from the same `P`. The builder
runs only on the admitted wrapper's first poll, so QueueFull/reservation/shutdown
failure drops the wrapper without exposing a token. A compile-fail fixture proves
safe callers cannot invoke its private constructor.

### Q: How are TaskId ABA and completion publication handled?
**A (high confidence):** `TypedJoinHandle` retains executor `TaskId` only for
diagnostics. It reads a distinct caller-owned `CompletionCell<T>`. The cell uses
EMPTY→RUNNING CAS once and never returns to EMPTY; a second spawn gets
`CompletionUnavailable`, so an old handle cannot observe a new task through cell
reuse. The unique wrapper writer stores `T`, then publishes READY with Release;
readers load state with Acquire. Drop performs only RUNNING→CANCELLED CAS and
cannot overwrite READY.

### Q: Can a pending typed wait make progress without a full-table scan?
**A (high confidence):** Yes under the executor's cooperative assumptions.
`TypedWait` self-wakes on every Pending poll, which engages the NOTIFIED state and
priority-aware requeue path. This costs one wake/state transition and another
queue dispatch per unsuccessful poll; it is progress-oriented busy waiting, not
an efficient target-to-waiter notification mechanism.

### Q: How is hostcall context propagated across blocks?
**A (high confidence):** Explicitly. `PriorityToken::metadata()` returns
`TaskMetadata { task_id, effective_priority }`, and `wire_metadata()` converts to
the shared protocol representation. There is no global current-context slot that
could alias tasks executing on different blocks. Metadata is a live context read,
not a spawn-time snapshot；terminal task返回error。Typed priority fixed使合法值与
admission一致；本结论不向没有 typed token/context 的 raw task 暗示 metadata API。

### Q: Is this a whole-program no-inversion proof?
**A (high confidence):** No. A safe caller may move an explicit token into other
code or another task, and untyped futures/channels/locks remain outside the API.
The guarantee covers edges constructed with the executor-supplied token used as
intended. It also provides no priority inheritance, queued relocation, mid-poll
preemption, or wall-time response bound.

## Verification
- `typed_priority_runner.sh`: PASS, 13/13. Real-API passes覆盖
  Low→Low/Normal/High、Normal→Normal/High、High→High、High fire-and-forget Low
  与 `#![no_std]` NVPTX graph；Real-API failures覆盖三个 downward waits、safe
  High-token forgery，并用错误 receiver mutation 证明诊断 gate 不会假阳性。
- `cargo test --manifest-path crates/core/gpu-runtime/Cargo.toml --lib`: PASS,
  debug/release 39/39（executor、typed、hostcall ownership/model）。
- `cargo clippy ... --lib -- -D warnings`: PASS.
- `cargo check/clippy ... --target nvptx64-nvidia-cuda -- -D warnings`: PASS.
- `cargo fmt --all -- --check`: PASS.
- Final `fb95559e...` RTX 4070 Phase3 safety：typed pending target 经 shutdown
  后首次 wait=`Cancelled`、第二次=`AlreadyJoined`，refs/reclaims `2→1→0`，
  spawned/completed `515/515`。

## Unexpected Discoveries
1. Calling the builder before admission would let a supposedly trusted token
   escape a rejected spawn. Deferring it to first poll closes that path.
2. A reusable completion cell merely moves ABA away from `TaskId`. Atomic
   single-use admission is required for the promised stable join capability.
3. Installing the built future changes an enum inside a pinned wrapper. Requiring
   the one-shot builder to be `Unpin` makes that move explicit; the resulting
   future is installed once and never moved afterward.
4. No target-owned waker registry exists, so correctness currently trades extra
   polls for self-wake progress.

## Remaining boundaries
- Decide whether a future scoped/non-delegatable context mechanism is worth the
  complexity; the current explicit token is deliberately a movable capability.
- Typed priority inheritance仍需要 generation-safe identity、atomic promotion、
  queue relocation 与 metadata refresh；当前 typed priority 是 fixed。
- `CompletionCell` currently supports only static, `Copy + Send + Sync` results;
  owned/dropping results need a separate allocation and ownership design.
- Phase3 first-Pending event 在 poll 内、真实 return 前发布；不声称 shutdown
  严格晚于 poll return，也不推出 wall-time/hard-real-time bound。
- Wire ID is `(namespace << 32) | local_sequence:u32`；local sequence 到
  `u32::MAX` 返回 `TraceIdExhausted` 且不回绕。其唯一域只覆盖一个已初始化
  executor namespace，不承诺跨 PTX reload/device/context。

## Impact on Downstream Tasks
- PriML-style code can opt into compile-time wait ordering only by keeping typed
  handles/tokens through the full dependency path; raw waits remain unchecked.
- Hostcalls can propagate explicit task trace and priority metadata across blocks
  without a second Priority type or an unsound ambient current-task variable.
- Runtime inheritance and a GPU completion test should build on this API rather
  than treating the compile-time relation as proof of scheduler preemption.
