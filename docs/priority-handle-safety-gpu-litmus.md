# Priority handle safety GPU litmus

## 简体中文摘要 / Chinese summary

该可证伪集成测试验证 executor authority、延迟 RawWaker 回收、release 构建中的
slot return、typed cancellation 与 multi-block progress。两 block、八个 warps 分三阶段
验证：terminal Waker pin 后只能接纳 255 个 High；唯一 drop 后同 slot generation
`1→2` 且旧 key 为 `Stale`；256 个 `TwoPoll` tasks 各 poll 两次/完成一次；
真实 typed High Pending target 在 shutdown 后由 Low waiter 依次观察
`Cancelled`、`AlreadyJoined`，最终 refs/reclaims/active 均归零。

边界：first-Pending 事件在 poll 内、真正 return 前发布，只证明进入必返 Pending
分支并 self-wake；teardown 仍要求 no-new-entry、稳定地址、所有 runner/warp 已退出与
device synchronization。本测试不证明 kernel/warp preemption、hard-real-time、
跨设备公平性或机器人安全认证。

This test is a falsifiable integration check for executor authority, delayed
RawWaker reclamation, release-build slot return, typed cancellation, and
multi-block progress. It does not add a production API or claim kernel/warp
preemption, hard-real-time bounds, or general GPU fairness.

## Run

Build and statically validate the IO PTX once, record its hash, then disable
automatic rebuilding for the dedicated selector:

```bash
./scripts/build-kernels.sh io
sha256sum crates/core/gpu-host/kernel_io.ptx
AUTO_BUILD_KERNEL=0 ONLY_TEST=priority_safety cargo run --release \
  --manifest-path crates/test/gpu-test-harness/Cargo.toml --bin gpu-tests
sha256sum crates/core/gpu-host/kernel_io.ptx
```

The harness launches `priority_handle_safety_stress` as grid `(2,1,1)`, block
`(128,1,1)`: warp 0 is the controller and the other seven warps call the same
mapped `GpuExecutor::run` through three phases. A 30-second kernel timeout is
reported as `UNKNOWN`; mapped buffers are intentionally retained because the
GPU may still own their pointers.

## Phase 1: terminal Waker and generation authority

1. A Future clones its real executor Waker into mapped `StoredWaker` storage.
   `EMPTY -> WRITING` CAS elects the sole byte writer; the winner moves the
   Waker and release-publishes `READY`, then returns `Ready`. A duplicate poll
   loses the CAS, drops its own clone, and fails without racing two writes.
2. The independent host oracle requires the terminal slot to remain
   unavailable with one pending reclaim. Calling the terminal Waker must leave
   active, completed, pending-reclaim, and spawned counters unchanged. Context
   refs are diagnostic while runner poll Wakers can be transiently live.
3. While the slot is pinned, exactly 255 High tasks fit and the next spawn must
   return `NoFreeSlots`.
4. The controller is the unique acquire/CAS taker and drops the Waker exactly
   once; only then may one additional spawn succeed. An exact transient
   `QueueFull` is retried with a finite bound, and each failed admission
   rollback advances the slot generation.
5. The new task must use the same slot and satisfy
   `new_generation = old_generation + 1 + queue_full_retries`; the old
   `TaskKey` must return `Stale`.

The host never interprets or drops the Waker bytes. It frees storage only after
CUDA synchronization. Timeout, launch-thread disconnect, or launch-thread
panic therefore cannot turn cleanup into a Waker/executor use-after-free.

## Release-build slot-return invariant

Terminal reclaim and trace-allocation rollback must call
`free_slots.push(slot)` unconditionally. The result is checked with a release-
retained `assert!`; putting the side effect inside `debug_assert!` is invalid
because release/NVPTX removes the expression. Host release tests cover both
paths, and this release GPU litmus proves the terminal slot becomes admissible
after its last Waker is dropped.

## Phase 2: cross-block wake/requeue progress

After Phase 1 drains, the controller admits 256 `TwoPoll` tasks. Every task
records its ID, self-wakes on the first poll, and returns `Ready` on the second.
The oracle requires exactly two polls and one completion for every ID, all 513
tasks admitted through Phase 2 completed, all seven runners entered/exited,
and both CUDA blocks performed at least one poll.

## Phase 3: typed shutdown and cancellation

1. A real typed Low donor receives its executor-issued `PriorityToken<Low>` in
   the spawn factory. It moves the token into mapped `StoredToken` storage via
   `EMPTY -> WRITING`, byte move, and Release `READY`; its completion handle is
   dropped immediately. The control warp is the sole Acquire/CAS taker. The
   host never interprets or drops token bytes.
2. A typed High target reaches a branch that necessarily returns Pending. The
   first poll winner issues `wake_by_ref`, writes the self-wake/event fields,
   then Release-publishes a READY marker. The controller only accepts READY
   with an acquire load, so those writes happen before its shutdown request.
   The marker is still written inside `poll` before `Poll::Pending` returns;
   the test does not claim that shutdown is ordered after the call returns.
3. While seven runners remain inside `run`, the controller calls `shutdown`.
   After cancellation drives active tasks to zero, the legal Low-to-High
   `wait_for` edge must return `TypedJoinError::Cancelled`; a second wait on the
   same handle must return `AlreadyJoined`.
4. The controller drops the remaining handle/token. Stable pending reclaims
   must change `2 -> 1 -> 0`; all seven runner exit bits must appear. Final
   spawned/completed are `515/515`, Ready-task executions total 514 because the
   cancelled target is completed without returning Ready, and final active,
   context refs, and pending reclaims are zero.

The target's typed completion wrapper also self-wakes on Pending, so this is a
cooperative busy-repoll path rather than an event-driven parked-I/O guarantee.

## Residency and teardown boundary

The phase barriers assume both blocks can be resident together. Timeout and
block masks test that configuration on the target RTX 4070; it is not a
portable forward-progress theorem for arbitrary grids, register pressure, or
devices.

`can_teardown=1` is only an internal snapshot. Phase 3 requests shutdown while
runners are active, releases escaped context refs, waits for all runner exit
bits, and only then samples it. Allocation release remains ordered after the
controller returns and CUDA synchronization completes. The result is not
permission to move/free storage without no-new-entry, stable-address, warp-
join, and device-sync guarantees.

## Result schema and oracle

The kernel writes a versioned 616-word observation array and uses 256 bytes of
mapped ownership storage. Kernel-side booleans are not trusted as an overall
verdict: the host duplicates the schema and recomputes masks, counts, slot/
generation relations, per-ID cardinality, Waker/token states, typed results,
event order, pending reclaim transitions, and final teardown values after
synchronization. Default-zero and early-terminal negative tests prove that the
new Phase 3 oracle does not accept absent work.

## Recorded result

On the c644 RTX 4070 snapshot, PTX `fb95559e93ee...` (4,821,968 bytes) passed
route tests 6/6 and `ptxas -arch=sm_89` in 196.71 seconds. The dedicated release
run passed in 25.10 seconds: all three runner masks were `0xfe/0xfe`, both
blocks polled (`0x3`), retained-Waker capacity was 255 plus one `NoFreeSlots`,
pending reclaim changed `1 -> 0`, slot 0 reused generation `1 -> 2`, and the
old key was stale. All 256 TwoPoll tasks observed two polls/one completion.
Phase 3 observed donor token publication, target polls=2, shutdown,
`Cancelled`, `AlreadyJoined`, pending `2 -> 1 -> 0`, event sequence `1..6`, all
runner exits, and final teardown state. See
`.research/findings/tasks/priority-handle-safety.3-final-gpu-evidence-c644.log`
(sha256 `8141df0cd9e0900aa09db7c9e7299acd3cee441874a03b2da72a00e3fb060f08`).
