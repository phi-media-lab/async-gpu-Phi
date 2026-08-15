# priority-handle-safety.3-c644: real-GPU executor lifetime litmus

**Cycle**: c644 | **Kind**: integration implementation/verification | **Status**: COMPLETE on the recorded RTX 4070 snapshot

## Summary

Added a dedicated three-phase IO kernel and host selector for lifetime and
authority gaps not covered by the obstacle/composed routes. One mapped
executor is driven by seven runner warps in two CUDA blocks. Phase 1 retains a
real cloned RawWaker past task completion and proves delayed reclaim, exact
capacity loss, same-slot/new-generation reuse, and stale-key rejection. Phase
2 requires 256 self-waking tasks to poll exactly twice and complete once.
Phase 3 moves a real executor-issued typed Low token through uniquely owned
mapped storage, shuts down a typed High Pending task while runners remain in
`run`, observes `Cancelled` then `AlreadyJoined`, releases both typed context
references, and requires runner exit plus the external teardown prerequisites.

The first release-GPU attempt also exposed a production P0: two free-slot
returns were hidden inside `debug_assert!`, so release/NVPTX removed the
`push` side effect. Both sites now unconditionally return the slot and retain
an `assert!` as a fail-stop invariant. Host/model, NVPTX, ptxas, composed,
safety, obstacle, and trace gates pass on the final source/PTX snapshot.

## Findings

### Q: does the test exercise a real escaped executor Waker?

**A — YES (high confidence on the recorded GPU run).** `CaptureWaker::poll`
clones `cx.waker()`, moves the owned value into separately mapped
`StoredWaker`, and release-publishes `READY`. An `EMPTY -> WRITING` CAS is the
sole-writer linearization point, so a duplicate poll becomes an observable
loser/error rather than racing two non-atomic Waker writes. The controller is
the unique acquire/CAS taker, invokes the terminal Waker, and drops it exactly
once. The host never interprets Waker bytes. Timeout, disconnect, or launch-
thread panic retain all mapped allocations and report UNKNOWN.

### Q: what falsifies delayed reclaim and generation authority?

**A — an exact capacity/reuse chain (high confidence).** While the terminal
Waker was held, the GPU observed one pending reclaim; exactly 255 High holders
succeeded and the next admission returned `NoFreeSlots`. The terminal wake
left active/completed/reclaim/spawn counters unchanged. After the sole drop,
pending reclaims became zero and one additional task reused slot 0 at
generation 2 after generation 1; an old-key setter returned `Stale`. Context
refs are diagnostic while runners poll, so the oracle relies on pending
reclaim, capacity, and authority relations instead of a transient ref count.

### Q: what is checked across blocks?

**A — concrete progress/cardinality, not portable fairness.** Seven runner
warps recorded phase enter/exit mask `0xfe/0xfe` in all three phases. Every
phase-2 ID observed `(polls, completions) = (2, 1)`, and polling block mask
`0x3` proved both launched blocks performed work. This falsifies lost wake
membership, duplicate task polling, and single-block-only execution for this
launch. It does not establish a response bound or occupancy guarantee on
arbitrary GPUs.

### Q: does Phase 3 use the real typed dependency and cancellation APIs?

**A — YES (high confidence on the recorded GPU run).** A typed Low donor moves
its sole executor-issued `PriorityToken<Low>` into `StoredToken`; CAS
`EMPTY -> WRITING`, byte move, Release `READY`, then Acquire/CAS take make the
transfer unique. Its completion handle is dropped immediately. A typed High
target enters its necessarily-Pending branch, issues `wake_by_ref`, records the
event, then release-publishes the first-poll READY marker. The controller only
accepts READY with an acquire load, requests shutdown, and uses the legal
Low-to-High `wait_for` edge. The first wait returns exactly `Cancelled`; replay
returns exactly `AlreadyJoined`. Dropping the remaining handle/token changes
stable pending reclaims `2 -> 1 -> 0`; all seven runners exit and final
active/context refs/pending reclaims are zero with spawned/completed `515/515`.

The first-Pending event is published inside `poll` before it returns
`Poll::Pending`. It proves entry into that code path and a real self-wake before
shutdown, not that the control warp observed the poll call return.

### Q: is teardown inferred from `can_teardown` alone?

**A — NO (high confidence).** Phase 3 deliberately calls shutdown while all
seven runner warps are still in `run`. They cancel/drop the CompletionTask but
cannot finish the strict gate while the controller owns typed references. The
controller consumes the cancelled join, drops the references, waits for all
runner exit bits, then samples `can_teardown`. The host frees executor, typed
storage, Waker storage, and results only after the control warp returns and
CUDA synchronization completes. No-new-entry, stable-address, warp-join, and
device-sync requirements remain external and mandatory.

## Verification

- Runtime host unit/model tests: debug **39/39 PASS**, release **39/39 PASS**.
- Typed real-API/diagnostic matrix: **13/13 PASS**, including all six allowed
  wait edges and three forbidden edges.
- Phase 3 publication/oracle tests: **3/3 PASS**; fmt and host Clippy with
  tests/`-D warnings`: **PASS**; host and NVPTX checks: **PASS** (the existing
  toolchain `ptx78` and composed dead-code warnings remain scoped warnings).
- Executor/priority hashes: `de59ac88a737...` / `e011b0621ca3...`.
- Safety kernel/host hashes: `b274c2b4a39b...` / `50ad86eb2534...`.
- IO PTX: `fb95559e93ee...`, 4,821,968 bytes; route **6/6 PASS** and safety
  entry unique; `ptxas -arch=sm_89`: **PASS**, 196.71 s.
- Same-PTX composed fresh/cached: **PASS**, 207.62 s / 1.84 s. Evidence hashes:
  `39f3d3a7188a...` / `61445c72f7d3...`.
- RTX 4070 safety: **PASS**, 25.10 s, exit 0; PTX hash/size/mtime unchanged.
  Canonical evidence:
  `.research/findings/tasks/priority-handle-safety.3-final-gpu-evidence-c644.log`,
  sha256 `8141df0cd9e0900aa09db7c9e7299acd3cee441874a03b2da72a00e3fb060f08`.
- Same-PTX obstacle: **PASS**, 0.60 s, evidence sha256 `32c504e37f4d...`;
  trace: **PASS**, 0.95 s and `32/32 + 32/32`, evidence sha256
  `4080a889df2b...`.

## Unexpected Discoveries

1. `debug_assert!(self.free_slots.push(slot))` erased the slot-return side
   effect in release/NVPTX. The fix evaluates `push` unconditionally and traps
   on duplicate return before decrementing pending/active bookkeeping.
2. A bounded MPMC queue can transiently return `QueueFull` while a consumer
   owns the next sequence cell. The litmus retries only that exact error and
   checks `new_generation = old + 1 + retries`; the final run saw zero retries.
3. Runner poll Wakers make context refs unsuitable as an intermediate exact
   gate. Pending reclaim, exact capacity, and generation authority are stable.
4. A non-cooperative grid-wide spin barrier can deadlock unless both blocks are
   resident. `(2,1,1) x (128,1,1)` is an RTX 4070 configuration, not a general
   CUDA scheduling theorem.
5. An acquire polling wrapper marked `readonly` could be hoisted from a spin
   loop. The kernel uses the non-readonly spin-load wrapper.
6. Publishing the first-poll flag before its event allowed shutdown/event
   sequence inversion. A WRITING state now hides the observation until
   self-wake and event writes precede Release READY; a barrier model covers it.

## Open Questions

- Priority-lock dequeue/requeue windows remain proven by source linearization
  plus host/model tests; this GPU litmus does not force every exact interleave.
- CUDA context reset or PTX module unload while a Waker/token is escaped remains
  outside the supported lifetime contract.
- A returned launch/synchronize error is handled after joining the launch
  thread but is not conservatively leaked like timeout/disconnect/panic; a
  stronger all-error fail-safe policy remains follow-up work.
- Typed tokens are movable capabilities, not ambient identity or a whole-
  program information-flow/inversion proof.
- The observed two-block progress is target/configuration evidence only, not
  hard-real-time response, preemption, or portable fairness proof.

## Impact on Downstream Tasks

- `priority_safety` is the canonical real-GPU gate for escaped-Waker reclaim,
  same-slot generation authority, typed shutdown/cancellation, and two-block
  executor progress.
- The compile runner now covers all six legal dependency edges through real
  `spawn_typed`/`wait_for` calls and keeps exact negative diagnostics.
- Obstacle covers cooperative admission/scheduling；composed 的 host events、metrics
  与 reuse oracle 覆盖 identity/reserve/cancel；trace 只覆盖 route/listener/kernel
  smoke。三者都通过同一 final PTX，但不能互相替代或由 trace 外推 hostcall 成功。
- Research state may mark `priority-handle-safety.3` and the typed lifecycle
  acceptance item complete only when it binds the hashes/evidence above;
  portable fairness and exact lock-window GPU forcing remain explicit residuals.
