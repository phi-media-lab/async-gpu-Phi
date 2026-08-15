# priority-executor.1: Cooperative three-level GPU executor
**Cycle**: 644 | **Theme**: priority-executor | **Kind**: experiment | **Status**: done

## Summary
Implemented fixed High/Normal/Low cooperative scheduling in `gpu-runtime::executor`.
Legacy `spawn()` remains Normal; callers can use `spawn_with_priority()` or
`spawn_with_options(TaskOptions)`. Admission reserves 16 slots for High and a
further 16 for Normal-or-High, while dispatch uses a work-conserving 8:4:1
service cycle. This is queue/poll-boundary priority, not mid-poll preemption.

The task state machine now preserves wakes that race with RUNNING polls through
NOTIFIED, coalesces duplicate wakes, and re-enqueues using effective priority.
Spawn failure fully drops/recycles the future, counters and shutdown admission
are atomic, Pending tasks are cancelled/recycled during terminal shutdown, and
completed futures run their destructor. The underlying bounded MPMC queue was
also corrected with per-cell sequence publication/recycle handshakes.

## Baseline
- `cargo test --manifest-path crates/core/gpu-runtime/Cargo.toml`: library had 0
  unit tests; 61 ignored doctests and one unrelated existing `std_future::block_on`
  doctest failure (missing example context/import).
- `cargo check ... --target nvptx64-nvidia-cuda`: PASS.
- Audit found non-atomic diagnostic counters, an unused shutdown flag, QueueFull
  slot/future leakage, RUNNING-wake loss, duplicate enqueue exposure, missing
  future Drop, and unsafe queue-cell reuse after publishing an advanced head.

## Findings

### Q: What priority contract is implemented?
**A (high confidence):** `gpu_protocol::Priority` is the single shared type.
`spawn()` defaults to Normal. Saturated service is 8 High : 4 Normal : 1 Low,
with empty classes skipped. A continuously runnable Low task is probed once per
13 successful dispatches per scheduler cursor; elapsed-time liveness still
depends on each cooperative `poll()` returning.

### Q: Can lower-priority admission block a later High event?
**A (high confidence):** Not through task-slot exhaustion under executor-owned
admission. Low stops at 224 active tasks, Normal at 240, High at 256. Low receives
`ReservedCapacity` while High can still use the final 32 slots.

### Q: Are wake-during-poll and duplicate wake races fixed?
**A (high confidence):** Yes for the current slot generation. RUNNING wake does
CAS to NOTIFIED; Pending then performs NOTIFIED→QUEUED exactly once. PARKED wake
performs PARKED→QUEUED; QUEUED/NOTIFIED duplicate wakes are ignored, and dequeue
must claim QUEUED→RUNNING before polling.

### Q: Is shutdown linearized against spawn?
**A (high confidence):** Yes. One lifecycle word contains SHUTDOWN_BIT and the
active count. CAS decides whether spawn reserved capacity before shutdown or is
rejected after it. Shutdown never interrupts a current poll; queued tasks receive
one poll and Pending/parked tasks are dropped and recycled.

### Q: Is this hard-real-time or preemptive priority?
**A (high confidence):** No. A newly admitted High task waits for the current
poll to return and, at worst in the quota cycle, five successful lower-priority
dispatches. No wall-time bound exists without bounding individual poll duration
and GPU/driver interference.

## Verification
- `cargo test --manifest-path crates/core/gpu-runtime/Cargo.toml --lib`: PASS,
  8/8 executor unit/model tests.
- `cargo check ... --target nvptx64-nvidia-cuda`: PASS.
- `cargo clippy ... --lib -- -D warnings`: PASS.
- `crates/kernel/gpu-kernel-core cargo build --release`: PASS, PTX linked.
- `crates/kernel/gpu-kernel-io cargo build --release`: PASS, PTX linked with the
  obstacle stress kernel.
- RTX 4070 obstacle stress: PASS. Low admission 224/224 with one
  `ReservedCapacity` rejection; High spawn accepted. High first poll followed
  injection by one dispatch (225→226), its self-wake resumed at 227, all 223 Low
  workers completed, and executor terminal count was 225/225 over 450 polls.
  Final rerun after the sequence-load/align changes measured 19.456 µs injection
  to first High poll. The bounded non-yielding negative ran Low sequence 1→2
  before High sequence 3 (~8.920 ms), confirming no mid-poll preemption; every
  semantic gate was 1.

## Unexpected Discoveries
1. The old WorkQueue advanced `head` before clearing a consumed ring cell. A
   producer could observe capacity, reuse that cell, then have its task erased by
   the old consumer. Per-cell generation sequences now publish and recycle safely.
2. `TaskSlot` only implied 8-byte alignment and never dropped futures. Inline
   storage is now explicitly 16-byte aligned; over-aligned futures are rejected.
3. Existing GeneratorTask/oneshot futures can return Pending without arranging a
   wake, so an empty-queue compatibility scan remains necessary.

## Open Questions
- RawWaker encodes executor pointer plus 8-bit slot index, not a slot generation;
  an externally retained stale waker could target a later occupant after reuse.
- 8:4:1 cursors are warp-local. They ensure bounded service attempts, but a formal
  multi-warp fairness proof and adversarial GPU stress remain downstream work.
- `set_effective_priority()` deliberately does not relocate an already QUEUED
  task; the new value applies at its next re-enqueue boundary.
- Hard response bounds require a maximum poll quantum plus platform-specific
  bounds for residency, memory contention, clocks, driver launch, and completion.

## Impact on Downstream Tasks
- obstacle-stress can fill Low admission to `ReservedCapacity`, inject High from
  inside a poll, and observe first-poll/waker/Low-liveness counters.
- hostcall priority metadata uses the same `gpu_protocol::Priority`, avoiding
  conversion drift between executor and host service paths.
- Any PriML-style inheritance can write effective priority now, but should add
  generation-safe task handles and queued-task relocation before claiming strict
  priority inheritance semantics.
