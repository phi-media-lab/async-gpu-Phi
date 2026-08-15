# Theme Synthesis: priority-handle-safety

## Implemented
- Public authority is executor/incarnation-bound `TaskKey`; queues use private `LocalTaskKey`.
- Stable WakerData delays terminal reuse; saturated refs permanently pin rather than wrap.
- Four `u64` bitmaps remove short-tag free-stack ABA; release slot returns are unconditional.
- Queue publication obligations survive priority locks and QueueFull rollback.
- Typed task priority is fixed before QUEUED; differing mutation returns `PriorityFixed`.
- Typed Pending self-wakes for cooperative busy-repoll progress.
- Live metadata rejects terminal tasks; namespace plus nonwrapping sequence identifies traces.
- System-scope acquire/release ordering covers Waker/token ownership publication.

## Evidence
- Runtime debug/release **39/39**; real typed API/diagnostic matrix **13/13**.
- Host Phase3 model/oracle **3/3**; fmt, host Clippy, NVPTX check, route **6/6** PASS.
- Final PTX `fb95559e...`; sm89 ptxas PASS; composed fresh/cached PASS.
- RTX 4070 safety PASS (25.10 s): Waker pin gives 255+NoFree, reclaim `1->0`.
- Same slot generation `1->2`, old key Stale; both blocks poll; 256 tasks poll2/complete1.
- Typed donor/target shutdown yields Cancelled then AlreadyJoined, pending `2->1->0`.
- All three runner masks are `fe/fe`; final tasks `515/515`, refs/pending/active zero.
- Same-PTX obstacle and trace PASS; safety evidence sha256 `8141df0cd9e0...`.

## Remaining boundary
- Teardown still requires no-new-entry, stable address, warp join, and device sync.
- Incarnation is not global across PTX module reload, device, or context.
- Typed tokens are movable capabilities, not ambient identity or whole-program inversion proof.
- GPU evidence is RTX 4070 configuration-specific, not a hard bound or portable fairness proof.
- Exact priority-lock windows remain source/model evidence, not forced GPU interleavings.
## Status
IMPLEMENTED; HOST/MODEL/NVPTX/FINAL-PTX/RTX4070 LIFETIME VERIFIED WITH THESE BOUNDARIES.
