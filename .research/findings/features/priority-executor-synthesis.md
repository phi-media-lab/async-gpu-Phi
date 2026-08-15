# Priority-executor synthesis

## 已落地 / Implemented
- High/Normal/Low API 保留 Normal-compatible `spawn()`；admission limits 为 Low224、Normal240、High256。
- 三个 MPMC queues 使用 8H:4N:1L cooperative service；priority 只影响下一 poll boundary。
- Public `TaskKey` 绑定 executor owner/incarnation/slot generation；private queue key 不泄漏 authority。
- Stable `WakerData` clone/wake/drop 引用阻止 terminal slot 在旧 waker 存活时复用。
- 4×`u64` free bitmap、sticky ref saturation 与无条件 release slot push 消除短 tag ABA/优化构建副作用。
- Priority-lock、wake-during-RUNNING、duplicate wake、QueueFull rollback 与 `ENQUEUE_RETRY` 保持唯一 queue membership。
- Typed priority 在 QUEUED 前固定；异值更新返回 `PriorityFixed`。
- Raw QUEUED priority update 返回 `DeferredUntilRequeue`，下一次 enqueue 使用新 class，不做原地 relocation。

## 最终证据 / Final evidence
- Runtime debug/release 39/39；typed compile matrix 13/13；x86/NVPTX check/clippy PASS。
- Final PTX `fb95559e...`；sm89 ptxas PASS；RTX 4070 obstacle 与 composed PASS。
- Multi-block safety：255 High + NoFreeSlots、reclaim `1→0`、slot generation `1→2`、old key `Stale`。
- 256 tasks 各 poll2/complete1；typed shutdown `Cancelled→AlreadyJoined`；final `515/515`。
- Safety evidence SHA-256 `8141df0c...`；official `scripts/ci-lint.sh` PASS。

## 边界 / Boundary
- 8H:4N:1L 是 dispatch-count property，不是 wall-time/WCET/hard-real-time bound。
- Long/non-yielding poll 不可被抢占；portable multi-warp fairness 仍属 `priority-progress-model`。
- Teardown 还要求 no-new-entry、stable address、runner/warp exit 与 device synchronization。
- Incarnation 不保证跨 PTX reload/device/context 全局唯一。

## 状态 / Status
GENERATION-SAFE EXECUTOR + FINAL RTX4070 LIFETIME VERIFIED；progress model pending。
