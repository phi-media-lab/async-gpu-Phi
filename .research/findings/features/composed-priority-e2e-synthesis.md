# Composed priority E2E synthesis

## 已落地 / Implemented
- Protocol service 24、schema v3 `u64[112]`、reuse schema `u64[16]` 与 ownership-safe async lease 已落地。
- Listener final-drain、generation-aware cancel/reclaim、独立 host event/audit 与 typed exactly-once join 已闭合。
- Oracle 接收独立 `ExpectedRun { mode, nonce }`；五种 mutation 要求 exact signature + prerequisite，不能由全零或无关错误假杀。
- Timeout gate 交叉核对 packet index、generation `1→2`、local `250→251`、process `1→2`，反证 stale write。
- Phase3 safety 真实执行 pending typed target → shutdown → `Cancelled` → `AlreadyJoined`，并让 refs/reclaims/active 归零。

## 最终证据 / Final evidence
- Protocol 9 unit + 7 doctest；runtime debug/release 39/39；typed compile matrix 13/13。
- Host priority 23 PASS/1 ignored；harness binary 21/21；composed oracle 10/10；route 6/6；safety oracle/model 3/3。
- Final PTX `fb95559e93eec1c...`，4,821,968 bytes；sm89 ptxas PASS 196.71s。
- RTX 4070：composed fresh 207.62s / cached 1.84s、Phase3 safety 25.10s、obstacle 0.60s、trace 0.95s，全部 PASS。
- Final logs SHA-256：fresh `39f3d3a7...`、cached `61445c72...`、safety `8141df0c...`、obstacle `32c504e3...`、trace `4080a889...`。
- 本次 fresh/cached 分别观察 4/5 次 Pending；它只证明 mandatory first Pending + total `>=1`，exact 次数不是契约。

## 边界 / Boundary
- 只证明 cooperative poll-boundary ordering，不是 hard real-time、预配置 deadline SLO 或机器人认证。
- Fresh/deadline threshold 是运行后根据观测构造的 post-hoc synthetic 值。
- Trace 仅为 IO route/kernel smoke；typed/cancel/reserve 结论来自 composed + safety 独立 gates。
- Shutdown 要求先 GPU synchronize 且无 late producer；CUDA launch/sync 异常路径的 bounded ownership-safe teardown 仍未由源码完全证明。
- 旧 `eefd4d97...`、`81be2f54...`、`51a9376c...` 日志保留为 superseded/candidate evidence，不与 final snapshot 混用。

## 状态 / Status
COMPOSED + TYPED CANCELLATION + SAME-PTX REGRESSION COMPLETE；progress/fairness model 仍是后续任务。
