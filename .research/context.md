# Current Research Context

**Cycle 644 — `invisible-exec / responsive-execution` active**（2026-08-15）。

## Current position
- 当前 step：`select`；下一 task：`priority-progress-model.1`。
- Brainstorm sequence 135；本轮没有新 epic，story `responsive-execution` 继续 active。
- `priority-hostcall-e2e` 与 `priority-handle-safety` features 已完成。
- `typed-priority` 核心发布 gate 已完成；PriML-style channel/CV 调研 `typed-priority.3` 仍 pending。
- `priority-progress-model.1-.3` 仍 pending，是当前主要能力缺口。

## Final verified snapshot
- Final IO PTX SHA-256：
  `fb95559e93eec1c07dbb4925c4564e43263cbd55de07a208ffa6b46b51e68531`，
  4,821,968 bytes；sm89 ptxas 196.71s PASS。
- Protocol 9 unit + 7 doctest；runtime debug/release 39/39；typed matrix 13/13。
- Host priority 23 PASS/1 ignored；harness binary 21/21；oracle 10/10；
  route 6/6；Phase3 safety model/oracle 3/3。
- fmt、git diff-check、scoped host/runtime/protocol/harness Clippy、
  runtime x86+NVPTX check/clippy 与官方 `scripts/ci-lint.sh` 全绿。
- Composed fresh/cached：207.62s / 1.84s PASS；五 mutation exact kill，
  timeout→reclaim→same packet generation `1→2` reuse。
- Phase3 safety：25.10s PASS；三阶段 masks `0xfe/0xfe`，
  stale key、256 exact TwoPoll tasks、typed `Cancelled→AlreadyJoined` 全绿。
- Same-PTX obstacle 0.60s PASS；canonical trace 0.95s，32/32 + 32/32 PASS。

## Evidence binding
- Composed fresh/cached logs：SHA-256 `39f3d3a7...` / `61445c72...`。
- Safety log：`8141df0c...`；obstacle：`32c504e3...`；trace：`4080a889...`。
- `priority-hostcall-e2e.3a-c644.md` 显式映射 state task
  `priority-hostcall-e2e.3` 的 v3 ownership 子阶段，不是 orphan task。
- `priority-handle-safety.3` 已加入 state，绑定 multi-block final GPU litmus。
- 旧 `eefd4d97...`、`81be2f54...`、`51a9376c...` 只保留为
  candidate/superseded evidence；final claims 只绑定 `fb95559e...`。
- GPU矩阵完成后的首次host定向命令漏设 `AUTO_BUILD_KERNEL=0`，build.rs只重抄
  byte-identical `fb95559e...` 并把文件mtime改为07:39；该命令不计入gate，
  修正命令后23+1与Clippy PASS。GPU日志中的pre/post mtime仍是当时真实07:20值。
- GPU/trace证据后的07:44 completion audit只删除 `gpu-host/src/hostcall.rs`
  `#[cfg(test)] mod tests` 内6个冗余 `as u64`（pre/post source `ed379140...`→
  `4a5d0fd2...`）；production handler/listener、协议语义与PTX均未改。精确hunk已
  归档在 `priority-hostcall-e2e.3a-post-evidence-host-test-only-delta-c644.log`。

## Established semantics
- Priority 是 cooperative poll-boundary ordering，不可抢占当前 poll。
- Low admission 224；Normal/High 各预留 16 slots；service cycle 8H:4N:1L。
- Typed wait lattice 接受所有 equal/upward edges，拒绝三条 downward edges；
  safe code不能伪造 token。
- Typed priority 固定；raw QUEUED update 返回明确 deferred result，不做原地 relocation。
- Mandatory first Pending + total Pending `>=1` 是契约；final fresh/cached 的
  4/5 只是观测值，不是精确 poll-count 保证。
- Fresh/deadline thresholds 是 post-hoc synthetic values，只演示数据流与 fallback；
  不证明预配置 deadline compliance。
- Fresh hazard 使用 GPU 结果；stale/deadline 丢弃结果并走
  `watchdog_conservative_stop`。

## Remaining boundaries
- 不声称 hard real-time、WCET、跨 GPU portable fairness 或机器人安全认证。
- Progress model 尚需明确 SIMT/warp residency、poll quantum、barrier/memory scope、
  multi-warp fairness 与 non-yielding telemetry。
- Host shutdown 仍依赖外部 GPU synchronize/no-new-producer；late producer 不在证明内。
- CUDA launch/synchronize Err 后的 mapped-memory quiescence 没有在源码内证明为
  所有路径 bounded ownership-safe。
- Full harness 无过滤会并发启动 38 个 legacy GPU tests，本轮 >60s 无结果后中止；
  目标内串行四链独立 PASS，不能外推为全 suite 通过。
- IO kernel整 crate Clippy `-D warnings`仍有18个既有pipeline/hybrid/example lint；
  IO NVPTX check与本轮 scoped gates PASS，未用allow掩盖遗留。
