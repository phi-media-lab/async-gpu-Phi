# typed-priority.2-c644：typed priority + v3 hostcall composed E2E

**Cycle** c644 | **Story** `responsive-execution` | **Kind** composed experiment | **Status** done（final static + same-PTX RTX 4070 matrix independently verified）

## Summary

composed E2E 的 protocol/runtime/host、独立 oracle 与真实 GPU 链已经闭合：v3 async
lease/future ownership、inline `PRIORITY_ECHO`、host generation/identity/provenance
event、quiescent pool audit、schema v3 `u64[112]`、五突变 exact oracle 和 timeout/reuse
均已落地。executor P0 经独立复验解冻后，从最终源码 fresh 构建 PTX；RTX 4070
canonical composed positive + 五 mutation + timeout/reuse 于 207.62 秒 exit 0；同一 PTX
的 cached、Phase3 safety、obstacle 与 trace 分别独立 PASS。Phase3 在真实 GPU 上覆盖
pending typed target → shutdown → `TypedJoinError::Cancelled` → `AlreadyJoined`，
typed compile matrix 也补齐为 13/13；因此本任务的目标内 gate 已完成。

## Baseline

- 基线 v3 hostcall 只有同步 submit/wait；无 unsubmitted lease、async Future、echo event/audit。
- typed API 已提供 live `PriorityToken::wire_metadata()` 与 exactly-once typed join，但尚无
  typed scheduler→wire→host→echo→join 的单一真实场景。
- 所有 raw/typed task 共用 monotonic trace id，故 223 raw Low 后 coordinator local=224，
  16 Normal 后 High local=241；原设计 local=2 已被源码反证并修正规格。
- Legacy trace 32/32 吞掉 device hostcall Result，只能作为 route/kernel smoke。
- v3 初始 listener 在 shutdown 前直接 break，可遗留 ready/CANCELLED credit；release 又先
  清 metadata 后置 IDLE，使稳定 READY 双快照可能接受半清字段。

## Changes

- Protocol service 24、8-slot echo contract、schema v3 `u64[112]` 与四个具名
  `DecisionRecord`；v3 持久保存 High inject timestamp、mandatory-first-Pending 与
  exact mutation-applied marker。
- Runtime `HostcallPacketLease/PendingHostcall`：first poll 必 Pending+self-wake；READY/ERROR
  device release；submitted timeout/Drop 只 CAS CANCELLED transfer；unsubmitted Drop直返。
- Host 独立记录 nonce/task namespace+local/priority/index/shared reserve/generation/process/error；
  `quiescent_pool_audit()` 检查 ready/free/control cardinality；fault hook 默认关闭。
- Listener shutdown 在 GPU sync/no-new-producer 前提下 final-drain 到 ready empty；精确交错
  测试覆盖 publish→timeout CANCELLED→shutdown→credit reclaim。
- Cancel-before-claim 的 echo 也进入 host 独立事件流，记录 timeout/generation/provenance且不写
  payload；`DelayNext` 在该 generation 被消费，不能泄漏到复用请求。Claim-before-cancel 则
  可记录成功 event，但 completion 看到 CANCELLED 后仍由 host 回收。
- Release 改为先 IDLE、再清 metadata、最后 push；echo 拒绝 generation=None legacy 冒充。
- Host oracle 接收独立 `ExpectedRun { mutation, nonce }`，并硬编码 5/4/1、224/16、
  local 241、gap1..6、join/packet/event/audit/decision gates；Action/AgeSource/end magic
  不复用 producer 值。schema坏为 UNKNOWN，语义坏为 FAIL，不信 kernel PASS 位。
- 五 mutation：隔离 raw scheduling-class substitution、wire downgrade、identity local0、host
  forced error、join replay；每项必须匹配 exact marker、预期错误与正向前置，默认零/早退/
  unrelated error 不再能假杀 designated gate。正例的 typed High priority 固定且不被
  mutation 破坏。
- Timeout/reuse 场景额外占住 general credit，强迫 High timeout 后等待 host 回收并复用同一
  reserved packet；oracle要求第二次 acquire 至少一次 busy、host local 250→251、process
  sequence递增，以新 generation/nonce 反证 stale write；16-word 输出已改为protocol共享具名
  schema，不再散落magic index。

## Verification / Evidence

```text
gpu-protocol: 9 unit + 7 doctest PASS
gpu-runtime --lib: debug 39/39 + release 39/39 PASS
typed real-API/diagnostic matrix: 13/13 PASS
gpu-host hostcall::priority_tests: 23 PASS / 1 ignored legacy GPU smoke
harness --bin gpu-tests: 21/21 PASS
harness composed_oracle: 10/10 PASS
harness kernel_routes: 6/6 PASS
harness Phase3 safety oracle/model: 3/3 PASS
fmt + git diff-check: PASS
host/runtime/protocol/harness scoped clippy -D warnings: PASS
gpu-runtime x86 + NVPTX check/clippy: PASS
scripts/ci-lint.sh: PASS
final IO PTX: fb95559e... / 4,821,968 bytes / six routes 6/6
ptxas sm_89: PASS / 196.71s / cubin 21,109,920 bytes
RTX 4070 composed fresh: PASS / 207.62s / exit 0
RTX 4070 composed cached: PASS / 1.84s / exit 0
RTX 4070 Phase3 priority_safety: PASS / 25.10s / exit 0
RTX 4070 obstacle: PASS / 0.60s / exit 0
RTX 4070 canonical trace: PASS / 0.95s / exit 0 / 32+32 threads
```

Final fresh transcript 由 `exec_command/write_stdin` 原始 chunk 直接进入持久 buffer，
再由该字节串生成 `apply_patch`；没有 UI 复制或事后 SSH 重读。完整证据：

- `typed-priority.2-phase3-final-fresh-gpu-evidence-c644.log`，SHA-256
  `39f3d3a7188a55eac06868758e43874563e4e5932d27ab0c57d4b7d44fbc4214`；
- `typed-priority.2-phase3-final-cached-gpu-evidence-c644.log`，SHA-256
  `61445c72f7d35059275bc8ccf92d05f75a2e6a63541a4b347957509903a0bfb6`；
- `priority-handle-safety.3-final-gpu-evidence-c644.log`，SHA-256
  `8141df0cd9e0900aa09db7c9e7299acd3cee441874a03b2da72a00e3fb060f08`；
- `obstacle-stress.1-phase3-final-gpu-evidence-c644.log`，SHA-256
  `32c504e37f4dbd9a003ac402b68c55931d879e909a133c3e1de1d997b0339661`；
- `priority-hostcall-e2e.2-phase3-final-trace-evidence-c644.log`，SHA-256
  `4080a889df2b5712da19d65acaac4599dc5219849051670bd39514cfec87944a`。

Fresh/cached positive 分别观察 `HIGH_PENDING_POLLS=4/5`；历史完整快照还有
1/3/4 等值，反证 exact poll-count 不变量。契约只要求 mandatory first Pending 与
total Pending `>=1`。

旧 `eefd4d97...`、`81be2f54...`、`51a9376c...` 日志保留但明确降级。
`81be2f54...` safety 发现 release build 删除
`debug_assert!(free_slots.push(...))` 副作用；`51a9376c...` 已过早期矩阵但缺
Phase3 cancellation/shutdown 与三条 compile-pass edge。只有 `fb95559e...`
是 final evidence snapshot。

## Findings

1. **Async Drop 必须按当前 control winner 分配 release 责任。** READY winner 仍由 device
   release；FILLED/HOST_OWNED cancel winner 才转 host。Confidence: 10/10。
2. **Shutdown 是 ownership protocol 的一部分。** kernel timeout 后 ready stack 中可能只剩
   CANCELLED credit；无 final drain 会永久丢容量。Confidence: 10/10。
3. **Identity 必须来自 live token，但 trace sequence 是全 task 共享。** High local=241 是当前
   实现的真实证据，不应为迎合旧摘要伪造 local=2。Confidence: 10/10。
4. **Host event 与 pool audit 必须独立。** 只回写 payload 会让 kernel 自证；host event、
   generation、provenance、process count 和 pool cardinality 才能杀死 wire/host faults。
   Confidence: 10/10。
5. **此实验不证明 hard real-time。** 8H:4N:1L 只给 cooperative dispatch-order bound，
   wall clock、长 poll、GPU/driver/OS 均在范围外。Confidence: 10/10。
6. **Safety record 不能只校验标签。** Host 必须从 sample/now 重算 age，并把 fresh/deadline
   时间回指真实 High timestamps；伪造 timestamp 的 fixture 已被独立 oracle 杀死。Confidence: 10/10。
7. **Mutation 必须杀死指定 gate。** 仅得到任意 FAIL 会掩盖 fault hook 未生效；harness现逐项
   重算 gap/wire/identity/host-event/join gate，无关 FAIL 反例不会被接受。Confidence: 10/10。
8. **Decision threshold 是 post-hoc synthetic，不是预配置 SLO。** fresh budget 取观测
   ready-first+1，deadline budget 取观测 first-inject-1；持久 inject timestamp 只加强数据锚点，
   不把该分支实验提升为 deadline compliance 或 hard-real-time 证明。Confidence: 10/10。
9. **位运算优先级也必须由 mutation exact oracle 约束。** 初版
   `!u32::MAX as u64` 先在 `u32` 上取反再 cast，结果为 0，错误清空整个 task id；exact
   namespace/local gate 在真实 GPU 上拒绝了该假 mutation kill。现使用协议层通用
   `replace_task_local_id()`，CPU 回归 exact 断言 namespace `0x0B57_A11E` 保留、local=0。
   Confidence: 10/10。
10. **不能把有副作用的正确性操作放进 `debug_assert!`。** Safety 真 GPU 在 release
    构建中观察到 terminal reclaim 只减 pending、slot bitmap 未恢复；根因是整个
    `free_slots.push()` 随 debug assertion 被删除，trace-id rollback 同样受影响。现为
    无条件 push + release fail-stop assertion，并以 debug/release 39/39 与 GPU generation
    reuse litmus 闭合。Confidence: 10/10。

## Unexpected Discoveries

- `ERR_OTHER` wire category 数值为 0，不能作为“nonzero forced error”；test hook 改用
  `ERR_IO_ERROR`。
- High reserve耗尽后会 fallback general；timeout same-packet gate 必须额外持有 general credit，
  否则第二 High 会使用另一个 packet，无法证明 reuse。
- Production typed priority 是固定契约，不能为负例加入 bypass。Mutation 1 的 raw probe先以
  High reserve admission，再设effective Normal；首个High entry只arm，16个Normal FIFO前驱
  使meaningful gap确定至少为18。
- Timeout 的 host event error 受 ownership CAS 线性化点影响：host先claim时为0、device先cancel
  时为16；device timeout、host cancelled metric、新generation第二成功event三者共同闭合语义。
- 第一份 schema v3 fresh candidate 在 mutation 3 以 194.74 秒 exit 1 结束；这不是 flaky
  GPU，而是 Rust cast/取反优先级的真实 producer P0。失败 transcript 被保留，不与后续
  final positive/mutation 片段拼接。

## Remaining boundaries / 非阻塞边界

- Typed `PriorityClass<P>` 是 fixed priority；不支持 typed dynamic promotion/demotion，
  也没有为 conformance test 暴露生产 bypass。
- Phase3 的 first-Pending event 写在 poll 内、实际 return 前；它证明进入必返
  Pending 分支并 self-wake，不证明 shutdown 严格晚于 poll return。
- Host shutdown API 尚未在类型上强制 `GPU synchronized/no-new-producer`；harness
  以顺序约束满足它，late producer 仍在证明范围外。
- CUDA launch/synchronize 返回 Err 时的 mapped-memory teardown 尚未由源码证明为所有路径
  bounded ownership-safe；正例 PASS 不扩张为该异常路径保证。
- 8H:4N:1L 与 gap 1..6 是 cooperative dispatch-count 语义，不是 wall-time 或
  hard-real-time 证明；跨 GPU 公平性仍属于 `priority-progress-model`。

## Impact on Downstream Tasks

- `priority-hostcall-e2e.3` 已使用 async lease、echo event、pool audit 与 timeout gate 完成 typed round-trip。
- `progress-model.1` 应把 gap 1..6 限定为本单 warp cooperative workload，不外推硬实时。
- typed handle 发布门禁必须同时包含 executor 独立安全证明、composed oracle 与真实 GPU
  safety litmus；任一 CPU fixture 绿灯都不能替代另一层证据。
