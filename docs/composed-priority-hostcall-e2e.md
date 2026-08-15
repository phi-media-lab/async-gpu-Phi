# Typed priority + v3 hostcall 组合 E2E 规格

## 1. 目标与结论边界

本规格验证一条可证伪（falsifiable）的组合链：executor 的 typed Low
coordinator 在 Low admission 已饱和且 Normal backlog 存活时，创建 typed
High；High 从自己的 live `PriorityToken::wire_metadata()` 取得 wire identity，
经 ownership-safe async hostcall 使用 shared High-reserved packet；Low 通过
typed dependency 恰好 join 一次，第二次 join 必须返回 `AlreadyJoined`。

priority 只在 `Future::poll` 边界改变调度顺序，不能抢占正在执行的 poll。
测试不证明 WCET、hard deadline、端到端机器人响应界或功能安全等级。

## 2. 机器人安全边界 / Robot safety boundary

- 只运行合成 future、`PRIORITY_ECHO` 与 mapped-memory 记录，不接传感器或执行器。
- 不发油门、转向、制动、急停命令；GPU PASS 不能作为机器人部署放行依据。
- `fresh hazard` 使用真实 GPU 时间并选择 `apply_brake_from_fresh_result`。
- `fresh safe` 使用真实 GPU 时间并选择 `no_brake`。
- `stale` 使用明确标记的注入 age；结果必须被丢弃并走
  `watchdog_conservative_stop`。
- `deadline` 使用真实 High first-poll latency，同时注入一个更小 threshold；
  结果必须被丢弃并走 `watchdog_conservative_stop`。
- `fresh_budget = observed(ready-first)+1`、`deadline_budget =
  observed(first-inject)-1` 都是在运行后由观测值构造的 synthetic threshold，
  只演示分支和数据流；它们不证明预配置 deadline compliance 或实时 SLO。
- 生产系统仍需要独立硬件 watchdog、freshness/deadline 监测与 fail-safe stop。

## 3. 固定配置与前提

- 单 block、单 warp、单 executor namespace `0x0B57_A11E`。
- 单 shard `HostcallBuffer::new_with_priority_reserve(5, 1)`：general packet
  index `0..=3`，shared High-only packet index `4`。
- 4 个 Low `HostcallPacketLease` 持有 general credits，不提交请求。
- 219 个 raw Low holder + 1 个 typed Low coordinator：Low active 总数 224。
- 第 225 个 Low 必须得到 `ExecutorError::ReservedCapacity`。
- 随后 16 个持续 self-wake 的 Normal holder；High 注入前 active 总数 240。
- 所有 raw 与 typed task 共用同一个 monotonic trace sequence：coordinator
  local id 224，16 个 Normal 后 High local id 241。
- typed `PriorityClass<P>` 的 priority 固定；异值 setter 必须返回 `PriorityFixed`。
  mutation 1 只使用隔离 raw probe 做 scheduling-class substitution，不修改 typed task。
- 正常 High wire task id 固定为 `(0x0B57_A11E << 32) | 241`。
- 8H:4N:1L cooperative service 下，本场景保守 first-poll gap 为 `1..=6`；
  这不是 wall-clock latency bound。
- host shutdown 前必须先 CUDA synchronize 并阻止新 producer；listener 的
  final drain 依赖这一 no-new-producer safety contract。

## 4. 正例时间线

1. 初始化 executor namespace、两个 single-use `CompletionCell<u64>` 与 schema。
2. admit 4 个 Low lease holder，分别取得 general packet；admit 219 Low holder。
3. admit typed Low coordinator；确认 Low 总数 224，第 225 个 Low 被预留容量拒绝。
4. admit 16 个 Normal holder；executor 开始 cooperative run。
5. coordinator 等到 `lease_count=4`、`lease_mask=0b1111`、`normal_alive=16`，
   再尝试 Low packet acquire 并观察 `PoolExhausted`。
6. coordinator 记录 inject sequence/time，`spawn_typed<High>`。
7. High 首次有效 poll 从 live token 取 metadata，取得 packet 4；构造
   `PendingHostcall`。其第一次 poll 发布请求、self-wake，并强制返回一次 Pending；
   `MANDATORY_FIRST_PENDING=1` 单独写入 raw observation marker。该 marker 仍由
   kernel 产生，只有结合 runtime 契约、host event 与 exact downstream signature 才构成证据。
   总 Pending 次数只要求 `>=1`，
   host 响应时机可使观察值变化，因此不设精确值或上界。
8. host 只在 generation-owned v3 packet 上执行 `PRIORITY_ECHO`，独立记录
   nonce、namespace/local id、priority、packet index、shared-reserve provenance、
   generation、process sequence/count/error，再回写 echo。
9. High 后续 cooperative poll 观察 READY/ERROR，释放 device-owned READY packet，
   并把 echo task id 作为 typed completion output。
10. Low token `wait_for(&mut HighHandle)` 得到一次结果；第二次 wait 必须
    `AlreadyJoined`。随后释放 lease/Normal/Low backlog。
11. executor 终态必须 `spawned=completed=241`、active=0；host 已在 GPU
    synchronize 后 shutdown+join listener，再做 pool audit。

## 5. v3 async ownership contract

`HostcallPacketLease` 表示“已 pop、尚未 publish”的 device-owned packet：

- unsubmitted lease Drop 可直接按 packet provenance 归还 free stack；
- `submit` 消费 lease，物理发布延迟到 `PendingHostcall` first poll；
- first poll 总是 Pending，即使 host 极快完成；后续 Pending poll self-wake；
- READY/ERROR 都由 device 读取后归还 packet；ERROR 对 Future caller 可见；
- submitted timeout/Drop 绝不直接 free，只能 CAS
  `FILLED|HOST_OWNED -> CANCELLED`，归还责任转给 host；
- 若 Drop 看到 host 已赢得 READY，device 仍负有 release 责任，必须归还；
- generation mismatch、同 generation CANCELLED/IDLE 表示 device 不再拥有 packet。

Host listener 在 shutdown 时不能立即退出：它必须 final-drain ready stacks，
使“已 publish、device 已 timeout CANCELLED、kernel 已结束”的 credit 被回收。
若 timeout CAS 先于 host claim，listener 仍须记录一个独立 timeout echo event，并消费
`DelayNext` 一次性 hook；该路径不改 payload。若 host claim 先发生，则 handler 可先记录
成功 event，但 completion CAS 看到 CANCELLED 后仍由 host 回收。两种合法交错都不得把
第一次请求的 delay 或 payload 写入复用后的 generation。

Release 顺序固定为：先 release-store IDLE control，再清 metadata，最后 push free。
因此 `packet_metadata()` 的双 acquire snapshot 不会在不变 READY 下接受半清字段。

## 6. `PRIORITY_ECHO` wire contract

Service id 为 24。lane-0 inline payload 共 8 个 `u64`：

| Slot | Request | Response |
|---:|---|---|
| 0 | nonzero nonce | echo nonce；CONTROL_ERROR 时为 encoded error |
| 1 | 0 | live header task id |
| 2 | 0 | raw effective priority |
| 3 | 0 | physical packet index |
| 4 | 0 | shared High reserve provenance |
| 5 | 0 | host process sequence |
| 6 | 0 | identity process count |
| 7 | 0 | error category |

Host 必须拒绝 generation=None、nonce/namespace/local id 为 0、非 High wire
priority 或非 shared-reserved packet。旧 legacy/v1 packet 不能冒充 composed v3。

## 7. 独立 host oracle

Oracle 不读取 kernel PASS 位。harness 以独立 `ExpectedRun { mutation, nonce }`
传入 CLI mode 与 host 生成的 nonce；raw 字段必须分别 exact 匹配，不能只在 producer
内部自洽。日志中的 `raw=[112]` 是 GPU 输出加上 host 对 slots 54..60 的 event/audit
augmentation，不应称为纯 GPU 原始数组。Oracle 从这组结构化字段重算，并直接交叉核对 host
`PriorityEchoEvent` 与 `HostcallPoolAudit`。固定字面断言包括：

- packets `5/4/1`、Low `224`、一次 ReservedCapacity、lease `4/0b1111`；
- Normal alive 16、general PoolExhausted、first-poll gap `1..=6`；
- mandatory first Pending marker =1 且 total Pending >=1、wire/echo/event task id
  local 241、priority High；
- inject→submit→ready→typed wait/join 时间线单调且 submit sequence 等于 High first poll；
- GPU/echo/event packet index 4 且来自 shared High reserve；
- host event/process count 1、error 0、nonce 四处一致；
- coordinator local 224、join success 1、第二次 `AlreadyJoined` 1；
- join output与 token/wire/echo/event identity 一致；
- executor spawned/completed 241、active 0、lease release 4、Normal release 16；
- 四个 safety record 的 age 必须由 sample/now 重算；fresh timestamps 回指真实
  High first/ready，deadline sample 回指持久保存的 High inject timestamp、now 回指
  High first，stale source 必须明确为 injected；Action、AgeSource、end magic 等语义值
  由 oracle 自己的字面常量判定，不复用 producer 常量；
- audit ready empty、5 个 packet 都 idle 且恰好出现一次、general mask
  `0b01111`、High mask `0b10000`、无 duplicate/missing。

判定：schema/header/end-magic 损坏为 **UNKNOWN**；schema 有效但任一语义不符为
**FAIL**；全部独立 gate 成立才为 **PASS**。CUDA load/launch/synchronize/timeout
也只能是 UNKNOWN，不能伪装为语义 FAIL 或 PASS。

## 8. 五个 fault mutations

所有 hook 默认关闭；harness 对 mode 0 期待 PASS，对 mode 1..5 期待独立 oracle FAIL：

| Mode | Mutation | 必须变红的证据 |
|---:|---|---|
| 1 | 隔离 raw scheduling-class substitution：probe 用 High reserve admission，随后 raw effective class 设为 Normal；首个不可迁移 High entry 只 arm，self-wake 后追加到 16 个 live Normal 之后；不创建或降级 typed High | exact mode/marker、完整终态与clean audit、无event，arm + 16 FIFO 前驱使 meaningful gap >= 18 |
| 2 | typed High 的 wire priority 改为 Normal | exact Normal=1、其余 task identity 正确、PoolExhausted=9、未submit/no event、typed join与终态成立 |
| 3 | wire local id 改为 0 | exact mutation marker、namespace正确且local/task低32位为0、High使用packet4/shared reserve、单event exact ERR_INVALID_INPUT=4 且generation非零 |
| 4 | host 强制 ERROR | 正常wire identity/High/reserved前置、单event与raw error exact ERR_IO_ERROR=5、typed join与终态成立 |
| 5 | 隐藏第二次 `AlreadyJoined` 证据 | 其余typed/echo/event/audit正例链成立、首次join成功、replay实际到达marker=5，仅外显AlreadyJoined证据exact为0 |

Mutation 通过（即 gate 未变红）是测试失败；mutation 输出 schema 损坏是 UNKNOWN，
也不算成功杀死 mutation。任意无关语义 FAIL 也不够：mode 1..5 必须分别让指定的
gap、wire priority、identity、host error event、AlreadyJoined gate 变红。

## 9. Timeout→reclaim→same-packet reuse gate

另一个 kernel 使用 `HostcallBuffer(2, reserve=1)`：Low lease 持有 general index 0；
第一次 High echo 使用 reserved index 1，host 对下一次 echo 延迟，device timeout CAS
CANCELLED。第二次 High acquire 因 general/high 都不可用而等待，直到 host handler
结束、识别 CANCELLED 并回收 index 1；随后同一 index 以新 generation 处理第二 nonce。

PASS 同时要求：device 第一次观察到 host-timeout category 16；第一个 host event 在
claim-before-cancel 时为 0、cancel-before-claim 时为 16，第二 event 必须为 0；两次 packet
index 都为 1、generation 非零且不同、第二 echo 只含第二 nonce、无 stale write、host metrics
`cancelled=1/completed=1/errors=0/stale=0`，最终 pool audit 为 general mask 1、
High mask 2、ready empty、2/2 idle。第二次 acquire 的 busy-attempt 必须非零；两个 host
event 还须分别匹配 raw first/second/response packet index、local 250/251、
High/shared-reserve、generation exact `1 -> 2`、process sequence exact `1 -> 2`
与单次处理。

Timeout/reuse 使用独立具名 `u64[16]` schema：

| Index | Name | Meaning |
|---:|---|---|
| 0 | `VERSION` | schema version 1 |
| 1..2 | `FIRST_NONCE`, `SECOND_NONCE` | 两个请求输入 nonce |
| 3..5 | `FIRST_PACKET_INDEX`, `FIRST_ERROR_CATEGORY`, `REACQUIRE_BUSY_ATTEMPTS` | timeout 与等待 host reclaim 证据 |
| 6..9 | `SECOND_PACKET_INDEX`, `SECOND_ECHO_NONCE`, `SECOND_RESPONSE_PACKET_INDEX`, `SECOND_PENDING_POLLS` | 同 packet 新 generation 的成功响应 |
| 10..12 | `REUSE_STALE_WRITE`, `SECOND_COMPLETED`, `GENERAL_GUARD_PACKET_INDEX` | stale-write、完成与 general-credit guard |
| 13..14 | `RESERVED_0`, `RESERVED_1` | 必须为零 |
| 15 | `KERNEL_ERROR` | 零或 encoded terminal error |

## 10. Phase3 typed cancellation / shutdown gate

专用 multi-block safety litmus 在同一最终 PTX 中补足 composed 正例没有覆盖的
typed cancellation：

1. Low donor 把唯一真实 `PriorityToken<Low>` move 到 mapped `StoredToken`；
   donor handle 随后丢弃，不能复制或重建 token。
2. typed High target 首次 poll 进入“必返 Pending”分支并显式 self-wake。
   `FIRST_PENDING_SEQUENCE=1` 写在 poll 函数内部、实际 return 前，因此这里只证明
   分支与 wake 已发布，不声称 shutdown 严格晚于 poll return。
3. controller 记录 `SHUTDOWN_SEQUENCE=2` 并调用 executor shutdown；Low token 对 High
   handle 的首次 wait 必须得到 `TypedJoinError::Cancelled`，第二次必须
   `AlreadyJoined`。
4. token/handle drop 后，pending reclaims `2→1→0`、context refs `2→1→0`、
   active=0、所有 runner 退出且 `can_teardown=true`。

Host oracle 独立读取 616-word safety schema，检查三阶段 runner mask、事件序列
`1..6`、typed terminal state、generation reuse 与最终资源计数；kernel PASS 文本
不参与判定。

## 11. Schema v3：`u64[112]`

| Index | Name | Meaning |
|---:|---|---|
| 0..15 | `VERSION, WORDS, PHASE, NONCE, NAMESPACE, EXPECTED_HIGH_LOCAL_ID, PACKET_COUNT, GENERAL_PACKET_COUNT, HIGH_RESERVED_COUNT, HARD_LOW_LIMIT, HARD_NORMAL_BACKLOG, HIGH_FIRST_POLL_GAP_MAX, FRESHNESS_BUDGET_TICKS, DEADLINE_BUDGET_TICKS, EXPECTED_END_MAGIC, END_MAGIC` | header/config/schema closure |
| 16..23 | `LOW_ADMITTED, RESERVED_CAPACITY_REJECTIONS, LEASE_ACQUIRED, LEASE_MASK, NORMAL_ALIVE, GENERAL_POOL_EXHAUSTED, DISPATCH_SEQUENCE, HIGH_INJECT_SEQUENCE` | admission 与注入 |
| 24..31 | `HIGH_FIRST_POLL_SEQUENCE, HIGH_SUBMIT_SEQUENCE, HIGH_READY_SEQUENCE, LOW_WAIT_START_SEQUENCE, LOW_JOIN_SEQUENCE, LOW_SECOND_JOIN_SEQUENCE, HIGH_FIRST_POLL_TIMESTAMP, HIGH_READY_TIMESTAMP` | raw poll/timestamp timeline |
| 32..39 | `HIGH_PENDING_POLLS, LEASE_RELEASED, NORMAL_RELEASED, EXECUTOR_SPAWNED, EXECUTOR_COMPLETED, READY_STACK_EMPTY, POOL_SEEN_MASK, POOL_SEEN_ONCE_MASK` | completion/pool raw evidence |
| 40..47 | `WIRE_TASK_ID, WIRE_NAMESPACE, WIRE_LOCAL_ID, WIRE_PRIORITY, GPU_PACKET_INDEX, GPU_SHARED_HIGH_RESERVED, ECHO_NONCE, ECHO_TASK_ID` | wire/device/echo identity |
| 48..55 | `ECHO_PRIORITY, ECHO_PACKET_INDEX, ECHO_SHARED_HIGH_RESERVED, HOST_PROCESS_SEQUENCE, HOST_PROCESS_COUNT, HOST_ERROR, HOST_EVENT_COUNT, HOST_AUDIT_READY_EMPTY` | host echo/event evidence |
| 56..63 | `HOST_AUDIT_IDLE_COUNT, HOST_AUDIT_GENERAL_MASK, HOST_AUDIT_HIGH_MASK, HOST_AUDIT_DUPLICATES, HOST_AUDIT_MISSING, HIGH_INJECT_TIMESTAMP, MANDATORY_FIRST_PENDING, MUTATION_APPLIED` | independent audit、持久inject时钟、首poll契约与exact mutation marker |
| 64..71 | `JOIN_SUCCESS, SECOND_JOIN_ALREADY_JOINED, WAITER_LOCAL_ID, EXECUTOR_ACTIVE_FINAL, LOW_TOKEN_CONSUMED, HIGH_OUTPUT, MUTATION_MODE, KERNEL_OBSERVATION_FLAGS` | typed join 与 raw diagnostics |
| 72..111 | 4 × `DecisionRecord[10]` | fresh hazard、fresh safe、stale、deadline |

每个 `DecisionRecord` 字段顺序为：`kind, action, use_gpu_result, age_source,
sample_timestamp, now_timestamp, age_ticks, budget_ticks, hazard, reserved`。

## 12. 验证命令与当前状态

```bash
cd /home/fbsh/async-gpu

# 纯 CPU / host gates
cargo test --manifest-path crates/core/gpu-protocol/Cargo.toml
cargo test --manifest-path crates/core/gpu-runtime/Cargo.toml --lib hostcall::tests
cargo test --manifest-path crates/core/gpu-host/Cargo.toml --lib \
  hostcall::priority_tests::priority_echo -- --nocapture
AUTO_BUILD_KERNEL=0 cargo test \
  --manifest-path crates/test/gpu-test-harness/Cargo.toml composed_oracle

# executor P0 独立复验通过后才允许 fresh PTX 与真 GPU
./scripts/build-kernels.sh io
timeout 300s env AUTO_BUILD_KERNEL=0 ONLY_TEST=composed_priority \
  cargo run --manifest-path crates/test/gpu-test-harness/Cargo.toml --bin gpu-tests
```

截至 2026-08-15，executor/typed/hostcall P0 已经实现并由独立 verifier
复核。最终 IO PTX SHA-256 为
`fb95559e93eec1c07dbb4925c4564e43263cbd55de07a208ffa6b46b51e68531`
（4,821,968 bytes；GPU 运行时 mtime `2026-08-15 07:20:10 +0800`）；
六个 canonical route tests 全绿，`ptxas -arch=sm_89` 196.71 秒 PASS，
cubin 21,109,920 bytes。

RTX 4070 final matrix：

- composed fresh 207.62 秒、exit 0；positive、五个 exact designated mutation
  与 timeout→reclaim→same-packet reuse 全部 PASS；
- 同一 PTX cached 复验 1.84 秒、exit 0；
- Phase3 priority safety 25.10 秒、exit 0：三阶段 runner masks 均
  `0xfe/0xfe`，typed wait 依次 `Cancelled`、`AlreadyJoined`，
  pending `2→1→0`、refs `2→1→0`、最终 teardown 成立；
- obstacle 0.60 秒、exit 0；canonical trace 0.95 秒、exit 0，
  两个 trace kernels 各 32/32。

Fresh/cached positive 分别观察 `HIGH_PENDING_POLLS=4/5`。历史完整快照还观察过
1、3、4 等值，直接反证 exact poll-count 不变量；强契约只有第一次 poll 必须
Pending 且 total Pending `>=1`。

完整机器直录证据：

- `.research/findings/tasks/typed-priority.2-phase3-final-fresh-gpu-evidence-c644.log`
  — SHA-256 `39f3d3a7188a55eac06868758e43874563e4e5932d27ab0c57d4b7d44fbc4214`；
- `.research/findings/tasks/typed-priority.2-phase3-final-cached-gpu-evidence-c644.log`
  — SHA-256 `61445c72f7d35059275bc8ccf92d05f75a2e6a63541a4b347957509903a0bfb6`；
- `.research/findings/tasks/priority-handle-safety.3-final-gpu-evidence-c644.log`
  — SHA-256 `8141df0cd9e0900aa09db7c9e7299acd3cee441874a03b2da72a00e3fb060f08`；
- `.research/findings/tasks/obstacle-stress.1-phase3-final-gpu-evidence-c644.log`
  — SHA-256 `32c504e37f4dbd9a003ac402b68c55931d879e909a133c3e1de1d997b0339661`；
- `.research/findings/tasks/priority-hostcall-e2e.2-phase3-final-trace-evidence-c644.log`
  — SHA-256 `4080a889df2b5712da19d65acaac4599dc5219849051670bd39514cfec87944a`。

旧 `eefd4d97...`、`81be2f54...`、`51a9376c...` transcripts 均保留，
但明确是 superseded/candidate evidence：`81be2f54...` 的 safety litmus 揭示 release
构建会删除 `debug_assert!(free_slots.push(...))` 中的真实副作用；production fix
把 push 移出宏，并以 debug/release 39/39 与 final GPU generation reuse 闭合。
`51a9376c...` 已过 composed/safety/obstacle/trace，但缺 Phase3 typed
cancellation/shutdown 与三条 compile-pass edge，因此也不能替代 `fb95559e...`。

第一次 schema-v3 candidate 还因 `!u32::MAX as u64` 的运算优先级把整个 task id
清零而在 mutation 3 精确 oracle 上失败；失败日志保留，修复使用通用
`replace_task_local_id()` 并由 CPU exact regression 验证 namespace 保留、local=0。

最终静态门禁包括：protocol 9 unit + 7 doctest、runtime debug/release 39/39、
typed compile matrix 13/13、host priority 23 PASS/1 ignored、harness binary
21/21、composed oracle 10/10、route 6/6、safety oracle/model 3/3、fmt、
targeted Clippy、runtime NVPTX check/clippy 与官方 `scripts/ci-lint.sh` 全绿。
Canonical trace 仍只证明 route/kernel smoke；typed cancellation、reserve 与 identity
结论分别来自 composed 和 Phase3 safety，不能由 trace 替代。
