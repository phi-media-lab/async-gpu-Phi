# 障碍事件优先级压力测试规格 / Obstacle-event priority stress specification

## 1. 结论边界

本规格验证一个可证伪的协作式调度命题：当 GPU executor 已被 Low 工作压到其准入上限后，运行中的 Low future 注入 High 障碍事件，预留容量仍允许 High 入队；在测试限定的“队列中只有 Low、每次 poll 有界且会让出”前提下，High 应成为下一次 poll，并且 High 的自唤醒仍保留 High priority。所有已接纳 Low 最终仍应完成。

这不是硬实时（hard real-time）保证。priority 只在两个 `Future::poll` 的边界生效，不能抢占正在执行的 poll；GPU 驱动、JIT、kernel launch、SIMT 收敛、显存/系统总线、操作系统和硬件故障均不在该保证内。

## 2. 机器人安全边界 / Robot safety boundary

测试只运行合成 future 并写 mapped-memory 结果：

- 不连接真实摄像头、雷达、编码器或障碍传感器；
- 不发送转向、油门、制动或急停命令；
- 不证明 worst-case execution time（WCET）、deadline 可达性或功能安全等级；
- `stale/deadline fallback` 只验证纯决策输出，不能控制 executor，更不能代替独立 watchdog；
- 真机必须由独立安全控制器/硬件 watchdog 监测 freshness 与 deadline，通信失联、结果过期或超时都进入 fail-safe conservative stop；
- GPU 新鲜且按时返回 hazard 时，路径是 `apply_brake_from_fresh_result`；结果 stale 或 deadline miss 时必须丢弃该 GPU 结果，路径是 `watchdog_conservative_stop`。两条路径都趋向安全，但数据来源与原因码不同。

因此，本测试通过只表示“当前实现满足下述协作式语义”，不能直接用于机器人部署放行。

## 3. 被测契约与假设

被测公开契约为：

- `Priority::{High, Normal, Low}`；
- `GpuExecutor::spawn_with_priority(future, priority)`；
- `MAX_TASKS = 256`；
- High 与 Normal 各预留 16 个 active slots，因此 Low admission limit 为 224；
- Low 达到 224 后返回 `ExecutorError::ReservedCapacity`，High 仍可进入剩余容量；
- 三个 ready queues 按 8H:4N:1L 加权服务；waker 根据 task slot 的 effective priority 重新入队；
- executor 是 cooperative/non-preemptive：一个 poll 返回前不会改 poll 另一任务。

测试成立所需假设：

1. 单个 block、单个完整 warp（32 lanes）共同调用 `run(mask)`，lane 0 执行 future poll。
2. 正例 initial ready set 只有 Low；不混入 Normal。由此 High 在注入当前 poll 返回后应成为下一次 poll，dispatch gap 上限为 1。
3. 若生产场景同时存在 Normal，8H:4N:1L 实现允许理论上最多出现 5 个 lower-priority polls；本测试的 `gap <= 1` 不可外推到该混合场景。
4. Low future 每次 poll 有界：第一次自唤醒并返回 `Pending`，第二次返回 `Ready`。
5. `%globaltimer` 只提供诊断时间。语义 gate 主要使用单调 dispatch sequence，不以某个微秒阈值声称 deadline 保证。
6. 使用当前构建生成的 `kernel_io.ptx`；host 必须从 `gpu_host::ptx::KERNEL_IO` 加载入口。

## 4. 可证伪假设 / Falsifiable hypotheses

| ID | 假设 | 反例（会使 gate 失败） |
|---|---|---|
| H1 | Low 最多接纳 224 个 active tasks，随后得到 `ReservedCapacity` | 接纳数量不是 224、得到其他错误或未拒绝 |
| H2 | Low 饱和后在 Low poll 内注入 High 仍成功 | High spawn 失败 |
| H3 | 纯 Low 前提下 High first poll dispatch gap ≤ 1 | 注入后先 poll 了其他任务，gap > 1，或 High 从未 poll |
| H4 | High 第一次 poll 自唤醒后仍按 High 入队，resume gap ≤ 1 | waker 降级/丢失，gap > 1，或未 resume |
| H5 | 所有被接纳的 Low 工作最终完成 | `low_completed != low_work_admitted` |
| H6 | priority 不会抢占当前 poll | 负例 High 在 Low return milestone 之前获得 poll |
| H7 | stale/deadline fallback 不采用过期 GPU 结果 | stale/deadline 输出 `used_gpu_result != 0` 或 action/reason 错误 |

## 5. 正例时间线

```text
T0  初始化 positive executor
T1  先以 Low 入队 LateHighInjector
T2  继续以 Low 入队 YieldingLow，直到第 225 个 Low active task 被 ReservedCapacity 拒绝
T3  executor 首次 poll injector；它 self-wake，返回 Pending，排到 Low queue 尾部
T4  223 个 Low work 各被首次 poll、self-wake、返回 Pending
T5  injector 第二次 poll：记录 inject_seq/ns，在 poll 内 spawn High，然后 Ready
T6  预期下一次 dispatch 就是 High first poll：记录 first_seq/ns，self-wake，Pending
T7  预期下一次 dispatch 是同一 High resume poll，然后 Ready
T8  继续 drain Low；全部 223 个 Low work 第二次 poll 后 Ready
T9  计算 admission、first-poll、waker、Low liveness gates
```

`first_poll_gap = high_first_seq - inject_seq`，`waker_gap = high_resume_seq - high_first_seq`。两者的 pass bound 都是 1 个 dispatch；wall-clock latency 仅打印。

## 6. 预期负例：non-yielding Low

第二块 executor 只入队一个 Low。它在同一次 poll 内：

1. 记录 `inject_seq/ns` 并 spawn High；
2. 执行 4096 次有界 `nanosleep.u32 1024` busy-work/yield hint；
3. 记录 `low_return_seq/ns`；
4. 返回 `Ready`；
5. High 才能获得 first poll 并记录 `high_first_seq/ns`。

结构性 pass 条件为：

```text
low_return_seq == inject_seq + 1
high_first_seq == low_return_seq + 1
high_first_ns >= low_return_ns >= inject_ns
```

这项“负例通过”证明 High 无法在当前 Low poll 中间抢占。busy-work 有界，避免用无限循环把 GPU/测试机挂死。`negative_latency_ns` 是诊断量，不设硬实时阈值。

## 7. Fresh/stale/deadline 决策实验

合成决策输入不操纵真实 executor 或 actuator：

| Case | 输入 | 期望 action | reason | `used_gpu_result` |
|---|---|---|---|---:|
| fresh hazard | age=10, first-poll=10, budgets=100/100 | `apply_brake_from_fresh_result` (1) | `fresh_hazard` (1) | 1 |
| stale result | age=101, freshness budget=100 | `watchdog_conservative_stop` (2) | `stale_result` (2) | 0 |
| deadline miss | age=10, first-poll=101, deadline=100 | `watchdog_conservative_stop` (2) | `deadline_miss` (4) | 0 |

关键安全断言：stale GPU 结果未被采用；deadline-missed GPU 结果也未被采用。这里验证的是 decision output，不是物理制动效果。

## 8. 结果 schema：`u64[48]`

kernel 与 host harness 都使用具名常量访问字段，不在判定逻辑中散落数字索引。

| Index | 字段 | 含义 |
|---:|---|---|
| 0 | `VERSION` | schema version，当前 1 |
| 1 | `PHASE` | 1=初始化，2=正例已入队，3=正例结束，4=负例已入队，5=负例结束，6=全部完成 |
| 2 | `LOW_ADMITTED_TOTAL` | Low active 总数，含 injector |
| 3 | `LOW_WORK_ADMITTED` | YieldingLow 数量 |
| 4 | `LOW_RESERVED_REJECTIONS` | `ReservedCapacity` 拒绝次数 |
| 5 | `LOW_OTHER_REJECTIONS` | 其他 Low admission 错误 |
| 6 | `HIGH_SPAWN_OK` | 正例 High 是否成功入队 |
| 7 | `POS_INJECT_SEQ` | High 注入 dispatch milestone |
| 8 | `POS_HIGH_FIRST_SEQ` | High first-poll milestone |
| 9 | `POS_HIGH_FIRST_GAP` | first - inject |
| 10 | `POS_HIGH_RESUME_SEQ` | High self-wake 后 resume milestone |
| 11 | `POS_HIGH_RESUME_GAP` | resume - first |
| 12 | `LOW_COMPLETED` | 完成的 YieldingLow 数量 |
| 13 | `POS_SPAWNED` | positive executor spawned counter |
| 14 | `POS_COMPLETED` | positive executor completed counter |
| 15 | `RESERVED_PASS` | H1+H2 gate |
| 16 | `FIRST_POLL_PASS` | H3 gate |
| 17 | `WAKER_PASS` | H4 gate |
| 18 | `LOW_EVENTUAL_PASS` | H5 gate |
| 19 | `POSITIVE_PASS` | 正例组合 gate |
| 20 | `POS_INJECT_NS` | 诊断时间戳 |
| 21 | `POS_HIGH_FIRST_NS` | 诊断时间戳 |
| 22 | `POS_LATENCY_NS` | first_ns - inject_ns，仅诊断 |
| 23 | `NEG_HIGH_SPAWN_OK` | 负例 High 是否入队 |
| 24 | `NEG_INJECT_SEQ` | 负例注入 milestone |
| 25 | `NEG_LOW_RETURN_SEQ` | Low 即将返回 milestone |
| 26 | `NEG_HIGH_FIRST_SEQ` | 负例 High first-poll milestone |
| 27 | `NEG_INJECT_NS` | 诊断时间戳 |
| 28 | `NEG_LOW_RETURN_NS` | 诊断时间戳 |
| 29 | `NEG_HIGH_FIRST_NS` | 诊断时间戳 |
| 30 | `NEG_LATENCY_NS` | first_ns - inject_ns，仅诊断 |
| 31 | `NEG_NO_MIDPOLL_PASS` | H6 gate |
| 32..34 | `FRESH_*` | fresh action/reason/use-result |
| 35..37 | `STALE_*` | stale action/reason/use-result |
| 38..40 | `DEADLINE_*` | deadline action/reason/use-result |
| 41 | `DECISION_PASS` | H7 gate |
| 42 | `POS_DISPATCH_SEQUENCE` | 正例最终 dispatch milestone |
| 43 | `NEG_DISPATCH_SEQUENCE` | 负例最终 milestone |
| 44 | `EXPECTED_LOW_LIMIT` | 期望 Low 上限，当前 224 |
| 45 | `NEG_BUSY_ITERS` | 负例有界工作次数，当前 4096 |
| 46 | `OVERALL_PASS` | 全部语义 gate 的 conjunction |
| 47 | `WORD_COUNT` | 固定为 48，检测 schema 漂移 |

## 9. 判定规则

**PASS**：CUDA kernel 正常结束、`VERSION=1`、`WORD_COUNT=48`、`PHASE=6`、各 action/reason/use-result 精确匹配，且 `OVERALL_PASS=1`。

**FAIL**：kernel 正常结束且 schema 有效，但任一可证伪语义 gate 为 0。host harness 返回 `GpuHostError::Verification` 并打印各 gate，不把失败包装成性能噪声。

**UNKNOWN**：PTX 构建失败、CUDA 初始化/加载/launch/synchronize 失败、入口不存在、launch 完成后 kernel 20 秒内未完成、schema version/phase 不可解释。这些结果不能证明调度语义成立或不成立。首次 PTX JIT 在实测机器上超过 90 秒；JIT/load 发生在 kernel phase 之前，只能由外层进程 timeout 保护，不能误判为语义 FAIL。kernel 超时路径不释放仍可能被 kernel 使用的 mapped memory，以避免 use-after-free；进程退出后由 CUDA/OS 回收。

**INCONCLUSIVE（只针对 wall-clock 推断）**：若 dispatch gates 通过但 wall-clock latency 抖动或重复实验分布变化，只能说明当前机器时间测量不稳定，不能据此推导硬实时。wall-clock 从来不是本测试的 pass gate。

## 10. PTX 路由与命令

入口属于 `gpu-kernel-io`，host harness 显式使用 `gpu_host::ptx::KERNEL_IO`。这同时修复本场景所需的 PTX module 路由，不改变其他历史测试的路由。

```bash
cd /home/fbsh/async-gpu
source ~/.cargo/env

# 构建唯一需要的 IO PTX，并复制到 gpu-host embed 路径
./scripts/build-kernels.sh io

# Host 构建不再隐式重建 kernel
AUTO_BUILD_KERNEL=0 cargo check -p gpu-test-harness

# 单次真实 GPU 测试；为 cold PTX JIT 预留 300 秒
timeout 300s env AUTO_BUILD_KERNEL=0 ONLY_TEST=obstacle_stress \
  cargo run -p gpu-test-harness --bin gpu-tests
```

PTX JIT cache 已热后重复 30 次检查稳定性；若 cache 被清理，应把每次外层 timeout 同样改为 300 秒：

```bash
for run in $(seq 1 30); do
  echo "obstacle stress run ${run}/30"
  timeout 60s env AUTO_BUILD_KERNEL=0 ONLY_TEST=obstacle_stress \
    cargo run -p gpu-test-harness --bin gpu-tests || exit 1
done
```

预期输出至少包含 admission、positive dispatch、Low completion、bounded non-yielding negative、decisions 和 gates 六行。若代码改动触及 IO kernel，必须先重建 IO PTX；否则旧 embedded PTX 会导致 `KernelNotFound` 或运行旧逻辑。

## 11. 可迁移结论与不可迁移结论

可迁移到后续设计：priority-aware admission、ready-queue selection、waker priority preservation、cooperative liveness 的结构性回归测试，以及 stale/deadline 决策不消费过期结果的契约测试。

不可迁移为安全承诺：本测试不能证明传感器到制动器的端到端 deadline，不能证明 non-yielding/发散/死锁 kernel 可被抢占，不能覆盖多 block、多 executor、mixed Normal 或 GPU reset，也不能替代硬件 watchdog、独立急停和系统级 hazard analysis。

## 12. 历史首轮实测证据（2026-08-15）

- 设备：NVIDIA GeForce RTX 4070，driver 595.84，compute capability 8.9。
- 首轮 `kernel_io.ptx`：3,038,167 bytes，SHA-256 `603023c75504092a8b26c2982726eac318ecb5b722b75b0c159580733c0c4ba7`，mtime `2026-08-15 00:58:56 +0800`。
- 完整命令：`/usr/bin/time -p timeout 300s env AUTO_BUILD_KERNEL=0 ONLY_TEST=obstacle_stress cargo run -p gpu-test-harness --bin gpu-tests`。
- 总时长：`real 128.99s`（其中绝大部分为 PTX JIT/load）；kernel 进入后在 harness bound 内完成。
- Admission：Low `224/224`，其中 workload 223；`ReservedCapacity` reject 1，other reject 0；High accepted 1。
- Positive dispatch：inject 225，High first 226（gap 1），High resume 227（waker gap 1）；Low `223/223` 完成，executor `225/225` 完成，共 450 dispatches。
- 诊断时间：High first-poll latency 19,456ns；负例 inject 1 → Low return 2 → High first 3，diagnostic latency 8,899,584ns。
- Decisions：fresh `(action=1, reason=1, use_gpu=1)`；stale `(2,2,0)`；deadline `(2,4,0)`。
- 最终 gates：`reserved=1 first_poll=1 waker=1 low_eventual=1 positive=1 no_midpoll=1 decisions=1 overall=1`。
- 另一次所谓“缓存复跑”以 60 秒外层 timeout 结束，`real 60.17s` 且始终无 phase；跨进程 JIT cache 未在该上限内命中，故该次严格归类 UNKNOWN，不改变最终 300 秒 run 的 PASS。

## 13. Final Phase3 同快照回归

- Final `kernel_io.ptx`：4,821,968 bytes，SHA-256
  `fb95559e93eec1c07dbb4925c4564e43263cbd55de07a208ffa6b46b51e68531`。
- 命令：`AUTO_BUILD_KERNEL=0 ONLY_TEST=obstacle_stress cargo run -p gpu-test-harness --bin gpu-tests`。
- RTX 4070：0.60 秒、exit 0；Low `224/224`，ReservedCapacity 1，
  High first/resume gap `1/1`，Low `223/223`，executor `225/225`，
  `reserved/first_poll/waker/low_eventual/positive/no_midpoll/decisions/overall`
  全部为 1。
- 机器直录日志：`.research/findings/tasks/obstacle-stress.1-phase3-final-gpu-evidence-c644.log`，
  SHA-256
  `32c504e37f4dbd9a003ac402b68c55931d879e909a133c3e1de1d997b0339661`。
- 旧 `603023c7...` 的 PASS/UNKNOWN 仍作为 JIT 与 baseline 历史保留，但所有
  “final snapshot”结论以 `fb95559e...` 为准。
