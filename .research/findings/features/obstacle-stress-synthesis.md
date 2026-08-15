# Obstacle-stress synthesis

- 入口：`gpu-kernel-io::obstacle_event_priority_stress`；host selector `ONLY_TEST=obstacle_stress` 显式加载 IO PTX。
- Low admission 上限 224（Normal/High 各预留 16）；第 225 个 Low 必须 `ReservedCapacity`，随后 High 仍能接纳。
- 纯 Low backlog 前提下，High first-poll gap 与 self-wake resume gap 都必须 `<=1`。
- 所有 223 个 yielding Low 最终完成；waker 按保存的 effective priority 回队。
- non-yielding Low 在 poll 内注入 High 后继续有界工作；High 只能在 Low 返回后运行，证明 priority 不是 preemption。
- fresh hazard 使用真实 GPU 结果并输出 `apply_brake_from_fresh_result`。
- stale/deadline 丢弃 GPU 结果并输出 `watchdog_conservative_stop`；二者理由与数据来源不同。
- `%globaltimer` 与 wall clock 仅诊断；本测试不连接传感器/执行器，不是 hard-real-time 或机器人放行证据。
- PASS/FAIL 使用结构化 dispatch/admission/decision gates；CUDA/timeout/schema 异常仍合法归类 UNKNOWN。
- Final PTX `fb95559e93eec1c...`（4,821,968 bytes）同快照回归：RTX 4070 0.60s、exit 0。
- 实测 Low 224/224、ReservedCapacity 1、High gap 1、wake gap 1、Low 223/223、executor 225/225、全部 gates=1。
- Final evidence：`.research/findings/tasks/obstacle-stress.1-phase3-final-gpu-evidence-c644.log`，SHA-256 `32c504e37f4dbd9a003ac402b68c55931d879e909a133c3e1de1d997b0339661`。
- 旧 `603023c7...` 的 128.99s PASS 与 60.17s UNKNOWN 保留为历史 JIT 证据，但不再是最终快照。
- 完整规格：`docs/obstacle-event-priority-stress.md`。
