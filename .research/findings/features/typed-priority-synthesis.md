# Typed-priority synthesis

## 已落地 / Implemented
- Sealed Low/Normal/High `PriorityLevel`、closed `MayWaitFor` lattice 与 executor-issued non-cloneable `PriorityToken<P>`。
- Safe typed path 接受 Low→Low/Normal/High、Normal→Normal/High、High→High；拒绝 High→Low/Normal 与 Normal→Low。
- High 可以 fire-and-forget Low；约束的是 wait dependency，不是 spawn target。
- Caller-owned single-use completion、generation-stable handle、Release/Acquire result publication 与 exactly-once join 已落地。
- Pending typed future 自唤醒以支持 cooperative busy-repoll；priority 在 admission 后固定，异值更新返回 `PriorityFixed`。
- Live token 产生 namespace/local id/priority wire metadata；safe code 不能伪造 High token。

## 最终证据 / Final evidence
- Runtime debug/release 39/39；typed real-API/diagnostic compile matrix 13/13。
- 13 项含 6 个合法 wait edge、High spawn Low、3 个非法 downward wait、token forge、错误 receiver 反证、no_std NVPTX graph。
- Phase3 RTX 4070：真实 High forever-Pending target 进入 must-return-Pending 路径并 self-wake；shutdown 后 Low waiter依次得到 `Cancelled`、`AlreadyJoined`。
- Phase3 pending reclaims `2→1→0`、context refs `2→1→0`、active=0、spawned/completed `515/515`、teardown=true。
- 第一事件写在 poll 内、实际 return 前；证据只说明进入必返 Pending 分支，不声称 shutdown 严格晚于 poll return。
- Composed 同快照证明 live token → reserved hostcall → host event/echo → Low exactly-once join。

## 边界 / Boundary
- Token 是可移动 capability，不是 ambient whole-program identity。
- Untyped Future/channel/lock 仍在 proof 外；没有 priority inheritance 或 preemption。
- Dynamic queued update 采用明确 deferred contract；typed priority 本身固定。
- GPU 证据限定 RTX 4070 配置，不构成 wall-time、portable fairness 或 hard-real-time bound。

## 状态 / Status
CORE TYPED PRIORITY + REAL-GPU READY/PENDING/CANCEL/SHUTDOWN/EXACTLY-ONCE VERIFIED；PriML-style channel/CV 调研仍 pending。
