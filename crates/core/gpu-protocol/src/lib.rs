//! Hostcall protocol shared definitions for GPU-host communication.
//!
//! This crate defines the wire protocol between GPU kernels running on
//! `nvptx64-nvidia-cuda` and the host CPU. It is `#![no_std]` so it compiles
//! for both targets.
//!
//! # Architecture
//!
//! Communication flows through a **hostcall buffer** — a CUDA mapped memory
//! region visible to both GPU and CPU. The buffer contains:
//!
//! 1. A **header** (64 bytes) with free/ready stacks and metadata
//! 2. A **packet pool** — fixed-size packets containing service ID + payload
//! 3. Optional **per-block shards** for reduced CAS contention at scale
//!
//! Each packet holds a 32-lane × 8-slot payload (2048 bytes), allowing a full
//! warp to communicate with the host in a single round-trip.
//!
//! # Sideband Buffer
//!
//! For bulk data transfer beyond the 56-byte packet payload limit, a separate
//! **sideband buffer** (default 1 MB) is allocated. GPU code uses a bump
//! allocator to reserve regions, then references them by offset in
//! `BULK_READ`/`BULK_WRITE` hostcalls.

#![no_std]

// ============================================================
// Service IDs
// ============================================================

/// No-op service — used for latency benchmarking.
pub const SERVICE_NOP: u32 = 0;
/// Print a message from GPU to host stdout.
pub const SERVICE_PRINT: u32 = 1;
/// Write data to an open file (up to 48 bytes inline).
pub const SERVICE_WRITE: u32 = 2;
/// Read data from an open file (up to 56 bytes inline).
pub const SERVICE_READ: u32 = 3;
/// Open a file by path, returning a file descriptor.
pub const SERVICE_OPEN: u32 = 4;
/// Close a file descriptor.
pub const SERVICE_CLOSE: u32 = 5;
/// Allocate host memory (reserved, not yet implemented).
pub const SERVICE_MALLOC: u32 = 6;
/// Free host memory (reserved, not yet implemented).
pub const SERVICE_FREE: u32 = 7;
/// Abort the GPU kernel (fatal).
pub const SERVICE_ABORT: u32 = 0xFF;
/// Read a line from host stdin.
pub const SERVICE_STDIN: u32 = 8;
/// Get current wall-clock time from host.
pub const SERVICE_TIME: u32 = 9;
/// Report a GPU panic (message + location), then trap.
pub const SERVICE_PANIC: u32 = 10;
/// Write bulk data from sideband buffer to a file.
pub const SERVICE_BULK_WRITE: u32 = 11;
/// Read bulk data from a file into sideband buffer.
pub const SERVICE_BULK_READ: u32 = 12;

// ============================================================
// TCP networking service IDs (16-23)
// ============================================================

/// Connect to a remote TCP address:port, returning a socket file descriptor.
///
/// Request (lane 0):
///   Slot 0: `port (u32) | addr_len (u32)` packed as `u64`
///   Slots 1-7: address string (up to 56 bytes, e.g. "127.0.0.1" or hostname)
///
/// Response:
///   Slot 0: socket fd (`u64`), or encoded error
pub const SERVICE_TCP_CONNECT: u32 = 16;

/// Write inline data to a TCP socket (up to 48 bytes).
///
/// Request (lane 0):
///   Slot 0: fd (`u64`)
///   Slot 1: data length (`u64`)
///   Slots 2-7: data bytes (up to 48 bytes)
///
/// Response:
///   Slot 0: bytes written (`u64`), or encoded error
pub const SERVICE_TCP_WRITE: u32 = 17;

/// Read inline data from a TCP socket (up to 56 bytes).
///
/// Request (lane 0):
///   Slot 0: fd (`u64`)
///   Slot 1: max bytes to read (`u64`)
///
/// Response:
///   Slot 0: bytes read (`u64`), or encoded error
///   Slots 1-7: data bytes (up to 56 bytes)
pub const SERVICE_TCP_READ: u32 = 18;

/// Close a TCP socket (both stream and listener).
///
/// Can also use `SERVICE_CLOSE` (5) — the fd namespace is shared between
/// files and sockets. This service ID exists for explicit type-checking
/// on the host side.
///
/// Request (lane 0):
///   Slot 0: fd (`u64`)
///
/// Response:
///   Slot 0: 0 on success, encoded error on failure
pub const SERVICE_TCP_CLOSE: u32 = 19;

/// Bind and listen on a local TCP address:port, returning a listener fd.
///
/// Request (lane 0):
///   Slot 0: `port (u32) | addr_len (u32)` packed as `u64`
///   Slots 1-7: bind address string (up to 56 bytes)
///
/// Response:
///   Slot 0: listener fd (`u64`), or encoded error
pub const SERVICE_TCP_BIND: u32 = 20;

/// Accept a connection on a TCP listener fd, returning a new stream fd.
///
/// Request (lane 0):
///   Slot 0: listener fd (`u64`)
///
/// Response:
///   Slot 0: stream fd (`u64`), or encoded error
pub const SERVICE_TCP_ACCEPT: u32 = 21;

/// Write bulk data from sideband buffer to a TCP socket.
///
/// Request (lane 0):
///   Slot 0: fd (`u64`)
///   Slot 1: sideband_offset (`u64`)
///   Slot 2: length (`u64`)
///
/// Response:
///   Slot 0: bytes written (`u64`), or encoded error
pub const SERVICE_TCP_BULK_WRITE: u32 = 22;

/// Read bulk data from a TCP socket into sideband buffer.
///
/// Request (lane 0):
///   Slot 0: fd (`u64`)
///   Slot 1: sideband_offset (`u64`)
///   Slot 2: max_length (`u64`)
///
/// Response:
///   Slot 0: bytes read (`u64`), or encoded error
pub const SERVICE_TCP_BULK_READ: u32 = 23;

/// Inline priority/identity echo used by the composed scheduler-hostcall
/// conformance test.
///
/// This service deliberately has no side effects.  The host validates the
/// generation-owned packet header, records an independent event, and writes
/// the observed identity/provenance back into the eight inline payload slots.
pub const SERVICE_PRIORITY_ECHO: u32 = 24;

/// Wire slots for [`SERVICE_PRIORITY_ECHO`].
pub mod priority_echo {
    /// Request and response slot containing a non-zero caller nonce.
    pub const NONCE: usize = 0;
    /// Response slot containing the task id read from the live packet header.
    pub const TASK_ID: usize = 1;
    /// Response slot containing the raw effective priority.
    pub const PRIORITY: usize = 2;
    /// Response slot containing the physical packet index.
    pub const PACKET_INDEX: usize = 3;
    /// Response slot: one iff the packet belongs to the shared High reserve.
    pub const SHARED_HIGH_RESERVED: usize = 4;
    /// Response slot containing the host processing sequence number.
    pub const PROCESS_SEQUENCE: usize = 5;
    /// Response slot containing the number of times this identity was seen.
    pub const PROCESS_COUNT: usize = 6;
    /// Response slot containing zero or an `ERR_*` category.
    pub const ERROR_CATEGORY: usize = 7;
    /// Number of inline `u64` slots in a packet lane.
    pub const SLOTS: usize = 8;
}

/// Named output schema shared by the composed obstacle-event GPU kernel and
/// its independent host oracle.  Keeping all offsets here prevents either
/// side from silently accepting an accidental layout change.
pub mod composed_priority_schema {
    /// Schema 版本值 / Schema version value.
    pub const VERSION_VALUE: u64 = 3;
    /// 输出字数 / Output word count.
    pub const WORD_COUNT: usize = 112;
    /// 固定任务命名空间 / Fixed task namespace.
    pub const NAMESPACE_VALUE: u64 = 0x0B57_A11E;
    /// Typed Low coordinator is the 224th admitted task.
    pub const COORDINATOR_LOCAL_ID_VALUE: u64 = 224;
    /// All task kinds share one monotonic trace sequence: 224 Low + 16 Normal.
    pub const EXPECTED_HIGH_LOCAL_ID_VALUE: u64 = 241;
    /// Packet 总数 / Total packet count.
    pub const PACKET_COUNT_VALUE: u64 = 5;
    /// General packet 数量 / General packet count.
    pub const GENERAL_PACKET_COUNT_VALUE: u64 = 4;
    /// High 预留 packet 数量 / High-reserved packet count.
    pub const HIGH_RESERVED_COUNT_VALUE: u64 = 1;
    /// Low 准入上限 / Low admission limit.
    pub const LOW_LIMIT_VALUE: u64 = 224;
    /// Normal backlog 数量 / Normal backlog size.
    pub const NORMAL_BACKLOG_VALUE: u64 = 16;
    /// High first-poll 最大 dispatch gap / Maximum High first-poll dispatch gap.
    pub const HIGH_FIRST_POLL_GAP_MAX_VALUE: u64 = 6;
    /// Schema 结束哨兵 / Schema end sentinel.
    pub const END_MAGIC_VALUE: u64 = 0x0C05_ED0E_2F1A_1644;

    /// Schema 版本字段索引 / Schema-version field index.
    pub const VERSION: usize = 0;
    /// 输出字数字段索引 / Word-count field index.
    pub const WORDS: usize = 1;
    /// Kernel 阶段字段索引 / Kernel-phase field index.
    pub const PHASE: usize = 2;
    /// 请求 nonce 字段索引 / Request-nonce field index.
    pub const NONCE: usize = 3;
    /// Task namespace 字段索引 / Task-namespace field index.
    pub const NAMESPACE: usize = 4;
    /// 预期 High local ID 字段索引 / Expected High local-ID field index.
    pub const EXPECTED_HIGH_LOCAL_ID: usize = 5;
    /// Packet 总数字段索引 / Packet-count field index.
    pub const PACKET_COUNT: usize = 6;
    /// General packet 数字段索引 / General-packet-count field index.
    pub const GENERAL_PACKET_COUNT: usize = 7;
    /// High reserve 数字段索引 / High-reserve-count field index.
    pub const HIGH_RESERVED_COUNT: usize = 8;
    /// Low 上限字段索引 / Low-limit field index.
    pub const HARD_LOW_LIMIT: usize = 9;
    /// Normal backlog 字段索引 / Normal-backlog field index.
    pub const HARD_NORMAL_BACKLOG: usize = 10;
    /// High dispatch gap 上限字段索引 / High dispatch-gap bound field index.
    pub const HIGH_FIRST_POLL_GAP_MAX: usize = 11;
    /// Freshness budget 字段索引 / Freshness-budget field index.
    pub const FRESHNESS_BUDGET_TICKS: usize = 12;
    /// Deadline budget 字段索引 / Deadline-budget field index.
    pub const DEADLINE_BUDGET_TICKS: usize = 13;
    /// 预期结束哨兵字段索引 / Expected-end-sentinel field index.
    pub const EXPECTED_END_MAGIC: usize = 14;
    /// 实际结束哨兵字段索引 / Actual-end-sentinel field index.
    pub const END_MAGIC: usize = 15;

    /// Low admitted 数字段索引 / Low-admitted-count field index.
    pub const LOW_ADMITTED: usize = 16;
    /// ReservedCapacity 拒绝数字段索引 / Reserved-capacity-rejection field index.
    pub const RESERVED_CAPACITY_REJECTIONS: usize = 17;
    /// 已取得 lease 数字段索引 / Acquired-lease-count field index.
    pub const LEASE_ACQUIRED: usize = 18;
    /// Lease packet mask 字段索引 / Lease-packet-mask field index.
    pub const LEASE_MASK: usize = 19;
    /// 存活 Normal 数字段索引 / Live-Normal-count field index.
    pub const NORMAL_ALIVE: usize = 20;
    /// General pool exhausted 标记索引 / General-pool-exhausted marker index.
    pub const GENERAL_POOL_EXHAUSTED: usize = 21;
    /// 全局 dispatch sequence 字段索引 / Global dispatch-sequence field index.
    pub const DISPATCH_SEQUENCE: usize = 22;
    /// High 注入 sequence 字段索引 / High-injection-sequence field index.
    pub const HIGH_INJECT_SEQUENCE: usize = 23;
    /// High first-poll sequence 字段索引 / High first-poll-sequence field index.
    pub const HIGH_FIRST_POLL_SEQUENCE: usize = 24;
    /// High hostcall submit sequence 字段索引 / High hostcall-submit-sequence field index.
    pub const HIGH_SUBMIT_SEQUENCE: usize = 25;
    /// High hostcall ready sequence 字段索引 / High hostcall-ready-sequence field index.
    pub const HIGH_READY_SEQUENCE: usize = 26;
    /// Low wait 起点 sequence 字段索引 / Low-wait-start-sequence field index.
    pub const LOW_WAIT_START_SEQUENCE: usize = 27;
    /// Low join sequence 字段索引 / Low-join-sequence field index.
    pub const LOW_JOIN_SEQUENCE: usize = 28;
    /// 第二次 Low join sequence 字段索引 / Second-Low-join-sequence field index.
    pub const LOW_SECOND_JOIN_SEQUENCE: usize = 29;
    /// High first-poll 时间戳字段索引 / High first-poll-timestamp field index.
    pub const HIGH_FIRST_POLL_TIMESTAMP: usize = 30;
    /// High ready 时间戳字段索引 / High-ready-timestamp field index.
    pub const HIGH_READY_TIMESTAMP: usize = 31;
    /// High Pending poll 数字段索引 / High-Pending-poll-count field index.
    pub const HIGH_PENDING_POLLS: usize = 32;
    /// 已释放 lease 数字段索引 / Released-lease-count field index.
    pub const LEASE_RELEASED: usize = 33;
    /// 已释放 Normal 数字段索引 / Released-Normal-count field index.
    pub const NORMAL_RELEASED: usize = 34;
    /// Executor spawned 数字段索引 / Executor-spawned-count field index.
    pub const EXECUTOR_SPAWNED: usize = 35;
    /// Executor completed 数字段索引 / Executor-completed-count field index.
    pub const EXECUTOR_COMPLETED: usize = 36;
    /// Ready stack 为空标记索引 / Ready-stack-empty marker index.
    pub const READY_STACK_EMPTY: usize = 37;
    /// Pool packet seen mask 字段索引 / Pool-packet-seen-mask field index.
    pub const POOL_SEEN_MASK: usize = 38;
    /// Pool packet seen-once mask 字段索引 / Pool-packet-seen-once-mask field index.
    pub const POOL_SEEN_ONCE_MASK: usize = 39;

    /// Wire task ID 字段索引 / Wire-task-ID field index.
    pub const WIRE_TASK_ID: usize = 40;
    /// Wire namespace 字段索引 / Wire-namespace field index.
    pub const WIRE_NAMESPACE: usize = 41;
    /// Wire local ID 字段索引 / Wire-local-ID field index.
    pub const WIRE_LOCAL_ID: usize = 42;
    /// Wire priority 字段索引 / Wire-priority field index.
    pub const WIRE_PRIORITY: usize = 43;
    /// GPU packet index 字段索引 / GPU-packet-index field index.
    pub const GPU_PACKET_INDEX: usize = 44;
    /// GPU shared-High provenance 字段索引 / GPU shared-High-provenance field index.
    pub const GPU_SHARED_HIGH_RESERVED: usize = 45;
    /// Echo nonce 字段索引 / Echo-nonce field index.
    pub const ECHO_NONCE: usize = 46;
    /// Echo task ID 字段索引 / Echo-task-ID field index.
    pub const ECHO_TASK_ID: usize = 47;
    /// Echo priority 字段索引 / Echo-priority field index.
    pub const ECHO_PRIORITY: usize = 48;
    /// Echo packet index 字段索引 / Echo-packet-index field index.
    pub const ECHO_PACKET_INDEX: usize = 49;
    /// Echo shared-High provenance 字段索引 / Echo shared-High-provenance field index.
    pub const ECHO_SHARED_HIGH_RESERVED: usize = 50;
    /// Host process sequence 字段索引 / Host-process-sequence field index.
    pub const HOST_PROCESS_SEQUENCE: usize = 51;
    /// Host process count 字段索引 / Host-process-count field index.
    pub const HOST_PROCESS_COUNT: usize = 52;
    /// Host error category 字段索引 / Host-error-category field index.
    pub const HOST_ERROR: usize = 53;
    /// Host event count 字段索引 / Host-event-count field index.
    pub const HOST_EVENT_COUNT: usize = 54;
    /// Host audit ready-empty 字段索引 / Host-audit-ready-empty field index.
    pub const HOST_AUDIT_READY_EMPTY: usize = 55;
    /// Host audit idle count 字段索引 / Host-audit-idle-count field index.
    pub const HOST_AUDIT_IDLE_COUNT: usize = 56;
    /// Host audit general mask 字段索引 / Host-audit-general-mask field index.
    pub const HOST_AUDIT_GENERAL_MASK: usize = 57;
    /// Host audit High mask 字段索引 / Host-audit-High-mask field index.
    pub const HOST_AUDIT_HIGH_MASK: usize = 58;
    /// Host audit duplicate count 字段索引 / Host-audit-duplicate-count field index.
    pub const HOST_AUDIT_DUPLICATES: usize = 59;
    /// Host audit missing count 字段索引 / Host-audit-missing-count field index.
    pub const HOST_AUDIT_MISSING: usize = 60;
    /// Persistent GPU clock sampled immediately before High admission.
    pub const HIGH_INJECT_TIMESTAMP: usize = 61;
    /// Raw observation marker that the async request's first poll returned Pending.
    pub const MANDATORY_FIRST_PENDING: usize = 62;
    /// Exact nonzero mutation whose designated hook reached its observation point.
    pub const MUTATION_APPLIED: usize = 63;

    /// 首次 join 成功标记索引 / First-join-success marker index.
    pub const JOIN_SUCCESS: usize = 64;
    /// 第二次 join AlreadyJoined 标记索引 / Second-join-AlreadyJoined marker index.
    pub const SECOND_JOIN_ALREADY_JOINED: usize = 65;
    /// Waiter local ID 字段索引 / Waiter-local-ID field index.
    pub const WAITER_LOCAL_ID: usize = 66;
    /// Executor final active 数字段索引 / Executor-final-active-count field index.
    pub const EXECUTOR_ACTIVE_FINAL: usize = 67;
    /// Low token consumed 标记索引 / Low-token-consumed marker index.
    pub const LOW_TOKEN_CONSUMED: usize = 68;
    /// High typed output 字段索引 / High-typed-output field index.
    pub const HIGH_OUTPUT: usize = 69;
    /// Mutation mode 字段索引 / Mutation-mode field index.
    pub const MUTATION_MODE: usize = 70;
    /// Kernel observation flags 字段索引 / Kernel-observation-flags field index.
    pub const KERNEL_OBSERVATION_FLAGS: usize = 71;

    /// Decision records 起始索引 / Decision-record base index.
    pub const DECISION_BASE: usize = 72;
    /// 每个 decision 的字数 / Words per decision record.
    pub const DECISION_WORDS: usize = 10;
    /// Decision record 数量 / Decision-record count.
    pub const DECISION_COUNT: usize = 4;
    /// Decision kind 相对索引 / Decision-kind relative index.
    pub const DECISION_KIND: usize = 0;
    /// Decision action 相对索引 / Decision-action relative index.
    pub const DECISION_ACTION: usize = 1;
    /// Use-GPU-result 相对索引 / Use-GPU-result relative index.
    pub const DECISION_USE_GPU_RESULT: usize = 2;
    /// Age source 相对索引 / Age-source relative index.
    pub const DECISION_AGE_SOURCE: usize = 3;
    /// Sample timestamp 相对索引 / Sample-timestamp relative index.
    pub const DECISION_SAMPLE_TIMESTAMP: usize = 4;
    /// Now timestamp 相对索引 / Now-timestamp relative index.
    pub const DECISION_NOW_TIMESTAMP: usize = 5;
    /// Age ticks 相对索引 / Age-ticks relative index.
    pub const DECISION_AGE_TICKS: usize = 6;
    /// Budget ticks 相对索引 / Budget-ticks relative index.
    pub const DECISION_BUDGET_TICKS: usize = 7;
    /// Hazard 标记相对索引 / Hazard-marker relative index.
    pub const DECISION_HAZARD: usize = 8;
    /// Reserved decision word 相对索引 / Reserved-decision-word relative index.
    pub const DECISION_RESERVED: usize = 9;

    /// Fresh-hazard decision ordinal / Fresh-hazard decision 序号。
    pub const DECISION_FRESH_HAZARD: usize = 0;
    /// Fresh-safe decision ordinal / Fresh-safe decision 序号。
    pub const DECISION_FRESH_SAFE: usize = 1;
    /// Stale decision ordinal / Stale decision 序号。
    pub const DECISION_STALE: usize = 2;
    /// Deadline decision ordinal / Deadline decision 序号。
    pub const DECISION_DEADLINE: usize = 3;

    /// 不制动 action / No-brake action.
    pub const ACTION_NO_BRAKE: u64 = 0;
    /// 使用 fresh GPU 结果制动 action / Apply-brake-from-fresh-result action.
    pub const ACTION_APPLY_BRAKE_FROM_FRESH_RESULT: u64 = 1;
    /// Watchdog 保守停止 action / Watchdog-conservative-stop action.
    pub const ACTION_WATCHDOG_CONSERVATIVE_STOP: u64 = 2;
    /// 真实 GPU 时钟 age source / Real-GPU-clock age source.
    pub const AGE_SOURCE_REAL_GPU_CLOCK: u64 = 1;
    /// 注入 stale age source / Injected-stale age source.
    pub const AGE_SOURCE_INJECTED_STALE: u64 = 2;
    /// 真实 first-poll latency age source / Real-first-poll-latency age source.
    pub const AGE_SOURCE_REAL_FIRST_POLL_LATENCY: u64 = 3;

    /// 计算 decision record 字段索引 / Compute a decision-record field index.
    #[inline(always)]
    pub const fn decision_word(decision: usize, field: usize) -> usize {
        DECISION_BASE + decision * DECISION_WORDS + field
    }
}

/// Named output schema for the timeout -> host reclaim -> same-packet reuse
/// GPU gate. This is deliberately separate from the 112-word composed result.
pub mod priority_echo_reuse_schema {
    /// 输出字数 / Output word count.
    pub const WORD_COUNT: usize = 16;
    /// Schema 版本值 / Schema version value.
    pub const VERSION_VALUE: u64 = 1;

    /// Schema version 字段索引 / Schema-version field index.
    pub const VERSION: usize = 0;
    /// 第一次请求 nonce 字段索引 / First-request-nonce field index.
    pub const FIRST_NONCE: usize = 1;
    /// 第二次请求 nonce 字段索引 / Second-request-nonce field index.
    pub const SECOND_NONCE: usize = 2;
    /// 第一次 packet index 字段索引 / First-packet-index field index.
    pub const FIRST_PACKET_INDEX: usize = 3;
    /// 第一次 error category 字段索引 / First-error-category field index.
    pub const FIRST_ERROR_CATEGORY: usize = 4;
    /// Reacquire busy attempt 数字段索引 / Reacquire-busy-attempt-count field index.
    pub const REACQUIRE_BUSY_ATTEMPTS: usize = 5;
    /// 第二次 packet index 字段索引 / Second-packet-index field index.
    pub const SECOND_PACKET_INDEX: usize = 6;
    /// 第二次 echo nonce 字段索引 / Second-echo-nonce field index.
    pub const SECOND_ECHO_NONCE: usize = 7;
    /// 第二次 response packet index 字段索引 / Second-response-packet-index field index.
    pub const SECOND_RESPONSE_PACKET_INDEX: usize = 8;
    /// 第二次 Pending poll 数字段索引 / Second-Pending-poll-count field index.
    pub const SECOND_PENDING_POLLS: usize = 9;
    /// Stale write 标记索引 / Stale-write marker index.
    pub const REUSE_STALE_WRITE: usize = 10;
    /// 第二次完成标记索引 / Second-completion marker index.
    pub const SECOND_COMPLETED: usize = 11;
    /// General guard packet index 字段索引 / General-guard-packet-index field index.
    pub const GENERAL_GUARD_PACKET_INDEX: usize = 12;
    /// 保留字段 0 索引 / Reserved-field-zero index.
    pub const RESERVED_0: usize = 13;
    /// 保留字段 1 索引 / Reserved-field-one index.
    pub const RESERVED_1: usize = 14;
    /// Kernel error 字段索引 / Kernel-error field index.
    pub const KERNEL_ERROR: usize = 15;
}

// ============================================================
// TCP service constants
// ============================================================

/// Maximum address string length for TCP connect/bind (7 slots x 8 bytes).
pub const TCP_MAX_ADDR_LEN: usize = 56;
/// Maximum inline write length for TCP write (6 slots x 8 bytes).
pub const TCP_MAX_WRITE_LEN: usize = 48;
/// Maximum inline read length for TCP read (7 slots x 8 bytes).
pub const TCP_MAX_READ_LEN: usize = 56;

// ============================================================
// Control bits (PacketHeader.control)
// ============================================================

/// Set by host when the response is ready for GPU to consume.
pub const CONTROL_READY: u32 = 1;
/// Set by host alongside `CONTROL_READY` to indicate an error occurred.
pub const CONTROL_ERROR: u32 = 2;
/// Set by GPU after filling the packet, before pushing to ready stack.
/// Host checks this bit before processing — skips if not set (stale re-visit).
pub const CONTROL_FILLED: u32 = 4;
/// Host has claimed a versioned request and exclusively owns its packet.
pub const CONTROL_HOST_OWNED: u32 = 8;
/// Device timeout won the cancellation race and transferred packet-release
/// responsibility to the host.
pub const CONTROL_CANCELLED: u32 = 16;
/// Low control-word bits reserved for request state and response flags.
pub const CONTROL_FLAGS_MASK: u32 = 0x1f;
/// Number of low bits occupied by [`CONTROL_FLAGS_MASK`].
pub const CONTROL_GENERATION_SHIFT: u32 = 5;
/// Number of request-generation bits stored atomically in the control word.
pub const CONTROL_GENERATION_BITS: u32 = 32 - CONTROL_GENERATION_SHIFT;
/// Complete request-generation width: 27 atomic low bits plus 16 metadata bits.
pub const REQUEST_GENERATION_BITS: u32 = CONTROL_GENERATION_BITS + 16;
/// Mask for the full 43-bit request generation.
pub const REQUEST_GENERATION_MASK: u64 = (1u64 << REQUEST_GENERATION_BITS) - 1;

/// Return the state/response flags from a versioned control word.
#[inline(always)]
pub const fn control_flags(control: u32) -> u32 {
    control & CONTROL_FLAGS_MASK
}

/// Encode a generation and state flags in the packet control word.
#[inline(always)]
pub const fn make_control(generation: u64, flags: u32) -> u32 {
    (((generation & ((1u64 << CONTROL_GENERATION_BITS) - 1)) as u32) << CONTROL_GENERATION_SHIFT)
        | (flags & CONTROL_FLAGS_MASK)
}

/// Decode the complete generation from a control word and metadata flags.
#[inline(always)]
pub const fn request_generation(control: u32, metadata_flags: u16) -> u64 {
    ((metadata_flags as u64) << CONTROL_GENERATION_BITS)
        | ((control >> CONTROL_GENERATION_SHIFT) as u64)
}

/// Return the metadata-flags portion of a complete request generation.
#[inline(always)]
pub const fn request_generation_metadata(generation: u64) -> u16 {
    ((generation & REQUEST_GENERATION_MASK) >> CONTROL_GENERATION_BITS) as u16
}

/// Advance a request generation, reserving zero for freshly initialized and
/// legacy packet slots.
#[inline(always)]
pub const fn next_request_generation(control: u32, metadata_flags: u16) -> u64 {
    let next =
        request_generation(control, metadata_flags).wrapping_add(1) & REQUEST_GENERATION_MASK;
    if next == 0 {
        1
    } else {
        next
    }
}

// ============================================================
// Hostcall priority and task metadata
// ============================================================

/// Scheduling priority carried end-to-end with a hostcall request.
///
/// The numeric order is part of the wire ABI. Consumers must decode values
/// through [`Priority::from_raw`] instead of transmuting an arbitrary byte.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// Best-effort work. It receives bounded service even under sustained load.
    Low = 0,
    /// Default priority used by all legacy APIs and packets.
    #[default]
    Normal = 1,
    /// Latency-sensitive work with reserved packet capacity.
    High = 2,
}

impl Priority {
    /// Decode a wire value, falling back to [`Priority::Normal`] for unknown
    /// values so a newer sender cannot accidentally gain elevated priority.
    #[inline(always)]
    pub const fn from_raw(value: u8) -> Self {
        match value {
            0 => Self::Low,
            2 => Self::High,
            _ => Self::Normal,
        }
    }

    /// Return the stable one-byte wire representation.
    #[inline(always)]
    pub const fn as_raw(self) -> u8 {
        self as u8
    }
}

/// Task identity and effective priority associated with one hostcall.
///
/// `task_id == 0` means that the request is not associated with an executor
/// task. Explicit trailing bytes make the C layout deterministic on both CPU
/// and GPU targets.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostcallMetadata {
    /// Executor task identifier, or zero when unavailable.
    pub task_id: u64,
    /// Priority after any runtime inheritance or aging adjustment.
    pub effective_priority: Priority,
    reserved: [u8; 7],
}

impl HostcallMetadata {
    /// Construct metadata for an executor task.
    #[inline(always)]
    pub const fn new(task_id: u64, effective_priority: Priority) -> Self {
        Self {
            task_id,
            effective_priority,
            reserved: [0; 7],
        }
    }
}

/// 保留任务命名空间（高 32 位）并替换本地标识（低 32 位）。
/// Preserve the task namespace (high 32 bits) while replacing its local ID.
#[inline(always)]
pub const fn replace_task_local_id(task_id: u64, local_id: u32) -> u64 {
    (task_id & !(u32::MAX as u64)) | local_id as u64
}

impl Default for HostcallMetadata {
    fn default() -> Self {
        Self::new(0, Priority::Normal)
    }
}

// ============================================================
// Tagged pointer constants
// ============================================================

/// Null sentinel for tagged pointers (no packet).
pub const NULL_INDEX: u16 = 0xFFFF;

// ============================================================
// Layout sizes
// ============================================================

/// Number of threads per warp (NVIDIA architecture constant).
pub const WARP_SIZE: u32 = 32;
/// Number of 8-byte slots per lane in a packet payload.
pub const SLOTS_PER_LANE: usize = 8;
/// Size of the buffer header in bytes.
pub const BUFFER_HEADER_SIZE: usize = 64;
/// Size of the packet header in bytes (before payload).
pub const PACKET_HEADER_SIZE: usize = 32;
/// Payload size: 32 lanes × 8 slots × 8 bytes = 2048 bytes.
pub const PACKET_PAYLOAD_SIZE: usize = (WARP_SIZE as usize) * SLOTS_PER_LANE * 8;
/// Total packet size: header (32) + payload (2048), rounded to 64-byte alignment = 2112.
pub const PACKET_SIZE: usize = 2112;

// ============================================================
// Buffer header field offsets (from buffer base pointer)
// ============================================================

/// Offset of the global free stack head (u64, tagged pointer).
pub const BUF_OFF_FREE_STACK: usize = 0;
/// Offset of the global ready stack head (u64, tagged pointer).
pub const BUF_OFF_READY_STACK: usize = 8;
/// Offset of the doorbell counter (u64, incremented by GPU on each push).
pub const BUF_OFF_DOORBELL: usize = 16;
/// Offset of the shutdown flag (u32, non-zero = listener should exit).
pub const BUF_OFF_SHUTDOWN: usize = 24;
/// Offset of the total packet count (u32).
pub const BUF_OFF_NUM_PACKETS: usize = 28;
/// Offset of the warp size field (u32, always 32).
pub const BUF_OFF_WARP_SIZE: usize = 32;
/// Offset of the shard count (u32, 0 = legacy single-stack mode).
pub const BUF_OFF_NUM_SHARDS: usize = 36;
/// Offset of the packets-per-shard count (u32).
pub const BUF_OFF_PKTS_PER_SHARD: usize = 40;
/// Offset of the shard array start offset (u32).
pub const BUF_OFF_SHARD_ARRAY_OFF: usize = 44;
/// Offset of the shared global priority-reserved free stack (u64, tagged pointer).
///
/// Version-zero buffers do not define this field and must ignore it.
pub const BUF_OFF_HIGH_FREE_STACK: usize = 48;
/// Offset of the hostcall protocol version (u32).
pub const BUF_OFF_PROTOCOL_VERSION: usize = 56;
/// Offset of the number each shard contributes to the shared High-only pool (u32).
/// Legacy unsharded mode is treated as one shard.
pub const BUF_OFF_HIGH_RESERVED_PER_SHARD: usize = 60;

/// First buffer protocol version with task metadata and High-only capacity.
pub const HOSTCALL_PRIORITY_PROTOCOL_VERSION: u32 = 2;
/// First buffer protocol version with generation-tagged cancellation.
pub const HOSTCALL_CANCELLATION_PROTOCOL_VERSION: u32 = 3;
/// Current buffer protocol version.
///
/// Version 3 adds a generation-tagged cancellation/ownership state machine
/// without changing any fixed header, packet, or payload size.
pub const HOSTCALL_PROTOCOL_VERSION: u32 = HOSTCALL_CANCELLATION_PROTOCOL_VERSION;
/// Priority-only metadata emitted by protocol-v2 device runtimes.
pub const PACKET_METADATA_PRIORITY_VERSION: u8 = 1;
/// Current packet metadata version. Version 2 opts into generation-tagged
/// control states and host-side return of cancelled packets.
pub const PACKET_METADATA_VERSION: u8 = 2;

/// Whether a buffer version has the priority/reserved-capacity extension.
/// Versions are matched explicitly because a future incompatible version must
/// be negotiated rather than accidentally accepted by a bare `>=` check.
#[inline(always)]
pub const fn hostcall_supports_priority(version: u32) -> bool {
    version == HOSTCALL_PRIORITY_PROTOCOL_VERSION
        || version == HOSTCALL_CANCELLATION_PROTOCOL_VERSION
}

/// Whether a buffer version supports generation-tagged cancellation.
#[inline(always)]
pub const fn hostcall_supports_cancellation(version: u32) -> bool {
    version == HOSTCALL_CANCELLATION_PROTOCOL_VERSION
}

// ============================================================
// Per-block sharding layout
// ============================================================
//
// Each shard entry is 16 bytes:
//   Offset 0: shard_free_stack  (u64) — tagged pointer
//   Offset 8: shard_ready_stack (u64) — tagged pointer

/// Size of one shard entry in bytes (free_stack + ready_stack).
pub const SHARD_ENTRY_SIZE: usize = 16;
/// Offset of the free stack within a shard entry.
pub const SHARD_OFF_FREE_STACK: usize = 0;
/// Offset of the ready stack within a shard entry.
pub const SHARD_OFF_READY_STACK: usize = 8;

/// Byte offset of shard entry `shard_idx` from buffer base.
#[inline(always)]
pub const fn shard_entry_offset(shard_array_offset: usize, shard_idx: u32) -> usize {
    shard_array_offset + (shard_idx as usize) * SHARD_ENTRY_SIZE
}

/// Byte offset of packet `index` in a sharded buffer.
///
/// Packets start after the shard array: `shard_array_offset + num_shards * SHARD_ENTRY_SIZE`.
#[inline(always)]
pub const fn packet_offset_sharded(
    index: u16,
    shard_array_offset: usize,
    num_shards: u32,
) -> usize {
    shard_array_offset + (num_shards as usize) * SHARD_ENTRY_SIZE + (index as usize) * PACKET_SIZE
}

/// Total buffer size for a sharded buffer with `num_packets` packets and `num_shards` shards.
#[inline(always)]
pub const fn buffer_size_sharded(num_packets: u16, num_shards: u32) -> usize {
    BUFFER_HEADER_SIZE
        + (num_shards as usize) * SHARD_ENTRY_SIZE
        + (num_packets as usize) * PACKET_SIZE
}

// ============================================================
// Packet field offsets (from packet base pointer)
// ============================================================

/// Offset of the next-pointer field (u64, tagged pointer for stack linkage).
pub const PKT_OFF_NEXT: usize = 0;
/// Offset of the active mask field (u32, which lanes are participating).
pub const PKT_OFF_ACTIVE_MASK: usize = 8;
/// Offset of the service ID field (u32).
pub const PKT_OFF_SERVICE: usize = 12;
/// Offset of the control flags field (u32).
pub const PKT_OFF_CONTROL: usize = 16;
/// Offset of the packet metadata version (u8).
pub const PKT_OFF_METADATA_VERSION: usize = 20;
/// Offset of the effective priority (u8; decode with [`Priority::from_raw`]).
pub const PKT_OFF_PRIORITY: usize = 21;
/// Offset of metadata flags (u16). Version-2 packets store the high 16 bits of
/// their request generation here.
pub const PKT_OFF_METADATA_FLAGS: usize = 22;
/// Offset of the executor task identifier (u64).
pub const PKT_OFF_TASK_ID: usize = 24;
/// Offset of the payload region (32 lanes × 8 slots × 8 bytes).
pub const PKT_OFF_PAYLOAD: usize = PACKET_HEADER_SIZE;

// Wire ABI assertions. These intentionally fail compilation if a future edit
// changes a shared type or moves metadata beyond the fixed-size headers.
const _: [(); 1] = [(); core::mem::size_of::<Priority>()];
const _: [(); 1] = [(); core::mem::align_of::<Priority>()];
const _: [(); 16] = [(); core::mem::size_of::<HostcallMetadata>()];
const _: [(); 8] = [(); core::mem::align_of::<HostcallMetadata>()];
const _: [(); 0] = [(); core::mem::offset_of!(HostcallMetadata, task_id)];
const _: [(); 8] = [(); core::mem::offset_of!(HostcallMetadata, effective_priority)];
const _: [(); PACKET_HEADER_SIZE] = [(); PKT_OFF_TASK_ID + core::mem::size_of::<u64>()];
const _: [(); BUFFER_HEADER_SIZE] =
    [(); BUF_OFF_HIGH_RESERVED_PER_SHARD + core::mem::size_of::<u32>()];

/// Return whether `index` belongs to the High-only packet reservation.
///
/// In sharded mode, the last `reserved_per_shard` indices of every shard are
/// reserved and linked into one global High free stack. In legacy mode, the
/// last `reserved_per_shard` indices of the whole pool are reserved.
#[inline(always)]
pub const fn is_high_reserved_packet(
    index: u16,
    num_packets: u16,
    num_shards: u32,
    pkts_per_shard: u32,
    reserved_per_shard: u32,
) -> bool {
    if reserved_per_shard == 0 {
        return false;
    }
    if num_shards == 0 {
        let reserved = if reserved_per_shard > num_packets as u32 {
            num_packets as u32
        } else {
            reserved_per_shard
        };
        return (index as u32) >= (num_packets as u32 - reserved);
    }
    if pkts_per_shard == 0 {
        return false;
    }
    let reserved = if reserved_per_shard > pkts_per_shard {
        pkts_per_shard
    } else {
        reserved_per_shard
    };
    let local_index = (index as u32) % pkts_per_shard;
    local_index >= pkts_per_shard - reserved
}

/// Total credits in the one shared global High-only pool.
///
/// In sharded mode each shard contributes `reserved_per_shard` tail packets,
/// but admission consumes them from a single global stack rather than from an
/// isolated reserve belonging to the submitting shard.
#[inline(always)]
pub const fn shared_high_reserved_packets(
    num_packets: u16,
    num_shards: u32,
    pkts_per_shard: u32,
    reserved_per_shard: u32,
) -> u32 {
    if num_shards == 0 {
        if reserved_per_shard > num_packets as u32 {
            num_packets as u32
        } else {
            reserved_per_shard
        }
    } else if pkts_per_shard == 0 {
        0
    } else {
        let contribution = if reserved_per_shard > pkts_per_shard {
            pkts_per_shard
        } else {
            reserved_per_shard
        };
        contribution.saturating_mul(num_shards)
    }
}

// ============================================================
// Tagged pointer helpers
// ============================================================
//
// Layout:
//   Bits 63..32: ABA tag (monotonically increasing)
//   Bits 31..16: reserved (zero)
//   Bits 15..0:  packet index (0..N-1), or 0xFFFF = NULL

/// Construct the next stack head for any push, pop, or whole-chain drain.
/// Every successful head mutation advances the tag even when the new index was
/// cached in an older packet `next` word.
#[inline(always)]
pub const fn advance_tagged_head(current: u64, new_index: u16) -> u64 {
    make_tagged(tagged_tag(current).wrapping_add(1), new_index)
}

/// Extract the packet index from a tagged pointer.
///
/// ```
/// use gpu_protocol::{make_tagged, tagged_index};
/// assert_eq!(tagged_index(make_tagged(5, 42)), 42);
/// ```
#[inline(always)]
pub const fn tagged_index(tagged: u64) -> u16 {
    (tagged & 0xFFFF) as u16
}

/// Extract the ABA tag from a tagged pointer.
///
/// ```
/// use gpu_protocol::{make_tagged, tagged_tag};
/// assert_eq!(tagged_tag(make_tagged(5, 42)), 5);
/// ```
#[inline(always)]
pub const fn tagged_tag(tagged: u64) -> u32 {
    (tagged >> 32) as u32
}

/// Construct a tagged pointer from a tag and packet index.
///
/// ```
/// use gpu_protocol::make_tagged;
/// let ptr = make_tagged(1, 0);
/// assert_eq!(ptr, 0x0000_0001_0000_0000);
/// ```
#[inline(always)]
pub const fn make_tagged(tag: u32, index: u16) -> u64 {
    ((tag as u64) << 32) | (index as u64)
}

/// Construct a null tagged pointer (index = `NULL_INDEX`).
///
/// ```
/// use gpu_protocol::{null_tagged, tagged_index, NULL_INDEX};
/// assert_eq!(tagged_index(null_tagged()), NULL_INDEX);
/// ```
#[inline(always)]
pub const fn null_tagged() -> u64 {
    make_tagged(0, NULL_INDEX)
}

// ============================================================
// Offset calculators
// ============================================================

/// Byte offset of packet `index` from buffer base (legacy non-sharded layout).
#[inline(always)]
pub const fn packet_offset(index: u16) -> usize {
    BUFFER_HEADER_SIZE + (index as usize) * PACKET_SIZE
}

/// Byte offset of a specific payload slot from the packet base.
///
/// `lane` is the warp lane (0..31), `slot` is the slot index (0..7).
#[inline(always)]
pub const fn payload_slot_offset(lane: u32, slot: usize) -> usize {
    PKT_OFF_PAYLOAD + (lane as usize) * SLOTS_PER_LANE * 8 + slot * 8
}

/// Total buffer size in bytes for `num_packets` packets (legacy non-sharded layout).
#[inline(always)]
pub const fn buffer_size(num_packets: u16) -> usize {
    BUFFER_HEADER_SIZE + (num_packets as usize) * PACKET_SIZE
}

// ============================================================
// PRINT service payload layout (lane 0)
// ============================================================

/// Maximum message length for `SERVICE_PRINT` (7 slots × 8 bytes).
pub const PRINT_MAX_MSG_LEN: usize = 56;

// ============================================================
// FILE I/O service payload layouts (lane 0)
// ============================================================
//
// SERVICE_OPEN request:
//   Slot 0: flags (u64) — see FILE_OPEN_* constants
//   Slot 1: path length (u64)
//   Slots 2-7: path bytes (up to 48 bytes)
// SERVICE_OPEN response:
//   Slot 0: file descriptor (u64), or FILE_ERROR_SENTINEL on error
//
// SERVICE_WRITE request:
//   Slot 0: fd (u64)
//   Slot 1: data length (u64)
//   Slots 2-7: data bytes (up to 48 bytes)
// SERVICE_WRITE response:
//   Slot 0: bytes written (u64), or FILE_ERROR_SENTINEL on error
//
// SERVICE_READ request:
//   Slot 0: fd (u64)
//   Slot 1: max bytes to read (u64)
// SERVICE_READ response:
//   Slot 0: bytes read (u64), or FILE_ERROR_SENTINEL on error
//   Slots 1-7: data bytes (up to 56 bytes)
//
// SERVICE_CLOSE request:
//   Slot 0: fd (u64)
// SERVICE_CLOSE response:
//   Slot 0: 0 on success, FILE_ERROR_SENTINEL on error

/// Maximum path length for `SERVICE_OPEN` (7 slots × 8 bytes).
pub const FILE_MAX_PATH_LEN: usize = 56;
/// Maximum inline write length for `SERVICE_WRITE` (6 slots × 8 bytes).
pub const FILE_MAX_WRITE_LEN: usize = 48;
/// Maximum inline read length for `SERVICE_READ` (7 slots × 8 bytes).
pub const FILE_MAX_READ_LEN: usize = 56;
/// Sentinel value indicating a file I/O error in response slot 0.
pub const FILE_ERROR_SENTINEL: u64 = u64::MAX;

/// Open a file for reading.
pub const FILE_OPEN_READ: u32 = 0;
/// Open a file for writing (create if not exists, truncate if exists).
pub const FILE_OPEN_WRITE_CREATE: u32 = 1;
/// Open a file for appending.
pub const FILE_OPEN_APPEND: u32 = 2;

// ============================================================
// Error encoding (structured error propagation)
// ============================================================
//
// When CONTROL_ERROR is set in the response, payload slot 0 contains:
//   Bits 63..32: reserved (zero)
//   Bits 31..16: raw OS errno (optional, 0 = not provided)
//   Bits 15..0:  error category (one of the ERR_* constants)

/// Generic/unknown error.
pub const ERR_OTHER: u16 = 0;
/// File or resource not found.
pub const ERR_NOT_FOUND: u16 = 1;
/// Permission denied.
pub const ERR_PERMISSION_DENIED: u16 = 2;
/// Resource already exists.
pub const ERR_ALREADY_EXISTS: u16 = 3;
/// Invalid input parameter.
pub const ERR_INVALID_INPUT: u16 = 4;
/// General I/O error.
pub const ERR_IO_ERROR: u16 = 5;
/// Operation timed out.
pub const ERR_TIMED_OUT: u16 = 6;
/// Operation would block (non-blocking mode).
pub const ERR_WOULD_BLOCK: u16 = 7;
/// Broken pipe.
pub const ERR_BROKEN_PIPE: u16 = 8;
/// Resource is busy.
pub const ERR_RESOURCE_BUSY: u16 = 9;
/// Storage is full.
pub const ERR_STORAGE_FULL: u16 = 10;
/// Too many open files.
pub const ERR_TOO_MANY_FILES: u16 = 11;
/// Out of memory.
pub const ERR_OUT_OF_MEMORY: u16 = 12;
/// Invalid file descriptor.
pub const ERR_INVALID_FD: u16 = 13;
/// Connection refused (reserved for future networking).
pub const ERR_CONNECTION_REFUSED: u16 = 14;
/// Path is a directory, not a file.
pub const ERR_IS_A_DIRECTORY: u16 = 15;
/// Host-side timeout processing the request.
pub const ERR_HOST_TIMEOUT: u16 = 16;
/// Operation not supported.
pub const ERR_UNSUPPORTED: u16 = 17;
/// TCP connection reset by peer.
pub const ERR_CONNECTION_RESET: u16 = 18;
/// TCP bind address already in use.
pub const ERR_ADDR_IN_USE: u16 = 19;
/// TCP bind address not available.
pub const ERR_ADDR_NOT_AVAILABLE: u16 = 20;
/// Socket is not connected.
pub const ERR_NOT_CONNECTED: u16 = 21;

/// Encode an error category and raw errno into payload slot 0 format.
///
/// ```
/// use gpu_protocol::{encode_error, error_category, error_raw_errno, ERR_NOT_FOUND};
/// let encoded = encode_error(ERR_NOT_FOUND, 2);
/// assert_eq!(error_category(encoded), ERR_NOT_FOUND);
/// assert_eq!(error_raw_errno(encoded), 2);
/// ```
#[inline(always)]
pub const fn encode_error(category: u16, raw_errno: u16) -> u64 {
    ((raw_errno as u64) << 16) | (category as u64)
}

/// Decode the error category from an encoded error value.
#[inline(always)]
pub const fn error_category(slot0: u64) -> u16 {
    (slot0 & 0xFFFF) as u16
}

/// Decode the raw OS errno from an encoded error value.
#[inline(always)]
pub const fn error_raw_errno(slot0: u64) -> u16 {
    ((slot0 >> 16) & 0xFFFF) as u16
}

// ============================================================
// GPU-side error types for Result-based error propagation
// ============================================================

/// GPU-side error returned by hostcall helpers.
///
/// Encodes an error category (from the ERR_* constants) and an optional
/// OS errno. Small enough (4 bytes) to be returned by value in Result.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuError {
    /// Error category (one of the ERR_* constants).
    pub category: u16,
    /// Raw OS errno from the host, or 0 if not applicable.
    pub raw_errno: u16,
}

impl GpuError {
    /// Create a new GpuError from category and errno.
    #[inline(always)]
    pub const fn new(category: u16, raw_errno: u16) -> Self {
        Self {
            category,
            raw_errno,
        }
    }

    /// Create a GpuError from an encoded error value (payload slot 0 format).
    #[inline(always)]
    pub const fn from_encoded(slot0: u64) -> Self {
        Self {
            category: error_category(slot0),
            raw_errno: error_raw_errno(slot0),
        }
    }

    /// Pool exhaustion — no free packets available.
    #[inline(always)]
    pub const fn pool_exhausted() -> Self {
        Self::new(ERR_RESOURCE_BUSY, 0)
    }

    /// Timeout waiting for host response.
    #[inline(always)]
    pub const fn timeout() -> Self {
        Self::new(ERR_HOST_TIMEOUT, 0)
    }

    /// The connected hostcall buffer predates a required wire capability.
    #[inline(always)]
    pub const fn unsupported() -> Self {
        Self::new(ERR_UNSUPPORTED, 0)
    }
}

/// Sentinel values for GpuKernelResult tag field.
pub const TAG_OK: u32 = 0;
/// Kernel returned an error.
pub const TAG_ERR: u32 = 1;
/// Buffer not yet written — kernel may have crashed.
pub const TAG_UNINIT: u32 = 0xDEAD_BEEF;

/// GPU kernel result buffer — 64 bytes, one cache line.
///
/// Passed as the last kernel parameter. Host allocates mapped memory,
/// initializes tag to TAG_UNINIT, launches kernel, then reads result
/// after synchronization.
///
/// Layout:
/// ```text
/// Offset  0: tag        (u32)  — TAG_OK, TAG_ERR, or TAG_UNINIT
/// Offset  4: category   (u16)  — ERR_* constant
/// Offset  6: raw_errno  (u16)  — OS errno
/// Offset  8: thread_idx (u16)  — threadIdx.x that produced error
/// Offset 10: block_idx  (u16)  — blockIdx.x that produced error
/// Offset 12: msg_len    (u32)  — message byte count (0..48)
/// Offset 16: msg_bytes  [48]   — UTF-8 message (truncated if needed)
/// ```
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct GpuKernelResult {
    /// TAG_OK, TAG_ERR, or TAG_UNINIT.
    pub tag: u32,
    /// ERR_* constant identifying the error category.
    pub category: u16,
    /// OS errno value from the GPU side.
    pub raw_errno: u16,
    /// `threadIdx.x` that produced this result.
    pub thread_idx: u16,
    /// `blockIdx.x` that produced this result.
    pub block_idx: u16,
    /// Byte count of the message in `msg_bytes` (0..48).
    pub msg_len: u32,
    /// UTF-8 message bytes (truncated to 48 bytes if needed).
    pub msg_bytes: [u8; 48],
}

impl GpuKernelResult {
    /// Create an uninitialized result (sentinel value).
    #[inline(always)]
    pub const fn uninit() -> Self {
        Self {
            tag: TAG_UNINIT,
            category: 0,
            raw_errno: 0,
            thread_idx: 0,
            block_idx: 0,
            msg_len: 0,
            msg_bytes: [0u8; 48],
        }
    }

    /// Write OK status.
    #[inline(always)]
    pub fn set_ok(&mut self) {
        self.tag = TAG_OK;
    }

    /// Write error status with a GpuError and optional message.
    #[inline(always)]
    pub fn set_err(&mut self, err: GpuError, thread_idx: u16, block_idx: u16, msg: &[u8]) {
        self.tag = TAG_ERR;
        self.category = err.category;
        self.raw_errno = err.raw_errno;
        self.thread_idx = thread_idx;
        self.block_idx = block_idx;
        let len = if msg.len() > 48 { 48 } else { msg.len() };
        self.msg_len = len as u32;
        let mut i = 0;
        while i < len {
            self.msg_bytes[i] = msg[i];
            i += 1;
        }
    }

    /// Check if the result indicates success.
    #[inline(always)]
    pub const fn is_ok(&self) -> bool {
        self.tag == TAG_OK
    }

    /// Check if the result indicates an error.
    #[inline(always)]
    pub const fn is_err(&self) -> bool {
        self.tag == TAG_ERR
    }

    /// Get the error message as a byte slice.
    #[inline(always)]
    pub fn message(&self) -> &[u8] {
        let len = if (self.msg_len as usize) > 48 {
            48
        } else {
            self.msg_len as usize
        };
        &self.msg_bytes[..len]
    }
}

// ============================================================
// PANIC service payload layout (lane 0)
// ============================================================
//
// Request:
//   Slot 0: metadata (u64)
//     - Bits 15..0:  threadIdx.x (u16)
//     - Bits 31..16: blockIdx.x (u16)
//     - Bits 47..32: message length (u16)
//     - Bits 63..48: reserved (zero)
//   Slots 1-7: panic message bytes (up to 56 bytes, truncated)
// Response:
//   (CONTROL_READY set — GPU thread will trap regardless)

/// Maximum panic message length (7 slots × 8 bytes).
pub const PANIC_MAX_MSG_LEN: usize = 56;

/// Encode panic metadata: thread index, block index, and message length.
///
/// ```
/// use gpu_protocol::{encode_panic_metadata, panic_thread_idx, panic_block_idx, panic_msg_len};
/// let meta = encode_panic_metadata(5, 3, 20);
/// assert_eq!(panic_thread_idx(meta), 5);
/// assert_eq!(panic_block_idx(meta), 3);
/// assert_eq!(panic_msg_len(meta), 20);
/// ```
#[inline(always)]
pub const fn encode_panic_metadata(thread_idx: u16, block_idx: u16, msg_len: u16) -> u64 {
    (thread_idx as u64) | ((block_idx as u64) << 16) | ((msg_len as u64) << 32)
}

/// Decode `threadIdx.x` from panic metadata.
#[inline(always)]
pub const fn panic_thread_idx(meta: u64) -> u16 {
    (meta & 0xFFFF) as u16
}

/// Decode `blockIdx.x` from panic metadata.
#[inline(always)]
pub const fn panic_block_idx(meta: u64) -> u16 {
    ((meta >> 16) & 0xFFFF) as u16
}

/// Decode message length from panic metadata.
#[inline(always)]
pub const fn panic_msg_len(meta: u64) -> u16 {
    ((meta >> 32) & 0xFFFF) as u16
}

// ============================================================
// Sideband buffer (bulk data transfer)
// ============================================================
//
// The sideband buffer is a separate CUDA mapped allocation used for
// transferring data larger than the 56-byte packet payload limit.
//
// Header (64 bytes):
//   Offset 0:  alloc_offset (u64) — GPU bump allocator position
//   Offset 8:  capacity (u64)     — total data region size in bytes
//   Offset 16: reserved (48 bytes)
//
// Data Region:
//   Starts at SIDEBAND_HEADER_SIZE from sideband base
//   Contiguous byte array, `capacity` bytes
//
// SERVICE_BULK_WRITE request (lane 0):
//   Slot 0: fd (u64)
//   Slot 1: sideband_offset (u64) — offset within data region
//   Slot 2: length (u64)
//   Response slot 0: bytes_written (u64), or FILE_ERROR_SENTINEL on error
//
// SERVICE_BULK_READ request (lane 0):
//   Slot 0: fd (u64)
//   Slot 1: sideband_offset (u64)
//   Slot 2: max_length (u64)
//   Response slot 0: bytes_read (u64), or FILE_ERROR_SENTINEL on error

/// Size of the sideband header in bytes.
pub const SIDEBAND_HEADER_SIZE: usize = 64;
/// Offset of the bump allocator position within the sideband header.
pub const SIDEBAND_OFF_ALLOC: usize = 0;
/// Offset of the capacity field within the sideband header.
pub const SIDEBAND_OFF_CAPACITY: usize = 8;
/// Byte offset where the sideband data region begins.
pub const SIDEBAND_DATA_OFFSET: usize = SIDEBAND_HEADER_SIZE;

/// Default sideband data region size: 1 MB.
pub const DEFAULT_SIDEBAND_SIZE: usize = 1024 * 1024;

// ============================================================
// GPU-side spin limit
// ============================================================

/// Maximum number of poll iterations before the GPU executor traps.
///
/// With a 64ns nanosleep between polls, this gives ~640ms before timeout.
pub const GPU_MAX_SPIN: u32 = 10_000_000;

// ============================================================
// STDIN service payload layout (lane 0)
// ============================================================

/// Maximum bytes readable from stdin in one hostcall (7 slots × 8 bytes).
pub const STDIN_MAX_READ_LEN: usize = 56;

// ============================================================
// TIME service payload layout (lane 0)
// ============================================================
// Request: (no payload needed)
// Response:
//   Slot 0: seconds since Unix epoch (u64)
//   Slot 1: nanoseconds within second (u64)

// ============================================================
// TRACE service — structured GPU trace events
// ============================================================

/// Emit a structured trace event from GPU to host.
///
/// Unlike `SERVICE_PRINT` (plain text), trace events carry structured
/// metadata: thread/block coordinates, severity level, and a GPU
/// timestamp. The host can collect, sort, and filter events.
pub const SERVICE_TRACE: u32 = 13;

/// Flush buffered print messages from sideband buffer.
///
/// GPU-side code accumulates multiple print messages in a per-thread
/// sideband buffer slot, then flushes them all in a single hostcall.
/// Messages are length-prefixed: `[u16 len][len bytes data]...`
///
/// Request (lane 0):
///   Slot 0: sideband_offset (u64) — start of this thread's buffer data in sideband
///   Slot 1: data_len (u64) — total bytes of length-prefixed messages
///   Slot 2: thread_idx (u32) | block_idx (u32) — packed metadata
///
/// Response: none (fire-and-forget, host sets CONTROL_READY).
pub const SERVICE_BULK_PRINT: u32 = 15;

// ============================================================
// TRACE service payload layout (lane 0)
// ============================================================
//
// Request:
//   Slot 0: metadata (u64)
//     - Bits 15..0:  threadIdx.x (u16)
//     - Bits 31..16: blockIdx.x (u16)
//     - Bits 39..32: trace level (u8) — see TRACE_LEVEL_* constants
//     - Bits 47..40: message length (u8, 0..48)
//     - Bits 63..48: warp lane ID (u16)
//   Slot 1: GPU timestamp (u64) — from %clock64 or %globaltimer
//   Slots 2-7: message bytes (up to 48 bytes, UTF-8)
// Response:
//   (CONTROL_READY set — no response data needed, fire-and-forget)

/// Trace level: debug (verbose, lowest priority).
pub const TRACE_LEVEL_DEBUG: u8 = 0;
/// Trace level: info (normal events).
pub const TRACE_LEVEL_INFO: u8 = 1;
/// Trace level: warn (potential issues).
pub const TRACE_LEVEL_WARN: u8 = 2;
/// Trace level: error (failures that don't trap).
pub const TRACE_LEVEL_ERROR: u8 = 3;

/// Maximum trace message length (6 slots × 8 bytes).
pub const TRACE_MAX_MSG_LEN: usize = 48;

/// Encode trace metadata: thread index, block index, level, msg_len, lane ID.
///
/// ```
/// use gpu_protocol::*;
/// let meta = encode_trace_metadata(5, 3, TRACE_LEVEL_INFO, 20, 0);
/// assert_eq!(trace_thread_idx(meta), 5);
/// assert_eq!(trace_block_idx(meta), 3);
/// assert_eq!(trace_level(meta), TRACE_LEVEL_INFO);
/// assert_eq!(trace_msg_len(meta), 20);
/// assert_eq!(trace_lane_id(meta), 0);
/// ```
#[inline(always)]
pub const fn encode_trace_metadata(
    thread_idx: u16,
    block_idx: u16,
    level: u8,
    msg_len: u8,
    lane_id: u16,
) -> u64 {
    (thread_idx as u64)
        | ((block_idx as u64) << 16)
        | ((level as u64) << 32)
        | ((msg_len as u64) << 40)
        | ((lane_id as u64) << 48)
}

/// Decode `threadIdx.x` from trace metadata.
#[inline(always)]
pub const fn trace_thread_idx(meta: u64) -> u16 {
    (meta & 0xFFFF) as u16
}

/// Decode `blockIdx.x` from trace metadata.
#[inline(always)]
pub const fn trace_block_idx(meta: u64) -> u16 {
    ((meta >> 16) & 0xFFFF) as u16
}

/// Decode trace level from trace metadata.
#[inline(always)]
pub const fn trace_level(meta: u64) -> u8 {
    ((meta >> 32) & 0xFF) as u8
}

/// Decode message length from trace metadata.
#[inline(always)]
pub const fn trace_msg_len(meta: u64) -> u8 {
    ((meta >> 40) & 0xFF) as u8
}

/// Decode warp lane ID from trace metadata.
#[inline(always)]
pub const fn trace_lane_id(meta: u64) -> u16 {
    ((meta >> 48) & 0xFFFF) as u16
}

/// Decode all fields from trace metadata in one call.
///
/// Returns `(thread_idx, block_idx, level, msg_len, lane_id)`.
#[inline(always)]
pub const fn decode_trace_metadata(meta: u64) -> (u16, u16, u8, u8, u16) {
    (
        trace_thread_idx(meta),
        trace_block_idx(meta),
        trace_level(meta),
        trace_msg_len(meta),
        trace_lane_id(meta),
    )
}

// ================================================================
// Command Buffer Protocol — Host→GPU command channel
// ================================================================

/// Command buffer header size (64 bytes, cache-line aligned).
pub const CMD_BUF_HEADER_SIZE: usize = 64;

/// Command slot size (64 bytes each).
pub const CMD_SLOT_SIZE: usize = 64;

/// Offset of `write_idx` (u64, atomic) in command buffer header.
/// Host increments after writing a command. GPU reads with Acquire.
pub const CMD_OFF_WRITE_IDX: usize = 0;

/// Offset of `read_idx` (u64, atomic) in command buffer header.
/// GPU increments after processing a command. Host reads for backpressure.
pub const CMD_OFF_READ_IDX: usize = 8;

/// Offset of `capacity` (u32) in command buffer header.
pub const CMD_OFF_CAPACITY: usize = 16;

/// Offset of `cmd_type` (u32) within a command slot.
pub const CMD_SLOT_OFF_TYPE: usize = 0;

/// Offset of payload within a command slot (56 bytes available).
pub const CMD_SLOT_OFF_PAYLOAD: usize = 8;

/// Maximum payload size per command slot.
pub const CMD_MAX_PAYLOAD: usize = CMD_SLOT_SIZE - CMD_SLOT_OFF_PAYLOAD;

/// Command type: no-op (for testing).
pub const CMD_NOP: u32 = 0;

/// Command type: execute computation on device buffer.
/// Payload: input_ptr (u64), output_ptr (u64), count (u32), op_code (u32).
pub const CMD_COMPUTE: u32 = 1;

/// Command type: print a message via hostcall.
/// Payload: msg_len (u32) + message bytes (up to 52 bytes).
pub const CMD_PRINT: u32 = 2;

/// Command type: exit the command processing loop.
pub const CMD_EXIT: u32 = 3;

// ================================================================
// Flight Recorder — GPU-side ring buffer for post-mortem trace events
// ================================================================

/// Size of the flight recorder header in bytes.
pub const FR_HEADER_SIZE: usize = 64;

/// Size of each flight recorder event slot in bytes.
pub const FR_SLOT_SIZE: usize = 64;

/// Offset of `write_idx` (u64, atomic) in the flight recorder header.
pub const FR_OFF_WRITE_IDX: usize = 0;

/// Offset of `capacity` (u32) in the flight recorder header.
pub const FR_OFF_CAPACITY: usize = 8;

/// Offset of `flags` (u32) in the flight recorder header.
pub const FR_OFF_FLAGS: usize = 12;

/// Offset of metadata (u64) in a flight recorder event slot.
pub const FR_SLOT_OFF_META: usize = 0;

/// Offset of timestamp (u64) in a flight recorder event slot.
pub const FR_SLOT_OFF_TIMESTAMP: usize = 8;

/// Offset of message bytes in a flight recorder event slot.
pub const FR_SLOT_OFF_MSG: usize = 16;

/// Maximum message length in a flight recorder event.
pub const FR_MAX_MSG_LEN: usize = FR_SLOT_SIZE - FR_SLOT_OFF_MSG;

/// Flag bit: kernel crashed (set by GPU before trap).
pub const FR_FLAG_CRASHED: u32 = 1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_wire_values_and_fallback_are_stable() {
        assert_eq!(Priority::Low.as_raw(), 0);
        assert_eq!(Priority::Normal.as_raw(), 1);
        assert_eq!(Priority::High.as_raw(), 2);
        assert_eq!(Priority::from_raw(0), Priority::Low);
        assert_eq!(Priority::from_raw(1), Priority::Normal);
        assert_eq!(Priority::from_raw(2), Priority::High);
        assert_eq!(Priority::from_raw(255), Priority::Normal);
        assert_eq!(Priority::default(), Priority::Normal);
    }

    #[test]
    fn hostcall_metadata_layout_is_fixed() {
        assert_eq!(core::mem::size_of::<Priority>(), 1);
        assert_eq!(core::mem::align_of::<Priority>(), 1);
        assert_eq!(core::mem::size_of::<HostcallMetadata>(), 16);
        assert_eq!(core::mem::align_of::<HostcallMetadata>(), 8);
        assert_eq!(core::mem::offset_of!(HostcallMetadata, task_id), 0);
        assert_eq!(
            core::mem::offset_of!(HostcallMetadata, effective_priority),
            8
        );
        assert_eq!(PKT_OFF_METADATA_VERSION, 20);
        assert_eq!(PKT_OFF_PRIORITY, 21);
        assert_eq!(PKT_OFF_METADATA_FLAGS, 22);
        assert_eq!(PKT_OFF_TASK_ID, 24);
        assert_eq!(
            PKT_OFF_TASK_ID + core::mem::size_of::<u64>(),
            PKT_OFF_PAYLOAD
        );
        assert_eq!(PACKET_HEADER_SIZE, 32);
        assert_eq!(PACKET_SIZE, 2112);
        assert_eq!(BUFFER_HEADER_SIZE, 64);
    }

    #[test]
    fn versioned_control_generation_round_trips_without_layout_growth() {
        assert_eq!(HOSTCALL_PRIORITY_PROTOCOL_VERSION, 2);
        assert_eq!(HOSTCALL_CANCELLATION_PROTOCOL_VERSION, 3);
        assert_eq!(HOSTCALL_PROTOCOL_VERSION, 3);
        assert_eq!(PACKET_METADATA_PRIORITY_VERSION, 1);
        assert_eq!(PACKET_METADATA_VERSION, 2);
        assert!(hostcall_supports_priority(2));
        assert!(hostcall_supports_priority(3));
        assert!(!hostcall_supports_priority(4));
        assert!(hostcall_supports_cancellation(3));
        assert!(!hostcall_supports_cancellation(2));
        assert!(!hostcall_supports_cancellation(4));

        let generations = [1, 17, (1 << 27) - 1, 1 << 27, REQUEST_GENERATION_MASK];
        for generation in generations {
            let metadata = request_generation_metadata(generation);
            let control = make_control(generation, CONTROL_HOST_OWNED);
            assert_eq!(control_flags(control), CONTROL_HOST_OWNED);
            assert_eq!(request_generation(control, metadata), generation);
        }

        let wrapped = next_request_generation(
            make_control(REQUEST_GENERATION_MASK, 0),
            request_generation_metadata(REQUEST_GENERATION_MASK),
        );
        assert_eq!(wrapped, 1);
        assert_eq!(
            make_control(9, CONTROL_READY | CONTROL_ERROR) & CONTROL_FLAGS_MASK,
            3
        );
    }

    #[test]
    fn every_stack_head_mutation_advances_tag_and_blocks_short_aba() {
        // T1 snapshots A(tag=7) -> B. T2 pops A and pushes A again before
        // T1's CAS. The index returns to A, but the tag advances twice.
        let t1_old = make_tagged(7, 4);
        let cached_next = make_tagged(2, 9);
        let after_t2_pop = advance_tagged_head(t1_old, tagged_index(cached_next));
        let after_t2_push = advance_tagged_head(after_t2_pop, 4);

        assert_eq!(tagged_index(after_t2_push), tagged_index(t1_old));
        assert_ne!(after_t2_push, t1_old);
        assert_eq!(tagged_tag(after_t2_pop), 8);
        assert_eq!(tagged_tag(after_t2_push), 9);

        let t1_pop_target = advance_tagged_head(t1_old, tagged_index(cached_next));
        assert_ne!(after_t2_push, t1_pop_target);
    }

    #[test]
    fn metadata_constructor_round_trips_values() {
        let metadata = HostcallMetadata::new(0x0123_4567_89ab_cdef, Priority::High);
        assert_eq!(metadata.task_id, 0x0123_4567_89ab_cdef);
        assert_eq!(metadata.effective_priority, Priority::High);

        let legacy = HostcallMetadata::default();
        assert_eq!(legacy.task_id, 0);
        assert_eq!(legacy.effective_priority, Priority::Normal);
    }

    #[test]
    fn replacing_task_local_id_preserves_namespace() {
        let task_id = (0x0B57_A11Eu64 << 32) | 241;
        let corrupted = replace_task_local_id(task_id, 0);

        assert_eq!(corrupted, 0x0B57_A11Eu64 << 32);
        assert_eq!(corrupted >> 32, 0x0B57_A11E);
        assert_eq!(corrupted as u32, 0);
    }

    #[test]
    fn high_reservation_covers_each_shard_tail() {
        for index in 0..12u16 {
            let expected = matches!(index, 3 | 7 | 11);
            assert_eq!(
                is_high_reserved_packet(index, 12, 3, 4, 1),
                expected,
                "index {index}"
            );
        }
        assert!(is_high_reserved_packet(7, 8, 0, 0, 1));
        assert!(!is_high_reserved_packet(6, 8, 0, 0, 1));
        assert!(!is_high_reserved_packet(7, 8, 0, 0, 0));
        assert_eq!(shared_high_reserved_packets(12, 3, 4, 1), 3);
        assert_eq!(shared_high_reserved_packets(8, 0, 0, 2), 2);
    }

    #[test]
    fn composed_priority_schema_is_exact_and_non_overlapping() {
        use composed_priority_schema as schema;

        assert_eq!(SERVICE_PRIORITY_ECHO, 24);
        assert_eq!(priority_echo::SLOTS, 8);
        assert_eq!(schema::VERSION_VALUE, 3);
        let fields = [
            schema::VERSION,
            schema::WORDS,
            schema::PHASE,
            schema::NONCE,
            schema::NAMESPACE,
            schema::EXPECTED_HIGH_LOCAL_ID,
            schema::PACKET_COUNT,
            schema::GENERAL_PACKET_COUNT,
            schema::HIGH_RESERVED_COUNT,
            schema::HARD_LOW_LIMIT,
            schema::HARD_NORMAL_BACKLOG,
            schema::HIGH_FIRST_POLL_GAP_MAX,
            schema::FRESHNESS_BUDGET_TICKS,
            schema::DEADLINE_BUDGET_TICKS,
            schema::EXPECTED_END_MAGIC,
            schema::END_MAGIC,
            schema::LOW_ADMITTED,
            schema::RESERVED_CAPACITY_REJECTIONS,
            schema::LEASE_ACQUIRED,
            schema::LEASE_MASK,
            schema::NORMAL_ALIVE,
            schema::GENERAL_POOL_EXHAUSTED,
            schema::DISPATCH_SEQUENCE,
            schema::HIGH_INJECT_SEQUENCE,
            schema::HIGH_FIRST_POLL_SEQUENCE,
            schema::HIGH_SUBMIT_SEQUENCE,
            schema::HIGH_READY_SEQUENCE,
            schema::LOW_WAIT_START_SEQUENCE,
            schema::LOW_JOIN_SEQUENCE,
            schema::LOW_SECOND_JOIN_SEQUENCE,
            schema::HIGH_FIRST_POLL_TIMESTAMP,
            schema::HIGH_READY_TIMESTAMP,
            schema::HIGH_PENDING_POLLS,
            schema::LEASE_RELEASED,
            schema::NORMAL_RELEASED,
            schema::EXECUTOR_SPAWNED,
            schema::EXECUTOR_COMPLETED,
            schema::READY_STACK_EMPTY,
            schema::POOL_SEEN_MASK,
            schema::POOL_SEEN_ONCE_MASK,
            schema::WIRE_TASK_ID,
            schema::WIRE_NAMESPACE,
            schema::WIRE_LOCAL_ID,
            schema::WIRE_PRIORITY,
            schema::GPU_PACKET_INDEX,
            schema::GPU_SHARED_HIGH_RESERVED,
            schema::ECHO_NONCE,
            schema::ECHO_TASK_ID,
            schema::ECHO_PRIORITY,
            schema::ECHO_PACKET_INDEX,
            schema::ECHO_SHARED_HIGH_RESERVED,
            schema::HOST_PROCESS_SEQUENCE,
            schema::HOST_PROCESS_COUNT,
            schema::HOST_ERROR,
            schema::HOST_EVENT_COUNT,
            schema::HOST_AUDIT_READY_EMPTY,
            schema::HOST_AUDIT_IDLE_COUNT,
            schema::HOST_AUDIT_GENERAL_MASK,
            schema::HOST_AUDIT_HIGH_MASK,
            schema::HOST_AUDIT_DUPLICATES,
            schema::HOST_AUDIT_MISSING,
            schema::HIGH_INJECT_TIMESTAMP,
            schema::MANDATORY_FIRST_PENDING,
            schema::MUTATION_APPLIED,
            schema::JOIN_SUCCESS,
            schema::SECOND_JOIN_ALREADY_JOINED,
            schema::WAITER_LOCAL_ID,
            schema::EXECUTOR_ACTIVE_FINAL,
            schema::LOW_TOKEN_CONSUMED,
            schema::HIGH_OUTPUT,
            schema::MUTATION_MODE,
            schema::KERNEL_OBSERVATION_FLAGS,
        ];
        for (expected, actual) in fields.into_iter().enumerate() {
            assert_eq!(actual, expected);
        }
        assert_eq!(fields.len(), schema::DECISION_BASE);
        for decision in 0..schema::DECISION_COUNT {
            for field in 0..schema::DECISION_WORDS {
                assert_eq!(
                    schema::decision_word(decision, field),
                    schema::DECISION_BASE + decision * schema::DECISION_WORDS + field,
                );
            }
        }
        assert_eq!(
            schema::DECISION_BASE + schema::DECISION_COUNT * schema::DECISION_WORDS,
            schema::WORD_COUNT,
        );
        assert_eq!(schema::LOW_LIMIT_VALUE, 224);
        assert_eq!(schema::NORMAL_BACKLOG_VALUE, 16);
        assert_ne!(schema::NAMESPACE_VALUE, 0);
        assert_eq!(schema::COORDINATOR_LOCAL_ID_VALUE, 224);
        assert_eq!(schema::EXPECTED_HIGH_LOCAL_ID_VALUE, 241);
    }

    #[test]
    fn priority_echo_reuse_schema_covers_every_word() {
        use priority_echo_reuse_schema as reuse;

        let fields = [
            reuse::VERSION,
            reuse::FIRST_NONCE,
            reuse::SECOND_NONCE,
            reuse::FIRST_PACKET_INDEX,
            reuse::FIRST_ERROR_CATEGORY,
            reuse::REACQUIRE_BUSY_ATTEMPTS,
            reuse::SECOND_PACKET_INDEX,
            reuse::SECOND_ECHO_NONCE,
            reuse::SECOND_RESPONSE_PACKET_INDEX,
            reuse::SECOND_PENDING_POLLS,
            reuse::REUSE_STALE_WRITE,
            reuse::SECOND_COMPLETED,
            reuse::GENERAL_GUARD_PACKET_INDEX,
            reuse::RESERVED_0,
            reuse::RESERVED_1,
            reuse::KERNEL_ERROR,
        ];
        for (expected, actual) in fields.into_iter().enumerate() {
            assert_eq!(actual, expected);
        }
        assert_eq!(fields.len(), reuse::WORD_COUNT);
        assert_eq!(reuse::VERSION_VALUE, 1);
    }
}
