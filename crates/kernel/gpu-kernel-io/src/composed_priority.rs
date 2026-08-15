//! Typed priority + v3 hostcall 的可证伪组合场景。
//!
//! 该场景只验证协作式 poll-boundary 调度、typed dependency 与 hostcall
//! ownership/provenance；不提供硬实时保证，也不控制真实机器人执行器。

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use gpu_atomics::{activemask, syncwarp};
use gpu_kernel_core::helpers::gpu_instant_nanos;
use gpu_protocol::{
    composed_priority_schema as schema, encode_error, priority_echo,
    priority_echo_reuse_schema as reuse, replace_task_local_id, HostcallMetadata, Priority,
    ERR_IO_ERROR, ERR_RESOURCE_BUSY, SERVICE_PRIORITY_ECHO,
};
use gpu_runtime::executor::{ExecutorError, GpuExecutor, PriorityUpdate};
use gpu_runtime::hostcall::{HostcallPacketLease, PendingHostcall};
use gpu_runtime::priority::{
    level, CompletionCell, PriorityClass, PriorityLevel, PriorityToken, TypedJoinError,
    TypedJoinHandle,
};

pub const MUTATION_NONE: u32 = 0;
pub const MUTATION_SCHEDULING_CLASS_SUBSTITUTION: u32 = 1;
pub const MUTATION_WIRE_DOWNGRADE: u32 = 2;
pub const MUTATION_IDENTITY_CORRUPTION: u32 = 3;
pub const MUTATION_HOST_ERROR: u32 = 4;
pub const MUTATION_JOIN_REPLAY: u32 = 5;

const LOW_RAW_HOLDERS: usize = 219;
const LEASE_HOLDERS: usize = 4;
const NORMAL_BACKLOG: usize = 16;
const RELEASE_BIT: u64 = 1;
const OBSERVATION_ERROR_BIT: u64 = 1 << 8;

#[inline(always)]
unsafe fn read_word(results: *mut u64, index: usize) -> u64 {
    core::ptr::read_volatile(results.add(index))
}

#[inline(always)]
unsafe fn write_word(results: *mut u64, index: usize, value: u64) {
    core::ptr::write_volatile(results.add(index), value);
}

#[inline(always)]
unsafe fn increment_word(results: *mut u64, index: usize) -> u64 {
    let value = read_word(results, index).wrapping_add(1);
    write_word(results, index, value);
    value
}

#[inline(always)]
unsafe fn dispatch(results: *mut u64) -> u64 {
    increment_word(results, schema::DISPATCH_SEQUENCE)
}

#[inline(always)]
fn task_id(local_id: u32) -> u64 {
    (schema::NAMESPACE_VALUE << 32) | local_id as u64
}

struct LowLeaseHolder {
    hostcall: *mut u8,
    results: *mut u64,
    ordinal: u32,
    lease: Option<HostcallPacketLease>,
}

impl Future for LowLeaseHolder {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = unsafe { self.get_unchecked_mut() };
        unsafe {
            dispatch(this.results);
            if this.lease.is_none() {
                match HostcallPacketLease::acquire(
                    this.hostcall,
                    HostcallMetadata::new(task_id(1_000 + this.ordinal), Priority::Low),
                ) {
                    Ok(lease) => {
                        let index = lease.packet_index();
                        write_word(
                            this.results,
                            schema::LEASE_MASK,
                            read_word(this.results, schema::LEASE_MASK) | (1u64 << index),
                        );
                        increment_word(this.results, schema::LEASE_ACQUIRED);
                        this.lease = Some(lease);
                    }
                    Err(error) => {
                        write_word(
                            this.results,
                            schema::KERNEL_OBSERVATION_FLAGS,
                            OBSERVATION_ERROR_BIT | error.category as u64,
                        );
                        return Poll::Ready(());
                    }
                }
            }
            if read_word(this.results, schema::KERNEL_OBSERVATION_FLAGS) & RELEASE_BIT != 0 {
                this.lease.take();
                increment_word(this.results, schema::LEASE_RELEASED);
                Poll::Ready(())
            } else {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
}

struct LowHolder {
    results: *mut u64,
}

impl Future for LowHolder {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        unsafe {
            dispatch(self.results);
            if read_word(self.results, schema::KERNEL_OBSERVATION_FLAGS) & RELEASE_BIT != 0 {
                Poll::Ready(())
            } else {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
}

struct NormalHolder {
    results: *mut u64,
    announced: bool,
}

impl Future for NormalHolder {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = unsafe { self.get_unchecked_mut() };
        unsafe {
            dispatch(this.results);
            if !this.announced {
                this.announced = true;
                increment_word(this.results, schema::NORMAL_ALIVE);
            }
            if read_word(this.results, schema::KERNEL_OBSERVATION_FLAGS) & RELEASE_BIT != 0 {
                increment_word(this.results, schema::NORMAL_RELEASED);
                Poll::Ready(())
            } else {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
}

/// 隔离的调度负例：raw probe 先使用 High reserve 完成 admission，再把
/// effective class 设为 Normal。不可迁移的首个 High entry 只负责 arm；
/// self-wake 后它以 Normal 排到 16 个 live Normal FIFO 前驱之后。
/// 它不持有 typed token，也不修改 production typed priority。
struct RawSchedulingClassSubstitution {
    results: *mut u64,
    armed: bool,
}

impl Future for RawSchedulingClassSubstitution {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = unsafe { self.get_unchecked_mut() };
        unsafe {
            let sequence = dispatch(this.results);
            if !this.armed {
                this.armed = true;
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }
            write_word(this.results, schema::HIGH_FIRST_POLL_SEQUENCE, sequence);
            write_word(
                this.results,
                schema::KERNEL_OBSERVATION_FLAGS,
                read_word(this.results, schema::KERNEL_OBSERVATION_FLAGS) | RELEASE_BIT,
            );
        }
        Poll::Ready(())
    }
}

struct TypedEcho<P: PriorityLevel> {
    token: PriorityToken<P>,
    hostcall: *mut u8,
    results: *mut u64,
    nonce: u64,
    mutation: u32,
    pending: Option<PendingHostcall>,
}

impl<P: PriorityLevel> Future for TypedEcho<P> {
    type Output = u64;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<u64> {
        let this = unsafe { self.get_unchecked_mut() };
        unsafe {
            let sequence = dispatch(this.results);

            if let Some(pending) = this.pending.as_mut() {
                let outcome = Pin::new_unchecked(&mut *pending).poll(cx);
                write_word(
                    this.results,
                    schema::HIGH_PENDING_POLLS,
                    pending.pending_polls() as u64,
                );
                return match outcome {
                    Poll::Pending => Poll::Pending,
                    Poll::Ready(Ok(response)) => {
                        write_word(this.results, schema::HIGH_READY_SEQUENCE, sequence);
                        write_word(
                            this.results,
                            schema::HIGH_READY_TIMESTAMP,
                            gpu_instant_nanos(),
                        );
                        write_word(
                            this.results,
                            schema::ECHO_NONCE,
                            response.payload[priority_echo::NONCE],
                        );
                        write_word(
                            this.results,
                            schema::ECHO_TASK_ID,
                            response.payload[priority_echo::TASK_ID],
                        );
                        write_word(
                            this.results,
                            schema::ECHO_PRIORITY,
                            response.payload[priority_echo::PRIORITY],
                        );
                        write_word(
                            this.results,
                            schema::ECHO_PACKET_INDEX,
                            response.payload[priority_echo::PACKET_INDEX],
                        );
                        write_word(
                            this.results,
                            schema::ECHO_SHARED_HIGH_RESERVED,
                            response.payload[priority_echo::SHARED_HIGH_RESERVED],
                        );
                        write_word(
                            this.results,
                            schema::HOST_PROCESS_SEQUENCE,
                            response.payload[priority_echo::PROCESS_SEQUENCE],
                        );
                        write_word(
                            this.results,
                            schema::HOST_PROCESS_COUNT,
                            response.payload[priority_echo::PROCESS_COUNT],
                        );
                        write_word(
                            this.results,
                            schema::HOST_ERROR,
                            response.payload[priority_echo::ERROR_CATEGORY],
                        );
                        Poll::Ready(response.payload[priority_echo::TASK_ID])
                    }
                    Poll::Ready(Err(error)) => {
                        write_word(this.results, schema::HIGH_READY_SEQUENCE, sequence);
                        write_word(
                            this.results,
                            schema::HIGH_READY_TIMESTAMP,
                            gpu_instant_nanos(),
                        );
                        write_word(this.results, schema::HOST_ERROR, error.category as u64);
                        Poll::Ready((1u64 << 63) | error.category as u64)
                    }
                };
            }

            write_word(this.results, schema::HIGH_FIRST_POLL_SEQUENCE, sequence);
            write_word(
                this.results,
                schema::HIGH_FIRST_POLL_TIMESTAMP,
                gpu_instant_nanos(),
            );
            let mut metadata = match this.token.wire_metadata() {
                Ok(metadata) => metadata,
                Err(_) => {
                    write_word(this.results, schema::HOST_ERROR, ERR_IO_ERROR as u64);
                    return Poll::Ready((1u64 << 63) | ERR_IO_ERROR as u64);
                }
            };
            if this.mutation == MUTATION_WIRE_DOWNGRADE {
                write_word(
                    this.results,
                    schema::MUTATION_APPLIED,
                    MUTATION_WIRE_DOWNGRADE as u64,
                );
                metadata.effective_priority = Priority::Normal;
            } else if this.mutation == MUTATION_IDENTITY_CORRUPTION {
                write_word(
                    this.results,
                    schema::MUTATION_APPLIED,
                    MUTATION_IDENTITY_CORRUPTION as u64,
                );
                metadata.task_id = replace_task_local_id(metadata.task_id, 0);
            } else if this.mutation == MUTATION_HOST_ERROR {
                write_word(
                    this.results,
                    schema::MUTATION_APPLIED,
                    MUTATION_HOST_ERROR as u64,
                );
            }
            write_word(this.results, schema::WIRE_TASK_ID, metadata.task_id);
            write_word(
                this.results,
                schema::WIRE_NAMESPACE,
                (metadata.task_id >> 32) as u32 as u64,
            );
            write_word(
                this.results,
                schema::WIRE_LOCAL_ID,
                metadata.task_id as u32 as u64,
            );
            write_word(
                this.results,
                schema::WIRE_PRIORITY,
                metadata.effective_priority.as_raw() as u64,
            );

            let lease = match HostcallPacketLease::acquire(this.hostcall, metadata) {
                Ok(lease) => lease,
                Err(error) => {
                    write_word(this.results, schema::HOST_ERROR, error.category as u64);
                    return Poll::Ready((1u64 << 63) | error.category as u64);
                }
            };
            write_word(
                this.results,
                schema::GPU_PACKET_INDEX,
                lease.packet_index() as u64,
            );
            write_word(
                this.results,
                schema::GPU_SHARED_HIGH_RESERVED,
                lease.is_shared_high_reserved() as u64,
            );
            let mut payload = [0u64; priority_echo::SLOTS];
            payload[priority_echo::NONCE] = this.nonce;
            let mut pending = lease.submit(SERVICE_PRIORITY_ECHO, payload, 10_000_000);
            let first = Pin::new_unchecked(&mut pending).poll(cx);
            write_word(this.results, schema::HIGH_SUBMIT_SEQUENCE, sequence);
            write_word(
                this.results,
                schema::HIGH_PENDING_POLLS,
                pending.pending_polls() as u64,
            );
            this.pending = Some(pending);
            match first {
                Poll::Pending => {
                    write_word(this.results, schema::MANDATORY_FIRST_PENDING, 1);
                    Poll::Pending
                }
                Poll::Ready(_) => {
                    write_word(
                        this.results,
                        schema::KERNEL_OBSERVATION_FLAGS,
                        read_word(this.results, schema::KERNEL_OBSERVATION_FLAGS)
                            | OBSERVATION_ERROR_BIT,
                    );
                    Poll::Ready((1u64 << 63) | ERR_IO_ERROR as u64)
                }
            }
        }
    }
}

struct LowCoordinator {
    token: PriorityToken<level::Low>,
    executor: *const GpuExecutor,
    hostcall: *mut u8,
    high_completion: *const CompletionCell<u64>,
    results: *mut u64,
    nonce: u64,
    mutation: u32,
    handle: Option<TypedJoinHandle<level::High, u64>>,
    join_started: bool,
    joined_value: Option<u64>,
    substitution_spawned: bool,
    checked_pool: bool,
}

impl Future for LowCoordinator {
    type Output = u64;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<u64> {
        let this = unsafe { self.get_unchecked_mut() };
        unsafe {
            let sequence = dispatch(this.results);
            if this.mutation == MUTATION_SCHEDULING_CLASS_SUBSTITUTION && this.substitution_spawned
            {
                if read_word(this.results, schema::HIGH_FIRST_POLL_SEQUENCE) != 0 {
                    return Poll::Ready(0);
                }
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }
            if this.handle.is_none() {
                if read_word(this.results, schema::LEASE_ACQUIRED) != LEASE_HOLDERS as u64
                    || read_word(this.results, schema::NORMAL_ALIVE) != NORMAL_BACKLOG as u64
                {
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                if !this.checked_pool {
                    this.checked_pool = true;
                    let exhausted = match HostcallPacketLease::acquire(
                        this.hostcall,
                        HostcallMetadata::new(task_id(2_000), Priority::Low),
                    ) {
                        Ok(lease) => {
                            drop(lease);
                            false
                        }
                        Err(error) => error.category == ERR_RESOURCE_BUSY,
                    };
                    write_word(
                        this.results,
                        schema::GENERAL_POOL_EXHAUSTED,
                        exhausted as u64,
                    );
                }

                if let Ok(metadata) = this.token.metadata() {
                    write_word(
                        this.results,
                        schema::WAITER_LOCAL_ID,
                        metadata.task_id as u32 as u64,
                    );
                }
                write_word(this.results, schema::HIGH_INJECT_SEQUENCE, sequence);
                let inject_ts = gpu_instant_nanos();
                write_word(this.results, schema::HIGH_INJECT_TIMESTAMP, inject_ts);

                let executor = &*this.executor;
                if this.mutation == MUTATION_SCHEDULING_CLASS_SUBSTITUTION {
                    match executor.spawn_with_priority(
                        RawSchedulingClassSubstitution {
                            results: this.results,
                            armed: false,
                        },
                        Priority::High,
                    ) {
                        Ok(task_id) => {
                            match executor.set_effective_priority(task_id, Priority::Normal) {
                                Ok(PriorityUpdate::DeferredUntilRequeue) => {
                                    write_word(
                                        this.results,
                                        schema::MUTATION_APPLIED,
                                        MUTATION_SCHEDULING_CLASS_SUBSTITUTION as u64,
                                    );
                                    this.substitution_spawned = true
                                }
                                _ => {
                                    write_word(
                                        this.results,
                                        schema::KERNEL_OBSERVATION_FLAGS,
                                        OBSERVATION_ERROR_BIT | RELEASE_BIT,
                                    );
                                    return Poll::Ready(0);
                                }
                            }
                        }
                        Err(error) => {
                            write_word(
                                this.results,
                                schema::KERNEL_OBSERVATION_FLAGS,
                                OBSERVATION_ERROR_BIT | error as u64 | RELEASE_BIT,
                            );
                            return Poll::Ready(0);
                        }
                    }
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                let completion = &*this.high_completion;
                let hostcall = this.hostcall;
                let results = this.results;
                let nonce = this.nonce;
                let mutation = this.mutation;
                let spawned = executor.spawn_typed(
                    PriorityClass::<level::High>::new(),
                    completion,
                    move |token| TypedEcho {
                        token,
                        hostcall,
                        results,
                        nonce,
                        mutation,
                        pending: None,
                    },
                );
                match spawned {
                    Ok(handle) => this.handle = Some(handle),
                    Err(error) => {
                        write_word(
                            this.results,
                            schema::KERNEL_OBSERVATION_FLAGS,
                            OBSERVATION_ERROR_BIT | error as u64,
                        );
                        write_word(
                            this.results,
                            schema::KERNEL_OBSERVATION_FLAGS,
                            read_word(this.results, schema::KERNEL_OBSERVATION_FLAGS) | RELEASE_BIT,
                        );
                        return Poll::Ready(0);
                    }
                }
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }

            if !this.join_started {
                write_word(this.results, schema::LOW_WAIT_START_SEQUENCE, sequence);
                this.join_started = true;
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }

            if let Some(value) = this.joined_value {
                let replay = {
                    let handle = this.handle.as_mut().expect("typed High handle");
                    let mut second = this.token.wait_for(handle);
                    Pin::new_unchecked(&mut second).poll(cx)
                };
                write_word(this.results, schema::LOW_SECOND_JOIN_SEQUENCE, sequence);
                let already_joined =
                    matches!(replay, Poll::Ready(Err(TypedJoinError::AlreadyJoined)));
                if this.mutation == MUTATION_JOIN_REPLAY && already_joined {
                    write_word(
                        this.results,
                        schema::MUTATION_APPLIED,
                        MUTATION_JOIN_REPLAY as u64,
                    );
                }
                write_word(
                    this.results,
                    schema::SECOND_JOIN_ALREADY_JOINED,
                    (already_joined && this.mutation != MUTATION_JOIN_REPLAY) as u64,
                );
                write_word(this.results, schema::LOW_TOKEN_CONSUMED, 1);
                write_word(
                    this.results,
                    schema::KERNEL_OBSERVATION_FLAGS,
                    read_word(this.results, schema::KERNEL_OBSERVATION_FLAGS) | RELEASE_BIT,
                );
                return Poll::Ready(value);
            }

            let join_poll = {
                let handle = this.handle.as_mut().expect("typed High handle");
                let mut wait = this.token.wait_for(handle);
                Pin::new_unchecked(&mut wait).poll(cx)
            };
            match join_poll {
                Poll::Pending => Poll::Pending,
                Poll::Ready(Ok(value)) => {
                    write_word(this.results, schema::LOW_JOIN_SEQUENCE, sequence);
                    write_word(this.results, schema::JOIN_SUCCESS, 1);
                    write_word(this.results, schema::HIGH_OUTPUT, value);
                    this.joined_value = Some(value);
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
                Poll::Ready(Err(_)) => {
                    write_word(
                        this.results,
                        schema::KERNEL_OBSERVATION_FLAGS,
                        read_word(this.results, schema::KERNEL_OBSERVATION_FLAGS)
                            | OBSERVATION_ERROR_BIT
                            | RELEASE_BIT,
                    );
                    Poll::Ready(0)
                }
            }
        }
    }
}

#[inline(always)]
unsafe fn write_decision(
    results: *mut u64,
    decision: usize,
    kind: u64,
    action: u64,
    use_gpu: u64,
    age_source: u64,
    sample: u64,
    now: u64,
    age: u64,
    budget: u64,
    hazard: u64,
) {
    write_word(
        results,
        schema::decision_word(decision, schema::DECISION_KIND),
        kind,
    );
    write_word(
        results,
        schema::decision_word(decision, schema::DECISION_ACTION),
        action,
    );
    write_word(
        results,
        schema::decision_word(decision, schema::DECISION_USE_GPU_RESULT),
        use_gpu,
    );
    write_word(
        results,
        schema::decision_word(decision, schema::DECISION_AGE_SOURCE),
        age_source,
    );
    write_word(
        results,
        schema::decision_word(decision, schema::DECISION_SAMPLE_TIMESTAMP),
        sample,
    );
    write_word(
        results,
        schema::decision_word(decision, schema::DECISION_NOW_TIMESTAMP),
        now,
    );
    write_word(
        results,
        schema::decision_word(decision, schema::DECISION_AGE_TICKS),
        age,
    );
    write_word(
        results,
        schema::decision_word(decision, schema::DECISION_BUDGET_TICKS),
        budget,
    );
    write_word(
        results,
        schema::decision_word(decision, schema::DECISION_HAZARD),
        hazard,
    );
}

/// 组合 E2E：224 Low admission、16 Normal backlog、typed High echo/join。
#[no_mangle]
pub unsafe extern "gpu-kernel" fn composed_priority_hostcall_e2e(
    executor_ptr: *mut u8,
    hostcall: *mut u8,
    low_completion_ptr: *mut CompletionCell<u64>,
    high_completion_ptr: *mut CompletionCell<u64>,
    results: *mut u64,
    nonce: u64,
    mutation: u32,
) {
    let lane = core::arch::nvptx::_thread_idx_x() as u32;
    if lane == 0 {
        let mut index = 0;
        while index < schema::WORD_COUNT {
            write_word(results, index, 0);
            index += 1;
        }
        core::ptr::write(low_completion_ptr, CompletionCell::new());
        core::ptr::write(high_completion_ptr, CompletionCell::new());
        write_word(results, schema::VERSION, schema::VERSION_VALUE);
        write_word(results, schema::WORDS, schema::WORD_COUNT as u64);
        write_word(results, schema::PHASE, 1);
        write_word(results, schema::NONCE, nonce);
        write_word(results, schema::NAMESPACE, schema::NAMESPACE_VALUE);
        write_word(
            results,
            schema::EXPECTED_HIGH_LOCAL_ID,
            schema::EXPECTED_HIGH_LOCAL_ID_VALUE,
        );
        write_word(results, schema::PACKET_COUNT, schema::PACKET_COUNT_VALUE);
        write_word(
            results,
            schema::GENERAL_PACKET_COUNT,
            schema::GENERAL_PACKET_COUNT_VALUE,
        );
        write_word(
            results,
            schema::HIGH_RESERVED_COUNT,
            schema::HIGH_RESERVED_COUNT_VALUE,
        );
        write_word(results, schema::HARD_LOW_LIMIT, schema::LOW_LIMIT_VALUE);
        write_word(
            results,
            schema::HARD_NORMAL_BACKLOG,
            schema::NORMAL_BACKLOG_VALUE,
        );
        write_word(
            results,
            schema::HIGH_FIRST_POLL_GAP_MAX,
            schema::HIGH_FIRST_POLL_GAP_MAX_VALUE,
        );
        write_word(results, schema::EXPECTED_END_MAGIC, schema::END_MAGIC_VALUE);
        write_word(results, schema::MUTATION_MODE, mutation as u64);
    }
    let mask = activemask();
    syncwarp(mask);

    let executor = &*(executor_ptr as *const GpuExecutor);
    if lane == 0 {
        executor.init_with_namespace(schema::NAMESPACE_VALUE as u32);
        let mut low_admitted = 0u64;
        let mut ordinal = 0usize;
        while ordinal < LEASE_HOLDERS {
            if executor
                .spawn_with_priority(
                    LowLeaseHolder {
                        hostcall,
                        results,
                        ordinal: ordinal as u32,
                        lease: None,
                    },
                    Priority::Low,
                )
                .is_ok()
            {
                low_admitted += 1;
            }
            ordinal += 1;
        }
        ordinal = 0;
        while ordinal < LOW_RAW_HOLDERS {
            if executor
                .spawn_with_priority(LowHolder { results }, Priority::Low)
                .is_ok()
            {
                low_admitted += 1;
            }
            ordinal += 1;
        }
        let low_completion = &*low_completion_ptr;
        let coordinator = executor.spawn_typed(
            PriorityClass::<level::Low>::new(),
            low_completion,
            move |token| LowCoordinator {
                token,
                executor,
                hostcall,
                high_completion: high_completion_ptr,
                results,
                nonce,
                mutation,
                handle: None,
                join_started: false,
                joined_value: None,
                substitution_spawned: false,
                checked_pool: false,
            },
        );
        if coordinator.is_ok() {
            low_admitted += 1;
        }
        drop(coordinator);
        write_word(results, schema::LOW_ADMITTED, low_admitted);

        match executor.spawn_with_priority(LowHolder { results }, Priority::Low) {
            Err(ExecutorError::ReservedCapacity) => {
                write_word(results, schema::RESERVED_CAPACITY_REJECTIONS, 1)
            }
            _ => write_word(
                results,
                schema::KERNEL_OBSERVATION_FLAGS,
                OBSERVATION_ERROR_BIT,
            ),
        }

        let mut normal = 0usize;
        while normal < NORMAL_BACKLOG {
            if executor
                .spawn_with_priority(
                    NormalHolder {
                        results,
                        announced: false,
                    },
                    Priority::Normal,
                )
                .is_err()
            {
                write_word(
                    results,
                    schema::KERNEL_OBSERVATION_FLAGS,
                    OBSERVATION_ERROR_BIT,
                );
            }
            normal += 1;
        }
        write_word(results, schema::PHASE, 2);
    }
    syncwarp(mask);

    let _stats = executor.run(mask);
    syncwarp(mask);

    if lane == 0 {
        write_word(
            results,
            schema::EXECUTOR_SPAWNED,
            executor.spawned_count() as u64,
        );
        write_word(
            results,
            schema::EXECUTOR_COMPLETED,
            executor.completed_count() as u64,
        );
        write_word(
            results,
            schema::EXECUTOR_ACTIVE_FINAL,
            executor.active_count() as u64,
        );

        let inject = read_word(results, schema::HIGH_INJECT_TIMESTAMP);
        let first = read_word(results, schema::HIGH_FIRST_POLL_TIMESTAMP);
        let ready = read_word(results, schema::HIGH_READY_TIMESTAMP);
        let age = ready.saturating_sub(first);
        let fresh_budget = age.saturating_add(1);
        let first_latency = first.saturating_sub(inject);
        let deadline_budget = first_latency.saturating_sub(1);
        write_word(results, schema::FRESHNESS_BUDGET_TICKS, fresh_budget);
        write_word(results, schema::DEADLINE_BUDGET_TICKS, deadline_budget);

        write_decision(
            results,
            schema::DECISION_FRESH_HAZARD,
            schema::DECISION_FRESH_HAZARD as u64,
            schema::ACTION_APPLY_BRAKE_FROM_FRESH_RESULT,
            1,
            schema::AGE_SOURCE_REAL_GPU_CLOCK,
            first,
            ready,
            age,
            fresh_budget,
            1,
        );
        write_decision(
            results,
            schema::DECISION_FRESH_SAFE,
            schema::DECISION_FRESH_SAFE as u64,
            schema::ACTION_NO_BRAKE,
            1,
            schema::AGE_SOURCE_REAL_GPU_CLOCK,
            first,
            ready,
            age,
            fresh_budget,
            0,
        );
        write_decision(
            results,
            schema::DECISION_STALE,
            schema::DECISION_STALE as u64,
            schema::ACTION_WATCHDOG_CONSERVATIVE_STOP,
            0,
            schema::AGE_SOURCE_INJECTED_STALE,
            ready.saturating_sub(fresh_budget.saturating_add(1)),
            ready,
            fresh_budget.saturating_add(1),
            fresh_budget,
            1,
        );
        write_decision(
            results,
            schema::DECISION_DEADLINE,
            schema::DECISION_DEADLINE as u64,
            schema::ACTION_WATCHDOG_CONSERVATIVE_STOP,
            0,
            schema::AGE_SOURCE_REAL_FIRST_POLL_LATENCY,
            inject,
            first,
            first_latency,
            deadline_budget,
            1,
        );
        write_word(results, schema::END_MAGIC, schema::END_MAGIC_VALUE);
        write_word(results, schema::PHASE, 4);
    }
}

/// Timeout→host reclaim→同一 reserved packet 复用，无复用后 stale write。
#[no_mangle]
pub unsafe extern "gpu-kernel" fn priority_echo_timeout_reuse(
    hostcall: *mut u8,
    results: *mut u64,
    first_nonce: u64,
    second_nonce: u64,
) {
    let lane = core::arch::nvptx::_thread_idx_x() as u32;
    if lane != 0 {
        return;
    }
    let mut index = 0;
    while index < reuse::WORD_COUNT {
        write_word(results, index, 0);
        index += 1;
    }
    write_word(results, reuse::VERSION, reuse::VERSION_VALUE);
    write_word(results, reuse::FIRST_NONCE, first_nonce);
    write_word(results, reuse::SECOND_NONCE, second_nonce);

    // Occupy the sole general credit so the second High request cannot fall
    // back to another packet while the reserved packet is host-owned.
    let general_guard = match HostcallPacketLease::acquire(
        hostcall,
        HostcallMetadata::new(task_id(249), Priority::Low),
    ) {
        Ok(lease) => lease,
        Err(error) => {
            write_word(
                results,
                reuse::KERNEL_ERROR,
                encode_error(error.category, error.raw_errno),
            );
            return;
        }
    };
    write_word(
        results,
        reuse::GENERAL_GUARD_PACKET_INDEX,
        general_guard.packet_index() as u64,
    );

    let first_metadata = HostcallMetadata::new(task_id(250), Priority::High);
    let first_lease = match HostcallPacketLease::acquire(hostcall, first_metadata) {
        Ok(lease) => lease,
        Err(error) => {
            write_word(
                results,
                reuse::KERNEL_ERROR,
                encode_error(error.category, error.raw_errno),
            );
            return;
        }
    };
    write_word(
        results,
        reuse::FIRST_PACKET_INDEX,
        first_lease.packet_index() as u64,
    );
    let mut first_payload = [0u64; priority_echo::SLOTS];
    first_payload[priority_echo::NONCE] = first_nonce;
    let first = first_lease.submit(SERVICE_PRIORITY_ECHO, first_payload, 1);
    let first_result = gpu_runtime::std_future::block_on_with(first, 32, 64);
    match first_result {
        Some(Err(error)) => write_word(results, reuse::FIRST_ERROR_CATEGORY, error.category as u64),
        _ => {
            write_word(results, reuse::KERNEL_ERROR, 1);
            return;
        }
    }

    let second_metadata = HostcallMetadata::new(task_id(251), Priority::High);
    let mut attempts = 0u64;
    let second_lease = loop {
        match HostcallPacketLease::acquire(hostcall, second_metadata) {
            Ok(lease) => break lease,
            Err(error) if error.category == ERR_RESOURCE_BUSY && attempts < 10_000_000 => {
                attempts += 1;
                gpu_runtime::thread::sleep_nanos(1_024);
            }
            Err(error) => {
                write_word(
                    results,
                    reuse::KERNEL_ERROR,
                    encode_error(error.category, error.raw_errno),
                );
                return;
            }
        }
    };
    write_word(results, reuse::REACQUIRE_BUSY_ATTEMPTS, attempts);
    write_word(
        results,
        reuse::SECOND_PACKET_INDEX,
        second_lease.packet_index() as u64,
    );
    let mut second_payload = [0u64; priority_echo::SLOTS];
    second_payload[priority_echo::NONCE] = second_nonce;
    let second = second_lease.submit(SERVICE_PRIORITY_ECHO, second_payload, 10_000_000);
    match gpu_runtime::std_future::block_on_with(second, 10_000_000, 64) {
        Some(Ok(response)) => {
            write_word(
                results,
                reuse::SECOND_ECHO_NONCE,
                response.payload[priority_echo::NONCE],
            );
            write_word(
                results,
                reuse::SECOND_RESPONSE_PACKET_INDEX,
                response.packet_index as u64,
            );
            write_word(
                results,
                reuse::SECOND_PENDING_POLLS,
                response.pending_polls as u64,
            );
            write_word(
                results,
                reuse::REUSE_STALE_WRITE,
                (response.payload[priority_echo::NONCE] != second_nonce) as u64,
            );
            write_word(results, reuse::SECOND_COMPLETED, 1);
        }
        Some(Err(error)) => write_word(
            results,
            reuse::KERNEL_ERROR,
            encode_error(error.category, error.raw_errno),
        ),
        None => write_word(results, reuse::KERNEL_ERROR, encode_error(ERR_IO_ERROR, 0)),
    }
    drop(general_guard);
}
