//! 障碍事件优先级压力场景 / Obstacle-event priority stress scenario.
//!
//! 该 kernel 只验证协作式执行器的可观测调度语义；它不连接传感器、制动器，
//! 也不构成硬实时或机器人安全认证。

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use gpu_atomics::{activemask, syncwarp};
use gpu_kernel_core::helpers::gpu_instant_nanos;
use gpu_runtime::executor::{
    ExecutorError, GpuExecutor, Priority, HIGH_PRIORITY_RESERVED_SLOTS, MAX_TASKS,
    NORMAL_PRIORITY_RESERVED_SLOTS,
};

/// Host/kernel 共享结果字数；host 端有同名 schema 常量。
const RESULT_WORDS: usize = 48;

/// 结果字段 / Result schema. 具名常量避免 kernel 内散落 magic indices。
mod result {
    pub const VERSION: usize = 0;
    pub const PHASE: usize = 1;
    pub const LOW_ADMITTED_TOTAL: usize = 2;
    pub const LOW_WORK_ADMITTED: usize = 3;
    pub const LOW_RESERVED_REJECTIONS: usize = 4;
    pub const LOW_OTHER_REJECTIONS: usize = 5;
    pub const HIGH_SPAWN_OK: usize = 6;
    pub const POS_INJECT_SEQ: usize = 7;
    pub const POS_HIGH_FIRST_SEQ: usize = 8;
    pub const POS_HIGH_FIRST_GAP: usize = 9;
    pub const POS_HIGH_RESUME_SEQ: usize = 10;
    pub const POS_HIGH_RESUME_GAP: usize = 11;
    pub const LOW_COMPLETED: usize = 12;
    pub const POS_SPAWNED: usize = 13;
    pub const POS_COMPLETED: usize = 14;
    pub const RESERVED_PASS: usize = 15;
    pub const FIRST_POLL_PASS: usize = 16;
    pub const WAKER_PASS: usize = 17;
    pub const LOW_EVENTUAL_PASS: usize = 18;
    pub const POSITIVE_PASS: usize = 19;
    pub const POS_INJECT_NS: usize = 20;
    pub const POS_HIGH_FIRST_NS: usize = 21;
    pub const POS_LATENCY_NS: usize = 22;
    pub const NEG_HIGH_SPAWN_OK: usize = 23;
    pub const NEG_INJECT_SEQ: usize = 24;
    pub const NEG_LOW_RETURN_SEQ: usize = 25;
    pub const NEG_HIGH_FIRST_SEQ: usize = 26;
    pub const NEG_INJECT_NS: usize = 27;
    pub const NEG_LOW_RETURN_NS: usize = 28;
    pub const NEG_HIGH_FIRST_NS: usize = 29;
    pub const NEG_LATENCY_NS: usize = 30;
    pub const NEG_NO_MIDPOLL_PASS: usize = 31;
    pub const FRESH_ACTION: usize = 32;
    pub const FRESH_REASON: usize = 33;
    pub const FRESH_USED_GPU_RESULT: usize = 34;
    pub const STALE_ACTION: usize = 35;
    pub const STALE_REASON: usize = 36;
    pub const STALE_USED_GPU_RESULT: usize = 37;
    pub const DEADLINE_ACTION: usize = 38;
    pub const DEADLINE_REASON: usize = 39;
    pub const DEADLINE_USED_GPU_RESULT: usize = 40;
    pub const DECISION_PASS: usize = 41;
    pub const POS_DISPATCH_SEQUENCE: usize = 42;
    pub const NEG_DISPATCH_SEQUENCE: usize = 43;
    pub const EXPECTED_LOW_LIMIT: usize = 44;
    pub const NEG_BUSY_ITERS: usize = 45;
    pub const OVERALL_PASS: usize = 46;
    pub const WORD_COUNT: usize = 47;
}

const SCHEMA_VERSION: u64 = 1;
const NEGATIVE_BUSY_ITERS: u32 = 4096;

const ACTION_APPLY_BRAKE_FROM_FRESH_RESULT: u64 = 1;
const ACTION_WATCHDOG_CONSERVATIVE_STOP: u64 = 2;
const REASON_FRESH_HAZARD: u64 = 1;
const REASON_STALE_RESULT: u64 = 2;
const REASON_DEADLINE_MISS: u64 = 4;

#[inline(always)]
unsafe fn read_word(results: *mut u64, field: usize) -> u64 {
    core::ptr::read_volatile(results.add(field))
}

#[inline(always)]
unsafe fn write_word(results: *mut u64, field: usize, value: u64) {
    core::ptr::write_volatile(results.add(field), value);
}

#[inline(always)]
unsafe fn next_sequence(results: *mut u64, field: usize) -> u64 {
    let next = read_word(results, field).wrapping_add(1);
    write_word(results, field, next);
    next
}

/// Low 工作：第一次 poll 主动唤醒并让出，第二次完成。
struct YieldingLow {
    results: *mut u64,
    yielded: bool,
}

impl Future for YieldingLow {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        unsafe {
            next_sequence(self.results, result::POS_DISPATCH_SEQUENCE);
        }
        if !self.yielded {
            self.yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        } else {
            unsafe {
                let completed = read_word(self.results, result::LOW_COMPLETED);
                write_word(self.results, result::LOW_COMPLETED, completed + 1);
            }
            Poll::Ready(())
        }
    }
}

/// High 事件：第一次 poll 自唤醒，第二次完成，用于验证 waker 保留 priority。
struct HighObstacleEvent {
    results: *mut u64,
    resumed: bool,
}

impl Future for HighObstacleEvent {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        unsafe {
            let sequence = next_sequence(self.results, result::POS_DISPATCH_SEQUENCE);
            if !self.resumed {
                write_word(self.results, result::POS_HIGH_FIRST_SEQ, sequence);
                write_word(self.results, result::POS_HIGH_FIRST_NS, gpu_instant_nanos());
                self.resumed = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            } else {
                write_word(self.results, result::POS_HIGH_RESUME_SEQ, sequence);
                Poll::Ready(())
            }
        }
    }
}

/// 首次让出并排到 Low backlog 尾部；再次获得 poll 时才注入 High 事件。
struct LateHighInjector {
    executor: *const GpuExecutor,
    results: *mut u64,
    armed: bool,
}

impl Future for LateHighInjector {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        unsafe {
            let sequence = next_sequence(self.results, result::POS_DISPATCH_SEQUENCE);
            if !self.armed {
                self.armed = true;
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }

            write_word(self.results, result::POS_INJECT_SEQ, sequence);
            write_word(self.results, result::POS_INJECT_NS, gpu_instant_nanos());
            let executor = &*self.executor;
            let spawned = executor
                .spawn_with_priority(
                    HighObstacleEvent {
                        results: self.results,
                        resumed: false,
                    },
                    Priority::High,
                )
                .is_ok();
            write_word(self.results, result::HIGH_SPAWN_OK, spawned as u64);
            Poll::Ready(())
        }
    }
}

/// 负例中的 High 事件只记录第一次 poll。
struct NegativeHighEvent {
    results: *mut u64,
}

impl Future for NegativeHighEvent {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        unsafe {
            let sequence = next_sequence(self.results, result::NEG_DISPATCH_SEQUENCE);
            write_word(self.results, result::NEG_HIGH_FIRST_SEQ, sequence);
            write_word(self.results, result::NEG_HIGH_FIRST_NS, gpu_instant_nanos());
        }
        Poll::Ready(())
    }
}

/// 有界 non-yielding Low：poll 内先入队 High，再继续工作，最后才返回。
/// 这不是无限挂死测试；它可证伪“priority 等于抢占”的错误假设。
struct BoundedNonYieldingLow {
    executor: *const GpuExecutor,
    results: *mut u64,
}

impl Future for BoundedNonYieldingLow {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        unsafe {
            let inject_sequence = next_sequence(self.results, result::NEG_DISPATCH_SEQUENCE);
            write_word(self.results, result::NEG_INJECT_SEQ, inject_sequence);
            write_word(self.results, result::NEG_INJECT_NS, gpu_instant_nanos());

            let executor = &*self.executor;
            let spawned = executor
                .spawn_with_priority(
                    NegativeHighEvent {
                        results: self.results,
                    },
                    Priority::High,
                )
                .is_ok();
            write_word(self.results, result::NEG_HIGH_SPAWN_OK, spawned as u64);

            let mut iteration = 0u32;
            while iteration < NEGATIVE_BUSY_ITERS {
                // `nanosleep` 是有界 busy-work/yield hint；结构性判据使用 poll sequence。
                gpu_runtime::thread::sleep_nanos(1024);
                iteration += 1;
            }

            let return_sequence = next_sequence(self.results, result::NEG_DISPATCH_SEQUENCE);
            write_word(self.results, result::NEG_LOW_RETURN_SEQ, return_sequence);
            write_word(self.results, result::NEG_LOW_RETURN_NS, gpu_instant_nanos());
        }
        Poll::Ready(())
    }
}

#[derive(Clone, Copy)]
struct SafetyDecision {
    action: u64,
    reason: u64,
    used_gpu_result: u64,
}

/// 纯决策函数：stale/deadline 时必须丢弃 GPU 结果并走 watchdog 保守停车。
fn decide_obstacle_action(
    fresh_hazard: bool,
    result_age_ns: u64,
    first_poll_latency_ns: u64,
    freshness_budget_ns: u64,
    deadline_ns: u64,
) -> SafetyDecision {
    if result_age_ns > freshness_budget_ns {
        return SafetyDecision {
            action: ACTION_WATCHDOG_CONSERVATIVE_STOP,
            reason: REASON_STALE_RESULT,
            used_gpu_result: 0,
        };
    }
    if first_poll_latency_ns > deadline_ns {
        return SafetyDecision {
            action: ACTION_WATCHDOG_CONSERVATIVE_STOP,
            reason: REASON_DEADLINE_MISS,
            used_gpu_result: 0,
        };
    }
    SafetyDecision {
        action: ACTION_APPLY_BRAKE_FROM_FRESH_RESULT,
        reason: REASON_FRESH_HAZARD,
        used_gpu_result: fresh_hazard as u64,
    }
}

unsafe fn write_decision(
    results: *mut u64,
    action_field: usize,
    reason_field: usize,
    used_field: usize,
    decision: SafetyDecision,
) {
    write_word(results, action_field, decision.action);
    write_word(results, reason_field, decision.reason);
    write_word(results, used_field, decision.used_gpu_result);
}

/// 真实 GPU 障碍事件优先级压力测试入口。
///
/// 两块 executor memory 均须至少 512 KiB；`results` 指向 u64[48] mapped memory。
#[no_mangle]
pub unsafe extern "gpu-kernel" fn obstacle_event_priority_stress(
    positive_executor_ptr: *mut u8,
    negative_executor_ptr: *mut u8,
    results: *mut u64,
) {
    let lane = core::arch::nvptx::_thread_idx_x() as u32;
    if lane == 0 {
        let mut field = 0usize;
        while field < RESULT_WORDS {
            write_word(results, field, 0);
            field += 1;
        }
        write_word(results, result::VERSION, SCHEMA_VERSION);
        write_word(results, result::PHASE, 1);
        write_word(results, result::WORD_COUNT, RESULT_WORDS as u64);
        write_word(
            results,
            result::EXPECTED_LOW_LIMIT,
            (MAX_TASKS - HIGH_PRIORITY_RESERVED_SLOTS - NORMAL_PRIORITY_RESERVED_SLOTS) as u64,
        );
        write_word(results, result::NEG_BUSY_ITERS, NEGATIVE_BUSY_ITERS as u64);
    }

    let mask = activemask();
    syncwarp(mask);

    let positive = &*(positive_executor_ptr as *const GpuExecutor);
    if lane == 0 {
        positive.init();
        let injector_ok = positive
            .spawn_with_priority(
                LateHighInjector {
                    executor: positive as *const GpuExecutor,
                    results,
                    armed: false,
                },
                Priority::Low,
            )
            .is_ok();

        let mut low_total = injector_ok as u64;
        let mut low_work = 0u64;
        let mut attempts = 0usize;
        while attempts <= MAX_TASKS {
            match positive.spawn_with_priority(
                YieldingLow {
                    results,
                    yielded: false,
                },
                Priority::Low,
            ) {
                Ok(_) => {
                    low_total += 1;
                    low_work += 1;
                }
                Err(ExecutorError::ReservedCapacity) => {
                    write_word(results, result::LOW_RESERVED_REJECTIONS, 1);
                    break;
                }
                Err(_) => {
                    write_word(results, result::LOW_OTHER_REJECTIONS, 1);
                    break;
                }
            }
            attempts += 1;
        }
        write_word(results, result::LOW_ADMITTED_TOTAL, low_total);
        write_word(results, result::LOW_WORK_ADMITTED, low_work);
        write_word(results, result::PHASE, 2);
    }
    syncwarp(mask);

    let _positive_stats = positive.run(mask);
    syncwarp(mask);

    if lane == 0 {
        let inject_sequence = read_word(results, result::POS_INJECT_SEQ);
        let first_sequence = read_word(results, result::POS_HIGH_FIRST_SEQ);
        let resume_sequence = read_word(results, result::POS_HIGH_RESUME_SEQ);
        let first_gap = first_sequence.saturating_sub(inject_sequence);
        let resume_gap = resume_sequence.saturating_sub(first_sequence);
        let inject_ns = read_word(results, result::POS_INJECT_NS);
        let first_ns = read_word(results, result::POS_HIGH_FIRST_NS);
        let low_total = read_word(results, result::LOW_ADMITTED_TOTAL);
        let low_work = read_word(results, result::LOW_WORK_ADMITTED);
        let low_completed = read_word(results, result::LOW_COMPLETED);
        let expected_low = read_word(results, result::EXPECTED_LOW_LIMIT);

        write_word(results, result::POS_HIGH_FIRST_GAP, first_gap);
        write_word(results, result::POS_HIGH_RESUME_GAP, resume_gap);
        write_word(
            results,
            result::POS_LATENCY_NS,
            first_ns.saturating_sub(inject_ns),
        );
        write_word(
            results,
            result::POS_SPAWNED,
            positive.spawned_count() as u64,
        );
        write_word(
            results,
            result::POS_COMPLETED,
            positive.completed_count() as u64,
        );

        let reserved_pass = low_total == expected_low
            && read_word(results, result::LOW_RESERVED_REJECTIONS) == 1
            && read_word(results, result::LOW_OTHER_REJECTIONS) == 0
            && read_word(results, result::HIGH_SPAWN_OK) == 1;
        let first_poll_pass =
            inject_sequence != 0 && first_sequence > inject_sequence && first_gap <= 1;
        let waker_pass = first_sequence != 0 && resume_sequence > first_sequence && resume_gap <= 1;
        let low_eventual_pass = low_work != 0 && low_completed == low_work;
        let positive_pass = reserved_pass
            && first_poll_pass
            && waker_pass
            && low_eventual_pass
            && positive.spawned_count() == positive.completed_count();

        write_word(results, result::RESERVED_PASS, reserved_pass as u64);
        write_word(results, result::FIRST_POLL_PASS, first_poll_pass as u64);
        write_word(results, result::WAKER_PASS, waker_pass as u64);
        write_word(results, result::LOW_EVENTUAL_PASS, low_eventual_pass as u64);
        write_word(results, result::POSITIVE_PASS, positive_pass as u64);
        write_word(results, result::PHASE, 3);
    }
    syncwarp(mask);

    let negative = &*(negative_executor_ptr as *const GpuExecutor);
    if lane == 0 {
        negative.init();
        let spawned = negative
            .spawn_with_priority(
                BoundedNonYieldingLow {
                    executor: negative as *const GpuExecutor,
                    results,
                },
                Priority::Low,
            )
            .is_ok();
        if !spawned {
            write_word(results, result::LOW_OTHER_REJECTIONS, 1);
        }
        write_word(results, result::PHASE, 4);
    }
    syncwarp(mask);

    let _negative_stats = negative.run(mask);
    syncwarp(mask);

    if lane == 0 {
        let inject_sequence = read_word(results, result::NEG_INJECT_SEQ);
        let return_sequence = read_word(results, result::NEG_LOW_RETURN_SEQ);
        let first_sequence = read_word(results, result::NEG_HIGH_FIRST_SEQ);
        let inject_ns = read_word(results, result::NEG_INJECT_NS);
        let return_ns = read_word(results, result::NEG_LOW_RETURN_NS);
        let first_ns = read_word(results, result::NEG_HIGH_FIRST_NS);
        let no_midpoll = read_word(results, result::NEG_HIGH_SPAWN_OK) == 1
            && inject_sequence != 0
            && return_sequence == inject_sequence + 1
            && first_sequence == return_sequence + 1
            && first_ns >= return_ns
            && return_ns >= inject_ns;
        write_word(
            results,
            result::NEG_LATENCY_NS,
            first_ns.saturating_sub(inject_ns),
        );
        write_word(results, result::NEG_NO_MIDPOLL_PASS, no_midpoll as u64);
        write_word(results, result::PHASE, 5);

        // 合成输入只验证决策输出；不会控制执行器、传感器或真实制动器。
        let fresh = decide_obstacle_action(true, 10, 10, 100, 100);
        let stale = decide_obstacle_action(true, 101, 10, 100, 100);
        let deadline = decide_obstacle_action(true, 10, 101, 100, 100);
        write_decision(
            results,
            result::FRESH_ACTION,
            result::FRESH_REASON,
            result::FRESH_USED_GPU_RESULT,
            fresh,
        );
        write_decision(
            results,
            result::STALE_ACTION,
            result::STALE_REASON,
            result::STALE_USED_GPU_RESULT,
            stale,
        );
        write_decision(
            results,
            result::DEADLINE_ACTION,
            result::DEADLINE_REASON,
            result::DEADLINE_USED_GPU_RESULT,
            deadline,
        );

        let decisions_pass = fresh.action == ACTION_APPLY_BRAKE_FROM_FRESH_RESULT
            && fresh.reason == REASON_FRESH_HAZARD
            && fresh.used_gpu_result == 1
            && stale.action == ACTION_WATCHDOG_CONSERVATIVE_STOP
            && stale.reason == REASON_STALE_RESULT
            && stale.used_gpu_result == 0
            && deadline.action == ACTION_WATCHDOG_CONSERVATIVE_STOP
            && deadline.reason == REASON_DEADLINE_MISS
            && deadline.used_gpu_result == 0;
        let overall =
            read_word(results, result::POSITIVE_PASS) == 1 && no_midpoll && decisions_pass;
        write_word(results, result::DECISION_PASS, decisions_pass as u64);
        write_word(results, result::OVERALL_PASS, overall as u64);
        write_word(results, result::PHASE, 6);
    }
}
