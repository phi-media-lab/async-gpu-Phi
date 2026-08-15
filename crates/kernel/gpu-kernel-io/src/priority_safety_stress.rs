//! Real-GPU executor authority and stale-waker safety litmus.
//!
//! This is a falsifiable integration test, not a production scheduling API.
//! One control warp coordinates seven executor warps across two blocks. The
//! result schema contains observations only; the host recomputes every gate.

use core::cell::UnsafeCell;
use core::future::Future;
use core::mem::MaybeUninit;
use core::pin::Pin;
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use gpu_atomics::{
    activemask, membar_sys, syncwarp, sys_cas_u32, sys_cas_u64, sys_fetch_add_u64,
    sys_load_acquire_u32, sys_load_acquire_u64, sys_spin_load_acquire_u32,
    sys_spin_load_acquire_u64, sys_store_release_u32, sys_store_release_u64,
};
use gpu_runtime::executor::{ExecutorError, GpuExecutor, Priority, TaskKeyError, MAX_TASKS};
use gpu_runtime::priority::{
    level::{High, Low},
    CompletionCell, PriorityClass, PriorityToken, TypedJoinError, TypedJoinHandle,
};

const SCHEMA_VERSION: u64 = 3;
const WARPS_PER_BLOCK: u32 = 4;
const EXPECTED_RUNNER_MASK: u64 = 0xFE;

const PHASE_ONE_ARM: u64 = 1;
const PHASE_ONE_GO: u64 = 2;
const PHASE_TWO_ARM: u64 = 3;
const PHASE_TWO_GO: u64 = 4;
const PHASE_THREE_ARM: u64 = 5;
const PHASE_THREE_GO: u64 = 6;
const PHASE_DONE: u64 = 7;
const MAX_REUSE_QUEUE_FULL_RETRIES: u64 = 65_536;
const MAX_PHASE_THREE_WAIT_ATTEMPTS: u64 = 1_000_000;

const TOKEN_OFFSET: usize = 64;
const DONOR_COMPLETION_OFFSET: usize = 128;
const TARGET_COMPLETION_OFFSET: usize = 160;

const WAKER_EMPTY: u32 = 0;
const WAKER_READY: u32 = 1;
const WAKER_TAKEN: u32 = 2;
const WAKER_WRITING: u32 = 3;
const FIRST_POLL_READY: u64 = 1;
const FIRST_POLL_WRITING: u64 = 2;

/// Host/kernel result schema. The host duplicates these indices deliberately
/// and rejects version/word-count drift before interpreting observations.
mod result {
    pub const VERSION: usize = 0;
    pub const WORDS: usize = 1;
    pub const PHASE: usize = 2;
    pub const KERNEL_ERROR: usize = 3;
    pub const EXPECTED_RUNNER_MASK: usize = 4;
    pub const PHASE_ONE_ENTER_MASK: usize = 5;
    pub const PHASE_ONE_EXIT_MASK: usize = 6;
    pub const PHASE_TWO_ENTER_MASK: usize = 7;
    pub const PHASE_TWO_EXIT_MASK: usize = 8;
    pub const POLL_BLOCK_MASK: usize = 9;
    pub const CAPTURE_BLOCK_MASK: usize = 10;
    pub const OLD_SLOT: usize = 11;
    pub const OLD_GENERATION: usize = 12;
    pub const PIN_CONTEXT_REFS: usize = 13;
    pub const PIN_PENDING_RECLAIMS: usize = 14;
    pub const PIN_ACTIVE_TASKS: usize = 15;
    pub const PIN_HIGH_SUCCESSES: usize = 16;
    pub const PIN_NO_FREE_SLOTS: usize = 17;
    pub const AFTER_DROP_CONTEXT_REFS: usize = 18;
    pub const AFTER_DROP_PENDING_RECLAIMS: usize = 19;
    pub const REUSE_SPAWN_OK: usize = 20;
    pub const NEW_SLOT: usize = 21;
    pub const NEW_GENERATION: usize = 22;
    pub const OLD_KEY_STALE: usize = 23;
    pub const PHASE_ONE_SPAWNED: usize = 24;
    pub const PHASE_ONE_COMPLETED: usize = 25;
    pub const PHASE_ONE_ACTIVE_FINAL: usize = 26;
    pub const PHASE_TWO_SPAWN_SUCCESSES: usize = 27;
    pub const TOTAL_SPAWNED: usize = 28;
    pub const TOTAL_COMPLETED: usize = 29;
    pub const TOTAL_ACTIVE_FINAL: usize = 30;
    pub const SHUTDOWN_COMPLETE: usize = 31;
    pub const WAKER_STATE_FINAL: usize = 32;
    pub const HOLD_RELEASE: usize = 33;
    pub const STORED_WAKER_SIZE: usize = 34;
    pub const STORED_WAKER_ALIGN: usize = 35;
    pub const TERMINAL_WAKE_ACTIVE_BEFORE: usize = 36;
    pub const TERMINAL_WAKE_ACTIVE_AFTER: usize = 37;
    pub const TERMINAL_WAKE_COMPLETED_BEFORE: usize = 38;
    pub const TERMINAL_WAKE_COMPLETED_AFTER: usize = 39;
    pub const RUNNER_TASKS_BASE: usize = 40;
    pub const RUNNER_POLLS_BASE: usize = 48;
    pub const TERMINAL_WAKE_RECLAIMS_BEFORE: usize = 56;
    pub const TERMINAL_WAKE_RECLAIMS_AFTER: usize = 57;
    pub const TERMINAL_WAKE_SPAWNED_BEFORE: usize = 58;
    pub const TERMINAL_WAKE_SPAWNED_AFTER: usize = 59;
    pub const REUSE_QUEUE_FULL_RETRIES: usize = 60;
    pub const REUSE_ERROR_CODE: usize = 61;
    pub const TWO_POLL_COUNTS_BASE: usize = 64;
    pub const TWO_COMPLETE_COUNTS_BASE: usize = 320;
    pub const PHASE_THREE_ENTER_MASK: usize = 576;
    pub const PHASE_THREE_EXIT_MASK: usize = 577;
    pub const TOKEN_STATE_FINAL: usize = 578;
    pub const STORED_TOKEN_SIZE: usize = 579;
    pub const STORED_TOKEN_ALIGN: usize = 580;
    pub const DONOR_SPAWN_OK: usize = 581;
    pub const DONOR_HANDLE_DROPPED: usize = 582;
    pub const DONOR_TOKEN_PUBLISHED: usize = 583;
    pub const TARGET_SPAWN_OK: usize = 584;
    pub const TARGET_FIRST_POLLED: usize = 585;
    pub const TARGET_POLL_COUNT: usize = 586;
    pub const SHUTDOWN_REQUESTED: usize = 587;
    pub const FIRST_WAIT_CANCELLED: usize = 588;
    pub const SECOND_WAIT_ALREADY_JOINED: usize = 589;
    pub const WAIT_POLL_ATTEMPTS: usize = 590;
    pub const TOKEN_DROPPED: usize = 591;
    pub const HANDLE_DROPPED: usize = 592;
    pub const FINAL_CONTEXT_REFS: usize = 593;
    pub const FINAL_PENDING_RECLAIMS: usize = 594;
    pub const FINAL_ACTIVE: usize = 595;
    pub const FINAL_SPAWNED: usize = 596;
    pub const FINAL_COMPLETED: usize = 597;
    pub const FINAL_CAN_TEARDOWN: usize = 598;
    pub const PHASE_THREE_SPAWN_SUCCESSES: usize = 599;
    pub const EVENT_SEQUENCE: usize = 600;
    pub const FIRST_PENDING_SEQUENCE: usize = 601;
    pub const SHUTDOWN_SEQUENCE: usize = 602;
    pub const CANCELLED_SEQUENCE: usize = 603;
    pub const ALREADY_JOINED_SEQUENCE: usize = 604;
    pub const DROP_SEQUENCE: usize = 605;
    pub const TEARDOWN_SEQUENCE: usize = 606;
    pub const PRE_JOIN_PENDING_RECLAIMS: usize = 607;
    pub const POST_JOIN_PENDING_RECLAIMS: usize = 608;
    pub const POST_DROP_PENDING_RECLAIMS: usize = 609;
    pub const PRE_JOIN_CONTEXT_REFS: usize = 610;
    pub const POST_JOIN_CONTEXT_REFS: usize = 611;
    pub const POST_DROP_CONTEXT_REFS: usize = 612;
    pub const TARGET_SELF_WAKE_ISSUED: usize = 613;
    pub const WORD_COUNT: usize = 616;
}

#[repr(C, align(16))]
struct StoredWaker {
    state: UnsafeCell<u32>,
    value: UnsafeCell<MaybeUninit<Waker>>,
}

unsafe impl Send for StoredWaker {}
unsafe impl Sync for StoredWaker {}

impl StoredWaker {
    const fn new() -> Self {
        Self {
            state: UnsafeCell::new(WAKER_EMPTY),
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    /// Publish the sole owned clone. The Waker bytes precede the release state.
    unsafe fn publish(&self, waker: Waker) -> Result<(), Waker> {
        if sys_cas_u32(self.state.get(), WAKER_EMPTY, WAKER_WRITING) != WAKER_EMPTY {
            return Err(waker);
        }
        core::ptr::write((*self.value.get()).as_mut_ptr(), waker);
        sys_store_release_u32(self.state.get(), WAKER_READY);
        Ok(())
    }

    /// The control warp is the unique taker. CAS makes accidental double-take
    /// observable; the preceding acquire pairs with `publish`'s release.
    unsafe fn take(&self) -> Option<Waker> {
        if sys_load_acquire_u32(self.state.get()) != WAKER_READY
            || sys_cas_u32(self.state.get(), WAKER_READY, WAKER_TAKEN) != WAKER_READY
        {
            return None;
        }
        membar_sys();
        Some(core::ptr::read((*self.value.get()).as_ptr()))
    }

    unsafe fn state(&self) -> u32 {
        sys_spin_load_acquire_u32(self.state.get())
    }
}

#[repr(C, align(16))]
struct StoredToken {
    state: UnsafeCell<u32>,
    value: UnsafeCell<MaybeUninit<PriorityToken<Low>>>,
}

unsafe impl Send for StoredToken {}
unsafe impl Sync for StoredToken {}

impl StoredToken {
    const fn new() -> Self {
        Self {
            state: UnsafeCell::new(WAKER_EMPTY),
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    /// Move the sole executor-issued token into mapped storage. A losing
    /// duplicate publisher retains ownership through the returned `Err`.
    unsafe fn publish(&self, token: PriorityToken<Low>) -> Result<(), PriorityToken<Low>> {
        if sys_cas_u32(self.state.get(), WAKER_EMPTY, WAKER_WRITING) != WAKER_EMPTY {
            return Err(token);
        }
        core::ptr::write((*self.value.get()).as_mut_ptr(), token);
        sys_store_release_u32(self.state.get(), WAKER_READY);
        Ok(())
    }

    /// The control warp is the unique taker. The acquire/CAS/read sequence
    /// transfers ownership; the host never interprets or drops these bytes.
    unsafe fn take(&self) -> Option<PriorityToken<Low>> {
        if sys_load_acquire_u32(self.state.get()) != WAKER_READY
            || sys_cas_u32(self.state.get(), WAKER_READY, WAKER_TAKEN) != WAKER_READY
        {
            return None;
        }
        membar_sys();
        Some(core::ptr::read((*self.value.get()).as_ptr()))
    }

    unsafe fn state(&self) -> u32 {
        sys_spin_load_acquire_u32(self.state.get())
    }
}

unsafe fn noop_waker_clone(data: *const ()) -> RawWaker {
    RawWaker::new(data, &NOOP_WAKER_VTABLE)
}

unsafe fn noop_waker_action(_data: *const ()) {}

static NOOP_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
    noop_waker_clone,
    noop_waker_action,
    noop_waker_action,
    noop_waker_action,
);

fn noop_waker() -> Waker {
    unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &NOOP_WAKER_VTABLE)) }
}

#[inline(always)]
unsafe fn word_ptr(results: *mut u64, index: usize) -> *mut u64 {
    results.add(index)
}

#[inline(always)]
unsafe fn read_word(results: *mut u64, index: usize) -> u64 {
    sys_load_acquire_u64(word_ptr(results, index))
}

#[inline(always)]
unsafe fn write_word(results: *mut u64, index: usize, value: u64) {
    sys_store_release_u64(word_ptr(results, index), value);
}

#[inline(always)]
unsafe fn add_word(results: *mut u64, index: usize, value: u64) -> u64 {
    sys_fetch_add_u64(word_ptr(results, index), value)
}

#[inline(always)]
unsafe fn record_event(results: *mut u64, index: usize) {
    let sequence = add_word(results, result::EVENT_SEQUENCE, 1) + 1;
    write_word(results, index, sequence);
}

#[inline(always)]
unsafe fn or_word(results: *mut u64, index: usize, bits: u64) {
    let ptr = word_ptr(results, index);
    loop {
        let old = sys_spin_load_acquire_u64(ptr);
        if sys_cas_u64(ptr, old, old | bits) == old {
            return;
        }
    }
}

#[inline(always)]
unsafe fn record_error(results: *mut u64, code: u64) {
    let ptr = word_ptr(results, result::KERNEL_ERROR);
    let _ = sys_cas_u64(ptr, 0, code);
}

const fn executor_error_code(error: ExecutorError) -> u64 {
    match error {
        ExecutorError::QueueFull => 1,
        ExecutorError::NoFreeSlots => 2,
        ExecutorError::FutureTooLarge => 3,
        ExecutorError::FutureTooAligned => 4,
        ExecutorError::ReservedCapacity => 5,
        ExecutorError::CompletionUnavailable => 6,
        ExecutorError::Shutdown => 7,
        ExecutorError::TraceIdExhausted => 8,
        ExecutorError::IncarnationExhausted => 9,
    }
}

#[inline(always)]
unsafe fn wait_for_phase(results: *mut u64, phase: u64) {
    while sys_spin_load_acquire_u64(word_ptr(results, result::PHASE)) < phase {}
}

#[inline(always)]
unsafe fn wait_for_mask(results: *mut u64, index: usize, expected: u64) {
    while sys_spin_load_acquire_u64(word_ptr(results, index)) & expected != expected {}
}

struct CaptureWaker {
    storage: *const StoredWaker,
    results: *mut u64,
}

impl Future for CaptureWaker {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        unsafe {
            let block = core::arch::nvptx::_block_idx_x() as u64;
            or_word(self.results, result::CAPTURE_BLOCK_MASK, 1 << block);
            if (&*self.storage).publish(cx.waker().clone()).is_err() {
                record_error(self.results, 1);
            }
        }
        Poll::Ready(())
    }
}

struct HoldUntilRelease {
    release: *const u64,
}

impl Future for HoldUntilRelease {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        if unsafe { sys_load_acquire_u64(self.release) } == 0 {
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    }
}

struct TwoPoll {
    id: usize,
    results: *mut u64,
    first: bool,
}

struct ExportToken {
    token: Option<PriorityToken<Low>>,
    storage: *const StoredToken,
    results: *mut u64,
}

impl Future for ExportToken {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        let Some(token) = self.token.take() else {
            unsafe { record_error(self.results, 6) };
            return Poll::Ready(());
        };
        unsafe {
            if (&*self.storage).publish(token).is_err() {
                record_error(self.results, 7);
            } else {
                write_word(self.results, result::DONOR_TOKEN_PUBLISHED, 1);
            }
        }
        Poll::Ready(())
    }
}

struct PendingUntilShutdown {
    results: *mut u64,
}

impl Future for PendingUntilShutdown {
    type Output = u64;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<u64> {
        unsafe {
            add_word(self.results, result::TARGET_POLL_COUNT, 1);
            if sys_cas_u64(
                word_ptr(self.results, result::TARGET_FIRST_POLLED),
                0,
                FIRST_POLL_WRITING,
            ) == 0
            {
                cx.waker().wake_by_ref();
                write_word(self.results, result::TARGET_SELF_WAKE_ISSUED, 1);
                record_event(self.results, result::FIRST_PENDING_SEQUENCE);
                write_word(self.results, result::TARGET_FIRST_POLLED, FIRST_POLL_READY);
            }
        }
        Poll::Pending
    }
}

impl Future for TwoPoll {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        unsafe {
            add_word(self.results, result::TWO_POLL_COUNTS_BASE + self.id, 1);
            let block = core::arch::nvptx::_block_idx_x() as u64;
            or_word(self.results, result::POLL_BLOCK_MASK, 1 << block);
        }
        if !self.first {
            self.first = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        } else {
            unsafe {
                add_word(self.results, result::TWO_COMPLETE_COUNTS_BASE + self.id, 1);
            }
            Poll::Ready(())
        }
    }
}

unsafe fn runner(
    executor: &GpuExecutor,
    results: *mut u64,
    global_warp: usize,
    lane: u32,
    mask: u32,
) {
    let bit = 1u64 << global_warp;

    wait_for_phase(results, PHASE_ONE_ARM);
    if lane == 0 {
        or_word(results, result::PHASE_ONE_ENTER_MASK, bit);
    }
    syncwarp(mask);
    wait_for_phase(results, PHASE_ONE_GO);
    let first = executor.run(mask);
    syncwarp(mask);
    if lane == 0 {
        write_word(
            results,
            result::RUNNER_TASKS_BASE + global_warp,
            first.tasks_executed as u64,
        );
        write_word(
            results,
            result::RUNNER_POLLS_BASE + global_warp,
            first.polls_total as u64,
        );
        or_word(results, result::PHASE_ONE_EXIT_MASK, bit);
    }
    syncwarp(mask);

    wait_for_phase(results, PHASE_TWO_ARM);
    if lane == 0 {
        or_word(results, result::PHASE_TWO_ENTER_MASK, bit);
    }
    syncwarp(mask);
    wait_for_phase(results, PHASE_TWO_GO);
    let second = executor.run(mask);
    syncwarp(mask);
    if lane == 0 {
        write_word(
            results,
            result::RUNNER_TASKS_BASE + global_warp,
            read_word(results, result::RUNNER_TASKS_BASE + global_warp)
                + second.tasks_executed as u64,
        );
        write_word(
            results,
            result::RUNNER_POLLS_BASE + global_warp,
            read_word(results, result::RUNNER_POLLS_BASE + global_warp) + second.polls_total as u64,
        );
        or_word(results, result::PHASE_TWO_EXIT_MASK, bit);
    }
    syncwarp(mask);

    wait_for_phase(results, PHASE_THREE_ARM);
    if lane == 0 {
        or_word(results, result::PHASE_THREE_ENTER_MASK, bit);
    }
    syncwarp(mask);
    wait_for_phase(results, PHASE_THREE_GO);
    let third = executor.run(mask);
    syncwarp(mask);
    if lane == 0 {
        write_word(
            results,
            result::RUNNER_TASKS_BASE + global_warp,
            read_word(results, result::RUNNER_TASKS_BASE + global_warp)
                + third.tasks_executed as u64,
        );
        write_word(
            results,
            result::RUNNER_POLLS_BASE + global_warp,
            read_word(results, result::RUNNER_POLLS_BASE + global_warp) + third.polls_total as u64,
        );
        or_word(results, result::PHASE_THREE_EXIT_MASK, bit);
    }
}

unsafe fn control(executor: &GpuExecutor, storage_ptr: *mut StoredWaker, results: *mut u64) {
    let mut index = 0usize;
    while index < result::WORD_COUNT {
        core::ptr::write_volatile(results.add(index), 0);
        index += 1;
    }
    let storage_bytes = storage_ptr.cast::<u8>();
    let token_ptr = storage_bytes.add(TOKEN_OFFSET).cast::<StoredToken>();
    let donor_completion_ptr = storage_bytes
        .add(DONOR_COMPLETION_OFFSET)
        .cast::<CompletionCell<()>>();
    let target_completion_ptr = storage_bytes
        .add(TARGET_COMPLETION_OFFSET)
        .cast::<CompletionCell<u64>>();
    core::ptr::write(storage_ptr, StoredWaker::new());
    core::ptr::write(token_ptr, StoredToken::new());
    core::ptr::write(donor_completion_ptr, CompletionCell::new());
    core::ptr::write(target_completion_ptr, CompletionCell::new());
    let storage = &*storage_ptr;
    let token_storage = &*token_ptr;
    let donor_completion: &'static CompletionCell<()> = &*donor_completion_ptr;
    let target_completion: &'static CompletionCell<u64> = &*target_completion_ptr;
    write_word(results, result::VERSION, SCHEMA_VERSION);
    write_word(results, result::WORDS, result::WORD_COUNT as u64);
    write_word(results, result::EXPECTED_RUNNER_MASK, EXPECTED_RUNNER_MASK);
    write_word(
        results,
        result::STORED_WAKER_SIZE,
        core::mem::size_of::<StoredWaker>() as u64,
    );
    write_word(
        results,
        result::STORED_WAKER_ALIGN,
        core::mem::align_of::<StoredWaker>() as u64,
    );
    write_word(
        results,
        result::STORED_TOKEN_SIZE,
        core::mem::size_of::<StoredToken>() as u64,
    );
    write_word(
        results,
        result::STORED_TOKEN_ALIGN,
        core::mem::align_of::<StoredToken>() as u64,
    );

    executor.init_with_namespace(0x5AFE_0001);
    let old_key =
        match executor.spawn_with_priority(CaptureWaker { storage, results }, Priority::High) {
            Ok(key) => key,
            Err(_) => {
                record_error(results, 2);
                write_word(results, result::PHASE, PHASE_DONE);
                return;
            }
        };
    write_word(results, result::OLD_SLOT, old_key.slot() as u64);
    write_word(results, result::OLD_GENERATION, old_key.generation());

    write_word(results, result::PHASE, PHASE_ONE_ARM);
    wait_for_mask(results, result::PHASE_ONE_ENTER_MASK, EXPECTED_RUNNER_MASK);
    write_word(results, result::PHASE, PHASE_ONE_GO);

    while storage.state() != WAKER_READY
        || executor.completed_count() != 1
        || executor.outstanding_context_refs() != 1
        || executor.pending_reclaims() != 1
    {
        gpu_runtime::thread::sleep_nanos(1_024);
    }
    write_word(
        results,
        result::PIN_CONTEXT_REFS,
        executor.outstanding_context_refs() as u64,
    );
    write_word(
        results,
        result::PIN_PENDING_RECLAIMS,
        executor.pending_reclaims() as u64,
    );
    write_word(
        results,
        result::PIN_ACTIVE_TASKS,
        executor.active_count() as u64,
    );

    let release = word_ptr(results, result::HOLD_RELEASE) as *const u64;
    let mut high_successes = 0usize;
    let mut terminal_error = 0u64;
    while high_successes < MAX_TASKS {
        match executor.spawn_with_priority(HoldUntilRelease { release }, Priority::High) {
            Ok(_) => high_successes += 1,
            Err(ExecutorError::NoFreeSlots) => {
                terminal_error = 1;
                break;
            }
            Err(_) => {
                terminal_error = 2;
                break;
            }
        }
    }
    write_word(results, result::PIN_HIGH_SUCCESSES, high_successes as u64);
    write_word(results, result::PIN_NO_FREE_SLOTS, terminal_error);

    match storage.take() {
        Some(waker) => {
            write_word(
                results,
                result::TERMINAL_WAKE_ACTIVE_BEFORE,
                executor.active_count() as u64,
            );
            write_word(
                results,
                result::TERMINAL_WAKE_COMPLETED_BEFORE,
                executor.completed_count() as u64,
            );
            write_word(
                results,
                result::TERMINAL_WAKE_RECLAIMS_BEFORE,
                executor.pending_reclaims() as u64,
            );
            write_word(
                results,
                result::TERMINAL_WAKE_SPAWNED_BEFORE,
                executor.spawned_count() as u64,
            );
            waker.wake_by_ref();
            write_word(
                results,
                result::TERMINAL_WAKE_ACTIVE_AFTER,
                executor.active_count() as u64,
            );
            write_word(
                results,
                result::TERMINAL_WAKE_COMPLETED_AFTER,
                executor.completed_count() as u64,
            );
            write_word(
                results,
                result::TERMINAL_WAKE_RECLAIMS_AFTER,
                executor.pending_reclaims() as u64,
            );
            write_word(
                results,
                result::TERMINAL_WAKE_SPAWNED_AFTER,
                executor.spawned_count() as u64,
            );
            drop(waker);
        }
        None => record_error(results, 3),
    }
    write_word(
        results,
        result::AFTER_DROP_CONTEXT_REFS,
        executor.outstanding_context_refs() as u64,
    );
    write_word(
        results,
        result::AFTER_DROP_PENDING_RECLAIMS,
        executor.pending_reclaims() as u64,
    );

    let mut queue_full_retries = 0u64;
    loop {
        match executor.spawn_with_priority(HoldUntilRelease { release }, Priority::High) {
            Ok(new_key) => {
                write_word(results, result::REUSE_SPAWN_OK, 1);
                write_word(results, result::NEW_SLOT, new_key.slot() as u64);
                write_word(results, result::NEW_GENERATION, new_key.generation());
                break;
            }
            Err(ExecutorError::QueueFull) if queue_full_retries < MAX_REUSE_QUEUE_FULL_RETRIES => {
                queue_full_retries += 1;
                gpu_runtime::thread::sleep_nanos(1_024);
            }
            Err(error) => {
                write_word(
                    results,
                    result::REUSE_ERROR_CODE,
                    executor_error_code(error),
                );
                record_error(results, 4);
                break;
            }
        }
    }
    write_word(
        results,
        result::REUSE_QUEUE_FULL_RETRIES,
        queue_full_retries,
    );
    write_word(
        results,
        result::OLD_KEY_STALE,
        matches!(
            executor.set_effective_priority(old_key, Priority::Low),
            Err(TaskKeyError::Stale)
        ) as u64,
    );

    write_word(results, result::HOLD_RELEASE, 1);
    while executor.active_count() != 0
        || executor.outstanding_context_refs() != 0
        || executor.pending_reclaims() != 0
    {
        gpu_runtime::thread::sleep_nanos(1_024);
    }
    wait_for_mask(results, result::PHASE_ONE_EXIT_MASK, EXPECTED_RUNNER_MASK);
    write_word(
        results,
        result::PHASE_ONE_SPAWNED,
        executor.spawned_count() as u64,
    );
    write_word(
        results,
        result::PHASE_ONE_COMPLETED,
        executor.completed_count() as u64,
    );
    write_word(
        results,
        result::PHASE_ONE_ACTIVE_FINAL,
        executor.active_count() as u64,
    );

    let mut two_poll_successes = 0usize;
    while two_poll_successes < MAX_TASKS {
        match executor.spawn_with_priority(
            TwoPoll {
                id: two_poll_successes,
                results,
                first: false,
            },
            Priority::High,
        ) {
            Ok(_) => two_poll_successes += 1,
            Err(_) => {
                record_error(results, 5);
                break;
            }
        }
    }
    write_word(
        results,
        result::PHASE_TWO_SPAWN_SUCCESSES,
        two_poll_successes as u64,
    );
    write_word(results, result::PHASE, PHASE_TWO_ARM);
    wait_for_mask(results, result::PHASE_TWO_ENTER_MASK, EXPECTED_RUNNER_MASK);
    write_word(results, result::PHASE, PHASE_TWO_GO);
    wait_for_mask(results, result::PHASE_TWO_EXIT_MASK, EXPECTED_RUNNER_MASK);

    let mut phase_three_spawn_successes = 0u64;
    match executor.spawn_typed(
        PriorityClass::<Low>::new(),
        donor_completion,
        move |token| ExportToken {
            token: Some(token),
            storage: token_ptr,
            results,
        },
    ) {
        Ok(handle) => {
            phase_three_spawn_successes += 1;
            write_word(results, result::DONOR_SPAWN_OK, 1);
            drop(handle);
            write_word(results, result::DONOR_HANDLE_DROPPED, 1);
        }
        Err(error) => record_error(results, 100 + executor_error_code(error)),
    }

    let mut target_handle: Option<TypedJoinHandle<High, u64>> = match executor.spawn_typed(
        PriorityClass::<High>::new(),
        target_completion,
        move |_token| PendingUntilShutdown { results },
    ) {
        Ok(handle) => {
            phase_three_spawn_successes += 1;
            write_word(results, result::TARGET_SPAWN_OK, 1);
            Some(handle)
        }
        Err(error) => {
            record_error(results, 120 + executor_error_code(error));
            None
        }
    };
    write_word(
        results,
        result::PHASE_THREE_SPAWN_SUCCESSES,
        phase_three_spawn_successes,
    );

    write_word(results, result::PHASE, PHASE_THREE_ARM);
    wait_for_mask(
        results,
        result::PHASE_THREE_ENTER_MASK,
        EXPECTED_RUNNER_MASK,
    );
    write_word(results, result::PHASE, PHASE_THREE_GO);

    let mut readiness_attempts = 0u64;
    while readiness_attempts < MAX_PHASE_THREE_WAIT_ATTEMPTS
        && (token_storage.state() != WAKER_READY
            || read_word(results, result::TARGET_FIRST_POLLED) != FIRST_POLL_READY)
    {
        readiness_attempts += 1;
        gpu_runtime::thread::sleep_nanos(1_024);
    }
    if token_storage.state() != WAKER_READY
        || read_word(results, result::TARGET_FIRST_POLLED) != FIRST_POLL_READY
    {
        record_error(results, 140);
    }

    executor.shutdown();
    write_word(results, result::SHUTDOWN_REQUESTED, 1);
    record_event(results, result::SHUTDOWN_SEQUENCE);

    let expected_completed = (MAX_TASKS as u32) * 2 + 3;
    let mut cancellation_attempts = 0u64;
    while cancellation_attempts < MAX_PHASE_THREE_WAIT_ATTEMPTS
        && (executor.active_count() != 0 || executor.completed_count() != expected_completed)
    {
        cancellation_attempts += 1;
        gpu_runtime::thread::sleep_nanos(1_024);
    }
    if executor.active_count() != 0 || executor.completed_count() != expected_completed {
        record_error(results, 141);
    }
    write_word(
        results,
        result::PRE_JOIN_PENDING_RECLAIMS,
        executor.pending_reclaims() as u64,
    );
    write_word(
        results,
        result::PRE_JOIN_CONTEXT_REFS,
        executor.outstanding_context_refs() as u64,
    );

    let token = token_storage.take();
    let mut wait_poll_attempts = 0u64;
    let mut cancelled = false;
    if let (Some(token_ref), Some(handle)) = (token.as_ref(), target_handle.as_mut()) {
        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        while wait_poll_attempts < MAX_PHASE_THREE_WAIT_ATTEMPTS {
            wait_poll_attempts += 1;
            let mut wait = token_ref.wait_for(handle);
            match Pin::new_unchecked(&mut wait).poll(&mut context) {
                Poll::Ready(Err(TypedJoinError::Cancelled)) => {
                    cancelled = true;
                    write_word(results, result::FIRST_WAIT_CANCELLED, 1);
                    record_event(results, result::CANCELLED_SEQUENCE);
                    break;
                }
                Poll::Ready(_) => {
                    record_error(results, 142);
                    break;
                }
                Poll::Pending => gpu_runtime::thread::sleep_nanos(1_024),
            }
        }
        if !cancelled {
            record_error(results, 143);
        }

        if cancelled {
            let mut replay = token_ref.wait_for(handle);
            match Pin::new_unchecked(&mut replay).poll(&mut context) {
                Poll::Ready(Err(TypedJoinError::AlreadyJoined)) => {
                    write_word(results, result::SECOND_WAIT_ALREADY_JOINED, 1);
                    record_event(results, result::ALREADY_JOINED_SEQUENCE);
                }
                _ => record_error(results, 144),
            }
        }
    } else {
        record_error(results, 145);
    }
    write_word(results, result::WAIT_POLL_ATTEMPTS, wait_poll_attempts);
    write_word(
        results,
        result::POST_JOIN_PENDING_RECLAIMS,
        executor.pending_reclaims() as u64,
    );
    write_word(
        results,
        result::POST_JOIN_CONTEXT_REFS,
        executor.outstanding_context_refs() as u64,
    );

    if let Some(handle) = target_handle.take() {
        drop(handle);
        write_word(results, result::HANDLE_DROPPED, 1);
    }
    if let Some(token) = token {
        drop(token);
        write_word(results, result::TOKEN_DROPPED, 1);
    }
    record_event(results, result::DROP_SEQUENCE);
    write_word(
        results,
        result::POST_DROP_PENDING_RECLAIMS,
        executor.pending_reclaims() as u64,
    );
    write_word(
        results,
        result::POST_DROP_CONTEXT_REFS,
        executor.outstanding_context_refs() as u64,
    );

    wait_for_mask(results, result::PHASE_THREE_EXIT_MASK, EXPECTED_RUNNER_MASK);
    write_word(
        results,
        result::FINAL_CONTEXT_REFS,
        executor.outstanding_context_refs() as u64,
    );
    write_word(
        results,
        result::FINAL_PENDING_RECLAIMS,
        executor.pending_reclaims() as u64,
    );
    write_word(
        results,
        result::FINAL_ACTIVE,
        executor.active_count() as u64,
    );
    write_word(
        results,
        result::FINAL_SPAWNED,
        executor.spawned_count() as u64,
    );
    write_word(
        results,
        result::FINAL_COMPLETED,
        executor.completed_count() as u64,
    );
    write_word(
        results,
        result::FINAL_CAN_TEARDOWN,
        executor.can_teardown() as u64,
    );
    record_event(results, result::TEARDOWN_SEQUENCE);

    write_word(
        results,
        result::TOTAL_SPAWNED,
        executor.spawned_count() as u64,
    );
    write_word(
        results,
        result::TOTAL_COMPLETED,
        executor.completed_count() as u64,
    );
    write_word(
        results,
        result::TOTAL_ACTIVE_FINAL,
        executor.active_count() as u64,
    );
    write_word(
        results,
        result::SHUTDOWN_COMPLETE,
        executor.can_teardown() as u64,
    );
    write_word(results, result::WAKER_STATE_FINAL, storage.state() as u64);
    write_word(
        results,
        result::TOKEN_STATE_FINAL,
        token_storage.state() as u64,
    );
    write_word(results, result::PHASE, PHASE_DONE);
}

/// Two-block, eight-warp real-GPU executor safety litmus.
///
/// `executor_ptr` must name at least 512 KiB of 256-byte-aligned mapped memory;
/// `waker_ptr` must name at least 256 bytes of 16-byte-aligned mapped memory;
/// `results` must name `u64[616]` mapped memory initialized to zero.
#[no_mangle]
pub unsafe extern "gpu-kernel" fn priority_handle_safety_stress(
    executor_ptr: *mut u8,
    waker_ptr: *mut u8,
    results: *mut u64,
) {
    let block = core::arch::nvptx::_block_idx_x() as u32;
    let thread = core::arch::nvptx::_thread_idx_x() as u32;
    let warp = thread / 32;
    let lane = thread % 32;
    let global_warp = (block * WARPS_PER_BLOCK + warp) as usize;
    let mask = activemask();
    let executor = &*(executor_ptr as *const GpuExecutor);

    if global_warp == 0 {
        if lane == 0 {
            control(executor, waker_ptr.cast::<StoredWaker>(), results);
        }
    } else {
        runner(executor, results, global_warp, lane, mask);
        syncwarp(mask);
    }
}
