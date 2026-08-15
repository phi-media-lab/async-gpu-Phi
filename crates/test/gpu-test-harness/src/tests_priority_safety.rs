//! Real-GPU stale-waker, generation, and multi-block executor safety litmus.

use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use gpu_host::error::{GpuHostError, Result};
use gpu_host::mapped_mem::{
    alloc_mapped_bytes, alloc_mapped_u64_array, free_mapped_bytes, free_mapped_u64_array,
};

const TEST_NAME: &str = "priority_handle_safety_stress";
const EXECUTOR_BYTES: usize = 512 * 1024;
const STORAGE_BYTES: usize = 256;
const TOKEN_STATE_OFFSET: usize = 64;
const RESULT_WORDS: usize = 616;
const KERNEL_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_TASKS: u64 = 256;
const EXPECTED_RUNNER_MASK: u64 = 0xFE;
const EXPECTED_BLOCK_MASK: u64 = 0b11;
const EXPECTED_PHASE_ONE_TASKS: u64 = 257;
const EXPECTED_FINAL_TASKS: u64 = 515;
const EXPECTED_RUNNER_READY_TASKS: u64 = 514;
const MINIMUM_TOTAL_POLLS: u64 = 769;
const MAX_REUSE_QUEUE_FULL_RETRIES: u64 = 65_536;
const MAX_PHASE_THREE_WAIT_ATTEMPTS: u64 = 1_000_000;

mod field {
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
    pub const RESERVED_PHASE_THREE_0: usize = 614;
}

struct MappedBytes {
    host: *mut u8,
    dev: cudarc::driver::sys::CUdeviceptr,
}

impl MappedBytes {
    unsafe fn allocate(dev: &Arc<CudaDevice>, bytes: usize) -> Result<Self> {
        let (host, device) = alloc_mapped_bytes(dev, bytes)?;
        Ok(Self { host, dev: device })
    }

    fn leak(self) {
        std::mem::forget(self);
    }
}

impl Drop for MappedBytes {
    fn drop(&mut self) {
        if let Err(error) = unsafe { free_mapped_bytes(self.host) } {
            eprintln!("mapped-byte cleanup failed: {error}");
        }
    }
}

struct MappedWords {
    host: *mut u64,
    dev: cudarc::driver::sys::CUdeviceptr,
}

impl MappedWords {
    unsafe fn allocate(dev: &Arc<CudaDevice>, words: usize) -> Result<Self> {
        let (host, device) = alloc_mapped_u64_array(dev, words)?;
        Ok(Self { host, dev: device })
    }

    fn leak(self) {
        std::mem::forget(self);
    }
}

impl Drop for MappedWords {
    fn drop(&mut self) {
        if let Err(error) = unsafe { free_mapped_u64_array(self.host) } {
            eprintln!("mapped-word cleanup failed: {error}");
        }
    }
}

struct SafetyResults {
    words: [u64; RESULT_WORDS],
}

impl SafetyResults {
    unsafe fn read_from(host: *mut u64) -> Self {
        let mut words = [0; RESULT_WORDS];
        for (index, word) in words.iter_mut().enumerate() {
            *word = std::ptr::read_volatile(host.add(index));
        }
        Self { words }
    }

    fn get(&self, index: usize) -> u64 {
        self.words[index]
    }
}

fn gate(failures: &mut Vec<String>, condition: bool, detail: String) {
    if !condition {
        failures.push(detail);
    }
}

fn check_phase_three(results: &SafetyResults, mapped_token_state: u32, failures: &mut Vec<String>) {
    let value = |index| results.get(index);
    for (name, index, expected) in [
        (
            "phase3 enter mask",
            field::PHASE_THREE_ENTER_MASK,
            EXPECTED_RUNNER_MASK,
        ),
        (
            "phase3 exit mask",
            field::PHASE_THREE_EXIT_MASK,
            EXPECTED_RUNNER_MASK,
        ),
        ("token state result", field::TOKEN_STATE_FINAL, 2),
        ("donor spawn", field::DONOR_SPAWN_OK, 1),
        ("donor handle drop", field::DONOR_HANDLE_DROPPED, 1),
        ("donor token publish", field::DONOR_TOKEN_PUBLISHED, 1),
        ("target spawn", field::TARGET_SPAWN_OK, 1),
        ("target first Pending", field::TARGET_FIRST_POLLED, 1),
        ("target self-wake issued", field::TARGET_SELF_WAKE_ISSUED, 1),
        ("shutdown requested", field::SHUTDOWN_REQUESTED, 1),
        ("first wait Cancelled", field::FIRST_WAIT_CANCELLED, 1),
        (
            "second wait AlreadyJoined",
            field::SECOND_WAIT_ALREADY_JOINED,
            1,
        ),
        ("token dropped", field::TOKEN_DROPPED, 1),
        ("handle dropped", field::HANDLE_DROPPED, 1),
        ("final context refs", field::FINAL_CONTEXT_REFS, 0),
        ("final pending reclaims", field::FINAL_PENDING_RECLAIMS, 0),
        ("final active", field::FINAL_ACTIVE, 0),
        ("final spawned", field::FINAL_SPAWNED, EXPECTED_FINAL_TASKS),
        (
            "final completed",
            field::FINAL_COMPLETED,
            EXPECTED_FINAL_TASKS,
        ),
        ("final can_teardown", field::FINAL_CAN_TEARDOWN, 1),
        (
            "phase3 spawn successes",
            field::PHASE_THREE_SPAWN_SUCCESSES,
            2,
        ),
        ("event sequence count", field::EVENT_SEQUENCE, 6),
        ("first Pending sequence", field::FIRST_PENDING_SEQUENCE, 1),
        ("shutdown sequence", field::SHUTDOWN_SEQUENCE, 2),
        ("Cancelled sequence", field::CANCELLED_SEQUENCE, 3),
        ("AlreadyJoined sequence", field::ALREADY_JOINED_SEQUENCE, 4),
        ("drop sequence", field::DROP_SEQUENCE, 5),
        ("teardown sequence", field::TEARDOWN_SEQUENCE, 6),
        (
            "pre-join pending reclaims",
            field::PRE_JOIN_PENDING_RECLAIMS,
            2,
        ),
        (
            "post-join pending reclaims",
            field::POST_JOIN_PENDING_RECLAIMS,
            1,
        ),
        (
            "post-drop pending reclaims",
            field::POST_DROP_PENDING_RECLAIMS,
            0,
        ),
    ] {
        gate(
            failures,
            value(index) == expected,
            format!("{name}={} (expected {expected})", value(index)),
        );
    }
    gate(
        failures,
        mapped_token_state == 2,
        format!("mapped token state={mapped_token_state} (expected 2)"),
    );
    gate(
        failures,
        value(field::STORED_TOKEN_ALIGN) == 16,
        format!(
            "StoredToken align={} (expected 16)",
            value(field::STORED_TOKEN_ALIGN)
        ),
    );
    let token_size = value(field::STORED_TOKEN_SIZE);
    gate(
        failures,
        token_size != 0 && token_size <= 64 && token_size.is_multiple_of(16),
        format!("StoredToken size={token_size} exceeds/alignment-mismatches 64 bytes"),
    );
    gate(
        failures,
        value(field::TARGET_POLL_COUNT) >= 1,
        format!(
            "target poll count={} (minimum 1)",
            value(field::TARGET_POLL_COUNT)
        ),
    );
    let wait_attempts = value(field::WAIT_POLL_ATTEMPTS);
    gate(
        failures,
        (1..=MAX_PHASE_THREE_WAIT_ATTEMPTS).contains(&wait_attempts),
        format!(
            "typed wait poll attempts={wait_attempts} (expected 1..={MAX_PHASE_THREE_WAIT_ATTEMPTS})"
        ),
    );
    gate(
        failures,
        results.words[field::RESERVED_PHASE_THREE_0..RESULT_WORDS]
            .iter()
            .all(|word| *word == 0),
        format!(
            "phase3 reserved words are {:?}",
            &results.words[field::RESERVED_PHASE_THREE_0..RESULT_WORDS]
        ),
    );
}

fn evaluate(
    results: &SafetyResults,
    mapped_waker_state: u32,
    mapped_token_state: u32,
) -> std::result::Result<(), String> {
    let mut failures = Vec::new();
    let value = |index| results.get(index);

    gate(
        &mut failures,
        value(field::VERSION) == 3,
        format!("schema version={} (expected 3)", value(field::VERSION)),
    );
    gate(
        &mut failures,
        value(field::WORDS) == RESULT_WORDS as u64,
        format!(
            "schema words={} (expected {RESULT_WORDS})",
            value(field::WORDS)
        ),
    );
    gate(
        &mut failures,
        value(field::PHASE) == 7,
        format!("terminal phase={} (expected 7)", value(field::PHASE)),
    );
    gate(
        &mut failures,
        value(field::KERNEL_ERROR) == 0,
        format!("kernel error={}", value(field::KERNEL_ERROR)),
    );
    gate(
        &mut failures,
        value(field::EXPECTED_RUNNER_MASK) == EXPECTED_RUNNER_MASK,
        format!(
            "kernel runner mask={:#x} (host expects {EXPECTED_RUNNER_MASK:#x})",
            value(field::EXPECTED_RUNNER_MASK)
        ),
    );
    for (name, index) in [
        ("phase1 enter", field::PHASE_ONE_ENTER_MASK),
        ("phase1 exit", field::PHASE_ONE_EXIT_MASK),
        ("phase2 enter", field::PHASE_TWO_ENTER_MASK),
        ("phase2 exit", field::PHASE_TWO_EXIT_MASK),
        ("phase3 enter", field::PHASE_THREE_ENTER_MASK),
        ("phase3 exit", field::PHASE_THREE_EXIT_MASK),
    ] {
        gate(
            &mut failures,
            value(index) == EXPECTED_RUNNER_MASK,
            format!(
                "{name} mask={:#x} (expected {EXPECTED_RUNNER_MASK:#x})",
                value(index)
            ),
        );
    }
    gate(
        &mut failures,
        value(field::POLL_BLOCK_MASK) == EXPECTED_BLOCK_MASK,
        format!(
            "poll block mask={:#x} (expected {EXPECTED_BLOCK_MASK:#x})",
            value(field::POLL_BLOCK_MASK)
        ),
    );
    let capture_mask = value(field::CAPTURE_BLOCK_MASK);
    gate(
        &mut failures,
        capture_mask != 0 && capture_mask & !EXPECTED_BLOCK_MASK == 0,
        format!("capture block mask={capture_mask:#x}"),
    );
    gate(
        &mut failures,
        value(field::OLD_SLOT) < MAX_TASKS,
        format!("old slot={} is out of range", value(field::OLD_SLOT)),
    );
    gate(
        &mut failures,
        value(field::OLD_GENERATION) != 0,
        "old generation is zero".to_owned(),
    );
    for (name, index, expected) in [
        ("pinned context refs", field::PIN_CONTEXT_REFS, 1),
        ("pinned pending reclaims", field::PIN_PENDING_RECLAIMS, 1),
        ("pinned active tasks", field::PIN_ACTIVE_TASKS, 0),
        (
            "high successes while pinned",
            field::PIN_HIGH_SUCCESSES,
            MAX_TASKS - 1,
        ),
        ("NoFreeSlots observation", field::PIN_NO_FREE_SLOTS, 1),
        (
            "pending reclaims after drop",
            field::AFTER_DROP_PENDING_RECLAIMS,
            0,
        ),
        ("reuse spawn", field::REUSE_SPAWN_OK, 1),
        ("old-key stale result", field::OLD_KEY_STALE, 1),
        (
            "phase1 spawned",
            field::PHASE_ONE_SPAWNED,
            EXPECTED_PHASE_ONE_TASKS,
        ),
        (
            "phase1 completed",
            field::PHASE_ONE_COMPLETED,
            EXPECTED_PHASE_ONE_TASKS,
        ),
        ("phase1 active", field::PHASE_ONE_ACTIVE_FINAL, 0),
        (
            "phase2 spawn successes",
            field::PHASE_TWO_SPAWN_SUCCESSES,
            MAX_TASKS,
        ),
        ("total spawned", field::TOTAL_SPAWNED, EXPECTED_FINAL_TASKS),
        (
            "total completed",
            field::TOTAL_COMPLETED,
            EXPECTED_FINAL_TASKS,
        ),
        ("total active", field::TOTAL_ACTIVE_FINAL, 0),
        ("shutdown complete", field::SHUTDOWN_COMPLETE, 1),
        ("stored waker final state", field::WAKER_STATE_FINAL, 2),
        ("holder release", field::HOLD_RELEASE, 1),
        (
            "terminal wake active before",
            field::TERMINAL_WAKE_ACTIVE_BEFORE,
            MAX_TASKS - 1,
        ),
        (
            "terminal wake active after",
            field::TERMINAL_WAKE_ACTIVE_AFTER,
            MAX_TASKS - 1,
        ),
        (
            "terminal wake completed before",
            field::TERMINAL_WAKE_COMPLETED_BEFORE,
            1,
        ),
        (
            "terminal wake completed after",
            field::TERMINAL_WAKE_COMPLETED_AFTER,
            1,
        ),
        (
            "terminal wake reclaims before",
            field::TERMINAL_WAKE_RECLAIMS_BEFORE,
            1,
        ),
        (
            "terminal wake reclaims after",
            field::TERMINAL_WAKE_RECLAIMS_AFTER,
            1,
        ),
        (
            "terminal wake spawned before",
            field::TERMINAL_WAKE_SPAWNED_BEFORE,
            MAX_TASKS,
        ),
        (
            "terminal wake spawned after",
            field::TERMINAL_WAKE_SPAWNED_AFTER,
            MAX_TASKS,
        ),
        ("reuse error code", field::REUSE_ERROR_CODE, 0),
    ] {
        gate(
            &mut failures,
            value(index) == expected,
            format!("{name}={} (expected {expected})", value(index)),
        );
    }
    gate(
        &mut failures,
        value(field::NEW_SLOT) == value(field::OLD_SLOT),
        format!(
            "slot reuse old={} new={}",
            value(field::OLD_SLOT),
            value(field::NEW_SLOT)
        ),
    );
    let expected_generation = value(field::OLD_GENERATION)
        .checked_add(1)
        .and_then(|generation| generation.checked_add(value(field::REUSE_QUEUE_FULL_RETRIES)));
    gate(
        &mut failures,
        expected_generation == Some(value(field::NEW_GENERATION)),
        format!(
            "generation reuse old={} retries={} new={}",
            value(field::OLD_GENERATION),
            value(field::REUSE_QUEUE_FULL_RETRIES),
            value(field::NEW_GENERATION)
        ),
    );
    gate(
        &mut failures,
        value(field::REUSE_QUEUE_FULL_RETRIES) <= MAX_REUSE_QUEUE_FULL_RETRIES,
        format!(
            "QueueFull retries={} (maximum {MAX_REUSE_QUEUE_FULL_RETRIES})",
            value(field::REUSE_QUEUE_FULL_RETRIES)
        ),
    );
    gate(
        &mut failures,
        value(field::STORED_WAKER_ALIGN) == 16,
        format!(
            "StoredWaker align={} (expected 16)",
            value(field::STORED_WAKER_ALIGN)
        ),
    );
    let stored_size = value(field::STORED_WAKER_SIZE);
    gate(
        &mut failures,
        stored_size != 0
            && stored_size <= TOKEN_STATE_OFFSET as u64
            && stored_size.is_multiple_of(16),
        format!(
            "StoredWaker size={stored_size} exceeds/alignment-mismatches {TOKEN_STATE_OFFSET} bytes"
        ),
    );
    gate(
        &mut failures,
        mapped_waker_state == 2,
        format!("mapped waker state={mapped_waker_state} (expected 2)"),
    );

    let runner_tasks: u64 = (0..8)
        .map(|warp| value(field::RUNNER_TASKS_BASE + warp))
        .sum();
    let runner_polls: u64 = (0..8)
        .map(|warp| value(field::RUNNER_POLLS_BASE + warp))
        .sum();
    gate(
        &mut failures,
        value(field::RUNNER_TASKS_BASE) == 0 && runner_tasks == EXPECTED_RUNNER_READY_TASKS,
        format!("runner task sum={runner_tasks} (expected {EXPECTED_RUNNER_READY_TASKS})"),
    );
    gate(
        &mut failures,
        value(field::RUNNER_POLLS_BASE) == 0 && runner_polls >= MINIMUM_TOTAL_POLLS,
        format!("runner poll sum={runner_polls} (minimum {MINIMUM_TOTAL_POLLS})"),
    );

    let mut poll_sum = 0u64;
    let mut completion_sum = 0u64;
    for id in 0..MAX_TASKS as usize {
        let polls = value(field::TWO_POLL_COUNTS_BASE + id);
        let completions = value(field::TWO_COMPLETE_COUNTS_BASE + id);
        poll_sum += polls;
        completion_sum += completions;
        if value(field::PHASE_TWO_SPAWN_SUCCESSES) == MAX_TASKS {
            gate(
                &mut failures,
                polls == 2 && completions == 1,
                format!("task {id}: polls={polls}, completions={completions}"),
            );
        }
    }
    gate(
        &mut failures,
        poll_sum == MAX_TASKS * 2 && completion_sum == MAX_TASKS,
        format!("two-poll sums polls={poll_sum}, completions={completion_sum}"),
    );
    gate(
        &mut failures,
        results.words[62..64].iter().all(|word| *word == 0),
        format!("reserved words 62..64 are {:?}", &results.words[62..64]),
    );
    check_phase_three(results, mapped_token_state, &mut failures);

    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; "))
    }
}

/// Run the two-block/8-warp executor safety litmus and independently evaluate
/// every result field after CUDA synchronization.
pub(crate) fn run_priority_handle_safety_stress(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- priority handle safety: stale Waker + generation + cross-block ---");
    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Io,
        "kernel_priority_safety",
        &[TEST_NAME],
    )?;

    let executor = unsafe { MappedBytes::allocate(&dev, EXECUTOR_BYTES)? };
    let stored_waker = unsafe { MappedBytes::allocate(&dev, STORAGE_BYTES)? };
    let results = unsafe { MappedWords::allocate(&dev, RESULT_WORDS)? };
    let function = dev
        .get_func("kernel_priority_safety", TEST_NAME)
        .ok_or(GpuHostError::KernelNotFound(TEST_NAME))?;

    let config = LaunchConfig {
        grid_dim: (2, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (status_tx, status_rx) = std::sync::mpsc::channel();
    let launch_device = Arc::clone(&dev);
    let executor_dev = executor.dev;
    let waker_dev = stored_waker.dev;
    let results_dev = results.dev;
    let launch_thread = std::thread::spawn(move || {
        let status = unsafe { function.launch(config, (executor_dev, waker_dev, results_dev)) }
            .and_then(|_| launch_device.synchronize())
            .map_err(|error| error.to_string());
        let _ = status_tx.send(status);
    });

    let start = Instant::now();
    let mut last_phase = 0u64;
    let launch_status = loop {
        match status_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(status) => break status,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                let thread_panicked = launch_thread.join().is_err();
                executor.leak();
                stored_waker.leak();
                results.leak();
                return Err(GpuHostError::Verification {
                    test: TEST_NAME,
                    detail: format!(
                        "GPU launch thread disconnected (panicked={thread_panicked}); result=UNKNOWN, mapped buffers intentionally retained"
                    ),
                });
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                let phase = unsafe { std::ptr::read_volatile(results.host.add(field::PHASE)) };
                if phase != last_phase {
                    println!("  phase={phase}");
                    last_phase = phase;
                }
                if start.elapsed() >= KERNEL_TIMEOUT {
                    executor.leak();
                    stored_waker.leak();
                    results.leak();
                    return Err(GpuHostError::Timeout {
                        test: TEST_NAME,
                        detail: format!(
                            "30s exceeded at phase={phase}; result=UNKNOWN, mapped buffers intentionally retained"
                        ),
                    });
                }
            }
        }
    };
    if launch_thread.join().is_err() {
        executor.leak();
        stored_waker.leak();
        results.leak();
        return Err(GpuHostError::Verification {
            test: TEST_NAME,
            detail: "GPU launch thread panicked after reporting status; result=UNKNOWN, mapped buffers intentionally retained"
                .to_owned(),
        });
    }
    launch_status.map_err(|detail| GpuHostError::Verification {
        test: TEST_NAME,
        detail: format!("launch/synchronize failed: {detail}; result=UNKNOWN"),
    })?;

    let observations = unsafe { SafetyResults::read_from(results.host) };
    let mapped_waker_state = unsafe { std::ptr::read_volatile(stored_waker.host.cast::<u32>()) };
    let mapped_token_state =
        unsafe { std::ptr::read_volatile(stored_waker.host.add(TOKEN_STATE_OFFSET).cast::<u32>()) };
    let runner_tasks: Vec<_> = (0..8)
        .map(|warp| observations.get(field::RUNNER_TASKS_BASE + warp))
        .collect();
    let runner_polls: Vec<_> = (0..8)
        .map(|warp| observations.get(field::RUNNER_POLLS_BASE + warp))
        .collect();
    println!(
        "  masks: p1={:#x}/{:#x} p2={:#x}/{:#x} p3={:#x}/{:#x} poll-blocks={:#x} capture-block={:#x}",
        observations.get(field::PHASE_ONE_ENTER_MASK),
        observations.get(field::PHASE_ONE_EXIT_MASK),
        observations.get(field::PHASE_TWO_ENTER_MASK),
        observations.get(field::PHASE_TWO_EXIT_MASK),
        observations.get(field::PHASE_THREE_ENTER_MASK),
        observations.get(field::PHASE_THREE_EXIT_MASK),
        observations.get(field::POLL_BLOCK_MASK),
        observations.get(field::CAPTURE_BLOCK_MASK),
    );
    println!(
        "  pin/reuse: refs={}->{} reclaims={}->{} High={} nofree={} queuefull-retries={} reuse-error={} slot={}→{} generation={}→{} stale={}",
        observations.get(field::PIN_CONTEXT_REFS),
        observations.get(field::AFTER_DROP_CONTEXT_REFS),
        observations.get(field::PIN_PENDING_RECLAIMS),
        observations.get(field::AFTER_DROP_PENDING_RECLAIMS),
        observations.get(field::PIN_HIGH_SUCCESSES),
        observations.get(field::PIN_NO_FREE_SLOTS),
        observations.get(field::REUSE_QUEUE_FULL_RETRIES),
        observations.get(field::REUSE_ERROR_CODE),
        observations.get(field::OLD_SLOT),
        observations.get(field::NEW_SLOT),
        observations.get(field::OLD_GENERATION),
        observations.get(field::NEW_GENERATION),
        observations.get(field::OLD_KEY_STALE),
    );
    println!(
        "  typed shutdown: token-state={}/{} donor={}/{} target={}/polls{} shutdown={} wait={}/{} pending={}->{}->{} refs={}->{}->{} seq=[{},{},{},{},{},{}] teardown={}",
        observations.get(field::TOKEN_STATE_FINAL),
        mapped_token_state,
        observations.get(field::DONOR_SPAWN_OK),
        observations.get(field::DONOR_TOKEN_PUBLISHED),
        observations.get(field::TARGET_SPAWN_OK),
        observations.get(field::TARGET_POLL_COUNT),
        observations.get(field::SHUTDOWN_REQUESTED),
        observations.get(field::FIRST_WAIT_CANCELLED),
        observations.get(field::SECOND_WAIT_ALREADY_JOINED),
        observations.get(field::PRE_JOIN_PENDING_RECLAIMS),
        observations.get(field::POST_JOIN_PENDING_RECLAIMS),
        observations.get(field::POST_DROP_PENDING_RECLAIMS),
        observations.get(field::PRE_JOIN_CONTEXT_REFS),
        observations.get(field::POST_JOIN_CONTEXT_REFS),
        observations.get(field::POST_DROP_CONTEXT_REFS),
        observations.get(field::FIRST_PENDING_SEQUENCE),
        observations.get(field::SHUTDOWN_SEQUENCE),
        observations.get(field::CANCELLED_SEQUENCE),
        observations.get(field::ALREADY_JOINED_SEQUENCE),
        observations.get(field::DROP_SEQUENCE),
        observations.get(field::TEARDOWN_SEQUENCE),
        observations.get(field::FINAL_CAN_TEARDOWN),
    );
    println!("  runner tasks={runner_tasks:?} polls={runner_polls:?}");

    evaluate(&observations, mapped_waker_state, mapped_token_state).map_err(|detail| {
        GpuHostError::Verification {
            test: TEST_NAME,
            detail,
        }
    })?;
    println!("  PASS: stale Waker/generation, 256 exact TwoPoll tasks, and typed shutdown Cancelled/AlreadyJoined gates hold");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Barrier;

    #[test]
    fn phase_three_oracle_rejects_default_zero() {
        let results = SafetyResults {
            words: [0; RESULT_WORDS],
        };
        let mut failures = Vec::new();
        check_phase_three(&results, 0, &mut failures);
        assert!(failures
            .iter()
            .any(|failure| failure.contains("donor spawn")));
        assert!(failures
            .iter()
            .any(|failure| failure.contains("first wait Cancelled")));
        assert!(failures
            .iter()
            .any(|failure| failure.contains("mapped token state")));
    }

    #[test]
    fn phase_three_oracle_rejects_early_terminal_markers() {
        let mut words = [0; RESULT_WORDS];
        words[field::PHASE_THREE_ENTER_MASK] = EXPECTED_RUNNER_MASK;
        words[field::PHASE_THREE_EXIT_MASK] = EXPECTED_RUNNER_MASK;
        words[field::TOKEN_STATE_FINAL] = 2;
        words[field::FINAL_CAN_TEARDOWN] = 1;
        words[field::EVENT_SEQUENCE] = 6;
        words[field::TEARDOWN_SEQUENCE] = 6;
        let results = SafetyResults { words };
        let mut failures = Vec::new();
        check_phase_three(&results, 2, &mut failures);
        assert!(failures
            .iter()
            .any(|failure| failure.contains("target first Pending")));
        assert!(failures
            .iter()
            .any(|failure| failure.contains("second wait AlreadyJoined")));
        assert!(failures
            .iter()
            .any(|failure| failure.contains("drop sequence")));
    }

    #[test]
    fn first_pending_publication_orders_wake_and_event_before_ready() {
        const READY: u64 = 1;
        const WRITING: u64 = 2;

        let state = Arc::new(AtomicU64::new(0));
        let wake_issued = Arc::new(AtomicU64::new(0));
        let event_written = Arc::new(AtomicU64::new(0));
        let writer_claimed = Arc::new(Barrier::new(2));
        let allow_publish = Arc::new(Barrier::new(2));

        let writer = {
            let state = Arc::clone(&state);
            let wake_issued = Arc::clone(&wake_issued);
            let event_written = Arc::clone(&event_written);
            let writer_claimed = Arc::clone(&writer_claimed);
            let allow_publish = Arc::clone(&allow_publish);
            std::thread::spawn(move || {
                assert_eq!(
                    state.compare_exchange(0, WRITING, Ordering::AcqRel, Ordering::Acquire),
                    Ok(0)
                );
                writer_claimed.wait();
                allow_publish.wait();
                wake_issued.store(1, Ordering::Relaxed);
                event_written.store(1, Ordering::Relaxed);
                state.store(READY, Ordering::Release);
            })
        };

        writer_claimed.wait();
        assert_eq!(state.load(Ordering::Acquire), WRITING);
        assert_eq!(wake_issued.load(Ordering::Relaxed), 0);
        assert_eq!(event_written.load(Ordering::Relaxed), 0);
        allow_publish.wait();
        while state.load(Ordering::Acquire) != READY {
            std::thread::yield_now();
        }
        assert_eq!(wake_issued.load(Ordering::Relaxed), 1);
        assert_eq!(event_written.load(Ordering::Relaxed), 1);
        writer.join().expect("publication writer panicked");
    }
}
