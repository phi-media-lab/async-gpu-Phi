//! 障碍事件优先级压力测试的 host harness。

use std::sync::mpsc::RecvTimeoutError;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use gpu_host::error::{GpuHostError, Result};
use gpu_host::hostcall::{reuse_schema, HostcallBuffer, PriorityEchoTestHook};
use gpu_host::mapped_mem::{
    alloc_mapped_bytes, alloc_mapped_u64_array, free_mapped_bytes, free_mapped_u64_array,
};

use crate::composed_oracle::{
    evaluate_composed, required_mutation_gate_is_red, ComposedResults, ExpectedRun, OracleVerdict,
};
use crate::harness_support::HostcallListener;

const TEST_NAME: &str = "obstacle_event_priority_stress";
const EXECUTOR_BYTES: usize = 512 * 1024;
const RESULT_WORDS: usize = 48;
const KERNEL_TIMEOUT: Duration = Duration::from_secs(20);

/// 与 kernel 同步维护的结果字段 / Result schema.
mod field {
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

const ACTION_APPLY_BRAKE_FROM_FRESH_RESULT: u64 = 1;
const ACTION_WATCHDOG_CONSERVATIVE_STOP: u64 = 2;
const REASON_FRESH_HAZARD: u64 = 1;
const REASON_STALE_RESULT: u64 = 2;
const REASON_DEADLINE_MISS: u64 = 4;

#[derive(Clone, Copy)]
struct StressResults {
    words: [u64; RESULT_WORDS],
}

impl StressResults {
    unsafe fn read_from(host_ptr: *mut u64) -> Self {
        let mut words = [0u64; RESULT_WORDS];
        let mut index = 0usize;
        while index < RESULT_WORDS {
            words[index] = std::ptr::read_volatile(host_ptr.add(index));
            index += 1;
        }
        Self { words }
    }

    fn get(&self, field: usize) -> u64 {
        self.words[field]
    }

    fn gate_summary(&self) -> String {
        format!(
            "reserved={} first_poll={} waker={} low_eventual={} positive={} no_midpoll={} decisions={} overall={}",
            self.get(field::RESERVED_PASS),
            self.get(field::FIRST_POLL_PASS),
            self.get(field::WAKER_PASS),
            self.get(field::LOW_EVENTUAL_PASS),
            self.get(field::POSITIVE_PASS),
            self.get(field::NEG_NO_MIDPOLL_PASS),
            self.get(field::DECISION_PASS),
            self.get(field::OVERALL_PASS),
        )
    }
}

/// 在真实 CUDA GPU 上运行协作式 priority/admission/waker 压力场景。
pub(crate) fn run_obstacle_event_stress(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- 障碍事件优先级压力测试 / Obstacle-event priority stress ---");
    println!("  语义范围：协作式调度，不是硬实时，也不控制真实机器人执行器。");

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Io,
        "kernel_obstacle_stress",
        &[TEST_NAME],
    )?;

    let (positive_host_ptr, positive_dev_ptr) =
        unsafe { alloc_mapped_bytes(&dev, EXECUTOR_BYTES)? };
    let (negative_host_ptr, negative_dev_ptr) =
        match unsafe { alloc_mapped_bytes(&dev, EXECUTOR_BYTES) } {
            Ok(allocation) => allocation,
            Err(error) => {
                unsafe { free_mapped_bytes(positive_host_ptr)? };
                return Err(error);
            }
        };
    let (results_host_ptr, results_dev_ptr) =
        match unsafe { alloc_mapped_u64_array(&dev, RESULT_WORDS) } {
            Ok(allocation) => allocation,
            Err(error) => {
                unsafe {
                    free_mapped_bytes(positive_host_ptr)?;
                    free_mapped_bytes(negative_host_ptr)?;
                }
                return Err(error);
            }
        };

    let function = match dev.get_func("kernel_obstacle_stress", TEST_NAME) {
        Some(function) => function,
        None => {
            unsafe {
                free_mapped_bytes(positive_host_ptr)?;
                free_mapped_bytes(negative_host_ptr)?;
                free_mapped_u64_array(results_host_ptr)?;
            }
            return Err(GpuHostError::KernelNotFound(TEST_NAME));
        }
    };

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (status_tx, status_rx) = std::sync::mpsc::channel();
    let launch_device = Arc::clone(&dev);
    let launch_thread = std::thread::spawn(move || {
        let status = unsafe {
            function.launch(
                config,
                (positive_dev_ptr, negative_dev_ptr, results_dev_ptr),
            )
        }
        .and_then(|_| launch_device.synchronize())
        .map_err(|error| format!("{error}"));
        let _ = status_tx.send(status);
    });

    let start = Instant::now();
    let mut last_phase = 0u64;
    let launch_status = loop {
        match status_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(status) => break status,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                break Err("GPU launch thread disconnected".to_owned());
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                let phase = unsafe { std::ptr::read_volatile(results_host_ptr.add(field::PHASE)) };
                if phase != last_phase {
                    println!("  phase={phase}");
                    last_phase = phase;
                }
                if start.elapsed() >= KERNEL_TIMEOUT {
                    // Kernel 仍可能访问 mapped memory；超时路径故意不 free，避免 UAF。
                    return Err(GpuHostError::Timeout {
                        test: TEST_NAME,
                        detail: format!(
                            "20s 内未完成，最后 phase={phase}；结果判定为 UNKNOWN，mapped buffers 保留至进程退出"
                        ),
                    });
                }
            }
        }
    };
    let _ = launch_thread.join();

    if let Err(detail) = launch_status {
        unsafe {
            free_mapped_bytes(positive_host_ptr)?;
            free_mapped_bytes(negative_host_ptr)?;
            free_mapped_u64_array(results_host_ptr)?;
        }
        return Err(GpuHostError::Verification {
            test: TEST_NAME,
            detail: format!("CUDA launch/synchronize failed: {detail}; result=UNKNOWN"),
        });
    }

    let results = unsafe { StressResults::read_from(results_host_ptr) };
    unsafe {
        free_mapped_bytes(positive_host_ptr)?;
        free_mapped_bytes(negative_host_ptr)?;
        free_mapped_u64_array(results_host_ptr)?;
    }

    println!(
        "  admission: Low total={}/{} (work={}), ReservedCapacity rejects={}, other rejects={}, High accepted={}",
        results.get(field::LOW_ADMITTED_TOTAL),
        results.get(field::EXPECTED_LOW_LIMIT),
        results.get(field::LOW_WORK_ADMITTED),
        results.get(field::LOW_RESERVED_REJECTIONS),
        results.get(field::LOW_OTHER_REJECTIONS),
        results.get(field::HIGH_SPAWN_OK),
    );
    println!(
        "  positive dispatch: inject={} first={} gap={} resume={} wake-gap={} timestamps={}ns→{}ns latency={}ns",
        results.get(field::POS_INJECT_SEQ),
        results.get(field::POS_HIGH_FIRST_SEQ),
        results.get(field::POS_HIGH_FIRST_GAP),
        results.get(field::POS_HIGH_RESUME_SEQ),
        results.get(field::POS_HIGH_RESUME_GAP),
        results.get(field::POS_INJECT_NS),
        results.get(field::POS_HIGH_FIRST_NS),
        results.get(field::POS_LATENCY_NS),
    );
    println!(
        "  Low completion: {}/{}; executor completed={}/{}; total dispatches={}",
        results.get(field::LOW_COMPLETED),
        results.get(field::LOW_WORK_ADMITTED),
        results.get(field::POS_COMPLETED),
        results.get(field::POS_SPAWNED),
        results.get(field::POS_DISPATCH_SEQUENCE),
    );
    println!(
        "  bounded non-yielding negative: High accepted={} inject={} low-return={} High-first={} timestamps={}ns→{}ns→{}ns latency={}ns busy-iters={} dispatches={}",
        results.get(field::NEG_HIGH_SPAWN_OK),
        results.get(field::NEG_INJECT_SEQ),
        results.get(field::NEG_LOW_RETURN_SEQ),
        results.get(field::NEG_HIGH_FIRST_SEQ),
        results.get(field::NEG_INJECT_NS),
        results.get(field::NEG_LOW_RETURN_NS),
        results.get(field::NEG_HIGH_FIRST_NS),
        results.get(field::NEG_LATENCY_NS),
        results.get(field::NEG_BUSY_ITERS),
        results.get(field::NEG_DISPATCH_SEQUENCE),
    );
    println!(
        "  decisions: fresh(action={},reason={},use_gpu={}); stale(action={},reason={},use_gpu={}); deadline(action={},reason={},use_gpu={})",
        results.get(field::FRESH_ACTION),
        results.get(field::FRESH_REASON),
        results.get(field::FRESH_USED_GPU_RESULT),
        results.get(field::STALE_ACTION),
        results.get(field::STALE_REASON),
        results.get(field::STALE_USED_GPU_RESULT),
        results.get(field::DEADLINE_ACTION),
        results.get(field::DEADLINE_REASON),
        results.get(field::DEADLINE_USED_GPU_RESULT),
    );
    println!("  gates: {}", results.gate_summary());

    let schema_ok = results.get(field::VERSION) == 1
        && results.get(field::WORD_COUNT) == RESULT_WORDS as u64
        && results.get(field::PHASE) == 6;
    let decisions_are_distinct = results.get(field::FRESH_ACTION)
        == ACTION_APPLY_BRAKE_FROM_FRESH_RESULT
        && results.get(field::FRESH_REASON) == REASON_FRESH_HAZARD
        && results.get(field::FRESH_USED_GPU_RESULT) == 1
        && results.get(field::STALE_ACTION) == ACTION_WATCHDOG_CONSERVATIVE_STOP
        && results.get(field::STALE_REASON) == REASON_STALE_RESULT
        && results.get(field::STALE_USED_GPU_RESULT) == 0
        && results.get(field::DEADLINE_ACTION) == ACTION_WATCHDOG_CONSERVATIVE_STOP
        && results.get(field::DEADLINE_REASON) == REASON_DEADLINE_MISS
        && results.get(field::DEADLINE_USED_GPU_RESULT) == 0;
    if !schema_ok || !decisions_are_distinct || results.get(field::OVERALL_PASS) != 1 {
        return Err(GpuHostError::Verification {
            test: TEST_NAME,
            detail: format!(
                "semantic gate failed: schema_ok={schema_ok} decisions_are_distinct={decisions_are_distinct}; {}",
                results.gate_summary()
            ),
        });
    }

    println!("  PASS：协作式 priority 语义成立；负例同时证明 poll 内不可抢占。");
    Ok(())
}

const COMPOSED_TEST_NAME: &str = "composed_priority_hostcall_e2e";
const TIMEOUT_REUSE_TEST_NAME: &str = "priority_echo_timeout_reuse";
const HOST_TIMEOUT_ERROR_CATEGORY: u16 = 16;
const COMPOSED_NAMESPACE: u64 = 0x0B57_A11E;
const TIMEOUT_FIRST_LOCAL_ID: u64 = 250;
const TIMEOUT_SECOND_LOCAL_ID: u64 = 251;
const EXPECTED_REUSE_SCHEMA_VERSION: u64 = 1;
const EXPECTED_PRIORITY_HIGH: u64 = 2;
const COMPLETION_BYTES: usize = 64;
const COMPOSED_TIMEOUT: Duration = Duration::from_secs(30);

fn composed_mutation_name(mutation: u32) -> &'static str {
    match mutation {
        0 => "positive typed High",
        1 => "isolated raw substitution (High admission, then Normal FIFO requeue)",
        2 => "wire priority downgrade",
        3 => "wire identity corruption",
        4 => "host forced error",
        5 => "join replay evidence",
        _ => "unknown mutation",
    }
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

fn composed_timeout_unknown(test: &'static str, detail: String) -> GpuHostError {
    GpuHostError::Timeout { test, detail }
}

fn run_composed_mode(dev: &Arc<CudaDevice>, mutation: u32) -> Result<(OracleVerdict, bool)> {
    let mutation_name = composed_mutation_name(mutation);
    let function = dev
        .get_func("kernel_composed_priority", COMPOSED_TEST_NAME)
        .ok_or(GpuHostError::KernelNotFound(COMPOSED_TEST_NAME))?;
    let buffer = Arc::new(HostcallBuffer::new_with_priority_reserve(5, 1)?);
    assert_eq!(buffer.num_packets(), 5);
    assert_eq!(buffer.high_reserved_packets(), 1);
    if mutation == 4 {
        buffer.set_priority_echo_test_hook(PriorityEchoTestHook::ForceError, 0);
    }
    let executor = unsafe { MappedBytes::allocate(dev, EXECUTOR_BYTES)? };
    let low_completion = unsafe { MappedBytes::allocate(dev, COMPLETION_BYTES)? };
    let high_completion = unsafe { MappedBytes::allocate(dev, COMPLETION_BYTES)? };
    let results =
        unsafe { MappedWords::allocate(dev, gpu_host::hostcall::composed_schema::WORD_COUNT)? };
    let nonce = 0xC05E_0000_0000_0000u64 | (mutation as u64 + 1);

    // Module load/symbol resolution happened before any listener/resource
    // thread starts. The host path synchronizes the GPU before shutdown.
    let listener = HostcallListener::start(Arc::clone(&buffer), |_| {});
    let (status_tx, status_rx) = std::sync::mpsc::channel();
    let launch_device = Arc::clone(dev);
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let executor_dev = executor.dev;
    let low_completion_dev = low_completion.dev;
    let high_completion_dev = high_completion.dev;
    let results_dev = results.dev;
    let hostcall_dev = buffer.dev_ptr();
    let launch_thread = std::thread::spawn(move || {
        let status = unsafe {
            function.launch(
                config,
                (
                    executor_dev,
                    hostcall_dev,
                    low_completion_dev,
                    high_completion_dev,
                    results_dev,
                    nonce,
                    mutation,
                ),
            )
        }
        .and_then(|_| launch_device.synchronize())
        .map_err(|error| error.to_string());
        if status_tx.send(status).is_err() {
            eprintln!("composed launch status receiver closed before delivery");
        }
    });

    let status = match status_rx.recv_timeout(COMPOSED_TIMEOUT) {
        Ok(status) => status,
        Err(RecvTimeoutError::Timeout) => {
            // GPU may still own every mapped pointer. Leak the guarded objects
            // until process exit instead of manufacturing a timeout UAF.
            std::mem::forget(listener);
            std::mem::forget(buffer);
            executor.leak();
            low_completion.leak();
            high_completion.leak();
            results.leak();
            return Err(composed_timeout_unknown(
                COMPOSED_TEST_NAME,
                format!("mutation={mutation} exceeded 30s; result=UNKNOWN, mappings intentionally retained"),
            ));
        }
        Err(RecvTimeoutError::Disconnected) => {
            let launch_panicked = launch_thread.join().is_err();
            std::mem::forget(listener);
            std::mem::forget(buffer);
            executor.leak();
            low_completion.leak();
            high_completion.leak();
            results.leak();
            return Err(GpuHostError::Verification {
                test: COMPOSED_TEST_NAME,
                detail: format!(
                    "mutation={mutation} launch status channel disconnected (thread_panicked={launch_panicked}); result=UNKNOWN, mappings intentionally retained"
                ),
            });
        }
    };
    launch_thread
        .join()
        .map_err(|_| GpuHostError::Verification {
            test: COMPOSED_TEST_NAME,
            detail: format!(
                "mutation={mutation} launch thread panicked after status delivery; result=UNKNOWN"
            ),
        })?;
    status.map_err(|detail| GpuHostError::Verification {
        test: COMPOSED_TEST_NAME,
        detail: format!("mutation={mutation} launch/synchronize failed: {detail}; result=UNKNOWN"),
    })?;
    listener.finish()?;

    let mut raw = unsafe { ComposedResults::read_from(results.host) };
    let events = buffer.priority_echo_events();
    let audit = buffer.quiescent_pool_audit();
    raw.apply_host_audit(events.len(), &audit);
    let expected = ExpectedRun { mutation, nonce };
    let verdict = evaluate_composed(expected, &raw, &events, &audit);
    let designated_gate_red =
        mutation != 0 && required_mutation_gate_is_red(expected, &raw, &events, &audit);
    println!(
        "  mutation={mutation} ({mutation_name}): dispatch={} inject={} first={} pending={} events={} audit={:?} verdict={verdict:?}",
        raw.get(gpu_host::hostcall::composed_schema::DISPATCH_SEQUENCE),
        raw.get(gpu_host::hostcall::composed_schema::HIGH_INJECT_SEQUENCE),
        raw.get(gpu_host::hostcall::composed_schema::HIGH_FIRST_POLL_SEQUENCE),
        raw.get(gpu_host::hostcall::composed_schema::HIGH_PENDING_POLLS),
        events.len(),
        audit,
    );
    println!("    raw={:?}", raw.words);
    println!("    host_events={events:?}");
    Ok((verdict, designated_gate_red))
}

fn run_timeout_reclaim_reuse(dev: &Arc<CudaDevice>) -> Result<()> {
    let function = dev
        .get_func("kernel_composed_priority", TIMEOUT_REUSE_TEST_NAME)
        .ok_or(GpuHostError::KernelNotFound(TIMEOUT_REUSE_TEST_NAME))?;
    let buffer = Arc::new(HostcallBuffer::new_with_priority_reserve(2, 1)?);
    buffer.set_priority_echo_test_hook(PriorityEchoTestHook::DelayNext, 50_000);
    let results = unsafe { MappedWords::allocate(dev, reuse_schema::WORD_COUNT)? };
    let first_nonce = 0xCA11_CE11u64;
    let second_nonce = 0xCA11_CE22u64;
    let listener = HostcallListener::start(Arc::clone(&buffer), |_| {});
    let (status_tx, status_rx) = std::sync::mpsc::channel();
    let launch_device = Arc::clone(dev);
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let results_dev = results.dev;
    let hostcall_dev = buffer.dev_ptr();
    let launch_thread = std::thread::spawn(move || {
        let status = unsafe {
            function.launch(
                config,
                (hostcall_dev, results_dev, first_nonce, second_nonce),
            )
        }
        .and_then(|_| launch_device.synchronize())
        .map_err(|error| error.to_string());
        if status_tx.send(status).is_err() {
            eprintln!("timeout/reuse launch status receiver closed before delivery");
        }
    });
    let status = match status_rx.recv_timeout(COMPOSED_TIMEOUT) {
        Ok(status) => status,
        Err(RecvTimeoutError::Timeout) => {
            std::mem::forget(listener);
            std::mem::forget(buffer);
            results.leak();
            return Err(composed_timeout_unknown(
                TIMEOUT_REUSE_TEST_NAME,
                "timeout/reclaim/reuse exceeded 30s; result=UNKNOWN, mapping retained".to_owned(),
            ));
        }
        Err(RecvTimeoutError::Disconnected) => {
            let launch_panicked = launch_thread.join().is_err();
            std::mem::forget(listener);
            std::mem::forget(buffer);
            results.leak();
            return Err(GpuHostError::Verification {
                test: TIMEOUT_REUSE_TEST_NAME,
                detail: format!(
                    "launch status channel disconnected (thread_panicked={launch_panicked}); result=UNKNOWN, mapping intentionally retained"
                ),
            });
        }
    };
    launch_thread
        .join()
        .map_err(|_| GpuHostError::Verification {
            test: TIMEOUT_REUSE_TEST_NAME,
            detail: "launch thread panicked after status delivery; result=UNKNOWN".to_owned(),
        })?;
    status.map_err(|detail| GpuHostError::Verification {
        test: TIMEOUT_REUSE_TEST_NAME,
        detail: format!("launch/synchronize failed: {detail}; result=UNKNOWN"),
    })?;
    listener.finish()?;

    let mut words = [0u64; reuse_schema::WORD_COUNT];
    for (index, word) in words.iter_mut().enumerate() {
        *word = unsafe { std::ptr::read_volatile(results.host.add(index)) };
    }
    let events = buffer.priority_echo_events();
    let audit = buffer.quiescent_pool_audit();
    let metrics = buffer.processing_metrics();
    let generations_are_exact_successors = matches!(events.as_slice(), [first, second]
        if first.generation == 1
            && second.generation == 2
            && second.generation == first.generation + 1);
    let event_chain_valid = matches!(events.as_slice(), [first, second]
        if first.task_id == ((COMPOSED_NAMESPACE << 32) | TIMEOUT_FIRST_LOCAL_ID)
            && second.task_id == ((COMPOSED_NAMESPACE << 32) | TIMEOUT_SECOND_LOCAL_ID)
            && first.namespace as u64 == COMPOSED_NAMESPACE
            && second.namespace as u64 == COMPOSED_NAMESPACE
            && first.local_id as u64 == TIMEOUT_FIRST_LOCAL_ID
            && second.local_id as u64 == TIMEOUT_SECOND_LOCAL_ID
            && first.priority.as_raw() as u64 == EXPECTED_PRIORITY_HIGH
            && second.priority.as_raw() as u64 == EXPECTED_PRIORITY_HIGH
            && first.shared_high_reserved
            && second.shared_high_reserved
            && first.echo_nonce == first_nonce
            && second.echo_nonce == second_nonce
            && matches!(first.error_category, 0 | HOST_TIMEOUT_ERROR_CATEGORY)
            && second.error_category == 0
            && first.process_sequence == 1
            && second.process_sequence == 2
            && first.process_count == 1
            && second.process_count == 1);
    let pass = words[reuse_schema::VERSION] == EXPECTED_REUSE_SCHEMA_VERSION
        && words[reuse_schema::FIRST_NONCE] == first_nonce
        && words[reuse_schema::SECOND_NONCE] == second_nonce
        && words[reuse_schema::FIRST_PACKET_INDEX] == 1
        && words[reuse_schema::FIRST_ERROR_CATEGORY] == HOST_TIMEOUT_ERROR_CATEGORY as u64
        && words[reuse_schema::REACQUIRE_BUSY_ATTEMPTS] >= 1
        && words[reuse_schema::SECOND_PACKET_INDEX] == 1
        && words[reuse_schema::SECOND_ECHO_NONCE] == second_nonce
        && words[reuse_schema::SECOND_RESPONSE_PACKET_INDEX] == 1
        && words[reuse_schema::SECOND_PENDING_POLLS] >= 1
        && words[reuse_schema::REUSE_STALE_WRITE] == 0
        && words[reuse_schema::SECOND_COMPLETED] == 1
        && words[reuse_schema::GENERAL_GUARD_PACKET_INDEX] == 0
        && words[reuse_schema::RESERVED_0] == 0
        && words[reuse_schema::RESERVED_1] == 0
        && words[reuse_schema::KERNEL_ERROR] == 0
        && events.len() == 2
        && events[0].request_nonce == first_nonce
        && events[1].request_nonce == second_nonce
        && events[0].packet_index as u64 == words[reuse_schema::FIRST_PACKET_INDEX]
        && events[1].packet_index as u64 == words[reuse_schema::SECOND_PACKET_INDEX]
        && events[1].packet_index as u64 == words[reuse_schema::SECOND_RESPONSE_PACKET_INDEX]
        && events[0].packet_index == events[1].packet_index
        && event_chain_valid
        && generations_are_exact_successors
        && metrics.cancelled == 1
        && metrics.completed == 1
        && metrics.errors == 0
        && metrics.stale_rejected == 0
        && audit.ready_empty
        && audit.idle_packets == 2
        && audit.general_mask == 0b01
        && audit.shared_high_mask == 0b10
        && audit.duplicate_entries == 0
        && audit.missing_packets == 0
        && audit.non_idle_controls == 0;
    println!(
        "  timeout/reclaim/reuse: raw={words:?} events={events:?} metrics={metrics:?} audit={audit:?}"
    );
    if !pass {
        return Err(GpuHostError::Verification {
            test: TIMEOUT_REUSE_TEST_NAME,
            detail: "timeout→host reclaim→same packet reuse/no-stale-write gate failed".to_owned(),
        });
    }
    Ok(())
}

/// 运行正常组合场景、五个 fault mutations 与 timeout/reclaim/reuse gate。
pub(crate) fn run_composed_priority_e2e(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- typed priority + v3 hostcall composed E2E ---");
    println!("  语义范围：协作式 poll-boundary 优先级；不声称硬实时响应界。 ");
    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Io,
        "kernel_composed_priority",
        &[COMPOSED_TEST_NAME, TIMEOUT_REUSE_TEST_NAME],
    )?;

    for mutation in 0..=5 {
        let (verdict, designated_gate_red) = run_composed_mode(&dev, mutation)?;
        let mutation_name = composed_mutation_name(mutation);
        match (mutation, verdict) {
            (0, OracleVerdict::Pass) => {}
            (0, OracleVerdict::Fail(failures)) => {
                return Err(GpuHostError::Verification {
                    test: COMPOSED_TEST_NAME,
                    detail: format!("positive semantics FAIL: {}", failures.join("; ")),
                });
            }
            (0, OracleVerdict::Unknown(detail)) => {
                return Err(GpuHostError::Verification {
                    test: COMPOSED_TEST_NAME,
                    detail: format!("positive result UNKNOWN: {detail}"),
                });
            }
            (_, OracleVerdict::Fail(_)) if designated_gate_red => {}
            (_, OracleVerdict::Fail(failures)) => {
                return Err(GpuHostError::Verification {
                    test: COMPOSED_TEST_NAME,
                    detail: format!(
                        "fault mutation {mutation} ({mutation_name}) failed only unrelated gates; designated gate stayed green: {}",
                        failures.join("; ")
                    ),
                });
            }
            (_, OracleVerdict::Pass) => {
                return Err(GpuHostError::Verification {
                    test: COMPOSED_TEST_NAME,
                    detail: format!(
                        "fault mutation {mutation} ({mutation_name}) survived independent oracle"
                    ),
                });
            }
            (_, OracleVerdict::Unknown(detail)) => {
                return Err(GpuHostError::Verification {
                    test: COMPOSED_TEST_NAME,
                    detail: format!(
                        "fault mutation {mutation} ({mutation_name}) produced UNKNOWN, not a killed gate: {detail}"
                    ),
                });
            }
        }
    }
    run_timeout_reclaim_reuse(&dev)?;
    println!("  PASS: positive + five killed mutations + cancel/reclaim/reuse。 ");
    Ok(())
}
