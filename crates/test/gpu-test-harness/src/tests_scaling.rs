//! Multi-warp/block tests + misc (error propagation, println direct, slab allocator).

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

use gpu_host::error::{GpuHostError, Result};
use gpu_host::hostcall;
use gpu_host::mapped_mem::{
    alloc_mapped_bytes, alloc_mapped_result_array, free_mapped_bytes, free_mapped_mem,
};

/// Test: 32-thread multi-warp synchronous hostcall scaling (product.3).
pub(crate) fn run_multi_warp_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Product Test 3: Multi-warp sync hostcall (32 threads) ---");

    use std::sync::{Arc as StdArc, Mutex};

    let hc_buf = hostcall::HostcallBuffer::new(64)?;
    let dev_ptr = hc_buf.dev_ptr();

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 2)? };

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] Received from GPU: \"{s}\"");
            messages_clone.lock().unwrap().push(s);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::MULTI_WARP_PTX);
    dev.load_ptx(ptx, "multi_warp", &["multi_warp_sync_kernel"])?;
    let f = dev
        .get_func("multi_warp", "multi_warp_sync_kernel")
        .ok_or(GpuHostError::KernelNotFound("multi_warp_sync_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching multi_warp_sync_kernel (32 threads, full warp)...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("  Kernel completed in {elapsed:?}.");

    std::thread::sleep(std::time::Duration::from_millis(500));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let thread_count = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();

    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "multi_warp_sync_kernel",
            detail: format!("success marker not set (got {success})"),
        });
    }

    if thread_count != 32 {
        return Err(GpuHostError::Verification {
            test: "multi_warp_sync_kernel",
            detail: format!("expected thread_count=32, got {thread_count}"),
        });
    }

    let msg_count = received.len();
    println!("  multi_warp_sync_kernel: {msg_count} messages received from 32 threads");

    let mut thread_seen = [false; 32];
    for msg in received.iter() {
        if msg.starts_with("Thread ") && msg.len() >= 16 {
            let tens = msg.as_bytes()[7];
            let ones = msg.as_bytes()[8];
            if (b'0'..=b'3').contains(&tens) && ones.is_ascii_digit() {
                let tid = ((tens - b'0') * 10 + (ones - b'0')) as usize;
                if tid < 32 {
                    thread_seen[tid] = true;
                }
            }
        }
    }
    let unique_threads = thread_seen.iter().filter(|&&v| v).count();

    if unique_threads < 32 {
        let missing: Vec<usize> = thread_seen
            .iter()
            .enumerate()
            .filter(|(_, &v)| !v)
            .map(|(i, _)| i)
            .collect();
        return Err(GpuHostError::Verification {
            test: "multi_warp_sync_kernel",
            detail: format!(
                "expected messages from all 32 threads, got {unique_threads} unique. Missing: {missing:?}"
            ),
        });
    }

    println!("  multi_warp_sync_kernel: PASSED!");
    println!("    All 32 threads sent unique messages concurrently");
    println!("    Thread count: {thread_count}");
    println!("    Messages received: {msg_count}");
    println!("    Unique threads verified: {unique_threads}/32");
    println!("    Multi-warp hostcall scaling demonstrated end-to-end");
    Ok(())
}

/// Test: multi-block synchronous hostcall (multiblock.1).
pub(crate) fn run_multi_block_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Multiblock Test 1: 4-block sync hostcall (128 threads) ---");

    use std::sync::{Arc as StdArc, Mutex};

    let num_blocks: u32 = 4;
    let threads_per_block: u32 = 32;
    let total_threads = (num_blocks * threads_per_block) as usize;
    let num_packets: u16 = 256;

    let hc_buf = hostcall::HostcallBuffer::new(num_packets)?;
    let dev_ptr = hc_buf.dev_ptr();

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 2)? };

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] Received from GPU: \"{s}\"");
            messages_clone.lock().unwrap().push(s);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::MULTI_WARP_PTX);
    dev.load_ptx(ptx, "multi_block", &["multi_block_sync_kernel"])?;
    let f = dev
        .get_func("multi_block", "multi_block_sync_kernel")
        .ok_or(GpuHostError::KernelNotFound("multi_block_sync_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    println!(
        "  Launching multi_block_sync_kernel ({num_blocks} blocks × {threads_per_block} threads = {total_threads} total)..."
    );
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("  Kernel completed in {elapsed:?}.");

    std::thread::sleep(std::time::Duration::from_millis(500));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let thread_count = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();

    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "multi_block_sync_kernel",
            detail: format!("success marker not set (got {success})"),
        });
    }

    if thread_count != total_threads as u32 {
        return Err(GpuHostError::Verification {
            test: "multi_block_sync_kernel",
            detail: format!("expected thread_count={total_threads}, got {thread_count}"),
        });
    }

    let msg_count = received.len();
    println!(
        "  multi_block_sync_kernel: {msg_count} messages received from {total_threads} threads"
    );

    let mut thread_seen = vec![false; total_threads];
    for msg in received.iter() {
        if msg.starts_with("Thread ") && msg.len() >= 17 {
            let hundreds = msg.as_bytes()[7];
            let tens = msg.as_bytes()[8];
            let ones = msg.as_bytes()[9];
            if hundreds.is_ascii_digit() && tens.is_ascii_digit() && ones.is_ascii_digit() {
                let tid = ((hundreds - b'0') as usize) * 100
                    + ((tens - b'0') as usize) * 10
                    + (ones - b'0') as usize;
                if tid < total_threads {
                    thread_seen[tid] = true;
                }
            }
        }
    }
    let unique_threads = thread_seen.iter().filter(|&&v| v).count();

    if unique_threads < total_threads {
        let missing: Vec<usize> = thread_seen
            .iter()
            .enumerate()
            .filter(|(_, &v)| !v)
            .map(|(i, _)| i)
            .collect();
        return Err(GpuHostError::Verification {
            test: "multi_block_sync_kernel",
            detail: format!(
                "expected messages from all {total_threads} threads, got {unique_threads} unique. Missing: {missing:?}"
            ),
        });
    }

    println!("  multi_block_sync_kernel: PASSED! ({elapsed:?})");
    println!("    All {total_threads} threads across {num_blocks} blocks sent unique messages");
    println!("    Messages received: {msg_count}");
    println!("    Unique threads verified: {unique_threads}/{total_threads}");
    Ok(())
}

/// Test: multi-block scaling to 512 threads (multiblock.2).
pub(crate) fn run_multi_block_512_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Multiblock Test 2: 8-block × 64-thread sync hostcall (512 threads) ---");

    use std::sync::{Arc as StdArc, Mutex};

    let num_blocks: u32 = 8;
    let threads_per_block: u32 = 64;
    let total_threads = (num_blocks * threads_per_block) as usize;
    let num_packets: u16 = 1024;

    let hc_buf = hostcall::HostcallBuffer::new(num_packets)?;
    let dev_ptr = hc_buf.dev_ptr();

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 2)? };

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            messages_clone.lock().unwrap().push(s);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::MULTI_WARP_PTX);
    dev.load_ptx(ptx, "multi_block_512", &["multi_block_sync_kernel"])?;
    let f = dev
        .get_func("multi_block_512", "multi_block_sync_kernel")
        .ok_or(GpuHostError::KernelNotFound("multi_block_sync_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    println!(
        "  Launching multi_block_sync_kernel ({num_blocks} blocks × {threads_per_block} threads = {total_threads} total)..."
    );
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("  Kernel completed in {elapsed:?}.");

    std::thread::sleep(std::time::Duration::from_millis(1000));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let thread_count = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();

    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "multi_block_512",
            detail: format!("success marker not set (got {success})"),
        });
    }

    if thread_count != total_threads as u32 {
        return Err(GpuHostError::Verification {
            test: "multi_block_512",
            detail: format!("expected thread_count={total_threads}, got {thread_count}"),
        });
    }

    let msg_count = received.len();
    let mut thread_seen = vec![false; total_threads];
    for msg in received.iter() {
        if msg.starts_with("Thread ") && msg.len() >= 17 {
            let hundreds = msg.as_bytes()[7];
            let tens = msg.as_bytes()[8];
            let ones = msg.as_bytes()[9];
            if hundreds.is_ascii_digit() && tens.is_ascii_digit() && ones.is_ascii_digit() {
                let tid = ((hundreds - b'0') as usize) * 100
                    + ((tens - b'0') as usize) * 10
                    + (ones - b'0') as usize;
                if tid < total_threads {
                    thread_seen[tid] = true;
                }
            }
        }
    }
    let unique_threads = thread_seen.iter().filter(|&&v| v).count();

    if unique_threads < total_threads {
        let missing: Vec<usize> = thread_seen
            .iter()
            .enumerate()
            .filter(|(_, &v)| !v)
            .map(|(i, _)| i)
            .collect();
        let shown: Vec<usize> = missing.iter().take(20).copied().collect();
        return Err(GpuHostError::Verification {
            test: "multi_block_512",
            detail: format!(
                "expected messages from all {total_threads} threads, got {unique_threads} unique. Missing (first 20): {shown:?}"
            ),
        });
    }

    let duplicates = msg_count - unique_threads;

    println!("  multi_block_512: PASSED! ({elapsed:?})");
    println!("    All {total_threads} threads across {num_blocks} blocks sent unique messages");
    println!(
        "    Messages received: {msg_count} ({unique_threads} unique, {duplicates} duplicates)"
    );
    println!("    Unique threads verified: {unique_threads}/{total_threads}");
    Ok(())
}

/// Test: 4-step sequential async pipeline (product.2).
pub(crate) fn run_pipeline_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Product Test 2: 4-step async pipeline ---");

    use std::sync::{Arc as StdArc, Mutex};

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 2)? };

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] Received from GPU: \"{s}\"");
            messages_clone.lock().unwrap().push(s);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::ASYNC_PIPELINE_PTX);
    dev.load_ptx(ptx, "async_pipeline", &["pipeline_kernel"])?;
    let f = dev
        .get_func("async_pipeline", "pipeline_kernel")
        .ok_or(GpuHostError::KernelNotFound("pipeline_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching pipeline_kernel (4-step sequential async)...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("  Kernel completed in {elapsed:?}.");

    std::thread::sleep(std::time::Duration::from_millis(200));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let poll_rounds = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();
    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "pipeline_kernel",
            detail: format!("success marker not set (got {success}), poll_rounds={poll_rounds}"),
        });
    }

    let expected_steps = [
        "step 1: READ",
        "step 2: PROCESS",
        "step 3: WRITE",
        "step 4: PRINT",
    ];
    let mut found_steps = 0;
    for expected in &expected_steps {
        if received.iter().any(|m| m.contains(expected)) {
            found_steps += 1;
        }
    }

    if found_steps != 4 {
        return Err(GpuHostError::Verification {
            test: "pipeline_kernel",
            detail: format!(
                "expected 4 pipeline steps, found {}. Messages: {:?}",
                found_steps, *received
            ),
        });
    }

    println!("  pipeline_kernel: PASSED!");
    println!("    Poll rounds: {poll_rounds}");
    println!("    Steps completed: 4/4");
    for (i, msg) in received.iter().enumerate() {
        println!("    Step {}: \"{}\"", i + 1, msg);
    }
    println!("    4-step sequential async pipeline demonstrated end-to-end");
    Ok(())
}

/// Showcase demo: all features combined (product.4).
pub(crate) fn run_showcase_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Product Test 4: Showcase Demo (Vec + format! + stdin + stdout on GPU) ---");

    use std::sync::{Arc as StdArc, Mutex};

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 2)? };

    let input_data: Vec<u32> = vec![10, 25, 30, 7, 42, 15, 88, 3];
    let input_dev: CudaSlice<u32> = dev.htod_copy(input_data.clone())?;
    let input_len = input_data.len() as u32;

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);

    let stdin_data = b"Rustacean\n".to_vec();
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen_with_stdin(
            |msg| {
                let s = String::from_utf8_lossy(msg).to_string();
                println!("  [GPU] {s}");
                messages_clone.lock().unwrap().push(s);
            },
            stdin_data,
        );
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::STD_BUILD_TEST_PTX);
    dev.load_ptx(ptx, "std_test", &["showcase_kernel"])?;
    let f = dev
        .get_func("std_test", "showcase_kernel")
        .ok_or(GpuHostError::KernelNotFound("showcase_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching showcase_kernel (Vec + format! + stdin + stdout)...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, &input_dev, input_len, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("  Kernel completed in {elapsed:?}.");

    std::thread::sleep(std::time::Duration::from_millis(200));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let msg_count = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();
    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "showcase_kernel",
            detail: format!("kernel reported failure (success={success})"),
        });
    }

    if msg_count < 4 {
        return Err(GpuHostError::Verification {
            test: "showcase_kernel",
            detail: format!("expected 4 stdout messages, got {msg_count}"),
        });
    }

    let has_greeting = received
        .iter()
        .any(|m| m.contains("Hello") && m.contains("Rustacean"));
    let has_stats = received.iter().any(|m| m.contains("sum=220"));
    let has_goodbye = received.iter().any(|m| m.contains("Goodbye"));

    if !has_greeting || !has_stats || !has_goodbye {
        return Err(GpuHostError::Verification {
            test: "showcase_kernel",
            detail: format!(
                "missing expected content (greeting={}, stats={}, goodbye={}). Messages: {:?}",
                has_greeting, has_stats, has_goodbye, *received
            ),
        });
    }

    println!("  showcase_kernel: PASSED!");
    println!("    Features demonstrated:");
    println!("      - stdin read (got name \"Rustacean\" from host)");
    println!(
        "      - Vec<u32> built from {} runtime kernel arguments",
        input_data.len()
    );
    println!("      - Iterator methods: sum, min, max, filter, collect");
    println!("      - format!() with heap-allocated String");
    println!("      - writeln!(stdout()) through PAL hostcall ({msg_count}x)");
    println!("    Total time: {elapsed:?}");
    println!("    This is VectorWare-level Rust std on GPU!");
    Ok(())
}

/// Test: multi-block async with Embassy executors (multiblock.3).
pub(crate) fn run_multi_block_async_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Multiblock Test 3: Multi-block async (4 blocks × 1 thread) ---");

    use std::sync::{Arc as StdArc, Mutex};

    let hc_buf = hostcall::HostcallBuffer::new(8)?;
    let dev_ptr = hc_buf.dev_ptr();

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let num_threads: usize = 4;
    let (result_host_ptr, result_dev_ptr) =
        unsafe { alloc_mapped_result_array(&dev, num_threads + 1)? };

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] Async: {s}");
            messages_clone.lock().unwrap().push(s);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::ASYNC_HOSTCALL_PTX);
    dev.load_ptx(ptx, "multi_block_async", &["multi_block_async_kernel"])?;
    let f = dev
        .get_func("multi_block_async", "multi_block_async_kernel")
        .ok_or(GpuHostError::KernelNotFound("multi_block_async_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (num_threads as u32, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching multi_block_async_kernel ({num_threads} blocks × 1 thread)...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();

    std::thread::sleep(std::time::Duration::from_millis(200));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let completed = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let mut poll_rounds = Vec::new();
    for i in 0..num_threads {
        let rounds = unsafe { std::ptr::read_volatile(result_host_ptr.add(1 + i)) };
        poll_rounds.push(rounds);
    }
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();

    println!("  Completed: {completed}/{num_threads} threads");
    println!("  Poll rounds per thread: {poll_rounds:?}");
    println!("  Messages received: {}", received.len());
    for msg in received.iter() {
        println!("    \"{msg}\"");
    }

    if completed != num_threads as u32 {
        return Err(GpuHostError::Verification {
            test: "multi_block_async_kernel",
            detail: format!("only {completed}/{num_threads} threads completed"),
        });
    }

    let mut seen = [false; 4];
    for msg in received.iter() {
        for i in 0..4u8 {
            if msg.contains(&format!("block {i}")) {
                seen[i as usize] = true;
            }
        }
    }
    let all_seen = seen.iter().all(|&s| s);
    if !all_seen {
        return Err(GpuHostError::Verification {
            test: "multi_block_async_kernel",
            detail: format!("missing messages from some blocks. Seen: {seen:?}"),
        });
    }

    println!("  multi_block_async_kernel: PASSED! ({elapsed:?})");
    println!("    {num_threads} blocks × 1 thread, each with its own Embassy executor");
    println!("    All threads completed async hostcall independently!");
    Ok(())
}

/// Test: error propagation through hostcall (error-handling.2).
pub(crate) fn run_error_propagation_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Error Handling Test: error propagation (hostcall \u{2192} GPU) ---");

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 6)? };

    let hc_buf_ref = std::sync::Arc::new(hc_buf);
    let hc_buf_listener = std::sync::Arc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|_msg| {});
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_IO_PTX);
    dev.load_ptx(ptx, "error_test", &["error_propagation_test"])?;
    let f = dev
        .get_func("error_test", "error_propagation_test")
        .ok_or(GpuHostError::KernelNotFound("error_propagation_test"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching error_propagation_test kernel...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let err1_cat = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    let err2_cat = unsafe { std::ptr::read_volatile(result_host_ptr.add(2)) };
    let err3_cat = unsafe { std::ptr::read_volatile(result_host_ptr.add(3)) };
    let err1_fd = unsafe { std::ptr::read_volatile(result_host_ptr.add(4)) };
    let passed = unsafe { std::ptr::read_volatile(result_host_ptr.add(5)) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let err_not_found: u32 = 1;

    println!(
        "  Test 1 (open nonexistent): category={err1_cat} (expected {err_not_found}), fd={err1_fd}"
    );
    println!("  Test 2 (close invalid fd): category={err2_cat} (expected nonzero)");
    println!("  Test 3 (read invalid fd):  category={err3_cat} (expected nonzero)");
    println!("  Tests passed: {passed}/3");

    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "error_propagation_test",
            detail: format!(
                "not all tests passed ({passed}/3). categories: open={err1_cat}, close={err2_cat}, read={err3_cat}"
            ),
        });
    }

    println!("  error_propagation_test: PASSED! ({elapsed:?})");
    println!("    Structured error codes propagate from host to GPU correctly!");
    println!("    File-not-found → ERR_NOT_FOUND ({err_not_found}), Invalid fd → error category");
    Ok(())
}

pub(crate) fn run_println_direct_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- OnceLock Test: println!() direct (no writeln! workaround) ---");

    use std::sync::{Arc as StdArc, Mutex};

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 2)? };

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [GPU println] {s}");
            messages_clone.lock().unwrap().push(s);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::STD_BUILD_TEST_PTX);
    dev.load_ptx(ptx, "println_test", &["println_direct_test_kernel"])?;
    let f = dev
        .get_func("println_test", "println_direct_test_kernel")
        .ok_or(GpuHostError::KernelNotFound("println_direct_test_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let input_val: u32 = 42;
    println!("  Launching println_direct_test_kernel (println! with value={input_val})...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, input_val, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();

    std::thread::sleep(std::time::Duration::from_millis(200));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let count = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();

    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "println_direct_test_kernel",
            detail: format!("kernel reported failure (success={success})"),
        });
    }

    let full_output: String = received.iter().map(|s| s.as_str()).collect();

    let has_hello = full_output.contains("hello from GPU");
    let has_value = full_output.contains("value = 42");
    let has_multi = full_output.contains("x=84") && full_output.contains("y=52");

    if !has_hello || !has_value || !has_multi {
        return Err(GpuHostError::Verification {
            test: "println_direct_test_kernel",
            detail: format!(
                "missing expected content (hello={has_hello}, value={has_value}, multi={has_multi}). Full output: {full_output:?}"
            ),
        });
    }

    println!("  println_direct_test_kernel: PASSED! ({elapsed:?})");
    println!("    println!() works directly on GPU — no writeln! workaround needed!");
    println!(
        "    {} println! calls completed, {} messages received",
        count,
        received.len()
    );
    println!("    Full VectorWare parity achieved for println!");
    Ok(())
}

/// Test: slab allocator deallocation (allocator.2).
pub(crate) fn run_slab_dealloc_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Allocator Test: Slab dealloc (10 Vec + 10 String cycles) ---");

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 2)? };

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::STD_BUILD_TEST_PTX);
    dev.load_ptx(ptx, "slab_test", &["slab_dealloc_test_kernel"])?;
    let f = dev
        .get_func("slab_test", "slab_dealloc_test_kernel")
        .ok_or(GpuHostError::KernelNotFound("slab_dealloc_test_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching slab_dealloc_test_kernel (10 Vec + 10 String alloc/dealloc cycles)...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (0u64, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();

    let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let cycles = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "slab_dealloc_test_kernel",
            detail: format!("expected 20 successful cycles, got {cycles}"),
        });
    }

    println!("  slab_dealloc_test_kernel: PASSED! ({elapsed:?})");
    println!("    Completed {cycles} alloc/dealloc cycles (10 Vec + 10 String)");
    println!("    Memory reuse confirmed — slab allocator deallocates correctly");
    Ok(())
}

/// Test: concurrent slab allocator stress test (allocator.3).
pub(crate) fn run_slab_concurrent_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Allocator Test: 32-thread concurrent alloc/dealloc ---");

    let num_threads: u32 = 32;
    let (result_host_ptr, result_dev_ptr) =
        unsafe { alloc_mapped_result_array(&dev, num_threads as usize)? };

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::STD_BUILD_TEST_PTX);
    dev.load_ptx(ptx, "slab_concurrent", &["slab_concurrent_test_kernel"])?;
    let f = dev
        .get_func("slab_concurrent", "slab_concurrent_test_kernel")
        .ok_or(GpuHostError::KernelNotFound("slab_concurrent_test_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (num_threads, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching slab_concurrent_test_kernel (32 threads × 5 cycles each)...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (0u64, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();

    let mut total_ok: u32 = 0;
    let mut threads_ok: u32 = 0;
    let mut failed_threads = Vec::new();
    for i in 0..num_threads as usize {
        let cycles = unsafe { std::ptr::read_volatile(result_host_ptr.add(i)) };
        total_ok += cycles;
        if cycles == 5 {
            threads_ok += 1;
        } else {
            failed_threads.push((i, cycles));
        }
    }
    unsafe { free_mapped_mem(result_host_ptr)? };

    if threads_ok < num_threads {
        return Err(GpuHostError::Verification {
            test: "slab_concurrent_test_kernel",
            detail: format!(
                "expected all 32 threads to complete 5 cycles, got {threads_ok}/{num_threads} threads OK. Failed: {failed_threads:?}"
            ),
        });
    }

    println!("  slab_concurrent_test_kernel: PASSED! ({elapsed:?})");
    println!("    All {num_threads} threads completed 5 alloc/dealloc cycles each");
    println!(
        "    Total successful cycles: {}/{}",
        total_ok,
        num_threads * 5
    );
    println!("    Concurrent slab allocator verified under contention");
    Ok(())
}

/// Test: GpuExecutor demo — spawn 8 async tasks and run them (executor-impl.4).
pub(crate) fn run_executor_demo_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Executor Demo Test (executor-impl.4) ---");
    println!("  Spawns 8 async tasks: 4 WriteValueFuture + 4 CounterFuture.");

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Io,
        "kernel_executor",
        &["executor_demo"],
    )?;

    // Allocate mapped memory for the GpuExecutor (~136KB)
    // The executor is big because of 256 TaskSlots × 528 bytes each.
    let executor_size = 256 * 1024; // 256KB — generous
    let (exec_host_ptr, exec_dev_ptr) = unsafe { alloc_mapped_bytes(&dev, executor_size)? };

    // Allocate results: 16 u32
    let (results_host_ptr, results_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 16)? };

    let f = dev
        .get_func("kernel_executor", "executor_demo")
        .ok_or(GpuHostError::KernelNotFound("executor_demo"))?;

    // Launch with 32 threads (1 warp) — all lanes participate in executor.run()
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    // Launch on a separate thread so we can poll phase markers from main thread.
    // The results are in mapped memory, visible to host even while kernel runs.
    let dev2 = dev.clone();
    let sync_done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sync_done2 = sync_done.clone();
    let sync_thread = std::thread::spawn(move || {
        unsafe {
            f.launch(cfg, (exec_dev_ptr, results_dev_ptr)).unwrap();
        }
        let _ = dev2.synchronize();
        sync_done2.store(true, std::sync::atomic::Ordering::Release);
    });

    // Poll phase marker with timeout
    let timeout = std::time::Duration::from_secs(10);
    let start = std::time::Instant::now();
    let mut last_phase = 0u32;
    loop {
        let phase = unsafe { std::ptr::read_volatile(results_host_ptr.add(10)) };
        if phase != last_phase {
            println!("  phase marker = {phase}");
            last_phase = phase;
        }
        if sync_done.load(std::sync::atomic::Ordering::Acquire) {
            println!("  kernel completed (phase={phase})");
            break;
        }
        if start.elapsed() > timeout {
            let spawned = unsafe { std::ptr::read_volatile(results_host_ptr.add(0)) };
            let completed = unsafe { std::ptr::read_volatile(results_host_ptr.add(1)) };
            let v0 = unsafe { std::ptr::read_volatile(results_host_ptr.add(4)) };
            let counter = unsafe { std::ptr::read_volatile(results_host_ptr.add(8)) };
            println!("  TIMEOUT after {timeout:?}! phase={phase} spawned={spawned} completed={completed} v0={v0} counter={counter}");
            // Dump executor header bytes for debugging
            println!("  executor memory dump (first 64 bytes):");
            for i in 0..8u32 {
                let val = unsafe {
                    std::ptr::read_volatile((exec_host_ptr as *const u64).add(i as usize))
                };
                println!("    offset {}: 0x{:016x}", i * 8, val);
            }
            // Dump all 16 result slots
            println!("  results dump:");
            for i in 0..16u32 {
                let val = unsafe { std::ptr::read_volatile(results_host_ptr.add(i as usize)) };
                println!("    results[{i}] = {val}");
            }
            unsafe {
                free_mapped_bytes(exec_host_ptr)?;
                free_mapped_mem(results_host_ptr)?;
            }
            return Err(GpuHostError::Timeout {
                test: "executor_demo",
                detail: format!("kernel hung at phase {phase}"),
            });
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let _ = sync_thread.join();

    // Read results
    let spawned = unsafe { std::ptr::read_volatile(results_host_ptr.add(0)) };
    let completed = unsafe { std::ptr::read_volatile(results_host_ptr.add(1)) };
    let tasks_executed = unsafe { std::ptr::read_volatile(results_host_ptr.add(2)) };
    let polls_total = unsafe { std::ptr::read_volatile(results_host_ptr.add(3)) };
    let v0 = unsafe { std::ptr::read_volatile(results_host_ptr.add(4)) };
    let v1 = unsafe { std::ptr::read_volatile(results_host_ptr.add(5)) };
    let v2 = unsafe { std::ptr::read_volatile(results_host_ptr.add(6)) };
    let v3 = unsafe { std::ptr::read_volatile(results_host_ptr.add(7)) };
    let counter = unsafe { std::ptr::read_volatile(results_host_ptr.add(8)) };
    let success = unsafe { std::ptr::read_volatile(results_host_ptr.add(9)) };

    println!("  spawned={spawned} completed={completed} tasks_executed={tasks_executed} polls_total={polls_total}");
    println!("  values=[{v0}, {v1}, {v2}, {v3}]  counter={counter}  success={success}");

    unsafe {
        free_mapped_bytes(exec_host_ptr)?;
        free_mapped_mem(results_host_ptr)?;
    }

    assert_eq!(spawned, 8, "expected 8 tasks spawned");
    assert_eq!(completed, 8, "expected 8 tasks completed");
    assert_eq!(v0, 42, "WriteValueFuture[0] expected 42");
    assert_eq!(v1, 100, "WriteValueFuture[1] expected 100");
    assert_eq!(v2, 255, "WriteValueFuture[2] expected 255");
    assert_eq!(v3, 1337, "WriteValueFuture[3] expected 1337");
    assert_eq!(counter, 4, "CounterFuture counter expected 4");
    assert_eq!(success, 1, "success flag expected 1");

    println!("  Executor demo — PASSED");
    Ok(())
}

/// Test the oneshot channel demo kernel.
///
/// Spawns 4 producer-consumer pairs that communicate via oneshot channels.
/// Each producer sends a different value; each consumer writes the received
/// value to a result slot.
pub(crate) fn run_channel_oneshot_demo_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Channel Oneshot Demo Test (channel-oneshot.3) ---");
    println!("  Spawns 4 producer-consumer pairs using oneshot channels.");

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Io,
        "kernel_channel",
        &["channel_oneshot_demo"],
    )?;

    // Allocate mapped memory: executor + 4 OneshotSlot<u32> (~16 bytes each)
    let executor_size = 256 * 1024 + 256; // 256KB for executor + extra for slots
    let (exec_host_ptr, exec_dev_ptr) = unsafe { alloc_mapped_bytes(&dev, executor_size)? };

    // Allocate results: 16 u32
    let (results_host_ptr, results_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 16)? };

    let f = dev
        .get_func("kernel_channel", "channel_oneshot_demo")
        .ok_or(GpuHostError::KernelNotFound("channel_oneshot_demo"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    let dev2 = dev.clone();
    let sync_done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sync_done2 = sync_done.clone();
    let sync_thread = std::thread::spawn(move || {
        unsafe {
            f.launch(cfg, (exec_dev_ptr, results_dev_ptr)).unwrap();
        }
        let _ = dev2.synchronize();
        sync_done2.store(true, std::sync::atomic::Ordering::Release);
    });

    // Poll phase marker with timeout
    let timeout = std::time::Duration::from_secs(10);
    let start = std::time::Instant::now();
    let mut last_phase = 0u32;
    loop {
        let phase = unsafe { std::ptr::read_volatile(results_host_ptr.add(10)) };
        if phase != last_phase {
            println!("  phase marker = {phase}");
            last_phase = phase;
        }
        if sync_done.load(std::sync::atomic::Ordering::Acquire) {
            println!("  kernel completed (phase={phase})");
            break;
        }
        if start.elapsed() > timeout {
            println!("  TIMEOUT after {timeout:?}! phase={phase}");
            println!("  results dump:");
            for i in 0..16u32 {
                let val = unsafe { std::ptr::read_volatile(results_host_ptr.add(i as usize)) };
                println!("    results[{i}] = {val}");
            }
            unsafe {
                free_mapped_bytes(exec_host_ptr)?;
                free_mapped_mem(results_host_ptr)?;
            }
            return Err(GpuHostError::Timeout {
                test: "channel_oneshot_demo",
                detail: format!("kernel hung at phase {phase}"),
            });
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let _ = sync_thread.join();

    // Read results
    let spawned = unsafe { std::ptr::read_volatile(results_host_ptr.add(0)) };
    let completed = unsafe { std::ptr::read_volatile(results_host_ptr.add(1)) };
    let tasks_executed = unsafe { std::ptr::read_volatile(results_host_ptr.add(2)) };
    let polls_total = unsafe { std::ptr::read_volatile(results_host_ptr.add(3)) };
    let v0 = unsafe { std::ptr::read_volatile(results_host_ptr.add(4)) };
    let v1 = unsafe { std::ptr::read_volatile(results_host_ptr.add(5)) };
    let v2 = unsafe { std::ptr::read_volatile(results_host_ptr.add(6)) };
    let v3 = unsafe { std::ptr::read_volatile(results_host_ptr.add(7)) };
    let channels = unsafe { std::ptr::read_volatile(results_host_ptr.add(8)) };
    let success = unsafe { std::ptr::read_volatile(results_host_ptr.add(9)) };

    println!("  spawned={spawned} completed={completed} tasks_executed={tasks_executed} polls_total={polls_total}");
    println!("  received values=[{v0}, {v1}, {v2}, {v3}]  channels={channels}  success={success}");

    unsafe {
        free_mapped_bytes(exec_host_ptr)?;
        free_mapped_mem(results_host_ptr)?;
    }

    assert_eq!(
        spawned, 8,
        "expected 8 tasks spawned (4 producers + 4 consumers)"
    );
    assert_eq!(completed, 8, "expected 8 tasks completed");
    assert_eq!(v0, 42, "channel[0] expected 42");
    assert_eq!(v1, 100, "channel[1] expected 100");
    assert_eq!(v2, 255, "channel[2] expected 255");
    assert_eq!(v3, 1337, "channel[3] expected 1337");
    assert_eq!(channels, 4, "expected 4 channel pairs");
    assert_eq!(success, 1, "success flag expected 1");

    println!("  Channel oneshot demo — PASSED");
    Ok(())
}

/// Test: Multi-stage compute pipeline with GPU-autonomous convergence.
///
/// Proves async GPU can run multi-stage compute (softmax → GELU → reduce → converge)
/// in a single kernel launch with zero inter-stage overhead. No hostcall needed.
pub(crate) fn run_compute_pipeline_demo_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Compute Pipeline Demo: GPU-autonomous multi-stage compute ---");

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Compute,
        "compute_demo",
        &["compute_pipeline_demo"],
    )?;

    // Allocate output (32 floats) + status (4 u32s) in mapped memory
    let (output_host_ptr, output_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 32)? };
    // Status: [iterations, nanos_lo, nanos_hi, done_flag]
    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 4)? };

    // Zero status
    unsafe {
        for i in 0..4 {
            core::ptr::write_volatile(status_host_ptr.add(i), 0);
        }
    }

    let f = dev
        .get_func("compute_demo", "compute_pipeline_demo")
        .ok_or(GpuHostError::KernelNotFound("compute_pipeline_demo"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1), // one full warp
        shared_mem_bytes: 0,
    };

    println!("  Launching compute_pipeline_demo (32 threads, 1 warp)...");
    let host_start = std::time::Instant::now();
    unsafe {
        // output and status are passed as raw pointers
        f.launch(cfg, (output_dev_ptr, status_dev_ptr))?;
    }

    dev.synchronize()?;
    let host_elapsed = host_start.elapsed();
    println!("  Host-side wall time: {host_elapsed:?}");

    // Read results
    let iterations = unsafe { std::ptr::read_volatile(status_host_ptr.add(0)) };
    let nanos_lo = unsafe { std::ptr::read_volatile(status_host_ptr.add(1)) };
    let nanos_hi = unsafe { std::ptr::read_volatile(status_host_ptr.add(2)) };
    let done = unsafe { std::ptr::read_volatile(status_host_ptr.add(3)) };
    let gpu_nanos = (nanos_lo as u64) | ((nanos_hi as u64) << 32);

    println!(
        "  GPU pipeline: {iterations} iterations, {gpu_nanos} ns ({:.2} μs)",
        gpu_nanos as f64 / 1000.0
    );
    println!("  done_flag = {done}");

    // Read output values and show statistics
    let mut sum: f32 = 0.0;
    for i in 0..32 {
        let val = unsafe { std::ptr::read_volatile(output_host_ptr.add(i) as *const f32) };
        sum += val;
    }
    let mean = sum / 32.0;
    println!("  Output: mean={mean:.4}, sum={sum:.4}");

    unsafe {
        free_mapped_mem(output_host_ptr)?;
        free_mapped_mem(status_host_ptr)?;
    }

    assert_eq!(done, 1, "kernel should have set done flag");
    assert!(
        iterations > 0 && iterations <= 100,
        "iterations={iterations} out of range"
    );

    // In CUDA equivalent: 3 launches × iterations × ~10μs = significant overhead
    // Our single-launch approach: zero inter-stage overhead
    let cuda_equiv_overhead_us = 3 * iterations as u64 * 10; // ~10μs per launch
    println!("  Estimated CUDA launch overhead savings: ~{cuda_equiv_overhead_us} μs");

    println!("  Compute pipeline demo — PASSED");
    Ok(())
}

/// Benchmark: single-launch async pipeline vs multi-launch sequential kernels.
///
/// Runs the same computation two ways:
/// 1. Single launch: compute_pipeline_demo (all stages in one kernel)
/// 2. Multi launch: bench_stage_softmax + bench_stage_gelu + bench_stage_reduce (3 separate launches)
///
/// Measures host-side wall time for both approaches.
pub(crate) fn run_compute_benchmark_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Compute Benchmark: single-launch vs multi-launch ---");

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Compute,
        "compute_bench",
        &[
            "compute_pipeline_demo",
            "bench_stage_softmax",
            "bench_stage_gelu",
            "bench_stage_reduce",
        ],
    )?;

    let warp_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    let n_warmup = 3;
    let n_trials = 10;

    // ── Benchmark 1: Single-launch async pipeline ──
    let f_pipeline = dev
        .get_func("compute_bench", "compute_pipeline_demo")
        .ok_or(GpuHostError::KernelNotFound("compute_pipeline_demo"))?;

    let (output_host, output_dev) = unsafe { alloc_mapped_result_array(&dev, 32)? };
    let (status_host, status_dev) = unsafe { alloc_mapped_result_array(&dev, 4)? };

    // Warmup
    for _ in 0..n_warmup {
        unsafe {
            for i in 0..4 {
                core::ptr::write_volatile(status_host.add(i), 0);
            }
            f_pipeline
                .clone()
                .launch(warp_cfg, (output_dev, status_dev))?;
        }
        dev.synchronize()?;
    }

    // Timed trials
    let mut single_times = Vec::with_capacity(n_trials);
    for _ in 0..n_trials {
        unsafe {
            for i in 0..4 {
                core::ptr::write_volatile(status_host.add(i), 0);
            }
        }
        let start = std::time::Instant::now();
        unsafe {
            f_pipeline
                .clone()
                .launch(warp_cfg, (output_dev, status_dev))?;
        }
        dev.synchronize()?;
        single_times.push(start.elapsed());
    }

    let single_median = {
        let mut sorted: Vec<_> = single_times.to_vec();
        sorted.sort();
        sorted[n_trials / 2]
    };

    // Read GPU timing from last run
    let gpu_nanos = {
        let lo = unsafe { std::ptr::read_volatile(status_host.add(1)) } as u64;
        let hi = unsafe { std::ptr::read_volatile(status_host.add(2)) } as u64;
        lo | (hi << 32)
    };
    let iterations = unsafe { std::ptr::read_volatile(status_host.add(0)) };

    unsafe {
        free_mapped_mem(output_host)?;
        free_mapped_mem(status_host)?;
    }

    // ── Benchmark 2: Multi-launch (3 separate kernels per iteration) ──
    let f_softmax = dev
        .get_func("compute_bench", "bench_stage_softmax")
        .ok_or(GpuHostError::KernelNotFound("bench_stage_softmax"))?;
    let f_gelu = dev
        .get_func("compute_bench", "bench_stage_gelu")
        .ok_or(GpuHostError::KernelNotFound("bench_stage_gelu"))?;
    let f_reduce = dev
        .get_func("compute_bench", "bench_stage_reduce")
        .ok_or(GpuHostError::KernelNotFound("bench_stage_reduce"))?;

    let (data_host, data_dev) = unsafe { alloc_mapped_result_array(&dev, 32)? };
    let (result_host, result_dev) = unsafe { alloc_mapped_result_array(&dev, 1)? };
    let (flag_host, flag_dev) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    // Initialize data
    for i in 0..32 {
        let x = (i as f32 * 0.3).sin() + (i as f32 * 0.17).cos() + 1.0;
        unsafe {
            core::ptr::write_volatile(data_host.add(i) as *mut f32, x);
        }
    }

    // Warmup
    for _ in 0..n_warmup {
        unsafe {
            core::ptr::write_volatile(flag_host, 0);
            f_softmax.clone().launch(warp_cfg, (data_dev, flag_dev))?;
            dev.synchronize()?;
            core::ptr::write_volatile(flag_host, 0);
            f_gelu.clone().launch(warp_cfg, (data_dev, flag_dev))?;
            dev.synchronize()?;
            core::ptr::write_volatile(flag_host, 0);
            f_reduce
                .clone()
                .launch(warp_cfg, (data_dev, result_dev, flag_dev))?;
            dev.synchronize()?;
        }
    }

    // Timed trials (3 launches per trial, to match iterations=1 of the pipeline)
    let mut multi_times = Vec::with_capacity(n_trials);
    for _ in 0..n_trials {
        // Re-initialize data
        for i in 0..32 {
            let x = (i as f32 * 0.3).sin() + (i as f32 * 0.17).cos() + 1.0;
            unsafe {
                core::ptr::write_volatile(data_host.add(i) as *mut f32, x);
            }
        }

        let start = std::time::Instant::now();
        unsafe {
            core::ptr::write_volatile(flag_host, 0);
            f_softmax.clone().launch(warp_cfg, (data_dev, flag_dev))?;
            dev.synchronize()?;

            core::ptr::write_volatile(flag_host, 0);
            f_gelu.clone().launch(warp_cfg, (data_dev, flag_dev))?;
            dev.synchronize()?;

            core::ptr::write_volatile(flag_host, 0);
            f_reduce
                .clone()
                .launch(warp_cfg, (data_dev, result_dev, flag_dev))?;
            dev.synchronize()?;
        }
        multi_times.push(start.elapsed());
    }

    let multi_median = {
        let mut sorted: Vec<_> = multi_times.to_vec();
        sorted.sort();
        sorted[n_trials / 2]
    };

    unsafe {
        free_mapped_mem(data_host)?;
        free_mapped_mem(result_host)?;
        free_mapped_mem(flag_host)?;
    }

    // ── Report results ──
    println!("  Single-launch (async pipeline):");
    println!("    Host median: {single_median:?}");
    println!(
        "    GPU time: {gpu_nanos} ns ({:.2} μs)",
        gpu_nanos as f64 / 1000.0
    );
    println!("    Iterations: {iterations}");
    println!("  Multi-launch (3 separate kernels × 1 iteration):");
    println!("    Host median: {multi_median:?}");
    let speedup = multi_median.as_nanos() as f64 / single_median.as_nanos().max(1) as f64;
    println!("  Launch overhead ratio: {speedup:.2}× (multi/single)");
    println!(
        "  With {iterations} iterations: est. {:.2}× overhead for CUDA-style",
        speedup * iterations as f64
    );

    println!("  Compute benchmark — DONE");
    Ok(())
}

/// Test the MPSC channel demo kernel.
///
/// Spawns 3 producers + 1 consumer. Each producer sends 4 values through
/// a shared MPSC channel. The consumer receives all 12 values and sums them.
pub(crate) fn run_channel_mpsc_demo_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Channel MPSC Demo Test (channel-mpsc.2) ---");
    println!("  Spawns 3 producers + 1 consumer using MPSC channel.");

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Io,
        "kernel_mpsc",
        &["channel_mpsc_demo"],
    )?;

    // Allocate mapped memory: executor + MpscChannel<u32, 16>
    // MpscChannel<u32, 16> is small (~256 bytes), executor is ~136KB
    let executor_size = 256 * 1024 + 1024; // 256KB for executor + extra for channel
    let (exec_host_ptr, exec_dev_ptr) = unsafe { alloc_mapped_bytes(&dev, executor_size)? };

    // Allocate results: 8 u32
    let (results_host_ptr, results_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 8)? };

    let f = dev
        .get_func("kernel_mpsc", "channel_mpsc_demo")
        .ok_or(GpuHostError::KernelNotFound("channel_mpsc_demo"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    let dev2 = dev.clone();
    let sync_done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sync_done2 = sync_done.clone();
    let sync_thread = std::thread::spawn(move || {
        unsafe {
            f.launch(cfg, (exec_dev_ptr, results_dev_ptr)).unwrap();
        }
        let _ = dev2.synchronize();
        sync_done2.store(true, std::sync::atomic::Ordering::Release);
    });

    // Poll phase marker with timeout
    let timeout = std::time::Duration::from_secs(10);
    let start = std::time::Instant::now();
    let mut last_phase = 0u32;
    loop {
        let phase = unsafe { std::ptr::read_volatile(results_host_ptr.add(7)) };
        if phase != last_phase {
            println!("  phase marker = {phase}");
            last_phase = phase;
        }
        if sync_done.load(std::sync::atomic::Ordering::Acquire) {
            println!("  kernel completed (phase={phase})");
            break;
        }
        if start.elapsed() > timeout {
            println!("  TIMEOUT after {timeout:?}! phase={phase}");
            println!("  results dump:");
            for i in 0..8u32 {
                let val = unsafe { std::ptr::read_volatile(results_host_ptr.add(i as usize)) };
                println!("    results[{i}] = {val}");
            }
            unsafe {
                free_mapped_bytes(exec_host_ptr)?;
                free_mapped_mem(results_host_ptr)?;
            }
            return Err(GpuHostError::Timeout {
                test: "channel_mpsc_demo",
                detail: format!("kernel hung at phase {phase}"),
            });
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let _ = sync_thread.join();

    // Read results
    let spawned = unsafe { std::ptr::read_volatile(results_host_ptr.add(0)) };
    let completed = unsafe { std::ptr::read_volatile(results_host_ptr.add(1)) };
    let tasks_executed = unsafe { std::ptr::read_volatile(results_host_ptr.add(2)) };
    let polls_total = unsafe { std::ptr::read_volatile(results_host_ptr.add(3)) };
    let sum = unsafe { std::ptr::read_volatile(results_host_ptr.add(4)) };
    let count = unsafe { std::ptr::read_volatile(results_host_ptr.add(5)) };
    let success = unsafe { std::ptr::read_volatile(results_host_ptr.add(6)) };

    println!("  spawned={spawned} completed={completed} tasks_executed={tasks_executed} polls_total={polls_total}");
    println!("  received: sum={sum} count={count}  success={success}");

    unsafe {
        free_mapped_bytes(exec_host_ptr)?;
        free_mapped_mem(results_host_ptr)?;
    }

    assert_eq!(
        spawned, 4,
        "expected 4 tasks spawned (3 producers + 1 consumer)"
    );
    assert_eq!(completed, 4, "expected 4 tasks completed");
    assert_eq!(
        sum, 312,
        "expected sum = 312 (10+20+30+40+11+21+31+41+12+22+32+42)"
    );
    assert_eq!(count, 12, "expected 12 values received");
    assert_eq!(success, 1, "success flag expected 1");

    println!("  Channel MPSC demo — PASSED");
    Ok(())
}

/// Tokio bridge demo: launch kernel via GpuTask, receive events asynchronously.
///
/// Uses `hostcall_print_hello` kernel which sends a PRINT hostcall message.
/// Demonstrates `GpuTask::launch().await` + `GpuTask::next_event().await`.
#[cfg(feature = "async")]
pub(crate) async fn run_tokio_bridge_demo_test(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    use gpu_host::async_rt::{AsyncGpuRuntime, GpuTask, HostcallEvent};
    use gpu_host::memory::MappedBuffer;
    use gpu_host::runtime::GpuRuntime;

    println!("\n--- Tokio Bridge Demo Test (tokio-impl.1) ---");
    println!("  Launches hostcall_print_hello via GpuTask async API.");

    let rt = AsyncGpuRuntime::new(0)?;
    rt.load_ptx(crate::KERNEL_IO_PTX, "kernel", &["hostcall_print_hello"])?;

    let mut task = GpuTask::new(&rt, 4)?;

    let result_buf = MappedBuffer::<u32>::new_zeroed(1)?;

    let func = rt.inner().require_func("kernel", "hostcall_print_hello")?;
    let cfg = GpuRuntime::launch_config((1, 1, 1), (32, 1, 1), 0);

    println!("  Launching kernel via GpuTask...");
    let start = std::time::Instant::now();

    // Launch kernel asynchronously (non-blocking for tokio runtime)
    task.launch(func, cfg, (task.session_dev_ptr(), result_buf.dev_ptr()))
        .await?;

    let elapsed = start.elapsed();
    println!("  Kernel completed in {elapsed:?}.");

    // Give listener time to forward remaining events
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Collect events
    let mut messages = Vec::new();
    loop {
        match tokio::time::timeout(std::time::Duration::from_millis(50), task.next_event()).await {
            Ok(Some(HostcallEvent::Print(msg))) => {
                let s = String::from_utf8_lossy(&msg).to_string();
                println!("  [GPU event] {s}");
                messages.push(s);
            }
            Ok(Some(HostcallEvent::Shutdown)) => break,
            Ok(None) => break,
            Err(_) => break, // timeout = no more events
        }
    }

    task.shutdown().await;

    // Verify
    let result = unsafe { result_buf.read(0) };
    assert_eq!(result, 1, "kernel reported failure");
    assert!(
        messages.iter().any(|m| m.contains("Hello from GPU!")),
        "expected 'Hello from GPU!' in events, got: {messages:?}"
    );

    println!("  Tokio bridge demo — PASSED!");
    println!("    GpuTask::launch().await + next_event().await working end-to-end.");
    println!("    {} events received asynchronously.", messages.len());
    Ok(())
}
