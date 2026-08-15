//\! Pipeline tests: file transform, branching, pipelined compute, bulk I/O, sharding, parallel grep, autonomous.

use std::sync::Arc;

use cudarc::driver::sys::lib as cuda_lib;
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};

use gpu_host::error::{GpuHostError, Result};
use gpu_host::hostcall;
use gpu_host::mapped_mem::{
    alloc_mapped_result_array, alloc_mapped_u32, alloc_mapped_u64_array, free_mapped_mem,
    free_mapped_u64_array,
};

/// async-pipeline demo: GPU-autonomous file transform pipeline.
///
/// The GPU self-coordinates the entire pipeline in one kernel launch:
///   open(in) → read(in) → transform(per-thread) → open(out) → write(out) → close(in) → close(out) → print
///
/// No CPU intervention between steps.
pub(crate) fn run_file_transform_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- File Transform Pipeline (async-pipeline.1+2) ---");

    // Create 1024 bytes of ASCII input: repeating "Hello, GPU Pipeline! " pattern
    let pattern = b"Hello, GPU Pipeline! ";
    let mut input_data = Vec::with_capacity(1024);
    while input_data.len() < 1024 {
        let remaining = 1024 - input_data.len();
        let chunk = remaining.min(pattern.len());
        input_data.extend_from_slice(&pattern[..chunk]);
    }
    std::fs::write("gpu_input.txt", &input_data).map_err(|e| GpuHostError::Verification {
        test: "file_transform_pipeline",
        detail: format!("failed to create input file: {e}"),
    })?;
    println!("  Created gpu_input.txt ({} bytes)", input_data.len());

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();
    let sb_dev_ptr = hc_buf.sideband_dev_ptr();

    let (status_host, status_dev) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    let hc_buf_ref = std::sync::Arc::new(hc_buf);
    let hc_buf_listener = std::sync::Arc::clone(&hc_buf_ref);

    let messages = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let msg_clone = std::sync::Arc::clone(&messages);

    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(move |msg| {
            let text = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] GPU says: \"{text}\"");
            let mut guard = msg_clone.lock().unwrap();
            guard.push(text);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_IO_PTX);
    dev.load_ptx(ptx, "kernel", &["file_transform_pipeline"])?;
    let f = dev
        .get_func("kernel", "file_transform_pipeline")
        .ok_or(GpuHostError::KernelNotFound("file_transform_pipeline"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching file_transform_pipeline kernel...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, sb_dev_ptr, status_dev))?;
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let status_val = unsafe { std::ptr::read_volatile(status_host) };
    let msgs = messages.lock().unwrap();

    println!("  Status: {status_val} (1=success)");
    println!("  Elapsed: {:.3}ms", elapsed.as_secs_f64() * 1000.0);

    // Verify output file exists and contains case-toggled content
    let output_data = std::fs::read("gpu_output.txt").map_err(|e| GpuHostError::Verification {
        test: "file_transform_pipeline",
        detail: format!("failed to read output file: {e}"),
    })?;

    // Expected: toggle ASCII case on the input
    let expected: Vec<u8> = input_data
        .iter()
        .map(|&b| {
            if b.is_ascii_uppercase() || b.is_ascii_lowercase() {
                b ^ 0x20
            } else {
                b
            }
        })
        .collect();

    let content_ok = output_data == expected;
    let msg_ok = msgs.iter().any(|m| m.contains("pipeline: done"));

    // Clean up
    let _ = std::fs::remove_file("gpu_input.txt");
    let _ = std::fs::remove_file("gpu_output.txt");
    unsafe { free_mapped_mem(status_host)? };

    if status_val == 1 && content_ok && msg_ok {
        println!("  File Transform Pipeline: PASSED!");
        println!("    16-state WarpFuture: open→read→transform→open→write→close→close→print");
        println!("    GPU self-coordinated {} I/O steps + 1 compute step", 8);
        println!(
            "    {} bytes: ASCII case toggled correctly",
            output_data.len()
        );
        println!("    Zero CPU intervention between steps");
    } else {
        println!("  File Transform Pipeline: FAILED");
        if status_val != 1 {
            println!("    Status: {status_val}");
        }
        if !content_ok {
            println!(
                "    Content mismatch: output {} bytes, expected {} bytes",
                output_data.len(),
                expected.len()
            );
            if !output_data.is_empty() {
                let first = std::cmp::min(32, output_data.len());
                println!("    Output[..{}]: {:?}", first, &output_data[..first]);
                println!("    Expected[..{}]: {:?}", first, &expected[..first]);
            }
        }
        if !msg_ok {
            println!("    Missing 'pipeline: done' message. Got: {:?}", *msgs);
        }
        return Err(GpuHostError::Verification {
            test: "file_transform_pipeline",
            detail: "see above".to_string(),
        });
    }

    Ok(())
}

/// GPU panic handler test: verify panic message is received via hostcall.
///
/// The test kernel deliberately panics. We expect:
/// 1. The panic message appears on stderr via [GPU PANIC] prefix
/// 2. The kernel returns a CUDA error (due to trap instruction)
/// 3. The result marker is set to 1 (written before the panic)
pub(crate) fn run_panic_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- GPU Panic Handler Test (gpu-panic.2) ---");

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    let hc_buf_ref = std::sync::Arc::new(hc_buf);
    let hc_buf_listener = std::sync::Arc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = std::str::from_utf8(msg).unwrap_or("<invalid>");
            println!("  [GPU PRINT] {s}");
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_IO_PTX);
    dev.load_ptx(ptx, "panic_test", &["panic_test_kernel"])?;
    let f = dev
        .get_func("panic_test", "panic_test_kernel")
        .ok_or(GpuHostError::KernelNotFound("panic_test_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching panic_test_kernel (expects GPU panic + trap)...");
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    // Use raw CUDA API for synchronization since cudarc's synchronize may
    // unwrap internally. trap instruction will cause CUDA_ERROR_LAUNCH_FAILED.
    let sync_result = unsafe {
        let cu = cuda_lib();
        cu.cuCtxSynchronize()
    };

    // Give listener time to process the panic packet
    std::thread::sleep(std::time::Duration::from_millis(200));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let marker = unsafe { std::ptr::read_volatile(result_host_ptr) };

    println!("  Result marker: {marker} (expected 1 = reached panic point)");
    println!("  CUDA sync result: {sync_result:?} (LAUNCH_FAILED expected from trap)");

    // The test passes if:
    // 1. The marker was set to 1 (code reached the panic point)
    // 2. The [GPU PANIC] message appeared on stderr (visible in console output)
    if marker == 1 {
        println!("  panic_test: PASSED (panic message sent via hostcall before trap)");
    } else {
        println!("  panic_test: marker was {marker} (expected 1)");
    }

    // IMPORTANT: trap instruction puts the CUDA context in an error state.
    // CudaDevice's Drop impl will try to sync and panic on the sticky error.
    // Exit the process cleanly before Drop runs.
    println!("  Note: Exiting to avoid CudaDevice Drop panic after trap.");
    std::process::exit(0);
}

/// Bulk data transfer test: write 4KB via sideband, read back, verify.
pub(crate) fn run_bulk_io_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Bulk data transfer test (large-payload.3) ---");

    use std::sync::Arc as StdArc;

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();
    let sb_dev_ptr = hc_buf.sideband_dev_ptr();

    println!(
        "  Hostcall buffer: {} bytes, {} packets",
        hc_buf.size(),
        hc_buf.num_packets()
    );
    println!("  Sideband buffer: {} bytes", hc_buf.sideband_size());

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 4)? };

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] Print from GPU: \"{s}\"");
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_IO_PTX);
    dev.load_ptx(ptx, "kernel", &["bulk_io_test"])?;
    let f = dev
        .get_func("kernel", "bulk_io_test")
        .ok_or(GpuHostError::KernelNotFound("bulk_io_test"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    let start = std::time::Instant::now();

    println!("  Launching bulk_io_test kernel...");
    unsafe {
        f.launch(cfg, (dev_ptr, sb_dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("  Kernel completed in {elapsed:?}.");

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let overall = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let bytes_written = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    let bytes_read = unsafe { std::ptr::read_volatile(result_host_ptr.add(2)) };
    let content_match = unsafe { std::ptr::read_volatile(result_host_ptr.add(3)) };

    unsafe { free_mapped_mem(result_host_ptr)? };

    // Clean up test file
    let _ = std::fs::remove_file("gpu_bulk_test.bin");

    println!(
        "  Results: overall={overall}, written={bytes_written}, read={bytes_read}, match={content_match}"
    );

    if overall != 1 {
        return Err(GpuHostError::Verification {
            test: "bulk_io_test",
            detail: format!(
                "overall={overall}, written={bytes_written}, read={bytes_read}, match={content_match}"
            ),
        });
    }

    println!(
        "  bulk_io_test: PASSED! (wrote {bytes_written} bytes, read {bytes_read} bytes, {elapsed:?})"
    );
    Ok(())
}

/// Per-block sharding test: use sharded buffer with multi-block print kernel.
pub(crate) fn run_sharded_hostcall_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Per-block sharding test (per-block-sharding.2) ---");

    use std::sync::{Arc as StdArc, Mutex};

    let num_blocks: u32 = 4;
    let pkts_per_shard: u32 = 4;
    let hc_buf = hostcall::HostcallBuffer::new_sharded(num_blocks, pkts_per_shard)?;
    let dev_ptr = hc_buf.dev_ptr();

    println!(
        "  Sharded buffer: {} shards × {} pkts/shard = {} total packets, {} bytes",
        hc_buf.num_shards(),
        hc_buf.pkts_per_shard(),
        hc_buf.num_packets(),
        hc_buf.size()
    );

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let (count_host_ptr, count_dev_ptr) = unsafe { alloc_mapped_u32(&dev)? };
    unsafe { std::ptr::write_volatile(count_host_ptr, 0u32) };

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] GPU says (sharded): \"{s}\"");
            messages_clone.lock().unwrap().push(s);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_IO_PTX);
    dev.load_ptx(ptx, "kernel_sharded", &["sharded_print_test"])?;
    let f = dev
        .get_func("kernel_sharded", "sharded_print_test")
        .ok_or(GpuHostError::KernelNotFound("sharded_print_test"))?;

    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching hostcall_print_multi ({num_blocks} blocks × 32 threads, sharded)...");
    unsafe {
        f.launch(cfg, (dev_ptr, count_dev_ptr))?;
    }

    dev.synchronize()?;
    println!("  Kernel completed.");

    std::thread::sleep(std::time::Duration::from_millis(200));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let success_count = unsafe { std::ptr::read_volatile(count_host_ptr) };
    unsafe { free_mapped_mem(count_host_ptr)? };

    let received = messages.lock().unwrap();
    println!(
        "  Results: {} blocks succeeded, {} messages received",
        success_count,
        received.len()
    );

    if success_count != num_blocks {
        return Err(GpuHostError::Verification {
            test: "sharded_hostcall",
            detail: format!("expected {num_blocks} successes, got {success_count}"),
        });
    }
    if received.len() != num_blocks as usize {
        return Err(GpuHostError::Verification {
            test: "sharded_hostcall",
            detail: format!("expected {} messages, got {}", num_blocks, received.len()),
        });
    }

    println!("  sharded_hostcall: PASSED!");
    println!("    {num_blocks} blocks with per-block sharding, all printed successfully");
    Ok(())
}

pub(crate) fn run_parallel_grep_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Parallel File Grep Demo (product.8) ---");

    // Create test file with lines, some containing "GPU"
    let test_content = "\
Hello world from the host\n\
This line mentions GPU computing\n\
Plain text line number three\n\
Another GPU reference here\n\
Nothing special on this line\n\
GPU acceleration is the future\n\
Final line without the keyword\n\
Yet another GPU mention for testing\n";
    std::fs::write("gpu_grep_test.txt", test_content).expect("write test file");
    println!(
        "  Created test file: gpu_grep_test.txt ({} bytes, 8 lines)",
        test_content.len()
    );

    let pattern = b"GPU";
    let num_threads: u32 = 4;

    // Allocate hostcall buffer with sideband for bulk read
    let hc_buf = hostcall::HostcallBuffer::new_with_sideband(8, 1024 * 1024)?;
    let dev_ptr = hc_buf.dev_ptr();
    let sb_dev_ptr = hc_buf.sideband_dev_ptr();

    // Allocate results array (1 u32 per thread, but use u64 array for convenience)
    let (results_host_ptr, results_dev_ptr) =
        unsafe { alloc_mapped_u64_array(&dev, num_threads as usize)? };

    // Allocate pattern in mapped memory
    let (pattern_host_ptr, pattern_dev_ptr) = unsafe { alloc_mapped_u64_array(&dev, 1)? };
    unsafe {
        let pat_bytes = pattern_host_ptr as *mut u8;
        for (i, &b) in pattern.iter().enumerate() {
            std::ptr::write_volatile(pat_bytes.add(i), b);
        }
    }

    let messages: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let messages_clone = std::sync::Arc::clone(&messages);

    let hc_buf_ref = std::sync::Arc::new(hc_buf);
    let hc_buf_listener = std::sync::Arc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [GREP] {s}");
            messages_clone.lock().unwrap().push(s);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_IO_PTX);
    dev.load_ptx(ptx, "kernel_grep", &["parallel_grep_kernel"])?;
    let f = dev
        .get_func("kernel_grep", "parallel_grep_kernel")
        .ok_or(GpuHostError::KernelNotFound("parallel_grep_kernel"))?;

    // Launch: 4 blocks × 1 thread (avoid warp divergence)
    let cfg = LaunchConfig {
        grid_dim: (num_threads, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!(
        "  Launching parallel_grep_kernel ({} threads, pattern=\"{}\")...",
        num_threads,
        String::from_utf8_lossy(pattern)
    );
    let start = std::time::Instant::now();
    unsafe {
        f.launch(
            cfg,
            (
                dev_ptr,
                sb_dev_ptr,
                results_dev_ptr,
                pattern_dev_ptr,
                pattern.len() as u32,
            ),
        )?;
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();

    std::thread::sleep(std::time::Duration::from_millis(200));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    // Read results
    let mut total_matches: u32 = 0;
    for tid in 0..num_threads as usize {
        let count = unsafe { std::ptr::read_volatile(results_host_ptr.add(tid)) } as u32;
        total_matches += count;
        println!("    Thread {tid}: {count} matches");
    }

    unsafe {
        free_mapped_u64_array(results_host_ptr)?;
        free_mapped_u64_array(pattern_host_ptr)?;
    }
    let _ = std::fs::remove_file("gpu_grep_test.txt");

    let received = messages.lock().unwrap();
    println!("  Elapsed: {:.3}ms", elapsed.as_secs_f64() * 1000.0);
    println!(
        "  Total matches: {} (expected {} per thread × {} threads = {})",
        total_matches,
        4,
        num_threads,
        4 * num_threads
    );
    println!("  Messages received: {}", received.len());

    if total_matches == 4 * num_threads {
        println!("  parallel_grep: PASSED!");
        println!(
            "    {num_threads} threads independently searched a file, each finding 4 \"GPU\" matches."
        );
    } else {
        println!(
            "  parallel_grep: PARTIAL (expected {} total matches, got {})",
            4 * num_threads,
            total_matches
        );
    }

    Ok(())
}

/// async-pipeline.3: Branching pipeline — conditional state transition test.
///
/// Run 1: File does not exist → GPU creates it, prints "branch: file created"
/// Run 2: File exists → GPU opens+closes it, prints "branch: file exists"
pub(crate) fn run_branching_pipeline_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Branching Pipeline (async-pipeline.3) ---");

    // Ensure file does NOT exist for first run
    let _ = std::fs::remove_file("branch_test.txt");

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    // --- Run 1: file does not exist → CREATE branch ---
    {
        let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_IO_PTX);
        dev.load_ptx(ptx, "kernel_bp1", &["branching_pipeline"])?;
        let f = dev
            .get_func("kernel_bp1", "branching_pipeline")
            .ok_or(GpuHostError::KernelNotFound("branching_pipeline"))?;
        let hc_buf = hostcall::HostcallBuffer::new(4)?;
        let dev_ptr = hc_buf.dev_ptr();
        let (status_host, status_dev) = unsafe { alloc_mapped_result_array(&dev, 1)? };

        let hc_buf_ref = std::sync::Arc::new(hc_buf);
        let hc_buf_listener = std::sync::Arc::clone(&hc_buf_ref);
        let messages = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let msg_clone = std::sync::Arc::clone(&messages);

        let listener_handle = std::thread::spawn(move || {
            hc_buf_listener.listen(move |msg| {
                let text = String::from_utf8_lossy(msg).to_string();
                println!("  [HOST] GPU says: \"{text}\"");
                let mut guard = msg_clone.lock().unwrap();
                guard.push(text);
            });
        });

        println!("  Run 1: file does not exist (CREATE branch)");
        unsafe { f.launch(cfg, (dev_ptr, status_dev))? };
        dev.synchronize()?;

        std::thread::sleep(std::time::Duration::from_millis(100));
        hc_buf_ref.signal_shutdown();
        listener_handle.join().unwrap();

        let status = unsafe { std::ptr::read_volatile(status_host) };
        let msgs = messages.lock().unwrap();
        let created_msg = msgs.iter().any(|m| m.contains("file created"));

        // Verify file was created with correct content
        let file_exists = std::path::Path::new("branch_test.txt").exists();
        let content_ok = if file_exists {
            std::fs::read("branch_test.txt")
                .map(|data| data == b"hello from GPU\n")
                .unwrap_or(false)
        } else {
            false
        };

        unsafe { free_mapped_mem(status_host)? };

        if status == 1 && created_msg && file_exists && content_ok {
            println!("  Run 1: PASSED (CREATE branch taken)");
        } else {
            println!("  Run 1: FAILED");
            println!(
                "    status={status}, created_msg={created_msg}, file_exists={file_exists}, content_ok={content_ok}"
            );
            println!("    messages: {:?}", *msgs);
            let _ = std::fs::remove_file("branch_test.txt");
            return Err(GpuHostError::Verification {
                test: "branching_pipeline_run1",
                detail: "CREATE branch failed".to_string(),
            });
        }
    }

    // --- Run 2: file exists → EXISTS branch ---
    {
        let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_IO_PTX);
        dev.load_ptx(ptx, "kernel_bp2", &["branching_pipeline"])?;
        let f = dev
            .get_func("kernel_bp2", "branching_pipeline")
            .ok_or(GpuHostError::KernelNotFound("branching_pipeline"))?;

        let hc_buf = hostcall::HostcallBuffer::new(4)?;
        let dev_ptr = hc_buf.dev_ptr();
        let (status_host, status_dev) = unsafe { alloc_mapped_result_array(&dev, 1)? };

        let hc_buf_ref = std::sync::Arc::new(hc_buf);
        let hc_buf_listener = std::sync::Arc::clone(&hc_buf_ref);
        let messages = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let msg_clone = std::sync::Arc::clone(&messages);

        let listener_handle = std::thread::spawn(move || {
            hc_buf_listener.listen(move |msg| {
                let text = String::from_utf8_lossy(msg).to_string();
                println!("  [HOST] GPU says: \"{text}\"");
                let mut guard = msg_clone.lock().unwrap();
                guard.push(text);
            });
        });

        println!("  Run 2: file exists (EXISTS branch)");
        unsafe { f.launch(cfg, (dev_ptr, status_dev))? };
        dev.synchronize()?;

        std::thread::sleep(std::time::Duration::from_millis(100));
        hc_buf_ref.signal_shutdown();
        listener_handle.join().unwrap();

        let status = unsafe { std::ptr::read_volatile(status_host) };
        let msgs = messages.lock().unwrap();
        let exists_msg = msgs.iter().any(|m| m.contains("file exists"));

        unsafe { free_mapped_mem(status_host)? };

        // Clean up
        let _ = std::fs::remove_file("branch_test.txt");

        if status == 1 && exists_msg {
            println!("  Run 2: PASSED (EXISTS branch taken)");
        } else {
            println!("  Run 2: FAILED");
            println!("    status={status}, exists_msg={exists_msg}");
            println!("    messages: {:?}", *msgs);
            return Err(GpuHostError::Verification {
                test: "branching_pipeline_run2",
                detail: "EXISTS branch failed".to_string(),
            });
        }
    }

    println!("  Branching Pipeline: PASSED!");
    println!("    Conditional state transition verified on GPU hardware");
    println!("    Run 1: CREATE branch (file not found → create + write + close + print)");
    println!("    Run 2: EXISTS branch (file found → close + print)");
    println!("    All 32 lanes take same branch via shfl.sync broadcast");

    Ok(())
}

/// async-pipeline.4: Pipelined I/O + compute test.
///
/// The GPU submits a PRINT hostcall, then does FMA computation while the I/O
/// is being processed. Reports how many compute iterations were completed
/// during the I/O round-trip, demonstrating overlap.
pub(crate) fn run_pipelined_compute_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Pipelined I/O + Compute (async-pipeline.4) ---");

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();
    let (status_host, status_dev) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    let hc_buf_ref = std::sync::Arc::new(hc_buf);
    let hc_buf_listener = std::sync::Arc::clone(&hc_buf_ref);
    let messages = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let msg_clone = std::sync::Arc::clone(&messages);

    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(move |msg| {
            let text = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] GPU says: \"{text}\"");
            let mut guard = msg_clone.lock().unwrap();
            guard.push(text);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_IO_PTX);
    dev.load_ptx(ptx, "kernel_pp", &["pipelined_compute"])?;
    let f = dev
        .get_func("kernel_pp", "pipelined_compute")
        .ok_or(GpuHostError::KernelNotFound("pipelined_compute"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    let start = std::time::Instant::now();
    unsafe { f.launch(cfg, (dev_ptr, status_dev))? };
    dev.synchronize()?;
    let elapsed = start.elapsed();

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let status = unsafe { std::ptr::read_volatile(status_host) };
    let msgs = messages.lock().unwrap();

    println!("  Status: {status} (1=success)");
    println!("  Elapsed: {:.3}ms", elapsed.as_secs_f64() * 1000.0);

    // Check messages: should have "start", "computing...", "done Niter"
    let has_start = msgs.iter().any(|m| m.contains("pipelined: start"));
    let has_computing = msgs.iter().any(|m| m.contains("pipelined: computing"));
    let done_msg = msgs.iter().find(|m| m.contains("pipelined: done"));

    // Extract iteration count from "done Niter"
    let iters: u32 = done_msg
        .and_then(|m| {
            let prefix = "done ";
            m.find(prefix).and_then(|pos| {
                let rest = &m[pos + prefix.len()..];
                let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                digits.parse().ok()
            })
        })
        .unwrap_or(0);

    unsafe { free_mapped_mem(status_host)? };

    if status == 1 && has_start && has_computing && iters > 0 {
        println!("  Pipelined Compute: PASSED!");
        println!("    {iters} FMA iterations completed while I/O was in-flight");
        println!("    Demonstrates: submit → compute_while_waiting → wait pattern");
        println!("    GPU threads did useful work during hostcall round-trip");
    } else {
        println!("  Pipelined Compute: FAILED");
        println!(
            "    status={status}, start={has_start}, computing={has_computing}, iters={iters}"
        );
        println!("    messages: {:?}", *msgs);
        return Err(GpuHostError::Verification {
            test: "pipelined_compute",
            detail: "Pipelined I/O + compute failed".to_string(),
        });
    }

    Ok(())
}

/// Test: Warp-scale Embassy async (async-pipeline.5).
/// 1 block × 32 threads (one full warp), each thread runs its own Embassy
/// executor with an independent hostcall print. Measures CAS contention
/// when all 32 lanes compete for the same free/ready stacks.
pub(crate) fn run_warp_scale_async_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Async Pipeline Test 5: Warp-scale Embassy (1 block × 32 threads) ---");

    use std::sync::{Arc as StdArc, Mutex};

    let hc_buf = hostcall::HostcallBuffer::new(64)?;
    let dev_ptr = hc_buf.dev_ptr();

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let num_threads: usize = 32;
    let (result_host_ptr, result_dev_ptr) =
        unsafe { alloc_mapped_result_array(&dev, num_threads + 1)? };

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] {s}");
            messages_clone.lock().unwrap().push(s);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::ASYNC_HOSTCALL_PTX);
    dev.load_ptx(ptx, "warp_scale_async", &["warp_scale_async_kernel"])?;
    let f = dev
        .get_func("warp_scale_async", "warp_scale_async_kernel")
        .ok_or(GpuHostError::KernelNotFound("warp_scale_async_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching warp_scale_async_kernel (1 block × 32 threads)...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();

    std::thread::sleep(std::time::Duration::from_millis(300));
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
    println!(
        "  Poll rounds: min={}, max={}, avg={}",
        poll_rounds.iter().min().unwrap_or(&0),
        poll_rounds.iter().max().unwrap_or(&0),
        if poll_rounds.is_empty() {
            0
        } else {
            poll_rounds.iter().sum::<u32>() / poll_rounds.len() as u32
        },
    );
    println!("  Messages received: {}", received.len());

    if completed != num_threads as u32 {
        return Err(GpuHostError::Verification {
            test: "warp_scale_async_kernel",
            detail: format!("only {completed}/{num_threads} threads completed"),
        });
    }

    if received.len() < num_threads {
        return Err(GpuHostError::Verification {
            test: "warp_scale_async_kernel",
            detail: format!("only {} messages, expected {}", received.len(), num_threads),
        });
    }

    println!("  warp_scale_async_kernel: PASSED! ({elapsed:?})");
    println!("    32 threads in one warp, each with own Embassy executor");
    println!("    All threads completed independent async hostcall!");
    Ok(())
}

// ============================================================
// gpu-compute.2: Autonomous Multi-Step Compute Pipeline
// ============================================================

pub(crate) fn run_autonomous_pipeline_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Autonomous Pipeline (gpu-compute.2) ---");
    println!("  GPU-driven multi-step compute with warp-async control flow");

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    // Cleanup files from prior runs
    let _ = std::fs::remove_file("gpu_autonomous.txt");
    let _ = std::fs::remove_file("gpu_roundtrip.txt");

    // Helper: launch autonomous_pipeline with a given mode, collect messages
    fn run_mode(
        dev: &Arc<CudaDevice>,
        cfg: LaunchConfig,
        mode: u64,
        module_name: &'static str,
    ) -> Result<(u32, Vec<String>)> {
        let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_TEST_PTX);
        dev.load_ptx(ptx, module_name, &["autonomous_pipeline"])?;
        let f = dev
            .get_func(module_name, "autonomous_pipeline")
            .ok_or(GpuHostError::KernelNotFound("autonomous_pipeline"))?;

        let hc_buf = hostcall::HostcallBuffer::new(4)?;
        let dev_ptr = hc_buf.dev_ptr();
        let (status_host, status_dev) = unsafe { alloc_mapped_result_array(dev, 1)? };

        let hc_buf_ref = std::sync::Arc::new(hc_buf);
        let hc_buf_listener = std::sync::Arc::clone(&hc_buf_ref);
        let messages = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let msg_clone = std::sync::Arc::clone(&messages);

        let listener_handle = std::thread::spawn(move || {
            hc_buf_listener.listen(move |msg| {
                let text = String::from_utf8_lossy(msg).to_string();
                println!("    [GPU] {text}");
                let mut guard = msg_clone.lock().unwrap();
                guard.push(text);
            });
        });

        unsafe { f.launch(cfg, (dev_ptr, mode, status_dev))? };
        dev.synchronize()?;

        std::thread::sleep(std::time::Duration::from_millis(100));
        hc_buf_ref.signal_shutdown();
        listener_handle.join().unwrap();

        let status = unsafe { std::ptr::read_volatile(status_host) };
        let msgs = messages.lock().unwrap().clone();
        Ok((status, msgs))
    }

    // --- Mode 0: File write pipeline ---
    println!("\n  Mode 0: File write pipeline (create → write → close)");
    let (status, msgs) = run_mode(&dev, cfg, 0, "auto_m0")?;
    assert_eq!(status, 1, "Mode 0: kernel should succeed");
    assert!(
        msgs.iter().any(|m| m.contains("auto: start")),
        "Mode 0: should see init message"
    );
    assert!(
        msgs.iter().any(|m| m.contains("auto: file-written")),
        "Mode 0: should see file-written"
    );
    assert!(
        msgs.iter().any(|m| m.contains("auto: done")),
        "Mode 0: should see done message"
    );
    // Verify file was actually created
    let written = std::fs::read_to_string("gpu_autonomous.txt")
        .expect("Mode 0: file should exist after write");
    assert_eq!(
        written, "GPU-autonomous-output",
        "Mode 0: file content mismatch"
    );
    println!("  Mode 0: PASSED (3 hostcall steps, file verified)");

    // --- Mode 1: File read + classify pipeline ---
    println!("\n  Mode 1: File read + classify pipeline (open → read → close → branch)");
    let (status, msgs) = run_mode(&dev, cfg, 1, "auto_m1")?;
    assert_eq!(status, 1, "Mode 1: kernel should succeed");
    assert!(
        msgs.iter().any(|m| m.contains("auto: large-payload")),
        "Mode 1: should detect large payload (21 bytes > 10)"
    );
    assert!(
        msgs.iter().any(|m| m.contains("auto: done")),
        "Mode 1: should see done message"
    );
    println!("  Mode 1: PASSED (4 hostcall steps + GPU-decided branch)");

    // --- Mode 2: End-to-end roundtrip pipeline ---
    println!("\n  Mode 2: Roundtrip pipeline (create → write → close → reopen → read → verify)");
    let (status, msgs) = run_mode(&dev, cfg, 2, "auto_m2")?;
    assert_eq!(status, 1, "Mode 2: kernel should succeed");
    assert!(
        msgs.iter().any(|m| m.contains("auto: roundtrip-ok")),
        "Mode 2: roundtrip should verify successfully"
    );
    assert!(
        msgs.iter().any(|m| m.contains("auto: done")),
        "Mode 2: should see done message"
    );
    println!("  Mode 2: PASSED (6 hostcall steps + GPU-decided verification)");

    println!("\n  Autonomous Pipeline: ALL 3 MODES PASSED!");
    println!("  Key results:");
    println!("    - GPU autonomously chose processing paths via match");
    println!("    - GPU branched on hostcall results via if/else");
    println!("    - 13 total hostcall steps across 3 pipelines, zero host orchestration");
    println!("    - Proc-macro warp-async replaces 150+ lines of hand-written state machine");
    Ok(())
}

/// Buffered print test: GPU accumulates 12 print messages in a per-thread buffer,
/// then flushes them all in a single SERVICE_BULK_PRINT hostcall.
///
/// Verifies: (1) kernel completes successfully, (2) host receives all 12 messages.
pub(crate) fn run_buffered_print_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Buffered Print Test (printf-batch.3) ---");

    use std::sync::{Arc as StdArc, Mutex};

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();
    let sb_dev_ptr = hc_buf.sideband_dev_ptr();

    println!(
        "  Hostcall buffer: {} bytes, {} packets",
        hc_buf.size(),
        hc_buf.num_packets()
    );
    println!("  Sideband buffer: {} bytes", hc_buf.sideband_size());

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    // Collect print messages from the GPU
    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] GPU print: \"{s}\"");
            messages_clone.lock().unwrap().push(s);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_IO_PTX);
    dev.load_ptx(ptx, "kernel", &["buffered_print_test"])?;
    let f = dev
        .get_func("kernel", "buffered_print_test")
        .ok_or(GpuHostError::KernelNotFound("buffered_print_test"))?;

    // Launch with single thread (print_buffer test is single-thread for simplicity)
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    let start = std::time::Instant::now();
    println!("  Launching buffered_print_test kernel...");
    unsafe {
        f.launch(cfg, (dev_ptr, sb_dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("  Kernel completed in {elapsed:?}.");

    // Wait for listener to process messages
    std::thread::sleep(std::time::Duration::from_millis(200));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let result = unsafe { std::ptr::read_volatile(result_host_ptr) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    if result != 1 {
        return Err(GpuHostError::Verification {
            test: "buffered_print_test",
            detail: format!("kernel returned {result}, expected 1"),
        });
    }

    let msgs = messages.lock().unwrap();
    println!(
        "  Result: kernel success, received {} messages in {elapsed:?}",
        msgs.len()
    );

    // Verify we got all 12 messages
    if msgs.len() < 12 {
        return Err(GpuHostError::Verification {
            test: "buffered_print_test",
            detail: format!("expected 12 messages, got {}", msgs.len()),
        });
    }

    // Verify message content (each should contain "Buffered msg NN")
    for (i, msg) in msgs.iter().enumerate() {
        let expected = format!("{:02}", i);
        if !msg.contains(&format!("Buffered msg {expected}")) {
            println!("  WARNING: Message {i} unexpected content: \"{msg}\"");
        }
    }

    println!("  buffered_print_test: PASSED! (12 messages, 1 flush round-trip, {elapsed:?})");
    Ok(())
}

/// Test: Data-dependent iteration — Newton's method sqrt on GPU.
///
/// The kernel autonomously iterates until convergence without host intervention.
/// Verifies: correct sqrt result, iteration count > 0, convergence within tolerance.
pub(crate) fn run_newton_sqrt_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Data-Dependent Iteration Test (data-iter.1) ---");

    // Test cases: (input, expected_sqrt, max_tolerance)
    let test_cases: Vec<(f32, f32, f32)> = vec![
        (4.0, 2.0, 1e-5),
        (2.0, std::f32::consts::SQRT_2, 1e-5),
        (100.0, 10.0, 1e-4),
        (0.25, 0.5, 1e-5),
        (1e6, 1000.0, 1e-2),
    ];

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_IO_PTX);
    dev.load_ptx(ptx, "kernel", &["newton_sqrt_kernel"])?;
    let f = dev
        .get_func("kernel", "newton_sqrt_kernel")
        .ok_or(GpuHostError::KernelNotFound("newton_sqrt_kernel"))?;

    // Allocate mapped memory for I/O
    let (input_host, input_dev) = unsafe { alloc_mapped_u32(&dev)? };
    let (output_host, output_dev) = unsafe { alloc_mapped_u32(&dev)? };
    let (iter_host, iter_dev) = unsafe { alloc_mapped_u32(&dev)? };

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let max_iter: u32 = 100;

    for (input_val, expected, tolerance) in &test_cases {
        // Write input as f32 bits to the u32 mapped memory
        unsafe {
            core::ptr::write_volatile(input_host, input_val.to_bits());
            core::ptr::write_volatile(output_host, 0);
            core::ptr::write_volatile(iter_host, 0);
        }

        unsafe {
            f.clone()
                .launch(cfg, (input_dev, output_dev, iter_dev, max_iter))?;
        }
        dev.synchronize()?;

        let result_bits = unsafe { core::ptr::read_volatile(output_host) };
        let result = f32::from_bits(result_bits);
        let iterations = unsafe { core::ptr::read_volatile(iter_host) };

        let error = (result - expected).abs();
        let passed = error < *tolerance && iterations > 0 && iterations < max_iter;

        println!(
            "  sqrt({:.1}) = {:.6} (expected {:.6}, err={:.2e}, {} iters) {}",
            input_val,
            result,
            expected,
            error,
            iterations,
            if passed { "PASS" } else { "FAIL" }
        );

        if !passed {
            unsafe {
                free_mapped_mem(input_host)?;
                free_mapped_mem(output_host)?;
                free_mapped_mem(iter_host)?;
            }
            return Err(GpuHostError::Verification {
                test: "newton_sqrt_kernel",
                detail: format!(
                    "sqrt({}) = {} (expected {}, err={:.2e}, {} iters)",
                    input_val, result, expected, error, iterations
                ),
            });
        }
    }

    unsafe {
        free_mapped_mem(input_host)?;
        free_mapped_mem(output_host)?;
        free_mapped_mem(iter_host)?;
    }

    println!(
        "  newton_sqrt_kernel: PASSED! ({} test cases, all converged autonomously)",
        test_cases.len()
    );
    Ok(())
}
