//! Hostcall + Embassy + Async hostcall tests.

use std::sync::Arc;

use cudarc::driver::sys::{self, lib as cuda_lib};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

use gpu_host::error::{GpuHostError, Result};
use gpu_host::hostcall;
use gpu_host::mapped_mem::{alloc_mapped_result_array, alloc_mapped_u32, free_mapped_mem};

/// Step 8: Hostcall print test (hostcall.4).
pub(crate) fn run_hostcall_print_hello(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Step 8: Hostcall print (single message) ---");

    use std::sync::{Arc as StdArc, Mutex};

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();

    println!(
        "  Hostcall buffer allocated: {} bytes, {} packets",
        hc_buf.size(),
        hc_buf.num_packets()
    );
    println!(
        "  Host ptr: {:p}, Device ptr: 0x{:016X}",
        hc_buf.host_ptr(),
        dev_ptr
    );

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_u32(&dev)? };
    unsafe { std::ptr::write_volatile(result_host_ptr, 0u32) };

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] Received from GPU: \"{s}\"");
            messages_clone.lock().unwrap().push(s);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_IO_PTX);
    dev.load_ptx(ptx, "kernel", &["hostcall_print_hello"])?;
    let f = dev
        .get_func("kernel", "hostcall_print_hello")
        .ok_or(GpuHostError::KernelNotFound("hostcall_print_hello"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching hostcall_print_hello kernel...");
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    println!("  Kernel completed.");

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let result_val = unsafe { std::ptr::read_volatile(result_host_ptr) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();
    if result_val != 1 {
        return Err(GpuHostError::Verification {
            test: "hostcall_print_hello",
            detail: format!("kernel reported failure (result={result_val})"),
        });
    }
    if received.len() != 1 || !received[0].contains("Hello from GPU!") {
        return Err(GpuHostError::Verification {
            test: "hostcall_print_hello",
            detail: format!("unexpected messages: {:?}", *received),
        });
    }

    println!("  hostcall_print_hello: PASSED!");
    println!("    GPU sent \"{}\" via hostcall protocol", received[0]);
    println!("    Host listener received and printed it correctly");
    Ok(())
}

/// Step 9: Multi-warp hostcall print test.
pub(crate) fn run_hostcall_print_multi(dev: Arc<CudaDevice>, num_blocks: u32) -> Result<()> {
    println!("\n--- Step 9: Hostcall print (multi-block, {num_blocks} blocks) ---");

    use std::sync::{Arc as StdArc, Mutex};

    let num_packets = (num_blocks as u16).max(8);
    let hc_buf = hostcall::HostcallBuffer::new(num_packets)?;
    let dev_ptr = hc_buf.dev_ptr();

    println!(
        "  Hostcall buffer: {} packets, {} bytes",
        num_packets,
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
            println!("  [HOST] GPU says: \"{s}\"");
            messages_clone.lock().unwrap().push(s);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_IO_PTX);
    dev.load_ptx(ptx, "kernel", &["hostcall_print_multi"])?;
    let f = dev
        .get_func("kernel", "hostcall_print_multi")
        .ok_or(GpuHostError::KernelNotFound("hostcall_print_multi"))?;

    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching hostcall_print_multi ({num_blocks} blocks × 32 threads)...");
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
            test: "hostcall_print_multi",
            detail: format!("expected {num_blocks} successes, got {success_count}"),
        });
    }
    if received.len() != num_blocks as usize {
        return Err(GpuHostError::Verification {
            test: "hostcall_print_multi",
            detail: format!("expected {} messages, got {}", num_blocks, received.len()),
        });
    }

    println!("  hostcall_print_multi: PASSED!");
    println!("    {num_blocks} concurrent warps printed via hostcall successfully");
    Ok(())
}

/// Test: ImmediateFuture — single spawn + poll, writes 1 on success.
pub(crate) fn run_embassy_immediate(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Embassy Test 1: ImmediateFuture (spawn + single poll) ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::EMBASSY_PTX);
    dev.load_ptx(ptx, "embassy", &["embassy_test_kernel"])?;

    let f = dev
        .get_func("embassy", "embassy_test_kernel")
        .ok_or(GpuHostError::KernelNotFound("embassy_test_kernel"))?;

    let mut result: CudaSlice<u32> = dev.alloc_zeros::<u32>(1)?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        f.launch(cfg, (&mut result,))?;
    }

    let host_result = dev.dtoh_sync_copy(&result)?;
    if host_result[0] != 1 {
        return Err(GpuHostError::Verification {
            test: "embassy_test_kernel",
            detail: format!("expected result=1, got {}", host_result[0]),
        });
    }

    println!("  embassy_test_kernel: PASSED (ImmediateFuture polled to Ready, result=1)");
    Ok(())
}

/// Test: CountdownFuture — multi-poll async task.
pub(crate) fn run_embassy_countdown(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Embassy Test 2: CountdownFuture (multi-poll) ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::EMBASSY_PTX);
    dev.load_ptx(ptx, "embassy", &["embassy_countdown_kernel"])?;

    let f = dev
        .get_func("embassy", "embassy_countdown_kernel")
        .ok_or(GpuHostError::KernelNotFound("embassy_countdown_kernel"))?;

    let mut result: CudaSlice<u32> = dev.alloc_zeros::<u32>(2)?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        f.launch(cfg, (&mut result,))?;
    }

    let host_result = dev.dtoh_sync_copy(&result)?;
    let poll_rounds = host_result[0];
    let success = host_result[1];

    println!("  Poll rounds executed: {poll_rounds}");

    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "embassy_countdown_kernel",
            detail: format!("success marker not set (got {success})"),
        });
    }

    println!(
        "  embassy_countdown_kernel: PASSED (CountdownFuture completed after {poll_rounds} poll rounds)"
    );
    Ok(())
}

/// Test: Two concurrent tasks on the same executor.
pub(crate) fn run_embassy_two_task(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Embassy Test 3: Two concurrent tasks ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::EMBASSY_PTX);
    dev.load_ptx(ptx, "embassy", &["embassy_two_task_kernel"])?;

    let f = dev
        .get_func("embassy", "embassy_two_task_kernel")
        .ok_or(GpuHostError::KernelNotFound("embassy_two_task_kernel"))?;

    let mut result: CudaSlice<u32> = dev.alloc_zeros::<u32>(2)?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        f.launch(cfg, (&mut result,))?;
    }

    let host_result = dev.dtoh_sync_copy(&result)?;
    let poll_rounds = host_result[0];
    let success = host_result[1];

    println!("  Poll rounds executed: {poll_rounds}");

    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "embassy_two_task_kernel",
            detail: format!("success marker not set (got {success})"),
        });
    }

    println!(
        "  embassy_two_task_kernel: PASSED (2 concurrent tasks completed after {poll_rounds} poll rounds)"
    );
    Ok(())
}

/// Test: Synchronous countdown baseline for register pressure comparison.
pub(crate) fn run_sync_countdown(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Embassy Test 4: Sync countdown baseline ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::EMBASSY_PTX);
    dev.load_ptx(ptx, "embassy", &["sync_countdown_kernel"])?;

    let f = dev
        .get_func("embassy", "sync_countdown_kernel")
        .ok_or(GpuHostError::KernelNotFound("sync_countdown_kernel"))?;

    let mut result: CudaSlice<u32> = dev.alloc_zeros::<u32>(2)?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        f.launch(cfg, (&mut result,))?;
    }

    let host_result = dev.dtoh_sync_copy(&result)?;
    let value = host_result[0];
    let success = host_result[1];

    if value != 42 || success != 1 {
        return Err(GpuHostError::Verification {
            test: "sync_countdown_kernel",
            detail: format!("expected (42, 1), got ({value}, {success})"),
        });
    }

    println!("  sync_countdown_kernel: PASSED (result=42, sync loop completed)");
    Ok(())
}

/// File I/O round-trip test: open → write → close → open → read → close → verify.
pub(crate) fn run_hostcall_file_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Hostcall file I/O test (gpu-std.3) ---");

    use std::sync::Arc as StdArc;

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();

    println!(
        "  Hostcall buffer allocated: {} bytes, {} packets",
        hc_buf.size(),
        hc_buf.num_packets()
    );

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
    dev.load_ptx(ptx, "kernel", &["hostcall_file_test"])?;
    let f = dev
        .get_func("kernel", "hostcall_file_test")
        .ok_or(GpuHostError::KernelNotFound("hostcall_file_test"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    let start = std::time::Instant::now();

    println!("  Launching hostcall_file_test kernel...");
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("  Kernel completed in {elapsed:?}.");

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let overall = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let bytes_written = unsafe { std::ptr::read_volatile(result_host_ptr.add(2)) };
    let bytes_read = unsafe { std::ptr::read_volatile(result_host_ptr.add(3)) };

    unsafe { free_mapped_mem(result_host_ptr)? };

    let file_content = std::fs::read_to_string("gpu_test_output.txt");
    if let Ok(content) = &file_content {
        println!("  File on disk: \"{}\"", content.trim_end());
    }
    let _ = std::fs::remove_file("gpu_test_output.txt");

    if overall != 1 {
        return Err(GpuHostError::Verification {
            test: "hostcall_file_test",
            detail: format!("overall={overall}, written={bytes_written}, read={bytes_read}"),
        });
    }

    println!(
        "  hostcall_file_test: PASSED! (wrote {bytes_written} bytes, read {bytes_read} bytes, {elapsed:?})"
    );
    Ok(())
}

/// Test: single async hostcall print via HostcallFuture + Embassy executor.
pub(crate) fn run_async_hostcall_single(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Integration Test 1: Async hostcall (single HostcallFuture) ---");

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

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::ASYNC_HOSTCALL_PTX);
    dev.load_ptx(ptx, "async_hostcall", &["async_hostcall_single_kernel"])?;
    let f = dev
        .get_func("async_hostcall", "async_hostcall_single_kernel")
        .ok_or(GpuHostError::KernelNotFound("async_hostcall_single_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching async_hostcall_single_kernel...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("  Kernel completed in {elapsed:?}.");

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let poll_rounds = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();
    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "async_hostcall_single_kernel",
            detail: format!("success marker not set (got {success}), poll_rounds={poll_rounds}"),
        });
    }
    if received.is_empty() || !received[0].contains("Async hello") {
        return Err(GpuHostError::Verification {
            test: "async_hostcall_single_kernel",
            detail: format!("unexpected messages: {:?}", *received),
        });
    }

    println!("  async_hostcall_single_kernel: PASSED!");
    println!("    Poll rounds: {poll_rounds}");
    println!("    Message: \"{}\"", received[0]);
    println!("    HostcallFuture correctly yielded and resumed across poll rounds");
    Ok(())
}

/// Test: two concurrent async hostcall prints via HostcallFuture + Embassy executor.
pub(crate) fn run_async_hostcall_two(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Integration Test 2: Async hostcall (two concurrent HostcallFutures) ---");

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

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::ASYNC_HOSTCALL_PTX);
    dev.load_ptx(ptx, "async_hostcall", &["async_hostcall_two_kernel"])?;
    let f = dev
        .get_func("async_hostcall", "async_hostcall_two_kernel")
        .ok_or(GpuHostError::KernelNotFound("async_hostcall_two_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching async_hostcall_two_kernel...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("  Kernel completed in {elapsed:?}.");

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let poll_rounds = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();
    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "async_hostcall_two_kernel",
            detail: format!("success marker not set (got {success}), poll_rounds={poll_rounds}"),
        });
    }
    if received.len() < 2 {
        return Err(GpuHostError::Verification {
            test: "async_hostcall_two_kernel",
            detail: format!(
                "expected 2 messages, got {}: {:?}",
                received.len(),
                *received
            ),
        });
    }

    println!("  async_hostcall_two_kernel: PASSED!");
    println!("    Poll rounds: {poll_rounds}");
    println!("    Messages: {:?}", *received);
    println!("    Two HostcallFutures ran concurrently on one Embassy executor!");
    Ok(())
}

/// Test: futures_util::future::join — third-party async combinator on GPU.
pub(crate) fn run_futures_join(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Integration Test 3: futures_util::future::join on GPU ---");

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

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::ASYNC_HOSTCALL_PTX);
    dev.load_ptx(ptx, "async_hostcall", &["futures_join_kernel"])?;
    let f = dev
        .get_func("async_hostcall", "futures_join_kernel")
        .ok_or(GpuHostError::KernelNotFound("futures_join_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching futures_join_kernel...");
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("  Kernel completed in {elapsed:?}.");

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let poll_rounds = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();
    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "futures_join_kernel",
            detail: format!("success marker not set (got {success}), poll_rounds={poll_rounds}"),
        });
    }
    if received.len() < 2 {
        return Err(GpuHostError::Verification {
            test: "futures_join_kernel",
            detail: format!(
                "expected 2 messages, got {}: {:?}",
                received.len(),
                *received
            ),
        });
    }

    println!("  futures_join_kernel: PASSED!");
    println!("    Poll rounds: {poll_rounds}");
    println!("    Messages: {:?}", *received);
    println!("    futures_util::future::join works on GPU! Third-party async crate confirmed.");
    Ok(())
}

/// Test: GPU %globaltimer + wall-clock time via hostcall.
pub(crate) fn run_hostcall_time_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- GPU-std Test: Instant (%globaltimer) + SystemTime (hostcall) ---");

    use std::sync::Arc as StdArc;

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();

    let (result_host_ptr, result_dev_ptr) = unsafe {
        let cu = cuda_lib();
        let size = 4 * std::mem::size_of::<u64>();

        let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;
        let result = cu.cuMemHostAlloc(&mut host_ptr, size, flags);
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(GpuHostError::CudaAlloc(result));
        }

        let mut d_ptr: sys::CUdeviceptr = 0;
        let result = cu.cuMemHostGetDevicePointer_v2(&mut d_ptr, host_ptr, 0);
        if result != sys::CUresult::CUDA_SUCCESS {
            cu.cuMemFreeHost(host_ptr);
            return Err(GpuHostError::CudaGetDevPtr(result));
        }

        std::ptr::write_bytes(host_ptr as *mut u8, 0, size);
        (host_ptr as *mut u64, d_ptr)
    };

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] Print from GPU: \"{s}\"");
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_IO_PTX);
    dev.load_ptx(ptx, "kernel", &["hostcall_stdin_time_test"])?;
    let f = dev
        .get_func("kernel", "hostcall_stdin_time_test")
        .ok_or(GpuHostError::KernelNotFound("hostcall_stdin_time_test"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    // Skip stdin (skip_stdin=1) to avoid blocking
    println!("  Launching hostcall_stdin_time_test (skip_stdin=1)...");
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr, 1u32))?;
    }

    dev.synchronize()?;
    println!("  Kernel completed.");

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let gpu_instant_delta = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let host_secs = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    let host_nanos = unsafe { std::ptr::read_volatile(result_host_ptr.add(2)) };

    unsafe {
        let cu = cuda_lib();
        cu.cuMemFreeHost(result_host_ptr as *mut std::ffi::c_void);
    }

    println!("  GPU Instant delta: {gpu_instant_delta} ns (time for 1000 add iterations)");
    println!("  Host wall-clock: epoch_secs={host_secs}, nanos={host_nanos}");

    if gpu_instant_delta == 0 {
        return Err(GpuHostError::Verification {
            test: "hostcall_stdin_time_test",
            detail: "GPU instant delta is 0 — %globaltimer may not be working".to_string(),
        });
    }
    if host_secs < 1700000000 {
        return Err(GpuHostError::Verification {
            test: "hostcall_stdin_time_test",
            detail: format!("host secs={host_secs} seems too low for epoch time"),
        });
    }

    println!("  hostcall_stdin_time_test: PASSED!");
    println!("    GPU %globaltimer works (non-zero delta)");
    println!("    Host SystemTime works (reasonable epoch seconds)");
    Ok(())
}

/// Test: 32 threads each emit a gpu_trace!() event, verify all 32 complete.
///
/// Trace events are printed to stderr by the host-side handler. We verify
/// that all 32 threads ran to completion via an atomic success counter.
pub(crate) fn run_trace_multithread_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Trace multi-thread test (32 threads) ---");

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Io,
        "kernel_trace_multithread",
        &["trace_multithread_test"],
    )?;
    let f = dev
        .get_func("kernel_trace_multithread", "trace_multithread_test")
        .ok_or(GpuHostError::KernelNotFound("trace_multithread_test"))?;

    let num_packets = 64u16; // Enough for 32 threads
    let hc_buf = hostcall::HostcallBuffer::new(num_packets)?;
    let dev_ptr = hc_buf.dev_ptr();

    let (count_host_ptr, count_dev_ptr) = unsafe { alloc_mapped_u32(&dev)? };
    unsafe { std::ptr::write_volatile(count_host_ptr, 0u32) };

    let hc_buf_ref = std::sync::Arc::new(hc_buf);
    let listener = crate::harness_support::HostcallListener::start(
        std::sync::Arc::clone(&hc_buf_ref),
        |_msg| {
            // Trace events go to stderr via handle_trace, not through on_print
        },
    );

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching trace_multithread_test (1 block × 32 threads)...");
    unsafe {
        f.launch(cfg, (dev_ptr, count_dev_ptr))?;
    }

    dev.synchronize()?;
    println!("  Kernel completed.");

    std::thread::sleep(std::time::Duration::from_millis(200));
    listener.finish()?;

    let success_count = unsafe { std::ptr::read_volatile(count_host_ptr) };
    unsafe { free_mapped_mem(count_host_ptr)? };

    println!("  Results: {success_count}/32 threads completed trace events");

    if success_count != 32 {
        return Err(GpuHostError::Verification {
            test: "trace_multithread_test",
            detail: format!("expected 32 successes, got {success_count}"),
        });
    }

    println!("  trace_multithread_test: PASSED!");
    Ok(())
}

/// Test: threads trace + assert (true condition). Verify assert does not trap.
pub(crate) fn run_trace_assert_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Trace + assert test (32 threads, assert true) ---");

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Io,
        "kernel_trace_assert",
        &["trace_assert_test"],
    )?;
    let f = dev
        .get_func("kernel_trace_assert", "trace_assert_test")
        .ok_or(GpuHostError::KernelNotFound("trace_assert_test"))?;

    let num_packets = 64u16;
    let hc_buf = hostcall::HostcallBuffer::new(num_packets)?;
    let dev_ptr = hc_buf.dev_ptr();

    let (count_host_ptr, count_dev_ptr) = unsafe { alloc_mapped_u32(&dev)? };
    unsafe { std::ptr::write_volatile(count_host_ptr, 0u32) };

    let hc_buf_ref = std::sync::Arc::new(hc_buf);
    let listener = crate::harness_support::HostcallListener::start(
        std::sync::Arc::clone(&hc_buf_ref),
        |_msg| {},
    );

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching trace_assert_test (1 block × 32 threads)...");
    unsafe {
        f.launch(cfg, (dev_ptr, count_dev_ptr))?;
    }

    dev.synchronize()?;
    println!("  Kernel completed (no trap — assert passed).");

    std::thread::sleep(std::time::Duration::from_millis(200));
    listener.finish()?;

    let success_count = unsafe { std::ptr::read_volatile(count_host_ptr) };
    unsafe { free_mapped_mem(count_host_ptr)? };

    println!("  Results: {success_count}/32 threads completed");

    if success_count != 32 {
        return Err(GpuHostError::Verification {
            test: "trace_assert_test",
            detail: format!("expected 32 successes, got {success_count}"),
        });
    }

    println!("  trace_assert_test: PASSED!");
    Ok(())
}

/// Test: HostcallSession persists across two kernel launches.
///
/// 1. Start a HostcallSession
/// 2. Launch Kernel A — prints message, writes 0xCAFE to shared mapped memory
/// 3. Synchronize, reinit packets
/// 4. Launch Kernel B — reads 0xCAFE, prints message, writes result
/// 5. Verify Kernel B read the correct value
pub(crate) fn run_session_multi_launch_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- HostcallSession multi-launch test ---");

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Io,
        "kernel_session",
        &["session_kernel_a", "session_kernel_b"],
    )?;

    let session = hostcall::HostcallSession::start(16).map_err(|e| GpuHostError::Verification {
        test: "session_multi_launch",
        detail: format!("session start failed: {e}"),
    })?;

    // Shared state: mapped u32 for cross-launch communication
    let (state_host_ptr, state_dev_ptr) = unsafe { alloc_mapped_u32(&dev)? };
    unsafe { std::ptr::write_volatile(state_host_ptr, 0u32) };

    // Result: mapped u32 for Kernel B's verification
    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_u32(&dev)? };
    unsafe { std::ptr::write_volatile(result_host_ptr, 0u32) };

    // --- Launch Kernel A ---
    let f_a = dev
        .get_func("kernel_session", "session_kernel_a")
        .ok_or(GpuHostError::KernelNotFound("session_kernel_a"))?;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching session_kernel_a...");
    unsafe {
        f_a.launch(cfg, (session.dev_ptr(), state_dev_ptr))?;
    }
    dev.synchronize()?;
    println!("  Kernel A completed.");

    // Check that Kernel A wrote the magic value
    let state_after_a = unsafe { std::ptr::read_volatile(state_host_ptr) };
    println!("  Shared state after Kernel A: 0x{state_after_a:X}");

    // --- Reinit packets for Kernel B ---
    session.reinit_packets();
    println!("  Packet pool reinitialized.");

    // --- Launch Kernel B ---
    let f_b = dev
        .get_func("kernel_session", "session_kernel_b")
        .ok_or(GpuHostError::KernelNotFound("session_kernel_b"))?;

    println!("  Launching session_kernel_b...");
    unsafe {
        f_b.launch(cfg, (session.dev_ptr(), state_dev_ptr, result_dev_ptr))?;
    }
    dev.synchronize()?;
    println!("  Kernel B completed.");

    // Give listener time to process the last print
    std::thread::sleep(std::time::Duration::from_millis(100));

    // --- Shutdown session ---
    session.shutdown();

    // --- Verify results ---
    let result = unsafe { std::ptr::read_volatile(result_host_ptr) };
    println!("  Kernel B result: {result} (1 = magic matched)");

    unsafe {
        free_mapped_mem(state_host_ptr)?;
        free_mapped_mem(result_host_ptr)?;
    }

    if result != 1 {
        return Err(GpuHostError::Verification {
            test: "session_multi_launch",
            detail: format!("Kernel B did not read correct magic value, result={result}"),
        });
    }

    println!("  session_multi_launch: PASSED!");
    println!("    Two kernels shared same HostcallSession");
    println!("    Cross-launch state verified (0xCAFE)");
    Ok(())
}

/// Test: Multi-command kernel processes COMPUTE, PRINT, EXIT from command buffer.
///
/// 1. Start HostcallSession + CommandBuffer
/// 2. Launch multi_cmd_kernel (thread 0 polls command buffer)
/// 3. Host submits: COMPUTE (double 4 values) → PRINT → EXIT
/// 4. Kernel processes all three, then exits
/// 5. Verify COMPUTE results (doubled values)
pub(crate) fn run_multi_cmd_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Multi-command kernel test ---");

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Io,
        "kernel_multi_cmd",
        &["multi_cmd_kernel"],
    )?;

    let session = hostcall::HostcallSession::start(16).map_err(|e| GpuHostError::Verification {
        test: "multi_cmd",
        detail: format!("session start failed: {e}"),
    })?;

    let cmd_buf = hostcall::CommandBuffer::new(8).map_err(|e| GpuHostError::Verification {
        test: "multi_cmd",
        detail: format!("command buffer alloc failed: {e}"),
    })?;

    // Allocate input/output arrays (4 u32s each)
    let count = 4u32;
    let (input_host, input_dev) = unsafe { alloc_mapped_result_array(&dev, count as usize)? };
    let (output_host, output_dev) = unsafe { alloc_mapped_result_array(&dev, count as usize)? };

    // Initialize input: [10, 20, 30, 40]
    unsafe {
        for i in 0..count as isize {
            std::ptr::write_volatile(input_host.offset(i), (i as u32 + 1) * 10);
        }
    }

    let f = dev
        .get_func("kernel_multi_cmd", "multi_cmd_kernel")
        .ok_or(GpuHostError::KernelNotFound("multi_cmd_kernel"))?;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching multi_cmd_kernel...");
    unsafe {
        f.launch(cfg, (session.dev_ptr(), cmd_buf.dev_ptr()))?;
    }

    // Give kernel time to start polling
    std::thread::sleep(std::time::Duration::from_millis(50));

    // Submit COMPUTE command: double the 4 input values
    println!("  Submitting COMPUTE command (double 4 values)...");
    cmd_buf.submit(&hostcall::Command::Compute {
        input_ptr: input_dev,
        output_ptr: output_dev,
        count,
        op_code: 0,
    });

    // Submit PRINT command
    println!("  Submitting PRINT command...");
    cmd_buf.submit(&hostcall::Command::Print {
        msg: b"hello from command buffer".to_vec(),
    });

    // Submit EXIT command
    println!("  Submitting EXIT command...");
    cmd_buf.submit(&hostcall::Command::Exit);

    // Wait for kernel to finish
    dev.synchronize()?;
    println!("  Kernel completed.");

    // Give listener time to process prints
    std::thread::sleep(std::time::Duration::from_millis(100));

    session.shutdown();

    // Verify COMPUTE results: [20, 40, 60, 80]
    let mut pass = true;
    println!("  Verifying COMPUTE results:");
    for i in 0..count as isize {
        let val = unsafe { std::ptr::read_volatile(output_host.offset(i)) };
        let expected = ((i as u32 + 1) * 10) * 2;
        println!("    output[{i}] = {val} (expected {expected})");
        if val != expected {
            pass = false;
        }
    }

    unsafe {
        free_mapped_mem(input_host)?;
        free_mapped_mem(output_host)?;
    }

    if !pass {
        return Err(GpuHostError::Verification {
            test: "multi_cmd",
            detail: "COMPUTE output mismatch".to_string(),
        });
    }

    println!("  multi_cmd: PASSED!");
    println!("    COMPUTE + PRINT + EXIT processed in single kernel launch");
    Ok(())
}

/// Test: Cross-launch pipeline — Kernel A writes data, Kernel B reads it.
///
/// Zero host-side copy: both kernels operate on the same mapped memory buffer.
/// HostcallSession persists across both launches.
pub(crate) fn run_cross_pipeline_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Cross-launch pipeline test ---");

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Io,
        "kernel_cross_pipeline",
        &["pipeline_writer_kernel", "pipeline_reader_kernel"],
    )?;

    let session = hostcall::HostcallSession::start(16).map_err(|e| GpuHostError::Verification {
        test: "cross_pipeline",
        detail: format!("session start failed: {e}"),
    })?;

    let count = 8u32;

    // Shared data buffer: Kernel A writes, Kernel B reads (zero host copy)
    let (data_host, data_dev) = unsafe { alloc_mapped_result_array(&dev, count as usize)? };
    // Result buffer: Kernel B writes final output
    let (result_host, result_dev) = unsafe { alloc_mapped_result_array(&dev, count as usize)? };

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // --- Stage 1: Writer kernel ---
    let f_writer = dev
        .get_func("kernel_cross_pipeline", "pipeline_writer_kernel")
        .ok_or(GpuHostError::KernelNotFound("pipeline_writer_kernel"))?;

    println!("  Stage 1: Launching pipeline_writer_kernel...");
    unsafe {
        f_writer.launch(cfg, (session.dev_ptr(), data_dev, count))?;
    }
    dev.synchronize()?;
    println!("  Writer completed.");

    // Verify writer output (peek — not a copy, just reading mapped memory)
    println!("  Data after writer:");
    for i in 0..count as isize {
        let val = unsafe { std::ptr::read_volatile(data_host.offset(i)) };
        println!("    data[{i}] = {val}");
    }

    // --- Reinit packets for Stage 2 ---
    session.reinit_packets();

    // --- Stage 2: Reader kernel ---
    let f_reader = dev
        .get_func("kernel_cross_pipeline", "pipeline_reader_kernel")
        .ok_or(GpuHostError::KernelNotFound("pipeline_reader_kernel"))?;

    println!("  Stage 2: Launching pipeline_reader_kernel...");
    unsafe {
        f_reader.launch(cfg, (session.dev_ptr(), data_dev, result_dev, count))?;
    }
    dev.synchronize()?;
    println!("  Reader completed.");

    std::thread::sleep(std::time::Duration::from_millis(100));
    session.shutdown();

    // Verify: result[i] = (i+1)*100*3 = (i+1)*300
    let mut pass = true;
    println!("  Verifying pipeline results:");
    for i in 0..count as isize {
        let val = unsafe { std::ptr::read_volatile(result_host.offset(i)) };
        let expected = (i as u32 + 1) * 300;
        println!("    result[{i}] = {val} (expected {expected})");
        if val != expected {
            pass = false;
        }
    }

    unsafe {
        free_mapped_mem(data_host)?;
        free_mapped_mem(result_host)?;
    }

    if !pass {
        return Err(GpuHostError::Verification {
            test: "cross_pipeline",
            detail: "Pipeline output mismatch".to_string(),
        });
    }

    println!("  cross_pipeline: PASSED!");
    println!("    Kernel A → Kernel B via shared device buffer, zero host copy");
    Ok(())
}

/// Test: Pipeline API — builder-style multi-stage kernel launch.
///
/// Uses Pipeline::new().stage().stage().run() to do the same
/// writer→reader pipeline as the manual test above.
pub(crate) fn run_pipeline_api_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Pipeline API test ---");

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Io,
        "kernel_pipeline_api",
        &["pipeline_writer_kernel", "pipeline_reader_kernel"],
    )?;

    let count = 4u32;

    // Allocate shared buffers
    let (data_host, data_dev) = unsafe { alloc_mapped_result_array(&dev, count as usize)? };
    let (result_host, result_dev) = unsafe { alloc_mapped_result_array(&dev, count as usize)? };

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let dev_a = Arc::clone(&dev);
    let dev_b = Arc::clone(&dev);

    println!("  Running 2-stage pipeline via Pipeline API...");
    hostcall::Pipeline::new(16)
        .map_err(|e| GpuHostError::Verification {
            test: "pipeline_api",
            detail: format!("pipeline create failed: {e}"),
        })?
        .stage(move |hc_ptr| {
            let f = dev_a
                .get_func("kernel_pipeline_api", "pipeline_writer_kernel")
                .ok_or(GpuHostError::KernelNotFound("pipeline_writer_kernel"))?;
            unsafe { f.launch(cfg, (hc_ptr, data_dev, count))? };
            dev_a.synchronize()?;
            println!("    Stage 1 (writer) completed.");
            Ok(())
        })
        .stage(move |hc_ptr| {
            let f = dev_b
                .get_func("kernel_pipeline_api", "pipeline_reader_kernel")
                .ok_or(GpuHostError::KernelNotFound("pipeline_reader_kernel"))?;
            unsafe { f.launch(cfg, (hc_ptr, data_dev, result_dev, count))? };
            dev_b.synchronize()?;
            println!("    Stage 2 (reader) completed.");
            Ok(())
        })
        .run()?;

    // Verify: result[i] = (i+1)*100*3
    let mut pass = true;
    println!("  Verifying pipeline results:");
    for i in 0..count as isize {
        let val = unsafe { std::ptr::read_volatile(result_host.offset(i)) };
        let expected = (i as u32 + 1) * 300;
        println!("    result[{i}] = {val} (expected {expected})");
        if val != expected {
            pass = false;
        }
    }

    unsafe {
        free_mapped_mem(data_host)?;
        free_mapped_mem(result_host)?;
    }

    if !pass {
        return Err(GpuHostError::Verification {
            test: "pipeline_api",
            detail: "Pipeline API output mismatch".to_string(),
        });
    }

    println!("  pipeline_api: PASSED!");
    println!("    Pipeline::new().stage(writer).stage(reader).run()");
    Ok(())
}

/// Test: Iterative convergence kernel — Newton's method sqrt with data-dependent iterations.
///
/// Tests with diverse inputs to verify iteration count varies with input value.
pub(crate) fn run_convergence_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Iterative convergence test ---");

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Io,
        "kernel_convergence",
        &["convergence_kernel"],
    )?;

    let session = hostcall::HostcallSession::start(16).map_err(|e| GpuHostError::Verification {
        test: "convergence",
        detail: format!("session start failed: {e}"),
    })?;

    // Test inputs: diverse values to show data-dependent iteration counts
    let inputs: Vec<u32> = vec![0, 1, 4, 9, 16, 100, 10000, 1000000];
    let count = inputs.len() as u32;

    let (input_host, input_dev) = unsafe { alloc_mapped_result_array(&dev, count as usize)? };
    let (output_host, output_dev) = unsafe { alloc_mapped_result_array(&dev, count as usize)? };
    let (iters_host, iters_dev) = unsafe { alloc_mapped_result_array(&dev, count as usize)? };

    // Write inputs
    for (i, &val) in inputs.iter().enumerate() {
        unsafe { std::ptr::write_volatile(input_host.add(i), val) };
    }

    let f = dev
        .get_func("kernel_convergence", "convergence_kernel")
        .ok_or(GpuHostError::KernelNotFound("convergence_kernel"))?;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching convergence_kernel...");
    unsafe {
        f.launch(
            cfg,
            (session.dev_ptr(), input_dev, output_dev, iters_dev, count),
        )?;
    }
    dev.synchronize()?;

    std::thread::sleep(std::time::Duration::from_millis(100));
    session.shutdown();

    // Expected: floor(sqrt(n))
    let expected_sqrt: Vec<u32> = vec![0, 1, 2, 3, 4, 10, 100, 1000];

    let mut pass = true;
    let mut saw_different_iters = false;
    let mut prev_iters = 0u32;

    println!("  Results:");
    for i in 0..count as usize {
        let out = unsafe { std::ptr::read_volatile(output_host.add(i)) };
        let it = unsafe { std::ptr::read_volatile(iters_host.add(i)) };
        let exp = expected_sqrt[i];
        let ok = if out == exp { "OK" } else { "FAIL" };
        println!(
            "    sqrt({}) = {} ({ok}), iterations = {it}",
            inputs[i], out
        );
        if out != exp {
            pass = false;
        }
        if i > 1 && it != prev_iters {
            saw_different_iters = true;
        }
        prev_iters = it;
    }

    unsafe {
        free_mapped_mem(input_host)?;
        free_mapped_mem(output_host)?;
        free_mapped_mem(iters_host)?;
    }

    if !pass {
        return Err(GpuHostError::Verification {
            test: "convergence",
            detail: "sqrt output mismatch".to_string(),
        });
    }

    if !saw_different_iters {
        return Err(GpuHostError::Verification {
            test: "convergence",
            detail: "iteration counts were all the same — not data-dependent".to_string(),
        });
    }

    println!("  convergence: PASSED!");
    println!(
        "    Data-dependent iteration verified (different inputs → different iteration counts)"
    );
    Ok(())
}

/// Test: Multi-step autonomous pipeline — Pipeline API + convergence + multiple datasets.
///
/// Two pipeline stages:
/// 1. Stage 1: process small dataset [4, 25, 144]
/// 2. Stage 2: process larger dataset [10000, 1000000, 99999999]
///
/// Demonstrates Pipeline API with data-dependent iteration across stages.
pub(crate) fn run_autonomous_pipeline_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Autonomous pipeline test ---");

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Io,
        "kernel_autonomous_pipeline",
        &["autonomous_pipeline_kernel"],
    )?;

    let dataset_a: Vec<u32> = vec![4, 25, 144];
    let dataset_b: Vec<u32> = vec![10000, 1000000, 99999999];
    let count_a = dataset_a.len() as u32;
    let count_b = dataset_b.len() as u32;

    // Allocate for dataset A
    let (in_a_host, in_a_dev) = unsafe { alloc_mapped_result_array(&dev, count_a as usize)? };
    let (out_a_host, out_a_dev) = unsafe { alloc_mapped_result_array(&dev, count_a as usize)? };
    let (iters_a_host, iters_a_dev) = unsafe { alloc_mapped_result_array(&dev, count_a as usize)? };
    let (total_a_host, total_a_dev) = unsafe { alloc_mapped_u32(&dev)? };

    // Allocate for dataset B
    let (in_b_host, in_b_dev) = unsafe { alloc_mapped_result_array(&dev, count_b as usize)? };
    let (out_b_host, out_b_dev) = unsafe { alloc_mapped_result_array(&dev, count_b as usize)? };
    let (iters_b_host, iters_b_dev) = unsafe { alloc_mapped_result_array(&dev, count_b as usize)? };
    let (total_b_host, total_b_dev) = unsafe { alloc_mapped_u32(&dev)? };

    // Write inputs
    for (i, &v) in dataset_a.iter().enumerate() {
        unsafe { std::ptr::write_volatile(in_a_host.add(i), v) };
    }
    for (i, &v) in dataset_b.iter().enumerate() {
        unsafe { std::ptr::write_volatile(in_b_host.add(i), v) };
    }

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let dev_a = Arc::clone(&dev);
    let dev_b = Arc::clone(&dev);

    println!("  Running 2-stage autonomous pipeline...");
    hostcall::Pipeline::new(16)
        .map_err(|e| GpuHostError::Verification {
            test: "autonomous_pipeline",
            detail: format!("pipeline create failed: {e}"),
        })?
        .stage(move |hc_ptr| {
            let f = dev_a
                .get_func("kernel_autonomous_pipeline", "autonomous_pipeline_kernel")
                .ok_or(GpuHostError::KernelNotFound("autonomous_pipeline_kernel"))?;
            unsafe {
                f.launch(
                    cfg,
                    (
                        hc_ptr,
                        in_a_dev,
                        out_a_dev,
                        iters_a_dev,
                        total_a_dev,
                        count_a,
                    ),
                )?;
            }
            dev_a.synchronize()?;
            println!("    Stage 1 completed (small dataset).");
            Ok(())
        })
        .stage(move |hc_ptr| {
            let f = dev_b
                .get_func("kernel_autonomous_pipeline", "autonomous_pipeline_kernel")
                .ok_or(GpuHostError::KernelNotFound("autonomous_pipeline_kernel"))?;
            unsafe {
                f.launch(
                    cfg,
                    (
                        hc_ptr,
                        in_b_dev,
                        out_b_dev,
                        iters_b_dev,
                        total_b_dev,
                        count_b,
                    ),
                )?;
            }
            dev_b.synchronize()?;
            println!("    Stage 2 completed (large dataset).");
            Ok(())
        })
        .run()?;

    // Verify results
    let expected_a: Vec<u32> = vec![2, 5, 12]; // floor(sqrt(4,25,144))
    let expected_b: Vec<u32> = vec![100, 1000, 9999]; // floor(sqrt(10000,1000000,99999999))

    let mut pass = true;
    println!("  Stage 1 results (small dataset):");
    let total_a = unsafe { std::ptr::read_volatile(total_a_host) };
    for i in 0..count_a as usize {
        let out = unsafe { std::ptr::read_volatile(out_a_host.add(i)) };
        let it = unsafe { std::ptr::read_volatile(iters_a_host.add(i)) };
        let exp = expected_a[i];
        let ok = if out == exp { "OK" } else { "FAIL" };
        println!("    sqrt({}) = {} ({ok}), iters = {it}", dataset_a[i], out);
        if out != exp {
            pass = false;
        }
    }
    println!("    Total iterations: {total_a}");

    println!("  Stage 2 results (large dataset):");
    let total_b = unsafe { std::ptr::read_volatile(total_b_host) };
    for i in 0..count_b as usize {
        let out = unsafe { std::ptr::read_volatile(out_b_host.add(i)) };
        let it = unsafe { std::ptr::read_volatile(iters_b_host.add(i)) };
        let exp = expected_b[i];
        let ok = if out == exp { "OK" } else { "FAIL" };
        println!("    sqrt({}) = {} ({ok}), iters = {it}", dataset_b[i], out);
        if out != exp {
            pass = false;
        }
    }
    println!("    Total iterations: {total_b}");

    // Verify different total iterations between stages (data-dependent)
    println!("  Total iters: stage1={total_a}, stage2={total_b}");

    unsafe {
        free_mapped_mem(in_a_host)?;
        free_mapped_mem(out_a_host)?;
        free_mapped_mem(iters_a_host)?;
        free_mapped_mem(total_a_host)?;
        free_mapped_mem(in_b_host)?;
        free_mapped_mem(out_b_host)?;
        free_mapped_mem(iters_b_host)?;
        free_mapped_mem(total_b_host)?;
    }

    if !pass {
        return Err(GpuHostError::Verification {
            test: "autonomous_pipeline",
            detail: "autonomous pipeline output mismatch".to_string(),
        });
    }

    if total_a == total_b {
        return Err(GpuHostError::Verification {
            test: "autonomous_pipeline",
            detail: "stages had same total iterations — not data-dependent".to_string(),
        });
    }

    println!("  autonomous_pipeline: PASSED!");
    println!("    2-stage pipeline with data-dependent iteration");
    println!("    Different datasets → different iteration counts");
    Ok(())
}

/// Test: Flight recorder captures trace events and dumps them.
///
/// 1. Allocate FlightRecorder (capacity 8)
/// 2. Launch kernel that writes 5 events + sets crash flag
/// 3. Verify event count and crash flag
/// 4. Dump events to stderr
pub(crate) fn run_flight_recorder_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Flight recorder test ---");

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Io,
        "kernel_flight_recorder",
        &["flight_recorder_test"],
    )?;

    let session = hostcall::HostcallSession::start(16).map_err(|e| GpuHostError::Verification {
        test: "flight_recorder",
        detail: format!("session start failed: {e}"),
    })?;

    let fr = hostcall::FlightRecorder::new(8).map_err(|e| GpuHostError::Verification {
        test: "flight_recorder",
        detail: format!("flight recorder alloc failed: {e}"),
    })?;

    // should_crash = 1 to test the crash flag
    let (crash_host, crash_dev) = unsafe { alloc_mapped_u32(&dev)? };
    unsafe { std::ptr::write_volatile(crash_host, 1u32) };

    let f = dev
        .get_func("kernel_flight_recorder", "flight_recorder_test")
        .ok_or(GpuHostError::KernelNotFound("flight_recorder_test"))?;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching flight_recorder_test kernel...");
    unsafe {
        f.launch(cfg, (session.dev_ptr(), fr.dev_ptr(), crash_dev))?;
    }
    dev.synchronize()?;

    std::thread::sleep(std::time::Duration::from_millis(100));
    session.shutdown();

    // Check results
    let event_count = fr.write_count();
    let crashed = fr.crashed();
    println!("  Events recorded: {event_count}");
    println!("  Crash flag: {crashed}");

    // Dump events
    fr.dump();

    unsafe { free_mapped_mem(crash_host)? };

    if event_count != 5 {
        return Err(GpuHostError::Verification {
            test: "flight_recorder",
            detail: format!("expected 5 events, got {event_count}"),
        });
    }

    if !crashed {
        return Err(GpuHostError::Verification {
            test: "flight_recorder",
            detail: "crash flag not set".to_string(),
        });
    }

    println!("  flight_recorder: PASSED!");
    println!("    5 events captured, crash flag set, dump printed");
    Ok(())
}

/// warp-future-bridge.1: Standard `impl Future` print test (single thread).
pub(crate) fn run_std_future_print_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- std Future print test (warp-future-bridge.1) ---");

    use std::sync::{Arc as StdArc, Mutex};

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Io,
        "kernel_std_future_print",
        &["std_future_print_kernel"],
    )?;
    let f = dev
        .get_func("kernel_std_future_print", "std_future_print_kernel")
        .ok_or(GpuHostError::KernelNotFound("std_future_print_kernel"))?;

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_u32(&dev)? };
    unsafe { std::ptr::write_volatile(result_host_ptr, 0u32) };

    let hc_buf_ref = StdArc::new(hc_buf);
    let listener =
        crate::harness_support::HostcallListener::start(StdArc::clone(&hc_buf_ref), move |msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] Received: \"{s}\"");
            messages_clone.lock().unwrap().push(s);
        });

    // Single thread — baseline test, no warp cooperation yet
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching std_future_print_kernel (1 thread)...");
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    println!("  Kernel completed.");

    std::thread::sleep(std::time::Duration::from_millis(100));
    listener.finish()?;

    let result_val = unsafe { std::ptr::read_volatile(result_host_ptr) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();
    if result_val != 1 {
        return Err(GpuHostError::Verification {
            test: "std_future_print",
            detail: format!("kernel reported failure (result={result_val})"),
        });
    }
    if received.is_empty() || !received[0].contains("Hello from std Future!") {
        return Err(GpuHostError::Verification {
            test: "std_future_print",
            detail: format!("unexpected messages: {:?}", *received),
        });
    }

    println!("  std_future_print: PASSED!");
    println!("    Standard impl Future polled to completion on GPU");
    println!("    Message: \"{}\"", received[0]);
    Ok(())
}

/// warp-future-bridge.2: Warp-cooperative polling of standard impl Future.
///
/// All 32 lanes enter. Lane 0 polls GpuPrintFuture, broadcasts result
/// via shfl.sync. All lanes converge and write their lane_id.
pub(crate) fn run_warp_cooperative_future_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Warp-cooperative Future test (warp-future-bridge.2) ---");

    use std::sync::{Arc as StdArc, Mutex};

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Io,
        "kernel_warp_cooperative_future",
        &["warp_cooperative_future_kernel"],
    )?;
    let f = dev
        .get_func(
            "kernel_warp_cooperative_future",
            "warp_cooperative_future_kernel",
        )
        .ok_or(GpuHostError::KernelNotFound(
            "warp_cooperative_future_kernel",
        ))?;

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_u32(&dev)? };
    unsafe { std::ptr::write_volatile(result_host_ptr, 0u32) };

    // Allocate mapped memory for 32 lane results
    let (lane_results_host_ptr, lane_results_dev_ptr) =
        unsafe { alloc_mapped_result_array(&dev, 32)? };
    for i in 0..32 {
        unsafe { std::ptr::write_volatile(lane_results_host_ptr.add(i), 0xFFFF_FFFFu32) };
    }

    let hc_buf_ref = StdArc::new(hc_buf);
    let listener =
        crate::harness_support::HostcallListener::start(StdArc::clone(&hc_buf_ref), move |msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] Received: \"{s}\"");
            messages_clone.lock().unwrap().push(s);
        });

    // Launch with exactly 32 threads = 1 full warp
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching warp_cooperative_future_kernel (32 threads = 1 warp)...");
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr, lane_results_dev_ptr))?;
    }

    dev.synchronize()?;
    println!("  Kernel completed.");

    std::thread::sleep(std::time::Duration::from_millis(100));
    listener.finish()?;

    let result_val = unsafe { std::ptr::read_volatile(result_host_ptr) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    // Check that all 32 lanes wrote their lane_id
    let mut all_lanes_ok = true;
    for i in 0..32u32 {
        let lane_val = unsafe { std::ptr::read_volatile(lane_results_host_ptr.add(i as usize)) };
        if lane_val != i {
            println!("  FAIL: lane {i} wrote {lane_val}, expected {i}");
            all_lanes_ok = false;
        }
    }
    unsafe { free_mapped_mem(lane_results_host_ptr)? };

    let received = messages.lock().unwrap();

    if result_val != 1 {
        return Err(GpuHostError::Verification {
            test: "warp_cooperative_future",
            detail: format!("kernel reported failure (result={result_val})"),
        });
    }
    if !all_lanes_ok {
        return Err(GpuHostError::Verification {
            test: "warp_cooperative_future",
            detail: "not all 32 lanes wrote their lane_id".to_string(),
        });
    }
    if received.is_empty() || !received[0].contains("Hello from warp-cooperative Future!") {
        return Err(GpuHostError::Verification {
            test: "warp_cooperative_future",
            detail: format!("unexpected messages: {:?}", *received),
        });
    }

    println!("  warp_cooperative_future: PASSED!");
    println!("    Lane 0 polled standard impl Future");
    println!("    Result broadcast via shfl.sync to all 32 lanes");
    println!("    All 32 lanes converged and wrote their lane_id");
    println!("    Message: \"{}\"", received[0]);
    Ok(())
}

/// warp-future-bridge.3: Two sequential warp-cooperative Future polls.
///
/// Simulates two sequential `.await` points with full warp convergence.
pub(crate) fn run_warp_cooperative_two_futures_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Warp-cooperative two-futures test (warp-future-bridge.3) ---");

    use std::sync::{Arc as StdArc, Mutex};

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Io,
        "kernel_warp_cooperative_two_futures",
        &["warp_cooperative_two_futures_kernel"],
    )?;
    let f = dev
        .get_func(
            "kernel_warp_cooperative_two_futures",
            "warp_cooperative_two_futures_kernel",
        )
        .ok_or(GpuHostError::KernelNotFound(
            "warp_cooperative_two_futures_kernel",
        ))?;

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_u32(&dev)? };
    unsafe { std::ptr::write_volatile(result_host_ptr, 0u32) };

    let (lane_results_host_ptr, lane_results_dev_ptr) =
        unsafe { alloc_mapped_result_array(&dev, 32)? };
    for i in 0..32 {
        unsafe { std::ptr::write_volatile(lane_results_host_ptr.add(i), 0xFFFF_FFFFu32) };
    }

    let hc_buf_ref = StdArc::new(hc_buf);
    let listener =
        crate::harness_support::HostcallListener::start(StdArc::clone(&hc_buf_ref), move |msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] Received: \"{s}\"");
            messages_clone.lock().unwrap().push(s);
        });

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching warp_cooperative_two_futures_kernel (32 threads)...");
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr, lane_results_dev_ptr))?;
    }

    dev.synchronize()?;
    println!("  Kernel completed.");

    std::thread::sleep(std::time::Duration::from_millis(100));
    listener.finish()?;

    let result_val = unsafe { std::ptr::read_volatile(result_host_ptr) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let mut all_lanes_ok = true;
    for i in 0..32u32 {
        let lane_val = unsafe { std::ptr::read_volatile(lane_results_host_ptr.add(i as usize)) };
        if lane_val != i {
            println!("  FAIL: lane {i} wrote {lane_val}, expected {i}");
            all_lanes_ok = false;
        }
    }
    unsafe { free_mapped_mem(lane_results_host_ptr)? };

    let received = messages.lock().unwrap();

    if result_val != 2 {
        return Err(GpuHostError::Verification {
            test: "warp_cooperative_two_futures",
            detail: format!("expected result=2, got {result_val}"),
        });
    }
    if !all_lanes_ok {
        return Err(GpuHostError::Verification {
            test: "warp_cooperative_two_futures",
            detail: "not all 32 lanes converged".to_string(),
        });
    }
    if received.len() < 2 {
        return Err(GpuHostError::Verification {
            test: "warp_cooperative_two_futures",
            detail: format!(
                "expected 2 messages, got {}: {:?}",
                received.len(),
                *received
            ),
        });
    }

    println!("  warp_cooperative_two_futures: PASSED!");
    println!("    Two sequential .await points with full warp convergence");
    println!("    All 32 lanes converged at both points");
    println!("    Messages: {:?}", *received);
    Ok(())
}

/// warp-future-bridge.4: Warp-cooperative Result<T, E> broadcasting with ? semantics.
pub(crate) fn run_warp_result_future_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Warp-cooperative Result future test (warp-future-bridge.4) ---");

    use std::sync::{Arc as StdArc, Mutex};

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Io,
        "kernel_warp_result_future",
        &["warp_result_future_kernel"],
    )?;
    let f = dev
        .get_func("kernel_warp_result_future", "warp_result_future_kernel")
        .ok_or(GpuHostError::KernelNotFound("warp_result_future_kernel"))?;

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_u32(&dev)? };
    unsafe { std::ptr::write_volatile(result_host_ptr, 0u32) };

    let (lane_results_host_ptr, lane_results_dev_ptr) =
        unsafe { alloc_mapped_result_array(&dev, 32)? };
    for i in 0..32 {
        unsafe { std::ptr::write_volatile(lane_results_host_ptr.add(i), 0xFFFF_FFFFu32) };
    }

    let hc_buf_ref = StdArc::new(hc_buf);
    let listener =
        crate::harness_support::HostcallListener::start(StdArc::clone(&hc_buf_ref), move |msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] Received: \"{s}\"");
            messages_clone.lock().unwrap().push(s);
        });

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching warp_result_future_kernel (32 threads)...");
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr, lane_results_dev_ptr))?;
    }

    dev.synchronize()?;
    println!("  Kernel completed.");

    std::thread::sleep(std::time::Duration::from_millis(100));
    listener.finish()?;

    let result_val = unsafe { std::ptr::read_volatile(result_host_ptr) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let mut all_lanes_ok = true;
    for i in 0..32u32 {
        let lane_val = unsafe { std::ptr::read_volatile(lane_results_host_ptr.add(i as usize)) };
        if lane_val != i {
            println!("  FAIL: lane {i} wrote {lane_val}, expected {i}");
            all_lanes_ok = false;
        }
    }
    unsafe { free_mapped_mem(lane_results_host_ptr)? };

    let received = messages.lock().unwrap();

    // result should be 2 (both Ok) — if error, high bit is set
    if result_val != 2 {
        return Err(GpuHostError::Verification {
            test: "warp_result_future",
            detail: format!(
                "expected result=2, got 0x{:08X} ({})",
                result_val,
                if result_val & 0x8000_0000 != 0 {
                    format!("Err({})", result_val & 0x7FFF_FFFF)
                } else {
                    format!("Ok({result_val})")
                }
            ),
        });
    }
    if !all_lanes_ok {
        return Err(GpuHostError::Verification {
            test: "warp_result_future",
            detail: "not all 32 lanes converged".to_string(),
        });
    }
    if received.len() < 2 {
        return Err(GpuHostError::Verification {
            test: "warp_result_future",
            detail: format!(
                "expected 2 messages, got {}: {:?}",
                received.len(),
                *received
            ),
        });
    }

    println!("  warp_result_future: PASSED!");
    println!("    Result<T, E> broadcasting with ? semantics");
    println!("    Both futures returned Ok, all 32 lanes converged");
    println!("    Messages: {:?}", *received);
    Ok(())
}

/// warp-future-bridge.1: Two sequential std Future prints.
pub(crate) fn run_std_future_two_prints_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- std Future two-prints test (warp-future-bridge.1) ---");

    use std::sync::{Arc as StdArc, Mutex};

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Io,
        "kernel_std_future_two_prints",
        &["std_future_two_prints_kernel"],
    )?;
    let f = dev
        .get_func(
            "kernel_std_future_two_prints",
            "std_future_two_prints_kernel",
        )
        .ok_or(GpuHostError::KernelNotFound("std_future_two_prints_kernel"))?;

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_u32(&dev)? };
    unsafe { std::ptr::write_volatile(result_host_ptr, 0u32) };

    let hc_buf_ref = StdArc::new(hc_buf);
    let listener =
        crate::harness_support::HostcallListener::start(StdArc::clone(&hc_buf_ref), move |msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] Received: \"{s}\"");
            messages_clone.lock().unwrap().push(s);
        });

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching std_future_two_prints_kernel (1 thread)...");
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    println!("  Kernel completed.");

    std::thread::sleep(std::time::Duration::from_millis(100));
    listener.finish()?;

    let result_val = unsafe { std::ptr::read_volatile(result_host_ptr) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    let received = messages.lock().unwrap();
    if result_val != 2 {
        return Err(GpuHostError::Verification {
            test: "std_future_two_prints",
            detail: format!("expected result=2, got {result_val}"),
        });
    }
    if received.len() < 2 {
        return Err(GpuHostError::Verification {
            test: "std_future_two_prints",
            detail: format!(
                "expected 2 messages, got {}: {:?}",
                received.len(),
                *received
            ),
        });
    }

    println!("  std_future_two_prints: PASSED!");
    println!("    Two sequential impl Future polls completed on GPU");
    println!("    Messages: {:?}", *received);
    Ok(())
}
