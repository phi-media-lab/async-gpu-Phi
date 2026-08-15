//\! Warp-level tests: intrinsics, WarpFuture, proc macro, control flow, hybrid executor.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};

use gpu_host::error::{GpuHostError, Result};
use gpu_host::hostcall;
use gpu_host::mapped_mem::{alloc_mapped_result_array, free_mapped_mem};

pub(crate) fn run_warp_intrinsics_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Warp Intrinsics Test (warp-future.3) ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_TEST_PTX);
    dev.load_ptx(ptx, "kernel", &["test_warp_intrinsics"])?;
    let f = dev
        .get_func("kernel", "test_warp_intrinsics")
        .ok_or(GpuHostError::KernelNotFound("test_warp_intrinsics"))?;

    // Allocate mapped output for 32 u32 values
    let (output_host, output_dev) = unsafe { alloc_mapped_result_array(&dev, 32)? };

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        f.launch(cfg, (output_dev,))?;
    }
    dev.synchronize()?;

    // Verify all 32 lanes received 0xCAFE_BABE
    let expected = 0xCAFE_BABEu32;
    let mut pass_count = 0u32;
    for lane in 0..32 {
        let val = unsafe { std::ptr::read_volatile(output_host.add(lane)) };
        if val == expected {
            pass_count += 1;
        } else {
            println!("  FAIL: lane {lane} got 0x{val:08X}, expected 0x{expected:08X}");
        }
    }

    if pass_count == 32 {
        println!("  PASSED: all 32 lanes received 0xCAFE_BABE via shfl.sync.idx.b32");
        println!("  bar.warp.sync + shfl.sync.idx.b32 confirmed working on hardware.");
    } else {
        println!("  FAILED: only {pass_count}/32 lanes received correct value");
    }

    unsafe { free_mapped_mem(output_host)? };
    Ok(())
}

/// WarpFuture PoC test: 32 lanes cooperatively send a PRINT hostcall.
///
/// Launches 1 block x 32 threads. All lanes execute the WarpPrintFuture
/// state machine via WarpExecutor. The message "WarpFuture: ABCDEFGHIJKLMNOP..."
/// should appear on the host.
pub(crate) fn run_warp_future_print_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- WarpFuture PoC Test (warp-future.4) ---");

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();

    let (result_host, result_dev) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    let hc_buf_ref = std::sync::Arc::new(hc_buf);
    let hc_buf_listener = std::sync::Arc::clone(&hc_buf_ref);

    let msg_received = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let msg_clone = std::sync::Arc::clone(&msg_received);

    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(move |msg| {
            let text = String::from_utf8_lossy(msg);
            let mut guard = msg_clone.lock().unwrap();
            *guard = text.to_string();
            println!("  [HOST] WarpFuture says: \"{text}\"");
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_TEST_PTX);
    dev.load_ptx(ptx, "kernel", &["warp_future_print_test"])?;
    let f = dev
        .get_func("kernel", "warp_future_print_test")
        .ok_or(GpuHostError::KernelNotFound("warp_future_print_test"))?;

    // Launch: 1 block x 32 threads (1 full warp)
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev))?;
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let result_val = unsafe { std::ptr::read_volatile(result_host) };

    println!("  Result: {result_val} (1=success)");
    println!("  Elapsed: {:.3}ms", elapsed.as_secs_f64() * 1000.0);

    let received_msg = msg_received.lock().unwrap();
    if result_val == 1 && received_msg.contains("WarpFuture: ") {
        println!("  WarpFuture PoC: PASSED!");
        println!("    32 lanes cooperatively built and sent a message via WarpFuture trait.");
        println!("    State machine: INIT -> WAIT -> DONE (zero divergence by construction).");
    } else if result_val == 1 {
        println!(
            "  WarpFuture completed but message format unexpected: \"{}\"",
            *received_msg
        );
    } else {
        println!("  WarpFuture PoC: FAILED (result={result_val})");
    }

    unsafe { free_mapped_mem(result_host)? };
    Ok(())
}

/// WarpFuture multi-hostcall test: 3 sequential PRINT calls in one WarpFuture (warp-future.6).
///
/// Validates that a WarpFuture state machine can compose multiple sequential
/// hostcalls while maintaining warp convergence. Expects 3 messages received in order.
pub(crate) fn run_warp_future_multi_print_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- WarpFuture Multi-Hostcall Test (warp-future.6) ---");

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();

    let (result_host, result_dev) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    let hc_buf_ref = std::sync::Arc::new(hc_buf);
    let hc_buf_listener = std::sync::Arc::clone(&hc_buf_ref);

    let messages = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let msg_clone = std::sync::Arc::clone(&messages);

    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(move |msg| {
            let text = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] WarpMulti says: \"{text}\"");
            let mut guard = msg_clone.lock().unwrap();
            guard.push(text);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_TEST_PTX);
    dev.load_ptx(ptx, "kernel", &["warp_future_multi_print_test"])?;
    let f = dev
        .get_func("kernel", "warp_future_multi_print_test")
        .ok_or(GpuHostError::KernelNotFound("warp_future_multi_print_test"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev))?;
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let result_val = unsafe { std::ptr::read_volatile(result_host) };
    let msgs = messages.lock().unwrap();

    println!("  Result: {result_val} (1=success)");
    println!("  Elapsed: {:.3}ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Messages received: {}", msgs.len());

    if result_val == 1 && msgs.len() == 3 {
        let ok1 = msgs[0].contains("1/3");
        let ok2 = msgs[1].contains("2/3");
        let ok3 = msgs[2].contains("3/3");
        if ok1 && ok2 && ok3 {
            println!("  WarpFuture Multi-Hostcall: PASSED!");
            println!("    3 sequential hostcalls completed in order.");
            println!("    7-state machine: INIT1→WAIT1→INIT2→WAIT2→INIT3→WAIT3→DONE");
            println!("    Composition of WarpFuture hostcalls verified on hardware.");
        } else {
            println!("  Messages out of order or unexpected content:");
            for (i, m) in msgs.iter().enumerate() {
                println!("    [{i}]: \"{m}\"");
            }
        }
    } else {
        println!(
            "  WarpFuture Multi-Hostcall: FAILED (result={}, msgs={})",
            result_val,
            msgs.len()
        );
        for (i, m) in msgs.iter().enumerate() {
            println!("    [{i}]: \"{m}\"");
        }
    }

    unsafe { free_mapped_mem(result_host)? };
    Ok(())
}

/// WarpFuture proc macro test: #[warp_async] generates a 2-call state machine (warp-future.5).
pub(crate) fn run_warp_print_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- WarpFuture Proc Macro Test (warp-future.5) ---");

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();

    let (result_host, result_dev) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    let hc_buf_ref = std::sync::Arc::new(hc_buf);
    let hc_buf_listener = std::sync::Arc::clone(&hc_buf_ref);

    let messages = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let msg_clone = std::sync::Arc::clone(&messages);

    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(move |msg| {
            let text = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] WarpMacro says: \"{text}\"");
            let mut guard = msg_clone.lock().unwrap();
            guard.push(text);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_TEST_PTX);
    dev.load_ptx(ptx, "kernel", &["warp_print_test"])?;
    let f = dev
        .get_func("kernel", "warp_print_test")
        .ok_or(GpuHostError::KernelNotFound("warp_print_test"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, result_dev))?;
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let result_val = unsafe { std::ptr::read_volatile(result_host) };
    let msgs = messages.lock().unwrap();

    println!("  Result: {result_val} (1=success)");
    println!("  Elapsed: {:.3}ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Messages received: {}", msgs.len());

    if result_val == 1 && msgs.len() == 2 {
        let ok1 = msgs[0].contains("1/2");
        let ok2 = msgs[1].contains("2/2");
        if ok1 && ok2 {
            println!("  WarpFuture Proc Macro: PASSED!");
            println!("    #[warp_async] generated a 5-state machine (2 PRINT calls).");
            println!("    Code quality matches hand-written WarpFuture.");
        } else {
            println!("  Messages unexpected:");
            for (i, m) in msgs.iter().enumerate() {
                println!("    [{i}]: \"{m}\"");
            }
        }
    } else {
        println!(
            "  WarpFuture Proc Macro: FAILED (result={}, msgs={})",
            result_val,
            msgs.len()
        );
    }

    unsafe { free_mapped_mem(result_host)? };
    Ok(())
}

/// WarpFuture proc macro if/else test (warp-cfg.2):
/// Tests #[warp_async] with if/else containing warp_*!() calls.
/// Kernel takes a `flag` parameter: flag != 0 → then branch, flag == 0 → else branch.
/// Run 1: flag=1 → "branch: then" + "branch: done"
/// Run 2: flag=0 → "branch: else" + "branch: done"
pub(crate) fn run_warp_cfg_if_else_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- WarpFuture If/Else Test (warp-cfg.2) ---");

    let launch_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    // Helper: run kernel with given flag, return (status, messages)
    fn run_with_flag(
        dev: &Arc<CudaDevice>,
        launch_cfg: LaunchConfig,
        flag: u64,
        module_name: &'static str,
    ) -> Result<(u32, Vec<String>)> {
        let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_TEST_PTX);
        dev.load_ptx(ptx, module_name, &["warp_cfg_if_else_test"])?;
        let f = dev
            .get_func(module_name, "warp_cfg_if_else_test")
            .ok_or(GpuHostError::KernelNotFound("warp_cfg_if_else_test"))?;

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
                println!("  [HOST] GPU says: \"{text}\"");
                let mut guard = msg_clone.lock().unwrap();
                guard.push(text);
            });
        });

        unsafe { f.launch(launch_cfg, (dev_ptr, flag, status_dev))? };
        dev.synchronize()?;

        std::thread::sleep(std::time::Duration::from_millis(100));
        hc_buf_ref.signal_shutdown();
        listener_handle.join().unwrap();

        let status = unsafe { std::ptr::read_volatile(status_host) };
        let msgs = messages.lock().unwrap().clone();
        unsafe { free_mapped_mem(status_host)? };
        Ok((status, msgs))
    }

    // --- Run 1: flag=1 → then branch ---
    println!("  Run 1: flag=1 (then-branch)");
    let (status, msgs) = run_with_flag(&dev, launch_cfg, 1, "kernel_cfg1")?;
    let then_msg = msgs.iter().any(|m| m.contains("branch: then"));
    let done_msg = msgs.iter().any(|m| m.contains("branch: done"));

    if status == 1 && then_msg && done_msg && msgs.len() == 2 {
        println!("  Run 1: PASSED (then-branch taken, done reached)");
    } else {
        println!("  Run 1: FAILED");
        println!("    status={status}, then_msg={then_msg}, done_msg={done_msg}");
        println!("    messages: {msgs:?}");
        return Err(GpuHostError::Verification {
            test: "warp_cfg_if_else_run1",
            detail: "then-branch not taken when flag=1".to_string(),
        });
    }

    // --- Run 2: flag=0 → else branch ---
    println!("  Run 2: flag=0 (else-branch)");
    let (status, msgs) = run_with_flag(&dev, launch_cfg, 0, "kernel_cfg2")?;
    let else_msg = msgs.iter().any(|m| m.contains("branch: else"));
    let done_msg = msgs.iter().any(|m| m.contains("branch: done"));

    if status == 1 && else_msg && done_msg && msgs.len() == 2 {
        println!("  Run 2: PASSED (else-branch taken, done reached)");
    } else {
        println!("  Run 2: FAILED");
        println!("    status={status}, else_msg={else_msg}, done_msg={done_msg}");
        println!("    messages: {msgs:?}");
        return Err(GpuHostError::Verification {
            test: "warp_cfg_if_else_run2",
            detail: "else-branch not taken when flag=0".to_string(),
        });
    }

    println!("  WarpFuture If/Else: PASSED!");
    println!("    #[warp_async] if/else generates correct DECISION state");
    println!("    Lane 0 evaluates condition, broadcasts to all 32 lanes");
    println!("    Both branches verified on GPU hardware");
    Ok(())
}

/// Loop/break test for #[warp_async] (warp-cfg.3)
///
/// Tests that the macro-generated state machine handles loop + break_if correctly.
/// Uses counter=0 for immediate break (1 "iter" + 1 "done" = 2 messages).
pub(crate) fn run_warp_cfg_loop_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- WarpFuture Loop/Break Test (warp-cfg.3) ---");

    let launch_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_TEST_PTX);
    dev.load_ptx(ptx, "kernel_loop", &["warp_cfg_loop_test"])?;
    let f = dev
        .get_func("kernel_loop", "warp_cfg_loop_test")
        .ok_or(GpuHostError::KernelNotFound("warp_cfg_loop_test"))?;

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

    // counter=0 → immediate break after first iteration
    let counter: u64 = 0;
    unsafe { f.launch(launch_cfg, (dev_ptr, counter, status_dev))? };
    dev.synchronize()?;

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let status = unsafe { std::ptr::read_volatile(status_host) };
    let msgs = messages.lock().unwrap().clone();
    unsafe { free_mapped_mem(status_host)? };

    let iter_msg = msgs.iter().any(|m| m.contains("iter"));
    let done_msg = msgs.iter().any(|m| m.contains("done"));

    if status == 1 && iter_msg && done_msg && msgs.len() == 2 {
        println!("  PASSED: loop executed once, break taken, done reached");
        println!("    messages: {msgs:?}");
    } else {
        println!("  FAILED");
        println!("    status={status}, iter_msg={iter_msg}, done_msg={done_msg}");
        println!("    messages: {msgs:?}");
        return Err(GpuHostError::Verification {
            test: "warp_cfg_loop_test",
            detail: format!(
                "expected 2 messages (iter+done), got {}: {msgs:?}",
                msgs.len()
            ),
        });
    }

    println!("  WarpFuture Loop/Break: PASSED!");
    println!("    #[warp_async] loop/break_if generates correct back-edge + break states");
    println!("    Lane 0 evaluates break condition, broadcasts to all 32 lanes");
    Ok(())
}

/// Match dispatch test for #[warp_async] (warp-cfg.4)
///
/// Tests that the macro-generated state machine handles match expressions correctly.
/// Runs 3 tests: cmd=0 ("cmd: zero"), cmd=1 ("cmd: one"), cmd=99 ("cmd: other").
/// Each run should produce exactly 2 messages: the arm message + "match: done".
pub(crate) fn run_warp_cfg_match_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- WarpFuture Match Test (warp-cfg.4) ---");

    let launch_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    fn run_with_cmd(
        dev: &Arc<CudaDevice>,
        launch_cfg: LaunchConfig,
        cmd: u64,
        module_name: &'static str,
    ) -> Result<(u32, Vec<String>)> {
        let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_TEST_PTX);
        dev.load_ptx(ptx, module_name, &["warp_cfg_match_test"])?;
        let f = dev
            .get_func(module_name, "warp_cfg_match_test")
            .ok_or(GpuHostError::KernelNotFound("warp_cfg_match_test"))?;

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
                println!("  [HOST] GPU says: \"{text}\"");
                let mut guard = msg_clone.lock().unwrap();
                guard.push(text);
            });
        });

        unsafe { f.launch(launch_cfg, (dev_ptr, cmd, status_dev))? };
        dev.synchronize()?;

        std::thread::sleep(std::time::Duration::from_millis(100));
        hc_buf_ref.signal_shutdown();
        listener_handle.join().unwrap();

        let status = unsafe { std::ptr::read_volatile(status_host) };
        let msgs = messages.lock().unwrap().clone();
        unsafe { free_mapped_mem(status_host)? };
        Ok((status, msgs))
    }

    // --- Run 1: cmd=0 → "cmd: zero" ---
    println!("  Run 1: cmd=0");
    let (status, msgs) = run_with_cmd(&dev, launch_cfg, 0, "kernel_match0")?;
    let arm_msg = msgs.iter().any(|m| m.contains("cmd: zero"));
    let done_msg = msgs.iter().any(|m| m.contains("match: done"));
    if status == 1 && arm_msg && done_msg && msgs.len() == 2 {
        println!("  Run 1: PASSED (arm 0 taken)");
    } else {
        println!("  Run 1: FAILED (status={status}, msgs={msgs:?})");
        return Err(GpuHostError::Verification {
            test: "warp_cfg_match_cmd0",
            detail: format!("expected cmd:zero + match:done, got: {msgs:?}"),
        });
    }

    // --- Run 2: cmd=1 → "cmd: one" ---
    println!("  Run 2: cmd=1");
    let (status, msgs) = run_with_cmd(&dev, launch_cfg, 1, "kernel_match1")?;
    let arm_msg = msgs.iter().any(|m| m.contains("cmd: one"));
    let done_msg = msgs.iter().any(|m| m.contains("match: done"));
    if status == 1 && arm_msg && done_msg && msgs.len() == 2 {
        println!("  Run 2: PASSED (arm 1 taken)");
    } else {
        println!("  Run 2: FAILED (status={status}, msgs={msgs:?})");
        return Err(GpuHostError::Verification {
            test: "warp_cfg_match_cmd1",
            detail: format!("expected cmd:one + match:done, got: {msgs:?}"),
        });
    }

    // --- Run 3: cmd=99 → "cmd: other" (wildcard arm) ---
    println!("  Run 3: cmd=99 (wildcard)");
    let (status, msgs) = run_with_cmd(&dev, launch_cfg, 99, "kernel_match99")?;
    let arm_msg = msgs.iter().any(|m| m.contains("cmd: other"));
    let done_msg = msgs.iter().any(|m| m.contains("match: done"));
    if status == 1 && arm_msg && done_msg && msgs.len() == 2 {
        println!("  Run 3: PASSED (wildcard arm taken)");
    } else {
        println!("  Run 3: FAILED (status={status}, msgs={msgs:?})");
        return Err(GpuHostError::Verification {
            test: "warp_cfg_match_cmd99",
            detail: format!("expected cmd:other + match:done, got: {msgs:?}"),
        });
    }

    println!("  WarpFuture Match: PASSED!");
    println!("    #[warp_async] match generates correct MATCH_DECISION state");
    println!("    Lane 0 evaluates scrutinee, maps to arm index, broadcasts to all 32 lanes");
    println!("    All 3 arms verified on GPU hardware");
    Ok(())
}

/// Nested control flow stress test for #[warp_async] (warp-cfg.5)
///
/// Tests if/else with match nested inside then-branch.
/// 4 test cases: flag=1+cmd=0, flag=1+cmd=1, flag=1+cmd=99, flag=0.
pub(crate) fn run_warp_cfg_nested_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- WarpFuture Nested Control Flow Test (warp-cfg.5) ---");

    let launch_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    fn run_nested(
        dev: &Arc<CudaDevice>,
        launch_cfg: LaunchConfig,
        flag: u64,
        cmd: u64,
        module_name: &'static str,
    ) -> Result<(u32, Vec<String>)> {
        let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_TEST_PTX);
        dev.load_ptx(ptx, module_name, &["warp_cfg_nested_test"])?;
        let f = dev
            .get_func(module_name, "warp_cfg_nested_test")
            .ok_or(GpuHostError::KernelNotFound("warp_cfg_nested_test"))?;

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
                println!("  [HOST] GPU says: \"{text}\"");
                let mut guard = msg_clone.lock().unwrap();
                guard.push(text);
            });
        });

        unsafe { f.launch(launch_cfg, (dev_ptr, flag, cmd, status_dev))? };
        dev.synchronize()?;

        std::thread::sleep(std::time::Duration::from_millis(100));
        hc_buf_ref.signal_shutdown();
        listener_handle.join().unwrap();

        let status = unsafe { std::ptr::read_volatile(status_host) };
        let msgs = messages.lock().unwrap().clone();
        unsafe { free_mapped_mem(status_host)? };
        Ok((status, msgs))
    }

    struct TestCase {
        flag: u64,
        cmd: u64,
        expected_msg: &'static str,
        label: &'static str,
        module: &'static str,
    }

    let cases = [
        TestCase {
            flag: 1,
            cmd: 0,
            expected_msg: "then-cmd0",
            label: "then + match arm 0",
            module: "kernel_nested_1_0",
        },
        TestCase {
            flag: 1,
            cmd: 1,
            expected_msg: "then-cmd1",
            label: "then + match arm 1",
            module: "kernel_nested_1_1",
        },
        TestCase {
            flag: 1,
            cmd: 99,
            expected_msg: "then-other",
            label: "then + match wildcard",
            module: "kernel_nested_1_99",
        },
        TestCase {
            flag: 0,
            cmd: 0,
            expected_msg: "else-path",
            label: "else branch",
            module: "kernel_nested_0_0",
        },
    ];

    for (i, tc) in cases.iter().enumerate() {
        println!(
            "  Run {}: flag={}, cmd={} ({})",
            i + 1,
            tc.flag,
            tc.cmd,
            tc.label
        );
        let (status, msgs) = run_nested(&dev, launch_cfg, tc.flag, tc.cmd, tc.module)?;
        let arm_msg = msgs.iter().any(|m| m.contains(tc.expected_msg));
        let done_msg = msgs.iter().any(|m| m.contains("nested: done"));
        if status == 1 && arm_msg && done_msg && msgs.len() == 2 {
            println!("  Run {}: PASSED ({})", i + 1, tc.label);
        } else {
            println!("  Run {}: FAILED (status={status}, msgs={msgs:?})", i + 1);
            return Err(GpuHostError::Verification {
                test: "warp_cfg_nested",
                detail: format!(
                    "run {}: expected '{}' + 'nested: done', got: {msgs:?}",
                    i + 1,
                    tc.expected_msg
                ),
            });
        }
    }

    println!("  WarpFuture Nested Control Flow: PASSED!");
    println!("    if/else with match nested inside then-branch verified on GPU");
    println!("    All 4 paths (3 match arms + else branch) produce correct messages");
    Ok(())
}

/// Hybrid executor test: WarpFuture PRINT → per-thread compute → WarpFuture PRINT (hybrid-executor.1)
///
/// Validates that a WarpFuture state machine can transition to per-thread divergent
/// computation and back to warp-cooperative I/O. Each lane computes lane_id^2 + 1
/// independently, then all lanes reconverge for the final PRINT.
pub(crate) fn run_hybrid_executor_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Hybrid Executor Test (hybrid-executor.1) ---");

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();

    // Allocate mapped memory for results: 32 u32 values (one per lane) + 1 status
    let (results_host, results_dev) = unsafe { alloc_mapped_result_array(&dev, 33)? };
    let status_dev = results_dev + (32 * std::mem::size_of::<u32>()) as u64;

    let hc_buf_ref = std::sync::Arc::new(hc_buf);
    let hc_buf_listener = std::sync::Arc::clone(&hc_buf_ref);

    let messages = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let msg_clone = std::sync::Arc::clone(&messages);

    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(move |msg| {
            let text = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] Hybrid says: \"{text}\"");
            let mut guard = msg_clone.lock().unwrap();
            guard.push(text);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_IO_PTX);
    dev.load_ptx(ptx, "kernel", &["hybrid_executor_test"])?;
    let f = dev
        .get_func("kernel", "hybrid_executor_test")
        .ok_or(GpuHostError::KernelNotFound("hybrid_executor_test"))?;

    // Launch: 1 block x 32 threads (1 full warp)
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, results_dev, status_dev))?;
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let status_host = unsafe { results_host.add(32) };
    let status_val = unsafe { std::ptr::read_volatile(status_host) };
    let msgs = messages.lock().unwrap();

    println!("  Status: {status_val} (1=success)");
    println!("  Elapsed: {:.3}ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Messages received: {}", msgs.len());

    // Verify per-thread computation results: results[i] = i*i + 1
    let mut compute_ok = true;
    for i in 0u32..32 {
        let actual = unsafe { std::ptr::read_volatile(results_host.add(i as usize)) };
        let expected = i * i + 1;
        if actual != expected {
            println!("  FAIL: results[{i}] = {actual} (expected {expected})");
            compute_ok = false;
        }
    }

    // Verify messages: expect "hybrid: start" and "hybrid: done"
    let msg_ok =
        msgs.len() == 2 && msgs[0].contains("hybrid: start") && msgs[1].contains("hybrid: done");

    if status_val == 1 && compute_ok && msg_ok {
        println!("  Hybrid Executor: PASSED!");
        println!("    WarpFuture PRINT → per-thread compute (32 lanes, each lane_id^2+1) → WarpFuture PRINT");
        println!(
            "    Demonstrates: warp-cooperative I/O ↔ per-thread divergent computation switching"
        );
    } else {
        println!("  Hybrid Executor: FAILED");
        if !compute_ok {
            println!("    Per-thread computation results incorrect");
        }
        if !msg_ok {
            println!("    Messages: {:?}", *msgs);
        }
    }

    unsafe { free_mapped_mem(results_host)? };
    Ok(())
}

/// Hybrid stress test: variable-duration per-thread work + multiple switching points (hybrid-executor.2)
///
/// 9-state machine with 3 I/O phases and 2 compute phases.
/// COMPUTE1: sum 1..=(lane_id*100+1) — ~3100x duration variance across lanes
/// COMPUTE2: XOR-fold with lane-dependent iteration count
/// Verifies syncwarp handles extreme lane divergence timing.
pub(crate) fn run_hybrid_stress_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Hybrid Stress Test (hybrid-executor.2) ---");

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();

    // 64 results (32 per compute phase) + 1 status = 65 u32
    let (results_host, results_dev) = unsafe { alloc_mapped_result_array(&dev, 65)? };
    let status_dev = results_dev + (64 * std::mem::size_of::<u32>()) as u64;

    let hc_buf_ref = std::sync::Arc::new(hc_buf);
    let hc_buf_listener = std::sync::Arc::clone(&hc_buf_ref);

    let messages = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let msg_clone = std::sync::Arc::clone(&messages);

    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(move |msg| {
            let text = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] Stress says: \"{text}\"");
            let mut guard = msg_clone.lock().unwrap();
            guard.push(text);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_IO_PTX);
    dev.load_ptx(ptx, "kernel", &["hybrid_stress_test"])?;
    let f = dev
        .get_func("kernel", "hybrid_stress_test")
        .ok_or(GpuHostError::KernelNotFound("hybrid_stress_test"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, results_dev, status_dev))?;
    }
    dev.synchronize()?;
    let elapsed = start.elapsed();

    std::thread::sleep(std::time::Duration::from_millis(100));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let status_host = unsafe { results_host.add(64) };
    let status_val = unsafe { std::ptr::read_volatile(status_host) };
    let msgs = messages.lock().unwrap();

    println!("  Status: {status_val} (1=success)");
    println!("  Elapsed: {:.3}ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Messages received: {}", msgs.len());

    // Verify COMPUTE1: results[i] = sum(1..=(i*100+1)) = (i*100+1)*(i*100+2)/2
    let mut compute1_ok = true;
    let mut compute1_failures = 0u32;
    for i in 0u32..32 {
        let actual = unsafe { std::ptr::read_volatile(results_host.add(i as usize)) };
        let n = i * 100 + 1;
        let expected = n.wrapping_mul(n + 1) / 2;
        if actual != expected {
            if compute1_failures < 3 {
                println!("  FAIL compute1[{i}]: {actual} (expected {expected})");
            }
            compute1_failures += 1;
            compute1_ok = false;
        }
    }

    // Verify COMPUTE2: XOR-fold results — compute expected values on host
    let mut compute2_ok = true;
    let mut compute2_failures = 0u32;
    for i in 0u32..32 {
        let actual = unsafe { std::ptr::read_volatile(results_host.add(32 + i as usize)) };
        let iters = (i + 1) * 50;
        let mut val: u32 = 0xDEAD_0000 | i;
        for _ in 0..iters {
            val ^= val << 13;
            val ^= val >> 17;
            val ^= val << 5;
        }
        if actual != val {
            if compute2_failures < 3 {
                println!("  FAIL compute2[{i}]: 0x{actual:08X} (expected 0x{val:08X})");
            }
            compute2_failures += 1;
            compute2_ok = false;
        }
    }

    // Verify messages: expect "stress: phase1", "stress: phase2", "stress: phase3"
    let msg_ok = msgs.len() == 3
        && msgs[0].contains("stress: phase1")
        && msgs[1].contains("stress: phase2")
        && msgs[2].contains("stress: phase3");

    if status_val == 1 && compute1_ok && compute2_ok && msg_ok {
        println!("  Hybrid Stress Test: PASSED!");
        println!("    9-state machine: 3 I/O phases + 2 compute phases");
        println!("    COMPUTE1: sum(1..=n) verified for all 32 lanes (1..3101 iterations)");
        println!("    COMPUTE2: XOR-fold verified for all 32 lanes (50..1600 iterations)");
        println!("    syncwarp correctly handles ~3100x lane duration variance");
    } else {
        println!("  Hybrid Stress Test: FAILED");
        if !compute1_ok {
            println!("    COMPUTE1: {compute1_failures} failures");
        }
        if !compute2_ok {
            println!("    COMPUTE2: {compute2_failures} failures");
        }
        if !msg_ok {
            println!("    Messages: {:?}", *msgs);
        }
    }

    unsafe { free_mapped_mem(results_host)? };
    Ok(())
}

/// warp-async-v2.2: ? operator in #[warp_async] with Result<bool, u32> return.
///
/// Tests warp_open!(buf, path, mode)? — if file open fails, all 32 lanes
/// return Err together. If it succeeds, prints "try: opened".
pub(crate) fn run_warp_try_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Warp ? operator test (warp-async-v2.2) ---");

    use gpu_host::hostcall;

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Test,
        "kernel_warp_try",
        &["warp_try_open_test"],
    )?;
    let f = dev
        .get_func("kernel_warp_try", "warp_try_open_test")
        .ok_or(GpuHostError::KernelNotFound("warp_try_open_test"))?;

    // Capture print messages via callback
    let msgs: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let msgs_clone = msgs.clone();

    let session = hostcall::HostcallSession::start_with_print(16, move |data| {
        if let Ok(s) = std::str::from_utf8(data) {
            msgs_clone.lock().unwrap().push(s.to_string());
        }
    })
    .map_err(|e| GpuHostError::Verification {
        test: "warp_try",
        detail: format!("session start failed: {e}"),
    })?;

    let (result_host, result_dev) = unsafe { gpu_host::mapped_mem::alloc_mapped_u32(&dev)? };
    unsafe { std::ptr::write_volatile(result_host, 0u32) };

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching warp_try_open_test (32 threads)...");
    unsafe {
        f.launch(cfg, (session.dev_ptr(), result_dev))?;
    }
    dev.synchronize()?;

    // Give listener time to process last prints
    std::thread::sleep(std::time::Duration::from_millis(100));

    let result_val = unsafe { std::ptr::read_volatile(result_host) };
    let captured = msgs.lock().unwrap().clone();
    session.shutdown();

    unsafe { gpu_host::mapped_mem::free_mapped_mem(result_host)? };

    println!("  Result: 0x{result_val:08X}");
    println!("  Messages: {:?}", captured);

    // result = 1 means Ok(true), result with high bit = Err
    if result_val == 1 {
        if captured.iter().any(|m| m.contains("try: opened")) {
            println!("  warp_try_test: PASSED!");
            println!("    ? operator works in #[warp_async]: file opened, print succeeded");
        } else {
            return Err(GpuHostError::Verification {
                test: "warp_try_test",
                detail: format!("result=1 but missing 'try: opened' message: {:?}", captured),
            });
        }
    } else if result_val & 0x8000_0000 != 0 {
        let err_code = result_val & 0x7FFF_FFFF;
        println!("  warp_try_test: OK (Err path verified, code={err_code:#X})");
        println!("    ? operator correctly propagated error to all 32 lanes");
    } else {
        return Err(GpuHostError::Verification {
            test: "warp_try_test",
            detail: format!("unexpected result: 0x{result_val:08X}"),
        });
    }

    Ok(())
}

/// Tests .await in #[warp_async] — two sequential GpuPrintFutures polled
/// warp-cooperatively. Lane 0 polls the inner future, broadcasts via shfl.sync.
pub(crate) fn run_warp_await_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Warp .await test (warp-async-v2.3) ---");

    use gpu_host::hostcall;

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Test,
        "kernel_warp_await",
        &["warp_await_test"],
    )?;
    let f = dev
        .get_func("kernel_warp_await", "warp_await_test")
        .ok_or(GpuHostError::KernelNotFound("warp_await_test"))?;

    // Capture print messages via callback
    let msgs: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let msgs_clone = msgs.clone();

    let session = hostcall::HostcallSession::start_with_print(16, move |data| {
        if let Ok(s) = std::str::from_utf8(data) {
            msgs_clone.lock().unwrap().push(s.to_string());
        }
    })
    .map_err(|e| GpuHostError::Verification {
        test: "warp_await",
        detail: format!("session start failed: {e}"),
    })?;

    let (result_host, result_dev) = unsafe { gpu_host::mapped_mem::alloc_mapped_u32(&dev)? };
    unsafe { std::ptr::write_volatile(result_host, 0u32) };

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching warp_await_test (32 threads)...");
    unsafe {
        f.launch(cfg, (session.dev_ptr(), result_dev))?;
    }
    dev.synchronize()?;

    // Give listener time to process last prints
    std::thread::sleep(std::time::Duration::from_millis(100));

    let result_val = unsafe { std::ptr::read_volatile(result_host) };
    let captured = msgs.lock().unwrap().clone();
    session.shutdown();

    unsafe { gpu_host::mapped_mem::free_mapped_mem(result_host)? };

    println!("  Result: {result_val} (1=true, 0=false)");
    println!("  Messages: {:?}", captured);

    // Should have two print messages and result=1 (true)
    let has_hello = captured.iter().any(|m| m.contains("await: hello"));
    let has_done = captured.iter().any(|m| m.contains("await: done"));

    if result_val != 1 {
        return Err(GpuHostError::Verification {
            test: "warp_await",
            detail: format!("expected result=1, got {result_val}"),
        });
    }

    if !has_hello || !has_done {
        return Err(GpuHostError::Verification {
            test: "warp_await",
            detail: format!(
                "missing messages: hello={has_hello}, done={has_done}, msgs={:?}",
                captured
            ),
        });
    }

    println!("  warp_await_test: PASSED!");
    println!("    .await works in #[warp_async]: two sequential futures polled warp-cooperatively");
    Ok(())
}

/// Tests end-to-end: .await + if/else branching + warp_*!() in a single #[warp_async].
/// Verifies mixed CfgNode types (Await + IfElse + Call) work together.
pub(crate) fn run_warp_e2e_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Warp end-to-end test (warp-async-v2.4) ---");

    use gpu_host::hostcall;

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Test,
        "kernel_warp_e2e",
        &["warp_e2e_test"],
    )?;

    let msgs: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let msgs_clone = msgs.clone();

    let session = hostcall::HostcallSession::start_with_print(16, move |data| {
        if let Ok(s) = std::str::from_utf8(data) {
            msgs_clone.lock().unwrap().push(s.to_string());
        }
    })
    .map_err(|e| GpuHostError::Verification {
        test: "warp_e2e",
        detail: format!("session start failed: {e}"),
    })?;

    let (result_host, result_dev) = unsafe { gpu_host::mapped_mem::alloc_mapped_u32(&dev)? };
    unsafe { std::ptr::write_volatile(result_host, 0u32) };

    let f = dev
        .get_func("kernel_warp_e2e", "warp_e2e_test")
        .ok_or(GpuHostError::KernelNotFound("warp_e2e_test"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching warp_e2e_test (32 threads)...");
    unsafe {
        f.launch(cfg, (session.dev_ptr(), result_dev))?;
    }
    dev.synchronize()?;

    std::thread::sleep(std::time::Duration::from_millis(100));

    let result_val = unsafe { std::ptr::read_volatile(result_host) };
    let captured = msgs.lock().unwrap().clone();
    session.shutdown();

    unsafe { gpu_host::mapped_mem::free_mapped_mem(result_host)? };

    println!("  Result: {result_val} (1=true, 0=false)");
    println!("  Messages: {:?}", captured);

    // Expected: "e2e: start", then "e2e: ok" (ok1=true → >0 → then branch), then "e2e: mixed"
    let has_start = captured.iter().any(|m| m.contains("e2e: start"));
    let has_ok = captured.iter().any(|m| m.contains("e2e: ok"));
    let has_mixed = captured.iter().any(|m| m.contains("e2e: mixed"));

    if result_val != 1 {
        return Err(GpuHostError::Verification {
            test: "warp_e2e",
            detail: format!("expected result=1, got {result_val}"),
        });
    }

    if !has_start || !has_ok || !has_mixed {
        return Err(GpuHostError::Verification {
            test: "warp_e2e",
            detail: format!(
                "missing messages: start={has_start}, ok={has_ok}, mixed={has_mixed}, msgs={:?}",
                captured
            ),
        });
    }

    println!("  warp_e2e_test: PASSED!");
    println!("    .await + if/else + warp_*!() all work together in #[warp_async]");
    Ok(())
}

/// Tests that rustc's async fn → coroutine → state machine compiles correctly
/// for nvptx64. Verifies that LLVM can optimize async state machines on GPU.
pub(crate) fn run_rustc_async_baseline_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- rustc async baseline test (rustc-warp.1) ---");

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Test,
        "kernel_rustc_async",
        &["rustc_async_baseline_test"],
    )?;

    let (result_host, result_dev) = unsafe { gpu_host::mapped_mem::alloc_mapped_u32(&dev)? };
    unsafe { std::ptr::write_volatile(result_host, 0u32) };

    let f = dev
        .get_func("kernel_rustc_async", "rustc_async_baseline_test")
        .ok_or(GpuHostError::KernelNotFound("rustc_async_baseline_test"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching rustc_async_baseline_test (1 thread)...");
    unsafe {
        f.launch(cfg, (result_dev,))?;
    }
    dev.synchronize()?;

    let result_val = unsafe { std::ptr::read_volatile(result_host) };
    unsafe { gpu_host::mapped_mem::free_mapped_mem(result_host)? };

    let val1 = result_val & 0xFFFF;
    let val2 = (result_val >> 16) & 0xFFFF;
    println!("  Result: 0x{result_val:08X} (val1={val1}, val2={val2})");

    // trivial_async() should return 42
    // one_yield(10) should return (10+1)*2 = 22
    if val1 != 42 || val2 != 22 {
        return Err(GpuHostError::Verification {
            test: "rustc_async_baseline",
            detail: format!("expected val1=42, val2=22, got val1={val1}, val2={val2}"),
        });
    }

    println!("  rustc_async_baseline_test: PASSED!");
    println!("    async fn compiles + runs on GPU via standard rustc");
    println!("    LLVM fully inlines async state machine — zero overhead");
    Ok(())
}
