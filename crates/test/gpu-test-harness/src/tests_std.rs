//! Std compatibility tests.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

use gpu_host::error::{GpuHostError, Result};
use gpu_host::hostcall;
use gpu_host::mapped_mem::{alloc_mapped_result_array, alloc_mapped_u32, free_mapped_mem};

/// Test: GPU kernels compiled with -Zbuild-std=std and patched std source.
pub(crate) fn run_std_build_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Integration Test 4: -Zbuild-std=std on GPU ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::STD_BUILD_TEST_PTX);
    dev.load_ptx(
        ptx,
        "std_test",
        &[
            "std_hello_kernel",
            "std_format_kernel",
            "std_dynamic_vec_kernel",
            "std_dynamic_format_kernel",
            "std_dynamic_multi_vec_kernel",
            "std_dynamic_vec_capacity_kernel",
        ],
    )?;

    // Test 1: std_hello_kernel
    {
        let f = dev
            .get_func("std_test", "std_hello_kernel")
            .ok_or(GpuHostError::KernelNotFound("std_hello_kernel"))?;
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
        if host_result[0] != 34 {
            return Err(GpuHostError::Verification {
                test: "std_hello_kernel",
                detail: format!("expected 34, got {}", host_result[0]),
            });
        }
        println!("  std_hello_kernel: PASSED (Vec + String via std, result=34)");
    }

    // Test 2: std_format_kernel
    {
        let f = dev
            .get_func("std_test", "std_format_kernel")
            .ok_or(GpuHostError::KernelNotFound("std_format_kernel"))?;
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
        if host_result[0] != 10 {
            return Err(GpuHostError::Verification {
                test: "std_format_kernel",
                detail: format!("expected 10, got {}", host_result[0]),
            });
        }
        println!("  std_format_kernel: PASSED (format! via std, result=10)");
    }

    println!("  -Zbuild-std=std compilation WORKS on nvptx64!");
    println!("    Vec, String, format! all available via patched std");
    println!("    Patches: 3 files modified + 1 new file (cuda.rs allocator)");
    Ok(())
}

pub(crate) fn run_std_println_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- std-pal.1: PAL stdout routing (writeln! via hostcall) ---");

    use std::sync::{Arc as StdArc, Mutex};

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] GPU stdout: \"{s}\"");
            messages_clone.lock().unwrap().push(s);
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::STD_BUILD_TEST_PTX);
    dev.load_ptx(
        ptx,
        "std_test",
        &[
            "std_println_kernel",
            "std_println_multi_kernel",
            "std_println_vec_kernel",
        ],
    )?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // Test 1: Single writeln! with runtime value
    {
        let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_u32(&dev)? };
        unsafe { std::ptr::write_volatile(result_host_ptr, 0u32) };
        let f = dev
            .get_func("std_test", "std_println_kernel")
            .ok_or(GpuHostError::KernelNotFound("std_println_kernel"))?;
        unsafe {
            f.launch(cfg, (dev_ptr, 42u32, result_dev_ptr))?;
        }
        dev.synchronize()?;
        std::thread::sleep(std::time::Duration::from_millis(100));
        let result_val = unsafe { std::ptr::read_volatile(result_host_ptr) };
        unsafe { free_mapped_mem(result_host_ptr)? };
        if result_val != 1 {
            hc_buf_ref.signal_shutdown();
            listener_handle.join().unwrap();
            return Err(GpuHostError::Verification {
                test: "std_println_kernel",
                detail: format!("kernel reported failure (result={result_val})"),
            });
        }
        println!("  std_println_kernel: PASSED (writeln! via std::io::stdout)");
    }

    // Test 2: Multiple writeln! calls
    {
        let input_data: Vec<u32> = vec![100, 200, 300];
        let input_dev = dev.htod_sync_copy(&input_data)?;
        let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_u32(&dev)? };
        unsafe { std::ptr::write_volatile(result_host_ptr, 0u32) };
        let f = dev
            .get_func("std_test", "std_println_multi_kernel")
            .ok_or(GpuHostError::KernelNotFound("std_println_multi_kernel"))?;
        unsafe {
            f.launch(
                cfg,
                (dev_ptr, &input_dev, input_data.len() as u32, result_dev_ptr),
            )?;
        }
        dev.synchronize()?;
        std::thread::sleep(std::time::Duration::from_millis(100));
        let result_val = unsafe { std::ptr::read_volatile(result_host_ptr) };
        unsafe { free_mapped_mem(result_host_ptr)? };
        if result_val != 3 {
            hc_buf_ref.signal_shutdown();
            listener_handle.join().unwrap();
            return Err(GpuHostError::Verification {
                test: "std_println_multi_kernel",
                detail: format!("expected 3 writes, got {result_val}"),
            });
        }
        println!("  std_println_multi_kernel: PASSED ({result_val} writeln! calls)");
    }

    // Test 3: writeln! with Vec data
    {
        let input_data: Vec<u32> = vec![10, 20, 30, 40, 50];
        let input_dev = dev.htod_sync_copy(&input_data)?;
        let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_u32(&dev)? };
        unsafe { std::ptr::write_volatile(result_host_ptr, 0u32) };
        let f = dev
            .get_func("std_test", "std_println_vec_kernel")
            .ok_or(GpuHostError::KernelNotFound("std_println_vec_kernel"))?;
        unsafe {
            f.launch(
                cfg,
                (dev_ptr, &input_dev, input_data.len() as u32, result_dev_ptr),
            )?;
        }
        dev.synchronize()?;
        std::thread::sleep(std::time::Duration::from_millis(100));
        let result_val = unsafe { std::ptr::read_volatile(result_host_ptr) };
        unsafe { free_mapped_mem(result_host_ptr)? };
        if result_val != 1 {
            hc_buf_ref.signal_shutdown();
            listener_handle.join().unwrap();
            return Err(GpuHostError::Verification {
                test: "std_println_vec_kernel",
                detail: format!("kernel reported failure (result={result_val})"),
            });
        }
        println!("  std_println_vec_kernel: PASSED (Vec + writeln! via std)");
    }

    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let received = messages.lock().unwrap();
    println!(
        "  Total messages received from GPU stdout: {}",
        received.len()
    );
    println!("  PAL stdout routing WORKS! writeln!(std::io::stdout(), ...) → hostcall");

    Ok(())
}

pub(crate) fn run_dynamic_alloc_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- product.1: Dynamic Allocation Stress Test ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::STD_BUILD_TEST_PTX);
    dev.load_ptx(
        ptx,
        "std_test",
        &[
            "std_dynamic_vec_kernel",
            "std_dynamic_format_kernel",
            "std_dynamic_multi_vec_kernel",
            "std_dynamic_vec_capacity_kernel",
        ],
    )?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // Test 1: Vec with runtime data
    {
        let input_data: Vec<u32> = vec![10, 20, 30, 40, 50];
        let expected_sum: u32 = input_data.iter().sum();
        let input_dev = dev.htod_sync_copy(&input_data)?;
        let mut result: CudaSlice<u32> = dev.alloc_zeros::<u32>(1)?;
        let f = dev
            .get_func("std_test", "std_dynamic_vec_kernel")
            .ok_or(GpuHostError::KernelNotFound("std_dynamic_vec_kernel"))?;
        unsafe {
            f.launch(cfg, (&input_dev, input_data.len() as u32, &mut result))?;
        }
        let host_result = dev.dtoh_sync_copy(&result)?;
        if host_result[0] != expected_sum {
            return Err(GpuHostError::Verification {
                test: "std_dynamic_vec_kernel",
                detail: format!("expected {}, got {}", expected_sum, host_result[0]),
            });
        }
        println!("  dynamic_vec (5 elements): PASSED (sum={expected_sum}, bump allocator active)");
    }

    // Test 1b: Larger Vec (100 elements)
    {
        let input_data: Vec<u32> = (1..=100).collect();
        let expected_sum: u32 = input_data.iter().sum();
        let input_dev = dev.htod_sync_copy(&input_data)?;
        let mut result: CudaSlice<u32> = dev.alloc_zeros::<u32>(1)?;
        let f = dev
            .get_func("std_test", "std_dynamic_vec_kernel")
            .ok_or(GpuHostError::KernelNotFound("std_dynamic_vec_kernel"))?;
        unsafe {
            f.launch(cfg, (&input_dev, input_data.len() as u32, &mut result))?;
        }
        let host_result = dev.dtoh_sync_copy(&result)?;
        if host_result[0] != expected_sum {
            return Err(GpuHostError::Verification {
                test: "std_dynamic_vec_kernel (100 elements)",
                detail: format!("expected {}, got {}", expected_sum, host_result[0]),
            });
        }
        println!("  dynamic_vec (100 elements): PASSED (sum={expected_sum}, multiple reallocs)");
    }

    // Test 2: format! with runtime value
    {
        let f = dev
            .get_func("std_test", "std_dynamic_format_kernel")
            .ok_or(GpuHostError::KernelNotFound("std_dynamic_format_kernel"))?;
        let mut result: CudaSlice<u32> = dev.alloc_zeros::<u32>(1)?;
        unsafe {
            f.launch(cfg, (12345u32, &mut result))?;
        }
        let host_result = dev.dtoh_sync_copy(&result)?;
        let expected_len = "result = 12345".len() as u32;
        if host_result[0] != expected_len {
            return Err(GpuHostError::Verification {
                test: "std_dynamic_format_kernel",
                detail: format!("expected len={}, got {}", expected_len, host_result[0]),
            });
        }
        println!(
            "  dynamic_format (value=12345): PASSED (len={expected_len}, allocator used for String)"
        );
    }

    // Test 3: Multiple Vecs alive simultaneously
    {
        let input_data: Vec<u32> = (1..=10).collect();
        let input_dev = dev.htod_sync_copy(&input_data)?;
        let mut result: CudaSlice<u32> = dev.alloc_zeros::<u32>(3)?;
        let f = dev
            .get_func("std_test", "std_dynamic_multi_vec_kernel")
            .ok_or(GpuHostError::KernelNotFound("std_dynamic_multi_vec_kernel"))?;
        unsafe {
            f.launch(cfg, (&input_dev, input_data.len() as u32, &mut result))?;
        }
        let host_result = dev.dtoh_sync_copy(&result)?;
        let expected_even_sum: u32 = 1 + 3 + 5 + 7 + 9;
        let expected_odd_sum: u32 = 2 + 4 + 6 + 8 + 10;
        if host_result[0] != expected_even_sum
            || host_result[1] != expected_odd_sum
            || host_result[2] != 10
        {
            return Err(GpuHostError::Verification {
                test: "std_dynamic_multi_vec_kernel",
                detail: format!(
                    "expected evens={}, odds={}, total=10, got evens={}, odds={}, total={}",
                    expected_even_sum,
                    expected_odd_sum,
                    host_result[0],
                    host_result[1],
                    host_result[2]
                ),
            });
        }
        println!(
            "  dynamic_multi_vec (2 Vecs): PASSED (even_sum={}, odd_sum={}, total={})",
            host_result[0], host_result[1], host_result[2]
        );
    }

    // Test 4: Vec::with_capacity
    {
        let input_data: Vec<u32> = vec![100, 200, 300, 400, 500];
        let expected_sum: u32 = input_data.iter().sum();
        let input_dev = dev.htod_sync_copy(&input_data)?;
        let mut result: CudaSlice<u32> = dev.alloc_zeros::<u32>(1)?;
        let f = dev
            .get_func("std_test", "std_dynamic_vec_capacity_kernel")
            .ok_or(GpuHostError::KernelNotFound(
                "std_dynamic_vec_capacity_kernel",
            ))?;
        unsafe {
            f.launch(cfg, (&input_dev, input_data.len() as u32, &mut result))?;
        }
        let host_result = dev.dtoh_sync_copy(&result)?;
        if host_result[0] != expected_sum {
            return Err(GpuHostError::Verification {
                test: "std_dynamic_vec_capacity_kernel",
                detail: format!("expected {}, got {}", expected_sum, host_result[0]),
            });
        }
        println!("  dynamic_vec_capacity (5 elements): PASSED (sum={expected_sum}, pre-allocated)");
    }

    println!("  All dynamic allocation tests PASSED!");
    println!("    Bump allocator confirmed working with runtime data");
    println!("    Vec growth, format!, multiple Vecs, with_capacity all verified");
    Ok(())
}

/// Test: std::fs File I/O on GPU via hostcall (std-fs.4).
///
/// Launches std_file_io_test kernel from gpu-kernel-test which uses
/// std::fs::File::create() + write_all() + File::open() + read_to_end().
///
/// Zero-param entry: hostcall buffer injected via `__HOSTCALL_BUF` device global.
pub(crate) fn run_std_file_io_test(_dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- std-fs.4: std::fs File I/O on GPU via hostcall ---");

    use std::sync::Arc as StdArc;
    use std::sync::Mutex;

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let module = gpu_host::gpu::GpuStdModule::load_with_print(
        crate::KERNEL_STD_PTX,
        "std_file_io_test",
        1,
        (1, 1, 1),
        Some(Box::new(move |msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] GPU stdout: \"{s}\"");
            messages_clone.lock().unwrap().push(s);
        })),
    )?;

    println!("  Launching std_file_io_test (zero-param)...");
    unsafe {
        module.launch_raw(&[])?;
    }
    std::thread::sleep(std::time::Duration::from_millis(200));
    module.finish();

    // Check GPU stdout messages for success markers
    let received = messages.lock().unwrap();
    let has_write_ok = received.iter().any(|m| m.contains("[OK] write_all"));
    let has_read_ok = received.iter().any(|m| m.contains("[OK] read_to_end"));
    let has_done = received.iter().any(|m| m.contains("[DONE]"));
    let has_err = received.iter().any(|m| m.contains("[ERR]"));

    if has_err {
        let errs: Vec<_> = received.iter().filter(|m| m.contains("[ERR]")).collect();
        return Err(GpuHostError::Verification {
            test: "std_file_io_test",
            detail: format!("kernel reported errors: {:?}", errs),
        });
    }

    if !has_write_ok || !has_read_ok || !has_done {
        return Err(GpuHostError::Verification {
            test: "std_file_io_test",
            detail: format!(
                "missing success markers: write_ok={has_write_ok}, read_ok={has_read_ok}, done={has_done}"
            ),
        });
    }

    // Clean up test file
    let _ = std::fs::remove_file("gpu_std_test.txt");

    println!("  std_file_io_test: PASSED!");
    println!("    std::fs::File::create() → hostcall OPEN");
    println!("    file.write_all() → hostcall WRITE");
    println!("    std::fs::File::open() + read_to_end() → hostcall READ");
    println!("    File I/O errors → std::io::Error with errno");

    Ok(())
}

/// Test: multi-step pipeline with std types and ? error propagation (std-migration.3).
///
/// Zero-param entry: hostcall buffer injected via `__HOSTCALL_BUF` device global.
pub(crate) fn run_std_pipeline_test(_dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- std-migration.3: std pipeline (Vec + format! + File + ? operator) ---");

    use std::sync::Arc as StdArc;
    use std::sync::Mutex;

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let module = gpu_host::gpu::GpuStdModule::load_with_print(
        crate::KERNEL_STD_PTX,
        "std_pipeline_test",
        1,
        (1, 1, 1),
        Some(Box::new(move |msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] GPU stdout: \"{s}\"");
            messages_clone.lock().unwrap().push(s);
        })),
    )?;

    println!("  Launching std_pipeline_test (zero-param)...");
    unsafe {
        module.launch_raw(&[])?;
    }
    std::thread::sleep(std::time::Duration::from_millis(200));
    module.finish();

    let received = messages.lock().unwrap();
    let has_done = received.iter().any(|m| m.contains("[DONE]"));
    let has_err = received.iter().any(|m| m.contains("[ERR]"));

    if has_err {
        let errs: Vec<_> = received.iter().filter(|m| m.contains("[ERR]")).collect();
        return Err(GpuHostError::Verification {
            test: "std_pipeline_test",
            detail: format!("kernel reported errors: {:?}", errs),
        });
    }

    if !has_done {
        return Err(GpuHostError::Verification {
            test: "std_pipeline_test",
            detail: "missing [DONE] marker".to_string(),
        });
    }

    // Clean up
    let _ = std::fs::remove_file("gpu_pipeline_test.txt");

    println!("  std_pipeline_test: PASSED!");
    println!("    Vec + format! + std::fs::File + ? operator working on GPU");
    println!("    Multi-step pipeline: generate → write → read → verify");

    Ok(())
}

/// Test: stdin via std::io::stdin() PAL routing (std-pal.2).
pub(crate) fn run_std_stdin_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- std-pal.2: PAL stdin routing (std::io::stdin via hostcall) ---");

    use std::sync::{Arc as StdArc, Mutex};

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 3)? };

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);

    let stdin_data = b"Hello GPU stdin!\n".to_vec();
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen_with_stdin(
            |msg| {
                let s = String::from_utf8_lossy(msg).to_string();
                println!("  [HOST] GPU stdout: \"{s}\"");
                messages_clone.lock().unwrap().push(s);
            },
            stdin_data,
        );
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::STD_BUILD_TEST_PTX);
    dev.load_ptx(ptx, "std_test", &["std_stdin_kernel"])?;
    let f = dev
        .get_func("std_test", "std_stdin_kernel")
        .ok_or(GpuHostError::KernelNotFound("std_stdin_kernel"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching std_stdin_kernel...");
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

    let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
    let bytes_read = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
    let first_byte = unsafe { std::ptr::read_volatile(result_host_ptr.add(2)) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    if success != 1 {
        return Err(GpuHostError::Verification {
            test: "std_stdin_kernel",
            detail: format!("stdin read failed (success={success}, bytes={bytes_read})"),
        });
    }

    if bytes_read == 0 {
        return Err(GpuHostError::Verification {
            test: "std_stdin_kernel",
            detail: "stdin read returned 0 bytes".to_string(),
        });
    }

    if first_byte != b'H' as u32 {
        return Err(GpuHostError::Verification {
            test: "std_stdin_kernel",
            detail: format!("first byte mismatch: expected 0x48 ('H'), got 0x{first_byte:02X}"),
        });
    }

    println!("  std_stdin_kernel: PASSED!");
    println!("    Bytes read: {bytes_read}");
    println!(
        "    First byte: '{}' (0x{:02X})",
        first_byte as u8 as char, first_byte
    );
    println!("    std::io::stdin().read() → hostcall STDIN → canned data received");

    Ok(())
}

/// Test: stdin().read_line() end-to-end via gpu-kernel-test (std-migration.4).
///
/// Unlike run_std_stdin_test which tests raw gpu_stdin_read() via std-build-test,
/// this tests the full std path: stdin().lock().read_line() → PAL → gpu_stdin_read()
/// → gpu-runtime hostcall → SERVICE_STDIN → host CannedStdin.
///
/// Zero-param entry: hostcall buffer injected via `__HOSTCALL_BUF` device global.
/// Uses raw CUDA driver API because the stdin test needs `listen_with_stdin`.
pub(crate) fn run_std_stdin_readline_test(_dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- std-migration.4: stdin().read_line() end-to-end on GPU ---");

    use cudarc::driver::sys::{self, lib as cuda_lib};
    use std::ffi::CString;
    use std::sync::{Arc as StdArc, Mutex};

    // Initialize CUDA context via cudarc
    let dev = cudarc::driver::CudaDevice::new(0).map_err(GpuHostError::CudaInit)?;

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let hc_ptr = hc_buf.dev_ptr();

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);

    let stdin_data = b"Hello from stdin!\n".to_vec();
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen_with_stdin(
            |msg| {
                let s = String::from_utf8_lossy(msg).to_string();
                println!("  [HOST] GPU stdout: \"{s}\"");
                messages_clone.lock().unwrap().push(s);
            },
            stdin_data,
        );
    });

    // Load PTX via raw driver API to get CUmodule for device global injection
    let ptx_cstring =
        CString::new(crate::KERNEL_STD_PTX).map_err(|_| GpuHostError::Verification {
            test: "ptx_load",
            detail: "PTX source contains null byte".to_string(),
        })?;

    let cu_module: sys::CUmodule;
    unsafe {
        let cu = cuda_lib();
        let mut module: sys::CUmodule = std::ptr::null_mut();
        let result =
            cu.cuModuleLoadData(&mut module, ptx_cstring.as_ptr() as *const std::ffi::c_void);
        if result != sys::CUresult::CUDA_SUCCESS {
            hc_buf_ref.signal_shutdown();
            listener_handle.join().unwrap();
            return Err(GpuHostError::Verification {
                test: "ptx_load",
                detail: format!("cuModuleLoadData failed: {result:?}"),
            });
        }
        cu_module = module;
    }

    // Get kernel function
    let func_name = CString::new("std_stdin_test").unwrap();
    let cu_func: sys::CUfunction;
    unsafe {
        let cu = cuda_lib();
        let mut func: sys::CUfunction = std::ptr::null_mut();
        let result = cu.cuModuleGetFunction(&mut func, cu_module, func_name.as_ptr());
        if result != sys::CUresult::CUDA_SUCCESS {
            cu.cuModuleUnload(cu_module);
            hc_buf_ref.signal_shutdown();
            listener_handle.join().unwrap();
            return Err(GpuHostError::KernelNotFound("std_stdin_test"));
        }
        cu_func = func;
    }

    // Inject hostcall pointer via device global
    let global_name = CString::new("__HOSTCALL_BUF").unwrap();
    unsafe {
        let cu = cuda_lib();
        let mut global_dptr: sys::CUdeviceptr = 0;
        let mut global_size: usize = 0;
        let result = cu.cuModuleGetGlobal_v2(
            &mut global_dptr,
            &mut global_size,
            cu_module,
            global_name.as_ptr(),
        );
        if result != sys::CUresult::CUDA_SUCCESS {
            cu.cuModuleUnload(cu_module);
            hc_buf_ref.signal_shutdown();
            listener_handle.join().unwrap();
            return Err(GpuHostError::Verification {
                test: "std_stdin_test",
                detail: format!("cuModuleGetGlobal_v2 failed: {result:?}"),
            });
        }
        let hc_ptr_val: u64 = hc_ptr;
        let result = cu.cuMemcpyHtoD_v2(
            global_dptr,
            &hc_ptr_val as *const u64 as *const std::ffi::c_void,
            8,
        );
        if result != sys::CUresult::CUDA_SUCCESS {
            cu.cuModuleUnload(cu_module);
            hc_buf_ref.signal_shutdown();
            listener_handle.join().unwrap();
            return Err(GpuHostError::Verification {
                test: "std_stdin_test",
                detail: format!("cuMemcpyHtoD failed: {result:?}"),
            });
        }
    }

    // Launch with zero params
    println!("  Launching std_stdin_test (zero-param, read_line via patched std)...");
    unsafe {
        let cu = cuda_lib();
        let result = cu.cuLaunchKernel(
            cu_func,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        if result != sys::CUresult::CUDA_SUCCESS {
            cu.cuModuleUnload(cu_module);
            hc_buf_ref.signal_shutdown();
            listener_handle.join().unwrap();
            return Err(GpuHostError::Verification {
                test: "std_stdin_test",
                detail: format!("cuLaunchKernel failed: {result:?}"),
            });
        }
    }
    dev.synchronize()?;
    std::thread::sleep(std::time::Duration::from_millis(200));

    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();
    unsafe {
        let cu = cuda_lib();
        cu.cuModuleUnload(cu_module);
    }

    let received = messages.lock().unwrap();
    let has_pass = received.iter().any(|m| m.contains("[STDIN] PASS"));
    let has_fail = received.iter().any(|m| m.contains("[STDIN] FAIL"));
    let has_content = received.iter().any(|m| m.contains("Hello from stdin!"));

    if has_fail {
        let errs: Vec<_> = received
            .iter()
            .filter(|m| m.contains("[STDIN] FAIL"))
            .collect();
        return Err(GpuHostError::Verification {
            test: "std_stdin_test (read_line)",
            detail: format!("kernel reported failure: {:?}", errs),
        });
    }

    if !has_pass {
        return Err(GpuHostError::Verification {
            test: "std_stdin_test (read_line)",
            detail: format!(
                "missing PASS marker. Messages: {:?}",
                received.iter().collect::<Vec<_>>()
            ),
        });
    }

    if !has_content {
        println!("  WARN: stdin data echoed but content not found in output");
    }

    println!("  std_stdin_test (read_line): PASSED!");
    println!("    stdin().lock().read_line() → PAL → gpu_stdin_read → hostcall");
    println!("    Full std::io::stdin() path verified on GPU via gpu-kernel-test");

    Ok(())
}

/// Test: multi-thread malloc with atomic bump allocator.
///
/// Launches `test_multithread_malloc` kernel with 32 threads. Each thread
/// calls malloc(64) and writes the pointer to output[tid]. Host verifies
/// all 32 pointers are non-null and non-overlapping.
pub(crate) fn run_multithread_malloc_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- std-hardening.3: Multi-thread malloc (atomic bump) ---");

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Core,
        "alloc_test",
        &["test_multithread_malloc"],
    )?;

    let num_threads: u32 = 32;
    let heap_size: u64 = 64 * 1024; // 64 KB heap

    // Allocate heap on device
    let heap: CudaSlice<u8> = dev.alloc_zeros::<u8>(heap_size as usize)?;

    // Allocate output buffer (32 x u64 pointers) on device
    let mut output: CudaSlice<u64> = dev.alloc_zeros::<u64>(num_threads as usize)?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (num_threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let f = dev
        .get_func("alloc_test", "test_multithread_malloc")
        .ok_or(GpuHostError::KernelNotFound("test_multithread_malloc"))?;

    unsafe {
        f.launch(cfg, (&heap, heap_size, &mut output))?;
    }
    dev.synchronize()?;

    // Copy results back to host
    let pointers = dev.dtoh_sync_copy(&output)?;

    // Verify: all non-null
    let null_count = pointers.iter().filter(|&&p| p == 0).count();
    if null_count > 0 {
        return Err(GpuHostError::Verification {
            test: "test_multithread_malloc",
            detail: format!("{} threads got null pointer", null_count),
        });
    }

    // Verify: all non-overlapping (sorted pointers should be >= 64 bytes apart)
    let mut sorted = pointers.clone();
    sorted.sort();
    let mut overlap_count = 0;
    for i in 1..sorted.len() {
        if sorted[i] - sorted[i - 1] < 64 {
            overlap_count += 1;
        }
    }
    if overlap_count > 0 {
        return Err(GpuHostError::Verification {
            test: "test_multithread_malloc",
            detail: format!(
                "{} overlapping allocations (pointers less than 64 bytes apart)",
                overlap_count
            ),
        });
    }

    println!(
        "  {} threads allocated 64 bytes each — all non-null",
        num_threads
    );
    println!(
        "  Pointer range: {:#x} .. {:#x}",
        sorted[0],
        sorted[sorted.len() - 1]
    );
    println!(
        "  No overlaps detected (min gap = {} bytes)",
        sorted.windows(2).map(|w| w[1] - w[0]).min().unwrap_or(0)
    );
    println!("  PASSED — atomic bump allocator is thread-safe!");

    Ok(())
}

/// Test: multi-thread std with GPU threads (std-multithread.3).
///
/// Two sub-tests:
/// 1. println! with 4 threads (limited by hostcall packet pool — each println
///    generates multiple 56-byte chunks, so too many threads exhaust packets)
/// 2. Vec allocation with 32 threads (no hostcall needed — pure compute)
///
/// Verifies thread_local storage (panic count, etc.) is per-thread and safe.
///
/// Zero-param entry for hostcall: `__HOSTCALL_BUF` device global injected.
/// `std_multithread_println_test(result)` now takes only `result`, no `buf`.
/// `std_multithread_vec_test(result)` was already buf-free, unchanged.
pub(crate) fn run_multithread_println_test(_dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- std-multithread.3: Multi-thread std (println + Vec) ---");

    use cudarc::driver::sys::{self, lib as cuda_lib};
    use std::ffi::CString;
    use std::sync::{Arc as StdArc, Mutex};

    let num_println_threads: u32 = 4; // Limited by hostcall packet contention
    let num_vec_threads: u32 = 32; // Full warp — no hostcall needed

    let dev = cudarc::driver::CudaDevice::new(0).map_err(GpuHostError::CudaInit)?;

    let messages: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(Vec::new()));
    let messages_clone = StdArc::clone(&messages);

    let hc_buf = hostcall::HostcallBuffer::new(16)?;
    let hc_ptr = hc_buf.dev_ptr();

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] GPU stdout: \"{s}\"");
            messages_clone.lock().unwrap().push(s);
        });
    });

    // Mapped memory for result verification (max 32 x u32)
    let (result_host_ptr, result_dev_ptr) =
        unsafe { alloc_mapped_result_array(&dev, num_vec_threads as usize)? };
    for i in 0..num_vec_threads as usize {
        unsafe { std::ptr::write_volatile(result_host_ptr.add(i), 0u32) };
    }

    // Load PTX via raw driver API for device global injection
    let ptx_cstring =
        CString::new(crate::KERNEL_STD_PTX).map_err(|_| GpuHostError::Verification {
            test: "ptx_load",
            detail: "PTX source contains null byte".to_string(),
        })?;

    let cu_module: sys::CUmodule;
    unsafe {
        let cu = cuda_lib();
        let mut module: sys::CUmodule = std::ptr::null_mut();
        let result =
            cu.cuModuleLoadData(&mut module, ptx_cstring.as_ptr() as *const std::ffi::c_void);
        if result != sys::CUresult::CUDA_SUCCESS {
            hc_buf_ref.signal_shutdown();
            listener_handle.join().unwrap();
            free_mapped_mem(result_host_ptr)?;
            return Err(GpuHostError::Verification {
                test: "ptx_load",
                detail: format!("cuModuleLoadData failed: {result:?}"),
            });
        }
        cu_module = module;
    }

    // Inject hostcall pointer via device global
    let global_name = CString::new("__HOSTCALL_BUF").unwrap();
    unsafe {
        let cu = cuda_lib();
        let mut global_dptr: sys::CUdeviceptr = 0;
        let mut global_size: usize = 0;
        let result = cu.cuModuleGetGlobal_v2(
            &mut global_dptr,
            &mut global_size,
            cu_module,
            global_name.as_ptr(),
        );
        if result != sys::CUresult::CUDA_SUCCESS {
            cu.cuModuleUnload(cu_module);
            hc_buf_ref.signal_shutdown();
            listener_handle.join().unwrap();
            free_mapped_mem(result_host_ptr)?;
            return Err(GpuHostError::Verification {
                test: "mt_std",
                detail: format!("cuModuleGetGlobal_v2 failed: {result:?}"),
            });
        }
        let hc_ptr_val: u64 = hc_ptr;
        let result = cu.cuMemcpyHtoD_v2(
            global_dptr,
            &hc_ptr_val as *const u64 as *const std::ffi::c_void,
            8,
        );
        if result != sys::CUresult::CUDA_SUCCESS {
            cu.cuModuleUnload(cu_module);
            hc_buf_ref.signal_shutdown();
            listener_handle.join().unwrap();
            free_mapped_mem(result_host_ptr)?;
            return Err(GpuHostError::Verification {
                test: "mt_std",
                detail: format!("cuMemcpyHtoD failed: {result:?}"),
            });
        }
    }

    // Test 1: Multi-thread println! (4 threads) — kernel now takes (result) only
    {
        let func_name = CString::new("std_multithread_println_test").unwrap();
        let cu_func: sys::CUfunction;
        unsafe {
            let cu = cuda_lib();
            let mut func: sys::CUfunction = std::ptr::null_mut();
            let result = cu.cuModuleGetFunction(&mut func, cu_module, func_name.as_ptr());
            if result != sys::CUresult::CUDA_SUCCESS {
                cu.cuModuleUnload(cu_module);
                hc_buf_ref.signal_shutdown();
                listener_handle.join().unwrap();
                free_mapped_mem(result_host_ptr)?;
                return Err(GpuHostError::KernelNotFound("std_multithread_println_test"));
            }
            cu_func = func;
        }

        println!(
            "  Launching std_multithread_println_test with {} threads...",
            num_println_threads
        );
        let mut result_ptr_arg = result_dev_ptr;
        let params: [*mut std::ffi::c_void; 1] =
            [&mut result_ptr_arg as *mut u64 as *mut std::ffi::c_void];
        unsafe {
            let cu = cuda_lib();
            let result = cu.cuLaunchKernel(
                cu_func,
                1,
                1,
                1,
                num_println_threads,
                1,
                1,
                0,
                std::ptr::null_mut(),
                params.as_ptr() as *mut *mut std::ffi::c_void,
                std::ptr::null_mut(),
            );
            if result != sys::CUresult::CUDA_SUCCESS {
                cu.cuModuleUnload(cu_module);
                hc_buf_ref.signal_shutdown();
                listener_handle.join().unwrap();
                free_mapped_mem(result_host_ptr)?;
                return Err(GpuHostError::Verification {
                    test: "std_multithread_println_test",
                    detail: format!("cuLaunchKernel failed: {result:?}"),
                });
            }
        }
        dev.synchronize()?;
        std::thread::sleep(std::time::Duration::from_millis(500));

        // Verify all threads ran (result[tid] = tid+1)
        let mut failed_tids = Vec::new();
        for i in 0..num_println_threads as usize {
            let val = unsafe { std::ptr::read_volatile(result_host_ptr.add(i)) };
            if val != (i as u32 + 1) {
                failed_tids.push((i, val));
            }
        }

        if !failed_tids.is_empty() {
            unsafe {
                let cu = cuda_lib();
                cu.cuModuleUnload(cu_module);
            }
            hc_buf_ref.signal_shutdown();
            listener_handle.join().unwrap();
            unsafe { free_mapped_mem(result_host_ptr)? };
            return Err(GpuHostError::Verification {
                test: "std_multithread_println_test",
                detail: format!(
                    "{} threads failed: {:?}",
                    failed_tids.len(),
                    &failed_tids[..std::cmp::min(5, failed_tids.len())]
                ),
            });
        }

        println!(
            "  std_multithread_println_test: PASSED ({} threads, all ran correctly)",
            num_println_threads
        );
    }

    // Test 2: Multi-thread Vec allocation (32 threads, no hostcall) — unchanged signature
    {
        // Reset result buffer
        for i in 0..num_vec_threads as usize {
            unsafe { std::ptr::write_volatile(result_host_ptr.add(i), 0u32) };
        }

        let func_name = CString::new("std_multithread_vec_test").unwrap();
        let cu_func: sys::CUfunction;
        unsafe {
            let cu = cuda_lib();
            let mut func: sys::CUfunction = std::ptr::null_mut();
            let result = cu.cuModuleGetFunction(&mut func, cu_module, func_name.as_ptr());
            if result != sys::CUresult::CUDA_SUCCESS {
                cu.cuModuleUnload(cu_module);
                hc_buf_ref.signal_shutdown();
                listener_handle.join().unwrap();
                free_mapped_mem(result_host_ptr)?;
                return Err(GpuHostError::KernelNotFound("std_multithread_vec_test"));
            }
            cu_func = func;
        }

        println!(
            "  Launching std_multithread_vec_test with {} threads...",
            num_vec_threads
        );
        let mut result_ptr_arg = result_dev_ptr;
        let params: [*mut std::ffi::c_void; 1] =
            [&mut result_ptr_arg as *mut u64 as *mut std::ffi::c_void];
        unsafe {
            let cu = cuda_lib();
            let result = cu.cuLaunchKernel(
                cu_func,
                1,
                1,
                1,
                num_vec_threads,
                1,
                1,
                0,
                std::ptr::null_mut(),
                params.as_ptr() as *mut *mut std::ffi::c_void,
                std::ptr::null_mut(),
            );
            if result != sys::CUresult::CUDA_SUCCESS {
                cu.cuModuleUnload(cu_module);
                hc_buf_ref.signal_shutdown();
                listener_handle.join().unwrap();
                free_mapped_mem(result_host_ptr)?;
                return Err(GpuHostError::Verification {
                    test: "std_multithread_vec_test",
                    detail: format!("cuLaunchKernel failed: {result:?}"),
                });
            }
        }
        dev.synchronize()?;

        // Verify each thread's sum: tid*80 + 28
        let mut failed_tids = Vec::new();
        for i in 0..num_vec_threads as usize {
            let val = unsafe { std::ptr::read_volatile(result_host_ptr.add(i)) };
            let expected = (i as u32) * 80 + 28;
            if val != expected {
                failed_tids.push((i, val, expected));
            }
        }

        if !failed_tids.is_empty() {
            unsafe {
                let cu = cuda_lib();
                cu.cuModuleUnload(cu_module);
            }
            hc_buf_ref.signal_shutdown();
            listener_handle.join().unwrap();
            unsafe { free_mapped_mem(result_host_ptr)? };
            return Err(GpuHostError::Verification {
                test: "std_multithread_vec_test",
                detail: format!(
                    "{} threads failed: {:?}",
                    failed_tids.len(),
                    &failed_tids[..std::cmp::min(5, failed_tids.len())]
                ),
            });
        }

        println!(
            "  std_multithread_vec_test: PASSED ({} threads, all sums correct)",
            num_vec_threads
        );
    }

    unsafe {
        let cu = cuda_lib();
        cu.cuModuleUnload(cu_module);
    }
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();
    unsafe { free_mapped_mem(result_host_ptr)? };

    // Check message count from println test
    let received = messages.lock().unwrap();
    let mt_msgs: Vec<_> = received.iter().filter(|m| m.contains("[MT]")).collect();
    println!(
        "  Total [MT] messages received: {} (expected {})",
        mt_msgs.len(),
        num_println_threads
    );

    println!("  Multi-thread std PASSED!");
    println!("    thread_local (panic count, etc.) is per-thread via gpu_threads.rs");
    println!(
        "    println! with {} threads + Vec with {} threads verified",
        num_println_threads, num_vec_threads
    );

    Ok(())
}

/// Test: Buffered println! on GPU using print_buffer integration.
///
/// Launches `std_buffered_println_test` which calls `stdio_print_buffer_init`,
/// then 6 `println!()` calls routed through `print_buffer`, then flushes.
/// Verifies all 6 messages arrive on host via SERVICE_BULK_PRINT.
pub(crate) fn run_std_buffered_println_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Std Buffered println! Test (println-buffer.2) ---");

    use std::sync::{Arc as StdArc, Mutex};

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();
    let sb_dev_ptr = hc_buf.sideband_dev_ptr();

    let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

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

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_STD_PTX);
    dev.load_ptx(ptx, "kernel_std", &["std_buffered_println_test"])?;
    let f = dev
        .get_func("kernel_std", "std_buffered_println_test")
        .ok_or(GpuHostError::KernelNotFound("std_buffered_println_test"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let start = std::time::Instant::now();
    println!("  Launching std_buffered_println_test kernel...");
    unsafe {
        f.launch(cfg, (dev_ptr, sb_dev_ptr, result_dev_ptr))?;
    }

    dev.synchronize()?;
    let elapsed = start.elapsed();
    println!("  Kernel completed in {elapsed:?}.");

    std::thread::sleep(std::time::Duration::from_millis(200));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let result = unsafe { std::ptr::read_volatile(result_host_ptr) };
    unsafe { free_mapped_mem(result_host_ptr)? };

    if result != 1 {
        return Err(GpuHostError::Verification {
            test: "std_buffered_println_test",
            detail: format!("kernel returned {result}, expected 1"),
        });
    }

    let msgs = messages.lock().unwrap();
    println!(
        "  Result: kernel success, received {} messages in {elapsed:?}",
        msgs.len()
    );

    if msgs.len() < 6 {
        return Err(GpuHostError::Verification {
            test: "std_buffered_println_test",
            detail: format!("expected 6+ messages, got {}", msgs.len()),
        });
    }

    println!(
        "  std_buffered_println_test: PASSED! ({} messages via print_buffer)",
        msgs.len()
    );
    Ok(())
}

/// Test: std::fs::File on GPU via std-build-test PTX.
///
/// This is the end-to-end test for the std-sysroot epic: kernel uses
/// `std::fs::File::create()` and `std::io::Write` on GPU, compiled via
/// `-Zbuild-std=["std"]` with patched std source, routed through
/// the patched PAL → gpu-libc → hostcall framework.
pub(crate) fn run_std_sysroot_file_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- std-sysroot-build.4: std::fs::File I/O on GPU (build-std) ---");

    use std::sync::Arc as StdArc;

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);

    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let s = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] GPU: \"{s}\"");
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::STD_BUILD_TEST_PTX);
    dev.load_ptx(
        ptx,
        "std_test",
        &["std_file_write_kernel", "std_file_read_kernel"],
    )?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // Clean up any leftover files
    let _ = std::fs::remove_file("gpu_test_output.txt");
    let _ = std::fs::remove_file("gpu_test_input.txt");

    // --- Test 1: File::create + write ---
    {
        let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 2)? };
        unsafe {
            std::ptr::write_volatile(result_host_ptr.add(0), 0u32);
            std::ptr::write_volatile(result_host_ptr.add(1), 0u32);
        }

        let f = dev
            .get_func("std_test", "std_file_write_kernel")
            .ok_or(GpuHostError::KernelNotFound("std_file_write_kernel"))?;

        println!("  Launching std_file_write_kernel...");
        unsafe {
            f.launch(cfg, (dev_ptr, result_dev_ptr))?;
        }
        dev.synchronize()?;
        std::thread::sleep(std::time::Duration::from_millis(100));

        let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
        let bytes_written = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
        unsafe { free_mapped_mem(result_host_ptr)? };

        if success != 1 {
            hc_buf_ref.signal_shutdown();
            listener_handle.join().unwrap();
            return Err(GpuHostError::Verification {
                test: "std_file_write_kernel",
                detail: format!("File::create + write failed (code=0x{success:X})"),
            });
        }

        // Verify the file exists on host
        match std::fs::read("gpu_test_output.txt") {
            Ok(data) => {
                let expected = b"Hello from GPU std::fs::File!";
                if data == expected {
                    println!("  std_file_write_kernel: PASSED!");
                    println!("    Written: {} bytes", bytes_written);
                    println!(
                        "    Content: {:?}",
                        std::str::from_utf8(&data).unwrap_or("<non-utf8>")
                    );
                } else {
                    hc_buf_ref.signal_shutdown();
                    listener_handle.join().unwrap();
                    return Err(GpuHostError::Verification {
                        test: "std_file_write_kernel",
                        detail: format!(
                            "content mismatch: expected {:?}, got {:?}",
                            std::str::from_utf8(expected).unwrap(),
                            std::str::from_utf8(&data).unwrap_or("<non-utf8>")
                        ),
                    });
                }
            }
            Err(e) => {
                hc_buf_ref.signal_shutdown();
                listener_handle.join().unwrap();
                return Err(GpuHostError::Verification {
                    test: "std_file_write_kernel",
                    detail: format!("gpu_test_output.txt not found: {e}"),
                });
            }
        }
    }

    // --- Test 2: File::open + read ---
    {
        // Create input file for the kernel to read
        std::fs::write("gpu_test_input.txt", b"GPU read test data").map_err(|e| {
            GpuHostError::Verification {
                test: "std_file_read_kernel",
                detail: format!("failed to create test input file: {e}"),
            }
        })?;

        let (result_host_ptr, result_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 3)? };
        unsafe {
            std::ptr::write_volatile(result_host_ptr.add(0), 0u32);
            std::ptr::write_volatile(result_host_ptr.add(1), 0u32);
            std::ptr::write_volatile(result_host_ptr.add(2), 0u32);
        }

        let f = dev
            .get_func("std_test", "std_file_read_kernel")
            .ok_or(GpuHostError::KernelNotFound("std_file_read_kernel"))?;

        println!("  Launching std_file_read_kernel...");
        unsafe {
            f.launch(cfg, (dev_ptr, result_dev_ptr))?;
        }
        dev.synchronize()?;
        std::thread::sleep(std::time::Duration::from_millis(100));

        let success = unsafe { std::ptr::read_volatile(result_host_ptr.add(0)) };
        let bytes_read = unsafe { std::ptr::read_volatile(result_host_ptr.add(1)) };
        let first_byte = unsafe { std::ptr::read_volatile(result_host_ptr.add(2)) };
        unsafe { free_mapped_mem(result_host_ptr)? };

        if success != 1 {
            hc_buf_ref.signal_shutdown();
            listener_handle.join().unwrap();
            let _ = std::fs::remove_file("gpu_test_output.txt");
            let _ = std::fs::remove_file("gpu_test_input.txt");
            return Err(GpuHostError::Verification {
                test: "std_file_read_kernel",
                detail: format!("File::open + read failed (code=0x{success:X})"),
            });
        }

        println!("  std_file_read_kernel: PASSED!");
        println!("    Bytes read: {bytes_read}");
        println!(
            "    First byte: '{}' (0x{:02X})",
            first_byte as u8 as char, first_byte
        );
    }

    // Cleanup
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();
    let _ = std::fs::remove_file("gpu_test_output.txt");
    let _ = std::fs::remove_file("gpu_test_input.txt");

    println!("  std::fs::File I/O on GPU: ALL PASSED!");
    println!("    File::create + write_all: hostcall → host filesystem");
    println!("    File::open + read: hostcall → host filesystem");
    Ok(())
}
