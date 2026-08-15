//! Basic kernel + atomics tests.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

use gpu_host::error::{GpuHostError, Result};
use gpu_host::mapped_mem::{alloc_mapped_u32, free_mapped_mem};

pub(crate) fn run_write_thread_idx(dev: Arc<CudaDevice>) -> Result<()> {
    const N: usize = 64;

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_CORE_PTX);
    dev.load_ptx(ptx, "kernel", &["write_thread_idx"])?;

    let f = dev
        .get_func("kernel", "write_thread_idx")
        .ok_or(GpuHostError::KernelNotFound("write_thread_idx"))?;

    let mut output: CudaSlice<u32> = dev.alloc_zeros::<u32>(N)?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (N as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        f.launch(cfg, (&mut output, N as u32))?;
    }

    let result: Vec<u32> = dev.dtoh_sync_copy(&output)?;

    println!("write_thread_idx output ({N} elements):");
    println!("  {:?}", &result[..N.min(16)]);

    for (i, &val) in result.iter().enumerate() {
        if val != i as u32 {
            return Err(GpuHostError::Verification {
                test: "write_thread_idx",
                detail: format!("index {i}: expected {i}, got {val}"),
            });
        }
    }
    println!("  Verification PASSED: all {N} elements correct");
    Ok(())
}

pub(crate) fn run_vector_add(dev: Arc<CudaDevice>) -> Result<()> {
    const N: usize = 128;

    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (N - i) as f32).collect();

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_CORE_PTX);
    dev.load_ptx(ptx, "kernel", &["vector_add"])?;

    let f = dev
        .get_func("kernel", "vector_add")
        .ok_or(GpuHostError::KernelNotFound("vector_add"))?;

    let a_dev: CudaSlice<f32> = dev.htod_sync_copy(&a_host)?;
    let b_dev: CudaSlice<f32> = dev.htod_sync_copy(&b_host)?;
    let mut c_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(N)?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (N as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        f.launch(cfg, (&a_dev, &b_dev, &mut c_dev, N as u32))?;
    }

    let c_host: Vec<f32> = dev.dtoh_sync_copy(&c_dev)?;

    println!("\nvector_add output (first 16 of {N} elements):");
    println!("  {:?}", &c_host[..16]);

    let expected = N as f32;
    for (i, &val) in c_host.iter().enumerate() {
        if (val - expected).abs() > 1e-5 {
            return Err(GpuHostError::Verification {
                test: "vector_add",
                detail: format!("index {i}: expected {expected}, got {val}"),
            });
        }
    }
    println!("  Verification PASSED: all {N} elements equal {expected}");
    Ok(())
}

/// Step 1 + Step 3: Test inline PTX asm kernels and inspect their PTX output.
pub(crate) fn run_asm_smoke_tests(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Step 1 / Step 3: Inline PTX asm smoke tests ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_CORE_PTX);
    dev.load_ptx(
        ptx,
        "kernel",
        &[
            "test_asm_membar_sys",
            "test_asm_st_release_sys",
            "test_asm_ld_acquire_sys",
            "test_asm_cas_sys",
            "test_read_volatile",
            "test_write_volatile",
        ],
    )?;

    // test_asm_membar_sys
    {
        let f = dev
            .get_func("kernel", "test_asm_membar_sys")
            .ok_or(GpuHostError::KernelNotFound("test_asm_membar_sys"))?;
        let mut out: CudaSlice<u32> = dev.alloc_zeros::<u32>(4)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (4, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { f.launch(cfg, (&mut out, 4u32))? };
        let result = dev.dtoh_sync_copy(&out)?;
        for &v in &result {
            if v != 0xDEAD_BEEFu32 {
                return Err(GpuHostError::Verification {
                    test: "test_asm_membar_sys",
                    detail: format!("expected 0xDEADBEEF, got 0x{v:08X}"),
                });
            }
        }
        println!(
            "  test_asm_membar_sys: PASSED (membar.sys + st.global.b32 works, result = 0xDEADBEEF)"
        );
    }

    // test_asm_st_release_sys
    {
        let f = dev
            .get_func("kernel", "test_asm_st_release_sys")
            .ok_or(GpuHostError::KernelNotFound("test_asm_st_release_sys"))?;
        let mut out: CudaSlice<u32> = dev.alloc_zeros::<u32>(1)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { f.launch(cfg, (&mut out, 42u32))? };
        let result = dev.dtoh_sync_copy(&out)?;
        if result[0] != 42 {
            return Err(GpuHostError::Verification {
                test: "test_asm_st_release_sys",
                detail: format!("expected 42, got {}", result[0]),
            });
        }
        println!("  test_asm_st_release_sys: PASSED (st.release.sys.global.u32 works, wrote 42)");
    }

    // test_asm_ld_acquire_sys
    {
        let f = dev
            .get_func("kernel", "test_asm_ld_acquire_sys")
            .ok_or(GpuHostError::KernelNotFound("test_asm_ld_acquire_sys"))?;
        let src: CudaSlice<u32> = dev.htod_sync_copy(&[99u32])?;
        let mut dst: CudaSlice<u32> = dev.alloc_zeros::<u32>(1)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { f.launch(cfg, (&src, &mut dst))? };
        let result = dev.dtoh_sync_copy(&dst)?;
        if result[0] != 99 {
            return Err(GpuHostError::Verification {
                test: "test_asm_ld_acquire_sys",
                detail: format!("expected 99, got {}", result[0]),
            });
        }
        println!("  test_asm_ld_acquire_sys: PASSED (ld.acquire.sys.global.u32 works, read 99)");
    }

    // test_asm_cas_sys
    {
        let f = dev
            .get_func("kernel", "test_asm_cas_sys")
            .ok_or(GpuHostError::KernelNotFound("test_asm_cas_sys"))?;
        let mut target: CudaSlice<u32> = dev.htod_sync_copy(&[7u32])?;
        let mut result_out: CudaSlice<u32> = dev.alloc_zeros::<u32>(1)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            f.launch(cfg, (&mut target, 7u32, 99u32, &mut result_out))?;
        }
        let old_val = dev.dtoh_sync_copy(&result_out)?[0];
        let new_val = dev.dtoh_sync_copy(&target)?[0];
        if old_val != 7 || new_val != 99 {
            return Err(GpuHostError::Verification {
                test: "test_asm_cas_sys",
                detail: format!("expected old=7 new=99, got old={old_val} new={new_val}"),
            });
        }
        println!("  test_asm_cas_sys: PASSED (atom.cas.sys.global.b32 works, old=7→new=99)");
    }

    // test_read_volatile / test_write_volatile
    {
        let f_write = dev
            .get_func("kernel", "test_write_volatile")
            .ok_or(GpuHostError::KernelNotFound("test_write_volatile"))?;
        let f_read = dev
            .get_func("kernel", "test_read_volatile")
            .ok_or(GpuHostError::KernelNotFound("test_read_volatile"))?;

        let mut buf: CudaSlice<u32> = dev.alloc_zeros::<u32>(1)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            f_write.launch(cfg, (&mut buf, 0xCAFE_BABEu32))?;
        }

        let mut dst: CudaSlice<u32> = dev.alloc_zeros::<u32>(1)?;
        unsafe {
            f_read.launch(cfg, (&buf, &mut dst))?;
        }

        let result = dev.dtoh_sync_copy(&dst)?;
        if result[0] != 0xCAFE_BABEu32 {
            return Err(GpuHostError::Verification {
                test: "test_read/write_volatile",
                detail: format!("expected 0xCAFEBABE, got 0x{:08X}", result[0]),
            });
        }
        println!("  test_write_volatile + test_read_volatile: PASSED");
        println!(
            "    st.volatile.global.b32 wrote 0xCAFEBABE, ld.volatile.global.b32 read it back"
        );
    }

    println!("  All asm smoke tests PASSED.");
    Ok(())
}

/// Step 5: Integration test using mapped host memory for GPU-CPU communication.
pub(crate) fn run_integration_sys_store(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Step 5: Integration test (GPU st.release.sys → CPU poll) ---");

    const EXPECTED_VALUE: u32 = 0xABCD_1234;
    const TIMEOUT_ITERS: usize = 100_000_000;

    let (data_host_ptr, data_dev_ptr) = unsafe { alloc_mapped_u32(&dev)? };
    let (flag_host_ptr, flag_dev_ptr) = unsafe { alloc_mapped_u32(&dev)? };

    unsafe {
        std::ptr::write_volatile(data_host_ptr, 0u32);
        std::ptr::write_volatile(flag_host_ptr, 0u32);
    }

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_CORE_PTX);
    dev.load_ptx(ptx, "kernel", &["integration_sys_store"])?;
    let f = dev
        .get_func("kernel", "integration_sys_store")
        .ok_or(GpuHostError::KernelNotFound("integration_sys_store"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching GPU kernel (thread 0 writes data + flag to pinned memory)...");
    unsafe {
        let data_u64 = data_dev_ptr;
        let flag_u64 = flag_dev_ptr;
        f.launch(cfg, (data_u64, flag_u64, EXPECTED_VALUE))?;
    }

    println!("  Host polling flag (with acquire semantics via AtomicU32::load)...");
    let flag_atomic = unsafe { &*(flag_host_ptr as *const AtomicU32) };
    let data_atomic = unsafe { &*(data_host_ptr as *const AtomicU32) };

    let mut flag_val = 0u32;
    for i in 0..TIMEOUT_ITERS {
        flag_val = flag_atomic.load(Ordering::Acquire);
        if flag_val == 1 {
            println!("  Flag became 1 after {i} poll iterations.");
            break;
        }
        std::hint::spin_loop();
        if i % 10_000_000 == 0 && i > 0 {
            println!("  ... still polling, iteration {i}");
        }
    }

    if flag_val != 1 {
        dev.synchronize()?;
        unsafe {
            free_mapped_mem(data_host_ptr)?;
            free_mapped_mem(flag_host_ptr)?;
        }
        return Err(GpuHostError::Timeout {
            test: "integration_sys_store",
            detail: format!("flag never became 1 after {TIMEOUT_ITERS} iters"),
        });
    }

    let data_val = data_atomic.load(Ordering::Acquire);

    unsafe {
        free_mapped_mem(data_host_ptr)?;
        free_mapped_mem(flag_host_ptr)?;
    }

    if data_val != EXPECTED_VALUE {
        return Err(GpuHostError::Verification {
            test: "integration_sys_store",
            detail: format!("expected data=0x{EXPECTED_VALUE:08X}, got 0x{data_val:08X}"),
        });
    }

    println!("  Integration test PASSED!");
    println!("    GPU wrote data=0x{EXPECTED_VALUE:08X} then set flag=1 via st.release.sys");
    println!("    CPU saw flag=1 and read data=0x{data_val:08X} correctly");
    println!("    Protocol: GPU st.release.sys → CPU Ordering::Acquire poll works correctly");
    Ok(())
}

/// Step 6: u64 atomics smoke tests (atomics.4).
pub(crate) fn run_u64_atomics_tests(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Step 6: u64 atomics smoke tests ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_CORE_PTX);
    dev.load_ptx(
        ptx,
        "kernel",
        &["test_u64_cas", "test_u64_fetch_add", "test_u64_exchange"],
    )?;

    // test_u64_cas
    {
        let f = dev
            .get_func("kernel", "test_u64_cas")
            .ok_or(GpuHostError::KernelNotFound("test_u64_cas"))?;
        let initial: u64 = 0x0000_0007_0000_0003;
        let expected: u64 = initial;
        let desired: u64 = 0x0000_0099_0000_0042;
        let mut target: CudaSlice<u64> = dev.htod_sync_copy(&[initial])?;
        let mut result_out: CudaSlice<u64> = dev.alloc_zeros::<u64>(1)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let expected_lo = expected as u32;
        let expected_hi = (expected >> 32) as u32;
        let desired_lo = desired as u32;
        let desired_hi = (desired >> 32) as u32;
        unsafe {
            f.launch(
                cfg,
                (
                    &mut target,
                    expected_lo,
                    expected_hi,
                    desired_lo,
                    desired_hi,
                    &mut result_out,
                ),
            )?;
        }
        let old_val = dev.dtoh_sync_copy(&result_out)?[0];
        let new_val = dev.dtoh_sync_copy(&target)?[0];
        if old_val != initial {
            return Err(GpuHostError::Verification {
                test: "test_u64_cas",
                detail: format!("expected old=0x{initial:016X}, got 0x{old_val:016X}"),
            });
        }
        if new_val != desired {
            return Err(GpuHostError::Verification {
                test: "test_u64_cas",
                detail: format!("expected new=0x{desired:016X}, got 0x{new_val:016X}"),
            });
        }
        println!(
            "  test_u64_cas: PASSED (atom.cas.sys.global.b64 works, 0x{initial:016X}→0x{desired:016X})"
        );
    }

    // test_u64_fetch_add
    {
        let f = dev
            .get_func("kernel", "test_u64_fetch_add")
            .ok_or(GpuHostError::KernelNotFound("test_u64_fetch_add"))?;
        let initial: u64 = 0x0000_0001_0000_0000;
        let addend: u64 = 0x0000_0000_FFFF_FFFF;
        let expected_new: u64 = initial + addend;
        let mut target: CudaSlice<u64> = dev.htod_sync_copy(&[initial])?;
        let mut result_out: CudaSlice<u64> = dev.alloc_zeros::<u64>(1)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let val_lo = addend as u32;
        let val_hi = (addend >> 32) as u32;
        unsafe {
            f.launch(cfg, (&mut target, val_lo, val_hi, &mut result_out))?;
        }
        let old_val = dev.dtoh_sync_copy(&result_out)?[0];
        let new_val = dev.dtoh_sync_copy(&target)?[0];
        if old_val != initial || new_val != expected_new {
            return Err(GpuHostError::Verification {
                test: "test_u64_fetch_add",
                detail: format!("old=0x{old_val:016X} new=0x{new_val:016X}"),
            });
        }
        println!("  test_u64_fetch_add: PASSED (atom.add.sys.global.u64 works, 0x{initial:016X}+0x{addend:016X}=0x{expected_new:016X})");
    }

    // test_u64_exchange
    {
        let f = dev
            .get_func("kernel", "test_u64_exchange")
            .ok_or(GpuHostError::KernelNotFound("test_u64_exchange"))?;
        let initial: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let new_value: u64 = 0x1234_5678_9ABC_DEF0;
        let mut target: CudaSlice<u64> = dev.htod_sync_copy(&[initial])?;
        let mut result_out: CudaSlice<u64> = dev.alloc_zeros::<u64>(1)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let val_lo = new_value as u32;
        let val_hi = (new_value >> 32) as u32;
        unsafe {
            f.launch(cfg, (&mut target, val_lo, val_hi, &mut result_out))?;
        }
        let old_val = dev.dtoh_sync_copy(&result_out)?[0];
        let final_val = dev.dtoh_sync_copy(&target)?[0];
        if old_val != initial || final_val != new_value {
            return Err(GpuHostError::Verification {
                test: "test_u64_exchange",
                detail: format!("old=0x{old_val:016X} final=0x{final_val:016X}"),
            });
        }
        println!(
            "  test_u64_exchange: PASSED (atom.exch.sys.global.b64 works, 0x{initial:016X}→0x{new_value:016X})"
        );
    }

    println!("  All u64 atomics smoke tests PASSED.");
    Ok(())
}

/// Step 7: Spin-load + warp intrinsics tests (atomics.4).
pub(crate) fn run_warp_intrinsics_tests(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Step 7: Spin-load + warp intrinsics tests ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_CORE_PTX);
    dev.load_ptx(
        ptx,
        "kernel",
        &["test_spin_load_u32", "test_activemask", "test_lane_id"],
    )?;

    // test_spin_load_u32
    {
        let f = dev
            .get_func("kernel", "test_spin_load_u32")
            .ok_or(GpuHostError::KernelNotFound("test_spin_load_u32"))?;
        let src: CudaSlice<u32> = dev.htod_sync_copy(&[0x42u32])?;
        let mut dst: CudaSlice<u32> = dev.alloc_zeros::<u32>(1)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            f.launch(cfg, (&src, &mut dst))?;
        }
        let result = dev.dtoh_sync_copy(&dst)?[0];
        if result != 0x42 {
            return Err(GpuHostError::Verification {
                test: "test_spin_load_u32",
                detail: format!("expected 0x42, got 0x{result:08X}"),
            });
        }
        println!("  test_spin_load_u32: PASSED (ld.acquire.sys + nanosleep works, read 0x42)");
    }

    // test_activemask: full warp
    {
        let f = dev
            .get_func("kernel", "test_activemask")
            .ok_or(GpuHostError::KernelNotFound("test_activemask"))?;
        let n: u32 = 32;
        let mut out: CudaSlice<u32> = dev.alloc_zeros::<u32>(n as usize)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (n, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            f.launch(cfg, (&mut out, n))?;
        }
        let result = dev.dtoh_sync_copy(&out)?;
        for (i, &mask) in result.iter().enumerate() {
            if mask != 0xFFFF_FFFF {
                return Err(GpuHostError::Verification {
                    test: "test_activemask",
                    detail: format!("thread {i} got mask=0x{mask:08X}, expected 0xFFFFFFFF"),
                });
            }
        }
        println!("  test_activemask: PASSED (activemask.b32 returns 0xFFFFFFFF for full warp)");
    }

    // test_activemask: partial warp (20 threads)
    {
        let f = dev
            .get_func("kernel", "test_activemask")
            .ok_or(GpuHostError::KernelNotFound("test_activemask"))?;
        let n: u32 = 20;
        let mut out: CudaSlice<u32> = dev.alloc_zeros::<u32>(n as usize)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (n, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            f.launch(cfg, (&mut out, n))?;
        }
        let result = dev.dtoh_sync_copy(&out)?;
        let expected_mask: u32 = (1u32 << 20) - 1;
        let actual_mask = result[0];
        println!(
            "  test_activemask (partial, 20 threads): mask=0x{actual_mask:08X} (expected ~0x{expected_mask:08X})"
        );
        if actual_mask == expected_mask {
            println!("    Exact match: only 20 lanes active (hardware launched partial warp)");
        } else if actual_mask == 0xFFFF_FFFF {
            println!("    Full warp launched (32 lanes), threads 20-31 predicated off by if-guard");
        }
    }

    // test_lane_id
    {
        let f = dev
            .get_func("kernel", "test_lane_id")
            .ok_or(GpuHostError::KernelNotFound("test_lane_id"))?;
        let n: u32 = 32;
        let mut out: CudaSlice<u32> = dev.alloc_zeros::<u32>(n as usize)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (n, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            f.launch(cfg, (&mut out, n))?;
        }
        let result = dev.dtoh_sync_copy(&out)?;
        for (i, &lid) in result.iter().enumerate() {
            if lid != i as u32 {
                return Err(GpuHostError::Verification {
                    test: "test_lane_id",
                    detail: format!("thread {i} got lane_id={lid}, expected {i}"),
                });
            }
        }
        println!("  test_lane_id: PASSED (lane_id returns 0..31 for single-warp block)");
    }

    println!("  All warp intrinsics tests PASSED.");
    Ok(())
}
