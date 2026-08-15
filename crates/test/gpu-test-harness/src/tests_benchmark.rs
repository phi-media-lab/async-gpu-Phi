//! Benchmark tests: hostcall latency, warp divergence, sharding, throughput, scalability.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};

use crate::bench_harness::{self, BenchmarkResult};
use gpu_host::error::{GpuHostError, Result};
use gpu_host::hostcall;
use gpu_host::mapped_mem::{alloc_mapped_u64_array, free_mapped_u64_array};

/// Compute percentile from a sorted slice (nearest-rank method).
pub(super) fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (p / 100.0 * sorted.len() as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(sorted.len() - 1);
    sorted[idx]
}

/// Run the hostcall latency benchmark at a specific thread count.
fn run_latency_bench_config(
    dev: &Arc<CudaDevice>,
    num_threads: u32,
    num_iters: u32,
    num_packets: u16,
) -> Result<()> {
    use std::sync::Arc as StdArc;

    let hc_buf = hostcall::HostcallBuffer::new(num_packets)?;
    let dev_ptr = hc_buf.dev_ptr();

    // Allocate results: 3 u64 per thread
    let results_count = (num_threads as usize) * 3;
    let (results_host_ptr, results_dev_ptr) =
        unsafe { alloc_mapped_u64_array(dev, results_count)? };

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|_msg| {
            // NOP service — no print messages expected
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_IO_PTX);
    dev.load_ptx(ptx, "kernel", &["hostcall_latency_bench"])?;
    let f = dev
        .get_func("kernel", "hostcall_latency_bench")
        .ok_or(GpuHostError::KernelNotFound("hostcall_latency_bench"))?;

    // Launch: 1 block with num_threads threads (up to 1024)
    let block_dim = num_threads.min(1024);
    let grid_dim = num_threads.div_ceil(block_dim);
    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, results_dev_ptr, num_iters))?;
    }
    dev.synchronize()?;
    let wall_elapsed = start.elapsed();

    // Give listener time to finish processing
    std::thread::sleep(std::time::Duration::from_millis(50));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    // Read results and compute statistics
    let mut latencies_ns: Vec<f64> = Vec::new();
    let mut total_retries: u64 = 0;
    let mut total_completed: u64 = 0;

    for tid in 0..num_threads as usize {
        let elapsed_ns = unsafe { std::ptr::read_volatile(results_host_ptr.add(tid * 3)) };
        let retries = unsafe { std::ptr::read_volatile(results_host_ptr.add(tid * 3 + 1)) };
        let completed = unsafe { std::ptr::read_volatile(results_host_ptr.add(tid * 3 + 2)) };

        total_retries += retries;
        total_completed += completed;

        if completed > 0 {
            let avg_ns = elapsed_ns as f64 / completed as f64;
            latencies_ns.push(avg_ns);
        }
    }

    latencies_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = percentile(&latencies_ns, 50.0);
    let p95 = percentile(&latencies_ns, 95.0);
    let p99 = percentile(&latencies_ns, 99.0);
    let mean = if latencies_ns.is_empty() {
        0.0
    } else {
        latencies_ns.iter().sum::<f64>() / latencies_ns.len() as f64
    };

    let total_hostcalls = total_completed;
    let throughput = if wall_elapsed.as_secs_f64() > 0.0 {
        total_hostcalls as f64 / wall_elapsed.as_secs_f64()
    } else {
        0.0
    };
    let cas_retry_rate = if total_completed > 0 {
        total_retries as f64 / total_completed as f64
    } else {
        0.0
    };

    println!(
        "    threads={:<4} iters={:<4} packets={:<3} | \
         p50={:>10.0}ns  p95={:>10.0}ns  p99={:>10.0}ns  mean={:>10.0}ns | \
         CAS retries/call={:.2}  throughput={:.0} calls/s  \
         completed={}/{}  wall={:.1}ms",
        num_threads,
        num_iters,
        num_packets,
        p50,
        p95,
        p99,
        mean,
        cas_retry_rate,
        throughput,
        total_completed,
        num_threads as u64 * num_iters as u64,
        wall_elapsed.as_secs_f64() * 1000.0,
    );

    unsafe { free_mapped_u64_array(results_host_ptr)? };

    Ok(())
}

/// Run the full hostcall latency benchmark suite.
pub(crate) fn run_hostcall_latency_benchmark(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Hostcall Latency Benchmark (benchmark.2) ---");
    println!("  Each thread performs N sequential NOP hostcalls with globaltimer timing.");
    println!("  Latency = per-thread average round-trip time.\n");

    let num_iters = 10;

    // Test matrix: thread counts × packet pool sizes
    let thread_counts = [1, 32, 128, 512];
    let packet_multipliers = [2, 4]; // packets = multiplier × thread_count (capped at 64)

    for &threads in &thread_counts {
        for &mult in &packet_multipliers {
            let packets = ((threads * mult) as u16).clamp(4, 64);
            run_latency_bench_config(&dev, threads, num_iters, packets)?;
        }
    }

    println!("\n  Benchmark complete.");
    Ok(())
}

/// Run the latency bench kernel with explicit grid/block dimensions.
/// Returns (wall_elapsed_ms, total_completed, total_retries, per_thread_latencies_ns).
pub(super) fn run_latency_bench_explicit(
    dev: &Arc<CudaDevice>,
    grid_dim: u32,
    block_dim: u32,
    num_iters: u32,
    num_packets: u16,
) -> Result<(f64, u64, u64, Vec<f64>)> {
    use std::sync::Arc as StdArc;

    let num_threads = grid_dim * block_dim;
    let hc_buf = hostcall::HostcallBuffer::new(num_packets)?;
    let dev_ptr = hc_buf.dev_ptr();

    let results_count = (num_threads as usize) * 3;
    let (results_host_ptr, results_dev_ptr) =
        unsafe { alloc_mapped_u64_array(dev, results_count)? };

    let hc_buf_ref = StdArc::new(hc_buf);
    let hc_buf_listener = StdArc::clone(&hc_buf_ref);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|_msg| {});
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_IO_PTX);
    dev.load_ptx(ptx, "kernel", &["hostcall_latency_bench"])?;
    let f = dev
        .get_func("kernel", "hostcall_latency_bench")
        .ok_or(GpuHostError::KernelNotFound("hostcall_latency_bench"))?;

    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, results_dev_ptr, num_iters))?;
    }
    dev.synchronize()?;
    let wall_elapsed = start.elapsed();

    std::thread::sleep(std::time::Duration::from_millis(50));
    hc_buf_ref.signal_shutdown();
    listener_handle.join().unwrap();

    let mut latencies_ns: Vec<f64> = Vec::new();
    let mut total_retries: u64 = 0;
    let mut total_completed: u64 = 0;

    for tid in 0..num_threads as usize {
        let elapsed_ns = unsafe { std::ptr::read_volatile(results_host_ptr.add(tid * 3)) };
        let retries = unsafe { std::ptr::read_volatile(results_host_ptr.add(tid * 3 + 1)) };
        let completed = unsafe { std::ptr::read_volatile(results_host_ptr.add(tid * 3 + 2)) };

        total_retries += retries;
        total_completed += completed;

        if completed > 0 {
            let avg_ns = elapsed_ns as f64 / completed as f64;
            latencies_ns.push(avg_ns);
        }
    }

    unsafe { free_mapped_u64_array(results_host_ptr)? };

    Ok((
        wall_elapsed.as_secs_f64() * 1000.0,
        total_completed,
        total_retries,
        latencies_ns,
    ))
}

/// Warp divergence measurement (warp-future.2):
/// Compare 1 block x 32 threads (intra-warp divergence) vs
/// 32 blocks x 1 thread (no intra-warp divergence) for hostcall NOP round-trips.
pub(crate) fn run_warp_divergence_measurement(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Warp Divergence Measurement (warp-future.2) ---");
    println!("  Comparing hostcall throughput: 1x32 (one warp) vs 32x1 (32 blocks)");
    println!("  Same total threads (32), same packet pool, same iterations.\n");

    let num_iters = 20;
    let num_packets: u16 = 32; // enough for all 32 threads

    // Warm-up run (1x1, 5 iters) to ensure PTX is loaded and JIT'd
    let _ = run_latency_bench_explicit(&dev, 1, 1, 5, 4);

    // Config A: 1 block x 32 threads → all threads in one warp
    // Expected: intra-warp divergence when threads are in different hostcall phases
    let mut a_wall_total = 0.0;
    let mut a_completed_total = 0u64;
    let mut a_retries_total = 0u64;
    let mut a_latencies_all: Vec<f64> = Vec::new();
    const RUNS: usize = 3;

    for _ in 0..RUNS {
        let (wall_ms, completed, retries, latencies) =
            run_latency_bench_explicit(&dev, 1, 32, num_iters, num_packets)?;
        a_wall_total += wall_ms;
        a_completed_total += completed;
        a_retries_total += retries;
        a_latencies_all.extend_from_slice(&latencies);
    }

    // Config B: 32 blocks x 1 thread → no intra-warp divergence
    // Expected: each thread is alone in its warp, no divergence
    let mut b_wall_total = 0.0;
    let mut b_completed_total = 0u64;
    let mut b_retries_total = 0u64;
    let mut b_latencies_all: Vec<f64> = Vec::new();

    for _ in 0..RUNS {
        let (wall_ms, completed, retries, latencies) =
            run_latency_bench_explicit(&dev, 32, 1, num_iters, num_packets)?;
        b_wall_total += wall_ms;
        b_completed_total += completed;
        b_retries_total += retries;
        b_latencies_all.extend_from_slice(&latencies);
    }

    // Compute statistics
    a_latencies_all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    b_latencies_all.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let a_mean = if a_latencies_all.is_empty() {
        0.0
    } else {
        a_latencies_all.iter().sum::<f64>() / a_latencies_all.len() as f64
    };
    let b_mean = if b_latencies_all.is_empty() {
        0.0
    } else {
        b_latencies_all.iter().sum::<f64>() / b_latencies_all.len() as f64
    };

    let a_p50 = percentile(&a_latencies_all, 50.0);
    let b_p50 = percentile(&b_latencies_all, 50.0);

    let a_throughput = if a_wall_total > 0.0 {
        a_completed_total as f64 / (a_wall_total / 1000.0)
    } else {
        0.0
    };
    let b_throughput = if b_wall_total > 0.0 {
        b_completed_total as f64 / (b_wall_total / 1000.0)
    } else {
        0.0
    };

    let a_cas = if a_completed_total > 0 {
        a_retries_total as f64 / a_completed_total as f64
    } else {
        0.0
    };
    let b_cas = if b_completed_total > 0 {
        b_retries_total as f64 / b_completed_total as f64
    } else {
        0.0
    };

    println!("  Config A (1 block x 32 threads — one warp, intra-warp divergence):");
    println!(
        "    wall={:.1}ms  completed={}/{}  mean={:.0}ns  p50={:.0}ns  CAS/call={:.2}  throughput={:.0}/s",
        a_wall_total / RUNS as f64,
        a_completed_total / RUNS as u64,
        32 * num_iters,
        a_mean,
        a_p50,
        a_cas,
        a_throughput / RUNS as f64,
    );

    println!("\n  Config B (32 blocks x 1 thread — no intra-warp divergence):");
    println!(
        "    wall={:.1}ms  completed={}/{}  mean={:.0}ns  p50={:.0}ns  CAS/call={:.2}  throughput={:.0}/s",
        b_wall_total / RUNS as f64,
        b_completed_total / RUNS as u64,
        32 * num_iters,
        b_mean,
        b_p50,
        b_cas,
        b_throughput / RUNS as f64,
    );

    println!("\n  --- Comparison ---");
    let throughput_ratio = if b_throughput > 0.0 {
        a_throughput / b_throughput
    } else {
        f64::NAN
    };
    let latency_ratio = if b_mean > 0.0 {
        a_mean / b_mean
    } else {
        f64::NAN
    };
    let cas_ratio = if b_cas > 0.0 { a_cas / b_cas } else { f64::NAN };

    println!("    Throughput ratio (A/B): {throughput_ratio:.2}x");
    println!("    Latency ratio (A/B):   {latency_ratio:.2}x");
    println!("    CAS retry ratio (A/B): {cas_ratio:.2}x");

    if throughput_ratio >= 0.8 {
        println!("\n  RESULT: Intra-warp divergence has MINIMAL impact (<20% throughput loss).");
        println!("  WarpFuture optimization may not be justified for hostcall-bound workloads.");
    } else if throughput_ratio >= 0.5 {
        println!(
            "\n  RESULT: Intra-warp divergence causes MODERATE impact ({:.0}% throughput loss).",
            (1.0 - throughput_ratio) * 100.0
        );
        println!("  WarpFuture could provide meaningful improvement.");
    } else {
        println!(
            "\n  RESULT: Intra-warp divergence causes SEVERE impact ({:.0}% throughput loss).",
            (1.0 - throughput_ratio) * 100.0
        );
        println!("  WarpFuture is strongly justified.");
    }

    println!("\n  Measurement complete.");
    Ok(())
}

/// Run a single sharding benchmark config using hostcall_latency_bench_v2.
/// Returns (wall_ms, total_completed, total_retries, per_thread_latencies_ns).
pub(super) fn run_sharding_bench_config(
    dev: &Arc<CudaDevice>,
    grid_dim: u32,
    block_dim: u32,
    num_iters: u32,
    hc_buf: &std::sync::Arc<hostcall::HostcallBuffer>,
) -> Result<(f64, u64, u64, Vec<f64>)> {
    let num_threads = grid_dim * block_dim;
    let dev_ptr = hc_buf.dev_ptr();

    let results_count = (num_threads as usize) * 3;
    let (results_host_ptr, results_dev_ptr) =
        unsafe { alloc_mapped_u64_array(dev, results_count)? };

    let hc_buf_listener = std::sync::Arc::clone(hc_buf);
    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|_msg| {});
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_IO_PTX);
    dev.load_ptx(ptx, "kernel_bench_v2", &["hostcall_latency_bench_v2"])?;
    let f = dev
        .get_func("kernel_bench_v2", "hostcall_latency_bench_v2")
        .ok_or(GpuHostError::KernelNotFound("hostcall_latency_bench_v2"))?;

    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, results_dev_ptr, num_iters))?;
    }
    dev.synchronize()?;
    let wall_elapsed = start.elapsed();

    std::thread::sleep(std::time::Duration::from_millis(50));
    hc_buf.signal_shutdown();
    listener_handle.join().unwrap();

    let mut latencies_ns: Vec<f64> = Vec::new();
    let mut total_retries: u64 = 0;
    let mut total_completed: u64 = 0;

    for tid in 0..num_threads as usize {
        let elapsed_ns = unsafe { std::ptr::read_volatile(results_host_ptr.add(tid * 3)) };
        let retries = unsafe { std::ptr::read_volatile(results_host_ptr.add(tid * 3 + 1)) };
        let completed = unsafe { std::ptr::read_volatile(results_host_ptr.add(tid * 3 + 2)) };

        total_retries += retries;
        total_completed += completed;

        if completed > 0 {
            let avg_ns = elapsed_ns as f64 / completed as f64;
            latencies_ns.push(avg_ns);
        }
    }

    unsafe { free_mapped_u64_array(results_host_ptr)? };

    Ok((
        wall_elapsed.as_secs_f64() * 1000.0,
        total_completed,
        total_retries,
        latencies_ns,
    ))
}

/// Benchmark: compare sharded vs unsharded hostcall pools at various thread counts.
pub(crate) fn run_sharding_benchmark(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Sharding Benchmark (per-block-sharding.3) ---");
    println!("  Comparing unsharded (global pool) vs sharded (per-block pool)");
    println!("  using NOP hostcalls with CAS retry counting.\n");

    let num_iters: u32 = 10;

    // Test configs: (num_blocks, threads_per_block, total_threads)
    let configs: &[(u32, u32)] = &[
        (1, 32),  // 32 threads, 1 block
        (4, 32),  // 128 threads, 4 blocks
        (16, 32), // 512 threads, 16 blocks
    ];

    for &(num_blocks, block_dim) in configs {
        let total_threads = num_blocks * block_dim;
        let packets_per_thread = 2u32;
        let total_packets = (total_threads * packets_per_thread).min(64) as u16;

        println!("  --- {num_blocks} blocks × {block_dim} threads = {total_threads} total ---");

        // Baseline: unsharded (global pool)
        let hc_buf_global = std::sync::Arc::new(hostcall::HostcallBuffer::new(total_packets)?);
        let (g_wall, g_completed, g_retries, mut g_latencies) =
            run_sharding_bench_config(&dev, num_blocks, block_dim, num_iters, &hc_buf_global)?;

        g_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let g_p50 = percentile(&g_latencies, 50.0);
        let g_p95 = percentile(&g_latencies, 95.0);
        let g_cas = if g_completed > 0 {
            g_retries as f64 / g_completed as f64
        } else {
            0.0
        };
        let g_throughput = if g_wall > 0.0 {
            g_completed as f64 / (g_wall / 1000.0)
        } else {
            0.0
        };

        // Sharded: one shard per block
        let pkts_per_shard = packets_per_thread;
        let hc_buf_sharded = std::sync::Arc::new(hostcall::HostcallBuffer::new_sharded(
            num_blocks,
            pkts_per_shard,
        )?);
        let (s_wall, s_completed, s_retries, mut s_latencies) =
            run_sharding_bench_config(&dev, num_blocks, block_dim, num_iters, &hc_buf_sharded)?;

        s_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let s_p50 = percentile(&s_latencies, 50.0);
        let s_p95 = percentile(&s_latencies, 95.0);
        let s_cas = if s_completed > 0 {
            s_retries as f64 / s_completed as f64
        } else {
            0.0
        };
        let s_throughput = if s_wall > 0.0 {
            s_completed as f64 / (s_wall / 1000.0)
        } else {
            0.0
        };

        println!(
            "    GLOBAL:  p50={:>10.0}ns  p95={:>10.0}ns  CAS/call={:>6.2}  throughput={:>8.0}/s  completed={}/{}  wall={:.1}ms",
            g_p50, g_p95, g_cas, g_throughput, g_completed, total_threads as u64 * num_iters as u64, g_wall,
        );
        println!(
            "    SHARDED: p50={:>10.0}ns  p95={:>10.0}ns  CAS/call={:>6.2}  throughput={:>8.0}/s  completed={}/{}  wall={:.1}ms",
            s_p50, s_p95, s_cas, s_throughput, s_completed, total_threads as u64 * num_iters as u64, s_wall,
        );

        let speedup = if s_wall > 0.0 { g_wall / s_wall } else { 0.0 };
        let cas_reduction = if g_cas > 0.0 {
            (1.0 - s_cas / g_cas) * 100.0
        } else {
            0.0
        };
        println!(
            "    DELTA:   wall speedup={:.2}x  CAS reduction={:.0}%  latency ratio(p50)={:.2}x",
            speedup,
            cas_reduction,
            if s_p50 > 0.0 { g_p50 / s_p50 } else { 0.0 },
        );
        println!();
    }

    println!("  Sharding benchmark complete.");
    Ok(())
}

// ============================================================
// New benchmarks (bench-suite.2): throughput + scalability
// ============================================================

/// Run the v3 per-iteration latency kernel and collect results into BenchmarkResult.
fn run_v3_bench(
    dev: &Arc<CudaDevice>,
    name: &str,
    grid_dim: u32,
    block_dim: u32,
    num_iters: u32,
    num_packets: u16,
) -> Result<BenchmarkResult> {
    let num_threads = grid_dim * block_dim;

    crate::kernel_routes::load_kernel(
        dev,
        crate::kernel_routes::KernelModule::Io,
        "kernel_bench_v3",
        &["hostcall_latency_bench_v3"],
    )?;
    let f = dev
        .get_func("kernel_bench_v3", "hostcall_latency_bench_v3")
        .ok_or(GpuHostError::KernelNotFound("hostcall_latency_bench_v3"))?;

    let hc_buf = hostcall::HostcallBuffer::new(num_packets)?;
    let dev_ptr = hc_buf.dev_ptr();

    // Allocate results: header (3 u64/thread) + per-iter (1 u64/thread/iter)
    let header_count = (num_threads as usize) * 3;
    let iter_count = (num_threads as usize) * (num_iters as usize);
    let total_results = header_count + iter_count;
    let (results_host_ptr, results_dev_ptr) =
        unsafe { alloc_mapped_u64_array(dev, total_results)? };

    // Zero out the results buffer
    unsafe {
        std::ptr::write_bytes(results_host_ptr, 0, total_results);
    }

    let hc_buf_ref = Arc::new(hc_buf);
    let listener =
        crate::harness_support::HostcallListener::start(Arc::clone(&hc_buf_ref), |_msg| {});

    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, results_dev_ptr, num_iters, num_threads))?;
    }
    dev.synchronize()?;
    let wall_elapsed = start.elapsed();

    std::thread::sleep(std::time::Duration::from_millis(50));
    listener.finish()?;

    // Read header results
    let mut per_thread_latencies_ns: Vec<f64> = Vec::new();
    let mut total_retries: u64 = 0;
    let mut total_completed: u64 = 0;

    for tid in 0..num_threads as usize {
        let elapsed_ns = unsafe { std::ptr::read_volatile(results_host_ptr.add(tid * 3)) };
        let retries = unsafe { std::ptr::read_volatile(results_host_ptr.add(tid * 3 + 1)) };
        let completed = unsafe { std::ptr::read_volatile(results_host_ptr.add(tid * 3 + 2)) };

        total_retries += retries;
        total_completed += completed;

        if completed > 0 {
            per_thread_latencies_ns.push(elapsed_ns as f64 / completed as f64);
        }
    }

    // Read per-iteration latencies
    let mut per_iter_latencies_ns: Vec<f64> = Vec::new();
    for tid in 0..num_threads as usize {
        let completed = unsafe { std::ptr::read_volatile(results_host_ptr.add(tid * 3 + 2)) };
        for iter in 0..completed as usize {
            let offset = header_count + tid * (num_iters as usize) + iter;
            let lat_ns = unsafe { std::ptr::read_volatile(results_host_ptr.add(offset)) };
            if lat_ns > 0 {
                per_iter_latencies_ns.push(lat_ns as f64);
            }
        }
    }

    unsafe { free_mapped_u64_array(results_host_ptr)? };

    Ok(BenchmarkResult {
        name: name.to_string(),
        num_threads,
        num_iters,
        num_packets,
        grid_dim,
        block_dim,
        wall_ms: wall_elapsed.as_secs_f64() * 1000.0,
        total_completed,
        total_retries,
        per_iter_latencies_ns,
        per_thread_latencies_ns,
    })
}

/// Sustained throughput benchmark (bench-suite.2).
///
/// Measures peak hostcalls/sec with many threads and many iterations,
/// ensuring the system reaches steady state.
pub(crate) fn run_throughput_benchmark(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Hostcall Throughput Benchmark (bench-suite.2) ---");
    println!("  Per-iteration timestamps via v3 kernel. Sustained load measurement.\n");

    // Warmup: 1 thread, 10 iters
    let _ = run_v3_bench(&dev, "warmup", 1, 1, 10, 4);

    let mut all_results: Vec<BenchmarkResult> = Vec::new();

    // Scenario 1: Single-thread micro-latency (true per-iteration distribution)
    let r = run_v3_bench(&dev, "nop_micro_1t", 1, 1, 100, 4)?;
    println!("  {}", r.summary_line());
    all_results.push(r);

    // Scenario 2: Single warp (32 threads) with enough packets
    let r = run_v3_bench(&dev, "nop_warp_32t", 1, 32, 50, 32)?;
    println!("  {}", r.summary_line());
    all_results.push(r);

    // Scenario 3: 4 blocks × 32 threads = 128 threads, sustained
    let r = run_v3_bench(&dev, "nop_sustained_128t", 4, 32, 50, 64)?;
    println!("  {}", r.summary_line());
    all_results.push(r);

    // Scenario 4: 16 blocks × 32 threads = 512 threads, stress test
    let r = run_v3_bench(&dev, "nop_stress_512t", 16, 32, 20, 64)?;
    println!("  {}", r.summary_line());
    all_results.push(r);

    // Print aggregate summary
    println!("\n  --- Throughput Summary ---");
    for r in &all_results {
        let stats = r.latency_stats();
        println!(
            "    {:<25} {:>8.0} calls/s  p50={:>8.0}ns  p99={:>8.0}ns  stddev={:>8.0}ns  CAS/call={:.2}",
            r.name,
            r.throughput(),
            stats.p50_ns,
            stats.p99_ns,
            stats.stddev_ns,
            r.cas_retry_rate(),
        );
    }

    // Write JSON results
    let json_path = std::path::Path::new("bench-results");
    if !json_path.exists() {
        let _ = std::fs::create_dir_all(json_path);
    }
    let _ = bench_harness::write_results_json(&json_path.join("throughput.json"), &all_results);

    println!("\n  Throughput benchmark complete.");
    Ok(())
}

/// Scalability curve benchmark (bench-suite.2).
///
/// Measures throughput and latency as a function of thread count,
/// sweeping from 1 to 1024 threads to find the saturation point.
pub(crate) fn run_scalability_benchmark(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Hostcall Scalability Benchmark (bench-suite.2) ---");
    println!("  Sweeping thread counts [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]");
    println!("  Fixed 20 iters/thread. Packet pool = min(2*threads, 64).\n");

    // Warmup
    let _ = run_v3_bench(&dev, "warmup", 1, 1, 5, 4);

    let thread_counts: &[u32] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
    let num_iters = 20;
    let mut all_results: Vec<BenchmarkResult> = Vec::new();

    println!(
        "  {:>6} {:>6} {:>6} {:>4} | {:>10} {:>10} {:>10} {:>10} | {:>8} {:>10}",
        "threads",
        "grid",
        "block",
        "pkts",
        "p50(ns)",
        "p95(ns)",
        "p99(ns)",
        "mean(ns)",
        "CAS/call",
        "throughput",
    );

    for &total_threads in thread_counts {
        let block_dim = total_threads.min(32); // keep warps compact
        let grid_dim = total_threads.div_ceil(block_dim);
        let num_packets = ((total_threads * 2) as u16).clamp(4, 64);

        let name = format!("scale_{total_threads}t");
        let r = run_v3_bench(&dev, &name, grid_dim, block_dim, num_iters, num_packets)?;

        let stats = r.latency_stats();
        println!(
            "  {:>6} {:>6} {:>6} {:>4} | {:>10.0} {:>10.0} {:>10.0} {:>10.0} | {:>8.2} {:>10.0}",
            total_threads,
            grid_dim,
            block_dim,
            num_packets,
            stats.p50_ns,
            stats.p95_ns,
            stats.p99_ns,
            stats.mean_ns,
            r.cas_retry_rate(),
            r.throughput(),
        );

        all_results.push(r);
    }

    // Identify saturation point
    if all_results.len() >= 2 {
        let mut peak_throughput = 0.0f64;
        let mut peak_threads = 0u32;
        for r in &all_results {
            if r.throughput() > peak_throughput {
                peak_throughput = r.throughput();
                peak_threads = r.num_threads;
            }
        }
        println!(
            "\n  Peak throughput: {:.0} calls/s at {} threads",
            peak_throughput, peak_threads,
        );

        // Check for saturation (throughput drops >10% from peak)
        let saturated = all_results
            .iter()
            .find(|r| r.num_threads > peak_threads && r.throughput() < peak_throughput * 0.9);
        if let Some(r) = saturated {
            println!(
                "  Saturation detected at {} threads ({:.0} calls/s, {:.0}% of peak)",
                r.num_threads,
                r.throughput(),
                r.throughput() / peak_throughput * 100.0,
            );
        }
    }

    // Write JSON
    let json_path = std::path::Path::new("bench-results");
    if !json_path.exists() {
        let _ = std::fs::create_dir_all(json_path);
    }
    let _ = bench_harness::write_results_json(&json_path.join("scalability.json"), &all_results);

    println!("\n  Scalability benchmark complete.");
    Ok(())
}

/// File I/O latency benchmark (bench-suite.3).
///
/// Measures per-phase latency for open/write/close/open/read/close cycles.
/// Single-thread only (file I/O is inherently serial).
pub(crate) fn run_file_io_benchmark(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- File I/O Latency Benchmark (bench-suite.3) ---");
    println!("  Thread 0 performs N rounds of open→write→close→open→read→close.");
    println!("  Per-phase timestamps via file_io_bench kernel.\n");

    crate::kernel_routes::load_kernel(
        &dev,
        crate::kernel_routes::KernelModule::Io,
        "kernel_file_bench",
        &["file_io_bench"],
    )?;
    let f = dev
        .get_func("kernel_file_bench", "file_io_bench")
        .ok_or(GpuHostError::KernelNotFound("file_io_bench"))?;

    let num_iters: u32 = 30;

    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();

    // Results: 2 header + 6 per iter
    let results_count = 2 + (num_iters as usize) * 6;
    let (results_host_ptr, results_dev_ptr) =
        unsafe { alloc_mapped_u64_array(&dev, results_count)? };

    // Zero out
    unsafe {
        std::ptr::write_bytes(results_host_ptr, 0, results_count);
    }

    let hc_buf_ref = Arc::new(hc_buf);
    let listener =
        crate::harness_support::HostcallListener::start(Arc::clone(&hc_buf_ref), |_msg| {});

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // Warmup run (3 iters)
    {
        let warmup_count = 2 + 3 * 6;
        let (warmup_host, warmup_dev) = unsafe { alloc_mapped_u64_array(&dev, warmup_count)? };
        unsafe {
            std::ptr::write_bytes(warmup_host, 0, warmup_count);
            f.clone().launch(cfg, (dev_ptr, warmup_dev, 3u32))?;
        }
        dev.synchronize()?;
        unsafe { free_mapped_u64_array(warmup_host)? };
    }

    // Main run
    let start = std::time::Instant::now();
    unsafe {
        f.launch(cfg, (dev_ptr, results_dev_ptr, num_iters))?;
    }
    dev.synchronize()?;
    let wall_elapsed = start.elapsed();

    std::thread::sleep(std::time::Duration::from_millis(50));
    listener.finish()?;

    // Read results
    let total_ns = unsafe { std::ptr::read_volatile(results_host_ptr.add(0)) };
    let completed = unsafe { std::ptr::read_volatile(results_host_ptr.add(1)) };

    let phase_names = [
        "open-write",
        "write",
        "close-write",
        "open-read",
        "read",
        "close-read",
    ];
    let mut phase_data: Vec<Vec<f64>> = vec![Vec::new(); 6];

    for iter in 0..completed as usize {
        let base = 2 + iter * 6;
        for phase in 0..6 {
            let ns = unsafe { std::ptr::read_volatile(results_host_ptr.add(base + phase)) };
            if ns > 0 {
                phase_data[phase].push(ns as f64);
            }
        }
    }

    unsafe { free_mapped_u64_array(results_host_ptr)? };

    // Print per-phase statistics
    println!(
        "  Completed {}/{} iterations, total={:.1}ms, wall={:.1}ms\n",
        completed,
        num_iters,
        total_ns as f64 / 1_000_000.0,
        wall_elapsed.as_secs_f64() * 1000.0,
    );

    println!(
        "  {:<12} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Phase", "p50(μs)", "p95(μs)", "p99(μs)", "mean(μs)", "stddev(μs)",
    );

    let mut total_mean = 0.0;
    for (i, name) in phase_names.iter().enumerate() {
        let stats = bench_harness::compute_stats(&phase_data[i]);
        println!(
            "  {:<12} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1}",
            name,
            stats.p50_ns / 1000.0,
            stats.p95_ns / 1000.0,
            stats.p99_ns / 1000.0,
            stats.mean_ns / 1000.0,
            stats.stddev_ns / 1000.0,
        );
        total_mean += stats.mean_ns;
    }

    println!(
        "\n  Total round-trip mean: {:.1}μs ({:.0}ns)",
        total_mean / 1000.0,
        total_mean,
    );
    println!(
        "  File I/O throughput: {:.0} round-trips/s",
        if total_mean > 0.0 {
            1_000_000_000.0 / total_mean
        } else {
            0.0
        },
    );

    // Clean up benchmark file
    let _ = std::fs::remove_file("gpu_bench_output.txt");

    println!("\n  File I/O benchmark complete.");
    Ok(())
}
