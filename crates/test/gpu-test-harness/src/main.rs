#![allow(clippy::needless_range_loop)]
mod bench_harness;
mod composed_oracle;
mod harness_support;
mod kernel_routes;
mod tests_basic;
mod tests_benchmark;
mod tests_cnn;
mod tests_gemm;
mod tests_hostcall;
mod tests_inference;
mod tests_obstacle;
mod tests_par_iter;
mod tests_pipeline;
mod tests_priority_safety;
mod tests_scaling;
mod tests_search;
mod tests_std;
mod tests_tokenizer;
mod tests_transformer;
mod tests_warp;

use cudarc::driver::{CudaDevice, DevicePtr};
use gpu_host::error::{GpuHostError, Result};
use std::sync::Arc;

// PTX constants re-exported from the library crate. Keep every canonical
// module explicit: a generic compute alias previously hid IO/test misroutes.
const KERNEL_CORE_PTX: &str = gpu_host::ptx::KERNEL_CORE;
const KERNEL_COMPUTE_PTX: &str = gpu_host::ptx::KERNEL_COMPUTE;
const EMBASSY_PTX: &str = gpu_host::ptx::EMBASSY_TEST;
const ASYNC_HOSTCALL_PTX: &str = gpu_host::ptx::ASYNC_HOSTCALL_TEST;
const STD_BUILD_TEST_PTX: &str = gpu_host::ptx::STD_BUILD_TEST;
const ASYNC_PIPELINE_PTX: &str = gpu_host::ptx::ASYNC_PIPELINE_TEST;
const MULTI_WARP_PTX: &str = gpu_host::ptx::MULTI_WARP_TEST;
const KERNEL_IO_PTX: &str = gpu_host::ptx::KERNEL_IO;
const KERNEL_TEST_PTX: &str = gpu_host::ptx::KERNEL_TEST;
const KERNEL_STD_PTX: &str = KERNEL_TEST_PTX;

fn main() -> Result<()> {
    println!("=== GPU Kernel Execution Test ===\n");

    let dev = CudaDevice::new(0).map_err(GpuHostError::CudaInit)?;
    println!("CUDA device initialized successfully");

    // Quick filter: ONLY_TEST=generation to skip to generation test
    if let Ok(only) = std::env::var("ONLY_TEST") {
        kernel_routes::validate_build_feature(&only)?;
        kernel_routes::validate_feature_gate(&only)?;
        if let Some(plan) = kernel_routes::selector_plan_for(&only) {
            println!("ONLY_TEST={only}: route plan={}", plan.name());
        }
        if let Some(route) = kernel_routes::route_for(&only) {
            println!(
                "ONLY_TEST={only}: explicit PTX route={} symbols={:?}",
                route.module.name(),
                route.symbols
            );
        }
        match only.as_str() {
            "generation" => {
                tests_inference::run_generation_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "forward" => {
                tests_inference::run_f32_forward_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "cpu_ref" => {
                tests_inference::run_cpu_f64_reference_test()?;
                return Ok(());
            }
            "mma_diag" => {
                tests_gemm::run_mma_diag(Arc::clone(&dev))?;
                return Ok(());
            }
            "splitk" => {
                tests_gemm::run_splitk_gemm_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "mma_fwd" => {
                tests_inference::run_mma_forward_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "kv_cache" => {
                tests_transformer::run_kv_cache_attention_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "kv_gen" => {
                tests_inference::run_kv_cached_generation_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "bf16" => {
                tests_gemm::run_bf16_gemm_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "bf16_fwd" => {
                tests_inference::run_bf16_forward_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "tf32" => {
                tests_gemm::run_tf32_gemm_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "std_fs" => {
                tests_std::run_std_file_io_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "std_sysroot_file" => {
                tests_std::run_std_sysroot_file_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "std_pipeline" => {
                tests_std::run_std_pipeline_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "std_stdin" => {
                tests_std::run_std_stdin_readline_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "mt_malloc" => {
                tests_std::run_multithread_malloc_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "mt_std" => {
                tests_std::run_multithread_println_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "trace" => {
                tests_hostcall::run_trace_multithread_test(Arc::clone(&dev))?;
                tests_hostcall::run_trace_assert_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "session" => {
                tests_hostcall::run_session_multi_launch_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "cmd" => {
                tests_hostcall::run_multi_cmd_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "pipeline" => {
                tests_hostcall::run_cross_pipeline_test(Arc::clone(&dev))?;
                tests_hostcall::run_pipeline_api_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "converge" => {
                tests_hostcall::run_convergence_test(Arc::clone(&dev))?;
                tests_hostcall::run_autonomous_pipeline_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "flight" => {
                tests_hostcall::run_flight_recorder_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "warp_try" => {
                tests_warp::run_warp_try_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "warp_await" => {
                tests_warp::run_warp_await_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "warp_e2e" => {
                tests_warp::run_warp_e2e_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "rustc_async" => {
                tests_warp::run_rustc_async_baseline_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "std_future" => {
                tests_hostcall::run_std_future_print_test(Arc::clone(&dev))?;
                tests_hostcall::run_std_future_two_prints_test(Arc::clone(&dev))?;
                tests_hostcall::run_warp_cooperative_future_test(Arc::clone(&dev))?;
                tests_hostcall::run_warp_cooperative_two_futures_test(Arc::clone(&dev))?;
                tests_hostcall::run_warp_result_future_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "throughput" => {
                tests_benchmark::run_throughput_benchmark(Arc::clone(&dev))?;
                return Ok(());
            }
            "scalability" => {
                tests_benchmark::run_scalability_benchmark(Arc::clone(&dev))?;
                return Ok(());
            }
            "file_io_bench" => {
                tests_benchmark::run_file_io_benchmark(Arc::clone(&dev))?;
                return Ok(());
            }
            "executor" => {
                tests_scaling::run_executor_demo_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "obstacle" | "obstacle_stress" => {
                tests_obstacle::run_obstacle_event_stress(Arc::clone(&dev))?;
                return Ok(());
            }
            "composed_priority" | "priority_e2e" => {
                tests_obstacle::run_composed_priority_e2e(Arc::clone(&dev))?;
                return Ok(());
            }
            "priority_safety" => {
                tests_priority_safety::run_priority_handle_safety_stress(Arc::clone(&dev))?;
                return Ok(());
            }
            "channel" | "channel_oneshot" => {
                tests_scaling::run_channel_oneshot_demo_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "channel_mpsc" | "mpsc" => {
                tests_scaling::run_channel_mpsc_demo_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "compute" | "compute_pipeline" => {
                tests_scaling::run_compute_pipeline_demo_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "compute_bench" => {
                tests_scaling::run_compute_benchmark_test(Arc::clone(&dev))?;
                return Ok(());
            }
            #[cfg(feature = "async")]
            "tokio_bridge" | "tokio" => {
                let tokio_rt = tokio::runtime::Runtime::new().expect("tokio runtime init");
                tokio_rt
                    .block_on(tests_scaling::run_tokio_bridge_demo_test())
                    .map_err(|e| GpuHostError::Verification {
                        test: "tokio_bridge",
                        detail: format!("{e}"),
                    })?;
                return Ok(());
            }
            "gemm_bench" => {
                tests_gemm::run_gemm_benchmark(Arc::clone(&dev))?;
                return Ok(());
            }
            #[cfg(feature = "cublas")]
            "attn_bench" => {
                tests_transformer::run_flash_attention_v3_bench(Arc::clone(&dev))?;
                return Ok(());
            }
            #[cfg(feature = "nn")]
            "elem_bench" => {
                tests_transformer::run_elementwise_benchmark(Arc::clone(&dev))?;
                return Ok(());
            }
            "thread_spawn" => {
                run_thread_spawn_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "gpu_run" => {
                run_gpu_api_test()?;
                return Ok(());
            }
            "cooperative" => {
                // Debug: check which warps participate
                println!("\n--- Cooperative Debug ---");
                let dbg: Vec<u32> =
                    gpu_host::gpu::launch("cooperative_debug", 4, 128).map_err(|e| {
                        GpuHostError::Verification {
                            test: "coop",
                            detail: format!("{e}"),
                        }
                    })?;
                println!("  Warp writes: {:?} (expect [100, 101, 102, 103])", dbg);

                println!("\n--- Cooperative Compute Test ---");
                let result: Vec<u32> = gpu_host::gpu::launch("cooperative_compute_test", 256, 128)
                    .map_err(|e| GpuHostError::Verification {
                        test: "coop",
                        detail: format!("{e}"),
                    })?;
                let mut ok = true;
                for i in 0..256usize {
                    let expected = (i * 2 + 1) as u32;
                    if result[i] != expected {
                        println!("  MISMATCH at {i}: got {}, expected {expected}", result[i]);
                        ok = false;
                    }
                }
                println!(
                    "  Cooperative compute: {}",
                    if ok { "PASSED" } else { "FAILED" }
                );
                assert!(ok);

                println!("\n--- Cooperative Map Test (no global atomics) ---");
                let map_result: Vec<u32> = gpu_host::gpu::launch("cooperative_map_test", 256, 128)
                    .map_err(|e| GpuHostError::Verification {
                        test: "coop_map",
                        detail: format!("{e}"),
                    })?;
                let mut map_ok = true;
                for i in 0..256usize {
                    let expected = (i * 2) as u32;
                    if map_result[i] != expected {
                        println!(
                            "  MISMATCH at {i}: got {}, expected {expected}",
                            map_result[i]
                        );
                        map_ok = false;
                    }
                }
                println!(
                    "  Cooperative map: {}",
                    if map_ok { "PASSED" } else { "FAILED" }
                );
                assert!(map_ok);

                println!("\n--- Cooperative Reduce Test (multi-warp sum) ---");
                let reduce_result: Vec<u64> =
                    gpu_host::gpu::launch("cooperative_reduce_test", 1, 128).map_err(|e| {
                        GpuHostError::Verification {
                            test: "coop_reduce",
                            detail: format!("{e}"),
                        }
                    })?;
                let total = reduce_result[0];
                let expected_sum: u64 = (0..256u64).sum(); // 32640
                println!("  Result: {total}, expected: {expected_sum}");
                println!(
                    "  Cooperative reduce: {}",
                    if total == expected_sum {
                        "PASSED"
                    } else {
                        "FAILED"
                    }
                );
                assert_eq!(total, expected_sum);

                println!("\n--- Cooperative Map with Params Test (scaled multiply) ---");
                let ext_result: Vec<u32> =
                    gpu_host::gpu::launch("cooperative_map_ext_test", 256, 128).map_err(|e| {
                        GpuHostError::Verification {
                            test: "coop_map_ext",
                            detail: format!("{e}"),
                        }
                    })?;
                let mut ext_ok = true;
                for i in 0..256usize {
                    let expected = (i * 7) as u32;
                    if ext_result[i] != expected {
                        println!(
                            "  MISMATCH at {i}: got {}, expected {expected}",
                            ext_result[i]
                        );
                        ext_ok = false;
                    }
                }
                println!(
                    "  Cooperative map_with_params: {}",
                    if ext_ok { "PASSED" } else { "FAILED" }
                );
                assert!(ext_ok);

                return Ok(());
            }
            "matmul" => {
                println!("\n--- Cooperative Matmul Test (coop-compute.2) ---");
                println!(
                    "  C[8×6] = A[8×4] × B[4×6], naive triple-loop via cooperative_map_with_params"
                );

                const M: usize = 8;
                const K: usize = 4;
                const N: usize = 6;

                // Launch kernel: 48 f32 output elements, 128 threads (4 warps)
                let result: Vec<f32> = gpu_host::gpu::launch("cooperative_matmul_test", M * N, 128)
                    .map_err(|e| GpuHostError::Verification {
                        test: "matmul",
                        detail: format!("{e}"),
                    })?;

                // CPU reference matmul
                // A[i][j] = (i * K + j + 1) as f32
                // B[i][j] = ((i * N + j + 1) * 2) as f32
                let mut a = [0.0f32; M * K];
                let mut b = [0.0f32; K * N];
                for i in 0..M {
                    for j in 0..K {
                        a[i * K + j] = (i * K + j + 1) as f32;
                    }
                }
                for i in 0..K {
                    for j in 0..N {
                        b[i * N + j] = ((i * N + j + 1) * 2) as f32;
                    }
                }

                // C = A × B (CPU reference)
                let mut expected = [0.0f32; M * N];
                for i in 0..M {
                    for j in 0..N {
                        let mut sum = 0.0f32;
                        for p in 0..K {
                            sum += a[i * K + p] * b[p * N + j];
                        }
                        expected[i * N + j] = sum;
                    }
                }

                // Verify GPU result against CPU reference
                let mut matmul_ok = true;
                let mut mismatch_count = 0usize;
                for i in 0..M {
                    for j in 0..N {
                        let idx = i * N + j;
                        let gpu_val = result[idx];
                        let cpu_val = expected[idx];
                        if (gpu_val - cpu_val).abs() > 1e-3 {
                            if mismatch_count < 5 {
                                println!("  MISMATCH at C[{i}][{j}]: GPU={gpu_val}, CPU={cpu_val}");
                            }
                            matmul_ok = false;
                            mismatch_count += 1;
                        }
                    }
                }

                if matmul_ok {
                    println!("  All 48 elements match CPU reference (tolerance 1e-3)");
                    println!(
                        "  Sample: C[0][0]={:.1}, C[7][5]={:.1}",
                        result[0],
                        result[M * N - 1]
                    );
                } else {
                    println!("  FAILED: {mismatch_count} mismatches out of {}", M * N);
                }

                println!(
                    "  Cooperative matmul: {}",
                    if matmul_ok { "PASSED" } else { "FAILED" }
                );
                assert!(matmul_ok);

                return Ok(());
            }
            "std_thread_demo" => {
                run_std_thread_spawn_demo(Arc::clone(&dev))?;
                return Ok(());
            }
            "real_std_thread" => {
                run_real_std_thread_spawn(Arc::clone(&dev))?;
                return Ok(());
            }
            "std_thread_minimal" => {
                run_std_thread_spawn_minimal(Arc::clone(&dev))?;
                return Ok(());
            }
            "par_iter" | "par_iter_fusion" => {
                tests_par_iter::run_all_par_iter_tests(Arc::clone(&dev))?;
                return Ok(());
            }
            "par_iter_1m" => {
                tests_par_iter::run_par_iter_1m_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "par_iter_bench" | "par_iter_rayon" => {
                tests_par_iter::run_par_iter_rayon_benchmark(Arc::clone(&dev))?;
                return Ok(());
            }
            "par_iter_multiblock" | "par_iter_mb" => {
                tests_par_iter::run_multiblock_benchmark(Arc::clone(&dev))?;
                return Ok(());
            }
            "kernel_std_smoke" => {
                run_kernel_std_smoke(Arc::clone(&dev))?;
                return Ok(());
            }
            "matmul_io" => {
                run_matmul_io_compute(Arc::clone(&dev))?;
                return Ok(());
            }
            #[cfg(feature = "cublas")]
            "fusion_bench" => {
                run_fusion_benchmark(Arc::clone(&dev))?;
                return Ok(());
            }
            #[cfg(feature = "cublas")]
            "sgemm_v4_bench" => {
                run_sgemm_v4_benchmark(Arc::clone(&dev))?;
                return Ok(());
            }
            "cnn" => {
                tests_cnn::run_batchnorm_silu_test(Arc::clone(&dev))?;
                tests_cnn::run_cnn_ops_test(Arc::clone(&dev))?;
                tests_cnn::run_conv2d_test(Arc::clone(&dev))?;
                tests_cnn::run_yolo_io_test()?;
                tests_cnn::run_yolo_backbone_test(Arc::clone(&dev))?;
                tests_cnn::run_detect_head_test(Arc::clone(&dev))?;
                tests_cnn::run_yolo_end_to_end_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "yolo" => {
                tests_cnn::run_yolo_end_to_end_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "bench" => {
                tests_benchmark::run_throughput_benchmark(Arc::clone(&dev))?;
                tests_benchmark::run_scalability_benchmark(Arc::clone(&dev))?;
                tests_benchmark::run_file_io_benchmark(Arc::clone(&dev))?;
                return Ok(());
            }
            "zero_param" => {
                run_zero_param_test()?;
                return Ok(());
            }
            "generator" | "coroutine" => {
                run_generator_tests()?;
                return Ok(());
            }
            _ => {
                return Err(GpuHostError::Verification {
                    test: "only_test",
                    detail: format!("unknown or unavailable ONLY_TEST selector `{only}`"),
                });
            }
        }
    }

    tests_basic::run_write_thread_idx(Arc::clone(&dev))?;
    tests_basic::run_vector_add(Arc::clone(&dev))?;
    tests_basic::run_asm_smoke_tests(Arc::clone(&dev))?;
    tests_basic::run_integration_sys_store(Arc::clone(&dev))?;
    tests_basic::run_u64_atomics_tests(Arc::clone(&dev))?;
    tests_basic::run_warp_intrinsics_tests(Arc::clone(&dev))?;

    // Hostcall tests (hostcall.4)
    tests_hostcall::run_hostcall_print_hello(Arc::clone(&dev))?;
    tests_hostcall::run_hostcall_print_multi(Arc::clone(&dev), 4)?;

    // Embassy async/await tests (async-runtime.3)
    tests_hostcall::run_embassy_immediate(Arc::clone(&dev))?;
    tests_hostcall::run_embassy_countdown(Arc::clone(&dev))?;
    tests_hostcall::run_embassy_two_task(Arc::clone(&dev))?;
    tests_hostcall::run_sync_countdown(Arc::clone(&dev))?;

    // File I/O test (gpu-std.3)
    tests_hostcall::run_hostcall_file_test(Arc::clone(&dev))?;

    // Async hostcall tests (integration.1)
    tests_hostcall::run_async_hostcall_single(Arc::clone(&dev))?;
    tests_hostcall::run_async_hostcall_two(Arc::clone(&dev))?;

    // futures_util::future::join on GPU (integration.2)
    tests_hostcall::run_futures_join(Arc::clone(&dev))?;

    // GPU Instant + Time test (gpu-std.4)
    tests_hostcall::run_hostcall_time_test(Arc::clone(&dev))?;

    // Std-build-test (integration.3): -Zbuild-std=std on GPU
    tests_std::run_std_build_test(Arc::clone(&dev))?;

    // Dynamic allocation stress test (product.1)
    tests_std::run_dynamic_alloc_test(Arc::clone(&dev))?;

    // Hostcall latency benchmark (benchmark.2) — run before PAL tests
    tests_benchmark::run_hostcall_latency_benchmark(Arc::clone(&dev))?;

    // Warp divergence measurement (warp-future.2)
    tests_benchmark::run_warp_divergence_measurement(Arc::clone(&dev))?;

    // Bulk data transfer test (large-payload.3)
    tests_pipeline::run_bulk_io_test(Arc::clone(&dev))?;

    // Per-block sharding test (per-block-sharding.2)
    tests_pipeline::run_sharded_hostcall_test(Arc::clone(&dev))?;

    // Sharding benchmark (per-block-sharding.3)
    tests_benchmark::run_sharding_benchmark(Arc::clone(&dev))?;

    // Throughput + scalability benchmarks (bench-suite.2)
    tests_benchmark::run_throughput_benchmark(Arc::clone(&dev))?;
    tests_benchmark::run_scalability_benchmark(Arc::clone(&dev))?;

    // File I/O latency benchmark (bench-suite.3)
    tests_benchmark::run_file_io_benchmark(Arc::clone(&dev))?;

    // Executor demo (executor-impl.4)
    tests_scaling::run_executor_demo_test(Arc::clone(&dev))?;

    // Channel oneshot demo (channel-oneshot.3)
    tests_scaling::run_channel_oneshot_demo_test(Arc::clone(&dev))?;

    // Compute pipeline demo (demo-pipeline.2)
    tests_scaling::run_compute_pipeline_demo_test(Arc::clone(&dev))?;

    // Parallel file grep demo (product.8)
    tests_pipeline::run_parallel_grep_test(Arc::clone(&dev))?;

    // Warp intrinsics test (warp-future.3): syncwarp + shfl.sync.idx
    tests_warp::run_warp_intrinsics_test(Arc::clone(&dev))?;

    // WarpFuture PoC test (warp-future.4)
    tests_warp::run_warp_future_print_test(Arc::clone(&dev))?;

    // WarpFuture multi-hostcall test (warp-future.6)
    tests_warp::run_warp_future_multi_print_test(Arc::clone(&dev))?;

    // WarpFuture proc macro test (warp-future.5)
    tests_warp::run_warp_print_test(Arc::clone(&dev))?;

    // WarpFuture if/else test (warp-cfg.2)
    tests_warp::run_warp_cfg_if_else_test(Arc::clone(&dev))?;

    // WarpFuture loop/break test (warp-cfg.3)
    tests_warp::run_warp_cfg_loop_test(Arc::clone(&dev))?;

    // WarpFuture match test (warp-cfg.4)
    tests_warp::run_warp_cfg_match_test(Arc::clone(&dev))?;

    // WarpFuture nested control flow test (warp-cfg.5)
    tests_warp::run_warp_cfg_nested_test(Arc::clone(&dev))?;

    // Hybrid executor test (hybrid-executor.1)
    tests_warp::run_hybrid_executor_test(Arc::clone(&dev))?;

    // Hybrid stress test (hybrid-executor.2)
    tests_warp::run_hybrid_stress_test(Arc::clone(&dev))?;

    // f32 math validation (ml-workload.1)
    tests_search::run_f32_math_test(Arc::clone(&dev))?;

    // Vector similarity search (ml-workload.2)
    tests_search::run_vector_search_test(Arc::clone(&dev))?;

    // Batch vector search (ml-workload.3)
    tests_search::run_batch_search_test(Arc::clone(&dev))?;

    // File transform pipeline (async-pipeline.1+2)
    tests_pipeline::run_file_transform_test(Arc::clone(&dev))?;

    // Branching pipeline (async-pipeline.3)
    tests_pipeline::run_branching_pipeline_test(Arc::clone(&dev))?;

    // Pipelined I/O + compute (async-pipeline.4)
    tests_pipeline::run_pipelined_compute_test(Arc::clone(&dev))?;

    // Warp-scale Embassy test (async-pipeline.5)
    tests_pipeline::run_warp_scale_async_test(Arc::clone(&dev))?;

    // Autonomous pipeline test (gpu-compute.2)
    tests_pipeline::run_autonomous_pipeline_test(Arc::clone(&dev))?;

    // Tensor Core MMA test (gpu-compute.3)
    tests_search::run_mma_test(Arc::clone(&dev))?;

    // Shared memory + bar.sync test (gpu-compute.4)
    tests_search::run_shared_memory_test(Arc::clone(&dev))?;

    // Tiled GEMM test (gpu-compute.5)
    tests_gemm::run_tiled_gemm_test(Arc::clone(&dev))?;

    // MMA fragment mapping test (gpu-pipeline.1)
    tests_search::run_mma_mapped_test(Arc::clone(&dev))?;

    // Softmax test (gpu-compute.6)
    tests_gemm::run_softmax_test(Arc::clone(&dev))?;

    // Multi-tile GEMM test (gpu-pipeline.2)
    tests_gemm::run_multi_tile_gemm_test(Arc::clone(&dev))?;

    // GEMM + softmax pipeline test (gpu-pipeline.3)
    tests_gemm::run_gemm_softmax_pipeline_test(Arc::clone(&dev))?;

    // Multi-warp GEMM test (gemm-scale.1)
    tests_gemm::run_multi_warp_gemm_test(Arc::clone(&dev))?;

    // Multi-block GEMM test (gemm-scale.2)
    tests_gemm::run_multi_block_gemm_test(Arc::clone(&dev))?;

    // Full GEMM 768x768 test (gemm-scale.3)
    tests_gemm::run_full_gemm_test(Arc::clone(&dev))?;

    // Full GEMM f32-input test (precision-fix.2)
    tests_gemm::run_full_gemm_f32in_test(Arc::clone(&dev))?;

    // LayerNorm test (transformer-layer.1)
    tests_transformer::run_layer_norm_test(Arc::clone(&dev))?;

    // GELU test (transformer-layer.2)
    tests_transformer::run_gelu_test(Arc::clone(&dev))?;

    // Attention test (transformer-layer.3)
    tests_transformer::run_attention_test(Arc::clone(&dev))?;

    // FlashAttention test (attention-scale.3)
    tests_transformer::run_flash_attention_test(Arc::clone(&dev))?;

    // FlashAttention scaling test (attention-scale.4)
    tests_transformer::run_flash_attention_scale_test(Arc::clone(&dev))?;

    // Embedding lookup test (full-inference.1)
    tests_transformer::run_embedding_test(Arc::clone(&dev))?;

    // FFN block test (transformer-layer.4)
    tests_transformer::run_ffn_test(Arc::clone(&dev))?;

    // Transformer layer test (transformer-layer.6)
    tests_transformer::run_transformer_layer_test(Arc::clone(&dev))?;

    // 12-layer GPT-2 forward pass (full-inference.2) + LM head (full-inference.3)
    tests_inference::run_full_forward_test(Arc::clone(&dev))?;

    // Greedy autoregressive generation (full-inference.4)
    tests_inference::run_generation_test(Arc::clone(&dev))?;

    // f32 GEMM forward pass (full-inference.6)
    tests_inference::run_f32_forward_test(Arc::clone(&dev))?;

    // CPU f64 reference forward pass (full-inference.8+9+10)
    tests_inference::run_cpu_f64_reference_test()?;

    // GPT-2 model weight loading test (model-loading.3)
    {
        let model_path =
            gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("model.safetensors");
        if model_path.exists() {
            println!("\n--- GPT-2 weight loading test (model-loading.3) ---");
            let weights = gpu_host::model::load_gpt2_weights(&model_path).map_err(|e| {
                GpuHostError::Verification {
                    test: "model_loading",
                    detail: format!("{e}"),
                }
            })?;
            println!("  Total params: {}", weights.total_params());
            println!("  Memory: {:.1} MB", weights.memory_bytes() as f64 / 1e6);
            println!("  Layers: {}", weights.layers.len());
            println!("  wte shape: [50257, 768] = {} elements", weights.wte.len());
            println!("  wpe shape: [1024, 768] = {} elements", weights.wpe.len());
            assert_eq!(weights.layers.len(), 12);
            assert_eq!(weights.wte.len(), 50257 * 768);
            assert_eq!(weights.wpe.len(), 1024 * 768);
            assert_eq!(weights.layers[0].c_attn_weight.len(), 768 * 2304);
            assert_eq!(weights.layers[0].mlp_fc_weight.len(), 768 * 3072);

            // Upload embedding table to GPU to verify transfer
            let wte_dev = dev.htod_sync_copy(&weights.wte)?;
            let wte_back: Vec<f32> = dev.dtoh_sync_copy(&wte_dev)?;
            assert_eq!(wte_back.len(), weights.wte.len());
            assert_eq!(wte_back[0], weights.wte[0]);
            assert_eq!(wte_back[1000], weights.wte[1000]);
            println!("  GPU round-trip verified for wte");
            println!("  GPT-2 weight loading — PASSED");
        } else {
            println!(
                "\n--- Skipping GPT-2 weight loading (models/model.safetensors not found) ---"
            );
        }
    }

    // GPT-2 tokenizer test (tokenizer.2)
    tests_tokenizer::run_tokenizer_test()?;

    // GPT-2 tokenizer validation (tokenizer.3)
    tests_tokenizer::run_tokenizer_validation()?;

    // GPU panic handler test (gpu-panic.2) — MUST BE LAST
    // since trap instruction calls process::exit(0)
    tests_pipeline::run_panic_test(Arc::clone(&dev))?;

    // PAL stdout routing test (std-pal.1)
    tests_std::run_std_println_test(Arc::clone(&dev))?;

    // PAL stdin routing test (std-pal.2)
    tests_std::run_std_stdin_test(Arc::clone(&dev))?;

    // std::fs File I/O test (std-fs.4)
    tests_std::run_std_file_io_test(Arc::clone(&dev))?;

    // std::fs File I/O via build-std (std-sysroot-build.4)
    tests_std::run_std_sysroot_file_test(Arc::clone(&dev))?;

    // std pipeline test (std-migration.3)
    tests_std::run_std_pipeline_test(Arc::clone(&dev))?;

    // stdin().read_line() e2e test (std-migration.4)
    tests_std::run_std_stdin_readline_test(Arc::clone(&dev))?;

    // Multi-thread malloc test (std-hardening.3)
    tests_std::run_multithread_malloc_test(Arc::clone(&dev))?;

    // Multi-warp scaling test (product.3)
    tests_scaling::run_multi_warp_test(Arc::clone(&dev))?;

    // 4-step async pipeline test (product.2)
    tests_scaling::run_pipeline_test(Arc::clone(&dev))?;

    // Showcase demo kernel (product.4)
    tests_scaling::run_showcase_test(Arc::clone(&dev))?;

    // Multi-block scaling test (multiblock.1)
    tests_scaling::run_multi_block_test(Arc::clone(&dev))?;

    // Multi-block 512-thread scaling test (multiblock.2)
    tests_scaling::run_multi_block_512_test(Arc::clone(&dev))?;

    // Slab allocator deallocation test (allocator.2)
    tests_scaling::run_slab_dealloc_test(Arc::clone(&dev))?;

    // Concurrent slab allocator test (allocator.3)
    tests_scaling::run_slab_concurrent_test(Arc::clone(&dev))?;

    tests_scaling::run_println_direct_test(Arc::clone(&dev))?;

    tests_scaling::run_error_propagation_test(Arc::clone(&dev))?;

    tests_scaling::run_multi_block_async_test(Arc::clone(&dev))?;

    // Standard impl Future on GPU (warp-future-bridge.1)
    tests_hostcall::run_std_future_print_test(Arc::clone(&dev))?;
    tests_hostcall::run_std_future_two_prints_test(Arc::clone(&dev))?;

    // Warp-cooperative Future polling (warp-future-bridge.2)
    tests_hostcall::run_warp_cooperative_future_test(Arc::clone(&dev))?;

    // Warp-cooperative two sequential futures (warp-future-bridge.3)
    tests_hostcall::run_warp_cooperative_two_futures_test(Arc::clone(&dev))?;

    // Warp-cooperative Result<T,E> broadcasting (warp-future-bridge.4)
    tests_hostcall::run_warp_result_future_test(Arc::clone(&dev))?;

    // Buffered print test (printf-batch.3)
    tests_pipeline::run_buffered_print_test(Arc::clone(&dev))?;

    // Std buffered println! test (println-buffer.2)
    tests_std::run_std_buffered_println_test(Arc::clone(&dev))?;

    // Data-dependent iteration test (data-iter.1)
    tests_pipeline::run_newton_sqrt_test(Arc::clone(&dev))?;

    println!("\nAll tests PASSED.");
    Ok(())
}

/// Benchmark fused LayerNorm + residual vs separate ops.
#[cfg(feature = "cublas")]
fn run_fusion_benchmark(dev: Arc<CudaDevice>) -> Result<()> {
    use gpu_host::nn::ops::norm::{layer_norm, layer_norm_residual};
    use gpu_host::nn::registry::KernelRegistry;
    use gpu_host::nn::tensor::GpuTensor;

    println!("\n--- Fused LayerNorm+Residual Benchmark (perf-fusion) ---");

    let registry = std::sync::Arc::new(
        KernelRegistry::new(dev.clone(), crate::KERNEL_COMPUTE_PTX).map_err(|e| {
            GpuHostError::Verification {
                test: "fusion",
                detail: format!("{e}"),
            }
        })?,
    );

    const ROWS: usize = 128;
    const D: usize = 768;
    let input_data: Vec<f32> = (0..ROWS * D)
        .map(|i| (i % 17) as f32 * 0.01 - 0.08)
        .collect();
    let residual_data: Vec<f32> = (0..ROWS * D)
        .map(|i| (i % 13) as f32 * 0.01 - 0.06)
        .collect();
    let gamma: Vec<f32> = vec![1.0; D];
    let beta: Vec<f32> = vec![0.0; D];

    let input = GpuTensor::from_host(&input_data, &[ROWS, D], &dev).map_err(|e| {
        GpuHostError::Verification {
            test: "fusion",
            detail: format!("{e}"),
        }
    })?;
    let residual = GpuTensor::from_host(&residual_data, &[ROWS, D], &dev).map_err(|e| {
        GpuHostError::Verification {
            test: "fusion",
            detail: format!("{e}"),
        }
    })?;
    let g = GpuTensor::from_host(&gamma, &[D], &dev).map_err(|e| GpuHostError::Verification {
        test: "fusion",
        detail: format!("{e}"),
    })?;
    let b = GpuTensor::from_host(&beta, &[D], &dev).map_err(|e| GpuHostError::Verification {
        test: "fusion",
        detail: format!("{e}"),
    })?;

    let iters = 100;

    // Benchmark separate LayerNorm
    for _ in 0..5 {
        let _ = layer_norm(&input, &g, &b, 1e-5, &registry);
    }
    dev.synchronize().map_err(|e| GpuHostError::Verification {
        test: "fusion",
        detail: format!("{e}"),
    })?;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = layer_norm(&input, &g, &b, 1e-5, &registry);
    }
    dev.synchronize().map_err(|e| GpuHostError::Verification {
        test: "fusion",
        detail: format!("{e}"),
    })?;
    let ln_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Benchmark fused LN+residual
    for _ in 0..5 {
        let _ = layer_norm_residual(&input, &residual, &g, &b, 1e-5, &registry);
    }
    dev.synchronize().map_err(|e| GpuHostError::Verification {
        test: "fusion",
        detail: format!("{e}"),
    })?;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = layer_norm_residual(&input, &residual, &g, &b, 1e-5, &registry);
    }
    dev.synchronize().map_err(|e| GpuHostError::Verification {
        test: "fusion",
        detail: format!("{e}"),
    })?;
    let fused_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let bytes = (ROWS * D * 4) as f64 * 3.0; // input + residual + output
    let ln_gbps = bytes / (ln_ms * 1e6);
    let fused_gbps = bytes / (fused_ms * 1e6);

    println!("  Shape: [{ROWS}, {D}] (GPT-2 typical)");
    println!("  Separate LN:  {ln_ms:.4} ms ({ln_gbps:.0} GB/s)");
    println!("  Fused LN+res: {fused_ms:.4} ms ({fused_gbps:.0} GB/s)");
    println!("  Speedup:      {:.2}x", ln_ms / fused_ms);
    // --- elementwise_add: in-place vs out-of-place ---
    use gpu_host::nn::ops::reshape::elementwise_add_out;

    const ELEM_N: usize = 786_432; // 128 * 768 * 8 (typical GPT-2 FFN)
    let a_data: Vec<f32> = (0..ELEM_N).map(|i| (i % 11) as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..ELEM_N).map(|i| (i % 7) as f32 * 0.1).collect();

    let a_elem =
        GpuTensor::from_host(&a_data, &[ELEM_N], &dev).map_err(|e| GpuHostError::Verification {
            test: "elem",
            detail: format!("{e}"),
        })?;
    let b_elem =
        GpuTensor::from_host(&b_data, &[ELEM_N], &dev).map_err(|e| GpuHostError::Verification {
            test: "elem",
            detail: format!("{e}"),
        })?;

    // Out-of-place benchmark
    for _ in 0..5 {
        let _ = elementwise_add_out(&a_elem, &b_elem, &registry);
    }
    dev.synchronize().map_err(|e| GpuHostError::Verification {
        test: "elem",
        detail: format!("{e}"),
    })?;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = elementwise_add_out(&a_elem, &b_elem, &registry);
    }
    dev.synchronize().map_err(|e| GpuHostError::Verification {
        test: "elem",
        detail: format!("{e}"),
    })?;
    let oop_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // In-place benchmark
    let mut a_inplace =
        GpuTensor::from_host(&a_data, &[ELEM_N], &dev).map_err(|e| GpuHostError::Verification {
            test: "elem",
            detail: format!("{e}"),
        })?;
    for _ in 0..5 {
        let _ = gpu_host::nn::ops::elementwise_add(&mut a_inplace, &b_elem, &registry);
    }
    dev.synchronize().map_err(|e| GpuHostError::Verification {
        test: "elem",
        detail: format!("{e}"),
    })?;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = gpu_host::nn::ops::elementwise_add(&mut a_inplace, &b_elem, &registry);
    }
    dev.synchronize().map_err(|e| GpuHostError::Verification {
        test: "elem",
        detail: format!("{e}"),
    })?;
    let ip_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let elem_bytes = (ELEM_N * 4) as f64;
    let oop_gbps = elem_bytes * 3.0 / (oop_ms * 1e6); // read a + read b + write c
    let ip_gbps = elem_bytes * 3.0 / (ip_ms * 1e6); // read a + read b + write a (RW conflict)

    println!("\n  elementwise_add ({ELEM_N} elements):");
    println!("  In-place (a+=b): {ip_ms:.4} ms ({ip_gbps:.0} GB/s)");
    println!("  Out-of-place:    {oop_ms:.4} ms ({oop_gbps:.0} GB/s)");
    println!("  Speedup:         {:.2}x", ip_ms / oop_ms);

    println!("\n  Fused LN+Residual + elementwise Benchmark — DONE");
    Ok(())
}

/// SGEMM V4 benchmark: compare V4 (BK=8) vs V4.1 (BK=16) vs cuBLAS at 4096^3.
#[cfg(feature = "cublas")]
fn run_sgemm_v4_benchmark(dev: Arc<CudaDevice>) -> Result<()> {
    use gpu_host::nn::ops::gemm;
    use gpu_host::nn::tensor::GpuTensor;

    println!("\n--- SGEMM V4 Double-Buffer Benchmark ---");

    let shapes: &[(usize, usize, usize, &str)] = &[
        (512, 512, 512, "512^3"),
        (1024, 1024, 1024, "1024^3"),
        (2048, 2048, 2048, "2048^3"),
        (4096, 4096, 4096, "4096^3"),
        (128, 768, 768, "GPT-2 128x768x768"),
    ];

    let warmup = 5;
    let iters = 20;

    println!(
        "  {:>22}  {:>10} {:>10} {:>10} {:>8} {:>8}",
        "Shape", "V4 (ms)", "V4.1 (ms)", "cuBLAS", "V4 GF", "V4.1 GF"
    );
    println!(
        "  {:-<22}  {:-<10} {:-<10} {:-<10} {:-<8} {:-<8}",
        "", "", "", "", "", ""
    );

    for &(m, k, n, label) in shapes {
        // Generate deterministic test data
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 + 3) % 17) as f32 * 0.1 - 0.8)
            .collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| ((i * 11 + 7) % 19) as f32 * 0.1 - 0.9)
            .collect();

        let a = TensorForBench::new(&a_data, &[m, k], &dev)?;
        let b = TensorForBench::new(&b_data, &[k, n], &dev)?;

        // --- V4 (original BK=8) ---
        for _ in 0..warmup {
            let _ = gemm::matmul_v4(&a.tensor, &b.tensor, m, k, n, &dev);
        }
        dev.synchronize().map_err(|e| GpuHostError::Verification {
            test: "sgemm",
            detail: format!("{e}"),
        })?;

        let start = std::time::Instant::now();
        for _ in 0..iters {
            let _ = gemm::matmul_v4(&a.tensor, &b.tensor, m, k, n, &dev);
        }
        dev.synchronize().map_err(|e| GpuHostError::Verification {
            test: "sgemm",
            detail: format!("{e}"),
        })?;
        let v4_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // --- V4.1 (BK=16, float4 A loads) ---
        for _ in 0..warmup {
            let _ = gemm::matmul_v4_1(&a.tensor, &b.tensor, m, k, n, &dev);
        }
        dev.synchronize().map_err(|e| GpuHostError::Verification {
            test: "sgemm",
            detail: format!("{e}"),
        })?;

        let start = std::time::Instant::now();
        for _ in 0..iters {
            let _ = gemm::matmul_v4_1(&a.tensor, &b.tensor, m, k, n, &dev);
        }
        dev.synchronize().map_err(|e| GpuHostError::Verification {
            test: "sgemm",
            detail: format!("{e}"),
        })?;
        let v41_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // --- cuBLAS reference ---
        use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
        let blas = CudaBlas::new(dev.clone()).map_err(|e| GpuHostError::Verification {
            test: "sgemm",
            detail: format!("{e:?}"),
        })?;
        let mut c_dev = dev
            .alloc_zeros::<f32>(m * n)
            .map_err(|e| GpuHostError::Verification {
                test: "sgemm",
                detail: format!("{e}"),
            })?;
        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0f32,
            lda: n as i32,
            ldb: k as i32,
            beta: 0.0f32,
            ldc: n as i32,
        };
        for _ in 0..warmup {
            unsafe {
                blas.gemm(cfg, b.tensor.data(), a.tensor.data(), &mut c_dev)
                    .ok();
            }
        }
        dev.synchronize().map_err(|e| GpuHostError::Verification {
            test: "sgemm",
            detail: format!("{e}"),
        })?;

        let start = std::time::Instant::now();
        for _ in 0..iters {
            unsafe {
                blas.gemm(cfg, b.tensor.data(), a.tensor.data(), &mut c_dev)
                    .ok();
            }
        }
        dev.synchronize().map_err(|e| GpuHostError::Verification {
            test: "sgemm",
            detail: format!("{e}"),
        })?;
        let cublas_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let v4_gflops = flops / (v4_ms * 1e6);
        let v41_gflops = flops / (v41_ms * 1e6);
        let cublas_gflops = flops / (cublas_ms * 1e6);

        println!(
            "  {label:>22}  {v4_ms:>10.3} {v41_ms:>10.3} {cublas_ms:>10.3} {v4_gflops:>7.0} {v41_gflops:>7.0}"
        );
        println!(
            "  {:>22}  {:>9.1}% {:>9.1}% {:>10} {:>8} {:>8}",
            "",
            v4_gflops / cublas_gflops * 100.0,
            v41_gflops / cublas_gflops * 100.0,
            format!("{cublas_gflops:.0} GF"),
            "",
            ""
        );

        // Correctness check: compare V4.1 output vs cuBLAS at small size
        if m <= 1024 {
            let v41_result =
                gemm::matmul_v4_1(&a.tensor, &b.tensor, m, k, n, &dev).map_err(|e| {
                    GpuHostError::Verification {
                        test: "sgemm",
                        detail: format!("{e}"),
                    }
                })?;
            let v41_host = v41_result
                .to_host()
                .map_err(|e| GpuHostError::Verification {
                    test: "sgemm",
                    detail: format!("{e}"),
                })?;
            let cublas_host: Vec<f32> =
                dev.dtoh_sync_copy(&c_dev)
                    .map_err(|e| GpuHostError::Verification {
                        test: "sgemm",
                        detail: format!("{e}"),
                    })?;
            let max_err = v41_host
                .iter()
                .zip(cublas_host.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let mean_val =
                cublas_host.iter().map(|v| v.abs()).sum::<f32>() / cublas_host.len() as f32;
            let rel_err = if mean_val > 0.0 {
                max_err / mean_val
            } else {
                max_err
            };
            println!(
                "  {:>22}  max_err={max_err:.4}, rel_err={rel_err:.6}",
                "correctness"
            );
        }
    }

    println!("\n  SGEMM V4 Benchmark — DONE");
    Ok(())
}

/// Helper for SGEMM benchmark: wraps GpuTensor creation.
#[cfg(feature = "cublas")]
struct TensorForBench {
    tensor: gpu_host::nn::tensor::GpuTensor,
}

#[cfg(feature = "cublas")]
impl TensorForBench {
    fn new(data: &[f32], shape: &[usize], dev: &Arc<CudaDevice>) -> Result<Self> {
        let tensor = gpu_host::nn::tensor::GpuTensor::from_host(data, shape, dev).map_err(|e| {
            GpuHostError::Verification {
                test: "sgemm",
                detail: format!("{e}"),
            }
        })?;
        Ok(Self { tensor })
    }
}

/// Demo: std::thread::spawn on GPU with println!
///
/// Zero-param entry for hostcall: `std_thread_spawn_demo(result)`.
fn run_std_thread_spawn_demo(_dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- std::thread::spawn Demo (std-thread-gpu) ---");

    let module =
        gpu_host::gpu::GpuStdModule::load(KERNEL_STD_PTX, "std_thread_spawn_demo", 128, (1, 1, 1))?;

    let result_dev: cudarc::driver::CudaSlice<u32> = module.device().alloc_zeros::<u32>(3)?;
    let mut result_ptr = *result_dev.device_ptr();

    unsafe {
        module.launch_raw(&[&mut result_ptr as *mut u64 as *mut std::ffi::c_void])?;
    }

    let result: Vec<u32> = module.device().dtoh_sync_copy(&result_dev)?;
    module.finish();

    println!("  Thread 1 (sum 0..10): {} (expected 45)", result[0]);
    println!("  Thread 2 (5!):        {} (expected 120)", result[1]);
    println!("  Combined:             {} (expected 165)", result[2]);

    assert_eq!(result[0], 45, "thread 1 wrong");
    assert_eq!(result[1], 120, "thread 2 wrong");
    assert_eq!(result[2], 165, "combined wrong");

    println!("  std::thread::spawn Demo — PASSED");
    Ok(())
}

/// Demo: REAL std::thread::spawn on GPU with println!
///
/// Zero-param entry for hostcall: `real_std_thread_spawn(result)`.
fn run_real_std_thread_spawn(_dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- REAL std::thread::spawn on GPU ---");

    let module = gpu_host::gpu::GpuStdModule::load_with_print(
        KERNEL_STD_PTX,
        "real_std_thread_spawn",
        128,
        (1, 1, 1),
        Some(Box::new(|msg| {
            let s = String::from_utf8_lossy(msg);
            println!("  [GPU] {}", s.trim());
        })),
    )?;

    let result_dev: cudarc::driver::CudaSlice<u32> = module.device().alloc_zeros::<u32>(3)?;
    let mut result_ptr = *result_dev.device_ptr();

    unsafe {
        module.launch_raw(&[&mut result_ptr as *mut u64 as *mut std::ffi::c_void])?;
    }
    // Brief sleep so hostcall listener can flush remaining messages
    std::thread::sleep(std::time::Duration::from_millis(100));

    let result: Vec<u32> = module.device().dtoh_sync_copy(&result_dev)?;
    module.finish();

    println!("  Thread 1 (sum 0..10): {} (expected 45)", result[0]);
    println!("  Thread 2 (5!):        {} (expected 120)", result[1]);
    println!("  Combined:             {} (expected 165)", result[2]);

    assert_eq!(result[0], 45, "thread 1 wrong");
    assert_eq!(result[1], 120, "thread 2 wrong");
    assert_eq!(result[2], 165, "combined wrong");

    println!("  REAL std::thread::spawn — PASSED");
    Ok(())
}

/// Minimal std::thread::spawn test — no println in closures.
///
/// Zero-param entry for hostcall: `std_thread_spawn_minimal(result)`.
fn run_std_thread_spawn_minimal(_dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Minimal std::thread::spawn on GPU ---");

    let module = gpu_host::gpu::GpuStdModule::load_with_print(
        KERNEL_STD_PTX,
        "std_thread_spawn_minimal",
        128,
        (1, 1, 1),
        Some(Box::new(|msg| {
            let s = String::from_utf8_lossy(msg);
            println!("  [GPU] {}", s.trim());
        })),
    )?;

    let result_dev: cudarc::driver::CudaSlice<u32> = module.device().alloc_zeros::<u32>(3)?;
    let mut result_ptr = *result_dev.device_ptr();

    unsafe {
        module.launch_raw(&[&mut result_ptr as *mut u64 as *mut std::ffi::c_void])?;
    }
    std::thread::sleep(std::time::Duration::from_millis(100));

    let result: Vec<u32> = module.device().dtoh_sync_copy(&result_dev)?;
    module.finish();

    println!("  Thread 1 (sum 0..10): {} (expected 45)", result[0]);
    println!("  Thread 2 (5!):        {} (expected 120)", result[1]);
    println!("  Combined:             {} (expected 165)", result[2]);

    assert_eq!(result[0], 45, "thread 1 wrong");
    assert_eq!(result[1], 120, "thread 2 wrong");
    assert_eq!(result[2], 165, "combined wrong");

    println!("  Minimal std::thread::spawn — PASSED");
    Ok(())
}

/// Smoke tests for kernel_std module.
///
/// `kernel_std_println_smoke(result)` now uses zero-param entry for hostcall.
fn run_kernel_std_smoke(_dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- kernel_std smoke tests ---");

    // Test 1: trivial write (1 thread, no hostcall) — uses cudarc (no device global needed)
    {
        use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
        let dev = CudaDevice::new(0).map_err(GpuHostError::CudaInit)?;
        let ptx = cudarc::nvrtc::Ptx::from_src(KERNEL_STD_PTX);
        dev.load_ptx(ptx, "kstd_smoke", &["kernel_std_smoke_test"])
            .map_err(|e| GpuHostError::Verification {
                test: "kstd_smoke",
                detail: format!("PTX load: {e}"),
            })?;
        println!("  Module load: OK");

        let mut result_dev: cudarc::driver::CudaSlice<u32> = dev.alloc_zeros::<u32>(1)?;
        let f = dev
            .get_func("kstd_smoke", "kernel_std_smoke_test")
            .ok_or(GpuHostError::KernelNotFound("kernel_std_smoke_test"))?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            f.launch(cfg, (&mut result_dev,))?;
        }
        dev.synchronize()?;
        let r = dev.dtoh_sync_copy(&result_dev)?;
        println!("  smoke_test: {} (expected 0xBEEFCAFE)", r[0]);
        assert_eq!(r[0], 0xBEEF_CAFE);
        println!("  smoke_test: PASSED");
    }

    // Test 2: println (1 thread, with hostcall via device global)
    {
        let module = gpu_host::gpu::GpuStdModule::load(
            KERNEL_STD_PTX,
            "kernel_std_println_smoke",
            1,
            (1, 1, 1),
        )?;
        let result_dev: cudarc::driver::CudaSlice<u32> = module.device().alloc_zeros::<u32>(1)?;
        let mut result_ptr = *result_dev.device_ptr();

        unsafe {
            module.launch_raw(&[&mut result_ptr as *mut u64 as *mut std::ffi::c_void])?;
        }
        let r = module.device().dtoh_sync_copy(&result_dev)?;
        module.finish();
        println!("  println_smoke: {} (expected 1)", r[0]);
        assert_eq!(r[0], 1);
        println!("  println_smoke: PASSED");
    }

    // Test 3: thread pool (128 threads = 4 warps, no spawn) — no hostcall needed
    println!("  pool_smoke: launching 128 threads...");
    {
        use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
        let dev = CudaDevice::new(0).map_err(GpuHostError::CudaInit)?;
        let ptx = cudarc::nvrtc::Ptx::from_src(KERNEL_STD_PTX);
        dev.load_ptx(ptx, "kstd_pool", &["kernel_std_pool_smoke"])
            .map_err(|e| GpuHostError::Verification {
                test: "kstd_pool",
                detail: format!("PTX load: {e}"),
            })?;

        let mut result_dev: cudarc::driver::CudaSlice<u32> = dev.alloc_zeros::<u32>(1)?;
        let f = dev
            .get_func("kstd_pool", "kernel_std_pool_smoke")
            .ok_or(GpuHostError::KernelNotFound("kernel_std_pool_smoke"))?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            f.launch(cfg, (&mut result_dev,))?;
        }
        dev.synchronize()?;
        let r = dev.dtoh_sync_copy(&result_dev)?;
        println!("  pool_smoke: {} (expected 42)", r[0]);
        assert_eq!(r[0], 42);
        println!("  pool_smoke: PASSED");
    }

    println!("  All smoke tests PASSED");
    Ok(())
}

/// Test zero-parameter kernel launch via device global injection.
///
/// The host writes the hostcall pointer to `__HOSTCALL_BUF` in the loaded
/// module via `cuModuleGetGlobal_v2` + `cuMemcpyHtoD`. The kernel reads it
/// at entry via `gpu_runtime::entry::auto_init()` — no kernel parameters.
fn run_zero_param_test() -> Result<()> {
    use gpu_host::gpu;

    println!("\n--- Zero-Parameter Kernel Entry Test (kernel-entry.2) ---");
    println!("  Hostcall buffer injected via __HOSTCALL_BUF device global");
    println!("  Kernel: zero_param_hello() — no parameters");

    gpu::run_zero_param(KERNEL_STD_PTX, "zero_param_hello")?;

    println!("  zero_param_hello: PASSED (println! works with zero kernel args)");
    println!("  Zero-param kernel entry test — PASSED");
    Ok(())
}

/// Test the gpu::run() one-liner API.
fn run_gpu_api_test() -> Result<()> {
    use gpu_host::gpu;

    println!("\n--- gpu::run() API test (native-rust-dx) ---");

    println!("  gpu::launch(\"thread_spawn_test\", 4, 128)...");
    let result: Vec<u32> =
        gpu::launch("thread_spawn_test", 4, 128).map_err(|e| GpuHostError::Verification {
            test: "gpu_run",
            detail: format!("{e}"),
        })?;
    println!("    result = {:?}", result);
    assert_eq!(result[0], 42, "thread 1 should return 42");
    assert_eq!(result[1], 99, "thread 2 should return 99");
    assert_eq!(result[2], 3, "available_parallelism should be 3");
    assert_eq!(result[3], 0, "main thread should be warp 0");
    println!("  gpu::launch — PASSED");

    println!("  gpu::launch(\"gpu_kernel_demo\", 2, 128)...");
    let result2: Vec<u32> =
        gpu::launch("gpu_kernel_demo", 2, 128).map_err(|e| GpuHostError::Verification {
            test: "gpu_kernel_demo",
            detail: format!("{e}"),
        })?;
    println!("    result = {:?}", result2);
    assert_eq!(result2[0], 42);
    assert_eq!(result2[1], 99);
    println!("  extern \"gpu-kernel\" ABI — PASSED");

    println!("  gpu::run() API test — ALL PASSED");
    Ok(())
}

/// Test thread::spawn on GPU — warp-as-thread model.
fn run_thread_spawn_test(dev: Arc<CudaDevice>) -> Result<()> {
    use cudarc::driver::{LaunchAsync, LaunchConfig};

    println!("\n--- thread::spawn test (std-thread-gpu) ---");

    kernel_routes::load_kernel(
        &dev,
        kernel_routes::KernelModule::Test,
        "thread_tests",
        &["thread_spawn_test", "thread_reuse_test"],
    )?;

    // --- Test 1: basic spawn + join ---
    println!("  Test 1: spawn 2 threads, join results...");
    let f = dev
        .get_func("thread_tests", "thread_spawn_test")
        .ok_or(GpuHostError::KernelNotFound("thread_spawn_test"))?;

    let mut result_dev: cudarc::driver::CudaSlice<u32> = dev.alloc_zeros::<u32>(4)?;

    // 4 warps = 128 threads: warp 0 = main, warps 1-3 = workers
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        f.launch(config, (&mut result_dev,))?;
    }
    dev.synchronize()?;

    let result: Vec<u32> = dev.dtoh_sync_copy(&result_dev)?;
    println!("    result[0] (thread 1) = {} (expected 42)", result[0]);
    println!("    result[1] (thread 2) = {} (expected 99)", result[1]);
    println!("    result[2] (parallelism) = {} (expected 3)", result[2]);
    println!("    result[3] (main tid) = {} (expected 0)", result[3]);

    assert_eq!(result[0], 42, "thread 1 returned wrong value");
    assert_eq!(result[1], 99, "thread 2 returned wrong value");
    assert_eq!(result[2], 3, "wrong available_parallelism");
    assert_eq!(result[3], 0, "main thread should be warp 0");

    println!("  Test 1 — PASSED");

    // --- Test 2: spawn + reuse (4 tasks on 3 warps) ---
    println!("  Test 2: spawn 4 tasks with reuse...");
    let f2 = dev
        .get_func("thread_tests", "thread_reuse_test")
        .ok_or(GpuHostError::KernelNotFound("thread_reuse_test"))?;

    let mut result2_dev: cudarc::driver::CudaSlice<u32> = dev.alloc_zeros::<u32>(5)?;
    unsafe {
        f2.launch(config, (&mut result2_dev,))?;
    }
    dev.synchronize()?;

    let result2: Vec<u32> = dev.dtoh_sync_copy(&result2_dev)?;
    println!(
        "    tasks: [{}, {}, {}, {}] total={} (expected [10,20,30,40] total=100)",
        result2[0], result2[1], result2[2], result2[3], result2[4]
    );

    assert_eq!(result2[0], 10, "task 0 wrong");
    assert_eq!(result2[1], 20, "task 1 wrong");
    assert_eq!(result2[2], 30, "task 2 wrong");
    assert_eq!(result2[3], 40, "task 3 wrong");
    assert_eq!(result2[4], 100, "total wrong");

    println!("  Test 2 — PASSED");
    println!("  thread::spawn test — ALL PASSED");
    Ok(())
}

/// North Star Litmus Test: File::read → matmul → File::write in ONE kernel.
///
/// Host creates matmul_a.bin (8×4) and matmul_b.bin (4×6), launches the
/// matmul_io_compute kernel, then reads matmul_c.bin and verifies against
/// a CPU reference matmul.
///
/// Zero-param entry for hostcall: `matmul_io_compute(dims, result)`.
fn run_matmul_io_compute(_dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- North Star: File::read → matmul → File::write (coop-demo.1) ---");

    const M: usize = 8;
    const K: usize = 4;
    const N: usize = 6;

    // === Step 1: Create input matrix files ===
    let mut a = [0.0f32; M * K];
    for i in 0..M {
        for j in 0..K {
            a[i * K + j] = (i * K + j + 1) as f32;
        }
    }
    let mut b = [0.0f32; K * N];
    for i in 0..K {
        for j in 0..N {
            b[i * N + j] = ((i * N + j + 1) * 2) as f32;
        }
    }

    let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write("matmul_a.bin", &a_bytes).map_err(|e| GpuHostError::Verification {
        test: "matmul_io",
        detail: format!("write matmul_a.bin: {e}"),
    })?;
    std::fs::write("matmul_b.bin", &b_bytes).map_err(|e| GpuHostError::Verification {
        test: "matmul_io",
        detail: format!("write matmul_b.bin: {e}"),
    })?;
    println!(
        "  Created matmul_a.bin ({}x{} = {} bytes)",
        M,
        K,
        a_bytes.len()
    );
    println!(
        "  Created matmul_b.bin ({}x{} = {} bytes)",
        K,
        N,
        b_bytes.len()
    );

    // === Step 2: Load module with device global injection ===
    let module = gpu_host::gpu::GpuStdModule::load_with_print(
        KERNEL_STD_PTX,
        "matmul_io_compute",
        128,
        (1, 1, 1),
        Some(Box::new(|msg| {
            let s = String::from_utf8_lossy(msg);
            println!("  [GPU] {}", s.trim());
        })),
    )?;
    println!("  Module loaded");

    // === Step 3: Set up device memory ===
    let dims_data = vec![M as u32, K as u32, N as u32];
    let dims_dev = module.device().htod_sync_copy(&dims_data)?;
    let result_dev: cudarc::driver::CudaSlice<u32> = module.device().alloc_zeros::<u32>(8)?;

    // === Step 4: Launch kernel with (dims, result) ===
    let mut dims_ptr = *dims_dev.device_ptr() as u64;
    let mut result_ptr = *result_dev.device_ptr();

    println!(
        "  Launching matmul_io_compute ({}x{} x {}x{} -> {}x{})...",
        M, K, K, N, M, N
    );
    let start = std::time::Instant::now();
    unsafe {
        module.launch_raw(&[
            &mut dims_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut result_ptr as *mut u64 as *mut std::ffi::c_void,
        ])?;
    }
    let elapsed = start.elapsed();
    println!("  Kernel completed in {elapsed:?}");

    // Brief sleep for hostcall listener to flush
    std::thread::sleep(std::time::Duration::from_millis(200));

    // Read result markers
    let result_vals = module.device().dtoh_sync_copy(&result_dev)?;
    module.finish();

    let success = result_vals[0];
    let n_elements = result_vals[1];

    if success != 1 {
        let _ = std::fs::remove_file("matmul_a.bin");
        let _ = std::fs::remove_file("matmul_b.bin");
        let _ = std::fs::remove_file("matmul_c.bin");
        return Err(GpuHostError::Verification {
            test: "matmul_io_compute",
            detail: format!("kernel failed (success={success}, elements={n_elements})"),
        });
    }
    println!("  Kernel success: {} elements written", n_elements);

    // === Step 5: Read matmul_c.bin and verify ===
    let c_bytes = std::fs::read("matmul_c.bin").map_err(|e| GpuHostError::Verification {
        test: "matmul_io_compute",
        detail: format!("read matmul_c.bin: {e}"),
    })?;
    println!("  Read matmul_c.bin: {} bytes", c_bytes.len());

    if c_bytes.len() != M * N * 4 {
        let _ = std::fs::remove_file("matmul_a.bin");
        let _ = std::fs::remove_file("matmul_b.bin");
        let _ = std::fs::remove_file("matmul_c.bin");
        return Err(GpuHostError::Verification {
            test: "matmul_io_compute",
            detail: format!(
                "output size mismatch: expected {} bytes, got {}",
                M * N * 4,
                c_bytes.len()
            ),
        });
    }

    let gpu_c: Vec<f32> = c_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let mut expected = [0.0f32; M * N];
    for i in 0..M {
        for j in 0..N {
            let mut sum = 0.0f32;
            for p in 0..K {
                sum += a[i * K + p] * b[p * N + j];
            }
            expected[i * N + j] = sum;
        }
    }

    let mut ok = true;
    let mut mismatch_count = 0usize;
    for i in 0..M {
        for j in 0..N {
            let idx = i * N + j;
            let gpu_val = gpu_c[idx];
            let cpu_val = expected[idx];
            if (gpu_val - cpu_val).abs() > 1e-3 {
                if mismatch_count < 5 {
                    println!("  MISMATCH at C[{i}][{j}]: GPU={gpu_val}, CPU={cpu_val}");
                }
                ok = false;
                mismatch_count += 1;
            }
        }
    }

    let _ = std::fs::remove_file("matmul_a.bin");
    let _ = std::fs::remove_file("matmul_b.bin");
    let _ = std::fs::remove_file("matmul_c.bin");

    if !ok {
        return Err(GpuHostError::Verification {
            test: "matmul_io_compute",
            detail: format!("{mismatch_count} mismatches out of {}", M * N),
        });
    }

    println!(
        "  All {} elements match CPU reference (tolerance 1e-3)",
        M * N
    );
    println!(
        "  Sample: C[0][0]={:.1}, C[7][5]={:.1}",
        gpu_c[0],
        gpu_c[M * N - 1]
    );
    println!("  ========================================");
    println!("  NORTH STAR LITMUS TEST — PASSED");
    println!("  File::read → matmul → File::write in ONE kernel");
    println!("  ========================================");
    Ok(())
}

// ============================================================
// GPU Coroutine Generator Tests (coro-impl.2)
// ============================================================

/// Run all GPU coroutine generator tests.
///
/// Tests:
/// 1. test_gpu_generator_fibonacci — FibGenerator streaming pipeline
/// 2. test_gpu_streaming_pipeline — CounterGenerator with square-and-accumulate consumer
/// 3. test_gpu_multi_generator — Multiple independent generators + edge cases
fn run_generator_tests() -> Result<()> {
    use gpu_host::gpu;

    println!("\n--- GPU Coroutine Generator Tests (coro-impl.2) ---");

    // Test 1: Fibonacci generator streaming pipeline
    println!("\n  Test 1: Fibonacci generator streaming pipeline...");
    gpu::run_zero_param(KERNEL_STD_PTX, "test_gpu_generator_fibonacci")?;
    println!("  test_gpu_generator_fibonacci — PASSED");

    // Test 2: Streaming pipeline with CounterGenerator
    println!("\n  Test 2: Streaming pipeline (counter → square → accumulate)...");
    gpu::run_zero_param(KERNEL_STD_PTX, "test_gpu_streaming_pipeline")?;
    println!("  test_gpu_streaming_pipeline — PASSED");

    // Test 3: Multiple generators (Counter + Fib + edge cases)
    println!("\n  Test 3: Multiple independent generators...");
    gpu::run_zero_param(KERNEL_STD_PTX, "test_gpu_multi_generator")?;
    println!("  test_gpu_multi_generator — PASSED");

    println!("\n  ========================================");
    println!("  GPU Coroutine Generator Tests — ALL PASSED");
    println!("  - Fibonacci streaming pipeline: zero-buffered producer→consumer");
    println!("  - Counter pipeline: yield → transform → accumulate");
    println!("  - Multiple generators: independent instances + edge cases");
    println!("  ========================================");
    Ok(())
}
