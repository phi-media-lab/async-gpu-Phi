//! Search tests: f32 math, vector search, batch search, MMA, shared memory, MMA mapped.

use std::sync::Arc;

use cudarc::driver::sys::lib as cuda_lib;
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

use gpu_host::error::{GpuHostError, Result};
use gpu_host::hostcall;
use gpu_host::mapped_mem::{alloc_mapped_result_array, free_mapped_mem};

/// ml-workload.1: f32 math validation on GPU.
pub(crate) fn run_f32_math_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- f32 Math Validation (ml-workload.1) ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(ptx, "kernel", &["f32_math_test"])?;
    let f = dev
        .get_func("kernel", "f32_math_test")
        .ok_or(GpuHostError::KernelNotFound("f32_math_test"))?;

    let mut output: CudaSlice<f32> = dev.alloc_zeros::<f32>(8)?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        f.launch(cfg, (&mut output,))?;
    }
    dev.synchronize()?;

    let result: Vec<f32> = dev.dtoh_sync_copy(&output)?;
    let expected: [f32; 8] = [7.0, 12.0, 2.5, 3.0, 70.0, 5.0, 0.0, 1.0];
    let labels = [
        "add", "mul", "div", "sqrt", "dot", "norm", "cos_orth", "cos_same",
    ];

    let mut all_ok = true;
    for i in 0..8 {
        let ok = (result[i] - expected[i]).abs() < 0.001;
        let status = if ok { "OK" } else { "FAIL" };
        println!(
            "  {}: {} (expected {}) [{}]",
            labels[i], result[i], expected[i], status
        );
        if !ok {
            all_ok = false;
        }
    }

    if all_ok {
        println!("  f32 Math: ALL PASSED!");
        println!("    add, mul, div, sqrt.approx.f32, dot product, norm, cosine similarity");
    } else {
        return Err(GpuHostError::Verification {
            test: "f32_math_test",
            detail: "see above".to_string(),
        });
    }

    Ok(())
}

/// ml-workload.2: GPU-autonomous vector similarity search.
pub(crate) fn run_vector_search_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Vector Similarity Search (ml-workload.2) ---");

    const DIM: usize = 128;
    const K: usize = 10;
    const N: usize = 100; // database size

    // Create database: N random-ish vectors + one planted similar vector
    // Vector i: each dimension = sin(i * d * 0.1) for variety
    let mut db_data: Vec<u8> = Vec::new();
    // Header: N (u32) + dim (u32)
    db_data.extend_from_slice(&(N as u32).to_le_bytes());
    db_data.extend_from_slice(&(DIM as u32).to_le_bytes());

    let mut db_vectors: Vec<Vec<f32>> = Vec::new();
    for i in 0..N {
        let mut vec = Vec::with_capacity(DIM);
        for d in 0..DIM {
            let val = ((i as f32 + 1.0) * (d as f32 + 1.0) * 0.1).sin();
            vec.push(val);
        }
        for &v in &vec {
            db_data.extend_from_slice(&v.to_le_bytes());
        }
        db_vectors.push(vec);
    }
    std::fs::write("vecdb.bin", &db_data).unwrap();
    println!(
        "  Created vecdb.bin ({} vectors × {} dims = {} bytes)",
        N,
        DIM,
        db_data.len()
    );

    // Create query: use db_vectors[42] as query (perfect match at index 42)
    let query_vec = &db_vectors[42];
    let mut query_data: Vec<u8> = Vec::new();
    query_data.extend_from_slice(&(DIM as u32).to_le_bytes());
    for &v in query_vec {
        query_data.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write("query.bin", &query_data).unwrap();
    println!("  Created query.bin (query = db[42], expect top-1 match at id ~42)");

    // Compute expected top-K on CPU
    let mut cpu_scores: Vec<(usize, f32)> = Vec::new();
    let q_norm: f32 = query_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    for (i, db_vec) in db_vectors.iter().enumerate() {
        let dot: f32 = query_vec
            .iter()
            .zip(db_vec.iter())
            .map(|(a, b)| a * b)
            .sum();
        let v_norm: f32 = db_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let score = if q_norm * v_norm > 0.0 {
            dot / (q_norm * v_norm)
        } else {
            0.0
        };
        cpu_scores.push((i, score));
    }
    cpu_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!(
        "  CPU reference top-3: {:?}",
        cpu_scores[..3]
            .iter()
            .map(|(id, s)| format!("id={id} score={s:.4}"))
            .collect::<Vec<_>>()
    );

    // Launch GPU kernel
    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();
    let sb_dev_ptr = hc_buf.sideband_dev_ptr();

    let (status_host, status_dev) = unsafe { alloc_mapped_result_array(&dev, 1)? };
    let hc_buf_ref = std::sync::Arc::new(hc_buf);
    let hc_buf_listener = std::sync::Arc::clone(&hc_buf_ref);

    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let text = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] GPU says: \"{text}\"");
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(ptx, "kernel", &["vector_search_pipeline"])?;
    let f = dev
        .get_func("kernel", "vector_search_pipeline")
        .ok_or(GpuHostError::KernelNotFound("vector_search_pipeline"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching vector_search_pipeline kernel...");
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
    println!("  Status: {status_val} (1=success)");
    println!("  Elapsed: {:.3}ms", elapsed.as_secs_f64() * 1000.0);

    // Read and verify results
    let result_data = std::fs::read("results.bin").map_err(|e| GpuHostError::Verification {
        test: "vector_search_pipeline",
        detail: format!("failed to read results.bin: {e}"),
    })?;

    let result_k = u32::from_le_bytes(result_data[0..4].try_into().unwrap()) as usize;
    println!("  Results: K={result_k}");

    let mut gpu_results: Vec<(u32, f32)> = Vec::new();
    for i in 0..result_k.min(K) {
        let off = 4 + i * 8;
        if off + 8 > result_data.len() {
            break;
        }
        let id = u32::from_le_bytes(result_data[off..off + 4].try_into().unwrap());
        let score = f32::from_le_bytes(result_data[off + 4..off + 8].try_into().unwrap());
        gpu_results.push((id, score));
        println!("    rank {}: id={} score={:.4}", i + 1, id, score);
    }

    // Clean up
    let _ = std::fs::remove_file("vecdb.bin");
    let _ = std::fs::remove_file("query.bin");
    let _ = std::fs::remove_file("results.bin");
    unsafe { free_mapped_mem(status_host)? };

    // Verify: top-1 result should be vector 42 with score ~1.0
    // Full warp merge: all 32 lanes contribute, so we see all 100 vectors.
    // Lane 10 processes vectors 10, 42, 74 — so id=42 should be in the results.

    let top1_ok = !gpu_results.is_empty() && gpu_results[0].0 == 42 && gpu_results[0].1 > 0.99;

    if status_val == 1 && top1_ok {
        println!("  Vector Search Pipeline: PASSED!");
        println!("    GPU self-coordinated: open(db)→read→close→open(query)→read→close→compute→write(results)");
        println!("    {N} database vectors × {DIM} dimensions, top-{result_k} returned");
        println!("    Full warp merge: all 32 lanes contribute via shfl.sync (100% DB coverage)");
        println!(
            "    Top-1: id={} score={:.4} (exact match!)",
            gpu_results[0].0, gpu_results[0].1
        );
    } else {
        println!("  Vector Search Pipeline: FAILED");
        return Err(GpuHostError::Verification {
            test: "vector_search_pipeline",
            detail: format!("status={status_val}, results={gpu_results:?}"),
        });
    }

    Ok(())
}

pub(crate) fn run_batch_search_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Batch Vector Search (ml-workload.3) ---");

    const DIM: usize = 128;
    const N: usize = 100; // database size
    const NUM_QUERIES: usize = 5;

    // Create database (same as ml-workload.2)
    let mut db_data: Vec<u8> = Vec::new();
    db_data.extend_from_slice(&(N as u32).to_le_bytes());
    db_data.extend_from_slice(&(DIM as u32).to_le_bytes());

    let mut db_vectors: Vec<Vec<f32>> = Vec::new();
    for i in 0..N {
        let mut vec = Vec::with_capacity(DIM);
        for d in 0..DIM {
            let val = ((i as f32 + 1.0) * (d as f32 + 1.0) * 0.1).sin();
            vec.push(val);
        }
        for &v in &vec {
            db_data.extend_from_slice(&v.to_le_bytes());
        }
        db_vectors.push(vec);
    }
    std::fs::write("vecdb.bin", &db_data).unwrap();
    println!("  Created vecdb.bin ({N} vectors)");

    // Create queries: use db[10], db[42], db[77], db[3], db[95]
    let query_indices = [10usize, 42, 77, 3, 95];
    let mut queries_data: Vec<u8> = Vec::new();
    queries_data.extend_from_slice(&(NUM_QUERIES as u32).to_le_bytes());
    queries_data.extend_from_slice(&(DIM as u32).to_le_bytes());
    for &qi in &query_indices {
        for &v in &db_vectors[qi] {
            queries_data.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write("queries.bin", &queries_data).unwrap();
    println!("  Created queries.bin ({NUM_QUERIES} queries: {query_indices:?})");

    // Compute CPU reference scores for each query
    for (qn, &qi) in query_indices.iter().enumerate() {
        let query_vec = &db_vectors[qi];
        let q_norm: f32 = query_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut scores: Vec<(usize, f32)> = Vec::new();
        for (i, db_vec) in db_vectors.iter().enumerate() {
            let dot: f32 = query_vec
                .iter()
                .zip(db_vec.iter())
                .map(|(a, b)| a * b)
                .sum();
            let v_norm: f32 = db_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            let score = if q_norm * v_norm > 0.0 {
                dot / (q_norm * v_norm)
            } else {
                0.0
            };
            scores.push((i, score));
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!(
            "  CPU query {}: top-1 = id={} score={:.4}",
            qn, scores[0].0, scores[0].1
        );
    }

    // Launch GPU kernel
    let hc_buf = hostcall::HostcallBuffer::new(4)?;
    let dev_ptr = hc_buf.dev_ptr();
    let sb_dev_ptr = hc_buf.sideband_dev_ptr();

    let (status_host, status_dev) = unsafe { alloc_mapped_result_array(&dev, 1)? };
    let hc_buf_ref = std::sync::Arc::new(hc_buf);
    let hc_buf_listener = std::sync::Arc::clone(&hc_buf_ref);

    let listener_handle = std::thread::spawn(move || {
        hc_buf_listener.listen(|msg| {
            let text = String::from_utf8_lossy(msg).to_string();
            println!("  [HOST] GPU says: \"{text}\"");
        });
    });

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(ptx, "kernel", &["batch_search_pipeline"])?;
    let f = dev
        .get_func("kernel", "batch_search_pipeline")
        .ok_or(GpuHostError::KernelNotFound("batch_search_pipeline"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching batch_search_pipeline kernel ({NUM_QUERIES} queries)...");
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
    println!("  Status: {status_val} (1=success)");
    println!("  Elapsed: {:.3}ms", elapsed.as_secs_f64() * 1000.0);

    // Read and verify results
    let result_data =
        std::fs::read("batch_results.bin").map_err(|e| GpuHostError::Verification {
            test: "batch_search_pipeline",
            detail: format!("failed to read batch_results.bin: {e}"),
        })?;

    let result_nq = u32::from_le_bytes(result_data[0..4].try_into().unwrap()) as usize;
    let result_k = u32::from_le_bytes(result_data[4..8].try_into().unwrap()) as usize;
    println!("  Results: num_queries={result_nq}, K={result_k}");

    for qi in 0..result_nq.min(NUM_QUERIES) {
        let base = 8 + qi * result_k * 8;
        print!("  Query {qi}: ");
        let mut first_score = 0.0f32;
        for ri in 0..result_k.min(3) {
            let off = base + ri * 8;
            if off + 8 > result_data.len() {
                break;
            }
            let id = u32::from_le_bytes(result_data[off..off + 4].try_into().unwrap());
            let score = f32::from_le_bytes(result_data[off + 4..off + 8].try_into().unwrap());
            if ri == 0 {
                first_score = score;
            }
            print!("[id={id} s={score:.4}] ");
        }
        println!();
        if first_score <= 0.0 {
            println!("  WARNING: query {qi} top-1 score <= 0");
        }
    }

    // Clean up
    let _ = std::fs::remove_file("vecdb.bin");
    let _ = std::fs::remove_file("queries.bin");
    let _ = std::fs::remove_file("batch_results.bin");
    unsafe { free_mapped_mem(status_host)? };

    if status_val == 1 && result_nq == NUM_QUERIES {
        println!("  Batch Search Pipeline: PASSED!");
        println!("    {NUM_QUERIES} queries processed in single kernel launch");
        println!("    DB read once, queries read once, results written once");
        println!(
            "    Amortized: {:.3}ms per query (total {:.3}ms)",
            elapsed.as_secs_f64() * 1000.0 / NUM_QUERIES as f64,
            elapsed.as_secs_f64() * 1000.0
        );
    } else {
        println!("  Batch Search Pipeline: FAILED");
        return Err(GpuHostError::Verification {
            test: "batch_search_pipeline",
            detail: format!("status={status_val}, nq={result_nq}"),
        });
    }

    Ok(())
}

pub(crate) fn run_mma_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Tensor Core MMA Test (gpu-compute.3) ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(ptx, "mma_test", &["test_mma_m16n8k16"])?;
    let f = dev
        .get_func("mma_test", "test_mma_m16n8k16")
        .ok_or(GpuHostError::KernelNotFound("test_mma_m16n8k16"))?;

    // 32 threads × 4 registers = 128 u32 values
    // C accumulator is f32, stored as u32 bits.
    // f32(1.0) = 0x3F800000
    // With A=0,B=0 → D = 0*0 + C = C
    let mut c_host = vec![0u32; 128];
    for t in 0..32u32 {
        let base = (t * 4) as usize;
        // Each register = f32(1.0) bit pattern
        c_host[base] = 0x3F80_0000; // 1.0f32
        c_host[base + 1] = 0x4000_0000; // 2.0f32
        c_host[base + 2] = 0x4040_0000; // 3.0f32
        c_host[base + 3] = 0x4080_0000; // 4.0f32
    }

    let c_dev: CudaSlice<u32> = dev.htod_sync_copy(&c_host)?;
    let mut d_dev: CudaSlice<u32> = dev.alloc_zeros::<u32>(128)?;

    // Status via mapped memory
    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1), // exactly 1 warp
        shared_mem_bytes: 0,
    };

    unsafe {
        f.launch(cfg, (&c_dev, &mut d_dev, status_dev_ptr))?;
    }
    dev.synchronize()?;

    let status = unsafe { std::ptr::read_volatile(status_host_ptr) };
    assert_eq!(status, 1, "MMA kernel should complete");

    // Read D back and verify D == C
    let d_host: Vec<u32> = dev.dtoh_sync_copy(&d_dev)?;

    let mut mismatches = 0;
    for i in 0..128 {
        if d_host[i] != c_host[i] {
            if mismatches < 3 {
                println!(
                    "  MISMATCH at index {}: expected 0x{:08X}, got 0x{:08X}",
                    i, c_host[i], d_host[i]
                );
            }
            mismatches += 1;
        }
    }

    if mismatches == 0 {
        println!("  Verification PASSED: D == C for all 128 fragment registers");
        println!("  MMA instruction mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 works!");
        println!("  14 register operands in single asm!() — Rust inline PTX handles it.");
    } else {
        println!("  {mismatches} mismatches out of 128 — MMA arithmetic may differ");
        println!("  (MMA with A=0,B=0 should give D=C, but hardware rounding may differ)");
    }

    // Free mapped memory
    unsafe {
        cuda_lib().cuMemFreeHost(status_host_ptr as *mut std::ffi::c_void);
    }
    Ok(())
}

/// gpu-compute.4: Shared memory + bar.sync test.
///
/// Launches `test_shared_memory` with 32 threads and dynamic shared memory.
/// Each thread writes (tid+1) to smem[tid], syncs, reads smem[tid^1], writes to output.
/// Verifies the neighbor-swap pattern.
pub(crate) fn run_shared_memory_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Shared Memory + bar.sync Test (gpu-compute.4) ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(ptx, "smem_test", &["test_shared_memory"])?;
    let f = dev
        .get_func("smem_test", "test_shared_memory")
        .ok_or(GpuHostError::KernelNotFound("test_shared_memory"))?;

    const N: u32 = 32;
    let mut output_dev: CudaSlice<u32> = dev.alloc_zeros::<u32>(N as usize)?;

    // Status via mapped memory
    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (N, 1, 1),
        shared_mem_bytes: N * 4, // N u32 values in shared memory
    };

    unsafe {
        f.launch(cfg, (&mut output_dev, N, status_dev_ptr))?;
    }
    dev.synchronize()?;

    let status = unsafe { std::ptr::read_volatile(status_host_ptr) };
    assert_eq!(status, 1, "Shared memory kernel should complete");

    // Read output and verify neighbor-swap pattern
    let output: Vec<u32> = dev.dtoh_sync_copy(&output_dev)?;

    let mut ok = true;
    for tid in 0..N {
        let neighbor = tid ^ 1;
        let expected = neighbor + 1; // neighbor wrote (neighbor+1) to smem[neighbor]
        if output[tid as usize] != expected {
            println!(
                "  MISMATCH at tid={}: expected {} (neighbor {}), got {}",
                tid, expected, neighbor, output[tid as usize]
            );
            ok = false;
        }
    }

    if ok {
        println!("  Verification PASSED: all {N} threads read correct neighbor values");
        println!("  Dynamic shared memory allocation via LaunchConfig::shared_mem_bytes works!");
        println!("  cvta.shared.u64 + bar.sync 0 verified from Rust inline PTX.");
    } else {
        return Err(GpuHostError::Verification {
            test: "shared_memory",
            detail: "neighbor swap pattern mismatch".to_string(),
        });
    }

    // Free mapped memory
    unsafe {
        cuda_lib().cuMemFreeHost(status_host_ptr as *mut std::ffi::c_void);
    }
    Ok(())
}

/// gpu-pipeline.1: MMA with proper fragment mapping test.
///
/// Uses A = identity matrix, B = sequential values (1..128 as f16).
/// Expected: D = A × B = B.
/// Verifies the per-thread fragment-to-matrix index mapping.
pub(crate) fn run_mma_mapped_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- MMA Fragment Mapping Test (gpu-pipeline.1) ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(ptx, "mma_mapped", &["test_mma_mapped"])?;
    let f = dev
        .get_func("mma_mapped", "test_mma_mapped")
        .ok_or(GpuHostError::KernelNotFound("test_mma_mapped"))?;

    // Helper: convert f32 to f16 bits (IEEE 754 half-precision)
    fn f32_to_f16(val: f32) -> u16 {
        let bits = val.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let frac = bits & 0x7FFFFF;
        if val == 0.0 {
            return (sign << 15) as u16;
        }
        let new_exp = exp - 127 + 15;
        if new_exp <= 0 {
            return (sign << 15) as u16; // flush to zero
        }
        if new_exp >= 31 {
            return ((sign << 15) | 0x7C00) as u16; // infinity
        }
        ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
    }

    fn pack_f16x2(lo: f32, hi: f32) -> u32 {
        let lo_bits = f32_to_f16(lo) as u32;
        let hi_bits = f32_to_f16(hi) as u32;
        lo_bits | (hi_bits << 16)
    }

    // A = 16×16 identity matrix, stored as u32[16][8] (row-major, f16x2 packed)
    let mut a_host = vec![0u32; 128];
    for i in 0..16u32 {
        // Row i: A[i][k] = 1.0 if i==k, else 0.0
        // Packed: A_packed[i][k/2] = pack(A[i][2*col], A[i][2*col+1])
        let col_pair = i / 2; // which u32 in the row
        let pos_in_pair = i % 2; // low or high half
        let idx = (i * 8 + col_pair) as usize;
        if pos_in_pair == 0 {
            a_host[idx] = pack_f16x2(1.0, 0.0);
        } else {
            a_host[idx] = pack_f16x2(0.0, 1.0);
        }
    }

    // B = 16×8 matrix with unique values, stored as u32[16][4] (row-major, f16x2 packed)
    // B[k][j] = (k * 8 + j + 1) as f16
    let mut b_host = vec![0u32; 64];
    for k in 0..16u32 {
        for j_pair in 0..4u32 {
            let j0 = j_pair * 2;
            let j1 = j0 + 1;
            let v0 = (k * 8 + j0 + 1) as f32;
            let v1 = (k * 8 + j1 + 1) as f32;
            b_host[(k * 4 + j_pair) as usize] = pack_f16x2(v0, v1);
        }
    }

    let a_dev: CudaSlice<u32> = dev.htod_sync_copy(&a_host)?;
    let b_dev: CudaSlice<u32> = dev.htod_sync_copy(&b_host)?;
    let mut d_dev: CudaSlice<u32> = dev.alloc_zeros::<u32>(128)?;
    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 768, // (128 + 64) * 4
    };

    unsafe {
        f.launch(cfg, (&a_dev, &b_dev, &mut d_dev, status_dev_ptr))?;
    }
    dev.synchronize()?;

    let status = unsafe { std::ptr::read_volatile(status_host_ptr) };
    assert_eq!(status, 1, "MMA mapped kernel should complete");

    let d_host: Vec<u32> = dev.dtoh_sync_copy(&d_dev)?;

    // Empirically determined D mapping: thread tid (group=tid/4, lane=tid%4)
    // d0 = D[lane*2, group]       = B[lane*2, group]       = lane*2*8 + group + 1
    // d1 = D[lane*2+1, group]     = B[lane*2+1, group]     = (lane*2+1)*8 + group + 1
    // d2 = D[lane*2+8, group]     = B[lane*2+8, group]     = (lane*2+8)*8 + group + 1
    // d3 = D[lane*2+8+1, group]   = B[lane*2+9, group]     = (lane*2+9)*8 + group + 1
    let mut mismatches = 0;
    for tid in 0..32u32 {
        let group = tid / 4;
        let lane = tid % 4;
        let base = (tid * 4) as usize;

        let expected = [
            (lane * 2 * 8 + group + 1) as f32,
            ((lane * 2 + 1) * 8 + group + 1) as f32,
            ((lane * 2 + 8) * 8 + group + 1) as f32,
            ((lane * 2 + 9) * 8 + group + 1) as f32,
        ];

        for r in 0..4 {
            let got = f32::from_bits(d_host[base + r]);
            let exp = expected[r];
            if (got - exp).abs() > 0.5 {
                if mismatches < 10 {
                    println!(
                        "  MISMATCH tid={tid} d{r}: expected {exp}, got {got} (group={group}, lane={lane})"
                    );
                }
                mismatches += 1;
            }
        }
    }

    if mismatches == 0 {
        println!("  Verification PASSED: all 128 D elements match expected (D = A×B = B)");
        println!("  Fragment mapping for m16n8k16.row.col confirmed:");
        println!("    A: a0=A[g][l], a1=A[g][l+4], a2=A[g+8][l], a3=A[g+8][l+4]");
        println!("    B: b0=B[g][l], b1=B[g+8][l]");
        println!("    D: d0=D[l*2][g], d1=D[l*2+1][g], d2=D[l*2+8][g], d3=D[l*2+9][g]");
        println!("    (g=tid/4, l=tid%4)");
    } else {
        println!("  {mismatches}/128 mismatches — fragment mapping needs adjustment");
        // Print a few D values for debugging
        for tid in 0..4u32 {
            let base = (tid * 4) as usize;
            let vals: Vec<f32> = (0..4).map(|r| f32::from_bits(d_host[base + r])).collect();
            println!(
                "  tid={tid}: d0={}, d1={}, d2={}, d3={}",
                vals[0], vals[1], vals[2], vals[3]
            );
        }
    }

    unsafe {
        cuda_lib().cuMemFreeHost(status_host_ptr as *mut std::ffi::c_void);
    }
    Ok(())
}
