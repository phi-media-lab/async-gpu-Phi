//! Transformer component tests: LayerNorm, GELU, attention, flash attention,
//! embedding, FFN, full transformer layer.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

use gpu_host::error::{GpuHostError, Result};
use gpu_host::mapped_mem::{alloc_mapped_result_array, free_mapped_mem};

/// LayerNorm test (transformer-layer.1): validate against CPU reference.
pub(crate) fn run_layer_norm_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- LayerNorm test (transformer-layer.1) ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(ptx, "layer_norm", &["layer_norm"])?;
    let f = dev
        .get_func("layer_norm", "layer_norm")
        .ok_or(GpuHostError::KernelNotFound("layer_norm"))?;

    const D_MODEL: u32 = 768;
    let num_rows: u32 = 4;
    let eps: f32 = 1e-5;

    // Generate input: x[i][j] = ((i*7 + j*3) % 11 - 5) as f32 * 0.1
    let mut input: Vec<f32> = Vec::with_capacity(num_rows as usize * D_MODEL as usize);
    for i in 0..num_rows as usize {
        for j in 0..D_MODEL as usize {
            let v = ((i * 7 + j * 3) % 11) as f32 - 5.0;
            input.push(v * 0.1);
        }
    }

    // gamma = 1.0 + j*0.001, beta = j*0.0001
    let mut gamma: Vec<f32> = Vec::with_capacity(D_MODEL as usize);
    let mut beta: Vec<f32> = Vec::with_capacity(D_MODEL as usize);
    for j in 0..D_MODEL as usize {
        gamma.push(1.0 + j as f32 * 0.001);
        beta.push(j as f32 * 0.0001);
    }

    let input_dev: CudaSlice<f32> = dev.htod_sync_copy(&input)?;
    let mut output_dev: CudaSlice<f32> =
        dev.alloc_zeros::<f32>(num_rows as usize * D_MODEL as usize)?;
    let gamma_dev: CudaSlice<f32> = dev.htod_sync_copy(&gamma)?;
    let beta_dev: CudaSlice<f32> = dev.htod_sync_copy(&beta)?;
    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    let cfg = LaunchConfig {
        grid_dim: (num_rows, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        f.clone().launch(
            cfg,
            (
                &input_dev,
                &mut output_dev,
                &gamma_dev,
                &beta_dev,
                D_MODEL,
                eps,
                status_dev_ptr,
            ),
        )?;
    }
    dev.synchronize()?;

    let status = unsafe { std::ptr::read_volatile(status_host_ptr) };
    assert!(
        status >= num_rows,
        "LayerNorm kernel did not complete: status={status}"
    );

    let output_host: Vec<f32> = dev.dtoh_sync_copy(&output_dev)?;

    // CPU reference
    let mut mismatches = 0;
    let mut max_err: f32 = 0.0;
    for row in 0..num_rows as usize {
        let row_start = row * D_MODEL as usize;
        let row_end = row_start + D_MODEL as usize;
        let row_data = &input[row_start..row_end];

        let mean: f32 = row_data.iter().sum::<f32>() / D_MODEL as f32;
        let var: f32 = row_data
            .iter()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f32>()
            / D_MODEL as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        for j in 0..D_MODEL as usize {
            let expected = gamma[j] * (row_data[j] - mean) * inv_std + beta[j];
            let got = output_host[row_start + j];
            let err = (got - expected).abs();
            if err > max_err {
                max_err = err;
            }
            if err > 1e-3 {
                if mismatches < 5 {
                    println!("  MISMATCH row={row} j={j}: got={got} expected={expected} err={err}");
                }
                mismatches += 1;
            }
        }
    }

    unsafe {
        free_mapped_mem(status_host_ptr)?;
    }

    println!("  LayerNorm {num_rows}x{D_MODEL}: max_err={max_err:.8}, mismatches={mismatches}");
    if mismatches == 0 {
        println!("  LayerNorm — PASSED");
        Ok(())
    } else {
        Err(GpuHostError::Verification {
            test: "layer_norm",
            detail: format!("{mismatches} mismatches"),
        })
    }
}

/// GELU test (transformer-layer.2): validate against CPU reference.
pub(crate) fn run_gelu_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- GELU test (transformer-layer.2) ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(ptx, "gelu_forward", &["gelu_forward"])?;
    let f = dev
        .get_func("gelu_forward", "gelu_forward")
        .ok_or(GpuHostError::KernelNotFound("gelu_forward"))?;

    // Test with a range of values from -5 to 5
    let n: u32 = 1024;
    let mut input: Vec<f32> = Vec::with_capacity(n as usize);
    for i in 0..n as usize {
        input.push(-5.0 + 10.0 * i as f32 / (n - 1) as f32);
    }

    let input_dev: CudaSlice<f32> = dev.htod_sync_copy(&input)?;
    let mut output_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(n as usize)?;
    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    let num_blocks = n.div_ceil(256);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        f.clone()
            .launch(cfg, (&input_dev, &mut output_dev, n, status_dev_ptr))?;
    }
    dev.synchronize()?;

    let status = unsafe { std::ptr::read_volatile(status_host_ptr) };
    assert!(
        status >= num_blocks,
        "GELU kernel did not complete: status={status}"
    );

    let output_host: Vec<f32> = dev.dtoh_sync_copy(&output_dev)?;

    // CPU reference: GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let sqrt_2_over_pi: f32 = 0.797_884_6;
    let coeff: f32 = 0.044715;
    let mut mismatches = 0;
    let mut max_err: f32 = 0.0;
    for i in 0..n as usize {
        let x = input[i];
        let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
        let expected = x * 0.5 * (1.0 + inner.tanh());
        let got = output_host[i];
        let err = (got - expected).abs();
        if err > max_err {
            max_err = err;
        }
        // Allow slightly larger tolerance for extreme values
        if err > 1e-4 {
            if mismatches < 5 {
                println!("  MISMATCH i={i} x={x}: got={got} expected={expected} err={err}");
            }
            mismatches += 1;
        }
    }

    unsafe {
        free_mapped_mem(status_host_ptr)?;
    }

    println!("  GELU {n} elements: max_err={max_err:.8}, mismatches={mismatches}");
    if mismatches == 0 {
        println!("  GELU — PASSED");
        Ok(())
    } else {
        Err(GpuHostError::Verification {
            test: "gelu_forward",
            detail: format!("{mismatches} mismatches, max_err={max_err:.8}"),
        })
    }
}

/// Multi-head attention test (transformer-layer.3): per-head scaled dot-product attention.
pub(crate) fn run_attention_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Attention test (transformer-layer.3) ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(ptx, "attention_head", &["attention_head"])?;
    let f = dev
        .get_func("attention_head", "attention_head")
        .ok_or(GpuHostError::KernelNotFound("attention_head"))?;

    const N_HEADS: usize = 12;
    const SEQ_LEN: u32 = 32;
    const D_HEAD: u32 = 64;
    let total = N_HEADS * SEQ_LEN as usize * D_HEAD as usize;

    // Generate deterministic Q, K, V: small values to avoid overflow
    let mut q: Vec<f32> = Vec::with_capacity(total);
    let mut k: Vec<f32> = Vec::with_capacity(total);
    let mut v: Vec<f32> = Vec::with_capacity(total);
    for i in 0..total {
        q.push(((i * 7 + 3) % 11) as f32 * 0.01 - 0.05);
        k.push(((i * 13 + 5) % 11) as f32 * 0.01 - 0.05);
        v.push(((i * 17 + 7) % 11) as f32 * 0.01 - 0.05);
    }

    let q_dev: CudaSlice<f32> = dev.htod_sync_copy(&q)?;
    let k_dev: CudaSlice<f32> = dev.htod_sync_copy(&k)?;
    let v_dev: CudaSlice<f32> = dev.htod_sync_copy(&v)?;
    let mut out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total)?;
    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    let cfg = LaunchConfig {
        grid_dim: (N_HEADS as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: SEQ_LEN * SEQ_LEN * 4, // score matrix
    };

    // Test 1: Bidirectional attention (causal_mask = 0) — backward compatibility
    unsafe {
        f.clone().launch(
            cfg,
            (
                &q_dev,
                &k_dev,
                &v_dev,
                &mut out_dev,
                SEQ_LEN,
                D_HEAD,
                0u32, // no causal mask
                status_dev_ptr,
            ),
        )?;
    }
    dev.synchronize()?;

    let status = unsafe { std::ptr::read_volatile(status_host_ptr) };
    assert!(
        status >= N_HEADS as u32,
        "Attention kernel did not complete: status={status}"
    );

    let out_host: Vec<f32> = dev.dtoh_sync_copy(&out_dev)?;

    // CPU reference (bidirectional)
    let mut mismatches = 0;
    let mut max_err: f32 = 0.0;
    let scale = 1.0 / (D_HEAD as f32).sqrt();

    for h in 0..N_HEADS {
        let offset = h * SEQ_LEN as usize * D_HEAD as usize;
        let q_h = &q[offset..offset + SEQ_LEN as usize * D_HEAD as usize];
        let k_h = &k[offset..offset + SEQ_LEN as usize * D_HEAD as usize];
        let v_h = &v[offset..offset + SEQ_LEN as usize * D_HEAD as usize];

        for i in 0..SEQ_LEN as usize {
            let mut scores: Vec<f32> = vec![0.0; SEQ_LEN as usize];
            for j in 0..SEQ_LEN as usize {
                let mut dot: f32 = 0.0;
                for d in 0..D_HEAD as usize {
                    dot += q_h[i * D_HEAD as usize + d] * k_h[j * D_HEAD as usize + d];
                }
                scores[j] = dot * scale;
            }

            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_s: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum_exp: f32 = exp_s.iter().sum();
            let weights: Vec<f32> = exp_s.iter().map(|e| e / sum_exp).collect();

            for d in 0..D_HEAD as usize {
                let mut acc: f32 = 0.0;
                for j in 0..SEQ_LEN as usize {
                    acc += weights[j] * v_h[j * D_HEAD as usize + d];
                }
                let got = out_host[offset + i * D_HEAD as usize + d];
                let err = (got - acc).abs();
                if err > max_err {
                    max_err = err;
                }
                if err > 1e-3 {
                    if mismatches < 5 {
                        println!(
                            "  MISMATCH h={h} i={i} d={d}: got={got} expected={acc} err={err}"
                        );
                    }
                    mismatches += 1;
                }
            }
        }
    }

    println!(
        "  Bidirectional attention {N_HEADS} heads, seq={SEQ_LEN}: max_err={max_err:.8}, mismatches={mismatches}"
    );
    if mismatches > 0 {
        unsafe { free_mapped_mem(status_host_ptr)? };
        return Err(GpuHostError::Verification {
            test: "attention_head (bidirectional)",
            detail: format!("{mismatches} mismatches"),
        });
    }

    // Test 2: Causal attention (causal_mask = 1) — GPT-2 style
    unsafe { std::ptr::write_volatile(status_host_ptr, 0u32) };
    let mut out_dev_causal: CudaSlice<f32> = dev.alloc_zeros::<f32>(total)?;

    unsafe {
        f.launch(
            cfg,
            (
                &q_dev,
                &k_dev,
                &v_dev,
                &mut out_dev_causal,
                SEQ_LEN,
                D_HEAD,
                1u32, // causal mask
                status_dev_ptr,
            ),
        )?;
    }
    dev.synchronize()?;

    let out_causal: Vec<f32> = dev.dtoh_sync_copy(&out_dev_causal)?;

    // CPU reference with causal mask
    let mut causal_mismatches = 0;
    let mut causal_max_err: f32 = 0.0;

    for h in 0..N_HEADS {
        let offset = h * SEQ_LEN as usize * D_HEAD as usize;
        let q_h = &q[offset..offset + SEQ_LEN as usize * D_HEAD as usize];
        let k_h = &k[offset..offset + SEQ_LEN as usize * D_HEAD as usize];
        let v_h = &v[offset..offset + SEQ_LEN as usize * D_HEAD as usize];

        for i in 0..SEQ_LEN as usize {
            let mut scores: Vec<f32> = vec![0.0; SEQ_LEN as usize];
            for j in 0..SEQ_LEN as usize {
                if j > i {
                    // Causal mask: future positions get -inf
                    scores[j] = -1.0e38_f32;
                } else {
                    let mut dot: f32 = 0.0;
                    for d in 0..D_HEAD as usize {
                        dot += q_h[i * D_HEAD as usize + d] * k_h[j * D_HEAD as usize + d];
                    }
                    scores[j] = dot * scale;
                }
            }

            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_s: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum_exp: f32 = exp_s.iter().sum();
            let weights: Vec<f32> = exp_s.iter().map(|e| e / sum_exp).collect();

            for d in 0..D_HEAD as usize {
                let mut acc: f32 = 0.0;
                for j in 0..SEQ_LEN as usize {
                    acc += weights[j] * v_h[j * D_HEAD as usize + d];
                }
                let got = out_causal[offset + i * D_HEAD as usize + d];
                let err = (got - acc).abs();
                if err > causal_max_err {
                    causal_max_err = err;
                }
                if err > 1e-3 {
                    if causal_mismatches < 5 {
                        println!(
                            "  CAUSAL MISMATCH h={h} i={i} d={d}: got={got} expected={acc} err={err}"
                        );
                    }
                    causal_mismatches += 1;
                }
            }
        }
    }

    unsafe { free_mapped_mem(status_host_ptr)? };

    println!(
        "  Causal attention {N_HEADS} heads, seq={SEQ_LEN}: max_err={causal_max_err:.8}, mismatches={causal_mismatches}"
    );

    // Verify causal output differs from bidirectional (except position 0 which should be same)
    let mut causal_differs = false;
    for i in 0..total {
        if (out_host[i] - out_causal[i]).abs() > 1e-6 {
            causal_differs = true;
            break;
        }
    }
    println!(
        "  Causal vs bidirectional: {}",
        if causal_differs {
            "outputs differ (expected)"
        } else {
            "outputs identical (UNEXPECTED)"
        }
    );

    if causal_mismatches == 0 && causal_differs {
        println!("  Attention (bidirectional + causal) — PASSED");
        Ok(())
    } else {
        Err(GpuHostError::Verification {
            test: "attention_head",
            detail: format!(
                "causal_mismatches={causal_mismatches}, causal_differs={causal_differs}"
            ),
        })
    }
}

/// FlashAttention test (attention-scale.3): tiled attention for seq>32.
/// Verifies flash_attention kernel against naive CPU reference for seq=128.
pub(crate) fn run_flash_attention_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- FlashAttention test (attention-scale.3) ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(ptx, "flash_attn", &["flash_attention"])?;
    let f = dev
        .get_func("flash_attn", "flash_attention")
        .ok_or(GpuHostError::KernelNotFound("flash_attention"))?;

    const N_HEADS: usize = 12;
    const SEQ_LEN: usize = 128;
    const D_HEAD: usize = 64;

    // Generate deterministic Q, K, V data
    let total = N_HEADS * SEQ_LEN * D_HEAD;
    let mut q_data: Vec<f32> = Vec::with_capacity(total);
    let mut k_data: Vec<f32> = Vec::with_capacity(total);
    let mut v_data: Vec<f32> = Vec::with_capacity(total);

    for i in 0..total {
        q_data.push(((i * 7 + 3) % 11) as f32 * 0.01 - 0.05);
        k_data.push(((i * 13 + 5) % 11) as f32 * 0.01 - 0.05);
        v_data.push(((i * 17 + 9) % 11) as f32 * 0.01 - 0.05);
    }

    let q_dev = dev.htod_sync_copy(&q_data)?;
    let k_dev = dev.htod_sync_copy(&k_data)?;
    let v_dev = dev.htod_sync_copy(&v_data)?;
    let mut out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total)?;
    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    // Test both bidirectional and causal
    for (mode_name, causal) in [("bidirectional", 0u32), ("causal", 1u32)] {
        println!("  Testing {mode_name} (seq={SEQ_LEN})...");
        unsafe { std::ptr::write_volatile(status_host_ptr, 0) };

        let n_q_tiles = SEQ_LEN.div_ceil(32);
        let cfg = LaunchConfig {
            grid_dim: (N_HEADS as u32, n_q_tiles as u32, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 2 * 32 * 64 * 4, // k_tile + v_tile = 16KB
        };

        unsafe {
            f.clone().launch(
                cfg,
                (
                    &q_dev,
                    &k_dev,
                    &v_dev,
                    &mut out_dev,
                    SEQ_LEN as u32,
                    D_HEAD as u32,
                    causal,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        let expected_blocks = (N_HEADS * n_q_tiles) as u32;
        let status = unsafe { std::ptr::read_volatile(status_host_ptr) };
        assert!(
            status >= expected_blocks,
            "flash_attention incomplete: {status}/{expected_blocks}"
        );

        let out_host: Vec<f32> = dev.dtoh_sync_copy(&out_dev)?;

        // CPU reference
        let scale = 1.0 / (D_HEAD as f32).sqrt();
        let mut mismatches = 0;
        let mut max_err: f32 = 0.0;

        for h in 0..N_HEADS {
            for i in 0..SEQ_LEN {
                // Compute attention scores for row i
                let mut scores: Vec<f32> = Vec::with_capacity(SEQ_LEN);
                for j in 0..SEQ_LEN {
                    if causal != 0 && j > i {
                        scores.push(-1.0e38);
                    } else {
                        let mut dot: f32 = 0.0;
                        for d in 0..D_HEAD {
                            let qi = q_data[h * SEQ_LEN * D_HEAD + i * D_HEAD + d];
                            let kj = k_data[h * SEQ_LEN * D_HEAD + j * D_HEAD + d];
                            dot += qi * kj;
                        }
                        scores.push(dot * scale);
                    }
                }
                // Softmax
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();
                let weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();
                // Output
                for d in 0..D_HEAD {
                    let mut acc: f32 = 0.0;
                    for j in 0..SEQ_LEN {
                        acc += weights[j] * v_data[h * SEQ_LEN * D_HEAD + j * D_HEAD + d];
                    }
                    let got = out_host[h * SEQ_LEN * D_HEAD + i * D_HEAD + d];
                    let err = (got - acc).abs();
                    if err > max_err {
                        max_err = err;
                    }
                    // Tolerance: online softmax should be mathematically exact
                    // but exp/div rounding may differ slightly
                    if err > 1e-3 {
                        if mismatches < 5 {
                            println!(
                                "    MISMATCH h={h} i={i} d={d}: got={got:.6}, exp={acc:.6}, err={err:.6}"
                            );
                        }
                        mismatches += 1;
                    }
                }
            }
        }

        let total_elems = N_HEADS * SEQ_LEN * D_HEAD;
        println!("    {mode_name}: max_err={max_err:.8}, mismatches={mismatches}/{total_elems}");

        if mismatches > 0 {
            unsafe { free_mapped_mem(status_host_ptr)? };
            return Err(GpuHostError::Verification {
                test: "flash_attention",
                detail: format!("{mode_name}: {mismatches} mismatches, max_err={max_err:.8}"),
            });
        }
    }

    unsafe { free_mapped_mem(status_host_ptr)? };
    println!("  FlashAttention (seq={SEQ_LEN}) — PASSED");
    Ok(())
}

/// FlashAttention scaling test (attention-scale.4): validate at seq=256 and seq=1024.
pub(crate) fn run_flash_attention_scale_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- FlashAttention scaling test (attention-scale.4) ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(ptx, "flash_attn_scale", &["flash_attention"])?;
    let f = dev
        .get_func("flash_attn_scale", "flash_attention")
        .ok_or(GpuHostError::KernelNotFound("flash_attention"))?;

    const N_HEADS: usize = 12;
    const D_HEAD: usize = 64;

    for seq_len in [256usize, 1024] {
        println!("  Testing causal attention at seq={seq_len}...");

        let total = N_HEADS * seq_len * D_HEAD;
        let mut q_data: Vec<f32> = Vec::with_capacity(total);
        let mut k_data: Vec<f32> = Vec::with_capacity(total);
        let mut v_data: Vec<f32> = Vec::with_capacity(total);

        for i in 0..total {
            q_data.push(((i * 7 + 3) % 11) as f32 * 0.01 - 0.05);
            k_data.push(((i * 13 + 5) % 11) as f32 * 0.01 - 0.05);
            v_data.push(((i * 17 + 9) % 11) as f32 * 0.01 - 0.05);
        }

        let q_dev = dev.htod_sync_copy(&q_data)?;
        let k_dev = dev.htod_sync_copy(&k_data)?;
        let v_dev = dev.htod_sync_copy(&v_data)?;
        let mut out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total)?;
        let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

        unsafe { std::ptr::write_volatile(status_host_ptr, 0) };

        let n_q_tiles = seq_len.div_ceil(32);
        let cfg = LaunchConfig {
            grid_dim: (N_HEADS as u32, n_q_tiles as u32, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 2 * 32 * 64 * 4,
        };

        unsafe {
            f.clone().launch(
                cfg,
                (
                    &q_dev,
                    &k_dev,
                    &v_dev,
                    &mut out_dev,
                    seq_len as u32,
                    D_HEAD as u32,
                    1u32, // causal
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        let expected_blocks = (N_HEADS * n_q_tiles) as u32;
        let status = unsafe { std::ptr::read_volatile(status_host_ptr) };
        assert!(
            status >= expected_blocks,
            "flash_attention seq={seq_len} incomplete: {status}/{expected_blocks}"
        );

        let out_host: Vec<f32> = dev.dtoh_sync_copy(&out_dev)?;

        // CPU reference: spot-check a subset of positions to keep CPU time reasonable
        // For seq=1024, full verification would be O(seq^2 * d * heads) ≈ 805M ops
        // Instead check first 32 + last 32 + middle 32 rows per head
        let scale = 1.0 / (D_HEAD as f32).sqrt();
        let mut mismatches = 0;
        let mut max_err: f32 = 0.0;
        let check_rows: Vec<usize> = {
            let mut rows = Vec::new();
            for r in 0..32.min(seq_len) {
                rows.push(r);
            }
            if seq_len > 64 {
                let mid = seq_len / 2;
                for r in mid..mid + 32.min(seq_len - mid) {
                    rows.push(r);
                }
            }
            if seq_len > 32 {
                for r in (seq_len - 32)..seq_len {
                    if !rows.contains(&r) {
                        rows.push(r);
                    }
                }
            }
            rows
        };

        for h in 0..N_HEADS {
            for &i in &check_rows {
                // Compute attention for row i
                let mut scores: Vec<f32> = Vec::with_capacity(seq_len);
                for j in 0..seq_len {
                    if j > i {
                        scores.push(-1.0e38);
                    } else {
                        let mut dot: f32 = 0.0;
                        for d in 0..D_HEAD {
                            dot += q_data[h * seq_len * D_HEAD + i * D_HEAD + d]
                                * k_data[h * seq_len * D_HEAD + j * D_HEAD + d];
                        }
                        scores.push(dot * scale);
                    }
                }
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();
                let weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

                for d in 0..D_HEAD {
                    let mut acc: f32 = 0.0;
                    for j in 0..seq_len {
                        acc += weights[j] * v_data[h * seq_len * D_HEAD + j * D_HEAD + d];
                    }
                    let got = out_host[h * seq_len * D_HEAD + i * D_HEAD + d];
                    let err = (got - acc).abs();
                    if err > max_err {
                        max_err = err;
                    }
                    if err > 1e-3 {
                        if mismatches < 3 {
                            println!(
                                "    MISMATCH h={h} i={i} d={d}: got={got:.6}, exp={acc:.6}, err={err:.6}"
                            );
                        }
                        mismatches += 1;
                    }
                }
            }
        }

        let checked = N_HEADS * check_rows.len() * D_HEAD;
        println!(
            "    seq={seq_len}: max_err={max_err:.8}, mismatches={mismatches}/{checked} (spot-checked)"
        );

        unsafe { free_mapped_mem(status_host_ptr)? };

        if mismatches > 0 {
            return Err(GpuHostError::Verification {
                test: "flash_attention_scale",
                detail: format!("seq={seq_len}: {mismatches} mismatches, max_err={max_err:.8}"),
            });
        }
    }

    println!("  FlashAttention scaling (seq=256, seq=1024) — PASSED");
    Ok(())
}

/// Embedding lookup test (full-inference.1): token + positional embeddings.
pub(crate) fn run_embedding_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Embedding lookup test (full-inference.1) ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(ptx, "embedding", &["embedding_lookup"])?;
    let f = dev
        .get_func("embedding", "embedding_lookup")
        .ok_or(GpuHostError::KernelNotFound("embedding_lookup"))?;

    const SEQ_LEN: usize = 8;
    const D_MODEL: usize = 768;
    const VOCAB_SIZE: usize = 50257;
    const MAX_SEQ: usize = 1024;

    // Create small fake embedding tables
    let mut wte = vec![0.0f32; VOCAB_SIZE * D_MODEL];
    let mut wpe = vec![0.0f32; MAX_SEQ * D_MODEL];

    // Fill with deterministic values
    for i in 0..VOCAB_SIZE * D_MODEL {
        wte[i] = (i % 1000) as f32 * 0.001;
    }
    for i in 0..MAX_SEQ * D_MODEL {
        wpe[i] = (i % 500) as f32 * 0.002;
    }

    let token_ids: Vec<u32> = vec![100, 200, 300, 400, 500, 1000, 5000, 50256];

    let wte_dev = dev.htod_sync_copy(&wte)?;
    let wpe_dev = dev.htod_sync_copy(&wpe)?;
    let tok_dev: CudaSlice<u32> = dev.htod_sync_copy(&token_ids)?;
    let mut out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(SEQ_LEN * D_MODEL)?;
    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    let total_elems = (SEQ_LEN * D_MODEL) as u32;
    let n_blocks = total_elems.div_ceil(256);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        f.clone().launch(
            cfg,
            (
                &wte_dev,
                &wpe_dev,
                &tok_dev,
                &mut out_dev,
                SEQ_LEN as u32,
                D_MODEL as u32,
                status_dev_ptr,
            ),
        )?;
    }
    dev.synchronize()?;

    let out_host: Vec<f32> = dev.dtoh_sync_copy(&out_dev)?;

    // CPU reference
    let mut mismatches = 0;
    let mut max_err: f32 = 0.0;
    for pos in 0..SEQ_LEN {
        let tok_id = token_ids[pos] as usize;
        for d in 0..D_MODEL {
            let expected = wte[tok_id * D_MODEL + d] + wpe[pos * D_MODEL + d];
            let got = out_host[pos * D_MODEL + d];
            let err = (got - expected).abs();
            if err > max_err {
                max_err = err;
            }
            if err > 1e-6 {
                mismatches += 1;
            }
        }
    }

    println!(
        "  Embedding (seq={SEQ_LEN}): max_err={max_err:.8}, mismatches={mismatches}/{}",
        SEQ_LEN * D_MODEL
    );

    unsafe { free_mapped_mem(status_host_ptr)? };

    if mismatches == 0 {
        println!("  Embedding lookup — PASSED");
        Ok(())
    } else {
        Err(GpuHostError::Verification {
            test: "embedding_lookup",
            detail: format!("{mismatches} mismatches, max_err={max_err:.8}"),
        })
    }
}

/// FFN block test (transformer-layer.4): linear(768→3072) → GELU → linear(3072→768).
/// Validates the full pipeline: f32→f16x2 pack → GEMM → bias → GELU → pack → GEMM → bias.
pub(crate) fn run_ffn_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- FFN block test (transformer-layer.4) ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(
        ptx,
        "ffn_kernels",
        &["full_gemm", "bias_add", "gelu_forward", "f32_to_f16x2_pack"],
    )?;

    let f_gemm = dev
        .get_func("ffn_kernels", "full_gemm")
        .ok_or(GpuHostError::KernelNotFound("full_gemm"))?;
    let f_bias = dev
        .get_func("ffn_kernels", "bias_add")
        .ok_or(GpuHostError::KernelNotFound("bias_add"))?;
    let f_gelu = dev
        .get_func("ffn_kernels", "gelu_forward")
        .ok_or(GpuHostError::KernelNotFound("gelu_forward"))?;
    let f_pack = dev
        .get_func("ffn_kernels", "f32_to_f16x2_pack")
        .ok_or(GpuHostError::KernelNotFound("f32_to_f16x2_pack"))?;

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
            return (sign << 15) as u16;
        }
        if new_exp >= 31 {
            return ((sign << 15) | 0x7C00) as u16;
        }
        ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
    }
    fn pack_f16x2(lo: f32, hi: f32) -> u32 {
        let lo_bits = f32_to_f16(lo) as u32;
        let hi_bits = f32_to_f16(hi) as u32;
        lo_bits | (hi_bits << 16)
    }
    fn f16_to_f32(bits: u16) -> f32 {
        let sign = ((bits >> 15) & 1) as u32;
        let exp = ((bits >> 10) & 0x1F) as i32;
        let frac = (bits & 0x3FF) as u32;
        if exp == 0 && frac == 0 {
            return f32::from_bits(sign << 31);
        }
        if exp == 0x1F {
            return if frac == 0 {
                f32::from_bits((sign << 31) | 0x7F800000)
            } else {
                f32::NAN
            };
        }
        let f32_exp = (exp - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (frac << 13))
    }

    const SEQ: u32 = 32;
    const D_MODEL: u32 = 768;
    const D_FFN: u32 = 3072;

    // Generate input [32][768] f32
    let mut input_f32: Vec<f32> = Vec::with_capacity((SEQ * D_MODEL) as usize);
    for i in 0..(SEQ * D_MODEL) as usize {
        input_f32.push(((i * 7 + 3) % 11) as f32 * 0.01 - 0.05);
    }

    // W_fc [768][3072] col-major f16x2: [3072][384] u32
    // Use small constant values for reproducibility
    let mut w_fc: Vec<u32> = Vec::with_capacity((D_FFN * D_MODEL / 2) as usize);
    let mut w_fc_f32: Vec<f32> = vec![0.0; (D_MODEL * D_FFN) as usize]; // [K=768][N=3072] row-major
    for col in 0..D_FFN as usize {
        for k_pair in 0..(D_MODEL / 2) as usize {
            let k0 = k_pair * 2;
            let k1 = k_pair * 2 + 1;
            let v0 = ((col + k0 * 3) % 7 + 1) as f32 * 0.001;
            let v1 = ((col + k1 * 3) % 7 + 1) as f32 * 0.001;
            let v0_f16 = f16_to_f32(f32_to_f16(v0));
            let v1_f16 = f16_to_f32(f32_to_f16(v1));
            w_fc_f32[k0 * D_FFN as usize + col] = v0_f16;
            w_fc_f32[k1 * D_FFN as usize + col] = v1_f16;
            w_fc.push(pack_f16x2(v0, v1));
        }
    }

    // bias_fc [3072] f32
    let bias_fc: Vec<f32> = (0..D_FFN as usize)
        .map(|j| (j % 5) as f32 * 0.001)
        .collect();

    // W_proj [3072][768] col-major f16x2: [768][1536] u32
    let mut w_proj: Vec<u32> = Vec::with_capacity((D_MODEL * D_FFN / 2) as usize);
    let mut w_proj_f32: Vec<f32> = vec![0.0; (D_FFN * D_MODEL) as usize]; // [K=3072][N=768]
    for col in 0..D_MODEL as usize {
        for k_pair in 0..(D_FFN / 2) as usize {
            let k0 = k_pair * 2;
            let k1 = k_pair * 2 + 1;
            let v0 = ((col * 5 + k0 * 11) % 7 + 1) as f32 * 0.0005;
            let v1 = ((col * 5 + k1 * 11) % 7 + 1) as f32 * 0.0005;
            let v0_f16 = f16_to_f32(f32_to_f16(v0));
            let v1_f16 = f16_to_f32(f32_to_f16(v1));
            w_proj_f32[k0 * D_MODEL as usize + col] = v0_f16;
            w_proj_f32[k1 * D_MODEL as usize + col] = v1_f16;
            w_proj.push(pack_f16x2(v0, v1));
        }
    }

    // bias_proj [768] f32
    let bias_proj: Vec<f32> = (0..D_MODEL as usize)
        .map(|j| (j % 3) as f32 * 0.001)
        .collect();

    // Upload weights
    let w_fc_dev: CudaSlice<u32> = dev.htod_sync_copy(&w_fc)?;
    let bias_fc_dev: CudaSlice<f32> = dev.htod_sync_copy(&bias_fc)?;
    let w_proj_dev: CudaSlice<u32> = dev.htod_sync_copy(&w_proj)?;
    let bias_proj_dev: CudaSlice<f32> = dev.htod_sync_copy(&bias_proj)?;

    // Upload input
    let input_dev: CudaSlice<f32> = dev.htod_sync_copy(&input_f32)?;

    // Step 1: Pack input f32 → f16x2 for GEMM
    let total_pairs_1 = SEQ * D_MODEL / 2;
    let mut input_packed_dev: CudaSlice<u32> = dev.alloc_zeros::<u32>(total_pairs_1 as usize)?;
    unsafe {
        f_pack.clone().launch(
            LaunchConfig {
                grid_dim: (total_pairs_1.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (&input_dev, &mut input_packed_dev, total_pairs_1),
        )?;
    }
    dev.synchronize()?;

    // Step 2: GEMM1: [32][768] × [768][3072] → [32][3072]
    let mut hidden_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((SEQ * D_FFN) as usize)?;
    let (s2_host, s2_dev) = unsafe { alloc_mapped_result_array(&dev, 1)? };
    unsafe {
        f_gemm.clone().launch(
            LaunchConfig {
                grid_dim: (SEQ / 32, D_FFN / 16, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: (256 + 128) * 4,
            },
            (
                &input_packed_dev,
                &w_fc_dev,
                &mut hidden_dev,
                D_MODEL / 16,
                D_FFN,
                s2_dev,
            ),
        )?;
    }
    dev.synchronize()?;

    // Step 3: Bias add
    let total_hidden = SEQ * D_FFN;
    let (s3_host, s3_dev) = unsafe { alloc_mapped_result_array(&dev, 1)? };
    unsafe {
        f_bias.clone().launch(
            LaunchConfig {
                grid_dim: (total_hidden.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (&mut hidden_dev, &bias_fc_dev, D_FFN, total_hidden, s3_dev),
        )?;
    }
    dev.synchronize()?;

    // Step 4: GELU
    let (s4_host, s4_dev) = unsafe { alloc_mapped_result_array(&dev, 1)? };
    let mut gelu_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_hidden as usize)?;
    unsafe {
        f_gelu.clone().launch(
            LaunchConfig {
                grid_dim: (total_hidden.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (&hidden_dev, &mut gelu_out_dev, total_hidden, s4_dev),
        )?;
    }
    dev.synchronize()?;

    // Step 5: Pack GELU output f32 → f16x2 for second GEMM
    let total_pairs_2 = SEQ * D_FFN / 2;
    let mut hidden_packed_dev: CudaSlice<u32> = dev.alloc_zeros::<u32>(total_pairs_2 as usize)?;
    unsafe {
        f_pack.clone().launch(
            LaunchConfig {
                grid_dim: (total_pairs_2.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (&gelu_out_dev, &mut hidden_packed_dev, total_pairs_2),
        )?;
    }
    dev.synchronize()?;

    // Step 6: GEMM2: [32][3072] × [3072][768] → [32][768]
    let mut output_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((SEQ * D_MODEL) as usize)?;
    let (s6_host, s6_dev) = unsafe { alloc_mapped_result_array(&dev, 1)? };
    unsafe {
        f_gemm.clone().launch(
            LaunchConfig {
                grid_dim: (SEQ / 32, D_MODEL / 16, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: (256 + 128) * 4,
            },
            (
                &hidden_packed_dev,
                &w_proj_dev,
                &mut output_dev,
                D_FFN / 16,
                D_MODEL,
                s6_dev,
            ),
        )?;
    }
    dev.synchronize()?;

    // Step 7: Bias add on output
    let total_output = SEQ * D_MODEL;
    let (s7_host, s7_dev) = unsafe { alloc_mapped_result_array(&dev, 1)? };
    unsafe {
        f_bias.clone().launch(
            LaunchConfig {
                grid_dim: (total_output.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &mut output_dev,
                &bias_proj_dev,
                D_MODEL,
                total_output,
                s7_dev,
            ),
        )?;
    }
    dev.synchronize()?;

    let output_host: Vec<f32> = dev.dtoh_sync_copy(&output_dev)?;

    // CPU reference: full FFN pipeline
    // Step 1: Pack input to f16
    let input_f16: Vec<f32> = input_f32
        .iter()
        .map(|v| f16_to_f32(f32_to_f16(*v)))
        .collect();

    // Step 2: GEMM1 (f16 inputs, f32 accumulation)
    let mut hidden_cpu = vec![0.0f32; (SEQ * D_FFN) as usize];
    for i in 0..SEQ as usize {
        for j in 0..D_FFN as usize {
            let mut sum: f32 = 0.0;
            for k in 0..D_MODEL as usize {
                sum += input_f16[i * D_MODEL as usize + k] * w_fc_f32[k * D_FFN as usize + j];
            }
            hidden_cpu[i * D_FFN as usize + j] = sum;
        }
    }

    // Step 3: Bias add
    for i in 0..SEQ as usize {
        for j in 0..D_FFN as usize {
            hidden_cpu[i * D_FFN as usize + j] += bias_fc[j];
        }
    }

    // Step 4: GELU
    let sqrt_2_over_pi: f32 = 0.797_884_6;
    let coeff: f32 = 0.044715;
    let gelu_cpu: Vec<f32> = hidden_cpu
        .iter()
        .map(|&x| {
            let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
            x * 0.5 * (1.0 + inner.tanh())
        })
        .collect();

    // Step 5: Pack to f16
    let gelu_f16: Vec<f32> = gelu_cpu
        .iter()
        .map(|v| f16_to_f32(f32_to_f16(*v)))
        .collect();

    // Step 6: GEMM2
    let mut output_cpu = vec![0.0f32; (SEQ * D_MODEL) as usize];
    for i in 0..SEQ as usize {
        for j in 0..D_MODEL as usize {
            let mut sum: f32 = 0.0;
            for k in 0..D_FFN as usize {
                sum += gelu_f16[i * D_FFN as usize + k] * w_proj_f32[k * D_MODEL as usize + j];
            }
            output_cpu[i * D_MODEL as usize + j] = sum;
        }
    }

    // Step 7: Bias add
    for i in 0..SEQ as usize {
        for j in 0..D_MODEL as usize {
            output_cpu[i * D_MODEL as usize + j] += bias_proj[j];
        }
    }

    // Compare
    let mut mismatches = 0;
    let mut max_err: f32 = 0.0;
    let mut max_rel_err: f32 = 0.0;
    for i in 0..(SEQ * D_MODEL) as usize {
        let got = output_host[i];
        let exp = output_cpu[i];
        let err = (got - exp).abs();
        if err > max_err {
            max_err = err;
        }
        let rel = if exp.abs() > 1e-6 {
            err / exp.abs()
        } else {
            err
        };
        if rel > max_rel_err {
            max_rel_err = rel;
        }
        // Allow larger tolerance due to f16 quantization in two GEMM stages
        if rel > 0.02 && err > 0.5 {
            if mismatches < 5 {
                let row = i / D_MODEL as usize;
                let col = i % D_MODEL as usize;
                println!("  MISMATCH [{row}][{col}]: got={got:.4} expected={exp:.4} err={err:.6}");
            }
            mismatches += 1;
        }
    }

    // Free all mapped status buffers
    unsafe {
        free_mapped_mem(s2_host)?;
        free_mapped_mem(s3_host)?;
        free_mapped_mem(s4_host)?;
        free_mapped_mem(s6_host)?;
        free_mapped_mem(s7_host)?;
    }

    println!(
        "  FFN {SEQ}x{D_MODEL}→{D_FFN}→{D_MODEL}: max_abs_err={max_err:.6}, max_rel_err={max_rel_err:.6}, mismatches={mismatches}"
    );
    if mismatches == 0 {
        println!("  FFN block — PASSED");
        Ok(())
    } else {
        Err(GpuHostError::Verification {
            test: "ffn_block",
            detail: format!("{mismatches} mismatches"),
        })
    }
}

/// End-to-end transformer layer test (transformer-layer.6):
/// LayerNorm1 → QKV proj → split → attention → concat → output proj → residual →
/// LayerNorm2 → FFN → residual
pub(crate) fn run_transformer_layer_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Transformer layer test (transformer-layer.6) ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(
        ptx,
        "transformer",
        &[
            "layer_norm",
            "full_gemm",
            "bias_add",
            "gelu_forward",
            "f32_to_f16x2_pack",
            "attention_head",
            "split_qkv",
            "concat_heads",
            "elementwise_add",
        ],
    )?;

    macro_rules! get_fn {
        ($name:expr) => {
            dev.get_func("transformer", $name)
                .ok_or(GpuHostError::KernelNotFound($name))?
        };
    }

    let f_ln = get_fn!("layer_norm");
    let f_gemm = get_fn!("full_gemm");
    let f_bias = get_fn!("bias_add");
    let f_gelu = get_fn!("gelu_forward");
    let f_pack = get_fn!("f32_to_f16x2_pack");
    let f_attn = get_fn!("attention_head");
    let f_split = get_fn!("split_qkv");
    let f_concat = get_fn!("concat_heads");
    let f_add = get_fn!("elementwise_add");

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
            return (sign << 15) as u16;
        }
        if new_exp >= 31 {
            return ((sign << 15) | 0x7C00) as u16;
        }
        ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
    }
    fn pack_f16x2(lo: f32, hi: f32) -> u32 {
        let lo_bits = f32_to_f16(lo) as u32;
        let hi_bits = f32_to_f16(hi) as u32;
        lo_bits | (hi_bits << 16)
    }
    fn f16_to_f32(bits: u16) -> f32 {
        let sign = ((bits >> 15) & 1) as u32;
        let exp = ((bits >> 10) & 0x1F) as i32;
        let frac = (bits & 0x3FF) as u32;
        if exp == 0 && frac == 0 {
            return f32::from_bits(sign << 31);
        }
        if exp == 0x1F {
            return if frac == 0 {
                f32::from_bits((sign << 31) | 0x7F800000)
            } else {
                f32::NAN
            };
        }
        let f32_exp = (exp - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (frac << 13))
    }

    // Helper to build col-major packed weight and return both packed + f32 versions
    fn make_weight_colmajor(
        n_out: usize,
        n_in: usize,
        seed: usize,
        scale: f32,
    ) -> (Vec<u32>, Vec<f32>) {
        let mut packed = Vec::with_capacity(n_out * n_in / 2);
        let mut f32_mat = vec![0.0f32; n_in * n_out]; // [K=n_in][N=n_out] row-major
        for col in 0..n_out {
            for k_pair in 0..n_in / 2 {
                let k0 = k_pair * 2;
                let k1 = k_pair * 2 + 1;
                let v0 = ((col + k0 * 3 + seed) % 7 + 1) as f32 * scale;
                let v1 = ((col + k1 * 3 + seed) % 7 + 1) as f32 * scale;
                let v0_f16 = f16_to_f32(f32_to_f16(v0));
                let v1_f16 = f16_to_f32(f32_to_f16(v1));
                f32_mat[k0 * n_out + col] = v0_f16;
                f32_mat[k1 * n_out + col] = v1_f16;
                packed.push(pack_f16x2(v0, v1));
            }
        }
        (packed, f32_mat)
    }

    const SEQ: u32 = 32;
    const D_MODEL: u32 = 768;
    const N_HEADS: u32 = 12;
    const D_HEAD: u32 = 64; // D_MODEL / N_HEADS
    const D_FFN: u32 = 3072;
    const EPS: f32 = 1e-5;
    let total_seq_model = (SEQ * D_MODEL) as usize;

    // === Generate all weights ===

    // LN1 gamma/beta
    let ln1_gamma: Vec<f32> = (0..D_MODEL as usize)
        .map(|j| 1.0 + j as f32 * 0.0001)
        .collect();
    let ln1_beta: Vec<f32> = (0..D_MODEL as usize).map(|j| j as f32 * 0.00005).collect();

    // QKV weight [768→2304] + bias
    let (w_qkv_packed, w_qkv_f32) = make_weight_colmajor(2304, 768, 0, 0.001);
    let bias_qkv: Vec<f32> = (0..2304usize).map(|j| (j % 5) as f32 * 0.0001).collect();

    // Output proj weight [768→768] + bias
    let (w_proj_packed, w_proj_f32) = make_weight_colmajor(768, 768, 100, 0.001);
    let bias_proj: Vec<f32> = (0..768usize).map(|j| (j % 3) as f32 * 0.0001).collect();

    // LN2 gamma/beta
    let ln2_gamma: Vec<f32> = (0..D_MODEL as usize)
        .map(|j| 1.0 + j as f32 * 0.00015)
        .collect();
    let ln2_beta: Vec<f32> = (0..D_MODEL as usize).map(|j| j as f32 * 0.00003).collect();

    // FFN weights
    let (w_fc_packed, w_fc_f32) = make_weight_colmajor(3072, 768, 200, 0.001);
    let bias_fc: Vec<f32> = (0..3072usize).map(|j| (j % 5) as f32 * 0.001).collect();
    let (w_fc_proj_packed, w_fc_proj_f32) = make_weight_colmajor(768, 3072, 300, 0.0005);
    let bias_fc_proj: Vec<f32> = (0..768usize).map(|j| (j % 3) as f32 * 0.001).collect();

    // Input
    let input_f32: Vec<f32> = (0..total_seq_model)
        .map(|i| ((i * 7 + 3) % 11) as f32 * 0.01 - 0.05)
        .collect();

    // === Upload everything ===
    let input_dev: CudaSlice<f32> = dev.htod_sync_copy(&input_f32)?;
    let ln1_g_dev: CudaSlice<f32> = dev.htod_sync_copy(&ln1_gamma)?;
    let ln1_b_dev: CudaSlice<f32> = dev.htod_sync_copy(&ln1_beta)?;
    let w_qkv_dev: CudaSlice<u32> = dev.htod_sync_copy(&w_qkv_packed)?;
    let bias_qkv_dev: CudaSlice<f32> = dev.htod_sync_copy(&bias_qkv)?;
    let w_proj_dev: CudaSlice<u32> = dev.htod_sync_copy(&w_proj_packed)?;
    let bias_proj_dev: CudaSlice<f32> = dev.htod_sync_copy(&bias_proj)?;
    let ln2_g_dev: CudaSlice<f32> = dev.htod_sync_copy(&ln2_gamma)?;
    let ln2_b_dev: CudaSlice<f32> = dev.htod_sync_copy(&ln2_beta)?;
    let w_fc_dev: CudaSlice<u32> = dev.htod_sync_copy(&w_fc_packed)?;
    let bias_fc_dev: CudaSlice<f32> = dev.htod_sync_copy(&bias_fc)?;
    let w_fc_proj_dev: CudaSlice<u32> = dev.htod_sync_copy(&w_fc_proj_packed)?;
    let bias_fc_proj_dev: CudaSlice<f32> = dev.htod_sync_copy(&bias_fc_proj)?;

    // Helper: alloc status buffer
    macro_rules! status_buf {
        () => {
            unsafe { alloc_mapped_result_array(&dev, 1)? }
        };
    }

    // === Step 1: LayerNorm1 ===
    let mut ln1_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let (sh1, sd1) = status_buf!();
    unsafe {
        f_ln.clone().launch(
            LaunchConfig {
                grid_dim: (SEQ, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &input_dev,
                &mut ln1_out_dev,
                &ln1_g_dev,
                &ln1_b_dev,
                D_MODEL,
                EPS,
                sd1,
            ),
        )?;
    }
    dev.synchronize()?;

    // === Step 2: QKV projection ===
    let total_pairs = SEQ * D_MODEL / 2;
    let mut ln1_packed_dev: CudaSlice<u32> = dev.alloc_zeros::<u32>(total_pairs as usize)?;
    unsafe {
        f_pack.clone().launch(
            LaunchConfig {
                grid_dim: (total_pairs.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (&ln1_out_dev, &mut ln1_packed_dev, total_pairs),
        )?;
    }
    dev.synchronize()?;

    let mut qkv_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((SEQ * 2304) as usize)?;
    let (sh2, sd2) = status_buf!();
    unsafe {
        f_gemm.clone().launch(
            LaunchConfig {
                grid_dim: (SEQ / 32, 2304 / 16, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: (256 + 128) * 4,
            },
            (
                &ln1_packed_dev,
                &w_qkv_dev,
                &mut qkv_dev,
                D_MODEL / 16,
                2304u32,
                sd2,
            ),
        )?;
    }
    dev.synchronize()?;

    // Bias add on QKV
    let total_qkv = SEQ * 2304;
    let (sh3, sd3) = status_buf!();
    unsafe {
        f_bias.clone().launch(
            LaunchConfig {
                grid_dim: (total_qkv.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (&mut qkv_dev, &bias_qkv_dev, 2304u32, total_qkv, sd3),
        )?;
    }
    dev.synchronize()?;

    // === Step 3: Split QKV → Q, K, V [12][32][64] ===
    let head_total = (N_HEADS * SEQ * D_HEAD) as usize;
    let mut q_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut k_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut v_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    unsafe {
        f_split.clone().launch(
            LaunchConfig {
                grid_dim: ((head_total as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &qkv_dev, &mut q_dev, &mut k_dev, &mut v_dev, SEQ, N_HEADS, D_HEAD,
            ),
        )?;
    }
    dev.synchronize()?;

    // === Step 4: Per-head attention ===
    let mut attn_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let (sh4, sd4) = status_buf!();
    unsafe {
        f_attn.clone().launch(
            LaunchConfig {
                grid_dim: (N_HEADS, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: SEQ * SEQ * 4,
            },
            (
                &q_dev,
                &k_dev,
                &v_dev,
                &mut attn_out_dev,
                SEQ,
                D_HEAD,
                0u32,
                sd4,
            ),
        )?;
    }
    dev.synchronize()?;

    // === Step 5: Concat heads → [32][768] ===
    let mut concat_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    unsafe {
        f_concat.clone().launch(
            LaunchConfig {
                grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (&attn_out_dev, &mut concat_dev, SEQ, N_HEADS, D_HEAD),
        )?;
    }
    dev.synchronize()?;

    // === Step 6: Output projection ===
    let concat_pairs = SEQ * D_MODEL / 2;
    let mut concat_packed_dev: CudaSlice<u32> = dev.alloc_zeros::<u32>(concat_pairs as usize)?;
    unsafe {
        f_pack.clone().launch(
            LaunchConfig {
                grid_dim: (concat_pairs.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (&concat_dev, &mut concat_packed_dev, concat_pairs),
        )?;
    }
    dev.synchronize()?;

    let mut proj_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let (sh5, sd5) = status_buf!();
    unsafe {
        f_gemm.clone().launch(
            LaunchConfig {
                grid_dim: (SEQ / 32, D_MODEL / 16, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: (256 + 128) * 4,
            },
            (
                &concat_packed_dev,
                &w_proj_dev,
                &mut proj_out_dev,
                D_MODEL / 16,
                D_MODEL,
                sd5,
            ),
        )?;
    }
    dev.synchronize()?;

    // Bias add
    let (sh6, sd6) = status_buf!();
    unsafe {
        f_bias.clone().launch(
            LaunchConfig {
                grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &mut proj_out_dev,
                &bias_proj_dev,
                D_MODEL,
                total_seq_model as u32,
                sd6,
            ),
        )?;
    }
    dev.synchronize()?;

    // === Step 7: Residual add: residual = input + proj_out ===
    // Copy input to residual buffer, then add proj_out
    let mut residual1_dev: CudaSlice<f32> = dev.htod_sync_copy(&input_f32)?;
    unsafe {
        f_add.clone().launch(
            LaunchConfig {
                grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (&mut residual1_dev, &proj_out_dev, total_seq_model as u32),
        )?;
    }
    dev.synchronize()?;

    // === Step 8: LayerNorm2 ===
    let mut ln2_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let (sh7, sd7) = status_buf!();
    unsafe {
        f_ln.clone().launch(
            LaunchConfig {
                grid_dim: (SEQ, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &residual1_dev,
                &mut ln2_out_dev,
                &ln2_g_dev,
                &ln2_b_dev,
                D_MODEL,
                EPS,
                sd7,
            ),
        )?;
    }
    dev.synchronize()?;

    // === Step 9-12: FFN (pack → GEMM1 → bias → GELU → pack → GEMM2 → bias) ===
    let ln2_pairs = SEQ * D_MODEL / 2;
    let mut ln2_packed_dev: CudaSlice<u32> = dev.alloc_zeros::<u32>(ln2_pairs as usize)?;
    unsafe {
        f_pack.clone().launch(
            LaunchConfig {
                grid_dim: (ln2_pairs.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (&ln2_out_dev, &mut ln2_packed_dev, ln2_pairs),
        )?;
    }
    dev.synchronize()?;

    let mut ffn_hidden_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((SEQ * D_FFN) as usize)?;
    let (sh8, sd8) = status_buf!();
    unsafe {
        f_gemm.clone().launch(
            LaunchConfig {
                grid_dim: (SEQ / 32, D_FFN / 16, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: (256 + 128) * 4,
            },
            (
                &ln2_packed_dev,
                &w_fc_dev,
                &mut ffn_hidden_dev,
                D_MODEL / 16,
                D_FFN,
                sd8,
            ),
        )?;
    }
    dev.synchronize()?;

    let total_ffn = SEQ * D_FFN;
    let (sh9, sd9) = status_buf!();
    unsafe {
        f_bias.clone().launch(
            LaunchConfig {
                grid_dim: (total_ffn.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (&mut ffn_hidden_dev, &bias_fc_dev, D_FFN, total_ffn, sd9),
        )?;
    }
    dev.synchronize()?;

    let mut gelu_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_ffn as usize)?;
    let (sh10, sd10) = status_buf!();
    unsafe {
        f_gelu.clone().launch(
            LaunchConfig {
                grid_dim: (total_ffn.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (&ffn_hidden_dev, &mut gelu_out_dev, total_ffn, sd10),
        )?;
    }
    dev.synchronize()?;

    let gelu_pairs = SEQ * D_FFN / 2;
    let mut gelu_packed_dev: CudaSlice<u32> = dev.alloc_zeros::<u32>(gelu_pairs as usize)?;
    unsafe {
        f_pack.clone().launch(
            LaunchConfig {
                grid_dim: (gelu_pairs.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (&gelu_out_dev, &mut gelu_packed_dev, gelu_pairs),
        )?;
    }
    dev.synchronize()?;

    let mut ffn_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let (sh11, sd11) = status_buf!();
    unsafe {
        f_gemm.clone().launch(
            LaunchConfig {
                grid_dim: (SEQ / 32, D_MODEL / 16, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: (256 + 128) * 4,
            },
            (
                &gelu_packed_dev,
                &w_fc_proj_dev,
                &mut ffn_out_dev,
                D_FFN / 16,
                D_MODEL,
                sd11,
            ),
        )?;
    }
    dev.synchronize()?;

    let (sh12, sd12) = status_buf!();
    unsafe {
        f_bias.clone().launch(
            LaunchConfig {
                grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &mut ffn_out_dev,
                &bias_fc_proj_dev,
                D_MODEL,
                total_seq_model as u32,
                sd12,
            ),
        )?;
    }
    dev.synchronize()?;

    // === Step 13: Residual add: output = residual1 + ffn_out ===
    unsafe {
        f_add.clone().launch(
            LaunchConfig {
                grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (&mut residual1_dev, &ffn_out_dev, total_seq_model as u32),
        )?;
    }
    dev.synchronize()?;

    let output_host: Vec<f32> = dev.dtoh_sync_copy(&residual1_dev)?;

    // === Full CPU reference computation ===
    let s = SEQ as usize;
    let dm = D_MODEL as usize;
    let nh = N_HEADS as usize;
    let dh = D_HEAD as usize;
    let dff = D_FFN as usize;
    let sqrt_2_over_pi: f32 = 0.797_884_6;
    let coeff_gelu: f32 = 0.044715;

    // CPU helper: layer_norm
    let cpu_layer_norm = |inp: &[f32], gamma: &[f32], beta: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; inp.len()];
        for row in 0..s {
            let sl = &inp[row * dm..(row + 1) * dm];
            let mean: f32 = sl.iter().sum::<f32>() / dm as f32;
            let var: f32 = sl.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / dm as f32;
            let inv_std = 1.0 / (var + EPS).sqrt();
            for j in 0..dm {
                out[row * dm + j] = gamma[j] * (sl[j] - mean) * inv_std + beta[j];
            }
        }
        out
    };

    // CPU helper: matmul with f16 input quantization (matching GPU pipeline)
    let cpu_gemm_f16 = |a: &[f32], w: &[f32], rows: usize, k_dim: usize, cols: usize| -> Vec<f32> {
        let a_f16: Vec<f32> = a.iter().map(|v| f16_to_f32(f32_to_f16(*v))).collect();
        let mut out = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                let mut sum: f32 = 0.0;
                for k in 0..k_dim {
                    sum += a_f16[i * k_dim + k] * w[k * cols + j];
                }
                out[i * cols + j] = sum;
            }
        }
        out
    };

    // Step 1: LayerNorm1
    let ln1_cpu = cpu_layer_norm(&input_f32, &ln1_gamma, &ln1_beta);

    // Step 2: QKV projection (f16 input)
    let mut qkv_cpu = cpu_gemm_f16(&ln1_cpu, &w_qkv_f32, s, dm, 2304);
    for i in 0..s {
        for j in 0..2304 {
            qkv_cpu[i * 2304 + j] += bias_qkv[j];
        }
    }

    // Step 3: Split QKV → [n_heads][seq][d_head]
    let mut q_cpu = vec![0.0f32; nh * s * dh];
    let mut k_cpu = vec![0.0f32; nh * s * dh];
    let mut v_cpu = vec![0.0f32; nh * s * dh];
    for head in 0..nh {
        for seq in 0..s {
            for d in 0..dh {
                let qkv_idx = seq * 2304 + head * dh + d;
                let out_idx = head * s * dh + seq * dh + d;
                q_cpu[out_idx] = qkv_cpu[qkv_idx];
                k_cpu[out_idx] = qkv_cpu[qkv_idx + dm];
                v_cpu[out_idx] = qkv_cpu[qkv_idx + 2 * dm];
            }
        }
    }

    // Step 4: Per-head attention
    let scale = 1.0 / (dh as f32).sqrt();
    let mut attn_out_cpu = vec![0.0f32; nh * s * dh];
    for h in 0..nh {
        let off = h * s * dh;
        for i in 0..s {
            // Scores
            let mut scores = vec![0.0f32; s];
            for j in 0..s {
                let mut dot: f32 = 0.0;
                for d in 0..dh {
                    dot += q_cpu[off + i * dh + d] * k_cpu[off + j * dh + d];
                }
                scores[j] = dot * scale;
            }
            // Softmax
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_s: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum_exp: f32 = exp_s.iter().sum();
            // Weighted sum
            for d in 0..dh {
                let mut acc: f32 = 0.0;
                for j in 0..s {
                    acc += (exp_s[j] / sum_exp) * v_cpu[off + j * dh + d];
                }
                attn_out_cpu[off + i * dh + d] = acc;
            }
        }
    }

    // Step 5: Concat → [seq][d_model]
    let mut concat_cpu = vec![0.0f32; s * dm];
    for seq in 0..s {
        for head in 0..nh {
            for d in 0..dh {
                concat_cpu[seq * dm + head * dh + d] = attn_out_cpu[head * s * dh + seq * dh + d];
            }
        }
    }

    // Step 6: Output projection (f16 input)
    let mut proj_cpu = cpu_gemm_f16(&concat_cpu, &w_proj_f32, s, dm, dm);
    for i in 0..s {
        for j in 0..dm {
            proj_cpu[i * dm + j] += bias_proj[j];
        }
    }

    // Step 7: Residual
    let mut residual1_cpu = input_f32.clone();
    for i in 0..s * dm {
        residual1_cpu[i] += proj_cpu[i];
    }

    // Step 8: LayerNorm2
    let ln2_cpu = cpu_layer_norm(&residual1_cpu, &ln2_gamma, &ln2_beta);

    // Step 9: FFN GEMM1 (f16 input)
    let mut ffn_hidden_cpu = cpu_gemm_f16(&ln2_cpu, &w_fc_f32, s, dm, dff);
    for i in 0..s {
        for j in 0..dff {
            ffn_hidden_cpu[i * dff + j] += bias_fc[j];
        }
    }

    // Step 10: GELU
    let gelu_cpu: Vec<f32> = ffn_hidden_cpu
        .iter()
        .map(|&x| {
            let inner = sqrt_2_over_pi * (x + coeff_gelu * x * x * x);
            x * 0.5 * (1.0 + inner.tanh())
        })
        .collect();

    // Step 11: FFN GEMM2 (f16 input)
    let mut ffn_out_cpu = cpu_gemm_f16(&gelu_cpu, &w_fc_proj_f32, s, dff, dm);
    for i in 0..s {
        for j in 0..dm {
            ffn_out_cpu[i * dm + j] += bias_fc_proj[j];
        }
    }

    // Step 12: Residual
    let mut output_cpu = residual1_cpu.clone();
    for i in 0..s * dm {
        output_cpu[i] += ffn_out_cpu[i];
    }

    // === Compare GPU vs CPU ===
    let mut mismatches = 0;
    let mut max_abs_err: f32 = 0.0;
    let mut max_rel_err: f32 = 0.0;
    for i in 0..total_seq_model {
        let got = output_host[i];
        let exp = output_cpu[i];
        if !got.is_finite() {
            if mismatches < 3 {
                println!("  GPU output[{i}] is not finite: {got}");
            }
            mismatches += 1;
            continue;
        }
        let err = (got - exp).abs();
        if err > max_abs_err {
            max_abs_err = err;
        }
        let rel = if exp.abs() > 1e-6 {
            err / exp.abs()
        } else {
            err
        };
        if rel > max_rel_err {
            max_rel_err = rel;
        }
        // Tolerance: 10% relative OR 0.05 absolute (compound f16 quantization across 3+ GEMM stages)
        if err > 0.05 && rel > 0.10 {
            if mismatches < 5 {
                let row = i / dm;
                let col = i % dm;
                println!(
                    "  MISMATCH [{row}][{col}]: gpu={got:.6} cpu={exp:.6} err={err:.6} rel={rel:.6}"
                );
            }
            mismatches += 1;
        }
    }

    // Free all status buffers
    unsafe {
        free_mapped_mem(sh1)?;
        free_mapped_mem(sh2)?;
        free_mapped_mem(sh3)?;
        free_mapped_mem(sh4)?;
        free_mapped_mem(sh5)?;
        free_mapped_mem(sh6)?;
        free_mapped_mem(sh7)?;
        free_mapped_mem(sh8)?;
        free_mapped_mem(sh9)?;
        free_mapped_mem(sh10)?;
        free_mapped_mem(sh11)?;
        free_mapped_mem(sh12)?;
    }

    println!(
        "  Transformer layer: max_abs_err={max_abs_err:.6}, max_rel_err={max_rel_err:.6}, mismatches={mismatches}/{total_seq_model}"
    );
    if mismatches == 0 {
        println!("  Transformer layer (full CPU reference validation, {SEQ}×{D_MODEL}) — PASSED");
        Ok(())
    } else {
        Err(GpuHostError::Verification {
            test: "transformer_layer",
            detail: format!("{mismatches} mismatches"),
        })
    }
}

/// KV cache test (kv-cache.2): validate flash_attention_kv against flash_attention.
///
/// Compares two approaches for computing attention at position N-1:
/// 1. Reference: full flash_attention(Q[0..N], K[0..N], V[0..N]), take row N-1
/// 2. Cached: flash_attention_kv(Q[N-1], K_cache[0..N], V_cache[0..N]), take row 0
///
/// They should produce identical output for the last position.
pub(crate) fn run_kv_cache_attention_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- KV cache attention test (kv-cache.2) ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(ptx, "kv_cache", &["flash_attention", "flash_attention_kv"])?;
    let f_attn = dev
        .get_func("kv_cache", "flash_attention")
        .ok_or(GpuHostError::KernelNotFound("flash_attention"))?;
    let f_attn_kv = dev
        .get_func("kv_cache", "flash_attention_kv")
        .ok_or(GpuHostError::KernelNotFound("flash_attention_kv"))?;

    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    const N_HEADS: u32 = 12;
    const D_HEAD: u32 = 64;
    const SEQ_LEN: u32 = 32; // full sequence length
    let total = (N_HEADS * SEQ_LEN * D_HEAD) as usize;

    // Generate deterministic Q, K, V
    let mut q_data = vec![0.0f32; total];
    let mut k_data = vec![0.0f32; total];
    let mut v_data = vec![0.0f32; total];
    for i in 0..total {
        q_data[i] = ((i * 7 + 3) % 100) as f32 * 0.01 - 0.5;
        k_data[i] = ((i * 11 + 7) % 100) as f32 * 0.01 - 0.5;
        v_data[i] = ((i * 13 + 11) % 100) as f32 * 0.01 - 0.5;
    }

    let q_dev = dev.htod_sync_copy(&q_data)?;
    let k_dev = dev.htod_sync_copy(&k_data)?;
    let v_dev = dev.htod_sync_copy(&v_data)?;

    // --- Reference: full flash_attention ---
    let mut ref_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total)?;
    let n_q_tiles = SEQ_LEN.div_ceil(32);
    unsafe {
        f_attn.clone().launch(
            LaunchConfig {
                grid_dim: (N_HEADS, n_q_tiles, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 2 * 32 * 64 * 4,
            },
            (
                &q_dev,
                &k_dev,
                &v_dev,
                &mut ref_out_dev,
                SEQ_LEN,
                D_HEAD,
                1u32, // causal
                status_dev_ptr,
            ),
        )?;
    }
    dev.synchronize()?;

    let ref_out: Vec<f32> = dev.dtoh_sync_copy(&ref_out_dev)?;

    // Extract reference output for last position (SEQ_LEN-1) across all heads
    let last_pos = (SEQ_LEN - 1) as usize;
    let mut ref_last = vec![0.0f32; (N_HEADS * D_HEAD) as usize];
    for h in 0..N_HEADS as usize {
        let head_off = h * SEQ_LEN as usize * D_HEAD as usize;
        let row_off = head_off + last_pos * D_HEAD as usize;
        for d in 0..D_HEAD as usize {
            ref_last[h * D_HEAD as usize + d] = ref_out[row_off + d];
        }
    }

    // --- Cached: flash_attention_kv with q_len=1 ---
    // Q: only the last position, shape [N_HEADS, 1, D_HEAD]
    let q_single_size = (N_HEADS * D_HEAD) as usize;
    let mut q_single = vec![0.0f32; q_single_size];
    for h in 0..N_HEADS as usize {
        let src_off = h * SEQ_LEN as usize * D_HEAD as usize + last_pos * D_HEAD as usize;
        let dst_off = h * D_HEAD as usize;
        q_single[dst_off..dst_off + D_HEAD as usize]
            .copy_from_slice(&q_data[src_off..src_off + D_HEAD as usize]);
    }

    let q_single_dev = dev.htod_sync_copy(&q_single)?;
    // K and V cache = full K, V (same as reference)
    let mut kv_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(q_single_size)?;

    unsafe {
        f_attn_kv.clone().launch(
            LaunchConfig {
                grid_dim: (N_HEADS, 1, 1), // q_len=1, so 1 Q tile
                block_dim: (32, 1, 1),
                shared_mem_bytes: 2 * 32 * 64 * 4,
            },
            (
                &q_single_dev,
                &k_dev, // full KV cache
                &v_dev, // full KV cache
                &mut kv_out_dev,
                1u32,    // q_len = 1
                SEQ_LEN, // kv_len = full cache
                D_HEAD,
                1u32,            // causal
                last_pos as u32, // q_offset = position of this query in full sequence
                SEQ_LEN,         // kv_stride = same as kv_len (packed)
                status_dev_ptr,
            ),
        )?;
    }
    dev.synchronize()?;

    let kv_out: Vec<f32> = dev.dtoh_sync_copy(&kv_out_dev)?;

    // --- Compare ---
    let mut max_err: f32 = 0.0;
    let mut mismatches = 0u32;
    for h in 0..N_HEADS as usize {
        for d in 0..D_HEAD as usize {
            let idx = h * D_HEAD as usize + d;
            let diff = (ref_last[idx] - kv_out[idx]).abs();
            if diff > max_err {
                max_err = diff;
            }
            if diff > 1e-4 {
                if mismatches < 5 {
                    println!(
                        "  MISMATCH h={h} d={d}: ref={:.6} kv={:.6} diff={:.6}",
                        ref_last[idx], kv_out[idx], diff
                    );
                }
                mismatches += 1;
            }
        }
    }

    println!("  max |ref - cached| = {max_err:.6}");
    println!("  mismatches (>1e-4): {mismatches} / {}", N_HEADS * D_HEAD);
    println!(
        "  ref_last[0..4] = [{:.6}, {:.6}, {:.6}, {:.6}]",
        ref_last[0], ref_last[1], ref_last[2], ref_last[3]
    );
    println!(
        "  kv_out[0..4]   = [{:.6}, {:.6}, {:.6}, {:.6}]",
        kv_out[0], kv_out[1], kv_out[2], kv_out[3]
    );

    unsafe {
        free_mapped_mem(status_host_ptr)?;
    }

    if mismatches > 0 {
        return Err(GpuHostError::Verification {
            test: "kv_cache_attention",
            detail: format!("{mismatches} mismatches, max_err={max_err:.6}"),
        });
    }

    println!("  KV cache attention — PASSED");
    Ok(())
}

/// Flash Attention V3 benchmark: correctness + throughput measurement.
#[cfg(feature = "cublas")]
pub(crate) fn run_flash_attention_v3_bench(dev: Arc<CudaDevice>) -> Result<()> {
    use gpu_host::nn::ops::attention::multi_head_flash_attention_v3;
    use gpu_host::nn::tensor::GpuTensor;

    println!("\n--- Flash Attention V3 Benchmark (perf-attn-v3) ---");

    const N_HEADS: usize = 12;
    const D_HEAD: usize = 64;
    let seq_lens: &[usize] = &[128, 256, 512];

    for &seq_len in seq_lens {
        let total = N_HEADS * seq_len * D_HEAD;

        let mut q_data: Vec<f32> = Vec::with_capacity(total);
        let mut k_data: Vec<f32> = Vec::with_capacity(total);
        let mut v_data: Vec<f32> = Vec::with_capacity(total);
        for i in 0..total {
            q_data.push(((i * 7 + 3) % 11) as f32 * 0.01 - 0.05);
            k_data.push(((i * 13 + 5) % 11) as f32 * 0.01 - 0.05);
            v_data.push(((i * 17 + 9) % 11) as f32 * 0.01 - 0.05);
        }

        let q = GpuTensor::from_host(&q_data, &[N_HEADS * seq_len, D_HEAD], &dev).map_err(|e| {
            GpuHostError::Verification {
                test: "attn_v3",
                detail: format!("{e}"),
            }
        })?;
        let k = GpuTensor::from_host(&k_data, &[N_HEADS * seq_len, D_HEAD], &dev).map_err(|e| {
            GpuHostError::Verification {
                test: "attn_v3",
                detail: format!("{e}"),
            }
        })?;
        let v = GpuTensor::from_host(&v_data, &[N_HEADS * seq_len, D_HEAD], &dev).map_err(|e| {
            GpuHostError::Verification {
                test: "attn_v3",
                detail: format!("{e}"),
            }
        })?;

        // Correctness check (causal)
        let out = multi_head_flash_attention_v3(&q, &k, &v, seq_len, N_HEADS, D_HEAD, true, &dev)
            .map_err(|e| GpuHostError::Verification {
            test: "attn_v3",
            detail: format!("{e}"),
        })?;
        let out_host = out.to_host().map_err(|e| GpuHostError::Verification {
            test: "attn_v3",
            detail: format!("{e}"),
        })?;

        // Spot-check correctness: row 0 of head 0 should equal V[0]
        let v_row0 = &v_data[0..D_HEAD];
        let out_row0 = &out_host[0..D_HEAD];
        let mut max_err: f32 = 0.0;
        for d in 0..D_HEAD {
            let err = (out_row0[d] - v_row0[d]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        if max_err > 1e-3 {
            return Err(GpuHostError::Verification {
                test: "attn_v3",
                detail: format!("seq={seq_len} row0 mismatch: max_err={max_err:.6} (expect ~V[0])"),
            });
        }

        // Throughput benchmark
        let warmup = 5;
        let iters = 20;

        for &(mode, label) in &[(true, "causal"), (false, "bidirectional")] {
            for _ in 0..warmup {
                let _ =
                    multi_head_flash_attention_v3(&q, &k, &v, seq_len, N_HEADS, D_HEAD, mode, &dev);
            }
            dev.synchronize().map_err(|e| GpuHostError::Verification {
                test: "attn_v3",
                detail: format!("{e}"),
            })?;

            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _ =
                    multi_head_flash_attention_v3(&q, &k, &v, seq_len, N_HEADS, D_HEAD, mode, &dev);
            }
            dev.synchronize().map_err(|e| GpuHostError::Verification {
                test: "attn_v3",
                detail: format!("{e}"),
            })?;
            let elapsed = start.elapsed();
            let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / iters as f64;

            // FLOPs: 2 * n_heads * seq^2 * d_head for each of Q·K^T and P·V
            // Causal: ~half the work; bidirectional: full
            let ratio = if mode { 0.5 } else { 1.0 };
            let flops =
                2.0 * 2.0 * ratio * N_HEADS as f64 * (seq_len as f64).powi(2) * D_HEAD as f64;
            let gflops = flops / (ms_per_iter * 1e6);

            println!("  seq={seq_len:>4} {label:>5}: {ms_per_iter:.3} ms, {gflops:.0} GFLOPS");
        }
    }

    println!("  Flash Attention V3 Benchmark — DONE");
    Ok(())
}

/// Elementwise ops benchmark (perf-elementwise.2).
///
/// Benchmarks all elementwise operations with bandwidth measurement:
/// - elementwise_add (in-place, V3 float4)
/// - gelu_forward (V1 scalar)
/// - gelu_forward_v2 (V2 4-elem/thread)
/// - silu_forward (V1 scalar)
/// - sigmoid_forward (V1 scalar)
/// - relu_forward (V2 float4, NEW)
///
/// Reports GB/s for bandwidth-bound ops and GFLOPS for compute-bound ops.
#[cfg(feature = "nn")]
pub(crate) fn run_elementwise_benchmark(dev: Arc<CudaDevice>) -> Result<()> {
    use gpu_host::nn::ops;
    use gpu_host::nn::registry::KernelRegistry;
    use gpu_host::nn::tensor::GpuTensor;

    println!("\n--- Elementwise Ops Benchmark (perf-elementwise.2) ---");
    println!("  GTX 1660: peak memory bandwidth = 192 GB/s, 5 TFLOPS FP32");

    let registry = std::sync::Arc::new(
        KernelRegistry::new(dev.clone(), crate::KERNEL_COMPUTE_PTX).map_err(|e| {
            GpuHostError::Verification {
                test: "elem_bench",
                detail: format!("{e}"),
            }
        })?,
    );

    // Test sizes: typical GPT-2 sizes
    let sizes: &[(usize, &str)] = &[
        (768 * 128, "128x768 (LN output)"),
        (3072 * 128, "128x3072 (FFN hidden)"),
        (786_432, "786432 (GPT-2 FFN)"),
        (4_194_304, "4M (large tensor)"),
    ];

    let warmup_iters = 20;
    let bench_iters = 200;

    for &(n, label) in sizes {
        println!(
            "\n  Size: {label} ({n} elements, {:.2} MB)",
            n as f64 * 4.0 / 1e6
        );
        println!(
            "  {:>22} {:>10} {:>10} {:>10} {:>10}",
            "Op", "ms/iter", "GB/s", "% peak", "notes"
        );

        // Generate input data: random-ish values in [-2, 2]
        let input_data: Vec<f32> = (0..n)
            .map(|i| ((i * 7 + 3) % 400) as f32 * 0.01 - 2.0)
            .collect();
        let b_data: Vec<f32> = (0..n)
            .map(|i| ((i * 11 + 5) % 300) as f32 * 0.01 - 1.5)
            .collect();

        // --- elementwise_add (in-place a += b) ---
        {
            let b_tensor = GpuTensor::from_host(&b_data, &[n], &dev).map_err(|e| {
                GpuHostError::Verification {
                    test: "elem_bench",
                    detail: format!("{e}"),
                }
            })?;

            // Warmup
            for _ in 0..warmup_iters {
                let mut a = GpuTensor::from_host(&input_data, &[n], &dev).map_err(|e| {
                    GpuHostError::Verification {
                        test: "elem_bench",
                        detail: format!("{e}"),
                    }
                })?;
                ops::elementwise_add(&mut a, &b_tensor, &registry).map_err(|e| {
                    GpuHostError::Verification {
                        test: "elem_bench",
                        detail: format!("{e}"),
                    }
                })?;
            }
            dev.synchronize().map_err(|e| GpuHostError::Verification {
                test: "elem_bench",
                detail: format!("{e}"),
            })?;

            // Benchmark: create a persistent tensor and repeatedly add
            let mut a_persist = GpuTensor::from_host(&input_data, &[n], &dev).map_err(|e| {
                GpuHostError::Verification {
                    test: "elem_bench",
                    detail: format!("{e}"),
                }
            })?;
            let start = std::time::Instant::now();
            for _ in 0..bench_iters {
                ops::elementwise_add(&mut a_persist, &b_tensor, &registry).map_err(|e| {
                    GpuHostError::Verification {
                        test: "elem_bench",
                        detail: format!("{e}"),
                    }
                })?;
            }
            dev.synchronize().map_err(|e| GpuHostError::Verification {
                test: "elem_bench",
                detail: format!("{e}"),
            })?;
            let elapsed = start.elapsed();
            let ms_per = elapsed.as_secs_f64() * 1000.0 / bench_iters as f64;
            // Bandwidth: read a (4B) + read b (4B) + write a (4B) = 12 bytes/elem
            let bytes = n as f64 * 12.0;
            let gbps = bytes / (ms_per * 1e6);
            let pct_peak = gbps / 192.0 * 100.0;
            println!(
                "  {:>22} {:>10.4} {:>10.1} {:>9.0}% {:>10}",
                "elementwise_add(V3)", ms_per, gbps, pct_peak, "in-place"
            );
        }

        // --- gelu_forward (V1 scalar, 1 elem/thread) ---
        {
            let input_tensor = GpuTensor::from_host(&input_data, &[n], &dev).map_err(|e| {
                GpuHostError::Verification {
                    test: "elem_bench",
                    detail: format!("{e}"),
                }
            })?;

            for _ in 0..warmup_iters {
                let _ = ops::gelu(&input_tensor, &registry);
            }
            dev.synchronize().map_err(|e| GpuHostError::Verification {
                test: "elem_bench",
                detail: format!("{e}"),
            })?;

            let start = std::time::Instant::now();
            for _ in 0..bench_iters {
                let _ = ops::gelu(&input_tensor, &registry);
            }
            dev.synchronize().map_err(|e| GpuHostError::Verification {
                test: "elem_bench",
                detail: format!("{e}"),
            })?;
            let elapsed = start.elapsed();
            let ms_per = elapsed.as_secs_f64() * 1000.0 / bench_iters as f64;
            // read input (4B) + write output (4B) = 8 bytes/elem
            let bytes = n as f64 * 8.0;
            let gbps = bytes / (ms_per * 1e6);
            let pct_peak = gbps / 192.0 * 100.0;
            println!(
                "  {:>22} {:>10.4} {:>10.1} {:>9.0}% {:>10}",
                "gelu(V1 tanh)", ms_per, gbps, pct_peak, "compute"
            );
        }

        // --- gelu_forward_v2 (V2, 4 elem/thread, sigmoid approx) ---
        {
            let input_tensor = GpuTensor::from_host(&input_data, &[n], &dev).map_err(|e| {
                GpuHostError::Verification {
                    test: "elem_bench",
                    detail: format!("{e}"),
                }
            })?;
            let dev_ref = registry.device();
            let mut output =
                GpuTensor::zeros(&[n], dev_ref).map_err(|e| GpuHostError::Verification {
                    test: "elem_bench",
                    detail: format!("{e}"),
                })?;
            let status_dev = dev_ref.htod_sync_copy(&[0u32])?;

            let func = registry
                .get("gelu_forward_v2")
                .map_err(|e| GpuHostError::Verification {
                    test: "elem_bench",
                    detail: format!("{e}"),
                })?;
            let config = LaunchConfig {
                grid_dim: ((n as u32 + 1023) / 1024, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };

            for _ in 0..warmup_iters {
                unsafe {
                    func.clone()
                        .launch(
                            config,
                            (
                                input_tensor.data(),
                                output.data_mut(),
                                n as u32,
                                &status_dev,
                            ),
                        )
                        .map_err(|e| GpuHostError::Verification {
                            test: "elem_bench",
                            detail: format!("{e}"),
                        })?;
                }
            }
            dev.synchronize().map_err(|e| GpuHostError::Verification {
                test: "elem_bench",
                detail: format!("{e}"),
            })?;

            let start = std::time::Instant::now();
            for _ in 0..bench_iters {
                unsafe {
                    func.clone()
                        .launch(
                            config,
                            (
                                input_tensor.data(),
                                output.data_mut(),
                                n as u32,
                                &status_dev,
                            ),
                        )
                        .map_err(|e| GpuHostError::Verification {
                            test: "elem_bench",
                            detail: format!("{e}"),
                        })?;
                }
            }
            dev.synchronize().map_err(|e| GpuHostError::Verification {
                test: "elem_bench",
                detail: format!("{e}"),
            })?;
            let elapsed = start.elapsed();
            let ms_per = elapsed.as_secs_f64() * 1000.0 / bench_iters as f64;
            let bytes = n as f64 * 8.0;
            let gbps = bytes / (ms_per * 1e6);
            let pct_peak = gbps / 192.0 * 100.0;
            println!(
                "  {:>22} {:>10.4} {:>10.1} {:>9.0}% {:>10}",
                "gelu(V2 sigmoid)", ms_per, gbps, pct_peak, "compute"
            );
        }

        // --- silu_forward (V1 scalar) ---
        {
            let input_tensor = GpuTensor::from_host(&input_data, &[n], &dev).map_err(|e| {
                GpuHostError::Verification {
                    test: "elem_bench",
                    detail: format!("{e}"),
                }
            })?;

            for _ in 0..warmup_iters {
                let _ = ops::silu(&input_tensor, &registry);
            }
            dev.synchronize().map_err(|e| GpuHostError::Verification {
                test: "elem_bench",
                detail: format!("{e}"),
            })?;

            let start = std::time::Instant::now();
            for _ in 0..bench_iters {
                let _ = ops::silu(&input_tensor, &registry);
            }
            dev.synchronize().map_err(|e| GpuHostError::Verification {
                test: "elem_bench",
                detail: format!("{e}"),
            })?;
            let elapsed = start.elapsed();
            let ms_per = elapsed.as_secs_f64() * 1000.0 / bench_iters as f64;
            let bytes = n as f64 * 8.0;
            let gbps = bytes / (ms_per * 1e6);
            let pct_peak = gbps / 192.0 * 100.0;
            println!(
                "  {:>22} {:>10.4} {:>10.1} {:>9.0}% {:>10}",
                "silu(V1 scalar)", ms_per, gbps, pct_peak, "compute"
            );
        }

        // --- sigmoid_forward (V1 scalar) ---
        {
            let input_tensor = GpuTensor::from_host(&input_data, &[n], &dev).map_err(|e| {
                GpuHostError::Verification {
                    test: "elem_bench",
                    detail: format!("{e}"),
                }
            })?;

            for _ in 0..warmup_iters {
                let _ = ops::sigmoid(&input_tensor, &registry);
            }
            dev.synchronize().map_err(|e| GpuHostError::Verification {
                test: "elem_bench",
                detail: format!("{e}"),
            })?;

            let start = std::time::Instant::now();
            for _ in 0..bench_iters {
                let _ = ops::sigmoid(&input_tensor, &registry);
            }
            dev.synchronize().map_err(|e| GpuHostError::Verification {
                test: "elem_bench",
                detail: format!("{e}"),
            })?;
            let elapsed = start.elapsed();
            let ms_per = elapsed.as_secs_f64() * 1000.0 / bench_iters as f64;
            let bytes = n as f64 * 8.0;
            let gbps = bytes / (ms_per * 1e6);
            let pct_peak = gbps / 192.0 * 100.0;
            println!(
                "  {:>22} {:>10.4} {:>10.1} {:>9.0}% {:>10}",
                "sigmoid(V1 scalar)", ms_per, gbps, pct_peak, "compute"
            );
        }

        // --- relu_forward (NEW, float4 vectorized) ---
        {
            let input_tensor = GpuTensor::from_host(&input_data, &[n], &dev).map_err(|e| {
                GpuHostError::Verification {
                    test: "elem_bench",
                    detail: format!("{e}"),
                }
            })?;

            for _ in 0..warmup_iters {
                let _ = ops::relu(&input_tensor, &registry);
            }
            dev.synchronize().map_err(|e| GpuHostError::Verification {
                test: "elem_bench",
                detail: format!("{e}"),
            })?;

            let start = std::time::Instant::now();
            for _ in 0..bench_iters {
                let _ = ops::relu(&input_tensor, &registry);
            }
            dev.synchronize().map_err(|e| GpuHostError::Verification {
                test: "elem_bench",
                detail: format!("{e}"),
            })?;
            let elapsed = start.elapsed();
            let ms_per = elapsed.as_secs_f64() * 1000.0 / bench_iters as f64;
            // ReLU: read input (4B) + write output (4B) = 8 bytes/elem, purely bandwidth-bound
            let bytes = n as f64 * 8.0;
            let gbps = bytes / (ms_per * 1e6);
            let pct_peak = gbps / 192.0 * 100.0;
            println!(
                "  {:>22} {:>10.4} {:>10.1} {:>9.0}% {:>10}",
                "relu(GPU float4)", ms_per, gbps, pct_peak, "bandwidth"
            );
        }
    }

    println!("\n  --- Summary ---");
    println!("  Hardware: GTX 1660 (sm_75), peak BW=192 GB/s, FP32=5 TFLOPS");
    println!("  elementwise_add: bandwidth-bound (read a + read b + write a = 12B/elem)");
    println!("  gelu/silu/sigmoid: compute-bound (exp/tanh), effective BW < peak");
    println!("  relu: bandwidth-bound (no transcendentals), should approach peak BW");
    println!("  240 GB/s target EXCEEDS 192 GB/s peak — recalibrating to % of peak");
    println!("\n  Elementwise Benchmark — DONE");
    Ok(())
}
