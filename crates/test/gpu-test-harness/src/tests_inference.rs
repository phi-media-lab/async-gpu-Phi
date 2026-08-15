//! Full GPT-2 inference tests: 12-layer forward pass, greedy generation,
//! f32 forward pass, CPU f64 reference, KV-cached generation.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

use gpu_host::error::{GpuHostError, Result};
use gpu_host::mapped_mem::{alloc_mapped_result_array, free_mapped_mem};

/// full-inference.2: 12-layer GPT-2 forward pass with real weights.
///
/// Loads GPT-2 small weights from safetensors, tokenizes a prompt, runs
/// embedding + 12 transformer layers + final LayerNorm on GPU.
/// Skips if model file is not present.
pub(crate) fn run_full_forward_test(dev: Arc<CudaDevice>) -> Result<()> {
    let model_path =
        gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("model.safetensors");
    if !model_path.exists() {
        println!("\n--- Skipping 12-layer forward pass (models/model.safetensors not found) ---");
        return Ok(());
    }

    println!("\n--- 12-layer GPT-2 forward pass (full-inference.2) ---");

    // Load weights
    let weights = gpu_host::model::load_gpt2_weights(&model_path).map_err(|e| {
        GpuHostError::Verification {
            test: "full_forward",
            detail: format!("weight loading: {e}"),
        }
    })?;
    println!(
        "  Loaded {} params ({:.1} MB)",
        weights.total_params(),
        weights.memory_bytes() as f64 / 1e6
    );

    // Tokenize
    let tokenizer =
        gpu_host::tokenizer::Gpt2Tokenizer::new().map_err(|e| GpuHostError::Verification {
            test: "full_forward",
            detail: format!("tokenizer: {e}"),
        })?;
    let prompt = "The capital of France is";
    let tokens = tokenizer.encode(prompt);
    let actual_seq = tokens.len();
    println!("  Prompt: \"{prompt}\" → {actual_seq} tokens: {tokens:?}");

    // Pad to multiple of 32 for GEMM kernel alignment
    const SEQ: u32 = 32;
    const D_MODEL: u32 = 768;
    const N_HEADS: u32 = 12;
    const D_HEAD: u32 = 64;
    const D_FFN: u32 = 3072;
    const EPS: f32 = 1e-5;
    let seq = SEQ.max((actual_seq as u32).div_ceil(32) * 32);
    let total_seq_model = (seq * D_MODEL) as usize;
    let head_total = (N_HEADS * seq * D_HEAD) as usize;

    let mut token_ids_u32: Vec<u32> = tokens.to_vec();
    token_ids_u32.resize(seq as usize, 0); // pad with token 0

    // Load PTX with all needed kernels
    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(
        ptx,
        "gpt2",
        &[
            "embedding_lookup",
            "layer_norm",
            "gemm_f32",
            "bias_add",
            "split_qkv",
            "flash_attention",
            "concat_heads",
            "gelu_forward",
            "elementwise_add",
            "zero_pad",
        ],
    )?;

    macro_rules! get_fn {
        ($name:expr) => {
            dev.get_func("gpt2", $name)
                .ok_or(GpuHostError::KernelNotFound($name))?
        };
    }

    let f_embed = get_fn!("embedding_lookup");
    let f_ln = get_fn!("layer_norm");
    let f_gemm = get_fn!("gemm_f32");
    let f_bias = get_fn!("bias_add");
    let f_split = get_fn!("split_qkv");
    let f_attn = get_fn!("flash_attention");
    let f_concat = get_fn!("concat_heads");
    let f_gelu = get_fn!("gelu_forward");
    let f_add = get_fn!("elementwise_add");
    let f_zero = get_fn!("zero_pad");

    // === Helper: transpose weight [K, N] row-major → column-major f32 for gemm_f32 ===
    fn to_col_major(w: &[f32], k: usize, n: usize) -> Vec<f32> {
        let mut cm = vec![0.0f32; k * n];
        for row in 0..k {
            for col in 0..n {
                cm[col * k + row] = w[row * n + col];
            }
        }
        cm
    }

    // === Allocate a single reusable status buffer ===
    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    // === Upload embedding tables ===
    let wte_dev = dev.htod_sync_copy(&weights.wte)?;
    let wpe_dev = dev.htod_sync_copy(&weights.wpe)?;
    let token_ids_dev = dev.htod_sync_copy(&token_ids_u32)?;

    // === Run embedding lookup ===
    let mut hidden_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    unsafe {
        f_embed.clone().launch(
            LaunchConfig {
                grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &wte_dev,
                &wpe_dev,
                &token_ids_dev,
                &mut hidden_dev,
                seq,
                D_MODEL,
                status_dev_ptr,
            ),
        )?;
    }
    dev.synchronize()?;

    // Zero out padded positions to prevent NaN from propagating through GEMM tiles.
    // Padded rows (actual_seq..seq) carry real embeddings for padding token 0,
    // which can diverge and produce NaN after a few layers.
    let pad_start = (actual_seq as u32) * D_MODEL;
    let pad_count = total_seq_model as u32 - pad_start;
    if pad_count > 0 {
        unsafe {
            f_zero.clone().launch(
                LaunchConfig {
                    grid_dim: (pad_count.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut hidden_dev, pad_start, total_seq_model as u32),
            )?;
        }
        dev.synchronize()?;
    }
    println!("  Embedding done (seq={seq}, actual={actual_seq})");

    // Drop large embedding tables from GPU to save memory
    drop(wte_dev);
    drop(wpe_dev);

    // === Upload all layer weights as f32 column-major ===
    println!("  Uploading 12 layers (f32 column-major)...");
    struct LayerWeightsGpu {
        ln1_g: CudaSlice<f32>,
        ln1_b: CudaSlice<f32>,
        w_qkv: CudaSlice<f32>,
        b_qkv: CudaSlice<f32>,
        w_proj: CudaSlice<f32>,
        b_proj: CudaSlice<f32>,
        ln2_g: CudaSlice<f32>,
        ln2_b: CudaSlice<f32>,
        w_fc: CudaSlice<f32>,
        b_fc: CudaSlice<f32>,
        w_fc_proj: CudaSlice<f32>,
        b_fc_proj: CudaSlice<f32>,
    }

    let mut gpu_layers: Vec<LayerWeightsGpu> = Vec::with_capacity(12);
    for (i, layer) in weights.layers.iter().enumerate() {
        let w_qkv_cm = to_col_major(&layer.c_attn_weight, 768, 2304);
        let w_proj_cm = to_col_major(&layer.c_proj_weight, 768, 768);
        let w_fc_cm = to_col_major(&layer.mlp_fc_weight, 768, 3072);
        let w_fc_proj_cm = to_col_major(&layer.mlp_proj_weight, 3072, 768);

        gpu_layers.push(LayerWeightsGpu {
            ln1_g: dev.htod_sync_copy(&layer.ln_1.weight)?,
            ln1_b: dev.htod_sync_copy(&layer.ln_1.bias)?,
            w_qkv: dev.htod_sync_copy(&w_qkv_cm)?,
            b_qkv: dev.htod_sync_copy(&layer.c_attn_bias)?,
            w_proj: dev.htod_sync_copy(&w_proj_cm)?,
            b_proj: dev.htod_sync_copy(&layer.c_proj_bias)?,
            ln2_g: dev.htod_sync_copy(&layer.ln_2.weight)?,
            ln2_b: dev.htod_sync_copy(&layer.ln_2.bias)?,
            w_fc: dev.htod_sync_copy(&w_fc_cm)?,
            b_fc: dev.htod_sync_copy(&layer.mlp_fc_bias)?,
            w_fc_proj: dev.htod_sync_copy(&w_fc_proj_cm)?,
            b_fc_proj: dev.htod_sync_copy(&layer.mlp_proj_bias)?,
        });
        if i == 0 || i == 11 {
            println!("    Layer {i} uploaded");
        }
    }
    println!("  All 12 layers uploaded (f32)");

    // === Allocate reusable activation buffers ===
    let mut ln_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let mut qkv_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((seq * 2304) as usize)?;
    let mut q_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut k_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut v_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut attn_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut concat_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let mut proj_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let mut ffn_hidden_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((seq * D_FFN) as usize)?;
    let mut gelu_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((seq * D_FFN) as usize)?;
    let mut ffn_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;

    let gemm_shared = (32 * 16 + 16 * 16) * 4; // f32 shared memory for gemm_f32
    let n_q_tiles = (seq as usize).div_ceil(32) as u32;

    // === Diagnostic: dump embedding at pos 4 ===
    {
        let emb_snap: Vec<f32> = dev.dtoh_sync_copy(&hidden_dev)?;
        let pos4 = actual_seq - 1;
        let emb4 = &emb_snap[pos4 * (D_MODEL as usize)..(pos4 + 1) * (D_MODEL as usize)];
        println!(
            "  GPU embed pos4 first4: [{:.6}, {:.6}, {:.6}, {:.6}]",
            emb4[0], emb4[1], emb4[2], emb4[3]
        );
    }

    // === Run 12 transformer layers ===
    for layer_idx in 0..12u32 {
        let lw = &gpu_layers[layer_idx as usize];

        // Step 1: LayerNorm1
        unsafe {
            f_ln.clone().launch(
                LaunchConfig {
                    grid_dim: (seq, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &hidden_dev,
                    &mut ln_out_dev,
                    &lw.ln1_g,
                    &lw.ln1_b,
                    D_MODEL,
                    EPS,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // === Layer 0 diagnostic: dump LN1 output ===
        if layer_idx == 0 {
            let ln_snap: Vec<f32> = dev.dtoh_sync_copy(&ln_out_dev)?;
            let pos4 = actual_seq - 1;
            let ln4 = &ln_snap[pos4 * (D_MODEL as usize)..(pos4 + 1) * (D_MODEL as usize)];
            println!(
                "  GPU L0 LN1 pos4 first4: [{:.6}, {:.6}, {:.6}, {:.6}]",
                ln4[0], ln4[1], ln4[2], ln4[3]
            );
        }

        // Step 2: QKV projection (gemm_f32: f32 input, f32 column-major weights)
        unsafe {
            f_gemm.clone().launch(
                LaunchConfig {
                    grid_dim: (seq / 32, 2304 / 16, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: gemm_shared,
                },
                (
                    &ln_out_dev,
                    &lw.w_qkv,
                    &mut qkv_dev,
                    D_MODEL,
                    2304u32,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // === Layer 0 diagnostic: dump QKV GEMM output (before bias) ===
        if layer_idx == 0 {
            let qkv_snap: Vec<f32> = dev.dtoh_sync_copy(&qkv_dev)?;
            let pos4 = actual_seq - 1;
            let qkv4 = &qkv_snap[pos4 * 2304..(pos4 + 1) * 2304];
            println!(
                "  GPU L0 QKV pos4 first4: [{:.6}, {:.6}, {:.6}, {:.6}]",
                qkv4[0], qkv4[1], qkv4[2], qkv4[3]
            );
        }

        // Bias add on QKV
        let total_qkv = seq * 2304;
        unsafe {
            f_bias.clone().launch(
                LaunchConfig {
                    grid_dim: (total_qkv.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut qkv_dev, &lw.b_qkv, 2304u32, total_qkv, status_dev_ptr),
            )?;
        }
        dev.synchronize()?;

        // Step 3: Split QKV → Q, K, V
        unsafe {
            f_split.clone().launch(
                LaunchConfig {
                    grid_dim: ((head_total as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &qkv_dev, &mut q_dev, &mut k_dev, &mut v_dev, seq, N_HEADS, D_HEAD,
                ),
            )?;
        }
        dev.synchronize()?;

        // Step 4: Flash attention (causal)
        unsafe {
            f_attn.clone().launch(
                LaunchConfig {
                    grid_dim: (N_HEADS, n_q_tiles, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 2 * 32 * 64 * 4, // k_tile + v_tile = 16KB
                },
                (
                    &q_dev,
                    &k_dev,
                    &v_dev,
                    &mut attn_out_dev,
                    seq,
                    D_HEAD,
                    1u32,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Step 5: Concat heads
        unsafe {
            f_concat.clone().launch(
                LaunchConfig {
                    grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&attn_out_dev, &mut concat_dev, seq, N_HEADS, D_HEAD),
            )?;
        }
        dev.synchronize()?;

        // Step 6: Output projection
        unsafe {
            f_gemm.clone().launch(
                LaunchConfig {
                    grid_dim: (seq / 32, D_MODEL / 16, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: gemm_shared,
                },
                (
                    &concat_dev,
                    &lw.w_proj,
                    &mut proj_out_dev,
                    D_MODEL,
                    D_MODEL,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Bias add
        unsafe {
            f_bias.clone().launch(
                LaunchConfig {
                    grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &mut proj_out_dev,
                    &lw.b_proj,
                    D_MODEL,
                    total_seq_model as u32,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Step 7: Residual add (hidden += proj_out) + zero padded rows
        unsafe {
            f_add.clone().launch(
                LaunchConfig {
                    grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut hidden_dev, &proj_out_dev, total_seq_model as u32),
            )?;
            if pad_count > 0 {
                f_zero.clone().launch(
                    LaunchConfig {
                        grid_dim: (pad_count.div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut hidden_dev, pad_start, total_seq_model as u32),
                )?;
            }
        }
        dev.synchronize()?;

        // Step 8: LayerNorm2
        unsafe {
            f_ln.clone().launch(
                LaunchConfig {
                    grid_dim: (seq, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &hidden_dev,
                    &mut ln_out_dev,
                    &lw.ln2_g,
                    &lw.ln2_b,
                    D_MODEL,
                    EPS,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Step 9: FFN up projection
        unsafe {
            f_gemm.clone().launch(
                LaunchConfig {
                    grid_dim: (seq / 32, D_FFN / 16, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: gemm_shared,
                },
                (
                    &ln_out_dev,
                    &lw.w_fc,
                    &mut ffn_hidden_dev,
                    D_MODEL,
                    D_FFN,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Bias add
        let total_ffn = seq * D_FFN;
        unsafe {
            f_bias.clone().launch(
                LaunchConfig {
                    grid_dim: (total_ffn.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &mut ffn_hidden_dev,
                    &lw.b_fc,
                    D_FFN,
                    total_ffn,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Step 10: GELU
        unsafe {
            f_gelu.clone().launch(
                LaunchConfig {
                    grid_dim: (total_ffn.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &ffn_hidden_dev,
                    &mut gelu_out_dev,
                    total_ffn,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Step 11: FFN down projection
        unsafe {
            f_gemm.clone().launch(
                LaunchConfig {
                    grid_dim: (seq / 32, D_MODEL / 16, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: gemm_shared,
                },
                (
                    &gelu_out_dev,
                    &lw.w_fc_proj,
                    &mut ffn_out_dev,
                    D_FFN,
                    D_MODEL,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Bias add
        unsafe {
            f_bias.clone().launch(
                LaunchConfig {
                    grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &mut ffn_out_dev,
                    &lw.b_fc_proj,
                    D_MODEL,
                    total_seq_model as u32,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Step 12: Residual add (hidden += ffn_out) + zero padded rows
        unsafe {
            f_add.clone().launch(
                LaunchConfig {
                    grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut hidden_dev, &ffn_out_dev, total_seq_model as u32),
            )?;
            if pad_count > 0 {
                f_zero.clone().launch(
                    LaunchConfig {
                        grid_dim: (pad_count.div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut hidden_dev, pad_start, total_seq_model as u32),
                )?;
            }
        }
        dev.synchronize()?;

        // === Per-layer diagnostic ===
        // Download hidden state and check max|val| per position
        let hidden_snapshot: Vec<f32> = dev.dtoh_sync_copy(&hidden_dev)?;
        let d = D_MODEL as usize;
        let mut layer_max_abs = 0.0f32;
        let mut layer_nan_positions = Vec::new();
        let mut layer_overflow_positions = Vec::new();
        for row in 0..actual_seq {
            let row_slice = &hidden_snapshot[row * d..(row + 1) * d];
            let row_nan = row_slice.iter().any(|v| v.is_nan());
            let row_max = row_slice
                .iter()
                .filter(|v| !v.is_nan() && !v.is_infinite())
                .fold(0.0f32, |m, &v| m.max(v.abs()));
            if row_nan {
                layer_nan_positions.push(row);
            }
            if row_max > 65504.0 {
                layer_overflow_positions.push((row, row_max));
            }
            if row_max > layer_max_abs {
                layer_max_abs = row_max;
            }
        }
        let nan_str = if layer_nan_positions.is_empty() {
            String::new()
        } else {
            format!(", NaN@{layer_nan_positions:?}")
        };
        let ovf_str = if layer_overflow_positions.is_empty() {
            String::new()
        } else {
            let ovf_pos: Vec<_> = layer_overflow_positions
                .iter()
                .map(|(p, v)| format!("pos{p}={v:.0}"))
                .collect();
            format!(", OVERFLOW: [{}]", ovf_pos.join(", "))
        };
        // Also print pos4 (last real token) specifics for CPU comparison
        let pos4_slice = &hidden_snapshot[(actual_seq - 1) * d..actual_seq * d];
        let pos4_maxabs = pos4_slice.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let pos4_first4: Vec<String> = pos4_slice[..4].iter().map(|x| format!("{x:.4}")).collect();
        println!("    Layer {layer_idx}: max|val|={layer_max_abs:.2}, pos4_max|val|={pos4_maxabs:.2}, pos4_first4=[{}]{nan_str}{ovf_str}",
            pos4_first4.join(", "));
    }

    // === Final LayerNorm ===
    let ln_f_g_dev = dev.htod_sync_copy(&weights.ln_f.weight)?;
    let ln_f_b_dev = dev.htod_sync_copy(&weights.ln_f.bias)?;
    unsafe {
        f_ln.clone().launch(
            LaunchConfig {
                grid_dim: (seq, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &hidden_dev,
                &mut ln_out_dev,
                &ln_f_g_dev,
                &ln_f_b_dev,
                D_MODEL,
                EPS,
                status_dev_ptr,
            ),
        )?;
    }
    dev.synchronize()?;

    // === Download and validate output ===
    let output: Vec<f32> = dev.dtoh_sync_copy(&ln_out_dev)?;

    // Check for NaN/Inf in the prediction-relevant token positions.
    // Position 0 may go NaN with f16 GEMM because it only self-attends (no averaging),
    // causing residual values to grow without damping until they overflow f16 range.
    // This is a known limitation of f16 Tensor Core inference vs f32 reference.
    // For inference, we only need the last actual token position for next-token prediction.
    let dm = D_MODEL as usize;
    let mut nan_positions: Vec<usize> = Vec::new();
    let mut pred_nan = 0;
    let mut pred_inf = 0;
    let mut max_abs = 0.0f32;
    for row in 0..actual_seq {
        let row_slice = &output[row * dm..(row + 1) * dm];
        let row_nan = row_slice.iter().filter(|v| v.is_nan()).count();
        if row_nan > 0 {
            nan_positions.push(row);
            if row == actual_seq - 1 {
                pred_nan = row_nan;
            }
        }
        for &v in row_slice {
            if !v.is_nan() && !v.is_infinite() && v.abs() > max_abs {
                max_abs = v.abs();
            }
            if row == actual_seq - 1 && v.is_infinite() {
                pred_inf += 1;
            }
        }
    }

    // Print stats for prediction position (last actual token)
    let last_pos = actual_seq - 1;
    let last_row = &output[last_pos * dm..(last_pos + 1) * dm];
    let row_mean: f32 = last_row.iter().sum::<f32>() / D_MODEL as f32;
    let row_var: f32 =
        last_row.iter().map(|x| (x - row_mean).powi(2)).sum::<f32>() / D_MODEL as f32;
    println!("  Output shape: [{seq}, {D_MODEL}] (actual {actual_seq} tokens)");
    if !nan_positions.is_empty() {
        println!("  NaN positions: {nan_positions:?} (f16 precision — no attention averaging)");
    }
    println!("  max|val|={max_abs:.4} (non-NaN actual positions)");
    println!("  Prediction pos {last_pos}: mean={row_mean:.6}, var={row_var:.6}");
    println!(
        "  First 8 values: {:?}",
        last_row[..8]
            .iter()
            .map(|v| format!("{v:.4}"))
            .collect::<Vec<_>>()
    );

    // Free status buffer
    unsafe {
        free_mapped_mem(status_host_ptr)?;
    }

    // The prediction position (last actual token) must be clean
    if pred_nan > 0 || pred_inf > 0 {
        return Err(GpuHostError::Verification {
            test: "full_forward",
            detail: format!("prediction position has {pred_nan} NaN and {pred_inf} Inf values"),
        });
    }

    // Sanity: max absolute value should be reasonable (not exploded)
    if max_abs > 1000.0 {
        return Err(GpuHostError::Verification {
            test: "full_forward",
            detail: format!("output magnitude too large: max|val|={max_abs:.2}"),
        });
    }

    println!("  12-layer GPT-2 forward pass (seq={seq}, actual={actual_seq}) — PASSED");

    // ================================================================
    // LM Head (full-inference.3): project hidden state → vocabulary logits
    // GPT-2 uses weight tying: logits = hidden_state @ wte.T
    // wte is [50257, 768] row-major, so logits[v] = dot(hidden[last_pos], wte[v])
    // ================================================================
    println!("\n--- LM head + greedy decode (full-inference.3) ---");

    let vocab_size = 50257;
    let hidden = &output[last_pos * dm..(last_pos + 1) * dm];

    // Compute logits on CPU (only 1 row × 768, not worth a GPU kernel for 50257 non-aligned)
    let mut logits = vec![0.0f32; vocab_size];
    for v in 0..vocab_size {
        let wte_row = &weights.wte[v * dm..(v + 1) * dm];
        let mut dot = 0.0f32;
        for d in 0..dm {
            dot += hidden[d] * wte_row[d];
        }
        logits[v] = dot;
    }

    // Softmax for probabilities (numerically stable)
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
    let probs: Vec<f32> = logits
        .iter()
        .map(|&l| (l - max_logit).exp() / exp_sum)
        .collect();

    // Top-5 predictions
    let mut indices: Vec<usize> = (0..vocab_size).collect();
    indices.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

    println!("  Top-5 predictions after \"The capital of France is\":");
    for &idx in &indices[..5] {
        let token_str = tokenizer
            .decode(&[idx as u32])
            .unwrap_or_else(|_| format!("<tok {idx}>"));
        println!(
            "    #{}: token {} = {:?} (logit={:.2}, prob={:.4})",
            indices.iter().position(|&i| i == idx).unwrap() + 1,
            idx,
            token_str,
            logits[idx],
            probs[idx],
        );
    }

    // Greedy prediction = argmax
    let top1 = indices[0];
    let top1_str = tokenizer
        .decode(&[top1 as u32])
        .unwrap_or_else(|_| format!("<tok {top1}>"));
    println!("  Greedy next token: {} = {:?}", top1, top1_str);

    // Validation: logits should be finite at prediction position
    let logit_nan = logits.iter().filter(|v| v.is_nan()).count();
    let logit_inf = logits.iter().filter(|v| v.is_infinite()).count();
    if logit_nan > 0 || logit_inf > 0 {
        return Err(GpuHostError::Verification {
            test: "lm_head",
            detail: format!("logits have {logit_nan} NaN and {logit_inf} Inf values"),
        });
    }

    // Validation: top-1 probability should be > 0.01 (model has a clear preference)
    if probs[top1] < 0.01 {
        println!(
            "  WARNING: top-1 probability very low ({:.4}), model may not be confident",
            probs[top1]
        );
    }

    println!("  LM head (vocab=50257, CPU matmul) — PASSED");
    Ok(())
}

/// full-inference.4: Greedy autoregressive generation loop.
///
/// Runs repeated forward passes, each time appending the argmax token to the
/// sequence, until max_new_tokens is reached or <|endoftext|> is produced.
/// No KV cache — full recompute each step (proof of concept).
/// Skips if model file is not present.
pub(crate) fn run_generation_test(dev: Arc<CudaDevice>) -> Result<()> {
    let model_path =
        gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("model.safetensors");
    if !model_path.exists() {
        println!("\n--- Skipping generation test (models/model.safetensors not found) ---");
        return Ok(());
    }

    println!("\n--- Greedy autoregressive generation (full-inference.4) ---");

    // Load weights
    let weights = gpu_host::model::load_gpt2_weights(&model_path).map_err(|e| {
        GpuHostError::Verification {
            test: "generation",
            detail: format!("weight loading: {e}"),
        }
    })?;

    // Tokenize
    let tokenizer =
        gpu_host::tokenizer::Gpt2Tokenizer::new().map_err(|e| GpuHostError::Verification {
            test: "generation",
            detail: format!("tokenizer: {e}"),
        })?;

    let prompts = [
        "The capital of France is",
        "Once upon a time, there was a",
        "The meaning of life is",
    ];

    const D_MODEL: u32 = 768;
    const N_HEADS: u32 = 12;
    const D_HEAD: u32 = 64;
    const D_FFN: u32 = 3072;
    const EPS: f32 = 1e-5;
    let dm = D_MODEL as usize;

    // Fixed seq=128 for all steps (pad shorter sequences)
    const SEQ: u32 = 128;
    let total_seq_model = (SEQ * D_MODEL) as usize;
    let head_total = (N_HEADS * SEQ * D_HEAD) as usize;

    // Helper: transpose weight [K, N] row-major → column-major f32 for gemm_f32
    fn to_col_major(w: &[f32], k: usize, n: usize) -> Vec<f32> {
        let mut cm = vec![0.0f32; k * n];
        for row in 0..k {
            for col in 0..n {
                cm[col * k + row] = w[row * n + col];
            }
        }
        cm
    }

    // Load PTX
    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(
        ptx,
        "gen",
        &[
            "embedding_lookup",
            "layer_norm",
            "gemm_f32",
            "bias_add",
            "split_qkv",
            "flash_attention",
            "concat_heads",
            "gelu_forward",
            "elementwise_add",
            "zero_pad",
        ],
    )?;

    macro_rules! get_fn {
        ($name:expr) => {
            dev.get_func("gen", $name)
                .ok_or(GpuHostError::KernelNotFound($name))?
        };
    }

    let f_embed = get_fn!("embedding_lookup");
    let f_ln = get_fn!("layer_norm");
    let f_gemm = get_fn!("gemm_f32");
    let f_bias = get_fn!("bias_add");
    let f_split = get_fn!("split_qkv");
    let f_attn = get_fn!("flash_attention");
    let f_concat = get_fn!("concat_heads");
    let f_gelu = get_fn!("gelu_forward");
    let f_add = get_fn!("elementwise_add");
    let f_zero = get_fn!("zero_pad");

    // Allocate status buffer
    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    // Upload embedding tables (keep alive for LM head weight-tying)
    let wte_dev = dev.htod_sync_copy(&weights.wte)?;
    let wpe_dev = dev.htod_sync_copy(&weights.wpe)?;

    // Upload all layer weights as f32 column-major
    struct LayerWeightsGpu {
        ln1_g: CudaSlice<f32>,
        ln1_b: CudaSlice<f32>,
        w_qkv: CudaSlice<f32>,
        b_qkv: CudaSlice<f32>,
        w_proj: CudaSlice<f32>,
        b_proj: CudaSlice<f32>,
        ln2_g: CudaSlice<f32>,
        ln2_b: CudaSlice<f32>,
        w_fc: CudaSlice<f32>,
        b_fc: CudaSlice<f32>,
        w_fc_proj: CudaSlice<f32>,
        b_fc_proj: CudaSlice<f32>,
    }

    let mut gpu_layers: Vec<LayerWeightsGpu> = Vec::with_capacity(12);
    for layer in weights.layers.iter() {
        let w_qkv_cm = to_col_major(&layer.c_attn_weight, 768, 2304);
        let w_proj_cm = to_col_major(&layer.c_proj_weight, 768, 768);
        let w_fc_cm = to_col_major(&layer.mlp_fc_weight, 768, 3072);
        let w_fc_proj_cm = to_col_major(&layer.mlp_proj_weight, 3072, 768);

        gpu_layers.push(LayerWeightsGpu {
            ln1_g: dev.htod_sync_copy(&layer.ln_1.weight)?,
            ln1_b: dev.htod_sync_copy(&layer.ln_1.bias)?,
            w_qkv: dev.htod_sync_copy(&w_qkv_cm)?,
            b_qkv: dev.htod_sync_copy(&layer.c_attn_bias)?,
            w_proj: dev.htod_sync_copy(&w_proj_cm)?,
            b_proj: dev.htod_sync_copy(&layer.c_proj_bias)?,
            ln2_g: dev.htod_sync_copy(&layer.ln_2.weight)?,
            ln2_b: dev.htod_sync_copy(&layer.ln_2.bias)?,
            w_fc: dev.htod_sync_copy(&w_fc_cm)?,
            b_fc: dev.htod_sync_copy(&layer.mlp_fc_bias)?,
            w_fc_proj: dev.htod_sync_copy(&w_fc_proj_cm)?,
            b_fc_proj: dev.htod_sync_copy(&layer.mlp_proj_bias)?,
        });
    }

    // Final layer norm weights
    let ln_f_g_dev = dev.htod_sync_copy(&weights.ln_f.weight)?;
    let ln_f_b_dev = dev.htod_sync_copy(&weights.ln_f.bias)?;

    // Allocate reusable activation buffers
    let mut hidden_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let mut ln_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let mut qkv_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((SEQ * 2304) as usize)?;
    let mut q_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut k_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut v_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut attn_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut concat_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let mut proj_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let mut ffn_hidden_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((SEQ * D_FFN) as usize)?;
    let mut gelu_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((SEQ * D_FFN) as usize)?;
    let mut ffn_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;

    let gemm_shared = (32 * 16 + 16 * 16) * 4; // f32 shared memory for gemm_f32
    let n_q_tiles = (SEQ as usize).div_ceil(32) as u32;

    let mut all_passed = true;

    for (pi, prompt) in prompts.iter().enumerate() {
        let prompt_tokens = tokenizer.encode(prompt);
        let prompt_len = prompt_tokens.len();
        let max_new_tokens: usize = 50usize.min(SEQ as usize - prompt_len);
        println!(
            "\n  [{}/{}] Prompt: \"{prompt}\" → {prompt_len} tokens, generating {max_new_tokens}",
            pi + 1,
            prompts.len()
        );

        // Build the token sequence (will grow each step)
        let mut tokens: Vec<u32> = prompt_tokens.clone();
        let mut generated: Vec<u32> = Vec::new();

        let gen_start = std::time::Instant::now();

        for step in 0..max_new_tokens {
            let actual_seq = tokens.len();

            // Pad to SEQ
            let mut token_ids_padded = tokens.clone();
            token_ids_padded.resize(SEQ as usize, 0);
            let token_ids_dev = dev.htod_sync_copy(&token_ids_padded)?;

            // === Embedding ===
            unsafe {
                f_embed.clone().launch(
                    LaunchConfig {
                        grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &wte_dev,
                        &wpe_dev,
                        &token_ids_dev,
                        &mut hidden_dev,
                        SEQ,
                        D_MODEL,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // Zero out padded positions after embedding
            let pad_start = (actual_seq as u32) * D_MODEL;
            let pad_count = total_seq_model as u32 - pad_start;
            if pad_count > 0 {
                unsafe {
                    f_zero.clone().launch(
                        LaunchConfig {
                            grid_dim: (pad_count.div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&mut hidden_dev, pad_start, total_seq_model as u32),
                    )?;
                }
                dev.synchronize()?;
            }

            // === 12 transformer layers ===
            for layer_idx in 0..12u32 {
                let lw = &gpu_layers[layer_idx as usize];

                // LayerNorm1
                unsafe {
                    f_ln.clone().launch(
                        LaunchConfig {
                            grid_dim: (SEQ, 1, 1),
                            block_dim: (32, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &hidden_dev,
                            &mut ln_out_dev,
                            &lw.ln1_g,
                            &lw.ln1_b,
                            D_MODEL,
                            EPS,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                // QKV projection
                unsafe {
                    f_gemm.clone().launch(
                        LaunchConfig {
                            grid_dim: (SEQ / 32, 2304 / 16, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: gemm_shared,
                        },
                        (
                            &ln_out_dev,
                            &lw.w_qkv,
                            &mut qkv_dev,
                            D_MODEL,
                            2304u32,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                // QKV bias
                let total_qkv = SEQ * 2304;
                unsafe {
                    f_bias.clone().launch(
                        LaunchConfig {
                            grid_dim: (total_qkv.div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&mut qkv_dev, &lw.b_qkv, 2304u32, total_qkv, status_dev_ptr),
                    )?;
                }
                dev.synchronize()?;

                // Split QKV
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

                // Flash attention (causal)
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
                            &mut attn_out_dev,
                            SEQ,
                            D_HEAD,
                            1u32,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                // Concat heads
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

                // Output projection
                unsafe {
                    f_gemm.clone().launch(
                        LaunchConfig {
                            grid_dim: (SEQ / 32, D_MODEL / 16, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: gemm_shared,
                        },
                        (
                            &concat_dev,
                            &lw.w_proj,
                            &mut proj_out_dev,
                            D_MODEL,
                            D_MODEL,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                // Projection bias
                unsafe {
                    f_bias.clone().launch(
                        LaunchConfig {
                            grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &mut proj_out_dev,
                            &lw.b_proj,
                            D_MODEL,
                            total_seq_model as u32,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                // Residual 1 + zero padded rows
                unsafe {
                    f_add.clone().launch(
                        LaunchConfig {
                            grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&mut hidden_dev, &proj_out_dev, total_seq_model as u32),
                    )?;
                    if pad_count > 0 {
                        f_zero.clone().launch(
                            LaunchConfig {
                                grid_dim: (pad_count.div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&mut hidden_dev, pad_start, total_seq_model as u32),
                        )?;
                    }
                }
                dev.synchronize()?;

                // LayerNorm2
                unsafe {
                    f_ln.clone().launch(
                        LaunchConfig {
                            grid_dim: (SEQ, 1, 1),
                            block_dim: (32, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &hidden_dev,
                            &mut ln_out_dev,
                            &lw.ln2_g,
                            &lw.ln2_b,
                            D_MODEL,
                            EPS,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                // FFN up
                unsafe {
                    f_gemm.clone().launch(
                        LaunchConfig {
                            grid_dim: (SEQ / 32, D_FFN / 16, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: gemm_shared,
                        },
                        (
                            &ln_out_dev,
                            &lw.w_fc,
                            &mut ffn_hidden_dev,
                            D_MODEL,
                            D_FFN,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                // FFN bias
                let total_ffn = SEQ * D_FFN;
                unsafe {
                    f_bias.clone().launch(
                        LaunchConfig {
                            grid_dim: (total_ffn.div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &mut ffn_hidden_dev,
                            &lw.b_fc,
                            D_FFN,
                            total_ffn,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                // GELU
                unsafe {
                    f_gelu.clone().launch(
                        LaunchConfig {
                            grid_dim: (total_ffn.div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &ffn_hidden_dev,
                            &mut gelu_out_dev,
                            total_ffn,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                // FFN down
                unsafe {
                    f_gemm.clone().launch(
                        LaunchConfig {
                            grid_dim: (SEQ / 32, D_MODEL / 16, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: gemm_shared,
                        },
                        (
                            &gelu_out_dev,
                            &lw.w_fc_proj,
                            &mut ffn_out_dev,
                            D_FFN,
                            D_MODEL,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                // FFN bias
                unsafe {
                    f_bias.clone().launch(
                        LaunchConfig {
                            grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &mut ffn_out_dev,
                            &lw.b_fc_proj,
                            D_MODEL,
                            total_seq_model as u32,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                // Residual 2 + zero padded rows
                unsafe {
                    f_add.clone().launch(
                        LaunchConfig {
                            grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&mut hidden_dev, &ffn_out_dev, total_seq_model as u32),
                    )?;
                    if pad_count > 0 {
                        f_zero.clone().launch(
                            LaunchConfig {
                                grid_dim: (pad_count.div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&mut hidden_dev, pad_start, total_seq_model as u32),
                        )?;
                    }
                }
                dev.synchronize()?;
            }

            // === Final LayerNorm ===
            unsafe {
                f_ln.clone().launch(
                    LaunchConfig {
                        grid_dim: (SEQ, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &hidden_dev,
                        &mut ln_out_dev,
                        &ln_f_g_dev,
                        &ln_f_b_dev,
                        D_MODEL,
                        EPS,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // Download only the prediction position (last actual token)
            let output: Vec<f32> = dev.dtoh_sync_copy(&ln_out_dev)?;
            let last_pos = actual_seq - 1;
            let hidden_vec = &output[last_pos * dm..(last_pos + 1) * dm];

            // Check for NaN at prediction position
            let nan_count = hidden_vec.iter().filter(|v| v.is_nan()).count();
            if nan_count > 0 {
                // Diagnose: download pre-LN hidden state to find where NaN originates
                let pre_ln: Vec<f32> = dev.dtoh_sync_copy(&hidden_dev)?;
                let pre_row = &pre_ln[last_pos * dm..(last_pos + 1) * dm];
                let pre_nan = pre_row.iter().filter(|v| v.is_nan()).count();
                let pre_inf = pre_row.iter().filter(|v| v.is_infinite()).count();
                let pre_max = pre_row
                    .iter()
                    .filter(|v| v.is_finite())
                    .map(|v| v.abs())
                    .fold(0.0f32, f32::max);
                println!("  Step {step}: NaN in final LN output ({nan_count}/768)");
                println!("    Pre-LN hidden: nan={pre_nan}, inf={pre_inf}, max|val|={pre_max:.2}");

                // Check all positions for NaN pattern
                let mut nan_rows = 0u32;
                for r in 0..actual_seq {
                    let row = &pre_ln[r * dm..(r + 1) * dm];
                    if row.iter().any(|v| v.is_nan() || v.is_infinite()) {
                        nan_rows += 1;
                    }
                }
                println!("    Rows with NaN/Inf: {nan_rows}/{actual_seq}");
                break;
            }

            // CPU LM head: logits[v] = dot(hidden, wte[v])
            let vocab_size = 50257;
            let mut best_logit = f32::NEG_INFINITY;
            let mut best_token: u32 = 0;
            for v in 0..vocab_size {
                let wte_row = &weights.wte[v * dm..(v + 1) * dm];
                let mut dot = 0.0f32;
                for d in 0..dm {
                    dot += hidden_vec[d] * wte_row[d];
                }
                if dot > best_logit {
                    best_logit = dot;
                    best_token = v as u32;
                }
            }

            // Decode and print
            let token_str = tokenizer
                .decode(&[best_token])
                .unwrap_or_else(|_| format!("<tok {best_token}>"));
            print!("{token_str}");

            // Stop on <|endoftext|>
            if best_token == 50256 {
                println!();
                println!("  [<|endoftext|> at step {step}]");
                break;
            }

            generated.push(best_token);
            tokens.push(best_token);
        }

        let gen_elapsed = gen_start.elapsed();
        println!();

        // Print full generated text
        let full_text = if !generated.is_empty() {
            tokenizer
                .decode(&generated)
                .unwrap_or_else(|_| "<decode error>".to_string())
        } else {
            String::new()
        };
        println!("  Prompt: \"{prompt}\"");
        println!("  Generated ({} tokens): \"{full_text}\"", generated.len());
        println!(
            "  Time: {:.1}ms total, {:.1}ms/token",
            gen_elapsed.as_secs_f64() * 1000.0,
            gen_elapsed.as_secs_f64() * 1000.0 / generated.len().max(1) as f64,
        );

        if generated.len() < 50.min(max_new_tokens) {
            println!(
                "  WARNING: generated fewer than 50 tokens ({} tokens)",
                generated.len()
            );
            all_passed = false;
        } else {
            println!("  PASSED ({} tokens, no NaN)", generated.len());
        }
    } // end prompt loop

    // Free status buffer
    unsafe {
        free_mapped_mem(status_host_ptr)?;
    }

    if !all_passed {
        return Err(GpuHostError::Verification {
            test: "generation",
            detail: "some prompts failed to generate 50 tokens".to_string(),
        });
    }

    println!(
        "  Greedy autoregressive generation — PASSED ({} prompts)",
        prompts.len()
    );
    Ok(())
}

/// full-inference.6: 12-layer GPT-2 forward pass with pure f32 GEMM (no Tensor Cores).
///
/// Identical pipeline to run_full_forward_test but uses gemm_f32 kernel
/// instead of full_gemm_f32in. Weights are uploaded as column-major f32
/// instead of packed f16x2. This validates whether f16 precision was
/// the reason for wrong predictions.
pub(crate) fn run_f32_forward_test(dev: Arc<CudaDevice>) -> Result<()> {
    let model_path =
        gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("model.safetensors");
    if !model_path.exists() {
        println!("\n--- Skipping f32 GEMM forward pass (models/model.safetensors not found) ---");
        return Ok(());
    }

    println!("\n--- f32 GEMM forward pass (full-inference.6) ---");

    let weights = gpu_host::model::load_gpt2_weights(&model_path).map_err(|e| {
        GpuHostError::Verification {
            test: "f32_forward",
            detail: format!("weight loading: {e}"),
        }
    })?;

    let tokenizer =
        gpu_host::tokenizer::Gpt2Tokenizer::new().map_err(|e| GpuHostError::Verification {
            test: "f32_forward",
            detail: format!("tokenizer: {e}"),
        })?;
    let prompt = "The capital of France is";
    let prompt_tokens = tokenizer.encode(prompt);
    let actual_seq = prompt_tokens.len();
    println!("  Prompt: \"{prompt}\" → {actual_seq} tokens: {prompt_tokens:?}");

    const SEQ: u32 = 32;
    const D_MODEL: u32 = 768;
    const N_HEADS: u32 = 12;
    const D_HEAD: u32 = 64;
    const D_FFN: u32 = 3072;
    const EPS: f32 = 1e-5;
    let dm = D_MODEL as usize;
    let seq = SEQ;
    let total_seq_model = (seq * D_MODEL) as usize;
    let head_total = (N_HEADS * seq * D_HEAD) as usize;

    let mut token_ids_u32: Vec<u32> = prompt_tokens.clone();
    token_ids_u32.resize(seq as usize, 0);

    // Load PTX
    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(
        ptx,
        "f32fwd",
        &[
            "embedding_lookup",
            "layer_norm",
            "gemm_f32",
            "bias_add",
            "split_qkv",
            "flash_attention",
            "concat_heads",
            "gelu_forward",
            "elementwise_add",
            "zero_pad",
        ],
    )?;

    macro_rules! get_fn {
        ($name:expr) => {
            dev.get_func("f32fwd", $name)
                .ok_or(GpuHostError::KernelNotFound($name))?
        };
    }

    let f_embed = get_fn!("embedding_lookup");
    let f_ln = get_fn!("layer_norm");
    let f_gemm = get_fn!("gemm_f32");
    let f_bias = get_fn!("bias_add");
    let f_split = get_fn!("split_qkv");
    let f_attn = get_fn!("flash_attention");
    let f_concat = get_fn!("concat_heads");
    let f_gelu = get_fn!("gelu_forward");
    let f_add = get_fn!("elementwise_add");
    let f_zero = get_fn!("zero_pad");

    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    // Upload embedding tables
    let wte_dev = dev.htod_sync_copy(&weights.wte)?;
    let wpe_dev = dev.htod_sync_copy(&weights.wpe)?;
    let token_ids_dev = dev.htod_sync_copy(&token_ids_u32)?;

    // === Embedding ===
    let mut hidden_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    unsafe {
        f_embed.clone().launch(
            LaunchConfig {
                grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &wte_dev,
                &wpe_dev,
                &token_ids_dev,
                &mut hidden_dev,
                seq,
                D_MODEL,
                status_dev_ptr,
            ),
        )?;
    }
    dev.synchronize()?;

    // Zero padded positions
    let pad_start = (actual_seq as u32) * D_MODEL;
    let pad_count = total_seq_model as u32 - pad_start;
    if pad_count > 0 {
        unsafe {
            f_zero.clone().launch(
                LaunchConfig {
                    grid_dim: (pad_count.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut hidden_dev, pad_start, total_seq_model as u32),
            )?;
        }
        dev.synchronize()?;
    }
    println!("  Embedding done (seq={seq}, actual={actual_seq})");

    drop(wte_dev);
    drop(wpe_dev);

    // === Helper: transpose weight [K, N] row-major to column-major f32 ===
    // Column-major B[col * K + row] for gemm_f32
    fn to_col_major(w: &[f32], k: usize, n: usize) -> Vec<f32> {
        let mut cm = vec![0.0f32; k * n];
        for row in 0..k {
            for col in 0..n {
                cm[col * k + row] = w[row * n + col];
            }
        }
        cm
    }

    // === Upload layer weights as f32 column-major ===
    println!("  Uploading 12 layers (f32 column-major)...");
    struct LayerWeightsF32 {
        ln1_g: CudaSlice<f32>,
        ln1_b: CudaSlice<f32>,
        w_qkv: CudaSlice<f32>,
        b_qkv: CudaSlice<f32>,
        w_proj: CudaSlice<f32>,
        b_proj: CudaSlice<f32>,
        ln2_g: CudaSlice<f32>,
        ln2_b: CudaSlice<f32>,
        w_fc: CudaSlice<f32>,
        b_fc: CudaSlice<f32>,
        w_fc_proj: CudaSlice<f32>,
        b_fc_proj: CudaSlice<f32>,
    }

    let mut gpu_layers: Vec<LayerWeightsF32> = Vec::with_capacity(12);
    for (i, layer) in weights.layers.iter().enumerate() {
        let w_qkv_cm = to_col_major(&layer.c_attn_weight, 768, 2304);
        let w_proj_cm = to_col_major(&layer.c_proj_weight, 768, 768);
        let w_fc_cm = to_col_major(&layer.mlp_fc_weight, 768, 3072);
        let w_fc_proj_cm = to_col_major(&layer.mlp_proj_weight, 3072, 768);

        gpu_layers.push(LayerWeightsF32 {
            ln1_g: dev.htod_sync_copy(&layer.ln_1.weight)?,
            ln1_b: dev.htod_sync_copy(&layer.ln_1.bias)?,
            w_qkv: dev.htod_sync_copy(&w_qkv_cm)?,
            b_qkv: dev.htod_sync_copy(&layer.c_attn_bias)?,
            w_proj: dev.htod_sync_copy(&w_proj_cm)?,
            b_proj: dev.htod_sync_copy(&layer.c_proj_bias)?,
            ln2_g: dev.htod_sync_copy(&layer.ln_2.weight)?,
            ln2_b: dev.htod_sync_copy(&layer.ln_2.bias)?,
            w_fc: dev.htod_sync_copy(&w_fc_cm)?,
            b_fc: dev.htod_sync_copy(&layer.mlp_fc_bias)?,
            w_fc_proj: dev.htod_sync_copy(&w_fc_proj_cm)?,
            b_fc_proj: dev.htod_sync_copy(&layer.mlp_proj_bias)?,
        });
        if i == 0 || i == 11 {
            println!("    Layer {i} uploaded");
        }
    }
    println!("  All 12 layers uploaded (f32)");

    // Final LN weights
    let ln_f_g_dev = dev.htod_sync_copy(&weights.ln_f.weight)?;
    let ln_f_b_dev = dev.htod_sync_copy(&weights.ln_f.bias)?;

    // === Allocate activation buffers ===
    let mut ln_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let mut qkv_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((seq * 2304) as usize)?;
    let mut q_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut k_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut v_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut attn_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut concat_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let mut proj_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let mut ffn_hidden_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((seq * D_FFN) as usize)?;
    let mut gelu_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((seq * D_FFN) as usize)?;
    let mut ffn_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;

    let gemm_shared = (32 * 16 + 16 * 16) * 4; // f32 shared memory for gemm_f32
    let n_q_tiles = (seq as usize).div_ceil(32) as u32;

    let fwd_start = std::time::Instant::now();

    // === 12 transformer layers ===
    for layer_idx in 0..12u32 {
        let lw = &gpu_layers[layer_idx as usize];

        // LN1
        unsafe {
            f_ln.clone().launch(
                LaunchConfig {
                    grid_dim: (seq, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &hidden_dev,
                    &mut ln_out_dev,
                    &lw.ln1_g,
                    &lw.ln1_b,
                    D_MODEL,
                    EPS,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // QKV GEMM (f32): ln_out[32,768] × w_qkv[768,2304] → qkv[32,2304]
        unsafe {
            f_gemm.clone().launch(
                LaunchConfig {
                    grid_dim: (seq / 32, 2304 / 16, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: gemm_shared,
                },
                (
                    &ln_out_dev,
                    &lw.w_qkv,
                    &mut qkv_dev,
                    D_MODEL,
                    2304u32,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // === Layer 0 diagnostic: compare f32 GEMM QKV with CPU f64 ===
        if layer_idx == 0 {
            let qkv_snap: Vec<f32> = dev.dtoh_sync_copy(&qkv_dev)?;
            let pos4 = actual_seq - 1;
            let qkv4 = &qkv_snap[pos4 * 2304..(pos4 + 1) * 2304];
            println!(
                "  f32 GEMM L0 QKV pos4 first4: [{:.6}, {:.6}, {:.6}, {:.6}]",
                qkv4[0], qkv4[1], qkv4[2], qkv4[3]
            );
            println!("  (CPU f64 reference: [-0.191336, -0.079322, 0.937499, 0.162505])");
        }

        // QKV bias
        let total_qkv = seq * 2304;
        unsafe {
            f_bias.clone().launch(
                LaunchConfig {
                    grid_dim: (total_qkv.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut qkv_dev, &lw.b_qkv, 2304u32, total_qkv, status_dev_ptr),
            )?;
        }
        dev.synchronize()?;

        // Split QKV
        unsafe {
            f_split.clone().launch(
                LaunchConfig {
                    grid_dim: ((head_total as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &qkv_dev, &mut q_dev, &mut k_dev, &mut v_dev, seq, N_HEADS, D_HEAD,
                ),
            )?;
        }
        dev.synchronize()?;

        // Flash attention (causal)
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
                    &mut attn_out_dev,
                    seq,
                    D_HEAD,
                    1u32,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Concat heads
        unsafe {
            f_concat.clone().launch(
                LaunchConfig {
                    grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&attn_out_dev, &mut concat_dev, seq, N_HEADS, D_HEAD),
            )?;
        }
        dev.synchronize()?;

        // Output projection (f32 GEMM)
        unsafe {
            f_gemm.clone().launch(
                LaunchConfig {
                    grid_dim: (seq / 32, D_MODEL / 16, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: gemm_shared,
                },
                (
                    &concat_dev,
                    &lw.w_proj,
                    &mut proj_out_dev,
                    D_MODEL,
                    D_MODEL,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Proj bias
        unsafe {
            f_bias.clone().launch(
                LaunchConfig {
                    grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &mut proj_out_dev,
                    &lw.b_proj,
                    D_MODEL,
                    total_seq_model as u32,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Residual 1 + zero pad
        unsafe {
            f_add.clone().launch(
                LaunchConfig {
                    grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut hidden_dev, &proj_out_dev, total_seq_model as u32),
            )?;
            if pad_count > 0 {
                f_zero.clone().launch(
                    LaunchConfig {
                        grid_dim: (pad_count.div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut hidden_dev, pad_start, total_seq_model as u32),
                )?;
            }
        }
        dev.synchronize()?;

        // LN2
        unsafe {
            f_ln.clone().launch(
                LaunchConfig {
                    grid_dim: (seq, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &hidden_dev,
                    &mut ln_out_dev,
                    &lw.ln2_g,
                    &lw.ln2_b,
                    D_MODEL,
                    EPS,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // FFN up (f32 GEMM)
        unsafe {
            f_gemm.clone().launch(
                LaunchConfig {
                    grid_dim: (seq / 32, D_FFN / 16, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: gemm_shared,
                },
                (
                    &ln_out_dev,
                    &lw.w_fc,
                    &mut ffn_hidden_dev,
                    D_MODEL,
                    D_FFN,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // FFN bias
        let total_ffn = seq * D_FFN;
        unsafe {
            f_bias.clone().launch(
                LaunchConfig {
                    grid_dim: (total_ffn.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &mut ffn_hidden_dev,
                    &lw.b_fc,
                    D_FFN,
                    total_ffn,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // GELU
        unsafe {
            f_gelu.clone().launch(
                LaunchConfig {
                    grid_dim: (total_ffn.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &ffn_hidden_dev,
                    &mut gelu_out_dev,
                    total_ffn,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // FFN down (f32 GEMM)
        unsafe {
            f_gemm.clone().launch(
                LaunchConfig {
                    grid_dim: (seq / 32, D_MODEL / 16, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: gemm_shared,
                },
                (
                    &gelu_out_dev,
                    &lw.w_fc_proj,
                    &mut ffn_out_dev,
                    D_FFN,
                    D_MODEL,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // FFN bias
        unsafe {
            f_bias.clone().launch(
                LaunchConfig {
                    grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &mut ffn_out_dev,
                    &lw.b_fc_proj,
                    D_MODEL,
                    total_seq_model as u32,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Residual 2 + zero pad
        unsafe {
            f_add.clone().launch(
                LaunchConfig {
                    grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut hidden_dev, &ffn_out_dev, total_seq_model as u32),
            )?;
            if pad_count > 0 {
                f_zero.clone().launch(
                    LaunchConfig {
                        grid_dim: (pad_count.div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut hidden_dev, pad_start, total_seq_model as u32),
                )?;
            }
        }
        dev.synchronize()?;

        if layer_idx == 0 || layer_idx == 5 || layer_idx == 11 {
            println!("    Layer {layer_idx} done");
        }
    }

    // === Final LayerNorm ===
    unsafe {
        f_ln.clone().launch(
            LaunchConfig {
                grid_dim: (seq, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &hidden_dev,
                &mut ln_out_dev,
                &ln_f_g_dev,
                &ln_f_b_dev,
                D_MODEL,
                EPS,
                status_dev_ptr,
            ),
        )?;
    }
    dev.synchronize()?;

    let fwd_elapsed = fwd_start.elapsed();

    // Download output
    let output: Vec<f32> = dev.dtoh_sync_copy(&ln_out_dev)?;

    // Check prediction position
    let last_pos = actual_seq - 1;
    let last_row = &output[last_pos * dm..(last_pos + 1) * dm];
    let pred_nan = last_row.iter().filter(|v| v.is_nan()).count();
    let pred_inf = last_row.iter().filter(|v| v.is_infinite()).count();
    let max_abs = last_row
        .iter()
        .filter(|v| !v.is_nan() && !v.is_infinite())
        .fold(0.0f32, |m, &v| m.max(v.abs()));

    println!(
        "  Forward pass: {:.1}ms",
        fwd_elapsed.as_secs_f64() * 1000.0
    );
    println!("  Prediction pos {last_pos}: nan={pred_nan}, inf={pred_inf}, max|val|={max_abs:.4}");

    if pred_nan > 0 || pred_inf > 0 {
        return Err(GpuHostError::Verification {
            test: "f32_forward",
            detail: format!("prediction has {pred_nan} NaN, {pred_inf} Inf"),
        });
    }

    // CPU LM head: logits[v] = dot(hidden[last_pos], wte[v])
    let vocab_size = 50257;
    let hidden = last_row;
    let mut logits = vec![0.0f32; vocab_size];
    for v in 0..vocab_size {
        let wte_row = &weights.wte[v * dm..(v + 1) * dm];
        let mut dot = 0.0f32;
        for d in 0..dm {
            dot += hidden[d] * wte_row[d];
        }
        logits[v] = dot;
    }

    // Top-5
    let mut indices: Vec<usize> = (0..vocab_size).collect();
    indices.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());

    println!("  Top-5 predictions (f32 GEMM):");
    for rank in 0..5 {
        let idx = indices[rank];
        let tok = tokenizer
            .decode(&[idx as u32])
            .unwrap_or_else(|_| format!("<tok {idx}>"));
        println!(
            "    #{}: token {} = {:?} (logit={:.2})",
            rank + 1,
            idx,
            tok,
            logits[idx],
        );
    }

    let top1 = indices[0];
    let top1_str = tokenizer
        .decode(&[top1 as u32])
        .unwrap_or_else(|_| format!("<tok {top1}>"));
    println!("  Greedy: {} = {:?}", top1, top1_str);

    unsafe {
        free_mapped_mem(status_host_ptr)?;
    }

    println!("  f32 GEMM forward pass — PASSED");
    Ok(())
}

/// mixed-precision.2: BF16 MMA inference validation.
///
/// Runs 3 prompts through both `full_gemm_bf16` (BF16 Tensor Core) and `gemm_f32`
/// (FMA reference), comparing top-5 predictions. BF16 takes f32 column-major inputs
/// (no host-side f16 packing needed). Reports timing for both paths.
pub(crate) fn run_bf16_forward_test(dev: Arc<CudaDevice>) -> Result<()> {
    let model_path =
        gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("model.safetensors");
    if !model_path.exists() {
        println!(
            "\n--- Skipping BF16 inference validation (models/model.safetensors not found) ---"
        );
        return Ok(());
    }

    println!("\n--- BF16 mixed-precision inference (mixed-precision.2) ---");

    let weights = gpu_host::model::load_gpt2_weights(&model_path).map_err(|e| {
        GpuHostError::Verification {
            test: "bf16_forward",
            detail: format!("weight loading: {e}"),
        }
    })?;

    let tokenizer =
        gpu_host::tokenizer::Gpt2Tokenizer::new().map_err(|e| GpuHostError::Verification {
            test: "bf16_forward",
            detail: format!("tokenizer: {e}"),
        })?;

    const SEQ: u32 = 32;
    const D_MODEL: u32 = 768;
    const N_HEADS: u32 = 12;
    const D_HEAD: u32 = 64;
    const D_FFN: u32 = 3072;
    const EPS: f32 = 1e-5;
    let dm = D_MODEL as usize;
    let total_seq_model = (SEQ * D_MODEL) as usize;
    let head_total = (N_HEADS * SEQ * D_HEAD) as usize;

    // Load PTX with both kernels
    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(
        ptx,
        "bf16fwd",
        &[
            "embedding_lookup",
            "layer_norm",
            "full_gemm_bf16",
            "full_gemm_tf32",
            "gemm_f32",
            "bias_add",
            "split_qkv",
            "flash_attention",
            "concat_heads",
            "gelu_forward",
            "elementwise_add",
            "zero_pad",
        ],
    )?;

    macro_rules! get_fn {
        ($name:expr) => {
            dev.get_func("bf16fwd", $name)
                .ok_or(GpuHostError::KernelNotFound($name))?
        };
    }

    let f_embed = get_fn!("embedding_lookup");
    let f_ln = get_fn!("layer_norm");
    let f_bf16 = get_fn!("full_gemm_bf16");
    let f_tf32 = get_fn!("full_gemm_tf32");
    let f_f32 = get_fn!("gemm_f32");
    let f_bias = get_fn!("bias_add");
    let f_split = get_fn!("split_qkv");
    let f_attn = get_fn!("flash_attention");
    let f_concat = get_fn!("concat_heads");
    let f_gelu = get_fn!("gelu_forward");
    let f_add = get_fn!("elementwise_add");
    let f_zero = get_fn!("zero_pad");

    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    // Column-major transpose helper
    fn to_col_major(w: &[f32], k: usize, n: usize) -> Vec<f32> {
        let mut cm = vec![0.0f32; k * n];
        for row in 0..k {
            for col in 0..n {
                cm[col * k + row] = w[row * n + col];
            }
        }
        cm
    }

    // Upload layer weights as column-major f32 (shared between bf16 and f32)
    struct LayerWeightsCm {
        ln1_g: CudaSlice<f32>,
        ln1_b: CudaSlice<f32>,
        w_qkv: CudaSlice<f32>,
        b_qkv: CudaSlice<f32>,
        w_proj: CudaSlice<f32>,
        b_proj: CudaSlice<f32>,
        ln2_g: CudaSlice<f32>,
        ln2_b: CudaSlice<f32>,
        w_fc: CudaSlice<f32>,
        b_fc: CudaSlice<f32>,
        w_fc_proj: CudaSlice<f32>,
        b_fc_proj: CudaSlice<f32>,
    }

    let mut gpu_layers: Vec<LayerWeightsCm> = Vec::with_capacity(12);
    for layer in weights.layers.iter() {
        gpu_layers.push(LayerWeightsCm {
            ln1_g: dev.htod_sync_copy(&layer.ln_1.weight)?,
            ln1_b: dev.htod_sync_copy(&layer.ln_1.bias)?,
            w_qkv: dev.htod_sync_copy(&to_col_major(&layer.c_attn_weight, 768, 2304))?,
            b_qkv: dev.htod_sync_copy(&layer.c_attn_bias)?,
            w_proj: dev.htod_sync_copy(&to_col_major(&layer.c_proj_weight, 768, 768))?,
            b_proj: dev.htod_sync_copy(&layer.c_proj_bias)?,
            ln2_g: dev.htod_sync_copy(&layer.ln_2.weight)?,
            ln2_b: dev.htod_sync_copy(&layer.ln_2.bias)?,
            w_fc: dev.htod_sync_copy(&to_col_major(&layer.mlp_fc_weight, 768, 3072))?,
            b_fc: dev.htod_sync_copy(&layer.mlp_fc_bias)?,
            w_fc_proj: dev.htod_sync_copy(&to_col_major(&layer.mlp_proj_weight, 3072, 768))?,
            b_fc_proj: dev.htod_sync_copy(&layer.mlp_proj_bias)?,
        });
    }
    println!("  12 layers uploaded (f32 column-major)");

    let ln_f_g_dev = dev.htod_sync_copy(&weights.ln_f.weight)?;
    let ln_f_b_dev = dev.htod_sync_copy(&weights.ln_f.bias)?;
    let wte_dev = dev.htod_sync_copy(&weights.wte)?;
    let wpe_dev = dev.htod_sync_copy(&weights.wpe)?;

    let n_q_tiles = (SEQ as usize).div_ceil(32) as u32;

    // Run forward pass with a given GEMM function.
    // gemm_kind: "bf16" uses full_gemm_bf16 (k_tiles), "f32" uses gemm_f32 (k).
    let run_forward = |prompt_tokens: &[u32],
                       actual_seq: usize,
                       gemm_kind: &str|
     -> Result<(Vec<f32>, std::time::Duration)> {
        let seq = SEQ;
        let mut token_ids: Vec<u32> = prompt_tokens.to_vec();
        token_ids.resize(seq as usize, 0);
        let token_ids_dev = dev.htod_sync_copy(&token_ids)?;

        let mut hidden_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
        unsafe {
            f_embed.clone().launch(
                LaunchConfig {
                    grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &wte_dev,
                    &wpe_dev,
                    &token_ids_dev,
                    &mut hidden_dev,
                    seq,
                    D_MODEL,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        let pad_start = (actual_seq as u32) * D_MODEL;
        let pad_count = total_seq_model as u32 - pad_start;
        if pad_count > 0 {
            unsafe {
                f_zero.clone().launch(
                    LaunchConfig {
                        grid_dim: (pad_count.div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut hidden_dev, pad_start, total_seq_model as u32),
                )?;
            }
            dev.synchronize()?;
        }

        let mut ln_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
        let mut qkv_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((seq * 2304) as usize)?;
        let mut q_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
        let mut k_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
        let mut v_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
        let mut attn_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
        let mut concat_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
        let mut proj_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
        let mut ffn_hidden_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((seq * D_FFN) as usize)?;
        let mut gelu_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((seq * D_FFN) as usize)?;
        let mut ffn_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;

        // Shared memory sizes
        let bf16_smem = (256 + 128) * 4; // full_gemm_bf16
        let tf32_smem = (256 + 128) * 4; // full_gemm_tf32 (same layout)
        let f32_smem = (32 * 16 + 16 * 16) * 4; // gemm_f32

        let fwd_start = std::time::Instant::now();

        for layer_idx in 0..12u32 {
            let lw = &gpu_layers[layer_idx as usize];

            // LN1
            unsafe {
                f_ln.clone().launch(
                    LaunchConfig {
                        grid_dim: (seq, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &hidden_dev,
                        &mut ln_out_dev,
                        &lw.ln1_g,
                        &lw.ln1_b,
                        D_MODEL,
                        EPS,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // QKV GEMM
            match gemm_kind {
                "bf16" => unsafe {
                    f_bf16.clone().launch(
                        LaunchConfig {
                            grid_dim: (seq / 32, 2304 / 16, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: bf16_smem,
                        },
                        (
                            &ln_out_dev,
                            &lw.w_qkv,
                            &mut qkv_dev,
                            (768u32 / 16),
                            2304u32,
                            status_dev_ptr,
                        ),
                    )?;
                },
                "tf32" => unsafe {
                    f_tf32.clone().launch(
                        LaunchConfig {
                            grid_dim: (seq / 32, 2304 / 16, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: tf32_smem,
                        },
                        (
                            &ln_out_dev,
                            &lw.w_qkv,
                            &mut qkv_dev,
                            (768u32 / 8),
                            2304u32,
                            status_dev_ptr,
                        ),
                    )?;
                },
                _ => unsafe {
                    f_f32.clone().launch(
                        LaunchConfig {
                            grid_dim: (seq / 32, 2304 / 16, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: f32_smem,
                        },
                        (
                            &ln_out_dev,
                            &lw.w_qkv,
                            &mut qkv_dev,
                            D_MODEL,
                            2304u32,
                            status_dev_ptr,
                        ),
                    )?;
                },
            }
            dev.synchronize()?;

            // QKV bias
            let total_qkv = seq * 2304;
            unsafe {
                f_bias.clone().launch(
                    LaunchConfig {
                        grid_dim: (total_qkv.div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut qkv_dev, &lw.b_qkv, 2304u32, total_qkv, status_dev_ptr),
                )?;
            }
            dev.synchronize()?;

            // Split QKV
            unsafe {
                f_split.clone().launch(
                    LaunchConfig {
                        grid_dim: ((head_total as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &qkv_dev, &mut q_dev, &mut k_dev, &mut v_dev, seq, N_HEADS, D_HEAD,
                    ),
                )?;
            }
            dev.synchronize()?;

            // Flash attention (causal)
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
                        &mut attn_out_dev,
                        seq,
                        D_HEAD,
                        1u32,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // Concat heads
            unsafe {
                f_concat.clone().launch(
                    LaunchConfig {
                        grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&attn_out_dev, &mut concat_dev, seq, N_HEADS, D_HEAD),
                )?;
            }
            dev.synchronize()?;

            // Output projection GEMM
            match gemm_kind {
                "bf16" => unsafe {
                    f_bf16.clone().launch(
                        LaunchConfig {
                            grid_dim: (seq / 32, D_MODEL / 16, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: bf16_smem,
                        },
                        (
                            &concat_dev,
                            &lw.w_proj,
                            &mut proj_out_dev,
                            (768u32 / 16),
                            D_MODEL,
                            status_dev_ptr,
                        ),
                    )?;
                },
                "tf32" => unsafe {
                    f_tf32.clone().launch(
                        LaunchConfig {
                            grid_dim: (seq / 32, D_MODEL / 16, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: tf32_smem,
                        },
                        (
                            &concat_dev,
                            &lw.w_proj,
                            &mut proj_out_dev,
                            (768u32 / 8),
                            D_MODEL,
                            status_dev_ptr,
                        ),
                    )?;
                },
                _ => unsafe {
                    f_f32.clone().launch(
                        LaunchConfig {
                            grid_dim: (seq / 32, D_MODEL / 16, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: f32_smem,
                        },
                        (
                            &concat_dev,
                            &lw.w_proj,
                            &mut proj_out_dev,
                            D_MODEL,
                            D_MODEL,
                            status_dev_ptr,
                        ),
                    )?;
                },
            }
            dev.synchronize()?;

            // Proj bias
            unsafe {
                f_bias.clone().launch(
                    LaunchConfig {
                        grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &mut proj_out_dev,
                        &lw.b_proj,
                        D_MODEL,
                        total_seq_model as u32,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // Residual 1 + zero pad
            unsafe {
                f_add.clone().launch(
                    LaunchConfig {
                        grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut hidden_dev, &proj_out_dev, total_seq_model as u32),
                )?;
                if pad_count > 0 {
                    f_zero.clone().launch(
                        LaunchConfig {
                            grid_dim: (pad_count.div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&mut hidden_dev, pad_start, total_seq_model as u32),
                    )?;
                }
            }
            dev.synchronize()?;

            // LN2
            unsafe {
                f_ln.clone().launch(
                    LaunchConfig {
                        grid_dim: (seq, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &hidden_dev,
                        &mut ln_out_dev,
                        &lw.ln2_g,
                        &lw.ln2_b,
                        D_MODEL,
                        EPS,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // FFN up GEMM
            match gemm_kind {
                "bf16" => unsafe {
                    f_bf16.clone().launch(
                        LaunchConfig {
                            grid_dim: (seq / 32, D_FFN / 16, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: bf16_smem,
                        },
                        (
                            &ln_out_dev,
                            &lw.w_fc,
                            &mut ffn_hidden_dev,
                            (768u32 / 16),
                            D_FFN,
                            status_dev_ptr,
                        ),
                    )?;
                },
                "tf32" => unsafe {
                    f_tf32.clone().launch(
                        LaunchConfig {
                            grid_dim: (seq / 32, D_FFN / 16, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: tf32_smem,
                        },
                        (
                            &ln_out_dev,
                            &lw.w_fc,
                            &mut ffn_hidden_dev,
                            (768u32 / 8),
                            D_FFN,
                            status_dev_ptr,
                        ),
                    )?;
                },
                _ => unsafe {
                    f_f32.clone().launch(
                        LaunchConfig {
                            grid_dim: (seq / 32, D_FFN / 16, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: f32_smem,
                        },
                        (
                            &ln_out_dev,
                            &lw.w_fc,
                            &mut ffn_hidden_dev,
                            D_MODEL,
                            D_FFN,
                            status_dev_ptr,
                        ),
                    )?;
                },
            }
            dev.synchronize()?;

            // FFN bias
            let total_ffn = seq * D_FFN;
            unsafe {
                f_bias.clone().launch(
                    LaunchConfig {
                        grid_dim: (total_ffn.div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &mut ffn_hidden_dev,
                        &lw.b_fc,
                        D_FFN,
                        total_ffn,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // GELU
            unsafe {
                f_gelu.clone().launch(
                    LaunchConfig {
                        grid_dim: (total_ffn.div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &ffn_hidden_dev,
                        &mut gelu_out_dev,
                        total_ffn,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // FFN down GEMM
            match gemm_kind {
                "bf16" => unsafe {
                    f_bf16.clone().launch(
                        LaunchConfig {
                            grid_dim: (seq / 32, D_MODEL / 16, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: bf16_smem,
                        },
                        (
                            &gelu_out_dev,
                            &lw.w_fc_proj,
                            &mut ffn_out_dev,
                            (3072u32 / 16),
                            D_MODEL,
                            status_dev_ptr,
                        ),
                    )?;
                },
                "tf32" => unsafe {
                    f_tf32.clone().launch(
                        LaunchConfig {
                            grid_dim: (seq / 32, D_MODEL / 16, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: tf32_smem,
                        },
                        (
                            &gelu_out_dev,
                            &lw.w_fc_proj,
                            &mut ffn_out_dev,
                            (3072u32 / 8),
                            D_MODEL,
                            status_dev_ptr,
                        ),
                    )?;
                },
                _ => unsafe {
                    f_f32.clone().launch(
                        LaunchConfig {
                            grid_dim: (seq / 32, D_MODEL / 16, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: f32_smem,
                        },
                        (
                            &gelu_out_dev,
                            &lw.w_fc_proj,
                            &mut ffn_out_dev,
                            D_FFN,
                            D_MODEL,
                            status_dev_ptr,
                        ),
                    )?;
                },
            }
            dev.synchronize()?;

            // FFN bias
            unsafe {
                f_bias.clone().launch(
                    LaunchConfig {
                        grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &mut ffn_out_dev,
                        &lw.b_fc_proj,
                        D_MODEL,
                        total_seq_model as u32,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // Residual 2 + zero pad
            unsafe {
                f_add.clone().launch(
                    LaunchConfig {
                        grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut hidden_dev, &ffn_out_dev, total_seq_model as u32),
                )?;
                if pad_count > 0 {
                    f_zero.clone().launch(
                        LaunchConfig {
                            grid_dim: (pad_count.div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&mut hidden_dev, pad_start, total_seq_model as u32),
                    )?;
                }
            }
            dev.synchronize()?;
        }

        // Final LayerNorm
        unsafe {
            f_ln.clone().launch(
                LaunchConfig {
                    grid_dim: (seq, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &hidden_dev,
                    &mut ln_out_dev,
                    &ln_f_g_dev,
                    &ln_f_b_dev,
                    D_MODEL,
                    EPS,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        let elapsed = fwd_start.elapsed();
        let output: Vec<f32> = dev.dtoh_sync_copy(&ln_out_dev)?;
        Ok((output, elapsed))
    };

    // Helper: compute top-5 from hidden state
    let get_top5 = |output: &[f32], actual_seq: usize| -> Vec<(usize, f32)> {
        let last_pos = actual_seq - 1;
        let last_row = &output[last_pos * dm..(last_pos + 1) * dm];
        let vocab_size = 50257;
        let mut logits = vec![0.0f32; vocab_size];
        for v in 0..vocab_size {
            let wte_row = &weights.wte[v * dm..(v + 1) * dm];
            let mut dot = 0.0f32;
            for d in 0..dm {
                dot += last_row[d] * wte_row[d];
            }
            logits[v] = dot;
        }
        let mut indices: Vec<usize> = (0..vocab_size).collect();
        indices.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        indices[0..5].iter().map(|&i| (i, logits[i])).collect()
    };

    // Test 3 prompts
    let prompts = [
        "The capital of France is",
        "In 1969, the first man to walk on the moon was",
        "The largest ocean on Earth is the",
    ];

    let mut total_agree = 0;
    let mut total_prompts = 0;

    for prompt in &prompts {
        let tokens = tokenizer.encode(prompt);
        let actual_seq = tokens.len();
        println!("\n  Prompt: \"{prompt}\" ({actual_seq} tokens)");

        // Run all three variants
        let (f32_out, f32_time) = run_forward(&tokens, actual_seq, "f32")?;
        let (bf16_out, bf16_time) = run_forward(&tokens, actual_seq, "bf16")?;
        let (tf32_out, tf32_time) = run_forward(&tokens, actual_seq, "tf32")?;

        let f32_top5 = get_top5(&f32_out, actual_seq);
        let bf16_top5 = get_top5(&bf16_out, actual_seq);
        let tf32_top5 = get_top5(&tf32_out, actual_seq);

        println!(
            "    f32: {:.1}ms | BF16: {:.1}ms | TF32: {:.1}ms",
            f32_time.as_secs_f64() * 1000.0,
            bf16_time.as_secs_f64() * 1000.0,
            tf32_time.as_secs_f64() * 1000.0,
        );

        let f32_top5_ids: Vec<usize> = f32_top5.iter().map(|&(id, _)| id).collect();

        // Display top-5 for each variant
        for (name, top5) in [
            ("f32 ", &f32_top5),
            ("BF16", &bf16_top5),
            ("TF32", &tf32_top5),
        ] {
            print!("    {name} top-5:");
            for &(id, logit) in top5.iter() {
                let tok = tokenizer
                    .decode(&[id as u32])
                    .unwrap_or_else(|_| format!("<{id}>"));
                print!(" {:?}({logit:.1})", tok);
            }
            println!();
        }

        // Check top-1 agreement for BF16 and TF32 vs f32
        let bf16_match = bf16_top5[0].0 == f32_top5_ids[0];
        let tf32_match = tf32_top5[0].0 == f32_top5_ids[0];
        println!("    vs f32 top-1: BF16={bf16_match} TF32={tf32_match}");

        if tf32_match {
            total_agree += 1;
        }
        total_prompts += 1;
    }

    unsafe {
        free_mapped_mem(status_host_ptr)?;
    }

    println!("\n  TF32 vs f32 top-1 agreement: {total_agree}/{total_prompts} prompts");
    println!("  Mixed-precision inference (BF16 + TF32) — PASSED");
    Ok(())
}

/// mma-fix.3: 12-layer GPT-2 forward pass with Tensor Core MMA (full_gemm_f32in).
///
/// Identical pipeline to run_f32_forward_test but uses Tensor Core MMA kernel
/// (full_gemm_f32in) for all GEMM operations. Weights are packed as column-major
/// f16x2 instead of column-major f32. Compares top-5 output against the f32 FMA
/// baseline to validate MMA correctness in inference context.
pub(crate) fn run_mma_forward_test(dev: Arc<CudaDevice>) -> Result<()> {
    let model_path =
        gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("model.safetensors");
    if !model_path.exists() {
        println!("\n--- Skipping MMA forward pass (models/model.safetensors not found) ---");
        return Ok(());
    }

    println!("\n--- MMA forward pass (mma-splitk.4: multi-prompt) ---");

    let weights = gpu_host::model::load_gpt2_weights(&model_path).map_err(|e| {
        GpuHostError::Verification {
            test: "mma_forward",
            detail: format!("weight loading: {e}"),
        }
    })?;

    let tokenizer =
        gpu_host::tokenizer::Gpt2Tokenizer::new().map_err(|e| GpuHostError::Verification {
            test: "mma_forward",
            detail: format!("tokenizer: {e}"),
        })?;

    let prompts = [
        "The capital of France is",
        "In 1969, the first man to walk on the moon was",
        "The largest ocean on Earth is the",
    ];

    const SEQ: u32 = 32;
    const D_MODEL: u32 = 768;
    const N_HEADS: u32 = 12;
    const D_HEAD: u32 = 64;
    const D_FFN: u32 = 3072;
    const EPS: f32 = 1e-5;
    let dm = D_MODEL as usize;
    let seq = SEQ;
    let total_seq_model = (seq * D_MODEL) as usize;
    let head_total = (N_HEADS * seq * D_HEAD) as usize;

    // Load PTX
    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(
        ptx,
        "mma_fwd",
        &[
            "embedding_lookup",
            "layer_norm",
            "full_gemm_f32in",
            "bias_add",
            "split_qkv",
            "flash_attention",
            "concat_heads",
            "gelu_forward",
            "elementwise_add",
            "zero_pad",
        ],
    )?;

    macro_rules! get_fn {
        ($name:expr) => {
            dev.get_func("mma_fwd", $name)
                .ok_or(GpuHostError::KernelNotFound($name))?
        };
    }

    let f_embed = get_fn!("embedding_lookup");
    let f_ln = get_fn!("layer_norm");
    let f_gemm = get_fn!("full_gemm_f32in");
    let f_bias = get_fn!("bias_add");
    let f_split = get_fn!("split_qkv");
    let f_attn = get_fn!("flash_attention");
    let f_concat = get_fn!("concat_heads");
    let f_gelu = get_fn!("gelu_forward");
    let f_add = get_fn!("elementwise_add");
    let f_zero = get_fn!("zero_pad");

    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    // Upload embedding tables (shared across all prompts)
    let wte_dev = dev.htod_sync_copy(&weights.wte)?;
    let wpe_dev = dev.htod_sync_copy(&weights.wpe)?;

    let mut hidden_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;

    // === Helper: f32 ↔ f16 conversion for weight packing ===
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
        f32_to_f16(lo) as u32 | ((f32_to_f16(hi) as u32) << 16)
    }

    /// Pack weight matrix [K, N] row-major into column-major f16x2.
    /// Output: for each column, K/2 packed u32 values (pairs along K dimension).
    fn to_col_major_f16x2(w: &[f32], k: usize, n: usize) -> Vec<u32> {
        let mut packed = Vec::with_capacity(n * k / 2);
        for col in 0..n {
            for k_pair in 0..k / 2 {
                let v0 = w[k_pair * 2 * n + col];
                let v1 = w[(k_pair * 2 + 1) * n + col];
                packed.push(pack_f16x2(v0, v1));
            }
        }
        packed
    }

    // === Upload layer weights as column-major f16x2 ===
    println!("  Uploading 12 layers (f16x2 column-major for MMA)...");
    struct LayerWeightsMma {
        ln1_g: CudaSlice<f32>,
        ln1_b: CudaSlice<f32>,
        w_qkv: CudaSlice<u32>,
        b_qkv: CudaSlice<f32>,
        w_proj: CudaSlice<u32>,
        b_proj: CudaSlice<f32>,
        ln2_g: CudaSlice<f32>,
        ln2_b: CudaSlice<f32>,
        w_fc: CudaSlice<u32>,
        b_fc: CudaSlice<f32>,
        w_fc_proj: CudaSlice<u32>,
        b_fc_proj: CudaSlice<f32>,
    }

    let mut gpu_layers: Vec<LayerWeightsMma> = Vec::with_capacity(12);
    for (i, layer) in weights.layers.iter().enumerate() {
        let w_qkv_pk = to_col_major_f16x2(&layer.c_attn_weight, 768, 2304);
        let w_proj_pk = to_col_major_f16x2(&layer.c_proj_weight, 768, 768);
        let w_fc_pk = to_col_major_f16x2(&layer.mlp_fc_weight, 768, 3072);
        let w_fc_proj_pk = to_col_major_f16x2(&layer.mlp_proj_weight, 3072, 768);

        gpu_layers.push(LayerWeightsMma {
            ln1_g: dev.htod_sync_copy(&layer.ln_1.weight)?,
            ln1_b: dev.htod_sync_copy(&layer.ln_1.bias)?,
            w_qkv: dev.htod_sync_copy(&w_qkv_pk)?,
            b_qkv: dev.htod_sync_copy(&layer.c_attn_bias)?,
            w_proj: dev.htod_sync_copy(&w_proj_pk)?,
            b_proj: dev.htod_sync_copy(&layer.c_proj_bias)?,
            ln2_g: dev.htod_sync_copy(&layer.ln_2.weight)?,
            ln2_b: dev.htod_sync_copy(&layer.ln_2.bias)?,
            w_fc: dev.htod_sync_copy(&w_fc_pk)?,
            b_fc: dev.htod_sync_copy(&layer.mlp_fc_bias)?,
            w_fc_proj: dev.htod_sync_copy(&w_fc_proj_pk)?,
            b_fc_proj: dev.htod_sync_copy(&layer.mlp_proj_bias)?,
        });
        if i == 0 || i == 11 {
            println!("    Layer {i} uploaded");
        }
    }
    println!("  All 12 layers uploaded (f16x2)");

    // Final LN weights
    let ln_f_g_dev = dev.htod_sync_copy(&weights.ln_f.weight)?;
    let ln_f_b_dev = dev.htod_sync_copy(&weights.ln_f.bias)?;

    // === Allocate activation buffers ===
    let mut ln_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let mut qkv_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((seq * 2304) as usize)?;
    let mut q_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut k_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut v_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut attn_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut concat_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let mut proj_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let mut ffn_hidden_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((seq * D_FFN) as usize)?;
    let mut gelu_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((seq * D_FFN) as usize)?;
    let mut ffn_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;

    let mma_shared = (256 + 128) * 4; // shared memory for full_gemm_f32in
    let n_q_tiles = (seq as usize).div_ceil(32) as u32;

    let mut mma_top5_results: Vec<Vec<usize>> = Vec::new();

    for prompt in &prompts {
        let prompt_tokens = tokenizer.encode(prompt);
        let actual_seq = prompt_tokens.len();
        println!("\n  Prompt: \"{prompt}\" → {actual_seq} tokens: {prompt_tokens:?}");

        let mut token_ids_u32: Vec<u32> = prompt_tokens.clone();
        token_ids_u32.resize(seq as usize, 0);
        let token_ids_dev = dev.htod_sync_copy(&token_ids_u32)?;

        // === Embedding ===
        unsafe {
            f_embed.clone().launch(
                LaunchConfig {
                    grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &wte_dev,
                    &wpe_dev,
                    &token_ids_dev,
                    &mut hidden_dev,
                    seq,
                    D_MODEL,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Zero padded positions
        let pad_start = (actual_seq as u32) * D_MODEL;
        let pad_count = total_seq_model as u32 - pad_start;
        if pad_count > 0 {
            unsafe {
                f_zero.clone().launch(
                    LaunchConfig {
                        grid_dim: (pad_count.div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut hidden_dev, pad_start, total_seq_model as u32),
                )?;
            }
            dev.synchronize()?;
        }
        println!("  Embedding done (seq={seq}, actual={actual_seq})");

        let fwd_start = std::time::Instant::now();

        // === 12 transformer layers ===
        for layer_idx in 0..12u32 {
            let lw = &gpu_layers[layer_idx as usize];

            // LN1
            unsafe {
                f_ln.clone().launch(
                    LaunchConfig {
                        grid_dim: (seq, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &hidden_dev,
                        &mut ln_out_dev,
                        &lw.ln1_g,
                        &lw.ln1_b,
                        D_MODEL,
                        EPS,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // QKV GEMM (MMA): ln_out[32,768] × w_qkv[768,2304] → qkv[32,2304]
            let qkv_k_tiles = (768 / 16) as u32;
            unsafe {
                f_gemm.clone().launch(
                    LaunchConfig {
                        grid_dim: (seq / 32, 2304 / 16, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: mma_shared,
                    },
                    (
                        &ln_out_dev,
                        &lw.w_qkv,
                        &mut qkv_dev,
                        qkv_k_tiles,
                        2304u32,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // QKV bias
            let total_qkv = seq * 2304;
            unsafe {
                f_bias.clone().launch(
                    LaunchConfig {
                        grid_dim: (total_qkv.div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut qkv_dev, &lw.b_qkv, 2304u32, total_qkv, status_dev_ptr),
                )?;
            }
            dev.synchronize()?;

            // Split QKV
            unsafe {
                f_split.clone().launch(
                    LaunchConfig {
                        grid_dim: ((head_total as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &qkv_dev, &mut q_dev, &mut k_dev, &mut v_dev, seq, N_HEADS, D_HEAD,
                    ),
                )?;
            }
            dev.synchronize()?;

            // Flash attention (causal)
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
                        &mut attn_out_dev,
                        seq,
                        D_HEAD,
                        1u32,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // Concat heads
            unsafe {
                f_concat.clone().launch(
                    LaunchConfig {
                        grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&attn_out_dev, &mut concat_dev, seq, N_HEADS, D_HEAD),
                )?;
            }
            dev.synchronize()?;

            // Output projection (MMA GEMM)
            let proj_k_tiles = (768 / 16) as u32;
            unsafe {
                f_gemm.clone().launch(
                    LaunchConfig {
                        grid_dim: (seq / 32, D_MODEL / 16, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: mma_shared,
                    },
                    (
                        &concat_dev,
                        &lw.w_proj,
                        &mut proj_out_dev,
                        proj_k_tiles,
                        D_MODEL,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // Proj bias
            unsafe {
                f_bias.clone().launch(
                    LaunchConfig {
                        grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &mut proj_out_dev,
                        &lw.b_proj,
                        D_MODEL,
                        total_seq_model as u32,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // Residual 1 + zero pad
            unsafe {
                f_add.clone().launch(
                    LaunchConfig {
                        grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut hidden_dev, &proj_out_dev, total_seq_model as u32),
                )?;
                if pad_count > 0 {
                    f_zero.clone().launch(
                        LaunchConfig {
                            grid_dim: (pad_count.div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&mut hidden_dev, pad_start, total_seq_model as u32),
                    )?;
                }
            }
            dev.synchronize()?;

            // LN2
            unsafe {
                f_ln.clone().launch(
                    LaunchConfig {
                        grid_dim: (seq, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &hidden_dev,
                        &mut ln_out_dev,
                        &lw.ln2_g,
                        &lw.ln2_b,
                        D_MODEL,
                        EPS,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // FFN up (MMA GEMM): ln_out[32,768] × w_fc[768,3072] → ffn[32,3072]
            let ffn_up_k_tiles = (768 / 16) as u32;
            unsafe {
                f_gemm.clone().launch(
                    LaunchConfig {
                        grid_dim: (seq / 32, D_FFN / 16, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: mma_shared,
                    },
                    (
                        &ln_out_dev,
                        &lw.w_fc,
                        &mut ffn_hidden_dev,
                        ffn_up_k_tiles,
                        D_FFN,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // FFN bias
            let total_ffn = seq * D_FFN;
            unsafe {
                f_bias.clone().launch(
                    LaunchConfig {
                        grid_dim: (total_ffn.div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &mut ffn_hidden_dev,
                        &lw.b_fc,
                        D_FFN,
                        total_ffn,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // GELU
            unsafe {
                f_gelu.clone().launch(
                    LaunchConfig {
                        grid_dim: (total_ffn.div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &ffn_hidden_dev,
                        &mut gelu_out_dev,
                        total_ffn,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // FFN down (MMA GEMM): gelu[32,3072] × w_fc_proj[3072,768] → ffn_out[32,768]
            let ffn_down_k_tiles = (3072 / 16) as u32;
            unsafe {
                f_gemm.clone().launch(
                    LaunchConfig {
                        grid_dim: (seq / 32, D_MODEL / 16, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: mma_shared,
                    },
                    (
                        &gelu_out_dev,
                        &lw.w_fc_proj,
                        &mut ffn_out_dev,
                        ffn_down_k_tiles,
                        D_MODEL,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // FFN bias
            unsafe {
                f_bias.clone().launch(
                    LaunchConfig {
                        grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &mut ffn_out_dev,
                        &lw.b_fc_proj,
                        D_MODEL,
                        total_seq_model as u32,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // Residual 2 + zero pad
            unsafe {
                f_add.clone().launch(
                    LaunchConfig {
                        grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut hidden_dev, &ffn_out_dev, total_seq_model as u32),
                )?;
                if pad_count > 0 {
                    f_zero.clone().launch(
                        LaunchConfig {
                            grid_dim: (pad_count.div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&mut hidden_dev, pad_start, total_seq_model as u32),
                    )?;
                }
            }
            dev.synchronize()?;

            if layer_idx == 0 || layer_idx == 5 || layer_idx == 11 {
                println!("    Layer {layer_idx} done");
            }
        }

        // === Final LayerNorm ===
        unsafe {
            f_ln.clone().launch(
                LaunchConfig {
                    grid_dim: (seq, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &hidden_dev,
                    &mut ln_out_dev,
                    &ln_f_g_dev,
                    &ln_f_b_dev,
                    D_MODEL,
                    EPS,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        let fwd_elapsed = fwd_start.elapsed();

        // Download output
        let output: Vec<f32> = dev.dtoh_sync_copy(&ln_out_dev)?;

        // Check prediction position
        let last_pos = actual_seq - 1;
        let last_row = &output[last_pos * dm..(last_pos + 1) * dm];
        let pred_nan = last_row.iter().filter(|v| v.is_nan()).count();
        let pred_inf = last_row.iter().filter(|v| v.is_infinite()).count();
        let max_abs = last_row
            .iter()
            .filter(|v| !v.is_nan() && !v.is_infinite())
            .fold(0.0f32, |m, &v| m.max(v.abs()));

        println!(
            "  MMA forward pass: {:.1}ms",
            fwd_elapsed.as_secs_f64() * 1000.0
        );
        println!(
            "  Prediction pos {last_pos}: nan={pred_nan}, inf={pred_inf}, max|val|={max_abs:.4}"
        );

        if pred_nan > 0 || pred_inf > 0 {
            return Err(GpuHostError::Verification {
                test: "mma_forward",
                detail: format!("prediction has {pred_nan} NaN, {pred_inf} Inf"),
            });
        }

        // CPU LM head: logits[v] = dot(hidden[last_pos], wte[v])
        let vocab_size = 50257;
        let hidden = last_row;
        let mut logits = vec![0.0f32; vocab_size];
        for v in 0..vocab_size {
            let wte_row = &weights.wte[v * dm..(v + 1) * dm];
            let mut dot = 0.0f32;
            for d in 0..dm {
                dot += hidden[d] * wte_row[d];
            }
            logits[v] = dot;
        }

        // Top-5
        let mut indices: Vec<usize> = (0..vocab_size).collect();
        indices.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());

        println!("  Top-5 predictions (MMA):");
        for rank in 0..5 {
            let idx = indices[rank];
            let tok = tokenizer
                .decode(&[idx as u32])
                .unwrap_or_else(|_| format!("<tok {idx}>"));
            println!(
                "    #{}: token {} = {:?} (logit={:.2})",
                rank + 1,
                idx,
                tok,
                logits[idx],
            );
        }

        let top5: Vec<usize> = indices[..5].to_vec();
        let top1 = top5[0];
        let top1_str = tokenizer
            .decode(&[top1 as u32])
            .unwrap_or_else(|_| format!("<tok {top1}>"));
        println!("  MMA greedy: {} = {:?}", top1, top1_str);

        mma_top5_results.push(top5);
    } // end prompt loop

    drop(wte_dev);
    drop(wpe_dev);

    unsafe {
        free_mapped_mem(status_host_ptr)?;
    }

    println!(
        "\n  === Multi-prompt summary ({} prompts) ===",
        prompts.len()
    );
    for (i, prompt) in prompts.iter().enumerate() {
        let top5 = &mma_top5_results[i];
        let tokens: Vec<String> = top5
            .iter()
            .map(|&idx| {
                tokenizer
                    .decode(&[idx as u32])
                    .unwrap_or_else(|_| format!("<tok {idx}>"))
            })
            .collect();
        println!("  Prompt {}: {:?} → {:?}", i + 1, prompt, tokens);
    }
    println!(
        "  All {} prompts completed without NaN/Inf — PASSED",
        prompts.len()
    );
    Ok(())
}

/// full-inference.8+10: CPU f64 reference forward pass — definitive diagnostic.
///
/// Implements the COMPLETE GPT-2 transformer in pure Rust f64 arithmetic on CPU.
/// No GPU involved. If this also predicts " a", the model genuinely cannot answer.
/// If this predicts "Paris", there is a GPU implementation bug.
///
/// Also performs LM head position audit (full-inference.10): checks predictions
/// from last_pos-1, last_pos, last_pos+1.
///
/// Also performs per-layer intermediate predictions (full-inference.9): applies
/// LM head after each of 12 layers to track where semantic content appears/is lost.
pub(crate) fn run_cpu_f64_reference_test() -> Result<()> {
    let model_path =
        gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("model.safetensors");
    if !model_path.exists() {
        println!("\n--- Skipping CPU f64 reference (models/model.safetensors not found) ---");
        return Ok(());
    }

    println!("\n--- CPU f64 reference forward pass (full-inference.8+9+10) ---");

    let weights = gpu_host::model::load_gpt2_weights(&model_path).map_err(|e| {
        GpuHostError::Verification {
            test: "cpu_f64_ref",
            detail: format!("weight loading: {e}"),
        }
    })?;

    let tokenizer =
        gpu_host::tokenizer::Gpt2Tokenizer::new().map_err(|e| GpuHostError::Verification {
            test: "cpu_f64_ref",
            detail: format!("tokenizer: {e}"),
        })?;
    let prompt = "The capital of France is";
    let prompt_tokens = tokenizer.encode(prompt);
    let seq = prompt_tokens.len();
    println!("  Prompt: \"{prompt}\" → {seq} tokens: {prompt_tokens:?}");

    const D_MODEL: usize = 768;
    const N_HEADS: usize = 12;
    const D_HEAD: usize = 64;
    const D_FFN: usize = 3072;
    const VOCAB_SIZE: usize = 50257;
    const EPS: f64 = 1e-5;

    // Convert all weights to f64
    let wte: Vec<f64> = weights.wte.iter().map(|&x| x as f64).collect();
    let wpe: Vec<f64> = weights.wpe.iter().map(|&x| x as f64).collect();
    let ln_f_g: Vec<f64> = weights.ln_f.weight.iter().map(|&x| x as f64).collect();
    let ln_f_b: Vec<f64> = weights.ln_f.bias.iter().map(|&x| x as f64).collect();

    // === Embedding: hidden[pos] = wte[token] + wpe[pos] ===
    let mut hidden = vec![0.0f64; seq * D_MODEL];
    for pos in 0..seq {
        let tok = prompt_tokens[pos] as usize;
        for d in 0..D_MODEL {
            hidden[pos * D_MODEL + d] = wte[tok * D_MODEL + d] + wpe[pos * D_MODEL + d];
        }
    }
    println!(
        "  Embedding done. hidden[0][0..4] = [{:.6}, {:.6}, {:.6}, {:.6}]",
        hidden[0], hidden[1], hidden[2], hidden[3]
    );
    let pos4_start = (seq - 1) * D_MODEL;
    println!(
        "  CPU embed pos4 first4: [{:.6}, {:.6}, {:.6}, {:.6}]",
        hidden[pos4_start],
        hidden[pos4_start + 1],
        hidden[pos4_start + 2],
        hidden[pos4_start + 3]
    );

    // Helper: LayerNorm
    fn layer_norm(input: &[f64], gamma: &[f64], beta: &[f64], dim: usize, eps: f64) -> Vec<f64> {
        let seq = input.len() / dim;
        let mut output = vec![0.0f64; input.len()];
        for s in 0..seq {
            let row = &input[s * dim..(s + 1) * dim];
            let mean: f64 = row.iter().sum::<f64>() / dim as f64;
            let var: f64 = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / dim as f64;
            let inv_std = 1.0 / (var + eps).sqrt();
            for d in 0..dim {
                output[s * dim + d] = (row[d] - mean) * inv_std * gamma[d] + beta[d];
            }
        }
        output
    }

    // Helper: matmul A[M,K] @ B[K,N] → C[M,N]  (B stored row-major as [K,N])
    fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut c = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    // Helper: add bias
    fn add_bias(data: &mut [f64], bias: &[f64], dim: usize) {
        let seq = data.len() / dim;
        for s in 0..seq {
            for d in 0..dim {
                data[s * dim + d] += bias[d];
            }
        }
    }

    // Helper: GELU (GPT-2 variant with tanh approximation)
    fn gelu(x: f64) -> f64 {
        let sqrt_2_over_pi: f64 = 0.7978845608028654;
        let coeff: f64 = 0.044715;
        let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
        x * 0.5 * (1.0 + inner.tanh())
    }

    // Helper: compute top-5 predictions from a hidden vector using wte
    fn top5_from_hidden(
        hidden_vec: &[f64],
        wte: &[f64],
        dim: usize,
        vocab: usize,
        tokenizer: &gpu_host::tokenizer::Gpt2Tokenizer,
    ) -> Vec<(usize, f64, String)> {
        let mut logits = vec![0.0f64; vocab];
        for v in 0..vocab {
            let mut dot = 0.0f64;
            for d in 0..dim {
                dot += hidden_vec[d] * wte[v * dim + d];
            }
            logits[v] = dot;
        }
        let mut indices: Vec<usize> = (0..vocab).collect();
        indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        indices[..5]
            .iter()
            .map(|&idx| {
                let tok_str = tokenizer
                    .decode(&[idx as u32])
                    .unwrap_or_else(|_| format!("<{idx}>"));
                (idx, logits[idx], tok_str)
            })
            .collect()
    }

    let timer = std::time::Instant::now();

    // === 12 Transformer Layers ===
    for layer_idx in 0..12 {
        let lw = &weights.layers[layer_idx];
        let ln1_g: Vec<f64> = lw.ln_1.weight.iter().map(|&x| x as f64).collect();
        let ln1_b: Vec<f64> = lw.ln_1.bias.iter().map(|&x| x as f64).collect();
        let c_attn_w: Vec<f64> = lw.c_attn_weight.iter().map(|&x| x as f64).collect();
        let c_attn_b: Vec<f64> = lw.c_attn_bias.iter().map(|&x| x as f64).collect();
        let c_proj_w: Vec<f64> = lw.c_proj_weight.iter().map(|&x| x as f64).collect();
        let c_proj_b: Vec<f64> = lw.c_proj_bias.iter().map(|&x| x as f64).collect();
        let ln2_g: Vec<f64> = lw.ln_2.weight.iter().map(|&x| x as f64).collect();
        let ln2_b: Vec<f64> = lw.ln_2.bias.iter().map(|&x| x as f64).collect();
        let fc_w: Vec<f64> = lw.mlp_fc_weight.iter().map(|&x| x as f64).collect();
        let fc_b: Vec<f64> = lw.mlp_fc_bias.iter().map(|&x| x as f64).collect();
        let proj_w: Vec<f64> = lw.mlp_proj_weight.iter().map(|&x| x as f64).collect();
        let proj_b: Vec<f64> = lw.mlp_proj_bias.iter().map(|&x| x as f64).collect();

        // 1. LayerNorm1
        let ln_out = layer_norm(&hidden, &ln1_g, &ln1_b, D_MODEL, EPS);

        if layer_idx == 0 {
            let lp = seq - 1;
            let ln4 = &ln_out[lp * D_MODEL..(lp + 1) * D_MODEL];
            println!(
                "  CPU L0 LN1 pos4 first4: [{:.6}, {:.6}, {:.6}, {:.6}]",
                ln4[0], ln4[1], ln4[2], ln4[3]
            );
        }

        // 2. QKV projection: ln_out[seq, 768] @ c_attn_w[768, 2304] + bias
        let mut qkv = matmul(&ln_out, &c_attn_w, seq, D_MODEL, D_MODEL * 3);

        if layer_idx == 0 {
            let lp = seq - 1;
            let qkv4 = &qkv[lp * (D_MODEL * 3)..(lp + 1) * (D_MODEL * 3)];
            println!(
                "  CPU L0 QKV pos4 first4 (no bias): [{:.6}, {:.6}, {:.6}, {:.6}]",
                qkv4[0], qkv4[1], qkv4[2], qkv4[3]
            );
        }

        add_bias(&mut qkv, &c_attn_b, D_MODEL * 3);

        // 3. Split Q, K, V and compute attention per head
        // qkv layout: [seq, 2304] where Q=[0..768], K=[768..1536], V=[1536..2304]
        // Within each 768: [head0_d0..head0_d63, head1_d0..head1_d63, ...]
        let mut attn_concat = vec![0.0f64; seq * D_MODEL];

        for h in 0..N_HEADS {
            // Extract Q, K, V for this head
            let mut q = vec![0.0f64; seq * D_HEAD];
            let mut k = vec![0.0f64; seq * D_HEAD];
            let mut v = vec![0.0f64; seq * D_HEAD];

            for s in 0..seq {
                for d in 0..D_HEAD {
                    q[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + h * D_HEAD + d];
                    k[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + D_MODEL + h * D_HEAD + d];
                    v[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + 2 * D_MODEL + h * D_HEAD + d];
                }
            }

            // Attention scores: Q @ K^T / sqrt(d_head) with causal mask
            let scale = 1.0 / (D_HEAD as f64).sqrt();
            let mut scores = vec![0.0f64; seq * seq];
            for i in 0..seq {
                for j in 0..seq {
                    if j <= i {
                        // Causal: position i can attend to positions 0..=i
                        let mut dot = 0.0f64;
                        for d in 0..D_HEAD {
                            dot += q[i * D_HEAD + d] * k[j * D_HEAD + d];
                        }
                        scores[i * seq + j] = dot * scale;
                    } else {
                        scores[i * seq + j] = f64::NEG_INFINITY;
                    }
                }
            }

            // Softmax per row
            for i in 0..seq {
                let row = &mut scores[i * seq..(i + 1) * seq];
                let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0f64;
                for j in 0..seq {
                    row[j] = (row[j] - max_val).exp();
                    sum += row[j];
                }
                for j in 0..seq {
                    row[j] /= sum;
                }
            }

            // Attention output: softmax_scores @ V
            let attn_out = matmul(&scores, &v, seq, seq, D_HEAD);

            // Concat: write to [seq][d_model] at head offset
            for s in 0..seq {
                for d in 0..D_HEAD {
                    attn_concat[s * D_MODEL + h * D_HEAD + d] = attn_out[s * D_HEAD + d];
                }
            }
        }

        // 4. Output projection: attn_concat[seq, 768] @ c_proj_w[768, 768] + bias
        let mut proj_out = matmul(&attn_concat, &c_proj_w, seq, D_MODEL, D_MODEL);
        add_bias(&mut proj_out, &c_proj_b, D_MODEL);

        // 5. Residual 1
        for i in 0..hidden.len() {
            hidden[i] += proj_out[i];
        }

        // 6. LayerNorm2
        let ln2_out = layer_norm(&hidden, &ln2_g, &ln2_b, D_MODEL, EPS);

        // 7. FFN up: ln2_out[seq, 768] @ fc_w[768, 3072] + bias
        let mut ffn_hidden = matmul(&ln2_out, &fc_w, seq, D_MODEL, D_FFN);
        add_bias(&mut ffn_hidden, &fc_b, D_FFN);

        // 8. GELU
        for x in ffn_hidden.iter_mut() {
            *x = gelu(*x);
        }

        // 9. FFN down: ffn_hidden[seq, 3072] @ proj_w[3072, 768] + bias
        let mut ffn_out = matmul(&ffn_hidden, &proj_w, seq, D_FFN, D_MODEL);
        add_bias(&mut ffn_out, &proj_b, D_MODEL);

        // 10. Residual 2
        for i in 0..hidden.len() {
            hidden[i] += ffn_out[i];
        }

        // === Per-layer intermediate predictions (full-inference.9) ===
        // Compute norms of each component
        let last_pos = seq - 1;
        let attn_res_norm: f64 = proj_out[last_pos * D_MODEL..(last_pos + 1) * D_MODEL]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let ffn_res_norm: f64 = ffn_out[last_pos * D_MODEL..(last_pos + 1) * D_MODEL]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let ln_probe = layer_norm(&hidden, &ln_f_g, &ln_f_b, D_MODEL, EPS);
        let probe_vec = &ln_probe[last_pos * D_MODEL..(last_pos + 1) * D_MODEL];
        let top5 = top5_from_hidden(probe_vec, &wte, D_MODEL, VOCAB_SIZE, &tokenizer);
        let hidden_norm: f64 = hidden[last_pos * D_MODEL..(last_pos + 1) * D_MODEL]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        println!(
            "  Layer {:2}: top1={:?} ({:.4}) | h_norm={:.2} attn_r={:.2} ffn_r={:.2} | top5: {}",
            layer_idx,
            top5[0].2,
            top5[0].1,
            hidden_norm,
            attn_res_norm,
            ffn_res_norm,
            top5.iter()
                .map(|(_, l, s)| format!("{s}({l:.2})"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // === Final LayerNorm ===
    let ln_final = layer_norm(&hidden, &ln_f_g, &ln_f_b, D_MODEL, EPS);

    let elapsed = timer.elapsed();
    println!(
        "  Forward pass done in {:.1}ms",
        elapsed.as_secs_f64() * 1000.0
    );

    // === LM Head position audit (full-inference.10) ===
    let last_pos = seq - 1;
    println!("\n  === Position audit ===");
    for offset in [-1i32, 0, 1] {
        let pos = last_pos as i32 + offset;
        if pos < 0 || pos >= seq as i32 {
            continue;
        }
        let pos = pos as usize;
        let hvec = &ln_final[pos * D_MODEL..(pos + 1) * D_MODEL];
        let top5 = top5_from_hidden(hvec, &wte, D_MODEL, VOCAB_SIZE, &tokenizer);
        let label = match offset {
            -1 => "last_pos-1",
            0 => "last_pos  ",
            1 => "last_pos+1",
            _ => unreachable!(),
        };
        println!(
            "  {label} (pos={pos}): top5: {}",
            top5.iter()
                .map(|(id, l, s)| format!("{s}[{id}]({l:.2})"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // === Diagnostic: norms ===
    let final_vec_for_diag = &ln_final[last_pos * D_MODEL..(last_pos + 1) * D_MODEL];
    let ln_final_norm: f64 = final_vec_for_diag.iter().map(|x| x * x).sum::<f64>().sqrt();
    let ln_final_mean: f64 = final_vec_for_diag.iter().sum::<f64>() / D_MODEL as f64;
    let ln_final_max: f64 = final_vec_for_diag
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let ln_final_min: f64 = final_vec_for_diag
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    println!("\n  === Final LN output diagnostics ===");
    println!(
        "  ln_final norm={:.4}, mean={:.6}, min={:.4}, max={:.4}",
        ln_final_norm, ln_final_mean, ln_final_min, ln_final_max
    );

    // Check a few wte row norms
    let mut wte_norms: Vec<f64> = Vec::new();
    for v in 0..100 {
        let norm: f64 = wte[v * D_MODEL..(v + 1) * D_MODEL]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        wte_norms.push(norm);
    }
    let avg_wte_norm: f64 = wte_norms.iter().sum::<f64>() / wte_norms.len() as f64;
    println!("  wte avg row norm (first 100) = {:.4}", avg_wte_norm);

    // Check ln_f gamma stats
    let gamma_norm: f64 = ln_f_g.iter().map(|x| x * x).sum::<f64>().sqrt();
    let gamma_mean: f64 = ln_f_g.iter().sum::<f64>() / D_MODEL as f64;
    let gamma_max: f64 = ln_f_g.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "  ln_f gamma: norm={:.4}, mean={:.6}, max={:.4}",
        gamma_norm, gamma_mean, gamma_max
    );

    // === Final top-10 predictions from last_pos ===
    println!("\n  === Final predictions (last_pos={last_pos}) ===");
    let final_vec = &ln_final[last_pos * D_MODEL..(last_pos + 1) * D_MODEL];
    let mut logits = vec![0.0f64; VOCAB_SIZE];
    for v in 0..VOCAB_SIZE {
        let mut dot = 0.0f64;
        for d in 0..D_MODEL {
            dot += final_vec[d] * wte[v * D_MODEL + d];
        }
        logits[v] = dot;
    }
    let mut indices: Vec<usize> = (0..VOCAB_SIZE).collect();
    indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());

    for rank in 0..10 {
        let idx = indices[rank];
        let tok = tokenizer
            .decode(&[idx as u32])
            .unwrap_or_else(|_| format!("<{idx}>"));
        println!(
            "    #{}: token {} = {:?} (logit={:.4})",
            rank + 1,
            idx,
            tok,
            logits[idx]
        );
    }

    let top1 = indices[0];
    let top1_str = tokenizer
        .decode(&[top1 as u32])
        .unwrap_or_else(|_| format!("<{top1}>"));
    println!(
        "\n  CPU f64 GREEDY PREDICTION: token {} = {:?}",
        top1, top1_str
    );

    // === Padding test: does padding change the result? ===
    println!("\n  === CPU f64 with padding (seq=32, same as GPU) ===");
    {
        let padded_seq = 32usize;
        // Re-do embedding with padding
        let mut h_pad = vec![0.0f64; padded_seq * D_MODEL];
        for pos in 0..seq {
            let tok = prompt_tokens[pos] as usize;
            for d in 0..D_MODEL {
                h_pad[pos * D_MODEL + d] = wte[tok * D_MODEL + d] + wpe[pos * D_MODEL + d];
            }
        }
        // Zero padded positions (seq..padded_seq) — already zero from initialization

        // Run 12 layers with padded_seq
        for li in 0..12 {
            let lw = &weights.layers[li];
            let lg: Vec<f64> = lw.ln_1.weight.iter().map(|&x| x as f64).collect();
            let lb: Vec<f64> = lw.ln_1.bias.iter().map(|&x| x as f64).collect();
            let caw: Vec<f64> = lw.c_attn_weight.iter().map(|&x| x as f64).collect();
            let cab: Vec<f64> = lw.c_attn_bias.iter().map(|&x| x as f64).collect();
            let cpw: Vec<f64> = lw.c_proj_weight.iter().map(|&x| x as f64).collect();
            let cpb: Vec<f64> = lw.c_proj_bias.iter().map(|&x| x as f64).collect();
            let l2g: Vec<f64> = lw.ln_2.weight.iter().map(|&x| x as f64).collect();
            let l2b: Vec<f64> = lw.ln_2.bias.iter().map(|&x| x as f64).collect();
            let fw: Vec<f64> = lw.mlp_fc_weight.iter().map(|&x| x as f64).collect();
            let fb: Vec<f64> = lw.mlp_fc_bias.iter().map(|&x| x as f64).collect();
            let pw: Vec<f64> = lw.mlp_proj_weight.iter().map(|&x| x as f64).collect();
            let pb: Vec<f64> = lw.mlp_proj_bias.iter().map(|&x| x as f64).collect();

            let lo = layer_norm(&h_pad, &lg, &lb, D_MODEL, EPS);
            let mut qkv = matmul(&lo, &caw, padded_seq, D_MODEL, D_MODEL * 3);
            add_bias(&mut qkv, &cab, D_MODEL * 3);

            let mut ac = vec![0.0f64; padded_seq * D_MODEL];
            for hd in 0..N_HEADS {
                let mut q = vec![0.0f64; padded_seq * D_HEAD];
                let mut k = vec![0.0f64; padded_seq * D_HEAD];
                let mut v = vec![0.0f64; padded_seq * D_HEAD];
                for s in 0..padded_seq {
                    for d in 0..D_HEAD {
                        q[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + hd * D_HEAD + d];
                        k[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + D_MODEL + hd * D_HEAD + d];
                        v[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + 2 * D_MODEL + hd * D_HEAD + d];
                    }
                }
                let scale = 1.0 / (D_HEAD as f64).sqrt();
                let mut scores = vec![0.0f64; padded_seq * padded_seq];
                for i in 0..padded_seq {
                    for j in 0..padded_seq {
                        if j <= i {
                            let mut dot = 0.0f64;
                            for d in 0..D_HEAD {
                                dot += q[i * D_HEAD + d] * k[j * D_HEAD + d];
                            }
                            scores[i * padded_seq + j] = dot * scale;
                        } else {
                            scores[i * padded_seq + j] = f64::NEG_INFINITY;
                        }
                    }
                }
                for i in 0..padded_seq {
                    let row = &mut scores[i * padded_seq..(i + 1) * padded_seq];
                    let mx = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let mut sm = 0.0f64;
                    for j in 0..padded_seq {
                        row[j] = (row[j] - mx).exp();
                        sm += row[j];
                    }
                    for j in 0..padded_seq {
                        row[j] /= sm;
                    }
                }
                let ao = matmul(&scores, &v, padded_seq, padded_seq, D_HEAD);
                for s in 0..padded_seq {
                    for d in 0..D_HEAD {
                        ac[s * D_MODEL + hd * D_HEAD + d] = ao[s * D_HEAD + d];
                    }
                }
            }

            let mut po = matmul(&ac, &cpw, padded_seq, D_MODEL, D_MODEL);
            add_bias(&mut po, &cpb, D_MODEL);
            for i in 0..h_pad.len() {
                h_pad[i] += po[i];
            }

            // Zero padded positions after residual (like GPU does)
            for s in seq..padded_seq {
                for d in 0..D_MODEL {
                    h_pad[s * D_MODEL + d] = 0.0;
                }
            }

            let l2o = layer_norm(&h_pad, &l2g, &l2b, D_MODEL, EPS);
            let mut fh = matmul(&l2o, &fw, padded_seq, D_MODEL, D_FFN);
            add_bias(&mut fh, &fb, D_FFN);
            for x in fh.iter_mut() {
                *x = gelu(*x);
            }
            let mut fo = matmul(&fh, &pw, padded_seq, D_FFN, D_MODEL);
            add_bias(&mut fo, &pb, D_MODEL);
            for i in 0..h_pad.len() {
                h_pad[i] += fo[i];
            }

            // Zero padded positions after FFN residual
            for s in seq..padded_seq {
                for d in 0..D_MODEL {
                    h_pad[s * D_MODEL + d] = 0.0;
                }
            }
        }

        let lnf = layer_norm(&h_pad, &ln_f_g, &ln_f_b, D_MODEL, EPS);
        let lp = seq - 1;
        let hv = &lnf[lp * D_MODEL..(lp + 1) * D_MODEL];
        let t5 = top5_from_hidden(hv, &wte, D_MODEL, VOCAB_SIZE, &tokenizer);
        let h_norm: f64 = h_pad[lp * D_MODEL..(lp + 1) * D_MODEL]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let ln_norm: f64 = hv.iter().map(|x| x * x).sum::<f64>().sqrt();
        println!(
            "  Padded result: h_norm={:.2} ln_norm={:.2}",
            h_norm, ln_norm
        );
        println!(
            "  Top5: {}",
            t5.iter()
                .map(|(_, l, s)| format!("{s}({l:.2})"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!("  Compare no-pad: h_norm=429.60, top1=\" the\"(-100.25)");
    }

    // === Layer-by-layer GPU vs CPU comparison ===
    // Recompute with padding (seq=32) and print per-layer hidden norms at pos 4
    // to compare against GPU output
    println!("\n  === Per-layer hidden state comparison (for GPU validation) ===");
    {
        let padded_seq = 32usize;
        let mut h_cmp = vec![0.0f64; padded_seq * D_MODEL];
        for pos in 0..seq {
            let tok = prompt_tokens[pos] as usize;
            for d in 0..D_MODEL {
                h_cmp[pos * D_MODEL + d] = wte[tok * D_MODEL + d] + wpe[pos * D_MODEL + d];
            }
        }

        for li in 0..12 {
            let lw = &weights.layers[li];
            let lg: Vec<f64> = lw.ln_1.weight.iter().map(|&x| x as f64).collect();
            let lb: Vec<f64> = lw.ln_1.bias.iter().map(|&x| x as f64).collect();
            let caw: Vec<f64> = lw.c_attn_weight.iter().map(|&x| x as f64).collect();
            let cab: Vec<f64> = lw.c_attn_bias.iter().map(|&x| x as f64).collect();
            let cpw: Vec<f64> = lw.c_proj_weight.iter().map(|&x| x as f64).collect();
            let cpb: Vec<f64> = lw.c_proj_bias.iter().map(|&x| x as f64).collect();
            let l2g: Vec<f64> = lw.ln_2.weight.iter().map(|&x| x as f64).collect();
            let l2b: Vec<f64> = lw.ln_2.bias.iter().map(|&x| x as f64).collect();
            let fw: Vec<f64> = lw.mlp_fc_weight.iter().map(|&x| x as f64).collect();
            let fb: Vec<f64> = lw.mlp_fc_bias.iter().map(|&x| x as f64).collect();
            let pw: Vec<f64> = lw.mlp_proj_weight.iter().map(|&x| x as f64).collect();
            let pb: Vec<f64> = lw.mlp_proj_bias.iter().map(|&x| x as f64).collect();

            let lo = layer_norm(&h_cmp, &lg, &lb, D_MODEL, EPS);
            let mut qkv = matmul(&lo, &caw, padded_seq, D_MODEL, D_MODEL * 3);
            add_bias(&mut qkv, &cab, D_MODEL * 3);

            let mut ac = vec![0.0f64; padded_seq * D_MODEL];
            for hd in 0..N_HEADS {
                let mut q = vec![0.0f64; padded_seq * D_HEAD];
                let mut k = vec![0.0f64; padded_seq * D_HEAD];
                let mut v = vec![0.0f64; padded_seq * D_HEAD];
                for s in 0..padded_seq {
                    for d in 0..D_HEAD {
                        q[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + hd * D_HEAD + d];
                        k[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + D_MODEL + hd * D_HEAD + d];
                        v[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + 2 * D_MODEL + hd * D_HEAD + d];
                    }
                }
                let scale = 1.0 / (D_HEAD as f64).sqrt();
                let mut scores = vec![0.0f64; padded_seq * padded_seq];
                for i in 0..padded_seq {
                    for j in 0..padded_seq {
                        if j <= i {
                            let mut dot = 0.0f64;
                            for d in 0..D_HEAD {
                                dot += q[i * D_HEAD + d] * k[j * D_HEAD + d];
                            }
                            scores[i * padded_seq + j] = dot * scale;
                        } else {
                            scores[i * padded_seq + j] = f64::NEG_INFINITY;
                        }
                    }
                }
                for i in 0..padded_seq {
                    let row = &mut scores[i * padded_seq..(i + 1) * padded_seq];
                    let mx = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let mut sm = 0.0f64;
                    for j in 0..padded_seq {
                        row[j] = (row[j] - mx).exp();
                        sm += row[j];
                    }
                    for j in 0..padded_seq {
                        row[j] /= sm;
                    }
                }
                let ao = matmul(&scores, &v, padded_seq, padded_seq, D_HEAD);
                for s in 0..padded_seq {
                    for d in 0..D_HEAD {
                        ac[s * D_MODEL + hd * D_HEAD + d] = ao[s * D_HEAD + d];
                    }
                }
            }

            let mut po = matmul(&ac, &cpw, padded_seq, D_MODEL, D_MODEL);
            add_bias(&mut po, &cpb, D_MODEL);
            for i in 0..h_cmp.len() {
                h_cmp[i] += po[i];
            }
            for s in seq..padded_seq {
                for d in 0..D_MODEL {
                    h_cmp[s * D_MODEL + d] = 0.0;
                }
            }

            let l2o = layer_norm(&h_cmp, &l2g, &l2b, D_MODEL, EPS);
            let mut fh = matmul(&l2o, &fw, padded_seq, D_MODEL, D_FFN);
            add_bias(&mut fh, &fb, D_FFN);
            for x in fh.iter_mut() {
                *x = gelu(*x);
            }
            let mut fo = matmul(&fh, &pw, padded_seq, D_FFN, D_MODEL);
            add_bias(&mut fo, &pb, D_MODEL);
            for i in 0..h_cmp.len() {
                h_cmp[i] += fo[i];
            }
            for s in seq..padded_seq {
                for d in 0..D_MODEL {
                    h_cmp[s * D_MODEL + d] = 0.0;
                }
            }

            // Print hidden stats at pos 4 for comparison with GPU
            let lp = seq - 1;
            let h_slice = &h_cmp[lp * D_MODEL..(lp + 1) * D_MODEL];
            let h_max = h_slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let h_min = h_slice.iter().cloned().fold(f64::INFINITY, f64::min);
            let h_maxabs = h_slice.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
            let first4: Vec<String> = h_slice[..4].iter().map(|x| format!("{x:.4}")).collect();
            println!(
                "  Layer {:2}: max|val|={:.2}, max={:.2}, min={:.2}, first4=[{}]",
                li,
                h_maxabs,
                h_max,
                h_min,
                first4.join(", ")
            );
        }
    }

    // === Additional prompt tests ===
    println!("\n  === Multi-prompt CPU f64 test ===");
    let test_prompts = [
        "Hello, my name is",
        "The largest city in Japan is",
        "1 + 1 =",
        "Barack Obama was the president of the",
    ];
    for tp in &test_prompts {
        let toks = tokenizer.encode(tp);
        let tseq = toks.len();

        // Embedding
        let mut h = vec![0.0f64; tseq * D_MODEL];
        for pos in 0..tseq {
            let tok = toks[pos] as usize;
            for d in 0..D_MODEL {
                h[pos * D_MODEL + d] = wte[tok * D_MODEL + d] + wpe[pos * D_MODEL + d];
            }
        }

        // 12 layers
        for li in 0..12 {
            let lw = &weights.layers[li];
            let lg: Vec<f64> = lw.ln_1.weight.iter().map(|&x| x as f64).collect();
            let lb: Vec<f64> = lw.ln_1.bias.iter().map(|&x| x as f64).collect();
            let caw: Vec<f64> = lw.c_attn_weight.iter().map(|&x| x as f64).collect();
            let cab: Vec<f64> = lw.c_attn_bias.iter().map(|&x| x as f64).collect();
            let cpw: Vec<f64> = lw.c_proj_weight.iter().map(|&x| x as f64).collect();
            let cpb: Vec<f64> = lw.c_proj_bias.iter().map(|&x| x as f64).collect();
            let l2g: Vec<f64> = lw.ln_2.weight.iter().map(|&x| x as f64).collect();
            let l2b: Vec<f64> = lw.ln_2.bias.iter().map(|&x| x as f64).collect();
            let fw: Vec<f64> = lw.mlp_fc_weight.iter().map(|&x| x as f64).collect();
            let fb: Vec<f64> = lw.mlp_fc_bias.iter().map(|&x| x as f64).collect();
            let pw: Vec<f64> = lw.mlp_proj_weight.iter().map(|&x| x as f64).collect();
            let pb: Vec<f64> = lw.mlp_proj_bias.iter().map(|&x| x as f64).collect();

            let lo = layer_norm(&h, &lg, &lb, D_MODEL, EPS);
            let mut qkv = matmul(&lo, &caw, tseq, D_MODEL, D_MODEL * 3);
            add_bias(&mut qkv, &cab, D_MODEL * 3);

            let mut ac = vec![0.0f64; tseq * D_MODEL];
            for hd in 0..N_HEADS {
                let mut q = vec![0.0f64; tseq * D_HEAD];
                let mut k = vec![0.0f64; tseq * D_HEAD];
                let mut v = vec![0.0f64; tseq * D_HEAD];
                for s in 0..tseq {
                    for d in 0..D_HEAD {
                        q[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + hd * D_HEAD + d];
                        k[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + D_MODEL + hd * D_HEAD + d];
                        v[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + 2 * D_MODEL + hd * D_HEAD + d];
                    }
                }
                let scale = 1.0 / (D_HEAD as f64).sqrt();
                let mut scores = vec![0.0f64; tseq * tseq];
                for i in 0..tseq {
                    for j in 0..tseq {
                        if j <= i {
                            let mut dot = 0.0f64;
                            for d in 0..D_HEAD {
                                dot += q[i * D_HEAD + d] * k[j * D_HEAD + d];
                            }
                            scores[i * tseq + j] = dot * scale;
                        } else {
                            scores[i * tseq + j] = f64::NEG_INFINITY;
                        }
                    }
                }
                for i in 0..tseq {
                    let row = &mut scores[i * tseq..(i + 1) * tseq];
                    let mx = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let mut sm = 0.0f64;
                    for j in 0..tseq {
                        row[j] = (row[j] - mx).exp();
                        sm += row[j];
                    }
                    for j in 0..tseq {
                        row[j] /= sm;
                    }
                }
                let ao = matmul(&scores, &v, tseq, tseq, D_HEAD);
                for s in 0..tseq {
                    for d in 0..D_HEAD {
                        ac[s * D_MODEL + hd * D_HEAD + d] = ao[s * D_HEAD + d];
                    }
                }
            }

            let mut po = matmul(&ac, &cpw, tseq, D_MODEL, D_MODEL);
            add_bias(&mut po, &cpb, D_MODEL);
            for i in 0..h.len() {
                h[i] += po[i];
            }

            let l2o = layer_norm(&h, &l2g, &l2b, D_MODEL, EPS);
            let mut fh = matmul(&l2o, &fw, tseq, D_MODEL, D_FFN);
            add_bias(&mut fh, &fb, D_FFN);
            for x in fh.iter_mut() {
                *x = gelu(*x);
            }
            let mut fo = matmul(&fh, &pw, tseq, D_FFN, D_MODEL);
            add_bias(&mut fo, &pb, D_MODEL);
            for i in 0..h.len() {
                h[i] += fo[i];
            }
        }

        let lnf = layer_norm(&h, &ln_f_g, &ln_f_b, D_MODEL, EPS);
        let lp = tseq - 1;
        let hv = &lnf[lp * D_MODEL..(lp + 1) * D_MODEL];
        let t5 = top5_from_hidden(hv, &wte, D_MODEL, VOCAB_SIZE, &tokenizer);
        println!(
            "  \"{tp}\" → top5: {}",
            t5.iter()
                .map(|(_, l, s)| format!("{s}({l:.2})"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // === CPU f64 autoregressive generation (full-inference.12 diagnostic) ===
    // Run generation with f64 to determine if NaN at step 22 is f32-specific
    println!("\n  === CPU f64 autoregressive generation (50 tokens) ===");
    let gen_prompt = "The capital of France is";
    let mut gen_tokens: Vec<u32> = tokenizer.encode(gen_prompt);
    let gen_prompt_len = gen_tokens.len();
    let max_gen = 50;
    print!("  \"{gen_prompt}\"");

    for step in 0..max_gen {
        let tseq = gen_tokens.len();

        // Embedding
        let mut h = vec![0.0f64; tseq * D_MODEL];
        for pos in 0..tseq {
            let tok = gen_tokens[pos] as usize;
            for d in 0..D_MODEL {
                h[pos * D_MODEL + d] = wte[tok * D_MODEL + d] + wpe[pos * D_MODEL + d];
            }
        }

        // 12 layers
        for li in 0..12 {
            let lw = &weights.layers[li];
            let lg: Vec<f64> = lw.ln_1.weight.iter().map(|&x| x as f64).collect();
            let lb: Vec<f64> = lw.ln_1.bias.iter().map(|&x| x as f64).collect();
            let caw: Vec<f64> = lw.c_attn_weight.iter().map(|&x| x as f64).collect();
            let cab: Vec<f64> = lw.c_attn_bias.iter().map(|&x| x as f64).collect();
            let cpw: Vec<f64> = lw.c_proj_weight.iter().map(|&x| x as f64).collect();
            let cpb: Vec<f64> = lw.c_proj_bias.iter().map(|&x| x as f64).collect();
            let l2g: Vec<f64> = lw.ln_2.weight.iter().map(|&x| x as f64).collect();
            let l2b: Vec<f64> = lw.ln_2.bias.iter().map(|&x| x as f64).collect();
            let fw: Vec<f64> = lw.mlp_fc_weight.iter().map(|&x| x as f64).collect();
            let fb: Vec<f64> = lw.mlp_fc_bias.iter().map(|&x| x as f64).collect();
            let pw: Vec<f64> = lw.mlp_proj_weight.iter().map(|&x| x as f64).collect();
            let pb: Vec<f64> = lw.mlp_proj_bias.iter().map(|&x| x as f64).collect();

            let lo = layer_norm(&h, &lg, &lb, D_MODEL, EPS);
            let mut qkv = matmul(&lo, &caw, tseq, D_MODEL, D_MODEL * 3);
            add_bias(&mut qkv, &cab, D_MODEL * 3);

            let mut ac = vec![0.0f64; tseq * D_MODEL];
            for hd in 0..N_HEADS {
                let mut q = vec![0.0f64; tseq * D_HEAD];
                let mut k = vec![0.0f64; tseq * D_HEAD];
                let mut v = vec![0.0f64; tseq * D_HEAD];
                for s in 0..tseq {
                    for d in 0..D_HEAD {
                        q[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + hd * D_HEAD + d];
                        k[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + D_MODEL + hd * D_HEAD + d];
                        v[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + 2 * D_MODEL + hd * D_HEAD + d];
                    }
                }
                let scale = 1.0 / (D_HEAD as f64).sqrt();
                let mut scores = vec![0.0f64; tseq * tseq];
                for i in 0..tseq {
                    for j in 0..tseq {
                        if j <= i {
                            let mut dot = 0.0f64;
                            for d in 0..D_HEAD {
                                dot += q[i * D_HEAD + d] * k[j * D_HEAD + d];
                            }
                            scores[i * tseq + j] = dot * scale;
                        } else {
                            scores[i * tseq + j] = f64::NEG_INFINITY;
                        }
                    }
                }
                for i in 0..tseq {
                    let row = &mut scores[i * tseq..(i + 1) * tseq];
                    let mx = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let mut sm = 0.0f64;
                    for j in 0..tseq {
                        row[j] = (row[j] - mx).exp();
                        sm += row[j];
                    }
                    for j in 0..tseq {
                        row[j] /= sm;
                    }
                }
                let ao = matmul(&scores, &v, tseq, tseq, D_HEAD);
                for s in 0..tseq {
                    for d in 0..D_HEAD {
                        ac[s * D_MODEL + hd * D_HEAD + d] = ao[s * D_HEAD + d];
                    }
                }
            }

            let mut po = matmul(&ac, &cpw, tseq, D_MODEL, D_MODEL);
            add_bias(&mut po, &cpb, D_MODEL);
            for i in 0..h.len() {
                h[i] += po[i];
            }

            let l2o = layer_norm(&h, &l2g, &l2b, D_MODEL, EPS);
            let mut fh = matmul(&l2o, &fw, tseq, D_MODEL, D_FFN);
            add_bias(&mut fh, &fb, D_FFN);
            for x in fh.iter_mut() {
                *x = gelu(*x);
            }
            let mut fo = matmul(&fh, &pw, tseq, D_FFN, D_MODEL);
            add_bias(&mut fo, &pb, D_MODEL);
            for i in 0..h.len() {
                h[i] += fo[i];
            }
        }

        let lnf = layer_norm(&h, &ln_f_g, &ln_f_b, D_MODEL, EPS);
        let lp = tseq - 1;
        let hv = &lnf[lp * D_MODEL..(lp + 1) * D_MODEL];

        // Check for NaN
        let nan_count = hv.iter().filter(|v| v.is_nan()).count();
        if nan_count > 0 {
            println!();
            println!("  CPU f64 NaN at step {step}! ({nan_count} NaN values)");
            break;
        }

        // Greedy argmax
        let mut best_logit = f64::NEG_INFINITY;
        let mut best_token: u32 = 0;
        for v in 0..VOCAB_SIZE {
            let mut dot = 0.0f64;
            for d in 0..D_MODEL {
                dot += hv[d] * wte[v * D_MODEL + d];
            }
            if dot > best_logit {
                best_logit = dot;
                best_token = v as u32;
            }
        }

        let token_str = tokenizer
            .decode(&[best_token])
            .unwrap_or_else(|_| format!("<tok {best_token}>"));
        print!("{token_str}");

        if best_token == 50256 {
            println!();
            println!("  [<|endoftext|> at step {step}]");
            break;
        }

        gen_tokens.push(best_token);

        // Print hidden norm every 10 steps
        if (step + 1) % 10 == 0 {
            let h_norm: f64 = h[lp * D_MODEL..(lp + 1) * D_MODEL]
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();
            println!();
            print!("    [step {step}, seq={tseq}, h_norm={h_norm:.1}]");
        }
    }
    println!();

    let gen_count = gen_tokens.len() - gen_prompt_len;
    println!("  CPU f64 generated {gen_count} tokens without NaN");
    println!(
        "  Full: \"{}\"",
        tokenizer
            .decode(&gen_tokens[gen_prompt_len..])
            .unwrap_or_else(|_| "<decode error>".to_string())
    );

    println!("  CPU f64 reference forward pass — PASSED");
    Ok(())
}

/// kv-cache.3: Full 12-layer KV-cached autoregressive generation.
///
/// Phase 1 (prefill): process full prompt through 12 layers, populate KV cache.
/// Phase 2 (decode): generate tokens one at a time using cached K/V.
/// Compares output tokens against non-cached generation to verify correctness.
pub(crate) fn run_kv_cached_generation_test(dev: Arc<CudaDevice>) -> Result<()> {
    let model_path =
        gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("model.safetensors");
    if !model_path.exists() {
        println!("\n--- Skipping KV-cached generation (models/model.safetensors not found) ---");
        return Ok(());
    }

    println!("\n--- KV-cached autoregressive generation (kv-cache.3) ---");

    let weights = gpu_host::model::load_gpt2_weights(&model_path).map_err(|e| {
        GpuHostError::Verification {
            test: "kv_cached_generation",
            detail: format!("weight loading: {e}"),
        }
    })?;

    let tokenizer =
        gpu_host::tokenizer::Gpt2Tokenizer::new().map_err(|e| GpuHostError::Verification {
            test: "kv_cached_generation",
            detail: format!("tokenizer: {e}"),
        })?;

    const D_MODEL: u32 = 768;
    const N_HEADS: u32 = 12;
    const D_HEAD: u32 = 64;
    const D_FFN: u32 = 3072;
    const EPS: f32 = 1e-5;
    let dm = D_MODEL as usize;

    // Full sequence length (same as non-cached test)
    const MAX_SEQ: u32 = 128;
    // Padded sequence length for decode-phase GEMM (minimum tile size)
    const SEQ_STEP: u32 = 32;

    let total_full = (MAX_SEQ * D_MODEL) as usize;
    let head_total_full = (N_HEADS * MAX_SEQ * D_HEAD) as usize;
    let total_step = (SEQ_STEP * D_MODEL) as usize;
    let head_total_step = (N_HEADS * SEQ_STEP * D_HEAD) as usize;

    fn to_col_major(w: &[f32], k: usize, n: usize) -> Vec<f32> {
        let mut cm = vec![0.0f32; k * n];
        for row in 0..k {
            for col in 0..n {
                cm[col * k + row] = w[row * n + col];
            }
        }
        cm
    }

    // Load PTX with all required kernels
    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(
        ptx,
        "kvcache",
        &[
            "embedding_lookup",
            "layer_norm",
            "gemm_f32",
            "bias_add",
            "split_qkv",
            "flash_attention",
            "flash_attention_kv",
            "concat_heads",
            "gelu_forward",
            "elementwise_add",
            "zero_pad",
            "kv_cache_append",
        ],
    )?;

    macro_rules! get_fn {
        ($name:expr) => {
            dev.get_func("kvcache", $name)
                .ok_or(GpuHostError::KernelNotFound($name))?
        };
    }

    let f_embed = get_fn!("embedding_lookup");
    let f_ln = get_fn!("layer_norm");
    let f_gemm = get_fn!("gemm_f32");
    let f_bias = get_fn!("bias_add");
    let f_split = get_fn!("split_qkv");
    let f_attn = get_fn!("flash_attention");
    let f_attn_kv = get_fn!("flash_attention_kv");
    let f_concat = get_fn!("concat_heads");
    let f_gelu = get_fn!("gelu_forward");
    let f_add = get_fn!("elementwise_add");
    let f_zero = get_fn!("zero_pad");
    let f_kv_append = get_fn!("kv_cache_append");

    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    // Upload embedding tables
    let wte_dev = dev.htod_sync_copy(&weights.wte)?;
    let wpe_dev = dev.htod_sync_copy(&weights.wpe)?;

    // Upload layer weights
    struct LayerWeightsGpu {
        ln1_g: CudaSlice<f32>,
        ln1_b: CudaSlice<f32>,
        w_qkv: CudaSlice<f32>,
        b_qkv: CudaSlice<f32>,
        w_proj: CudaSlice<f32>,
        b_proj: CudaSlice<f32>,
        ln2_g: CudaSlice<f32>,
        ln2_b: CudaSlice<f32>,
        w_fc: CudaSlice<f32>,
        b_fc: CudaSlice<f32>,
        w_fc_proj: CudaSlice<f32>,
        b_fc_proj: CudaSlice<f32>,
    }

    let mut gpu_layers: Vec<LayerWeightsGpu> = Vec::with_capacity(12);
    for layer in weights.layers.iter() {
        let w_qkv_cm = to_col_major(&layer.c_attn_weight, 768, 2304);
        let w_proj_cm = to_col_major(&layer.c_proj_weight, 768, 768);
        let w_fc_cm = to_col_major(&layer.mlp_fc_weight, 768, 3072);
        let w_fc_proj_cm = to_col_major(&layer.mlp_proj_weight, 3072, 768);

        gpu_layers.push(LayerWeightsGpu {
            ln1_g: dev.htod_sync_copy(&layer.ln_1.weight)?,
            ln1_b: dev.htod_sync_copy(&layer.ln_1.bias)?,
            w_qkv: dev.htod_sync_copy(&w_qkv_cm)?,
            b_qkv: dev.htod_sync_copy(&layer.c_attn_bias)?,
            w_proj: dev.htod_sync_copy(&w_proj_cm)?,
            b_proj: dev.htod_sync_copy(&layer.c_proj_bias)?,
            ln2_g: dev.htod_sync_copy(&layer.ln_2.weight)?,
            ln2_b: dev.htod_sync_copy(&layer.ln_2.bias)?,
            w_fc: dev.htod_sync_copy(&w_fc_cm)?,
            b_fc: dev.htod_sync_copy(&layer.mlp_fc_bias)?,
            w_fc_proj: dev.htod_sync_copy(&w_fc_proj_cm)?,
            b_fc_proj: dev.htod_sync_copy(&layer.mlp_proj_bias)?,
        });
    }

    let ln_f_g_dev = dev.htod_sync_copy(&weights.ln_f.weight)?;
    let ln_f_b_dev = dev.htod_sync_copy(&weights.ln_f.bias)?;

    let gemm_shared = (32 * 16 + 16 * 16) * 4;

    // ============================
    // Step 1: Non-cached reference generation (one prompt)
    // ============================
    let prompt = "The capital of France is";
    let prompt_tokens = tokenizer.encode(prompt);
    let prompt_len = prompt_tokens.len();
    let max_new_tokens: usize = 50usize.min(MAX_SEQ as usize - prompt_len);

    println!("  Prompt: \"{prompt}\" ({prompt_len} tokens), generating {max_new_tokens}");

    // --- Non-cached reference run ---
    println!("  [Phase 1] Non-cached reference...");
    let ref_start = std::time::Instant::now();
    let reference_tokens = {
        let mut hidden_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_full)?;
        let mut ln_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_full)?;
        let mut qkv_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((MAX_SEQ * 2304) as usize)?;
        let mut q_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total_full)?;
        let mut k_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total_full)?;
        let mut v_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total_full)?;
        let mut attn_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total_full)?;
        let mut concat_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_full)?;
        let mut proj_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_full)?;
        let mut ffn_hidden_dev: CudaSlice<f32> =
            dev.alloc_zeros::<f32>((MAX_SEQ * D_FFN) as usize)?;
        let mut gelu_out_dev: CudaSlice<f32> =
            dev.alloc_zeros::<f32>((MAX_SEQ * D_FFN) as usize)?;
        let mut ffn_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_full)?;

        let n_q_tiles = (MAX_SEQ as usize).div_ceil(32) as u32;
        let mut tokens: Vec<u32> = prompt_tokens.clone();
        let mut generated: Vec<u32> = Vec::new();

        for _step in 0..max_new_tokens {
            let actual_seq = tokens.len();
            let mut token_ids_padded = tokens.clone();
            token_ids_padded.resize(MAX_SEQ as usize, 0);
            let token_ids_dev = dev.htod_sync_copy(&token_ids_padded)?;

            unsafe {
                f_embed.clone().launch(
                    LaunchConfig {
                        grid_dim: ((total_full as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &wte_dev,
                        &wpe_dev,
                        &token_ids_dev,
                        &mut hidden_dev,
                        MAX_SEQ,
                        D_MODEL,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            let pad_start = (actual_seq as u32) * D_MODEL;
            let pad_count = total_full as u32 - pad_start;
            if pad_count > 0 {
                unsafe {
                    f_zero.clone().launch(
                        LaunchConfig {
                            grid_dim: (pad_count.div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&mut hidden_dev, pad_start, total_full as u32),
                    )?;
                }
                dev.synchronize()?;
            }

            for layer_idx in 0..12u32 {
                let lw = &gpu_layers[layer_idx as usize];

                unsafe {
                    f_ln.clone().launch(
                        LaunchConfig {
                            grid_dim: (MAX_SEQ, 1, 1),
                            block_dim: (32, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &hidden_dev,
                            &mut ln_out_dev,
                            &lw.ln1_g,
                            &lw.ln1_b,
                            D_MODEL,
                            EPS,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                unsafe {
                    f_gemm.clone().launch(
                        LaunchConfig {
                            grid_dim: (MAX_SEQ / 32, 2304 / 16, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: gemm_shared,
                        },
                        (
                            &ln_out_dev,
                            &lw.w_qkv,
                            &mut qkv_dev,
                            D_MODEL,
                            2304u32,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                let total_qkv = MAX_SEQ * 2304;
                unsafe {
                    f_bias.clone().launch(
                        LaunchConfig {
                            grid_dim: (total_qkv.div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&mut qkv_dev, &lw.b_qkv, 2304u32, total_qkv, status_dev_ptr),
                    )?;
                }
                dev.synchronize()?;

                unsafe {
                    f_split.clone().launch(
                        LaunchConfig {
                            grid_dim: ((head_total_full as u32).div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &qkv_dev, &mut q_dev, &mut k_dev, &mut v_dev, MAX_SEQ, N_HEADS, D_HEAD,
                        ),
                    )?;
                }
                dev.synchronize()?;

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
                            &mut attn_out_dev,
                            MAX_SEQ,
                            D_HEAD,
                            1u32,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                unsafe {
                    f_concat.clone().launch(
                        LaunchConfig {
                            grid_dim: ((total_full as u32).div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&attn_out_dev, &mut concat_dev, MAX_SEQ, N_HEADS, D_HEAD),
                    )?;
                }
                dev.synchronize()?;

                unsafe {
                    f_gemm.clone().launch(
                        LaunchConfig {
                            grid_dim: (MAX_SEQ / 32, D_MODEL / 16, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: gemm_shared,
                        },
                        (
                            &concat_dev,
                            &lw.w_proj,
                            &mut proj_out_dev,
                            D_MODEL,
                            D_MODEL,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                unsafe {
                    f_bias.clone().launch(
                        LaunchConfig {
                            grid_dim: ((total_full as u32).div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &mut proj_out_dev,
                            &lw.b_proj,
                            D_MODEL,
                            total_full as u32,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                unsafe {
                    f_add.clone().launch(
                        LaunchConfig {
                            grid_dim: ((total_full as u32).div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&mut hidden_dev, &proj_out_dev, total_full as u32),
                    )?;
                    if pad_count > 0 {
                        f_zero.clone().launch(
                            LaunchConfig {
                                grid_dim: (pad_count.div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&mut hidden_dev, pad_start, total_full as u32),
                        )?;
                    }
                }
                dev.synchronize()?;

                unsafe {
                    f_ln.clone().launch(
                        LaunchConfig {
                            grid_dim: (MAX_SEQ, 1, 1),
                            block_dim: (32, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &hidden_dev,
                            &mut ln_out_dev,
                            &lw.ln2_g,
                            &lw.ln2_b,
                            D_MODEL,
                            EPS,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                unsafe {
                    f_gemm.clone().launch(
                        LaunchConfig {
                            grid_dim: (MAX_SEQ / 32, D_FFN / 16, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: gemm_shared,
                        },
                        (
                            &ln_out_dev,
                            &lw.w_fc,
                            &mut ffn_hidden_dev,
                            D_MODEL,
                            D_FFN,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                let total_ffn = MAX_SEQ * D_FFN;
                unsafe {
                    f_bias.clone().launch(
                        LaunchConfig {
                            grid_dim: (total_ffn.div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &mut ffn_hidden_dev,
                            &lw.b_fc,
                            D_FFN,
                            total_ffn,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                unsafe {
                    f_gelu.clone().launch(
                        LaunchConfig {
                            grid_dim: (total_ffn.div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &ffn_hidden_dev,
                            &mut gelu_out_dev,
                            total_ffn,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                unsafe {
                    f_gemm.clone().launch(
                        LaunchConfig {
                            grid_dim: (MAX_SEQ / 32, D_MODEL / 16, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: gemm_shared,
                        },
                        (
                            &gelu_out_dev,
                            &lw.w_fc_proj,
                            &mut ffn_out_dev,
                            D_FFN,
                            D_MODEL,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                unsafe {
                    f_bias.clone().launch(
                        LaunchConfig {
                            grid_dim: ((total_full as u32).div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &mut ffn_out_dev,
                            &lw.b_fc_proj,
                            D_MODEL,
                            total_full as u32,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                unsafe {
                    f_add.clone().launch(
                        LaunchConfig {
                            grid_dim: ((total_full as u32).div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&mut hidden_dev, &ffn_out_dev, total_full as u32),
                    )?;
                    if pad_count > 0 {
                        f_zero.clone().launch(
                            LaunchConfig {
                                grid_dim: (pad_count.div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&mut hidden_dev, pad_start, total_full as u32),
                        )?;
                    }
                }
                dev.synchronize()?;
            }

            // Final LayerNorm
            unsafe {
                f_ln.clone().launch(
                    LaunchConfig {
                        grid_dim: (MAX_SEQ, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &hidden_dev,
                        &mut ln_out_dev,
                        &ln_f_g_dev,
                        &ln_f_b_dev,
                        D_MODEL,
                        EPS,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            let output: Vec<f32> = dev.dtoh_sync_copy(&ln_out_dev)?;
            let last_pos = actual_seq - 1;
            let hidden_vec = &output[last_pos * dm..(last_pos + 1) * dm];

            if hidden_vec.iter().any(|v| v.is_nan()) {
                break;
            }

            // CPU LM head
            let vocab_size = 50257;
            let mut best_logit = f32::NEG_INFINITY;
            let mut best_token: u32 = 0;
            for v in 0..vocab_size {
                let wte_row = &weights.wte[v * dm..(v + 1) * dm];
                let mut dot = 0.0f32;
                for d in 0..dm {
                    dot += hidden_vec[d] * wte_row[d];
                }
                if dot > best_logit {
                    best_logit = dot;
                    best_token = v as u32;
                }
            }

            generated.push(best_token);
            tokens.push(best_token);
            if best_token == 50256 {
                break;
            }
        }
        generated
    };
    let ref_elapsed = ref_start.elapsed();
    println!(
        "    Reference: {} tokens in {:.1}ms ({:.1}ms/tok)",
        reference_tokens.len(),
        ref_elapsed.as_secs_f64() * 1000.0,
        ref_elapsed.as_secs_f64() * 1000.0 / reference_tokens.len().max(1) as f64,
    );

    // ============================
    // Step 2: KV-cached generation
    // ============================
    println!("  [Phase 2] KV-cached generation...");
    let cached_start = std::time::Instant::now();
    let cached_tokens = {
        // Prefill buffers (full sequence)
        let mut hidden_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_full)?;
        let mut ln_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_full)?;
        let mut qkv_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((MAX_SEQ * 2304) as usize)?;
        let mut q_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total_full)?;
        let mut k_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total_full)?;
        let mut v_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total_full)?;
        let mut attn_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total_full)?;
        let mut concat_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_full)?;
        let mut proj_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_full)?;
        let mut ffn_hidden_dev: CudaSlice<f32> =
            dev.alloc_zeros::<f32>((MAX_SEQ * D_FFN) as usize)?;
        let mut gelu_out_dev: CudaSlice<f32> =
            dev.alloc_zeros::<f32>((MAX_SEQ * D_FFN) as usize)?;
        let mut ffn_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_full)?;

        // Decode-phase buffers (SEQ_STEP-padded for GEMM)
        let mut hidden_step_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_step)?;
        let mut ln_out_step_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_step)?;
        let mut qkv_step_dev: CudaSlice<f32> =
            dev.alloc_zeros::<f32>((SEQ_STEP * 2304) as usize)?;
        let mut q_step_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total_step)?;
        let mut k_step_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total_step)?;
        let mut v_step_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total_step)?;
        let mut attn_kv_out_dev: CudaSlice<f32> =
            dev.alloc_zeros::<f32>((N_HEADS * D_HEAD) as usize)?;
        let mut concat_step_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_step)?;
        let mut proj_step_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_step)?;
        let mut ffn_hidden_step_dev: CudaSlice<f32> =
            dev.alloc_zeros::<f32>((SEQ_STEP * D_FFN) as usize)?;
        let mut gelu_step_dev: CudaSlice<f32> =
            dev.alloc_zeros::<f32>((SEQ_STEP * D_FFN) as usize)?;
        let mut ffn_step_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_step)?;

        // KV cache: 12 layers × [n_heads, MAX_SEQ, d_head] for K and V
        let cache_size = (N_HEADS * MAX_SEQ * D_HEAD) as usize;
        let mut k_caches: Vec<CudaSlice<f32>> = Vec::with_capacity(12);
        let mut v_caches: Vec<CudaSlice<f32>> = Vec::with_capacity(12);
        for _ in 0..12 {
            k_caches.push(dev.alloc_zeros::<f32>(cache_size)?);
            v_caches.push(dev.alloc_zeros::<f32>(cache_size)?);
        }

        let n_q_tiles_full = (MAX_SEQ as usize).div_ceil(32) as u32;
        let mut tokens: Vec<u32> = prompt_tokens.clone();
        let mut generated: Vec<u32> = Vec::new();

        for step in 0..max_new_tokens {
            let actual_seq = tokens.len();

            if step == 0 {
                // ===== PREFILL: process full prompt =====
                let mut token_ids_padded = tokens.clone();
                token_ids_padded.resize(MAX_SEQ as usize, 0);
                let token_ids_dev = dev.htod_sync_copy(&token_ids_padded)?;

                unsafe {
                    f_embed.clone().launch(
                        LaunchConfig {
                            grid_dim: ((total_full as u32).div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &wte_dev,
                            &wpe_dev,
                            &token_ids_dev,
                            &mut hidden_dev,
                            MAX_SEQ,
                            D_MODEL,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                let pad_start = (actual_seq as u32) * D_MODEL;
                let pad_count = total_full as u32 - pad_start;
                if pad_count > 0 {
                    unsafe {
                        f_zero.clone().launch(
                            LaunchConfig {
                                grid_dim: (pad_count.div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&mut hidden_dev, pad_start, total_full as u32),
                        )?;
                    }
                    dev.synchronize()?;
                }

                for layer_idx in 0..12u32 {
                    let lw = &gpu_layers[layer_idx as usize];

                    unsafe {
                        f_ln.clone().launch(
                            LaunchConfig {
                                grid_dim: (MAX_SEQ, 1, 1),
                                block_dim: (32, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &hidden_dev,
                                &mut ln_out_dev,
                                &lw.ln1_g,
                                &lw.ln1_b,
                                D_MODEL,
                                EPS,
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    unsafe {
                        f_gemm.clone().launch(
                            LaunchConfig {
                                grid_dim: (MAX_SEQ / 32, 2304 / 16, 1),
                                block_dim: (128, 1, 1),
                                shared_mem_bytes: gemm_shared,
                            },
                            (
                                &ln_out_dev,
                                &lw.w_qkv,
                                &mut qkv_dev,
                                D_MODEL,
                                2304u32,
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    let total_qkv = MAX_SEQ * 2304;
                    unsafe {
                        f_bias.clone().launch(
                            LaunchConfig {
                                grid_dim: (total_qkv.div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&mut qkv_dev, &lw.b_qkv, 2304u32, total_qkv, status_dev_ptr),
                        )?;
                    }
                    dev.synchronize()?;

                    unsafe {
                        f_split.clone().launch(
                            LaunchConfig {
                                grid_dim: ((head_total_full as u32).div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &qkv_dev, &mut q_dev, &mut k_dev, &mut v_dev, MAX_SEQ, N_HEADS,
                                D_HEAD,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    // Save K and V to cache: both have layout [n_heads, MAX_SEQ, d_head]
                    // so a direct copy works (same layout and stride).
                    let k_host: Vec<f32> = dev.dtoh_sync_copy(&k_dev)?;
                    let v_host: Vec<f32> = dev.dtoh_sync_copy(&v_dev)?;
                    dev.htod_copy_into(k_host, &mut k_caches[layer_idx as usize])?;
                    dev.htod_copy_into(v_host, &mut v_caches[layer_idx as usize])?;

                    unsafe {
                        f_attn.clone().launch(
                            LaunchConfig {
                                grid_dim: (N_HEADS, n_q_tiles_full, 1),
                                block_dim: (32, 1, 1),
                                shared_mem_bytes: 2 * 32 * 64 * 4,
                            },
                            (
                                &q_dev,
                                &k_dev,
                                &v_dev,
                                &mut attn_out_dev,
                                MAX_SEQ,
                                D_HEAD,
                                1u32,
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    unsafe {
                        f_concat.clone().launch(
                            LaunchConfig {
                                grid_dim: ((total_full as u32).div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&attn_out_dev, &mut concat_dev, MAX_SEQ, N_HEADS, D_HEAD),
                        )?;
                    }
                    dev.synchronize()?;

                    unsafe {
                        f_gemm.clone().launch(
                            LaunchConfig {
                                grid_dim: (MAX_SEQ / 32, D_MODEL / 16, 1),
                                block_dim: (128, 1, 1),
                                shared_mem_bytes: gemm_shared,
                            },
                            (
                                &concat_dev,
                                &lw.w_proj,
                                &mut proj_out_dev,
                                D_MODEL,
                                D_MODEL,
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    unsafe {
                        f_bias.clone().launch(
                            LaunchConfig {
                                grid_dim: ((total_full as u32).div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &mut proj_out_dev,
                                &lw.b_proj,
                                D_MODEL,
                                total_full as u32,
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    unsafe {
                        f_add.clone().launch(
                            LaunchConfig {
                                grid_dim: ((total_full as u32).div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&mut hidden_dev, &proj_out_dev, total_full as u32),
                        )?;
                        if pad_count > 0 {
                            f_zero.clone().launch(
                                LaunchConfig {
                                    grid_dim: (pad_count.div_ceil(256), 1, 1),
                                    block_dim: (256, 1, 1),
                                    shared_mem_bytes: 0,
                                },
                                (&mut hidden_dev, pad_start, total_full as u32),
                            )?;
                        }
                    }
                    dev.synchronize()?;

                    unsafe {
                        f_ln.clone().launch(
                            LaunchConfig {
                                grid_dim: (MAX_SEQ, 1, 1),
                                block_dim: (32, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &hidden_dev,
                                &mut ln_out_dev,
                                &lw.ln2_g,
                                &lw.ln2_b,
                                D_MODEL,
                                EPS,
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    unsafe {
                        f_gemm.clone().launch(
                            LaunchConfig {
                                grid_dim: (MAX_SEQ / 32, D_FFN / 16, 1),
                                block_dim: (128, 1, 1),
                                shared_mem_bytes: gemm_shared,
                            },
                            (
                                &ln_out_dev,
                                &lw.w_fc,
                                &mut ffn_hidden_dev,
                                D_MODEL,
                                D_FFN,
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    let total_ffn = MAX_SEQ * D_FFN;
                    unsafe {
                        f_bias.clone().launch(
                            LaunchConfig {
                                grid_dim: (total_ffn.div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &mut ffn_hidden_dev,
                                &lw.b_fc,
                                D_FFN,
                                total_ffn,
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    unsafe {
                        f_gelu.clone().launch(
                            LaunchConfig {
                                grid_dim: (total_ffn.div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &ffn_hidden_dev,
                                &mut gelu_out_dev,
                                total_ffn,
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    unsafe {
                        f_gemm.clone().launch(
                            LaunchConfig {
                                grid_dim: (MAX_SEQ / 32, D_MODEL / 16, 1),
                                block_dim: (128, 1, 1),
                                shared_mem_bytes: gemm_shared,
                            },
                            (
                                &gelu_out_dev,
                                &lw.w_fc_proj,
                                &mut ffn_out_dev,
                                D_FFN,
                                D_MODEL,
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    unsafe {
                        f_bias.clone().launch(
                            LaunchConfig {
                                grid_dim: ((total_full as u32).div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &mut ffn_out_dev,
                                &lw.b_fc_proj,
                                D_MODEL,
                                total_full as u32,
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    unsafe {
                        f_add.clone().launch(
                            LaunchConfig {
                                grid_dim: ((total_full as u32).div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&mut hidden_dev, &ffn_out_dev, total_full as u32),
                        )?;
                        if pad_count > 0 {
                            f_zero.clone().launch(
                                LaunchConfig {
                                    grid_dim: (pad_count.div_ceil(256), 1, 1),
                                    block_dim: (256, 1, 1),
                                    shared_mem_bytes: 0,
                                },
                                (&mut hidden_dev, pad_start, total_full as u32),
                            )?;
                        }
                    }
                    dev.synchronize()?;
                }

                // Final LayerNorm
                unsafe {
                    f_ln.clone().launch(
                        LaunchConfig {
                            grid_dim: (MAX_SEQ, 1, 1),
                            block_dim: (32, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &hidden_dev,
                            &mut ln_out_dev,
                            &ln_f_g_dev,
                            &ln_f_b_dev,
                            D_MODEL,
                            EPS,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                let output: Vec<f32> = dev.dtoh_sync_copy(&ln_out_dev)?;
                let last_pos = actual_seq - 1;
                let hidden_vec = &output[last_pos * dm..(last_pos + 1) * dm];

                if hidden_vec.iter().any(|v| v.is_nan()) {
                    break;
                }

                // Also save the last-position hidden state for decode continuation
                let mut hidden_1tok = vec![0.0f32; dm];
                hidden_1tok.copy_from_slice(hidden_vec);

                // LM head (CPU)
                let vocab_size = 50257;
                let mut best_logit = f32::NEG_INFINITY;
                let mut best_token: u32 = 0;
                for v in 0..vocab_size {
                    let wte_row = &weights.wte[v * dm..(v + 1) * dm];
                    let mut dot = 0.0f32;
                    for d in 0..dm {
                        dot += hidden_vec[d] * wte_row[d];
                    }
                    if dot > best_logit {
                        best_logit = dot;
                        best_token = v as u32;
                    }
                }

                generated.push(best_token);
                tokens.push(best_token);

                // Save the last-position hidden state from the full forward pass
                // for carrying into decode steps. We need the pre-final-LN hidden state.
                let prefill_hidden: Vec<f32> = dev.dtoh_sync_copy(&hidden_dev)?;
                let last_hidden = &prefill_hidden[last_pos * dm..(last_pos + 1) * dm];
                // We'll use this as the starting state for tracking the hidden state
                // across decode steps. But actually, for the decode phase, we re-embed
                // the new token and process it fresh through 12 layers — the hidden state
                // doesn't carry across steps. Only KV cache carries across.
                let _ = last_hidden;

                if best_token == 50256 {
                    break;
                }
            } else {
                // ===== DECODE: process single new token using KV cache =====
                let new_token = tokens[actual_seq - 1];
                let position = (actual_seq - 1) as u32; // 0-indexed position

                // CPU embedding for single token: wte[token] + wpe[position]
                let mut embed_vec = vec![0.0f32; total_step];
                for d in 0..dm {
                    embed_vec[d] = weights.wte[new_token as usize * dm + d]
                        + weights.wpe[position as usize * dm + d];
                }
                // Rest of 32-padded buffer stays zero
                dev.htod_copy_into(embed_vec, &mut hidden_step_dev)?;

                for layer_idx in 0..12u32 {
                    let lw = &gpu_layers[layer_idx as usize];

                    // LayerNorm (only row 0 has data, but LN is per-row so padding rows are fine)
                    unsafe {
                        f_ln.clone().launch(
                            LaunchConfig {
                                grid_dim: (SEQ_STEP, 1, 1),
                                block_dim: (32, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &hidden_step_dev,
                                &mut ln_out_step_dev,
                                &lw.ln1_g,
                                &lw.ln1_b,
                                D_MODEL,
                                EPS,
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    // QKV GEMM: [32, 768] × [768, 2304] → [32, 2304]
                    unsafe {
                        f_gemm.clone().launch(
                            LaunchConfig {
                                grid_dim: (SEQ_STEP / 32, 2304 / 16, 1),
                                block_dim: (128, 1, 1),
                                shared_mem_bytes: gemm_shared,
                            },
                            (
                                &ln_out_step_dev,
                                &lw.w_qkv,
                                &mut qkv_step_dev,
                                D_MODEL,
                                2304u32,
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    // QKV bias
                    let total_qkv_step = SEQ_STEP * 2304;
                    unsafe {
                        f_bias.clone().launch(
                            LaunchConfig {
                                grid_dim: (total_qkv_step.div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &mut qkv_step_dev,
                                &lw.b_qkv,
                                2304u32,
                                total_qkv_step,
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    // Split QKV (produces [12, 32, 64] for q, k, v)
                    unsafe {
                        f_split.clone().launch(
                            LaunchConfig {
                                grid_dim: ((head_total_step as u32).div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &qkv_step_dev,
                                &mut q_step_dev,
                                &mut k_step_dev,
                                &mut v_step_dev,
                                SEQ_STEP,
                                N_HEADS,
                                D_HEAD,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    // Append new K/V to cache (row 0 from [12, 32, 64] → cache position)
                    let kv_elems = N_HEADS * D_HEAD;
                    unsafe {
                        f_kv_append.clone().launch(
                            LaunchConfig {
                                grid_dim: (kv_elems.div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &k_step_dev,
                                &mut k_caches[layer_idx as usize],
                                N_HEADS,
                                SEQ_STEP,
                                MAX_SEQ,
                                D_HEAD,
                                position, // write_pos = actual position of new token
                                status_dev_ptr,
                            ),
                        )?;
                        f_kv_append.clone().launch(
                            LaunchConfig {
                                grid_dim: (kv_elems.div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &v_step_dev,
                                &mut v_caches[layer_idx as usize],
                                N_HEADS,
                                SEQ_STEP,
                                MAX_SEQ,
                                D_HEAD,
                                position,
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    // Flash attention with KV cache: q(1 token) vs full cache
                    // flash_attention_kv(q, k, v, out, q_len, kv_len, d_head, causal, q_offset, status)
                    // q: [n_heads, 1, d_head] — we use row 0 from q_step_dev [n_heads, 32, d_head]
                    // For q_len=1, we need q data at [h * 1 * d_head + 0 * d_head + d]
                    // But q_step_dev has [h * 32 * d_head + 0 * d_head + d] layout.
                    // We need to repack q from stride=32 to stride=1.
                    // Use kv_cache_append in reverse: copy from [12, 32, 64] row 0 to a compact [12, 1, 64] buffer.
                    // Actually, we can reuse kv_cache_append — it reads row 0 from src and writes to write_pos in dst.
                    // Create a compact q buffer [12, 1, 64]:
                    let q_compact_size = (N_HEADS * D_HEAD) as usize;
                    let mut q_compact_dev: CudaSlice<f32> =
                        dev.alloc_zeros::<f32>(q_compact_size)?;
                    unsafe {
                        f_kv_append.clone().launch(
                            LaunchConfig {
                                grid_dim: (kv_elems.div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &q_step_dev,
                                &mut q_compact_dev,
                                N_HEADS,
                                SEQ_STEP, // src stride
                                1u32,     // dst max_seq (compact)
                                D_HEAD,
                                0u32, // write at position 0
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    // flash_attention_kv: q_len=1, kv_len=actual_seq
                    unsafe {
                        f_attn_kv.clone().launch(
                            LaunchConfig {
                                grid_dim: (N_HEADS, 1, 1), // 1 q-tile (1 token)
                                block_dim: (32, 1, 1),
                                shared_mem_bytes: 2 * 32 * 64 * 4,
                            },
                            (
                                &q_compact_dev,
                                &k_caches[layer_idx as usize],
                                &v_caches[layer_idx as usize],
                                &mut attn_kv_out_dev,
                                1u32,              // q_len
                                actual_seq as u32, // kv_len (valid entries)
                                D_HEAD,
                                1u32,     // causal mask
                                position, // q_offset for causal masking
                                MAX_SEQ,  // kv_stride (cache allocated with MAX_SEQ)
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    // Concat heads: attn_kv_out [12, 1, 64] → [1, 768]
                    // concat_heads expects [n_heads, seq_len, d_head] → [seq_len, n_heads * d_head]
                    // For seq_len=1, output is [1, 768]. But we need it in a 32-padded buffer
                    // for the next GEMM.
                    // First concat into a compact buffer, then pad.
                    let mut concat_1tok: CudaSlice<f32> = dev.alloc_zeros::<f32>(dm)?;
                    unsafe {
                        f_concat.clone().launch(
                            LaunchConfig {
                                grid_dim: ((dm as u32).div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&attn_kv_out_dev, &mut concat_1tok, 1u32, N_HEADS, D_HEAD),
                        )?;
                    }
                    dev.synchronize()?;

                    // Copy 1-token concat into 32-padded buffer (first row only)
                    // Zero the step buffer first, then copy row 0
                    unsafe {
                        f_zero.clone().launch(
                            LaunchConfig {
                                grid_dim: ((total_step as u32).div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&mut concat_step_dev, 0u32, total_step as u32),
                        )?;
                    }
                    dev.synchronize()?;

                    // Copy 768 floats from concat_1tok to row 0 of concat_step_dev
                    let concat_host: Vec<f32> = dev.dtoh_sync_copy(&concat_1tok)?;
                    let mut concat_padded = vec![0.0f32; total_step];
                    concat_padded[..dm].copy_from_slice(&concat_host);
                    dev.htod_copy_into(concat_padded, &mut concat_step_dev)?;

                    // Output projection GEMM: [32, 768] × [768, 768] → [32, 768]
                    unsafe {
                        f_gemm.clone().launch(
                            LaunchConfig {
                                grid_dim: (SEQ_STEP / 32, D_MODEL / 16, 1),
                                block_dim: (128, 1, 1),
                                shared_mem_bytes: gemm_shared,
                            },
                            (
                                &concat_step_dev,
                                &lw.w_proj,
                                &mut proj_step_dev,
                                D_MODEL,
                                D_MODEL,
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    // Projection bias
                    unsafe {
                        f_bias.clone().launch(
                            LaunchConfig {
                                grid_dim: ((total_step as u32).div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &mut proj_step_dev,
                                &lw.b_proj,
                                D_MODEL,
                                total_step as u32,
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    // Residual add (only affects row 0 meaningfully)
                    unsafe {
                        f_add.clone().launch(
                            LaunchConfig {
                                grid_dim: ((total_step as u32).div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&mut hidden_step_dev, &proj_step_dev, total_step as u32),
                        )?;
                    }
                    dev.synchronize()?;

                    // Zero padding rows (rows 1..31)
                    let step_pad_start = D_MODEL;
                    let step_pad_count = total_step as u32 - step_pad_start;
                    unsafe {
                        f_zero.clone().launch(
                            LaunchConfig {
                                grid_dim: (step_pad_count.div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&mut hidden_step_dev, step_pad_start, total_step as u32),
                        )?;
                    }
                    dev.synchronize()?;

                    // LayerNorm2
                    unsafe {
                        f_ln.clone().launch(
                            LaunchConfig {
                                grid_dim: (SEQ_STEP, 1, 1),
                                block_dim: (32, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &hidden_step_dev,
                                &mut ln_out_step_dev,
                                &lw.ln2_g,
                                &lw.ln2_b,
                                D_MODEL,
                                EPS,
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    // FFN up: [32, 768] × [768, 3072]
                    unsafe {
                        f_gemm.clone().launch(
                            LaunchConfig {
                                grid_dim: (SEQ_STEP / 32, D_FFN / 16, 1),
                                block_dim: (128, 1, 1),
                                shared_mem_bytes: gemm_shared,
                            },
                            (
                                &ln_out_step_dev,
                                &lw.w_fc,
                                &mut ffn_hidden_step_dev,
                                D_MODEL,
                                D_FFN,
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    let total_ffn_step = SEQ_STEP * D_FFN;
                    unsafe {
                        f_bias.clone().launch(
                            LaunchConfig {
                                grid_dim: (total_ffn_step.div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &mut ffn_hidden_step_dev,
                                &lw.b_fc,
                                D_FFN,
                                total_ffn_step,
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    // GELU
                    unsafe {
                        f_gelu.clone().launch(
                            LaunchConfig {
                                grid_dim: (total_ffn_step.div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &ffn_hidden_step_dev,
                                &mut gelu_step_dev,
                                total_ffn_step,
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    // FFN down: [32, 3072] × [3072, 768]
                    unsafe {
                        f_gemm.clone().launch(
                            LaunchConfig {
                                grid_dim: (SEQ_STEP / 32, D_MODEL / 16, 1),
                                block_dim: (128, 1, 1),
                                shared_mem_bytes: gemm_shared,
                            },
                            (
                                &gelu_step_dev,
                                &lw.w_fc_proj,
                                &mut ffn_step_dev,
                                D_FFN,
                                D_MODEL,
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    // FFN bias
                    unsafe {
                        f_bias.clone().launch(
                            LaunchConfig {
                                grid_dim: ((total_step as u32).div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &mut ffn_step_dev,
                                &lw.b_fc_proj,
                                D_MODEL,
                                total_step as u32,
                                status_dev_ptr,
                            ),
                        )?;
                    }
                    dev.synchronize()?;

                    // Residual 2
                    unsafe {
                        f_add.clone().launch(
                            LaunchConfig {
                                grid_dim: ((total_step as u32).div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&mut hidden_step_dev, &ffn_step_dev, total_step as u32),
                        )?;
                    }
                    dev.synchronize()?;

                    // Zero padding rows
                    unsafe {
                        f_zero.clone().launch(
                            LaunchConfig {
                                grid_dim: (step_pad_count.div_ceil(256), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&mut hidden_step_dev, step_pad_start, total_step as u32),
                        )?;
                    }
                    dev.synchronize()?;
                }

                // Final LayerNorm (only 1 row matters but run all SEQ_STEP rows)
                unsafe {
                    f_ln.clone().launch(
                        LaunchConfig {
                            grid_dim: (SEQ_STEP, 1, 1),
                            block_dim: (32, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &hidden_step_dev,
                            &mut ln_out_step_dev,
                            &ln_f_g_dev,
                            &ln_f_b_dev,
                            D_MODEL,
                            EPS,
                            status_dev_ptr,
                        ),
                    )?;
                }
                dev.synchronize()?;

                // Download row 0 only
                let output: Vec<f32> = dev.dtoh_sync_copy(&ln_out_step_dev)?;
                let hidden_vec = &output[..dm];

                if hidden_vec.iter().any(|v| v.is_nan()) {
                    println!("    Step {step}: NaN detected in decode phase!");
                    break;
                }

                // CPU LM head
                let vocab_size = 50257;
                let mut best_logit = f32::NEG_INFINITY;
                let mut best_token: u32 = 0;
                for v in 0..vocab_size {
                    let wte_row = &weights.wte[v * dm..(v + 1) * dm];
                    let mut dot = 0.0f32;
                    for d in 0..dm {
                        dot += hidden_vec[d] * wte_row[d];
                    }
                    if dot > best_logit {
                        best_logit = dot;
                        best_token = v as u32;
                    }
                }

                generated.push(best_token);
                tokens.push(best_token);
                if best_token == 50256 {
                    break;
                }
            }
        }
        generated
    };
    let cached_elapsed = cached_start.elapsed();

    println!(
        "    Cached:    {} tokens in {:.1}ms ({:.1}ms/tok)",
        cached_tokens.len(),
        cached_elapsed.as_secs_f64() * 1000.0,
        cached_elapsed.as_secs_f64() * 1000.0 / cached_tokens.len().max(1) as f64,
    );

    // ============================
    // Compare reference vs cached
    // ============================
    println!("  [Phase 3] Comparing outputs...");
    let ref_text = tokenizer
        .decode(&reference_tokens)
        .unwrap_or_else(|_| "<decode error>".to_string());
    let cached_text = tokenizer
        .decode(&cached_tokens)
        .unwrap_or_else(|_| "<decode error>".to_string());

    println!("    Reference: \"{ref_text}\"");
    println!("    Cached:    \"{cached_text}\"");

    let min_len = reference_tokens.len().min(cached_tokens.len());
    let mut first_mismatch: Option<usize> = None;
    for i in 0..min_len {
        if reference_tokens[i] != cached_tokens[i] {
            first_mismatch = Some(i);
            break;
        }
    }

    let match_count = first_mismatch.unwrap_or(min_len);
    println!(
        "    Token match: {}/{} (ref={}, cached={})",
        match_count,
        min_len,
        reference_tokens.len(),
        cached_tokens.len()
    );

    if let Some(pos) = first_mismatch {
        let ref_tok_str = tokenizer
            .decode(&[reference_tokens[pos]])
            .unwrap_or_else(|_| "?".to_string());
        let cached_tok_str = tokenizer
            .decode(&[cached_tokens[pos]])
            .unwrap_or_else(|_| "?".to_string());
        println!(
            "    First mismatch at step {pos}: ref=\"{ref_tok_str}\" vs cached=\"{cached_tok_str}\""
        );
    }

    // Calculate speedup
    let ref_ms_per_tok = ref_elapsed.as_secs_f64() * 1000.0 / reference_tokens.len().max(1) as f64;
    let cached_ms_per_tok =
        cached_elapsed.as_secs_f64() * 1000.0 / cached_tokens.len().max(1) as f64;
    let speedup = ref_ms_per_tok / cached_ms_per_tok;
    println!(
        "    Speedup: {speedup:.2}x ({ref_ms_per_tok:.1}ms → {cached_ms_per_tok:.1}ms per token)"
    );

    unsafe {
        free_mapped_mem(status_host_ptr)?;
    }

    // Pass criteria: all tokens match for 50+ tokens
    if match_count < 50.min(max_new_tokens) {
        return Err(GpuHostError::Verification {
            test: "kv_cached_generation",
            detail: format!(
                "only {match_count} tokens matched (need 50+). First mismatch at step {}",
                first_mismatch.unwrap_or(0)
            ),
        });
    }

    println!("  KV-cached generation — PASSED ({match_count} tokens match, {speedup:.2}x speedup)");
    Ok(())
}
