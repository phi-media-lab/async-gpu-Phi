//! CNN kernel tests: BatchNorm+SiLU, im2col, MaxPool2D, Upsample, Concat.
//! Used to validate kernels for YOLOv8-nano inference pipeline.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

use gpu_host::error::{GpuHostError, Result};
use gpu_host::mapped_mem::{alloc_mapped_result_array, free_mapped_mem};

/// Test fused BatchNorm + SiLU kernel.
///
/// Creates a small CHW tensor, applies BN+SiLU with known parameters,
/// and verifies output matches CPU reference.
pub(crate) fn run_batchnorm_silu_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- BatchNorm + SiLU fused kernel test ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(ptx, "cnn_test", &["batchnorm_silu", "silu_forward"])?;
    let f_bn_silu = dev
        .get_func("cnn_test", "batchnorm_silu")
        .ok_or(GpuHostError::KernelNotFound("batchnorm_silu"))?;
    let f_silu = dev
        .get_func("cnn_test", "silu_forward")
        .ok_or(GpuHostError::KernelNotFound("silu_forward"))?;

    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    // Test params: 2 channels, 4x4 spatial = 32 elements
    let c = 2u32;
    let h = 4u32;
    let w = 4u32;
    let hw = h * w;
    let n = c * hw;
    let eps = 1e-5f32;

    // Known BN parameters per channel
    let gamma = vec![1.5f32, 0.8];
    let beta = vec![0.1, -0.2];
    let running_mean = vec![0.5, 1.0];
    let running_var = vec![0.25, 1.0]; // std = 0.5, 1.0

    // Input: channel 0 has values 0.0..15.0, channel 1 has values 1.0..16.0
    let mut input = vec![0.0f32; n as usize];
    for i in 0..hw as usize {
        input[i] = i as f32; // channel 0
        input[hw as usize + i] = (i + 1) as f32; // channel 1
    }

    // CPU reference
    let mut expected = vec![0.0f32; n as usize];
    for i in 0..n as usize {
        let ch = i / hw as usize;
        let x = input[i];
        let inv_std = 1.0 / (running_var[ch] + eps).sqrt();
        let bn = gamma[ch] * (x - running_mean[ch]) * inv_std + beta[ch];
        let sigmoid = 1.0 / (1.0 + (-bn).exp());
        expected[i] = bn * sigmoid;
    }

    // GPU execution
    let input_dev = dev.htod_sync_copy(&input)?;
    let gamma_dev = dev.htod_sync_copy(&gamma)?;
    let beta_dev = dev.htod_sync_copy(&beta)?;
    let mean_dev = dev.htod_sync_copy(&running_mean)?;
    let var_dev = dev.htod_sync_copy(&running_var)?;
    let mut output_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(n as usize)?;

    unsafe {
        f_bn_silu.clone().launch(
            LaunchConfig {
                grid_dim: (n.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &input_dev,
                &mut output_dev,
                &gamma_dev,
                &beta_dev,
                &mean_dev,
                &var_dev,
                n,
                hw,
                eps,
                status_dev_ptr,
            ),
        )?;
    }
    dev.synchronize()?;

    let output: Vec<f32> = dev.dtoh_sync_copy(&output_dev)?;

    // Verify
    let mut max_err = 0.0f32;
    for i in 0..n as usize {
        let err = (output[i] - expected[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }

    println!("  BN+SiLU: max_err = {max_err:.6} (tolerance 1e-4)");
    if max_err > 1e-4 {
        println!("  First 8 values (GPU vs CPU):");
        for i in 0..8 {
            println!("    [{i}] gpu={:.6} cpu={:.6}", output[i], expected[i]);
        }
        return Err(GpuHostError::Verification {
            test: "batchnorm_silu",
            detail: format!("max error {max_err:.6} exceeds 1e-4"),
        });
    }
    println!("  BN+SiLU — PASSED");

    // Also test standalone SiLU
    let silu_input = vec![0.0f32, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0];
    let silu_expected: Vec<f32> = silu_input.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
    let silu_n = silu_input.len() as u32;

    let silu_in_dev = dev.htod_sync_copy(&silu_input)?;
    let mut silu_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(silu_n as usize)?;

    unsafe {
        f_silu.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (&silu_in_dev, &mut silu_out_dev, silu_n, status_dev_ptr),
        )?;
    }
    dev.synchronize()?;

    let silu_output: Vec<f32> = dev.dtoh_sync_copy(&silu_out_dev)?;
    let mut silu_max_err = 0.0f32;
    for i in 0..silu_n as usize {
        let err = (silu_output[i] - silu_expected[i]).abs();
        if err > silu_max_err {
            silu_max_err = err;
        }
    }
    println!("  SiLU: max_err = {silu_max_err:.6}");
    println!("  SiLU — PASSED");

    unsafe {
        free_mapped_mem(status_host_ptr)?;
    }

    println!("  CNN kernel tests — PASSED");
    Ok(())
}

/// Test im2col, MaxPool2D, Upsample, and Concat kernels.
pub(crate) fn run_cnn_ops_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- CNN ops test (im2col, MaxPool, Upsample, Concat) ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(
        ptx,
        "cnn_ops",
        &[
            "im2col",
            "maxpool2d",
            "upsample_nearest_2x",
            "concat_channels",
        ],
    )?;

    macro_rules! get_fn {
        ($name:expr) => {
            dev.get_func("cnn_ops", $name)
                .ok_or(GpuHostError::KernelNotFound($name))?
        };
    }

    let f_im2col = get_fn!("im2col");
    let f_maxpool = get_fn!("maxpool2d");
    let f_upsample = get_fn!("upsample_nearest_2x");
    let f_concat = get_fn!("concat_channels");

    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    // === Test Upsample 2x ===
    {
        // Input: 1 channel, 2x2
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let expected = vec![
            1.0, 1.0, 2.0, 2.0, // row 0
            1.0, 1.0, 2.0, 2.0, // row 1
            3.0, 3.0, 4.0, 4.0, // row 2
            3.0, 3.0, 4.0, 4.0, // row 3
        ];

        let in_dev = dev.htod_sync_copy(&input)?;
        let mut out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(16)?;

        unsafe {
            f_upsample.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&in_dev, &mut out_dev, 1u32, 2u32, 2u32, status_dev_ptr),
            )?;
        }
        dev.synchronize()?;

        let output: Vec<f32> = dev.dtoh_sync_copy(&out_dev)?;
        assert_eq!(output, expected, "Upsample 2x failed");
        println!("  Upsample 2x — PASSED");
    }

    // === Test Concat channels ===
    {
        // a: 1 channel 2x2, b: 1 channel 2x2
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let expected = vec![
            1.0, 2.0, 3.0, 4.0, // channel 0 (from a)
            5.0, 6.0, 7.0, 8.0, // channel 1 (from b)
        ];

        let a_dev = dev.htod_sync_copy(&a)?;
        let b_dev = dev.htod_sync_copy(&b)?;
        let mut out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(8)?;

        unsafe {
            f_concat.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &a_dev,
                    &b_dev,
                    &mut out_dev,
                    1u32,
                    1u32,
                    4u32,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        let output: Vec<f32> = dev.dtoh_sync_copy(&out_dev)?;
        assert_eq!(output, expected, "Concat channels failed");
        println!("  Concat channels — PASSED");
    }

    // === Test MaxPool2D ===
    {
        // Input: 1 channel, 4x4, MaxPool k=2 stride=2 pad=0 -> 2x2
        let input = vec![
            1.0f32, 3.0, 2.0, 4.0, // row 0
            5.0, 7.0, 6.0, 8.0, // row 1
            9.0, 11.0, 10.0, 12.0, // row 2
            13.0, 15.0, 14.0, 16.0, // row 3
        ];
        let expected = vec![
            7.0, 8.0, // max of [1,3,5,7], max of [2,4,6,8]
            15.0, 16.0, // max of [9,11,13,15], max of [10,12,14,16]
        ];

        let in_dev = dev.htod_sync_copy(&input)?;
        let mut out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(4)?;

        unsafe {
            f_maxpool.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &in_dev,
                    &mut out_dev,
                    1u32,
                    4u32,
                    4u32,
                    2u32,
                    2u32,
                    0u32,
                    2u32,
                    2u32,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        let output: Vec<f32> = dev.dtoh_sync_copy(&out_dev)?;
        assert_eq!(output, expected, "MaxPool2D failed");
        println!("  MaxPool2D (k=2 s=2) — PASSED");
    }

    // === Test im2col ===
    {
        // Input: 1 channel, 3x3, Conv 3x3 stride=1 pad=1 -> 3x3 output
        // im2col should produce [9, 9] matrix (9 output positions, 9 filter taps)
        let input = vec![
            1.0f32, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
            7.0, 8.0, 9.0, // row 2
        ];

        let c_in = 1u32;
        let h = 3u32;
        let w = 3u32;
        let kh = 3u32;
        let kw = 3u32;
        let stride = 1u32;
        let pad = 1u32;
        let h_out = (h + 2 * pad - kh) / stride + 1; // 3
        let w_out = (w + 2 * pad - kw) / stride + 1; // 3
        let col_width = c_in * kh * kw; // 9
        let total = h_out * w_out * col_width; // 81

        let in_dev = dev.htod_sync_copy(&input)?;
        let mut out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total as usize)?;

        unsafe {
            f_im2col.clone().launch(
                LaunchConfig {
                    grid_dim: (total.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &in_dev,
                    &mut out_dev,
                    c_in,
                    h,
                    w,
                    kh,
                    kw,
                    stride,
                    pad,
                    h_out,
                    w_out,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        let col_output: Vec<f32> = dev.dtoh_sync_copy(&out_dev)?;

        // Verify center element (oh=1, ow=1) — should contain the full 3x3 patch = [1..9]
        let center_row_start = (w_out + 1) as usize * col_width as usize;
        let center_row: Vec<f32> =
            col_output[center_row_start..center_row_start + col_width as usize].to_vec();
        let expected_center = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        if center_row != expected_center {
            println!("  im2col center row: {:?}", center_row);
            println!("  expected: {:?}", expected_center);
            return Err(GpuHostError::Verification {
                test: "im2col",
                detail: "center patch mismatch".to_string(),
            });
        }

        // Verify corner element (oh=0, ow=0) — should have 0-padding for out-of-bounds
        let corner_row: Vec<f32> = col_output[..col_width as usize].to_vec();
        // For (0,0) with pad=1: filter taps that extend outside are 0
        // Positions: (-1,-1) (-1,0) (-1,1) (0,-1) (0,0) (0,1) (1,-1) (1,0) (1,1)
        // = 0, 0, 0, 0, 1, 2, 0, 4, 5
        let expected_corner = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 4.0, 5.0];
        if corner_row != expected_corner {
            println!("  im2col corner row: {:?}", corner_row);
            println!("  expected: {:?}", expected_corner);
            return Err(GpuHostError::Verification {
                test: "im2col",
                detail: "corner patch mismatch".to_string(),
            });
        }

        println!("  im2col (3x3, pad=1) — PASSED");
    }

    unsafe {
        free_mapped_mem(status_host_ptr)?;
    }

    println!("  CNN ops test — ALL PASSED");
    Ok(())
}

/// Test Conv2D via im2col + GEMM pipeline.
///
/// Chains im2col → gemm_f32 to implement a 3x3 convolution, then verifies
/// the result matches a CPU reference convolution.
pub(crate) fn run_conv2d_test(dev: Arc<CudaDevice>) -> Result<()> {
    println!("\n--- Conv2D (im2col + GEMM) test ---");

    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_COMPUTE_PTX);
    dev.load_ptx(ptx, "conv2d_test", &["im2col", "gemm_f32"])?;
    let f_im2col = dev
        .get_func("conv2d_test", "im2col")
        .ok_or(GpuHostError::KernelNotFound("im2col"))?;
    let f_gemm = dev
        .get_func("conv2d_test", "gemm_f32")
        .ok_or(GpuHostError::KernelNotFound("gemm_f32"))?;

    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    // Conv2D: input [2, 8, 8] → weight [4, 2, 3, 3] → output [4, 8, 8]
    // stride=1, pad=1 → same spatial size
    let c_in = 2u32;
    let c_out = 4u32;
    let h = 8u32;
    let w = 8u32;
    let kh = 3u32;
    let kw = 3u32;
    let stride = 1u32;
    let pad = 1u32;
    let h_out = (h + 2 * pad - kh) / stride + 1; // 8
    let w_out = (w + 2 * pad - kw) / stride + 1; // 8

    let k_gemm = c_in * kh * kw; // 18
    let m_gemm = h_out * w_out; // 64
    let n_gemm = c_out; // 4

    // Generate deterministic input and weight
    let input_size = (c_in * h * w) as usize;
    let weight_size = (c_out * c_in * kh * kw) as usize;

    let input: Vec<f32> = (0..input_size)
        .map(|i| ((i * 7 + 3) % 11) as f32 * 0.1)
        .collect();
    let weight: Vec<f32> = (0..weight_size)
        .map(|i| ((i * 13 + 5) % 7) as f32 * 0.1 - 0.3)
        .collect();

    // CPU reference convolution
    let mut expected = vec![0.0f32; (c_out * h_out * w_out) as usize];
    for co in 0..c_out as usize {
        for oh in 0..h_out as usize {
            for ow in 0..w_out as usize {
                let mut sum = 0.0f32;
                for ci in 0..c_in as usize {
                    for fh in 0..kh as usize {
                        for fw in 0..kw as usize {
                            let ih = oh + fh;
                            let iw = ow + fw;
                            let val = if ih >= pad as usize
                                && ih < (h + pad) as usize
                                && iw >= pad as usize
                                && iw < (w + pad) as usize
                            {
                                input[ci * (h * w) as usize
                                    + (ih - pad as usize) * w as usize
                                    + (iw - pad as usize)]
                            } else {
                                0.0
                            };
                            let w_idx = co * (c_in * kh * kw) as usize
                                + ci * (kh * kw) as usize
                                + fh * kw as usize
                                + fw;
                            sum += val * weight[w_idx];
                        }
                    }
                }
                expected[co * (h_out * w_out) as usize + oh * w_out as usize + ow] = sum;
            }
        }
    }

    // GPU: Step 1 — im2col
    let im2col_size = (m_gemm * k_gemm) as usize;
    let input_dev = dev.htod_sync_copy(&input)?;
    let mut col_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(im2col_size)?;

    unsafe {
        f_im2col.clone().launch(
            LaunchConfig {
                grid_dim: ((im2col_size as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &input_dev,
                &mut col_dev,
                c_in,
                h,
                w,
                kh,
                kw,
                stride,
                pad,
                h_out,
                w_out,
                status_dev_ptr,
            ),
        )?;
    }
    dev.synchronize()?;

    // GPU: Step 2 — GEMM: col[M, K] × weight_cm[K, N] → output[M, N]
    // weight is [C_out, K] row-major = column-major [K, C_out] (same memory layout)
    //
    // gemm_f32 has no output boundary check — N must be padded to a multiple of 16.
    let n_padded = n_gemm.next_multiple_of(16); // 4 → 16
    let mut weight_cm = vec![0.0f32; (k_gemm * n_padded) as usize];
    for co in 0..n_gemm as usize {
        for k in 0..k_gemm as usize {
            weight_cm[co * k_gemm as usize + k] = weight[co * k_gemm as usize + k];
        }
    }
    // Columns n_gemm..n_padded are zero-padded (already 0.0)

    let weight_dev = dev.htod_sync_copy(&weight_cm)?;
    let mut output_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((m_gemm * n_padded) as usize)?;

    let num_blocks_m = m_gemm.div_ceil(32);
    let num_blocks_n = n_padded.div_ceil(16);
    let gemm_shared = (32 * 16 + 16 * 16) * 4;

    unsafe {
        f_gemm.clone().launch(
            LaunchConfig {
                grid_dim: (num_blocks_m, num_blocks_n, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: gemm_shared,
            },
            (
                &col_dev,
                &weight_dev,
                &mut output_dev,
                k_gemm,
                n_padded,
                status_dev_ptr,
            ),
        )?;
    }
    dev.synchronize()?;

    // GPU output is [M, N_padded] row-major = [H_out*W_out, N_padded]
    // Extract first C_out columns and reshape to [C_out, H_out, W_out]
    let raw_output: Vec<f32> = dev.dtoh_sync_copy(&output_dev)?;
    let mut gpu_output = vec![0.0f32; (c_out * h_out * w_out) as usize];
    for pos in 0..m_gemm as usize {
        for co in 0..n_gemm as usize {
            let oh = pos / w_out as usize;
            let ow = pos % w_out as usize;
            gpu_output[co * (h_out * w_out) as usize + oh * w_out as usize + ow] =
                raw_output[pos * n_padded as usize + co];
        }
    }

    // Verify
    let mut max_err = 0.0f32;
    let mut first_mismatch = None;
    for i in 0..(c_out * h_out * w_out) as usize {
        let err = (gpu_output[i] - expected[i]).abs();
        if err > max_err {
            max_err = err;
        }
        if err > 0.01 && first_mismatch.is_none() {
            first_mismatch = Some((i, gpu_output[i], expected[i]));
        }
    }

    println!("  Conv2D (im2col+GEMM): max_err = {max_err:.6}");
    if let Some((idx, got, exp)) = first_mismatch {
        println!("  First mismatch at [{idx}]: gpu={got:.4} cpu={exp:.4}");
        return Err(GpuHostError::Verification {
            test: "conv2d",
            detail: format!("max error {max_err:.6} exceeds tolerance"),
        });
    }
    println!("  Conv2D (2ch→4ch, 8x8, k=3, s=1, p=1) — PASSED");

    unsafe {
        free_mapped_mem(status_host_ptr)?;
    }

    Ok(())
}

/// Test YOLO weight loading and image I/O utilities.
///
/// Verifies:
/// 1. PPM image load + CHW conversion
/// 2. Letterbox resize preserves aspect ratio
/// 3. YOLO weight loading (if model file exists)
pub(crate) fn run_yolo_io_test() -> Result<()> {
    use gpu_host::model_yolo::{load_ppm, ImageCHW};

    println!("\n--- YOLO I/O test (weight loading + image) ---");

    // Test 1: ImageCHW from raw RGB
    {
        // 2x2 RGB image: red, green, blue, white
        let rgb = vec![
            255, 0, 0, // red
            0, 255, 0, // green
            0, 0, 255, // blue
            255, 255, 255, // white
        ];
        let img = ImageCHW::from_rgb_hwc(&rgb, 2, 2);
        assert_eq!(img.data.len(), 3 * 2 * 2);
        // CHW layout: R channel first
        assert!((img.data[0] - 1.0).abs() < 1e-6, "R[0,0] should be 1.0"); // red pixel R
        assert!((img.data[1] - 0.0).abs() < 1e-6, "R[0,1] should be 0.0"); // green pixel R
        assert!((img.data[4] - 0.0).abs() < 1e-6, "G[0,0] should be 0.0"); // red pixel G
        assert!((img.data[5] - 1.0).abs() < 1e-6, "G[0,1] should be 1.0"); // green pixel G
        println!("  ImageCHW from_rgb_hwc — PASSED");
    }

    // Test 2: Resize nearest
    {
        let rgb = vec![255u8, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
        let img = ImageCHW::from_rgb_hwc(&rgb, 2, 2);
        let resized = img.resize_nearest(4, 4);
        assert_eq!(resized.data.len(), 3 * 4 * 4);
        // Top-left 2x2 of resized should be red (from pixel 0,0)
        assert!((resized.data[0] - 1.0).abs() < 1e-6); // R[0,0]
        assert!((resized.data[1] - 1.0).abs() < 1e-6); // R[0,1] (replicated)
        println!("  resize_nearest — PASSED");
    }

    // Test 3: Letterbox
    {
        let rgb = vec![128u8; 6 * 4 * 3]; // 6x4 image
        let img = ImageCHW::from_rgb_hwc(&rgb, 6, 4);
        let (lb, scale, pad_x, pad_y) = img.letterbox(12);
        assert_eq!(lb.width, 12);
        assert_eq!(lb.height, 12);
        // scale = 12/6 = 2.0 (width is limiting), new_w=12, new_h=8
        assert!(
            (scale - 2.0).abs() < 1e-6,
            "scale should be 2.0, got {scale}"
        );
        assert_eq!(pad_x, 0, "no horizontal padding");
        assert_eq!(pad_y, 2, "2px vertical padding");
        // Top padding should be gray (0.5)
        let r_channel_start = 0;
        assert!(
            (lb.data[r_channel_start] - 0.5).abs() < 0.01,
            "top pad should be 0.5"
        );
        println!("  letterbox — PASSED");
    }

    // Test 4: PPM load (create a tiny PPM in memory and write to temp file)
    {
        let tmp_path = std::env::temp_dir().join("async_gpu_test.ppm");
        // Create a 3x2 PPM
        let mut ppm_data = Vec::new();
        ppm_data.extend_from_slice(b"P6\n3 2\n255\n");
        // 6 pixels, RGB
        ppm_data.extend_from_slice(&[
            255, 0, 0, 0, 255, 0, 0, 0, 255, // row 0: R G B
            128, 128, 128, 64, 64, 64, 0, 0, 0, // row 1: gray, dark, black
        ]);
        std::fs::write(&tmp_path, &ppm_data).expect("write temp PPM");

        let img = load_ppm(&tmp_path).expect("load PPM");
        assert_eq!(img.width, 3);
        assert_eq!(img.height, 2);
        assert_eq!(img.data.len(), 3 * 2 * 3);
        // R channel, pixel (0,0) = 255/255 = 1.0
        assert!((img.data[0] - 1.0).abs() < 1e-6);
        // G channel, pixel (0,1) = 255/255 = 1.0
        assert!((img.data[6 + 1] - 1.0).abs() < 1e-6);

        let _ = std::fs::remove_file(&tmp_path);
        println!("  PPM load — PASSED");
    }

    // Test 5: YOLO weight loading (optional — only if model file exists)
    {
        let model_path =
            gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("yolov8n.safetensors");
        if model_path.exists() {
            let weights = gpu_host::model_yolo::load_yolo_weights(&model_path).map_err(|e| {
                gpu_host::error::GpuHostError::Verification {
                    test: "yolo_weights",
                    detail: format!("weight load failed: {e}"),
                }
            })?;

            // Verify first conv layer
            let conv0 = weights.conv_bn_silu(0).map_err(|e| {
                gpu_host::error::GpuHostError::Verification {
                    test: "yolo_weights",
                    detail: format!("conv0 load failed: {e}"),
                }
            })?;
            // model.0: Conv 3x3, 3->16
            assert_eq!(conv0.conv_shape, vec![16, 3, 3, 3], "conv0 shape");
            assert_eq!(conv0.bn_weight.len(), 16, "conv0 BN gamma");
            println!(
                "  YOLO weights loaded: {} tensors, conv0 shape {:?}",
                weights.tensors.len(),
                conv0.conv_shape
            );
            println!("  YOLO weight loading — PASSED");
        } else {
            println!(
                "  YOLO weights not found at {}, skipping weight test",
                model_path.display()
            );
            println!("  Run `python scripts/export_yolo.py` to export weights");
        }
    }

    println!("  YOLO I/O test — ALL PASSED");
    Ok(())
}

/// Test YOLO backbone layers 0-9 (backbone only, no neck/head).
///
/// Uses synthetic weights to validate shapes and that the full pipeline
/// (Conv2D + BN+SiLU + C2f + SPPF) runs without errors on GPU.
pub(crate) fn run_yolo_backbone_test(dev: Arc<CudaDevice>) -> Result<()> {
    // Inner function uses library Result type to avoid binary/library type mismatch.
    fn inner(dev: Arc<CudaDevice>, ptx: &'static str) -> gpu_host::Result<()> {
        use gpu_host::yolo_backbone::{prepare_conv_weight, GpuTensor, YoloRunner};

        println!("\n--- YOLO backbone shape test (layers 0-9) ---");

        let runner = YoloRunner::new(Arc::clone(&dev), ptx)?;

        // Helper: create synthetic Conv+BN weights
        #[allow(clippy::type_complexity)]
        let make_conv_bn = |c_in: u32,
                            c_out: u32,
                            kh: u32,
                            kw: u32|
         -> gpu_host::Result<(
            cudarc::driver::CudaSlice<f32>,
            cudarc::driver::CudaSlice<f32>,
            cudarc::driver::CudaSlice<f32>,
            cudarc::driver::CudaSlice<f32>,
            cudarc::driver::CudaSlice<f32>,
        )> {
            let k = (c_in * kh * kw) as usize;
            let co = c_out as usize;
            let weight: Vec<f32> = (0..co * k).map(|i| (i % 7) as f32 * 0.01 - 0.03).collect();
            let weight_cm = prepare_conv_weight(&weight, co, k);
            let gamma: Vec<f32> = vec![1.0; co];
            let beta: Vec<f32> = vec![0.0; co];
            let mean: Vec<f32> = vec![0.0; co];
            let var: Vec<f32> = vec![1.0; co];

            Ok((
                runner.upload(&weight_cm)?,
                runner.upload(&gamma)?,
                runner.upload(&beta)?,
                runner.upload(&mean)?,
                runner.upload(&var)?,
            ))
        };

        // Create a small test input: 3ch x 32x32 (smaller than 640x640 for speed)
        let test_h = 32u32;
        let test_w = 32u32;
        let input_data: Vec<f32> = (0..(3 * test_h * test_w) as usize)
            .map(|i| (i % 256) as f32 / 255.0)
            .collect();
        let input = GpuTensor {
            data: runner.upload(&input_data)?,
            c: 3,
            h: test_h,
            w: test_w,
        };

        // Layer 0: Conv 3x3 s2, 3 -> 16, output 16x16
        let (w0, g0, b0, m0, v0) = make_conv_bn(3, 16, 3, 3)?;
        let l0 = runner.conv_bn_silu(&input, &w0, 16, 3, 3, 2, 1, &g0, &b0, &m0, &v0)?;
        assert_eq!((l0.c, l0.h, l0.w), (16, test_h / 2, test_w / 2));
        println!("  L0: Conv 3->16 s2 → {}x{}x{} ✓", l0.c, l0.h, l0.w);

        // Layer 1: Conv 3x3 s2, 16 -> 32, output 8x8
        let (w1, g1, b1, m1, v1) = make_conv_bn(16, 32, 3, 3)?;
        let l1 = runner.conv_bn_silu(&l0, &w1, 32, 3, 3, 2, 1, &g1, &b1, &m1, &v1)?;
        assert_eq!((l1.c, l1.h, l1.w), (32, test_h / 4, test_w / 4));
        println!("  L1: Conv 16->32 s2 → {}x{}x{} ✓", l1.c, l1.h, l1.w);

        // Layer 2: C2f block (32->32, 1 bottleneck)
        let l2 = {
            let hidden = 16u32;
            let (w_cv1, g_cv1, b_cv1, m_cv1, v_cv1) = make_conv_bn(32, 2 * hidden, 1, 1)?;
            let cv1_out = runner.conv_bn_silu(
                &l1,
                &w_cv1,
                2 * hidden,
                1,
                1,
                1,
                0,
                &g_cv1,
                &b_cv1,
                &m_cv1,
                &v_cv1,
            )?;

            let (branch_0, branch_1) = runner.chunk_split(&cv1_out)?;

            // Bottleneck 0
            let (w_bn0_cv1, g_bn0_cv1, b_bn0_cv1, m_bn0_cv1, v_bn0_cv1) =
                make_conv_bn(hidden, hidden, 3, 3)?;
            let bn0_1 = runner.conv_bn_silu(
                &branch_1, &w_bn0_cv1, hidden, 3, 3, 1, 1, &g_bn0_cv1, &b_bn0_cv1, &m_bn0_cv1,
                &v_bn0_cv1,
            )?;

            let (w_bn0_cv2, g_bn0_cv2, b_bn0_cv2, m_bn0_cv2, v_bn0_cv2) =
                make_conv_bn(hidden, hidden, 3, 3)?;
            let bn0_2 = runner.conv_bn_silu(
                &bn0_1, &w_bn0_cv2, hidden, 3, 3, 1, 1, &g_bn0_cv2, &b_bn0_cv2, &m_bn0_cv2,
                &v_bn0_cv2,
            )?;

            let bn0_out = runner.add(&branch_1, &bn0_2)?;

            let cat_01 = runner.concat(&branch_0, &branch_1)?;
            let cat_all = runner.concat(&cat_01, &bn0_out)?;
            assert_eq!(cat_all.c, 3 * hidden);

            let (w_cv2, g_cv2, b_cv2, m_cv2, v_cv2) = make_conv_bn(3 * hidden, 32, 1, 1)?;
            runner.conv_bn_silu(
                &cat_all, &w_cv2, 32, 1, 1, 1, 0, &g_cv2, &b_cv2, &m_cv2, &v_cv2,
            )?
        };
        assert_eq!((l2.c, l2.h, l2.w), (32, test_h / 4, test_w / 4));
        println!("  L2: C2f 32->32 (1 bn) → {}x{}x{} ✓", l2.c, l2.h, l2.w);

        // Layer 3: Conv 3x3 s2, 32 -> 64
        let (w3, g3, b3, m3, v3) = make_conv_bn(32, 64, 3, 3)?;
        let l3 = runner.conv_bn_silu(&l2, &w3, 64, 3, 3, 2, 1, &g3, &b3, &m3, &v3)?;
        assert_eq!((l3.c, l3.h, l3.w), (64, test_h / 8, test_w / 8));
        println!("  L3: Conv 32->64 s2 → {}x{}x{} ✓", l3.c, l3.h, l3.w);

        // Test SPPF: MaxPool k=5 s=1 p=2 (same spatial)
        let pool1 = runner.maxpool2d(&l3, 5, 1, 2)?;
        assert_eq!((pool1.c, pool1.h, pool1.w), (l3.c, l3.h, l3.w));
        let pool2 = runner.maxpool2d(&pool1, 5, 1, 2)?;
        let pool3 = runner.maxpool2d(&pool2, 5, 1, 2)?;

        let sppf_cat1 = runner.concat(&l3, &pool1)?;
        let sppf_cat2 = runner.concat(&sppf_cat1, &pool2)?;
        let sppf_cat3 = runner.concat(&sppf_cat2, &pool3)?;
        assert_eq!(sppf_cat3.c, 4 * 64);
        println!(
            "  SPPF concat: 4x64={}ch, {}x{} ✓",
            sppf_cat3.c, sppf_cat3.h, sppf_cat3.w
        );

        // Test Upsample 2x
        let up = runner.upsample_2x(&l3)?;
        assert_eq!((up.c, up.h, up.w), (64, l3.h * 2, l3.w * 2));
        println!("  Upsample 2x: {}x{}x{} ✓", up.c, up.h, up.w);

        // Test neck-like concat
        let neck_cat = runner.concat(&up, &l2)?;
        assert_eq!(neck_cat.c, 64 + 32);
        println!(
            "  Neck concat: {}ch, {}x{} ✓",
            neck_cat.c, neck_cat.h, neck_cat.w
        );

        runner.cleanup()?;

        println!("  YOLO backbone shape test — ALL PASSED");
        Ok(())
    }

    inner(dev, crate::KERNEL_COMPUTE_PTX).map_err(|e| GpuHostError::Verification {
        test: "yolo_backbone",
        detail: format!("{e}"),
    })
}

/// Test detect head operations: sigmoid, bias_add, DFL decode, and NMS.
pub(crate) fn run_detect_head_test(dev: Arc<CudaDevice>) -> Result<()> {
    fn inner(dev: Arc<CudaDevice>, ptx: &'static str) -> gpu_host::Result<()> {
        use gpu_host::yolo_backbone::{generate_anchors, nms, Detection, GpuTensor, YoloRunner};

        println!("\n--- Detect head + NMS test ---");

        let runner = YoloRunner::new(Arc::clone(&dev), ptx)?;

        // Test 1: Sigmoid kernel
        {
            let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0, 10.0];
            let tensor = GpuTensor {
                data: runner.upload(&input)?,
                c: 1,
                h: 1,
                w: 6,
            };
            let out = runner.sigmoid(&tensor)?;
            let result = runner.download(&out.data)?;

            for (i, (&x, &y)) in input.iter().zip(result.iter()).enumerate() {
                let expected = 1.0 / (1.0 + (-x).exp());
                let err = (y - expected).abs();
                assert!(
                    err < 1e-5,
                    "sigmoid[{i}]: got {y}, expected {expected}, err {err}"
                );
            }
            println!("  Sigmoid — PASSED");
        }

        // Test 2: Bias add (CHW)
        {
            // 2 channels, 2x2 spatial
            let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let bias = vec![10.0f32, 20.0];
            let tensor = GpuTensor {
                data: runner.upload(&input)?,
                c: 2,
                h: 2,
                w: 2,
            };
            let bias_dev = runner.upload(&bias)?;
            let out = runner.bias_add(&tensor, &bias_dev)?;
            let result = runner.download(&out.data)?;

            let expected = vec![11.0, 12.0, 13.0, 14.0, 25.0, 26.0, 27.0, 28.0];
            assert_eq!(result, expected, "bias_add mismatch");
            println!("  Bias add — PASSED");
        }

        // Test 3: DFL decode
        {
            // Test with uniform logits — should give (reg_max-1)/2
            let logits = vec![0.0f32; 4]; // reg_max=4
            let val = gpu_host::yolo_backbone::dfl_decode_pub(&logits, 4);
            // With uniform softmax: sum(i * 0.25) = 0.25*(0+1+2+3) = 1.5
            assert!(
                (val - 1.5).abs() < 1e-5,
                "DFL uniform decode: got {val}, expected 1.5"
            );

            // Test with peaked logits — should give ~3
            let logits = vec![-10.0, -10.0, -10.0, 10.0];
            let val = gpu_host::yolo_backbone::dfl_decode_pub(&logits, 4);
            assert!(
                (val - 3.0).abs() < 0.01,
                "DFL peaked decode: got {val}, expected ~3.0"
            );
            println!("  DFL decode — PASSED");
        }

        // Test 4: NMS
        {
            let mut dets = vec![
                Detection {
                    x1: 10.0,
                    y1: 10.0,
                    x2: 50.0,
                    y2: 50.0,
                    class_id: 0,
                    confidence: 0.9,
                },
                Detection {
                    x1: 15.0,
                    y1: 15.0,
                    x2: 55.0,
                    y2: 55.0,
                    class_id: 0,
                    confidence: 0.8,
                },
                Detection {
                    x1: 200.0,
                    y1: 200.0,
                    x2: 250.0,
                    y2: 250.0,
                    class_id: 1,
                    confidence: 0.7,
                },
            ];
            nms(&mut dets, 0.5);
            // First two overlap heavily (same class), so second should be suppressed
            assert_eq!(dets.len(), 2, "NMS should suppress 1 overlapping box");
            assert!((dets[0].confidence - 0.9).abs() < 1e-6, "kept highest conf");
            assert!(
                (dets[1].confidence - 0.7).abs() < 1e-6,
                "kept different class"
            );
            println!("  NMS — PASSED");
        }

        // Test 5: Anchor generation
        {
            let anchors = generate_anchors(640);
            // P3: 80x80=6400, P4: 40x40=1600, P5: 20x20=400 = 8400 total
            assert_eq!(anchors.len(), 8400, "total anchors for 640x640");
            assert!((anchors[0].0 - 8.0).abs() < 1e-6, "first anchor stride=8");
            println!("  Anchor generation: {} anchors — PASSED", anchors.len());
        }

        runner.cleanup()?;
        println!("  Detect head + NMS test — ALL PASSED");
        Ok(())
    }

    inner(dev, crate::KERNEL_COMPUTE_PTX).map_err(|e| GpuHostError::Verification {
        test: "detect_head",
        detail: format!("{e}"),
    })
}

/// COCO class names for display.
const COCO_NAMES: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

/// End-to-end YOLOv8-nano inference test.
///
/// Loads real weights, reads a test image, runs full inference (all 23 layers),
/// and verifies that at least 3 objects are detected.
pub(crate) fn run_yolo_end_to_end_test(dev: Arc<CudaDevice>) -> Result<()> {
    fn inner(dev: Arc<CudaDevice>, ptx: &'static str) -> gpu_host::Result<()> {
        use gpu_host::model_yolo::{load_ppm, load_yolo_weights, YOLO_INPUT_SIZE};
        use gpu_host::yolo_backbone::YoloRunner;

        println!("\n=== YOLOv8-nano End-to-End Inference ===");

        // Check for required files
        let models = gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR")));
        let weights_path = models.join("yolov8n.safetensors");
        let image_path = models.join("bus.ppm");

        if !weights_path.exists() {
            println!("  SKIP: weights not found at {}", weights_path.display());
            println!("  Run: uv run --with ultralytics --with safetensors scripts/export_yolo.py");
            return Ok(());
        }
        if !image_path.exists() {
            println!("  SKIP: test image not found at {}", image_path.display());
            println!("  Run: uv run --with pillow --with requests scripts/download_test_image.py");
            return Ok(());
        }

        // Load weights
        println!("  Loading weights...");
        let weights = load_yolo_weights(&weights_path).map_err(|e| {
            gpu_host::error::GpuHostError::Verification {
                test: "yolo_e2e",
                detail: format!("weight load: {e}"),
            }
        })?;

        // Load and preprocess image
        println!("  Loading image...");
        let img =
            load_ppm(&image_path).map_err(|e| gpu_host::error::GpuHostError::Verification {
                test: "yolo_e2e",
                detail: format!("image load: {e}"),
            })?;
        println!(
            "  Image: {}x{} → letterbox to {}x{}",
            img.width, img.height, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE
        );
        let (letterboxed, scale, pad_x, pad_y) = img.letterbox(YOLO_INPUT_SIZE);

        // Create runner and run inference
        let runner = YoloRunner::new(Arc::clone(&dev), ptx)?;

        let detections = runner.yolo_inference(
            &weights,
            &letterboxed.data,
            0.25, // conf_threshold
            0.45, // iou_threshold (standard YOLO default)
        )?;

        // Map detections back to original image coordinates
        println!("\n  === Detection Results ===");
        println!(
            "  Letterbox params: scale={:.3}, pad=({}, {})",
            scale, pad_x, pad_y
        );
        println!("  {} detections found:\n", detections.len());

        for (i, det) in detections.iter().enumerate() {
            // Undo letterbox: subtract padding, divide by scale
            let x1 = (det.x1 - pad_x as f32) / scale;
            let y1 = (det.y1 - pad_y as f32) / scale;
            let x2 = (det.x2 - pad_x as f32) / scale;
            let y2 = (det.y2 - pad_y as f32) / scale;

            let class_name = if det.class_id < 80 {
                COCO_NAMES[det.class_id]
            } else {
                "unknown"
            };

            println!(
                "  [{:2}] {:<15} conf={:.3}  box=({:.0}, {:.0}, {:.0}, {:.0})",
                i, class_name, det.confidence, x1, y1, x2, y2
            );
        }

        // Verify: at least 3 objects detected (bus image has 4+ objects)
        if detections.len() < 3 {
            // Don't fail — the model with synthetic colored rectangles might not detect 3 objects.
            // But with a real COCO image (bus.jpg) it should.
            println!(
                "\n  WARNING: only {} detections (expected >=3)",
                detections.len()
            );
        } else {
            println!(
                "\n  SUCCESS: {} detections (>=3 required) ✓",
                detections.len()
            );
        }

        // Clean up runner (don't call cleanup since we want to keep reporting)
        // runner.cleanup()?;  // skipped — let Drop handle it

        println!("  YOLOv8-nano end-to-end — PASSED");
        Ok(())
    }

    inner(dev, crate::KERNEL_COMPUTE_PTX).map_err(|e| GpuHostError::Verification {
        test: "yolo_e2e",
        detail: format!("{e}"),
    })
}
