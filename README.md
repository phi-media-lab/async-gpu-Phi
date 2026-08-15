# async_gpu — Rust Async/Await on NVIDIA GPUs

[![CI](https://github.com/DaLaw2/async-gpu/actions/workflows/build.yml/badge.svg)](https://github.com/DaLaw2/async-gpu/actions/workflows/build.yml)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE-MIT)

**Write plain Rust. The compiler handles GPU.** Transparent data (`GpuArray<T>` with automatic host-device sync), auto-fused kernels, `Box<dyn Trait>` on GPU, runtime auto-tuning, `par_iter` — all powered by a custom rustc MIR pass that turns standard `async fn` into warp-cooperative state machines.

async_gpu makes GPU programming feel like normal Rust: **async/await runs natively on NVIDIA GPUs**, `Vec`, `HashMap`, `File`, and `thread::spawn` work out of the box, and 245+ compute kernels deliver **GPT-2 inference in 25ms** (8.8x optimized), **YOLOv8-nano object detection**, **Conv2D at 54.8% peak** (Winograd batched GEMM), SGEMM at **90% of cuBLAS**, and **Monte Carlo simulations** (129x throughput).

```rust
// GPU kernel — looks like normal Rust, runs on GPU
#[no_mangle]
pub unsafe extern "gpu-kernel" fn matmul_pipeline(buf: *mut u8, result: *mut u32) {
    use std::fs::File;
    use std::io::{Read, Write};
    use std::thread;

    // Read matrices from files — real std::fs on GPU
    let a = read_matrix(File::open("a.bin").unwrap());  // M×K
    let b = read_matrix(File::open("b.bin").unwrap());  // K×N

    // Matrix multiply — all warps cooperate in parallel
    let mut c = vec![0.0f32; a.rows * b.cols];
    thread::cooperative(|| {
        let wid = thread::current_id() as usize;
        let n_warps = thread::available_parallelism() + 1;
        for row in (wid..a.rows).step_by(n_warps) {
            for col in 0..b.cols {
                let mut sum = 0.0;
                for k in 0..a.cols { sum += a[(row, k)] * b[(k, col)]; }
                c[row * b.cols + col] = sum;
            }
        }
    });

    // Write result — same std::fs, back to a file
    File::create("c.bin").unwrap().write_all(as_bytes(&c)).unwrap();
    println!("[GPU] {}×{} matmul complete", a.rows, b.cols);
}
```

```rust
// Host side — one line launches the entire pipeline
fn main() -> async_gpu::Result<()> {
    async_gpu::gpu::run("matmul_pipeline")
}
```

Kernel entry uses `extern "gpu-kernel"` — no custom attribute macros needed. A **custom rustc MIR pass** auto-applies to all `async fn` on the `nvptx64` target, inserting `bar.warp.sync` + `shfl.sync` at every `.await` point for warp convergence. Standard Rust syntax, standard `Future` trait.

## Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) with nightly toolchain: `rustup toolchain install nightly-2026-06-03`
- nvptx64 target: `rustup target add nvptx64-nvidia-cuda --toolchain nightly-2026-06-03`
- Rust nightly src (for `-Zbuild-std`): `rustup component add rust-src --toolchain nightly-2026-06-03`
- NVIDIA GPU (SM 70+) with CUDA driver (runtime driver sufficient; CUDA toolkit optional)

### Run an Example

Each example is self-contained with automated PTX compilation via `build.rs`:

```bash
git clone https://github.com/DaLaw2/async-gpu.git
cd async-gpu

# Hello GPU — GPU print, file I/O, thread::spawn
cargo run --manifest-path examples/hostcall/hello-gpu/host/Cargo.toml

# Thread Demo — std::thread::spawn on GPU, join results
cargo run --manifest-path examples/std/thread-demo/Cargo.toml

# Vector Math — SAXPY, dot product, softmax
cargo run --manifest-path examples/hostcall/vector-math/host/Cargo.toml

# GPT-2 Inference — full transformer generation
cargo run --release --manifest-path examples/std/gpt2-inference/Cargo.toml
```

<details>
<summary>All 24 examples</summary>

| Example | Description | Toolchain |
|---------|-------------|-----------|
| **Hostcall examples** (`examples/hostcall/`) | | |
| `hello-gpu` | GPU print, file I/O, thread::spawn (`gpu::run_with_output` API) | Stock nightly |
| `async-pipeline` | Warp-cooperative async pipelines (`gpu::run_with_output` API) | Patched rustc |
| `async-io` | Multi-file write pipeline + read-transform-write | Stock nightly |
| `parallel-search` | 32-lane GPU grep with `shfl.sync` warp reduction (`gpu::custom` API) | Stock nightly |
| `vector-math` | SAXPY, dot product, softmax (`gpu::custom` builder API) | Stock nightly |
| `tcp-echo` | GPU-initiated TCP networking (`gpu::custom` + hostcall) | Stock nightly |
| `tokio-offload` | Async kernel launch from tokio runtime | Stock nightly |
| `structured-concurrency` | Block-scoped spawn, oneshot channels, shared memory (`gpu::custom` API) | Stock nightly |
| `gpu-channels` | MPSC channels + `GpuExecutor` multi-task scheduling (`gpu::custom` API) | Stock nightly |
| `warp-cooperative` | Cooperative compute showcase + MIR pass verification (`gpu::custom` API) | Patched rustc |
| **Std / NN API examples** (`examples/std/`) | | |
| `thread-demo` | `std::thread::spawn` on GPU — spawn, join, warp reuse (`gpu::launch` API) | Stock nightly |
| `gpt2-inference` | GPT-2 Small text generation using `nn` module | Stock nightly |
| `yolo-detect` | YOLOv8-nano object detection using `nn` module | Stock nightly |
| `mnist-train` | MNIST MLP training (91.2% accuracy in 5 epochs) | Stock nightly |
| `cifar-train` | CIFAR-10 tiny CNN training with loss convergence | Stock nightly |
| `gpt2-lora` | GPT-2 LoRA fine-tuning on WikiText-2 (ppl 128→16, rank=8) | Stock nightly |
| `mnist-cnn` | MNIST CNN training (96.4% accuracy, 2.62x GPU speedup) | Stock nightly |
| `resnet-cifar` | ResNet-18 pretrained inference (91.3% CIFAR-10) + ONNX inference (91.2%) + full conv training | Stock nightly |
| `gpu-rag` | GPU-Autonomous RAG: 1030-chunk vector search + GPT-2 generation | Stock nightly |
| `diff-physics` | Differentiable 2D spring-mass / N-body gravity (47.1x GPU speedup) | Stock nightly |
| `dynamic-control` | Data-dependent GPU control flow: variable-length gen, early exit, sampling | Stock nightly |
| `graph-algorithms` | GPU BFS + PageRank on RMAT graphs (CSR, 1M+ vertices, 4.3x speedup) | Stock nightly |
| `monte-carlo` | GPU Monte Carlo: Black-Scholes pricing (129x), Pi estimation (12x) | Stock nightly |
| `benchmark` | SGEMM/Conv2D/Attention vs cuBLAS, memory bandwidth, GPT-2 profiling | Stock nightly |

</details>

### Patched Toolchain (for async warp convergence)

Most examples work with stock nightly. The `async-pipeline` and `warp-cooperative` examples need the MIR pass: `bash scripts/build-toolchain.sh` (Linux) or `.\scripts\build-toolchain.bat` (Windows). Requires ~30GB disk, cmake, ninja, clang/gcc. The example `build.rs` auto-detects it.

## Feature Matrix

| Category | Feature | Description |
|----------|---------|-------------|
| **Data** | [`GpuArray<T>`](crates/core/gpu-host/src/gpu_array.rs) | Transparent host-device data — 4-state residency, auto sync, zero-copy below 64 KiB |
| **Data** | [`SharedRef` / `GlobalRef`](crates/core/gpu-runtime/src/tiered_mem.rs) | Tiered GPU memory — address-space-aware pointers emitting `ld.shared`/`ld.global` |
| **Data** | [`GpuHashMap`](crates/core/gpu-runtime/src/collections.rs) | Fixed-capacity lock-free GPU hash map (CAS-based concurrent insert/get) |
| **Runtime** | `gpu::run` / `launch` / `custom` | One-liner, pure-compute, and builder kernel launch APIs |
| **Runtime** | `extern "gpu-kernel"` | Native GPU entry — no proc macros needed |
| **Runtime** | Real `std` on GPU | `Vec`, `String`, `HashMap`, `Mutex`, `println!`, `File`, `stdin` |
| **Runtime** | `std::thread::spawn` | Warp-as-thread with `JoinHandle::join()` and warp reuse |
| **Runtime** | GPU async/await | `async fn` with warp-cooperative state machines (MIR pass) |
| **Runtime** | [Structured concurrency](examples/hostcall/structured-concurrency/) | `BlockScope`, `GridScope`, [unified channels](crates/core/gpu-runtime/src/unified_channel.rs) (auto shared/global transport) |
| **Runtime** | [`AutoScheduler`](crates/core/gpu-host/src/scheduler.rs) | Unified CPU/GPU work routing — `par_map` auto-selects by data size |
| **Runtime** | [`par_iter`](crates/core/gpu-host/src/memory.rs) | GPU parallel iterators — `map`, `filter`, `fold`, `collect`, `zip` on GPU slices |
| **Runtime** | [Cross-block work dispatch](crates/core/gpu-runtime/src/grid_work.rs) | `GridScope` coordinator/worker pattern without cooperative launch |
| **Safety** | GPU panic transparency | `panic!`/`unwrap`/`assert!` with `GpuKernelResult` block/warp/lane metadata |
| **Safety** | GPU generics | `fn kernel<T: Add + Copy>` — full trait system + `GpuReducible`/`GpuTransformable` on GPU |
| **Safety** | Dynamic dispatch | [`&dyn Trait`](crates/kernel/gpu-kernel-test/src/lib.rs), `Box<dyn Trait>`, `Box<dyn Fn>`, vtable + Drop — all on GPU |
| **Safety** | Type-level safety | `DisjointSlice<T>` race-freedom, `WarpIndex<'scope>` — [safety.rs](crates/core/gpu-runtime/src/safety.rs) |
| **Safety** | GPU coroutines | [`GpuGenerator`](crates/core/gpu-runtime/src/generator.rs) with `yield_value()` / `resume_warp()` — warp-cooperative |
| **Safety** | Compile-time cost model | ptxas-based `KernelResources` / `SmConfig` / `OccupancyLevel` with [`KernelWarning`](crates/core/gpu-host/src/resource_report.rs) diagnostics |
| **Perf** | [`AutoTuner`](crates/core/gpu-host/src/auto_tune.rs) | Warmup-based block-size search + `TuningCache` — 1.4x on compute-bound kernels |
| **Perf** | [Tape-level fusion](crates/core/gpu-host/src/nn/fusion.rs) | Autograd `FusionPlan` — greedy longest-match (MatmulBiasGelu, ElemAddLayerNorm) |
| **Perf** | [Gradient checkpointing](crates/core/gpu-host/src/nn/autograd/checkpoint.rs) | Trade compute for memory — re-executes forward during backward |
| **Debug** | [`FlightRecorder`](crates/core/gpu-runtime/src/flight_recorder.rs) | Mapped-memory ring buffer — fire-and-forget GPU trace events for post-mortem |
| **Debug** | [`#[gpu_test]`](crates/test/gpu-test-macro/) | GPU test macro — `#[gpu_test]` with custom thread/grid config |
| **I/O** | File I/O + TCP | `std::fs::File`, GPU-initiated TCP networking via hostcall |
| **I/O** | Hostcall protocol | ROCm-inspired lock-free GPU-host RPC (TLA+ verified, 367M states) |
| **Compute** | Monte Carlo / Graph / Physics | Black-Scholes 129x, PageRank 4.3x, N-body 47.1x over CPU |
| **ML** | GPT-2 + YOLOv8 + ResNet-18 | Full inference, KV cache, DFL decode, NMS — pure Rust PTX |
| **ML** | Autograd + ONNX + INT4 | Tape-based AD, 43 ONNX ops, W4A16 quantized inference |
| **ML** | [hashbrown on GPU](crates/kernel/gpu-kernel-test/src/lib.rs) | Third-party `#![no_std]` crates with internal `&dyn FnMut` work unmodified on GPU |

<details>
<summary>Additional feature details</summary>

**GPU Compute Kernels (pure Rust inline PTX):** SGEMM (f32 FMA + f16 Tensor Core + INT8 dp4a), FlashAttention (tiled online softmax, causal, KV cache), Conv2D (im2col + GEMM + Winograd F(4x4)), BatchNorm+SiLU (fused), LayerNorm, GELU, Softmax, MaxPool2D, Upsample, Embedding — 245+ kernels total.

**Additional ML features:**
- ResNet-18 inference: Pretrained CIFAR-10 (91.3%), full conv training — [resnet-cifar](examples/std/resnet-cifar/)
- GPU-RAG pipeline: 1030-chunk vector search + GPT-2 generation — [gpu-rag](examples/std/gpu-rag/)
- LoRA fine-tuning: GPT-2 LoRA on WikiText-2 (ppl 128 to 16, rank=8) — [gpt2-lora](examples/std/gpt2-lora/)

**Additional compute patterns:**
- SAXPY / dot product / softmax — [vector-math](examples/hostcall/vector-math/)
- Warp-parallel search (32-lane GPU grep with `shfl.sync`) — [parallel-search](examples/hostcall/parallel-search/)
- Dynamic control flow (variable-length generation, top-k sampling, early exit) — [dynamic-control](examples/std/dynamic-control/)
- Kernel fusion (fused GEMM+bias+GELU in a single kernel launch) — [gpu-rag](examples/std/gpu-rag/) `--bench-fused`
- GPU benchmarks (SGEMM/Conv2D/Attention vs cuBLAS, memory bandwidth profiling) — [benchmark](examples/std/benchmark/)

**Additional runtime features:**
- GPU async executor: `GpuExecutor` — multi-task scheduling on GPU — [gpu-channels](examples/hostcall/gpu-channels/)
- Parallel iterator: `par_iter` — `map`, `filter`, `fold`, `collect`, `zip` on GPU slices
- Unified channels: auto-selects shared (block scope) vs global (grid scope) transport
- AutoScheduler: `par_map` auto-routes to CPU or GPU based on data size
- Tokio integration: `AsyncGpuRuntime`, `GpuTask`, non-blocking kernel launch — [tokio-offload](examples/hostcall/tokio-offload/)

</details>

## Performance

| Metric | Value |
|--------|-------|
| GPT-2 forward (seq=128, optimized) | **25.1ms** (8.8x over baseline) |
| SGEMM 4096^3 | **2,691 GFLOPS** (90% of cuBLAS) |
| Conv2D 3x3 (Winograd batched GEMM) | **2,753 GFLOPS** (54.8% peak, YOLO P4 shape) |
| Flash Attention V3 (seq=512, causal) | **559 GFLOPS** (47-60% of cuDNN FA2) |
| Auto-tuning (compute-bound kernel) | **1.4x** best-vs-worst, 16% free speedup vs default |
| Dyn dispatch (`&dyn Trait` on GPU) | **<1.15x** overhead vs monomorphized (near-zero per-call) |
| Monte Carlo Black-Scholes | **129x** throughput over CPU |
| N-body gravity (4096 particles) | **47.1x** GPU vs CPU |
| MNIST MLP training (5 epochs) | **5.6x** GPU speedup |

<details>
<summary>Full benchmark results</summary>

**Inference** (RTX 3060, SM 86):

| Metric | Value |
|--------|-------|
| GPT-2 per-token f32 FMA (KV cache) | ~68ms/token |
| GPT-2 per-token f16 MMA (Tensor Core) | ~26ms/token (2.18x over f32 FMA) |
| YOLOv8-nano inference | 374ms, 34 detections on 640x640 |
| ResNet-18 pretrained (CIFAR-10) | 91.3% accuracy, 16.0ms/image |
| Compute pipeline speedup | 1.91x vs multi-launch |
| N-body gravity (4096 particles) | 47.1x GPU vs CPU |
| **ONNX Runtime** (ResNet-18, 48 nodes) | 42ms/inference, 91.2% CIFAR-10 (matches ORT) |
| **ONNX Runtime** (GPT-2, 1107 nodes) | 150ms/forward pass, text generation works |
| **ONNX Runtime** (MobileNetV2, 209 nodes) | 409ms/inference, 1000-class output verified |
| **INT4 GPT-2** (W4A16 quantized) | 43ms/token, 7.5x memory reduction (45MB vs 340MB) |
| **GPU PageRank** (1M vertices, 16M edges) | 4.3x speedup over CPU (scale=22) |
| **GPU Monte Carlo** (Black-Scholes, f32) | 129x throughput speedup, 0.004% error |

**Kernel Performance vs cuBLAS / cuDNN** (NVIDIA A2 SM 86 unless noted):

| Kernel | async-gpu | cuBLAS/cuDNN | % of Reference | Improvement |
|--------|-----------|-------------|----------------|-------------|
| **GPT-2 forward** (seq=128) | **25.1ms** | ~20ms est. | — | **8.8x** over baseline |
| **GPT-2 forward** (seq=128)^1 | **39.4ms** | — | — | **5.6x** over baseline |
| **SGEMM** (4096^3) | 2,691 GFLOPS | 2,987 GFLOPS | **90%** | 17.1x over v1 |
| **Flash Attention V3** (seq=512, causal)^1 | 559 GFLOPS | ~1,000-1,200 est. | **47-60%** | V3 rewrite |
| **Flash Attention** (seq=64) | 0.056ms | 0.030ms (FA2) | **54%** | 8.2x over v1 |
| **Flash Attention** (seq=128) | 0.134ms | 0.048ms (FA2) | **36%** | 9.3x over v1 |
| **Conv2D** (128->128, 28^2) | 425 GFLOPS | 522 GFLOPS | **81%** | 3.9x over v1 |
| **Conv2D** (256->256, 14^2) | 556 GFLOPS | 243 GFLOPS | **229%** | 4.9x over v1 |
| **Conv2D Winograd** (YOLO P4, 128->128 @ 40^2) | 2,753 GFLOPS | 5,027 peak | **54.8%** | Winograd batched GEMM |
| **Conv2D Winograd** (YOLO P3, 64->64 @ 80^2) | 2,273 GFLOPS | 5,027 peak | **45.2%** | F(4x4) dispatch |
| **LayerNorm** (128x768)^1 | 199 GB/s eff. | 200 GB/s peak | **~100%** | 6.6x over v1 |
| **Fused LN+residual**^1 | 154 GB/s eff. | — | — | 2.01x speedup |
| **elementwise_add** (in-place)^1 | 160 GB/s | 192 GB/s peak | **83%** | 1.5x over PyTorch |

^1 Measured on GTX 1660 (SM 75, 192 GB/s). FA V3 % is vs estimated cuDNN FA2 on SM 75 (no tensor cores).

**Training** (GPU matmul + autograd tape):

| Example | CPU | GPU | Speedup | Accuracy |
|---------|-----|-----|---------|----------|
| MNIST MLP (60K, 5 epochs) | 44.0s (8.8s/ep) | 7.8s (1.6s/ep) | **5.6x** | 91.2% |
| MNIST CNN (60K, 5 epochs) | 541.3s (107.5s/ep) | 206.7s (41.3s/ep) | **2.62x** | 96.4% |
| CIFAR-10 CNN (2K, 10 epochs) | 6.5s (0.7s/ep) | 7.2s (0.7s/ep) | 0.90x | 27.2%/21.0% |
| Mini-ResNet (2K, 20 ep, full conv bwd) | — | 468.9s | — | 32.1% |

MNIST MLP shows clear GPU advantage for matmul-heavy workloads (batch=64, 784x128 GPU GEMM). MNIST CNN uses full GPU conv2d backward (im2col + matmul + col2im) — 2.62x over CPU. CIFAR-10 GPU produces **identical** loss/accuracy curves to CPU. All use `--cpu` for comparison.

**Hostcall**:

| Metric | Value |
|--------|-------|
| Round-trip (1 thread) | ~42-101 us, 10-15K calls/s |
| Round-trip (32 threads) | ~1.1 ms, 20-23K calls/s |

</details>

<details>
<summary>Progressive examples</summary>

Six snippets, increasing complexity. Each is extracted from a runnable example or working API.

**1. Hello GPU** -- spawn threads on GPU, join results. Looks like normal Rust. ([hello-gpu](examples/hostcall/hello-gpu/), [thread-demo](examples/std/thread-demo/))

```rust
// GPU kernel — spawn two warps as threads, join results
#[no_mangle]
pub unsafe extern "gpu-kernel" fn thread_spawn_test(result: *mut u32) {
    thread::gpu_main(|| {
        let h1 = thread::spawn(|| 42u32);
        let h2 = thread::spawn(|| 99u32);
        let (r1, r2) = (h1.join(), h2.join());
    });
}

// Host — one line launches the kernel and downloads results
let result: Vec<u32> = gpu::launch("thread_spawn_test", 4, 128)?;
assert_eq!(result[0], 42);
```

**2. Transparent Data** -- `GpuArray<T>` with automatic host-device sync. No manual uploads. ([gpu_array.rs](crates/core/gpu-host/src/gpu_array.rs))

```rust
use async_gpu::GpuArray;

// Create from a Vec — data lives on host only
let mut data = GpuArray::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);

// Deref transparently reads host data — no API ceremony
assert_eq!(data[0], 1.0);

// DerefMut marks host as dirty — next kernel launch auto-uploads
data[0] = 42.0;

// Pass to kernel — runtime ensures device residency automatically
// Below 64 KiB: zero-copy over PCIe. Above: explicit VRAM copy.
let result = data.map_gpu(ptx::KERNEL, "vector_add", 256)?;
```

**3. Auto-Tuning** -- `AutoTuner` finds optimal launch params via warmup-based search. ([auto_tune.rs](crates/core/gpu-host/src/auto_tune.rs))

```rust
use gpu_host::auto_tune::{AutoTuner, TuningCache};

let cache = TuningCache::new();
let tuner = AutoTuner::new();

// Tune block size for a kernel — benchmarks [32, 64, 128, 256, 512, 1024]
let best = tuner.tune_block_size(
    65536,             // problem size (elements)
    None,              // optional occupancy filter from KernelResources
    &|block_size| {    // benchmark: launch kernel, measure elapsed
        launch_kernel(block_size);
        Some(elapsed)
    },
);

// Cache result — keyed by (kernel, problem-size bucket, device)
if let Some(result) = best {
    cache.insert_config("vector_add", 65536, 0, result);
}
// Typical result: 1.4x speedup on compute-bound kernels
```

**4. Dynamic Dispatch on GPU** -- `Box<dyn Trait>` with vtable, closures, and Drop — all on GPU. ([gpu-kernel-test](crates/kernel/gpu-kernel-test/src/lib.rs))

```rust
// Trait works on GPU — vtable lives in .global memory
trait Compute { fn eval(&self, x: u32) -> u32; }

struct Linear { slope: u32, intercept: u32 }
impl Compute for Linear {
    fn eval(&self, x: u32) -> u32 { self.slope * x + self.intercept }
}

// &dyn Trait — stack-allocated data, vtable dispatch
let obj: &dyn Compute = &Linear { slope: 2, intercept: 10 };
assert_eq!(obj.eval(16), 42);  // indirect call via vtable

// Box<dyn Trait> — heap-allocated via GPU allocator
let boxed: Box<dyn Compute> = Box::new(Linear { slope: 3, intercept: 0 });
assert_eq!(boxed.eval(14), 42);  // vtable dispatch + heap alloc

// Box<dyn Fn> — closures on GPU
let f: Box<dyn Fn(u32) -> u32> = Box::new(|x| x + 5);
assert_eq!(f(100), 105);

// Overhead: <1.15x vs monomorphized (near-zero per-call)
```

**5. Cooperative Compute** -- all warps process data in parallel, then return to sequential. ([warp-cooperative](examples/hostcall/warp-cooperative/))

```rust
// All warps cooperate: each handles rows where row % n_warps == warp_id
thread::cooperative(|| {
    let wid = thread::current_id() as usize;
    let n_warps = thread::available_parallelism() + 1;
    for row in (wid..M).step_by(n_warps) {
        for col in 0..N {
            let mut sum = 0.0;
            for k in 0..K { sum += a[row * K + k] * b[k * N + col]; }
            c[row * N + col] = sum;
        }
    }
});
```

**6. Structured Concurrency Pipeline** -- scoped spawn, oneshot channels, lifetime-bounded shared memory. ([structured-concurrency](examples/hostcall/structured-concurrency/))

```rust
// Block-scoped producer-consumer on GPU — memory freed when scope exits
block_scope(|scope| {
    let data: &mut [u32] = scope.alloc::<u32>(64);   // shared memory
    let (tx, rx) = block_oneshot(scope.alloc_slot()); // oneshot channel

    scope.spawn(move || {                 // producer warp
        for i in 0..64 { data[i] = i; }
        tx.send(1);                       // signal completion
    });
    scope.spawn(move || -> u32 {          // consumer warp
        let _signal = rx.recv_spin();     // wait for data
        data.iter().sum()                 // sum = 2016
    });
});
```

</details>

<details>
<summary>GPT-2 inference details</summary>

End-to-end transformer inference — real HuggingFace weights, custom BPE tokenizer, 12 transformer layers, KV-cached autoregressive generation. Available via both the raw kernel API and the composable **nn module** (`Linear`, `LayerNorm`, `MultiHeadAttention`, `Gpt2Model`). All compute kernels in pure Rust with inline PTX, no CUDA C++ or cuBLAS.

```
--- Greedy autoregressive generation (with KV cache) ---
  [1/3] Prompt: "The capital of France is" -> 5 tokens, generating 50
  Generated: " the capital of the French Republic, and the capital of
  the French Republic is the capital of the French Republic..."
  Time: 3400ms total, 68ms/token  (2.07x faster with KV cache)
  PASSED (50 tokens, no NaN)
```

GPU compute kernels: GEMM (f32 FMA + f16 Tensor Core MMA with split-K + INT8 dp4a), FlashAttention (tiled online softmax, causal masking, KV cache), LayerNorm, GELU, Softmax, Embedding, fused GEMM+bias+activation — all in Rust inline PTX.

Standalone example: `cargo run --manifest-path examples/std/gpt2-inference/Cargo.toml --release` (requires `models/model.safetensors` — run `bash scripts/download-models.sh`).

### GPU-Autonomous File Transform

Single kernel launch, 8-step I/O pipeline + compute — zero CPU intervention:

```
--- File Transform Pipeline ---
  16-state WarpFuture: open->read->transform->open->write->close->close->print
  1024 bytes: ASCII case toggled correctly, Elapsed: 4.183ms
```

### GPU-Autonomous Vector Search

20-state WarpFuture: open database, read vectors, cosine similarity across 32 lanes, merge top-K via warp shuffle, write results — one kernel launch:

```
--- Vector Similarity Search ---
  rank 1: id=42 score=1.0000, rank 2: id=82 score=0.2103, rank 3: id=18 score=0.0913
  Elapsed: 6.434ms
```

### Compute Pipeline (1.91x vs multi-launch)

Newton-Raphson sqrt with warp-cooperative convergence — single-launch async (24.1 us) vs multi-launch CUDA-style (46.1 us, 3 separate kernels).

</details>

<details>
<summary>YOLOv8-nano detection details</summary>

End-to-end real-time object detection — SafeTensors weights, 23-layer backbone/neck, decoupled detect head with DFL decode + NMS. All compute kernels in pure Rust inline PTX, no cuDNN or cuBLAS.

```
--- YOLOv8-nano end-to-end inference ---
  Image: 810x1080 → letterbox 640x640
  7 detections found:
  [ 0] person          conf=0.931  box=(672, 391, 810, 877)
  [ 1] person          conf=0.925  box=(222, 409, 344, 856)
  [ 2] person          conf=0.878  box=(53, 400, 243, 905)
  [ 3] bus             conf=0.865  box=(32, 237, 797, 747)
  [ 4] person          conf=0.508  box=(1, 548, 59, 877)
  [ 5] car             conf=0.469  box=(686, 505, 778, 680)
  [ 6] tie             conf=0.298  box=(135, 477, 152, 518)
```

GPU compute kernels: Conv2D (im2col + GEMM), BatchNorm+SiLU (fused elementwise), MaxPool2D, Upsample (nearest-neighbor), C2f blocks, SPPF, Sigmoid — all in Rust inline PTX.

Standalone example: `cargo run --manifest-path examples/std/yolo-detect/Cargo.toml --release` (requires `models/yolov8n.safetensors` — run `uv run --with ultralytics --with safetensors scripts/export_yolo.py`).

</details>

<details>
<summary>std on GPU example</summary>

GPU kernels can use **actual Rust standard library** types and traits — not custom wrappers:

```rust
// This runs on the GPU, using real std
println!("[GPU] Hello from Rust std on GPU!");

let mut data = Vec::new();
for i in 0..10 {
    data.push(format!("item-{}", i));
}

let file = std::fs::File::create("gpu_output.txt")?;
std::io::Write::write_all(&mut &file, b"Written from GPU")?;

let line = std::io::stdin().lock().lines().next().unwrap()?;
println!("[GPU] Read from stdin: {}", line);
```

This works via a **patched std** (`-Zbuild-std=std`) with a CUDA platform adaptation layer (PAL) that routes `sys` calls through the hostcall protocol.

**What works** (multi-thread safe): `println!`, `format!`, `Vec`, `String`, `Box`, `HashMap`, `Mutex`, `std::fs::File` (create/read/write), `std::io::stdin().read_line()`, `std::thread::spawn` + `JoinHandle::join()`, `Result<T, E>` with `?` operator and `std::io::Error`.

</details>

<details>
<summary>How it works — architecture</summary>

### Host SDK

**One-liner API** (`async_gpu::gpu`):

| Function | Purpose |
|----------|---------|
| `gpu::run("kernel")` | Hostcall-enabled kernel (supports `println!`, file I/O) |
| `gpu::run_with_output("kernel", n)` | Hostcall + output buffer, returns `Vec<T>` |
| `gpu::launch("kernel", n, threads)` | Pure compute with output buffer, no hostcall |
| `gpu::custom("kernel")` | Builder API for multi-argument kernels (`.ptx()`, `.threads()`, `.hostcall()`, `.prepare()`) |

**Core types**:

| Type | Purpose |
|------|---------|
| `GpuRuntime` | Device init, PTX loading, kernel launch, multi-GPU support |
| `GpuArray<T>` | Transparent data with 4-state residency tracking and auto host-device sync |
| `GpuVec<T>` | GPU-backed vector with kernel-side push/pop |
| `Pipeline` | Multi-stage kernel pipeline with automatic dependency tracking |
| `FlightRecorder` | Mapped-memory ring buffer for fire-and-forget GPU trace events |
| `HostcallBuffer` | GPU-host RPC communication (print, file I/O, stdin) |
| `MappedBuffer<T>` | RAII pinned device-mapped memory (auto-freed on drop) |
| `GpuStream` | CUDA stream wrapper for overlapping compute and I/O |
| `GpuContext` | Prepared launch context from `gpu::custom()` — upload, alloc, launch |
| `GpuResult` | Post-launch handle for downloading device buffers |

### Lock-Free Hostcall Protocol

GPU-host communication uses a ROCm-inspired two-stack design over CUDA mapped memory:

- **Free stack**: Available packets for GPU to claim (one CAS per warp)
- **Ready stack**: Filled packets for host to process
- **Per-block sharding**: Reduces CAS contention at scale
- **Sideband buffer**: Separate mapped memory for bulk data beyond the 56-byte packet payload

Formally verified with TLA+ (367M safety states, 337K liveness states, 0 violations). See [`formal/`](formal/).

### Async on GPU

The custom MIR pass auto-applies to **all** `async fn` on the `nvptx64` target — no attributes needed. It inserts `bar.warp.sync` at each `.await` point so all 32 SIMT lanes always agree on the current state.

Standard `async fn` + `.await` is the only path needed — no proc macros required.

### GPU Threading Model

`std::thread::spawn` works on GPU — each warp (32 SIMT lanes) acts as a single thread:

| API | GPU Behavior |
|-----|-------------|
| `thread::spawn(closure)` | Wakes a sleeping warp, assigns closure, returns `JoinHandle` |
| `handle.join()` | Blocks parent warp until child completes, returns result |
| `thread::available_parallelism()` | Returns number of free warps |
| `thread::current()` / `thread::yield_now()` | Thread identity and cooperative yield |

Warp 0 runs `main()`, other warps sleep until `thread::spawn()` wakes them. Warps return to the idle pool after their closure completes, enabling reuse.

</details>

<details>
<summary>Neural network module (nn)</summary>

PyTorch-style composable layers and autograd, running on GPU via the kernel registry:

```rust
use async_gpu::nn::{GpuTensor, KernelRegistry, Module};
use async_gpu::nn::layers::{Linear, LayerNorm, GELU};
use async_gpu::nn::models::gpt2::Gpt2Model;

// Build model from safetensors weights — no raw kernel launches needed
let model = Gpt2Model::from_weights(&weights, config, &registry)?;
let tokens = model.generate(&prompt_tokens, 50)?;
```

**Layers**: `Linear`, `Conv2d`, `LayerNorm`, `BatchNorm2d`, `Embedding`, `MultiHeadAttention`, `GELU`, `SiLU`, `Sigmoid`, `ReLU`, `MaxPool2d`, `Sequential`, `Int4Linear`.

**ONNX Runtime** (`async_gpu::onnx`):
- Load any `.onnx` file via prost protobuf parser (no protoc needed)
- 43 ONNX operators: Conv (incl. grouped/depthwise), MatMul, Gemm, Relu, BatchNorm, LayerNorm, Softmax, Add, Mul, Sub, Reshape, Transpose, Gather, Split, Where, Concat, Identity, GlobalAveragePool, ReduceMean, and more
- `OnnxSession`: initializer caching + weight prepadding for repeated inference
- Graph fusion pass: MatMul+Add+Activation pattern matching
- GPT-2 ONNX text generation verified (150ms/forward, 1107 nodes)
- ResNet-18 ONNX: 91.2% CIFAR-10 accuracy (matches ORT exactly)
- MobileNetV2 ONNX: 209 nodes, 1000-class output, end-to-end verified

**Autograd** (tape-based reverse-mode AD):
- Forward ops automatically record on a thread-local tape when `requires_grad = true`
- `backward()` traverses tape in reverse with chain rule dispatch
- Backward kernels: GELU, SiLU, sigmoid, ReLU, matmul, LayerNorm, BatchNorm (GPU), Conv2d (im2col), MaxPool2d (gradient routing), UpsampleNearest (4-to-1), bias_add, elementwise_add
- Optimizers: SGD (with momentum), Adam
- Losses: cross-entropy, MSE
- Verified via numerical gradient checks (finite differences)

</details>

## Architecture

Two-workspace build model: host crates compile with the standard `x86_64` target, GPU crates compile with `-Zbuild-std=std` targeting `nvptx64-nvidia-cuda`. A custom rustc MIR pass inserts warp-convergence barriers into async state machines. PTX is compiled at build time via `build.rs` and embedded as string constants in the host binary. The hostcall protocol bridges GPU-host I/O over CUDA mapped memory.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full compilation pipeline, crate layout, subsystem internals, and data flow diagrams.

## Crate Map

19 crates organized by layer: **Facade** (1) → **Core** (5) → **Kernel** (4) → **Test** (9).

```
Facade
  crates/async-gpu/        User-facing crate — re-exports GpuArray, AutoScheduler, FlightRecorder, nn, async_rt

Core (host + GPU runtime + support)
  crates/core/gpu-host/    Host SDK: gpu:: API, GpuRuntime, GpuArray, GpuVec, Pipeline, FlightRecorder
    gpu_array.rs           GpuArray<T> — 4-state residency, auto host-device sync, zero-copy <64 KiB
    auto_tune.rs           AutoTuner, TuningCache — warmup-based block-size search
    scheduler.rs           AutoScheduler, CpuScheduler, GpuScheduler — unified work routing
    resource_report.rs     KernelResources, SmConfig, OccupancyLevel, KernelWarning
    nn/                    Neural network: GpuTensor, KernelRegistry, layers, models, autograd, fusion
    onnx_rt/               ONNX Runtime: prost parser, graph executor (43 ops), fusion pass
  crates/core/gpu-runtime/ GPU-side runtime: index, math, warp, block, thread, nn, executor
    scope.rs               BlockScope, GridScope — structured concurrency with lifetime-bounded memory
    tiered_mem.rs           SharedRef/GlobalRef — address-space-aware GPU pointers (ld.shared/ld.global)
    unified_channel.rs     Auto-selects shared vs global transport for channels
    flight_recorder.rs     Fire-and-forget GPU trace ring buffer
    collections.rs         GpuHashMap — lock-free concurrent hash map (CAS-based)
    par_iter.rs            GPU parallel iterators — map, filter, fold, collect, zip
    grid_work.rs           Cross-block coordinator/worker dispatch
    generator.rs           GpuGenerator — warp-cooperative coroutines
    safety.rs              DisjointSlice, WarpIndex — type-level race-freedom
  crates/core/gpu-protocol/ Shared constants: packet layout, service IDs, error codes
  crates/core/gpu-atomics/  System-scope GPU atomics via inline PTX (CAS, shfl, activemask)
  crates/core/gpu-libc/     Minimal libc shim for GPU: routes sys calls to hostcall

Kernel (PTX code compiled for nvptx64)
  crates/kernel/gpu-kernel-core/    Shared helpers + basic kernels
  crates/kernel/gpu-kernel-compute/ Compute kernels (fused ops, physics, transformer, persistent)
  crates/kernel/gpu-kernel-io/      I/O kernels (hostcall, hybrid, async pipeline)
  crates/kernel/gpu-kernel-test/    Test/demo kernels (std, dyn dispatch, hashbrown, par_iter)

Test (9 integration test crates)
  crates/test/               gpu-test-macro (#[gpu_test]), gpu-test-harness, gpu-std-test,
                             async-hostcall-test, async-pipeline-test, embassy-test,
                             gpu-critical-section, multi-warp-test, std-build-test

Other
  rustc-patches/       Custom MIR pass patches for rustc
  examples/hostcall/   10 hostcall examples | examples/std/  14 std/nn examples
  formal/              TLA+ specification and model-checking config
  docs/                ARCHITECTURE.md, CHANGELOG.md, getting-started.md
```

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) — hostcall protocol, MIR pass, runtime internals
- [Changelog](docs/CHANGELOG.md) — version history and shipped stories
- [Getting Started](docs/getting-started.md) — step-by-step first kernel tutorial
- [Composed typed-priority + hostcall E2E](docs/composed-priority-hostcall-e2e.md) —
  typed metadata、reserved credit、mutation oracle 与 cancellation/shutdown 边界
- [Obstacle-event priority stress](docs/obstacle-event-priority-stress.md) —
  cooperative admission/dispatch、non-yielding negative 与 watchdog decisions
- [Priority handle safety GPU litmus](docs/priority-handle-safety-gpu-litmus.md) —
  multi-block stale-waker generation reuse 与 typed `Cancelled→AlreadyJoined`

## Limitations

- **Nightly Rust**: Requires `asm_experimental_arch`, `abi_gpu_kernel`, `-Zbuild-std`. Async warp convergence MIR pass needs patched rustc
- **NVIDIA only**: `nvptx64-nvidia-cuda` target, SM 70+ GPU required
- **Hostcall latency**: ~20-100 us round-trip per call, not suitable for per-element I/O in hot loops
- **Partial std**: `Vec`, `HashMap`, `Mutex`, `File`, `println!` work; `OsRng`/`getrandom`, networking (beyond hostcall TCP) not available
- **f32 + f16 only**: f32 FMA and f16 Tensor Core MMA both supported; BF16/TF32/FP8 not yet implemented
- **Single GPU**: Multi-GPU device selection works, but no cross-device kernel orchestration or peer-to-peer transfers

## Acknowledgements

Inspired by [VectorWare](https://www.vectorware.com/)'s work on [Rust std on GPU](https://www.vectorware.com/blog/rust-std-on-gpu/) and [Async/Await on GPU](https://www.vectorware.com/blog/async-await-on-gpu/).

## License

MIT OR Apache-2.0
