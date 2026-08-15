//! GPU Runtime — facade crate for writing GPU kernels with async hostcall support.
//!
//! Re-exports all necessary GPU-side APIs so kernel authors only need
//! one dependency instead of four.
//!
//! # Usage
//!
//! ```toml
//! [dependencies]
//! gpu-runtime = { path = "../gpu-runtime" }
//! ```
//!
//! ```rust,ignore
//! #![no_std]
//! #![feature(abi_gpu_kernel)]
//!
//! use gpu_runtime::prelude::*;
//!
//! #[no_mangle]
//! pub unsafe extern "gpu-kernel" fn my_kernel(buf: *mut u8, result: *mut u32) {
//!     let msg = b"Hello from GPU!";
//!     gpu_hostcall_print(buf, msg.as_ptr(), msg.len() as u32);
//!     core::ptr::write_volatile(result, 1);
//! }
//! ```

#![no_std]
#![allow(clippy::missing_safety_doc)]
#![cfg_attr(target_arch = "nvptx64", feature(stdarch_nvptx))]
#![cfg_attr(target_arch = "nvptx64", feature(asm_experimental_arch))]

// GPU intrinsic wrappers -- stubs on non-nvptx targets for doc builds.
mod nvptx_shim;

// Re-export sub-crates
pub use gpu_atomics;
pub use gpu_protocol;

// Ensure critical-section is linked (needed for Embassy executor)
extern crate gpu_critical_section;

// ============================================================
// GPU Compute Utilities — public API for kernel authors
// ============================================================

// -- Compute utilities --
/// Thread, block, and grid indexing helpers.
///
/// Safe wrappers around `nvptx` intrinsics. All functions return 0/1 on
/// non-nvptx targets for doc builds.
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::index;
///
/// let tid = index::thread_idx_x();
/// let gid = index::global_thread_idx();
/// let n_threads = index::global_thread_count();
/// if gid < n {
///     output[gid] = input[gid] * 2.0;
/// }
/// ```
pub mod index;

/// GPU math intrinsics — fast approximate f32 math via PTX special function units.
///
/// All functions are safe (no pointers, no thread coordination). They use PTX
/// approximate instructions which are fast (~1-4 cycles) but have limited precision
/// (~1 ULP for most operations).
///
/// On non-nvptx targets, functions return 0.0 (for compilation/doc builds only).
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::math;
///
/// let x = math::exp_f32(-1.0);     // e^(-1) ≈ 0.3679
/// let y = math::sqrt_f32(2.0);     // √2 ≈ 1.4142
/// let z = math::fma_f32(a, b, c);  // a*b + c (single instruction)
/// ```
pub mod math;

/// Warp-level compute primitives — reductions, shuffle variants, vote operations.
///
/// All functions are `unsafe` because they require warp-level coordination:
/// all participating lanes must call the function with the same mask.
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::warp;
///
/// let my_val: f32 = compute_something();
/// let sum = unsafe { warp::reduce_sum_f32(my_val) }; // sum across all 32 lanes
/// let max = unsafe { warp::reduce_max_f32(my_val) }; // max across all 32 lanes
/// ```
pub mod warp;

/// GPU thread pool — `thread::spawn()` maps to warp execution.
///
/// Each GPU warp (32 SIMT lanes) acts as a single logical "thread".
/// Warp 0 runs the user's main function; other warps park until work
/// is assigned via `spawn()`.
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::thread;
///
/// thread::gpu_main(|| {
///     let h = thread::spawn(|| 42u32);
///     assert_eq!(h.join(), 42);
/// });
/// ```
pub mod thread;

/// Block-level compute primitives — synchronization, shared memory, reductions.
///
/// Functions require all threads in the block to participate (unless noted).
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::block;
///
/// unsafe {
///     let smem = block::shared_mem_ptr();
///     // ... write to shared memory ...
///     block::sync();
///     // ... read from shared memory ...
/// }
/// ```
pub mod block;

/// Block-level structured concurrency with lifetime-bounded shared memory.
///
/// Provides `BlockScope` — a scoped concurrency primitive that owns a region
/// of shared memory and guarantees all spawned work completes before the
/// scope exits. The `'scope` lifetime prevents shared memory references from
/// escaping their hardware scope.
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::scope::block_scope;
///
/// block_scope(|scope| {
///     let buf = scope.alloc::<f32>(256);
///     scope.spawn_all(|wid, n_warps| {
///         let mut i = wid as usize;
///         while i < 256 {
///             buf[i] = i as f32;
///             i += n_warps as usize;
///         }
///     });
/// });
/// ```
pub mod scope;

/// Type-level safety primitives for race-free GPU programming.
///
/// Provides [`WarpIndex`], [`DisjointSlice`], and [`WarpHandle`] — compile-time
/// witnesses that prevent data races by construction. Inspired by cuda-oxide's
/// 3-tier safety model, adapted for async-gpu's warp-cooperative execution.
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::scope::block_scope;
/// use gpu_runtime::safety::{WarpIndex, DisjointSlice, WarpHandle};
///
/// block_scope(|scope| {
///     let buf = scope.alloc::<f32>(256);
///     let disjoint = scope.disjoint_slice(buf);
///
///     scope.spawn_all_indexed(|widx, warp| {
///         let my_part = disjoint.get_mut(&widx);
///         for slot in my_part.iter_mut() {
///             *slot = 1.0;
///         }
///     });
/// });
/// ```
pub mod safety;

/// Lifetime-parameterized memory tier types for GPU address spaces.
///
/// Provides [`GpuRef`] — a lifetime-bounded, address-space-aware GPU memory
/// reference that encodes shared vs global at the type level. This enables
/// the compiler to emit `ld.shared`/`st.shared` or `ld.global`/`st.global`
/// instead of generic loads, avoiding runtime address-space resolution.
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::scope::block_scope;
/// use gpu_runtime::tiered_mem::SharedRef;
///
/// block_scope(|scope| {
///     let buf: SharedRef<'_, f32> = scope.alloc_shared::<f32>(256);
///     buf.write(0, 42.0);  // emits st.shared
///     let val = buf.read(0);  // emits ld.shared
/// });
/// ```
pub mod tiered_mem;

/// User-extensible GPU traits for generic kernel algorithms.
///
/// Provides [`GpuReducible`] and [`GpuTransformable`] — traits that enable
/// writing generic kernel functions with user-defined behavior. The compiler
/// monomorphizes each instantiation to type-specific PTX instructions.
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::traits::GpuReducible;
///
/// // Generic reduce — works for any T: GpuReducible
/// #[inline(always)]
/// fn reduce<T: GpuReducible>(data: &[T]) -> T {
///     let mut acc = T::identity();
///     for &x in data {
///         acc = acc.combine(x);
///     }
///     acc
/// }
/// ```
pub mod traits;

/// Cross-block work coordination for GridScope.
///
/// Provides work-dispatch primitives that let a coordinator block distribute
/// work to independently-scheduled worker blocks via global memory. Designed
/// for SM75 without cooperative launch.
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::scope::grid_scope;
/// use gpu_runtime::grid_work::{self, BlockWorkSlot};
///
/// // Coordinator (block 0):
/// unsafe {
///     grid_scope(pool, pool_size, |gscope| {
///         let slots = gscope.alloc_work_slots(num_workers);
///         // dispatch work, wait for results...
///     });
/// }
///
/// // Worker (block N):
/// unsafe { grid_work::grid_worker_loop(slot_ptr); }
/// ```
pub mod grid_work;

/// Neural network building blocks — activation functions and warp-cooperative ops.
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::{math, nn};
///
/// let x: f32 = input[tid];
/// let activated = nn::gelu_f32(x);
/// let clipped = nn::relu_f32(x);
/// ```
pub mod nn;

// -- Hostcall / I/O --
/// Hostcall helpers for GPU-side hostcall protocol operations.
///
/// These functions implement the lock-free two-stack hostcall protocol
/// (pop from free stack, fill packet, push to ready stack, spin-wait for response).
pub mod hostcall;

/// Sideband buffer helpers for bulk data transfer (>56 bytes).
///
/// The sideband buffer is a separate CUDA mapped allocation with a bump allocator.
/// GPU threads allocate regions via atomic fetch_add, write/read data, and use
/// hostcall packets to coordinate with the host.
pub mod sideband;

/// GPU-side buffered print — accumulate messages and flush via sideband.
///
/// Instead of one hostcall per `gpu_hostcall_print()` (~20-100us each),
/// messages are buffered in a per-thread sideband slot and flushed in
/// a single `SERVICE_BULK_PRINT` round-trip.
///
/// # Usage
///
/// ```ignore
/// // At kernel start:
/// print_buffer::init(sideband, thread_count);
///
/// // During kernel:
/// print_buffer::print(buf, sideband, msg.as_ptr(), msg.len() as u32);
///
/// // At kernel end (required!):
/// print_buffer::flush(buf, sideband);
/// ```
pub mod print_buffer;

/// GPU panic handler that sends panic messages via hostcall before trapping.
///
/// # Usage
///
/// 1. Call `gpu_panic_init(buf)` at the start of your kernel to register the
///    hostcall buffer pointer. If not called, panics will trap without sending
///    a message.
///
/// 2. Add `gpu_runtime::panic_handler!();` in your kernel crate to install the
///    panic handler (replaces the `loop {}` handler).
pub mod panic;

// -- Crate-root macros --

/// Macro to install the GPU panic handler provided by gpu-runtime.
///
/// Place `gpu_runtime::panic_handler!();` at the top level of your kernel crate.
/// This replaces the `loop {}` panic handler with one that sends the panic message
/// via hostcall before trapping. Call `gpu_panic_init(buf)` at kernel entry.
#[macro_export]
macro_rules! panic_handler {
    () => {
        #[panic_handler]
        fn _gpu_panic_handler(info: &core::panic::PanicInfo) -> ! {
            unsafe {
                // Format the panic message into a fixed-size buffer
                let mut pbuf = $crate::panic::PanicBuf::new();
                use core::fmt::Write;
                let _ = write!(pbuf, "{}", info);
                let msg = pbuf.as_slice();

                // Write to kernel result buffer (if registered)
                $crate::panic::write_panic_to_result(msg);

                // Send via hostcall (if registered)
                let buf = $crate::panic::panic_buf();
                if !buf.is_null() {
                    $crate::panic::send_panic_hostcall(buf, msg);
                }

                // Signal that this warp has trapped, so BlockScope::join_all()
                // can detect the dead warp instead of spinning forever.
                $crate::panic::set_warp_trapped();

                // Terminate this GPU thread
                #[cfg(target_arch = "nvptx64")]
                core::arch::asm!("trap;", options(noreturn));
                #[cfg(not(target_arch = "nvptx64"))]
                panic!("GPU trap");
            }
        }
    };
}

/// Emit a structured trace event from GPU to host.
///
/// Usage:
/// ```rust,ignore
/// gpu_trace!(buf, INFO, "processing item {}", idx);
/// gpu_trace!(buf, DEBUG, "loop iteration");
/// gpu_trace!(buf, WARN, "buffer nearly full");
/// gpu_trace!(buf, ERROR, "unexpected value");
/// ```
///
/// `buf` is the hostcall buffer pointer. Level is one of DEBUG, INFO, WARN, ERROR.
/// The message is formatted into a fixed-size buffer (max 48 bytes) and sent via
/// SERVICE_TRACE hostcall with thread/block/warp metadata and GPU timestamp.
///
/// When the `gpu-trace` feature is disabled, this macro compiles to a no-op
/// for zero overhead in release builds.
#[cfg(feature = "gpu-trace")]
#[macro_export]
macro_rules! gpu_trace {
    ($buf:expr, DEBUG, $($arg:tt)*) => {
        $crate::_gpu_trace_impl!($buf, $crate::prelude::TRACE_LEVEL_DEBUG, $($arg)*)
    };
    ($buf:expr, INFO, $($arg:tt)*) => {
        $crate::_gpu_trace_impl!($buf, $crate::prelude::TRACE_LEVEL_INFO, $($arg)*)
    };
    ($buf:expr, WARN, $($arg:tt)*) => {
        $crate::_gpu_trace_impl!($buf, $crate::prelude::TRACE_LEVEL_WARN, $($arg)*)
    };
    ($buf:expr, ERROR, $($arg:tt)*) => {
        $crate::_gpu_trace_impl!($buf, $crate::prelude::TRACE_LEVEL_ERROR, $($arg)*)
    };
}

/// No-op version of `gpu_trace!` when `gpu-trace` feature is disabled.
#[cfg(not(feature = "gpu-trace"))]
#[macro_export]
macro_rules! gpu_trace {
    ($buf:expr, $level:ident, $($arg:tt)*) => {
        // Compiled out — zero overhead
        let _ = &$buf;
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _gpu_trace_impl {
    ($buf:expr, $level:expr, $($arg:tt)*) => {{
        let mut tbuf = $crate::panic::PanicBuf::new();
        {
            use core::fmt::Write;
            let _ = write!(tbuf, $($arg)*);
        }
        let msg = tbuf.as_slice();
        let _ = unsafe {
            $crate::hostcall::gpu_hostcall_trace($buf, $level, msg.as_ptr(), msg.len() as u32)
        };
    }};
}

// -- Async / Future --
/// Warp-level Future — SIMT-convergent async on GPU.
///
/// A `WarpFuture` represents an entire warp (32 lanes) executing in lockstep.
/// Unlike `core::future::Future` where each thread has its own state machine,
/// a WarpFuture has ONE state discriminant shared across all 32 lanes via
/// `shfl.sync.idx.b32`. All lanes enter the same match arm on every poll;
/// only per-lane data differs (SIMD semantics).
///
/// # Key Concepts
///
/// - **WarpPoll**: `Ready(T)` or `Pending`, analogous to `core::task::Poll`.
/// - **WarpContext**: Provides `lane_id` and `active_mask`. No `Waker` —
///   warp futures use synchronous spin-poll.
/// - **WarpFuture trait**: `poll_warp(&mut self, wcx: &mut WarpContext) -> WarpPoll<T>`.
/// - **WarpExecutor**: Simple loop that polls a WarpFuture until Ready.
///
/// # Example
///
/// ```rust,ignore
/// // All 32 lanes call this simultaneously
/// let mut future = MyWarpFuture::new(buf);
/// let result = WarpExecutor::run(&mut future);
/// ```
pub mod warp_future;

/// Standard `core::future::Future` wrappers for hostcall operations.
///
/// These types implement `core::future::Future` — they are per-thread,
/// independent, and have NO warp awareness. They can be polled by any
/// single-threaded executor (Embassy, manual spin-poll, etc.).
///
/// The key design insight: inner futures are standard per-thread futures.
/// Warp cooperation is added automatically by the rustc MIR pass for all
/// async fn on nvptx64 (no annotation needed).
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::std_future::GpuPrintFuture;
/// use core::future::Future;
///
/// // Poll manually (single-thread):
/// let mut future = GpuPrintFuture::new(buf, b"Hello!");
/// loop {
///     match Pin::new_unchecked(&mut future).poll(&mut cx) {
///         Poll::Ready(ok) => break,
///         Poll::Pending => { /* yield / nanosleep */ }
///     }
/// }
/// ```
pub mod std_future;

/// Warp-cooperative wrapper for standard `core::future::Future`.
///
/// This is the key proof for warp-future-bridge: standard per-thread futures
/// can be polled warp-cooperatively. Lane 0 polls the inner future, broadcasts
/// the `Poll` discriminant via `shfl.sync`, and all lanes converge.
///
/// This bridges the gap between `core::future::Future` (per-thread) and
/// warp-cooperative execution (SIMT lockstep).
pub mod warp_cooperative;

/// Warp-cooperative sequential executor for two standard futures.
///
/// Simulates two sequential `.await` points: runs F1 to completion
/// (warp-cooperatively), then runs F2 to completion. All lanes
/// stay converged throughout.
///
/// This is the manual proof of what the MIR pass now generates
/// automatically for all async fn on nvptx64.
pub mod warp_sequential;

/// Warp-cooperative Result broadcasting for `? operator` across .await boundaries.
///
/// Extends the warp-cooperative model to handle `Result<T, E>` — when lane 0
/// polls a future that returns `Result`, the discriminant (Ok/Err) and error
/// code are broadcast to all lanes. If Err, all lanes can early-return together.
pub mod warp_result;

// -- Kernel entry --
/// Implicit hostcall injection via device global (`__HOSTCALL_BUF`).
///
/// The host writes the hostcall buffer pointer to a device global before
/// kernel launch. The kernel calls `entry::auto_init()` to read it and
/// initialize all subsystems — no explicit `buf` parameter needed.
///
/// # Example
///
/// ```rust,ignore
/// #[no_mangle]
/// pub unsafe extern "gpu-kernel" fn my_kernel() {
///     gpu_runtime::entry::auto_init();
///     gpu_runtime::thread::gpu_main_poll(|| {
///         println!("Hello from zero-param kernel!");
///     });
/// }
/// ```
pub mod entry;

// -- Infrastructure --
/// Command buffer polling — GPU-side API for host→GPU command channel.
///
/// The host submits commands to a mapped-memory ring buffer; the GPU kernel
/// polls with `cmd_poll()` and acknowledges with `cmd_ack()`.
pub mod cmd;

/// Flight recorder — GPU-side ring buffer for post-mortem trace events.
///
/// Unlike `gpu_trace!()` which sends events to the host via hostcall,
/// the flight recorder writes directly to mapped memory with no round-trip.
/// On kernel crash, the host can dump the last N events for post-mortem analysis.
pub mod flight_recorder;

/// GPU-side async task executor with dynamic spawning.
///
/// Provides a work-stealing executor where warps dequeue tasks from a lock-free
/// MPMC queue, poll type-erased futures, and recycle slots on completion. Tasks
/// can spawn new tasks dynamically via `GpuExecutor::spawn()`.
///
/// # Architecture
///
/// - **WorkQueue**: Bounded FIFO using tagged CAS (same pattern as hostcall protocol)
/// - **TaskSlot**: Fixed-size arena slot with type-erased `poll_fn` + inline future bytes
/// - **Free list**: Tagged CAS stack for slot recycling (no general allocator needed)
/// - **Scheduling**: Lane 0 dequeues, broadcasts task ID via `shfl.sync` to all 32 lanes
///
/// # Safety
///
/// The executor must reside in GPU global memory (device or mapped). All warps
/// entering `run()` must have all 32 lanes active.
pub mod executor;

/// Typed cooperative-priority task contexts and dependency-checked joins.
///
/// The typed API makes priority-inverting `wait_for` edges fail to compile,
/// while keeping task context and hostcall metadata explicit across blocks.
pub mod priority;

/// GPU async channels — oneshot and mpsc for inter-task communication.
///
/// Provides channels for sending values between GPU async tasks running in the
/// executor. Channels use atomic state machines with acquire/release semantics
/// for cross-warp visibility.
///
/// # Channel Types
///
/// - **Oneshot**: Single-value, single-use channel. Sender sends one value,
///   receiver polls until available. No CAS needed (SPSC).
/// - **Mpsc** (planned): Multi-producer, single-consumer ring buffer.
///
/// # Memory Requirements
///
/// Channel slots must reside in GPU global memory (device or mapped).
/// Values must be `Copy` — no Drop support on GPU.
pub mod channel;

/// Block-scoped GPU channels — shared-memory backed with CTA-scope atomics.
///
/// Provides intra-block channels that use CTA-scope atomics (~2-6 cycles)
/// instead of system-scope atomics (~100 cycles), yielding 20-50x latency
/// improvement for communication between warps within the same block.
///
/// # Channel Types
///
/// - **BlockOneshotSlot**: Single-value, single-use channel in shared memory.
/// - **BlockMpscChannel**: Multi-producer, single-consumer ring buffer in shared memory.
///
/// # Safety
///
/// All channel storage must reside in shared memory. The `'scope` lifetime
/// on sender/receiver types prevents references from escaping the block.
/// Values must be `Copy` — no Drop support on GPU.
pub mod block_channel;

/// Unified channel API — auto-selects transport based on scope.
///
/// Provides `ScopedOneshotSender`/`ScopedOneshotReceiver` and
/// `ScopedMpscSender`/`ScopedMpscReceiver` that dispatch to the correct
/// transport (shared memory CTA atomics or global memory system atomics)
/// based on how they were created.
///
/// Instead of choosing between `block_channel` and `channel` explicitly,
/// users call `scope.oneshot()` / `gscope.oneshot()` and the right
/// transport is selected automatically.
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::scope::block_scope;
///
/// // Block scope: auto-selects shared memory + CTA-scope atomics
/// block_scope(|scope| {
///     let (tx, rx) = scope.oneshot::<u32>();
///     scope.spawn(move || unsafe { tx.send(42) });
///     let val = unsafe { rx.recv_spin() }; // Ok(42)
/// });
/// ```
pub mod unified_channel;

/// GPU synchronization primitives — Mutex, MutexGuard.
///
/// Provides cross-warp/cross-block mutual exclusion on GPU using system-scope
/// atomic CAS spin-locks. Designed for protecting shared data structures in
/// global (mapped) memory.
///
/// # Design Notes
///
/// - Uses `sys_cas_u32` for lock acquisition (system-scope CAS)
/// - Uses `sys_store_release_u32` for unlock (release semantics)
/// - Spin-loop includes `nanosleep` yield via `sys_spin_load_acquire_u32`
/// - Safe across warps and blocks (different warps have independent PCs)
/// - **Not recommended for intra-warp use** — warp-cooperative patterns
///   (lane 0 acts, `shfl.sync` broadcasts) are strictly superior
/// - No poisoning semantics (GPU panics trap the thread)
pub mod sync;

/// GPU-side hash map with fixed-capacity open addressing.
///
/// Provides a concurrent insert/lookup map for GPU kernels. Uses
/// atomic CAS for lock-free inserts and atomic loads for lock-free reads.
/// The capacity is fixed at construction time — no resizing is supported
/// (bump allocator constraint).
///
/// # Design
///
/// - Open addressing with linear probing
/// - Keys must be non-zero `u32` (zero is the empty sentinel)
/// - Values are `u32` (kept simple for GPU; wrap in newtype for semantics)
/// - Lock-free concurrent inserts via `sys_cas_u32`
/// - Lock-free concurrent reads via volatile load
/// - No delete support (use tombstone pattern if needed in future)
///
/// # Capacity
///
/// The map is backed by a fixed array of `(key, value)` pairs. Load factor
/// should be kept below 0.7 to avoid excessive probe chain lengths. With
/// linear probing, performance degrades rapidly above 0.7 load factor.
pub mod collections;

/// GPU parallel iterator — lazy, fused, data-parallel iterator chains.
///
/// Provides a Rayon-like parallel iterator API for GPU kernels. Iterator
/// chains are built lazily via adapter methods (`map`, `enumerate`, `zip`)
/// and executed via terminal methods (`for_each`, `collect_into`, `fold`,
/// `sum`). All closures are fused at compile time via monomorphization.
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::par_iter::*;
///
/// let data = unsafe { GpuSlice::from_raw_parts(input_ptr, len) };
/// let out = unsafe { GpuSliceMut::from_raw_parts(output_ptr, len) };
///
/// data.par_iter()
///     .map(|x| x * 2.0)
///     .map(|x| x + 1.0)
///     .collect_into(out);
/// ```
pub mod par_iter;

/// GPU stdio infrastructure — hostcall-backed stdout/stdin for patched std.
///
/// Provides `#[no_mangle]` functions (`gpu_stdout_write`, `gpu_stdin_read`)
/// that the patched Rust std PAL calls for I/O on `nvptx64-nvidia-cuda`.
///
/// # Example
///
/// ```rust,ignore
/// // At kernel entry:
/// gpu_runtime::stdio::stdio_init(hostcall_buf);
///
/// // For buffered printing:
/// gpu_runtime::stdio::stdio_print_buffer_init(buf, sideband, 1);
/// // ... println!() calls routed through print_buffer ...
/// gpu_runtime::stdio::gpu_print_buffer_flush();
/// ```
pub mod stdio;

/// GPU coroutine generators — warp-cooperative generator/yield pattern.
///
/// Provides `GpuGenerator`, a warp-cooperative generator trait that produces
/// a stream of values using warp-cooperative execution with zero buffering.
/// Generators yield values broadcast from lane 0 to all lanes, enabling
/// data-parallel consumption.
///
/// # Key Types
///
/// - `WarpCoroutineState`: Yielded(Y) or Complete(R) — result of resuming
/// - `GpuGenerator`: trait with `resume_warp()` — the generator protocol
/// - `WarpBroadcast`: trait for lane-0-to-all broadcast (shfl.sync based)
/// - `GeneratorTask`: adapter wrapping generators as Futures for the executor
/// - `for_each_yield`: zero-buffered streaming pipeline combinator
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::generator::*;
/// use gpu_runtime::warp_future::WarpContext;
///
/// let mut gen = CounterGenerator::new(10);
/// let mut wcx = unsafe { WarpContext::new() };
/// let sum = unsafe {
///     for_each_yield(&mut gen, |val, _wcx| {
///         // all 32 lanes see the same val
///     }, &mut wcx)
/// };
/// // sum == 0+1+2+...+9 == 45
/// ```
pub mod generator;

/// Prelude — import everything you need for a basic GPU kernel.
///
/// The prelude exports high-level APIs for common tasks. For low-level
/// access (atomics, protocol constants, packet layout), use the module
/// paths directly: `gpu_runtime::hostcall::*`, `gpu_atomics::*`,
/// `gpu_protocol::*`.
///
/// ```rust,ignore
/// use gpu_runtime::prelude::*;
/// ```
pub mod prelude;
