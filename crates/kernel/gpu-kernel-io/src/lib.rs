// GPU hostcall, I/O, and pipeline kernels.
//
// This crate contains kernel entry points for hostcall protocol testing,
// file I/O pipelines (WarpFuture-based), and hybrid executor kernels.
// Extracted from the former gpu-kernel-std (now gpu-kernel-test) as part of the kernel-split epic.

#![no_main]
#![feature(restricted_std)]
#![feature(abi_gpu_kernel)]
#![feature(stdarch_nvptx)]
// NOTE: Under restricted_std, the standard library provides #[panic_handler].

// Re-export gpu-kernel-core so its kernel symbols are linked into this crate's cdylib.
extern crate gpu_kernel_core;

mod composed_priority;
mod hostcall_kernels;
mod hybrid;
mod obstacle_stress;
mod pipeline;
mod priority_safety_stress;

// Force-link stdio symbols from gpu-runtime. These are called by the patched
// std PAL via `extern "C"` blocks, so LTO would strip them without this anchor.
#[used]
static _KEEP_STDOUT: unsafe fn(*const u8, usize) -> usize = gpu_runtime::stdio::gpu_stdout_write;
#[used]
static _KEEP_STDIN: unsafe fn(*mut u8, usize) -> usize = gpu_runtime::stdio::gpu_stdin_read;
