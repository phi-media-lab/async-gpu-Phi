//! Canonical PTX routes used by the host test harness.
//!
//! A kernel name alone is not enough: the same workspace builds four PTX
//! modules, and loading a symbol from the wrong one used to fail only after a
//! long CUDA JIT. Keep the module choice next to every routed symbol.

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use cudarc::nvrtc::Ptx;
use gpu_host::error::{GpuHostError, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum KernelModule {
    Core,
    Compute,
    Io,
    Test,
}

impl KernelModule {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Core => "core",
            Self::Compute => "compute",
            Self::Io => "io",
            Self::Test => "test",
        }
    }

    pub(crate) const fn ptx(self) -> &'static str {
        match self {
            Self::Core => gpu_host::ptx::KERNEL_CORE,
            Self::Compute => gpu_host::ptx::KERNEL_COMPUTE,
            Self::Io => gpu_host::ptx::KERNEL_IO,
            Self::Test => gpu_host::ptx::KERNEL_TEST,
        }
    }

    fn declares(self, symbol: &str) -> bool {
        let marker = format!(".entry {symbol}(");
        self.ptx().lines().any(|line| line.contains(&marker))
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct KernelRoute {
    pub(crate) selectors: &'static [&'static str],
    pub(crate) module: KernelModule,
    pub(crate) symbols: &'static [&'static str],
}

#[derive(Clone, Copy, Debug)]
struct FeatureGatedRoute {
    selectors: &'static [&'static str],
    module: KernelModule,
    feature: &'static str,
    required_symbols: &'static [&'static str],
}

impl FeatureGatedRoute {
    const fn new(
        selectors: &'static [&'static str],
        module: KernelModule,
        feature: &'static str,
        required_symbols: &'static [&'static str],
    ) -> Self {
        Self {
            selectors,
            module,
            feature,
            required_symbols,
        }
    }

    fn validate(self, selector: &str) -> Result<()> {
        let missing: Vec<_> = self
            .required_symbols
            .iter()
            .copied()
            .filter(|symbol| !self.module.declares(symbol))
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        Err(GpuHostError::Verification {
            test: "ptx_feature_gate",
            detail: format!(
                "ONLY_TEST={selector} requires kernel feature `{}`; {} PTX is missing entries {missing:?}",
                self.feature,
                self.module.name()
            ),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SelectorPlan {
    Module(KernelModule),
    Standalone(&'static str),
    RuntimeNvrtc,
    MixedComputeNvrtc,
    AutoDiscovery,
    HostOnly,
}

impl SelectorPlan {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Module(module) => module.name(),
            Self::Standalone(name) => name,
            Self::RuntimeNvrtc => "runtime-nvrtc",
            Self::MixedComputeNvrtc => "compute+runtime-nvrtc",
            Self::AutoDiscovery => "auto-discovery",
            Self::HostOnly => "host-only",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SelectorPlanEntry {
    selectors: &'static [&'static str],
    plan: SelectorPlan,
}

impl SelectorPlanEntry {
    const fn new(selectors: &'static [&'static str], plan: SelectorPlan) -> Self {
        Self { selectors, plan }
    }
}

/// Complete audit of every selector accepted by the `ONLY_TEST` dispatch in
/// `main.rs`. This table records top-level ownership even for selectors whose
/// implementation loads several symbols or delegates to an auto-discovery API.
const ONLY_TEST_PLANS: &[SelectorPlanEntry] = &[
    SelectorPlanEntry::new(
        &[
            "generation",
            "forward",
            "mma_diag",
            "splitk",
            "mma_fwd",
            "kv_cache",
            "kv_gen",
            "bf16",
            "bf16_fwd",
            "tf32",
            "compute",
            "compute_pipeline",
            "compute_bench",
            "gemm_bench",
            "elem_bench",
            "cnn",
            "yolo",
        ],
        SelectorPlan::Module(KernelModule::Compute),
    ),
    SelectorPlanEntry::new(
        &[
            "trace",
            "session",
            "cmd",
            "pipeline",
            "converge",
            "flight",
            "std_future",
            "throughput",
            "scalability",
            "file_io_bench",
            "executor",
            "obstacle",
            "obstacle_stress",
            "composed_priority",
            "priority_e2e",
            "priority_safety",
            "channel",
            "channel_oneshot",
            "channel_mpsc",
            "mpsc",
            "tokio_bridge",
            "tokio",
            "bench",
        ],
        SelectorPlan::Module(KernelModule::Io),
    ),
    SelectorPlanEntry::new(
        &[
            "std_fs",
            "std_pipeline",
            "std_stdin",
            "mt_std",
            "warp_try",
            "warp_await",
            "warp_e2e",
            "rustc_async",
            "thread_spawn",
            "std_thread_demo",
            "real_std_thread",
            "std_thread_minimal",
            "par_iter",
            "par_iter_fusion",
            "par_iter_1m",
            "par_iter_bench",
            "par_iter_rayon",
            "par_iter_multiblock",
            "par_iter_mb",
            "kernel_std_smoke",
            "matmul_io",
            "zero_param",
            "generator",
            "coroutine",
        ],
        SelectorPlan::Module(KernelModule::Test),
    ),
    SelectorPlanEntry::new(&["mt_malloc"], SelectorPlan::Module(KernelModule::Core)),
    SelectorPlanEntry::new(
        &["std_sysroot_file"],
        SelectorPlan::Standalone("std-build-test"),
    ),
    SelectorPlanEntry::new(
        &["attn_bench", "sgemm_v4_bench"],
        SelectorPlan::RuntimeNvrtc,
    ),
    SelectorPlanEntry::new(&["fusion_bench"], SelectorPlan::MixedComputeNvrtc),
    SelectorPlanEntry::new(&["cpu_ref"], SelectorPlan::HostOnly),
    SelectorPlanEntry::new(
        &["gpu_run", "cooperative", "matmul"],
        SelectorPlan::AutoDiscovery,
    ),
];

const FEATURE_GATED_ROUTES: &[FeatureGatedRoute] = &[
    FeatureGatedRoute::new(
        &["mma_diag", "mma_fwd", "gemm_bench"],
        KernelModule::Compute,
        "sm_80",
        &["full_gemm_f32in"],
    ),
    FeatureGatedRoute::new(
        &["splitk"],
        KernelModule::Compute,
        "sm_80",
        &[
            "full_gemm_splitk",
            "full_gemm_f32in",
            "multi_block_gemm",
            "mma_diag",
        ],
    ),
    FeatureGatedRoute::new(
        &["bf16"],
        KernelModule::Compute,
        "sm_80",
        &["full_gemm_bf16", "full_gemm_f32in"],
    ),
    FeatureGatedRoute::new(
        &["bf16_fwd", "tf32"],
        KernelModule::Compute,
        "sm_80",
        &["full_gemm_bf16", "full_gemm_tf32"],
    ),
];

impl KernelRoute {
    const fn new(
        selectors: &'static [&'static str],
        module: KernelModule,
        symbols: &'static [&'static str],
    ) -> Self {
        Self {
            selectors,
            module,
            symbols,
        }
    }
}

/// Routes whose selector is especially sensitive to module split drift.
/// Standalone PTX selectors keep using their named PTX constants at call sites.
pub(crate) const CANONICAL_ROUTES: &[KernelRoute] = &[
    KernelRoute::new(
        &["trace"],
        KernelModule::Io,
        &["trace_multithread_test", "trace_assert_test"],
    ),
    KernelRoute::new(
        &["std_future"],
        KernelModule::Io,
        &[
            "std_future_print_kernel",
            "std_future_two_prints_kernel",
            "warp_cooperative_future_kernel",
            "warp_cooperative_two_futures_kernel",
            "warp_result_future_kernel",
        ],
    ),
    KernelRoute::new(&["warp_e2e"], KernelModule::Test, &["warp_e2e_test"]),
    KernelRoute::new(
        &["rustc_async"],
        KernelModule::Test,
        &["rustc_async_baseline_test"],
    ),
    KernelRoute::new(&["executor"], KernelModule::Io, &["executor_demo"]),
    KernelRoute::new(
        &["channel", "channel_oneshot"],
        KernelModule::Io,
        &["channel_oneshot_demo"],
    ),
    KernelRoute::new(
        &["channel_mpsc", "mpsc"],
        KernelModule::Io,
        &["channel_mpsc_demo"],
    ),
    KernelRoute::new(
        &["obstacle", "obstacle_stress"],
        KernelModule::Io,
        &["obstacle_event_priority_stress"],
    ),
    KernelRoute::new(
        &["composed_priority", "priority_e2e"],
        KernelModule::Io,
        &[
            "composed_priority_hostcall_e2e",
            "priority_echo_timeout_reuse",
        ],
    ),
    KernelRoute::new(
        &["priority_safety"],
        KernelModule::Io,
        &["priority_handle_safety_stress"],
    ),
    KernelRoute::new(
        &["compute", "compute_pipeline", "compute_bench"],
        KernelModule::Compute,
        &[
            "compute_pipeline_demo",
            "bench_stage_softmax",
            "bench_stage_gelu",
            "bench_stage_reduce",
        ],
    ),
    KernelRoute::new(
        &["mt_malloc"],
        KernelModule::Core,
        &["test_multithread_malloc"],
    ),
    KernelRoute::new(
        &["thread_spawn"],
        KernelModule::Test,
        &["thread_spawn_test", "thread_reuse_test"],
    ),
    KernelRoute::new(
        &["session"],
        KernelModule::Io,
        &["session_kernel_a", "session_kernel_b"],
    ),
    KernelRoute::new(&["cmd"], KernelModule::Io, &["multi_cmd_kernel"]),
    KernelRoute::new(
        &["pipeline"],
        KernelModule::Io,
        &["pipeline_writer_kernel", "pipeline_reader_kernel"],
    ),
    KernelRoute::new(
        &["converge"],
        KernelModule::Io,
        &["convergence_kernel", "autonomous_pipeline_kernel"],
    ),
    KernelRoute::new(&["flight"], KernelModule::Io, &["flight_recorder_test"]),
    KernelRoute::new(&["warp_try"], KernelModule::Test, &["warp_try_open_test"]),
    KernelRoute::new(&["warp_await"], KernelModule::Test, &["warp_await_test"]),
    KernelRoute::new(
        &["throughput", "scalability"],
        KernelModule::Io,
        &["hostcall_latency_bench_v3"],
    ),
    KernelRoute::new(&["file_io_bench"], KernelModule::Io, &["file_io_bench"]),
    KernelRoute::new(
        &["bench"],
        KernelModule::Io,
        &["hostcall_latency_bench_v3", "file_io_bench"],
    ),
];

pub(crate) fn route_for(selector: &str) -> Option<&'static KernelRoute> {
    CANONICAL_ROUTES
        .iter()
        .find(|route| route.selectors.contains(&selector))
}

pub(crate) fn selector_plan_for(selector: &str) -> Option<SelectorPlan> {
    ONLY_TEST_PLANS
        .iter()
        .find(|entry| entry.selectors.contains(&selector))
        .map(|entry| entry.plan)
}

pub(crate) fn validate_feature_gate(selector: &str) -> Result<()> {
    match FEATURE_GATED_ROUTES
        .iter()
        .find(|route| route.selectors.contains(&selector))
    {
        Some(route) => route.validate(selector),
        None => Ok(()),
    }
}

pub(crate) fn validate_build_feature(selector: &str) -> Result<()> {
    #[cfg(not(feature = "async"))]
    if matches!(selector, "tokio_bridge" | "tokio") {
        return unavailable_build_feature(selector, "async");
    }
    #[cfg(not(feature = "cublas"))]
    if matches!(selector, "attn_bench" | "fusion_bench" | "sgemm_v4_bench") {
        return unavailable_build_feature(selector, "cublas");
    }
    #[cfg(not(feature = "nn"))]
    if selector == "elem_bench" {
        return unavailable_build_feature(selector, "nn");
    }
    let _ = selector;
    Ok(())
}

#[cfg(any(not(feature = "async"), not(feature = "cublas"), not(feature = "nn")))]
fn unavailable_build_feature(selector: &str, feature: &str) -> Result<()> {
    Err(GpuHostError::Verification {
        test: "only_test_feature",
        detail: format!("ONLY_TEST={selector} requires harness feature `{feature}`"),
    })
}

/// Validate the textual route, load PTX, and resolve every symbol before the
/// caller allocates listener-owned resources or starts a background thread.
pub(crate) fn load_kernel(
    dev: &Arc<CudaDevice>,
    module: KernelModule,
    module_name: &'static str,
    symbols: &'static [&'static str],
) -> Result<()> {
    for &symbol in symbols {
        if !module.declares(symbol) {
            return Err(GpuHostError::Verification {
                test: "ptx_route",
                detail: format!(
                    "kernel `{symbol}` is not declared by the explicit `{}` PTX module",
                    module.name()
                ),
            });
        }
    }

    dev.load_ptx(Ptx::from_src(module.ptx()), module_name, symbols)?;
    for &symbol in symbols {
        if dev.get_func(module_name, symbol).is_none() {
            return Err(GpuHostError::KernelNotFound(symbol));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn canonical_routes_are_unique_and_symbols_exist() {
        let mut selectors = HashSet::new();
        for route in CANONICAL_ROUTES {
            for selector in route.selectors {
                assert!(selectors.insert(*selector), "duplicate selector {selector}");
            }
            for symbol in route.symbols {
                assert!(
                    route.module.declares(symbol),
                    "{} must declare {symbol}",
                    route.module.name()
                );
            }
        }
    }

    #[test]
    fn split_sensitive_only_test_routes_are_explicit() {
        for (selector, expected) in [
            ("trace", KernelModule::Io),
            ("std_future", KernelModule::Io),
            ("warp_e2e", KernelModule::Test),
            ("rustc_async", KernelModule::Test),
            ("executor", KernelModule::Io),
            ("channel", KernelModule::Io),
            ("channel_mpsc", KernelModule::Io),
            ("obstacle", KernelModule::Io),
            ("priority_safety", KernelModule::Io),
            ("compute", KernelModule::Compute),
            ("compute_bench", KernelModule::Compute),
            ("mt_malloc", KernelModule::Core),
            ("thread_spawn", KernelModule::Test),
            ("session", KernelModule::Io),
            ("cmd", KernelModule::Io),
            ("pipeline", KernelModule::Io),
            ("converge", KernelModule::Io),
            ("flight", KernelModule::Io),
            ("warp_try", KernelModule::Test),
            ("warp_await", KernelModule::Test),
            ("throughput", KernelModule::Io),
            ("file_io_bench", KernelModule::Io),
            ("bench", KernelModule::Io),
        ] {
            assert_eq!(
                route_for(selector).map(|route| route.module),
                Some(expected)
            );
        }
    }

    #[test]
    fn missing_sm80_symbols_fail_as_a_feature_gate() {
        let result = validate_feature_gate("mma_diag");
        if KernelModule::Compute.declares("full_gemm_f32in") {
            assert!(result.is_ok());
        } else {
            let error = result.expect_err("missing sm_80 entry must be explicit");
            assert!(error.to_string().contains("sm_80"));
            assert!(error.to_string().contains("full_gemm_f32in"));
        }
    }

    #[cfg(not(feature = "cublas"))]
    #[test]
    fn disabled_selector_fails_instead_of_running_the_full_suite() {
        let error = validate_build_feature("fusion_bench")
            .expect_err("disabled feature must reject its selector");
        assert!(error.to_string().contains("cublas"));
    }

    #[test]
    fn every_only_test_selector_has_one_plan() {
        let mut selectors = HashSet::new();
        for entry in ONLY_TEST_PLANS {
            for selector in entry.selectors {
                assert!(selectors.insert(*selector), "duplicate selector {selector}");
            }
        }
        assert_eq!(selectors.len(), 73, "update the route audit with main.rs");
    }

    #[test]
    fn trace_cannot_silently_fall_back_to_compute() {
        assert!(KernelModule::Io.declares("trace_multithread_test"));
        assert!(!KernelModule::Compute.declares("trace_multithread_test"));
    }
}
