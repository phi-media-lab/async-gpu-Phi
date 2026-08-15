//! Host-side hostcall listener and buffer management.
//!
//! This module implements the host half of the GPU-host hostcall protocol.
//! It allocates a pinned, device-mapped packet buffer and runs a listener
//! thread that polls for GPU requests and dispatches them to service handlers.
//!
//! # Hostcall protocol overview
//!
//! The GPU and host communicate through a shared memory region containing a
//! pool of fixed-size packet slots. Each packet has a control word, a service
//! ID, and a 56-byte payload. The protocol flow is:
//!
//! 1. **GPU** acquires a free packet slot from the lock-free stack
//! 2. **GPU** writes the service ID + payload, then sets the doorbell flag
//! 3. **Host listener** polls doorbell flags in a tight loop, detects the request
//! 4. **Host** dispatches to the appropriate service handler (print, file I/O,
//!    TCP networking, stdin, etc.)
//! 5. **Host** writes the response payload and clears the doorbell (ACK)
//! 6. **GPU** reads the response and releases the packet back to the free stack
//!
//! For bulk data exceeding 56 bytes, a separate sideband buffer provides a
//! bump-allocated scratch region shared between GPU and host.
//!
//! # Sharding
//!
//! The buffer can be sharded across CUDA blocks (`new_sharded`) so that each
//! block uses `blockIdx.x % num_shards`, reducing contention on the free stack.
//!
//! # FdResource model
//!
//! File descriptors returned to the GPU live in a unified fd table that holds
//! three resource types: [`std::fs::File`], [`std::net::TcpStream`], and
//! [`std::net::TcpListener`]. The GPU uses the same fd namespace for all I/O
//! operations (read, write, close) regardless of the underlying resource type.
//! File handles persist across kernel launches within the same [`HostcallSession`].
//!
//! # Key types
//!
//! - [`HostcallBuffer`] — Pinned shared-memory packet pool with host + device pointers
//! - [`HostcallSession`] — Persistent listener that survives across kernel launches
//! - [`Pipeline`] — Multi-stage kernel pipeline with automatic packet reinitialization
//! - [`CommandBuffer`] — Host-to-GPU command ring buffer
//! - [`FlightRecorder`] — Mapped-memory ring buffer for post-mortem GPU tracing
//! - [`HostcallError`] — Error type for buffer allocation failures

use cudarc::driver::sys::{self, lib as cuda_lib};
pub use gpu_protocol::composed_priority_schema as composed_schema;
pub use gpu_protocol::priority_echo_reuse_schema as reuse_schema;
pub use gpu_protocol::Priority as EchoPriority;
use gpu_protocol::*;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

// ================================================================
// Unified fd resource table (files + TCP sockets)
// ================================================================

/// A resource held in the fd table — either a file, TCP stream, or TCP listener.
enum FdResource {
    /// An open file handle.
    File(File),
    /// A connected TCP stream.
    TcpStream(TcpStream),
    /// A bound TCP listener.
    TcpListener(TcpListener),
}

// ================================================================
// Stdin abstraction (host-scaling.2 Phase A)
// ================================================================

/// Trait for providing stdin data to the listener.
/// Implementations must be `Send` to allow I/O thread offloading.
pub trait StdinSource: Send {
    /// Read up to `buf.len()` bytes of stdin input. Returns bytes written.
    fn read_line_bytes(&mut self, buf: &mut [u8]) -> usize;
}

/// Real stdin — reads from `std::io::stdin()`. Blocks until input available.
pub struct RealStdin;

impl StdinSource for RealStdin {
    fn read_line_bytes(&mut self, buf: &mut [u8]) -> usize {
        let mut line = String::new();
        match std::io::stdin().read_line(&mut line) {
            Ok(0) | Err(_) => 0,
            Ok(n) => {
                let bytes = line.as_bytes();
                let copy = n.min(buf.len());
                buf[..copy].copy_from_slice(&bytes[..copy]);
                copy
            }
        }
    }
}

/// Canned stdin — returns pre-loaded data once, then EOF on subsequent reads.
pub struct CannedStdin {
    data: Vec<u8>,
    consumed: bool,
}

impl CannedStdin {
    /// Create a `CannedStdin` that returns `data` on first read, then EOF.
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            consumed: false,
        }
    }
}

impl StdinSource for CannedStdin {
    fn read_line_bytes(&mut self, buf: &mut [u8]) -> usize {
        if self.consumed {
            return 0;
        }
        self.consumed = true;
        let copy = self.data.len().min(buf.len());
        buf[..copy].copy_from_slice(&self.data[..copy]);
        copy
    }
}

/// Request sent from the listener thread to the blocking I/O service thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IoRequest {
    pkt_idx: u16,
    service: u32,
    metadata: HostcallMetadata,
    generation: Option<u64>,
    pool: PacketPool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PacketPool {
    HighShared,
    General { shard: u32 },
}

enum PacketClaim {
    Claimed(IoRequest),
    Cancelled(IoRequest),
    Ignore,
}

#[derive(Default)]
struct HostcallLifecycleState {
    listener_active: bool,
    freeze_requested: bool,
    freeze_epoch: u64,
    acknowledged_epoch: u64,
    dispatching: usize,
    queued: usize,
    inflight: usize,
}

#[derive(Default)]
struct HostcallLifecycle {
    state: Mutex<HostcallLifecycleState>,
    changed: Condvar,
}

impl HostcallLifecycle {
    fn lock(&self) -> std::sync::MutexGuard<'_, HostcallLifecycleState> {
        self.state.lock().unwrap_or_else(|error| error.into_inner())
    }

    fn listener_started(&self) {
        let mut state = self.lock();
        while state.freeze_requested {
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(|error| error.into_inner());
        }
        assert!(
            !state.listener_active,
            "only one hostcall listener is supported"
        );
        state.listener_active = true;
        self.changed.notify_all();
    }

    fn listener_stopped(&self) {
        let mut state = self.lock();
        state.listener_active = false;
        state.dispatching = 0;
        self.changed.notify_all();
    }

    fn dispatch_started(&self, count: usize) {
        let mut state = self.lock();
        state.dispatching += count;
        self.changed.notify_all();
    }

    fn dispatch_finished(&self, count: usize) {
        let mut state = self.lock();
        state.dispatching = state.dispatching.saturating_sub(count);
        self.changed.notify_all();
    }

    fn queued(&self) {
        let mut state = self.lock();
        state.queued += 1;
        self.changed.notify_all();
    }

    fn begin_inflight(&self) {
        let mut state = self.lock();
        state.queued = state.queued.saturating_sub(1);
        state.inflight += 1;
        self.changed.notify_all();
    }

    fn finish_inflight(&self) {
        let mut state = self.lock();
        state.inflight = state.inflight.saturating_sub(1);
        self.changed.notify_all();
    }

    fn request_freeze(&self) -> u64 {
        let mut state = self.lock();
        while state.freeze_requested {
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(|error| error.into_inner());
        }
        state.freeze_epoch = state.freeze_epoch.wrapping_add(1);
        state.freeze_requested = true;
        let epoch = state.freeze_epoch;
        self.changed.notify_all();
        epoch
    }

    fn try_request_freeze(&self) -> Result<u64, ReinitBusy> {
        let mut state = self.lock();
        if state.freeze_requested
            || state.listener_active
            || state.dispatching != 0
            || state.queued != 0
            || state.inflight != 0
        {
            return Err(ReinitBusy {
                listener_active: state.listener_active,
                dispatching: state.dispatching,
                queued: state.queued,
                inflight: state.inflight,
            });
        }
        state.freeze_epoch = state.freeze_epoch.wrapping_add(1);
        state.freeze_requested = true;
        let epoch = state.freeze_epoch;
        self.changed.notify_all();
        Ok(epoch)
    }

    fn freeze_requested(&self) -> bool {
        self.lock().freeze_requested
    }

    fn acknowledge_and_wait(&self, shutdown: &AtomicU32) {
        let mut state = self.lock();
        if !state.freeze_requested {
            return;
        }
        state.acknowledged_epoch = state.freeze_epoch;
        self.changed.notify_all();
        while state.freeze_requested && shutdown.load(Ordering::Acquire) == 0 {
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(|error| error.into_inner());
        }
    }

    fn wait_quiescent(&self, epoch: u64) {
        let mut state = self.lock();
        while (state.listener_active && state.acknowledged_epoch < epoch)
            || state.dispatching != 0
            || state.queued != 0
            || state.inflight != 0
        {
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(|error| error.into_inner());
        }
    }

    fn thaw(&self) {
        let mut state = self.lock();
        state.freeze_requested = false;
        self.changed.notify_all();
    }

    fn notify_shutdown(&self) {
        self.changed.notify_all();
    }
}

struct ListenerLifecycleGuard<'a> {
    lifecycle: &'a HostcallLifecycle,
}

impl Drop for ListenerLifecycleGuard<'_> {
    fn drop(&mut self) {
        self.lifecycle.listener_stopped();
    }
}

struct InflightLifecycleGuard<'a> {
    lifecycle: &'a HostcallLifecycle,
}

struct DispatchBatchGuard<'a> {
    lifecycle: &'a HostcallLifecycle,
    remaining: usize,
}

impl DispatchBatchGuard<'_> {
    fn finished_one(&mut self) {
        self.lifecycle.dispatch_finished(1);
        self.remaining = self.remaining.saturating_sub(1);
    }
}

impl Drop for DispatchBatchGuard<'_> {
    fn drop(&mut self) {
        self.lifecycle.dispatch_finished(self.remaining);
    }
}

impl Drop for InflightLifecycleGuard<'_> {
    fn drop(&mut self) {
        self.lifecycle.finish_inflight();
    }
}

/// Why a non-blocking packet-pool reinitialization could not prove quiescence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReinitBusy {
    /// Whether the persistent listener is still active.
    pub listener_active: bool,
    /// Number of requests held by a listener-local drained chain.
    pub dispatching: usize,
    /// Number of slow requests waiting in the priority queue.
    pub queued: usize,
    /// Number of slow requests currently executing a host handler.
    pub inflight: usize,
}

impl fmt::Display for ReinitBusy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "hostcall not quiescent: listener={}, dispatching={}, queued={}, inflight={}",
            self.listener_active, self.dispatching, self.queued, self.inflight
        )
    }
}

impl std::error::Error for ReinitBusy {}

/// Error-visible counters from successfully claimed hostcall requests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostcallProcessingMetrics {
    /// Requests whose completion was successfully published to the device.
    pub completed: u64,
    /// Published completions carrying `CONTROL_ERROR`.
    pub errors: u64,
    /// Timed-out requests reclaimed by the host.
    pub cancelled: u64,
    /// Queue entries rejected because their generation or metadata was stale.
    pub stale_rejected: u64,
}

/// One independently observed `PRIORITY_ECHO` request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PriorityEchoEvent {
    /// Nonce read before the host mutates the payload.
    pub request_nonce: u64,
    /// Nonce returned to the device (may differ only under a test hook).
    pub echo_nonce: u64,
    /// Complete task id read from the claimed packet header.
    pub task_id: u64,
    /// Upper 32 bits of `task_id`.
    pub namespace: u32,
    /// Lower 32 bits of `task_id`.
    pub local_id: u32,
    /// Effective priority read from the claimed packet header.
    pub priority: Priority,
    /// Physical packet index claimed by the listener.
    pub packet_index: u16,
    /// Complete v3 request generation claimed by the listener.
    pub generation: u64,
    /// Whether the index belongs to the shared High-only pool.
    pub shared_high_reserved: bool,
    /// Monotonic sequence assigned by the host listener.
    pub process_sequence: u64,
    /// Number of events seen for this nonce/task identity.
    pub process_count: u64,
    /// Zero on success, otherwise the emitted `ERR_*` category.
    pub error_category: u16,
}

/// Explicit, default-off fault hooks for the composed GPU conformance test.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PriorityEchoTestHook {
    #[default]
    /// Production/default behavior.
    Disabled = 0,
    /// Publish an explicit host error.
    ForceError = 1,
    /// Return a different nonce without changing packet identity.
    CorruptEchoNonce = 2,
    /// Delay exactly the next echo; the handler atomically disables this hook.
    DelayNext = 3,
}

impl PriorityEchoTestHook {
    fn from_raw(raw: u32) -> Self {
        match raw {
            1 => Self::ForceError,
            2 => Self::CorruptEchoNonce,
            3 => Self::DelayNext,
            _ => Self::Disabled,
        }
    }
}

/// Quiescent packet-pool snapshot used by the independent host oracle.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HostcallPoolAudit {
    /// Every ready stack was empty at the snapshot.
    pub ready_empty: bool,
    /// Number of packet controls with no active state flags.
    pub idle_packets: u16,
    /// Free general-pool indices (limited to the first 64 packets).
    pub general_mask: u64,
    /// Free shared-High indices (limited to the first 64 packets).
    pub shared_high_mask: u64,
    /// Duplicate, cyclic, or invalid free-stack entries.
    pub duplicate_entries: u16,
    /// Allocated packets absent from every free stack.
    pub missing_packets: u16,
    /// Packet controls that still carry a non-idle state.
    pub non_idle_controls: u16,
}

/// Weighted service cycle: latency-sensitive work gets most service slots,
/// while Normal and Low retain bounded opportunities under sustained High load.
const IO_SERVICE_CYCLE: [Priority; 13] = [
    Priority::High,
    Priority::High,
    Priority::High,
    Priority::High,
    Priority::High,
    Priority::High,
    Priority::High,
    Priority::High,
    Priority::Normal,
    Priority::Normal,
    Priority::Normal,
    Priority::Normal,
    Priority::Low,
];

#[derive(Default)]
struct PriorityIoState {
    high: VecDeque<IoRequest>,
    normal: VecDeque<IoRequest>,
    low: VecDeque<IoRequest>,
    cycle_cursor: usize,
    closed: bool,
}

impl PriorityIoState {
    fn queue_mut(&mut self, priority: Priority) -> &mut VecDeque<IoRequest> {
        match priority {
            Priority::High => &mut self.high,
            Priority::Normal => &mut self.normal,
            Priority::Low => &mut self.low,
        }
    }

    fn push(&mut self, request: IoRequest) {
        self.queue_mut(request.metadata.effective_priority)
            .push_back(request);
    }

    fn is_empty(&self) -> bool {
        self.high.is_empty() && self.normal.is_empty() && self.low.is_empty()
    }

    fn pop_weighted(&mut self) -> Option<IoRequest> {
        if self.is_empty() {
            return None;
        }

        for _ in 0..IO_SERVICE_CYCLE.len() {
            let priority = IO_SERVICE_CYCLE[self.cycle_cursor];
            self.cycle_cursor = (self.cycle_cursor + 1) % IO_SERVICE_CYCLE.len();
            if let Some(request) = self.queue_mut(priority).pop_front() {
                return Some(request);
            }
        }
        None
    }
}

/// Blocking, stable, priority-aware queue shared by the listener and service
/// threads. Poisoned locks are recovered because dropping all queued GPU
/// requests would otherwise leave device callers spinning forever.
struct PriorityIoQueue {
    state: Mutex<PriorityIoState>,
    available: Condvar,
    lifecycle: Arc<HostcallLifecycle>,
}

impl PriorityIoQueue {
    fn new(lifecycle: Arc<HostcallLifecycle>) -> Self {
        Self {
            state: Mutex::new(PriorityIoState::default()),
            available: Condvar::new(),
            lifecycle,
        }
    }

    fn push(&self, request: IoRequest) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.closed {
            return false;
        }
        state.push(request);
        self.lifecycle.queued();
        self.available.notify_one();
        true
    }

    fn pop(&self) -> Option<IoRequest> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(request) = state.pop_weighted() {
                self.lifecycle.begin_inflight();
                return Some(request);
            }
            if state.closed {
                return None;
            }
            state = self
                .available
                .wait(state)
                .unwrap_or_else(|e| e.into_inner());
        }
    }

    fn close(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.closed = true;
        self.available.notify_all();
    }
}

struct QueueCloseGuard<'a>(&'a PriorityIoQueue);

impl Drop for QueueCloseGuard<'_> {
    fn drop(&mut self) {
        self.0.close();
    }
}

/// Decode versioned task metadata from a packet header.
///
/// Old device code leaves the reserved header bytes at zero, which maps to the
/// compatibility default rather than to Low priority.
unsafe fn read_packet_metadata(pkt: *const u8) -> HostcallMetadata {
    let version = std::ptr::read_volatile(pkt.add(PKT_OFF_METADATA_VERSION));
    if version != PACKET_METADATA_PRIORITY_VERSION && version != PACKET_METADATA_VERSION {
        return HostcallMetadata::default();
    }
    let priority = Priority::from_raw(std::ptr::read_volatile(pkt.add(PKT_OFF_PRIORITY)));
    let task_id = std::ptr::read_volatile(pkt.add(PKT_OFF_TASK_ID) as *const u64);
    HostcallMetadata::new(task_id, priority)
}

/// Map a std::io::Error to an error category code for hostcall error propagation.
fn io_error_to_category(e: &std::io::Error) -> u16 {
    use std::io::ErrorKind;
    match e.kind() {
        ErrorKind::NotFound => ERR_NOT_FOUND,
        ErrorKind::PermissionDenied => ERR_PERMISSION_DENIED,
        ErrorKind::AlreadyExists => ERR_ALREADY_EXISTS,
        ErrorKind::InvalidInput => ERR_INVALID_INPUT,
        ErrorKind::TimedOut => ERR_TIMED_OUT,
        ErrorKind::WouldBlock => ERR_WOULD_BLOCK,
        ErrorKind::BrokenPipe => ERR_BROKEN_PIPE,
        ErrorKind::OutOfMemory => ERR_OUT_OF_MEMORY,
        ErrorKind::Unsupported => ERR_UNSUPPORTED,
        ErrorKind::ConnectionRefused => ERR_CONNECTION_REFUSED,
        ErrorKind::ConnectionReset => ERR_CONNECTION_RESET,
        ErrorKind::AddrInUse => ERR_ADDR_IN_USE,
        ErrorKind::AddrNotAvailable => ERR_ADDR_NOT_AVAILABLE,
        ErrorKind::NotConnected => ERR_NOT_CONNECTED,
        _ => ERR_OTHER,
    }
}

/// Encode an io::Error into the hostcall error format and write it to payload slot 0.
/// Returns `true` to signal CONTROL_ERROR should be set.
unsafe fn write_error_response(payload: *mut u8, e: &std::io::Error) -> bool {
    let category = io_error_to_category(e);
    let raw_errno = e.raw_os_error().unwrap_or(0) as u16;
    // SAFETY: Caller guarantees payload points to slot 0 within a valid packet's
    // payload region (PKT_OFF_PAYLOAD offset from a valid packet pointer).
    // Volatile write ensures the GPU observes the error value.
    std::ptr::write_volatile(payload as *mut u64, encode_error(category, raw_errno));
    true
}

/// Errors that can occur during hostcall buffer allocation.
#[derive(Debug)]
pub enum HostcallError {
    /// `cuMemHostAlloc` failed — could not allocate pinned GPU-visible memory.
    CudaAlloc(sys::CUresult),
    /// `cuMemHostGetDevicePointer_v2` failed — could not obtain device-side pointer.
    CudaGetDevPtr(sys::CUresult),
}

impl fmt::Display for HostcallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CudaAlloc(r) => write!(f, "cuMemHostAlloc failed: {r:?}"),
            Self::CudaGetDevPtr(r) => {
                write!(f, "cuMemHostGetDevicePointer_v2 failed: {r:?}")
            }
        }
    }
}

impl std::error::Error for HostcallError {}

/// Hostcall buffer handle with both host and device pointers.
pub struct HostcallBuffer {
    /// Host-side pointer to the pinned hostcall buffer.
    pub(crate) host_ptr: *mut u8,
    /// Device-side pointer to the hostcall buffer (for kernel launch args).
    pub(crate) dev_ptr: sys::CUdeviceptr,
    /// Total size of the hostcall buffer in bytes.
    pub(crate) size: usize,
    /// Number of packet slots in the buffer.
    pub(crate) num_packets: u16,
    /// Number of shards (0 = legacy unsharded mode).
    pub(crate) num_shards: u32,
    /// Packets assigned to each shard (only meaningful when num_shards > 0).
    pub(crate) pkts_per_shard: u32,
    /// Number contributed by each shard to one shared global High-only pool.
    pub(crate) high_reserved_per_shard: u32,
    /// Host-side pointer to the sideband buffer for bulk data transfer (>56 bytes).
    pub(crate) sideband_host_ptr: *mut u8,
    /// Device-side pointer to the sideband buffer (for kernel launch args).
    pub(crate) sideband_dev_ptr: sys::CUdeviceptr,
    /// Total size of the sideband buffer in bytes.
    pub(crate) sideband_size: usize,
    /// Host-only listener/worker quiescence coordination.
    lifecycle: Arc<HostcallLifecycle>,
    processed_completed: AtomicU64,
    processed_errors: AtomicU64,
    processed_cancelled: AtomicU64,
    stale_rejected: AtomicU64,
    priority_echo_events: Mutex<Vec<PriorityEchoEvent>>,
    priority_echo_sequence: AtomicU64,
    priority_echo_hook: AtomicU32,
    priority_echo_delay_micros: AtomicU64,
}

// SAFETY: The buffer is pinned memory shared between host and GPU.
// We ensure single-writer access via the protocol (GPU writes packet,
// host reads; host writes control, GPU reads).
unsafe impl Send for HostcallBuffer {}
unsafe impl Sync for HostcallBuffer {}

/// Existing constructors preserve every packet as a general credit. Priority
/// reservation is opt-in through the `_with_priority_reserve` constructors.
const COMPAT_DEFAULT_HIGH_RESERVED_PER_SHARD: u32 = 0;

impl HostcallBuffer {
    /// Allocate and initialize a hostcall buffer with `num_packets` packet slots
    /// and a default-sized sideband buffer (1MB) for bulk data transfer.
    /// Legacy (unsharded) mode. All packets remain general-purpose for backward
    /// compatibility; use [`Self::new_with_priority_reserve`] to opt in.
    ///
    /// Uses cuMemHostAlloc with DEVICEMAP|PORTABLE flags for GPU-CPU shared access.
    pub fn new(num_packets: u16) -> Result<Self, HostcallError> {
        Self::new_with_sideband(num_packets, DEFAULT_SIDEBAND_SIZE)
    }

    /// Allocate an unsharded buffer with an explicit High-only packet reserve.
    pub fn new_with_priority_reserve(
        num_packets: u16,
        high_reserved_packets: u32,
    ) -> Result<Self, HostcallError> {
        Self::new_with_sideband_and_priority_reserve(
            num_packets,
            DEFAULT_SIDEBAND_SIZE,
            high_reserved_packets,
        )
    }

    /// Allocate a legacy (unsharded) hostcall buffer with custom sideband size.
    pub fn new_with_sideband(
        num_packets: u16,
        sideband_data_size: usize,
    ) -> Result<Self, HostcallError> {
        Self::new_with_sideband_and_priority_reserve(
            num_packets,
            sideband_data_size,
            COMPAT_DEFAULT_HIGH_RESERVED_PER_SHARD,
        )
    }

    /// Allocate an unsharded buffer with custom sideband and High reserve sizes.
    pub fn new_with_sideband_and_priority_reserve(
        num_packets: u16,
        sideband_data_size: usize,
        high_reserved_packets: u32,
    ) -> Result<Self, HostcallError> {
        Self::alloc_internal(num_packets, 0, 0, high_reserved_packets, sideband_data_size)
    }

    /// Allocate a sharded hostcall buffer with `num_shards` shards.
    ///
    /// Each shard gets `pkts_per_shard` general-purpose packets. Total packets =
    /// num_shards * pkts_per_shard. Each CUDA block uses shard
    /// `blockIdx.x % num_shards`. Use
    /// [`Self::new_sharded_with_priority_reserve`] to opt in to High reserve.
    pub fn new_sharded(num_shards: u32, pkts_per_shard: u32) -> Result<Self, HostcallError> {
        Self::new_sharded_with_sideband(num_shards, pkts_per_shard, DEFAULT_SIDEBAND_SIZE)
    }

    /// Allocate a sharded buffer whose shards each contribute packets to one
    /// shared global High-only reserve. This is not per-shard admission.
    pub fn new_sharded_with_priority_reserve(
        num_shards: u32,
        pkts_per_shard: u32,
        high_reserved_per_shard: u32,
    ) -> Result<Self, HostcallError> {
        Self::new_sharded_with_sideband_and_priority_reserve(
            num_shards,
            pkts_per_shard,
            DEFAULT_SIDEBAND_SIZE,
            high_reserved_per_shard,
        )
    }

    /// Allocate a sharded hostcall buffer with custom sideband size.
    pub fn new_sharded_with_sideband(
        num_shards: u32,
        pkts_per_shard: u32,
        sideband_data_size: usize,
    ) -> Result<Self, HostcallError> {
        Self::new_sharded_with_sideband_and_priority_reserve(
            num_shards,
            pkts_per_shard,
            sideband_data_size,
            COMPAT_DEFAULT_HIGH_RESERVED_PER_SHARD,
        )
    }

    /// Allocate a sharded buffer with custom sideband and High reserve sizes.
    pub fn new_sharded_with_sideband_and_priority_reserve(
        num_shards: u32,
        pkts_per_shard: u32,
        sideband_data_size: usize,
        high_reserved_per_shard: u32,
    ) -> Result<Self, HostcallError> {
        let total_packets = num_shards * pkts_per_shard;
        assert!(total_packets <= 0xFFFE, "too many packets (max 65534)");
        Self::alloc_internal(
            total_packets as u16,
            num_shards,
            pkts_per_shard,
            high_reserved_per_shard,
            sideband_data_size,
        )
    }

    /// Internal allocation — handles both legacy and sharded modes.
    fn alloc_internal(
        num_packets: u16,
        num_shards: u32,
        pkts_per_shard: u32,
        high_reserved_per_shard: u32,
        sideband_data_size: usize,
    ) -> Result<Self, HostcallError> {
        let packets_in_shard = if num_shards == 0 {
            num_packets as u32
        } else {
            pkts_per_shard
        };
        assert!(
            packets_in_shard > 0,
            "hostcall buffer needs at least one packet"
        );
        assert!(
            high_reserved_per_shard < packets_in_shard,
            "High reserve must leave at least one general packet per shard"
        );
        let size = if num_shards == 0 {
            buffer_size(num_packets)
        } else {
            buffer_size_sharded(num_packets, num_shards)
        };
        // SAFETY: cuda_lib() returns the lazily-loaded CUDA driver function table.
        let cu = unsafe { cuda_lib() };
        let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;

        // Allocate hostcall buffer (pinned, device-mapped)
        let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: cuMemHostAlloc writes a valid pointer to host_ptr on success.
        // The allocation is `size` bytes with DEVICEMAP|PORTABLE flags.
        let result = unsafe { cu.cuMemHostAlloc(&mut host_ptr, size, flags) };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(HostcallError::CudaAlloc(result));
        }

        let mut dev_ptr: sys::CUdeviceptr = 0;
        // SAFETY: host_ptr was allocated with DEVICEMAP flag, so the driver can
        // provide a GPU-visible address.
        let result = unsafe { cu.cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0) };
        if result != sys::CUresult::CUDA_SUCCESS {
            // SAFETY: host_ptr was allocated above; freeing on error path.
            unsafe { cu.cuMemFreeHost(host_ptr) };
            return Err(HostcallError::CudaGetDevPtr(result));
        }

        // SAFETY: host_ptr is valid for `size` bytes. No kernel is running yet.
        unsafe {
            std::ptr::write_bytes(host_ptr as *mut u8, 0, size);
        }

        // Allocate sideband buffer for bulk data transfer
        let sideband_total = SIDEBAND_HEADER_SIZE + sideband_data_size;
        let mut sb_host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: Same CUDA alloc pattern as the hostcall buffer above.
        let result = unsafe { cu.cuMemHostAlloc(&mut sb_host_ptr, sideband_total, flags) };
        if result != sys::CUresult::CUDA_SUCCESS {
            unsafe { cu.cuMemFreeHost(host_ptr) };
            return Err(HostcallError::CudaAlloc(result));
        }

        let mut sb_dev_ptr: sys::CUdeviceptr = 0;
        // SAFETY: sb_host_ptr was allocated with DEVICEMAP flag.
        let result = unsafe { cu.cuMemHostGetDevicePointer_v2(&mut sb_dev_ptr, sb_host_ptr, 0) };
        if result != sys::CUresult::CUDA_SUCCESS {
            // SAFETY: Both pointers were allocated above; freeing on error path.
            unsafe {
                cu.cuMemFreeHost(sb_host_ptr);
                cu.cuMemFreeHost(host_ptr);
            }
            return Err(HostcallError::CudaGetDevPtr(result));
        }

        // SAFETY: sb_host_ptr is valid for `sideband_total` bytes. No kernel running.
        // SIDEBAND_OFF_CAPACITY is within the sideband header region.
        unsafe {
            std::ptr::write_bytes(sb_host_ptr as *mut u8, 0, sideband_total);
            let cap_ptr = (sb_host_ptr as *mut u8).add(SIDEBAND_OFF_CAPACITY) as *mut u64;
            std::ptr::write_volatile(cap_ptr, sideband_data_size as u64);
        }

        let buf = Self {
            host_ptr: host_ptr as *mut u8,
            dev_ptr,
            size,
            num_packets,
            num_shards,
            pkts_per_shard,
            high_reserved_per_shard,
            sideband_host_ptr: sb_host_ptr as *mut u8,
            sideband_dev_ptr: sb_dev_ptr,
            sideband_size: sideband_total,
            lifecycle: Arc::new(HostcallLifecycle::default()),
            processed_completed: AtomicU64::new(0),
            processed_errors: AtomicU64::new(0),
            processed_cancelled: AtomicU64::new(0),
            stale_rejected: AtomicU64::new(0),
            priority_echo_events: Mutex::new(Vec::new()),
            priority_echo_sequence: AtomicU64::new(0),
            priority_echo_hook: AtomicU32::new(PriorityEchoTestHook::Disabled as u32),
            priority_echo_delay_micros: AtomicU64::new(0),
        };

        buf.init();

        Ok(buf)
    }

    /// Rebuild general, High-reserved, and ready stacks without changing the
    /// fixed buffer layout. The High reservation is global for acquisition but
    /// draws the tail packets evenly from every shard.
    unsafe fn rebuild_packet_stacks(&self) {
        let base = self.host_ptr;
        let mut high_head = null_tagged();

        std::ptr::write_volatile(base.add(BUF_OFF_READY_STACK) as *mut u64, null_tagged());

        if self.num_shards == 0 {
            let general_packets = self.num_packets as u32 - self.high_reserved_per_shard;
            for i in 0..self.num_packets as u32 {
                let pkt_idx = i as u16;
                let pkt = base.add(packet_offset(pkt_idx));
                let next = if i < general_packets {
                    if i + 1 < general_packets {
                        make_tagged(0, (i + 1) as u16)
                    } else {
                        null_tagged()
                    }
                } else {
                    let previous = high_head;
                    high_head = make_tagged(0, pkt_idx);
                    previous
                };
                std::ptr::write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, next);
                std::ptr::write_volatile(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);
                std::ptr::write_volatile(pkt.add(PKT_OFF_METADATA_VERSION), 0);
                std::ptr::write_volatile(pkt.add(PKT_OFF_PRIORITY), Priority::Normal.as_raw());
                std::ptr::write_volatile(pkt.add(PKT_OFF_METADATA_FLAGS) as *mut u16, 0);
                std::ptr::write_volatile(pkt.add(PKT_OFF_TASK_ID) as *mut u64, 0);
            }
            std::ptr::write_volatile(base.add(BUF_OFF_FREE_STACK) as *mut u64, make_tagged(0, 0));
        } else {
            let shard_array_off = BUFFER_HEADER_SIZE;
            let general_packets = self.pkts_per_shard - self.high_reserved_per_shard;

            std::ptr::write_volatile(base.add(BUF_OFF_FREE_STACK) as *mut u64, null_tagged());
            for shard in 0..self.num_shards {
                let base_pkt = shard * self.pkts_per_shard;
                let entry_off = shard_entry_offset(shard_array_off, shard);

                for local in 0..self.pkts_per_shard {
                    let pkt_idx = (base_pkt + local) as u16;
                    let pkt = base.add(packet_offset_sharded(
                        pkt_idx,
                        shard_array_off,
                        self.num_shards,
                    ));
                    let next = if local < general_packets {
                        if local + 1 < general_packets {
                            make_tagged(0, (base_pkt + local + 1) as u16)
                        } else {
                            null_tagged()
                        }
                    } else {
                        let previous = high_head;
                        high_head = make_tagged(0, pkt_idx);
                        previous
                    };
                    std::ptr::write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, next);
                    std::ptr::write_volatile(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);
                    std::ptr::write_volatile(pkt.add(PKT_OFF_METADATA_VERSION), 0);
                    std::ptr::write_volatile(pkt.add(PKT_OFF_PRIORITY), Priority::Normal.as_raw());
                    std::ptr::write_volatile(pkt.add(PKT_OFF_METADATA_FLAGS) as *mut u16, 0);
                    std::ptr::write_volatile(pkt.add(PKT_OFF_TASK_ID) as *mut u64, 0);
                }

                std::ptr::write_volatile(
                    base.add(entry_off + SHARD_OFF_FREE_STACK) as *mut u64,
                    make_tagged(0, base_pkt as u16),
                );
                std::ptr::write_volatile(
                    base.add(entry_off + SHARD_OFF_READY_STACK) as *mut u64,
                    null_tagged(),
                );
            }
        }

        std::ptr::write_volatile(base.add(BUF_OFF_HIGH_FREE_STACK) as *mut u64, high_head);
    }

    /// Initialize the hostcall buffer: set up free stack, ready stack, etc.
    /// Handles both legacy (num_shards == 0) and sharded modes.
    fn init(&self) {
        let base = self.host_ptr;
        // SAFETY: All pointer arithmetic below stays within the cuMemHostAlloc region
        // of `self.size` bytes. The buffer layout offsets (BUF_OFF_*, PKT_OFF_*) are
        // computed by gpu_protocol to fit within buffer_size(num_packets). write_volatile
        // is used because the GPU may read this memory once a kernel is launched.
        // This is called during construction, before any kernel launch.
        unsafe {
            // Common header fields
            let doorbell = base.add(BUF_OFF_DOORBELL) as *mut u64;
            let shutdown = base.add(BUF_OFF_SHUTDOWN) as *mut u32;
            let num_packets_field = base.add(BUF_OFF_NUM_PACKETS) as *mut u32;
            let warp_size_field = base.add(BUF_OFF_WARP_SIZE) as *mut u32;
            let num_shards_field = base.add(BUF_OFF_NUM_SHARDS) as *mut u32;
            let pkts_per_shard_field = base.add(BUF_OFF_PKTS_PER_SHARD) as *mut u32;
            let shard_array_off_field = base.add(BUF_OFF_SHARD_ARRAY_OFF) as *mut u32;
            let protocol_version_field = base.add(BUF_OFF_PROTOCOL_VERSION) as *mut u32;
            let high_reserved_field = base.add(BUF_OFF_HIGH_RESERVED_PER_SHARD) as *mut u32;

            std::ptr::write_volatile(doorbell, 0u64);
            std::ptr::write_volatile(shutdown, 0u32);
            std::ptr::write_volatile(num_packets_field, self.num_packets as u32);
            std::ptr::write_volatile(warp_size_field, WARP_SIZE);
            std::ptr::write_volatile(num_shards_field, self.num_shards);
            std::ptr::write_volatile(pkts_per_shard_field, self.pkts_per_shard);
            std::ptr::write_volatile(shard_array_off_field, BUFFER_HEADER_SIZE as u32);
            std::ptr::write_volatile(protocol_version_field, HOSTCALL_PROTOCOL_VERSION);
            std::ptr::write_volatile(high_reserved_field, self.high_reserved_per_shard);
            self.rebuild_packet_stacks();
        }
    }

    /// Reinitialize packet pool for reuse between kernel launches.
    ///
    /// Resets free stacks (all packets available), ready stacks (empty),
    /// and all packet control flags. Resets sideband bump allocator.
    ///
    /// The caller must first synchronize the GPU and prevent every producer
    /// from starting a new hostcall until this method returns. The host-only
    /// freeze handshake cannot stop an arbitrary persistent GPU kernel.
    ///
    /// A persistent listener is supported: it drains its local ready chain and
    /// freezes, then this method waits for queued and in-flight I/O to finish
    /// before rebuilding any packet or sideband state. Blocking stdin/accept
    /// can therefore make this compatibility method wait without a bound; use
    /// [`Self::try_reinit_packets`] when rejection is preferable.
    pub fn reinit_packets(&self) {
        let epoch = self.lifecycle.request_freeze();
        self.lifecycle.wait_quiescent(epoch);
        self.reset_quiescent_packets();
        self.lifecycle.thaw();
    }

    /// Reinitialize only when no listener, local dispatch, queued request, or
    /// I/O handler is active. This non-blocking form reports the exact busy
    /// counts and never races a slow host service.
    ///
    /// The same external GPU-idle/no-new-submit precondition as
    /// [`Self::reinit_packets`] applies.
    pub fn try_reinit_packets(&self) -> Result<(), ReinitBusy> {
        self.lifecycle.try_request_freeze()?;
        self.reset_quiescent_packets();
        self.lifecycle.thaw();
        Ok(())
    }

    fn reset_quiescent_packets(&self) {
        let base = self.host_ptr;
        // SAFETY: The public entry points establish both the external GPU-idle
        // precondition and host listener/worker quiescence before reaching here.
        unsafe {
            // Reset shutdown flag (may have been set by previous session use)
            std::ptr::write_volatile(base.add(BUF_OFF_SHUTDOWN) as *mut u32, 0);
            self.rebuild_packet_stacks();

            // Reset sideband bump allocator
            if !self.sideband_host_ptr.is_null() {
                std::ptr::write_volatile(
                    self.sideband_host_ptr.add(SIDEBAND_OFF_ALLOC) as *mut u64,
                    0u64,
                );
            }
        }
    }

    // ── Public accessors ────────────────────────────────────────

    /// Host-side pointer to the pinned hostcall buffer.
    pub fn host_ptr(&self) -> *mut u8 {
        self.host_ptr
    }

    /// Device-side pointer to the hostcall buffer (for kernel launch args).
    pub fn dev_ptr(&self) -> sys::CUdeviceptr {
        self.dev_ptr
    }

    /// Total size of the hostcall buffer in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Number of packet slots in the buffer.
    pub fn num_packets(&self) -> u16 {
        self.num_packets
    }

    /// Number of shards (0 = legacy unsharded mode).
    pub fn num_shards(&self) -> u32 {
        self.num_shards
    }

    /// Packets assigned to each shard (only meaningful when num_shards > 0).
    pub fn pkts_per_shard(&self) -> u32 {
        self.pkts_per_shard
    }

    /// Number each shard contributes to the shared global High-only pool.
    /// Total shared reserve is this value multiplied by `num_shards` (or this
    /// value directly in unsharded mode).
    pub fn high_reserved_per_shard(&self) -> u32 {
        self.high_reserved_per_shard
    }

    /// Total packet credits in the shared global High-only pool.
    pub fn high_reserved_packets(&self) -> u32 {
        shared_high_reserved_packets(
            self.num_packets,
            self.num_shards,
            self.pkts_per_shard,
            self.high_reserved_per_shard,
        )
    }

    /// Return task metadata currently stored in a packet slot.
    ///
    /// This is primarily useful to completion/wake integrations that retain a
    /// packet index. Returns `None` while the slot is FREE or if control changes
    /// during the snapshot; recognized legacy metadata decodes to Normal/task 0.
    pub fn packet_metadata(&self, index: u16) -> Option<HostcallMetadata> {
        if index >= self.num_packets {
            return None;
        }
        let pkt = self.packet_ptr(index);
        // A double acquire snapshot rejects a concurrent publish, completion,
        // cancellation, or reuse instead of returning a half-written task id.
        let control = unsafe { &*(pkt.add(PKT_OFF_CONTROL) as *const AtomicU32) };
        let before = control.load(Ordering::Acquire);
        let flags = control_flags(before);
        if flags != CONTROL_FILLED && flags != CONTROL_HOST_OWNED && flags & CONTROL_READY == 0 {
            return None;
        }
        // SAFETY: index was bounds-checked and the control acquire pairs with
        // the device's release publication of metadata_version.
        let metadata = unsafe { read_packet_metadata(pkt) };
        let after = control.load(Ordering::Acquire);
        if before != after {
            return None;
        }
        Some(metadata)
    }

    /// Host-side pointer to the sideband buffer for bulk data transfer.
    pub fn sideband_host_ptr(&self) -> *mut u8 {
        self.sideband_host_ptr
    }

    /// Device-side pointer to the sideband buffer (for kernel launch args).
    pub fn sideband_dev_ptr(&self) -> sys::CUdeviceptr {
        self.sideband_dev_ptr
    }

    /// Total size of the sideband buffer in bytes.
    pub fn sideband_size(&self) -> usize {
        self.sideband_size
    }

    // ── Internal helpers ───────────────────────────────────────

    /// Get a reference to the doorbell as an AtomicU64.
    fn doorbell(&self) -> &AtomicU64 {
        // SAFETY: BUF_OFF_DOORBELL is within the buffer header. The pointer is
        // 8-byte aligned (buffer is page-aligned from cuMemHostAlloc). The
        // resulting reference is valid for the lifetime of HostcallBuffer.
        unsafe { &*(self.host_ptr.add(BUF_OFF_DOORBELL) as *const AtomicU64) }
    }

    /// Get a reference to the ready_stack as an AtomicU64.
    fn ready_stack(&self) -> &AtomicU64 {
        // SAFETY: Same as doorbell() — BUF_OFF_READY_STACK is within the header,
        // 8-byte aligned, valid for the buffer's lifetime.
        unsafe { &*(self.host_ptr.add(BUF_OFF_READY_STACK) as *const AtomicU64) }
    }

    /// Acquire-snapshot whether every ready stack is empty. Shutdown callers
    /// establish the no-new-producer condition by synchronizing the GPU first;
    /// the listener repeats final scans until this predicate holds.
    fn ready_stacks_empty(&self) -> bool {
        if self.num_shards == 0 {
            return tagged_index(self.ready_stack().load(Ordering::Acquire)) == NULL_INDEX;
        }
        for shard in 0..self.num_shards {
            let entry_off = shard_entry_offset(BUFFER_HEADER_SIZE, shard);
            let ready = unsafe {
                &*(self.host_ptr.add(entry_off + SHARD_OFF_READY_STACK) as *const AtomicU64)
            };
            if tagged_index(ready.load(Ordering::Acquire)) != NULL_INDEX {
                return false;
            }
        }
        true
    }

    /// Get a reference to the shutdown flag as an AtomicU32.
    fn shutdown(&self) -> &AtomicU32 {
        // SAFETY: BUF_OFF_SHUTDOWN is within the buffer header, 4-byte aligned,
        // valid for the buffer's lifetime.
        unsafe { &*(self.host_ptr.add(BUF_OFF_SHUTDOWN) as *const AtomicU32) }
    }

    /// Get pointer to a packet by index. Handles both legacy and sharded layouts.
    fn packet_ptr(&self, index: u16) -> *mut u8 {
        // SAFETY: packet_offset / packet_offset_sharded compute offsets that stay
        // within the allocated buffer (index must be < num_packets, which is
        // guaranteed by the lock-free stack protocol — only indices that were
        // originally placed in the free stack can appear in the ready stack).
        unsafe {
            if self.num_shards == 0 {
                self.host_ptr.add(packet_offset(index))
            } else {
                self.host_ptr.add(packet_offset_sharded(
                    index,
                    BUFFER_HEADER_SIZE,
                    self.num_shards,
                ))
            }
        }
    }

    fn packet_pool(&self, index: u16) -> PacketPool {
        if is_high_reserved_packet(
            index,
            self.num_packets,
            self.num_shards,
            self.pkts_per_shard,
            self.high_reserved_per_shard,
        ) {
            PacketPool::HighShared
        } else {
            let shard = if self.num_shards == 0 || self.pkts_per_shard == 0 {
                0
            } else {
                (index as u32) / self.pkts_per_shard
            };
            PacketPool::General { shard }
        }
    }

    fn stack_for_pool(&self, pool: PacketPool) -> &AtomicU64 {
        let offset = match pool {
            PacketPool::HighShared => BUF_OFF_HIGH_FREE_STACK,
            PacketPool::General { shard } if self.num_shards != 0 => {
                shard_entry_offset(BUFFER_HEADER_SIZE, shard) + SHARD_OFF_FREE_STACK
            }
            PacketPool::General { .. } => BUF_OFF_FREE_STACK,
        };
        // SAFETY: every selected stack head is an aligned u64 inside the fixed
        // hostcall header/shard array and lives as long as this allocation.
        unsafe { &*(self.host_ptr.add(offset) as *const AtomicU64) }
    }

    fn push_free_from_host(&self, request: IoRequest) {
        let Some(generation) = request.generation else {
            return;
        };
        let pkt = self.packet_ptr(request.pkt_idx);
        // Publish IDLE first so packet_metadata() cannot accept half-cleared
        // fields under an unchanged READY snapshot. The packet remains
        // unreachable until the free-stack push below. Generation high bits
        // stay in metadata_flags so the next v3 request can advance all 43 bits.
        let control = unsafe { &*(pkt.add(PKT_OFF_CONTROL) as *const AtomicU32) };
        control.store(make_control(generation, 0), Ordering::Release);
        unsafe {
            std::ptr::write_volatile(pkt.add(PKT_OFF_METADATA_VERSION), 0);
            std::ptr::write_volatile(pkt.add(PKT_OFF_PRIORITY), Priority::Normal.as_raw());
            std::ptr::write_volatile(pkt.add(PKT_OFF_TASK_ID) as *mut u64, 0);
        }

        let stack = self.stack_for_pool(request.pool);
        loop {
            let old_head = stack.load(Ordering::Acquire);
            unsafe {
                std::ptr::write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, old_head);
            }
            let new_head = advance_tagged_head(old_head, request.pkt_idx);
            if stack
                .compare_exchange(old_head, new_head, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break;
            }
        }
    }

    fn drain_ready_stack(&self, stack: &AtomicU64) -> u64 {
        loop {
            let old_head = stack.load(Ordering::Acquire);
            if tagged_index(old_head) == NULL_INDEX {
                return old_head;
            }
            let empty = advance_tagged_head(old_head, NULL_INDEX);
            if stack
                .compare_exchange(old_head, empty, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return old_head;
            }
        }
    }

    unsafe fn snapshot_request(&self, index: u16, generation: Option<u64>) -> IoRequest {
        let pkt = self.packet_ptr(index);
        IoRequest {
            pkt_idx: index,
            service: std::ptr::read_volatile(pkt.add(PKT_OFF_SERVICE) as *const u32),
            metadata: read_packet_metadata(pkt),
            generation,
            pool: self.packet_pool(index),
        }
    }

    unsafe fn claim_packet(&self, index: u16) -> PacketClaim {
        let pkt = self.packet_ptr(index);
        let control = &*(pkt.add(PKT_OFF_CONTROL) as *const AtomicU32);
        let observed = control.load(Ordering::Acquire);
        let metadata_version = std::ptr::read_volatile(pkt.add(PKT_OFF_METADATA_VERSION));

        if metadata_version != PACKET_METADATA_VERSION {
            return if observed & CONTROL_FILLED != 0 {
                PacketClaim::Claimed(self.snapshot_request(index, None))
            } else {
                PacketClaim::Ignore
            };
        }

        let generation_metadata =
            std::ptr::read_volatile(pkt.add(PKT_OFF_METADATA_FLAGS) as *const u16);
        let generation = request_generation(observed, generation_metadata);
        match control_flags(observed) {
            CONTROL_FILLED => {
                let owned = make_control(generation, CONTROL_HOST_OWNED);
                match control.compare_exchange(observed, owned, Ordering::AcqRel, Ordering::Acquire)
                {
                    Ok(_) => PacketClaim::Claimed(self.snapshot_request(index, Some(generation))),
                    Err(actual) if control_flags(actual) == CONTROL_CANCELLED => {
                        PacketClaim::Cancelled(self.snapshot_request(index, Some(generation)))
                    }
                    Err(_) => PacketClaim::Ignore,
                }
            }
            CONTROL_CANCELLED => {
                PacketClaim::Cancelled(self.snapshot_request(index, Some(generation)))
            }
            _ => PacketClaim::Ignore,
        }
    }

    unsafe fn request_snapshot_matches(&self, request: &IoRequest, expected_flags: u32) -> bool {
        let Some(generation) = request.generation else {
            return true;
        };
        let pkt = self.packet_ptr(request.pkt_idx);
        let control = (&*(pkt.add(PKT_OFF_CONTROL) as *const AtomicU32)).load(Ordering::Acquire);
        if control_flags(control) != expected_flags
            || std::ptr::read_volatile(pkt.add(PKT_OFF_METADATA_VERSION)) != PACKET_METADATA_VERSION
            || request_generation(
                control,
                std::ptr::read_volatile(pkt.add(PKT_OFF_METADATA_FLAGS) as *const u16),
            ) != generation
            || std::ptr::read_volatile(pkt.add(PKT_OFF_SERVICE) as *const u32) != request.service
            || read_packet_metadata(pkt) != request.metadata
        {
            return false;
        }
        true
    }

    unsafe fn begin_request_processing(&self, request: IoRequest) -> bool {
        if self.request_snapshot_matches(&request, CONTROL_HOST_OWNED) {
            return true;
        }
        let Some(generation) = request.generation else {
            self.stale_rejected.fetch_add(1, Ordering::Relaxed);
            return false;
        };
        let pkt = self.packet_ptr(request.pkt_idx);
        let control = (&*(pkt.add(PKT_OFF_CONTROL) as *const AtomicU32)).load(Ordering::Acquire);
        let metadata_generation =
            std::ptr::read_volatile(pkt.add(PKT_OFF_METADATA_FLAGS) as *const u16);
        if request_generation(control, metadata_generation) == generation
            && control_flags(control) == CONTROL_CANCELLED
        {
            if request.service == SERVICE_PRIORITY_ECHO {
                self.record_cancelled_priority_echo(request);
            }
            self.processed_cancelled.fetch_add(1, Ordering::Relaxed);
            self.push_free_from_host(request);
        } else {
            self.stale_rejected.fetch_add(1, Ordering::Relaxed);
        }
        false
    }

    unsafe fn complete_request(&self, request: IoRequest, has_error: bool) {
        let pkt = self.packet_ptr(request.pkt_idx);
        let flags = if has_error {
            CONTROL_READY | CONTROL_ERROR
        } else {
            CONTROL_READY
        };
        let control = &*(pkt.add(PKT_OFF_CONTROL) as *const AtomicU32);

        let Some(generation) = request.generation else {
            control.store(flags, Ordering::Release);
            self.processed_completed.fetch_add(1, Ordering::Relaxed);
            if has_error {
                self.processed_errors.fetch_add(1, Ordering::Relaxed);
            }
            return;
        };

        let metadata_generation =
            std::ptr::read_volatile(pkt.add(PKT_OFF_METADATA_FLAGS) as *const u16);
        if std::ptr::read_volatile(pkt.add(PKT_OFF_METADATA_VERSION)) != PACKET_METADATA_VERSION
            || request_generation(
                make_control(generation, CONTROL_HOST_OWNED),
                metadata_generation,
            ) != generation
            || std::ptr::read_volatile(pkt.add(PKT_OFF_SERVICE) as *const u32) != request.service
            || read_packet_metadata(pkt) != request.metadata
        {
            self.stale_rejected.fetch_add(1, Ordering::Relaxed);
            return;
        }

        let expected = make_control(generation, CONTROL_HOST_OWNED);
        let desired = make_control(generation, flags);
        match control.compare_exchange(expected, desired, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {
                self.processed_completed.fetch_add(1, Ordering::Relaxed);
                if has_error {
                    self.processed_errors.fetch_add(1, Ordering::Relaxed);
                }
            }
            Err(actual)
                if request_generation(
                    actual,
                    std::ptr::read_volatile(pkt.add(PKT_OFF_METADATA_FLAGS) as *const u16),
                ) == generation
                    && control_flags(actual) == CONTROL_CANCELLED =>
            {
                self.processed_cancelled.fetch_add(1, Ordering::Relaxed);
                self.push_free_from_host(request);
            }
            Err(_) => {
                self.stale_rejected.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Return error-visible host processing counters.
    pub fn processing_metrics(&self) -> HostcallProcessingMetrics {
        HostcallProcessingMetrics {
            completed: self.processed_completed.load(Ordering::Acquire),
            errors: self.processed_errors.load(Ordering::Acquire),
            cancelled: self.processed_cancelled.load(Ordering::Acquire),
            stale_rejected: self.stale_rejected.load(Ordering::Acquire),
        }
    }

    /// Snapshot all host-observed priority echo events.
    pub fn priority_echo_events(&self) -> Vec<PriorityEchoEvent> {
        self.priority_echo_events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Clear prior echo evidence before a logically independent experiment.
    pub fn clear_priority_echo_events(&self) {
        self.priority_echo_events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        self.priority_echo_sequence.store(0, Ordering::Release);
    }

    /// Configure an explicit default-off test hook. `delay_micros` is used only
    /// by [`PriorityEchoTestHook::DelayNext`].
    pub fn set_priority_echo_test_hook(&self, hook: PriorityEchoTestHook, delay_micros: u64) {
        self.priority_echo_delay_micros
            .store(delay_micros, Ordering::Release);
        self.priority_echo_hook
            .store(hook as u32, Ordering::Release);
    }

    fn audit_free_stack(
        &self,
        mut head: u64,
        shared_high: bool,
        seen: &mut [bool],
        audit: &mut HostcallPoolAudit,
    ) {
        let mut traversed = 0usize;
        while tagged_index(head) != NULL_INDEX {
            let index = tagged_index(head);
            let index_usize = index as usize;
            if index_usize >= seen.len() || traversed >= seen.len() {
                audit.duplicate_entries = audit.duplicate_entries.saturating_add(1);
                break;
            }
            if seen[index_usize] {
                audit.duplicate_entries = audit.duplicate_entries.saturating_add(1);
                break;
            }
            seen[index_usize] = true;
            if index < 64 {
                if shared_high {
                    audit.shared_high_mask |= 1u64 << index;
                } else {
                    audit.general_mask |= 1u64 << index;
                }
            }
            // SAFETY: index was checked against the allocation's packet count.
            head = unsafe {
                std::ptr::read_volatile(self.packet_ptr(index).add(PKT_OFF_NEXT) as *const u64)
            };
            traversed += 1;
        }
    }

    /// Inspect packet ownership after GPU synchronization and listener join.
    ///
    /// This method does not create quiescence; the caller must establish it.
    /// The audit walks every free stack, rejects duplicate/cyclic entries, and
    /// separately checks that all packet controls are idle.
    pub fn quiescent_pool_audit(&self) -> HostcallPoolAudit {
        let mut audit = HostcallPoolAudit {
            ready_empty: true,
            ..HostcallPoolAudit::default()
        };
        let mut seen = vec![false; self.num_packets as usize];

        if self.num_shards == 0 {
            audit.ready_empty =
                tagged_index(self.ready_stack().load(Ordering::Acquire)) == NULL_INDEX;
            self.audit_free_stack(
                self.stack_for_pool(PacketPool::General { shard: 0 })
                    .load(Ordering::Acquire),
                false,
                &mut seen,
                &mut audit,
            );
        } else {
            for shard in 0..self.num_shards {
                let entry_off = shard_entry_offset(BUFFER_HEADER_SIZE, shard);
                let ready = unsafe {
                    &*(self.host_ptr.add(entry_off + SHARD_OFF_READY_STACK) as *const AtomicU64)
                };
                audit.ready_empty &= tagged_index(ready.load(Ordering::Acquire)) == NULL_INDEX;
                self.audit_free_stack(
                    self.stack_for_pool(PacketPool::General { shard })
                        .load(Ordering::Acquire),
                    false,
                    &mut seen,
                    &mut audit,
                );
            }
        }
        self.audit_free_stack(
            self.stack_for_pool(PacketPool::HighShared)
                .load(Ordering::Acquire),
            true,
            &mut seen,
            &mut audit,
        );

        for (index, was_seen) in seen.into_iter().enumerate() {
            let pkt = self.packet_ptr(index as u16);
            let control = unsafe {
                (&*(pkt.add(PKT_OFF_CONTROL) as *const AtomicU32)).load(Ordering::Acquire)
            };
            if control_flags(control) == 0 {
                audit.idle_packets = audit.idle_packets.saturating_add(1);
            } else {
                audit.non_idle_controls = audit.non_idle_controls.saturating_add(1);
            }
            if !was_seen {
                audit.missing_packets = audit.missing_packets.saturating_add(1);
            }
        }
        audit
    }

    /// Signal shutdown to the GPU.
    pub fn signal_shutdown(&self) {
        self.shutdown().store(1, Ordering::Release);
        self.lifecycle.notify_shutdown();
    }

    /// Run the host listener loop with real stdin. Blocks until shutdown is signaled.
    ///
    /// `on_print` is called for each PRINT service request with the message bytes.
    pub fn listen<F>(&self, on_print: F)
    where
        F: FnMut(&[u8]),
    {
        self.listen_unified(on_print, RealStdin);
    }

    /// Unified listener with I/O thread separation (host-scaling.3, ADR-6).
    ///
    /// Fast services (NOP, PRINT, TIME, PANIC) are handled inline on the listener thread.
    /// Blocking services (FILE I/O, STDIN) are offloaded to a dedicated I/O
    /// thread through a stable weighted-priority queue. Shutdown joins that
    /// worker; an OS call such as blocking stdin or accept is not forcibly
    /// cancellable and can therefore make teardown unbounded. The listener's
    /// `Arc<HostcallBuffer>` keeps mapped memory alive throughout that wait.
    pub fn listen_unified<F, S>(&self, mut on_print: F, stdin: S)
    where
        F: FnMut(&[u8]),
        S: StdinSource,
    {
        self.lifecycle.listener_started();
        let _listener_guard = ListenerLifecycleGuard {
            lifecycle: &self.lifecycle,
        };
        let io_queue = Arc::new(PriorityIoQueue::new(Arc::clone(&self.lifecycle)));

        std::thread::scope(|scope| {
            // Spawn I/O thread for blocking operations (FILE, STDIN)
            let worker_queue = Arc::clone(&io_queue);
            scope.spawn(move || {
                self.io_thread_loop(&worker_queue, stdin);
            });
            // Close on normal exit and while unwinding a listener callback, so
            // the scoped worker is not left blocked on an empty queue.
            let _queue_close_guard = QueueCloseGuard(&io_queue);

            let mut last_doorbell: u64 = 0;
            let mut idle_spins: u32 = 0;

            // Adaptive polling: spin fast for SPIN_PHASE_LIMIT iterations,
            // then switch to sleeping SLEEP_DURATION between polls.
            const SPIN_PHASE_LIMIT: u32 = 1_000; // ~10µs at ~100ns/spin
            const SLEEP_DURATION: std::time::Duration = std::time::Duration::from_micros(100);

            loop {
                // Shutdown is a drain request, not an immediate break. A GPU
                // may have published a generation-tagged packet and then won
                // its timeout CAS just before the kernel completed. The final
                // scan must observe that CANCELLED entry and return its credit.
                let shutting_down = self.shutdown().load(Ordering::Acquire) != 0;
                let force_drain = self.lifecycle.freeze_requested();
                let current_doorbell = self.doorbell().load(Ordering::Acquire);
                if current_doorbell == last_doorbell && !force_drain && !shutting_down {
                    idle_spins += 1;
                    if idle_spins <= SPIN_PHASE_LIMIT {
                        std::hint::spin_loop();
                    } else {
                        std::thread::sleep(SLEEP_DURATION);
                    }
                    continue;
                }

                last_doorbell = current_doorbell;
                idle_spins = 0;

                // Drain all ready stacks (1 global or N shard stacks)
                let stacks_to_scan = if self.num_shards == 0 {
                    1
                } else {
                    self.num_shards
                };
                for s in 0..stacks_to_scan {
                    // Atomically drain the ready stack: swap head with NULL to
                    // claim all enqueued packets. AcqRel ordering ensures we see
                    // all writes the GPU made before pushing to the ready stack.
                    let ready_head = if self.num_shards == 0 {
                        self.drain_ready_stack(self.ready_stack())
                    } else {
                        let entry_off = shard_entry_offset(BUFFER_HEADER_SIZE, s);
                        // SAFETY: entry_off + SHARD_OFF_READY_STACK is within the
                        // shard array region of the buffer, 8-byte aligned.
                        let shard_ready = unsafe {
                            &*(self.host_ptr.add(entry_off + SHARD_OFF_READY_STACK)
                                as *const AtomicU64)
                        };
                        self.drain_ready_stack(shard_ready)
                    };
                    if tagged_index(ready_head) == NULL_INDEX {
                        continue;
                    }

                    // Treiber push order is newest-first. Reverse each claimed
                    // chain before dispatch so requests of the same priority are
                    // enqueued in their stack linearization order rather than LIFO.
                    let mut drained = Vec::new();
                    let mut current = ready_head;
                    while tagged_index(current) != NULL_INDEX {
                        let idx = tagged_index(current);
                        let pkt = self.packet_ptr(idx);
                        // SAFETY: idx came from a ready stack and therefore names
                        // a packet inside this allocation.
                        let next =
                            unsafe { std::ptr::read_volatile(pkt.add(PKT_OFF_NEXT) as *const u64) };
                        drained.push(idx);
                        current = next;
                    }

                    self.lifecycle.dispatch_started(drained.len());
                    let mut dispatch_guard = DispatchBatchGuard {
                        lifecycle: &self.lifecycle,
                        remaining: drained.len(),
                    };
                    for idx in drained.into_iter().rev() {
                        unsafe {
                            match self.claim_packet(idx) {
                                PacketClaim::Cancelled(request) => {
                                    if request.service == SERVICE_PRIORITY_ECHO {
                                        self.record_cancelled_priority_echo(request);
                                    }
                                    self.processed_cancelled.fetch_add(1, Ordering::Relaxed);
                                    self.push_free_from_host(request);
                                }
                                PacketClaim::Ignore => {}
                                PacketClaim::Claimed(request) => {
                                    if self.begin_request_processing(request) {
                                        let pkt = self.packet_ptr(idx);
                                        match request.service {
                                            SERVICE_NOP => self.complete_request(request, false),
                                            SERVICE_PRINT => {
                                                self.handle_print(pkt, &mut on_print);
                                                self.complete_request(request, false);
                                            }
                                            SERVICE_TIME => {
                                                let has_error = self.handle_time(pkt);
                                                self.complete_request(request, has_error);
                                            }
                                            SERVICE_PANIC => {
                                                self.handle_panic(pkt);
                                                self.complete_request(request, false);
                                            }
                                            SERVICE_TRACE => {
                                                self.handle_trace(pkt);
                                                self.complete_request(request, false);
                                            }
                                            SERVICE_PRIORITY_ECHO => {
                                                let has_error =
                                                    self.handle_priority_echo(pkt, request);
                                                self.complete_request(request, has_error);
                                            }
                                            SERVICE_BULK_PRINT => {
                                                self.handle_bulk_print(pkt, &mut on_print);
                                                self.complete_request(request, false);
                                            }
                                            SERVICE_OPEN
                                            | SERVICE_WRITE
                                            | SERVICE_READ
                                            | SERVICE_CLOSE
                                            | SERVICE_STDIN
                                            | SERVICE_BULK_WRITE
                                            | SERVICE_BULK_READ
                                            | SERVICE_TCP_CONNECT
                                            | SERVICE_TCP_WRITE
                                            | SERVICE_TCP_READ
                                            | SERVICE_TCP_CLOSE
                                            | SERVICE_TCP_BIND
                                            | SERVICE_TCP_ACCEPT
                                            | SERVICE_TCP_BULK_WRITE
                                            | SERVICE_TCP_BULK_READ => {
                                                if !io_queue.push(request) {
                                                    std::ptr::write_volatile(
                                                        pkt.add(PKT_OFF_PAYLOAD) as *mut u64,
                                                        encode_error(ERR_RESOURCE_BUSY, 0),
                                                    );
                                                    self.complete_request(request, true);
                                                }
                                            }
                                            _ => {
                                                std::ptr::write_volatile(
                                                    pkt.add(PKT_OFF_PAYLOAD) as *mut u64,
                                                    encode_error(ERR_UNSUPPORTED, 0),
                                                );
                                                self.complete_request(request, true);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        dispatch_guard.finished_one();
                    }
                }

                if force_drain {
                    self.lifecycle.acknowledge_and_wait(self.shutdown());
                }
                if shutting_down && self.ready_stacks_empty() {
                    break;
                }
            }

            // QueueCloseGuard drains queued work before the scoped worker joins.
        });
    }

    /// I/O thread loop — processes blocking FILE and STDIN operations.
    ///
    /// Runs until the listener closes the queue and all queued work is drained.
    fn io_thread_loop<S: StdinSource>(&self, queue: &PriorityIoQueue, mut stdin: S) {
        let mut fd_table: HashMap<u64, FdResource> = HashMap::new();
        let mut next_fd: u64 = 1; // fd 0 is reserved

        while let Some(req) = queue.pop() {
            let _inflight_guard = InflightLifecycleGuard {
                lifecycle: &self.lifecycle,
            };
            // SAFETY: request was claimed by this listener. Generation and
            // metadata validation rejects cancellation-before-dequeue and any
            // stale queue entry before a service can touch the packet payload.
            if !unsafe { self.begin_request_processing(req) } {
                continue;
            }
            let pkt = self.packet_ptr(req.pkt_idx);
            // SAFETY: pkt_idx came from the listener's ready stack traversal, so
            // it is a valid packet index. All service handlers read/write within
            // the packet's payload region (offsets < PACKET_SIZE).
            let has_error = unsafe {
                match req.service {
                    SERVICE_OPEN => self.handle_open(pkt, &mut fd_table, &mut next_fd),
                    SERVICE_WRITE => self.handle_write(pkt, &mut fd_table),
                    SERVICE_READ => self.handle_read(pkt, &mut fd_table),
                    SERVICE_CLOSE => self.handle_close(pkt, &mut fd_table),
                    SERVICE_STDIN => self.handle_stdin_from_source(pkt, &mut stdin),
                    SERVICE_BULK_WRITE => self.handle_bulk_write(pkt, &mut fd_table),
                    SERVICE_BULK_READ => self.handle_bulk_read(pkt, &mut fd_table),
                    // TCP services
                    SERVICE_TCP_CONNECT => {
                        self.handle_tcp_connect(pkt, &mut fd_table, &mut next_fd)
                    }
                    SERVICE_TCP_WRITE => self.handle_tcp_write(pkt, &mut fd_table),
                    SERVICE_TCP_READ => self.handle_tcp_read(pkt, &mut fd_table),
                    SERVICE_TCP_CLOSE => self.handle_tcp_close(pkt, &mut fd_table),
                    SERVICE_TCP_BIND => self.handle_tcp_bind(pkt, &mut fd_table, &mut next_fd),
                    SERVICE_TCP_ACCEPT => self.handle_tcp_accept(pkt, &mut fd_table, &mut next_fd),
                    SERVICE_TCP_BULK_WRITE => self.handle_tcp_bulk_write(pkt, &mut fd_table),
                    SERVICE_TCP_BULK_READ => self.handle_tcp_bulk_read(pkt, &mut fd_table),
                    _ => true,
                }
            };

            // SAFETY: the pre-handler generation snapshot remained exclusively
            // host-owned. A concurrent timeout can only change HOST_OWNED to
            // CANCELLED; complete_request then discards and returns the packet.
            unsafe { self.complete_request(req, has_error) };
        }
    }

    /// Handle a PRINT service request.
    ///
    /// Reads message from lane 0's payload slots and calls the callback.
    unsafe fn handle_print<F>(&self, pkt: *mut u8, on_print: &mut F)
    where
        F: FnMut(&[u8]),
    {
        // SAFETY (applies to all service handlers): pkt points to a valid packet
        // obtained via packet_ptr(). PKT_OFF_PAYLOAD is within the packet. All
        // read_volatile/write_volatile target payload slots 0-7 which occupy bytes
        // [PKT_OFF_PAYLOAD .. PKT_OFF_PAYLOAD + 64) within the packet — well within
        // the packet's total size. Volatile access is required because the GPU wrote
        // these values and we must observe them.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        // Slot 0 = message length (u64)
        let msg_len = std::ptr::read_volatile(payload as *const u64) as usize;
        let msg_len = msg_len.min(PRINT_MAX_MSG_LEN);

        // Slots 1-7 = message bytes (up to 56 bytes)
        let msg_ptr = payload.add(8); // skip slot 0
        let mut msg_buf = [0u8; PRINT_MAX_MSG_LEN];
        for i in 0..msg_len {
            msg_buf[i] = std::ptr::read_volatile(msg_ptr.add(i));
        }

        // Read thread/block metadata from payload+64 (lane 1 area)
        let block_idx = std::ptr::read_volatile(payload.add(64) as *const u32);
        let thread_idx = std::ptr::read_volatile(payload.add(68) as *const u32);

        // Format: [B{block}.T{thread}] message
        let prefix = format!("[B{block_idx}.T{thread_idx}] ");
        let mut full_msg = Vec::with_capacity(prefix.len() + msg_len);
        full_msg.extend_from_slice(prefix.as_bytes());
        full_msg.extend_from_slice(&msg_buf[..msg_len]);

        on_print(&full_msg);
    }

    /// Handle SERVICE_BULK_PRINT: flush a buffer of length-prefixed print messages.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: sideband_offset — offset in sideband data region
    ///   Slot 1: data_len — total bytes of length-prefixed messages
    ///   Slot 2: block_idx (high 32) | thread_idx (low 32)
    ///
    /// Message format in sideband: `[u16 len][len bytes data]...`
    unsafe fn handle_bulk_print<F>(&self, pkt: *mut u8, on_print: &mut F)
    where
        F: FnMut(&[u8]),
    {
        // SAFETY: Same payload access pattern as handle_print. Additionally,
        // sideband_host_ptr + SIDEBAND_DATA_OFFSET + sideband_offset is within
        // the sideband buffer (the GPU's bump allocator ensures offset < capacity).
        let payload = pkt.add(PKT_OFF_PAYLOAD);
        let sideband_offset = std::ptr::read_volatile(payload as *const u64) as usize;
        let data_len = std::ptr::read_volatile(payload.add(8) as *const u64) as usize;
        let metadata = std::ptr::read_volatile(payload.add(16) as *const u64);
        let thread_idx = (metadata & 0xFFFF_FFFF) as u32;
        let block_idx = (metadata >> 32) as u32;

        if data_len == 0 || self.sideband_host_ptr.is_null() {
            return;
        }

        // Read messages from sideband
        let data_ptr = self
            .sideband_host_ptr
            .add(SIDEBAND_DATA_OFFSET + sideband_offset);
        let prefix = format!("[B{block_idx}.T{thread_idx}] ");

        let mut pos = 0;
        while pos + 2 <= data_len {
            let msg_len = u16::from_le_bytes([*data_ptr.add(pos), *data_ptr.add(pos + 1)]) as usize;
            if msg_len == 0 || pos + 2 + msg_len > data_len {
                break;
            }

            let mut full_msg = Vec::with_capacity(prefix.len() + msg_len);
            full_msg.extend_from_slice(prefix.as_bytes());
            for i in 0..msg_len {
                full_msg.push(*data_ptr.add(pos + 2 + i));
            }
            on_print(&full_msg);

            pos += 2 + msg_len;
        }
    }

    // ================================================================
    // FILE I/O handlers (gpu-std.3)
    // ================================================================

    /// Handle SERVICE_OPEN: open or create a file.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: low 32 bits = path length, high 32 bits = flags
    ///   Slots 1-7: path bytes (up to 56 bytes)
    /// Response payload (lane 0):
    ///   Slot 0: fd on success, FILE_ERROR_SENTINEL on error
    ///
    /// Returns true if the service itself encountered an error (not a file error).
    unsafe fn handle_open(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, FdResource>,
        next_fd: &mut u64,
    ) -> bool {
        // SAFETY: Same payload access pattern as handle_print — all slot reads/writes
        // are within the 64-byte payload region. Path bytes are clamped to
        // FILE_MAX_PATH_LEN (56 bytes = slots 1-7).
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        // Read slot 0: path_len (low 32) + flags (high 32)
        let slot0 = std::ptr::read_volatile(payload as *const u64);
        let path_len = (slot0 & 0xFFFF_FFFF) as usize;
        let flags = (slot0 >> 32) as u32;
        let path_len = path_len.min(FILE_MAX_PATH_LEN);

        // Read path bytes from slots 1-7
        let path_ptr = payload.add(8);
        let mut path_buf = [0u8; FILE_MAX_PATH_LEN];
        for i in 0..path_len {
            path_buf[i] = std::ptr::read_volatile(path_ptr.add(i));
        }

        let path_str = match std::str::from_utf8(&path_buf[..path_len]) {
            Ok(s) => s,
            Err(_) => {
                let e = std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid UTF-8 path");
                eprintln!("  [HOST] FILE OPEN ERROR: invalid UTF-8 path");
                return write_error_response(payload, &e);
            }
        };

        let file_result = match flags {
            FILE_OPEN_READ => File::open(path_str),
            FILE_OPEN_WRITE_CREATE => File::create(path_str),
            FILE_OPEN_APPEND => OpenOptions::new().append(true).create(true).open(path_str),
            _ => {
                let e = std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid open flags");
                eprintln!("  [HOST] FILE OPEN ERROR: invalid flags={flags}");
                return write_error_response(payload, &e);
            }
        };

        match file_result {
            Ok(file) => {
                let fd = *next_fd;
                *next_fd += 1;
                fd_table.insert(fd, FdResource::File(file));
                std::ptr::write_volatile(payload as *mut u64, fd);
                println!("  [HOST] FILE OPEN: \"{path_str}\" flags={flags} -> fd={fd}");
                false
            }
            Err(e) => {
                eprintln!("  [HOST] FILE OPEN ERROR: \"{path_str}\": {e}");
                write_error_response(payload, &e)
            }
        }
    }

    /// Handle SERVICE_WRITE: write data to an open file.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: fd (u64)
    ///   Slot 1: data length (u64)
    ///   Slots 2-7: data bytes (up to 48 bytes)
    /// Response payload (lane 0):
    ///   Slot 0: bytes written on success, FILE_ERROR_SENTINEL on error
    unsafe fn handle_write(&self, pkt: *mut u8, fd_table: &mut HashMap<u64, FdResource>) -> bool {
        // SAFETY: Same as handle_open — payload slot reads within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);
        let data_len = std::ptr::read_volatile(payload.add(8) as *const u64) as usize;
        let data_len = data_len.min(FILE_MAX_WRITE_LEN);

        // Read data bytes from slots 2-7
        let data_ptr = payload.add(16);
        let mut data_buf = [0u8; FILE_MAX_WRITE_LEN];
        for i in 0..data_len {
            data_buf[i] = std::ptr::read_volatile(data_ptr.add(i));
        }

        let file = match fd_table.get_mut(&fd) {
            Some(FdResource::File(f)) => f,
            Some(_) => {
                eprintln!("  [HOST] FILE WRITE ERROR: fd={fd} is not a file");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
                return true;
            }
            None => {
                eprintln!("  [HOST] FILE WRITE ERROR: invalid fd={fd}");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                return true;
            }
        };

        match file.write(&data_buf[..data_len]) {
            Ok(n) => {
                // Flush to ensure data is persisted
                let _ = file.flush();
                println!("  [HOST] FILE WRITE: fd={fd} {n} bytes written");
                std::ptr::write_volatile(payload as *mut u64, n as u64);
                false
            }
            Err(e) => {
                eprintln!("  [HOST] FILE WRITE ERROR: fd={fd}: {e}");
                write_error_response(payload, &e)
            }
        }
    }

    /// Handle SERVICE_READ: read data from an open file.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: fd (u64)
    ///   Slot 1: max bytes to read (u64)
    /// Response payload (lane 0):
    ///   Slot 0: bytes read on success, FILE_ERROR_SENTINEL on error
    ///   Slots 1-7: data bytes (up to 56 bytes)
    unsafe fn handle_read(&self, pkt: *mut u8, fd_table: &mut HashMap<u64, FdResource>) -> bool {
        // SAFETY: Same as handle_open — payload slot reads/writes within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);
        let max_len = std::ptr::read_volatile(payload.add(8) as *const u64) as usize;
        let max_len = max_len.min(FILE_MAX_READ_LEN);

        let file = match fd_table.get_mut(&fd) {
            Some(FdResource::File(f)) => f,
            Some(_) => {
                eprintln!("  [HOST] FILE READ ERROR: fd={fd} is not a file");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
                return true;
            }
            None => {
                eprintln!("  [HOST] FILE READ ERROR: invalid fd={fd}");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                return true;
            }
        };

        let mut read_buf = [0u8; FILE_MAX_READ_LEN];
        match file.read(&mut read_buf[..max_len]) {
            Ok(n) => {
                println!("  [HOST] FILE READ: fd={fd} {n} bytes read");
                // Write response: slot 0 = bytes read
                std::ptr::write_volatile(payload as *mut u64, n as u64);
                // Slots 1-7 = data bytes
                let dst = payload.add(8);
                for i in 0..n {
                    std::ptr::write_volatile(dst.add(i), read_buf[i]);
                }
                false
            }
            Err(e) => {
                eprintln!("  [HOST] FILE READ ERROR: fd={fd}: {e}");
                write_error_response(payload, &e)
            }
        }
    }

    /// Handle SERVICE_TIME: return wall-clock time.
    ///
    /// Response payload (lane 0):
    ///   Slot 0: seconds since Unix epoch (u64)
    ///   Slot 1: nanoseconds within second (u64)
    unsafe fn handle_time(&self, pkt: *mut u8) -> bool {
        // SAFETY: Same as handle_open — payload slot writes within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            Ok(duration) => {
                let secs = duration.as_secs();
                let nanos = duration.subsec_nanos() as u64;
                std::ptr::write_volatile(payload as *mut u64, secs);
                std::ptr::write_volatile(payload.add(8) as *mut u64, nanos);
                println!("  [HOST] TIME: epoch_secs={secs} nanos={nanos}");
            }
            Err(_) => {
                std::ptr::write_volatile(payload as *mut u64, FILE_ERROR_SENTINEL);
                std::ptr::write_volatile(payload.add(8) as *mut u64, 0);
            }
        }
        false
    }

    fn delay_next_priority_echo_if_requested(&self) {
        if self
            .priority_echo_hook
            .compare_exchange(
                PriorityEchoTestHook::DelayNext as u32,
                PriorityEchoTestHook::Disabled as u32,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            std::thread::sleep(std::time::Duration::from_micros(
                self.priority_echo_delay_micros.load(Ordering::Acquire),
            ));
        }
    }

    fn record_priority_echo_event(
        &self,
        request: IoRequest,
        request_nonce: u64,
        echo_nonce: u64,
        error_category: u16,
    ) -> (u64, u64) {
        let namespace = (request.metadata.task_id >> 32) as u32;
        let local_id = request.metadata.task_id as u32;
        let shared_high_reserved = request.pool == PacketPool::HighShared;
        let process_sequence = self
            .priority_echo_sequence
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1);
        let mut events = self
            .priority_echo_events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let process_count = events
            .iter()
            .filter(|event| {
                event.request_nonce == request_nonce && event.task_id == request.metadata.task_id
            })
            .count() as u64
            + 1;
        let event = PriorityEchoEvent {
            request_nonce,
            echo_nonce,
            task_id: request.metadata.task_id,
            namespace,
            local_id,
            priority: request.metadata.effective_priority,
            packet_index: request.pkt_idx,
            generation: request.generation.unwrap_or(0),
            shared_high_reserved,
            process_sequence,
            process_count,
            error_category,
        };
        events.push(event);
        drop(events);
        (process_sequence, process_count)
    }

    /// Record a submitted v3 echo that the device cancelled before the host
    /// could claim it. The host must not mutate its payload, but the independent
    /// event stream still records the timeout and consumes a pending DelayNext
    /// hook so a reused packet cannot inherit the prior request's test fault.
    unsafe fn record_cancelled_priority_echo(&self, request: IoRequest) {
        let payload = self.packet_ptr(request.pkt_idx).add(PKT_OFF_PAYLOAD) as *const u64;
        let request_nonce = std::ptr::read_volatile(payload.add(priority_echo::NONCE));
        self.delay_next_priority_echo_if_requested();
        self.record_priority_echo_event(request, request_nonce, request_nonce, ERR_HOST_TIMEOUT);
    }

    /// Validate and echo the identity/provenance of a live v3 request.
    unsafe fn handle_priority_echo(&self, pkt: *mut u8, request: IoRequest) -> bool {
        let payload = pkt.add(PKT_OFF_PAYLOAD) as *mut u64;
        let request_nonce = std::ptr::read_volatile(payload.add(priority_echo::NONCE));
        let namespace = (request.metadata.task_id >> 32) as u32;
        let local_id = request.metadata.task_id as u32;
        let shared_high_reserved = request.pool == PacketPool::HighShared;

        let hook = PriorityEchoTestHook::from_raw(self.priority_echo_hook.load(Ordering::Acquire));
        self.delay_next_priority_echo_if_requested();

        let mut error_category = 0u16;
        if request.generation.is_none()
            || request_nonce == 0
            || namespace == 0
            || local_id == 0
            || request.metadata.effective_priority != Priority::High
            || !shared_high_reserved
        {
            error_category = ERR_INVALID_INPUT;
        }
        if hook == PriorityEchoTestHook::ForceError {
            error_category = ERR_IO_ERROR;
        }

        let echo_nonce = if hook == PriorityEchoTestHook::CorruptEchoNonce {
            request_nonce ^ 1
        } else {
            request_nonce
        };
        let (process_sequence, process_count) =
            self.record_priority_echo_event(request, request_nonce, echo_nonce, error_category);

        std::ptr::write_volatile(payload.add(priority_echo::NONCE), echo_nonce);
        std::ptr::write_volatile(
            payload.add(priority_echo::TASK_ID),
            request.metadata.task_id,
        );
        std::ptr::write_volatile(
            payload.add(priority_echo::PRIORITY),
            request.metadata.effective_priority.as_raw() as u64,
        );
        std::ptr::write_volatile(
            payload.add(priority_echo::PACKET_INDEX),
            request.pkt_idx as u64,
        );
        std::ptr::write_volatile(
            payload.add(priority_echo::SHARED_HIGH_RESERVED),
            shared_high_reserved as u64,
        );
        std::ptr::write_volatile(
            payload.add(priority_echo::PROCESS_SEQUENCE),
            process_sequence,
        );
        std::ptr::write_volatile(payload.add(priority_echo::PROCESS_COUNT), process_count);
        std::ptr::write_volatile(
            payload.add(priority_echo::ERROR_CATEGORY),
            error_category as u64,
        );
        if error_category != 0 {
            std::ptr::write_volatile(payload, encode_error(error_category, 0));
        }
        error_category != 0
    }

    /// Handle SERVICE_PANIC: receive and display a GPU panic message.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: metadata (threadIdx.x, blockIdx.x, msg_len packed)
    ///   Slots 1-7: panic message bytes (up to 56 bytes)
    /// Response: CONTROL_READY (no error — GPU will trap regardless)
    unsafe fn handle_panic(&self, pkt: *mut u8) -> bool {
        // SAFETY: Same as handle_open — payload slot reads within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        // Decode metadata from slot 0
        let meta = std::ptr::read_volatile(payload as *const u64);
        let thread_idx = panic_thread_idx(meta);
        let block_idx = panic_block_idx(meta);
        let msg_len = panic_msg_len(meta) as usize;
        let msg_len = msg_len.min(PANIC_MAX_MSG_LEN);

        // Read message bytes from slots 1-7
        let msg_ptr = payload.add(8);
        let mut msg_buf = [0u8; PANIC_MAX_MSG_LEN];
        for i in 0..msg_len {
            msg_buf[i] = std::ptr::read_volatile(msg_ptr.add(i));
        }

        let msg = std::str::from_utf8(&msg_buf[..msg_len]).unwrap_or("<invalid UTF-8>");
        eprintln!("\x1b[1;31m[GPU PANIC]\x1b[0m block={block_idx} thread={thread_idx}: {msg}");

        false // No error — GPU thread will trap after receiving response
    }

    /// Handle SERVICE_TRACE: receive and display a structured trace event.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: metadata (threadIdx:16 | blockIdx:16 | level:8 | msg_len:8 | lane_id:16)
    ///   Slot 1: clock64 timestamp (u64)
    ///   Slots 2-7: message bytes (up to 48 bytes)
    /// Response: CONTROL_READY (no error)
    unsafe fn handle_trace(&self, pkt: *mut u8) -> bool {
        // SAFETY: Same as handle_open — payload slot reads within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        // Decode metadata from slot 0
        let meta = std::ptr::read_volatile(payload as *const u64);
        let thread_idx = trace_thread_idx(meta);
        let block_idx = trace_block_idx(meta);
        let level = trace_level(meta);
        let msg_len = trace_msg_len(meta) as usize;
        let msg_len = msg_len.min(TRACE_MAX_MSG_LEN);

        // Slot 1: timestamp
        let timestamp = std::ptr::read_volatile(payload.add(8) as *const u64);

        // Slots 2-7: message bytes (starting at offset 16)
        let msg_ptr = payload.add(16);
        let mut msg_buf = [0u8; TRACE_MAX_MSG_LEN];
        for i in 0..msg_len {
            msg_buf[i] = std::ptr::read_volatile(msg_ptr.add(i));
        }

        let msg = std::str::from_utf8(&msg_buf[..msg_len]).unwrap_or("<invalid UTF-8>");
        let level_str = match level {
            TRACE_LEVEL_DEBUG => "DEBUG",
            TRACE_LEVEL_INFO => "INFO",
            TRACE_LEVEL_WARN => "WARN",
            TRACE_LEVEL_ERROR => "ERROR",
            _ => "UNKNOWN",
        };
        let color = match level {
            TRACE_LEVEL_DEBUG => "\x1b[36m",   // cyan
            TRACE_LEVEL_INFO => "\x1b[32m",    // green
            TRACE_LEVEL_WARN => "\x1b[33m",    // yellow
            TRACE_LEVEL_ERROR => "\x1b[1;31m", // bold red
            _ => "\x1b[0m",
        };
        eprintln!("{color}[GPU {level_str}]\x1b[0m B{block_idx}.T{thread_idx} @{timestamp}: {msg}");

        false
    }

    /// Handle SERVICE_CLOSE: close an open file.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: fd (u64)
    /// Response payload (lane 0):
    ///   Slot 0: 0 on success, FILE_ERROR_SENTINEL on error
    unsafe fn handle_close(&self, pkt: *mut u8, fd_table: &mut HashMap<u64, FdResource>) -> bool {
        // SAFETY: Same as handle_open — payload slot reads/writes within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);

        match fd_table.remove(&fd) {
            Some(resource) => {
                // Resource is dropped here, which closes it
                let kind = match &resource {
                    FdResource::File(_) => "FILE",
                    FdResource::TcpStream(_) => "TCP STREAM",
                    FdResource::TcpListener(_) => "TCP LISTENER",
                };
                drop(resource);
                println!("  [HOST] CLOSE: fd={fd} ({kind}) closed");
                std::ptr::write_volatile(payload as *mut u64, 0);
                false
            }
            None => {
                eprintln!("  [HOST] CLOSE ERROR: invalid fd={fd}");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                true
            }
        }
    }
    /// Handle SERVICE_STDIN using a `StdinSource` abstraction.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: max bytes to read (u64)
    /// Response payload (lane 0):
    ///   Slot 0: bytes read (u64)
    ///   Slots 1-7: data bytes (up to 56 bytes)
    unsafe fn handle_stdin_from_source<S: StdinSource>(&self, pkt: *mut u8, stdin: &mut S) -> bool {
        // SAFETY: Same as handle_open — payload slot reads/writes within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);
        let max_len = std::ptr::read_volatile(payload as *const u64) as usize;
        let max_len = max_len.min(STDIN_MAX_READ_LEN);

        let mut buf = [0u8; STDIN_MAX_READ_LEN];
        let n = stdin.read_line_bytes(&mut buf[..max_len]);

        println!("  [HOST] STDIN: {n} bytes");
        std::ptr::write_volatile(payload as *mut u64, n as u64);
        let dst = payload.add(8);
        for i in 0..n {
            std::ptr::write_volatile(dst.add(i), buf[i]);
        }
        false
    }

    /// Handle SERVICE_BULK_WRITE: write sideband data to an open file.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: fd (u64)
    ///   Slot 1: sideband_offset (u64)
    ///   Slot 2: length (u64)
    /// Response payload (lane 0):
    ///   Slot 0: bytes written on success, FILE_ERROR_SENTINEL on error
    unsafe fn handle_bulk_write(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, FdResource>,
    ) -> bool {
        // SAFETY: Payload slot reads within packet bounds. Sideband access at
        // SIDEBAND_DATA_OFFSET + sb_offset is bounds-checked against sideband
        // capacity below. from_raw_parts creates a slice within the sideband region.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);
        let sb_offset = std::ptr::read_volatile(payload.add(8) as *const u64) as usize;
        let length = std::ptr::read_volatile(payload.add(16) as *const u64) as usize;

        // Bounds check against sideband capacity
        let capacity = std::ptr::read_volatile(
            self.sideband_host_ptr.add(SIDEBAND_OFF_CAPACITY) as *const u64
        ) as usize;
        if sb_offset + length > capacity {
            eprintln!(
                "  [HOST] BULK WRITE ERROR: offset={sb_offset} + len={length} > capacity={capacity}"
            );
            std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
            return true;
        }

        let file = match fd_table.get_mut(&fd) {
            Some(FdResource::File(f)) => f,
            Some(_) => {
                eprintln!("  [HOST] BULK WRITE ERROR: fd={fd} is not a file");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
                return true;
            }
            None => {
                eprintln!("  [HOST] BULK WRITE ERROR: invalid fd={fd}");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                return true;
            }
        };

        let data_ptr = self.sideband_host_ptr.add(SIDEBAND_DATA_OFFSET + sb_offset);
        let data = std::slice::from_raw_parts(data_ptr, length);

        match file.write_all(data) {
            Ok(()) => {
                let _ = file.flush();
                println!("  [HOST] BULK WRITE: fd={fd} {length} bytes written");
                std::ptr::write_volatile(payload as *mut u64, length as u64);
                false
            }
            Err(e) => {
                eprintln!("  [HOST] BULK WRITE ERROR: fd={fd}: {e}");
                write_error_response(payload, &e)
            }
        }
    }

    /// Handle SERVICE_BULK_READ: read file data into sideband buffer.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: fd (u64)
    ///   Slot 1: sideband_offset (u64)
    ///   Slot 2: max_length (u64)
    /// Response payload (lane 0):
    ///   Slot 0: bytes read on success, FILE_ERROR_SENTINEL on error
    unsafe fn handle_bulk_read(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, FdResource>,
    ) -> bool {
        // SAFETY: Same as handle_bulk_write — payload reads within packet bounds,
        // sideband access is bounds-checked against capacity below.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);
        let sb_offset = std::ptr::read_volatile(payload.add(8) as *const u64) as usize;
        let max_length = std::ptr::read_volatile(payload.add(16) as *const u64) as usize;

        // Bounds check
        let capacity = std::ptr::read_volatile(
            self.sideband_host_ptr.add(SIDEBAND_OFF_CAPACITY) as *const u64
        ) as usize;
        if sb_offset + max_length > capacity {
            eprintln!(
                "  [HOST] BULK READ ERROR: offset={sb_offset} + len={max_length} > capacity={capacity}"
            );
            std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
            return true;
        }

        let file = match fd_table.get_mut(&fd) {
            Some(FdResource::File(f)) => f,
            Some(_) => {
                eprintln!("  [HOST] BULK READ ERROR: fd={fd} is not a file");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
                return true;
            }
            None => {
                eprintln!("  [HOST] BULK READ ERROR: invalid fd={fd}");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                return true;
            }
        };

        let data_ptr = self.sideband_host_ptr.add(SIDEBAND_DATA_OFFSET + sb_offset);
        let buf = std::slice::from_raw_parts_mut(data_ptr, max_length);

        match file.read(buf) {
            Ok(n) => {
                println!("  [HOST] BULK READ: fd={fd} {n} bytes read");
                std::ptr::write_volatile(payload as *mut u64, n as u64);
                false
            }
            Err(e) => {
                eprintln!("  [HOST] BULK READ ERROR: fd={fd}: {e}");
                write_error_response(payload, &e)
            }
        }
    }

    // ================================================================
    // TCP networking handlers
    // ================================================================

    /// Extract an address string from packet payload slots 1-7.
    ///
    /// Slot 0 contains `port(u32) | addr_len(u32)` packed as `u64`.
    /// Returns `(address_string, port)` or writes an error response and returns `None`.
    unsafe fn extract_tcp_addr(&self, payload: *mut u8) -> Option<(String, u16)> {
        // SAFETY: payload points to a valid packet payload region. Slot reads
        // (0 and 1-7) are within the 64-byte payload. addr_len is clamped to
        // TCP_MAX_ADDR_LEN (56 bytes).
        let slot0 = std::ptr::read_volatile(payload as *const u64);
        let port = (slot0 & 0xFFFF_FFFF) as u32;
        let addr_len = ((slot0 >> 32) & 0xFFFF_FFFF) as usize;
        let addr_len = addr_len.min(TCP_MAX_ADDR_LEN);

        // Read address bytes from slots 1-7
        let addr_ptr = payload.add(8);
        let mut addr_buf = [0u8; TCP_MAX_ADDR_LEN];
        for i in 0..addr_len {
            addr_buf[i] = std::ptr::read_volatile(addr_ptr.add(i));
        }

        match std::str::from_utf8(&addr_buf[..addr_len]) {
            Ok(s) => Some((s.to_string(), port as u16)),
            Err(_) => None,
        }
    }

    /// Handle SERVICE_TCP_CONNECT: connect to a remote TCP address:port.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: `port(u32) | addr_len(u32)` packed as `u64`
    ///   Slots 1-7: address string (up to 56 bytes)
    /// Response payload (lane 0):
    ///   Slot 0: socket fd on success, encoded error on failure
    unsafe fn handle_tcp_connect(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, FdResource>,
        next_fd: &mut u64,
    ) -> bool {
        // SAFETY: Same as handle_open — payload slot reads/writes within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let (addr, port) = match self.extract_tcp_addr(payload) {
            Some(v) => v,
            None => {
                eprintln!("  [HOST] TCP CONNECT ERROR: invalid UTF-8 address");
                let e = std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid UTF-8 addr");
                return write_error_response(payload, &e);
            }
        };

        let socket_addr = format!("{addr}:{port}");
        match TcpStream::connect(&socket_addr) {
            Ok(stream) => {
                let fd = *next_fd;
                *next_fd += 1;
                fd_table.insert(fd, FdResource::TcpStream(stream));
                std::ptr::write_volatile(payload as *mut u64, fd);
                println!("  [HOST] TCP CONNECT: \"{socket_addr}\" -> fd={fd}");
                false
            }
            Err(e) => {
                eprintln!("  [HOST] TCP CONNECT ERROR: \"{socket_addr}\": {e}");
                write_error_response(payload, &e)
            }
        }
    }

    /// Handle SERVICE_TCP_WRITE: write inline data to a TCP socket (up to 48 bytes).
    ///
    /// Request payload (lane 0):
    ///   Slot 0: fd (u64)
    ///   Slot 1: data length (u64)
    ///   Slots 2-7: data bytes (up to 48 bytes)
    /// Response payload (lane 0):
    ///   Slot 0: bytes written on success, encoded error on failure
    unsafe fn handle_tcp_write(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, FdResource>,
    ) -> bool {
        // SAFETY: Same as handle_open — payload slot reads/writes within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);
        let data_len = std::ptr::read_volatile(payload.add(8) as *const u64) as usize;
        let data_len = data_len.min(TCP_MAX_WRITE_LEN);

        // Read data bytes from slots 2-7
        let data_ptr = payload.add(16);
        let mut data_buf = [0u8; TCP_MAX_WRITE_LEN];
        for i in 0..data_len {
            data_buf[i] = std::ptr::read_volatile(data_ptr.add(i));
        }

        let stream = match fd_table.get_mut(&fd) {
            Some(FdResource::TcpStream(s)) => s,
            Some(_) => {
                eprintln!("  [HOST] TCP WRITE ERROR: fd={fd} is not a TCP stream");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
                return true;
            }
            None => {
                eprintln!("  [HOST] TCP WRITE ERROR: invalid fd={fd}");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                return true;
            }
        };

        match stream.write(&data_buf[..data_len]) {
            Ok(n) => {
                let _ = stream.flush();
                println!("  [HOST] TCP WRITE: fd={fd} {n} bytes written");
                std::ptr::write_volatile(payload as *mut u64, n as u64);
                false
            }
            Err(e) => {
                eprintln!("  [HOST] TCP WRITE ERROR: fd={fd}: {e}");
                write_error_response(payload, &e)
            }
        }
    }

    /// Handle SERVICE_TCP_READ: read inline data from a TCP socket (up to 56 bytes).
    ///
    /// Request payload (lane 0):
    ///   Slot 0: fd (u64)
    ///   Slot 1: max bytes to read (u64)
    /// Response payload (lane 0):
    ///   Slot 0: bytes read on success, encoded error on failure
    ///   Slots 1-7: data bytes (up to 56 bytes)
    unsafe fn handle_tcp_read(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, FdResource>,
    ) -> bool {
        // SAFETY: Same as handle_open — payload slot reads/writes within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);
        let max_len = std::ptr::read_volatile(payload.add(8) as *const u64) as usize;
        let max_len = max_len.min(TCP_MAX_READ_LEN);

        let stream = match fd_table.get_mut(&fd) {
            Some(FdResource::TcpStream(s)) => s,
            Some(_) => {
                eprintln!("  [HOST] TCP READ ERROR: fd={fd} is not a TCP stream");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
                return true;
            }
            None => {
                eprintln!("  [HOST] TCP READ ERROR: invalid fd={fd}");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                return true;
            }
        };

        let mut read_buf = [0u8; TCP_MAX_READ_LEN];
        match stream.read(&mut read_buf[..max_len]) {
            Ok(n) => {
                println!("  [HOST] TCP READ: fd={fd} {n} bytes read");
                // Write response: slot 0 = bytes read
                std::ptr::write_volatile(payload as *mut u64, n as u64);
                // Slots 1-7 = data bytes
                let dst = payload.add(8);
                for i in 0..n {
                    std::ptr::write_volatile(dst.add(i), read_buf[i]);
                }
                false
            }
            Err(e) => {
                eprintln!("  [HOST] TCP READ ERROR: fd={fd}: {e}");
                write_error_response(payload, &e)
            }
        }
    }

    /// Handle SERVICE_TCP_CLOSE: close a TCP socket (stream or listener).
    ///
    /// The fd namespace is shared between files and sockets. This handler
    /// specifically expects a TCP resource. For generic close, use SERVICE_CLOSE.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: fd (u64)
    /// Response payload (lane 0):
    ///   Slot 0: 0 on success, encoded error on failure
    unsafe fn handle_tcp_close(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, FdResource>,
    ) -> bool {
        // SAFETY: Same as handle_open — payload slot reads/writes within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);

        match fd_table.remove(&fd) {
            Some(resource) => {
                let kind = match &resource {
                    FdResource::TcpStream(_) => "TCP STREAM",
                    FdResource::TcpListener(_) => "TCP LISTENER",
                    FdResource::File(_) => "FILE (via TCP_CLOSE)",
                };
                drop(resource);
                println!("  [HOST] TCP CLOSE: fd={fd} ({kind}) closed");
                std::ptr::write_volatile(payload as *mut u64, 0);
                false
            }
            None => {
                eprintln!("  [HOST] TCP CLOSE ERROR: invalid fd={fd}");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                true
            }
        }
    }

    /// Handle SERVICE_TCP_BIND: bind and listen on a local TCP address:port.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: `port(u32) | addr_len(u32)` packed as `u64`
    ///   Slots 1-7: bind address string (up to 56 bytes)
    /// Response payload (lane 0):
    ///   Slot 0: listener fd on success, encoded error on failure
    unsafe fn handle_tcp_bind(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, FdResource>,
        next_fd: &mut u64,
    ) -> bool {
        // SAFETY: Same as handle_open — payload slot reads/writes within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let (addr, port) = match self.extract_tcp_addr(payload) {
            Some(v) => v,
            None => {
                eprintln!("  [HOST] TCP BIND ERROR: invalid UTF-8 address");
                let e = std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid UTF-8 addr");
                return write_error_response(payload, &e);
            }
        };

        let socket_addr = format!("{addr}:{port}");
        match TcpListener::bind(&socket_addr) {
            Ok(listener) => {
                let fd = *next_fd;
                *next_fd += 1;
                fd_table.insert(fd, FdResource::TcpListener(listener));
                std::ptr::write_volatile(payload as *mut u64, fd);
                println!("  [HOST] TCP BIND: \"{socket_addr}\" -> fd={fd}");
                false
            }
            Err(e) => {
                eprintln!("  [HOST] TCP BIND ERROR: \"{socket_addr}\": {e}");
                write_error_response(payload, &e)
            }
        }
    }

    /// Handle SERVICE_TCP_ACCEPT: accept a connection on a TCP listener fd.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: listener fd (u64)
    /// Response payload (lane 0):
    ///   Slot 0: new stream fd on success, encoded error on failure
    unsafe fn handle_tcp_accept(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, FdResource>,
        next_fd: &mut u64,
    ) -> bool {
        // SAFETY: Same as handle_open — payload slot reads/writes within packet bounds.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let listener_fd = std::ptr::read_volatile(payload as *const u64);

        // We need to borrow the listener immutably, then insert the new stream.
        // accept() takes &self on TcpListener, so no mutable borrow conflict.
        let accept_result = match fd_table.get(&listener_fd) {
            Some(FdResource::TcpListener(l)) => l.accept(),
            Some(_) => {
                eprintln!("  [HOST] TCP ACCEPT ERROR: fd={listener_fd} is not a TCP listener");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
                return true;
            }
            None => {
                eprintln!("  [HOST] TCP ACCEPT ERROR: invalid fd={listener_fd}");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                return true;
            }
        };

        match accept_result {
            Ok((stream, peer_addr)) => {
                let fd = *next_fd;
                *next_fd += 1;
                fd_table.insert(fd, FdResource::TcpStream(stream));
                std::ptr::write_volatile(payload as *mut u64, fd);
                println!("  [HOST] TCP ACCEPT: listener fd={listener_fd} -> stream fd={fd} from {peer_addr}");
                false
            }
            Err(e) => {
                eprintln!("  [HOST] TCP ACCEPT ERROR: fd={listener_fd}: {e}");
                write_error_response(payload, &e)
            }
        }
    }

    /// Handle SERVICE_TCP_BULK_WRITE: write sideband buffer data to a TCP socket.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: fd (u64)
    ///   Slot 1: sideband_offset (u64)
    ///   Slot 2: length (u64)
    /// Response payload (lane 0):
    ///   Slot 0: bytes written on success, encoded error on failure
    unsafe fn handle_tcp_bulk_write(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, FdResource>,
    ) -> bool {
        // SAFETY: Same as handle_bulk_write — payload reads within packet bounds,
        // sideband access is bounds-checked against capacity below.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);
        let sb_offset = std::ptr::read_volatile(payload.add(8) as *const u64) as usize;
        let length = std::ptr::read_volatile(payload.add(16) as *const u64) as usize;

        // Bounds check against sideband capacity
        let capacity = std::ptr::read_volatile(
            self.sideband_host_ptr.add(SIDEBAND_OFF_CAPACITY) as *const u64
        ) as usize;
        if sb_offset + length > capacity {
            eprintln!(
                "  [HOST] TCP BULK WRITE ERROR: offset={sb_offset} + len={length} > capacity={capacity}"
            );
            std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
            return true;
        }

        let stream = match fd_table.get_mut(&fd) {
            Some(FdResource::TcpStream(s)) => s,
            Some(_) => {
                eprintln!("  [HOST] TCP BULK WRITE ERROR: fd={fd} is not a TCP stream");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
                return true;
            }
            None => {
                eprintln!("  [HOST] TCP BULK WRITE ERROR: invalid fd={fd}");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                return true;
            }
        };

        let data_ptr = self.sideband_host_ptr.add(SIDEBAND_DATA_OFFSET + sb_offset);
        let data = std::slice::from_raw_parts(data_ptr, length);

        match stream.write_all(data) {
            Ok(()) => {
                let _ = stream.flush();
                println!("  [HOST] TCP BULK WRITE: fd={fd} {length} bytes written");
                std::ptr::write_volatile(payload as *mut u64, length as u64);
                false
            }
            Err(e) => {
                eprintln!("  [HOST] TCP BULK WRITE ERROR: fd={fd}: {e}");
                write_error_response(payload, &e)
            }
        }
    }

    /// Handle SERVICE_TCP_BULK_READ: read from a TCP socket into sideband buffer.
    ///
    /// Request payload (lane 0):
    ///   Slot 0: fd (u64)
    ///   Slot 1: sideband_offset (u64)
    ///   Slot 2: max_length (u64)
    /// Response payload (lane 0):
    ///   Slot 0: bytes read on success, encoded error on failure
    unsafe fn handle_tcp_bulk_read(
        &self,
        pkt: *mut u8,
        fd_table: &mut HashMap<u64, FdResource>,
    ) -> bool {
        // SAFETY: Same as handle_bulk_write — payload reads within packet bounds,
        // sideband access is bounds-checked against capacity below.
        let payload = pkt.add(PKT_OFF_PAYLOAD);

        let fd = std::ptr::read_volatile(payload as *const u64);
        let sb_offset = std::ptr::read_volatile(payload.add(8) as *const u64) as usize;
        let max_length = std::ptr::read_volatile(payload.add(16) as *const u64) as usize;

        // Bounds check
        let capacity = std::ptr::read_volatile(
            self.sideband_host_ptr.add(SIDEBAND_OFF_CAPACITY) as *const u64
        ) as usize;
        if sb_offset + max_length > capacity {
            eprintln!(
                "  [HOST] TCP BULK READ ERROR: offset={sb_offset} + len={max_length} > capacity={capacity}"
            );
            std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
            return true;
        }

        let stream = match fd_table.get_mut(&fd) {
            Some(FdResource::TcpStream(s)) => s,
            Some(_) => {
                eprintln!("  [HOST] TCP BULK READ ERROR: fd={fd} is not a TCP stream");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_INPUT, 0));
                return true;
            }
            None => {
                eprintln!("  [HOST] TCP BULK READ ERROR: invalid fd={fd}");
                std::ptr::write_volatile(payload as *mut u64, encode_error(ERR_INVALID_FD, 0));
                return true;
            }
        };

        let data_ptr = self.sideband_host_ptr.add(SIDEBAND_DATA_OFFSET + sb_offset);
        let buf = std::slice::from_raw_parts_mut(data_ptr, max_length);

        match stream.read(buf) {
            Ok(n) => {
                println!("  [HOST] TCP BULK READ: fd={fd} {n} bytes read");
                std::ptr::write_volatile(payload as *mut u64, n as u64);
                false
            }
            Err(e) => {
                eprintln!("  [HOST] TCP BULK READ ERROR: fd={fd}: {e}");
                write_error_response(payload, &e)
            }
        }
    }

    /// Listen with both a print callback and a canned stdin provider.
    ///
    /// Convenience wrapper around `listen_unified` with `CannedStdin`.
    pub fn listen_with_stdin<F>(&self, on_print: F, stdin_data: Vec<u8>)
    where
        F: FnMut(&[u8]),
    {
        self.listen_unified(on_print, CannedStdin::new(stdin_data));
    }
}

// ================================================================
// HostcallSession — persistent listener across kernel launches
// ================================================================

/// A persistent hostcall session that keeps the listener thread alive
/// across multiple kernel launches.
///
/// Lifecycle:
/// ```text
/// let session = HostcallSession::start(64)?;
/// // Launch kernel A
/// launch_kernel(session.dev_ptr(), ...);
/// dev.synchronize()?;
/// session.reinit_packets();
/// // Launch kernel B (same hostcall buffer, same listener)
/// launch_kernel(session.dev_ptr(), ...);
/// dev.synchronize()?;
/// session.shutdown();
/// ```
///
/// File handles opened by one kernel persist for subsequent kernels
/// within the same session.
pub struct HostcallSession {
    buf: std::sync::Arc<HostcallBuffer>,
    listener_handle: Option<std::thread::JoinHandle<()>>,
}

impl HostcallSession {
    /// Start a new session with the given packet count.
    /// Spawns listener + I/O threads immediately.
    pub fn start(num_packets: u16) -> Result<Self, HostcallError> {
        let buf = HostcallBuffer::new(num_packets)?;
        let buf = std::sync::Arc::new(buf);
        let buf_listener = std::sync::Arc::clone(&buf);

        let listener_handle = std::thread::spawn(move || {
            buf_listener.listen(|_msg| {
                // Print messages handled by handle_print — on_print is for test capture
            });
        });

        Ok(Self {
            buf,
            listener_handle: Some(listener_handle),
        })
    }

    /// Start a new session with a custom print callback.
    pub fn start_with_print<F>(num_packets: u16, on_print: F) -> Result<Self, HostcallError>
    where
        F: FnMut(&[u8]) + Send + 'static,
    {
        let buf = HostcallBuffer::new(num_packets)?;
        let buf = std::sync::Arc::new(buf);
        let buf_listener = std::sync::Arc::clone(&buf);

        let listener_handle = std::thread::spawn(move || {
            buf_listener.listen(on_print);
        });

        Ok(Self {
            buf,
            listener_handle: Some(listener_handle),
        })
    }

    /// Get the device pointer for kernel launch args.
    pub fn dev_ptr(&self) -> sys::CUdeviceptr {
        self.buf.dev_ptr
    }

    /// Get the sideband device pointer for bulk transfer args.
    pub fn sideband_dev_ptr(&self) -> sys::CUdeviceptr {
        self.buf.sideband_dev_ptr
    }

    /// Reinitialize packet pool between kernel launches.
    ///
    /// MUST be called after `dev.synchronize()` and before the next kernel launch.
    /// No persistent producer may submit while the reset is in progress.
    /// Resets free/ready stacks and sideband allocator.
    /// File handles opened by previous kernels are NOT closed.
    pub fn reinit_packets(&self) {
        self.buf.reinit_packets();
    }

    /// Shut down the session. Stops listener + I/O threads, closes all files.
    ///
    /// This waits for the worker. A blocking stdin/accept syscall is not
    /// interrupted by the protocol, so the wait has no finite upper bound.
    /// Keeping the join (rather than detaching a raw-pointer worker) guarantees
    /// the mapped buffer cannot be freed while a handler still accesses it.
    pub fn shutdown(mut self) {
        self.buf.signal_shutdown();
        if let Some(handle) = self.listener_handle.take() {
            // Wait briefly for listener to drain, then join
            std::thread::sleep(std::time::Duration::from_millis(100));
            let _ = handle.join();
        }
    }
}

impl Drop for HostcallSession {
    fn drop(&mut self) {
        self.buf.signal_shutdown();
        if let Some(handle) = self.listener_handle.take() {
            std::thread::sleep(std::time::Duration::from_millis(50));
            let _ = handle.join();
        }
    }
}

// ================================================================
// Pipeline — Multi-stage kernel launch with shared hostcall session
// ================================================================

type PipelineStage =
    Box<dyn FnOnce(sys::CUdeviceptr) -> std::result::Result<(), crate::GpuHostError>>;

/// A pipeline of kernel stages that share a single [`HostcallSession`].
///
/// Each stage is a closure that launches a kernel using the session's
/// hostcall buffer device pointer. The pipeline handles synchronization
/// and packet reinitialization between stages automatically.
pub struct Pipeline {
    session: HostcallSession,
    stages: Vec<PipelineStage>,
}

impl Pipeline {
    /// Create a new pipeline with the given hostcall packet count.
    pub fn new(num_packets: u16) -> std::result::Result<Self, HostcallError> {
        let session = HostcallSession::start(num_packets)?;
        Ok(Self {
            session,
            stages: Vec::new(),
        })
    }

    /// Add a stage to the pipeline.
    ///
    /// The closure receives the hostcall buffer device pointer and should
    /// launch a kernel + synchronize. The pipeline reinits packets between stages.
    pub fn stage<F>(mut self, f: F) -> Self
    where
        F: FnOnce(sys::CUdeviceptr) -> std::result::Result<(), crate::GpuHostError> + 'static,
    {
        self.stages.push(Box::new(f));
        self
    }

    /// Execute all stages sequentially with automatic synchronization.
    ///
    /// Between stages, the hostcall packet pool is reinitialized.
    /// After all stages complete, the session is shut down.
    pub fn run(self) -> std::result::Result<(), crate::GpuHostError> {
        let hc_ptr = self.session.dev_ptr();
        let stages = self.stages;
        let session = self.session;

        for (i, stage) in stages.into_iter().enumerate() {
            if i > 0 {
                session.reinit_packets();
            }
            stage(hc_ptr)?;
        }

        std::thread::sleep(std::time::Duration::from_millis(100));
        session.shutdown();
        Ok(())
    }
}

// ================================================================
// CommandBuffer — Host→GPU command channel
// ================================================================

/// A mapped-memory command buffer for host→GPU command submission.
///
/// The host writes commands to a ring buffer; the GPU kernel polls
/// `write_idx` and processes commands sequentially.
pub struct CommandBuffer {
    host_ptr: *mut u8,
    dev_ptr: sys::CUdeviceptr,
    _size: usize,
    capacity: u32,
}

// SAFETY: CommandBuffer wraps pinned CUDA mapped memory (cuMemHostAlloc with DEVICEMAP).
// The raw pointer is valid for the lifetime of the struct and freed in Drop.
// Thread safety is ensured by the protocol: only one writer (host submit) at a time,
// and the GPU reads via the device pointer after observing write_idx updates.
unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}

/// Command to submit to the GPU via command buffer.
pub enum Command {
    /// No-op (for testing).
    Nop,
    /// Execute a computation. op_code 0 = vector add, 1 = scalar multiply, etc.
    Compute {
        /// Device pointer to input data.
        input_ptr: u64,
        /// Device pointer to output buffer.
        output_ptr: u64,
        /// Number of elements to process.
        count: u32,
        /// Operation code (application-defined).
        op_code: u32,
    },
    /// Print a message via hostcall (max 52 bytes).
    Print {
        /// Message bytes to print.
        msg: Vec<u8>,
    },
    /// Exit the command processing loop.
    Exit,
}

impl CommandBuffer {
    /// Allocate a command buffer with the given slot capacity.
    pub fn new(capacity: u32) -> Result<Self, HostcallError> {
        let size = CMD_BUF_HEADER_SIZE + (capacity as usize) * CMD_SLOT_SIZE;

        let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: cuMemHostAlloc allocates `size` bytes of pinned device-mapped memory.
        // write_bytes zero-initializes the region; no kernel is running yet.
        unsafe {
            let cu = cuda_lib();
            let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;
            let r = cu.cuMemHostAlloc(&mut host_ptr, size, flags);
            if r != sys::CUresult::CUDA_SUCCESS {
                return Err(HostcallError::CudaAlloc(r));
            }
            std::ptr::write_bytes(host_ptr as *mut u8, 0, size);
        }

        let mut dev_ptr: sys::CUdeviceptr = 0;
        // SAFETY: host_ptr was allocated with DEVICEMAP flag above.
        // On failure, we free host_ptr before returning.
        unsafe {
            let cu = cuda_lib();
            let r = cu.cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0);
            if r != sys::CUresult::CUDA_SUCCESS {
                cu.cuMemFreeHost(host_ptr);
                return Err(HostcallError::CudaGetDevPtr(r));
            }
        }

        let host_ptr = host_ptr as *mut u8;

        // SAFETY: CMD_OFF_CAPACITY is within the header region of the command buffer.
        unsafe {
            std::ptr::write_volatile(host_ptr.add(CMD_OFF_CAPACITY) as *mut u32, capacity);
        }

        Ok(Self {
            host_ptr,
            dev_ptr,
            _size: size,
            capacity,
        })
    }

    /// Get device pointer for kernel arg.
    pub fn dev_ptr(&self) -> sys::CUdeviceptr {
        self.dev_ptr
    }

    /// Submit a command to the buffer.
    ///
    /// Blocks if the buffer is full (busy-waits for GPU to drain).
    pub fn submit(&self, cmd: &Command) {
        // SAFETY: All read_volatile/write_volatile target offsets within the command
        // buffer's allocated region (CMD_OFF_WRITE_IDX, CMD_OFF_READ_IDX are in the
        // header; slot_ptr is computed from CMD_BUF_HEADER_SIZE + slot_idx * CMD_SLOT_SIZE
        // where slot_idx < capacity). Volatile access is required because the GPU
        // reads these values concurrently. The final AtomicU64 store with Release
        // ordering ensures the GPU sees the command data before the index update.
        unsafe {
            // Wait for space (backpressure)
            loop {
                let write_idx =
                    std::ptr::read_volatile(self.host_ptr.add(CMD_OFF_WRITE_IDX) as *const u64);
                let read_idx =
                    std::ptr::read_volatile(self.host_ptr.add(CMD_OFF_READ_IDX) as *const u64);
                if (write_idx - read_idx) < self.capacity as u64 {
                    break;
                }
                std::thread::yield_now();
            }

            let write_idx =
                std::ptr::read_volatile(self.host_ptr.add(CMD_OFF_WRITE_IDX) as *const u64);
            let slot_idx = (write_idx % self.capacity as u64) as usize;
            let slot_ptr = self
                .host_ptr
                .add(CMD_BUF_HEADER_SIZE + slot_idx * CMD_SLOT_SIZE);

            // Write command type and payload
            match cmd {
                Command::Nop => {
                    std::ptr::write_volatile(slot_ptr.add(CMD_SLOT_OFF_TYPE) as *mut u32, CMD_NOP);
                }
                Command::Compute {
                    input_ptr,
                    output_ptr,
                    count,
                    op_code,
                } => {
                    std::ptr::write_volatile(
                        slot_ptr.add(CMD_SLOT_OFF_TYPE) as *mut u32,
                        CMD_COMPUTE,
                    );
                    let payload = slot_ptr.add(CMD_SLOT_OFF_PAYLOAD);
                    std::ptr::write_volatile(payload as *mut u64, *input_ptr);
                    std::ptr::write_volatile(payload.add(8) as *mut u64, *output_ptr);
                    std::ptr::write_volatile(payload.add(16) as *mut u32, *count);
                    std::ptr::write_volatile(payload.add(20) as *mut u32, *op_code);
                }
                Command::Print { msg } => {
                    std::ptr::write_volatile(
                        slot_ptr.add(CMD_SLOT_OFF_TYPE) as *mut u32,
                        CMD_PRINT,
                    );
                    let payload = slot_ptr.add(CMD_SLOT_OFF_PAYLOAD);
                    let len = msg.len().min(CMD_MAX_PAYLOAD - 4) as u32;
                    std::ptr::write_volatile(payload as *mut u32, len);
                    for i in 0..len as usize {
                        std::ptr::write_volatile(payload.add(4 + i), msg[i]);
                    }
                }
                Command::Exit => {
                    std::ptr::write_volatile(slot_ptr.add(CMD_SLOT_OFF_TYPE) as *mut u32, CMD_EXIT);
                }
            }

            // Increment write_idx with Release semantics
            let write_idx_ptr = &*(self.host_ptr.add(CMD_OFF_WRITE_IDX) as *const AtomicU64);
            write_idx_ptr.store(write_idx + 1, Ordering::Release);
        }
    }

    /// Reset indices to 0 (between kernel launches).
    pub fn reset(&self) {
        // SAFETY: CMD_OFF_WRITE_IDX and CMD_OFF_READ_IDX are within the header.
        // Must only be called when no kernel is accessing the buffer (after sync).
        unsafe {
            std::ptr::write_volatile(self.host_ptr.add(CMD_OFF_WRITE_IDX) as *mut u64, 0);
            std::ptr::write_volatile(self.host_ptr.add(CMD_OFF_READ_IDX) as *mut u64, 0);
        }
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        // SAFETY: host_ptr was allocated by cuMemHostAlloc in new() and has not
        // been freed yet (Drop is called exactly once).
        unsafe {
            let cu = cuda_lib();
            cu.cuMemFreeHost(self.host_ptr as *mut std::ffi::c_void);
        }
    }
}

// ================================================================
// FlightRecorder — Post-mortem trace event ring buffer
// ================================================================

/// A mapped-memory ring buffer that stores the last N trace events.
///
/// Unlike hostcall-based tracing, the flight recorder writes directly to
/// mapped memory with no round-trip. After a kernel crash, call [`dump()`]
/// to print the last N events for post-mortem analysis.
///
/// [`dump()`]: FlightRecorder::dump
pub struct FlightRecorder {
    host_ptr: *mut u8,
    dev_ptr: sys::CUdeviceptr,
    _size: usize,
    capacity: u32,
}

// SAFETY: FlightRecorder wraps pinned CUDA mapped memory (cuMemHostAlloc with DEVICEMAP).
// The raw pointer is valid for the lifetime of the struct and freed in Drop.
// The GPU writes events atomically via write_idx; the host only reads after
// kernel completion (cuCtxSynchronize), so there is no data race.
unsafe impl Send for FlightRecorder {}
unsafe impl Sync for FlightRecorder {}

impl FlightRecorder {
    /// Allocate a flight recorder with the given event slot capacity.
    pub fn new(capacity: u32) -> Result<Self, HostcallError> {
        let size = FR_HEADER_SIZE + (capacity as usize) * FR_SLOT_SIZE;

        let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: Same CUDA alloc pattern as CommandBuffer::new — allocates pinned
        // device-mapped memory of `size` bytes and zero-initializes it.
        unsafe {
            let cu = cuda_lib();
            let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;
            let r = cu.cuMemHostAlloc(&mut host_ptr, size, flags);
            if r != sys::CUresult::CUDA_SUCCESS {
                return Err(HostcallError::CudaAlloc(r));
            }
            std::ptr::write_bytes(host_ptr as *mut u8, 0, size);
        }

        let mut dev_ptr: sys::CUdeviceptr = 0;
        // SAFETY: host_ptr was allocated with DEVICEMAP flag. On failure, we free it.
        unsafe {
            let cu = cuda_lib();
            let r = cu.cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0);
            if r != sys::CUresult::CUDA_SUCCESS {
                cu.cuMemFreeHost(host_ptr);
                return Err(HostcallError::CudaGetDevPtr(r));
            }
        }

        let host_ptr = host_ptr as *mut u8;

        // SAFETY: FR_OFF_CAPACITY is within the flight recorder header.
        unsafe {
            std::ptr::write_volatile(host_ptr.add(FR_OFF_CAPACITY) as *mut u32, capacity);
        }

        Ok(Self {
            host_ptr,
            dev_ptr,
            _size: size,
            capacity,
        })
    }

    /// Get device pointer for kernel arg.
    pub fn dev_ptr(&self) -> sys::CUdeviceptr {
        self.dev_ptr
    }

    /// Check if the kernel set the crashed flag.
    pub fn crashed(&self) -> bool {
        // SAFETY: FR_OFF_FLAGS is within the flight recorder header, 4-byte aligned.
        let flags =
            unsafe { std::ptr::read_volatile(self.host_ptr.add(FR_OFF_FLAGS) as *const u32) };
        (flags & FR_FLAG_CRASHED) != 0
    }

    /// Get the number of events written (may exceed capacity for wrap-around).
    pub fn write_count(&self) -> u64 {
        // SAFETY: FR_OFF_WRITE_IDX is within the header, 8-byte aligned.
        unsafe { std::ptr::read_volatile(self.host_ptr.add(FR_OFF_WRITE_IDX) as *const u64) }
    }

    /// Dump all recorded events to stderr.
    ///
    /// Events are printed in chronological order. If the buffer has wrapped
    /// around, only the last `capacity` events are shown.
    pub fn dump(&self) {
        let write_idx = self.write_count();
        if write_idx == 0 {
            eprintln!("=== Flight Recorder: no events ===");
            return;
        }

        let start = write_idx.saturating_sub(self.capacity as u64);

        let crashed = self.crashed();
        eprintln!(
            "=== Flight Recorder Dump ({} events{}) ===",
            write_idx - start,
            if crashed { ", CRASHED" } else { "" }
        );

        for i in start..write_idx {
            let slot_idx = (i % self.capacity as u64) as usize;
            // SAFETY: slot_idx < capacity, so FR_HEADER_SIZE + slot_idx * FR_SLOT_SIZE
            // is within the allocated flight recorder buffer. All slot offset reads
            // (FR_SLOT_OFF_META, FR_SLOT_OFF_TIMESTAMP, FR_SLOT_OFF_MSG) are within
            // FR_SLOT_SIZE bytes of the slot start.
            let slot = unsafe { self.host_ptr.add(FR_HEADER_SIZE + slot_idx * FR_SLOT_SIZE) };

            let meta = unsafe { std::ptr::read_volatile(slot.add(FR_SLOT_OFF_META) as *const u64) };
            let timestamp =
                unsafe { std::ptr::read_volatile(slot.add(FR_SLOT_OFF_TIMESTAMP) as *const u64) };

            let (tid, bid, level, msg_len, lane) = decode_trace_metadata(meta);

            let msg_len = (msg_len as usize).min(FR_MAX_MSG_LEN);
            let mut msg_buf = vec![0u8; msg_len];
            for j in 0..msg_len {
                msg_buf[j] = unsafe { std::ptr::read_volatile(slot.add(FR_SLOT_OFF_MSG + j)) };
            }

            let level_str = match level {
                TRACE_LEVEL_DEBUG => "DEBUG",
                TRACE_LEVEL_INFO => "INFO",
                TRACE_LEVEL_WARN => "WARN",
                TRACE_LEVEL_ERROR => "ERROR",
                _ => "???",
            };

            let msg = String::from_utf8_lossy(&msg_buf);
            eprintln!(
                "  [{ts}] T{tid}.B{bid}.L{lane} {lvl}: {msg}",
                ts = timestamp,
                lvl = level_str,
            );
        }

        eprintln!("=== End Flight Recorder ===");
    }

    /// Reset the flight recorder (between kernel launches).
    pub fn reset(&self) {
        // SAFETY: FR_OFF_WRITE_IDX and FR_OFF_FLAGS are within the header.
        // Must only be called when no kernel is accessing the buffer (after sync).
        unsafe {
            std::ptr::write_volatile(self.host_ptr.add(FR_OFF_WRITE_IDX) as *mut u64, 0);
            std::ptr::write_volatile(self.host_ptr.add(FR_OFF_FLAGS) as *mut u32, 0);
        }
    }
}

impl Drop for FlightRecorder {
    fn drop(&mut self) {
        // SAFETY: host_ptr was allocated by cuMemHostAlloc in new() and has not
        // been freed yet (Drop is called exactly once).
        unsafe {
            let cu = cuda_lib();
            cu.cuMemFreeHost(self.host_ptr as *mut std::ffi::c_void);
        }
    }
}

impl Drop for HostcallBuffer {
    fn drop(&mut self) {
        // SAFETY: Both host_ptr and sideband_host_ptr were allocated by
        // cuMemHostAlloc in alloc_internal() and have not been freed yet.
        unsafe {
            let cu = cuda_lib();
            cu.cuMemFreeHost(self.host_ptr as *mut std::ffi::c_void);
            if !self.sideband_host_ptr.is_null() {
                cu.cuMemFreeHost(self.sideband_host_ptr as *mut std::ffi::c_void);
            }
        }
    }
}

#[cfg(test)]
mod priority_tests {
    use super::*;
    use std::mem::ManuallyDrop;

    struct ListenerGuard {
        buffer: Arc<HostcallBuffer>,
        handle: Option<std::thread::JoinHandle<()>>,
    }

    impl ListenerGuard {
        fn start(buffer: Arc<HostcallBuffer>) -> Self {
            let listener_buffer = Arc::clone(&buffer);
            let handle = std::thread::spawn(move || listener_buffer.listen(|_| {}));
            Self {
                buffer,
                handle: Some(handle),
            }
        }

        fn finish(mut self) -> std::thread::Result<()> {
            self.buffer.signal_shutdown();
            self.handle
                .take()
                .expect("listener handle is present")
                .join()
        }
    }

    impl Drop for ListenerGuard {
        fn drop(&mut self) {
            self.buffer.signal_shutdown();
            if let Some(handle) = self.handle.take() {
                match handle.join() {
                    Ok(()) => {}
                    Err(_) => eprintln!("priority hostcall listener panicked during cleanup"),
                }
            }
        }
    }

    fn request(task_id: u64, priority: Priority) -> IoRequest {
        IoRequest {
            pkt_idx: task_id as u16,
            service: SERVICE_NOP,
            metadata: HostcallMetadata::new(task_id, priority),
            generation: None,
            pool: PacketPool::General { shard: 0 },
        }
    }

    fn fake_buffer(
        num_packets: u16,
        high_reserved: u32,
    ) -> (Vec<u64>, ManuallyDrop<HostcallBuffer>) {
        let size = buffer_size(num_packets);
        let mut storage = vec![0u64; size.div_ceil(core::mem::size_of::<u64>())];
        let buffer = ManuallyDrop::new(HostcallBuffer {
            host_ptr: storage.as_mut_ptr() as *mut u8,
            dev_ptr: 0,
            size,
            num_packets,
            num_shards: 0,
            pkts_per_shard: 0,
            high_reserved_per_shard: high_reserved,
            sideband_host_ptr: core::ptr::null_mut(),
            sideband_dev_ptr: 0,
            sideband_size: 0,
            lifecycle: Arc::new(HostcallLifecycle::default()),
            processed_completed: AtomicU64::new(0),
            processed_errors: AtomicU64::new(0),
            processed_cancelled: AtomicU64::new(0),
            stale_rejected: AtomicU64::new(0),
            priority_echo_events: Mutex::new(Vec::new()),
            priority_echo_sequence: AtomicU64::new(0),
            priority_echo_hook: AtomicU32::new(PriorityEchoTestHook::Disabled as u32),
            priority_echo_delay_micros: AtomicU64::new(0),
        });
        buffer.init();
        (storage, buffer)
    }

    unsafe fn seed_versioned_request(
        buffer: &HostcallBuffer,
        index: u16,
        generation: u64,
        state: u32,
        metadata: HostcallMetadata,
        service: u32,
    ) -> IoRequest {
        let pkt = buffer.packet_ptr(index);
        std::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, service);
        std::ptr::write_volatile(
            pkt.add(PKT_OFF_PRIORITY),
            metadata.effective_priority.as_raw(),
        );
        std::ptr::write_volatile(
            pkt.add(PKT_OFF_METADATA_FLAGS) as *mut u16,
            request_generation_metadata(generation),
        );
        std::ptr::write_volatile(pkt.add(PKT_OFF_TASK_ID) as *mut u64, metadata.task_id);
        std::ptr::write_volatile(pkt.add(PKT_OFF_METADATA_VERSION), PACKET_METADATA_VERSION);
        (&*(pkt.add(PKT_OFF_CONTROL) as *const AtomicU32))
            .store(make_control(generation, state), Ordering::Release);
        buffer.snapshot_request(index, Some(generation))
    }

    fn remove_test_packet_from_pool(buffer: &HostcallBuffer, pool: PacketPool) {
        buffer
            .stack_for_pool(pool)
            .store(make_tagged(1, NULL_INDEX), Ordering::Release);
    }

    #[test]
    fn same_priority_requests_are_stable() {
        let mut state = PriorityIoState::default();
        for task_id in 1..=5 {
            state.push(request(task_id, Priority::High));
        }
        let actual: Vec<_> = (0..5)
            .map(|_| state.pop_weighted().unwrap().metadata.task_id)
            .collect();
        assert_eq!(actual, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn weighted_cycle_is_eight_four_one() {
        let mut state = PriorityIoState::default();
        for task_id in 100..110 {
            state.push(request(task_id, Priority::High));
        }
        for task_id in 200..210 {
            state.push(request(task_id, Priority::Normal));
        }
        for task_id in 300..302 {
            state.push(request(task_id, Priority::Low));
        }

        let actual: Vec<_> = (0..13)
            .map(|_| state.pop_weighted().unwrap().metadata.task_id)
            .collect();
        assert_eq!(
            actual,
            vec![100, 101, 102, 103, 104, 105, 106, 107, 200, 201, 202, 203, 300]
        );
    }

    #[test]
    fn sustained_high_load_cannot_starve_normal_or_low() {
        let mut state = PriorityIoState::default();
        for task_id in 0..64 {
            state.push(request(task_id, Priority::High));
            state.push(request(1_000 + task_id, Priority::Normal));
        }
        state.push(request(2_000, Priority::Low));
        state.push(request(2_001, Priority::Low));

        let first_cycle: Vec<_> = (0..13)
            .map(|_| state.pop_weighted().unwrap().metadata)
            .collect();
        assert_eq!(
            first_cycle
                .iter()
                .filter(|m| m.effective_priority == Priority::High)
                .count(),
            8
        );
        assert_eq!(
            first_cycle
                .iter()
                .filter(|m| m.effective_priority == Priority::Normal)
                .count(),
            4
        );
        assert_eq!(first_cycle.last().unwrap().task_id, 2_000);

        let second_cycle: Vec<_> = (0..13)
            .map(|_| state.pop_weighted().unwrap().metadata)
            .collect();
        assert_eq!(second_cycle.last().unwrap().task_id, 2_001);
    }

    #[test]
    fn queue_close_drains_existing_requests_then_exits() {
        let queue = PriorityIoQueue::new(Arc::new(HostcallLifecycle::default()));
        assert!(queue.push(request(7, Priority::Normal)));
        queue.close();
        assert_eq!(queue.pop().unwrap().metadata.task_id, 7);
        assert!(queue.pop().is_none());
        assert!(!queue.push(request(8, Priority::High)));
    }

    #[test]
    fn legacy_and_invalid_packet_metadata_default_safely() {
        let mut words = [0u64; PACKET_HEADER_SIZE / core::mem::size_of::<u64>()];
        let pkt = words.as_mut_ptr() as *mut u8;
        // SAFETY: words is 8-byte aligned and exactly PACKET_HEADER_SIZE bytes.
        unsafe {
            std::ptr::write_volatile(pkt.add(PKT_OFF_METADATA_VERSION), 0);
            assert_eq!(read_packet_metadata(pkt), HostcallMetadata::default());

            std::ptr::write_volatile(pkt.add(PKT_OFF_METADATA_VERSION), PACKET_METADATA_VERSION);
            std::ptr::write_volatile(pkt.add(PKT_OFF_PRIORITY), 255);
            std::ptr::write_volatile(pkt.add(PKT_OFF_TASK_ID) as *mut u64, 42);
            assert_eq!(
                read_packet_metadata(pkt),
                HostcallMetadata::new(42, Priority::Normal)
            );
        }
    }

    #[test]
    fn legacy_default_keeps_all_general_credits() {
        assert_eq!(COMPAT_DEFAULT_HIGH_RESERVED_PER_SHARD, 0);
    }

    #[test]
    fn explicit_reserve_builds_a_disjoint_high_credit() {
        let size = buffer_size(4);
        let mut storage = vec![0u64; size.div_ceil(core::mem::size_of::<u64>())];
        let buffer = ManuallyDrop::new(HostcallBuffer {
            host_ptr: storage.as_mut_ptr() as *mut u8,
            dev_ptr: 0,
            size,
            num_packets: 4,
            num_shards: 0,
            pkts_per_shard: 0,
            high_reserved_per_shard: 1,
            sideband_host_ptr: core::ptr::null_mut(),
            sideband_dev_ptr: 0,
            sideband_size: 0,
            lifecycle: Arc::new(HostcallLifecycle::default()),
            processed_completed: AtomicU64::new(0),
            processed_errors: AtomicU64::new(0),
            processed_cancelled: AtomicU64::new(0),
            stale_rejected: AtomicU64::new(0),
            priority_echo_events: Mutex::new(Vec::new()),
            priority_echo_sequence: AtomicU64::new(0),
            priority_echo_hook: AtomicU32::new(PriorityEchoTestHook::Disabled as u32),
            priority_echo_delay_micros: AtomicU64::new(0),
        });

        // SAFETY: storage is 8-byte aligned and large enough for the complete
        // four-packet layout. ManuallyDrop prevents CUDA deallocation.
        unsafe {
            buffer.rebuild_packet_stacks();
            assert_eq!(buffer.high_reserved_packets(), 1);
            let general =
                std::ptr::read_volatile(buffer.host_ptr.add(BUF_OFF_FREE_STACK) as *const u64);
            let high =
                std::ptr::read_volatile(buffer.host_ptr.add(BUF_OFF_HIGH_FREE_STACK) as *const u64);
            assert_eq!(tagged_index(general), 0);
            assert_eq!(tagged_index(high), 3);

            let general_tail = buffer.host_ptr.add(packet_offset(2));
            let high_packet = buffer.host_ptr.add(packet_offset(3));
            assert_eq!(
                tagged_index(std::ptr::read_volatile(
                    general_tail.add(PKT_OFF_NEXT) as *const u64
                )),
                NULL_INDEX
            );
            assert_eq!(
                tagged_index(std::ptr::read_volatile(
                    high_packet.add(PKT_OFF_NEXT) as *const u64
                )),
                NULL_INDEX
            );
        }
    }

    #[test]
    fn cancel_before_dequeue_returns_general_packet_to_its_pool() {
        let (_storage, buffer) = fake_buffer(1, 0);
        remove_test_packet_from_pool(&buffer, PacketPool::General { shard: 0 });
        let generation = 41;
        let request = unsafe {
            seed_versioned_request(
                &buffer,
                0,
                generation,
                CONTROL_HOST_OWNED,
                HostcallMetadata::new(7, Priority::Normal),
                SERVICE_READ,
            )
        };
        let pkt = buffer.packet_ptr(0);
        let control = unsafe { &*(pkt.add(PKT_OFF_CONTROL) as *const AtomicU32) };
        assert!(control
            .compare_exchange(
                make_control(generation, CONTROL_HOST_OWNED),
                make_control(generation, CONTROL_CANCELLED),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok());

        assert!(!unsafe { buffer.begin_request_processing(request) });
        assert_eq!(
            tagged_index(buffer.stack_for_pool(request.pool).load(Ordering::Acquire)),
            0
        );
        assert_eq!(control_flags(control.load(Ordering::Acquire)), 0);
        assert_eq!(buffer.processing_metrics().cancelled, 1);
    }

    #[test]
    fn cancel_inflight_beats_completion_and_host_releases_once() {
        let (_storage, buffer) = fake_buffer(1, 0);
        remove_test_packet_from_pool(&buffer, PacketPool::General { shard: 0 });
        let generation = 77;
        let request = unsafe {
            seed_versioned_request(
                &buffer,
                0,
                generation,
                CONTROL_HOST_OWNED,
                HostcallMetadata::new(9, Priority::Normal),
                SERVICE_READ,
            )
        };
        assert!(unsafe { buffer.begin_request_processing(request) });
        let pkt = buffer.packet_ptr(0);
        let control = unsafe { &*(pkt.add(PKT_OFF_CONTROL) as *const AtomicU32) };
        assert!(control
            .compare_exchange(
                make_control(generation, CONTROL_HOST_OWNED),
                make_control(generation, CONTROL_CANCELLED),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok());
        unsafe { buffer.complete_request(request, false) };

        assert_eq!(buffer.processing_metrics().cancelled, 1);
        assert_eq!(buffer.processing_metrics().completed, 0);
        assert_eq!(
            tagged_index(buffer.stack_for_pool(request.pool).load(Ordering::Acquire)),
            0
        );
    }

    #[test]
    fn completion_and_cancel_cas_have_exactly_one_winner() {
        let (_storage, buffer) = fake_buffer(2, 0);
        let first = unsafe {
            seed_versioned_request(
                &buffer,
                0,
                101,
                CONTROL_HOST_OWNED,
                HostcallMetadata::new(1, Priority::Normal),
                SERVICE_NOP,
            )
        };
        unsafe { buffer.complete_request(first, false) };
        let first_control =
            unsafe { &*(buffer.packet_ptr(0).add(PKT_OFF_CONTROL) as *const AtomicU32) };
        assert!(first_control
            .compare_exchange(
                make_control(101, CONTROL_HOST_OWNED),
                make_control(101, CONTROL_CANCELLED),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err());
        assert_eq!(
            control_flags(first_control.load(Ordering::Acquire)),
            CONTROL_READY
        );

        remove_test_packet_from_pool(&buffer, PacketPool::General { shard: 0 });
        let second = unsafe {
            seed_versioned_request(
                &buffer,
                1,
                102,
                CONTROL_HOST_OWNED,
                HostcallMetadata::new(2, Priority::Normal),
                SERVICE_NOP,
            )
        };
        let second_control =
            unsafe { &*(buffer.packet_ptr(1).add(PKT_OFF_CONTROL) as *const AtomicU32) };
        assert!(second_control
            .compare_exchange(
                make_control(102, CONTROL_HOST_OWNED),
                make_control(102, CONTROL_CANCELLED),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok());
        unsafe { buffer.complete_request(second, false) };
        assert_eq!(control_flags(second_control.load(Ordering::Acquire)), 0);
        assert_eq!(buffer.processing_metrics().completed, 1);
        assert_eq!(buffer.processing_metrics().cancelled, 1);
    }

    #[test]
    fn stale_io_request_cannot_write_after_packet_reuse() {
        let (_storage, buffer) = fake_buffer(1, 0);
        let stale = unsafe {
            seed_versioned_request(
                &buffer,
                0,
                500,
                CONTROL_HOST_OWNED,
                HostcallMetadata::new(10, Priority::Low),
                SERVICE_READ,
            )
        };
        let pkt = buffer.packet_ptr(0);
        unsafe {
            std::ptr::write_volatile(pkt.add(PKT_OFF_PAYLOAD) as *mut u64, 0xfeed_beef);
            let _fresh = seed_versioned_request(
                &buffer,
                0,
                501,
                CONTROL_HOST_OWNED,
                HostcallMetadata::new(11, Priority::High),
                SERVICE_WRITE,
            );
        }

        assert!(!unsafe { buffer.begin_request_processing(stale) });
        assert_eq!(
            unsafe { std::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64) },
            0xfeed_beef
        );
        assert_eq!(buffer.processing_metrics().stale_rejected, 1);
    }

    #[test]
    fn cancelled_high_packet_returns_to_shared_high_pool() {
        let (_storage, buffer) = fake_buffer(2, 1);
        remove_test_packet_from_pool(&buffer, PacketPool::HighShared);
        let request = unsafe {
            seed_versioned_request(
                &buffer,
                1,
                900,
                CONTROL_CANCELLED,
                HostcallMetadata::new(12, Priority::High),
                SERVICE_NOP,
            )
        };
        buffer.push_free_from_host(request);
        assert_eq!(
            tagged_index(
                buffer
                    .stack_for_pool(PacketPool::HighShared)
                    .load(Ordering::Acquire)
            ),
            1
        );
        assert_eq!(
            tagged_index(
                buffer
                    .stack_for_pool(PacketPool::General { shard: 0 })
                    .load(Ordering::Acquire)
            ),
            0
        );
    }

    #[test]
    fn try_reinit_rejects_slow_io_and_succeeds_after_quiescence() {
        let (_storage, buffer) = fake_buffer(1, 0);
        buffer.lifecycle.begin_inflight();
        let busy = buffer.try_reinit_packets().unwrap_err();
        assert_eq!(busy.inflight, 1);
        buffer.lifecycle.finish_inflight();
        buffer.try_reinit_packets().unwrap();
    }

    #[test]
    fn packet_metadata_requires_a_stable_owned_control_snapshot() {
        let (_storage, buffer) = fake_buffer(1, 0);
        let metadata = HostcallMetadata::new(0x1234, Priority::High);
        unsafe {
            seed_versioned_request(&buffer, 0, 33, CONTROL_HOST_OWNED, metadata, SERVICE_NOP);
        }
        assert_eq!(buffer.packet_metadata(0), Some(metadata));
        let control = unsafe { &*(buffer.packet_ptr(0).add(PKT_OFF_CONTROL) as *const AtomicU32) };
        control.store(make_control(33, 0), Ordering::Release);
        assert_eq!(buffer.packet_metadata(0), None);
    }

    #[test]
    fn release_idle_publication_precedes_metadata_clear_and_reuse() {
        let (_storage, buffer) = fake_buffer(1, 0);
        remove_test_packet_from_pool(&buffer, PacketPool::General { shard: 0 });
        let metadata = HostcallMetadata::new(0x55, Priority::High);
        let request =
            unsafe { seed_versioned_request(&buffer, 0, 34, CONTROL_READY, metadata, SERVICE_NOP) };
        assert_eq!(buffer.packet_metadata(0), Some(metadata));
        buffer.push_free_from_host(request);
        assert_eq!(buffer.packet_metadata(0), None);
        let control = unsafe {
            (&*(buffer.packet_ptr(0).add(PKT_OFF_CONTROL) as *const AtomicU32))
                .load(Ordering::Acquire)
        };
        assert_eq!(control_flags(control), 0);
        assert_eq!(
            tagged_index(
                buffer
                    .stack_for_pool(PacketPool::General { shard: 0 })
                    .load(Ordering::Acquire)
            ),
            0
        );
    }

    #[test]
    fn host_metrics_make_error_completion_observable() {
        let (_storage, buffer) = fake_buffer(1, 0);
        let request = unsafe {
            seed_versioned_request(
                &buffer,
                0,
                63,
                CONTROL_HOST_OWNED,
                HostcallMetadata::new(3, Priority::Normal),
                SERVICE_TIME,
            )
        };
        unsafe { buffer.complete_request(request, true) };
        assert_eq!(
            buffer.processing_metrics(),
            HostcallProcessingMetrics {
                completed: 1,
                errors: 1,
                cancelled: 0,
                stale_rejected: 0,
            }
        );
    }

    #[test]
    fn priority_echo_records_live_identity_and_shared_reserve_provenance() {
        let (_storage, buffer) = fake_buffer(5, 1);
        let initial = buffer.quiescent_pool_audit();
        assert!(initial.ready_empty);
        assert_eq!(initial.idle_packets, 5);
        assert_eq!(initial.general_mask, 0b0_1111);
        assert_eq!(initial.shared_high_mask, 0b1_0000);
        assert_eq!(initial.duplicate_entries, 0);
        assert_eq!(initial.missing_packets, 0);

        let task_id = (composed_priority_schema::NAMESPACE_VALUE << 32)
            | composed_priority_schema::EXPECTED_HIGH_LOCAL_ID_VALUE;
        remove_test_packet_from_pool(&buffer, PacketPool::HighShared);
        let request = unsafe {
            seed_versioned_request(
                &buffer,
                4,
                41,
                CONTROL_HOST_OWNED,
                HostcallMetadata::new(task_id, Priority::High),
                SERVICE_PRIORITY_ECHO,
            )
        };
        let nonce = 0xA11C_E55E_u64;
        unsafe {
            let pkt = buffer.packet_ptr(4);
            std::ptr::write_volatile(pkt.add(PKT_OFF_PAYLOAD) as *mut u64, nonce);
            assert!(!buffer.handle_priority_echo(pkt, request));
            buffer.complete_request(request, false);
            let payload = pkt.add(PKT_OFF_PAYLOAD) as *const u64;
            assert_eq!(
                std::ptr::read_volatile(payload.add(priority_echo::NONCE)),
                nonce
            );
            assert_eq!(
                std::ptr::read_volatile(payload.add(priority_echo::PACKET_INDEX)),
                4
            );
            assert_eq!(
                std::ptr::read_volatile(payload.add(priority_echo::SHARED_HIGH_RESERVED)),
                1
            );
        }

        let events = buffer.priority_echo_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].task_id, task_id);
        assert_eq!(
            events[0].namespace as u64,
            composed_priority_schema::NAMESPACE_VALUE
        );
        assert_eq!(
            events[0].local_id as u64,
            composed_priority_schema::EXPECTED_HIGH_LOCAL_ID_VALUE
        );
        assert_eq!(events[0].priority, Priority::High);
        assert_eq!(events[0].packet_index, 4);
        assert!(events[0].shared_high_reserved);
        assert_eq!(events[0].process_count, 1);
        assert_eq!(events[0].error_category, 0);
    }

    #[test]
    fn priority_echo_corruption_hook_is_explicit_and_oracle_visible() {
        let (_storage, buffer) = fake_buffer(5, 1);
        buffer.set_priority_echo_test_hook(PriorityEchoTestHook::CorruptEchoNonce, 0);
        remove_test_packet_from_pool(&buffer, PacketPool::HighShared);
        let task_id = (composed_priority_schema::NAMESPACE_VALUE << 32)
            | composed_priority_schema::EXPECTED_HIGH_LOCAL_ID_VALUE;
        let request = unsafe {
            seed_versioned_request(
                &buffer,
                4,
                42,
                CONTROL_HOST_OWNED,
                HostcallMetadata::new(task_id, Priority::High),
                SERVICE_PRIORITY_ECHO,
            )
        };
        unsafe {
            let pkt = buffer.packet_ptr(4);
            std::ptr::write_volatile(pkt.add(PKT_OFF_PAYLOAD) as *mut u64, 7);
            assert!(!buffer.handle_priority_echo(pkt, request));
        }
        let events = buffer.priority_echo_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].request_nonce, 7);
        assert_eq!(events[0].echo_nonce, 6);
    }

    #[test]
    fn priority_echo_rejects_legacy_unversioned_identity() {
        let (_storage, buffer) = fake_buffer(5, 1);
        let task_id = (composed_priority_schema::NAMESPACE_VALUE << 32)
            | composed_priority_schema::EXPECTED_HIGH_LOCAL_ID_VALUE;
        let request = IoRequest {
            pkt_idx: 4,
            service: SERVICE_PRIORITY_ECHO,
            metadata: HostcallMetadata::new(task_id, Priority::High),
            generation: None,
            pool: PacketPool::HighShared,
        };
        unsafe {
            let pkt = buffer.packet_ptr(4);
            std::ptr::write_volatile(pkt.add(PKT_OFF_PAYLOAD) as *mut u64, 99);
            assert!(buffer.handle_priority_echo(pkt, request));
        }
        let events = buffer.priority_echo_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].error_category, ERR_INVALID_INPUT);
    }

    #[test]
    fn priority_echo_forced_error_is_recorded_independently() {
        let (_storage, buffer) = fake_buffer(5, 1);
        buffer.set_priority_echo_test_hook(PriorityEchoTestHook::ForceError, 0);
        assert_eq!(buffer.priority_echo_hook.load(Ordering::Acquire), 1);
        let task_id = (composed_priority_schema::NAMESPACE_VALUE << 32)
            | composed_priority_schema::EXPECTED_HIGH_LOCAL_ID_VALUE;
        let request = unsafe {
            seed_versioned_request(
                &buffer,
                4,
                43,
                CONTROL_HOST_OWNED,
                HostcallMetadata::new(task_id, Priority::High),
                SERVICE_PRIORITY_ECHO,
            )
        };
        unsafe {
            let pkt = buffer.packet_ptr(4);
            std::ptr::write_volatile(pkt.add(PKT_OFF_PAYLOAD) as *mut u64, 123);
            assert!(buffer.handle_priority_echo(pkt, request));
        }
        let events = buffer.priority_echo_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].request_nonce, 123);
        assert_eq!(events[0].error_category, ERR_IO_ERROR);
    }

    #[test]
    fn cancelled_priority_echo_is_recorded_and_delay_does_not_leak_to_reuse() {
        let (_storage, buffer) = fake_buffer(5, 1);
        buffer.set_priority_echo_test_hook(PriorityEchoTestHook::DelayNext, 0);
        remove_test_packet_from_pool(&buffer, PacketPool::HighShared);
        let task_id = (composed_priority_schema::NAMESPACE_VALUE << 32)
            | composed_priority_schema::EXPECTED_HIGH_LOCAL_ID_VALUE;
        unsafe {
            seed_versioned_request(
                &buffer,
                4,
                51,
                CONTROL_CANCELLED,
                HostcallMetadata::new(task_id, Priority::High),
                SERVICE_PRIORITY_ECHO,
            );
            let pkt = buffer.packet_ptr(4);
            std::ptr::write_volatile(pkt.add(PKT_OFF_PAYLOAD) as *mut u64, 0xCA11_CE11);
            std::ptr::write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, null_tagged());
            buffer
                .ready_stack()
                .store(make_tagged(1, 4), Ordering::Release);
            buffer.doorbell().store(1, Ordering::Release);
            buffer.shutdown().store(1, Ordering::Release);
        }

        buffer.listen_unified(|_| {}, CannedStdin::new(Vec::new()));

        let events = buffer.priority_echo_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].request_nonce, 0xCA11_CE11);
        assert_eq!(events[0].echo_nonce, 0xCA11_CE11);
        assert_eq!(events[0].task_id, task_id);
        assert_eq!(events[0].generation, 51);
        assert_eq!(events[0].packet_index, 4);
        assert!(events[0].shared_high_reserved);
        assert_eq!(events[0].process_count, 1);
        assert_eq!(events[0].error_category, ERR_HOST_TIMEOUT);
        assert_eq!(
            buffer.priority_echo_hook.load(Ordering::Acquire),
            PriorityEchoTestHook::Disabled as u32
        );
        assert_eq!(buffer.processing_metrics().cancelled, 1);
        let audit = buffer.quiescent_pool_audit();
        assert!(audit.ready_empty);
        assert_eq!(audit.shared_high_mask, 0b1_0000);
        assert_eq!(audit.missing_packets, 0);
        assert_eq!(audit.non_idle_controls, 0);
    }

    #[test]
    fn cancellation_between_claim_and_dispatch_is_recorded_before_reclaim() {
        let (_storage, buffer) = fake_buffer(5, 1);
        buffer.set_priority_echo_test_hook(PriorityEchoTestHook::DelayNext, 0);
        remove_test_packet_from_pool(&buffer, PacketPool::HighShared);
        let task_id = (composed_priority_schema::NAMESPACE_VALUE << 32)
            | composed_priority_schema::EXPECTED_HIGH_LOCAL_ID_VALUE;
        let request = unsafe {
            let request = seed_versioned_request(
                &buffer,
                4,
                52,
                CONTROL_HOST_OWNED,
                HostcallMetadata::new(task_id, Priority::High),
                SERVICE_PRIORITY_ECHO,
            );
            let pkt = buffer.packet_ptr(4);
            std::ptr::write_volatile(pkt.add(PKT_OFF_PAYLOAD) as *mut u64, 0xCA11_CE12);
            (&*(pkt.add(PKT_OFF_CONTROL) as *const AtomicU32))
                .store(make_control(52, CONTROL_CANCELLED), Ordering::Release);
            request
        };

        assert!(!unsafe { buffer.begin_request_processing(request) });

        let events = buffer.priority_echo_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].request_nonce, 0xCA11_CE12);
        assert_eq!(events[0].generation, 52);
        assert_eq!(events[0].error_category, ERR_HOST_TIMEOUT);
        assert_eq!(
            buffer.priority_echo_hook.load(Ordering::Acquire),
            PriorityEchoTestHook::Disabled as u32
        );
        assert_eq!(buffer.processing_metrics().cancelled, 1);
        let audit = buffer.quiescent_pool_audit();
        assert_eq!(audit.shared_high_mask, 0b1_0000);
        assert_eq!(audit.missing_packets, 0);
        assert_eq!(audit.non_idle_controls, 0);
    }

    #[test]
    fn shutdown_final_drain_reclaims_cancelled_ready_credit() {
        let (_storage, buffer) = fake_buffer(1, 0);
        remove_test_packet_from_pool(&buffer, PacketPool::General { shard: 0 });
        unsafe {
            seed_versioned_request(
                &buffer,
                0,
                73,
                CONTROL_CANCELLED,
                HostcallMetadata::new(9, Priority::Normal),
                SERVICE_NOP,
            );
            let pkt = buffer.packet_ptr(0);
            std::ptr::write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, null_tagged());
            buffer
                .ready_stack()
                .store(make_tagged(1, 0), Ordering::Release);
            buffer.doorbell().store(1, Ordering::Release);
            buffer.shutdown().store(1, Ordering::Release);
        }

        buffer.listen_unified(|_| {}, CannedStdin::new(Vec::new()));

        assert!(buffer.ready_stacks_empty());
        assert_eq!(
            tagged_index(
                buffer
                    .stack_for_pool(PacketPool::General { shard: 0 })
                    .load(Ordering::Acquire)
            ),
            0
        );
        assert_eq!(buffer.processing_metrics().cancelled, 1);
        let audit = buffer.quiescent_pool_audit();
        assert_eq!(audit.missing_packets, 0);
        assert_eq!(audit.non_idle_controls, 0);
    }

    /// Correct-route compatibility gate for the legacy metadata path.
    ///
    /// Kept ignored because loading the 3 MiB I/O PTX can take minutes on a
    /// cold CUDA JIT cache. Run it explicitly under an outer timeout.
    #[test]
    #[ignore = "requires a CUDA GPU and a potentially slow PTX JIT"]
    fn legacy_trace_e2e_uses_io_ptx_and_all_general_credits() {
        use crate::mapped_mem::{alloc_mapped_u32, free_mapped_mem};
        use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
        use cudarc::nvrtc::Ptx;

        eprintln!("phase=ptx_load bytes={}", crate::ptx::KERNEL_IO.len());

        let dev = CudaDevice::new(0).expect("initialize CUDA device 0");
        dev.load_ptx(
            Ptx::from_src(crate::ptx::KERNEL_IO),
            "priority_hostcall_io",
            &["trace_multithread_test"],
        )
        .expect("load I/O PTX and resolve trace_multithread_test");
        let function = dev
            .get_func("priority_hostcall_io", "trace_multithread_test")
            .expect("resolved trace_multithread_test function");
        eprintln!("phase=module_ready");

        let buffer = Arc::new(HostcallBuffer::new(64).expect("allocate legacy hostcall buffer"));
        assert_eq!(buffer.high_reserved_per_shard(), 0);
        assert_eq!(buffer.num_packets(), 64);
        // SAFETY: the fixed header belongs to this live mapped allocation.
        let high_head = unsafe {
            std::ptr::read_volatile(buffer.host_ptr().add(BUF_OFF_HIGH_FREE_STACK) as *const u64)
        };
        assert_eq!(tagged_index(high_head), NULL_INDEX);

        // SAFETY: CUDA is initialized and the returned mapping is freed below.
        let (count_host_ptr, count_dev_ptr) =
            unsafe { alloc_mapped_u32(&dev).expect("allocate mapped completion counter") };
        unsafe { std::ptr::write_volatile(count_host_ptr, 0) };

        // Module loading and symbol resolution intentionally happen before the
        // listener starts. ListenerGuard guarantees shutdown + join on unwind.
        let listener = ListenerGuard::start(Arc::clone(&buffer));
        eprintln!("phase=launch");
        let run_result = unsafe {
            function.launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (buffer.dev_ptr(), count_dev_ptr),
            )
        }
        .and_then(|_| dev.synchronize());
        run_result.expect("launch and synchronize trace_multithread_test");
        eprintln!("phase=kernel_complete");

        listener.finish().expect("join hostcall listener");
        let completed = unsafe { std::ptr::read_volatile(count_host_ptr) };
        // SAFETY: count_host_ptr came from alloc_mapped_u32 and is no longer in use.
        unsafe { free_mapped_mem(count_host_ptr).expect("free mapped completion counter") };
        assert_eq!(completed, 32);
        eprintln!("phase=pass completed={completed}/32");
    }
}
