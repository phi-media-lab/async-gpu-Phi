use core::cell::UnsafeCell;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use gpu_atomics::{lane_id, shfl_sync_idx_u32, syncwarp};

#[cfg(not(target_arch = "nvptx64"))]
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

pub use gpu_protocol::Priority;

/// Maximum number of tasks the executor can hold.
pub const MAX_TASKS: usize = 256;

/// Maximum size of a spawned future in bytes.
pub const TASK_FUTURE_MAX_SIZE: usize = 512;

/// Maximum alignment supported by inline future storage.
pub const TASK_FUTURE_MAX_ALIGN: usize = 16;

/// Slots that can only be consumed by [`Priority::High`] tasks.
///
/// This admission reserve prevents a flood of lower-priority work from making
/// a later latency-sensitive spawn fail before it reaches the scheduler.
pub const HIGH_PRIORITY_RESERVED_SLOTS: usize = 16;

/// Additional slots reserved for [`Priority::Normal`] or [`Priority::High`]
/// tasks. Low-priority tasks cannot consume this capacity.
pub const NORMAL_PRIORITY_RESERVED_SLOTS: usize = 16;

/// Sentinel value broadcast when no queue entry is available.
pub const EMPTY_SENTINEL: u32 = u32::MAX;

const EMPTY_TASK_KEY: u64 = u64::MAX;
const TASK_SLOT_BITS: u32 = 8;
const TASK_SLOT_MASK: u64 = (1 << TASK_SLOT_BITS) - 1;
const TASK_GENERATION_MAX: u64 = (1 << (64 - TASK_SLOT_BITS)) - 2;
const INITIAL_GENERATION: u64 = 1;

// Task slot states
const SLOT_FREE: u32 = 0;
const SLOT_QUEUED: u32 = 1;
const SLOT_RUNNING: u32 = 2;
/// Task returned Pending — waiting for a waker to re-enqueue it.
pub const SLOT_PARKED: u32 = 3;
/// A wake raced with a poll while the task was RUNNING.
const SLOT_NOTIFIED: u32 = 4;
/// Re-enqueue was temporarily unable to reserve a queue cell.
const SLOT_ENQUEUE_RETRY: u32 = 5;
/// Terminal owner has claimed the slot and may drop/recycle its future.
const SLOT_COMPLETING: u32 = 6;

const SLOT_STATE_MASK: u64 = 0x7F;
const SLOT_PRIORITY_LOCK: u64 = 0x80;

const CONTEXT_UNUSED: u32 = 0;
const CONTEXT_LIVE: u32 = 1;
const CONTEXT_RETIRING: u32 = 2;
const CONTEXT_RETIRED: u32 = 3;
const CONTEXT_RECLAIMED: u32 = 4;

const SHUTDOWN_BIT: u32 = 1 << 31;
const ACTIVE_COUNT_MASK: u32 = !SHUTDOWN_BIT;

/// Weighted, work-conserving service cycle.
///
/// A continuously runnable Low task is considered at least once per 13
/// successful dispatches, while High receives 8/13 of saturated service.
const SERVICE_PATTERN: [Priority; 13] = [
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

/// Sentinel: no waker registered.
pub const NO_WAKER: u32 = EMPTY_SENTINEL;

/// Error type for executor operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutorError {
    /// Work queue is full.
    QueueFull,
    /// No free task slots available.
    NoFreeSlots,
    /// Future exceeds `TASK_FUTURE_MAX_SIZE` bytes.
    FutureTooLarge,
    /// Future alignment exceeds [`TASK_FUTURE_MAX_ALIGN`].
    FutureTooAligned,
    /// Capacity exists, but it is reserved for a higher priority class.
    ReservedCapacity,
    /// A typed completion cell was already used by another spawn.
    CompletionUnavailable,
    /// The executor has started shutting down and rejects new work.
    Shutdown,
    /// The configured executor namespace exhausted its 32-bit trace sequence.
    TraceIdExhausted,
    /// The process/module-wide executor incarnation sequence is exhausted.
    IncarnationExhausted,
}

/// Why a generational task authority no longer names a live task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskKeyError {
    /// The encoded slot is outside the executor arena.
    InvalidSlot,
    /// The slot belongs to another generation.
    Stale,
    /// The named generation has entered terminal cleanup.
    Terminated,
    /// This authority belongs to another executor allocation.
    WrongExecutor,
    /// This authority belongs to a previous initialization epoch.
    WrongIncarnation,
    /// Typed tasks keep their type-level priority fixed for sound wait edges.
    PriorityFixed,
}

/// Scheduling effect of a successful effective-priority update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriorityUpdate {
    /// No existing queue membership needs relocation; the next enqueue observes
    /// the new value.
    Applied,
    /// The task already has one immutable queue entry at its previous class.
    /// It keeps that entry exactly once and uses the new class after its next
    /// poll boundary.
    DeferredUntilRequeue,
}

/// Options controlling cooperative task scheduling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskOptions {
    /// Base priority assigned when the task is admitted.
    pub priority: Priority,
}

impl TaskOptions {
    /// Create options with the given base priority.
    #[inline(always)]
    pub const fn new(priority: Priority) -> Self {
        Self { priority }
    }

    /// Set the base priority.
    #[inline(always)]
    pub const fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }
}

impl Default for TaskOptions {
    fn default() -> Self {
        Self::new(Priority::Normal)
    }
}

/// Queue-local slot/generation identity.
///
/// This type is deliberately private: a local key is not authority without
/// the executor allocation and initialization epoch that issued it.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct LocalTaskKey(u64);

impl LocalTaskKey {
    #[inline(always)]
    const fn from_parts(slot: u32, generation: u64) -> Self {
        Self((generation << TASK_SLOT_BITS) | slot as u64)
    }

    #[inline(always)]
    const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    #[inline(always)]
    const fn slot(self) -> u32 {
        (self.0 & TASK_SLOT_MASK) as u32
    }

    #[inline(always)]
    const fn generation(self) -> u64 {
        self.0 >> TASK_SLOT_BITS
    }

    #[inline(always)]
    const fn raw(self) -> u64 {
        self.0
    }
}

/// Generational authority for one admitted task.
///
/// Authority is bound to an executor address, a non-repeating initialization
/// incarnation, and a queue-local slot/generation. Only the issuing executor
/// can construct this value. The address is an identity token, not permission
/// to dereference memory.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TaskKey {
    owner: u64,
    incarnation: u64,
    local: LocalTaskKey,
}

impl TaskKey {
    #[inline(always)]
    const fn bind(owner: u64, incarnation: u64, local: LocalTaskKey) -> Self {
        Self {
            owner,
            incarnation,
            local,
        }
    }

    /// Arena slot used for diagnostics. It is not sufficient authority alone.
    #[inline(always)]
    pub const fn slot(self) -> u32 {
        self.local.slot()
    }

    /// Admission generation used for diagnostics.
    #[inline(always)]
    pub const fn generation(self) -> u64 {
        self.local.generation()
    }

    #[inline(always)]
    const fn local(self) -> LocalTaskKey {
        self.local
    }
}

/// Compatibility name retained for existing spawn call sites.
///
/// Unlike the former tuple struct, this alias is generational and cannot be
/// constructed from a slot-only `u32`.
pub type TaskId = TaskKey;

/// Statistics returned when a warp exits the executor loop.
#[derive(Clone, Copy, Debug)]
pub struct ExecutorStats {
    /// Number of tasks this warp executed to completion.
    pub tasks_executed: u32,
    /// Total number of poll calls this warp made.
    pub polls_total: u32,
}

// ================================================================
// NVPTX system-scope acquire/release atomics
// ================================================================

/// Load a repeatedly-polled u32 without LLVM's `readonly` LICM permission.
#[inline(always)]
pub(crate) unsafe fn atomic_load_acquire_u32(ptr: *const u32) -> u32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let value: u32;
        core::arch::asm!(
            "ld.acquire.sys.global.u32 {value}, [{ptr}];",
            value = out(reg32) value,
            ptr = in(reg64) ptr,
            options(nostack),
        );
        value
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        (&*(ptr as *const AtomicU32)).load(Ordering::Acquire)
    }
}

#[inline(always)]
pub(crate) unsafe fn atomic_load_acquire_u64(ptr: *const u64) -> u64 {
    #[cfg(target_arch = "nvptx64")]
    {
        let value: u64;
        core::arch::asm!(
            "ld.acquire.sys.global.u64 {value}, [{ptr}];",
            value = out(reg64) value,
            ptr = in(reg64) ptr,
            options(nostack),
        );
        value
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        (&*(ptr as *const AtomicU64)).load(Ordering::Acquire)
    }
}

#[inline(always)]
pub(crate) unsafe fn atomic_store_release_u32(ptr: *mut u32, value: u32) {
    #[cfg(target_arch = "nvptx64")]
    core::arch::asm!(
        "st.release.sys.global.u32 [{ptr}], {value};",
        ptr = in(reg64) ptr,
        value = in(reg32) value,
        options(nostack),
    );
    #[cfg(not(target_arch = "nvptx64"))]
    (&*(ptr as *const AtomicU32)).store(value, Ordering::Release);
}

#[inline(always)]
pub(crate) unsafe fn atomic_store_release_u64(ptr: *mut u64, value: u64) {
    #[cfg(target_arch = "nvptx64")]
    core::arch::asm!(
        "st.release.sys.global.u64 [{ptr}], {value};",
        ptr = in(reg64) ptr,
        value = in(reg64) value,
        options(nostack),
    );
    #[cfg(not(target_arch = "nvptx64"))]
    (&*(ptr as *const AtomicU64)).store(value, Ordering::Release);
}

#[inline(always)]
pub(crate) unsafe fn atomic_cas_acq_rel_u32(ptr: *mut u32, expected: u32, desired: u32) -> u32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let old: u32;
        core::arch::asm!(
            "atom.acq_rel.sys.global.cas.b32 {old}, [{ptr}], {expected}, {desired};",
            old = out(reg32) old,
            ptr = in(reg64) ptr,
            expected = in(reg32) expected,
            desired = in(reg32) desired,
            options(nostack),
        );
        old
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        match (&*(ptr as *const AtomicU32)).compare_exchange(
            expected,
            desired,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(old) | Err(old) => old,
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn atomic_cas_acq_rel_u64(ptr: *mut u64, expected: u64, desired: u64) -> u64 {
    #[cfg(target_arch = "nvptx64")]
    {
        let old: u64;
        core::arch::asm!(
            "atom.acq_rel.sys.global.cas.b64 {old}, [{ptr}], {expected}, {desired};",
            old = out(reg64) old,
            ptr = in(reg64) ptr,
            expected = in(reg64) expected,
            desired = in(reg64) desired,
            options(nostack),
        );
        old
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        match (&*(ptr as *const AtomicU64)).compare_exchange(
            expected,
            desired,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(old) | Err(old) => old,
        }
    }
}

/// Process/module-wide allocator for executor initialization epochs.
///
/// It never wraps: exhaustion is sticky so no old public authority can revive.
#[repr(transparent)]
struct IncarnationSequence(UnsafeCell<u64>);

unsafe impl Sync for IncarnationSequence {}

static EXECUTOR_INCARNATION_SEQUENCE: IncarnationSequence = IncarnationSequence(UnsafeCell::new(0));

#[inline(always)]
unsafe fn allocate_nonwrapping_incarnation(sequence: *mut u64) -> Option<u64> {
    loop {
        let old = atomic_load_acquire_u64(sequence as *const u64);
        if old == u64::MAX {
            return None;
        }
        let next = old + 1;
        if atomic_cas_acq_rel_u64(sequence, old, next) == old {
            return Some(next);
        }
    }
}

#[inline(always)]
unsafe fn allocate_executor_incarnation() -> Result<u64, ExecutorError> {
    allocate_nonwrapping_incarnation(EXECUTOR_INCARNATION_SEQUENCE.0.get())
        .ok_or(ExecutorError::IncarnationExhausted)
}
#[inline(always)]
pub(crate) unsafe fn atomic_fetch_add_u32(ptr: *mut u32, value: u32) -> u32 {
    loop {
        let old = atomic_load_acquire_u32(ptr as *const u32);
        if atomic_cas_acq_rel_u32(ptr, old, old.wrapping_add(value)) == old {
            return old;
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn atomic_fetch_sub_u32(ptr: *mut u32, value: u32) -> u32 {
    loop {
        let old = atomic_load_acquire_u32(ptr as *const u32);
        if atomic_cas_acq_rel_u32(ptr, old, old.wrapping_sub(value)) == old {
            return old;
        }
    }
}
/// Retain one reference without ever wrapping back to zero.
///
/// Returns false after the counter reaches the permanent saturated sentinel.
#[inline(always)]
unsafe fn saturating_retain_u32(ptr: *mut u32) -> bool {
    loop {
        let old = atomic_load_acquire_u32(ptr as *const u32);
        if old == u32::MAX {
            return false;
        }
        let next = old + 1;
        if atomic_cas_acq_rel_u32(ptr, old, next) == old {
            return next != u32::MAX;
        }
    }
}

/// Release one finite reference, returning its value before release.
///
/// A saturated counter is permanently pinned and is never decremented.
#[inline(always)]
unsafe fn saturating_release_u32(ptr: *mut u32) -> Option<u32> {
    loop {
        let old = atomic_load_acquire_u32(ptr as *const u32);
        if old == u32::MAX {
            return None;
        }
        debug_assert!(old != 0, "reference counter underflow");
        if old == 0 {
            return None;
        }
        if atomic_cas_acq_rel_u32(ptr, old, old - 1) == old {
            return Some(old);
        }
    }
}

#[inline(always)]
unsafe fn permanently_pin_u32(ptr: *mut u32) {
    loop {
        let old = atomic_load_acquire_u32(ptr as *const u32);
        if old == u32::MAX {
            return;
        }
        if atomic_cas_acq_rel_u32(ptr, old, u32::MAX) == old {
            return;
        }
    }
}

/// Signed modular distance used by the bounded MPMC cell protocol.
#[inline(always)]
const fn sequence_difference(sequence: u64, expected: u64) -> i64 {
    sequence.wrapping_sub(expected) as i64
}

#[inline(always)]
const fn identity_value(generation: u64, state: u32) -> u64 {
    (generation << TASK_SLOT_BITS) | state as u64
}

#[inline(always)]
const fn identity_generation(identity: u64) -> u64 {
    identity >> TASK_SLOT_BITS
}

#[inline(always)]
const fn identity_state(identity: u64) -> u32 {
    (identity & SLOT_STATE_MASK) as u32
}

#[inline(always)]
const fn identity_is_locked(identity: u64) -> bool {
    identity & SLOT_PRIORITY_LOCK != 0
}

#[inline(always)]
const fn next_generation(generation: u64) -> Option<u64> {
    if generation >= TASK_GENERATION_MAX {
        None
    } else {
        Some(generation + 1)
    }
}

// ================================================================
// WorkQueue — bounded lock-free MPMC FIFO
// ================================================================

/// Bounded lock-free MPMC FIFO queue of generational task keys.
#[repr(C)]
struct WorkQueue {
    /// Monotonic consumer position.
    head: UnsafeCell<u64>,
    /// Monotonic producer position.
    tail: UnsafeCell<u64>,
    /// Circular buffer of immutable task keys.
    buffer: [UnsafeCell<u64>; MAX_TASKS],
    /// Per-cell publication/recycle sequence for bounded MPMC correctness.
    sequence: [UnsafeCell<u64>; MAX_TASKS],
}

#[allow(clippy::new_without_default)]
impl WorkQueue {
    /// Create a new empty work queue.
    pub const fn new() -> Self {
        #[allow(clippy::declare_interior_mutable_const)]
        const EMPTY: UnsafeCell<u64> = UnsafeCell::new(EMPTY_TASK_KEY);
        #[allow(clippy::declare_interior_mutable_const)]
        const ZERO: UnsafeCell<u64> = UnsafeCell::new(0);
        let mut queue = Self {
            head: UnsafeCell::new(0),
            tail: UnsafeCell::new(0),
            buffer: [EMPTY; MAX_TASKS],
            sequence: [ZERO; MAX_TASKS],
        };
        let mut index = 0;
        while index < MAX_TASKS {
            queue.sequence[index] = UnsafeCell::new(index as u64);
            index += 1;
        }
        queue
    }

    /// Initialize queue positions and per-cell sequence numbers.
    ///
    /// # Safety
    /// Must be called exactly once, before concurrent queue access begins.
    pub unsafe fn init(&self) {
        core::ptr::write_volatile(self.head.get(), 0);
        core::ptr::write_volatile(self.tail.get(), 0);
        let mut index = 0;
        while index < MAX_TASKS {
            core::ptr::write_volatile(self.buffer[index].get(), EMPTY_TASK_KEY);
            core::ptr::write_volatile(self.sequence[index].get(), index as u64);
            index += 1;
        }
    }

    /// Enqueue an immutable task key. Returns Err if the queue is full.
    ///
    /// # Safety
    /// Must be called from a single lane (typically lane 0).
    #[inline(always)]
    unsafe fn enqueue(&self, key: LocalTaskKey) -> Result<(), ExecutorError> {
        let tail_ptr = self.tail.get();

        loop {
            let position = atomic_load_acquire_u64(tail_ptr as *const _);
            let cell = (position as usize) & (MAX_TASKS - 1);
            let sequence = atomic_load_acquire_u64(self.sequence[cell].get() as *const u64);
            let difference = sequence_difference(sequence, position);

            if difference < 0 {
                return Err(ExecutorError::QueueFull);
            }
            if difference > 0 {
                continue;
            }

            if atomic_cas_acq_rel_u64(tail_ptr, position, position.wrapping_add(1)) == position {
                // Publish data before advancing the cell sequence. Consumers
                // never observe a reserved-but-unwritten cell as ready.
                core::ptr::write_volatile(self.buffer[cell].get(), key.raw());
                atomic_store_release_u64(self.sequence[cell].get(), position.wrapping_add(1));
                return Ok(());
            }
        }
    }

    /// Dequeue a task key. Returns `None` if the queue is empty.
    ///
    /// # Safety
    /// Must be called from a single lane (typically lane 0).
    #[inline(always)]
    unsafe fn dequeue(&self) -> Option<LocalTaskKey> {
        let head_ptr = self.head.get();

        loop {
            let position = atomic_load_acquire_u64(head_ptr as *const _);
            let cell = (position as usize) & (MAX_TASKS - 1);
            let sequence = atomic_load_acquire_u64(self.sequence[cell].get() as *const u64);
            let difference = sequence_difference(sequence, position.wrapping_add(1));

            if difference < 0 {
                return None;
            }
            if difference > 0 {
                continue;
            }

            if atomic_cas_acq_rel_u64(head_ptr, position, position.wrapping_add(1)) == position {
                let raw = core::ptr::read_volatile(self.buffer[cell].get());
                core::ptr::write_volatile(self.buffer[cell].get(), EMPTY_TASK_KEY);
                // Only now may a producer reuse the cell. Publishing `head`
                // before clearing is unsafe without this per-cell handshake.
                atomic_store_release_u64(
                    self.sequence[cell].get(),
                    position.wrapping_add(MAX_TASKS as u64),
                );
                return Some(LocalTaskKey::from_raw(raw));
            }
        }
    }
}

// ================================================================
// TaskSlot — type-erased future storage
// ================================================================

/// Type alias for a type-erased poll function pointer.
pub type PollFn = unsafe fn(*mut u8, &mut Context<'_>) -> Poll<()>;

/// Type alias for a type-erased future destructor.
pub type DropFn = unsafe fn(*mut u8);

/// Aligned inline bytes used by every [`TaskSlot`].
#[repr(C, align(16))]
pub struct TaskStorage(pub [u8; TASK_FUTURE_MAX_SIZE]);

/// Stable per-slot record referenced by RawWakers and typed task contexts.
///
/// The record is not overwritten until its previous generation is retired and
/// every reference has been released.
#[repr(C, align(16))]
pub(crate) struct WakerData {
    executor: UnsafeCell<u64>,
    incarnation: UnsafeCell<u64>,
    key: UnsafeCell<u64>,
    trace_id: UnsafeCell<u64>,
    refs: UnsafeCell<u32>,
    phase: UnsafeCell<u32>,
}

impl WakerData {
    const fn new() -> Self {
        Self {
            executor: UnsafeCell::new(0),
            incarnation: UnsafeCell::new(0),
            key: UnsafeCell::new(EMPTY_TASK_KEY),
            trace_id: UnsafeCell::new(0),
            refs: UnsafeCell::new(0),
            phase: UnsafeCell::new(CONTEXT_UNUSED),
        }
    }
}

/// A fixed-size slot for storing a type-erased future.
///
/// The future bytes are stored inline. The `poll_fn` pointer provides
/// type-erased access to `Future::poll()`.
#[repr(C)]
pub struct TaskSlot {
    /// Atomic generation, priority-lock bit, and task state.
    identity_state: UnsafeCell<u64>,
    /// Priority chosen at spawn. Immutable until the slot is recycled.
    pub base_priority: UnsafeCell<u32>,
    /// Type-erased poll function.
    pub poll_fn: UnsafeCell<Option<PollFn>>,
    /// Size of the stored future in bytes (for debugging).
    pub future_size: UnsafeCell<u32>,
    /// Priority used for the next enqueue. May differ through inheritance.
    pub effective_priority: UnsafeCell<u32>,
    /// Nonzero when type-level dependency rules require a fixed priority.
    priority_fixed: UnsafeCell<u32>,
    /// Type-erased destructor used on completion, cancellation, and rollback.
    pub drop_fn: UnsafeCell<Option<DropFn>>,
    /// Stable ownership record used by all wakers and typed contexts.
    waker_data: WakerData,
    /// Inline storage for the future. The preceding pointer keeps it 8-byte aligned.
    pub future_bytes: UnsafeCell<TaskStorage>,
}

#[allow(clippy::new_without_default)]
impl TaskSlot {
    /// Create a new free task slot.
    pub const fn new() -> Self {
        Self {
            identity_state: UnsafeCell::new(identity_value(INITIAL_GENERATION, SLOT_FREE)),
            base_priority: UnsafeCell::new(Priority::Normal as u32),
            poll_fn: UnsafeCell::new(None),
            future_size: UnsafeCell::new(0),
            effective_priority: UnsafeCell::new(Priority::Normal as u32),
            priority_fixed: UnsafeCell::new(0),
            drop_fn: UnsafeCell::new(None),
            waker_data: WakerData::new(),
            future_bytes: UnsafeCell::new(TaskStorage([0u8; TASK_FUTURE_MAX_SIZE])),
        }
    }
}

/// Type-erased poll trampoline. Casts the raw bytes back to `F` and polls.
///
/// # Safety
/// `ptr` must point to a valid `F` that was previously copied into the slot.
#[inline(always)]
unsafe fn erased_poll<F: Future<Output = ()>>(ptr: *mut u8, cx: &mut Context<'_>) -> Poll<()> {
    let future = &mut *(ptr as *mut F);
    Pin::new_unchecked(future).poll(cx)
}

/// Type-erased destructor for a future stored in a [`TaskSlot`].
///
/// # Safety
/// `ptr` must point to a live `F` that has not previously been dropped.
#[inline(always)]
unsafe fn erased_drop<F>(ptr: *mut u8) {
    core::ptr::drop_in_place(ptr as *mut F);
}

// ================================================================
// Free slot bitmap (no linked-node ABA)
// ================================================================

const FREE_BITMAP_WORDS: usize = MAX_TASKS / 64;
const FREE_NULL: u16 = u16::MAX;

/// Four independent availability words. A set bit denotes a free slot.
#[repr(C)]
pub struct FreeSlotBitmap {
    words: [UnsafeCell<u64>; FREE_BITMAP_WORDS],
}

impl FreeSlotBitmap {
    pub const fn empty() -> Self {
        #[allow(clippy::declare_interior_mutable_const)]
        const ZERO: UnsafeCell<u64> = UnsafeCell::new(0);
        Self {
            words: [ZERO; FREE_BITMAP_WORDS],
        }
    }

    /// Mark slots `[0, count)` available before concurrent use.
    ///
    /// # Safety
    /// Called by exactly one initializer before publication.
    unsafe fn init(&self, count: usize) {
        let mut word = 0;
        while word < FREE_BITMAP_WORDS {
            let first = word * 64;
            let available = count.saturating_sub(first).min(64);
            let bits = if available == 64 {
                u64::MAX
            } else if available == 0 {
                0
            } else {
                (1u64 << available) - 1
            };
            core::ptr::write_volatile(self.words[word].get(), bits);
            word += 1;
        }
    }

    /// Atomically claim any set bit.
    #[inline(always)]
    unsafe fn pop(&self) -> u16 {
        let mut word = 0;
        while word < FREE_BITMAP_WORDS {
            let ptr = self.words[word].get();
            let old = atomic_load_acquire_u64(ptr as *const u64);
            if old != 0 {
                let bit = old.trailing_zeros();
                let mask = 1u64 << bit;
                if atomic_cas_acq_rel_u64(ptr, old, old & !mask) == old {
                    return (word * 64 + bit as usize) as u16;
                }
                continue;
            }
            word += 1;
        }
        FREE_NULL
    }

    /// Publish one unavailable bit as free. Returns false on double-free.
    #[inline(always)]
    unsafe fn push(&self, slot_idx: u16) -> bool {
        let word = slot_idx as usize / 64;
        let mask = 1u64 << (slot_idx as usize % 64);
        let ptr = self.words[word].get();
        loop {
            let old = atomic_load_acquire_u64(ptr as *const u64);
            if old & mask != 0 {
                return false;
            }
            if atomic_cas_acq_rel_u64(ptr, old, old | mask) == old {
                return true;
            }
        }
    }
}

// ================================================================
// GpuExecutor — the main executor struct
// ================================================================

/// GPU-side async task executor with work-stealing.
///
/// Allocated in global or device-visible mapped memory. Initialization must run
/// on the device so the owner token uses the same device pointer value later
/// seen by spawn, wake, and run. Host-side initialization of mapped memory is
/// not supported when host and device virtual addresses can differ.
///
/// After the first spawn or run, the allocation must not move until shutdown,
/// all executor warps have exited, and the host/device completion is synchronized.
#[repr(C, align(256))]
pub struct GpuExecutor {
    /// Normal-priority MPMC queue.
    work_queue: WorkQueue,
    /// High-priority MPMC queue.
    high_priority_queue: WorkQueue,
    /// Low-priority MPMC queue.
    low_priority_queue: WorkQueue,
    /// ABA-free slot availability bitmap.
    free_slots: FreeSlotBitmap,
    /// Non-repeating process/module-local initialization epoch.
    incarnation: UnsafeCell<u64>,
    /// Shutdown bit plus atomically maintained active-task count.
    ///
    /// Keeping both values in one CAS word linearizes spawn admission against
    /// shutdown: either a spawn reserves a count first, or shutdown rejects it.
    tasks_active: UnsafeCell<u32>,
    /// Total tasks spawned (diagnostic counter).
    tasks_spawned: UnsafeCell<u32>,
    /// Total tasks completed (diagnostic counter).
    tasks_completed: UnsafeCell<u32>,
    /// Caller-provided launch/executor namespace for host correlation.
    trace_namespace: UnsafeCell<u32>,
    /// Monotonic, non-wrapping sequence within `trace_namespace`.
    trace_sequence: UnsafeCell<u32>,
    /// Waker and typed-context references that can dereference this executor.
    outstanding_context_refs: UnsafeCell<u32>,
    /// Terminal slots whose stable context record is not yet reclaimable.
    pending_reclaims: UnsafeCell<u32>,
    /// Task slot arena.
    pub slots: [TaskSlot; MAX_TASKS],
}

// SAFETY: GpuExecutor is designed for concurrent access across warps/blocks.
// All mutable state is protected by atomic CAS operations.
unsafe impl Send for GpuExecutor {}
unsafe impl Sync for GpuExecutor {}
unsafe impl Send for WorkQueue {}
unsafe impl Sync for WorkQueue {}
unsafe impl Send for TaskSlot {}
unsafe impl Sync for TaskSlot {}
unsafe impl Send for WakerData {}
unsafe impl Sync for WakerData {}
unsafe impl Send for FreeSlotBitmap {}
unsafe impl Sync for FreeSlotBitmap {}

// ================================================================
// GPU Waker — state-aware re-enqueue into the effective-priority queue
// ================================================================

/// Owned reference to one stable per-slot context record.
///
/// The record pins the executor allocation and prevents slot reuse until this
/// reference is dropped. It is crate-private because only executor admission
/// may issue task-context capabilities.
pub(crate) struct TaskContextRef {
    data: *const WakerData,
}

unsafe impl Send for TaskContextRef {}
unsafe impl Sync for TaskContextRef {}

impl TaskContextRef {
    #[inline(always)]
    pub(crate) unsafe fn acquire(data: *const WakerData) -> Self {
        retain_context(data);
        Self { data }
    }

    #[inline(always)]
    pub(crate) fn key(&self) -> TaskKey {
        unsafe {
            TaskKey::bind(
                atomic_load_acquire_u64((*self.data).executor.get()),
                atomic_load_acquire_u64((*self.data).incarnation.get()),
                LocalTaskKey::from_raw(atomic_load_acquire_u64((*self.data).key.get())),
            )
        }
    }

    #[inline(always)]
    pub(crate) fn trace_id(&self) -> u64 {
        unsafe { atomic_load_acquire_u64((*self.data).trace_id.get()) }
    }

    #[inline(always)]
    pub(crate) fn effective_priority(&self) -> Result<Priority, TaskKeyError> {
        unsafe {
            let executor = atomic_load_acquire_u64((*self.data).executor.get());
            if executor == 0 {
                return Err(TaskKeyError::Terminated);
            }
            (&*(executor as *const GpuExecutor)).effective_priority(self.key())
        }
    }
}

impl Drop for TaskContextRef {
    #[inline(always)]
    fn drop(&mut self) {
        unsafe { release_context(self.data) }
    }
}

#[inline(always)]
unsafe fn retain_context(data: *const WakerData) {
    let finite = saturating_retain_u32((*data).refs.get());
    let executor_ptr = atomic_load_acquire_u64((*data).executor.get());
    if executor_ptr != 0 {
        let global = (*(executor_ptr as *const GpuExecutor))
            .outstanding_context_refs
            .get();
        if finite {
            let _ = saturating_retain_u32(global);
        } else {
            // A saturated per-slot count can never reach zero. Pin the global
            // teardown gate as well so it can never incorrectly report safe.
            permanently_pin_u32(global);
        }
    }
}

#[inline(always)]
unsafe fn release_context(data: *const WakerData) {
    let executor_ptr = atomic_load_acquire_u64((*data).executor.get());
    let key = TaskKey::bind(
        executor_ptr,
        atomic_load_acquire_u64((*data).incarnation.get()),
        LocalTaskKey::from_raw(atomic_load_acquire_u64((*data).key.get())),
    );
    let Some(old) = saturating_release_u32((*data).refs.get()) else {
        // Saturation is a deliberate permanent leak: RawWaker::clone cannot
        // report allocation/refcount failure, so wrapping would permit UAF.
        return;
    };
    if executor_ptr != 0 {
        let executor = &*(executor_ptr as *const GpuExecutor);
        let _ = saturating_release_u32(executor.outstanding_context_refs.get());
        if old == 1 {
            executor.try_reclaim(key);
        }
    }
}

unsafe fn raw_waker_clone(data: *const ()) -> core::task::RawWaker {
    retain_context(data as *const WakerData);
    core::task::RawWaker::new(data, &GPU_WAKER_VTABLE)
}

unsafe fn raw_waker_wake(data: *const ()) {
    wake_task_from_data(data);
    release_context(data as *const WakerData);
}

unsafe fn raw_waker_wake_by_ref(data: *const ()) {
    wake_task_from_data(data);
}

unsafe fn raw_waker_drop(data: *const ()) {
    release_context(data as *const WakerData);
}

const GPU_WAKER_VTABLE: core::task::RawWakerVTable = core::task::RawWakerVTable::new(
    raw_waker_clone,
    raw_waker_wake,
    raw_waker_wake_by_ref,
    raw_waker_drop,
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WakeTransition {
    Ignore,
    NotifyRunning,
    EnqueueParked,
}

#[inline(always)]
const fn wake_transition(state: u32) -> WakeTransition {
    match state {
        SLOT_RUNNING => WakeTransition::NotifyRunning,
        SLOT_PARKED | SLOT_ENQUEUE_RETRY => WakeTransition::EnqueueParked,
        _ => WakeTransition::Ignore,
    }
}

/// Terminal result for the obligation to publish one runnable queue entry.
///
/// A caller may stop retrying only after the entry was published (or another
/// linearized publisher owns it), publication was converted to the scanner's
/// explicit retry state, or the matching task generation became terminal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RequeueOutcome {
    PublishedByMe,
    DelegatedToWinner,
    Retry,
    TerminalOrStale,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PendingTransition {
    Park,
    Requeue,
    Invalid,
}

#[inline(always)]
const fn pending_transition(state: u32) -> PendingTransition {
    match state {
        SLOT_RUNNING => PendingTransition::Park,
        SLOT_NOTIFIED => PendingTransition::Requeue,
        _ => PendingTransition::Invalid,
    }
}

/// Wake through a stable record. The embedded generation makes stale wakers a
/// no-op even after a slot is eventually reused.
///
/// A wake while RUNNING sets `NOTIFIED`; the polling warp observes that state
/// after `Poll::Pending` and re-enqueues exactly once. QUEUED/NOTIFIED wakes are
/// coalesced, preventing duplicate queue entries.
#[inline(always)]
unsafe fn wake_task_from_data(data: *const ()) {
    if data.is_null() {
        return;
    }
    let data = data as *const WakerData;
    if atomic_load_acquire_u32((*data).phase.get()) != CONTEXT_LIVE {
        return;
    }
    let executor_ptr = atomic_load_acquire_u64((*data).executor.get());
    if executor_ptr != 0 {
        let key = TaskKey::bind(
            executor_ptr,
            atomic_load_acquire_u64((*data).incarnation.get()),
            LocalTaskKey::from_raw(atomic_load_acquire_u64((*data).key.get())),
        );
        (&*(executor_ptr as *const GpuExecutor)).wake_task(key);
    }
}

/// Create one owned waker reference for a live task.
#[inline(always)]
unsafe fn make_task_waker(data: *const WakerData) -> core::task::Waker {
    retain_context(data);
    unsafe {
        core::task::Waker::from_raw(core::task::RawWaker::new(
            data as *const (),
            &GPU_WAKER_VTABLE,
        ))
    }
}

#[allow(clippy::new_without_default)]
impl GpuExecutor {
    /// Create a new executor with all slots free.
    ///
    /// After construction, call `init()` to set up the free slot linked list.
    pub const fn new() -> Self {
        #[allow(clippy::declare_interior_mutable_const)]
        const SLOT: TaskSlot = TaskSlot::new();
        Self {
            work_queue: WorkQueue::new(),
            high_priority_queue: WorkQueue::new(),
            low_priority_queue: WorkQueue::new(),
            incarnation: UnsafeCell::new(0),
            free_slots: FreeSlotBitmap::empty(),
            tasks_active: UnsafeCell::new(0),
            tasks_spawned: UnsafeCell::new(0),
            tasks_completed: UnsafeCell::new(0),
            trace_namespace: UnsafeCell::new(0),
            trace_sequence: UnsafeCell::new(0),
            outstanding_context_refs: UnsafeCell::new(0),
            pending_reclaims: UnsafeCell::new(0),
            slots: [SLOT; MAX_TASKS],
        }
    }
    #[inline(always)]
    fn owner_id(&self) -> u64 {
        self as *const Self as u64
    }

    #[inline(always)]
    unsafe fn current_incarnation(&self) -> u64 {
        atomic_load_acquire_u64(self.incarnation.get())
    }

    #[inline(always)]
    unsafe fn bind_local_key(&self, local: LocalTaskKey) -> TaskKey {
        TaskKey::bind(self.owner_id(), self.current_incarnation(), local)
    }

    #[inline(always)]
    unsafe fn validate_key_binding(&self, key: TaskKey) -> Result<LocalTaskKey, TaskKeyError> {
        if key.owner != self.owner_id() {
            return Err(TaskKeyError::WrongExecutor);
        }
        if key.incarnation != self.current_incarnation() {
            return Err(TaskKeyError::WrongIncarnation);
        }
        if key.slot() as usize >= MAX_TASKS {
            return Err(TaskKeyError::InvalidSlot);
        }
        Ok(key.local())
    }

    /// Initialize the executor. Must be called once before `spawn()` or `run()`.
    ///
    /// Sets up the free slot linked list.
    ///
    /// # Safety
    /// Must be called by exactly one thread (e.g., lane 0 of the first warp).
    /// It must run on the device. Before reinitialization, shutdown must be
    /// complete, all executor warps must have exited, and the launch/device
    /// must be externally synchronized.
    pub unsafe fn init(&self) {
        self.init_with_namespace(0);
    }

    /// Initialize with a caller-assigned launch/executor namespace.
    ///
    /// Trace IDs are `(namespace << 32) | sequence`. Namespace zero, used by
    /// [`GpuExecutor::init`], is unique only within this executor epoch; callers
    /// requiring cross-block or cross-launch correlation must supply a unique
    /// nonzero namespace.
    ///
    /// # Safety
    /// No task, waker, typed handle, or executor warp from a previous epoch may
    /// remain. [`GpuExecutor::can_teardown`] must be true before reinitializing.
    /// The call must execute on the device, at the same stable address used by
    /// future spawn, wake, and run operations. External warp join and device
    /// synchronization remain mandatory even after the internal gate is true.
    pub unsafe fn init_with_namespace(&self, namespace: u32) {
        let incarnation = allocate_executor_incarnation().expect("executor incarnation exhausted");
        core::ptr::write_volatile(self.incarnation.get(), incarnation);
        self.high_priority_queue.init();
        self.work_queue.init();
        self.low_priority_queue.init();
        self.free_slots.init(MAX_TASKS);
        let mut index = 0;
        while index < MAX_TASKS {
            let slot = &self.slots[index];
            core::ptr::write_volatile(
                slot.identity_state.get(),
                identity_value(INITIAL_GENERATION, SLOT_FREE),
            );
            core::ptr::write(slot.poll_fn.get(), None);
            core::ptr::write(slot.drop_fn.get(), None);
            core::ptr::write_volatile(slot.future_size.get(), 0);
            core::ptr::write_volatile(slot.base_priority.get(), Priority::Normal as u32);
            core::ptr::write_volatile(slot.effective_priority.get(), Priority::Normal as u32);
            core::ptr::write_volatile(slot.priority_fixed.get(), 0);
            core::ptr::write_volatile(slot.waker_data.executor.get(), 0);
            core::ptr::write_volatile(slot.waker_data.incarnation.get(), 0);
            core::ptr::write_volatile(slot.waker_data.key.get(), EMPTY_TASK_KEY);
            core::ptr::write_volatile(slot.waker_data.trace_id.get(), 0);
            core::ptr::write_volatile(slot.waker_data.refs.get(), 0);
            core::ptr::write_volatile(slot.waker_data.phase.get(), CONTEXT_UNUSED);
            index += 1;
        }
        core::ptr::write_volatile(self.tasks_active.get(), 0);
        core::ptr::write_volatile(self.tasks_spawned.get(), 0);
        core::ptr::write_volatile(self.tasks_completed.get(), 0);
        core::ptr::write_volatile(self.trace_namespace.get(), namespace);
        core::ptr::write_volatile(self.trace_sequence.get(), 0);
        core::ptr::write_volatile(self.outstanding_context_refs.get(), 0);
        core::ptr::write_volatile(self.pending_reclaims.get(), 0);
    }

    /// Allocate a launch-local logical task ID for tracing and hostcall
    /// correlation. Unlike [`TaskId`], this value is never used to locate a
    /// slot, so slot recycling cannot redirect a typed dependency.
    #[inline(always)]
    pub(crate) unsafe fn allocate_trace_id(&self) -> Result<u64, ExecutorError> {
        loop {
            let old = atomic_load_acquire_u32(self.trace_sequence.get());
            if old == u32::MAX {
                return Err(ExecutorError::TraceIdExhausted);
            }
            let sequence = old + 1;
            if atomic_cas_acq_rel_u32(self.trace_sequence.get(), old, sequence) == old {
                let namespace = atomic_load_acquire_u32(self.trace_namespace.get());
                return Ok(((namespace as u64) << 32) | sequence as u64);
            }
        }
    }

    #[inline(always)]
    const fn admission_limit(priority: Priority) -> u32 {
        match priority {
            Priority::High => MAX_TASKS as u32,
            Priority::Normal => (MAX_TASKS - HIGH_PRIORITY_RESERVED_SLOTS) as u32,
            Priority::Low => {
                (MAX_TASKS - HIGH_PRIORITY_RESERVED_SLOTS - NORMAL_PRIORITY_RESERVED_SLOTS) as u32
            }
        }
    }

    /// Reserve one active-task count while atomically excluding shutdown.
    #[inline(always)]
    unsafe fn reserve_active(&self, priority: Priority) -> Result<(), ExecutorError> {
        loop {
            let old = atomic_load_acquire_u32(self.tasks_active.get());
            if old & SHUTDOWN_BIT != 0 {
                return Err(ExecutorError::Shutdown);
            }
            let active = old & ACTIVE_COUNT_MASK;
            if active >= Self::admission_limit(priority) {
                return if active >= MAX_TASKS as u32 {
                    Err(ExecutorError::NoFreeSlots)
                } else {
                    Err(ExecutorError::ReservedCapacity)
                };
            }
            if atomic_cas_acq_rel_u32(self.tasks_active.get(), old, old + 1) == old {
                return Ok(());
            }
        }
    }

    #[inline(always)]
    unsafe fn release_active(&self) {
        loop {
            let old = atomic_load_acquire_u32(self.tasks_active.get());
            let active = old & ACTIVE_COUNT_MASK;
            if active == 0 {
                return;
            }
            let new = (old & SHUTDOWN_BIT) | (active - 1);
            if atomic_cas_acq_rel_u32(self.tasks_active.get(), old, new) == old {
                return;
            }
        }
    }

    #[inline(always)]
    unsafe fn enqueue_task(
        &self,
        key: LocalTaskKey,
        priority: Priority,
    ) -> Result<(), ExecutorError> {
        match priority {
            Priority::High => self.high_priority_queue.enqueue(key),
            Priority::Normal => self.work_queue.enqueue(key),
            Priority::Low => self.low_priority_queue.enqueue(key),
        }
    }

    #[inline(always)]
    unsafe fn dequeue_priority(&self, priority: Priority) -> Option<LocalTaskKey> {
        match priority {
            Priority::High => self.high_priority_queue.dequeue(),
            Priority::Normal => self.work_queue.dequeue(),
            Priority::Low => self.low_priority_queue.dequeue(),
        }
    }

    /// Dequeue according to the 8:4:1 work-conserving service cycle.
    #[inline(always)]
    unsafe fn dequeue_weighted(&self, cursor: &mut usize) -> Option<LocalTaskKey> {
        let mut tried = 0;
        while tried < SERVICE_PATTERN.len() {
            let pattern_index = *cursor;
            *cursor += 1;
            if *cursor == SERVICE_PATTERN.len() {
                *cursor = 0;
            }
            if let Some(key) = self.dequeue_priority(SERVICE_PATTERN[pattern_index]) {
                return Some(key);
            }
            tried += 1;
        }
        None
    }

    /// Remove stale duplicate entries until a QUEUED task is claimed RUNNING.
    #[inline(always)]
    unsafe fn dequeue_runnable(&self, cursor: &mut usize) -> Option<TaskKey> {
        let mut stale = 0;
        while stale < MAX_TASKS * 3 {
            let key = self.dequeue_weighted(cursor)?;
            if key.slot() as usize >= MAX_TASKS {
                stale += 1;
                continue;
            }
            let identity = &self.slots[key.slot() as usize].identity_state;
            // Dequeue consumed the generation's sole queue-membership entry.
            // A priority accessor may transiently lock the identity word, but
            // that lock does not transfer this publication obligation. Wait
            // for it rather than misclassifying the entry as stale.
            loop {
                let observed = atomic_load_acquire_u64(identity.get());
                if identity_generation(observed) != key.generation() {
                    break;
                }
                if identity_is_locked(observed) {
                    core::hint::spin_loop();
                    continue;
                }
                if identity_state(observed) != SLOT_QUEUED {
                    break;
                }
                if atomic_cas_acq_rel_u64(
                    identity.get(),
                    observed,
                    identity_value(key.generation(), SLOT_RUNNING),
                ) == observed
                {
                    return Some(self.bind_local_key(key));
                }
            }
            stale += 1;
        }
        None
    }

    /// Lock a live generation while reading or mutating its priority fields.
    #[inline(always)]
    unsafe fn lock_priority(&self, key: TaskKey) -> Result<(&TaskSlot, u64), TaskKeyError> {
        let index = self.validate_key_binding(key)?.slot() as usize;
        let slot = &self.slots[index];
        loop {
            let identity = atomic_load_acquire_u64(slot.identity_state.get());
            if identity_generation(identity) != key.generation() {
                return Err(TaskKeyError::Stale);
            }
            if identity_is_locked(identity) {
                continue;
            }
            match identity_state(identity) {
                SLOT_FREE | SLOT_COMPLETING => return Err(TaskKeyError::Terminated),
                _ => {}
            }
            if atomic_cas_acq_rel_u64(
                slot.identity_state.get(),
                identity,
                identity | SLOT_PRIORITY_LOCK,
            ) == identity
            {
                return Ok((slot, identity));
            }
        }
    }

    #[inline(always)]
    unsafe fn unlock_priority(&self, slot: &TaskSlot, identity: u64) {
        atomic_store_release_u64(slot.identity_state.get(), identity);
    }

    #[inline(always)]
    unsafe fn wake_task(&self, key: TaskKey) {
        let Ok(local) = self.validate_key_binding(key) else {
            return;
        };
        let index = local.slot() as usize;
        let slot = &self.slots[index];
        loop {
            let identity = atomic_load_acquire_u64(slot.identity_state.get());
            if identity_generation(identity) != key.generation() {
                return;
            }
            if identity_is_locked(identity) {
                continue;
            }
            let (transition, next_state) = match wake_transition(identity_state(identity)) {
                WakeTransition::Ignore => return,
                WakeTransition::NotifyRunning => (WakeTransition::NotifyRunning, SLOT_NOTIFIED),
                WakeTransition::EnqueueParked => {
                    let _ = self.requeue_slot(key, identity_state(identity));
                    return;
                }
            };
            let next = identity_value(key.generation(), next_state);
            if atomic_cas_acq_rel_u64(slot.identity_state.get(), identity, next) != identity {
                continue;
            }
            debug_assert_eq!(transition, WakeTransition::NotifyRunning);
            return;
        }
    }

    #[inline(always)]
    unsafe fn clear_slot(&self, slot: &TaskSlot) {
        let drop_fn = core::ptr::read(slot.drop_fn.get());
        if let Some(drop_fn) = drop_fn {
            let ptr = (*slot.future_bytes.get()).0.as_mut_ptr();
            drop_fn(ptr);
        }
        core::ptr::write(slot.poll_fn.get(), None);
        core::ptr::write(slot.drop_fn.get(), None);
        core::ptr::write_volatile(slot.future_size.get(), 0);
        core::ptr::write_volatile(slot.base_priority.get(), Priority::Normal as u32);
        core::ptr::write_volatile(slot.effective_priority.get(), Priority::Normal as u32);
        core::ptr::write_volatile(slot.priority_fixed.get(), 0);
    }

    /// Move an exclusively claimed task into delayed terminal reclamation.
    #[inline(always)]
    unsafe fn retire_claimed(&self, key: TaskKey, count_terminal: bool) {
        let slot = &self.slots[key.slot() as usize];
        atomic_fetch_add_u32(self.pending_reclaims.get(), 1);
        let phase =
            atomic_cas_acq_rel_u32(slot.waker_data.phase.get(), CONTEXT_LIVE, CONTEXT_RETIRING);
        debug_assert_eq!(phase, CONTEXT_LIVE);
        self.clear_slot(slot);
        atomic_store_release_u32(slot.waker_data.phase.get(), CONTEXT_RETIRED);
        self.release_active();
        if count_terminal {
            atomic_fetch_add_u32(self.tasks_completed.get(), 1);
        }
        self.try_reclaim(key);
    }

    /// Publish a terminal slot to the free bitmap only after every escaped
    /// RawWaker and typed context has released the stable record.
    #[inline(always)]
    unsafe fn try_reclaim(&self, key: TaskKey) {
        let Ok(local) = self.validate_key_binding(key) else {
            return;
        };
        let index = local.slot() as usize;
        let slot = &self.slots[index];
        if atomic_load_acquire_u64(slot.waker_data.key.get()) != key.local().raw()
            || atomic_load_acquire_u32(slot.waker_data.phase.get()) != CONTEXT_RETIRED
            || atomic_load_acquire_u32(slot.waker_data.refs.get()) != 0
        {
            return;
        }
        if atomic_cas_acq_rel_u32(
            slot.waker_data.phase.get(),
            CONTEXT_RETIRED,
            CONTEXT_RECLAIMED,
        ) != CONTEXT_RETIRED
        {
            return;
        }

        let expected = identity_value(key.generation(), SLOT_COMPLETING);
        let Some(generation) = next_generation(key.generation()) else {
            // Generation exhaustion permanently retires this slot rather than
            // reviving an old authority through wraparound.
            atomic_fetch_sub_u32(self.pending_reclaims.get(), 1);
            return;
        };
        if atomic_cas_acq_rel_u64(
            slot.identity_state.get(),
            expected,
            identity_value(generation, SLOT_FREE),
        ) != expected
        {
            return;
        }
        let returned = self.free_slots.push(index as u16);
        assert!(returned, "terminal slot was already free");
        atomic_fetch_sub_u32(self.pending_reclaims.get(), 1);
    }

    #[inline(always)]
    unsafe fn claim_running_for_completion(&self, key: TaskKey) -> bool {
        let slot = &self.slots[key.slot() as usize];
        loop {
            let identity = atomic_load_acquire_u64(slot.identity_state.get());
            if identity_generation(identity) != key.generation() {
                return false;
            }
            if identity_is_locked(identity) {
                continue;
            }
            match identity_state(identity) {
                SLOT_RUNNING | SLOT_NOTIFIED => {
                    if atomic_cas_acq_rel_u64(
                        slot.identity_state.get(),
                        identity,
                        identity_value(key.generation(), SLOT_COMPLETING),
                    ) == identity
                    {
                        return true;
                    }
                }
                _ => return false,
            }
        }
    }

    #[inline(always)]
    unsafe fn mark_enqueue_retry(&self, key: TaskKey) -> RequeueOutcome {
        let slot = &self.slots[key.slot() as usize];
        loop {
            let identity = atomic_load_acquire_u64(slot.identity_state.get());
            if identity_generation(identity) != key.generation() {
                return RequeueOutcome::TerminalOrStale;
            }
            if identity_is_locked(identity) {
                core::hint::spin_loop();
                continue;
            }
            match identity_state(identity) {
                SLOT_QUEUED => {
                    if atomic_cas_acq_rel_u64(
                        slot.identity_state.get(),
                        identity,
                        identity_value(key.generation(), SLOT_ENQUEUE_RETRY),
                    ) == identity
                    {
                        return RequeueOutcome::Retry;
                    }
                }
                SLOT_ENQUEUE_RETRY => return RequeueOutcome::Retry,
                _ => return RequeueOutcome::TerminalOrStale,
            }
        }
    }

    #[inline(always)]
    unsafe fn requeue_slot(&self, key: TaskKey, from_state: u32) -> RequeueOutcome {
        let slot = &self.slots[key.slot() as usize];
        loop {
            let identity = atomic_load_acquire_u64(slot.identity_state.get());
            if identity_generation(identity) != key.generation() {
                return RequeueOutcome::TerminalOrStale;
            }
            if identity_is_locked(identity) {
                core::hint::spin_loop();
                continue;
            }
            match identity_state(identity) {
                state if state == from_state => {
                    if atomic_cas_acq_rel_u64(
                        slot.identity_state.get(),
                        identity,
                        identity_value(key.generation(), SLOT_QUEUED),
                    ) == identity
                    {
                        break;
                    }
                }
                // The winner owns the obligation until it either publishes the
                // queue cell or leaves a stable ENQUEUE_RETRY marker.
                SLOT_QUEUED => return RequeueOutcome::DelegatedToWinner,
                SLOT_ENQUEUE_RETRY => return RequeueOutcome::Retry,
                _ => return RequeueOutcome::TerminalOrStale,
            }
        }

        let priority = match self.effective_priority(key) {
            Ok(priority) => priority,
            Err(_) => return self.mark_enqueue_retry(key),
        };
        match self.enqueue_task(key.local(), priority) {
            Ok(()) => RequeueOutcome::PublishedByMe,
            Err(_) => self.mark_enqueue_retry(key),
        }
    }

    #[inline(always)]
    unsafe fn requeue_parked_tasks(&self) {
        let mut index = 0;
        while index < MAX_TASKS {
            let slot = &self.slots[index];
            let identity = atomic_load_acquire_u64(slot.identity_state.get());
            let state = identity_state(identity);
            if !identity_is_locked(identity)
                && ((state == SLOT_PARKED
                    && atomic_load_acquire_u32(slot.waker_data.refs.get()) == 0)
                    || state == SLOT_ENQUEUE_RETRY)
            {
                self.requeue_slot(
                    self.bind_local_key(LocalTaskKey::from_parts(
                        index as u32,
                        identity_generation(identity),
                    )),
                    state,
                );
            }
            index += 1;
        }
    }

    /// Cancel parked work during terminal shutdown. RUNNING tasks are claimed
    /// by their polling warp after the current cooperative poll returns.
    #[inline(always)]
    unsafe fn cancel_parked_tasks(&self) {
        let mut index = 0;
        while index < MAX_TASKS {
            let slot = &self.slots[index];
            let identity = atomic_load_acquire_u64(slot.identity_state.get());
            let state = identity_state(identity);
            if !identity_is_locked(identity)
                && (state == SLOT_PARKED || state == SLOT_ENQUEUE_RETRY)
                && atomic_cas_acq_rel_u64(
                    slot.identity_state.get(),
                    identity,
                    identity_value(identity_generation(identity), SLOT_COMPLETING),
                ) == identity
            {
                self.retire_claimed(
                    self.bind_local_key(LocalTaskKey::from_parts(
                        index as u32,
                        identity_generation(identity),
                    )),
                    true,
                );
            }
            index += 1;
        }
    }

    /// Spawn a new async task onto the executor.
    ///
    /// The future is copied into a task slot and enqueued for execution.
    /// Any warp currently in `run()` may pick it up.
    ///
    /// # Safety
    /// - `self` must point to valid executor memory in global/mapped space
    /// - The future must be safe to poll from any warp
    /// - Should be called from lane 0 only (single-lane operation)
    /// - The executor address must remain stable from this call until shutdown
    ///   is complete, every executor warp has exited, and execution is
    ///   externally synchronized
    #[inline(always)]
    pub unsafe fn spawn<F: Future<Output = ()>>(&self, future: F) -> Result<TaskId, ExecutorError> {
        self.spawn_with_options(future, TaskOptions::default())
    }

    /// Spawn with an explicit fixed priority.
    ///
    /// Priority is cooperative: it changes dispatch order at poll boundaries,
    /// but never interrupts a currently executing `Future::poll` call.
    #[inline(always)]
    pub unsafe fn spawn_with_priority<F: Future<Output = ()>>(
        &self,
        future: F,
        priority: Priority,
    ) -> Result<TaskId, ExecutorError> {
        self.spawn_with_options(future, TaskOptions::new(priority))
    }

    /// Spawn with [`TaskOptions`].
    #[inline(always)]
    pub unsafe fn spawn_with_options<F: Future<Output = ()>>(
        &self,
        future: F,
        options: TaskOptions,
    ) -> Result<TaskId, ExecutorError> {
        self.spawn_with_factory(options, false, move |_, _, _| (future, ()))
            .map(|(key, ())| key)
    }

    /// Admit a task whose wrapper and returned capability need the executor's
    /// stable context record. This is crate-private so safe callers cannot mint
    /// task identity or priority credentials.
    #[inline(always)]
    pub(crate) unsafe fn spawn_with_factory<Build, F, Extra>(
        &self,
        options: TaskOptions,
        priority_fixed: bool,
        build: Build,
    ) -> Result<(TaskKey, Extra), ExecutorError>
    where
        Build: FnOnce(TaskKey, u64, *const WakerData) -> (F, Extra),
        F: Future<Output = ()>,
    {
        let size = core::mem::size_of::<F>();
        if size > TASK_FUTURE_MAX_SIZE {
            return Err(ExecutorError::FutureTooLarge);
        }
        if core::mem::align_of::<F>() > TASK_FUTURE_MAX_ALIGN {
            return Err(ExecutorError::FutureTooAligned);
        }

        self.reserve_active(options.priority)?;

        let slot_idx = self.free_slots.pop();
        if slot_idx == FREE_NULL {
            self.release_active();
            return Err(ExecutorError::NoFreeSlots);
        }

        let slot = &self.slots[slot_idx as usize];
        let free_identity = atomic_load_acquire_u64(slot.identity_state.get());
        debug_assert_eq!(identity_state(free_identity), SLOT_FREE);
        debug_assert!(!identity_is_locked(free_identity));
        let key = self.bind_local_key(LocalTaskKey::from_parts(
            slot_idx as u32,
            identity_generation(free_identity),
        ));
        let trace_id = match self.allocate_trace_id() {
            Ok(trace_id) => trace_id,
            Err(error) => {
                let returned = self.free_slots.push(slot_idx);
                assert!(returned, "trace rollback slot was already free");
                self.release_active();
                return Err(error);
            }
        };

        core::ptr::write_volatile(slot.waker_data.executor.get(), self as *const Self as u64);
        core::ptr::write_volatile(slot.waker_data.incarnation.get(), key.incarnation);
        core::ptr::write_volatile(slot.waker_data.key.get(), key.local().raw());
        core::ptr::write_volatile(slot.waker_data.trace_id.get(), trace_id);
        core::ptr::write_volatile(slot.waker_data.refs.get(), 0);
        atomic_store_release_u32(slot.waker_data.phase.get(), CONTEXT_LIVE);

        let (future, extra) = build(key, trace_id, &slot.waker_data);

        // Copy future bytes into the slot
        core::ptr::copy_nonoverlapping(
            &future as *const F as *const u8,
            (*slot.future_bytes.get()).0.as_mut_ptr(),
            size,
        );
        core::mem::forget(future); // ownership transferred to slot

        // Publish metadata before transitioning to QUEUED.
        core::ptr::write(slot.poll_fn.get(), Some(erased_poll::<F> as _));
        core::ptr::write(slot.drop_fn.get(), Some(erased_drop::<F> as _));
        core::ptr::write_volatile(slot.future_size.get(), size as u32);
        core::ptr::write_volatile(slot.base_priority.get(), options.priority as u32);
        core::ptr::write_volatile(slot.effective_priority.get(), options.priority as u32);
        core::ptr::write_volatile(slot.priority_fixed.get(), priority_fixed as u32);

        // Publish the new generation only after future and context metadata.
        atomic_store_release_u64(
            slot.identity_state.get(),
            identity_value(key.generation(), SLOT_QUEUED),
        );

        if let Err(error) = self.enqueue_task(key.local(), options.priority) {
            let queued = identity_value(key.generation(), SLOT_QUEUED);
            let claimed = atomic_cas_acq_rel_u64(
                slot.identity_state.get(),
                queued,
                identity_value(key.generation(), SLOT_COMPLETING),
            );
            debug_assert_eq!(claimed, queued);
            self.retire_claimed(key, false);
            drop(extra);
            return Err(error);
        }

        atomic_fetch_add_u32(self.tasks_spawned.get(), 1);

        Ok((key, extra))
    }

    /// Enter the executor loop with waker support.
    ///
    /// Dequeues and polls tasks. When a future returns `Poll::Pending`, the
    /// task is parked (not re-polled). Wakers re-enqueue parked tasks.
    /// Legacy futures that do not register a waker are cooperatively re-polled
    /// whenever all queues become empty. The loop exits when no task is active.
    /// After [`GpuExecutor::shutdown`], it rejects new tasks, drains queued
    /// work, and cancels tasks that remain Pending without waiting for another
    /// task wake. Returning from `run` and satisfying the teardown snapshot
    /// still require every escaped waker/context reference to be released.
    /// Saturated reference counts deliberately pin storage permanently as a
    /// fail-safe, so shutdown has no unconditional time bound.
    ///
    /// `mask` must be the warp's active lane mask (from `activemask()`).
    /// Taking it as a parameter avoids GPU hangs caused by LLVM nvptx
    /// codegen issues with `activemask()` called inside inlined methods.
    ///
    /// # Safety
    /// - Must be called by all active lanes of a warp simultaneously
    /// - `self` must point to valid executor memory in global/mapped space
    #[inline(always)]
    pub unsafe fn run(&self, mask: u32) -> ExecutorStats {
        let lid = lane_id();
        let mut tasks_executed: u32 = 0;
        let mut polls_total: u32 = 0;
        let mut schedule_cursor: usize = 0;

        loop {
            // Lane 0 dequeues and broadcasts the complete generational key.
            let mut raw_key = EMPTY_TASK_KEY;
            if lid == 0 {
                if let Some(key) = self.dequeue_runnable(&mut schedule_cursor) {
                    raw_key = key.local().raw();
                }
            }
            let low = shfl_sync_idx_u32(mask, raw_key as u32, 0);
            let high = shfl_sync_idx_u32(mask, (raw_key >> 32) as u32, 0);
            raw_key = ((high as u64) << 32) | low as u64;
            syncwarp(mask);

            if raw_key == EMPTY_TASK_KEY {
                let mut should_exit: u32 = 0;
                if lid == 0 {
                    let lifecycle = atomic_load_acquire_u32(self.tasks_active.get());
                    let shutting_down = lifecycle & SHUTDOWN_BIT != 0;
                    let active = lifecycle & ACTIVE_COUNT_MASK;
                    if shutting_down {
                        self.cancel_parked_tasks();
                        if self.shutdown_complete() {
                            should_exit = 1;
                        }
                    } else if active == 0
                        && atomic_load_acquire_u32(self.outstanding_context_refs.get()) == 0
                        && atomic_load_acquire_u32(self.pending_reclaims.get()) == 0
                    {
                        should_exit = 1;
                    } else {
                        // Compatibility path for cooperative futures such as
                        // GeneratorTask and oneshot, which intentionally return
                        // Pending without arranging an external wake.
                        self.requeue_parked_tasks();
                    }
                }
                let should_exit = shfl_sync_idx_u32(mask, should_exit, 0);
                syncwarp(mask);
                if should_exit != 0 {
                    break;
                }
                #[cfg(target_arch = "nvptx64")]
                core::arch::asm!("nanosleep.u32 1000;", options(nostack));
                continue;
            }

            let key = self.bind_local_key(LocalTaskKey::from_raw(raw_key));
            if key.slot() as usize >= MAX_TASKS {
                continue;
            }

            let slot = &self.slots[key.slot() as usize];

            let poll_fn = core::ptr::read_volatile(slot.poll_fn.get());
            if poll_fn.is_none() {
                if lid == 0 && self.claim_running_for_completion(key) {
                    self.retire_claimed(key, true);
                }
                syncwarp(mask);
                continue;
            }
            let poll_fn = poll_fn.unwrap();
            let future_ptr = (*slot.future_bytes.get()).0.as_mut_ptr();

            // Poll once — if Pending, park the task (waker will re-enqueue)
            let mut is_ready: u32 = 0;
            if lid == 0 {
                let waker = make_task_waker(&slot.waker_data);
                let mut cx = Context::from_waker(&waker);
                let result = poll_fn(future_ptr, &mut cx);
                is_ready = match result {
                    Poll::Ready(()) => 1,
                    Poll::Pending => 0,
                };
            }
            let is_ready = shfl_sync_idx_u32(mask, is_ready, 0);
            syncwarp(mask);
            polls_total += 1;

            if is_ready != 0 {
                if lid == 0 && self.claim_running_for_completion(key) {
                    self.retire_claimed(key, true);
                }
                syncwarp(mask);
                tasks_executed += 1;
            } else {
                if lid == 0 {
                    if self.is_shutting_down() {
                        if self.claim_running_for_completion(key) {
                            self.retire_claimed(key, true);
                        }
                    } else {
                        loop {
                            let identity = atomic_load_acquire_u64(slot.identity_state.get());
                            if identity_generation(identity) != key.generation() {
                                break;
                            }
                            if identity_is_locked(identity) {
                                continue;
                            }
                            match pending_transition(identity_state(identity)) {
                                PendingTransition::Park => {
                                    if atomic_cas_acq_rel_u64(
                                        slot.identity_state.get(),
                                        identity,
                                        identity_value(key.generation(), SLOT_PARKED),
                                    ) == identity
                                    {
                                        break;
                                    }
                                }
                                PendingTransition::Requeue => {
                                    self.requeue_slot(key, SLOT_NOTIFIED);
                                    break;
                                }
                                PendingTransition::Invalid => break,
                            }
                        }
                    }
                }
                syncwarp(mask);
            }
        }

        ExecutorStats {
            tasks_executed,
            polls_total,
        }
    }

    /// Signal terminal shutdown.
    ///
    /// New spawns are rejected atomically. Queued tasks receive their next
    /// cooperative poll; tasks that remain Pending are dropped and recycled.
    /// A currently executing poll is never preempted.
    ///
    /// # Safety
    /// Must be called by lane 0 of exactly one warp.
    #[inline(always)]
    pub unsafe fn shutdown(&self) {
        loop {
            let old = atomic_load_acquire_u32(self.tasks_active.get());
            if old & SHUTDOWN_BIT != 0 {
                return;
            }
            if atomic_cas_acq_rel_u32(self.tasks_active.get(), old, old | SHUTDOWN_BIT) == old {
                return;
            }
        }
    }

    /// Return whether terminal shutdown has started.
    #[inline(always)]
    pub unsafe fn is_shutting_down(&self) -> bool {
        atomic_load_acquire_u32(self.tasks_active.get()) & SHUTDOWN_BIT != 0
    }

    /// Get the number of admitted tasks that have not completed or cancelled.
    #[inline(always)]
    pub unsafe fn active_count(&self) -> u32 {
        atomic_load_acquire_u32(self.tasks_active.get()) & ACTIVE_COUNT_MASK
    }

    /// Get the number of tasks spawned (diagnostic).
    pub unsafe fn spawned_count(&self) -> u32 {
        atomic_load_acquire_u32(self.tasks_spawned.get())
    }

    /// Get the number of terminal tasks, including shutdown cancellations.
    pub unsafe fn completed_count(&self) -> u32 {
        atomic_load_acquire_u32(self.tasks_completed.get())
    }

    /// Number of escaped waker and typed-context references.
    #[inline(always)]
    pub unsafe fn outstanding_context_refs(&self) -> u32 {
        atomic_load_acquire_u32(self.outstanding_context_refs.get())
    }

    /// Terminal slots still pinned by escaped references.
    #[inline(always)]
    pub unsafe fn pending_reclaims(&self) -> u32 {
        atomic_load_acquire_u32(self.pending_reclaims.get())
    }

    /// Diagnostic quiescence snapshot.
    ///
    /// This does not freeze admission and does not count executor warps that
    /// are still inside run. It is never sufficient authority to free or move
    /// the allocation.
    #[inline(always)]
    pub unsafe fn is_quiescent_snapshot(&self) -> bool {
        self.active_count() == 0
            && self.outstanding_context_refs() == 0
            && self.pending_reclaims() == 0
    }

    /// Strict internal teardown gate after admission has been frozen.
    ///
    /// Even when true, the caller must first join/synchronize every executor
    /// warp and prove that no new run, spawn, or wake can begin before moving,
    /// reinitializing, or freeing the allocation.
    #[inline(always)]
    pub unsafe fn can_teardown(&self) -> bool {
        self.is_shutting_down() && self.is_quiescent_snapshot()
    }

    /// True when shutdown was requested and task/context teardown is complete.
    ///
    /// This has the same external warp-join and synchronization preconditions
    /// as can_teardown.
    #[inline(always)]
    pub unsafe fn shutdown_complete(&self) -> bool {
        self.can_teardown()
    }

    /// Read a live task's base priority.
    #[inline(always)]
    pub unsafe fn base_priority(&self, key: TaskId) -> Result<Priority, TaskKeyError> {
        let (slot, identity) = self.lock_priority(key)?;
        let priority = Priority::from_raw(core::ptr::read_volatile(slot.base_priority.get()) as u8);
        self.unlock_priority(slot, identity);
        Ok(priority)
    }

    /// Read a live task's effective priority.
    #[inline(always)]
    pub unsafe fn effective_priority(&self, key: TaskId) -> Result<Priority, TaskKeyError> {
        let (slot, identity) = self.lock_priority(key)?;
        let priority =
            Priority::from_raw(core::ptr::read_volatile(slot.effective_priority.get()) as u8);
        self.unlock_priority(slot, identity);
        Ok(priority)
    }

    /// Update the priority used the next time a live task is enqueued.
    ///
    /// If the task is already QUEUED, the change takes effect after its next
    /// poll boundary; queue relocation is deliberately avoided to preserve the
    /// one-entry-per-task invariant.
    #[inline(always)]
    pub unsafe fn set_effective_priority(
        &self,
        key: TaskId,
        priority: Priority,
    ) -> Result<PriorityUpdate, TaskKeyError> {
        let (slot, identity) = self.lock_priority(key)?;
        let current =
            Priority::from_raw(core::ptr::read_volatile(slot.effective_priority.get()) as u8);
        if core::ptr::read_volatile(slot.priority_fixed.get()) != 0 && current != priority {
            self.unlock_priority(slot, identity);
            return Err(TaskKeyError::PriorityFixed);
        }
        let update = if identity_state(identity) == SLOT_QUEUED {
            PriorityUpdate::DeferredUntilRequeue
        } else {
            PriorityUpdate::Applied
        };
        core::ptr::write_volatile(slot.effective_priority.get(), priority as u32);
        self.unlock_priority(slot, identity);
        Ok(update)
    }

    #[cfg(test)]
    pub(crate) unsafe fn test_poll_next(&self) -> Option<(TaskKey, bool)> {
        let mut cursor = 0;
        let key = self.dequeue_runnable(&mut cursor)?;
        let slot = &self.slots[key.slot() as usize];
        let poll_fn = core::ptr::read_volatile(slot.poll_fn.get())?;
        let future_ptr = (*slot.future_bytes.get()).0.as_mut_ptr();
        let ready = {
            let waker = make_task_waker(&slot.waker_data);
            let mut cx = Context::from_waker(&waker);
            matches!(poll_fn(future_ptr, &mut cx), Poll::Ready(()))
        };
        if ready {
            if self.claim_running_for_completion(key) {
                self.retire_claimed(key, true);
            }
        } else {
            loop {
                let identity = atomic_load_acquire_u64(slot.identity_state.get());
                match pending_transition(identity_state(identity)) {
                    PendingTransition::Park => {
                        if atomic_cas_acq_rel_u64(
                            slot.identity_state.get(),
                            identity,
                            identity_value(key.generation(), SLOT_PARKED),
                        ) == identity
                        {
                            break;
                        }
                    }
                    PendingTransition::Requeue => {
                        self.requeue_slot(key, SLOT_NOTIFIED);
                        break;
                    }
                    PendingTransition::Invalid => break,
                }
            }
        }
        Some((key, ready))
    }

    #[cfg(test)]
    pub(crate) unsafe fn test_retire_queued(&self, key: TaskKey) -> bool {
        let Ok(local) = self.validate_key_binding(key) else {
            return false;
        };
        let slot = &self.slots[local.slot() as usize];
        let queued = identity_value(key.generation(), SLOT_QUEUED);
        if atomic_cas_acq_rel_u64(
            slot.identity_state.get(),
            queued,
            identity_value(key.generation(), SLOT_COMPLETING),
        ) != queued
        {
            return false;
        }
        self.retire_claimed(key, true);
        true
    }

    #[cfg(test)]
    pub(crate) unsafe fn test_fill_normal_queue(&self) {
        let key = LocalTaskKey::from_parts(0, INITIAL_GENERATION);
        let mut count = 0;
        while count < MAX_TASKS {
            self.work_queue.enqueue(key).unwrap();
            count += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use std::boxed::Box;
    use std::sync::{
        atomic::{AtomicBool, AtomicU32 as HostAtomicU32, Ordering as HostOrdering},
        Barrier,
    };
    use std::thread;
    use std::vec::Vec;

    fn initialized_executor() -> Box<GpuExecutor> {
        let executor = Box::new(GpuExecutor::new());
        unsafe { executor.init() };
        executor
    }

    struct DropProbe<'a>(&'a HostAtomicU32);

    impl Future for DropProbe<'_> {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
            Poll::Ready(())
        }
    }

    impl Drop for DropProbe<'_> {
        fn drop(&mut self) {
            self.0.fetch_add(1, HostOrdering::SeqCst);
        }
    }

    struct Message {
        payload: UnsafeCell<u32>,
        sequence: UnsafeCell<u32>,
        acknowledged: UnsafeCell<u32>,
    }

    unsafe impl Sync for Message {}

    impl Message {
        unsafe fn publish(&self, value: u32) {
            while atomic_load_acquire_u32(self.acknowledged.get()) != value - 1 {
                core::hint::spin_loop();
            }
            core::ptr::write_volatile(self.payload.get(), value);
            atomic_store_release_u32(self.sequence.get(), value);
        }

        unsafe fn consume(&self, expected: u32) {
            while atomic_load_acquire_u32(self.sequence.get()) != expected {
                core::hint::spin_loop();
            }
            assert_eq!(core::ptr::read_volatile(self.payload.get()), expected);
            atomic_store_release_u32(self.acknowledged.get(), expected);
        }
    }

    #[test]
    fn default_spawn_options_are_normal() {
        assert_eq!(TaskOptions::default().priority, Priority::Normal);
        assert_eq!(
            TaskOptions::default()
                .with_priority(Priority::High)
                .priority,
            Priority::High
        );
    }

    #[test]
    fn service_cycle_is_eight_four_one() {
        let mut counts = [0u32; 3];
        for priority in SERVICE_PATTERN {
            counts[priority.as_raw() as usize] += 1;
        }
        assert_eq!(counts[Priority::Low.as_raw() as usize], 1);
        assert_eq!(counts[Priority::Normal.as_raw() as usize], 4);
        assert_eq!(counts[Priority::High.as_raw() as usize], 8);
        assert_eq!(SERVICE_PATTERN[0], Priority::High);
    }

    #[test]
    fn service_cycle_bounds_low_liveness_and_high_delay() {
        let low_index = SERVICE_PATTERN
            .iter()
            .position(|priority| *priority == Priority::Low)
            .unwrap();
        assert_eq!(low_index + 1, SERVICE_PATTERN.len());

        for start in 0..SERVICE_PATTERN.len() {
            let mut lower_before_high = 0;
            while SERVICE_PATTERN[(start + lower_before_high) % SERVICE_PATTERN.len()]
                != Priority::High
            {
                lower_before_high += 1;
            }
            assert!(lower_before_high <= 5);
        }
    }

    #[test]
    fn admission_reserves_capacity_for_normal_and_high() {
        assert_eq!(
            GpuExecutor::admission_limit(Priority::Low),
            (MAX_TASKS - HIGH_PRIORITY_RESERVED_SLOTS - NORMAL_PRIORITY_RESERVED_SLOTS) as u32
        );
        assert_eq!(
            GpuExecutor::admission_limit(Priority::Normal),
            (MAX_TASKS - HIGH_PRIORITY_RESERVED_SLOTS) as u32
        );
        assert_eq!(
            GpuExecutor::admission_limit(Priority::High),
            MAX_TASKS as u32
        );
    }

    #[test]
    fn running_wake_is_not_lost_or_duplicated() {
        assert_eq!(wake_transition(SLOT_RUNNING), WakeTransition::NotifyRunning);
        assert_eq!(
            pending_transition(SLOT_NOTIFIED),
            PendingTransition::Requeue
        );
        assert_eq!(wake_transition(SLOT_NOTIFIED), WakeTransition::Ignore);
        assert_eq!(wake_transition(SLOT_QUEUED), WakeTransition::Ignore);
    }

    #[test]
    fn parked_wake_requeues_once() {
        assert_eq!(wake_transition(SLOT_PARKED), WakeTransition::EnqueueParked);
        assert_eq!(pending_transition(SLOT_RUNNING), PendingTransition::Park);
        assert_eq!(wake_transition(SLOT_FREE), WakeTransition::Ignore);
        assert_eq!(wake_transition(SLOT_COMPLETING), WakeTransition::Ignore);
    }

    #[test]
    fn inline_storage_supports_documented_alignment() {
        assert_eq!(core::mem::align_of::<TaskStorage>(), TASK_FUTURE_MAX_ALIGN);
        assert_eq!(core::mem::size_of::<TaskStorage>(), TASK_FUTURE_MAX_SIZE);
        assert_eq!(core::mem::align_of::<GpuExecutor>(), 256);
    }

    #[test]
    fn mpmc_cell_sequence_distinguishes_ready_full_and_empty() {
        let enqueue_position = 0u64;
        assert_eq!(sequence_difference(0, enqueue_position), 0);

        // After publication, the matching consumer expects position + 1.
        assert_eq!(sequence_difference(1, enqueue_position + 1), 0);

        // A producer one ring later still sees the old published generation:
        // the negative distance means full until the consumer recycles it.
        assert!(sequence_difference(1, MAX_TASKS as u64) < 0);

        // Once consumed, sequence advances one complete ring and is reusable.
        assert_eq!(sequence_difference(MAX_TASKS as u64, MAX_TASKS as u64), 0);

        // The wrapping arithmetic preserves the same relations at u32 rollover.
        assert_eq!(sequence_difference(0, 0), 0);
        assert!(sequence_difference(u64::MAX, 0) < 0);
    }

    #[test]
    fn raw_waker_clone_wake_and_drop_balance_context_refs() {
        let executor = initialized_executor();
        unsafe {
            let key = executor.spawn(async {}).unwrap();
            let mut cursor = 0;
            assert_eq!(executor.dequeue_runnable(&mut cursor), Some(key));
            let slot = &executor.slots[key.slot() as usize];
            let waker = make_task_waker(&slot.waker_data);
            let clone = waker.clone();
            assert_eq!(executor.outstanding_context_refs(), 2);

            clone.wake_by_ref();
            assert_eq!(
                identity_state(atomic_load_acquire_u64(slot.identity_state.get())),
                SLOT_NOTIFIED
            );
            assert_eq!(executor.outstanding_context_refs(), 2);
            drop(clone);
            assert_eq!(executor.outstanding_context_refs(), 1);
            waker.wake();
            assert_eq!(executor.outstanding_context_refs(), 0);

            assert!(executor.claim_running_for_completion(key));
            executor.retire_claimed(key, true);
            assert!(executor.is_quiescent_snapshot());
        }
    }

    #[test]
    fn dequeued_membership_waits_for_priority_lock_then_polls_once() {
        let executor = initialized_executor();
        let finished = AtomicBool::new(false);
        unsafe {
            let key = executor.spawn(async {}).unwrap();
            let slot = &executor.slots[key.slot() as usize];
            let queued = identity_value(key.generation(), SLOT_QUEUED);
            assert_eq!(
                atomic_cas_acq_rel_u64(
                    slot.identity_state.get(),
                    queued,
                    queued | SLOT_PRIORITY_LOCK,
                ),
                queued
            );

            thread::scope(|scope| {
                let worker = scope.spawn(|| {
                    let result = executor.test_poll_next();
                    finished.store(true, HostOrdering::Release);
                    result
                });

                // The queue cell has been recycled, so the worker exclusively
                // owns this generation's consumed membership obligation.
                while atomic_load_acquire_u64(executor.work_queue.head.get()) == 0 {
                    thread::yield_now();
                }
                thread::yield_now();
                assert!(!finished.load(HostOrdering::Acquire));

                atomic_store_release_u64(slot.identity_state.get(), queued);
                assert_eq!(worker.join().unwrap(), Some((key, true)));
            });

            assert_eq!(executor.completed_count(), 1);
            assert!(executor.is_quiescent_snapshot());
        }
    }

    #[test]
    fn notified_requeue_waits_for_priority_lock_and_publishes_once() {
        let executor = initialized_executor();
        let entered = Barrier::new(2);
        let finished = AtomicBool::new(false);
        unsafe {
            let key = executor.spawn(async {}).unwrap();
            let mut cursor = 0;
            assert_eq!(executor.dequeue_runnable(&mut cursor), Some(key));
            let slot = &executor.slots[key.slot() as usize];
            let running = identity_value(key.generation(), SLOT_RUNNING);
            let notified = identity_value(key.generation(), SLOT_NOTIFIED);
            assert_eq!(
                atomic_cas_acq_rel_u64(slot.identity_state.get(), running, notified),
                running
            );

            // Model the polling owner having already observed NOTIFIED, followed
            // by a priority accessor acquiring the identity lock before the
            // owner enters requeue_slot.
            assert_eq!(atomic_load_acquire_u64(slot.identity_state.get()), notified);
            assert_eq!(
                atomic_cas_acq_rel_u64(
                    slot.identity_state.get(),
                    notified,
                    notified | SLOT_PRIORITY_LOCK,
                ),
                notified
            );

            thread::scope(|scope| {
                let worker = scope.spawn(|| {
                    entered.wait();
                    let result = executor.requeue_slot(key, SLOT_NOTIFIED);
                    finished.store(true, HostOrdering::Release);
                    result
                });
                entered.wait();
                thread::yield_now();
                assert!(!finished.load(HostOrdering::Acquire));

                atomic_store_release_u64(slot.identity_state.get(), notified);
                assert_eq!(worker.join().unwrap(), RequeueOutcome::PublishedByMe);
            });

            assert_eq!(executor.dequeue_runnable(&mut cursor), Some(key));
            assert!(executor.claim_running_for_completion(key));
            executor.retire_claimed(key, true);
            assert!(executor.is_quiescent_snapshot());
        }
    }

    #[test]
    fn enqueue_failure_rollback_waits_for_priority_lock() {
        let executor = initialized_executor();
        let entered = Barrier::new(2);
        let finished = AtomicBool::new(false);
        unsafe {
            let key = executor.spawn(async {}).unwrap();
            assert_eq!(executor.work_queue.dequeue(), Some(key.local()));
            let slot = &executor.slots[key.slot() as usize];
            let queued = identity_value(key.generation(), SLOT_QUEUED);
            assert_eq!(
                atomic_cas_acq_rel_u64(
                    slot.identity_state.get(),
                    queued,
                    queued | SLOT_PRIORITY_LOCK,
                ),
                queued
            );

            thread::scope(|scope| {
                let worker = scope.spawn(|| {
                    entered.wait();
                    let result = executor.mark_enqueue_retry(key);
                    finished.store(true, HostOrdering::Release);
                    result
                });
                entered.wait();
                thread::yield_now();
                assert!(!finished.load(HostOrdering::Acquire));

                atomic_store_release_u64(slot.identity_state.get(), queued);
                assert_eq!(worker.join().unwrap(), RequeueOutcome::Retry);
            });

            assert_eq!(
                identity_state(atomic_load_acquire_u64(slot.identity_state.get())),
                SLOT_ENQUEUE_RETRY
            );
            assert_eq!(
                atomic_cas_acq_rel_u64(
                    slot.identity_state.get(),
                    identity_value(key.generation(), SLOT_ENQUEUE_RETRY),
                    identity_value(key.generation(), SLOT_COMPLETING),
                ),
                identity_value(key.generation(), SLOT_ENQUEUE_RETRY)
            );
            executor.retire_claimed(key, true);
            assert!(executor.is_quiescent_snapshot());
        }
    }

    #[test]
    fn enqueue_retry_with_retained_waker_recovers_without_second_wake() {
        let executor = initialized_executor();
        unsafe {
            let key = executor.spawn(async {}).unwrap();
            let mut cursor = 0;
            assert_eq!(executor.dequeue_runnable(&mut cursor), Some(key));
            let slot = &executor.slots[key.slot() as usize];
            let retained = make_task_waker(&slot.waker_data);
            let running = identity_value(key.generation(), SLOT_RUNNING);
            assert_eq!(
                atomic_cas_acq_rel_u64(
                    slot.identity_state.get(),
                    running,
                    identity_value(key.generation(), SLOT_PARKED),
                ),
                running
            );
            executor.test_fill_normal_queue();

            retained.wake_by_ref();
            assert_eq!(
                identity_state(atomic_load_acquire_u64(slot.identity_state.get())),
                SLOT_ENQUEUE_RETRY
            );
            while executor.work_queue.dequeue().is_some() {}
            executor.requeue_parked_tasks();
            assert_eq!(executor.dequeue_runnable(&mut cursor), Some(key));

            drop(retained);
            assert!(executor.claim_running_for_completion(key));
            executor.retire_claimed(key, true);
            assert!(executor.is_quiescent_snapshot());
        }
    }

    #[test]
    fn retained_waker_excludes_task_from_legacy_rescan() {
        let executor = initialized_executor();
        unsafe {
            let key = executor.spawn(async {}).unwrap();
            let mut cursor = 0;
            assert_eq!(executor.dequeue_runnable(&mut cursor), Some(key));
            let slot = &executor.slots[key.slot() as usize];
            let waker = make_task_waker(&slot.waker_data);
            assert_eq!(
                atomic_cas_acq_rel_u64(
                    slot.identity_state.get(),
                    identity_value(key.generation(), SLOT_RUNNING),
                    identity_value(key.generation(), SLOT_PARKED),
                ),
                identity_value(key.generation(), SLOT_RUNNING)
            );
            executor.requeue_parked_tasks();
            assert_eq!(
                identity_state(atomic_load_acquire_u64(slot.identity_state.get())),
                SLOT_PARKED
            );
            drop(waker);
            executor.requeue_parked_tasks();
            assert_eq!(executor.dequeue_runnable(&mut cursor), Some(key));
            assert!(executor.claim_running_for_completion(key));
            executor.retire_claimed(key, true);
            assert!(executor.is_quiescent_snapshot());
        }
    }

    #[test]
    fn terminal_reclaim_waits_for_old_waker_and_stale_key_cannot_wake_reuse() {
        let executor = initialized_executor();
        unsafe {
            let old_key = executor.spawn(async {}).unwrap();
            let mut cursor = 0;
            assert_eq!(executor.dequeue_runnable(&mut cursor), Some(old_key));
            let old_slot = &executor.slots[old_key.slot() as usize];
            let old_waker = make_task_waker(&old_slot.waker_data);

            assert!(executor.claim_running_for_completion(old_key));
            executor.retire_claimed(old_key, true);
            executor.shutdown();
            assert_eq!(executor.pending_reclaims(), 1);
            assert!(!executor.can_teardown());
            assert!(!executor.shutdown_complete());
            assert_eq!(
                identity_state(atomic_load_acquire_u64(old_slot.identity_state.get())),
                SLOT_COMPLETING
            );

            drop(old_waker);
            assert_eq!(executor.pending_reclaims(), 0);
            assert!(executor.shutdown_complete());

            // Reinitialize only after quiescence, then demonstrate that the old
            // generation has no authority over a later occupant of slot zero.
            executor.init();
            let new_key = executor.spawn(async {}).unwrap();
            assert_eq!(new_key.slot(), old_key.slot());
            // init starts a new epoch, so use an in-epoch recycle below to
            // exercise the generation check rather than comparing epochs.
            assert!(executor.test_retire_queued(new_key));
            let reused_key = executor.spawn(async {}).unwrap();
            assert_eq!(reused_key.slot(), new_key.slot());
            assert_ne!(reused_key.generation(), new_key.generation());
            let queued = identity_value(reused_key.generation(), SLOT_QUEUED);
            let parked = identity_value(reused_key.generation(), SLOT_PARKED);
            assert_eq!(
                atomic_cas_acq_rel_u64(
                    executor.slots[reused_key.slot() as usize]
                        .identity_state
                        .get(),
                    queued,
                    parked,
                ),
                queued
            );
            executor.wake_task(new_key);
            assert_eq!(
                identity_state(atomic_load_acquire_u64(
                    executor.slots[reused_key.slot() as usize]
                        .identity_state
                        .get(),
                )),
                SLOT_PARKED
            );
            assert_eq!(
                executor.set_effective_priority(new_key, Priority::High),
                Err(TaskKeyError::Stale)
            );
            assert_eq!(
                executor.effective_priority(reused_key),
                Ok(Priority::Normal)
            );
            assert_eq!(
                atomic_cas_acq_rel_u64(
                    executor.slots[reused_key.slot() as usize]
                        .identity_state
                        .get(),
                    parked,
                    identity_value(reused_key.generation(), SLOT_COMPLETING),
                ),
                parked
            );
            executor.retire_claimed(reused_key, true);
            assert!(executor.is_quiescent_snapshot());
        }
    }

    #[test]
    fn bitmap_and_generation_survive_more_than_u16_tag_space() {
        let executor = initialized_executor();
        let first = unsafe { executor.spawn(async {}).unwrap() };
        unsafe {
            assert_eq!(executor.test_poll_next(), Some((first, true)));
        }
        let mut latest = first;
        for _ in 0..70_000 {
            latest = unsafe { executor.spawn(async {}).unwrap() };
            assert_eq!(latest.slot(), first.slot());
            unsafe {
                assert_eq!(executor.test_poll_next(), Some((latest, true)));
            }
        }
        assert!(latest.generation() > 65_536);
        assert_eq!(
            unsafe { executor.set_effective_priority(first, Priority::High) },
            Err(TaskKeyError::Stale)
        );
        assert!(unsafe { executor.is_quiescent_snapshot() });
    }

    #[test]
    fn queue_full_rollback_drops_future_and_restores_all_accounting() {
        let executor = initialized_executor();
        let dropped = HostAtomicU32::new(0);
        unsafe {
            executor.test_fill_normal_queue();
            assert_eq!(
                executor.spawn(DropProbe(&dropped)),
                Err(ExecutorError::QueueFull)
            );
            assert_eq!(dropped.load(HostOrdering::SeqCst), 1);
            assert_eq!(executor.active_count(), 0);
            assert_eq!(executor.outstanding_context_refs(), 0);
            assert_eq!(executor.pending_reclaims(), 0);
            assert!(executor.is_quiescent_snapshot());
            assert_ne!(
                atomic_load_acquire_u64(executor.free_slots.words[0].get()) & 1,
                0
            );
        }
    }

    #[test]
    fn queued_priority_update_reports_deferred_without_duplicate_membership() {
        let executor = initialized_executor();
        unsafe {
            let key = executor.spawn(async {}).unwrap();
            assert_eq!(
                executor.set_effective_priority(key, Priority::High),
                Ok(PriorityUpdate::DeferredUntilRequeue)
            );
            assert_eq!(executor.effective_priority(key), Ok(Priority::High));
            assert_eq!(executor.test_poll_next(), Some((key, true)));
            assert_eq!(executor.completed_count(), 1);
            assert!(executor.is_quiescent_snapshot());
        }
    }

    #[test]
    fn namespace_trace_ids_are_distinct_and_exhaustion_does_not_wrap() {
        let executor = Box::new(GpuExecutor::new());
        unsafe {
            executor.init_with_namespace(0xA5A5_1234);
            let key = executor.spawn(async {}).unwrap();
            let trace = atomic_load_acquire_u64(
                executor.slots[key.slot() as usize]
                    .waker_data
                    .trace_id
                    .get(),
            );
            assert_eq!(trace >> 32, 0xA5A5_1234);
            assert_eq!(trace as u32, 1);
            assert!(executor.test_retire_queued(key));

            atomic_store_release_u32(executor.trace_sequence.get(), u32::MAX);
            assert_eq!(
                executor.spawn(async {}),
                Err(ExecutorError::TraceIdExhausted)
            );
            assert_eq!(executor.active_count(), 0);
            assert!(executor.is_quiescent_snapshot());
        }
    }

    #[test]
    fn slot_return_side_effect_survives_release_build() {
        let executor = initialized_executor();
        unsafe {
            let first = executor
                .spawn_with_priority(async {}, Priority::High)
                .unwrap();
            assert!(executor.test_retire_queued(first));
            let recycled = executor
                .spawn_with_priority(async {}, Priority::High)
                .unwrap();
            assert_eq!(recycled.slot(), first.slot());
            assert_eq!(recycled.generation(), first.generation() + 1);
            assert!(executor.test_retire_queued(recycled));
            assert!(executor.is_quiescent_snapshot());
        }

        // Use a fresh queue: `test_retire_queued` deliberately leaves stale
        // queue entries for dequeue validation, which is unrelated to the
        // trace-allocation rollback being checked here.
        let trace_executor = initialized_executor();
        unsafe {
            atomic_store_release_u32(trace_executor.trace_sequence.get(), u32::MAX);
            assert_eq!(
                trace_executor.spawn_with_priority(async {}, Priority::High),
                Err(ExecutorError::TraceIdExhausted)
            );
            atomic_store_release_u32(trace_executor.trace_sequence.get(), 0);

            let mut keys = Vec::with_capacity(MAX_TASKS);
            for _ in 0..MAX_TASKS {
                keys.push(
                    trace_executor
                        .spawn_with_priority(async {}, Priority::High)
                        .unwrap(),
                );
            }
            assert_eq!(
                trace_executor.spawn_with_priority(async {}, Priority::High),
                Err(ExecutorError::NoFreeSlots)
            );
            for key in keys {
                assert!(trace_executor.test_retire_queued(key));
            }
            assert!(trace_executor.is_quiescent_snapshot());
        }
    }

    #[test]
    fn free_slot_bitmap_return_is_one_shot() {
        let bitmap = FreeSlotBitmap::empty();
        unsafe {
            bitmap.init(1);
            assert_eq!(bitmap.pop(), 0);
            assert!(bitmap.push(0));
            assert!(!bitmap.push(0));
        }
    }

    #[test]
    fn strict_teardown_gate_requires_shutdown() {
        let executor = initialized_executor();
        unsafe {
            assert!(executor.is_quiescent_snapshot());
            assert!(!executor.can_teardown());
            assert!(!executor.shutdown_complete());
            executor.shutdown();
            assert!(executor.can_teardown());
            assert!(executor.shutdown_complete());
        }
    }

    #[test]
    fn saturated_global_refcount_never_unpins_on_finite_release() {
        let executor = initialized_executor();
        unsafe {
            let data = &executor.slots[0].waker_data;
            core::ptr::write_volatile(data.executor.get(), executor.owner_id());
            core::ptr::write_volatile(data.incarnation.get(), executor.current_incarnation());
            core::ptr::write_volatile(
                data.key.get(),
                LocalTaskKey::from_parts(0, INITIAL_GENERATION).raw(),
            );
            atomic_store_release_u32(data.refs.get(), 1);
            atomic_store_release_u32(executor.outstanding_context_refs.get(), u32::MAX);

            release_context(data);
            assert_eq!(atomic_load_acquire_u32(data.refs.get()), 0);
            assert_eq!(executor.outstanding_context_refs(), u32::MAX);
        }
    }

    #[test]
    fn context_refcount_saturation_permanently_pins_teardown_gate() {
        let executor = initialized_executor();
        unsafe {
            let data = &executor.slots[0].waker_data;
            core::ptr::write_volatile(data.executor.get(), executor.owner_id());
            core::ptr::write_volatile(data.incarnation.get(), executor.current_incarnation());
            core::ptr::write_volatile(
                data.key.get(),
                LocalTaskKey::from_parts(0, INITIAL_GENERATION).raw(),
            );
            atomic_store_release_u32(data.phase.get(), CONTEXT_LIVE);
            atomic_store_release_u32(data.refs.get(), u32::MAX - 1);
            atomic_store_release_u32(executor.outstanding_context_refs.get(), u32::MAX - 1);

            retain_context(data);
            assert_eq!(atomic_load_acquire_u32(data.refs.get()), u32::MAX);
            assert_eq!(executor.outstanding_context_refs(), u32::MAX);

            release_context(data);
            assert_eq!(atomic_load_acquire_u32(data.refs.get()), u32::MAX);
            assert_eq!(executor.outstanding_context_refs(), u32::MAX);
            assert!(!executor.is_quiescent_snapshot());
        }
    }

    #[test]
    fn public_task_key_is_bound_to_issuing_executor() {
        let first = initialized_executor();
        let second = initialized_executor();
        unsafe {
            let first_key = first.spawn(async {}).unwrap();
            let second_key = second.spawn(async {}).unwrap();
            assert_eq!(first_key.slot(), second_key.slot());
            assert_eq!(first_key.generation(), second_key.generation());
            assert_eq!(
                second.set_effective_priority(first_key, Priority::High),
                Err(TaskKeyError::WrongExecutor)
            );
            assert_eq!(
                first.set_effective_priority(second_key, Priority::High),
                Err(TaskKeyError::WrongExecutor)
            );
            assert!(first.test_retire_queued(first_key));
            assert!(second.test_retire_queued(second_key));
        }
    }

    #[test]
    fn old_public_task_key_is_rejected_after_reinit() {
        let executor = initialized_executor();
        unsafe {
            let old_key = executor.spawn(async {}).unwrap();
            assert_eq!(executor.test_poll_next(), Some((old_key, true)));
            executor.shutdown();
            assert!(executor.shutdown_complete());
            executor.init();
            let new_key = executor.spawn(async {}).unwrap();
            assert_eq!(old_key.slot(), new_key.slot());
            assert_eq!(old_key.generation(), new_key.generation());
            assert_eq!(
                executor.set_effective_priority(old_key, Priority::High),
                Err(TaskKeyError::WrongIncarnation)
            );
            assert_eq!(executor.effective_priority(new_key), Ok(Priority::Normal));
            assert!(executor.test_retire_queued(new_key));
        }
    }

    #[test]
    fn incarnation_allocator_exhaustion_is_sticky() {
        let sequence = UnsafeCell::new(u64::MAX - 1);
        unsafe {
            assert_eq!(
                allocate_nonwrapping_incarnation(sequence.get()),
                Some(u64::MAX)
            );
            assert_eq!(allocate_nonwrapping_incarnation(sequence.get()), None);
            assert_eq!(atomic_load_acquire_u64(sequence.get()), u64::MAX);
        }
    }

    #[test]
    fn release_acquire_message_passing_litmus() {
        let message = Message {
            payload: UnsafeCell::new(0),
            sequence: UnsafeCell::new(0),
            acknowledged: UnsafeCell::new(0),
        };
        const ITERATIONS: u32 = 20_000;
        thread::scope(|scope| {
            scope.spawn(|| unsafe {
                let mut expected = 1;
                while expected <= ITERATIONS {
                    message.publish(expected);
                    expected += 1;
                }
            });
            scope.spawn(|| unsafe {
                let mut expected = 1;
                while expected <= ITERATIONS {
                    message.consume(expected);
                    expected += 1;
                }
            });
        });
    }
}
