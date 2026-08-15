//! Typed cooperative-priority dependencies for [`GpuExecutor`].
//!
//! This module closes one specific priority-inversion footgun: code using
//! [`PriorityToken::wait_for`] cannot make a higher-priority task wait for a
//! lower-priority typed handle. It does not claim whole-program control over
//! ordinary `Future` implementations, raw channels, locks, or direct polling.
//!
//! A token is an explicit, non-cloneable capability rather than transparent
//! per-warp state. Safe code can still move that capability into another task,
//! so the guarantee applies to dependency edges formed with the token supplied
//! to that task's builder; it is not an ambient identity or information-flow
//! proof. Metadata is read from the live executor context and fails once that
//! task generation has entered terminal cleanup.

use core::cell::UnsafeCell;
use core::future::Future;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::pin::Pin;
use core::task::{Context, Poll};

use gpu_protocol::HostcallMetadata;

use crate::executor::{
    atomic_cas_acq_rel_u32, atomic_load_acquire_u32, atomic_store_release_u32, ExecutorError,
    GpuExecutor, Priority, TaskContextRef, TaskId, TaskKeyError, TaskOptions,
};

/// Error returned when a typed context no longer names a live task generation.
pub type TaskContextError = TaskKeyError;

const COMPLETION_EMPTY: u32 = 0;
const COMPLETION_RUNNING: u32 = 1;
const COMPLETION_READY: u32 = 2;
const COMPLETION_CANCELLED: u32 = 3;

mod sealed {
    pub trait Sealed {}
}

/// Type-level task-priority markers.
pub mod level {
    /// Best-effort task marker.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Low;

    /// Default task marker.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Normal;

    /// Latency-sensitive task marker.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct High;

    impl super::sealed::Sealed for Low {}
    impl super::sealed::Sealed for Normal {}
    impl super::sealed::Sealed for High {}
}

/// Sealed type-level mapping to the shared runtime/wire [`Priority`].
pub trait PriorityLevel: sealed::Sealed + Clone + Copy + 'static {
    /// Runtime priority used by the executor and hostcall wire metadata.
    const RUNTIME: Priority;
}

impl PriorityLevel for level::Low {
    const RUNTIME: Priority = Priority::Low;
}

impl PriorityLevel for level::Normal {
    const RUNTIME: Priority = Priority::Normal;
}

impl PriorityLevel for level::High {
    const RUNTIME: Priority = Priority::High;
}

/// Compile-time permission for a waiter to depend on a target task.
///
/// The implementations are intentionally explicit and closed: a waiter may
/// only wait for a task at the same or a higher priority.
pub trait MayWaitFor<Target: PriorityLevel>: PriorityLevel {}

impl MayWaitFor<level::Low> for level::Low {}
impl MayWaitFor<level::Normal> for level::Low {}
impl MayWaitFor<level::High> for level::Low {}
impl MayWaitFor<level::Normal> for level::Normal {}
impl MayWaitFor<level::High> for level::Normal {}
impl MayWaitFor<level::High> for level::High {}

/// Publicly constructible request for the priority of a task being spawned.
///
/// This is not a current-task credential: any task may fire-and-forget work at
/// any class. Only [`PriorityToken`] represents the executor-issued context of
/// a task that was actually admitted at a particular priority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PriorityClass<P: PriorityLevel> {
    marker: PhantomData<P>,
}

impl<P: PriorityLevel> PriorityClass<P> {
    /// Request task admission at `P`.
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    /// Return the runtime/wire priority represented by this class.
    #[inline(always)]
    pub const fn runtime_priority(self) -> Priority {
        P::RUNTIME
    }
}

impl<P: PriorityLevel> Default for PriorityClass<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Explicit metadata propagated from a typed task to a hostcall.
///
/// `task_id` is a monotonic logical trace ID allocated by the executor. It is
/// never interpreted as a recyclable task-slot index.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskMetadata {
    /// Logical task/trace identifier.
    pub task_id: u64,
    /// Priority that is effective for this typed context.
    pub effective_priority: Priority,
}

impl TaskMetadata {
    #[inline(always)]
    const fn new(task_id: u64, effective_priority: Priority) -> Self {
        Self {
            task_id,
            effective_priority,
        }
    }

    /// Convert to the shared hostcall wire representation.
    #[inline(always)]
    pub const fn wire(self) -> HostcallMetadata {
        HostcallMetadata::new(self.task_id, self.effective_priority)
    }
}

impl From<TaskMetadata> for HostcallMetadata {
    fn from(metadata: TaskMetadata) -> Self {
        metadata.wire()
    }
}

/// Executor-issued proof of the current typed task's priority.
///
/// There is deliberately no public constructor. A token is handed to the
/// builder passed to [`GpuExecutor::spawn_typed`] only after its wrapper task
/// has been admitted using the same type-level priority.
pub struct PriorityToken<P: PriorityLevel> {
    context: TaskContextRef,
    marker: PhantomData<P>,
}

impl<P: PriorityLevel> PriorityToken<P> {
    #[inline(always)]
    fn from_context(context: TaskContextRef) -> Self {
        Self {
            context,
            marker: PhantomData,
        }
    }

    /// Live metadata to pass explicitly into a GPU hostcall.
    #[inline(always)]
    pub fn metadata(&self) -> Result<TaskMetadata, TaskContextError> {
        Ok(TaskMetadata::new(
            self.context.trace_id(),
            self.context.effective_priority()?,
        ))
    }

    /// Shared hostcall wire metadata.
    #[inline(always)]
    pub fn wire_metadata(&self) -> Result<HostcallMetadata, TaskContextError> {
        self.metadata().map(TaskMetadata::wire)
    }

    /// Monotonic logical trace ID of the admitted task.
    #[inline(always)]
    pub fn trace_id(&self) -> u64 {
        self.context.trace_id()
    }

    /// Effective runtime/wire priority of the admitted task.
    #[inline(always)]
    pub fn effective_priority(&self) -> Result<Priority, TaskContextError> {
        self.context.effective_priority()
    }

    /// Create the only safe await adapter for a typed task handle.
    ///
    /// This method is unavailable when `P` is higher than `Target`, causing a
    /// compile-time error before a priority-inverting dependency can be formed.
    #[inline(always)]
    pub fn wait_for<'a, Target, T>(
        &self,
        handle: &'a mut TypedJoinHandle<Target, T>,
    ) -> TypedWait<'a, Target, T>
    where
        P: MayWaitFor<Target>,
        Target: PriorityLevel,
        T: Copy + Send + Sync + 'static,
    {
        TypedWait { handle }
    }
}

/// Alias emphasizing that the token is an explicitly propagated task context.
pub type TaskContext<P> = PriorityToken<P>;

/// Caller-owned, single-writer completion storage for a typed task.
///
/// The cell avoids using a recyclable [`TaskId`] as a join capability. It must
/// reside in aligned global/mapped memory and remain valid for the full task
/// lifetime. A cell is single-use: its state never returns to EMPTY, which
/// prevents an old handle from observing a later spawn through completion-cell
/// reuse. `spawn_typed` therefore requires a `'static` reference; creating that
/// reference from raw GPU memory remains part of the method's unsafe contract.
#[repr(C)]
pub struct CompletionCell<T: Copy + Send + Sync + 'static> {
    state: UnsafeCell<u32>,
    value: UnsafeCell<MaybeUninit<T>>,
}

impl<T: Copy + Send + Sync + 'static> CompletionCell<T> {
    /// Construct a single-use cell. [`GpuExecutor::spawn_typed`] prepares it.
    pub const fn new() -> Self {
        Self {
            state: UnsafeCell::new(COMPLETION_EMPTY),
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    #[inline(always)]
    unsafe fn prepare(&self) -> Result<(), ExecutorError> {
        if atomic_cas_acq_rel_u32(self.state.get(), COMPLETION_EMPTY, COMPLETION_RUNNING)
            == COMPLETION_EMPTY
        {
            Ok(())
        } else {
            Err(ExecutorError::CompletionUnavailable)
        }
    }

    /// Publish the only result. The wrapper task is the unique writer.
    #[inline(always)]
    unsafe fn complete(&self, value: T) {
        core::ptr::write_volatile((*self.value.get()).as_mut_ptr(), value);
        atomic_store_release_u32(self.state.get(), COMPLETION_READY);
    }

    /// Cancellation never overwrites a result that reached READY.
    #[inline(always)]
    unsafe fn cancel(&self) {
        let _ = atomic_cas_acq_rel_u32(self.state.get(), COMPLETION_RUNNING, COMPLETION_CANCELLED);
    }

    #[inline(always)]
    unsafe fn poll_result(&self) -> CompletionPoll<T> {
        match atomic_load_acquire_u32(self.state.get()) {
            COMPLETION_READY => {
                CompletionPoll::Ready(core::ptr::read_volatile((*self.value.get()).as_ptr()))
            }
            COMPLETION_CANCELLED => CompletionPoll::Cancelled,
            _ => CompletionPoll::Pending,
        }
    }
}

impl<T: Copy + Send + Sync + 'static> Default for CompletionCell<T> {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<T: Copy + Send + Sync + 'static> Send for CompletionCell<T> {}
unsafe impl<T: Copy + Send + Sync + 'static> Sync for CompletionCell<T> {}

enum CompletionPoll<T> {
    Pending,
    Ready(T),
    Cancelled,
}

/// Error returned by a typed join operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypedJoinError {
    /// Executor shutdown or spawn rollback cancelled the target task.
    Cancelled,
    /// This handle already yielded its single result.
    AlreadyJoined,
}

/// Typed join capability for an admitted task.
///
/// This type intentionally does not implement [`Future`]. A caller must use
/// [`PriorityToken::wait_for`], where the [`MayWaitFor`] bound is enforced.
pub struct TypedJoinHandle<P, T>
where
    P: PriorityLevel,
    T: Copy + Send + Sync + 'static,
{
    completion: &'static CompletionCell<T>,
    executor_task_id: TaskId,
    context: Option<TaskContextRef>,
    joined: bool,
    marker: PhantomData<P>,
}

impl<P, T> TypedJoinHandle<P, T>
where
    P: PriorityLevel,
    T: Copy + Send + Sync + 'static,
{
    /// Generational executor key for diagnostics and priority operations.
    #[inline(always)]
    pub const fn executor_task_id(&self) -> TaskId {
        self.executor_task_id
    }

    /// Metadata read from the target's live task context.
    #[inline(always)]
    pub fn metadata(&self) -> Result<TaskMetadata, TaskContextError> {
        let context = self.context.as_ref().ok_or(TaskKeyError::Terminated)?;
        Ok(TaskMetadata::new(
            context.trace_id(),
            context.effective_priority()?,
        ))
    }
}

/// Future produced only after a valid typed dependency has been checked.
pub struct TypedWait<'a, P, T>
where
    P: PriorityLevel,
    T: Copy + Send + Sync + 'static,
{
    handle: &'a mut TypedJoinHandle<P, T>,
}

impl<P, T> Future for TypedWait<'_, P, T>
where
    P: PriorityLevel,
    T: Copy + Send + Sync + 'static,
{
    type Output = Result<T, TypedJoinError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };
        if this.handle.joined {
            return Poll::Ready(Err(TypedJoinError::AlreadyJoined));
        }

        match unsafe { this.handle.completion.poll_result() } {
            CompletionPoll::Ready(value) => {
                this.handle.joined = true;
                this.handle.context.take();
                Poll::Ready(Ok(value))
            }
            CompletionPoll::Cancelled => {
                this.handle.joined = true;
                this.handle.context.take();
                Poll::Ready(Err(TypedJoinError::Cancelled))
            }
            CompletionPoll::Pending => {
                // Explicit self-wake gives this dependency progress independent
                // of the executor's compatibility scan for legacy futures.
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
}

enum CompletionTaskState<Build, F> {
    Building(Option<Build>),
    Running(F),
}

struct CompletionTask<Build, F, P, T>
where
    P: PriorityLevel,
    T: Copy + Send + Sync + 'static,
{
    state: CompletionTaskState<Build, F>,
    token: Option<PriorityToken<P>>,
    completion: &'static CompletionCell<T>,
    terminal: bool,
}

impl<Build, F, P, T> Future for CompletionTask<Build, F, P, T>
where
    Build: FnOnce(PriorityToken<P>) -> F + Unpin,
    F: Future<Output = T>,
    P: PriorityLevel,
    T: Copy + Send + Sync + 'static,
{
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = unsafe { self.get_unchecked_mut() };

        if let CompletionTaskState::Building(builder) = &mut this.state {
            let builder = builder
                .take()
                .expect("typed task builder polled after consumption");
            let token = this
                .token
                .take()
                .expect("typed task token consumed before builder");
            let future = builder(token);
            this.state = CompletionTaskState::Running(future);
        }

        let CompletionTaskState::Running(future) = &mut this.state else {
            unreachable!();
        };
        match unsafe { Pin::new_unchecked(future) }.poll(cx) {
            Poll::Ready(value) => {
                unsafe { this.completion.complete(value) };
                this.terminal = true;
                Poll::Ready(())
            }
            Poll::Pending => {
                // Typed context refs intentionally keep this task out of the
                // legacy refs==0 scanner. Self-wake guarantees progress for an
                // inner future that returns Pending without retaining a waker.
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
}

impl<Build, F, P, T> Drop for CompletionTask<Build, F, P, T>
where
    P: PriorityLevel,
    T: Copy + Send + Sync + 'static,
{
    fn drop(&mut self) {
        if !self.terminal {
            unsafe { self.completion.cancel() };
        }
    }
}

impl GpuExecutor {
    /// Spawn a result-producing task with a compiler-checked priority context.
    ///
    /// The builder is not invoked until the admitted task receives its first
    /// poll. Consequently, an executor-issued token cannot escape from a spawn
    /// that was rejected. The token marker, actual queue priority, handle type,
    /// and wire metadata are all derived from the same `PriorityClass<P>`.
    /// The builder must be `Unpin` because it is moved out of the pinned wrapper
    /// exactly once, before the resulting future is installed and pinned.
    ///
    /// # Safety
    /// In addition to [`GpuExecutor::spawn_with_priority`] requirements:
    /// - `completion` must not have been used by an earlier typed spawn;
    /// - it must live in aligned GPU global/mapped memory for the entire task;
    /// - the builder and resulting future must be safe to run on any executor
    ///   warp and must not retain references shorter than the task lifetime.
    #[inline(always)]
    pub unsafe fn spawn_typed<P, Build, F, T>(
        &self,
        _class: PriorityClass<P>,
        completion: &'static CompletionCell<T>,
        builder: Build,
    ) -> Result<TypedJoinHandle<P, T>, ExecutorError>
    where
        P: PriorityLevel,
        Build: FnOnce(PriorityToken<P>) -> F + Unpin,
        F: Future<Output = T>,
        T: Copy + Send + Sync + 'static,
    {
        completion.prepare()?;
        let spawned = self.spawn_with_factory(
            TaskOptions::new(P::RUNTIME),
            true,
            move |key, _trace_id, data| {
                let token = PriorityToken::<P>::from_context(TaskContextRef::acquire(data));
                let handle_context = TaskContextRef::acquire(data);
                (
                    CompletionTask {
                        state: CompletionTaskState::Building(Some(builder)),
                        token: Some(token),
                        completion,
                        terminal: false,
                    },
                    (key, handle_context),
                )
            },
        );

        match spawned {
            Ok((_key, (task_id, context))) => Ok(TypedJoinHandle {
                completion,
                executor_task_id: task_id,
                context: Some(context),
                joined: false,
                marker: PhantomData,
            }),
            Err(error) => {
                completion.cancel();
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use std::boxed::Box;
    use std::task::{RawWaker, RawWakerVTable, Waker};

    unsafe fn noop_clone(data: *const ()) -> RawWaker {
        RawWaker::new(data, &NOOP_VTABLE)
    }

    unsafe fn noop(_data: *const ()) {}

    static NOOP_VTABLE: RawWakerVTable = RawWakerVTable::new(noop_clone, noop, noop, noop);

    fn noop_waker() -> Waker {
        unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &NOOP_VTABLE)) }
    }

    struct PendingOnce {
        polled: bool,
    }

    impl Future for PendingOnce {
        type Output = u32;

        fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            if self.polled {
                Poll::Ready(9)
            } else {
                self.polled = true;
                Poll::Pending
            }
        }
    }
    fn assert_wait_allowed<Waiter, Target>()
    where
        Waiter: PriorityLevel + MayWaitFor<Target>,
        Target: PriorityLevel,
    {
    }

    #[test]
    fn runtime_priority_mapping_is_shared_with_wire_protocol() {
        assert_eq!(level::Low::RUNTIME, Priority::Low);
        assert_eq!(level::Normal::RUNTIME, Priority::Normal);
        assert_eq!(level::High::RUNTIME, Priority::High);
    }

    #[test]
    fn allowed_wait_edges_are_present() {
        assert_wait_allowed::<level::Low, level::Low>();
        assert_wait_allowed::<level::Low, level::Normal>();
        assert_wait_allowed::<level::Low, level::High>();
        assert_wait_allowed::<level::Normal, level::Normal>();
        assert_wait_allowed::<level::Normal, level::High>();
        assert_wait_allowed::<level::High, level::High>();
    }

    #[test]
    fn task_metadata_round_trips_to_wire() {
        let metadata = TaskMetadata::new(42, Priority::High);
        let wire = metadata.wire();
        assert_eq!(metadata.task_id, 42);
        assert_eq!(metadata.effective_priority, Priority::High);
        assert_eq!(wire.task_id, 42);
        assert_eq!(wire.effective_priority, Priority::High);
    }

    #[test]
    fn typed_metadata_tracks_live_effective_priority_and_rejects_terminal_context() {
        let executor = Box::new(GpuExecutor::new());
        let completion: &'static CompletionCell<u32> =
            Box::leak(Box::new(CompletionCell::<u32>::new()));
        unsafe { executor.init_with_namespace(0x1234_5678) };
        let handle = unsafe {
            executor
                .spawn_typed(PriorityClass::<level::High>::new(), completion, |_| async {
                    7
                })
                .unwrap()
        };
        let key = handle.executor_task_id();
        let initial = handle.metadata().unwrap();
        assert_eq!(initial.task_id >> 32, 0x1234_5678);
        assert_eq!(initial.effective_priority, Priority::High);
        assert_eq!(initial.wire().effective_priority, Priority::High);
        assert_eq!(unsafe { executor.outstanding_context_refs() }, 2);
        assert_eq!(
            unsafe { executor.set_effective_priority(key, Priority::High) },
            Ok(crate::executor::PriorityUpdate::DeferredUntilRequeue)
        );

        assert_eq!(
            unsafe { executor.set_effective_priority(key, Priority::Normal) },
            Err(TaskKeyError::PriorityFixed)
        );
        assert_eq!(
            handle.metadata().unwrap().effective_priority,
            Priority::High
        );

        assert!(unsafe { executor.test_retire_queued(key) });
        assert_eq!(handle.metadata(), Err(TaskKeyError::Terminated));
        assert_eq!(unsafe { executor.outstanding_context_refs() }, 1);
        assert_eq!(unsafe { executor.pending_reclaims() }, 1);
        drop(handle);
        assert!(unsafe { executor.is_quiescent_snapshot() });
    }

    #[test]
    fn typed_pending_without_inner_wake_is_requeued() {
        let executor = Box::new(GpuExecutor::new());
        let completion: &'static CompletionCell<u32> =
            Box::leak(Box::new(CompletionCell::<u32>::new()));
        unsafe { executor.init() };
        let handle = unsafe {
            executor
                .spawn_typed(PriorityClass::<level::Normal>::new(), completion, |_| {
                    PendingOnce { polled: false }
                })
                .unwrap()
        };
        let key = handle.executor_task_id();

        assert_eq!(unsafe { executor.test_poll_next() }, Some((key, false)));
        assert_eq!(unsafe { executor.test_poll_next() }, Some((key, true)));
        assert_eq!(unsafe { executor.completed_count() }, 1);
        drop(handle);
        assert!(unsafe { executor.is_quiescent_snapshot() });
    }

    #[test]
    fn normal_typed_join_releases_last_context_reference() {
        let executor = Box::new(GpuExecutor::new());
        let completion: &'static CompletionCell<u32> =
            Box::leak(Box::new(CompletionCell::<u32>::new()));
        unsafe { executor.init() };
        let mut handle = unsafe {
            executor
                .spawn_typed(PriorityClass::<level::High>::new(), completion, |_| async {
                    7
                })
                .unwrap()
        };
        assert_eq!(unsafe { executor.outstanding_context_refs() }, 2);
        assert_eq!(
            unsafe { executor.test_poll_next() },
            Some((handle.executor_task_id(), true))
        );
        // Completion dropped the task-owned token; the handle alone pins the
        // retired context until the successful join consumes it.
        assert_eq!(unsafe { executor.outstanding_context_refs() }, 1);
        assert_eq!(unsafe { executor.pending_reclaims() }, 1);

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        let mut wait = TypedWait {
            handle: &mut handle,
        };
        assert_eq!(
            unsafe { Pin::new_unchecked(&mut wait) }.poll(&mut cx),
            Poll::Ready(Ok(7))
        );
        assert_eq!(unsafe { executor.outstanding_context_refs() }, 0);
        assert_eq!(unsafe { executor.pending_reclaims() }, 0);
        assert!(unsafe { executor.is_quiescent_snapshot() });
    }

    #[test]
    fn typed_spawn_queue_full_rollback_releases_both_context_references() {
        let executor = Box::new(GpuExecutor::new());
        let completion: &'static CompletionCell<u32> =
            Box::leak(Box::new(CompletionCell::<u32>::new()));
        unsafe {
            executor.init();
            executor.test_fill_normal_queue();
        }
        let result = unsafe {
            executor.spawn_typed(
                PriorityClass::<level::Normal>::new(),
                completion,
                |_| async { 7 },
            )
        };
        assert!(matches!(result, Err(ExecutorError::QueueFull)));
        assert!(matches!(
            unsafe { completion.poll_result() },
            CompletionPoll::Cancelled
        ));
        assert_eq!(unsafe { executor.outstanding_context_refs() }, 0);
        assert_eq!(unsafe { executor.pending_reclaims() }, 0);
        assert!(unsafe { executor.is_quiescent_snapshot() });
    }
}
