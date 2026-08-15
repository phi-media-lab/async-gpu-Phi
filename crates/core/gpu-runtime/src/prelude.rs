// --- High-level hostcall API ---
pub use crate::hostcall::{
    gpu_hostcall_print, gpu_hostcall_release, gpu_hostcall_request, gpu_hostcall_trace,
};
pub use crate::panic::{gpu_panic_init, gpu_result_init};
pub use crate::print_buffer;
pub use crate::sideband::{gpu_bulk_read, gpu_bulk_write, sideband_alloc, sideband_reset};

// --- WarpFuture API ---
pub use crate::warp_future::{
    broadcast_u32, warp_hostcall_submit, warp_hostcall_wait_u64, WarpContext, WarpExecutor,
    WarpFuture, WarpPoll,
};

// --- Error types ---
pub use gpu_protocol::{GpuError, GpuKernelResult, TAG_ERR, TAG_OK, TAG_UNINIT};

// --- Commonly needed protocol constants ---
pub use gpu_protocol::{
    CONTROL_ERROR, CONTROL_FILLED, CONTROL_READY, FILE_ERROR_SENTINEL, FILE_MAX_PATH_LEN,
    FILE_MAX_READ_LEN, FILE_MAX_WRITE_LEN, FILE_OPEN_APPEND, FILE_OPEN_READ,
    FILE_OPEN_WRITE_CREATE, NULL_INDEX, PACKET_SIZE, PKT_OFF_ACTIVE_MASK, PKT_OFF_CONTROL,
    PKT_OFF_PAYLOAD, PKT_OFF_SERVICE, PRINT_MAX_MSG_LEN, SERVICE_BULK_PRINT, SERVICE_BULK_READ,
    SERVICE_BULK_WRITE, SERVICE_CLOSE, SERVICE_OPEN, SERVICE_PANIC, SERVICE_PRINT, SERVICE_READ,
    SERVICE_STDIN, SERVICE_TCP_BULK_READ, SERVICE_TCP_BULK_WRITE, SERVICE_TCP_CLOSE,
    SERVICE_TCP_CONNECT, SERVICE_TCP_READ, SERVICE_TCP_WRITE, SERVICE_TIME, SERVICE_TRACE,
    SERVICE_WRITE, TCP_MAX_ADDR_LEN, TCP_MAX_READ_LEN, TCP_MAX_WRITE_LEN, TRACE_LEVEL_DEBUG,
    TRACE_LEVEL_ERROR, TRACE_LEVEL_INFO, TRACE_LEVEL_WARN,
};

// --- Warp intrinsics ---
pub use gpu_atomics::{activemask, lane_id, shfl_sync_idx_u32, syncwarp};

// --- Command buffer polling ---
pub use crate::cmd::{cmd_ack, cmd_poll, cmd_yield};

// --- Command buffer constants ---
pub use gpu_protocol::{CMD_COMPUTE, CMD_EXIT, CMD_NOP, CMD_PRINT};

// --- Commonly needed atomics ---
pub use gpu_atomics::{sys_load_acquire_u32, sys_store_release_u32};

// --- Async executor + futures ---
pub use crate::std_future::{
    block_on, GpuBulkReadFuture, GpuBulkWriteFuture, GpuCloseFuture, GpuOpenFuture, GpuReadFuture,
    GpuTcpBulkReadFuture, GpuTcpBulkWriteFuture, GpuTcpCloseFuture, GpuTcpConnectFuture,
    GpuTcpReadFuture, GpuTcpWriteFuture, GpuWriteFuture,
};

// --- Sync primitives ---
pub use crate::sync::{Mutex, MutexGuard};

// --- Executor ---
pub use crate::executor::{
    ExecutorError, ExecutorStats, GpuExecutor, Priority, PriorityUpdate, TaskId, TaskKey,
    TaskKeyError, TaskOptions, HIGH_PRIORITY_RESERVED_SLOTS, NORMAL_PRIORITY_RESERVED_SLOTS,
};
pub use crate::priority::{
    CompletionCell, MayWaitFor, PriorityClass, PriorityLevel, PriorityToken, TaskContext,
    TaskContextError, TaskMetadata, TypedJoinError, TypedJoinHandle, TypedWait,
};

// --- Collections ---
pub use crate::collections::GpuHashMap;

// --- Parallel iterators ---
pub use crate::par_iter::{
    GpuFilter, GpuFilterMap, GpuMaxValue, GpuMinValue, GpuOne, GpuParIter, GpuParallelIterator,
    GpuSlice, GpuSliceMut, GpuZero, SendPtr, SendPtrMut,
};

// --- Stdio ---
pub use crate::stdio::{gpu_print_buffer_flush, stdio_init, stdio_print_buffer_init};

// --- Safety types ---
pub use crate::safety::{DisjointSlice, WarpHandle, WarpIndex};

// --- Tiered memory types ---
pub use crate::tiered_mem::{Global, GlobalRef, GpuRef, MemoryTier, Shared, SharedRef};

// --- Generator / coroutine ---
pub use crate::generator::{
    for_each_yield, CounterGenerator, FibGenerator, GeneratorTask, GpuGenerator, WarpBroadcast,
    WarpCoroutineState,
};

// --- GPU traits for generic algorithms ---
pub use crate::traits::{GpuReducible, GpuTransformable};

// --- Unified channels ---
pub use crate::unified_channel::{
    ScopedMpscReceiver, ScopedMpscSendError, ScopedMpscSender, ScopedOneshotClosed,
    ScopedOneshotReceiver, ScopedOneshotSender,
};
