use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use gpu_atomics::{activemask, sys_fetch_add_u64, sys_load_acquire_u32, sys_store_release_u32};
use gpu_protocol::*;

/// State machine for async hostcall print.
enum PrintState {
    /// Initial: need to allocate packet and submit request.
    Init,
    /// Packet submitted, waiting for host response.
    Waiting { pkt_idx: u16 },
    /// Completed.
    Done,
}

/// A `core::future::Future` that performs a hostcall print asynchronously.
///
/// On first poll: allocates a packet, fills the PRINT payload, submits
/// to the ready stack, and rings the doorbell. Returns `Poll::Pending`.
///
/// On subsequent polls: checks the control word for `CONTROL_READY`.
/// Returns `Poll::Ready(true)` on success, `Poll::Ready(false)` on error,
/// or `Poll::Pending` if the host hasn't responded yet.
///
/// This is a standard per-thread future — no warp cooperation.
pub struct GpuPrintFuture {
    buf: *mut u8,
    msg: *const u8,
    msg_len: u32,
    state: PrintState,
}

// SAFETY: On GPU, all threads access the same global memory.
// The future is only used by one thread at a time.
unsafe impl Send for GpuPrintFuture {}

impl GpuPrintFuture {
    /// Create a new GpuPrintFuture.
    ///
    /// `buf` is the hostcall buffer (mapped memory).
    /// `msg` is the message to print (max 56 bytes, truncated if longer).
    #[inline(always)]
    pub fn new(buf: *mut u8, msg: &[u8]) -> Self {
        Self {
            buf,
            msg: msg.as_ptr(),
            msg_len: msg.len() as u32,
            state: PrintState::Init,
        }
    }
}

impl Future for GpuPrintFuture {
    type Output = bool;

    #[inline(always)]
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<bool> {
        // SAFETY: This future has no self-referential fields (all fields are
        // raw pointers, integers, or a plain enum), so unpinning is safe.
        // The future is never moved after first poll — the executor (block_on
        // / SpinExecutor) pins it in place and polls via Pin::as_mut().
        //
        // All subsequent GPU futures (GpuOpenFuture, GpuReadFuture, etc.)
        // use the same get_unchecked_mut() pattern with identical structural
        // pinning justification and are not re-documented individually.
        let this = unsafe { self.get_unchecked_mut() };

        match this.state {
            PrintState::Init => unsafe {
                // Pop a free packet using the hostcall module's helpers
                let (num_shards, shard_array_off, _) =
                    crate::hostcall::read_shard_info(this.buf as *const u8);
                let free_ptr =
                    crate::hostcall::get_free_stack_ptr(this.buf, num_shards, shard_array_off);

                let pkt_idx = crate::hostcall::hc_pop_free_from(
                    this.buf,
                    free_ptr,
                    num_shards,
                    shard_array_off,
                );
                if pkt_idx == NULL_INDEX {
                    // Pool exhausted — backpressure, retry on next poll
                    return Poll::Pending;
                }

                let pkt_off = crate::hostcall::pkt_offset(this.buf as *const u8, pkt_idx);
                let pkt = this.buf.add(pkt_off);

                // Fill packet header
                let mask = activemask();
                core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, mask);
                core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, SERVICE_PRINT);
                sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

                // Fill payload: slot 0 = message length, then message bytes
                let payload = pkt.add(PKT_OFF_PAYLOAD);
                core::ptr::write_volatile(payload as *mut u64, this.msg_len as u64);

                let copy_len = if this.msg_len > PRINT_MAX_MSG_LEN as u32 {
                    PRINT_MAX_MSG_LEN as u32
                } else {
                    this.msg_len
                };
                let dst = payload.add(8);
                let mut i: u32 = 0;
                while i < copy_len {
                    core::ptr::write_volatile(dst.add(i as usize), *this.msg.add(i as usize));
                    i += 1;
                }

                // Thread/block metadata
                let block_idx = crate::nvptx_shim::block_idx_x();
                let thread_idx = crate::nvptx_shim::thread_idx_x();
                core::ptr::write_volatile(payload.add(64) as *mut u32, block_idx);
                core::ptr::write_volatile(payload.add(68) as *mut u32, thread_idx);

                // Mark filled, push to ready stack, ring doorbell
                sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);

                let ready_ptr =
                    crate::hostcall::get_ready_stack_ptr(this.buf, num_shards, shard_array_off);
                crate::hostcall::hc_push(ready_ptr, this.buf, pkt_idx);
                sys_fetch_add_u64(this.buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

                this.state = PrintState::Waiting { pkt_idx };
                Poll::Pending
            },

            PrintState::Waiting { pkt_idx } => unsafe {
                let pkt_off = crate::hostcall::pkt_offset(this.buf as *const u8, pkt_idx);
                let pkt = this.buf.add(pkt_off);
                let ctrl = sys_load_acquire_u32(pkt.add(PKT_OFF_CONTROL) as *const u32);

                if ctrl & CONTROL_READY != 0 {
                    let success = (ctrl & CONTROL_ERROR) == 0;
                    // Release packet back to free stack
                    crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                    this.state = PrintState::Done;
                    Poll::Ready(success)
                } else {
                    Poll::Pending
                }
            },

            PrintState::Done => Poll::Ready(false),
        }
    }
}

/// Wrapper around `GpuPrintFuture` that returns `Result<bool, u32>`.
///
/// Maps `true` → `Ok(true)`, `false` → `Err(1)` (print failure).
/// Used for testing warp-cooperative `?` operator broadcasting.
pub struct GpuPrintResultFuture {
    inner: GpuPrintFuture,
}

impl GpuPrintResultFuture {
    /// Create a new Result-returning print future.
    #[inline(always)]
    pub fn new(buf: *mut u8, msg: &[u8]) -> Self {
        Self {
            inner: GpuPrintFuture::new(buf, msg),
        }
    }
}

impl Future for GpuPrintResultFuture {
    type Output = Result<bool, u32>;

    #[inline(always)]
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<bool, u32>> {
        // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
        let inner = unsafe { &mut self.get_unchecked_mut().inner };
        match unsafe { Pin::new_unchecked(inner) }.poll(cx) {
            Poll::Ready(true) => Poll::Ready(Ok(true)),
            Poll::Ready(false) => Poll::Ready(Err(1)), // print failure → error
            Poll::Pending => Poll::Pending,
        }
    }
}

// ================================================================
// Async I/O Futures — generic hostcall futures for file operations
// ================================================================

/// Internal state for all hostcall futures.
enum HostcallState {
    /// Initial: need to allocate packet and submit request.
    Init,
    /// Packet submitted, waiting for host response.
    Waiting { pkt_idx: u16 },
    /// Completed.
    Done,
}

/// Submit a hostcall packet and transition to Waiting state.
///
/// Returns `Poll::Pending` on success, `Poll::Pending` on pool exhaustion (retry),
/// or the new state.
#[inline(always)]
unsafe fn submit_hostcall(
    buf: *mut u8,
    service: u32,
    fill_payload: impl FnOnce(*mut u8),
) -> Result<u16, ()> {
    let (num_shards, shard_array_off, _) = crate::hostcall::read_shard_info(buf as *const u8);
    let free_ptr = crate::hostcall::get_free_stack_ptr(buf, num_shards, shard_array_off);

    let pkt_idx = crate::hostcall::hc_pop_free_from(buf, free_ptr, num_shards, shard_array_off);
    if pkt_idx == NULL_INDEX {
        return Err(()); // Pool exhausted — retry on next poll
    }

    let pkt_off = crate::hostcall::pkt_offset(buf as *const u8, pkt_idx);
    let pkt = buf.add(pkt_off);

    // Fill header
    let mask = activemask();
    core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, mask);
    core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, service);
    sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);

    // Fill payload
    fill_payload(pkt.add(PKT_OFF_PAYLOAD));

    // Mark filled, push to ready stack, ring doorbell
    sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);
    let ready_ptr = crate::hostcall::get_ready_stack_ptr(buf, num_shards, shard_array_off);
    crate::hostcall::hc_push(ready_ptr, buf, pkt_idx);
    sys_fetch_add_u64(buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

    Ok(pkt_idx)
}

/// Check if a hostcall packet has a response ready.
/// Returns `Some(pkt_ptr)` if ready, `None` if still pending.
#[inline(always)]
unsafe fn check_response(buf: *mut u8, pkt_idx: u16) -> Option<*mut u8> {
    let pkt_off = crate::hostcall::pkt_offset(buf as *const u8, pkt_idx);
    let pkt = buf.add(pkt_off);
    let ctrl = sys_load_acquire_u32(pkt.add(PKT_OFF_CONTROL) as *const u32);
    if ctrl & CONTROL_READY != 0 {
        Some(pkt)
    } else {
        None
    }
}

/// A `Future` that opens a file via hostcall.
///
/// On first poll: submits SERVICE_OPEN request with path and flags.
/// On subsequent polls: checks for response.
/// Returns `Ok(fd)` on success, `Err(errno)` on failure.
pub struct GpuOpenFuture {
    buf: *mut u8,
    path: *const u8,
    path_len: u32,
    flags: u32,
    state: HostcallState,
}

unsafe impl Send for GpuOpenFuture {}

impl GpuOpenFuture {
    /// Create a new open future.
    ///
    /// `flags` uses gpu-protocol FILE_OPEN_* constants.
    #[inline(always)]
    pub fn new(buf: *mut u8, path: &[u8], flags: u32) -> Self {
        Self {
            buf,
            path: path.as_ptr(),
            path_len: path.len() as u32,
            flags,
            state: HostcallState::Init,
        }
    }
}

impl Future for GpuOpenFuture {
    type Output = Result<i32, i32>;

    #[inline(always)]
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
        let this = unsafe { self.get_unchecked_mut() };
        match this.state {
            HostcallState::Init => {
                let path_ptr = this.path;
                let path_len = this.path_len;
                let flags = this.flags;
                match unsafe {
                    submit_hostcall(this.buf, SERVICE_OPEN, |payload| {
                        let slot0 = (path_len as u64) | ((flags as u64) << 32);
                        core::ptr::write_volatile(payload as *mut u64, slot0);
                        let dst = payload.add(8);
                        let mut i: u32 = 0;
                        while i < path_len && i < FILE_MAX_PATH_LEN as u32 {
                            core::ptr::write_volatile(
                                dst.add(i as usize),
                                *path_ptr.add(i as usize),
                            );
                            i += 1;
                        }
                    })
                } {
                    Ok(idx) => {
                        this.state = HostcallState::Waiting { pkt_idx: idx };
                        Poll::Pending
                    }
                    Err(()) => Poll::Pending, // pool exhausted, retry
                }
            }
            HostcallState::Waiting { pkt_idx } => unsafe {
                match check_response(this.buf, pkt_idx) {
                    Some(pkt) => {
                        let ctrl = core::ptr::read_volatile(pkt.add(PKT_OFF_CONTROL) as *const u32);
                        let fd = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                        crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                        this.state = HostcallState::Done;
                        if ctrl & CONTROL_ERROR != 0 {
                            Poll::Ready(Err(fd as i32))
                        } else {
                            Poll::Ready(Ok(fd as i32))
                        }
                    }
                    None => Poll::Pending,
                }
            },
            HostcallState::Done => Poll::Ready(Err(-1)),
        }
    }
}

/// A `Future` that writes data to a file descriptor via hostcall.
///
/// Returns `Ok(bytes_written)` on success, `Err(errno)` on failure.
pub struct GpuWriteFuture {
    buf: *mut u8,
    fd: i32,
    data: *const u8,
    data_len: u32,
    state: HostcallState,
}

unsafe impl Send for GpuWriteFuture {}

impl GpuWriteFuture {
    #[inline(always)]
    pub fn new(buf: *mut u8, fd: i32, data: &[u8]) -> Self {
        Self {
            buf,
            fd,
            data: data.as_ptr(),
            data_len: data.len() as u32,
            state: HostcallState::Init,
        }
    }
}

impl Future for GpuWriteFuture {
    type Output = Result<usize, i32>;

    #[inline(always)]
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
        let this = unsafe { self.get_unchecked_mut() };
        match this.state {
            HostcallState::Init => {
                let fd = this.fd;
                let data_ptr = this.data;
                let write_len = if this.data_len as usize > FILE_MAX_WRITE_LEN {
                    FILE_MAX_WRITE_LEN as u32
                } else {
                    this.data_len
                };
                match unsafe {
                    submit_hostcall(this.buf, SERVICE_WRITE, |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd as u64);
                        core::ptr::write_volatile(payload.add(8) as *mut u64, write_len as u64);
                        let dst = payload.add(16);
                        let mut i: u32 = 0;
                        while i < write_len {
                            core::ptr::write_volatile(
                                dst.add(i as usize),
                                *data_ptr.add(i as usize),
                            );
                            i += 1;
                        }
                    })
                } {
                    Ok(idx) => {
                        this.state = HostcallState::Waiting { pkt_idx: idx };
                        Poll::Pending
                    }
                    Err(()) => Poll::Pending,
                }
            }
            HostcallState::Waiting { pkt_idx } => unsafe {
                match check_response(this.buf, pkt_idx) {
                    Some(pkt) => {
                        let ctrl = core::ptr::read_volatile(pkt.add(PKT_OFF_CONTROL) as *const u32);
                        let written =
                            core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                        crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                        this.state = HostcallState::Done;
                        if ctrl & CONTROL_ERROR != 0 {
                            Poll::Ready(Err(written as i32))
                        } else {
                            Poll::Ready(Ok(written as usize))
                        }
                    }
                    None => Poll::Pending,
                }
            },
            HostcallState::Done => Poll::Ready(Err(-1)),
        }
    }
}

/// A `Future` that reads data from a file descriptor via hostcall.
///
/// Returns `Ok((bytes_read, data))` on success. The data is copied into
/// the caller-provided buffer.
pub struct GpuReadFuture {
    buf: *mut u8,
    fd: i32,
    out_buf: *mut u8,
    max_len: u32,
    state: HostcallState,
}

unsafe impl Send for GpuReadFuture {}

impl GpuReadFuture {
    #[inline(always)]
    pub fn new(buf: *mut u8, fd: i32, out_buf: &mut [u8]) -> Self {
        Self {
            buf,
            fd,
            out_buf: out_buf.as_mut_ptr(),
            max_len: out_buf.len() as u32,
            state: HostcallState::Init,
        }
    }
}

impl Future for GpuReadFuture {
    type Output = Result<usize, i32>;

    #[inline(always)]
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
        let this = unsafe { self.get_unchecked_mut() };
        match this.state {
            HostcallState::Init => {
                let fd = this.fd;
                let max_len = if this.max_len as usize > FILE_MAX_READ_LEN {
                    FILE_MAX_READ_LEN as u32
                } else {
                    this.max_len
                };
                match unsafe {
                    submit_hostcall(this.buf, SERVICE_READ, |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd as u64);
                        core::ptr::write_volatile(payload.add(8) as *mut u64, max_len as u64);
                    })
                } {
                    Ok(idx) => {
                        this.state = HostcallState::Waiting { pkt_idx: idx };
                        Poll::Pending
                    }
                    Err(()) => Poll::Pending,
                }
            }
            HostcallState::Waiting { pkt_idx } => unsafe {
                match check_response(this.buf, pkt_idx) {
                    Some(pkt) => {
                        let ctrl = core::ptr::read_volatile(pkt.add(PKT_OFF_CONTROL) as *const u32);
                        let bytes_read =
                            core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                        if ctrl & CONTROL_ERROR != 0 {
                            crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                            this.state = HostcallState::Done;
                            Poll::Ready(Err(bytes_read as i32))
                        } else {
                            // Copy data from payload to output buffer
                            let src = pkt.add(PKT_OFF_PAYLOAD).add(8);
                            let n = core::cmp::min(bytes_read as usize, this.max_len as usize);
                            let mut i = 0;
                            while i < n {
                                core::ptr::write_volatile(
                                    this.out_buf.add(i),
                                    core::ptr::read_volatile(src.add(i)),
                                );
                                i += 1;
                            }
                            crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                            this.state = HostcallState::Done;
                            Poll::Ready(Ok(n))
                        }
                    }
                    None => Poll::Pending,
                }
            },
            HostcallState::Done => Poll::Ready(Err(-1)),
        }
    }
}

/// A `Future` that closes a file descriptor via hostcall.
///
/// Returns `Ok(())` on success, `Err(errno)` on failure.
pub struct GpuCloseFuture {
    buf: *mut u8,
    fd: i32,
    state: HostcallState,
}

unsafe impl Send for GpuCloseFuture {}

impl GpuCloseFuture {
    #[inline(always)]
    pub fn new(buf: *mut u8, fd: i32) -> Self {
        Self {
            buf,
            fd,
            state: HostcallState::Init,
        }
    }
}

impl Future for GpuCloseFuture {
    type Output = Result<(), i32>;

    #[inline(always)]
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
        let this = unsafe { self.get_unchecked_mut() };
        match this.state {
            HostcallState::Init => {
                let fd = this.fd;
                match unsafe {
                    submit_hostcall(this.buf, SERVICE_CLOSE, |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd as u64);
                    })
                } {
                    Ok(idx) => {
                        this.state = HostcallState::Waiting { pkt_idx: idx };
                        Poll::Pending
                    }
                    Err(()) => Poll::Pending,
                }
            }
            HostcallState::Waiting { pkt_idx } => unsafe {
                match check_response(this.buf, pkt_idx) {
                    Some(pkt) => {
                        let ctrl = core::ptr::read_volatile(pkt.add(PKT_OFF_CONTROL) as *const u32);
                        crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                        this.state = HostcallState::Done;
                        if ctrl & CONTROL_ERROR != 0 {
                            Poll::Ready(Err(-1))
                        } else {
                            Poll::Ready(Ok(()))
                        }
                    }
                    None => Poll::Pending,
                }
            },
            HostcallState::Done => Poll::Ready(Err(-1)),
        }
    }
}

// ================================================================
// Async Bulk I/O Futures — sideband-based large data transfers
// ================================================================

/// A `Future` that writes data to a file via sideband bulk transfer.
///
/// On first poll: allocates sideband space, copies data, submits SERVICE_BULK_WRITE.
/// On subsequent polls: checks for response.
/// Returns `Ok(bytes_written)` on success, `Err(-1)` on failure.
pub struct GpuBulkWriteFuture {
    buf: *mut u8,
    sideband: *mut u8,
    fd: u64,
    src: *const u8,
    len: usize,
    sideband_offset: u64,
    state: HostcallState,
}

unsafe impl Send for GpuBulkWriteFuture {}

impl GpuBulkWriteFuture {
    /// Create a new bulk write future.
    ///
    /// `buf` is the hostcall buffer, `sideband` is the sideband buffer,
    /// `fd` is the file descriptor, `src`/`len` describe the data to write.
    #[inline(always)]
    pub fn new(buf: *mut u8, sideband: *mut u8, fd: u64, src: *const u8, len: usize) -> Self {
        Self {
            buf,
            sideband,
            fd,
            src,
            len,
            sideband_offset: 0,
            state: HostcallState::Init,
        }
    }
}

impl Future for GpuBulkWriteFuture {
    type Output = Result<usize, i32>;

    #[inline(always)]
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
        let this = unsafe { self.get_unchecked_mut() };
        match this.state {
            HostcallState::Init => {
                if this.len == 0 {
                    this.state = HostcallState::Done;
                    return Poll::Ready(Ok(0));
                }

                // Allocate sideband space
                let offset =
                    unsafe { crate::sideband::sideband_alloc(this.sideband, this.len as u64) };
                if offset == u64::MAX {
                    return Poll::Pending; // Retry on next poll
                }
                this.sideband_offset = offset;

                // Copy data to sideband
                unsafe {
                    let dst = this.sideband.add(SIDEBAND_DATA_OFFSET + offset as usize);
                    let mut i = 0;
                    while i < this.len {
                        core::ptr::write_volatile(dst.add(i), *this.src.add(i));
                        i += 1;
                    }
                }

                // Submit hostcall
                let fd = this.fd;
                let sb_offset = this.sideband_offset;
                let len = this.len;
                match unsafe {
                    submit_hostcall(this.buf, SERVICE_BULK_WRITE, |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                        core::ptr::write_volatile(payload.add(8) as *mut u64, sb_offset);
                        core::ptr::write_volatile(payload.add(16) as *mut u64, len as u64);
                    })
                } {
                    Ok(idx) => {
                        this.state = HostcallState::Waiting { pkt_idx: idx };
                        Poll::Pending
                    }
                    Err(()) => Poll::Pending,
                }
            }
            HostcallState::Waiting { pkt_idx } => unsafe {
                match check_response(this.buf, pkt_idx) {
                    Some(pkt) => {
                        let written =
                            core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                        crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                        this.state = HostcallState::Done;
                        if written == FILE_ERROR_SENTINEL {
                            Poll::Ready(Err(-1))
                        } else {
                            Poll::Ready(Ok(written as usize))
                        }
                    }
                    None => Poll::Pending,
                }
            },
            HostcallState::Done => Poll::Ready(Err(-1)),
        }
    }
}

/// A `Future` that reads data from a file via sideband bulk transfer.
///
/// On first poll: allocates sideband space, submits SERVICE_BULK_READ.
/// On subsequent polls: checks for response, copies data from sideband.
/// Returns `Ok(bytes_read)` on success, `Err(-1)` on failure.
pub struct GpuBulkReadFuture {
    buf: *mut u8,
    sideband: *mut u8,
    fd: u64,
    dst: *mut u8,
    max_len: usize,
    sideband_offset: u64,
    state: HostcallState,
}

unsafe impl Send for GpuBulkReadFuture {}

impl GpuBulkReadFuture {
    /// Create a new bulk read future.
    ///
    /// `buf` is the hostcall buffer, `sideband` is the sideband buffer,
    /// `fd` is the file descriptor, `dst`/`max_len` describe the output buffer.
    #[inline(always)]
    pub fn new(buf: *mut u8, sideband: *mut u8, fd: u64, dst: *mut u8, max_len: usize) -> Self {
        Self {
            buf,
            sideband,
            fd,
            dst,
            max_len,
            sideband_offset: 0,
            state: HostcallState::Init,
        }
    }
}

impl Future for GpuBulkReadFuture {
    type Output = Result<usize, i32>;

    #[inline(always)]
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
        let this = unsafe { self.get_unchecked_mut() };
        match this.state {
            HostcallState::Init => {
                if this.max_len == 0 {
                    this.state = HostcallState::Done;
                    return Poll::Ready(Ok(0));
                }

                // Allocate sideband space for response data
                let offset =
                    unsafe { crate::sideband::sideband_alloc(this.sideband, this.max_len as u64) };
                if offset == u64::MAX {
                    return Poll::Pending; // Retry on next poll
                }
                this.sideband_offset = offset;

                // Submit hostcall
                let fd = this.fd;
                let sb_offset = this.sideband_offset;
                let max_len = this.max_len;
                match unsafe {
                    submit_hostcall(this.buf, SERVICE_BULK_READ, |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                        core::ptr::write_volatile(payload.add(8) as *mut u64, sb_offset);
                        core::ptr::write_volatile(payload.add(16) as *mut u64, max_len as u64);
                    })
                } {
                    Ok(idx) => {
                        this.state = HostcallState::Waiting { pkt_idx: idx };
                        Poll::Pending
                    }
                    Err(()) => Poll::Pending,
                }
            }
            HostcallState::Waiting { pkt_idx } => unsafe {
                match check_response(this.buf, pkt_idx) {
                    Some(pkt) => {
                        let bytes_read =
                            core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                        crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                        this.state = HostcallState::Done;

                        if bytes_read == FILE_ERROR_SENTINEL || bytes_read == 0 {
                            if bytes_read == 0 {
                                return Poll::Ready(Ok(0)); // EOF
                            }
                            return Poll::Ready(Err(-1));
                        }

                        // Copy data from sideband to destination
                        let src = this
                            .sideband
                            .add(SIDEBAND_DATA_OFFSET + this.sideband_offset as usize);
                        let n = bytes_read as usize;
                        let mut i = 0;
                        while i < n {
                            core::ptr::write_volatile(
                                this.dst.add(i),
                                core::ptr::read_volatile(src.add(i)),
                            );
                            i += 1;
                        }

                        Poll::Ready(Ok(n))
                    }
                    None => Poll::Pending,
                }
            },
            HostcallState::Done => Poll::Ready(Err(-1)),
        }
    }
}

// ================================================================
// Async TCP Futures — inline data and sideband-based transfers
// ================================================================

/// A `Future` that connects to a remote TCP address:port via hostcall.
///
/// On first poll: submits SERVICE_TCP_CONNECT with address and port.
/// On subsequent polls: checks for response.
/// Returns `Ok(fd)` on success, `Err(errno)` on failure.
pub struct GpuTcpConnectFuture {
    buf: *mut u8,
    addr: *const u8,
    addr_len: u32,
    port: u32,
    state: HostcallState,
}

// SAFETY: On GPU, all threads access the same global memory.
// The future is only used by one thread at a time.
unsafe impl Send for GpuTcpConnectFuture {}

impl GpuTcpConnectFuture {
    /// Create a new TCP connect future.
    ///
    /// `addr` is the address string (e.g. "127.0.0.1"), max 56 bytes.
    /// `port` is the TCP port number.
    #[inline(always)]
    pub fn new(buf: *mut u8, addr: &[u8], port: u32) -> Self {
        Self {
            buf,
            addr: addr.as_ptr(),
            addr_len: addr.len() as u32,
            port,
            state: HostcallState::Init,
        }
    }
}

impl Future for GpuTcpConnectFuture {
    type Output = Result<u64, i32>;

    #[inline(always)]
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
        let this = unsafe { self.get_unchecked_mut() };
        match this.state {
            HostcallState::Init => {
                let addr_ptr = this.addr;
                let addr_len = this.addr_len;
                let port = this.port;
                match unsafe {
                    submit_hostcall(this.buf, SERVICE_TCP_CONNECT, |payload| {
                        // slot0 = port (low 32) | addr_len (high 32)
                        let slot0 = (port as u64) | ((addr_len as u64) << 32);
                        core::ptr::write_volatile(payload as *mut u64, slot0);
                        // slots 1-7: address string (max 56 bytes)
                        let dst = payload.add(8);
                        let mut i: u32 = 0;
                        while i < addr_len && i < TCP_MAX_ADDR_LEN as u32 {
                            core::ptr::write_volatile(
                                dst.add(i as usize),
                                *addr_ptr.add(i as usize),
                            );
                            i += 1;
                        }
                    })
                } {
                    Ok(idx) => {
                        this.state = HostcallState::Waiting { pkt_idx: idx };
                        Poll::Pending
                    }
                    Err(()) => Poll::Pending, // pool exhausted, retry
                }
            }
            HostcallState::Waiting { pkt_idx } => unsafe {
                match check_response(this.buf, pkt_idx) {
                    Some(pkt) => {
                        let ctrl = core::ptr::read_volatile(pkt.add(PKT_OFF_CONTROL) as *const u32);
                        let fd = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                        crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                        this.state = HostcallState::Done;
                        if ctrl & CONTROL_ERROR != 0 {
                            Poll::Ready(Err(fd as i32))
                        } else {
                            Poll::Ready(Ok(fd))
                        }
                    }
                    None => Poll::Pending,
                }
            },
            HostcallState::Done => Poll::Ready(Err(-1)),
        }
    }
}

/// A `Future` that writes inline data to a TCP socket via hostcall.
///
/// Returns `Ok(bytes_written)` on success, `Err(errno)` on failure.
/// Maximum inline write is 48 bytes (6 slots).
pub struct GpuTcpWriteFuture {
    buf: *mut u8,
    fd: u64,
    src: *const u8,
    len: u32,
    state: HostcallState,
}

unsafe impl Send for GpuTcpWriteFuture {}

impl GpuTcpWriteFuture {
    #[inline(always)]
    pub fn new(buf: *mut u8, fd: u64, data: &[u8]) -> Self {
        Self {
            buf,
            fd,
            src: data.as_ptr(),
            len: data.len() as u32,
            state: HostcallState::Init,
        }
    }
}

impl Future for GpuTcpWriteFuture {
    type Output = Result<usize, i32>;

    #[inline(always)]
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
        let this = unsafe { self.get_unchecked_mut() };
        match this.state {
            HostcallState::Init => {
                let fd = this.fd;
                let src_ptr = this.src;
                let write_len = if this.len as usize > TCP_MAX_WRITE_LEN {
                    TCP_MAX_WRITE_LEN as u32
                } else {
                    this.len
                };
                match unsafe {
                    submit_hostcall(this.buf, SERVICE_TCP_WRITE, |payload| {
                        // slot0 = fd, slot1 = len
                        core::ptr::write_volatile(payload as *mut u64, fd);
                        core::ptr::write_volatile(payload.add(8) as *mut u64, write_len as u64);
                        // slots 2-7: data bytes (max 48 bytes)
                        let dst = payload.add(16);
                        let mut i: u32 = 0;
                        while i < write_len {
                            core::ptr::write_volatile(
                                dst.add(i as usize),
                                *src_ptr.add(i as usize),
                            );
                            i += 1;
                        }
                    })
                } {
                    Ok(idx) => {
                        this.state = HostcallState::Waiting { pkt_idx: idx };
                        Poll::Pending
                    }
                    Err(()) => Poll::Pending,
                }
            }
            HostcallState::Waiting { pkt_idx } => unsafe {
                match check_response(this.buf, pkt_idx) {
                    Some(pkt) => {
                        let ctrl = core::ptr::read_volatile(pkt.add(PKT_OFF_CONTROL) as *const u32);
                        let written =
                            core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                        crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                        this.state = HostcallState::Done;
                        if ctrl & CONTROL_ERROR != 0 {
                            Poll::Ready(Err(written as i32))
                        } else {
                            Poll::Ready(Ok(written as usize))
                        }
                    }
                    None => Poll::Pending,
                }
            },
            HostcallState::Done => Poll::Ready(Err(-1)),
        }
    }
}

/// A `Future` that reads inline data from a TCP socket via hostcall.
///
/// Returns `Ok(bytes_read)` on success. Data is copied into the
/// caller-provided buffer. Maximum inline read is 56 bytes (7 slots).
pub struct GpuTcpReadFuture {
    buf: *mut u8,
    fd: u64,
    dst: *mut u8,
    max_len: u32,
    state: HostcallState,
}

unsafe impl Send for GpuTcpReadFuture {}

impl GpuTcpReadFuture {
    #[inline(always)]
    pub fn new(buf: *mut u8, fd: u64, out_buf: &mut [u8]) -> Self {
        Self {
            buf,
            fd,
            dst: out_buf.as_mut_ptr(),
            max_len: out_buf.len() as u32,
            state: HostcallState::Init,
        }
    }
}

impl Future for GpuTcpReadFuture {
    type Output = Result<usize, i32>;

    #[inline(always)]
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
        let this = unsafe { self.get_unchecked_mut() };
        match this.state {
            HostcallState::Init => {
                let fd = this.fd;
                let max_len = if this.max_len as usize > TCP_MAX_READ_LEN {
                    TCP_MAX_READ_LEN as u32
                } else {
                    this.max_len
                };
                match unsafe {
                    submit_hostcall(this.buf, SERVICE_TCP_READ, |payload| {
                        // slot0 = fd, slot1 = max_len
                        core::ptr::write_volatile(payload as *mut u64, fd);
                        core::ptr::write_volatile(payload.add(8) as *mut u64, max_len as u64);
                    })
                } {
                    Ok(idx) => {
                        this.state = HostcallState::Waiting { pkt_idx: idx };
                        Poll::Pending
                    }
                    Err(()) => Poll::Pending,
                }
            }
            HostcallState::Waiting { pkt_idx } => unsafe {
                match check_response(this.buf, pkt_idx) {
                    Some(pkt) => {
                        let ctrl = core::ptr::read_volatile(pkt.add(PKT_OFF_CONTROL) as *const u32);
                        let bytes_read =
                            core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                        if ctrl & CONTROL_ERROR != 0 {
                            crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                            this.state = HostcallState::Done;
                            Poll::Ready(Err(bytes_read as i32))
                        } else {
                            // Copy data from payload slots 1-7 to destination buffer
                            let src = pkt.add(PKT_OFF_PAYLOAD).add(8);
                            let n = core::cmp::min(bytes_read as usize, this.max_len as usize);
                            let mut i = 0;
                            while i < n {
                                core::ptr::write_volatile(
                                    this.dst.add(i),
                                    core::ptr::read_volatile(src.add(i)),
                                );
                                i += 1;
                            }
                            crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                            this.state = HostcallState::Done;
                            Poll::Ready(Ok(n))
                        }
                    }
                    None => Poll::Pending,
                }
            },
            HostcallState::Done => Poll::Ready(Err(-1)),
        }
    }
}

/// A `Future` that closes a TCP socket via hostcall.
///
/// Returns `Ok(())` on success, `Err(errno)` on failure.
pub struct GpuTcpCloseFuture {
    buf: *mut u8,
    fd: u64,
    state: HostcallState,
}

unsafe impl Send for GpuTcpCloseFuture {}

impl GpuTcpCloseFuture {
    #[inline(always)]
    pub fn new(buf: *mut u8, fd: u64) -> Self {
        Self {
            buf,
            fd,
            state: HostcallState::Init,
        }
    }
}

impl Future for GpuTcpCloseFuture {
    type Output = Result<(), i32>;

    #[inline(always)]
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
        let this = unsafe { self.get_unchecked_mut() };
        match this.state {
            HostcallState::Init => {
                let fd = this.fd;
                match unsafe {
                    submit_hostcall(this.buf, SERVICE_TCP_CLOSE, |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                    })
                } {
                    Ok(idx) => {
                        this.state = HostcallState::Waiting { pkt_idx: idx };
                        Poll::Pending
                    }
                    Err(()) => Poll::Pending,
                }
            }
            HostcallState::Waiting { pkt_idx } => unsafe {
                match check_response(this.buf, pkt_idx) {
                    Some(pkt) => {
                        let ctrl = core::ptr::read_volatile(pkt.add(PKT_OFF_CONTROL) as *const u32);
                        crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                        this.state = HostcallState::Done;
                        if ctrl & CONTROL_ERROR != 0 {
                            Poll::Ready(Err(-1))
                        } else {
                            Poll::Ready(Ok(()))
                        }
                    }
                    None => Poll::Pending,
                }
            },
            HostcallState::Done => Poll::Ready(Err(-1)),
        }
    }
}

/// A `Future` that writes data to a TCP socket via sideband bulk transfer.
///
/// On first poll: allocates sideband space, copies data, submits SERVICE_TCP_BULK_WRITE.
/// On subsequent polls: checks for response.
/// Returns `Ok(bytes_written)` on success, `Err(-1)` on failure.
pub struct GpuTcpBulkWriteFuture {
    buf: *mut u8,
    sideband: *mut u8,
    fd: u64,
    src: *const u8,
    len: usize,
    sideband_offset: u64,
    state: HostcallState,
}

unsafe impl Send for GpuTcpBulkWriteFuture {}

impl GpuTcpBulkWriteFuture {
    /// Create a new TCP bulk write future.
    ///
    /// `buf` is the hostcall buffer, `sideband` is the sideband buffer,
    /// `fd` is the socket fd, `src`/`len` describe the data to write.
    #[inline(always)]
    pub fn new(buf: *mut u8, sideband: *mut u8, fd: u64, src: *const u8, len: usize) -> Self {
        Self {
            buf,
            sideband,
            fd,
            src,
            len,
            sideband_offset: 0,
            state: HostcallState::Init,
        }
    }
}

impl Future for GpuTcpBulkWriteFuture {
    type Output = Result<usize, i32>;

    #[inline(always)]
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
        let this = unsafe { self.get_unchecked_mut() };
        match this.state {
            HostcallState::Init => {
                if this.len == 0 {
                    this.state = HostcallState::Done;
                    return Poll::Ready(Ok(0));
                }

                // Allocate sideband space
                let offset =
                    unsafe { crate::sideband::sideband_alloc(this.sideband, this.len as u64) };
                if offset == u64::MAX {
                    return Poll::Pending; // Retry on next poll
                }
                this.sideband_offset = offset;

                // Copy data to sideband
                unsafe {
                    let dst = this.sideband.add(SIDEBAND_DATA_OFFSET + offset as usize);
                    let mut i = 0;
                    while i < this.len {
                        core::ptr::write_volatile(dst.add(i), *this.src.add(i));
                        i += 1;
                    }
                }

                // Submit hostcall
                let fd = this.fd;
                let sb_offset = this.sideband_offset;
                let len = this.len;
                match unsafe {
                    submit_hostcall(this.buf, SERVICE_TCP_BULK_WRITE, |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                        core::ptr::write_volatile(payload.add(8) as *mut u64, sb_offset);
                        core::ptr::write_volatile(payload.add(16) as *mut u64, len as u64);
                    })
                } {
                    Ok(idx) => {
                        this.state = HostcallState::Waiting { pkt_idx: idx };
                        Poll::Pending
                    }
                    Err(()) => Poll::Pending,
                }
            }
            HostcallState::Waiting { pkt_idx } => unsafe {
                match check_response(this.buf, pkt_idx) {
                    Some(pkt) => {
                        let written =
                            core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                        crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                        this.state = HostcallState::Done;
                        if written == FILE_ERROR_SENTINEL {
                            Poll::Ready(Err(-1))
                        } else {
                            Poll::Ready(Ok(written as usize))
                        }
                    }
                    None => Poll::Pending,
                }
            },
            HostcallState::Done => Poll::Ready(Err(-1)),
        }
    }
}

/// A `Future` that reads data from a TCP socket via sideband bulk transfer.
///
/// On first poll: allocates sideband space, submits SERVICE_TCP_BULK_READ.
/// On subsequent polls: checks for response, copies data from sideband.
/// Returns `Ok(bytes_read)` on success, `Err(-1)` on failure.
pub struct GpuTcpBulkReadFuture {
    buf: *mut u8,
    sideband: *mut u8,
    fd: u64,
    dst: *mut u8,
    max_len: usize,
    sideband_offset: u64,
    state: HostcallState,
}

unsafe impl Send for GpuTcpBulkReadFuture {}

impl GpuTcpBulkReadFuture {
    /// Create a new TCP bulk read future.
    ///
    /// `buf` is the hostcall buffer, `sideband` is the sideband buffer,
    /// `fd` is the socket fd, `dst`/`max_len` describe the output buffer.
    #[inline(always)]
    pub fn new(buf: *mut u8, sideband: *mut u8, fd: u64, dst: *mut u8, max_len: usize) -> Self {
        Self {
            buf,
            sideband,
            fd,
            dst,
            max_len,
            sideband_offset: 0,
            state: HostcallState::Init,
        }
    }
}

impl Future for GpuTcpBulkReadFuture {
    type Output = Result<usize, i32>;

    #[inline(always)]
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: Same structural pinning argument as GpuPrintFuture::poll.
        let this = unsafe { self.get_unchecked_mut() };
        match this.state {
            HostcallState::Init => {
                if this.max_len == 0 {
                    this.state = HostcallState::Done;
                    return Poll::Ready(Ok(0));
                }

                // Allocate sideband space for response data
                let offset =
                    unsafe { crate::sideband::sideband_alloc(this.sideband, this.max_len as u64) };
                if offset == u64::MAX {
                    return Poll::Pending; // Retry on next poll
                }
                this.sideband_offset = offset;

                // Submit hostcall
                let fd = this.fd;
                let sb_offset = this.sideband_offset;
                let max_len = this.max_len;
                match unsafe {
                    submit_hostcall(this.buf, SERVICE_TCP_BULK_READ, |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                        core::ptr::write_volatile(payload.add(8) as *mut u64, sb_offset);
                        core::ptr::write_volatile(payload.add(16) as *mut u64, max_len as u64);
                    })
                } {
                    Ok(idx) => {
                        this.state = HostcallState::Waiting { pkt_idx: idx };
                        Poll::Pending
                    }
                    Err(()) => Poll::Pending,
                }
            }
            HostcallState::Waiting { pkt_idx } => unsafe {
                match check_response(this.buf, pkt_idx) {
                    Some(pkt) => {
                        let bytes_read =
                            core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                        crate::hostcall::gpu_hostcall_release(this.buf, pkt);
                        this.state = HostcallState::Done;

                        if bytes_read == FILE_ERROR_SENTINEL || bytes_read == 0 {
                            if bytes_read == 0 {
                                return Poll::Ready(Ok(0)); // Connection closed / EOF
                            }
                            return Poll::Ready(Err(-1));
                        }

                        // Copy data from sideband to destination
                        let src = this
                            .sideband
                            .add(SIDEBAND_DATA_OFFSET + this.sideband_offset as usize);
                        let n = bytes_read as usize;
                        let mut i = 0;
                        while i < n {
                            core::ptr::write_volatile(
                                this.dst.add(i),
                                core::ptr::read_volatile(src.add(i)),
                            );
                            i += 1;
                        }

                        Poll::Ready(Ok(n))
                    }
                    None => Poll::Pending,
                }
            },
            HostcallState::Done => Poll::Ready(Err(-1)),
        }
    }
}

/// Default maximum poll iterations before timeout.
const DEFAULT_MAX_POLLS: u32 = 10_000_000;

/// Default nanosleep duration between polls (nanoseconds).
const DEFAULT_NANOSLEEP_NS: u32 = 1000;

/// No-op waker vtable for GPU (no real wake mechanism).
const NOOP_VTABLE: core::task::RawWakerVTable = core::task::RawWakerVTable::new(
    |_| core::task::RawWaker::new(core::ptr::null(), &NOOP_VTABLE),
    |_| {},
    |_| {},
    |_| {},
);

/// Create a no-op waker suitable for GPU spin-polling.
#[inline(always)]
fn noop_waker() -> core::task::Waker {
    unsafe {
        core::task::Waker::from_raw(core::task::RawWaker::new(core::ptr::null(), &NOOP_VTABLE))
    }
}

/// Run a future to completion by spin-polling with nanosleep yield.
///
/// This is the primary way to drive async code on GPU. Replaces the
/// manual waker/context/pin/poll boilerplate that every kernel previously
/// needed.
///
/// Returns `Some(output)` on completion, `None` on timeout (10M polls).
///
/// # Safety
/// The caller must ensure the future is safe to poll on the current thread
/// and that any raw pointers captured by the future remain valid.
///
/// # Example
/// ```no_run
/// use core::future::ready;
/// use gpu_runtime::std_future::block_on;
///
/// let result = unsafe { block_on(ready(7u32)) };
/// assert_eq!(result, Some(7));
/// ```
#[inline(always)]
pub unsafe fn block_on<F: Future>(future: F) -> Option<F::Output> {
    block_on_with(future, DEFAULT_MAX_POLLS, DEFAULT_NANOSLEEP_NS)
}

/// Run a future with custom poll limit and nanosleep duration.
///
/// # Safety
/// Same as [`block_on`].
#[inline(always)]
pub unsafe fn block_on_with<F: Future>(
    future: F,
    max_polls: u32,
    _nanosleep_ns: u32,
) -> Option<F::Output> {
    let mut future = future;
    // SAFETY: The future is stack-pinned here and never moved — we poll it
    // in a loop via Pin::as_mut() until completion or timeout.
    let mut future = Pin::new_unchecked(&mut future);
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);

    let mut polls: u32 = 0;
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(output) => return Some(output),
            Poll::Pending => {
                polls += 1;
                if polls >= max_polls {
                    return None;
                }
                #[cfg(target_arch = "nvptx64")]
                {
                    // Yield SM scheduler slot — gives host time to respond
                    // and allows other warps to execute.
                    match _nanosleep_ns {
                        64 => core::arch::asm!("nanosleep.u32 64;", options(nostack)),
                        1000 => core::arch::asm!("nanosleep.u32 1000;", options(nostack)),
                        _ => core::arch::asm!("nanosleep.u32 1000;", options(nostack)),
                    }
                }
            }
        }
    }
}

/// Minimal spin-poll executor for a single `Future`.
///
/// No waker, no task queue — just polls in a loop with nanosleep yield.
/// Prefer the free function [`block_on`] for simpler usage.
pub struct SpinExecutor;

impl SpinExecutor {
    /// Run a future to completion by spin-polling.
    ///
    /// Returns the future's output, or `None` if MAX_POLLS exceeded.
    ///
    /// # Safety
    /// The future must be safe to poll repeatedly on the current thread.
    #[inline(always)]
    pub unsafe fn run<F: Future>(future: &mut F) -> Option<F::Output> {
        // SAFETY: The caller provides a mutable reference that we pin in place.
        // We only poll via Pin::as_mut() and never move the future.
        let mut future = Pin::new_unchecked(future);
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        let mut polls: u32 = 0;
        loop {
            match future.as_mut().poll(&mut cx) {
                Poll::Ready(output) => return Some(output),
                Poll::Pending => {
                    polls += 1;
                    if polls >= DEFAULT_MAX_POLLS {
                        return None;
                    }
                    #[cfg(target_arch = "nvptx64")]
                    core::arch::asm!("nanosleep.u32 1000;", options(nostack));
                }
            }
        }
    }
}
