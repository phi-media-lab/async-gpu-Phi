use core::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

#[cfg(not(test))]
use gpu_atomics::{
    activemask, membar_sys, sys_cas_u32, sys_cas_u64, sys_fetch_add_u64, sys_load_acquire_u32,
    sys_load_acquire_u64, sys_spin_load_acquire_u32, sys_store_release_u32,
};
use gpu_protocol::*;

// Host-side unit tests exercise the ownership state machine against ordinary
// atomics. Production and NVPTX builds continue to use system-scope GPU
// atomics; this shim exists only because gpu-atomics intentionally traps when
// invoked on a CPU.
#[cfg(test)]
mod host_test_atomics {
    use core::sync::atomic::{fence, AtomicU32, AtomicU64, Ordering};

    pub fn activemask() -> u32 {
        u32::MAX
    }

    pub fn membar_sys() {
        fence(Ordering::SeqCst);
    }

    pub unsafe fn sys_load_acquire_u32(ptr: *const u32) -> u32 {
        (&*(ptr as *const AtomicU32)).load(Ordering::Acquire)
    }

    pub unsafe fn sys_spin_load_acquire_u32(ptr: *const u32) -> u32 {
        sys_load_acquire_u32(ptr)
    }

    pub unsafe fn sys_store_release_u32(ptr: *mut u32, value: u32) {
        (&*(ptr as *const AtomicU32)).store(value, Ordering::Release);
    }

    pub unsafe fn sys_cas_u32(ptr: *mut u32, current: u32, new: u32) -> u32 {
        (&*(ptr as *const AtomicU32))
            .compare_exchange(current, new, Ordering::AcqRel, Ordering::Acquire)
            .unwrap_or_else(|actual| actual)
    }

    pub unsafe fn sys_load_acquire_u64(ptr: *const u64) -> u64 {
        (&*(ptr as *const AtomicU64)).load(Ordering::Acquire)
    }

    pub unsafe fn sys_cas_u64(ptr: *mut u64, current: u64, new: u64) -> u64 {
        (&*(ptr as *const AtomicU64))
            .compare_exchange(current, new, Ordering::AcqRel, Ordering::Acquire)
            .unwrap_or_else(|actual| actual)
    }

    pub unsafe fn sys_fetch_add_u64(ptr: *mut u64, value: u64) -> u64 {
        (&*(ptr as *const AtomicU64)).fetch_add(value, Ordering::AcqRel)
    }
}

#[cfg(test)]
use host_test_atomics::*;

// ================================================================
// Sharding helpers — compute shard index and resolve packet offsets
// ================================================================

/// Read sharding metadata from buffer header. Returns (num_shards, shard_array_offset, pkts_per_shard).
/// If num_shards == 0, this is a legacy (unsharded) buffer.
#[inline(always)]
pub unsafe fn read_shard_info(buf: *const u8) -> (u32, u32, u32) {
    let num_shards = core::ptr::read_volatile(buf.add(BUF_OFF_NUM_SHARDS) as *const u32);
    if num_shards == 0 {
        return (0, BUFFER_HEADER_SIZE as u32, 0);
    }
    let pkts_per_shard = core::ptr::read_volatile(buf.add(BUF_OFF_PKTS_PER_SHARD) as *const u32);
    let shard_array_off = core::ptr::read_volatile(buf.add(BUF_OFF_SHARD_ARRAY_OFF) as *const u32);
    (num_shards, shard_array_off, pkts_per_shard)
}

/// Read priority-reservation metadata from a versioned buffer header.
///
/// Version-zero buffers predate priority hostcalls. Treating them as having no
/// reservation keeps a new device runtime compatible with an old host buffer.
#[inline(always)]
pub unsafe fn read_priority_info(buf: *const u8) -> (u32, u32, u16) {
    let version = core::ptr::read_volatile(buf.add(BUF_OFF_PROTOCOL_VERSION) as *const u32);
    let num_packets = core::ptr::read_volatile(buf.add(BUF_OFF_NUM_PACKETS) as *const u32) as u16;
    let reserved_per_shard = if hostcall_supports_priority(version) {
        core::ptr::read_volatile(buf.add(BUF_OFF_HIGH_RESERVED_PER_SHARD) as *const u32)
    } else {
        0
    };
    (version, reserved_per_shard, num_packets)
}

/// Decode task metadata from a packet. Legacy or unknown metadata versions
/// safely fall back to the default Normal-priority, anonymous task context.
#[inline(always)]
pub unsafe fn gpu_hostcall_packet_metadata(pkt: *const u8) -> HostcallMetadata {
    let version = core::ptr::read_volatile(pkt.add(PKT_OFF_METADATA_VERSION));
    if version != PACKET_METADATA_PRIORITY_VERSION && version != PACKET_METADATA_VERSION {
        return HostcallMetadata::default();
    }
    let priority = Priority::from_raw(core::ptr::read_volatile(pkt.add(PKT_OFF_PRIORITY)));
    let task_id = core::ptr::read_volatile(pkt.add(PKT_OFF_TASK_ID) as *const u64);
    HostcallMetadata::new(task_id, priority)
}

/// Store task metadata before the packet's CONTROL_FILLED release-store.
#[inline(always)]
unsafe fn write_packet_metadata(
    pkt: *mut u8,
    metadata: HostcallMetadata,
    metadata_version: u8,
    generation: u64,
) {
    // Publish the version last. Host readers first acquire a stable control
    // ownership state, then treat a recognized version as the commit marker.
    core::ptr::write_volatile(
        pkt.add(PKT_OFF_PRIORITY),
        metadata.effective_priority.as_raw(),
    );
    let generation_metadata = if metadata_version == PACKET_METADATA_VERSION {
        request_generation_metadata(generation)
    } else {
        0
    };
    core::ptr::write_volatile(
        pkt.add(PKT_OFF_METADATA_FLAGS) as *mut u16,
        generation_metadata,
    );
    core::ptr::write_volatile(pkt.add(PKT_OFF_TASK_ID) as *mut u64, metadata.task_id);
    core::ptr::write_volatile(pkt.add(PKT_OFF_METADATA_VERSION), metadata_version);
}

/// Require the generation/cancellation protocol before consuming a packet.
/// A finite timeout cannot be made safe against an older host that has no
/// ownership-transfer state, so compatibility is an explicit pre-submit error.
#[inline(always)]
unsafe fn require_cancellation_protocol(buf: *const u8) -> Result<(), GpuError> {
    let version = core::ptr::read_volatile(buf.add(BUF_OFF_PROTOCOL_VERSION) as *const u32);
    if !hostcall_supports_cancellation(version) {
        Err(GpuError::unsupported())
    } else {
        Ok(())
    }
}

/// Initialize a freshly popped packet and return its next non-zero generation.
#[inline(always)]
unsafe fn prepare_versioned_packet(pkt: *mut u8, metadata: HostcallMetadata) -> u64 {
    let control_ptr = pkt.add(PKT_OFF_CONTROL) as *mut u32;
    let old_control = sys_load_acquire_u32(control_ptr as *const u32);
    let old_generation_metadata =
        core::ptr::read_volatile(pkt.add(PKT_OFF_METADATA_FLAGS) as *const u16);
    let generation = next_request_generation(old_control, old_generation_metadata);
    sys_store_release_u32(control_ptr, make_control(generation, 0));
    write_packet_metadata(pkt, metadata, PACKET_METADATA_VERSION, generation);
    generation
}

/// Compute the byte offset of a packet from buf base, handling both legacy and sharded layouts.
#[inline(always)]
pub unsafe fn pkt_offset(buf: *const u8, idx: u16) -> usize {
    let (num_shards, shard_array_off, _) = read_shard_info(buf);
    if num_shards == 0 {
        packet_offset(idx)
    } else {
        packet_offset_sharded(idx, shard_array_off as usize, num_shards)
    }
}

/// Get the free stack pointer for the current block's shard (or global if unsharded).
#[inline(always)]
pub unsafe fn get_free_stack_ptr(buf: *mut u8, num_shards: u32, shard_array_off: u32) -> *mut u64 {
    if num_shards == 0 {
        buf.add(BUF_OFF_FREE_STACK) as *mut u64
    } else {
        let shard_idx = crate::nvptx_shim::block_idx_x() % num_shards;
        let entry_off = shard_entry_offset(shard_array_off as usize, shard_idx);
        buf.add(entry_off + SHARD_OFF_FREE_STACK) as *mut u64
    }
}

/// Get the ready stack pointer for the current block's shard (or global if unsharded).
#[inline(always)]
pub unsafe fn get_ready_stack_ptr(buf: *mut u8, num_shards: u32, shard_array_off: u32) -> *mut u64 {
    if num_shards == 0 {
        buf.add(BUF_OFF_READY_STACK) as *mut u64
    } else {
        let shard_idx = crate::nvptx_shim::block_idx_x() % num_shards;
        let entry_off = shard_entry_offset(shard_array_off as usize, shard_idx);
        buf.add(entry_off + SHARD_OFF_READY_STACK) as *mut u64
    }
}

/// Get the global High-only free stack pointer.
#[inline(always)]
unsafe fn get_high_free_stack_ptr(buf: *mut u8) -> *mut u64 {
    buf.add(BUF_OFF_HIGH_FREE_STACK) as *mut u64
}

// ================================================================
// Core stack operations
// ================================================================

/// Pop a packet from the free stack. Returns packet index or NULL_INDEX.
#[inline(always)]
pub unsafe fn hc_pop_free(buf: *mut u8) -> u16 {
    let (num_shards, shard_array_off, _) = read_shard_info(buf as *const u8);
    let free_ptr = get_free_stack_ptr(buf, num_shards, shard_array_off);
    hc_pop_free_from(buf, free_ptr, num_shards, shard_array_off)
}

/// Pop a packet for a request, honoring the shared global High-only reservation
/// when the buffer advertises protocol v2/v3. High requests fall back to the
/// caller's general shard after the shared stack is exhausted; Normal and Low
/// never consume reserved packets.
#[inline(always)]
unsafe fn hc_pop_for_metadata(
    buf: *mut u8,
    metadata: HostcallMetadata,
    num_shards: u32,
    shard_array_off: u32,
) -> u16 {
    let (version, reserved_per_shard, _) = read_priority_info(buf as *const u8);
    if metadata.effective_priority == Priority::High
        && hostcall_supports_priority(version)
        && reserved_per_shard != 0
    {
        let idx = hc_pop_free_from(
            buf,
            get_high_free_stack_ptr(buf),
            num_shards,
            shard_array_off,
        );
        if idx != NULL_INDEX {
            return idx;
        }
    }

    let free_ptr = get_free_stack_ptr(buf, num_shards, shard_array_off);
    hc_pop_free_from(buf, free_ptr, num_shards, shard_array_off)
}

/// Pop a packet from a specific free stack pointer.
#[inline(always)]
pub unsafe fn hc_pop_free_from(
    buf: *mut u8,
    free_ptr: *mut u64,
    num_shards: u32,
    shard_array_off: u32,
) -> u16 {
    // SAFETY: Lock-free Treiber stack pop with ABA prevention.
    //
    // The stack head is a tagged pointer: bits 32-63 = epoch tag, bits 0-15
    // = packet index (NULL_INDEX 0xFFFF = empty). The CAS compares the full
    // 64-bit tagged value, so even if a packet is popped and re-pushed at
    // the same index, the epoch tag will differ, preventing ABA.
    //
    // Acquire load on the head ensures we see the `next` pointer written by
    // the most recent push. System-scope CAS (sys_cas_u64) provides
    // visibility across all GPU SMs and the host CPU.
    //
    // Ownership protocol: a successful CAS transfers exclusive ownership of
    // the packet to the popping thread. The packet is not on any stack until
    // explicitly pushed back.
    loop {
        let old_head = sys_load_acquire_u64(free_ptr as *const u64);
        let idx = tagged_index(old_head);
        if idx == NULL_INDEX {
            return NULL_INDEX;
        }
        let pkt_off = if num_shards == 0 {
            packet_offset(idx)
        } else {
            packet_offset_sharded(idx, shard_array_off as usize, num_shards)
        };
        let pkt = buf.add(pkt_off);
        let next = core::ptr::read_volatile(pkt.add(PKT_OFF_NEXT) as *const u64);
        // A pop is also a stack-head mutation and must advance the ABA tag.
        // Installing the cached `next` word verbatim would restore an older tag
        // and permit a short T1-pop/T2-pop+push/T1-CAS ABA cycle.
        let new_head = advance_tagged_head(old_head, tagged_index(next));
        if sys_cas_u64(free_ptr, old_head, new_head) == old_head {
            return idx;
        }
    }
}

/// Push a packet onto a tagged-pointer stack (free or ready).
#[inline(always)]
pub unsafe fn hc_push(stack_ptr: *mut u64, buf: *mut u8, pkt_idx: u16) {
    let (num_shards, shard_array_off, _) = read_shard_info(buf as *const u8);
    hc_push_with(stack_ptr, buf, pkt_idx, num_shards, shard_array_off);
}

/// Push with pre-computed sharding info.
#[inline(always)]
pub(crate) unsafe fn hc_push_with(
    stack_ptr: *mut u64,
    buf: *mut u8,
    pkt_idx: u16,
    num_shards: u32,
    shard_array_off: u32,
) {
    // SAFETY: Lock-free Treiber stack push with ABA prevention.
    //
    // We write the current head into the packet's `next` field, then CAS
    // the stack head from old_head to a new tagged value containing our
    // packet index and an incremented epoch tag. The epoch tag (bits 32-63)
    // is incremented on every push to prevent ABA: even if a concurrent
    // pop+push restores the same index, the tag will differ.
    //
    // The caller must have exclusive ownership of the packet (obtained via
    // hc_pop_free_from or initial allocation). After a successful CAS, the
    // packet is visible to other threads via the stack.
    let pkt_off = if num_shards == 0 {
        packet_offset(pkt_idx)
    } else {
        packet_offset_sharded(pkt_idx, shard_array_off as usize, num_shards)
    };
    let pkt = buf.add(pkt_off);
    loop {
        let old_head = sys_load_acquire_u64(stack_ptr as *const u64);
        core::ptr::write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, old_head);
        // Publish `next` and any packet cleanup before the new head becomes
        // visible to another SM or the host CPU.
        membar_sys();
        let new_tagged = advance_tagged_head(old_head, pkt_idx);
        if sys_cas_u64(stack_ptr, old_head, new_tagged) == old_head {
            break;
        }
    }
}

/// Release a packet index to the pool from which it was allocated.
#[inline(always)]
unsafe fn hc_release_index(buf: *mut u8, pkt_idx: u16, num_shards: u32, shard_array_off: u32) {
    // Publish IDLE before clearing metadata. A concurrent packet_metadata()
    // double-snapshot must see either the old READY word followed by IDLE, or
    // IDLE immediately; it must never accept half-cleared metadata under an
    // unchanged READY word. The packet is not reusable until the later stack
    // push, so clearing the fields after IDLE is safe.
    let pkt = buf.add(pkt_offset(buf as *const u8, pkt_idx));
    let metadata_version = core::ptr::read_volatile(pkt.add(PKT_OFF_METADATA_VERSION));
    let generation_metadata =
        core::ptr::read_volatile(pkt.add(PKT_OFF_METADATA_FLAGS) as *const u16);
    let control_ptr = pkt.add(PKT_OFF_CONTROL) as *mut u32;
    let control = sys_load_acquire_u32(control_ptr as *const u32);
    let idle_control = if metadata_version == PACKET_METADATA_VERSION {
        make_control(request_generation(control, generation_metadata), 0)
    } else {
        0
    };
    sys_store_release_u32(control_ptr, idle_control);
    core::ptr::write_volatile(pkt.add(PKT_OFF_METADATA_VERSION), 0);
    core::ptr::write_volatile(pkt.add(PKT_OFF_PRIORITY), Priority::Normal.as_raw());
    if metadata_version != PACKET_METADATA_VERSION {
        core::ptr::write_volatile(pkt.add(PKT_OFF_METADATA_FLAGS) as *mut u16, 0);
    }
    core::ptr::write_volatile(pkt.add(PKT_OFF_TASK_ID) as *mut u64, 0);

    let (version, reserved_per_shard, num_packets) = read_priority_info(buf as *const u8);
    let (_, _, pkts_per_shard) = read_shard_info(buf as *const u8);
    let reserved = hostcall_supports_priority(version)
        && is_high_reserved_packet(
            pkt_idx,
            num_packets,
            num_shards,
            pkts_per_shard,
            reserved_per_shard,
        );
    let free_ptr = if reserved {
        get_high_free_stack_ptr(buf)
    } else if num_shards != 0 && pkts_per_shard != 0 {
        // Release by packet provenance, not by the block currently executing
        // this function. A typed/raw completion handle may be consumed after
        // task migration to another scheduler block.
        let original_shard = (pkt_idx as u32) / pkts_per_shard;
        let entry_off = shard_entry_offset(shard_array_off as usize, original_shard);
        buf.add(entry_off + SHARD_OFF_FREE_STACK) as *mut u64
    } else {
        get_free_stack_ptr(buf, num_shards, shard_array_off)
    };
    hc_push_with(free_ptr, buf, pkt_idx, num_shards, shard_array_off);
}

/// Release a packet back to the free stack.
#[inline(always)]
pub unsafe fn gpu_hostcall_release(buf: *mut u8, pkt: *mut u8) {
    let (num_shards, shard_array_off, _) = read_shard_info(buf as *const u8);
    let pkt_offset_bytes = (pkt as usize) - (buf as usize);
    let packet_base = if num_shards == 0 {
        BUFFER_HEADER_SIZE
    } else {
        shard_array_off as usize + (num_shards as usize) * SHARD_ENTRY_SIZE
    };
    let idx = ((pkt_offset_bytes - packet_base) / PACKET_SIZE) as u16;
    hc_release_index(buf, idx, num_shards, shard_array_off);
}

/// Maximum spin iterations before declaring timeout.
pub const GPU_MAX_SPIN: u32 = 10_000_000;

/// A packet acquired from the v3 hostcall pool but not yet published to the
/// host.  Dropping this value is ownership-safe: an unsubmitted packet is
/// returned directly to its provenance pool.
pub struct HostcallPacketLease {
    packet: Option<LeasedPacket>,
}

#[derive(Clone, Copy)]
struct LeasedPacket {
    buf: *mut u8,
    pkt_idx: u16,
    pkt: *mut u8,
    generation: u64,
    num_shards: u32,
    shard_array_off: u32,
    shared_high_reserved: bool,
}

/// Inline response produced by a v3 asynchronous hostcall.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostcallResponse {
    /// Eight lane-zero payload slots after host completion.
    pub payload: [u64; 8],
    /// Physical packet index used by the request.
    pub packet_index: u16,
    /// Whether the packet came from the shared High-only reserve.
    pub shared_high_reserved: bool,
    /// Number of cooperative pending polls, including the mandatory first one.
    pub pending_polls: u32,
}

impl HostcallPacketLease {
    /// Acquire and generation-initialize one packet without submitting it.
    ///
    /// # Safety
    ///
    /// `buf` must be a live mapped v3 [`gpu_protocol`] hostcall buffer for the
    /// whole lifetime of this lease and any future created from it.
    #[inline(always)]
    pub unsafe fn acquire(buf: *mut u8, metadata: HostcallMetadata) -> Result<Self, GpuError> {
        require_cancellation_protocol(buf as *const u8)?;
        let (num_shards, shard_array_off, pkts_per_shard) = read_shard_info(buf as *const u8);
        let (version, reserved_per_shard, num_packets) = read_priority_info(buf as *const u8);
        let pkt_idx = hc_pop_for_metadata(buf, metadata, num_shards, shard_array_off);
        if pkt_idx == NULL_INDEX {
            return Err(GpuError::pool_exhausted());
        }

        let pkt = buf.add(pkt_offset(buf as *const u8, pkt_idx));
        let generation = prepare_versioned_packet(pkt, metadata);
        let shared_high_reserved = hostcall_supports_priority(version)
            && is_high_reserved_packet(
                pkt_idx,
                num_packets,
                num_shards,
                pkts_per_shard,
                reserved_per_shard,
            );
        Ok(Self {
            packet: Some(LeasedPacket {
                buf,
                pkt_idx,
                pkt,
                generation,
                num_shards,
                shard_array_off,
                shared_high_reserved,
            }),
        })
    }

    /// Physical packet selected for this lease.
    #[inline(always)]
    pub fn packet_index(&self) -> u16 {
        self.packet.as_ref().map_or(NULL_INDEX, |p| p.pkt_idx)
    }

    /// True iff this lease consumed the shared High-only reserve.
    #[inline(always)]
    pub fn is_shared_high_reserved(&self) -> bool {
        self.packet.as_ref().is_some_and(|p| p.shared_high_reserved)
    }

    /// Convert the lease into a cooperative v3 request future.
    ///
    /// Submission occurs on the first poll, which always returns `Pending` and
    /// self-wakes. `max_response_polls` is checked only on later polls.
    #[inline(always)]
    pub fn submit(
        mut self,
        service: u32,
        payload: [u64; 8],
        max_response_polls: u32,
    ) -> PendingHostcall {
        PendingHostcall {
            packet: self.packet.take(),
            service,
            payload,
            max_response_polls,
            pending_polls: 0,
            submitted: false,
        }
    }
}

impl Drop for HostcallPacketLease {
    #[inline(always)]
    fn drop(&mut self) {
        if let Some(packet) = self.packet.take() {
            unsafe {
                hc_release_index(
                    packet.buf,
                    packet.pkt_idx,
                    packet.num_shards,
                    packet.shard_array_off,
                );
            }
        }
    }
}

/// A self-waking asynchronous v3 hostcall with explicit ownership transfer.
///
/// Once submitted, timeout/Drop never pushes the packet directly to a free
/// stack.  Instead they win `FILLED|HOST_OWNED -> CANCELLED`; the host then
/// acknowledges the matching generation and reclaims the packet.
pub struct PendingHostcall {
    packet: Option<LeasedPacket>,
    service: u32,
    payload: [u64; 8],
    max_response_polls: u32,
    pending_polls: u32,
    submitted: bool,
}

impl PendingHostcall {
    /// Number of polls that returned `Pending`.
    #[inline(always)]
    pub fn pending_polls(&self) -> u32 {
        self.pending_polls
    }

    /// Physical packet selected for this request.
    #[inline(always)]
    pub fn packet_index(&self) -> u16 {
        self.packet.as_ref().map_or(NULL_INDEX, |p| p.pkt_idx)
    }

    /// True iff this request uses the shared High-only reserve.
    #[inline(always)]
    pub fn is_shared_high_reserved(&self) -> bool {
        self.packet.as_ref().is_some_and(|p| p.shared_high_reserved)
    }

    #[inline(always)]
    unsafe fn publish(&mut self) {
        let packet = self.packet.expect("pending hostcall packet");
        core::ptr::write_volatile(
            packet.pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32,
            activemask(),
        );
        core::ptr::write_volatile(packet.pkt.add(PKT_OFF_SERVICE) as *mut u32, self.service);
        let payload_ptr = packet.pkt.add(PKT_OFF_PAYLOAD) as *mut u64;
        let mut slot = 0;
        while slot < self.payload.len() {
            core::ptr::write_volatile(payload_ptr.add(slot), self.payload[slot]);
            slot += 1;
        }
        sys_store_release_u32(
            packet.pkt.add(PKT_OFF_CONTROL) as *mut u32,
            make_control(packet.generation, CONTROL_FILLED),
        );
        let ready_ptr = get_ready_stack_ptr(packet.buf, packet.num_shards, packet.shard_array_off);
        hc_push_with(
            ready_ptr,
            packet.buf,
            packet.pkt_idx,
            packet.num_shards,
            packet.shard_array_off,
        );
        sys_fetch_add_u64(packet.buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);
        self.submitted = true;
    }

    #[inline(always)]
    unsafe fn completed_response(&mut self) -> Result<HostcallResponse, GpuError> {
        let packet = self.packet.expect("completed hostcall packet");
        let mut payload = [0u64; 8];
        let payload_ptr = packet.pkt.add(PKT_OFF_PAYLOAD) as *const u64;
        let mut slot = 0;
        while slot < payload.len() {
            payload[slot] = core::ptr::read_volatile(payload_ptr.add(slot));
            slot += 1;
        }
        let control = sys_load_acquire_u32(packet.pkt.add(PKT_OFF_CONTROL) as *const u32);
        let response = HostcallResponse {
            payload,
            packet_index: packet.pkt_idx,
            shared_high_reserved: packet.shared_high_reserved,
            pending_polls: self.pending_polls,
        };
        let error = if control_flags(control) & CONTROL_ERROR != 0 {
            Some(GpuError::from_encoded(payload[0]))
        } else {
            None
        };
        hc_release_index(
            packet.buf,
            packet.pkt_idx,
            packet.num_shards,
            packet.shard_array_off,
        );
        self.packet.take();
        match error {
            Some(error) => Err(error),
            None => Ok(response),
        }
    }

    /// Cancel a submitted same-generation request.  `true` means ownership was
    /// transferred to the host; `false` means the state changed and must be
    /// observed again by the caller.
    #[inline(always)]
    unsafe fn try_cancel(&self, control: u32) -> bool {
        let packet = self.packet.expect("cancelled hostcall packet");
        let flags = control_flags(control);
        if flags != CONTROL_FILLED && flags != CONTROL_HOST_OWNED {
            return false;
        }
        sys_cas_u32(
            packet.pkt.add(PKT_OFF_CONTROL) as *mut u32,
            control,
            make_control(packet.generation, CONTROL_CANCELLED),
        ) == control
    }

    #[inline(always)]
    unsafe fn cancel_for_drop(&mut self) {
        let Some(packet) = self.packet else {
            return;
        };
        if !self.submitted {
            hc_release_index(
                packet.buf,
                packet.pkt_idx,
                packet.num_shards,
                packet.shard_array_off,
            );
            self.packet.take();
            return;
        }

        loop {
            let control = sys_load_acquire_u32(packet.pkt.add(PKT_OFF_CONTROL) as *const u32);
            if !generation_matches(packet.pkt, control, packet.generation) {
                self.packet.take();
                return;
            }
            let flags = control_flags(control);
            if flags & CONTROL_READY != 0 {
                hc_release_index(
                    packet.buf,
                    packet.pkt_idx,
                    packet.num_shards,
                    packet.shard_array_off,
                );
                self.packet.take();
                return;
            }
            if flags == CONTROL_CANCELLED || flags == 0 {
                self.packet.take();
                return;
            }
            if self.try_cancel(control) {
                self.packet.take();
                return;
            }
        }
    }
}

impl Future for PendingHostcall {
    type Output = Result<HostcallResponse, GpuError>;

    #[inline(always)]
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        unsafe {
            if !this.submitted {
                this.publish();
                this.pending_polls = 1;
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }

            let Some(packet) = this.packet else {
                return Poll::Ready(Err(GpuError::unsupported()));
            };

            let control = sys_spin_load_acquire_u32(packet.pkt.add(PKT_OFF_CONTROL) as *const u32);
            if !generation_matches(packet.pkt, control, packet.generation) {
                this.packet.take();
                return Poll::Ready(Err(GpuError::timeout()));
            }
            let flags = control_flags(control);
            if flags & CONTROL_READY != 0 {
                return Poll::Ready(this.completed_response());
            }
            if flags == CONTROL_CANCELLED || flags == 0 {
                this.packet.take();
                return Poll::Ready(Err(GpuError::timeout()));
            }
            if this.pending_polls >= this.max_response_polls && this.try_cancel(control) {
                this.packet.take();
                return Poll::Ready(Err(GpuError::timeout()));
            }

            this.pending_polls = this.pending_polls.saturating_add(1);
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

impl Drop for PendingHostcall {
    #[inline(always)]
    fn drop(&mut self) {
        unsafe { self.cancel_for_drop() }
    }
}

struct SubmittedPacket {
    pkt_idx: u16,
    pkt: *mut u8,
    generation: u64,
    num_shards: u32,
    shard_array_off: u32,
}

enum WaitOutcome {
    Ready,
    HostError(GpuError),
    Cancelled,
}

#[inline(always)]
unsafe fn generation_matches(pkt: *const u8, control: u32, generation: u64) -> bool {
    let generation_metadata =
        core::ptr::read_volatile(pkt.add(PKT_OFF_METADATA_FLAGS) as *const u16);
    request_generation(control, generation_metadata) == generation
}

/// Submit one generation-tagged request. A successful return transfers the
/// packet to the host; only completion or a successful cancellation CAS can
/// transfer ownership again.
#[inline(always)]
unsafe fn submit_versioned_request(
    buf: *mut u8,
    service: u32,
    metadata: HostcallMetadata,
    fill_payload: impl FnOnce(*mut u8),
) -> Result<SubmittedPacket, GpuError> {
    require_cancellation_protocol(buf as *const u8)?;

    let (num_shards, shard_array_off, _) = read_shard_info(buf as *const u8);
    let ready_ptr = get_ready_stack_ptr(buf, num_shards, shard_array_off);
    let pkt_idx = hc_pop_for_metadata(buf, metadata, num_shards, shard_array_off);
    if pkt_idx == NULL_INDEX {
        return Err(GpuError::pool_exhausted());
    }

    let pkt = buf.add(pkt_offset(buf as *const u8, pkt_idx));
    core::ptr::write_volatile(pkt.add(PKT_OFF_ACTIVE_MASK) as *mut u32, activemask());
    core::ptr::write_volatile(pkt.add(PKT_OFF_SERVICE) as *mut u32, service);
    let generation = prepare_versioned_packet(pkt, metadata);
    fill_payload(pkt.add(PKT_OFF_PAYLOAD));

    sys_store_release_u32(
        pkt.add(PKT_OFF_CONTROL) as *mut u32,
        make_control(generation, CONTROL_FILLED),
    );
    hc_push_with(ready_ptr, buf, pkt_idx, num_shards, shard_array_off);
    sys_fetch_add_u64(buf.add(BUF_OFF_DOORBELL) as *mut u64, 1);

    Ok(SubmittedPacket {
        pkt_idx,
        pkt,
        generation,
        num_shards,
        shard_array_off,
    })
}

/// Wait for completion, or atomically transfer packet-release responsibility
/// to the host when the finite spin budget expires.
#[inline(always)]
unsafe fn wait_versioned_response(request: &SubmittedPacket, max_spin: u32) -> WaitOutcome {
    let control_ptr = request.pkt.add(PKT_OFF_CONTROL) as *mut u32;
    let mut spins = 0u32;

    loop {
        let control = sys_spin_load_acquire_u32(control_ptr as *const u32);
        if !generation_matches(request.pkt, control, request.generation) {
            // A matching cancellation may already have been acknowledged,
            // released, and reused before this lane observes its CAS result.
            return WaitOutcome::Cancelled;
        }

        let flags = control_flags(control);
        if flags & CONTROL_READY != 0 {
            if flags & CONTROL_ERROR != 0 {
                let slot0 =
                    core::ptr::read_volatile(request.pkt.add(PKT_OFF_PAYLOAD) as *const u64);
                return WaitOutcome::HostError(GpuError::from_encoded(slot0));
            }
            return WaitOutcome::Ready;
        }
        if flags == CONTROL_CANCELLED || flags == 0 {
            return WaitOutcome::Cancelled;
        }

        spins = spins.saturating_add(1);
        if spins < max_spin {
            continue;
        }

        if flags == CONTROL_FILLED || flags == CONTROL_HOST_OWNED {
            let cancelled = make_control(request.generation, CONTROL_CANCELLED);
            if sys_cas_u32(control_ptr, control, cancelled) == control {
                return WaitOutcome::Cancelled;
            }
            // Completion and cancellation race through the same control word;
            // retry to observe which side won.
            continue;
        }

        // Unknown same-generation state: transfer ownership if possible rather
        // than publishing the packet to a free stack behind the host's back.
        let cancelled = make_control(request.generation, CONTROL_CANCELLED);
        if sys_cas_u32(control_ptr, control, cancelled) == control {
            return WaitOutcome::Cancelled;
        }
    }
}

#[inline(always)]
unsafe fn finish_request_wait(
    buf: *mut u8,
    request: &SubmittedPacket,
    max_spin: u32,
) -> Result<(), GpuError> {
    match wait_versioned_response(request, max_spin) {
        WaitOutcome::Ready => Ok(()),
        WaitOutcome::HostError(error) => {
            hc_release_index(
                buf,
                request.pkt_idx,
                request.num_shards,
                request.shard_array_off,
            );
            Err(error)
        }
        WaitOutcome::Cancelled => Err(GpuError::timeout()),
    }
}

/// Submit a hostcall request: pop packet, fill header + payload, push to ready
/// stack, ring doorbell, spin-wait for response.
///
/// Returns `Ok(pkt_ptr)` on success — the payload contains the host's response.
/// Returns `Err(GpuError)` on failure: pool exhaustion, timeout, or host-side error
/// (decoded from CONTROL_ERROR + payload slot 0).
/// Caller must call `gpu_hostcall_release(buf, pkt)` after reading the response.
#[inline(always)]
pub unsafe fn gpu_hostcall_request(
    buf: *mut u8,
    service: u32,
    fill_payload: impl FnOnce(*mut u8),
) -> Result<*mut u8, GpuError> {
    gpu_hostcall_request_with_metadata(buf, service, HostcallMetadata::default(), fill_payload)
}

/// Submit a hostcall request with explicit executor task metadata.
///
/// The metadata remains in the packet through completion and can be recovered
/// with [`gpu_hostcall_packet_metadata`] before releasing the packet. This is
/// the bridge used by async completion paths to wake the originating task at
/// its effective priority.
#[inline(always)]
pub unsafe fn gpu_hostcall_request_with_metadata(
    buf: *mut u8,
    service: u32,
    metadata: HostcallMetadata,
    fill_payload: impl FnOnce(*mut u8),
) -> Result<*mut u8, GpuError> {
    let request = submit_versioned_request(buf, service, metadata, fill_payload)?;
    finish_request_wait(buf, &request, GPU_MAX_SPIN)?;
    Ok(request.pkt)
}

/// Submit a hostcall request with a longer spin-wait timeout.
///
/// Identical to [`gpu_hostcall_request`] but uses `max_spin` iterations instead
/// of the default `GPU_MAX_SPIN`. Useful for blocking host operations like stdin
/// that may take longer to complete due to I/O thread routing.
#[inline(always)]
pub unsafe fn gpu_hostcall_request_with_timeout(
    buf: *mut u8,
    service: u32,
    max_spin: u32,
    fill_payload: impl FnOnce(*mut u8),
) -> Result<*mut u8, GpuError> {
    gpu_hostcall_request_with_timeout_and_metadata(
        buf,
        service,
        max_spin,
        HostcallMetadata::default(),
        fill_payload,
    )
}

/// Submit a hostcall request with a caller-specified timeout and task metadata.
#[inline(always)]
pub unsafe fn gpu_hostcall_request_with_timeout_and_metadata(
    buf: *mut u8,
    service: u32,
    max_spin: u32,
    metadata: HostcallMetadata,
    fill_payload: impl FnOnce(*mut u8),
) -> Result<*mut u8, GpuError> {
    let request = submit_versioned_request(buf, service, metadata, fill_payload)?;
    finish_request_wait(buf, &request, max_spin)?;
    Ok(request.pkt)
}

/// Send a PRINT hostcall with a short message (max 56 bytes).
/// Returns `Ok(())` on success, `Err(GpuError)` on pool exhaustion or timeout.
#[inline(always)]
pub unsafe fn gpu_hostcall_print(
    buf: *mut u8,
    msg: *const u8,
    msg_len: u32,
) -> Result<(), GpuError> {
    gpu_hostcall_print_with_metadata(buf, msg, msg_len, HostcallMetadata::default())
}

/// Send a PRINT hostcall with explicit task metadata.
#[inline(always)]
pub unsafe fn gpu_hostcall_print_with_metadata(
    buf: *mut u8,
    msg: *const u8,
    msg_len: u32,
    metadata: HostcallMetadata,
) -> Result<(), GpuError> {
    let request = submit_versioned_request(buf, SERVICE_PRINT, metadata, |payload| {
        core::ptr::write_volatile(payload as *mut u64, msg_len as u64);

        let copy_len = if msg_len > PRINT_MAX_MSG_LEN as u32 {
            PRINT_MAX_MSG_LEN as u32
        } else {
            msg_len
        };
        let dst = payload.add(8);
        let mut i = 0u32;
        while i < copy_len {
            core::ptr::write_volatile(dst.add(i as usize), *msg.add(i as usize));
            i += 1;
        }

        core::ptr::write_volatile(
            payload.add(64) as *mut u32,
            crate::nvptx_shim::block_idx_x(),
        );
        core::ptr::write_volatile(
            payload.add(68) as *mut u32,
            crate::nvptx_shim::thread_idx_x(),
        );
    })?;
    finish_request_wait(buf, &request, GPU_MAX_SPIN)?;
    hc_release_index(
        buf,
        request.pkt_idx,
        request.num_shards,
        request.shard_array_off,
    );

    Ok(())
}

/// Send a TRACE hostcall with a structured trace event.
///
/// Emits a trace event with thread/block/warp metadata and GPU timestamp.
/// Fire-and-forget: the host acknowledges but no response data is used.
///
/// # Arguments
/// - `buf`: hostcall buffer pointer
/// - `level`: trace level (TRACE_LEVEL_DEBUG/INFO/WARN/ERROR)
/// - `msg`: message bytes (max 48 bytes, truncated if longer)
/// - `msg_len`: message length
#[inline(always)]
pub unsafe fn gpu_hostcall_trace(
    buf: *mut u8,
    level: u8,
    msg: *const u8,
    msg_len: u32,
) -> Result<(), GpuError> {
    gpu_hostcall_trace_with_metadata(buf, level, msg, msg_len, HostcallMetadata::default())
}

/// Send a TRACE hostcall with explicit task metadata.
#[inline(always)]
pub unsafe fn gpu_hostcall_trace_with_metadata(
    buf: *mut u8,
    level: u8,
    msg: *const u8,
    msg_len: u32,
    metadata: HostcallMetadata,
) -> Result<(), GpuError> {
    let request = submit_versioned_request(buf, SERVICE_TRACE, metadata, |payload| {
        let thread_idx = crate::nvptx_shim::thread_idx_x() as u16;
        let block_idx = crate::nvptx_shim::block_idx_x() as u16;
        let lane = gpu_atomics::lane_id() as u16;
        let copy_len = if msg_len > TRACE_MAX_MSG_LEN as u32 {
            TRACE_MAX_MSG_LEN as u32
        } else {
            msg_len
        };
        let meta = encode_trace_metadata(thread_idx, block_idx, level, copy_len as u8, lane);
        core::ptr::write_volatile(payload as *mut u64, meta);

        let timestamp: u64;
        #[cfg(target_arch = "nvptx64")]
        {
            core::arch::asm!("mov.u64 {}, %clock64;", out(reg64) timestamp);
        }
        #[cfg(not(target_arch = "nvptx64"))]
        {
            timestamp = 0;
        }
        core::ptr::write_volatile(payload.add(8) as *mut u64, timestamp);

        let dst = payload.add(16);
        let mut i = 0u32;
        while i < copy_len {
            core::ptr::write_volatile(dst.add(i as usize), *msg.add(i as usize));
            i += 1;
        }
    })?;
    finish_request_wait(buf, &request, GPU_MAX_SPIN)?;
    hc_release_index(
        buf,
        request.pkt_idx,
        request.num_shards,
        request.shard_array_off,
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use core::task::{RawWaker, RawWakerVTable, Waker};

    unsafe fn noop_clone(data: *const ()) -> RawWaker {
        RawWaker::new(data, &NOOP_VTABLE)
    }
    unsafe fn noop(_data: *const ()) {}
    static NOOP_VTABLE: RawWakerVTable = RawWakerVTable::new(noop_clone, noop, noop, noop);

    fn test_waker() -> Waker {
        unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &NOOP_VTABLE)) }
    }

    fn test_buffer() -> std::vec::Vec<u64> {
        let size = buffer_size(1);
        let mut storage = std::vec![0u64; size.div_ceil(core::mem::size_of::<u64>())];
        let buf = storage.as_mut_ptr() as *mut u8;
        unsafe {
            core::ptr::write_volatile(
                buf.add(BUF_OFF_PROTOCOL_VERSION) as *mut u32,
                HOSTCALL_PROTOCOL_VERSION,
            );
            core::ptr::write_volatile(buf.add(BUF_OFF_NUM_PACKETS) as *mut u32, 1);
            core::ptr::write_volatile(buf.add(BUF_OFF_NUM_SHARDS) as *mut u32, 0);
            core::ptr::write_volatile(buf.add(BUF_OFF_PKTS_PER_SHARD) as *mut u32, 0);
            core::ptr::write_volatile(
                buf.add(BUF_OFF_SHARD_ARRAY_OFF) as *mut u32,
                BUFFER_HEADER_SIZE as u32,
            );
            core::ptr::write_volatile(buf.add(BUF_OFF_HIGH_RESERVED_PER_SHARD) as *mut u32, 0);
            core::ptr::write_volatile(buf.add(BUF_OFF_FREE_STACK) as *mut u64, make_tagged(0, 0));
            core::ptr::write_volatile(buf.add(BUF_OFF_HIGH_FREE_STACK) as *mut u64, null_tagged());
            core::ptr::write_volatile(buf.add(BUF_OFF_READY_STACK) as *mut u64, null_tagged());
            let pkt = buf.add(packet_offset(0));
            core::ptr::write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, null_tagged());
        }
        storage
    }

    #[test]
    fn unsubmitted_lease_drop_returns_packet_directly() {
        let mut storage = test_buffer();
        let buf = storage.as_mut_ptr() as *mut u8;
        let lease = unsafe {
            HostcallPacketLease::acquire(buf, HostcallMetadata::new(7, Priority::Low)).unwrap()
        };
        assert_eq!(lease.packet_index(), 0);
        assert_eq!(
            unsafe {
                tagged_index(core::ptr::read_volatile(
                    buf.add(BUF_OFF_FREE_STACK) as *const u64
                ))
            },
            NULL_INDEX
        );
        drop(lease);
        assert_eq!(
            unsafe {
                tagged_index(core::ptr::read_volatile(
                    buf.add(BUF_OFF_FREE_STACK) as *const u64
                ))
            },
            0
        );
    }

    #[test]
    fn async_request_first_poll_is_pending_then_ready_releases_once() {
        let mut storage = test_buffer();
        let buf = storage.as_mut_ptr() as *mut u8;
        let lease = unsafe {
            HostcallPacketLease::acquire(buf, HostcallMetadata::new(9, Priority::High)).unwrap()
        };
        let mut pending = lease.submit(SERVICE_PRIORITY_ECHO, [17; 8], 8);
        let packet = pending.packet.expect("packet");
        let waker = test_waker();
        let mut cx = Context::from_waker(&waker);
        assert!(Pin::new(&mut pending).poll(&mut cx).is_pending());
        assert_eq!(pending.pending_polls(), 1);

        unsafe {
            let payload = packet.pkt.add(PKT_OFF_PAYLOAD) as *mut u64;
            core::ptr::write_volatile(payload, 23);
            sys_store_release_u32(
                packet.pkt.add(PKT_OFF_CONTROL) as *mut u32,
                make_control(packet.generation, CONTROL_READY),
            );
        }
        let Poll::Ready(Ok(response)) = Pin::new(&mut pending).poll(&mut cx) else {
            panic!("ready completion expected");
        };
        assert_eq!(response.payload[0], 23);
        assert_eq!(response.pending_polls, 1);
        assert_eq!(
            unsafe {
                tagged_index(core::ptr::read_volatile(
                    buf.add(BUF_OFF_FREE_STACK) as *const u64
                ))
            },
            0
        );
        assert_eq!(
            Pin::new(&mut pending).poll(&mut cx),
            Poll::Ready(Err(GpuError::unsupported()))
        );
    }

    #[test]
    fn ready_error_and_ready_drop_both_return_device_owned_packet() {
        for drop_instead_of_poll in [false, true] {
            let mut storage = test_buffer();
            let buf = storage.as_mut_ptr() as *mut u8;
            let lease =
                unsafe { HostcallPacketLease::acquire(buf, HostcallMetadata::default()).unwrap() };
            let mut pending = lease.submit(SERVICE_PRIORITY_ECHO, [0; 8], 8);
            let packet = pending.packet.expect("packet");
            let waker = test_waker();
            let mut cx = Context::from_waker(&waker);
            assert!(Pin::new(&mut pending).poll(&mut cx).is_pending());
            unsafe {
                core::ptr::write_volatile(
                    packet.pkt.add(PKT_OFF_PAYLOAD) as *mut u64,
                    encode_error(ERR_OTHER, 0),
                );
                sys_store_release_u32(
                    packet.pkt.add(PKT_OFF_CONTROL) as *mut u32,
                    make_control(packet.generation, CONTROL_READY | CONTROL_ERROR),
                );
            }
            if drop_instead_of_poll {
                drop(pending);
            } else {
                assert_eq!(
                    Pin::new(&mut pending).poll(&mut cx),
                    Poll::Ready(Err(GpuError::new(ERR_OTHER, 0)))
                );
            }
            assert_eq!(
                unsafe {
                    tagged_index(core::ptr::read_volatile(
                        buf.add(BUF_OFF_FREE_STACK) as *const u64
                    ))
                },
                0
            );
        }
    }

    #[test]
    fn submitted_drop_transfers_filled_and_host_owned_packets_to_host() {
        for host_owned in [false, true] {
            let mut storage = test_buffer();
            let buf = storage.as_mut_ptr() as *mut u8;
            let lease =
                unsafe { HostcallPacketLease::acquire(buf, HostcallMetadata::default()).unwrap() };
            let mut pending = lease.submit(SERVICE_PRIORITY_ECHO, [0; 8], 8);
            let packet = pending.packet.expect("packet");
            let waker = test_waker();
            let mut cx = Context::from_waker(&waker);
            assert!(Pin::new(&mut pending).poll(&mut cx).is_pending());
            if host_owned {
                unsafe {
                    sys_store_release_u32(
                        packet.pkt.add(PKT_OFF_CONTROL) as *mut u32,
                        make_control(packet.generation, CONTROL_HOST_OWNED),
                    );
                }
            }
            drop(pending);
            let control =
                unsafe { sys_load_acquire_u32(packet.pkt.add(PKT_OFF_CONTROL) as *const u32) };
            assert_eq!(control_flags(control), CONTROL_CANCELLED);
            assert_eq!(
                unsafe {
                    tagged_index(core::ptr::read_volatile(
                        buf.add(BUF_OFF_FREE_STACK) as *const u64
                    ))
                },
                NULL_INDEX
            );
        }
    }
}
