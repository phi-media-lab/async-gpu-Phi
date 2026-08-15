//! Composed priority E2E 的独立 host oracle。
//!
//! 判定只消费 raw schema、host 独立事件与 quiescent pool audit；不读取
//! kernel 生成的 PASS 位。schema 损坏为 UNKNOWN，语义反例为 FAIL。

use gpu_host::hostcall::{composed_schema as schema, HostcallPoolAudit, PriorityEchoEvent};

// Independent semantic literals. Layout offsets may be shared with the
// producer, but the oracle must not inherit producer values that it verifies.
const EXPECTED_SCHEMA_VERSION: u64 = 3;
const EXPECTED_SCHEMA_WORDS: u64 = 112;
const EXPECTED_PHASE_COMPLETE: u64 = 4;
const EXPECTED_END_MAGIC: u64 = 0x0C05_ED0E_2F1A_1644;
const EXPECTED_NAMESPACE: u64 = 0x0B57_A11E;
const EXPECTED_HIGH_LOCAL_ID: u64 = 241;
const EXPECTED_HIGH_TASK_ID: u64 = (EXPECTED_NAMESPACE << 32) | EXPECTED_HIGH_LOCAL_ID;
const PRIORITY_NORMAL: u64 = 1;
const PRIORITY_HIGH: u64 = 2;
const ERR_INVALID_INPUT: u64 = 4;
const ERR_IO_ERROR: u64 = 5;
const ERR_RESOURCE_BUSY: u64 = 9;
const ACTION_NO_BRAKE: u64 = 0;
const ACTION_APPLY_BRAKE_FROM_FRESH_RESULT: u64 = 1;
const ACTION_WATCHDOG_CONSERVATIVE_STOP: u64 = 2;
const AGE_SOURCE_REAL_GPU_CLOCK: u64 = 1;
const AGE_SOURCE_INJECTED_STALE: u64 = 2;
const AGE_SOURCE_REAL_FIRST_POLL_LATENCY: u64 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExpectedRun {
    pub mutation: u32,
    pub nonce: u64,
}

#[derive(Clone, Copy)]
pub(crate) struct ComposedResults {
    pub words: [u64; schema::WORD_COUNT],
}

impl ComposedResults {
    pub unsafe fn read_from(host_ptr: *mut u64) -> Self {
        let mut words = [0u64; schema::WORD_COUNT];
        for (index, word) in words.iter_mut().enumerate() {
            *word = std::ptr::read_volatile(host_ptr.add(index));
        }
        Self { words }
    }

    pub fn get(&self, index: usize) -> u64 {
        self.words[index]
    }

    pub fn apply_host_audit(&mut self, event_count: usize, audit: &HostcallPoolAudit) {
        self.words[schema::HOST_EVENT_COUNT] = event_count as u64;
        self.words[schema::HOST_AUDIT_READY_EMPTY] = audit.ready_empty as u64;
        self.words[schema::HOST_AUDIT_IDLE_COUNT] = audit.idle_packets as u64;
        self.words[schema::HOST_AUDIT_GENERAL_MASK] = audit.general_mask;
        self.words[schema::HOST_AUDIT_HIGH_MASK] = audit.shared_high_mask;
        self.words[schema::HOST_AUDIT_DUPLICATES] = audit.duplicate_entries as u64;
        self.words[schema::HOST_AUDIT_MISSING] = audit.missing_packets as u64;
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum OracleVerdict {
    Pass,
    Fail(Vec<String>),
    Unknown(String),
}

/// Verify that a fault mode turned its designated independent gate red, rather
/// than merely causing an unrelated semantic failure elsewhere in the run.
pub(crate) fn required_mutation_gate_is_red(
    expected: ExpectedRun,
    results: &ComposedResults,
    events: &[PriorityEchoEvent],
    audit: &HostcallPoolAudit,
) -> bool {
    if results.get(schema::MUTATION_MODE) != expected.mutation as u64
        || results.get(schema::NONCE) != expected.nonce
        || results.get(schema::MUTATION_APPLIED) != expected.mutation as u64
    {
        return false;
    }

    let inject = results.get(schema::HIGH_INJECT_SEQUENCE);
    let first = results.get(schema::HIGH_FIRST_POLL_SEQUENCE);
    let bounded_high_entry = first > inject && (1..=6).contains(&(first - inject));
    let terminal_clean = results.get(schema::EXECUTOR_SPAWNED) == 241
        && results.get(schema::EXECUTOR_COMPLETED) == 241
        && results.get(schema::EXECUTOR_ACTIVE_FINAL) == 0
        && results.get(schema::LEASE_RELEASED) == 4
        && results.get(schema::NORMAL_RELEASED) == 16
        && results.get(schema::KERNEL_OBSERVATION_FLAGS) == 1;
    let audit_clean = audit.ready_empty
        && audit.idle_packets == 5
        && audit.general_mask == 0b1111
        && audit.shared_high_mask == 0b1_0000
        && audit.duplicate_entries == 0
        && audit.missing_packets == 0
        && audit.non_idle_controls == 0;
    let typed_join_clean = results.get(schema::JOIN_SUCCESS) == 1
        && results.get(schema::SECOND_JOIN_ALREADY_JOINED) == 1
        && results.get(schema::LOW_TOKEN_CONSUMED) == 1;
    let wire_identity_clean = results.get(schema::WIRE_TASK_ID) == EXPECTED_HIGH_TASK_ID
        && results.get(schema::WIRE_NAMESPACE) == EXPECTED_NAMESPACE
        && results.get(schema::WIRE_LOCAL_ID) == EXPECTED_HIGH_LOCAL_ID;
    let reserved_packet_clean = results.get(schema::GPU_PACKET_INDEX) == 4
        && results.get(schema::GPU_SHARED_HIGH_RESERVED) == 1;

    match expected.mutation {
        1 => {
            // One immutable High entry arms the raw probe; its self-wake then
            // appends behind all 16 live Normal holders. Meaningful first poll
            // therefore needs at least 1 + 16 + 1 dispatches after injection.
            first > inject
                && first - inject >= 18
                && results.get(schema::HIGH_PENDING_POLLS) == 0
                && results.get(schema::MANDATORY_FIRST_PENDING) == 0
                && results.get(schema::HIGH_SUBMIT_SEQUENCE) == 0
                && results.get(schema::HIGH_READY_SEQUENCE) == 0
                && results.get(schema::LOW_WAIT_START_SEQUENCE) == 0
                && results.get(schema::LOW_JOIN_SEQUENCE) == 0
                && results.get(schema::LOW_SECOND_JOIN_SEQUENCE) == 0
                && results.get(schema::WIRE_TASK_ID) == 0
                && results.get(schema::GPU_PACKET_INDEX) == 0
                && results.get(schema::ECHO_TASK_ID) == 0
                && results.get(schema::JOIN_SUCCESS) == 0
                && results.get(schema::LOW_TOKEN_CONSUMED) == 0
                && events.is_empty()
                && terminal_clean
                && audit_clean
        }
        2 => {
            bounded_high_entry
                && wire_identity_clean
                && results.get(schema::WIRE_PRIORITY) == PRIORITY_NORMAL
                && results.get(schema::HOST_ERROR) == ERR_RESOURCE_BUSY
                && results.get(schema::HIGH_SUBMIT_SEQUENCE) == 0
                && results.get(schema::HIGH_READY_SEQUENCE) == 0
                && results.get(schema::MANDATORY_FIRST_PENDING) == 0
                && results.get(schema::HIGH_PENDING_POLLS) == 0
                && results.get(schema::GPU_PACKET_INDEX) == 0
                && results.get(schema::GPU_SHARED_HIGH_RESERVED) == 0
                && events.is_empty()
                && typed_join_clean
                && terminal_clean
                && audit_clean
        }
        3 => {
            bounded_high_entry
                && results.get(schema::WIRE_TASK_ID) == EXPECTED_NAMESPACE << 32
                && results.get(schema::WIRE_NAMESPACE) == EXPECTED_NAMESPACE
                && results.get(schema::WIRE_LOCAL_ID) == 0
                && results.get(schema::WIRE_PRIORITY) == PRIORITY_HIGH
                && reserved_packet_clean
                && results.get(schema::MANDATORY_FIRST_PENDING) == 1
                && results.get(schema::HIGH_PENDING_POLLS) >= 1
                && results.get(schema::HIGH_SUBMIT_SEQUENCE) == first
                && results.get(schema::HIGH_READY_SEQUENCE) > first
                && results.get(schema::HOST_ERROR) == ERR_INVALID_INPUT
                && matches!(events, [event]
                if event.request_nonce == expected.nonce
                    && event.task_id == EXPECTED_NAMESPACE << 32
                    && event.namespace as u64 == EXPECTED_NAMESPACE
                    && event.local_id == 0
                    && event.priority.as_raw() as u64 == PRIORITY_HIGH
                    && event.packet_index == 4
                    && event.shared_high_reserved
                    && event.generation != 0
                    && event.process_count == 1
                    && event.error_category as u64 == ERR_INVALID_INPUT)
                && typed_join_clean
                && terminal_clean
                && audit_clean
        }
        4 => {
            bounded_high_entry
                && wire_identity_clean
                && results.get(schema::WIRE_PRIORITY) == PRIORITY_HIGH
                && reserved_packet_clean
                && results.get(schema::MANDATORY_FIRST_PENDING) == 1
                && results.get(schema::HIGH_PENDING_POLLS) >= 1
                && results.get(schema::HIGH_SUBMIT_SEQUENCE) == first
                && results.get(schema::HIGH_READY_SEQUENCE) > first
                && results.get(schema::HOST_ERROR) == ERR_IO_ERROR
                && matches!(events, [event]
                    if event.request_nonce == expected.nonce
                        && event.task_id == EXPECTED_HIGH_TASK_ID
                        && event.namespace as u64 == EXPECTED_NAMESPACE
                        && event.local_id as u64 == EXPECTED_HIGH_LOCAL_ID
                        && event.priority.as_raw() as u64 == PRIORITY_HIGH
                        && event.packet_index == 4
                        && event.shared_high_reserved
                        && event.generation != 0
                        && event.process_count == 1
                        && event.error_category as u64 == ERR_IO_ERROR)
                && typed_join_clean
                && terminal_clean
                && audit_clean
        }
        5 => {
            bounded_high_entry
                && wire_identity_clean
                && results.get(schema::WIRE_PRIORITY) == PRIORITY_HIGH
                && reserved_packet_clean
                && results.get(schema::MANDATORY_FIRST_PENDING) == 1
                && results.get(schema::HIGH_PENDING_POLLS) >= 1
                && results.get(schema::HIGH_SUBMIT_SEQUENCE) == first
                && results.get(schema::HIGH_READY_SEQUENCE) > first
                && results.get(schema::HOST_ERROR) == 0
                && results.get(schema::ECHO_NONCE) == expected.nonce
                && results.get(schema::ECHO_TASK_ID) == EXPECTED_HIGH_TASK_ID
                && results.get(schema::ECHO_PRIORITY) == PRIORITY_HIGH
                && results.get(schema::ECHO_PACKET_INDEX) == 4
                && results.get(schema::ECHO_SHARED_HIGH_RESERVED) == 1
                && results.get(schema::HOST_PROCESS_COUNT) == 1
                && matches!(events, [event]
                    if event.request_nonce == expected.nonce
                        && event.echo_nonce == expected.nonce
                        && event.task_id == EXPECTED_HIGH_TASK_ID
                        && event.namespace as u64 == EXPECTED_NAMESPACE
                        && event.local_id as u64 == EXPECTED_HIGH_LOCAL_ID
                        && event.priority.as_raw() as u64 == PRIORITY_HIGH
                        && event.packet_index == 4
                        && event.shared_high_reserved
                        && event.error_category == 0
                        && event.generation != 0
                        && event.process_count == 1
                        && results.get(schema::HOST_PROCESS_SEQUENCE) == event.process_sequence)
                && results.get(schema::JOIN_SUCCESS) == 1
                && results.get(schema::LOW_TOKEN_CONSUMED) == 1
                && results.get(schema::HIGH_OUTPUT) == EXPECTED_HIGH_TASK_ID
                && results.get(schema::LOW_JOIN_SEQUENCE)
                    > results.get(schema::LOW_WAIT_START_SEQUENCE)
                && results.get(schema::LOW_SECOND_JOIN_SEQUENCE)
                    > results.get(schema::LOW_JOIN_SEQUENCE)
                && results.get(schema::SECOND_JOIN_ALREADY_JOINED) == 0
                && terminal_clean
                && audit_clean
        }
        _ => false,
    }
}

fn expect_eq(failures: &mut Vec<String>, name: &str, actual: u64, expected: u64) {
    if actual != expected {
        failures.push(format!("{name}: expected={expected}, actual={actual}"));
    }
}

fn decision(results: &ComposedResults, which: usize, field: usize) -> u64 {
    results.get(schema::decision_word(which, field))
}

fn expect_age_consistent(
    failures: &mut Vec<String>,
    name: &str,
    results: &ComposedResults,
    which: usize,
) {
    let sample = decision(results, which, schema::DECISION_SAMPLE_TIMESTAMP);
    let now = decision(results, which, schema::DECISION_NOW_TIMESTAMP);
    let age = decision(results, which, schema::DECISION_AGE_TICKS);
    if sample == 0 || now < sample {
        failures.push(format!(
            "{name} timestamps invalid: sample={sample} now={now}"
        ));
    } else {
        expect_eq(failures, &format!("{name} derived age"), age, now - sample);
    }
}

pub(crate) fn evaluate_composed(
    expected: ExpectedRun,
    results: &ComposedResults,
    events: &[PriorityEchoEvent],
    audit: &HostcallPoolAudit,
) -> OracleVerdict {
    if results.get(schema::VERSION) != EXPECTED_SCHEMA_VERSION
        || results.get(schema::WORDS) != EXPECTED_SCHEMA_WORDS
        || results.get(schema::PHASE) != EXPECTED_PHASE_COMPLETE
        || results.get(schema::EXPECTED_END_MAGIC) != EXPECTED_END_MAGIC
        || results.get(schema::END_MAGIC) != EXPECTED_END_MAGIC
    {
        return OracleVerdict::Unknown(format!(
            "schema/header damaged: version={} words={} phase={} expected_magic={:#x} end={:#x}",
            results.get(schema::VERSION),
            results.get(schema::WORDS),
            results.get(schema::PHASE),
            results.get(schema::EXPECTED_END_MAGIC),
            results.get(schema::END_MAGIC),
        ));
    }

    let mut failures = Vec::new();
    let nonce = results.get(schema::NONCE);
    expect_eq(&mut failures, "input nonce", nonce, expected.nonce);
    expect_eq(
        &mut failures,
        "CLI mutation mode",
        results.get(schema::MUTATION_MODE),
        expected.mutation as u64,
    );
    expect_eq(
        &mut failures,
        "mutation hook reached its exact observation point",
        results.get(schema::MUTATION_APPLIED),
        expected.mutation as u64,
    );
    if expected.nonce == 0 {
        failures.push("host-provided expected nonce must be non-zero".to_owned());
    }

    // These are intentionally host literals, not values copied from the
    // kernel header. A coordinated schema corruption cannot move the oracle.
    expect_eq(
        &mut failures,
        "packet_count",
        results.get(schema::PACKET_COUNT),
        5,
    );
    expect_eq(
        &mut failures,
        "general_packet_count",
        results.get(schema::GENERAL_PACKET_COUNT),
        4,
    );
    expect_eq(
        &mut failures,
        "shared_high_reserved_count",
        results.get(schema::HIGH_RESERVED_COUNT),
        1,
    );
    expect_eq(
        &mut failures,
        "hard_low_limit",
        results.get(schema::HARD_LOW_LIMIT),
        224,
    );
    expect_eq(
        &mut failures,
        "hard_normal_backlog",
        results.get(schema::HARD_NORMAL_BACKLOG),
        16,
    );
    expect_eq(
        &mut failures,
        "High first-poll gap bound",
        results.get(schema::HIGH_FIRST_POLL_GAP_MAX),
        6,
    );
    expect_eq(
        &mut failures,
        "low_admitted",
        results.get(schema::LOW_ADMITTED),
        224,
    );
    expect_eq(
        &mut failures,
        "ReservedCapacity count",
        results.get(schema::RESERVED_CAPACITY_REJECTIONS),
        1,
    );
    expect_eq(
        &mut failures,
        "lease_count",
        results.get(schema::LEASE_ACQUIRED),
        4,
    );
    expect_eq(
        &mut failures,
        "lease_mask",
        results.get(schema::LEASE_MASK),
        0b1111,
    );
    expect_eq(
        &mut failures,
        "normal_alive",
        results.get(schema::NORMAL_ALIVE),
        16,
    );
    expect_eq(
        &mut failures,
        "general_pool_exhausted",
        results.get(schema::GENERAL_POOL_EXHAUSTED),
        1,
    );

    let inject = results.get(schema::HIGH_INJECT_SEQUENCE);
    let first = results.get(schema::HIGH_FIRST_POLL_SEQUENCE);
    let gap = first.saturating_sub(inject);
    if inject == 0 || first <= inject || !(1..=6).contains(&gap) {
        failures.push(format!(
            "High first-poll gap must be 1..=6, got inject={inject} first={first} gap={gap}"
        ));
    }
    let inject_timestamp = results.get(schema::HIGH_INJECT_TIMESTAMP);
    let first_timestamp = results.get(schema::HIGH_FIRST_POLL_TIMESTAMP);
    if inject_timestamp == 0 || first_timestamp < inject_timestamp {
        failures.push(format!(
            "High inject/first timestamps invalid: inject={inject_timestamp} first={first_timestamp}"
        ));
    }
    expect_eq(
        &mut failures,
        "mandatory first async poll returned Pending",
        results.get(schema::MANDATORY_FIRST_PENDING),
        1,
    );
    if results.get(schema::HIGH_PENDING_POLLS) < 1 {
        failures.push("async echo total Pending polls must be at least one".to_owned());
    }
    expect_eq(
        &mut failures,
        "High submit sequence",
        results.get(schema::HIGH_SUBMIT_SEQUENCE),
        first,
    );
    let ready_sequence = results.get(schema::HIGH_READY_SEQUENCE);
    if ready_sequence <= first {
        failures.push(format!(
            "High ready must follow submit: first={first} ready={ready_sequence}"
        ));
    }
    let wait_start = results.get(schema::LOW_WAIT_START_SEQUENCE);
    let join = results.get(schema::LOW_JOIN_SEQUENCE);
    let second_join = results.get(schema::LOW_SECOND_JOIN_SEQUENCE);
    if wait_start <= inject || join <= wait_start || second_join <= join {
        failures.push(format!(
            "typed wait/join timeline invalid: inject={inject} wait={wait_start} join={join} second={second_join}"
        ));
    }

    let expected_task_id = EXPECTED_HIGH_TASK_ID;
    expect_eq(
        &mut failures,
        "namespace",
        results.get(schema::NAMESPACE),
        EXPECTED_NAMESPACE,
    );
    expect_eq(
        &mut failures,
        "expected High local id header",
        results.get(schema::EXPECTED_HIGH_LOCAL_ID),
        EXPECTED_HIGH_LOCAL_ID,
    );
    expect_eq(
        &mut failures,
        "wire task id",
        results.get(schema::WIRE_TASK_ID),
        expected_task_id,
    );
    expect_eq(
        &mut failures,
        "wire namespace",
        results.get(schema::WIRE_NAMESPACE),
        EXPECTED_NAMESPACE,
    );
    expect_eq(
        &mut failures,
        "wire local id",
        results.get(schema::WIRE_LOCAL_ID),
        EXPECTED_HIGH_LOCAL_ID,
    );
    expect_eq(
        &mut failures,
        "wire priority",
        results.get(schema::WIRE_PRIORITY),
        PRIORITY_HIGH,
    );
    expect_eq(
        &mut failures,
        "GPU packet index",
        results.get(schema::GPU_PACKET_INDEX),
        4,
    );
    expect_eq(
        &mut failures,
        "GPU shared reserve provenance",
        results.get(schema::GPU_SHARED_HIGH_RESERVED),
        1,
    );
    expect_eq(
        &mut failures,
        "echo nonce",
        results.get(schema::ECHO_NONCE),
        nonce,
    );
    expect_eq(
        &mut failures,
        "echo task id",
        results.get(schema::ECHO_TASK_ID),
        expected_task_id,
    );
    expect_eq(
        &mut failures,
        "echo priority",
        results.get(schema::ECHO_PRIORITY),
        2,
    );
    expect_eq(
        &mut failures,
        "echo packet index",
        results.get(schema::ECHO_PACKET_INDEX),
        4,
    );
    expect_eq(
        &mut failures,
        "echo shared reserve provenance",
        results.get(schema::ECHO_SHARED_HIGH_RESERVED),
        1,
    );
    expect_eq(
        &mut failures,
        "echo process count",
        results.get(schema::HOST_PROCESS_COUNT),
        1,
    );
    expect_eq(
        &mut failures,
        "echo host error",
        results.get(schema::HOST_ERROR),
        0,
    );

    expect_eq(
        &mut failures,
        "join success",
        results.get(schema::JOIN_SUCCESS),
        1,
    );
    expect_eq(
        &mut failures,
        "second join AlreadyJoined",
        results.get(schema::SECOND_JOIN_ALREADY_JOINED),
        1,
    );
    expect_eq(
        &mut failures,
        "waiter local id",
        results.get(schema::WAITER_LOCAL_ID),
        224,
    );
    expect_eq(
        &mut failures,
        "join output identity",
        results.get(schema::HIGH_OUTPUT),
        expected_task_id,
    );
    expect_eq(
        &mut failures,
        "executor spawned",
        results.get(schema::EXECUTOR_SPAWNED),
        241,
    );
    expect_eq(
        &mut failures,
        "executor completed",
        results.get(schema::EXECUTOR_COMPLETED),
        241,
    );
    expect_eq(
        &mut failures,
        "executor active final",
        results.get(schema::EXECUTOR_ACTIVE_FINAL),
        0,
    );
    expect_eq(
        &mut failures,
        "lease release",
        results.get(schema::LEASE_RELEASED),
        4,
    );
    expect_eq(
        &mut failures,
        "normal release",
        results.get(schema::NORMAL_RELEASED),
        16,
    );
    expect_eq(
        &mut failures,
        "Low token consumed",
        results.get(schema::LOW_TOKEN_CONSUMED),
        1,
    );
    expect_eq(
        &mut failures,
        "kernel observation flags",
        results.get(schema::KERNEL_OBSERVATION_FLAGS),
        1,
    );

    expect_eq(&mut failures, "host event count", events.len() as u64, 1);
    if let [event] = events {
        expect_eq(
            &mut failures,
            "host request nonce",
            event.request_nonce,
            nonce,
        );
        expect_eq(&mut failures, "host echo nonce", event.echo_nonce, nonce);
        expect_eq(
            &mut failures,
            "host task id",
            event.task_id,
            expected_task_id,
        );
        expect_eq(
            &mut failures,
            "host namespace",
            event.namespace as u64,
            EXPECTED_NAMESPACE,
        );
        expect_eq(
            &mut failures,
            "host local id",
            event.local_id as u64,
            EXPECTED_HIGH_LOCAL_ID,
        );
        expect_eq(
            &mut failures,
            "host priority",
            event.priority.as_raw() as u64,
            PRIORITY_HIGH,
        );
        expect_eq(
            &mut failures,
            "host packet index",
            event.packet_index as u64,
            4,
        );
        expect_eq(
            &mut failures,
            "host shared reserve provenance",
            event.shared_high_reserved as u64,
            1,
        );
        if event.generation == 0 {
            failures.push("host event generation must be non-zero v3 ownership".to_owned());
        }
        if event.process_sequence == 0 {
            failures.push("host process sequence must be non-zero".to_owned());
        }
        expect_eq(&mut failures, "host process count", event.process_count, 1);
        expect_eq(
            &mut failures,
            "host error event",
            event.error_category as u64,
            0,
        );
        expect_eq(
            &mut failures,
            "echo/host process sequence",
            results.get(schema::HOST_PROCESS_SEQUENCE),
            event.process_sequence,
        );
    }

    expect_eq(
        &mut failures,
        "audit ready empty",
        audit.ready_empty as u64,
        1,
    );
    expect_eq(
        &mut failures,
        "audit idle packets",
        audit.idle_packets as u64,
        5,
    );
    expect_eq(
        &mut failures,
        "audit general mask",
        audit.general_mask,
        0b1111,
    );
    expect_eq(
        &mut failures,
        "audit High mask",
        audit.shared_high_mask,
        1 << 4,
    );
    expect_eq(
        &mut failures,
        "audit duplicates",
        audit.duplicate_entries as u64,
        0,
    );
    expect_eq(
        &mut failures,
        "audit missing",
        audit.missing_packets as u64,
        0,
    );
    expect_eq(
        &mut failures,
        "audit non-idle controls",
        audit.non_idle_controls as u64,
        0,
    );

    let fresh_hazard = schema::DECISION_FRESH_HAZARD;
    let fresh_safe = schema::DECISION_FRESH_SAFE;
    let stale = schema::DECISION_STALE;
    let deadline = schema::DECISION_DEADLINE;
    for (which, kind, hazard, name) in [
        (fresh_hazard, 0, 1, "fresh hazard"),
        (fresh_safe, 1, 0, "fresh safe"),
        (stale, 2, 1, "stale"),
        (deadline, 3, 1, "deadline"),
    ] {
        expect_eq(
            &mut failures,
            &format!("{name} kind"),
            decision(results, which, schema::DECISION_KIND),
            kind,
        );
        expect_eq(
            &mut failures,
            &format!("{name} hazard"),
            decision(results, which, schema::DECISION_HAZARD),
            hazard,
        );
        expect_age_consistent(&mut failures, name, results, which);
    }
    expect_eq(
        &mut failures,
        "fresh hazard action",
        decision(results, fresh_hazard, schema::DECISION_ACTION),
        ACTION_APPLY_BRAKE_FROM_FRESH_RESULT,
    );
    expect_eq(
        &mut failures,
        "fresh hazard uses GPU",
        decision(results, fresh_hazard, schema::DECISION_USE_GPU_RESULT),
        1,
    );
    expect_eq(
        &mut failures,
        "fresh safe action",
        decision(results, fresh_safe, schema::DECISION_ACTION),
        ACTION_NO_BRAKE,
    );
    expect_eq(
        &mut failures,
        "fresh safe uses GPU",
        decision(results, fresh_safe, schema::DECISION_USE_GPU_RESULT),
        1,
    );
    for which in [fresh_hazard, fresh_safe] {
        expect_eq(
            &mut failures,
            "fresh real age source",
            decision(results, which, schema::DECISION_AGE_SOURCE),
            AGE_SOURCE_REAL_GPU_CLOCK,
        );
        if decision(results, which, schema::DECISION_AGE_TICKS)
            > decision(results, which, schema::DECISION_BUDGET_TICKS)
        {
            failures.push("fresh decision age exceeds budget".to_owned());
        }
        expect_eq(
            &mut failures,
            "fresh sample is real High first timestamp",
            decision(results, which, schema::DECISION_SAMPLE_TIMESTAMP),
            results.get(schema::HIGH_FIRST_POLL_TIMESTAMP),
        );
        expect_eq(
            &mut failures,
            "fresh now is real High ready timestamp",
            decision(results, which, schema::DECISION_NOW_TIMESTAMP),
            results.get(schema::HIGH_READY_TIMESTAMP),
        );
        expect_eq(
            &mut failures,
            "fresh budget matches header",
            decision(results, which, schema::DECISION_BUDGET_TICKS),
            results.get(schema::FRESHNESS_BUDGET_TICKS),
        );
        expect_eq(
            &mut failures,
            "fresh budget is post-hoc observed age plus one",
            decision(results, which, schema::DECISION_BUDGET_TICKS),
            decision(results, which, schema::DECISION_AGE_TICKS).saturating_add(1),
        );
    }
    expect_eq(
        &mut failures,
        "stale watchdog action",
        decision(results, stale, schema::DECISION_ACTION),
        ACTION_WATCHDOG_CONSERVATIVE_STOP,
    );
    expect_eq(
        &mut failures,
        "stale must discard GPU result",
        decision(results, stale, schema::DECISION_USE_GPU_RESULT),
        0,
    );
    expect_eq(
        &mut failures,
        "stale injection source",
        decision(results, stale, schema::DECISION_AGE_SOURCE),
        AGE_SOURCE_INJECTED_STALE,
    );
    if decision(results, stale, schema::DECISION_AGE_TICKS)
        <= decision(results, stale, schema::DECISION_BUDGET_TICKS)
    {
        failures.push("stale injected age must exceed budget".to_owned());
    }
    expect_eq(
        &mut failures,
        "stale now is real High ready timestamp",
        decision(results, stale, schema::DECISION_NOW_TIMESTAMP),
        results.get(schema::HIGH_READY_TIMESTAMP),
    );
    expect_eq(
        &mut failures,
        "stale budget matches freshness header",
        decision(results, stale, schema::DECISION_BUDGET_TICKS),
        results.get(schema::FRESHNESS_BUDGET_TICKS),
    );
    expect_eq(
        &mut failures,
        "deadline watchdog action",
        decision(results, deadline, schema::DECISION_ACTION),
        ACTION_WATCHDOG_CONSERVATIVE_STOP,
    );
    expect_eq(
        &mut failures,
        "deadline must discard GPU result",
        decision(results, deadline, schema::DECISION_USE_GPU_RESULT),
        0,
    );
    expect_eq(
        &mut failures,
        "deadline real latency source",
        decision(results, deadline, schema::DECISION_AGE_SOURCE),
        AGE_SOURCE_REAL_FIRST_POLL_LATENCY,
    );
    if decision(results, deadline, schema::DECISION_AGE_TICKS)
        <= decision(results, deadline, schema::DECISION_BUDGET_TICKS)
    {
        failures.push("deadline latency must exceed injected smaller threshold".to_owned());
    }
    expect_eq(
        &mut failures,
        "deadline now is real High first timestamp",
        decision(results, deadline, schema::DECISION_NOW_TIMESTAMP),
        results.get(schema::HIGH_FIRST_POLL_TIMESTAMP),
    );
    expect_eq(
        &mut failures,
        "deadline sample is persistent High inject timestamp",
        decision(results, deadline, schema::DECISION_SAMPLE_TIMESTAMP),
        results.get(schema::HIGH_INJECT_TIMESTAMP),
    );
    expect_eq(
        &mut failures,
        "deadline budget matches header",
        decision(results, deadline, schema::DECISION_BUDGET_TICKS),
        results.get(schema::DEADLINE_BUDGET_TICKS),
    );
    expect_eq(
        &mut failures,
        "deadline budget is post-hoc observed latency minus one",
        decision(results, deadline, schema::DECISION_BUDGET_TICKS),
        decision(results, deadline, schema::DECISION_AGE_TICKS).saturating_sub(1),
    );

    if failures.is_empty() {
        OracleVerdict::Pass
    } else {
        OracleVerdict::Fail(failures)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::hostcall::EchoPriority;

    fn expected_run(mutation: u32) -> ExpectedRun {
        ExpectedRun {
            mutation,
            nonce: 77,
        }
    }

    fn valid_fixture() -> (ComposedResults, Vec<PriorityEchoEvent>, HostcallPoolAudit) {
        let mut results = ComposedResults {
            words: [0; schema::WORD_COUNT],
        };
        let w = &mut results.words;
        w[schema::VERSION] = EXPECTED_SCHEMA_VERSION;
        w[schema::WORDS] = 112;
        w[schema::PHASE] = 4;
        w[schema::NONCE] = 77;
        w[schema::NAMESPACE] = 0x0B57_A11E;
        w[schema::EXPECTED_HIGH_LOCAL_ID] = 241;
        w[schema::PACKET_COUNT] = 5;
        w[schema::GENERAL_PACKET_COUNT] = 4;
        w[schema::HIGH_RESERVED_COUNT] = 1;
        w[schema::HARD_LOW_LIMIT] = 224;
        w[schema::HARD_NORMAL_BACKLOG] = 16;
        w[schema::HIGH_FIRST_POLL_GAP_MAX] = 6;
        w[schema::FRESHNESS_BUDGET_TICKS] = 11;
        w[schema::DEADLINE_BUDGET_TICKS] = 7;
        w[schema::EXPECTED_END_MAGIC] = EXPECTED_END_MAGIC;
        w[schema::END_MAGIC] = EXPECTED_END_MAGIC;
        w[schema::LOW_ADMITTED] = 224;
        w[schema::RESERVED_CAPACITY_REJECTIONS] = 1;
        w[schema::LEASE_ACQUIRED] = 4;
        w[schema::LEASE_MASK] = 0b1111;
        w[schema::NORMAL_ALIVE] = 16;
        w[schema::GENERAL_POOL_EXHAUSTED] = 1;
        w[schema::HIGH_INJECT_SEQUENCE] = 100;
        w[schema::HIGH_FIRST_POLL_SEQUENCE] = 106;
        w[schema::HIGH_SUBMIT_SEQUENCE] = 106;
        w[schema::HIGH_READY_SEQUENCE] = 110;
        w[schema::LOW_WAIT_START_SEQUENCE] = 107;
        w[schema::LOW_JOIN_SEQUENCE] = 111;
        w[schema::LOW_SECOND_JOIN_SEQUENCE] = 112;
        w[schema::HIGH_INJECT_TIMESTAMP] = 992;
        w[schema::HIGH_FIRST_POLL_TIMESTAMP] = 1_000;
        w[schema::HIGH_READY_TIMESTAMP] = 1_010;
        w[schema::MANDATORY_FIRST_PENDING] = 1;
        w[schema::HIGH_PENDING_POLLS] = 1;
        let task_id = (0x0B57_A11E_u64 << 32) | 241;
        w[schema::WIRE_TASK_ID] = task_id;
        w[schema::WIRE_NAMESPACE] = 0x0B57_A11E;
        w[schema::WIRE_LOCAL_ID] = 241;
        w[schema::WIRE_PRIORITY] = 2;
        w[schema::GPU_PACKET_INDEX] = 4;
        w[schema::GPU_SHARED_HIGH_RESERVED] = 1;
        w[schema::ECHO_NONCE] = 77;
        w[schema::ECHO_TASK_ID] = task_id;
        w[schema::ECHO_PRIORITY] = 2;
        w[schema::ECHO_PACKET_INDEX] = 4;
        w[schema::ECHO_SHARED_HIGH_RESERVED] = 1;
        w[schema::HOST_PROCESS_SEQUENCE] = 1;
        w[schema::HOST_PROCESS_COUNT] = 1;
        w[schema::HOST_ERROR] = 0;
        w[schema::JOIN_SUCCESS] = 1;
        w[schema::SECOND_JOIN_ALREADY_JOINED] = 1;
        w[schema::WAITER_LOCAL_ID] = 224;
        w[schema::EXECUTOR_SPAWNED] = 241;
        w[schema::EXECUTOR_COMPLETED] = 241;
        w[schema::EXECUTOR_ACTIVE_FINAL] = 0;
        w[schema::LOW_TOKEN_CONSUMED] = 1;
        w[schema::HIGH_OUTPUT] = task_id;
        w[schema::LEASE_RELEASED] = 4;
        w[schema::NORMAL_RELEASED] = 16;
        w[schema::KERNEL_OBSERVATION_FLAGS] = 1;
        w[schema::MUTATION_MODE] = 0;
        w[schema::MUTATION_APPLIED] = 0;

        let mut set_decision =
            |which, action, use_gpu, source, sample, now, age, budget, hazard| {
                w[schema::decision_word(which, schema::DECISION_KIND)] = which as u64;
                w[schema::decision_word(which, schema::DECISION_ACTION)] = action;
                w[schema::decision_word(which, schema::DECISION_USE_GPU_RESULT)] = use_gpu;
                w[schema::decision_word(which, schema::DECISION_AGE_SOURCE)] = source;
                w[schema::decision_word(which, schema::DECISION_SAMPLE_TIMESTAMP)] = sample;
                w[schema::decision_word(which, schema::DECISION_NOW_TIMESTAMP)] = now;
                w[schema::decision_word(which, schema::DECISION_AGE_TICKS)] = age;
                w[schema::decision_word(which, schema::DECISION_BUDGET_TICKS)] = budget;
                w[schema::decision_word(which, schema::DECISION_HAZARD)] = hazard;
            };
        set_decision(
            schema::DECISION_FRESH_HAZARD,
            ACTION_APPLY_BRAKE_FROM_FRESH_RESULT,
            1,
            AGE_SOURCE_REAL_GPU_CLOCK,
            1_000,
            1_010,
            10,
            11,
            1,
        );
        set_decision(
            schema::DECISION_FRESH_SAFE,
            ACTION_NO_BRAKE,
            1,
            AGE_SOURCE_REAL_GPU_CLOCK,
            1_000,
            1_010,
            10,
            11,
            0,
        );
        set_decision(
            schema::DECISION_STALE,
            ACTION_WATCHDOG_CONSERVATIVE_STOP,
            0,
            AGE_SOURCE_INJECTED_STALE,
            998,
            1_010,
            12,
            11,
            1,
        );
        set_decision(
            schema::DECISION_DEADLINE,
            ACTION_WATCHDOG_CONSERVATIVE_STOP,
            0,
            AGE_SOURCE_REAL_FIRST_POLL_LATENCY,
            992,
            1_000,
            8,
            7,
            1,
        );

        let event = PriorityEchoEvent {
            request_nonce: 77,
            echo_nonce: 77,
            task_id,
            namespace: 0x0B57_A11E,
            local_id: 241,
            priority: EchoPriority::High,
            packet_index: 4,
            generation: 1,
            shared_high_reserved: true,
            process_sequence: 1,
            process_count: 1,
            error_category: 0,
        };
        let audit = HostcallPoolAudit {
            ready_empty: true,
            idle_packets: 5,
            general_mask: 0b1111,
            shared_high_mask: 1 << 4,
            duplicate_entries: 0,
            missing_packets: 0,
            non_idle_controls: 0,
        };
        (results, vec![event], audit)
    }

    fn mutation_fixture(
        mutation: u32,
    ) -> (ComposedResults, Vec<PriorityEchoEvent>, HostcallPoolAudit) {
        let (mut results, mut events, audit) = valid_fixture();
        results.words[schema::MUTATION_MODE] = mutation as u64;
        results.words[schema::MUTATION_APPLIED] = mutation as u64;
        match mutation {
            1 => {
                results.words[schema::HIGH_FIRST_POLL_SEQUENCE] = 118;
                results.words[schema::HIGH_PENDING_POLLS] = 0;
                results.words[schema::MANDATORY_FIRST_PENDING] = 0;
                results.words[schema::HIGH_SUBMIT_SEQUENCE] = 0;
                results.words[schema::HIGH_READY_SEQUENCE] = 0;
                results.words[schema::LOW_WAIT_START_SEQUENCE] = 0;
                results.words[schema::LOW_JOIN_SEQUENCE] = 0;
                results.words[schema::LOW_SECOND_JOIN_SEQUENCE] = 0;
                results.words[schema::WIRE_TASK_ID] = 0;
                results.words[schema::GPU_PACKET_INDEX] = 0;
                results.words[schema::ECHO_TASK_ID] = 0;
                results.words[schema::JOIN_SUCCESS] = 0;
                results.words[schema::LOW_TOKEN_CONSUMED] = 0;
                events.clear();
            }
            2 => {
                results.words[schema::WIRE_PRIORITY] = PRIORITY_NORMAL;
                results.words[schema::HOST_ERROR] = ERR_RESOURCE_BUSY;
                results.words[schema::HIGH_SUBMIT_SEQUENCE] = 0;
                results.words[schema::HIGH_READY_SEQUENCE] = 0;
                results.words[schema::HIGH_PENDING_POLLS] = 0;
                results.words[schema::MANDATORY_FIRST_PENDING] = 0;
                results.words[schema::GPU_PACKET_INDEX] = 0;
                results.words[schema::GPU_SHARED_HIGH_RESERVED] = 0;
                events.clear();
            }
            3 => {
                results.words[schema::WIRE_TASK_ID] = EXPECTED_NAMESPACE << 32;
                results.words[schema::WIRE_LOCAL_ID] = 0;
                results.words[schema::HOST_ERROR] = ERR_INVALID_INPUT;
                events[0].task_id = EXPECTED_NAMESPACE << 32;
                events[0].namespace = EXPECTED_NAMESPACE as u32;
                events[0].local_id = 0;
                events[0].error_category = ERR_INVALID_INPUT as u16;
            }
            4 => {
                results.words[schema::HOST_ERROR] = ERR_IO_ERROR;
                events[0].error_category = ERR_IO_ERROR as u16;
            }
            5 => results.words[schema::SECOND_JOIN_ALREADY_JOINED] = 0,
            _ => unreachable!(),
        }
        (results, events, audit)
    }

    #[test]
    fn valid_fixture_passes_without_kernel_pass_bit() {
        let (results, events, audit) = valid_fixture();
        assert_eq!(
            evaluate_composed(expected_run(0), &results, &events, &audit),
            OracleVerdict::Pass
        );
    }

    #[test]
    fn schema_damage_is_unknown() {
        let (mut results, events, audit) = valid_fixture();
        results.words[schema::WORDS] = 111;
        assert!(matches!(
            evaluate_composed(expected_run(0), &results, &events, &audit),
            OracleVerdict::Unknown(_)
        ));
    }

    #[test]
    fn forged_safety_timestamp_is_a_semantic_failure() {
        let (mut results, events, audit) = valid_fixture();
        results.words[schema::decision_word(
            schema::DECISION_FRESH_HAZARD,
            schema::DECISION_SAMPLE_TIMESTAMP,
        )] -= 1;
        assert!(matches!(
            evaluate_composed(expected_run(0), &results, &events, &audit),
            OracleVerdict::Fail(_)
        ));
    }

    #[test]
    fn persistent_inject_and_mandatory_first_pending_are_independent_gates() {
        let (mut forged_inject, events, audit) = valid_fixture();
        forged_inject.words[schema::HIGH_INJECT_TIMESTAMP] += 1;
        assert!(matches!(
            evaluate_composed(expected_run(0), &forged_inject, &events, &audit),
            OracleVerdict::Fail(_)
        ));

        let (mut no_first_pending, events, audit) = valid_fixture();
        no_first_pending.words[schema::MANDATORY_FIRST_PENDING] = 0;
        assert!(matches!(
            evaluate_composed(expected_run(0), &no_first_pending, &events, &audit),
            OracleVerdict::Fail(_)
        ));
    }

    #[test]
    fn zero_generation_event_cannot_masquerade_as_live_v3() {
        let (results, mut events, audit) = valid_fixture();
        events[0].generation = 0;
        assert!(matches!(
            evaluate_composed(expected_run(0), &results, &events, &audit),
            OracleVerdict::Fail(_)
        ));
    }

    #[test]
    fn unrelated_failure_does_not_count_as_a_killed_mutation() {
        let (mut results, events, audit) = valid_fixture();
        results.words[schema::KERNEL_OBSERVATION_FLAGS] = 0;
        assert!(matches!(
            evaluate_composed(expected_run(0), &results, &events, &audit),
            OracleVerdict::Fail(_)
        ));
        assert!(!required_mutation_gate_is_red(
            expected_run(1),
            &results,
            &events,
            &audit,
        ));
    }

    #[test]
    fn gap_above_positive_bound_but_before_fifo_probe_is_not_designated_kill() {
        let (mut results, events, audit) = valid_fixture();
        results.words[schema::HIGH_FIRST_POLL_SEQUENCE] = 107;
        assert!(matches!(
            evaluate_composed(expected_run(0), &results, &events, &audit),
            OracleVerdict::Fail(_)
        ));
        assert!(!required_mutation_gate_is_red(
            expected_run(1),
            &results,
            &events,
            &audit,
        ));
    }

    #[test]
    fn expected_nonce_and_cli_mode_are_independent_inputs() {
        let (results, events, audit) = valid_fixture();
        assert!(matches!(
            evaluate_composed(
                ExpectedRun {
                    mutation: 0,
                    nonce: 78,
                },
                &results,
                &events,
                &audit,
            ),
            OracleVerdict::Fail(_)
        ));
        assert!(matches!(
            evaluate_composed(expected_run(1), &results, &events, &audit),
            OracleVerdict::Fail(_)
        ));
    }

    #[test]
    fn default_and_wrong_signatures_cannot_fake_designated_kills() {
        let (_, _, audit) = valid_fixture();
        let zero = ComposedResults {
            words: [0; schema::WORD_COUNT],
        };
        for mutation in 1..=5 {
            assert!(!required_mutation_gate_is_red(
                expected_run(mutation),
                &zero,
                &[],
                &audit,
            ));
        }

        let (mut mode1, events1, audit1) = mutation_fixture(1);
        mode1.words[schema::HIGH_FIRST_POLL_SEQUENCE] = 0;
        assert!(!required_mutation_gate_is_red(
            expected_run(1),
            &mode1,
            &events1,
            &audit1,
        ));

        let (mut mode2, events2, audit2) = mutation_fixture(2);
        mode2.words[schema::WIRE_PRIORITY] = 0;
        assert!(!required_mutation_gate_is_red(
            expected_run(2),
            &mode2,
            &events2,
            &audit2,
        ));

        let (mut mode3, events3, audit3) = mutation_fixture(3);
        mode3.words[schema::MUTATION_APPLIED] = 0;
        assert!(!required_mutation_gate_is_red(
            expected_run(3),
            &mode3,
            &events3,
            &audit3,
        ));

        let (mut mode4, mut events4, audit4) = mutation_fixture(4);
        mode4.words[schema::HOST_ERROR] = ERR_INVALID_INPUT;
        events4[0].error_category = ERR_INVALID_INPUT as u16;
        assert!(!required_mutation_gate_is_red(
            expected_run(4),
            &mode4,
            &events4,
            &audit4,
        ));

        let (mut mode5, events5, audit5) = mutation_fixture(5);
        mode5.words[schema::JOIN_SUCCESS] = 0;
        assert!(!required_mutation_gate_is_red(
            expected_run(5),
            &mode5,
            &events5,
            &audit5,
        ));
    }

    #[test]
    fn every_required_fault_mutation_turns_its_gate_red() {
        for mutation in 1..=5 {
            let (results, events, audit) = mutation_fixture(mutation);
            assert!(
                matches!(
                    evaluate_composed(expected_run(mutation), &results, &events, &audit),
                    OracleVerdict::Fail(_)
                ),
                "mutation {mutation} unexpectedly survived"
            );
            assert!(
                required_mutation_gate_is_red(expected_run(mutation), &results, &events, &audit,),
                "mutation {mutation} failed without turning its designated gate red"
            );
        }
    }
}
