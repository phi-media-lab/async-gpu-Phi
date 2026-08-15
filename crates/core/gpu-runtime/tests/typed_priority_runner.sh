#!/usr/bin/env bash
# Compile-pass/fail coverage for the real PriorityToken::wait_for API.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
FAIL=0
PASS=0

TEST_TMP=$(mktemp -d)
trap 'rm -rf "$TEST_TMP"' EXIT
mkdir -p "$TEST_TMP/src"

GPU_RUNTIME_ABS="$(cd crates/core/gpu-runtime && pwd)"
cat > "$TEST_TMP/Cargo.toml" <<TOML
[workspace]

[package]
name = "typed-priority-compile-tests"
version = "0.0.0"
edition = "2021"

[lib]
path = "src/lib.rs"

[dependencies]
gpu-runtime = { path = "$GPU_RUNTIME_ABS" }
TOML

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_DIR="$SCRIPT_DIR/typed_priority"

expect_pass() {
    local name="$1"
    local file="$2"
    cp "$file" "$TEST_TMP/src/lib.rs"
    if cargo +stable check --quiet --manifest-path "$TEST_TMP/Cargo.toml"; then
        echo -e "  ${GREEN}OK${NC}   $name"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}FAIL${NC} $name (expected compilation success)"
        FAIL=$((FAIL + 1))
    fi
}

render_json_errors() {
    local log="$1"
    jq -c '
        select(.reason == "compiler-message")
        | .message
        | select(.level == "error")
        | {
            code: .code.code,
            message,
            primary: [.spans[] | select(.is_primary)]
        }
    ' "$log"
    jq -r '
        select(.reason == "compiler-message")
        | .message
        | select(.level == "error")
        | .rendered
    ' "$log" | sed -n '1,120p'
}

wait_diagnostic_matches() {
    local log="$1"
    local wait_line="$2"
    jq -se --argjson line "$wait_line" '
        .[] |
        select(.reason == "compiler-message")
        | .message
        | select(.code.code == "E0277")
        | select(.message | contains("MayWaitFor"))
        | select(any(
            .spans[];
            .is_primary
            and (.file_name | endswith("src/lib.rs"))
            and .line_start == $line
            and any(.text[]; .text | contains(".wait_for::<"))
        ))
    ' "$log" > /dev/null
}

forge_diagnostic_matches() {
    local log="$1"
    local forge_line="$2"
    jq -se --argjson line "$forge_line" '
        .[] |
        select(.reason == "compiler-message")
        | .message
        | select(.code.code == "E0599")
        | select(.message | contains("PriorityToken"))
        | select(any(
            .spans[];
            .is_primary
            and (.file_name | endswith("src/lib.rs"))
            and .line_start == $line
            and any(.text[]; .text | contains("PriorityToken::<High>::from_spawn"))
        ))
    ' "$log" > /dev/null
}

expect_wait_rejected() {
    local name="$1"
    local file="$2"
    local log="$TEST_TMP/$name.json"
    local cargo_stderr="$TEST_TMP/$name.cargo.stderr"
    local wait_line
    wait_line="$(grep -n '\.wait_for' "$file" | head -n 1 | cut -d: -f1)"
    cp "$file" "$TEST_TMP/src/lib.rs"
    if cargo +stable check --quiet --message-format=json \
        --manifest-path "$TEST_TMP/Cargo.toml" > "$log" 2> "$cargo_stderr"; then
        echo -e "  ${RED}FAIL${NC} $name (forbidden wait compiled)"
        FAIL=$((FAIL + 1))
    elif wait_diagnostic_matches "$log" "$wait_line"; then
        echo -e "  ${GREEN}OK${NC}   $name (rejected by MayWaitFor)"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}FAIL${NC} $name (failed for an unrelated reason)"
        render_json_errors "$log"
        sed -n '1,120p' "$cargo_stderr"
        FAIL=$((FAIL + 1))
    fi
}

expect_private_token_rejected() {
    local name="$1"
    local file="$2"
    local log="$TEST_TMP/$name.json"
    local cargo_stderr="$TEST_TMP/$name.cargo.stderr"
    local forge_line
    forge_line="$(grep -n 'from_spawn' "$file" | tail -n 1 | cut -d: -f1)"
    cp "$file" "$TEST_TMP/src/lib.rs"
    if cargo +stable check --quiet --message-format=json \
        --manifest-path "$TEST_TMP/Cargo.toml" > "$log" 2> "$cargo_stderr"; then
        echo -e "  ${RED}FAIL${NC} $name (forged token compiled)"
        FAIL=$((FAIL + 1))
    elif forge_diagnostic_matches "$log" "$forge_line"; then
        echo -e "  ${GREEN}OK${NC}   $name (token constructor is private)"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}FAIL${NC} $name (failed for an unrelated reason)"
        render_json_errors "$log"
        sed -n '1,120p' "$cargo_stderr"
        FAIL=$((FAIL + 1))
    fi
}

expect_forge_receiver_mutation_rejected() {
    local file="$1"
    local log="$TEST_TMP/Forge_receiver_mutation.json"
    local cargo_stderr="$TEST_TMP/Forge_receiver_mutation.cargo.stderr"
    local forge_line
    forge_line="$(grep -n 'from_spawn' "$file" | tail -n 1 | cut -d: -f1)"
    sed 's/PriorityToken::<High>/String/' "$file" > "$TEST_TMP/src/lib.rs"
    if cargo +stable check --quiet --message-format=json \
        --manifest-path "$TEST_TMP/Cargo.toml" > "$log" 2> "$cargo_stderr"; then
        echo -e "  ${RED}FAIL${NC} Forge_receiver_mutation (mutation unexpectedly compiled)"
        FAIL=$((FAIL + 1))
    elif forge_diagnostic_matches "$log" "$forge_line"; then
        echo -e "  ${RED}FAIL${NC} Forge_receiver_mutation (wrong receiver fooled gate)"
        FAIL=$((FAIL + 1))
    else
        echo -e "  ${GREEN}OK${NC}   Forge_receiver_mutation (unrelated E0599 rejected)"
        PASS=$((PASS + 1))
    fi
}

expect_nvptx_pass() {
    local name="$1"
    local file="$2"
    cp "$file" "$TEST_TMP/src/lib.rs"
    if cargo check --quiet --lib --target nvptx64-nvidia-cuda \
        --manifest-path "$TEST_TMP/Cargo.toml"; then
        echo -e "  ${GREEN}OK${NC}   $name"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}FAIL${NC} $name (no_std NVPTX graph did not compile)"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Typed-priority dependency compile tests ==="
expect_pass "Low_to_Low" "$TEST_DIR/valid_low_low.rs"
expect_pass "Low_to_Normal" "$TEST_DIR/valid_low_normal.rs"
expect_pass "Low_to_High" "$TEST_DIR/valid_low_high.rs"
expect_pass "Normal_to_Normal" "$TEST_DIR/valid_normal_normal.rs"
expect_pass "Normal_to_High" "$TEST_DIR/valid_normal_high.rs"
expect_pass "High_to_High" "$TEST_DIR/valid_high_high.rs"
expect_pass "High_spawns_Low" "$TEST_DIR/valid_high_spawns_low.rs"
expect_wait_rejected "High_to_Low" "$TEST_DIR/invalid_high_low.rs"
expect_wait_rejected "High_to_Normal" "$TEST_DIR/invalid_high_normal.rs"
expect_wait_rejected "Normal_to_Low" "$TEST_DIR/invalid_normal_low.rs"
expect_private_token_rejected "Cannot_forge_High_token" "$TEST_DIR/invalid_forge_high_token.rs"
expect_forge_receiver_mutation_rejected "$TEST_DIR/invalid_forge_high_token.rs"
expect_nvptx_pass "NoStd_NVPTX_graph" "$TEST_DIR/valid_nvptx.rs"

if [ "$FAIL" -ne 0 ]; then
    echo -e "${RED}$FAIL failed, $PASS passed${NC}"
    exit 1
fi
echo -e "${GREEN}All $PASS typed-priority compile tests passed${NC}"
