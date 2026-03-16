#!/bin/bash
# Per-pass test runner for metal-ir-pipeline.
#
# Each .test file contains:
#   Line 1: PASS: <pass-name>
#   Lines starting with CHECK: — strings that MUST appear in output
#   Lines starting with CHECK-NOT: — strings that must NOT appear in output
#   Lines starting with INPUT: — (optional) inline .ll input
#   Otherwise: the .ll input file is <testname>.ll in the same directory
#
# Usage:
#   ./run_tests.sh /path/to/metal-ir-opt [test1.test test2.test ...]
#   If no test files given, runs all *.test in this directory.

set -euo pipefail

OPT="${1:?Usage: $0 /path/to/metal-ir-opt [test files...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -eq 0 ]; then
  TESTS=("$SCRIPT_DIR"/*.test)
else
  TESTS=("$@")
fi

PASS_COUNT=0
FAIL_COUNT=0

for TEST_FILE in "${TESTS[@]}"; do
  TEST_NAME="$(basename "$TEST_FILE" .test)"

  # Parse directives
  PASS_NAME=""
  CHECKS=()
  CHECK_NOTS=()
  INPUT_FILE=""
  HAS_INLINE_INPUT=false

  while IFS= read -r line; do
    if [[ "$line" =~ ^PASS:\ *(.*) ]]; then
      PASS_NAME="${BASH_REMATCH[1]}"
    elif [[ "$line" =~ ^CHECK:\ *(.*) ]]; then
      CHECKS+=("${BASH_REMATCH[1]}")
    elif [[ "$line" =~ ^CHECK-NOT:\ *(.*) ]]; then
      CHECK_NOTS+=("${BASH_REMATCH[1]}")
    elif [[ "$line" =~ ^INPUT:\ *(.*) ]]; then
      INPUT_FILE="${BASH_REMATCH[1]}"
    fi
  done < "$TEST_FILE"

  if [ -z "$PASS_NAME" ]; then
    echo "SKIP $TEST_NAME (no PASS: directive)"
    continue
  fi

  # Determine input file
  if [ -z "$INPUT_FILE" ]; then
    INPUT_FILE="$(dirname "$TEST_FILE")/${TEST_NAME}.ll"
  fi

  if [ ! -f "$INPUT_FILE" ]; then
    echo "FAIL $TEST_NAME (input not found: $INPUT_FILE)"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    continue
  fi

  # Run pass
  OUTPUT=$("$OPT" "$INPUT_FILE" --pass="$PASS_NAME" --emit-llvm 2>&1) || {
    echo "FAIL $TEST_NAME (metal-ir-opt crashed)"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    continue
  }

  # Check assertions
  FAILED=false
  if [ ${#CHECKS[@]} -gt 0 ]; then
    for CHECK in "${CHECKS[@]}"; do
      if ! echo "$OUTPUT" | grep -qF "$CHECK"; then
        echo "FAIL $TEST_NAME: CHECK not found: $CHECK"
        FAILED=true
      fi
    done
  fi

  if [ ${#CHECK_NOTS[@]} -gt 0 ]; then
    for CHECK_NOT in "${CHECK_NOTS[@]}"; do
      if echo "$OUTPUT" | grep -qF "$CHECK_NOT"; then
        echo "FAIL $TEST_NAME: CHECK-NOT found: $CHECK_NOT"
        FAILED=true
      fi
    done
  fi

  if $FAILED; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
  else
    PASS_COUNT=$((PASS_COUNT + 1))
  fi
done

echo ""
echo "Results: $PASS_COUNT passed, $FAIL_COUNT failed"
exit $FAIL_COUNT
