# Test infrastructure

# Per-pass tests (FileCheck-style via run_tests.sh)
add_test(NAME pass_tests
  COMMAND ${CMAKE_SOURCE_DIR}/test/pass/run_tests.sh $<TARGET_FILE:metal-ir-opt>
)
