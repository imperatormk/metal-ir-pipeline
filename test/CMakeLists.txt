# Test infrastructure

# Smoke test: full pipeline
add_test(NAME smoke_test
  COMMAND metal-ir-opt ${CMAKE_SOURCE_DIR}/test/simple_kernel.ll --emit-llvm -o /dev/null
)

# Per-pass tests (FileCheck-style via run_tests.sh)
add_test(NAME pass_tests
  COMMAND ${CMAKE_SOURCE_DIR}/test/pass/run_tests.sh $<TARGET_FILE:metal-ir-opt>
)
