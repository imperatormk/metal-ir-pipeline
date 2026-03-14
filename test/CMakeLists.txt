# Test infrastructure — for now just a simple smoke test
add_test(NAME smoke_test
  COMMAND metal-ir-opt ${CMAKE_SOURCE_DIR}/test/simple_kernel.ll --emit-llvm -o /dev/null
)
