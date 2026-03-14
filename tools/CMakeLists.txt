llvm_map_components_to_libnames(llvm_libs
  Core Support IRReader BitWriter Option Passes Analysis
)

add_executable(metal-ir-opt
  metal-ir-opt.cpp
)

target_link_libraries(metal-ir-opt
  MetalIRTransforms
  MetalIRSerialization
  ${llvm_libs}
)
