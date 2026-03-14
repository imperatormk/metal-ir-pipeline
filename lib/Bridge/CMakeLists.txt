llvm_map_components_to_libnames(llvm_libs
  Core Support IRReader BitWriter TransformUtils Analysis Passes
)

add_library(MetalIRBridge SHARED
  metalir_bridge.cpp
)

target_link_libraries(MetalIRBridge
  MetalIRTransforms
  MetalIRSerialization
  ${llvm_libs}
)

target_include_directories(MetalIRBridge PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(MetalIRBridge PROPERTIES
  OUTPUT_NAME "MetalIRBridge"
  PREFIX "lib"
  CXX_VISIBILITY_PRESET hidden
  C_VISIBILITY_PRESET hidden
)

# Hide all LLVM symbols — only expose metalir_compile/metalir_free
target_link_options(MetalIRBridge PRIVATE
  -Wl,-exported_symbol,_metalir_compile
  -Wl,-exported_symbol,_metalir_free
)
