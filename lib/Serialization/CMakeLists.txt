llvm_map_components_to_libnames(llvm_libs
  Core Support BitWriter BitstreamReader
)

add_library(MetalIRSerialization
  BitcodeEmitter.cpp
  ValueEnumerator.cpp
  TypeTableWriter.cpp
  ConstantsWriter.cpp
  MetadataWriter.cpp
  FunctionWriter.cpp
  MetallibWriter.cpp
)

target_link_libraries(MetalIRSerialization ${llvm_libs})
target_include_directories(MetalIRSerialization PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
