llvm_map_components_to_libnames(llvm_libs
  Core Support IRReader BitWriter TransformUtils Analysis
)

add_library(MetalIRTransforms
  Pipeline.cpp
  InlineNonKernel.cpp
  DecomposeStructPhis.cpp
  PtrPhiToI64.cpp
  BarrierRename.cpp
  TGBarrierInsert.cpp
  NaNMinMax.cpp
  LowerFNeg.cpp
  BitcastZeroInit.cpp
  LLVMToAIRIntrinsics.cpp
  LowerIntMinMax.cpp
  LowerAtomicRMW.cpp
  TGGlobalDeadElim.cpp
  TGGlobalCoalesce.cpp
  TGGlobalGEPRewrite.cpp
  DeviceLoadsVolatile.cpp
  WidenDeviceLoads.cpp
  ScalarStoreGuard.cpp
  SplitI64Shuffle.cpp
  BFloat16CastDecompose.cpp
  InferTypedPointers.cpp
  ScalarBufferPacking.cpp
  AIRSystemValues.cpp
  NormalizeI1Pointers.cpp
  NormalizeAllocas.cpp
  PointeeTypeMap.cpp
  KernelProfile.cpp
)

target_link_libraries(MetalIRTransforms ${llvm_libs})
target_include_directories(MetalIRTransforms PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
