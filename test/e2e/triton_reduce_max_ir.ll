target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

source_filename = "LLVMDialectModule"

declare float @air.simd_shuffle_xor.f32(float, i16)
declare [3 x i32] @air.thread_position_in_threadgroup()
declare [3 x i32] @air.threadgroup_position_in_grid()

define void @reduce_max_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(2) %2) {
  %4 = load i32, ptr addrspace(2) %2, align 4
  %5 = call [3 x i32] @air.threadgroup_position_in_grid()
  %6 = extractvalue [3 x i32] %5, 0
  %7 = mul i32 %6, 32
  %8 = call [3 x i32] @air.thread_position_in_threadgroup()
  %9 = extractvalue [3 x i32] %8, 0
  %10 = zext i32 %9 to i64
  %11 = trunc i64 %10 to i32
  %12 = and i32 %11, 127
  %13 = urem i32 %12, 32
  %14 = call [3 x i32] @air.thread_position_in_threadgroup()
  %15 = extractvalue [3 x i32] %14, 0
  %16 = udiv i32 %15, 32
  %17 = shl i32 %13, 0
  %18 = or i32 0, %17
  %19 = shl i32 %16, 5
  %20 = or i32 %18, %19
  %21 = and i32 %20, 31
  %22 = lshr i32 %21, 0
  %23 = or disjoint i32 %22, 0
  %24 = xor i32 0, %23
  %25 = xor i32 %24, 0
  %26 = add i32 %25, 0
  %27 = add i32 %7, %26
  %28 = icmp slt i32 %27, %4
  %29 = getelementptr float, ptr addrspace(1) %0, i32 %27
  %30 = load float, ptr addrspace(1) %29, align 4
  %31 = select i1 %28, float %30, float 0xFFF0000000000000
  %32 = call float @air.simd_shuffle_xor.f32(float %31, i16 16)
  %33 = call float @llvm.maxnum.f32(float %31, float %32)
  %34 = call float @air.simd_shuffle_xor.f32(float %33, i16 8)
  %35 = call float @llvm.maxnum.f32(float %33, float %34)
  %36 = call float @air.simd_shuffle_xor.f32(float %35, i16 4)
  %37 = call float @llvm.maxnum.f32(float %35, float %36)
  %38 = call float @air.simd_shuffle_xor.f32(float %37, i16 2)
  %39 = call float @llvm.maxnum.f32(float %37, float %38)
  %40 = call float @air.simd_shuffle_xor.f32(float %39, i16 1)
  %41 = call float @llvm.maxnum.f32(float %39, float %40)
  %42 = getelementptr float, ptr addrspace(1) %1, i32 %6
  store float %41, ptr addrspace(1) %42, align 4
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #0

attributes #0 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
