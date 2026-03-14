target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@global_smem = internal addrspace(3) global [16 x i8] undef, align 16

declare void @air.wg.barrier(i32, i32)
declare float @air.simd_shuffle_xor.f32(float, i16)
declare [3 x i32] @air.thread_position_in_threadgroup()
declare [3 x i32] @air.threadgroup_position_in_grid()

define void @reduce_sum_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(2) %2) {
  %4 = load i32, ptr addrspace(2) %2, align 4
  %5 = call [3 x i32] @air.threadgroup_position_in_grid()
  %6 = extractvalue [3 x i32] %5, 0
  %7 = mul i32 %6, 128
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
  %21 = and i32 %20, 127
  %22 = lshr i32 %21, 0
  %23 = or disjoint i32 %22, 0
  %24 = xor i32 0, %23
  %25 = xor i32 %24, 0
  %26 = add i32 %25, 0
  %27 = add i32 %7, %26
  %28 = icmp slt i32 %27, %4
  %29 = getelementptr float, ptr addrspace(1) %0, i32 %27
  %30 = load float, ptr addrspace(1) %29, align 4
  %31 = select i1 %28, float %30, float 0.000000e+00
  %32 = call float @air.simd_shuffle_xor.f32(float %31, i16 16)
  %33 = fadd float %31, %32
  %34 = call float @air.simd_shuffle_xor.f32(float %33, i16 8)
  %35 = fadd float %33, %34
  %36 = call float @air.simd_shuffle_xor.f32(float %35, i16 4)
  %37 = fadd float %35, %36
  %38 = call float @air.simd_shuffle_xor.f32(float %37, i16 2)
  %39 = fadd float %37, %38
  %40 = call float @air.simd_shuffle_xor.f32(float %39, i16 1)
  %41 = fadd float %39, %40
  %42 = call [3 x i32] @air.thread_position_in_threadgroup()
  %43 = extractvalue [3 x i32] %42, 0
  %44 = zext i32 %43 to i64
  %45 = trunc i64 %44 to i32
  %46 = and i32 %45, 127
  %47 = urem i32 %46, 32
  %48 = call [3 x i32] @air.thread_position_in_threadgroup()
  %49 = extractvalue [3 x i32] %48, 0
  %50 = udiv i32 %49, 32
  %51 = shl i32 %47, 0
  %52 = or i32 0, %51
  %53 = shl i32 %50, 5
  %54 = or i32 %52, %53
  %55 = and i32 %54, 96
  %56 = lshr i32 %55, 3
  %57 = or disjoint i32 0, %56
  %58 = xor i32 0, %57
  %59 = xor i32 %58, 0
  %60 = xor i32 %59, 0
  %61 = add i32 %60, 0
  %62 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %61
  %63 = insertelement <1 x float> undef, float %41, i32 0
  store <1 x float> %63, ptr addrspace(3) %62, align 4
  call void @air.wg.barrier(i32 1, i32 1)
  %64 = call [3 x i32] @air.thread_position_in_threadgroup()
  %65 = extractvalue [3 x i32] %64, 0
  %66 = zext i32 %65 to i64
  %67 = trunc i64 %66 to i32
  %68 = and i32 %67, 127
  %69 = urem i32 %68, 32
  %70 = call [3 x i32] @air.thread_position_in_threadgroup()
  %71 = extractvalue [3 x i32] %70, 0
  %72 = udiv i32 %71, 32
  %73 = shl i32 %69, 0
  %74 = or i32 0, %73
  %75 = shl i32 %72, 5
  %76 = or i32 %74, %75
  %77 = and i32 %76, 3
  %78 = shl i32 %77, 2
  %79 = or disjoint i32 %78, 0
  %80 = xor i32 0, %79
  %81 = xor i32 %80, 0
  %82 = xor i32 %81, 0
  %83 = add i32 %82, 0
  %84 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %83
  %85 = load <1 x float>, ptr addrspace(3) %84, align 4
  %86 = extractelement <1 x float> %85, i32 0
  %87 = call float @air.simd_shuffle_xor.f32(float %86, i16 2)
  %88 = fadd float %86, %87
  %89 = call float @air.simd_shuffle_xor.f32(float %88, i16 1)
  %90 = fadd float %88, %89
  %91 = getelementptr float, ptr addrspace(1) %1, i32 %6
  store float %90, ptr addrspace(1) %91, align 4
  ret void
}
