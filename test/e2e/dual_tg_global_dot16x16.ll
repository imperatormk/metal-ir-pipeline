target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@__tg_cvt_0 = internal addrspace(3) global [256 x float] undef, align 4
@__tg_dot_ab_0 = internal addrspace(3) global [256 x float] undef, align 4
@global_smem = internal addrspace(3) global [1024 x i8] undef, align 16

declare void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float>, ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)
declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
declare void @air.simdgroup.barrier(i32, i32)
declare void @air.threadgroup.barrier(i32, i32)
declare i32 @air.thread_index_in_simdgroup()
declare [3 x i32] @air.thread_position_in_threadgroup()

define void @dot_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2) {
  %4 = call [3 x i32] @air.thread_position_in_threadgroup()
  %5 = extractvalue [3 x i32] %4, 0
  %6 = zext i32 %5 to i64
  %7 = trunc i64 %6 to i32
  %8 = and i32 %7, 127
  %9 = urem i32 %8, 32
  %10 = call [3 x i32] @air.thread_position_in_threadgroup()
  %11 = extractvalue [3 x i32] %10, 0
  %12 = udiv i32 %11, 32
  %13 = shl i32 %9, 0
  %14 = or i32 0, %13
  %15 = shl i32 %12, 5
  %16 = or i32 %14, %15
  %17 = and i32 %16, 120
  %18 = lshr i32 %17, 3
  %19 = or disjoint i32 %18, 0
  %20 = xor i32 0, %19
  %21 = xor i32 %20, 0
  %22 = add i32 %21, 0
  %23 = mul i32 %22, 16
  %24 = getelementptr float, ptr addrspace(1) %0, i32 %23
  %25 = call [3 x i32] @air.thread_position_in_threadgroup()
  %26 = extractvalue [3 x i32] %25, 0
  %27 = zext i32 %26 to i64
  %28 = trunc i64 %27 to i32
  %29 = and i32 %28, 127
  %30 = urem i32 %29, 32
  %31 = call [3 x i32] @air.thread_position_in_threadgroup()
  %32 = extractvalue [3 x i32] %31, 0
  %33 = udiv i32 %32, 32
  %34 = shl i32 %30, 0
  %35 = or i32 0, %34
  %36 = shl i32 %33, 5
  %37 = or i32 %35, %36
  %38 = and i32 %37, 7
  %39 = shl i32 %38, 1
  %40 = or disjoint i32 %39, 0
  %41 = xor i32 0, %40
  %42 = xor i32 %41, 0
  %43 = xor i32 %41, 1
  %44 = add i32 %42, 0
  %45 = add i32 %43, 0
  %46 = getelementptr float, ptr addrspace(1) %24, i32 %44
  %47 = getelementptr float, ptr addrspace(1) %24, i32 %45
  %48 = load float, ptr addrspace(1) %46, align 4
  %49 = load float, ptr addrspace(1) %47, align 4
  %50 = getelementptr float, ptr addrspace(1) %1, i32 %23
  %51 = getelementptr float, ptr addrspace(1) %50, i32 %44
  %52 = getelementptr float, ptr addrspace(1) %50, i32 %45
  %53 = load float, ptr addrspace(1) %51, align 4
  %54 = load float, ptr addrspace(1) %52, align 4
  %55 = call i32 @air.thread_index_in_simdgroup()
  %56 = call [3 x i32] @air.thread_position_in_threadgroup()
  %57 = extractvalue [3 x i32] %56, 0
  %58 = udiv i32 %57, 32
  %59 = udiv i32 %55, 8
  %60 = urem i32 %55, 8
  %61 = mul i32 %58, 4
  %62 = add i32 %61, %59
  %63 = mul i32 %60, 2
  %64 = add i32 0, %63
  %65 = udiv i32 %55, 16
  %66 = urem i32 %55, 16
  %67 = mul i32 %58, 2
  %68 = add i32 %67, %65
  %69 = add i32 0, %66
  %70 = mul i32 %62, 16
  %71 = add i32 %70, %64
  %72 = zext i32 %71 to i64
  %73 = getelementptr float, ptr addrspace(3) @__tg_dot_ab_0, i64 %72
  store float %48, ptr addrspace(3) %73, align 4
  %74 = add i32 %64, 1
  %75 = add i32 %70, %74
  %76 = zext i32 %75 to i64
  %77 = getelementptr float, ptr addrspace(3) @__tg_dot_ab_0, i64 %76
  store float %49, ptr addrspace(3) %77, align 4
  call void @air.threadgroup.barrier(i32 1, i32 4)
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %78 = mul i32 %68, 16
  %79 = add i32 %78, %69
  %80 = zext i32 %79 to i64
  %81 = getelementptr float, ptr addrspace(3) @__tg_dot_ab_0, i64 %80
  store float 0.000000e+00, ptr addrspace(3) %81, align 4
  %82 = add i32 %68, 8
  %83 = mul i32 %82, 16
  %84 = add i32 %83, %69
  %85 = zext i32 %84 to i64
  %86 = getelementptr float, ptr addrspace(3) @__tg_dot_ab_0, i64 %85
  store float 0.000000e+00, ptr addrspace(3) %86, align 4
  call void @air.threadgroup.barrier(i32 1, i32 4)
  call void @air.threadgroup.barrier(i32 1, i32 4)
  store float %53, ptr addrspace(3) %73, align 4
  store float %54, ptr addrspace(3) %77, align 4
  call void @air.threadgroup.barrier(i32 1, i32 4)
  call void @air.threadgroup.barrier(i32 1, i32 4)
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %87 = load float, ptr addrspace(3) %81, align 4
  %88 = load float, ptr addrspace(3) %86, align 4
  %89 = getelementptr float, ptr addrspace(1) %2, i32 %23
  %90 = getelementptr float, ptr addrspace(1) %89, i32 %44
  %91 = getelementptr float, ptr addrspace(1) %89, i32 %45
  %92 = call i32 @air.thread_index_in_simdgroup()
  %93 = call [3 x i32] @air.thread_position_in_threadgroup()
  %94 = extractvalue [3 x i32] %93, 0
  %95 = udiv i32 %94, 32
  %96 = udiv i32 %92, 16
  %97 = urem i32 %92, 16
  %98 = mul i32 %95, 2
  %99 = add i32 %98, %96
  %100 = add i32 0, %97
  %101 = udiv i32 %92, 8
  %102 = urem i32 %92, 8
  %103 = mul i32 %95, 4
  %104 = add i32 %103, %101
  %105 = mul i32 %102, 2
  %106 = add i32 0, %105
  br i1 true, label %107, label %117

107:
  %108 = mul i32 %99, 16
  %109 = add i32 %108, %100
  %110 = zext i32 %109 to i64
  %111 = getelementptr float, ptr addrspace(3) @__tg_cvt_0, i64 %110
  store float %87, ptr addrspace(3) %111, align 4
  %112 = add i32 %99, 8
  %113 = mul i32 %112, 16
  %114 = add i32 %113, %100
  %115 = zext i32 %114 to i64
  %116 = getelementptr float, ptr addrspace(3) @__tg_cvt_0, i64 %115
  store float %88, ptr addrspace(3) %116, align 4
  br label %117

117:
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %118 = mul i32 %104, 16
  %119 = add i32 %118, %106
  %120 = zext i32 %119 to i64
  %121 = getelementptr float, ptr addrspace(3) @__tg_cvt_0, i64 %120
  %122 = load float, ptr addrspace(3) %121, align 4
  %123 = add i32 %106, 1
  %124 = add i32 %118, %123
  %125 = zext i32 %124 to i64
  %126 = getelementptr float, ptr addrspace(3) @__tg_cvt_0, i64 %125
  %127 = load float, ptr addrspace(3) %126, align 4
  store float %122, ptr addrspace(1) %90, align 4
  store float %127, ptr addrspace(1) %91, align 4
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
