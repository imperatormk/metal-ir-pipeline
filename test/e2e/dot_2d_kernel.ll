target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@__tg_cvt_0 = internal addrspace(3) global [256 x float] undef, align 4
@__tg_dot_c_0 = internal addrspace(3) global [256 x float] undef, align 4
@__tg_dot_b_0 = internal addrspace(3) global [256 x float] undef, align 4
@__tg_dot_a_0 = internal addrspace(3) global [256 x float] undef, align 4

declare void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float>, ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)
declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
declare void @air.simdgroup.barrier(i32, i32)
declare void @air.threadgroup.barrier(i32, i32)
declare i32 @air.thread_index_in_simdgroup()
declare [3 x i32] @air.thread_position_in_grid()
declare [3 x i32] @air.threadgroup_position_in_grid()

define void @dot_2d(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2) {
  %4 = call [3 x i32] @air.threadgroup_position_in_grid()
  %5 = extractvalue [3 x i32] %4, 0
  %6 = call [3 x i32] @air.threadgroup_position_in_grid()
  %7 = extractvalue [3 x i32] %6, 1
  %8 = mul i32 %5, 16
  %9 = call [3 x i32] @air.thread_position_in_grid()
  %10 = extractvalue [3 x i32] %9, 0
  %11 = zext i32 %10 to i64
  %12 = trunc i64 %11 to i32
  %13 = and i32 %12, 127
  %14 = urem i32 %13, 32
  %15 = call [3 x i32] @air.thread_position_in_grid()
  %16 = extractvalue [3 x i32] %15, 0
  %17 = udiv i32 %16, 32
  %18 = shl i32 %14, 0
  %19 = or i32 0, %18
  %20 = shl i32 %17, 5
  %21 = or i32 %19, %20
  %22 = and i32 %21, 120
  %23 = lshr i32 %22, 3
  %24 = or disjoint i32 %23, 0
  %25 = xor i32 0, %24
  %26 = xor i32 %25, 0
  %27 = add i32 %26, 0
  %28 = call [3 x i32] @air.thread_position_in_grid()
  %29 = extractvalue [3 x i32] %28, 0
  %30 = zext i32 %29 to i64
  %31 = trunc i64 %30 to i32
  %32 = and i32 %31, 127
  %33 = urem i32 %32, 32
  %34 = call [3 x i32] @air.thread_position_in_grid()
  %35 = extractvalue [3 x i32] %34, 0
  %36 = udiv i32 %35, 32
  %37 = shl i32 %33, 0
  %38 = or i32 0, %37
  %39 = shl i32 %36, 5
  %40 = or i32 %38, %39
  %41 = and i32 %40, 7
  %42 = shl i32 %41, 1
  %43 = or disjoint i32 %42, 0
  %44 = xor i32 0, %43
  %45 = xor i32 %44, 0
  %46 = xor i32 %44, 1
  %47 = add i32 %45, 0
  %48 = add i32 %46, 0
  %49 = add i32 %8, %27
  %50 = mul i32 %7, 16
  %51 = add i32 %50, %47
  %52 = add i32 %50, %48
  %53 = mul i32 %49, 16
  %54 = getelementptr float, ptr addrspace(1) %0, i32 %53
  %55 = getelementptr float, ptr addrspace(1) %54, i32 %47
  %56 = getelementptr float, ptr addrspace(1) %54, i32 %48
  %57 = load float, ptr addrspace(1) %55, align 4
  %58 = load float, ptr addrspace(1) %56, align 4
  %59 = mul i32 %27, 32
  %60 = getelementptr float, ptr addrspace(1) %1, i32 %59
  %61 = getelementptr float, ptr addrspace(1) %60, i32 %51
  %62 = getelementptr float, ptr addrspace(1) %60, i32 %52
  %63 = load float, ptr addrspace(1) %61, align 4
  %64 = load float, ptr addrspace(1) %62, align 4
  %65 = call i32 @air.thread_index_in_simdgroup()
  %66 = call [3 x i32] @air.thread_position_in_grid()
  %67 = extractvalue [3 x i32] %66, 0
  %68 = udiv i32 %67, 32
  %69 = udiv i32 %65, 8
  %70 = urem i32 %65, 8
  %71 = mul i32 %68, 4
  %72 = add i32 %71, %69
  %73 = mul i32 %70, 2
  %74 = add i32 0, %73
  %75 = udiv i32 %65, 16
  %76 = urem i32 %65, 16
  %77 = mul i32 %68, 2
  %78 = add i32 %77, %75
  %79 = add i32 0, %76
  %80 = mul i32 %72, 16
  %81 = add i32 %80, %74
  %82 = zext i32 %81 to i64
  %83 = getelementptr float, ptr addrspace(3) @__tg_dot_a_0, i64 %82
  store float %57, ptr addrspace(3) %83, align 4
  %84 = add i32 %74, 1
  %85 = add i32 %80, %84
  %86 = zext i32 %85 to i64
  %87 = getelementptr float, ptr addrspace(3) @__tg_dot_a_0, i64 %86
  store float %58, ptr addrspace(3) %87, align 4
  %88 = getelementptr float, ptr addrspace(3) @__tg_dot_b_0, i64 %82
  store float %63, ptr addrspace(3) %88, align 4
  %89 = getelementptr float, ptr addrspace(3) @__tg_dot_b_0, i64 %86
  store float %64, ptr addrspace(3) %89, align 4
  %90 = mul i32 %78, 16
  %91 = add i32 %90, %79
  %92 = zext i32 %91 to i64
  %93 = getelementptr float, ptr addrspace(3) @__tg_dot_c_0, i64 %92
  store float 0.000000e+00, ptr addrspace(3) %93, align 4
  %94 = add i32 %78, 8
  %95 = mul i32 %94, 16
  %96 = add i32 %95, %79
  %97 = zext i32 %96 to i64
  %98 = getelementptr float, ptr addrspace(3) @__tg_dot_c_0, i64 %97
  store float 0.000000e+00, ptr addrspace(3) %98, align 4
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %99 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_c_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer)
  %100 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_a_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer)
  %101 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_b_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer)
  %102 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %100, <64 x float> %101, <64 x float> %99)
  %103 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_a_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 8, i64 0>)
  %104 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_b_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 0, i64 8>)
  %105 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %103, <64 x float> %104, <64 x float> %102)
  call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float> %105, ptr addrspace(3) @__tg_dot_c_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer)
  %106 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_c_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 8, i64 0>)
  %107 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_a_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer)
  %108 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_b_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 8, i64 0>)
  %109 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %107, <64 x float> %108, <64 x float> %106)
  %110 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_a_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 8, i64 0>)
  %111 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_b_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 0, i64 8>)
  %112 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %110, <64 x float> %111, <64 x float> %109)
  call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float> %112, ptr addrspace(3) @__tg_dot_c_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 8, i64 0>)
  %113 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_c_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 0, i64 8>)
  %114 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_a_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 0, i64 8>)
  %115 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_b_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer)
  %116 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %114, <64 x float> %115, <64 x float> %113)
  %117 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_a_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> splat (i64 8))
  %118 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_b_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 0, i64 8>)
  %119 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %117, <64 x float> %118, <64 x float> %116)
  call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float> %119, ptr addrspace(3) @__tg_dot_c_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 0, i64 8>)
  %120 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_c_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> splat (i64 8))
  %121 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_a_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 0, i64 8>)
  %122 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_b_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 8, i64 0>)
  %123 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %121, <64 x float> %122, <64 x float> %120)
  %124 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_a_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> splat (i64 8))
  %125 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_b_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> splat (i64 8))
  %126 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %124, <64 x float> %125, <64 x float> %123)
  call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float> %126, ptr addrspace(3) @__tg_dot_c_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> splat (i64 8))
  call void @air.simdgroup.barrier(i32 2, i32 4)
  %127 = load float, ptr addrspace(3) %93, align 4
  %128 = load float, ptr addrspace(3) %98, align 4
  %129 = mul i32 %49, 32
  %130 = getelementptr float, ptr addrspace(1) %2, i32 %129
  %131 = getelementptr float, ptr addrspace(1) %130, i32 %51
  %132 = getelementptr float, ptr addrspace(1) %130, i32 %52
  %133 = call i32 @air.thread_index_in_simdgroup()
  %134 = call [3 x i32] @air.thread_position_in_grid()
  %135 = extractvalue [3 x i32] %134, 0
  %136 = udiv i32 %135, 32
  %137 = udiv i32 %133, 16
  %138 = urem i32 %133, 16
  %139 = mul i32 %136, 2
  %140 = add i32 %139, %137
  %141 = add i32 0, %138
  %142 = udiv i32 %133, 8
  %143 = urem i32 %133, 8
  %144 = mul i32 %136, 4
  %145 = add i32 %144, %142
  %146 = mul i32 %143, 2
  %147 = add i32 0, %146
  %148 = mul i32 %140, 16
  %149 = add i32 %148, %141
  %150 = zext i32 %149 to i64
  %151 = getelementptr float, ptr addrspace(3) @__tg_cvt_0, i64 %150
  store float %127, ptr addrspace(3) %151, align 4
  %152 = add i32 %140, 8
  %153 = mul i32 %152, 16
  %154 = add i32 %153, %141
  %155 = zext i32 %154 to i64
  %156 = getelementptr float, ptr addrspace(3) @__tg_cvt_0, i64 %155
  store float %128, ptr addrspace(3) %156, align 4
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %157 = mul i32 %145, 16
  %158 = add i32 %157, %147
  %159 = zext i32 %158 to i64
  %160 = getelementptr float, ptr addrspace(3) @__tg_cvt_0, i64 %159
  %161 = load float, ptr addrspace(3) %160, align 4
  %162 = add i32 %147, 1
  %163 = add i32 %157, %162
  %164 = zext i32 %163 to i64
  %165 = getelementptr float, ptr addrspace(3) @__tg_cvt_0, i64 %164
  %166 = load float, ptr addrspace(3) %165, align 4
  store float %161, ptr addrspace(1) %131, align 4
  store float %166, ptr addrspace(1) %132, align 4
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
