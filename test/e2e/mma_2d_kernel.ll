target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@__tg_a = internal addrspace(3) global [64 x float] undef, align 4
@__tg_b = internal addrspace(3) global [64 x float] undef, align 4
@__tg_c = internal addrspace(3) global [64 x float] undef, align 4

declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)
declare void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float>, ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
declare void @air.threadgroup.barrier(i32, i32)
declare i32 @air.thread_index_in_simdgroup()
declare [3 x i32] @air.thread_position_in_grid()

define void @mma_2d_kernel(ptr addrspace(1) %out) {
  %tid3 = call [3 x i32] @air.thread_position_in_grid()
  %tid = extractvalue [3 x i32] %tid3, 0
  %sl = call i32 @air.thread_index_in_simdgroup()
  ; Each thread writes 2 elements to fill all 64
  %idx0 = mul i32 %sl, 2
  %idx1 = add i32 %idx0, 1
  %idx0_64 = zext i32 %idx0 to i64
  %idx1_64 = zext i32 %idx1 to i64
  %a0 = getelementptr float, ptr addrspace(3) @__tg_a, i64 %idx0_64
  %a1 = getelementptr float, ptr addrspace(3) @__tg_a, i64 %idx1_64
  %b0 = getelementptr float, ptr addrspace(3) @__tg_b, i64 %idx0_64
  %b1 = getelementptr float, ptr addrspace(3) @__tg_b, i64 %idx1_64
  %c0 = getelementptr float, ptr addrspace(3) @__tg_c, i64 %idx0_64
  %c1 = getelementptr float, ptr addrspace(3) @__tg_c, i64 %idx1_64
  store float 1.000000e+00, ptr addrspace(3) %a0, align 4
  store float 1.000000e+00, ptr addrspace(3) %a1, align 4
  store float 1.000000e+00, ptr addrspace(3) %b0, align 4
  store float 1.000000e+00, ptr addrspace(3) %b1, align 4
  store float 0.000000e+00, ptr addrspace(3) %c0, align 4
  store float 0.000000e+00, ptr addrspace(3) %c1, align 4
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %mma_a = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_a, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
  %mma_b = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_b, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
  %mma_c = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_c, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
  %mma_r = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %mma_a, <64 x float> %mma_b, <64 x float> %mma_c)
  call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float> %mma_r, ptr addrspace(3) @__tg_c, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
  ; Read back result — each thread reads its 2 elements
  %r0 = load float, ptr addrspace(3) %c0, align 4
  %r1 = load float, ptr addrspace(3) %c1, align 4
  %out0 = getelementptr float, ptr addrspace(1) %out, i32 %idx0
  %out1 = getelementptr float, ptr addrspace(1) %out, i32 %idx1
  store float %r0, ptr addrspace(1) %out0, align 4
  store float %r1, ptr addrspace(1) %out1, align 4
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
