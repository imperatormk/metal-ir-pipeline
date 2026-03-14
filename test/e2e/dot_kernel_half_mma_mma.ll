target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

source_filename = "LLVMDialectModule"

@__tg_dot_ab_0 = internal addrspace(3) global [512 x float] undef, align 4

declare void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float>, ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)
declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
declare void @air.wg.barrier(i32, i32)
declare [3 x i32] @air.thread_position_in_threadgroup()

define void @mma_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2) {
  %tid = call [3 x i32] @air.thread_position_in_threadgroup()
  %tidx = extractvalue [3 x i32] %tid, 0
  %lane = urem i32 %tidx, 32
  %a_idx = mul i32 %lane, 8
  %a_gep = getelementptr half, ptr addrspace(1) %0, i32 %a_idx
  %a_val = load half, ptr addrspace(1) %a_gep, align 2

  %mma_base = getelementptr float, ptr addrspace(3) @__tg_dot_ab_0, i64 0
  %mma_a = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(
    ptr addrspace(3) %mma_base,
    <2 x i64> <i64 64, i64 8>, <2 x i64> <i64 1, i64 64>, <2 x i64> zeroinitializer)

  %mma_c = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(
    <64 x float> %mma_a, <64 x float> %mma_a, <64 x float> zeroinitializer)

  call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(
    <64 x float> %mma_c, ptr addrspace(3) %mma_base,
    <2 x i64> <i64 64, i64 8>, <2 x i64> <i64 1, i64 64>, <2 x i64> zeroinitializer)

  ; Store widened float output (real MMA kernels accumulate in float)
  %a_f32 = fpext half %a_val to float
  %out_gep = getelementptr float, ptr addrspace(1) %2, i32 %a_idx
  store float %a_f32, ptr addrspace(1) %out_gep, align 4
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
