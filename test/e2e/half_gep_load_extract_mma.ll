target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

source_filename = "LLVMDialectModule"

@__tg_buf = internal addrspace(3) global [64 x float] undef, align 4

declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
declare void @air.threadgroup.barrier(i32, i32)
declare i32 @air.thread_index_in_simdgroup()

define void @half_extract_kernel(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %lane = call i32 @air.thread_index_in_simdgroup()

  ; Variable-index half GEP — this triggers lshr + load-and-extract
  %p = getelementptr half, ptr addrspace(1) %in, i32 %lane
  %v = load half, ptr addrspace(1) %p
  %vf = fpext half %v to float

  ; Store to TG
  %lane64 = zext i32 %lane to i64
  %tg_p = getelementptr float, ptr addrspace(3) @__tg_buf, i64 %lane64
  store float %vf, ptr addrspace(3) %tg_p

  call void @air.threadgroup.barrier(i32 1, i32 4)

  ; MMA load (forces MMA path, validates typed pointer compatibility)
  %mma = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(
    ptr addrspace(3) @__tg_buf,
    <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)

  ; Write loaded half value to output for verification
  %out_p = getelementptr float, ptr addrspace(1) %out, i32 %lane
  store float %vf, ptr addrspace(1) %out_p
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
