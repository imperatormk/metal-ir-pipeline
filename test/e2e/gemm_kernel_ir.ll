target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Minimal MMA kernel: C[8x8] = A[8x8] * B[8x8]
source_filename = "LLVMDialectModule"

@__tg_dot_a = internal addrspace(3) global [64 x float] undef, align 4
@__tg_dot_b = internal addrspace(3) global [64 x float] undef, align 4
@__tg_dot_c = internal addrspace(3) global [64 x float] undef, align 4

declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)
declare void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float>, ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
declare void @air.threadgroup.barrier(i32, i32)
declare i32 @air.thread_index_in_simdgroup()
declare [3 x i32] @air.thread_position_in_grid()

define void @matmul_kernel(ptr addrspace(1) %A, ptr addrspace(1) %B, ptr addrspace(1) %C) {
  ; Load A[lane] and B[lane] from device memory into TG
  %tid3 = call [3 x i32] @air.thread_position_in_grid()
  %tid = extractvalue [3 x i32] %tid3, 0
  %lane = call i32 @air.thread_index_in_simdgroup()
  ; Each thread loads 2 elements: A[lane*2] and A[lane*2+1]
  %a_idx0 = mul i32 %lane, 2
  %a_idx1 = add i32 %a_idx0, 1
  %a_ptr0 = getelementptr float, ptr addrspace(1) %A, i32 %a_idx0
  %a_ptr1 = getelementptr float, ptr addrspace(1) %A, i32 %a_idx1
  %a_val0 = load float, ptr addrspace(1) %a_ptr0, align 4
  %a_val1 = load float, ptr addrspace(1) %a_ptr1, align 4
  %a_idx0_64 = zext i32 %a_idx0 to i64
  %a_idx1_64 = zext i32 %a_idx1 to i64
  %tg_a0 = getelementptr float, ptr addrspace(3) @__tg_dot_a, i64 %a_idx0_64
  %tg_a1 = getelementptr float, ptr addrspace(3) @__tg_dot_a, i64 %a_idx1_64
  store float %a_val0, ptr addrspace(3) %tg_a0, align 4
  store float %a_val1, ptr addrspace(3) %tg_a1, align 4
  ; Same for B
  %b_ptr0 = getelementptr float, ptr addrspace(1) %B, i32 %a_idx0
  %b_ptr1 = getelementptr float, ptr addrspace(1) %B, i32 %a_idx1
  %b_val0 = load float, ptr addrspace(1) %b_ptr0, align 4
  %b_val1 = load float, ptr addrspace(1) %b_ptr1, align 4
  %tg_b0 = getelementptr float, ptr addrspace(3) @__tg_dot_b, i64 %a_idx0_64
  %tg_b1 = getelementptr float, ptr addrspace(3) @__tg_dot_b, i64 %a_idx1_64
  store float %b_val0, ptr addrspace(3) %tg_b0, align 4
  store float %b_val1, ptr addrspace(3) %tg_b1, align 4
  ; Zero C
  %tg_c0 = getelementptr float, ptr addrspace(3) @__tg_dot_c, i64 %a_idx0_64
  %tg_c1 = getelementptr float, ptr addrspace(3) @__tg_dot_c, i64 %a_idx1_64
  store float 0.0, ptr addrspace(3) %tg_c0, align 4
  store float 0.0, ptr addrspace(3) %tg_c1, align 4
  call void @air.threadgroup.barrier(i32 1, i32 4)
  ; MMA: C = A * B + C
  %mA = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_a, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
  %mB = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_b, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
  %mC = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_c, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
  %mR = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %mA, <64 x float> %mB, <64 x float> %mC)
  call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float> %mR, ptr addrspace(3) @__tg_dot_c, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
  ; Store C back to device
  %c_val0 = load float, ptr addrspace(3) %tg_c0, align 4
  %c_val1 = load float, ptr addrspace(3) %tg_c1, align 4
  %c_ptr0 = getelementptr float, ptr addrspace(1) %C, i32 %a_idx0
  %c_ptr1 = getelementptr float, ptr addrspace(1) %C, i32 %a_idx1
  store float %c_val0, ptr addrspace(1) %c_ptr0, align 4
  store float %c_val1, ptr addrspace(1) %c_ptr1, align 4
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
