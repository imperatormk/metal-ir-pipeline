target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

source_filename = "LLVMDialectModule"

@__tg_dot_ab_0 = internal addrspace(3) global [4096 x float] undef, align 4

declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
declare void @air.wg.barrier(i32, i32)

define void @test_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2) {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv_next, %loop ]
  %p0 = phi ptr addrspace(1) [ %0, %entry ], [ %p0_next, %loop ]

  %p0_next = getelementptr half, ptr addrspace(1) %p0, i32 64

  %tg_ptr = getelementptr float, ptr addrspace(3) @__tg_dot_ab_0, i32 0
  %a = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(
    ptr addrspace(3) %tg_ptr,
    <2 x i64> <i64 8, i64 0>, <2 x i64> <i64 1, i64 0>, <2 x i64> <i64 0, i64 0>)

  call void @air.wg.barrier(i32 2, i32 1)

  %iv_next = add i32 %iv, 1
  %cond = icmp slt i32 %iv_next, 8
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
