target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

declare [3 x i32] @air.thread_position_in_threadgroup()

define void @struct_phi_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1) {
entry:
  %tid_arr = call [3 x i32] @air.thread_position_in_threadgroup()
  %tid = extractvalue [3 x i32] %tid_arr, 0
  %tid64 = zext i32 %tid to i64
  %a_init = getelementptr float, ptr addrspace(1) %0, i64 %tid64
  %s0 = insertvalue { ptr addrspace(1) } undef, ptr addrspace(1) %a_init, 0
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv_next, %loop ]
  %acc = phi float [ 0.000000e+00, %entry ], [ %acc_next, %loop ]
  %ptrs = phi { ptr addrspace(1) } [ %s0, %entry ], [ %ptrs_next, %loop ]
  %p0 = extractvalue { ptr addrspace(1) } %ptrs, 0
  %val = load float, ptr addrspace(1) %p0, align 4
  %acc_next = fadd float %acc, %val
  %p0_next = getelementptr float, ptr addrspace(1) %p0, i64 1
  %ptrs_next = insertvalue { ptr addrspace(1) } %ptrs, ptr addrspace(1) %p0_next, 0
  %iv_next = add i32 %iv, 1
  %cond = icmp slt i32 %iv_next, 4
  br i1 %cond, label %loop, label %exit

exit:
  %out = getelementptr float, ptr addrspace(1) %1, i64 %tid64
  store float %acc_next, ptr addrspace(1) %out, align 4
  ret void
}
