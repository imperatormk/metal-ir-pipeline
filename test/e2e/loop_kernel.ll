target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @sum_kernel(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %tid) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %acc_next, %loop ]
  %ptr = getelementptr float, ptr addrspace(1) %in, i32 %i
  %val = load float, ptr addrspace(1) %ptr, align 4
  %acc_next = fadd float %acc, %val
  %i_next = add i32 %i, 1
  %cond = icmp slt i32 %i_next, 4
  br i1 %cond, label %loop, label %exit

exit:
  %out_ptr = getelementptr float, ptr addrspace(1) %out, i32 %tid
  store float %acc_next, ptr addrspace(1) %out_ptr, align 4
  ret void
}

!llvm.module.flags = !{!0}
!air.kernel = !{!1}
!air.version = !{!8}

!0 = !{i32 7, !"frame-pointer", i32 0}
!1 = !{ptr @sum_kernel, !2, !3}
!2 = !{}
!3 = !{!4, !5, !6}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!6 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!8 = !{i32 2, i32 8, i32 0}
