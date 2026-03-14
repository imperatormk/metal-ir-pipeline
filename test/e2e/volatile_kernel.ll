; Test: DeviceLoadsVolatile pass
; Load+store to same device pointer inside loop → load should become volatile

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

define void @kernel_add(ptr addrspace(1) %buf, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %p = getelementptr float, ptr addrspace(1) %buf, i32 %i
  %v = load float, ptr addrspace(1) %p, align 4
  %v2 = fadd float %v, 1.0
  store float %v2, ptr addrspace(1) %p, align 4
  %i_next = add i32 %i, 1
  %cond = icmp slt i32 %i_next, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}

!air.kernel = !{!1}
!air.version = !{!8}
!1 = !{ptr @kernel_add, !2, !3}
!2 = !{}
!3 = !{!4, !10}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"buf"}
!10 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"n"}
!8 = !{i32 2, i32 8, i32 0}
