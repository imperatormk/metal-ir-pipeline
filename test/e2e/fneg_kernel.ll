; Test: LowerFNeg — fneg → fsub -0.0

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

define void @kernel_add(ptr addrspace(1) %A, ptr addrspace(1) %B, i32 %tid_x) {
entry:
  %pa = getelementptr float, ptr addrspace(1) %A, i32 %tid_x
  %v = load float, ptr addrspace(1) %pa, align 4
  %neg = fneg float %v
  %pb = getelementptr float, ptr addrspace(1) %B, i32 %tid_x
  store float %neg, ptr addrspace(1) %pb, align 4
  ret void
}

!air.kernel = !{!1}
!air.version = !{!8}
!1 = !{ptr @kernel_add, !2, !3}
!2 = !{}
!3 = !{!4, !5, !10}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"A"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"B"}
!10 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_x"}
!8 = !{i32 2, i32 8, i32 0}
