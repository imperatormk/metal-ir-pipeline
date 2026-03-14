; Test: TGBarrierInsert pass
; TG store without preceding barrier → barrier should be inserted

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

@shared = internal addrspace(3) global [128 x float] undef

define void @kernel_add(ptr addrspace(1) %buf, i32 %tid_x) {
entry:
  %p_dev = getelementptr float, ptr addrspace(1) %buf, i32 %tid_x
  %v = load float, ptr addrspace(1) %p_dev, align 4
  %p_tg = getelementptr [128 x float], ptr addrspace(3) @shared, i32 0, i32 %tid_x
  store float %v, ptr addrspace(3) %p_tg, align 4
  %v2 = load float, ptr addrspace(3) %p_tg, align 4
  store float %v2, ptr addrspace(1) %p_dev, align 4
  ret void
}

!air.kernel = !{!1}
!air.version = !{!8}
!1 = !{ptr @kernel_add, !2, !3}
!2 = !{}
!3 = !{!4, !10}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"buf"}
!10 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_x"}
!8 = !{i32 2, i32 8, i32 0}
