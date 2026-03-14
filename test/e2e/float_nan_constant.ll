source_filename = "test_nan"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @nan_kernel(ptr addrspace(1) %buf, i32 %tid_x) {
  %p = getelementptr float, ptr addrspace(1) %buf, i32 %tid_x
  store float 0x7FF8000000000000, ptr addrspace(1) %p, align 4
  ret void
}

!llvm.module.flags = !{!0}
!air.kernel = !{!1}
!air.version = !{!5}

!0 = !{i32 7, !"frame-pointer", i32 0}
!1 = !{ptr @nan_kernel, !2, !3}
!2 = !{}
!3 = !{!4, !6}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"buf"}
!6 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_x"}
!5 = !{i32 2, i32 8, i32 0}
