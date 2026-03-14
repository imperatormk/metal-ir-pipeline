; Simple Metal kernel IR — matches MetalASM's working testAddKernelTritonIR format.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @kernel_add(ptr addrspace(1) %A, ptr addrspace(1) %B, ptr addrspace(1) %C, i32 %tid_x) {
entry:
  %a_ptr = getelementptr float, ptr addrspace(1) %A, i32 %tid_x
  %b_ptr = getelementptr float, ptr addrspace(1) %B, i32 %tid_x
  %a = load float, ptr addrspace(1) %a_ptr, align 4
  %b = load float, ptr addrspace(1) %b_ptr, align 4
  %c = fadd float %a, %b
  %c_ptr = getelementptr float, ptr addrspace(1) %C, i32 %tid_x
  store float %c, ptr addrspace(1) %c_ptr, align 4
  ret void
}

!llvm.module.flags = !{!0}
!air.kernel = !{!1}
!air.version = !{!8}

!0 = !{i32 7, !"frame-pointer", i32 0}
!1 = !{ptr @kernel_add, !2, !3}
!2 = !{}
!3 = !{!4, !5, !6, !10}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"A"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"B"}
!6 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"C"}
!10 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_x"}
!8 = !{i32 2, i32 8, i32 0}
