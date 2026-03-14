; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @add_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, i32 %tid_x, i32 %tgid_x) {
  %block_offset = mul i32 %tgid_x, 128
  %idx = add i32 %block_offset, %tid_x
  %p0 = getelementptr float, ptr addrspace(1) %0, i32 %idx
  %v0 = load float, ptr addrspace(1) %p0, align 4
  %p1 = getelementptr float, ptr addrspace(1) %1, i32 %idx
  %v1 = load float, ptr addrspace(1) %p1, align 4
  %sum = fadd float %v0, %v1
  %p2 = getelementptr float, ptr addrspace(1) %2, i32 %idx
  store float %sum, ptr addrspace(1) %p2, align 4
  ret void
}

!llvm.module.flags = !{!0}
!air.kernel = !{!1}
!air.version = !{!8}

!0 = !{i32 7, !"frame-pointer", i32 0}
!1 = !{ptr @add_kernel, !2, !3}
!2 = !{}
!3 = !{!4, !5, !6, !10, !11}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"a"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"b"}
!6 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"c"}
!10 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_x"}
!11 = !{i32 4, !"air.threadgroup_position_in_grid", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"tgid_x"}
!8 = !{i32 2, i32 8, i32 0}
