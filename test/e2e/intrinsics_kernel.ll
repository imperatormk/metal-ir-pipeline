; Test: LLVM intrinsic rename, int min/max lowering, bitcast zero, barrier rename

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

declare float @llvm.sin.f32(float)
declare float @llvm.sqrt.f32(float)
declare float @llvm.fabs.f32(float)
declare i32 @llvm.smin.i32(i32, i32)
declare i32 @llvm.umax.i32(i32, i32)
declare void @air.threadgroup.barrier(i32, i32)

define void @kernel_add(ptr addrspace(1) %A, i32 %tid_x) {
entry:
  %pa = getelementptr float, ptr addrspace(1) %A, i32 %tid_x
  %a = load float, ptr addrspace(1) %pa, align 4
  %s = call float @llvm.sin.f32(float %a)
  %sq = call float @llvm.sqrt.f32(float %s)
  %ab = call float @llvm.fabs.f32(float %sq)
  store float %ab, ptr addrspace(1) %pa, align 4
  call void @air.threadgroup.barrier(i32 1, i32 4)
  ret void
}

!air.kernel = !{!1}
!air.version = !{!8}
!1 = !{ptr @kernel_add, !2, !3}
!2 = !{}
!3 = !{!4, !10}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"A"}
!10 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_x"}
!8 = !{i32 2, i32 8, i32 0}
