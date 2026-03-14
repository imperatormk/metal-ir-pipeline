target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@__tg_cvt_0 = internal addrspace(3) global [32 x bfloat] undef, align 4

declare void @air.threadgroup.barrier(i32, i32)

define void @test_kernel(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %tid_x, i32 %tid_y, i32 %tid_z) {
entry:
  %tid64 = zext i32 %tid_x to i64
  %in_gep = getelementptr bfloat, ptr addrspace(1) %in, i64 %tid64
  %val = load bfloat, ptr addrspace(1) %in_gep, align 2
  %tg_gep = getelementptr bfloat, ptr addrspace(3) @__tg_cvt_0, i64 %tid64
  store bfloat %val, ptr addrspace(3) %tg_gep, align 2
  call void @air.threadgroup.barrier(i32 2, i32 1)
  %read_idx = sub i32 31, %tid_x
  %read_idx64 = zext i32 %read_idx to i64
  %tg_gep2 = getelementptr bfloat, ptr addrspace(3) @__tg_cvt_0, i64 %read_idx64
  %val2 = load bfloat, ptr addrspace(3) %tg_gep2, align 2
  %out_gep = getelementptr bfloat, ptr addrspace(1) %out, i64 %tid64
  store bfloat %val2, ptr addrspace(1) %out_gep, align 2
  ret void
}
!air.kernel = !{!0}
!air.version = !{!7}
!air.language_version = !{!8}
!llvm.module.flags = !{!9, !10, !11, !12, !13, !14}
!0 = !{void (ptr addrspace(1), ptr addrspace(1), i32, i32, i32)* @test_kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !15}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"bfloat", !"air.arg_name", !"in"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"bfloat", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_x"}
!6 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_y"}
!15 = !{i32 4, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_z"}
!7 = !{i32 2, i32 8, i32 0}
!8 = !{!"Metal", i32 3, i32 2, i32 0}
!9 = !{i32 7, !"air.max_device_buffers", i32 31}
!10 = !{i32 7, !"air.max_constant_buffers", i32 31}
!11 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!12 = !{i32 7, !"air.max_textures", i32 128}
!13 = !{i32 7, !"air.max_read_write_textures", i32 8}
!14 = !{i32 7, !"air.max_samplers", i32 16}
