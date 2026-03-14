source_filename = "codebook_test"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@CODEBOOK = internal addrspace(2) constant [4 x float] [
  float 1.0, float 2.0, float 3.0, float 4.0
]

define void @codebook_lookup(
  i8 addrspace(1)* noundef "air-buffer-no-alias" %input_raw,
  i8 addrspace(1)* noundef "air-buffer-no-alias" %output_raw,
  i32 %tid_scalar
) {
entry:
  %input = bitcast i8 addrspace(1)* %input_raw to i32 addrspace(1)*
  %output = bitcast i8 addrspace(1)* %output_raw to float addrspace(1)*
  %idx_ptr = getelementptr i32, i32 addrspace(1)* %input, i32 %tid_scalar
  %idx = load i32, i32 addrspace(1)* %idx_ptr
  %masked = and i32 %idx, 3
  %cb_ptr = getelementptr [4 x float], [4 x float] addrspace(2)* @CODEBOOK, i32 0, i32 %masked
  %val = load float, float addrspace(2)* %cb_ptr
  %out_ptr = getelementptr float, float addrspace(1)* %output, i32 %tid_scalar
  store float %val, float addrspace(1)* %out_ptr
  ret void
}

!air.kernel = !{!1}
!llvm.module.flags = !{!10, !11, !12, !13, !14, !15, !16}
!air.version = !{!20}
!air.language_version = !{!21}
!air.compile_options = !{!30}
!1 = !{void (i8 addrspace(1)*, i8 addrspace(1)*, i32)* @codebook_lookup, !2, !3}
!2 = !{}
!3 = !{!4, !5, !6}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"input"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"output"}
!6 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{i32 7, !"air.max_device_buffers", i32 31}
!12 = !{i32 7, !"air.max_constant_buffers", i32 31}
!13 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!14 = !{i32 7, !"air.max_textures", i32 128}
!15 = !{i32 7, !"air.max_read_write_textures", i32 8}
!16 = !{i32 7, !"air.max_samplers", i32 16}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
!30 = !{!"air.compile.fast_math_enable"}
