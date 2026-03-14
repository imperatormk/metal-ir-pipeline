target triple = "air64_v28-apple-macosx26.0.0"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"

define void @kernel(float addrspace(1)* %0, i32 addrspace(1)* %1, <3 x i32> %2) {
  %4 = extractelement <3 x i32> %2, i32 0
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %body, label %exit

body:
  br label %loop

loop:
  %count = phi i32 [ 8, %body ], [ %newcount, %loop ]
  %val = load float, float addrspace(1)* %0, align 4
  %newacc = fadd float %val, 1.000000e+00
  store float %newacc, float addrspace(1)* %0, align 4
  %newcount = sub i32 %count, 2
  %test = icmp sgt i32 %newcount, 0
  br i1 %test, label %loop, label %done

done:
  ret void

exit:
  ret void
}

!air.kernel = !{!0}
!air.version = !{!6}
!air.language_version = !{!7}
!llvm.module.flags = !{!8, !9, !10, !11, !12, !13}

!0 = !{void (float addrspace(1)*, i32 addrspace(1)*, <3 x i32>)* @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"data"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"n"}
!5 = !{i32 2, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"tidtg"}
!6 = !{i32 2, i32 8, i32 0}
!7 = !{!"Metal", i32 3, i32 2, i32 0}
!8 = !{i32 7, !"air.max_device_buffers", i32 31}
!9 = !{i32 7, !"air.max_constant_buffers", i32 31}
!10 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!11 = !{i32 7, !"air.max_textures", i32 128}
!12 = !{i32 7, !"air.max_read_write_textures", i32 8}
!13 = !{i32 7, !"air.max_samplers", i32 16}
