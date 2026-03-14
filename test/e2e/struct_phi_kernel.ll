; Test: DecomposeStructPhis pass
; Struct phi nodes from insertvalue chains → scalar phis
; Also tests full pipeline → metallib → GPU PSO creation

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

define void @kernel_add(ptr addrspace(1) %out, i32 %tid_x) {
entry:
  %i0 = insertvalue { float, float } undef, float 0.0, 0
  %init = insertvalue { float, float } %i0, float 1.0, 1
  br label %loop

loop:
  %s = phi { float, float } [ %init, %entry ], [ %updated, %loop ]
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]

  %a = extractvalue { float, float } %s, 0
  %b = extractvalue { float, float } %s, 1
  %new_a = fadd float %a, 1.0
  %new_b = fadd float %b, 2.0

  %t1 = insertvalue { float, float } undef, float %new_a, 0
  %updated = insertvalue { float, float } %t1, float %new_b, 1

  %i_next = add i32 %i, 1
  %cond = icmp slt i32 %i_next, 4
  br i1 %cond, label %loop, label %exit

exit:
  %final = extractvalue { float, float } %s, 0
  %p = getelementptr float, ptr addrspace(1) %out, i32 %tid_x
  store float %final, ptr addrspace(1) %p, align 4
  ret void
}

!air.kernel = !{!1}
!air.version = !{!8}

!1 = !{ptr @kernel_add, !2, !3}
!2 = !{}
!3 = !{!4, !10}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!10 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_x"}
!8 = !{i32 2, i32 8, i32 0}
