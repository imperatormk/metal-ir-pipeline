target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

declare i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1), ptr, i32, i32, i32, i32, i1)
declare i32 @air.atomic.global.xchg.i32(ptr addrspace(1), i32, i32, i32, i1)

define void @volatile_cas_repro(ptr addrspace(1) %flag,
                                ptr addrspace(1) %data,
                                ptr addrspace(1) %out,
                                i32 %tid_x) {
entry:
  %is_writer = icmp eq i32 %tid_x, 0
  br i1 %is_writer, label %writer, label %check_reader

check_reader:
  %is_reader = icmp eq i32 %tid_x, 1
  br i1 %is_reader, label %reader, label %exit

writer:
  %data0 = getelementptr float, ptr addrspace(1) %data, i32 0
  store float 4.200000e+01, ptr addrspace(1) %data0, align 4
  %flag0 = getelementptr i32, ptr addrspace(1) %flag, i32 0
  %publish = call i32 @air.atomic.global.xchg.i32(ptr addrspace(1) %flag0,
                                                  i32 1, i32 0, i32 1,
                                                  i1 true)
  br label %exit

reader:
  %flag1 = getelementptr i32, ptr addrspace(1) %flag, i32 0
  %expected = alloca i32, align 4
  br label %spin

spin:
  store i32 1, ptr %expected, align 4
  %seen = call i32 @air.atomic.global.cmpxchg.weak.i32(
      ptr addrspace(1) %flag1, ptr %expected, i32 1, i32 0, i32 0, i32 2,
      i1 true)
  %ready = icmp eq i32 %seen, 1
  br i1 %ready, label %consume, label %spin

consume:
  %data1 = getelementptr float, ptr addrspace(1) %data, i32 0
  %val = load float, ptr addrspace(1) %data1, align 4
  %out0 = getelementptr float, ptr addrspace(1) %out, i32 0
  store float %val, ptr addrspace(1) %out0, align 4
  br label %exit

exit:
  ret void
}

!llvm.module.flags = !{!0}
!air.kernel = !{!1}
!air.version = !{!8}

!0 = !{i32 7, !"frame-pointer", i32 0}
!1 = !{ptr @volatile_cas_repro, !2, !3}
!2 = !{}
!3 = !{!4, !5, !6, !7}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"flag"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"data"}
!6 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!7 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_x"}
!8 = !{i32 2, i32 8, i32 0}
