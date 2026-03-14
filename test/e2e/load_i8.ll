target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define void @load_i8_test(ptr addrspace(1) %0, ptr addrspace(1) %1) {
  %tid = call [3 x i32] @air.thread_position_in_threadgroup()
  %idx = extractvalue [3 x i32] %tid, 0
  %p = getelementptr i8, ptr addrspace(1) %0, i32 %idx
  %v = load i8, ptr addrspace(1) %p, align 1
  %ext = zext i8 %v to i32
  %po = getelementptr i32, ptr addrspace(1) %1, i32 %idx
  store i32 %ext, ptr addrspace(1) %po, align 4
  ret void
}

declare [3 x i32] @air.thread_position_in_threadgroup()
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
