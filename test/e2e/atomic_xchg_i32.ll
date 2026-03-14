target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

source_filename = "LLVMDialectModule"

declare i32 @air.atomic.global.xchg.i32(ptr addrspace(1), i32, i32, i32, i1)

define void @xchg_kernel(ptr addrspace(1) %0, i32 %1) {
  %3 = call i32 @air.atomic.global.xchg.i32(ptr addrspace(1) %0, i32 %1, i32 0, i32 2, i1 true)
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
