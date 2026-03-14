target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

source_filename = "LLVMDialectModule"
define void @bisect_b(ptr addrspace(1) %cond, ptr addrspace(1) %x, ptr addrspace(1) %y, ptr addrspace(1) %out) {
  %cp = getelementptr i1, ptr addrspace(1) %cond, i32 0
  %cv = load i8, ptr addrspace(1) %cp, align 1
  %cb = icmp ne i8 %cv, 0
  %xp = getelementptr i32, ptr addrspace(1) %x, i32 0
  %xv = load i32, ptr addrspace(1) %xp, align 4
  %yp = getelementptr i32, ptr addrspace(1) %y, i32 0
  %yv = load i32, ptr addrspace(1) %yp, align 4
  %sel = select i1 %cb, i32 %xv, i32 %yv
  %op = getelementptr i32, ptr addrspace(1) %out, i32 0
  store i32 %sel, ptr addrspace(1) %op, align 4
  ret void
}
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
