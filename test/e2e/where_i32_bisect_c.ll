target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

source_filename = "LLVMDialectModule"
@global_smem = internal addrspace(3) global [4096 x i8] undef, align 16
declare void @air.wg.barrier(i32, i32)
define void @bisect_c(ptr addrspace(1) %cond, ptr addrspace(1) %x, ptr addrspace(1) %y, ptr addrspace(1) %out) {
  %cp = getelementptr i1, ptr addrspace(1) %cond, i32 0
  %cv = load i8, ptr addrspace(1) %cp, align 1
  %cb = icmp ne i8 %cv, 0
  %xp = getelementptr i32, ptr addrspace(1) %x, i32 0
  %xv = load i32, ptr addrspace(1) %xp, align 4
  %yp = getelementptr i32, ptr addrspace(1) %y, i32 0
  %yv = load i32, ptr addrspace(1) %yp, align 4
  %sel = select i1 %cb, i32 %xv, i32 %yv
  %v0 = insertelement <4 x i32> undef, i32 %sel, i32 0
  %v1 = insertelement <4 x i32> %v0, i32 %sel, i32 1
  %v2 = insertelement <4 x i32> %v1, i32 %sel, i32 2
  %v3 = insertelement <4 x i32> %v2, i32 %sel, i32 3
  %tg = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 0
  store <4 x i32> %v3, ptr addrspace(3) %tg, align 16
  call void @air.wg.barrier(i32 1, i32 1)
  %ld = load <4 x i32>, ptr addrspace(3) %tg, align 16
  %r = extractelement <4 x i32> %ld, i32 0
  %op = getelementptr i32, ptr addrspace(1) %out, i32 0
  store i32 %r, ptr addrspace(1) %op, align 4
  ret void
}
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
