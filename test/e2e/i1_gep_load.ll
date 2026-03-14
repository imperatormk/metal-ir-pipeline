target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

source_filename = "LLVMDialectModule"

define void @where_i32_kernel(ptr addrspace(1) %cond, ptr addrspace(1) %x, ptr addrspace(1) %y, ptr addrspace(1) %out) {
  %cond_ptr = getelementptr i1, ptr addrspace(1) %cond, i32 0
  %cond_val = load i8, ptr addrspace(1) %cond_ptr, align 1
  %cond_bool = icmp ne i8 %cond_val, 0
  %x_ptr = getelementptr i32, ptr addrspace(1) %x, i32 0
  %x_val = load i32, ptr addrspace(1) %x_ptr, align 4
  %y_ptr = getelementptr i32, ptr addrspace(1) %y, i32 0
  %y_val = load i32, ptr addrspace(1) %y_ptr, align 4
  %sel = select i1 %cond_bool, i32 %x_val, i32 %y_val
  %out_ptr = getelementptr i32, ptr addrspace(1) %out, i32 0
  store i32 %sel, ptr addrspace(1) %out_ptr, align 4
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
