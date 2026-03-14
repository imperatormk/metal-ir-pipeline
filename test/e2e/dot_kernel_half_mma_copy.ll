target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

source_filename = "LLVMDialectModule"

declare [3 x i32] @air.thread_position_in_threadgroup()

define void @copy_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1) {
  %tid = call [3 x i32] @air.thread_position_in_threadgroup()
  %tidx = extractvalue [3 x i32] %tid, 0

  ; Load half from input (chained GEPs like real dot kernel)
  %base_gep = getelementptr half, ptr addrspace(1) %0, i32 0
  %elem_gep = getelementptr half, ptr addrspace(1) %base_gep, i32 %tidx
  %val = load half, ptr addrspace(1) %elem_gep, align 2

  ; Store half to output (store-only ptr, chained GEPs)
  %out_base = getelementptr half, ptr addrspace(1) %1, i32 0
  %out_elem = getelementptr half, ptr addrspace(1) %out_base, i32 %tidx
  store half %val, ptr addrspace(1) %out_elem, align 2

  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
