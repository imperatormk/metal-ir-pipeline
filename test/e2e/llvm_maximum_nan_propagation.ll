target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

source_filename = "test"

declare float @llvm.maximum.f32(float, float)

define void @max_nan_kernel(ptr addrspace(1) %A, ptr addrspace(1) %B, ptr addrspace(1) %C) {
  %tid = call [3 x i32] @air.thread_position_in_threadgroup()
  %tid_x = extractvalue [3 x i32] %tid, 0
  %idx = zext i32 %tid_x to i64
  %pa = getelementptr float, ptr addrspace(1) %A, i64 %idx
  %a = load float, ptr addrspace(1) %pa
  %pb = getelementptr float, ptr addrspace(1) %B, i64 %idx
  %b = load float, ptr addrspace(1) %pb
  %r = call float @llvm.maximum.f32(float %a, float %b)
  %pc = getelementptr float, ptr addrspace(1) %C, i64 %idx
  store float %r, ptr addrspace(1) %pc
  ret void
}

declare [3 x i32] @air.thread_position_in_threadgroup()
