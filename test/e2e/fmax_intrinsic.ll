target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

source_filename = "LLVMDialectModule"

declare float @llvm.maxnum.f32(float, float)
declare float @air.simd_shuffle_xor.f32(float, i16)

define void @fmax_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1) {
  %tid = call [3 x i32] @air.thread_position_in_threadgroup()
  %tid_x = extractvalue [3 x i32] %tid, 0
  %idx = zext i32 %tid_x to i64
  %p0 = getelementptr float, ptr addrspace(1) %0, i64 %idx
  %v0 = load float, ptr addrspace(1) %p0
  %idx1 = add i64 %idx, 1
  %p1 = getelementptr float, ptr addrspace(1) %0, i64 %idx1
  %v1 = load float, ptr addrspace(1) %p1
  %mx = call float @llvm.maxnum.f32(float %v0, float %v1)
  ; Also test shuffle
  %sh = call float @air.simd_shuffle_xor.f32(float %mx, i16 1)
  %final = fadd float %mx, %sh
  %p2 = getelementptr float, ptr addrspace(1) %1, i64 %idx
  store float %final, ptr addrspace(1) %p2
  ret void
}

declare [3 x i32] @air.thread_position_in_threadgroup()
