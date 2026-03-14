target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@global_smem = internal addrspace(3) global [16 x i8] undef, align 16
declare float @air.simd_shuffle_xor.f32(float, i16)
declare void @air.wg.barrier(i32, i32)
declare [3 x i32] @air.threadgroup_position_in_grid()
declare [3 x i32] @air.thread_position_in_threadgroup()
define void @kern(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(2) %2) {
  %n = load i32, ptr addrspace(2) %2, align 4
  %pid3 = call [3 x i32] @air.threadgroup_position_in_grid()
  %pid = extractvalue [3 x i32] %pid3, 0
  %tid3 = call [3 x i32] @air.thread_position_in_threadgroup()
  %tid = extractvalue [3 x i32] %tid3, 0
  %s1 = call float @air.simd_shuffle_xor.f32(float 1.0, i16 1)
  %warpid = udiv i32 %tid, 32
  %offset = shl i32 %warpid, 2
  %ptr = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %offset
  %vec = insertelement <1 x float> undef, float %s1, i32 0
  store <1 x float> %vec, ptr addrspace(3) %ptr, align 4
  call void @air.wg.barrier(i32 1, i32 1)
  %lane = urem i32 %tid, 32
  %roff = shl i32 %lane, 2
  %rptr = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %roff
  %ld = load <1 x float>, ptr addrspace(3) %rptr, align 4
  %val = extractelement <1 x float> %ld, i32 0
  %s2 = call float @air.simd_shuffle_xor.f32(float %val, i16 1)
  %sum = fadd float %val, %s2
  store float %sum, ptr addrspace(1) %1, align 4
  ret void
}
