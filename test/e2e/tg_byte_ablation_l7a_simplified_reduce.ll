target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@global_smem = internal addrspace(3) global [16 x i8] undef, align 16
declare void @air.wg.barrier(i32, i32)
declare float @air.simd_shuffle_xor.f32(float, i16)
declare [3 x i32] @air.thread_position_in_threadgroup()
declare [3 x i32] @air.threadgroup_position_in_grid()
define void @kern(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(2) %2) {
  %n = load i32, ptr addrspace(2) %2, align 4
  %pid3 = call [3 x i32] @air.threadgroup_position_in_grid()
  %pid = extractvalue [3 x i32] %pid3, 0
  %tid3 = call [3 x i32] @air.thread_position_in_threadgroup()
  %tid = extractvalue [3 x i32] %tid3, 0
  %idx = add i32 %pid, %tid
  %inptr = getelementptr float, ptr addrspace(1) %0, i32 %idx
  %inval = load float, ptr addrspace(1) %inptr, align 4
  %cmp = icmp slt i32 %idx, %n
  %val = select i1 %cmp, float %inval, float 0.0
  %s1 = call float @air.simd_shuffle_xor.f32(float %val, i16 16)
  %a1 = fadd float %val, %s1
  %s2 = call float @air.simd_shuffle_xor.f32(float %a1, i16 8)
  %a2 = fadd float %a1, %s2
  %s3 = call float @air.simd_shuffle_xor.f32(float %a2, i16 4)
  %a3 = fadd float %a2, %s3
  %s4 = call float @air.simd_shuffle_xor.f32(float %a3, i16 2)
  %a4 = fadd float %a3, %s4
  %s5 = call float @air.simd_shuffle_xor.f32(float %a4, i16 1)
  %a5 = fadd float %a4, %s5
  %warpid = udiv i32 %tid, 32
  %woff = shl i32 %warpid, 2
  %tgptr = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %woff
  %vec = insertelement <1 x float> undef, float %a5, i32 0
  store <1 x float> %vec, ptr addrspace(3) %tgptr, align 4
  call void @air.wg.barrier(i32 1, i32 1)
  %lane = urem i32 %tid, 32
  %roff = shl i32 %lane, 2
  %rptr = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %roff
  %ld = load <1 x float>, ptr addrspace(3) %rptr, align 4
  %ldval = extractelement <1 x float> %ld, i32 0
  %r1 = call float @air.simd_shuffle_xor.f32(float %ldval, i16 2)
  %b1 = fadd float %ldval, %r1
  %r2 = call float @air.simd_shuffle_xor.f32(float %b1, i16 1)
  %b2 = fadd float %b1, %r2
  %outptr = getelementptr float, ptr addrspace(1) %1, i32 %pid
  store float %b2, ptr addrspace(1) %outptr, align 4
  ret void
}
