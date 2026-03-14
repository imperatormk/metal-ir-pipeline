target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@global_smem = internal addrspace(3) global [16 x i8] undef, align 16
declare void @air.wg.barrier(i32, i32)
define void @kern(ptr addrspace(1) %0) {
  %ptr = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 0
  %vec = insertelement <1 x float> undef, float 42.0, i32 0
  store <1 x float> %vec, ptr addrspace(3) %ptr, align 4
  call void @air.wg.barrier(i32 1, i32 1)
  %ld = load <1 x float>, ptr addrspace(3) %ptr, align 4
  %val = extractelement <1 x float> %ld, i32 0
  store float %val, ptr addrspace(1) %0, align 4
  ret void
}
