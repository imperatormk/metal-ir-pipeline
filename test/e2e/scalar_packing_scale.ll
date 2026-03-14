target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @scale_kernel(ptr addrspace(1) %in, ptr addrspace(1) %out, float %scale) {
  %tid3 = call <3 x i32> @air.thread_position_in_grid.v3i32()
  %idx = extractelement <3 x i32> %tid3, i32 0
  %p_in = getelementptr inbounds float, ptr addrspace(1) %in, i32 %idx
  %val = load float, ptr addrspace(1) %p_in, align 4
  %mul = fmul float %val, %scale
  %p_out = getelementptr inbounds float, ptr addrspace(1) %out, i32 %idx
  store float %mul, ptr addrspace(1) %p_out, align 4
  ret void
}
declare <3 x i32> @air.thread_position_in_grid.v3i32()
