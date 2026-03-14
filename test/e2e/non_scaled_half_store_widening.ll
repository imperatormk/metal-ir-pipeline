target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@__tg_dot_ab_0 = internal addrspace(3) global [64 x float] undef, align 4

declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)
declare [3 x i32] @air.thread_position_in_threadgroup()

define void @test_nonscaled_half(ptr addrspace(1) %out) {
  %tid_s = call [3 x i32] @air.thread_position_in_threadgroup()
  %tid = extractvalue [3 x i32] %tid_s, 0

  ; Flat indices: idx0 = tid*2, idx1 = tid*2+1
  %idx0 = mul i32 %tid, 2
  %idx1 = add i32 %idx0, 1

  ; Values: val0 = tid*10, val1 = tid*10+5
  %v0_i = mul i32 %tid, 10
  %v1_i = add i32 %v0_i, 5
  %v0_f = uitofp i32 %v0_i to float
  %v1_f = uitofp i32 %v1_i to float
  %val0 = fptrunc float %v0_f to half
  %val1 = fptrunc float %v1_f to half

  ; Chained GEP with constant-0 second index (identity GEP).
  ; The constant-0 GEP must propagate the half-scaled info from base.
  %base0 = getelementptr half, ptr addrspace(1) %out, i32 %idx0
  %addr0 = getelementptr half, ptr addrspace(1) %base0, i32 0
  store half %val0, ptr addrspace(1) %addr0, align 2

  %base1 = getelementptr half, ptr addrspace(1) %out, i32 %idx1
  %addr1 = getelementptr half, ptr addrspace(1) %base1, i32 0
  store half %val1, ptr addrspace(1) %addr1, align 2

  ret void
}
