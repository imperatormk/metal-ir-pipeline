target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@__tg_dot_ab_0 = internal addrspace(3) global [64 x float] undef, align 4

declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)
declare [3 x i32] @air.thread_position_in_threadgroup()
declare i32 @air.thread_index_in_simdgroup()

define void @test_chained_half_gep(ptr addrspace(1) %out) {
  %tid_s = call [3 x i32] @air.thread_position_in_threadgroup()
  %tid = extractvalue [3 x i32] %tid_s, 0
  %lane = call i32 @air.thread_index_in_simdgroup()
  %warp = udiv i32 %tid, 32

  ; Dest encoding: spt=[2,1] tpw=[8,4] wpc=[1,4] order=[0,1]
  %lR = urem i32 %lane, 8
  %lC = udiv i32 %lane, 8
  %row0 = mul i32 %lR, 2
  %row1 = add i32 %row0, 1
  %wC = mul i32 %warp, 4
  %col = add i32 %wC, %lC

  ; Values: val[row][col] = row*100 + col
  %v0_r = mul i32 %row0, 100
  %v0_i = add i32 %v0_r, %col
  %v1_r = mul i32 %row1, 100
  %v1_i = add i32 %v1_r, %col
  %v0_f = uitofp i32 %v0_i to float
  %v1_f = uitofp i32 %v1_i to float
  %val0 = fptrunc float %v0_f to half
  %val1 = fptrunc float %v1_f to half

  ; Col-major store via chained GEP: out[row + col*16]
  ; First GEP: base = out + row
  ; Second GEP: addr = base + col*16
  %col16 = mul i32 %col, 16
  %base0 = getelementptr half, ptr addrspace(1) %out, i32 %row0
  %addr0 = getelementptr half, ptr addrspace(1) %base0, i32 %col16
  store half %val0, ptr addrspace(1) %addr0, align 2

  %base1 = getelementptr half, ptr addrspace(1) %out, i32 %row1
  %addr1 = getelementptr half, ptr addrspace(1) %base1, i32 %col16
  store half %val1, ptr addrspace(1) %addr1, align 2

  ret void
}
