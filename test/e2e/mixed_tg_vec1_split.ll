; Regression test: mixed-type TG global with <1 x float> and <1 x i32> stores.
; The split pass must scalarize vec1 BEFORE checking for mixed types,
; otherwise <1 x float> isn't detected as a scalar type and the global
; doesn't get split → materializeAll.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

@global_smem = internal addrspace(3) global [32 x i8] undef, align 16

declare void @air.wg.barrier(i32, i32)
declare [3 x i32] @air.thread_position_in_threadgroup()

define void @kernel(ptr addrspace(1) %out_f, ptr addrspace(1) %out_i) {
entry:
  %tid = call [3 x i32] @air.thread_position_in_threadgroup()
  %tid_x = extractvalue [3 x i32] %tid, 0

  ; Store <1 x float> at offset 0
  %p0 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 0
  %val_f = insertelement <1 x float> undef, float 42.0, i32 0
  store <1 x float> %val_f, ptr addrspace(3) %p0, align 4

  ; Store <1 x i32> at offset 16
  %p16 = getelementptr inbounds [32 x i8], ptr addrspace(3) @global_smem, i64 0, i64 16
  %val_i = insertelement <1 x i32> undef, i32 7, i32 0
  store <1 x i32> %val_i, ptr addrspace(3) %p16, align 4

  call void @air.wg.barrier(i32 2, i32 1)

  ; Load back
  %ld_f = load <1 x float>, ptr addrspace(3) %p0, align 4
  %sf = extractelement <1 x float> %ld_f, i32 0
  store float %sf, ptr addrspace(1) %out_f, align 4

  %ld_i = load <1 x i32>, ptr addrspace(3) %p16, align 4
  %si = extractelement <1 x i32> %ld_i, i32 0
  store i32 %si, ptr addrspace(1) %out_i, align 4

  ret void
}
