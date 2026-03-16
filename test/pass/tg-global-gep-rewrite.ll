; Minimal test for TGGlobalGEPRewrite: [N x i8] TG global → [M x float]

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

@global_smem = internal addrspace(3) global [16 x i8] undef, align 16

define void @test_kernel(ptr addrspace(1) %out) {
entry:
  %ptr = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 0
  store float 4.200000e+01, ptr addrspace(3) %ptr, align 4
  %val = load float, ptr addrspace(3) %ptr, align 4
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}
