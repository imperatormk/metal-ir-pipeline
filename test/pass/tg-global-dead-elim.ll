; Minimal test for TGGlobalDeadElim: unused AS3 global → removed

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

@used_smem = internal addrspace(3) global [64 x float] undef
@dead_smem = internal addrspace(3) global [128 x float] undef

define void @test_kernel(ptr addrspace(1) %out) {
entry:
  %p = getelementptr [64 x float], ptr addrspace(3) @used_smem, i32 0, i32 0
  %v = load float, ptr addrspace(3) %p
  store float %v, ptr addrspace(1) %out
  ret void
}
