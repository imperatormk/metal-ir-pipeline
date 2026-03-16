; Minimal test for NaNMinMax: llvm.minimum → air.fmin + NaN select

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

declare float @llvm.minimum.f32(float, float)

define void @test_kernel(ptr addrspace(1) %A, ptr addrspace(1) %B, ptr addrspace(1) %C, i32 %tid) {
entry:
  %idx = zext i32 %tid to i64
  %pa = getelementptr float, ptr addrspace(1) %A, i64 %idx
  %a = load float, ptr addrspace(1) %pa
  %pb = getelementptr float, ptr addrspace(1) %B, i64 %idx
  %b = load float, ptr addrspace(1) %pb
  %r = call float @llvm.minimum.f32(float %a, float %b)
  %pc = getelementptr float, ptr addrspace(1) %C, i64 %idx
  store float %r, ptr addrspace(1) %pc
  ret void
}
