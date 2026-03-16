; Minimal test for LLVMToAIRIntrinsics: llvm.sin → air.fast_sin etc.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

declare float @llvm.sin.f32(float)
declare float @llvm.cos.f32(float)
declare float @llvm.sqrt.f32(float)
declare float @llvm.exp.f32(float)
declare float @llvm.log.f32(float)

define void @test_kernel(ptr addrspace(1) %A, ptr addrspace(1) %B, i32 %tid) {
entry:
  %p = getelementptr float, ptr addrspace(1) %A, i32 %tid
  %v = load float, ptr addrspace(1) %p, align 4
  %s = call float @llvm.sin.f32(float %v)
  %c = call float @llvm.cos.f32(float %s)
  %sq = call float @llvm.sqrt.f32(float %c)
  %e = call float @llvm.exp.f32(float %sq)
  %l = call float @llvm.log.f32(float %e)
  %po = getelementptr float, ptr addrspace(1) %B, i32 %tid
  store float %l, ptr addrspace(1) %po, align 4
  ret void
}
