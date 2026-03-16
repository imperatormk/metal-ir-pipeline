; Minimal test for InlineNonKernelFunctions: helper → inlined + removed

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

define internal float @helper(float %x) {
  %r = fadd float %x, 1.0
  ret float %r
}

define void @test_kernel(ptr addrspace(1) %buf, i32 %tid) {
entry:
  %p = getelementptr float, ptr addrspace(1) %buf, i32 %tid
  %v = load float, ptr addrspace(1) %p, align 4
  %r = call float @helper(float %v)
  store float %r, ptr addrspace(1) %p, align 4
  ret void
}
