; Negative test: fsub (not fneg) should NOT be transformed

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

define void @test_kernel(ptr addrspace(1) %A, ptr addrspace(1) %B, i32 %tid) {
entry:
  %pa = getelementptr float, ptr addrspace(1) %A, i32 %tid
  %v = load float, ptr addrspace(1) %pa, align 4
  %sub = fsub float %v, 1.0
  %pb = getelementptr float, ptr addrspace(1) %B, i32 %tid
  store float %sub, ptr addrspace(1) %pb, align 4
  ret void
}
