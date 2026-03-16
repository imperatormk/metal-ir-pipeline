; Test for ScalarBufferPacking: scalar params → one device buffer + GEP/load

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

define void @test_kernel(ptr addrspace(1) %out, i32 %n, float %scale) {
entry:
  %p = getelementptr float, ptr addrspace(1) %out, i32 %n
  %v = fmul float %scale, 2.0
  store float %v, ptr addrspace(1) %p
  ret void
}
