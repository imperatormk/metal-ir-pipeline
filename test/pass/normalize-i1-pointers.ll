; Minimal test for NormalizeI1Pointers: i1 GEP source → i8

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

define void @test_kernel(ptr addrspace(1) %mask, ptr addrspace(1) %out, i32 %tid) {
entry:
  %p = getelementptr i1, ptr addrspace(1) %mask, i32 %tid
  %v = load i1, ptr addrspace(1) %p, align 1
  %ext = zext i1 %v to i32
  %po = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store i32 %ext, ptr addrspace(1) %po, align 4
  ret void
}
