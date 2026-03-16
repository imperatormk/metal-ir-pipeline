; Test for NormalizeAllocas: alloca with i64 size → i32 size

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

define void @test_kernel(ptr addrspace(1) %out, i32 %tid) {
entry:
  %buf = alloca float, i64 4, align 4
  %p = getelementptr float, ptr %buf, i32 0
  store float 1.0, ptr %p, align 4
  %v = load float, ptr %p, align 4
  %po = getelementptr float, ptr addrspace(1) %out, i32 %tid
  store float %v, ptr addrspace(1) %po, align 4
  ret void
}
