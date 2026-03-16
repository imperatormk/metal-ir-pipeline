; Minimal test for ScalarStoreGuard: scalar kernel → tid==0 guard.
; This kernel has no per-thread indexing (no air.thread_position_*),
; so it should get a tid==0 guard around its store.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

define void @test_kernel(ptr addrspace(1) %out) {
entry:
  store float 4.200000e+01, ptr addrspace(1) %out, align 4
  ret void
}
