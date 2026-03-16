; Minimal test for LowerAtomicRMW: atomicrmw → air.atomic.* calls

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

define void @test_kernel(ptr addrspace(1) %buf, i32 %tid) {
entry:
  %p = getelementptr i32, ptr addrspace(1) %buf, i32 %tid
  %old = atomicrmw add ptr addrspace(1) %p, i32 1 seq_cst
  ret void
}
