; Minimal test for BarrierRename: air.threadgroup.barrier → air.wg.barrier

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

declare void @air.threadgroup.barrier(i32, i32)

define void @test_kernel(ptr addrspace(1) %buf, i32 %tid) {
entry:
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %p = getelementptr float, ptr addrspace(1) %buf, i32 %tid
  %v = load float, ptr addrspace(1) %p, align 4
  store float %v, ptr addrspace(1) %p, align 4
  ret void
}
