; Negative test: kernel WITH per-thread index should NOT get tid==0 guard

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

declare [3 x i32] @air.thread_position_in_threadgroup()

define void @test_kernel(ptr addrspace(1) %out) {
entry:
  %tid = call [3 x i32] @air.thread_position_in_threadgroup()
  %tid_x = extractvalue [3 x i32] %tid, 0
  %p = getelementptr float, ptr addrspace(1) %out, i32 %tid_x
  store float 1.0, ptr addrspace(1) %p
  ret void
}
