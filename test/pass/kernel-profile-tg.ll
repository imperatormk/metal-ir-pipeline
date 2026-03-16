; Test KernelProfile: TG kernel with per-thread index (not scalar)

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

@shared = internal addrspace(3) global [128 x float] undef

declare [3 x i32] @air.thread_position_in_threadgroup()

define void @test_kernel(ptr addrspace(1) %buf) {
entry:
  %tid = call [3 x i32] @air.thread_position_in_threadgroup()
  %tid_x = extractvalue [3 x i32] %tid, 0
  %p_dev = getelementptr float, ptr addrspace(1) %buf, i32 %tid_x
  %v = load float, ptr addrspace(1) %p_dev
  %p_tg = getelementptr [128 x float], ptr addrspace(3) @shared, i32 0, i32 %tid_x
  store float %v, ptr addrspace(3) %p_tg
  %v2 = load float, ptr addrspace(3) %p_tg
  store float %v2, ptr addrspace(1) %p_dev
  ret void
}
