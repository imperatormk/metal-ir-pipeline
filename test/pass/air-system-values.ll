; Test for AIRSystemValues: system value calls → kernel params + metadata
; Pre-condition: scalar-buffer-packing has already run.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

declare [3 x i32] @air.threadgroup_position_in_grid()

define void @test_kernel(ptr addrspace(1) %out) {
entry:
  %pid = call [3 x i32] @air.threadgroup_position_in_grid()
  %pid_x = extractvalue [3 x i32] %pid, 0
  %p = getelementptr float, ptr addrspace(1) %out, i32 %pid_x
  store float 1.0, ptr addrspace(1) %p
  ret void
}
