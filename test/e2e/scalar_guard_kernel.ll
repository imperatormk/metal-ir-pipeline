; Test: ScalarStoreGuard + AIRSystemValues cooperation
; Kernel uses only pid (threadgroup_position), stores to device → needs tid.x==0 guard
; ScalarStoreGuard adds the tidtg call, AIRSystemValues converts both to params

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

declare { i32, i32, i32 } @air.threadgroup_position_in_grid()

define void @kernel_add(ptr addrspace(1) %out) {
entry:
  %pid_struct = call { i32, i32, i32 } @air.threadgroup_position_in_grid()
  %pid_x = extractvalue { i32, i32, i32 } %pid_struct, 0
  %p = getelementptr float, ptr addrspace(1) %out, i32 %pid_x
  store float 42.0, ptr addrspace(1) %p, align 4
  ret void
}
