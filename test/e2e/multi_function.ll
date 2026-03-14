target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

declare [3 x i32] @air.threadgroup_position_in_grid()

define void @kernel(ptr addrspace(1) %0) {
  %2 = call [3 x i32] @air.threadgroup_position_in_grid()
  %3 = extractvalue [3 x i32] %2, 0
  %4 = load i32, ptr addrspace(1) %0, align 4
  %5 = call i32 @add_fn(i32 %4, i32 %3)
  store i32 %5, ptr addrspace(1) %0, align 4
  ret void
}

define i32 @add_fn(i32 %0, i32 %1) {
  %3 = icmp eq i32 %1, 0
  %4 = select i1 %3, i32 1, i32 2
  %5 = add i32 %0, %4
  ret i32 %5
}
