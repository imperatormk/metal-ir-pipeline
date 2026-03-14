target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

declare [3 x i32] @air.thread_position_in_threadgroup()

define void @loop_store_load(ptr addrspace(1) %0) {
  %tid = call [3 x i32] @air.thread_position_in_threadgroup()
  %tidx = extractvalue [3 x i32] %tid, 0
  br label %loop
loop:
  %i = phi i32 [ %i_next, %loop ], [ 0, %1 ]
  %val = load volatile float, ptr addrspace(1) %0, align 4
  %val1 = fadd float %val, 1.000000e+00
  store volatile float %val1, ptr addrspace(1) %0, align 4
  %i_next = add i32 %i, 1
  %cmp = icmp slt i32 %i_next, 40
  br i1 %cmp, label %loop, label %done
done:
  ret void
}
