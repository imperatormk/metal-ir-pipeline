target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

declare i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1), ptr, i32, i32, i32, i32, i1)

define void @kernel(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %tid = call [3 x i32] @air.thread_position_in_threadgroup()
  %tidx = extractvalue [3 x i32] %tid, 0
  %cond = icmp eq i32 %tidx, 0
  br i1 %cond, label %body, label %exit

body:
  ; bfloat load
  %gep = getelementptr bfloat, ptr addrspace(1) %in, i32 0
  %val = load bfloat, ptr addrspace(1) %gep, align 2

  ; alloca for cmpxchg
  %expected = alloca i32, align 4
  store i32 0, ptr %expected, align 4

  ; cmpxchg
  %result = call i32 @air.atomic.global.cmpxchg.weak.i32(
    ptr addrspace(1) %out, ptr %expected,
    i32 42, i32 0, i32 0, i32 2, i1 true)

  ; store i32 result (not bfloat)
  %gep2 = getelementptr i32, ptr addrspace(1) %out, i32 0
  store i32 %result, ptr addrspace(1) %gep2, align 4
  ret void

exit:
  ret void
}

declare [3 x i32] @air.thread_position_in_threadgroup()
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
