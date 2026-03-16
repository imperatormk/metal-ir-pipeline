; Test for PtrPhiToI64: convert ptr phis to i64 when count > 32 per block.
; This test has only 2 ptr phis so it should NOT convert (threshold is 32).

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

define void @test_kernel(ptr addrspace(1) %a, ptr addrspace(1) %b, i32 %n) {
entry:
  br label %loop

loop:
  %p1 = phi ptr addrspace(1) [ %a, %entry ], [ %p1_next, %loop ]
  %p2 = phi ptr addrspace(1) [ %b, %entry ], [ %p2_next, %loop ]
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %v = load float, ptr addrspace(1) %p1
  store float %v, ptr addrspace(1) %p2
  %p1_next = getelementptr float, ptr addrspace(1) %p1, i32 1
  %p2_next = getelementptr float, ptr addrspace(1) %p2, i32 1
  %i_next = add i32 %i, 1
  %cond = icmp slt i32 %i_next, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}
