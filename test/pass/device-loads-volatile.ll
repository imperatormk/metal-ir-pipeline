; Test for DeviceLoadsVolatile: device loads in loops with stores → volatile

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

define void @test_kernel(ptr addrspace(1) %buf, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %v = load float, ptr addrspace(1) %buf, align 4
  %inc = fadd float %v, 1.0
  store float %inc, ptr addrspace(1) %buf, align 4
  %i_next = add i32 %i, 1
  %cond = icmp slt i32 %i_next, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}
