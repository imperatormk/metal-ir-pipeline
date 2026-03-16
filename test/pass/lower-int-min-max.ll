; Minimal test for LowerIntMinMax: llvm.smin/smax → icmp + select

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

declare i32 @llvm.smin.i32(i32, i32)
declare i32 @llvm.smax.i32(i32, i32)

define void @test_kernel(ptr addrspace(1) %A, ptr addrspace(1) %B, ptr addrspace(1) %C, i32 %tid) {
entry:
  %pa = getelementptr i32, ptr addrspace(1) %A, i32 %tid
  %a = load i32, ptr addrspace(1) %pa
  %pb = getelementptr i32, ptr addrspace(1) %B, i32 %tid
  %b = load i32, ptr addrspace(1) %pb
  %mn = call i32 @llvm.smin.i32(i32 %a, i32 %b)
  %mx = call i32 @llvm.smax.i32(i32 %a, i32 %b)
  %sum = add i32 %mn, %mx
  %pc = getelementptr i32, ptr addrspace(1) %C, i32 %tid
  store i32 %sum, ptr addrspace(1) %pc
  ret void
}
