; Test for SplitI64Shuffle: air.simd_shuffle.*.i64 → 2x i32 shuffles

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

declare i64 @air.simd_shuffle.i64(i64, i16)

define void @test_kernel(ptr addrspace(1) %out, i32 %tid) {
entry:
  %p = getelementptr i64, ptr addrspace(1) %out, i32 %tid
  %v = load i64, ptr addrspace(1) %p
  %shuf = call i64 @air.simd_shuffle.i64(i64 %v, i16 1)
  store i64 %shuf, ptr addrspace(1) %p
  ret void
}
