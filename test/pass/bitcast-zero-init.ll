; Minimal test for BitcastZeroInit: bitcast zeroinitializer → zero of dest type

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

define void @test_kernel(ptr addrspace(1) %out, i32 %tid) {
entry:
  %zero_i32 = bitcast float 0.000000e+00 to i32
  %p = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store i32 %zero_i32, ptr addrspace(1) %p
  ret void
}
