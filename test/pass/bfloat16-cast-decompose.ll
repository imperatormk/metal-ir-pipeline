; Test for BFloat16CastDecompose: iN→bfloat via iN→f32→bfloat

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

define void @test_kernel(ptr addrspace(1) %out, i32 %tid) {
entry:
  %cast = sitofp i32 %tid to bfloat
  %p = getelementptr bfloat, ptr addrspace(1) %out, i32 %tid
  store bfloat %cast, ptr addrspace(1) %p
  ret void
}
