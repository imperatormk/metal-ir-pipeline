; Test KernelProfile: scalar kernel (device writes, no per-thread index)

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

define void @test_kernel(ptr addrspace(1) %out) {
entry:
  store float 4.200000e+01, ptr addrspace(1) %out, align 4
  ret void
}
