target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@__tg_cvt_0 = internal addrspace(3) global [2 x i32] undef, align 4
@global_smem = internal addrspace(3) global [8 x i8] undef, align 16

declare void @air.threadgroup.barrier(i32, i32)
declare [3 x i32] @air.thread_position_in_threadgroup()
declare i32 @air.thread_index_in_simdgroup()

define void @kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2) {
  %4 = load i32, ptr addrspace(1) %0, align 4
  %5 = load i32, ptr addrspace(1) %1, align 4
  %6 = call i32 @air.thread_index_in_simdgroup()
  %7 = call [3 x i32] @air.thread_position_in_threadgroup()
  %8 = urem i32 %6, 2
  %9 = add i32 0, %8
  store i32 %4, ptr addrspace(3) @__tg_cvt_0, align 4
  store i32 %5, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__tg_cvt_0, i64 4), align 4
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %10 = add i32 0, %9
  %11 = zext i32 %10 to i64
  %12 = getelementptr i32, ptr addrspace(3) @__tg_cvt_0, i64 %11
  %13 = load i32, ptr addrspace(3) %12, align 4
  %14 = call [3 x i32] @air.thread_position_in_threadgroup()
  %15 = extractvalue [3 x i32] %14, 0
  %16 = zext i32 %15 to i64
  %17 = trunc i64 %16 to i32
  %18 = and i32 %17, 127
  %19 = urem i32 %18, 32
  %20 = call [3 x i32] @air.thread_position_in_threadgroup()
  %21 = extractvalue [3 x i32] %20, 0
  %22 = udiv i32 %21, 32
  %23 = shl i32 %19, 0
  %24 = or i32 0, %23
  %25 = shl i32 %22, 5
  %26 = or i32 %24, %25
  %27 = and i32 %26, 1
  %28 = icmp eq i32 %27, 0
  %29 = select i1 %28, i32 0, i32 1
  %30 = or disjoint i32 %29, 0
  %31 = xor i32 0, %30
  %32 = xor i32 %31, 0
  %33 = add i32 %32, 0
  %34 = getelementptr float, ptr addrspace(1) %2, i32 %33
  %35 = sitofp i32 %13 to float
  store float %35, ptr addrspace(1) %34, align 4
  ret void
}
