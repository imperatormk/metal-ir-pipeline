; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@global_smem = internal addrspace(3) global [8 x i8] undef, align 16

declare [3 x i32] @air.thread_position_in_threadgroup()

declare [3 x i32] @air.threadgroup_position_in_grid()

define void @add_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2) {
  %4 = call [3 x i32] @air.threadgroup_position_in_grid()
  %5 = extractvalue [3 x i32] %4, 0
  %6 = mul i32 %5, 1024
  %7 = call [3 x i32] @air.thread_position_in_threadgroup()
  %8 = extractvalue [3 x i32] %7, 0
  %9 = zext i32 %8 to i64
  %10 = trunc i64 %9 to i32
  %11 = and i32 %10, 127
  %12 = urem i32 %11, 32
  %13 = call [3 x i32] @air.thread_position_in_threadgroup()
  %14 = extractvalue [3 x i32] %13, 0
  %15 = udiv i32 %14, 32
  %16 = shl i32 %12, 0
  %17 = or i32 0, %16
  %18 = shl i32 %15, 5
  %19 = or i32 %17, %18
  %20 = and i32 %19, 127
  %21 = shl i32 %20, 2
  %22 = or disjoint i32 %21, 0
  %23 = xor i32 0, %22
  %24 = xor i32 %23, 0
  %25 = xor i32 %23, 1
  %26 = xor i32 %23, 2
  %27 = xor i32 %23, 3
  %28 = xor i32 %23, 512
  %29 = xor i32 %23, 513
  %30 = xor i32 %23, 514
  %31 = xor i32 %23, 515
  %32 = add i32 %24, 0
  %33 = add i32 %25, 0
  %34 = add i32 %26, 0
  %35 = add i32 %27, 0
  %36 = add i32 %28, 0
  %37 = add i32 %29, 0
  %38 = add i32 %30, 0
  %39 = add i32 %31, 0
  %40 = add i32 %6, %32
  %41 = add i32 %6, %33
  %42 = add i32 %6, %34
  %43 = add i32 %6, %35
  %44 = add i32 %6, %36
  %45 = add i32 %6, %37
  %46 = add i32 %6, %38
  %47 = add i32 %6, %39
  %48 = getelementptr float, ptr addrspace(1) %0, i32 %40
  %49 = getelementptr float, ptr addrspace(1) %0, i32 %41
  %50 = getelementptr float, ptr addrspace(1) %0, i32 %42
  %51 = getelementptr float, ptr addrspace(1) %0, i32 %43
  %52 = getelementptr float, ptr addrspace(1) %0, i32 %44
  %53 = getelementptr float, ptr addrspace(1) %0, i32 %45
  %54 = getelementptr float, ptr addrspace(1) %0, i32 %46
  %55 = getelementptr float, ptr addrspace(1) %0, i32 %47
  %56 = load float, ptr addrspace(1) %48, align 4
  %57 = load float, ptr addrspace(1) %49, align 4
  %58 = load float, ptr addrspace(1) %50, align 4
  %59 = load float, ptr addrspace(1) %51, align 4
  %60 = load float, ptr addrspace(1) %52, align 4
  %61 = load float, ptr addrspace(1) %53, align 4
  %62 = load float, ptr addrspace(1) %54, align 4
  %63 = load float, ptr addrspace(1) %55, align 4
  %64 = getelementptr float, ptr addrspace(1) %1, i32 %40
  %65 = getelementptr float, ptr addrspace(1) %1, i32 %41
  %66 = getelementptr float, ptr addrspace(1) %1, i32 %42
  %67 = getelementptr float, ptr addrspace(1) %1, i32 %43
  %68 = getelementptr float, ptr addrspace(1) %1, i32 %44
  %69 = getelementptr float, ptr addrspace(1) %1, i32 %45
  %70 = getelementptr float, ptr addrspace(1) %1, i32 %46
  %71 = getelementptr float, ptr addrspace(1) %1, i32 %47
  %72 = load float, ptr addrspace(1) %64, align 4
  %73 = load float, ptr addrspace(1) %65, align 4
  %74 = load float, ptr addrspace(1) %66, align 4
  %75 = load float, ptr addrspace(1) %67, align 4
  %76 = load float, ptr addrspace(1) %68, align 4
  %77 = load float, ptr addrspace(1) %69, align 4
  %78 = load float, ptr addrspace(1) %70, align 4
  %79 = load float, ptr addrspace(1) %71, align 4
  %80 = getelementptr float, ptr addrspace(1) %2, i32 %40
  %81 = getelementptr float, ptr addrspace(1) %2, i32 %41
  %82 = getelementptr float, ptr addrspace(1) %2, i32 %42
  %83 = getelementptr float, ptr addrspace(1) %2, i32 %43
  %84 = getelementptr float, ptr addrspace(1) %2, i32 %44
  %85 = getelementptr float, ptr addrspace(1) %2, i32 %45
  %86 = getelementptr float, ptr addrspace(1) %2, i32 %46
  %87 = getelementptr float, ptr addrspace(1) %2, i32 %47
  %88 = fadd float %56, %72
  %89 = fadd float %57, %73
  %90 = fadd float %58, %74
  %91 = fadd float %59, %75
  %92 = fadd float %60, %76
  %93 = fadd float %61, %77
  %94 = fadd float %62, %78
  %95 = fadd float %63, %79
  store float %88, ptr addrspace(1) %80, align 4
  store float %89, ptr addrspace(1) %81, align 4
  store float %90, ptr addrspace(1) %82, align 4
  store float %91, ptr addrspace(1) %83, align 4
  store float %92, ptr addrspace(1) %84, align 4
  store float %93, ptr addrspace(1) %85, align 4
  store float %94, ptr addrspace(1) %86, align 4
  store float %95, ptr addrspace(1) %87, align 4
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
