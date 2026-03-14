; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@global_smem = internal addrspace(3) global [128 x i8] undef, align 16

declare void @air.wg.barrier(i32, i32)

declare float @air.simd_shuffle_xor.f32(float, i16)

declare [3 x i32] @air.thread_position_in_threadgroup()

define void @kernel(ptr addrspace(1) %0, ptr addrspace(1) %1) {
  %3 = call [3 x i32] @air.thread_position_in_threadgroup()
  %4 = extractvalue [3 x i32] %3, 0
  %5 = zext i32 %4 to i64
  %6 = trunc i64 %5 to i32
  %7 = and i32 %6, 127
  %8 = urem i32 %7, 32
  %9 = call [3 x i32] @air.thread_position_in_threadgroup()
  %10 = extractvalue [3 x i32] %9, 0
  %11 = udiv i32 %10, 32
  %12 = shl i32 %8, 0
  %13 = or i32 0, %12
  %14 = shl i32 %11, 5
  %15 = or i32 %13, %14
  %16 = and i32 %15, 31
  %17 = lshr i32 %16, 0
  %18 = or disjoint i32 %17, 0
  %19 = xor i32 0, %18
  %20 = xor i32 %19, 0
  %21 = add i32 %20, 0
  %22 = call [3 x i32] @air.thread_position_in_threadgroup()
  %23 = extractvalue [3 x i32] %22, 0
  %24 = zext i32 %23 to i64
  %25 = trunc i64 %24 to i32
  %26 = and i32 %25, 127
  %27 = urem i32 %26, 32
  %28 = call [3 x i32] @air.thread_position_in_threadgroup()
  %29 = extractvalue [3 x i32] %28, 0
  %30 = udiv i32 %29, 32
  %31 = shl i32 %27, 0
  %32 = or i32 0, %31
  %33 = shl i32 %30, 5
  %34 = or i32 %32, %33
  %35 = and i32 %34, 120
  %36 = lshr i32 %35, 3
  %37 = or disjoint i32 %36, 0
  %38 = xor i32 0, %37
  %39 = xor i32 %38, 0
  %40 = xor i32 %38, 16
  %41 = add i32 %39, 0
  %42 = add i32 %40, 0
  %43 = mul i32 %41, 32
  %44 = mul i32 %42, 32
  %45 = getelementptr float, ptr addrspace(1) %0, i32 %43
  %46 = getelementptr float, ptr addrspace(1) %0, i32 %44
  %47 = call [3 x i32] @air.thread_position_in_threadgroup()
  %48 = extractvalue [3 x i32] %47, 0
  %49 = zext i32 %48 to i64
  %50 = trunc i64 %49 to i32
  %51 = and i32 %50, 127
  %52 = urem i32 %51, 32
  %53 = call [3 x i32] @air.thread_position_in_threadgroup()
  %54 = extractvalue [3 x i32] %53, 0
  %55 = udiv i32 %54, 32
  %56 = shl i32 %52, 0
  %57 = or i32 0, %56
  %58 = shl i32 %55, 5
  %59 = or i32 %57, %58
  %60 = and i32 %59, 7
  %61 = shl i32 %60, 2
  %62 = or disjoint i32 %61, 0
  %63 = xor i32 0, %62
  %64 = xor i32 %63, 0
  %65 = xor i32 %63, 1
  %66 = xor i32 %63, 2
  %67 = xor i32 %63, 3
  %68 = add i32 %64, 0
  %69 = add i32 %65, 0
  %70 = add i32 %66, 0
  %71 = add i32 %67, 0
  %72 = getelementptr float, ptr addrspace(1) %45, i32 %68
  %73 = getelementptr float, ptr addrspace(1) %45, i32 %69
  %74 = getelementptr float, ptr addrspace(1) %45, i32 %70
  %75 = getelementptr float, ptr addrspace(1) %45, i32 %71
  %76 = getelementptr float, ptr addrspace(1) %46, i32 %68
  %77 = getelementptr float, ptr addrspace(1) %46, i32 %69
  %78 = getelementptr float, ptr addrspace(1) %46, i32 %70
  %79 = getelementptr float, ptr addrspace(1) %46, i32 %71
  %80 = load float, ptr addrspace(1) %72, align 4
  %81 = load float, ptr addrspace(1) %73, align 4
  %82 = load float, ptr addrspace(1) %74, align 4
  %83 = load float, ptr addrspace(1) %75, align 4
  %84 = load float, ptr addrspace(1) %76, align 4
  %85 = load float, ptr addrspace(1) %77, align 4
  %86 = load float, ptr addrspace(1) %78, align 4
  %87 = load float, ptr addrspace(1) %79, align 4
  %88 = fadd float %80, %81
  %89 = fadd float %82, %83
  %90 = fadd float %88, %89
  %91 = fadd float %84, %85
  %92 = fadd float %86, %87
  %93 = fadd float %91, %92
  %94 = call float @air.simd_shuffle_xor.f32(float %90, i16 4)
  %95 = fadd float %90, %94
  %96 = call float @air.simd_shuffle_xor.f32(float %95, i16 2)
  %97 = fadd float %95, %96
  %98 = call float @air.simd_shuffle_xor.f32(float %97, i16 1)
  %99 = fadd float %97, %98
  %100 = call float @air.simd_shuffle_xor.f32(float %93, i16 4)
  %101 = fadd float %93, %100
  %102 = call float @air.simd_shuffle_xor.f32(float %101, i16 2)
  %103 = fadd float %101, %102
  %104 = call float @air.simd_shuffle_xor.f32(float %103, i16 1)
  %105 = fadd float %103, %104
  %106 = getelementptr float, ptr addrspace(1) %1, i32 %21
  %107 = call [3 x i32] @air.thread_position_in_threadgroup()
  %108 = extractvalue [3 x i32] %107, 0
  %109 = zext i32 %108 to i64
  %110 = trunc i64 %109 to i32
  %111 = and i32 %110, 127
  %112 = urem i32 %111, 32
  %113 = call [3 x i32] @air.thread_position_in_threadgroup()
  %114 = extractvalue [3 x i32] %113, 0
  %115 = udiv i32 %114, 32
  %116 = shl i32 %112, 0
  %117 = or i32 0, %116
  %118 = shl i32 %115, 5
  %119 = or i32 %117, %118
  %120 = and i32 %119, 120
  %121 = lshr i32 %120, 0
  %122 = or disjoint i32 0, %121
  %123 = xor i32 0, %122
  %124 = xor i32 %123, 0
  %125 = xor i32 %124, 0
  %126 = add i32 %125, 0
  %127 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %126
  %128 = insertelement <2 x float> undef, float %99, i32 0
  %129 = insertelement <2 x float> %128, float %105, i32 1
  br i1 true, label %130, label %131

130:                                              ; preds = %2
  store <2 x float> %129, ptr addrspace(3) %127, align 8
  br label %131

131:                                              ; preds = %130, %2
  call void @air.wg.barrier(i32 2, i32 1)
  %132 = call [3 x i32] @air.thread_position_in_threadgroup()
  %133 = extractvalue [3 x i32] %132, 0
  %134 = zext i32 %133 to i64
  %135 = trunc i64 %134 to i32
  %136 = and i32 %135, 127
  %137 = urem i32 %136, 32
  %138 = call [3 x i32] @air.thread_position_in_threadgroup()
  %139 = extractvalue [3 x i32] %138, 0
  %140 = udiv i32 %139, 32
  %141 = shl i32 %137, 0
  %142 = or i32 0, %141
  %143 = shl i32 %140, 5
  %144 = or i32 %142, %143
  %145 = and i32 %144, 15
  %146 = shl i32 %145, 3
  %147 = and i32 %144, 16
  %148 = icmp eq i32 %147, 0
  %149 = select i1 %148, i32 0, i32 4
  %150 = or disjoint i32 %146, %149
  %151 = or disjoint i32 %150, 0
  %152 = xor i32 0, %151
  %153 = xor i32 %152, 0
  %154 = xor i32 %153, 0
  %155 = add i32 %154, 0
  %156 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %155
  %157 = load <1 x float>, ptr addrspace(3) %156, align 4
  %158 = extractelement <1 x float> %157, i32 0
  store float %158, ptr addrspace(1) %106, align 4
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
