; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@__tg_dot_c_0 = internal addrspace(3) global [129 x float] undef, align 4
@__tg_dot_a_0 = internal addrspace(3) global [129 x float] undef, align 4
@__tg_cvt_1 = internal addrspace(3) global [257 x float] undef, align 4
@__tg_cvt_0 = internal addrspace(3) global [257 x float] undef, align 4
@global_smem = internal addrspace(3) global [1024 x i8] undef, align 16

declare void @air.wg.barrier(i32, i32)

declare void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float>, ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)

declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)

declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)

declare void @air.simdgroup.barrier(i32, i32)

declare void @air.threadgroup.barrier(i32, i32)

declare i32 @air.thread_index_in_simdgroup()

declare [3 x i32] @air.thread_position_in_threadgroup()

declare [3 x i32] @air.threadgroup_position_in_grid()

define void @dot_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2) {
  %4 = call [3 x i32] @air.threadgroup_position_in_grid()
  %5 = extractvalue [3 x i32] %4, 0
  %6 = mul i32 %5, 16
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
  %20 = and i32 %19, 120
  %21 = lshr i32 %20, 3
  %22 = or disjoint i32 %21, 0
  %23 = xor i32 0, %22
  %24 = xor i32 %23, 0
  %25 = add i32 %24, 0
  %26 = add i32 %6, %25
  %27 = mul i32 %26, 16
  %28 = getelementptr float, ptr addrspace(1) %0, i32 %27
  %29 = call [3 x i32] @air.thread_position_in_threadgroup()
  %30 = extractvalue [3 x i32] %29, 0
  %31 = zext i32 %30 to i64
  %32 = trunc i64 %31 to i32
  %33 = and i32 %32, 127
  %34 = urem i32 %33, 32
  %35 = call [3 x i32] @air.thread_position_in_threadgroup()
  %36 = extractvalue [3 x i32] %35, 0
  %37 = udiv i32 %36, 32
  %38 = shl i32 %34, 0
  %39 = or i32 0, %38
  %40 = shl i32 %37, 5
  %41 = or i32 %39, %40
  %42 = and i32 %41, 7
  %43 = shl i32 %42, 1
  %44 = or disjoint i32 %43, 0
  %45 = xor i32 0, %44
  %46 = xor i32 %45, 0
  %47 = xor i32 %45, 1
  %48 = add i32 %46, 0
  %49 = add i32 %47, 0
  %50 = getelementptr float, ptr addrspace(1) %28, i32 %48
  %51 = getelementptr float, ptr addrspace(1) %28, i32 %49
  %52 = load float, ptr addrspace(1) %50, align 4
  %53 = load float, ptr addrspace(1) %51, align 4
  %54 = mul i32 %25, 16
  %55 = getelementptr float, ptr addrspace(1) %1, i32 %54
  %56 = getelementptr float, ptr addrspace(1) %55, i32 %48
  %57 = getelementptr float, ptr addrspace(1) %55, i32 %49
  %58 = load float, ptr addrspace(1) %56, align 4
  %59 = load float, ptr addrspace(1) %57, align 4
  %60 = call i32 @air.thread_index_in_simdgroup()
  %61 = call [3 x i32] @air.thread_position_in_threadgroup()
  %62 = extractvalue [3 x i32] %61, 0
  %63 = udiv i32 %62, 32
  %64 = udiv i32 %60, 8
  %65 = urem i32 %60, 8
  %66 = mul i32 %63, 4
  %67 = add i32 %66, %64
  %68 = mul i32 %65, 2
  %69 = add i32 0, %68
  %70 = icmp uge i32 %67, 0
  %71 = icmp ult i32 %67, 16
  %72 = and i1 %70, %71
  %73 = mul i32 %67, 16
  %74 = add i32 %73, %69
  %75 = zext i32 %74 to i64
  %76 = select i1 %72, i64 %75, i64 256
  %77 = getelementptr float, ptr addrspace(3) @__tg_cvt_0, i64 %76
  store float %52, ptr addrspace(3) %77, align 4
  %78 = add i32 %69, 1
  %79 = add i32 %73, %78
  %80 = zext i32 %79 to i64
  %81 = select i1 %72, i64 %80, i64 256
  %82 = getelementptr float, ptr addrspace(3) @__tg_cvt_0, i64 %81
  store float %53, ptr addrspace(3) %82, align 4
  call void @air.threadgroup.barrier(i32 1, i32 4)
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %83 = call i32 @air.thread_index_in_simdgroup()
  %84 = call [3 x i32] @air.thread_position_in_threadgroup()
  %85 = extractvalue [3 x i32] %84, 0
  %86 = udiv i32 %85, 32
  %87 = udiv i32 %83, 8
  %88 = urem i32 %83, 8
  %89 = mul i32 %86, 4
  %90 = add i32 %89, %87
  %91 = mul i32 %88, 2
  %92 = add i32 0, %91
  %93 = icmp uge i32 %90, 0
  %94 = icmp ult i32 %90, 16
  %95 = and i1 %93, %94
  %96 = mul i32 %90, 16
  %97 = add i32 %96, %92
  %98 = zext i32 %97 to i64
  %99 = select i1 %95, i64 %98, i64 256
  %100 = getelementptr float, ptr addrspace(3) @__tg_cvt_1, i64 %99
  store float %58, ptr addrspace(3) %100, align 4
  %101 = add i32 %92, 1
  %102 = add i32 %96, %101
  %103 = zext i32 %102 to i64
  %104 = select i1 %95, i64 %103, i64 256
  %105 = getelementptr float, ptr addrspace(3) @__tg_cvt_1, i64 %104
  store float %59, ptr addrspace(3) %105, align 4
  call void @air.threadgroup.barrier(i32 1, i32 4)
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %106 = call i32 @air.thread_index_in_simdgroup()
  %107 = call [3 x i32] @air.thread_position_in_threadgroup()
  %108 = extractvalue [3 x i32] %107, 0
  %109 = udiv i32 %108, 32
  %110 = udiv i32 %106, 8
  %111 = urem i32 %106, 8
  %112 = mul i32 %109, 4
  %113 = add i32 %112, %110
  %114 = mul i32 %111, 2
  %115 = add i32 0, %114
  %116 = and i32 %106, 7
  %117 = lshr i32 %106, 3
  %118 = udiv i32 %109, 2
  %119 = urem i32 %109, 2
  %120 = mul i32 %118, 8
  %121 = add i32 %120, %117
  %122 = mul i32 %119, 8
  %123 = add i32 %122, %116
  %124 = icmp uge i32 %121, 0
  %125 = icmp ult i32 %121, 8
  %126 = and i1 %124, %125
  %127 = mul i32 %121, 16
  %128 = add i32 %127, %123
  %129 = zext i32 %128 to i64
  %130 = select i1 %126, i64 %129, i64 128
  %131 = getelementptr float, ptr addrspace(3) @__tg_dot_c_0, i64 %130
  store float 0.000000e+00, ptr addrspace(3) %131, align 4
  %132 = add i32 %121, 4
  %133 = icmp uge i32 %132, 0
  %134 = icmp ult i32 %132, 8
  %135 = and i1 %133, %134
  %136 = mul i32 %132, 16
  %137 = add i32 %136, %123
  %138 = zext i32 %137 to i64
  %139 = select i1 %135, i64 %138, i64 128
  %140 = getelementptr float, ptr addrspace(3) @__tg_dot_c_0, i64 %139
  store float 0.000000e+00, ptr addrspace(3) %140, align 4
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %141 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_c_0, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer)
  %142 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_c_0, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 8, i64 0>)
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %143 = icmp uge i32 %121, 8
  %144 = icmp ult i32 %121, 16
  %145 = and i1 %143, %144
  %146 = add i32 %121, -8
  %147 = mul i32 %146, 16
  %148 = add i32 %147, %123
  %149 = zext i32 %148 to i64
  %150 = select i1 %145, i64 %149, i64 128
  %151 = getelementptr float, ptr addrspace(3) @__tg_dot_c_0, i64 %150
  store float 0.000000e+00, ptr addrspace(3) %151, align 4
  %152 = icmp uge i32 %132, 8
  %153 = icmp ult i32 %132, 16
  %154 = and i1 %152, %153
  %155 = add i32 %121, -4
  %156 = mul i32 %155, 16
  %157 = add i32 %156, %123
  %158 = zext i32 %157 to i64
  %159 = select i1 %154, i64 %158, i64 128
  %160 = getelementptr float, ptr addrspace(3) @__tg_dot_c_0, i64 %159
  store float 0.000000e+00, ptr addrspace(3) %160, align 4
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %161 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_c_0, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer)
  %162 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_c_0, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 8, i64 0>)
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %163 = icmp uge i32 %113, 0
  %164 = icmp ult i32 %113, 8
  %165 = and i1 %163, %164
  %166 = mul i32 %113, 16
  %167 = add i32 %166, %115
  %168 = zext i32 %167 to i64
  %169 = select i1 %165, i64 %168, i64 128
  %170 = getelementptr float, ptr addrspace(3) @__tg_dot_a_0, i64 %169
  store float %52, ptr addrspace(3) %170, align 4
  %171 = add i32 %115, 1
  %172 = add i32 %166, %171
  %173 = zext i32 %172 to i64
  %174 = select i1 %165, i64 %173, i64 128
  %175 = getelementptr float, ptr addrspace(3) @__tg_dot_a_0, i64 %174
  store float %53, ptr addrspace(3) %175, align 4
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %176 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_a_0, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer)
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %177 = icmp uge i32 %113, 8
  %178 = icmp ult i32 %113, 16
  %179 = and i1 %177, %178
  %180 = add i32 %113, -8
  %181 = mul i32 %180, 16
  %182 = add i32 %181, %115
  %183 = zext i32 %182 to i64
  %184 = select i1 %179, i64 %183, i64 128
  %185 = getelementptr float, ptr addrspace(3) @__tg_dot_a_0, i64 %184
  store float %52, ptr addrspace(3) %185, align 4
  %186 = add i32 %181, %171
  %187 = zext i32 %186 to i64
  %188 = select i1 %179, i64 %187, i64 128
  %189 = getelementptr float, ptr addrspace(3) @__tg_dot_a_0, i64 %188
  store float %53, ptr addrspace(3) %189, align 4
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %190 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_a_0, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer)
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %191 = getelementptr float, ptr addrspace(3) @__tg_dot_c_0, i64 %169
  store float %58, ptr addrspace(3) %191, align 4
  %192 = getelementptr float, ptr addrspace(3) @__tg_dot_c_0, i64 %174
  store float %59, ptr addrspace(3) %192, align 4
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %193 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_c_0, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer)
  %194 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %176, <64 x float> %193, <64 x float> %141)
  %195 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %190, <64 x float> %193, <64 x float> %161)
  %196 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_c_0, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 8, i64 0>)
  %197 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %176, <64 x float> %196, <64 x float> %142)
  %198 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %190, <64 x float> %196, <64 x float> %162)
  call void @air.threadgroup.barrier(i32 1, i32 4)
  store float %52, ptr addrspace(3) %170, align 4
  store float %53, ptr addrspace(3) %175, align 4
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %199 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_a_0, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 8, i64 0>)
  call void @air.threadgroup.barrier(i32 1, i32 4)
  store float %52, ptr addrspace(3) %185, align 4
  store float %53, ptr addrspace(3) %189, align 4
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %200 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_a_0, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 8, i64 0>)
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %201 = getelementptr float, ptr addrspace(3) @__tg_dot_c_0, i64 %184
  store float %58, ptr addrspace(3) %201, align 4
  %202 = getelementptr float, ptr addrspace(3) @__tg_dot_c_0, i64 %188
  store float %59, ptr addrspace(3) %202, align 4
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %203 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_c_0, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer)
  %204 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %199, <64 x float> %203, <64 x float> %194)
  %205 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %200, <64 x float> %203, <64 x float> %195)
  %206 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_c_0, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 8, i64 0>)
  %207 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %199, <64 x float> %206, <64 x float> %197)
  %208 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %200, <64 x float> %206, <64 x float> %198)
  call void @air.threadgroup.barrier(i32 1, i32 4)
  call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float> %204, ptr addrspace(3) @__tg_dot_c_0, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer)
  call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float> %207, ptr addrspace(3) @__tg_dot_c_0, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 8, i64 0>)
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %209 = load float, ptr addrspace(3) %131, align 4
  %210 = select i1 %126, float %209, float 0.000000e+00
  %211 = load float, ptr addrspace(3) %140, align 4
  %212 = select i1 %135, float %211, float 0.000000e+00
  call void @air.threadgroup.barrier(i32 1, i32 4)
  call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float> %205, ptr addrspace(3) @__tg_dot_c_0, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer)
  call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float> %208, ptr addrspace(3) @__tg_dot_c_0, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 8, i64 0>)
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %213 = load float, ptr addrspace(3) %151, align 4
  %214 = select i1 %145, float %213, float %210
  %215 = load float, ptr addrspace(3) %160, align 4
  %216 = select i1 %154, float %215, float %212
  call void @air.threadgroup.barrier(i32 1, i32 4)
  %217 = getelementptr float, ptr addrspace(1) %2, i32 %27
  %218 = getelementptr float, ptr addrspace(1) %217, i32 %48
  %219 = getelementptr float, ptr addrspace(1) %217, i32 %49
  %220 = call [3 x i32] @air.thread_position_in_threadgroup()
  %221 = extractvalue [3 x i32] %220, 0
  %222 = zext i32 %221 to i64
  %223 = trunc i64 %222 to i32
  %224 = and i32 %223, 127
  %225 = urem i32 %224, 32
  %226 = call [3 x i32] @air.thread_position_in_threadgroup()
  %227 = extractvalue [3 x i32] %226, 0
  %228 = udiv i32 %227, 32
  %229 = shl i32 %225, 0
  %230 = or i32 0, %229
  %231 = shl i32 %228, 5
  %232 = or i32 %230, %231
  %233 = and i32 %232, 6
  %234 = shl i32 %233, 3
  %235 = and i32 %232, 24
  %236 = lshr i32 %235, 1
  %237 = and i32 %232, 1
  %238 = icmp eq i32 %237, 0
  %239 = select i1 %238, i32 0, i32 576
  %240 = and i32 %232, 32
  %241 = icmp eq i32 %240, 0
  %242 = select i1 %241, i32 0, i32 64
  %243 = and i32 %232, 64
  %244 = icmp eq i32 %243, 0
  %245 = select i1 %244, i32 0, i32 256
  %246 = xor i32 %234, %236
  %247 = xor i32 %239, %242
  %248 = xor i32 %246, %247
  %249 = or disjoint i32 %245, %248
  %250 = xor i32 0, %249
  %251 = xor i32 %250, 0
  %252 = xor i32 %251, 0
  %253 = add i32 %252, 0
  %254 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %253
  %255 = insertelement <1 x float> undef, float %214, i32 0
  br i1 true, label %256, label %257

256:                                              ; preds = %3
  store <1 x float> %255, ptr addrspace(3) %254, align 4
  br label %257

257:                                              ; preds = %256, %3
  %258 = add i32 %252, 128
  %259 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %258
  %260 = insertelement <1 x float> undef, float %216, i32 0
  br i1 true, label %261, label %262

261:                                              ; preds = %257
  store <1 x float> %260, ptr addrspace(3) %259, align 4
  br label %262

262:                                              ; preds = %261, %257
  call void @air.wg.barrier(i32 2, i32 1)
  %263 = call [3 x i32] @air.thread_position_in_threadgroup()
  %264 = extractvalue [3 x i32] %263, 0
  %265 = zext i32 %264 to i64
  %266 = trunc i64 %265 to i32
  %267 = and i32 %266, 127
  %268 = urem i32 %267, 32
  %269 = call [3 x i32] @air.thread_position_in_threadgroup()
  %270 = extractvalue [3 x i32] %269, 0
  %271 = udiv i32 %270, 32
  %272 = shl i32 %268, 0
  %273 = or i32 0, %272
  %274 = shl i32 %271, 5
  %275 = or i32 %273, %274
  %276 = and i32 %275, 7
  %277 = shl i32 %276, 4
  %278 = and i32 %275, 96
  %279 = shl i32 %278, 2
  %280 = and i32 %275, 24
  %281 = lshr i32 %280, 1
  %282 = xor i32 %277, %281
  %283 = or disjoint i32 %279, %282
  %284 = xor i32 0, %283
  %285 = xor i32 %284, 0
  %286 = xor i32 %285, 0
  %287 = add i32 %286, 0
  %288 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %287
  %289 = load <1 x float>, ptr addrspace(3) %288, align 4
  %290 = extractelement <1 x float> %289, i32 0
  %291 = xor i32 %285, 576
  %292 = add i32 %291, 0
  %293 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %292
  %294 = load <1 x float>, ptr addrspace(3) %293, align 4
  %295 = extractelement <1 x float> %294, i32 0
  store float %290, ptr addrspace(1) %218, align 4
  store float %295, ptr addrspace(1) %219, align 4
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
