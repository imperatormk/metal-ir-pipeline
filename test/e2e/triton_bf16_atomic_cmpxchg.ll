; ModuleID = 'triton_kernel'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.3.0"

declare i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1), ptr, i32, i32, i32, i32, i1)

declare float @air.simd_shuffle_xor.f32(float, i16)

define void @kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, <3 x i32> %tidtg) {
  %tidtg_z = extractelement <3 x i32> %tidtg, i32 2
  %tidtg_y = extractelement <3 x i32> %tidtg, i32 1
  %tidtg_x = extractelement <3 x i32> %tidtg, i32 0
  %4 = zext i32 %tidtg_x to i64
  %5 = trunc i64 %4 to i32
  %6 = and i32 %5, 127
  %7 = urem i32 %6, 32
  %8 = udiv i32 %tidtg_x, 32
  %9 = shl i32 %7, 0
  %10 = or i32 0, %9
  %11 = shl i32 %8, 5
  %12 = or i32 %10, %11
  %13 = and i32 %12, 1
  %14 = icmp eq i32 %13, 0
  %15 = select i1 %14, i32 0, i32 1
  %16 = or i32 %15, 0
  %17 = xor i32 0, %16
  %18 = xor i32 %17, 0
  %19 = add i32 %18, 0
  %20 = zext i32 %tidtg_x to i64
  %21 = trunc i64 %20 to i32
  %22 = and i32 %21, 127
  %23 = urem i32 %22, 32
  %24 = udiv i32 %tidtg_x, 32
  %25 = shl i32 %23, 0
  %26 = or i32 0, %25
  %27 = shl i32 %24, 5
  %28 = or i32 %26, %27
  %29 = and i32 %28, 14
  %30 = lshr i32 %29, 1
  %31 = or i32 %30, 0
  %32 = xor i32 0, %31
  %33 = xor i32 %32, 0
  %34 = add i32 %33, 0
  %35 = mul i32 %34, 2
  %36 = getelementptr bfloat, ptr addrspace(1) %1, i32 %35
  %37 = zext i32 %tidtg_x to i64
  %38 = trunc i64 %37 to i32
  %39 = and i32 %38, 127
  %40 = urem i32 %39, 32
  %41 = udiv i32 %tidtg_x, 32
  %42 = shl i32 %40, 0
  %43 = or i32 0, %42
  %44 = shl i32 %41, 5
  %45 = or i32 %43, %44
  %46 = and i32 %45, 1
  %47 = icmp eq i32 %46, 0
  %48 = select i1 %47, i32 0, i32 1
  %49 = or i32 %48, 0
  %50 = xor i32 0, %49
  %51 = xor i32 %50, 0
  %52 = add i32 %51, 0
  %53 = getelementptr bfloat, ptr addrspace(1) %36, i32 %52
  %54 = load bfloat, ptr addrspace(1) %53, align 2
  %55 = fpext bfloat %54 to float
  %56 = call float @air.simd_shuffle_xor.f32(float %55, i16 8)
  %57 = fadd float %55, %56
  %58 = call float @air.simd_shuffle_xor.f32(float %57, i16 4)
  %59 = fadd float %57, %58
  %60 = call float @air.simd_shuffle_xor.f32(float %59, i16 2)
  %61 = fadd float %59, %60
  %62 = fptrunc float %61 to bfloat
  %63 = getelementptr bfloat, ptr addrspace(1) %0, i32 %19
  %64 = urem i32 %tidtg_x, 32
  %65 = udiv i32 %tidtg_x, 32
  %66 = and i32 %64, 30
  %67 = icmp eq i32 %66, 0
  %68 = and i32 %65, 3
  %69 = icmp eq i32 %68, 0
  %70 = and i1 %67, %69
  %71 = and i1 %70, true
  br i1 %71, label %72, label %101

72:                                               ; preds = %3
  %73 = alloca i32, align 4
  %74 = ptrtoint ptr addrspace(1) %63 to i64
  %75 = and i64 %74, 3
  %76 = and i64 %74, -4
  %77 = inttoptr i64 %76 to ptr addrspace(1)
  %78 = trunc i64 %75 to i32
  %79 = mul i32 %78, 8
  %80 = shl i32 65535, %79
  %81 = xor i32 %80, -1
  %82 = fpext bfloat %62 to float
  %83 = load i32, ptr addrspace(1) %77, align 4
  br label %84

84:                                               ; preds = %84, %72
  %85 = phi i32 [ %97, %84 ], [ %83, %72 ]
  %86 = lshr i32 %85, %79
  %87 = trunc i32 %86 to i16
  %88 = bitcast i16 %87 to bfloat
  %89 = fpext bfloat %88 to float
  %90 = fadd float %89, %82
  %91 = fptrunc float %90 to bfloat
  %92 = bitcast bfloat %91 to i16
  %93 = zext i16 %92 to i32
  %94 = shl i32 %93, %79
  %95 = and i32 %85, %81
  %96 = or i32 %95, %94
  store i32 %85, ptr %73, align 4
  %97 = call i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1) %77, ptr %73, i32 %96, i32 0, i32 0, i32 2, i1 true)
  %98 = icmp eq i32 %97, %85
  br i1 %98, label %99, label %84

99:                                               ; preds = %84
  %100 = phi bfloat [ %88, %84 ]
  br label %101

101:                                              ; preds = %99, %3
  %102 = phi bfloat [ %100, %99 ], [ 0xR0000, %3 ]
  %103 = getelementptr bfloat, ptr addrspace(1) %2, i32 %19
  %104 = bitcast ptr addrspace(1) %103 to ptr addrspace(1)
  store bfloat %102, ptr addrspace(1) %104, align 2
  ret void
}

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}
!air.kernel = !{!7}
!air.version = !{!14}
!air.language_version = !{!15}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 7, !"air.max_device_buffers", i32 31}
!2 = !{i32 7, !"air.max_constant_buffers", i32 31}
!3 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!4 = !{i32 7, !"air.max_textures", i32 128}
!5 = !{i32 7, !"air.max_read_write_textures", i32 8}
!6 = !{i32 7, !"air.max_samplers", i32 16}
!7 = !{ptr @kernel, !8, !9}
!8 = !{}
!9 = !{!10, !11, !12, !13}
!10 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"0"}
!11 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"1"}
!12 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"2"}
!13 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"tidtg"}
!14 = !{i32 2, i32 8, i32 0}
!15 = !{!"Metal", i32 3, i32 2, i32 0}
