target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

source_filename = "LLVMDialectModule"

@global_smem = internal addrspace(3) global [1024 x i8] undef, align 16

declare void @air.wg.barrier(i32, i32)
declare [3 x i32] @air.thread_position_in_threadgroup()

define void @kernel(ptr addrspace(1) %0, ptr addrspace(1) %1) {
  %3 = call [3 x i32] @air.thread_position_in_threadgroup()
  %4 = extractvalue [3 x i32] %3, 0
  %5 = and i32 %4, 31

  ; Store to base of global_smem
  %6 = getelementptr float, ptr addrspace(3) @global_smem, i32 %5
  store float 1.0, ptr addrspace(3) %6, align 4

  ; Store to global_smem + 512 bytes — byte-offset GEP into middle of TG global
  %base2 = getelementptr i8, ptr addrspace(3) @global_smem, i64 512
  %7 = getelementptr float, ptr addrspace(3) %base2, i32 %5
  store float 2.0, ptr addrspace(3) %7, align 4

  call void @air.wg.barrier(i32 1, i32 1)

  %8 = load float, ptr addrspace(3) %6, align 4
  %9 = load float, ptr addrspace(3) %7, align 4
  %10 = fadd float %8, %9

  %11 = getelementptr float, ptr addrspace(1) %1, i32 %5
  store float %10, ptr addrspace(1) %11, align 4
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
