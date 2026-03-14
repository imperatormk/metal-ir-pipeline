target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@global_smem = internal addrspace(3) global [16 x i8] undef, align 16

declare void @air.wg.barrier(i32, i32)
declare [3 x i32] @air.thread_position_in_threadgroup()

define void @kernel(ptr addrspace(1) %0) {
  %tid3 = call [3 x i32] @air.thread_position_in_threadgroup()
  %tid = extractvalue [3 x i32] %tid3, 0
  %is0 = icmp eq i32 %tid, 0

  ; Thread 0: store 42 at TG offset 0
  br i1 %is0, label %store_vals, label %after_store

store_vals:
  %slot0 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 0
  %v42 = insertelement <1 x i32> undef, i32 42, i32 0
  store <1 x i32> %v42, ptr addrspace(3) %slot0, align 4

  ; Store 99 at TG offset 8 using constant GEP expression
  %slot1 = getelementptr inbounds i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @global_smem, i64 8), i32 0
  %v99 = insertelement <1 x i32> undef, i32 99, i32 0
  store <1 x i32> %v99, ptr addrspace(3) %slot1, align 4
  br label %after_store

after_store:
  call void @air.wg.barrier(i32 1, i32 1)

  ; Thread 0: load both back and write to output
  br i1 %is0, label %load_vals, label %done

load_vals:
  %rd0 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 0
  %ld0 = load <1 x i32>, ptr addrspace(3) %rd0, align 4
  %val0 = extractelement <1 x i32> %ld0, i32 0

  %rd1 = getelementptr inbounds i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @global_smem, i64 8), i32 0
  %ld1 = load <1 x i32>, ptr addrspace(3) %rd1, align 4
  %val1 = extractelement <1 x i32> %ld1, i32 0

  %out0 = getelementptr i32, ptr addrspace(1) %0, i32 0
  store i32 %val0, ptr addrspace(1) %out0, align 4
  %out1 = getelementptr i32, ptr addrspace(1) %0, i32 1
  store i32 %val1, ptr addrspace(1) %out1, align 4
  br label %done

done:
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
