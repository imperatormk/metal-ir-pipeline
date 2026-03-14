source_filename = "test_simple"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @test_kernel(ptr addrspace(1) %buf) #0 {
entry:
  ret void
}

attributes #0 = { convergent mustprogress nounwind willreturn "frame-pointer"="none" }

!llvm.module.flags = !{!0}
!air.kernel = !{!1}
!air.version = !{!5}

!0 = !{i32 7, !"frame-pointer", i32 0}
!1 = !{ptr @test_kernel, !2, !3}
!2 = !{}
!3 = !{!4}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"buf"}
!5 = !{i32 2, i32 8, i32 0}
