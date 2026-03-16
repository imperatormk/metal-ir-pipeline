; Test for TGBarrierInsert: inserts barrier before TG store.
; Pre-condition: barriers already renamed (BarrierRename runs first).

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

@shared = internal addrspace(3) global [128 x float] undef

define void @test_kernel(ptr addrspace(1) %buf, i32 %tid) {
entry:
  %p_dev = getelementptr float, ptr addrspace(1) %buf, i32 %tid
  %v = load float, ptr addrspace(1) %p_dev, align 4
  %p_tg = getelementptr [128 x float], ptr addrspace(3) @shared, i32 0, i32 %tid
  store float %v, ptr addrspace(3) %p_tg, align 4
  %v2 = load float, ptr addrspace(3) %p_tg, align 4
  store float %v2, ptr addrspace(1) %p_dev, align 4
  ret void
}
