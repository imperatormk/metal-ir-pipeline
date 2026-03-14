; Test: PtrPhiToI64 pass
; 40 ptr phis in one block — exceeds 32 limit, should convert to i64

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
target triple = "air64_v28-apple-macosx26.0.0"

define void @test_ptr_phis(ptr addrspace(1) %a, ptr addrspace(1) %b, i1 %cond) {
entry:
  br i1 %cond, label %left, label %right

left:
  %la0 = getelementptr float, ptr addrspace(1) %a, i32 0
  %la1 = getelementptr float, ptr addrspace(1) %a, i32 1
  %la2 = getelementptr float, ptr addrspace(1) %a, i32 2
  %la3 = getelementptr float, ptr addrspace(1) %a, i32 3
  %la4 = getelementptr float, ptr addrspace(1) %a, i32 4
  %la5 = getelementptr float, ptr addrspace(1) %a, i32 5
  %la6 = getelementptr float, ptr addrspace(1) %a, i32 6
  %la7 = getelementptr float, ptr addrspace(1) %a, i32 7
  %la8 = getelementptr float, ptr addrspace(1) %a, i32 8
  %la9 = getelementptr float, ptr addrspace(1) %a, i32 9
  %la10 = getelementptr float, ptr addrspace(1) %a, i32 10
  %la11 = getelementptr float, ptr addrspace(1) %a, i32 11
  %la12 = getelementptr float, ptr addrspace(1) %a, i32 12
  %la13 = getelementptr float, ptr addrspace(1) %a, i32 13
  %la14 = getelementptr float, ptr addrspace(1) %a, i32 14
  %la15 = getelementptr float, ptr addrspace(1) %a, i32 15
  %la16 = getelementptr float, ptr addrspace(1) %a, i32 16
  %la17 = getelementptr float, ptr addrspace(1) %a, i32 17
  %la18 = getelementptr float, ptr addrspace(1) %a, i32 18
  %la19 = getelementptr float, ptr addrspace(1) %a, i32 19
  %la20 = getelementptr float, ptr addrspace(1) %a, i32 20
  %la21 = getelementptr float, ptr addrspace(1) %a, i32 21
  %la22 = getelementptr float, ptr addrspace(1) %a, i32 22
  %la23 = getelementptr float, ptr addrspace(1) %a, i32 23
  %la24 = getelementptr float, ptr addrspace(1) %a, i32 24
  %la25 = getelementptr float, ptr addrspace(1) %a, i32 25
  %la26 = getelementptr float, ptr addrspace(1) %a, i32 26
  %la27 = getelementptr float, ptr addrspace(1) %a, i32 27
  %la28 = getelementptr float, ptr addrspace(1) %a, i32 28
  %la29 = getelementptr float, ptr addrspace(1) %a, i32 29
  %la30 = getelementptr float, ptr addrspace(1) %a, i32 30
  %la31 = getelementptr float, ptr addrspace(1) %a, i32 31
  %la32 = getelementptr float, ptr addrspace(1) %a, i32 32
  %la33 = getelementptr float, ptr addrspace(1) %a, i32 33
  br label %merge

right:
  %rb0 = getelementptr float, ptr addrspace(1) %b, i32 0
  %rb1 = getelementptr float, ptr addrspace(1) %b, i32 1
  %rb2 = getelementptr float, ptr addrspace(1) %b, i32 2
  %rb3 = getelementptr float, ptr addrspace(1) %b, i32 3
  %rb4 = getelementptr float, ptr addrspace(1) %b, i32 4
  %rb5 = getelementptr float, ptr addrspace(1) %b, i32 5
  %rb6 = getelementptr float, ptr addrspace(1) %b, i32 6
  %rb7 = getelementptr float, ptr addrspace(1) %b, i32 7
  %rb8 = getelementptr float, ptr addrspace(1) %b, i32 8
  %rb9 = getelementptr float, ptr addrspace(1) %b, i32 9
  %rb10 = getelementptr float, ptr addrspace(1) %b, i32 10
  %rb11 = getelementptr float, ptr addrspace(1) %b, i32 11
  %rb12 = getelementptr float, ptr addrspace(1) %b, i32 12
  %rb13 = getelementptr float, ptr addrspace(1) %b, i32 13
  %rb14 = getelementptr float, ptr addrspace(1) %b, i32 14
  %rb15 = getelementptr float, ptr addrspace(1) %b, i32 15
  %rb16 = getelementptr float, ptr addrspace(1) %b, i32 16
  %rb17 = getelementptr float, ptr addrspace(1) %b, i32 17
  %rb18 = getelementptr float, ptr addrspace(1) %b, i32 18
  %rb19 = getelementptr float, ptr addrspace(1) %b, i32 19
  %rb20 = getelementptr float, ptr addrspace(1) %b, i32 20
  %rb21 = getelementptr float, ptr addrspace(1) %b, i32 21
  %rb22 = getelementptr float, ptr addrspace(1) %b, i32 22
  %rb23 = getelementptr float, ptr addrspace(1) %b, i32 23
  %rb24 = getelementptr float, ptr addrspace(1) %b, i32 24
  %rb25 = getelementptr float, ptr addrspace(1) %b, i32 25
  %rb26 = getelementptr float, ptr addrspace(1) %b, i32 26
  %rb27 = getelementptr float, ptr addrspace(1) %b, i32 27
  %rb28 = getelementptr float, ptr addrspace(1) %b, i32 28
  %rb29 = getelementptr float, ptr addrspace(1) %b, i32 29
  %rb30 = getelementptr float, ptr addrspace(1) %b, i32 30
  %rb31 = getelementptr float, ptr addrspace(1) %b, i32 31
  %rb32 = getelementptr float, ptr addrspace(1) %b, i32 32
  %rb33 = getelementptr float, ptr addrspace(1) %b, i32 33
  br label %merge

merge:
  %p0 = phi ptr addrspace(1) [ %la0, %left ], [ %rb0, %right ]
  %p1 = phi ptr addrspace(1) [ %la1, %left ], [ %rb1, %right ]
  %p2 = phi ptr addrspace(1) [ %la2, %left ], [ %rb2, %right ]
  %p3 = phi ptr addrspace(1) [ %la3, %left ], [ %rb3, %right ]
  %p4 = phi ptr addrspace(1) [ %la4, %left ], [ %rb4, %right ]
  %p5 = phi ptr addrspace(1) [ %la5, %left ], [ %rb5, %right ]
  %p6 = phi ptr addrspace(1) [ %la6, %left ], [ %rb6, %right ]
  %p7 = phi ptr addrspace(1) [ %la7, %left ], [ %rb7, %right ]
  %p8 = phi ptr addrspace(1) [ %la8, %left ], [ %rb8, %right ]
  %p9 = phi ptr addrspace(1) [ %la9, %left ], [ %rb9, %right ]
  %p10 = phi ptr addrspace(1) [ %la10, %left ], [ %rb10, %right ]
  %p11 = phi ptr addrspace(1) [ %la11, %left ], [ %rb11, %right ]
  %p12 = phi ptr addrspace(1) [ %la12, %left ], [ %rb12, %right ]
  %p13 = phi ptr addrspace(1) [ %la13, %left ], [ %rb13, %right ]
  %p14 = phi ptr addrspace(1) [ %la14, %left ], [ %rb14, %right ]
  %p15 = phi ptr addrspace(1) [ %la15, %left ], [ %rb15, %right ]
  %p16 = phi ptr addrspace(1) [ %la16, %left ], [ %rb16, %right ]
  %p17 = phi ptr addrspace(1) [ %la17, %left ], [ %rb17, %right ]
  %p18 = phi ptr addrspace(1) [ %la18, %left ], [ %rb18, %right ]
  %p19 = phi ptr addrspace(1) [ %la19, %left ], [ %rb19, %right ]
  %p20 = phi ptr addrspace(1) [ %la20, %left ], [ %rb20, %right ]
  %p21 = phi ptr addrspace(1) [ %la21, %left ], [ %rb21, %right ]
  %p22 = phi ptr addrspace(1) [ %la22, %left ], [ %rb22, %right ]
  %p23 = phi ptr addrspace(1) [ %la23, %left ], [ %rb23, %right ]
  %p24 = phi ptr addrspace(1) [ %la24, %left ], [ %rb24, %right ]
  %p25 = phi ptr addrspace(1) [ %la25, %left ], [ %rb25, %right ]
  %p26 = phi ptr addrspace(1) [ %la26, %left ], [ %rb26, %right ]
  %p27 = phi ptr addrspace(1) [ %la27, %left ], [ %rb27, %right ]
  %p28 = phi ptr addrspace(1) [ %la28, %left ], [ %rb28, %right ]
  %p29 = phi ptr addrspace(1) [ %la29, %left ], [ %rb29, %right ]
  %p30 = phi ptr addrspace(1) [ %la30, %left ], [ %rb30, %right ]
  %p31 = phi ptr addrspace(1) [ %la31, %left ], [ %rb31, %right ]
  %p32 = phi ptr addrspace(1) [ %la32, %left ], [ %rb32, %right ]
  %p33 = phi ptr addrspace(1) [ %la33, %left ], [ %rb33, %right ]
  %v = load float, ptr addrspace(1) %p0, align 4
  store float %v, ptr addrspace(1) %p33, align 4
  ret void
}
