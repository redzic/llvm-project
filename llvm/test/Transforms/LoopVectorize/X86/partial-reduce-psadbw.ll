; RUN: opt -passes=loop-vectorize -S -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64 < %s | FileCheck %s --check-prefixes=CHECK,SSE2
; RUN: opt -passes=loop-vectorize -S -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64-v3 < %s | FileCheck %s --check-prefixes=CHECK,AVX2
; RUN: opt -passes=loop-vectorize -S -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64-v4 -mattr=-prefer-256-bit < %s | FileCheck %s --check-prefixes=CHECK,AVX512
; RUN: opt -passes=loop-vectorize -S -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64-v3 \
; RUN:   -vectorizer-maximize-bandwidth-for-partial-reductions=false < %s | FileCheck %s --check-prefixes=CHECK,NOMAXBW

; Byte sums and sums of absolute byte differences reduce a whole vector of
; bytes per PSADBW, so the VF is chosen by the bytes and the reduction becomes a
; partial reduction into an accumulator as wide as one vector of bytes.

; sum += abs(a[i] - b[i])
define i32 @sad(ptr %a, ptr %b, i64 %n) {
; CHECK-LABEL: @sad(
; SSE2:        call <4 x i32> @llvm.vector.partial.reduce.add.v4i32.v16i32(<4 x i32> {{.*}}, <16 x i32>
; AVX2:        call <8 x i32> @llvm.vector.partial.reduce.add.v8i32.v32i32(<8 x i32> {{.*}}, <32 x i32>
; AVX512:      call <16 x i32> @llvm.vector.partial.reduce.add.v16i32.v64i32(<16 x i32> {{.*}}, <64 x i32>
; NOMAXBW-NOT: partial.reduce
; NOMAXBW:     call <8 x i32> @llvm.abs.v8i32(
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  %pa = getelementptr inbounds i8, ptr %a, i64 %i
  %pb = getelementptr inbounds i8, ptr %b, i64 %i
  %va = load i8, ptr %pa, align 1
  %vb = load i8, ptr %pb, align 1
  %ea = zext i8 %va to i32
  %eb = zext i8 %vb to i32
  %d = sub nsw i32 %ea, %eb
  %abs = call i32 @llvm.abs.i32(i32 %d, i1 true)
  %sum.next = add nuw nsw i32 %abs, %sum
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret i32 %sum.next
}

; sum += a[i], into i32
define i32 @bsum(ptr %a, i64 %n) {
; CHECK-LABEL: @bsum(
; SSE2:        call <4 x i32> @llvm.vector.partial.reduce.add.v4i32.v16i32(<4 x i32> {{.*}}, <16 x i32>
; AVX2:        call <8 x i32> @llvm.vector.partial.reduce.add.v8i32.v32i32(<8 x i32> {{.*}}, <32 x i32>
; AVX512:      call <16 x i32> @llvm.vector.partial.reduce.add.v16i32.v64i32(<16 x i32> {{.*}}, <64 x i32>
; NOMAXBW-NOT: partial.reduce
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  %pa = getelementptr inbounds i8, ptr %a, i64 %i
  %va = load i8, ptr %pa, align 1
  %ea = zext i8 %va to i32
  %sum.next = add i32 %sum, %ea
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret i32 %sum.next
}

; sum += a[i], into i64
define i64 @bsum64(ptr %a, i64 %n) {
; CHECK-LABEL: @bsum64(
; SSE2:        call <2 x i64> @llvm.vector.partial.reduce.add.v2i64.v16i64(<2 x i64> {{.*}}, <16 x i64>
; AVX2:        call <4 x i64> @llvm.vector.partial.reduce.add.v4i64.v32i64(<4 x i64> {{.*}}, <32 x i64>
; AVX512:      call <8 x i64> @llvm.vector.partial.reduce.add.v8i64.v64i64(<8 x i64> {{.*}}, <64 x i64>
; NOMAXBW-NOT: partial.reduce
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi i64 [ 0, %entry ], [ %sum.next, %loop ]
  %pa = getelementptr inbounds i8, ptr %a, i64 %i
  %va = load i8, ptr %pa, align 1
  %ea = zext i8 %va to i64
  %sum.next = add i64 %sum, %ea
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret i64 %sum.next
}

; An i16 accumulator has no PSADBW form, so the VF stays sized by the i16 type.
define i16 @bsum16(ptr %a, i64 %n) {
; CHECK-LABEL: @bsum16(
; CHECK-NOT:   partial.reduce
; AVX2:        add <16 x i16>
; CHECK:       ret i16
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi i16 [ 0, %entry ], [ %sum.next, %loop ]
  %pa = getelementptr inbounds i8, ptr %a, i64 %i
  %va = load i8, ptr %pa, align 1
  %ea = zext i8 %va to i16
  %sum.next = add i16 %sum, %ea
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret i16 %sum.next
}
