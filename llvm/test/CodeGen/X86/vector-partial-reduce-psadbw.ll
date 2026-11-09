; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64 | FileCheck %s --check-prefixes=SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=sandybridge | FileCheck %s --check-prefixes=AVX1
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64-v3 | FileCheck %s --check-prefixes=AVX2
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64-v4 -mattr=-prefer-256-bit | FileCheck %s --check-prefixes=AVX512
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2,+avxvnniint8 | FileCheck %s --check-prefixes=AVXVNNIINT8

; Byte sums and sums of absolute byte differences, as the loop vectorizer emits
; them, become one PSADBW per vector and an add into the accumulator.

; Byte sum into i32: psadbw against zero.

define <4 x i32> @bsum_v4i32(<4 x i32> %acc, <16 x i8> %a) {
; SSE2-LABEL: bsum_v4i32:
; SSE2:       pxor %xmm2, %xmm2
; SSE2-NEXT:  psadbw %xmm1, %xmm2
; SSE2-NEXT:  paddd %xmm2, %xmm0
; SSE2-NEXT:  retq
;
; AVX2-LABEL: bsum_v4i32:
; AVX2:       vpxor %xmm2, %xmm2, %xmm2
; AVX2-NEXT:  vpsadbw %xmm2, %xmm1, %xmm1
; AVX2-NEXT:  vpaddd %xmm1, %xmm0, %xmm0
; AVX2-NEXT:  retq
  %ext = zext <16 x i8> %a to <16 x i32>
  %r = call <4 x i32> @llvm.vector.partial.reduce.add.v4i32.v16i32(<4 x i32> %acc, <16 x i32> %ext)
  ret <4 x i32> %r
}

define <8 x i32> @bsum_v8i32(<8 x i32> %acc, <32 x i8> %a) {
; SSE2-LABEL: bsum_v8i32:
; SSE2-COUNT-2: psadbw
; SSE2-NOT:   punpck
; SSE2:       retq
;
; AVX1-LABEL: bsum_v8i32:
; AVX1-COUNT-2: vpsadbw {{.*}}%xmm
; AVX1:       retq
;
; AVX2-LABEL: bsum_v8i32:
; AVX2:       vpxor %xmm2, %xmm2, %xmm2
; AVX2-NEXT:  vpsadbw %ymm2, %ymm1, %ymm1
; AVX2-NEXT:  vpaddd %ymm1, %ymm0, %ymm0
; AVX2-NEXT:  retq
;
; AVXVNNIINT8-LABEL: bsum_v8i32:
; AVXVNNIINT8:       vpsadbw {{.*}}%ymm
; AVXVNNIINT8-NOT:   vpdpbuud
; AVXVNNIINT8:       retq
  %ext = zext <32 x i8> %a to <32 x i32>
  %r = call <8 x i32> @llvm.vector.partial.reduce.add.v8i32.v32i32(<8 x i32> %acc, <32 x i32> %ext)
  ret <8 x i32> %r
}

define <16 x i32> @bsum_v16i32(<16 x i32> %acc, <64 x i8> %a) {
; AVX2-LABEL: bsum_v16i32:
; AVX2-COUNT-2: vpsadbw {{.*}}%ymm
; AVX2:       retq
;
; AVX512-LABEL: bsum_v16i32:
; AVX512:       vpxor %xmm2, %xmm2, %xmm2
; AVX512-NEXT:  vpsadbw %zmm2, %zmm1, %zmm1
; AVX512-NEXT:  vpaddd %zmm1, %zmm0, %zmm0
; AVX512-NEXT:  retq
  %ext = zext <64 x i8> %a to <64 x i32>
  %r = call <16 x i32> @llvm.vector.partial.reduce.add.v16i32.v64i32(<16 x i32> %acc, <64 x i32> %ext)
  ret <16 x i32> %r
}

; Byte sum into i64: the psadbw lanes are the accumulator lanes.

define <4 x i64> @bsum_v4i64(<4 x i64> %acc, <32 x i8> %a) {
; AVX2-LABEL: bsum_v4i64:
; AVX2:       vpxor %xmm2, %xmm2, %xmm2
; AVX2-NEXT:  vpsadbw %ymm2, %ymm1, %ymm1
; AVX2-NEXT:  vpaddq %ymm1, %ymm0, %ymm0
; AVX2-NEXT:  retq
  %ext = zext <32 x i8> %a to <32 x i64>
  %r = call <4 x i64> @llvm.vector.partial.reduce.add.v4i64.v32i64(<4 x i64> %acc, <32 x i64> %ext)
  ret <4 x i64> %r
}

; Sum of absolute differences, in the form the loop vectorizer produces for
; abs(sub(zext(a), zext(b))): psadbw of both inputs.

define <4 x i32> @sad_v4i32(<4 x i32> %acc, <16 x i8> %a, <16 x i8> %b) {
; SSE2-LABEL: sad_v4i32:
; SSE2:       psadbw %xmm2, %xmm1
; SSE2-NEXT:  paddd %xmm1, %xmm0
; SSE2-NEXT:  retq
  %max = call <16 x i8> @llvm.umax.v16i8(<16 x i8> %a, <16 x i8> %b)
  %min = call <16 x i8> @llvm.umin.v16i8(<16 x i8> %a, <16 x i8> %b)
  %abd = sub <16 x i8> %max, %min
  %ext = zext <16 x i8> %abd to <16 x i32>
  %r = call <4 x i32> @llvm.vector.partial.reduce.add.v4i32.v16i32(<4 x i32> %acc, <16 x i32> %ext)
  ret <4 x i32> %r
}

define <8 x i32> @sad_v8i32(<8 x i32> %acc, <32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: sad_v8i32:
; AVX1-COUNT-2: vpsadbw {{.*}}%xmm
; AVX1-NOT:   vpmaxub
; AVX1:       retq
;
; AVX2-LABEL: sad_v8i32:
; AVX2:       vpsadbw %ymm2, %ymm1, %ymm1
; AVX2-NEXT:  vpaddd %ymm1, %ymm0, %ymm0
; AVX2-NEXT:  retq
;
; AVXVNNIINT8-LABEL: sad_v8i32:
; AVXVNNIINT8:       vpsadbw %ymm2, %ymm1, %ymm1
; AVXVNNIINT8-NEXT:  vpaddd %ymm1, %ymm0, %ymm0
; AVXVNNIINT8-NEXT:  retq
  %max = call <32 x i8> @llvm.umax.v32i8(<32 x i8> %a, <32 x i8> %b)
  %min = call <32 x i8> @llvm.umin.v32i8(<32 x i8> %b, <32 x i8> %a)
  %abd = sub <32 x i8> %max, %min
  %ext = zext <32 x i8> %abd to <32 x i32>
  %r = call <8 x i32> @llvm.vector.partial.reduce.add.v8i32.v32i32(<8 x i32> %acc, <32 x i32> %ext)
  ret <8 x i32> %r
}

define <8 x i64> @sad_v8i64(<8 x i64> %acc, <64 x i8> %a, <64 x i8> %b) {
; AVX512-LABEL: sad_v8i64:
; AVX512:       vpsadbw %zmm2, %zmm1, %zmm1
; AVX512-NEXT:  vpaddq %zmm1, %zmm0, %zmm0
; AVX512-NEXT:  retq
  %max = call <64 x i8> @llvm.umax.v64i8(<64 x i8> %a, <64 x i8> %b)
  %min = call <64 x i8> @llvm.umin.v64i8(<64 x i8> %a, <64 x i8> %b)
  %abd = sub <64 x i8> %max, %min
  %ext = zext <64 x i8> %abd to <64 x i64>
  %r = call <8 x i64> @llvm.vector.partial.reduce.add.v8i64.v64i64(<8 x i64> %acc, <64 x i64> %ext)
  ret <8 x i64> %r
}

; Negative tests.

; A sign extended byte sum is not an unsigned PSADBW.
define <8 x i32> @bsum_sext_v8i32(<8 x i32> %acc, <32 x i8> %a) {
; AVX2-LABEL: bsum_sext_v8i32:
; AVX2-NOT:   vpsadbw
; AVX2:       retq
  %ext = sext <32 x i8> %a to <32 x i32>
  %r = call <8 x i32> @llvm.vector.partial.reduce.add.v8i32.v32i32(<8 x i32> %acc, <32 x i32> %ext)
  ret <8 x i32> %r
}

; An i16 accumulator is not handled.
define <16 x i16> @bsum_v16i16(<16 x i16> %acc, <32 x i8> %a) {
; AVX2-LABEL: bsum_v16i16:
; AVX2-NOT:   vpsadbw
; AVX2:       retq
  %ext = zext <32 x i8> %a to <32 x i16>
  %r = call <16 x i16> @llvm.vector.partial.reduce.add.v16i16.v32i16(<16 x i16> %acc, <32 x i16> %ext)
  ret <16 x i16> %r
}

; A narrower accumulator than the input is not handled.
define <4 x i32> @bsum_v4i32_v32i8(<4 x i32> %acc, <32 x i8> %a) {
; AVX2-LABEL: bsum_v4i32_v32i8:
; AVX2-NOT:   vpsadbw
; AVX2:       retq
  %ext = zext <32 x i8> %a to <32 x i32>
  %r = call <4 x i32> @llvm.vector.partial.reduce.add.v4i32.v32i32(<4 x i32> %acc, <32 x i32> %ext)
  ret <4 x i32> %r
}

; The i8 multiply is not a byte sum, and stays a pmaddwd.
define <8 x i32> @dot_v8i32(<8 x i32> %acc, <32 x i8> %a, <32 x i8> %b) {
; AVX2-LABEL: dot_v8i32:
; AVX2-NOT:   vpsadbw
; AVX2:       vpmaddwd
; AVX2:       retq
  %ea = zext <32 x i8> %a to <32 x i32>
  %eb = zext <32 x i8> %b to <32 x i32>
  %mul = mul <32 x i32> %ea, %eb
  %r = call <8 x i32> @llvm.vector.partial.reduce.add.v8i32.v32i32(<8 x i32> %acc, <32 x i32> %mul)
  ret <8 x i32> %r
}
