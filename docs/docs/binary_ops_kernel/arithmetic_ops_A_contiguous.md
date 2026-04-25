---
title: arithmetic_ops_A_contiguous
version: 0.26.2.0
slug: arithmetic_ops_A_contiguous
type: function
namespace: tenmo.binary_ops_kernel
lang: mojo
show_stability_marker: 
description: "Mojo function `tenmo.binary_ops_kernel.arithmetic_ops_A_contiguous` documentation"
---

<section class='mojo-docs'>


<div class='mojo-function-detail'>

<div class="mojo-function-sig">

``arithmetic_ops_A_contiguous[op_code: Int, dtype: DType, simd_width: Int = simd_width_of[dtype](), simd_vectors_per_thread: Int = (2 * simd_width)](result: UnsafePointer[Scalar[dtype], MutAnyOrigin], A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin], B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin], result_shape: Array, B_strides: Array, size: Int, rank: Int, epsilon: Scalar[dtype] = Epsilon.value())``


</div>








</div>


</section>