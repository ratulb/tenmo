---
title: cnn
version: 0.26.2.0
type: module
namespace: tenmo
lang: mojo
description: "Mojo module tenmo.cnn documentation"
---

<section class='mojo-docs'>



<div class='mojo-module-detail'><!-- here only for Listing component -->








</div>


## Structs

* [​`Conv2dFused`](./Conv2dFused): Batched, multi-channel, multi-filter 2D convolution using fused im2col + matmul + bias. Args:     image: (N, C_in, H_in, W_in).     kernel: (C_out, C_in, KH, KW).     bias: Optional (C_out,).     stride: Stride for spatial dimensions.     dilation: Dilation factor for atrous convolution.     padding: 'valid', 'same', int, tuple, or list of tuples. Returns:     output: (N, C_out, H_out, W_out).
* [​`FusedCol2ImBackward`](./FusedCol2ImBackward): Convolution backward pass.
* [​`FusedIm2Col`](./FusedIm2Col): Fused Im2Col + Conv + Bias operation.
* [​`FusedIm2ColBwdArg`](./FusedIm2ColBwdArg): 

</section>