---
title: unary_ops_kernel
version: 0.26.2.0
type: module
namespace: tenmo
lang: mojo
description: "Mojo module tenmo.unary_ops_kernel documentation"
---

<section class='mojo-docs'>



<div class='mojo-module-detail'><!-- here only for Listing component -->








</div>


## Structs

* [​`UnaryOpsKernel`](./UnaryOpsKernel): 

## Functions

* [​`float_unary_ops`](./float_unary_ops): Floating point unary ops kernel — LOG, EXP, TANH, SIGMOID. Requires dtype.is_floating_point() — supported in Mojo 0.26.2+.
* [​`invert_bool`](./invert_bool): Logical NOT for bool stored as uint8. 0 -> 1, 1 -> 0.
* [​`unary_ops`](./unary_ops): Generic unary ops kernel — SQRT, NEGATE, ABS, RELU. LOG, EXP, TANH, SIGMOID are handled by float_unary_ops. RELU = max(x, 0) — pure arithmetic, safe for any dtype.
* [​`unary_ops_with_mask`](./unary_ops_with_mask): Single-pass kernel: compute activation output AND gradient mask.

</section>