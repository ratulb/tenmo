---
title: pad
version: 0.26.2.0
type: module
namespace: tenmo
lang: mojo
description: "PADDING SPECIFICATION:"
---

<section class='mojo-docs'>



<div class='mojo-module-detail'><!-- here only for Listing component -->


PADDING SPECIFICATION:

For N-dimensional tensor, padding is specified as a list of tuples:
[(before_0, after_0), (before_1, after_1), ..., (before_N-1, after_N-1)]

Or as a flat list (PyTorch style, applied from last to first dimension):
[before_last, after_last, before_second_last, after_second_last, ...]

Examples:
- 2D tensor (H, W): pad = [(1, 2), (3, 4)] means:
  - Dimension 0 (H): add 1 before, 2 after
  - Dimension 1 (W): add 3 before, 4 after

- 4D tensor (N, C, H, W): pad = [(0, 0), (0, 0), (1, 1), (2, 2)] means:
  - No padding on batch and channel dimensions
  - Pad H with 1 on each side
  - Pad W with 2 on each side



</div>


## `comptime` values

<div class='mojo-alias-header'>

###  `Padding`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime Padding = Variant[String, Int, Tuple[Int, Int], List[Tuple[Int, Int]]] ``

</div>


</div>


## Structs

* [​`Pad`](./Pad): Generalized padding operation supporting: - Arbitrary dimensions. - Asymmetric padding (different on each side). - Multiple padding modes. - Proper gradient flow in backward pass.
* [​`PadBackward`](./PadBackward): Backward pass for padding operation - handles all modes.

</section>