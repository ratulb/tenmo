---
title: argminmax
version: 0.26.2.0
type: module
namespace: tenmo
lang: mojo
description: "Mojo module tenmo.argminmax documentation"
---

<section class='mojo-docs'>



<div class='mojo-module-detail'><!-- here only for Listing component -->








</div>


## Structs

* [​`Argmax`](./Argmax): 
* [​`Argmin`](./Argmin): 
* [​`ArgMinMaxReducer`](./ArgMinMaxReducer): Unified CPU + GPU argmin/argmax on NDBuffer. Returns an NDBuffer[DType.int32] with the output shape.

## Functions

* [​`reduce_argminmax`](./reduce_argminmax): One block per output element. Each thread strides over the reduction axis, tracking local best value and its index. Then a two-array shared-memory tree reduction picks the global best index for this output slot.

</section>