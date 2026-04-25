---
title: ReLU
version: 0.26.2.0
slug: ReLU
type: struct
namespace: tenmo.relu
lang: mojo
show_stability_marker: 
description: "Mojo struct `tenmo.relu.ReLU` documentation"
---

<section class='mojo-docs'>




<div class="mojo-function-sig">

`` struct ReLU[dtype: DType] ``

</div>










## Implemented traits

[`AnyType`](/mojo/std/builtin/anytype/AnyType), 
[`Copyable`](/mojo/std/builtin/value/Copyable), 
[`ImplicitlyCopyable`](/mojo/std/builtin/value/ImplicitlyCopyable), 
[`ImplicitlyDestructible`](/mojo/std/builtin/anytype/ImplicitlyDestructible), 
[`Movable`](/mojo/std/builtin/value/Movable), 
[`RegisterPassable`](/mojo/std/builtin/value/RegisterPassable)


## Methods

### `forward`

<div class='mojo-function-detail'>


<div class="mojo-function-sig">

`` static forward[track_grad: Bool = True](self: Tensor[dtype], requires_grad: Optional[Bool] = None) -> Tensor[dtype] ``


</div>

Apply ReLU activation: max(0, x).

Computes output and mask simultaneously for efficiency.
Routes through NDBuffer.unary_ops_with_mask — GPU and CPU aware.



**Args:**

*   ​<b>self</b> ([``Tensor``](/mojo/tenmo/tensor/Tensor)): Input tensor.
*   ​<b>requires_grad</b> ([``Optional``](/mojo/std/collections/optional/Optional)): Override gradient tracking (default: inherit from input).

**Returns:**

[``Tensor``](/mojo/tenmo/tensor/Tensor): Output tensor with ReLU applied.

</div>





</section>