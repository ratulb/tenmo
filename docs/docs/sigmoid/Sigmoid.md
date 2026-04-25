---
title: Sigmoid
version: 0.26.2.0
slug: Sigmoid
type: struct
namespace: tenmo.sigmoid
lang: mojo
show_stability_marker: 
description: "Mojo struct `tenmo.sigmoid.Sigmoid` documentation"
---

<section class='mojo-docs'>




<div class="mojo-function-sig">

`` struct Sigmoid[dtype: DType] ``

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

`` static forward[track_grad: Bool = True](self: Tensor[dtype], requires_grad: Optional[Bool] = None) -> Tensor[dtype] where dtype.is_floating_point() ``


</div>






**Returns:**

[``Tensor``](/mojo/tenmo/tensor/Tensor)

</div>





</section>