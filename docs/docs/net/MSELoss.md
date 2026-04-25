---
title: MSELoss
version: 0.26.2.0
slug: MSELoss
type: struct
namespace: tenmo.net
lang: mojo
show_stability_marker: 
description: "Mojo struct `tenmo.net.MSELoss` documentation"
---

<section class='mojo-docs'>




<div class="mojo-function-sig">

`` struct MSELoss[dtype: DType = DType.float32] ``

</div>










## Fields

* ​<b>training</b> (``Bool``): 

## Implemented traits

[`AnyType`](/mojo/std/builtin/anytype/AnyType), 
[`ImplicitlyDestructible`](/mojo/std/builtin/anytype/ImplicitlyDestructible), 
[`Movable`](/mojo/std/builtin/value/Movable), 
[`RegisterPassable`](/mojo/std/builtin/value/RegisterPassable)


## Methods

### `__init__`

<div class='mojo-function-detail'>


<div class="mojo-function-sig">

`` __init__() -> Self ``


</div>







</div>



### `__call__`

<div class='mojo-function-detail'>


<div class="mojo-function-sig">

`` __call__(self, preds: Tensor[dtype], target: Tensor[dtype]) -> Tensor[dtype] ``


</div>






**Returns:**

[``Tensor``](/mojo/tenmo/tensor/Tensor)

</div>



### `train`

<div class='mojo-function-detail'>


<div class="mojo-function-sig">

`` train(mut self) ``


</div>







</div>



### `eval`

<div class='mojo-function-detail'>


<div class="mojo-function-sig">

`` eval(mut self) ``


</div>







</div>





</section>