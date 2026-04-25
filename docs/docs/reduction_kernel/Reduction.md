---
title: Reduction
version: 0.26.2.0
slug: Reduction
type: struct
namespace: tenmo.reduction_kernel
lang: mojo
show_stability_marker: 
description: "Mojo struct `tenmo.reduction_kernel.Reduction` documentation"
---

<section class='mojo-docs'>




<div class="mojo-function-sig">

`` struct Reduction[dtype: DType = DType.float32] ``

</div>










## Implemented traits

[`AnyType`](/mojo/std/builtin/anytype/AnyType), 
[`Copyable`](/mojo/std/builtin/value/Copyable), 
[`ImplicitlyCopyable`](/mojo/std/builtin/value/ImplicitlyCopyable), 
[`ImplicitlyDestructible`](/mojo/std/builtin/anytype/ImplicitlyDestructible), 
[`Movable`](/mojo/std/builtin/value/Movable), 
[`RegisterPassable`](/mojo/std/builtin/value/RegisterPassable)


## Methods

### `launch`

<div class='mojo-function-detail'>


<div class="mojo-function-sig">

`` static launch[max_block_width: Int = 512, mean: Bool = False](A: NDBuffer[dtype], normalized_axes: IntArray, keepdims: Bool) -> NDBuffer[dtype] ``


</div>






**Returns:**

[``NDBuffer``](/mojo/tenmo/ndbuffer/NDBuffer)

</div>



### `launch_log_sum`

<div class='mojo-function-detail'>


<div class="mojo-function-sig">

`` static launch_log_sum[max_block_width: Int = 512, epsilon: Scalar[dtype] = Epsilon.value()](A: NDBuffer[dtype], normalized_axes: IntArray, keepdims: Bool) -> NDBuffer[dtype] ``


</div>






**Returns:**

[``NDBuffer``](/mojo/tenmo/ndbuffer/NDBuffer)

</div>



### `launch_config`

<div class='mojo-function-detail'>


<div class="mojo-function-sig">

`` static launch_config[max_block_size: Int](total_output: Int, reduced_volume: Int) -> Tuple[Int, Int] ``


</div>






**Returns:**

``Tuple``

</div>





</section>