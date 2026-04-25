---
title: device
version: 0.26.2.0
type: module
namespace: tenmo
lang: mojo
description: "Mojo module tenmo.device documentation"
---

<section class='mojo-docs'>



<div class='mojo-module-detail'><!-- here only for Listing component -->








</div>


## `comptime` values

<div class='mojo-alias-header'>

###  `DeviceType`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime DeviceType = Variant[CPU, GPU] ``

</div>


</div>


## Structs

* [​`CPU`](./CPU): 
* [​`Device`](./Device): 
* [​`DeviceState`](./DeviceState): GPU device state for NDBuffer. DType.bool is stored internally as DType.uint8 since DeviceBuffer[DType.bool] is unsupported on GPU. All buffer operations cast accordingly.
* [​`GPU`](./GPU): Essentially a shared DeviceContext.

</section>