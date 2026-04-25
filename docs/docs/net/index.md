---
title: net
version: 0.26.2.0
type: module
namespace: tenmo
lang: mojo
description: "Mojo module tenmo.net documentation"
---

<section class='mojo-docs'>



<div class='mojo-module-detail'><!-- here only for Listing component -->








</div>


## `comptime` values

<div class='mojo-alias-header'>

###  `Layer`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime Layer[dtype: DType] = Variant[Linear[dtype], LinearBLAS[dtype], ReLU[dtype], Sigmoid[dtype], Tanh[dtype], Dropout[dtype], Conv2D[dtype], Flatten[dtype], MaxPool2d[dtype]] ``

</div>



#### Parameters

*   ​<b>dtype</b> ([``DType``](/std/builtin/dtype/DType)): 
</div>


## Structs

* [​`BCELoss`](./BCELoss): 
* [​`BCEWithLogitsLoss`](./BCEWithLogitsLoss): 
* [​`Conv2D`](./Conv2D): Conv2D layer wrapper for Sequential integration.
* [​`Flatten`](./Flatten): Flatten spatial dimensions: (N, C, H, W) → (N, C*H*W).
* [​`Linear`](./Linear): Fully connected layer: y = xW + b.
* [​`LinearBLAS`](./LinearBLAS): Fully connected layer: y = xW + b.
* [​`Module`](./Module): 
* [​`MSELoss`](./MSELoss): 
* [​`Profile`](./Profile): Profile for a specific batch size.
* [​`ReLU`](./ReLU): 
* [​`Sequential`](./Sequential): 
* [​`SequentialBLAS`](./SequentialBLAS): 
* [​`Sigmoid`](./Sigmoid): 
* [​`Tanh`](./Tanh): 

</section>