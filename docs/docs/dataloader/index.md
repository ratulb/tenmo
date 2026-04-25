---
title: dataloader
version: 0.26.2.0
type: module
namespace: tenmo
lang: mojo
description: "Mojo module tenmo.dataloader documentation"
---

<section class='mojo-docs'>



<div class='mojo-module-detail'><!-- here only for Listing component -->








</div>


## `comptime` values

<div class='mojo-alias-header'>

###  `CIFAR10_MEAN`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime CIFAR10_MEAN = Tuple(VariadicPack(0.4914, 0.48220000000000002, 0.44650000000000001)) ``

</div>


</div>

<div class='mojo-alias-header'>

###  `CIFAR10_STD`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime CIFAR10_STD = Tuple(VariadicPack(0.247, 0.24349999999999999, 0.2616)) ``

</div>


</div>

<div class='mojo-alias-header'>

###  `FASHION_MNIST_MEAN`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime FASHION_MNIST_MEAN = 0.28599999999999998 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `FASHION_MNIST_STD`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime FASHION_MNIST_STD = 0.35299999999999998 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `IMAGENET_MEAN`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime IMAGENET_MEAN = Tuple(VariadicPack(0.48499999999999999, 0.45600000000000002, 0.40600000000000003)) ``

</div>


</div>

<div class='mojo-alias-header'>

###  `IMAGENET_STD`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime IMAGENET_STD = Tuple(VariadicPack(0.22900000000000001, 0.224, 0.22500000000000001)) ``

</div>


</div>

<div class='mojo-alias-header'>

###  `MNIST_MEAN`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime MNIST_MEAN = 0.13070000000000001 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `MNIST_STD`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime MNIST_STD = 0.30809999999999998 ``

</div>


</div>


## Structs

* [​`Batch`](./Batch): 
* [​`DataLoader`](./DataLoader): Zero-copy batched data loading that preserves tensor shapes.
* [​`NumpyDataset`](./NumpyDataset): Dataset that preserves original tensor shapes.
* [​`TensorDataset`](./TensorDataset): Dataset from tensors. References existing data (no copy).

## Traits

* [​`Dataset`](./Dataset): 

</section>