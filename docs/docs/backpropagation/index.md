---
title: backpropagation
version: 0.26.2.0
type: module
namespace: tenmo
lang: mojo
description: "Mojo module tenmo.backpropagation documentation"
---

<section class='mojo-docs'>



<div class='mojo-module-detail'><!-- here only for Listing component -->








</div>


## `comptime` values

<div class='mojo-alias-header'>

###  `BACKWARD_ADD`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_ADD = 0 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_ADD_BROADCAST`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_ADD_BROADCAST = 23 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_ADD_SCALAR`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_ADD_SCALAR = 22 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_CE_CLASS_INDICES`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_CE_CLASS_INDICES = 9 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_CE_PROBABILITIES`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_CE_PROBABILITIES = 10 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_CLIP`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_CLIP = 40 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_CONCAT`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_CONCAT = 44 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_CONTIGUOUS`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_CONTIGUOUS = 18 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_DEVICE_TRANSFER`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_DEVICE_TRANSFER = 51 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_DIV_SCALAR`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_DIV_SCALAR = 27 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_DIVIDE`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_DIVIDE = 19 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_DOT`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_DOT = 30 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_DROPOUT`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_DROPOUT = 49 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_EXPAND`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_EXPAND = 31 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_EXPONENTIAL`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_EXPONENTIAL = 50 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_EXPONENTIATION`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_EXPONENTIATION = 29 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_FLATTEN`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_FLATTEN = 32 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_FUSED_CONV`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_FUSED_CONV = 47 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_LOG`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_LOG = 38 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_LOG_SOFTMAX`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_LOG_SOFTMAX = 17 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_MATMUL_2D`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_MATMUL_2D = 4 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_MATMUL_ND`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_MATMUL_ND = 3 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_MATRIX_VECTOR_MUL`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_MATRIX_VECTOR_MUL = 20 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_MAX_SCALAR`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_MAX_SCALAR = 52 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_MAXPOOL2D`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_MAXPOOL2D = 48 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_MEAN`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_MEAN = 15 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_MIN_SCALAR`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_MIN_SCALAR = 53 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_MINMAX`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_MINMAX = 36 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_MULTIPLY`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_MULTIPLY = 1 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_MULTIPLY_BROADCAST`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_MULTIPLY_BROADCAST = 54 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_MULTIPLY_SCALAR`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_MULTIPLY_SCALAR = 24 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_PAD`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_PAD = 46 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_PERMUTE`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_PERMUTE = 6 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_RELU`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_RELU = 2 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_RESHAPE`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_RESHAPE = 13 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_RIGHT_DIV_SCALAR`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_RIGHT_DIV_SCALAR = 28 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_SHUFFLE`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_SHUFFLE = 35 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_SIGMOID`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_SIGMOID = 7 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_SOFTMAX`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_SOFTMAX = 8 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_SQRT`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_SQRT = 39 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_SQUEEZE`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_SQUEEZE = 33 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_STACK`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_STACK = 45 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_STD`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_STD = 42 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_SUB`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_SUB = 12 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_SUB_SCALAR`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_SUB_SCALAR = 25 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_SUBTRACT_BROADCAST`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_SUBTRACT_BROADCAST = 26 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_SUM`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_SUM = 16 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_TANH`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_TANH = 11 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_TILE`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_TILE = 37 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_TRANSPOSE`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_TRANSPOSE = 5 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_UNSQUEEZE`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_UNSQUEEZE = 34 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_VARIANCE`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_VARIANCE = 41 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_VECTOR_MATMUL`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_VECTOR_MATMUL = 21 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BACKWARD_VIEW`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BACKWARD_VIEW = 14 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BLAS_BACKWARD_MATMUL_2D`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BLAS_BACKWARD_MATMUL_2D = 43 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `CopyFn`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime CopyFn = def(UnsafePointer[UInt8, MutAnyOrigin]) -> UnsafePointer[UInt8, MutAnyOrigin] ``

</div>


</div>

<div class='mojo-alias-header'>

###  `DestroyerFn`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime DestroyerFn = def(UnsafePointer[UInt8, MutAnyOrigin]) -> None ``

</div>


</div>


## Structs

* [​`Backward`](./Backward): 
* [​`BackwardFnArg`](./BackwardFnArg): 
* [​`BlasArg`](./BlasArg): 
* [​`Boolean`](./Boolean): 
* [​`BufferArg`](./BufferArg): 
* [​`ClipArg`](./ClipArg): 
* [​`IntArrayArg`](./IntArrayArg): 
* [​`Integer`](./Integer): 
* [​`MinMaxArg`](./MinMaxArg): 
* [​`NDBufferArg`](./NDBufferArg): 
* [​`NullArg`](./NullArg): 
* [​`PadArg`](./PadArg): 
* [​`ReductionArg`](./ReductionArg): 
* [​`ScalarArg`](./ScalarArg): 
* [​`ShuffleArg`](./ShuffleArg): 
* [​`SoftmaxArg`](./SoftmaxArg): 
* [​`StackArg`](./StackArg): 
* [​`StdArg`](./StdArg): 
* [​`TilesArg`](./TilesArg): 
* [​`ViewArg`](./ViewArg): 

## Traits

* [​`ArgumentType`](./ArgumentType): 

## Functions

* [​`make_copier`](./make_copier): 
* [​`make_destroyer`](./make_destroyer): 

</section>