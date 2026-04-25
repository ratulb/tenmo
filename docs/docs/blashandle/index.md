---
title: blashandle
version: 0.26.2.0
type: module
namespace: tenmo
lang: mojo
description: "Mojo module tenmo.blashandle documentation"
---

<section class='mojo-docs'>



<div class='mojo-module-detail'><!-- here only for Listing component -->








</div>


## `comptime` values

<div class='mojo-alias-header'>

###  `BLAS_PATH`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BLAS_PATH = get_defined_string["BLAS_PATH", "/lib/x86_64-linux-gnu/libopenblas.so.0"]() ``

</div>


</div>

<div class='mojo-alias-header'>

###  `CBLAS_DGEMM_FN`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime CBLAS_DGEMM_FN = def(Int32, Int32, Int32, Int32, Int32, Int32, Float64, UnsafePointer[Float64, MutAnyOrigin], Int32, UnsafePointer[Float64, MutAnyOrigin], Int32, Float64, UnsafePointer[Float64, MutAnyOrigin], Int32) -> None ``

</div>


</div>

<div class='mojo-alias-header'>

###  `CBLAS_SGEMM_FN`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime CBLAS_SGEMM_FN = def(Int32, Int32, Int32, Int32, Int32, Int32, Float32, UnsafePointer[Float32, MutAnyOrigin], Int32, UnsafePointer[Float32, MutAnyOrigin], Int32, Float32, UnsafePointer[Float32, MutAnyOrigin], Int32) -> None ``

</div>


</div>

<div class='mojo-alias-header'>

###  `CblasColMajor`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime CblasColMajor = 102 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `CblasNoTrans`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime CblasNoTrans = 111 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `CblasRowMajor`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime CblasRowMajor = 101 ``

</div>


</div>

<div class='mojo-alias-header'>

###  `CblasTrans`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime CblasTrans = 112 ``

</div>


</div>


## Structs

* [​`BLASHandle`](./BLASHandle): 
* [​`BLASHandleLite`](./BLASHandleLite): 
* [​`BLASMatmul2dBackward`](./BLASMatmul2dBackward): 

</section>