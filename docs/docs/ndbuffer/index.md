---
title: ndbuffer
version: 0.26.2.0
type: module
namespace: tenmo
lang: mojo
description: "Mojo module tenmo.ndbuffer documentation"
---

<section class='mojo-docs'>



<div class='mojo-module-detail'><!-- here only for Listing component -->








</div>


## `comptime` values

<div class='mojo-alias-header'>

###  `TILE_SIZE`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime TILE_SIZE = 32 ``

</div>


</div>


## Structs

* [​`Layout`](./Layout): Pure metadata describing how data is laid out in memory. No data, no device, no allocation. Device-agnostic — same for CPU and GPU.
* [​`NDBuffer`](./NDBuffer): 
* [​`Storage`](./Storage): Pure data carrier — CPU buffer or GPU device state. No shape knowledge. No layout knowledge. copy() is cheap — just a refcount bump.

</section>