---
title: common_utils
version: 0.26.2.0
type: module
namespace: tenmo
lang: mojo
description: "Mojo module tenmo.common_utils documentation"
---

<section class='mojo-docs'>



<div class='mojo-module-detail'><!-- here only for Listing component -->








</div>


## `comptime` values

<div class='mojo-alias-header'>

###  `BLUE`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BLUE = "\1B[34m" ``

</div>


</div>

<div class='mojo-alias-header'>

###  `BRIGHT_BLUE`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime BRIGHT_BLUE = "\1B[94m" ``

</div>


</div>

<div class='mojo-alias-header'>

###  `CYAN`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime CYAN = "\1B[36m" ``

</div>


</div>

<div class='mojo-alias-header'>

###  `Idx`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime Idx = Variant[Int, IntArray, Slice, NewAxis] ``

</div>


</div>

<div class='mojo-alias-header'>

###  `log`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime log = Logger(stdout, "", False) ``

</div>


</div>

<div class='mojo-alias-header'>

###  `LOG_LEVEL`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime LOG_LEVEL = get_defined_string["LOGGING_LEVEL", "INFO"]() ``

</div>


</div>

<div class='mojo-alias-header'>

###  `MAGENTA`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime MAGENTA = "\1B[35m" ``

</div>


</div>

<div class='mojo-alias-header'>

###  `newaxis`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime newaxis = Variant.__init__[NewAxis](NewAxis()) ``

</div>


</div>

<div class='mojo-alias-header'>

###  `RED`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime RED = "\1B[31m" ``

</div>


</div>

<div class='mojo-alias-header'>

###  `RESET`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime RESET = "\1B[0m" ``

</div>


</div>

<div class='mojo-alias-header'>

###  `YELLOW`



</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">


`` comptime YELLOW = "\1B[33m" ``

</div>


</div>


## Structs

* [​`Epsilon`](./Epsilon): 
* [​`IDGen`](./IDGen): 
* [​`NewAxis`](./NewAxis): 
* [​`One`](./One): 
* [​`Slicer`](./Slicer): 
* [​`Zero`](./Zero): 

## Functions

* [​`addr`](./addr): 
* [​`addrs`](./addrs): 
* [​`assert_grad`](./assert_grad): 
* [​`binary_accuracy`](./binary_accuracy): 
* [​`copy`](./copy): General-purpose optimized copy with smart defaults.
* [​`do_assert`](./do_assert): 
* [​`i`](./i): 
* [​`id`](./id): 
* [​`il`](./il): 
* [​`inf`](./inf): Gets a +inf value for the given dtype.
* [​`is_null`](./is_null): 
* [​`isinf`](./isinf): 
* [​`isnan`](./isnan): 
* [​`log_debug`](./log_debug): 
* [​`log_info`](./log_info): 
* [​`log_warning`](./log_warning): 
* [​`multiclass_accuracy`](./multiclass_accuracy): 
* [​`nan`](./nan): Gets a NaN value for the given dtype.
* [​`now`](./now): 
* [​`panic`](./panic): 
* [​`print_buffer`](./print_buffer): 
* [​`print_summary`](./print_summary): 
* [​`s`](./s): 
* [​`str_repeat`](./str_repeat): 

</section>