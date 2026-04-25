"""
# Core Types
- `Tensor` — the central type. Tracks gradients, owns data, participates in the autograd graph. Supports CPU and GPU.
- `NDBuffer` — shape, strides, offset, and data. Single source of truth for memory layout. Shared between tensors and views via ref-counting.
- `Gradbox` — gradient storage. Independently ref-counted — survives ASAP tensor destruction. Views have their own independent gradboxes.
- `Ancestor` — lightweight parent handle in the autograd graph. Carries id, grad routing, a refcounted gradbox pointer, `Layout`, and `Storage`. Does not copy full tensors.
- `Ancestors` — the ancestry list for a tensor. Carries the `BackwardFnArg` directly — not stored on `Tensor`.
- `Layout` — shape, strides, offset, contiguity. Pure metadata, no data.
- `Storage` — CPU `Buffer` or GPU `DeviceState`. Refcount bump on copy.
- `Buffer` — linear, SIMD-capable storage. Ref-counted when shared via views.

## Quick Start

```mojo
from tenmo.tensor import Tensor

fn main() raises:
    var a = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = a * 2
    var c = a * 3
    var d = b + c
    var loss = d.sum()
    loss.backward()
    # a.grad() == [5.0, 5.0, 5.0]
    a.grad().print()
```

## GPU Example

```mojo
from tenmo.tensor import Tensor
from tenmo.shapes import Shape

fn main() raises:
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = Tensor.full_gpu(Shape.of(2, 2), 2.0)
    var c_gpu = a_gpu * b_gpu
    var loss = c_gpu.sum()
    loss.backward()
    a.grad().print()
```

## Training Example

```mojo
from tenmo.tensor import Tensor
from tenmo.net import Sequential, Linear, ReLU
from tenmo.sgd import SGD

fn main() raises:
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](784, 128).into(), ReLU[DType.float32]().into())
    var optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # ... training loop
```
"""

from  .addition import *
from  .ancestry import *
from  .argminmax import *
from  .array import *
from  .backpropagation import *
from  .binary_inplace_ops_kernel import *
from  .binary_ops_kernel import *
from  .blashandle import *
from  .broadcastbackward import *
from  .broadcasthelper import *
from  .buffers import *
from  .clip import *
from  .cnn import *
from  .common_utils import *
from  .compare_kernel import *
from  .concate import *
from  .contiguous import *
from  .crossentropy import *
from  .dataloader import *
from  .device import *
from  .device_transfer import *
from  .division import *
from  .dotproduct import *
from  .dropout import *
from  .expand import *
from  .exponential import *
from  .exponentiator import *
from  .filler import *
from  .flatten import *
from  .forwards import *
from  .gradbox import *
from  .indexhelper import *
from  .intarray import *
from  .logarithm import *
from  .matmul import *
from  .matmul_kernel import *
from  .matrixshapevalidator import *
from  .matrixvector import *
from  .matrixvector_kernel import *
from  .maxmin_scalar import *
from  .mean_reduction import *
from  .minmax import *
from  .minmax_kernel import *
from  .minmax_reducer import *
from  .mnemonics import *
from  .mse import *
from  .multiplication import *
from  .ndbuffer import *
from  .net import *
from  .numpy_interop import *
from  .pad import *
from  .permute import *
from  .pooling import *
from  .reduction_kernel import *
from  .relu import *
from  .repeat import *
from  .reshape import *
from  .scalar_inplace_ops_kernel import *
from  .scalar_ops_kernel import *
from  .sgd import *
from  .shapes import *
from  .shuffle import *
from  .sigmoid import *
from  .softmax import *
from  .squareroot import *
from  .squeeze import *
from  .stack import *
from  .std_deviation import *
from  .strides import *
from  .subtraction import *
from  .summation import *
from  .tanh import *
from  .tensor import *
from  .tiles import *
from  .transpose import *
from  .unary_ops_kernel import *
from  .unsqueeze import *
from  .validators import *
from  .variance import *
from  .vectormatrix import *
from  .vectormatrix_kernel import *
from  .views import *
from  .walkback import *


