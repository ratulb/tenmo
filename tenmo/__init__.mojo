"""

# Core Types

- `Tensor` — the central type. Tracks gradients, owns data, participates in the autograd graph. Supports CPU and GPU.
- `NDBuffer` — shape, strides, offset, and data. Single source of truth for memory layout. Shared between tensors and views via ref-counting.
- `Gradbox` — gradient storage. Independently ref-counted — survives ASAP tensor destruction. Views have their own independent gradboxes.
- `Ancestor` — lightweight parent handle in the autograd graph. Carries id, grad routing, a refcounted gradbox pointer, shape, strides, offset, buffer, and device state. Does not copy full tensors.
- `Ancestors` — the ancestry list for a tensor. Carries the `BackwardFnArg` directly — not stored on `Tensor`.
- `Buffer` — linear, SIMD-capable storage. Ref-counted when shared via views.

## Quick Start

```mojo
from tenmo.tensor import Tensor

def main() raises:
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

def main() raises:
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = Tensor.full(Shape(2, 2), 2.0).to_gpu()
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

def main() raises:
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](784, 128).into(), ReLU[DType.float32]().into())
    var optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # ... training loop
```
"""

# Core Types - explicit exports for documentation links
from .tensor import Tensor
from .ndbuffer import NDBuffer
from .gradbox import Gradbox
from .ancestry import Ancestor, Ancestors

from .buffers import Buffer
from .shapes import Shape
from .strides import Strides
from .intarray import IntArray
from .array import Array

# Modules
from .addition import *
from .ancestry import *
from .argminmax import *

# from  .array import *
from .backpropagation import *
from .kernels.binary_inplace_ops_kernel import *
from .kernels.binary_ops_kernel import *
from .blashandle import *
from .broadcast import *
from .broadcasthelper import *

from .clip import *
from .cnn import *
from .common_utils import *
from .kernels.compare_kernel import *
from .concate import *
from .contiguous import *
from .crossentropy import *
from .dataloader import *
from .device import *
from .device_transfer import *
from .division import *
from .dotproduct import *
from .dropout import *
from .expand import *
from .exponential import *
from .exponentiator import *
from .filler import *
from .flatten import *
from .forwards import *

from .indexhelper import *

from .logarithm import *
from .matmul import *
from .matmul_cpu import MmCpu2d, MmCpuNd
from .kernels.matmul_kernel import *
from .matrixshapevalidator import *
from .matrixvector import *
from .kernels.matrixvector_kernel import *
from .maxmin_scalar import *
from .mean_reduction import *
from .minmax import *
from .kernels.minmax_kernel import *
from .minmax_reducer import *
from .mnemonics import *
from .named_parameter import NamedParameter
from .mse import *
from .multiplication import *
from .ndbuffer import *
from .net import *
from .numpy_interop import *
from .pad import *
from .permute import *
from .pooling import *
from .kernels.reduction_kernel import *
from .relu import *
from .repeat import *
from .reshape import *
from .kernels.scalar_inplace_ops_kernel import *
from .kernels.scalar_ops_kernel import *
from .sgd import *

from .shuffle import *
from .sigmoid import *
from .softmax import *
from .squareroot import *
from .squeeze import *
from .stack import *
from .std_deviation import *
from .strides import *
from .subtraction import *
from .sum_mean_reduction import *
from .sum_reduction import *
from .product_reduction import *
from .tanh import *

from .tiles import *
from .transpose import *
from .kernels.unary_ops_kernel import *
from .unsqueeze import *
from .validators import *
from .variance import *
from .welford import *
from .vectormatrix import *
from .kernels.vectormatrix_kernel import *
from .views import *
from .walkback import *
