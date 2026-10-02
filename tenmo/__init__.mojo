"""
Core Types.

- `Tensor` — the central type. Tracks gradients, owns data, participates in the autograd graph. Supports CPU and GPU.
- `NDBuffer` — shape, strides, offset, and data. Single source of truth for memory layout. Shared between tensors and views via ref-counting.
- `Gradbox` — gradient storage. Independently ref-counted — survives ASAP tensor destruction. Views have their own independent gradboxes.
- `Ancestor` — lightweight parent handle in the autograd graph. Carries id, grad routing, a refcounted gradbox pointer, shape, strides, offset, buffer, and device state. Does not copy full tensors.
- `Ancestors` — refcounted handle to a tensor's parent list + `BackwardFn` (stored in the private heap-only `Ancestry` record). Copies are O(1) refcount bumps; not stored on `Tensor` beyond the handle.
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
from tenmo.shared.shapes import Shape

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
from tenmo.optim import SGD

def main() raises:
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](784, 128).into(), ReLU[DType.float32]().into())
    var optimizer = SGD[DType.float32](model.parameters(), lr=0.01, momentum=0.9)
    # ... training loop
```

## Import discipline

This package init is intentionally EMPTY — no re-exports, no helpers.

Mojo 1.0 / MAX evaluates a package's `__init__.mojo` when any submodule is
imported (Python-style parent-init semantics). The old init re-exported the
whole library, so importing even a leaf kernel module (e.g.
`from tenmo.kernels.unary_device import float_unary_ops`) dragged in the entire
106-module tensor core and, on a GPU box, JIT-compiled every kernel. Keeping
this file docstring-only makes any `tenmo.*` submodule import cheap and keeps
kernel tests decoupled.

Import everything from its DEFINING module (the repo's lean-import rule):

- `from tenmo.tensor import Tensor`
- `from tenmo.ndbuffer import NDBuffer`
- `from tenmo.shared.shapes import Shape`
- `from tenmo.shared.intarray import IntArray`
- `from tenmo.shared.buffers import Buffer`
- `from tenmo.shared import Reduction`
- `from tenmo.accuracy import Accuracy`
- `from tenmo.shared.mnemonics import DEFAULT_INDEX_DTYPE`
- `from tenmo.kernels.unary_device import float_unary_ops`
"""
