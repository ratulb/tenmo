# Tenmo
## Build Status

![Mojo Tests](https://github.com/ratulb/tenmo/actions/workflows/test.yml/badge.svg?branch=development)
![Last Commit](https://img.shields.io/github/last-commit/ratulb/tenmo/development)
![License](https://img.shields.io/github/license/ratulb/tenmo)
![Language](https://img.shields.io/badge/language-Mojo%20🔥-orange)
![Open Issues](https://img.shields.io/github/issues/ratulb/tenmo)

**A lean tensor library and neural network framework built entirely in Mojo 🔥**

Tenmo brings modern, ergonomic ML abstractions to Mojo with automatic differentiation, modular neural networks, GPU acceleration, and end-to-end training pipelines — built from first principles with full visibility into every layer of the stack.

> ⚠️ **Development Status**: Tenmo is actively evolving alongside Mojo itself. The API is subject to change as we incorporate improvements from the Mojo ecosystem. Not production-ready yet, but excellent for learning, experimentation, and systems-level exploration.

---

## What's New

The library has undergone significant architectural work since the benchmarks below were recorded. The changes prioritize correctness, safety, and GPU support — at a measured cost to raw CPU throughput. The benchmarks are preserved for historical reference but **no longer reflect current performance**. Updated numbers will be published once GPU training benchmarks are available.

### Major Recent Work

**Backward system redesign** — moved from stateful handler instances embedded in a fat `Delegate` variant to pure static methods with a type-erased `BackwardFnArg`. Dispatch is now a direct integer-tag jump table. No variant extraction, no handler instances, no redundant copies.

**Ancestry redesign** — `Ancestors` no longer stores full `Tensor` copies. Each ancestor is now a lightweight `Ancestor` handle carrying only what backward needs: an id, a `requires_grad` flag, a refcounted gradbox pointer, a `Layout` (shape/strides/offset), and a `Storage` (refcount bump — CPU or GPU). The recursive deep-copy explosion on every `add_ancestry` call is gone.

**`BackwardFnArg` off `Tensor`** — the backward function argument now lives on `Ancestors` directly, set atomically at ancestry registration time. Intermediate tensors carry zero backward function overhead.

**Gradbox refcounting** — `Gradbox` now has its own atomic refcount independent of `Buffer`. Intermediate tensors can be ASAP-destroyed by Mojo without dangling pointers in the ancestry graph. Last owner frees.

**`Layout` and `Storage`** — introduced as first-class types in the NDBuffer module. `Layout` carries shape, strides, offset, and contiguity. `Storage` carries the CPU `Buffer` or GPU `DeviceState`. `Ancestor` composes these directly — no full `NDBuffer` machinery needed on each ancestry entry.

**GPU support** — tensor operations, backward passes, and gradient flow now work on GPU. `DType.bool` is handled correctly on GPU via internal `uint8` storage throughout kernels, `DeviceState`, and result construction. `INVERT`, `SQRT`, `NEGATE`, `ABS`, `RELU` all have GPU kernel support. `SQRT_BACKWARD` uses `rsqrt` to avoid float64 GPU intrinsic limitations.

---

## ⚡︎ Performance

### MNIST Training Benchmark (15 Epochs, 105K Parameters)

> ⚠️ **These benchmarks reflect an earlier version of Tenmo.** The architectural work described above has added overhead on the CPU path. Updated benchmarks — including GPU training numbers — are in progress.

Training the same 4-layer MLP (784→128→32→10) on identical hardware:

| Platform | Device | Avg Epoch Time | Total Time | Final Test Acc |
|----------|--------|----------------|------------|----------------|
| **Tenmo** | **CPU (Mojo)** | **11.4s** | **171s** | **97.44%** |
| PyTorch | CPU | 14.5s | 218s | 98.26% |
| PyTorch | GPU (Tesla T4) | 15.2s | 227s | 97.87% |

**Key Observations (historical):**
- ⚡︎ **1.3× faster than PyTorch CPU** — Pure Mojo with SIMD optimization
- ⚡︎ **Competitive with PyTorch GPU for small models**
- 🎯︎ **97.4% accuracy** — Comparable to PyTorch with proper initialization
- 📉 **Zero Python overhead** — Runs entirely in compiled Mojo

*All runs were performed sequentially on the same system, batch_size=64. The MNIST example does not use BLAS — pure Mojo only.*

**Why was Tenmo competitive?**
- GPU overhead (kernel launch + data movement) dominates for small MNIST models
- Zero Python overhead
- SIMD-vectorized operations on contiguous buffers
- Zero-copy batch loading
- Compile-time specialization eliminates graph overhead in eval mode

---

## Quick Start

#### Tensor operation with backpropagation
```mojo
from testing import assert_true
from tenmo import Tensor

fn main() raises:
    # Defaults to DType.float32

    var a = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)

    # a is used in two places
    var b = a * 2  # ∂b/∂a = 2
    var c = a * 3  # ∂c/∂a = 3

    var d = b + c  # ∂d/∂a = ∂b/∂a + ∂c/∂a = 2 + 3 = 5

    var loss = d.sum()
    loss.backward()

    # Final grad: ∂d/∂a = [5, 5, 5]
    assert_true(a.grad().all_close(Tensor.d1([5.0, 5.0, 5.0])), "∂d/∂a = 5")
```

#### Broadcast matmul
```mojo
from tenmo import Tensor

fn main() raises:
    """Broadcasting (2,3) @ (1,3,4)."""
    var A = Tensor.ones(2, 3, requires_grad=True)
    var B = Tensor.ones(1, 3, 4)
    var result = A.matmul(B)
    var loss = result.sum()
    loss.backward()
    print("Broadcast matmul result")
    result.print()
    print("\nA's gradients")
    A.grad().print()
```

#### GPU tensor operations
```mojo
from tenmo import Tensor

fn main() raises:
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var a_gpu = a.to_gpu()

    var b_gpu = Tensor.full_gpu(Shape.of(2, 2), 2.0)
    var c_gpu = a_gpu * b_gpu

    var loss = c_gpu.sum()
    loss.backward()

    # Gradients flow back to original CPU tensor
    a.grad().print()
```

#### Solve XOR
```mojo
from tenmo import Tensor
from net import Sequential, Linear, Sigmoid, MSELoss
from sgd import SGD

fn main():
    """
    Classic non-linearly separable XOR problem requiring hidden layers.
    """
    alias dtype = DType.float64

    var X = Tensor[dtype].d2([[0, 0], [0, 1], [1, 0], [1, 1]])
    var y = Tensor[dtype].d2([[0], [1], [1], [0]])

    var model = Sequential[dtype]()
    model.append(
        Linear[dtype](2, 4, init_method="xavier").into(),
        Sigmoid[dtype]().into(),
        Linear[dtype](4, 1, init_method="xavier").into(),
        Sigmoid[dtype]().into(),
    )

    var criterion = MSELoss[dtype]()
    var optimizer = SGD(model.parameters(), lr=0.5, momentum=0.9)

    model.train()
    criterion.train()

    for epoch in range(200):
        var pred = model(X)
        var loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    var final_pred = model(X)
    var final_loss = criterion(final_pred, y)

    print("Final loss: ", final_loss.item())
    # Final loss: 0.028409039159250152
    # Accuracy: 100.0%
    # Success: Network learned XOR perfectly
```

---

## Running Tests
```bash
curl -fsSL https://pixi.sh/install.sh | sh
source ~/.bashrc
git clone https://github.com/ratulb/tenmo -b development
cd tenmo
pixi shell
./execute.sh all
```

---

## Why Tenmo?

**Full-stack visibility**: Every operation is implemented in pure Mojo — no hidden BLAS calls, no opaque kernels. You can read, understand, and optimize every line.

**Correctness first**: The recent architecture overhaul prioritizes memory safety (gradbox refcounting, ASAP-destruction safety), correctness of gradient flow through complex graphs, and GPU/CPU parity — before raw throughput.

**GPU-native**: Tensor operations, autograd, and kernel dispatch work on GPU. `DType.bool` GPU support with correct `uint8` internal representation throughout. Gradient flow across CPU/GPU boundaries.

**Mojo-native design**: Compile-time metaprogramming eliminates graph overhead in eval mode. Zero Python overhead. SIMD-vectorized kernels. Systems-level control that Python-based frameworks cannot match.

**Forward-looking**: The architecture is designed to support distributed training, model checkpointing, and production use — not just research exploration.

---

## 🏗️ Architecture

Tenmo's design prioritizes memory efficiency, correctness, and GPU support through careful separation of concerns.

### Core Types
```
Tensor[dtype: DType]
├── _id: UInt                     # Unique identity for graph traversal
├── buffer: NDBuffer              # Shape, strides, offset, data
├── requires_grad: Bool           # Gradient tracking flag
├── gradbox: UnsafePointer        # Refcounted — allocated only when needed
└── ancestors: Optional           # Lightweight ancestry for autograd graph

Ancestors[dtype: DType]
├── refs: List[Ancestor]          # Lightweight parent handles
└── backward_arg: BackwardFnArg   # Type-erased op argument — not Optional!

Ancestor[dtype: DType]
├── _id: UInt                     # Graph traversal
├── requires_grad: Bool           # Grad routing
├── gradbox: UnsafePointer        # Refcounted raw pointer — safe across lifetimes
├── layout: Layout                # Shape + strides + offset (deep copied once)
├── storage: Storage              # Buffer/DeviceState (refcount bump only)
└── parents: Optional[Ancestors]  # Recursive graph structure

Layout
├── shape: Shape                  # Tensor dimensions
├── strides: Strides              # Memory layout
├── offset: Int                   # View offset
└── _contiguous: Bool             # Cached contiguity flag

Storage[dtype: DType]
├── buffer: Buffer                # CPU data (ref-counted)
└── device_state: Optional        # GPU data (ArcPointer)

Gradbox[dtype: DType]
├── buffer: NDBuffer              # Contiguous gradient storage
└── _refcount: UnsafePointer      # Independent refcount — survives tensor ASAP destruction

NDBuffer[dtype: DType]
├── shape: Shape                  # Tensor dimensions
├── strides: Strides              # Memory layout
├── offset: Int                   # View offset
├── buffer: Buffer                # CPU data (ref-counted for views)
├── _contiguous: Bool             # Cached contiguity
└── device_state: Optional        # GPU device state
```

### Design Rationale

**Ancestor is not Tensor**
The old design stored full `Tensor` copies in ancestry — triggering recursive `__copyinit__` chains that allocated new gradboxes, copied ancestor lists, and copied `BackwardFnArg` heap blocks at every op. `Ancestor` carries only what backward actually needs: identity, grad routing, a refcounted gradbox pointer, layout metadata, and a storage refcount bump. The recursive explosion is gone.

**Gradbox refcounting is independent**
`Gradbox` has its own atomic refcount separate from `Buffer`. When Mojo ASAP-destroys an intermediate tensor, its `Tensor.__del__` decrements the gradbox refcount. If `Ancestor` copies in the graph still reference it, refcount stays above zero and the gradbox lives. Last owner frees. Views have independent gradboxes — zeroed after passing gradients to parents.

**`BackwardFnArg` lives on `Ancestors`, not `Tensor`**
Set atomically at `add_ancestry` time. Intermediate tensors carry zero backward function overhead. The argument is never redundantly copied through `Tensor` fields.

**`Layout` + `Storage` separation**
`Layout` owns all metadata — shape, strides, offset, contiguity. `Storage` owns all data — CPU `Buffer` or GPU `DeviceState`. `Ancestor` composes them directly, shedding the full `NDBuffer` operation machinery. `NDBuffer` remains the coordinator for indexing, arithmetic, and device transfer — but ancestry entries are lighter for it.

**Views are cheap**
`Buffer` is linear and becomes reference-counted when views are created. Views share storage without copying — zero-cost slicing.

**Compile-time graph elimination**
The `track_grad` compile-time parameter eliminates graph-building code entirely during evaluation — not a runtime branch, but a compile-time specialization that produces a pure forward pass binary.

---

## Tensor Capabilities

### Core Tensor Operations
- **Automatic differentiation** with dynamic computational graph
- **Broadcasting** for arithmetic operations (`+`, `-`, `*`, `/`)
- **SIMD-optimized** kernels with manual vectorization
- **Views and slicing** with zero-copy memory sharing
- **Comprehensive constructors**: `zeros`, `ones`, `rand`, `randn`, `arange`, `linspace`, `full`
- **Indexing**: Advanced slicing, `getitem`, `setitem`, and view operations
- **Reductions**: `sum`, `mean`, `max`, `min`, `argmax`, `argmin` (with axis support)
- **Reshaping**: `reshape`, `view`, `transpose`, `permute`, `squeeze`, `unsqueeze`, `flatten`
- **Statistical ops**: `variance`, `std` (numerically stable)
- **Comparison ops**: `eq`, `ne`, `all`, `any`, `all_close`
- **Unary ops**: `sqrt`, `abs`, `negate`, `relu`, `log`, `exp`, `tanh`, `sigmoid`
- **Bitwise/logical**: `~` (invert) for integer and bool tensors
- **Utility ops**: `concat`, `stack`, `vstack`, `hstack`, `chunk`, `tile`, `repeat`

### GPU Operations
- **Device transfer**: `to_gpu()`, `to_cpu()`, transparent CPU↔GPU
- **GPU arithmetic**: All binary and unary ops with GPU kernel dispatch
- **GPU reductions**: `sum`, `mean` with multi-axis support
- **GPU matmul**: Tiled GPU matmul kernel
- **GPU comparisons**: `eq`, `ne`, `gt`, `lt`, `ge`, `le` — including `DType.bool` with correct `uint8` internal representation
- **Gradient flow**: Backward passes run on GPU; gradients flow back to originating CPU tensors
- **Bool on GPU**: Full `DType.bool` support via `uint8` internal storage throughout all kernel paths

### Neural Network Components

**Layers:**
- `Linear` — Fully connected with Xavier/He initialization
- `ReLU`, `Sigmoid`, `Tanh` — Standard activations
- `Flatten` — Spatial to vector conversion
- `MaxPool2d` — 2D max pooling with stride/padding support
- `Conv2d` — 2D convolution (functional, optimization in progress)
- `Dropout` — Regularization layer
- `Sequential` — Layer composition container

**Loss Functions:**
- `MSELoss` — Mean squared error
- `BCELoss` — Binary cross-entropy
- `CrossEntropyLoss` — Multi-class classification

**Optimizers:**
- `SGD` — Stochastic gradient descent with momentum

**Training Utilities:**
- `.train()` / `.eval()` mode switching
- `DataLoader` with optimized batching
- `TensorDataset`, `NumpyDataset` wrappers

### BLAS Integration
Tenmo supports configurable BLAS backends. When `BLAS_PATH` is set, `LinearBLAS` dispatches to the configured library. The pure Mojo implementation is recommended — BLAS support is maintained for compatibility and benchmarking.

---

## 🔬 Advanced Features

### Compile-Time Optimization
```mojo
# Training: builds computational graph
model.train()
criterion.train()
var loss = criterion(pred, target)
loss.backward()

# Evaluation: zero overhead — no graph, no tracking, pure forward pass
model.eval()
criterion.eval()
var loss = criterion(pred, target)
```

### Memory-Efficient Data Loading
```mojo
var train_loader = train_dataset.into_loader(
    batch_size=64,
    shuffle=True,
    drop_last=False
)

for batch in train_loader:
    var pred = model(batch.features)
    var loss = criterion(pred, batch.labels)
    # ...
```

**DataLoader Optimization:**
- Pre-allocated batch buffers — zero allocations during iteration
- Bulk `memcpy` for sequential access (validation)
- Row-by-row `memcpy` for shuffled access (training)
- Built-in shuffling without data movement

---

## 📖 Examples

### Prerequisites
- Mojo 0.25.7.0+
- Python 3.10–3.12 (for NumPy interop in examples only)

### Setup
```bash
git clone https://github.com/ratulb/tenmo.git
cd tenmo

./example.sh xor
./example.sh mnist
./example.sh spiral
```

### XOR Problem
Binary classification demonstrating non-linear decision boundaries. Perfect separation in ~200 epochs with a 2-layer network.

### Spiral Dataset
Multi-class classification with complex decision boundaries — 99%+ accuracy on 2-rotation spirals.
```
Final Validation Loss: 0.022977224874494065
Final Validation Accuracy: 99.2%
Average time per epoch: 206.8ms
```

### MNIST Digit Classification
Full pipeline with data loading, batching, and validation.

**Architecture**: 784 → 128 → 32 → 10 | **15 epochs** | **batch_size=64** | **lr=0.01**

```
Epoch 1:  Loss: 0.711, Train: 76.96%, Test: 89.40%, Time: 11.7s
Epoch 5:  Loss: 0.158, Train: 95.40%, Test: 95.76%, Time: 11.0s
Epoch 10: Loss: 0.091, Train: 97.38%, Test: 97.13%, Time: 11.6s
Epoch 15: Loss: 0.059, Train: 98.38%, Test: 97.44%, Time: 11.7s
```

> These timings reflect a previous version of the codebase. Current timings will differ.

---

## 🚧 Roadmap

### In Progress
- [x] GPU tensor operations and kernel dispatch
- [x] GPU autograd — backward passes on GPU
- [x] `DType.bool` GPU support with correct `uint8` internals
- [x] Lightweight ancestry (`Ancestor` replacing fat `Tensor` copies)
- [x] Gradbox refcounting — safe across ASAP tensor destruction
- [x] `BackwardFnArg` off `Tensor` — lives on `Ancestors` directly
- [x] `Layout` + `Storage` as first-class types
- [ ] GPU training loop — gradient accumulation on GPU, explicit grad retrieval
- [ ] Linear layer on GPU
- [ ] DataLoader GPU data movement with prefetching

### Near Term
- [ ] More optimizers: Adam, RMSprop, AdamW
- [ ] Model checkpointing — serialization and loading
- [ ] Additional layers: BatchNorm, LayerNorm
- [ ] Updated GPU training benchmarks

### Medium Term
- [ ] `NDBuffer` internal refactor to `Layout` + `Storage` fields
- [ ] `Layout` consuming `IndexCalculator` / `ShapeBroadcaster` responsibilities
- [ ] Distributed training — multi-device and multi-node

### Long Term
- [ ] Attention mechanisms, transformer blocks
- [ ] Model Zoo — pre-trained architectures
- [ ] Production readiness — API stabilization

---

## 💡 Inspirations & Acknowledgments

Tenmo is built with a simple goal: **understand, control, and optimize the full ML stack from the ground up** — from memory layout to backpropagation to GPU kernel dispatch — while remaining lightweight and ergonomically familiar.

This project stands on the shoulders of giants:

- **[Mojo by Modular](https://www.modular.com/mojo)** — for proving that systems programming can wear Python's ergonomics, making SIMD and GPU programming genuinely accessible
- **[PyTorch](https://pytorch.org/)** — for its intuitive API design and elegant autograd architecture
- **[NumPy](https://numpy.org/)** — for defining the standard in array operations and broadcasting semantics
- **[Karpathy's llm.c](https://github.com/karpathy/llm.c)** — for championing radical transparency: understanding beats abstraction

---

## 🤝 Contributing

Tenmo welcomes contributions! Given the experimental nature of both the library and Mojo itself, we particularly value:

1. **Bug reports** with reproducible examples
2. **Performance optimizations** for existing operations
3. **Documentation** and examples
4. **Additional layers and operations**
5. **GPU training examples and benchmarks**

Please ensure any contributions maintain API consistency and include appropriate tests.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

**⭐ Building ML systems in Mojo? Star this repo and follow along as we push toward production-grade GPU training!**
