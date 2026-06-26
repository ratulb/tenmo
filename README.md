# Tenmo
## Build Status

[![Mojo Tests](https://github.com/ratulb/tenmo/actions/workflows/test.yml/badge.svg)](https://github.com/ratulb/tenmo/actions/workflows/test.yml)
![Last Commit](https://img.shields.io/github/last-commit/ratulb/tenmo/development)
![License](https://img.shields.io/github/license/ratulb/tenmo)
![Language](https://img.shields.io/badge/language-Mojo%20🔥-orange)
![Open Issues](https://img.shields.io/github/issues/ratulb/tenmo)

**A lean tensor library and neural network framework built entirely in Mojo 🔥**

Tenmo brings modern, ergonomic ML abstractions to Mojo with automatic differentiation, modular neural networks, and end-to-end training pipelines—aiming for performance competitive with modern ML systems.

> ⚠️ **Development Status**: Tenmo is actively evolving alongside Mojo itself. The API is subject to change as we incorporate improvements from the Mojo ecosystem. Not production-ready yet, but excellent for learning, experimentation, and systems-level exploration.


---

## ⚡︎ Performance

### MNIST Training Benchmark (15 Epochs, 105K Parameters)

Training the same 4-layer MLP (784→128→32→10) on identical hardware, all runs sequential:

| Platform | Device | Avg Epoch Time | Total Time | Final Val Acc |
|----------|--------|----------------|------------|---------------|
| **Tenmo** | **CPU (Mojo)** | **5.5s** | **82.3s** | **98.14%** |
| **Tenmo** | **GPU (Mojo)** | **6.0s** | **90.1s** | **98.00%** |
| PyTorch | GPU (CUDA) | 14.5s | 217.2s | 98.18% |
| PyTorch | CPU | 15.4s | 231.5s | 98.12% |

**Key Observations:**
- ⚡︎ **2.8× faster than PyTorch CPU** & **2.4× faster than PyTorch GPU** — Pure Mojo SIMD on CPU, native kernel compilation on GPU
- 🎯︎ **98.14% validation accuracy** — Matches PyTorch precision on identical hardware
- 💡 **CPU beats GPU for this model** — At 104K params the SIMD CPU kernels saturate the machine before GPU launch overhead pays off
- 📉 **Zero Python overhead** — Runs entirely in compiled Mojo

*Batch_size=64. The MNIST example does not use BLAS — pure Mojo end-to-end.*

**Training Progression** (Tenmo CPU):
```
Epoch 1:  Loss: 0.323, Train: 90.18%, Val: 95.19%, Time: 5.40s
Epoch 5:  Loss: 0.051, Train: 98.46%, Val: 97.28%, Time: 5.49s
Epoch 10: Loss: 0.018, Train: 99.48%, Val: 97.60%, Time: 5.49s
Epoch 15: Loss: 0.006, Train: 99.93%, Val: 98.14%, Time: 5.47s
```

**Why is Tenmo competitive?**
- Zero Python overhead — no interpreter, no dispatch
- SIMD-vectorized operations on contiguous buffers
- Zero-copy batch loading
- Compile-time specialization eliminates graph overhead in eval mode
- GPU kernels compile directly from the same Mojo source — no CUDA/C++ bridge

---

## What's New

The library has undergone significant architectural work. The changes prioritize correctness, safety, and GPU support.

### Major Recent Work

**Backward system redesign** — moved from stateful handler instances to pure static methods with a type-erased `BackwardFnArg`. Dispatch is now a direct integer-tag jump table. No variant extraction, no handler instances, no redundant copies.

**Ancestry redesign** — `Ancestors` no longer stores full `Tensor` copies. Each ancestor is now a lightweight `Ancestor` handle carrying only what backward needs: an id, `requires_grad`, a refcounted gradbox pointer, and a shared `NDBuffer`. The recursive deep-copy explosion on every `add_ancestry` call is gone.

**GPU support** — tensor operations, backward passes, and gradient flow now work on GPU. `DType.bool` is handled correctly via internal `uint8` storage throughout kernels.

> 📖 **Deep Dive**: For a complete explanation of forward and backward pass mechanics, see [`README_AUTOGRAD.md`](README_AUTOGRAD.md).

---

## Quick Start

#### Tensor operation with backpropagation
```mojo
from std.testing import assert_true
from tenmo.tensor import Tensor

def main() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)

    # a is used in two places
    var b = a * 2  # ∂b/∂a = 2
    var c = a * 3  # ∂c/∂a = 3

    var d = b + c  # ∂d/∂a = ∂b/∂a + ∂c/∂a = 2 + 3 = 5

    d.backward()

    # Final grad: ∂d/∂a = [5, 5, 5]
    assert_true(a.grad().all_close(Tensor.d1([5.0, 5.0, 5.0])), "∂d/∂a = 5")
```
#### Broadcast matmul
```mojo

from tenmo.tensor import Tensor

def main() raises:
    """Broadcasting (2,3) @ (1,3,4)."""
    comptime dtype = DType.float32
    var A = Tensor[dtype].ones(2, 3, requires_grad=True)
    var B = Tensor[dtype].ones(1, 3, 4)
    var result = A.matmul(B)
    result.backward()
    print(" Broadcast matmul result")
    result.print()
    print(" \nA's gradients")
    A.grad().print()

 Broadcast matmul result

 [3D Tensor(1, 2, 4), strides: (8, 4, 1), offset: 0, Type: float32, requires_grad: True]
  [
    [
      [3.0, 3.0, 3.0, 3.0],
      [3.0, 3.0, 3.0, 3.0]
    ]
  ]
A's gradients

 [2D Gradbox(2, 3), Type: float32, Shared : True, Strides : (3, 1), Offset : 0]
  [
    [4.0, 4.0, 4.0],
    [4.0, 4.0, 4.0]
  ]

```
#### Solve XOR
```mojo
from tenmo.tensor import Tensor
from tenmo.net import Sequential, Linear, Sigmoid, MSELoss
from tenmo.optim import SGD

def main() raises:
    """
    Classic non-linearly separable XOR problem requiring hidden layers.
    """
    comptime dtype = DType.float64

    # XOR truth table
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

    # Final evaluation
    model.eval()
    var final_pred = model(X)
    var final_loss = criterion(final_pred, y)

    var correct = 0
    var total_error = 0.0
    for i in range(4):
        var pred_class = 1 if final_pred[i, 0] > 0.5 else 0
        var true_class = Int(y[i, 0])
        if pred_class == true_class:
            correct += 1
        total_error += abs(final_pred[i, 0] - y[i, 0])

    print("Final loss: ", final_loss.item())
    print("Accuracy: ", 100.0 * correct / 4, "%")

    if correct == 4:
        print("Success: Network learned XOR perfectly")
    else:
        print("Failed: Network did not learn XOR")


    Final loss: 0.028409039159250152
    Accuracy: 100.0%

    Success: Network learned XOR perfectly
```
## Running Tests
```bash
curl -fsSL https://pixi.sh/install.sh | sh
source ~/.bashrc
git clone https://github.com/ratulb/tenmo -b main
cd tenmo
pixi shell
./execute.sh all
```

---

## Why Tenmo?

**Performance without compromise**: 2.8× faster than PyTorch CPU and 2.4× faster than PyTorch GPU on MNIST, with zero Python overhead and full SIMD optimization.

**Transparency you can trust**: Every operation is implemented in pure Mojo — no hidden BLAS calls, no opaque kernels. Perfect for learning and optimization.

**Forward-looking design**: Competitive with PyTorch today; GPU support already benchmarks faster than PyTorch GPU on the same hardware.

**Mojo-native**: Leverages compile-time metaprogramming, zero-cost abstractions, and systems-level control that Python-based frameworks can't match.

---
## Tensor Capabilities

Tenmo provides a broad set of tensor operations. Below is a representative (not exhaustive) selection:

### Core Tensor Operations
- **Automatic differentiation** with dynamic computational graph
- **Broadcasting** for arithmetic operations (`+`, `-`, `*`, `/`)
- **SIMD-optimized** kernels with manual vectorization
- **Views and slicing** with zero-copy memory sharing
- **Comprehensive constructors**: `zeros`, `ones`, `rand`, `randn`, `arange`, `linspace`, `full`
- **Indexing**: Advanced `slicing`, `getitem`, `setitem`, and view operations
- **Reductions**: `sum`, `mean`, `max`, `min`, `argmax`, `argmin` (with axis support)
- **Reshaping**: `reshape`, `view`, `transpose`, `permute`, `squeeze`, `unsqueeze`, `flatten`
- **Statistical ops**: `variance`, `std` (numerically stable algorithms)
- **Comparison ops**: `eq`, `ne`, `all`, `any`, `all_close`
- **Utility ops**: `concat`, `stack`, `vstack`, `hstack`, `chunk`, `tile`, `repeat`

### Neural Network Components

**Layers:**
- `Linear` - Fully connected with Xavier/He initialization
- `ReLU`, `Sigmoid`, `Tanh` - Standard activations
- `Flatten` - Spatial to vector conversion
- `MaxPool2d` - 2D max pooling with stride/padding support
- `Conv2d` - 2D convolution
- `Dropout` - Regularization layer
- `Sequential` - Layer composition container

**Loss Functions:**
- `MSELoss` - Mean squared error
- `BCELoss` - Binary cross-entropy
- `CrossEntropyLoss` - Multi-class classification

**Optimizers:**
- `SGD` - Stochastic gradient descent with momentum

**Training Utilities:**
- `.train()` / `.eval()` mode switching
- `DataLoader` with optimized batching
- `TensorDataset`, `NumpyDataset` wrappers

### BLAS Integration

Tenmo supports configurable BLAS backends for linear algebra operations. Use `SequentialBLAS` with `LinearBLAS` layers for automatic BLAS acceleration:

- **Auto-profiling**: `LinearBLAS` automatically profiles native Mojo vs BLAS matmul at runtime and selects the faster path
- **Runtime dispatch**: No compile-time configuration needed — profiling happens on first forward calls
- **Gradient-aware**: Full backward pass support through BLAS for training

```mojo
var model = SequentialBLAS[dtype]()
model.append(LinearBLAS[dtype](784, 128, profile_samples=10).into())
```

### Installation

**Option 1 — System OpenBLAS (recommended, works out of the box):**
```bash
sudo apt-get update && sudo apt-get install -y libopenblas-dev
```
Installs to `/lib/x86_64-linux-gnu/libopenblas.so.0` — the default path Tenmo looks for, no `-D` flag needed.

**Option 2 — Pixi-managed OpenBLAS (conda):**
```bash
pixi add openblas
# Then pass the path explicitly:
mojo -I . -D BLAS_PATH=$(find $CONDA_PREFIX/lib -name "libopenblas.so" | head -1) ...
```

---

## 🏗️ Architecture

Tenmo's design prioritizes memory efficiency and performance through careful separation of concerns - organized around a few tightly scoped core building blocks:

### Core Types
```
Tensor[dtype: DType]
├── _id: UInt                              # Unique identifier
├── buffer: NDBuffer[dtype]                # Data + layout (shape/strides/offset)
├── requires_grad: Bool                    # Gradient tracking flag
├── gradbox: Optional[Gradbox[dtype]]      # Gradient storage (only if requires_grad=True)
└── ancestors: Optional[Ancestors]         # Computation graph parents

Gradbox[dtype: DType]
├── _ndb_ptr: Optional[UnsafePointer[NDBuffer]]   # Heap NDBuffer (combined alloc)
└── _refcount: Optional[UnsafePointer[Atomic]]     # Atomic refcount (combined alloc)

Ancestor[dtype: DType]
├── _id: UInt                              # Graph traversal key
├── requires_grad: Bool                    # Skip gradient update if False
├── gradbox: Optional[Gradbox[dtype]]      # Gradient storage (inline via Optional)
├── ndb: Optional[NDBuffer[dtype]]         # Data+layout (None unless needs_parent_data=True)
└── parents: Optional[Ancestors[dtype]]    # Recursive ancestry chain

NDBuffer[dtype: DType]
├── shape: Shape                           # Tensor dimensions
├── strides: Strides                       # Memory layout
├── offset: Int                            # View offset
├── _contiguous: Bool                      # Cached contiguous flag
├── buffer: Buffer[dtype]                  # CPU data
└── device_state: Optional[DeviceState]    # GPU storage
```

### Design Rationale

**Gradbox is not a Tensor**
Gradients don't need the full Tensor API. A `Gradbox` encapsulates only an `NDBuffer`, keeping gradient storage minimal and explicit — **70% less code than full Tensors**. Gradbox buffers are always ref-counted — gradients land in the right place regardless of how many tensor copies or views exist.

**`Tensor.grad()` returns an independent deep copy**
Calling `A.grad()` returns a detached `Gradbox` with its own data via `Gradbox.detach()`, which deep-copies the underlying buffer (CPU: `memcpy`, GPU: `enqueue_copy_to`). The tensor's internal Gradbox is unaffected by subsequent `zero_grad()` or `.backward()` calls on the returned copy — safe to snapshot gradients mid-training.

**Ancestors is not a Tensor**
The autograd graph no longer stores full `Tensor` copies. An `Ancestor` handle carries only what backward needs: an id, `requires_grad` flag, a refcounted gradbox pointer, and a shared `NDBuffer`(if backward needs). This eliminates the recursive deep-copy explosion on every `add_ancestry` call.

**NDBuffer as Single Source of Truth**
Shape, strides, and offset logic is centralized in `NDBuffer`, which serves both `Tensor` and `Gradbox`. This ensures views, slicing, and broadcasting behave consistently across the system.

**Views are cheap**
`Buffer` is linear and becomes reference-counted when views are created. Views share storage without copying — which provides zero-cost slicing.

**Backpropagation**
The gradbox pointer is the single link between the autograd graph and gradient storage. It is refcounted independently of tensor lifetime — gradients flow to the right place regardless of whether the original tensor is still alive.

**Minimal Module System**
Tenmo includes a minimal neural network module system: `Sequential`, `Linear`, `LinearBLAS`, `ReLU`, `Sigmoid`, `Tanh`, `Dropout`, `Conv2d`, `Flatten`, `MaxPool2d`, and loss functions. Intentionally minimal — build on top as needed.

This architecture keeps the system **explicit, predictable, and close to the metal**.

---

## 📖 Examples

### Prerequisites
- Mojo 1.0.0b1 (linux-64 only)
- Python 3.10-3.14 (for NumPy interop in examples)

### Setup
```bash
git clone https://github.com/ratulb/tenmo.git
cd tenmo

# Run examples
./example.sh xor
./example.sh mnist
./example.sh spiral
```

### 1. XOR Problem
Binary classification demonstrating non-linear decision boundaries. Perfect separation achieved in ~2000 epochs with a simple 2-layer network.
```bash
./example.sh xor

Epoch 1999 predictions:
  (0,0) → 0 | 0.0107 (err: 0.0107)
  (0,1) → 1 | 0.9845 (err: 0.0154)
  (1,0) → 1 | 0.9880 (err: 0.0119)
  (1,1) → 0 | 0.0166 (err: 0.0166)

```

### 2. Spiral Dataset
Multi-class classification with complex decision boundaries:
- 2 rotations: 99% accuracy
- 3 rotations: Requires deeper architecture
```bash
./example.sh spiral
Final Validation Loss: 0.022977224874494065
Final Validation Accuracy: 99.2 %

================================================================================
Performance Summary
================================================================================
Total epochs: 3000
Total batches processed: 96000
Average time per batch: 6.463713670645833 ms
Average time per epoch: 206.83883746066667 ms

✓ Training successful! Model learned the spiral pattern.
================================================================================
```

### 3. MNIST Digit Classification

Full training pipeline with data loading, batching, and validation:
```bash
./example.sh mnist          # CPU
./example.sh mnist_gpu      # GPU (requires CUDA-capable device)
```

**Architecture**: 784 → 128 → 32 → 10
**Training**: 15 epochs, batch_size=64, lr=0.01, momentum=0.9
**Results**: 98.14% validation accuracy in **82 seconds** (CPU) / **90 seconds** (GPU)

See the [Performance section](#-performance) for full CPU & GPU benchmarks vs PyTorch.
All core tensor operations are in pure Mojo with no external dependencies. NumPy is only used for loading MNIST data in the examples.

---

## 🔬 Advanced Features

### Compile-Time Optimization

The `track_grad` compile-time parameter eliminates graph overhead during evaluation:
```mojo
# Training: builds computational graph
model.train()
criterion.train()
loss = criterion(pred, target)  # Graph tracking enabled
loss.backward()

# Evaluation: zero overhead
model.eval()
criterion.eval()
loss = criterion(pred, target)  # Pure forward pass, no graph, utilizes Mojo's compile-time metaprogramming that eliminates generation of grad tracking code
```

### Memory-Efficient Data Loading
```mojo
var train_loader = train_dataset.into_loader(
    batch_size=64,
    shuffle=True,
    drop_last=False
)

# Pre-allocated batch buffers reused across iterations
for batch in train_loader:
    var pred = model(batch.features)
    var loss = criterion(pred, batch.labels)
    # ... training step
```

**DataLoader Optimization:**
- Pre-allocated batch buffers (zero allocations during iteration)
- Bulk `memcpy` for sequential access (validation: single copy per batch)
- Row-by-row `memcpy` for shuffled access (training: 64 copies per batch)
- Built-in shuffling without data movement
---

## 🚧 Roadmap

### Near Term
- [ ] More Optimizers: Adam, RMSprop, AdamW
- [ ] Aggressive performance optimization of core components
- [ ] Checkpointing: Model serialization and loading
- [ ] Additional Layers: BatchNorm, LayerNorm
- [ ] GPU transfer optimization: pinned memory, async transfers, stream pipelining
- [ ] GPU synchronization: explicit stream management and async kernel launch

### Medium Term
- [ ] Transparent GPU Support: Unified CPU/GPU tensor operations
- [ ] Zero-copy ancestry tracking: eliminate remaining deep copies on forward pass

### Long Term
- [ ] Distributed Training: Multi-device and multi-node support
- [ ] Advanced Operations: Attention mechanisms, transformer blocks
- [ ] Model Zoo: Pre-trained models and architectures
- [ ] Production Readiness: API stabilization and comprehensive testing

---

## 💡 Inspirations & Acknowledgments

Tenmo is built with a simple goal: **understand, control, and optimize the full ML stack from the ground up** — from memory layout to backpropagation — while remaining lightweight and ergonomically familiar.

This project stands on the shoulders of giants:

- **[Mojo by Modular](https://www.modular.com/mojo)** for proving that systems programming can wear Python's ergonomics, making SIMD and GPU programming genuinely accessible
- **[PyTorch](https://pytorch.org/)** for its intuitive API design and elegant autograd architecture that made deep learning feel natural
- **[NumPy](https://numpy.org/)** for defining the standard in array operations and broadcasting semantics
- **[Karpathy's llm.c](https://github.com/karpathy/llm.c)** for championing radical transparency: showing that understanding beats abstraction

---

## 🤝 Contributing

Tenmo welcomes contributions! Given the experimental nature of both the library and Mojo itself, we particularly value:

1. **Bug reports** with reproducible examples.
2. **Performance optimizations** for existing operations.
3. **Documentation** and examples.
4. **Additional layers and operations**.

Please ensure any contributions maintain API consistency and include appropriate tests.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**⭐ Building ML systems in Mojo? Star this repo to follow along as we push toward production-grade performance!**
