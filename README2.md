# Tenmo 🔥

**A high-performance, lean tensor library and neural network framework built entirely in Mojo 🔥**

Tenmo brings modern, ergonomic ML abstractions to Mojo with automatic differentiation, modular neural networks, and end-to-end training pipelines—aiming for performance competitive with modern ML systems.

> ⚠️ **Development Status**: Tenmo is actively evolving alongside Mojo itself. The API is subject to change as we incorporate improvements from the Mojo ecosystem. Not recommended for production use yet, but great for learning, experimentation, and systems-level exploration.

---

## ⚡︎ Performance

### MNIST Training Benchmark (15 Epochs, 105K Parameters)

Training the same 4-layer MLP (784→128→32→10) on identical hardware:

| Platform | Device | Avg Epoch Time | Total Time | Final Test Acc |
|----------|--------|----------------|------------|----------------|
| **Tenmo** | **CPU (Mojo)** | **11.4s** | **171s** | **97.44%** |
| PyTorch | CPU | 14.5s | 218s | 98.26% |
| PyTorch | GPU (Tesla T4) | 15.2s | 227s | 97.87% |

**Key Observations:**
- ⚡︎ **1.3x faster than PyTorch CPU** - Pure Mojo implementation with SIMD optimization
- ⚡︎ **1.3x faster than PyTorch GPU** - Efficient CPU utilization outperforms GPU on small models
- 🎯︎ **97.4% accuracy** - Comparable to PyTorch with proper initialization
- 📉 **Zero Python overhead** - Runs entirely in compiled Mojo

*All runs were performed sequentially on the same system, batch_size=64*

**Why is Tenmo competitive?**
- GPU overhead (kernel launch + data movement) dominates for small MNIST models
- Tenmo benefits from:
  - Zero Python overhead 
  - SIMD-vectorized operations on contiguous buffers
  - Zero-copy batch loading
  - Compile-time specialization (eliminates graph overhead in eval mode)

 The MNIST example does not use BLAS - it executes pure mojo code.

For larger models where GPU compute advantages outweigh transfer costs, GPU acceleration becomes more beneficial. Tenmo's current focus is proving out the fundamentals on CPU before adding GPU support.

---

## Quick Start

#### Tensor operation with backpropgation
```mojo
from testing import assert_true
from tenmo import Tensor

fn main() raises:
    # Default DType is DType.float32

    var a = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)

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

from tenmo import Tensor

fn main() raises:
    """Broadcasting (2,3) @ (1,3,4)."""
    var A = Tensor.ones(2, 3, requires_grad=True)
    var B = Tensor.ones(1, 3, 4)
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

 [2D Gradbox(2, 3), Type: float32, Shared : False, Strides : (3, 1), Offset : 0]
  [
    [4.0, 4.0, 4.0],
    [4.0, 4.0, 4.0]
  ]  

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
---

## Features

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
- `Conv2d` - 2D convolution (functional, optimization in progress)
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
Tenmo supports configurable BLAS backends for linear algebra operations. When the `BLAS_PATH` environment variable is specified, the `LinearBLAS` layer will dispatch operations to the configured BLAS library. This remains optional, as Tenmo's pure Mojo implementation provides competitive performance and is the primary focus.

---

## 🏗️ Architecture

Tenmo's design prioritizes memory efficiency and performance through careful separation of concerns - organized around a small number of tightly scoped core building blocks:

### Core Types (Conceptual)
```
Tensor[dtype: DType]
├── buffer: NDBuffer          # Single source of truth for shape/strides/offset
├── requires_grad: Bool       # Gradient tracking flag
├── gradbox: UnsafePointer    # Gradients (only allocated when needed)
├── ancestors: Optional       # Parent tensors in computation graph
└── backwardFn: Optional      # Backward pass function

Gradbox[dtype: DType]
└── buffer: NDBuffer          # Contiguous gradient storage

NDBuffer[dtype: DType]
├── buffer: Buffer            # Underlying data (ref-counted for views)
├── shape: Shape              # Tensor dimensions
├── strides: Strides          # Memory layout
└── offset: Int               # View offset into parent buffer

Buffer[dtype: DType]
└── UnsafePointer[Scalar]     # SIMD-capable linear storage
```

### Design Rationale

**Gradbox is not a Tensor**  
Gradients don't need the full Tensor API. A `Gradbox` encapsulates only an `NDBuffer`, keeping gradient storage minimal and explicit — **70% less code than full Tensors**.

**NDBuffer as Single Source of Truth**  
Shape, strides, and offset logic is centralized in `NDBuffer`, which serves both `Tensor` and `Gradbox`. This ensures views, slicing, and broadcasting behave consistently across the system.

**Views are cheap**  
`Buffer` is linear and becomes reference-counted when views are created. Views share storage without copying — which provides zero-cost slicing.

**Backpropagation**  
The innermost linear buffer is shared (ref-counted) between user tensors and the autograd engine, ensuring gradients flow correctly through the computation graph.

This architecture keeps the system **explicit, predictable, and close to the metal**.

---

## 📖 Examples

### 1. XOR Problem
Binary classification demonstrating non-linear decision boundaries. Perfect separation achieved in ~2000 epochs with a simple 2-layer network.
```bash
./example.sh xor
```

### 2. Spiral Dataset
Multi-class classification with complex decision boundaries:
- 2 rotations: 99% accuracy, quick convergence
- 3 rotations: Requires deeper architecture
```bash
./example.sh spiral
```

### 3. MNIST Digit Classification

Full training pipeline with data loading, batching, and validation:
```bash
./example.sh mnist
```

**Architecture**: 784 → 128 → 32 → 10  
**Training**: 15 epochs, batch_size=64, lr=0.01, momentum=0.9  
**Results**: 97.44% test accuracy in 171 seconds

**Training Progression** (Tenmo on CPU):
```
Epoch 1:  Loss: 0.711, Train: 76.96%, Test: 89.40%, Time: 11.7s
Epoch 5:  Loss: 0.158, Train: 95.40%, Test: 95.76%, Time: 11.0s
Epoch 10: Loss: 0.091, Train: 97.38%, Test: 97.13%, Time: 11.6s
Epoch 15: Loss: 0.059, Train: 98.38%, Test: 97.44%, Time: 11.7s
```

---

## 🔧 Installation

### Prerequisites
- Mojo 0.25.7.0
- Python 3.10-3.12 (for NumPy interop in examples)

### Setup
```bash
git clone https://github.com/ratulb/tenmo.git
cd tenmo

# Run examples
./example.sh xor
./example.sh mnist
./example.sh spiral
```

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
loss = criterion(pred, target)  # Pure forward pass, no graph, utilizes Mojo's compile-time metaprogramming that eliminates generation of grad tracking code path 
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

### Numerically Stable Statistics
```mojo
var mean = tensor.mean()
var std = tensor.std()
var variance = tensor.variance()
```

---

## 🚧 Roadmap

### Near Term
- [ ] More Optimizers: Adam, RMSprop, AdamW
- [ ] Aggressive performance optimization of core components
- [ ] Checkpointing: Model serialization and loading
- [ ] Additional Layers: BatchNorm, LayerNorm
      
### Medium Term
- [ ] Transparent GPU Support: Unified CPU/GPU tensor operations
- [ ] Investigate memory-efficient ancestry tracking for autodiff.

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
