# Tenmo 🔥

**A high-performance tensor library and neural network framework built entirely in Mojo 🔥**

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

*All benchmarks performed on the same machine (Intel Xeon CPU, Tesla T4 GPU), batch_size=64*

**Why is Tenmo competitive?**

The performance comes from eliminating Python's interpreter overhead and data movement costs. For MNIST-scale models, the time spent in Python's runtime (scheduling, GIL, type checking) and GPU data transfers dominates actual compute. Tenmo's compiled Mojo code runs these operations directly on CPU with:
- SIMD-vectorized operations (manual loop unrolling)
- Zero-copy batch loading (bulk memcpy at ~0.03ms/batch)
- Cache-friendly memory layouts
- Compile-time specialization (eliminates graph overhead in eval mode)

For larger models where GPU compute advantages outweigh transfer costs, GPU acceleration becomes more beneficial. Tenmo's current focus is proving out the fundamentals on CPU before adding GPU support.

---

## ⚡︎ Quick Start

### XOR Problem - Neural Network in 15 Lines
```mojo
from tenmo import Tensor, Sequential, Linear, Sigmoid, MSELoss, SGD

var model = Sequential[DType.float32]()
model.append(
    Linear[DType.float32](2, 4).into(),
    Sigmoid[DType.float32]().into(),
    Linear[DType.float32](4, 1).into(),
    Sigmoid[DType.float32]().into()
)

var X = Tensor[DType.float32].d2([[0, 0], [0, 1], [1, 0], [1, 1]])
var y = Tensor[DType.float32].d2([[0], [1], [1], [0]])

var optimizer = SGD(model.parameters(), lr=0.5, momentum=0.9)
var criterion = MSELoss[DType.float32]()

for epoch in range(2000):
    model.train()
    criterion.train()
    var pred = model(X)
    var loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
print(model(X))  # Perfect XOR solution!
```

### Creating and Operating on Tensors
```mojo
from tenmo import Tensor

# Create tensors with various constructors
var a = Tensor[DType.float32].arange(6, requires_grad=True).reshape(2, 3)
a.print()
# [2D Tensor(2, 3), Type: float32, requires_grad: True]
#   [
#     [0.0, 1.0, 2.0],
#     [3.0, 4.0, 5.0],
#   ]

# Broadcasting and element-wise operations
var b = a + 1.0
var c = a * 2.0

# Automatic differentiation
var y = c.sum()
y.backward()
a.grad[].print()  # Gradients computed!

# Views and slicing (memory-efficient)
var v1 = a.view([5, 2], offset=2)   # Slice starting from a[2]
var v2 = v1.view([2, 5])            # Reshape
var v3 = v2.view([10])              # Flatten

# More constructors
var zeros = Tensor[DType.float32].zeros(3, 4)
var ones = Tensor[DType.float32].ones(2, 2)
var randn = Tensor[DType.float32].randn(5, 5)
var linspace = Tensor[DType.float32].linspace(0, 10, 100)
```

---

## ⚡︎ Features

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

Tenmo's design prioritizes memory efficiency and performance through careful separation of concerns:

### Memory Layout
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

**Key Design Decisions:**

1. **Tensor vs Gradbox**: Gradients don't need to be full-fledged Tensors, so they live in a lightweight `Gradbox` struct. This results in 70 % code reduction for Gradboxes!

2. **NDBuffer as Single Source of Truth**: Shape, strides, and offset logic is centralized in `NDBuffer`, which serves both `Tensor` and `Gradbox`. This ensures views, slicing, and broadcasting behave consistently.

3. **Reference-Counted Buffer**: The innermost `Buffer` becomes ref-counted when views are created, enabling zero-copy slicing while maintaining memory safety.

4. **Manual Broadcasting**: Broadcasting is explicitly handled in `NDBuffer` operations rather than being implicit, giving fine-grained control over memory layouts.

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
- Mojo 0.25.7.0 or newer
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
# Welford's algorithm with Kahan summation
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
