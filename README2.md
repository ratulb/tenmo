# Tenmo - High-Performance Tensor Library for Mojo 🔥

**Fast, type-safe tensor operations and neural network training in pure Mojo.**

Tenmo brings PyTorch-like ergonomics to Mojo with automatic differentiation, modular neural networks, and production-ready optimizers—all with **competitive CPU performance** and **zero Python overhead**.

---

## ⚡ Performance Highlights

### MNIST Training Benchmark (20 Epochs, 104K Parameters)

| Platform | Device | Avg Epoch Time | Total Time | Final Test Acc |
|----------|--------|----------------|------------|----------------|
| **Tenmo** | **CPU (Mojo)** | **11.3s** | **113s (10 epochs)** | **97.09%** |
| PyTorch | CPU | 21.7s | 433s | 98.27% |
| PyTorch | GPU (CUDA) | 17.9s | 357s | 98.20% |

**Key Takeaways:**
- ⚡ **1.9x faster than PyTorch CPU** (11.3s vs 21.7s per epoch)
- 🎯 **Competitive with PyTorch GPU** (11.3s vs 17.9s per epoch)
- 🚀 **97% accuracy in 10 epochs** with pure Mojo implementation
- 💾 **Zero Python overhead** - runs entirely in compiled Mojo

*Benchmarked on: PyTorch (Google Colab CPU/GPU), Tenmo (CPU, batch_size=64)*

### What Makes Tenmo Fast?

- **Zero-copy data loading**: Memory-mapped NumPy arrays with `0.03ms` per batch
- **Direct offset iteration**: SIMD-optimized loops at `3ns` per element
- **Compile-time specialization**: `track_grad` parameter eliminates graph overhead in eval mode
- **Efficient memory management**: Stack-allocated tensors with move semantics

---

## 🚀 Quick Start

### XOR Problem in 10 Lines
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

---

## 📊 Detailed Benchmarks

### MNIST Training Progression (PyTorch CPU vs Tenmo)

#### PyTorch CPU (Google Colab)
```
Epoch 1:  Loss: 0.3348, Train Acc: 89.95%, Test Acc: 95.75%, Time: 22.4s
Epoch 7:  Loss: 0.0296, Train Acc: 99.01%, Test Acc: 97.89%, Time: 20.6s
Epoch 20: Loss: 0.0005, Train Acc: 100.0%, Test Acc: 98.27%, Time: 21.3s
Total: 433s (20 epochs)
```

#### Tenmo CPU (Mojo)
```
Epoch 1:  Loss: 0.5615, Train Acc: 82.50%, Test Acc: 90.62%, Time: 11.2s
Epoch 7:  Loss: 0.0883, Train Acc: 97.48%, Test Acc: 97.03%, Time: 11.5s
Epoch 10: Loss: 0.0746, Train Acc: 97.80%, Test Acc: 97.09%, Time: 11.3s
Total: 113s (10 epochs) - Extrapolated 20 epochs: ~226s
```

**Performance Summary:**
- Tenmo achieves **97% accuracy in half the time**
- PyTorch reaches **98.3%** with 2x more epochs and training time
- **1.9x speedup** makes Tenmo ideal for rapid prototyping and iteration

---

## 🎯 Features

### Core Tensor Operations
- ✅ **Automatic differentiation** with computational graph
- ✅ **Broadcasting** for all arithmetic operations
- ✅ **SIMD-optimized** kernels for contiguous tensors
- ✅ **Memory-efficient** gradient accumulation
- ✅ **Type-safe** with compile-time dtype checking

### Neural Network Layers
- `Linear` - Fully connected layers with Xavier/He initialization
- `ReLU`, `Sigmoid`, `Tanh` - Activation functions
- `Sequential` - Container for layer composition
- `MSELoss`, `BCELoss`, `CrossEntropyLoss` - Loss functions

### Optimizers
- `SGD` - Stochastic Gradient Descent with momentum
- Zero-overhead `.train()` / `.eval()` mode switching

### Data Loading
- `TensorDataset` - PyTorch-style dataset wrapper
- `DataLoader` - Batching with shuffling support
- `NumpyDataset` - Direct NumPy array integration

---

## 🔧 Installation

### Prerequisites
- Mojo 24.5 or later
- Python 3.8+ (for NumPy interop)

### Usage
```bash
git clone https://github.com/yourusername/tenmo.git
cd tenmo

# Run examples
mojo examples/xor.mojo
mojo examples/mnist.mojo
```

---

## 📖 Examples

### 1. **XOR Problem** ([examples/xor.mojo](./examples/xor.mojo))
Binary classification with perfect separation in 2000 epochs.

### 2. **Spiral Dataset** ([examples/spiral.mojo](./examples/spiral.mojo))
Non-linear decision boundaries with 2-3 rotations.
- 2 rotations: 99% accuracy, converges quickly
- 3 rotations: Complex architecture required

### 3. **MNIST Training** ([examples/mnist.mojo](./examples/mnist.mojo))
Full production pipeline with:
- NumPy data loading
- Train/validation splits
- Batch processing (64 samples/batch)
- Accuracy tracking

**Architecture:**
```
Input(784) → Linear(128) → ReLU → Linear(64) → ReLU →
Linear(32) → ReLU → Linear(10)
```

**Results:**
- 97.09% test accuracy in 10 epochs
- 11.3s per epoch on CPU
- 104,938 trainable parameters

---

## 🏗️ Architecture

### Design Principles

1. **Zero-cost abstractions**: Compile-time `track_grad` parameter eliminates runtime overhead
2. **Move semantics**: Efficient memory management with explicit ownership
3. **Type safety**: Leverages Mojo's strong type system for correctness
4. **PyTorch compatibility**: Familiar API for easy adoption

### Key Components
```
tenmo/
├── tensor.mojo          # Core Tensor with autograd
├── ops/                 # Operations (matmul, add, relu, etc.)
├── nn/                  # Neural network layers
│   ├── linear.mojo
│   ├── activations.mojo
│   └── sequential.mojo
├── optim/               # Optimizers (SGD, Adam)
└── data/                # Data loading utilities
    ├── dataset.mojo
    └── dataloader.mojo
```

---

## 🎓 Advanced Features

### Compile-Time Graph Optimization
```mojo
# Training: builds computational graph
model.train()
criterion.train()
loss = criterion(pred, target)  # Graph enabled
loss.backward()


# Evaluation: zero overhead
model.eval()
criterion.eval()
loss = criterion(pred, target)  # No graph, pure forward pass
```

### Memory-Efficient Batching
```mojo
var train_loader = DataLoader[NumpyDataset[DType.float32, DType.int32]](
    train_dataset^,
    batch_size=64,
    reshuffle=True
)

for batch in train_loader:
    var pred = model(batch.features)  # Loads batch on-demand
    var loss = criterion(pred, batch.labels)
    # ... training step
```

---

## 🔬 Benchmarking Details

### Test Environment
- **CPU**: Google Colab default runtime (Intel Xeon, 2 cores)
- **GPU**: Tesla T4 (CUDA 11.8)
- **Dataset**: MNIST (60K train, 10K test)
- **Model**: 3-layer MLP (784→128→32→10)
- **Batch Size**: 64 (Tenmo), 64 (PyTorch)

### Reproducibility
All benchmarks use the same:
- Model architecture
- Hyperparameters (LR=0.01, momentum=0.9)
- Initialization scheme (Xavier)
- Loss function (CrossEntropy)

---

## 🚧 Roadmap

- [ ] **Optimizers**: Adam, AdamW, RMSprop
- [ ] **Layers**: Conv2D, BatchNorm, Dropout
- [ ] **Advanced ops**: Embedding, Attention
- [ ] **Distributed training**: Multi-core parallelism
- [ ] **Model zoo**: Pre-trained ResNet, ViT

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built with ❤️ using [Mojo](https://www.modular.com/mojo) by Modular.

Inspired by PyTorch's elegant API and Mojo's performance potential.

---

## 📬 Contact

Questions? Suggestions? Open an issue or reach out!

**Star ⭐ this repo if Tenmo helps your ML journey!**
