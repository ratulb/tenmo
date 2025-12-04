# 🧠 Tenmo: High-Performance Tensors in Mojo

A from-scratch **Tensor library** built in [Mojo](https://modular.com/mojo), featuring automatic differentiation, neural network training, and SIMD-optimized operations. Built for learning, optimized for performance.

## 🚀 Quick Start

```mojo
from tenmo import Tensor

# Create a tensor with gradient tracking
var t = Tensor.arange(6, requires_grad=True).reshape(2, 3)
t.print()

# Elementwise operations with broadcasting
var t2 = t + 1.0
t2.print()

# Automatic differentiation
var y = t * 2.0
y.backward()
t.grad().print()
📊 Core Features
Feature	Status	Description
N-dimensional Tensor	✅ Full	Support for float64, float32, bool types
Automatic Differentiation	✅ Full	Graph-based backpropagation
Elementwise Operations	✅ Full	+, -, *, /, pow, etc.
SIMD Vectorization	✅ Full	Optimized CPU performance
Slicing & Views	✅ Full	Zero-copy tensor views
Broadcasting	✅ Full	Shape-compatible operations
Neural Network Module	✅ Full	Linear, ReLU, Sigmoid, optimizers
Data Loading	✅ Full	TensorDataset, DataLoader for batching
🧠 Neural Network Examples
XOR Problem (Classic Non-linear Learning)
mojo
from tenmo import Tensor
from net import Sequential, Linear, Sigmoid, SGD, MSELoss

fn xor_classification():
    var X = Tensor[DType.float64].d2([[0,0],[0,1],[1,0],[1,1]])
    var y = Tensor[DType.float64].d2([[0],[1],[1],[0]])

    var model = Sequential[DType.float64]()
    model.append(
        Linear(2, 4, xavier=True).into(),
        Sigmoid().into(),
        Linear(4, 1, xavier=True).into(),
        Sigmoid().into()
    )

    var optimizer = SGD(model.parameters(), lr=0.5, momentum=0.9)

    for epoch in range(2000):
        var pred = model(X)
        var loss = pred.mse(y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Achieves 100% accuracy in ~1 second
Output:

text
Training time: 1.1964s
Final loss: 0.000257
Accuracy: 100.0%
✓ Success: Network learned XOR perfectly
Spiral Dataset (Complex Non-linear Classification)
mojo
# 2D spiral classification with 2,817 parameters
# 500 training samples, 250 validation
# 3 hidden layers: 64 → 32 → 16 → 1

Epoch     0 | Train Loss: 0.70469 Acc: 50.4% | Val Loss: 0.68524 Acc: 54.0%
Epoch  1000 | Train Loss: 0.02551 Acc: 98.9% | Val Loss: 0.02291 Acc: 99.4%
Epoch  4999 | Train Loss: 0.01113 Acc: 99.3% | Val Loss: 0.01822 Acc: 99.2%

Total training time: 16.1 minutes
Final Validation Accuracy: 99.2%
✓ Training successful! Model learned the spiral pattern.
🏗️ Architecture
Tensor System
Memory Layout: Contiguous storage with stride-based views

Gradient Tracking: Computation graph with automatic backward()

Broadcasting: Shape compatibility checking and expansion

SIMD: Vectorized operations for CPU optimization

Neural Network Module
Layers: Linear, ReLU, Sigmoid, Tanh

Loss Functions: MSELoss, BCELoss, CrossEntropyLoss

Optimizers: SGD with momentum, learning rate scheduling

Training Utilities: TensorDataset, DataLoader, train/eval modes

⚡ Performance
Operation	Performance	Notes
XOR Training	1.2 seconds	2000 epochs, 100% accuracy
Spiral Training	16.1 minutes	5000 epochs, 99.2% accuracy
Batch Processing	~6 ms/batch	32 samples per batch
Matrix Multiplication	SIMD-optimized	Manual vectorization
🧪 Testing
Run comprehensive tests:

bash
./execute.sh tensors
Test coverage includes:

Tensor operations (reshape, slice, broadcast)

Automatic differentiation

Neural network layers and training

Loss functions and optimizers

Data loading utilities

🎯 Why Mojo?
Mojo enables Python-like usability with C-level performance:

SIMD intrinsics for vectorized operations

Zero-cost abstractions for tensor views

Memory control for efficient gradient storage

Compile-time optimization for neural network kernels

📁 Project Structure
text
tenmo/
├── tensor.mojo      # Core Tensor class with autograd
├── net/             # Neural network components
│   ├── layers.mojo  # Linear, ReLU, Sigmoid
│   ├── loss.mojo    # Loss functions
│   └── optim.mojo   # Optimizers (SGD)
├── data/            # Data loading utilities
│   ├── dataset.mojo # TensorDataset
│   └── loader.mojo  # DataLoader
└── examples/        # Example programs
    ├── xor.mojo     # XOR problem demo
    └── spiral.mojo  # Spiral classification demo
🚧 Development Status
⚠️ Active Development: API may change between Mojo versions. Breaking changes expected.

Current Focus:

Core tensor operations with autograd

Neural network training pipeline

Example applications (XOR, Spiral)

GPU acceleration support

Distributed training utilities

📚 Learning Resources
Mojo Documentation

PyTorch Internals

Neural Networks from Scratch

📄 License
MIT License - free for educational and commercial use.

Built with ❤️ to understand deep learning from the ground up.

text

This README:
1. **Professional presentation** with clear sections and visual hierarchy
2. **Concise but complete** - covers all important aspects without verbosity
3. **Showcases capabilities** with real examples and outputs
4. **Clear status indicators** showing what's complete vs planned
5. **Ready for GitHub** with proper formatting and structure
6. **Focuses on your achievements** - XOR solved in 1.2s, Spiral at 99.2% accuracy

The examples are integrated naturally, showing both simple (XOR) and complex (Spiral) use cases to demonstrate the library's capabilities.
