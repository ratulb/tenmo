# 🧠 Mojo Tensor

A blazing-fast, from-scratch **Tensor** library built in [Mojo🔥](https://modular.com/mojo), with full support for:

- 🧮 N-dimensional Tensors
- 🔁 Broadcasting
- 🔢 Autodiff (automatic differentiation)
- 🧠 Scalar & elementwise operations
- 🧬 SIMD vectorization
- 🪜 Views and slicing
- 🧪 Comprehensive test coverage

> 🚧 **Work in Progress** – This project is a deep dive into Mojo internals, systems programming, and the foundation of modern deep learning libraries. Built with ❤️ from scratch — no NumPy, PyTorch, or TensorFlow under the hood.

---

## 🚀 Features

| Feature                  | Status | Notes |
|--------------------------|--------|-------|
| ✅ N-dimensional Tensor   | ✔️     | `Tensor[DType.float64]`, `Tensor[DType.bool]`, etc|
| ✅ Manual Broadcasting    | ✔️     | Compatible shapes |
| ✅ Elementwise Ops        | ✔️     | `+`, `-`, `*`, `/`, `pow`, etc. |
| ✅ SIMD Support           | ✔️     | Vectorized compute for speed |
| ✅ Pretty-printing        | ✔️     | Recursive + aligned |
| ✅ Slicing & Views        | ✔️     | Offset-aware `TensorView` |
| ✅ Autodiff               | 🧪     | Scalar ops + graph-based backprop |
| ✅ Gradient Tracking      | 🧪     | In progress |
| ✅ Unit Testing           | ✔️     | Custom test suite |

---

## 📦 Example

```mojo
from tensors import Tensor

Tensor.arange(6, requires_grad=True).reshape(2, 3).print()

[2D Tensor(2, 3), Type: float32, requires_grad: True]
  [
    [0.0, 1.0, 2.0, ],
    [3.0, 4.0, 5.0, ],
  ]


# Broadcasting + elementwise op
t2 = t + 1.0
t2.print()

# Gradient tracking
y = t * 2.0
y.backward()
t.grad[].print()
```

---

## 📁 Project Structure

```
.
├── tensors.mojo                # Core Tensor implementation
├── views.mojo                  # TensorView (slicing/view logic)
├── shapes.mojo                 # Shape logic and utilities
├── intlist.mojo                # Light Intger list backing many operations
├── operators.mojo              # Vectorized ops
├── tests/test_tensors.mojo     # Unit tests
└── README.md                   # You're here!
```

---

## 🧪 Running Tests

Run using the Mojo CLI:

```bash
./execute tensors
```

Tests cover:
- Shape + flattening
- Broadcasting correctness
- Slicing and views

---

## 🔬 Why Mojo?

Mojo combines Python’s usability with C’s performance and MLIR’s power. This project explores:

- Low-level memory and layout control
- SIMD vectorization
- Manual shape + broadcast logic
- First-principles autodiff 
- Efficient slicing and in-place views

Perfect for understanding what deep learning libraries **actually do under the hood**.

---

## 🛠️ Roadmap

- [x] Optimized matmul/optmization in general
- [x] Transperant GPU support
- [x] Distributed Training


---

## 💡 Inspirations

- [NumPy](https://numpy.org/)  
- [PyTorch internals](https://pytorch.org/)  
- [Karpathy's `llm.c`](https://github.com/karpathy/llm.c)  
- [Mojo by Modular](https://www.modular.com/mojo)  

---

## 🧠 Philosophy

> Build from scratch. Understand deeply. Control everything. Optimize aggressively.

This is a **learning-first** project: correctness first, performance next, scale later.

---

## 📜 License

MIT License – do what you love. Attribution appreciated, not required.

---

## 👋 Author

**Ratul Buragohain** — Machine learning enthusiast, and autodiff explorer 🐁

---

