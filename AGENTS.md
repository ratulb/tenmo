# Tenmo — Agents Guide

A lean tensor library and neural network framework built in Mojo.

## Documentation Conventions (mojodoc)

Follow [mojodoc](https://github.com/ehsanmok/mojodoc) format for all documentation:

```mojo
"""Brief one-line summary.

Detailed description with more context about what
this function does and when to use it.

Args:
    name: The user's name.
    age: The user's age in years.

Returns:
    A greeting message.

Raises:
    Error: If name is empty.

Example:
    ```mojo
    var greeting = greet("Alice", 30)
    print(greeting)  # Hello, Alice!
    ```
"""
```

Key terminology rules:
- Use **"instance"** NOT "object" (Mojo has structs, not objects)
- Use **"Mojo List"** for `List[Int]` type, NOT "Python list" or just "list"
- Use **"IntArray"** for the tenmo IntArray type
- Document all public methods with Args, Returns, Raises, and Notes where applicable
- Keep docstrings concise but complete

## Setup

```bash
curl -fsSL https://pixi.sh/install.sh | sh
source ~/.bashrc
pixi install
pixi shell
```

## Running Tests

Tests live in `tests/test_*.mojo`.

```bash
./execute.sh <testname>           # basic test runner
./execute_advanced.sh <testname> # colored output + timing
```

Debug mode (add `d` as second arg):
```bash
./execute.sh tensors d
```

Run all tests:
```bash
./execute.sh all
```

## Running Examples

Examples live in `tenmo/examples/*.mojo`.

```bash
./example.sh xor      # XOR training
./example.sh mnist    # MNIST training
./example.sh spiral  # Spiral classification
./example.sh cifar_10 # CIFAR-10 training
```

Available test targets (from `execute.sh` line 6): `unary`, `sqrt`, `inplace`, `sigmoid`, `summean`, `exp`, `transmute`, `count_unique`, `compare`, `allany`, `power`, `onehot`, `maxmin_scalar`, `contiguous`, `item`, `scalar`, `broadcast`, `gpu_expand`, `expand`, `gpusummean`, `gpu`, `npiop`, `sgd`, `matmul`, `cnn`, `chunk`, `fill`, `pad`, `logarithm`, `stack`, `concat`, `std_variance`, `blas`, `dropout`, `indexhelper`, `utils`, `variance`, `tanh`, `losses`, `data`, `intarray`, `tensors`, `mmnd`, `mm2d`, `mv`, `vm`, `argminmax`, `minmax`, `repeat`, `tiles`, `slice`, `linspace`, `softmax`, `relu`, `shuffle`, `buffers`, `flatten`, `permute`, `squeeze`, `unsqueeze`, `views`, `gradbox`, `ndb`, `transpose`, `shapes`, `strides`, `bench`, `validators`, `ce`, `synth_mnist`, `shapebroadcast`, `all`

## Architecture

- Main library: `tenmo/`
- Tests: `tests/test_*.mojo`
- Examples: `tenmo/examples/*.mojo`
- Core types: `Tensor`, `NDBuffer`, `Gradbox` — defined in `tenmo/tensor.mojo`, `tenmo/ndbuffer.mojo`, `tenmo/gradbox.mojo`
- Import: `from tenmo import Tensor` or `from tenmo.tensor import Tensor`

## Mojo Version

Requires `mojo == 0.26.2` (from `pixi.toml`). The API is subject to change with Mojo nightly.

## CI

- Runs on every push/PR to `development` and `main` via `.github/workflows/test.yml`
- All test targets run via `pixi run ./execute.sh <test>`
- GPU tests are disabled (`if: false`) until a self-hosted runner is available