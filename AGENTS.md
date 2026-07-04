# AGENTS.md — Tenmo

## ⛔ CRITICAL: NEVER launch tests in parallel
Mojo compilation is memory-bound. Running more than one `./execute.sh` or `./fire.sh` at a time will OOM the machine.

## Quick Start
```bash
pixi shell                # enter dev environment (linux-64 only, Mojo 1.0.0b2)
pixi install              # install deps
./execute.sh quick        # fast sanity check (tensors + shapes + strides + summean)
```

## Test Commands
```bash
./execute.sh <name>       # single test (e.g. tensors, matmul, softmax)
./execute.sh cpu_all      # all CPU tests (chunks sequentially)
./execute.sh cpu_all N    # chunk N only
./execute.sh gpu_all      # all GPU tests (chunks sequentially)
./execute.sh -d <name>    # debug mode (-D LOGGING_LEVEL=debug)
./execute.sh from <name>  # run <name> and all tests after it
./example.sh xor          # run example: xor|mnist|mnist_unified|spiral|cifar_10|imdb|mnist_gpu|mnist_conv2d|word2vec_cbow
./fire.sh                 # quick-run debug.mojo (or pass another file)
```

### Test names ≠ filenames
Use test **names** (not filenames) with `execute.sh`:
- `ce` → `test_cross_entropy.mojo`
- `npiop` → `test_numpy_interop.mojo`
- `shapebroadcast` → `test_broadcaster.mojo`

### Timeout warning
Mojo compiles the entire library from scratch on every run. **Always use ≥12 min (720s) timeout.** Single tests take ~5 min to compile; `tensors` takes ~8 min. Logs go to `logs/<test_name>.log`.

## Architecture
```
tenmo/                    # library source (~94 .mojo files)
  __init__.mojo           # re-exports everything
  tensor.mojo             # Tensor[dtype] — main type, autograd, CPU+GPU
  ndbuffer.mojo           # NDBuffer — shape/strides/offset (single source of truth)
  gradbox.mojo            # Gradbox — gradient storage, independent refcount
  ancestry.mojo           # Ancestor/Ancestors — lightweight parent handles
  backpropagation.mojo    # backward dispatch (jump table on integer op_code)
  net.mojo                # NN modules: Sequential, Linear, ReLU, Conv2d, etc.
  optim.mojo              # SGD optimizer with momentum
  cpu_arithmetics.mojo    # CPU arithmetic dispatch (CpuArithmeticOps)
  dataloader.mojo         # DataLoader, TensorDataset, NumpyDataset
  blashandle.mojo         # BLAS integration (OpenBLAS)
  kernels/                # GPU kernel implementations
  nlp/                    # NLP modules (Embedding, etc.)
tests/                    # ~112 test files
examples/                 # training examples
nanogpt/                  # GPT implementation (WIP)
```

## Key Design Facts
- `Tensor[dtype]` is generic over `DType` — use `alias dtype = DType.float32` for comptime dtype
- `track_grad: Bool` is a **compile-time** parameter — `model.eval()` eliminates graph overhead entirely
- `Gradbox` has its own atomic refcount separate from `Tensor` — survives Mojo's ASAP destruction of intermediates
- `Ancestor.ndb` is `Optional[NDBuffer]` — populated only for ops with `needs_parent_data=True`
- Backward dispatch: integer `op_code` jump table in `backpropagation.mojo:357` (58 ops). No variant extraction.
- `DType.bool` stored as `uint8` in GPU kernels (`DeviceBuffer[DType.bool]` unsupported)
- Gradboxes are always contiguous with zero offset
- Views share `Buffer` via refcounting (zero-copy), but get their own independent `Gradbox`
- GPU transfer always makes data contiguous

## Forward Pass Pattern
Every differentiable operation follows this pattern:
1. Compute output on `NDBuffer`
2. Create `Tensor(ndbuffer^, requires_grad=False)`
3. If grad required: `out.requires_grad_(True)` → allocates gradbox (zeros), then `out.add_ancestry(backwardFnArg^, parents...)`
4. Return `out^`

**BackwardFnArg** stores operation metadata (`op_code` + type-erased payload). Most ops store only parameters (axes, shape). Three ops store their **output NDBuffer** because backward needs the forward output values: **Sigmoid, Tanh, Exp** (plus ReLU stores mask, Softmax stores output).

## Backward Pass
`tensor.backward()` works on **any** tensor, not just scalars — equivalent to `tensor.sum().backward()` in PyTorch.

Phase 1: seed gradient. Phase 2: DFS graph collection (fanin tracking, reverse topological order). Phase 3: reverse topological execution via ready queue — each node calls `Backward.invoke()` which dispatches to the handler via op_code, which accumulates into parent gradboxes.

## GPU Sync Conventions
GPU ops queue asynchronously. `sync: Bool` controls CPU wait:
- Tensor dunders (`+`, `+=`, `*`, etc.): `sync=True` (safe by default)
- Gradbox dunders (backward accumulation): `sync=False`
- `backward()` entry: `sync=True` (fence before forward→backward transition)
- `to_gpu()` / `to_cpu()`: `sync=True` (pass `sync=False` for async batch transfer)
- NDBuffer dispatch methods: `sync=False` (caller manages)

## GPU Device Transfer
| Path | Mechanism |
|---|---|
| CPU → GPU | `DeviceState.fill(ndb)` reads logical view (strides/offset) |
| GPU → CPU (contiguous) | `DeviceState.into(shape)` direct memcpy |
| GPU → CPU (strided) | Bring flat to CPU → create view → materialize contiguous copy |
| GPU → GPU (same) | No-op, returns self |
| GPU → GPU (different) | Round-trip through CPU |

`to_gpu()/to_cpu()` accept `stop_grad` (default `False`). `stop_grad=True` makes destination a new leaf. Recommended pattern: `model.to_gpu(stop_grad=True)` once, train entirely on GPU.

### Known GPU gaps (CPU fallbacks)
- **Conv2D, MaxPool2d** — no GPU path (use `data_ptr()` + `parallelize` on CPU)
- **Tensor.rand/randn/arange/linspace** — no `device` param (workaround: `.to_gpu()` after construction)
- **Tensor.onehot** — per-element `map_to_host()` round trips on GPU
- **CE probability targets** — forward falls back to GPU→CPU transfer + CPU fused kernel (backward is fine)

## BLAS
Default path: `/lib/x86_64-linux-gnu/libopenblas.so.0` (system install via `apt-get install libopenblas-dev`). Override at compile time:
```bash
mojo -I . -D BLAS_PATH=/path/to/libopenblas.so tests/test_cpu_all_3.mojo
```
`SequentialBLAS` with `LinearBLAS` layers auto-profile native Mojo vs BLAS matmul at runtime on first forward.

## Mojo 1.0.0b2 Notes
- **`fn` is an error** — use `def` for all function/struct method declarations and function pointer types (`fn(...) thin` → `def(...) thin`)
- **`Movable.__init__`**: `take` → `move` (rename parameter)
- **`mojo package` → `mojo precompile`**, `.mojopkg` → `.mojoc`
- **Implicit `std` imports are now an error** — must fully qualify: `from std.algorithm import ...`
- **Deprecated** (fix when modifying nearby code):
  - `compile_function[func, func]()` → `compile_function[func]()`
  - `as_any_origin()` → `as_unsafe_any_origin()`
  - `unsafe_origin_cast[MutAnyOrigin]()` → `unsafe_origin_cast[MutUnsafeAnyOrigin]()`
  - `MutAnyOrigin/ImmutAnyOrigin` → `MutUnsafeAnyOrigin/ImmutUnsafeAnyOrigin`
  - `reflect[T]()` → `reflect[T]` (no parens)
  - `Idx[value]()` → `Idx[value]` (no parens for comptime coords)
  - `constrained[cond, msg]()` → `comptime assert cond, msg`
  - `register_passable` keyword removed (implicitly computed)
- **No `address_of`** — use `UnsafePointer(to=...)` or `Pointer(to=...)`
- **`.unsafe_value()`** not `.value()` for `Optional[UnsafePointer[...]]`
- **`def(...) thin` function pointers** — named functions only, no lambdas
- **Atomic API**: `from std.atomic`, `Atomic[DType.uint64]`, `Ordering.RELAXED/RELEASE/ACQUIRE`
- **Cache debug**: `mojo --print-cache-location` and `mojo --clear-cache [-f]`

## CI (`.github/workflows/test.yml`)
- Each test is a separate matrix job on `ubuntu-latest` (never parallel within a job)
- Environment: `MOJO_STACK_SIZE=67108864`, `ulimit -s unlimited`, `libopenblas-dev`
- Tests retry up to `MAX_RETRIES=2` on failure
- GPU job exists but is disabled (`if: false`) — needs a self-hosted runner

## Environment
- Platform: **linux-64 only**
- Mojo `==1.0.0b2` from `conda.modular.com/max-nightly`
- Python 3.10–3.12
- Key PyPI deps: `mnist-datasets`, `pure-cifar-10`, `tiktoken`
- `pixi run docs` — generate docs via mojodoc (from `tenmo/`)
- `pixi run docs-build` — build docs to `target/doc/`
