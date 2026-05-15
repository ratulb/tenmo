# AGENTS.md — Tenmo

Tenmo is a tensor library and neural network framework built in **Mojo**. Autograd, GPU support, SIMD-optimized kernels, and a minimal NN module system (Sequential, Linear, Conv2d, etc.) — all pure Mojo, no BLAS by default.

## Quick Commands

```bash
pixi shell                # enter dev environment (Mojo 0.26.2, linux-64 only)
pixi install              # install deps
./execute.sh all          # run all tests sequentially
./execute.sh <name>       # run single test (e.g. tensors, matmul, softmax)
./execute.sh quick        # fast sanity: tensors + shapes + strides + summean
./execute.sh gpu          # run all GPU-guarded tests
./execute.sh -p all       # parallel test run
./execute.sh -d <name>    # debug mode (-D LOGGING_LEVEL=debug)
./execute.sh from <name>  # run <name> and all tests after it in order
./example.sh xor          # run an example (xor|mnist|spiral|cifar_10|imdb|mnist_gpu|mnist_conv2d)
./example.sh <name> d     # example with debug logging
./fire.sh                 # quick-run scratchpad.mojo (or pass another file)
```

## Test Runner Details (`execute.sh`)

- Tests are compiled with `mojo -I .` — the repo root must be on the include path
- Logs go to `logs/<test_name>.log`
- Test names in `execute.sh` do **not** match filenames 1:1 (e.g. `ce` → `test_cross_entropy.mojo`, `npiop` → `test_numpy_interop.mojo`, `shapebroadcast` → `test_broadcaster.mojo`)
- Use the test **name** (not filename) as the argument to `execute.sh`
- `synth_mnist` is commented out in the test list
- The `gpu` test suite is a defined subset of tests that have GPU guards — not all tests run on GPU

## CI (`.github/workflows/test.yml`)

- Each test runs as a separate matrix job on `ubuntu-latest`
- `MOJO_STACK_SIZE=67108864` (64MB) and `ulimit -s unlimited` are required
- `libopenblas-dev` is installed as a system dependency
- Tests retry up to `MAX_RETRIES=2` on failure
- GPU job exists but is disabled (`if: false`) — needs a self-hosted runner

## Architecture

```
tenmo/          # library source (~93 .mojo files)
  __init__.mojo # re-exports everything; `from tenmo.tensor import Tensor`
  tensor.mojo   # Tensor[dtype] — main type, autograd, CPU+GPU
  ndbuffer.mojo # NDBuffer — shape/strides/offset, single source of truth
  gradbox.mojo  # Gradbox — gradient storage, independent refcount
  ancestry.mojo # Ancestor/Ancestors — lightweight parent handles
  backpropagation.mojo  # backward dispatch (jump table via op_code)
  net.mojo      # NN modules: Sequential, Linear, ReLU, Sigmoid, Conv2d, etc.
  sgd.mojo      # SGD optimizer with momentum
  dataloader.mojo       # DataLoader, TensorDataset, NumpyDataset
  nlp/          # NLP-specific modules
tests/          # ~90 test files
examples/       # training examples (xor, mnist, spiral, cifar_10, imdb)
```

Key design facts:
- `Tensor[dtype]` is generic over `DType` — use `alias dtype = DType.float32` or `comptime dtype = DType.float32`
- `track_grad: Bool` is a **compile-time** parameter — `model.eval()` sets it to `False`, eliminating graph overhead entirely
- `Gradbox` has its own refcount separate from `Tensor` — survives Mojo's ASAP destruction of intermediates
- `Ancestor` stores only what backward needs (id, gradbox ptr, layout, storage) — no full Tensor copies
- Backward dispatch uses a jump table on integer `op_code` — no variant extraction
- Views share `Buffer` via ref-counting — zero-copy slicing
- `DType.bool` is stored internally as `uint8` in GPU kernels

## Tensor Constructors

**Direct constructors** (`__init__`):
- `Tensor(shape, requires_grad=False)` — empty tensor of given `Shape`
- `Tensor(*axes_spans, requires_grad=False)` — variadic int dims
- `Tensor(row, requires_grad=False)` — from a 1D `List[Scalar]`
- `Tensor(ptr, shape, strides?, offset?, requires_grad?, copy=True)` — from raw pointer
- `Tensor(buffer, requires_grad=False)` — from `NDBuffer`

**Static constructors** (called as `Tensor[dtype].method(...)`):

| Constructor | Signature | Description |
|---|---|---|
| `d1` | `(row, requires_grad)` | 1D from `List[Scalar]` |
| `d2` | `(rows, requires_grad)` | 2D from `List[List[Scalar]]` |
| `d3` | `(blocks, requires_grad)` | 3D from `List[List[List[Scalar]]]` |
| `d4` | `(blockgrid, requires_grad)` | 4D from deeply nested lists |
| `zeros` | `(*dims/shape, requires_grad, device?)` | Filled with 0 |
| `zeros_like` | `(like, requires_grad?, device?)` | Zeros matching shape |
| `ones` | `(*dims/shape, requires_grad, device?)` | Filled with 1 |
| `ones_like` | `(like, requires_grad?, device?)` | Ones matching shape |
| `full` | `(shape, value, requires_grad, device?)` | Filled with scalar |
| `full_like` | `(like, value, requires_grad, device?)` | Fill matching shape |
| `rand` | `(*dims/shape, low=0, high=1, init_seed?, requires_grad)` | Uniform random |
| `randn` | `(*dims/shape, mean=0, std=1, init_seed?, requires_grad)` | Normal distribution (Box-Muller) |
| `arange` | `(*args, requires_grad)` | Evenly spaced 1D (`arange(stop)`, `arange(start, stop)`, `arange(start, stop, step)`) |
| `linspace` | `(start, end, steps, requires_grad)` | Linearly spaced 1D, inclusive |
| `eye` | `(n, requires_grad, device?)` | n×n identity matrix |
| `onehot` | `(indices, num_classes, device?, ignore_index?)` | One-hot encoding |
| `scalar` | `(val, requires_grad)` | 0D tensor |
| `from_device_buffer` | `(buffer, shape?, strides?, offset?, requires_grad)` | From GPU device buffer |

- All constructors accept `requires_grad: Bool` (default `False`)
- Shape-accepting constructors take `Shape`, `List[Int]`, or variadic `*dims: Int`
- `device` parameter defaults to CPU; pass `Device` for GPU allocation

## Tensor / Gradbox / NDBuffer Relationship

**NDBuffer** is the single source of truth for shape, strides, and offset. Both **Tensor** and **Gradbox** sit on top of it.

### Tensor Structure (`tensor.mojo`)
```
Tensor[dtype]
├── _id: UInt                          # unique identifier
├── buffer: NDBuffer[dtype]            # data + layout (shape/strides/offset)
├── requires_grad: Bool                # compile-time track_grad gate
├── gradbox: UnsafePointer[Gradbox]    # allocated only if requires_grad=True
└── ancestors: Optional[Ancestors]     # computation graph parents
```

### Gradbox Structure (`gradbox.mojo`)
```
Gradbox[dtype]
├── buffer: NDBuffer[dtype]            # gradient data + layout
└── _refcount: Atomic[UInt64]          # independent refcount from Tensor
```

### Key Invariants

1. **Gradbox initialized upfront** — When `requires_grad=True`, `init_gradbox()` allocates gradient storage immediately (zeros). On GPU, a `DeviceState` is allocated; on CPU, a contiguous `NDBuffer` of zeros.

2. **Independent refcounting** — Gradbox has its own atomic refcount separate from Tensor. This survives Mojo's ASAP destruction of intermediates, ensuring gradients persist through backward pass even when Tensor temporaries are freed.

3. **Views share data, own gradboxes** — When a view is created via `View.forward()`:
   - CPU: `buffer.share()` enables refcounting on the underlying `Buffer` — zero-copy slice
   - GPU: `DeviceBuffer` (Mojo GPU built-in) is always refcounted
   - The view gets its own independent `Gradbox` if `requires_grad=True` (allocated via `requires_grad_(True)` → `init_gradbox()`)
   - The view registers a `ViewArg` ancestry entry pointing to the parent tensor

4. **Views release gradients after backward** — During `ViewBackward.backward()`, the view's gradient is scattered back to the parent's gradbox, then the view's gradbox is zeroed (`ZeroGrad` op). Views don't retain gradients once they've doled them out to parents.

5. **Gradboxes are always contiguous with zero offset** — This is a hard invariant:
   - `Gradbox.__init__(shape)` creates a contiguous `NDBuffer(shape)` with default strides and offset 0
   - `init_gradbox()` for GPU creates a `DeviceState` via `.new(numels, 0)` — contiguous allocation
   - `Gradbox.as_tensor()` calls `.buffer.contiguous()` if not already contiguous before converting
   - All Gradbox operations (`transpose`, `permute`, `detach`, `reshape`) return `share=False` — owned contiguous copies

6. **GPU transfer makes data contiguous** — When `to_device()` transfers to GPU:
   - CPU → GPU: `DeviceState.fill(ndb)` reads the logical view (respects strides/offset), writes contiguously to GPU buffer. The resulting NDBuffer has offset=0, default strides
   - GPU → CPU (contiguous): `DeviceState.into(shape)` direct memcpy
   - GPU → CPU (strided): brings flat to CPU → creates view → materializes contiguous copy
   - GPU → different GPU: round-trips through CPU

### Layer 1: `Buffer` (CPU raw memory) — `tenmo/buffers.mojo`
- Flat allocation of `Scalar[dtype]` on CPU only
- Optional atomic refcount: `shared()` transforms layout from `[data]` → `[refcount][data]`
- `__copyinit__`: shared → refcount bump; unshared → deep copy
- SIMD-optimized arithmetic, comparison, unary ops (add, mul, exp, log, tanh, etc.)
- No shape, stride, or device knowledge

### Layer 2: `DeviceState` (GPU raw memory) — `tenmo/device.mojo`
- Wraps `DeviceBuffer[dtype]` + `GPU` context
- **`DType.bool` stored as `uint8`** — `DeviceBuffer[DType.bool]` unsupported; `datatype` comptime field handles cast
- `fill(source: NDBuffer)`: copies from CPU logical view (respects strides/offset)
- `into(shape)`: maps GPU buffer to host → CPU `Buffer` → `NDBuffer`
- `map_to_host()`: required for any CPU-side access to GPU memory

### Layer 3: `NDBuffer` (shaped tensor view) — `tenmo/ndbuffer.mojo`
Combines **Layout** + **Storage** into the single source of truth:

```
NDBuffer
├── Layout (metadata only)
│   ├── shape: Shape
│   ├── strides: Strides
│   ├── offset: Int
│   └── _contiguous: Bool
└── Storage (data carrier)
    ├── buffer: Buffer[dtype]          ← CPU
    └── device_state: DeviceState[dtype] ← GPU (Optional)
```

**Key rules:**
- `Storage` holds **either** CPU buffer **or** GPU device state — never both active
- `copy()` is always cheap — refcount bump on both Buffer and DeviceState
- `share(shape?, strides?, offset?)` creates a view: enables refcount on CPU, copies device_state ref
- `transpose(shared=True)` returns a view with permuted shape/strides; `shared=False` returns contiguous copy
- `to_device()` handles all transfer paths:
  - CPU → GPU: allocate DeviceState, `fill(self)` from logical view, return new NDBuffer
  - GPU → CPU: `device_state.into(shape)` (contiguous) or materialize with stride respect
  - GPU → different GPU: round-trip through CPU

### Device Transfer Summary
| Path | Mechanism |
|---|---|
| CPU → GPU | `DeviceState.fill(ndb)` reads logical view (strides/offset) |
| GPU → CPU (contiguous) | `DeviceState.into(shape)` direct memcpy |
| GPU → CPU (strided) | Bring flat to CPU → create view → materialize contiguous |
| GPU → GPU (same) | No-op, returns self |
| GPU → GPU (different) | GPU → CPU → new GPU |

### Bool Handling on GPU
- `DeviceState` comptime `datatype = DType.uint8 if dtype == DType.bool else dtype`
- All GPU fills, loads, stores cast between bool ↔ uint8
- `DeviceState.into()` converts uint8 0/1 back to bool on CPU path

## Forward & Backward Pass (Autograd)

### Operation Dispatch: Summer → NDBuffer → CPU/GPU

**Full call chain** for `tensor.sum(axes, keepdims)`:

```
Tensor.sum() → Summer.forward() → tensor.buffer.reduce[SUM](axes, keepdims)
```

#### NDBuffer Device Dispatch (`ndbuffer.mojo:1237`)

```mojo
fn reduce[op_code: Int = SUM](self, normalized_axes, keepdims) -> NDBuffer:
    comptime if has_accelerator():
        if self.is_on_gpu():
            out = Reduction[Self.dtype].launch[op_code](self, normalized_axes, keepdims)
        else:
            out = self.reduce_cpu[op_code](normalized_axes, keepdims)
    else:
        out = self.reduce_cpu[op_code](normalized_axes, keepdims)
```

- `has_accelerator()`: **comptime** check — GPU code is compiled out entirely on CPU-only systems
- `is_on_gpu()`: runtime check — `device_state is not None`
- Both paths return an `NDBuffer` — device is transparent to the caller

#### CPU Reduction Kernel (`ndbuffer.mojo:1267`)

`reduce_cpu[op_code]` iterates over output coordinates and reduced coordinates explicitly:

1. Compute `out_shape` via `shape.compute_output_shape(axes, keepdims)`
2. Allocate output `NDBuffer.zeros(out_shape)`
3. For each `out_coord` in `out_shape`:
   - Accumulate over `red_coord` in `reduction_axes_shape`
   - Build `self_coord` via `out_coord.replace(axes, red_coord)` (keepdims) or `out_coord.insert(axes, red_coord)` (no keepdims)
   - `accum += self[self_coord]` — strided indexing through `__getitem__`
4. If `op_code == MEAN`, divide by `reduced_volume`

**Strided indexing** (`ndbuffer.mojo:606`):
- `__getitem__(indices: IntArray)` → `IndexCalculator.flatten_index(shape, indices, strides, offset)` → `self.get(flat_index)`
- `get(index)` → `self.buffer[index]` — direct Buffer access

#### GPU Reduction Kernel (`reduction_kernel.mojo`)

**Launcher** (`Reduction.launch[op_code]`, line 1000):

1. Compute `output_shape`, `reduced_shape`, `total_output`, `reduced_volume`
2. Determine launch config: `(threads_per_block, num_blocks) = launch_config(total_output, reduced_volume)`
3. Allocate GPU output buffer via `device_context.enqueue_create_buffer(total_output)`
4. Compile and enqueue the `reduce` kernel with args: `result_buffer, A_buffer, in_shape, in_strides, reduction_axes, total_output, reduced_volume`
5. `device_context.synchronize()` — blocks until kernel completes
6. Wrap result in `DeviceState` → `NDBuffer.with_device_state()`

**Kernel** (`reduce[dtype, max_block_size, op_code]`, line 176):

One block per output element. Threads stripe across the reduced volume:

```
smem = shared memory[max_block_size]         # AddressSpace.SHARED
tid = thread_idx.x, block_size = block_dim.x
out_idx = block_idx.x                        # which output element this block handles

# Phase 1: Each thread stripes across reduced_volume
input_base = output_to_input_base(out_idx, in_shape, in_strides, reduction_axes)
local = 0
for rank = tid, tid+block_size, tid+2*block_size, ... < reduced_volume:
    offset = rank_to_reduced_offset(rank, in_shape, in_strides, reduction_axes)
    local += in_buffer[input_base + offset]

smem[tid] = local
barrier()

# Phase 2: Parallel tree reduction in shared memory
for stride = block_size/2, block_size/4, ..., 1:
    if tid < stride: smem[tid] += smem[tid + stride]
    barrier()

# Phase 3: Thread 0 writes result
if tid == 0:
    out_buffer[out_idx] = smem[0]             # SUM
    out_buffer[out_idx] = smem[0] / volume    # MEAN
```

**Index helpers**:
- `output_to_input_base(out_idx, ...)`: decomposes output index into non-reduced coordinates, returns base flat offset into input
- `rank_to_reduced_offset(rank, ...)`: decomposes linear rank into reduced coordinates, returns stride-based offset

**Launch config** (`launch_config[max_block_size]`, line 1489):
- `block_size` starts at 1, doubles until it reaches `min(reduced_volume, max_block_size)` (default 512)
- `num_blocks = total_output` — one block per output element
- Power-of-2 block sizes enable efficient tree reduction

**PRODUCT kernel** (`product_reduce`, line 298):
- Uses **float64 log-space** accumulation for overflow safety across all dtypes
- Three shared memory arrays: `smem_log` (float64), `smem_neg` (int32), `smem_zero` (int32)
- Final write: `zero_count > 0 → 0`, else `sign * exp(log_abs_sum)` cast back to dtype
- int64/uint64 values beyond 2^53 are approximate (documented limitation)

### Forward Pass Pattern

Every differentiable operation follows this pattern:

```mojo
fn forward[track_grad: Bool = True](self, ...) -> Tensor[Self.dtype]:
    # 1. Compute output buffer
    var nd_buffer = ...  # operation on NDBuffer

    # 2. Create output tensor (no grad tracking yet)
    var out = Tensor[Self.dtype](nd_buffer^, requires_grad=False)

    # 3. Optionally register ancestry for backward
    comptime if track_grad:
        grad_required = requires_grad.or_else(self.requires_grad)
        if grad_required:
            out.requires_grad_(True)                          # allocates gradbox
            var backwardFnArg = BackwardFnArg[Self.dtype](
                BACKWARD_*, OperationArg(...)                  # op_code + payload
            )
            out.add_ancestry(backwardFnArg^, self, other)     # register parents
    return out^
```

**Example: `Summer.forward()`** (`tenmo/summation.mojo`):
1. Normalize reduction axes via `Validator.normalize_reduction_axes()`
2. Compute reduction: `tensor.buffer.reduce[SUM](axes, keepdims)`
3. Create output tensor with `requires_grad=False`
4. If grad required: `out.requires_grad_(True)` → allocates gradbox (zeros)
5. Register ancestry: `BackwardFnArg(BACKWARD_SUM, ReductionArg(axes, keepdims))`
6. `out.add_ancestry(backwardFnArg^, tensor)` — stores parent handle + backward metadata

### `add_ancestry()` (`tensor.mojo:1080`)

Registers parent tensors and backward function on the output tensor:

```mojo
fn add_ancestry(mut self, var backwardFnArg: BackwardFnArg[Self.dtype], *parents: Tensor[Self.dtype]):
```

- Creates `Ancestors` with the `backwardFnArg` if not already present
- For each parent:
  - If parent buffer is **not shared**: copies it and calls `buffer.share()` to enable refcounting, then appends the copy
  - If parent buffer **is already shared**: appends the parent directly
- `Ancestors.append()` converts `Tensor` → `Ancestor` via `Ancestor.from_tensor()`

### `Ancestor` (`ancestry.mojo`)

Lightweight handle carrying everything backward needs — no full Tensor copy:

```
Ancestor[dtype]
├── _id: UInt                           # graph traversal key
├── requires_grad: Bool                 # skip gradient update if False
├── gradbox: UnsafePointer[Gradbox]     # gradient storage (refcount bumped)
├── layout: Layout                      # shape, strides, offset, _contiguous
├── storage: Storage[dtype]             # buffer or device_state (refcount bumped)
└── parents: Optional[Ancestors[dtype]] # recursive ancestry chain
```

- `__copyinit__`: bumps gradbox refcount via `fetch_add[MONOTONIC](1)`
- `__del__`: decrements refcount via `fetch_sub[RELEASE](1)`; destroys gradbox when count hits 0 (with `ACQUIRE` fence)
- `from_tensor()`: extracts layout/storage from Tensor's NDBuffer, bumps gradbox refcount if present
- `buffer()`: reconstructs `NDBuffer` on-demand from layout + storage
- `update_grad(incoming, op_code, extra_arg)`: applies gradient to gradbox via op_code dispatch:
  - `AddTensor`: `gradbox += incoming`
  - `SubtractTensor`: `gradbox -= incoming`
  - `ZeroGrad`: `gradbox.zero_grad()`
  - `ScatterAddTensor`: scatter-add via `Filler.scatter_add()`, optionally zero padding row

### `Ancestors` (`ancestry.mojo`)

Holds `List[Ancestor]` + `BackwardFnArg`:

- `append(parent)`: converts `Tensor` → `Ancestor` via `from_tensor()`
- `tensor(idx)`: reconstructs `Tensor` from `Ancestor` at index
- `backward_fn_arg()`: returns reference to stored `BackwardFnArg`

### `BackwardFnArg` (`backpropagation.mojo`)

Type-erased container for backward operation arguments:

```
BackwardFnArg[dtype]
├── op_code: Int                        # dispatch key (BACKWARD_*)
├── ptr: UnsafePointer[UInt8]           # type-erased payload
├── destroy: DestroyerFn                # type-specific destructor
└── copy_fn: CopyFn                     # type-specific deep copy
```

**Payload types** (all implement `ArgumentType`):

| Payload | Used By | Fields |
|---|---|---|
| `NullArg` | ADD, MULTIPLY | (empty) |
| `Boolean` | DROPOUT | `is_true` |
| `ScalarArg` | *_SCALAR ops | `value: Scalar` |
| `Integer` | integer params | `value: Int` |
| `IntArrayArg` | array params | `array: IntArray` |
| `ReductionArg` | SUM, MEAN | `axes: IntArray`, `keepdims: Bool` |
| `ViewArg` | VIEW | `shape`, `strides`, `offset` |
| `BlasArg` | BLAS_MATMUL_2D | `transpose_A`, `transpose_B`, `blas` |
| `StackArg` | STACK | `axis`, `num_tensors` |
| `ShuffleArg` | SHUFFLE | `axis`, `permutation` |
| `PadArg` | PAD | `pad: List[(Int,Int)]`, `mode` |
| `MinMaxArg` | MINMAX | `axes`, `keepdims`, `mask: NDBuffer` |
| `SoftmaxArg` | SOFTMAX | `axes`, `softmax_out: NDBuffer` |
| `ClipArg` | CLIP | `min_val`, `max_val` |
| `TilesArg` | TILE | `repeat`, `orig_shape` |
| `StdArg` | STD | `axis`, `unbiased`, `keepdims`, `epsilon` |
| `GatherArg` | GATHER | `indices`, `axis`, `padding_idx` |

- `get[T]()`: bitcasts type-erased ptr back to concrete type
- `__del__`: calls `destroy()` which invokes `T.__del__` on the payload

### Backward Pass (`tensor.mojo:3073`)

```mojo
fn backward[graph_size: Int = 50](mut output, start_grad: Scalar = 1.0)
fn backward[graph_size: Int = 50](mut output, seed_tensor: Tensor)
```

**Phase 1: Seed gradients**
- `output.seed_grad(seed_tensor)` — copies seed values into output's gradbox
- If gradbox doesn't exist, `requires_grad_()` allocates it (zeros) first

**Phase 2: DFS graph collection**
- Build `node_list: List[Ancestor]`, `id_to_index: Dict[UInt, Int]`, `fanin: Dict[UInt, Int]`
- Start from output's `Ancestor`, DFS through parents via `ancestry()`
- Track `fanin` count (number of children depending on each node)
- Record reverse topological order in `topo_ids`

**Phase 3: Reverse topological execution**
- `ready_queue` starts with output node
- For each node popped from queue:
  1. Call `Backward.invoke(node)` — jump table dispatch on `op_code`
  2. Backward handler returns `List[Tuple[Ancestor, Gradbox, Int]]` (target, gradient, op_code)
  3. For each result:
     - Extract `extra_arg` from output's `backward_fn_arg()` if `op_code == ScatterAddTensor`
     - `target.update_grad(grad, op_code, extra_arg)` — accumulates into parent's gradbox
     - Decrement `fanin[target_id]`; when it hits 0 and target has ancestry, add to `ready_queue`

### `Backward.invoke()` (`backpropagation.mojo:337`)

Jump table dispatcher with 58 operation codes:

```mojo
fn invoke(output: Ancestor[dtype]) -> List[Tuple[Ancestor, Gradbox, Int]]:
    ref arg = output.ancestry().backward_fn_arg()
    var op_code = arg.op_code
    if op_code == BACKWARD_SUM:
        return SumBackward[dtype].backward(output)
    # ... 57 more branches
```

- Guards: returns empty list if `!output.has_ancestry()`
- Each backward struct implements `static fn backward(output: Ancestor) -> List[Tuple[Ancestor, Gradbox, Int]]`
- All backward handlers are in separate modules, imported via `walkback.mojo`

### Example: `SumBackward` (`tenmo/summation.mojo`)

```mojo
fn backward(output: Ancestor) -> List[Tuple[Ancestor, Gradbox, Int]]:
    # 1. Extract reduction parameters from BackwardFnArg
    ref bwd_arg = output.ancestry().backward_fn_arg().get[ReductionArg]()
    var (axes, keepdims) = bwd_arg.axes, bwd_arg.keepdims

    # 2. Get the gradient and original shape
    ref gradbox = output.gradients()[]
    var ancestor = output.ancestry().get(0)
    ref shape = ancestor.shape()

    # 3. Broadcast gradient back to original shape
    if gradbox.shape() == Shape():  # scalar case
        grad_contrib = Gradbox.full(shape, gradbox.item(), share=False, device=...)
    else if not keepdims:
        # Unsqueeze reduced dimensions back to size 1
        axes = gradbox.shape().intarray().insert(axes, IntArray.filled(len(axes), 1))
        unsqueezed_grad = gradbox.reshape(Shape(axes))
        grad_contrib = unsqueezed_grad.broadcast_to(shape, share=False)
    else:
        grad_contrib = gradbox.broadcast_to(shape, share=False)

    # 4. Return (parent, gradient, accumulate_op)
    return [(ancestor^, grad_contrib^, AddTensor)]
```

### `seed_grad()` (`tensor.mojo:1226`)

- `seed_grad(with_tensor)`: copies gradient values from tensor into gradbox
- `seed_grad(value)`: fills gradbox with scalar value via `Tensor.full()`
- If gradbox doesn't exist, calls `requires_grad_()` to allocate it

### `update_grad[opcode]()` (`tensor.mojo:3186`)

Direct gradient update on Tensor (used by some paths):
- `MulTensor`: `gradbox *= incoming`
- `AddTensor`: `gradbox += incoming`
- `SubtractTensor`: `gradbox -= incoming`
- `ZeroGrad`: `self.zero_grad()`

## GPU Notes

- `to_gpu()` and `to_cpu()` accept `stop_grad` parameter (default `False`)
- `stop_grad=False`: gradient flows across device boundary transparently
- `stop_grad=True`: destination becomes a new leaf, gradient stays on that device
- Recommended training pattern: `model.to_gpu(stop_grad=True)` once, train entirely on GPU, `model.to_cpu(stop_grad=True)` to persist
- `Conv2d`, `MaxPool2d` GPU migration is WIP

## Pixi / Environment

- Platform: **linux-64 only**
- Mojo pinned to `==0.26.2` from `conda.modular.com/max-nightly`
- Python 3.10–3.14
- PyPI deps: `mnist-datasets`, `pure-cifar-10`, `tiktoken`
- Dev feature includes `mojodoc` and a local `bpe` at `../bpe`
- `pixi run docs` — generate docs via mojodoc

## BLAS

- `SequentialBLAS` with `LinearBLAS` layers auto-profile native Mojo vs BLAS matmul at runtime
- Profiling happens on first forward calls, then selects faster path
- Full backward pass support through BLAS
- Set `BLAS_PATH` env var for custom BLAS library path

## Style Conventions

- Mojo structs use `snake_case` for methods and fields
- `fn main() raises:` is the standard entrypoint pattern (tests use `raises`)
- `comptime` blocks for compile-time branching (e.g. `comptime if track_grad:`)
- `^` suffix for owned value transfer (e.g. `out.add_ancestry(backwardFnArg^, self)`)
- `ref` for borrow references (e.g. `ref gradbox = output.gradients()`)
