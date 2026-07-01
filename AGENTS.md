# AGENTS.md — Tenmo

## ⛔ CRITICAL: NEVER launch multiple tests in parallel — Mojo compilation is memory-bound and will crash the machine. Run ONE `./execute.sh` or `./fire.sh` at a time, wait for it to complete, then run the next.

Tenmo is a tensor library and neural network framework built in **Mojo**. Autograd, GPU support, SIMD-optimized kernels, and a minimal NN module system (Sequential, Linear, Conv2d, etc.) — all pure Mojo, no BLAS by default.

## Quick Commands

```bash
pixi shell                # enter dev environment (Mojo 0.26.2, linux-64 only)
pixi install              # install deps
./execute.sh all          # run all tests sequentially
# ⛔ CRITICAL: NEVER launch multiple tests in parallel — Mojo compilation is memory-bound and will crash the machine. Run ONE `./execute.sh` or `./fire.sh` at a time, wait for it to complete, then run the next.
./execute.sh <name>       # run single test (e.g. tensors, matmul, softmax)
./execute.sh quick        # fast sanity: tensors + shapes + strides + summean
./execute.sh gpu          # run all GPU-guarded tests
./execute.sh cpu_all      # run all CPU tests (all chunks sequentially)
./execute.sh cpu_all N    # run chunk N (e.g. cpu_all 2)
./execute.sh cpu_all M..N # run chunks M through N (e.g. cpu_all 2..4)
./execute.sh cpu_all M N  # run specific chunks (e.g. cpu_all 1 4)
./execute.sh gpu_all      # run all GPU tests (all chunks sequentially)
./execute.sh gpu_all N    # run chunk N (e.g. gpu_all 2)
./execute.sh gpu_all M..N # run chunks M through N (e.g. gpu_all 2..4)
./execute.sh gpu_all M N  # run specific chunks (e.g. gpu_all 1 4)
./execute.sh -p all       # parallel test run
./execute.sh -d <name>    # debug mode (-D LOGGING_LEVEL=debug)
./execute.sh from <name>  # run <name> and all tests after it in order
./example.sh xor          # run an example (word2vec_cbow|xor|mnist|mnist_unified|spiral|cifar_10|imdb|mnist_gpu|mnist_conv2d|mnist_gpu_prof)
./example.sh <name> d     # example with debug logging
./fire.sh                 # quick-run scratchpad.mojo (or pass another file)
```

## ⛔ CRITICAL: NEVER launch multiple tests in parallel — Mojo compilation is memory-bound and will crash the machine. Run ONE `./execute.sh` or `./fire.sh` at a time, wait for it to complete, then run the next.

## Test Runner Details (`execute.sh`)

- Tests are compiled with `mojo -I .` — the repo root must be on the include path
- Logs go to `logs/<test_name>.log`
- GPU access: launch a Kaggle GPU notebook, then SSH into it (no local GPU)
- Test names in `execute.sh` do **not** match filenames 1:1 (e.g. `ce` → `test_cross_entropy.mojo`, `npiop` → `test_numpy_interop.mojo`, `shapebroadcast` → `test_broadcaster.mojo`)
- Use the test **name** (not filename) as the argument to `execute.sh`
- `synth_mnist` is commented out in the test list
- The `gpu` test suite is a defined subset of tests that have GPU guards — not all tests run on GPU
- The `gpu_all` test alias runs all GPU-guarded tests via chunked files (`tests/test_gpu_all_{1..N}.mojo`)
- `./execute.sh gpu_all` — run all chunks sequentially
- `./execute.sh gpu_all N` — run chunk N only
- `./execute.sh gpu_all M..N` — run chunks M through N (inclusive)
- `./execute.sh gpu_all M N O` — run specific chunks
- Chunked files are generated via `python3 scripts/generate_gpu_test_suite.py --chunks N`
- The monolithic `test_gpu_all.mojo` has been removed — compiling all 1138 GPU tests as one unit consumes 20GB+ RAM and takes >1hr. Use chunks instead.
- The `cpu_all` test alias runs all CPU tests via chunked files (`tests/test_cpu_all_{1..N}.mojo`)
- `./execute.sh cpu_all` — run all chunks sequentially
- `./execute.sh cpu_all N` — run chunk N only
- `./execute.sh cpu_all M..N` — run chunks M through N (inclusive)
- `./execute.sh cpu_all M N O` — run specific chunks
- Chunked files are generated via `python3 scripts/generate_cpu_test_suite.py --chunks N`
- Generating chunks requires much less RAM than the monolithic file: ch2 compiled in 433s with ~8GB peak
- `--only` flag for filtering individual tests is **not supported** in Mojo's TestSuite — run individual test files via `./execute.sh <name>` instead
- **Always use ≥12 min timeout** (720s) when launching tests via `mojo` or `./execute.sh`. Mojo compiles the entire library from scratch each time — single-test suites take ~5 min to compile, `tensors` takes ~8 min.

## CI (`.github/workflows/test.yml`)

- Each test runs as a separate matrix job on `ubuntu-latest`
- `MOJO_STACK_SIZE=67108864` (64MB) and `ulimit -s unlimited` are required
- `libopenblas-dev` is installed as a system dependency
- Tests retry up to `MAX_RETRIES=2` on failure
- GPU job exists but is disabled (`if: false`) — needs a self-hosted runner

## ⛔ CRITICAL: NEVER launch multiple tests in parallel — Mojo compilation is memory-bound and will crash the machine. Run ONE `./execute.sh` or `./fire.sh` at a time, wait for it to complete, then run the next.

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
  optim.mojo      # SGD optimizer with momentum
  dataloader.mojo       # DataLoader, TensorDataset, NumpyDataset
  nlp/          # NLP-specific modules
tests/          # ~90 test files
examples/       # training examples (xor, mnist, spiral, cifar_10, imdb)
```

Key design facts:
- `Tensor[dtype]` is generic over `DType` — use `alias dtype = DType.float32` or `comptime dtype = DType.float32`
- `track_grad: Bool` is a **compile-time** parameter — `model.eval()` sets it to `False`, eliminating graph overhead entirely
- `Gradbox` has its own refcount separate from `Tensor` — survives Mojo's ASAP destruction of intermediates
- `Ancestor.ndb` is `Optional[NDBuffer]` — populated only for ops with `needs_parent_data=True`; `to_ancestor()` creates ancestors without ndb
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
├── gradbox: Optional[Gradbox[dtype]]  # gradient storage (only if requires_grad=True)
└── ancestors: Optional[Ancestors]     # computation graph parents
```

### Gradbox Structure (`gradbox.mojo`)

```
Gradbox[dtype]
├── _ndb_ptr: Optional[UnsafePointer[NDBuffer]]   # pointer to heap NDBuffer
└── _refcount: Optional[UnsafePointer[Atomic]]     # pointer to heap atomic refcount
```

Buffer-like combined heap allocation: `[Atomic[DType.uint64] | NDBuffer]`.
- `_refcount` points to the Atomic at the start.
- `_ndb_ptr` points to the NDBuffer after the Atomic.
- `buffer()` method returns `ref self._ndb_ptr.unsafe_value()[]` — the NDBuffer reference.
- Stored inline in both `Tensor` and `Ancestor` as `Optional[Gradbox[dtype]]` — no separate heap allocation for the Gradbox wrapper itself.

### Key Invariants (Updated)

1. **Gradbox initialized upfront** — When `requires_grad=True`, `init_gradbox()` allocates gradient storage immediately (zeros) via combined heap allocation (`[Atomic | NDBuffer]`). On GPU, a `DeviceState` is allocated; on CPU, a contiguous `NDBuffer` of zeros.

2. **Independent refcounting** — Gradbox has its own atomic refcount (`_refcount`) separate from Tensor. This survives Mojo's ASAP destruction of intermediates, ensuring gradients persist through backward pass even when Tensor temporaries are freed. `__copyinit__` bumps the refcount; `__del__` decrements and frees when last handle drops. Stored inline in Tensor and Ancestor as `Optional[Gradbox[dtype]]`.

3. **Views share data, own gradboxes** — When a view is created via `View.forward()`:
   - CPU: `buffer.share()` enables refcounting on the underlying `Buffer` — zero-copy slice
   - GPU: `DeviceBuffer` (Mojo GPU built-in) is always refcounted
   - The view gets its own independent `Gradbox` if `requires_grad=True` (allocated via `requires_grad_(True)` → `init_gradbox()`)
   - The view registers a `ViewArg` ancestry entry pointing to the parent tensor

4. **Views release gradients after backward** — During `ViewBackward.backward()`, the view's gradient is scattered back to the parent's gradbox, then the view's gradbox is zeroed (`ZeroGrad` op). Views don't retain gradients once they've doled them out to parents.

5. **Gradboxes are always contiguous with zero offset** — This is a hard invariant:
   - `Gradbox.__init__(shape)` creates a contiguous `NDBuffer(shape)` with default strides and offset 0
   - `init_gradbox()` for GPU creates a `DeviceState` via `.new(numels, 0)` — contiguous allocation
   - `Gradbox.as_tensor()` calls `.buffer().contiguous()` if not already contiguous before converting
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
Single source of truth, combining metadata and data:

```
NDBuffer[dtype]
├── shape: Shape                    # Tensor dimensions
├── strides: Strides                # Memory layout
├── offset: Int                     # View offset
├── _contiguous: Bool               # Cache
├── buffer: Buffer[dtype]           # CPU data
└── device_state: Optional[DeviceState]  # GPU data
```

**Key rules:**
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

## GPU Lazy Evaluation (`sync` Conventions)

GPU operations queue asynchronously. A `sync: Bool` parameter controls when the CPU waits for GPU completion. Three tiers of defaults:

| Tier | Default | Rationale |
|---|---|---|
| **Tensor dunders** (`+`, `+=`, `*`, etc.) | `sync=True` | User-facing operations should be safe by default. Every op waits. |
| **Forward structs** (`Adder.forward`, etc.) | sync threaded from caller | Tensor dunders → `sync=True`, Gradbox dunders → `sync=False`. |
| **Gradbox dunders** (`gradbox += x`) | `sync=False` | Backward accumulation queues ops without blocking. Final sync at `backward()` entry. |
| **`backward()` entry point** | `sync=True` (fence before `seed_grad`) | Option A: fences GPU before forward→backward transition. CPU graph traversal overlaps with GPU seed copy. |
| **`to_gpu()` / `to_cpu()`** | `sync=True` | Data transfer and readback are sync points by default. Pass `sync=False` for async batch transfer. |
| **NDBuffer dispatch methods** (`arithmetic_ops`, `scalar_ops`, etc.) | `sync=False` | Called from forward structs which manage sync themselves. |
| **Backward handlers** (jump table functions) | hardcoded `sync=False` | Called after backward entry fence — no further sync needed. |

**Key pattern:** User code uses Tensor dunders (safe sync). Autograd backward uses Gradbox dunders (no sync). `backward(sync=True)` ensures forward GPU work completes before backward starts.

**Training loop with lazy GPU:**
```mojo
var features_gpu = batch.features.to_gpu(gpu, sync=False)  # async queue copy
var labels_gpu = batch.labels.to_gpu(gpu, sync=False)
var pred = model(features_gpu)                               # queues on GPU
var loss = criterion(pred, labels_gpu)
loss.backward()                                              # sync before backward
optimizer.step()                                             # GPU-side param update
loss.item()                                                  # sync: read scalar
pred.to_cpu()                                                # sync: bring to CPU
```

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
def forward[track_grad: Bool = True](self, ...) -> Tensor[Self.dtype]:
    # 1. Compute output buffer
    var nd_buffer = ...  # operation on NDBuffer

    # 2. Create output tensor (no grad tracking yet)
    var out = Tensor[Self.dtype](nd_buffer^, requires_grad=False)

    # 3. Optionally register ancestry for backward
    comptime if track_grad:
        var grad_required = requires_grad.or_else(self.requires_grad)
        if grad_required:
            out.requires_grad_(True)                          # allocates gradbox
            var backwardFnArg = BackwardFnArg[Self.dtype](
                BACKWARD_*, OperationArg(...),                  # op_code + payload
                needs_parent_data=True                          # True if backward reads parent shape/buffer
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
  - If `backwardFnArg.needs_parent_data == True`: copies parent's buffer and calls `buffer.share()` to enable refcounting, then appends with populated ndb
  - If `needs_parent_data == False`: appends the parent via `to_ancestor()` — no ndb copy
- `Ancestors.append()` converts `Tensor` → `Ancestor` via `Tensor.to_ancestor()`

### `Ancestor` (`ancestry.mojo`)

Lightweight handle carrying only what backward needs — ndb is Optional, populated only when required:

```
Ancestor[dtype]
├── _id: UInt                           # graph traversal key
├── requires_grad: Bool                 # skip gradient update if False
├── gradbox: Optional[Gradbox[dtype]]   # gradient storage (inline via Optional)
├── ndb: Optional[NDBuffer[dtype]]      # data+layout (None unless needs_parent_data=True)
└── parents: Optional[Ancestors[dtype]] # recursive ancestry chain
```

- `__copyinit__`: bumps gradbox refcount via `fetch_add[MONOTONIC](1)`
- `__del__`: decrements refcount via `fetch_sub[RELEASE](1)`; destroys gradbox when count hits 0 (with `ACQUIRE` fence)
- `buffer()`: returns `ref[self.ndb.value()]NDBuffer[Self.dtype]` — reference, no copy; **panics if ndb is empty**
- `shape()`, `strides()`, `offset()`, `max_index()`: delegate to `ndb.value()` — panic if empty
- `is_on_gpu()`: safe — checks `if self.ndb:` first, returns `False` when empty
- `update_grad(incoming, op_code, extra_arg)`: applies gradient to gradbox via op_code dispatch:
  - `AddTensor`: `gradbox += incoming`
  - `SubtractTensor`: `gradbox -= incoming`
  - `ZeroGrad`: `gradbox.zero_grad()`
  - `ScatterAddTensor`: scatter-add via `Filler.scatter_add()`, optionally zero padding row

### `Ancestors` (`ancestry.mojo`)

Holds `List[Ancestor]` + `BackwardFnArg`:

- `append(parent)`: converts `Tensor` → `Ancestor` via `to_ancestor()`
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
| `NDBufferArg` | SIGMOID_BACKWARD, TANH_BACKWARD, BACKWARD_EXPONENTIAL, RELU | `ndb: NDBuffer` |
| `ClipArg` | CLIP | `min_val`, `max_val` |
| `TilesArg` | TILE | `repeat`, `orig_shape` |
| `StdArg` | STD | `axis`, `unbiased`, `keepdims`, `epsilon` |
| `GatherArg` | GATHER | `indices`, `axis`, `padding_idx` |

- `get[T]()`: bitcasts type-erased ptr back to concrete type
- `__del__`: calls `destroy()` which invokes `T.__del__` on the payload

### Backward Pass (`tensor.mojo:3073`)

```mojo
def backward[graph_size: Int = 50](mut output, start_grad: Scalar = 1.0, sync: Bool = True)
def backward[graph_size: Int = 50](mut output, seed_tensor: Tensor, sync: Bool = True)
```

`sync=True` (default): GPU sync fence before `seed_grad()` — all forward GPU work completes before backward starts (Option A). CPU graph traversal overlaps with GPU seed copy.

`sync=False`: no fence — caller must ensure GPU forward ops are complete before backward. Use only when managing sync externally.

**Phase 1: Seed gradients**
- `output.seed_grad(seed_tensor)` — copies seed values into output's gradbox
- If gradbox doesn't exist, `requires_grad_()` allocates it (zeros) first

**Phase 2: DFS graph collection**
- Build `node_list: List[Ancestor]`, `id_to_index: Dict[UInt, Int]`, `fanin: Dict[UInt, Int]`
- Start from output's `Ancestor` via `to_ancestor()`, then inject ndb: `root.ndb = output.buffer.copy()` (root always needs data)
- DFS through parents via `ancestry()` — parent ancestors carry ndb only if their `BackwardFnArg.needs_parent_data` was True
- Track `fanin` count (number of children depending on each node)
- Record reverse topological order in `topo_ids`

**Phase 3: Reverse topological execution**
- `ready_queue` starts with output node, `parent_ids: List[UInt]` reused per node
- For each node popped from queue:
  1. `parent_ids.clear()`
  2. Call `Backward.invoke(node, parent_ids)` — jump table dispatch on `op_code`. Each backward handler:
     - Reads gradient from `output.gradbox[]`
     - Computes parent gradient contributions
     - Calls `parent.update_grad(grad, op_code, extra_arg)` to accumulate into parent's gradbox
     - Appends parent `_id` to `parent_ids`
  3. For each `target_id` in `parent_ids`:
     - Decrement `fanin[target_id]`; when it hits 0 and target has ancestry, add to `ready_queue`

### `Backward.invoke()` (`backpropagation.mojo:357`)

Jump table dispatcher with 58 operation codes:

```mojo
def invoke(
    output: Ancestor[Self.dtype],
    mut parent_ids: List[UInt],
):
    if not output.has_ancestry():
        return
    ref arg = output.ancestry().backward_fn_arg()
    var op_code = arg.op_code
    if op_code == BACKWARD_SUM:
        SumBackward[Self.dtype].backward(output, parent_ids)
    # ... 57 more branches
```

- Guards: returns early if `!output.has_ancestry()`
- Each backward struct implements `def backward(output: Ancestor[Self.dtype], mut parent_ids: List[UInt], retain_graph: Bool = False)` — handlers call `parent.update_grad()` and `parent_ids.append()`
- Backward handlers that call `parent.shape()`, `parent.buffer()`, `parent.strides()`, `parent.offset()`, or `parent.max_index()` require `needs_parent_data=True` on the `BackwardFnArg`
- 3 ops (sigmoid, tanh, exp) circumvent this by storing their own output NDBuffer in the payload — their backward reads from the payload, not from `output.buffer()`
- All backward handlers are in separate modules, imported via `walkback.mojo`

### Example: `SumBackward` (`tenmo/sum_reduction.mojo`)

```mojo
def backward(
    output: Ancestor[Self.dtype],
    mut parent_ids: List[UInt],
    retain_graph: Bool = False,
):
    # 1. Extract reduction parameters from BackwardFnArg
    ref bwd_arg = output.ancestry().backward_fn_arg().get[ReductionArg]()
    var (axes, keepdims) = bwd_arg.axes, bwd_arg.keepdims

    # 2. Get the gradient and original shape
    ref gradbox = output.gradients()
    var ancestor = output.ancestry().get(0)
    ref shape = ancestor.shape()

    # 3. Broadcast gradient back to original shape
    if gradbox.shape() == Shape():  # scalar case
        grad_contrib = Gradbox.full(shape, gradbox.item(), device=...)
    else if not keepdims:
        # Unsqueeze reduced dimensions back to size 1
        axes = gradbox.shape().intarray().insert(axes, IntArray.filled(len(axes), 1))
        unsqueezed_shape = Shape(axes)
        unsqueezed_grad = gradbox.reshape(unsqueezed_shape)
        grad_contrib = unsqueezed_grad.broadcast_to(shape)
    else:
        grad_contrib = gradbox.broadcast_to(shape)

    # 4. Accumulate into parent's gradbox and register parent for fanin
    if ancestor.requires_grad:
        ancestor.update_grad(grad_contrib^, AddTensor, None)
    parent_ids.append(ancestor._id)

    # 5. Zero gradient if not retaining graph
    if not retain_graph:
        gradbox.zero_grad()
```

### Ops That Store Output in BackwardFnArg Payload

Most ops store only operation metadata in the payload (axes, keepdims, shape, etc.). Three ops store their **output NDBuffer** because their backward handler needs the forward output values:

| Op | Payload Type | Why |
|---|---|---|
| Sigmoid | `NDBufferArg` | Backward formula: `grad * out * (1 - out)` — needs sigmoid output |
| Tanh | `NDBufferArg` | Backward formula: `grad * (1 - out²)` — needs tanh output |
| Exp | `NDBufferArg` | Backward formula: `grad * out` — needs exp output |

These ops set `needs_parent_data=False` (they don't need their parent's ndb). The output ndb is stored during forward via `from_ndbuffer()`:

```mojo
var out_ndb = out.buffer.copy()
var backwardFnArg = BackwardFnArg[Self.dtype].from_ndbuffer(
    BACKWARD_SIGMOID, out_ndb^
)
out.add_ancestry(backwardFnArg^, self)
```

The backward handler reads from the payload:
```mojo
ref bwd_arg = output.ancestry().backward_fn_arg().get[NDBufferArg[Self.dtype]]()
var out_ndb = bwd_arg.ndb
var ndb = out_ndb.arithmetic_ops[SIGMOID_BACKWARD](gradbox.buffer())
```

This is identical to how ReLU stores its mask and Softmax stores its output via `SoftmaxArg`.

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
- Mojo pinned to `==1.0.0b2` from `conda.modular.com/max-nightly`
- Python 3.10–3.14
- PyPI deps: `mnist-datasets`, `pure-cifar-10`, `tiktoken`
- Dev feature includes `mojodoc` and a local `bpe` at `../bpe`
- `pixi run docs` — generate docs via mojodoc

## BLAS

Tenmo's BLAS integration is in `tenmo/blashandle.mojo`. At compile time it looks up
`BLAS_PATH` via:

```mojo
comptime BLAS_PATH = get_defined_string["BLAS_PATH", "/lib/x86_64-linux-gnu/libopenblas.so.0"]()
```

The second argument is the **fallback default** — used when `-D BLAS_PATH=...` is not
passed on the command line.

### Option 1 — System OpenBLAS (default, works out of the box)

```bash
sudo apt-get update
sudo apt-get install -y libopenblas-dev
```

This installs to `/lib/x86_64-linux-gnu/libopenblas.so.0` which is the default fallback
path — no `-D` flag needed.

### Option 2 — Pixi-managed OpenBLAS (conda)

```bash
pixi add openblas
# pixi.toml entry (auto-generated): openblas = ">=0.3.33,<0.4"
find $CONDA_PREFIX/lib -name "libopenblas.so" | head -1
```

Then pass the path explicitly:

```bash
mojo -I . -D BLAS_PATH=$HOME/tenmo/.pixi/envs/default/lib/libopenblas.so tests/test_cpu_all_3.mojo
# or with auto-discovery:
mojo -I . -D BLAS_PATH=$(find $CONDA_PREFIX/lib -name "libopenblas.so" | head -1) tests/test_cpu_all_3.mojo
```

### Runtime Behaviour

- `SequentialBLAS` with `LinearBLAS` layers auto-profile native Mojo vs BLAS matmul
  at runtime on the first forward call, then select the faster path.
- Full backward pass support through BLAS (gradient flows through BLAS matmul).

## Style Conventions

- Mojo structs use `snake_case` for methods and fields
- `fn main() raises:` is the standard entrypoint pattern (tests use `raises`)
- `comptime` blocks for compile-time branching (e.g. `comptime if track_grad:`)
- `^` suffix for owned value transfer (e.g. `out.add_ancestry(backwardFnArg^, self)`)
- `ref` for borrow references (e.g. `ref gradbox = output.gradients()`)

## Gradbox Testing Convention

**Gradboxes are shared at birth** — `Gradbox.__init__` always calls `.share()` on the underlying NDBuffer's Buffer so that references survive Mojo's ASAP destruction.

**`Tensor.grad()` returns an independent deep copy** — it calls `Gradbox.detach()` internally, which allocates fresh CPU or GPU storage and copies data. Slicing or indexing the returned Gradbox is safe:

```mojo
var grad = tensor.grad()   # already a detached deep copy
assert_true(grad[i(0), s()].all_close(Tensor[dtype].d1([1.0, 1.0])))
```

`Gradbox.detach()` (`gradbox.mojo:136`) handles both CPU and GPU:
- CPU: `NDBuffer.contiguous_buffer()` — allocates `Buffer(size)` + `memcpy`
- GPU: `NDBuffer.contiguous_device_state()` — fresh `DeviceState` + GPU→GPU copy

No `.copy()` call is needed after `.grad()` — it's already independent. Use `.copy()` only when you need a second handle to the *same* independent data.

## Autograd & Optimizer Notes

### `SGD[dtype: DType, //]` — Witness Parameter

The `//` after `dtype` means it is a **witness** parameter — it **is** auto-inferred from `__init__` arguments:

```mojo
var opt = SGD(params, lr=0.01)                           # dtype auto-inferred — OK
var opt = SGD[DType.float32](params, lr=0.01)            # explicit positional — OK
var opt = SGD[dtype=DType.float32](params, lr=0.01)      # explicit keyword — OK
```

Mojo deduces `dtype` from the element type of the parameter list (e.g., from the embedding weight's dtype). Witness parameters configure the struct at compile time and are derivable from runtime constructor args.

### `.backward()` on Non-Scalar Tensors

`tensor.backward()` works on **any** tensor, not just scalars. When called on a non-scalar tensor `C` of shape `(N,)`, it computes `d(sum(C))/d(leaf)` — equivalent to `C.sum().backward()` in PyTorch.

```mojo
var A = Tensor[dtype].rand(shape)
var B = Tensor[dtype].rand(shape)
var C = A.matmul[mode=mv](B)          # shape (N,) — non-scalar
C.backward()                          # OK — computes d(sum(C))/d(A), d(sum(C))/d(B)
```

This is convenient for element-wise losses: `loss.backward()` without an explicit `.sum()`.

### `Embedding.from_pretrained(weights, freeze=True)`

Loads a pretrained weight matrix (e.g. GloVe, fastText, word2vec) into an `Embedding`:

```mojo
var emb = Embedding[dtype].from_pretrained(glove_weights)       # frozen — no gradient
var emb = Embedding[dtype].from_pretrained(glove_weights, freeze=False)  # fine-tuned
```

- **`freeze=True` (default):** `weight.requires_grad = False`. Embeddings act as fixed lookups — no gradients, no SGD updates. Use for feature extraction where pretrained vectors should stay intact.
- **`freeze=False`:** `weight.requires_grad = True`. Gradients flow into the weight during `.backward()`. `SGD.step()` updates the pretrained vectors. Use for domain adaptation or fine-tuning.

Args:
- `weights`: Pretrained `Tensor` — must be 2D `(vocab_size, embedding_dim)`.
- `padding_idx`: Row to keep zeroed (never receives gradient).
- `freeze`: If `True` (default), `requires_grad=False` — no gradient flow.

## Lessons from Sparse Embedding Training

### Manual `scatter_add` vs Autograd `Embedding` + `SGD`

Two approaches were implemented for word2vec-style negative sampling training on IMDB (252K vocab, 100 hidden dim):

| Approach | Per-step cost | Time for 5K reviews × 2 iters |
|---|---|---|
| **Manual** (gather[track_grad=False], manual backward, Filler.scatter_add) | ~3.6 ms | ~2 hours |
| **Autograd** (Embedding, BCEWithLogitsLoss, .backward(), SGD.step()) | ~60 ms | too slow — aborted |

**Why autograd is 16× slower:**

`SGD.step()` iterates over **all** 25M weight parameters (252K × 100 × 2) on every training step. For word2vec, only ~10 rows (1000 elements) actually changed per step — a 25000× waste. The manual `scatter_add` touches only the rows that received gradient updates.

**When to use each approach:**

- **Dense architectures** (MLP, CNN, Transformer in full-batch): every parameter updates each step → `SGD.step()` cost is amortized → autograd + optimizer is appropriate.
- **Sparse lookup tasks** (word2vec, large-vocab embeddings): most rows are untouched each step → manual `scatter_add` is the correct pattern.

**Current Tenmo limitation:** No sparse optimizer exists. `SGD.step()` always touches every parameter, even if only a few rows have non-zero gradients.

## Performance Analysis

See [`PERFORMANCE_BOTTLENECKS.md`](PERFORMANCE_BOTTLENECKS.md) for a ranked
list of all performance bottlenecks across GPU kernels, autograd, memory
management, and SIMD paths — with root causes and fix suggestions for each.

See [`GPU_SYNCHRONIZATION.md`](GPU_SYNCHRONIZATION.md) for a complete map of
all GPU sync call sites, the kernel launch pipeline, compound operations, CPU-
GPU transfer sync, correctness model, and a phased optimization plan.

## CpuArithmeticOps — CPU Arithmetic Dispatch

All CPU per-element arithmetic is centralized in `CpuArithmeticOps[dtype]` at `tenmo/cpu_arithmetics.mojo:40`. It supersedes the old `CpuBroadcast` struct and handles three categories:

| Method | Line | Description |
|---|---|---|
| `compute[op_code](a, b, epsilon?)` | 408 | Element-wise binary op, same-shape or scalar-broadcast |
| `broadcast[op_code](a, b, epsilon?)` | 52 | Entry point: dispatches to `broadcast_scalar` or `broadcast_nd` |
| `broadcast_scalar[op_code](a, b, epsilon?)` | 92 | SIMD-splat for scalar-size operands |
| `broadcast_nd[op_code](a, b, epsilon?)` | 170 | ND broadcast with three-tier dispatch |
| `unary_ops[op_code](a)` | 507 | Element-wise unary ops |
| `unary_ops_constrained[op_code](a, low, high)` | 531 | Clamp/clip unary ops |

`epsilon` (for divide stability) is threaded through all broadcast/compute paths — see `tenmo/ndbuffer.mojo` callers for usage.

### `broadcast_nd` — Three-Tier Dispatch

`CpuArithmeticOps.broadcast_nd` (`cpu_arithmetics.mojo:170`) handles ND-broadcast arithmetic (both operands non-scalar, different shapes):

1. **Both unit-stride in last dim** (`a_eff[-1]==1 AND b_eff[-1]==1`) — SIMD-SIMD tile inner dimension. Reads `simd_width` consecutive elements from both buffers, vector op, vector store. Outer dims iterate via odometer.
2. **One broadcasts, one unit-stride** (`a_eff[-1]==1 AND b_eff[-1]==0`, or vice versa) — splat scalar from broadcasting side, SIMD-load from contiguous side, vector op, vector store.
3. **Scalar odometer** — per-element loop with incremental offset updates via effective strides. No `translate_index`/`flatten_index` overhead.

#### Correctness guarantees (generically proven by effective strides)

For any valid broadcast pair `(shape_a, shape_b)` producing `result_shape`:
- Each output coordinate maps to `base_a + Σ coord[d] × eff_stride_a[d]` in operand A, where `eff_stride[d] = 0` if dim `d` is broadcast, else original stride.
- All three tiers evaluate the same mapping — only iteration strategy differs.
- Result is allocated contiguous; output offset is always the flat linear index.

#### Performance boundaries (where Path 3 applies)

| Condition | Cause |
|---|---|
| `a_eff[-1] != 1 AND b_eff[-1] != 1` | Neither operand has unit stride in last dim (e.g. transposed views, both broadcast in last dim) |
| `dtype == DType.bool` | `simd_width` forced to 1 |
| `last_dim < simd_width` | Remainder loop — implicit, correct |
| Uncommon op codes (SIGMOID_BACKWARD, POW, etc.) | Fall through to scalar-per-tile in inner SIMD loop |

None of these affect correctness — only throughput. See `tests/bench_broadcast.mojo` for benchmark patterns.

### In‑place Operations

In‑place CPU ops (`+=`, `*=` etc.) live on **`CpuArithmeticOps`** as `inplace_ops` / `inplace_scalar_ops` at `cpu_arithmetics.mojo:557/633`. The `@staticmethod` parameter `self: NDBuffer[Self.dtype]` is a value copy, but `Buffer` owns its data through a refcounted pointer — writing to `self.buffer[index]` mutates the shared memory. `NDBuffer.inplace_ops` in `ndbuffer.mojo:1704` dispatches GPU paths and falls back to `CpuArithmeticOps.inplace_ops` for CPU. The broadcast sub-path within inplace reuses `CpuArithmeticOps.broadcast` (returns a new buffer), then writes the result via `copy_from_alike`.

## ⛔ CRITICAL: NEVER launch multiple tests in parallel — Mojo compilation is memory-bound and will crash the machine. Run ONE `./execute.sh` or `./fire.sh` at a time, wait for it to complete, then run the next.

## Running Selective Tests

**`./execute.sh <name>`** — runs a single test alias from the test runner (e.g. `./execute.sh gather`)

**`./fire.sh <file.mojo>`** — runs any `.mojo` file directly with `mojo -I .` (e.g. `./fire.sh test_gather_memcpy_new.mojo`)

To run specific test functions within a file, the `TestSuite` discovers all `fn test_*` functions automatically. You can selectively run tests using CLI arguments:
- `mojo run <file.mojo> --only test_foo test_bar` — run only specified tests
- `mojo run <file.mojo> --skip test_slow test_flaky` — skip specified tests
- See `TestSuite` documentation for full CLI filtering capabilities.

## Mojo 1.0.0b1 → 1.0.0b2 Migration Notes

### Breaking Changes (compile errors)

- **`fn` is now an error** — Use `def` instead of `fn` for all function/struct method declarations and function pointer types (`fn(...) thin` → `def(...) thin`).
- **`Movable.__init__` argument renamed `take` → `move`** — All `def __init__(out self, *, deinit take: Self)` must become `def __init__(out self, *, deinit move: Self)`. Internal `take.xxx` references must become `move.xxx`.

### Deprecation Warnings (still compiles, fix when modifying nearby code)

- **`compile_function[func, func]()` → `compile_function[func]()`** — Single-arg form replaces the old double-arg pattern. 86 call sites in 31 kernel files.
- **`as_any_origin()` → `as_unsafe_any_origin()`** — 17 calls in `net.mojo`, `layernorm.mojo`, `embedding.mojo`.
- **`unsafe_origin_cast[MutAnyOrigin]()` → `unsafe_origin_cast[MutUnsafeAnyOrigin]()`** — 36 calls across 8 files.
- **`MutAnyOrigin`/`ImmutAnyOrigin` → `MutUnsafeAnyOrigin`/`ImmutUnsafeAnyOrigin`** — Hundreds of call sites. Slated for future deprecation; compiles silently in b2.
- **`ExternalOrigin`/`MutExternalOrigin`/`ImmutExternalOrigin` → `UntrackedOrigin`/`MutUntrackedOrigin`/`ImmutUntrackedOrigin`** — Already migrated (0 hits in codebase).

### Persistent b1 Notes (still relevant)

- **No `address_of`** — Use `UnsafePointer(to=...)` or `Pointer(to=...)` instead.
- **`alloc[T](count)` returns `UnsafePointer[T, MutAnyOrigin]`**
- **`.unsafe_value()` not `.value()`** for `Optional[UnsafePointer[...]]`
- **`def(...) thin` function pointers** — Replace broken `fn(...)` types. Named functions only (no lambdas).
- **Atomic API** — `from std.atomic`, `Atomic[DType.uint64]`, `Ordering.RELAXED`/`RELEASE`/`ACQUIRE`.
- **`UnsafePointer(...)()` null constructor removed** — Use `Optional[UnsafePointer[...]]` with `{}`.
- **`alias` is deprecated** — Use `comptime` instead (already fully migrated).
- **`@parameter` decorator is deprecated** — Use `comptime if` / `comptime for` directly (already fully migrated).

## Known Issues / Future Work

### Broadcast Dispatch Bug Fix

**Location:** `tenmo/kernels/binary_ops_kernel.mojo:556,592`

**Bug:** PATH 2 and PATH 3 in the broadcast arithmetic dispatch checked
`not B_is_contiguous` / `not A_is_contiguous` instead of
`B_shape != broadcast_shape` / `A_shape != broadcast_shape`. Every broadcast
operation (bias_add, all crossentropy sub-ops, layer norm, etc.) fell through
to the general `both_strided` fallback PATH 4.

**Impact:** 3 bias_add operations in MNIST MLP (784→128→32→10, batch=64) went
from expected ~0.05ms each to ~6.5ms, ~1.2ms, ~1.2ms. ~9ms/batch waste on GPU.

**Fix applied:** Two-line change. After fix, bias_add dispatches through PATH 2
(one operand unit-stride, one broadcasts) using SIMD-splat for the scalar side.

### CrossEntropy GPU Path — Planned Fused Kernel

**Problem:** `CEClassIndicesForward.forward` on GPU triggers ~18 separate
kernel launches + 1 CPU-fallback onehot loop for a single crossentropy call.
Profiling shows 29ms per (64, 10) call — dominated by kernel launch overhead
and CPU-fallback onehot (64 per-element DeviceState round trips).

**Fix (planned):** Write a fused forward kernel at
`tenmo/kernels/crossentropy_fused_kernel.mojo` that computes max, exp,
sum_exp, softmax, log_softmax_target, and per-sample loss in 1 launch:
- Thread-block-per-row pattern (M blocks)
- Shared-memory tree reduction for max and sum_exp
- `Atomic.fetch_add` for scalar_loss and valid_count
- Handles ignore_index and label_smoothing in-kernel
- Replaces the existing ~18 kernel decomposition + CPU onehot entirely

Backward stays unchanged — 4 GPU arithmetic ops using stored softmax.

### Onehot Missing GPU Kernel

`NDBuffer.onehot` at `ndbuffer.mojo:411` has no GPU path. For GPU-resident
tensors, it iterates coordinates on CPU and writes elements via per-element
`DeviceState.__setitem__()` — each a `map_to_host()` round trip. The fused
crossentropy kernel eliminates this bottleneck by computing NLL in-kernel
instead of via onehot masking.

### GPU `scatter_add` only supports axis=0 — Fixed

GPU kernels `scatter_add_rows_kernel` and `scatter_add_broadcast_kernel` (`tenmo/kernels/filler_kernel.mojo`) compute target flat indices assuming row-major layout where `indices` pick rows (axis=0). For axis != 0, `Filler.scatter_add` (`tenmo/filler.mojo:138`) fell back to `_scatter_add_cpu`, which accessed GPU memory element-by-element via `device_state[idx]` — correct but very slow.

**Fix:** Added `scatter_add_nd_kernel` (`tenmo/kernels/filler_kernel.mojo:120`) — a general N-dimensional GPU kernel that accepts shape and strides arrays and uses the same coordinate decomposition as the CPU path. `FillerGpu._scatter_add_nd_gpu` copies shape/strides to GPU device buffers and dispatches the kernel with one block per index.

**Dispatch:** `Filler.scatter_add` routes axis != 0 through the new general kernel; axis == 0 still uses the existing fast path. All 5 new GPU gather backward tests for axis=1 and axis=2 pass.

### GPU Multiply Broadcast Backward — Two Bugs Fixed

**Location:** `tenmo/broadcast.mojo:90-113` (`upstream_grad_share`), `tenmo/kernels/binary_ops_kernel.mojo` (GPU broadcast kernel dispatch), `tenmo/multiplication.mojo:85-90` (`MultiplyBroadcastBackward`)

**Symptom:** 7 GPU multiply broadcast backward tests failed originally.

**Bug 1 — Gradbox one-liner destruction race (fixed):**

The one-liner `grad_contrib = Gradbox[Self.dtype](upstream_grad * other)` destroyed the temporary NDBuffer (Mojo ASAP) before `Gradbox.__init__` finished copying data from the GPU buffer. Fixed by using an intermediate variable:

```mojo
var product_ndb = upstream_grad * other
grad_contrib = Gradbox[Self.dtype](product_ndb^)
```

**Bug 2 — GPU broadcast arithmetic kernel indexing error (fixed):**

Root cause: the broadcast operand's GPU `DeviceBuffer` was freed and reused before the kernel read it, returning stale data. The multiply kernel was queued asynchronously, but the parent tensor's GPU buffer (shared via `ancestor.ndb`) was freed by Mojo's ASAP destruction before the kernel actually executed. Fix: ensure the parent NDBuffer's GPU device state survives through the kernel launch by extending the lifetime of the broadcast operand.

**All 7 GPU multiply broadcast backward tests now pass.**

### CE Probability-Target GPU Forward — CPU Fallback (Technical Debt)

**Location:** `tenmo/crossentropy.mojo:1141-1152` (`CEProbabilitiesForward.forward`)

**Debt description:** `CEProbabilitiesForward.forward` has no proper GPU path. When inputs are GPU-resident, the fix at lines 1146-1152 transfers `logits` and `target` from GPU → CPU, then calls the CPU-only `_fused_forward_probabilities`. The backward is unaffected — it uses `arithmetic_ops[Subtract]` which dispatches correctly to CPU.

**Root cause:** `_fused_forward_probabilities` (line 468) is entirely CPU-native: it accesses logits via `data_ptr()[offset]` for raw pointer arithmetic during max/exp/sum/softmax computation. NDBuffers on GPU have `device_state` set and `buffer` empty (zero-size). Calling `data_ptr()` on a GPU NDBuffer through the CPU fused function dereferences the empty CPU Buffer — segfault at line 523.

**Contrast with `CEClassIndicesForward.forward`:** That path has proper GPU dispatch using `CrossEntropyFusedKernel` (class indices only). The class-indices GPU kernel was written because it's straightforward — a single scalar index per row. Probability targets require element-wise operations on the full `(M, C)` target matrix, which is a fundamentally different kernel.

**Cost breakdown (accruing):**
- Each GPU cross-entropy call with probability targets triggers:
  1. `to_cpu(sync=True)` — GPU→CPU transfer of `(M, C)` logits
  2. `to_cpu(sync=True)` — GPU→CPU transfer of `(M, C)` targets
  3. CPU fused forward — O(4MC) flops on CPU
  4. Backward `arithmetic_ops[Subtract]` — CPU ops (but backward was already CPU for the forward output)
  5. Result consumed on CPU — must be transferred back to GPU via implicit `to_gpu()` if chained with GPU ops
- Total: 2 full GPU→CPU transfers + 1 CPU forward + 1 CPU backward + 1 implicit CPU→GPU transfer ≈ O(3MC) cross-device bandwidth + O(4MC) CPU flops
- By contrast, `CEClassIndicesForward` runs entirely on GPU: O(MC) GPU flops, zero cross-device transfers.

**Justification for a separate GPU kernel:**

Probability-target CE cannot reuse the existing `CrossEntropyFusedKernel` (`tenmo/kernels/crossentropy_fused_kernel.mojo:279`) — that kernel is designed for class indices (`target_1d: NDBuffer[DType.int32]`):

```mojo
# CrossEntropyFusedKernel — class indices only
target_value = target_buffer[row * target_stride0 + col * target_stride1]
```

Probability targets need a per-element kernel where each class dimension `c` is a separate target value. The fused kernel structure (thread-block-per-row, shared-memory tree reduction for max and sum_exp, register-level log_softmax) can be adapted, but:

1. **Input format:** Instead of `target_1d: NDBuffer[DType.int32]`, the kernel must accept `target_2d: NDBuffer[Self.dtype]` (same dtype as logits).
2. **NLL computation:** Instead of `nll = -log_softmax[row, target[row]]`, compute `nll_row = Σₛ target[row, c] * log_softmax[row, c]` — requires a second reduction tree (sum over classes) per row.
3. **Gradient computation:** Backward for probability targets is `softmax - target` — a per-element subtraction, same formula regardless of target type. The existing backward via `arithmetic_ops[Subtract]` is already correct.
4. **Label smoothing with probabilities:** The CPU function applies label smoothing in-kernel by blending targets with uniform distribution. The GPU kernel must also handle this — either accept a `smoothed_target` pre-computed or blend in registers.

**Priority:** Low. The CPU fallback is functionally correct. Fix when:
- Probability-target CE becomes a training bottleneck on GPU (profile: if `to_cpu` + CPU forward dominates step time).
- A user requests GPU probability-target CE for large-scale training.

**Implementation sketch for the GPU kernel:**

```mojo
struct CEProbabilityFusedKernel[dtype: DType, max_block_size: Int]:
    @staticmethod
    def launch(
        logits: NDBuffer[dtype],        # (M, C), GPU-resident
        target: NDBuffer[dtype],         # (M, C), GPU-resident
        label_smoothing: Scalar[dtype],  # blending factor
        device_context: GPU,
    ) raises -> Tuple[NDBuffer[dtype], NDBuffer[dtype], NDBuffer[dtype]]:
        # Returns: softmax_out, smoothed_target (for backward), loss (M,)

    # Per-row kernel:
    # 1. Shared-memory tree reduction for max over logits[row, :]
    # 2. Shared-memory tree reduction for sum_exp = Σ exp(logits[row, c] - max)
    # 3. For each c: log_softmax = logits[row, c] - max - log(sum_exp)
    # 4. If label_smoothing: blend target with uniform
    # 5. Second tree reduction: loss_row = Σ target[row, c] * log_softmax[c]
    # 6. Write loss_row to output, softmax_out[row, :] = exp(log_softmax)
    # 7. Write smoothed_target[row, :] to output for backward
```

**Blocks:** None. Code compiles, all 139 CE tests pass (including the 3 GPU-guarded prob tests which now skip gracefully with the CPU fallback instead of crashing).

### GPU Capability Gaps — CPU-Only Operations

These operations have no GPU dispatch path. When called on GPU-resident tensors they either crash (dereference empty CPU buffer), produce wrong results, or silently fall back to slow per-element CPU round trips. Each is a numbered action item.

| ID | Operation | File | Status | Root Cause |
|----|-----------|------|--------|------------|
| KI-01 | **Concate / Stack / vstack / hstack** | `concate.mojo:136`, `stack.mojo:19` | ✅ Done | Fused GPU copy kernel (`ConcateGpuKernel`) with coordinate decomposition. Forward + backward for all axes. Stack uses same kernel via temp unsqueeze + contiguous squeeze. |
| KI-02 | **Pad** (constant mode) | `pad.mojo:369`, `pad_kernel.mojo` | ✅ Done (GPU tests added) | Fused GPU copy kernel (`PadConstantGpuKernel`) with coordinate decomposition via `Array`. Forward + backward for constant mode. Replicate/reflect/circular still CPU. |
| KI-03 | **Conv2D** (Conv2dFused) | `cnn.mojo` | ❌ No GPU path (WIP) | `FusedIm2Col` uses `data_ptr()` for all reads/writes. SIMD + `parallelize`. Documented WIP |
| KI-04 | **MaxPool2d** | `pooling.mojo` | ❌ No GPU path | Pooling kernels use `data_ptr()` with stride arithmetic + `parallelize`. No GPU guard |
| KI-05 | **Tensor.rand** | `tensor.mojo:1301` | ❌ No GPU allocation | No `device` parameter. Creates CPU `Buffer` and fills with `random_float64` in a `for` loop |
| KI-06 | **Tensor.randn** | `tensor.mojo:1360` | ❌ No GPU allocation | No `device` parameter. CPU-only Box-Muller via `NDBuffer.randn` |
| KI-07 | **Tensor.arange / linspace** | `ndbuffer.mojo` | ❌ No GPU allocation | `NDBuffer.arange`/`linspace` create CPU `Buffer` directly, no device parameter |
| KI-08 | **Tensor.onehot** | `ndbuffer.mojo:411` | ❌ No GPU path (known) | CPU coordinate iteration with per-element `DeviceState.__setitem__` — each write is a `map_to_host()` round trip |
| KI-09 | **Tensor.eye** | `tensor.mojo` | ⚠️ Partial | Allocates on GPU via `device` param, but fill loop (`for i: out[i,i]=1`) hits per-element CPU round trips; no GPU kernel |

**Priority ordering:** KI-01 (Concate/Stack) blocks any GPU training path that gathers per-sample outputs into batches. KI-02 (Pad) blocks Conv2D on GPU. KI-03/KI-04 are WIP. KI-05/KI-06/KI-07 are convenience — workaround is `.to_gpu()` after construction. KI-08/KI-09 are slow but functional.

## Bias Implementation — `Linear` / `LinearBLAS` / `Conv2D`

### Status

- `Conv2D` uses **runtime `Bool` + sentinel** (0-rank `Tensor.scalar(0)`) — implemented.
- `Linear` / `LinearBLAS` — **not yet implemented** (pending decision).
- This section documents the full design conversation so the rationale is not lost.

### Problem

Transformers and modern architectures often eliminate bias from linear/attention layers entirely (e.g. GPT-2, Llama, most post-2020 ViTs). When a user writes `Linear(in_features, out_features, bias=False)`, we need:

1. **No bias storage** — don't allocate memory for bias weights.
2. **No bias computation** — skip the bias-add kernel entirely.
3. **No gradient flow** — don't allocate a gradbox for the bias.
4. **Clean introspection** — `parameters()` returns only the weight; `num_parameters()` counts only the weight.

### Three Approaches Considered

#### Approach A: Runtime `Bool` + sentinel 0-rank Tensor (Conv2D's current pattern)

```mojo
def __init__(..., bias: Bool = True):
    if bias:
        self.bias = Tensor[Self.dtype].zeros(Shape(out_features), requires_grad=True)
    else:
        self.bias = Tensor[Self.dtype].scalar(0)  # 0-rank, requires_grad=False

def __call__(...):
    result = matmul_out^
    if self.bias.shape().rank() > 0:
        result = Adder.forward(result^, self.bias)
```

**Pros:**
- Zero changes outside the struct — `Layer` Variant, `Module` dispatch, TAGs, `Sequential`, `SequentialBLAS` all untouched.
- ~12 lines of changes, all in `net.mojo`.
- Same pattern as `Conv2D` — consistent.

**Cons:**
- **Dummy allocation** — `Tensor.scalar(0)` still allocates a 1-element `Buffer`, takes a full `Tensor` slot in the struct, and participates in `to_gpu()`/`to_cpu()` transfers (1 element, negligible but inelegant).
- **Runtime guards** — every access (forward, `parameters()`, `num_parameters()`, transfer) requires `if self.bias.shape().rank() > 0:`.
- `num_parameters()` must subtract the 1 from the sentinel scalar: `self.weight.numels() + (self.bias.numels() if self.bias.shape().rank() > 0 else 0)`.

---

## ⚠️ Empirical Finding: Mojo Cannot Conditionally Declare Fields

**Tested 2026-06-15.** The following was confirmed to fail at compile time:

```mojo
struct Foo[use_extra: Bool]:
    var always: Int
    comptime if use_extra:     # ERROR: recursive reference to declaration
        var extra: Int
```

Both `comptime if` and `@parameter if` inside struct body fail with "attempt to resolve a recursive reference to declaration." A field always exists in the struct layout regardless of comptime parameters. Only method bodies and inline expressions support `comptime if` branching.

**Implication for bias design:** No approach can eliminate the bias field at compile time. Phase 1 (runtime `Optional[Tensor]`) and Phase 2 (comptime `use_bias`) both have the field present as `Optional[Tensor]`. The comptime flag can only eliminate *guards* (the `if self.bias:` branches), never the storage itself.

---

#### Approach B: Runtime `Optional[Tensor]` (Phase 1)

```mojo
var bias: Optional[Tensor[Self.dtype]]

def __init__(..., bias: Bool = True):
    if bias:
        self.bias = Optional(Tensor[Self.dtype].zeros(Shape(out_features), requires_grad=True))
    else:
        self.bias = None  # no allocation at all

def __call__(...):
    result = matmul_out^
    if self.bias:   # self.bias.is_some()
        result = Adder.forward(result^, self.bias.value())

def parameters(...):
    params.append(weight_ptr)
    if self.bias:
        params.append(bias_ptr)

def num_parameters():
    count = self.weight.numels()
    if self.bias:
        count += self.bias.value().numels()
    return count
```

**Pros:**
- **Zero allocation** when `bias=False` — no `Tensor`, no `Buffer`, no `gradbox`.
- Guards are idiomatic `if self.bias:` — no `shape().rank() > 0` hack.
- `Optional` tag byte adds ~1 byte to the struct (negligible).
- Same changeset as Approach A — ~12 insertions.
- No changes to `Layer`, `Module`, `Sequential`, TAGs, etc.
- API unchanged: `Linear(in, out, bias=False)` still works.

**Cons:**
- `Optional` access via `.value()` requires an extra call vs direct field access.
- Still requires runtime `if` branches (same as sentinel approach).
- Bias field is always present in the struct (as `Optional[Tensor]`), not eliminated at compile time — but Mojo can't eliminate it anyway.

#### Approach C: Comptime `use_bias` + `Optional[Tensor]` (Phase 2)

Since Mojo cannot eliminate fields, the comptime approach controls *guards only*:

```mojo
struct Linear[dtype: DType, mode: Int = mm, use_bias: Bool = True]:
    comptime TAG = LINEAR if use_bias else LINEAR_NO_BIAS

    var weight: Tensor[Self.dtype]
    var bias: Optional[Tensor[Self.dtype]]    # always Optional, can't eliminate

    def __init__(..., bias_zero: Bool = True):
        # No runtime `bias` param — bias is a type parameter
        comptime if use_bias:
            if bias_zero:
                self.bias = Optional(Tensor[...].zeros(Shape(out_features), requires_grad=True))
            else:
                self.bias = Optional(Tensor[...].rand(Shape(out_features), ..., requires_grad=True))
        else:
            self.bias = None

    def __call__(...):
        var result = Matmul.forward(...)
        comptime if use_bias:
            # No `if self.bias:` needed — always Some when use_bias=True
            result = Adder.forward(result^, self.bias.value())

    def parameters(...):
        params.append(weight_ptr)
        comptime if use_bias:
            params.append(bias_ptr)   # guard eliminated at compile time
```

**Pros:**
- **Zero runtime overhead** — all bias guards are `comptime if` branches, eliminated from the binary when `use_bias=False`.
- `Optional[Tensor]` when `use_bias=False` is always `None` and no code reads it — dead data.
- Zero allocation + zero instruction overhead for bias=False models (Transformers).
- Field type is always `Optional[Tensor]` — same struct shape regardless of `use_bias`.

**Cons — the comptime cascade:**

The type signature gains a comptime parameter: `Linear[dtype, mode]` → `Linear[dtype, mode, use_bias]`. API changes from `Linear(in, out, bias=False)` to `Linear[dtype, mode, use_bias=False](in, out)`.

This cascades through every layer that references a concrete `Linear`:

| Layer | Impact |
|---|---|
| `Layer` Variant (`net.mojo:772`) | Must list both `Linear[dtype, mm, True]` and `Linear[dtype, mm, False]`, same for `LinearBLAS` and `Conv2D`. Doubles the Variant entries for each. |
| TAG constants (`mnemonics.mojo:33-43`) | +2 new constants per struct (`LINEAR_NO_BIAS`, `LINEAR_BLAS_NO_BIAS`, `CONV2D_NO_BIAS`). Every subsequent constant shifts — check all `mnemonics.mojo` consumers. |
| `Module.__call__` (`net.mojo:792`) | +2 branches per struct: `elif tag == LINEAR_NO_BIAS: return self.layer[Linear[Self.dtype, mm, False]](xs)` |
| `Module.parameters()` (`net.mojo:822`) | +2 branches |
| `Module.named_parameters()` (`net.mojo:839`) | +2 branches |
| `Module.num_parameters()` (`net.mojo:853`) | +2 branches |
| `Module.train()` (`net.mojo:885`) | +2 branches |
| `Module.eval()` (`net.mojo:910`) | +2 branches |
| `Module.to_gpu()` (`net.mojo:935`) | +2 branches |
| `Module.to_cpu()` (`net.mojo:981`) | +2 branches |
| `Sequential.append()` (`net.mojo:1011`) | Guards against `LINEAR_BLAS` — must also check `LINEAR_BLAS_NO_BIAS` |
| `SequentialBLAS.append()` (`net.mojo:1137`) | Same guard expansion |
| Every `layer[Linear[Self.dtype]]` extraction | Must specify `layer[Linear[Self.dtype, mm, True]]` vs `layer[Linear[Self.dtype, mm, False]]` — must match TAG exactly or panic at runtime. |
| Tests (`test_cpu_all_3.mojo`, `test_checkpoint.mojo`) | `layer[Conv2D[DType.float32]]` → `layer[Conv2D[DType.float32, True]]` (or change to `False`). |
| Examples (`mnist_conv2d.mojo`) | Construction syntax changes: no `bias=` kwarg. |

**Rough change count: ~30+ lines per struct across `net.mojo` + `mnemonics.mojo` + tests/examples.** For all three (Linear, LinearBLAS, Conv2D): ~80-100 total.

The key combinatorial problem: **every comptime knob on `Linear` doubles the Variant entries and the `Module` dispatch branches.** If we later add `use_layer_norm: Bool`, `use_scale: Bool`, etc., the Variant explodes exponentially.

---

### Two-Phase Implementation Plan

| Phase | What | Cost | Benefit | API Changes |
|---|---|---|---|---|
| **1** | Runtime `Optional[Tensor]` + runtime `bias: Bool` | ~11 changes per struct, all local | Zero allocation for `bias=False` | None |
| **2** | Comptime `use_bias` on top of Phase 1 | ~30+ changes per struct + cascade | Zero runtime branches for `bias=False` | `bias=` kwarg → type param |

### Decision Record (2026-06-15)

| Decision | Choice |
|---|---|
| **Phase 1 implementation** | Approach B — `Optional[Tensor]` with runtime `bias: Bool` |
| **Phase 2 goal** | Approach C — comptime `use_bias` controlling `Optional[Tensor]` guards |
| **Trigger for Phase 2** | When a Transformer block is written and bias elimination matters for correctness (not just performance). At that point, approach B → C is a mechanical transform within each method. |

**Justification for Phase 1 now (Approach B):**

1. **Same changeset size** as the sentinel hack (~11 lines) — no extra work.
2. **Zero allocation** — `Optional[Tensor]` stores `None` when `bias=False`. No Buffer, no gradbox.
3. **Clean guards** — `if self.bias:` is idiomatic Mojo.
4. **Easy Phase 2 migration** — when we write the Transformer block:
   - Change each `if self.bias:` → `comptime if use_bias:` (same condition, just earlier binding)
   - Add `use_bias` to type signature
   - Absorb the TAG/Variant/Module cascade (~30 changes per struct)
   - The internal logic per method is identical between phases.
5. **No cascade risk** — Phase 1 touches zero files outside `net.mojo`.

**Rejected approaches:**

- **Sentinel 0-rank Tensor (Approach A / Conv2D current pattern):** Still allocates 1 element. Inelegant. The `shape().rank() > 0` guard is less idiomatic than `Optional`. Only reason Conv2D uses it is historical precedent — no reason to repeat it.
- **Always-allocated bias (current state):** Wastes memory, compute, and gradient storage for `bias=False` callers. Unacceptable for modern architectures.

### Phase 1 Scope (Per Struct)

All changes in `net.mojo` only:

| Method | Current → After |
|---|---|
| Field | `var bias: Tensor` → `var bias: Optional[Tensor]` |
| `__init__` | `self.bias = Tensor.zeros/rand/scalar(...)` → `if bias: self.bias = Optional(...)` else `self.bias = None` |
| `__call__` (forward) | `Adder.forward(..., self.bias)` → `if self.bias: result = Adder.forward(result^, self.bias.value())` |
| `parameters()` | `params.append(self.bias)` → `if self.bias: params.append(bias_ptr)` |
| `named_parameters()` | bias NamedParameter → `if self.bias:` |
| `num_parameters()` | `+ self.bias.numels()` → `+ (self.bias.value().numels() if self.bias else 0)` |
| `to_gpu()` | `self.bias.to_gpu(...)` → `if self.bias: out.bias = Optional(self.bias.value().to_gpu(...))` else `out.bias = None` |
| `to_cpu()` | same → same guard |

For `Conv2D` specifically: replaces the sentinel (`Tensor.scalar(0)`) with `None` — same Optional pattern, no more dummy allocation.

### Phase 2 Scope (Per Struct)

Stacked on top of Phase 1:

1. Add `use_bias: Bool = True` as a comptime type parameter
2. Change `TAG` to ternary: `LINEAR if use_bias else LINEAR_NO_BIAS`
3. Replace all `if self.bias:` with `comptime if use_bias:` — no runtime check
4. Remove runtime `bias: Bool` from `__init__` (replaced by type param)
5. Add `LINEAR_NO_BIAS` (and equivalents) to `mnemonics.mojo`
6. Add both type instantiations to `Layer` Variant
7. Add dispatch branches to all 8 `Module` methods
8. Update `Sequential.append()` guards
9. Update tests/examples that construct or extract the layer from Variant

API change:
```mojo
# Phase 1 (runtime):
Linear[DType.float32, mm](in, out, bias=False)

# Phase 2 (comptime):
Linear[DType.float32, mm, use_bias=False](in, out)
```

### When to Move to Phase 2

1. **A Transformer block (`Attention`, `TransformerBlock`) is written.** At that point we know the exact dispatch surface and can evaluate whether comptime elimination is worth the Variant expansion.
2. **The Layer Variant combinatorics become a problem.** If we have 3+ comptime knobs (e.g. `use_bias`, `use_layer_norm`, `use_scale`), the 2^N explosion may force a redesign of how `Layer`/`Module` dispatch works — possibly switching from a flat Variant to a trait-based or delegate-based architecture.
3. **A measured performance regression from the `Optional` tag.** Unlikely, but if a profile shows the extra byte or `.value()` call matters in a tight Transformer loop, Phase 2 eliminates it.

### Type Dispatch Chain (Reference)

```
Linear[dtype, mode] / LinearBLAS[dtype, mode]
    ↓ into()
Layer[dtype] = Variant[Linear[dtype, mm], LinearBLAS[dtype, mm], ...]
    ↓ stored in
Module[dtype] { layer: Layer, tag: Int }
    ↓ tag dispatch to Variant.get[ConcreteType]()
__call__ / parameters / named_parameters / num_parameters / train / eval / to_gpu / to_cpu
    ↓ consumed by
Sequential[dtype] / SequentialBLAS[dtype]
```

With Phase 2 (comptime `use_bias`), `Module` would need:

```mojo
if tag == LINEAR:        self.layer[Linear[dtype, mm, True]](xs)
if tag == LINEAR_NO_BIAS: self.layer[Linear[dtype, mm, False]](xs)
```

The `Linear[dtype, mm, True]` and `Linear[dtype, mm, False]` are different concrete types — the Variant holds one or the other, and `Variant.get[WrongType]()` panics at runtime.

### Conv2D Phase 1 Migration Note

When `Conv2D` is updated from sentinel to `Optional[Tensor]`:
- `var bias: Tensor` → `var bias: Optional[Tensor]`
- `self.bias = Tensor.scalar(0)` → `self.bias = None`
- `self.bias = Tensor(...)` → `self.bias = Optional(Tensor(...))`
- `self.bias.shape().rank() > 0` → `self.bias`
- `.to_gpu()`/`.to_cpu()`: guard transfer with `if self.bias:` (same pattern)
- `parameters()`: guard with `if self.bias:` (already done, just change condition)

This is a low-priority cleanup — sentinel works, it just wastes 1 element.

## ⛔ CRITICAL: NEVER launch multiple tests in parallel — Mojo compilation is memory-bound and will crash the machine. Run ONE `./execute.sh` or `./fire.sh` at a time, wait for it to complete, then run the next.

