# From Bytes to Gradients: Tracing a Neural Network Through Tenmo, One Layer at a Time

When you call `loss.backward()` in PyTorch, a C++ autograd engine climbs the computation graph in reverse, multiplying Jacobians until every leaf tensor has its gradient filled in. It works. It's fast. But the graph lives in C++ libraries you never see — `torch::autograd::Engine`, `THPVariable`, `VariableType` — hundreds of thousands of lines built over a decade.

What if you could read *every line* of the system between `loss.backward()` and the weight update? That's the premise of Tenmo, a tensor library and neural network framework written entirely in Mojo. Every autograd dispatch, every SIMD matmul kernel, every GPU launch is in one repository in 130+ source files.

This post traces one MNIST training step — `matmul → bias_add → relu → matmul → bias_add → relu → matmul → bias_add → cross_entropy` — through every layer of the system. We'll start with raw memory allocation and end with the final parameter update, showing the real code at each stage.

<blockquote style="color: #d4d4d8; text-shadow: 0 1px 3px rgba(0,0,0,0.4);">
  <strong>Audience note:</strong> this post assumes familiarity with autograd and basic SIMD concepts.
</blockquote>

<blockquote style="color: #d4d4d8; text-shadow: 0 1px 3px rgba(0,0,0,0.4);">
  <strong>TL;DR:</strong> This post traces one MNIST batch through Tenmo's full stack — memory allocation, SIMD matmul, autograd graph traversal, SGD — with real code at each step. Skip to <a href="#8-putting-it-all-together">§8</a> for the unified training loop or <a href="#what-the-benchmarks-say">What the Benchmarks Say</a> for the numbers.
</blockquote>

- [1. The Memory Model — Buffer](#1-the-memory-model--buffer)
- [2. Shape + Strides + Views — NDBuffer](#2-shape--strides--views--ndbuffer)
- [3. Tensor — The User-Facing Type](#3-tensor--the-user-facing-type)
- [4. Forward Pass — A Real MNIST Step](#4-forward-pass--a-real-mnist-step)
- [5. The Backward Graph](#5-the-backward-graph)
- [6. The Optimizer — SGD Step](#6-the-optimizer--sgd-step)
- [7. GPU Transfer](#7-gpu-transfer)
- [8. Putting It All Together](#8-putting-it-all-together)
- [What the Benchmarks Say](#what-the-benchmarks-say)
- [Common Pitfalls](#common-pitfalls)
- [Try It Yourself](#try-it-yourself)

<!-- Line numbers referenced throughout (e.g., "buffers.mojo:122", "tensor.mojo:1080") point to specific snapshots in the Tenmo source and may shift as the codebase evolves. -->

## 1. The Memory Model — Buffer

Every tensor operation eventually reads or writes a flat array of scalars. In Tenmo, that flat array is a `Buffer[dtype]` — a CPU-only, shape-agnostic block of memory with one optional feature: reference counting.

```mojo
struct Buffer[dtype: DType = DType.float32]:
    var size: Int
    var data: Optional[UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]]
    var _refcount: Optional[UnsafePointer[Atomic[DType.uint64], MutAnyOrigin]]
    var external: Bool
```

A `Buffer` has two modes. **Unshared**: a single allocated block of `Scalar[dtype]` elements with no reference counting. `__init__(*, copy:)` deep-copies the data — malloc + memcpy. **Shared**: the allocation layout is `[refcount: Atomic(UInt64)] | [data array]`, and `__init__(*, copy:)` merely bumps the atomic counter. `__del__` decrements; when it hits zero, the combined allocation is freed in one shot.

The `shared()` method transforms an unshared buffer in-place (line 122 of `buffers.mojo`):

```mojo
def shared(mut self):
    if self.is_shared():
        return
    var refcount_size = size_of[Atomic[DType.uint64]]()
    var data_size = self.size * size_of[Scalar[Self.dtype]]()
    var total_size = refcount_size + data_size
    var new_alloc = alloc[UInt8](total_size)
    var refcount_ptr = new_alloc.bitcast[Atomic[DType.uint64]]()
    refcount_ptr[] = Atomic[DType.uint64](1)
    var new_data = (new_alloc + refcount_size).bitcast[Scalar[Self.dtype]]()
    memcpy(dest=new_data, src=self.data, count=self.size)
    self.data.unsafe_value().free()
    self.data = new_data
    self._refcount = refcount_ptr
```

This allocation layout matters because views share the same Buffer via refcount bump. When we slice a tensor, the new tensor's NDBuffer points to the same underlying Buffer with a refcount of 2. The memory stays alive as long as any view holds a reference, regardless of Mojo's aggressive destruction of intermediate tensors.

There's also a static `Buffer.shared(size)` constructor that allocates the combined layout from the start, avoiding the O(n) reallocation that the instance `shared()` method performs. This is the fast path used by `Gradbox.__init__`.

## 2. Shape + Strides + Views — NDBuffer

A flat Buffer doesn't know about dimensions. That's the job of `NDBuffer[dtype]` — the single source of truth for shape, strides, offset, and device location.

```mojo
struct NDBuffer[dtype: DType]:
    var shape: Shape
    var strides: Strides
    var offset: Int
    var _contiguous: Bool
    var buffer: Buffer[dtype]      # CPU data
    var device_state: Optional[DeviceState]  # GPU data
```

The key insight: `NDBuffer` doesn't own the data. It points into a `Buffer` at some `offset`, interpreting the flat memory through `strides`. A contiguous tensor `(3, 4)` with strides `(4, 1)` and offset `0` maps element `(i, j)` to `buffer[i*4 + j]`. A transposed view of the same tensor has strides `(1, 4)` and offset `0` — element `(i, j)` maps to `buffer[i*1 + j*4]`.

Zero-copy slicing uses `share()`:

```mojo
def share(
    self, new_shape: Shape, new_strides: Strides, new_offset: Int
) -> NDBuffer[Self.dtype]:
    # Enables refcounting on the CPU Buffer (first call does the transform)
    self.buffer.shared()
    # Returns a new NDBuffer pointing at the same Buffer
    return NDBuffer(...)
```

On GPU, there's no separate sharing step — `DeviceBuffer` (Mojo's GPU built-in) is always refcounted. The `device_state` is simply copied by pointer.

`reshape()` exploits this: if the new shape's `max_index` fits within the underlying `buffer_size`, it returns a zero-copy view with new strides and offset. Only when the view would require discontiguous access does it materialize a contiguous copy.

This is the foundation for the "reshape is free" property of the autograd graph. A `ReshapeBackward` handler (in `reshape.mojo`) does nothing but reshape the gradient tensor to the parent's shape — no data transformation, just a new `Shape` and `Strides` object.

## 3. Tensor — The User-Facing Type

The `Tensor[dtype]` struct bundles an NDBuffer with autograd metadata:

```mojo
struct Tensor[dtype: DType]:
    var _id: UInt
    var buffer: NDBuffer[Self.dtype]
    var requires_grad: Bool
    var gradbox: Optional[Gradbox[Self.dtype]]
    var ancestors: Optional[Ancestors[Self.dtype]]
```

Two of these fields deserve a closer look.

**Gradbox** — this is not Tensor, and that matters. Tensor is 4543 lines of code; Gradbox is 1526. Gradbox doesn't need  reductions, trig, comparisons, or many of the 200-odd operations Tensor supports. It only needs gradient storage shapes, accumulation (add, subtract, zero), reshape, broadcast, and device transfer. That's it. A lean container specialized for one job.

Technically, Gradbox is a combined heap allocation of `[Atomic(UInt64)] | [NDBuffer]`. The atomic refcount is *independent* of the Tensor's refcount. When Mojo's ASAP destruction drops an intermediate tensor, the Gradbox survives if other handles (Ancestor copies in the graph) still reference it. This prevents dangling pointers in the autograd graph.

```mojo
struct Gradbox[dtype: DType]:
    var _ndb_ptr: Optional[UnsafePointer[NDBuffer, MutAnyOrigin]]
    var _refcount: Optional[UnsafePointer[Atomic[DType.uint64], MutAnyOrigin]]
```

In `__init__(shape)` (line 33 of `gradbox.mojo`), it allocates one block, initializes the atomic to 1, and constructs the NDBuffer via move-init. `__init__(*, copy:)` bumps the atomic via `fetch_add[RELAXED](1)`. `__del__` decrements via `fetch_sub[RELEASE](1)`; if the result is 1 (meaning this was the last handle), it destroys the NDBuffer and frees the combined allocation.

When you need to convert between the two, `Gradbox.as_tensor()` (`gradbox.mojo:118`) materializes a contiguous copy of the gradient data as a Tensor, and `Tensor.as_gradbox()` (`tensor.mojo:135`) consumes the Tensor's NDBuffer to produce a Gradbox. This metamorphosis between types is explicit — you don't accidentally use a gradient storage container as a full tensor.

**Ancestor** — The old Tenmo design stored full `Tensor` copies at every `add_ancestry` call, triggering recursive deep copies, gradbox allocations, and heap blocks. The current design uses a lightweight handle:

```mojo
struct Ancestor[dtype: DType]:
    var _id: UInt
    var requires_grad: Bool
    var gradbox: Optional[Gradbox[Self.dtype]]
    var ndb: Optional[NDBuffer[Self.dtype]]
    var parents: Optional[Ancestors[Self.dtype]]
```

The `ndb` field is only populated when `needs_parent_data=True` — most operations don't need it. Addition doesn't need the parent's buffer; it just passes the gradient through unchanged. Matmul does need the parent's data (to compute `grad × B^T`), so `needs_parent_data=True` is set on its `BackwardFnArg`.

## 4. Forward Pass — A Real MNIST Step

With the data structures in hand, let's trace one batch through the MNIST model. The architecture is `784 → 128 → ReLU → 32 → ReLU → 10`, built as a `Sequential`:

```mojo
var model = Sequential[dtype]()
model.append(
    Linear[dtype](784, 128).into(),
    ReLU[dtype]().into(),
    Linear[dtype](128, 32).into(),
    ReLU[dtype]().into(),
    Linear[dtype](32, 10).into(),
)
```

A forward call `model(x)` dispatches through each layer in sequence. The heaviest operation by far is `matmul` — three of them per batch, each computing `(batch_size, in_features) × (in_features, out_features)`.

### Matmul — The CPU Kernel

The CPU matmul lives in `matmul_cpu.mojo`, struct `MmCpu2d`. It selects from 18 tile configurations based on the matrix dimensions (`m`, `n`, `p`):

```mojo
var tile_m = 128 if m > 256 else (64 if m > 64 else 32)
var tile_n = 64  if n > 64  else 32
var tile_p = 256 if p > 256 else (128 if p > 64 else 64)
```

For the first layer `(64, 784) × (784, 128)`, `m=64, n=784, p=128`. Tracing through the selection (matmul_cpu.mojo:87–89):

- `tile_m = 128 if m > 256 else (64 if m > 64 else 32)` — `m=64`: `64 > 256` false → `64 > 64` false → **tile_m=32**
- `tile_n = 64 if n > 64 else 32` — `n=784 > 64` → **tile_n=64**
- `tile_p = 256 if p > 256 else (128 if p > 64 else 64)` — `p=128`: `128 > 256` false → `128 > 64` true → **tile_p=128**

Result: `MmCpu2d[float32, 32, 64, 128]` — the `tile_m=32` branch of the 18-way dispatch table.

Note the `tile_p=128` choice. The `p > 64` check that picks 128 over 256 when `p=128` is about L1 cache capacity, not SIMD utilization. Tile_P controls the outer `j_tile` stride — how many columns of B are loaded per `k_tile` pass and reused across all rows in the tile. With `TILE_N=64` and `TILE_P=256`, the B j-tile is `64 × 256 × 4 bytes = 64 KB`, which overflows L1 data cache (32 KB). With `TILE_P=128`, it's `64 × 128 × 4 = 32 KB`, fitting perfectly. The inner SIMD unrolled loop (32 columns per iteration) is equally efficient in either case — `j_end = min(j_tile + TILE_P, p)` caps it at the actual 128 columns regardless of `TILE_P`, so 4 iterations of 32 columns fully cover the output with no tail.

Inside the selected tile configuration, the hot loop processes columns in groups of `simd_unroll = simdwidth × UNROLL` (for float32 with AVX2: `8 × 4 = 32` columns per iteration):

```mojo
# Unrolled SIMD: 4 independent accumulators fill the FMA pipeline
var acc0: SIMD[Self.dtype, simdwidth]
var acc1: SIMD[Self.dtype, simdwidth]
var acc2: SIMD[Self.dtype, simdwidth]
var acc3: SIMD[Self.dtype, simdwidth]

if k_tile == 0:
    acc0 = SIMD[Self.dtype, simdwidth](0)  # C is zeroed, skip load
else:
    acc0 = C_data.load[width=simdwidth](cj)

for k in range(k_tile, k_end):
    var a_ik = SIMD[Self.dtype, simdwidth](A_data[a_row_base + k])
    var b_base = k * B_stride0 + B_offset + j
    acc0 = math.fma(a_ik, B_data.load[width=simdwidth](b_base), acc0)
    acc1 = math.fma(a_ik, B_data.load[width=simdwidth](b_base + simdwidth), acc1)
    acc2 = math.fma(a_ik, B_data.load[width=simdwidth](b_base + simdwidth * 2), acc2)
    acc3 = math.fma(a_ik, B_data.load[width=simdwidth](b_base + simdwidth * 3), acc3)
```

Each iteration: one broadcast of `a_ik` (scalar→SIMD), four SIMD loads from B, four FMA instructions. For float32 with `simdwidth=8`: **32 FMAs per inner iteration**. The `k_tile==0` optimization skips loading C (it starts zeroed), saving 4 vector reads on the first tile pass.

Rows are parallelized across physical cores using `parallelize` from Mojo's standard library — each core processes a contiguous block of `TILE_M` rows with its own cache-hot k-strip and j-tile.

### Bias Add — Broadcast Arithmetic

After matmul, bias addition broadcasts a `(128,)` vector across the batch dimension. This dispatches through `CpuArithmeticOps.broadcast` (`cpu_arithmetics.mojo`) which selects Tier 2: one operand has unit stride in the last dimension, the other broadcasts (stride 0).

```mojo
# Tier 2: SIMD splat from broadcasting side
var scalar_vec = SIMD[Self.dtype, simd_width](scalar_v)
while j + simd_width <= last_dim:
    var vec = b.buffer.load[simdwidth=simd_width](b_off + j)
    var op_result = simd_op[op_code, Self.dtype, simd_width](vec, scalar_vec)
    buffer.store[simdwidth=simd_width](out_base + j, op_result)
    j += simd_width
```

A single scalar is splatted into a SIMD register, then the contiguous side is SIMD-loaded and vector-added. This is the same mechanism used by every broadcasting op in the system — bias add, layer norm, cross-entropy sub-ops.

### Cross-Entropy — Fused GPU Kernel

The final layer produces logits `(64, 10)`. `CrossEntropyLoss` dispatches through `CrossEntropyFusedKernel` on GPU (at `tenmo/kernels/crossentropy_fused_kernel.mojo`). This fused kernel computes max-reduce, exp, sum-exp, softmax, and NLL in a single GPU launch:

- Thread-block-per-row pattern (M = 64 blocks)
- Shared-memory tree reduction for max and sum_exp
- Register-level log_softmax computation
- Single scalar write per block for the loss value

Without this fusion, `cross_entropy` would trigger ~18 separate kernel launches plus a CPU onehot fallback. The fused kernel reduces it to 1 launch + 4 backward arithmetic ops.

On CPU, cross-entropy uses an analogous fused path that walks rows with SIMD vectorization, computing the max, exp, sum, log, and NLL in a single row loop.

## 5. The Backward Graph

Every forward operation that needs gradient tracking registers a `BackwardFnArg` and parent `Ancestor` handles on the output tensor. Let's see what happens when we call `loss.backward()`.

### What `add_ancestry` Stores

When `Multiplicator.forward()` registers `c = a * b`, it creates:

```mojo
var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(BACKWARD_MULTIPLY)
backwardFnArg.needs_parent_data = True  # backward needs parent buffer
out.add_ancestry(backwardFnArg^, self, other)
```

The `BackwardFnArg` is the dispatch key — a type-erased container packing the integer `op_code` together with a destructor function and copier function for whatever payload it carries. The 58 operation codes are defined as `comptime` constants in `backpropagation.mojo` (e.g. `BACKWARD_ADD = 0`, `BACKWARD_MATMUL_2D = 4`, `BACKWARD_SIGMOID = 7`).

`add_ancestry()` (`tensor.mojo:1080`) converts each parent Tensor into an `Ancestor` handle. When `needs_parent_data=True`, it copies the parent's NDBuffer and calls `buffer.share()` to enable refcounting. When `False` (most ops), it creates the ancestor with no ndb — just the `_id`, `requires_grad` flag, and gradbox pointer.

### The Backward Pass — Phase by Phase

The `backward()` method at `tensor.mojo:3160` proceeds in three phases:

**Phase 1: Seed gradient.** `output.seed_grad(1.0)` allocates the output's gradbox (if needed) and fills it with 1.0. On GPU, `sync=True` fences all pending GPU work before the seed — ensuring forward kernel outputs are visible before backward reads them.

**Phase 2: DFS graph collection.** Starting from the output's `Ancestor`, the code walks parent references recursively, building three parallel structures:

```mojo
var node_list = List[Ancestor[Self.dtype]]
var fanin = Dict[UInt, Int]()
var id_to_index = Dict[UInt, Int]()

# DFS: push root, pop, visit parents
var root = output.to_ancestor()
root.ndb = output.buffer.copy()  # root always gets data
dfs_stack.append(root._id)
while len(dfs_stack) > 0:
    var node_id = dfs_stack.pop()
    if node_id in visited:
        continue
    visited.add(node_id)
    topo_ids.append(node_id)
    if node.has_ancestry():
        for parent in node.ancestry():
            var parent_id = parent._id
            fanin[parent_id] = fanin.get(parent_id, 0) + 1
            if parent_id not in id_to_index:
                node_list.append(parent.copy())
                id_to_index[parent_id] = new_idx
                dfs_stack.append(parent_id)
```

`fanin` counts how many children depend on each node. The root has fanin 0. A matmul node may have fanin 0 (no one depends on its gradient) or 1 (a ReLU sits on top).

**Phase 3: Reverse topological execution.** A `ready_queue` starts with the root. For each popped node:

1. `Backward.invoke(node, parent_ids)` dispatches via a 58-way jump table on `op_code` to the appropriate backward handler
2. The handler reads `output.gradients()`, computes parent gradient contributions, calls `parent.update_grad(grad, op_code, extra_arg)` to accumulate into each parent's gradbox
3. For each parent that received gradient, its `_id` is appended to `parent_ids`
4. Each parent's fanin is decremented; when it hits 0 and the parent has ancestry, it's enqueued

### Example: Multiply Broadcast Backward

When `c = a * b` with broadcasting (e.g. `a` is `(3, 1)` and `b` is `(1, 4)`), the backward handler at `multiplication.mojo:85` is aliased to `BroadcastBackward`. This handler:

1. Extracts the upstream gradient `∂loss/∂c` from the output's gradbox
2. Broadcasts/unbroadcasts it to each parent's shape
3. If the op is multiplication, scales by the other parent's values: `∂loss/∂a = ∂loss/∂c * b`
4. Calls `ancestor.update_grad(grad_contrib, AddTensor, None)` for each parent

The `update_grad` method at `ancestry.mojo:72` dispatches on the `op_code` parameter:
- `AddTensor`: `gradbox += incoming` (in-place addition)
- `ScatterAddTensor`: `Filler.scatter_add()` for sparse gradient accumulation (used by Gather backward)
- `ZeroGrad`: `gradbox.zero_grad()`

### The "Aha" Moment — Reshape Backward

`ReshapeBackward` (`reshape.mojo:13`) is the simplest backward in the system:

```mojo
def backward(output, mut parent_ids, retain_graph=False):
    ref gradbox = output.gradients()
    var ancestor = output.ancestry().get(0)
    if ancestor.requires_grad:
        var reshaped = gradbox.reshape(ancestor.shape())
        ancestor.update_grad(reshaped^, AddTensor, None)
```

It just reshapes the gradient tensor to the parent's shape. No data transformation — a new `Shape` and `Strides` object, same Buffer, same values. If your forward was `(2,6) → reshape(3,4)`, backward is just `gradient(3,4) → reshape(2,6)`. The gradient values pass through unchanged.

This contradicts the naive intuition that "reshape is a math op that rearranges data". It's a metadata op. The backward proves it.

## 6. The Optimizer — SGD Step

After backward fills every gradient, `SGD.step()` updates the parameters. The optimizer struct at `optim.mojo:10` holds pointers to parameters, velocity buffers (for momentum), and hyperparameters.

```mojo
struct SGD[dtype: DType, //]:
    var parameters: List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]
    var lr: Scalar[Self.dtype]
    var momentum: Scalar[Self.dtype]
    var weight_decay: Scalar[Self.dtype]
    var velocities: List[Gradbox[Self.dtype]]
```

The `step()` method iterates each parameter, checks `requires_grad && has_grad()`, and runs the update. On CPU, it's SIMD-vectorized:

```mojo
def _step_no_momentum[simd_w: Int](self, param_ptr, grad_ptr, num_elements):
    var lr_vec = SIMD[Self.dtype, simd_w](self.lr)
    var wd_vec = SIMD[Self.dtype, simd_w](self.weight_decay)
    for j in range(0, vec_end, simd_w):
        var p_vec = param_ptr.load[width=simd_w](j)
        var g_vec = grad_ptr.load[width=simd_w](j)
        if self.weight_decay > 0:
            g_vec += p_vec * wd_vec
        p_vec -= lr_vec * g_vec
        param_ptr.store[width=simd_w](j, p_vec)
```

On GPU, the update launches an in-place kernel (`sgd_kernel.mojo`) without any CPU round-trip. The kernel reads `param` and `grad` from GPU memory, applies the update, and writes back — all on-device:

```mojo
def sgd_step_no_momentum_kernel[dtype: DType](
    param: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grad: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    num_elements: Int, lr: Scalar[dtype], weight_decay: Scalar[dtype],
):
    var gtid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    var i = gtid
    while i < num_elements:
        var p = param[i]
        var g = grad[i]
        if weight_decay > 0:
            g += p * weight_decay
        param[i] = p - lr * g
        i += stride
```

Each thread handles strided elements across the parameter array — a classic GPU element-wise pattern. The momentum variant adds a velocity buffer read/write and the momentum term `v = momentum * v + g`.

The optimizer supports sparse row-wise updates for embedding layers: when `indices` are provided, only specific rows of 2D parameters are updated. This was critical for word2vec-style training where only ~10 rows out of 252K receive gradient each step — a 25000× reduction in write traffic.

## 7. GPU Transfer

Tensor transfer between CPU and GPU goes through `DeviceState` at `device.mojo:229`:

**CPU → GPU:** `DeviceState.fill(ndb)` copies data from the CPU NDBuffer's logical view to a GPU device buffer. If the source is contiguous, it's a direct `memcpy` to a mapped device buffer. If strided, it iterates via `index_iterator()` and writes each element.

**GPU → CPU:** `DeviceState.into(shape)` calls `map_to_host()` to bring the GPU buffer to host-accessible memory, then `memcpy` back to a CPU Buffer.

`DType.bool` is stored as `uint8` internally — a limitation of Mojo's `DeviceBuffer` which doesn't support `DType.bool`. The `datatype` comptime field on `DeviceState` handles the cast transparently.

The `stop_grad` parameter controls whether a device transfer registers a backward node. With `stop_grad=False` (default), the transfer creates a `DeviceTransferBackward` node, so gradients tunnel transparently across device boundaries. With `stop_grad=True`, no backward node is registered — the destination becomes a new leaf on the target device.

The recommended training pattern transfers model weights to GPU once:

```mojo
model = model.to_gpu(stop_grad=True)    # weights become GPU leaves
# ... entire training loop on GPU ...
model = model.to_cpu(stop_grad=True)    # persist back to CPU
```

## 8. Putting It All Together

The unified MNIST example at `examples/mnist.mojo` (151 lines) ties everything together:

```mojo
def train_mnist() raises:
    comptime dtype = DType.float32
    # ... data loading via numpy interop ...

    var model = Sequential[dtype]()
    model.append(
        Linear[dtype](784, 128).into(),
        ReLU[dtype]().into(),
        Linear[dtype](128, 32).into(),
        ReLU[dtype]().into(),
        Linear[dtype](32, 10).into(),
    )
    comptime if has_accelerator():
        model = model.to_gpu(stop_grad=True)

    var opt = SGD(model.parameters(), lr=0.01, momentum=0.9)
    var loss_fn = CrossEntropyLoss[dtype]()

    for epoch in range(epochs):
        train_loader.reset()
        while train_loader.__has_next__():
            ref batch = train_loader.__next__()
            var x = batch.features
            var y = batch.labels
            comptime if has_accelerator():
                x = x.to_gpu(sync=False)
                y = y.to_gpu(sync=False)
            var pred = model(x)
            var loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
```

The loop is under 80 lines. Everything we traced — Buffer allocation, NDBuffer strides, Gradbox refcounting, SIMD matmul, broadcast arithmetic, fused CE kernel, autograd graph traversal, SGD vectorized update — collapses into this tight loop.

The `comptime if has_accelerator()` pattern is key: on a CPU-only system, the GPU branch compiles away entirely. No runtime dispatch, no dead code. The same source file runs on both platforms.

## What the Benchmarks Say

Training the same 4-layer MLP on identical hardware (15 epochs, batch_size=64, all runs sequential):

| Platform | Device | Avg Epoch Time | Total Time | Final Val Acc |
|---|---|---|---|---|
| Tenmo | CPU (Mojo) | 5.5s | 82.3s | 98.14% |
| Tenmo | GPU (Mojo) | 6.0s | 90.1s | 98.00% |
| PyTorch | GPU (CUDA) | 14.5s | 217.2s | 98.18% |
| PyTorch | CPU | 15.4s | 231.5s | 98.12% |

**2.8× faster than PyTorch CPU, 2.4× faster than PyTorch GPU.** The CPU result is the headline: pure Mojo SIMD on a 104K-parameter model saturates the machine[^1] before GPU launch overhead pays off. On a model this small, each GPU kernel launch has too few elements to amortize its dispatch cost — the MNIST MLP does 13 kernels per forward/backward step, each with 64 rows or fewer, and the cumulative launch latency exceeds the compute time. We include the GPU number because it's an honest measurement: Tenmo's GPU path is correct and matches PyTorch GPU behavior, but small models don't benefit. The fusion work described in the Cross-Entropy section is exactly the strategy that will close this gap.

Each design choice has a measurable payoff:

| Choice | Payoff |
|---|---|
| Ref-counted Buffer sharing | Reshape is free — no alloc, no copy |
| SIMD-tiled matmul + FMA + UNROLL=4 | 32 FMAs per iteration, saturates the CPU |
| Lightweight Ancestor handles | No Tensor copy in the graph — just `_id` + gradbox |
| Fused CE GPU kernel | 1 launch instead of 18 |
| In-place GPU SGD step | No CPU round-trip for parameter updates |
| Gradbox independent refcount | Survives Mojo's ASAP destruction — gradients persist |
| Comptime graph elimination | Zero backward overhead in eval mode |

These aren't abstract architectural claims. Every line of code is in the repository.

---

## Common Pitfalls

**Gradbox lifespan confusion.** Gradboxes have their own refcount. If you save `tensor.grad()` to a variable, it returns a deep copy via `Gradbox.detach()` — a fresh allocation with independent data. The internal gradbox remains untouched by subsequent `zero_grad()` calls. The detached copy is safe to use, but it's not linked to the parameter anymore.

**`stop_grad=True` breaks graph flow.** If you transfer weights to GPU with `stop_grad=True`, the model's parameters become GPU leaves. Input tensors transferred with `stop_grad=False` (default) can still carry gradients from the loss back to their CPU origin, but the weights' gradients accumulate on the GPU parameters. This is usually what you want, but it means `model.to_cpu(stop_grad=True)` creates new CPU leaves — the GPU weight values are copied, but the CPU copy won't receive future gradients.

---

[^1]: "CPU's SIMD vector units sustain peak arithmetic throughput — no stalls from cache misses or memory bandwidth — because the entire 104K-parameter model (~1 MB) fits in L3 cache, so every cycle does useful FMA. On GPU, the same model dispatches 13 kernels per step with at most 64 rows each; kernel launch latency (~10–50 μs per launch) exceeds the GPU's compute time, leaving the hardware underutilized. For larger models (millions of parameters), the GPU's massive parallelism eventually dominates.

## Try It Yourself

The complete source is on GitHub at [ratulb/tenmo](https://github.com/ratulb/tenmo). To train the MNIST model from this post without building from source:

<pre style="background: #0d2b4e;"><code style="color: #e6edf3;">docker run -it ratulb/tenmo:latest /app/bin/mnist</code></pre>


This runs the MNIST CPU example from `examples/mnist.mojo` — the same 784→128→ReLU→32→ReLU→10 architecture traced above — compiled into a static binary inside the container. Corresponding PyTorch is [script](https://github.com/ratulb/tenmo/blob/main/mnist_pytorch.py).
