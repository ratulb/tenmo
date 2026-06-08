# Performance Bottlenecks

Ranked in decreasing order of severity. Covers GPU kernels, autograd overhead,
memory management, SIMD paths, and system-level issues.

---

## 🔴 Critical (hit on every op, every iteration)

### 1. GPU `device_context.synchronize()` after every kernel launch

**Files:** All GPU launchers (`reduction_kernel.mojo`, `matmul_kernel.mojo`,
`bce_kernel.mojo`, `unary_ops_kernel.mojo`, `binary_ops_kernel.mojo`,
`binary_inplace_ops_kernel.mojo`, `compare_kernel.mojo`, `dropout_kernel.mojo`,
`division_kernel.mojo`, `scalar_ops_kernel.mojo`, `layernorm_kernel.mojo`,
`minmax_kernel.mojo`, `std_variance_backward_kernel.mojo`,
`matrixvector_kernel.mojo`, `vectormatrix_kernel.mojo`)

**Problem:** Every GPU launcher calls `device_context.synchronize()` before
returning to the host. This serializes all GPU work — the host blocks until the
kernel completes. For any multi-operation pipeline (e.g. forward → loss →
backward → step), each step round-trips through PCIe. No CUDA stream
concurrency is possible.

**Root cause:** `synchronize()` is called unconditionally at the end of every
`launch_*` method. The `DeviceState` API supports a `sync: Bool` parameter but
callers never pass `False`.

**Fix:** Remove `synchronize()` from individual launchers and batch into a
stream-enqueue pattern. Let the host enqueue multiple kernels, then
synchronize once at a natural barrier (e.g. before reading results or at
`loss.item()`). Use separate streams for independent work.

---

### 2. `add_ancestry` deep copy of parent tensors

**File:** `tenmo/tensor.mojo:1106`

**Problem:** `add_ancestry()` deep-copies every parent tensor that is not
already shared. For a 100MB Embedding weight, this copies 100MB on every
forward operation (e.g. every `gather()` call). This is the single biggest
training bottleneck for any model with large weights.

**Root cause:**
```mojo
if not parent.buffer.shared():
    var parent_copy = parent.copy()        # full data deep copy
    parent_copy.buffer.buffer.shared()
    ancestors.append(parent_copy^)
```
The copy exists because silently converting the user's tensor to shared
ownership would violate the principle of least surprise.

**Fix:** Layer structs (Embedding, Linear, Conv2d) that "take over" weights
from the user at init time should call
`self.weight.buffer.buffer.shared()` internally — the user handed off
ownership, so sharing is fine. Then `add_ancestry` hits the `else` branch
(zero-cost refcount bump).

---

### 3. CPU reductions (`reduce_cpu`) have no SIMD fast path

**File:** `tenmo/ndbuffer.mojo:1269` (and surrounding)

**Status: 🟢 FIXED** — Fast path added for contiguous suffix-axis reductions.
~840× speedup (75.9 ms → 0.09 ms for 512×1024 sum over last axis).

**Problem (original):** The multi-element reduction path (non-scalar output)
iterates coord-by-coord via `self[self_coord]`, calling
`IndexCalculator.flatten_index` with bounds checking for every element. Even
for fully contiguous inputs, the inner reduction loop is a scalar nested
`for red_coord` — no delegation to `Buffer.sum()` / `Buffer.product()`.

Note: the **scalar** output path (`out_shape == Shape()`) already uses SIMD
via `self.sum_all()` → `Buffer.sum()` with vector loads. The gap was only in
the multi-element path.

**Fix applied:** When `self.is_contiguous()` and the reduced axes form a
suffix of the shape dimensions (e.g., the last axis, or last N axes), each
output element maps to a contiguous block of `reduced_volume` elements in
memory. The fast path computes the flat offset and calls `Buffer.sum()`
per output element, using SIMD vector loads internally.

**Remaining gap:** Non-suffix reduction axes (e.g., reduce over axis 0 of a
2D tensor) and non-contiguous inputs still use the coord-by-coord fallback.
`welford_cpu` and `product_cpu` also have the same pattern and could benefit
from a similar fix.

---

### 4. GPU reduction kernel: one block per output element; element-wise ops cap blocks at 512

**Files:** `tenmo/reduction_kernel.mojo:1489` (one-block-per-output),
`tenmo/unary_ops_kernel.mojo:507` (512 block cap),
`tenmo/scalar_ops_kernel.mojo`, `tenmo/bce_kernel.mojo:1081`

**Problem (reductions):** The reduction launch config returns
`num_blocks = total_output` — one block per output element. When
`total_output` is tiny (e.g. scalar sum: `total_output=1`), only 1 block is
launched with up to 512 threads, grossly under-utilizing the GPU.

**Problem (element-wise):** The 3-tier heuristic in unary/scalar ops caps
`num_blocks` at 512:
```mojo
num_blocks = min((total_chunks + 255) // 256, 512)
```
For a 10M-element tensor, this limits parallelism to 512 blocks × 256 threads
= 131K threads, when the hardware can support 10K+ blocks.

**Fix:** For reductions with tiny output, split the reduced dimension across
multiple blocks and do a second kernel to combine. For element-wise ops,
remove the hard cap or use a much higher one (e.g. 65535) based on the actual
work size.

---

### 5. Matmul kernel: 1 output per thread, no register tiling, TILE_SIZE=32 fixed

**File:** `tenmo/matmul_kernel.mojo:38-122`

**Problem:** The tiled matmul kernel assigns one output element per thread.
Each thread issues 2 shared memory loads + 1 FMA per iteration (arithmetic
intensity ~64 FLOPs per smem access = 1:1 FLOPS:B). No vectorized smem loads,
no bank conflict avoidance (no padding), and TILE_SIZE is fixed at 16 or 32
with no adaptive selection.

**Root cause:** Classic "naive tiled matmul" without modern GPU optimization
patterns (register tiling, vectorized smem, bank conflict padding, WMMA).

**Fix:**
- Register tiling: each thread computes a 2×2, 4×4, or 8×8 tile
- Vectorized smem loads (`float4`)
- Pad shared memory dims by 1 to avoid bank conflicts (`smem_A[TILE_SIZE][TILE_SIZE+1]`)
- Support TILE_SIZE=64 or 128 for large matrices
- Add warp-level matrix multiply (WMMA / tensor core) path

---

## 🟠 High (major impact on common patterns)

### 6. Reduction kernel strided global memory access

**File:** `tenmo/reduction_kernel.mojo:224-236` (and similar in all kernel variants)

**Problem:** Each thread calls `rank_to_reduced_offset()` per element, which
decomposes a linear rank with a `for` loop over all dimensions using
modulo/division. Adjacent threads access non-adjacent memory locations —
fully uncoalesced global loads.

**Root cause:** The reduction kernel is written generically for any reduction
axis. The helper functions reconstruct coordinates from a linear rank for
every element.

**Fix:** For the common case (reduce over last axis), use contiguous loads
where adjacent threads access adjacent elements. For arbitrary axes,
precompute per-block base offsets into constant memory to avoid repeated
decomposition.

---

### 7. No warp-level primitives in reduction/minmax kernels

**Files:** `tenmo/reduction_kernel.mojo`, `tenmo/minmax_kernel.mojo`,
`tenmo/layernorm_kernel.mojo`, `tenmo/std_variance_backward_kernel.mojo`

**Problem:** The reduction, minmax, layernorm, and std_variance_backward
kernels use shared memory + barriers for tree reduction even when
`block_size ≤ 32` (single warp). A warp-level reduction using
`__shfl_down_sync` would be 10-20× faster for small block sizes and free
shared memory for other uses. (Note: `dotproduct.mojo` already uses warp
shuffle — the gap is in the generic reduction kernels.)

**Root cause:** Shared memory tree is the generic pattern; warp intrinsics
were never added as an optimization path in these kernels.

**Fix:** Add a comptime check: `if block_size <= 32` → warp shuffle
reduction; else → shared memory tree.

---

### 8. Backward dispatch is a 61-branch linear `elif` chain

**File:** `tenmo/backpropagation.mojo:360-490`

**Problem:** `Backward.invoke()` dispatches on `op_code` with a linear
`if/elif` cascade of 61 branches. Worst case: ~30 comparisons per dispatch.
For a deep graph of 100 ops, that's ~3000 comparisons.

**Root cause:** Simple `elif` chain — Mojo does not automatically convert
this to a jump table.

**Fix:** Generate a comptime array of function pointers indexed by `op_code`,
giving O(1) dispatch.

---

### 9. `ancestor.buffer()` reconstructs NDBuffer with 2-3 heap allocations

**File:** `tenmo/ancestry.mojo:70-79`

**Problem:** Every call to `ancestor.buffer()` allocates:
1. Shape copy (`IntArray` heap allocation + memcpy)
2. Strides copy (`IntArray` heap allocation + memcpy)
3. Buffer copy (refcount bump if shared; full `alloc`+memcpy if not shared)
4. DeviceState copy (refcount bump — 0 allocs)

**Total: 2–3 heap allocations minimum** per call. This is called multiple
times per backward handler (once per parent). Not as expensive as initially
estimated, but still adds allocation pressure on every backward node.

---

### 10. `Gradbox.transpose()` / `permute()` / `reshape()` always deep-copies

**File:** `tenmo/gradbox.mojo` (various methods)

**Problem:** Every shape manipulation on a Gradbox first deep-copies the
buffer, then operates with `share=False`. This means every reshape/transpose
during backward (which happens in most backward handlers) is a full data
allocation + copy.

**Root cause:** Invariant: Gradbox data must always be contiguous with offset
0. The copy ensures this, but it is unnecessarily conservative — if the
gradbox is already contiguous, reshape could be a view.

**Fix:** For `reshape` on a contiguous gradbox, create a view (no copy).
For `transpose`, only copy if the result needs to be contiguous for the
next operation; otherwise return a view.

---

### 11. `seed_grad(Scalar)` allocates a temporary full-size tensor

**File:** `tenmo/tensor.mojo:1245`

**Problem:**
```mojo
fn seed_grad(mut self, value: Scalar[Self.dtype]):
    var seed = Tensor[Self.dtype].full(self.shape(), value)   # allocates full-size tensor
    self.seed_grad(seed^)                                      # then copies into gradbox
```
Two allocations when zero would suffice.

**Fix:** Add a `seed_grad(value: Scalar)` overload on Gradbox that directly
fills the buffer with the scalar, avoiding the temporary tensor.

---

### 12. Non-contiguous GPU `fill()` is single-threaded CPU loop

**File:** `tenmo/device.mojo:289-302`

**Problem:** When copying a non-contiguous CPU tensor to GPU, `DeviceState.fill()`
falls back to a host-side element-by-element for loop over `index_iterator`.
For large non-contiguous tensors this is catastrophically slow.

**Fix:** Write a small GPU kernel that handles strided copies by accepting
shape/strides/offset as kernel arguments.

---

## 🟡 Medium (specific scenarios, notable impact)

### 13. No multi-dimensional GPU gather kernel

**File:** `tenmo/gather.mojo`

**Problem:** `Gather._gather_copy` supports multi-dimensional index shapes on
CPU natively (`indices_shape` parameter), but on GPU it falls back to flat
gather + tracked reshape. The reshape is zero-cost (view), but the tracked
graph node adds unnecessary complexity.

**Root cause:** The `gather_gpu` kernel only accepts flat 1D `IntArray`
indices and produces `(n, cols)` output. When the input indices are a
multi-dimensional tensor (e.g. `(B, T)` from batched Embedding), the caller
must flatten, gather, then reshape.

**Fix:** Write a GPU gather kernel that accepts an `indices_shape` parameter
(like the CPU path) and produces `(*indices_shape, *src.shape[ax+1:])` output
directly, avoiding the extra reshape node in the autograd graph.

---

### 14. Broadcast arithmetic always scalar — no SIMD

**Files:** `tenmo/ndbuffer.mojo:2680-2730` (`broadcast_nd_buffer`,
`broadcast_scalar_buffer`)

**Problem:** Even `tensor + scalar` with contiguous input uses a coord-iterator
with per-element `flatten_index`. No delegation to `Buffer.arithmetic_ops_scalar`.

**Fix:** For scalar broadcast, delegate directly to
`Buffer.arithmetic_ops_scalar[op_code]`. For matching-contiguous broadcasts,
expand the smaller operand into contiguous form and SIMD.

---

### 15. Ancestor recursive deep copy

**File:** `tenmo/ancestry.mojo:48`

**Problem:** `Ancestor.__copyinit__` deep-copies the entire parent ancestry
chain recursively (`self.parents = copy.parents.copy()`). For a node in the
middle of a deep graph, this reconstructs all ancestor nodes.

**Fix:** Shallow-copy the parent chain (refcount bump only) rather than
deep-copying every ancestor.

---

### 16. `float_unary_ops` parallelization capped at 2 cores

**File:** `tenmo/buffers.mojo:1470`

**Problem:** `min(num_physical_cores(), 2)` hard-caps parallelization to
2 cores, regardless of actual core count.

**Fix:** Remove the cap, or set it dynamically based on CPU topology.

---

### 17. BCE backward always allocates full-size gradient buffer

**Files:** `tenmo/bceloss.mojo:186,195,216,223`

**Problem:** Even for mean/sum reduction where the upstream gradient is a
scalar, `BCE*Backward.backward()` allocates a full-size gradient buffer.

**Fix:** For mean/sum, the backward handler receives a reduced gradient; the
full-size allocation is correct (it must be full-size to add back to the
parent). However, the allocation can be deferred or fused with the kernel
computation.

---

### 18. `product_reduce` uses float64 log-space for ALL dtypes

**File:** `tenmo/reduction_kernel.mojo:298`

**Problem:** Product reduction converts to float64, takes log, accumulates,
then exp + cast back. For int8/int16, direct integer multiplication would
be ~100× cheaper and safe for reasonable tensor sizes.

**Fix:** Add dtype-specific product kernels (int path for integer types,
float64 log-space for floating-point types).

---

### 19. `__getitem__` bounds check per element on every NDBuffer access

**File:** `tenmo/indexhelper.mojo:286`

**Problem:** `IndexCalculator.flatten_index` validates `if idx < 0 or idx >= dim_size`
for every dimension on every call. In hot paths like `reduce_cpu`, this is
called millions of times with valid indices.

**Fix:** Add an unchecked variant for internal use (or use `@always_inline`
with comptime-gated bounds checking).

---

## 🔵 Low (minor or rare)

### 20. `except` with `panic()` in hot inner loops for iterator mismatch

**Files:** `tenmo/ndbuffer.mojo:2237-2253` (`copy_from_alike`),
`tenmo/ndbuffer.mojo:2412-2421` (`inplace_ops_cpu`),
`tenmo/ndbuffer.mojo:2653-2664` (`arithmetic_ops_cpu`)

**Problem:** The both-strided path uses lockstep dual-iteration: it iterates
`self.index_iterator()` in a `for` loop while manually calling
`other.__next__()` inside the loop body, wrapped in `except e: panic()`. The
exception mechanism is only a defensive guard — the iterators should never
desync — but the `try`/`except` overhead is incurred on every element.

**Fix:** Use an explicit `while` loop with `__has_next__()` checks on both
iterators, avoiding the exception mechanism entirely.

---

### 21. GPU dropout reuses same 4 random values across SIMD lanes

**File:** `tenmo/dropout_kernel.mojo:83-88`

**Problem:** `rng.step_uniform()` returns 4 values, but the code reuses them
across all lanes via `rand_f32[lane % 4]`. Multiple SIMD lanes share the same
random value, reducing statistical quality when `simd_width > 4`.

**Fix:** Call `step_uniform()` multiple times (once per group of 4 lanes) or
use a different subsequence per group.

---

### 22. `atomic_and` implemented as CAS loop instead of native instruction

**File:** `tenmo/compare_kernel.mojo:30-36`

**Problem:** The all_close kernel emulates `atomicAnd` with a CAS loop.
NVIDIA GPUs have a native `atomicAnd` instruction.

**Fix:** Use the native `atomicAnd` intrinsic if available (Mojo GPU
intrinsics).

---

### 23. `excl_product_kernel` does 2 full passes over data

**File:** `tenmo/reduction_kernel.mojo:453-608`

**Problem:** Pass 1 accumulates totals, Pass 2 computes per-element
excl_product. This doubles memory traffic vs a single-pass approach
using running prefix/suffix products.

**Fix:** Rewrite as a single-pass scan using shared memory for per-block
prefix/suffix.

---

### 24. Dead SIMD strided load code

**File:** `tenmo/buffers.mojo:279-302`

**Problem:** The SIMD strided `__getitem__` path is stringified via
`_ = """..."""` — never compiled. All non-unit-step slicing uses a
scalar loop regardless of data size.

**Fix:** Compile and use the SIMD strided load path (re-enable the block
with proper stride computation).

---

### 25. Duplicated 4-case contiguity dispatch code

**Files:** `tenmo/ndbuffer.mojo` — `arithmetic_ops_cpu`, `inplace_ops_cpu`,
`copy_from_alike`

**Problem:** The 4-case contiguity dispatch (both contiguous, A strided,
B strided, both strided) is ~120 lines duplicated across 3 functions.

**Fix:** Extract a shared helper or comptime-parameterized template.

---

### 26. Fused CrossEntropy kernel computes softmax even for ignored rows

**File:** `tenmo/kernels/crossentropy_fused_kernel.mojo` (lines 87–148)

**Problem:** The fused kernel unconditionally computes exp, log-sum-exp, and
normalized softmax for ALL rows, even when `ignore_index` makes the target
invalid. Thread 0 skips the atomic accumulation for ignored rows, but all
threads in the block still run the shared-memory tree reduction (max, sum_exp,
sum_logits) and write back normalized softmax. On datasets with many ignored
positions (e.g. padded sequences), this wastes GPU cycles.

**Fix:** Add a warp-level or block-level early exit at the top of Phase 2 when
`target[row] == ignore_index`. The block can skip the softmax computation and
directly write 0 to `per_sample_loss[row]` and 0-filled softmax to
`softmax_out[row * C : (row+1) * C]`. The `scalar_loss` and `valid_count`
atomics are already correctly guarded by the `is_valid` check.

---

## Top 4 Highest-ROI Fixes

1. **Layer structs proactively share buffers** → eliminates `add_ancestry`
   deep copy (biggest win for training throughput, especially with large
   weights)
2. **Remove `synchronize()` from GPU launchers** → replace with
   stream-enqueue pattern and synchronize at natural barriers
3. ~~Add SIMD fast path to CPU `reduce_cpu`~~ **🟢 DONE** — ~840× speedup
   for contiguous suffix-axis reductions. Next: apply same pattern to
   `welford_cpu` and `product_cpu`.
