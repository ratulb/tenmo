# Performance Bottlenecks

Ranked in decreasing order of severity. Covers GPU kernels, autograd overhead,
memory management, SIMD paths, and system-level issues.

---

## 🔴 Critical (hit on every op, every iteration)

### 1. GPU `device_context.synchronize()` after every kernel launch

**Status: 🟢 FIXED — Phase 1 complete.**

All 41 kernel launchers now default `sync=False`. Sync is only triggered
when callers explicitly pass `sync=True`. NDBuffer dispatch methods all default
`sync=False`. Tensor dunders default `sync=True` and thread it to forward
structs, but most forward structs don't actually pass sync to NDBuffer calls,
so the effective behavior is async at the kernel level.

**Current per-batch sync profile in MNIST GPU training:**
- `backward(sync=True)`: 1 sync fence at entry (drains forward queue)
- `optimizer.step()`: **async GPU kernel** — no sync (SGD GPU kernel)
- `loss.item()`: 1 sync via `map_to_host()` (pipeline already drained)
- `compute_accuracy_gpu()`: 1 sync via `map_to_host()` (reads 1 Int back)
- **Total: ~2 sync barriers per batch** (down from ~1 barrier + N parameter syncs)

The per-op sync concern in the original analysis does not apply in practice —
most forward structs accept `sync` but don't thread it to NDBuffer, which
defaults to `sync=False`. The pipeline runs async by default.

---

### 2. `add_ancestry` deep copy of parent tensors

**Status: 🟢 FIXED** — Layer structs (Linear, Embedding, Conv2d) call
`self.weight.buffer.buffer.shared()` at init time. `add_ancestry` hits the
`else` branch (zero-cost refcount bump) for all weight tensors.

**File:** `tenmo/tensor.mojo:1106`

**Original problem:** `add_ancestry()` deep-copied every parent tensor that
was not already shared. For a 100MB Embedding weight, this meant 100MB copy
per forward call.

**Fix:** Layer init calls `.share()` on weight buffers so `add_ancestry`
never hits the copy path for parameters.

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

## Top 3 Highest-ROI Fixes

### 1. `compute_accuracy` CPU round-trip — write GPU kernel

**Status: 🟡 IN PROGRESS**

**Problem:** The training loop calls `pred.to_cpu()` every batch (line 188 of
`examples/mnist_gpu.mojo`) just to compute argmax in a scalar CPU loop.
This transfers 640 floats (64×10) from GPU→CPU and runs a per-element Python-
style loop. The sync from `map_to_host()` is cheap (pipeline already drained
by `backward(sync=True)` + `step()`), but the unnecessary data transfer is
wasteful.

**Fix:** Single GPU kernel: one thread per sample computes argmax, compares
with label, atomically increments counter. Single `Int` read-back instead of
full 640-element transfer. Eliminates `pred.to_cpu()` from the training loop.

**Design:**
- Kernel file: `tenmo/kernels/accuracy_kernel.mojo`
- Launcher: `Accuracy[dtype].launch(pred_ndb, labels_ndb)` → returns `Int`
- Launch config: 1 block, `batch_size` threads
- Each thread: argmax over row → compare with label → `Atomic.add` to result
- Single `Int` read back via `map_to_host`
- Also remove `pred.to_cpu()` from training loop in `examples/mnist_gpu.mojo`

**ROI:** Eliminates 1 `to_cpu()` per batch × 938 batches/epoch = 938 unnecessary
GPU→CPU transfers. Net impact on epoch time: small for MNIST (640 elements is
tiny), but critical for larger models with more classes or wider outputs.

---

### 2. `SGD.step()` CPU read-modify-write — GPU kernel

**Status: 🟢 FIXED**

**Problem:** `SGD.step()` (`tenmo/sgd.mojo:403`) mapped all GPU parameters and
gradients to CPU via `map_to_host()`, applied momentum, weight decay, and
gradient updates on the CPU, then wrote back to GPU via `param_ds.sync()`.
This round-tripped every parameter across PCIe on every batch.

**Fix (`tenmo/kernels/sgd_kernel.mojo`):** Two GPU kernels — `sgd_step_no_momentum_kernel`
and `sgd_step_momentum_kernel` — apply updates in-place on GPU with a grid-stride
loop (256 threads/block). The launcher `SGDStep[dtype].launch_no_momentum()` /
`.launch_momentum()` takes NDBuffer copies (cheap refcount bump) and enqueues
async (`sync=False`). Replaces 6 `map_to_host()` round-trips per batch for the
3-layer MNIST model (104K params) with 6 async GPU kernel launches.

**Remaining:** `clip_gradients()` still uses `map_to_host()` on GPU — low priority
since MNIST doesn't use clipping (both values default to 0).

---

### 3. Matmul forward sync-threading bug

**Status: 🟢 FIXED**

**Problem:** `Matmul.forward` (`tenmo/matmul.mojo:205`) accepted `sync` but
did not pass it to any sub-dispatch: `MatmulNd.forward` (line 224),
`Matmul2d.forward` (line 152 via MatmulNd short-circuit),
`VectorMatmulNd.forward` (line 218), `MatrixVectorMulNd.forward` (line 221),
or `A.dot` (line 215). All dispatch targets accept `sync` and some use it
(e.g. GPU matmul launchers).

**Fix:** Thread `sync` through every dispatch point. Six `sync=sync` keyword
additions across `Matmul.forward` (dot/vm/mv/mm) and `MatmulNd.forward`
(short-circuit to `Matmul2d`).

No functional impact — NDBuffer dispatch methods (including matmul) all
default `sync=False`, so the pipeline was already async. Correctness fix
for any future caller that explicitly passes `sync=True`.

---

### Additional completed work

- ~~Remove `synchronize()` from GPU launchers~~ **🟢 DONE** — All 41 launchers
  default `sync=False` (Phase 1)
- ~~Layer structs proactively share buffers~~ **🟢 DONE** — `add_ancestry` no
  longer deep-copies weights that are already shared
- ~~Add SIMD fast path to CPU `reduce_cpu`~~ **🟢 DONE** — ~840× speedup for
  contiguous suffix-axis reductions
- ~~Fused crossentropy backward kernel~~ **🟢 DONE** — op=9 reduced from
  ~15.5ms to ~0.5ms
- ~~`NDBuffer.to_device()` default `sync=True`→`False`~~ **🟢 DONE**
- ~~Redundant `synchronize()` in `DeviceState.into()`~~ **🟢 DONE** — removed
- ~~SGD GPU kernel (dense no-momentum + momentum)~~ **🟢 DONE** — replaces
  `map_to_host()` round-trips with async GPU kernel launches per parameter
- ~~Matmul forward sync-threading~~ **🟢 DONE** — `sync` threaded through
  `Matmul.forward` → all 4 dispatch paths (dot/vm/mv/mm)

### Remaining work (low priority / not blocking MNIST GPU)

| Item | Priority | Why left |
|---|---|---|
| `clip_gradients()` GPU kernel (`sgd.mojo:291-400`) | Low | Not used by MNIST (both clip_norm/clip_value default 0). Still uses `map_to_host()`. |
| Sparse SGD GPU kernel (sgd.mojo:440-479) | Low | Only needed for embedding/word2vec sparse training. Not used by MLP. |
| `backward(sync=True)` redundant sync in same-stream execution | Very low | Kept as safety fence. Removal is 0.5µs savings. Not measurable. |
| Tensor dunders `sync=True` default → not threaded to NDBuffer | Very low | No functional impact (NDBuffer defaults `sync=False`). Pipeline runs async regardless. |

---

## CPU Optimization Opportunities

Ranked by estimated performance impact. Covers all CPU-only code paths found during codebase audit.

### CPU-1 [H1] 1. Fuse crossentropy CPU path — 10-20× speedup

**File:** `tenmo/crossentropy.mojo`, `_forward_cpu_impl` (L472-547)

**Current:** CPU crossentropy goes through `compute_log_softmax_and_softmax()` → `onehot()` → NLL via `arithmetic_ops[Multiply]` + `SumMeanReduction.sum` + `unary_ops[NEGATE]` — ~18 separate NDBuffer operations (each allocating, dispatching, looping) for a single loss computation.

**Fix:** Write a fused `_forward_cpu_fused()` that computes softmax (max subtraction, exp, sum, divide), target NLL (or class-index lookup), and optional label smoothing in a single CPU pass over the (M, C) logits and (M,) targets. Mirror the already-written GPU fused kernel (`crossentropy_fused_kernel.mojo`).

**Also:** `_forward_cpu_impl` (L472) is duplicated between `track_grad=True` and `track_grad=False` paths in `CEClassIndicesForward.forward` (L654 and L665) — factor out.

**Estimated impact:** 10-20× faster CPU crossentropy during training. Each call replaces ~18 full NDBuffer passes with 2-3 fused passes.

**Effort:** Medium. Need to write the fused loop, handle ignore_index and label_smoothing, keep the decomposed path as fallback.

---

### CPU-2 [H2] 2. Unify reduction CPU loops — 2-5× compile-time, 1.5× runtime

**Files:**
- `tenmo/sum_mean_reduction.mojo` → `reduce_cpu` (L69-145): SUM/MEAN
- `tenmo/welford.mojo` → `forward_cpu` (L73-122): Welford mean+variance
- `tenmo/softmax.mojo` → `_log_sum_cpu` (L64-92): log-sum-exp
- `tenmo/minmax.mojo` → (inferred): min/max reduction

**Current:** All four implement the same double-nested coord-by-coord reduction pattern independently:
```
for out_coord in out_shape:
    var accum = 0  # or max, or min, or Pair
    for red_coord in reduction_axes_shape:
        var self_coord = out_coord.replace/insert(axes, red_coord)
        accum += ndb[self_coord]  # or accum = max(accum, ...)
    out[out_coord] = fn(accum)
```

~30 lines each, 4+ copies. The index arithmetic (`out_coord.replace/insert`) and the double-nested coordinate iteration are identical across all.

**Fix:** Create a shared `_reduce_cpu_loop[op_code](ndb, axes, keepdims, out_shape)` that takes a comptime op_code for the reduction operation (SUM, MEAN, MAX, MIN, LOG_SUM_EXP, WELFORD). This eliminates duplication and makes optimizations (SIMD suffix fast path, parallelization) apply to all reductions at once.

**Estimated impact:** ~30% compile-time reduction from eliminating duplicated comptime reduction code. Runtime benefits when suffix fast path is enhanced.

**Effort:** Medium. Need to design the shared interface, migrate all callers, handle differing accumulator types (Scalar vs Pair for Welford).

---

### CPU-3 [H3] 3. Share `simd_op` helper in CPU broadcast dispatch

**Files:**
- `tenmo/cpu_broadcast.mojo` → `apply_nd` (L167-476): CPU broadcast (3 tiers)
- `tenmo/kernels/binary_ops_kernel.mojo` → 6 GPU kernels (L28-937): GPU broadcast (4 paths)

**Current:** `cpu_broadcast.mojo` has its own inline comptime op_code dispatch (ADD, SUB, MUL, DIV, MAX, MIN, ReverseSubtract, ReverseDivide — 8 opcodes) in:
- Tier 1 SIMD path (L249-276): 8 comptime branches
- Tier 2 SIMD path (L354-408): 8 comptime branches × 2 operands
- Tier 3 scalar path (L459-460): calls `scalar_fn`

Meanwhile, `kernel_helpers.mojo` has `simd_op[op_code]()` and `scalar_op[op_code]()` that handle the same 8 opcodes. `cpu_broadcast.mojo` does not use them — it duplicates the dispatch logic inline.

**Fix:** Replace the inline comptime branches in `cpu_broadcast.mojo` Tier 1 and Tier 2 with calls to the shared `simd_op` / `scalar_op` helpers. This eliminates ~200 lines of duplicated comptime conditionals and ensures consistent op-code handling across CPU and GPU paths.

**Bug note:** The current code hands uncommon ops (SIGMOID_BACKWARD, TANH_BACKWARD, POW) to per-element `ScalarOps.scalar_fn` in Tier 1 (L264-275) and Tier 2 (L393-408) instead of the more efficient `simd_op` — a missed optimization.

**Effort:** Low-Medium. Mechanical replacement of comptime branches with helper calls.

---

### CPU-4 [H4] 4. Factor binary ops GPU kernel vs inplace kernel — ~400 lines duplicated

**Files:**
- `tenmo/kernels/binary_ops_kernel.mojo` (937 lines): 6 out-of-place kernels
- `tenmo/kernels/binary_inplace_ops_kernel.mojo` (723 lines): 6 in-place kernels

**Current:** Two files, each with 6 kernel variants that share 100% identical coordinate decomposition logic (i % inner_dim, outer_remaining, row-boundary handling, per-lane scalar fallback). The only difference: out-of-place stores to `result`, in-place stores back to `A`.

- `arithmetic_ops_both_contiguous`: binary L28 vs inplace L20 — identical except store target
- `arithmetic_ops_A_contiguous_lastdim_contiguous_B`: binary L287 vs inplace L84 — identical coord decomp
- `arithmetic_ops_A_contiguous`: binary L350 vs inplace L144 — identical
- `arithmetic_ops_B_contiguous`: binary L453 vs inplace L223 — identical
- `both_strided`: binary L556 vs inplace L302 — identical
- `both_strided_inner_path`: binary L687 vs inplace L424 — identical

**Fix:** Factor the coordinate decomposition and inner loop body into a shared comptime helper parameterized by inplace: Bool. The helper handles the SIMD loop, stride computation, and remainder — the caller just specifies which buffer receives the result.

**Alternatively:** Generate one file from the other via a comptime parameter (`mode: Int = 0 for out-of-place, 1 for in-place`).

**Effort:** Medium. Requires careful refactoring of ~400 lines. High maintenance value — the in-place variants were recently bug-fixed independently of out-of-place.

---

### CPU-5 [H5] 5. Factor scalar ops GPU kernel vs inplace kernel — ~100 lines duplicated

**Files:**
- `tenmo/kernels/scalar_ops_kernel.mojo` (336 lines): `scalar_ops`, `pow_op_f32/f64`, `ScalarOperations.launch`
- `tenmo/kernels/scalar_inplace_ops_kernel.mojo` (158 lines): `inplace_scalar_ops`, `InplaceScalarOperations.launch`

**Current:** The inner loop is identical (SIMD with 2×simd_width chunking, comptime op dispatch, tail handling). The launch method (L92-137 vs L193-250) is identical — same launch_config, same compile+enqueue pattern. Inplace supports 4 opcodes (Add/Sub/Mul/Div) vs out-of-place's 6 (Add/Sub/RevSub/Mul/Div/MAX/MIN).

**Fix:** Factor into a shared `scalar_ops_launch[inplace: Bool](...)`. Also check if the missing `ReverseSubtract`/`MAX`/`MIN` in inplace is a correctness gap (likely unused, but inconsistent).

**Effort:** Low.

---

### CPU-6 [H6] 6. Template contiguous-check pattern in `ndbuffer.mojo` — 6× duplication

**File:** `tenmo/ndbuffer.mojo`

**Current:** The same `if is_contiguous(): Buffer-level SIMD else: index_iterator scalar` pattern appears at least 6 times:
- `arithmetic_ops_cpu` (L2001-2064)
- `scalar_ops_cpu` (L2158-2186)
- `unary_ops_cpu` (L2219-2239)
- `float_unary_ops_cpu` (L2276-2298)
- `clamp` (L2300-2325)
- `clamp_fixed_minmax` (L2327-2340)

Each has ~30 lines of: check contiguity, call Buffer method, or iterate via index_iterator with per-element scalar operations.

**Fix:** Create a shared `_cpu_map[op_code](self, args...) -> NDBuffer` template:
```mojo
if self.is_contiguous():
    return contig_path  # Buffer-level SIMD
else:
    return noncontig_path  # index_iterator loop
```

Eliminates ~180 lines of duplicated pattern.

**Effort:** Low.

---

### CPU-7 [H7] 7. Share comptime op dispatch between `Buffer.arithmetic_ops` and `Buffer.arithmetic_ops_scalar`

**File:** `tenmo/buffers.mojo`

**Current:** Two methods (L656-788 and L791-875) both have:
- SIMD loop: load → comptime op dispatch → store
- Scalar tail: comptime op dispatch

The op dispatch chains (Multiply, Add, Subtract, Divide, plus backward ops) are independent but near-identical. Difference: binary (two input buffers) vs scalar (one buffer + scalar value).

**Fix:** Factor the comptime op dispatch into a shared helper that works for both `SIMD + SIMD` and `SIMD + Scalar` forms. 8 opcodes × 2 (SIMD + scalar tail) × 2 (binary + scalar) = 32 comptime branches to maintain.

**Effort:** Low.

---

### CPU-8 [H8] 8. Eliminate unnecessary Buffer copy in broadcast scalar path — 1.5-2× speedup

**File:** `tenmo/cpu_broadcast.mojo`, `apply_scalar` (L88-144)

**Current:**
```mojo
if is_contiguous:
    buffer = (
        b.buffer.copied(offset, offset + numels)  # <-- ALWAYS COPIES
    ).arithmetic_ops_scalar[op_code](item)
```

Calls `copied()` which allocates a new Buffer and memcpy's the data, then applies the scalar op to produce another Buffer. This is **two allocations + one full memcpy + one SIMD pass** for every scalar broadcast operation.

**Fix:** Pass offset directly to `arithmetic_ops_scalar`:
```mojo
buffer = b.buffer.arithmetic_ops_scalar[op_code](item, offset, offset + numels)
```

`arithmetic_ops_scalar` already accepts `start`/`end` parameters (L796). This avoids the intermediate copy entirely.

**Estimated impact:** 1.5-2× speedup for every scalar broadcast operation (bias_add, scalar arithmetic on non-contiguous tensors, etc.). Low effort — 3 lines to change.

**Effort:** Low.

---

### CPU-9 [H9] 9. Parallelize CPU reduction suffix-axis fast path — 2-4× speedup

**File:** `tenmo/sum_mean_reduction.mojo`, `reduce_cpu` (L114-128)

**Current:** The suffix-axis fast path calls `Buffer.sum()` per output element sequentially:
```mojo
for oi in range(num_out):
    var base = ndb.offset + oi * reduced_numels
    out.buffer[oi] = ndb.buffer.sum(base, base + reduced_numels)
```

Each output element's reduction is independent — the outer loop is purely serial.

**Fix:** Use `parallelize` to process output rows in parallel across CPU cores — identical to matmul's tile processing pattern. Each thread handles a subset of output elements.

```mojo
parallelize[reduce_row_fn](num_out, num_physical_cores())
```

**Estimated impact:** 2-4× for reductions over large innermost dimensions (e.g., softmax over 252K vocab in word2vec, or large classifier heads). The SIMD suffix path already runs at ~0.09ms for 512×1024 sum — parallelization would help for much larger tensors.

**Effort:** Low.

---

### CPU-10 [M10] 10. Factor matmul inner j-loop — remove ~600 lines duplication

**File:** `tenmo/matmul_cpu.mojo`

**Current:** The inner j-loop (SIMD unrolled + SIMD single + scalar tail with k_tile==0 branching) is written 4 times:
- MmCpu2d Path 1a (L275-351): A contiguous
- MmCpu2d Path 1b (L401-468): A non-contiguous
- MmCpuNd Path 1a (L761-828): A contiguous
- MmCpuNd Path 1b (L859-925): A non-contiguous

Each is ~75 lines, total ~300 lines of nearly-identical SIMD code. The only difference is how A elements are loaded:
- Contiguous: `A_data[a_row_base + k]`
- Non-contiguous: `A_data[a_row_base + k * A_stride1]`

Factor into a single comptime-parameterized function:
```mojo
def _matmul_inner_loop[A_contiguous: Bool](params...):
    # SIMD unrolled over n-dim
    for j in range(0, n_tile_minus_last, simdwidth):
        if A_contiguous:
            A_elements = A_data.load[width=simdwidth](a_row_base + k)
        else:
            for lane in range(simdwidth):
                A_elements[lane] = A_data[a_row_base + (k + lane) * A_stride1]
        ...
```

Also: the 18-combination tile dispatch (`tile_m → tile_n → tile_p`) in `MmCpu2d.tiled_matmul` (L80-135) and `MmCpuNd.tiled_matmul` (L568-625) is identical — factor into a shared function.

**Estimated impact:** Significant compile-time reduction (currently 36 nearly-identical comptime instantiations). Maintenance: one copy of the SIMD inner loop to tune and bug-fix.

**Effort:** Medium.

---

### CPU-11 [L11] 11. Gather embedding-bag: avoid duplicate index re-reads

**File:** `tenmo/gather.mojo`, `_gather_copy` embedding-bag path (L438-451)

**Current:**
```mojo
var sorted = normalized.sorted()
for row in sorted:
    var row_offset = base_offset + row * cols
    res_buffer += self_buffer[row_offset : row_offset + cols]
```

Sorts indices (O(K log K)) then does K Buffer `+=` operations. If indices have duplicates, duplicates re-read the same row multiple times.

**Fix:** Use a row-count accumulator: first pass builds `Dict[Int, Int]` (index → count), second pass iterates unique indices with `res_buffer += count * embedding_row`. Eliminates duplicate reads + Buffer `+=` overhead for repeated indices.

**Estimated impact:** Meaningful when embedding-bag has many repeated indices (common in NLP bag-of-words). 1.5× for datasets with average duplicate rate.

**Effort:** Low.

---

### CPU-12 [L12] 12. Dropout CPU: replace scalar-per-lane RNG with SIMD Philox — 1.2-2× speedup

**File:** `tenmo/dropout.mojo`, CPU path (L186-205)

**Current:**
```mojo
for lane in range(simd_w):
    rand_vec[lane] = random_float64(0.0, 1.0).cast[Self.dtype]()
```

The RNG call is scalar per lane — `random_float64` returns one value, called simd_width times per SIMD iteration. On AVX2 (simd_width=8), 7 of 8 RNG calls are wasted overhead.

**Fix:** Use `PhiloxRandom` from `std.random.philox` which supports SIMD-width generation via `.rand()`:
```mojo
var philox = PhiloxRandom(seed, increment)
var rand_vec = philox.rand[simd_w]()
```

**Estimated impact:** 1.2-2× for dropout layers during training. Critical for models using dropout regularization (common in MLP, Transformer).

**Effort:** Low.

---

### CPU-13 [L13] 13. Noncontiguous element-wise SIMD: partial SIMD when one operand is contiguous

**File:** `tenmo/ndbuffer.mojo`, `arithmetic_ops_cpu` (L2001-2064) and 3 other methods

**Current:** When `is_contiguous()` is false, ALL element-wise ops fall back to scalar `index_iterator` loop with per-element function calls. No SIMD is used even when one of the two operands IS contiguous.

**Fix:** For `arithmetic_ops_cpu` when one operand is contiguous, use SIMD for the contiguous side with per-element index computation for the non-contiguous side:
```mojo
if self.is_contiguous() and not other.is_contiguous():
    # SIMD load from self, scalar index-lookup from other
elif other.is_contiguous() and not self.is_contiguous():
    # scalar index-lookup from self, SIMD load from other
```

**Estimated impact:** Low — noncontiguous views are rare in hot paths. But `broadcast_to()` followed by element-wise ops creates strided views, which is common. Relevant for broadcasting edge cases.

**Effort:** Medium.

---

### CPU-14 [L14] 14. LayerNorm CPU: parallelize outer row loop — 2-4× speedup

**File:** `tenmo/layernorm.mojo`, `normalize_cpu` (L118-134)

**Current:**
```mojo
for row in range(outer_size):
    for i in range(D):
        var x_i = x.buffer[x.offset + row_base + i]
```

The outer `row` loop is serial. For large batch sizes (B=1024, D=768) this is 1024 serial iterations of a D=768 inner loop. Each row is fully independent.

**Fix:** Use `parallelize[process_row](outer_size, num_physical_cores())` — same pattern as matmul's tile processing. Each thread handles a subset of rows. Inner loop over D can also use SIMD via `float_unary_ops` on row slices.

**Estimated impact:** 2-4× for LayerNorm at scale. Important for Transformer training on CPU.

**Effort:** Low.

---

### CPU-15 [L15] 15. Filler scalar fill: use SIMD store instead of scalar loop — 2-3× speedup

**File:** `tenmo/filler.mojo`, `_fill_scalar_cpu` (L162-167)

**Current:**
```mojo
if strides.is_contiguous(shape):
    ref buffer = target.data_buffer()
    var end = absolute_offset + shape.num_elements()
    for idx in range(absolute_offset, end):
        buffer[idx] = value
```

Scalar loop filling contiguous memory with a constant value. No SIMD.

**Fix:** Use SIMD store:
```mojo
var vec = SIMD[Self.dtype, simd_width](value)
for idx in range(0, num_elements, simd_width):
    buffer.store[width=simd_width](absolute_offset + idx, vec)
```

Same fix applies to the non-contiguous `IndexIterator` path (L170-176).

**Estimated impact:** 2-3× for buffer initialization / fill operations. Memory-bound — speedup depends on memory bandwidth.

**Effort:** Low.

---

### CPU-16 [L16] 16. Matmul: eliminate `k_tile==0` branch from inner loop — 5-10% inner loop speedup

**File:** `tenmo/matmul_cpu.mojo`, inner j-loop in all 4 paths (e.g., L280, L406, L766, L864)

**Current:**
```mojo
if k_tile == 0:
    acc0 = SIMD[Self.dtype, simdwidth](0)    # first tile: no C to load
else:
    acc0 = C_data.load[width=simdwidth](cj)  # subsequent tiles: accumulate
```

The `k_tile == 0` check is evaluated every SIMD vector iteration of the inner loop. It's a runtime check for the first k-tile only.

**Fix:** Split the inner loop into two phases:
1. First k-tile (k_tile == 0): pure accumulate from zero — no C load, no C store
2. Remaining k-tiles: load C, accumulate, store back

A comment in `matmul_cpu.mojo` (L159-160) already acknowledges this as a known optimization. Same pattern appears 6 times (2 paths × 2D/ND × 2 sub-loops).

**Estimated impact:** 5-10% inner loop speedup for small matrices or many k-tiles. Negligible for large contiguous matmuls.

**Effort:** Low.

---

### CPU-17 [L17] 17. `all_close` CPU fallback: missing SIMD path

**Files:** `tenmo/kernels/compare_kernel.mojo` (L39-220), `tenmo/buffers.mojo`, `tenmo/ndbuffer.mojo`

**Current:** `all_close` GPU kernel (L39) is a device function using `thread_idx.x`, `block_dim.x`, etc. — can't run on CPU. The `AllClose.launch()` method only has a GPU launch path. On CPU, the Tensor-level `.all_close()` may fall through to a slow decomposed path.

**Fix:** Add a CPU SIMD `all_close` function in `buffers.mojo` that:
1. Vector-compares element pairs with `SIMD.gt(atol_mask)` / `SIMD.lt(atol_mask)`
2. Accumulates mismatches via `reduce_add`
3. Short-circuits when mismatch count exceeds `max_mismatches`
4. Handles NaN (NaN != NaN per IEEE 754)

**Estimated impact:** 3-5× for test assertions. Test-only, not hot-path training.

**Effort:** Low.

---

### CPU-18 [L18] 18. Pooling/CNN CPU: parallelize outer loops — 2-4× speedup

**Files:** `tenmo/pooling.mojo`, `tenmo/cnn.mojo`

**Current:** Conv2d, MaxPool2d, AvgPool2d CPU forward/backward use raw `data_ptr()` loops with nested for loops over batch, input channels, output channels, spatial dimensions (e.g., `cnn.mojo` L375-378, L748-749, L844-846). All loops are serial.

**Fix:** Use `parallelize` over batch or output-channel dimensions. Each batch element or output channel is independent.

**Estimated impact:** 2-4× for CPU Conv2d/MaxPool2d at scale. Rarely trained on CPU at scale, but useful for small models and inference.

**Effort:** Medium.
