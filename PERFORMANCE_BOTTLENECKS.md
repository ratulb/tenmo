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

**Files:** `tenmo/cpu_arithmetics.mojo:52-405` (`CpuArithmeticOps.broadcast*`)

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

### 26. Fused CrossEntropy GPU kernel: class indices only — probabilities path has no fused kernel

**Files:**
- `tenmo/kernels/crossentropy_fused_kernel.mojo` — fused kernel (class indices only)
- `tenmo/crossentropy.mojo`, `CEProbabilitiesForward.forward` (L798-873)

**Problem (class indices, existing):** The fused kernel unconditionally computes
exp, log-sum-exp, and normalized softmax for ALL rows, even when `ignore_index`
makes the target invalid. Thread 0 skips the atomic accumulation for ignored
rows, but all threads in the block still run the shared-memory tree reduction
(max, sum_exp, sum_logits) and write back normalized softmax. On datasets with
many ignored positions (e.g. padded sequences), this wastes GPU cycles.

**Fix (class indices):** Add a warp-level or block-level early exit at the top
of Phase 2 when `target[row] == ignore_index`. The block can skip the softmax
computation and directly write 0 to `per_sample_loss[row]` and 0-filled softmax
to `softmax_out[row * C : (row+1) * C]`. The `scalar_loss` and `valid_count`
atomics are already correctly guarded by the `is_valid` check.

**Problem (probabilities, NEW):** `CEProbabilitiesForward.forward` (L798-873)
has **no** fused GPU kernel at all. It always runs the decomposed path:
`compute_log_softmax_and_softmax()` (~7 GPU kernel launches) → label smooth
target (2) → `arithmetic_ops[Multiply]` (1) → `SumMeanReduction.sum` (1) →
`unary_ops[NEGATE]` (1) = **~12 separate GPU kernel launches** per loss call.
Each launch has overhead (~3-10µs) plus full allocation + dispatch. Unlike
`CEClassIndicesForward.forward` (L610) which dispatches to
`CrossEntropyFusedKernel.launch()` on GPU, the probabilities path has no
`comptime if has_accelerator()` guard.

**Fix (probabilities):** Write `CEProbabilitiesFusedKernel` — a fused GPU
forward kernel at `tenmo/kernels/crossentropy_fused_kernel.mojo` that computes
softmax, inline label-smoothed target (`target * (1-ls) + ls/C`), and
`-sum(smoothed_target * log_softmax, axis=1)` in a single GPU pass per row.
Design mirrors the class-indices kernel: M blocks, C threads, shared-memory
tree reduction for max/sum_exp. Returns (softmax_probs, per_sample_loss,
scalar_loss). Wire into `CEProbabilitiesForward.forward` with the same
`comptime if has_accelerator()` + `is_on_gpu()` pattern.

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
- ~~Fused crossentropy backward kernel (class indices)~~ **🟢 DONE** — op=9 reduced from
  ~15.5ms to ~0.5ms
- ~~`NDBuffer.to_device()` default `sync=True`→`False`~~ **🟢 DONE**
- ~~Redundant `synchronize()` in `DeviceState.into()`~~ **🟢 DONE** — removed
- ~~SGD GPU kernel (dense no-momentum + momentum)~~ **🟢 DONE** — replaces
  `map_to_host()` round-trips with async GPU kernel launches per parameter
- ~~Matmul forward sync-threading~~ **🟢 DONE** — `sync` threaded through
  `Matmul.forward` → all 4 dispatch paths (dot/vm/mv/mm)

### Remaining work (low priority / not blocking MNIST GPU)

| Item | Priority | Why left |
|---|---|---|---|
| `clip_gradients()` GPU kernel (`sgd.mojo:291-400`) | Low | Not used by MNIST (both clip_norm/clip_value default 0). Still uses `map_to_host()`. |
| Sparse SGD GPU kernel (sgd.mojo:440-479) | Low | Only needed for embedding/word2vec sparse training. Not used by MLP. |
| Fused GPU forward kernel for CEProbabilities (`crossentropy.mojo:829-845`) | Medium | Not used by MNIST (uses class-indices CE). ~12 GPU launches per loss call on probabilities path. |
| `backward(sync=True)` redundant sync in same-stream execution | Very low | Kept as safety fence. Removal is 0.5µs savings. Not measurable. |
| Tensor dunders `sync=True` default → not threaded to NDBuffer | Very low | No functional impact (NDBuffer defaults `sync=False`). Pipeline runs async regardless. |

---

## CPU Optimization Opportunities

Ranked by estimated performance impact. Covers all CPU-only code paths found during codebase audit.

| Item | Priority | Status |
|---|---|---|
| **CPU-1** Fuse crossentropy CPU paths | H1 | ✅ **DONE** |
| **CPU-2** Add SIMD suffix-contiguous fast paths (softmax, welford) | H2 | ✅ **DONE** |
| **CPU-3** Share `simd_op` helper in CPU broadcast dispatch | H3 | ✅ **DONE** |
| **CPU-4** Factor binary ops GPU kernel vs inplace | H4 | ❌ **CANCELLED** — opted for targeted porting between files instead |
| **—** Port outer-base decomposition to out-of-place `both_strided` | H4 | ✅ **DONE** (same session as CPU-4 cancellation) |
| **—** Fix scalar-splat bug in out-of-place `B_contiguous` | H4 | ✅ **DONE** (same session) |
| **—** Add bias_add kernel to in-place `A_contiguous` launcher | H4 | ✅ **DONE** (same session) |
| **CPU-5** Factor scalar ops — single source of truth for op dispatch (5a ✅ 5b✅ 5c✅ 5d✅) | H5 | ✅ 5a ✅ / 5b ✅ / 5c ✅ / 5d ✅ |
| **CPU-6** Phase 3 — Standalone free functions for remaining ScalarOps dispatch chains | H6 | ✅ **DONE** |
| **CPU-7** Phase 1 — Replace buffer comptime op dispatch with simd_op/scalar_op (12→0 copies) | H7 | ✅ **DONE** |
| **CPU-8** Eliminate Buffer copy in broadcast scalar path | H8 | ✅ **FIXED** |
| **CPU-9** Parallelize CPU reduction suffix-axis fast path | H9 | ✅ **DONE** |
| **CPU-10** Factor matmul inner j-loop | M10 | ❌ Pending |
| **CPU-11** Gather embedding-bag duplicate reads | L11 | ❌ Pending |
| **CPU-12** Dropout CPU: SIMD Philox RNG | L12 | ❌ Pending |
| **CPU-13** Noncontiguous element-wise SIMD (partial) | L13 | ❌ Pending |
| **CPU-14** LayerNorm CPU: parallelize outer row loop | L14 | ❌ Pending |
| **CPU-15** Filler scalar fill: SIMD store | L15 | ❌ Pending |
| **CPU-16** Matmul: eliminate k_tile==0 branch | L16 | ❌ Pending |
| **CPU-17** all_close CPU SIMD path | L17 | ✅ **FIXED** (already in place) |
| **CPU-18** Pooling/CNN CPU: parallelize outer loops | L18 | ❌ Pending |

### CPU-1 [H1] 1. Fuse crossentropy CPU paths (class indices + probabilities) — 10-20× speedup

**Files:** `tenmo/crossentropy.mojo` — two distinct paths:
- **Class indices** `_forward_cpu_impl` (L472-547): ~18 NDBuffer ops
- **Probabilities** `CEProbabilitiesForward.forward` (L798-873): ~10-14 NDBuffer ops

**Current (class indices):** `compute_log_softmax_and_softmax()` (7 ops: minmax, subtract, exp, log_sum, sum, subtract, divide) → `onehot()` (1) → `build_ignore_mask` (2) → `SumMeanReduction.sum` (1) → `arithmetic_ops[Multiply]` (1) → `SumMeanReduction.sum` (1) → `unary_ops[NEGATE]` (1) → optional label smoothing multiply/subtract (2-3) → `losses * ignore_mask` (1) → `apply_reduction` (1-2) = ~18 ops.

**Current (probabilities):** `compute_log_softmax_and_softmax()` (7 ops) → label smooth target (2) → `arithmetic_ops[Multiply]` (1) → `SumMeanReduction.sum` (1) → `unary_ops[NEGATE]` (1) → `apply_reduction` (1-2) = ~14 ops.

**Fix:** Write two fused CPU functions — one per path — that mirror the GPU fused kernel pattern (`crossentropy_fused_kernel.mojo`). Below is the full task breakdown:

---

### CPU-1 Todo — Fused CPU CrossEntropy — ✅ DONE (Phases A-D)

**Status:** All 139 CE tests pass. Both fused forward functions (`_fused_forward_class_indices` and `_fused_forward_probabilities`) are wired into their respective forward structs, replacing the decomposed `_forward_cpu_impl` and `compute_log_softmax_and_softmax`+label smoothing paths.

**Phase A: Core fused loops (eliminate 15–21 allocations → 2–3 passes) — ✅ DONE**

- [x] **A1. Fused class-indices forward `_fused_forward_class_indices`** (`crossentropy.mojo:283`)
  3 passes per row: max → exp+sum+loss → softmax normalize. O(1) loss via direct target index lookup. Replaces ~18 NDBuffer ops.

- [x] **A2. Fused probabilities forward `_fused_forward_probabilities`** (`crossentropy.mojo:467`)
  3 passes per row: max → exp+sum → softmax+smoothed_target+loss. O(C) loss via element-wise multiply-accumulate. Replaces ~14 NDBuffer ops.

**Phase B: Allocation eliminations — ✅ DONE**

- [x] **B1. Skip onehot** — direct `logits[row, target[row]]` strided read in row loop.
- [x] **B2. Skip `ignore_mask`** — `if target[row] != ignore_index` inline in row loop.
- [x] **B3. Skip `valid_count` reduction** — increment `valid_count` as scalar in row loop.
- [x] **B4. Skip redundant `log_sum` + `sum`** — compute `sum_exp` once, `log(sum_exp)` once.
- [x] **B5. Conditional softmax** — `comptime if track_grad:` gating softmax write in both functions.
- [x] **B6. Inline loss reduction** — accumulate `scalar_loss` or write `per_sample_loss[row]` directly in row loop.
- [x] **B7. Inline label smoothing** — compute adjusted loss in-place using `sum_logits / C - max_val - log_sum_exp`.

**Phase C: Code organization — ✅ DONE**

- [x] **C1. Factor duplication** — Forward structs call the fused function once (CPU path in single call, no track_grad branching).
- [x] **C2. Decomposed fallback** — `_forward_cpu_impl` (L779) and `compute_log_softmax_and_softmax` (L409) preserved as fallback.

**Phase D: Follow-up refinements — ✅ DONE**

- [x] **D1. SIMD inner loop** — Both functions use `comptime SIMD_WIDTH = simd_width_of[Self.dtype]()` with `simd_end` guard for remainder handling.
- [x] **D2. Early exit** — `comptime if track_grad:` eliminates softmax and smoothed_target allocations in eval mode. `reduction='none'` + `track_grad=False` allocates only `per_sample_loss`.

**Files:** `tenmo/crossentropy.mojo` — `_fused_forward_class_indices` (L283), `_fused_forward_probabilities` (L467). Old decomposed path preserved at L779.

**Estimated impact:** ~10-20× faster CPU crossentropy (both paths) by replacing ~10-18 full NDBuffer passes with 3 fused SIMD passes per row, eliminating 15-21 heap allocations per loss call.

---

### CPU-2 [H2] 2. Add missing SIMD suffix-contiguous fast paths — ~1.5× runtime for affected ops

**Files:**
- `tenmo/softmax.mojo` → `_log_sum_cpu` (L64-103): log-sum-exp (**SIMD added**)
- `tenmo/welford.mojo` → `forward_cpu` (L73-130): Welford mean+variance (**suffix-contiguous path added**)
- `tenmo/sum_mean_reduction.mojo` → `reduce_cpu` (L69-145): SUM/MEAN (already had SIMD, unchanged)
- `tenmo/minmax_reducer.mojo` → already parallelized, out of scope

**Changes:**

| Function | Before | After |
|---|---|---|
| `_log_sum_cpu` | Scalar coord loop (strided index per element) | Suffix-contiguous SIMD fast path + SIMD scalar case. exp() applied to SIMD vectors, `.reduce_add()` for horizontal sum. |
| `welford.forward_cpu` | Scalar coord loop (strided index per element) | Suffix-contiguous fast path eliminates coord index overhead. No SIMD — Welford recurrence (`mean += delta / count`) is inherently sequential. |
| `sum_mean_reduction.reduce_cpu` | Already had suffix-contiguous SIMD via `Buffer.sum()` | Unchanged |
| scalar case (`_log_sum_cpu`, `out_shape == Shape()`) | Element-by-element `ndb.buffer[index]` | SIMD vector loop with scalar remainder |

**Why not a shared loop:** `comptime if` does not suppress type-checking of dead branches — Mojo still resolves `exp(int_val)` and fails. Factoring would require two separate entry points (int + float) connected via `where` constraints that cascade to all callers, multiplying complexity for ~3 lines of saved loop header code. Individual fast paths are simpler and isolate the `where` constraint to float-only functions that already have it.

**Estimated impact:**
- `_log_sum_cpu` (softmax forward): ~2-4× faster on contiguous suffix reductions (common case: last-axis reduction on post-Linear activations). exp + reduce_add dominates the original coord-index computation.
- `welford.forward_cpu`: ~1.5-2× faster on contiguous suffix reductions (eliminates coord-index overhead). SIMD not applicable due to recurrence.
- Scalar log-sum-exp case: SIMD acceleration for global reductions.

**Effort:** Low. ~30 lines added across two files.

---

### CPU-3 [H3] 3. Share `simd_op` helper in CPU broadcast dispatch ✅ DONE

**Files:**
- `tenmo/cpu_broadcast.mojo` → `apply_nd` (3 tiers)
- `tenmo/shared/scalar_ops.mojo` → `simd_op`, `scalar_op` (standalone dispatch helpers)
- `tenmo/kernels/binary_ops_kernel.mojo`, `binary_inplace_ops_kernel.mojo` → GPU callers

**What was done:**
1. Added `ReverseSubtract`, `ReverseDivide`, `MAX`, `MIN`, `POW` to `simd_op` and `scalar_op` in `tenmo/shared/scalar_ops.mojo` (were missing).
2. Moved `simd_op`/`scalar_op` from `tenmo/kernels/kernel_helpers.mojo` to `tenmo/shared/scalar_ops.mojo` as standalone functions (precursor). GPU kernel files already import from there.
3. Replaced Tier 1 inline dispatch (8 comptime branches + per-element scalar fallback for uncommon ops) → single `simd_op` call.
4. Replaced Tier 2 inline dispatch (8 comptime branches × operand-ordering + per-element scalar fallback) → two `simd_op` calls with `a_broadcasts_last` guard.
5. Bonus: uncommon ops (SIGMOID_BACKWARD, TANH_BACKWARD, SQRT_BACKWARD, POW, LOG_BACKWARD) now get SIMD instead of per-element scalar — faster.

**Effort:** Low-Medium. Mechanical replacement of comptime branches with helper calls.

---

### CPU-4 [H4] 4. Factor binary ops GPU kernel vs inplace kernel — ~400 lines duplicated ❌ CANCELLED

**Reason:** In-place versions have outer-base decomposition optimizations that out-of-place versions lack (and vice versa — out-of-place has the `lastdim_contiguous_B` bias_add path that in-place lacks). Both are correct in both files; they just ended up at different optimization levels. Unifying would require either accepting the slower path in one mode or adding comptime branches that negate the savings. Not worth the complexity — focus on porting optimizations between the two files individually.

---

### CPU-5 [H5] 5. Scalar ops — single source of truth for comptime op dispatch

**Files:**
- `tenmo/kernels/scalar_ops_kernel.mojo` — GPU `scalar_ops` + `scalar_ops_strided` + `ScalarOperations.launch`
- `tenmo/kernels/scalar_inplace_ops_kernel.mojo` — GPU `inplace_scalar_ops` + `inplace_scalar_ops_strided` + `InplaceScalarOperations.launch`
- `tenmo/shared/scalar_ops.mojo` — `simd_op` / `scalar_op` standalone dispatch functions ✅ established
- `tenmo/ndbuffer.mojo` — CPU and GPU dispatch (*_cpu methods, GPU dispatch at L1828-1842)
- `tenmo/buffers.mojo` — CPU `arithmetic_ops`, `arithmetic_ops_scalar`, `inplace_ops`, `inplace_ops_scalar`

**Overall goal:** Eliminate all duplicated comptime `op_code` dispatch chains by routing through `simd_op`/`scalar_op` from `tenmo/shared/scalar_ops.mojo`. Completed — GPU kernel factoring (CPU-5a ✅), CPU buffer factoring (CPU-7 ✅), CPU NDBuffer strided fallback factoring (CPU-5c Phase 2 ✅). Every per-element operation now dispatches through a single shared function.

#### CPU-5a — GPU non-contiguous strided kernel ✅ DONE

**Severity: correctness bug** (not triggered by current tests, but latent).

**Fix applied:**
- **Inplace path:** Added `inplace_scalar_ops_strided` kernel — outer-base decomposition per SIMD vector. `InplaceScalarOperations.launch` checks `A.is_contiguous()`: contiguous → flat linear kernel, non-contiguous → strided kernel.
- **Out-of-place path:** Added `scalar_ops_strided` kernel (writes to separate result buffer, reads strided input). `ScalarOperations.launch` dispatches on contiguity.
- **GPU inline dispatch replaced:** Both `scalar_ops` and `inplace_scalar_ops` comptime chains replaced with `simd_op`/`scalar_op` calls.
- **Tests:** `tests/test_scalar_gpu.mojo` generated (222 GPU tests covering all ops, dims 1-4, contiguous/transposed/permuted/sliced layouts, f32/f64, edge cases). Registered as `scalar_gpu` in `execute.sh` + `GPU_TESTS` array.

#### CPU-5b — Missing opcodes in inplace kernel (GPU) ✅ DONE

**Status:** 🔶 In progress — implementation underway.

**Background:** We started by surveying which opcodes were missing from the GPU inplace kernel and expecting to simply "wire them up" through the generic `simd_op`/`scalar_op` dispatch. However, POW required dedicated typed kernels (float32/float64) using the GPU `pow()` intrinsic — the generic kernel's `a ** b` approach is not reliably supported for all dtypes on GPU backends. This led us to add 4 new kernels (contiguous + strided, f32 + f64) and a `launch_inplace_pow` method, mirroring the out-of-place `ScalarOperations.launch_pow` pattern.

**Why POW is special:** The out-of-place `ScalarOperations.launch_pow` (`tenmo/kernels/scalar_ops_kernel.mojo:366`) forces contiguous via `contiguous_device_state()` and writes to a fresh result buffer — safe for out-of-place since the result is a new allocation. But for inplace, we must write back to the original memory (which may be strided). Forcing contiguous, computing pow, then scattering back is wasteful — so we need both contiguous and strided pow kernels.

**Before / After:**

| Opcode | Before | After | Why no kernel change? |
|---|---|---|---|
| ReverseSubtract | Wrong (`scalar / vec_a` = ReverseDivide) | ✅ Correct via `simd_op`/`scalar_op` | Generic kernel passes `op_code` through |
| ReverseDivide | Wrong (`scalar / vec_a` again) | ✅ Correct via `simd_op`/`scalar_op` | Same — generic dispatch |
| MAX | Wrong (falls to Divide path) | ✅ Correct via `simd_op`/`scalar_op` | Same |
| MIN | Wrong (falls to Divide path) | ✅ Correct via `simd_op`/`scalar_op` | Same |
| POW | Wrong (falls to Divide path) | ✅ New typed kernels (f32/f64) | `pow()` intrinsic needed for GPU |

**Execution plan (added incrementally to `tenmo/kernels/scalar_inplace_ops_kernel.mojo`):**

1. `inplace_pow_op_f32` — contiguous float32, stores back to `A`
2. `inplace_pow_op_f64` — contiguous float64
3. `inplace_pow_op_f32_strided` — strided float32, outer-base decomposition
4. `inplace_pow_op_f64_strided` — strided float64
5. `InplaceScalarOperations.launch_inplace_pow` — dtype guard + contiguity dispatch

**Then in `tenmo/ndbuffer.mojo`:**
6. `inplace_scalar_ops` — add `comptime if op_code == POW:` branch

**Files affected:**
- `tenmo/kernels/scalar_inplace_ops_kernel.mojo` — +4 kernels, +1 launcher
- `tenmo/ndbuffer.mojo` — +1 comptime branch in `inplace_scalar_ops`

#### CPU-5c — Phase 2: Replace strided fallback dispatch in ndbuffer.mojo *cpu methods ✅ DONE

**What was done:** All 8 uses of `ScalarOps[Self.dtype].scalar_fn[op_code](...)` in the strided fallback paths of 4 `*_cpu` methods were replaced with `scalar_op[op_code, Self.dtype](...)`.

| Method | Lines | Uses replaced |
|---|---|---|
| `inplace_ops_cpu` | 1790–1814 | 3 (self-contig, other-contig, both-strided) |
| `inplace_scalar_ops_cpu` | 1864–1866 | 1 (self strided) |
| `arithmetic_ops_cpu` | 2034–2061 | 3 (self-contig, other-contig, both-strided) |
| `scalar_ops_cpu` | 2181–2183 | 1 (self strided) |

Both `ScalarOps.scalar_fn` and `scalar_op` implement the same comptime op dispatch. The change routes every CPU path through the shared `scalar_op` function, making the dispatch chain truly single-sourced.

**Bonus:** `Epsilon` fix for integer types (INT_MIN → 0) was applied during Phase 1 — uncovered by int64 divide tests in `test_buffers.mojo`.

#### CPU-5d — Test coverage (scalar GPU tests)

**Status:** ✅ DONE — `tests/test_scalar_gpu.mojo` generated via `scripts/gen_scalar_gpu_tests.py`. 222 tests covering all 8 ops (Add, Subtract, ReverseSubtract, Multiply, Divide, ReverseDivide, MAX, MIN), dimensions 1D-4D, contiguous + transposed + permuted + sliced layouts, f32 + f64, edge cases (negative scalars, medium/big sizes).

**Remaining gaps (not high priority):**
- **CPU scalar tests** — existing `test_ndbuffer_arithmetic_gpu.mojo` and `test_ndbuffer_inplace_gpu.mojo` test NDBuffer-level dispatch on GPU. CPU-side unit tests for `Buffer.arithmetic_ops*` are minimal.
- **POW GPU test** — POW is handled by separate f32/f64 kernels. Not tested in `test_scalar_gpu.mojo`.

---

### CPU-6 [H6] Standalone free functions for ScalarOps dispatch chains ✅ DONE

**What was done:** Added `unary_op`, `float_unary_op`, and `compare_op` standalone free functions to `tenmo/shared/scalar_ops.mojo` — mirroring the existing `scalar_op`/`simd_op` pattern. Then replaced the remaining 4 `ScalarOps` struct method calls in `ndbuffer.mojo`'s strided fallback paths:

| Method | Old call | New call | Lines |
|---|---|---|---|
| `unary_ops_cpu` | `ScalarOps[...].unary_fn_helper[op_code]` | `unary_op[op_code, Self.dtype]` | 2239 |
| `float_unary_ops_cpu` | `ScalarOps[...].float_unary_fn_helper[op_code, epsilon]` | `float_unary_op[op_code, Self.dtype, epsilon]` | 2298 |
| `compare_cpu` | `ScalarOps[...].compare_pair[op_code]` | `compare_op[op_code, Self.dtype]` | 2431 |
| `compare_scalar_cpu` | `ScalarOps[...].compare_pair[op_code]` | `compare_op[op_code, Self.dtype]` | 2487 |

After CPU-7 (Phase 1), CPU-5c (Phase 2), and CPU-6 (Phase 3), **every per-element operation** in `ndbuffer.mojo`'s strided fallback paths dispatches through a standalone free function — no struct-method dispatch chains remain.

**Skeleton duplication remains:** The `if is_contiguous(): buffer.SIMD else: idx-iter` skeleton still appears 18 times (5-15 lines each). Mojo's `def(…) thin` fn pointers cannot carry comptime parameters (op_code, dtype), so extracting the skeleton into a shared helper is not viable without Mojo macros or comptime-lambdas, which don't exist yet. The remaining boilerplate is mechanical but clear — each method reads as a simple `if is_contiguous()` branch with a well-known buffer-level SIMD path and a trivial per-element scalar fallback.

---

### CPU-7 [H7] Phase 1 — Replace all comptime op dispatch chains in `buffers.mojo` with `simd_op`/`scalar_op` calls ✅ DONE

**Files:** `tenmo/buffers.mojo`, `tenmo/common_utils.mojo`

**What was done:**

All 4 core arithmetic methods in `Buffer` had their SIMD vector block and scalar tail replaced with calls to `simd_op`/`scalar_op` from `tenmo/shared/scalar_ops.mojo`:

| Method | Before | After |
|---|---|---|
| `inplace_ops_scalar` | 14-line SIMD chain + 8-line tail chain | `simd_op(...)` + `scalar_op(...)` |
| `inplace_ops` | 30-line SIMD chain + 12-line tail chain (incl Overwrite) | `simd_op`/Overwrite if + `scalar_op`/Overwrite if |
| `arithmetic_ops` | 35-line SIMD chain + 30-line tail chain (7 opcodes) | `simd_op(...)` + `scalar_op(...)` |
| `arithmetic_ops_scalar` | 18-line SIMD chain + 13-line tail chain (8 opcodes) | `simd_op(...)` + `scalar_op(...)` |

~110 lines of duplicated comptime dispatch eliminated.

**Bonus bug fix:** `Epsilon` for integer types was returning `min_finite[int]()` = INT_MIN (e.g. `-9223372036854775808` for int64). When `simd_op[Divide]` computed `a / (b + epsilon)`, this caused integer overflow, making all int64 divide results zero. Fixed in `common_utils.mojo:224` — integer types now return 0, which preserves integer division semantics (`b + 0 == b`).

**Result:** 12 copies → 0 copies. Single source of truth in `shared/scalar_ops.mojo`. `test_buffers.mojo`: 298/298 passed.

---

### CPU-8 [H8] 8. Eliminate unnecessary Buffer copy in broadcast scalar path — 1.5-2× speedup ✅ FIXED

**File:** `tenmo/cpu_broadcast.mojo`, `apply_scalar` (L89-116)

**Problem:** `apply_scalar` called `copied(offset, offset+numels).arithmetic_ops_scalar[op_code](item)` — two allocations + one full memcpy + one SIMD pass per scalar broadcast.

**Fix:** `arithmetic_ops_scalar` already accepts `(item, start, end)`. Changed to `arithmetic_ops_scalar[op_code](item, offset, offset + numels)` directly, eliminating the intermediate `copied()` allocation and memcpy.

**Impact:** 1.5-2× speedup for every scalar broadcast operation (bias_add, scalar arithmetic, etc.). Ordering for ReverseSubtract/ReverseDivide is preserved — `arithmetic_ops_scalar` reads from the ND tensor's buffer as `self[i]` and applies the scalar, exactly as the old `.copied()` path did.

---

### CPU-9 [H9] 9. Parallelize CPU reduction suffix-axis fast path — 2-4× speedup ✅ DONE

**File:** `tenmo/sum_mean_reduction.mojo`, `reduce_cpu` (L114-128)

**Fix:** Replaced `for oi in range(num_out)` with `parallelize[reduce_row_fn](num_out, num_physical_cores())` using `@parameter def reduce_row_fn`. Each output element's `Buffer.sum()` is independent — 2-4× speedup for large reductions.

**Verified:** `./execute.sh summean` passes all tests.

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

### CPU-17 [L17] 17. `all_close` CPU fallback: missing SIMD path ✅ FIXED

**Files:** `tenmo/buffers.mojo` → `all_close` (L2046-2083)

**Problem:** `AllClose.launch()` only had a GPU path. CPU path fell through to slow decomposed operation.

**Fix:** `Buffer.all_close` already uses SIMD via `load[simdwidth]` → `le(tolerance).reduce_and()` with `@parameter` dispatch. Short-circuits on first mismatch, handles NaN correctly (NaN != NaN). No changes needed — the SIMD path was already in place.

---

### CPU-18 [L18] 18. Pooling/CNN CPU: parallelize outer loops — 2-4× speedup

**Files:** `tenmo/pooling.mojo`, `tenmo/cnn.mojo`

**Current:** Conv2d, MaxPool2d, AvgPool2d CPU forward/backward use raw `data_ptr()` loops with nested for loops over batch, input channels, output channels, spatial dimensions (e.g., `cnn.mojo` L375-378, L748-749, L844-846). All loops are serial.

**Fix:** Use `parallelize` over batch or output-channel dimensions. Each batch element or output channel is independent.

**Estimated impact:** 2-4× for CPU Conv2d/MaxPool2d at scale. Rarely trained on CPU at scale, but useful for small models and inference.

**Effort:** Medium.
