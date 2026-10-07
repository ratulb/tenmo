# `GatherKernel.gather_gpu` Double-Dispatches the 2D Row-Gather Fast Path

**Status:** fixed in `03ea4e8` and validated on GPU (2× T4, Mojo 1.1.0) — `dispatched_2d_fastpath` flag gates the generic rank dispatch; all 36 tests in `tests/test_gather_gpu.mojo` pass. Exact-one-launch assertion still open (needs a runtime launch-count hook).
**Severity:** performance only — results are correct on every input.
**File:** `tenmo/kernels/gather_kernel.mojo`
**Found:** 2026-10-04, while tracing the kernel dispatch for [device-resident data loading](device-resident-dataloading.md).

---

## Summary

`gather_gpu` selected a specialized 2D row-gather kernel and then **fell through** to the generic rank-dispatch kernel, enqueuing both into the same output buffer. The two kernels computed identical values, so the result was correct — but the "fast path" performed 2× the memory traffic it existed to avoid, on exactly the inputs it was written for. (Fixed in `03ea4e8`; this document keeps the pre-fix shape for context and records the as-shipped fix in [The Fix](#the-fix).)

The adjacent `comptime for r in range(1, MAX_RANK + 1)` block was written as if it were an unconditional fallback. It was not guarded by `else` and the fast path did not `return`.

---

## The Defect (pre-fix; fixed in `03ea4e8`)

Pre-fix `tenmo/kernels/gather_kernel.mojo:287-326` (current dispatch flag at `:287-289`, guarded generic dispatch at `:313`):

```mojo
        var out_dev = ctx.enqueue_create_buffer[datatype](total_output)

        if rank == 2 and axis == 0 and tensor_layout.shape[1] <= 512:
            var in_cols = tensor_layout.shape[1]
            var block_cols = _gather_2d_block_cols(in_cols)
            var compiled = ctx.compile_function[
                gather_rows_2d_kernel[datatype, Self.index_dtype],
            ]()
            ctx.enqueue_function(
                compiled,
                out_dev,
                in_dev,
                Int64(tensor_layout.shape[0]),
                Int64(in_cols),
                Int64(tensor_layout.strides[0]),
                idx_dev,
                Int64(n_indices),
                Int64(out_strides[0]),
                grid_dim=n_indices,
                block_dim=block_cols,
            )
        # Rank dispatch generated from MAX_RANK (shared.constants) — one
        # arm per rank, selected at runtime. Lowering MAX_RANK keeps the
        # instantiations (dead arms never fire; over-rank inputs panic
        # above); raising it generates new arms automatically.
        comptime for r in range(1, MAX_RANK + 1):
            if rank == r:
                _launch_gather_generic[datatype, r, Self.index_dtype](
                    ctx,
                    out_dev,
                    in_dev,
                    ...
                )
```

Line 287 opened an `if`. Line 305 closed the `enqueue_function` call. Line 306 started the rank dispatch with **no `else`, no `return`**. When `rank == 2`, the condition on line 287 was true *and* the comptime loop's `rank == r` arm matched `r == 2`, so both fired.

Contrast the embedding-bag branch immediately above, which gets this right — it `return`s inside the `if` (gather_kernel.mojo:227-276, `return (` at `:273`):

```mojo
            if sync:
                ctx.synchronize()
            var out_shape = Shape(in_cols)
            var result_state = DeviceState[Self.dtype].__init__[special=True](
                out_dev^, gpu
            )
            return (
                Layout(out_shape),
                result_state^,
            )
```

That branch is unreachable-from-the-row-gather branch (it requires `reduction.is_sum() or reduction.is_mean()`, the row-gather path is only reached with `reduction == NONE`), so **the row-gather branch was the only defective dispatch in the function.**

---

## Why the Result Was Still Correct

Both kernels wrote the same `out_dev`, over the same `n_indices × in_cols` region, with the same values. Neither read `out_dev`, so the second write was idempotent.

**Region.** For `rank == 2, axis == 0`, line 278-283 builds:

```mojo
        var out_shape_arr = IntArray.with_capacity(rank)
        for d in range(rank):
            out_shape_arr.append(n_indices if d == axis else tensor_layout.shape[d])
        var out_shape = Shape(out_shape_arr)
        var out_strides = Strides.default(out_shape)
        var total_output = out_shape.num_elements()

        var out_dev = ctx.enqueue_create_buffer[datatype](total_output)
```

so `out_shape == [n_indices, cols]` and `out_strides == [cols, 1]`. The 2D kernel's maximum write index is `(n_indices - 1) * cols + (cols - 1) == n_indices * cols - 1 == total_output - 1` — in bounds, and it covers the buffer exactly once.

**Values.** `gather_rows_2d_kernel` (gather_kernel.mojo:65-99; `def` at `:65`), one block per output row:

```mojo
    var src_row = indices_buffer[unsafe_offset=row]
    if src_row < 0:
        src_row += Scalar[index_dtype](in_rows)
    ...
    out_buffer[unsafe_offset=row * out_row_stride + c] = in_buffer[unsafe_offset=
        Int(src_row) * in_row_stride + c
    ]
```

`gather_gpu_kernel` via `_launch_gather_generic` (gather_kernel.mojo:19-64 kernel, launcher at `:149-177`), one thread per output element:

```mojo
        var src_coords = out_coords
        var idx_val = indices_buffer[unsafe_offset=out_coords[axis]]
        if idx_val < 0:
            idx_val += Scalar[index_dtype](in_shape[axis])
        src_coords.storage[axis] = Int(idx_val)

        var src_flat = in_strides.fma(src_coords, in_offset)
        var dst_flat = out_strides.fma(out_coords, 0)
        out_buffer[unsafe_offset=dst_flat] = in_buffer[unsafe_offset=src_flat]
```

With `axis == 0`: `out_coords[0]` is the destination row, `src_coords[0]` becomes `indices[out_coords[0]]`, and `dst_flat == out_coords[0] * cols + out_coords[1] == row * out_row_stride + c`. Identical mapping, including negative-index normalization.

**Cost.** For MNIST-shaped inputs the generic kernel additionally paid full grid-stride launch overhead (`elementwise_launch_config(total_output, simdwidth)`, gather_kernel.mojo:166) and recomputed `RankArray` coordinates per element, where the 2D kernel's threads read a single scalar each.

---

## Trigger Conditions

All four must hold:

| # | Condition | Source |
|---|---|---|
| 1 | Host has an accelerator | `has_accelerator()` — gather.mojo:461 |
| 2 | Source tensor `is_on_gpu()` | gather.mojo:462 |
| 3 | `rank == 2 and axis == 0 and cols <= 512` | gather_kernel.mojo:288 (flag at `:287-289`, guard at `:313`) |
| 4 | `reduction == NONE` (`Reduction(2)`) | `Reduction.is_none()` — shared/__init__.mojo:45 |

Condition 4 is implied by the call path. `Gather._gather_copy`'s fused fast path (gather.mojo:405-425) requires `_is_fast_path(reduction, ax, rank)`, which demands `is_sum()` or `is_mean()` (gather.mojo:345-352) — so it always lands in the early-returning embedding-bag branch. The general path (gather.mojo:461-487) passes `reduction=Reduction(2)` explicitly:

```mojo
                    var result = GatherKernel[
                        Self.dtype, Self.index_dtype
                    ].gather_gpu(
                        self.buffer.layout(),
                        self.buffer.device_state.value(),
                        ax,
                        normalized,
                        Reduction(2),
                        sync=sync and not has_followup,
                    )
```

`Reduction(2)` is `none` (shared/__init__.mojo:13-28 ctor, `:45`). So **every plain 2D row-gather on a GPU tensor with ≤ 512 columns double-dispatched (pre-fix).**

### Blast radius

- **Not hit:** MNIST (784 cols > 512). `examples/mnist_gpu.mojo`, `mnist_conv2d_gpu.mojo`, `mnist_conv_tt_gpu.mojo` are unaffected.
- **Hit:** any 2D GPU tensor with ≤ 512 columns gathered along axis 0 with no reduction — `Tensor.gather`, `embedding`, and the token-embedding path in the LLM examples. Narrow 2D data (sequence models, `examples/sort_sequence.mojo`, `examples/reverse_sequence.mojo`, IMDb) is squarely in range.
- **Also hit (pre-fix):** whatever device-resident loader work consumed row-gathers through `gather_gpu`. Resolved by construction: the landed `GatherKernel.gather_rows_2d_into` (gather_kernel.mojo:341) calls `gather_rows_2d_kernel` directly and bypasses the `gather_gpu` dispatch entirely, so the loader path cannot double-dispatch.

---

## Why No Existing Test Caught It (pre-fix analysis)

Two independent reasons, both verified at the time.

**1. The defect was invisible to value assertions.** Both kernels wrote the same values, so every correctness test passed. All of `tests/test_gather_gpu.mojo` — `test_gather_gpu_2d_axis0_irregular_copy` (:18), `test_mcpy_gpu_2d_multi_row_reversed` (:115), `test_mcpy_gpu_2d_duplicate_indices` (:132), `test_gather_gpu_2d_axis0_cols512_boundary` (:170), the fused sum/mean tests (:337-458), and parity tests (:459+) — asserted values and kept passing after the fix. **A regression test for this defect cannot be a correctness test.** (Test names/lines are post-fix positions; pre-fix the file had 2–3-col fixtures at those early lines.)

**2. Pre-fix, no test exercised the other side of the threshold.** Every 2D fixture in `test_gather_gpu.mojo` was 2-3 columns wide (`Tensor[dtype].d2([[1.0, 2.0, 3.0], ...])` and similar). There was no `cols > 512` 2D case anywhere in the file, so the generic rank-2 arm was **never tested at all** — the double dispatch meant the generic kernel silently substituted for the fast one in every existing test, and the fast path's exclusive behavior was unverified. **Closed with the fix:** `test_gather_gpu_2d_axis0_cols512_boundary` (:170), `test_gather_gpu_2d_axis0_cols513_above_boundary` (:191), `test_gather_gpu_2d_axis0_cols784_mnist_like` (:214), and `test_gather_rows_2d_into_matches_gather` (:233).

---

## The Fix

> **As shipped (`03ea4e8`) vs as proposed.** The proposal below extracts a
> `_use_row_2d_fast_path()` predicate with a named `_ROW_2D_MAX_COLS`
> constant and a local `dispatched` flag. What actually landed is the
> minimal variant: an inline `dispatched_2d_fastpath` flag
> (gather_kernel.mojo:287-289) with the literal `<= 512` predicate kept in
> place, and the generic dispatch guarded by
> `if rank == r and not dispatched_2d_fastpath` (`:313`). No predicate
> helper, no named constant, no launch counters. The rest of this section is
> the original proposal, kept for the launch-count work that is still open.

Make double dispatch structurally impossible. Extract the predicate so it is named, testable, and stated once:

```mojo
comptime _ROW_2D_MAX_COLS = 512


@always_inline
def _use_row_2d_fast_path(rank: Int, axis: Int, cols: Int) -> Bool:
    """True when `gather_rows_2d_kernel` handles this gather.

    Coalesced one-block-per-row access beats the generic element-wise
    kernel for narrow rows. The threshold is a block-width tradeoff
    (`_gather_2d_block_cols` caps at 512), not a correctness limit.
    """
    return rank == 2 and axis == 0 and cols <= _ROW_2D_MAX_COLS
```

then gate the rank dispatch on it having *not* fired:

```mojo
        var dispatched = False
        if _use_row_2d_fast_path(rank, axis, tensor_layout.shape[1]):
            var in_cols = tensor_layout.shape[1]
            var block_cols = _gather_2d_block_cols(in_cols)
            var compiled = ctx.compile_function[
                gather_rows_2d_kernel[datatype, Self.index_dtype],
            ]()
            ctx.enqueue_function(
                compiled,
                out_dev,
                in_dev,
                Int64(tensor_layout.shape[0]),
                Int64(in_cols),
                Int64(tensor_layout.strides[0]),
                idx_dev,
                Int64(n_indices),
                Int64(out_strides[0]),
                grid_dim=n_indices,
                block_dim=block_cols,
            )
            dispatched = True

        if not dispatched:
            # Rank dispatch generated from MAX_RANK (shared.constants) — one
            # arm per rank, selected at runtime. Lowering MAX_RANK keeps the
            # instantiations (dead arms never fire; over-rank inputs panic
            # above); raising it generates new arms automatically.
            comptime for r in range(1, MAX_RANK + 1):
                if rank == r:
                    _launch_gather_generic[datatype, r, Self.index_dtype](
                        ctx,
                        out_dev,
                        in_dev,
                        tensor_layout.shape.array(),
                        tensor_layout.strides.array(),
                        tensor_layout.offset,
                        idx_dev,
                        n_indices,
                        axis,
                        out_shape.array(),
                        out_strides.array(),
                        total_output,
                    )
```

`dispatched` is set and read on the host, in straight-line control flow — the compiler cannot sink the flag past the launch, so exactly one path runs.

An `elif` is *not* usable here: the rank dispatch is a `comptime for` statement, not an `if` arm, so it cannot be made a sibling branch syntactically. A `return` after the 2D launch would work but would duplicate the `if sync` and `DeviceState` construction tail that both paths share.

**Do not** put the counter described below inside `elementwise_launch_config` (tenmo/gpu/runtime.mojo:9-105). It has 64 call sites across every element-wise kernel, so that is a 64-site blast radius for a one-file defect.

---

## Test Strategy

Three layers, because layers 1 and 2 alone cannot see the regression.

### Layer 1 — Predicate unit test (CPU-only, no GPU required; NOT landed — helper does not exist)

Pins the dispatch *condition*, including the threshold boundary, in `tests/test_gather.mojo` or `tests/test_gather_gpu.mojo`. Requires extracting `_use_row_2d_fast_path` first (see [The Fix](#the-fix) note):

```mojo
def test_gather_row2d_fast_path_predicate() raises:
    from tenmo.kernels.gather_kernel import _use_row_2d_fast_path
    # Fast path: rank 2, axis 0, narrow.
    assert_true(_use_row_2d_fast_path(2, 0, 1))
    assert_true(_use_row_2d_fast_path(2, 0, 512))
    # Generic: wrong rank, wrong axis, or too wide.
    assert_false(_use_row_2d_fast_path(2, 1, 128))
    assert_false(_use_row_2d_fast_path(3, 0, 128))
    assert_false(_use_row_2d_fast_path(2, 0, 513))
```

The `512` / `513` pair is the boundary that matters — the bare literal `512` at gather_kernel.mojo:288 is still unnamed (no `_ROW_2D_MAX_COLS` constant landed) and has no predicate unit test, though value coverage on both sides now exists (Layer 2 tests above).

### Layer 2 — Launch-count regression test (GPU; the only layer that catches the bug; NOT landed)

Module-scope counters in `gather_kernel.mojo` — one file, host-side, one increment per kernel launch, negligible next to the launch itself. (Not implemented — no such counters exist in the file today.)

```mojo
var _row2d_launches: Int = 0
var _generic_launches: Int = 0


def _launch_counters_reset():
    _row2d_launches = 0
    _generic_launches = 0
```

incremented at the 2D launch site and at the top of `_launch_gather_generic`. Then, in `tests/test_gather_gpu.mojo`:

```mojo
def test_gather_gpu_row2d_dispatches_exactly_one_kernel() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        # 384 cols <= 512 → fast path must win and the generic arm must not fire.
        var a = Tensor[dtype].from_list(
            List[Float32](
                [Float32(i % 97) for i in range(64 * 384)]
            )
        ).reshape(Shape(64, 384))
        var gpu = GPU()
        var a_gpu = a.to_gpu(gpu)
        var idx = IntArray()
        for i in range(8):
            idx.append(Int(63 - i))

        _launch_counters_reset()
        var result = a_gpu.gather(idx, axis=0)
        assert_equal(_row2d_launches, 1)
        assert_equal(_generic_launches, 0)
        _ = result
```

And the mirror case at 768 cols, asserting `_generic_launches == 1` and `_row2d_launches == 0` — which also closes the coverage gap from [Why No Existing Test Caught It](#why-no-existing-test-caught-it).

### Layer 3 — Recorded benchmark (manual gate, not a CI assertion)

`tests/benchmark.mojo` is CPU-only (no `has_accelerator`, no `to_gpu`), so there is no GPU benchmark harness to extend. Adding one is out of scope for a bug fix.

Instead: time `Tensor.gather` on a `(4096, 384)` GPU tensor, 200 iterations, before and after the fix, and record both numbers in this document. Timing thresholds are too flaky to assert in CI; the launch counters in Layer 2 are the enforceable regression guard, and Layer 3 only quantifies the win.

---

## Checklist

- [ ] Extract `_use_row_2d_fast_path` with the named `_ROW_2D_MAX_COLS` constant (shipped instead as inline `dispatched_2d_fastpath` + literal `512` in `03ea4e8`)
- [x] Gate the rank dispatch on `not dispatched` (landed as `not dispatched_2d_fastpath`, gather_kernel.mojo:313)
- [ ] Add launch counters + reset/read helpers to `gather_kernel.mojo`
- [ ] Layer 1 predicate test (CPU; blocked on predicate extraction)
- [ ] Layer 2 launch-count test, both sides of the 512 threshold (GPU)
- [x] Add a `cols > 512` 2D GPU value test (landed: cols512 `:170`, cols513 `:191`, cols784 `:214`, plus `gather_rows_2d_into_matches_gather` `:233`)
- [ ] Record before/after timings in this document
- [ ] Confirm no behavior change in `tests/test_gather.mojo` and `tests/test_embedding.mojo`
- [x] Re-check that `GatherKernel.gather_into` (device-resident loader work) inherits the corrected dispatch — vacuous: `gather_rows_2d_into` (`:341`) calls the 2D kernel directly, bypassing `gather_gpu` dispatch

---

## References (line numbers refreshed 2026-10-07; `gather_kernel.mojo` is 397 lines)

- `tenmo/kernels/gather_kernel.mojo:287-289,313` — the defect site (pre-fix `:287-326`); flag + guard as shipped
- `tenmo/kernels/gather_kernel.mojo:65-99` — `gather_rows_2d_kernel`
- `tenmo/kernels/gather_kernel.mojo:19-64` — `gather_gpu_kernel`
- `tenmo/kernels/gather_kernel.mojo:149-177` — `_launch_gather_generic` (`elementwise_launch_config` at `:166`)
- `tenmo/kernels/gather_kernel.mojo:193-340` — `gather_gpu` entry, per-call allocations
- `tenmo/kernels/gather_kernel.mojo:227-276` — the embedding-bag branch that gets the early return right (`return (` at `:273`)
- `tenmo/kernels/gather_kernel.mojo:341-397` — `gather_rows_2d_into` (bypasses `gather_gpu` dispatch)
- `tenmo/gather.mojo:5` — the fast path this comment advertises
- `tenmo/gather.mojo:345-352` — `_is_fast_path`
- `tenmo/gather.mojo:461-487` — the general path that passes `Reduction(2)`
- `tenmo/shared/__init__.mojo:13-28`, `45` — `Reduction(2) == none`
- `tenmo/gpu/runtime.mojo:1-105` — `elementwise_launch_config` (deliberately *not* instrumented)
- `tests/test_gather_gpu.mojo` — 36 tests; boundary coverage at `:170` (512), `:191` (513), `:214` (784), `:233` (into-vs-gather)