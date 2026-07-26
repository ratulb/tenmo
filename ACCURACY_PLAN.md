# Accuracy Enhancement Plan

## Goal
Add `token_accuracy` and `sequence_accuracy` methods to `Accuracy` struct, replacing
the ad-hoc `compute_accuracy` functions in `examples/sort_sequence.mojo` and
`examples/reverse_sequence.mojo`.

## Design Decisions

### 1. Device check: `and` (both on same device), panic on mixed
`Accuracy.compute` uses `or` (transfers CPU tensor to GPU if one is on GPU). The new
methods follow the codebase convention from `NDBuffer.arithmetic_ops`:
- If both on GPU → use GPU path
- If both on CPU → use CPU path
- If mixed → `panic()` with message
This avoids silent cross-device transfers during evaluation loops.

### 2. Copy mechanism: `Tensor(copy=orig)`
Mojo's copy constructor (`__init__(out self, *, copy: Self)`) creates a deep copy with
independent buffer. Device residency is preserved (GPU copy stays on GPU).
Usage: `var flat = Tensor[DTYPE](copy=pred)` then `flat.reshape(...)`.

### 3. GPU block size: 256 threads/block (not raw batch_size)
`AccuracyGpu` uses `block_dim=batch_size` which hits the 1024 hardware ceiling.
Following the dominant pattern (used by Compare, ScalarOperations, BinaryOperations):
- `block_dim = 256` (fixed)
- `grid_dim = ceil(batch_size / 256)`
- Kernel uses grid-stride loop with bounds check — safe for any grid size

### 4. Kernel indexing: direct arithmetic (no Int() cast)
Modern Mojo handles `thread_idx.x + block_dim.x * block_idx.x` without explicit `Int()`
wrapping. Only `Int()` needed for comparison: `Int(labels[tid * seq_len + t])`.

### 5. No NaN handling, ties to first argmax found
Matches existing `AccuracyGpu` and `argmax` conventions.

---

## Files to Modify (6)

### 1. `tenmo/accuracy.mojo` — 4 new methods

**`token_accuracy(pred, target, sync) → Float64`**
```
var num_classes = pred.shape()[pred.shape().ndim() - 1]
var flat_pred = Tensor[Self.dtype](copy=pred)
flat_pred.reshape(target.numels(), num_classes)
var flat_target = Tensor[Self.index_dtype](copy=target)
flat_target.reshape(target.numels())
var correct = Self.compute(flat_pred, flat_target, sync)
return Float64(correct) / Float64(target.numels())
```
Device check: panic if mixed. Copy+flatten preserves device.

**`sequence_accuracy(pred, target, sync) → Float64`**
```
if both GPU: _sequence_accuracy_gpu
if both CPU: _sequence_accuracy_cpu
if mixed: panic
return Float64(correct) / Float64(pred.shape()[0])
```

**`_sequence_accuracy_cpu(pred, target) → Int`**
```
var preds = pred.argmax[Self.index_dtype](axis=-1)  # (B, T)
var B = preds.shape()[0], T = preds.shape()[1]
var correct = 0
for b in range(B):
    var ok = True
    for t in range(T):
        if Int(preds[b, t]) != Int(target[b, t]):
            ok = False; break
    if ok: correct += 1
return correct
```

**`_sequence_accuracy_gpu(pred, target, sync) → Int`**
```
from tenmo.kernels import SequenceAccuracyGpu
return SequenceAccuracyGpu[Self.dtype, Self.index_dtype].launch(
    pred.buffer, target.buffer, sync=sync
)
```

### 2. `tenmo/kernels/accuracy_kernel.mojo` — Add kernel + struct

**`sequence_accuracy_kernel`** function:
```mojo
def sequence_accuracy_kernel[
    dtype: DType,
    index_dtype: DType = DEFAULT_INDEX_DTYPE,
](
    result: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    pred: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    labels: UnsafePointer[Scalar[index_dtype], ImmutAnyOrigin],
    batch_size: Int,
    seq_len: Int,
    num_classes: Int,
):
    var tid = thread_idx.x + block_dim.x * block_idx.x
    if tid >= batch_size: return
    if tid == 0: result.store(0)
    barrier()
    var seq_base = tid * seq_len * num_classes
    for t in range(seq_len):
        var pos_base = seq_base + t * num_classes
        var max_val = pred[pos_base]
        var max_idx = 0
        for j in range(1, num_classes):
            var val = pred[pos_base + j]
            if val > max_val:
                max_val = val
                max_idx = j
        if max_idx != Int(labels[tid * seq_len + t]):
            return
    _ = Atomic.fetch_add[ordering=Ordering.RELAXED](result, 1)
```

**`SequenceAccuracyGpu`** struct (launch wrapper):
- Uses `pred.shape[0]` for batch_size, `pred.shape[1]` for seq_len, `pred.shape[2]` for num_classes
- block_dim = 256, grid_dim = ceil(batch_size / 256)
- Matches `AccuracyGpu` boilerplate (device_state, compile_function, enqueue, map_to_host)

### 3. `tenmo/kernels/__init__.mojo` — Add export
```
from .accuracy_kernel import AccuracyGpu, SequenceAccuracyGpu
```

### 4. `tests/test_accuracy.mojo` — Add tests

**CPU tests** (7 new functions, no guard needed):
| Test | pred shape | setup | expected |
|---|---|---|---|
| `test_token_accuracy_perfect` | (2,2,3) | all correct | 1.0 |
| `test_token_accuracy_half` | (2,2,3) | half correct | 0.5 |
| `test_token_accuracy_all_wrong` | (2,2,3) | none correct | 0.0 |
| `test_seq_accuracy_perfect` | (2,3,4) | both full seq correct | 1.0 |
| `test_seq_accuracy_half` | (2,3,4) | 1 of 2 seq correct | 0.5 |
| `test_seq_accuracy_none` | (2,3,4) | 0 of 2 seq correct | 0.0 |
| `test_token_vs_seq_mismatch` | (2,4,5) | 80% tok, 0% seq | tok=0.8, seq=0.0 |

**GPU tests** (3 new functions, under `comptime if has_accelerator()`):
| Test | what it checks |
|---|---|
| `test_gpu_token_accuracy` | GPU tok_acc matches expected |
| `test_gpu_seq_accuracy` | GPU seq_acc matches expected |
| `test_gpu_token_seq_parity` | GPU tok_acc == CPU tok_acc on same data |

Test pattern follows existing file: `def test_*()` with `assert_equal()` from
`std.testing`. GPU tests use `gpu_parity`-style helper or `comptime if` guard.

### 5. `examples/sort_sequence.mojo` — Replace local `compute_accuracy`
```mojo
def compute_accuracy(
    logits: Tensor[DTYPE], targets: Tensor[IDTYPE]
) -> Tuple[Float64, Float64]:
    from tenmo.accuracy import Accuracy
    return (
        Accuracy[DTYPE].token_accuracy(logits, targets),
        Accuracy[DTYPE].sequence_accuracy(logits, targets),
    )
```

### 6. `examples/reverse_sequence.mojo` — Same replacement

---

## What Won't Change
- `Accuracy.compute` remains unchanged (and its `or` dispatch pattern is preserved)
- Existing test_accuracy.mojo tests for `compute` stay as-is
- Training loop structure in sort_sequence.mojo stays identical
- No new files created

---

## Assumptions (to document in AGENTS.md)

1. **Contiguous input for GPU**: `SequenceAccuracyGpu` kernel assumes contiguous
   (B, T, V) layout — same assumption as `AccuracyGpu` for (N, C). For strided input,
   caller should `.contiguous()` first.
2. **`token_accuracy` copies + flattens**: Always produces contiguous input to
   `compute` — safe regardless of original layout.
3. **Same-device requirement**: Mixed CPU/GPU raises panic (matching arithmetic_ops).
4. **Ties**: First argmax index wins (matching existing convention).
5. **No NaN handling**: Undefined results if pred contains NaN.
6. **Block size 256**: Matches `elementwise_launch_config` convention used by 5+
   kernel structs. Grid-stride kernel is correct for any grid size.
