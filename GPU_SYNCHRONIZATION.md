# GPU Synchronization Architecture

## 1. Architecture Overview

GPU operations in Tenmo flow through a 4-layer stack:

```
Tensor → NDBuffer → DeviceState (wraps DeviceBuffer) → Mojo GPU API
```

### Key Types

**`GPU`** (`device.mojo:72`) — wraps Mojo's `DeviceContext` + a unique ID.
`GPU.__getitem__()` returns `self.device_context.copy()` — every kernel call
site gets a fresh handle to the same default stream.

**`DeviceState[dtype]`** (`device.mojo:121`) — wraps
`DeviceBuffer[datatype]` + owning `GPU`. Handles allocate, fill, copy-to-host,
and device-to-device transfer. `DType.bool` stored as `uint8`.

**`NDBuffer`** (`ndbuffer.mojo:66`) — single source of truth: shape, strides,
offset, CPU `Buffer`, optional `DeviceState`. All device dispatch happens here.

**`DeviceContext`** (Mojo std lib `std.gpu.host`) — opaque handle. Provides
`enqueue_function`, `enqueue_create_buffer`, `enqueue_fill`,
`enqueue_copy_to`, `compile_function`, `synchronize`. Single default stream
— kernels execute in submission order.

### Dispatch Flow

```mojo
tensor.sum() → NDBuffer.reduce[SUM]()
  → comptime has_accelerator() && is_on_gpu()
    → Reduction[dtype].launch[op_code](self, axes, keepdims, sync)
      → device_context.enqueue_function(compiled_func, args...)
      → if sync: device_context.synchronize()
```

---

## 2. Kernel Launch Pattern

Every GPU kernel launcher follows a standard 4-step pattern:

```mojo
# Step 1: Get DeviceContext from GPU handle
ref gpu = device_state.get_gpu()
var ctx = gpu[]

# Step 2: Allocate output buffer
var result = ctx.enqueue_create_buffer[dtype](num_elements)

# Step 3: Compile and enqueue kernel
var compiled = ctx.compile_function[kernel_fn, kernel_fn]()
ctx.enqueue_function(compiled, args..., grid_dim=NB, block_dim=TB)

# Step 4: Optionally synchronize
if sync: ctx.synchronize()
```

All 42 kernel launchers previously defaulted `sync: Bool = True`. Phase 1
changed all defaults to `sync: Bool = False`. The body is uniformly
`if sync: device_context.synchronize()` — callers pass `sync` explicitly
when coordination is needed.

---

## 3. Sync Policy

### 3a. The Core Tension

Eager sync after every op (current default):
```
result = matmul(A, B)  → kernel launch → SYNC → result ready
result = relu(result)  → kernel launch → SYNC → result ready
N ops = N round trips  → safe, simple, slow
```

No sync:
```
result = matmul(A, B)  → kernel launch queued
result = relu(result)  → kernel launch queued
Fast. But result.item() silently reads stale data.
```

### 3b. How PyTorch Solves It — Lazy Sync

Sync only when CPU genuinely needs the data:
```python
c = torch.matmul(a, b)    # queued
d = torch.relu(c)         # queued
e = torch.sum(d)          # queued
val = e.item()            # ← sync happens HERE
```

PyTorch sync points: `.item()`, `.numpy()`, `.cpu()`, `print()`,
`loss.backward()`. Everything else stays async on GPU.

### 3c. Tenmo Sync Policy

**ALWAYS sync** (CPU reads GPU data):
- `.item()` — scalar extraction crosses GPU→CPU
- `.to_cpu()` / `.numpy()` / `to_ndarray()` — explicit device transfer
- `print()` — materializes CPU copy via `.to_cpu()` then prints
- `assert` on GPU tensor — needs actual value
- `AllClose` / equality checks on GPU — result crosses to CPU

**SYNC if crossing device boundary:**
- Mixed-device ops (CPU index → GPU tensor gather)
- CPU-side data fed to GPU kernel parameters (indices, scalars)

**NEVER sync:**
- Pure GPU→GPU ops (matmul, relu, add, etc.)
- Backward pass — gradient accumulation stays on GPU
- `.gradients()` — raw ref to Gradbox, used by SGD internally

**USER-CONTROLLED:**
- `device_context.synchronize()` — explicit flush

### 3d. `grad()` vs `gradients()`

**`tensor.grad()`** — For debugging and visualization only.
- Calls `Gradbox.detach()` which deep-copies gradient data
- Returns a CPU-side Tensor (syncs internally via `map_to_host`)
- Always safe for display, `.copy()`, `.item()`
- **Not used in training loops** — too expensive (deep copy + sync per call)

**`tensor.gradients()`** — For actual gradient updates in training.
- Returns a raw `ref Gradbox` — no copy, no sync
- Stays entirely on GPU
- `SGD.step()` accesses this internally without CPU round-trip
- If CPU needs the values (e.g., gradient clipping stats), caller must sync
  explicitly

> **Rule:** Use `gradients()` in hot loops. Use `grad()` only for debugging.

---

## 4. Sync Call Sites

### 4a. Kernel Launchers — 42 functions across 17 files

All default `sync=False` (Phase 1). Every function calls `synchronize()` when
`sync` is `True` — callers pass `sync=True` explicitly.

| File | Functions | Lines |
|---|---|---|
| `reduction_kernel.mojo` | `Reduction.launch`, `launch_product`, `compute_excl_product`, `launch_log_sum`, `launch_welford` | 962, 1043, 1198, 1263, 1358 |
| `binary_ops_kernel.mojo` | `BinaryOperations.launch` | 476 |
| `binary_inplace_ops_kernel.mojo` | `BinaryInplaceOperations.launch` | 386 |
| `scalar_ops_kernel.mojo` | `ScalarOperations.launch`, `launch_pow` | 193, 246 |
| `scalar_inplace_ops_kernel.mojo` | `InplaceScalarOperations.launch` | 92 |
| `unary_ops_kernel.mojo` | `UnaryOpsKernel.launch`, `launch_with_mask` | 307, 406 |
| `division_kernel.mojo` | `DivisionKernel.launch_rdiv_scalar_backward`, `launch_divide_backward` | 114, 180 |
| `compare_kernel.mojo` | `AllClose.launch`, `Compare.launch`, `CompareScalar.launch` | 150, 313, 458 |
| `filler_kernel.mojo` | `_fill_scalar_gpu`, `_fill_buffer_gpu`, `_scatter_add_gpu`, `_scatter_add_nd_gpu` | 128, 167, 238, 347 |
| `gather_kernel.mojo` | `GatherGpu.gather_gpu` | 182 |
| `shuffle_kernel.mojo` | `ShuffleGpu.launch_gather`, `launch_scatter` | 85, 136 |
| `dropout_kernel.mojo` | `DropoutKernel.launch` | 126 |
| `minmax_kernel.mojo` | `MinMaxGpu.launch` | 227 |
| `argminmax_kernel.mojo` | `ArgMinMaxGpu._gpu_reduce` | 108 |
| `matmul_kernel.mojo` | `MatmulNdGpu.launch` | 128 |
| `vectormatrix_kernel.mojo` | `VectorMatmulNdGpu.launch` | 85 |
| `matrixvector_kernel.mojo` | `MatrixVectorNdGpu.launch` | 85 |
| `dotproduct_kernel.mojo` | `DotproductKernel.launch` | 115 |
| `layernorm_kernel.mojo` | `LayerNormGpu.launch` | 116 |
| `std_variance_backward_kernel.mojo` | `launch_variance_backward`, `launch_std_backward` | 296, 371 |
| `bce_kernel.mojo` | `launch_forward_with_logits`, `launch_forward_with_logits_reduce`, `launch_forward`, `launch_forward_reduce`, `launch_bce_with_logits_backward`, `launch_bce_with_logits_backward_scaled`, `launch_bce_backward`, `launch_bce_backward_scaled` | 511, 585, 671, 744, 828, 894, 959, 1025 |
| `ndbuffer.mojo` onehot | 411 | **No GPU kernel** — CPU-fallback per-element DeviceState access. Fused into crossentropy kernel. |

### 4b. DeviceState Methods — 4 default `sync=False`, `into()` no sync param

| Method | Line | What it does |
|---|---|---|
| `DeviceState.new(size, value, sync=False)` | 189 | Allocates + fills GPU buffer. `enqueue_fill` + optional sync |
| `DeviceState.fill(value, sync=False)` | 229 | Scalar fill via `map_to_host` + `enqueue_fill` + sync |
| `DeviceState.fill(source: NDBuffer, sync=False)` | 243 | CPU→GPU copy. Reads CPU logical view → writes GPU contiguously |
| `DeviceState.into(shape)` | 306 | GPU→CPU copy. `map_to_host` → memcpy. No sync param (Phase 0 removed redundant `if sync: sync()`) |
| `DeviceState.map_where(ndb, pred, value, sync=False)` | 211 | Scalar predicate fill via `map_to_host` |

### 4c. NDBuffer GPU Dispatch Methods — 12 functions

These are the public API boundary for GPU ops. Each calls one kernel launcher
and passes `sync` through:

| NDBuffer method | Line | Kernel launcher |
|---|---|---|
| `inplace_ops[op_code]` | 1720 | `BinaryInplaceOperations.launch` |
| `inplace_scalar_ops[op_code]` | 1828 | `InplaceScalarOperations.launch` |
| `arithmetic_ops[op_code]` | 1968 | `BinaryOperations.launch` |
| `scalar_ops[POW]` | 2129 | `ScalarOperations.launch_pow` |
| `scalar_ops[op_code]` | 2133 | `ScalarOperations.launch` |
| `unary_ops[op_code]` | 2194 | `UnaryOpsKernel.launch` |
| `float_unary_ops[op_code]` | 2251 | `UnaryOpsKernel.launch` |
| `compare[op_code]` | 2371 | `Compare.launch` |
| `compare_scalar[op_code]` | 2444 | `CompareScalar.launch` |
| `all_close` | 2508 | `AllClose.launch` |
| `matmul_2d` | 2641 | `MatmulNdGpu.launch` |
| `matmul_nd` | 2699 | `MatmulNdGpu.launch` |

### 4d. GPU Context Manager

```mojo
def __exit__(mut self):
    self.device_context.synchronize()   # device.mojo:112
```

Rarely exercised — most code accesses GPU through `DeviceState.gpu` directly,
not through `with GPU() as gpu:` blocks.

---

## 5. Compound Operations (Multi-Kernel Sequences)

These functions launch multiple GPU kernels in sequence within a single
logical operation — prime candidates for batching sync.

### 5a. Welford.forward_gpu (`welford.mojo:53-66`)

```mojo
def forward_gpu(..., sync: Bool = True) raises:
    var (mean_ndb, M2_ndb) = Reduction.launch_welford(ndb, axes, keepdims, sync=False)
    var var_ndb = M2_ndb.scalar_ops[Divide](divisor, sync=False)
    if sync and var_ndb.is_on_gpu():
        var_ndb.sync()
    return (mean_ndb^, var_ndb^)
```

Welford exposes `sync` to its callers. Internally, both kernels (welford +
divide) run without sync; one sync at the end when `sync=True`.

### 5b. StdDev.forward (`std_deviation.mojo:147-156`)

```mojo
Welford.forward(..., sync=False)
var_ndb.unary_ops[SQRT](sync=False)
ndb.sync()
```

2 kernels (Welford + SQRT), 1 sync.

### 5c. LayerNorm.forward (`layernorm.mojo:261-278`)

```mojo
Welford.forward(..., sync=False)
LayerNormCpu.normalize(..., sync=False)  # calls LayerNormKernel.launch
ndb.sync()
```

2 kernels (Welford + LayerNormKernel), 1 sync.

### 5d. Gather with reduction (`gather.mojo:428-468`)

Two paths:
- **Fast path (rank==2, axis=0, mean):** `gather_gpu` + `result /= scalar`
  (scalar_ops) → 2 kernels, 2 syncs (not yet batched — scalar_ops path
  is trivially fast).
- **General path:** `gather_gpu` (sync=False) + `.sum()/.mean()` (sync) → 2
  kernels, 1 sync. Batched: gather's internal sync suppressed since sum/mean
  handles the synchronize.

### 5e. CrossEntropyLoss Forward — Pre-Fusion (~18 kernels → 1)

**Pre-fusion state:** `CEClassIndicesForward.forward` decomposes into ~18
kernel launches + 1 CPU-fallback onehot loop for a (64, 10) GPU tensor:

| Decomposition step | GPU ops | Source |
|---|---|---|
| `_softmax_components` (max stability) | minmax + fill(0) + mask + sub + exp | `softmax.mojo:119` |
| log-sum-exp + softmax | `log_sum` + sum + sub + div | `softmax.mojo:95` |
| ignore_mask | `compare_scalar` + `to_dtype` | `crossentropy.mojo:246` |
| valid_count | `sum` + **`.item()` sync** | `crossentropy.mojo:525` |
| onehot | **CPU fallback** — per-element DeviceState round trips | `ndbuffer.mojo:411` |
| NLL | `onehot * log_probs` + sum + negate | `crossentropy.mojo:546` |
| zero_ignored | `losses * ignore_mask` | `crossentropy.mojo:549` |
| reduction (mean) | sum + scalar divide | `crossentropy.mojo:274` |

**Root cause:** 29ms measured for (64, 10) on T4 — dominated by kernel launch
overhead (~5µs each × ~18 = ~90µs dispatch) + CPU-fallback onehot (64
per-element DeviceState round trips) + avoidable syncs.

**Fusion plan:** Single kernel replaces all ~18 launches + CPU onehot:
- 1 block per row (M blocks), shared-memory tree reduction for max and sum_exp
- `Atomic.fetch_add` for scalar_loss and valid_count (avoids `.item()` sync)
- Label smoothing and ignore_index handled in-kernel
- Outputs: softmax (M,C) + per_sample_loss (M,) + scalar_loss (1,) + valid_count (1,)

**Est. sync reduction:** 2 per forward call (valid_count `.item()` + final
`.sync()`) → 1 (mean divisor readback, batched with final sync).

The backward pass stays **unchanged** — reads stored `softmax_probs` from
payload, does 4 GPU arithmetic ops. No fusion needed.

---

## 6. Backward Pass Sync (`tensor.mojo:2870-2952`)

`backward()` dispatches backward kernels through the graph in topological
order. Each backward handler calls `parent.update_grad()` which dispatches
GPU ops (add, copy, etc.) to accumulate gradients.

**No sync at end of backward, by design.** After backward completes:
- All gradients have been queued as GPU operations on the default stream
- The CPU has NOT waited for them to finish — and should not
- The backward pass is pure GPU work; sync would be wasted
- If the caller needs gradients on CPU (e.g., for monitoring), they call
  `.grad()` which syncs internally, or `.gradients().to_cpu()` explicitly
- `SGD.step()` reads gradients via `.gradients()` → raw GPU ref → no sync
  needed for the step itself

---

## 7. CPU-GPU Data Transfer Sync

`map_to_host` is the implicit sync primitive. Mojo guarantees GPU work
completes before the mapped memory is coherent on the CPU side. Every GPU→CPU
transfer eventually calls `map_to_host`. The `synchronize()` after
`map_to_host` in `DeviceState.into()` was redundant and has been removed
(Phase 0).

| Operation | CPU reads GPU? | Sync mechanism |
|---|---|---|
| `.backward()` return | **No** (by design) | Pure GPU — no sync needed |
| `.grad()` | Yes | `Gradbox.detach()` → `contiguous_device_state()` → `map_to_host` (implicit sync) |
| `.gradients()` | **No** | Raw ref to Gradbox — stays on GPU |
| `.item()` | Yes | `DeviceState.__getitem__` → `map_to_host` |
| `.to_cpu()` | Yes | `DeviceState.into` → `map_to_host` + `memcpy` |
| `.numpy()` | Yes | Calls `.to_cpu()` first |
| `__str__()` / `print()` | Yes | `__str__()` is metadata-only (no GPU data). `print()` calls `.to_cpu()` then formats on CPU (Phase 3) |
| `to_ndarray()` | Yes | Calls `.to_cpu()` first (Phase 3) |
| `SGD.step()` | Yes (write-back) | `map_to_host` read-modify-write + explicit `sync()` after write |
| `AllClose` / compare | Yes | Result buffer via `map_to_host` |
| `ndb[idx]` element read | Yes | `DeviceState.__getitem__` → `map_to_host` |

---

## 8. Correctness Model

### 8a. Same-Stream Ordering

All GPU ops go through `GPU.__getitem__()` which returns
`self.device_context.copy()`. Since every `GPU` owns exactly one
`DeviceContext`, and `DeviceContext` operations are serialized on a single
default stream:

**Kernel A enqueued → Kernel B enqueued → Kernel B always starts after
Kernel A finishes**

Removing `synchronize()` between A and B is **safe** — the GPU pipeline is
preserved. The sync only blocks the CPU.

### 8b. When Sync IS Required

A sync is required when and only when the **CPU reads data produced by a GPU
kernel**. In Tenmo, `map_to_host` is the implicit sync primitive that
guarantees coherence at GPU→CPU boundaries:

```
GPU: Kernel produces result R
CPU: Reads R  ← map_to_host triggers implicit sync before this point
```

**Sync always happens at:**
1. `.to_cpu()` / `.numpy()` / `to_ndarray()` — `DeviceState.into()` → `map_to_host` + `memcpy`
2. `.item()` — `DeviceState.__getitem__` → `map_to_host`
3. `AllClose` / compare — result buffer via `map_to_host`
4. `print()` — routes through `.to_cpu()` (Phase 3)
5. `SGD.step()` write-back — `map_to_host` + explicit `sync()` after write

**Sync never happens at:**
- Pure GPU→GPU ops — same-stream ordering guarantees correctness
- `.backward()` — gradients stay on GPU
- `.gradients()` — raw GPU ref, no copy

### 8c. Data Race Scenarios

| Scenario | Risk | Why safe |
|---|---|---|
| Op A → Op B (both GPU, B reads A's output) | None | Same-stream ordering |
| Fill GPU buffer → Kernel reads it | None | Fill uses `enqueue_fill` on same stream |
| Kernel writes → `map_to_host` reads | Low | `map_to_host` is an implicit sync |
| Kernel writes → CPU reads via `into()` | None | `map_to_host` is an implicit sync (Phase 0 removed redundant `sync()`) |
| Kernel A → CPU reads via `item()` | None | `map_to_host` is an implicit sync |

---

## 9. Optimization Plan

### Phase 0 — Remove redundant sync in `DeviceState.into()` ✅

`DeviceState.into()` (device.mojo:306) called `map_to_host` which is already an
implicit sync, then called `synchronize()` again redundantly. Removed the extra
sync and the now-dead `sync` parameter entirely. Saves 1 sync per `.to_cpu()`,
`.numpy()`, or explicit `into()` call.

### Phase 1 — Change all kernel launcher defaults (`sync=True` → `sync=False`) ✅

46 signature changes across 22 files. Pure mechanical — no behavior change
because NDBuffer callers pass `sync` explicitly.

**Files:**
- 16 kernel files (42 launchers) listed in §4a
- `device.mojo` (5 DeviceState methods) listed in §4b

### Phase 2 — Remove redundant syncs from compound operations ✅

- [x] `welford.mojo` — added `sync: Bool = True` param, batched internal sync
- [x] `std_deviation.mojo` — passes `sync=False` to Welford + SQRT, syncs once
- [x] `layernorm.mojo` — passes `sync=False` to Welford + LayerNormKernel, syncs once
- [x] `gather.mojo:428-468` — pass `sync=False` to gather_gpu in general GPU path when follow-up sum/mean handles sync

### Phase 3 — Route `print()`, `to_ndarray()` through `.to_cpu()` ✅

**Current problems:**

- **`Tensor.print()`** (`tensor.mojo:2800`): Calls `print_buffer` which reads
  elements one-by-one via `NDBuffer.__getitem__[]` → `DeviceState.__getitem__`
  → **per-element `map_to_host()`**. An N-element GPU tensor triggers N syncs.
- **`to_ndarray()`** (`numpy_interop.mojo:59`): For contiguous GPU tensors,
  reads `ndb.data_ptr()` which returns the **empty CPU buffer** — a bug that
  copies garbage. For non-contiguous GPU tensors, falls through to per-element
  `__getitem__` → N syncs.

**Changes made:**

1. **`Tensor.print()`**: Before calling `print_buffer`, checks `is_on_gpu()`. If
   on GPU, calls `self.buffer.to_cpu()` directly (avoids autograd overhead of
   `Tensor.to_cpu()`), then prints from the CPU copy. Replaces N per-element
   `map_to_host()` calls with 1 bulk transfer via `DeviceState.into()`.

2. **`to_ndarray()`**: When the NDBuffer is on GPU, calls `.to_cpu()` first
   before reading `data_ptr()`. This fixes the contiguous-GPU bug (was reading
   empty CPU buffer) and replaces per-element access with bulk transfer for
   non-contiguous GPU.

3. **`Tensor.item()`** and **`DeviceState.__getitem__()`**: Leave as-is.
   Single-element reads through `map_to_host()` are the most efficient path —
   no bulk transfer overhead.

**Keep `map_to_host()` call sites (not redundant):**

| Location | Reason |
|---|---|
| `DeviceState.fill(Scalar)` | `enqueue_fill` is async GPU op; sync ensures completion |
| `DeviceState.fill(NDBuffer)` — CPU→GPU path | CPU writes to mapped memory; sync ensures GPU visibility |
| `DeviceState.__setitem__` | CPU writes to mapped memory; sync ensures GPU visibility |
| `DeviceState.__getitem__` | Single scalar read — most efficient path |
| `DeviceState.load` / `store` | SIMD bulk access on full buffer |
| `DeviceState.all_true` / `any_true` | Scalar result from GPU; needs sync for correctness |
| `SGD.step()` | CPU-side read-modify-write of params/grads; sync ensures coherence |
| `gather_kernel.mojo` idx upload | CPU writes indices to GPU; sync ensures kernel reads correct data |
| `filler_kernel.mojo` idx upload | Same — CPU writes indices to GPU |
| `compare_kernel.mojo` result read | Scalar result crosses GPU→CPU |

**Deferred (not part of this phase):** `DeviceState.load` / `store` / `all_true`
/ `any_true` are used by CPU-side SIMD kernels and unique/count_nonzero
operations. These could also route through `.to_cpu()` if they become hot paths,
but they're niche operations.

### Phase 4 — Async training loop

**Goal:** Thread `sync=False` through the entire autograd chain so forward +
backward run entirely async on GPU, with a single sync at `SGD.step()`.

**Current state:**
- NDBuffer GPU dispatch methods default `sync=True` (callers must opt out)
- Kernel launchers default `sync=False` (Phase 1 — callers pass sync explicitly)
- Compound ops batch internal syncs (Phase 2)

**What needs to change:**

1. **Add `sync: Bool = True` (comptime) to every Tensor op**: `sum`, `mean`,
   `matmul`, `relu`, `sigmoid`, `tanh`, `add`, `mul`, `sub`, `div`, `view`,
   `gather`, `stack`, `shuffle`, `dropout`, `pad`, `softmax`, `cross_entropy`,
   `bce`, etc. — ~30 methods. Parameter flows down to NDBuffer methods, which
   pass to kernel launchers. With kernel launcher default `False`, passing
   `sync=False` genuinely skips the sync.

2. **Add `sync` to every backward handler**: All 58 op code handlers
   (`SumBackward`, `AddBackward`, `MulBackward`, `MatmulBackward`, etc.).
   Each handler calls `parent.update_grad()` which dispatches GPU ops (add,
   scatter-add, copy). These must run async.

3. **Thread through `Ancestor.update_grad()`**: Its op dispatch (`AddTensor`,
   `ScatterAddTensor`, `ZeroGrad`, etc.) launches GPU operations that should
   batch under a single sync.

4. **`SGD.step()`** remains the single sync point. Its 7 `map_to_host()` call
   sites for CPU-side read-modify-write of params/grads synchronize the entire
   stream before reading.

5. **`Tensor.backward()`** — the outermost caller. Accept `sync: Bool = True`,
   thread through the DFS loop:
   ```mojo
   def backward[sync: Bool = True](mut self, ...):
       # Phase 2: DFS graph collection (no GPU ops)
       # Phase 3: reverse topological execution via Backward.invoke
       #   Each handler syncs only when asked
       if sync and self.is_on_gpu():
           self.device_context.synchronize()
   ```

**Design chosen — runtime `sync: Bool` (see §9a for full rationale):**

- Default `True` at user-facing layer (Tensor ops, `backward()`)
- Default `False` at all internal layers (kernel launchers, NDBuffer dispatch, backward handlers)
- Top-down percolation — single `sync` param threads from Tensor → NDBuffer → kernel launcher
- Branch cost: single `if sync:` per kernel launch, always correctly predicted — negligible

**Rejected — comptime `sync`:** Incompatible with runtime decisions like `sync_on_odd_steps`.
Requires awkward workarounds for `__iadd__` operators. Double compile time.

**Rejected — global `GPU.sync_mode` flag:** Atomic read per op. Breaks composability
(mixed sync/async within same graph impossible).

### §9a. Design Decision: Comptime vs Runtime `sync`

**Question:** Should `sync` be a comptime parameter (like `track_grad`) or a runtime `Bool`?

**Analysis:**

| Aspect | Runtime `Bool` | Comptime `Bool` |
|---|---|---|
| Stack consistency | Full — NDBuffer/kernel layer already uses runtime `Bool` | Mixed — NDBuffer dispatch & kernel launchers are runtime `Bool`, so comptime at Tensor ops still flows to a runtime check |
| Gradbox operators | Works — `__iadd__` accepts `sync: Bool` arg | Broken — `__iadd__` has no comptime parameter slot. Requires awkward `add_grad[sync]()` methods alongside `+=` |
| Compile time | No change | ~2× from double specialization (sync=True + sync=False for every function) |
| Branch elimination | Runtime `if sync:` — single local `Bool` check | Dead-code eliminated — `if True/False:` removed |
| Signature changes | ~100 (same count either way) | ~100 (same count either way) |
| `track_grad` consistency | No — different paradigms | Yes — matches existing pattern |

**The branch cost:** A single `if sync:` on a local variable at ~100 kernel launches per step. Even at 1000 steps/s, that's ~100K branches — all correctly predicted (always not-taken in training). Negligible — single-digit microseconds.

**The real cost** is `ctx.synchronize()` itself, a full GPU pipeline drain. Comptime vs runtime doesn't change how many times that's called.

**Verdict:** Proceed with **runtime `sync: Bool`**. The comptime complexity — mixed layers, operator workarounds, compile-time blowup — outweighs the marginal branch-elimination benefit.

### Sync Propagation Rules

**Rule 1 — Runtime `Bool`.** `sync` is always a runtime `Bool` parameter. No comptime variants (see §9a for rationale).

**Rule 2 — Top-down percolation.** `sync` enters at the outermost API (Tensor ops, `backward()`) and flows down through every layer to the kernel launcher. No layer decides independently whether to sync.

**Rule 3 — User-facing defaults `sync=True`.** Tensor ops (`sum`, `matmul`, `add`, etc.), `Tensor.backward()`, and `Tensor.print()` all default `sync=True`. Safe by default — callers opt into async explicitly.

**Rule 4 — Internal layers default `sync=False`.** Kernel launchers (Phase 1), DeviceState methods (Phase 1), NDBuffer dispatch methods, backward handlers, and `Ancestor.update_grad()` all default `sync=False`. This is already the state for kernel launchers and DeviceState methods. The hidden layers never sync unless explicitly told to by the layer above.

**Why asymmetric defaults:**

```mojo
# User-facing — safe by default
tensor.sum(...)                            # sync=True → result ready for CPU
loss.backward()                            # sync=True

# Optimized path — opt in
loss.backward(sync=False)                  # async backward
optimizer.step()                           # single sync at step time

# Runtime decisions still work
var should_sync = step % 100 == 0
loss.backward(sync=should_sync)            # periodic sync for monitoring
```

**Backward sync design:**

`Tensor.backward()` does NOT pass `sync` to individual handlers. Instead:

```mojo
def backward(self, seed: Scalar = 1.0, sync: Bool = True):
    # All 58 backward handlers launch kernels — each passes sync=False internally
    self._run_backward_graph(seed)
    # Single sync at the end if requested
    if sync and self.is_on_gpu():
        self.device_context.synchronize()
```

This means backward handlers don't need `sync` in their signatures at all — they always run with internal `sync=False` (Rule 4). The backward pass queues all gradient operations asynchronously, then the outer `backward()` drains once. This avoids unnecessary per-op syncs during graph traversal of potentially hundreds of nodes.

**§9a continues with the implementation plan.**

### §9b. Async Training Code Path Walkthrough

Example: `C = A + B` → `C.backward(sync=False)` → `A.grad().print()`

**Forward — `C = A + B`:**

```
Addition.forward[track_grad=True](A, B, sync=False)
  → NDBuffer.arithmetic_ops[ADD](A.buffer, B.buffer, sync=False)
    → BinaryOperations.launch[ADD](A, B, sync=False)
      → ctx.enqueue_function(kernel_add, args...)    # queued, no sync
```

One kernel queued. CPU returns immediately.

**Backward — `C.backward(sync=False)`:**

```
Phase 1: seed_grad — scalar fill (kernel queued, no sync)
Phase 2: DFS graph collection (CPU-only)
Phase 3: reverse topological execution:
  AddBackward.backward(output, parent_ids)
    → ancestor.update_grad(grad_contrib, AddTensor, None)
      → gradbox.__iadd__(incoming)
        → NDBuffer.arithmetic_ops[ADD](..., sync=False)
          → BinaryOperations.launch[ADD](..., sync=False)
            → ctx.enqueue_function(kernel_add, args...)    # queued, no sync
No sync at end — returns immediately
```

All backward kernels queued. Zero CPU syncs.

**Read — `A.grad()`:**

```
Gradbox.detach()
  → contiguous_device_state()
    → DeviceState.into(shape)
      → map_to_host()    ← IMPLICIT SYNC — GPU pipeline drains here
      → memcpy to CPU
```

Exact 1 sync — the unavoidable GPU→CPU barrier.

**Print — `.print()`:**

Data already on CPU from `.grad()` — zero syncs.

| Step | GPU kernels | Syncs |
|---|---|---|
| `C = A + B` | 1 (add) | 0 |
| `C.backward(sync=False)` | 1 (gradbox add) | 0 |
| `A.grad()` | 1 (detach copy) | 1 (implicit via map_to_host) |
| `.print()` | 0 | 0 |
| **Total** | **3 kernels** | **1 sync** |

Compare to default `sync=True` path: ~3 separate syncs (1 per kernel launch) + 1 at grad() = 4 syncs. Async collapses to 1.

### Rough change count

| Layer | Methods to change |
|---|---|
| Tensor ops (~30) | sum, mean, matmul, add, mul, sub, div, neg, relu, sigmoid, tanh, exp, log, view, gather, stack, shuffle, dropout, pad, softmax, cross_entropy, bce, min, max, argmin, argmax, squeeze, unsqueeze, transpose, permute, reshape |
| NDBuffer dispatch (10) | Already have `sync` param — just thread through |
| Backward handlers (58) | One per op_code — add `sync` to each `backward()` signature |
| Ancestor/Ancestors | `update_grad()`, grad dispatch |
| `Backward.invoke()` | Plumbing through jump table |

### §9c. Unaddressed Gaps

**Gap 1 — NDBuffer dispatch defaults must flip.** 11 NDBuffer methods default `sync=True`. Even if backward handlers don't take sync params, the ops they call (`gradbox += incoming` → `NDBuffer.arithmetic_ops`) trigger per-op syncs. The async path is dead unless these flip to `sync=False`. Forward ops then explicitly pass `sync=True`.

| Layer | Current default | Phase 4 target |
|---|---|---|
| NDBuffer dispatch (11 methods) | `sync=True` | **`sync=False`** |
| Tensor ops (~35) | no sync param | `sync: Bool = True` |
| Forward structs (~25) | mostly no sync param | thread `sync` through |
| Kernel launchers (42) | `sync=False` ✅ | no change |
| Backward handlers (58) | no sync param | no change — hardcode `sync=False` internally |

**Gap 2 — Compound ops without sync threading.** `StdDev.forward` lacks a `sync` param entirely — hardcodes `sync=False` on Welford + `.sync()` at end. `Gather.forward` also lacks `sync`. These need `sync` threading to be consistent with top-down percolation.

**Gap 3 — Low-level helpers default `sync=True`.** ~15 functions across `filler.mojo`, `bce_kernel.mojo`, `division_kernel.mojo`, `minmax_kernel.mojo`, `sum_mean_reduction.mojo`, `dropout_kernel.mojo`, `variance.mojo`, `softmax.mojo` default `sync=True`. Called by both forward and backward paths — must accept threaded `sync` arg.

**Gap 4 — `Ancestor.update_grad()` and `Filler.scatter_add`.** The `ScatterAddTensor` branch calls `Filler.scatter_add` which defaults `sync=True`. If backward runs async, this must be called with `sync=False`. Needs threading through `update_grad`'s internal calls even when `update_grad` itself doesn't expose sync.

**Gap 5 — No async tests.** All existing tests use default `sync=True`. Need GPU-guarded tests that run with `sync=False` and verify results via `to_cpu()` (the sync barrier).

**Gap 6 — What's NOT changing (confirmed):**
- `Tensor.backward()` — sync at end only, per-handler threading skipped ✅
- `SGD.step()` — already syncs via `map_to_host` + explicit sync, no change ✅
- Kernel launchers — already `sync=False` (Phase 1) ✅
- CPU path — unaffected, `sync` param ignored ✅

### §9d. BLOCKER — Dunders Don't Propagate Sync

The `sync` parameter cannot be threaded through operator-overload dunder methods (`__add__`, `__sub__`, `__mul__`, `__truediv__`, `__neg__`, `__pow__`, `__iadd__`, `__isub__`, `__imul__`, `__itruediv__`) as runtime parameters because Mojo restricts the number of runtime operands dunders may accept (`__add__` = 2, `__neg__` = 1, etc.). This creates a chain of silent sync-loss across three layers.

#### Current State

| Layer | Dunders | Sync | How called |
|---|---|---|---|
| **Tensor** | `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__neg__`, `__pow__` (regular) + `__iadd__`, `__isub__`, `__imul__`, `__itruediv__` (inplace) | No sync param — always default | User-facing `a + b`, `a += b`, backward handlers |
| **Gradbox** | `__iadd__`, `__isub__`, `__imul__`, `__itruediv__` (scalar + Gradbox overloads) | No sync param — always default | `gradbox += incoming` in backward handlers |
| **NDBuffer** | `__add__`, `__sub__`, `__mul__`, `__neg__`, `__truediv__`, `__pow__`, `__rmul__` (regular) + `__iadd__`, `__isub__`, `__imul__`, `__itruediv__` (inplace) | No sync param — calls `arithmetic_ops[Op](other)` with default `sync=False` (Phase 3 default) | Backward handlers via `gradbox.buffer() * mask_ndb`, Tensor inplace via `self.buffer.__iadd__(other.buffer)` |

#### The Gap

When a user (or internal code) calls a Tensor dunder:
```
a + b                    # Tensor.__add__ — can't pass sync
loss += grad             # Tensor.__iadd__ — can't pass sync
```

The internal chain is:
```
Tensor.__add__ → Adder.forward[track_grad](self, other) → ... → Adder calls NDBuffer ops with sync=False (default)
Tensor.__iadd__ → self.buffer.__iadd__(other.buffer) → NDBuffer.__iadd__ → arithmetic_ops[sync=False]
```

Even if `Adder.forward` has `sync` as a runtime param, the dunder can't pass it through because `__add__` can't accept the extra param. The `sync` value is silently lost — `sync=True` (intended by user) becomes `sync=False` (the NDBuffer default from Phase 3).

#### All NDBuffer Dunder Call Sites (44 total, 11 files)

**Operator syntax on NDBuffer from backward handlers (29 sites, 8 files):**

| File | Lines | Dunders | Context |
|---|---|---|---|
| `softmax.mojo` | 128, 160–169 | `__sub__`, `__mul__` | `SoftmaxBackwardDelegate.backward()` + `_softmax_components` |
| `exponentiator.mojo` | 33, 39, 65 | `__pow__`, `__mul__` | `ExponentiationBackward.backward()` + `Exponentiator.forward()` |
| `relu.mojo` | 35 | `__mul__` | `ReLU.backward()` |
| `dropout.mojo` | 58, 62 | `__mul__` | `Dropout.backward()` |
| `product_reduction.mojo` | 410 | `__mul__` | `ProductBackward.backward()` |
| `maxmin_scalar.mojo` | 52, 94 | `__mul__` | `MaxBackwardScalar.backward()`, `MinBackwardScalar.backward()` |
| `exponential.mojo` | 22 | `__mul__` | `ExponentialBackward.backward()` |
| `crossentropy.mojo` | 542–549, 718–723 | `__mul__`, `__neg__`, `__truediv__`, `__sub__`, `__add__`, `__rmul__` | `_ce_loss_class_indices`, `_ce_loss_probabilities` helpers |

**Dunders called by name from Tensor/Gradbox (15 sites, 3 files):**

| File | Lines | Dunder | Caller |
|---|---|---|---|
| `tensor.mojo` | 2372, 2389, 2397, 2414, 2431, 2771, 2787, 2803, 2819 | `__iadd__`, `__isub__`, `__imul__`, `__itruediv__` on `self.buffer` | `Tensor.__iadd__` (Tensor + scalar), `Tensor.__isub__` (Tensor + scalar + Gradbox), `Tensor.__imul__` (Tensor + scalar), `Tensor.__itruediv__` (Tensor + scalar) |
| `tensor.mojo` | 2530 | `__abs__` on `self.buffer` | `Tensor.__abs__` |
| `gradbox.mojo` | 191, 500 | `__abs__`, `__getitem__` on `self.buffer()` | `Gradbox.__abs__`, `Gradbox.__getitem__` |
| `views.mojo` | 320, 346 | `__getitem__` on `tensor.buffer` | `Viewer.forward` |

#### Resolution Options

**Option A — Comptime sync on all NDBuffer dunders (rejected).**
Add `sync: Bool = False` as comptime param to all 16 NDBuffer dunder methods. Each passes it through to the internal `arithmetic_ops[Op](sync=sync)` call. Tensor/Gradbox dunders and forward structs then explicitly pass `sync` to NDBuffer dunders.

*Pro:* Minimal code changes at call sites (just add `[sync]` to existing calls).
*Con:* Adds surface area to 16 trivial wrappers. Backward handlers (29 sites) all compile with default `False` — no actual improvement there, just noise.

**Option B — Replace NDBuffer dunder calls with explicit ops (preferred).**
Do NOT add `sync` to NDBuffer dunders. Instead, replace all forward-path NDBuffer dunder calls with explicit `arithmetic_ops[Op](sync=...)` / `scalar_ops[Op](sync=...)` / `unary_ops[Op](sync=...)` calls. NDBuffer dunders remain as backward-only convenience wrappers — backward handlers don't need sync (they run inside `backward()` which syncs once at end).

*Pro:* No new surface area on NDBuffer. Forward paths use the sync-aware API directly. Clear semantic boundary: dunders = backward (sync=False), explicit methods = forward (sync-aware).
*Con:* 44 call sites to audit and convert. Some conversions lose syntactic readability (`a * b` → `a.arithmetic_ops[Multiply](b, sync=sync)`).

Backward-handler sites (29 of 44) stay as operator syntax — they're correct with `sync=False`. Tensor/Gradbox inplace dunders and forward struct operators (15 of 44) convert to explicit calls that accept and propagate `sync`.

| Category | Sites | Action |
|---|---|---|
| Backward handlers (29) | `softmax.mojo`, `relu.mojo`, `dropout.mojo`, `product_reduction.mojo`, `maxmin_scalar.mojo`, `exponential.mojo`, `exponentiator.mojo:33,39`, `crossentropy.mojo` (backward helpers) | Keep as-is — dunders, default `sync=False` |
| Forward struct operators (3) | `exponentiator.mojo:65` (`**`), `softmax.mojo:128` (`-`), `crossentropy.mojo` (~14 forward-helper calls) | Replace with `.__op__[Op]` dunder-name call or `arithmetic_ops` |
| Tensor inplace dunders (10) | `tensor.mojo` — `self.buffer.__iadd__` etc. | Replace with `self.buffer.arithmetic_ops[Add](..., sync=sync)` |
| Gradbox inplace dunders (2) | `gradbox.mojo` — `self.buffer().__iadd__` etc. | Replace with explicit call |

**Decision path:** Option B preferred. Start with a feasibility study (Action Item 1) — catalog every site and confirm replacement is safe. Then mechanical conversion (Action Items 2–4).

### §9e. Phase 4 Action Items

Consolidated itemized action items in recommended execution order:

 | # | Action | Files affected | Change type |
|---|---|---|---|---|
| 1 | **Feasibility study: catalog all NDBuffer dunder calls and confirm replacement with explicit `arithmetic_ops[Op]` / `scalar_ops[Op]` / `unary_ops[Op]`** | All `.mojo` files | Audit only — no code changes |
| 2 | **BLOCKER: Replace forward-path NDBuffer dunder calls with explicit ops** — forward structs (exponentiator `**` → `.scalar_ops[POW]`, softmax `-` → `.arithmetic_ops[Subtract]`, crossentropy ~14 operator calls → explicit ops), Tensor inplace dunders (`self.buffer.__iadd__` → `self.buffer.arithmetic_ops[Add]`), Gradbox inplace dunders | `tensor.mojo`, `gradbox.mojo`, `exponentiator.mojo`, `softmax.mojo`, `crossentropy.mojo` | Replace operator syntax with explicit method call, thread sync |
| 3 | **BLOCKER: Add comptime `sync` to Tensor dunders** — add `sync: Bool = True` to all Tensor regular and inplace dunders; forward struct calls pass `sync` as runtime arg | `tensor.mojo` | Add comptime param + thread to callees |
| 4 | **BLOCKER: Add comptime `sync` to Gradbox dunders** — add `sync: Bool = True` to all Gradbox inplace dunders; thread to explicit `arithmetic_ops[Op](sync=sync)` calls | `gradbox.mojo` | Add comptime param + thread to callees |
| 5 | Flip NDBuffer dispatch defaults (`sync=True` → `sync=False`) | `ndbuffer.mojo` | 11 method signatures — default value only |
| 6 | Add `sync: Bool = True` to all Tensor ops + thread to forward structs | `tensor.mojo` + ~25 forward files | Add param + pass-through |
| 7 | Thread `sync` through low-level helpers (Filler, Bce, SumMeanReduction, Div, Minmax, Dropout, Variance, Softmax, ArgMinMax) | ~10 files | Add param to callers; callees already have `sync` |
| 8 | Add `sync` param to compound ops that lack it (StdDev, Gather forward) | `std_deviation.mojo`, `gather.mojo` | Add param + thread internal calls |
| 9 | Thread `sync=False` through `Ancestor.update_grad()` internal Filler call | `ancestry.mojo` | Pass `sync=False` to `scatter_add` |
| 10 | Add GPU-guarded async tests | `tests/test_gpu_all.mojo` or new file | 5–10 tests exercising `sync=False` path |
| 11 | Validate: run full CPU test suite with default `sync=True` — zero regressions | CI | Compile + execute all |

**Execution order rationale:** 1 is audit-only — do it first to confirm the scope and viability of Option B. Blockers 2→3→4 fix the fundamental dunder sync gap using Option B (replace forward-path dunder calls with explicit ops, then add comptime sync to Tensor/Gradbox dunders). 5→6→7→8 is bottom-up (dependencies first). 9 is independent. 10 is last (needs all changes in place). 11 runs at each step to catch regressions early.

| Phase | Syncs removed | Speedup | Complexity |
|---|---|---|---|---|---|
| 0 | 1 per `into()` | <1% | Trivial |
| 1 | 0 (behavioral no-op) | 0% | Trivial (mechanical) |
| 2 | ~2 per compound op | <1% | Trivial |
| 3 | N per print/to_ndarray call | ~N× per call (N elem) | Low |
| 4 | ~N per training step | **2–5×** | High |

### §9f. Empirical Findings — Async Data Transfer Alone Doesn't Improve Performance

**Test setup:** MNIST GPU training, 784→128→32→10, batch_size=64, 15 epochs.
- **Version A:** `to_gpu(sync=True)` — per-batch sync
- **Version B:** `to_gpu(sync=False)` — async batch transfer, only sync at `backward(sync=True)`

**Result:** Both ~29.5s/epoch. No measurable difference.

**Root cause:** Tensor dunders (`+`, `*`, `@`, etc.) default `sync=True`. The very first forward operation (`matmul`) forces a GPU pipeline drain — by which time the async data copy (microseconds for ~200KB) has long completed. Every subsequent `__add__`, `__sub__`, `relu()`, etc. adds another sync. The `backward(sync=True)` fence is irrelevant — the per-op syncs during forward are the bottleneck.

```
to_gpu(sync=False)      → queued (0µs CPU)
matmul[track_grad]      → enqueue kernel → SYNC → drain all prior work
__add__[sync=True]      → enqueue kernel → SYNC
relu[sync=True]         → enqueue kernel → SYNC
...                     → N syncs for N ops
```

**Key insight:** You must eliminate syncs from the forward path. Async data transfer alone is meaningless when every operation immediately syncs.

**How to fix — three options considered:**

| Option | Approach | Ergonomics | Complexity |
|---|---|---|---|
| A — Manual per-op | `x.__add__[sync=False](b1)` everywhere | Fragile — one missed sync stalls pipeline | None |
| B — Default `sync=False` on dunders | Change all 16 dunder defaults | Breaks safety contract — silent stale reads | Trivial |
| C — Comptime threading through Sequential | `model[sync=False](x)` threads through all layers | Ergonomically clean, but requires every layer to accept `[sync]` | High |

**Option C design (chosen for implementation):**

```
Sequential.__call__[sync: Bool = True](xs)
  → m[sync](out)                                          # Module dispatch
    → Module.__call__[sync: Bool = True](xs)
      → self.layer[Linear[...]]()[sync](xs^)              # Variant extract + call
        → Linear.__call__[sync: Bool = True](xs)
          → Matmul.forward[track_grad, sync=sync](xs, weight)
          → Adder.forward[track_grad, sync=sync](matmul_out, bias)
```

Every internal op uses `ForwardStruct.forward[..., sync=sync](...)` — never dunders. This guarantees every kernel in the forward path receives `sync` from the top-level caller.

**Variant dispatch nuance:** `self.layer[ConcreteType]()` extracts the concrete layer from the `Variant`. Then `[sync](xs)` calls `__call__[sync]` on the extracted value. All 10 layer types must accept `[sync: Bool = True]` on their `__call__`.

**Risk:** Any layer that internally uses a dunder (`x + y` instead of `Adder.forward[sync=sync](x, y)`) silently defaults to `sync=True`, creating an unexpected sync point. All internal arithmetic across all layers must be converted to explicit forward struct calls.

**Training loop with Option C:**
```mojo
var pred = model[sync=False](features_gpu)   # whole forward async
var loss = criterion(pred, labels_gpu)        # loss must also accept [sync]
loss.backward(sync=True)                       # single sync point
```

**What's needed:**

| Component | Change |
|---|---|
| `Sequential.__call__` | Add `[sync: Bool = True]`, thread to `m[sync](out)` |
| `Module.__call__` | Add `[sync: Bool = True]`, dispatch with `[sync]` |
| All 10 layer `__call__` | Add `[sync: Bool = True]`, thread to forward structs |
| Forward struct calls in layers | Replace all dunders with `Forward.forward[..., sync=sync]()` |
| Criterion/loss functions | Add `[sync]` param |
| `SequentialBLAS.__call__` | Same threading |

---

## 10. Summary Statistics

| Metric | Count |
|---|---|---|
| GPU kernel launchers | 42 |
| DeviceState methods with sync param | 5 (into() sync param removed in Phase 0) |
| NDBuffer GPU dispatch methods | 12 |
| Compound multi-kernel ops (batched) | 5 (Welford, StdDev, LayerNorm, Gather, CrossEntropy) |
| Total sync call sites | ~50 |
| Sync-default functions (kernel + DeviceState) | 0 — all `False` after Phase 1 |
| Sync-default functions (NDBuffer layer) | 12 — currently `True`, target `False` in Phase 4 |
| Backward handlers | 58 — internal `sync=False`, no sync parameter needed |
| GPU kernel files | 17 |
