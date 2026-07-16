# Checkpoint System — Gap Analysis & Architecture

## Current State

The checkpoint system lives in `tenmo/numpy_interop.mojo` (~206 lines) and consists of:

| Function | Scope | Format |
|---|---|---|
| `save(model, path)` | Single `Tensor` → `.npz` | NumPy `.npz` |
| `load(path) -> Tensor` | Single `Tensor` ← `.npz` | NumPy `.npz` |
| `save_checkpoint(model, path)` | `Sequential` model weights → `.npy` | Flat dict `{name: ndarray}` |
| `load_checkpoint(model, path)` | `Sequential` model weights ← `.npy` | Flat dict `{name: ndarray}` |

**Underlying infrastructure:**
- `NamedParameter` struct (`tenmo/named_parameter.mojo`) — pairs `name: String` + `tensor_ptr: UnsafePointer[Tensor]`
- `named_parameters(prefix)` — every module struct implements this to enumerate its trainable parameters
- `to_ndarray()` / `from_ndarray()` — convert between Mojo `Tensor`/`NDBuffer`/`Gradbox` and NumPy ndarrays

**Key enablers already in place:**
- NumPy interop with `memcpy`-based data transfer (CPU and GPU→CPU)
- PyTorch-compatible key naming (`"0.weight"`, `"0.bias"`)
- `load_checkpoint` silently skips missing keys (safe partial loading)

---

## Gap Analysis

| Ideal (from the passage) | Current | Gap |
|---|---|---|
| **Model state_dict** — all weights | Only `Sequential` containers. **`Embedding.weight` excluded** — `ModuleWrapper.named_parameters()` falls to `else` (empty) for `EMBEDDING` tag. | Missing Embedding |
| **Optimizer state_dict** — Adam moments / momentum velocities | `SGD` has `velocities: List[Gradbox]` but **zero serialization methods**. Velocities lost on save/load. Resume → cold momentum. | Full gap |
| **Scheduler state_dict** — LR, step count | **No LR scheduler exists** anywhere in the codebase. All LR management manual (`SGD.set_lr()`). | Full gap |
| **Resume training** — epoch, step | No epoch/step metadata saved. Loaded weights resume from step 0. | Full gap |
| **Best model tracking** — validation loss | Not implemented. Only latest weights saved. | Full gap |
| **Branching point** — step-numbered checkpoints | No step-based checkpoint naming or periodic save. | Full gap |
| **Provenance** — config, seed, git hash, dataset | Not captured. No audit trail for a given weights file. | Full gap |

### Summary: ~30% complete. Phases 1–4a done. Phase 5 done.

---

## Proposed Format

Expand from flat `{name: ndarray}` to structured dict:

```python
{
  "metadata": {
    "epoch": 10,
    "step": 5000,
    "best_loss": 0.123,
    "config": {"seed": 42, "dataset": "mnist", "arch": "784-128-10"},
    "timestamp": "2026-07-13T12:00:00",
    "git_hash": "abc123def"
  },
  "model": {
    "0.weight": ndarray,
    "0.bias": ndarray,
    "1.weight": ndarray,
    ...
  },
  "optimizer": {
    "type": "SGD",
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "clip_norm": 0.0,
    "clip_value": 0.0,
    "velocities": [ndarray, ...]
  },
  "scheduler": {                    # optional
    "type": "StepLR",
    "step_size": 30,
    "gamma": 0.1,
    "last_epoch": 5
  }
}
```

Backward compatible: `load_checkpoint` detects old format (flat keys with `"0.weight"` at top) vs new format (`"model"` key present).

---

## Files

| File | Status | Purpose |
|---|---|---|
| `tenmo/embedding.mojo` | **Edit** | Add `named_parameters()` |
| `tenmo/net.mojo` | **Edit** | Add `EMBEDDING` case to `ModuleWrapper.named_parameters()` |
| `tenmo/optim.mojo` | **Edit** | Add `SGD.state_dict()` + `SGD.load_state_dict()` |
| `tenmo/scheduler.mojo` | **New** | `Scheduler` trait + StepLR / MultiStepLR / CosineAnnealingLR |
| `tenmo/checkpoint.mojo` | **New** | `Checkpoint` struct + all save/load variants |
| `tenmo/numpy_interop.mojo` | **Edit** | Deprecate old fns → delegate to checkpoint module |
| `tenmo/__init__.mojo` | **Edit** | Export new symbols |

---

## Architecture: `tenmo/checkpoint.mojo`

```
Checkpoint
├── model_state: PythonObject      # dict of ndarrays
├── optimizer_state: PythonObject  # dict including velocities
├── scheduler_state: PythonObject  # dict (optional)
├── metadata: PythonObject         # dict w/ epoch, step, loss, config, hash

save_checkpoint(path, model, optimizer?, scheduler?,
                epoch?, step?, best_loss?, config?)
    → serializes full Checkpoint to .npy

load_checkpoint(path) → Checkpoint
    → deserializes, detects old vs new format

apply_to_model(mut model, checkpoint)
    → copies model_state into model's named_parameters

apply_to_optimizer(mut optimizer, parameters, checkpoint)
    → reconstructs SGD from optimizer_state + parameter list

apply_to_scheduler(mut scheduler, checkpoint)
    → restores scheduler internal counters
```

### Helper functions

```
save_weights(path, model)           # model-only, for inference
load_weights(mut model, path)       # model-only, backwards compat

save_best_if_improved(path_prefix, model, optimizer,
                      scheduler, current_loss, ...)
    → writes {path_prefix}_latest.ckpt always
    → writes {path_prefix}_best.ckpt only if current_loss < best_loss
    → returns new best_loss

save_step_checkpoint(path_template, step, model, optimizer, ...)
    → writes checkpoint_step_{step}.ckpt
```

---

## Implementation Phases

### Phase 1 — Embedding checkpointing (2 edits)
- `tenmo/embedding.mojo`: add `named_parameters(prefix)` returning `prefix + "weight"`
- `tenmo/net.mojo:936`: add `EMBEDDING` case to `ModuleWrapper.named_parameters()`

### Phase 2 — Optimizer serialization (1 edit)
- `tenmo/optim.mojo`: add `SGD.state_dict() -> PythonObject`
- `tenmo/optim.mojo`: add `SGD.load_state_dict(state, parameters) -> SGD`

### Phase 3 — LR Scheduler (NEW: `tenmo/scheduler.mojo`, ~250 lines)

Three structs, each using `@fieldwise_init` + `ImplicitlyCopyable & Movable` (same pattern as `SGD`).

All schedulers are **self-contained** — `step(mut self) -> Scalar[dtype]` returns the new LR. User applies it:
```mojo
optimizer.set_lr(scheduler.step())
```

#### `StepLR[dtype]`

| Field | Type | Description |
|---|---|---|
| `base_lr` | `Scalar[dtype]` | Initial LR |
| `step_size` | `Int` | Epochs between decays |
| `gamma` | `Scalar[dtype]` | Multiplicative factor |
| `last_epoch` | `Int` | Epoch counter (starts at -1, PyTorch convention) |

```
lr = base_lr * gamma ** (epoch // step_size)
```

#### `MultiStepLR[dtype]`

| Field | Type | Description |
|---|---|---|
| `base_lr` | `Scalar[dtype]` | Initial LR |
| `milestones` | `List[Int]` | Sorted epoch thresholds |
| `gamma` | `Scalar[dtype]` | Multiplicative factor |
| `last_epoch` | `Int` | Epoch counter |

```
lr = base_lr * gamma ** count(milestones <= epoch)
```

#### `CosineAnnealingLR[dtype]`

| Field | Type | Description |
|---|---|---|
| `base_lr` | `Scalar[dtype]` | Initial LR |
| `T_max` | `Int` | Half-cycle length (epochs) |
| `eta_min` | `Scalar[dtype]` | Minimum LR |
| `last_epoch` | `Int` | Epoch counter |

```
lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2
```

Requires `from std.math import cos, pi`.

#### Serialization (CHECKPOINT.md format)

Each scheduler implements:
- `state_dict() raises -> PythonObject` — encodes hyperparams + counters (Scalar → `np.array([Float64(x)])`, same encoding as `SGD.state_dict()`)
- `load_state_dict(mut self, state: PythonObject) raises` — mutable restore
- `@staticmethod from_state_dict(state: PythonObject) raises -> Self` — convenience factory

Format per-scheduler:
```
StepLR         → {"type": "StepLR", "lr": [f64], "step_size": 30, "gamma": [f64], "last_epoch": 5}
MultiStepLR    → {"type": "MultiStepLR", "lr": [f64], "milestones": [30,60,80], "gamma": [f64], "last_epoch": 5}
CosineAnnealingLR → {"type": "CosineAnnealingLR", "lr": [f64], "T_max": 50, "eta_min": [f64], "last_epoch": 10}
```

#### Integration

- `tenmo/__init__.mojo`: add `from .scheduler import StepLR, MultiStepLR, CosineAnnealingLR`

#### Tests — `tests/test_scheduler.mojo`

| # | Test | Verifies |
|---|---|---|
| 1 | `test_steplr_basic` | StepLR: 1.0 × γ at every `step_size` boundary |
| 2 | `test_steplr_state_dict` | StepLR: state_dict roundtrip preserves `last_epoch` + params |
| 3 | `test_multisteplr_basic` | MultiStepLR: decays only past milestones |
| 4 | `test_multisteplr_no_milestones` | MultiStepLR: empty milestones = no decay |
| 5 | `test_multisteplr_state_dict` | MultiStepLR: state_dict roundtrip |
| 6 | `test_cosineannealinglr_basic` | CosineAnnealingLR: base_lr at epoch 0, eta_min at epoch T_max |
| 7 | `test_cosineannealinglr_state_dict` | CosineAnnealingLR: state_dict roundtrip |
| 8 | `test_scheduler_integration` | End-to-end: `scheduler.step()` → `optimizer.set_lr()` → produces correct `optimizer.lr` |

Run with: `./execute.sh scheduler`

#### Risks

| Risk | Mitigation |
|---|---|
| `Scalar ** Int` (`pow`) not supported | Loop-based multiplication: `gamma_pow = 1; for _ in range(n): gamma_pow *= gamma` |
| `cos`/`pi` not in Mojo 1.0.0b2 `std.math` | Fallback via `Python.import_module("math")` |
| `Float64 → Scalar[float32]` truncation in `load_state_dict` | Acceptable for LR values (same precision used in `SGD.load_state_dict`); test in float32 |

### Phase 4 — Checkpoint module (NEW: `tenmo/checkpoint.mojo`)

Split into two sub-phases for pragmatic delivery:

#### Phase 4a — CPU-only checkpointing (the immediately useful 90%)

**Scope:** `Checkpoint` struct + save/load/apply helpers for CPU models. No GPU awareness. No scheduler integration (scheduler slot reserved, always empty). Best-model tracking deferred to Phase 5.

**Design rationale:**
- `Checkpoint` struct is NOT dtype-parameterized — holds opaque `PythonObject` dicts with CPU ndarrays
- `save_state` / `load_state` ARE dtype-parameterized (need `to_ndarray`, `ndarray_ptr`)
- `apply_to_model` uses direct `memcpy` (CPU only — same as old `load_checkpoint`)
- `save_weights` / `load_weights` convenience helpers for inference (model only)
- Old `save_checkpoint`/`load_checkpoint` in `numpy_interop.mojo` left intact — no delegation (avoids circular import since `checkpoint.mojo` depends on `numpy_interop.mojo` for `to_ndarray`/`ndarray_ptr`)
- Scheduler functions (`apply_to_scheduler`) not implemented — user calls `scheduler.load_state_dict(ckpt.scheduler_state)` directly
- Optimizer functions (`apply_to_optimizer`) not implemented — user calls `opt = SGD.load_state_dict(...)` directly

**API:**
```mojo
Checkpoint                  # struct with model_state, optimizer_state, scheduler_state, metadata
save_state(path, model, metadata?)           # model weights + metadata
save_state(path, model, optimizer, metadata?) # model + optimizer + metadata
load_state(path) -> Checkpoint               # auto-detects old/new format
apply_to_model(mut model, checkpoint)        # CPU memcpy (Phase 4b: GPU-aware)
save_weights(path, model)                    # model-only convenience
load_weights(mut model, path)                # model-only convenience (detects old/new format)
```

**Format (backward-compat auto-detected via `"model"` key presence):**
```python
{
  "model":      {"0.weight": ndarray, ...},
  "optimizer":  {"type": "SGD", "lr": [f64], ...},   # optional
  "scheduler":  {},                                     # slot for Phase 4b
  "metadata":   {"epoch": 10, "step": 5000, ...}       # optional
}
```

**Files to touch:**

| File | Change |
|---|---|
| `tenmo/checkpoint.mojo` | **New** — `Checkpoint` struct + `save_state`/`load_state`/`apply_to_model`/`save_weights`/`load_weights` |
| `tenmo/__init__.mojo` | Export `Checkpoint`, `save_state`, `load_state`, `apply_to_model`, `save_weights`, `load_weights` |
| `tests/test_checkpoint.mojo` | Add tests for new API |

#### Phase 4b — Device-aware checkpointing (deferred)

**Scope:** GPU-aware `apply_to_model`, `apply_to_optimizer`, `apply_to_scheduler`, Scheduler state serialization roundtrip.

**Why deferred:**

`apply_to_model` on GPU is not just "call `to_gpu` on the loaded tensor". Mojo's autograd graph holds ownership references via `Ancestor` handles and `Gradbox` refcounts. Replacing a GPU tensor in-place (e.g. `t = gpu_t^`) must:
- Not orphan the existing `Gradbox` (gradients in-flight during training)
- Not break the `Ancestor` parent chain that `backward()` walks
- Handle the case where the tensor is a view of another buffer (shared `Buffer` refcount)
- Coordinate with `SGD.load_state_dict` which does raw `memcpy` into `Gradbox.buffer()` — that path assumes CPU `UnsafePointer`

The current CPU path avoids all this because `memcpy` into `data_ptr()` does not change the Tensor identity or its graph connections. A GPU equivalent would need either: (a) a new `to_cpu().copy_into()`-style copy that preserves graph, or (b) compiling transfer into a backward-visible `DeviceTransfer` node.

Separately, GPU velocity restore in `SGD.load_state_dict` requires reading velocity bytes from a CPU ndarray into a GPU `Gradbox.buffer()`, which involves CUDA memcpy — not yet wired.

All of this is doable, but it's a focused yak-shave with its own test infrastructure. Phase 4a unblocks all CPU training pipelines in the meantime.

**Deferred until:**
- A real GPU training loop needs checkpoint resume and hits this gap
- Or someone volunteers to write the GPU-specific test fixtures

**Key challenges:**
- `apply_to_model` on GPU: in-place tensor replacement vs autograd ownership
- `SGD.load_state_dict` needs device-aware velocity buffer restore
- Scheduler `apply_to_scheduler` requires a common trait or dispatcher

#### Risks (Phase 4a)

| Risk | Mitigation |
|---|---|
| `PythonObject = {}` default in function signature not supported in Mojo 1.0.0b2 | Overload two `save_state` variants (3-arg / 4-arg) — metadata always a positional param in both |
| Circular import `checkpoint.mojo` ↔ `numpy_interop.mojo` | Old `save_checkpoint`/`load_checkpoint` in `numpy_interop.mojo` remain untouched (no delegation) |
| Name collision `save_checkpoint` (new module) vs old `numpy_interop` | New API uses `save_state` / `load_state` — no overlap |

### Phase 5 — Best model & step-based tracking

Adds two high-level helper functions to `tenmo/checkpoint.mojo`. Both build on Phase 4a's `save_state` — no additional module needed.

#### `save_best_if_improved`

```mojo
# Without optimizer
def save_best_if_improved[
    dtype: DType, //
](
    path_prefix: String,
    model: Sequential[dtype],
    current_loss: Float64,
    best_loss: Float64,
    metadata: PythonObject,
) raises -> Float64:
    # Always saves {path_prefix}_latest.npy
    # If current_loss < best_loss, saves {path_prefix}_best.npy
    # Returns min(current_loss, best_loss)

# With optimizer
def save_best_if_improved[
    dtype: DType, //
](
    path_prefix: String,
    model: Sequential[dtype],
    optimizer: SGD[dtype],
    current_loss: Float64,
    best_loss: Float64,
    metadata: PythonObject,
) raises -> Float64:
```

Usage pattern:
```mojo
var best_loss = Float64(inf)
for epoch in range(num_epochs):
    train()
    var val_loss = validate()
    best_loss = save_best_if_improved("/tmp/model", model, val_loss, best_loss, {})
```

#### `save_step_checkpoint`

```mojo
# Without optimizer
def save_step_checkpoint[
    dtype: DType, //
](
    path_prefix: String,
    step: Int,
    model: Sequential[dtype],
    metadata: PythonObject,
) raises:

# With optimizer
def save_step_checkpoint[
    dtype: DType, //
](
    path_prefix: String,
    step: Int,
    model: Sequential[dtype],
    optimizer: SGD[dtype],
    metadata: PythonObject,
) raises:
```

Saves to `{path_prefix}_step_{step}.npy`.

#### Files to touch

| File | Change |
|---|---|
| `tenmo/checkpoint.mojo` | Add `save_best_if_improved` + `save_step_checkpoint` functions |
| `tenmo/__init__.mojo` | Export new symbols |
| `tests/test_checkpoint.mojo` | Add tests for both new functions |

#### Tests

| Test | Verifies |
|---|---|
| `test_save_best_if_improved_saves_latest` | Always saves `_latest.npy` regardless of loss |
| `test_save_best_if_improved_saves_best` | Saves `_best.npy` only when `current_loss < best_loss` |
| `test_save_best_if_improved_no_improvement` | Does NOT overwrite `_best.npy` when loss worsens |
| `test_save_best_if_improved_returns_min` | Returns `min(current_loss, best_loss)` |
| `test_save_best_if_improved_with_optimizer` | Full state saved in both files |
| `test_save_step_checkpoint_basic` | Saves `{prefix}_step_{N}.npy` |
| `test_save_step_checkpoint_with_optimizer` | Optimizer state included |

### Phase 6 — Provenance & examples
- Git hash capture, config dict, example usage, extended tests
