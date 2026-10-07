# Device-Resident Data Loading for `NativeLoader` and `DataLoader`

**Status:** Phase X + 5a + steps 0–2 + 5b + norm-hoist + `mnist_gpu` migration + 5c landed (`03ea4e8`→`e512431`, 2026-10-04); tree clean (see §12).
**Date:** 2026-10-04 (status refreshed 2026-10-07)
**Line numbers:** stale — `dataloader.mojo` grew ~+30–280 lines as the device path landed. Symbol names are authoritative; treat `:line` refs as pre-5b approximations.
**Scope:** `tenmo/dataloader.mojo` (`Dataset` trait, `NativeLoader`, `DataLoader`, `TensorDataset`, `NumpyDataset`), the gather kernel family, and the Python binding surface.
**Reference:** `mnist_pytorch_optimized.py` vs `examples/mnist_gpu.mojo`.
**Related:** [`gather-kernel-double-dispatch.md`](gather-kernel-double-dispatch.md) — a pre-existing perf defect in the kernel this work extends (Phase X).

- [1. The Premise](#1-the-premise)
- [2. Findings — Why Device Residency Is Impossible Today](#2-findings--why-device-residency-is-impossible-today)
- [3. The Central Design Question](#3-the-central-design-question)
- [4. Sequencing: Which Loader First](#4-sequencing-which-loader-first)
- [5. Design](#5-design)
- [6. The Sync Budget After the Change](#6-the-sync-budget-after-the-change)
- [7. Alternatives Considered and Rejected](#7-alternatives-considered-and-rejected)
- [8. Open Questions and Risks](#8-open-questions-and-risks)
- [9. Call-Site Migration](#9-call-site-migration)
- [10. Test Strategy](#10-test-strategy)
- [11. Rollout](#11-rollout)
- [12. Implementation Status](#12-implementation-status-2026-10-04-refreshed-2026-10-07)
- [Appendix A — Related Defect: `GatherKernel` Double-Dispatch](#appendix-a--related-defect-gatherkernel-double-dispatch)
- [Appendix B — Reference Inventory](#appendix-b--reference-inventory)

---

## 1. The Premise

`mnist_pytorch_optimized.py` wins by doing three things the Mojo version cannot currently do at all:

**1. One-time device residency.** `prep()` (mnist_pytorch_optimized.py:17-21) normalizes on-device once and leaves `X_train` / `y_train` resident for the whole run:

```python
def prep(ds):
    x = ds.data.to(device=device, dtype=torch.float32).div_(255.0)
    x = x.sub_(0.1307).div_(0.3081).view(len(ds), -1)  # (N, 784)
    y = ds.targets.to(device)
    return x.contiguous(), y
```

**2. Zero-copy eval batches.** The test loop slices, never gathers (mnist_pytorch_optimized.py:87-90):

```python
data, target = X_test[i : i + batch_size], y_test[i : i + batch_size]
```

**3. One sync per epoch, not per batch.** Loss and accuracy accumulate into device scalars (mnist_pytorch_optimized.py:63-64) and are read back once at (mnist_pytorch_optimized.py:78-79):

```python
loss_sum = torch.zeros((), device=device)
correct = torch.zeros((), device=device, dtype=torch.long)
...
train_loss = loss_sum.item() / n_train_batches
train_acc = 100.0 * correct.item() / n_train
```

`examples/mnist_gpu.mojo` currently transfers **every batch, every epoch** (mnist_gpu.mojo:176-177, repeated at :204-205):

```mojo
var features_gpu = batch.features.to_gpu(gpu, sync=False)
var labels_gpu = batch.labels.to_gpu(gpu, sync=False)
```

That is 14,070 batch transfers over a 15-epoch run, and it is the entire remaining gap. The header comment in mnist_gpu.mojo:3-4 already claims "running entirely on GPU after a one-time parameter transfer" — one-time for the *parameters*, not the data.

**Memory budget** (float32, MNIST): train features 60000×784×4 = **188.2 MB**, test features **31.4 MB**, train labels 60000×8 = **0.5 MB**. Comfortable. This will not hold for CIFAR-10 or ImageNet at full resolution — see [§3](#3-the-central-design-question).

---

## 2. Findings — Why Device Residency Is Impossible Today

`tenmo/dataloader.mojo` contains **zero** device awareness: no `is_on_gpu`, no `device:` parameter, no `Optional[Device]`, no import of `Device`/`GPU`. It is a host-memory module that happens to store `Tensor`s.

### 2.1 The `Dataset` trait is defined by a host-pointer contract

The trait exposes raw pointers as its only data accessors (dataloader.mojo:49-59):

```mojo
    def get_features_ptr(
        ref self,
    ) -> Pointer[Scalar[Self._sample_dtype], ImmutAnyOrigin]:
        """Get raw pointer to feature data."""
        ...
```

Both shipped conformers implement them as a bare host pointer dereference — `NumpyDataset` at dataloader.mojo:502-510, `TensorDataset` at dataloader.mojo:667-675:

```mojo
    def get_features_ptr(
        ref self,
    ) -> Pointer[Scalar[Self.sample_dtype], ImmutAnyOrigin]:
        return self._features.data_ptr().as_imm()
```

### 2.2 A GPU tensor has no host pointer to return

This is the hard blocker. `NDBuffer` carries an optional `device_state`, and `data_ptr()` unconditionally projects the **host** `Buffer` (ndbuffer.mojo:1546-1549):

```mojo
    @always_inline
    def data_ptr(ref self) -> Pointer[Scalar[Self.dtype], MutAnyOrigin]:
        return self.buffer.unsafe_ptr()
```

But a GPU-resident `NDBuffer` is constructed *without* a CPU buffer. `to_device` is explicit about this (ndbuffer.mojo:555-560):

```mojo
            # Create new NDBuffer:
            #   - contiguous
            #   - offset = 0
            #   - no CPU buffer
            var result = NDBuffer[Self.dtype].with_device_state(
                new_device_state, self.shape
            )
```

So for a GPU tensor, `buffer` is an empty `Buffer` and `data_ptr()` yields a null/dangling host address. `is_on_gpu()` is the discriminator (ndbuffer.mojo:605-607) and nothing in the loader consults it.

**Consequence:** handing a GPU-resident tensor to `NumpyDataset` or `TensorDataset` today does not error — it silently `unsafe_memcpy`s from a dangling pointer. That is a segfault or silent corruption, not a clean panic. This is the single most important thing to fix first, because the current failure mode is unsafe rather than loud.

### 2.3 Inventory of host-memory assumptions

Every row is a line that must be revisited.

| Location | Assumption | Breaks on GPU? |
|---|---|---|
| dataloader.mojo:502-510, 667-675 | `get_*_ptr()` → `data_ptr()` | **Yes — dangling** |
| dataloader.mojo:321-359 | `NativeLoader._fill_batch` bulk `unsafe_memcpy` | **Yes** |
| dataloader.mojo:362-381 | `NativeLoader._fill_batch` row-by-row `unsafe_memcpy` | **Yes** |
| dataloader.mojo:384-407 | SIMD host normalization on raw pointer | **Yes** |
| dataloader.mojo:1117-1148 | `DataLoader._fill_batch` row `unsafe_memcpy` | **Yes** |
| dataloader.mojo:532-535, 717-720 | `Tensor.zeros(shape)` in `__getitem__` — no `device=` | Allocates on CPU |
| dataloader.mojo:204-209, 235-240 | `NativeLoader` batch buffers — no `device=` | Allocates on CPU |
| dataloader.mojo:933-936, 951-956 | `DataLoader.__init__` batch buffers — no `device=` | Allocates on CPU |
| dataloader.mojo:985-1015 | `_make_buffers` (via `set_shuffle`) — no `device=` | Allocates on CPU |
| dataloader.mojo:871-881 | `src.contiguous()` on a possibly-strided GPU source | Full-dataset device copy ([§5.6](#56-normalization-must-leave-the-loader)) |
| dataloader.mojo:809 | documented contract: "per batch: zero allocations" | Violated by any allocate-per-batch gather |

`Tensor.zeros` already accepts `device:` (tensor.mojo:1545-1548), so the buffer-allocation rows are one-line fixes.

### 2.4 What already works — the good news

Not everything needs work. Two paths are already device-clean:

**`slice()` on a GPU tensor is metadata-only.** `Tensor.slice` (tensor.mojo:3583-3613) delegates to `View.forward`, which calls `NDBuffer.share()`; `share` copies the `Buffer` handle *and* the `device_state`, and moves no data (ndbuffer.mojo:1438-1448):

```mojo
        var ndb = NDBuffer[Self.dtype](
            buffer=self.buffer.copy(),
            shape=new_shape,
            strides=new_strides,
            offset=offset,
        )
        ndb.device_state = self.device_state.copy()
```

**`is_contiguous()` is layout-only** (ndbuffer.mojo:857-859) and ignores device:

```mojo
    def is_contiguous(self) -> Bool:
        return self.strides.is_contiguous(self.shape)
```

Therefore the `DataLoader` sequential path (dataloader.mojo:1085-1098) — which already slices rather than memcpys — **works unchanged on a device-resident source**. This is precisely the PyTorch eval-loop shape, and it is already implemented. That is a meaningful chunk of [§5.3](#53-sequential-path-delete-the-copy) for free.

`NativeLoader` does *not* get this for free: its non-shuffled branch bulk-memcpys (dataloader.mojo:340-359) even when the source is contiguous and sequential.

### 2.5 The existing GPU gather allocates per call

`GatherKernel.gather_gpu` (gather_kernel.mojo:189-225) allocates **two** device buffers and does an element-wise index upload on *every* invocation:

```mojo
        var idx_dev = ctx.enqueue_create_buffer[Self.index_dtype](n_indices)
        with idx_dev.map_to_host() as host_idx:
            for k in range(n_indices):
                host_idx[k] = Scalar[Self.index_dtype](indices[k])
        ...
        var out_dev = ctx.enqueue_create_buffer[datatype](total_output)
```

For MNIST at batch 64 that is 938 × 2 = 1,876 device allocations per epoch, 28,140 over the run — plus 14,070 host→device index sweeps.

The index *volume* is trivial: 64 × 8 B = 512 B per batch, **7.2 MB across the entire 15-epoch run**. The cost is not bandwidth; it is the allocation churn and the loss of the persistent-buffer contract at dataloader.mojo:809. This distinction drives the whole design in [§5.4](#54-shuffled-path-gather_into--with-a-device-resident-permutation).

### 2.6 The Python surface cannot express any of this

Verified in `python-binding/tenmo.py`:

- `DataLoader.__init__` (tenmo.py:1611-1620) accepts only `features, labels, batch_size, shuffle, drop_last, transform`. **No `device`.**
- The Python `Tensor` exposes `device` as a **read-only property** (tenmo.py:679-682) and has **no `to_gpu` / `to_cpu` / `to_device`**:

```python
    @property
    def device(self) -> str:
        """Return the device string ('cpu' or 'cuda:0')."""
        return str(self._raw.device())
```

So a Python user cannot even move a tensor to the GPU, let alone ask a loader to batch it there. Mojo-side work alone leaves the binding broken.

> **Partially landed (`e512431`):** `DataLoader.to_gpu` / `to_cpu` / `device`
> (tenmo.py) now exist; the `Tensor`-level `to_gpu`/`to_cpu` move methods and
> the `device=`/`residency=` ctor sugar are still missing (deferred by design, §4.5).

`_LOADER_PAIRS` (tenmo.py:1593-1596) is compile-time dtype dispatch, not device dispatch:

```python
_LOADER_PAIRS = {
    ("float32", "int64"): _tenmo.DataLoader,
    ("float32", "float32"): _tenmo.DataLoaderProb,
}
```

Device is a *runtime* property, so no new pair registrations are needed — but the `device` argument must be plumbed through `_init_data_loader` in `tenmo_bind.mojo`.

---

## 3. The Central Design Question

Before any code: **what does "device-resident" mean for the loader?** Three coherent models, and they are not equally supportable.

**Model A — Whole-dataset residency.** The entire dataset lives on the device; the loader gathers/slices in place. Per-batch cost is one kernel launch, zero transfers. Requires dataset ≤ device memory. This is what `mnist_pytorch_optimized.py` does.

**Model B — Per-batch residency (today).** The dataset stays on the host; each batch is copied per iteration. Cost is one H2D transfer per batch. Unbounded dataset size, but the transfer is on the critical path 14,070 times.

**Model C — Pinned-host staging.** Host data in page-locked memory, batch copies issued async on a separate stream, overlapped with compute. The standard PyTorch remedy for Model B's latency problem. Requires pinned allocation, a second stream, and event-based ordering — none of which exist in `tenmo/gpu/`.

**Decision: implement A as the target, keep B as the default for out-of-core datasets, and do not build C.** The gate between A and B is a memory-capacity check that must be *explicit and visible*, because silently choosing is how you get an unexplained 188 MB allocation. `into_loader(device=)` therefore does **not** silently mean "copy everything to the GPU" — that gets its own named parameter (below), because conflating "put my batches on the GPU" with "hold my whole dataset in VRAM" is exactly the kind of default that OOMs on someone else's data.

---

## 4. Sequencing: Which Loader First

Both loaders are amenable to every change in [§5](#5-design). They are **not** equally amenable, and the asymmetry is structural rather than incidental — so the order matters.

### 4.1 Both are amenable — the same five changes apply to each

| Change | `NativeLoader` | `DataLoader` |
|---|---|---|
| `Dataset.device()` on the trait | **Required** — trait-generic, must query `DatasetSource` | **Not needed** — holds `Tensor` by value, `self.features.device()` is direct |
| `device=` on batch-buffer allocation | dataloader.mojo:204-209, 235-240 | dataloader.mojo:933-936, 951-956, 985-1015 |
| Sequential → view slices | dataloader.mojo:339-359 (convert) | **Already done** — dataloader.mojo:1085-1098 |
| Shuffled → `gather_into` | dataloader.mojo:362-381 | dataloader.mojo:1110-1148 |
| Hoist normalization out | **Only consumer** — dataloader.mojo:384-407 | Never had it |
| Python binding | Not bound ("not directly Python-bound", dataloader.mojo:100) | `_LOADER_PAIRS`, python-binding/tenmo.py:1593 |

Neither is blocked by the other. `DataLoader`'s concrete `sample_dtype`/`label_dtype` parameters and by-value tensor storage make it the *easier* of the two to convert — every change is local to one struct, with no trait surface. `NativeLoader` additionally needs the trait surgery and the pointer guards.

### 4.2 But all the value is in `NativeLoader`

Verified by usage, not assumed:

- **All four GPU examples use `into_loader`, i.e. `NativeLoader`** — `mnist_gpu.mojo`, `mnist_conv2d_gpu.mojo`, `mnist_conv_tt_gpu.mojo`, `mnist_gpu_prof.mojo`. `DataLoader` appears in exactly one example (`mnist_native.mojo`, CPU-only).
- **All 11 `normalize_mean=` / `normalize_std=` call sites** are `into_loader` (NativeLoader) call sites: `mnist.mojo`, `mnist_adamw.mojo`, `mnist_conv2d.mojo`, `mnist_conv2d_gpu.mojo`, `mnist_conv_tt_gpu.mojo`, `mnist_gelu.mojo`, `mnist_gpu.mojo`, `mnist_gpu_prof.mojo`, `mnist_mixed.mojo`, `mnist_mixed_dtypes.mojo`, `mnist_quant.mojo`, `mnist_unified.mojo`.
- **`DataLoader` has one example and one Python test suite**; `NativeLoader` has three examples.

Converting `DataLoader` first would leave every GPU training path in the repo — including the flagship `mnist_gpu.mojo` this work is motivated by — still doing 14,070 per-batch transfers. It would optimize the loader nobody uses on a GPU.

### 4.3 Recommendation: `NativeLoader` first, with the shared kernel pulled ahead of both

The shared kernel (`gather_rows_2d_into` + the device-resident permutation, [§5.4](#54-shuffled-path-gather_into--with-a-device-resident-permutation)) is the only genuinely new code and is consumed by both. Sequence it as its own unit **before** either loader:

> **Phase 5a — kernel, standalone.** Add `GatherKernel.gather_rows_2d_into` and test it directly against `Tensor.gather` as the oracle. No loader touched.
> **Phase 5b — `NativeLoader`.** Consume the kernel; add `Dataset.to_gpu`; hoist normalization.
> **Phase 5c — `DataLoader`.** Consume the same kernel; derive the device from the source tensors (no `device=` param — see §4.5).

Why this order:

1. **The kernel is where the uncertainty is.** It is the only new GPU code, and the only part that cannot be validated by the existing CPU test suite. Landing and testing it standalone means the loaders become pure wiring.
2. **`NativeLoader` is the harder consumer, so it should drive the kernel's design.** It needs both halves — the kernel *and* a device-resident permutation — and it is the one with the normalization interaction. `DataLoader` needs only the kernel. Designing against `DataLoader` first risks building the kernel to the easier spec and then retrofitting permutation management.
3. **The trait's `device()` contract gets proven against the harder case.** If `DataLoader` goes first, `device()` is never exercised through the trait at all (DataLoader reads `self.features.device()` directly), and the defaulted trait method ships untested against its only real consumer. Doing `NativeLoader` first means the default and the override are both validated by the same test run.
4. **Normalization removal is `NativeLoader`-only and is the largest mechanical change** (11 files). [§9](#9-call-site-migration) notes this is *not* a pure signature deletion — each site needs an equivalent prep step. Doing it while the diff is otherwise small, and before the `DataLoader` changes land, keeps the numerics review separable.

The counter-argument, stated fairly: **`DataLoader` is the better test oracle.** Its sequential path already works on device ([§2.4](#24-what-already-works--the-good-news)), and `tests/python/test_data_loader.py` gives an end-to-end GPU check with no new harness. That is exactly why Phase 5a validates the kernel *before* either loader — the oracle is available without making it the first thing converted.

### 4.4 One asymmetry the conversion introduces

`DataLoader` carries two pieces of bookkeeping `NativeLoader` lacks: `_buffers_owned` (dataloader.mojo:837) and `_make_buffers` (dataloader.mojo:959-1017), used by `set_shuffle` (dataloader.mojo:1162-1175) to rebuild gather buffers after the sequential path has replaced them with views (dataloader.mojo:1089 sets `_buffers_owned = False`).

Converting `NativeLoader`'s sequential path to view slices ([§5.3](#53-sequential-path-delete-the-copy)) makes its batch buffers hold **aliases of the source dataset**, not owned storage. `NativeLoader` fixes `shuffle` at construction and has no `set_shuffle`, so nothing can flip modes today and there is no immediate bug — but the invariant is now implicit and unrecorded.

**Action:** when converting, either port `_buffers_owned` to `NativeLoader` for parity, or state in the docstring that a sequential `NativeLoader` batch aliases the dataset and the loader is single-mode. Do not leave it undocumented — this is the class of bug that appears later when someone adds `set_shuffle` for parity and gets stale view buffers back.

### 4.5 `DataLoader` derives its device — no `device=` param

Decision (agreed 2026-10-04): `DataLoader` reads the device from its source tensors instead of taking a `device=` constructor parameter. Rationale:

1. **One source of truth.** A stored device duplicates reality (`"I asked for cuda:0"` vs `"sources are on CPU"`), and every buffer-replacing path (`__init__`, `_make_buffers`, copy-init) must reconcile the two. Deriving from `self.features.device()` makes the `set_shuffle` rebuild-wrong-device bug from §4.4 unexpressible.
2. **Transfers stay visible.** `loader.to_gpu(gpu)` puts the one big copy at the call site, mirroring `NativeLoader`'s `dataset.to_gpu(gpu)` pattern and `mnist_gpu.mojo`'s `to_gpu(gpu).normalized()`. A ctor that silently moves gigabytes hides cost inside what looks like cheap construction.
3. **The silent-CPU footgun is cheap to fix.** `__init__` panics if `features.device() != labels.device()`; pure-CPU-in means CPU-loader, which is a legitimate configuration (all local tests run that way).

Resulting 5c shape:

- `__init__`: unchanged signature. Normalize sources as today, then assert device match and allocate `_batch`/`_last_batch` on the sources' device via `Tensor.zeros(..., device=)`.
- `to_gpu(gpu)` / `to_cpu()`: move both sources, rebuild both buffers on-device. This is also the entire Python story — the binding exposes the same method, so no ctor plumbing through `register_data_loader`. A `device="cuda:0"` Python kwarg is deferred as sugar over construct-then-move.
- `_fill_batch`: device branch via `gather_rows_2d_into` (5a kernel); host branch keeps `unsafe_memcpy`.
- Tests: CPU↔GPU batch parity plus buffer-device assertions across `set_shuffle` toggles and `to_gpu`→`to_cpu` round trips.

---

## 5. Design

### 5.1 Extend the trait with `device()` — do not widen the pointer contract

The pointer methods must **stay**: they are the storage contract for `SlidingWindowDataset` and its `WindowLoader`, which deliberately avoid the `Dataset` trait for exactly this reason (dataloader.mojo:1210-1212 — "Deliberately NOT a `Dataset`-trait conformer — the trait's flat-pointer contract is what forces the legacy O(N·T) layout"). Replacing them would cascade into `LLMDataset`, `RandomSlidingWindowDataset`, `spiral.mojo`, and ~40 call sites in `tests/test_data.mojo`.

Instead, add one defaulted method to the trait (dataloader.mojo:42-92):

```mojo
trait Dataset(Sized & Copyable):
    ...
    def device(ref self) -> Device:
        """Device the bulk data lives on. Defaults to CPU.

        Overridden by conformers that hold device-resident tensors. Loaders
        allocate their batch buffers here and dispatch their gather here.
        """
        return CPU().into()
```

A **defaulted** method keyed on `Device` means every existing conformer keeps compiling and keeps its current behavior. `NumpyDataset` and `TensorDataset` override it in one line each:

```mojo
    def device(ref self) -> Device:
        return self._features.device()
```

Separately, make the pointer accessors fail loudly rather than dangling, which closes the unsafe-failure-mode gap from [§2.2](#22-a-gpu-tensor-has-no-host-pointer-to-return):

```mojo
    def get_features_ptr(
        ref self,
    ) -> Pointer[Scalar[Self.sample_dtype], ImmutAnyOrigin]:
        if self.device().is_gpu():
            panic(
                "NumpyDataset.get_features_ptr: bulk data is device-resident;"
                " the flat host pointer contract does not apply. Use the"
                " loader's tensor-level path (into_loader) or .to_cpu() first."
            )
        return self._features.data_ptr().as_imm()
```

### 5.2 Datasets get `to_gpu()`, mirroring `Tensor` and `Module`

Naming is not free: `Tensor.to_gpu(...)` (tensor.mojo:772) and `Module.to_gpu` (net.mojo:960; `Sequential.to_gpu` at net.mojo:1099) already own this verb. Datasets should match, and the signature should be a mirror of the `Tensor` one:

```mojo
    def to_gpu(
        ref self, gpu: Optional[GPU] = None, sync: Bool = True
    ) raises -> Self:
        """Return a dataset whose bulk data is resident on `gpu`.

        The returned dataset owns its tensors; the receiver is unchanged.
        """
        var target = gpu.or_else(GPU())
        var f = self._features.to_gpu(target, sync=sync)
        var y = self._labels.to_gpu(target, sync=sync)
        return Self(f, y)

    def to_cpu(ref self, sync: Bool = True) raises -> Self:
        return Self(self._features.to_cpu(sync=sync), self._labels.to_cpu(sync=sync))
```

Two details that matter:

- **`ref self`, not `mut self`, and return `Self`.** A moved dataset is a *value*, so lifetime is the ordinary Mojo rule. This is exactly why [§5.5](#55-into_loaderdevice-as-sugar-not-the-primitive) makes `to_gpu` the primitive: `into_loader` stores `Pointer(to=self)` with `origin_of(self)` (dataloader.mojo:96, 584-591), so a loader over a moved dataset is safe as long as the moved dataset is a named local that outlives the loader. Keeping the move explicit means the user can see that lifetime in their own code.
- **The pre-normalized tensors move with it.** This is what lets normalization leave the loader ([§5.6](#56-normalization-must-leave-the-loader)) — you normalize on CPU once, then move, rather than maintaining a CPU and a GPU copy of the normalized data.

`__getitem__` / `sample` (dataloader.mojo:532-535, 717-720, 1336-1337) allocate sample tensors with a bare `Tensor.zeros(...)`. Once device-aware, each becomes:

```mojo
        var sample_feature = Tensor[Self.dtype].zeros(
            self._feature_shape, device=self._features.device()
        )
```

(rarely hit — the loaders do not use `__getitem__` — but it is the same bug and belongs in the same change.)

### 5.3 Sequential path: delete the copy

The `DataLoader` sequential path already works on device ([§2.4](#24-what-already-works--the-good-news)). `NativeLoader` needs to be converted to match, and the conversion is a **deletion**, not an addition. Replace dataloader.mojo:339-359:

```mojo
        # Bulk copy if not shuffled
        if not self.shuffle_data:
            var first_sample_idx = self._indices[start_idx]
            ...
```

with a view slice, mirroring `DataLoader.__next__` (dataloader.mojo:1085-1098):

```mojo
        if not self.shuffle_data:
            # Sequential (eval): zero-copy view slices of the source tensor.
            batch.features = self._features.slice(
                start=start_idx, end=start_idx + actual_batch_size, step=1, axis=0
            )
            batch.labels = self._labels.slice(
                start=start_idx, end=start_idx + actual_batch_size, step=1, axis=0
            )
            return
```

This deletes the two `unsafe_memcpy` calls, both `get_*_ptr()` uses in the hot path, and the bulk-copy branch — a net reduction in code that also makes eval free on both devices.

One contract consequence: a sequential batch is now an **alias of the source**, valid indefinitely, instead of a private copy valid until the next `__next__()`. That asymmetry already exists in `DataLoader` and is documented at dataloader.mojo:813-819. `NativeLoader`'s docstring must be updated to match, and `Batch` consumers must be read-only (they already are — see `epochs.mojo`, which clones off the gather buffer).

### 5.4 Shuffled path: `gather_into` with a device-resident permutation

This is the only genuinely new kernel work. Three layers.

**Layer 1 — an out-variant row-gather kernel.** `gather_rows_2d_kernel` (gather_kernel.mojo:66-99) already has the right shape; it needs to write into a caller-supplied buffer at a caller-supplied index offset. Add to `GatherKernel`:

```mojo
    @staticmethod
    def gather_rows_2d_into(
        out_dev: DeviceBuffer[datatype],
        in_dev: DeviceBuffer[datatype],
        in_rows: Int, in_cols: Int, in_row_stride: Int,
        idx_dev: DeviceBuffer[index_dtype], idx_offset: Int,
        out_rows: Int, out_row_stride: Int,
        ctx: DeviceContext, sync: Bool = False,
    ) raises
```

launching `gather_rows_2d_kernel` with `indices_buffer[row + idx_offset]`. Launch config is already right: `grid_dim=out_rows`, `block_dim=_gather_2d_block_cols(in_cols)`.

**Layer 2 — a device-resident permutation, uploaded once per epoch.** The loader keeps the epoch permutation as a device `DeviceBuffer[int64]` instead of a host `List[Int]`, and the batch's indices are a *sub-buffer view*:

```mojo
var idx_sub = self._perm_dev.create_sub_buffer[index_dtype](start, bs)
```

`create_sub_buffer[dtype](start, count)` zero-copy views are established practice
(`contiguous_device_state` uses one at its view offset before the D2D copy).
That precedent removes the need for an `idx_offset` kernel parameter entirely — the sub-buffer's pointer already points at `start`. **Per batch: zero allocations, zero transfers.** Compare [§2.5](#25-the-existing-gpu-gather-allocates-per-call)'s 2 allocations + 1 upload per batch.

> Note: this section once cited `to_dtype`'s same-dtype path as the precedent;
> that path now routes through `CastKernel` (fresh buffer, no aliasing) — the
> `contiguous_device_state` site above is the surviving precedent. (The
> sub-buffer permutation sketched here was itself never implemented — the
> landed `gather_rows_2d_into` takes `IntArray` indices; see §12.)

Note that the permutation must be re-uploaded on `reset()`/`__iter__()` (dataloader.mojo:418-423, 1056-1060, 1156-1160) — 0.5 MB per epoch for MNIST, 7.2 MB over the run. Keep `std.random.shuffle` host-side; it is microseconds and keeps the loader host-light. This is a deliberate divergence from `torch.randperm(n, device=device)` (mnist_pytorch_optimized.py:62), and it is the right one: 0.5 MB/epoch is not where the time goes.

**Layer 3 — loader wiring.** `_fill_batch` becomes:

```mojo
    def _fill_batch(self, batch: Batch[...], start: Int, bs: Int) raises:
        var idx_sub = self._perm_dev.create_sub_buffer[index_dtype](start, bs)
        GatherKernel.gather_rows_2d_into(
            batch.features.buffer.device_state.value().buffer,   # preallocated
            self._features.buffer.device_state.value().buffer,
            self._features.shape()[0], self._features_per_sample, ...,
            idx_sub, bs, self._features_per_sample,
            ctx, sync=False,
        )
        # same for labels
```

and the preallocated batch buffers gain `device=self.features.device()` (dataloader.mojo:933-936, 951-956, and 985-1015 for `_make_buffers`). This is what actually preserves the dataloader.mojo:809 contract: *"per batch: zero allocations (persistent buffers reused)"*.

**Dispatch.** `_fill_batch` branches once, at the top:

```mojo
        if self.device().is_gpu():
            Self._fill_batch_device(batch, start, bs)
        else:
            Self._fill_batch_host(batch, start, bs)   # today's unsafe_memcpy loop
```

Host behavior stays byte-for-byte identical, so the CPU path and its ~40 tests are untouched. This also means `Gather` (gather.mojo) does **not** need to change — the loader uses the kernel directly rather than the autograd-tracked `Gather.forward` op. Batches are read-only training inputs; there is nothing to differentiate.

### 5.5 `into_loader(device=)` as sugar, not the primitive

> **Not landed as proposed.** All `into_loader` overloads still take only
> `(batch_size, shuffle, drop_last)`; the `device=`/`residency=` knobs below
> were not implemented. What landed instead is the §4.5 derive-from-source
> pattern everywhere: move the dataset (`dataset.to_gpu(gpu)` /
> `DataLoader.to_gpu(gpu)`), then build the loader with no new argument.

```mojo
    def into_loader(
        ref self,
        batch_size: Int,
        shuffle: Bool = True,
        drop_last: Bool = False,
        device: Optional[Device] = None,
        residency: Residency = Residency.INFER,
    ) -> NativeLoader[Self, origin_of(self)]:
```

with

```mojo
struct Residency:
    """How much of the dataset to place on `device`."""
    var INFER = Residency(0)   # follow the source tensors (default)
    var FULL   = Residency(1)   # move the whole dataset (Model A)
    var NONE   = Residency(2)   # keep on host, transfer per batch (Model B)
```

Two knobs rather than one, because they answer different questions:

- **`device=`** — *where should batches be produced?* This is the common case and the reason someone reached for this feature.
- **`residency=`** — *how much data should live there?* `FULL` is the PyTorch-style upfront move. `NONE` with `device=gpu` is Model B. Defaulting `device=` to `FULL` would mean "pass `device=cuda`" silently allocates your entire dataset in VRAM; making it explicit is the [§3](#3-the-central-design-question) gate, made visible.

`FULL` should refuse rather than OOM:

```mojo
            if self._features.num_elements() * size_of[Scalar[Self.sample_dtype]]() > free_bytes:
                panic(
                    "into_loader(residency=FULL): dataset does not fit in"
                    " device memory. Use residency=NONE for per-batch"
                    " transfer, or .to_cpu() to page it out."
                )
```

`INFER` is the default so the documented pattern — move the dataset, then build the loader — needs **no new argument at all**:

```mojo
var train_dataset = TensorDataset[FEATURE_DTYPE, LABEL_DTYPE](
    X_train.to_gpu(gpu), y_train.to_gpu(gpu)     # or train_dataset.to_gpu(gpu)
)
var train_loader = train_dataset.into_loader(batch_size=64, shuffle=True)
```

### 5.6 Normalization must leave the loader

`NativeLoader._fill_batch` (dataloader.mojo:384-407) applies `(x - mean) * inv_std` through a raw SIMD loop over a host pointer. There is no GPU equivalent of "poke a host pointer", and adding one would be the wrong abstraction anyway — it re-normalizes the same row on every epoch it appears in.

`mnist_pytorch_optimized.py:17-21` normalizes once, upfront, in place. Match it:

- **Delete** `normalize_mean` / `normalize_std` from `NativeLoader` and `into_loader` (dataloader.mojo:83-84, 155-160, 384-407, 581-582). They are on ~15 example call sites, all mechanical.
- **Add** normalization to the data-prep step in the GPU examples, exactly as `prep()` does.
- Keep the *constants* (`MNIST_MEAN`, `MNIST_STD`, and the Fashion/CIFAR/ImageNet sets at dataloader.mojo:11-24) — they are the vocabulary for writing the prep step.

This is a **behavior-preserving** change for numerics: today the loader applies `(x/255 - mean)/std` per batch, so hoisting it to prep produces bit-identical inputs. The one thing to preserve is float ordering — `(x/255.0 - 0.1307)/0.3081` as written in mnist_native.mojo:71-74, not `x/255.0/0.3081 - 0.1307/0.3081`, if exact parity with the current loader matters.

`DataLoader` never had normalization, so it is unaffected — it is already the "pre-normalize, then load" engine.

### 5.7 What stays host-side, deliberately

| State | Location | Why |
|---|---|---|
| `_indices` (epoch permutation) | host `List[Int]` | reshuffle is µs; device upload is 0.5 MB/epoch |
| `__len__`, `__has_next__`, batch shapes | host scalars | control flow |
| `Batch.batch_size` | host `Int` | already is (dataloader.mojo:31) |
| Dataset metadata (shapes, per-sample counts) | host | pure metadata |

Only bulk sample data and the permutation cross to the device.

### 5.8 Python binding

```python
class Tensor:
    def to(self, device: str) -> "Tensor": ...      # 'cpu' | 'cuda:0'
    def to_gpu(self) -> "Tensor": ...
    def to_cpu(self) -> "Tensor": ...

class DataLoader:
    def __init__(self, features, labels, *, batch_size=64, shuffle=True,
                 drop_last=False, transform=None,
                 device: str | None = None,
                 residency: str = "infer"): ...
```

`device`/`residency` are runtime strings, so `_LOADER_PAIRS` (tenmo.py:1593-1596) needs no new entries — but `tenmo_bind.mojo`'s `_init_data_loader` must forward them. Precedent for adding a string-valued kwarg to a bound type already exists: `CrossEntropyLoss(reduction: String)` (crossentropy.mojo:1311-1322).

The `Tensor.to_gpu` / `to_cpu` bindings are the harder half and are worth doing regardless of this feature — the Python API currently cannot move a tensor to a GPU at all ([§2.6](#26-the-python-surface-cannot-express-any-of-this)).

---

## 6. The Sync Budget After the Change

This is the part that decides whether the change actually buys anything. Device-resident data removes the *transfers*, but two per-batch **synchronizations** remain, and they will dominate a small model:

| Op | Location | Cost |
|---|---|---|
| `loss.item()` | mnist_gpu.mojo:187, 210 | full device sync, returns host `Float32` |
| `Accuracy.compute(..., sync=True)` | mnist_gpu.mojo:189, 212 | full device sync, returns host `Float64` |

Both are inherent to their current signatures. `Tensor.item()` → `NDBuffer.item()` → `get(0)` → host read (tensor.mojo:698, ndbuffer.mojo:922). `Accuracy.compute` is declared `-> raises -> Float64` (accuracy.mojo:13), a host scalar; its GPU kernel launches and then reads the result back (accuracy_kernel.mojo:126-132).

To reach PyTorch's one-sync-per-epoch (mnist_pytorch_optimized.py:78-79), two additions are needed. Both are feasible with what already exists:

**Device-resident loss accumulator.** `CrossEntropyLoss` already supports `reduction="none"`, returning the per-sample vector (crossentropy.mojo:269-279, constructor at crossentropy.mojo:1311-1322):

```mojo
var per_sample = criterion_none(pred, batch.labels)   # (B,) on device
loss_acc = loss_acc + per_sample.sum()                 # stays on device
```

then read `loss_acc.item()` **once per epoch**.

**Device-resident accuracy accumulator.** `Accuracy.compute` needs an out-variant sibling, since it hardcodes the `Float64` return:

```mojo
    @staticmethod
    def compute_into(
        pred: Tensor[Self.dtype], target: Tensor[Self.index_dtype],
        acc: Tensor[DType.int64],
    ) raises -> Tensor[DType.int64]: ...
```

adding into `acc` on device, read back once per epoch.

**This is out of scope for the loader** — it touches `Accuracy` and the training loops, not `dataloader.mojo`. But it is the remaining half of "match the optimized PyTorch script," and shipping device-resident data without it will show a much smaller speedup than the PyTorch comparison implies. Sequence it explicitly: land data residency first (it is a strict improvement on its own), then the accumulators.

---

## 7. Alternatives Considered and Rejected

**Reuse `Tensor.gather` per batch.** Already GPU-capable (gather.mojo:461-487) and needs zero new kernels. Rejected as the fast path because it allocates two device buffers and does a host→device index upload per call ([§2.5](#25-the-existing-gpu-gather-allocates-per-call)), violating the documented persistent-buffer contract (dataloader.mojo:809). **Retained as the reference oracle** the new `gather_into` path is tested against — for `rank==2, axis==0` the two must agree element-for-element.

**Pre-gather the whole epoch into a permuted device buffer, then slice sequentially.** One gather per epoch instead of per batch; the sequential path is then free. Rejected as the default: it needs a second full copy of the dataset (another 188 MB for MNIST train) and its epoch barrier serializes the permutation against compute. Worth an opt-in flag for datasets that comfortably fit — the per-batch launch is ~5 µs and the gather is ~200 KB, so for small batches the launch overhead may dominate. **Measure before adding the flag.**

**Upload the permutation as a `Tensor[int64]` and use `Tensor.gather` with a tensor index.** Reuses the multi-dim `Gather.forward` overload (gather.mojo:234-305). Rejected: it routes through `IntArray` + `reshape` anyway (gather.mojo:279-305) and inherits the same per-call allocation.

**Make `Dataset.get_*_ptr()` return a `DeviceBuffer` instead of a `Pointer`.** Maximum generality. Rejected: it breaks the `Dataset` contract for `LLMDataset` / `RandomSlidingWindowDataset`, which are host-only by construction (dataloader.mojo:1191-1197), and it would force every conformer to be device-capable to satisfy the trait.

**Pinned-host staging (Model C).** Rejected for now: no pinned-allocation API, no second stream, and no event-based ordering in `tenmo/gpu/`. It is the right answer for datasets that do not fit in VRAM, and it should be a follow-on design, not a retrofit into this one.

---

## 8. Open Questions and Risks

**Q1 — Does `map_to_host` synchronize?** *Unverified; MAX sources are not in this repo.* Evidence is mixed:

- `accuracy_kernel.mojo:126-132` synchronizes **explicitly** immediately before mapping — which suggests `map_to_host` does *not* sync on its own.
- `optim.mojo:322-324` maps a gradient device buffer to host with **no** preceding synchronize, while the enclosing SGD step has kernels in flight — which suggests it *does*, or that this is a latent staleness bug.

**Resolution:** design so the question does not matter. [§5.4](#54-shuffled-path-gather_into--with-a-device-resident-permutation) keeps the permutation device-resident precisely so the hot path never calls `map_to_host`. **Action:** measure with a 1000-iteration microbenchmark before and after; if it turns out `map_to_host` does sync, that also explains part of the current per-batch cost and should be recorded against `Tensor.gather`.

**Q2 — `Tensor.contiguous()` copies the whole dataset on GPU.** `DataLoader.__init__` (dataloader.mojo:871-881) calls `contiguous()` when the source `requires_grad` or is non-contiguous. On GPU that is **not** a no-op even for contiguous input — `NDBuffer.contiguous` always materializes a fresh independent `DeviceState` (ndbuffer.mojo:1802-1825), and the contiguous fast path is a full device-to-device `enqueue_copy_to` (ndbuffer.mojo:1747-1770). A non-contiguous GPU source additionally round-trips through the host (gpu/transfer.mojo:117-155).

So a user who passes a *strided* GPU tensor pays a full-dataset copy — and possibly a host round-trip — before the loader ever starts. **Action:** guard the call so an already-contiguous, grad-free source is adopted as-is (the loader holds it by value, so the "must own contiguous storage" invariant at dataloader.mojo:871-873 is about batching correctness, not aliasing), and log a warning when a real materialization happens.

**Q3 — Sub-buffer lifetime.** [§5.4](#54-shuffled-path-gather_into--with-a-device-resident-permutation) relies on the epoch permutation `DeviceBuffer` outliving every batch sub-buffer taken from it. Since the loader owns the permutation for the loader's lifetime and the loader outlives its batches, this holds — **but** it must be asserted, not assumed, because `Batch` is documented as aliasing loader-owned buffers (dataloader.mojo:813-819). Add a debug assertion in the sub-buffer path.

**Q4 — `DType.bool` labels.** `DeviceState` stores `bool` as `uint8` (gpu/transfer.mojo:44, 142) and every `map_to_host`/`materialize_contiguous` site branches on it. A new kernel must either reject `bool` or handle the storage cast. MNIST labels are `int64` so this is not on the critical path, but `DataLoaderProb` (float32/float32) and any boolean-mask loader will hit it.

**Q5 — Multi-GPU.** The proposed `into_loader(device=)` would take a `Device`, and `to_device` already handles GPU→different-GPU by materializing through the host (ndbuffer.mojo:567-585). Multi-GPU *training* (DDP-style) is out of scope; multi-GPU *inference over a loader* should work but is untested.

**Q6 — The `epochs.mojo` clone pattern.** `tenmo/epochs.mojo` documents that it "clone[s] batches off the loader's persistent gather buffers before use." On GPU that clone is a full device allocation per batch (tensor.mojo `clone`), reintroducing exactly the churn [§5.4](#54-shuffled-path-gather_into--with-a-device-resident-permutation) removes. **Action:** re-examine whether the clone is still necessary once the gather buffer is device-resident and the consumer is read-only; if it is not, remove it and update the docstring.

---

## 9. Call-Site Migration

> **Landed (`4441e93`, `345dee9`).** The deletions below are done;
> `normalize_mean`/`normalize_std` no longer exist and `mnist_gpu.mojo`
> already uses resident batches. Keep this section as the migration record.

`examples/mnist_gpu.mojo`, the reference GPU example (pre-migration shape):

```mojo
-    # Normalize to [0, 1]
-    X_train = X_train / 255.0
-    X_test = X_test / 255.0
+    # Normalize ONCE, upfront — the loader no longer normalizes per batch.
+    # Ordering matches the removed loader SIMD loop for bit-identical inputs.
+    X_train = (X_train / 255.0 - Float32(MNIST_MEAN)) / Float32(MNIST_STD)
+    X_test = (X_test / 255.0 - Float32(MNIST_MEAN)) / Float32(MNIST_STD)

+    # One-time device residency. Must precede into_loader so the loader
+    # allocates its batch buffers on the device too.
+    var gpu = GPU()
+    X_train = X_train.to_gpu(gpu, sync=True, stop_grad=True)
+    y_train = y_train.to_gpu(gpu, sync=True, stop_grad=True)
+    X_test = X_test.to_gpu(gpu, sync=True, stop_grad=True)
+    y_test = y_test.to_gpu(gpu, sync=True, stop_grad=True)

     var train_dataset = NumpyDataset[FEATURE_DTYPE, LABEL_DTYPE](X_train, y_train)
     var test_dataset = NumpyDataset[FEATURE_DTYPE, LABEL_DTYPE](X_test, y_test)

     var train_loader = train_dataset.into_loader(
         batch_size=train_batch_size,
         shuffle=True,
         drop_last=False,
-        normalize_mean=Float32(MNIST_MEAN),
-        normalize_std=Float32(MNIST_STD),
     )
     var test_loader = test_dataset.into_loader(
         batch_size=test_batch_size,
         shuffle=False,
         drop_last=False,
-        normalize_mean=Float32(MNIST_MEAN),
-        normalize_std=Float32(MNIST_STD),
     )

-    print("Transferring model parameters to GPU...")
-    var gpu = GPU()
     model = model.to_gpu(gpu, stop_grad=True)
```

and in **both** loops (mnist_gpu.mojo:176-177 and :204-205):

```mojo
-            var features_gpu = batch.features.to_gpu(gpu, sync=False)
-            var labels_gpu = batch.labels.to_gpu(gpu, sync=False)
-
-            var pred = model(features_gpu)
-            var loss = criterion(pred, labels_gpu)
+            # Batches are already device-resident; no per-batch transfer.
+            var pred = model(batch.features)
+            var loss = criterion(pred, batch.labels)
```

with `Accuracy.compute(pred, labels_gpu, sync=True)` → `Accuracy.compute(pred, batch.labels, sync=True)` at mnist_gpu.mojo:189 and :212.

Unchanged: the `has_accelerator()` guard (mnist_gpu.mojo:23-26), `model.to_gpu(gpu, stop_grad=True)` (:132), and `model.to_cpu()` (:251).

**Call sites needing the mechanical `normalize_*` removal** (all `into_loader` callers passing both kwargs — since removed in `4441e93`): `mnist_gpu.mojo`, `mnist_conv2d_gpu.mojo`, `mnist_conv_tt_gpu.mojo`, `mnist_mixed.mojo`, `mnist_quant.mojo`, `mnist_gelu.mojo`, `mnist_conv2d.mojo`, `mnist_adamw.mojo`, `mnist_mixed_dtypes.mojo`, `mnist.mojo`, `cifar_10.mojo`, `sort_sequence.mojo`, `reverse_sequence.mojo`, `spiral.mojo`, plus `tests/test_data.mojo`. Each must gain the equivalent prep-step normalization or its numerics change — **do not treat this as a pure signature deletion.**

---

## 10. Test Strategy

`tests/test_data.mojo` has ~90+ loader tests (was ~40 when written; 94 defs at refresh), all CPU initially, several asserting the pointer and persistent-buffer contracts directly. Those must stay green unchanged — they are the regression net for the host path.

New coverage, gated on `comptime if has_accelerator():` and following the repo's GPU test conventions (`scripts/run_gpu_tests.sh`, `scripts/gpu_test_files.txt`):

1. **Differential test, shuffled.** Same seed, same dataset, host loader vs device loader; assert batches are element-for-element identical. This is the single most valuable test — it catches index-offset bugs, sub-buffer mis-slicing, and kernel argument errors all at once. Run against `DataLoader` first ([§4.3](#43-recommendation-nativeloader-first-with-the-shared-kernel-pulled-ahead-of-both)): its sequential path already works on device, so it is the cheapest oracle for the Phase 5a kernel.
 2. **`gather_into` vs `Tensor.gather`.** Random `(N, C)` for `C` in {1, 32, 64, 128, 512, 784, 1024} to straddle both the `cols <= 512` fast path and the generic rank-dispatch path ([`gather-kernel-double-dispatch.md`](gather-kernel-double-dispatch.md)), with repeated and out-of-order indices. The `cols > 512` 2D gap in `tests/test_gather_gpu.mojo` was closed with the fix (cols-512/513/784 boundary tests).
3. **Allocation-count regression.** Assert the per-batch path issues zero `enqueue_create_buffer` calls — this is the contract at dataloader.mojo:809 and the entire point of the change. Needs a counter in the GPU runtime or a test-only hook. The same mechanism, in `gather_kernel.mojo`, is what makes the double-dispatch fix testable.
4. **Sequential == shuffled-set.** Sequential batches over the full dataset must equal the concatenation of shuffled batches for the same permutation — validates [§5.3](#53-sequential-path-delete-the-copy)'s view-slice conversion, and the `_buffers_owned` invariant in [§4.4](#44-one-asymmetry-the-conversion-introduces).
5. **Partial last batch.** `drop_last=False` with `N % batch_size != 0` on device (MNIST train is 60000 = 937×64 + 32, so this path is live).
6. **`to_gpu` / `to_cpu` round-trip.** Numerical identity after round-trip; receiver unchanged; lifetime of the returned dataset outliving its loader.
 7. **`residency=FULL` capacity guard (proposed; `residency=` not implemented).** If the `FULL` knob is ever added, it must panic cleanly on an oversized request rather than OOM-ing the device.
8. **Dtype matrix.** `(float32, int64)`, `(float32, float32)`, and the `bool`-as-`uint8` storage path (Q4).
 9. **Python.** `DataLoader(device="cuda")` and `Tensor.to_gpu()` parity with the Mojo path, via `tests/python/test_data_loader.py`. (Still open at refresh: `DataLoader.to_gpu/to_cpu` landed, ctor `device=` sugar and `Tensor.to_gpu` did not.)

---

## 11. Rollout

Sequenced so each step is independently valuable and independently testable.

| Phase | Change | Value if we stop here |
|---|---|---|
| **0** | Panic in `get_*_ptr()` on device sources ([§5.1](#51-extend-the-trait-with-device--do-not-widen-the-pointer-contract)) | Turns silent corruption into a loud error. ~10 lines. **Ship first, alone.** |
| **1** | `device()` on the trait; `device=` on all `Tensor.zeros` batch/sample allocations | CPU behavior identical; device sources stop landing in the wrong place |
| **2** | `Dataset.to_gpu` / `to_cpu`; guard `contiguous()` (Q2) | Callers can move data; the documented pattern works |
| **3** | Sequential path → view slices in `NativeLoader` ([§5.3](#53-sequential-path-delete-the-copy)) | Eval becomes free on both devices. Net code deletion |
| **4** | Hoist normalization out of the loader; migrate all call sites ([§5.6](#56-normalization-must-leave-the-loader)) | Removes per-batch recompute; matches PyTorch's `prep()` |
| **5a** | `GatherKernel.gather_rows_2d_into` standalone, tested against `Tensor.gather` ([§5.4](#54-shuffled-path-gather_into--with-a-device-resident-permutation)) | New GPU code landed and validated before anything depends on it |
| **5b** | `NativeLoader` consumes the kernel + device-resident permutation | **This is the phase that removes the 14,070 transfers** — every GPU example is fixed |
| **5c** | `DataLoader` consumes the same kernel | Feature reachable from `examples/mnist_native.mojo` and from Python |
| **6** | Python `device=` plumbing + `Tensor.to_gpu` / `to_cpu` ([§5.8](#58-python-binding)) | Feature reachable from Python |
| **7** | Device-resident loss / accuracy accumulators ([§6](#6-the-sync-budget-after-the-change)) | One sync per epoch. **Separate concern — own PR** |
| **X** | `GatherKernel` double-dispatch fix ([`gather-kernel-double-dispatch.md`](gather-kernel-double-dispatch.md)) | Independent perf fix. **Land before or with 5a** — same function |

Phases 0-4 are prerequisites for 5b and are individually landable. Phase 5 splits because the kernel is shared and the loaders are not interchangeable consumers — see [§4](#4-sequencing-which-loader-first). Phase 7 is orthogonal and should not be bundled.

Phase X is a separate concern with its own document and its own test strategy, but it edits `gather_kernel.mojo` and Phase 5a extends the same dispatch, so sequencing it first avoids two conflicting diffs in one file.

---

## 12. Implementation Status (2026-10-04; refreshed 2026-10-07)

> **Refresh note (2026-10-07):** the three "uncommitted / remaining" items below all landed
> same-day (`30162c4`, `4441e93`, `345dee9`, `e512431`). Tree is clean. Only
> genuinely open item is the Phase X launch-count regression test.

- **Phase X + 5a — committed (`03ea4e8`).** `dispatched_2d_fastpath` flag; `GatherKernel.gather_rows_2d_into` reusing `gather_rows_2d_kernel`; cols-512/513/784 boundary tests + into-vs-`Tensor.gather` oracle. GPU: `test_gather_gpu.mojo` 35–36 passed.
- **Steps 0–2 — committed (`c5aad88`).** Trait `device()` (CPU default), GPU pointer guards, `Dataset.to_gpu`/`to_cpu`, device-aware batch/sample allocations. `test_data.mojo` 87/87 on CPU and GPU.
- **5b device path — committed (`30162c4`).** `get_features`/`get_labels`, `_next_batch_device` (sequential = aliasing view slices; shuffled = in-place `gather_rows_2d_into` via `_fill_batch_device`). Relaxed `gather_rows_2d_into` to flat-row (rank-N, offset-0, contiguous, numel-checked) — takes `IntArray` indices, not the `DeviceBuffer` + sub-buffer permutation sketched in §5.4.
- **Offset defects fixed along the way (same batch).** `ScalarKernel.launch` PATH 1 and `contiguous_device_state` fast path ignored view offsets (read/copied from buffer base); both now read from a sub-buffer at the offset. Regression tests: `test_oop_offset_view_sub_mul_matches_cpu`, `test_contig_gpu_2d_offset_slice_values`.
- **Host norm-tail defect fixed.** `_fill_batch` SIMD remainder started at `total - simd + 1`, double-normalizing the tail whenever `total` was a multiple of `simd_width` (all other repo loops already used the correct `total - total % simd` idiom). Caught by the new `test_nativeloader_device_parity_normalized`. GPU: `test_data.mojo` 91/91, `test_contiguous.mojo` 39/39.
- **Remaining (as of refresh):** launch-count regression test for Phase X. Done since: Phase 5c (`e512431`), `mnist_gpu` resident-batch migration (`345dee9`), normalization hoist (below) — all committed 2026-10-04.
- **View-offset defects in compute kernels — found by parity, fixed, validating.** Strict loss parity (same torch init weights + sequential order) matched batch 0 to 7 digits, then diverged: Mojo epoch means collapsed (0.07/0.0003 vs torch 0.34/0.13) with suspiciously smooth batch losses — the signature of re-served rows. Root cause, same bug class as the loader fixes: `MatmulKernel.launch` built per-batch base offsets from batch coordinates only, dropping `Layout.offset`, so every offset row-slice read from the buffer base (batch 0 always looked right). Fixed by adding the view offset to each base offset (covers forward + `matmul_2d` backward, incl. transposed-view dW — `transpose()` preserves offset). `AccuracyKernel`/`SequenceAccuracyKernel.launch` had the same defect on label views (eval accuracy) — fixed via offset sub-buffers. CE fused kernels are safe (`materialize_contiguous` is layout-aware); `blas_matmul` is CPU-only with loud offset guards. Regression tests: `test_matmul2d_gpu_offset_row_slice` (CPU oracle + hand-checked dot), `test_accuracy_gpu_offset_label_view` (decoy labels: 0.4 pre-fix vs 0.7 post-fix).
- **Follow-up (tracked): audit remaining GPU kernels for dropped view offsets.** Any kernel taking raw `device_buffer()` + manual indexing is suspect (ReLU/Adder/etc. currently only see fresh offset-0 tensors in the MLP path, but conv/NLP paths may differ).
- **Normalization hoisted — committed (`4441e93`; `mnist_gpu` migration `345dee9`; 5c `e512431`).** `normalize_mean/std` deleted from the trait `into_loader`, `NativeLoader` (fields, both fill paths), both tensor-dataset overrides, and both NLP wrappers. New `NumpyDataset`/`TensorDataset.normalized(mean, std) -> Self` (value semantics mirroring `to_gpu`/`to_cpu`; `(x-mean)*inv` via Tensor scalar kernels with `track_grad=False`, wherever the data lives; fresh features, shared labels). All 12 examples migrated (eager `normalized()` before `into_loader`); `mnist_unified.mojo` chains it on the temporary. Python binding applies it eagerly in `_train_epoch`/`_eval_epoch`, so `examples/mnist.py` is untouched. `test_nativeloader_device_parity_normalized` rewritten: bulk spot-checks + host-vs-device bulk match + sequential lockstep through param-free loaders.
- **Known pre-existing failures (not this work):** 45 `test_ip_*` in-place GPU scalar tests fail on device (`test_scalar_gpu.mojo`: 182 passed / 45 failed). Untouched code (`ScalarInplaceKernel`, HEAD-committed tests); pristine-HEAD baseline run on the same box fails the identical 45 (226 run: 181 passed / 45 failed — the +1 delta is the new offset test).
- **Follow-up (tracked, not this phase): fix the `test_ip_*` in-place GPU scalar failures.** All 45 fail in `ScalarInplaceKernel.launch`'s generic path (add/sub/mul/div/reverse-sub + `pow(2)` scalar across 1d/2d/3d/4d/tail/edge shapes; the dedicated `launch_inplace_pow` strided path passes). Suspect the `inplace_scalar_ops` kernel launch configuration (`simd_vectors_per_thread = 2 * simdwidth` chunking vs `launch_config`) rather than any loader work — the loader never calls this kernel. Repro: `pixi run mojo -I . tests/test_scalar_gpu.mojo` on a T4 box, filter `test_ip_`. Owner: whoever owns `scalar_inplace_ops_kernel.mojo`.

---

## Appendix A — Related Defect: `GatherKernel` Double-Dispatch

**Full write-up, fix, and test plan: [`gather-kernel-double-dispatch.md`](gather-kernel-double-dispatch.md).**

Found while tracing the kernel dispatch for this design; it is a pre-existing defect, not a consequence of it, but Phase 5a touches the same function and must inherit the corrected dispatch.

> **Fixed in `03ea4e8` (Phase X).** The snippet below is the pre-fix shape, kept
> for context — current code sets `dispatched_2d_fastpath` and guards the
> generic dispatch with `if rank == r and not dispatched_2d_fastpath`.

`GatherKernel.gather_gpu` enqueued the 2D row-gather fast path **and then fell through** to the generic rank-dispatch kernel — there was no `else` and no early return (pre-fix gather_kernel.mojo:287 onward):

```mojo
        var out_dev = ctx.enqueue_create_buffer[datatype](total_output)

        if rank == 2 and axis == 0 and tensor_layout.shape[1] <= 512:
            ...
            ctx.enqueue_function(
                compiled, out_dev, in_dev, ..., grid_dim=n_indices, block_dim=block_cols,
            )
        # Rank dispatch generated from MAX_RANK — one arm per rank
        comptime for r in range(1, MAX_RANK + 1):
            if rank == r:
                _launch_gather_generic[datatype, r, Self.index_dtype](ctx, out_dev, ...)
```

Both kernels wrote the same `out_dev` with the same values, so **results were correct** — but the "Optimized 2D row-gather fast path" advertised at gather.mojo:5 did 2× the memory traffic it was meant to save, on exactly the inputs it targets.

**Scope:** `rank==2, axis==0, cols <= 512, reduction=NONE`, GPU. MNIST is 784 columns, so it is *not* hit by `mnist_gpu.mojo` — but it is hit by any narrow 2D dataset (`sort_sequence.mojo`, `reverse_sequence.mojo`, IMDb) and by the token-embedding path. `GatherKernel` is reached from `Gather._gather_copy` (gather.mojo:409-418, 467-476), so `Tensor.gather` and `embedding` are both affected.

**Why it survived:** the defect is invisible to value assertions — every test in `tests/test_gather_gpu.mojo` asserts values and keeps passing after a fix. A regression test for it *cannot* be a correctness test; it has to assert launch counts. Pre-fix, all 2D fixtures in that file were also 2-3 columns wide, so the generic rank-2 arm had no coverage at all (cols-512/513/784 boundary tests were added with the fix). The dedicated document works this through, specifies the `dispatched`-flag fix, and specifies a three-layer test strategy (CPU predicate unit test, GPU launch-count regression test, recorded before/after benchmark).

---

## Appendix B — Reference Inventory

> Line numbers below are pre-5b approximations (see header note). Prefer
> symbol search over exact `:line` refs.

**Blockers**
- `tenmo/dataloader.mojo:42-92` — `Dataset` trait, host-pointer contract
- `tenmo/dataloader.mojo:49-59` — `get_features_ptr` / `get_labels_ptr`
- `tenmo/dataloader.mojo:502-510`, `667-675` — conformer pointer impls
- `tenmo/dataloader.mojo:321-407` — `NativeLoader._fill_batch` (memcpy + host SIMD normalize)
- `tenmo/dataloader.mojo:1110-1148` — `DataLoader._fill_batch`
- `tenmo/ndbuffer.mojo:1546-1549` — `data_ptr()` projects host `Buffer`
- `tenmo/ndbuffer.mojo:554-560` — GPU `NDBuffer` has no CPU buffer
- `tenmo/ndbuffer.mojo:605-607` — `is_on_gpu()`

**Already device-clean**
- `tenmo/dataloader.mojo:1085-1098` — sequential `slice()` path
- `tenmo/tensor.mojo:3583-3613` — `slice` → `View.forward`
- `tenmo/ndbuffer.mojo:1438-1448` — `share()` copies `device_state`, no data
- `tenmo/ndbuffer.mojo:857-859` — `is_contiguous()` is layout-only
- `tenmo/tensor.mojo:1545-1548`, `1564-1584` — `zeros` / `zeros_like` already take `device:`

**Sequencing evidence**
- `examples/mnist_gpu.mojo:80`, `mnist_conv2d_gpu.mojo:84`, `mnist_conv_tt_gpu.mojo:92`, `mnist_gpu_prof.mojo` — every GPU example uses `into_loader` (NativeLoader)
- `examples/mnist_native.mojo:105-114` — the only `DataLoader[...]` example (CPU)
- `tenmo/dataloader.mojo:100` — `NativeLoader` is "not directly Python-bound"
- `tenmo/dataloader.mojo:837`, `959-1017`, `1089`, `1162-1175` — `DataLoader`'s `_buffers_owned` / `_make_buffers` / `set_shuffle`, absent from `NativeLoader`

**New work**
- `tenmo/kernels/gather_kernel.mojo:66-99` — `gather_rows_2d_kernel` (the kernel to extend)
- `tenmo/kernels/gather_kernel.mojo:287-326` — the double-dispatch defect (Phase X)
- `tenmo/kernels/gather_kernel.mojo:189-225` — `gather_gpu`, per-call allocations
- `tenmo/gpu/device.mojo:191-197` — `create_sub_buffer[dtype](start, count)`
- `tenmo/ndbuffer.mojo:1306-1320` — GPU `create_sub_buffer` zero-copy precedent

**Sync budget**
- `tenmo/tensor.mojo:698` / `tenmo/ndbuffer.mojo:922` — `item()` → host read
- `tenmo/accuracy.mojo:9-23` — `compute(...) -> Float64`
- `tenmo/kernels/accuracy_kernel.mojo:126-132` — explicit sync then `map_to_host`
- `tenmo/crossentropy.mojo:269-279`, `1311-1322` — `reduction="none"` per-sample vector

**Open questions**
- `tenmo/optim.mojo:322-324` — `map_to_host` with no preceding sync (Q1)
- `tenmo/ndbuffer.mojo:1747-1770`, `1802-1825` — GPU `contiguous()` full copy (Q2)
- `tenmo/gpu/transfer.mojo:117-155` — `materialize_contiguous` host round-trip for strided
- `tenmo/epochs.mojo` — clone-off-buffer pattern to re-examine (Q6)

**Python surface**
- `python-binding/tenmo.py:1611-1620` — `DataLoader.__init__`, no `device=`
- `python-binding/tenmo.py:679-682` — `Tensor.device`, read-only, no transfer methods
- `python-binding/tenmo.py:1593-1596` — `_LOADER_PAIRS` (dtype, not device, dispatch)

**Call sites**
- `examples/mnist_gpu.mojo:62-63` (normalize), `:84-85`/`:92-93` (`normalize_*`), `:131-132` (`to_gpu`), `:176-177`/`:204-205` (per-batch transfer), `:187`/`:189`/`:210`/`:212` (syncs), `:251` (`to_cpu`)
- `mnist_pytorch_optimized.py:17-21` (`prep`), `:62` (`randperm`), `:63-64` (accumulators), `:78-79` (one sync/epoch), `:87-90` (eval slices)