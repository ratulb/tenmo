# In-place GPU scalar failures (`test_ip_*`) — investigation notes

Working note. If the session ends before this is resolved, start here.
Status: `test_scalar_gpu.mojo` — 104 `test_ip_*` tests, **45 fail on GPU**.
Pristine-baseline run (pre-change HEAD) fails the byte-identical set:
pre-existing, not caused by device-resident loading work.
Old per-test logs were wiped with the box; `ip_list.log` re-run launched
2026-10-04 to recapture the exact FAIL list — read that first.

## 1. Confirmed by reading (no run needed)

`ScalarInplaceKernel` (`tenmo/kernels/scalar_inplace_ops_kernel.mojo`)
drops view offsets in every launch path — sixth instance of the
offset-drop class (cf. matmul, accuracy, scalar-oop, contiguous,
fill_device_state):

- `launch` PATH 1 (contiguous, :391-398): flat kernel from `A_buffer`
  base, `A_layout.offset` never added.
- `launch` PATH 2 (strided, :410-420) and `launch_inplace_pow` both
  paths: strided kernels take
  `(base_ptr, scalar, shape, strides, numels, rank)` — **no offset
  parameter exists**; `a_base` accumulates from 0.

Fix shape — IMPLEMENTED 2026-10-04 (committed, GPU proof pending):
PATH 1 mirrors `ScalarKernel.launch`
(`scalar_ops_kernel.mojo:271-278`, sub-buffer at offset) in both
`launch` and `launch_inplace_pow`; both strided kernels gained a
trailing `offset_: Int64` arg initializing the two `a_base` (one per
kernel) + four `a_idx` (slow-path + tail per kernel) sites via
`Int(offset_)`, threaded from both launchers. Kernels are only
compiled at those two sites. Regression tests added to
`test_scalar_gpu.mojo` §Z2 (`test_ip_offset_row_slice_add/pow_matches_cpu`:
device row-slice with real offset, CPU-slice oracle, first-element
offset pin). Strided+offset path changed but has no dedicated
regression test yet — add one (transpose-of-slice on device) when
the box is back.

## 2. Twist: the offset bug is likely dormant in exactly these tests

`NDBuffer.to_device` CPU→GPU (`ndbuffer.mojo:554-567`) **densifies**:
`fill_device_state` into a fresh buffer, bare `self.shape` attached —
offset 0, default strides. Every test does `a.to_gpu(gpu)` from CPU,
so every device tensor under test is dense offset-0 and the dropped
offset cannot fire. The offset fix is still correct and still wanted
— `NativeLoader`/`DataLoader` sequential batches are device views
with *real* offsets, the first consumers that would trip it — but it
likely explains ~0 of the 45.

Also cleared by reading: launch coverage (grid-stride loop,
documented in `gpu/runtime.mojo`), CPU oracle path, `sync=True`
plumbing. Name-count hypothesis only: 34 view-flavored
(`transposed`/sliced) + 13 `f64`-named − overlap ≈ 45, i.e. failures
may be *views + non-view f64*. The f64 half has no cause yet —
candidates: f64 H2D/D2H width, `simd_op[*, float64, *]` codegen,
`all_close` on f64.

## 3. Shared-by-default hypothesis (leading suspect for the remainder)

History: CPU `Buffer`/`NDBuffer` became **shared by default**; sharing
used to happen only via views/slices. Trail:
`1a9a745` Gradbox shared-from-birth, `64f7bf4` Gradbox shared
throughout, `772c0d6` "Got rid share flag", `0e58e4e` "`into_view`
not adding backward if already shared — GPU tensors are always
shared". The GPU suites were never thoroughly re-run after this
change.

Why it smells: `is_shared()` is used as a proxy for "is a view" in
fast paths. Now that *everything* is shared, those proxies
misclassify:

- `NDBuffer.reshape` lazy path (`ndbuffer.mojo:1595-1596`):
  `is_contiguous() and is_shared()` → returns an **alias** instead of
  a copy. Any downstream inplace op then mutates the base.
- `Tensor.into_view` backward-skip (`0e58e4e`): shared ⇒ no backward
  node. On GPU (always shared) this changes grad flow vs CPU
  expectations wherever tests compare grads.
- `contiguous(owned=False)` aliasing paths and `copy()` depth
  (shallow vs deep) now matter everywhere, not just for views.
- In-place ops on shared storage mutate all aliases: if `expected`
  and `a` (or `a_gpu`) ever alias through a shared buffer, the
  oracle comparison is meaningless (usually spurious-pass, but
  combined with transfers it can fail either way).

Concretely to check when resumed: does `NDBuffer.copy()` deep-copy
on all paths? Does `a.to_gpu(gpu)` of an *already-shared* CPU
buffer alias or copy? Does `expected.inplace_scalar_ops` (CPU) mutate
storage that `a_gpu`'s transfer later reads? Print `ref_count()` /
identity around the copy→transfer→inplace sequence for one failing
case.

## 4. Coverage gap (bigger than this file)

`scripts/gpu_test_files.txt` lists ~40 GPU suites. Since the
shared-by-default change, validated on hardware: `test_data`,
`test_contiguous`, `test_gather_gpu`, `test_matmul` (mm subset),
`test_accuracy` (mm subset), `test_reshape`, `test_scalar_gpu`
(partial). **Everything else is unrun in the new regime** — more
failures of the same family (offset-drop, shared-aliasing) may be
sitting in `test_ndbuffer_inplace_gpu`, `test_inplace`,
`test_cross_entropy`, `test_device_transfer_gradflow`, etc. After the
inplace fix lands, sweep the full GPU list before declaring the
regime clean; budget ~one session, mostly compiles.

## 5. VERDICT (2026-10-04): broken oracle, kernel innocent — FIXED

`ip_list.log`: all 45 FAILs are contiguous `a.copy()`-oracle tests
with non-idempotent ops; max/min pass by value-idempotence,
transposed pass via `contiguous()` (deep on the non-contiguous
path). Local probes (seconds, no GPU) proved it:

- `a.copy()` + inplace → `a[0] == 6.0`: **copy-init aliases**
  (`Buffer.__init__(copy:)` copies the pointer + refcount bump;
  deep copy is the explicitly-named `clone()`).
- `a.contiguous()` on contiguous input aliases too
  (`buffer[start:end]` is a view).
- `a.clone()` is genuinely deep (`a[0] == 1.0` preserved).

Mechanism: `expected = a.copy()` aliases `a`; CPU inplace mutates
both; `to_gpu` transfers the already-modified data; GPU applies the
op a second time. Fix (committed): 63× `a.copy()` → `a.clone()` in
`test_ip_*` oracle snapshots. Local 227/227; GPU proof run queued
(`ip_fix.log`, expect 45 → 0).

Corollary for §3: `Tensor.buffer.copy()`-style snapshots anywhere
in training code alias live storage. Notably `optim.mojo`'s
dense-GPU "copies" (`parameter.buffer.copy()` etc.) alias the
parameters — the earlier SGD write-back suspect is therefore
benign-by-aliasing, not a discarded update. Do NOT "fix" the
optimizer to deep-copy without re-proving parity: deep copies there
would *introduce* the write-back bug. The rule: `.copy()` = new
handle, same memory; `.clone()` = new memory. Grep training paths
for copy-then-mutate assumptions before trusting them.

Resume checklist (remaining — all need the GPU box, currently
connection-refused on :9191):

1. Confirm `ip_fix.log` 229/229 on GPU (`test_scalar_gpu`: 45
   oracle fixes + 2 new §Z2 offset regressions).
2. Run `test_ndbuffer_inplace_gpu` on GPU (expect 199/199 after
   its 88× `clone()` fix; same oracle disease, fixed same day).
3. Add a strided+offset regression test (transpose-of-slice on
   device) for the changed strided kernels.
4. Full GPU suite sweep (§4).
5. Known sibling: oop `scalar_ops_strided` PATH 2
   (`scalar_ops_kernel.mojo:300-322`) passes the base buffer with
   no offset either — same latent class, out of scope here.

## 6. Deferred: Option A (Python-boundary auto-D2H)

Decided 2026-10-04: `_loader_next` copies device-resident batches to
CPU before wrapping (Python iteration is a host operation), plus
register `device` for int dtypes (currently
`where dtype.is_floating_point()` only). Implement + local `.so`
build + Python CPU tests, but **do not spend a box `.so` rebuild on
it alone** — batch with the next session needing one. Reason:
`.numpy()` on a GPU tensor segfaults (no graceful error), and
`Int64Tensor` lacks `.device`; until Option A lands, a `to_gpu()`d
Python-iterated loader is a core dump waiting to happen — document
as unsupported in the meantime.
