# GPU kernel offset-drop audit (`tenmo/kernels/`)

Working note. Date: 2026-10-05. Status: audit complete (code reading
only, zero GPU spent); fixes NOT yet written except where noted.
Start here when scheduling kernel hardening.

## Background

Two bug classes found while chasing the `test_ip_*` failures
(see `docs/inplace-gpu-failures.md`):

- **Offset-drop**: a launcher receives a caller device buffer (via
  `DeviceState`) PLUS a `Layout` describing a possibly-offset VIEW,
  but indexes from the buffer base, ignoring `Layout.offset`.
- **Shared-from-birth aliasing**: since `Buffer`/`NDBuffer` became
  shared by default, `.copy()` is a new handle on the SAME memory;
  only `.clone()` is deep. Any copy-then-mutate sequence that
  assumes independence is broken (in tests: spurious results; in
  lib code: potentially live misbehavior).

Both classes are largely *dormant* today because most callers
densify through `to_device` (offset 0). Live consumers of real
device offsets: `Tensor.slice` on GPU, `NativeLoader`/`DataLoader`
sequential device batches, `contiguous_device_state` views.

## Reference fix shapes (already landed, copy these)

- Contiguous fast path: operate on
  `buffer.create_sub_buffer[datatype](layout.offset, numels)`
  (cf. `scalar_ops_kernel.mojo:277-281`,
  `scalar_inplace_ops_kernel.mojo` PATH 1).
- Strided paths: trailing `offset_: Int64` kernel param seeding all
  `a_base`/`a_idx` initializers (cf. `inplace_scalar_ops_strided`,
  `inplace_pow_op_strided`); or the `std_variance` pattern
  (explicit `in_offset` param, launcher passes `x_layout.offset`).
- Regression test pattern: device row-slice with real offset vs
  CPU-slice oracle + first-element offset pin (cf.
  `test_scalar_gpu.mojo` §Z2).

## Verdict table

`reads-views?` = reads caller buffer via `Layout`. `mutates-caller?`
= writes caller buffer (vs fresh `enqueue_create_buffer`, which is
output-safe). HONORED = sub-buffer / offset param / panic on
nonzero / `materialize_contiguous` / offset-carried DevicePointer.

### Already fixed — do not touch

| file | status |
|---|---|
| `matmul_kernel.mojo` | fixed |
| `accuracy_kernel.mojo` | fixed |
| `scalar_ops_kernel.mojo` PATH 1 | fixed via sub-buffer |
| `scalar_inplace_ops_kernel.mojo` all paths | fixed (sub-buffer + `offset_`) |
| `ndbuffer` contiguous path, `fill_device_state` | fixed |
| `gather_rows_2d_into` | safe via `panic` on nonzero offset |

### Needs fixing

| file : launcher | reads | mutates | paths affected | notes |
|---|---|---|---|---|
| `binary_ops_kernel.mojo : launch:700` | A,B | no | ALL (PATH 1,2,3,4,5) | hottest path: every oop tensor add/sub/mul/div on offset views |
| `binary_inplace_ops_kernel.mojo : launch:538` | A,B | yes (A) | ALL 4 paths | two operands; same fix shape as scalar |
| `scalar_ops_kernel.mojo : launch` PATH 2 | A | no | strided only | PATH 1 fixed; strided still base-indexed |
| `sgd_kernel.mojo` : both launchers | param,grad[,vel] | yes | all operands | dormant (params dense offset-0); harden + re-prove parity |
| `adamw_kernel.mojo : launch:49` | param,grad,m,v | yes | all 4 | same as SGD |
| `reduction_kernel.mojo` : 6 enqueues | A | no | all | `.sum()` over device slice reduces wrong rows |
| `minmax_kernel.mojo : launch:189` | A | no | both passes | same reduction-helper pattern |
| `filler_kernel.mojo : _fill_scalar_gpu` | — | yes (target) | contiguous path only | strided path honored via IndexIterator |
| `filler_kernel.mojo : _scatter_add_gpu` | target+source | yes | both branches | `_scatter_add_nd` next door is honored — inconsistent |
| `compare_kernel.mojo : Compare:349` | A,B | no | strided | kernel HAS `A_offset/B_offset`; launcher hardcodes `0,0` (:385-386) — one-line fix |
| `compare_kernel.mojo : AllClose:178`, `CompareScalar:484` | A(,B) | no | all | linear from base |
| `dotproduct_kernel.mojo : launch:115` | A,B | no | both paths | |
| `matrixvector_kernel.mojo : launch:89` | M,v | no | all | also assumes inner-contiguous |
| `vectormatrix_kernel.mojo : launch:89` | v,M | no | all | mirror of above |
| `argminmax_kernel.mojo : _gpu_reduce:115` | A | no | all | |
| `shuffle_kernel.mojo` : gather+scatter | A / grad | scatter target | both | coord*strides from base |
| `gather_kernel.mojo` : embedding_bag, gather_rows_2d | yes | no | all | generic `gather_gpu` is honored; these two lack the offset param |
| `layernorm_kernel.mojo : launch:113` | x,mean,var,gamma,beta | no | aux inputs | x honored via materialize; mean/var/gamma/beta read from base |
| `conv_tt.mojo : forward` | image,kernel,bias | no | bias only | image/kernel honored via offset-carried DevicePointer |
| `pad_kernel.mojo`, `concate_kernel.mojo` | src,dst | yes (dst) | dst writes | src honored via materialize; dst base assumed fresh-contig |
| `cast_kernel.mojo` | — | — | — | mentions offset only in a comment; uses materialize — HONORED, listed here to close the loop |

### Safe — no action

`bce` (8 launchers), `crossentropy_fused` (2), `unary_ops`
(incl. masked), `clip` (2), `cumsum` (2), `division` backward (2),
`dropout`, `where`, `trilu` (4), `conv_gpu`, `pool_tt` (2),
`conv_tt.backward` (3), `multinomial`, `onehot`, `random` (N/A —
pure generator), `cast_device`/`unary_device` (device fns, no
launcher), `kernel_helpers` (helpers, no enqueue).

## Shared-aliasing review (lib code, unresolved)

- Benign-by-aliasing, do NOT "fix" without re-proving parity:
  `optim.mojo` dense-GPU copies, `adamw.mojo:399-402`.
- Intent unchecked (copy-then-mutate that may require a true
  snapshot): `conv.mojo:238,308-310` (`kern_shadow`),
  `bceloss.mojo:1324,1415` (`target_copy`),
  `crossentropy.mojo:466`, `dataloader.mojo` index/offset copies.
  Grep training paths for copy-then-mutate assumptions before
  trusting them. Rule: `.copy()` = new handle, same memory;
  `.clone()` = new memory.

## Fix order (proposed, risk x size)

1. `binary_ops` + `binary_inplace` (shared pattern, highest traffic).
2. `compare` hardcoded zeros + `filler` contiguous fill (tiny).
3. Reductions/minmax + dotproduct/matvec/vecmat.
4. Optimizers (harden, parity proof).
5. Long tail (shuffle, gather variants, layernorm aux, conv bias,
   pad/concate dst).
6. Lib copy-then-mutate intent review.

Also open (no owner yet): strided+offset transpose-of-slice test,
scalar PATH2 OOP coverage, full 35-suite sweep (see §validation log
for the 692-test baseline).

Each fix ships with a §Z2-style device-slice regression test;
GPU-proof in <=50-test chunks (per-binary compile is ~1000s flat,
so chunking is for early-exit granularity, not speed).

## Open item: `NDBuffer.to_dtype` GPU path (study, 2026-10-05)

Status: studied, NOT changed. Current code
(`tenmo/ndbuffer.mojo:1309-1354`): same-dtype on GPU takes a
zero-copy view via `create_sub_buffer[NewType](0, len)`; cross-dtype
on GPU does a CPU round-trip (D2H, CPU cast, H2D). An earlier
experiment tried `create_sub_buffer` for the cross-dtype path and
hit inconsistencies — correctly abandoned. Findings (all from the
official `DeviceBuffer` contract,
`max/gpu/host/device_context/DeviceBuffer`):

1. **Reinterpret is not convert.** `create_sub_buffer[view_type]`
   "creates a new buffer that references a subset of the memory
   ... with a different element dtype" — same bytes re-read as a
   new type, no value conversion. A real cast changes bytes per
   element AND converts each value; a sub-buffer can do neither.
   Wrong tool for cross-dtype, full stop.
2. **Units trap.** `offset`/`size` are in **view_type elements**
   (`__len__` = bytes / sizeof(dtype)). Source-dtype counts
   passed for a differently-sized view read out of bounds or half
   the data — the likely shape of the earlier inconsistency.
3. **Aliasing even when sizes match.** The sub-buffer "shares the
   underlying memory" (same as the copy-constructor: "new reference
   to the same memory"). So the live same-dtype branch aliases the
   source while the CPU path allocates fresh — a caller mutating
   the "cast" corrupts the original. Minimum: comment it; safer:
   `clone()` there.

Fix recipe (when scheduled): replace the round-trip
(`ndbuffer.mojo:1335-1350`) with `CastKernel.launch`
(`tenmo/kernels/cast_kernel.mojo:30`), which already does
`materialize_contiguous` + elementwise convert into a FRESH
buffer. No sub-buffer involved.

Bool caveat: Mojo supports no `DType.bool` in GPU kernels, which is
why `CastKernel.launch` remaps bool <-> uint8 storage
(`src_datatype`/`dst_datatype`, `:48-53`) and allocates the result
in the storage dtype. Any `NDBuffer[DType.bool].to_dtype` wiring
must preserve that mapping on both sides (storage dtype for
alloc/launch, logical dtype for the returned Layout) — verify with
a bool round-trip test, not just f32/f16.

## GPU validation log (2026-10-05, 2x T4)

All green, 692 tests total. Per-binary compile ~1000s flat
regardless of test count (measured: 7 tests 1001s, 96 tests
1049s) — chunking buys early-exit granularity, not speed.

- `test_scalar_gpu` **229/229**: proof 79 + 3x50 chunks.
- `test_ndbuffer_inplace_gpu` **199/199**: proof 7 + 96 + 48 + 48.
- `test_ndbuffer_arithmetic_gpu` **82/82** (binary oop baseline).
- `test_broadcast` **113/113**.
- `test_inplace` **69/69** (Tensor-level iop).

Baseline meaning: the arith/broadcast/inplace suites pass
*without* the `binary_ops`/`binary_inplace` offset fixes — their
view tests densify on `to_gpu` transfer (offset 0), same dormancy
as §1 of `inplace-gpu-failures.md`. The kernel fixes (§Fix order)
are still wanted for live device-view consumers; no failing
baseline currently pins them.

Chunk files are box-only (`/root/tenmo/tests/`) + `/tmp/opencode/`
locally — deliberately uncommitted. Box logs: `/root/ip_proof.log`,
`nd_proof.log`, `nd_chunk*.log`, `ip_fix` superseded,
`arith_chunk*.log`, `bcast_chunk*.log`, `iop_chunk*.log`,
`scalar_chunk*.log`.

## Item 1 proven (2026-10-05 rental, 2x T4)

`BinaryKernel` + `BinaryInplaceKernel` offset fixes (commit
`46c5c75`): `test_bin_proof` 6/6 (§Z3 device-slice regressions:
oop+inplace x contig/broadcast/strided — fail pre-fix, green
post-fix), `test_ndbuffer_arithmetic_gpu` 85/85,
`test_ndbuffer_inplace_gpu` 202/202. Box logs: `/root/bin_proof.log`
(first attempt EXIT=137, empty log — cause undetermined, box showed
no memory pressure; relaunch clean), `/root/rq_arith*.log`,
`/root/rq_ndip*.log`. One process note: chunk splitter must bundle
non-test helper defs with the chunk that uses them.

## Item 2 proven: filler + compare (2026-10-06 rental, 2x T4)

`FillerKernel` + `Compare/AllClose/CompareScalar` offset fixes
(commit `c1be6d1`, test-bug fixes `d255bab`/`caac944`):
`test_cf_proof` 5/5 (§Z4 device-slice regressions), `test_fill`
**36/36** full-suite green (`/root/fill_bin2_run.log`).

Flake post-mortem (recorded so nobody re-chases it):
`test_scatter_add_offset_gpu` failed 5/5 in the 36-test binary while
passing 6/6 in small binaries. Dumps proved GPU `got` == CPU `want`
exactly (kernel correct). Root cause was the TEST: `Tensor(Shape)`
leaves memory uninitialized; big-binary pages carried earlier tests'
fill values (9.0/10.0), failing the untouched-row pins that assumed
zeros. Small binaries passed on fresh zero pages. Fix (commit
`e1f99de`): sentinel-fill `T` with 7.0, pins expect 7.0 untouched /
8.0 hit. Lesson: every GPU test must initialize its tensors —
zero-assumption pins are process-history coin flips.

Still open from item 2: NONE — `test_compare` no-regression chunks
`test_cmp_chunk1/2` 29/29 each (`/root/cmp1_run.log`,
`/root/cmp2_run.log`). Item 2 fully closed.

## Item 3 in progress: reductions/minmax/dot/matvec/vecmat (2026-10-06)

Audit verdict: ALL ignored `Layout.offset` — `reduce`, `product_reduce`,
`excl_product_kernel`, `log_sum_exp_f32/f64`, `welford_reduce`
(`reduction_kernel.mojo`); `reduce_minmax` + `build_minmax_mask`
(`minmax_kernel.mojo`); `reduce_argminmax`; `dot_product_32/64`
(offset AND strides — dense `a[i]*b[i]`); `matrix_vector_nd`
(`M_base`/`v_base` started at 0); `vector_matmul_nd` (same).
Dispatch sites pass whole `Layout`s that the launchers then discard;
CPU paths honor offsets. Transfer densifies, so the bug bites on
on-device slices.

Fix shape (commits `9406b62`, `d8a82c8`): trailing `offset_: Int64`
param on every device fn, seeded into the read base, `Int64(layout.offset)`
from each launcher — uniform, no sub-buffer needed (unlike items 1-2).
Dot additionally takes `a/b_stride_` (used `strides[0]`). Contiguous-inner
assumptions in matvec/vecmat left as-is (pre-existing, out of scope).

Write-side trap (SECOND instance — first was the minmax mask in item 2):
`excl_product_kernel` writes a FRESH view-sized buffer, so the offset seed
applies to `in_buffer` reads only; seeding the write ran out of bounds and
failed product-backward grads on the first proof run. Fixed with a separate
0-based `write_base`. Rule, now twice-earned: fresh output buffers are
always 0-based — only seed reads from shared storage.

Z5 regression tests (11, all GPU-slice vs CPU-oracle + hand pins):
sum/mean offset rows, minmax offset (999.0 planted OUTSIDE the slice),
dot offset slices, matvec offset rows, vecmat offset batch, argmax offset,
product offset fwd + backward-recompute (`store_excl_product=False`),
softmax offset, variance offset.

Pin-arithmetic lesson: minmax and matvec "failures" were MY pins wrong
(row-4 max is 19 not 23; rows map to 40r+20 not 40r+30) while GPU==CPU
oracle asserts passed. When the oracle passes and pins fail, check your
arithmetic before blaming the kernel.

Proof status: new rental (old 12h box expired). `test_gpu_all_24`
(dot) 55/55, `test_gpu_all_25` (argmax) 60/60 green. Chunk 1 rerun
green 64/64 — product-forward AND backward-recompute proven, i.e. the
excl write-base fix works on hardware. Chunks 5/6 rerun in flight
(`/root/run_56.sh`, ~35 min).
Full 1751-test sweep (22 remaining chunks) deferred — GPU time.

Stale-chunk lesson: the 5/6 rerun above first ran with STALE generated
files (pins fixed in the source test files but chunks generated before
the fix) and failed on the old arithmetic. Regenerating
(`generate_gpu_test_suite.py --chunks 30`, membership unchanged) and
re-shipping fixed it. Rule: any edit to an embedded test INVALIDATES
the generated chunks — regenerate + re-ship before every box run.

---

## Appendix: runbook for the fixer

Everything below is what the 2026-10-05 session learned the hard
way. Follow it literally; all commands are copy-paste ready.

### A. GPU box workflow

Box: `ssh -p 9191 root@127.0.0.1` (2x Tesla T4, 31 GB RAM).
Every box shell needs:

```bash
export PATH=/root/.pixi/bin:/opt/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
```

Sources live at `/root/tenmo` (plain files, NOT a git repo).
Transfer additional files as a tarball:

```bash
# local:
tar -czf /tmp/opencode/more.tgz tests/<files...> tenmo/<files...>
scp -P 9191 /tmp/opencode/more.tgz root@127.0.0.1:/root/more.tgz
# box:
tar -xzf /root/more.tgz -C /root/tenmo
```

Per-binary compile is ~1000s FLAT regardless of test count
(measured 7→1001s, 96→1049s). Chunking buys early-exit
granularity, not speed. Keep chunks <= 50 tests.

Split any test file into chunks with (keep header + `main`,
keep non-test helper defs WITH the chunk that uses them —
`iop_close` once shipped without its helper and burned a cycle):

```bash
python3 -c "
import re
src = open('tests/<file>.mojo').read()
m = re.search(r'(?m)^def ', src)
header, rest = src[:m.start()], src[m.start():]
blocks = [b for b in re.split(r'(?m)(?=^def )', rest) if b.strip()]
byname, order = {}, []
for b in blocks:
    mm = re.match(r'def (\w+)', b)
    if mm and mm.group(1) != 'main':
        byname[mm.group(1)] = b; order.append(mm.group(1))
for i, ns in enumerate([order[j:j+50] for j in range(0, len(order), 50)]):
    out = header + ''.join(byname[n] for n in ns) + '''
def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
'''
    open(f'/tmp/opencode/<tag>{i+1}.mojo','w').write(out)
"
```

Verify each chunk LOCALLY first (`pixi run mojo -I . <file>` —
GPU tests prune, runs in seconds), then ship + launch with
`setsid` so snapped connections don't kill the run, one chunk at
a time:

```bash
scp -P 9191 <chunk> root@127.0.0.1:/root/tenmo/tests/
ssh -p 9191 root@127.0.0.1 "export PATH=... ; setsid nohup bash -c \
  'cd /root/tenmo && pixi run mojo -I . tests/<chunk> \
  > /root/<tag>.log 2>&1; echo EXIT=\$? >> /root/<tag>.log' \
  >/dev/null 2>&1 </dev/null & disown; echo launched"
# poll:
ssh -p 9191 root@127.0.0.1 \
  "grep -E 'FAIL|Summary|EXIT' /root/<tag>.log | tail -5"
```

Pitfalls seen: `pixi run --manifest-path` runs in the WRONG cwd
(`cd /root/tenmo` inside the wrapped bash instead); chaining
`verify && ship` after `grep -E "error|..."` always succeeds
because grep matching "error" exits 0 — verify and ship as
SEPARATE steps.

### B. Fix recipes (issue -> exact fix per item)

General rule. Contiguous fast path: build
`buffer.create_sub_buffer[DeviceState[dtype].datatype](layout.offset,
numels)` when `layout.offset != 0` and pass THAT as the kernel
pointer (read-write safe: sub-buffers share storage at the
offset). Strided paths: append trailing `offset_: Int64` kernel
params (offsetS for two-operand kernels), seed every
`a_base`/`b_base`/`a_idx`/`b_idx` zero-initializer with
`Int(offset_)`, thread `Int64(layout.offset)` from the launcher
after the last shape/strides arg. Then add a §Z2 regression test
(see C) and prove in chunks (see A).

1. **`binary_inplace_ops_kernel.mojo : launch:538`** (mutates A).
   PATH 1 (:593-609): sub-buffers at `A_layout.offset` AND
   `B_layout.offset`. Strided kernels and their zero-initializers:
   `arithmetic_ops_A_contiguous` (:62; `b_base` :115, `b_idx`
   :154/:175), `arithmetic_ops_B_contiguous` (:191; `a_base` :253,
   `a_idx` :282/:299/:318), `arithmetic_ops_both_strided` (:336;
   `a_base`/`b_base` :388-389, `a_idx`/`b_idx` :404/:407,
   :424-425, :446-447),
   `arithmetic_ops_A_contiguous_lastdim_contiguous_B` (:466;
   `b_idx` :506 is `i % last_dim`, needs `+ Int(b_offset_)`).
   Launchers: PATH 2 `:673-683`, PATH 3 `:708-718`, PATH 4
   `:739-750`, lastdim `:658-666`.
2. **`binary_ops_kernel.mojo : launch:700`** (fresh output, reads
   A,B). Same shape with two offsets. Kernels:
   `arithmetic_ops_both_contiguous` (:23),
   `arithmetic_ops_both_contiguous_broadcast` (:70),
   `arithmetic_ops_A_contiguous` (:255; initializers
   :319/:354/:372), `arithmetic_ops_A_contiguous_lastdim_contiguous_B`
   (:390; :434), `arithmetic_ops_B_contiguous` (:461; :512,
   :534/:548/:566), `arithmetic_ops_both_strided` (:584; :623-624,
   :641/:644, :658-659). Launch sites: :744, :789, :806, :831,
   :856, :881.
3. **`compare_kernel.mojo : Compare.launch:349`**: replace
   hardcoded `Int64(0), Int64(0)` (:385-386) with
   `Int64(A_layout.offset), Int64(B_layout.offset)` — kernel
   already supports it. Then `AllClose.launch:178`,
   `CompareScalar.launch:484` (linear-from-base; sub-buffer or
   offset param).
4. **`filler_kernel.mojo`**: `_fill_scalar_gpu:216` contiguous
   enqueues (:241/:250) need target sub-buffer at
   `target_layout.offset`; `_scatter_add_gpu:351` both branches
   (:379/:393) need target+source offsets (mirror the honored
   `_scatter_add_nd_gpu:408`, which passes
   `target.offset/source.offset` at :471-472).
5. **Reductions** (`reduction_kernel.mojo` launches :880/:967/
   :1053/:1099/:1194/:1286, `minmax_kernel.mojo :189`,
   `dotproduct :115`, `matrixvector`/`vectormatrix :89`,
   `argminmax _gpu_reduce:115`, `shuffle` :86/:141,
   `gather` embedding_bag :238/:255): all pass base buffer +
   shape/strides with no offset. Same recipe; reductions thread
   one offset, scatter/gather two.
6. **Optimizers** (`sgd_kernel.mojo` :60/:95,
   `adamw_kernel.mojo` :49): harden the same way AND re-prove
   torch loss parity afterward (sequential, batch 64, SGD
   lr=0.01 momentum=0.9, same prep, seed 0, 2 epochs; reference:
   E1 0.341914, E2 0.128499, B0 2.330240). Params are dense
   offset-0 today so this is hardening, not a live fix.
7. **OOP scalar PATH 2** (`scalar_ops_kernel.mojo :300-322`):
   same recipe as B.1 single-operand.
8. **Partials** (layernorm aux :200-203, conv_tt bias, pad/concate
   dst :149): sub-buffer the remaining base-buffer reads/writes.

### C. Regression test recipe (§Z2 pattern)

For each fixed kernel, append to the matching test file:

```mojo
def test_<op>_offset_row_slice_matches_cpu() raises:
    comptime if has_accelerator():
        from tenmo.tensor import Tensor
        from tenmo.shared.shapes import Shape
        # 1. Build small CPU tensor with position-encoding values.
        # 2. to_gpu(), then .slice() ON DEVICE (real offset view).
        # 3. Run the fixed op on the device view (sync=True).
        # 4. Same op on the CPU slice; all_close full tensors.
        # 5. Pin the offset: first sliced element == expected value,
        #    row 0 untouched (pre-fix wrote rows 0..N).
```

Why this shape: `to_gpu` DENSIFIES (offset 0), so slicing must
happen on-device; position-encoding values (`r*W+c`) make wrong-row
reads/writes visible; the row-0 assertion catches exactly the
base-indexing bug. Still missing: a strided+offset test
(transpose-of-slice on device) for the strided kernels changed in
`scalar_inplace_ops_kernel.mojo`.

### D. Copy-then-mutate review procedure

For each site (`conv.mojo:238,308-310`, `bceloss.mojo:1324,1415`,
`crossentropy.mojo:466`, `dataloader.mojo` index/offset copies,
`adamw.mojo:399-402`): answer ONE question — does any code path
mutate the copy while the original must stay intact (needs
`.clone()`), or is the mutation MEANT to be visible through the
original (aliasing is load-bearing, leave it)? The tell: after
`var x = y.copy()`, does `y` get read again expecting old values
(bug: needs clone) or is `y` never touched / expected updated
(fine as-is)? Optimizer + AdamW copies are the second kind —
verified by torch parity. Flip nothing without a test proving the
new behavior.

### E. Option A pointers (Python-boundary auto-D2H)

- `_loader_next` at `python-binding/tenmo_bind.mojo:2794`: copy
  device-resident batches via `.to_cpu()` BEFORE wrapping in
  Python objects (Python iteration is a host operation).
- Register `device` for int dtypes: `register_data_loader`
  (`tenmo_bind.mojo:2901`) only registers
  `_loader_device[...]` (`:2919`) and `_tensor_device` for
  float32 (`:2971`) / float64 (`:3030`) — the
  `where dtype.is_floating_point()` guards at :822-931 exclude
  int64 labels. Add int registrations mirroring the float ones.
- Verify LOCALLY only: box `.so` rebuild (~2h wall) must be
  batched with other box work. Until it lands, document
  `to_gpu()`d Python-iterated loaders as unsupported (`.numpy()`
  on a GPU tensor segfaults; no graceful error).
