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

Each fix ships with a §Z2-style device-slice regression test;
GPU-proof in <=50-test chunks (per-binary compile is ~1000s flat,
so chunking is for early-exit granularity, not speed).

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
