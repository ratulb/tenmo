# Loading GPT-2 Weights into Tenmo

Applies to tenmo rev `49b35a8` (paths below are relative to the tenmo repo
unless marked `book:` for the tokens-to-transformers repo). Symbol names are
authoritative; line numbers are approximate.

This guide is for a complete novice. It assumes you know what a tensor and a
model parameter are, and nothing else — not what a "model release" is, not
Hugging Face, not safetensors, not Tenmo's checkpoint format. By the end you
will have OpenAI's trained GPT-2 small speaking through Tenmo code, know how
to save and resume your own training, know how to repeat the process for
GPT-2 medium, and know what every failure message means.

---

## 1. What "loading weights" means, and why

A model has two halves: the **architecture** (the recipe — which layers, in
which order, which sizes) and the **weights** (the numbers inside those
layers). Tenmo's `GPTModel` (`tenmo/gpt.mojo`) is the architecture. When you
construct one, its weights are born random:

```mojo
var model = GPTModel[DType.float32, DType.int64](
    50257, 1024, 768, 12, 12, tie_weights=True, qkv_bias=True
)
```

Every weight is drawn from the Xavier distribution (`init_method="xavier"`,
the default). A randomly-weighted GPT babbles — it stalls in repetition
loops. Training is the process of nudging ~124 million numbers, one small
step at a time, until babble becomes language. For GPT-2 that took a
datacenter.

OpenAI then did something generous: they **published the final numbers**.
Loading weights means pouring their trained numbers into our identically
shaped container, so the model speaks immediately with zero training. The
guarantee this guide builds toward: **same numbers + same program = same
scores.** If our 124,439,808 numbers equal theirs and our code computes the
same math, our outputs match theirs exactly — and we verify that, not just
claim it (§8, §11).

---

## 2. Background concepts

Four ideas you need, each in a few sentences. If you already know Hugging
Face, skip to §3.

### 2.1. Model releases: the recipe vs the baked cake

In 2019 OpenAI published the paper *Language Models are Unsupervised
Multitask Learners* describing the GPT-2 architecture — the recipe. Later
they released the trained weights for four model sizes — the baked cakes.
Both are public; either alone is useless. This guide reunites them: Tenmo
re-implements the oven, this process fetches their cake.

### 2.2. Hugging Face Hub: where the numbers live

The Hugging Face Hub is a public file host for models. The `gpt2` repository
there holds two files we care about: `model.safetensors` (the weights) and
`config.json` (the sizes). Every file is versioned by git commit SHA. Ours is
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, recorded in the book's manifest
(`book: validate/ch04/MANIFEST`) — a pinned revision means anyone, anywhere,
downloads byte-identical files. We use the Hub as a dumb downloader via
`snapshot_download`; you need to know no Hub API beyond that call.

### 2.3. Safetensors: the upstream file format

Safetensors is a simple container: a dictionary mapping **tensor name → typed
array**, plus a header. It exists because the older way of shipping weights
(Python pickles) can execute malicious code on load; safetensors files are
plain data, memory-mappable and safe. What you must grasp: we open the file
with `load_file` and get back an ordinary Python dict of 160 named NumPy
arrays. Everything after that is dict-shuffling Python — renames and
shape checks, no machine learning.

### 2.4. Tenmo's checkpoint: our file format

A Tenmo checkpoint is a NumPy-saved dict with exactly four keys:

| Key | Contents |
|---|---|
| `model` | name → array, one per model parameter (the weights) |
| `optimizer` | the optimizer's state (learning rate, momentum buffers, step count) |
| `scheduler` | the learning-rate schedule's clock position |
| `metadata` | anything the caller wants (epoch, loss, config, source revision) |

The converter (§4) rewrites the upstream dict with new key names and wraps it
in this four-key envelope. The loader (§6) only ever looks at `model` when
you call `load_weights`; the other three keys matter when *resuming training*
(§9). The format is deliberately boring: any NumPy reader can open it.

> **Sidebar: what is pickling?**
> Pickle is Python's built-in way of freezing arbitrary objects (here: our
> dict of arrays) into bytes and thawing them back. `np.save` uses it
> whenever you hand it a dict instead of a plain array, and that is why the
> loader calls `np.load(path, allow_pickle=True)` (`tenmo/checkpoint.mojo`,
> `load_state`). Thawing an *untrusted* pickle can run malicious code, so
> NumPy makes you opt in with the flag. It is safe here because we unpickle a
> file we created ourselves two steps earlier with the converter — our own
> cooking, never a download. Note the neat symmetry: the upstream format is
> called *safe*tensors precisely because downloads need no pickling, and the
> only pickle in our whole pipeline is the local cache we build ourselves.

---

## 3. The pipeline map: two halves

```
huggingface.co/gpt2 ──snapshot_download──▶ model.safetensors (160 tensors)
        │
        │  book: validate/ch04/fetch_gpt2.py  (Python, once per machine)
        ▼
book: validate/ch04/data/gpt2_small_tenmo.npy  (~500 MB, gitignored)
        │
        │  load_weights(lm, ...)  (Mojo, every run)
        ▼
speaking GPTModel  ──check_top20.py──▶ matches HF reference 16/16
```

**Half 1 — fetch and convert** (Python, run once per machine): download the
upstream file, rename every tensor to Tenmo's naming scheme, verify every
shape, write the local `.npy`. **Half 2 — load and verify** (Mojo, runs
every time): pour the `.npy` into a `GPTModel`, run it, compare against an
independent reference. The `.npy` is a *cache*, never committed (it is
gitignored); anyone reproduces it with one command (§8).

---

## 4. GPT-2 small, fully worked

The GPT-2 family comes in four sizes. Small is the one this guide loads end
to end:

| Model | Layers | Width | Heads | Context | Vocab | Params |
|---|---|---|---|---|---|---|
| small (124M) | 12 | 768 | 12 | 1024 | 50257 | 124,439,808 |
| medium (355M) | 24 | 1024 | 16 | 1024 | 50257 | 354,823,168 |
| large (774M) | 36 | 1280 | 20 | 1024 | 50257 | ~774M |
| xl (1.5B) | 48 | 1600 | 25 | 1024 | 50257 | ~1.5B |

The constructor call spells the config out loud — vocabulary, context,
width, heads, layers:

```mojo
var lm = GPTModel[DType.float32, DType.int64](
    50257, 1024, 768, 12, 12, tie_weights=True, qkv_bias=True
)
```

(Constructor documented at `tenmo/gpt.mojo`, `GPTModel.__init__`: `n_vocab`,
`n_ctx`, `n_embd`, `n_head`, `n_layer`, `tie_weights=True` default,
`qkv_bias=False` default. GPT-2's layout needs `qkv_bias=True` — §7.)

Where does 124,439,808 come from? Add the shapes up:

- Word table `wte`: 50257 × 768 = 38,597,376
- Position table `wpe`: 1024 × 768 = 786,432
- One block: two norms (2 × 2 × 768) + attention (`c_attn` 768×2304 +
  bias, `c_proj` 768×768 + bias) + MLP (`c_fc` 768×3072 + bias, `c_proj`
  3072×768 + bias) = 7,087,872
- Twelve blocks: 85,054,464; final norm: 1,536
- Total: 38,597,376 + 786,432 + 85,054,464 + 1,536 = **124,439,808**

The parameter count is 148 tensors: 2 tables + 12 blocks × 12 tensors +
2 final-norm tensors. The generator asserts all three numbers every run
(`book: validate/ch04/generate_ch04_numbers.mojo`).

---

## 5. How `load_weights` works

Three functions in `tenmo/checkpoint.mojo`, each doing one job:

1. **`load_state(path)`** — deserializes. `np.load(path,
   allow_pickle=True).item()` yields the four-key dict; a `"model"` key
   means the structured layout, otherwise the whole dict is treated as
   legacy flat weights. Returns a `Checkpoint` struct.
2. **`apply_to_model(model, checkpoint)`** — copies. It walks
   `model.named_parameters("")` (every parameter with its dotted name,
   e.g. `h.5.mlp.c_fc.weight`), looks each name up in `model_state`, and
   `unsafe_memcpy`s the matching array into the tensor after a dtype/size
   check. Name-keyed, so **order in the file does not matter**; keys the
   model lacks are silently skipped, so partial loads work (e.g. MLM
   weights into a classifier fill the shared trunk and skip the fresh
   head).
3. **`load_weights(model, path)`** — the one you call. `load_state` then
   `apply_to_model`, model-only restore for inference. (The `GPTModel`
   overload lives at the bottom of `checkpoint.mojo`; `Sequential`, BERT
   MLM, and BERT classifier have their own identical-shape overloads.)

Tied weights need no re-tying step: `named_parameters` lists `wte` once,
so save/load stays symmetric (`checkpoint.mojo`, `save_state` docstring).

---

## 6. The naming bridge: upstream names ↔ Tenmo names

Two codebases, same math, different names. The converter's map
(`book: validate/ch04/fetch_gpt2.py`, `hf_name_to_tenmo`) is small:

| Upstream (`model.safetensors`) | Tenmo (`named_parameters("")`) | Note |
|---|---|---|
| `transformer.wte.weight` | `wte_wpe.wte.weight` | gains `wte_wpe.` prefix |
| `transformer.wpe.weight` | `wte_wpe.wpe.weight` | same |
| `transformer.h.{i}.ln_1.weight` / `.bias` | `h.{i}.ln_1.gamma` / `.beta` | `weight→gamma`, `bias→beta` |
| `transformer.h.{i}.attn.c_attn.*` | `h.{i}.attn.c_attn.*` | identical path |
| `transformer.h.{i}.attn.c_proj.*` | `h.{i}.attn.c_proj.*` | identical path |
| `transformer.h.{i}.ln_2.*` | `h.{i}.ln_2.gamma` / `.beta` | same rename |
| `transformer.h.{i}.mlp.c_fc.*` | `h.{i}.mlp.c_fc.*` | identical |
| `transformer.h.{i}.mlp.c_proj.*` | `h.{i}.mlp.c_proj.*` | identical |
| `transformer.ln_f.weight` / `.bias` | `ln_f.gamma` / `.beta` | same rename |

Three non-obvious facts:

- **No transpose anywhere.** Hugging Face `Conv1D` weights are already
  `[in, out]`, exactly like Tenmo's `Linear`. A transpose here would be a
  bug, not a fix.
- **The `wte_wpe.` prefix** exists because Tenmo nests both embedding
  tables in one `GPTEmbedding` struct (`tenmo/gpt.mojo`,
  `GPTModel.named_parameters`); the name is just the path down the struct
  tree: `wte_wpe.` + block `h.{i}.` + layer + leaf.
- **Every tensor is shape-checked, not assumed.** The converter asserts
  each array against `expected_shapes()` and asserts the final set equals
  the wanted set exactly — an unmapped or missing tensor fails loudly.

---

## 7. Why 12 tensors are skipped (and the totals that confuse everyone)

Upstream ships **160** tensors; Tenmo has **148** parameters. The 12 extras
are `h.{i}.attn.bias` — the causal-mask triangle. That is not learned data;
it is a fixed table derived from the sequence shape, which Tenmo rebuilds
itself. The converter skips them explicitly (`if name in ("h.%d.attn.bias"
...)`, never silently) and asserts `148 mapped / 12 skipped`.

The general rule: **skip anything derived, never anything learned** — and
make the counts assert so a silent drop is impossible.

Then the totals. Three numbers circulate and all three are correct in
context:

| Configuration | Total | When |
|---|---|---|
| tied (`tie_weights=True`) | 124,439,808 | the released GPT-2; head reuses `wte` |
| untied, bias-free head | 163,037,184 | the figure prose quotes (+38,597,376 = one word table) |
| untied, Tenmo default head | 163,087,441 | Tenmo's separate `lm_head` carries its own 50,257-bias |

Plus one flag: Tenmo defaults to `qkv_bias=False` (bias-free attention) but
GPT-2's layout keeps the QKV bias, so the loader constructs with
`qkv_bias=True`. **Rule: the constructor config must match the weights'
family**, or shapes will not line up and the asserts will tell you.

---

## 8. Step-by-step: do it yourself (small)

From the book repo root (`tokens-to-transformers/`):

1. **Fetch + convert** (downloads ~500 MB, needs the `[pypi-dependencies]`
   torch/transformers/safetensors/hub from `book: pixi.toml`):
   ```
   pixi run python validate/ch04/fetch_gpt2.py
   ```
   Success looks like:
   ```
   upstream: gpt2 @ 607a30d... (model.safetensors, 160 tensors)
   mapped tensors: 148 (expect 148), mask buffers skipped: 12 (expect 12)
   total elements: 124439808 (expect 124439808)
   wrote validate/ch04/data/gpt2_small_tenmo.npy (... bytes)
   ---- MANIFEST (record in validate/ch04/MANIFEST) ----
   ```
   Record the printed manifest (source SHA, size, sha256) in
   `validate/ch04/MANIFEST`.
2. **Load** (in any Mojo program, from the repo root so the relative path
   resolves):
   ```mojo
   from tenmo.checkpoint import load_weights
   var lm = GPTModel[DType.float32, DType.int64](
       50257, 1024, 768, 12, 12, tie_weights=True, qkv_bias=True
   )
   load_weights(lm, "validate/ch04/data/gpt2_small_tenmo.npy")
   lm.eval()
   ```
   `lm.eval()` flips training → inference mode (dropout off; ours is 0.0
   anyway — the call states intent).
3. **Run**: feed integer token IDs of shape `(1, T)`, get logits `(1, T,
   50257)` — one score per vocabulary word per position. Argmax each
   position for the model's top pick.
4. **Verify**: the independent reference (`validate/ch04/reference_gpt2.py`,
   Hugging Face `transformers` on the same weights) wrote
   `out/reference_top20.txt`. The check:
   ```
   pixi run python validate/ch04/check_top20.py
   ```
   Success looks like:
   ```
   positions with id mismatch: 0/16
   max abs score diff: 0.000128
   ```
   IDs identical at all 16 probe positions; scores agree to ~1e-4 (different
   accumulation order in Tenmo vs torch BLAS on scores near 100 — ~1e-6
   relative). The full pipeline, generator included, is one command:
   `pixi run validate`.

---

## 9. Saving and resuming your own training (full training state)

Loading released weights is only half the checkpoint story. When *you*
train, the file must capture everything needed to resume bit-exactly: not
just weights, but the optimizer's momentum, the scheduler's clock, and your
bookkeeping. That is what the other three keys are for.

### 9.1. Saving: `save_state`

The GPT + AdamW overload (`tenmo/checkpoint.mojo`, `save_state`) writes all
four keys in one `.npy`:

```mojo
var meta: PythonObject = {}
meta["phase"] = "pilot_8k"
meta["vocab_size"] = N_VOCAB
meta["best_eval_loss"] = np.array([best])   # 1-elem array idiom, see below
meta["eval_loss"] = eval_loss
var _ = save_state(path, model, optim, sched.state_dict(), meta)
```

Real call site: `examples/tinystories_pilot_8k.mojo` (`_save_snapshot`).
What each piece holds:

- **model** — `named_parameters("")` → arrays, same as the loader reads.
- **optimizer** — `AdamW.state_dict()` (`tenmo/adamw.mojo`): `type`,
  `lr`, `beta1`, `beta2`, `eps`, `weight_decay`, `clip_norm`,
  `clip_value`, `step_count` (each a 1-elem float64 array — exact below
  2⁵³, the standard scalar idiom), plus `m_states` / `v_states`, the two
  momentum buffers per parameter. Without these, a resume restarts with
  amnesiac momentum and the loss curve jumps.
- **scheduler** — `WarmupCosineLR.state_dict()` (`tenmo/scheduler.mojo`):
  `type`, `max_lr`, `min_lr`, `warmup_steps`, `max_steps`, `last_step`.
  `last_step` counts *optimizer steps* (-1 before the first). A resume that
  resets it would replay warmup mid-training and silently spike the
  learning rate — the struct's docstring calls this load-bearing.
- **metadata** — yours. Convention from the pilots: phase name, config,
  eval loss, and the best-tracker as a 1-elem float64 array (plain floats
  do not survive the NumPy round trip; arrays do).

**Which saver when:** `save_weights` (model only) for inference and
deployment — that is what the GPT-2 `.npy` effectively is, with empty
optimizer/scheduler and source metadata. `save_state` for anything you may
resume. `save_best_if_improved` (always writes `{prefix}_latest.npy`, adds
`{prefix}_best.npy` when the loss improves) and `save_step_checkpoint`
(`{prefix}_step_{step}.npy`) encode the two standard policies so every
driver names files the same way.

### 9.2. Resuming: the four-step dance

From `examples/tinystories_pilot_8k.mojo` (the `RESUME` block), in order:

```mojo
var resumed = load_state(CKPT_LATEST)          // 1. read the file
apply_to_model(model, resumed)                 // 2. restore weights
var params0 = model.parameters()
adamw = AdamW[DType.float32].load_state_dict(  // 3. rebuild optimizer
    resumed.optimizer_state, params0^
)
sched.load_state_dict(resumed.scheduler_state) // 4. restore the clock
global_step = sched.last_step + 1
best = Float64(ndarray_ptr[DType.float64](
    resumed.metadata["best_eval_loss"]).unsafe_load())
```

Note `AdamW.load_state_dict` is a static constructor (`tenmo/adamw.mojo`):
it takes the saved state *plus your live parameter list* and re-attaches
the momentum buffers to them — the optimizer is rebuilt around the model,
not beside it. Then the subtle bit the pilot gets right: `_latest`'s
best-tracker only refreshes at epoch end, so a crash-resume cycle would
forget a `_best` set by an earlier run — the code folds `CKPT_BEST`'s
tracker back in when present. That is metadata being load-bearing, and the
reason the best-tracker rides in `meta` as an array.

### 9.3. Round-trip gates: proof it works

`examples/tinystories_pilot.mojo` (gates section) asserts the whole
contract after every training run:

- **Gate 1:** fresh model + `apply_to_model(fresh, load_state(path))` →
  forward sum-difference versus the trained model is **exactly 0**.
- **Gate 2:** fresh scheduler + `load_state_dict(ckpt.scheduler_state)` →
  `last_step` equal; fresh `AdamW.load_state_dict(ckpt.optimizer_state)`
  → learning rate equal.

The unit suite (`tests/test_cpu_all_23.mojo`, checkpoint tests) covers the
same ground per-function: SGD state round trip, model-only save, optimizer
save, metadata save, legacy flat-format backward compatibility for both
`load_state` and `load_weights`, and `save_weights` round trip. If you
change anything in `checkpoint.mojo`, these are the tests that must stay
green.

---

## 10. What you get, and how to play

At the end of §8 you hold `lm`: 8 token IDs in, `(8, 4, 50257)`-shaped
logits out on the book's probe batch (or `(1, T, 50257)` on your own
input), bit-compatible with Hugging Face's GPT-2 small. Playgrounds in
increasing daring — all five exist as copy-pasteable code in
`book: validate/ch04/generate_ch04_numbers.mojo`:

1. **Greedy continuation (the speaking demo).** Append each argmax pick and
   feed the grown context back. The first twelve continuation IDs on the
   probe are pinned in the generator — yours must match.
2. **Top-k inspection.** Dump the top-20 IDs + scores per position and
   diff against `out/reference_top20.txt` (§8 step 4) — the scores that
   prove the loaded model thinks with OpenAI's numbers.
3. **Per-block trace.** Run the embedding + 12 blocks manually, keeping
   each block's output; compare hidden states block by block. This is the
   instrument you debug with.
4. **Corrupt and trace (the debugging recipe).** Scale one doorway
   (`h.5.ln_2.gamma × 1.5`), watch logits drift ~9.7, trace the first
   diverging block (5), reload clean and confirm 0.0. If a future code
   change ever breaks the reference match, this recipe names the culprit.
5. **Freeze and count.** `model.freeze()` cascades through every
   sub-layer; `named_parameters` then yields zero trainables while
   `num_parameters()` still reports the full 124,439,808 — the difference
   between *stored numbers* and *trained numbers*.
6. **Save and resume (yours).** After any finetuning, `save_state` the
   full snapshot per §9.1, resume per §9.2, and confirm Gate-1-style
   identical forward. You now own both directions of the pipeline.

---

## 11. Replicate on medium (355M): the worked checklist

Everything in §§3–8 repeated for the next size up. The loader code, the
name map, the skip rule, and the tying logic are all size-agnostic — **only
constants change**. Each item gives small-value → medium-value:

| What | Small | Medium |
|---|---|---|
| Upstream repo (`REPO_ID`) | `gpt2` | `gpt2-medium` |
| Constructor `(vocab, ctx, embd, heads, layers)` | `(50257, 1024, 768, 12, 12)` | `(50257, 1024, 1024, 16, 24)` |
| `N_LAYER` in `fetch_gpt2.py` | 12 | 24 |
| `expected_shapes()` dims | width 768, MLP 3072 | width 1024, MLP 4096 |
| Assert: mapped tensors | 148 (= 2 + 12·12 + 2) | **292** (= 2 + 24·12 + 2) |
| Assert: skipped masks | 12 | **24** |
| Assert: elements | 124,439,808 | **354,823,168** |
| Output file | `gpt2_small_tenmo.npy` | `gpt2_medium_tenmo.npy` (new name) |
| `qkv_bias` / `tie_weights` | `True` / `True` | unchanged (`True` / `True`) |

The medium element total falls out of the §4 formula with width 1024 and
MLP inner 4096: tables 51,463,168 + 1,048,576, per-block 12,596,224 × 24,
final norm 2,048 = **354,823,168** — the "355M" name. (Recompute from the
formula at write time; never trust a remembered integer.)

Expect and verify:

- **Download ~1.5 GB, `.npy` ~1.4 GB** — and enough RAM to hold both the
  safetensors dict and the converted dict simultaneously during
  conversion. This is the one genuinely new constraint versus small.
- **Same asserts, new numbers:** the converter prints `mapped tensors:
  292 (expect 292), mask buffers skipped: 24 (expect 24), total elements:
  354823168`. Any deviation fails loudly per §6.
- **Fresh manifest** with the `gpt2-medium` SHA — a different repo, a
  different pin.
- **Reference rerun:** point `reference_gpt2.py` at `gpt2-medium`,
  regenerate the top-20 file, rerun `check_top20.py` — same bars (IDs
  exact, scores within 1e-3).
- **Training-state note:** a medium *training* snapshot is ~1.4 GB for
  weights alone plus AdamW's two momentum buffers per parameter (~2.8 GB
  more) — size your disk and your patience before `save_state` on medium.

Large (774M) and xl (1.5B) follow the same checklist with their row from
the §4 table; only the memory arithmetic changes (xl ≈ 6 GB weights in
fp32). Re-verify every config value against upstream `config.json` — do
not scale the table from memory.

---

## 12. Failure gallery

| Symptom | Cause | Fix |
|---|---|---|
| Fetch asserts `tuple(arr.shape) == want[tname]` | Wrong family for the constants (e.g. medium weights, small shapes) | Match `REPO_ID` to `expected_shapes()` / constructor per §11 |
| `unmapped tensor: ...` | Upstream added or renamed a key | Inspect the name; extend `hf_name_to_tenmo` if learned, skip-list if derived |
| `missing: {...}` after conversion | A wanted tensor absent upstream | Same as above, in reverse — never silence it |
| Count assert `!= 148` (small) | Tenmo gained/lost a parameter since the pin | Update the three asserts together; say why in the commit |
| `check_top20` ID mismatch | Wrong weights in the model (stale `.npy`, untied-head model fed tied file layout, or `qkv_bias` mismatch) | Re-run fetch; check constructor flags; reload |
| Score diff in (1e-3, 1e-1) with IDs matching | dtype or accumulation-order change in either implementation | Bisect: exhibit-5 trace (§10.4) names the first diverging block |
| `FileNotFoundError: ... gpt2_small_tenmo.npy` | Forgot the fetch step — the file is gitignored by design | Run §8 step 1 |
| Resume loss jumps after reload | Optimizer/scheduler state not restored (weights-only resume) | Do all four dance steps (§9.2), not just `apply_to_model` |
| LR spikes mid-training after resume | `last_step` reset → warmup replayed | Restore scheduler state; confirm `last_step` per Gate 2 (§9.3) |
| Config change + old checkpoint goes stale | Shapes no longer line up | Start fresh or surgery the dict explicitly — `apply_to_model` skips missing keys but cannot reshape |

---

## Appendix A. File inventory

| Path | One line |
|---|---|
| `book: validate/ch04/fetch_gpt2.py` | Downloader + converter + manifest printer (164 lines) |
| `book: validate/ch04/reference_gpt2.py` | Independent `transformers` reference → `out/reference_top20.txt` |
| `book: validate/ch04/check_top20.py` | Exhibit-4 check: IDs exact, scores within 1e-3 |
| `book: validate/ch04/generate_ch04_numbers.mojo` | Mojo generator: asserts every number, incl. load + trace + speech |
| `book: validate/ch04/MANIFEST` | Pinned source SHA, size, sha256, licence |
| `book: validate/ch04/data/gpt2_small_tenmo.npy` | The converted weights (gitignored, ~500 MB) |
| `tenmo/checkpoint.mojo` | `Checkpoint`, `save/load_state`, `apply_to_model`, `save/load_weights`, best/step helpers |
| `tenmo/gpt.mojo` (`GPTModel`) | Architecture + constructor flags + `named_parameters` tree |
| `tenmo/adamw.mojo` (`state_dict`/`load_state_dict`) | Optimizer half of training snapshots |
| `tenmo/scheduler.mojo` (`WarmupCosineLR`) | Scheduler clock half of training snapshots |
| `examples/tinystories_pilot_8k.mojo` | Real save (`_save_snapshot`) + resume (`RESUME` block) call sites |
| `examples/tinystories_pilot.mojo` | Round-trip gates (forward-identical, sched+optim restored) |
| `examples/tinystories_generate.mojo` | Inference-side `load_weights` call site |
| `tests/test_cpu_all_23.mojo` (checkpoint tests) | Per-function round-trip + backward-compat suite |
