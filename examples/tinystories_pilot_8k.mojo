"""
Stage-2b pilot: custom-vocab fork.
=========================================================================
Same shrink architecture and training protocol as `tinystories_pilot`
(6×256×8h ctx256, B=8 T=256 stride128, dropout 0.1, tie, xavier/seed 7,
peak 6e-4, warmup 1%) with ONE variable changed: the tokenizer is a
home-trained 8k BPE (`examples/data/tinystories_8k.tiktoken`, built by
`./example.sh tinystories_vocab`) instead of r50k_base. Expected
6,852,608 params (17,670,400 − 50,257×256 + 8,000×256 — the tied head
means only the shared table changes); asserted below, so silent config
drift fails loud.

Metric discipline: loss-per-token and tok/s are NOT comparable across
vocabs (fewer, longer tokens). Compare per character: divide loss by
chars/token (printed for both splits), and throughput in chars/s.
The 8k bet: ~6× smaller (B,T,V) traffic attacks the measured backward
hog (CE over 103M logits) and head forward (755 ms) at ~equal
compression (4.18 vs 4.1 chars/token on val).

Vocab bootstrap: `load_tiktoken`, falling back to train+save on the
train split when the file is absent — the example is self-contained.
Checkpointing: every
CKPT_EVERY_BATCHES training batches — plus the epoch end — saves
`pilot_8k_latest.npy` (weights + AdamW moments + scheduler clock +
best-tracker, self-described via metadata); `pilot_8k_best.npy` is
written only when eval improves, so a worse run never clobbers a
better checkpoint. RESUME=True restarts from _latest (weights,
moments, LR clock; the epoch itself restarts at batch 0 with a fresh
shuffle). `pilot_8k.npy` on disk is the frozen stage-2b artifact and
is never overwritten. All three files gitignored.
Scoreboard (SCOREBOARD_EVERY_BATCHES > 0): per-boundary val eval +
`pilot_8k_snap_<step>.npy` snapshot + fixed-prompt sample (also
gitignored); SCOREBOARD_EVERY_BATCHES = 0 keeps end-of-epoch eval only.

Run from the repo root: ./example.sh tinystories_pilot_8k.
"""

from bpe.tokenizer import Tokenizers, BPETokenizer
from tenmo.adamw import AdamW
from tenmo.checkpoint import save_state, load_state, apply_to_model
from tenmo.crossentropy import CrossEntropyLoss
from tenmo.dataloader import SlidingWindowDataset
from tenmo.epochs import train_epoch_sched, eval_epoch
from tenmo.generate import generate
from tenmo.gpt import GPTModel
from tenmo.numpy_interop import ndarray_ptr
from tenmo.scheduler import WarmupCosineLR
from tenmo.tensor import Tensor
from std.pathlib import Path
from std.python import Python, PythonObject
from std.utils.numerics import max_finite
from std.time import perf_counter_ns

comptime N_VOCAB = 8000  # home-trained 8k BPE (tinystories_vocab)
comptime N_PARAMS = 6852608
comptime N_LAYER = 6
comptime N_EMBD = 256
comptime N_HEAD = 8
comptime N_CTX = 256
comptime SEQ = 256
comptime BATCH = 8
comptime STRIDE = 128
comptime PEAK_LR = 6e-4
comptime FLOOR_LR = 3e-5
comptime VOCAB_PATH = "examples/data/tinystories_8k.tiktoken"
comptime CKPT_LATEST = "examples/data/pilot_8k_latest.npy"
comptime CKPT_BEST = "examples/data/pilot_8k_best.npy"
# Snapshot every K training batches (0 = only the end-of-epoch
# _latest). 200 x ~3.5 s/batch ~= 12 min of crash exposure.
comptime CKPT_EVERY_BATCHES = 200
# Resume weights + AdamW moments + LR clock from _latest (the epoch
# itself restarts at batch 0 with a fresh shuffle; the schedule
# continues from the restored clock, never replays warmup).
comptime RESUME = False
# Scoreboard (eval harness): when SCOREBOARD_EVERY_BATCHES > 0, every
# chunk boundary that lands on a multiple also runs a val eval, saves a
# full-state snapshot `pilot_8k_snap_<done>.npy` (post-hoc regeneration;
# gitignored like the other checkpoints), and prints a fixed-prompt
# sample — loss curve + perplexity + visible progression, not just an
# end-state sample. 0 = end-of-epoch eval only (default; zero behavior
# change). A step-0 eval+sample anchors the curve when enabled.
comptime SCOREBOARD_EVERY_BATCHES = 200
comptime SCOREBOARD_PROMPT = "Once upon a time"
comptime SCOREBOARD_MAX_NEW = 60
comptime SCOREBOARD_SEED = 7
# Same knob contract as the r50k pilot: -1 = full epoch, N > 0 = cap.
comptime PILOT_MAX_BATCHES = -1
comptime PILOT_LOG_EVERY = 5 if PILOT_MAX_BATCHES > 0 else 100


def _encode(text: String) raises -> List[Scalar[DType.int64]]:
    # 8k vocab gate: loud raise here, never an embedding crash later.
    var tok = BPETokenizer[Tokenizers.gpt2]()
    tok.load_tiktoken(VOCAB_PATH)
    var ids = tok.encode(text)
    var out = List[Scalar[DType.int64]](capacity=len(ids))
    for i in range(len(ids)):
        var v = ids[i]
        if v < 0 or v >= N_VOCAB:
            raise Error("tinystories_pilot_8k: token id out of 8k range")
        out.append(Scalar[DType.int64](v))
    return out^


def _shrink_model() -> GPTModel[DType.float32]:
    var model = GPTModel[DType.float32](
        n_vocab=N_VOCAB,
        n_ctx=N_CTX,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout_p=0.1,
        tie_weights=True,
        init_method="xavier",
        init_seed=7,
        qkv_bias=True,  # matches pilot_8k_*.npy snapshots + N_PARAMS guard
    )
    model.train()
    return model^


def _save_snapshot(
    model: GPTModel[DType.float32],
    optim: AdamW[DType.float32],
    sched: WarmupCosineLR[DType.float32],
    path: String,
    best: Float64,
    eval_loss: Float64,
) raises:
    """Full training snapshot: weights + AdamW moments + sched clock.

    `best` round-trips as a 1-elem float64 array (the
    scheduler/optimizer state_dict idiom) so resume restores the
    best-tracker exactly; `eval_loss` is informational (this save's own
    eval, -1.0 when the save is mid-epoch and unevaluated).
    """
    var np = Python.import_module("numpy")
    var meta: PythonObject = {}
    meta["phase"] = "pilot_8k"
    meta["vocab_size"] = N_VOCAB
    meta["best_eval_loss"] = np.array([best])
    meta["eval_loss"] = eval_loss
    var _ = save_state(path, model, optim, sched.state_dict(), meta)


def _scoreboard_sample(mut model: GPTModel[DType.float32], step: Int) raises:
    """Fixed-prompt sample at one scoreboard step (eval mode, fixed seed).

    Dropout is on during training, so sampling flips to eval and back;
    `eval_epoch` does the same internally, hence no mode surprises on
    either side of this call. The tokenizer reloads from VOCAB_PATH
    (already on disk this late) to keep the helper self-contained like
    `_encode`.
    """
    var tok = BPETokenizer[Tokenizers.gpt2]()
    tok.load_tiktoken(VOCAB_PATH)
    var prompt = Tensor[DType.int64].from_list[DType.int64](
        _encode(SCOREBOARD_PROMPT)
    )
    model.eval()
    var out = generate(
        model,
        prompt,
        max_new_tokens=SCOREBOARD_MAX_NEW,
        temperature=0.8,
        top_k=50,
        init_seed=SCOREBOARD_SEED,
    )
    model.train()
    var gen = List[Int](capacity=out.numels())
    for i in range(out.numels()):
        gen.append(Int(out.get(i)))
    print("---- sample @ step", step, "----")
    print(tok.decode(gen^))
    print("--------------------------------")


def main() raises:
    print("pilot_8k pins: 6x256x8h ctx256 V8000 drop0.1 tie xavier seed7")
    # ---- Corpus ----
    var train_text = Path("examples/data/tinystories_train.txt").read_text()
    var val_text = Path("examples/data/tinystories_val.txt").read_text()

    # ---- Vocab bootstrap: load, else train on the train split + save ----
    var boot = BPETokenizer[Tokenizers.gpt2]()
    try:
        boot.load_tiktoken(VOCAB_PATH)
        print("vocab: loaded", VOCAB_PATH)
    except e:
        print("vocab: training fresh 8k BPE —", e)
        var corpus = List[String]()
        corpus.append(train_text)
        boot.train(Span[String](corpus), N_VOCAB)
        boot.save_tiktoken(VOCAB_PATH)
        print("vocab: saved", VOCAB_PATH)

    var t0 = perf_counter_ns()
    var train_ids = _encode(train_text)
    var val_ids = _encode(val_text)
    var enc_s = Float64(perf_counter_ns() - t0) / 1e9
    print(
        "encode: train", len(train_ids), "val", len(val_ids), "tokens in",
        enc_s, "s (",
        Int(Float64(len(train_ids) + len(val_ids)) / enc_s), "tok/s)",
    )
    print(
        "compression: train",
        Float64(train_text.byte_length()) / Float64(len(train_ids)),
        "val",
        Float64(val_text.byte_length()) / Float64(len(val_ids)),
        "chars/token",
    )
    var train_ds = SlidingWindowDataset[DType.int64](
        train_ids^, seq_length=SEQ, stride=STRIDE
    )
    var val_ds = SlidingWindowDataset[DType.int64](
        val_ids^, seq_length=SEQ, stride=STRIDE
    )
    print("windows: train", len(train_ds), "val", len(val_ds))

    # ---- Model / loss / optimizer / schedule (pins mirror r50k pilot) ----
    var model = _shrink_model()
    print("params:", model.num_parameters())
    if model.num_parameters() != N_PARAMS:
        raise Error("tinystories_pilot_8k: param count drift — pins changed?")
    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var params = model.parameters()
    var adamw = AdamW[DType.float32](
        params^,
        lr=Scalar[DType.float32](0.0),  # schedule owns the LR from step 0
        clip_norm=Scalar[DType.float32](1),
        clip_value=Scalar[DType.float32](0.5),
    )
    var probe = train_ds.into_loader(batch_size=BATCH, shuffle=False)
    var n_batches = 0
    while probe.__has_next__():
        _ = probe.__next__()
        n_batches += 1
    print("batches/epoch:", n_batches)
    var warmup = n_batches // 100
    if warmup < 10:
        warmup = 10
    var sched = WarmupCosineLR[DType.float32](
        max_lr=Scalar[DType.float32](PEAK_LR),
        min_lr=Scalar[DType.float32](FLOOR_LR),
        warmup_steps=warmup,
        max_steps=n_batches,
    )
    var global_step = 0

    # ---- Resume: weights + moments + LR clock from _latest ----
    # The epoch itself always restarts at batch 0 with a fresh shuffle
    # (the loader cannot seek); the schedule continues from the
    # restored clock, so warmup never replays. Total steps may then
    # exceed max_steps by up to one epoch — the scheduler clamps to the
    # floor there by construction. global_step re-derives from the
    # clock (the two advance in lockstep, one per optimizer step).
    var best = Float64(max_finite[DType.float64]())
    comptime if RESUME:
        try:
            var resumed = load_state(CKPT_LATEST)
            apply_to_model(model, resumed)
            var params0 = model.parameters()
            adamw = AdamW[DType.float32].load_state_dict(
                resumed.optimizer_state, params0^
            )
            sched.load_state_dict(resumed.scheduler_state)
            global_step = sched.last_step + 1
            best = Float64(
                ndarray_ptr[DType.float64](
                    resumed.metadata["best_eval_loss"]
                ).unsafe_load()
            )
            print(
                "resumed:", CKPT_LATEST, "sched step", sched.last_step,
                "best eval", best,
            )
            # _latest's tracker only refreshes at epoch end, so a
            # crash-resume cycle would forget a _best set by an earlier
            # run — fold it back in when present.
            try:
                var prior = load_state(CKPT_BEST)
                var prior_best = Float64(
                    ndarray_ptr[DType.float64](
                        prior.metadata["best_eval_loss"]
                    ).unsafe_load()
                )
                if prior_best < best:
                    best = prior_best
                print("resume: best tracker", best, "from", CKPT_BEST)
            except e_best:
                print(
                    "resume: no usable", CKPT_BEST,
                    "- tracker stays", best,
                )
        except e:
            print("resume: no usable", CKPT_LATEST, "- fresh start:", e)

    # ---- Train one epoch in checkpointed chunks, timed ----
    # Chunking changes nothing mathematically: the loader's shuffle
    # permutation is fixed at construction (WindowLoader shuffles up
    # front), so successive capped calls consume exactly the batch
    # sequence one full call would. Each boundary saves _latest, so a
    # NaN halt keeps the last good state.
    # Capped runs train a prefix (smoke tests); the runtime clamp
    # covers cap > epoch. comptime-if/else assigns exactly once per
    # knob state, so no branch reads as a dead assignment.
    var total: Int
    comptime if PILOT_MAX_BATCHES > 0:
        total = PILOT_MAX_BATCHES
    else:
        total = n_batches
    if total > n_batches:
        total = n_batches
    var loader = train_ds.into_loader(batch_size=BATCH, shuffle=True)
    var t1 = perf_counter_ns()
    var done = 0
    var tot_loss = 0.0
    var tot_correct = 0.0
    var tot_tok = 0
    var have_ckpt = False
    # Scoreboard step-0 anchor: init loss + init gibberish before any
    # update, so the curve starts at step 0, not at the first boundary.
    comptime if SCOREBOARD_EVERY_BATCHES > 0:
        var ev0_loader = val_ds.into_loader(batch_size=BATCH, shuffle=False)
        var ev0 = eval_epoch(model, criterion, ev0_loader)
        print("scoreboard step 0 eval loss:", ev0[0], "acc:", ev0[1])
        _scoreboard_sample(model, 0)
    while done < total:
        var take = total - done
        comptime if CKPT_EVERY_BATCHES > 0:
            if take > CKPT_EVERY_BATCHES:
                take = CKPT_EVERY_BATCHES
        try:
            var r = train_epoch_sched(
                model, criterion, adamw, sched, global_step, loader,
                max_batches=take,
                log_every=PILOT_LOG_EVERY,
            )
            # Chunk means recombine by batch count (loader batches are
            # full short of a possible short tail — negligible here).
            tot_loss += r[0] * Float64(take)
            tot_tok += take * BATCH * SEQ
            tot_correct += r[1] * Float64(take * BATCH * SEQ)
            done += take
        except e:
            if have_ckpt:
                print(
                    "train halted at batch", done + 1,
                    "- last good:", CKPT_LATEST,
                )
            else:
                print(
                    "train halted at batch", done + 1,
                    "- no checkpoint yet",
                )
            raise e
        _save_snapshot(model, adamw, sched, CKPT_LATEST, best, -1.0)
        have_ckpt = True
        print("checkpoint:", CKPT_LATEST, "at batch", done, "of", total)
        # Scoreboard boundary: cadence rides the chunk boundaries (a
        # boundary fires when `done` lands on a multiple). Val eval is
        # forward-only and short next to a training chunk; the snapshot
        # is full state so any point regenerates post-hoc.
        comptime if SCOREBOARD_EVERY_BATCHES > 0:
            if done % SCOREBOARD_EVERY_BATCHES == 0:
                var evb_loader = val_ds.into_loader(
                    batch_size=BATCH, shuffle=False
                )
                var evb = eval_epoch(model, criterion, evb_loader)
                print(
                    "scoreboard step", done, "eval loss:", evb[0],
                    "acc:", evb[1],
                )
                _save_snapshot(
                    model,
                    adamw,
                    sched,
                    "examples/data/pilot_8k_snap_"
                    + String(done)
                    + ".npy",
                    best,
                    evb[0],
                )
                _scoreboard_sample(model, done)
    var train_s = Float64(perf_counter_ns() - t1) / 1e9
    var train_toks = total * BATCH * SEQ
    print("train loss:", tot_loss / Float64(total), "acc:", tot_correct / Float64(tot_tok))
    print(
        "throughput:", Int(Float64(train_toks) / train_s), "tok/s over",
        train_s, "s",
    )

    # ---- Eval + best tracking (save_best_if_improved contract) ----
    # _latest is already on disk (final chunk boundary): the resume
    # artifact, always overwritten. _best updates only on improvement —
    # a worse eval never clobbers a better checkpoint. (The library
    # overloads cover Sequential/SGD only, so the GPT-model form lives
    # here until a stage-3 driver reuses it.)
    var ev_loader = val_ds.into_loader(batch_size=BATCH, shuffle=False)
    var ev = eval_epoch(model, criterion, ev_loader)
    print("eval loss:", ev[0], "acc:", ev[1])
    if ev[0] < best:
        best = ev[0]
        _save_snapshot(model, adamw, sched, CKPT_BEST, best, ev[0])
        print("new best eval loss:", best, "- saved:", CKPT_BEST)
    else:
        print(
            "eval", ev[0], "- best unchanged:", best, "- kept:", CKPT_BEST
        )

    # ---- Round-trip gates (same contract as the r50k pilot) ----
    # Reload from _latest: proves the resume artifact itself carries
    # weights + sched clock + optimizer state, not just the model.
    var ckpt = load_state(CKPT_LATEST)
    var fresh = _shrink_model()
    fresh.eval()
    apply_to_model(fresh, ckpt)
    var gate_ds = SlidingWindowDataset[DType.int64](
        _encode(String(val_text[codepoint=0:4096])),
        seq_length=SEQ,
        stride=STRIDE,
    )
    var gate_loader = gate_ds.into_loader(batch_size=2, shuffle=False)
    ref gb = gate_loader.__next__()
    var xb = gb.features.clone()
    model.eval()
    var a = model(xb)
    var b = fresh(xb)
    var diff = (a - b).abs().sum().item()
    model.train()
    print("reload sum-diff:", diff)
    if diff != 0:
        raise Error("tinystories_pilot_8k: reloaded forward differs")
    var sched2 = WarmupCosineLR[DType.float32](
        max_lr=Scalar[DType.float32](PEAK_LR),
        min_lr=Scalar[DType.float32](FLOOR_LR),
        warmup_steps=warmup,
        max_steps=n_batches,
    )
    sched2.load_state_dict(ckpt.scheduler_state)
    if sched2.last_step != sched.last_step:
        raise Error("tinystories_pilot_8k: scheduler step not restored")
    var params2 = fresh.parameters()
    var adamw2 = AdamW[DType.float32].load_state_dict(
        ckpt.optimizer_state, params2
    )
    if Float64(adamw2.lr) != Float64(adamw.lr):
        raise Error("tinystories_pilot_8k: optimizer LR not restored")
    print("round-trip gates passed: forward identical, sched+optim restored")

    print("tinystories_pilot_8k passed: 8k pilot trains + checkpoints.")
