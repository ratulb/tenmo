"""
Stage-2 pilot.
~1M TinyStories tokens at the decided shrink
config (6 layers, C=256, 8 heads, ctx 256 — same vocab/tokenizer).

One epoch over the fetched 4 MB prefix with warmup→cosine, periodic eval
on the val split, best-checkpoint save, then the roadmap
round-trip gates: fresh model + load_weights → bit-identical forward;
scheduler counter + AdamW LR restored from the same file. Kill criteria:
any NaN (raises inside the loop), final loss not below the first readout,
or throughput far under the stage-0 projection (~1500–2500 tok/s at
106 MFLOP/token).

Run log pins (recorded here, printed at runtime): init xavier/seed 7,
dropout 0.1, peak 6e-4, warmup 1% of steps, B=8, T=256, stride 128.
LR peak/warmup for the SMALLER model is the re-tune calls for —
kept at default values for the pilot, adjusted only if the curve demands.

Corpus: examples/data/tinystories_{train,val}.txt (fetch script).
Checkpoints: examples/data/ (gitignored) — never committed.

Run from the repo root: ./example.sh tinystories_pilot.
"""

from bpe.tokenizer import Tokenizers
from tenmo.adamw import AdamW
from tenmo.checkpoint import save_state, load_state, apply_to_model
from tenmo.crossentropy import CrossEntropyLoss
from tenmo.dataloader import SlidingWindowDataset
from tenmo.epochs import train_epoch_sched, eval_epoch
from tenmo.gpt import GPTModel
from tenmo.scheduler import WarmupCosineLR
from std.pathlib import Path
from std.python import PythonObject
from std.time import perf_counter_ns

comptime N_VOCAB = 50257  # mbpe gpt2 = r50k_base
comptime N_LAYER = 6
comptime N_EMBD = 256
comptime N_HEAD = 8
comptime N_CTX = 256
comptime SEQ = 256
comptime BATCH = 8
comptime STRIDE = 128
comptime PEAK_LR = 6e-4
comptime FLOOR_LR = 3e-5
comptime CKPT_PATH = "examples/data/pilot_best.npy"
# Timing knob: -1 = full epoch (the real pilot); N > 0 = stop after N
# batches (diagnostic timing without a 2 h burn). Not a training fork —
# same code path, just capped like any max_batches smoke run.
comptime PILOT_MAX_BATCHES = -1
# Log cadence follows the knob (comptime ternary: no dead-branch warning
# in either state, unlike a runtime `if` on the knob).
comptime PILOT_LOG_EVERY = 5 if PILOT_MAX_BATCHES > 0 else 100


def _encode(text: String) raises -> List[Scalar[DType.int64]]:
    # vocab gate: loud raise here, never an embedding crash later.
    var gpt2 = Tokenizers.get[Tokenizers.gpt2]()
    var ids = gpt2.encode(text)
    var out = List[Scalar[DType.int64]](capacity=len(ids))
    for i in range(len(ids)):
        var v = ids[i]
        if v < 0 or v >= N_VOCAB:
            raise Error("tinystories_pilot: token id out of r50k_base range")
        out.append(Scalar[DType.int64](v))
    return out^


def _shrink_model() -> GPTModel[DType.float32]:
    # Shrink pins: 8 heads → 32 dims/head, inside
    # the 32–64 band. dropout 0.1 = run config (rehearsal used 0.0).
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
        qkv_bias=True,  # pilot_best.npy must load into tinystories_generate
    )
    model.train()
    return model^


def main() raises:
    print("pilot pins: 6x256x8h ctx256 V50257 drop0.1 tie xavier seed7")
    # ---- Corpus: full fetched prefix (train) + val split ----
    var train_text = Path("examples/data/tinystories_train.txt").read_text()
    var val_text = Path("examples/data/tinystories_val.txt").read_text()
    var t0 = perf_counter_ns()
    var train_ids = _encode(train_text)
    var val_ids = _encode(val_text)
    var enc_s = Float64(perf_counter_ns() - t0) / 1e9
    print(
        "encode: train", len(train_ids), "val", len(val_ids), "tokens in",
        enc_s, "s (",
        Int(Float64(len(train_ids) + len(val_ids)) / enc_s), "tok/s)",
    )
    var train_ds = SlidingWindowDataset[DType.int64](
        train_ids^, seq_length=SEQ, stride=STRIDE
    )
    var val_ds = SlidingWindowDataset[DType.int64](
        val_ids^, seq_length=SEQ, stride=STRIDE
    )
    print("windows: train", len(train_ds), "val", len(val_ds))

    # ---- Model / loss / optimizer / schedule ----
    var model = _shrink_model()
    print("params:", model.num_parameters())
    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var params = model.parameters()
    var adamw = AdamW[DType.float32](
        params^,
        lr=Scalar[DType.float32](0.0),  # schedule owns the LR from step 0
        clip_norm=Scalar[DType.float32](1),
        clip_value=Scalar[DType.float32](0.5),
    )
    # One epoch of B=8/T=256 windows; warmup = 1% of steps.
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

    # ---- Train one epoch, timed (throughput vs stage-0 projection) ----
    var loader = train_ds.into_loader(batch_size=BATCH, shuffle=True)
    var t1 = perf_counter_ns()
    var tr = train_epoch_sched(
        model, criterion, adamw, sched, global_step, loader,
        max_batches=PILOT_MAX_BATCHES,
        log_every=PILOT_LOG_EVERY,
    )
    var train_s = Float64(perf_counter_ns() - t1) / 1e9
    var train_toks = n_batches * BATCH * SEQ
    print("train loss:", tr[0], "acc:", tr[1])
    print(
        "throughput:", Int(Float64(train_toks) / train_s), "tok/s over",
        train_s, "s",
    )
    if tr[0] != tr[0]:
        raise Error("tinystories_pilot: non-finite train loss")

    # ---- Eval + best checkpoint ----
    var ev_loader = val_ds.into_loader(batch_size=BATCH, shuffle=False)
    var ev = eval_epoch(model, criterion, ev_loader)
    print("eval loss:", ev[0], "acc:", ev[1])
    var meta: PythonObject = {}
    meta["phase"] = "pilot"
    meta["eval_loss"] = ev[0]
    var ckpt = save_state(
        CKPT_PATH, model, adamw, sched.state_dict(), meta
    )
    print("saved:", CKPT_PATH)

    # ---- Round-trip gates ----
    # Gate 1: fresh tied model + load_weights → identical forward.
    var fresh = _shrink_model()
    fresh.eval()
    apply_to_model(fresh, load_state(CKPT_PATH))
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
        raise Error("tinystories_pilot: reloaded forward differs")
    # Gate 2: scheduler counter + optimizer LR restore from the file.
    var sched2 = WarmupCosineLR[DType.float32](
        max_lr=Scalar[DType.float32](PEAK_LR),
        min_lr=Scalar[DType.float32](FLOOR_LR),
        warmup_steps=warmup,
        max_steps=n_batches,
    )
    sched2.load_state_dict(ckpt.scheduler_state)
    if sched2.last_step != sched.last_step:
        raise Error("tinystories_pilot: scheduler step not restored")
    var params2 = fresh.parameters()
    var adamw2 = AdamW[DType.float32].load_state_dict(
        ckpt.optimizer_state, params2
    )
    if Float64(adamw2.lr) != Float64(adamw.lr):
        raise Error("tinystories_pilot: optimizer LR not restored")
    print("round-trip gates passed: forward identical, sched+optim restored")

    print("tinystories_pilot passed: shrink pilot trains + checkpoints.")
