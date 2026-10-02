"""
Stage-1 dress rehearsal: every new piece on a small.
scale before TinyStories scale-up.

Exercises, in one run: mbpe `gpt2` tokenization of real story text (tokenizer
output, vocab gate = every id < 50257 or loud raise), the `from_list` →
`SlidingWindowDataset` → `WindowLoader` path, a real-vocab model ctor
(V=50257, C=64, 2 layers — full `wte`/head width, toy compute), and the
`WarmupCosineLR` → `train_epoch_sched` hook with a driver-owned
global step. Pass criteria: finite loss that falls pass-over-pass, the
schedule counter advancing exactly with batches run, and the optimizer LR
matching the schedule.

Deliberate non-run-config choices: `dropout_p=0.0` for a
deterministic gate (the run uses 0.1); tiny C/layers for minutes not hours.
Corpus: `examples/data/tinystories_train.txt` (fetched by
`scripts/fetch_tinystories.py`, gitignored), capped to REHEARSAL_CHARS so
mbpe-encode time stays trivial — full-text encode throughput is a stage-2
measurement, not asserted here.

Run from the repo root: `./example.sh tinystories_smoke`.
"""

from bpe.tokenizer import Tokenizers
from tenmo.adamw import AdamW
from tenmo.crossentropy import CrossEntropyLoss
from tenmo.dataloader import SlidingWindowDataset
from tenmo.epochs import train_epoch_sched
from tenmo.gpt import GPTModel
from tenmo.scheduler import WarmupCosineLR
from std.pathlib import Path

# Rehearsal scale: chars of corpus encoded (rest of the file untouched).
comptime REHEARSAL_CHARS = 100000
comptime N_VOCAB = 50257  # mbpe gpt2 = r50k_base


def main() raises:
    # ---- Corpus: fetched TinyStories train prefix, capped ----
    var full = Path("examples/data/tinystories_train.txt").read_text()
    # Codepoint slice: "chars" in the sense (stories are ASCII, so
    # codepoints == bytes here, but the semantic stays correct regardless).
    var text = full[codepoint=0:REHEARSAL_CHARS]
    print("corpus chars (capped):", text.byte_length())

    # ---- Tokenize: mbpe gpt2  ----
    var gpt2 = Tokenizers.get[Tokenizers.gpt2]()
    var ids = gpt2.encode(text)
    print("tokens:", len(ids))
    # vocab gate: loud failure here, never an embedding crash later.
    var scalars = List[Scalar[DType.int64]](capacity=len(ids))
    for i in range(len(ids)):
        var v = ids[i]
        if v < 0 or v >= N_VOCAB:
            raise Error(
                "tinystories_smoke: token id out of r50k_base range"
            )
        scalars.append(Scalar[DType.int64](v))
    var ds = SlidingWindowDataset[DType.int64](
        scalars^, seq_length=64, stride=16
    )
    print("windows:", len(ds))

    # ---- Model: real vocab width, toy compute (see module doc) ----
    var model = GPTModel[DType.float32](
        n_vocab=N_VOCAB,
        n_ctx=128,
        n_embd=64,
        n_head=2,
        n_layer=2,
        dropout_p=0.0,
        tie_weights=True,
        init_method="xavier",
        init_seed=1234,
        qkv_bias=True,  # mirror the pilot config (dropout excepted)
    )
    print("params:", model.num_parameters())
    model.train()
    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var params = model.parameters()
    var adamw = AdamW[DType.float32](
        params^,
        lr=Scalar[DType.float32](0.0),  # schedule owns the LR from step 0
        clip_norm=Scalar[DType.float32](1),
        clip_value=Scalar[DType.float32](0.5),
    )
    var sched = WarmupCosineLR[DType.float32](
        max_lr=Scalar[DType.float32](3e-4),
        min_lr=Scalar[DType.float32](3e-5),
        warmup_steps=10,
        max_steps=100,
    )
    var global_step = 0

    # ---- Two capped passes: loss must fall, schedule must advance ----
    var prev = 0.0
    for epoch in range(2):
        var loader = ds.into_loader(batch_size=8, shuffle=True)
        var r = train_epoch_sched(
            model, criterion, adamw, sched, global_step, loader,
            max_batches=30,
        )
        print(
            "pass", epoch + 1, "loss:", r[0], "acc:", r[1],
            "steps:", global_step,
            "lr:", Float64(sched.get_last_lr()),
        )
        if r[0] != r[0]:
            raise Error("tinystories_smoke: non-finite loss")
        if epoch == 1 and r[0] >= prev:
            raise Error(
                "tinystories_smoke: loss did not fall pass-over-pass"
            )
        prev = r[0]
    if global_step != 60:
        raise Error("tinystories_smoke: schedule ran wrong batch count")
    if Float64(adamw.lr) != Float64(sched.get_last_lr()):
        raise Error("tinystories_smoke: optimizer LR != schedule")

    print("tinystories_smoke passed: Ep-18 pieces train on real text.")
