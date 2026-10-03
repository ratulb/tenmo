"""
IMDB BERT MLM pretraining smoke.

The mask game: per batch, ~15% of non-special positions are selected;
80% become `[MASK]`, 10% a random vocab id, 10% stay (BERT's 80/10/10);
labels hold the ORIGINAL ids at selected positions and `IGNORE`
elsewhere, so `CrossEntropyLoss(ignore_index)` scores masked slots
only. Encoder (bidirectional — no causal op anywhere) + `BertMLMHead`
predict; `AdamW` + `WarmupCosineLR` (schedule owns LR from step 0,
pilot pattern) update.

Scale discipline: tiny smoke (2Lx128, T=128, B=8) with `MAX_BATCHES`
capping the run (default 100 ≈ minutes on CPU); `-1` = full 25k-epoch.
Eval (loss + masked-acc) runs on a FIXED pre-masked slice of
`aclImdb/test` (frozen, never trained on — I4), so epoch-to-epoch
numbers compare fairly. Best checkpoint by eval loss; `_latest` every
`CKPT_EVERY_BATCHES` for crash resume (weights + AdamW moments + LR
clock, pilot-8k pattern).

Run from the repo root: `./example.sh imdb_bert_pretrain`.
Requires `examples/data/imdb_8k.tiktoken` (`./example.sh
imdb_bert_vocab` first).
"""

from tenmo.adamw import AdamW
from tenmo.argminmax import Argmax
from tenmo.checkpoint import save_state, load_state, apply_to_model
from tenmo.crossentropy import CrossEntropyLoss
from tenmo.dataloader import TensorDataset
from tenmo.encoder import BertForMLM, make_padding_mask
from tenmo.tensor import Tensor
from tenmo.nlp.bert_data import (
    IMDB_VOCAB_SIZE,
    IMDB_PAD_ID,
    IMDB_CLS_ID,
    IMDB_SEP_ID,
    IMDB_MASK_ID,
    IMDB_MODEL_VOCAB,
    IMDB_MAX_LEN,
    load_imdb_vocab,
    ensure_aclImdb,
    read_imdb_split,
    encode_review,
    materialize_batch,
)
from tenmo.scheduler import WarmupCosineLR
from std.random import seed, random_float64
from std.time import perf_counter_ns
from std.python import Python, PythonObject
from std.utils.numerics import max_finite

comptime N_LAYER = 2
comptime N_EMBD = 128
comptime N_HEAD = 4
# Pinned (pilot-8k pattern): wte 1024512 + tte 256 + wpe 16384 +
# emb-LN 256 + 2×198272 blocks + head 1049284. Drift fails loud.
comptime N_PARAMS = 2487236
comptime T = IMDB_MAX_LEN
comptime B = 8
comptime MASK_P = 0.15
comptime IGNORE = -100
comptime PEAK_LR = 1e-4
comptime FLOOR_LR = 1e-6
comptime SEED = 7
# Smoke cap: N > 0 = batches this run, -1 = full epoch (~3125 @ B=8).
# 100 = the cheap smoke (default); raise for a longer run.
comptime MAX_BATCHES = 100
comptime EVAL_EVERY = 25
comptime EVAL_N = 256
comptime EVAL_B = 32
comptime CKPT_LATEST = "examples/data/imdb_bert_mlm_latest.npy"
comptime CKPT_BEST = "examples/data/imdb_bert_mlm_best.npy"
comptime CKPT_EVERY_BATCHES = 50
comptime RESUME = False


def apply_mlm_mask(
    mut xb: Tensor[DType.int64],
    mask_p: Float64 = MASK_P,
) raises -> Tensor[DType.int64]:
    """80/10/10 BERT masking, in place on a CLONED batch.

    `[CLS]`/`[SEP]`/`[PAD]` are never selected (masking the
    summary token or filler teaches nothing). Returns `(B,T)` int64
    labels: original ids at selected positions, `IGNORE` elsewhere —
    the loss's `ignore_index` then scores masked slots only. Caller
    must pass a clone: the loader reuses its buffers across batches,
    and mutating them would corrupt the dataset (epochs.mojo clones
    for the same reason).
    """
    var nb = xb.shape()[0]
    var nt = xb.shape()[1]
    var rows = List[List[Scalar[DType.int64]]](capacity=nb)
    for b in range(nb):
        var row = List[Scalar[DType.int64]](capacity=nt)
        for t in range(nt):
            var orig = Int(xb[b, t])
            var label = IGNORE
            if (
                orig != IMDB_CLS_ID
                and orig != IMDB_SEP_ID
                and orig != IMDB_PAD_ID
                and random_float64(0.0, 1.0) < mask_p
            ):
                label = orig
                var r = random_float64(0.0, 1.0)
                if r < 0.8:
                    xb[b, t] = Scalar[DType.int64](IMDB_MASK_ID)
                elif r < 0.9:
                    xb[b, t] = Scalar[DType.int64](
                        Int(random_float64(0.0, Float64(IMDB_VOCAB_SIZE)))
                    )
                # else: keep original (the 10% the model must copy).
            row.append(Scalar[DType.int64](label))
        rows.append(row^)
    return Tensor[DType.int64].d2(rows^)


def masked_accuracy(
    logits: Tensor[DType.float32], labels: Tensor[DType.int64]
) -> Float64:
    """Fraction correct over `labels != IGNORE` positions.

    chance is ~1/8000, so anything in the percents is real
    learning. `argmax` over the vocab axis picks each position's top
    guess; unmasked slots never count.
    """
    var pred = Argmax[DType.float32].argmax(logits, axis=2)
    var nb = labels.shape()[0]
    var nt = labels.shape()[1]
    var correct = 0
    var total = 0
    for b in range(nb):
        for t in range(nt):
            var y = Int(labels[b, t])
            if y != IGNORE:
                total += 1
                if Int(pred[b, t]) == y:
                    correct += 1
    if total == 0:
        return 0.0
    return Float64(correct) / Float64(total)


def evaluate(
    mut model: BertForMLM[DType.float32],
    mut criterion: CrossEntropyLoss[DType.float32],
    eval_ids: Tensor[DType.int64],
    eval_mlm: Tensor[DType.int64],
    eval_lens: Tensor[DType.int64],
) raises -> Tuple[Float64, Float64]:
    """Fixed-mask eval: mean CE loss + masked accuracy, no grads."""
    model.eval()
    criterion.eval()
    var tot_loss = 0.0
    var tot_acc = 0.0
    var n = 0
    var N = eval_ids.shape()[0]
    var b = 0
    while b < N:
        var e = min(b + EVAL_B, N)
        var nb = e - b
        var xb = eval_ids.slice[track_grad=False](b, e, axis=0).clone()
        var yb = eval_mlm.slice[track_grad=False](b, e, axis=0).clone()
        var lb = eval_lens.slice[track_grad=False](b, e, axis=0).clone()
        _ = nb
        var mask = make_padding_mask(lb, T)
        var logits = model.forward_padded(xb, mask)
        var loss = criterion(logits.permute([0, 2, 1]), yb)
        tot_loss += Float64(loss.item())
        tot_acc += masked_accuracy(logits, yb)
        n += 1
        b = e
    model.train()
    criterion.train()
    return (tot_loss / Float64(n), tot_acc / Float64(n))


def main() raises:
    seed(SEED)
    ensure_aclImdb()
    var tok = load_imdb_vocab()

    # ---- Data: full train encode (order-preserving, T=128) ----
    print("reading + encoding train split...")
    var train_pair = read_imdb_split("/tmp/aclImdb/train")
    var train_texts = train_pair[0].copy()
    var rows = List[List[Int]]()
    var dummy = List[Int]()
    var done_enc = 0
    for i in range(len(train_texts)):
        var stext = train_texts[i]
        var enc = encode_review(tok, stext, T)
        rows.append(enc[0].copy())
        dummy.append(0)
        done_enc += 1
        if done_enc % 5000 == 0:
            print("  encoded", done_enc, "/", len(train_texts))
    var batch3 = materialize_batch(rows^, dummy^, T)
    var train_ids = batch3[0].clone()
    var train_lens = batch3[2].clone()
    print("train rows:", train_ids.shape()[0])

    # ---- Eval: fixed pre-masked slice of the FROZEN test split ----
    print("preparing fixed eval set...")
    var test_pair = read_imdb_split("/tmp/aclImdb/test")
    var test_texts = test_pair[0].copy()
    var erows = List[List[Int]]()
    var edummy = List[Int]()
    var ne = min(EVAL_N, len(test_texts))
    for i in range(ne):
        var stext = test_texts[i]
        var enc = encode_review(tok, stext, T)
        erows.append(enc[0].copy())
        edummy.append(0)
    var ebatch = materialize_batch(erows^, edummy^, T)
    var eval_ids = ebatch[0].clone()
    var eval_lens = ebatch[2].clone()
    # Fixed masks: sampled ONCE here, so every eval call scores the
    # same slots (comparable numbers across the run).
    var eval_mlm = apply_mlm_mask(eval_ids)
    print("eval rows:", ne)

    # ---- Model / loss / optimizer / schedule ----
    var model = BertForMLM[DType.float32](
        n_vocab=IMDB_MODEL_VOCAB,
        n_ctx=T,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        padding_idx=Optional[Int](IMDB_PAD_ID),
        dropout_p=0.1,
        init_seed=SEED,
        init_method="xavier",
    )
    model.train()
    print("params:", model.num_parameters())
    if model.num_parameters() != N_PARAMS:
        raise Error("imdb_bert_pretrain: param count drift — pins changed?")
    var criterion = CrossEntropyLoss[DType.float32](
        ignore_index=IGNORE, reduction="mean"
    )
    var params = model.parameters()
    var optim = AdamW[DType.float32](
        params^,
        lr=Scalar[DType.float32](0.0),  # schedule owns LR from step 0
        clip_norm=Scalar[DType.float32](1),
        clip_value=Scalar[DType.float32](0.5),
    )
    var dataset = TensorDataset[DType.int64, DType.int64](
        train_ids, train_lens
    )
    var probe = dataset.into_loader(batch_size=B, shuffle=False)
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
    var best = Float64(max_finite[DType.float64]())
    comptime if RESUME:
        try:
            var resumed = load_state(CKPT_LATEST)
            apply_to_model(model, resumed)
            var params0 = model.parameters()
            optim = AdamW[DType.float32].load_state_dict(
                resumed.optimizer_state, params0^
            )
            sched.load_state_dict(resumed.scheduler_state)
            global_step = sched.last_step + 1
            print("resumed:", CKPT_LATEST, "sched step", sched.last_step)
        except e:
            print("resume: no usable", CKPT_LATEST, "- fresh start:", e)

    # ---- Train ----
    var total = n_batches
    comptime if MAX_BATCHES > 0:
        total = MAX_BATCHES
    if total > n_batches:
        total = n_batches
    var loader = dataset.into_loader(
        batch_size=B, shuffle=True, drop_last=True
    )
    var np = Python.import_module("numpy")
    var t0 = perf_counter_ns()
    var step = 0
    while loader.__has_next__() and step < total:
        ref bb = loader.__next__()
        var xb = bb.features.clone()
        var lb = bb.labels.clone()
        var yb = apply_mlm_mask(xb)
        var mask = make_padding_mask(lb, T)
        var logits = model.forward_padded(xb, mask)
        var loss = criterion(logits.permute([0, 2, 1]), yb)
        var v = Float64(loss.item())
        if v != v:
            raise Error("imdb_bert_pretrain: loss NaN at step " + String(step + 1))
        var acc = masked_accuracy(logits, yb)
        optim.zero_grad()
        loss.backward()
        var lr_now = sched.step()
        optim.set_lr(lr_now)
        global_step += 1
        optim.step()
        step += 1

        if step % 5 == 0 or step == total:
            var el = Float64(perf_counter_ns() - t0) / 1e9
            print(
                "  step", step, "/", total, "loss:", v, "masked-acc:", acc,
                "lr:", Float64(lr_now),
                "elapsed_s:", el,
            )
        if step % CKPT_EVERY_BATCHES == 0:
            var meta: PythonObject = {}
            meta["phase"] = "imdb_bert_mlm"
            meta["best_eval_loss"] = np.array([best])
            meta["eval_loss"] = -1.0
            _ = save_state(CKPT_LATEST, model, optim, sched.state_dict(), meta)
            print("  saved", CKPT_LATEST)
        if step % EVAL_EVERY == 0 or step == total:
            var ev = evaluate(model, criterion, eval_ids, eval_mlm, eval_lens)
            print("  eval loss:", ev[0], "eval masked-acc:", ev[1])
            if ev[0] < best:
                best = ev[0]
                var meta: PythonObject = {}
                meta["phase"] = "imdb_bert_mlm"
                meta["best_eval_loss"] = np.array([best])
                meta["eval_loss"] = ev[0]
                _ = save_state(
                    CKPT_BEST, model, optim, sched.state_dict(), meta
                )
                print("  new best ->", CKPT_BEST)

    print("done: steps", step, "best eval loss:", best)
    right_context_probe(model)


def right_context_probe(mut model: BertForMLM[DType.float32]) raises:
    """Non-causality probe.
    Does the encoder really
    use RIGHT-side context?

    A causal (GPT-style) model physically cannot see future
    tokens, so hiding them changes nothing. A bidirectional encoder
    SHOULD get worse when the right side is taken away. For 64 fresh
    test reviews: mask one middle position, score its per-slot loss
    twice — full review vs everything right of the slot replaced by
    `[PAD]` (lengths shortened so attention truly ignores it) — and
    report the mean degradation. Positive = right context mattered =
    bidirectionality is real, not leaky plumbing.
    """
    ensure_aclImdb()
    var tok = load_imdb_vocab()
    var test_pair = read_imdb_split("/tmp/aclImdb/test")
    var test_texts = test_pair[0].copy()
    var probe_criterion = CrossEntropyLoss[DType.float32](
        ignore_index=IGNORE, reduction="none"
    )
    probe_criterion.eval()
    model.eval()
    var delta_sum = 0.0
    var n_helped = 0
    var probed = 0
    var ri = 0
    while probed < 64 and ri < len(test_texts):
        var stext = test_texts[ri]
        ri += 1
        var enc = encode_review(tok, stext, T)
        var true_len = enc[1]
        # Need a middle slot with >= 16 real tokens on the right.
        if true_len < 48 or true_len > T:
            continue
        var m = true_len // 2
        var rows = List[List[Int]]()
        rows.append(enc[0].copy())
        var lab = List[Int]()
        lab.append(0)
        var b3 = materialize_batch(rows^, lab^, T)
        var ids = b3[0].clone()
        var target = Int(ids[0, m])
        if target == IMDB_PAD_ID:
            continue
        ids[0, m] = Scalar[DType.int64](IMDB_MASK_ID)
        var lens_full = List[Scalar[DType.int64]]()
        lens_full.append(Scalar[DType.int64](true_len))
        var full_mask = make_padding_mask(
            Tensor[DType.int64].d1(lens_full^), T
        )
        var logits_full = model.forward_padded(ids, full_mask)
        # Right-ablated twin: positions past m become PAD, length
        # shortens to m+1 so attention cannot see them at all.
        var cut = ids.clone()
        for t in range(m + 1, T):
            cut[0, t] = Scalar[DType.int64](IMDB_PAD_ID)
        var lens_cut = List[Scalar[DType.int64]]()
        lens_cut.append(Scalar[DType.int64](m + 1))
        var cut_mask = make_padding_mask(
            Tensor[DType.int64].d1(lens_cut^), T
        )
        var logits_cut = model.forward_padded(cut, cut_mask)
        var lab_rows = List[List[Scalar[DType.int64]]](capacity=1)
        var lab_row = List[Scalar[DType.int64]](capacity=T)
        for t in range(T):
            if t == m:
                lab_row.append(Scalar[DType.int64](target))
            else:
                lab_row.append(Scalar[DType.int64](IGNORE))
        lab_rows.append(lab_row^)
        var labels = Tensor[DType.int64].d2(lab_rows^)
        var lf = probe_criterion(logits_full.permute([0, 2, 1]), labels)
        var lc = probe_criterion(logits_cut.permute([0, 2, 1]), labels)
        # `none` reduction keeps spatial dims: (B,T) per-slot
        # losses, so slot m of the single probe row is [0, m].
        if lf.shape()[0] != 1 or lf.shape()[1] != T:
            raise Error("right_context_probe: unexpected none-loss shape")
        delta_sum += Float64(lc[0, m]) - Float64(lf[0, m])
        if Float64(lc[0, m]) > Float64(lf[0, m]):
            n_helped += 1
        probed += 1
    model.train()
    print(
        "right-context probe:", probed, "slots, mean loss degradation",
        delta_sum / Float64(probed),
        "(> 0 = right side helped = bidirectional),",
        "slots degraded:",
        n_helped,
        "/",
        probed,
    )
