"""
IMDB BERT sentiment fine-tune.

Two phases, the documented order: (A) HEAD-ONLY — only head params
reach the optimizer while the fresh `[CLS]` head learns the sentiment
mapping (the encoder still runs forward/backward underneath; its
grads are simply never stepped); (B) FULL-STACK — all weights refine
at a lower LR. Flip `FROM_SCRATCH=True` for the control run:
identical code, random encoder, no MLM weights — the load-bearing
comparison for the transfer claim.

Transfer is by NAME: `apply_to_model` fills the `emb.`/`blk{i}.` keys
it finds in the MLM checkpoint and skips the fresh `head.clf.` keys
(checkpoint.mojo documents the contract); the run prints a transfer
audit (keys filled vs skipped) so silent mismatch fails loud.

Eval is ALWAYS the frozen `aclImdb/test` split (I4): a 2048-review
slice during training, the full 25k once at the end. Majority class
is 50% — anything must beat that to count. No weight saving in the
smoke (runs are bit-deterministic from SEED — re-run to reproduce).

Run from the repo root: `./example.sh imdb_bert`.
Requires the MLM checkpoint (`./example.sh imdb_bert_pretrain`
first) unless `FROM_SCRATCH=True`.
"""

from bpe.tokenizer import Tokenizers, BPETokenizer
from tenmo.accuracy import Accuracy
from tenmo.adamw import AdamW
from tenmo.checkpoint import load_state, apply_to_model
from tenmo.crossentropy import CrossEntropyLoss
from tenmo.dataloader import TensorDataset
from tenmo.encoder import BertForSequenceClassification, make_padding_mask
from tenmo.nlp.bert_data import (
    IMDB_PAD_ID,
    IMDB_MODEL_VOCAB,
    IMDB_MAX_LEN,
    load_imdb_vocab,
    ensure_aclImdb,
    read_imdb_split,
    encode_review,
    materialize_batch,
    lengths_of,
)
from tenmo.scheduler import WarmupCosineLR
from tenmo.tensor import Tensor
from std.random import seed
from std.time import perf_counter_ns

comptime N_LAYER = 2
comptime N_EMBD = 128
comptime N_HEAD = 4
comptime T = IMDB_MAX_LEN
comptime B = 16
comptime N_LABELS = 2
comptime SEED = 11
# False = transfer (needs `imdb_bert_pretrain` checkpoint); True =
# from-scratch control (runs standalone). 6k-subset frozen-test at
# matched LR: scratch 0.766 / transfer 0.758 (majority 0.5). Default
# True so the example runs without the checkpoint.
comptime FROM_SCRATCH = True
comptime HEAD_STEPS = 100
comptime FULL_STEPS = 1500
comptime HEAD_LR = 2e-4
comptime FULL_LR = 2e-4
comptime FLOOR_LR = 1e-6
comptime EVAL_EVERY = 375
# Smoke subset: contiguous TRAIN rows straddling the pos/neg boundary
# (row 12500), so 6000 balanced reviews with zero concat ops. Both
# arms train on the SAME subset; eval stays the FULL frozen test.
# Full-25k training is scale-up, not smoke.
comptime SUB_A = 9500
comptime SUB_B = 15500
comptime EVAL_B = 64
comptime CKPT_MLM_BEST = "examples/data/imdb_bert_mlm_best.npy"


def encode_split(
    mut tok: BPETokenizer[Tokenizers.gpt2],
    texts: List[String],
    labels: List[Int],
    tag: String,
) raises -> Tuple[Tensor[DType.int64], Tensor[DType.int64]]:
    """Order-preserving encode of a full split → (ids, labels).

    Lengths are derived per batch by `lengths_of`, not stored.
    """
    var rows = List[List[Int]]()
    var labs = List[Int]()
    for i in range(len(texts)):
        var stext = texts[i]
        var enc = encode_review(tok, stext, T)
        rows.append(enc[0].copy())
        labs.append(labels[i])
        if (i + 1) % 5000 == 0:
            print(" ", tag, "encoded", i + 1, "/", len(texts))
    var both = materialize_batch(rows^, labs^, T)
    return (both[0].clone(), both[1].clone())


def evaluate(
    mut model: BertForSequenceClassification[DType.float32],
    ids: Tensor[DType.int64],
    labels: Tensor[DType.int64],
    batch_size: Int,
) raises -> Tuple[Float64, Float64]:
    """Mean CE loss + accuracy over a split, no grads."""
    model.eval()
    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    criterion.eval()
    var tot_loss = 0.0
    var tot_acc = 0.0
    var n = 0
    var N = ids.shape()[0]
    var b = 0
    while b < N:
        var e = min(b + batch_size, N)
        var xb = ids.slice[track_grad=False](b, e, axis=0).clone()
        var yb = labels.slice[track_grad=False](b, e, axis=0).clone()
        var mask = make_padding_mask(lengths_of(xb), T)
        var logits = model.forward_padded(xb, mask)
        tot_loss += Float64(criterion(logits, yb).item())
        tot_acc += Accuracy[DType.float32].compute(logits, yb)
        n += 1
        b = e
    model.train()
    return (tot_loss / Float64(n), tot_acc / Float64(n))


def eval_balanced(
    mut model: BertForSequenceClassification[DType.float32],
    a_ids: Tensor[DType.int64],
    a_labels: Tensor[DType.int64],
    b_ids: Tensor[DType.int64],
    b_labels: Tensor[DType.int64],
) raises -> Tuple[Float64, Float64]:
    """Mean of per-half (loss.
    Acc): the halves are pos-only and
    neg-only by construction, so the average is a balanced accuracy
    even though each half alone is trivially gameable."""
    var ea = evaluate(model, a_ids, a_labels, EVAL_B)
    var eb = evaluate(model, b_ids, b_labels, EVAL_B)
    # Halves printed separately (not just the mean): (1.0, 0.0) means
    # a collapsed constant predictor, (0.5, 0.5) uniform mush — the
    # mean alone cannot tell them apart.
    print("    halves: pos-loss", ea[0], "pos-acc", ea[1],
          " neg-loss", eb[0], "neg-acc", eb[1])
    return ((ea[0] + eb[0]) / 2.0, (ea[1] + eb[1]) / 2.0)


def train_phase(
    mut model: BertForSequenceClassification[DType.float32],
    head_only: Bool,
    steps: Int,
    peak_lr: Float64,
    var dataset: TensorDataset[DType.int64, DType.int64],
    eval_a_ids: Tensor[DType.int64],
    eval_a_labels: Tensor[DType.int64],
    eval_b_ids: Tensor[DType.int64],
    eval_b_labels: Tensor[DType.int64],
    tag: String,
) raises:
    """One phase (head-only or full-stack). Loader built inside, so.
    each phase gets a fresh shuffle.

    ITERATION CONTRACT (learned the hard way): `NativeLoader` shuffles
    in `__iter__`, so batches MUST come from `for batch in loader`.
    Calling `loader.__next__()` directly skips the shuffle and serves
    identity order — on pos-first data the model then "learns" the
    constant-positive predictor while every scoreboard applauds.
    One `for` pass = exactly ONE epoch
    (`__next__` raises StopIteration at the end), so multi-epoch
    training needs `while done < steps:` outside + `for` inside; each
    re-entry reshuffles for free. `break` at the cap is the shape.
    """
    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    criterion.train()
    var opt_params = model.parameters()
    if head_only:
        opt_params = model.head.parameters()
    var optim = AdamW[DType.float32](
        opt_params^,
        lr=Scalar[DType.float32](0.0),
        clip_norm=Scalar[DType.float32](1),
        clip_value=Scalar[DType.float32](0.5),
    )
    var sched = WarmupCosineLR[DType.float32](
        max_lr=Scalar[DType.float32](peak_lr),
        min_lr=Scalar[DType.float32](FLOOR_LR),
        warmup_steps=10,
        max_steps=steps,
    )
    var loader = dataset.into_loader(batch_size=B, shuffle=True, drop_last=True)
    print(" ", tag, "loader batches/epoch:", len(loader), "steps requested:", steps)
    var done = 0
    var epoch = 0
    var t0 = perf_counter_ns()
    while done < steps:
        epoch += 1
        for batch in loader:
            if done >= steps:
                break
            var xb = batch.features.clone()
            var yb = batch.labels.clone()
            var mask = make_padding_mask(lengths_of(xb), T)
            var logits = model.forward_padded(xb, mask)
            var loss = criterion(logits, yb)
            var v = Float64(loss.item())
            if v != v:
                raise Error(
                    "imdb_bert: loss NaN at " + tag + " step " + String(done + 1)
                )
            optim.zero_grad()
            loss.backward()
            optim.set_lr(sched.step())
            optim.step()
            done += 1
            if done % 10 == 0 or done == steps:
                var el = Float64(perf_counter_ns() - t0) / 1e9
                var acc = Accuracy[DType.float32].compute(logits, yb)
                print(
                    " ", tag, "step", done, "/", steps, "loss:", v,
                    "train-acc:", acc, "elapsed_s:", el,
                )
            if done % EVAL_EVERY == 0 or done == steps:
                var ev = eval_balanced(
                    model, eval_a_ids, eval_a_labels, eval_b_ids, eval_b_labels
                )
                print(" ", tag, "eval loss:", ev[0], "eval acc:", ev[1])
            if done >= steps:
                break
    var el_done = Float64(perf_counter_ns() - t0) / 1e9
    print(" ", tag, "done:", done, "/", steps, "epochs:", epoch, "elapsed_s:", el_done)


def main() raises:
    seed(SEED)
    ensure_aclImdb()
    var tok = load_imdb_vocab()

    print("encoding train subset...")
    var train_pair = read_imdb_split("/tmp/aclImdb/train")
    var train_texts_all = train_pair[0].copy()
    var train_tags_all = train_pair[1].copy()
    var train_texts = List[String]()
    var train_tags = List[Int]()
    for i in range(SUB_A, min(SUB_B, len(train_texts_all))):
        var t = train_texts_all[i]
        train_texts.append(t)
        train_tags.append(train_tags_all[i])
    print("subset rows:", len(train_texts))
    var tm = encode_split(tok, train_texts, train_tags, "train")
    var train_ids = tm[0].copy()
    var train_labels = tm[1].copy()

    print("encoding frozen test split...")
    var test_pair = read_imdb_split("/tmp/aclImdb/test")
    var test_texts = test_pair[0].copy()
    var test_tags = test_pair[1].copy()
    var em = encode_split(tok, test_texts, test_tags, "test")
    var test_ids = em[0].copy()
    var test_labels = em[1].copy()

    var model = BertForSequenceClassification[DType.float32](
        n_vocab=IMDB_MODEL_VOCAB,
        n_ctx=T,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        n_labels=N_LABELS,
        padding_idx=Optional[Int](IMDB_PAD_ID),
        dropout_p=0.1,
        init_seed=SEED,
        init_method="xavier",
    )
    model.train()
    print("params:", model.num_parameters())

    # ---- Transfer (or the honest skip for the control run) ----
    comptime if FROM_SCRATCH:
        print("FROM_SCRATCH control: random encoder, no MLM weights.")
    else:
        var ckpt = load_state(CKPT_MLM_BEST)
        var filled = 0
        var skipped = 0
        var names = model.named_parameters("")
        for i in range(len(names)):
            if ckpt.model_state.__contains__(names[i].name):
                filled += 1
            else:
                skipped += 1
        print("transfer audit: filled", filled, "skipped", skipped)
        if filled == 0:
            raise Error("imdb_bert: MLM checkpoint matched nothing — abort")
        apply_to_model(model, ckpt)
        print("transferred encoder from", CKPT_MLM_BEST)

    # Eval halves: first 1024 TEST-pos rows + first 1024 TEST-neg
    # rows (test order is pos-first, so a head slice would be
    # all-positive and gameable). FIXED for the whole run.
    var eval_a_ids = test_ids.slice[track_grad=False](0, 1024, axis=0).clone()
    var eval_a_labels = test_labels.slice[track_grad=False](0, 1024, axis=0).clone()
    var eval_b_ids = test_ids.slice[track_grad=False](12500, 13524, axis=0).clone()
    var eval_b_labels = test_labels.slice[track_grad=False](12500, 13524, axis=0).clone()

    var dataset = TensorDataset[DType.int64, DType.int64](
        train_ids, train_labels
    )

    # ---- Phase A: head-only, then Phase B: full-stack ----
    train_phase(
        model, True, HEAD_STEPS, HEAD_LR, dataset,
        eval_a_ids, eval_a_labels, eval_b_ids, eval_b_labels, "head",
    )
    var ev_a = eval_balanced(
        model, eval_a_ids, eval_a_labels, eval_b_ids, eval_b_labels
    )
    print("after head-only: eval loss", ev_a[0], "eval acc", ev_a[1])
    train_phase(
        model, False, FULL_STEPS, FULL_LR, dataset,
        eval_a_ids, eval_a_labels, eval_b_ids, eval_b_labels, "full",
    )
    var ev_b = eval_balanced(
        model, eval_a_ids, eval_a_labels, eval_b_ids, eval_b_labels
    )
    print("after full-stack: eval loss", ev_b[0], "eval acc", ev_b[1])

    # ---- Final: FULL frozen test, once ----
    var final = evaluate(model, test_ids, test_labels, EVAL_B)
    print("FINAL frozen-test loss:", final[0], "acc:", final[1])
    print("majority-class baseline: 0.5")
