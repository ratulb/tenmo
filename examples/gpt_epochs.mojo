"""
GPT multi-epoch training demo.
===============================================================
Trains the same tiny GPT as `gpt_overfit.mojo` on the full vendored
tiny-shakespeare excerpt through the shared `tenmo/epochs.mojo` loops:
two `train_epoch` passes (fresh `WindowLoader` per epoch, shuffled)
with an `eval_epoch` probe after each. Loss must fall epoch-over-epoch
on real text at a sane lr — otherwise the loop (not the graph) is wrong.

Run from the repo root: `./example.sh gpt_epochs`.
"""

from bpe.tokenizer import BPETokenizer
from tenmo.adamw import AdamW
from tenmo.crossentropy import CrossEntropyLoss
from tenmo.dataloader import SlidingWindowDataset
from tenmo.epochs import eval_epoch, train_epoch
from tenmo.gpt import GPTModel
from tenmo.tensor import Tensor
from std.pathlib import Path


def main() raises:
    # ---- Corpus: vendored tiny-shakespeare excerpt (~12 KB) ----
    var text = Path("examples/data/tinyshakespeare_excerpt.txt").read_text()
    print("corpus bytes:", text.byte_length())

    # ---- Tokenize: mbpe trained on the excerpt itself ----
    var tokenizer = BPETokenizer()
    var corpus = List[String]()
    corpus.append(text)
    tokenizer.train(corpus, 512)
    var ids = tokenizer.encode(text)
    print("tokens:", len(ids))

    var scalars = List[Scalar[DType.int64]](capacity=len(ids))
    for i in range(len(ids)):
        var v = ids[i]
        if v < 0 or v >= 512:
            raise Error("gpt_epochs: token ID out of vocab range")
        scalars.append(Scalar[DType.int64](v))
    var ds = SlidingWindowDataset[DType.int64](scalars^, seq_length=32, stride=8)
    print("windows:", len(ds))

    # ---- Model: same tiny config as the overfit smoke ----
    var model = GPTModel[DType.float32](
        n_vocab=512,
        n_ctx=32,
        n_embd=64,
        n_head=2,
        n_layer=2,
        tie_weights=True,
        init_seed=42,
    )
    model.train()
    var params = model.parameters()
    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var adamw = AdamW[DType.float32](
        params,
        lr=Scalar[DType.float32](3e-4),
        clip_norm=Scalar[DType.float32](1),
        clip_value=Scalar[DType.float32](0.5),
    )

    # ---- Two shuffled epochs, eval probe after each ----
    var prev_train = 0.0
    for epoch in range(2):
        var loader = ds.into_loader(batch_size=4, shuffle=True)
        var tr = train_epoch(model, criterion, adamw, loader, log_every=100)
        print("epoch", epoch + 1, "train loss:", tr[0], "acc:", tr[1])
        var ev_loader = ds.into_loader(batch_size=4, shuffle=False)
        var ev = eval_epoch(model, criterion, ev_loader)
        print("epoch", epoch + 1, "eval  loss:", ev[0], "acc:", ev[1])
        if ev[0] != ev[0] or ev[1] < 0.0 or ev[1] > 1.0:
            raise Error("gpt_epochs: eval readout not finite/in-range")
        if epoch == 1 and tr[0] >= prev_train:
            raise Error(
                "gpt_epochs: train loss did not fall epoch-over-epoch ("
                + String(prev_train)
                + " -> "
                + String(tr[0])
                + ")"
            )
        prev_train = tr[0]

    print("gpt_epochs passed: shared loops train on real text.")
