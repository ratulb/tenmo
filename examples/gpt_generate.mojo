"""
GPT generate demo.
==================================================
Closes the prompt→text loop on the vendored tiny-shakespeare excerpt:
mbpe tokenize -> one `train_epoch` (shared loop) -> `generate` 30 tokens
from a text prompt (τ=0.8, top-k 50, seeded) -> mbpe decode -> print.

Assertions are structural only (length, prompt prefix, id range) — one
epoch does not teach Shakespeare; it proves the sampler, the sliding
window, and the decode path compose end to end.

Run from the repo root: `./example.sh gpt_generate`.
"""

from bpe.tokenizer import BPETokenizer
from tenmo.adamw import AdamW
from tenmo.crossentropy import CrossEntropyLoss
from tenmo.dataloader import SlidingWindowDataset
from tenmo.epochs import train_epoch
from tenmo.generate import generate
from tenmo.gpt import GPTModel
from tenmo.tensor import Tensor
from std.pathlib import Path


def main() raises:
    # ---- Corpus + tokenizer (same setup as the epoch demo) ----
    var text = Path("examples/data/tinyshakespeare_excerpt.txt").read_text()
    var tokenizer = BPETokenizer()
    var corpus = List[String]()
    corpus.append(text)
    tokenizer.train(corpus, 512)
    var ids = tokenizer.encode(text)
    print("tokens:", len(ids))

    var scalars = List[Scalar[DType.int64]](capacity=len(ids))
    for i in range(len(ids)):
        scalars.append(Scalar[DType.int64](ids[i]))
    var ds = SlidingWindowDataset[DType.int64](scalars^, seq_length=32, stride=8)

    # ---- Model + one epoch (enough to bias the sampler off uniform) ----
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
    var loader = ds.into_loader(batch_size=4, shuffle=True)
    var tr = train_epoch(model, criterion, adamw, loader, log_every=100)
    print("train loss:", tr[0], "acc:", tr[1])

    # ---- Prompt -> generate -> decode ----
    var prompt_text = "To be"
    var prompt_ids = tokenizer.encode(prompt_text)
    print("prompt:", prompt_text, "->", len(prompt_ids), "ids")
    var ps = List[Scalar[DType.int64]](capacity=len(prompt_ids))
    for i in range(len(prompt_ids)):
        ps.append(Scalar[DType.int64](prompt_ids[i]))
    var prompt = Tensor[DType.int64].from_list[DType.int64](ps^)

    var out = generate(
        model,
        prompt,
        max_new_tokens=30,
        temperature=0.8,
        top_k=50,
        init_seed=1234,
    )
    if out.numels() != len(prompt_ids) + 30:
        raise Error("gpt_generate: output length mismatch")
    for i in range(len(prompt_ids)):
        if Int(out.get(i)) != prompt_ids[i]:
            raise Error("gpt_generate: prompt prefix not preserved")
    var gen = List[Int](capacity=out.numels())
    for i in range(out.numels()):
        var v = Int(out.get(i))
        if v < 0 or v >= 512:
            raise Error("gpt_generate: id out of vocab range")
        gen.append(v)
    var continuation = tokenizer.decode(gen^)
    print("---- continuation ----")
    print(continuation)
    print("----------------------")
    print("gpt_generate passed: prompt -> text end to end.")
