"""
Char-fronted GPT: train + generate end to end (REFERENCE SOLUTION).

DO NOT PEEK until you have written your own `MyCharPT` — this file
exists so a stuck learner can compare, not so a hasty one can copy.
It is deliberately NOT wired into `example.sh`; run it with
`./fire.sh examples/gpt_char_pretrain.mojo` from the repo root.

Pipeline (mirrors `gpt_overfit.mojo`, char-swapped, schedule-trained,
with the `gpt_generate.mojo` tail): vendored tiny-shakespeare excerpt
-> hand-written character pre-tokenizer -> BPE train (request 512,
observe EXACTLY 256: the excerpt is pure ASCII, so every word is one
byte, no adjacent pair ever exists, and zero merges happen) ->
SlidingWindow -> 2 scheduled epochs -> cached generate 20 tokens from
"To be" -> decode -> print.

Expected (seed-pinned): n_vocab == 256; epoch-2 loss <
epoch-1 loss (3.38 -> 2.72); a 25-id output (5 prompt + 20 new) that
decodes to char-level gibberish ("To bees tase y :lbo inn t").
"""

from bpe.pretokenizer import (
    PreTokenizer,
    ByteMapping,
    utf8_codepoint_byte_length,
)
from bpe.tokenizer import BPETokenizer
from tenmo.adamw import AdamW
from tenmo.crossentropy import CrossEntropyLoss
from tenmo.dataloader import SlidingWindowDataset
from tenmo.epochs import train_epoch_sched
from tenmo.generate import generate_cached
from tenmo.gpt import GPTModel
from tenmo.scheduler import WarmupCosineLR
from tenmo.tensor import Tensor
from std.pathlib import Path


struct MyCharPT(PreTokenizer):
    comptime byte_map: ByteMapping = ByteMapping.SEQUENTIAL

    def __init__(out self):
        pass

    @staticmethod
    def name() -> String:
        return String("mychar")

    def split[
        mut: Bool, //, origin: Origin[mut=mut]
    ](self, text: StringSpan[origin]) raises -> List[StringSpan[origin]]:
        var result = List[StringSpan[origin]]()
        var n = text.byte_length()
        if n == 0:
            return result^
        var span = text.as_bytes()
        var pos = 0
        while pos < n:
            # One "word" per codepoint. Clamp so a truncated trailing
            # sequence never reads past the end.
            var end = pos + utf8_codepoint_byte_length(span[pos])
            if end > n:
                end = n
            result.append(StringSpan(unsafe_from_utf8=span[pos:end]))
            pos = end
        return result^

    def write_to[T: Writer](self, mut writer: T):
        writer.write(String("MyCharPT"))


def main() raises:
    # ---- Checkpoint 1: the splitter counts codepoints, not bytes ----
    var pt = MyCharPT()
    var pieces = pt.split("a中😀")
    if len(pieces) != 3:
        raise Error("char_pretrain: expected 3 codepoint pieces")
    print("split ok: a中😀 -> 3 pieces")

    # ---- Corpus: vendored tiny-shakespeare excerpt (~12 KB) ----
    var text = Path("examples/data/tinyshakespeare_excerpt.txt").read_text()
    print("corpus bytes:", text.byte_length())

    # ---- Tokenize: OUR front end, BPE trained on the excerpt ----
    var tokenizer = BPETokenizer[MyCharPT]()
    var corpus = List[String]()
    corpus.append(text)
    tokenizer.train(corpus, 512)
    var nv = len(tokenizer)
    print("n_vocab after train (requested 512):", nv)
    # Pure-ASCII corpus -> every word is one byte -> no pairs exist ->
    # zero merges -> exactly the 256 base bytes. If this is not 256,
    # the corpus (or the splitter) changed: investigate, don't adjust.
    if nv != 256:
        raise Error("char_pretrain: expected exactly 256 (ASCII => no merges)")

    var ids = tokenizer.encode(text)
    print("tokens:", len(ids))
    var id_max = ids[0]
    for i in range(len(ids)):
        if ids[i] > id_max:
            id_max = ids[i]
    if id_max >= nv:
        raise Error("char_pretrain: token ID out of vocab range")

    # ---- Windows: flat ID stream -> shift-by-one windows ----
    var scalars = List[Scalar[DType.int64]](capacity=len(ids))
    for i in range(len(ids)):
        scalars.append(Scalar[DType.int64](ids[i]))
    var ds = SlidingWindowDataset[DType.int64](scalars^, seq_length=32, stride=8)
    print("windows:", len(ds))

    # ---- Model: tiny, vocab sized from the TRAINED tokenizer ----
    var model = GPTModel[DType.float32](
        n_vocab=nv,
        n_ctx=32,
        n_embd=64,
        n_head=2,
        n_layer=2,
        tie_weights=True,
        init_seed=42,
    )
    model.train()

    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    criterion.train()
    var params = model.parameters()
    var adamw = AdamW[DType.float32](
        params,
        lr=Scalar[DType.float32](0.0),
        clip_norm=Scalar[DType.float32](1),
        clip_value=Scalar[DType.float32](0.5),
    )
    var batches_per_epoch = len(ds) // 4 + 1
    var sched = WarmupCosineLR[DType.float32](
        max_lr=Scalar[DType.float32](3e-4),
        min_lr=Scalar[DType.float32](1e-6),
        warmup_steps=10,
        max_steps=2 * batches_per_epoch,
    )
    var global_step = 0

    var prev_loss = 1e30
    for epoch in range(2):
        var loader = ds.into_loader(batch_size=4, shuffle=True)
        var tr = train_epoch_sched(
            model, criterion, adamw, sched, global_step, loader,
            log_every=100,
        )
        print("epoch", epoch, "loss:", tr[0], "acc:", tr[1])
        if tr[0] >= prev_loss:
            raise Error("char_pretrain: loss did not fall across epochs")
        prev_loss = tr[0]

    # ---- Prompt -> cached generate -> decode ----
    # "To be" is 5 codepoints = 5 ids; 5 + 20 <= n_ctx=32 (cached
    # generation never slides, so the budget must fit).
    var prompt_text = "To be"
    var prompt_ids = tokenizer.encode(prompt_text)
    print("prompt:", prompt_text, "->", len(prompt_ids), "ids")
    var ps = List[Scalar[DType.int64]](capacity=len(prompt_ids))
    for i in range(len(prompt_ids)):
        ps.append(Scalar[DType.int64](prompt_ids[i]))
    var prompt = Tensor[DType.int64].from_list[DType.int64](ps^)

    var out = generate_cached(
        model,
        prompt,
        max_new_tokens=20,
        temperature=0.8,
        top_k=50,
        init_seed=1234,
    )
    if out.numels() != len(prompt_ids) + 20:
        raise Error("char_pretrain: output length mismatch")
    for i in range(len(prompt_ids)):
        if Int(out.get(i)) != prompt_ids[i]:
            raise Error("char_pretrain: prompt prefix not preserved")
    var gen = List[Int](capacity=out.numels())
    for i in range(out.numels()):
        var v = Int(out.get(i))
        if v < 0 or v >= nv:
            raise Error("char_pretrain: id out of vocab range")
        gen.append(v)
    var continuation = tokenizer.decode(gen^)
    print("---- continuation (char front end, 2 epochs) ----")
    print(continuation)
    print("-------------------------------------------------")
    print("char_pretrain passed: own front end -> train -> text.")
