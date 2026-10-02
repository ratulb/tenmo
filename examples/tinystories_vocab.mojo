"""
TinyStories vocab spike (custom-vocab fork, tooling question).
=========================================================================
Answers one question only: can mbpe train a BPE on our fetch, save it
as a `.tiktoken` file, and reload it into a working tokenizer?

Pipeline: `examples/data/tinystories_train.txt` -> `train` (8k merges,
GPT-2 pretokenizer — same splitting as the r50k run) ->
`save_tiktoken` -> fresh tokenizer + `load_tiktoken` -> gates:
reloaded ids match in-memory ids exactly, decode round-trips, all ids
in `[0, 8000)` -> compression (chars/token on the VAL split, unseen
in training) printed next to the r50k reference (~4.1).

This changes nothing about the run config: no model, no training loop,
no spec edits. The `.tiktoken` artifact lands in gitignored
`examples/data/`. A real custom-vocab pilot (new example files,
per-char-normalized comparison) is a separate, sequenced decision.

Run from the repo root: `./example.sh tinystories_vocab`.
"""

from bpe.tokenizer import Tokenizers, BPETokenizer
from std.pathlib import Path
from std.time import perf_counter_ns


comptime VOCAB_SIZE = 8000
comptime TRAIN_PATH = "examples/data/tinystories_train.txt"
comptime VAL_PATH = "examples/data/tinystories_val.txt"
comptime OUT_PATH = "examples/data/tinystories_8k.tiktoken"
comptime PROBE = "Once upon a time, Lily was very happy."


def main() raises:
    var train_text = Path(TRAIN_PATH).read_text()
    var val_text = Path(VAL_PATH).read_text()

    # ---- Train on the fetch (single-element corpus: whole file) ----
    var corpus = List[String]()
    corpus.append(train_text)
    var t0 = perf_counter_ns()
    var tok = BPETokenizer[Tokenizers.gpt2]()
    tok.train(Span[String](corpus), VOCAB_SIZE)
    var train_s = Float64(perf_counter_ns() - t0) / 1e9
    print("trained", VOCAB_SIZE, "vocab in", train_s, "s")

    # ---- Save + reload into a fresh tokenizer ----
    tok.save_tiktoken(OUT_PATH)
    var reloaded = BPETokenizer[Tokenizers.gpt2]()
    reloaded.load_tiktoken(OUT_PATH)
    print("saved + reloaded:", OUT_PATH)

    # ---- Gate 1: reloaded encoder agrees id-for-id, decode round-trips --
    var ids_live = tok.encode(PROBE)
    var ids_file = reloaded.encode(PROBE)
    if len(ids_live) != len(ids_file):
        raise Error("tinystories_vocab: reload length mismatch")
    for i in range(len(ids_live)):
        if ids_live[i] != ids_file[i]:
            raise Error("tinystories_vocab: reload id mismatch")
        if ids_file[i] < 0 or ids_file[i] >= VOCAB_SIZE:
            raise Error("tinystories_vocab: id out of vocab range")
    if reloaded.decode(ids_file) != PROBE:
        raise Error("tinystories_vocab: decode round-trip failed")
    print("probe:", PROBE, "->", len(ids_file), "ids, round-trip exact")

    # ---- Compression on the unseen val split vs the r50k reference ----
    var val_ids = reloaded.encode(val_text)
    var cpt = Float64(val_text.byte_length()) / Float64(len(val_ids))
    print(
        "val compression:", cpt, "chars/token over", len(val_ids),
        "tokens (r50k reference ~4.1)",
    )
    print("tinystories_vocab passed: train -> save -> reload -> encode.")
