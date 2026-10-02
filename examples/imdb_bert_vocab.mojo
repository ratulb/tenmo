"""
IMDB BERT vocabulary builder.

Trains an 8k BPE on RAW `aclImdb/train` review text (raw, not cleaned:
BPE learns its own normalization from bytes, and the GPT-2
pretokenizer already handles punctuation/casing splits — cleaning
would only burn signal), appends the four BERT specials beyond the
trained range, saves `examples/data/imdb_8k.tiktoken`, and gates the
artifact: reload id-equality, decode round-trip, id range, specials
survival (via re-registration), and chars/token compression on the
unseen TEST split (stats only — no test text enters training, I4).

Dataset: downloaded on first run to /tmp/aclImdb (same source as
`imdb_sentiment_v1.mojo`), ~84MB tarball, 25k train + 25k test.

Run from the repo root: `./example.sh imdb_bert_vocab`.
"""

from bpe.tokenizer import Tokenizers, BPETokenizer
from std.time import perf_counter_ns
from tenmo.nlp.bert_data import (
    IMDB_VOCAB_SIZE,
    IMDB_MODEL_VOCAB,
    IMDB_CLS_ID,
    IMDB_SEP_ID,
    IMDB_VOCAB_PATH,
    register_imdb_specials,
    load_imdb_vocab,
    log_corpus_stats,
    ensure_aclImdb,
    read_imdb_split,
)

comptime TRAIN_DIR = "/tmp/aclImdb/train"
comptime TEST_DIR = "/tmp/aclImdb/test"
comptime PROBE = "This movie was brilliant, a wonderful surprise."


def main() raises:
    ensure_aclImdb()
    print("dataset present:", TRAIN_DIR)

    # ---- Corpus: raw train text, concatenated single-element (the
    # tinystories_vocab shape — one big String, not 25k spans) ----
    print("reading train split...")
    var train_pair = read_imdb_split(TRAIN_DIR)
    var train_texts = train_pair[0].copy()
    print("train reviews:", len(train_texts))
    var corpus_text = String("")
    for ti in range(len(train_texts)):
        var t = train_texts[ti]
        corpus_text += t
        corpus_text += "\n"
    print("corpus bytes:", corpus_text.byte_length())

    var corpus = List[String]()
    corpus.append(corpus_text^)

    var t0 = perf_counter_ns()
    var tok = BPETokenizer[Tokenizers.gpt2]()
    tok.train(Span[String](corpus), IMDB_VOCAB_SIZE)
    print(
        "trained",
        IMDB_VOCAB_SIZE,
        "BPE in",
        Float64(perf_counter_ns() - t0) / 1e9,
        "s",
    )
    register_imdb_specials(tok)
    tok.save_tiktoken(IMDB_VOCAB_PATH)
    print("saved:", IMDB_VOCAB_PATH, "(specials NOT persisted — by design)")

    # ---- Gate 1: reload id-equality + decode round-trip + range ----
    var reloaded = load_imdb_vocab(IMDB_VOCAB_PATH)
    var ids_live = tok.encode(PROBE)
    var ids_file = reloaded.encode(PROBE)
    if len(ids_live) != len(ids_file):
        raise Error("imdb_bert_vocab: reload length mismatch")
    for i in range(len(ids_live)):
        if ids_live[i] != ids_file[i]:
            raise Error("imdb_bert_vocab: reload id mismatch")
        if ids_file[i] < 0 or ids_file[i] >= IMDB_MODEL_VOCAB:
            raise Error("imdb_bert_vocab: id out of model-vocab range")
    if reloaded.decode(ids_file) != PROBE:
        raise Error("imdb_bert_vocab: decode round-trip failed")
    print("probe:", PROBE, "->", len(ids_file), "ids, round-trip exact")

    # ---- Gate 2: specials survived via re-registration ----
    var wrapped = "[CLS] " + PROBE + " [SEP]"
    var wids = reloaded.encode(wrapped)
    if wids[0] != IMDB_CLS_ID or wids[len(wids) - 1] != IMDB_SEP_ID:
        raise Error("imdb_bert_vocab: CLS/SEP boundary ids wrong")
    if reloaded.decode(wids) != wrapped:
        raise Error("imdb_bert_vocab: specials decode round-trip failed")
    print("specials: [CLS]/[SEP] wrap exact,", len(wids), "ids")

    # ---- Stats: compression on the unseen test split + train lengths --
    print("reading test split (compression stats only)...")
    var test_pair = read_imdb_split(TEST_DIR)
    var test_texts = test_pair[0].copy()
    var test_blob = String("")
    for ti in range(len(test_texts)):
        var t = test_texts[ti]
        test_blob += t
        test_blob += "\n"
    var test_ids = reloaded.encode(test_blob)
    print(
        "test compression:",
        Float64(test_blob.byte_length()) / Float64(len(test_ids)),
        "chars/token over",
        len(test_ids),
        "tokens",
    )
    # Length stats over a 2k-review sample (full 25k encode is MLM-loop
    # work, not vocab work — the sample sizes T=128 here).
    var len_probe = List[Int]()
    var sample_n = min(2000, len(train_texts))
    for s in range(sample_n):
        var stext = train_texts[s]
        var rids = reloaded.encode(stext)
        len_probe.append(len(rids) + 2)  # +CLS/SEP, per-encode shape
    log_corpus_stats(len_probe, 128)
    print("imdb_bert_vocab passed: train -> specials -> save -> reload.")
