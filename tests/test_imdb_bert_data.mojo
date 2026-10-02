"""
Tests for the IMDB BERT data layer (`tenmo/nlp/bert_data.mojo`).

Deliberately hermetic: the real 8k vocab needs the network + 25k
reviews, so this suite trains a tiny BPE (300 entries) on a synthetic
corpus and exercises the SAME helpers the real pipeline uses
(`register_imdb_specials`, `load_imdb_vocab(path)`, `encode_review`,
`materialize_batch`). The real-artifact gates (reload id-equality,
compression) live in `examples/imdb_bert_vocab.mojo`, mirroring the
`tinystories_vocab.mojo` split between example-gates and CI tests.
"""

from std.testing import assert_true, TestSuite
from bpe.tokenizer import Tokenizers, BPETokenizer
from tenmo.encoder import make_padding_mask
from tenmo.nlp.bert_data import (
    IMDB_PAD_ID,
    IMDB_CLS_ID,
    IMDB_SEP_ID,
    register_imdb_specials,
    load_imdb_vocab,
    encode_review,
    materialize_batch,
    log_corpus_stats,
)

comptime TOY_VOCAB = 300
comptime TOY_PATH = "/tmp/imdb_bert_data_test.tiktoken"
comptime TOY_CORPUS = "the movie was great. the acting was terrible. a brilliant wonderful film. a boring dull story. great acting, great film. terrible story, dull movie."


def _toy_tokenizer() raises -> BPETokenizer[Tokenizers.gpt2]:
    """Train the toy BPE once per test (seconds, no network)."""
    var corpus = List[String]()
    corpus.append(String(TOY_CORPUS))
    var tok = BPETokenizer[Tokenizers.gpt2]()
    tok.train(Span[String](corpus), TOY_VOCAB)
    register_imdb_specials(tok)
    return tok^


def test_bert_specials_reregistered_after_load() raises:
    """The F11 footgun, pinned.
    Save drops specials, so a raw load
    encodes `[CLS]` as ordinary subwords — only `load_imdb_vocab`
    (load + re-register) restores the boundary ids."""
    var tok = _toy_tokenizer()
    tok.save_tiktoken(TOY_PATH)

    var reloaded = load_imdb_vocab(TOY_PATH)
    var ids = reloaded.encode("[CLS] hello [SEP]")
    assert_true(ids[0] == IMDB_CLS_ID)
    assert_true(ids[len(ids) - 1] == IMDB_SEP_ID)
    assert_true(reloaded.decode(ids) == "[CLS] hello [SEP]")


def test_bert_encode_review_shape_and_wrap() raises:
    """`encode_review` wraps `[CLS]..[SEP]`.
    Right-pads to exactly
    `max_len`, and reports the pre-pad length."""
    var tok = _toy_tokenizer()
    comptime T = 16
    var pair = encode_review(tok, "the movie was great", T)
    var ids = pair[0].copy()
    var true_len = pair[1]
    assert_true(len(ids) == T)
    assert_true(ids[0] == IMDB_CLS_ID)
    assert_true(ids[true_len - 1] == IMDB_SEP_ID)
    for k in range(true_len, T):
        assert_true(ids[k] == IMDB_PAD_ID)
    # true_len = CLS + 4 words (or fewer if merged) + SEP; bounded check
    assert_true(true_len >= 3 and true_len <= T)


def test_bert_encode_review_truncates_head() raises:
    """Over-long reviews keep the FIRST (T-2) content ids (head.
    truncation, doc §4.7) — never the tail, never an error."""
    var tok = _toy_tokenizer()
    comptime T = 8
    var long = String(
        "the movie was great and the acting was wonderful and brilliant"
    )
    var short = String("the movie was great and the acting was wonderful")
    var got_long = encode_review(tok, long, T)[0].copy()
    var got_short = encode_review(tok, short, T)[0].copy()
    assert_true(len(got_long) == T)
    # Both share the head: first T ids identical (CLS + first T-2 + SEP
    # once truncated to the same window).
    for k in range(T):
        assert_true(got_long[k] == got_short[k])


def test_bert_order_preserved_regression() raises:
    """Anti-F10 regression.
    Word-order swaps must change the id
    sequence (the v1 `Set`-dedup collapsed them to identical bags)."""
    var tok = _toy_tokenizer()
    var a = encode_review(tok, "great movie, not terrible", 16)[0].copy()
    var b = encode_review(tok, "terrible movie, not great", 16)[0].copy()
    var differ = False
    for k in range(len(a)):
        if a[k] != b[k]:
            differ = True
    assert_true(differ)


def test_bert_materialize_batch_shapes() raises:
    """`materialize_batch` stacks rows into `(N,T)` ids, `(N,)` labels.
    `(N,)` lengths; lengths count non-PAD (mask-ready)."""
    var tok = _toy_tokenizer()
    comptime T = 12
    var rows = List[List[Int]]()
    var labels = List[Int]()
    var r0 = encode_review(tok, "great film", T)
    rows.append(r0[0].copy())
    labels.append(1)
    var r1 = encode_review(tok, "dull story indeed", T)
    rows.append(r1[0].copy())
    labels.append(0)
    var batch = materialize_batch(rows^, labels^, T)
    ref x = batch[0]
    ref y = batch[1]
    ref lens = batch[2]
    assert_true(x.shape()[0] == 2 and x.shape()[1] == T)
    assert_true(y.shape()[0] == 2)
    assert_true(Int(lens[0]) == r0[1])
    assert_true(Int(lens[1]) == r1[1])
    assert_true(Int(y[0]) == 1 and Int(y[1]) == 0)


def test_bert_materialize_rejects_ragged() raises:
    """Ragged rows raise instead of silently stacking (pad-first.
    contract). Uses the panic-spawner discipline: this path is pure
    `Error`, so `assert_raises` sees it in-process."""
    var tok = _toy_tokenizer()
    var rows = List[List[Int]]()
    var full = encode_review(tok, "great film", 12)
    rows.append(full[0].copy())
    var short = List[Int]()
    short.append(IMDB_CLS_ID)
    short.append(IMDB_SEP_ID)
    rows.append(short^)
    var labels = List[Int]()
    labels.append(1)
    labels.append(0)
    var raised = False
    try:
        _ = materialize_batch(rows^, labels^, 12)
    except:
        raised = True
    assert_true(raised)


def test_bert_mask_integration() raises:
    """End-to-end builder→mask link.
    `make_padding_mask` over
    materialized lengths is True exactly on non-PAD columns, i.e. the
    padding the encoder ignores (via `forward_padded`) is exactly the
    padding the builder emitted."""
    var tok = _toy_tokenizer()
    comptime T = 12
    var rows = List[List[Int]]()
    var labels = List[Int]()
    var r0 = encode_review(tok, "great film", T)
    rows.append(r0[0].copy())
    labels.append(1)
    var r1 = encode_review(tok, "a boring dull story indeed", T)
    rows.append(r1[0].copy())
    labels.append(0)
    var batch = materialize_batch(rows^, labels^, T)
    ref lens_only = batch[2]
    var mask = make_padding_mask(lens_only, T)
    # Rebuild expectation from the raw rows (independent path).
    for r in range(2):
        var text = String("a boring dull story indeed")
        if r == 0:
            text = String("great film")
        var pair = encode_review(tok, text, T)
        var row = pair[0].copy()
        for t in range(T):
            var expect_live = row[t] != IMDB_PAD_ID
            assert_true(mask[r, t] == expect_live)


def test_bert_stats_smoke() raises:
    """`log_corpus_stats` runs without error on a small sample (output.
    is human-read; the gate is 'does not crash on real lengths')."""
    var lens = List[Int]()
    lens.append(5)
    lens.append(130)
    lens.append(12)
    log_corpus_stats(lens, 128)


def main() raises:
    """Run every `test_bert_*` function in this module."""
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll imdb_bert_data tests passed!")
