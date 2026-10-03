"""
IMDB BERT data support: vocab constants, loader, order-preserving encoder.

How this file fits the pipeline:
`examples/imdb_bert_vocab.mojo` trains the 8k BPE and saves it to
`IMDB_VOCAB_PATH`; everything downstream — the data-builder test, the
MLM pretrain example, the classifier example — loads it through
`load_imdb_vocab()` here and never touches raw `load_tiktoken`.

Why does one helper own every load? The `.tiktoken` file format
silently DROPS special tokens on save (verified against the mbpe writer). A loader that forgets to re-register them
shifts every special id's meaning without any error. Centralizing
load-then-re-register in one function makes that bug impossible.
"""

from bpe.tokenizer import Tokenizers, BPETokenizer
from ..tensor import Tensor
from std.pathlib import Path
from std.os.process import Process

# The vocabulary contract, in one place. BPE training fills ids
# 0..7999 contiguously with NO room reserved for specials, so the four
# BERT markers are appended BEYOND the trained range (in-range
# registration would overwrite a real BPE entry). The model
# embedding tables must therefore be sized IMDB_MODEL_VOCAB (8004),
# and every `id < V` gate in the codebase means `id < 8004`.
comptime IMDB_VOCAB_SIZE = 8000
comptime IMDB_PAD_ID = 8000
comptime IMDB_CLS_ID = 8001
comptime IMDB_SEP_ID = 8002
comptime IMDB_MASK_ID = 8003
comptime IMDB_MODEL_VOCAB = 8004
comptime IMDB_VOCAB_PATH = "examples/data/imdb_8k.tiktoken"
comptime IMDB_MAX_LEN = 128


def register_imdb_specials(
    mut tok: BPETokenizer[Tokenizers.gpt2],
) raises:
    """Append the four BERT markers beyond the trained BPE range.

    Must run AFTER `train` (which resets the token table but keeps the
    specials dict) and AFTER every `load_tiktoken` (which restores
    neither). Idempotent per tokenizer instance — registering twice
    raises a duplicate-token error, which is intentional (a loud
    failure beats a silent semantic shift).
    """
    var specials = Dict[String, Int]()
    specials["[PAD]"] = IMDB_PAD_ID
    specials["[CLS]"] = IMDB_CLS_ID
    specials["[SEP]"] = IMDB_SEP_ID
    specials["[MASK]"] = IMDB_MASK_ID
    tok.register_special_tokens(specials)


def load_imdb_vocab(
    path: String = IMDB_VOCAB_PATH,
) raises -> BPETokenizer[Tokenizers.gpt2]:
    """The ONLY sanctioned way to load the IMDB BPE: file first, then
    re-register specials (risk R1). Every example uses this; nobody
    calls raw `load_tiktoken`."""
    var tok = BPETokenizer[Tokenizers.gpt2]()
    tok.load_tiktoken(path)
    register_imdb_specials(tok)
    return tok^


def encode_review(
    mut tok: BPETokenizer[Tokenizers.gpt2],
    text: String,
    max_len: Int = IMDB_MAX_LEN,
) raises -> Tuple[List[Int], Int]:
    """Order-preserving encode.
    `[CLS] ids [SEP]`, head-truncated,
    right-padded with `[PAD]` to exactly `max_len`.

    This is the anti-v1 function. The old
    `imdb_sentiment_v1.mojo` pushed ids through a `Set` (presence-only
    bag-of-words), which DESTROYS word order — "good, not bad" and
    "bad, not good" become identical. An encoder's whole point is
    reading order, so every id stays in place here.

    Returns `(padded_ids, true_len)` where `true_len` counts the real
    tokens including `[CLS]`/`[SEP]` but excluding padding — feed it
    straight into `make_padding_mask`.
    """
    var raw = tok.encode(text)
    var out = List[Int]()
    out.append(IMDB_CLS_ID)
    # Head truncation: keep the FIRST (max_len - 2) ids. Documented
    # first cut; length stats from the vocab example decide
    # whether tail-truncation is worth revisiting.
    var keep = min(len(raw), max_len - 2)
    for k in range(keep):
        out.append(raw[k])
    out.append(IMDB_SEP_ID)
    var true_len = len(out)
    while len(out) < max_len:
        out.append(IMDB_PAD_ID)
    return (out^, true_len)


def materialize_batch(
    encoded: List[List[Int]],
    labels: List[Int],
    max_len: Int,
) raises -> Tuple[Tensor[DType.int64], Tensor[DType.int64], Tensor[DType.int64]]:
    """Stack padded rows into `(N,T)[int64]` ids, `(N,)[int64]` labels,
    and `(N,)[int64]` true lengths (non-`[PAD]` counts, `make_padding_mask`
    -ready).

    `Tensor.d2` needs `List[Scalar[int64]]` rows, but the
    tokenizer speaks plain `Int` — hence the explicit per-element
    conversion loop. Ragged rows raise instead of silently stacking:
    every row must already be exactly `max_len` (that is
    `encode_review`'s contract).
    """
    if len(encoded) != len(labels):
        raise Error("materialize_batch: ids/labels count mismatch")
    var rows = List[Tensor[DType.int64].Row]()
    var lengths = List[Scalar[DType.int64]]()
    var flat_labels = List[Scalar[DType.int64]]()
    for r in range(len(encoded)):
        ref row = encoded[r]
        if len(row) != max_len:
            raise Error("materialize_batch: ragged row (pad first)")
        var conv = List[Scalar[DType.int64]](capacity=max_len)
        var true_len = 0
        for k in range(len(row)):
            conv.append(Scalar[DType.int64](row[k]))
            if row[k] != IMDB_PAD_ID:
                true_len += 1
        rows.append(conv^)
        lengths.append(Scalar[DType.int64](true_len))
        flat_labels.append(Scalar[DType.int64](labels[r]))
    return (
        Tensor[DType.int64].d2(rows^),
        Tensor[DType.int64].d1(flat_labels^),
        Tensor[DType.int64].d1(lengths^),
    )


def _lt(x: Int, y: Int) -> Bool:
    return x < y


def lengths_of(xb: Tensor[DType.int64]) -> Tensor[DType.int64]:
    """Per-row non-`[PAD]` counts of an `(N,T)` id batch.

    the dataset stores ids + sentiment labels; lengths are
    DERIVED from the ids (a row is `[CLS] ... [SEP] [PAD]*`, so
    non-PAD count == true length) instead of carried as a third
    tensor — one fewer thing that can disagree. Feed straight into
    `make_padding_mask`.
    """
    var nb = xb.shape()[0]
    var nt = xb.shape()[1]
    var out = List[Scalar[DType.int64]](capacity=nb)
    for b in range(nb):
        var L = 0
        for t in range(nt):
            if Int(xb[b, t]) != IMDB_PAD_ID:
                L += 1
        out.append(Scalar[DType.int64](L))
    return Tensor[DType.int64].d1(out^)


def ensure_aclImdb() raises:
    """Fetch + extract aclImdb to /tmp once (idempotent).

    the 84MB tarball lives OUTSIDE the repo (too big for git)
    at a fixed `/tmp` path — every IMDB example calls this first, so a
    fresh machine self-provisions on first run and skips after.
    """
    var extracted = Path("/tmp/aclImdb")
    var tarball = Path("/tmp/aclImdb_v1.tar.gz")
    if extracted.exists() and extracted.is_dir():
        return
    if not (tarball.exists() and tarball.is_file()):
        print("downloading aclImdb (~84MB)...")
        var args = List[String]()
        args.append("-P")
        args.append("/tmp")
        args.append(
            "https://huggingface.co/datasets/NolanChai/aclImdb_v1/resolve/main/aclImdb_v1.tar.gz"
        )
        _ = Process.run("wget", args^)
    print("extracting aclImdb...")
    _ = Process.run("tar", ["-xzf", "/tmp/aclImdb_v1.tar.gz", "-C", "/tmp"])


def read_imdb_split(dir: String) raises -> Tuple[List[String], List[Int]]:
    """Read every review under `<dir>/{pos,neg}` (order-preserving).

    Returns `(texts, labels)` with 1 = positive, 0 = negative. No
    shuffling here — the LOADER shuffles per epoch, so the on-disk
    order stays a stable frame of reference (and the frozen test split
    is never shuffled at all).
    """
    var texts = List[String]()
    var labels = List[Int]()
    var subs = List[String]()
    subs.append("pos")
    subs.append("neg")
    for si in range(len(subs)):
        var folder = Path(dir) / subs[si]
        var items = folder.listdir()
        for ii in range(len(items)):
            var name = items[ii].name()
            var text = folder.joinpath(name).read_text()
            texts.append(text^)
            labels.append(1 if si == 0 else 0)
    return (texts^, labels^)


def log_corpus_stats(lengths: List[Int], max_len: Int):
    """Log length percentiles, truncation rate, and pad fraction.

    These are the numbers the old v1/v2 examples never printed —
    without them you cannot tell whether T=128 truncates
    half the corpus or pads 90% air. `lengths` holds per-review
    true_lens (CLS/SEP included, PAD excluded).
    """
    var n = len(lengths)
    if n == 0:
        print("log_corpus_stats: empty corpus")
        return
    var ordered = lengths.copy()
    sort(ordered, _lt)
    var total = 0
    var total_kept = 0
    var truncated = 0
    for L in ordered:
        total += L
        if L > max_len:
            truncated += 1
            total_kept += max_len
        else:
            total_kept += L
    print("reviews:", n)
    print(
        "len p50/median:",
        ordered[n // 2],
        " p90:",
        ordered[(n * 9) // 10],
        " p99:",
        ordered[(n * 99) // 100],
        " max:",
        ordered[n - 1],
    )
    print("mean len:", Float64(total) / Float64(n))
    print(
        "truncated at T=",
        max_len,
        ":",
        truncated,
        " (",
        Float64(truncated) / Float64(n),
        ")",
    )
    print(
        "pad fraction at T=",
        max_len,
        ":",
        1.0 - Float64(total_kept) / Float64(n * max_len),
        "(post-truncation: truncated tokens are cut, not padded)",
    )
    print("UNK rate: 0.0 by construction (byte-level BPE has no UNK)")
