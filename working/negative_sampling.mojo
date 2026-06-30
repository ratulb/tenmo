from tenmo.shapes import Shape
from std.math import sqrt
from tenmo.tensor import Tensor
from std.pathlib import Path
from std.python import Python, PythonObject
from std.sys import has_accelerator
from std.os.process import Process
from std.random import shuffle, random_float64, seed
from tenmo.mnemonics import mv
from tenmo.shared import Reduction
from tenmo.filler import Filler
from tenmo.intarray import IntArray
from std.collections import Set
#from bpe import BasicTokenizer
from std.time import perf_counter_ns

comptime STOPWORDS = """
{
    'the','a','an','and','or','but','in','on','at','to',
    'for','of','with','by','from','is','it','as','be',
    'was','are','were','has','had','have','this','that',
    'i','he','she','they','we','you','his','her','their',
    'its','my','our','your','so','do','did','not','no',
    'if','up','out','about','than','into','then','there',
    'what','which','who','how','when','where','will','would',
    'could','should','may','might','also','just','been','after',
    'before','more','all','one','can','get','got','him','them'
}
"""
comptime UNK = "<|unk|>"


@fieldwise_init
struct Tokenizer(Sized & ImplicitlyCopyable & Movable):
    var str_to_int: Dict[String, Int]
    var int_to_str: Dict[Int, String]

    def __init__(
        out self,
        var vocab: Dict[String, Int],
    ) raises:
        self.int_to_str = {item.value: item.key for item in vocab.items()}
        self.str_to_int = vocab^

    def __init__(out self, *, copy: Self):
        self.int_to_str = copy.int_to_str.copy()
        self.str_to_int = copy.str_to_int.copy()

    def __init__(out self, *, deinit take: Self):
        self.int_to_str = take.int_to_str^
        self.str_to_int = take.str_to_int^

    @staticmethod
    def clean_text(
        line: String,
    ) raises -> PythonObject:
        var py = Python.import_module("builtins")
        var re = Python.import_module("re")

        var py_str = py.str(line)

        # Strip HTML tags — <br />, <p>, <a href="...">, </div> etc.
        #    [^>]+ matches one or more characters that are not >
        py_str = re.sub(r"<[^>]+>", " ", py_str)

        # Strip URLs — http://example.com or www.example.com
        #    \S+ matches any non-whitespace run after http or www.
        py_str = re.sub(r"http\S+|www\.\S+", " ", py_str)

        # Strip digit sequences — 2024, 10, 3rd etc.
        #    \d+ matches one or more consecutive digit characters
        py_str = re.sub(r"\d+", " ", py_str)

        # Strip stray apostrophes not part of a contraction.
        # Preserves: don't, it's, they're (apostrophe between letters)
        # Removes:   'hello' (leading), dogs' (trailing)
        py_str = re.sub(r"(?<!\w)'|'(?!\w)", " ", py_str)

        # Collapse multiple spaces/newlines → single space, strip ends
        py_str = re.sub(r"\s+", " ", py_str).strip()

        # Split and filter — drop words shorter than 2 characters.
        var filter_fn = Python.evaluate(
            "lambda words: [w for w in words.split() if len(w) >= 2]"
        )
        var words = filter_fn(py_str)

        _ = """var stopwords = Python.evaluate(STOPWORDS)
        filter_fn = Python.evaluate(
            "lambda words, sw: [w for w in words if w not in sw]"
        )

        return filter_fn(words, stopwords)"""
        return words

    @staticmethod
    def from_text_lines(
        lines: List[String],
    ) raises -> Self:
        var py = Python.import_module("builtins")
        var re = Python.import_module("re")
        var unique_words: PythonObject = []

        for line in lines:
            unique_words.extend(Self.clean_text(line))

        unique_words = py.list(py.set(unique_words))
        unique_words = py.sorted(unique_words)
        var extension: PythonObject = [UNK]
        unique_words.extend(extension)
        var vocab = {
            String(token): Int(index)
            for index, token in enumerate(unique_words.__iter__())
        }
        return Self(vocab^)

    def encode(self, text: String) raises -> List[Int]:
        var tokens = Self.clean_text(text)
        var token_ids = List[Int](capacity=len(tokens))
        for token in tokens:
            var token_str = String(token)
            token_ids.append(
                self.str_to_int[token_str] if token_str
                in self.str_to_int else self.str_to_int[UNK]
            )
        return token_ids^

    def decode(self, token_ids: List[Int]) raises -> String:
        var text = " ".join([self.int_to_str[id] for id in token_ids])
        return text^

    def __len__(self) -> Int:
        return len(self.int_to_str)


# IMDB dataset loader
struct NegativeSampler:
    var input_dataset: List[List[Int]]
    var concatenated: List[Int]

    def __init__(out self):
        self.input_dataset = List[List[Int]](capacity=50000)
        self.concatenated = List[Int](capacity=5000000)

    def init_tokenizer_and_datasets(
        mut self, folder_path: String
    ) raises -> Tokenizer:
        """Load reviews from pos/neg folders."""
        download()
        var pos_path = Path("/tmp") / folder_path / "pos"
        var neg_path = Path("/tmp") / folder_path / "neg"

        var comments = List[String](capacity=50000)

        # Load positive reviews
        if pos_path.exists():
            for item in pos_path.listdir():
                var rating = self.extract_rating(item.name())
                if rating >= 7:  # Positive review threshold
                    var comment = pos_path.joinpath(item.name()).read_text()
                    comments.append(comment)

        # Load negative reviews
        if neg_path.exists():
            for item in neg_path.listdir():
                var rating = self.extract_rating(item.name())
                if rating <= 4:  # Negative review threshold
                    var comment = neg_path.joinpath(item.name()).read_text()
                    comments.append(comment)

        var tokenizer = Tokenizer.from_text_lines(comments)

        for comment in comments:
            var indices = tokenizer.encode(comment)

            # Debug: check immediately after encode
            var max_id = 0
            for id in indices:
                if id > max_id:
                    max_id = id
            if max_id >= len(tokenizer):
                print("BAD ENCODE! max_id:", max_id, "vocab:", len(tokenizer))
                print("Offending comment:", comment[byte=0:50])
                break

            if len(indices) == 0:
                continue
            self.input_dataset.append(indices.copy())
            self.concatenated.extend(indices^)

        # Sanity check — all word IDs must be < vocab_size
        var vocab_size = len(tokenizer)
        var break_out = False
        for review in self.input_dataset:
            if break_out:
                break
            for word_id in review:
                if word_id >= vocab_size:
                    print("BAD ID:", word_id, "for vocab size: ", vocab_size)
                    break_out = True
                    break
        # shuffle(self.input_dataset)

        return tokenizer

    def extract_rating(self, filename: String) -> Int:
        """Extract rating from filename like '9992_10.txt'."""
        var undscore_sep = StringSlice("_")
        var dot_sep = StringSlice(".txt")
        var split = filename.split(undscore_sep)
        if len(split) >= 2:
            var rating_str = split[1].split(dot_sep)[0]
            try:
                return Int(rating_str)
            except e:
                print(e)
                print("Error extracting rating")
                return 0
        return 0


def build_target_samples(
    ref review: List[Int],
    target_i: Int,
    ref concatenated: List[Int],
    negative: Int,
) -> List[Int]:
    var concat_len = Float64(len(concatenated))
    var samples = [
        concatenated[
            min(Int(random_float64() * concat_len), len(concatenated) - 1)
        ]
        for _ in range(negative)
    ]
    samples.insert(0, review[target_i])
    return samples^


comptime TRAIN_FOLDER = "aclImdb/train"
comptime LEARNING_RATE: Float32 = 0.01
comptime ITERATIONS = 5
comptime HIDDEN_SIZE = 100
comptime TEST_SIZE = 1000  # last N reviews held out for testinddg
comptime RANDOM_SEED_W01 = 42
comptime RANDOM_SEED_W12 = 24
comptime window, negative = (3, 5)
comptime MAX_REVIEWS = 5000  # train only first N reviews for speed


def main() raises:
    comptime dtype = DType.float32
    var sys = Python.import_module("sys")
    seed()
    # ── Device detection ──────────────────────────────────────────────────────
    comptime if has_accelerator():
        print("Device: GPU")
    else:
        print("Device: CPU")

    # ── Load and preprocess ───────────────────────────────────────────────────
    print("Loading reviews from:", TRAIN_FOLDER)
    var sampler = NegativeSampler()
    var tokenizer = sampler.init_tokenizer_and_datasets(TRAIN_FOLDER)
    ref input_dataset = sampler.input_dataset
    ref concatenated = sampler.concatenated
    print("Vocab size: ", len(tokenizer))

    # ── Weight initialisation ─────────────────────────────────────────────────
    # weights_0_1: random uniform [-0.1, 0.1]  (vocab, hidden) — input embeddings
    # weights_1_2: zeros                       (vocab, hidden) — output embeddings
    # layer_2_target: [1, 0, 0, ..., 0]       (negative+1,)   — positive at idx 0
    var vocab_size = len(tokenizer)
    var weights_0_1 = Tensor[dtype].rand(
        Shape(vocab_size, HIDDEN_SIZE),
        min=-0.1,
        max=0.1,
        init_seed=RANDOM_SEED_W01,
    )
    var weights_1_2 = Tensor[dtype].zeros(
        Shape(vocab_size, HIDDEN_SIZE),
    )
    var layer_2_target = Tensor[dtype].zeros(negative + 1)
    layer_2_target[0] = 1

    comptime if has_accelerator():
        weights_0_1 = weights_0_1.to_gpu()
        weights_1_2 = weights_1_2.to_gpu()
        layer_2_target = layer_2_target.to_gpu()

    var num_reviews = min(MAX_REVIEWS, len(input_dataset))
    print("Training configuration:")
    print("  Iterations:   ", ITERATIONS)
    print("  Hidden size:  ", HIDDEN_SIZE)
    print("  Learning rate:", LEARNING_RATE)
    print("  Vocab size:   ", vocab_size)
    print("  Reviews:      ", num_reviews, "of", len(input_dataset))

    # Quick sanity: capture initial weights sum to verify updates later
    var initial_wsum = weights_0_1.sum[track_grad=False]().item()

    # ── Training loop (manual backprop, no autograd) ──────────────────────────
    # Skip-gram with negative sampling (Grokking DL Algorithm 12-1).
    # Each review: for each word as target, average its context embeddings,
    # score against target + negative samples, compute deltas manually,
    # scatter-add updates via Filler (sparse row ops, GPU-accelerated kernel).
    # ──────────────────────────────────────────────────────────────────────────
    var _similar_embeddings: Tensor[dtype]
    comptime if has_accelerator():
        _similar_embeddings = weights_0_1.to_cpu()
    else:
        _similar_embeddings = weights_0_1

    for iteration in range(ITERATIONS):
        shuffle(input_dataset)
        var num_reviews = min(MAX_REVIEWS, len(input_dataset))

        for rev_i in range(num_reviews):
            ref review = input_dataset[rev_i]
            for target_i in range(len(review)):
                var left = slice(max(0, target_i - window), target_i)
                var right = slice(
                    target_i + 1, min(len(review), target_i + window)
                )
                if left.start == left.end and right.start == right.end:
                    continue
                var context_indices = review[left].copy()
                context_indices.extend(review[right].copy())
                var context_len = len(context_indices)
                if context_len == 0:
                    continue

                var target_samples = build_target_samples(
                    review, target_i, concatenated, negative
                )

                # ── Forward ───────────────────────────────────────────────────
                var context_embed = weights_0_1.gather[track_grad=False](
                    context_indices, reduction=Reduction(1)
                )
                var layer_1 = context_embed / Float32(context_len)

                var target_embed = weights_1_2.gather[track_grad=False](
                    target_samples
                )
                var layer_2 = target_embed.matmul[mode=mv, track_grad=False](
                    layer_1
                ).sigmoid()

                # ── Backward (manual) ─────────────────────────────────────────
                var layer_2_delta = layer_2 - layer_2_target  # (n+1,)
                var layer_1_delta = target_embed.transpose().matmul[
                    mode=mv, track_grad=False
                ](
                    layer_2_delta
                )  # (h,)

                # ── Update weights_0_1: context word rows ─────────────────────
                # All context words get the same delta (layer_1_delta * lr).
                # Broadcast to (context_len, hidden) via repeat and scatter-add
                # once — avoids per-index scatter_add calls.
                var delta_01_raw = -layer_1_delta * Float32(LEARNING_RATE)
                var unsqueezed = delta_01_raw.unsqueeze(0)
                var delta_01 = unsqueezed.repeat[track_grad=False](
                    context_len, 1
                )
                var ctx_arr = IntArray(context_indices)
                Filler[dtype].scatter_add(
                    weights_0_1.buffer, delta_01.buffer, ctx_arr
                )

                # ── Update weights_1_2: target sample rows ────────────────────
                # outer(-layer_2_delta * lr, layer_1) → (n+1, hidden)
                var delta_12_raw = (
                    -layer_2_delta.unsqueeze(1)
                    * layer_1.unsqueeze(0)
                    * Float32(LEARNING_RATE)
                )
                var tgt_arr = IntArray(target_samples)
                Filler[dtype].scatter_add(
                    weights_1_2.buffer, delta_12_raw.buffer, tgt_arr
                )

            # Progress
            if (rev_i + 1) % 25 == 0:
                var pct = Float32(rev_i + 1) / Float32(num_reviews) * 100.0
                sys.stdout.write(
                    "\rIter:"
                    + String(iteration)
                    + "  Progress: "
                    + String(Int(pct))
                    + "%"
                )

        var final_wsum = weights_0_1.sum[track_grad=False]().item()
        var wsum_delta = final_wsum - initial_wsum
        print(
            "\n  weight sum change:",
            wsum_delta,
            "(should be != 0 — proves gradients flow)",
        )

        comptime if has_accelerator():
            _similar_embeddings = weights_0_1.to_cpu()
        else:
            _similar_embeddings = weights_0_1
        var neighbours = similar(tokenizer, _similar_embeddings, "terrible")
        print("Words similar to 'terrible':")
        for item in neighbours:
            print("  ", item[0], item[1])


def similar(
    tokenizer: Tokenizer,
    ref embeddings: Tensor[DType.float32],
    target: String = "beautiful",
    top_n: Int = 10,
) raises -> List[Tuple[String, Float32]]:
    var target_ids = tokenizer.encode(target)
    var target_vec = embeddings.gather[track_grad=False](target_ids)
    if len(target_ids) > 1:
        target_vec = target_vec.mean[track_grad=False](
            IntArray(0), keepdims=True
        )
    var diff = embeddings - target_vec
    var distances = (
        (diff * diff)
        .sum[track_grad=False](IntArray(1))
        .sqrt[track_grad=False]()
    )
    var result = List[Tuple[String, Float32]](capacity=len(tokenizer))
    for ref pair in tokenizer.str_to_int.items():
        var word = pair.key
        var index = pair.value
        if word == target or "_" in word:
            continue
        result.append((word, -distances[index]))
    sort[cmp_fn=compare](result)
    var top = List[Tuple[String, Float32]](capacity=min(top_n, len(result)))
    for k in range(min(top_n, len(result))):
        top.append(result[k])
    return top^


def compare(
    x: Tuple[String, Float32], y: Tuple[String, Float32]
) capturing -> Bool:
    return x[1] > y[1]


def download(
    url: String = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
) raises:
    var file_path = Path("/tmp/aclImdb_v1.tar.gz")
    var extracted_path = Path("/tmp/aclImdb/")

    if extracted_path.exists() and extracted_path.is_dir():
        print("Extracted folder", extracted_path.name(), "exists")
        return
    elif file_path.exists() and file_path.is_file():
        print(file_path.__fspath__(), " file exists - extracting")
        _ = Process.run("tar", ["-xzf", "/tmp/aclImdb_v1.tar.gz", "-C", "/tmp"])
    else:
        print("File or folder is not present - downloading")
        # var args = ["-P", "/tmp", url] # strangely this does not work!
        var download_to = "/tmp"
        var args = ["-P", download_to, url]
        _ = Process.run("wget", args)
        print("Downloaded - extracting now")
        _ = Process.run("tar", ["-xzf", "/tmp/aclImdb_v1.tar.gz", "-C", "/tmp"])

    print("done")
