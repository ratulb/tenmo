from tenmo.shapes import Shape
from std.math import sqrt
from tenmo.tensor import Tensor
from std.pathlib import Path
from std.python import Python, PythonObject
from std.sys import has_accelerator
from std.os.process import Process
from std.random import shuffle, random_float64, seed
from tenmo.mnemonics import mv
from tenmo.filler import Filler
from tenmo.intarray import IntArray
from std.collections import Set
from bpe import BasicTokenizer
from std.time import perf_counter_ns
from tenmo.embedding import Embedding
from tenmo.shared import Reduction
from tenmo.sgd import SGD
from tenmo.net import BCEWithLogitsLoss

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

    def __copyinit__(out self, copy: Self):
        self.int_to_str = copy.int_to_str.copy()
        self.str_to_int = copy.str_to_int.copy()

    def __moveinit__(out self, deinit take: Self):
        self.int_to_str = take.int_to_str^
        self.str_to_int = take.str_to_int^

    @staticmethod
    def clean_text(
        line: String,
    ) raises -> PythonObject:
        var py = Python.import_module("builtins")
        var re = Python.import_module("re")

        var py_str = py.str(line)

        py_str = re.sub(r"<[^>]+>", " ", py_str)
        py_str = re.sub(r"http\S+|www\.\S+", " ", py_str)
        py_str = re.sub(r"\d+", " ", py_str)
        py_str = re.sub(r"(?<!\w)'|'(?!\w)", " ", py_str)
        py_str = re.sub(r"\s+", " ", py_str).strip()

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


struct NegativeSampler:
    var input_dataset: List[List[Int]]
    var concatenated: List[Int]

    def __init__(out self):
        self.input_dataset = List[List[Int]](capacity=50000)
        self.concatenated = List[Int](capacity=5000000)

    def init_tokenizer_and_datasets(
        mut self, folder_path: String
    ) raises -> Tokenizer:
        download()
        var pos_path = Path("/tmp") / folder_path / "pos"
        var neg_path = Path("/tmp") / folder_path / "neg"

        var comments = List[String](capacity=50000)

        if pos_path.exists():
            for item in pos_path.listdir():
                var rating = self.extract_rating(item.name())
                if rating >= 7:
                    var comment = pos_path.joinpath(item.name()).read_text()
                    comments.append(comment)

        if neg_path.exists():
            for item in neg_path.listdir():
                var rating = self.extract_rating(item.name())
                if rating <= 4:
                    var comment = neg_path.joinpath(item.name()).read_text()
                    comments.append(comment)

        var tokenizer = Tokenizer.from_text_lines(comments)

        for comment in comments:
            var indices = tokenizer.encode(comment)

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
        shuffle(self.input_dataset)

        return tokenizer

    def extract_rating(self, filename: String) -> Int:
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
comptime LEARNING_RATE: Float32 = 0.02
comptime ITERATIONS = 5
comptime HIDDEN_SIZE = 100
comptime RANDOM_SEED_W01 = 42
comptime RANDOM_SEED_W12 = 24
comptime window, negative = (4, 6)
comptime MAX_REVIEWS = 10000


def main() raises:
    comptime dtype = DType.float32
    var sys = Python.import_module("sys")
    seed()

    comptime if has_accelerator():
        print("Device: GPU")
    else:
        print("Device: CPU")

    print("Loading reviews from:", TRAIN_FOLDER)
    var sampler = NegativeSampler()
    var tokenizer = sampler.init_tokenizer_and_datasets(TRAIN_FOLDER)
    ref input_dataset = sampler.input_dataset
    ref concatenated = sampler.concatenated
    var vocab_size = len(tokenizer)
    print("Vocab size: ", vocab_size)

    var num_reviews = min(MAX_REVIEWS, len(input_dataset))
    print("Training configuration:")
    print("  Iterations:   ", ITERATIONS)
    print("  Hidden size:  ", HIDDEN_SIZE)
    print("  Learning rate:", LEARNING_RATE)
    print("  Vocab size:   ", vocab_size)
    print("  Reviews:      ", num_reviews, "of", len(input_dataset))
    print("  Optimizer:    SGD with momentum=0.9")
    print("  Loss:         BCEWithLogits (autograd)")

    # ── Embedding + Optimizer setup ────────────────────────────────────────────
    # Input embeddings: kaiming init (~N(0, 0.14)) — close to original U(-0.1, 0.1)
    # Output embeddings: zero init — matches original
    var emb_input = Embedding[dtype](
        vocab_size, HIDDEN_SIZE,
        init_method="kaiming",
        init_seed=RANDOM_SEED_W01,
        reduction=Reduction(1),
    )
    var emb_output = Embedding[dtype](
        vocab_size, HIDDEN_SIZE,
        init_method="zero",
    )
    var layer_2_target = Tensor[dtype].zeros(negative + 1)
    layer_2_target[0] = 1

    comptime if has_accelerator():
        emb_input = emb_input.to_gpu()
        emb_output = emb_output.to_gpu()
        layer_2_target = layer_2_target.to_gpu()

    # Two optimizers — each embedding with its own sparse row indices
    var opt_in = SGD[dtype=dtype](
        emb_input.parameters(), lr=LEARNING_RATE, momentum=0.9
    )
    var opt_out = SGD[dtype=dtype](
        emb_output.parameters(), lr=LEARNING_RATE, momentum=0.9
    )

    # ── Quick gradient flow check ──────────────────────────────────────────────
    var init_sum_in = emb_input.weight.sum[track_grad=False]().item()
    var init_sum_out = emb_output.weight.sum[track_grad=False]().item()
    print("Initial emb_input.weight sum:", init_sum_in)
    print("Initial emb_output.weight sum:", init_sum_out)

    # ── Snapshot for similar() ─────────────────────────────────────────────────
    var _similar_embeddings: Tensor[dtype]
    comptime if has_accelerator():
        _similar_embeddings = emb_input.weight.to_cpu()
    else:
        _similar_embeddings = emb_input.weight

    # ── Training loop (autograd + SGD) ─────────────────────────────────────────
    for iteration in range(ITERATIONS):
        shuffle(input_dataset)
        shuffle(input_dataset)

        for rev_i in range(num_reviews):
            ref review = input_dataset[rev_i]
            var loss: Tensor[dtype] = Tensor[dtype].scalar(0)
            for target_i in range(len(review)):
                var left = slice(max(0, target_i - window), target_i)
                var right = slice(target_i + 1, min(len(review), target_i + window))
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

                # ── Forward (autograd tracked) ─────────────────────────────────
                var context_embed = emb_input(context_indices)
                if rev_i == 0 and target_i == 0:
                    print("context_embed shape: ", context_embed.shape())
                var layer_1 = context_embed

                var target_embed = emb_output(target_samples)
                if rev_i == 0 and target_i == 0:
                    print("target_embed shape: ", target_embed.shape())

                var logits = target_embed.matmul[
                    mode=mv
                ](layer_1)
                if rev_i == 0 and target_i == 0:
                    print("logits shape: ", logits.shape())

                loss = BCEWithLogitsLoss.forward[track_grad=True](
                    logits, layer_2_target
                )
                if rev_i == 0 and target_i == 0:
                    print("loss shape: ", loss.shape())

                # ── Backward + sparse optimizer step ───────────────────────────
                # loss is non-scalar — backward computes d(sum(loss))/d(leaf)
                loss.backward()
                opt_in.step(IntArray(context_indices))
                opt_in.zero_grad(IntArray(context_indices))
                opt_out.step(IntArray(target_samples))
                opt_out.zero_grad(IntArray(target_samples))

            # Progress
            if (rev_i + 1) % 25 == 0:
                var pct = (
                    Float32(rev_i + 1) / Float32(num_reviews) * 100.0
                )
                sys.stdout.write(
                    "\rIter:"
                    + String(iteration)
                    + "  Progress: "
                    + String(Int(pct))
                    + "%"
                    + "loss: "
                    + String(loss.item())
                )

        var final_sum_in = emb_input.weight.sum[track_grad=False]().item()
        var final_sum_out = emb_output.weight.sum[track_grad=False]().item()
        print("Final emb_input.weight sum:", final_sum_in, "(delta:", final_sum_in - init_sum_in, ")")
        print("Final emb_output.weight sum:", final_sum_out, "(delta:", final_sum_out - init_sum_out, ")")

        print()
        comptime if has_accelerator():
            _similar_embeddings = emb_input.weight.to_cpu()
        else:
            _similar_embeddings = emb_input.weight
        var neighbours = similar(
            tokenizer, _similar_embeddings, "terrible"
        )
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
    var distances = (diff * diff).sum[track_grad=False](
        IntArray(1)
    ).sqrt[track_grad=False]()
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
        var download_to = "/tmp"
        var args = ["-P", download_to, url]
        _ = Process.run("wget", args)
        print("Downloaded - extracting now")
        _ = Process.run("tar", ["-xzf", "/tmp/aclImdb_v1.tar.gz", "-C", "/tmp"])

    print("done")
