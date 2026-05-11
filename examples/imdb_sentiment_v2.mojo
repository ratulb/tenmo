from tenmo.shapes import Shape
from std.math import abs, sqrt
from tenmo.sgd import SGD
from std.random import seed
from tenmo.tensor import Tensor
from std.pathlib import Path
from std.collections import Set
from std.python import Python
from std.sys import has_accelerator
from std.os.process import Process
from tenmo.common_utils import (
    SimpleTokenizer,
    DEFAULT_SPLITTER,
    DEFAULT_SUBSTITUTION,
    DEFAULT_UNK,
    END_OF_TEXT,
)


@fieldwise_init
struct Review(ImplicitlyCopyable, Movable):
    var rating: Int
    var comment: String


# IMDB dataset loader
struct IMDBPreprocessor:
    var reviews: Optional[List[Review]]

    fn __init__(out self):
        self.reviews = Optional(List[Review](capacity=50000))

    fn load_from_folder(mut self, folder_path: String) raises:
        """Load reviews from pos/neg folders."""
        download()
        var pos_path = Path("/tmp") / folder_path / "pos"
        var neg_path = Path("/tmp") / folder_path / "neg"
        ref reviews = self.reviews.value()

        # Load positive reviews
        if pos_path.exists():
            for item in pos_path.listdir():
                var rating = self.extract_rating(item.name())
                if rating >= 7:  # Positive review threshold
                    var comment = pos_path.joinpath(item.name()).read_text()
                    reviews.append(Review(rating, comment))

        # Load negative reviews
        if neg_path.exists():
            for item in neg_path.listdir():
                var rating = self.extract_rating(item.name())
                if rating <= 4:  # Negative review threshold
                    var comment = neg_path.joinpath(item.name()).read_text()
                    reviews.append(Review(rating, comment))

    fn extract_rating(self, filename: String) -> Int:
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

    fn get_labels(self) -> List[Int]:
        """Extract labels (1 for positive, 0 for negative)."""
        var labels = List[Int]()
        for review in self.reviews.value():
            labels.append(1 if review.rating >= 7 else 0)
        return labels^

    fn init_tokenizer(
        self,
    ) raises -> SimpleTokenizer[
        DEFAULT_SPLITTER, DEFAULT_SUBSTITUTION, DEFAULT_UNK, END_OF_TEXT
    ]:
        """Initialize tokenizer from review content."""
        var lines = [
            StringSlice(review.comment) for review in self.reviews.value()
        ]
        return SimpleTokenizer.from_text_lines_min_freq(lines^)

    fn build_datasets(
        self,
        tokenizer: SimpleTokenizer,
        shuffle: Bool = True,
        random_seed: Int = 42,
    ) -> Tuple[List[List[Int]], List[Int]]:
        """Build input_dataset and target_dataset from loaded reviews.

        Args:
            tokenizer: Already initialised tokenizer.
            shuffle: Whether to shuffle datasets.
            random_seed: An integer value used in shuffle.

        Returns:
            Tuple of (input_dataset, target_dataset).
            input_dataset:  one List[Int] of deduplicated token ids per review.
            target_dataset: 1 for positive (rating>=7), 0 for negative (rating<=4).
        """
        var input_dataset = List[List[Int]]()
        var target_dataset = List[Int]()

        for review in self.reviews.value():
            try:
                var ids = tokenizer.encode(review.comment)
                var seen = Set[Int]()
                var deduped = List[Int]()
                for id in ids:
                    if id not in seen:
                        seen.add(id)
                        deduped.append(id)
                input_dataset.append(deduped^)
            except:
                # If encoding fails for any review, append empty to keep alignment
                input_dataset.append(List[Int]())

            target_dataset.append(1 if review.rating >= 7 else 0)

        # Shuffle once after all reviews are encoded
        if shuffle:
            self.shuffle_datasets(input_dataset, target_dataset, random_seed)

        return (input_dataset^, target_dataset^)

    def shuffle_datasets(
        self,
        mut input_dataset: List[List[Int]],
        mut target_dataset: List[Int],
        random_seed: Int = 42,
    ):
        """Shuffle input_dataset and target_dataset(labels) together."""
        var rng_seed = random_seed
        for idx in range(len(input_dataset) - 1, 0, -1):
            rng_seed = (rng_seed * 1664525 + 1013904223) % (2**31)
            var j = rng_seed % (idx + 1)
            input_dataset.swap_elements(idx, j)
            target_dataset.swap_elements(idx, j)


comptime TRAIN_FOLDER = "aclImdb/train"
comptime LEARNING_RATE: Float32 = 0.01
comptime ITERATIONS = 3
comptime HIDDEN_SIZE = 100
comptime TEST_SIZE = 1000  # last N reviews held out for testing
comptime RANDOM_SEED_W01 = 42
comptime RANDOM_SEED_W12 = 24


def main() raises:
    comptime dtype = DType.float32
    var sys = Python.import_module("sys")

    # ── Device detection ──────────────────────────────────────────────────────
    comptime if has_accelerator():
        print("Device: GPU")
    else:
        print("Device: CPU")

    # ── Load and preprocess ───────────────────────────────────────────────────
    print("Loading reviews from:", TRAIN_FOLDER)
    var preprocessor = IMDBPreprocessor()
    preprocessor.load_from_folder(TRAIN_FOLDER)

    print("Building vocabulary...")
    var tokenizer = preprocessor.init_tokenizer()
    var vocab_size = len(tokenizer)
    print("Vocabulary size:", vocab_size)

    print("Encoding reviews...")
    var datasets = preprocessor.build_datasets(
        tokenizer, shuffle=True, random_seed=42
    )
    ref (token_id_sets, labels) = datasets

    # Free up all the review strings because we are done with them
    _ = preprocessor.reviews.take()
    print("Reviews loaded:", len(token_id_sets))
    print("Labels  loaded:", len(labels))

    # ── Weight initialisation ─────────────────────────────────────────────────
    # Weights are initialised on CPU then moved to GPU if available.
    # The optimizer holds UnsafePointers to the original Tensor variables —
    # moving to GPU updates the tensor's internal buffer in-place so the
    # pointers remain valid.
    var weights_0_1 = Tensor[dtype].rand(
        Shape(vocab_size, HIDDEN_SIZE),
        min=-0.1,
        max=0.1,
        init_seed=RANDOM_SEED_W01,
        requires_grad=True,
    )
    var weights_1_2 = Tensor[dtype].rand(
        Shape(HIDDEN_SIZE),
        min=-0.1,
        max=0.1,
        init_seed=RANDOM_SEED_W12,
        requires_grad=True,
    )

    comptime if has_accelerator():
        weights_0_1 = weights_0_1.to_gpu()
        weights_1_2 = weights_1_2.to_gpu()

    var optimizer = SGD(
        parameters=[
            UnsafePointer(to=weights_0_1)
            .mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin](),
            UnsafePointer(to=weights_1_2)
            .mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin](),
        ],
        lr=Scalar[dtype](LEARNING_RATE),
        momentum=Scalar[dtype](0.9),
    )

    var train_size = len(token_id_sets) - TEST_SIZE

    print("Training configuration:")
    print("  Iterations:    ", ITERATIONS)
    print("  Hidden size:   ", HIDDEN_SIZE)
    print("  Learning rate: ", LEARNING_RATE)
    print("  Train samples: ", train_size)
    print("  Test samples:  ", TEST_SIZE)

    # ── Training loop ─────────────────────────────────────────────────────────
    for iteration in range(ITERATIONS):
        var num_correct = 0
        var num_seen = 0

        for sample_idx in range(train_size):
            ref token_ids = token_id_sets[sample_idx]

            # target is a scalar tensor — move to GPU if needed
            var target = Tensor[dtype].scalar(Float32(labels[sample_idx]))
            comptime if has_accelerator():
                target = target.to_gpu()

            # ── Forward pass ──────────────────────────────────────────────────
            # gather: (n_tokens, hidden_size)  — on same device as weights_0_1
            # sum:    (hidden_size,)
            # sigmoid:(hidden_size,)
            # dot:    scalar
            # sigmoid:scalar
            var hidden = (
                weights_0_1.gather(token_ids)
                .sum(axes=[0], keepdims=False)
                .sigmoid()
            )
            var prediction = hidden.dot(weights_1_2).sigmoid()

            # ── Loss — MSE ────────────────────────────────────────────────────
            var diff = prediction - target
            var loss = diff**2

            # ── Backward + optimizer step ─────────────────────────────────────
            # GatherBackward fires ScatterAddTensor — sparsely updates only
            # the embedding rows for tokens present in this review.
            # Filler.fill uses NDBuffer.get/set which are GPU-transparent.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ── Progress tracking ─────────────────────────────────────────────
            # diff.item() calls NDBuffer.get(0) — GPU-transparent
            if abs(diff.item()) < 0.5:
                num_correct += 1
            num_seen += 1

            if (sample_idx % 10) == 9:
                var pct = (
                    Float32(sample_idx) / Float32(len(token_id_sets)) * 100.0
                )
                var pct_whole = Int(pct)
                var pct_frac = Int((pct - Float32(pct_whole)) * 100.0)
                var decimal_str = ("0" if pct_frac < 10 else "") + String(
                    pct_frac
                )
                var train_acc = Float32(num_correct) / Float32(num_seen)
                sys.stdout.write(
                    "\rIter:"
                    + String(iteration)
                    + "  Progress: "
                    + String(pct_whole)
                    + "."
                    + decimal_str
                    + "%"
                    + "  Training Accuracy: "
                    + String(train_acc)
                )

        print()

    # ── Test loop ─────────────────────────────────────────────────────────────
    print("Evaluating on test set (" + String(TEST_SIZE) + " reviews)...")
    var num_correct = 0
    var num_seen = 0

    for sample_idx in range(train_size, len(token_id_sets)):
        ref token_ids = token_id_sets[sample_idx]
        ref true_label = labels[sample_idx]

        # No grad tracking needed for inference
        var hidden = (
            weights_0_1.gather[track_grad=False](token_ids)
            .sum[track_grad=False](axes=[0], keepdims=False)
            .sigmoid[track_grad=False]()
        )
        var prediction = hidden.dot[track_grad=False](weights_1_2).sigmoid[
            track_grad=False
        ]()

        # prediction.item() is GPU-transparent
        if abs(prediction.item() - Float32(true_label)) < 0.5:
            num_correct += 1
        num_seen += 1

    var test_accuracy = Float32(num_correct) / Float32(num_seen)
    print("Test Accuracy: " + String(test_accuracy))

    # ── Similarity search — verify embedding quality ──────────────────────────
    # Similarity search scans the entire vocabulary — bring weights to CPU
    # once to avoid per-word GPU round-trips.
    var embeddings_cpu: Tensor[dtype]
    comptime if has_accelerator():
        embeddings_cpu = weights_0_1.to_cpu()
    else:
        embeddings_cpu = weights_0_1

    var query_words = [
        "beautiful",
        "wonderful",
        "outstanding",
        "boring",
        "terrible",
    ]
    for target in query_words:
        print("\nTarget: ", target)
        var neighbours = similar(tokenizer, embeddings_cpu, target)
        for item in neighbours:
            print(" ", item[0], item[1])


def similar(
    tokenizer: SimpleTokenizer[_, _, _, _],
    ref embeddings: Tensor[DType.float32],
    target: String = "beautiful",
    top_n: Int = 10,
) raises -> List[Tuple[String, Float32]]:
    """Find top_n words with embeddings most similar to target.

    Args:
        tokenizer:  Trained tokenizer with str_to_int vocab.
        embeddings: CPU embedding tensor (vocab_size, hidden_size).
                    Caller must ensure this is on CPU — pass weights_0_1.to_cpu()
                    when running on GPU to avoid per-word device round-trips.
        target:     Word to find neighbours for.
        top_n:      Number of nearest neighbours to return.

    Returns:
        List of (word, negative_distance) sorted by descending score.
    """
    var target_ids = tokenizer.encode(target)
    var scores = Dict[String, Float32]()

    # Compute target embedding once outside the loop
    var target_row = embeddings.gather[track_grad=False](target_ids)

    for ref pair in tokenizer.str_to_int.items():
        var word = pair.key
        var index = pair.value
        if word == target:
            continue
        var raw_diff = target_row - embeddings.gather[track_grad=False]([index])
        scores[word] = -sqrt((raw_diff * raw_diff).sum().item())

    # Sort descending by score and return top_n
    var result = [(item.key, item.value) for item in scores.items()]
    sort[cmp_fn=compare](result)

    var top = List[Tuple[String, Float32]]()
    for k in range(min(top_n, len(result))):
        top.append(result[k])
    return top^


def compare(
    x: Tuple[String, Float32], y: Tuple[String, Float32]
) capturing -> Bool:
    return x[1] > y[1]


fn download(
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
