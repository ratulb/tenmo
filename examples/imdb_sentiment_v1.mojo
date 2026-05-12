"""
IMDB Sentiment Classifier
==================================================

A from-scratch two-layer neural network trained on the IMDB movie review
dataset for binary sentiment classification (positive / negative).

Architecture
────────────
    Input  : bag-of-words embedding lookup  (vocab_size → hidden_size)
    Layer 1: sum-pooled embedding + sigmoid activation  (hidden_size,)
    Layer 2: dot product + sigmoid activation           (scalar)

Training
────────
    Manual gradient computation — no autograd.
    SGD with sparse embedding updates: only the rows of weights_0_1
    corresponding to tokens present in the current review are updated.
    This mirrors the original NumPy implementation from Trask's
    "Grokking Deep Learning", Chapter 11.

Dataset
───────
    IMDB large movie review dataset (Maas et al., 2011).
    Folder structure expected:
        <folder>/pos/*.txt   — reviews rated 7–10
        <folder>/neg/*.txt   — reviews rated 1–4
    Download: https://ai.stanford.edu/~amaas/data/sentiment/

Usage
─────
    Adjust TRAIN_FOLDER, LEARNING_RATE, ITERATIONS, HIDDEN_SIZE as needed,
    then run:
        pix run mojo -I . imdb_sentiment.mojo
"""

from tenmo.tensor import Tensor
from tenmo.shapes import Shape
from tenmo.intarray import IntArray
from tenmo.nlp import (
    IMDBTextCleaner,
    DefaultTokenizer,
)
from tenmo.common_utils import (
    i,
    s,
)
from std.pathlib import Path
from std.math import sqrt
from std.python import Python
from std.collections import Set
from std.os.process import Process

# =============================================================================
# Review and dataset structures
# =============================================================================


@fieldwise_init
struct Review(ImplicitlyCopyable, Movable):
    """A single IMDB review with its numeric rating and raw comment text."""

    var rating: Int  # 1–10 as extracted from the filename
    var comment: String  # raw review text


struct IMDBPreprocessor:
    """Loads IMDB reviews from the standard folder layout and builds
    the tokenizer and datasets needed for training.

    Folder layout expected:
        <folder>/pos/<id>_<rating>.txt
        <folder>/neg/<id>_<rating>.txt

    Only reviews with rating >= 7 (positive) or <= 4 (negative) are loaded,
    matching the IMDB dataset's own labelling convention.
    """

    var reviews: Optional[List[Review]]

    fn __init__(out self):
        self.reviews = Optional(List[Review](capacity=50000))

    # ── Loading ───────────────────────────────────────────────────────────────

    fn load_from_folder(mut self, folder_path: String) raises:
        """Recursively load all pos/neg reviews under folder_path.

        Args:
            folder_path: Root of the split directory (e.g. .../aclImdb/train).
        """
        download()
        var pos_path = Path("/tmp") / folder_path / "pos"
        var neg_path = Path("/tmp") / folder_path / "neg"
        ref reviews = self.reviews.value()

        if pos_path.exists():
            for item in pos_path.listdir():
                var rating = self.extract_rating(item.name())
                if rating >= 7:
                    var comment = pos_path.joinpath(item.name()).read_text()
                    reviews.append(Review(rating, comment))

        if neg_path.exists():
            for item in neg_path.listdir():
                var rating = self.extract_rating(item.name())
                if rating <= 4:
                    var comment = neg_path.joinpath(item.name()).read_text()
                    reviews.append(Review(rating, comment))

    fn extract_rating(self, filename: String) -> Int:
        """Parse the numeric rating embedded in an IMDB filename.

        IMDB filenames follow the pattern '<review_id>_<rating>.txt',
        e.g. '9992_10.txt' has rating 10.

        Args:
            filename: Bare filename (no directory component).

        Returns:
            Parsed integer rating, or 0 on parse failure.
        """
        var parts = filename.split(StringSlice("_"))
        if len(parts) >= 2:
            var rating_str = parts[1].split(StringSlice(".txt"))[0]
            try:
                return Int(rating_str)
            except:
                return 0
        return 0

    # ── Tokenizer ─────────────────────────────────────────────────────────────

    fn init_tokenizer(
        self,
        min_freq: Int = 5,
        max_n: Int = 1,
    ) raises -> DefaultTokenizer:
        """Build a vocabulary from all loaded review texts.

        Applies lowercasing, HTML stripping, digit removal, and a minimum
        frequency filter (min_freq=2) to keep the vocabulary compact.

        Returns:
            Trained DefaultTokenizer ready to encode new text.
        """
        var lines = [review.comment for review in self.reviews.value()]
        return DefaultTokenizer.from_text_lines(
            lines^, IMDBTextCleaner(), min_freq=min_freq, max_n=max_n
        )

    # ── Dataset builder ───────────────────────────────────────────────────────

    fn build_datasets(
        self,
        tokenizer: DefaultTokenizer,
        shuffle: Bool = True,
        random_seed: Int = 42,
    ) -> Tuple[List[List[Int]], List[Int]]:
        """Encode all reviews into token-id sets and produce binary labels.

        Each review is encoded then deduplicated — only the *presence* of a
        token matters, not its frequency. This bag-of-words representation
        is the key insight from Trask Ch. 11: removing frequency bias makes
        sentiment much easier to learn.

        Args:
            tokenizer: Tokenizer built from the same review corpus.
            shuffle: Should we shuffle the datasets.
            random_seed: Random seed used for shuffle.

        Returns:
            T(t)oken_id_sets : List[List[Int]] — One deduplicated id list per review.
            L(l)abels        : List[Int]       — 1 for positive, 0 for negative.
        """
        var token_id_sets = List[List[Int]]()
        var labels = List[Int]()

        for review in self.reviews.value():
            try:
                var ids = tokenizer.encode(review.comment)

                # Deduplicate: presence-only, order does not matter
                var seen = Set[Int]()
                var deduped = List[Int]()
                for token_id in ids:
                    if token_id not in seen:
                        seen.add(token_id)
                        deduped.append(token_id)
                token_id_sets.append(deduped^)
            except:
                # Keep list lengths aligned even if a review fails to encode
                token_id_sets.append(List[Int]())

            labels.append(1 if review.rating >= 7 else 0)

        if shuffle:
            self.shuffle_datasets(token_id_sets, labels, random_seed)
        return (token_id_sets^, labels^)

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

    fn get_labels(self) -> List[Int]:
        """Return binary labels for all loaded reviews (convenience method).

        Returns:
            List of ints: 1 = positive (rating >= 7), 0 = negative (rating <= 4).
        """
        var labels = List[Int]()
        for review in self.reviews.value():
            labels.append(1 if review.rating >= 7 else 0)
        return labels^


# =============================================================================
# Sigmoid activation
# =============================================================================


fn sigmoid[
    dtype: DType
](x: Tensor[dtype]) -> Tensor[dtype] where dtype.is_floating_point():
    """Element-wise sigmoid: σ(x) = 1 / (1 + e^(−x)).

    Args:
        x: Input tensor of any shape.

    Returns:
        Tensor of same shape with values in (0, 1).
    """
    var ones = Tensor[dtype].ones_like(x)
    var neg_x = x * Tensor[dtype].scalar(-1.0)
    return ones / (ones + neg_x.exp())


# =============================================================================
# Configuration
# =============================================================================

comptime TRAIN_FOLDER = "aclImdb/train"
comptime LEARNING_RATE: Float32 = 0.01
comptime ITERATIONS = 3
comptime HIDDEN_SIZE = 100
comptime TEST_SIZE = 1000  # last N reviews held out for testing
comptime RANDOM_SEED_W01 = 1
comptime RANDOM_SEED_W12 = 2
comptime WORD_RETENTION_MIN_FREQ = 5
comptime NGRAM_SIZE = 3



# =============================================================================
# Training and evaluation
# =============================================================================


def main() raises:
    comptime dtype = DType.float32
    var sys = Python.import_module("sys")

    # ── Load and preprocess data ──────────────────────────────────────────────
    print("Loading reviews from:", TRAIN_FOLDER)
    var preprocessor = IMDBPreprocessor()
    preprocessor.load_from_folder(TRAIN_FOLDER)

    print("Building vocabulary with retention frequency: ", WORD_RETENTION_MIN_FREQ, " and ngram_size: ", NGRAM_SIZE)
    var tokenizer = preprocessor.init_tokenizer(min_freq=WORD_RETENTION_MIN_FREQ, max_n=NGRAM_SIZE)
    var vocab_size = len(tokenizer)
    print("Vocabulary size:", vocab_size)

    print("Encoding reviews...")
    var datasets = preprocessor.build_datasets(
        tokenizer, shuffle=False, random_seed=42
    )
    var token_id_sets = datasets[0].copy()
    var labels = datasets[1].copy()
    _ = preprocessor.reviews.take()  # free raw review text — no longer needed
    print("Reviews loaded:", len(token_id_sets))
    print("Labels  loaded:", len(labels))

    # ── Weight initialisation ─────────────────────────────────────────────────
    # Uniform in [-0.1, 0.1) — equivalent to 0.2*random() - 0.1 in NumPy.
    # weights_0_1 : embedding matrix  (vocab_size  × hidden_size)
    # weights_1_2 : output projection (hidden_size × 1)
    var weights_0_1 = Tensor[dtype].rand(
        Shape(vocab_size, HIDDEN_SIZE),
        min=-0.1,
        max=0.1,
        init_seed=RANDOM_SEED_W01,
    )
    var weights_1_2 = Tensor[dtype].rand(
        Shape(HIDDEN_SIZE, 1),
        min=-0.1,
        max=0.1,
        init_seed=RANDOM_SEED_W12,
    )

    var alpha = Tensor[dtype].scalar(LEARNING_RATE)
    var train_size = len(token_id_sets) - TEST_SIZE

    print("Training configuration:")
    print("  Iterations:    ", ITERATIONS)
    print("  Hidden size:   ", HIDDEN_SIZE)
    print("  Learning rate: ", LEARNING_RATE)
    print("  Train samples: ", train_size)
    print("  Test samples:  ", TEST_SIZE)

    # ── Training loop ─────────────────────────────────────────────────────────
    for iteration in range(ITERATIONS):
        preprocessor.shuffle_datasets(token_id_sets, labels, iteration)
        var num_correct = 0
        var num_seen = 0

        for sample_idx in range(train_size):
            ref token_ids = token_id_sets[sample_idx]
            var target = Tensor[dtype].scalar(Float32(labels[sample_idx]))

            # ── Forward pass ──────────────────────────────────────────────────
            #
            # 1. Gather the embedding rows for every token present in this
            #    review and sum them into a single hidden vector.
            #    Shape: (len(token_ids), hidden_size) → sum → (hidden_size,)
            var embedding_sum = weights_0_1.gather[track_grad=False](
                token_ids
            ).sum[track_grad=False](axes=[0], keepdims=False)
            var hidden = sigmoid(embedding_sum)  # (hidden_size,)

            # 2. Project hidden → scalar prediction via dot product.
            var output_weights_flat = weights_1_2.squeeze[
                track_grad=False
            ]()  # (hidden_size,)
            var prediction = sigmoid(
                hidden.dot[track_grad=False](output_weights_flat)
            )  # scalar

            # ── Backward pass (manual) ────────────────────────────────────────
            #
            # Output layer error: dL/d(prediction) for BCE loss
            # For BCE + sigmoid output, the gradient simplifies to:
            # delta_output = prediction - target
            var output_delta = prediction - target  # scalar

            # Propagate error to hidden layer (before activation)
            # dL/dz1 = W2^T @ output_delta
            var hidden_error = (
                output_delta * output_weights_flat
            )  # (hidden_size,)

            # Apply sigmoid derivative for hidden layer activation
            # For hidden = sigmoid(z1), we need to multiply by sigmoid'(z1)
            # sigmoid'(z1) = hidden * (1 - hidden)
            var sigmoid_grad = hidden * (
                Tensor[dtype].scalar(1.0) - hidden
            )  # (hidden_size,)
            var hidden_delta = hidden_error * sigmoid_grad  # (hidden_size,)

            # ── Weight updates ────────────────────────────────────────────────
            #
            # weights_1_2 -= outer(hidden, output_delta) * lr
            # outer(hidden, output_delta): (hidden_size,) x scalar → (hidden_size, 1)
            var unsqueeze_axes = IntArray()
            unsqueeze_axes.append(1)
            var hidden_col = hidden.unsqueeze[track_grad=False](
                unsqueeze_axes
            )  # (hidden_size, 1)
            var outer_product = hidden_col * output_delta  # (hidden_size, 1)
            weights_1_2 = weights_1_2 - outer_product * alpha

            # Sparse embedding update: only rows touched by this review's tokens
            # weights_0_1[token_id] -= hidden_delta * lr
            # Note: Now hidden_delta includes the sigmoid derivative for the hidden layer
            for token_id in token_ids:
                var embedding_row = weights_0_1[
                    i(token_id), s()
                ]  # (hidden_size,) view
                var updated_row = embedding_row - hidden_delta * alpha
                weights_0_1.fill(updated_row, i(token_id), s())

            # ── Progress tracking ─────────────────────────────────────────────
            if abs(output_delta.item()) < 0.5:
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

        var embedding_sum = weights_0_1.gather[track_grad=False](token_ids).sum(
            axes=[0], keepdims=False
        )
        var hidden = sigmoid(embedding_sum)
        var output_weights_flat = weights_1_2.squeeze[track_grad=False]()
        var prediction = sigmoid(
            hidden.dot[track_grad=False](output_weights_flat)
        )

        if abs(prediction.item() - Float32(true_label)) < 0.5:
            num_correct += 1
        num_seen += 1

    var test_accuracy = Float32(num_correct) / Float32(num_seen)
    print("Test Accuracy: " + String(test_accuracy))

    # ── Similarity search — verify embedding quality ──────────────────────────

    var query_words = [
        "beautiful",
        "wonderful",
        "outstanding",
        "boring",
        "terrible",
    ]
    for target in query_words:
        print("\nTarget: ", target)
        var neighbours = similar(tokenizer, weights_0_1, target)
        for item in neighbours:
            print(" ", item[0], item[1])


def similar(
    tokenizer: DefaultTokenizer,
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
        if word == target or '_' in word:
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
    to: String = "/tmp",
    # url: String = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    url: String = "https://huggingface.co/datasets/NolanChai/aclImdb_v1/resolve/main/aclImdb_v1.tar.gz",
) raises:
    var download_to = "/tmp"
    var file_path = Path("/tmp/aclImdb_v1.tar.gz")
    var extracted_path = Path("/tmp/aclImdb/")

    if extracted_path.exists() and extracted_path.is_dir():
        print("Extracted folder", extracted_path.name(), "exists")
        return
    elif file_path.exists() and file_path.is_file():
        print("/tmp/aclImdb_v1.tar.gz file exists - extracting")
        _ = Process.run("tar", ["-xzf", "/tmp/aclImdb_v1.tar.gz", "-C", "/tmp"])
    else:
        print("File or folder is not present - downloading")
        var args = ["-P", download_to, url]
        _ = Process.run("wget", args)
        print("Downloaded - extracting now")
        _ = Process.run("tar", ["-xzf", "/tmp/aclImdb_v1.tar.gz", "-C", "/tmp"])

    print("done")


# Loading reviews from: aclImdb/train
# Building vocabulary...
# Vocabulary size: 49301
# Encoding reviews...
# Reviews loaded: 25000
# Labels  loaded: 25000
# Iter:0  Progress: 95.99%  Training Accuracy: 0.99858333
# Iter:1  Progress: 95.99%  Training Accuracy: 0.99720836
# Evaluating on test set (1000 reviews)...
# Test Accuracy: 1.0
