"""
IMDB Sentiment Classifier — Tenmo Tensor Showcase
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
    "Grokking Deep Learning", Chapter 12.

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
from tenmo.common_utils import (
    SimpleTokenizer,
    DEFAULT_SPLITTER,
    DEFAULT_SUBSTITUTION,
    DEFAULT_UNK,
    END_OF_TEXT,
    i,
    s,
)
from std.pathlib import Path
from std.math import abs
from std.python import Python


# =============================================================================
# Configuration
# =============================================================================

# comptime TRAIN_FOLDER = "/home/tenmoomnet/aclImdb/train"
comptime TRAIN_FOLDER = "aclImdb/train"
comptime LEARNING_RATE: Float32 = 0.01
comptime ITERATIONS = 2
comptime HIDDEN_SIZE = 100
comptime TEST_SIZE = 1000  # last N reviews held out for testing
comptime RANDOM_SEED_W01 = 1
comptime RANDOM_SEED_W12 = 2


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
        var pos_path = Path.home() / folder_path / "pos"
        var neg_path = Path.home() / folder_path / "neg"
        ref reviews = self.reviews.value()

        if pos_path.exists():
            for item in pos_path.listdir():
                var rating = self._extract_rating(item.name())
                if rating >= 7:
                    var comment = pos_path.joinpath(item.name()).read_text()
                    reviews.append(Review(rating, comment))

        if neg_path.exists():
            for item in neg_path.listdir():
                var rating = self._extract_rating(item.name())
                if rating <= 4:
                    var comment = neg_path.joinpath(item.name()).read_text()
                    reviews.append(Review(rating, comment))

    fn _extract_rating(self, filename: String) -> Int:
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
    ) raises -> SimpleTokenizer[
        DEFAULT_SPLITTER, DEFAULT_SUBSTITUTION, DEFAULT_UNK, END_OF_TEXT
    ]:
        """Build a vocabulary from all loaded review texts.

        Applies lowercasing, HTML stripping, digit removal, and a minimum
        frequency filter (min_freq=2) to keep the vocabulary compact.

        Returns:
            Trained SimpleTokenizer ready to encode new text.
        """
        var lines = [
            StringSlice(review.comment) for review in self.reviews.value()
        ]
        return SimpleTokenizer.from_text_lines_min_freq(lines^)

    # ── Dataset builder ───────────────────────────────────────────────────────

    fn build_datasets(
        self,
        tokenizer: SimpleTokenizer,
    ) -> Tuple[List[List[Int]], List[Int]]:
        """Encode all reviews into token-id sets and produce binary labels.

        Each review is encoded then deduplicated — only the *presence* of a
        token matters, not its frequency. This bag-of-words representation
        is the key insight from Trask Ch. 11: removing frequency bias makes
        sentiment much easier to learn.

        Args:
            tokenizer: Tokenizer built from the same review corpus.

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
                var seen = Dict[Int, Bool]()
                var deduped = List[Int]()
                for token_id in ids:
                    if token_id not in seen:
                        seen[token_id] = True
                        deduped.append(token_id)
                token_id_sets.append(deduped^)
            except:
                # Keep list lengths aligned even if a review fails to encode
                token_id_sets.append(List[Int]())

            labels.append(1 if review.rating >= 7 else 0)

        return (token_id_sets^, labels^)

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
# Training and evaluation
# =============================================================================


def main() raises:
    comptime dtype = DType.float32
    var sys = Python.import_module("sys")

    # ── Load and preprocess data ──────────────────────────────────────────────
    print("Loading reviews from:", TRAIN_FOLDER)
    var preprocessor = IMDBPreprocessor()
    preprocessor.load_from_folder(TRAIN_FOLDER)

    print("Building vocabulary...")
    var tokenizer = preprocessor.init_tokenizer()
    var vocab_size = len(tokenizer)
    print("Vocabulary size:", vocab_size)

    print("Encoding reviews...")
    ref (token_id_sets, labels) = preprocessor.build_datasets(tokenizer)
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

    # ── Training loop ─────────────────────────────────────────────────────────
    for iteration in range(ITERATIONS):
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
            var embedding_sum = weights_0_1.gather(token_ids).sum(
                axes=[0], keepdims=False
            )
            var hidden = sigmoid(embedding_sum)  # (hidden_size,)

            # 2. Project hidden → scalar prediction via dot product.
            var output_weights_flat = weights_1_2.squeeze()  # (hidden_size,)
            var prediction = sigmoid(hidden.dot(output_weights_flat))  # scalar

            # ── Backward pass (manual) ────────────────────────────────────────
            #
            # Output layer error: dL/d(prediction) * sigmoid'
            # For MSE + sigmoid output: delta = prediction - target
            var output_delta = prediction - target  # scalar

            # Hidden layer error: back-propagate through output weights
            var hidden_delta = (
                output_delta * output_weights_flat
            )  # (hidden_size,)

            # ── Weight updates ────────────────────────────────────────────────
            #
            # weights_1_2 -= outer(hidden, output_delta) * lr
            # outer(hidden, output_delta): (hidden_size,) x scalar → (hidden_size, 1)
            var unsqueeze_axes = IntArray()
            unsqueeze_axes.append(1)
            var hidden_col = hidden.unsqueeze(
                unsqueeze_axes
            )  # (hidden_size, 1)
            var outer_product = hidden_col * output_delta  # (hidden_size, 1)
            weights_1_2 = weights_1_2 - outer_product * alpha

            # Sparse embedding update: only rows touched by this review's tokens
            # weights_0_1[token_id] -= hidden_delta * lr
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
                var raw_pct = Float32(sample_idx) / Float32(len(token_id_sets))
                var pct_int = Int(raw_pct * 10000)
                var pct_whole = pct_int // 100
                var pct_decimal = pct_int % 100
                var decimal_str = ("0" if pct_decimal < 10 else "") + String(
                    pct_decimal
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

        var embedding_sum = weights_0_1.gather(token_ids).sum(
            axes=[0], keepdims=False
        )
        var hidden = sigmoid(embedding_sum)
        var output_weights_flat = weights_1_2.squeeze()
        var prediction = sigmoid(hidden.dot(output_weights_flat))

        if abs(prediction.item() - Float32(true_label)) < 0.5:
            num_correct += 1
        num_seen += 1

    var test_accuracy = Float32(num_correct) / Float32(num_seen)
    print("Test Accuracy: " + String(test_accuracy))
