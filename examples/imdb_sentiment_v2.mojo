from tenmo.shapes import Shape
from std.math import abs
from tenmo.sgd import SGD
from std.random import shuffle, seed
from tenmo.tensor import Tensor
from tenmo.common_utils import *
from tenmo.mnemonics import *
from std.pathlib import Path
from std.collections import Set
from std.python import Python
from tenmo.common_utils import (
    SimpleTokenizer,
    DEFAULT_SPLITTER,
    DEFAULT_SUBSTITUTION,
    DEFAULT_UNK,
    END_OF_TEXT,
)


# comptime TRAIN_FOLDER = "/home/tenmoomnet/aclImdb/train"
comptime TRAIN_FOLDER = "aclImdb/train"
comptime LEARNING_RATE: Float32 = 0.005
comptime ITERATIONS = 5
comptime HIDDEN_SIZE = 100
comptime TEST_SIZE = 1000  # last N reviews held out for testing
comptime RANDOM_SEED_W01 = 42
comptime RANDOM_SEED_W12 = 24


@fieldwise_init
struct Review(ImplicitlyCopyable, Movable):
    var rating: Int
    var comment: String


# Extended IMDD dataset loader
struct IMDBPreprocessor:
    var reviews: Optional[List[Review]]

    fn __init__(out self):
        self.reviews = Optional(List[Review](capacity=50000))

    fn load_from_folder(mut self, folder_path: String) raises:
        """Load reviews from pos/neg folders."""
        var pos_path = Path.home() / folder_path / "pos"
        var neg_path = Path.home() / folder_path / "neg"
        ref reviews = self.reviews.value()

        # Load positive reviews
        if pos_path.exists():
            for item in pos_path.listdir():
                var rating = self._extract_rating(item.name())
                if rating >= 7:  # Positive review threshold
                    var comment = pos_path.joinpath(item.name()).read_text()
                    reviews.append(Review(rating, comment))

        # Load negative reviews
        if neg_path.exists():
            for item in neg_path.listdir():
                var rating = self._extract_rating(item.name())
                if rating <= 4:  # Negative review threshold
                    var comment = neg_path.joinpath(item.name()).read_text()
                    reviews.append(Review(rating, comment))

    fn _extract_rating(self, filename: String) -> Int:
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
            # Rating >=7 is positive (1), <=4 is negative (0)
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
    ) -> Tuple[List[List[Int]], List[Int]]:
        """Build input_dataset and target_dataset from loaded reviews.

        Args:
            tokenizer: Already initialised tokenizer.

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

        return (input_dataset^, target_dataset^)


def main() raises:
    comptime dtype = DType.float32
    var sys = Python.import_module("sys")

    # ── Load and preprocess ───────────────────────────────────────────────────
    print("Loading reviews from:", TRAIN_FOLDER)
    var preprocessor = IMDBPreprocessor()
    preprocessor.load_from_folder(TRAIN_FOLDER)

    print("Building vocabulary...")
    var tokenizer = preprocessor.init_tokenizer()
    var vocab_size = len(tokenizer)
    print("Vocabulary size:", vocab_size)

    print("Encoding reviews...")
    ref (token_id_sets, labels) = preprocessor.build_datasets(tokenizer)
    # Free up all the review strings because we are done with them
    _ = preprocessor.reviews.take()
    print("Reviews loaded:", len(token_id_sets))
    print("Labels  loaded:", len(labels))
    # ── Weight initialisation ─────────────────────────────────────────────────
    # requires_grad=True — autograd will accumulate gradients into these
    var weights_0_1 = Tensor[dtype].rand(
        Shape(vocab_size, HIDDEN_SIZE),
        min=-0.1,
        max=0.1,
        init_seed=RANDOM_SEED_W01,
        requires_grad=True,
    )
    var weights_1_2 = Tensor[dtype].rand(
        Shape(HIDDEN_SIZE, 1),
        min=-0.1,
        max=0.1,
        init_seed=RANDOM_SEED_W12,
        requires_grad=True,
    )

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

    # ── Training loop ─────────────────────────────────────────────────────────
    for iteration in range(ITERATIONS):
        var num_correct = 0
        var num_seen = 0
        seed()

        for sample_idx in range(train_size):
            ref token_ids = token_id_sets[sample_idx]
            var target = Tensor[dtype].scalar(Float32(labels[sample_idx]))

            # ── Forward pass ──────────────────────────────────────────────────
            # gather() copies — we don't know which embedding rows would be picked up! Strides do not allow that to express!
            # gather() + sum() + sigmoid() + dot() + sigmoid() all track grad normally.
            var hidden = (
                weights_0_1.gather(token_ids, requires_grad=True)
                .sum(axes=[0], keepdims=False)
                .sigmoid()
            )

            var prediction = hidden.dot(
                weights_1_2.squeeze()
            ).sigmoid()  # scalar

            # ── Loss ──────────────────────────────────────────────────────────
            # MSE: (prediction - target)^2
            var diff = prediction - target
            var diff_sqrd = diff * diff
            var loss = diff_sqrd.squeeze()  # scalar Tensor for backward
            # ── Backward pass ─────────────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ── Progress tracking ─────────────────────────────────────────────
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

        # Inference — no grad tracking needed
        var hidden = (
            weights_0_1.gather[track_grad=False](token_ids)
            .sum[track_grad=False](axes=[0], keepdims=False)
            .sigmoid[track_grad=False]()
        )
        var prediction = hidden.dot[track_grad=False](
            weights_1_2.squeeze()
        ).sigmoid[track_grad=False]()

        if abs(prediction.item() - Float32(true_label)) < 0.5:
            num_correct += 1
        num_seen += 1

    var test_accuracy = Float32(num_correct) / Float32(num_seen)
    print("Test Accuracy: " + String(test_accuracy))


# ====================Outputs========================="
# Loading reviews from: /home/tenmoomnet/aclImdb/train
# Building vocabulary...
# Vocabulary size: 49301
# Encoding reviews...
# IMDBPreprocessor reviews freed up?  True
# Reviews loaded: 25000
# Labels  loaded: 25000
# Iter:0  Progress: 95.99%  Training Accuracy: 0.99433334
# Iter:1  Progress: 95.99%  Training Accuracy: 0.98862534
# Iter:2  Progress: 95.99%  Training Accuracy: 0.98825516
# Iter:3  Progress: 95.99%  Training Accuracy: 0.98787594
# Iter:4  Progress: 95.99%  Training Accuracy: 0.98754176
# Evaluating on test set (1000 reviews)...
# Test Accuracy: 1.0
