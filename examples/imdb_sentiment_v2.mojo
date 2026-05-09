from tenmo.shapes import Shape
from std.math import abs, sqrt
from tenmo.sgd import SGD
from std.random import seed
from tenmo.tensor import Tensor
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
        var pos_path = Path.home() / folder_path / "pos"
        var neg_path = Path.home() / folder_path / "neg"
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

        # Shuffle datasets
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
            rng_seed = (rng_seed * 1664525 + 1013904223) % (
                2**31
            )  # ← evolves
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

        for sample_idx in range(train_size):
            ref token_ids = token_id_sets[sample_idx]
            var target = Tensor[dtype].scalar(Float32(labels[sample_idx]))

            # ── Forward pass ──────────────────────────────────────────────────
            var hidden = (
                weights_0_1.gather(token_ids)
                .sum(axes=[0], keepdims=False)
                .sigmoid()
            )
            print("Forward: hidden is on gpu: ", hidden.is_on_gpu())
            comptime if has_accelerator():
                if not hidden.is_on_gpu():
                    hidden = hidden.to_gpu()
            var prediction = hidden.dot(weights_1_2).sigmoid()

            # ── Loss ──────────────────────────────────────────────────────────
            var diff = prediction - target
            var loss = diff**2

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

        var hidden = (
            weights_0_1.gather[track_grad=False](token_ids)
            .sum[track_grad=False](axes=[0], keepdims=False)
            .sigmoid[track_grad=False]()
        )

        print("hidden is on gpu: ", hidden.is_on_gpu(), "weights_1_2: ", weights_1_2.is_on_gpu())
        comptime if has_accelerator():
            if not hidden.is_on_gpu():
                hidden = hidden.to_gpu()
        var prediction = hidden.dot[track_grad=False](weights_1_2).sigmoid[
            track_grad=False
        ]()

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
    tokenizer: SimpleTokenizer[_, _, _, _],
    ref embeddings: Tensor[DType.float32],
    target: String = "beautiful",
    top_n: Int = 10,
) raises -> List[Tuple[String, Float32]]:
    var target_ids = tokenizer.encode(target)
    var scores = Dict[String, Float32]()

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

#Loading reviews from: aclImdb/train
#Building vocabulary...
#Vocabulary size: 29777
#Encoding reviews...
#Reviews loaded: 25000
#Labels  loaded: 25000
#Iter:0  Progress: 95.99%  Training Accuracy: 0.82791677
#Iter:1  Progress: 95.99%  Training Accuracy: 0.89787554
#Iter:2  Progress: 95.99%  Training Accuracy: 0.92966664
#Evaluating on test set (1000 reviews)...
#Test Accuracy: 0.863
#
#Target:  beautiful
#  cry -0.8139669
#  innocent -0.82132995
#  tragic -0.82555395
#  impressive -0.83826303
#  magic -0.84857774
#  unique -0.851256
#  stunning -0.8520091
#  compelling -0.8524668
#  buddy -0.85552156
#  superb -0.8582328
#
#Target:  wonderful
#  refreshing -0.81821996
#  definitely -0.8412058
#  rare -0.87024164
#  funniest -0.8810015
#  gem -0.88207245
#  surprised -0.8925017
#  enjoyable -0.89441526
#  incredible -0.8946219
#  fantastic -0.9034469
#  wonderfully -0.90369225
#
#Target:  outstanding
#  offers -0.7104493
#  appreciated -0.7180212
#  sweet -0.74203795
#  fantastic -0.7430425
#  games -0.74329716
#  beautifully -0.75147206
#  realistic -0.75587887
#  noir -0.7573669
#  surprisingly -0.760731
#  originally -0.7609394
#
#Target:  boring
#  poorly -0.8065185
#  mess -0.8126137
#  disappointing -0.818716
#  awful -0.8211603
#  terrible -0.8514237
#  disappointment -0.88710153
#  dull -0.8926999
#  supposed -0.90556884
#  horrible -0.91566694
#  badly -0.9355349
#
#Target:  terrible
#  disappointment -0.77307105
#  dull -0.7917348
#  poorly -0.8179629
#  badly -0.82152003
#  mess -0.8331556
#  disappointing -0.83548564
#  worse -0.83746415
#  fails -0.8376444
#  awful -0.84623975
#  boring -0.8514237
