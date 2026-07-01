"""
Word2Vec Implementation in Mojo
============================================================
This implementation trains word embeddings using the CBOW
(Continuous Bag of Words) architecture with Negative Sampling.
It's designed for the IMDB movie review dataset.

Key Concepts:
   - CBOW: Predict the target word from averaged context words
   - Negative Sampling: Train on positive pairs + random negatives
   - Word Embeddings: Dense vector representations of words

Architecture:
   input_embeddings: Input embeddings (vocab_size × hidden_size)
   output_embeddings: Output embeddings (vocab_size × hidden_size)
   Training: Context words predict the target word (CBOW)
"""

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

# ============================================================
# CONFIGURATION
# ============================================================

# Common words to filter out during training
comptime STOPWORDS = """
{
    'the','a','an','and','or','but','in','on','at','to',
    'for','of','with','by','from','is','it','as','be',
    'was','were','has','had','have','this','that',
    'i','he','she','they','we','you','his','her','their',
    'its','my','our','your','so','do','did','not','no',
    'if','up','out','about','than','into','then','there',
    'what','which','who','how','when','where','will','would',
    'could','should','may','might','also','just','been','after',
    'before','more','all','one','can','get','got','him','them'
}
"""

# Special token for unknown words (not in vocabulary)
comptime UNKNOWN_TOKEN = "<|unk|>"

# ============================================================
# TEXT TOKENIZER
# ============================================================


@fieldwise_init
struct Tokenizer(Sized & ImplicitlyCopyable & Movable):
    """Converts between text strings and integer token IDs.

    The Tokenizer handles:
        1. Cleaning raw text (removing HTML, URLs, digits, etc.)
        2. Building a vocabulary from text corpus.
        3. Encoding text → token IDs.
        4. Decoding token IDs → text.

    Attributes:
        word_to_id: Maps word strings to unique integer IDs.
        id_to_word: Maps integer IDs back to word strings.
    """

    var word_to_id: Dict[String, Int]
    var id_to_word: Dict[Int, String]

    def __init__(
        out self,
        var vocabulary: Dict[String, Int],
    ) raises:
        """Initialize tokenizer from a vocabulary dictionary."""
        self.id_to_word = {item.value: item.key for item in vocabulary.items()}
        self.word_to_id = vocabulary^

    def __init__(out self, *, copy: Self):
        """Copy constructor."""
        self.id_to_word = copy.id_to_word.copy()
        self.word_to_id = copy.word_to_id.copy()

    def __init__(out self, *, deinit move: Self):
        """Move constructor."""
        self.id_to_word = move.id_to_word^
        self.word_to_id = move.word_to_id^

    @staticmethod
    def clean_text(
        raw_text: String,
    ) raises -> PythonObject:
        """Clean raw text by removing noise and normalizing.

        Processing Steps:
            1. Remove HTML tags: <br />, <p>, <a href="...">, etc.
            2. Remove URLs: http://example.com or www.example.com.
            3. Remove digit sequences: 2024, 10, 3rd, etc.
            4. Remove stray apostrophes (preserving contractions).
            5. Collapse multiple spaces and newlines.
            6. Filter out words shorter than 2 characters.

        Args:
            raw_text: The input text string to clean.

        Returns:
            PythonObject: List of cleaned word tokens.
        """
        var py = Python.import_module("builtins")
        var regex = Python.import_module("re")

        var text = py.str(raw_text)

        # Remove HTML tags — <br />, <p>, <a href="...">, </div> etc.
        # Pattern: [^>]+ matches one or more characters that are not >
        text = regex.sub(r"<[^>]+>", " ", text)

        # Remove URLs — http://example.com or www.example.com
        # Pattern: \S+ matches any non-whitespace run after http or www
        text = regex.sub(r"http\S+|www\.\S+", " ", text)

        # Remove digit sequences — 2024, 10, 3rd etc.
        # Pattern: \d+ matches one or more consecutive digit characters
        text = regex.sub(r"\d+", " ", text)

        # Remove stray apostrophes (preserve contractions)
        # Preserves: don't, it's, they're (apostrophe between letters)
        # Removes:   'hello' (leading), dogs' (trailing)
        text = regex.sub(r"(?<!\w)'|'(?!\w)", " ", text)

        # Collapse multiple spaces/newlines → single space, strip ends
        text = regex.sub(r"\s+", " ", text).strip()

        # Filter out words shorter than 2 characters
        var filter_short_words = Python.evaluate(
            "lambda words: [w for w in words.split() if len(w) >= 2]"
        )
        var words = filter_short_words(text)

        # Optional: Remove stopwords (commented out for flexibility)
        # var stopwords = Python.evaluate(STOPWORDS)
        # filter_fn = Python.evaluate(
        #     "lambda words, sw: [w for w in words if w not in sw]"
        # )
        # return filter_fn(words, stopwords)

        return words

    @staticmethod
    def from_text_lines(
        text_lines: List[String],
    ) raises -> Self:
        """Build a tokenizer from a list of text lines.

        Process:
            1. Clean all text lines.
            2. Collect unique words.
            3. Sort vocabulary alphabetically.
            4. Assign each word a unique integer ID.
            5. Add UNKNOWN token for out-of-vocabulary words.

        Args:
            text_lines: List of text strings to build vocabulary from.

        Returns:
            Tokenizer: Initialized tokenizer with full vocabulary.
        """
        var py = Python.import_module("builtins")
        var regex = Python.import_module("re")
        var all_words: PythonObject = []

        # Collect all words from all text lines
        for line in text_lines:
            all_words.extend(Tokenizer.clean_text(line))

        # Create unique vocabulary, sorted alphabetically
        all_words = py.list(py.set(all_words))
        all_words = py.sorted(all_words)

        # Add UNKNOWN token for words not in vocabulary
        var vocab_with_unknown: PythonObject = [UNKNOWN_TOKEN]
        vocab_with_unknown.extend(all_words)

        # Map each word to a unique integer ID
        var vocabulary = {
            String(token): Int(index)
            for index, token in enumerate(vocab_with_unknown.__iter__())
        }

        return Self(vocabulary^)

    def encode(self, text: String) raises -> List[Int]:
        """Convert text to a list of token IDs.

        Args:
            text: Input text string.

        Returns:
            List[Int]: Token IDs for each word in the text.
        """
        # Clean the text and split into words
        var words = Tokenizer.clean_text(text)

        # Convert each word to its token ID (or UNKNOWN if not found)
        var token_ids = List[Int](capacity=len(words))
        for word in words:
            var word_str = String(word)
            token_ids.append(
                self.word_to_id[word_str] if word_str
                in self.word_to_id else self.word_to_id[UNKNOWN_TOKEN]
            )

        return token_ids^

    def decode(self, token_ids: List[Int]) raises -> String:
        """Convert token IDs back to text.

        Args:
            token_ids: List of token IDs.

        Returns:
            String: The reconstructed text string.
        """
        var text = " ".join([self.id_to_word[id] for id in token_ids])
        return text^

    def __len__(self) -> Int:
        """Return vocabulary size."""
        return len(self.id_to_word)


# ============================================================
# NEGATIVE SAMPLING DATA LOADER
# ============================================================


struct NegativeSampler:
    """Loads and prepares training data for Word2Vec with negative sampling.

    The NegativeSampler:
        1. Loads IMDB reviews from disk.
        2. Tokenizes text using the Tokenizer.
        3. Builds training dataset of word sequences.
        4. Provides negative samples for training.

    Attributes:
        tokenized_reviews: List of token ID sequences per review.
        concatenated_tokens: All token IDs concatenated into one array.
    """

    var tokenized_reviews: List[List[Int]]
    var concatenated_tokens: List[Int]

    def __init__(out self):
        """Initialize empty data structures."""
        self.tokenized_reviews = List[List[Int]](capacity=50000)
        self.concatenated_tokens = List[Int](capacity=5000000)

    def init_tokenizer_and_datasets(
        mut self, dataset_folder: String
    ) raises -> Tokenizer:
        """Load IMDB reviews and create tokenizer and datasets.

        The IMDB dataset has:
            - pos/ folder: Positive reviews (rating 7-10).
            - neg/ folder: Negative reviews (rating 1-4).

        Args:
            dataset_folder: Path to the IMDB dataset folder.

        Returns:
            Tokenizer: The trained tokenizer for this dataset.
        """
        # Ensure dataset is downloaded
        self._download_imdb_dataset()

        # Build paths
        var positive_path = Path("/tmp") / dataset_folder / "pos"
        var negative_path = Path("/tmp") / dataset_folder / "neg"

        var all_comments = List[String](capacity=50000)

        # Load positive reviews (rating 7-10)
        if positive_path.exists():
            for file in positive_path.listdir():
                var rating = self._extract_rating_from_filename(file.name())
                if rating >= 7:  # Positive review threshold
                    var comment = positive_path.joinpath(
                        file.name()
                    ).read_text()
                    all_comments.append(comment)

        # Load negative reviews (rating 1-4)
        if negative_path.exists():
            for file in negative_path.listdir():
                var rating = self._extract_rating_from_filename(file.name())
                if rating <= 4:  # Negative review threshold
                    var comment = negative_path.joinpath(
                        file.name()
                    ).read_text()
                    all_comments.append(comment)

        # Build tokenizer from all comments
        var tokenizer = Tokenizer.from_text_lines(all_comments)

        # Tokenize all comments and build datasets
        for comment in all_comments:
            var token_ids = tokenizer.encode(comment)

            # Validate token IDs are within vocabulary range
            var max_id = 0
            for id in token_ids:
                if id > max_id:
                    max_id = id
            if max_id >= len(tokenizer):
                print("ERROR: Token ID out of vocabulary range!")
                print("Max ID:", max_id, "Vocab size:", len(tokenizer))
                print("Offending comment:", comment[byte=0:50])
                break

            # Skip empty reviews
            if len(token_ids) == 0:
                continue

            # Store tokenized review and add to concatenated corpus
            self.tokenized_reviews.append(token_ids.copy())
            self.concatenated_tokens.extend(token_ids^)

        # Final validation: all token IDs must be < vocabulary size
        var vocabulary_size = len(tokenizer)
        var has_error = False
        for review in self.tokenized_reviews:
            if has_error:
                break
            for word_id in review:
                if word_id >= vocabulary_size:
                    print("ERROR: Invalid token ID found!")
                    print("Word ID:", word_id, "Vocab size:", vocabulary_size)
                    has_error = True
                    break

        # Shuffle reviews for better training
        # shuffle(self.tokenized_reviews)

        return tokenizer

    def _extract_rating_from_filename(self, filename: String) -> Int:
        """Extract rating from IMDB filename.

        IMDB filenames follow the pattern: '1234_8.txt'.
        where '8' is the movie rating (1-10).

        Args:
            filename: The filename string (e.g., '9992_10.txt').

        Returns:
            Int: The extracted rating (0 if extraction fails).
        """
        var underscore_separator = StringSlice("_")
        var dot_separator = StringSlice(".txt")

        var parts = filename.split(underscore_separator)
        if len(parts) >= 2:
            var rating_part = parts[1].split(dot_separator)[0]
            try:
                return Int(rating_part)
            except e:
                print("Warning: Could not extract rating from:", filename)
                return 0

        return 0

    def _download_imdb_dataset(
        self,
        download_url: String = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    ) raises:
        """Download and extract the IMDB dataset if not already present.

        Args:
            download_url: URL to download the dataset from.
        """
        var archive_path = Path("/tmp/aclImdb_v1.tar.gz")
        var extracted_path = Path("/tmp/aclImdb/")

        # Check if already extracted
        if extracted_path.exists() and extracted_path.is_dir():
            print("IMDB dataset already extracted at:", extracted_path.name())
            return

        # Check if archive exists
        if archive_path.exists() and archive_path.is_file():
            print("Archive exists, extracting...")
            _ = Process.run(
                "tar", ["-xzf", "/tmp/aclImdb_v1.tar.gz", "-C", "/tmp"]
            )
        else:
            # Download the dataset
            print("Downloading IMDB dataset...")
            var download_to = "/tmp"
            var args = ["-P", download_to, download_url]
            _ = Process.run("wget", args)
            print("Download complete, extracting now...")
            _ = Process.run(
                "tar", ["-xzf", "/tmp/aclImdb_v1.tar.gz", "-C", "/tmp"]
            )

        print("Dataset ready!")


# ============================================================
# TRAINING HELPERS
# ============================================================


def generate_negative_samples(
    current_review: List[Int],
    target_position: Int,
    all_tokens: List[Int],
    num_negative_samples: Int,
) -> List[Int]:
    """Generate negative samples for Word2Vec training.

    Negative sampling randomly selects words from the corpus.
    that are NOT the target word. This helps the model.
    learn to distinguish meaningful context from random noise.

    Args:
        current_review: Token IDs in the current review.
        target_position: Position of the target word in the review.
        all_tokens: All token IDs in the corpus (for random sampling).
        num_negative_samples: Number of negative samples to generate.

    Returns:
        List[Int]: List containing [target_token] + negative_samples.
    """
    var corpus_length = Float64(len(all_tokens))
    var negative_samples = [
        all_tokens[
            min(Int(random_float64() * corpus_length), len(all_tokens) - 1)
        ]
        for _ in range(num_negative_samples)
    ]

    # Insert the target word at position 0 (positive sample)
    negative_samples.insert(0, current_review[target_position])

    return negative_samples^


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

comptime TRAIN_FOLDER = "aclImdb/train"
comptime LEARNING_RATE: Float32 = 0.01
comptime TRAINING_ITERATIONS = 5
comptime EMBEDDING_DIMENSION = 100
comptime TEST_SIZE = 1000  # Last N reviews held out for testing
comptime RANDOM_SEED_INPUT = 42
comptime RANDOM_SEED_OUTPUT = 24
comptime CONTEXT_WINDOW_SIZE = 3
comptime NUM_NEGATIVE_SAMPLES = 5
comptime MAX_REVIEWS_TO_USE = 5000  # Limit training dataset size for speed


# ============================================================
# WORD2VEC MODEL
# ============================================================


struct Word2Vec[dtype: DType]:
    """CBOW with Negative Sampling — manual gradients, scatter_add updates.

    Two embedding matrices trained via CBOW:
      forward() → average context, score targets, apply sigmoid
      step()    → backward + scatter-add gradients to both tables

    Uses rank-1 broadcast in scatter_add so context rows share one gradient vector
    without needing to allocate a repeated matrix.
    """

    var input_embeddings: Tensor[Self.dtype]
    var output_embeddings: Tensor[Self.dtype]

    # Cached forward intermediates for step()
    var cached_avg: Tensor[Self.dtype]
    var cached_tgt_emb: Tensor[Self.dtype]

    def __init__(
        out self,
        vocab_size: Int,
        embedding_dim: Int,
        init_seed: Int = 42,
    ):
        self.input_embeddings = Tensor[Self.dtype].rand(
            Shape(vocab_size, embedding_dim),
            min=-0.1,
            max=0.1,
            init_seed=init_seed,
        )
        self.output_embeddings = Tensor[Self.dtype].zeros(
            Shape(vocab_size, embedding_dim),
        )
        self.cached_avg = Tensor[Self.dtype]()
        self.cached_tgt_emb = Tensor[Self.dtype]()

    def to_gpu(mut self, sync: Bool = True) raises:
        self.input_embeddings = self.input_embeddings.to_gpu(sync=sync)
        self.output_embeddings = self.output_embeddings.to_gpu(sync=sync)

    def forward(
        mut self,
        context_indices: List[Int],
        target_indices: List[Int],
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        """Average context embeddings, score targets, apply sigmoid.

        Caches averaged_context and target_embeddings for step().
        """
        self.cached_avg = self.input_embeddings.gather[track_grad=False](
            context_indices, reduction=Reduction(0)
        )
        self.cached_tgt_emb = self.output_embeddings.gather[track_grad=False](
            target_indices
        )
        var scores = self.cached_tgt_emb.matmul[mode=mv, track_grad=False](
            self.cached_avg
        )
        return scores.sigmoid[track_grad=False]()

    def step(
        mut self,
        scores: Tensor[Self.dtype],
        labels: Tensor[Self.dtype],
        context_indices: List[Int],
        target_indices: List[Int],
        lr: Scalar[Self.dtype],
    ):
        """Backward + scatter-add updates for both embedding tables.

        Input table: all context rows get the same gradient (rank-1 broadcast).
        Output table: each target row gets outer(gradient, avg_context).
        """
        var context_length = len(context_indices)
        var gradient = scores - labels
        var grad_ctx = self.cached_tgt_emb.transpose[track_grad=False]().matmul[
            mode=mv, track_grad=False
        ](gradient)

        # Input embeddings — rank-1 source broadcasts to all context rows
        var ctx_update = -grad_ctx * lr / Scalar[Self.dtype](context_length)
        Filler[Self.dtype].scatter_add(
            self.input_embeddings.buffer,
            ctx_update.buffer,
            IntArray(context_indices),
        )

        # Output embeddings — outer product, each target row gets its own
        var out_update = (
            -gradient.unsqueeze(1) * self.cached_avg.unsqueeze(0) * lr
        )
        Filler[Self.dtype].scatter_add(
            self.output_embeddings.buffer,
            out_update.buffer,
            IntArray(target_indices),
        )


# ============================================================
# MAIN TRAINING LOOP
# ============================================================


def main() raises:
    """Main training function for Word2Vec embedding model.

    Training Process:
        1. Load and tokenize IMDB dataset.
        2. Initialize input and output embedding matrices.
        3. For each word in each review:
            a. Gather context words from the window.
            b. Calculate average context embedding.
            c. Calculate scores for positive + negative samples.
            d. Compute gradients manually (no autograd).
            e. Update embeddings using scatter_add.
        4. After training, find similar words to test embeddings.
    """
    comptime dtype = DType.float32
    var python_sys = Python.import_module("sys")
    seed()

    # ── Device Detection ──────────────────────────────────────────────────────
    comptime if has_accelerator():
        print("Device: GPU")
    else:
        print("Device: CPU")

    # ── Load and Preprocess Data ─────────────────────────────────────────────
    print("Loading reviews from:", TRAIN_FOLDER)
    var sampler = NegativeSampler()
    var tokenizer = sampler.init_tokenizer_and_datasets(TRAIN_FOLDER)
    ref tokenized_reviews = sampler.tokenized_reviews
    ref all_tokens = sampler.concatenated_tokens
    var vocabulary_size = len(tokenizer)
    print("Vocabulary size: ", vocabulary_size)

    # ── Initialize Model ──────────────────────────────────────────────────────
    var model = Word2Vec[dtype](
        vocabulary_size, EMBEDDING_DIMENSION, init_seed=RANDOM_SEED_INPUT
    )
    var fixed_target = Tensor[dtype].zeros(NUM_NEGATIVE_SAMPLES + 1)
    fixed_target[0] = 1

    comptime if has_accelerator():
        model.to_gpu()
        fixed_target = fixed_target.to_gpu()

    var num_reviews = min(MAX_REVIEWS_TO_USE, len(tokenized_reviews))

    print("\nTraining Configuration:")
    print("  Training Iterations:  ", TRAINING_ITERATIONS)
    print("  Embedding Dimension:  ", EMBEDDING_DIMENSION)
    print("  Learning Rate:        ", LEARNING_RATE)
    print("  Vocabulary Size:      ", vocabulary_size)
    print("  Reviews Used:         ", num_reviews, "of", len(tokenized_reviews))

    var initial_weight_sum = model.input_embeddings.sum[
        track_grad=False
    ]().item()

    for iteration in range(TRAINING_ITERATIONS):
        shuffle(tokenized_reviews)
        var num_reviews = min(MAX_REVIEWS_TO_USE, len(tokenized_reviews))

        for review_idx in range(num_reviews):
            ref review = tokenized_reviews[review_idx]

            for word_position in range(len(review)):
                var left = slice(
                    max(0, word_position - CONTEXT_WINDOW_SIZE), word_position
                )
                var right = slice(
                    word_position + 1,
                    min(len(review), word_position + CONTEXT_WINDOW_SIZE),
                )
                if left.start == left.end and right.start == right.end:
                    continue

                var ctx = review[left].copy()
                ctx.extend(review[right].copy())
                if len(ctx) == 0:
                    continue

                var tgt = generate_negative_samples(
                    review, word_position, all_tokens, NUM_NEGATIVE_SAMPLES
                )

                var scores = model.forward(ctx, tgt)
                model.step(
                    scores, fixed_target, ctx, tgt, Float32(LEARNING_RATE)
                )

            if (review_idx + 1) % 25 == 0:
                var pct = Float32(review_idx + 1) / Float32(num_reviews) * 100.0
                python_sys.stdout.write(
                    "\rIteration:"
                    + String(iteration)
                    + "  Progress: "
                    + String(Int(pct))
                    + "%"
                )

        var final_sum = model.input_embeddings.sum[track_grad=False]().item()
        print(
            "\n  ✓ Weight sum change:",
            final_sum - initial_weight_sum,
            "(should be != 0 — proves gradients are flowing!)",
        )

        print("\nWords similar to 'terrible':")
        comptime if has_accelerator():
            var cpu_emb = model.input_embeddings.to_cpu()
            for item in find_similar_words(tokenizer, cpu_emb, "terrible"):
                print("  ", item[0], "→ similarity:", item[1])
        else:
            for item in find_similar_words(
                tokenizer, model.input_embeddings, "terrible"
            ):
                print("  ", item[0], "→ similarity:", item[1])


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def find_similar_words(
    tokenizer: Tokenizer,
    ref embeddings: Tensor[DType.float32],
    query_word: String = "beautiful",
    top_n: Int = 10,
) raises -> List[Tuple[String, Float32]]:
    """Find words with embeddings most similar to the query word.

    Uses Euclidean distance (L2 norm) to measure similarity.
    Lower distance = more similar.

    Args:
        tokenizer: The tokenizer with vocabulary.
        embeddings: The word embedding matrix.
        query_word: Word to find similar words for.
        top_n: Number of similar words to return.

    Returns:
        List[Tuple[String, Float32]]: Top N similar words with similarity scores.
    """
    # Get embedding for the query word
    var query_ids = tokenizer.encode(query_word)
    var query_embedding = embeddings.gather[track_grad=False](query_ids)

    # If multiple tokens (unlikely for single word), average them
    if len(query_ids) > 1:
        query_embedding = query_embedding.mean[track_grad=False](
            IntArray(0), keepdims=True
        )

    # Compute Euclidean distance to all other words
    var differences = embeddings - query_embedding
    var distances = (
        (differences * differences)
        .sum[track_grad=False](IntArray(1))
        .sqrt[track_grad=False]()
    )

    # Build results list (convert to negative for sorting)
    var results = List[Tuple[String, Float32]](capacity=len(tokenizer))
    for ref pair in tokenizer.word_to_id.items():
        var word = pair.key
        var index = pair.value

        # Skip the query word itself and words with underscores (symbols)
        if word == query_word or "_" in word:
            continue

        results.append((word, -distances[index]))

    # Sort by similarity (ascending order, so negative distance works)
    sort[cmp_fn=compare_by_similarity](results)

    # Return top N results
    var top_results = List[Tuple[String, Float32]](
        capacity=min(top_n, len(results))
    )
    for k in range(min(top_n, len(results))):
        top_results.append(results[k])

    return top_results^


def compare_by_similarity(
    left: Tuple[String, Float32], right: Tuple[String, Float32]
) capturing -> Bool:
    """Comparison function for sorting by similarity score."""
    return left[1] > right[1]
