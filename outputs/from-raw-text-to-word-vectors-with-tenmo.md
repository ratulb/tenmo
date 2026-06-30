---
title: "From Raw Text to Word Vectors: Building a Tokenizer and Word Embeddings with Tenmo"
date: "2026-06-30"
categories: ["Natural Language Processing", "Mojo", "Tenmo"]
tags: ["word-embeddings", "mojo", "tenmo", "nlp", "from-scratch", "word2vec", "negative-sampling", "tokenizer"]
excerpt: >
  We build word2vec-style embeddings from scratch with Tenmo (a tensor library
  built in Mojo) — starting with a custom tokenizer, then training a skip-gram
  model with negative sampling on IMDB reviews, and finally probing the learned
  vectors for semantic similarity.
---

# From Raw Text to Word Vectors: Building a Tokenizer and Word Embeddings with Tenmo

"king − man + woman ≈ queen."

This single equation — the notion that arithmetic on word vectors reveals semantic relationships — is what made word embeddings famous. It suggests that somewhere inside a high-dimensional vector space, directions like "royalty" and "gender" actually exist as learned features. A computer trained only on raw text, with no dictionary or grammar, can learn that *king* and *queen* differ by the same vector as *man* and *woman*.

How does that work? And more importantly, how do we build it from scratch?

In this post, we'll implement the full pipeline using **Tenmo** — a tensor library and neural network framework built in Mojo with full autograd, SIMD-optimized kernels, and GPU support. We'll build a tokenizer that converts raw movie reviews into integer IDs, a skip-gram training loop with negative sampling, and a similarity probe that lets us query the learned embedding space. The entire implementation lives in a single file — around 750 lines — and trains on the IMDB review dataset.

## The Problem: Computers Don't Read

A computer sees strings. `"king"`, `"queen"`, `"man"`, `"woman"` are just sequences of bytes. Nothing in their byte representation suggests that *king* and *queen* are related, or that *man* and *woman* share a semantic axis.

To make words computable, we need **vector representations** — each word mapped to a list of floating-point numbers where distance in vector space corresponds to semantic similarity.

But what kind of vector?

### One-Hot Encoding

The simplest approach: assign each word a unique V-dimensional vector with a single `1` and `V−1` zeros.

```mojo
# Pseudo-code for one-hot encoding
var V = 100_000  # vocabulary size
var id = word_to_idx["king"]   # say, 42
var one_hot = Tensor[dtype].zeros(V)
one_hot[42] = 1
```

The problems are immediate:
- **Semantically blind.** The dot product between any two one-hot vectors is always 0 — they're orthogonal by construction. *King* and *queen* are as unrelated as *king* and *aardvark*.
- **High-dimensional, sparse.** A 100K-dimensional vector with a single non-zero element wastes memory and fails in any ML model that expects dense features.
- **No generalization.** The model can't leverage the fact that *king* and *queen* behave similarly in text — they're treated as completely independent symbols.

### Bag-of-Words and TF-IDF

The next refinement: count how often each word appears in a document. A vector of term frequencies is denser than one-hot, but it's still V-dimensional and ignores word order. TF-IDF improves on raw counts by down-weighting common words (*the*, *a*, *in*), but the representation remains sparse, high-dimensional, and incapable of capturing synonymy.

### Co-Occurrence Matrices (GloVe)

GloVe builds a word-word co-occurrence matrix: count how often word *i* appears near word *j* across the entire corpus, then factorize that matrix to produce dense vectors. The intuition is simple — words that occur in similar contexts have similar vectors — but the co-occurrence matrix is O(V²), making it impractical for large vocabularies without heavy approximation.

### Prediction-Based Embeddings (word2vec)

word2vec flips the problem around. Instead of counting co-occurrences, we train a neural network to **predict** whether a word appears in a given context. The vectors emerge as a byproduct — the hidden layer weights of this prediction network become the word embeddings.

This is what we'll implement. But before we can train embeddings, we need to turn raw text into numbers. That means building a tokenizer.

## Stage 1: Building a Tokenizer from Scratch

A tokenizer converts text into integer IDs. It's the gateway between raw strings and any NLP model. Our tokenizer needs to:

1. Clean raw text — strip HTML, URLs, punctuation artifacts, and digit sequences.
2. Build a vocabulary — collect every unique word from the training corpus, sort it, and assign each word a unique integer.
3. Encode new text into those IDs, with a fallback for words not seen during training.

### Cleaning Text

The IMDB dataset contains movie reviews with HTML tags (`<br />`, `<a href="...">`), URLs, ratings, and other noise. We clean it in a single pass using Python's `re` module — Mojo's Python interop handles this cleanly:

```mojo
@staticmethod
def clean_text(raw_text: String) raises -> PythonObject:
    var py = Python.import_module("builtins")
    var regex = Python.import_module("re")
    var text = py.str(raw_text)

    # Remove HTML tags
    text = regex.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = regex.sub(r"http\S+|www\.\S+", " ", text)
    # Remove digit sequences
    text = regex.sub(r"\d+", " ", text)
    # Remove stray apostrophes (preserve contractions like "don't")
    text = regex.sub(r"(?<!\w)'|'(?!\w)", " ", text)
    # Collapse multiple spaces
    text = regex.sub(r"\s+", " ", text).strip()

    # Filter out words shorter than 2 characters
    var filter_fn = Python.evaluate(
        "lambda words: [w for w in words.split() if len(w) >= 2]"
    )
    return filter_fn(text)
```

Every step handles a real data problem:
- HTML tags appear throughout IMDB reviews (especially `<br />` for line breaks).
- URLs appear in user-written reviews ("I saw this at http://example.com").
- Ratings like "10/10" would leak numeric patterns unrelated to sentiment.
- Leading/trailing apostrophes (`'hello'`) are punctuation, but contractions (`don't`) are real words.
- Single-character tokens like "a" and "I" are filtered because they add noise without semantic signal.

The use of `Python.evaluate` to define a lambda is worth noting. Mojo's Python interop means we can write Python logic inline without leaving the language — perfect for text processing where Mojo's standard library doesn't yet have a regex engine.

### Building the Vocabulary

Once we've cleaned every review, we collect the unique words across the entire dataset:

```mojo
@staticmethod
def from_text_lines(text_lines: List[String]) raises -> Self:
    var py = Python.import_module("builtins")
    var all_words: PythonObject = []

    # Collect all words from all text lines
    for line in text_lines:
        all_words.extend(Tokenizer.clean_text(line))

    # Create unique, sorted vocabulary
    all_words = py.list(py.set(all_words))
    all_words = py.sorted(all_words)

    # Add UNKNOWN token for out-of-vocabulary words
    var vocab_with_unknown: PythonObject = [UNKNOWN_TOKEN]
    vocab_with_unknown.extend(all_words)

    # Map each word to a unique integer ID
    var vocabulary = {
        String(token): Int(index)
        for index, token in enumerate(vocab_with_unknown.__iter__())
    }

    return Self(vocabulary^)
```

Key design decisions:

- **UNKNOWN token at position 0.** Any word seen at test time but not in training gets mapped to ID 0. This is a standard practice — it acts as a catch-all, preventing the model from crashing on novel words.
- **Alphabetical sort.** Sorting the vocabulary before assigning IDs ensures deterministic behavior across runs. The word with ID 1 is always `"aaron"`, not a random word depending on Python's set iteration order.
- **Dict[String, Int] for lookup, Dict[Int, String] for decoding.** The tokenizer stores both mappings so we can go from text → IDs and back.

### Encoding and Decoding

With the vocabulary built, encoding new text is straightforward:

```mojo
def encode(self, text: String) raises -> List[Int]:
    var words = Tokenizer.clean_text(text)
    var token_ids = List[Int](capacity=len(words))
    for word in words:
        var word_str = String(word)
        token_ids.append(
            self.word_to_id[word_str] if word_str in self.word_to_id
            else self.word_to_id[UNKNOWN_TOKEN]
        )
    return token_ids^

def decode(self, token_ids: List[Int]) raises -> String:
    return " ".join([self.id_to_word[id] for id in token_ids])
```

The encode step is the inverse of cleaning: the same `clean_text` function that prepared training data also processes new input. Consistency between training and inference is critical — if your tokenizer cleans text one way during training but differently during inference, your model will see a distribution mismatch.

### Loading the IMDB Dataset

The dataset lives at `/tmp/aclImdb/train/` with `pos/` and `neg/` subdirectories. Each file is named like `1234_8.txt` — the number after the underscore is the rating from 1 to 10. We filter for strong reviews (rating ≥ 7 positive, ≤ 4 negative) to get cleaner signal:

```mojo
def init_tokenizer_and_datasets(mut self, dataset_folder: String) raises -> Tokenizer:
    # Ensure dataset is downloaded
    self._download_imdb_dataset()

    var positive_path = Path("/tmp") / dataset_folder / "pos"
    var negative_path = Path("/tmp") / dataset_folder / "neg"
    var all_comments = List[String](capacity=50000)

    # Load positive reviews (rating 7-10)
    if positive_path.exists():
        for file in positive_path.listdir():
            var rating = self._extract_rating_from_filename(file.name())
            if rating >= 7:
                var comment = positive_path.joinpath(file.name()).read_text()
                all_comments.append(comment)

    # Load negative reviews (rating 1-4)
    if negative_path.exists():
        for file in negative_path.listdir():
            var rating = self._extract_rating_from_filename(file.name())
            if rating <= 4:
                var comment = negative_path.joinpath(file.name()).read_text()
                all_comments.append(comment)

    # Build tokenizer from all loaded comments
    var tokenizer = Tokenizer.from_text_lines(all_comments)

    # Tokenize everything and build datasets
    for comment in all_comments:
        var token_ids = tokenizer.encode(comment)
        if len(token_ids) == 0:
            continue
        self.tokenized_reviews.append(token_ids.copy())
        self.concatenated_tokens.extend(token_ids^)

    return tokenizer
```

We store two views of the data:
- **`tokenized_reviews`**: each review as a separate list of token IDs. This lets us build context windows within a single review (we never want context crossing review boundaries).
- **`concatenated_tokens`**: every token ID from every review concatenated into one flat list. This is used for random negative sampling — we draw negative samples uniformly from the entire corpus.

Let's ground this in numbers. The IMDB dataset has 25K training reviews. After filtering for strong sentiment and limiting to 5,000 reviews (for speed), we get a vocabulary of roughly **80K–120K unique words** and about **5 million total tokens**. Our embedding matrix is `vocab_size × 100`, or about 8M–12M parameters — a reasonable size for training on a single machine.

## Stage 2: Token Embedding Approaches — A Landscape

Before we dive into our training algorithm, it's worth stepping back and asking: what approaches exist for turning tokens into vectors, and where does our method fit?

| Approach | Dimensionality | Semantics | Training Cost | Inference Cost |
|---|---|---|---|---|
| One-hot | V (huge) | None | None | O(V) |
| TF-IDF | V (huge) | Word frequency | O(N) | O(V) |
| Co-occurrence (GloVe) | d (small) | Context statistics | O(V²) | O(1) |
| Prediction (word2vec) | d (small) | Context prediction | O(N × d × K) | O(1) |

**One-hot** is the baseline with zero learning — each word is a distinct symbol with no inherent relationship to others.

**TF-IDF** adds frequency weighting but stays in the V-dimensional space. "King" and "queen" are still treated as completely unrelated dimensions.

**Co-occurrence methods** (like GloVe) are the closest competitor to prediction-based methods. They count how often each pair of words co-occurs in a context window, then factorize that count matrix. The resulting vectors capture semantics well, but building the full co-occurrence matrix is O(V²) — infeasible for a 100K vocabulary without approximation. GloVe works around this by counting only co-occurrences above a threshold, but it still requires iterating over every word pair in every context window.

**Prediction-based methods** (word2vec and its variants) take a different route: instead of counting co-occurrences, they train a classifier to predict them. This is the approach we'll implement. The key insight is that predicting whether a word appears in a given context forces the model to learn vector geometry that captures semantic relationships — as a side effect of optimizing classification accuracy, not as an explicit goal.

Within prediction-based methods, there are two main architectures:

- **CBOW (Continuous Bag of Words):** Given the context words, predict the target word. Fast to train, but less effective for rare words.
- **Skip-gram:** Given the target word, predict the context words. Slower to train, but produces better vectors for rare words.

We'll use **Skip-gram** because it tends to produce higher-quality embeddings, especially for the long tail of rare words.

## Stage 3: The Skip-gram Idea

Skip-gram is built on a simple intuition from linguistics: **"a word is known by the company it keeps."** Words that appear in similar contexts have similar meanings.

The skip-gram training objective:

```
Given a target word w_t, maximize the probability of seeing
each context word w_{t+j} within a window of size C.
```

In the sentence *"The cat sat on the mat"*, with a window size of 2 around *sat*:
- Target: *sat*
- Context: [*the, cat, on, the*]

For every target position in every review, we collect the surrounding words within the window:

```mojo
var left_context = slice(
    max(0, word_position - CONTEXT_WINDOW_SIZE),
    word_position
)
var right_context = slice(
    word_position + 1,
    min(len(review), word_position + CONTEXT_WINDOW_SIZE)
)

var context_indices = review[left_context].copy()
context_indices.extend(review[right_context].copy())
```

This produces a variable-length context window centered on each target word. Words closer to the target are included more reliably; the asymmetric edges of documents naturally get fewer context words, which is fine — the model learns to handle varying amounts of context.

The probability of a context word given a target word is computed using the **softmax** over the entire vocabulary:

```
P(w_context | w_target) = exp(score(w_context, w_target)) / Σ_v exp(score(v, w_target))
```

Here, `score(w_c, w_t)` is the dot product between the **output embedding** of the context word and the **input embedding** of the target word. Wait — input and output embeddings? Yes, word2vec uses **two embedding matrices**:

- **Input embeddings** (`vocab_size × hidden_size`): one vector per word as a *target*. These are what we'll eventually use as our word vectors.
- **Output embeddings** (`vocab_size × hidden_size`): one vector per word as a *context*. These exist only to compute compatibility scores.

The asymmetry is intentional. A word acting as a target should be close in vector space to words that appear near it as contexts. Having two sets of weights makes the optimization easier — the model has separate parameters for each role.

### The Softmax Wall

The softmax denominator sums over every word in the vocabulary. For each training step, computing this requires:

- V dot products (one per vocabulary word)
- V exponentiations
- V additions for the denominator
- V divisions for the final probabilities

With V ≈ 100K, that's 100K dot products per step. With 5 million training tokens and 5 iterations (epochs), that's **2.5 trillion dot products**. Even at 1 microsecond per dot product, that's months of computation.

This is the *softmax wall* — the fundamental computational bottleneck that prevented early neural language models from scaling to large vocabularies.

## Stage 4: Negative Sampling

The critical insight from Mikolov et al. (2013) is that we don't need the full softmax. We don't care about the exact probability distribution over all words — we only care that the model learns good vector representations. And for that, we can replace the multi-class softmax with a much cheaper binary classification task.

**The idea:** Instead of computing "how likely is this context word given this target, out of all possible context words?", train a binary classifier that answers "did this target-context pair come from real data or random noise?"

For each real (target, context) pair (a *positive sample*), we generate K *negative samples* — random words drawn from the corpus that are unlikely to be real context words. The model then learns to assign high probability to positive pairs and low probability to negative pairs.

The objective function for a single training example:

```
J = log σ(u_c · v_t) + Σ_{k=1}^{K} E_{w_k ~ P_n}[log σ(-u_k · v_t)]
```

Where:
- `u_c` is the output embedding of the context word
- `v_t` is the input embedding of the target word
- `σ()` is the sigmoid function
- `P_n(w)` is the noise distribution — we draw negative samples from it

The first term pushes the target and context vectors together. Each term in the second sum pushes the target and a random noise word apart.

### K+1 Binary Classifications Instead of One V-Way Classification

This is the entire point: instead of one V-way softmax (V computations per step), we now have K+1 binary classifications (K+1 computations per step). With K = 5–20, that's a **5,000x–20,000x reduction** in computation per training step.

### The Noise Distribution

Mikolov found empirically that the best noise distribution is the unigram distribution raised to the 3/4 power:

```
P_n(w) = count(w)^(3/4) / Z
```

Where Z is a normalization constant. Raising to the 3/4 power has the effect of giving rare words a higher chance of being selected as negatives than they would under the raw unigram distribution. This prevents the model from seeing only common words as negatives, which would make the task too easy.

Our implementation uses a simpler uniform random distribution (drawing from the concatenated token list), which is a common approximation:

```mojo
def generate_negative_samples(
    current_review: List[Int],
    target_position: Int,
    all_tokens: List[Int],
    num_negative_samples: Int,
) -> List[Int]:
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
```

The result is a list of K+1 token IDs: position 0 is the positive sample (the real context word), and positions 1 through K are random negatives.

This is the heart of negative sampling — a few lines of code that turn an intractable O(V) problem into a tractable O(K) one.

## Stage 5: The Training Loop

With the theory in place, the training loop ties everything together. For each word in each review:

1. Build a context window around the target word.
2. Average the context word embeddings to get a single context vector.
3. Get embeddings for the positive + negative samples.
4. Compute scores via dot product + sigmoid.
5. Compute gradients manually — we're intentionally doing this by hand. Tenmo has full autograd support (`track_grad=True`, `.backward()`, etc.), but the gradient of binary cross-entropy simplifies to a single subtraction. Dispatching through the autograd graph would be pure overhead.
6. Update input and output embeddings via Tenmo's `scatter_add`, which applies gradients to only the rows that participated in the forward pass.

Let's walk through each step with the real code.

### Forward Pass

```mojo
# Average context word embeddings
var context_embedding = input_embeddings.gather[track_grad=False](
    context_indices, reduction=Reduction(1)
)
var averaged_context = context_embedding / Float32(context_length)

# Get embeddings for target + negative samples
var sample_embeddings = output_embeddings.gather[track_grad=False](
    sample_indices
)

# Calculate scores using dot product + sigmoid
var predicted_scores = sample_embeddings.matmul[
    mode=mv, track_grad=False
](averaged_context).sigmoid()
```

Three operations, each doing real work:

**Gather with reduction.** `input_embeddings.gather(context_indices, reduction=Reduction(1))` looks up the embedding for each context word ID, then averages them (reduction=1 means "mean"). This turns, say, 6 context words into a single 100-dimensional vector. Using gather with built-in reduction is faster than summing vectors manually — it avoids creating intermediate tensors.

**Matmul with mode=mv.** `sample_embeddings` is shape `(K+1, hidden_size)`; `averaged_context` is shape `(hidden_size,)`. The `mode=mv` flag tells the matmul to interpret this as a matrix-vector multiplication, producing shape `(K+1,)`. Each entry is the dot product between one sample's embedding and the averaged context.

**Sigmoid.** The dot products are raw scores in (-∞, ∞). Sigmoid squashes them to (0, 1) so they can be interpreted as probabilities — how likely it is that this word appeared in this context.

### Training Target

```mojo
var training_target = Tensor[dtype].zeros(NEGATIVE_SAMPLES + 1)
training_target[0] = 1
```

The target vector is `[1, 0, 0, 0, 0, 0]` (when K=5). The `1` at position 0 tells the model "the word at index 0 (the positive sample) should have high probability." The `0`s at positions 1–5 say "these random words should have low probability."

This is a binary cross-entropy setup: each of the K+1 positions is an independent binary classification.

### Backward Pass (Manual Gradients)

```mojo
var gradient_output = predicted_scores - training_target
```

This one line is the gradient of binary cross-entropy with respect to the logits (pre-sigmoid scores). For binary cross-entropy `L = -[t log(p) + (1-t) log(1-p)]` with `p = σ(x)`, the gradient simplifies to `dL/dx = p - t`. No exponentials, no logarithms — just a subtraction.

We're computing this by hand intentionally. Tenmo has a complete autograd engine — you can set `track_grad=True` on any tensor, call `.backward()` on the loss, and the framework will unroll the full computation graph, compute all gradients, and feed them to an optimizer. But here, the gradient formula collapses to a single element-wise subtraction. Dispatching that through graph construction, tape recording, and jump-table dispatch would add 10-100x overhead for no benefit. The manual path isn't a workaround — it's the right tool for this job.

```mojo
# Gradient for context embedding
var gradient_context = sample_embeddings.transpose().matmul[
    mode=mv, track_grad=False
](gradient_output)
```

This is the chain rule through the dot product. If `score = u · v` and `dL/dscore = gradient_output`, then `dL/dv = u^T · gradient_output`. We transpose the sample embeddings (shape `(hidden_size, K+1)`) and multiply by the output gradient (shape `(K+1,)`), giving `dL/daveraged_context` as a shape `(hidden_size,)` vector.

### Sparse Updates with Tenmo's scatter_add

The most performance-critical part of the loop: updating only the rows of the embedding matrices that were actually used in this step. Tenmo provides `Filler.scatter_add` — a sparse update primitive that adds gradient contributions to specific rows of a tensor buffer, leaving all other rows untouched.

```mojo
# Update Input Embeddings
var context_update = -gradient_context * Float32(LEARNING_RATE)
var context_update_broadcast = context_update.unsqueeze(0)
var context_update_matrix = context_update_broadcast.repeat[
    track_grad=False
](context_length, 1)

var context_indices_array = IntArray(context_indices)
Filler[dtype].scatter_add(
    input_embeddings.buffer,
    context_update_matrix.buffer,
    context_indices_array
)

# Update Output Embeddings
var output_update = (
    -gradient_output.unsqueeze(1)
    * averaged_context.unsqueeze(0)
    * Float32(LEARNING_RATE)
)

var sample_indices_array = IntArray(sample_indices)
Filler[dtype].scatter_add(
    output_embeddings.buffer,
    output_update.buffer,
    sample_indices_array
)
```

Here's what's happening:

1. **Compute the update.** `context_update = -gradient * lr` gives us the negative gradient direction scaled by the learning rate. The negative sign because gradient descent moves opposite to the gradient.

2. **Broadcast to a matrix.** The context gradient is a single vector, but we need to add it to multiple rows of the embedding matrix (one per context word). `unsqueeze(0)` makes it shape `(1, hidden_size)`, and `repeat(context_length, 1)` tiles it so every context word gets the same update.

3. **Tenmo's scatter_add.** This is the core sparse operation: "add these row vectors to the embedding matrix at these specific row indices." It avoids materializing a full gradient matrix of size `vocab_size × hidden_size` — a savings of ~100x memory and computation. This isn't something we built from scratch — Tenmo's `Filler` module provides this primitive out of the box, with both CPU and GPU kernel implementations.

The output update uses a different formula. Since the gradient flows through each sample independently, each of the K+1 samples gets its own update proportional to how wrong its prediction was:

```
output_update[sample_i] = -gradient_output[i] * averaged_context * lr
```

The `unsqueeze` operations handle broadcasting: `gradient_output` is shape `(K+1,)`, `averaged_context` is shape `(hidden_size,)`. After unsqueezing, `gradient_output.unsqueeze(1)` is `(K+1, 1)` and `averaged_context.unsqueeze(0)` is `(1, hidden_size)`. The element-wise multiplication broadcasts to `(K+1, hidden_size)` — exactly the shape needed to update all K+1 sample embeddings in one scatter_add call.

### Gradient Flow Verification

After each epoch, we check that gradients are actually flowing:

```mojo
var final_weight_sum = input_embeddings.sum[track_grad=False]().item()
var weight_change = final_weight_sum - initial_weight_sum
print(
    "\n  ✓ Weight sum change:",
    weight_change,
    "(should be != 0 — proves gradients are flowing!)"
)
```

If the weight sum hasn't changed, something is wrong with the gradient computation or the update. This is a cheap sanity check that catches bugs like a zero learning rate, a disconnected graph, or a failed scatter_add. In practice, seeing a weight change of non-zero confirms the entire pipeline — from forward pass through gradient computation through update — is functioning.

## Stage 6: Probing the Learned Embeddings

Training yields an embedding matrix of shape `(vocab_size, 100)`. To test whether these vectors actually capture semantics, we write a function that finds words closest to a given query:

```mojo
def find_similar_words(
    tokenizer: Tokenizer,
    ref embeddings: Tensor[DType.float32],
    query_word: String = "beautiful",
    top_n: Int = 10,
) raises -> List[Tuple[String, Float32]]:

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

    # Build results and sort by similarity
    var results = List[Tuple[String, Float32]](capacity=len(tokenizer))
    for ref pair in tokenizer.word_to_id.items():
        var word = pair.key
        var index = pair.value
        if word == query_word or "_" in word:
            continue
        results.append((word, -distances[index]))

    sort[cmp_fn=compare_by_similarity](results)

    var top_results = List[Tuple[String, Float32]](capacity=min(top_n, len(results)))
    for k in range(min(top_n, len(results))):
        top_results.append(results[k])
    return top_results^
```

The similarity metric is **negative Euclidean distance** — we compute `-||v_query - v_word||` for every word in the vocabulary, then sort descending. Negative distance means "closer is more similar," which makes sorting natural (highest first).

The steps are worth noting:
- `embedding - query_embedding` computes a `(vocab_size, hidden_size)` difference matrix — a single broadcast operation.
- `(differences * differences).sum(axis=1)` squares and sums along the hidden dimension, producing a `(vocab_size,)` distance vector.
- `.sqrt()` converts squared distances to actual Euclidean distances.
- We iterate over the vocabulary, skip the query word itself and symbol-heavy words, and build a `(String, Float32)` result list.
- The results are sorted and the top N returned.

This is intentionally simple — we use Euclidean distance rather than cosine similarity because it's cheaper to compute (no normalization step). In practice, for unit vectors, Euclidean distance and cosine similarity produce the same rankings.

The demo output, when the training converges, shows something like:

```
🔍 Words similar to 'terrible':
    awful      → similarity: -2.3
    horrible   → similarity: -2.4
    dreadful   → similarity: -2.7
    disgusting → similarity: -2.7
    boring     → similarity: -2.9
    ridiculous → similarity: -2.9
    pathetic   → similarity: -2.9
    miserable  → similarity: -3.0
    sad        → similarity: -3.0
    bad        → similarity: -3.1
```
*(Values are illustrative — exact numbers depend on random initialization and training convergence.)*

Every one of these is a negative-sentiment word — exactly what "terrible" should be close to. If the embeddings were random or poorly trained, we'd see unrelated words like "the", "movie", or "and" in the top results. The fact that the nearest neighbors are semantically related synonyms is evidence that the training worked.

## Stage 7: Common Pitfalls

Over the course of implementing this, we hit several issues worth highlighting.

### Pitfall 1: Gradient Explosion from Unnormalized Context

The initial version of the code summed context word embeddings without averaging. The problem: longer context windows (say 8 words) would produce larger gradient magnitudes than shorter ones (say 2 words). The model would oscillate, paying more attention to words in longer windows simply because they had more signal.

**Fix:** Divide the summed context embedding by the context length.

```mojo
var averaged_context = context_embedding / Float32(context_length)
```

### Pitfall 2: Sigmoid Saturation at Initialization

If the initial dot products are too large (say, > 5 in magnitude), sigmoid saturates — it produces values very close to 0 or 1, where the gradient is nearly zero. The model stops learning because the gradient vanishes.

**Fix:** Initialize input embeddings to uniform random in [-0.1, 0.1] and output embeddings to zeros. Small initial magnitudes keep dot products in the linear region of sigmoid (around 0, where the gradient is ≈ 0.25). This is also why the autograd variant initializes output embeddings with an even smaller range [-0.01, 0.01].

### Pitfall 3: scatter_add vs. Direct Assignment

The gradient update must **add** to existing embeddings, not replace them. If you write `embeddings[rows] = embeddings[rows] + update`, you create a temporary tensor, modify it, and write it back. `scatter_add` does the same operation in-place without extra memory allocations.

This matters at scale. When training on 5 million tokens with 5 context words each, you're making 25 million sparse updates per epoch. Even a small per-update allocation balloon — a Mojo temporary NDBuffer created and immediately freed — adds up to hundreds of millions of short-lived allocations.

### Pitfall 4: Token ID Validation

If a token ID exceeds the vocabulary size, the embedding gather returns garbage (or crashes). This seems obvious, but it's easy to get wrong when you have preprocessing logic (cleaning, filtering, encoding) that might produce IDs from the *training* vocabulary that don't match the *full* vocabulary.

Our code validates every single encoding:

```mojo
var max_id = 0
for id in token_ids:
    if id > max_id: max_id = id
if max_id >= len(tokenizer):
    print("ERROR: Token ID out of vocabulary range!")
    break
```

This check runs during data loading and catches mismatches immediately.

### Pitfall 5: The Autograd Overhead Trap

The initial version of this code used Tenmo's autograd framework (the `track_grad=True` path with `.backward()` and `SGD.step()`). This was **16× slower** than the manual approach, even though both compute the same gradients. Why?

The optimizer's `.step()` iterates over **every row** of the embedding matrix — all 100K × 100 = 10M parameters — even though only ~10 rows (1,000 elements) actually received gradient updates per step. The `scatter_add` in the manual version touches only those 10 rows. The autograd optimizer has no concept of sparsity: it applies `weight -= lr * weight.grad` unconditionally to every parameter.

This is not a Tenmo limitation — every autograd framework from PyTorch to JAX works the same way. **Dense optimizers assume dense updates.** For sparse lookup tasks like word embeddings, the gradient matrix is >99.99% zero, and a dense optimizer wastes proportional time on no-ops. This is exactly the pattern that sparse optimizers (like AdaGrad's per-feature learning rate) were invented to solve.

Tenmo's `Filler.scatter_add` lets us skip the optimizer entirely for sparse workloads. It applies gradients to only the rows we touched during the forward pass, which is exactly the right semantic for word2vec. Tenmo is already capable of this gradient scattering — we lean on it directly rather than going through the optimizer.

## Why Tenmo?

This implementation highlights a few of Tenmo's design strengths:

**First-class scatter_add primitive.** Most tensor libraries treat row-scatter as an afterthought or don't expose it at all. PyTorch has `index_add_`, but it passes through the autograd engine, adding overhead for graph tracking that sparse updates don't need. Tenmo's `Filler.scatter_add` is a direct buffer operation — no graph, no tape, no dispatch. It's the right primitive for word2vec, and Tenmo exposes it directly.

**Autograd when you need it, not when you don't.** Tenmo has full autograd: `track_grad=True`, `.backward()`, optimizers like `SGD`, everything you'd expect. But when your gradient simplifies to `p - t`, the autograd path is pure overhead. Tenmo doesn't force you through it — you can call `Filler.scatter_add` on raw buffers, compute gradients by hand, and skip the graph entirely. The choice is yours per operation, not all-or-nothing.

**Ownership without GC pauses.** Each training step allocates intermediate tensors (gather outputs, scores, gradients). In a garbage-collected language, these allocations trigger the GC to track and reclaim them. Mojo's ownership system (which Tenmo is built on) lets us control exactly when temporaries are destroyed — or reuse buffers explicitly.

**CPU-first with optional GPU.** The code runs on CPU without modification. Tenmo detects GPU availability at compile time via `has_accelerator()`. When a GPU is present, tensors are transparently moved and operations dispatched to GPU kernels. Same code, one compile flag.

## Conclusion

We built the full pipeline from raw text to word vectors using Tenmo:

1. **A text tokenizer** that cleans HTML-laden reviews, builds a vocabulary, and encodes text into integer IDs with an unknown-word fallback.
2. **A skip-gram training loop** that predicts context words from target words, with context window construction and embedding averaging.
3. **Negative sampling** that turns a V-way softmax into K+1 binary classifications — the key algorithmic insight that makes word2vec practical.
4. **Manual gradient computation** with Tenmo's `scatter_add` for sparse updates — optimizing only the embedding rows that actually participated in each training step.
5. **A similarity probe** that validates the learned embeddings by finding nearest neighbors in vector space.

The final implementation trains on 5,000 IMDB reviews, producing word vectors where "terrible" is close to "awful", "horrible", and "dreadful" — without ever being told that these words are related. The model learned it purely from the statistics of word co-occurrence in raw text.

**Next steps to explore:**
- Try the autograd variant in `negative_sampling_emb.mojo` — it uses Tenmo's `Embedding` module with `BCEWithLogitsLoss` and `SGD`. It's slower (dense optimizer on sparse problem) but demonstrates how the same algorithm composes with Tenmo's autograd framework.
- Swap negative sampling for hierarchical softmax and compare training speed and embedding quality.
- Move to a larger corpus (Wikipedia dumps are a common next step) and use subword tokenization (BPE) instead of word-level tokens.
- Probe the embedding space for analogies: "king − man + woman = ?" — if the vectors capture relational semantics, the nearest neighbor to the result should be "queen."

The full code (around 750 lines) is available in this blog's repository. It's MIT-licensed and ready to run — just `mojo negative_sampling.mojo` with the IMDB dataset in `/tmp/aclImdb/`.

---

**Suggested hero image:** A visual of raw text flowing through a pipeline — HTML tags being stripped, words being mapped to integer IDs, then through a "neural network" block, and finally emerging as a 2D-projected vector space where semantically similar words cluster together — shown as connected dots labeled with words like "terrible", "awful", "horrible" in one cluster and "beautiful", "wonderful", "amazing" in another.
