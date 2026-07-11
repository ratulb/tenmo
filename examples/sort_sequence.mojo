"""A two-layer single-head transformer that learns to sort 8-digit sequences.

Inspired by ATTN/11 ("Paper Tape Is All You Need", https://github.com/dbrll/ATTN-11),
a transformer written in PDP-11 assembly language. This is the same idea — a minimal
end-to-end transformer that solves a concrete task — implemented in Mojo + tenmo.
The task is sorting rather than reversal — a fundamentally harder problem because
the routing depends on token VALUES, not just positions.

Usage:
  pixi run mojo -I . examples/sort_sequence.mojo

Expected behavior (default comptime values, D_MODEL=32, training WITH duplicates):
  • Best seq accuracy: 95.9%–100% over 10 runs (mean 98.8%, median 99.3%)
  • 9/10 runs ≥97.1% best seq accuracy
  • 1/10 hits 100%
  • Momentum overshoots visible at epochs 8–14, model always recovers
  Each epoch takes ~7 seconds on CPU (two attention layers + 5000 samples);
  total runtime ~3 minutes for 25 epochs.

── Why Two Layers ──────────────────────────────────────────────────────────────────

Sorting benefits enormously from a second attention layer because the task
decomposes naturally into two stages:

  Stage 1 (Layer 1): Compare — Each position's query attends to all keys to
  determine which tokens are smaller/larger. The context vector encodes relative
  ordering information. The residual preserves the original token identity.

  Stage 2 (Layer 2): Route — Each output position (0 = smallest, 7 = largest)
  reads the enriched representations from stage 1 and decides which input token
  to copy. Since each position now has access to both "what token is here" and
  "how does it compare to others," the routing decision is well-informed.

A single layer must do both comparison AND routing within one attention
computation — forcing the same 64-dim Q/K/V vectors to serve dual roles.
Two layers lift this bottleneck, letting each layer specialize.

Empirical result at epoch 10 with D_MODEL=64, 5000 samples:
  • 1 layer: 75.3% token / 18.5% seq
  • 2 layers: 99.5% token / 95.9% seq
That is a ~10x improvement in sequence accuracy, far exceeding what the
parameter increase alone (14K → 26K) would suggest.

── Data ──────────────────────────────────────────────────────────────────────────

Data is generated procedurally — no files, no Python dependency.

Each training sample is built from a shuffled [0..9] list using random repeat
counts to create sequences with possible duplicates:

1. Start with the list [0, 1, 2, ..., 9] (vocab = 10 digits).
2. Shuffle it into a random permutation using std.random.shuffle on List[Int].
3. Walk through the shuffled list. For each value, sample a repeat count
   (1-3) via std.random.random_ui64. Append that many copies to the sequence.
4. Stop when SEQ_LEN elements are filled. Later values in the shuffled list
   are skipped (they don't fit).

This produces sequences where some digits repeat 2-3 times and others don't
appear at all — e.g., [3, 3, 7, 1, 1, 1, 9, 0]. The target y is the sorted
version of the same sequence via bubble sort: [0, 1, 1, 1, 3, 3, 7, 9].

Inputs and targets are stored as Tensor[idx_dtype] of shape
(num_samples, seq_len) = (NUM_TRAIN, SEQ_LEN) for training,
(NUM_TEST, SEQ_LEN) for testing.

Why this matters: training on permutations only (all unique) means the model
never sees duplicate values, so it can't learn to handle ties. Training with
repeats forces the model to learn true value-based sorting that generalizes
to arbitrary token frequencies.

Why sorting is harder than reversal: reversal routes by position (t → T-1-t), a
fixed pattern. Sorting requires comparing token values and routing each token to
its rank position — the attention pattern must be content-dependent, not fixed.

── Architecture ──────────────────────────────────────────────────────────────────

All shapes assume batch size B and sequence length T = SEQ_LEN.

1. Token Embedding      tok_embed: Embedding(VOCAB -> D_MODEL)
2. Position Embedding   pos_embed: Embedding(SEQ_LEN -> D_MODEL)
3. Layer 1 QKV          w_q1, w_k1, w_v1: Linear(D_MODEL, D_MODEL, bias=False)
4. Layer 1 Attention    Scaled Dot-Product + Residual
5. Layer 2 QKV          w_q2, w_k2, w_v2: Linear(D_MODEL, D_MODEL, bias=False)
6. Layer 2 Attention    Scaled Dot-Product + Residual
7. Output Projection    w_out: Linear(D_MODEL, VOCAB, bias=True)

Total parameters (defaults): 7050 (93% from the six QKV projections).

── Forward Pass Walkthrough (per batch) ──────────────────────────────────────────

The forward pass stacks two identical attention blocks. Each block is:
  x ← x + softmax(Q @ K^T / sqrt(d_model)) @ V

The first block computes initial pairwise comparisons between token values. The
residual preserves the original token embeddings, so the second block can refine
the routing decisions based on the first block's output. This is more powerful
than a single block because the second attention layer sees a richer
representation — each position's vector now encodes both its own token identity
and weighted context from the first layer.

Step 1 — Embeddings
───────────────────

Input batch.features has shape (B, SEQ_LEN) — each element is an integer token ID.

  var tok_x = tok_embed(batch.features)     # (B, SEQ_LEN, D_MODEL)

The embedding layer is a learned lookup table of shape (VOCAB, D_MODEL). For each
position in each sequence, it fetches row[v] from the table. Every position now
has a D_MODEL-dimensional dense vector instead of a raw integer.

Self-attention has no inherent notion of order (it's permutation-invariant), so we
inject position information:

  var pos_ids = make_pos_indices(B, SEQ_LEN)    # (B, SEQ_LEN) — [0,1,2,...,SEQ_LEN-1]
  var x = tok_x + pos_embed(pos_ids)        # (B, SEQ_LEN, D_MODEL)

pos_embed is a second Embedding(SEQ_LEN, D_MODEL) that maps each position index to
a learned D_MODEL-dim vector. Element-wise addition fuses identity and position.

Step 2 — QKV Projections
────────────────────────

  Q = w_q(x)    # (B, SEQ_LEN, D_MODEL)
  K = w_k(x)    # (B, SEQ_LEN, D_MODEL)
  V = w_v(x)    # (B, SEQ_LEN, D_MODEL)

Each is x @ W^T with no bias. Xavier/Glorot init to keep activation scales stable.

Step 3 — Scaled Dot-Product Attention
─────────────────────────────────────

  K_t = K.transpose(0, 2, 1)               # (B, D_MODEL, SEQ_LEN)
  scores = Q @ K_t                          # (B, SEQ_LEN, SEQ_LEN)
  scores /= sqrt(D_MODEL)                   # scale to prevent softmax saturation
  attn = softmax(scores, axis=-1)           # (B, SEQ_LEN, SEQ_LEN) — rows sum to 1

Each cell scores[b, i, j] = dot(query[b,i], key[b,j]) — how much output position
i should attend to input position j. For sorting, the model must learn to route
the smallest-valued input token to output position 0, the second smallest to
output position 1, etc. This means the attention pattern depends on the token
values, not just on fixed position pairs.

Step 4 — Context
────────────────

  ctx = attn @ V                            # (B, SEQ_LEN, D_MODEL)

Each output position is a weighted sum of ALL value vectors, with weights from the
attention distribution. For sorting, output position 0 should attend primarily to
whichever input position holds the smallest value; output position 1 to the input
with the second smallest value, etc.

Step 5 — Residual Connection
────────────────────────────

  combined = x + ctx                        # (B, SEQ_LEN, D_MODEL)

The residual serves two purposes:
  • Gradient flow — gradients bypass the attention layer during backprop,
    preventing vanishing gradients.
  • Identity preservation — the model can pass through unmodified information
    if attention doesn't help.

Step 6 — Output Projection
──────────────────────────

  logits = w_out(combined)                  # (B, SEQ_LEN, VOCAB)
  preds  = logits.argmax(axis=-1)           # (B, SEQ_LEN) — predicted token IDs

w_out is Linear(D_MODEL -> VOCAB, bias=True). Each position's D_MODEL-dim
representation is projected to a VOCAB-dim logits vector. argmax picks the most
likely token ID.

── Loss ─────────────────────────────────────────────────────────────────────────

tenmo's CrossEntropyLoss expects (N, C, d1, ..., dk) layout — class dimension
at position 1. Our logits are (B, T, V), so we transpose dims 1 and 2:

  logits_v = logits.transpose(0, 2, 1)      # (B, VOCAB, SEQ_LEN)
  loss = criterion(logits_v, batch.labels)

Internally cross-entropy:
  1. Applies log_softmax over the class dimension (C=VOCAB).
  2. For each position in each sample, indexes the log-probability at the true
     target class.
  3. Negates and averages over all positions and batch items → scalar loss.

The loss is minimized when the model assigns high probability to the correct token
at each of the SEQ_LEN positions for every sequence in the batch.

── Backward Pass ─────────────────────────────────────────────────────────────────

loss.backward() traverses the computation graph in reverse topological order,
dispatching to the backward handler for each op via an integer op_code jump table
(backpropagation.mojo:357). Key gradients:

  • softmax backward: dL/dlogits from cross-entropy, propagated through softmax.
  • matmul backward: dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC.
  • add backward: gradient is split to both the residual path and the attention
    path (ctx).
  • transpose backward: inverse transpose (same permutation).
  • embedding backward: gradient accumulates into the table at the looked-up
    indices.
  • linear backward: computes dL/dW, dL/db, dL/dx.

── Optimizer ─────────────────────────────────────────────────────────────────────

SGD with momentum (lr=0.01, momentum=0.9, weight_decay=1e-4):
  v = momentum * v - lr * (grad + weight_decay * param)
  param += v

All parameters across 8 modules are updated every step.

── Eval ──────────────────────────────────────────────────────────────────────────

Same forward pass with track_grad=False — no computation graph is built.
model.eval() disables gradient tracking in all layers at compile time.

Accuracy.token_accuracy and Accuracy.sequence_accuracy provide the metrics:
  • Token accuracy: fraction of individual positions correct.
  • Sequence accuracy: fraction of sequences where ALL positions are correct.

The test set is 1000 samples (no shuffle), so evaluation is deterministic.

── Q&A: Feeding Shorter Sequences (Inference) ─────────────────────────────────────

Q: Can the current model (SEQ_LEN=8) sort a 5-digit sequence at inference?

A: Yes — the model handles any length ≤ SEQ_LEN without code changes.

Why it works:
  • tok_embed is element-wise — looks up each token ID independently, no length
    dependence. A (1, 5) input produces (1, 5, D_MODEL) embeddings.
  • pos_embed maps indices 0..4, all within its (8, 32) table — fine.
  • QKV projections are per-position Linear layers — length-agnostic.
  • Attention produces (1, 5, 5) scores — smaller matrix, same arithmetic.
  • w_out projects each position independently — works on any T.

What wouldn't work: a sequence longer than 8 — pos_embed would OOB on index 8+.

What you'd need to change to actually use it:
  1. The inference input tensor: Shape(1, 5) instead of Shape(1, SEQ_LEN).
  2. Position IDs: make_pos_indices(1, 5) instead of make_pos_indices(1, SEQ_LEN).

The catch: training on length 8 and inferring on length 5 is not useful — the
model learned ranking patterns for 8 elements, not 5. You'd want to set SEQ_LEN=5
and retrain, or train on mixed lengths.

── Q&A: Max Context Length vs Fixed Sequence Length ───────────────────────────────

Q: In this implementation, sequence length == context length. If we had a max
   context length with variable-length sequences, would we need padding? How
   would that work?

A: Yes — the current code has no distinction between "model capacity" (max
   context) and "data length." SEQ_LEN is a single compile-time constant that
   sets the position embedding table size, the training data length, and every
   intermediate tensor shape to (B, SEQ_LEN, ...).

   To support variable-length sequences up to a max context length M:

   1. Padding token
      Add a special <PAD> token ID outside the data vocabulary. For example,
      VOCAB=11 where 10 is <PAD>. Shorter sequences are right-padded to length
      M. The position embedding grows to Embedding(M, D_MODEL) — padded
      positions still get a position vector, but the attention mask suppresses
      them.

   2. Attention mask
      Without masking, padded positions attend to real tokens and vice versa —
      the model learns garbage associations. The fix is a binary mask applied
      before softmax:
        scores = Q @ K^T / sqrt(d_model)             # (B, M, M)
        mask = make_padding_mask(lengths)             # (B, M, M) — 1 for real, 0 for pad
        scores = scores * mask + (-inf) * (1 - mask)  # masked → -inf → softmax zeros

   3. Loss mask
      Cross-entropy must ignore padding positions. In PyTorch this is
      ignore_index=PAD_ID. In tenmo, if CrossEntropyLoss doesn't support it,
      you'd manually mask:
        loss = (raw_loss * padding_mask).sum() / mask.sum()

   Why the current code doesn't need this: every training example is exactly
   SEQ_LEN tokens, so there is never a padding token, never a variable-length
   batch, and no masking at any point. Tying SEQ_LEN to both capacity and data
   shape is fine when every input has the same length.

── Q&A: Could This Be Done With Position Embeddings Alone? ────────────────────────

Q: Would the model work with only position embeddings — no token embeddings?

A: No — and this is even more true for sorting than for reversal.

   For reversal, position embeddings alone fail because they carry zero token
   identity information. Every input would look like [0, 1, 2, ..., 7], so the
   model could never distinguish one permutation from another.

   For sorting, the problem is even deeper. Sorting requires comparing token
   VALUES and ordering them — a purely content-based operation. Without token
   embeddings, there are no values to compare. The attention mechanism has
   nothing to route by. The model could only emit a fixed output independent
   of the input.

   The value pathway is:
     ctx[t] = Σⱼ attn[t, j] · w_v(token_embed[j] + pos_embed[j])

   Only token_embed[j] tells the model "which digit is at position j." For
   sorting, the attention weights must encode "which input position has the
   k-th smallest value" — and the only way to determine "smallest" is through
   the token embedding vectors. Without them, the model has no basis for
   comparison.

── Q&A: Training With Duplicates ──────────────────────────────────────────────────

Q: This version trains on sequences WITH duplicate values (not just permutations).
   Why? And how does the model handle ties?

A: Training on permutations only (all unique) means the model never learns to
   handle repeated digits. The learned solution would be "attend to the k-th
   smallest distinct value" — but with duplicates, the same value occupies
   multiple output slots (e.g., [0, 0, 1, 1, 3, 3, 9, 9]).

   The data generation algorithm solves this by constructing sequences from a
   shuffled [0..9] with random repeat counts (1-3 per value from random_ui64).
   This exposes the model to triples (e.g., [4,4,4]), pairs, singletons, and
   homogeneous sequences ([0,0,0,0,0,0,0,0]) during training.

   How the model learns to handle ties: since identical tokens share the same
   embedding vector, the attention mechanism sees them as interchangeable.
   For sorting, the output needs k copies of value v in adjacent positions.
   The model learns that when two input positions both contain "1", output
   positions k and k+1 should each predict "1". The attention weights distribute
   across the copies — each copy attends to both, and the residual ensures
   both output positions have access to a "1" embedding. The two-layer
   architecture helps: layer 1 detects "how many copies of each value exist,"
   and layer 2 assigns them to consecutive output slots.

── Q&A: Why Sorting Is Harder Than Reversal ──────────────────────────────────────

Q: Why does sorting take longer to converge than reversal? And why do two
   transformer layers help so dramatically (10x faster than one)?

A: Sorting is a fundamentally harder learning problem for a transformer.

   Reversal requires learning a fixed positional mapping: position t routes to
   position T-1-t. The attention pattern is an anti-diagonal — the same for
   every input. The model only needs to embed position identity into Q and K
   such that dot(Q[t], K[T-1-t]) is maximal.

   Sorting requires learning a content-dependent mapping: the smallest input
   token (wherever it sits) must route to output position 0, the second smallest
   to position 1, etc. The attention pattern changes for every input sequence.
   The Q and K vectors must encode token VALUES (not just position), and the
   dot products must reflect a comparison (ordering) relationship.

   This means:
   • The attention weights cannot be hardcoded by position — they must be
     computed dynamically per input.
   • The model must learn embeddings where dot(Q[a], K[b]) correlates with
     "is token[a] smaller than token[b]?"
   • The V vectors must preserve enough information about token identity for
     the output projection to produce the correct sorted token.

   Concretely, for reversal the model can memorize 8 pairwise attention targets
   (0→7, 1→6, etc.). For sorting it must learn to compare any two of the 10
   possible token values and route correctly — a combinatorial generalization
   problem, not a pattern-memorization one.

   Why two layers help: the first attention layer can focus on pairwise value
   comparisons — each position's context vector encodes which tokens are smaller
   or larger than itself. The residual preserves the original token embedding.
   The second attention layer reads these enriched representations and makes
   the final routing decision. This two-stage decomposition (compare → route)
   is more sample-efficient than forcing a single layer to do both simultaneously.

── Q&A: Scaling to Longer/Wider Vocabularies ─────────────────────────────────────

Q: What changes to sort 20 numbers (vocab 0..19, seq_len 20)?

A: Two mandatory changes, plus more aggressive hyperparameter tuning than for
   reversal.

Mandatory — compile-time constants:
  VOCAB = 10  →  VOCAB = 20
  SEQ_LEN = 8 →  SEQ_LEN = 20

These propagate automatically:
  • tok_embed:  Embedding(VOCAB, D_MODEL)        — table (10,32) → (20,32)
  • pos_embed:  Embedding(SEQ_LEN, D_MODEL)       — table (8,32)  → (20,32)
  • w_out:      Linear(D_MODEL, VOCAB)             — weight (32,10) → (32,20)
  • generate_sort_data(SEQ_LEN, VOCAB)             — already parameterized

Total parameters (at D_MODEL=64) jump from 26378 to ~55K (two layers at D_MODEL=128):
  tok_embed    640 → 1280
  pos_embed    512 → 1280
  w_q1/k1/v1  each 4096 → 16384 (at D_MODEL=128)
  w_q2/k2/v2  each 4096 → 16384 (at D_MODEL=128)
  w_out        650 → 2580

Likely hyperparameter changes (sorting is harder than reversal, and the input
space grows factorially):
  • D_MODEL = 32 → 64 or 128 (sorting needs more capacity for value comparison)
  • NUM_TRAIN = 2000 → 20000+ (far more diverse sequences)
  • NUM_EPOCHS = 10 → 100+ (much slower convergence)
  • LR = 0.01 → 0.001 (larger models need smaller learning rates)

What does NOT need changing:
  • generate_sort_data — takes vocab, seq_len as runtime params
  • make_pos_indices — works for any B, T
  • Accuracy.token_accuracy / Accuracy.sequence_accuracy — generic over dimensions
  • Softmax, CrossEntropyLoss, matmul, transpose, residual, SGD — all
    dimension-agnostic

Main risk: d_model=32 will almost certainly be insufficient for sorting 20
tokens at 20 positions. The Q/K vectors must embed a total ordering over 20
values — much harder than 10 values for reversal. Bump d_model to at least 64.
"""

from tenmo.tensor import Tensor
from tenmo.optim import SGD
from tenmo.net import Linear
from tenmo.embedding import Embedding
from tenmo.crossentropy import CrossEntropyLoss
from tenmo.softmax import Softmax
from tenmo.intarray import IntArray
from tenmo.shapes import Shape
from tenmo.dataloader import TensorDataset
from tenmo.mnemonics import DEFAULT_INDEX_DTYPE as idx_dtype
from tenmo.accuracy import Accuracy
from std.random import shuffle, random_ui64
from std.time import perf_counter_ns
from std.math import sqrt


comptime dtype = DType.float32

comptime VOCAB = 10
comptime SEQ_LEN = 8
comptime D_MODEL = 32
comptime BATCH_SIZE = 32
comptime NUM_TRAIN = 5000
comptime NUM_TEST = 1000
comptime NUM_EPOCHS = 25
comptime LR = 0.01
comptime MOMENTUM = 0.9


def generate_sort_data(
    num_samples: Int,
    seq_len: Int,
    vocab: Int,
    allow_duplicates: Bool = True,
) -> Tuple[Tensor[idx_dtype], Tensor[idx_dtype]]:
    var x = Tensor[idx_dtype](Shape(num_samples, seq_len))
    var y = Tensor[idx_dtype](Shape(num_samples, seq_len))

    var base = List[Int](capacity=vocab)
    for k in range(vocab):
        base.append(k)

    for i in range(num_samples):
        var raw = List[Int](capacity=seq_len)

        if allow_duplicates:
            var perm = base.copy()
            shuffle(perm)
            var vi = 0
            while len(raw) < seq_len:
                var remaining = seq_len - len(raw)
                var max_r = remaining
                if max_r > 3:
                    max_r = 3
                var repeat = Int(random_ui64(1, UInt64(max_r)))
                if repeat > remaining:
                    repeat = remaining
                for _ in range(repeat):
                    raw.append(perm[vi % vocab])
                vi += 1
        else:
            var perm = base.copy()
            shuffle(perm)
            for j in range(seq_len):
                raw.append(perm[j])

        for j in range(seq_len):
            x[i, j] = Scalar[idx_dtype](raw[j])

        var n = seq_len
        for a in range(n):
            for b in range(a + 1, n):
                if raw[a] > raw[b]:
                    var tmp = raw[a]
                    raw[a] = raw[b]
                    raw[b] = tmp

        for j in range(seq_len):
            y[i, j] = Scalar[idx_dtype](raw[j])

    return (x^, y^)


def make_pos_indices(batch_size: Int, seq_len: Int) -> Tensor[idx_dtype]:
    return Tensor[idx_dtype].stack(
        [
            Tensor[idx_dtype].arange(Scalar[idx_dtype](seq_len))
            for _ in range(batch_size)
        ]
    )


def compute_accuracy(
    logits: Tensor[dtype], targets: Tensor[idx_dtype]
) raises -> Tuple[Float64, Float64]:
    return (
        Accuracy[dtype].token_accuracy(logits, targets),
        Accuracy[dtype].sequence_accuracy(logits, targets),
    )


def make_repeat_tests(vocab: Int, seq_len: Int) -> Tensor[idx_dtype]:
    var T = Tensor[idx_dtype](Shape(5, seq_len))
    for t in range(5):
        for p in range(seq_len):
            T[t, p] = Scalar[idx_dtype](0)

    var p = 0
    var v = 0
    while p < seq_len:
        var r = min(2, seq_len - p)
        for _ in range(r):
            T[0, p] = Scalar[idx_dtype](v % vocab)
            p += 1
        v += 1

    p = 0
    v = 0
    while p < seq_len:
        var r = min(3, seq_len - p)
        for _ in range(r):
            T[1, p] = Scalar[idx_dtype](v % vocab)
            p += 1
        v += 1

    for p in range(seq_len):
        T[2, p] = Scalar[idx_dtype](0)

    for p in range(seq_len):
        T[3, p] = Scalar[idx_dtype](max(vocab - 1 - p, 0))

    p = 0
    v = 0
    while p < seq_len:
        var r = min(2 + (v % 2), seq_len - p)
        for _ in range(r):
            T[4, p] = Scalar[idx_dtype](v % vocab)
            p += 1
        v += 1

    return T^

def test_descriptions(seq_len: Int) -> List[String]:
    var d = List[String](capacity=5)
    d.append("pairs  (2-repeats of sequential low values)")
    d.append("triples (3-repeats of sequential low values)")
    d.append("all-0")
    d.append("unique (descending from V-1)")
    d.append("mixed  (2s and 3s interleaved)")
    return d^


def main() raises:
    print("=" * 58)
    print("Sequence Sort Transformer - Tenmo")
    print("=" * 58)
    print("  d_model =", D_MODEL, "| vocab =", VOCAB, "| seq_len =", SEQ_LEN)
    print("  train =", NUM_TRAIN, "| test =", NUM_TEST)
    print("  batch_size =", BATCH_SIZE, "| epochs =", NUM_EPOCHS)
    print("  lr =", LR, "| momentum =", MOMENTUM)
    print()

    # ── Data ──────────────────────────────────────────────────────────
    print("Generating", NUM_TRAIN, "training +", NUM_TEST, "test samples...")
    var train_x, train_y = generate_sort_data(NUM_TRAIN, SEQ_LEN, VOCAB)
    var test_x, test_y = generate_sort_data(NUM_TEST, SEQ_LEN, VOCAB)

    var demo_count = min(5, NUM_TEST)
    var demo_x = Tensor[idx_dtype](Shape(demo_count, SEQ_LEN))
    var demo_y = Tensor[idx_dtype](Shape(demo_count, SEQ_LEN))
    for idx in range(demo_count):
        for t in range(SEQ_LEN):
            demo_x[idx, t] = test_x[idx, t]
            demo_y[idx, t] = test_y[idx, t]

    var train_dataset = TensorDataset[idx_dtype, idx_dtype](train_x, train_y)
    var test_dataset = TensorDataset[idx_dtype, idx_dtype](test_x, test_y)
    var train_loader = train_dataset.into_loader(
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    var test_loader = test_dataset.into_loader(
        batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )
    print()

    # ── Model ──────────────────────────────────────────────────────────
    var tok_embed = Embedding[dtype, idx_dtype](VOCAB, D_MODEL, init_method="uniform")
    var pos_embed = Embedding[dtype, idx_dtype](SEQ_LEN, D_MODEL, init_method="uniform")
    var w_q1 = Linear[dtype](D_MODEL, D_MODEL, bias=False, init_method="xavier")
    var w_k1 = Linear[dtype](D_MODEL, D_MODEL, bias=False, init_method="xavier")
    var w_v1 = Linear[dtype](D_MODEL, D_MODEL, bias=False, init_method="xavier")
    var w_q2 = Linear[dtype](D_MODEL, D_MODEL, bias=False, init_method="xavier")
    var w_k2 = Linear[dtype](D_MODEL, D_MODEL, bias=False, init_method="xavier")
    var w_v2 = Linear[dtype](D_MODEL, D_MODEL, bias=False, init_method="xavier")
    var w_out = Linear[dtype](D_MODEL, VOCAB, bias=True, init_method="xavier")

    # ── Parameter collection ──────────────────────────────────────────

    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.extend(tok_embed.parameters())
    params.extend(pos_embed.parameters())
    params.extend(w_q1.parameters())
    params.extend(w_k1.parameters())
    params.extend(w_v1.parameters())

    params.extend(w_q2.parameters())
    params.extend(w_k2.parameters())
    params.extend(w_v2.parameters())

    params.extend(w_out.parameters())

    var optimizer = SGD[dtype](
        params^, lr=LR, momentum=MOMENTUM, weight_decay=1e-4
    )

    var total_params = 0
    for p in optimizer.parameters:
        total_params += p[].numels()
    print("Parameters:", total_params)
    print()

    var criterion = CrossEntropyLoss[dtype, idx_dtype]()

    # ── Training ──────────────────────────────────────────────────────
    print("Training...")
    print()
    var best_seq_acc = Float64(0.0)
    var t_start = perf_counter_ns()

    for epoch in range(NUM_EPOCHS):
        var t_epoch_start = perf_counter_ns()
        # ── Train phase ──
        tok_embed.train()
        pos_embed.train()
        w_q1.train()
        w_k1.train()
        w_v1.train()
        w_q2.train()
        w_k2.train()
        w_v2.train()
        w_out.train()
        criterion.train()

        train_loader.reset()
        var epoch_loss = Float64(0.0)

        while train_loader.__has_next__():
            ref batch = train_loader.__next__()
            var B = batch.features.shape()[0]

            var tok_x = tok_embed(batch.features)
            var pos_ids = make_pos_indices(B, SEQ_LEN)
            var x = tok_x + pos_embed(pos_ids)

            # Layer 1
            var Q = w_q1(x)
            var K = w_k1(x)
            var K_t = K.transpose(0, 2, 1)
            var scores = Q.matmul(K_t) / sqrt(Float32(D_MODEL))
            var attn = Softmax[dtype].forward(scores, axes=IntArray(-1))
            var ctx = attn.matmul(w_v1(x))
            x = x + ctx

            # Layer 2
            Q = w_q2(x)
            K = w_k2(x)
            K_t = K.transpose(0, 2, 1)
            scores = Q.matmul(K_t) / sqrt(Float32(D_MODEL))
            attn = Softmax[dtype].forward(scores, axes=IntArray(-1))
            ctx = attn.matmul(w_v2(x))
            x = x + ctx

            var logits = w_out(x)

            var logits_v = logits.transpose(0, 2, 1)
            var loss = criterion(logits_v, batch.labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += Float64(loss.item()) * Float64(B)

        # ── Eval phase ──
        tok_embed.eval()
        pos_embed.eval()
        w_q1.eval()
        w_k1.eval()
        w_v1.eval()
        w_q2.eval()
        w_k2.eval()
        w_v2.eval()
        w_out.eval()
        criterion.eval()

        test_loader.reset()
        var token_acc = Float64(0.0)
        var seq_acc = Float64(0.0)
        var n_batches = 0

        while test_loader.__has_next__():
            ref batch = test_loader.__next__()
            var B = batch.features.shape()[0]

            var tok_x = tok_embed(batch.features)
            var pos_ids = make_pos_indices(B, SEQ_LEN)
            var x = tok_x + pos_embed(pos_ids)

            # Layer 1
            var Q = w_q1(x)
            var K = w_k1(x)
            var K_t = K.transpose(0, 2, 1)
            var scores = Q.matmul(K_t) / sqrt(Float32(D_MODEL))
            var attn = Softmax[dtype].forward[track_grad=False](
                scores, axes=IntArray(-1)
            )
            var ctx = attn.matmul(w_v1(x))
            x = x + ctx

            # Layer 2
            Q = w_q2(x)
            K = w_k2(x)
            K_t = K.transpose(0, 2, 1)
            scores = Q.matmul(K_t) / sqrt(Float32(D_MODEL))
            attn = Softmax[dtype].forward[track_grad=False](
                scores, axes=IntArray(-1)
            )
            ctx = attn.matmul(w_v2(x))
            x = x + ctx

            var logits = w_out(x)

            var ta, sa = compute_accuracy(logits, batch.labels)
            token_acc += ta
            seq_acc += sa
            n_batches += 1

        token_acc /= Float64(n_batches)
        seq_acc /= Float64(n_batches)
        if seq_acc > best_seq_acc:
            best_seq_acc = seq_acc

        var epoch_time = Float64(perf_counter_ns() - t_epoch_start) / 1e9
        print(
            "  Epoch",
            epoch + 1,
            "/",
            NUM_EPOCHS,
            "| loss:",
            epoch_loss / Float64(NUM_TRAIN),
            "| tok_acc:",
            token_acc * 100,
            "%",
            "| seq_acc:",
            seq_acc * 100,
            "%",
            "|",
            epoch_time,
            "s",
        )

    # ── Final results ─────────────────────────────────────────────────
    var total_time = Float64(perf_counter_ns() - t_start) / 1e9
    print()
    print("=" * 58)
    print("Best sequence accuracy:", best_seq_acc * 100, "%")
    print("Total time:", total_time, "s")
    print()

    tok_embed.eval()
    pos_embed.eval()
    w_q1.eval()
    w_k1.eval()
    w_v1.eval()
    w_q2.eval()
    w_k2.eval()
    w_v2.eval()
    w_out.eval()

    print("Sample predictions (test set):")
    for idx in range(demo_count):
        var inp = Tensor[idx_dtype](Shape(1, SEQ_LEN))
        for t in range(SEQ_LEN):
            inp[0, t] = demo_x[idx, t]

        var tok_x = tok_embed(inp)
        var pos_ids = make_pos_indices(1, SEQ_LEN)
        var x = tok_x + pos_embed(pos_ids)

        # Layer 1
        var Q = w_q1(x)
        var K = w_k1(x)
        var K_t = K.transpose(0, 2, 1)
        var scores = Q.matmul(K_t) / sqrt(Float32(D_MODEL))
        var attn = Softmax[dtype].forward[track_grad=False](
            scores, axes=IntArray(-1)
        )
        var ctx = attn.matmul(w_v1(x))
        x = x + ctx

        # Layer 2
        Q = w_q2(x)
        K = w_k2(x)
        K_t = K.transpose(0, 2, 1)
        scores = Q.matmul(K_t) / sqrt(Float32(D_MODEL))
        attn = Softmax[dtype].forward[track_grad=False](
            scores, axes=IntArray(-1)
        )
        ctx = attn.matmul(w_v2(x))
        x = x + ctx

        var logits = w_out(x)
        var preds = logits.argmax[idx_dtype](axis=-1)

        print("  Example", idx + 1)
        for t in range(SEQ_LEN):
            print(
                "    Pos",
                t,
                "| in:",
                Int(demo_x[idx, t]),
                "-> pred:",
                Int(preds[0, t]),
                "(target:",
                Int(demo_y[idx, t]),
                ")",
            )
        print()

    # ── Repeat-pattern tests ──
    print("Repeat-pattern inference tests:")

    var repeat_vals = make_repeat_tests(VOCAB, SEQ_LEN)
    var test_descs = test_descriptions(SEQ_LEN)

    for test_idx in range(5):
        var inp = Tensor[idx_dtype](Shape(1, SEQ_LEN))
        for t in range(SEQ_LEN):
            inp[0, t] = repeat_vals[test_idx, t]

        var tok_x = tok_embed(inp)
        var pos_ids = make_pos_indices(1, SEQ_LEN)
        var x = tok_x + pos_embed(pos_ids)

        var Q = w_q1(x)
        var K = w_k1(x)
        var K_t = K.transpose(0, 2, 1)
        var scores = Q.matmul(K_t) / sqrt(Float32(D_MODEL))
        var attn = Softmax[dtype].forward[track_grad=False](
            scores, axes=IntArray(-1)
        )
        var ctx = attn.matmul(w_v1(x))
        x = x + ctx

        Q = w_q2(x)
        K = w_k2(x)
        K_t = K.transpose(0, 2, 1)
        scores = Q.matmul(K_t) / sqrt(Float32(D_MODEL))
        attn = Softmax[dtype].forward[track_grad=False](
            scores, axes=IntArray(-1)
        )
        ctx = attn.matmul(w_v2(x))
        x = x + ctx

        var logits = w_out(x)
        var preds = logits.argmax[idx_dtype](axis=-1)

        var expected = List[Int](capacity=SEQ_LEN)
        for t in range(SEQ_LEN):
            expected.append(Int(repeat_vals[test_idx, t]))
        var n = SEQ_LEN
        for a in range(n):
            for b in range(a + 1, n):
                if expected[a] > expected[b]:
                    var tmp = expected[a]
                    expected[a] = expected[b]
                    expected[b] = tmp

        print("  Test", test_idx + 1, ":", test_descs[test_idx])
        print("    Input  : [", end="")
        for t in range(SEQ_LEN):
            if t > 0:
                print(", ", end="")
            print(Int(repeat_vals[test_idx, t]), end="")
        print("]")
        print("    Predict: [", end="")
        for t in range(SEQ_LEN):
            if t > 0:
                print(", ", end="")
            print(Int(preds[0, t]), end="")
        print("]")
        print("    Expect : [", end="")
        for t in range(SEQ_LEN):
            if t > 0:
                print(", ", end="")
            print(expected[t], end="")
        print("]")
        var ok = True
        for t in range(SEQ_LEN):
            if Int(preds[0, t]) != expected[t]:
                ok = False
        print("    Correct:", ok)
        print()

    print("Done.")
