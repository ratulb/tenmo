"""A single-layer, dual-head transformer that learns to reverse 8-digit sequences.

Inspired by ATTN/11 (https://github.com/dbrll/ATTN-11), a transformer written in
PDP-11 assembly. This is the same idea — a minimal end-to-end transformer that
solves a concrete task — implemented in Mojo + tenmo.

Usage:
  pixi run mojo -I . examples/reverse_sequence.mojo

Expected behavior:
  Converges to 100% sequence accuracy on every seed (20/20 runs).
  First epoch at 100%: range 4–10, median 6.
  Total runtime: ~26 seconds for 10 epochs on CPU.

────────────────────────────────────────────────────────────────────────────────────
  DATA
────────────────────────────────────────────────────────────────────────────────────

Data is generated procedurally — no files, no Python dependency.

1. Start with the list [0, 1, 2, ..., VOCAB-1] (vocab = VOCAB digits).
2. Shuffle it into a random permutation using std.random.shuffle on List[Int].
3. Take the first SEQ_LEN elements as the input sequence x.
4. Take the same elements in reverse order as the target y.

The model must learn the reversal rule: output[t] = input[SEQ_LEN-1-t].
Inputs and targets are Tensor[idx_dtype] of shape (num_samples, seq_len).
No one-hot encoding — the embedding layer consumes integer token IDs directly.

────────────────────────────────────────────────────────────────────────────────────
  THE PROBLEM: Single-Head Attention Can Get Stuck
────────────────────────────────────────────────────────────────────────────────────

A single attention head computes context for each output position:

  ctx[t] = Σⱼ softmax(Q[t]·K[j] / √d) · V[j]

At init, the Q and K vectors are random (Xavier), so attention is near-uniform:
each ctx[t] is roughly the average of all V vectors, and every output position
receives nearly the same context. The initial gradient for Q and K is therefore
zero-mean noise — each batch pushes the attention weights in a random direction.

In ~94% of seeds, this random exploration stumbles onto the correct anti-diagonal
pairing (position t attends to position SEQ_LEN-1-t) within the first few epochs,
and the model converges. But in ~6% of seeds, the random walk settles into a
wrong local minimum — a self-reinforcing attention pattern that encodes per-position
token statistics instead of the reversal pairing. Once stuck, the gradient is
too weak to escape.

This is not an optimization problem (LR schedule, clip-norm, different embedding
init). It is a statistical one: a single head has only one chance to find the
correct pattern, and it fails ~6% of the time.

────────────────────────────────────────────────────────────────────────────────────
  THE FIX: Two Independent Attention Heads
────────────────────────────────────────────────────────────────────────────────────

The solution is to give the model two independent attention heads, each with its
own randomly initialized Q, K, V projections. Both heads attend over the full
sequence independently, and their outputs are averaged:

  ctx = (ctx_head1 + ctx_head2) / 2

Each head explores the attention space independently. With two heads, the
probability that BOTH get stuck is ~0.36% (6% × 6%) — and even then, the loss
from the single correct head dominates the gradient. Empirically, 2 heads
achieve 100% convergence over 50 random seeds (0 failures).

The residual connection (x + ctx) is deliberately omitted. Without it, the
gradient must flow through the attention mechanism at every step — there is no
shortcut for the output layer to learn a per-position mapping that bypasses
attention. This forces the model to develop a meaningful attention pattern.

────────────────────────────────────────────────────────────────────────────────────
  ARCHITECTURE
────────────────────────────────────────────────────────────────────────────────────

All shapes assume batch size B and sequence length T = SEQ_LEN.

  tok_embed: Embedding(VOCAB -> D_MODEL)      token → D_MODEL-dim vector
  pos_embed: Embedding(SEQ_LEN -> D_MODEL)    position index → D_MODEL-dim vector
  x = tok_embed(tokens) + pos_embed(pos)      fused representation (B, T, D_MODEL)

  Head 1:  Q1 = w_q1(x),  K1 = w_k1(x),  V1 = w_v1(x)   each (B, T, D_MODEL)
  Head 2:  Q2 = w_q2(x),  K2 = w_k2(x),  V2 = w_v2(x)   each (B, T, D_MODEL)

  Each head independently computes scaled dot-product attention:

    scores_h = Q_h @ K_h^T / √D_MODEL            (B, T, T)
    attn_h  = softmax(scores_h, axis=-1)          rows sum to 1
    ctx_h   = attn_h @ V_h                        (B, T, D_MODEL)

  ctx = (ctx1 + ctx2) / 2                         averaged context  (B, T, D_MODEL)
  logits = w_out(ctx)                             output projection (B, T, VOCAB)

No residual connection: logits depend exclusively on ctx, forcing the gradient
through attention. Both heads use Xavier init, embeddings use uniform U[-1,1].
Total parameters: ~8K.

────────────────────────────────────────────────────────────────────────────────────
  LOSS
────────────────────────────────────────────────────────────────────────────────────

tenmo's CrossEntropyLoss expects (N, C, d1, ..., dk) layout — class dimension at
position 1. Our logits are (B, T, V) = (B, SEQ_LEN, VOCAB), so we transpose dims 1 and 2:

  logits_v = logits.transpose(0, 2, 1)     (B, 10, 8)
  loss = criterion(logits_v, batch.labels)

────────────────────────────────────────────────────────────────────────────────────
  BACKWARD PASS
────────────────────────────────────────────────────────────────────────────────────

loss.backward() traverses the computation graph in reverse topological order,
dispatching to the backward handler for each op via an integer op_code jump table
(backpropagation.mojo:357). Key gradient flows:

  • dL/dctx = w_out^T @ dL/dlogits — gradient reaches ctx, then splits to both
    heads (division by 2 in the average).
  • dL/dattn_h = dL/dctx_h @ V_h^T — gradient reaches each head's attention.
  • dL/dQ_h, dL/dK_h propagate through the softmax Jacobian, driving the
    attention weights toward the correct anti-diagonal pairing.
  • The two heads receive independent gradients (different V_h), so even if one
    head gets stuck, the other head's gradient remains informative.

────────────────────────────────────────────────────────────────────────────────────
  OPTIMIZER
────────────────────────────────────────────────────────────────────────────────────

SGD with momentum (lr=LR, momentum=MOMENTUM, weight_decay=1e-4, clip_norm=5.0):

  v = momentum * v - lr * (grad + weight_decay * param)
  param += v

All parameters across 9 modules (tok_embed, pos_embed, 2× QKV sets, w_out)
are updated every step. The LR schedule decays to LR×0.2 after epoch 8.

────────────────────────────────────────────────────────────────────────────────────
  RELATED WORK
────────────────────────────────────────────────────────────────────────────────────

- ATTN/11 (https://github.com/dbrll/ATTN-11): PDP-11 assembly transformer,
  same reversal task.
- "Attention Is All You Need" (Vaswani et al. 2017): the original transformer.
- "The Annotated Transformer" (Harvard NLP): line-by-line implementation in
  PyTorch. This file is the Mojo/tenmo equivalent.

────────────────────────────────────────────────────────────────────────────────────
  Q&A: FEEDING SHORTER SEQUENCES
────────────────────────────────────────────────────────────────────────────────────

Q: Can the trained model (SEQ_LEN=8) reverse a 5-digit sequence at inference?
A: Yes — the model handles any length ≤ SEQ_LEN without code changes.

Why it works: tok_embed is element-wise (looks up each token ID independently),
pos_embed maps indices within its (8, 32) table, and all projections are
per-position Linear layers — all length-agnostic. Attention produces a smaller
matrix (1, 5, 5) instead of (1, 8, 8), same arithmetic.

What wouldn't work: a sequence longer than 8 — pos_embed would OOB on index 8+.
What you'd change: the inference input tensor shape and position IDs.

The catch: training on length 8 and inferring on length 5 means the model
learned to pair position t with 7-t, not 4-t. Not useful in practice — you'd
set SEQ_LEN=5 and retrain, or train on mixed lengths.

────────────────────────────────────────────────────────────────────────────────────
  Q&A: PADDING FOR VARIABLE-LENGTH SEQUENCES
────────────────────────────────────────────────────────────────────────────────────

Q: With a max context length M and variable-length sequences, would we need
   padding? How?
A: Yes. SEQ_LEN currently sets both model capacity and data length. To support
   variable-length input up to M:

   1. Add a <PAD> token outside the data vocabulary (VOCAB+1).
   2. Right-pad shorter sequences to length M.
   3. Mask padding positions in the attention: add -inf to padded positions
      before softmax so they contribute zero weight.
   4. Mask padding positions in the loss so they are not counted.

   The current code doesn't need this: every training example is exactly
   SEQ_LEN tokens, so there is never a padding token or variable-length batch.

────────────────────────────────────────────────────────────────────────────────────
  Q&A: POSITION EMBEDDINGS ALONE
────────────────────────────────────────────────────────────────────────────────────

Q: Would the model work with only position embeddings — no token embeddings?
A: No. Position embeddings encode "I am position t" but carry zero information
   about which digit lives there. Without token embeddings, every input sequence
   would look identical: [0, 1, 2, ..., 7] regardless of the actual digits.
   The value pathway is:

     ctx[t] = Σⱼ attn[t, j] · w_v(token_embed[j] + pos_embed[j])

   token_embed[j] is the only thing that carries "which digit is at position j."
   Without it, w_v(pos_embed[j]) is a function of position only, so ctx[t] is
   independent of the input — the model can only emit a constant output.

────────────────────────────────────────────────────────────────────────────────────
  Q&A: REPEATED DIGITS
────────────────────────────────────────────────────────────────────────────────────

Q: What happens with repeated digits, like [1, 1, 3, 3, 9, 9, 0, 0]?
A: It works perfectly — the model reverses [1, 1, 3, 3, 9, 9, 0, 0] →
   [0, 0, 9, 9, 3, 3, 1, 1] without error.

   Reason: the model learned a positional mapping, not a token-frequency one.
   Position t attends to position T-1-t via the attention mechanism, then copies
   whatever token it finds at that position. Since repeated tokens share the same
   embedding vector, the lookup is unaffected by duplicates. The attention weights
   are determined by position, not by what token occupies each position.

────────────────────────────────────────────────────────────────────────────────────
  Q&A: SCALING TO LONGER / WIDER VOCABULARIES
────────────────────────────────────────────────────────────────────────────────────

Q: What changes to reverse 20 numbers (VOCAB=20, SEQ_LEN=20)?
A: Two mandatory changes, two likely hyperparameter tweaks.

   Mandatory:
     VOCAB = 10  →  VOCAB = 20
     SEQ_LEN = 8 →  SEQ_LEN = 20

   These propagate: tok_embed grows to (20, 32), pos_embed to (20, 32), and
   w_out to (32, 20). Total parameters jump from ~8K to ~12K.

   Likely hyperparameter changes (the task is harder — P(20,20) ≈ 2.4e18 vs
   P(10,8) ≈ 1.8e6, and the 20×20 attention matrix has 6.25× more pairs):
     NUM_TRAIN = 2000 → 10000–20000
     NUM_EPOCHS = 10 → 50–100

   Main risk: at D_MODEL=32, each head's 32-dim Q/K space must disambiguate 20
   positions. If the model struggles, bump D_MODEL to 48 or add a third head.
"""

from tenmo.tensor import Tensor
from tenmo.optim import SGD
from tenmo.net import Linear
from tenmo.embedding import Embedding
from tenmo.crossentropy import CrossEntropyLoss
from tenmo.softmax import Softmax
from tenmo.intarray import IntArray
from tenmo.dataloader import TensorDataset
from tenmo.mnemonics import DEFAULT_INDEX_DTYPE as idx_dtype
from tenmo.accuracy import Accuracy
from std.random import shuffle
from std.time import perf_counter_ns
from std.math import sqrt


comptime dtype = DType.float32

comptime VOCAB = 10
comptime SEQ_LEN = 8
comptime D_MODEL = 32
comptime BATCH_SIZE = 32
comptime NUM_TRAIN = 2000
comptime NUM_TEST = 500
comptime NUM_EPOCHS = 10
comptime LR = 0.01
comptime MOMENTUM = 0.9


def generate_reversal_data(
    num_samples: Int,
    seq_len: Int,
    vocab: Int,
) -> Tuple[Tensor[idx_dtype], Tensor[idx_dtype]]:
    var x = Tensor[idx_dtype](num_samples, seq_len)
    var y = Tensor[idx_dtype](num_samples, seq_len)

    var base = [i for i in range(vocab)]

    for i in range(num_samples):
        var perm = base.copy()
        shuffle(perm)
        for j in range(seq_len):
            x[i, j] = Scalar[idx_dtype](perm[j])
            y[i, j] = Scalar[idx_dtype](perm[seq_len - 1 - j])

    return (x^, y^)

def make_pos_indices(batch_size: Int, seq_len: Int) -> Tensor[idx_dtype]:
    return Tensor[idx_dtype].stack[False](
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


def make_reversal_test(vocab: Int, seq_len: Int) -> Tensor[idx_dtype]:
    var T = Tensor[idx_dtype](1, seq_len)
    var p = 0
    var v = 0
    while p < seq_len:
        var r = min(2, seq_len - p)
        for _ in range(r):
            T[0, p] = Scalar[idx_dtype](v % vocab)
            p += 1
        v += 1
    return T^


def main() raises:
    print("=" * 58)
    print("Sequence Reversal Transformer - Tenmo")
    print("=" * 58)
    print("  d_model =", D_MODEL, "| vocab =", VOCAB, "| seq_len =", SEQ_LEN)
    print("  train =", NUM_TRAIN, "| test =", NUM_TEST)
    print("  batch_size =", BATCH_SIZE, "| epochs =", NUM_EPOCHS)
    print("  lr =", LR, "| momentum =", MOMENTUM)
    print()

    # ── Data ──────────────────────────────────────────────────────────
    print("Generating", NUM_TRAIN, "training +", NUM_TEST, "test samples...")
    var train_x, train_y = generate_reversal_data(NUM_TRAIN, SEQ_LEN, VOCAB)
    var test_x, test_y = generate_reversal_data(NUM_TEST, SEQ_LEN, VOCAB)

    var demo_count = min(5, NUM_TEST)
    var demo_x = Tensor[idx_dtype](demo_count, SEQ_LEN)
    var demo_y = Tensor[idx_dtype](demo_count, SEQ_LEN)
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
    # Embeddings: default "normal" init (N(0,1)) produces attention scores with
    # std ≈ 2.0, causing softmax saturation and occasional gradient vanishing.
    # "uniform" init (U[-1,1]) gives scores with std ≈ 0.67 — a well-conditioned
    # softmax with strong position-specific gradients. We add a second independent
    # attention head to eliminate the ~6% failure rate from single-head gradient noise.
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
        params^,
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=1e-4,
        clip_norm=Float32(5.0),
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
        # ── LR schedule: high LR for exploration, decay for fine-tuning ──
        if epoch < 8:
            optimizer.set_lr(LR)
        else:
            optimizer.set_lr(LR * 0.2)

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

        # ── reset would shuffle dataset ──
        train_loader.reset()
        var epoch_loss = Float64(0.0)
        # We could use a for loop - that would copy - we want to avoid copy - hence __has_next__ ──
        while train_loader.__has_next__():
            ref batch = train_loader.__next__()
            var B = batch.features.shape()[0]

            var tok_x = tok_embed(batch.features)
            var pos_indices = make_pos_indices(B, SEQ_LEN)
            var x = tok_x + pos_embed(pos_indices)

            var Q1 = w_q1(x)
            var K1 = w_k1(x)
            var K1_t = K1.transpose(0, 2, 1)
            var scores1 = Q1.matmul(K1_t) / sqrt(Float32(D_MODEL))
            var attn1 = Softmax[dtype].forward[track_grad=True](
                scores1, axes=IntArray(-1)
            )
            var ctx1 = attn1.matmul(w_v1(x))

            var Q2 = w_q2(x)
            var K2 = w_k2(x)
            var K2_t = K2.transpose(0, 2, 1)
            var scores2 = Q2.matmul(K2_t) / sqrt(Float32(D_MODEL))
            var attn2 = Softmax[dtype].forward[track_grad=True](
                scores2, axes=IntArray(-1)
            )
            var ctx2 = attn2.matmul(w_v2(x))

            var ctx = (ctx1 + ctx2) / Float32(2.0)
            var combined = ctx
            var logits = w_out(combined)

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
            var pos_indices = make_pos_indices(B, SEQ_LEN)
            var x = tok_x + pos_embed(pos_indices)

            var Q1 = w_q1(x)
            var K1 = w_k1(x)
            var K1_t = K1.transpose(0, 2, 1)
            var scores1 = Q1.matmul(K1_t) / sqrt(Float32(D_MODEL))
            var attn1 = Softmax[dtype].forward[track_grad=False](
                scores1, axes=IntArray(-1)
            )
            var ctx1 = attn1.matmul(w_v1(x))

            var Q2 = w_q2(x)
            var K2 = w_k2(x)
            var K2_t = K2.transpose(0, 2, 1)
            var scores2 = Q2.matmul(K2_t) / sqrt(Float32(D_MODEL))
            var attn2 = Softmax[dtype].forward[track_grad=False](
                scores2, axes=IntArray(-1)
            )
            var ctx2 = attn2.matmul(w_v2(x))

            var ctx = (ctx1 + ctx2) / Float32(2.0)
            var combined = ctx
            var logits = w_out(combined)

            var ta, sa = compute_accuracy(logits, batch.labels)
            token_acc += ta
            seq_acc += sa
            n_batches += 1


        token_acc /= Float64(n_batches)
        seq_acc /= Float64(n_batches)
        if seq_acc > best_seq_acc:
            best_seq_acc = seq_acc

        var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
        print(
            "  Epoch", epoch + 1, "/", NUM_EPOCHS,
            "| loss:", epoch_loss / Float64(NUM_TRAIN),
            "| token:", token_acc * 100, "%",
            "| seq:", seq_acc * 100, "%",
            "|", elapsed, "s",
        )

    # ── Final results ─────────────────────────────────────────────────
    print()
    print("=" * 58)
    print("Best sequence accuracy:", best_seq_acc * 100, "%")
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
        var inp = Tensor[idx_dtype](1, SEQ_LEN)
        for t in range(SEQ_LEN):
            inp[0, t] = demo_x[idx, t]

        var tok_x = tok_embed(inp)
        var pos_indices = make_pos_indices(1, SEQ_LEN)
        var x = tok_x + pos_embed(pos_indices)

        var Q1 = w_q1(x)
        var K1 = w_k1(x)
        var K1_t = K1.transpose(0, 2, 1)
        var scores1 = Q1.matmul(K1_t) / sqrt(Float32(D_MODEL))
        var attn1 = Softmax[dtype].forward[track_grad=False](
            scores1, axes=IntArray(-1)
        )
        var ctx1 = attn1.matmul(w_v1(x))

        var Q2 = w_q2(x)
        var K2 = w_k2(x)
        var K2_t = K2.transpose(0, 2, 1)
        var scores2 = Q2.matmul(K2_t) / sqrt(Float32(D_MODEL))
        var attn2 = Softmax[dtype].forward[track_grad=False](
            scores2, axes=IntArray(-1)
        )
        var ctx2 = attn2.matmul(w_v2(x))

        var ctx = (ctx1 + ctx2) / Float32(2.0)
        var combined = ctx
        var logits = w_out(combined)
        var preds = logits.argmax[idx_dtype](axis=-1)

        print("  Example", idx + 1)
        for t in range(SEQ_LEN):
            print(
                "    Pos", t,
                "| in:", Int(demo_x[idx, t]),
                "-> pred:", Int(preds[0, t]),
                "(target:", Int(demo_y[idx, t]), ")",
            )
        print()

    # ── Repeated-digit test ──
    print("Repeated-digit inference test:")
    var dup_input = make_reversal_test(VOCAB, SEQ_LEN)
    var dup_tok = tok_embed(dup_input)
    var dup_pos = make_pos_indices(1, SEQ_LEN)
    var dup_x = dup_tok + pos_embed(dup_pos)
    var dup_Q1 = w_q1(dup_x)
    var dup_K1 = w_k1(dup_x)
    var dup_K1_t = dup_K1.transpose(0, 2, 1)
    var dup_scores1 = dup_Q1.matmul(dup_K1_t) / sqrt(Float32(D_MODEL))
    var dup_attn1 = Softmax[dtype].forward[track_grad=False](
        dup_scores1, axes=IntArray(-1)
    )
    var dup_ctx1 = dup_attn1.matmul(w_v1(dup_x))

    var dup_Q2 = w_q2(dup_x)
    var dup_K2 = w_k2(dup_x)
    var dup_K2_t = dup_K2.transpose(0, 2, 1)
    var dup_scores2 = dup_Q2.matmul(dup_K2_t) / sqrt(Float32(D_MODEL))
    var dup_attn2 = Softmax[dtype].forward[track_grad=False](
        dup_scores2, axes=IntArray(-1)
    )
    var dup_ctx2 = dup_attn2.matmul(w_v2(dup_x))

    var dup_ctx = (dup_ctx1 + dup_ctx2) / Float32(2.0)
    var dup_combined = dup_ctx
    var dup_logits = w_out(dup_combined)
    var dup_preds = dup_logits.argmax[idx_dtype](axis=-1)
    print("  Input     : [", end="")
    for t in range(SEQ_LEN):
        if t > 0: print(", ", end="")
        print(Int(dup_input[0, t]), end="")
    print("]")
    print("  Expected  : [", end="")
    for t in range(SEQ_LEN):
        if t > 0: print(", ", end="")
        print(Int(dup_input[0, SEQ_LEN - 1 - t]), end="")
    print("]")
    print("  Predicted : [", end="")
    for t in range(SEQ_LEN):
        if t > 0: print(", ", end="")
        print(Int(dup_preds[0, t]), end="")
    print("]")
    var dup_ok = True
    for t in range(SEQ_LEN):
        if Int(dup_preds[0, t]) != Int(dup_input[0, SEQ_LEN - 1 - t]):
            dup_ok = False
    print("  Correct:", dup_ok)
    print()

    print("Done.")
