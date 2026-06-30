from tenmo import Tensor, Embedding, Linear, CrossEntropyLoss
from tokenizer import SimpleTokenizer
from tenmo.mnemonics import DEFAULT_INDEX_DTYPE as index_dtype
from std.pathlib import Path
from tenmo.nlp import RandomSlidingWindowDataset as WindowingDataset
from tenmo.common_utils import s, i, panic
from tenmo.dataloader import DataLoader
from tenmo.optim import SGD


comptime dtype: DType = DType.float32


# ─────────────────────────────────────────────────────────────────────────────
#  NGramModel
#
#  Predicts the next character given the previous (n-1) characters.
#
#  Architecture:
#    indices [B, T]
#    → Embedding lookup → [B, T, emb_dim]
#    → for each position t, take context window of (n-1) tokens ending at t-1
#      → context embeddings [B, T, (n-1)*emb_dim]
#    → Linear projection  → [B, T, vocab_size]  (logits)
#
#  Special cases:
#    n=2 (bigram):  context = 1 token  → embed dim = emb_dim
#    n=3 (trigram): context = 2 tokens → embed dim = 2*emb_dim
#    n=N:           context = N-1 tokens → embed dim = (N-1)*emb_dim
#
#  The sequence must be padded at the start so every position has a full
#  context window. We pad with token 0 (typically newline/start token).
#
#  DataLoader note:
#    seq_length should be >= n so each window contains enough context.
#    The dataloader's sliding window already provides this — each row of
#    features has seq_length tokens, and each position predicts the next.
# ─────────────────────────────────────────────────────────────────────────────
@fieldwise_init
struct NGramModel[dtype: DType](ImplicitlyCopyable & Movable):
    var n: Int  # n-gram order (2=bigram, 3=trigram, ...)
    var vocab_size: Int
    var emb_dim: Int
    var embedding: Embedding[Self.dtype]
    var projection: Linear[Self.dtype]  # [(n-1)*emb_dim] → [vocab_size]

    def __init__(
        out self,
        vocab_size: Int,
        n: Int = 2,
        emb_dim: Int = 32,
    ):
        """
        Args:
            vocab_size: Number of unique tokens (e.g. 65 for tiny Shakespeare).
            n: N-gram order. n=2 is bigram, n=3 is trigram, etc.
               Must be >= 2.
            emb_dim: Embedding dimension per token. For n=2 this is also the
                     projection input size. For n>2, projection input = (n-1)*emb_dim.
        """
        if n < 2:
            panic("NGramModel: n must be >= 2 (n=2 is bigram)")

        self.n = n
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.embedding = Embedding[Self.dtype](
            vocab_size,
            emb_dim,
            init_method="kaiming",
            init_seed=42,
        )
        # Input to projection = concatenation of (n-1) context embeddings
        var context_size = (n - 1) * emb_dim
        self.projection = Linear[Self.dtype](
            context_size,
            vocab_size,
            bias=True,
            init_seed=42,
        )

    def parameters(
        ref self,
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = self.embedding.parameters()
        for p in self.projection.parameters():
            params.append(p)
        return params^

    def train(mut self):
        self.embedding.train()
        self.projection.train()

    def eval(mut self):
        self.embedding.eval()
        self.projection.eval()

    def __call__(
        mut self,
        indices: Tensor[index_dtype],  # [B, T]
        targets: Optional[Tensor[index_dtype]] = None,
    ) -> Tuple[
        Tensor[Self.dtype], Optional[Tensor[Self.dtype]]
    ] where Self.dtype.is_floating_point():
        ref shape = indices.shape()
        var B = shape[0]
        var T = shape[1]
        var context_len = self.n - 1  # number of preceding tokens to use

        # ── Step 1: embed all tokens ──────────────────────────────────────────
        # embs: [B, T, emb_dim]
        var embs = self.embedding(indices)

        # ── Step 2: build context windows ────────────────────────────────────
        # For each position t in [0, T), gather the (n-1) preceding embeddings.
        # Positions before the start of the sequence are padded with zeros
        # (equivalent to a learned "start of sequence" — token 0's embedding
        # would also work but zero padding is simpler and common for n-grams).
        #
        # context: [B, T, (n-1)*emb_dim]
        # For t=0:   pad (n-1) slots with zeros
        # For t=1:   pad (n-2) slots, use embs[:, 0, :]
        # For t=k:   use embs[:, max(0,k-context_len):k, :] + left-zero-pad if needed
        var context_dim = context_len * self.emb_dim
        var context = Tensor[Self.dtype].zeros(B, T, context_dim)

        for t in range(T):
            # How many real (non-padded) context tokens are available at position t?
            var available = min(t, context_len)
            # How many zero-padding slots at the left?
            var padding = context_len - available

            if available > 0:
                # Copy available preceding embeddings into the right slots
                # of the context vector for position t.
                # Slots [padding*emb_dim .. context_dim) get real embeddings.
                # Slots [0 .. padding*emb_dim) stay zero (already zeroed).
                for k in range(available):
                    # k-th real token is at position t - available + k
                    var src_t = t - available + k
                    # Destination slot in context vector
                    var dst_slot = padding + k
                    # context[b, t, dst_slot*emb_dim : (dst_slot+1)*emb_dim]
                    #   = embs[b, src_t, :]
                    var dst_start = dst_slot * self.emb_dim
                    var dst_end = dst_start + self.emb_dim
                    # Slice assignment: all batches at once
                    context.fill(
                        embs[s(), i(src_t), s()],
                        s(),
                        i(t),
                        s(dst_start, dst_end),
                    )

        # ── Step 3: project context to vocab logits ───────────────────────────
        # context: [B, T, (n-1)*emb_dim]
        # project each (B*T) context vector independently
        var context_flat = context.reshape(
            B * T, context_dim
        )  # [B*T, context_dim]
        var logits_flat = self.projection(context_flat)  # [B*T, vocab_size]
        var logits = logits_flat.reshape(
            B, T, self.vocab_size
        )  # [B, T, vocab_size]

        # ── Step 4: compute loss if targets provided ──────────────────────────
        var loss: Optional[Tensor[Self.dtype]] = None
        if targets:
            var targets_unwrapped = targets.value()
            var targets_flat = targets_unwrapped.reshape(B * T)  # [B*T]
            loss = CrossEntropyLoss[Self.dtype]()(logits_flat, targets_flat)

        return logits, loss

    def generate(
        mut self,
        indices: Tensor[index_dtype],
        max_new_tokens: Int,
    ) raises -> Tensor[index_dtype] where Self.dtype.is_floating_point():
        """
        Autoregressively generate max_new_tokens tokens.

        The full sequence is passed each step so the context window at the
        last position always has the correct (n-1) preceding tokens.
        For large max_new_tokens this grows the sequence — acceptable for
        generation, not for training.
        """
        var result = indices
        for _ in range(max_new_tokens):
            var logits, _loss = self(result)
            # Take last position's logits: [B, T, C] → [B, C]
            var logits_last = logits[s(), i(-1), s()]
            var probs = logits_last.softmax[track_grad=False](axes=[-1])
            var next_index = Tensor[Self.dtype].multinomial(
                probs, num_samples=1
            )  # [B, 1]
            result = Tensor[index_dtype].concat[track_grad=False](
                [result, next_index], axis=1
            )
        return result


# ─────────────────────────────────────────────────────────────────────────────
#  estimate_loss
#
#  Uses loader.sample() — draws random batches, no iterator state consumed.
#  Sets model to eval mode, restores train mode after.
# ─────────────────────────────────────────────────────────────────────────────
def estimate_loss[
    dtype: DType
](
    mut model: NGramModel[dtype],
    train_loader: DataLoader[WindowingDataset[index_dtype], _],
    val_loader: DataLoader[WindowingDataset[index_dtype], _],
    eval_iters: Int,
) raises -> Dict[String, Scalar[dtype]] where dtype.is_floating_point():
    var out = Dict[String, Scalar[dtype]]()
    model.eval()

    for split in ["train", "val"]:
        var total_loss: Scalar[dtype] = 0

        for _ in range(eval_iters):
            var X, Y = (
                train_loader.sample() if split
                == "train" else val_loader.sample()
            )
            X = X.unsqueeze(0)  # Add batch
            Y = Y.unsqueeze(0)
            var _logits, loss = model(X, Optional(Y))
            if loss:
                total_loss += loss.value().item()

        out[String(split)] = total_loss / Scalar[dtype](eval_iters)

    model.train()
    return out^


# ─────────────────────────────────────────────────────────────────────────────
#  generate_text
# ─────────────────────────────────────────────────────────────────────────────
def generate_text[
    dtype: DType
](
    mut model: NGramModel[dtype],
    tokenizer: SimpleTokenizer,
    max_new_tokens: Int = 200,
    start_token: Int = 0,
) raises -> String where dtype.is_floating_point():
    var ctx = Tensor[index_dtype].zeros(1, 1)
    if start_token != 0:
        ctx[0, 0] = Scalar[index_dtype](start_token)
    var generated = model.generate(ctx, max_new_tokens=max_new_tokens)
    var row = generated[i(0), s()]
    var tokens = row.tolist()
    return tokenizer.decode_from(tokens)


def main() raises:
    # ── Data preparation ──────────────────────────────────────────────────────
    var text = Path("input.txt").read_text()
    var tokenizer = SimpleTokenizer(text)
    var vocab_size = len(tokenizer)

    print("Vocab size: ", vocab_size)
    print("Text length:", text.count_codepoints(), "characters")

    var data = tokenizer.encode_as[dtype=index_dtype](text)
    var split_idx = Int(0.9 * Float64(len(data)))
    var train_data = Tensor[index_dtype].from_list(data[:split_idx])
    var val_data = Tensor[index_dtype].from_list(data[split_idx:])

    print("Train tokens:", split_idx)
    print("Val tokens:  ", len(data) - split_idx)

    # ── Hyperparameters ───────────────────────────────────────────────────────
    var n = 8  # 2=bigram, 3=trigram, 4=4-gram, ...
    var emb_dim = 32  # embedding dimension per token
    var seq_length = 16  # must be >= n so each window has enough context
    var batch_size = 64
    var num_epochs = 30
    var eval_interval = 500
    var eval_iters = 50
    var lr: Scalar[dtype] = 1e-2

    print(
        "\nModel: ", n, "-gram | emb_dim:", emb_dim, "| seq_length:", seq_length
    )

    # ── Dataloaders ───────────────────────────────────────────────────────────
    var train_ds = WindowingDataset[index_dtype](train_data, seq_length)
    var val_ds = WindowingDataset[index_dtype](val_data, seq_length)

    var train_loader = train_ds.into_loader(batch_size, shuffle=True)
    var val_loader = val_ds.into_loader(batch_size, shuffle=False)

    # ── Model + optimiser ─────────────────────────────────────────────────────
    var model = NGramModel[dtype](vocab_size, n=n, emb_dim=emb_dim)
    var optimizer = SGD(model.parameters(), lr=lr)

    var context_size = (n - 1) * emb_dim
    print(
        "Parameters:",
        vocab_size * emb_dim,  # embedding table
        "+",
        context_size * vocab_size,  # projection weights
        "+",
        vocab_size,  # projection bias
        "=",
        vocab_size * emb_dim + context_size * vocab_size + vocab_size,
    )

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\n--- Baseline (before training) ---")
    var pre = estimate_loss(model, train_loader, val_loader, eval_iters)
    print("train loss", pre["train"], "| val loss", pre["val"])
    # Expected: ~log(65) ≈ 4.17 for random weights regardless of n
    print("Generated (random weights):")
    print(generate_text(model, tokenizer, max_new_tokens=100))

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\n--- Training (", n, "-gram) ---")
    var global_step = 0

    for epoch in range(num_epochs):
        var epoch_loss: Scalar[dtype] = 0
        var epoch_batches: Int = 0

        for batch in train_loader:
            # Periodic eval
            if global_step % eval_interval == 0:
                var losses = estimate_loss(
                    model, train_loader, val_loader, eval_iters
                )
                print(
                    "Epoch",
                    epoch + 1,
                    "| step",
                    global_step,
                    "| train loss",
                    losses["train"],
                    "| val loss",
                    losses["val"],
                )

                print(generate_text(model, tokenizer, max_new_tokens=300))

            # Forward
            var X = batch.features  # [B, T]
            var Y = batch.labels  # [B, T]
            var _logits, loss = model(X, Optional(Y))

            # Backward + update
            optimizer.zero_grad()
            if loss:
                var l = loss.value()
                l.backward()
                optimizer.step()
                epoch_loss += l.item()

            epoch_batches += 1
            global_step += 1

        var avg = epoch_loss / Scalar[dtype](max(epoch_batches, 1))
        print(
            "Epoch",
            epoch + 1,
            "complete |",
            epoch_batches,
            "batches |",
            "avg train loss",
            avg,
        )

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n--- After training ---")
    var post = estimate_loss(model, train_loader, val_loader, eval_iters)
    print("train loss", post["train"], "| val loss", post["val"])
    print("\nGenerated (trained", n, "-gram):")
    print(generate_text(model, tokenizer, max_new_tokens=500))
    print("\nDone.")
