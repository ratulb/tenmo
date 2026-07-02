from tenmo import Tensor, Embedding, Linear, CrossEntropyLoss
from tokenizer import SimpleTokenizer
from tenmo.mnemonics import DEFAULT_INDEX_DTYPE as index_dtype
from std.pathlib import Path
from tenmo.nlp import RandomSlidingWindowDataset as WindowingDataset
from tenmo.common_utils import s, i, panic
from tenmo.dataloader import DataLoader
from tenmo.optim import SGD

# ─────────────────────────────────────────────────────────────────────────────
#  NeuralProbabilisticLM (Character-Level)
#
#  Predicts next character using a neural network with a hidden layer.
#  Based on Bengio et al. (2003) "A Neural Probabilistic Language Model"
#  but adapted for character-level modeling.
#
#  Architecture:
#    indices [B, T]
#    → Embedding lookup → [B, T, emb_dim]
#    → Concatenate context of (n-1) embeddings → [B, T, (n-1)*emb_dim]
#    → Hidden layer with tanh activation → [B, T, hidden_dim]
#    → Output projection → [B, T, vocab_size]  (logits)
#
#  Key differences from simple N-gram:
#    1. Non-linear hidden layer (captures complex interactions between chars)
#    2. Optional direct input-output connections (skip connections)
#    3. Parameter tying (share embedding matrix with output)
# ─────────────────────────────────────────────────────────────────────────────
@fieldwise_init
struct NeuralProbabilisticLM[dtype: DType](ImplicitlyCopyable & Movable):
    var n: Int  # n-gram order
    var vocab_size: Int
    var emb_dim: Int
    var hidden_dim: Int
    var embedding: Embedding[Self.dtype]
    var hidden_layer: Linear[Self.dtype]  # [(n-1)*emb_dim] → [hidden_dim]
    var output_layer: Linear[Self.dtype]  # [hidden_dim] → [vocab_size]
    # Optional direct connection (skip connection from context to output)
    var direct_connection: Optional[Linear[Self.dtype]]  # [(n-1)*emb_dim] → [vocab_size]
    var use_direct: Bool
    var tie_weights: Bool

    def __init__(
        out self,
        vocab_size: Int,
        n: Int = 3,
        emb_dim: Int = 32,
        hidden_dim: Int = 64,
        use_direct: Bool = True,
        tie_weights: Bool = False,
    ):
        """
        Args:
            vocab_size: Number of unique characters.
            n: N-gram order. n=2 is bigram, n=3 is trigram.
            emb_dim: Embedding dimension per character.
            hidden_dim: Size of the hidden layer.
            use_direct: Whether to have direct input→output connections.
            tie_weights: Whether to tie input embeddings to output weights.
        """
        if n < 2:
            panic("NeuralProbabilisticLM: n must be >= 2")

        self.n = n
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.use_direct = use_direct
        self.tie_weights = tie_weights

        # Input embedding layer
        self.embedding = Embedding[Self.dtype](
            vocab_size,
            emb_dim,
            init_method="kaiming",
            init_seed=42,
        )

        var context_size = (n - 1) * emb_dim

        # Hidden layer with tanh activation
        self.hidden_layer = Linear[Self.dtype](
            context_size,
            hidden_dim,
            bias=True,
            init_seed=42,
        )

        # Output layer (hidden → vocab)
        if tie_weights:
            # If tying weights, use embedding matrix transposed
            # In practice, this means output weights = embedding.weight.T
            # and we don't learn separate output weights
            self.output_layer = Linear[Self.dtype](
                hidden_dim,
                vocab_size,
                bias=True,
                init_seed=42,
            )
            # We'll need to handle weight tying in the forward pass
        else:
            self.output_layer = Linear[Self.dtype](
                hidden_dim,
                vocab_size,
                bias=True,
                init_seed=42,
            )

        # Optional direct connection
        if use_direct:
            self.direct_connection = Linear[Self.dtype](
                context_size,
                vocab_size,
                bias=True,
                init_seed=42,
            )
        else:
            self.direct_connection = None

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
        var context_len = self.n - 1

        # ── Step 1: embed all tokens ──────────────────────────────────────────
        var embs = self.embedding(indices)  # [B, T, emb_dim]

        # ── Step 2: build context windows ────────────────────────────────────
        var context_dim = context_len * self.emb_dim
        var context = Tensor[Self.dtype].zeros(B, T, context_dim)

        for t in range(T):
            var available = min(t, context_len)
            var padding = context_len - available

            if available > 0:
                for k in range(available):
                    var src_t = t - available + k
                    var dst_slot = padding + k
                    var dst_start = dst_slot * self.emb_dim
                    var dst_end = dst_start + self.emb_dim
                    context.fill(
                        embs[s(), i(src_t), s()],
                        s(),
                        i(t),
                        s(dst_start, dst_end),
                    )

        # ── Step 3: reshape for linear layers ────────────────────────────────
        var context_flat = context.reshape(B * T, context_dim)  # [B*T, context_dim]

        # ── Step 4: hidden layer with tanh activation ────────────────────────
        var hidden = self.hidden_layer(context_flat)  # [B*T, hidden_dim]
        var hidden_activated = hidden.tanh()  # Non-linearity!

        # ── Step 5: output layer ─────────────────────────────────────────────
        var logits_flat = self.output_layer(hidden_activated)  # [B*T, vocab_size]

        # ── Step 6: optional direct connection (skip connection) ────────────
        if self.use_direct:
            var direct_logits = self.direct_connection.value()(context_flat)
            logits_flat = logits_flat + direct_logits  # Add them together

        # ── Step 7: optionally tie weights ────────────────────────────────────
        if self.tie_weights:
            # In practice, this means using self.embedding.weight.T
            # for the output projection from hidden layer
            # This is a simplified version; actual implementation is more complex
            pass

        # Reshape back to [B, T, vocab_size]
        var logits = logits_flat.reshape(B, T, self.vocab_size)

        # ── Step 8: compute loss ─────────────────────────────────────────────
        var loss: Optional[Tensor[Self.dtype]] = None
        if targets:
            var targets_unwrapped = targets.value()
            var targets_flat = targets_unwrapped.reshape(B * T)
            loss = CrossEntropyLoss[Self.dtype]()(logits_flat, targets_flat)

        return logits, loss

def main() raises:
    pass
