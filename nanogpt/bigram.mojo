from tenmo import Tensor, Embedding, CrossEntropyLoss
from tokenizer import SimpleTokenizer
from tenmo.mnemonics import DEFAULT_INDEX_DTYPE as index_dtype
from std.pathlib import Path
from tenmo.nlp import RandomSlidingWindowDataset as WindowingDataset
from tenmo.common_utils import s, i
from tenmo.dataloader import DataLoader
from tenmo.optim import SGD


comptime dtype: DType = DType.float32


# ─────────────────────────────────────────────────────────────────────────────
#  BigramModel
#
#  The simplest possible language model:
#    embedding_table[i] = logit distribution over next character given char i
#
#  Forward:
#    indices [B, T] → lookup → logits [B, T, C]
#    if targets given: reshape to [B*T, C] / [B*T], compute CE loss
#    always return original [B, T, C] logits + Optional loss
#
#  Generate:
#    autoregressively append tokens by sampling from last-position logits
# ─────────────────────────────────────────────────────────────────────────────
@fieldwise_init
struct BigramModel[dtype: DType](ImplicitlyCopyable & Movable):
    var lookup_table: Embedding[Self.dtype]

    def __init__(out self, vocab_size: Int):
        self.lookup_table = Embedding[Self.dtype](
            vocab_size,
            vocab_size,
            init_method="kaiming",
            init_seed=42,
        )

    def parameters(
        ref self,
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        return self.lookup_table.parameters()

    def train(mut self):
        self.lookup_table.train()

    def eval(mut self):
        self.lookup_table.eval()

    def __call__(
        mut self,
        indices: Tensor[index_dtype],
        targets: Optional[Tensor[index_dtype]] = None,
    ) -> Tuple[
        Tensor[Self.dtype], Optional[Tensor[Self.dtype]]
    ] where Self.dtype.is_floating_point():
        # logits: [B, T, C]  where C = vocab_size
        var logits = self.lookup_table(indices)

        var loss: Optional[Tensor[Self.dtype]] = None
        if targets:
            ref shape = logits.shape()
            var B = shape[0]
            var T = shape[1]
            var C = shape[2]
            # Cross entropy expects [N, C] logits and [N] targets
            var logits_flat = logits.reshape(B * T, C)
            var targets_unwrapped = targets.value()
            var targets_flat = targets_unwrapped.reshape(B * T)
            loss = CrossEntropyLoss[Self.dtype]()(logits_flat, targets_flat)

        # Always return [B, T, C] logits — never the reshaped version.
        # Callers (generate, estimate_loss) always want the original shape.
        return logits, loss

    def generate(
        mut self,
        indices: Tensor[index_dtype],
        max_new_tokens: Int,
    ) raises -> Tensor[index_dtype] where Self.dtype.is_floating_point():
        """Autoregressively append max_new_tokens tokens to indices [B, T]."""
        var result = indices
        for _ in range(max_new_tokens):
            var logits, _loss = self(result)
            # Take only the last time step: [B, T, C] → [B, C]
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
#  Evaluate mean loss over eval_iters random batches on each split.
#  Uses loader.sample() — draws one random batch per call, no iterator state.
#  Sets model to eval mode, restores train mode after.
# ─────────────────────────────────────────────────────────────────────────────
def estimate_loss[
    dtype: DType
](
    mut model: BigramModel[dtype],
    train_loader: DataLoader[WindowingDataset[index_dtype], _],
    val_loader: DataLoader[WindowingDataset[index_dtype], _],
    eval_iters: Int,
) raises -> Dict[String, Scalar[dtype]] where dtype.is_floating_point():
    var out = Dict[String, Scalar[dtype]]()
    model.eval()

    for split in ["train", "val"]:
        var total_loss: Scalar[dtype] = 0
        # ref loader = train_loader if split == "train" else val_loader

        for _ in range(eval_iters):
            # .sample() draws a random record without advancing iterator state
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
#  generate_text — decode a generated sequence back to a string
# ─────────────────────────────────────────────────────────────────────────────
def generate_text[
    dtype: DType
](
    mut model: BigramModel[dtype],
    tokenizer: SimpleTokenizer,
    max_new_tokens: Int = 100,
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
    _ = """tokenizer = SimpleTokenizer.from_url(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/inp
    )"""
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

    # ── Dataloaders ───────────────────────────────────────────────────────────
    var seq_length = 8
    var batch_size = 32

    var train_ds = WindowingDataset[index_dtype](train_data, seq_length)
    var val_ds = WindowingDataset[index_dtype](val_data, seq_length)

    # shuffle=True for training — different order each epoch
    var train_loader = train_ds.into_loader(batch_size, shuffle=True)
    var val_loader = val_ds.into_loader(batch_size, shuffle=False)

    # ── Model + optimiser ─────────────────────────────────────────────────────
    var model = BigramModel[dtype](vocab_size)
    var optimizer = SGD(model.parameters(), lr=Scalar[dtype](1e-2))

    # ── Hyperparameters ───────────────────────────────────────────────────────
    var num_epochs = 50  # outer loop — full passes over train set
    var eval_interval = 500  # report every N steps within an epoch
    var eval_iters = 50  # batches averaged for loss estimate

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\n--- Baseline (before training) ---")
    var pre = estimate_loss(model, train_loader, val_loader, eval_iters)
    print("train loss", pre["train"], "| val loss", pre["val"])
    print("Expected ~4.17 = log(65) for random weights")
    print(generate_text(model, tokenizer, max_new_tokens=100))

    # ── Training loop — epoch-based ───────────────────────────────────────────
    #
    # Outer loop: epochs — each full pass through train_loader.
    # Inner loop: batches within one epoch.
    # eval_interval: report train/val loss every N steps (global step count).
    #
    # Using epochs rather than a raw step budget:
    #   - One epoch = one full pass over all (seq_length)-length windows
    #   - Multiple epochs let the model revisit data with different shuffle order
    #   - Global step counter used for eval_interval reporting across epochs
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Training ---")
    var global_step = 0

    for epoch in range(num_epochs):
        var epoch_loss: Scalar[dtype] = 0
        var epoch_batches = 0

        for batch in train_loader:
            # ── Periodic evaluation ───────────────────────────────────────────
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
                print("\n Generated\n")
                print(generate_text(model, tokenizer, max_new_tokens=100))

            # ── Forward ───────────────────────────────────────────────────────
            var X = batch.features  # [B, T]
            var Y = batch.labels  # [B, T]
            var _logits, loss = model(X, Optional(Y))

            # ── Backward + update ─────────────────────────────────────────────
            optimizer.zero_grad()
            if loss:
                var l = loss.value()
                l.backward()
                optimizer.step()
                epoch_loss += l.item()

            epoch_batches += 1
            global_step += 1


        # ── End of epoch summary ──────────────────────────────────────────────
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
    print("\nGenerated (trained):")
    print(generate_text(model, tokenizer, max_new_tokens=500))
    print("\nDone.")
