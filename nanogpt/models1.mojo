from tenmo import Tensor, Embedding, CrossEntropyLoss, SGD
from tokenizer import SimpleTokenizer
from tenmo.mnemonics import DEFAULT_INDEX_DTYPE as index_dtype
from std.pathlib import Path
from tenmo.nlp import RandomSlidingWindowDataset as WindowingDataset
from tenmo.common_utils import s, i
from tenmo.dataloader import DataLoader


comptime dtype: DType = DType.float32


@fieldwise_init
struct BigramLanguageModel[dtype: DType](ImplicitlyCopyable & Movable):
    var lookup_table: Embedding[Self.dtype]
    var cross_entropy: CrossEntropyLoss[Self.dtype]

    def __init__(out self, vocab_size: Int):
        self.lookup_table = Embedding[Self.dtype](
            vocab_size,
            vocab_size,
            init_method="kaiming",
            init_seed=42,
        )
        self.cross_entropy = CrossEntropyLoss[Self.dtype]()

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
        mut targets: Optional[Tensor[index_dtype]],
    ) -> Tuple[
        Tensor[Self.dtype], Optional[Tensor[Self.dtype]]
    ] where Self.dtype.is_floating_point():
        var logits = self.lookup_table(indices)
        if targets:
            ref shape = logits.shape()
            var B, T, C = shape[0], shape[1], shape[2]
            var logits_reshaped = logits.reshape(B * T, C)
            var targets_reshaped = targets.value().reshape(B * T)
            var loss = self.cross_entropy(logits_reshaped, targets_reshaped^)
            return logits_reshaped, loss
        else:
            return logits, None

    def generate(
        mut self, indices: Tensor[index_dtype], max_new_tokens: Int
    ) raises -> Tensor[index_dtype] where Self.dtype.is_floating_point():
        var indices_appended = indices
        for _ in range(max_new_tokens):
            var targets: Optional[Tensor[index_dtype]] = None
            var logits, _loss = self(indices, targets)
            logits = logits[s(), i(-1), s()]
            var probs = logits.softmax[track_grad=False](axes=[-1])
            var indices_next = Tensor[Self.dtype].multinomial(
                probs, num_samples=1
            )
            indices_appended = Tensor[index_dtype].concat[track_grad=False](
                [indices_appended, indices_next], axis=1
            )
        return indices_appended


def estimate_loss[
    dtype: DType
](
    mut model: BigramLanguageModel[dtype],
    train_loader: DataLoader[WindowingDataset[index_dtype], _],
    val_loader: DataLoader[WindowingDataset[index_dtype], _],
    eval_iters: Int,
) raises -> Dict[String, Scalar[dtype]] where dtype.is_floating_point():
    var out = Dict[String, Scalar[dtype]]()
    model.eval()
    for split in ["train", "val"]:
        var losses = Tensor[dtype].zeros(eval_iters)
        for k in range(eval_iters):
            var X_raw, Y_raw = (
                train_loader.sample() if split
                == "train" else val_loader.sample()
            )
            var X = X_raw.unsqueeze(0)
            var Y = Y_raw.unsqueeze(0)
            var targets = Optional(Y)
            var _logits, loss = model(X, targets)
            losses[k] = loss.value().item()
        out[String(split)] = losses.mean().item()
    model.train()
    return out^


def main() raises:
    var file_name = "input.txt"
    var path = Path(file_name)
    var text = path.read_text()
    var tokenizer = SimpleTokenizer(text)
    var vocab_size = len(tokenizer)

    var data = tokenizer.encode_as[dtype=index_dtype](text)
    var split = Int(0.9 * Float64(len(data)))
    var train_data = data[:split]
    var validation_data = data[split:]

    var seq_length = 8
    var batch_size = 32

    max_iters = 100
    eval_interval = 4
    learning_rate = Scalar[dtype](1e-2)
    var eval_iters = 200
    var max_new_tokens = 200

    var train_ds = WindowingDataset[index_dtype](
        Tensor[index_dtype].from_list(train_data), seq_length
    )
    var val_ds = WindowingDataset[index_dtype](
        Tensor[index_dtype].from_list(validation_data), seq_length
    )

    var train_loader = train_ds.into_loader(batch_size, shuffle=False)
    var val_loader = val_ds.into_loader(batch_size, shuffle=False)

    var model = BigramLanguageModel[dtype](vocab_size)
    var optimizer = SGD(model.parameters(), lr=learning_rate)
    # Generate before training
    var gen_ctx = Tensor[index_dtype].zeros(1, 1)
    var generated = model.generate(gen_ctx, max_new_tokens=max_new_tokens)
    var gen_row = generated[i(0), s()]
    var gen_tokens = gen_row.tolist()

    print("Generated before training:")
    print(tokenizer.decode_from(gen_tokens))

    for epoch in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if epoch % eval_interval == 0:
            var losses = estimate_loss(
                model, train_loader, val_loader, eval_iters
            )
            print(
                "Epoc:",
                epoch,
                "train loss "
                + String(losses["train"])
                + ", val loss "
                + String(losses["val"]),
            )

        for batch in train_loader:
            # sample a batch of data
            var xb, yb = batch.features, batch.labels
            var targets = Optional(yb)
            # evaluate the loss
            var _logits, loss = model(xb, targets)
            optimizer.zero_grad()
            loss.value().backward()
            optimizer.step()

        # Generate after completion of each epoch
        gen_ctx = Tensor[index_dtype].zeros(1, 1)
        generated = model.generate(gen_ctx, max_new_tokens=max_new_tokens)
        gen_row = generated[i(0), s()]
        gen_tokens = gen_row.tolist()
        print("Generated after epoch:", epoch + 1)
        print(tokenizer.decode_from(gen_tokens))

    print("All iterations done")
