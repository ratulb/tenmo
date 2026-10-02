"""BERT-style encoder stack: bidirectional blocks, embeddings, heads.

The understanding counterpart to `gpt.mojo`: every block runs
`SelfAttention(causal=False)`, so each position attends the whole
sequence. No new kernels or autograd primitives — everything composes
existing CPU+GPU layers.

    make_padding_mask  — (N,) lengths + T -> (N,T) bool key-padding mask.
    EncoderBlock       — pre-norm residual block, bidirectional attention.
    BertEmbeddings     — token + segment + position tables, sum + LayerNorm
                         + dropout.
    BertMLMHead        — Linear -> GeLU -> LayerNorm -> Linear(vocab) over
                         all positions (masked-LM pretraining).
    BertClassifierHead — first-token slice -> dropout -> Linear(n_labels).

Ids `(B,T)` in, and the same encoder weights serve both heads:
`BertMLMHead` scores every position for MLM pretraining,
`BertClassifierHead` reads only the first (`[CLS]`) vector for
classification.
"""

from .shared.mnemonics import DEFAULT_INDEX_DTYPE
from .shared.panic import panic
from .shared.shapes import Shape
from .tensor import Tensor
from .layer_trait import LayerTrait
from .net import Linear, GeLU
from .dropout import Dropout
from .layernorm import LayerNorm
from .attention import SelfAttention
from .embedding import Embedding
from .positional import PositionalEmbedding
from .gpt import MLP
from .named_parameter import NamedParameter
from .gpu.device import GPU


def make_padding_mask(
    lengths: Tensor[DType.int64], T: Int
) -> Tensor[DType.bool]:
    """Build a `(N,T)` key-padding mask from per-row lengths.

    `lengths` is `(N,)` int64; entry `[b, t]` is True iff
    `t < min(lengths[b], T)`. Lengths past `T` clamp. Starts as all-False
    and each real position is flipped True, so rows beyond their length
    keep their initial False.

    Host-side scalar loop — mask building is data prep, not the hot path.
    """
    var N = lengths.shape()[0]
    var mask = Tensor[DType.bool].zeros(Shape(N, T))
    for b in range(N):
        var L = Int(lengths[b])
        var lim = L if L < T else T
        for t in range(lim):
            mask[b, t] = True
    return mask^


struct EncoderBlock[dtype: DType](LayerTrait):
    """One bidirectional transformer block (pre-norm, residual).

    Composition (mirror of `TransformerBlock`):

        h = x + Attention(LayerNorm(x), causal=False)   # full context
        x = h + MLP(LayerNorm(h))

    Pre-norm (each sub-layer renormalizes its own input first) rather than
    the original BERT's post-norm, which trains worse — a documented
    deviation. Attention is constructed with `causal=False`, so unlike the
    GPT stack every query position attends every key position. For padded
    batches use `forward_padded` (separate entry point because
    `LayerTrait` pins the exact `__call__(x, sync)` signature — the same
    constraint that forced `SelfAttention.forward_padded`).
    """

    comptime InputDType = Self.dtype

    var ln_1: LayerNorm[Self.dtype]
    var attn: SelfAttention[Self.dtype]
    var ln_2: LayerNorm[Self.dtype]
    var mlp: MLP[Self.dtype]
    var n_embd: Int
    var n_head: Int
    var training: Bool

    def __init__(
        out self,
        n_embd: Int,
        n_head: Int,
        dropout_p: Float32 = 0.0,
        init_seed: Optional[Int] = None,
        init_method: String = "xavier",
    ):
        """Create an encoder block (attention is bidirectional)."""
        self.n_embd = n_embd
        self.n_head = n_head
        self.training = True
        self.ln_1 = LayerNorm[Self.dtype](n_embd)
        self.attn = SelfAttention[Self.dtype](
            n_embd,
            n_head,
            dropout_p=dropout_p,
            init_seed=init_seed,
            init_method=init_method,
            causal=False,
            # BERT keeps the QKV bias: opt in explicitly, since the
            # library default is now bias-free.
            qkv_bias=True,
        )
        self.ln_2 = LayerNorm[Self.dtype](n_embd)
        self.mlp = MLP[Self.dtype](
            n_embd,
            dropout_p=dropout_p,
            init_seed=init_seed,
            init_method=init_method,
        )

    def __call__(
        mut self, x: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        if self.training:
            return self._forward[track_grad=True](x, sync=sync)
        else:
            return self._forward[track_grad=False](x, sync=sync)

    def forward_padded(
        mut self,
        x: Tensor[Self.dtype],
        pad_mask: Tensor[DType.bool],
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Run the block with a `(B,T)` key-padding mask (True = real).

        Same computation as `__call__`, except the mask travels
        all the way down into attention (`_forward` ->
        `SelfAttention.forward_padded`), where padded keys are zeroed
        out. Everything else — norms, MLP, residuals — is untouched, so
        padded *queries* still produce outputs; the caller simply never
        reads those rows (loss and pooling only look at real positions).

        See `SelfAttention.forward_padded` for why this is a separate
        entry point rather than a `__call__` parameter.
        """
        if self.training:
            return self._forward[track_grad=True](
                x, pad_mask, sync=sync
            )
        else:
            return self._forward[track_grad=False](
                x, pad_mask, sync=sync
            )

    def _forward[track_grad: Bool](
        mut self,
        x: Tensor[Self.dtype],
        pad_mask: Optional[Tensor[DType.bool]] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Block computation: pre-norm + residual two-step. Each residual
        `+` is an explicit grad-tracked add under this call's
        `track_grad`; sub-layers fork on their own `training` flags (same
        division of labor as `TransformerBlock._forward`).
        """
        comptime assert Self.dtype.is_floating_point()
        var h = self.ln_1(x, sync=sync)
        var attn_out: Tensor[Self.dtype]
        if pad_mask:
            attn_out = self.attn.forward_padded(h, pad_mask.value(), sync=sync)
        else:
            attn_out = self.attn(h, sync=sync)
        var residual_1 = x.__add__[track_grad=track_grad](attn_out)
        var h2 = self.ln_2(residual_1, sync=sync)
        var mlp_out = self.mlp(h2, sync=sync)
        var out = residual_1.__add__[track_grad=track_grad](mlp_out)
        return out^

    def parameters(
        ref self,
    ) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = self.ln_1.parameters()
        var attn = self.attn.parameters()
        for p in range(len(attn)):
            params.append(attn[p])
        var ln2 = self.ln_2.parameters()
        for p in range(len(ln2)):
            params.append(ln2[p])
        var mlp = self.mlp.parameters()
        for p in range(len(mlp)):
            params.append(mlp[p])
        return params^

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        var result = self.ln_1.named_parameters(prefix + "ln_1.")
        var attn = self.attn.named_parameters(prefix + "attn.")
        for p in range(len(attn)):
            result.append(attn[p])
        var ln2 = self.ln_2.named_parameters(prefix + "ln_2.")
        for p in range(len(ln2)):
            result.append(ln2[p])
        var mlp = self.mlp.named_parameters(prefix + "mlp.")
        for p in range(len(mlp)):
            result.append(mlp[p])
        return result^

    def num_parameters(self) -> Int:
        return (
            self.ln_1.num_parameters()
            + self.attn.num_parameters()
            + self.ln_2.num_parameters()
            + self.mlp.num_parameters()
        )

    def train(mut self):
        self.training = True
        self.ln_1.train()
        self.attn.train()
        self.ln_2.train()
        self.mlp.train()

    def eval(mut self):
        self.training = False
        self.ln_1.eval()
        self.attn.eval()
        self.ln_2.eval()
        self.mlp.eval()

    def to_gpu(self, gpu: Optional[GPU] = None) raises -> Self:
        var out = self
        out.ln_1 = self.ln_1.to_gpu(gpu=gpu)
        out.attn = self.attn.to_gpu(gpu=gpu)
        out.ln_2 = self.ln_2.to_gpu(gpu=gpu)
        out.mlp = self.mlp.to_gpu(gpu=gpu)
        return out^

    def to_cpu(self) raises -> Self:
        var out = self
        out.ln_1 = self.ln_1.to_cpu()
        out.attn = self.attn.to_cpu()
        out.ln_2 = self.ln_2.to_cpu()
        out.mlp = self.mlp.to_cpu()
        return out^


struct BertEmbeddings[dtype: DType, index_dtype: DType = DEFAULT_INDEX_DTYPE](
    LayerTrait
):
    """BERT embedding recipe: token + segment + position, LayerNorm, dropout.

    `out[b,t] = drop(LN(wte[ids[b,t]] + tte[seg[b,t]] + wpe[t]))`.
    `wte` carries `padding_idx` (that row stays zeros, grad-free);
    `tte` is the 2-row segment table (single-segment input uses all-zero
    ids); `wpe` is learned absolute positions. `T > n_ctx` panics (same
    entrance guard as `GPTEmbedding` — `LayerTrait.__call__` is
    non-raising, so `raise` is impossible there).

    `tte` covers the sentence-A/B case; IMDB is single-segment, so it uses
    all-zero ids and the table is there for API completeness.

    `__call__(x)` takes token ids with implicit all-zero segments;
    `forward_segments(x, seg)` is the general entry point.
    """

    comptime InputDType = Self.index_dtype
    comptime OutputDType = Self.dtype

    var wte: Embedding[Self.dtype, Self.index_dtype]
    var tte: Embedding[Self.dtype, Self.index_dtype]
    var wpe: PositionalEmbedding[Self.dtype, Self.index_dtype]
    var ln: LayerNorm[Self.dtype]
    var drop: Dropout[Self.dtype]
    var n_ctx: Int
    var n_embd: Int
    var training: Bool

    def __init__(
        out self,
        n_vocab: Int,
        n_ctx: Int,
        n_embd: Int,
        padding_idx: Optional[Int] = None,
        dropout_p: Float32 = 0.0,
        init_seed: Optional[Int] = None,
        init_method: String = "xavier",
    ):
        """Create the three tables + norm + dropout."""
        if n_vocab < 1 or n_ctx < 1 or n_embd < 1:
            panic(
                "BertEmbeddings: n_vocab, n_ctx, n_embd must be >= 1, got",
                String(n_vocab),
                String(n_ctx),
                String(n_embd),
            )
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.training = True
        self.wte = Embedding[Self.dtype, Self.index_dtype](
            num_embeddings=n_vocab,
            embedding_dim=n_embd,
            padding_idx=padding_idx,
            init_seed=init_seed,
            init_method=init_method,
        )
        self.tte = Embedding[Self.dtype, Self.index_dtype](
            num_embeddings=2,
            embedding_dim=n_embd,
            init_seed=init_seed,
            init_method=init_method,
        )
        self.wpe = PositionalEmbedding[Self.dtype, Self.index_dtype](
            n_ctx,
            n_embd,
            init_seed=init_seed,
            init_method=init_method,
        )
        self.ln = LayerNorm[Self.dtype](n_embd)
        self.drop = Dropout[Self.dtype](Scalar[Self.dtype](dropout_p))

    def __call__(
        mut self, x: Tensor[Self.index_dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        if self.training:
            return self._forward[track_grad=True](x, sync=sync)
        else:
            return self._forward[track_grad=False](x, sync=sync)

    def forward_segments(
        mut self,
        x: Tensor[Self.index_dtype],
        seg: Tensor[Self.index_dtype],
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Embed with explicit `(B,T)` segment ids (0/1).

        Use this when the input has two spans (e.g. sentence
        pairs); for plain IMDB reviews `__call__` (all-zero segments)
        is equivalent and shorter.
        """
        if self.training:
            return self._forward[track_grad=True](x, seg, sync=sync)
        else:
            return self._forward[track_grad=False](x, seg, sync=sync)

    def _forward[track_grad: Bool](
        mut self,
        x: Tensor[Self.index_dtype],
        seg: Optional[Tensor[Self.index_dtype]] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """`drop(LN(wte(x) + tte(seg|0) + wpe(0..T-1)))` with two explicit
        grad-tracked adds, so `.backward()` reaches all three tables.
        """
        var B = x.shape()[0]
        var T = x.shape()[1]
        if T > self.n_ctx:
            panic(
                "BertEmbeddings: sequence length T = "
                + String(T)
                + " exceeds n_ctx = "
                + String(self.n_ctx)
            )
        var seg_ids = Tensor[Self.index_dtype].zeros(Shape(B, T))
        if seg:
            seg_ids = seg.value()
        var ids: Tensor[Self.index_dtype] = Tensor[Self.index_dtype].zeros(
            Shape(B, T)
        )
        try:
            ids = PositionalEmbedding[
                Self.dtype, Self.index_dtype
            ].position_ids(self.n_ctx, B, T)
        except e:
            print(e)
            panic("BertEmbeddings.position_ids failed")
        var tok = self.wte(x, sync=sync)
        var typ = self.tte(seg_ids, sync=sync)
        var pos = self.wpe(ids, sync=sync)
        var s1 = Tensor[Self.dtype].add[track_grad=track_grad](
            tok, typ, sync=sync
        )
        var s2 = Tensor[Self.dtype].add[track_grad=track_grad](
            s1, pos, sync=sync
        )
        var normed = self.ln(s2, sync=sync)
        return self.drop(normed, sync=sync)

    def parameters(
        ref self,
    ) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = self.wte.parameters()
        var tte = self.tte.parameters()
        for p in range(len(tte)):
            params.append(tte[p])
        var wpe = self.wpe.parameters()
        for p in range(len(wpe)):
            params.append(wpe[p])
        var ln = self.ln.parameters()
        for p in range(len(ln)):
            params.append(ln[p])
        return params^

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        var result = self.wte.named_parameters(prefix + "wte.")
        var tte = self.tte.named_parameters(prefix + "tte.")
        for p in range(len(tte)):
            result.append(tte[p])
        var wpe = self.wpe.named_parameters(prefix + "wpe.")
        for p in range(len(wpe)):
            result.append(wpe[p])
        var ln = self.ln.named_parameters(prefix + "ln.")
        for p in range(len(ln)):
            result.append(ln[p])
        return result^

    def num_parameters(self) -> Int:
        return (
            self.wte.num_parameters()
            + self.tte.num_parameters()
            + self.wpe.num_parameters()
            + self.ln.num_parameters()
        )

    def train(mut self):
        self.training = True
        self.wte.train()
        self.tte.train()
        self.wpe.train()
        self.ln.train()
        self.drop.train()

    def eval(mut self):
        self.training = False
        self.wte.eval()
        self.tte.eval()
        self.wpe.eval()
        self.ln.eval()
        self.drop.eval()

    def to_gpu(self, gpu: Optional[GPU] = None) raises -> Self:
        var out = self
        out.wte = self.wte.to_gpu(gpu=gpu)
        out.tte = self.tte.to_gpu(gpu=gpu)
        out.wpe = self.wpe.to_gpu(gpu=gpu)
        out.ln = self.ln.to_gpu(gpu=gpu)
        return out^

    def to_cpu(self) raises -> Self:
        var out = self
        out.wte = self.wte.to_cpu()
        out.tte = self.tte.to_cpu()
        out.wpe = self.wpe.to_cpu()
        out.ln = self.ln.to_cpu()
        return out^


struct BertMLMHead[dtype: DType](LayerTrait):
    """Masked-LM head: `Linear(C,C) -> GeLU -> LayerNorm -> Linear(C,V)`
    over all positions. Output is `(B,T,V)` logits, scored with
    `CrossEntropyLoss(ignore_index=...)` on masked positions only. No
    residual adds, so the sub-layers' own training fork governs graph
    construction.
    """

    comptime InputDType = Self.dtype

    var fc: Linear[Self.dtype]
    var gelu: GeLU[Self.dtype]
    var ln: LayerNorm[Self.dtype]
    var head: Linear[Self.dtype]
    var n_vocab: Int
    var training: Bool

    def __init__(
        out self,
        n_embd: Int,
        n_vocab: Int,
        init_seed: Optional[Int] = None,
        init_method: String = "xavier",
    ):
        """Create the MLM head."""
        if n_embd < 1 or n_vocab < 1:
            panic(
                "BertMLMHead: n_embd and n_vocab must be >= 1, got",
                String(n_embd),
                String(n_vocab),
            )
        self.n_vocab = n_vocab
        self.training = True
        self.fc = Linear[Self.dtype](
            in_features=n_embd,
            out_features=n_embd,
            init_seed=init_seed,
            init_method=init_method,
        )
        self.gelu = GeLU[Self.dtype]()
        self.ln = LayerNorm[Self.dtype](n_embd)
        self.head = Linear[Self.dtype](
            in_features=n_embd,
            out_features=n_vocab,
            init_seed=init_seed,
            init_method=init_method,
        )

    def __call__(
        mut self, x: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        var h = self.fc(x, sync=sync)
        var g = self.gelu(h, sync=sync)
        var n = self.ln(g, sync=sync)
        return self.head(n, sync=sync)

    def parameters(
        ref self,
    ) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = self.fc.parameters()
        var ln = self.ln.parameters()
        for p in range(len(ln)):
            params.append(ln[p])
        var head = self.head.parameters()
        for p in range(len(head)):
            params.append(head[p])
        return params^

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        var result = self.fc.named_parameters(prefix + "fc.")
        var ln = self.ln.named_parameters(prefix + "ln.")
        for p in range(len(ln)):
            result.append(ln[p])
        var head = self.head.named_parameters(prefix + "head.")
        for p in range(len(head)):
            result.append(head[p])
        return result^

    def num_parameters(self) -> Int:
        return (
            self.fc.num_parameters()
            + self.ln.num_parameters()
            + self.head.num_parameters()
        )

    def train(mut self):
        self.training = True
        self.fc.train()
        self.gelu.train()
        self.ln.train()
        self.head.train()

    def eval(mut self):
        self.training = False
        self.fc.eval()
        self.gelu.eval()
        self.ln.eval()
        self.head.eval()

    def to_gpu(self, gpu: Optional[GPU] = None) raises -> Self:
        var out = self
        out.fc = self.fc.to_gpu(gpu=gpu)
        out.ln = self.ln.to_gpu(gpu=gpu)
        out.head = self.head.to_gpu(gpu=gpu)
        return out^

    def to_cpu(self) raises -> Self:
        var out = self
        out.fc = self.fc.to_cpu()
        out.ln = self.ln.to_cpu()
        out.head = self.head.to_cpu()
        return out^


struct BertClassifierHead[dtype: DType](LayerTrait):
    """Sequence head: `[CLS]` slice -> dropout -> `Linear(C, n_labels)`.
    `(B,T,C)` float in, `(B,n_labels)` logits out
    for 2-class CE. No pooling helper exists (`pooling.mojo` is
    vision-only), so extraction is an explicit `slice` + `reshape` view,
    both graph-tracked.
    """

    comptime InputDType = Self.dtype

    var drop: Dropout[Self.dtype]
    var clf: Linear[Self.dtype]
    var n_labels: Int
    var training: Bool

    def __init__(
        out self,
        n_embd: Int,
        n_labels: Int = 2,
        dropout_p: Float32 = 0.0,
        init_seed: Optional[Int] = None,
        init_method: String = "xavier",
    ):
        """Create the classifier head."""
        if n_embd < 1 or n_labels < 1:
            panic(
                "BertClassifierHead: n_embd and n_labels must be >= 1, got",
                String(n_embd),
                String(n_labels),
            )
        self.n_labels = n_labels
        self.training = True
        self.drop = Dropout[Self.dtype](Scalar[Self.dtype](dropout_p))
        self.clf = Linear[Self.dtype](
            in_features=n_embd,
            out_features=n_labels,
            init_seed=init_seed,
            init_method=init_method,
        )

    def __call__(
        mut self, x: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        # The first-token slice/reshape below is an explicit tensor op
        # (not hidden inside a sub-layer), so unlike the MLM head this
        # module forks on `track_grad` itself: training records the
        # slice for `.backward()`; evaluation skips the bookkeeping
        # (house rule I5: eval runs graph-free).
        if self.training:
            return self._forward[track_grad=True](x, sync=sync)
        else:
            return self._forward[track_grad=False](x, sync=sync)

    def _forward[track_grad: Bool](
        mut self, x: Tensor[Self.dtype], sync: Bool
    ) -> Tensor[Self.dtype]:
        var B = x.shape()[0]
        var C = x.shape()[2]
        var first = x.slice[track_grad=track_grad](0, 1, axis=1).reshape[
            track_grad=track_grad
        ](Shape(B, C), sync=sync)
        var dropped = self.drop(first, sync=sync)
        return self.clf(dropped, sync=sync)

    def parameters(
        ref self,
    ) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        return self.clf.parameters()

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        return self.clf.named_parameters(prefix + "clf.")

    def num_parameters(self) -> Int:
        return self.clf.num_parameters()

    def train(mut self):
        self.training = True
        self.drop.train()
        self.clf.train()

    def eval(mut self):
        self.training = False
        self.drop.eval()
        self.clf.eval()

    def to_gpu(self, gpu: Optional[GPU] = None) raises -> Self:
        var out = self
        out.clf = self.clf.to_gpu(gpu=gpu)
        return out^

    def to_cpu(self) raises -> Self:
        var out = self
        out.clf = self.clf.to_cpu()
        return out^


struct BertForMLM[dtype: DType](LayerTrait):
    """Full MLM-pretraining model: embeddings + N encoder blocks + MLM head.

    Ids `(B,T)` in, vocabulary logits `(B,T,V)` out.

    The wrapper exists so that (1) the optimizer gets one flat parameter
    list, (2) checkpointing keys agree on the `emb.`/`blk{i}.` prefix
    scheme, and (3) one `model.train()`/`eval()` flips every submodule
    together. Submodules still fork their own forward on their own flag,
    so the wrapper just chains calls.
    """

    comptime InputDType = DType.int64
    comptime OutputDType = Self.dtype

    var emb: BertEmbeddings[Self.dtype]
    var blocks: List[EncoderBlock[Self.dtype]]
    var head: BertMLMHead[Self.dtype]
    var training: Bool
    var n_layer: Int

    def __init__(
        out self,
        n_vocab: Int,
        n_ctx: Int,
        n_embd: Int,
        n_head: Int,
        n_layer: Int,
        padding_idx: Optional[Int] = None,
        dropout_p: Float32 = 0.1,
        init_seed: Optional[Int] = None,
        init_method: String = "xavier",
    ):
        """Create embeddings + `n_layer` blocks + MLM head.

        `padding_idx` forwards to the token table (that row's grad is
        zeroed — pass the corpus PAD id so filler never learns).
        """
        if n_layer < 1:
            panic("BertForMLM: n_layer must be >= 1")
        self.emb = BertEmbeddings[Self.dtype](
            n_vocab,
            n_ctx,
            n_embd,
            padding_idx=padding_idx,
            dropout_p=dropout_p,
            init_seed=init_seed,
            init_method=init_method,
        )
        self.blocks = List[EncoderBlock[Self.dtype]](capacity=n_layer)
        for _ in range(n_layer):
            self.blocks.append(
                EncoderBlock[Self.dtype](
                    n_embd,
                    n_head,
                    dropout_p=dropout_p,
                    init_seed=init_seed,
                    init_method=init_method,
                )
            )
        self.head = BertMLMHead[Self.dtype](
            n_embd, n_vocab, init_seed=init_seed, init_method=init_method
        )
        self.training = True
        self.n_layer = n_layer

    def __init__(out self, *, copy: Self):
        # Explicit copy ctor (the GPTModel precedent): the `blocks`
        # List is not implicitly copyable, so the compiler cannot
        # synthesize this — `.copy()` the list, assign the rest.
        self.emb = copy.emb
        self.blocks = copy.blocks.copy()
        self.head = copy.head
        self.training = copy.training
        self.n_layer = copy.n_layer

    def __call__(
        mut self, x: Tensor[Self.InputDType], sync: Bool = True
    ) -> Tensor[Self.OutputDType]:
        """Dense forward (no padding mask) — ids `(B,T)` to logits."""
        var h = self.emb(x, sync=sync)
        for i in range(len(self.blocks)):
            ref blk = self.blocks[i]
            h = blk(h, sync=sync)
        return self.head(h, sync=sync)

    def forward_padded(
        mut self,
        x: Tensor[Self.InputDType],
        pad_mask: Tensor[DType.bool],
        sync: Bool = True,
    ) -> Tensor[Self.OutputDType]:
        """Padded forward — the training entry point.

        `pad_mask` (from `make_padding_mask`) travels into
        every block's `forward_padded`, so padded keys contribute zero
        attention weight. Separate from `__call__` because `LayerTrait`
        pins the `__call__(x, sync)` signature.
        """
        var h = self.emb(x, sync=sync)
        for i in range(len(self.blocks)):
            ref blk = self.blocks[i]
            h = blk.forward_padded(h, pad_mask, sync=sync)
        return self.head(h, sync=sync)

    def parameters(
        ref self,
    ) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = self.emb.parameters()
        for i in range(len(self.blocks)):
            ref blk = self.blocks[i]
            var bp = blk.parameters()
            for p in range(len(bp)):
                params.append(bp[p])
        var hp = self.head.parameters()
        for p in range(len(hp)):
            params.append(hp[p])
        return params^

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        var result = self.emb.named_parameters(prefix + "emb.")
        for i in range(len(self.blocks)):
            ref blk = self.blocks[i]
            var bp = blk.named_parameters(
                prefix + "blk" + String(i) + "."
            )
            for p in range(len(bp)):
                result.append(bp[p])
        var hp = self.head.named_parameters(prefix + "head.")
        for p in range(len(hp)):
            result.append(hp[p])
        return result^

    def num_parameters(self) -> Int:
        var n = self.emb.num_parameters() + self.head.num_parameters()
        for i in range(len(self.blocks)):
            ref blk = self.blocks[i]
            n += blk.num_parameters()
        return n

    def train(mut self):
        self.training = True
        self.emb.train()
        for i in range(len(self.blocks)):
            ref blk = self.blocks[i]
            blk.train()
        self.head.train()

    def eval(mut self):
        self.training = False
        self.emb.eval()
        for i in range(len(self.blocks)):
            ref blk = self.blocks[i]
            blk.eval()
        self.head.eval()

    def to_gpu(self, gpu: Optional[GPU] = None) raises -> Self:
        var out = self
        out.emb = self.emb.to_gpu(gpu=gpu)
        var moved = List[EncoderBlock[Self.dtype]](capacity=len(self.blocks))
        for i in range(len(self.blocks)):
            ref blk = self.blocks[i]
            moved.append(blk.to_gpu(gpu=gpu))
        out.blocks = moved^
        out.head = self.head.to_gpu(gpu=gpu)
        return out^

    def to_cpu(self) raises -> Self:
        var out = self
        out.emb = self.emb.to_cpu()
        var moved = List[EncoderBlock[Self.dtype]](capacity=len(self.blocks))
        for i in range(len(self.blocks)):
            ref blk = self.blocks[i]
            moved.append(blk.to_cpu())
        out.blocks = moved^
        out.head = self.head.to_cpu()
        return out^


struct BertForSequenceClassification[dtype: DType](LayerTrait):
    """Full sentiment model: embeddings + N encoder blocks + CLS head.

    Ids `(B,T)` in, sentiment logits `(B, n_labels)` out. Same encoder as
    `BertForMLM` — only the head differs — so MLM weights transfer by name:
    `apply_to_model` fills the `emb.`/`blk{i}.` keys it finds and leaves
    the fresh `clf.` head untouched. Set `n_labels=2` for IMDB.
    """

    comptime InputDType = DType.int64
    comptime OutputDType = Self.dtype

    var emb: BertEmbeddings[Self.dtype]
    var blocks: List[EncoderBlock[Self.dtype]]
    var head: BertClassifierHead[Self.dtype]
    var training: Bool
    var n_layer: Int

    def __init__(
        out self,
        n_vocab: Int,
        n_ctx: Int,
        n_embd: Int,
        n_head: Int,
        n_layer: Int,
        n_labels: Int = 2,
        padding_idx: Optional[Int] = None,
        dropout_p: Float32 = 0.1,
        init_seed: Optional[Int] = None,
        init_method: String = "xavier",
    ):
        if n_layer < 1:
            panic("BertForSequenceClassification: n_layer must be >= 1")
        self.emb = BertEmbeddings[Self.dtype](
            n_vocab,
            n_ctx,
            n_embd,
            padding_idx=padding_idx,
            dropout_p=dropout_p,
            init_seed=init_seed,
            init_method=init_method,
        )
        self.blocks = List[EncoderBlock[Self.dtype]](capacity=n_layer)
        for _ in range(n_layer):
            self.blocks.append(
                EncoderBlock[Self.dtype](
                    n_embd,
                    n_head,
                    dropout_p=dropout_p,
                    init_seed=init_seed,
                    init_method=init_method,
                )
            )
        self.head = BertClassifierHead[Self.dtype](
            n_embd,
            n_labels,
            dropout_p=dropout_p,
            init_seed=init_seed,
            init_method=init_method,
        )
        self.training = True
        self.n_layer = n_layer

    def __init__(out self, *, copy: Self):
        # Same explicit copy ctor as BertForMLM (List is not
        # implicitly copyable — the GPTModel precedent).
        self.emb = copy.emb
        self.blocks = copy.blocks.copy()
        self.head = copy.head
        self.training = copy.training
        self.n_layer = copy.n_layer

    def __call__(
        mut self, x: Tensor[Self.InputDType], sync: Bool = True
    ) -> Tensor[Self.OutputDType]:
        var h = self.emb(x, sync=sync)
        for i in range(len(self.blocks)):
            ref blk = self.blocks[i]
            h = blk(h, sync=sync)
        return self.head(h, sync=sync)

    def forward_padded(
        mut self,
        x: Tensor[Self.InputDType],
        pad_mask: Tensor[DType.bool],
        sync: Bool = True,
    ) -> Tensor[Self.OutputDType]:
        var h = self.emb(x, sync=sync)
        for i in range(len(self.blocks)):
            ref blk = self.blocks[i]
            h = blk.forward_padded(h, pad_mask, sync=sync)
        return self.head(h, sync=sync)

    def parameters(
        ref self,
    ) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = self.emb.parameters()
        for i in range(len(self.blocks)):
            ref blk = self.blocks[i]
            var bp = blk.parameters()
            for p in range(len(bp)):
                params.append(bp[p])
        var hp = self.head.parameters()
        for p in range(len(hp)):
            params.append(hp[p])
        return params^

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        var result = self.emb.named_parameters(prefix + "emb.")
        for i in range(len(self.blocks)):
            ref blk = self.blocks[i]
            var bp = blk.named_parameters(
                prefix + "blk" + String(i) + "."
            )
            for p in range(len(bp)):
                result.append(bp[p])
        var hp = self.head.named_parameters(prefix + "head.")
        for p in range(len(hp)):
            result.append(hp[p])
        return result^

    def num_parameters(self) -> Int:
        var n = self.emb.num_parameters() + self.head.num_parameters()
        for i in range(len(self.blocks)):
            ref blk = self.blocks[i]
            n += blk.num_parameters()
        return n

    def train(mut self):
        self.training = True
        self.emb.train()
        for i in range(len(self.blocks)):
            ref blk = self.blocks[i]
            blk.train()
        self.head.train()

    def eval(mut self):
        self.training = False
        self.emb.eval()
        for i in range(len(self.blocks)):
            ref blk = self.blocks[i]
            blk.eval()
        self.head.eval()

    def to_gpu(self, gpu: Optional[GPU] = None) raises -> Self:
        var out = self
        out.emb = self.emb.to_gpu(gpu=gpu)
        var moved = List[EncoderBlock[Self.dtype]](capacity=len(self.blocks))
        for i in range(len(self.blocks)):
            ref blk = self.blocks[i]
            moved.append(blk.to_gpu(gpu=gpu))
        out.blocks = moved^
        out.head = self.head.to_gpu(gpu=gpu)
        return out^

    def to_cpu(self) raises -> Self:
        var out = self
        out.emb = self.emb.to_cpu()
        var moved = List[EncoderBlock[Self.dtype]](capacity=len(self.blocks))
        for i in range(len(self.blocks)):
            ref blk = self.blocks[i]
            moved.append(blk.to_cpu())
        out.blocks = moved^
        out.head = self.head.to_cpu()
        return out^
