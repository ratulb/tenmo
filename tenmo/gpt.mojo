"""GPT stack — MLP, TransformerBlock, GPTEmbedding, GPTModel.

Compositions over existing layers, so no new autograd primitive is needed.
Mapping to GPT-2: `MLP` = `transformer.h.{i}.mlp`, `TransformerBlock` =
`transformer.h.{i}`, `GPTEmbedding` = `wte + wpe`, `GPTModel` = the lot.

`__call__` forks on `self.training` into `_forward[track_grad=...]`; sub-
layers fork on their own flags, so `train()`/`eval()` cascades down.
"""

from .shared.mnemonics import DEFAULT_INDEX_DTYPE
from .shared.panic import panic
from .tensor import Tensor
from .shared.shapes import Shape
from .layer_trait import LayerTrait
from .net import Linear, GeLU
from .dropout import Dropout
from .layernorm import LayerNorm
from .attention import SelfAttention
from .embedding import Embedding
from .positional import PositionalEmbedding
from .matmul import Matmul
from .named_parameter import NamedParameter
from .gpu.device import GPU

@fieldwise_init
struct MLP[dtype: DType](LayerTrait):
    """Positionwise feed-forward MLP (GPT-2 `transformer.h.{i}.mlp`).

    `c_fc` expands `n_embd -> 4*n_embd`, GeLU, `c_proj` contracts back to
    `n_embd`, then dropout on the projection output.

    Args:
        n_embd:     Hidden/embedding dimension (C); the MLP expands it 4x.
        dropout_p:  MLP-output dropout rate (training only, default 0.0).
        init_seed / init_method: Forwarded to both Linear sub-layers.
    """

    # Homogeneous layer: consumes and produces Tensor[Self.dtype];
    # OutputDType inherits InputDType (LayerTrait default).
    comptime InputDType = Self.dtype

    var c_fc: Linear[Self.dtype]
    var gelu: GeLU[Self.dtype]
    var c_proj: Linear[Self.dtype]
    var drop: Dropout[Self.dtype]
    var n_embd: Int
    var training: Bool

    def __init__(
        out self,
        n_embd: Int,
        dropout_p: Float32 = 0.0,
        init_seed: Optional[Int] = None,
        init_method: String = "xavier",
    ):
        """Create the positionwise MLP.

        Args:
            n_embd: Hidden dimension `C`; also constructs the 4C
                intermediate width.
            dropout_p: Residual-path dropout rate (training only).
            init_seed: Forwarded to both inner Linears.
            init_method: Forwarded to both inner Linears.
        """
        if n_embd < 1:
            panic(
                "MLP: n_embd must be >= 1, got",
                String(n_embd),
            )
        self.n_embd = n_embd
        self.training = True
        self.c_fc = Linear[Self.dtype](
            in_features=n_embd,
            out_features=4 * n_embd,
            init_seed=init_seed,
            init_method=init_method,
        )
        self.gelu = GeLU[Self.dtype]()
        self.c_proj = Linear[Self.dtype](
            in_features=4 * n_embd,
            out_features=n_embd,
            init_seed=init_seed,
            init_method=init_method,
        )
        self.drop = Dropout[Self.dtype](Scalar[Self.dtype](dropout_p))

    def __call__(
        mut self, x: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        if self.training:
            return self._forward[track_grad=True](x, sync=sync)
        else:
            return self._forward[track_grad=False](x, sync=sync)

    def _forward[track_grad: Bool](
        mut self,
        x: Tensor[Self.dtype],
        sync: Bool,
    ) -> Tensor[Self.dtype]:
        """MLP computation: `drop(c_proj(gelu(c_fc(x))))`.

        The two Linears branch on their own `training` flags (kept in sync
        by the `train()`/`eval()` cascade), so `track_grad` here is
        bookkeeping consistency rather than a live input.
        """
        var h = self.c_fc(x, sync=sync)
        h = self.gelu(h, sync=sync)
        h = self.c_proj(h, sync=sync)
        return self.drop(h, sync=sync)

    def parameters(
        ref self,
    ) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = self.c_fc.parameters()
        var proj = self.c_proj.parameters()
        for p in range(len(proj)):
            params.append(proj[p])
        return params^

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        var result = self.c_fc.named_parameters(prefix + "c_fc.")
        ref proj = self.c_proj.named_parameters(prefix + "c_proj.")
        for p in range(len(proj)):
            result.append(proj[p])
        return result^

    def num_parameters(self) -> Int:
        return self.c_fc.num_parameters() + self.c_proj.num_parameters()

    def train(mut self):
        self.training = True
        self.c_fc.train()
        self.gelu.train()
        self.c_proj.train()
        self.drop.train()

    def eval(mut self):
        self.training = False
        self.c_fc.eval()
        self.gelu.eval()
        self.c_proj.eval()
        self.drop.eval()

    def to_gpu(self, gpu: Optional[GPU] = None) raises -> MLP[Self.dtype]:
        var out = self
        out.c_fc = self.c_fc.to_gpu(gpu=gpu)
        out.c_proj = self.c_proj.to_gpu(gpu=gpu)
        return out^

    def to_cpu(self) raises -> MLP[Self.dtype]:
        var out = self
        out.c_fc = self.c_fc.to_cpu()
        out.c_proj = self.c_proj.to_cpu()
        return out^

@fieldwise_init
struct TransformerBlock[dtype: DType](LayerTrait):
    """One GPT-2 transformer block (Pre-LayerNorm, residual).

    Composition:

        h = x + Attention(LayerNorm(x))      # ln_1 → attn, residual add
        x = h + MLP(LayerNorm(h))            # ln_2 → mlp, residual add

    Args:
        n_embd:      Hidden/embedding dimension (C).
        n_head:      Attention heads (must divide n_embd evenly).
        dropout_p:   Dropout rate shared by attention probabilities, MLP
                     output, and (through GPTEmbedding) embeddings.
        init_seed / init_method: Forwarded down.
        qkv_bias:   If True, the block's attention keeps a QKV (`c_attn`)
                     bias (GPT-2 layout); default False is bias-free.
    """

    # Homogeneous layer: consumes and produces Tensor[Self.dtype];
    # OutputDType inherits InputDType (LayerTrait default).
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
        qkv_bias: Bool = False,
    ):
        """Create a transformer block."""
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
            qkv_bias=qkv_bias,
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

    def _forward[track_grad: Bool](
        mut self,
        x: Tensor[Self.dtype],
        sync: Bool,
    ) -> Tensor[Self.dtype]:
        """Block computation: the pre-norm + residual two-step shape above.

        Each residual `+` is an explicit grad-tracked add so the recorded
        graph uniformly entries this call's `track_grad` (the sub-layers
        themselves make the same comptime fork internally).
        """
        comptime assert Self.dtype.is_floating_point()
        var h = self.ln_1(x, sync=sync)
        var attn_out = self.attn(h, sync=sync)
        var residual_1 = x.__add__[track_grad=track_grad](
            attn_out
        )
        var h2 = self.ln_2(residual_1, sync=sync)
        var mlp_out = self.mlp(h2, sync=sync)
        var out = residual_1.__add__[track_grad=track_grad](
            mlp_out
        )
        return out^

    def forward_step(
        mut self,
        x_1: Tensor[Self.dtype],
        k_prev: Optional[Tensor[Self.dtype]],
        v_prev: Optional[Tensor[Self.dtype]],
        sync: Bool = True,
    ) -> Tuple[
        Tensor[Self.dtype], Tensor[Self.dtype], Tensor[Self.dtype]
    ]:
        """Single-token block step: threads the K/V pair through.

        `x_1` is `(B, 1, C)`; returns `(out, k_full, v_full)` with the same
        pre-norm residual order as `_forward`. Inference-only
        (attention step is grad-free; adds are `track_grad=False`).
        """
        comptime assert Self.dtype.is_floating_point()
        var h = self.ln_1(x_1, sync=sync)
        var (attn_out, k_full, v_full) = self.attn.forward_step(
            h, k_prev, v_prev, sync
        )
        var residual_1 = x_1.__add__[track_grad=False](attn_out)
        var h2 = self.ln_2(residual_1, sync=sync)
        var mlp_out = self.mlp(h2, sync=sync)
        var out = residual_1.__add__[track_grad=False](mlp_out)
        return (out^, k_full^, v_full^)

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

@fieldwise_init
struct GPTEmbedding[
    dtype: DType, index_dtype: DType = DEFAULT_INDEX_DTYPE
](LayerTrait):
    """Token + position embeddings with embedding dropout (GPT-2 `wte`/`wpe`).

        x[t, b] = wte[tokens[b, t]] + wpe[position_id(b, t)], then dropout.

    The deliberate input-swallowing `LayerTrait` exception: consumes
    `Tensor[index_dtype]` token ids, emits float activations.

    Args:
        n_vocab:     Vocabulary size (rows of `wte`).
        n_ctx:       Context length (rows of `wpe`); also the max `T`.
        n_embd:      Embedding dimension (cols of both tables).
        dropout_p:   Embedding-output dropout rate (training only).
        init_seed / init_method: Forwarded to both tables.
    """

    # Index-honest boundary: consumes raw token-id tensors (index_dtype),
    # produces Tensor[dtype] — no internal cast hack (contrast Embedding).
    # A MixedSequential seam into this layer is a leaf cast: indices carry
    # no gradients.
    comptime InputDType = Self.index_dtype
    comptime OutputDType = Self.dtype

    var wte: Embedding[Self.dtype, Self.index_dtype]
    var wpe: PositionalEmbedding[Self.dtype, Self.index_dtype]
    var drop: Dropout[Self.dtype]
    var n_ctx: Int
    var n_embd: Int
    var training: Bool

    def __init__(
        out self,
        n_vocab: Int,
        n_ctx: Int,
        n_embd: Int,
        dropout_p: Float32 = 0.0,
        init_seed: Optional[Int] = None,
        init_method: String = "xavier",
    ):
        """Create the token + position embedding pair."""
        if n_vocab < 1 or n_ctx < 1 or n_embd < 1:
            panic(
                "GPTEmbedding: n_vocab, n_ctx, n_embd must be >= 1, got",
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
            init_seed=init_seed,
            init_method=init_method,
        )
        self.wpe = PositionalEmbedding[Self.dtype, Self.index_dtype](
            n_ctx,
            n_embd,
            init_seed=init_seed,
            init_method=init_method,
        )
        self.drop = Dropout[Self.dtype](Scalar[Self.dtype](dropout_p))

    def __call__(
        mut self, x: Tensor[Self.index_dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        if self.training:
            return self._forward[track_grad=True](x, sync=sync)
        else:
            return self._forward[track_grad=False](x, sync=sync)

    def embed_step(
        mut self, x_1: Tensor[Self.index_dtype], pos: Int, sync: Bool = True
    ) raises -> Tensor[Self.dtype]:
        """Single-token embedding at absolute position `pos`.

        `x_1` is `(B, 1)` ids; the position row is built as a `(B, 1)`
        batch filled with `pos` (never 0-based: the cache counts from the
        prompt start, so the gathered `wpe` row must match the row the
        batched `_forward` would gather at this position). Same op
        sequence as `_forward` (`wte` + `wpe` + embedding dropout, all
        grad-free); dropout is the eval no-op under `eval()`. Raises when
        `pos` is outside `[0, n_ctx)`.
        """
        if pos < 0 or pos >= self.n_ctx:
            raise Error(
                "GPTEmbedding.embed_step: pos "
                + String(pos)
                + " outside [0, "
                + String(self.n_ctx)
                + ")"
            )
        var B = x_1.shape()[0]
        var ids = List[Scalar[Self.index_dtype]](capacity=B)
        for _ in range(B):
            ids.append(Scalar[Self.index_dtype](pos))
        var pos_batch = Tensor[Self.index_dtype].from_list[
            Self.index_dtype
        ](ids^).reshape(B, 1)
        var tok = self.wte(x_1, sync=sync)
        var p = self.wpe(pos_batch, sync=sync)
        var summed = tok.__add__[track_grad=False](p)
        return self.drop(summed, sync=sync)

    def _forward[track_grad: Bool](
        mut self,
        x: Tensor[Self.index_dtype],
        sync: Bool,
    ) -> Tensor[Self.dtype]:
        """Embedding computation: `drop(wte(x) + wpe(position_ids(B,T)))`.

        `T > n_ctx` aborts via `panic` here — `LayerTrait.__call__` is
        non-raising, so `raise` is impossible at this entrance. The entrance
        guard localizes the failure message rather than surfacing it as a
        gather bounds error two layers down.
        """
        var B = x.shape()[0]
        var T = x.shape()[1]
        if T > self.n_ctx:
            panic(
                "GPTEmbedding: sequence length T = "
                + String(T)
                + " exceeds n_ctx = "
                + String(self.n_ctx)
            )
        var ids: Tensor[Self.index_dtype] = Tensor[Self.index_dtype].zeros(
            Shape(B, T)
        )
        try:
            ids = PositionalEmbedding[
                Self.dtype, Self.index_dtype
            ].position_ids(self.n_ctx, B, T)
        except e:
            print(e)
            panic("GPTEmbedding.position_ids failed")
        var tok = self.wte(x, sync=sync)
        var pos = self.wpe(ids, sync=sync)
        var x_out = Tensor[Self.dtype].add[track_grad=track_grad](tok, pos, sync=sync)
        return self.drop(x_out, sync=sync)

    @always_inline
    def wte_weight(ref self) -> ref[self.wte.weight] Tensor[Self.dtype]:
        """The `(n_vocab, n_embd)` token table (forwards to wte.weight).

        GPTModel uses this both for the forward gather *and* (when tied)
        as the output projection's weight. Its gradient accumulates from
        both paths into one gradbox.
        """
        return self.wte.weight


    def parameters(
        ref self,
    ) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = self.wte.parameters()
        var wpe = self.wpe.parameters()
        for p in range(len(wpe)):
            params.append(wpe[p])
        return params^

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        var result = self.wte.named_parameters(prefix + "wte.")
        var wpe = self.wpe.named_parameters(prefix + "wpe.")
        for p in range(len(wpe)):
            result.append(wpe[p])
        return result^

    def num_parameters(self) -> Int:
        return self.wte.num_parameters() + self.wpe.num_parameters()

    def train(mut self):
        self.training = True
        self.wte.train()
        self.wpe.train()
        self.drop.train()

    def eval(mut self):
        self.training = False
        self.wte.eval()
        self.wpe.eval()
        self.drop.eval()

    def to_gpu(self, gpu: Optional[GPU] = None) raises -> Self:
        var out = self
        out.wte = self.wte.to_gpu(gpu=gpu)
        out.wpe = self.wpe.to_gpu(gpu=gpu)
        return out^

    def to_cpu(self) raises -> Self:
        var out = self
        out.wte = self.wte.to_cpu()
        out.wpe = self.wpe.to_cpu()
        return out^


struct KVCache[dtype: DType]:
    """Per-layer K/V cache for cached generation.

    `keys[i]` / `values[i]` hold layer `i`'s full key/value history as
    `(B, h, S, dh)` — the exact post-split layout `forward_step`
    consumes. Entries are appended lazily by layer index on the first
    pass (`Shape(0)` is illegal, so a missing entry is absence from the
    list, never an empty tensor); later steps overwrite entries in place.
    One cache serves one sequence: batch rows are never mixed across
    calls.
    """

    var keys: List[Tensor[Self.dtype]]
    var values: List[Tensor[Self.dtype]]

    def __init__(out self):
        self.keys = List[Tensor[Self.dtype]]()
        self.values = List[Tensor[Self.dtype]]()

    def depth(self) -> Int:
        return len(self.keys)

    def seq_len(self) -> Int:
        if len(self.keys) == 0:
            return 0
        return self.keys[0].shape()[2]

@fieldwise_init
struct GPTModel[
    dtype: DType, index_dtype: DType = DEFAULT_INDEX_DTYPE
](LayerTrait):
    """A complete GPT — embeddings, block stack, final norm, tied head.

    Forward:

        x  = GPTEmbedding(tokens)                    # wte + wpe, dropout
        for block i: x = TransformerBlock(block_i)(x)
        x  = LayerNorm(x)                            # ln_f (final norm)
        logits = x @ wte^T   (tied, default)  or  logits = lm_head(x)

    With `tie_weights == True` the output projection is the *transpose* of
    the token-embedding table, so there is no separate `lm_head`. The
    cross-entropy gradient reaches `wte` from both the forward gather and
    the head, accumulating additively into the one shared gradbox.

    The block stack is runtime-erased, so `n_layer` is arbitrary and no
    comptime value crosses the loop's store/load boundary.

    Args:
        n_vocab:     Vocabulary size (rows of `wte`).
        n_ctx:       Context length (rows of `wpe`; max sequence length).
        n_embd:      Hidden/embedding dimension (C).
        n_head:      Attention heads per block.
        n_layer:     Number of transformer blocks.
        dropout_p:   Dropout rate forwarded to embeddings, blocks, MLPs.
        tie_weights: If True (default), the output projection reuses the
                     token-embedding table (no separate lm_head).
        qkv_bias:    If True, every block's attention keeps a QKV bias
                     (GPT-2 layout); default False is bias-free.
        init_seed / init_method: Forwarded down the whole tree. One scheme
                     end to end ("xavier" by default), so no layer can
                     silently fall back to a different distribution.
    """

    # Index-honest boundary like GPTEmbedding: raw token-id tensor in,
    # logits out. A MixedSequential seam into the model is a leaf cast:
    # indices carry no gradients.
    comptime InputDType = Self.index_dtype
    comptime OutputDType = Self.dtype

    var wte_wpe: GPTEmbedding[Self.dtype, Self.index_dtype]
    var h: List[TransformerBlock[Self.dtype]]
    var ln_f: LayerNorm[Self.dtype]
    var lm_head: Optional[Linear[Self.dtype]]
    var tie_weights: Bool
    var n_vocab: Int
    var n_ctx: Int
    var n_embd: Int
    var n_layer: Int
    var training: Bool

    def __init__(
        out self,
        n_vocab: Int,
        n_ctx: Int,
        n_embd: Int,
        n_head: Int,
        n_layer: Int,
        dropout_p: Float32 = 0.0,
        tie_weights: Bool = True,
        init_seed: Optional[Int] = None,
        init_method: String = "xavier",
        qkv_bias: Bool = False,
    ):
        """Create a full GPT model."""
        if n_layer < 1:
            panic(
                "GPTModel: n_layer must be >= 1, got",
                String(n_layer),
            )
        self.tie_weights = tie_weights
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.training = True
        self.wte_wpe = GPTEmbedding[Self.dtype, Self.index_dtype](
            n_vocab,
            n_ctx,
            n_embd,
            dropout_p=dropout_p,
            init_seed=init_seed,
            init_method=init_method,
        )
        self.h = List[TransformerBlock[Self.dtype]]()
        for _ in range(n_layer):
            self.h.append(
                TransformerBlock[Self.dtype](
                    n_embd,
                    n_head,
                    dropout_p=dropout_p,
                    init_seed=init_seed,
                    init_method=init_method,
                    qkv_bias=qkv_bias,
                )
            )
        self.ln_f = LayerNorm[Self.dtype](n_embd)
        if tie_weights:
            self.lm_head = None
        else:
            self.lm_head = Linear[Self.dtype](
                in_features=n_embd,
                out_features=n_vocab,
                init_seed=init_seed,
                init_method=init_method,
            )

    def __init__(out self, *, copy: Self):
        self.wte_wpe = copy.wte_wpe
        self.h = copy.h.copy()
        self.ln_f = copy.ln_f
        self.lm_head = copy.lm_head
        self.tie_weights = copy.tie_weights
        self.n_vocab = copy.n_vocab
        self.n_ctx = copy.n_ctx
        self.n_embd = copy.n_embd
        self.n_layer = copy.n_layer
        self.training = copy.training

    def __call__(
        mut self, x: Tensor[Self.index_dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        if self.training:
            return self._forward[track_grad=True](x, sync=sync)
        else:
            return self._forward[track_grad=False](x, sync=sync)

    def _forward[track_grad: Bool](
        mut self,
        x: Tensor[Self.index_dtype],
        sync: Bool,
    ) -> Tensor[Self.dtype]:
        """Full-model forward: embeddings, blocks, final norm, head.

        Block loop note: each `self.h[i](...)` is an ordinary call into a
        block whose `__call__` forks its own comptime `_forward[track_grad]`
        on its own `training` flag — consistent with this call's because
        `train()`/`eval()` cascade. The tied output head MUST pass this
        call's `track_grad` explicitly: it is a raw tensor-level matmul,
        not a layer call, so nothing else would record it.
        """
        comptime assert Self.dtype.is_floating_point()
        var x_act = self.wte_wpe(x, sync=sync)
        for i in range(self.n_layer):
            ref block = self.h[i]
            x_act = block(x_act, sync=sync)
        x_act = self.ln_f(x_act, sync=sync)
        if self.tie_weights:
            var wt = self.wte_wpe.wte.weight.transpose[
                track_grad=track_grad
            ](sync=sync)
            var logits = Matmul[Self.dtype].forward[
                track_grad=track_grad
            ](x_act, wt, sync=sync)
            return logits^
        else:
            var logits = self.lm_head.value()(x_act, sync=sync)
            return logits^

    def forward_step(
        mut self,
        x_1: Tensor[Self.index_dtype],
        pos: Int,
        mut cache: KVCache[Self.dtype],
        sync: Bool = True,
    ) raises -> Tensor[Self.dtype]:
        """Single-token full-model step: embed, blocks, norm, head.

        `x_1` is `(B, 1)` ids at absolute position `pos`; `cache` gains
        one layer entry per block on the first pass and grows every step.
        Returns `(B, 1, V)` logits. The head mirrors `_forward` exactly
        (tied transpose-matmul, grad-free, or the `lm_head` layer).
        Inference-only; raises on out-of-range `pos` (via `embed_step`).
        """
        comptime assert Self.dtype.is_floating_point()
        var x_act = self.wte_wpe.embed_step(x_1, pos, sync=sync)
        for i in range(self.n_layer):
            ref block = self.h[i]
            if i >= cache.depth():
                var (b_out, k_full, v_full) = block.forward_step(
                    x_act, None, None, sync
                )
                cache.keys.append(k_full)
                cache.values.append(v_full)
                x_act = b_out
            else:
                var (b_out, k_full, v_full) = block.forward_step(
                    x_act,
                    Optional(cache.keys[i]),
                    Optional(cache.values[i]),
                    sync,
                )
                cache.keys[i] = k_full
                cache.values[i] = v_full
                x_act = b_out
        x_act = self.ln_f(x_act, sync=sync)
        if self.tie_weights:
            var wt = self.wte_wpe.wte.weight.transpose[
                track_grad=False
            ](sync=sync)
            return Matmul[Self.dtype].forward[track_grad=False](
                x_act, wt, sync=sync
            )
        else:
            return self.lm_head.value()(x_act, sync=sync)

    @always_inline
    def wte_weight(ref self) -> ref[self.wte_wpe.wte.weight] Tensor[Self.dtype]:
        """The shared token table `(n_vocab, n_embd)` (forwards to wte).

        Exists so parametrized code (e.g. a checkpoint loader) can read
        the tied head's weight without conditional logic on `tie_weights`.
        """
        return self.wte_wpe.wte.weight


    def parameters(
        ref self,
    ) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = self.wte_wpe.parameters()
        for i in range(len(self.h)):
            ref block = self.h[i]
            ref block_params = block.parameters()
            for p in range(len(block_params)):
                params.append(block_params[p])
        ref ln = self.ln_f.parameters()
        for p in range(len(ln)):
            params.append(ln[p])
        if not self.tie_weights:
            ref head = self.lm_head.value()
            ref head_params = head.parameters()
            for p in range(len(head_params)):
                params.append(head_params[p])
        return params^

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        var result = self.wte_wpe.named_parameters(prefix + "wte_wpe.")
        for i in range(len(self.h)):
            ref block = self.h[i]
            var block_names = block.named_parameters(
                prefix + "h." + String(i) + "."
            )
            for p in range(len(block_names)):
                result.append(block_names[p])
        var ln = self.ln_f.named_parameters(prefix + "ln_f.")
        for p in range(len(ln)):
            result.append(ln[p])
        if not self.tie_weights:
            ref head = self.lm_head.value()
            var head_names = head.named_parameters(
                prefix + "lm_head."
            )
            for p in range(len(head_names)):
                result.append(head_names[p])
        return result^

    def num_parameters(self) -> Int:
        var count = self.wte_wpe.num_parameters()
        for i in range(len(self.h)):
            ref block = self.h[i]
            count += block.num_parameters()
        count += self.ln_f.num_parameters()
        if not self.tie_weights:
            ref head = self.lm_head.value()
            count += head.num_parameters()
        return count

    def train(mut self):
        self.training = True
        self.wte_wpe.train()
        for i in range(len(self.h)):
            ref block = self.h[i]
            block.train()
        self.ln_f.train()
        if not self.tie_weights:
            ref head = self.lm_head.value()
            head.train()

    def eval(mut self):
        self.training = False
        self.wte_wpe.eval()
        for i in range(len(self.h)):
            ref block = self.h[i]
            block.eval()
        self.ln_f.eval()
        if not self.tie_weights:
            ref head = self.lm_head.value()
            head.eval()

    def to_gpu(
        self, gpu: Optional[GPU] = None
    ) raises -> GPTModel[Self.dtype, Self.index_dtype]:
        var out = GPTModel[Self.dtype, Self.index_dtype](
            self.n_vocab, self.n_ctx, self.n_embd, 1, 1,
            tie_weights=self.tie_weights,
        )
        out.wte_wpe = self.wte_wpe.to_gpu(gpu=gpu)
        var blocks = List[TransformerBlock[Self.dtype]]()
        for i in range(len(self.h)):
            ref block = self.h[i]
            blocks.append(block.to_gpu(gpu=gpu))
        out.h = blocks^
        out.n_layer = self.n_layer
        out.training = self.training
        out.ln_f = self.ln_f.to_gpu(gpu=gpu)
        if not self.tie_weights:
            ref head = self.lm_head.value()
            out.lm_head = head.to_gpu(gpu=gpu)
        return out^

    def to_cpu(self) raises -> GPTModel[Self.dtype, Self.index_dtype]:
        var out = GPTModel[Self.dtype, Self.index_dtype](
            self.n_vocab, self.n_ctx, self.n_embd, 1, 1,
            tie_weights=self.tie_weights,
        )
        out.wte_wpe = self.wte_wpe.to_cpu()
        var blocks = List[TransformerBlock[Self.dtype]]()
        for i in range(len(self.h)):
            ref block = self.h[i]
            blocks.append(block.to_cpu())
        out.h = blocks^
        out.n_layer = self.n_layer
        out.training = self.training
        out.ln_f = self.ln_f.to_cpu()
        if not self.tie_weights:
            ref head = self.lm_head.value()
            out.lm_head = head.to_cpu()
        return out^
