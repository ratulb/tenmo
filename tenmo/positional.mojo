"""PositionalEmbedding — learned position table (wpe) for the GPT stack.

GPT-2-style learned positional embeddings: a `(n_ctx, n_embd)` table keyed by
position index, added elementwise to the token embeddings `wte`. This is a
thin LayerTrait composition over the existing, heavily-tested `Embedding`,
so it inherits the fast gather forward and the (now-parallel) scatter-add
backward with no new primitive.

Forward:  wpe(positions)  ->  (..., n_embd)  via a gather of rows by
          position id (positions are 0..n_ctx-1, broadcast over batch).
The GPT block forms `x = wte(tokens) + wpe(positions)`.
"""

from .tensor import Tensor
from .shared.shapes import Shape
from .shared.intarray import IntArray
from .named_parameter import NamedParameter
from .embedding import Embedding
from .layer_trait import LayerTrait
from .gpu.device import GPU
from .shared.mnemonics import DEFAULT_INDEX_DTYPE
from .shared.indexhelper import i, s
from .shared.panic import panic


@fieldwise_init
struct PositionalEmbedding[
    dtype: DType, index_dtype: DType = DEFAULT_INDEX_DTYPE
](LayerTrait):
    """Learned positional embedding table `(n_ctx, n_embd)`.

    One row per position; `wpe[position_id]` is added to the token embedding
    `wte[token_id]` at that position. Position ids are `0..T-1` broadcast
    across the batch, shared with every sequence.

    Implemented by composing `Embedding(num_embeddings=n_ctx,
    embedding_dim=n_embd)`; all LayerTrait methods forward to it, so it drops
    into `Sequential`/`ModuleWrapper` and `optim.mojo` unchanged.
    """

    # Float-carrier convention (see __call__ below): Tensor[dtype] holding
    # integer-valued data, cast to index_dtype internally.
    comptime InputDType = Self.dtype

    var emb: Embedding[Self.dtype, Self.index_dtype]
    var n_ctx: Int
    var n_embd: Int
    var training: Bool

    def __init__(
        out self,
        n_ctx: Int,
        n_embd: Int,
        init_seed: Optional[Int] = None,
        init_method: String = "xavier",
        unsafe_freeze: Bool = False,
    ):
        """Create a positional embedding table.

        Args:
            n_ctx:      Number of positions (rows). Also the max context len.
            n_embd:     Embedding dimension (cols).
            init_seed:  Random seed for weight init.
            init_method: Weight init strategy (passed to Embedding).
            unsafe_freeze: If True, requires_grad=False — no gradient.
        """
        if n_ctx < 1 or n_embd < 1:
            panic(
                "PositionalEmbedding: n_ctx and n_embd must be >= 1, got",
                String(n_ctx),
                String(n_embd),
            )
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.training = True
        self.emb = Embedding[Self.dtype, Self.index_dtype](
            num_embeddings=n_ctx,
            embedding_dim=n_embd,
            init_seed=init_seed,
            init_method=init_method,
            unsafe_freeze=unsafe_freeze,
        )

    def __call__(
        mut self,
        positions: Tensor[Self.dtype],
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Satisfy the LayerTrait `__call__` signature by casting to index dtype.

        Accepts a float tensor of position ids (mirrors Embedding) and converts
        to the index dtype before the gather.
        """
        var casted = positions.to_dtype[Self.index_dtype]()
        return self.__call__(casted, sync=sync)

    def __call__(
        mut self,
        positions: Tensor[Self.index_dtype],
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Look up position vectors for the given position-id tensor.

        Args:
            positions: Position ids in [0, n_ctx). A `(T,)` tensor gives a
                `(T, n_embd)` output; a `(B, T)` tensor gives `(B, T, n_embd)`.
            sync: Whether to synchronize the GPU operation.

        Returns:
            The looked-up position embeddings.
        """
        return self.emb.__call__(positions, sync=sync)

    def __call__(
        mut self,
        positions: IntArray,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Look up position vectors for the given position-id array."""
        return self.emb.__call__(positions, sync=sync)

    def __call__(
        mut self,
        positions: List[Int],
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Look up position vectors for the given position-id list."""
        return self.emb.__call__(positions, sync=sync)

    # Position-id builder

    @staticmethod
    def position_ids(
        n_ctx: Int,
        B: Int,
        T: Int,
    ) raises -> Tensor[Self.index_dtype]:
        """Build a `(B, T)` tensor of position ids `0..T-1` broadcast over B.

        Each row of the batch holds the same sequence positions 0..T-1, and
        `wpe` is shared across the batch.

        Args:
            n_ctx: Table size; guards `T`, raises if `T > n_ctx`.
            B:     Batch size.
            T:     Sequence length (must be <= n_ctx).

        Returns:
            A `(B, T)` int tensor; row b is `[0, 1, ..., T-1]`.
        """
        if T > n_ctx:
            raise Error(
                "PositionalEmbedding.position_ids: T "
                + String(T)
                + " exceeds n_ctx "
                + String(n_ctx)
            )
        # Materialize a CONTIGUOUS (B, T) tensor: every row is 0..T-1.
        # We must NOT use a stride-0 broadcast view — the Gather index read
        # (`Tensor.get`, flat) only handles physically-contiguous index
        # tensors, so a broadcast view (numels = B*T but a T-element backing
        # buffer) would read out of bounds on rows after the first.
        var base = Tensor[Self.index_dtype].arange(
            Scalar[Self.index_dtype](0), Scalar[Self.index_dtype](T)
        )  # (T,)
        var ids = Tensor[Self.index_dtype].zeros(Shape(B, T))
        for b in range(B):
            ids.fill(base, i(b), s())
        return ids

    # Weight accessor

    @always_inline
    def weight(ref self) -> ref[self.emb.weight] Tensor[Self.dtype]:
        """The `(n_ctx, n_embd)` position table (forwards to the inner Embedding).
        """
        return self.emb.weight

    # Layer protocol (forwarded to the inner Embedding)

    def parameters(
        ref self,
    ) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        return self.emb.parameters()

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        return self.emb.named_parameters(prefix)

    def num_parameters(self) -> Int:
        return self.emb.num_parameters()

    def train(mut self):
        self.training = True
        self.emb.train()

    def eval(mut self):
        self.training = False
        self.emb.eval()

    def to_gpu(
        self,
        gpu: Optional[GPU] = None,
    ) raises -> PositionalEmbedding[Self.dtype, Self.index_dtype]:
        var out = self
        out.emb = self.emb.to_gpu(gpu=gpu)
        return out^

    def to_cpu(
        self,
    ) raises -> PositionalEmbedding[Self.dtype, Self.index_dtype]:
        var out = self
        out.emb = self.emb.to_cpu()
        return out^
