from .tensor import Tensor
from .shared.intarray import IntArray
from .named_parameter import NamedParameter
from .gather import Gather
from .layer_trait import LayerTrait
from .gpu.device import GPU
from .shared.mnemonics import DEFAULT_INDEX_DTYPE
from .shared.indexhelper import i, s
from .shared.panic import panic
from .shared import Reduction
from .weight_init import Weights

@fieldwise_init
struct Embedding[dtype: DType, index_dtype: DType = DEFAULT_INDEX_DTYPE](
    LayerTrait
):
    """Lookup table mapping integer indices to dense embedding vectors.

    Forward: out[i] = weight[indices[i]]  — a gather along axis=0
    Backward: ScatterAddTensor — sparse gradient update into weight rows

    Equivalent to PyTorch nn.Embedding but with:
    - Optional fused gather+sum (embedding_bag) via reduction=Reduction(1)
    - padding_idx support — that row's grad is zeroed after each update
    - Pretrained weight loading via from_pretrained()
    - Standard init methods matching nn.init.*
    """

    # Float-carrier convention: the LayerTrait __call__ takes Tensor[dtype]
    # holding integer-valued data and casts to index_dtype internally
    # (mirrors PositionalEmbedding.__call__). GPTEmbedding instead declares
    # InputDType = index_dtype and takes raw index tensors.
    comptime InputDType = Self.dtype

    var weight: Tensor[Self.dtype]  # (num_embeddings, embedding_dim)
    var num_embeddings: Int
    var embedding_dim: Int
    var padding_idx: Optional[Int]  # row frozen to zeros if set
    var max_norm: Optional[Float64]  # renormalise rows if set
    var norm_type: Float64  # L2 norm order (default 2.0)
    var training: Bool
    var reduction: Reduction

    def __init__(
        out self,
        num_embeddings: Int,
        embedding_dim: Int,
        padding_idx: Optional[Int] = None,
        max_norm: Optional[Float64] = None,
        norm_type: Float64 = 2.0,
        init_seed: Optional[Int] = None,
        init_method: String = "normal",
        unsafe_freeze: Bool = False,
        reduction: Reduction = Reduction(2),
    ):
        """
        Args:
            num_embeddings: Vocabulary size — number of rows in weight.
            embedding_dim:  Embedding dimension — number of cols in weight.
            padding_idx:    If set, weight[padding_idx] is always zeros
                            and receives no gradient.
            max_norm:       If set, renormalise each looked-up embedding
                            to have norm <= max_norm (in-place, after lookup).
            norm_type:      Order of norm for max_norm (default L2).
            init_seed:      Random seed.
            init_method:    Weight init strategy (see Weights.initialize):
                            "normal", "uniform", "xavier", "kaiming"/"he", "zero"
            unsafe_freeze:         If True, requires_grad=False — no gradient.
            reduction:      How to reduce gathered rows (NONE=2, SUM=1, MEAN=0).
                            Default NONE preserves standard gather behavior.
                            Set to Reduction(1) for bag-of-words sum.
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.training = True
        self.reduction = reduction

        var grad_required = not unsafe_freeze
        self.weight, _ = Weights[Self.dtype].initialize(
            num_embeddings,
            embedding_dim,
            init_seed=init_seed,
            init_method=init_method,
            requires_grad=grad_required,
            bias=False,
            bias_zero=True,
            device=None,
        )

        # Zero out padding row if set
        if padding_idx:
            self._zero_padding_row()

    # Forward
    def __call__(
        mut self,
        x: Tensor[Self.dtype],
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var xs_casted = x.to_dtype[Self.index_dtype]()
        return self.__call__(xs_casted, sync=sync)

    def __call__(
        mut self,
        indices: List[Int],
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        return self.__call__(IntArray(indices), sync=sync)

    def __call__(
        mut self,
        indices: IntArray,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Lookup embeddings for given indices.

        Reduction is determined by self.reduction:
            NONE (0) — standard gather, output shape (n, embedding_dim).
            SUM  (1) — fused gather+sum (embedding_bag), output (embedding_dim,).
            MEAN (2) — like SUM but divided by n.

        Args:
            indices:  Token ids to look up. Each must be in [0, num_embeddings).
            sync:     Whether to synchronize the GPU operation.

        Returns:
            (len(indices), embedding_dim) when reduction is NONE.
            (embedding_dim,) when reduction is SUM or MEAN.
        """
        var result: Tensor[Self.dtype]
        if self.training:
            result = Gather[Self.dtype, Self.index_dtype].forward[
                track_grad=True
            ](
                self.weight,
                indices,
                axis=0,
                reduction=self.reduction,
                padding_idx=self.padding_idx,
                sync=sync,
            )
        else:
            result = Gather[Self.dtype, Self.index_dtype].forward[
                track_grad=False
            ](
                self.weight,
                indices,
                axis=0,
                reduction=self.reduction,
                sync=sync,
            )

        # Apply max_norm renormalisation if set (in-place on result)
        if self.max_norm:
            self._apply_max_norm(result)

        return result^

    def __call__(
        mut self,
        indices: Tensor[Self.index_dtype],
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Lookup embeddings for indices given as an int64 Tensor.
        Enables batched transformer-style usage:
            embedding(token_ids_tensor)
        Preserves input shape: e.g. (B, T) → (B, T, embed_dim).
        """
        var result: Tensor[Self.dtype]
        if self.training:
            result = Gather[Self.dtype, Self.index_dtype].forward[
                track_grad=True
            ](
                self.weight,
                indices,
                axis=0,
                reduction=self.reduction,
                padding_idx=self.padding_idx,
                sync=sync,
            )
        else:
            result = Gather[Self.dtype, Self.index_dtype].forward[
                track_grad=False
            ](
                self.weight,
                indices,
                axis=0,
                reduction=self.reduction,
                sync=sync,
            )
        if self.max_norm:
            self._apply_max_norm(result)
        return result^

    # Utilities

    def _zero_padding_row(self):
        """Zero out the padding_idx row — called after init and after updates.
        """
        if self.padding_idx:
            var pad = self.padding_idx.value()
            self.weight.fill(Scalar[Self.dtype](0), i(pad), s())

    def _apply_max_norm(self, mut result: Tensor[Self.dtype]):
        """Renormalise looked-up rows to have norm <= max_norm in-place."""
        if self.max_norm:
            var mn = Scalar[Self.dtype](self.max_norm.value())
            var n = result.shape()[0]
            for k in range(n):
                var row = result[i(k), s()]
                var norm = row.norm(p=self.norm_type).item()
                if norm > mn:
                    result.fill(row * Scalar[Self.dtype](mn / norm), i(k), s())

    def unsafe_freeze(mut self):
        """Freeze all embeddings — no gradient computed."""
        self.weight.requires_grad = False

    def ununsafe_freeze(mut self):
        """Ununsafe_freeze embeddings — gradient computed."""
        self.weight.requires_grad = True
        if self.padding_idx:
            # padding row must stay frozen even when rest is unfrozen
            # handled in ScatterAddTensor by zeroing that row's grad
            pass

    # Pretrained loading

    @staticmethod
    def from_pretrained(
        weights: Tensor[Self.dtype],
        padding_idx: Optional[Int] = None,
        max_norm: Optional[Float64] = None,
        norm_type: Float64 = 2.0,
        unsafe_freeze: Bool = True,  # frozen by default — PyTorch convention
        reduction: Reduction = Reduction(2),
    ) -> Embedding[Self.dtype]:
        """Load pretrained embeddings (GloVe, fastText, word2vec etc.).

        Args:
            weights:     Pretrained weight tensor, shape (vocab_size, dim).
            padding_idx: Row to keep zeroed.
            max_norm:    Renormalisation threshold.
            norm_type:   Type of norm to use (default: 2.0).
            unsafe_freeze:      If True (default), embeddings are not updated during training.
                         Set False to fine-tune pretrained embeddings.
            reduction:   How to reduce gathered rows (NONE/SUM/MEAN).

        Returns:
            Embedding with weights copied from pretrained tensor.
        """
        var shape = weights.shape()
        if shape.rank() != 2:
            panic("Embedding.from_pretrained: weights must be 2D (vocab, dim)")

        var emb = Embedding[Self.dtype](
            num_embeddings=shape[0],
            embedding_dim=shape[1],
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            init_method="zero",  # allocate, then overwrite
            unsafe_freeze=unsafe_freeze,
            reduction=reduction,
        )
        # Copy pretrained weights in
        emb.weight.fill(weights, s(), s())
        if padding_idx:
            emb._zero_padding_row()
        return emb^

    # Layer protocol

    def parameters(
        ref self,
    ) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]()
        if self.weight.requires_grad:
            params.append(
                Pointer(to=self.weight)
                .unsafe_mut_cast[True]()
                .as_unsafe_any_origin()
            )
        return params^

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        var result = List[NamedParameter[Self.dtype]]()
        if self.weight.requires_grad:
            result.append(
                NamedParameter[Self.dtype](
                    prefix + "weight",
                    Pointer(to=self.weight)
                    .unsafe_mut_cast[True]()
                    .as_unsafe_any_origin()
                    .unsafe_origin_cast[MutUntrackedOrigin](),
                )
            )
        return result^

    def num_parameters(self) -> Int:
        return self.weight.numels()

    def train(mut self):
        self.training = True

    def eval(mut self):
        self.training = False

    def to_gpu(
        self,
        gpu: Optional[GPU] = None,
    ) raises -> Embedding[Self.dtype, Self.index_dtype]:
        var weight_gpu = self.weight.to_gpu(gpu=gpu, stop_grad=True)
        var out = self
        out.weight = weight_gpu^
        return out^

    def to_cpu(self) raises -> Embedding[Self.dtype, Self.index_dtype]:
        var weight_cpu = self.weight.to_cpu(stop_grad=True)
        var out = self
        out.weight = weight_cpu^
        return out^
