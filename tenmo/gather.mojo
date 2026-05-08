from .tensor import Tensor
from .gradbox import Gradbox
from .intarray import IntArray
from .ancestry import Ancestor
from .mnemonics import ScatterAddTensor, ZeroGrad
from .backpropagation import IntArrayArg, BACKWARD_GATHER
from .common_utils import panic


@fieldwise_init
struct GatherArg(ArgumentType):
    """Carries the axis and normalized indices for a gather operation.
    Used by both GatherBackward (to scatter grad) and the engine's
    ScatterAddTensor branch (to know which rows to update).
    """

    var axis: Int
    var indices: IntArray


@fieldwise_init
struct GatherBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        # ── Indices live on the output node's own BackwardFnArg ───────────────
        # Registered during forward — array[0]=axis, array[1:]=indices
        var parent = output.ancestry().get(0)
        ref incoming_grad = output.gradbox[]
        # incoming_grad is already (n_indices, hidden_size) — exactly what
        # ScatterAddTensor needs. No new allocation at all.

        return [
            (
                parent^,
                incoming_grad,
                ScatterAddTensor,
            ),  # sparse — engine uses indices from output node
            (output, incoming_grad, ZeroGrad),
        ]


@fieldwise_init
struct Gather[dtype: DType](Copyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        indices: IntArray,
        axis: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Gather slices along `axis` at the given indices.

        Always copies data into a fresh contiguous output tensor.
        Output has no grad connection to input — requires_grad is always False.

        Args:
            indices: Indices to gather along `axis`. May contain negative values,
                     which are normalized to axis_dim + index.
            axis:    Axis to gather along. May be negative. Defaults to 0.

        Returns:
            A new contiguous tensor with shape identical to self except
            axis dimension is replaced by len(indices).

        Panics:
            - axis out of bounds
            - indices is empty
            - any index out of bounds after normalization
        """
        var rank = self.shape().rank()

        var ax = axis if axis >= 0 else axis + rank
        if ax < 0 or ax >= rank:
            panic(
                "gather: axis ",
                String(axis),
                " out of bounds for rank ",
                String(rank),
            )

        if len(indices) == 0:
            panic("gather: indices cannot be empty")
        var ax_dim = self.shape()[ax]
        var normalized = IntArray.with_capacity(len(indices))

        for k in range(len(indices)):
            var idx = indices[k]
            if idx < 0:
                idx += ax_dim
            if idx < 0 or idx >= ax_dim:
                panic(
                    "gather: index ",
                    String(indices[k]),
                    " out of bounds for axis ",
                    String(ax),
                    " with size ",
                    String(ax_dim),
                )
            normalized.append(idx)

        var out = Self._gather_copy(self, ax, normalized)
        # print("Normalized: ", normalized)
        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_GATHER, GatherArg(ax, normalized)
                )
                out.add_ancestry(backwardFnArg^, self)

        return out^

    @staticmethod
    fn _gather_copy(
        self: Tensor[Self.dtype],
        ax: Int,
        normalized: IntArray,
    ) -> Tensor[Self.dtype]:
        """Copy-based gather — always produces a contiguous output tensor.

        Args:
            ax:         Normalized (non-negative) axis to gather along.
            normalized: Validated, normalized gather indices.

        Returns:
            Fresh contiguous tensor. Shape is identical to self except
            axis `ax` has dimension len(normalized).
        """
        var rank = self.shape().rank()

        var out_shape_arr = IntArray.with_capacity(rank)
        for d in range(rank):
            out_shape_arr.append(
                len(normalized) if d == ax else self.shape()[d]
            )

        var result = Tensor[Self.dtype].zeros(
            Shape(out_shape_arr), requires_grad=False
        )
        var total = result.shape().num_elements()

        for flat in range(total):
            var coords = IntArray.with_capacity(rank)
            var rem = flat
            for d in range(rank - 1, -1, -1):
                coords.prepend(rem % result.shape()[d])
                rem //= result.shape()[d]

            var src_idx = normalized[coords[ax]]

            var src_offset = self.offset()
            for d in range(rank):
                src_offset += (
                    src_idx if d == ax else coords[d]
                ) * self.strides()[d]

            result.set(flat, self.get(src_offset))

        return result^
