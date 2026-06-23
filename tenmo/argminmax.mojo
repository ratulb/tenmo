from .tensor import Tensor
from .intarray import IntArray
from .common_utils import panic
from .shapes import Shape
from std.utils.numerics import max_finite, min_finite
from .ndbuffer import NDBuffer
from std.sys import has_accelerator
from .kernels.argminmax_kernel import ArgMinMaxGpu
from tenmo.mnemonics import DEFAULT_INDEX_DTYPE


@fieldwise_init
struct ArgMinMaxReducer[dtype: DType, index_dtype: DType = DEFAULT_INDEX_DTYPE](
    ImplicitlyCopyable, RegisterPassable
):
    """
    Unified CPU + GPU argmin/argmax on NDBuffer.
    Returns an NDBuffer[Self.index_dtype] with the output shape.
    """

    @staticmethod
    def reduce[
        is_max: Bool,
        max_block_size: Int = 512,
    ](
        A: NDBuffer[Self.dtype],
        axis: Int,
        keepdims: Bool = False,
        sync: Bool = True,
    ) raises -> NDBuffer[Self.index_dtype]:
        var shape = A.shape
        var rank = shape.rank()
        var ax = axis if axis >= 0 else axis + rank
        if ax < 0 or ax >= rank:
            panic(
                "ArgMinMaxReducer: axis",
                String(axis),
                "out of range for rank",
                String(rank),
            )

        # Build output shape
        var out_axes = IntArray()
        for i in range(rank):
            if i == ax:
                if keepdims:
                    out_axes.append(1)
            else:
                out_axes.append(shape[i])
        var out_shape = Shape(out_axes)
        var reduced_volume = shape[ax]
        var total_output = out_shape.num_elements()

        comptime if has_accelerator():
            if A.is_on_gpu():
                return ArgMinMaxGpu[Self.dtype, Self.index_dtype]._gpu_reduce[
                    is_max, max_block_size
                ](
                    A,
                    ax,
                    keepdims,
                    out_shape,
                    total_output,
                    reduced_volume,
                    sync=sync,
                )

        return Self._cpu_reduce[is_max](A, ax, keepdims, out_shape)

    # ── CPU path ──────────────────────────────────────────────────────────────

    @staticmethod
    def _cpu_reduce[
        is_max: Bool,
    ](
        A: NDBuffer[Self.dtype],
        ax: Int,
        keepdims: Bool,
        out_shape: Shape,
    ) -> NDBuffer[Self.index_dtype]:
        var shape = A.shape
        var out = NDBuffer[Self.index_dtype].zeros(out_shape)

        for out_idx in out_shape:
            var best_val: Scalar[Self.dtype]
            var best_pos = 0

            comptime if is_max:
                best_val = min_finite[Self.dtype]()
            else:
                best_val = max_finite[Self.dtype]()

            for idx in range(shape[ax]):
                var full_idx = out_idx.insert(
                    ax, idx
                ) if not keepdims else out_idx.replace(ax, idx)
                var val = A[full_idx]

                comptime if is_max:
                    if val > best_val:
                        best_val = val
                        best_pos = idx
                else:
                    if val < best_val:
                        best_val = val
                        best_pos = idx

            if keepdims:
                var write_idx = out_idx.replace(ax, 0)
                out[write_idx] = Scalar[Self.index_dtype](best_pos)
            else:
                out[out_idx] = Scalar[Self.index_dtype](best_pos)

        return out^


# ── Public structs (thin wrappers) ────────────────────────────────────────────


struct Argmin[dtype: DType, index_dtype: DType = DEFAULT_INDEX_DTYPE]:
    @staticmethod
    def argmin(
        ndb: NDBuffer[Self.dtype],
        axis: Int = 0,
        keepdims: Bool = False,
    ) -> NDBuffer[Self.index_dtype]:
        try:
            return ArgMinMaxReducer[Self.dtype, Self.index_dtype].reduce[
                is_max=False
            ](ndb, axis, keepdims)
        except e:
            print(e)
            panic("Argmin failed at ArgMinMaxReducer reduce")
            # Unreachable
            return NDBuffer[Self.index_dtype].zeros(Shape())

    # Tensor convenience overload
    @staticmethod
    def argmin(
        tensor: Tensor[Self.dtype],
        axis: Int = 0,
        keepdims: Bool = False,
    ) -> Tensor[Self.index_dtype]:
        try:
            var result_ndb = ArgMinMaxReducer[
                Self.dtype, Self.index_dtype
            ].reduce[is_max=False](tensor.buffer, axis, keepdims)
            return Tensor[Self.index_dtype](result_ndb^, requires_grad=False)
        except e:
            print(e)
            panic("Argmin tensor failed at ArgMinMaxReducer reduce")
            # Unreachable
            return Tensor[Self.index_dtype].scalar(0)


struct Argmax[dtype: DType, index_dtype: DType = DEFAULT_INDEX_DTYPE]:
    @staticmethod
    def argmax(
        ndb: NDBuffer[Self.dtype],
        axis: Int = 0,
        keepdims: Bool = False,
    ) -> NDBuffer[Self.index_dtype]:
        try:
            return ArgMinMaxReducer[Self.dtype, Self.index_dtype].reduce[
                is_max=True
            ](ndb, axis, keepdims)
        except e:
            print(e)
            panic("Argmax failed at ArgMinMaxReducer reduce")
            # Unreachable
            return NDBuffer[Self.index_dtype].zeros(Shape())

    # Tensor convenience overload
    @staticmethod
    def argmax(
        tensor: Tensor[Self.dtype],
        axis: Int = 0,
        keepdims: Bool = False,
    ) -> Tensor[Self.index_dtype]:
        try:
            var result_ndb = ArgMinMaxReducer[
                Self.dtype, Self.index_dtype
            ].reduce[is_max=True](tensor.buffer, axis, keepdims)
            return Tensor[Self.index_dtype](result_ndb^, requires_grad=False)
        except e:
            print(e)
            panic("Argmax tensor failed at ArgMinMaxReducer reduce")
            # Unreachable
            return Tensor[Self.index_dtype].scalar(0)
