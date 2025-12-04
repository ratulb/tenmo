### Mojo Tensor Gradbox
### Implement tensor library in mojo from first principles
from shapes import Shape
from common_utils_imports import *
from operators import *
from validators import Validator
from tenmo import Tensor
from intarray import IntArray
from ndbuffer import NDBuffer
from broadcasthelper import ShapeBroadcaster
from strides import Strides
from sys import simd_width_of
from matmul import Matmul
from random import seed, random_float64
from buffers import Buffer
from forwards import Mean
from utilities import Utils


struct Gradbox[dtype: DType](
    ImplicitlyCopyable
    & Movable
    & Sized
    & Stringable
    & Representable
    & Writable
    & EqualityComparable
    & Absable
):
    var buffer: NDBuffer[dtype]

    fn __init__(out self, shape: Shape, share: Bool = True):
        buffer = NDBuffer[dtype](shape)
        self.buffer = buffer.share() if share else buffer^

    fn __init__(out self, var buffer: NDBuffer[dtype], share: Bool = True):
        self.buffer = buffer.share() if share else buffer^

    fn __moveinit__(out self, deinit other: Self):
        self.buffer = other.buffer^

    fn __copyinit__(out self, other: Self):
        self.buffer = other.buffer.copy()

    @always_inline
    fn as_tensor(deinit self, requires_grad: Bool = False) -> Tensor[dtype]:
        return Tensor[dtype](
            self^.buffer.contiguous(), requires_grad=requires_grad
        )

    @always_inline
    fn transpose(self, axes: IntArray) -> Gradbox[dtype]:
        shape = self.shape()
        normalized_axes = (
            Validator.validate_and_normalize_axes(
                shape, axes, ordered=False, fill_missing=True
            ) if len(axes)
            > 0 else IntArray.range(0, shape.rank()).reversed()
        )

        # Permute shape and create default strides and permute
        var new_shape = shape.permute(normalized_axes)
        var new_strides = self.strides().permute(normalized_axes)

        buffer = self.buffer.contiguous_buffer()
        nd_buffer = NDBuffer[dtype](
            buffer^, Optional(new_shape^), Optional(new_strides^), offset=0
        )
        return Gradbox[dtype](nd_buffer^, share=False)

    fn __abs__(self) -> Gradbox[dtype]:
        var buffer = self.buffer.map[
            Utils[dtype].abs_buffer, Utils[dtype].abs_scalar
        ]()
        var nd_buffer = NDBuffer[dtype](buffer^, self.buffer.shape)
        return Gradbox[dtype](nd_buffer^, share=False)

    @staticmethod
    fn arange(
        *args: Scalar[dtype],
    ) -> Gradbox[dtype]:
        nd_buffer = NDBuffer[dtype].arange(args)
        return Gradbox[dtype](nd_buffer^, share=False)

    fn flatten(
        self, start_dim: Int = 0, end_dim: Optional[Int] = None
    ) -> Gradbox[dtype]:
        flattened_buffer = self.buffer.flatten(start_dim, end_dim)
        return Gradbox[dtype](flattened_buffer^, share=False)

    @always_inline
    fn squeeze(self, axes: List[Int] = []) -> Gradbox[dtype]:
        return self.squeeze(IntArray(axes))

    @always_inline
    fn squeeze(self, axes: IntArray) -> Gradbox[dtype]:
        shape = self.shape()
        rank = shape.rank()

        # Validate axes
        var validated_axes = IntArray()
        if len(axes) == 0:
            # Default: squeeze all dimensions with size 1
            for i in range(rank):
                if shape[i] == 1:
                    validated_axes.append(i)
        else:
            validated_axes = Validator.validate_and_normalize_axes(
                shape, axes, ordered=True
            )

            # ensure all are truly size-1 dimensions
            for ax in validated_axes:
                if shape[ax] != 1:
                    panic(
                        "Gradbox.squeeze(): cannot squeeze non-unit dimension "
                        + ax.__str__()
                        + " (size="
                        + shape[ax].__str__()
                        + ")"
                    )

        # construct new shape
        var new_dims = IntArray()
        for i in range(rank):
            if i not in validated_axes:
                new_dims.append(shape[i])

        var new_shape = Shape(new_dims)
        var buffer = self.buffer.contiguous_buffer()
        var nd_buffer = NDBuffer[dtype](buffer^, new_shape^)

        return Gradbox[dtype](nd_buffer^, share=False)

    fn unsqueeze(self, axes: List[Int]) -> Self:
        return self.unsqueeze(IntArray(axes))

    fn unsqueeze(self, axes: IntArray) -> Self:
        var rank = self.rank()
        var sorted_axes = axes.sorted()
        var new_rank = rank + sorted_axes.size()

        # Normalize negative axes
        for i in range(sorted_axes.size()):
            var ax = sorted_axes[i]
            if ax < 0:
                ax += new_rank
                # Validate
                if ax < 0 or ax > new_rank:
                    panic(
                        "Gradbox → unsqueeze: invalid axis",
                        sorted_axes[i].__str__(),
                    )
            sorted_axes[i] = ax

        # Build new shape
        shape = self.shape()
        var new_shape_dims = IntArray.with_capacity(new_rank)
        var src_i = 0
        for dst_i in range(new_rank):
            if dst_i in sorted_axes:
                new_shape_dims.append(1)
            else:
                new_shape_dims.append(shape[src_i])
                src_i += 1

        # Allocate new gradbox
        new_shape = Shape(new_shape_dims)

        # Copy elements preserving order
        # Unsqueeze doesn't change total number of elements.
        # So we can safely copy flattened contents.
        nd_buffer = self.buffer.contiguous(new_shape)
        return Gradbox[dtype](nd_buffer^, share=False)

    @always_inline
    fn permute(self, axes: IntArray) -> Gradbox[dtype]:
        # Validate rank / axis count
        var rank = self.rank()
        if len(axes) != rank:
            panic(
                "Gradbox → permute: number of axes (",
                len(axes).__str__(),
                ") must match rank (",
                rank.__str__(),
                ")",
            )

        # Validate permutation (no duplicates, indices in range)
        var seen = IntArray.with_capacity(rank)
        for axis in axes:
            if axis < 0 or axis >= rank:
                panic(
                    "Gradbox → permute: invalid axis index ",
                    axis.__str__(),
                    " for rank ",
                    rank.__str__(),
                )
            if axis in seen:
                panic(
                    "Gradbox → permute: duplicate axis in permutation ",
                    axis.__str__(),
                )
            seen.append(axis)

        # Build permuted shape and strides
        var new_shape = IntArray.with_capacity(rank)
        var new_strides = IntArray.with_capacity(rank)
        for axis in axes:
            new_shape.append(self.shape()[axis])
            new_strides.append(self.strides()[axis])

        # Create a contiguous backing buffer (value semantics) and construct a new NDBuffer
        var buffer = self.buffer.contiguous_buffer()
        var nd_buffer = NDBuffer[dtype](
            buffer=buffer^,
            shape=Shape(new_shape^),
            strides=Optional(Strides(new_strides^)),
            offset=0,
        )

        # Return a value-style Gradbox (share=False) — no ArcPointer, no autograd, no view-sharing
        return Gradbox[dtype](nd_buffer^, share=False)

    @staticmethod
    @always_inline
    fn full(
        shape: Shape, scalar: Scalar[dtype], share: Bool = False
    ) -> Gradbox[dtype]:
        return Gradbox[dtype](NDBuffer.full(shape, scalar), share=share)

    @staticmethod
    @always_inline
    fn zeros(shape: Shape, share: Bool = False) -> Gradbox[dtype]:
        return Gradbox[dtype](
            NDBuffer.full(shape, Scalar[dtype](0)), share=share
        )

    @staticmethod
    fn rand(
        shape: Shape,
        min: Scalar[dtype] = 0,
        max: Scalar[dtype] = 1,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
        share: Bool = False,
    ) -> Gradbox[dtype]:
        if init_seed:
            seed(init_seed.value())
        else:
            seed()
        numels = shape.num_elements()
        buffer = Buffer[dtype](numels)
        for i in range(numels):
            buffer[i] = random_float64(
                min.cast[DType.float64](), max.cast[DType.float64]()
            ).cast[dtype]()

        return Gradbox[dtype](NDBuffer[dtype](buffer^, shape), share=share)

    @always_inline
    fn unshared(self) -> Gradbox[dtype]:
        return Gradbox[dtype](self.buffer.contiguous(), share=False)

    fn shared(self) -> Bool:
        return self.buffer.shared()

    @always_inline
    fn sum(
        self, axes: IntArray = IntArray(), keepdims: Bool = False
    ) -> Gradbox[dtype]:
        var nd_buffer = self.buffer.sum(reduction_axes=axes, keepdims=keepdims)
        return Gradbox[dtype](nd_buffer^, share=False)

    @always_inline
    fn mean(
        self, axes: IntArray = IntArray(), keepdims: Bool = False
    ) -> Gradbox[dtype]:
        return Mean[dtype].forward(self, axes=axes, keepdims=keepdims)

    @always_inline
    fn sum_over_broadcasted_axes(
        extended_grad: Gradbox[dtype], target_shape: Shape
    ) -> Gradbox[dtype]:
        var nd_buffer = extended_grad.buffer.sum_over_broadcasted_axes(
            target_shape
        )
        return Gradbox[dtype](nd_buffer^, share=False)

    fn broadcast_to(
        self, target_shape: Shape, share: Bool = False
    ) -> Gradbox[dtype]:
        if not ShapeBroadcaster.broadcastable(self.shape(), target_shape):
            panic(
                "Gradbox → broadcast_to: shape "
                + self.shape().__str__()
                + " not broadcastable to "
                + target_shape.__str__()
            )

        broadcasted_buffer = self.buffer.broadcast_to(target_shape)
        out = Gradbox[dtype](broadcasted_buffer^, share=share)
        return out^

    fn __getitem__(self, *indices: Idx) -> Gradbox[dtype]:
        if not self.shared():
            panic(
                "Gradbox -> __getitem__(self, *indices: Idx): can not call on"
                " an unshared gradbox"
            )
        # Compute view metadata
        view_shape, view_strides, relative_offset = (
            Validator.validate_and_compute_advanced_indexing_metadata(
                self.shape(), self.strides(), indices
            )
        )

        # Handle scalar (rank-0) case
        is_scalar = len(view_shape) == 0
        shape = Shape() if is_scalar else view_shape
        strides = Strides() if is_scalar else view_strides
        abs_offset = self.offset() + relative_offset
        shared_buffer = self.buffer.buffer.copy()
        ndb = NDBuffer[dtype](
            shared_buffer^, shape=shape^, strides=strides^, offset=abs_offset
        )

        return Gradbox[dtype](ndb^, share=False)

    fn contiguous(self) -> Gradbox[dtype]:
        return Gradbox[dtype](self.buffer.contiguous(), share=False)

    @always_inline
    fn __getitem__(self, indices: List[Int]) -> Scalar[dtype]:
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __getitem__(List): Scalar gradbox expects empty"
                " indices"
            )

        return self.buffer[indices]

    @always_inline
    fn __getitem__(self, indices: IntArray) -> Scalar[dtype]:
        if self.rank() == 0 and indices.size() != 0:
            panic(
                "Gradbox → __getitem__(IntArray): Scalar gradbox expects empty"
                " indices"
            )

        return self.buffer[indices]

    @always_inline
    fn __getitem__(self, *indices: Int) -> Scalar[dtype]:
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __getitem__(*Int): Scalar gradbox expects empty"
                " indices - please use __getitem__([])"
            )

        return self.buffer[indices]

    @always_inline
    fn __setitem__(self, indices: List[Int], value: Scalar[dtype]):
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __setitem__(List[Int]): Scalar gradbox expects empty"
                " indices"
            )

        self.buffer[indices] = value

    @always_inline
    fn __setitem__(self, indices: IntArray, value: Scalar[dtype]):
        if self.rank() == 0 and indices.size() != 0:
            panic(
                "Gradbox → __setitem__(IntArray): Scalar gradbox expects empty"
                " indices"
            )
        self.buffer[indices] = value

    @always_inline
    fn __setitem__(self, *indices: Int, value: Scalar[dtype]):
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __setitem__(*Int): Scalar gradbox expects empty"
                " indices - please use __setitem__([], value)"
            )
        self.buffer[indices] = value

    @always_inline
    fn load[
        simdwidth: Int = simd_width_of[dtype](), validated: Bool = False
    ](self, row: Int, col: Int) -> SIMD[dtype, simdwidth]:
        """SIMD load of a row segment from a 2D Gradbox.

        Preconditions:
            - Gradbox must be 2D.
            - Columns must be contiguous (stride[1] == 1) for SIMD loads.
            - `col + simdwidth` must not exceed the number of columns.
        """
        return self.buffer.load[simdwidth, validated](row, col)

    @always_inline
    fn store[
        simdwidth: Int = simd_width_of[dtype](), validated: Bool = False
    ](self, row: Int, col: Int, value: SIMD[dtype, simdwidth]):
        """SIMD store of a row segment into a 2D Gradbox.

        Preconditions:
            - Gradbox must be 2D.
            - Columns must be contiguous for SIMD stores (stride[1] == 1).
            - Caller may set validated=True if these checks are already ensured.
        """
        self.buffer.store[simdwidth, validated](row, col, value)

    fn item(self) -> Scalar[dtype]:
        return self.buffer.item()

    @always_inline
    fn is_scalar(self) -> Bool:
        return self.buffer.is_scalar()

    @always_inline
    fn numels(self) -> Int:
        return self.buffer.numels()

    @always_inline
    fn __len__(self) -> Int:
        return self.buffer.numels()

    @always_inline
    fn is_contiguous(self) -> Bool:
        return self.strides().is_contiguous(self.shape())

    @always_inline
    fn rank(self) -> Int:
        return self.buffer.rank()

    @always_inline
    fn offset(self) -> Int:
        return self.buffer.offset

    @always_inline
    fn shape(ref self) -> ref [self.buffer.shape] Shape:
        return self.buffer.shape

    @always_inline
    fn strides(ref self) -> ref [self.buffer.strides] Strides:
        return self.buffer.strides

    fn __eq__(self, other: Gradbox[dtype]) -> Bool:
        if self.shape() != other.shape():
            panic(
                "Gradbox → __eq__(other): shape mismatch",
                self.shape().__str__(),
                "≠",
                other.shape().__str__(),
            )
        return self.buffer.compare[Equal](other.buffer).buffer.all_true()

    fn __ne__(self, other: Gradbox[dtype]) -> Bool:
        if self.shape() != other.shape():
            panic(
                "Gradbox → __ne__(other): shape mismatch",
                self.shape().__str__(),
                "≠",
                other.shape().__str__(),
            )
        return self.buffer.compare[NotEqual](other.buffer).buffer.all_true()

    fn __str__(self) -> String:
        rank = self.rank()
        s = String("[")
        if rank == 1:
            s += "1D Gradbox"
        elif rank == 2:
            s += "2D Gradbox"
        elif rank == 3:
            s += "3D Gradbox"
        elif rank == 4:
            s += "4D Gradbox"
        elif rank == 5:
            s += "5D Gradbox"
        else:
            s += "Gradbox"
        s += self.shape().__str__()
        s += ", Type: " + dtype.__str__()
        s += ", Shared : " + self.shared().__str__()
        s += ", Strides : " + self.strides().__str__()
        s += ", Offset : " + self.offset().__str__()
        s += "]"
        return s

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    @always_inline
    fn seed_grad(self, value: Scalar[dtype]):
        self.buffer.fill(value)

    @always_inline
    fn seed_grad(self, with_tensor: Tensor[dtype]):
        self.buffer.fill(with_tensor.buffer)

    @always_inline
    fn zero_grad(self):
        self.buffer.zero()

    fn __mul__(self, scalar: Scalar[dtype]) -> Gradbox[dtype]:
        return Gradbox[dtype](
            self.buffer.scalar_ops[Multiply](scalar), share=False
        )

    fn __rmul__(self, scalar: Scalar[dtype]) -> Gradbox[dtype]:
        return self.__mul__(scalar)

    fn __add__(self, scalar: Scalar[dtype]) -> Gradbox[dtype]:
        return Gradbox[dtype](self.buffer.scalar_ops[Add](scalar), share=False)

    fn __radd__(self, scalar: Scalar[dtype]) -> Gradbox[dtype]:
        return self.__add__(scalar)

    fn __sub__(self, scalar: Scalar[dtype]) -> Gradbox[dtype]:
        return Gradbox[dtype](
            self.buffer.scalar_ops[Subtract](scalar), share=False
        )

    fn __rsub__(self, scalar: Scalar[dtype]) -> Gradbox[dtype]:
        return Gradbox[dtype](
            self.buffer.scalar_ops[ReverseSubtract](scalar), share=False
        )

    fn __truediv__(self, scalar: Scalar[dtype]) -> Gradbox[dtype]:
        if scalar == Scalar[dtype](0):
            panic("Gradbox → __truediv__(scalar): can not divide by zero")
        return Gradbox[dtype](
            self.buffer.scalar_ops[Divide](scalar), share=False
        )

    fn __rtruediv__(self, scalar: Scalar[dtype]) -> Gradbox[dtype]:
        return Gradbox[dtype](
            self.buffer.scalar_ops[ReverseDivide](scalar), share=False
        )

    fn __mul__(self, other: Self) -> Gradbox[dtype]:
        return Gradbox[dtype](
            self.buffer.arithmetic_ops[Multiply](other.buffer), share=False
        )

    fn __mul__(self, other: Tensor[dtype]) -> Gradbox[dtype]:
        return Gradbox[dtype](
            self.buffer.arithmetic_ops[Multiply](other.buffer), share=False
        )

    fn matmul(A: Gradbox[dtype], B: Tensor[dtype]) -> Gradbox[dtype]:
        return Matmul[dtype].forward(A, B)

    fn __add__(self, other: Self) -> Gradbox[dtype]:
        return Gradbox[dtype](
            self.buffer.arithmetic_ops[Add](other.buffer), share=False
        )

    fn __sub__(self, other: Self) -> Gradbox[dtype]:
        return Gradbox[dtype](
            self.buffer.arithmetic_ops[Subtract](other.buffer), share=False
        )

    fn __sub__(self, other: Tensor[dtype]) -> Gradbox[dtype]:
        return Gradbox[dtype](
            self.buffer.arithmetic_ops[Subtract](other.buffer), share=False
        )

    fn __truediv__(self, other: Self) -> Gradbox[dtype]:
        return Gradbox[dtype](
            self.buffer.arithmetic_ops[Divide](other.buffer), share=False
        )

    fn __imul__(self, scalar: Scalar[dtype]):
        self.buffer.inplace_scalar_ops[Multiply](scalar)

    fn __iadd__(self, scalar: Scalar[dtype]):
        self.buffer.inplace_scalar_ops[Add](scalar)

    fn __isub__(self, scalar: Scalar[dtype]):
        self.buffer.inplace_scalar_ops[Subtract](scalar)

    fn __itruediv__(self, scalar: Scalar[dtype]):
        self.buffer.inplace_scalar_ops[Divide](scalar)

    @always_inline
    fn __imul__(self, incoming: Gradbox[dtype]):
        # self.buffer.inplace_ops[Multiply](incoming.buffer)
        var multiplied = self.buffer.buffer * incoming.buffer.buffer
        var numels = self.buffer.buffer.size
        self.buffer.buffer.overwrite(multiplied, 0, numels)

    @always_inline
    fn __iadd__(self, incoming: Gradbox[dtype]):
        # self.buffer.inplace_ops[Add](incoming.buffer)
        var added = self.buffer.buffer + incoming.buffer.buffer
        var numels = self.buffer.buffer.size
        self.buffer.buffer.overwrite(added, 0, numels)

    @always_inline
    fn __isub__(self, incoming: Gradbox[dtype]):
        # self.buffer.inplace_ops[Subtract](incoming.buffer)
        var subtracted = self.buffer.buffer - incoming.buffer.buffer
        var numels = self.buffer.buffer.size
        self.buffer.buffer.overwrite(subtracted, 0, numels)

    @always_inline
    fn __itruediv__(self, incoming: Gradbox[dtype]):
        self.buffer.inplace_ops[Divide](incoming.buffer)

    fn all_close[
        rtol: Scalar[dtype] = 1e-5,
        atol: Scalar[dtype] = 1e-8,
    ](self, other: Self) -> Bool:
        constrained[
            dtype.is_floating_point(),
            "Gradbox → all_close(Self): is for floating point data types only",
        ]()
        if self.shape() != other.shape():
            panic(
                "Gradbox → all_close(Self): expects same shaped gradboxes: "
                + self.shape().__str__()
                + ", "
                + other.shape().__str__()
            )

        return self.buffer.all_close[rtol=rtol, atol=atol](other.buffer)

    fn all_close[
        rtol: Scalar[dtype] = 1e-5,
        atol: Scalar[dtype] = 1e-8,
    ](self, other: Tensor[dtype]) -> Bool:
        constrained[
            dtype.is_floating_point(),
            (
                "Gradbox → all_close(Tensor): is for floating point data types"
                " only"
            ),
        ]()
        if self.shape() != other.shape():
            panic(
                "Gradbox → all_close(Tensor): expects same shaped tensor: "
                + self.shape().__str__()
                + ", "
                + other.shape().__str__()
            )

        return self.buffer.all_close[rtol=rtol, atol=atol](other.buffer)

    @always_inline
    fn reshape(self) -> Gradbox[dtype]:
        if self.numels() != 1:
            panic(
                "Gradbox → reshape: only gradbox with single element can be"
                " reshaped to scalar gradbox"
            )
        return self.reshape(Shape(), validated=True)

    @always_inline
    fn reshape(
        self,
        new_shape: Shape,
        validated: Bool = False,
    ) -> Gradbox[dtype]:
        shape = new_shape if validated else Validator.validate_and_construct_new_shape(
            self.shape(), new_shape.intarray()
        )
        buffer = self.buffer.contiguous_buffer()
        nd_buffer = NDBuffer[dtype](buffer^, shape^)

        return Gradbox[dtype](nd_buffer^)

    fn __eq__(self, tensor: Tensor[dtype]) -> Bool:
        if self.shape() != tensor.shape():
            panic(
                "Gradbox __eq__(tensor) → dimension mismatch:",
                self.shape().__str__(),
                ",",
                tensor.shape().__str__(),
            )
        return self.buffer.compare[Equal](tensor.buffer).buffer.all_true()

    fn print(self, num_first: Int = 10, num_last: Int = 10):
        print(
            "\n",
            self.__str__(),
            end="\n",
        )
        empty = List[Int]()
        print_gradbox_recursive[dtype](
            UnsafePointer(to=self),
            empty,
            1,
            num_first=num_first,
            num_last=num_last,
        )

    fn __del__(deinit self):
        _ = self.buffer^


from common_utils import s, il


fn main() raises:
    alias dtype = DType.float32
    gb = Gradbox[dtype].arange(12).reshape(Shape([3, 4]))
    r = gb[il(2), s()]
    gb.print()

    print(gb.buffer)
    print(r.buffer)

    r.print()

    print(gb.buffer is r.buffer)

    a = Tensor.arange(12).reshape(3, 4)
    a_slice = a[il(1), s()]

    a_slice.print()

    gg = Gradbox[dtype](Shape(3, 4), share=True)
    gg.print()
    gg_copied = gg.copy()
    gg_copied.print()

    x = Tensor.arange(12, requires_grad=True)
    x.print()
    y = x.copy()
    y.print()

    x.gradients()[].print()
    y.gradients()[].print()


from testing import assert_true
