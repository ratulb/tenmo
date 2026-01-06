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
from forwards import Mean, Sqrt
from utilities import Utils
from indexhelper import IndexIterator
from filler import Filler


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
    var buffer: NDBuffer[Self.dtype]

    fn __init__(out self, shape: Shape, share: Bool = True):
        buffer = NDBuffer[Self.dtype](shape)
        self.buffer = buffer.share() if share else buffer^

    fn __init__(out self, var buffer: NDBuffer[Self.dtype], share: Bool = True):
        self.buffer = buffer.share() if share else buffer^

    fn __moveinit__(out self, deinit other: Self):
        self.buffer = other.buffer^

    fn __copyinit__(out self, other: Self):
        self.buffer = other.buffer.copy()

    @always_inline
    fn as_tensor(
        deinit self, requires_grad: Bool = False
    ) -> Tensor[Self.dtype]:
        return Tensor[Self.dtype](
            self^.buffer.contiguous(), requires_grad=requires_grad
        )

    @always_inline
    fn transpose(self, axes: IntArray) -> Gradbox[Self.dtype]:
        """Fused transpose and then contiguous."""
        shape = self.shape()
        normalized_axes = (
            Validator.validate_and_normalize_axes(
                shape, axes, ordered=False, fill_missing=True
            ) if len(axes)
            > 0 else IntArray.range(0, shape.rank()).reversed()
        )

        # Permute shape and strides
        var new_shape = shape.permute(normalized_axes)
        var new_strides = self.strides().permute(normalized_axes)
        var iterator = IndexIterator(
            Pointer(to=new_shape), Pointer(to=new_strides), start_offset=0
        )
        var buffer = Buffer[Self.dtype](self.numels())
        var index = 0
        for idx in iterator:
            buffer[index] = self.buffer.buffer[idx]
            index += 1
        nd_buffer = NDBuffer[Self.dtype](buffer^, new_shape^, None, offset=0)
        return Gradbox[Self.dtype](nd_buffer^, share=False)

    fn __abs__(self) -> Gradbox[Self.dtype]:
        var buffer = self.buffer.map[
            Utils[Self.dtype].abs_buffer, Utils[dtype].abs_scalar
        ]()
        var nd_buffer = NDBuffer[Self.dtype](buffer^, self.buffer.shape)
        return Gradbox[Self.dtype](nd_buffer^, share=False)

    fn sqrt(
        self,
        epsilon: Scalar[Self.dtype] = Scalar[Self.dtype](1e-12),
    ) -> Gradbox[Self.dtype]:
        return Sqrt[Self.dtype].forward(self, epsilon)

    fn norm(
        self,
        p: Float64 = 2.0,
        axis: Optional[Int] = None,
        keepdims: Bool = False,
    ) -> Gradbox[Self.dtype]:
        """Compute Lp norm. Supports L2 (p=2) only."""
        if p == 2.0:
            # L2 norm: sqrt(sum(x²))
            var squared = self.__mul__(self)
            var dim = IntArray(axis.value()) if axis else IntArray()
            var sum_sq = squared.sum(dim, keepdims=keepdims)
            return sum_sq.sqrt()
        else:
            panic("Only L2 norm (p=2) currently supported")
            return Gradbox[Self.dtype](Shape(), share=False)

    @staticmethod
    fn arange(
        *args: Scalar[Self.dtype],
    ) -> Gradbox[Self.dtype]:
        nd_buffer = NDBuffer[Self.dtype].arange(args)
        return Gradbox[Self.dtype](nd_buffer^, share=False)

    fn flatten(
        self, start_dim: Int = 0, end_dim: Optional[Int] = None
    ) -> Gradbox[Self.dtype]:
        flattened_buffer = self.buffer.flatten(start_dim, end_dim)
        return Gradbox[Self.dtype](flattened_buffer^, share=False)

    @always_inline
    fn squeeze(self, axes: List[Int] = []) -> Gradbox[Self.dtype]:
        return self.squeeze(IntArray(axes))

    @always_inline
    fn squeeze(self, axes: IntArray) -> Gradbox[Self.dtype]:
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
        var nd_buffer = NDBuffer[Self.dtype](buffer^, new_shape^)

        return Gradbox[Self.dtype](nd_buffer^, share=False)

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
        return Gradbox[Self.dtype](nd_buffer^, share=False)

    @always_inline
    fn permute(self, axes: IntArray) -> Gradbox[Self.dtype]:
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

        var buffer = self.buffer.contiguous_buffer()
        var nd_buffer = NDBuffer[Self.dtype](
            buffer=buffer^,
            shape=Shape(new_shape^),
            strides=Optional(Strides(new_strides^)),
            offset=0,
        )

        return Gradbox[Self.dtype](nd_buffer^, share=False)

    @staticmethod
    @always_inline
    fn full(
        shape: Shape, scalar: Scalar[Self.dtype], share: Bool = False
    ) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](NDBuffer.full(shape, scalar), share=share)

    @staticmethod
    @always_inline
    fn zeros(shape: Shape, share: Bool = False) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            NDBuffer.full(shape, Scalar[Self.dtype](0)), share=share
        )

    @staticmethod
    fn rand(
        shape: Shape,
        min: Scalar[Self.dtype] = 0,
        max: Scalar[Self.dtype] = 1,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
        share: Bool = False,
    ) -> Gradbox[Self.dtype]:
        if init_seed:
            seed(init_seed.value())
        else:
            seed()
        numels = shape.num_elements()
        buffer = Buffer[Self.dtype](numels)
        for i in range(numels):
            buffer[i] = random_float64(
                min.cast[Self.dtype.float64](), max.cast[DType.float64]()
            ).cast[Self.dtype]()

        return Gradbox[Self.dtype](
            NDBuffer[Self.dtype](buffer^, shape), share=share
        )

    @always_inline
    fn unshared(self) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](self.buffer.contiguous(), share=False)

    fn shared(self) -> Bool:
        return self.buffer.shared()

    @always_inline
    fn sum(
        self, axes: IntArray = IntArray(), keepdims: Bool = False
    ) -> Gradbox[Self.dtype]:
        var nd_buffer = self.buffer.sum(reduction_axes=axes, keepdims=keepdims)
        return Gradbox[Self.dtype](nd_buffer^, share=False)

    @always_inline
    fn mean(
        self, axes: IntArray = IntArray(), keepdims: Bool = False
    ) -> Gradbox[Self.dtype]:
        return Mean[Self.dtype].forward(self, axes=axes, keepdims=keepdims)

    @always_inline
    fn sum_over_broadcasted_axes(
        extended_grad: Gradbox[Self.dtype], target_shape: Shape
    ) -> Gradbox[Self.dtype]:
        var nd_buffer = extended_grad.buffer.sum_over_broadcasted_axes(
            target_shape
        )
        return Gradbox[Self.dtype](nd_buffer^, share=False)

    fn broadcast_to(
        self, target_shape: Shape, share: Bool = False
    ) -> Gradbox[Self.dtype]:
        if not ShapeBroadcaster.broadcastable(self.shape(), target_shape):
            panic(
                "Gradbox → broadcast_to: shape "
                + self.shape().__str__()
                + " not broadcastable to "
                + target_shape.__str__()
            )

        broadcasted_buffer = self.buffer.broadcast_to(target_shape)
        out = Gradbox[Self.dtype](broadcasted_buffer^, share=share)
        return out^

    fn __getitem__(self, *indices: Idx) -> Gradbox[Self.dtype]:
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
        ndb = NDBuffer[Self.dtype](
            shared_buffer^, shape=shape^, strides=strides^, offset=abs_offset
        )

        return Gradbox[Self.dtype](ndb^, share=False)

    @always_inline
    fn __getitem__(self, indices: List[Int]) -> Scalar[Self.dtype]:
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __getitem__(List): Scalar gradbox expects empty"
                " indices"
            )

        return self.buffer[indices]

    @always_inline
    fn __getitem__(self, indices: IntArray) -> Scalar[Self.dtype]:
        if self.rank() == 0 and indices.size() != 0:
            panic(
                "Gradbox → __getitem__(IntArray): Scalar gradbox expects empty"
                " indices"
            )

        return self.buffer[indices]

    @always_inline
    fn __getitem__(self, *indices: Int) -> Scalar[Self.dtype]:
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __getitem__(*Int): Scalar gradbox expects empty"
                " indices - please use __getitem__([])"
            )

        return self.buffer[indices]

    @always_inline
    fn __setitem__(self, indices: List[Int], value: Scalar[Self.dtype]):
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __setitem__(List[Int]): Scalar gradbox expects empty"
                " indices"
            )

        self.buffer[indices] = value

    @always_inline
    fn __setitem__(self, indices: IntArray, value: Scalar[Self.dtype]):
        if self.rank() == 0 and indices.size() != 0:
            panic(
                "Gradbox → __setitem__(IntArray): Scalar gradbox expects empty"
                " indices"
            )
        self.buffer[indices] = value

    @always_inline
    fn __setitem__(self, *indices: Int, value: Scalar[Self.dtype]):
        if self.rank() == 0 and len(indices) != 0:
            panic(
                "Gradbox → __setitem__(*Int): Scalar gradbox expects empty"
                " indices - please use __setitem__([], value)"
            )
        self.buffer[indices] = value

    @always_inline
    fn load[
        simdwidth: Int = simd_width_of[Self.dtype](), validated: Bool = False
    ](self, row: Int, col: Int) -> SIMD[Self.dtype, simdwidth]:
        """SIMD load of a row segment from a 2D Gradbox.

        Preconditions:
            - Gradbox must be 2D.
            - Columns must be contiguous (stride[1] == 1) for SIMD loads.
            - `col + simdwidth` must not exceed the number of columns.
        """
        return self.buffer.load[simdwidth, validated](row, col)

    @always_inline
    fn store[
        simdwidth: Int = simd_width_of[Self.dtype](), validated: Bool = False
    ](self, row: Int, col: Int, value: SIMD[Self.dtype, simdwidth]):
        """SIMD store of a row segment into a 2D Gradbox.

        Preconditions:
            - Gradbox must be 2D.
            - Columns must be contiguous for SIMD stores (stride[1] == 1).
            - Caller may set validated=True if these checks are already ensured.
        """
        self.buffer.store[simdwidth, validated](row, col, value)

    fn item(self) -> Scalar[Self.dtype]:
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

    @always_inline
    fn index_iterator(
        self,
    ) -> IndexIterator[
        origin_of(self.buffer.shape), origin_of(self.buffer.strides)
    ]:
        return self.buffer.index_iterator()

    fn __eq__(self, other: Gradbox[Self.dtype]) -> Bool:
        if self.shape() != other.shape():
            panic(
                "Gradbox → __eq__(other): shape mismatch",
                self.shape().__str__(),
                "≠",
                other.shape().__str__(),
            )
        return self.buffer.compare[Equal](other.buffer).buffer.all_true()

    fn __ne__(self, other: Gradbox[Self.dtype]) -> Bool:
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
        s += ", Type: " + Self.dtype.__str__()
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
    fn seed_grad(self, value: Scalar[Self.dtype]):
        self.buffer.fill(value)

    @always_inline
    fn seed_grad(self, with_tensor: Tensor[Self.dtype]):
        self.buffer.fill(with_tensor.buffer)

    @always_inline
    fn zero_grad(self):
        self.buffer.zero()

    fn fill(self, value: Scalar[Self.dtype], *indices: Idx):
        Filler[Self.dtype].fill(self.buffer, value, indices)

    fn fill(self, tensor: Tensor[Self.dtype], *indices: Idx):
        Filler[Self.dtype].fill(self.buffer, tensor.buffer, indices)

    fn fill(self, gradbox: Gradbox[Self.dtype], *indices: Idx):
        Filler[Self.dtype].fill(self.buffer, gradbox.buffer, indices)

    @always_inline
    fn clamp_in_place(
        self, lower_bound: Scalar[Self.dtype], upper_bound: Scalar[Self.dtype]
    ):
        self.buffer.clamp_in_place(lower_bound, upper_bound)

    fn __mul__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.scalar_ops[Multiply](scalar), share=False
        )

    fn __rmul__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        return self.__mul__(scalar)

    fn __add__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.scalar_ops[Add](scalar), share=False
        )

    fn __radd__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        return self.__add__(scalar)

    fn __sub__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.scalar_ops[Subtract](scalar), share=False
        )

    fn __rsub__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.scalar_ops[ReverseSubtract](scalar), share=False
        )

    fn __truediv__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        if scalar == Scalar[Self.dtype](0):
            panic("Gradbox → __truediv__(scalar): can not divide by zero")
        return Gradbox[Self.dtype](
            self.buffer.scalar_ops[Divide](scalar), share=False
        )

    fn __rtruediv__(self, scalar: Scalar[Self.dtype]) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.scalar_ops[ReverseDivide](scalar), share=False
        )

    fn __mul__(self, other: Self) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.arithmetic_ops[Multiply](other.buffer), share=False
        )

    fn __mul__(self, other: Tensor[Self.dtype]) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.arithmetic_ops[Multiply](other.buffer), share=False
        )

    fn matmul(
        A: Gradbox[Self.dtype], mut B: Tensor[dtype]
    ) -> Gradbox[Self.dtype]:
        return Matmul[Self.dtype].forward(A, B)

    fn __add__(self, other: Self) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.arithmetic_ops[Add](other.buffer), share=False
        )

    fn __add__(self, other: Tensor[Self.dtype]) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.arithmetic_ops[Add](other.buffer), share=False
        )

    fn __sub__(self, other: Self) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.arithmetic_ops[Subtract](other.buffer), share=False
        )

    fn __sub__(self, other: Tensor[Self.dtype]) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.arithmetic_ops[Subtract](other.buffer), share=False
        )

    fn __truediv__(self, other: Self) -> Gradbox[Self.dtype]:
        return Gradbox[Self.dtype](
            self.buffer.arithmetic_ops[Divide](other.buffer), share=False
        )

    fn __imul__(self, scalar: Scalar[Self.dtype]):
        self.buffer.inplace_scalar_ops[Multiply](scalar)

    fn __iadd__(self, scalar: Scalar[Self.dtype]):
        self.buffer.inplace_scalar_ops[Add](scalar)

    fn __isub__(self, scalar: Scalar[Self.dtype]):
        self.buffer.inplace_scalar_ops[Subtract](scalar)

    fn __itruediv__(self, scalar: Scalar[Self.dtype]):
        self.buffer.inplace_scalar_ops[Divide](scalar)

    @always_inline
    fn __imul__(self, incoming: Gradbox[Self.dtype]):
        # self.buffer.inplace_ops[Multiply](incoming.buffer)
        var multiplied = self.buffer.buffer * incoming.buffer.buffer
        var numels = self.buffer.buffer.size
        self.buffer.buffer.overwrite(multiplied, 0, numels)

    @always_inline
    fn __iadd__(self, incoming: Gradbox[Self.dtype]):
        # self.buffer.inplace_ops[Add](incoming.buffer)
        var added = self.buffer.buffer + incoming.buffer.buffer
        var numels = self.buffer.buffer.size
        self.buffer.buffer.overwrite(added, 0, numels)

    @always_inline
    fn __isub__(self, incoming: Gradbox[Self.dtype]):
        # self.buffer.inplace_ops[Subtract](incoming.buffer)
        var subtracted = self.buffer.buffer - incoming.buffer.buffer
        var numels = self.buffer.buffer.size
        self.buffer.buffer.overwrite(subtracted, 0, numels)

    @always_inline
    fn __itruediv__(self, incoming: Gradbox[Self.dtype]):
        self.buffer.inplace_ops[Divide](incoming.buffer)

    fn all_close[
        rtol: Scalar[Self.dtype] = 1e-5,
        atol: Scalar[Self.dtype] = 1e-8,
    ](self, other: Self) -> Bool:
        constrained[
            Self.dtype.is_floating_point(),
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
        rtol: Scalar[Self.dtype] = 1e-5,
        atol: Scalar[Self.dtype] = 1e-8,
    ](self, other: Tensor[Self.dtype]) -> Bool:
        constrained[
            Self.dtype.is_floating_point(),
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
    fn reshape(self) -> Gradbox[Self.dtype]:
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
    ) -> Gradbox[Self.dtype]:
        shape = new_shape if validated else Validator.validate_and_construct_new_shape(
            self.shape(), new_shape.intarray()
        )
        buffer = self.buffer.contiguous_buffer()
        nd_buffer = NDBuffer[Self.dtype](buffer^, shape^)

        return Gradbox[Self.dtype](nd_buffer^)

    fn __eq__(self, tensor: Tensor[Self.dtype]) -> Bool:
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
        print_gradbox_recursive[Self.dtype](
            UnsafePointer(to=self),
            empty,
            1,
            num_first=num_first,
            num_last=num_last,
        )

    fn __del__(deinit self):
        _ = self.buffer^


fn main():
    alias dtype = DType.float32
    g = Gradbox[dtype].arange(-10, 20)
    g.print()
    g.clamp_in_place(-5, 8)
    g.print()
    g = Gradbox[dtype].zeros(Shape(), share=False)
    g.print()
