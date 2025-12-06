from shapes import Shape
from strides import Strides
from buffers import Buffer
from intarray import IntArray
from indexhelper import IndexCalculator
from broadcasthelper import ShapeBroadcaster
from common_utils import panic
from memory import memcpy
from collections import Set
from sys import simd_width_of
from operators import (
    Multiply,
    Add,
    Subtract,
    ReverseSubtract,
    Divide,
    Overwrite,
    ReverseDivide,
    Equal,
    NotEqual,
    LessThan,
    LessThanEqual,
    GreaterThan,
    GreaterThanEqual,
)


struct NDBuffer[dtype: DType](
    Copyable
    & Movable
    & EqualityComparable
    & Stringable
    & Representable
    & Writable
):
    var shape: Shape
    var strides: Strides
    var offset: Int
    var buffer: Buffer[dtype]
    var _contiguous: Bool

    fn __init__(out self, *values: Scalar[dtype]):
        buffer = Buffer[dtype](len(values))
        for i in range(len(values)):
            buffer[i] = values[i]
        self = NDBuffer[dtype](buffer^)

    fn __init__(
        out self,
        var buffer: Buffer[dtype],
        shape: Optional[Shape] = None,
        strides: Optional[Strides] = None,
        offset: Int = 0,
    ):
        if buffer.size == 0:
            panic(
                "NDBuffer →__init__(Buffer, ...): zero sized buffer not allowed"
            )
        _shape = shape.or_else(Shape(buffer.size))
        self.shape = _shape.copy()
        self.buffer = buffer^
        self.strides = strides.or_else(Strides.default(_shape))
        self.offset = offset
        self._contiguous = False
        self._contiguous = self.is_contiguous()

    fn __init__(
        out self,
        shape: Shape,
        strides: Optional[Strides] = None,
        offset: Int = 0,
    ):
        self.buffer = Buffer[dtype](shape.num_elements())
        self.shape = shape
        self.strides = strides.or_else(Strides.default(shape))
        self.offset = offset
        self._contiguous = False
        self._contiguous = self.is_contiguous()

    fn __moveinit__(out self, deinit other: Self):
        self.buffer = other.buffer^
        self.shape = other.shape^
        self.strides = other.strides^
        self.offset = other.offset
        self._contiguous = other._contiguous

    fn __copyinit__(out self, other: Self):
        """Copy NDBuffer - buffer handles ref counting automatically!."""
        self.buffer = (
            other.buffer.copy()
        )  # Buffer copy handles shared/unshared!
        self.shape = other.shape.copy()
        self.strides = other.strides.copy()
        self.offset = other.offset
        self._contiguous = other._contiguous

    @staticmethod
    @always_inline
    fn zeros(shape: Shape) -> NDBuffer[dtype]:
        var buffer = Buffer[dtype].zeros(shape.num_elements())
        return NDBuffer[dtype](buffer^, shape)

    @staticmethod
    @always_inline
    fn full(shape: Shape, scalar: Scalar[dtype]) -> NDBuffer[dtype]:
        var buffer = Buffer[dtype].full(scalar, shape.num_elements())
        return NDBuffer[dtype](buffer^, shape)

    @staticmethod
    @always_inline
    fn arange(
        args: VariadicList[Scalar[dtype]],
    ) -> NDBuffer[dtype]:
        var buffer = Buffer[dtype].arange(args)
        var shape = Shape(buffer.size)
        return NDBuffer[dtype](buffer^, shape^)

    @staticmethod
    @always_inline
    fn linspace(
        start: Scalar[dtype],
        end: Scalar[dtype],
        steps: Int,
    ) -> NDBuffer[dtype]:
        var buffer = Buffer[dtype].linspace(start, end, steps)
        var shape = Shape(buffer.size)
        return NDBuffer[dtype](buffer^, shape^)

    @always_inline
    fn is_contiguous(self) -> Bool:
        return self.strides.is_contiguous(self.shape)

    @always_inline
    fn size(self) -> Int:
        return self.buffer.size

    @always_inline
    fn __getitem__(self, indices: IntArray) -> Scalar[dtype]:
        index = IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )
        return self.buffer[index]

    @always_inline
    fn __setitem__(self, indices: IntArray, value: Scalar[dtype]):
        index = IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )
        self.buffer[index] = value

    @always_inline
    fn __getitem__(self, indices: List[Int]) -> Scalar[dtype]:
        index = IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )
        return self.buffer[index]

    @always_inline
    fn __setitem__(self, indices: List[Int], value: Scalar[dtype]):
        index = IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )
        self.buffer[index] = value

    @always_inline
    fn __getitem__(self, indices: VariadicList[Int]) -> Scalar[dtype]:
        index = IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )
        return self.buffer[index]

    @always_inline
    fn __setitem__(self, indices: VariadicList[Int], value: Scalar[dtype]):
        index = IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )
        self.buffer[index] = value

    @always_inline
    fn item(self) -> Scalar[dtype]:
        if self.shape != Shape(1) and self.shape != Shape():
            panic(
                "NDBuffer → item(self): only valid for zero dim"
                " buffer/singleton, got shape: "
                + self.shape.__str__()
            )
        if self.shape == Shape(1):
            return self[IntArray(0)]
        else:
            return self[IntArray()]

    @always_inline
    fn load[
        simdwidth: Int = simd_width_of[dtype](), validated: Bool = False
    ](self, row: Int, col: Int) -> SIMD[dtype, simdwidth]:
        """SIMD load of a row segment from a 2D NDBuffer."""

        constrained[
            simdwidth.is_power_of_two(),
            "NDBuffer → load: SIMD width must be a power of 2",
        ]()

        @parameter
        if not validated:
            var rank = self.rank()
            ref shape = self.shape

            if rank != 2:
                panic("NDBuffer → load: Only 2D buffers are supported.")

            if (
                row < 0
                or row >= shape[0]
                or col < 0
                or col + simdwidth > shape[1]
            ):
                panic(
                    "NDBuffer → load: Out-of-bounds access. "
                    + "Attempted row "
                    + row.__str__()
                    + ", col range ["
                    + col.__str__()
                    + ", "
                    + (col + simdwidth).__str__()
                    + ") "
                    + "for shape "
                    + shape.__str__()
                    + "."
                )

            if simdwidth > 1 and self.strides[1] != 1:
                panic(
                    "NDBuffer → SIMD load requires contiguous column access. "
                    + "Expected stride[1] == 1 but got "
                    + self.strides[1].__str__()
                    + ". "
                    + "Use .contiguous() or scalar loads."
                )

        # Direct field access - no copy!
        var addr = row * self.strides[0] + col * self.strides[1] + self.offset
        return self.buffer.load[simdwidth](addr)

    @always_inline
    fn store[
        simdwidth: Int = simd_width_of[dtype](), validated: Bool = False
    ](self, row: Int, col: Int, value: SIMD[dtype, simdwidth]):
        """SIMD store of a row segment into a 2D NDBuffer."""

        constrained[
            simdwidth.is_power_of_two(),
            "NDBuffer → store: SIMD width must be a power of 2",
        ]()

        @parameter
        if not validated:
            var rank = self.rank()
            ref shape = self.shape

            if rank != 2:
                panic("NDBuffer → store: Only 2D buffers are supported.")

            if (
                row < 0
                or row >= shape[0]
                or col < 0
                or col + simdwidth > shape[1]
            ):
                panic(
                    "NDBuffer → store: Out-of-bounds access. "
                    + "Attempted row "
                    + row.__str__()
                    + ", col range ["
                    + col.__str__()
                    + ", "
                    + (col + simdwidth).__str__()
                    + ") "
                    + "for shape "
                    + shape.__str__()
                    + "."
                )

            if simdwidth > 1 and self.strides[1] != 1:
                panic(
                    "NDBuffer → SIMD store requires contiguous column access. "
                    + "Expected stride[1] == 1 but got "
                    + self.strides[1].__str__()
                    + ". "
                    + "Use .contiguous() or scalar stores."
                )

        # Direct field access - no copy!
        var addr = row * self.strides[0] + col * self.strides[1] + self.offset
        self.buffer.store[simdwidth](addr, value)

    fn __str__(self) -> String:
        s = String("NDBuffer [")
        s += "Shape: " + self.shape.__str__()
        s += ", Type: " + dtype.__str__()
        s += ", Shared : " + self.shared().__str__()
        s += ", Strides : " + self.strides.__str__()
        s += ", Offset : " + self.offset.__str__()
        s += ", Contiguous : " + self.is_contiguous().__str__()
        s += ", Buffer size : " + self.size().__str__()
        s += "]"
        return s

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    fn data_buffer(ref self) -> ref [self.buffer] Buffer[dtype]:
        return self.buffer

    @always_inline
    fn is_scalar(self) -> Bool:
        return self.numels() == 1 and self.shape == Shape()

    @always_inline
    fn numels(self) -> Int:
        return self.shape.num_elements()

    @always_inline
    fn rank(self) -> Int:
        return self.shape.rank()

    @always_inline
    fn max_index(self) -> Int:
        var max_index = self.offset
        for i in range(self.shape.rank()):
            max_index += (self.shape[i] - 1) * abs(self.strides[i])
        return max_index

    @always_inline
    fn offset_at(self, indices: IntArray) -> Int:
        """Return the absolute linear offset in the underlying buffer
        for the given multidimensional indices."""
        if indices.size() != self.rank():
            panic("NDBuffer.offset_at: Incorrect number of indices")

        return IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )

    @always_inline
    fn to_dtype[NewType: DType](self) -> NDBuffer[NewType]:
        new_buffer = self.contiguous_buffer().to_dtype[NewType]()
        return NDBuffer[NewType](new_buffer^, self.shape)

    @always_inline
    fn __imul__(self, other: NDBuffer[dtype]):
        self.inplace_ops[Multiply](other)

    @always_inline
    fn __iadd__(self, other: NDBuffer[dtype]):
        self.inplace_ops[Add](other)

    @always_inline
    fn __isub__(self, other: NDBuffer[dtype]):
        self.inplace_ops[Subtract](other)

    @always_inline
    fn __itruediv__(self, other: NDBuffer[dtype]):
        self.inplace_ops[Divide](other)

    @always_inline
    fn shared(self) -> Bool:
        """Check if underlying buffer is shared."""
        return self.buffer.is_shared()

    fn share(
        mut self,
        shape: Optional[Shape] = None,
        strides: Optional[Strides] = None,
        offset: Int = 0,
    ) -> NDBuffer[dtype]:
        """
        Create shared view of this buffer.
        First call enables ref counting. Subsequent calls just create views.
        """
        # Enable ref counting if not already shared
        if not self.shared():
            self.buffer.shared()

        new_shape = shape.or_else(self.shape)
        new_strides = strides.or_else(Strides.default(new_shape))
        return NDBuffer[dtype](
            buffer=self.buffer.copy(),
            shape=new_shape,
            strides=new_strides,
            offset=offset,
        )

    @always_inline
    fn __is__(self, other: NDBuffer[dtype]) -> Bool:
        return self.buffer.data == other.buffer.data

    @always_inline
    fn zero(self):
        self.fill(Scalar[dtype](0))

    @always_inline
    fn fill(self, value: Scalar[dtype]):
        if self.is_contiguous():
            self.buffer.fill(value, self.offset, self.offset + self.numels())
        else:
            for coord in self.shape:
                self[coord] = value

    @always_inline
    fn contiguous(self, new_shape: Optional[Shape] = None) -> NDBuffer[dtype]:
        target_shape = new_shape.or_else(self.shape)
        if (
            self.is_contiguous()
            and not self.shared()
            and target_shape == self.shape
        ):
            return self.copy()
        return NDBuffer[dtype](self.contiguous_buffer(), target_shape)

    @always_inline
    fn map[
        map_buffer: fn (Buffer[dtype]) -> Buffer[dtype],
        map_element: fn (Scalar[dtype]) -> Scalar[dtype],
    ](self) -> Buffer[dtype]:
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            return map_buffer(self.buffer[start:end])
        else:
            var buffer = Buffer[dtype](self.numels())
            var index = 0
            for coord in self.shape:
                buffer[index] = map_element(self[coord])
                index += 1
            return buffer^

    @always_inline
    fn reduce[
        reduce_buffer: fn (Buffer[dtype], Int, Optional[Int]) -> Scalar[dtype],
        reduce_elements: fn (Scalar[dtype], Scalar[dtype]) -> Scalar[dtype],
        unit: Scalar[dtype] = Scalar[dtype](0),
    ](self) -> Scalar[dtype]:
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            return reduce_buffer(self.buffer, start, end)
        else:
            var accum: Scalar[dtype] = unit
            for coord in self.shape:
                accum = reduce_elements(self[coord], accum)
            return accum

    @always_inline
    fn sum_all(self) -> Scalar[dtype]:
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            return self.buffer.sum(start, end)
        else:
            var accum_sum: Scalar[dtype] = Scalar[dtype](0)
            for coord in self.shape:
                accum_sum += self[coord]
            return accum_sum

    fn sum(self, reduction_axes: IntArray, keepdims: Bool) -> NDBuffer[dtype]:
        # Step 1: Normalize and validate reduction axes
        var normalized_axes = self._normalize_reduction_axes(reduction_axes)

        # Step 2: Compute output shape with pre-validated axes
        var out_shape = self.shape.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )

        var out = NDBuffer[dtype].zeros(out_shape)

        # Step 3: Handle scalar output cases
        if out_shape == Shape():
            # This covers both scalar input AND full reduction cases
            out[IntArray()] = self.sum_all()
        else:
            # Step 4: Handle partial reduction with proper coordinate mapping
            reduction_axes_shape = self.shape.reduced_shape(normalized_axes)

            for out_coord in out_shape:
                var accum_sum = Scalar[dtype](0)
                for red_coord in reduction_axes_shape:
                    # Use normalized_axes (sorted) for coordinate reconstruction
                    var self_coord = out_coord.replace(
                        normalized_axes, red_coord
                    ) if keepdims else out_coord.insert(
                        normalized_axes, red_coord
                    )
                    accum_sum += self[self_coord]
                out[out_coord] = accum_sum

        return out^

    fn _normalize_reduction_axes(self, axes: IntArray) -> IntArray:
        """Normalize reduction axes: handle empty list, negative indices, sort, and deduplicate.
        """
        var rank = self.rank()

        # Empty axes list means reduce over all dimensions
        if len(axes) == 0:
            return IntArray.range(start=0, end=rank, step=1)

        # Normalize negative indices and validate bounds
        var normalized = IntArray.with_capacity(len(axes))
        for axis in axes:
            var norm_axis = axis
            if norm_axis < 0:
                norm_axis = rank + norm_axis
            if norm_axis < 0 or norm_axis >= rank:
                panic(
                    "Reduction axis out of bounds: "
                    + axis.__str__()
                    + " for rank "
                    + rank.__str__()
                )
            normalized.append(norm_axis)

        # Sort and remove duplicates
        normalized.sort(asc=True)
        var result = IntArray.with_capacity(len(normalized))
        var prev = -1
        for axis in normalized:
            if axis != prev:
                result.append(axis)
                prev = axis

        return result^

    fn flatten(
        self,
        start_dim: Int = 0,
        end_dim: Optional[Int] = None,
    ) -> NDBuffer[dtype]:
        rank = self.rank()
        if rank == 0:
            return self.contiguous()
        var endd = end_dim.or_else(rank - 1)

        if endd < start_dim:
            panic("NDBuffer → flatten: end_dim must be >= start_dim")

        var original_shape = self.shape
        # compute new shape
        collapsed = original_shape[start_dim : endd + 1].product()
        new_shape = (
            original_shape[:start_dim]
            + [collapsed]
            + original_shape[endd + 1 :]
        )
        return self.contiguous(new_shape)

    @always_inline
    fn contiguous_buffer(self) -> Buffer[dtype]:
        """Returns a contiguous copy of the buffer with the same data."""
        # - same shape
        # - contiguous strides
        # - offset = 0
        # - copies data from original
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            return self.buffer[start:end]
        else:
            var buffer = Buffer[dtype](self.numels())
            var index = 0
            for coord in self.shape:
                buffer[index] = self[coord]
                index += 1
            return buffer^

    fn count(self, key: Scalar[dtype]) -> Int:
        """Count occurence of the key in the buffer."""
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            return self.buffer.count(key, start, end)
        else:
            var _count = 0
            for coord in self.shape:
                if self[coord] == key:
                    _count += 1
            return _count

    fn unique(self) -> NDBuffer[dtype]:
        """Get the unique values in the buffer."""
        var uniques = Set[Scalar[dtype]]()
        if self.is_contiguous():
            if not self.shared():
                for i in range(self.numels()):
                    uniques.add(self.buffer[i])
            else:
                var start = self.offset
                var end = start + self.numels()
                for i in range(start, end):
                    uniques.add(self.buffer[i])
        else:
            for coord in self.shape:
                uniques.add(self[coord])
        var distincts = List[Scalar[dtype]](capacity=UInt(len(uniques)))
        for elem in uniques:
            distincts.append(elem)
        var unique_shape = Shape(len(distincts))
        return NDBuffer[dtype](Buffer[dtype](distincts), unique_shape)

    @always_inline
    fn copy_from_alike[
        overwrite: Bool = True, validate: Bool = True
    ](self: NDBuffer[dtype], other: NDBuffer[dtype]):
        @parameter
        if validate:
            if not self.shape == other.shape:
                panic(
                    (
                        "NDBuffer → copy_from_alike(other):"
                        " dimension mismatch: self shape"
                    ),
                    self.shape.__str__(),
                    "≠",
                    "other shape",
                    other.shape.__str__(),
                )

        if self.is_contiguous() and other.is_contiguous():
            self_start = self.offset
            other_start = other.offset
            self_end = self_start + self.numels()
            other_end = other_start + other.numels()

            @parameter
            if overwrite:
                self.buffer.inplace_ops[Overwrite, validate=validate](
                    other.buffer, self_start, self_end, other_start, other_end
                )
            else:
                self.buffer.inplace_ops[Add, validate=validate](
                    other.buffer, self_start, self_end, other_start, other_end
                )

        elif self.is_contiguous() and not other.is_contiguous():
            var index = self.offset
            for coord in other.shape:

                @parameter
                if overwrite:
                    self.buffer[index] = other[coord]
                else:
                    self.buffer[index] += other[coord]
                index += 1

        elif not self.is_contiguous() and other.is_contiguous():
            var index = other.offset
            for coord in self.shape:

                @parameter
                if overwrite:
                    self[coord] = other.buffer[index]
                else:
                    self[coord] += other.buffer[index]
                index += 1

        else:
            for coord in self.shape:

                @parameter
                if overwrite:
                    self[coord] = other[coord]
                else:
                    self[coord] += other[coord]

    @always_inline
    fn fill(self, other: NDBuffer[dtype]):
        if self.__is__(other):
            panic("NDBuffer → fill: cannot fill with self")

        if other.is_scalar() or other.shape == Shape.Unit():
            self.fill(
                other.item()
            )  # Scalar/Singleton NDBuffer - shared or otherwise
            return

        if self.shape == other.shape:
            self.copy_from_alike[overwrite=True, validate=True](other)
        else:
            # Handle broadcast
            if not ShapeBroadcaster.broadcastable(self.shape, other.shape):
                panic(
                    "NDBuffer → fill(other): dimension mismatch: self shape",
                    self.shape.__str__(),
                    "≠",
                    "other shape",
                    other.shape.__str__(),
                )
            var broadcast_shape = ShapeBroadcaster.broadcast_shape(
                self.shape, other.shape
            )
            if broadcast_shape != self.shape:
                panic(
                    "NDBuffer → fill: broadcasted shape must match receiver"
                    " shape"
                )

            # self.shape -> Target shape
            # other.shape -> Source shape

            mask = ShapeBroadcaster.broadcast_mask(other.shape, self.shape)
            for coord in self.shape:
                src_coord = ShapeBroadcaster.translate_index(
                    other.shape, coord, mask, self.shape
                )
                self[coord] = other[src_coord]

    @always_inline
    fn inplace_ops[
        op_code: Int,
    ](self: NDBuffer[dtype], other: NDBuffer[dtype]):
        # Broadcast validation
        if not ShapeBroadcaster.broadcastable(self.shape, other.shape):
            panic(
                "NDBuffer → inplace_ops: dimension mismatch: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )

        # Handle broadcasting case
        if self.shape != other.shape:
            broadcast_shape = ShapeBroadcaster.broadcast_shape(
                self.shape, other.shape
            )

            # PyTorch's rule: broadcasted shape must match receiver shape
            if broadcast_shape != self.shape:
                panic(
                    "NDBuffer → inplace_ops: broadcasted shape "
                    + broadcast_shape.__str__()
                    + " must match receiver shape "
                    + self.shape.__str__()
                )

            # Get the broadcasted result which is now of self's shape
            var broadcast_result = self.broadcast_buffer[op_code](other)
            self.copy_from_alike[overwrite=True, validate=False](
                broadcast_result^
            )

        else:
            # Same shape case
            if self.is_contiguous() and other.is_contiguous():
                self_start = self.offset
                self_end = self_start + self.numels()
                other_start = other.offset
                other_end = other_start + other.numels()
                self.buffer.inplace_ops[op_code](
                    other.buffer, self_start, self_end, other_start, other_end
                )

            elif self.is_contiguous() and not other.is_contiguous():
                var index = self.offset
                for coord in other.shape:
                    self.buffer[index] = Self.scalar_fn[op_code](
                        self.buffer[index], other[coord]
                    )
                    index += 1

            elif not self.is_contiguous() and other.is_contiguous():
                var index = other.offset
                for coord in self.shape:
                    self[coord] = Self.scalar_fn[op_code](
                        self[coord], other.buffer[index]
                    )
                    index += 1
            else:
                for coord in self.shape:
                    self[coord] = Self.scalar_fn[op_code](
                        self[coord], other[coord]
                    )

    @always_inline
    fn inplace_scalar_ops[
        op_code: Int,
    ](self: NDBuffer[dtype], scalar: Scalar[dtype]):
        @parameter
        if op_code == Divide:
            if scalar == Scalar[dtype](0):
                panic("NDBuffer → inplace_scalar_ops: cannot divide by zero")

        if self.is_contiguous():
            start = self.offset
            end = start + self.numels()
            self.buffer.inplace_ops_scalar[op_code](scalar, start, end)

        else:
            for coord in self.shape:
                self[coord] = Self.scalar_fn[op_code](self[coord], scalar)

    @always_inline
    fn __add__(self, other: NDBuffer[dtype]) -> NDBuffer[dtype]:
        return self.arithmetic_ops[Add](other)

    @always_inline
    fn __mul__(self, other: NDBuffer[dtype]) -> NDBuffer[dtype]:
        return self.arithmetic_ops[Multiply](other)

    @always_inline
    fn __mul__(self, scalar: Scalar[dtype]) -> NDBuffer[dtype]:
        return self.scalar_ops[Multiply](scalar)

    @always_inline
    fn __rmul__(self, scalar: Scalar[dtype]) -> NDBuffer[dtype]:
        return self.__mul__(scalar)

    @always_inline
    fn __sub__(self, other: NDBuffer[dtype]) -> NDBuffer[dtype]:
        return self.arithmetic_ops[Subtract](other)

    @always_inline
    fn __truediv__(self, other: NDBuffer[dtype]) -> NDBuffer[dtype]:
        return self.arithmetic_ops[Divide](other)

    @always_inline
    fn arithmetic_ops[
        op_code: Int,
    ](self: NDBuffer[dtype], other: NDBuffer[dtype]) -> NDBuffer[dtype]:
        # Broadcast validation
        if not ShapeBroadcaster.broadcastable(self.shape, other.shape):
            panic(
                "NDBuffer → arithmetic_ops(self, other): dimension mismatch: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )

        # Handle broadcasting case
        if self.shape != other.shape:
            return self.broadcast_buffer[op_code](other)

        if self.is_contiguous() and other.is_contiguous():
            self_start = self.offset
            self_end = self_start + self.numels()
            other_start = other.offset
            other_end = other_start + other.numels()

            var result_buffer = self.buffer.arithmetic_ops[op_code](
                other.buffer, self_start, self_end, other_start, other_end
            )
            return NDBuffer[dtype](result_buffer^, self.shape)

        else:
            var result_buffer = Buffer[dtype](self.numels())
            var index = 0

            if self.is_contiguous() and not other.is_contiguous():
                var offset = self.offset
                for coord in other.shape:
                    result_buffer[index] = Self.scalar_fn[op_code](
                        self.buffer[offset + index], other[coord]
                    )
                    index += 1

            elif not self.is_contiguous() and other.is_contiguous():
                var offset = other.offset
                for coord in self.shape:
                    result_buffer[index] = Self.scalar_fn[op_code](
                        self[coord], other.buffer[offset + index]
                    )
                    index += 1

            else:
                for coord in self.shape:
                    result_buffer[index] = Self.scalar_fn[op_code](
                        self[coord], other[coord]
                    )
                    index += 1

            return NDBuffer[dtype](result_buffer^, self.shape)

    @always_inline
    fn broadcast_buffer[
        op_code: Int,
    ](self: NDBuffer[dtype], other: NDBuffer[dtype]) -> NDBuffer[dtype]:
        if self.shape.rank() == 0 or other.shape.rank() == 0:
            return self.broadcast_scalar_buffer[op_code](other)
        else:
            return self.broadcast_nd_buffer[op_code](other)

    @always_inline
    fn broadcast_scalar_buffer[
        op_code: Int
    ](self: NDBuffer[dtype], other: NDBuffer[dtype]) -> NDBuffer[dtype]:
        result_shape = other.shape if self.shape.rank() == 0 else self.shape
        var buffer = Buffer[dtype](result_shape.num_elements())
        strides = Strides.default(result_shape)

        for coord in result_shape:
            self_val = self.item() if self.shape.rank() == 0 else self[coord]
            other_val = (
                other.item() if other.shape.rank() == 0 else other[coord]
            )
            index = IndexCalculator.flatten_index(
                result_shape, coord, strides, 0
            )

            buffer[index] = Self.scalar_fn[op_code](self_val, other_val)

        return NDBuffer[dtype](buffer^, result_shape)

    @always_inline
    fn broadcast_nd_buffer[
        op_code: Int
    ](self: NDBuffer[dtype], other: NDBuffer[dtype]) -> NDBuffer[dtype]:
        result_shape = ShapeBroadcaster.broadcast_shape(self.shape, other.shape)

        mask1 = ShapeBroadcaster.broadcast_mask(self.shape, result_shape)
        mask2 = ShapeBroadcaster.broadcast_mask(other.shape, result_shape)

        var buffer = Buffer[dtype](result_shape.num_elements())
        strides = Strides.default(result_shape)

        for coord in result_shape:
            self_coord = ShapeBroadcaster.translate_index(
                self.shape, coord, mask1, result_shape
            )
            other_coord = ShapeBroadcaster.translate_index(
                other.shape, coord, mask2, result_shape
            )
            index = IndexCalculator.flatten_index(
                result_shape, coord, strides, 0
            )

            buffer[index] = Self.scalar_fn[op_code](
                self[self_coord], other[other_coord]
            )
        return NDBuffer[dtype](buffer^, result_shape)

    @always_inline
    fn broadcast_to(self, target_shape: Shape) -> NDBuffer[dtype]:
        own_shape = self.shape
        if not ShapeBroadcaster.broadcastable(own_shape, target_shape):
            panic(
                "NDBuffer → broadcast_to(target_shape): "
                + own_shape.__str__()
                + " not broadcastable to "
                + target_shape.__str__()
            )

        mask = ShapeBroadcaster.broadcast_mask(own_shape, target_shape)
        out = NDBuffer[dtype].zeros(target_shape)

        for target_coord in target_shape:
            src_coord = ShapeBroadcaster.translate_index(
                own_shape, target_coord, mask, target_shape
            )
            out[target_coord] = self[src_coord]

        return out^

    @always_inline
    @staticmethod
    fn scalar_fn[
        op_code: Int
    ](lhs: Scalar[dtype], rhs: Scalar[dtype]) -> Scalar[dtype]:
        @parameter
        if op_code == Add:
            return lhs + rhs
        elif op_code == Subtract:
            return lhs - rhs
        elif op_code == ReverseSubtract:
            return rhs - lhs
        elif op_code == Multiply:
            return lhs * rhs
        elif op_code == Divide:
            return lhs / rhs
        else:  # op_code == ReverseDivide
            return rhs / lhs

    @always_inline
    fn scalar_ops[
        op_code: Int,
    ](self: NDBuffer[dtype], scalar: Scalar[dtype]) -> NDBuffer[dtype]:
        @parameter
        if op_code == Divide:
            if scalar == Scalar[dtype](0):
                panic("NDBuffer → scalar_ops: cannot divide by zero")

        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            var result_buffer = self.buffer.arithmetic_ops_scalar[op_code](
                scalar, start, end
            )
            return NDBuffer[dtype](result_buffer^, self.shape)

        else:
            var index = 0
            var result_buffer = Buffer[dtype](self.numels())

            for coord in self.shape:
                result_buffer[index] = Self.scalar_fn[op_code](
                    self[coord], scalar
                )
                index += 1

            return NDBuffer[dtype](result_buffer^, self.shape)

    fn __eq__(self, other: Self) -> Bool:
        return self.compare[Equal](other).buffer.all_true()

    fn __ne__(self, other: Self) -> Bool:
        return self.compare[NotEqual](other).buffer.all_true()

    fn compare[
        op_code: Int,
    ](self: NDBuffer[dtype], other: NDBuffer[dtype]) -> NDBuffer[DType.bool]:
        if not self.shape == other.shape:
            panic(
                "NDBuffer → compare(self, other): dimension mismatch: "
                + self.shape.__str__()
                + "≠"
                + other.shape.__str__()
            )

        if self.is_contiguous() and other.is_contiguous():
            var self_contiguous = self.contiguous_buffer()
            var other_contiguous = other.contiguous_buffer()
            var result_buffer = self_contiguous.compare_buffer_full[op_code](
                other_contiguous
            )
            return NDBuffer[DType.bool](result_buffer^, self.shape)

        else:
            var index = 0
            var buffer = Buffer[DType.bool](self.numels())

            for coord in self.shape:
                var self_val = self[coord]
                var other_val = other[coord]

                @parameter
                if op_code == Equal:
                    buffer[index] = self_val == other_val
                elif op_code == NotEqual:
                    buffer[index] = self_val != other_val
                elif op_code == LessThan:
                    buffer[index] = self_val < other_val
                elif op_code == LessThanEqual:
                    buffer[index] = self_val <= other_val
                elif op_code == GreaterThan:
                    buffer[index] = self_val > other_val
                else:  # GreaterThanEqual
                    buffer[index] = self_val >= other_val

                index += 1

            return NDBuffer[DType.bool](buffer^, self.shape)

    @always_inline
    fn compare_scalar[
        op_code: Int,
    ](self: NDBuffer[dtype], scalar: Scalar[dtype]) -> NDBuffer[DType.bool]:
        if self.is_contiguous():
            var contiguous_data = self.contiguous_buffer()
            var result_buffer = contiguous_data.compare_scalar_full[op_code](
                scalar
            )
            return NDBuffer[DType.bool](result_buffer^, self.shape)

        else:
            var index = 0
            var buffer = Buffer[DType.bool](self.numels())

            for coord in self.shape:
                var value = self[coord]

                @parameter
                if op_code == Equal:
                    buffer[index] = value == scalar
                elif op_code == NotEqual:
                    buffer[index] = value != scalar
                elif op_code == LessThan:
                    buffer[index] = value < scalar
                elif op_code == LessThanEqual:
                    buffer[index] = value <= scalar
                elif op_code == GreaterThan:
                    buffer[index] = value > scalar
                else:  # GreaterThanEqual
                    buffer[index] = value >= scalar

                index += 1

            return NDBuffer[DType.bool](buffer^, self.shape)

    @always_inline
    fn all_close[
        rtol: Scalar[dtype] = 1e-5,
        atol: Scalar[dtype] = 1e-8,
    ](self, other: Self) -> Bool:
        constrained[
            dtype.is_floating_point(),
            "NDBuffer → all_close is for floating point data types only",
        ]()

        if self.shape != other.shape:
            panic(
                "NDBuffer → all_close(other) expects same shaped buffers: "
                + self.shape.__str__()
                + "≠"
                + other.shape.__str__()
            )

        return self.contiguous_buffer().all_close[rtol=rtol, atol=atol](
            other.contiguous_buffer()
        )

    @always_inline
    fn element_at(self, index: Int) -> Scalar[dtype]:
        idx = index + self.max_index() if index < 0 else index
        if idx < 0 or idx > self.max_index():
            panic(
                "NDBuffer → element_at: index out of bounds.",
                "NDBuffer max index",
                self.max_index().__str__(),
                ", provided index",
                index.__str__(),
            )
        return self.buffer[idx]

    @always_inline
    fn sum_over_broadcasted_axes(
        extended_buffer: NDBuffer[dtype], target_shape: Shape
    ) -> NDBuffer[dtype]:
        result = extended_buffer.contiguous()
        current_shape = extended_buffer.shape
        # Sum over extra leading dimensions
        while len(current_shape) > len(target_shape):
            result = result.sum(reduction_axes=IntArray(0), keepdims=False)
            current_shape = result.shape
        # Sum over mismatched dimensions
        for i in range(len(target_shape)):
            if current_shape[i] != target_shape[i] and current_shape[i] > 1:
                result = result.sum(reduction_axes=IntArray(i), keepdims=True)
                current_shape = result.shape
        return result^


fn main() raises:
    pass
