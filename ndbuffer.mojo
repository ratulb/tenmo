from shapes import Shape
from strides import Strides
from intlist import IntList
from buffers import Buffer
from layout.int_tuple import IntArray
from indexhelper import IndexCalculator
from broadcasthelper import ShapeBroadcaster
from common_utils import panic
from memory import memcpy, ArcPointer
from collections import Set
from operators import (
    Multiply,
    Add,
    Subtract,
    ReverseSubtract,
    Divide,
    ReverseDivide,
    Equal,
    NotEqual,
    LessThan,
    LessThanEqual,
    GreaterThan,
    GreaterThanEqual,
)


struct NDBuffer[dtype: DType](Copyable & Movable & EqualityComparable):
    var shape: Shape
    var strides: Strides
    var offset: Int
    var buffer: Optional[Buffer[dtype]]
    var shared_buffer: Optional[ArcPointer[Buffer[dtype]]]
    var _contiguous: Bool

    fn __init__(
        out self,
    ):
        self.buffer = Buffer[dtype](1)
        self.shared_buffer = None
        self.shape = Shape()
        self.strides = Strides.Zero()
        self.offset = 0
        self._contiguous = True

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
        _shape = shape.value() if shape else Shape(buffer.size)
        self.shape = _shape.copy()
        self.buffer = Optional(buffer^)
        self.shared_buffer = None
        self.strides = strides.value() if strides else Strides.default(_shape)
        self.offset = offset
        self._contiguous = False
        self._contiguous = self.is_contiguous()

    fn __init__(
        out self,
        shape: Shape,
        strides: Optional[Strides] = None,
        offset: Int = 0,
    ):
        self.buffer = Optional(Buffer[dtype](shape.num_elements()))
        self.shared_buffer = None
        self.shape = shape
        self.strides = strides.value() if strides else Strides.default(shape)
        self.offset = offset
        self._contiguous = False
        self._contiguous = self.is_contiguous()

    fn __init__(
        out self,
        shared_buffer: Optional[ArcPointer[Buffer[dtype]]],
        shape: Shape,
        strides: Optional[Strides] = None,
        offset: Int = 0,
    ):
        self.buffer = None
        self.shared_buffer = shared_buffer.copy()
        self.shape = shape.copy()
        self.strides = strides.value() if strides else Strides.default(shape)
        self.offset = offset
        self._contiguous = False
        self._contiguous = self.is_contiguous()

    fn __moveinit__(out self, deinit other: Self):
        self.buffer = other.buffer^
        self.shared_buffer = other.shared_buffer^
        self.shape = other.shape^
        self.strides = other.strides^
        self.offset = other.offset
        self._contiguous = other._contiguous

    fn __copyinit__(out self, other: Self):
        self.buffer = other.buffer.copy()
        self.shared_buffer = other.shared_buffer.copy()
        self.shape = other.shape.copy()
        self.strides = other.strides.copy()
        self.offset = other.offset
        self._contiguous = other._contiguous

    fn __del__(deinit self):
        _ = self.buffer^
        _ = self.shared_buffer^
        _ = self.shape^
        _ = self.strides^

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

    @always_inline
    fn is_contiguous(self) -> Bool:
        return self.strides.is_contiguous(self.shape)

    @always_inline
    fn data(
        ref self,
    ) -> ref [
        origin_of(self.buffer.value(), self.shared_buffer.value()[])
    ] Buffer[dtype]:
        if self.buffer:
            return self.buffer.value()
        return self.shared_buffer.value()[]

    @always_inline
    fn __getitem__(self, indices: IntArray) -> Scalar[dtype]:
        index = IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )
        return self.data()[index]

    @always_inline
    fn __setitem__(self, indices: IntArray, value: Scalar[dtype]):
        index = IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )
        self.data()[index] = value

    @always_inline
    fn __getitem__(self, indices: List[Int]) -> Scalar[dtype]:
        index = IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )
        return self.data()[index]

    @always_inline
    fn __setitem__(self, indices: List[Int], value: Scalar[dtype]):
        index = IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )
        self.data()[index] = value

    @always_inline
    fn __getitem__(self, indices: IntList) -> Scalar[dtype]:
        index = IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )
        return self.data()[index]

    @always_inline
    fn __setitem__(self, indices: IntList, value: Scalar[dtype]):
        index = IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )
        self.data()[index] = value

    @always_inline
    fn __getitem__(self, indices: VariadicList[Int]) -> Scalar[dtype]:
        index = IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )
        return self.data()[index]

    @always_inline
    fn __setitem__(self, indices: VariadicList[Int], value: Scalar[dtype]):
        index = IndexCalculator.flatten_index(
            self.shape, indices, self.strides, self.offset
        )
        self.data()[index] = value

    @always_inline
    fn item(self) -> Scalar[dtype]:
        if self.shape != Shape(1) and self.shape != Shape():
            panic(
                "NDBuffer → item(self): only valid for zero dim"
                " buffer/singleton, got shape: "
                + self.shape.__str__()
            )
        if self.shape == Shape(1):
            index = IntArray(size=1)
            index[0] = 0
            return self[index]
        else:
            return self[IntArray()]

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
        return self.buffer is None and self.shared_buffer is not None

    fn share(
        mut self,
        shape: Optional[Shape] = None,
        strides: Optional[Strides] = None,
        offset: Int = 0,
    ) -> NDBuffer[dtype]:
        if not self.shared():
            self.shared_buffer = Optional(self.buffer.unsafe_take().shared())

        new_shape = shape.value() if shape else self.shape
        return NDBuffer[dtype](self.shared_buffer, new_shape, strides, offset)

    @always_inline
    fn zero(self):
        self.fill(Scalar[dtype](0))

    @always_inline
    fn fill(self, value: Scalar[dtype]):
        if self._contiguous:
            self.data().fill(value, self.offset, self.offset + self.numels())
        else:
            for coord in self.shape:
                self[coord] = value

    @always_inline
    fn contiguous(self) -> NDBuffer[dtype]:
        return NDBuffer[dtype](self.contiguous_buffer(), self.shape)

    @always_inline
    fn map[
        map_buffer: fn (Buffer[dtype]) -> Buffer[dtype],
        map_element: fn (Scalar[dtype]) -> Scalar[dtype],
    ](self) -> Buffer[dtype]:
        if self._contiguous:
            var start = self.offset
            var end = start + self.numels()
            return map_buffer(self.data()[start:end])
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
        if self._contiguous:
            var start = self.offset
            var end = start + self.numels()
            return reduce_buffer(self.data(), start, end)
        else:
            var accum: Scalar[dtype] = unit
            for coord in self.shape:
                accum = reduce_elements(self[coord], accum)
            return accum

    @always_inline
    fn sum_all(self) -> Scalar[dtype]:
        if self._contiguous:
            var start = self.offset
            var end = start + self.numels()
            return self.data().sum(start, end)
        else:
            var accum_sum: Scalar[dtype] = Scalar[dtype](0)
            for coord in self.shape:
                accum_sum += self[coord]
            return accum_sum

    fn sum(self, reduction_axes: IntList, keepdims: Bool) -> NDBuffer[dtype]:
        var out_shape = self.shape.compute_output_shape(
            reduction_axes, keepdims
        )
        var out = NDBuffer[dtype].zeros(out_shape)
        if out_shape == Shape():
            # We're producing a scalar (either from scalar NDBuffer or full reduction)
            out[IntArray()] = self.sum_all()
        else:
            reduction_axes_shape = Shape(
                self.shape.axes_spans.select(reduction_axes)
            )
            for out_coord in out_shape.indices():
                var accum_sum = Scalar[dtype](0)
                for red_coord in reduction_axes_shape.indices():
                    self_coord = out_coord.replace(
                        reduction_axes, red_coord
                    ) if keepdims else out_coord.insert(
                        reduction_axes, red_coord
                    )
                    accum_sum += self[self_coord]
                out[out_coord] = accum_sum

        return out^

    @always_inline
    fn contiguous_buffer(self) -> Buffer[dtype]:
        """Returns a contiguous copy of the buffer with the same data."""
        # - same shape
        # - contiguous strides
        # - offset = 0
        # - copies data from original
        if self._contiguous:
            var start = self.offset
            var end = start + self.numels()
            return self.data()[start:end]
        else:
            var buffer = Buffer[dtype](self.numels())
            var index = 0
            for coord in self.shape:
                buffer[index] = self[coord]
                index += 1
            return buffer^

    fn count(self, key: Scalar[dtype]) -> Int:
        """Count occurence of the key in the buffer."""
        if self._contiguous:
            var start = self.offset
            var end = start + self.numels()
            return self.data().count(key, start, end)
        else:
            var _count = 0
            for coord in self.shape:
                if self[coord] == key:
                    _count += 1
            return _count

    fn unique(self) -> NDBuffer[dtype]:
        """Get the unique values in the buffer."""
        var uniques = Set[Scalar[dtype]]()
        if self._contiguous:
            if not self.shared():
                for i in range(self.numels()):
                    uniques.add(self.data()[i])
            else:
                var start = self.offset
                var end = start + self.numels()
                for i in range(start, end):
                    uniques.add(self.data()[i])
        else:
            for coord in self.shape:
                uniques.add(self[coord])
        var distincts = List[Scalar[dtype]](capacity=UInt(len(uniques)))
        for elem in uniques:
            distincts.append(elem)
        return NDBuffer[dtype](Buffer[dtype](distincts), self.shape)

    @always_inline
    fn fill(lhs, other: NDBuffer[dtype]):
        if not ShapeBroadcaster.broadcastable(lhs.shape, other.shape):
            panic(
                "NDBuffer → fill(lhs, other): dimension mismatch: lhs shape",
                lhs.shape.__str__(),
                "≠",
                "other shape",
                other.shape.__str__(),
            )
        if other.is_scalar() or other.shape == Shape.Unit():
            lhs.fill(
                other.item()
            )  # Scalar/Singleton NDBuffer - shared or otherwise
            return
        var broadcast_shape = ShapeBroadcaster.broadcast_shape(
            lhs.shape, other.shape
        )
        if broadcast_shape != lhs.shape:
            panic(
                "NDBuffer → fill: broadcasted shape must match receiver shape"
            )

        # Make other contiguous because it could be same storage and makes processing simpler/faster
        var rhs = other.contiguous()

        if lhs.shape == rhs.shape:
            if lhs._contiguous:
                lhs.data().overwrite(
                    rhs.data(), lhs.offset, lhs.offset + lhs.numels()
                )
            else:
                var index = 0
                for coord in lhs.shape:
                    lhs[coord] = rhs.data()[index]
                    index += 1
        else:  # Handle broadcast
            # lhs.shape -> Target shape
            # rhs.shape -> Source shape
            mask = ShapeBroadcaster.broadcast_mask(rhs.shape, lhs.shape)
            for coord in lhs.shape:
                src_coord = ShapeBroadcaster.translate_index(
                    rhs.shape, coord, mask, lhs.shape
                )
                lhs[coord] = rhs[src_coord]

    @always_inline
    fn inplace_ops[
        opcode: Int,
    ](lhs: NDBuffer[dtype], rhs: NDBuffer[dtype]):
        # Broadcast validation
        if not ShapeBroadcaster.broadcastable(lhs.shape, rhs.shape):
            panic(
                "NDBuffer → inplace_ops: dimension mismatch: "
                + lhs.shape.__str__()
                + ", "
                + rhs.shape.__str__()
            )

        # Handle broadcasting case
        if lhs.shape != rhs.shape:
            broadcast_shape = ShapeBroadcaster.broadcast_shape(
                lhs.shape, rhs.shape
            )

            # PyTorch's rule: broadcasted shape must match receiver shape
            if broadcast_shape != lhs.shape:
                panic(
                    "NDBuffer → inplace_ops: broadcasted shape "
                    + broadcast_shape.__str__()
                    + " must match receiver shape "
                    + lhs.shape.__str__()
                )

            # Use existing broadcast operation
            var broadcast_result = lhs.broadcast_buffer[opcode](rhs)

            if lhs._contiguous:
                result_buffer = broadcast_result.take_buffer()
                lhs.data().overwrite(
                    result_buffer, lhs.offset, lhs.offset + lhs.numels()
                )
            else:
                for coord in lhs.shape:
                    lhs[coord] = broadcast_result[coord]

        else:
            # Same shape case - use hybrid approach
            if lhs._contiguous and rhs._contiguous:
                # Fast path: Use out-of-place operation + overwrite
                var result = lhs.arithmetic_ops[opcode](rhs)
                result_buffer = result.take_buffer()
                lhs.data().overwrite(
                    result_buffer, lhs.offset, lhs.offset + lhs.numels()
                )

            else:
                for coord in lhs.shape:
                    var lhs_val = lhs[coord]
                    var rhs_val = rhs[coord]

                    @parameter
                    if opcode == Multiply:
                        lhs[coord] = lhs_val * rhs_val
                    elif opcode == Add:
                        lhs[coord] = lhs_val + rhs_val
                    elif opcode == Subtract:
                        lhs[coord] = lhs_val - rhs_val
                    else:  # Divide
                        lhs[coord] = lhs_val / rhs_val

    @always_inline
    fn take_buffer(mut self) -> Buffer[dtype]:
        """Take the underlying buffer out, panicking if None."""
        if not self.buffer:
            panic("NDBuffer: expected buffer to be present")
        return self.buffer.unsafe_take()

    @always_inline
    fn inplace_scalar_ops[
        opcode: Int,
    ](self: NDBuffer[dtype], scalar: Scalar[dtype]):
        @parameter
        if opcode == Divide:
            if scalar == Scalar[dtype](0):
                panic("NDBuffer → inplace_scalar_ops: cannot divide by zero")

        if self._contiguous:
            start = self.offset
            end = start + self.numels()
            var result = self.scalar_ops[opcode](scalar)
            self.data().overwrite(result.take_buffer(), start, end)

        else:
            for coord in self.shape:

                @parameter
                if opcode == Multiply:
                    self[coord] *= scalar
                elif opcode == Add:
                    self[coord] += scalar
                elif opcode == Subtract:
                    self[coord] -= scalar
                else:  # Divide
                    self[coord] /= scalar

    @always_inline
    fn __add__(self, other: NDBuffer[dtype]) -> NDBuffer[dtype]:
        return self.arithmetic_ops[Add](other)

    @always_inline
    fn __mul__(self, other: NDBuffer[dtype]) -> NDBuffer[dtype]:
        return self.arithmetic_ops[Multiply](other)

    @always_inline
    fn __sub__(self, other: NDBuffer[dtype]) -> NDBuffer[dtype]:
        return self.arithmetic_ops[Subtract](other)

    @always_inline
    fn __truediv__(self, other: NDBuffer[dtype]) -> NDBuffer[dtype]:
        return self.arithmetic_ops[Divide](other)

    @always_inline
    fn arithmetic_ops[
        opcode: Int,
    ](lhs: NDBuffer[dtype], rhs: NDBuffer[dtype]) -> NDBuffer[dtype]:
        # Broadcast validation
        if not ShapeBroadcaster.broadcastable(lhs.shape, rhs.shape):
            panic(
                "NDBuffer → arithmetic_ops(lhs, rhs): dimension mismatch: "
                + lhs.shape.__str__()
                + ", "
                + rhs.shape.__str__()
            )

        # Handle broadcasting case
        if lhs.shape != rhs.shape:
            return lhs.broadcast_buffer[opcode](rhs)

        # Same shape case - use hybrid approach
        if lhs._contiguous and rhs._contiguous:
            var lhs_contiguous = lhs.contiguous_buffer()
            var rhs_contiguous = rhs.contiguous_buffer()
            var result_buffer: Buffer[dtype]

            @parameter
            if opcode == Multiply:
                result_buffer = lhs_contiguous * rhs_contiguous
            elif opcode == Add:
                result_buffer = lhs_contiguous + rhs_contiguous
            elif opcode == Subtract:
                result_buffer = lhs_contiguous - rhs_contiguous
            else:  # Divide
                result_buffer = lhs_contiguous / rhs_contiguous

            return NDBuffer[dtype](result_buffer^, lhs.shape)

        else:
            var buffer = Buffer[dtype](lhs.numels())
            var index = 0

            for coord in lhs.shape:
                var lhs_val = lhs[coord]
                var rhs_val = rhs[coord]

                @parameter
                if opcode == Multiply:
                    buffer[index] = lhs_val * rhs_val
                elif opcode == Add:
                    buffer[index] = lhs_val + rhs_val
                elif opcode == Subtract:
                    buffer[index] = lhs_val - rhs_val
                else:  # Divide
                    buffer[index] = lhs_val / rhs_val

                index += 1

            return NDBuffer[dtype](buffer^, lhs.shape)

    @always_inline
    fn broadcast_buffer[
        opcode: Int,
    ](lhs: NDBuffer[dtype], rhs: NDBuffer[dtype]) -> NDBuffer[dtype]:
        if lhs.shape.rank() == 0 or rhs.shape.rank() == 0:
            return lhs.broadcast_scalar_buffer[opcode](rhs)
        else:
            return lhs.broadcast_nd_buffer[opcode](rhs)

    @always_inline
    fn broadcast_scalar_buffer[
        opcode: Int
    ](lhs: NDBuffer[dtype], rhs: NDBuffer[dtype]) -> NDBuffer[dtype]:
        result_shape = rhs.shape if lhs.shape.rank() == 0 else lhs.shape
        var buffer = Buffer[dtype](result_shape.num_elements())
        strides = Strides.default(result_shape)

        for coord in result_shape:
            lhs_val = lhs.item() if lhs.shape.rank() == 0 else lhs[coord]
            rhs_val = rhs.item() if rhs.shape.rank() == 0 else rhs[coord]
            index = IndexCalculator.flatten_index(
                result_shape, coord, strides, 0
            )

            buffer[index] = Self.scalar_fn[opcode](lhs_val, rhs_val)

        return NDBuffer[dtype](buffer^, result_shape)

    @always_inline
    fn broadcast_nd_buffer[
        opcode: Int
    ](lhs: NDBuffer[dtype], rhs: NDBuffer[dtype]) -> NDBuffer[dtype]:
        result_shape = ShapeBroadcaster.broadcast_shape(lhs.shape, rhs.shape)

        mask1 = ShapeBroadcaster.broadcast_mask(lhs.shape, result_shape)
        mask2 = ShapeBroadcaster.broadcast_mask(rhs.shape, result_shape)

        var buffer = Buffer[dtype](result_shape.num_elements())
        strides = Strides.default(result_shape)

        for coord in result_shape:
            lhs_coord = ShapeBroadcaster.translate_index(
                lhs.shape, coord, mask1, result_shape
            )
            rhs_coord = ShapeBroadcaster.translate_index(
                rhs.shape, coord, mask2, result_shape
            )
            index = IndexCalculator.flatten_index(
                result_shape, coord, strides, 0
            )

            buffer[index] = Self.scalar_fn[opcode](
                lhs[lhs_coord], rhs[rhs_coord]
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
        opcode: Int
    ](lhs: Scalar[dtype], rhs: Scalar[dtype]) -> Scalar[dtype]:
        var result = Scalar[dtype](0)

        @parameter
        if opcode == Add:
            result = lhs + rhs

        @parameter
        if opcode == Subtract:
            result = lhs - rhs

        @parameter
        if opcode == Multiply:
            result = lhs * rhs

        @parameter
        if opcode == Divide:
            result = lhs / rhs
        return result

    @always_inline
    fn scalar_ops[
        opcode: Int,
    ](self: NDBuffer[dtype], scalar: Scalar[dtype]) -> NDBuffer[dtype]:
        @parameter
        if opcode == Divide:
            if scalar == Scalar[dtype](0):
                panic("NDBuffer → scalar_ops: cannot divide by zero")

        if self._contiguous:
            var contiguous_data = self.contiguous_buffer()
            var result_buffer: Buffer[dtype]

            @parameter
            if opcode == Multiply:
                result_buffer = contiguous_data * scalar
            elif opcode == Add:
                result_buffer = contiguous_data + scalar
            elif opcode == Subtract:
                result_buffer = contiguous_data - scalar
            elif opcode == ReverseSubtract:
                result_buffer = scalar - contiguous_data
            elif opcode == ReverseDivide:
                result_buffer = scalar / contiguous_data
            else:  # Divide
                result_buffer = contiguous_data / scalar

            return NDBuffer[dtype](result_buffer^, self.shape)

        else:
            var index = 0
            var result_buffer = Buffer[dtype](self.numels())

            for coord in self.shape:
                var value = self[coord]

                @parameter
                if opcode == Multiply:
                    result_buffer[index] = value * scalar
                elif opcode == Add:
                    result_buffer[index] = value + scalar
                elif opcode == Subtract:
                    result_buffer[index] = value - scalar
                elif opcode == ReverseDivide:
                    result_buffer[index] = scalar / value
                else:  # Divide
                    result_buffer[index] = value / scalar

                index += 1

            return NDBuffer[dtype](result_buffer^, self.shape)

    fn __eq__(self, other: Self) -> Bool:
        return self.compare[Equal](other).buffer.value().all_true()

    fn __ne__(self, other: Self) -> Bool:
        return self.compare[NotEqual](other).buffer.value().all_true()

    fn compare[
        opcode: Int,
    ](lhs: NDBuffer[dtype], rhs: NDBuffer[dtype]) -> NDBuffer[DType.bool]:
        if not lhs.shape == rhs.shape:
            panic(
                "NDBuffer → compare(lhs, rhs): dimension mismatch: "
                + lhs.shape.__str__()
                + "≠"
                + rhs.shape.__str__()
            )

        if lhs._contiguous and rhs._contiguous:
            var lhs_contiguous = lhs.contiguous_buffer()
            var rhs_contiguous = rhs.contiguous_buffer()
            var result_buffer: Buffer[DType.bool]

            @parameter
            if opcode == Equal:
                result_buffer = lhs_contiguous.eq(rhs_contiguous)
            elif opcode == NotEqual:
                result_buffer = lhs_contiguous.ne(rhs_contiguous)
            elif opcode == LessThan:
                result_buffer = lhs_contiguous.lt(rhs_contiguous)
            elif opcode == LessThanEqual:
                result_buffer = lhs_contiguous.le(rhs_contiguous)
            elif opcode == GreaterThan:
                result_buffer = lhs_contiguous.gt(rhs_contiguous)
            else:  # opcode == GreaterThanEqual
                result_buffer = lhs_contiguous.ge(rhs_contiguous)

            return NDBuffer[DType.bool](result_buffer^, lhs.shape)

        else:
            var index = 0
            var buffer = Buffer[DType.bool](lhs.numels())

            for coord in lhs.shape:
                var lhs_val = lhs[coord]
                var rhs_val = rhs[coord]

                @parameter
                if opcode == Equal:
                    buffer[index] = lhs_val == rhs_val
                elif opcode == NotEqual:
                    buffer[index] = lhs_val != rhs_val
                elif opcode == LessThan:
                    buffer[index] = lhs_val < rhs_val
                elif opcode == LessThanEqual:
                    buffer[index] = lhs_val <= rhs_val
                elif opcode == GreaterThan:
                    buffer[index] = lhs_val > rhs_val
                else:  # GreaterThanEqual
                    buffer[index] = lhs_val >= rhs_val

                index += 1

            return NDBuffer[DType.bool](buffer^, lhs.shape)

    @always_inline
    fn compare_scalar[
        opcode: Int,
    ](self: NDBuffer[dtype], scalar: Scalar[dtype]) -> NDBuffer[DType.bool]:
        if self._contiguous:
            var contiguous_data = self.contiguous_buffer()
            var result_buffer: Buffer[DType.bool]

            @parameter
            if opcode == Equal:
                result_buffer = contiguous_data.eq(scalar)
            elif opcode == NotEqual:
                result_buffer = contiguous_data.ne(scalar)
            elif opcode == LessThan:
                result_buffer = contiguous_data.lt(scalar)
            elif opcode == LessThanEqual:
                result_buffer = contiguous_data.le(scalar)
            elif opcode == GreaterThan:
                result_buffer = contiguous_data.gt(scalar)
            else:  # opcode == GreaterThanEqual
                result_buffer = contiguous_data.ge(scalar)

            return NDBuffer[DType.bool](result_buffer^, self.shape)

        else:
            var index = 0
            var buffer = Buffer[DType.bool](self.numels())

            for coord in self.shape:
                var value = self[coord]

                @parameter
                if opcode == Equal:
                    buffer[index] = value == scalar
                elif opcode == NotEqual:
                    buffer[index] = value != scalar
                elif opcode == LessThan:
                    buffer[index] = value < scalar
                elif opcode == LessThanEqual:
                    buffer[index] = value <= scalar
                elif opcode == GreaterThan:
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
                + ", "
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
        return self.data()[idx]


fn main() raises:
    var runs = 1
    alias _dtype = DType.float32

    for _ in range(runs):
        _ = """test_ndbuffer_set_get()
        test_scalar_buffer()
        test_fill_2()
        test_broadcast_fill()
        test_zero()
        test_add()
        test_equal()
        test_dtype_conversion()
        test_element_at()
        test_ndbuffer_inplace_ops()
        test_count()
        test_unique()
        test_inplace_operations()
        test_inplace_broadcast_operations()
        test_ndbuffer_broadcast_ops()
        test_scalar_ops()
        test_compare_scalar()
        test_compare_buffer()
        test_buffer_overwrite()
        test_scalar_inplace_update()
        test_ndbuffer_fill()"""
        test_buffer_sum_all()
        test_buffer_sum()
    pass


from testing import assert_true, assert_false


fn test_buffer_sum() raises:
    print("test_buffer_sum")
    alias dtype = DType.int32
    size = 21
    l = List[Scalar[dtype]](capacity=UInt(size))
    for i in range(size):
        l.append(i)

    buffer = Buffer[dtype](l)
    ndb = NDBuffer[dtype](buffer^, Shape(3, 7))
    result = ndb.sum(IntList(0), True)
    assert_true(result.data() == Buffer[dtype]([21, 24, 27, 30, 33, 36, 39]))

    result = ndb.sum(IntList(0), False)
    assert_true(result.data() == Buffer[dtype]([21, 24, 27, 30, 33, 36, 39]))

    result = ndb.sum(IntList(0, 1), True)
    assert_true(result.data() == Buffer[dtype]([210]))

    result = ndb.sum(IntList(1), True)
    assert_true(result.data() == Buffer[dtype]([21, 70, 119]))


fn test_buffer_sum_all() raises:
    print("test_buffer_sum_all")
    alias dtype = DType.int32
    size = 21
    l = List[Scalar[dtype]](capacity=UInt(size))
    for i in range(size):
        l.append(i)

    buffer = Buffer[dtype](l)
    ndb = NDBuffer[dtype](buffer^, Shape(3, 7))

    assert_true(ndb.sum_all() == 210)
    shared = ndb.share(Shape(5, 2), offset=1, strides=Strides(2, 2))
    assert_true(shared.sum_all() == 60)
    # Scalar
    ndb = NDBuffer[dtype](Shape())
    ndb.fill(42)
    assert_true(ndb.sum_all() == 42)
    shared = ndb.share()
    assert_true(
        shared.sum_all() == 42 and shared.item() == 42 and ndb.item() == 42
    )
    # Shape(1)
    ndb = NDBuffer[dtype](Shape(1))
    ndb.fill(39)
    assert_true(ndb.sum_all() == 39 and ndb[IntList(0)] == 39)
    shared = ndb.share()
    assert_true(
        shared.sum_all() == 39 and shared.item() == 39 and ndb.item() == 39
    )


fn test_buffer_overwrite() raises:
    alias dtype = DType.int32
    size = 21
    l = List[Scalar[dtype]](capacity=UInt(size))
    for i in range(size):
        l.append(i)

    buffer = Buffer[dtype](l)
    ndb = NDBuffer[dtype](buffer^, Shape(3, 7))
    result = Buffer[dtype]([42, 42, 42])
    ndb.data().overwrite(result, 3, 6)


fn test_compare_buffer() raises:
    print("test_compare_buffer")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6]), Shape(2, 3))
    ndb2 = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 3, 4, 6]), Shape(2, 3))
    result = ndb.compare[GreaterThan](ndb2)
    assert_true(
        result.data()
        == Buffer[DType.bool]([False, False, False, True, True, False])
    )


fn test_compare_scalar() raises:
    print("test_compare_scalar")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6]), Shape(2, 3))
    result = ndb.compare_scalar[GreaterThan](3)
    assert_true(
        result.data()
        == Buffer[DType.bool]([False, False, False, True, True, True])
    )

    shared = ndb.share(Shape(1, 3), strides=Strides(1, 2), offset=1)
    result = shared.compare_scalar[Equal](4)
    assert_true(result.data() == Buffer[DType.bool]([False, True, False]))


fn test_inplace_broadcast_operations() raises:
    print("test_inplace_broadcast_operations")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6]), Shape(2, 3))
    ndb2 = NDBuffer[dtype](Buffer[dtype]([1, 2, 3]), Shape(3))
    ndb += ndb2
    assert_true(ndb.data() == Buffer[dtype]([2, 4, 6, 5, 7, 9]))

    ndb -= ndb2
    assert_true(ndb.data() == Buffer[dtype]([1, 2, 3, 4, 5, 6]))

    ndb_shared = ndb.share()
    ndb2_shared = ndb2.share()

    ndb_shared += ndb2_shared
    assert_true(ndb.data() == Buffer[dtype]([2, 4, 6, 5, 7, 9]))

    ndb_shared -= ndb2_shared
    assert_true(ndb.data() == Buffer[dtype]([1, 2, 3, 4, 5, 6]))


fn test_inplace_operations() raises:
    print("test_inplace_operations")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](
        Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]), Shape(3, 3)
    )
    ndb2 = NDBuffer[dtype](
        Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]), Shape(3, 3)
    )
    ndb += ndb2
    assert_true(ndb.data() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]) * 2)
    ndb -= ndb2
    assert_true(ndb.data() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    ndb *= ndb2
    assert_true(ndb.data() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]) ** 2)
    ndb /= ndb2
    assert_true(ndb.data() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]))

    shared = ndb.share()

    ndb += ndb2
    assert_true(shared.data() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]) * 2)
    ndb -= ndb2
    assert_true(shared.data() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    ndb *= ndb2
    assert_true(
        shared.data() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]) ** 2
    )
    ndb /= ndb2
    assert_true(shared.data() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]))

    shared2 = ndb.share(Shape(2, 3), offset=3)
    ndb2_shared = ndb2.share(Shape(2, 3))

    shared2 += ndb2_shared

    assert_true(ndb.data() == Buffer[dtype]([1, 2, 3, 5, 7, 9, 11, 13, 15]))
    shared2 -= ndb2_shared

    assert_true(ndb.data() == Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8, 9]))

    shared3 = ndb.share(Shape(1, 3), offset=3, strides=Strides(1, 2))
    shared4 = ndb2.share(Shape(1, 3), strides=Strides(1, 3))

    shared3 += shared4

    assert_true(ndb.data() == Buffer[dtype]([1, 2, 3, 5, 5, 10, 7, 15, 9]))


fn test_unique() raises:
    print("test_unique")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([2, 2, 3, 4, 2, 6]), Shape(2, 3))
    assert_true(ndb.unique().data() == Buffer[dtype]([2, 3, 4, 6]))


fn test_count() raises:
    print("test_count")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([2, 2, 3, 4, 2, 6]), Shape(2, 3))
    assert_true(ndb.count(2) == 3)
    shared = ndb.share()
    assert_true(
        shared.count(2) == 3 and ndb.count(2) == 3 and ndb.count(3) == 1
    )
    share2 = shared.share(Shape(5, 1), offset=1)
    assert_true(share2.count(2) == 2)
    share3 = ndb.share(Shape(2))
    assert_true(share3.count(2) == 2)
    share4 = ndb.share(Shape(1))
    assert_true(share4.count(2) == 1)


fn test_scalar_inplace_update() raises:
    print("test_scalar_inplace_update")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6]), Shape(2, 3))
    ndb.inplace_scalar_ops[Add](99)
    assert_true(ndb.data() == Buffer[dtype]([100, 101, 102, 103, 104, 105]))
    shared = ndb.share(Shape(3, 1), offset=3)
    shared.inplace_scalar_ops[Add](10)
    assert_true(ndb.data() == Buffer[dtype]([100, 101, 102, 113, 114, 115]))

    shared2 = ndb.share(Shape(1, 3), offset=0, strides=Strides(1, 2))
    shared2.inplace_scalar_ops[Add](100)
    assert_true(ndb.data() == Buffer[dtype]([200, 101, 202, 113, 214, 115]))


fn test_element_at() raises:
    print("test_element_at")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6]), Shape(2, 3))
    shared = ndb.share(Shape(3, 1), offset=3)
    assert_true(
        shared.max_index() == 5 and shared.element_at(shared.max_index()) == 6
    )


fn test_scalar_ops() raises:
    print("test_scalar_ops")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6]), Shape(2, 3))
    ndb_shared = ndb.share(Shape(1, 3), offset=3)
    result = ndb_shared.scalar_ops[Add](42)
    assert_true(result.data() == Buffer[dtype]([46, 47, 48]))


fn test_dtype_conversion() raises:
    print("test_dtype_conversion")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6]), Shape(2, 3))
    ndb_shared = ndb.share(Shape(1, 3), offset=3)
    converted = ndb_shared.to_dtype[DType.float64]()

    assert_true(
        converted.data() == Buffer[DType.float64]([4, 5, 6])
        and not converted.shared()
        and converted.strides == Strides(3, 1)
        and converted._contiguous
    )


fn test_equal() raises:
    print("test_equal")
    alias dtype = DType.float32
    ndb1 = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6]), Shape(2, 3))
    ndb1_shared = ndb1.share(Shape(1, 3), offset=3)
    ndb2 = NDBuffer[dtype](Buffer[dtype]([4, 10, 6]), Shape(1, 3))
    result = ndb1_shared.compare[Equal](ndb2)
    assert_true(result.data() == Buffer[DType.bool]([True, False, True]))


fn test_add() raises:
    print("test_add")
    alias dtype = DType.float32
    ndb1 = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6]), Shape(2, 3))
    ndb1_shared = ndb1.share(Shape(1, 3), offset=3)
    ndb2 = NDBuffer[dtype](Buffer[dtype]([10, 20, 30]), Shape(1, 3))

    result = ndb1_shared + ndb2
    assert_true(
        result.data() == Buffer[dtype]([14, 25, 36])
        and result.shared() == False
    )


fn test_zero() raises:
    print("test_zero")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Shape(2, 3))
    ndb.fill(42)
    shared = ndb.share(Shape(3), offset=3)
    shared.zero()
    assert_true(ndb.data() == Buffer[dtype]([42, 42, 42, 0, 0, 0]))


fn test_broadcast_fill() raises:
    print("test_broadcast_fill")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Shape(2, 3))
    filler = NDBuffer[dtype](Shape(2, 1))
    filler.fill(42)
    ndb.fill(filler)
    assert_true(ndb.data() == Buffer[dtype]([42, 42, 42, 42, 42, 42]))

    filler.fill(89)
    shared = filler.share()
    ndb.fill(shared)
    assert_true(ndb.data() == Buffer[dtype]([89, 89, 89, 89, 89, 89]))


fn test_fill_2() raises:
    print("test_fill_2")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Shape(2, 3))
    filler = NDBuffer[dtype](Shape(2, 3))
    filler.fill(91)
    ndb.fill(filler)

    assert_true(ndb.data() == Buffer[dtype].full(91, 6))

    shared1 = ndb.share(Shape(3), offset=3)
    filler = NDBuffer[dtype](Shape(3))
    filler.fill(92)
    shared1.fill(filler)

    assert_true(shared1.data() == Buffer[dtype]([91, 91, 91, 92, 92, 92]))
    assert_true(ndb.data() == Buffer[dtype]([91, 91, 91, 92, 92, 92]))

    # Left contiguous, right non-contiguous
    ndb = NDBuffer[dtype](Shape(2, 2))
    filler = NDBuffer[dtype](Shape(2, 1, 4))
    filler.fill(102)
    filler_shared = filler.share(Shape(2, 2), offset=4)
    ndb.fill(filler_shared)

    assert_true(ndb.data() == Buffer[dtype]([102, 102, 102, 102]))
    # Both shared
    ndb = NDBuffer[dtype](Shape(2, 2))
    filler_shared.fill(31)
    ndb_shared = ndb.share()
    ndb_shared.fill(filler_shared)
    assert_true(ndb.data() == Buffer[dtype]([31, 31, 31, 31]))

    filler = NDBuffer[dtype](Shape(2, 1, 4))
    filler.fill(1919)
    filler_shared = filler.share(Shape(2, 2), strides=Strides(1, 2))
    ndb_shared.fill(filler_shared)

    assert_true(
        ndb.data() == Buffer[dtype]([1919, 1919, 1919, 1919])
        and not filler_shared._contiguous,
    )
    # Left non-contiguous and right contiguous
    filler1 = NDBuffer[dtype](Shape(2, 2))
    filler1.fill(47)

    ndb1 = NDBuffer[dtype](Shape(2, 1, 4))
    ndb1.fill(1)
    ndb_shared1 = ndb1.share(Shape(2, 2), strides=Strides(1, 2), offset=1)
    ndb_shared1.fill(filler1)

    assert_true(ndb1.data() == Buffer[dtype]([1, 47, 47, 47, 47, 1, 1, 1]))

    # left and right No contiguous

    ndb1 = NDBuffer[dtype](
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
    )
    ndb1_shared = ndb1.share(Shape(2, 3), strides=Strides(1, 2), offset=12)

    ndb2 = NDBuffer[dtype](
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160
    )
    ndb2_shared = ndb2.share(Shape(2, 3), strides=Strides(1, 3), offset=0)

    ndb1_shared.fill(ndb2_shared)

    assert_true(
        ndb1.data()
        == Buffer[dtype](
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                10,
                20,
                40,
                50,
                70,
                80,
                18,
                19,
                20,
                21,
                22,
                23,
            ]
        )
    )

    ndb = NDBuffer[dtype](Buffer[dtype]([1, 2, 3, 4, 5, 6, 7, 8]))
    ndb_shared = ndb.share(Shape(3), offset=2)
    ndb_shared.fill(42)
    assert_true(ndb_shared.data() == Buffer[dtype]([1, 2, 42, 42, 42, 6, 7, 8]))


fn test_scalar_buffer() raises:
    print("test_scalar_buffer")
    ndb = NDBuffer[DType.bool]()
    assert_true(ndb.is_scalar())
    ndb.fill(True)
    assert_true(ndb.item() == True)
    assert_true(ndb[IntArray()] == True)


fn test_ndbuffer_fill() raises:
    print("test_ndbuffer_fill")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype](Shape(8))
    ndb.fill(42)
    expected = Buffer[dtype].full(42, 8)
    assert_true(ndb.data() == expected, "NDBuffer fill assertion 1 failed")
    assert_false(ndb.shared(), "NDBuffer not shared assertion failed")
    shared = ndb.share()
    assert_true(ndb.shared(), "NDBuffer shared assertion failed - post sharing")
    shared.fill(91)
    expected = Buffer[dtype].full(91, 8)
    assert_true(ndb.data() == expected, "NDBuffer fill assertion 2 failed")
    share2 = ndb.share(Shape(3), Strides(2), offset=2)
    share2.fill(81)
    var l: List[Scalar[dtype]] = [91, 91, 81, 91, 81, 91, 81, 91]

    expected = Buffer[dtype](l)
    assert_true(
        share2.data() == expected
        and ndb.data() == expected
        and shared.data() == expected,
        "Fill via shape, strides and offset failed",
    )
    ndb = NDBuffer[dtype]()
    filler = NDBuffer[dtype]()
    filler.fill(39)
    ndb.fill(filler)
    assert_true(ndb.item() == 39)

    filler = NDBuffer[dtype](Shape(1))
    filler.fill(42)
    ndb.fill(filler)
    assert_true(ndb.item() == 42)
    shared = ndb.share()

    filler.fill(101)
    shared.fill(filler)

    assert_true(ndb.item() == 101)

    alias _Bool = Scalar[DType.bool]

    _list = List[Scalar[DType.bool]](
        [
            _Bool(True),
            _Bool(True),
            _Bool(True),
            _Bool(True),
            _Bool(True),
            _Bool(True),
            _Bool(True),
            _Bool(True),
            _Bool(True),
        ]
    )
    buff = Buffer[DType.bool](_list.copy())
    ndb_bool = NDBuffer[DType.bool](buff.copy())
    ndb_bool_shared = ndb_bool.share(Shape(5), offset=1)
    ndb_bool_shared.fill(False)
    assert_true(
        ndb_bool.data()
        == Buffer[DType.bool](
            [True, False, False, False, False, False, True, True, True]
        )
    )


fn test_ndbuffer_broadcast_ops() raises:
    print("test_ndbuffer_broadcast_ops")

    alias dtype = DType.float32
    buffer1 = Buffer[dtype]([42, 42, 42, 42, 42, 42])
    shape1 = Shape(2, 3)
    ndbuffer1 = NDBuffer[dtype](buffer1^, shape1)

    buffer2 = Buffer[dtype]([3, 3, 3])
    shape2 = Shape(3)
    ndbuffer2 = NDBuffer[dtype](buffer2^, shape2)

    result = ndbuffer1.arithmetic_ops[Add](ndbuffer2)
    assert_true(result.data() == (Buffer[dtype]([42, 42, 42, 42, 42, 42]) + 3))

    result = result.arithmetic_ops[Subtract](ndbuffer2)
    assert_true(result.data() == Buffer[dtype]([42, 42, 42, 42, 42, 42]))


fn test_ndbuffer_inplace_ops() raises:
    print("test_ndbuffer_inplace_ops")

    alias dtype = DType.float32
    buffer1 = Buffer[dtype](30)
    buffer1.fill(42)
    shape = Shape(5, 6)
    ndbuffer1 = NDBuffer[dtype](buffer1^, shape, None)
    index1 = IntArray(2)
    index1[0] = 4
    index1[1] = 5
    assert_true(ndbuffer1[index1] == 42, "NDBuffer get failed")

    buffer2 = Buffer[dtype](30)
    buffer2.fill(24)
    shape1 = Shape(5, 6)
    ndbuffer2 = NDBuffer[dtype](buffer2^, shape1, None)

    _shared = ndbuffer1.share(shape1)
    ndbuffer1 += ndbuffer2
    # ndbuffer1.__iadd__[check_contiguity=False](ndbuffer2)

    expected = Buffer[dtype].full(66, 30)

    assert_true(
        ndbuffer1.data() == expected, "In place add failed for NDBuffer"
    )

    shared_buffer = ndbuffer1.share(shape1)
    assert_true(shared_buffer.data() == expected, "NDBuffer sharing failed")
    assert_true(ndbuffer1.shared(), "NDBuffer buffer nullification failed")


fn test_ndbuffer_set_get() raises:
    print("test_ndbuffer_set_get")

    alias dtype = DType.float32
    buffer = Buffer[dtype](1)
    buffer[0] = 42
    shape = Shape()
    ndbuffer = NDBuffer[dtype](buffer.copy(), shape, None)
    assert_true(ndbuffer[IntArray()] == 42, "NDBuffer get failed")
    ndbuffer[IntArray()] = 97
    assert_true(ndbuffer[IntArray()] == 97, "NDBuffer get failed post update")
    assert_true(ndbuffer.item() == 97, "NDBuffer item() failed post update")
    assert_true(ndbuffer.is_scalar(), "NDBuffer is_scalar check failed")
