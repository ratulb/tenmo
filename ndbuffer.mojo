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
        self.shape = shape
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
    fn size(self) -> Int:
        if self.buffer:
            return self.buffer.value().size
        else:
            return self.shared_buffer.value()[].size

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

        new_shape = shape.or_else(self.shape)
        new_strides = strides.or_else(self.strides)
        return NDBuffer[dtype](
            shared_buffer=self.shared_buffer,
            shape=new_shape,
            strides=new_strides,
            offset=offset,
        )

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
    fn contiguous(self, new_shape: Optional[Shape] = None) -> NDBuffer[dtype]:
        shape = new_shape.value() if new_shape else self.shape
        return NDBuffer[dtype](self.contiguous_buffer(), shape)

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

    @always_inline
    fn sum_over_broadcasted_axes(
        extended_buffer: NDBuffer[dtype], target_shape: Shape
    ) -> NDBuffer[dtype]:
        result = extended_buffer.contiguous()
        current_shape = extended_buffer.shape

        # Sum over extra leading dimensions
        while len(current_shape) > len(target_shape):
            result = result.sum(reduction_axes=IntList(0), keepdims=False)
            current_shape = result.shape

        # Sum over mismatched dimensions
        for i in range(len(target_shape)):
            if current_shape[i] != target_shape[i] and current_shape[i] > 1:
                result = result.sum(reduction_axes=IntList(i), keepdims=True)
                current_shape = result.shape
        return result^


fn main() raises:
    pass
