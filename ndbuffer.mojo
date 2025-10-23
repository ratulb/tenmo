from shapes import Shape
from strides import Strides
from buffers import Buffer
from layout.int_tuple import IntArray
from indexhelper import IndexCalculator
from broadcasthelper import ShapeBroadcaster
from common_utils import panic, IntArrayHelper
from sys import simd_width_of
from algorithm import vectorize
from memory import memcpy, ArcPointer
from operators import (
    Multiply,
    Add,
    Subtract,
    Divide,
    Equal,
    NotEqual,
    LessThan,
    LessThanEqual,
    GreaterThan,
    GreaterThanEqual,
)


struct NDBuffer[dtype: DType](Copyable & Movable):
    var shape: Shape
    var strides: Strides
    var offset: Int
    var buffer: Optional[Buffer[dtype]]
    var shared_buffer: Optional[ArcPointer[Buffer[dtype]]]
    var contiguous: Bool

    fn __init__(
        out self,
    ):
        self.buffer = Buffer[dtype](Shape.Void().num_elements())
        self.shared_buffer = None
        self.shape = Shape.Void()
        self.strides = Strides.Zero()
        self.offset = 0
        self.contiguous = True

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
        self.contiguous = False
        self.contiguous = self.is_contiguous()

    fn __init__(
        out self,
        shape: Shape,
        strides: Optional[Strides] = None,
        offset: Int = 0,
    ):
        self.buffer = Optional(Buffer[dtype](shape.num_elements()))
        self.shared_buffer = None
        self.shape = shape.copy()
        self.strides = strides.value() if strides else Strides.default(shape)
        self.offset = offset
        self.contiguous = False
        self.contiguous = self.is_contiguous()

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
        self.contiguous = False
        self.contiguous = self.is_contiguous()

    fn __moveinit__(out self, deinit other: Self):
        self.buffer = other.buffer^
        self.shared_buffer = other.shared_buffer^
        self.shape = other.shape^
        self.strides = other.strides^
        self.offset = other.offset
        self.contiguous = other.contiguous

    fn __copyinit__(out self, other: Self):
        self.buffer = other.buffer.copy()
        self.shared_buffer = other.shared_buffer.copy()
        self.shape = other.shape.copy()
        self.strides = other.strides.copy()
        self.offset = other.offset
        self.contiguous = other.contiguous

    fn detach(self) -> NDBuffer[dtype]:
        out = NDBuffer[dtype]()
        out.buffer = self.buffer.copy()
        out.shared_buffer = Optional(
            self.data().copy().shared()
        ) if self.shared() else None
        out.shape = self.shape.copy()
        out.strides = self.strides.copy()
        out.offset = self.offset
        out.contiguous = self.contiguous
        return out^

    fn __del__(deinit self):
        _ = self.buffer^
        _ = self.shared_buffer^
        _ = self.shape^
        _ = self.strides^

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

    fn item(self) -> Scalar[dtype]:
        if self.shape != Shape(1) and self.shape != Shape():
            panic(
                "NDBuffer → item(self): only valid for zero dim buffer,"
                " got shape: "
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
    fn __imul__[
        validate_shape: Bool = True, check_contiguity: Bool = True
    ](self, other: NDBuffer[dtype]):
        self.inplace_ops[Multiply, validate_shape, check_contiguity](other)

    @always_inline
    fn __iadd__[
        validate_shape: Bool = True, check_contiguity: Bool = True
    ](self, other: NDBuffer[dtype]):
        self.inplace_ops[Add, validate_shape, check_contiguity](other)

    @always_inline
    fn __isub__[
        validate_shape: Bool = True, check_contiguity: Bool = True
    ](self, other: NDBuffer[dtype]):
        self.inplace_ops[Subtract, validate_shape, check_contiguity](other)

    @always_inline
    fn shared(self) -> Bool:
        return self.buffer is None and self.shared_buffer is not None

    @staticmethod
    fn share(
        source: UnsafePointer[Self],
        shape: Shape,
        strides: Optional[Strides] = None,
        offset: Int = 0,
    ) -> NDBuffer[dtype]:
        shared = source[].shared()
        ref buffer = source[].buffer
        ref shared_buffer = source[].shared_buffer

        if not shared:
            if buffer is None:
                buffer = Optional(Buffer[dtype]())
            shared_buffer = Optional(buffer.unsafe_take().shared())

        return NDBuffer[dtype](shared_buffer, shape, strides, offset)

    fn share(
        mut self,
        shape: Optional[Shape] = None,
        strides: Optional[Strides] = None,
        offset: Int = 0,
    ) -> NDBuffer[dtype]:
        if not self.shared():
            if self.buffer is None:
                self.buffer = Optional(Buffer[dtype]())
            self.shared_buffer = Optional(self.buffer.unsafe_take().shared())

        new_shape = shape.value() if shape else self.shape
        return NDBuffer[dtype](self.shared_buffer, new_shape, strides, offset)

    @always_inline
    fn size(self) -> Int:
        return self.shape.num_elements()

    @always_inline
    fn zero(self):
        self.fill(Scalar[dtype](0))

    @always_inline
    fn fill(self, value: Scalar[dtype]):
        if self.contiguous:
            if not self.shared():
                self.data().fill(value)
            else:
                var start = self.offset
                var end = start + self.numels()
                ref buffer = self.shared_buffer.value()[]
                for i in range(start, end):
                    buffer[i] = value
        else:
            for coord in self.shape:
                self[coord] = value

    @always_inline
    fn __is__(self, other: NDBuffer[dtype]) -> Bool:
        if self.shared() and other.shared():
            return self.shared_buffer.value() is other.shared_buffer.value()
        return False

    @always_inline
    fn fill(lhs, other: NDBuffer[dtype]):
        var src_shape = other.shape
        var target_shape = lhs.shape
        if not ShapeBroadcaster.broadcastable(src_shape, target_shape):
            panic(
                "NDBuffer → fill(lhs, other): dimension mismatch: lhs shape",
                target_shape.__str__(),
                "≠",
                "other shape",
                src_shape.__str__(),
            )
        if other.is_scalar() or src_shape == Shape.Unit():
            lhs.fill(
                other.item()
            )  # Scalar/Singleton NDBuffer - shared or otherwise
            return
        # Detach if same storage
        rhs = other.detach() if other is lhs else other.copy()
        if lhs.numels() == rhs.numels():
            if lhs.contiguous and rhs.contiguous:
                count = lhs.size()
                ref dest = lhs.data()
                ref src = rhs.data()
                offset_dest = lhs.offset
                offset_src = rhs.offset

                memcpy(
                    dest=dest.data + offset_dest,
                    src=src.data + offset_src,
                    count=count,
                )
            elif lhs.contiguous and not rhs.contiguous:
                index = 0
                offset = lhs.offset
                ref dest = lhs.data()
                for coord in src_shape:
                    dest[index + offset] = rhs[coord]
                    index += 1

            elif not lhs.contiguous and rhs.contiguous:
                index = 0
                offset = rhs.offset
                ref src = rhs.data()
                for coord in target_shape:
                    lhs[coord] = src[index + offset]
                    index += 1

            else:
                for coord in target_shape:
                    lhs[coord] = rhs[coord]

        else:  # Handle broadcast
            mask = ShapeBroadcaster.broadcast_mask(src_shape, target_shape)
            for coord in target_shape:
                src_coord = ShapeBroadcaster.translate_index(
                    src_shape, coord, mask, target_shape
                )
                lhs[coord] = rhs[src_coord]

    @always_inline
    fn buffer_scalar_arithmetic_ops[
        opcode: Int
    ](lhs: NDBuffer[dtype], scalar: Scalar[dtype]) -> NDBuffer[dtype]:
        var buffer: Buffer[dtype]

        if lhs.contiguous:
            var lhs_buffer: Buffer[dtype]
            if not lhs.shared():
                lhs_buffer = lhs.data().copy()
            else:
                offset = lhs.offset
                numels = lhs.numels()
                lhs_buffer = lhs.data()[offset : offset + numels]

            @parameter
            if opcode == Multiply:
                buffer = lhs_buffer * scalar

            elif opcode == Add:
                buffer = lhs_buffer + scalar

            elif opcode == Subtract:
                buffer = lhs_buffer - scalar

            elif opcode == Divide:
                buffer = lhs_buffer / scalar

            else:
                buffer = scalar / lhs_buffer
        else:
            buffer = Buffer[dtype](lhs.numels())
            var index = 0
            for coord in lhs.shape:

                @parameter
                if opcode == Multiply:
                    buffer[index] = lhs[coord] * scalar

                elif opcode == Add:
                    buffer[index] = lhs[coord] + scalar

                elif opcode == Subtract:
                    buffer[index] = lhs[coord] - scalar

                elif opcode == Divide:
                    buffer[index] = lhs[coord] / scalar

                else:
                    buffer[index] = scalar / lhs[coord]

                index += 1

        return NDBuffer[dtype](buffer^, lhs.shape)

    @always_inline
    fn inplace_ops[
        opcode: Int,
        validate_shape: Bool = True,
        check_contiguity: Bool = True,
        simd_width: Int = simd_width_of[dtype](),
    ](lhs: NDBuffer[dtype], rhs: NDBuffer[dtype]):
        @parameter
        if validate_shape:
            if lhs.shape != rhs.shape:
                panic(
                    "NDBuffer → inplace_ops(lhs, rhs): dimension mismatch: "
                    + lhs.shape.__str__()
                    + ", "
                    + rhs.shape.__str__()
                    + ", opcode = "
                    + opcode.__str__()
                )

        @parameter
        fn update_inplace[width: Int](idx: Int):
            var vec_rhs = rhs.data().load[width](idx)
            var vec_lhs = lhs.data().load[width](idx)

            @parameter
            if opcode == Multiply:
                lhs.data().store[width](idx, vec_rhs * vec_lhs)

            @parameter
            if opcode == Add:
                lhs.data().store[width](idx, vec_rhs + vec_lhs)

            @parameter
            if opcode == Subtract:
                lhs.data().store[width](idx, vec_lhs - vec_rhs)

            @parameter
            if opcode == Divide:
                lhs.data().store[width](idx, vec_lhs / vec_rhs)

        @parameter
        if not check_contiguity:
            vectorize[update_inplace, simd_width](lhs.size())

        @parameter
        if check_contiguity:
            if lhs.contiguous and rhs.contiguous:
                if not lhs.shared() and not rhs.shared():
                    vectorize[update_inplace, simd_width](lhs.size())
                else:
                    numels = lhs.numels()
                    lhs_offset = lhs.offset
                    rhs_offset = rhs.offset

                    @parameter
                    fn update_segment[width: Int](idx: Int):
                        lhs_segment = lhs.data().load[width](lhs_offset + idx)
                        rhs_segment = rhs.data().load[width](rhs_offset + idx)

                        @parameter
                        if opcode == Multiply:
                            lhs.data().store[width](
                                lhs_offset + idx, lhs_segment * rhs_segment
                            )

                        @parameter
                        if opcode == Add:
                            lhs.data().store[width](
                                lhs_offset + idx, lhs_segment + rhs_segment
                            )

                        @parameter
                        if opcode == Subtract:
                            lhs.data().store[width](
                                lhs_offset + idx, lhs_segment - rhs_segment
                            )

                        @parameter
                        if opcode == Divide:
                            lhs.data().store[width](
                                lhs_offset + idx, lhs_segment / rhs_segment
                            )

                    vectorize[update_segment, simd_width](numels)

            else:
                for coord in lhs.shape:

                    @parameter
                    if opcode == Multiply:
                        lhs[coord] = lhs[coord] * rhs[coord]

                    @parameter
                    if opcode == Add:
                        lhs[coord] = lhs[coord] + rhs[coord]

                    @parameter
                    if opcode == Subtract:
                        lhs[coord] = lhs[coord] - rhs[coord]

                    @parameter
                    if (
                        opcode != Multiply
                        and opcode != Add
                        and opcode != Subtract
                    ):
                        panic(
                            "Unknown opcode:[",
                            opcode.__str__(),
                            "] in NDBuffer inplace_ops",
                        )

    @always_inline
    fn __add__[
        validate_shape: Bool = True,
        check_contiguity: Bool = True,
        broadcast: Bool = True,
    ](self, other: NDBuffer[dtype]) -> NDBuffer[dtype]:
        return self.buffer_arithmetic_ops[
            Add, validate_shape, check_contiguity, broadcast
        ](other)

    @always_inline
    fn __mul__[
        validate_shape: Bool = True,
        check_contiguity: Bool = True,
        broadcast: Bool = True,
    ](self, other: NDBuffer[dtype]) -> NDBuffer[dtype]:
        return self.buffer_arithmetic_ops[
            Multiply, validate_shape, check_contiguity, broadcast
        ](other)

    @always_inline
    fn __sub__[
        validate_shape: Bool = True,
        check_contiguity: Bool = True,
        broadcast: Bool = True,
    ](self, other: NDBuffer[dtype]) -> NDBuffer[dtype]:
        return self.buffer_arithmetic_ops[
            Subtract, validate_shape, check_contiguity, broadcast
        ](other)

    @always_inline
    fn __truediv__[
        validate_shape: Bool = True,
        check_contiguity: Bool = True,
        broadcast: Bool = True,
    ](self, other: NDBuffer[dtype]) -> NDBuffer[dtype]:
        return self.buffer_arithmetic_ops[
            Divide, validate_shape, check_contiguity, broadcast
        ](other)

    @always_inline
    fn buffer_arithmetic_ops[
        opcode: Int,
        validate_shape: Bool = True,
        check_contiguity: Bool = True,
        broadcast: Bool = True,
        simd_width: Int = simd_width_of[dtype](),
    ](lhs: NDBuffer[dtype], rhs: NDBuffer[dtype]) -> NDBuffer[dtype]:
        @parameter
        if validate_shape:
            if not ShapeBroadcaster.broadcastable(lhs.shape, rhs.shape):
                panic(
                    "NDBuffer → buffer_arithmetic_ops(lhs, rhs): dimension"
                    " mismatch: "
                    + lhs.shape.__str__()
                    + ", "
                    + rhs.shape.__str__()
                    + ", opcode = "
                    + opcode.__str__()
                )

        @parameter
        if broadcast:
            if lhs.shape != rhs.shape:
                return lhs.broadcast_buffer[opcode](rhs)

        var buffer = Buffer[dtype](lhs.numels())

        @parameter
        fn buffer_arithmetic_op_fn[width: Int](idx: Int):
            var vec_rhs = rhs.data().load[width](idx)
            var vec_lhs = lhs.data().load[width](idx)

            @parameter
            if opcode == Multiply:
                buffer.store[width](idx, vec_rhs * vec_lhs)

            @parameter
            if opcode == Add:
                buffer.store[width](idx, vec_rhs + vec_lhs)

            @parameter
            if opcode == Subtract:
                buffer.store[width](idx, vec_lhs - vec_rhs)

            @parameter
            if opcode == Divide:
                buffer.store[width](idx, vec_lhs / vec_rhs)

            @parameter
            if (
                opcode != Multiply
                and opcode != Add
                and opcode != Subtract
                and opcode != Divide
            ):
                panic(
                    "Unknown opcode:[",
                    opcode.__str__(),
                    "] in NDBuffer buffer_arithmetic_ops",
                )

        @parameter
        if not check_contiguity:
            vectorize[buffer_arithmetic_op_fn, simd_width](lhs.size())

        @parameter
        if check_contiguity:
            if lhs.contiguous and rhs.contiguous:
                if not lhs.shared() and not rhs.shared():
                    vectorize[buffer_arithmetic_op_fn, simd_width](lhs.size())
                    return NDBuffer[dtype](buffer^, lhs.shape)
                else:
                    numels = lhs.numels()
                    lhs_offset = lhs.offset
                    rhs_offset = rhs.offset
                    var lhs_buffer = lhs.data()[
                        lhs_offset : lhs_offset + numels
                    ]
                    var rhs_buffer = rhs.data()[
                        rhs_offset : rhs_offset + numels
                    ]
                    var buffer: Buffer[dtype]

                    @parameter
                    if opcode == Multiply:
                        buffer = lhs_buffer * rhs_buffer

                    elif opcode == Add:
                        buffer = lhs_buffer + rhs_buffer

                    elif opcode == Subtract:
                        buffer = lhs_buffer - rhs_buffer

                    else:
                        buffer = lhs_buffer / rhs_buffer

                    return NDBuffer[dtype](buffer^, lhs.shape)

            else:
                var index = 0
                for coord in lhs.shape:

                    @parameter
                    if opcode == Multiply:
                        buffer[index] = lhs[coord] * rhs[coord]

                    @parameter
                    if opcode == Add:
                        buffer[index] = lhs[coord] + rhs[coord]

                    @parameter
                    if opcode == Subtract:
                        buffer[index] = lhs[coord] - rhs[coord]

                    @parameter
                    if opcode == Divide:
                        buffer[index] = lhs[coord] / rhs[coord]

                    @parameter
                    if (
                        opcode != Multiply
                        and opcode != Add
                        and opcode != Subtract
                        and opcode != Divide
                    ):
                        panic(
                            "Unknown opcode:[",
                            opcode.__str__(),
                            "] in NDBuffer buffer_arithmetic_ops",
                        )
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
    fn compare[
        opcode: Int,
        validate_shape: Bool = True,
        simd_width: Int = simd_width_of[dtype](),
    ](lhs: NDBuffer[dtype], rhs: NDBuffer[dtype]) -> NDBuffer[DType.bool]:
        @parameter
        if validate_shape:
            if not lhs.shape == rhs.shape:
                panic(
                    "NDBuffer → compare(lhs, rhs): dimension mismatch: "
                    + lhs.shape.__str__()
                    + "≠"
                    + rhs.shape.__str__(),
                    "opcode: " + opcode.__str__(),
                )

        if lhs.contiguous and rhs.contiguous:
            var lhs_buffer: Buffer[dtype]
            var rhs_buffer: Buffer[dtype]
            if not lhs.shared() and not rhs.shared():
                lhs_buffer = lhs.data().copy()
                rhs_buffer = rhs.data().copy()
            else:
                numels = lhs.numels()
                lhs_offset = lhs.offset
                rhs_offset = rhs.offset
                lhs_buffer = lhs.data()[lhs_offset : lhs_offset + numels]
                rhs_buffer = rhs.data()[rhs_offset : rhs_offset + numels]

            @parameter
            if opcode == Equal:
                buffer = lhs_buffer.eq[simd_width](rhs_buffer)

            elif opcode == NotEqual:
                buffer = lhs_buffer.ne[simd_width](rhs_buffer)

            elif opcode == LessThan:
                buffer = lhs_buffer.lt[simd_width](rhs_buffer)

            elif opcode == LessThanEqual:
                buffer = lhs_buffer.le[simd_width](rhs_buffer)

            elif opcode == GreaterThan:
                buffer = lhs_buffer.gt[simd_width](rhs_buffer)

            else:  # opcode == GreaterThanEqual:
                buffer = lhs_buffer.ge[simd_width](rhs_buffer)

            return NDBuffer[DType.bool](buffer^, lhs.shape)

        else:
            var index = 0
            var buffer = Buffer[DType.bool](lhs.numels())
            for coord in lhs.shape:

                @parameter
                if opcode == Equal:
                    buffer[index] = lhs[coord] == rhs[coord]

                elif opcode == NotEqual:
                    buffer[index] = lhs[coord] != rhs[coord]

                elif opcode == LessThan:
                    buffer[index] = lhs[coord] < rhs[coord]

                elif opcode == LessThanEqual:
                    buffer[index] = lhs[coord] <= rhs[coord]

                elif opcode == GreaterThan:
                    buffer[index] = lhs[coord] > rhs[coord]

                else:
                    buffer[index] = lhs[coord] >= rhs[coord]

                index += 1

            return NDBuffer[DType.bool](buffer^, lhs.shape)


fn main() raises:
    var runs = 1
    for _ in range(runs):
        _ = """test_ndbuffer_set_get()
        test_ndbuffer_inplace_ops()
        test_ndbuffer_broadcast_ops()
        test_is()
        test_ndbuffer_fill()
        test_detach()
        test_scalar_buffer()
        test_fill_2()
        test_broadcast_fill()
        test_zero()
        test_add()"""
        test_equal()


from testing import assert_true, assert_false

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
    filler_shared = filler.share(Shape(2, 2), strides=Strides.of(1, 2))
    ndb_shared.fill(filler_shared)

    assert_true(
        ndb.data() == Buffer[dtype]([1919, 1919, 1919, 1919])
        and not filler_shared.contiguous,
    )
    # Left non-contiguous and right contiguous
    filler1 = NDBuffer[dtype](Shape(2, 2))
    filler1.fill(47)

    ndb1 = NDBuffer[dtype](Shape(2, 1, 4))
    ndb1.fill(1)
    ndb_shared1 = ndb1.share(Shape(2, 2), strides=Strides.of(1, 2), offset=1)
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
    ndb1_shared = ndb1.share(Shape(2, 3), strides=Strides.of(1, 2), offset=12)

    ndb2 = NDBuffer[dtype](
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160
    )
    ndb2_shared = ndb2.share(Shape(2, 3), strides=Strides.of(1, 3), offset=0)

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


fn test_scalar_buffer() raises:
    print("test_scalar_buffer")
    ndb = NDBuffer[DType.bool]()
    assert_true(ndb.is_scalar())
    ndb.fill(True)
    assert_true(ndb.item() == True)
    assert_true(ndb[IntArray()] == True)


fn test_detach() raises:
    print("test_detach")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype]()
    ndb.fill(42)
    detached = ndb.detach()
    assert_false(
        ndb is detached,
        "Unshared NDBuffer is check False assertion 1 failed for detach buffer",
    )
    shared = ndb.share()
    shared_detached = shared.detach()
    assert_false(
        shared is shared_detached,
        "Shared buffer detach __is__ entirely different assertion failed",
    )


fn test_is() raises:
    print("test_is")
    alias dtype = DType.float32
    ndb = NDBuffer[dtype]()
    ndb.fill(42)
    assert_true(
        ndb.size() == 1 and ndb.is_scalar(), "Unit size buffer assertion failed"
    )
    copied = ndb.copy()
    assert_false(
        ndb is copied, "Unshared NDBuffer is check False assertion 1 failed"
    )
    assert_false(
        copied is ndb, "Unshared NDBuffer is check False assertion 2 failed"
    )
    copied = ndb.share()
    assert_true(copied is ndb, "Shared NDBuffer is check True assertion failed")

    new_ndb = NDBuffer[dtype]()
    new_ndb_shared = new_ndb.share()
    assert_true(
        new_ndb is new_ndb_shared,
        "Empty NDBuffer sharing and __is__ True assertion failed",
    )


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
    share2 = ndb.share(Shape(3), Strides.of(2), offset=2)
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


fn test_ndbuffer_broadcast_ops() raises:
    print("test_ndbuffer_broadcast_ops")

    alias dtype = DType.float32
    buffer1 = Buffer[dtype](6)
    buffer1.fill(42)
    shape1 = Shape(2, 3)
    ndbuffer1 = NDBuffer[dtype](buffer1^, shape1, None)

    buffer2 = Buffer[dtype](3)
    buffer2.fill(3)
    shape2 = Shape(3)
    ndbuffer2 = NDBuffer[dtype](buffer2^, shape2, None)

    _result = ndbuffer1.buffer_arithmetic_ops[Add](ndbuffer2)

    buffer3 = Buffer[dtype](1)
    buffer3.fill(-3)
    shape3 = Shape()
    ndbuffer3 = NDBuffer[dtype](buffer3^, shape3, None)

    _result2 = ndbuffer1.buffer_arithmetic_ops[Add](ndbuffer3)


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

    _shared = NDBuffer[dtype].share(UnsafePointer(to=ndbuffer1), shape1)
    ndbuffer1 += ndbuffer2
    # ndbuffer1.__iadd__[check_contiguity=False](ndbuffer2)

    expected = Buffer[dtype].full(66, 30)

    assert_true(
        ndbuffer1.data() == expected, "In place add failed for NDBuffer"
    )

    shared_buffer = NDBuffer[dtype].share(UnsafePointer(to=ndbuffer1), shape1)
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
