from algorithm import vectorize
from sys import simdwidthof
from memory import memset_zero, memcpy
from math import exp

# from runtime.asyncrt import num_physical_cores
from sys import num_logical_cores, num_physical_cores
from common_utils import log_debug, panic


struct Buffer[dtype: DType = DType.float32](
    Copyable & Movable & Sized & Stringable & Writable & Representable & Absable
):
    var size: Int
    var data: UnsafePointer[Scalar[dtype]]
    alias Empty = Buffer[dtype]()

    fn __init__(out self):
        self.size = 0
        self.data = UnsafePointer[Scalar[dtype]]()

    fn __init__(out self, size: Int):
        self.data = UnsafePointer[Scalar[dtype]].alloc(size)
        self.size = size

    fn __init__(out self, size: Int, data: UnsafePointer[Scalar[dtype]]):
        self.size = size
        self.data = data

    fn __moveinit__(out self, var other: Self):
        self.size = other.size
        self.data = UnsafePointer[Scalar[dtype]].alloc(other.size)
        memcpy(self.data, other.data, other.size)

    fn __copyinit__(out self, other: Self):
        self.size = other.size
        self.data = UnsafePointer[Scalar[dtype]].alloc(other.size)
        memcpy(self.data, other.data, other.size)

    fn clone(self) -> Buffer[dtype]:
        data = UnsafePointer[Scalar[dtype]].alloc(self.size)
        memcpy(data, self.data, self.size)
        clone = Buffer[dtype](self.size, data)
        return clone

    fn __len__(self) -> Int:
        return self.size

    fn __getitem__(self, slice: Slice) -> Buffer[dtype]:
        var start, end, step = slice.indices(len(self))
        var spread = range(start, end, step)

        if not len(spread):
            return Buffer[dtype].Empty

        # Calculate the correct size based on the actual number of elements
        var result_size = len(spread)
        var result = Buffer[dtype](result_size)

        # Use consecutive indices for the result buffer
        var result_index = 0
        for i in spread:
            result[result_index] = self[
                i
            ]  # Copy the element from source to result
            result_index += 1

        return result^

    fn __getitem__(self, index: Int) -> Scalar[dtype]:
        return self.data.load[width=1, volatile=True](index)

    fn __setitem__(self, index: Int, scalar: Scalar[dtype]):
        self.data.store[width=1, volatile=True](index, scalar)

    @always_inline
    fn load[simdwidth: Int = 1](self, offset: Int) -> SIMD[dtype, simdwidth]:
        return self.data.load[width=simdwidth, volatile=True](offset)

    @always_inline
    fn store[
        simdwidth: Int = 1
    ](self, offset: Int, values: SIMD[dtype, simdwidth]):
        self.data.store[width=simdwidth, volatile=True](offset, values)

    fn __add__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[dtype]:
        out = Buffer[dtype](lhs.size)

        @parameter
        fn add_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) + rhs.load[simdwidth](idx)),
            )

        vectorize[add_elems, simd_width](lhs.size)
        return out

    fn __iadd__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]):
        @parameter
        fn inplace_add_elems[simdwidth: Int](idx: Int):
            lhs.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) + rhs.load[simdwidth](idx)),
            )

        vectorize[inplace_add_elems, simd_width](lhs.size)

    fn __isub__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]):
        @parameter
        fn inplace_sub_elems[simdwidth: Int](idx: Int):
            lhs.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) - rhs.load[simdwidth](idx)),
            )

        vectorize[inplace_sub_elems, simd_width](lhs.size)

    fn __sub__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[dtype]:
        out = Buffer[dtype](lhs.size)

        @parameter
        fn subtract_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) - rhs.load[simdwidth](idx)),
            )

        vectorize[subtract_elems, simd_width](lhs.size)
        return out

    fn __mul__(
        lhs: Buffer[DType.bool], rhs: Buffer[DType.bool]
    ) -> Buffer[DType.bool]:
        if not lhs.size == rhs.size:
            panic(
                (
                    "Buffer → __mul__(Buffer[DType.bool]: buffer size does not"
                    " match -> lhs:"
                ),
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size
        out = Buffer[DType.bool](total)
        _ = """@parameter
        fn mul_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) * rhs.load[simdwidth](idx)),
            )

        vectorize[mul_elems, simdwidthof[DType.bool]()](lhs.size)"""
        alias simd_width = simdwidthof[DType.bool]()
        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx) * rhs.load[simd_width](idx)
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
            # out.store[simd_width](idx, cmp)
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, lhs.load(k) == rhs.load(k))
        return out

    fn __mul__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[dtype]:
        out = Buffer[dtype](lhs.size)

        @parameter
        fn mul_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) * rhs.load[simdwidth](idx)),
            )

        vectorize[mul_elems, simd_width](lhs.size)
        return out

    fn __imul__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]):
        @parameter
        fn inplace_mul_elems[simdwidth: Int](idx: Int):
            lhs.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) * rhs.load[simdwidth](idx)),
            )

        vectorize[inplace_mul_elems, simd_width](lhs.size)

    fn __imul__(lhs: Buffer[DType.bool], rhs: Buffer[DType.bool]):
        if not lhs.size == rhs.size:
            panic(
                (
                    "Buffer → __imul__(Buffer[DType.bool]: buffer size does not"
                    " match -> lhs:"
                ),
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )

        _ = """@parameter
        fn inplace_mul_elems[simdwidth: Int](idx: Int):
            lhs.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) * rhs.load[simdwidth](idx)),
            )

        vectorize[inplace_mul_elems, simd_width](lhs.size)"""
        total = lhs.size
        alias simd_width: Int = simdwidthof[dtype]()
        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx) * rhs.load[simd_width](idx)
            for k in range(simd_width):
                lhs.store[simd_width](idx + k, cmp[idx + k])
            # out.store[simd_width](idx, cmp)
        i = simd_blocks * simd_width

        for k in range(i, total):
            lhs.store(k, lhs.load(k) == rhs.load(k))

    fn __radd__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype]:
        return this.__add__(scalar)

    fn __add__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype]:
        var out = Buffer[dtype](this.size)

        @parameter
        fn add_scalar[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, this.load[simdwidth](idx) + scalar)

        vectorize[add_scalar, simd_width](this.size)
        return out

    fn __iadd__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]):
        @parameter
        fn inplace_add_scalar[simdwidth: Int](idx: Int):
            this.store[simdwidth](idx, this.load[simdwidth](idx) + scalar)

        vectorize[inplace_add_scalar, simd_width](this.size)

    fn __rsub__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype]:
        var out = Buffer[dtype](this.size)

        @parameter
        fn sub_from_scalar[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, scalar - this.load[simdwidth](idx))

        vectorize[sub_from_scalar, simd_width](this.size)
        return out

    fn __sub__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype]:
        var out = Buffer[dtype](this.size)

        @parameter
        fn sub_scalar[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, this.load[simdwidth](idx) - scalar)

        vectorize[sub_scalar, simd_width](this.size)
        return out

    fn __rmul__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], factor: Scalar[dtype]) -> Buffer[dtype]:
        return this.__mul__(factor)

    fn __mul__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], factor: Scalar[dtype]) -> Buffer[dtype]:
        var out = Buffer[dtype](this.size)

        @parameter
        fn mul_by_factor[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, this.load[simdwidth](idx) * factor)

        vectorize[mul_by_factor, simd_width](this.size)
        return out

    fn __pow__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], exponent: Scalar[dtype]) -> Buffer[dtype]:
        var out = Buffer[dtype](this.size)

        @parameter
        fn raise_to_power[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx, this.load[simdwidth](idx).__pow__(exponent)
            )

        vectorize[raise_to_power, simd_width](this.size)
        return out

    fn __truediv__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], divisor: Scalar[dtype]) -> Buffer[dtype]:
        var out = Buffer[dtype](this.size)

        @parameter
        fn divide_by_divisor[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx, this.load[simdwidth](idx).__truediv__(divisor)
            )

        vectorize[divide_by_divisor, simd_width](this.size)
        return out

    fn __rtruediv__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype]:
        var out = Buffer[dtype](this.size)

        @parameter
        fn divide_scalar[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx, this.load[simdwidth](idx).__rtruediv__(scalar)
            )

        vectorize[divide_scalar, simd_width](this.size)
        return out

    fn __abs__(self) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → __abs__ is for numeric data types only",
        ]()
        total = self.size
        out = Buffer[dtype](total)

        @parameter
        fn absolute_value[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, self.load[simdwidth](idx).__abs__())

        vectorize[absolute_value, simdwidthof[dtype]()](total)
        return out

    fn exp(self) -> Buffer[dtype]:
        total = self.size
        out = Buffer[dtype](total)

        @parameter
        fn exp_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, exp(self.load[simdwidth](idx)))

        vectorize[exp_elems, simdwidthof[dtype]()](total)
        return out

    @staticmethod
    fn full[
        simd_width: Int = simdwidthof[dtype]()
    ](value: Scalar[dtype], size: Int) -> Buffer[dtype]:
        buffer = Buffer[dtype](size)
        buffer.fill[simd_width](value)
        return buffer

    fn fill[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], value: Scalar[dtype]):
        @parameter
        fn set_scalar[simdwidth: Int](idx: Int):
            this.store[simdwidth](idx, value)

        vectorize[set_scalar, simd_width](this.size)

    fn zero(this: Buffer[dtype]):
        memset_zero(this.data, this.size)

    fn __neg__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype]) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → __neg__ is for numeric data types only",
        ]()

        var out = Buffer[dtype](this.size)

        @parameter
        fn negate_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, this.load[simdwidth](idx).__neg__())

        vectorize[negate_elems, simd_width](this.size)
        return out

    fn __invert__[
        simd_width: Int = simdwidthof[DType.bool]()
    ](this: Buffer[DType.bool]) -> Buffer[DType.bool]:
        total = this.size
        out = Buffer[DType.bool](total)

        _ = """@parameter
        fn invert_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, this.load[simdwidth](idx).__invert__())

        vectorize[invert_elems, simd_width](total)"""

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx).__invert__()
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, this.load(k).__invert__())
        return out

    fn __eq__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        total = this.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx) == scalar
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, this.load(k) == scalar)
        return out

    fn __ne__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        total = this.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx) != scalar
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, this.load(k) != scalar)
        return out

    fn __lt__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        total = this.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx) < scalar
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, this.load(k) < scalar)
        return out

    fn __le__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        total = this.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx) <= scalar
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, this.load(k) <= scalar)
        return out

    fn __gt__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        total = this.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx) > scalar
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, this.load(k) > scalar)
        return out

    fn __ge__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        total = this.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx) >= scalar
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, this.load(k) >= scalar)
        return out

    fn __eq__[
        dtype: DType,
        simd_width: Int = simdwidthof[dtype](),
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → __eq__: buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx) == rhs.load[simd_width](idx)
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
            # out.store[simd_width](idx, cmp)
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, lhs.load(k) == rhs.load(k))
        return out

    fn __ne__[
        dtype: DType,
        simd_width: Int = simdwidthof[dtype](),
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → __ne__: buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx) != rhs.load[simd_width](idx)
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, lhs.load(k) != rhs.load(k))
        return out

    fn __lt__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → __lt__: buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx) < rhs.load[simd_width](idx)
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, lhs.load(k) < rhs.load(k))
        return out

    fn __le__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → __le__: buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx) <= rhs.load[simd_width](idx)
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, lhs.load(k) <= rhs.load(k))
        return out

    fn __gt__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → __gt__: buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx) > rhs.load[simd_width](idx)
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, lhs.load(k) > rhs.load(k))
        return out

    fn __ge__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → __ge__: buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx) >= rhs.load[simd_width](idx)
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, lhs.load(k) >= rhs.load(k))
        return out

    fn float(self) -> Buffer[DType.float32]:
        if dtype == DType.float32:
            return rebind[Buffer[DType.float32]](self)
        return self.to_dtype[DType.float32]()

    fn float64(self) -> Buffer[DType.float64]:
        if dtype == DType.float64:
            return rebind[Buffer[DType.float64]](self)
        return self.to_dtype[DType.float64]()

    fn to_dtype[NewType: DType](self) -> Buffer[NewType]:
        total = self.size
        out = Buffer[NewType](total)

        @parameter
        if NewType != DType.bool:

            @parameter
            fn cast_values[simd_width: Int](idx: Int):
                out.store[simd_width](
                    idx, self.load[simd_width](idx).cast[NewType]()
                )

            vectorize[cast_values, simdwidthof[NewType]()](self.size)
        else:
            alias simd_width: Int = simdwidthof[NewType]()
            simd_blocks = total // simd_width
            for block in range(simd_blocks):
                idx = block * simd_width
                cmp = self.load[simd_width](idx).cast[NewType]()
                for k in range(simd_width):
                    out.store[simd_width](idx + k, cmp[idx + k])
            i = simd_blocks * simd_width

            for k in range(i, total):
                out.store(k, self.load(k).cast[NewType]())

        return out

    fn sum[
        simd_width: Int = simdwidthof[dtype]()
    ](
        this: Buffer[dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ) -> Scalar[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → sum is for numeric data types only",
        ]()
        var summ = Scalar[dtype](0)
        total = (end_index.value() if end_index else this.size) - start_index
        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = start_index + block * simd_width
            summ += this.load[simd_width](idx).reduce_add()
        i = simd_blocks * simd_width

        for k in range(i, total):
            summ += this.load(k + start_index)
        return summ

    fn dot(lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Scalar[dtype]:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → dot: buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )

        return (lhs * rhs).sum()

    fn product[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype]) -> Scalar[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → product is for numeric data types only",
        ]()

        var product = Scalar[dtype](1)
        total = this.size
        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            product *= this.load[simd_width](idx).reduce_mul()
        i = simd_blocks * simd_width

        for k in range(i, total):
            product *= this.load(k)
        return product

    fn all_close[
        simd_width: Int = simdwidthof[dtype](),
    ](
        lhs: Buffer[dtype],
        rhs: Buffer[dtype],
        rtol: Scalar[dtype] = 1e-5,
        atol: Scalar[dtype] = 1e-8,
    ) -> Bool:
        constrained[
            dtype.is_floating_point(),
            "Buffer → all_close is for floating point data types only",
        ]()

        num_elems = len(lhs)
        simd_blocks = num_elems // simd_width
        tail_start = simd_blocks * simd_width

        for i in range(simd_blocks):
            vector1 = lhs.load[simd_width](i * simd_width)
            vector2 = rhs.load[simd_width](i * simd_width)
            diff = abs(vector1 - vector2)
            tolerance = atol + rtol * abs(vector2)
            if (diff > tolerance).reduce_or():
                return False
        for k in range(tail_start, num_elems):
            value1 = lhs[simd_blocks * simd_width + k]
            value2 = rhs[simd_blocks * simd_width + k]
            if abs(value1 - value2) > atol + rtol * abs(value2):
                return False

        return True

    fn any[
        simd_width: Int = simdwidthof[dtype](),
    ](
        this,
        scalar_pred: fn (Scalar[dtype]) -> Bool,
        simd_pred: Optional[
            fn (SIMD[dtype, simd_width]) -> SIMD[DType.bool, simd_width]
        ] = None,
    ) -> Bool:
        num_elems = len(this)
        simd_blocks = num_elems // simd_width
        remaining = num_elems % simd_width

        for i in range(simd_blocks):
            vector = this.load[simd_width](i * simd_width)
            if simd_pred:
                any_true = simd_pred.value()(vector)
                if any_true.reduce_or():
                    return True
            else:
                for j in range(simd_width):
                    if scalar_pred(vector[j]):
                        return True

        for k in range(remaining):
            scalar = this.load(simd_blocks * simd_width + k)
            if scalar_pred(scalar):
                return True

        return False

    fn for_all[
        simd_width: Int = simdwidthof[dtype](),
    ](this: Buffer[dtype], pred: fn (Scalar[dtype]) -> Bool) -> Buffer[
        DType.bool
    ]:
        total = this.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            vector = this.load[simd_width](idx)

            for k in range(simd_width):
                out.store[simd_width](idx + k, pred(vector[k]))

        i = simd_blocks * simd_width
        for k in range(i, total):
            out.store(k, pred(this.load(k)))

        return out

    fn all_true[
        simd_width: Int = simdwidthof[dtype](),
    ](self: Buffer[dtype], pred: fn (Scalar[dtype]) -> Bool) -> Bool:
        total = self.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            vals = self.load[simd_width](idx)
            for k in range(simd_width):
                if not pred(vals[k]):
                    return False

        i = simd_blocks * simd_width
        for k in range(i, total):
            if not pred(self.load(k)):
                return False

        return True

    fn all_true[
        simd_width: Int = simdwidthof[DType.bool](),
    ](buf: Buffer[DType.bool]) -> Bool:
        fn pred(scalar: Scalar[DType.bool]) -> Bool:
            return scalar == True

        return buf.all_true[simd_width](pred)

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        length = len(self)
        writer.write("Buffer[")
        if length <= 60:
            for i in range(length):
                writer.write(self.load(i))
                if i < length - 1:
                    writer.write(", ")
        else:
            for i in range(15):
                writer.write(self.load(i))
                writer.write(", ")

            writer.write("..., ")
            for i in range(length - 15, length):
                writer.write(self.load(i))
                if i < length - 1:
                    writer.write(", ")

        writer.write(", dtype=", self.dtype, ", size=", length, "]")

    @no_inline
    fn __repr__(self) -> String:
        return self.__str__()

    fn free(owned this):
        for i in range(len(this)):
            (this.data + i).destroy_pointee()
        this.data.free()
        log_debug("Buffer__del__ → freed data pointees")
        _ = this^

    @staticmethod
    fn of(elems: List[Scalar[dtype]]) -> Buffer[dtype]:
        buffer = Buffer[dtype](len(elems))
        memcpy(buffer.data, elems.data, len(elems))
        return buffer^


fn main() raises:
    test_buffer_slice()
    test_buffer_buffer_mul()
    test_buffer_buffer_add()
    test_buffer_buffer_mul()
    test_buffer_scalar_float_greater_than()
    test_buffer_scalar_float_less_than_eq()
    test_buffer_scalar_float_greater_than_eq()
    test_buffer_scalar_float_less_than()
    test_buffer_scalar_float_equality()
    test_buffer_scalar_float_inequality()
    test_buffer_float_equality()
    test_buffer_dot()
    test_buffer_prod()
    test_buffer_sum()
    test_buffer_float_greater_than_eq()
    test_buffer_float_greater_than()
    test_buffer_float_less_than()
    test_buffer_float_inequality()
    test_buffer_float_less_eq_than()


from testing import assert_true, assert_false


fn test_buffer_slice() raises:
    buff = Buffer.of([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(buff)
    sliced = buff[1::2]
    for i in range(len(sliced)):
        print(sliced[i])
    print(sliced)


fn test_buffer_buffer_mul() raises:
    x = Buffer[DType.bool](129)
    x.fill(True)
    y = Buffer[DType.bool](129)
    y.fill(True)
    expect = Buffer[DType.bool](129)
    expect.fill(True)
    mul_result = x * y
    cmp_result = mul_result == expect

    assert_true(
        cmp_result.all_true(),
        "Buffer buffer mul for boolean - assertion failed",
    )
    y.fill(False)
    expect.fill(True)
    mul_result = x * y
    cmp_result = mul_result == expect
    assert_false(
        cmp_result.all_true(),
        "Buffer buffer mul for boolean(True * False) - assertion failed",
    )


fn test_buffer_buffer_add() raises:
    a = Buffer(72)
    a.fill(43.0)
    b = Buffer(72)
    b.fill(43.0)
    expected = Buffer(72)
    expected.fill(86)
    added = a + b
    result = added == expected
    assert_true(
        result.all_true(),
        "Buffer buffer add assertion failed",
    )


fn test_buffer_scalar_float_greater_than_eq() raises:
    a = Buffer(72)
    a.fill(43.0)
    result1 = a >= 42
    a.fill(42.0)
    result2 = a >= 42
    assert_true(
        result1.all_true() and result2.all_true(),
        "Buffer scalar float greater than eq assertion failed",
    )


fn test_buffer_scalar_float_less_than_eq() raises:
    a = Buffer(72)
    a.fill(42.0)
    result1 = a <= 43
    result2 = a <= 42
    assert_true(
        result1.all_true() and result2.all_true(),
        "Buffer scalar float less than eq assertion failed",
    )


fn test_buffer_scalar_float_greater_than() raises:
    a = Buffer(72)
    a.fill(42.0)
    result = a > 41
    assert_true(
        result.all_true(), "Buffer scalar float greater than assertion failed"
    )


fn test_buffer_scalar_float_less_than() raises:
    a = Buffer(72)
    a.fill(42.0)
    result = a < 43
    assert_true(
        result.all_true(), "Buffer scalar float less than assertion failed"
    )


fn test_buffer_scalar_float_inequality() raises:
    a = Buffer(72)
    a.fill(42.0)
    result = a != 43
    assert_true(
        result.all_true(), "Buffer scalar float inequality assertion failed"
    )


fn test_buffer_scalar_float_equality() raises:
    a = Buffer(72)
    a.fill(42.0)
    result = a == 42
    assert_true(
        result.all_true(), "Buffer scalar float equality assertion failed"
    )


fn test_buffer_dot() raises:
    a = Buffer(33)
    a.fill(42.0)
    b = Buffer(33)
    b.fill(2.0)
    assert_true(
        a.dot(b) == b.dot(a) and a.dot(b) == 2772, "dot assertion failed"
    )


fn test_buffer_prod() raises:
    a = Buffer(2)
    a.fill(42.0)
    result = a.product()
    assert_true(result == 1764, "prod assertion failed")


fn test_buffer_sum() raises:
    a = Buffer(72)
    a.fill(42.0)
    result = a.sum()
    assert_true(result == 3024, "Sum assertion failed")


fn test_buffer_float_greater_than_eq() raises:
    a = Buffer(72)
    a.fill(42.0)
    b = Buffer(72)
    b.fill(420)
    result = b >= a
    assert_true(result.all_true(), "72 float greater than eq assertion failed")

    a = Buffer(31)
    a.fill(42.0)
    b = Buffer(31)
    b.fill(42)
    result = b >= a
    assert_true(result.all_true(), "31 float greater than eq assertion failed")


fn test_buffer_float_greater_than() raises:
    a = Buffer(72)
    a.fill(42.0)
    b = Buffer(72)
    b.fill(420)
    result = b > a
    assert_true(result.all_true(), "72 float greater than assertion failed")


fn test_buffer_float_less_eq_than() raises:
    a = Buffer(72)
    a.fill(42.0)
    b = Buffer(72)
    b.fill(420)
    result = a <= b
    assert_true(result.all_true(), "72 float less than eq assertion failed")

    a = Buffer(65)
    a.fill(42.0)
    b = Buffer(65)
    b.fill(42)
    result = a <= b
    assert_true(result.all_true(), "65 float less than eq assertion failed")


fn test_buffer_float_less_than() raises:
    a = Buffer(72)
    a.fill(42.0)
    b = Buffer(72)
    b.fill(420)
    result = a < b
    assert_true(result.all_true(), "72 float less than assertion failed")


fn test_buffer_float_equality() raises:
    a = Buffer(72)
    a.fill(42.0)
    b = Buffer(72)
    b.fill(42)
    result = a == b
    assert_true(result.all_true(), "72 float equality assertion failed")

    a = Buffer(1)
    a.fill(42.0)
    b = Buffer(1)
    b.fill(42)
    result = a == b
    assert_true(result.all_true(), "1 float equality assertion failed")

    a = Buffer(1024)
    a.fill(42.0)
    b = Buffer(1024)
    b.fill(42)
    result = a == b
    assert_true(result.all_true(), "1024 float equality assertion failed")


fn test_buffer_float_inequality() raises:
    a = Buffer(72)
    a.fill(42.0)
    b = Buffer(72)
    b.fill(420)
    result = a != b
    assert_true(result.all_true(), "72 float inequality assertion failed")

    a = Buffer(1)
    a.fill(42.0)
    b = Buffer(1)
    b.fill(420)
    result = a != b
    assert_true(result.all_true(), "1 float inequality assertion failed")

    a = Buffer(1024)
    a.fill(42.0)
    b = Buffer(1024)
    b.fill(420)
    result = a != b
    assert_true(result.all_true(), "1024 float inequality assertion failed")
