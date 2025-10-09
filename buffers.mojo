from algorithm import vectorize
from sys import simdwidthof
from memory import memset_zero, memcpy, ArcPointer
from math import exp, log
from common_utils import log_debug, panic


struct Buffer[dtype: DType = DType.float32](
    Copyable & Movable & Sized & Stringable & Writable & Representable & Absable
):
    var size: Int
    var data: UnsafePointer[Scalar[dtype]]
    var external: Bool

    fn __init__(out self):
        self.size = 0
        self.data = UnsafePointer[Scalar[dtype]]()
        self.external = False

    fn __init__(out self, size: Int, external: Bool = False):
        if size < 0:
            panic("Buffer size must be >= 0")
        self.size = size
        self.external = external

        if size == 0:
            self.data = UnsafePointer[Scalar[dtype]]()
        else:
            if external:
                # Expect data pointer to be set later
                self.data = UnsafePointer[Scalar[dtype]]()
            else:
                self.data = UnsafePointer[Scalar[dtype]].alloc(size)

    fn __init__(out self, elems: List[Scalar[dtype]]):
        length = len(elems)
        self.data = UnsafePointer[Scalar[dtype]].alloc(length)
        self.size = length
        memcpy(self.data, elems._data, length)
        self.external = False

    fn __init__(
        out self,
        size: Int,
        data: UnsafePointer[Scalar[dtype]],
        copy: Bool = False,
    ):
        self.size = size
        if copy:
            self.data = UnsafePointer[Scalar[dtype]].alloc(size)
            memcpy(self.data, data, size)
            self.external = False
        else:
            self.data = data
            self.external = True

    fn __moveinit__(out self, deinit other: Self):
        self.size = other.size
        self.data = other.data
        self.external = other.external

    fn __copyinit__(out self, other: Self):
        self.size = other.size
        self.data = UnsafePointer[Scalar[dtype]].alloc(other.size)
        memcpy(self.data, other.data, other.size)
        self.external = other.external

    fn shared(self) -> ArcPointer[Buffer[dtype]]:
        return ArcPointer(self)

    fn __len__(self) -> Int:
        return self.size

    fn __iter__(ref self) -> Iterator[dtype, __origin_of(self)]:
        return Iterator(0, Pointer(to=self))

    fn __getitem__(self, slice: Slice) -> Buffer[dtype]:
        var start, end, step = slice.indices(len(self))
        var spread = range(start, end, step)

        if not len(spread):
            return Buffer[dtype]()

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

        return result

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

    @always_inline
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

    @always_inline
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

    @always_inline
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

    @always_inline
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

        @parameter
        fn mul_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) * rhs.load[simdwidth](idx)),
            )

        # vectorize[mul_elems, simdwidthof[DType.bool]()](lhs.size)
        vectorize[mul_elems, 1](lhs.size)
        _ = """alias simd_width = simdwidthof[DType.bool]()
        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx) * rhs.load[simd_width](idx)
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
            # out.store[simd_width](idx, cmp)
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](lhs.load(k) == rhs.load(k)))"""
        return out

    @always_inline
    fn __mul__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[dtype]:
        if not lhs.size == rhs.size:
            panic(
                (
                    "Buffer → __mul__(Buffer[dtype]: buffer size does not"
                    " match -> lhs:"
                ),
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )

        out = Buffer[dtype](lhs.size)

        @parameter
        fn mul_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) * rhs.load[simdwidth](idx)),
            )

        vectorize[mul_elems, simd_width](lhs.size)
        return out

    @always_inline
    fn __imul__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]):
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
            lhs.store(k, Scalar[DType.bool](lhs.load(k) == rhs.load(k)))

    fn __truediv__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[dtype]:
        if not lhs.size == rhs.size:
            panic(
                (
                    "Buffer → __trudiv__(Buffer[dtype]: buffer size does not"
                    " match -> lhs:"
                ),
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )

        out = Buffer[dtype](lhs.size)

        @parameter
        fn mul_elems[simdwidth: Int](idx: Int):
            out.store[simdwidth](
                idx,
                (lhs.load[simdwidth](idx) / rhs.load[simdwidth](idx)),
            )

        vectorize[mul_elems, simd_width](lhs.size)
        return out

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
    @always_inline
    fn full[
        simd_width: Int = simdwidthof[dtype]()
    ](value: Scalar[dtype], size: Int) -> Buffer[dtype]:
        buffer = Buffer[dtype](size)
        buffer.fill[simd_width](value)
        return buffer

    @staticmethod
    @always_inline
    fn zeros[
        simd_width: Int = simdwidthof[dtype]()
    ](size: Int) -> Buffer[dtype]:
        buffer = Buffer[dtype](size)
        memset_zero(buffer.data, size)
        return buffer

    @always_inline
    fn fill[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], value: Scalar[dtype]):
        @parameter
        fn set_scalar[simdwidth: Int](idx: Int):
            this.store[simdwidth](idx, value)

        if dtype == DType.bool:
            vectorize[set_scalar, 1](this.size)
        else:
            vectorize[set_scalar, simd_width](this.size)

    @always_inline
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

    fn log[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype]) -> Buffer[dtype]:
        constrained[
            dtype.is_numeric(),
            "Buffer → log is for numeric data types only",
        ]()

        total = this.size
        out = Buffer[dtype](total)

        @parameter
        fn log_of[simdwidth: Int](idx: Int):
            out.store[simdwidth](idx, log(this.load[simdwidth](idx)))

        vectorize[log_of, simd_width](total)
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
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Bool:
        total = this.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            if not this.load[simd_width](idx) == scalar:
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not this.load(k) == scalar:
                return False
        return True

    fn eq[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        total = this.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx).eq(scalar)
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](this.load(k).eq(scalar)))
        return out

    fn __ne__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Bool:
        total = this.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            if not this.load[simd_width](idx) != scalar:
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not this.load(k) != scalar:
                return False
        return True

    fn ne[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        total = this.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx).ne(scalar)
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](this.load(k).ne(scalar)))
        return out

    fn lt[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        total = this.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx).lt(scalar)
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, this.load(k).lt(scalar))
        return out

    fn __lt__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Bool:
        total = this.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx).lt(scalar)
            if cmp == Scalar[DType.bool](False):
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not this.load(k).lt(scalar):
                return False
        return True

    fn __le__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Bool:
        total = this.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx).le(scalar)
            if cmp == Scalar[DType.bool](False):
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not this.load(k).le(scalar):
                return False
        return True

    fn le[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        total = this.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx).le(scalar)
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](this.load(k).le(scalar)))
        return out

    fn __gt__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Bool:
        total = this.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx).gt(scalar)
            if cmp == Scalar[DType.bool](False):
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not this.load(k).gt(scalar):
                return False
        return True

    fn gt[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        total = this.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx).gt(scalar)
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](this.load(k).gt(scalar)))
        return out

    fn __ge__[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Bool:
        total = this.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx).ge(scalar)
            if cmp == Scalar[DType.bool](False):
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not this.load(k).ge(scalar):
                return False
        return True

    fn ge[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], scalar: Scalar[dtype]) -> Buffer[dtype.bool]:
        total = this.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = this.load[simd_width](idx).ge(scalar)
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](this.load(k).ge(scalar)))
        return out

    fn eq[
        simd_width: Int = simdwidthof[dtype](),
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → eq: buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx).eq(rhs.load[simd_width](idx))
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
            # out.store[simd_width](idx, cmp)
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](lhs.load(k).eq(rhs.load(k))))
        return out

    @always_inline
    fn __eq__[
        simd_width: Int = simdwidthof[dtype](),
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Bool:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → __eq__(buffer): Buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            if not lhs.load[simd_width](idx) == rhs.load[simd_width](idx):
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not lhs.load(k) == rhs.load(k):
                return False
        return True

    fn ne[
        simd_width: Int = simdwidthof[dtype](),
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → __ne__(buffer): Buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx).ne(rhs.load[simd_width](idx))
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](lhs.load(k).ne(rhs.load(k))))
        return out

    fn __ne__[
        simd_width: Int = simdwidthof[dtype](),
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Bool:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → __ne__(buffer): Buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx).ne(rhs.load[simd_width](idx))
            if cmp == Scalar[DType.bool](False):
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not lhs.load(k).ne(rhs.load(k)):
                return False
        return True

    fn lt[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → lt(buffer): Buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx).lt(rhs.load[simd_width](idx))
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](lhs.load(k).lt(rhs.load(k))))
        return out

    fn __lt__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Bool:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → __lt__(buffer): Buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx).lt(rhs.load[simd_width](idx))
            if cmp == Scalar[DType.bool](False):
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not lhs.load(k).lt(rhs.load(k)):
                return False
        return True

    fn le[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → le(buffer): Buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx).le(rhs.load[simd_width](idx))
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](lhs.load(k).le(rhs.load(k))))
        return out

    fn __le__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Bool:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → __le__(buffer): Buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            if not lhs.load[simd_width](idx).le(rhs.load[simd_width](idx)):
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not lhs.load(k) <= rhs.load(k):
                return False
        return True

    fn __gt__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Bool:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → __gt__(buffer): Buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx).gt(rhs.load[simd_width](idx))
            if cmp == Scalar[DType.bool](False):
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not lhs.load(k).gt(rhs.load(k)):
                return False
        return True

    fn gt[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → gt(buffer): Buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx).gt(rhs.load[simd_width](idx))
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](lhs.load(k).gt(rhs.load(k))))
        return out

    fn __ge__[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Bool:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → __ge__(buffer): Buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx).ge(rhs.load[simd_width](idx))
            if cmp == Scalar[DType.bool](False):
                return False
        i = simd_blocks * simd_width

        for k in range(i, total):
            if not lhs.load(k).ge(rhs.load(k)):
                return False
        return True

    fn ge[
        simd_width: Int = simdwidthof[dtype]()
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Buffer[DType.bool]:
        if not lhs.size == rhs.size:
            panic(
                "Buffer → ge: buffer size does not match -> lhs:",
                lhs.size.__str__(),
                "vs. rhs:",
                rhs.size.__str__(),
            )
        total = lhs.size
        out = Buffer[DType.bool](total)

        simd_blocks = total // simd_width
        for block in range(simd_blocks):
            idx = block * simd_width
            cmp = lhs.load[simd_width](idx).ge(rhs.load[simd_width](idx))
            for k in range(simd_width):
                out.store[simd_width](idx + k, cmp[idx + k])
        i = simd_blocks * simd_width

        for k in range(i, total):
            out.store(k, Scalar[DType.bool](lhs.load(k).ge(rhs.load(k))))
        return out

    fn float(self) -> Buffer[DType.float32]:
        if dtype == DType.float32:
            return rebind[Buffer[DType.float32]](self)
        return self.to_dtype[DType.float32]()

    fn float64(self) -> Buffer[DType.float64]:
        if dtype == DType.float64:
            return rebind[Buffer[DType.float64]](self)
        return self.to_dtype[DType.float64]()

    fn to_dtype[
        NewType: DType, simdwidth: Int = simdwidthof[NewType]()
    ](self) -> Buffer[NewType]:
        total = self.size
        out = Buffer[NewType](total)

        @parameter
        if NewType != DType.bool:

            @parameter
            fn cast_values[simd_width: Int](idx: Int):
                out.store[simd_width](
                    idx, self.load[simd_width](idx).cast[NewType]()
                )

            vectorize[cast_values, simdwidth](self.size)
        else:
            simd_blocks = total // simdwidth
            for block in range(simd_blocks):
                idx = block * simdwidth
                cmp = self.load[simdwidth](idx).cast[NewType]()
                for k in range(simdwidth):
                    out.store[simdwidth](idx + k, cmp[idx + k])
            i = simd_blocks * simdwidth

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

    fn count[
        simd_width: Int = simdwidthof[dtype]()
    ](this: Buffer[dtype], key: Scalar[dtype]) -> Int:
        total = 0

        @parameter
        fn matches[simdwidth: Int](idx: Int):
            block = this.load[simdwidth](idx)
            result = block == key
            if result:
                total += simdwidth
            elif not result and simdwidth > 1:
                for i in range(simdwidth):
                    if block[i] == key:
                        total += 1

        @parameter
        if dtype == DType.bool:
            vectorize[matches, 1](this.size)
        else:
            vectorize[matches, simd_width](this.size)

        return total

    fn all_close[
        simd_width: Int = simdwidthof[dtype](),
        rtol: Scalar[dtype] = 1e-5,
        atol: Scalar[dtype] = 1e-8,
    ](lhs: Buffer[dtype], rhs: Buffer[dtype]) -> Bool:
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
            if not diff.le(tolerance).reduce_and():
                return False

        # Handle tail (non-SIMD leftover)
        for k in range(tail_start, num_elems):
            value1 = lhs[k]
            value2 = rhs[k]
            if abs(value1 - value2).gt(atol + rtol * abs(value2)):
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
        out = Buffer[DType.bool](len(this))

        for i in range(len(this)):
            out[i] = pred(this[i])

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

    fn all_true(buf: Buffer[DType.bool]) -> Bool:
        fn pred(scalar: Scalar[DType.bool]) -> Bool:
            return scalar == Scalar[DType.bool](True)

        return buf.all_true[1](pred)

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

    fn __del__(deinit self):
        should_delete = (
            self.data.__as_bool__() and self.size > 0 and not self.external
        )
        if should_delete:
            for i in range(len(self)):
                (self.data + i).destroy_pointee()
            self.data.free()
            log_debug("Buffer__del__ → freed data pointees")


struct Iterator[
    dtype: DType,
    origin: Origin[False],
](Sized & Copyable):
    var index: Int
    var src: Pointer[Buffer[dtype], origin]

    fn __init__(out self, idx: Int, src: Pointer[Buffer[dtype], origin]):
        self.src = src
        self.index = idx

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) -> Scalar[dtype]:
        self.index += 1
        return self.src[][self.index - 1]

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        return len(self.src[]) - self.index


fn main() raises:
    _ = """l = List[Scalar[DType.int32]](capacity=10)
    for i in range(10):
        l.append(Scalar[DType.int32](i))
    b = Buffer[DType.int32](l)"""
    pass


from testing import assert_true, assert_false
